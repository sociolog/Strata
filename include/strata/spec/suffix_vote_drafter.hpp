// include/strata/spec/suffix_vote_drafter.hpp - 4Mile: a suffix drafter for agent / coding turns (SuffixDecoding-style).
//
// The prompt-lookup drafter (suffix_drafter.hpp) takes the single longest of a trigram's 4 most recent earlier
// occurrences and copies what followed it. In an OpenCode session the context is mostly code and tool output the
// model quotes back (edit's oldString, a file it rewrites, a command it repeats), so most continuations have MANY
// earlier occurrences, and the right one is often not among the 4 most recent of a common trigram ("    return ").
//
// This one indexes EVERY earlier occurrence of every trigram (a chain through the history: prev_[p] = the previous
// end position with the same trigram, so memory is 4 bytes a token beside a fixed table, and appends are O(1)),
// scores up to `max_cand` of them by the length of their match with the current suffix, and walks the continuation
// as a vote: at each depth the candidates that still agree vote for their next token, weighted by match length; the
// winner is drafted while its share of the vote stays at or above `min_share`. A long exact match therefore drafts a
// long run (a file quoted back), and a short ambiguous one drafts only as far as its candidates agree.
//
// It only proposes; the verify window decides, so the output is the same with or without it.
#pragma once

#include "strata/spec/suffix_drafter.hpp"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <vector>

namespace strata::spec {

struct SuffixVoteParams {
    int min_match = 3;        // a proposal needs a match of at least this many tokens
    int max_match = 96;       // matches are measured up to this length
    int max_cand = 128;       // earlier occurrences scored per proposal (most recent first)
    double min_share = 0.6;   // the drafted token's least share of the vote (after the first)
    int keep_div = 2;         // candidates voting: those with a match >= best / keep_div (and >= min_match)
};

class SuffixVoteDrafter {
public:
    using Params = SuffixVoteParams;

    explicit SuffixVoteDrafter(Params p = Params{}, size_t capacity_tokens = 1u << 19);

    void reset();
    void append(const int32_t* tokens, size_t n);
    void append(int32_t token) { append(&token, 1); }

    /// Write up to `max_k` proposed next tokens to `out`; returns how many (0 = no match of at least min_match).
    int propose(int max_k, int32_t* out);
    /// As propose(), for the history followed by `extra` (drafts not verified yet); the history is unchanged.
    int propose_after(const int32_t* extra, int n_extra, int max_k, int32_t* out);

    int last_match() const { return last_match_; }       // the best match behind the last proposal
    int last_support() const { return last_support_; }   // candidates that agreed with the whole proposal
    size_t size() const { return hist_.size(); }
    const std::vector<int32_t>& history() const { return hist_; }
    const Params& params() const { return p_; }

private:
    uint64_t key_at(size_t end) const;
    uint32_t* head_slot(uint64_t key, bool insert);
    void index(size_t end, size_t limit);

    Params p_;
    std::vector<int32_t> hist_;
    std::vector<uint32_t> prev_;               // [end]: the previous end position of the same trigram (+1; 0 = none)
    std::vector<uint64_t> keys_;               // table: trigram key + 1 (0 = empty)
    std::vector<uint32_t> heads_;              // table: the latest end position of that trigram (+1)
    size_t mask_ = 0, indexed_ = 0;            // positions [0, indexed_) are in the chains
    int last_match_ = 0, last_support_ = 0;
    struct Cand { uint32_t end; int len; };
    std::vector<Cand> cand_;
    std::vector<int32_t> votes_tok_;
    std::vector<double> votes_w_;
};

/// The engine's lookup drafter: the prompt-lookup one (always kept: --lookup-chain reads it), and with
/// STRATA_SFX_VOTE=1 the vote drafter beside it, which then makes the proposals.  Both see every token.
class LookupDrafter {
public:
    LookupDrafter(int min_match, int max_match, size_t capacity_tokens) : old_(min_match, max_match, capacity_tokens) {
        if (vote_on()) {
            SuffixVoteParams p;
            p.min_match = min_match;
            if (const char* v = std::getenv("STRATA_SFX_VOTE_CAND")) p.max_cand = std::max(1, std::atoi(v));
            if (const char* v = std::getenv("STRATA_SFX_VOTE_SHARE")) p.min_share = std::atof(v);
            vote_ = std::make_unique<SuffixVoteDrafter>(p, capacity_tokens);
        }
    }
    static bool vote_on() {
        static const bool on = [] { const char* v = std::getenv("STRATA_SFX_VOTE"); return v != nullptr && v[0] == '1'; }();
        return on;
    }
    void reset() { old_.reset(); if (vote_) vote_->reset(); }
    void append(const int32_t* tokens, size_t n) { old_.append(tokens, n); if (vote_) vote_->append(tokens, n); }
    void append(int32_t token) { append(&token, 1); }
    int propose(int max_k, int32_t* out) { return vote_ ? vote_->propose(max_k, out) : old_.propose(max_k, out); }
    int propose_after(const int32_t* extra, int n_extra, int max_k, int32_t* out) {
        return vote_ ? vote_->propose_after(extra, n_extra, max_k, out) : old_.propose_after(extra, n_extra, max_k, out);
    }
    int last_match() const { return vote_ ? vote_->last_match() : old_.last_match(); }
    size_t size() const { return old_.size(); }
    const std::vector<int32_t>& history() const { return old_.history(); }
    SuffixDrafter& base() { return old_; }
    const SuffixDrafter& base() const { return old_; }

private:
    SuffixDrafter old_;
    std::unique_ptr<SuffixVoteDrafter> vote_;
};

}  // namespace strata::spec
