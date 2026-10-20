// src/spec/suffix_vote_drafter.cpp - see include/strata/spec/suffix_vote_drafter.hpp
#include "strata/spec/suffix_vote_drafter.hpp"

#include <algorithm>

namespace strata::spec {

namespace {
uint64_t mix(uint64_t x) {
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdull;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ull;
    x ^= x >> 33;
    return x;
}
}  // namespace

SuffixVoteDrafter::SuffixVoteDrafter(Params p, size_t capacity_tokens) : p_(p) {
    size_t cap = 1;
    while (cap < capacity_tokens * 2) cap <<= 1;   // distinct trigrams <= tokens: load factor <= 0.5 at capacity
    keys_.assign(cap, 0);
    heads_.assign(cap, 0);
    mask_ = cap - 1;
    hist_.reserve(capacity_tokens);
    prev_.reserve(capacity_tokens);
}

void SuffixVoteDrafter::reset() {
    hist_.clear();
    prev_.clear();
    std::fill(keys_.begin(), keys_.end(), 0);
    std::fill(heads_.begin(), heads_.end(), 0);
    indexed_ = 0;
    last_match_ = last_support_ = 0;
}

uint64_t SuffixVoteDrafter::key_at(size_t end) const {
    const uint64_t a = (uint32_t) hist_[end - 2], b = (uint32_t) hist_[end - 1], c = (uint32_t) hist_[end];
    return mix(a * 0x9E3779B97F4A7C15ull ^ mix(b + 0x632BE59BD9B4E019ull) ^ (c << 1)) | 1ull;   // never 0
}

uint32_t* SuffixVoteDrafter::head_slot(uint64_t key, bool insert) {
    for (size_t i = key & mask_, probes = 0; probes <= mask_; i = (i + 1) & mask_, ++probes) {
        if (keys_[i] == key) return &heads_[i];
        if (keys_[i] == 0) {
            if (!insert) return nullptr;
            keys_[i] = key;
            return &heads_[i];
        }
    }
    return nullptr;   // the table is full: the history outgrew its capacity (later trigrams are not indexed)
}

void SuffixVoteDrafter::index(size_t end, size_t limit) {
    // positions are chained once; `limit` keeps propose_after()'s temporary tokens out of the chains
    for (; indexed_ < limit && indexed_ <= end; ++indexed_) {
        const size_t e = indexed_;
        uint32_t link = 0;
        if (e >= 2)
            if (uint32_t* h = head_slot(key_at(e), true)) {
                link = *h;
                *h = (uint32_t) e + 1;
            }
        prev_.push_back(link);
    }
}

void SuffixVoteDrafter::append(const int32_t* tokens, size_t n) {
    hist_.insert(hist_.end(), tokens, tokens + n);
    if (!hist_.empty()) index(hist_.size() - 1, hist_.size());
}

int SuffixVoteDrafter::propose(int max_k, int32_t* out) {
    last_match_ = last_support_ = 0;
    const size_t n = hist_.size();
    if (n < 4 || max_k <= 0) return 0;
    const size_t cur = n - 1;
    // the current trigram's chain: its latest indexed occurrence, unless that is the current position itself
    const uint32_t* h = nullptr;
    {
        const uint64_t key = key_at(cur);
        for (size_t i = key & mask_, probes = 0; probes <= mask_; i = (i + 1) & mask_, ++probes) {
            if (keys_[i] == key) { h = &heads_[i]; break; }
            if (keys_[i] == 0) break;
        }
    }
    if (h == nullptr || *h == 0) return 0;
    uint32_t link = *h;
    cand_.clear();
    int best = 0;
    for (int seen = 0; link != 0 && seen < p_.max_cand; ++seen) {
        const size_t p = link - 1;
        link = p < prev_.size() ? prev_[p] : 0;
        if (p >= cur) continue;   // the current suffix itself (it is indexed when it is history)
        int len = 0;
        while (len < p_.max_match && len <= (int) p && hist_[p - len] == hist_[cur - len]) ++len;
        if (len < p_.min_match) continue;
        cand_.push_back({(uint32_t) p, len});
        best = std::max(best, len);
    }
    if (best < p_.min_match) return 0;
    last_match_ = best;
    // the candidates that vote: a match of at least best / keep_div
    const int keep = std::max(p_.min_match, best / std::max(1, p_.keep_div));
    size_t m = 0;
    for (const Cand& c : cand_)
        if (c.len >= keep) cand_[m++] = c;
    cand_.resize(m);
    int k = 0;
    for (; k < max_k && !cand_.empty(); ++k) {
        // tally the next token of every candidate still in agreement (most recent first: ties keep the newer)
        votes_tok_.clear();
        votes_w_.clear();
        double total = 0.0;
        for (const Cand& c : cand_) {
            const size_t q = (size_t) c.end + 1 + (size_t) k;
            if (q > cur) continue;   // its continuation reaches the present: no vote (periodic text still reads up to cur)
            const int32_t t = hist_[q];
            const double w = (double) c.len;
            total += w;
            size_t j = 0;
            while (j < votes_tok_.size() && votes_tok_[j] != t) ++j;
            if (j == votes_tok_.size()) { votes_tok_.push_back(t); votes_w_.push_back(w); }
            else votes_w_[j] += w;
        }
        if (votes_tok_.empty()) break;
        size_t win = 0;
        for (size_t j = 1; j < votes_tok_.size(); ++j)
            if (votes_w_[j] > votes_w_[win]) win = j;
        if (k > 0 && votes_w_[win] < p_.min_share * total) break;
        const int32_t t = votes_tok_[win];
        out[k] = t;
        size_t keep_n = 0;
        for (const Cand& c : cand_) {
            const size_t q = (size_t) c.end + 1 + (size_t) k;
            if (q <= cur && hist_[q] == t) cand_[keep_n++] = c;
        }
        cand_.resize(keep_n);
    }
    last_support_ = (int) cand_.size();
    return k;
}

int SuffixVoteDrafter::propose_after(const int32_t* extra, int n_extra, int max_k, int32_t* out) {
    if (n_extra <= 0) return propose(max_k, out);
    // the extra tokens only form the suffix being matched: they are not chained (indexed_ stays where it is)
    const size_t n0 = hist_.size();
    hist_.insert(hist_.end(), extra, extra + n_extra);
    const int k = propose(max_k, out);
    hist_.resize(n0);
    return k;
}

}  // namespace strata::spec
