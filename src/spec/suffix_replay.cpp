// src/spec/suffix_replay.cpp - 4Mile: replays recorded agent sessions through the lookup drafters (no model).
//
//   suffix_replay [--k K] [--drafter old|vote] [--cand N] [--share F] [--keep-div D] [--min-match M] [--global]
//                 SESSION.stream.i32 SESSION.steps.txt [...]
//
// A session is its whole conversation as token ids (stream.i32) and its assistant steps ("ctx gen" a line: the
// step's prompt is stream[0, ctx), it generated stream[ctx, ctx + gen)). Each step is decoded again as the engine
// would: at every position the drafter proposes up to K tokens from the history (the step's prompt, as the engine
// gives it a request's prompt, plus what the step has generated so far); a proposal whose first token is right (the
// engine takes a lookup window only where the MTP's first draft agrees) is verified: its accepted prefix plus the
// window's own token commit at once. Elsewhere the token is left to the MTP. --global keeps one history over the
// whole session instead (a store that outlives a compaction).
//
// Prints, per drafter setting: the tokens, the share committed by lookup windows, the windows by size and their mean
// accepted drafts - the inputs to tokens/s with the measured window costs.
#include "strata/spec/suffix_drafter.hpp"
#include "strata/spec/suffix_vote_drafter.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {
struct Session {
    std::string name;
    std::vector<int32_t> stream;
    std::vector<std::pair<size_t, size_t>> steps;
};

bool load(const char* ids, const char* steps, Session& s) {
    std::ifstream f(ids, std::ios::binary);
    if (!f) return false;
    f.seekg(0, std::ios::end);
    const size_t bytes = (size_t) f.tellg();
    f.seekg(0);
    s.stream.resize(bytes / 4);
    f.read((char*) s.stream.data(), (std::streamsize) bytes);
    std::ifstream g(steps);
    size_t c, n;
    while (g >> c >> n) s.steps.push_back({c, n});
    s.name = ids;
    return true;
}

constexpr int kMaxK = 64;
// the measured window cost on the rig (RTX 3090 + 3060, 6.05 at 262K, 10-20: T 2/3/4/6/8 = 27.8/35.1/42.9/56.6/72.4 ms)
double cost_ms(int t, double c0, double c1) { return c0 + c1 * (double) t; }
// a lookup window's policy: per match-length bucket, the acceptance of a drafted token (geometric), learned online
struct SimPolicy {
    static constexpr int kB = 6;
    double ok[kB] = {}, bad[kB] = {};
    static int bucket(int m) { return m < 6 ? 0 : m < 12 ? 1 : m < 24 ? 2 : m < 48 ? 3 : m < 96 ? 4 : 5; }
    double q(int m) const {
        static const double prior[kB] = {0.75, 0.85, 0.9, 0.94, 0.96, 0.97};
        const int b = bucket(m);
        return (ok[b] + 4.0 * prior[b]) / (ok[b] + bad[b] + 4.0);
    }
    // the drafts to verify (0 = leave it to the MTP): best expected tokens per ms against the MTP's rate
    int choose(int k, int m, double mtp_ms_tok, double c0, double c1, double margin) const {
        const double qq = q(m);
        double e = 1.0, qi = 1.0, best = 1.0 / mtp_ms_tok * (1.0 + margin);
        int pick = 0;
        for (int j = 1; j <= k; ++j) {
            qi *= qq;
            e += qi;
            const double r = e / cost_ms(j + 1, c0, c1);
            if (r > best) { best = r; pick = j; }
        }
        return pick;
    }
    void observe(int m, int drafted, int accepted) {
        const int b = bucket(m);
        ok[b] = 0.99 * ok[b] + accepted;
        bad[b] = 0.99 * bad[b] + (accepted < drafted ? 1.0 : 0.0);
    }
};
struct Stats {
    long long tokens = 0, lookup_tokens = 0, windows = 0, drafted = 0, accepted = 0;
    long long by_t[kMaxK + 2] = {}, acc_by_t[kMaxK + 2] = {};
    double ms = 0.0;
    // the cost model: modelled decode time with the policy, and the oracle's (it knows how far a proposal holds)
    double t_policy = 0.0, t_oracle = 0.0, t_mtp = 0.0;
    long long pol_windows = 0, pol_tokens = 0, pol_drafted = 0;
};
}  // namespace

int main(int argc, char** argv) {
    int K = 5;
    std::string which = "old";
    bool global = false;
    strata::spec::SuffixVoteDrafter::Params vp;
    int old_min = 3;
    double c0 = 13.1, c1 = 7.4, mtp_ms_tok = 18.9, margin = 0.03;
    std::vector<Session> ss;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() { return i + 1 < argc ? argv[++i] : (char*) ""; };
        if (a == "--k") K = std::min(kMaxK, std::atoi(next()));
        else if (a == "--drafter") which = next();
        else if (a == "--cand") vp.max_cand = std::atoi(next());
        else if (a == "--share") vp.min_share = std::atof(next());
        else if (a == "--keep-div") vp.keep_div = std::atoi(next());
        else if (a == "--min-match") vp.min_match = old_min = std::atoi(next());
        else if (a == "--max-match") vp.max_match = std::atoi(next());
        else if (a == "--global") global = true;
        else if (a == "--c0") c0 = std::atof(next());
        else if (a == "--c1") c1 = std::atof(next());
        else if (a == "--mtp-ms") mtp_ms_tok = std::atof(next());
        else if (a == "--margin") margin = std::atof(next());
        else {
            Session s;
            if (i + 1 >= argc || !load(argv[i], argv[i + 1], s)) { std::fprintf(stderr, "cannot read %s\n", argv[i]); return 2; }
            ++i;
            ss.push_back(std::move(s));
        }
    }
    const bool vote = which == "vote";
    // mode 0: every proposal whose first token is right is verified whole (the drafter's reach);
    // mode 1: the policy picks how many drafts to verify where they pay (the cost model); mode 2: the oracle
    for (int mode = 0; mode < 3; ++mode) {
        Stats st;
        double t_model = 0.0;
        std::vector<int32_t> buf(kMaxK);
        for (const Session& s : ss) {
            strata::spec::SuffixDrafter old_d(old_min, 64, s.stream.size() + 4096);
            strata::spec::SuffixVoteDrafter vote_d(vp, s.stream.size() + 4096);
            SimPolicy pol;
            size_t fed = 0;
            for (const auto& [ctx, gen] : s.steps) {
                const auto t0 = std::chrono::steady_clock::now();
                if (!global || fed > ctx) {
                    if (vote) vote_d.reset(); else old_d.reset();
                    fed = 0;
                }
                if (vote) vote_d.append(s.stream.data() + fed, ctx - fed); else old_d.append(s.stream.data() + fed, ctx - fed);
                fed = ctx;
                const int32_t* g = s.stream.data() + ctx;
                size_t i = 0;
                while (i < gen) {
                    const int room = (int) std::min<size_t>((size_t) K, gen - i - 1);
                    int k = 0;
                    if (room > 0) k = vote ? vote_d.propose(room, buf.data()) : old_d.propose(room, buf.data());
                    const int m = vote ? vote_d.last_match() : old_d.last_match();
                    size_t commit = 1;
                    double cost = mtp_ms_tok;   // the token left to the MTP
                    if (k > 0 && buf[0] == g[i]) {
                        int a = 0;
                        while (a < k && buf[(size_t) a] == g[i + (size_t) a]) ++a;
                        int j = k;   // the drafts verified
                        if (mode == 1) {
                            j = pol.choose(k, m, mtp_ms_tok, c0, c1, margin);
                            if (j > 0) {
                                a = std::min(a, j);
                                pol.observe(m, j, a);
                            }
                        } else if (mode == 2) {
                            j = (a > 0 && (double) (a + 1) / cost_ms(a + 1, c0, c1) > 1.0 / mtp_ms_tok) ? a : 0;
                        }
                        if (j > 0) {
                            commit = (size_t) a + 1;
                            cost = cost_ms(j + 1, c0, c1);
                            ++st.windows;
                            st.drafted += j;
                            st.accepted += a;
                            ++st.by_t[std::min(j + 1, kMaxK + 1)];
                            st.acc_by_t[std::min(j + 1, kMaxK + 1)] += a;
                            st.lookup_tokens += (long long) commit;
                        }
                    }
                    t_model += cost;
                    if (vote) vote_d.append(g + i, commit); else old_d.append(g + i, commit);
                    i += commit;
                }
                fed += gen;
                st.tokens += (long long) gen;
                st.ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            }
        }
        static const char* names[3] = {"reach", "policy", "oracle"};
        std::printf("%-6s drafter %s K %d%s: %.1f %% of %lld tokens by %lld lookup windows (%.2f a window), drafts %lld "
                    "accepted %lld (%.1f %%) | model %.1f tok/s, x%.2f against the MTP alone\n", names[mode], which.c_str(), K,
                    global ? " global" : "", 100.0 * (double) st.lookup_tokens / (double) std::max(1LL, st.tokens), st.tokens,
                    st.windows, (double) st.lookup_tokens / (double) std::max(1LL, st.windows), st.drafted, st.accepted,
                    100.0 * (double) st.accepted / (double) std::max(1LL, st.drafted), 1000.0 * (double) st.tokens / t_model,
                    (double) st.tokens * mtp_ms_tok / t_model);
        if (mode == 1) {
            std::printf("       windows by size T (count : mean accepted):");
            for (int t = 2; t <= K + 1; ++t)
                if (st.by_t[t] > 0) std::printf(" T%d %lld:%.2f", t, st.by_t[t], (double) st.acc_by_t[t] / (double) st.by_t[t]);
            std::printf("\n");
        }
    }
    return 0;
}
