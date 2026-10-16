// src/kernels/cpu/exl3_lane_moe.cpp - EXL3 routed experts on the CPU expert pool (see exl3_lane_moe.hpp).
#include "strata/kernels/cpu/exl3_lane_moe.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>

namespace strata::kernels::cpu {

namespace {
constexpr int kGuPiece = 16;     // 16-input slices a gate/up piece: 256 inputs, two input blocks
constexpr int kDownPiece = 8;    // a down piece: 128 inputs, one input block
float silu(float g) { return g / (1.0f + std::exp(-g)); }
}  // namespace

bool Exl3LaneMoe::init(int n_embd, int n_ff, std::string& err) {
    if (n_embd <= 0 || n_ff <= 0 || n_embd % 256 != 0 || n_ff % 128 != 0 || n_ff > n_embd) {
        err = "EXL3 CPU lane: n_embd " + std::to_string(n_embd) + " / n_ff " + std::to_string(n_ff) +
              " do not cut into 256-input / 128-input pieces (or n_ff > n_embd)";
        return false;
    }
    h_ = n_embd;
    ff_ = n_ff;
    pgu_ = n_embd / (16 * kGuPiece);
    pdn_ = n_ff / (16 * kDownPiece);
    hb_ = n_embd / 128;
    const size_t B = kMaxBatch, T = MAXT;
    const size_t n_xg = B * T * (size_t) h_, n_pg = B * (size_t) pgu_ * T * (size_t) ff_, n_xd = B * T * (size_t) ff_,
                 n_pd = B * (size_t) pdn_ * T * (size_t) h_;
    buf_.assign(2 * n_xg + 2 * n_pg + n_xd + n_pd + 16, 0.0f);
    // every region is a multiple of 128 floats: a 64-byte aligned start keeps every row of every region aligned
    off_xg_ = (16 - ((uintptr_t) buf_.data() & 63u) / sizeof(float)) % 16;
    off_xu_ = off_xg_ + n_xg;
    off_pg_ = off_xu_ + n_xg;
    off_pu_ = off_pg_ + n_pg;
    off_xd_ = off_pu_ + n_pg;
    off_pd_ = off_xd_ + n_xd;
    bs_.assign(B * T * 3 * (size_t) hb_, 0.0f);
    return true;
}

bool Exl3LaneMoe::check(const Exl3ExpertJob& j, std::string& err) const {
    auto mat = [&](const Exl3Part& m, int k, int n, const char* what) {
        if (m.trellis != nullptr && m.suh != nullptr && m.svh != nullptr && m.k == k && m.n == n && m.K >= 1 && m.K <= 8)
            return true;
        err = std::string("EXL3 CPU lane: an expert's ") + what + " matrix is " + std::to_string(m.k) + "x" +
              std::to_string(m.n) + " K" + std::to_string(m.K) + (m.trellis == nullptr ? " (no data)" : "") +
              ", expected " + std::to_string(k) + "x" + std::to_string(n) + " K 1..8";
        return false;
    };
    if (j.nt < 1 || j.nt > MAXT) {
        err = "EXL3 CPU lane: an expert job has " + std::to_string(j.nt) + " tokens (1.." + std::to_string(MAXT) + ")";
        return false;
    }
    for (int t = 0; t < j.nt; ++t)
        if (j.x[t] == nullptr || j.out[t] == nullptr) {
            err = "EXL3 CPU lane: an expert job's token " + std::to_string(t) + " has no activation or output row";
            return false;
        }
    return mat(j.e.gate, h_, ff_, "gate") && mat(j.e.up, h_, ff_, "up") && mat(j.e.down, ff_, h_, "down");
}

bool Exl3LaneMoe::run(ExpertPool& pool, const Exl3ExpertJob* jobs, int n, std::string& err) {
    if (h_ == 0) {
        err = "EXL3 CPU lane: not initialised";
        return false;
    }
    for (int i = 0; i < n; ++i)
        if (!check(jobs[i], err)) return false;
    for (int b0 = 0; b0 < n; b0 += kMaxBatch) batch(&pool, jobs + b0, std::min(kMaxBatch, n - b0));
    return true;
}

bool Exl3LaneMoe::run_serial(const Exl3ExpertJob* jobs, int n, std::string& err) {
    if (h_ == 0) {
        err = "EXL3 CPU lane: not initialised";
        return false;
    }
    for (int i = 0; i < n; ++i)
        if (!check(jobs[i], err)) return false;
    for (int b0 = 0; b0 < n; b0 += kMaxBatch) batch(nullptr, jobs + b0, std::min(kMaxBatch, n - b0));
    return true;
}

void Exl3LaneMoe::batch(ExpertPool* pool, const Exl3ExpertJob* jobs, int n) {
    jobs_ = jobs;
    auto phase = [&](int tasks, ExpertPool::TaskFn fn, double& ms) {
        const auto t0 = std::chrono::steady_clock::now();
        if (pool != nullptr) pool->run_tasks(tasks, fn, this);
        else
            for (int i = 0; i < tasks; ++i) fn(this, i, 0);
        ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    };
    phase(n * 2 * pgu_, &task_gu, ms_gu);
    phase(n * (ff_ / 128), &task_mid, ms_mid);
    phase(n * pdn_, &task_down, ms_down);
    phase(n * hb_, &task_out, ms_out);
    for (int i = 0; i < n; ++i) {
        ++experts;
        expert_tokens += jobs[i].nt;
        const Exl3Expert& e = jobs[i].e;
        for (const Exl3Part* m : {&e.gate, &e.up, &e.down})
            bytes += (int64_t) m->k / 16 * (m->n / 16) * 32 * m->K + 2 * (m->k + m->n);
    }
    jobs_ = nullptr;
}

// gate or up, piece p of expert b: its two input blocks transformed (x * suh, H128; their sums kept), then the
// piece's 16 slices for every column and every token
void Exl3LaneMoe::task_gu(void* ctx, int task, int /*thread*/) {
    Exl3LaneMoe& s = *(Exl3LaneMoe*) ctx;
    const int b = task / (2 * s.pgu_), mat = (task / s.pgu_) % 2, p = task % s.pgu_;
    const Exl3ExpertJob& j = s.jobs_[b];
    const Exl3Part& M = mat == 0 ? j.e.gate : j.e.up;
    const float* xh[MAXT] = {};
    float* acc[MAXT] = {};
    for (int t = 0; t < j.nt; ++t) {
        float* xe = mat == 0 ? s.xg(b, t) : s.xu(b, t);
        for (int blk = 2 * p; blk < 2 * p + 2; ++blk) {
            const float sum = exl3_cpu_prepare_block(M.suh + 128 * blk, j.x[t] + 128 * blk, xe + 128 * blk);
            (mat == 0 ? s.bsg(b, t, blk) : s.bsu(b, t, blk)) = sum;
        }
        xh[t] = xe;
        acc[t] = mat == 0 ? s.pg(b, p, t) : s.pu(b, p, t);
    }
    exl3_cpu_rows_acc_n(M, xh, j.nt, kGuPiece * p, kGuPiece * (p + 1), acc);
}

// band c of expert b: gate's and up's pieces summed in order, their outputs (band_out), SwiGLU, and the down matrix's
// input block c (x * suh, H128; its sum kept)
void Exl3LaneMoe::task_mid(void* ctx, int task, int /*thread*/) {
    Exl3LaneMoe& s = *(Exl3LaneMoe*) ctx;
    const int bands = s.ff_ / 128;
    const int b = task / bands, c = task % bands, c0 = 128 * c;
    const Exl3ExpertJob& j = s.jobs_[b];
    for (int t = 0; t < j.nt; ++t) {
        alignas(64) float ag[128], au[128], g[128], u[128], h[128];
        std::memcpy(ag, s.pg(b, 0, t) + c0, sizeof ag);
        std::memcpy(au, s.pu(b, 0, t) + c0, sizeof au);
        for (int p = 1; p < s.pgu_; ++p) {
            const float* qg = s.pg(b, p, t) + c0;
            const float* qu = s.pu(b, p, t) + c0;
            for (int i = 0; i < 128; ++i) {
                ag[i] += qg[i];
                au[i] += qu[i];
            }
        }
        double sg = 0.0, su = 0.0;   // exl3_cpu_prepare's sum: the block sums in order, in double
        for (int blk = 0; blk < s.h_ / 128; ++blk) {
            sg += s.bsg(b, t, blk);
            su += s.bsu(b, t, blk);
        }
        exl3_cpu_band_out(j.e.gate, ag, (float) sg, c0, g);
        exl3_cpu_band_out(j.e.up, au, (float) su, c0, u);
        for (int i = 0; i < 128; ++i) h[i] = silu(g[i]) * u[i];
        s.bsd(b, t, c) = exl3_cpu_prepare_block(j.e.down.suh + c0, h, s.xd(b, t) + c0);
    }
}

// down, piece q of expert b: its 8 slices (input block q) for every column and every token
void Exl3LaneMoe::task_down(void* ctx, int task, int /*thread*/) {
    Exl3LaneMoe& s = *(Exl3LaneMoe*) ctx;
    const int b = task / s.pdn_, q = task % s.pdn_;
    const Exl3ExpertJob& j = s.jobs_[b];
    const float* xh[MAXT] = {};
    float* acc[MAXT] = {};
    for (int t = 0; t < j.nt; ++t) {
        xh[t] = s.xd(b, t);
        acc[t] = s.pd(b, q, t);
    }
    exl3_cpu_rows_acc_n(j.e.down, xh, j.nt, kDownPiece * q, kDownPiece * (q + 1), acc);
}

// band c of expert b's output: down's pieces summed in order, band_out into the token's row
void Exl3LaneMoe::task_out(void* ctx, int task, int /*thread*/) {
    Exl3LaneMoe& s = *(Exl3LaneMoe*) ctx;
    const int b = task / s.hb_, c = task % s.hb_, c0 = 128 * c;
    const Exl3ExpertJob& j = s.jobs_[b];
    for (int t = 0; t < j.nt; ++t) {
        alignas(64) float a[128];
        std::memcpy(a, s.pd(b, 0, t) + c0, sizeof a);
        for (int q = 1; q < s.pdn_; ++q) {
            const float* p = s.pd(b, q, t) + c0;
            for (int i = 0; i < 128; ++i) a[i] += p[i];
        }
        double sd = 0.0;
        for (int blk = 0; blk < s.ff_ / 128; ++blk) sd += s.bsd(b, t, blk);
        exl3_cpu_band_out(j.e.down, a, (float) sd, c0, j.out[t] + c0);
    }
}

}  // namespace strata::kernels::cpu
