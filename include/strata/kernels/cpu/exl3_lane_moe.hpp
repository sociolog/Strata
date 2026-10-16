// include/strata/kernels/cpu/exl3_lane_moe.hpp - EXL3 routed experts on the CPU expert pool.
//
// A batch of experts (a layer's CPU share), each with the tokens routed to it, in four phases of ExpertPool::run_tasks:
//   gate/up  split-K pieces of 256 inputs; each piece first transforms its own two input blocks (x * suh, H128)
//   mid      per 128-column band: the pieces summed in order, the outputs' transform (band_out), SwiGLU, and the
//            down matrix's input block for that band (the down input's 128-blocks are the gate/up output bands)
//   down     split-K pieces of 128 inputs (one input block each)
//   out      per 128-column band: the pieces summed in order, the outputs' transform
// The cuts are fixed by the shapes, never by the thread count, so the rows are the same whatever the pool's size.  The
// rows are UNWEIGHTED: the engine's moe_combine applies the router weights, as for every other expert path.
#pragma once

#include "strata/kernels/cpu/exl3_lane.hpp"
#include "strata/kernels/cpu/pool.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace strata::kernels::cpu {

/// One routed expert's three matrices: gate and up n_embd -> n_ff, down n_ff -> n_embd.
struct Exl3Expert {
    Exl3Part gate, up, down;
};

/// One expert for the `nt` tokens routed to it: x[t] its n_embd F32 activation, out[t] its n_embd F32 output row
/// (overwritten, unweighted).
struct Exl3ExpertJob {
    Exl3Expert e;
    int nt = 0;
    const float* x[MAXT] = {};
    float* out[MAXT] = {};
};

class Exl3LaneMoe {
public:
    /// The model's shape: n_embd a multiple of 256, n_ff a multiple of 128 (the cuts).  False and a message otherwise.
    bool init(int n_embd, int n_ff, std::string& err);
    /// The `n` jobs on `pool`, kMaxBatch experts at a time.  False and a message (nothing half-written is left as a
    /// result: the caller fails the layer) when a job does not match init's shape.
    bool run(ExpertPool& pool, const Exl3ExpertJob* jobs, int n, std::string& err);
    /// The same on the calling thread alone, the same cuts and order (tests: the rows equal run's bit for bit).
    bool run_serial(const Exl3ExpertJob* jobs, int n, std::string& err);
    static constexpr int kMaxBatch = 32;

    /// Accumulated wall time of the four phases (ms), and what went through them.
    double ms_gu = 0, ms_mid = 0, ms_down = 0, ms_out = 0;
    int64_t experts = 0, expert_tokens = 0, bytes = 0;

private:
    bool check(const Exl3ExpertJob& j, std::string& err) const;
    void batch(ExpertPool* pool, const Exl3ExpertJob* jobs, int n);
    static void task_gu(void* ctx, int task, int thread);
    static void task_mid(void* ctx, int task, int thread);
    static void task_down(void* ctx, int task, int thread);
    static void task_out(void* ctx, int task, int thread);
    float* xg(int b, int t) { return buf_.data() + off_xg_ + ((size_t) b * MAXT + t) * (size_t) h_; }
    float* xu(int b, int t) { return buf_.data() + off_xu_ + ((size_t) b * MAXT + t) * (size_t) h_; }
    float* pg(int b, int p, int t) { return buf_.data() + off_pg_ + (((size_t) b * pgu_ + p) * MAXT + t) * (size_t) ff_; }
    float* pu(int b, int p, int t) { return buf_.data() + off_pu_ + (((size_t) b * pgu_ + p) * MAXT + t) * (size_t) ff_; }
    float* xd(int b, int t) { return buf_.data() + off_xd_ + ((size_t) b * MAXT + t) * (size_t) ff_; }
    float* pd(int b, int q, int t) { return buf_.data() + off_pd_ + (((size_t) b * pdn_ + q) * MAXT + t) * (size_t) h_; }
    float& bsg(int b, int t, int blk) { return bs_[(((size_t) b * MAXT + t) * 3 + 0) * (size_t) hb_ + blk]; }
    float& bsu(int b, int t, int blk) { return bs_[(((size_t) b * MAXT + t) * 3 + 1) * (size_t) hb_ + blk]; }
    float& bsd(int b, int t, int blk) { return bs_[(((size_t) b * MAXT + t) * 3 + 2) * (size_t) hb_ + blk]; }

    int h_ = 0, ff_ = 0;
    int pgu_ = 0, pdn_ = 0;     // split-K pieces: gate/up (256 inputs each), down (128 each)
    int hb_ = 0;                // 128-blocks of n_embd (>= those of n_ff)
    size_t off_xg_ = 0, off_xu_ = 0, off_pg_ = 0, off_pu_ = 0, off_xd_ = 0, off_pd_ = 0;
    std::vector<float> buf_;
    std::vector<float> bs_;
    const Exl3ExpertJob* jobs_ = nullptr;   // the batch in flight
};

}  // namespace strata::kernels::cpu
