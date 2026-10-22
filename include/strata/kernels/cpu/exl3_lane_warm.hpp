// include/strata/kernels/cpu/exl3_lane_warm.hpp - proto STRATA_EXL3_LANE_WARM: the EXL3 CPU lane's cache warm-up.
//
// Between two layers of a verify window the lane's workers have nothing to do while the GPU runs the next layer's
// attention and router (~0.28 ms a layer on the 6.05, 3090).  After layer l's lane the workers score layer l + 1's
// router on layer l's expert input (the residual stream changes little from one layer to the next: a prediction only),
// take every token's top k, rank the predicted experts that only the lane holds by their summed weight, and read the
// likeliest ones into their caches - gate/up first, the order the lane computes them - until the next layer's doorbell
// stops them.  Layer 0 (whose input is the next round's first token) gets the last round's layer-0 lane experts.  Only
// caches are warmed: which experts run, where and how, is unchanged, so the rows are bit for bit the same.
//
// Ported from Basalt's CPU tier warm-up (src/core/verify.cpp CpuWarm, warm_logits / warm_select / warm_job; MIT, see
// third_party/basalt/LICENSE): the router in int8 with one scale per 32 values, the AVX2 dot products, the
// selection.  4Mile's version takes the lane's F32 rows (quantized here, not the ring's q8_1), warms several experts
// in rank order rather than one, runs as an ExpertPool background phase, and keeps hit statistics (--stats).
#pragma once

#include "strata/kernels/cpu/pool.hpp"

#include <atomic>
#include <cstdint>
#include <string>
#include <vector>

namespace strata::kernels::cpu {

class Exl3LaneWarm {
public:
    /// The expert's RAM blob when the lane would run it (no VRAM tier holds it), else null; `ctx` is the caller's.
    using LaneBlobFn = const uint8_t* (*)(void* ctx, int layer, int e);
    ~Exl3LaneWarm() { if (pool_ != nullptr) pool_->stop_background(); }
    /// `routers[l]`: layer l's router (ffn_gate_inp) as BF16 bits, n_expert rows of n_embd (n_embd a multiple of 32).
    /// `k`: the model's experts per token; `nsel`: how many predicted experts at most are warmed a layer.
    bool init(const std::vector<std::vector<uint16_t>>& routers, int n_embd, int n_expert, int k, int nsel,
              LaneBlobFn lane_blob, void* ctx, std::string& err);
    bool ready() const { return n_layers_ > 0; }
    /// After layer `layer`'s lane, on the host thread: predict layer + 1 from `x` (n_tok rows n_embd apart, the
    /// layer's expert input) and start the warm-up on `pool`'s workers (the host returns at once).  For the last
    /// layer: layer 0's experts of the last round (`first_blobs`).  `blob_bytes(l)`: a layer's blob size.
    void start(ExpertPool& pool, int layer, const float* x, int n_tok, uint64_t blob_bytes_next);
    /// Before layer `layer`'s lane (the warm-up may still run: only a finished prediction is read): score the
    /// prediction made for it against the experts the lane really runs (`lane`, n), and remember layer 0's for the next
    /// round.
    void score(int layer, const int32_t* lane, int n);
    // --stats
    int64_t st_layers = 0, st_late = 0, st_pred = 0, st_lane = 0, st_hit = 0, st_warmed_hit = 0, st_warm_starts = 0;
    std::atomic<int64_t> st_bytes{0}, st_done{0};   // bytes read by the workers; warm-ups that read all their experts

private:
    static void task(void* ctx, int task, int thread);
    void logits(int e0, int e1);
    void select();
    std::vector<int8_t> rq_;   // [n_layers][n_expert][n_embd]
    std::vector<float> rs_;    // [n_layers][n_expert][n_embd / 32]
    int n_layers_ = 0, n_expert_ = 0, n_embd_ = 0, k_ = 10, nsel_ = 3;
    LaneBlobFn lane_blob_ = nullptr;
    void* ctx_ = nullptr;
    ExpertPool* pool_ = nullptr;
    // the job in flight
    int layer_ = -1, n_tok_ = 0, W_ = 1;
    uint64_t bytes_ = 0;
    std::vector<int8_t> xq_;   // [MAXT][n_embd]
    std::vector<float> xs_;    // [MAXT][n_embd / 32]
    std::vector<float> lg_;    // [MAXT][n_expert]
    const float* x_ = nullptr;
    alignas(64) std::atomic<int> xready_{0};
    alignas(64) std::atomic<int> arrive_{0};
    alignas(64) std::atomic<int> ready_{0};
    alignas(64) std::atomic<int> left_{0};   // tasks still reading (the last one counts a warm-up read in full)
    const uint8_t* blobs_[16] = {};
    int nblobs_ = 0;
    int32_t sel_[16] = {};
    bool history_ = false;   // this job warms layer 0 from the last round
    // the prediction per layer (for score), and layer 0's lane experts of the last round
    std::vector<std::vector<int32_t>> pred_, warmed_;
    std::vector<int32_t> first_;
    std::atomic<uint64_t> sink_{0};
};

}  // namespace strata::kernels::cpu
