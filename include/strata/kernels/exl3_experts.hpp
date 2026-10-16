// include/strata/kernels/exl3_experts.hpp - one layer's routed EXL3 experts resident on the GPU.
//
// The engine's CPU expert pool is blob/quantized-activation oriented; EXL3 experts are per-expert
// trellis/suh/svh tensors (docs/EXL3.md), so this is the GPU-side counterpart: upload a layer's experts
// once and run the MoE (`exl3_moe_ffn`) for a token's routed experts.
#pragma once

#include <cstdint>
#include <string>

#include "strata/kernels/exl3.hpp"

namespace strata::kernels {

class Exl3ExpertStore {
public:
    // Uploads `n_experts` experts of `layer` from the EXL3 model at `model_dir` to the GPU.  `stream` may
    // be null (default stream).  `layer` is the transformer layer index; expert keys are
    // `model.language_model.layers.<layer>.mlp.experts.<e>.{gate,up,down}_proj`.
    Exl3ExpertStore(const std::string& model_dir, int layer, int n_experts, void* stream);
    ~Exl3ExpertStore();
    Exl3ExpertStore(const Exl3ExpertStore&) = delete;
    Exl3ExpertStore& operator=(const Exl3ExpertStore&) = delete;

    int n_experts() const { return n_experts_; }
    // Bytes of GPU memory the layer's experts occupy (for the engine's budget).
    size_t bytes() const { return bytes_; }

    // out = sum_i weights[i] * ffn(ids[i], x); x: 2560 fp16, out: 2560 fp16.  ids are expert indices in
    // [0, n_experts), k of them.  `stream` may be null.
    void run(const uint16_t* x, const int* ids, const float* weights, int k, uint16_t* out, void* stream) const;

private:
    struct Impl;
    Impl* impl_ = nullptr;
    int n_experts_ = 0;
    size_t bytes_ = 0;
};

// Streaming EXL3 experts for the engine's pool: reads a token's routed experts straight from the model's
// safetensors (mmap), uploads just those, runs them, and hands back the k UNWEIGHTED rows in fp32 (the
// engine's `moe_combine` applies the router weights).  One instance serves all layers.
class Exl3ExpertStream {
public:
    /// `slots_per_layer`: routed experts kept resident per layer (the same idea as the GGUF path's
    /// `--expert-cache`, counted per layer because the stream is).  0 keeps none: every use reads and
    /// uploads the expert again.  -1 sizes the slots from the free VRAM after this instance's own buffers.
    /// `n_layers` is the model's layer count (the cache spans all of them).
    /// `ram_cache_bytes` is the host-side expert cache (0 = off).  **THE PAGE CACHE IS NOT A CACHE YOU CAN
    /// COUNT ON**: on the development machine a 10.6 GB read of the routed experts did not stay resident, so
    /// every miss was a SATA read (measured 8-91 ms of uploads per token against 1.4 ms of FFN).  A managed
    /// host cache sits outside whatever reclaim policy evicts those pages and is the design the GGUF path
    /// already uses (`--resident-cpu-experts`): a miss is read into RAM, and every later miss of the same
    /// expert is a memcpy at RAM speed (0.2 ms per 1.86 MB against 3.7 ms from the disk).
    Exl3ExpertStream(const std::string& model_dir, int n_embd, int n_layers, int slots_per_layer, void* stream,
                     size_t ram_cache_bytes = 0);
    ~Exl3ExpertStream();
    Exl3ExpertStream(const Exl3ExpertStream&) = delete;
    Exl3ExpertStream& operator=(const Exl3ExpertStream&) = delete;

    // out[i*n_embd] = ffn(layer, expert ids[i], x) for i in [0,k).  `x_f32` and `out` are HOST fp32.
    bool run(int layer, const float* x_f32, const int* ids, int k, float* out, void* stream);

    // ---- --exl3-lane's GPU share: the VRAM cache holds each layer's hottest experts (seeded once, never evicted),
    // the CPU lane computes the rest from its RAM tier.  Nothing is read from the file on these paths.
    /// Seeds layer `layer`'s free slots with `experts` in order (the rest stay on the CPU once the slots are full);
    /// `pieces` holds 9 host pointers an expert: gate trellis, suh, svh, then up's, then down's (the sizes are the
    /// slots').  False and a message if the cache is off or a copy fails.
    bool preload(int layer, const int* experts, int n, const void* const* pieces, std::string& err);
    /// Whether layer `layer`'s VRAM cache holds expert `e` (no LRU update, nothing read).
    bool resident(int layer, int e) const;
    /// Enqueues on `stream` the FFN of the experts ids[i] with is_res[i] != 0 (each must be resident) and the copies of
    /// their rows into out + i * n_embd (HOST fp32, as `run`); returns at once, so the CPU can work meanwhile.
    bool launch_resident(int layer, const float* x_f32, const int* ids, const uint8_t* is_res, int k, float* out,
                         void* stream);
    /// Waits for launch_resident's rows.
    bool finish_resident(void* stream);
    /// launch_resident's experts so far (the hit count of the GPU share).
    long long resident_runs() const;
    size_t bytes() const;
    int slots_per_layer() const;
    /// the host cache's slots per layer and its bytes (0 when it is off)
    int ram_slots_per_layer() const;
    size_t ram_bytes() const;

private:
    struct Impl;
    Impl* impl_ = nullptr;
};

}  // namespace strata::kernels
