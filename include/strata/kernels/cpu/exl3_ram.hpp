// include/strata/kernels/cpu/exl3_ram.hpp - the resident RAM tier of an EXL3 model's routed experts.
//
// Every routed expert of every layer is copied once out of the model's safetensors (its nine tensors: the three
// trellises and the six scale vectors) into one anonymous mapping, transparent huge pages asked for, so the CPU lane
// (exl3_lane_moe.hpp) reads RAM and never the file.  One expert is one blob - gate, up, down trellis, then suh / svh of
// gate, up, down - every part 64-byte aligned: the file keeps its trellises at offsets 2 mod 4, which never reach the
// kernels.  The reads run on several threads in file order and drop the page cache behind them, so the load does not
// hold the model twice.  Plain (not pinned) memory: the GPU never reads this copy (the PCIe share and the helper card
// are later steps).
#pragma once

#include "strata/kernels/cpu/exl3_lane_moe.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace strata::kernels::cpu {

class Exl3ExpertRam {
public:
    Exl3ExpertRam() = default;
    ~Exl3ExpertRam();
    Exl3ExpertRam(const Exl3ExpertRam&) = delete;
    Exl3ExpertRam& operator=(const Exl3ExpertRam&) = delete;

    /// Copies the routed experts of layers [0, n_layers) (`model.language_model.layers.L.mlp.experts.E.*`) into RAM on
    /// `threads` threads.  `budget` > 0 refuses a tier larger than that many bytes.  False and a message on a missing
    /// tensor, a codebook other than mul1, an expert whose shape differs from its layer's, a short read or a failed
    /// allocation; nothing is left half-loaded.
    bool open(const std::string& model_dir, int n_layers, int n_expert, int threads, uint64_t budget, std::string& err);
    bool loaded() const { return base_ != nullptr; }
    /// Pointers into the tier (valid while it lives); layer and expert must be in range.
    Exl3Expert expert(int layer, int e) const;
    uint64_t bytes() const { return bytes_; }
    double load_seconds() const { return load_s_; }
    int n_layers() const { return (int) lay_.size(); }
    int n_expert() const { return n_expert_; }
    /// e.g. "48 layers x 512 experts, gate/up/down K 4/4/4 (2.48 MB an expert), 56.7 GiB"
    std::string describe() const;

private:
    struct LayerFmt {
        int kg = 0, ku = 0, kd = 0;          // K of gate, up, down
        int h = 0, ff = 0;                   // n_embd, n_ff
        uint64_t blob = 0;                   // bytes an expert
        uint64_t up = 0, down = 0, suh_g = 0, svh_g = 0, suh_u = 0, svh_u = 0, suh_d = 0, svh_d = 0;   // offsets in the blob
        uint64_t base = 0;                   // the layer's first expert in the tier
    };
    std::vector<LayerFmt> lay_;
    int n_expert_ = 0;
    uint8_t* base_ = nullptr;
    uint64_t bytes_ = 0, map_bytes_ = 0;
    double load_s_ = 0.0;
};

}  // namespace strata::kernels::cpu
