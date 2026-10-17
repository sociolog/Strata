// include/strata/kernels/exl3_tier.hpp - the VRAM tier of an EXL3 model's routed experts (--exl3-lane with
// --exl3-lane-profile): the experts a routing profile ranks hottest, copied once from the CPU lane's RAM tier
// (kernels/cpu/exl3_ram.hpp) into one device allocation of equal slots, each slot one expert blob in the RAM tier's
// layout.  The verify windows run them on the GPU (kernels/exl3_dev.hpp: grouped_experts, inside the window's graph)
// while the CPU lane runs the others; the single-token path runs them with the same kernels.  The table is static - no
// swaps - so which experts a run computes where, and with it every row's arithmetic, is fixed for the run.
#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace strata::kernels::cpu { class Exl3ExpertRam; }

namespace strata::kernels {

class Exl3VramTier {
public:
    Exl3VramTier() = default;
    ~Exl3VramTier();
    Exl3VramTier(const Exl3VramTier&) = delete;
    Exl3VramTier& operator=(const Exl3VramTier&) = delete;

    /// As many slots as `budget` bytes hold (each the RAM tier's largest expert), filled in `ranked` order ((layer,
    /// expert) pairs, hottest first; pairs out of range or repeated are skipped, and so are those `exclude` - another
    /// tier's table, [n_layers][n_expert] - holds) from `ram`.  One allocation on `device` (-1: the current one), the
    /// residency table there too - everything before any graph capture.  A budget below one slot leaves the tier empty
    /// (true).  False and a message on a failed allocation or copy, with nothing left allocated.
    bool init(const cpu::Exl3ExpertRam& ram, const std::vector<std::pair<int32_t, int32_t>>& ranked, uint64_t budget,
              std::string& err, int device = -1, const int32_t* exclude = nullptr);
    int device() const { return device_; }

    int64_t slots() const { return slots_; }
    uint64_t slot_bytes() const { return slot_bytes_; }
    uint64_t bytes() const { return (uint64_t) slots_ * slot_bytes_; }
    const uint8_t* base() const { return base_; }
    int64_t n_layers() const { return n_layers_; }
    int64_t n_expert() const { return n_expert_; }
    double load_seconds() const { return load_s_; }
    /// The residency table [n_layers][n_expert]: an expert's slot, or -1 (host; the device copy for the windows).
    const int32_t* h_res() const { return h_res_.empty() ? nullptr : h_res_.data(); }
    const int32_t* d_res() const { return d_res_; }
    /// The expert's blob in VRAM, or null when it is not resident (layer and e in range).
    const uint8_t* blob(int64_t layer, int64_t e) const {
        const int32_t s = h_res_[(size_t) (layer * n_expert_ + e)];
        return s < 0 ? nullptr : base_ + (uint64_t) s * slot_bytes_;
    }

private:
    void release();
    uint8_t* base_ = nullptr;
    int32_t* d_res_ = nullptr;
    std::vector<int32_t> h_res_;
    int device_ = 0;
    int64_t slots_ = 0, n_layers_ = 0, n_expert_ = 0;
    uint64_t slot_bytes_ = 0;
    double load_s_ = 0.0;
};

}  // namespace strata::kernels
