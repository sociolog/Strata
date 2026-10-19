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
    /// (true).  False and a message on a failed allocation or copy, with nothing left allocated.  `spares`: that many of
    /// the slots are left empty, for an adaptive caller (assign / release_slot); they count in the budget.
    bool init(const cpu::Exl3ExpertRam& ram, const std::vector<std::pair<int32_t, int32_t>>& ranked, uint64_t budget,
              std::string& err, int device = -1, const int32_t* exclude = nullptr, int64_t spares = 0);
    /// proto (the elastic K/V in --serve): the slots in memory the caller owns - a VMM range of `budget` bytes, all
    /// mapped - instead of the tier's own allocation; the tier never frees it.  Call before init.
    void use_external(uint8_t* base) { ext_base_ = base; }
    uint64_t slot_offset(int64_t slot) const { return (uint64_t) slot * slot_bytes_; }
    const int32_t* h_res_table() const { return h_res_.data(); }
    int device() const { return device_; }

    int64_t slots() const { return slots_; }               ///< every slot, the spares included
    int64_t filled() const { return filled_; }             ///< the slots filled from the ranking at init
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
    /// Adaptive use (--exl3-tier-adapt): a slot's expert (layer * n_expert + e) or -1 (a spare); the slot's bytes; an
    /// expert put in a spare slot its blob was copied into, and a slot given back as a spare.  Both tables change -
    /// the host's and the device copy (synchronously) - so a caller does it only while no window or token graph runs.
    int32_t owner(int64_t slot) const { return owner_[(size_t) slot]; }
    uint8_t* slot_ptr(int64_t slot) const { return base_ + (uint64_t) slot * slot_bytes_; }
    // `upload` false: the host table only - the caller uploads the whole table once after a batch (upload_table)
    bool assign(int64_t slot, int32_t key, std::string& err, bool upload = true);
    bool release_slot(int64_t slot, std::string& err, bool upload = true);

    /// --serve: the batched prompt path borrows the slots [first, slots()) for its buffers, and they are refilled
    /// before a window reads the tier again.  `lend` takes the experts those slots hold out of both tables (one upload
    /// of the device copy; they read as host experts, the lane's, until `restore`) and returns what each slot held -
    /// a spare slot lends nothing.  `restore` copies every one back into its own slot from `ram` (asynchronously on the
    /// tier's copy stream, then waits) and puts it back in both tables (one upload), so the tier is what it was before
    /// the loan.  Only while no window or token graph runs, and with no adaptive copy in flight into the range.
    struct Lent {
        int32_t slot, key;   // key: layer * n_expert + e
    };
    bool lend(int64_t first, std::vector<Lent>& out, std::string& err);
    bool lend_range(int64_t first, int64_t last, std::vector<Lent>& out, std::string& err);   ///< [first, last)
    bool restore(const std::vector<Lent>& lent, const cpu::Exl3ExpertRam& ram, std::string& err);
    double restore_seconds() const { return restore_s_; }   ///< the last restore's copies and table upload

    bool upload_table(std::string& err);   // the host table to the device copy (synchronous)

private:
    void release();
    uint8_t* base_ = nullptr;
    uint8_t* ext_base_ = nullptr;   // use_external: base_ points here and is not freed
    int32_t* d_res_ = nullptr;
    void* copy_stream_ = nullptr;   // restore's copies (cudaStream_t), made at its first use
    std::vector<int32_t> h_res_;
    std::vector<int32_t> owner_;   // slot -> its expert, -1 a spare
    int device_ = 0;
    int64_t slots_ = 0, filled_ = 0, n_layers_ = 0, n_expert_ = 0;
    uint64_t slot_bytes_ = 0;
    double load_s_ = 0.0, restore_s_ = 0.0;
};

}  // namespace strata::kernels
