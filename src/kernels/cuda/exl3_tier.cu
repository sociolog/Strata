// src/kernels/cuda/exl3_tier.cu - the VRAM tier of an EXL3 model's routed experts (include/strata/kernels/exl3_tier.hpp).
#include "strata/kernels/exl3_tier.hpp"

#include "strata/kernels/cpu/exl3_ram.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>

namespace strata::kernels {

namespace {
// the tier's device current for a scope, the caller's back after it
struct DeviceScope {
    int prev = -1;
    explicit DeviceScope(int dev) {
        if (dev < 0 || cudaGetDevice(&prev) != cudaSuccess || prev == dev) { prev = -1; return; }
        (void) cudaSetDevice(dev);
    }
    ~DeviceScope() { if (prev >= 0) (void) cudaSetDevice(prev); }
};
}  // namespace

Exl3VramTier::~Exl3VramTier() { release(); }

void Exl3VramTier::release() {
    const DeviceScope on(device_);
    if (copy_stream_ != nullptr) {
        (void) cudaStreamSynchronize((cudaStream_t) copy_stream_);
        (void) cudaStreamDestroy((cudaStream_t) copy_stream_);
    }
    if (base_ != nullptr) (void) cudaFree(base_);
    if (d_res_ != nullptr) (void) cudaFree(d_res_);
    copy_stream_ = nullptr;
    base_ = nullptr;
    d_res_ = nullptr;
    h_res_.clear();
    owner_.clear();
    slots_ = 0;
    filled_ = 0;
}

bool Exl3VramTier::init(const cpu::Exl3ExpertRam& ram, const std::vector<std::pair<int32_t, int32_t>>& ranked,
                        uint64_t budget, std::string& err, int device, const int32_t* exclude, int64_t spares) {
    release();
    const auto t0 = std::chrono::steady_clock::now();
    if (device < 0 && cudaGetDevice(&device) != cudaSuccess) {
        err = "EXL3 VRAM tier: no current device";
        return false;
    }
    device_ = device;
    const DeviceScope on(device_);
    n_layers_ = ram.n_layers();
    n_expert_ = ram.n_expert();
    if (n_layers_ <= 0 || n_expert_ <= 0) {
        err = "EXL3 VRAM tier: the RAM tier is empty";
        return false;
    }
    slot_bytes_ = 0;
    for (int64_t l = 0; l < n_layers_; ++l) slot_bytes_ = std::max<uint64_t>(slot_bytes_, ram.blob_bytes((int) l));
    // slots start 256-byte aligned (every part of a blob is 64-byte aligned within it, as in the RAM tier)
    slot_bytes_ = (slot_bytes_ + 255) & ~(uint64_t) 255;
    h_res_.assign((size_t) (n_layers_ * n_expert_), -1);
    const int64_t fit = (int64_t) (budget / slot_bytes_);
    const int64_t n_spare = std::clamp<int64_t>(spares, 0, fit);
    const int64_t want = std::min<int64_t>(fit - n_spare, (int64_t) ranked.size());
    std::vector<std::pair<int32_t, int32_t>> take;
    take.reserve((size_t) std::max<int64_t>(want, 0));
    for (const auto& pr : ranked) {
        if ((int64_t) take.size() >= want) break;
        if (pr.first < 0 || pr.first >= n_layers_ || pr.second < 0 || pr.second >= n_expert_) continue;
        const size_t i = (size_t) (pr.first * n_expert_ + pr.second);
        if (exclude != nullptr && exclude[i] >= 0) continue;   // another tier holds it
        int32_t& r = h_res_[i];
        if (r >= 0) continue;   // a repeated pair
        r = (int32_t) take.size();
        take.push_back(pr);
    }
    filled_ = (int64_t) take.size();
    slots_ = filled_ + n_spare;
    owner_.assign((size_t) slots_, -1);
    for (int64_t s = 0; s < filled_; ++s) owner_[(size_t) s] = take[(size_t) s].first * (int32_t) n_expert_ + take[(size_t) s].second;
    auto fail = [&](const std::string& what, cudaError_t e) {
        err = "EXL3 VRAM tier (device " + std::to_string(device_) + "): " + what + ": " + cudaGetErrorString(e);
        (void) cudaGetLastError();
        release();
        return false;
    };
    if (cudaError_t e = cudaMalloc((void**) &d_res_, h_res_.size() * sizeof(int32_t)); e != cudaSuccess)
        return fail("the residency table", e);
    if (slots_ > 0) {
        if (cudaError_t e = cudaMalloc((void**) &base_, (size_t) slots_ * slot_bytes_); e != cudaSuccess)
            return fail(std::to_string(slots_) + " slots of " + std::to_string(slot_bytes_) + " bytes", e);
        // one copy a slot, in order, from the RAM tier (page-locked or not: the driver stages a pageable source)
        for (int64_t s = 0; s < filled_; ++s) {
            const auto [l, e] = take[(size_t) s];
            if (cudaError_t ce = cudaMemcpy(base_ + (uint64_t) s * slot_bytes_, ram.blob(l, e), (size_t) ram.blob_bytes(l),
                                            cudaMemcpyHostToDevice);
                ce != cudaSuccess)
                return fail("filling slot " + std::to_string(s), ce);
        }
    }
    if (cudaError_t e = cudaMemcpy(d_res_, h_res_.data(), h_res_.size() * sizeof(int32_t), cudaMemcpyHostToDevice);
        e != cudaSuccess)
        return fail("the residency table's upload", e);
    load_s_ = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return true;
}

bool Exl3VramTier::assign(int64_t slot, int32_t key, std::string& err) {
    if (slot < 0 || slot >= slots_ || owner_[(size_t) slot] >= 0 || key < 0 || key >= (int32_t) h_res_.size() ||
        h_res_[(size_t) key] >= 0) {
        err = "EXL3 VRAM tier: assign of expert " + std::to_string(key) + " to slot " + std::to_string(slot) +
              ": the slot is taken or the expert resident already";
        return false;
    }
    h_res_[(size_t) key] = (int32_t) slot;
    owner_[(size_t) slot] = key;
    const DeviceScope on(device_);
    if (const cudaError_t e = cudaMemcpy(d_res_ + key, &h_res_[(size_t) key], sizeof(int32_t), cudaMemcpyHostToDevice);
        e != cudaSuccess) {
        err = std::string("EXL3 VRAM tier: the residency table's update: ") + cudaGetErrorString(e);
        return false;
    }
    return true;
}

bool Exl3VramTier::release_slot(int64_t slot, std::string& err) {
    const int32_t key = slot >= 0 && slot < slots_ ? owner_[(size_t) slot] : -1;
    if (key < 0) {
        err = "EXL3 VRAM tier: slot " + std::to_string(slot) + " holds no expert";
        return false;
    }
    h_res_[(size_t) key] = -1;
    owner_[(size_t) slot] = -1;
    const DeviceScope on(device_);
    if (const cudaError_t e = cudaMemcpy(d_res_ + key, &h_res_[(size_t) key], sizeof(int32_t), cudaMemcpyHostToDevice);
        e != cudaSuccess) {
        err = std::string("EXL3 VRAM tier: the residency table's update: ") + cudaGetErrorString(e);
        return false;
    }
    return true;
}

bool Exl3VramTier::upload_table(std::string& err) {
    const DeviceScope on(device_);
    // a pageable copy can return before its DMA has landed: wait for it (the legacy stream) before a kernel reads it
    cudaError_t e = cudaMemcpy(d_res_, h_res_.data(), h_res_.size() * sizeof(int32_t), cudaMemcpyHostToDevice);
    if (e == cudaSuccess) e = cudaStreamSynchronize(nullptr);
    if (e != cudaSuccess) {
        err = std::string("EXL3 VRAM tier: the residency table's upload: ") + cudaGetErrorString(e);
        return false;
    }
    return true;
}

bool Exl3VramTier::lend(int64_t first, std::vector<Lent>& out, std::string& err) {
    out.clear();
    if (first < 0 || first > slots_) {
        err = "EXL3 VRAM tier: a loan from slot " + std::to_string(first) + " of " + std::to_string(slots_);
        return false;
    }
    for (int64_t s = first; s < slots_; ++s) {
        const int32_t key = owner_[(size_t) s];
        if (key < 0) continue;   // a spare: nothing to give back
        out.push_back({(int32_t) s, key});
        h_res_[(size_t) key] = -1;
        owner_[(size_t) s] = -1;
    }
    return out.empty() || upload_table(err);
}

bool Exl3VramTier::restore(const std::vector<Lent>& lent, const cpu::Exl3ExpertRam& ram, std::string& err) {
    restore_s_ = 0.0;
    if (lent.empty()) return true;
    const auto t0 = std::chrono::steady_clock::now();
    if (ram.n_layers() != n_layers_ || ram.n_expert() != n_expert_) {
        err = "EXL3 VRAM tier: the refill's RAM tier is not the one the tier was filled from";
        return false;
    }
    for (const Lent& x : lent)
        if (x.slot < 0 || x.slot >= slots_ || owner_[(size_t) x.slot] >= 0 || x.key < 0 ||
            x.key >= (int32_t) h_res_.size() || h_res_[(size_t) x.key] >= 0) {
            err = "EXL3 VRAM tier: refilling expert " + std::to_string(x.key) + " into slot " + std::to_string(x.slot) +
                  ": the slot is taken or the expert resident already";
            return false;
        }
    const DeviceScope on(device_);
    if (copy_stream_ == nullptr) {
        cudaStream_t s = nullptr;
        if (const cudaError_t e = cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking); e != cudaSuccess) {
            err = std::string("EXL3 VRAM tier: the refill's copy stream: ") + cudaGetErrorString(e);
            return false;
        }
        copy_stream_ = s;
    }
    const cudaStream_t s = (cudaStream_t) copy_stream_;
    // each expert into its own slot again (the tables still read it as a host expert until every copy has landed)
    for (const Lent& x : lent) {
        const int l = (int) (x.key / n_expert_), e = (int) (x.key % n_expert_);
        if (const cudaError_t ce = cudaMemcpyAsync(slot_ptr(x.slot), ram.blob(l, e), (size_t) ram.blob_bytes(l),
                                                   cudaMemcpyHostToDevice, s);
            ce != cudaSuccess) {
            (void) cudaStreamSynchronize(s);
            err = "EXL3 VRAM tier: refilling slot " + std::to_string(x.slot) + ": " + cudaGetErrorString(ce);
            return false;
        }
    }
    if (const cudaError_t e = cudaStreamSynchronize(s); e != cudaSuccess) {
        err = std::string("EXL3 VRAM tier: the refill's copies: ") + cudaGetErrorString(e);
        return false;
    }
    for (const Lent& x : lent) {
        owner_[(size_t) x.slot] = x.key;
        h_res_[(size_t) x.key] = x.slot;
    }
    if (!upload_table(err)) return false;
    restore_s_ = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return true;
}

}  // namespace strata::kernels
