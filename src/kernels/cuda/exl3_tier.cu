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
    if (base_ != nullptr) (void) cudaFree(base_);
    if (d_res_ != nullptr) (void) cudaFree(d_res_);
    base_ = nullptr;
    d_res_ = nullptr;
    h_res_.clear();
    slots_ = 0;
}

bool Exl3VramTier::init(const cpu::Exl3ExpertRam& ram, const std::vector<std::pair<int32_t, int32_t>>& ranked,
                        uint64_t budget, std::string& err, int device, const int32_t* exclude) {
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
    const int64_t want = std::min<int64_t>((int64_t) (budget / slot_bytes_), (int64_t) ranked.size());
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
    slots_ = (int64_t) take.size();
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
        for (int64_t s = 0; s < slots_; ++s) {
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

}  // namespace strata::kernels
