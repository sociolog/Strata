// src/kernels/cuda/dense_tc.cu - see include/strata/kernels/dense_tc.hpp.
//
// Ported from Basalt (github.com/jesdga95/basalt @ fb0944c: src/kernels/cuda/dense_tc.cu, the ring kernel
// dense_tc_block / launch_s and the Q8_0 branch of half_sb_terms).  Copyright (c) 2026 Jesús Gulfo, MIT License; the
// full notice is in third_party/basalt/LICENSE.  Changes: Q8_0 only; the __CUDA_ARCH__ >= 900 guards are >= 800 (the
// ring uses only cp.async.cg and mma.sync, both sm_80); no programmatic dependent launch (sm_90 only); the TMA
// one-tile / whole-tile / pair / L2-prefetch paths are dropped (sm_90+); the launch shape is picked per shape for
// sm_86 by STRATA's own benchmark (dense_tc_bench).
//
// A tile is 16 MT rows; its 512 KS-value chunks are one ring slot each.  Warp w of a tile takes m-tile (w / 4) % MT
// (16 rows) and blocks 4 (w % 4) .. 4 (w % 4) + 3 of every 512-value half (KS = 2: warp group w / (4 MT) takes half
// ks of the 1024-value chunk).  One mma per block gives the exact integer dots of 16 rows x 8 columns, and each lane
// adds its four (row, column) terms in a fixed order: blocks in ascending order within its four, chunks in order.  The
// four warps of an m-tile are then added in order, then the KS groups.  The result does not depend on MT, the grid,
// or the column count.
//
// Persistent: block b takes tiles b, b + G, ...; their chunks stream through one ring of S slots (16-byte cp.async per
// thread, issued S - 1 chunks ahead across tile boundaries).  The q8_1 columns are read through L1.
#include "strata/kernels/dense_tc.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace strata::kernels {
namespace {

constexpr int kMaxCols = 8;
[[maybe_unused]] constexpr int kQ81 = 36;   // bytes of a q8_1 block: half2 (d, d * sum), 32 codes
constexpr int kSB = 272;   // eight 34-byte Q8_0 blocks (256 values)

// a staged row of a slot: KS * 2 "super-blocks" from the 16-byte unit holding their first byte, padded to an odd number
// of 16-byte units so the eight rows a load touches fall in different banks
__host__ __device__ constexpr int row_stride(int sb) {
    const int r = (2 * sb + 15 + 15) & ~15;
    return ((r / 16) & 1) ? r : r + 16;
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
__device__ __forceinline__ void mma_k32(int (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                        uint32_t b1) {
    asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};\n"
        : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(0));
}
__device__ __forceinline__ uint32_t ld32(const uint8_t* p) { return *reinterpret_cast<const uint32_t*>(p); }
__device__ __forceinline__ uint32_t ld32_b2(const uint8_t* p) {   // 2-byte aligned
    const uint16_t* q = reinterpret_cast<const uint16_t*>(p);
    return (uint32_t) q[0] | ((uint32_t) q[1] << 16);
}

// The column data of a warp's four blocks: column g's codes 4t.. (b0) and 16 + 4t.. (b1), and the q8_1 scales of
// columns 2t and 2t + 1.
struct XFrag {
    uint32_t b0[4], b1[4];
    float d8[4][2];
};

// Blocks 4n..4n + 3 of the 256-value groups at p[0] (row g) and p[1] (row g + 8): the four (row, column) terms added
// to acc.  Q8_0: d, qs[32] per block; 2-byte aligned (the odd blocks' codes 4-byte).
__device__ __forceinline__ void q8_0_terms(float (&acc)[4], const uint8_t* const (&p)[2], int n, int t, const XFrag& x) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int j = 4 * n + i;
        uint32_t lo[2], hi[2];
        float dw[2];
#pragma unroll
        for (int h = 0; h < 2; ++h) {   // values 4t.. and 16 + 4t.. of block j
            const uint8_t* bp = p[h] + 34 * j;
            dw[h] = __half2float(*reinterpret_cast<const __half*>(bp));
            lo[h] = (i & 1) ? ld32(bp + 2 + 4 * t) : ld32_b2(bp + 2 + 4 * t);
            hi[h] = (i & 1) ? ld32(bp + 18 + 4 * t) : ld32_b2(bp + 18 + 4 * t);
        }
        int d[4];
        mma_k32(d, lo[0], lo[1], hi[0], hi[1], x.b0[i], x.b1[i]);
#pragma unroll
        for (int q = 0; q < 4; ++q) acc[q] += x.d8[i][q & 1] * (dw[q >> 1] * (float) d[q]);
    }
}

// block blk of nblk; smem: smem_bytes<S, MT, KS>() bytes, 128-byte aligned
template<int S, int MT, int KS>
__device__ __forceinline__ void dense_tc_block(int blk, int nblk, uint8_t* smem, const uint8_t* __restrict__ w,
                                               const uint8_t* __restrict__ x, float* __restrict__ y, int n_in,
                                               int n_out, int ncols) {
    constexpr int SB = kSB, CB = 2 * KS * SB, RS = row_stride(KS * SB), R = 16 * MT, SLOT = R * RS;
    constexpr int NU = RS / 16, NT = MT * KS * 128;
    uint8_t* ring = smem;
    float* red = reinterpret_cast<float*>(smem + S * SLOT);   // [warp][row][column]
    const int tid = threadIdx.x, wp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
    const int mt = (wp >> 2) % MT, ks = wp / (4 * MT), hb = wp & 3, sbi = 2 * ks + (hb >> 1), n = hb & 1;
    const int nchunk = n_in / (512 * KS), nb32 = n_in / 32;
    const size_t row_bytes = (size_t) (n_in / 256) * SB;
    const int ntiles = (n_out + R - 1) / R;
    const int mine = blk < ntiles ? (ntiles - 1 - blk) / nblk + 1 : 0;
    const int nunits = mine * nchunk;
    auto issue = [&](int u) {   // unit u = (tile, chunk) into slot u % S
        const int tile = blk + (u / nchunk) * nblk, c = u % nchunk;
        uint8_t* dst = ring + (u % S) * SLOT;
        for (int q = tid; q < R * NU; q += NT) {
            const int r = q / NU, k = q - r * NU;
            const int row = min(tile * R + r, n_out - 1);
            const uintptr_t src = (uintptr_t) (w + (size_t) row * row_bytes + (size_t) c * CB);
            const uintptr_t a0 = src & ~(uintptr_t) 15;
            if (k * 16 < (int) (src - a0) + CB)
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                             ::"r"((unsigned) __cvta_generic_to_shared(dst + r * RS + 16 * k)), "l"(a0 + 16 * k)
                             : "memory");
        }
    };
    auto commit = [] { asm volatile("cp.async.commit_group;\n" ::: "memory"); };
#pragma unroll
    for (int q = 0; q < S - 1; ++q) {
        if (q < nunits) issue(q);
        commit();
    }
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int u = 0; u < nunits; ++u) {
        const int tile = blk + (u / nchunk) * nblk, c = u % nchunk;
        XFrag xf;   // read before the wait: it does not depend on the slot
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int k = 16 * (KS * c + ks) + 4 * hb + i;   // the 32-value block of the columns
            xf.b0[i] = xf.b1[i] = 0;
            if (g < ncols) {
                const uint8_t* xb = x + ((size_t) g * nb32 + k) * kQ81;
                xf.b0[i] = __ldg(reinterpret_cast<const unsigned int*>(xb + 4) + t);
                xf.b1[i] = __ldg(reinterpret_cast<const unsigned int*>(xb + 20) + t);
            }
#pragma unroll
            for (int e = 0; e < 2; ++e) {
                const int col = 2 * t + e;
                xf.d8[i][e] = col < ncols ? __low2float(__ldg(reinterpret_cast<const half2*>(
                                                x + ((size_t) col * nb32 + k) * kQ81)))
                                          : 0.0f;
            }
        }
        asm volatile("cp.async.wait_group %0;\n" ::"n"(S - 2) : "memory");
        __syncthreads();   // unit u is in for every thread; the slot of unit u - 1 is free
        if (u + S - 1 < nunits) issue(u + S - 1);
        commit();
        const int row0 = tile * R;
        const int rA = min(row0 + 16 * mt + g, n_out - 1), rB = min(row0 + 16 * mt + g + 8, n_out - 1);
        const uint8_t* slot = ring + (u % S) * SLOT;
        const uint8_t* const p[2] = {
            slot + (16 * mt + g) * RS + ((uintptr_t) (w + (size_t) rA * row_bytes + (size_t) c * CB) & 15) + sbi * SB,
            slot + (16 * mt + g + 8) * RS + ((uintptr_t) (w + (size_t) rB * row_bytes + (size_t) c * CB) & 15) +
                sbi * SB};
        q8_0_terms(acc, p, n, t, xf);
        if (c == nchunk - 1) {   // the tile: each m-tile's warp sums, added in warp order (hb, then ks)
            float* rw = red + wp * 16 * kMaxCols;
            rw[g * kMaxCols + 2 * t] = acc[0];
            rw[g * kMaxCols + 2 * t + 1] = acc[1];
            rw[(g + 8) * kMaxCols + 2 * t] = acc[2];
            rw[(g + 8) * kMaxCols + 2 * t + 1] = acc[3];
            acc[0] = acc[1] = acc[2] = acc[3] = 0.0f;
            __syncthreads();
            const int row = tid / kMaxCols, col = tid % kMaxCols;   // R * kMaxCols threads
            if (tid < R * kMaxCols && col < ncols && row0 + row < n_out) {
                float v = 0.0f;
#pragma unroll
                for (int q = 0; q < KS; ++q) {
                    const float* rr = red + ((4 * (q * MT + row / 16)) * 16 + row % 16) * kMaxCols + col;
                    const float vq = ((rr[0] + rr[16 * kMaxCols]) + rr[32 * kMaxCols]) + rr[48 * kMaxCols];
                    v = q == 0 ? vq : v + vq;
                }
                y[(size_t) col * n_out + row0 + row] = v;
            }
            // (the next tile's red writes come after at least one more __syncthreads in the loop above)
        }
    }
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
}
#endif

template<int S, int MT, int KS>
__global__ void __launch_bounds__(MT * KS * 128) dense_tc_kernel(const uint8_t* __restrict__ w,
                                                                 const uint8_t* __restrict__ x, float* __restrict__ y,
                                                                 int n_in, int n_out, int ncols) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    extern __shared__ __align__(128) uint8_t smem[];
    dense_tc_block<S, MT, KS>((int) blockIdx.x, (int) gridDim.x, smem, w, x, y, n_in, n_out, ncols);
#elif defined(__CUDA_ARCH__)
    __trap();
#endif
}

template<int S, int MT, int KS>
size_t smem_bytes() {
    return (size_t) S * 16 * MT * row_stride(KS * kSB) + (size_t) MT * KS * 4 * 16 * kMaxCols * 4;
}

struct DevInfo {
    int sms = 0;
    int cc = 0;
};
const DevInfo& dev_info() {
    static DevInfo info[64];
    static std::atomic<bool> done[64];
    int dev = 0;
    cudaGetDevice(&dev);
    if (dev < 0 || dev >= 64) dev = 0;
    if (!done[dev].load(std::memory_order_acquire)) {
        int sms = 0, maj = 0, mnr = 0;
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
        cudaDeviceGetAttribute(&maj, cudaDevAttrComputeCapabilityMajor, dev);
        cudaDeviceGetAttribute(&mnr, cudaDevAttrComputeCapabilityMinor, dev);
        info[dev].sms = sms;
        info[dev].cc = 10 * maj + mnr;
        done[dev].store(true, std::memory_order_release);
    }
    return info[dev];
}

// The dynamic shared-memory opt-in and the occupancy of a launch shape, once per device (a graph capture may run on
// several threads, one per card).
template<int S, int MT, int KS>
int occupancy() {
    static std::atomic<int> occ[64];
    int dev = 0;
    cudaGetDevice(&dev);
    if (dev < 0 || dev >= 64) dev = 0;
    int o = occ[dev].load(std::memory_order_acquire);
    if (o == 0) {
        const size_t smem = smem_bytes<S, MT, KS>();
        cudaError_t e = cudaSuccess;
        if (smem > 48 * 1024)
            e = cudaFuncSetAttribute(dense_tc_kernel<S, MT, KS>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem);
        if (e == cudaSuccess)
            e = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&o, dense_tc_kernel<S, MT, KS>, MT * KS * 128, smem);
        if (e != cudaSuccess) throw std::runtime_error(std::string("dense_tc: ") + cudaGetErrorString(e));
        if (o < 1) o = 1;
        occ[dev].store(o, std::memory_order_release);
    }
    return o;
}

// GRID 0: persistent, min(tiles, occupancy x SMs) blocks (Basalt's); 1: one block a tile; 2: persistent with the tiles
// spread evenly (the fewest blocks that keep the same tiles per block)
template<int S, int MT, int KS, int GRID = 0>
void launch_s(const void* w, const void* x, float* y, int n_in, int n_out, int ncols, cudaStream_t s) {
    const int occ = occupancy<S, MT, KS>();
    const int ntiles = (n_out + 16 * MT - 1) / (16 * MT);
    const int cap = occ * dev_info().sms;
    int grid = ntiles < cap ? ntiles : cap;
    if (GRID == 1) grid = ntiles;
    if (GRID == 2) { const int per = (ntiles + grid - 1) / grid; grid = (ntiles + per - 1) / per; }
    dense_tc_kernel<S, MT, KS><<<grid, MT * KS * 128, smem_bytes<S, MT, KS>(), s>>>(
        (const uint8_t*) w, (const uint8_t*) x, y, n_in, n_out, ncols);
}

struct Cfg {
    const char* name;
    void (*fn)(const void*, const void*, float*, int, int, int, cudaStream_t);
    int ks;
};
const Cfg kCfgs[] = {
    {"default", nullptr, 1},
    {"S3 MT1 KS1", launch_s<3, 1, 1>, 1}, {"S4 MT1 KS1", launch_s<4, 1, 1>, 1}, {"S6 MT1 KS1", launch_s<6, 1, 1>, 1},
    {"S3 MT2 KS1", launch_s<3, 2, 1>, 1}, {"S4 MT2 KS1", launch_s<4, 2, 1>, 1}, {"S3 MT1 KS2", launch_s<3, 1, 2>, 2},
    {"S4 MT1 KS2", launch_s<4, 1, 2>, 2},
    {"S3 MT1 KS1 g1", launch_s<3, 1, 1, 1>, 1}, {"S3 MT1 KS1 g2", launch_s<3, 1, 1, 2>, 1},
    {"S3 MT2 KS1 g1", launch_s<3, 2, 1, 1>, 1}, {"S3 MT1 KS2 g1", launch_s<3, 1, 2, 1>, 2},
    {"S2 MT1 KS1 g1", launch_s<2, 1, 1, 1>, 1}, {"S2 MT1 KS2 g1", launch_s<2, 1, 2, 1>, 2},
};
constexpr int kNCfg = (int) (sizeof(kCfgs) / sizeof(kCfgs[0]));

// The launch shape follows the shape alone (never the column count), so a column's sums are the same for every T.
// Picked on a 3090 (dense_tc_bench, T = 1..8): n_in % 1024 == 0 and at most 4 x SMs 16-row tiles (the out-projections):
// two warp groups a tile (Basalt's KS = 2) with a 2-slot ring, one block a tile; many tiles (QSA q): 32-row tiles; very
// many (the head): 32-row tiles, one block a tile; else Basalt's default (16-row tiles, persistent).
void launch_default(const void* w, const void* x, float* y, int n_in, int n_out, int ncols, cudaStream_t s) {
    const int t16 = (n_out + 15) / 16, sms = dev_info().sms;
    if (n_in % 1024 == 0 && t16 <= 4 * sms) launch_s<2, 1, 2, 1>(w, x, y, n_in, n_out, ncols, s);
    else if (t16 > 64 * sms) launch_s<3, 2, 1, 1>(w, x, y, n_in, n_out, ncols, s);
    else if (t16 > 8 * sms) launch_s<3, 2, 1, 0>(w, x, y, n_in, n_out, ncols, s);
    else launch_s<3, 1, 1, 0>(w, x, y, n_in, n_out, ncols, s);
}

void check_args(const void* w, const void* x, int n_in, int n_out, int ncols) {
    if (!dense_tc_q8_0_ok(n_in) || ncols < 1 || ncols > kMaxCols || n_out < 1 || ((uintptr_t) w & 3) != 0 ||
        ((uintptr_t) x & 3) != 0) {
        char b[160];
        std::snprintf(b, sizeof b, "dense_tc_q8_0: n_in %d n_out %d ncols %d not supported", n_in, n_out, ncols);
        throw std::runtime_error(b);
    }
}

}  // namespace

bool dense_tc_q8_0_ok(int n_in) {
#if defined(STRATA_USE_HIP)
    return false;
#endif
    return n_in > 0 && n_in % 512 == 0 && dev_info().cc >= 80;
}

void dense_tc_q8_0(const void* w, const void* x, float* y, int n_in, int n_out, int ncols, void* stream) {
    check_args(w, x, n_in, n_out, ncols);
    launch_default(w, x, y, n_in, n_out, ncols, (cudaStream_t) stream);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string("dense_tc_q8_0: ") + cudaGetErrorString(e));
}

bool dense_tc_q8_0_cfg(int cfg, const void* w, const void* x, float* y, int n_in, int n_out, int ncols, void* stream) {
    if (cfg < 0 || cfg >= kNCfg) return false;
    if (cfg == 0) {
        dense_tc_q8_0(w, x, y, n_in, n_out, ncols, stream);
        return true;
    }
    if (n_in % (512 * kCfgs[cfg].ks) != 0) return false;
    check_args(w, x, n_in, n_out, ncols);
    kCfgs[cfg].fn(w, x, y, n_in, n_out, ncols, (cudaStream_t) stream);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string("dense_tc_q8_0: ") + cudaGetErrorString(e));
    return true;
}

const char* dense_tc_q8_0_cfg_name(int cfg) { return cfg >= 0 && cfg < kNCfg ? kCfgs[cfg].name : nullptr; }

}  // namespace strata::kernels
