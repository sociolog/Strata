// src/kernels/cuda/fused_gr.cu - see include/strata/kernels/fused_gr.hpp.
#include "strata/core/emulate.hpp"
#include "strata/core/sm75_series.hpp"
#include "strata/kernels/q8_1_finite.hpp"   // #606: q8_1_ds
#include "strata/core/geometry.hpp"
#include "strata/kernels/fused_gr.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/verify_kernels.hpp"
#include "strata/kernels/gfx_arch.hpp"
#include "s26_tsum.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <atomic>
#include <cfloat>
#include <type_traits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace strata::kernels {
namespace {

constexpr int N = 2560;         // n_embd
constexpr int HC = 4;           // streams
constexpr int D = N * HC;       // 10240
constexpr int LR = 320;         // hc_lr
// the latency-hidden norm/up (gr_norm_fast_kernel / gr_up_fast_kernel, STRATA_GR_FAST): gfx906's, built for CUDA too
#if defined(STRATA_HIP_GFX906) || !defined(__HIPCC__)
#define STRATA_GR_FAST_BUILD 1
#else
#define STRATA_GR_FAST_BUILD 0
#endif
static_assert(N == strata::core::kCompiledGeometry.n_embd && HC == strata::core::kCompiledGeometry.hc && LR == strata::core::kCompiledGeometry.hc_lr, "fused_gr is compiled for core/geometry.hpp");
constexpr int THREADS = 256;
constexpr int WARPS = THREADS / 32;
constexpr int DOWN_BLOCKS = LR / WARPS;          // 40 blocks of 8 rows; one more for the inject rows
constexpr int UP_COLS = 32;                      // columns d per `up` block (x 4 streams = 128 rows)
constexpr int UP_BLOCKS = N / UP_COLS;           // 80

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float sigmoidf_(float x) { return 1.0f / (1.0f + __expf(-x)); }

// 8 bf16 packed in a uint4, unpacked to two float4 once when reused across tokens.
struct Bf16x8 { float4 a, b; };
__device__ __forceinline__ Bf16x8 unpack8(const uint4 w) {
    return {
        make_float4(__uint_as_float(w.x << 16), __uint_as_float(w.x & 0xffff0000u),
                    __uint_as_float(w.y << 16), __uint_as_float(w.y & 0xffff0000u)),
        make_float4(__uint_as_float(w.z << 16), __uint_as_float(w.z & 0xffff0000u),
                    __uint_as_float(w.w << 16), __uint_as_float(w.w & 0xffff0000u))
    };
}
__device__ __forceinline__ float dot8u(const Bf16x8& w, const float4 x0, const float4 x1) {
    float acc = 0.0f;
    acc = fmaf(w.a.x, x0.x, acc);
    acc = fmaf(w.a.y, x0.y, acc);
    acc = fmaf(w.a.z, x0.z, acc);
    acc = fmaf(w.a.w, x0.w, acc);
    acc = fmaf(w.b.x, x1.x, acc);
    acc = fmaf(w.b.y, x1.y, acc);
    acc = fmaf(w.b.z, x1.z, acc);
    acc = fmaf(w.b.w, x1.w, acc);
    return acc;
}
__device__ __forceinline__ float dot8u_ptr(const Bf16x8& w, const float* x) {
    const float4* x4 = reinterpret_cast<const float4*>(x);
    return dot8u(w, x4[0], x4[1]);
}

// 8 bf16 packed in a uint4 against 8 16-byte-aligned floats (same 8-FMA order).
__device__ __forceinline__ float dot8(const uint4 w, const float* x) {
    return dot8u_ptr(unpack8(w), x);
}

// STRATA_HC_Q8: a weight chunk of 8 values in one uint4 register either way - bf16 (4 words), or Q8 (8 int8 codes in
// .x/.y and the chunk's 32-value block scale in .z), so the kernels keep their register budget.
__device__ __forceinline__ float dot8q(const uint32_t a, const uint32_t b, const float* x) {
    float acc = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) acc = fmaf((float) (int8_t) (a >> (8 * i)), x[i], acc);
#pragma unroll
    for (int i = 0; i < 4; ++i) acc = fmaf((float) (int8_t) (b >> (8 * i)), x[4 + i], acc);
    return acc;
}
/// STRATA_HC_REQ8: the chunk at element `e` (a multiple of 8) of a matrix - bf16 `wb`, or (Q8 and `sq` non-null)
/// int8 codes with scales `sq`, packed into the same uint4 register as 8 int8 codes in .x/.y and the 32-value block's
/// scale in .z.  Q8 is a template parameter: the BF16 instantiations compile exactly as before.
template <bool Q8>
__device__ __forceinline__ uint4 ldw(const uint16_t* wb, const float* sq, size_t e) {
    if constexpr (Q8) {
        if (sq != nullptr) {
            const uint2 v = __ldg(reinterpret_cast<const uint2*>(reinterpret_cast<const int8_t*>(wb) + e));
            return make_uint4(v.x, v.y, __float_as_uint(__ldg(sq + e / 32)), 0u);
        }
    }
    return __ldg(reinterpret_cast<const uint4*>(wb + e));
}
template <bool Q8>
__device__ __forceinline__ float dotw(const bool q8, const uint4 w, const float* x) {
    if constexpr (Q8) {
        if (q8) return __uint_as_float(w.z) * dot8q(w.x, w.y, x);
    }
    return dot8(w, x);
}

template <bool Q8 = false>
__global__ void __launch_bounds__(THREADS) gr_down_kernel(FusedGrArgs a) {
    __shared__ __align__(16) float xn[D];
    __shared__ float part[WARPS][HC];
    __shared__ float s_rs[HC];
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    float gw[HC];
#pragma unroll
    for (int c = 0; c < HC; ++c) gw[c] = a.apply ? 2.0f * sigmoidf_(a.inj_prev[c] / (float) HC) : 0.0f;
    // 1. R' * w_norm into shared memory, and the per-stream sums of squares of R'.
    float ss[HC] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int i = t * 4; i < D; i += THREADS * 4) {
        const int c = i / N, d = i - c * N;
        float4 r = *reinterpret_cast<const float4*>(a.R + i);
        if (a.apply) {
            const float4 b = *reinterpret_cast<const float4*>(a.bo_prev + d);
            r.x = fmaf(b.x, gw[c], r.x); r.y = fmaf(b.y, gw[c], r.y);
            r.z = fmaf(b.z, gw[c], r.z); r.w = fmaf(b.w, gw[c], r.w);
        }
        const float4 g = *reinterpret_cast<const float4*>(a.w_norm + i);
        float sq = r.x * r.x + r.y * r.y + r.z * r.z + r.w * r.w;
#pragma unroll
        for (int cc = 0; cc < HC; ++cc) if (cc == c) ss[cc] += sq;
        *reinterpret_cast<float4*>(xn + i) = make_float4(r.x * g.x, r.y * g.y, r.z * g.z, r.w * g.w);
    }
#pragma unroll
    for (int c = 0; c < HC; ++c) {
        const float v = warp_sum(ss[c]);
        if (lane == 0) part[warp][c] = v;
    }
    __syncthreads();
    if (t < HC) {
        float s = 0.0f;
        for (int w = 0; w < WARPS; ++w) s += part[w][t];
        s_rs[t] = rsqrtf(s / (float) N + a.eps);
        if (blockIdx.x == 0) a.rs[t] = s_rs[t];
    }
    __syncthreads();
    for (int i = t; i < D; i += THREADS) xn[i] *= s_rs[i / N];
    __syncthreads();
    // 2. one warp per output row: 10240 bf16 = 1280 chunks of 8, 40 per lane.
    const bool inject_block = blockIdx.x == DOWN_BLOCKS;
    const int row = inject_block ? warp : blockIdx.x * WARPS + warp;
    if (inject_block && (a.w_inject == nullptr || warp >= HC)) return;
    const uint16_t* wmat = inject_block ? a.w_inject : a.w_down;
    const float* sq = (Q8 && !inject_block) ? a.s_down : nullptr;   // the inject rows stay bf16
    const bool q8 = Q8 && !inject_block;
    float acc = 0.0f;
#pragma unroll 4
    for (int j = lane; j < D / 8; j += 32) acc += dotw<Q8>(q8, ldw<Q8>(wmat, sq, (size_t) row * D + (size_t) j * 8), xn + j * 8);
    acc = warp_sum(acc);
    if (lane != 0) return;
    if (inject_block) {
        a.inject_out[row] = acc;
    } else {
        const float x = acc / (float) HC;
        a.lo[row] = x / (1.0f + __expf(-x));
    }
}

template <bool Q8 = false>
__global__ void __launch_bounds__(THREADS) gr_up_kernel(FusedGrArgs a) {
    __shared__ __align__(16) float lo[LR];
    __shared__ float g[HC][UP_COLS];
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int d0 = blockIdx.x * UP_COLS;
    for (int k = t; k < LR; k += THREADS) lo[k] = a.lo[k];
    __syncthreads();
    // 128 rows (4 streams x 32 columns), 16 per warp: 320 bf16 = 40 chunks of 8.
    for (int r = warp; r < HC * UP_COLS; r += WARPS) {
        const int c = r / UP_COLS, dd = r - c * UP_COLS, i = c * N + d0 + dd;
        const bool q8 = Q8;
        float acc = dotw<Q8>(q8, ldw<Q8>(a.w_up, a.s_up, (size_t) i * LR + (size_t) lane * 8), lo + lane * 8);
        if (lane < LR / 8 - 32)
            acc += dotw<Q8>(q8, ldw<Q8>(a.w_up, a.s_up, (size_t) i * LR + (size_t) (32 + lane) * 8), lo + (32 + lane) * 8);
        acc = warp_sum(acc);
        if (lane == 0) {
            float rv = a.R[i];
            if (a.apply) {
                rv = fmaf(a.bo_prev[d0 + dd], 2.0f * sigmoidf_(a.inj_prev[c] / (float) HC), rv);
                a.R_out[i] = rv;                       // this block owns column d0+dd of every stream
            }
            const float x = rv * a.w_norm[i] * a.rs[c];
            g[c][dd] = x * sigmoidf_(acc);
        }
    }
    __syncthreads();
    if (t < UP_COLS) {
        float s = 0.0f;
#pragma unroll
        for (int c = 0; c < HC; ++c) s += g[c][t];
        a.mixed[d0 + t] = s / (float) HC;
    }
}

// ================================ plan v0.3 P6: T tokens, one weight read ================================
struct GrMulti {
    FusedGrArgs a[kFusedGrMaxT];
    float* xn;
    int T;
};


// S26 STRATA_QFUSE=1: the q8_1 image of `mixed`, written by the up kernels below. A q8_1 block is 32 columns and an
// up block owns UPM_COLS = 16, so the second of the two blocks that own a 32-column group (a per-group counter,
// incremented after the block's writes are fenced, reset by that block for the next launch) reads the 32 values back
// and quantizes them with native_quantize_q8_1_kernel's quantizer: one warp per token, the same XOR-tree max and sum,
// d = amax / 127, roundf(x / d), ds = (d, sum) - the bytes the separate launch writes.
struct GrQ81 { half2 ds; int8_t qs[32]; };
static_assert(2 * 16 == 32, "two up blocks per q8_1 group");
__device__ __forceinline__ void gr_q8_tail(const GrMulti& m, int d0) {
    __shared__ int last;
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned* c = m.a[0].q8_cnt + d0 / 32;
        last = atomicAdd(c, 1u) == 1u;
        if (last) atomicExch(c, 0u);
    }
    __syncthreads();
    if (!last) return;
    __threadfence();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if (warp >= m.T) return;
    const int c0 = (d0 / 32) * 32;
    const float xi = *reinterpret_cast<const volatile float*>(m.a[warp].mixed + c0 + lane);
    float amax = fabsf(xi), sum = xi;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
    const float d = q8_1_finite(amax / 127.0f);   // #606: as native_quantize_q8_1_kernel - finite blocks bit for bit
    const int8_t q = q8_1_quant(xi, d, amax);
    GrQ81* y = reinterpret_cast<GrQ81*>(m.a[warp].q8_mixed) + c0 / 32;
    y->qs[lane] = q;
    if (lane == 0) y->ds = q8_1_ds(d, sum);   // #606: clamped scale/sum - an unclamped pair NaN-poisons the dot path
}
// Step 1 of `gr_down_kernel`, one block per token, same threads and reduction order: rs[t] and xn[t] to global.
__global__ void __launch_bounds__(THREADS) gr_norm_multi_kernel(GrMulti m) {
    __shared__ float part[WARPS][HC];
    __shared__ float s_rs[HC];
    const FusedGrArgs& a = m.a[blockIdx.x];
    float* xn = m.xn + (size_t) blockIdx.x * D;
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    float gw[HC];
#pragma unroll
    for (int c = 0; c < HC; ++c) gw[c] = a.apply ? 2.0f * sigmoidf_(a.inj_prev[c] / (float) HC) : 0.0f;
    float ss[HC] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int i = t * 4; i < D; i += THREADS * 4) {
        const int c = i / N, d = i - c * N;
        float4 r = *reinterpret_cast<const float4*>(a.R + i);
        if (a.apply) {
            const float4 b = *reinterpret_cast<const float4*>(a.bo_prev + d);
            r.x = fmaf(b.x, gw[c], r.x); r.y = fmaf(b.y, gw[c], r.y);
            r.z = fmaf(b.z, gw[c], r.z); r.w = fmaf(b.w, gw[c], r.w);
        }
        const float4 g = *reinterpret_cast<const float4*>(a.w_norm + i);
        float sq = r.x * r.x + r.y * r.y + r.z * r.z + r.w * r.w;
#pragma unroll
        for (int cc = 0; cc < HC; ++cc) if (cc == c) ss[cc] += sq;
        *reinterpret_cast<float4*>(xn + i) = make_float4(r.x * g.x, r.y * g.y, r.z * g.z, r.w * g.w);
    }
#pragma unroll
    for (int c = 0; c < HC; ++c) {
        const float v = warp_sum(ss[c]);
        if (lane == 0) part[warp][c] = v;
    }
    __syncthreads();
    if (t < HC) {
        float s = 0.0f;
        for (int w = 0; w < WARPS; ++w) s += part[w][t];
        s_rs[t] = rsqrtf(s / (float) N + a.eps);
        a.rs[t] = s_rs[t];
    }
    __syncthreads();
    for (int i = t; i < D; i += THREADS) xn[i] *= s_rs[i / N];
}

// Step 2 of `gr_down_kernel` for T tokens.  One warp per row (so each lane accumulates the same chunks in the
// same order as the single-token kernel); per tile the lane's weight chunks are loaded BEFORE the activation
// tile is staged, so the DRAM and L2 traffic are in flight together.
//
// TILEV = xn floats per token staged at a time.  The tile only changes the staging granularity: the lane's chunk
// order (lane + 32*q within the tile, tiles ascending) is strictly increasing for either value, so the results are
// bitwise identical to `gr_down_kernel` for both.  2560 stages 320 chunks of 8 per tile (10 per lane); cards whose
// opt-in below 8 * 2560 * 4 B slices the tokens - sm_75 (64 KiB) carries 6 tokens of it - run TILEV 1280 instead,
// which fits all eight tokens in one 40 KiB launch and stages 160 chunks of 8 (5 per lane, half the registers held
// for the weight prefetch); smaller tiles raise how many blocks share an SM (the 41-block grid), e.g. three blocks
// of four tokens instead of one on sm_75.
template <int TILEV, int MAX_T = kFusedGrMaxT, bool Q8 = false>
__global__ void __launch_bounds__(THREADS) gr_down_multi_kernel(GrMulti m) {
    constexpr int TQ = TILEV / 8 / 32;      // uint4 weight chunks per lane per tile
    extern __shared__ __align__(16) float tile[];   // [T][TILEV]
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int T = m.T;
    const bool inject_block = blockIdx.x == DOWN_BLOCKS;
    const int row = inject_block ? warp : blockIdx.x * WARPS + warp;
    const bool active = !(inject_block && (m.a[0].w_inject == nullptr || warp >= HC));
    const uint16_t* wmat = inject_block ? m.a[0].w_inject : m.a[0].w_down;
    const float* sq = (Q8 && !inject_block) ? m.a[0].s_down : nullptr;   // the inject rows stay bf16
    const bool q8 = Q8 && !inject_block;
    const size_t row0 = (size_t) (active ? row : 0) * D;
    float acc[MAX_T];
#pragma unroll
    for (int k = 0; k < MAX_T; ++k) acc[k] = 0.0f;
    for (int base = 0; base < D; base += TILEV) {
        uint4 wv[TQ];
        if (active) {
#pragma unroll
            for (int q = 0; q < TQ; ++q) wv[q] = ldw<Q8>(wmat, sq, row0 + (size_t) base + (size_t) (lane + 32 * q) * 8);
        }
        __syncthreads();                                   // the previous tile is consumed
        const float4* src4 = reinterpret_cast<const float4*>(m.xn);
        float4* tile4 = reinterpret_cast<float4*>(tile);
        for (int i = t; i < T * (TILEV / 4); i += THREADS) {
            const int k = i / (TILEV / 4), off = i - k * (TILEV / 4);
            tile4[i] = src4[((size_t) k * D + base) / 4 + off];
        }
        __syncthreads();
        if (!active) continue;
#pragma unroll
        for (int q = 0; q < TQ; ++q) {
            const int j = lane + 32 * q;
#pragma unroll
            for (int k = 0; k < MAX_T; ++k)
                if (k < T) acc[k] += dotw<Q8>(q8, wv[q], tile + k * TILEV + j * 8);
        }
    }
    if (!active) return;
    float s[MAX_T];
#pragma unroll
    for (int k = 0; k < MAX_T; ++k) s[k] = k < T ? warp_sum(acc[k]) : 0.0f;
    // lane k writes token k (every lane holds every sum after the xor reduction)
#pragma unroll
    for (int k = 0; k < MAX_T; ++k) {
        if (k >= T || lane != k) continue;
        if (inject_block) {
            m.a[k].inject_out[row] = s[k];
        } else {
            const float x = s[k] / (float) HC;
            m.a[k].lo[row] = x / (1.0f + __expf(-x));
        }
    }
}

constexpr int UPM_COLS = 16;                      // columns per block (x 4 streams = 64 rows, 8 per warp)
constexpr int UPM_BLOCKS = N / UPM_COLS;          // 160

// `gr_up_kernel` for T tokens: each row of w_up read once; the T dots reduced by xor so every lane holds every
// sum, and lane k runs token k's epilogue - the T epilogues in parallel instead of one after another.
template <int MAX_T = kFusedGrMaxT, bool EXACT_T = false, bool Q8 = false>
__global__ void __launch_bounds__(THREADS) gr_up_multi_kernel(GrMulti m) {
    __shared__ __align__(16) float lo[MAX_T][LR];
    __shared__ float g[MAX_T][HC][UPM_COLS];
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int T = EXACT_T ? MAX_T : m.T;
    const int d0 = blockIdx.x * UPM_COLS;
    for (int i = t; i < T * LR; i += THREADS) lo[i / LR][i % LR] = m.a[i / LR].lo[i % LR];
    __syncthreads();
    for (int r = warp; r < HC * UPM_COLS; r += WARPS) {
        const int c = r / UPM_COLS, dd = r - c * UPM_COLS, i = c * N + d0 + dd;
        uint4 qa = make_uint4(0, 0, 0, 0), qb = make_uint4(0, 0, 0, 0);
        Bf16x8 wa{}, wb{};
        if constexpr (Q8) {
            qa = ldw<true>(m.a[0].w_up, m.a[0].s_up, (size_t) i * LR + (size_t) lane * 8);
            if (lane < LR / 8 - 32) qb = ldw<true>(m.a[0].w_up, m.a[0].s_up, (size_t) i * LR + (size_t) (32 + lane) * 8);
        } else {
            const uint4* w4 = reinterpret_cast<const uint4*>(m.a[0].w_up + (size_t) i * LR);
            wa = unpack8(__ldg(w4 + lane));
            wb = unpack8(lane < LR / 8 - 32 ? __ldg(w4 + 32 + lane) : make_uint4(0, 0, 0, 0));
        }
        // the epilogue inputs of this lane's token, fetched while the dots run
        float rv = 0.0f, wn = 0.0f, rsc = 0.0f, bo = 0.0f, ip = 0.0f;
        bool apply = false;
        if (lane < T) {
            const FusedGrArgs& a = m.a[lane];
            rv = a.R[i];
            wn = a.w_norm[i];
            rsc = a.rs[c];
            apply = a.apply;
            if (apply) { bo = a.bo_prev[d0 + dd]; ip = a.inj_prev[c]; }
        }
        float mine = 0.0f;
#pragma unroll
        for (int k = 0; k < MAX_T; ++k) {
            if (!EXACT_T && k >= T) break;
            float acc;
            if constexpr (Q8) {
                acc = dotw<true>(true, qa, lo[k] + lane * 8);
                if (lane < LR / 8 - 32) acc += dotw<true>(true, qb, lo[k] + (32 + lane) * 8);
            } else {
                acc = dot8u_ptr(wa, lo[k] + lane * 8);
                if (lane < LR / 8 - 32) acc += dot8u_ptr(wb, lo[k] + (32 + lane) * 8);
            }
            acc = warp_sum(acc);
            if (lane == k) mine = acc;
        }
        if (lane < T) {
            if (apply) {
                rv = fmaf(bo, 2.0f * sigmoidf_(ip / (float) HC), rv);
                m.a[lane].R_out[i] = rv;
            }
            const float x = rv * wn * rsc;
            g[lane][c][dd] = x * sigmoidf_(mine);
        }
    }
    __syncthreads();
    for (int i = t; i < T * UPM_COLS; i += THREADS) {
        const int k = i / UPM_COLS, col = i - k * UPM_COLS;
        float s = 0.0f;
#pragma unroll
        for (int c = 0; c < HC; ++c) s += g[k][c][col];
        m.a[k].mixed[d0 + col] = s / (float) HC;
    }
    if (m.a[0].q8_mixed != nullptr) gr_q8_tail(m, d0);   // S26 STRATA_QFUSE
}

// Turing (sm_75): gr_up_multi_kernel's rows, sums and epilogue with the latency taken out of the warp's chain.  A warp's
// 8 rows of w_up are all loaded before any dot (was: one row's load after the previous row's epilogue: ~9k of the
// warp's ~22k cycles were waits on memory, ncu on an RTX 2080 Ti); a token's 8 activations per lane are read from
// shared memory once for the 8 rows; and the 8 rows' xor reductions run interleaved (8 independent shuffles per stage
// instead of one chain of 5 per row).  A row's last 8 chunks (lanes 0-7 of gr_up_multi_kernel) are held 4 rows to a
// uint4 - lane 8 i + j holds chunk 32 + j of row i - and their dot is shuffled to lane j before it is added, so lane j
// still sums its two dots in gr_up_multi_kernel's order.  Every row's lane sums, xor tree and epilogue are
// gr_up_multi_kernel's: the same bits.  STRATA_GR_UP75=0 keeps gr_up_multi_kernel.
template <int MAX_T = kFusedGrMaxT, bool EXACT_T = false>
__global__ void __launch_bounds__(THREADS, 4) gr_up75_kernel(GrMulti m) {
    constexpr int RPW = HC * UPM_COLS / WARPS;    // 8 rows per warp: row r = warp + WARPS i -> stream i / 2
    static_assert(RPW == 8 && UPM_COLS == 2 * WARPS && HC == 4 && LR / 8 - 32 == 8, "geometry");
    __shared__ __align__(16) float lo[MAX_T][LR];
    __shared__ float g[MAX_T][HC][UPM_COLS];
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int T = EXACT_T ? MAX_T : m.T;
    const int d0 = blockIdx.x * UPM_COLS;
    const int tj = lane & 7, ti = lane >> 3;       // tail: chunk 32 + tj of row 4 b + ti
    float mine[RPW];
#pragma unroll
    for (int i = 0; i < RPW; ++i) mine[i] = 0.0f;
#pragma unroll
    for (int bt = 0; bt < 2; ++bt) {               // two batches of 4 rows: 4 loads in flight, registers for 4 blocks
        uint4 wa[4], wt;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int r = 4 * bt + i, c = r >> 1, dd = warp + WARPS * (r & 1), row = c * N + d0 + dd;
            wa[i] = __ldg(reinterpret_cast<const uint4*>(m.a[0].w_up + (size_t) row * LR) + lane);
        }
        {
            const int r = 4 * bt + ti, c = r >> 1, dd = warp + WARPS * (r & 1), row = c * N + d0 + dd;
            wt = __ldg(reinterpret_cast<const uint4*>(m.a[0].w_up + (size_t) row * LR) + 32 + tj);
        }
        if (bt == 0) {
            for (int i = t; i < T * LR; i += THREADS) lo[i / LR][i % LR] = m.a[i / LR].lo[i % LR];
            __syncthreads();
        }
#pragma unroll 1
        for (int k = 0; k < T; ++k) {
            const float4* la = reinterpret_cast<const float4*>(lo[k] + lane * 8);
            const float4* lb = reinterpret_cast<const float4*>(lo[k] + (32 + tj) * 8);
            const float tb = dot8u(unpack8(wt), lb[0], lb[1]);
            const float4 x0 = la[0], x1 = la[1];
            float p[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float b = __shfl_sync(0xffffffffu, tb, i * 8 + tj);
                float acc = dot8u(unpack8(wa[i]), x0, x1);
                if (lane < 8) acc += b;
                p[i] = acc;
            }
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) {
#pragma unroll
                for (int i = 0; i < 4; ++i) p[i] += __shfl_xor_sync(0xffffffffu, p[i], o);
            }
            if (lane == k) {
#pragma unroll
                for (int i = 0; i < 4; ++i) mine[4 * bt + i] = p[i];
            }
        }
    }
    if (lane < T) {
        const FusedGrArgs& a = m.a[lane];
        const bool apply = a.apply;
        float rv[RPW], wn[RPW], bo[RPW];
#pragma unroll
        for (int i = 0; i < RPW; ++i) {
            const int c = i >> 1, dd = warp + WARPS * (i & 1), row = c * N + d0 + dd;
            rv[i] = a.R[row];
            wn[i] = a.w_norm[row];
            bo[i] = apply ? a.bo_prev[d0 + dd] : 0.0f;
        }
#pragma unroll
        for (int i = 0; i < RPW; ++i) {
            const int c = i >> 1, dd = warp + WARPS * (i & 1), row = c * N + d0 + dd;
            float x0 = rv[i];
            if (apply) {
                x0 = fmaf(bo[i], 2.0f * sigmoidf_(a.inj_prev[c] / (float) HC), x0);
                a.R_out[row] = x0;
            }
            const float x = x0 * wn[i] * a.rs[c];
            g[lane][c][dd] = x * sigmoidf_(mine[i]);
        }
    }
    __syncthreads();
    for (int i = t; i < T * UPM_COLS; i += THREADS) {
        const int k = i / UPM_COLS, col = i - k * UPM_COLS;
        float s = 0.0f;
#pragma unroll
        for (int c = 0; c < HC; ++c) s += g[k][c][col];
        m.a[k].mixed[d0 + col] = s / (float) HC;
    }
    if (m.a[0].q8_mixed != nullptr) gr_q8_tail(m, d0);   // S26 STRATA_QFUSE, as gr_up_multi_kernel
}


// 143-amdk (STRATA_QFUSE_HC): gr_up_multi_kernel with 32 columns a block (512 threads, 80 blocks, the same 8 rows a warp), so a block owns a whole
// q8_1 group of `mixed` and writes its image itself.  STRATA_QFUSE's version (16 columns a block) needs a counter, two fences and a last block
// to finish the group; this one needs none.  Every row's math is gr_up_multi_kernel's.  Needs q8_mixed set for every token.
constexpr int Q32_COLS = 32, Q32_THREADS = 512, Q32_WARPS = Q32_THREADS / 32, Q32_BLOCKS = N / Q32_COLS;
template <int MAX_T = kFusedGrMaxT, bool EXACT_T = false>
__global__ void __launch_bounds__(Q32_THREADS) gr_up_q32_kernel(GrMulti m) {
    __shared__ __align__(16) float lo[MAX_T][LR];
    __shared__ float g[MAX_T][HC][Q32_COLS];
    __shared__ float mixed_s[MAX_T][Q32_COLS];
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int T = EXACT_T ? MAX_T : m.T;
    const int d0 = blockIdx.x * Q32_COLS;
    for (int i = t; i < T * LR; i += Q32_THREADS) lo[i / LR][i % LR] = m.a[i / LR].lo[i % LR];
    __syncthreads();
    for (int r = warp; r < HC * Q32_COLS; r += Q32_WARPS) {
        const int c = r / Q32_COLS, dd = r - c * Q32_COLS, i = c * N + d0 + dd;
        const uint4* w4 = reinterpret_cast<const uint4*>(m.a[0].w_up + (size_t) i * LR);
        const Bf16x8 wa = unpack8(__ldg(w4 + lane));
        const Bf16x8 wb = unpack8(lane < LR / 8 - 32 ? __ldg(w4 + 32 + lane) : make_uint4(0, 0, 0, 0));
        // the epilogue inputs of this lane's token, fetched while the dots run
        float rv = 0.0f, wn = 0.0f, rsc = 0.0f, bo = 0.0f, ip = 0.0f;
        bool apply = false;
        if (lane < T) {
            const FusedGrArgs& a = m.a[lane];
            rv = a.R[i];
            wn = a.w_norm[i];
            rsc = a.rs[c];
            apply = a.apply;
            if (apply) { bo = a.bo_prev[d0 + dd]; ip = a.inj_prev[c]; }
        }
        float mine = 0.0f;
#pragma unroll
        for (int k = 0; k < MAX_T; ++k) {
            if (!EXACT_T && k >= T) break;
            float acc = dot8u_ptr(wa, lo[k] + lane * 8);
            if (lane < LR / 8 - 32) acc += dot8u_ptr(wb, lo[k] + (32 + lane) * 8);
            acc = warp_sum(acc);
            if (lane == k) mine = acc;
        }
        if (lane < T) {
            if (apply) {
                rv = fmaf(bo, 2.0f * sigmoidf_(ip / (float) HC), rv);
                m.a[lane].R_out[i] = rv;
            }
            const float x = rv * wn * rsc;
            g[lane][c][dd] = x * sigmoidf_(mine);
        }
    }
    __syncthreads();
    for (int i = t; i < T * Q32_COLS; i += Q32_THREADS) {
        const int k = i / Q32_COLS, col = i - k * Q32_COLS;
        float s = 0.0f;
#pragma unroll
        for (int c = 0; c < HC; ++c) s += g[k][c][col];
        const float v = s / (float) HC;
        m.a[k].mixed[d0 + col] = v;
        mixed_s[k][col] = v;
    }
    __syncthreads();
    // the block owns the whole q8_1 group (32 columns): quantize it here, no counter, no fence.  One warp a token, the
    // quantizer of gr_q8_tail (native_quantize_q8_1's: XOR-tree max and sum, d = amax / 127, roundf(x / d)).
    if (warp < T) {
        const float xi = mixed_s[warp][lane];
        float amax = fabsf(xi), sum = xi;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
        const float d = q8_1_finite(amax / 127.0f);
        const int8_t q = q8_1_quant(xi, d, amax);
        GrQ81* y = reinterpret_cast<GrQ81*>(m.a[warp].q8_mixed) + d0 / 32;
        y->qs[lane] = q;
        if (lane == 0) y->ds = q8_1_ds(d, sum);
    }
}



// ================================ hc read v3 (opt-in: STRATA_GR_V3=1) - two kernels, stream-split ================
// The norm kernel runs on only T blocks (~16 us of pure latency per call) and `down` on 41 blocks (~280 GB/s).
// v3: `down` is split over (row group, stream[, column half]) = 164 or 328 blocks; each block stages its slice of
// R' * w_norm for the T tokens, reduces that slice's sum of squares itself, and writes UNSCALED partial dots.  The
// rms scale is per stream, so  w_down . xn = sum_c rs[c] * (w_down[:, c] . (R'[c] * w_norm[c]))  - `up` applies it
// in its prologue (lo, inject, rs).  Same maths, ANOTHER SUMMATION ORDER: not bitwise the default kernels, hence
// opt-in.  Dynamic shared memory is T * (N / S) floats: S (1 or 2 column halves) is the smallest that fits the
// card's opt-in limit at kFusedGrMaxT tokens (Ampere 99 KB: S = 1; Turing / HIP 64 KB: S = 2); a card where
// neither fits keeps the default kernels.
constexpr int PR = LR + HC;                        // partial rows per (token, stream): 320 down + 4 inject
constexpr int TQ3 = N / 8 / 32;                    // uint4 weight chunks per lane in one stream's slice (10)

template <int S, int MAX_T = kFusedGrMaxT, bool EXACT_T = false>
__global__ void __launch_bounds__(THREADS) gr_down_v3_kernel(GrMulti m, float* __restrict__ part, float* __restrict__ ssg) {
    extern __shared__ __align__(16) float xs[];    // [T][N / S]
    constexpr int R2 = 1;                          // down rows per warp
    __shared__ float red[WARPS][MAX_T];
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int T = EXACT_T ? MAX_T : m.T;
    // S = 2: each stream's 2560 columns in two halves (blockIdx.y = stream * S + half): twice the blocks
    constexpr int SL = N / S, TQS = SL / 8 / 32;
    const int rg = blockIdx.x, c = blockIdx.y / S, h = blockIdx.y - (blockIdx.y / S) * S;
    constexpr int NDB = LR / (WARPS * R2);          // down row blocks per stream; block NDB = the inject rows
    const bool inject_block = rg == NDB;
    // warp w owns rows row0 + w * R2 + r (r < R2); the inject block: warps 0-3, one row each
    const int row0 = inject_block ? warp : (rg * WARPS + warp) * R2;
    const int nrows = inject_block ? ((m.a[0].w_inject != nullptr && warp < HC) ? 1 : 0) : R2;
    const bool active = nrows > 0;
    const uint16_t* wbase = inject_block ? m.a[0].w_inject : m.a[0].w_down;
    float ssp[MAX_T];
#pragma unroll
    for (int k = 0; k < MAX_T; ++k) {
        ssp[k] = 0.0f;
        if (!EXACT_T && k >= T) continue;
        const FusedGrArgs& a = m.a[k];
        const float gw = a.apply ? 2.0f * sigmoidf_(a.inj_prev[c] / (float) HC) : 0.0f;
        const float4* R4 = reinterpret_cast<const float4*>(a.R + (size_t) c * N + (size_t) h * SL);
        const float4* G4 = reinterpret_cast<const float4*>(a.w_norm + (size_t) c * N + (size_t) h * SL);
        const float4* B4 = reinterpret_cast<const float4*>(a.bo_prev + (size_t) h * SL);
        float4* X4 = reinterpret_cast<float4*>(xs + (size_t) k * SL);
        for (int i = t; i < SL / 4; i += THREADS) {
            float4 r = R4[i];
            if (a.apply) {
                const float4 b = B4[i];
                r.x = fmaf(b.x, gw, r.x); r.y = fmaf(b.y, gw, r.y);
                r.z = fmaf(b.z, gw, r.z); r.w = fmaf(b.w, gw, r.w);
            }
            const float4 g = G4[i];
            ssp[k] += r.x * r.x + r.y * r.y + r.z * r.z + r.w * r.w;
            X4[i] = make_float4(r.x * g.x, r.y * g.y, r.z * g.z, r.w * g.w);
        }
    }
#pragma unroll
    for (int k = 0; k < MAX_T; ++k) {
        if (!EXACT_T && k >= T) break;
        const float v = warp_sum(ssp[k]);
        if (lane == 0) red[warp][k] = v;
    }
    __syncthreads();
    if (rg == 0 && t < T) {
        float sum = 0.0f;
        for (int w = 0; w < WARPS; ++w) sum += red[w][t];
        ssg[(t * HC + c) * S + h] = sum;
    }
    if (!active) return;
#pragma unroll
    for (int r = 0; r < R2; ++r) {
        if (r >= nrows) break;
        const uint4* w4 = reinterpret_cast<const uint4*>(wbase + (size_t) (row0 + r) * D + (size_t) c * N + (size_t) h * SL);
        float acc[MAX_T];
#pragma unroll
        for (int k = 0; k < MAX_T; ++k) acc[k] = 0.0f;
#pragma unroll
        for (int q = 0; q < TQS; ++q) {
            const int j = lane + 32 * q;
            const Bf16x8 wvq = unpack8(__ldg(w4 + j));
#pragma unroll
            for (int k = 0; k < MAX_T; ++k)
                if (EXACT_T || k < T) acc[k] += dot8u_ptr(wvq, xs + (size_t) k * SL + j * 8);
        }
        const int prow = inject_block ? LR + warp : row0 + r;
#pragma unroll
        for (int k = 0; k < MAX_T; ++k) {
            if (!EXACT_T && k >= T) break;
            const float v = warp_sum(acc[k]);
            if (lane == 0) part[(((size_t) k * HC + c) * S + h) * PR + prow] = v;
        }
    }
}

template <int S, int MAX_T = kFusedGrMaxT, bool EXACT_T = false>
__global__ void __launch_bounds__(THREADS) gr_up_v3_kernel(GrMulti m, const float* __restrict__ part,
                                                           const float* __restrict__ ssg) {
    __shared__ __align__(16) float lo[MAX_T][LR];
    __shared__ float rsS[MAX_T][HC];
    __shared__ float g[MAX_T][HC][UPM_COLS];
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int T = EXACT_T ? MAX_T : m.T;
    const int d0 = blockIdx.x * UPM_COLS;
    constexpr int RPW = HC * UPM_COLS / WARPS;     // 8 rows per warp
    if (t < T * HC) {
        const int k = t / HC, c = t - k * HC;
        float ss = 0.0f;
#pragma unroll
        for (int h = 0; h < S; ++h) ss += ssg[t * S + h];
        const float r = rsqrtf(ss / (float) N + m.a[k].eps);
        rsS[k][c] = r;
        if (blockIdx.x == 0) m.a[k].rs[c] = r;
    }
    __syncthreads();
    for (int i = t; i < T * LR; i += THREADS) {
        const int k = i / LR, r = i - k * LR;
        float sum = 0.0f;
#pragma unroll
        for (int c = 0; c < HC; ++c) {
            float p = 0.0f;
#pragma unroll
            for (int h = 0; h < S; ++h) p += part[(((size_t) k * HC + c) * S + h) * PR + r];
            sum = fmaf(rsS[k][c], p, sum);
        }
        const float x = sum / (float) HC;
        lo[k][r] = x / (1.0f + __expf(-x));
    }
    if (blockIdx.x == 0 && t < T * HC) {
        const int k = t / HC, cc = t - k * HC;
        if (m.a[k].w_inject != nullptr) {
            float sum = 0.0f;
#pragma unroll
            for (int c = 0; c < HC; ++c) {
                float p = 0.0f;
#pragma unroll
                for (int h = 0; h < S; ++h) p += part[(((size_t) k * HC + c) * S + h) * PR + LR + cc];
                sum = fmaf(rsS[k][c], p, sum);
            }
            m.a[k].inject_out[cc] = sum;
        }
    }
    __syncthreads();
#pragma unroll
    for (int q = 0; q < RPW; ++q) {
        const int r = warp + q * WARPS;
        const int c = r / UPM_COLS, dd = r - c * UPM_COLS, i = c * N + d0 + dd;
        const uint4* w4 = reinterpret_cast<const uint4*>(m.a[0].w_up + (size_t) i * LR);
        const Bf16x8 wa = unpack8(__ldg(w4 + lane));
        const Bf16x8 wb = unpack8(lane < LR / 8 - 32 ? __ldg(w4 + 32 + lane) : make_uint4(0, 0, 0, 0));
        float rv = 0.0f, wn = 0.0f, bo = 0.0f, ip = 0.0f;
        bool apply = false;
        if (lane < T) {
            const FusedGrArgs& a = m.a[lane];
            rv = a.R[i];
            wn = a.w_norm[i];
            apply = a.apply;
            if (apply) { bo = a.bo_prev[d0 + dd]; ip = a.inj_prev[c]; }
        }
        float mine = 0.0f;
#pragma unroll
        for (int k = 0; k < MAX_T; ++k) {
            if (!EXACT_T && k >= T) break;
            float acc = dot8u_ptr(wa, lo[k] + lane * 8);
            if (lane < LR / 8 - 32) acc += dot8u_ptr(wb, lo[k] + (32 + lane) * 8);
            acc = warp_sum(acc);
            if (lane == k) mine = acc;
        }
        if (lane < T) {
            if (apply) {
                rv = fmaf(bo, 2.0f * sigmoidf_(ip / (float) HC), rv);
                m.a[lane].R_out[i] = rv;
            }
            const float x = rv * wn * rsS[lane][c];
            g[lane][c][dd] = x * sigmoidf_(mine);
        }
    }
    __syncthreads();
    for (int i = t; i < T * UPM_COLS; i += THREADS) {
        const int k = i / UPM_COLS, col = i - k * UPM_COLS;
        float sum = 0.0f;
#pragma unroll
        for (int c = 0; c < HC; ++c) sum += g[k][c][col];
        m.a[k].mixed[d0 + col] = sum / (float) HC;
    }
}
// ================================ #315: split / staged - the same arithmetic, split differently ==================
// The multi read's variants beside the plain read (0.1.31's default; not main's opt-in STRATA_GR_V3 read, which sums
// in another order).  Every variant computes each output with exactly the plain read's operations in its order (so
// bitwise the plain read and the single-token read); only who does what, and when, changes.  `fused_gr_check`
// compares them on the card before a verify window uses them (STRATA_HC_SPLIT below).
//  - the norm (split and staged): one block per token AND stream instead of one per token.  Each thread visits
//    exactly the elements, in the order, it visited for that stream in the plain read, so every sum of squares and rs
//    is the same; it stores R' * w_norm and scales it in place, with the plain read's two roundings.
//    Before this scratch reuse, the split norm measured 0.23 ms per 4-token round (96 launches) on an RTX 4080
//    SUPER, compared with 0.58 ms for the plain read's norm.
//  - the down projection, split: the plain read's kernel.
//  - the down projection, staged: the plain read's 8 rows per block and lane order, but the activations arrive by
//    cp.async in half-stream tiles, two in flight, so no thread waits on a chain of loads, and they sit in shared
//    memory as two planes (floats 0-3 and 4-7 of every 8), so a warp's reads hit no bank twice.  The next tile's
//    weights are loaded while the current one is used.  A tile is 160 = 5 x 32 chunks of 8, so a lane still
//    accumulates its chunks lane + 32 q in ascending order, as in the plain read with either tile.  Before sm_80
//    (and on HIP) the staging is a plain copy: the same bits, only not asynchronous.
constexpr int kHcPlain = 1, kHcSplit = 2, kHcStaged = 3;
constexpr int H_TILE = 1280;                          // staged tile: half a stream = 160 chunks of 8, 5 per lane
constexpr int HQ = H_TILE / 8 / 32;
constexpr int N_HTILES = D / H_TILE;                  // 8
static_assert(N % H_TILE == 0, "a staged tile never straddles two streams");
static_assert(H_TILE % 256 == 0, "a staged tile holds whole rounds of 32 chunks: the plain read's lane order");

__global__ void __launch_bounds__(THREADS) gr_norm_split_kernel(GrMulti m) {
    __shared__ float part[WARPS];
    __shared__ float s_rs;
    const FusedGrArgs& a = m.a[blockIdx.x];
    const int c = blockIdx.y;
    float* xn = m.xn + (size_t) blockIdx.x * D;
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const float gw = a.apply ? 2.0f * sigmoidf_(a.inj_prev[c] / (float) HC) : 0.0f;
    float ss = 0.0f;
    for (int i = t * 4; i < D; i += THREADS * 4) {
        if (i / N != c) continue;                       // another stream's element: another block's
        const int d = i - c * N;
        float4 r = *reinterpret_cast<const float4*>(a.R + i);
        if (a.apply) {
            const float4 b = *reinterpret_cast<const float4*>(a.bo_prev + d);
            r.x = fmaf(b.x, gw, r.x); r.y = fmaf(b.y, gw, r.y);
            r.z = fmaf(b.z, gw, r.z); r.w = fmaf(b.w, gw, r.w);
        }
        const float sq = r.x * r.x + r.y * r.y + r.z * r.z + r.w * r.w;
        ss += sq;
        // Retain the unscaled product in the existing scratch instead of reloading R
        // and recomputing the pending write after the reduction.
        const float4 g = *reinterpret_cast<const float4*>(a.w_norm + i);
        *reinterpret_cast<float4*>(xn + i) = make_float4(r.x * g.x, r.y * g.y, r.z * g.z, r.w * g.w);
    }
    const float v = warp_sum(ss);
    if (lane == 0) part[warp] = v;
    __syncthreads();
    if (t == 0) {
        float s = 0.0f;
        for (int w = 0; w < WARPS; ++w) s += part[w];
        s_rs = rsqrtf(s / (float) N + a.eps);
        a.rs[c] = s_rs;
    }
    __syncthreads();
    const float rs = s_rs;
    for (int d = t; d < N; d += THREADS) xn[c * N + d] *= rs;
}

// 143-amdk: gr_norm_split_kernel with the loads of a thread's (at most three) float4 issued together and the scaled
// product written once from registers.  The same elements go through the same operations in the same order (the
// thread's own sum of squares ascends in k; the warp sum, part[] sum and rsqrt are the kernel's), so rs and xn are
// bitwise the same; what goes is the loop's one-iteration-at-a-time loads and the second pass that read xn back from
// global memory (a write, a barrier, a read: two memory latencies).
__global__ void __launch_bounds__(THREADS) gr_norm_split_fast_kernel(GrMulti m) {
    __shared__ float part[WARPS];
    __shared__ float s_rs;
    const FusedGrArgs& a = m.a[blockIdx.x];
    const int c = blockIdx.y;
    float* xn = m.xn + (size_t) blockIdx.x * D;
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const float gw = a.apply ? 2.0f * sigmoidf_(a.inj_prev[c] / (float) HC) : 0.0f;
    // the thread's float4 of stream c: i = 4 t + 1024 k in [c N, (c + 1) N), ascending k; N / 1024 = 2.5 -> at most 3
    constexpr int KS = 3;
    int k0 = (c * N - 4 * t + THREADS * 4 - 1) / (THREADS * 4);
    if (k0 < 0) k0 = 0;
    float4 r[KS], g[KS];
    bool ok[KS];
#pragma unroll
    for (int j = 0; j < KS; ++j) {
        const int i = t * 4 + (k0 + j) * THREADS * 4;
        ok[j] = i < (c + 1) * N;
        if (ok[j]) {
            r[j] = *reinterpret_cast<const float4*>(a.R + i);
            g[j] = *reinterpret_cast<const float4*>(a.w_norm + i);
            if (a.apply) {
                const float4 b = *reinterpret_cast<const float4*>(a.bo_prev + (i - c * N));
                r[j].x = fmaf(b.x, gw, r[j].x); r[j].y = fmaf(b.y, gw, r[j].y);
                r[j].z = fmaf(b.z, gw, r[j].z); r[j].w = fmaf(b.w, gw, r[j].w);
            }
        }
    }
    float ss = 0.0f;
#pragma unroll
    for (int j = 0; j < KS; ++j) {
        if (ok[j]) {
            const float sq = r[j].x * r[j].x + r[j].y * r[j].y + r[j].z * r[j].z + r[j].w * r[j].w;
            ss += sq;
            r[j] = make_float4(r[j].x * g[j].x, r[j].y * g[j].y, r[j].z * g[j].z, r[j].w * g[j].w);
        }
    }
    const float v = warp_sum(ss);
    if (lane == 0) part[warp] = v;
    __syncthreads();
    if (t == 0) {
        float s = 0.0f;
        for (int w = 0; w < WARPS; ++w) s += part[w];
        s_rs = rsqrtf(s / (float) N + a.eps);
        a.rs[c] = s_rs;
    }
    __syncthreads();
    const float rs = s_rs;
#pragma unroll
    for (int j = 0; j < KS; ++j)
        if (ok[j])
            *reinterpret_cast<float4*>(xn + t * 4 + (k0 + j) * THREADS * 4) =
                make_float4(r[j].x * rs, r[j].y * rs, r[j].z * rs, r[j].w * rs);
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800 && !defined(__HIPCC__)
#define STRATA_GR_CP_ASYNC 1
#endif
__device__ __forceinline__ void cp_async16(void* smem, const void* gmem) {
#if defined(STRATA_GR_CP_ASYNC)
    const unsigned sa = (unsigned) __cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(sa), "l"(gmem) : "memory");
#else
    *reinterpret_cast<float4*>(smem) = *reinterpret_cast<const float4*>(gmem);
#endif
}
__device__ __forceinline__ void cp_async_commit() {
#if defined(STRATA_GR_CP_ASYNC)
    asm volatile("cp.async.commit_group;\n" ::: "memory");
#endif
}
__device__ __forceinline__ void cp_async_wait1() {
#if defined(STRATA_GR_CP_ASYNC)
    asm volatile("cp.async.wait_group 1;\n" ::: "memory");
#endif
}
__device__ __forceinline__ void cp_async_wait0() {
#if defined(STRATA_GR_CP_ASYNC)
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
#endif
}

// `dot8` with its 8 activations as two float4: the same eight fmaf in the same order
__device__ __forceinline__ float dot8v(const uint4 w, const float4 x0, const float4 x1) {
    float acc = 0.0f;
    acc = fmaf(__uint_as_float(w.x << 16), x0.x, acc);
    acc = fmaf(__uint_as_float(w.x & 0xffff0000u), x0.y, acc);
    acc = fmaf(__uint_as_float(w.y << 16), x0.z, acc);
    acc = fmaf(__uint_as_float(w.y & 0xffff0000u), x0.w, acc);
    acc = fmaf(__uint_as_float(w.z << 16), x1.x, acc);
    acc = fmaf(__uint_as_float(w.z & 0xffff0000u), x1.y, acc);
    acc = fmaf(__uint_as_float(w.w << 16), x1.z, acc);
    acc = fmaf(__uint_as_float(w.w & 0xffff0000u), x1.w, acc);
    return acc;
}

// Stage tile `h` of every token into `buf`: [T][2 planes][160 chunks] float4, plane 0 = floats 0-3 of a chunk.
__device__ __forceinline__ void stage_htile(const GrMulti& m, int T, int h, float4* buf, int t, int k0 = 0) {
    for (int i = t; i < T * (H_TILE / 4); i += THREADS) {
        const int k = i / (H_TILE / 4), s4 = i - k * (H_TILE / 4);   // s4: float4 of the tile, chunk s4/2, half s4&1
        const float* src = m.xn + (size_t) (k0 + k) * D + (size_t) h * H_TILE + (size_t) s4 * 4;
        cp_async16(buf + (size_t) k * (H_TILE / 4) + (s4 & 1) * (H_TILE / 8) + (s4 >> 1), src);
    }
}

template <int MAX_T = kFusedGrMaxT, bool EXACT_T = false>
__global__ void __launch_bounds__(THREADS) gr_down_staged_kernel(GrMulti m) {
    extern __shared__ __align__(16) float4 hbuf[];      // 2 buffers x [T][2][160] float4
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int T = EXACT_T ? MAX_T : m.T;
    // sm_75: a launch of EXACT_T groups of MAX_T tokens, one per blockIdx.y (gridDim.y 1: the whole launch); every
    // (row, token) is summed by one warp exactly as before - only more blocks share the card (#down-tg)
    const int k0 = EXACT_T ? (int) blockIdx.y * MAX_T : 0;
    const bool inject_block = blockIdx.x == DOWN_BLOCKS;
    const int row = inject_block ? warp : blockIdx.x * WARPS + warp;
    const bool active = !(inject_block && (m.a[0].w_inject == nullptr || warp >= HC));
    const uint16_t* wrow = (inject_block ? m.a[0].w_inject : m.a[0].w_down) + (size_t) (active ? row : 0) * D;
    const uint4* w4 = reinterpret_cast<const uint4*>(wrow);
    const size_t buf_f4 = (size_t) T * (H_TILE / 4);    // float4 per buffer (the second one follows the first)
    float acc[MAX_T];
#pragma unroll
    for (int k = 0; k < MAX_T; ++k) acc[k] = 0.0f;
    uint4 wv[HQ], wnext[HQ];
    if (active) {
#pragma unroll
        for (int q = 0; q < HQ; ++q) wv[q] = __ldg(w4 + lane + 32 * q);
    }
    stage_htile(m, T, 0, hbuf, t, k0);
    cp_async_commit();
    stage_htile(m, T, 1, hbuf + buf_f4, t, k0);
    cp_async_commit();
#pragma unroll 1
    for (int h = 0; h < N_HTILES; ++h) {
        if (active && h + 1 < N_HTILES) {
#pragma unroll
            for (int q = 0; q < HQ; ++q) wnext[q] = __ldg(w4 + (h + 1) * (H_TILE / 8) + lane + 32 * q);
        }
        if (h + 1 < N_HTILES) cp_async_wait1();         // tile h has landed (h + 1 may still be on its way)
        else cp_async_wait0();
        __syncthreads();
        const float4* cur = hbuf + (h & 1) * buf_f4;
        if (active) {
#pragma unroll
            for (int q = 0; q < HQ; ++q) {
                const int j = lane + 32 * q;
                const Bf16x8 wvq = unpack8(wv[q]);
#pragma unroll
                for (int k = 0; k < MAX_T; ++k) {
                    if (EXACT_T || k < T) {
                        const float4* pk = cur + (size_t) k * (H_TILE / 4);
                        acc[k] += dot8u(wvq, pk[j], pk[H_TILE / 8 + j]);
                    }
                }
            }
        }
        __syncthreads();                                // every warp is done with buffer h & 1
        if (h + 2 < N_HTILES) stage_htile(m, T, h + 2, hbuf + (h & 1) * buf_f4, t, k0);
        cp_async_commit();                              // an empty group at the end keeps the wait counts simple
        if (h + 1 < N_HTILES) {
#pragma unroll
            for (int q = 0; q < HQ; ++q) wv[q] = wnext[q];
        }
    }
    if (!active) return;
    float s[MAX_T];
#pragma unroll
    for (int k = 0; k < MAX_T; ++k) s[k] = (EXACT_T || k < T) ? warp_sum(acc[k]) : 0.0f;
    // lane k writes token k (every lane holds every sum after the xor reduction)
#pragma unroll
    for (int k = 0; k < MAX_T; ++k) {
        if ((!EXACT_T && k >= T) || lane != k) continue;
        if (inject_block) {
            m.a[k0 + k].inject_out[row] = s[k];
        } else {
            const float x = s[k] / (float) HC;
            m.a[k0 + k].lo[row] = x / (1.0f + __expf(-x));
        }
    }
}

// the current device is Volta (sm_70): the fast norm/up is its default
bool cur_dev_volta() {
#if defined(__HIPCC__)
    return false;
#else
    static int per_dev[64];   // 0 unknown, 1 no, 2 yes
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) return false;
    if (!per_dev[dev]) {
        int major = 0, minor = 0;
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev);
        per_dev[dev] = major == 7 && minor == 0 ? 2 : 1;
    }
    return per_dev[dev] == 2;
#endif
}

// The tokens a down kernel may carry in one launch on the current card, and the plain read's tile: the plain read
// stages n_tok * TILEV floats (1280 on sm_75, 2560 elsewhere), staged two tiles of n_tok * H_TILE.  The shared-memory
// opt-in is set here once per device (a per-DEVICE setting: a layer split runs these kernels on two cards).
int down_chunk(bool staged, int* tile_out) {
    static bool attr[64] = {};
    static int chunk[64] = {};   // tokens the plain down kernel may carry in one launch on this card
    static int chunk_staged[64] = {};  // the same for staged
    static int tile[64] = {};    // the plain down kernel's TILEV on this card (1280 on sm_75, 2560 elsewhere)
    int dev = 0;
    cudaGetDevice(&dev);
    if (dev < 0 || dev >= 64) {
        *tile_out = 2560;
        return kFusedGrMaxT;
    }
    if (!attr[dev]) {
        int optin = 0;
        cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
        optin = strata::smem_optin_of(optin);   // STRATA_EMULATE_CC (tests only)
        // the down kernel stages n_tok*TILEV floats of dynamic shared memory - 80 KB at the full 8 tokens of the
        // CUDA tile.  sm_75 gets the smaller tile: all eight tokens fit one 40 KiB launch there (no more slicing),
        // the TQ-5 prefetch holds half the registers, and the smaller blocks raise how many of the 41-block grid
        // share an SM.  Cards whose opt-in is still below that (or that report no opt-in at all) slice the tokens;
        // the down kernel's outputs (lo, inject_out) are strictly per-token, so the chunk boundaries are safe, and
        // the up kernel below still sees every token of the batch in one launch.
#if defined(__HIPCC__)
#if defined(STRATA_HIP_GFX906)
        const bool small_tile = false;  // gfx906: 64 KiB LDS, the 2560 tile as before (8 tokens slice into launches)
#else
        const bool small_tile = true;   // all eight tokens fit gfx1100's 64 KiB LDS at this tile
#endif
#else
        int cc_maj = 0, cc_min = 0;
        cudaDeviceGetAttribute(&cc_maj, cudaDevAttrComputeCapabilityMajor, dev);
        cudaDeviceGetAttribute(&cc_min, cudaDevAttrComputeCapabilityMinor, dev);
        const bool small_tile = strata::cc_major_of(cc_maj) * 10 + strata::cc_minor_of(cc_min) == 75;
#endif
        const int tv = small_tile ? 1280 : 2560;
        tile[dev] = tv;
        int want = (int) (kFusedGrMaxT * tv * sizeof(float));
        if (optin > 0 && want > optin) want = optin;
        if (small_tile) {
            cudaFuncSetAttribute(gr_down_multi_kernel<1280>, cudaFuncAttributeMaxDynamicSharedMemorySize, want);
            cudaFuncSetAttribute(gr_down_multi_kernel<1280, kFusedGrMaxT, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, want);
            cudaFuncSetAttribute(gr_down_multi_kernel<1280, 4, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, want);
        } else {
            cudaFuncSetAttribute(gr_down_multi_kernel<2560>, cudaFuncAttributeMaxDynamicSharedMemorySize, want);
            cudaFuncSetAttribute(gr_down_multi_kernel<2560, kFusedGrMaxT, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, want);
            cudaFuncSetAttribute(gr_down_multi_kernel<2560, 4, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, want);
        }
        auto set_staged_attr = [&](auto kernel, int max_t) {
            int ws = (int) (2 * max_t * H_TILE * sizeof(float));
            if (optin > 0 && ws > optin) ws = optin;
            if (ws > 48 * 1024)
                cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, ws);
            cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
        };
        set_staged_attr(gr_down_staged_kernel<1, true>, 1);
        set_staged_attr(gr_down_staged_kernel<2, true>, 2);
        set_staged_attr(gr_down_staged_kernel<3, true>, 3);
        set_staged_attr(gr_down_staged_kernel<4, true>, 4);
        set_staged_attr(gr_down_staged_kernel<5, true>, 5);
        set_staged_attr(gr_down_staged_kernel<6, true>, 6);
        set_staged_attr(gr_down_staged_kernel<4, false>, 4);
        set_staged_attr(gr_down_staged_kernel<kFusedGrMaxT, false>, kFusedGrMaxT);
        cudaGetLastError();      // drop any error the attempt left behind
        // The opt-in is a promise a pre-Volta card does not keep: an sm_60 answers 65536 and accepts the
        // cudaFuncSetAttribute for 61440 B, then fails the LAUNCH with "invalid argument".  What such a card
        // will launch is its per-block limit, so the capacity comes from that below sm_70 - the same tokens
        // the "no opt-in" branch assumes, but taken from the attribute that is actually enforced.
#if defined(__HIPCC__)
        const int usable = optin > 0 ? optin : 48 * 1024;
#else
        int cc = 0, per_block = 0;
        cudaDeviceGetAttribute(&cc, cudaDevAttrComputeCapabilityMajor, dev);
        cudaDeviceGetAttribute(&per_block, cudaDevAttrMaxSharedMemoryPerBlock, dev);
        cc = strata::cc_major_of(cc);
        const int usable = (cc >= 7 && optin > 0) ? optin : per_block;
#endif
        const int capacity = usable / (int) (tv * sizeof(float));
        chunk[dev] = capacity < 1 ? 1 : (capacity > kFusedGrMaxT ? kFusedGrMaxT : capacity);
        const int capacity_staged = usable / (int) (2 * H_TILE * sizeof(float));
        chunk_staged[dev] = capacity_staged < 1 ? 1 : (capacity_staged > kFusedGrMaxT ? kFusedGrMaxT : capacity_staged);
        attr[dev] = true;
    }
    *tile_out = tile[dev] ? tile[dev] : 2560;
    return staged ? chunk_staged[dev] : chunk[dev];
}

// The multi read as `variant` (kHcPlain, kHcSplit or kHcStaged): the norm, the down projection in launches of as
// many tokens as fit the card, the up projection; the profile's stamps after the norm and after the down projection.
// kHcPlain is the default read exactly as before (never main's opt-in STRATA_GR_V3 path, which fused_gr_read_multi
// takes first).
#if STRATA_GR_FAST_BUILD
__global__ void __launch_bounds__(THREADS) gr_norm_fast_kernel(GrMulti m);   // below, with the AMD fast path
__global__ void __launch_bounds__(THREADS) gr_up_fast_kernel(GrMulti m);
}  // namespace
static bool gr_fast();
namespace {
#endif
// STRATA_GR_DOWN_MAX4=1 (opt-in): launches of up to 4 tokens hold 4 tokens' sums per thread instead of kFusedGrMaxT (the
// same bits; checked bitwise on the 5070 for 0.1.40). Default off: a 10-pair interleaved decode A/B on the RTX 5070 (Q2_0,
// default config) did not show the +2.8-3.5% of bench #832 (medians 0.97-0.99 of the off arm).
bool gr_down_max4() {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_GR_DOWN_MAX4");
        return v != nullptr && std::atoi(v) != 0;
    }();
    return on;
}

// the split norm with registers (gr_norm_split_fast_kernel): gfx12 only until another arch is measured;
// STRATA_GR_NORM_FAST=0 / 1 forces it off / on
bool norm_fast_on() {
#if defined(__HIPCC__)
    static const int env = [] { const char* e = std::getenv("STRATA_GR_NORM_FAST"); return e != nullptr ? (std::atoi(e) != 0 ? 1 : 0) : -1; }();
    if (env >= 0) return env != 0;
    static int per_dev[64];   // 0 unknown, 1 no, 2 yes
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) return false;
    if (!per_dev[dev]) {
        cudaDeviceProp p;
        per_dev[dev] = cudaGetDeviceProperties(&p, dev) == cudaSuccess && strata::kernels::gfx_arch_is_gfx12(p.gcnArchName) ? 2 : 1;
    }
    return per_dev[dev] == 2;
#else
    return false;
#endif
}

// STRATA_QFUSE_HC_LOCAL=0 puts the atomic-tail up kernel back under STRATA_QFUSE_HC (default: the 32-column kernel)
bool qfuse_hc_local() {
    static const bool on = [] { const char* e = std::getenv("STRATA_QFUSE_HC_LOCAL"); return e == nullptr || std::atoi(e) != 0; }();
    return on;
}
// Turing: gr_up75_kernel for 3+ tokens (bitwise gr_up_multi_kernel; fused_gr_bench cold on an RTX 2080 Ti, up incl. a
// stamp: T 3 27.3 -> 24.6 us, T 4 33.2 -> 28.7, T 6 46.6 -> 36.6); STRATA_GR_UP75=0 / 1 forces it off / on
bool up75_on() {
#if defined(__HIPCC__)
    return false;
#else
    static const int env = [] { const char* v = std::getenv("STRATA_GR_UP75"); return v ? (std::atoi(v) != 0 ? 1 : 0) : -1; }();
    if (env >= 0) return env != 0;
    static int per_dev[64];   // 0 unknown, 1 no, 2 yes
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) return false;
    if (!per_dev[dev]) {
        int major = 0, minor = 0;
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev);
        per_dev[dev] = strata::sm75_default(major, minor) ? 2 : 1;   // Turing, STRATA_SM75_PERF=1
    }
    return per_dev[dev] == 2;
#endif
}
// sm_75: the staged down kernel's tokens per block (gridDim.y = ct / tg), 0 = one block row for all of them.
// STRATA_GR_DOWN_TG=n forces n where it divides the launch (0 = off).
int down_tg(int ct) {
#if defined(__HIPCC__)
    (void) ct;
    return 0;
#else
    static const int env = [] { const char* v = std::getenv("STRATA_GR_DOWN_TG"); return v ? std::atoi(v) : -1; }();
    if (ct < 2) return 0;
    if (env >= 0) return env > 0 && env <= 3 && ct % env == 0 ? env : 0;
    // fused_gr_bench cold on an RTX 2080 Ti (down incl. a stamp): T 4 29.1 -> 27.3 us, T 6 38.5 -> 32.2 with pairs;
    // one token per block is slower at every T (the weights are read from L2 once per group)
    static int per_dev[64];   // 0 unknown, 1 no, 2 sm_75
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) return 0;
    if (!per_dev[dev]) {
        int major = 0, minor = 0;
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev);
        per_dev[dev] = strata::sm75_default(major, minor) ? 2 : 1;   // Turing, STRATA_SM75_PERF=1
    }
    return per_dev[dev] == 2 && (ct == 4 || ct == 6) ? 2 : 0;
#endif
}
void launch_multi(const GrMulti& m, int variant, cudaStream_t st, unsigned long long* stamp_buf, int stamp_i0) {
    const int n_tok = m.T;
    const bool q8w = m.a[0].s_down != nullptr;   // STRATA_HC_REQ8: w_down / w_up hold int8 codes
#if STRATA_GR_FAST_BUILD
    // gfx906: the latency-hidden norm/up (STRATA_GR_FAST=0: off) - the same sums in the same order as the kernels
    // they replace (gr_parity checks the multi-token read against single-token calls bitwise)
    const bool fast = gr_fast();
    if (variant >= kHcSplit) {
        if (norm_fast_on()) gr_norm_split_fast_kernel<<<dim3((unsigned) n_tok, HC), THREADS, 0, st>>>(m);
        else gr_norm_split_kernel<<<dim3((unsigned) n_tok, HC), THREADS, 0, st>>>(m);
    } else if (fast) gr_norm_fast_kernel<<<n_tok, THREADS, 0, st>>>(m);
    else gr_norm_multi_kernel<<<n_tok, THREADS, 0, st>>>(m);
#else
    if (variant >= kHcSplit) {
        if (norm_fast_on()) gr_norm_split_fast_kernel<<<dim3((unsigned) n_tok, HC), THREADS, 0, st>>>(m);
        else gr_norm_split_kernel<<<dim3((unsigned) n_tok, HC), THREADS, 0, st>>>(m);
    } else gr_norm_multi_kernel<<<n_tok, THREADS, 0, st>>>(m);
#endif
    if (stamp_buf) gpu_stamp(stamp_buf, stamp_i0, (void*) st);
    // STRATA_HC_Q8: the staged down kernel reads bf16 only, so Q8 weights take split's down kernel (the plain one)
    const bool staged = variant >= kHcStaged && m.a[0].s_down == nullptr;
    int tv = 2560;
    const int chunk_tok = down_chunk(staged, &tv);
    const size_t per_tok = (staged ? (size_t) 2 * H_TILE : (size_t) tv) * sizeof(float);
    static const bool no_multi_gr = [] {
        const char* no = std::getenv("STRATA_NO_MULTI_GR");
        return no != nullptr && no[0] != '\0' && no[0] != '0';
    }();
    const bool exact_t = !no_multi_gr;
    for (int c0 = 0; c0 < n_tok; c0 += chunk_tok) {
        const int ct = n_tok - c0 < chunk_tok ? n_tok - c0 : chunk_tok;
        GrMulti c{};
        if (ct == n_tok) {
            c = m;
        } else {
            c.xn = m.xn + (size_t) c0 * D;
            c.T = ct;
            for (int k = 0; k < ct; ++k) c.a[k] = m.a[c0 + k];
        }
        const size_t smem = (size_t) ct * per_tok;
        // #443, STRATA_GR_DOWN_MAX4: launches of up to 4 tokens hold 4 tokens' sums per thread instead of
        // kFusedGrMaxT - the same tile, block size, accumulation order and plain/split/staged path, so the same bits
        // (decided once per device: this runs per layer when decode is not captured)
        const bool max4 = ct <= 4 && gr_down_max4();
        if (staged) {
            // #783 PR-g (stuchapin909): a launch of exactly ct <= 6 tokens is its own instantiation, the loop bounds
            // are compile-time (the same sums in the same order); STRATA_NO_MULTI_GR=1 keeps the generic kernels
            const int tg = down_tg(ct);
            if (exact_t && tg > 0 && tg < ct) {
                const dim3 grid(DOWN_BLOCKS + 1, (unsigned) (ct / tg));
                const size_t sm_g = (size_t) tg * per_tok;
                if (tg == 1) gr_down_staged_kernel<1, true><<<grid, THREADS, sm_g, st>>>(c);
                else if (tg == 2) gr_down_staged_kernel<2, true><<<grid, THREADS, sm_g, st>>>(c);
                else gr_down_staged_kernel<3, true><<<grid, THREADS, sm_g, st>>>(c);
            } else
            if (exact_t && ct == 1) gr_down_staged_kernel<1, true><<<DOWN_BLOCKS + 1, THREADS, smem, st>>>(c);
            else if (exact_t && ct == 2) gr_down_staged_kernel<2, true><<<DOWN_BLOCKS + 1, THREADS, smem, st>>>(c);
            else if (exact_t && ct == 3) gr_down_staged_kernel<3, true><<<DOWN_BLOCKS + 1, THREADS, smem, st>>>(c);
            else if (exact_t && ct == 4) gr_down_staged_kernel<4, true><<<DOWN_BLOCKS + 1, THREADS, smem, st>>>(c);
            else if (exact_t && ct == 5) gr_down_staged_kernel<5, true><<<DOWN_BLOCKS + 1, THREADS, smem, st>>>(c);
            else if (exact_t && ct == 6) gr_down_staged_kernel<6, true><<<DOWN_BLOCKS + 1, THREADS, smem, st>>>(c);
            else if (max4) gr_down_staged_kernel<4><<<DOWN_BLOCKS + 1, THREADS, smem, st>>>(c);
            else gr_down_staged_kernel<><<<DOWN_BLOCKS + 1, THREADS, smem, st>>>(c);
        } else if (tv == 1280) {
            if (q8w) {   // STRATA_HC_REQ8
                if (max4) gr_down_multi_kernel<1280, 4, true><<<DOWN_BLOCKS + 1, THREADS, smem, st>>>(c);
                else gr_down_multi_kernel<1280, kFusedGrMaxT, true><<<DOWN_BLOCKS + 1, THREADS, smem, st>>>(c);
            } else if (max4) gr_down_multi_kernel<1280, 4><<<DOWN_BLOCKS + 1, THREADS, smem, st>>>(c);
            else gr_down_multi_kernel<1280><<<DOWN_BLOCKS + 1, THREADS, smem, st>>>(c);
        } else {
            if (q8w) {   // STRATA_HC_REQ8
                if (max4) gr_down_multi_kernel<2560, 4, true><<<DOWN_BLOCKS + 1, THREADS, smem, st>>>(c);
                else gr_down_multi_kernel<2560, kFusedGrMaxT, true><<<DOWN_BLOCKS + 1, THREADS, smem, st>>>(c);
            } else if (max4) gr_down_multi_kernel<2560, 4><<<DOWN_BLOCKS + 1, THREADS, smem, st>>>(c);
            else gr_down_multi_kernel<2560><<<DOWN_BLOCKS + 1, THREADS, smem, st>>>(c);
        }
    }
    if (stamp_buf) gpu_stamp(stamp_buf, stamp_i0 + 1, (void*) st);
#if STRATA_GR_FAST_BUILD
    if (fast && !q8w) gr_up_fast_kernel<<<UPM_BLOCKS, THREADS, 0, st>>>(m);
    else
#endif
#if !STRATA_GR_FAST_BUILD
    if (!q8w && m.a[0].q8_mixed != nullptr && qfuse_hc_local()) {
        switch (no_multi_gr ? kFusedGrMaxT : n_tok) {
            case 1: gr_up_q32_kernel<1, true><<<Q32_BLOCKS, Q32_THREADS, 0, st>>>(m); break;
            case 2: gr_up_q32_kernel<2, true><<<Q32_BLOCKS, Q32_THREADS, 0, st>>>(m); break;
            case 3: gr_up_q32_kernel<3, true><<<Q32_BLOCKS, Q32_THREADS, 0, st>>>(m); break;
            case 4: gr_up_q32_kernel<4, true><<<Q32_BLOCKS, Q32_THREADS, 0, st>>>(m); break;
            case 5: gr_up_q32_kernel<5, true><<<Q32_BLOCKS, Q32_THREADS, 0, st>>>(m); break;
            case 6: gr_up_q32_kernel<6, true><<<Q32_BLOCKS, Q32_THREADS, 0, st>>>(m); break;
            default: gr_up_q32_kernel<kFusedGrMaxT, false><<<Q32_BLOCKS, Q32_THREADS, 0, st>>>(m); break;
        }
        return;
    }
#endif
    if (q8w) {   // STRATA_HC_REQ8: the up kernel on int8 rows
        switch (no_multi_gr ? kFusedGrMaxT : n_tok) {
            case 1: gr_up_multi_kernel<1, true, true><<<UPM_BLOCKS, THREADS, 0, st>>>(m); break;
            case 2: gr_up_multi_kernel<2, true, true><<<UPM_BLOCKS, THREADS, 0, st>>>(m); break;
            case 3: gr_up_multi_kernel<3, true, true><<<UPM_BLOCKS, THREADS, 0, st>>>(m); break;
            case 4: gr_up_multi_kernel<4, true, true><<<UPM_BLOCKS, THREADS, 0, st>>>(m); break;
            case 5: gr_up_multi_kernel<5, true, true><<<UPM_BLOCKS, THREADS, 0, st>>>(m); break;
            case 6: gr_up_multi_kernel<6, true, true><<<UPM_BLOCKS, THREADS, 0, st>>>(m); break;
            default: gr_up_multi_kernel<kFusedGrMaxT, false, true><<<UPM_BLOCKS, THREADS, 0, st>>>(m); break;
        }
    } else if (n_tok >= 3 && up75_on()) {   // T 1-2: gr_up_multi_kernel is as fast or faster (fused_gr_bench, cold)
        switch (no_multi_gr ? kFusedGrMaxT : n_tok) {
            case 1: gr_up75_kernel<1, true><<<UPM_BLOCKS, THREADS, 0, st>>>(m); break;
            case 2: gr_up75_kernel<2, true><<<UPM_BLOCKS, THREADS, 0, st>>>(m); break;
            case 3: gr_up75_kernel<3, true><<<UPM_BLOCKS, THREADS, 0, st>>>(m); break;
            case 4: gr_up75_kernel<4, true><<<UPM_BLOCKS, THREADS, 0, st>>>(m); break;
            case 5: gr_up75_kernel<5, true><<<UPM_BLOCKS, THREADS, 0, st>>>(m); break;
            case 6: gr_up75_kernel<6, true><<<UPM_BLOCKS, THREADS, 0, st>>>(m); break;
            default: gr_up75_kernel<kFusedGrMaxT, false><<<UPM_BLOCKS, THREADS, 0, st>>>(m); break;
        }
    } else
    switch (no_multi_gr ? kFusedGrMaxT : n_tok) {
        case 1: gr_up_multi_kernel<1, true><<<UPM_BLOCKS, THREADS, 0, st>>>(m); break;
        case 2: gr_up_multi_kernel<2, true><<<UPM_BLOCKS, THREADS, 0, st>>>(m); break;
        case 3: gr_up_multi_kernel<3, true><<<UPM_BLOCKS, THREADS, 0, st>>>(m); break;
        case 4: gr_up_multi_kernel<4, true><<<UPM_BLOCKS, THREADS, 0, st>>>(m); break;
        case 5: gr_up_multi_kernel<5, true><<<UPM_BLOCKS, THREADS, 0, st>>>(m); break;
        case 6: gr_up_multi_kernel<6, true><<<UPM_BLOCKS, THREADS, 0, st>>>(m); break;
        default: gr_up_multi_kernel<kFusedGrMaxT, false><<<UPM_BLOCKS, THREADS, 0, st>>>(m); break;
    }
}

/// STRATA_HC_SPLIT: unset or 2 = the newest the check accepts (staged), 1 = at most split, 0 = the plain read
int env_variant() {
    const char* e = std::getenv("STRATA_HC_SPLIT");
    if (e == nullptr || e[0] == '\0') return kHcStaged;
    if (e[0] == '0') return kHcPlain;
    if (e[0] == '1') return kHcSplit;
    return kHcStaged;
}

// per device: the variant `fused_gr_check` chose (0 = not checked yet)
std::atomic<int> g_variant[64];


#if STRATA_GR_FAST_BUILD
// ---- AMD fast path (STRATA_GR_FAST, default 1; 0 = the old kernels): the same arithmetic per value in the same order, latency
// hidden.  gr_norm: one block per token (as before) but a thread's 10 float4 of R / bo / w_norm are loaded
// before any is used, and xn stays in registers until rs is known (was: store, then re-read the 40 KB).
// gr_up: a warp's 8 rows of w_up and their epilogue inputs are loaded before the dots (was: one row's load chain
// after another).
constexpr int NQ = D / (THREADS * 4);   // 10 float4 per thread
static_assert(D % (THREADS * 4) == 0, "whole float4 per thread");
__global__ void __launch_bounds__(THREADS) gr_norm_fast_kernel(GrMulti m) {
    __shared__ float part[WARPS][HC];
    __shared__ float s_rs[HC];
    const FusedGrArgs& a = m.a[blockIdx.x];
    float* xn = m.xn + (size_t) blockIdx.x * D;
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    float gw[HC];
#pragma unroll
    for (int c = 0; c < HC; ++c) gw[c] = a.apply ? 2.0f * sigmoidf_(a.inj_prev[c] / (float) HC) : 0.0f;
    float4 r[NQ], g[NQ];
#pragma unroll
    for (int k = 0; k < NQ; ++k) {
        const int i = t * 4 + k * THREADS * 4;
        r[k] = *reinterpret_cast<const float4*>(a.R + i);
        g[k] = *reinterpret_cast<const float4*>(a.w_norm + i);
    }
    if (a.apply) {
#pragma unroll
        for (int k = 0; k < NQ; ++k) {
            const int i = t * 4 + k * THREADS * 4, c = i / N, d = i - c * N;
            const float4 b = *reinterpret_cast<const float4*>(a.bo_prev + d);
            r[k].x = fmaf(b.x, gw[c], r[k].x); r[k].y = fmaf(b.y, gw[c], r[k].y);
            r[k].z = fmaf(b.z, gw[c], r[k].z); r[k].w = fmaf(b.w, gw[c], r[k].w);
        }
    }
    float ss[HC] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
    for (int k = 0; k < NQ; ++k) {
        const int i = t * 4 + k * THREADS * 4, c = i / N;
        const float sq = r[k].x * r[k].x + r[k].y * r[k].y + r[k].z * r[k].z + r[k].w * r[k].w;
#pragma unroll
        for (int cc = 0; cc < HC; ++cc) if (cc == c) ss[cc] += sq;
        r[k] = make_float4(r[k].x * g[k].x, r[k].y * g[k].y, r[k].z * g[k].z, r[k].w * g[k].w);
    }
#pragma unroll
    for (int c = 0; c < HC; ++c) {
        const float v = warp_sum(ss[c]);
        if (lane == 0) part[warp][c] = v;
    }
    __syncthreads();
    if (t < HC) {
        float s = 0.0f;
        for (int w = 0; w < WARPS; ++w) s += part[w][t];
        s_rs[t] = rsqrtf(s / (float) N + a.eps);
        a.rs[t] = s_rs[t];
    }
    __syncthreads();
#pragma unroll
    for (int k = 0; k < NQ; ++k) {
        const int i = t * 4 + k * THREADS * 4;
        const float sr = s_rs[i / N];
        *reinterpret_cast<float4*>(xn + i) = make_float4(r[k].x * sr, r[k].y * sr, r[k].z * sr, r[k].w * sr);
    }
}
// gr_up: 8 lanes per row instead of a 32-lane warp.  Lane j holds the old lanes j, j+8, j+16, j+24 (and chunk
// 32+j, the old lane j's second chunk): the xor tree's stages 16 and 8 are its own adds (a + b == b + a), stages
// 4, 2, 1 are shuffles inside the 8 - the same tree, bitwise the same sum, 3 shuffles a token instead of 5 and no
// idle lanes on the tail chunks.  A block's 64 rows are 2 passes of 32 groups, their weights loaded up front.
__device__ __forceinline__ float xor8(float v) {
#pragma unroll
#if defined(__HIPCC__)
    for (int o = 4; o > 0; o >>= 1) v += __shfl_xor(v, o, 64);
#else
    for (int o = 4; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
#endif
    return v;
}
__global__ void __launch_bounds__(THREADS) gr_up_fast_kernel(GrMulti m) {
    __shared__ __align__(16) float lo[kFusedGrMaxT][LR];
    __shared__ float g[kFusedGrMaxT][HC][UPM_COLS];
    const int t = threadIdx.x, j = t & 7, grp = t >> 3;
    const int T = m.T;
    const int d0 = blockIdx.x * UPM_COLS;
    static_assert(LR == 40 * 8 && HC * UPM_COLS == 64 && THREADS == 256, "geometry");
    uint4 w[2][5];
    float rv[2] = {0.0f, 0.0f}, wn[2] = {0.0f, 0.0f}, rsc[2] = {0.0f, 0.0f}, bo[2] = {0.0f, 0.0f}, ip[2] = {0.0f, 0.0f};
    const bool apply = j < T && m.a[j < T ? j : 0].apply;
#pragma unroll
    for (int p = 0; p < 2; ++p) {
        const int r = grp + 32 * p, c = r / UPM_COLS, dd = r - c * UPM_COLS, i = c * N + d0 + dd;
        const uint4* w4 = reinterpret_cast<const uint4*>(m.a[0].w_up + (size_t) i * LR);
#pragma unroll
        for (int q = 0; q < 4; ++q) w[p][q] = __ldg(w4 + j + 8 * q);
        w[p][4] = __ldg(w4 + 32 + j);
        if (j < T) {
            const FusedGrArgs& a = m.a[j];
            rv[p] = a.R[i];
            wn[p] = a.w_norm[i];
            rsc[p] = a.rs[c];
            if (apply) { bo[p] = a.bo_prev[d0 + dd]; ip[p] = a.inj_prev[c]; }
        }
    }
    for (int i = t; i < T * LR; i += THREADS) lo[i / LR][i % LR] = m.a[i / LR].lo[i % LR];
    __syncthreads();
#pragma unroll
    for (int p = 0; p < 2; ++p) {
        const int r = grp + 32 * p, c = r / UPM_COLS, dd = r - c * UPM_COLS, i = c * N + d0 + dd;
        float mine = 0.0f;
#pragma unroll
        for (int k = 0; k < kFusedGrMaxT; ++k) {
            if (k >= T) break;
            const float* l = lo[k];
            const float p0 = dot8(w[p][0], l + j * 8) + dot8(w[p][4], l + (32 + j) * 8);   // old lane j
            const float p1 = dot8(w[p][1], l + (j + 8) * 8);                               // old lane j + 8
            const float p2 = dot8(w[p][2], l + (j + 16) * 8);                              // old lane j + 16
            const float p3 = dot8(w[p][3], l + (j + 24) * 8);                              // old lane j + 24
            const float s = xor8((p0 + p2) + (p1 + p3));   // stage 16: (j, j+16), (j+8, j+24); stage 8; then 4, 2, 1
            if (j == k) mine = s;
        }
        if (j < T) {
            float x0 = rv[p];
            if (apply) {
                x0 = fmaf(bo[p], 2.0f * sigmoidf_(ip[p] / (float) HC), x0);
                m.a[j].R_out[i] = x0;
            }
            const float x = x0 * wn[p] * rsc[p];
            g[j][c][dd] = x * sigmoidf_(mine);
        }
    }
    __syncthreads();
    for (int i = t; i < T * UPM_COLS; i += THREADS) {
        const int k = i / UPM_COLS, col = i - k * UPM_COLS;
        float s = 0.0f;
#pragma unroll
        for (int c = 0; c < HC; ++c) s += g[k][c][col];
        m.a[k].mixed[d0 + col] = s / (float) HC;
    }
    if (m.a[0].q8_mixed != nullptr) gr_q8_tail(m, d0);   // S26 STRATA_QFUSE, as gr_up_multi_kernel
}
#endif

#if defined(STRATA_HIP_GFX906)
// ---- AMD: `gr_down_multi` split along K.  The CUDA kernel is one warp per row of w_down (320 + 4 rows), 41
// blocks: on a 60-CU gfx906 that left most of the card idle and read the 6.5 MB matrix at ~100 GB/s.  Here a block
// is 8 wavefronts = 8 rows of one K-slice of 2048 (the slice of every token's xn staged in LDS once), 41 x 5
// blocks, and a second kernel adds the 5 slice sums in a fixed order and runs the epilogue.  A token's result does
// not depend on how many tokens share the window (each column is reduced on its own).
constexpr int GS_SL = 2048;                   // K per slice
constexpr int GS_S = D / GS_SL;               // 5 slices
constexpr int GS_ROWS = LR + HC;              // 324: the down rows, then the inject rows
constexpr int GS_WAVES = 8;
static_assert(D % GS_SL == 0 && GS_SL == 64 * 8 * 4, "a lane takes 4 chunks of 8 per slice");
// the slice sums: a module-scope device array is per device by construction, and needs no allocation during a
// graph capture
__device__ float g_gr_part[GS_S * kFusedGrMaxT * GS_ROWS];
__global__ void __launch_bounds__(GS_WAVES * 64) gr_down_split_kernel(GrMulti m) {
    float* part = g_gr_part;
    extern __shared__ __align__(16) float tile[];   // [T][GS_SL]
    const int t = threadIdx.x, lane = t & 63, wave = t >> 6;
    const int T = m.T, sl = blockIdx.y;
    const bool inject_block = blockIdx.x == DOWN_BLOCKS;
    const int row = inject_block ? wave : blockIdx.x * GS_WAVES + wave;
    const bool active = !(inject_block && (m.a[0].w_inject == nullptr || wave >= HC));
    const uint16_t* wrow = (inject_block ? m.a[0].w_inject : m.a[0].w_down) + (size_t) (active ? row : 0) * D;
    const uint4* w4 = reinterpret_cast<const uint4*>(wrow) + sl * (GS_SL / 8);
    uint4 wv[4];
    if (active) {
#pragma unroll
        for (int q = 0; q < 4; ++q) wv[q] = __ldg(w4 + lane + 64 * q);
    }
    const float4* src4 = reinterpret_cast<const float4*>(m.xn);
    float4* tile4 = reinterpret_cast<float4*>(tile);
    for (int i = t; i < T * (GS_SL / 4); i += GS_WAVES * 64) {
        const int k = i / (GS_SL / 4), off = i - k * (GS_SL / 4);
        tile4[i] = src4[((size_t) k * D + (size_t) sl * GS_SL) / 4 + off];
    }
    __syncthreads();
    if (!active) return;
    float acc[kFusedGrMaxT];
#pragma unroll
    for (int k = 0; k < kFusedGrMaxT; ++k) acc[k] = 0.0f;
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        const int j = lane + 64 * q;
#pragma unroll
        for (int k = 0; k < kFusedGrMaxT; ++k)
            if (k < T) acc[k] += dot8(wv[q], tile + k * GS_SL + j * 8);
    }
#pragma unroll
    for (int k = 0; k < kFusedGrMaxT; ++k) {
        if (k >= T) break;
        float v = acc[k];
#pragma unroll
        for (int off = 32; off > 0; off >>= 1) v += __shfl_xor(v, off, 64);
        if (lane == 0) part[((size_t) sl * kFusedGrMaxT + k) * GS_ROWS + (inject_block ? LR + row : row)] = v;
    }
}
__global__ void gr_down_finish_kernel(GrMulti m) {
    const float* part = g_gr_part;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int k = i / GS_ROWS, r = i - k * GS_ROWS;
    if (k >= m.T) return;
    const bool inject = r >= LR;
    if (inject && m.a[0].w_inject == nullptr) return;
    float s = 0.0f;
#pragma unroll
    for (int sl = 0; sl < GS_S; ++sl) s += part[((size_t) sl * kFusedGrMaxT + k) * GS_ROWS + r];
    if (inject) {
        m.a[k].inject_out[r - LR] = s;
    } else {
        const float x = s / (float) HC;
        m.a[k].lo[r] = x / (1.0f + __expf(-x));
    }
}

#endif

// ================================ S23 experiment (STRATA_HC_Q8=1): the read with the GGUF's Q8_0 projections ======
// The pack holds the hyper-connection projections as BF16 (iq_pack.py --compat-bf16 rounds the GGUF's Q8_0 values to
// BF16: 1.20 GiB per verify step); this read takes the Q8_0 bytes as stored (0.65 GiB - and the GGUF's own values,
// not a rounding of them).  The layout is the stream split of the v3 read above, at 4 chunks per stream: `down` runs
// on (10 groups of 32 rows + the inject rows) x 16 chunks of 640 columns = 176 blocks, each staging its chunk of
// R' * w_norm (unscaled) for the T tokens and writing the chunk's partial dots and sums of squares; `up` sums them
// per stream in a fixed order, applies rs, SwiGLU-free silu, and runs the default epilogue.  Another summation order
// and other weights than the default read: an output-changing step (measured, quality-checked).
constexpr int Q8B = 34;                             // Q8_0 block: fp16 d + 32 int8
constexpr int Q8_RPW = 4;                           // down rows per warp
constexpr int Q8_RG = LR / (WARPS * Q8_RPW);        // 10 row groups (+1: the inject rows)
constexpr int Q8_KC = 640, Q8_NKC = D / Q8_KC;      // 16 chunks, 4 per stream
constexpr int Q8_CPS = N / Q8_KC;                   // chunks per stream
constexpr int Q8_SPB = Q8_KC / 128;                 // 5 steps of 4 Q8_0 blocks (8 lanes x 4 values each)
__device__ __forceinline__ uint32_t q8_ld16(const uint8_t* p) { return *(const uint16_t*) p; }
__device__ __forceinline__ float q8_half(uint32_t h) { return __half2float(__ushort_as_half((unsigned short) h)); }
__device__ __forceinline__ float4 q8_four(uint32_t q, float d) {
    return make_float4(d * (float) (int8_t) (q & 255u), d * (float) (int8_t) ((q >> 8) & 255u),
                       d * (float) (int8_t) ((q >> 16) & 255u), d * (float) (int8_t) (q >> 24));
}

template <int T>
__global__ void __launch_bounds__(THREADS) gr_down_q8_kernel(GrMulti m, float* __restrict__ part, float* __restrict__ ssg) {
    __shared__ __align__(16) float xs[T][Q8_KC];
    __shared__ float red[WARPS][T];
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int rg = blockIdx.x, kc = blockIdx.y, c = kc / Q8_CPS;
    const bool inj = rg == Q8_RG;
    // S25: the inject rows are F32 in the GGUF with BF16-exact values, so by default they are read as the pack's BF16
    // (exact); a Q8_0 copy (STRATA_HC_Q8_INJECT=1) is the other option
    const bool inj_bf16 = inj && m.a[0].q8_inject == nullptr;
    const int nrows = inj ? (((m.a[0].q8_inject != nullptr || m.a[0].w_inject != nullptr) && warp < HC) ? 1 : 0) : Q8_RPW;
    const int row0 = inj ? warp : (rg * WARPS + warp) * Q8_RPW;
    const uint8_t* wb = inj ? m.a[0].q8_inject : m.a[0].q8_down;
    const int sub = lane & 7, bl = lane >> 3;       // lanes 8b..8b+7: one Q8_0 block, 4 values each
    uint32_t q[Q8_RPW][Q8_SPB];
    float dq[Q8_RPW][Q8_SPB];
#pragma unroll
    for (int r = 0; r < Q8_RPW; ++r) {
        if (r >= nrows || inj_bf16) break;
        const uint8_t* row = wb + (size_t) (row0 + r) * (D / 32) * Q8B + (size_t) kc * (Q8_KC / 32) * Q8B;
#pragma unroll
        for (int s = 0; s < Q8_SPB; ++s) {
            const uint8_t* blk = row + (4 * s + bl) * Q8B;
            dq[r][s] = q8_half(q8_ld16(blk));
            q[r][s] = q8_ld16(blk + 2 + 4 * sub) | q8_ld16(blk + 4 + 4 * sub) << 16;
        }
    }
    float ssp[T];
#pragma unroll
    for (int k = 0; k < T; ++k) {
        ssp[k] = 0.0f;
        const FusedGrArgs& a = m.a[k];
        const float gw = a.apply ? 2.0f * sigmoidf_(a.inj_prev[c] / (float) HC) : 0.0f;
        for (int i = t; i < Q8_KC / 4; i += THREADS) {
            const int col = kc * Q8_KC + 4 * i;
            float4 r = *reinterpret_cast<const float4*>(a.R + col);
            if (a.apply) {
                const float4 b = *reinterpret_cast<const float4*>(a.bo_prev + (col - c * N));
                r.x = fmaf(b.x, gw, r.x); r.y = fmaf(b.y, gw, r.y);
                r.z = fmaf(b.z, gw, r.z); r.w = fmaf(b.w, gw, r.w);
            }
            const float4 g = *reinterpret_cast<const float4*>(a.w_norm + col);
            ssp[k] += r.x * r.x + r.y * r.y + r.z * r.z + r.w * r.w;
            *reinterpret_cast<float4*>(&xs[k][4 * i]) = make_float4(r.x * g.x, r.y * g.y, r.z * g.z, r.w * g.w);
        }
    }
#pragma unroll
    for (int k = 0; k < T; ++k) {
        const float v = warp_sum(ssp[k]);
        if (lane == 0) red[warp][k] = v;
    }
    __syncthreads();
    if (rg == 0 && t < T) {
        float s = 0.0f;
        for (int w = 0; w < WARPS; ++w) s += red[w][t];
        ssg[t * Q8_NKC + kc] = s;
    }
    if (nrows == 0) return;
#pragma unroll
    for (int r = 0; r < Q8_RPW; ++r) {
        if (r >= nrows) break;
        float acc[T];
#pragma unroll
        for (int k = 0; k < T; ++k) acc[k] = 0.0f;
#pragma unroll
        for (int s = 0; s < Q8_SPB; ++s) {
            const int v = 32 * (4 * s + bl) + 4 * sub;
            float4 w;
            if (inj_bf16) {
                const uint2 b = *reinterpret_cast<const uint2*>(m.a[0].w_inject + (size_t) row0 * D + (size_t) kc * Q8_KC + v);
                w = make_float4(__uint_as_float(b.x << 16), __uint_as_float(b.x & 0xffff0000u),
                                __uint_as_float(b.y << 16), __uint_as_float(b.y & 0xffff0000u));
            } else {
                w = q8_four(q[r][s], dq[r][s]);
            }
#pragma unroll
            for (int k = 0; k < T; ++k) {
                const float4 x = *reinterpret_cast<const float4*>(&xs[k][v]);
                acc[k] = fmaf(w.x, x.x, acc[k]); acc[k] = fmaf(w.y, x.y, acc[k]);
                acc[k] = fmaf(w.z, x.z, acc[k]); acc[k] = fmaf(w.w, x.w, acc[k]);
            }
        }
        const int prow = inj ? LR + warp : row0 + r;
#pragma unroll
        for (int k = 0; k < T; ++k) {
            const float v = warp_sum(acc[k]);
            if (lane == 0) part[((size_t) k * Q8_NKC + kc) * PR + prow] = v;
        }
    }
}

template <int T>
__global__ void __launch_bounds__(THREADS) gr_up_q8_kernel(GrMulti m, const float* __restrict__ part,
                                                           const float* __restrict__ ssg) {
    __shared__ __align__(16) float lo[T][LR];
    __shared__ float rsS[T][HC];
    __shared__ float g[T][HC][UPM_COLS];
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int d0 = blockIdx.x * UPM_COLS;
    if (t < T * HC) {
        const int k = t / HC, c = t - k * HC;
        float ss = 0.0f;
#pragma unroll
        for (int j = 0; j < Q8_CPS; ++j) ss += ssg[k * Q8_NKC + c * Q8_CPS + j];
        const float r = rsqrtf(ss / (float) N + m.a[k].eps);
        rsS[k][c] = r;
        if (blockIdx.x == 0) m.a[k].rs[c] = r;
    }
    __syncthreads();
    for (int i = t; i < T * PR; i += THREADS) {
        const int k = i / PR, r = i - k * PR;
        if (r >= LR && (blockIdx.x != 0 || m.a[k].w_inject == nullptr)) continue;
        float sum = 0.0f;
#pragma unroll
        for (int c = 0; c < HC; ++c) {
            float p = 0.0f;
#pragma unroll
            for (int j = 0; j < Q8_CPS; ++j) p += part[((size_t) k * Q8_NKC + c * Q8_CPS + j) * PR + r];
            sum = fmaf(rsS[k][c], p, sum);
        }
        if (r < LR) {
            const float x = sum / (float) HC;
            lo[k][r] = x / (1.0f + __expf(-x));
        } else {
            m.a[k].inject_out[r - LR] = sum;
        }
    }
    __syncthreads();
    constexpr int RPW8 = HC * UPM_COLS / WARPS;     // 8 rows per warp
#pragma unroll 1
    for (int qq = 0; qq < RPW8; ++qq) {
        const int r = warp + qq * WARPS;
        const int c = r / UPM_COLS, dd = r - c * UPM_COLS, i = c * N + d0 + dd;
        const uint8_t* row = m.a[0].q8_up + (size_t) i * (LR / 32) * Q8B;
        float4 w[3];
#pragma unroll
        for (int s = 0; s < 3; ++s) {   // lane: values 4 lane + 128 s (the third step: lanes 0-15)
            const int v = 4 * lane + 128 * s;
            if (v < LR) {
                const uint8_t* blk = row + (v >> 5) * Q8B;
                w[s] = q8_four(q8_ld16(blk + 2 + (v & 31)) | q8_ld16(blk + 4 + (v & 31)) << 16, q8_half(q8_ld16(blk)));
            } else {
                w[s] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
        }
        float rv = 0.0f, wn = 0.0f, bo = 0.0f, ip = 0.0f;
        bool apply = false;
        if (lane < T) {
            const FusedGrArgs& a = m.a[lane];
            rv = a.R[i];
            wn = a.w_norm[i];
            apply = a.apply;
            if (apply) { bo = a.bo_prev[d0 + dd]; ip = a.inj_prev[c]; }
        }
        float mine = 0.0f;
#pragma unroll
        for (int k = 0; k < T; ++k) {
            float acc = 0.0f;
#pragma unroll
            for (int s = 0; s < 3; ++s) {
                const int v = 4 * lane + 128 * s;
                if (v < LR) {
                    const float4 x = *reinterpret_cast<const float4*>(&lo[k][v]);
                    acc = fmaf(w[s].x, x.x, acc); acc = fmaf(w[s].y, x.y, acc);
                    acc = fmaf(w[s].z, x.z, acc); acc = fmaf(w[s].w, x.w, acc);
                }
            }
            acc = warp_sum(acc);
            if (lane == k) mine = acc;
        }
        if (lane < T) {
            if (apply) {
                rv = fmaf(bo, 2.0f * sigmoidf_(ip / (float) HC), rv);
                m.a[lane].R_out[i] = rv;
            }
            const float x = rv * wn * rsS[lane][c];
            g[lane][c][dd] = x * sigmoidf_(mine);
        }
    }
    __syncthreads();
    for (int i = t; i < T * UPM_COLS; i += THREADS) {
        const int k = i / UPM_COLS, col = i - k * UPM_COLS;
        float sum = 0.0f;
#pragma unroll
        for (int c = 0; c < HC; ++c) sum += g[k][c][col];
        m.a[k].mixed[d0 + col] = sum / (float) HC;
    }
    if (m.a[0].q8_mixed != nullptr) gr_q8_tail(m, d0);   // S26 STRATA_QFUSE
}


// STRATA_HC_Q8 on HIP gfx12: gr_up_q8_kernel with 32 columns / 512 threads a block, so a block owns a whole q8_1 group
// (STRATA_QFUSE_HC: no counter, no fence; the same arithmetic per value as gr_up_q8_kernel)
template <int T>
__global__ void __launch_bounds__(Q32_THREADS) gr_up_q8_q32_kernel(GrMulti m, const float* __restrict__ part,
                                                           const float* __restrict__ ssg) {
    __shared__ __align__(16) float lo[T][LR];
    __shared__ float rsS[T][HC];
    __shared__ float g[T][HC][Q32_COLS];
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int d0 = blockIdx.x * Q32_COLS;
    if (t < T * HC) {
        const int k = t / HC, c = t - k * HC;
        float ss = 0.0f;
#pragma unroll
        for (int j = 0; j < Q8_CPS; ++j) ss += ssg[k * Q8_NKC + c * Q8_CPS + j];
        const float r = rsqrtf(ss / (float) N + m.a[k].eps);
        rsS[k][c] = r;
        if (blockIdx.x == 0) m.a[k].rs[c] = r;
    }
    __syncthreads();
    for (int i = t; i < T * PR; i += Q32_THREADS) {
        const int k = i / PR, r = i - k * PR;
        if (r >= LR && (blockIdx.x != 0 || m.a[k].w_inject == nullptr)) continue;
        float sum = 0.0f;
#pragma unroll
        for (int c = 0; c < HC; ++c) {
            float p = 0.0f;
#pragma unroll
            for (int j = 0; j < Q8_CPS; ++j) p += part[((size_t) k * Q8_NKC + c * Q8_CPS + j) * PR + r];
            sum = fmaf(rsS[k][c], p, sum);
        }
        if (r < LR) {
            const float x = sum / (float) HC;
            lo[k][r] = x / (1.0f + __expf(-x));
        } else {
            m.a[k].inject_out[r - LR] = sum;
        }
    }
    __syncthreads();
    constexpr int RPW8 = HC * Q32_COLS / Q32_WARPS;     // 8 rows per warp
#pragma unroll 1
    for (int qq = 0; qq < RPW8; ++qq) {
        const int r = warp + qq * Q32_WARPS;
        const int c = r / Q32_COLS, dd = r - c * Q32_COLS, i = c * N + d0 + dd;
        const uint8_t* row = m.a[0].q8_up + (size_t) i * (LR / 32) * Q8B;
        float4 w[3];
#pragma unroll
        for (int s = 0; s < 3; ++s) {   // lane: values 4 lane + 128 s (the third step: lanes 0-15)
            const int v = 4 * lane + 128 * s;
            if (v < LR) {
                const uint8_t* blk = row + (v >> 5) * Q8B;
                w[s] = q8_four(q8_ld16(blk + 2 + (v & 31)) | q8_ld16(blk + 4 + (v & 31)) << 16, q8_half(q8_ld16(blk)));
            } else {
                w[s] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
        }
        float rv = 0.0f, wn = 0.0f, bo = 0.0f, ip = 0.0f;
        bool apply = false;
        if (lane < T) {
            const FusedGrArgs& a = m.a[lane];
            rv = a.R[i];
            wn = a.w_norm[i];
            apply = a.apply;
            if (apply) { bo = a.bo_prev[d0 + dd]; ip = a.inj_prev[c]; }
        }
        float mine = 0.0f;
#pragma unroll
        for (int k = 0; k < T; ++k) {
            float acc = 0.0f;
#pragma unroll
            for (int s = 0; s < 3; ++s) {
                const int v = 4 * lane + 128 * s;
                if (v < LR) {
                    const float4 x = *reinterpret_cast<const float4*>(&lo[k][v]);
                    acc = fmaf(w[s].x, x.x, acc); acc = fmaf(w[s].y, x.y, acc);
                    acc = fmaf(w[s].z, x.z, acc); acc = fmaf(w[s].w, x.w, acc);
                }
            }
            acc = warp_sum(acc);
            if (lane == k) mine = acc;
        }
        if (lane < T) {
            if (apply) {
                rv = fmaf(bo, 2.0f * sigmoidf_(ip / (float) HC), rv);
                m.a[lane].R_out[i] = rv;
            }
            const float x = rv * wn * rsS[lane][c];
            g[lane][c][dd] = x * sigmoidf_(mine);
        }
    }
    __syncthreads();
    for (int i = t; i < T * Q32_COLS; i += Q32_THREADS) {
        const int k = i / Q32_COLS, col = i - k * Q32_COLS;
        float sum = 0.0f;
#pragma unroll
        for (int c = 0; c < HC; ++c) sum += g[k][c][col];
        m.a[k].mixed[d0 + col] = sum / (float) HC;
    }
    __syncthreads();
    // the block owns the whole q8_1 group (32 columns): quantize it here (gr_up_q32_kernel's tail), no counter, no fence
    if (m.a[0].q8_mixed != nullptr && warp < T) {
        const float xi = m.a[warp].mixed[d0 + lane];
        float amax = fabsf(xi), sum = xi;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
        const float d = q8_1_finite(amax / 127.0f);
        const int8_t q = q8_1_quant(xi, d, amax);
        GrQ81* y = reinterpret_cast<GrQ81*>(m.a[warp].q8_mixed) + d0 / 32;
        y->qs[lane] = q;
        if (lane == 0) y->ds = q8_1_ds(d, sum);
    }
}



// S26 STRATA_TSUM=1: gr_down_q8_kernel / gr_up_q8_kernel with each row's T per-token warp sums (and the sums of
// squares) as one transposed butterfly (s26_tsum.cuh): bitwise the same values (S26 ws4 harness, every output equal,
// T 1-4: 1.00 / 1.03 / 1.03 / 1.15x per read). PF (all rows' weights in registers before the part / lo phase) was
// slower (0.92-1.03x) and is off.
using s26ts::tsum;
using s26ts::tsum_token;
using s26ts::tsum_lane;
using s26ts::pow2_ceil;
// gr_down_q8_kernel / gr_up_q8_kernel with every per-token warp sum done as one transposed butterfly per row (tsum)
template <int T, int RPW = Q8_RPW>
__global__ void __launch_bounds__(THREADS) gr_down_q8_fast_kernel(GrMulti m, float* __restrict__ part, float* __restrict__ ssg) {
    constexpr int P = pow2_ceil(T);
    constexpr int RG_ = LR / (WARPS * RPW);
    __shared__ __align__(16) float xs[T][Q8_KC];
    __shared__ float red[WARPS][T];
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int rg = blockIdx.x, kc = blockIdx.y, c = kc / Q8_CPS;
    const bool inj = rg == RG_;
    const bool inj_bf16 = inj && m.a[0].q8_inject == nullptr;
    const int nrows = inj ? (((m.a[0].q8_inject != nullptr || m.a[0].w_inject != nullptr) && warp < HC) ? 1 : 0) : RPW;
    const int row0 = inj ? warp : (rg * WARPS + warp) * RPW;
    const uint8_t* wb = inj ? m.a[0].q8_inject : m.a[0].q8_down;
    const int sub = lane & 7, bl = lane >> 3;
    const int tk = tsum_token<P>(lane);
    const bool owner = lane == tsum_lane<P>(tk) && tk < T;
    uint32_t q[RPW][Q8_SPB];
    float dq[RPW][Q8_SPB];
#pragma unroll
    for (int r = 0; r < RPW; ++r) {
        if (r >= nrows || inj_bf16) break;
        const uint8_t* row = wb + (size_t) (row0 + r) * (D / 32) * Q8B + (size_t) kc * (Q8_KC / 32) * Q8B;
#pragma unroll
        for (int s = 0; s < Q8_SPB; ++s) {
            const uint8_t* blk = row + (4 * s + bl) * Q8B;
            dq[r][s] = q8_half(q8_ld16(blk));
            q[r][s] = q8_ld16(blk + 2 + 4 * sub) | q8_ld16(blk + 4 + 4 * sub) << 16;
        }
    }
    float ssp[P];
#pragma unroll
    for (int k = 0; k < P; ++k) ssp[k] = 0.0f;
#pragma unroll
    for (int k = 0; k < T; ++k) {
        const FusedGrArgs& a = m.a[k];
        const float gw = a.apply ? 2.0f * sigmoidf_(a.inj_prev[c] / (float) HC) : 0.0f;
        for (int i = t; i < Q8_KC / 4; i += THREADS) {
            const int col = kc * Q8_KC + 4 * i;
            float4 r = *reinterpret_cast<const float4*>(a.R + col);
            if (a.apply) {
                const float4 b = *reinterpret_cast<const float4*>(a.bo_prev + (col - c * N));
                r.x = fmaf(b.x, gw, r.x); r.y = fmaf(b.y, gw, r.y);
                r.z = fmaf(b.z, gw, r.z); r.w = fmaf(b.w, gw, r.w);
            }
            const float4 g = *reinterpret_cast<const float4*>(a.w_norm + col);
            ssp[k] += r.x * r.x + r.y * r.y + r.z * r.z + r.w * r.w;
            *reinterpret_cast<float4*>(&xs[k][4 * i]) = make_float4(r.x * g.x, r.y * g.y, r.z * g.z, r.w * g.w);
        }
    }
    {
        const float v = tsum<P>(ssp, lane);
        if (owner) red[warp][tk] = v;
    }
    __syncthreads();
    if (rg == 0 && t < T) {
        float s = 0.0f;
        for (int w = 0; w < WARPS; ++w) s += red[w][t];
        ssg[t * Q8_NKC + kc] = s;
    }
    if (nrows == 0) return;
#pragma unroll
    for (int r = 0; r < RPW; ++r) {
        if (r >= nrows) break;
        float acc[P];
#pragma unroll
        for (int k = 0; k < P; ++k) acc[k] = 0.0f;
#pragma unroll
        for (int s = 0; s < Q8_SPB; ++s) {
            const int v = 32 * (4 * s + bl) + 4 * sub;
            float4 w;
            if (inj_bf16) {
                const uint2 b = *reinterpret_cast<const uint2*>(m.a[0].w_inject + (size_t) row0 * D + (size_t) kc * Q8_KC + v);
                w = make_float4(__uint_as_float(b.x << 16), __uint_as_float(b.x & 0xffff0000u),
                                __uint_as_float(b.y << 16), __uint_as_float(b.y & 0xffff0000u));
            } else {
                w = q8_four(q[r][s], dq[r][s]);
            }
#pragma unroll
            for (int k = 0; k < T; ++k) {
                const float4 x = *reinterpret_cast<const float4*>(&xs[k][v]);
                acc[k] = fmaf(w.x, x.x, acc[k]); acc[k] = fmaf(w.y, x.y, acc[k]);
                acc[k] = fmaf(w.z, x.z, acc[k]); acc[k] = fmaf(w.w, x.w, acc[k]);
            }
        }
        const int prow = inj ? LR + warp : row0 + r;
        const float v = tsum<P>(acc, lane);
        if (owner) part[((size_t) tk * Q8_NKC + kc) * PR + prow] = v;
    }
}

template <int T, bool PF = false>
__global__ void __launch_bounds__(THREADS) gr_up_q8_fast_kernel(GrMulti m, const float* __restrict__ part,
                                                                const float* __restrict__ ssg) {
    constexpr int P = pow2_ceil(T);
    constexpr int RPW8 = HC * UPM_COLS / WARPS;
    __shared__ __align__(16) float lo[T][LR];
    __shared__ float rsS[T][HC];
    __shared__ float g[T][HC][UPM_COLS];
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int d0 = blockIdx.x * UPM_COLS;
    const int tk = tsum_token<P>(lane);
    const bool owner = lane == tsum_lane<P>(tk) && tk < T;
    // PF: every row's weights in registers before the part / lo phase (its latency hides the weight read)
    uint32_t pq[PF ? RPW8 : 1][3];
    float pd[PF ? RPW8 : 1][3];
    if constexpr (PF) {
#pragma unroll
        for (int qq = 0; qq < RPW8; ++qq) {
            const int r = warp + qq * WARPS;
            const int c = r / UPM_COLS, dd = r - c * UPM_COLS, i = c * N + d0 + dd;
            const uint8_t* row = m.a[0].q8_up + (size_t) i * (LR / 32) * Q8B;
#pragma unroll
            for (int s = 0; s < 3; ++s) {
                const int v = 4 * lane + 128 * s;
                if (v < LR) {
                    const uint8_t* blk = row + (v >> 5) * Q8B;
                    pq[qq][s] = q8_ld16(blk + 2 + (v & 31)) | q8_ld16(blk + 4 + (v & 31)) << 16;
                    pd[qq][s] = q8_half(q8_ld16(blk));
                } else {
                    pq[qq][s] = 0u; pd[qq][s] = 0.0f;
                }
            }
        }
    }
    if (t < T * HC) {
        const int k = t / HC, c = t - k * HC;
        float ss = 0.0f;
#pragma unroll
        for (int j = 0; j < Q8_CPS; ++j) ss += ssg[k * Q8_NKC + c * Q8_CPS + j];
        const float r = rsqrtf(ss / (float) N + m.a[k].eps);
        rsS[k][c] = r;
        if (blockIdx.x == 0) m.a[k].rs[c] = r;
    }
    __syncthreads();
    for (int i = t; i < T * PR; i += THREADS) {
        const int k = i / PR, r = i - k * PR;
        if (r >= LR && (blockIdx.x != 0 || m.a[k].w_inject == nullptr)) continue;
        float sum = 0.0f;
#pragma unroll
        for (int c = 0; c < HC; ++c) {
            float p = 0.0f;
#pragma unroll
            for (int j = 0; j < Q8_CPS; ++j) p += part[((size_t) k * Q8_NKC + c * Q8_CPS + j) * PR + r];
            sum = fmaf(rsS[k][c], p, sum);
        }
        if (r < LR) {
            const float x = sum / (float) HC;
            lo[k][r] = x / (1.0f + __expf(-x));
        } else {
            m.a[k].inject_out[r - LR] = sum;
        }
    }
    __syncthreads();
#pragma unroll(PF ? RPW8 : 2)
    for (int qq = 0; qq < RPW8; ++qq) {
        const int r = warp + qq * WARPS;
        const int c = r / UPM_COLS, dd = r - c * UPM_COLS, i = c * N + d0 + dd;
        const uint8_t* row = m.a[0].q8_up + (size_t) i * (LR / 32) * Q8B;
        float4 w[3];
#pragma unroll
        for (int s = 0; s < 3; ++s) {
            const int v = 4 * lane + 128 * s;
            if (PF) {
                w[s] = v < LR ? q8_four(pq[PF ? qq : 0][s], pd[PF ? qq : 0][s]) : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            } else if (v < LR) {
                const uint8_t* blk = row + (v >> 5) * Q8B;
                w[s] = q8_four(q8_ld16(blk + 2 + (v & 31)) | q8_ld16(blk + 4 + (v & 31)) << 16, q8_half(q8_ld16(blk)));
            } else {
                w[s] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
        }
        float rv = 0.0f, wn = 0.0f, bo = 0.0f, ip = 0.0f;
        bool apply = false;
        if (owner) {
            const FusedGrArgs& a = m.a[tk];
            rv = a.R[i];
            wn = a.w_norm[i];
            apply = a.apply;
            if (apply) { bo = a.bo_prev[d0 + dd]; ip = a.inj_prev[c]; }
        }
        float acc[P];
#pragma unroll
        for (int k = 0; k < P; ++k) acc[k] = 0.0f;
#pragma unroll
        for (int k = 0; k < T; ++k) {
#pragma unroll
            for (int s = 0; s < 3; ++s) {
                const int v = 4 * lane + 128 * s;
                if (v < LR) {
                    const float4 x = *reinterpret_cast<const float4*>(&lo[k][v]);
                    acc[k] = fmaf(w[s].x, x.x, acc[k]); acc[k] = fmaf(w[s].y, x.y, acc[k]);
                    acc[k] = fmaf(w[s].z, x.z, acc[k]); acc[k] = fmaf(w[s].w, x.w, acc[k]);
                }
            }
        }
        const float mine = tsum<P>(acc, lane);
        if (owner) {
            if (apply) {
                rv = fmaf(bo, 2.0f * sigmoidf_(ip / (float) HC), rv);
                m.a[tk].R_out[i] = rv;
            }
            const float x = rv * wn * rsS[tk][c];
            g[tk][c][dd] = x * sigmoidf_(mine);
        }
    }
    __syncthreads();
    for (int i = t; i < T * UPM_COLS; i += THREADS) {
        const int k = i / UPM_COLS, col = i - k * UPM_COLS;
        float sum = 0.0f;
#pragma unroll
        for (int c = 0; c < HC; ++c) sum += g[k][c][col];
        m.a[k].mixed[d0 + col] = sum / (float) HC;
    }
    if (m.a[0].q8_mixed != nullptr) gr_q8_tail(m, d0);   // S26 STRATA_QFUSE
}

template <int T> void launch_q8_t(const GrMulti& m, float* part, float* ssg, cudaStream_t st, unsigned long long* stamp_buf,
                                  int stamp_i0) {
    static const bool ts = [] { const char* v = std::getenv("STRATA_TSUM"); return v && v[0] == '1'; }();
    if (stamp_buf) gpu_stamp(stamp_buf, stamp_i0, (void*) st);
    if (ts) gr_down_q8_fast_kernel<T, 4><<<dim3(LR / (WARPS * 4) + 1, Q8_NKC), THREADS, 0, st>>>(m, part, ssg);
    else gr_down_q8_kernel<T><<<dim3(Q8_RG + 1, Q8_NKC), THREADS, 0, st>>>(m, part, ssg);
    if (stamp_buf) gpu_stamp(stamp_buf, stamp_i0 + 1, (void*) st);
#if !STRATA_GR_FAST_BUILD
    if (m.a[0].q8_mixed != nullptr && qfuse_hc_local()) {   // HIP (not gfx906): the block owns a q8_1 group
        gr_up_q8_q32_kernel<T><<<Q32_BLOCKS, Q32_THREADS, 0, st>>>(m, part, ssg);
        return;
    }
#endif
    if (ts) gr_up_q8_fast_kernel<T, false><<<UPM_BLOCKS, THREADS, 0, st>>>(m, part, ssg);
    else gr_up_q8_kernel<T><<<UPM_BLOCKS, THREADS, 0, st>>>(m, part, ssg);
}
void launch_q8(const GrMulti& m, float* scratch, cudaStream_t st, unsigned long long* stamp_buf, int stamp_i0) {
    float* part = scratch;
    float* ssg = scratch + (size_t) m.T * Q8_NKC * PR;
    switch (m.T) {
        case 1: launch_q8_t<1>(m, part, ssg, st, stamp_buf, stamp_i0); break;
        case 2: launch_q8_t<2>(m, part, ssg, st, stamp_buf, stamp_i0); break;
        case 3: launch_q8_t<3>(m, part, ssg, st, stamp_buf, stamp_i0); break;
        case 4: launch_q8_t<4>(m, part, ssg, st, stamp_buf, stamp_i0); break;
        case 5: launch_q8_t<5>(m, part, ssg, st, stamp_buf, stamp_i0); break;
        case 6: launch_q8_t<6>(m, part, ssg, st, stamp_buf, stamp_i0); break;
        case 7: launch_q8_t<7>(m, part, ssg, st, stamp_buf, stamp_i0); break;
        default: launch_q8_t<8>(m, part, ssg, st, stamp_buf, stamp_i0); break;
    }
}

// ================================ the hc8 read (STRATA_EXL3_HC8=1) ================================
// Ported from Basalt (github.com/jesdga95/basalt @ fb0944c: src/kernels/cuda/hc8_gr.cu, the read's three kernels and
// launcher, and src/kernels/cuda/hc_fused.cu's hc8_quant_kernel).  Copyright (c) 2026 Jesús Gulfo, MIT License; the
// full notice is in third_party/basalt/LICENSE.  Changes: the 8-bit copies come as explicit q / s pointers in
// FusedGrArgs (no hc8_view registry), `mixed`'s q8_1 image (STRATA_QFUSE) is written by the up kernel's last block of a
// 32-column group, and the GPU stamps of the verify profile are kept.
//
//   norm : R' = R + bo_prev * 2 sigmoid(inj_prev / hc) into R_out (when apply),  rs[c] = rsqrt(mean(R'[c]^2) + eps)
//   down : lo[k] = silu((q_down[k] . xn) / hc),  inject[c] = q_inject[c] . xn,  xn = R' * w_norm * rs (on the fly)
//   up   : mixed[d] = mean_c xn[c,d] * sigmoid(q_up[c*n_embd + d] . lo)
//
// The codes are int8 [rows][cols] with one f16 scale per 32 consecutive values of a row (s = f16(absmax / 127),
// q = rint(x / s)).  Every token's values come from the same operations in the same order whatever the token count.
// Not bitwise the BF16 read (weight rounding, ~1e-3 KL): opt-in.  It does not write xn (no consumer reads it after
// the read).
#if !defined(__HIPCC__)
namespace hc8 {
constexpr int MT = kFusedGrMaxT;
constexpr int MAX_HC = 4;
constexpr int MAX_LR = 512;
constexpr int G8 = 32;
constexpr int DOWN_ROWS = 2;      // rows per down CTA
constexpr int DOWN_THREADS = 160;
constexpr int UP_THREADS = 256;
constexpr int UP_COLS8 = UP_THREADS / 32;   // columns per up CTA (a warp a column)
static_assert(N % UP_COLS8 == 0 && 32 % UP_COLS8 == 0, "an up CTA never straddles a q8_1 group");
static_assert(HC <= MAX_HC && LR <= MAX_LR && LR % 4 == 0 && N % G8 == 0, "hc8 geometry");

struct P {
    const float* R[MT];
    float* R_out[MT];
    const float* bo[MT];
    const float* inj_prev[MT];
    float* lo[MT];
    float* rs[MT];
    float* inj_out[MT];
    float* mixed[MT];
    uint8_t* q8_mixed[MT];
    unsigned* q8_cnt;
    const float* w_norm;
    const int8_t* qd;
    const uint16_t* sd;
    const int8_t* qu;
    const uint16_t* su;
    const int8_t* qi;
    const uint16_t* si;
    int sd_row, su_row, si_row;
    int T, n, hc, lr, ninj;
    bool apply;
    float eps;
};

__device__ __forceinline__ const float* src_of(const P& p, int t) { return p.apply ? p.R_out[t] : p.R[t]; }
__device__ __forceinline__ float s8(uint32_t w, int j) { return (float) (int) (int8_t) (w >> (8 * j)); }
__device__ __forceinline__ float f16f(uint16_t h) { return __half2float(__ushort_as_half(h)); }

// one CTA per (stream, token)
__global__ void norm_kernel(P p) {
    __shared__ float red[32];
    const int c = blockIdx.x, t = blockIdx.y;
    const int n = p.n;
    const float* r = p.R[t] + (size_t) c * n;
    float w = 0.0f;
    if (p.apply) w = 2.0f / (1.0f + expf(-p.inj_prev[t][c] / (float) p.hc));
    float ss = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float v = r[i];
        if (p.apply) {
            v = fmaf(p.bo[t][i], w, v);
            p.R_out[t][(size_t) c * n + i] = v;
        }
        ss = fmaf(v, v, ss);
    }
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = ss;
    __syncthreads();
    if (threadIdx.x == 0) {
        float s = 0.0f;
        for (int k = 0; k < (int) (blockDim.x >> 5); ++k) s += red[k];
        p.rs[t][c] = rsqrtf(s / (float) n + p.eps);
    }
}

// DOWN_ROWS rows of [q_down; q_inject] per CTA; a thread takes whole scale groups
__global__ void __launch_bounds__(DOWN_THREADS) down_kernel(P p) {
    __shared__ float red[DOWN_THREADS / 32][DOWN_ROWS][MT];
    const int blk = blockIdx.x;
    const int D8 = p.n * p.hc, gpr = D8 / G8, nrows = p.lr + p.ninj;
    const int8_t* q[DOWN_ROWS];
    const uint16_t* s[DOWN_ROWS];
#pragma unroll
    for (int r = 0; r < DOWN_ROWS; ++r) {
        const int k = min(blk * DOWN_ROWS + r, nrows - 1);
        if (k < p.lr) { q[r] = p.qd + (size_t) k * D8; s[r] = p.sd + (size_t) k * p.sd_row; }
        else { q[r] = p.qi + (size_t) (k - p.lr) * D8; s[r] = p.si + (size_t) (k - p.lr) * p.si_row; }
    }
    float acc[DOWN_ROWS][MT];
#pragma unroll
    for (int r = 0; r < DOWN_ROWS; ++r)
#pragma unroll
        for (int t = 0; t < MT; ++t) acc[r][t] = 0.0f;
    for (int g = threadIdx.x; g < gpr; g += blockDim.x) {
        const int col0 = g * G8, c = col0 / p.n;
        float wq[DOWN_ROWS][G8];
        float sc[DOWN_ROWS];
        {
            float wn[G8];
            const float4* w4 = reinterpret_cast<const float4*>(p.w_norm + col0);
#pragma unroll
            for (int j = 0; j < G8 / 4; ++j) {
                const float4 v = w4[j];
                wn[4 * j] = v.x; wn[4 * j + 1] = v.y; wn[4 * j + 2] = v.z; wn[4 * j + 3] = v.w;
            }
#pragma unroll
            for (int r = 0; r < DOWN_ROWS; ++r) {
                const uint4 a = reinterpret_cast<const uint4*>(q[r] + col0)[0];
                const uint4 b = reinterpret_cast<const uint4*>(q[r] + col0)[1];
                const uint32_t w[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
#pragma unroll
                for (int i = 0; i < G8; ++i) wq[r][i] = s8(w[i >> 2], i & 3) * wn[i];
                sc[r] = f16f(s[r][g]);
            }
        }
#pragma unroll
        for (int t = 0; t < MT; ++t) {
            if (t < p.T) {
                const float4* x4 = reinterpret_cast<const float4*>(src_of(p, t) + col0);
                float d[DOWN_ROWS];
#pragma unroll
                for (int r = 0; r < DOWN_ROWS; ++r) d[r] = 0.0f;
#pragma unroll
                for (int j = 0; j < G8 / 4; ++j) {
                    const float4 v = x4[j];
#pragma unroll
                    for (int r = 0; r < DOWN_ROWS; ++r) {
                        d[r] = fmaf(wq[r][4 * j], v.x, d[r]);
                        d[r] = fmaf(wq[r][4 * j + 1], v.y, d[r]);
                        d[r] = fmaf(wq[r][4 * j + 2], v.z, d[r]);
                        d[r] = fmaf(wq[r][4 * j + 3], v.w, d[r]);
                    }
                }
                const float rs = p.rs[t][c];
#pragma unroll
                for (int r = 0; r < DOWN_ROWS; ++r) acc[r][t] = fmaf(sc[r] * rs, d[r], acc[r][t]);
            }
        }
    }
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
#pragma unroll
    for (int r = 0; r < DOWN_ROWS; ++r)
#pragma unroll
        for (int t = 0; t < MT; ++t) {
            if (t < p.T) {
                float v = acc[r][t];
                for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
                if (lane == 0) red[warp][r][t] = v;
            }
        }
    __syncthreads();
    if (threadIdx.x < DOWN_ROWS * MT) {
        const int r = threadIdx.x / MT, t = threadIdx.x % MT, k = blk * DOWN_ROWS + r;
        if (t < p.T && k < nrows) {
            float v = 0.0f;
            for (int w = 0; w < (int) (blockDim.x >> 5); ++w) v += red[w][r][t];
            if (k < p.lr) {
                const float x = v / (float) p.hc;
                p.lo[t][k] = x / (1.0f + expf(-x));
            } else {
                p.inj_out[t][k - p.lr] = v;
            }
        }
    }
}

// STRATA_QFUSE: the q8_1 image of `mixed` (gr_q8_tail's quantizer, the bytes native_quantize_q8_1 writes): the last of
// the 32 / UP_COLS8 CTAs that own a 32-column group (a per-group counter, reset by that CTA) reads the group back.
__device__ __forceinline__ void q8_tail(const P& p, int d0) {
    __shared__ int last;
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned* c = p.q8_cnt + d0 / 32;
        last = atomicAdd(c, 1u) == (unsigned) (32 / UP_COLS8 - 1);
        if (last) atomicExch(c, 0u);
    }
    __syncthreads();
    if (!last) return;
    __threadfence();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if (warp >= p.T) return;
    const int c0 = (d0 / 32) * 32;
    const float xi = *reinterpret_cast<const volatile float*>(p.mixed[warp] + c0 + lane);
    float amax = fabsf(xi), sum = xi;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
    const float d = q8_1_finite(amax / 127.0f);
    const int8_t q = q8_1_quant(xi, d, amax);
    GrQ81* y = reinterpret_cast<GrQ81*>(p.q8_mixed[warp]) + c0 / 32;
    y->qs[lane] = q;
    if (lane == 0) y->ds = q8_1_ds(d, sum);
}

// one warp per output column d: the hc rows c*n + d, lo staged in shared memory; a lane takes 4-value chunks
__global__ void __launch_bounds__(UP_THREADS) up_kernel(P p) {
    __shared__ __align__(16) float slo[MT * MAX_LR];
    const int lr = p.lr;
    for (int i = threadIdx.x; i < p.T * lr; i += blockDim.x) slo[i] = p.lo[i / lr][i % lr];
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int d0 = blockIdx.x * UP_COLS8;
    const int d = d0 + warp;   // n % UP_COLS8 == 0: every warp has a column
    const int nch = lr / 4;   // chunks of 4 per row
    float part[MAX_HC][MT];
#pragma unroll
    for (int c = 0; c < MAX_HC; ++c)
#pragma unroll
        for (int t = 0; t < MT; ++t) part[c][t] = 0.0f;
#pragma unroll
    for (int c = 0; c < MAX_HC; ++c) {
        if (c < p.hc) {
            const size_t row = (size_t) c * p.n + d;
            const uint32_t* qr = reinterpret_cast<const uint32_t*>(p.qu + row * lr);
            const uint16_t* sr = p.su + row * p.su_row;
#pragma unroll
            for (int j = 0; j < MAX_LR / 128; ++j) {
                const int ch = lane + 32 * j;
                if (ch < nch) {
                    const uint32_t w = qr[ch];
                    const float sc = f16f(sr[ch / (G8 / 4)]);
                    const float q0 = s8(w, 0), q1 = s8(w, 1), q2 = s8(w, 2), q3 = s8(w, 3);
#pragma unroll
                    for (int t = 0; t < MT; ++t) {
                        if (t < p.T) {
                            const float4 l = *reinterpret_cast<const float4*>(slo + t * lr + 4 * ch);
                            float v = q0 * l.x;
                            v = fmaf(q1, l.y, v);
                            v = fmaf(q2, l.z, v);
                            v = fmaf(q3, l.w, v);
                            part[c][t] = fmaf(sc, v, part[c][t]);
                        }
                    }
                }
            }
        }
    }
#pragma unroll
    for (int c = 0; c < MAX_HC; ++c)
#pragma unroll
        for (int t = 0; t < MT; ++t)
            if (c < p.hc && t < p.T)
                for (int o = 16; o > 0; o >>= 1) part[c][t] += __shfl_xor_sync(0xffffffffu, part[c][t], o);
#pragma unroll
    for (int t = 0; t < MT; ++t) {
        if (t < p.T && lane == t) {
            const float* src = src_of(p, t);
            float m = 0.0f;
#pragma unroll
            for (int c = 0; c < MAX_HC; ++c) {
                if (c < p.hc) {
                    const size_t i = (size_t) c * p.n + d;
                    const float xn = src[i] * p.w_norm[i] * p.rs[t][c];
                    m = fmaf(xn, 1.0f / (1.0f + expf(-part[c][t])), m);
                }
            }
            p.mixed[t][d] = m / (float) p.hc;
        }
    }
    if (p.q8_cnt != nullptr) q8_tail(p, d0);
}

// hc_fused.cu's hc8_quant_kernel: q = round(w / s) in [-127, 127], s = f16(absmax / 127) per G8 consecutive values
// of a row; a row's `gpr` scales start every `srow` f16
__global__ void quant_kernel(const uint16_t* w, int64_t groups, int gpr, int srow, int8_t* q, uint16_t* sc, float* s32) {
    const int64_t gi = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (gi >= groups) return;
    float x[G8], amax = 0.0f;
    for (int i = 0; i < G8; ++i) {
        x[i] = __uint_as_float((uint32_t) w[gi * G8 + i] << 16);
        amax = fmaxf(amax, fabsf(x[i]));
    }
    const __half h = __float2half_rn(amax / 127.0f);
    const float s = __half2float(h);
    sc[gi / gpr * srow + gi % gpr] = __half_as_ushort(h);
    if (s32 != nullptr) s32[gi] = s;
    for (int i = 0; i < G8; ++i) {
        const float v = s > 0.0f ? fminf(fmaxf(rintf(x[i] / s), -127.0f), 127.0f) : 0.0f;
        q[gi * G8 + i] = (int8_t) v;
    }
}

// The read on the 8-bit copies; true when it wrote the q8_1 images (every token's q8_mixed set).
bool launch(const GrMulti& m, cudaStream_t st, unsigned long long* stamp_buf, int stamp_i0) {
    const FusedGrArgs& a0 = m.a[0];
    const bool inj = a0.w_inject != nullptr || a0.hc8_qi != nullptr;
    if (a0.hc8_qd == nullptr || a0.hc8_qu == nullptr || a0.hc8_sd == nullptr || a0.hc8_su == nullptr ||
        (inj && (a0.hc8_qi == nullptr || a0.hc8_si == nullptr)) || a0.hc8_sd_row < D / 32 ||
        a0.hc8_su_row < LR / 32 || (inj && a0.hc8_si_row < D / 32)) {
        std::fprintf(stderr, "hc8 read: a read's matrices are not all 8-bit copies (down %p up %p inject %p)\n",
                     (const void*) a0.hc8_qd, (const void*) a0.hc8_qu, (const void*) a0.hc8_qi);
        std::exit(1);
    }
    P p{};
    bool q8 = a0.q8_cnt != nullptr;
    for (int t = 0; t < m.T; ++t) {
        const FusedGrArgs& x = m.a[t];
        if ((inj && !x.inject_out) || x.apply != a0.apply || x.hc8_qd != a0.hc8_qd || x.hc8_qu != a0.hc8_qu ||
            x.hc8_qi != a0.hc8_qi) {
            std::fprintf(stderr, "hc8 read: invalid arguments for token %d\n", t);
            std::exit(1);
        }
        p.R[t] = x.R; p.R_out[t] = x.R_out; p.bo[t] = x.bo_prev; p.inj_prev[t] = x.inj_prev;
        p.lo[t] = x.lo; p.rs[t] = x.rs; p.inj_out[t] = x.inject_out; p.mixed[t] = x.mixed;
        p.q8_mixed[t] = x.q8_mixed;
        q8 = q8 && x.q8_mixed != nullptr;
    }
    p.q8_cnt = q8 ? a0.q8_cnt : nullptr;
    p.w_norm = a0.w_norm;
    p.qd = a0.hc8_qd; p.sd = a0.hc8_sd; p.sd_row = a0.hc8_sd_row;
    p.qu = a0.hc8_qu; p.su = a0.hc8_su; p.su_row = a0.hc8_su_row;
    if (inj) { p.qi = a0.hc8_qi; p.si = a0.hc8_si; p.si_row = a0.hc8_si_row; }
    p.T = m.T; p.n = N; p.hc = HC; p.lr = LR; p.ninj = inj ? HC : 0;
    p.apply = a0.apply; p.eps = a0.eps;
    norm_kernel<<<dim3((unsigned) HC, (unsigned) m.T), 256, 0, st>>>(p);
    if (stamp_buf) gpu_stamp(stamp_buf, stamp_i0, (void*) st);
    down_kernel<<<(unsigned) ((p.lr + p.ninj + DOWN_ROWS - 1) / DOWN_ROWS), DOWN_THREADS, 0, st>>>(p);
    if (stamp_buf) gpu_stamp(stamp_buf, stamp_i0 + 1, (void*) st);
    up_kernel<<<(unsigned) (N / UP_COLS8), UP_THREADS, 0, st>>>(p);
    return q8;
}
}  // namespace hc8
#endif

// ================================ STRATA_HC_FUSED: the fused hyper-connection read (+ router + top-10) ================
// The work-queue design of Basalt's one-launch HC read (src/kernels/cuda/hc_fused.cu, github.com/jesdga95/basalt,
// Copyright (c) 2026 Jesús Gulfo, MIT License; the full notice is in third_party/basalt/LICENSE), redesigned for sm_80+
// cards without TMA / mbarrier / PDL (the 3090's 82 SMs): one persistent kernel whose CTAs claim items in order through
// a global ticket and wait only on items claimed before their own, so the read finishes however few CTAs are resident:
//
//   norm items   (token, stream)                         rs, and xn (bf16) / R' (hc8)
//   down items   8 rows of [w_down; w_inject] (a warp a row) for one half of the window's tokens (T >= 2): lo, inject
//                                                         wait: every norm item
//   up items     32 columns (x 4 streams), four copy groups  mixed, R <- R' (bf16)   wait: every down item
//   router items 8 rows of ffn_gate_inp each               logits      wait: every up item  (the FFN-side read, T >= 2)
//   top-10       run by the last router item to finish     ids, weights
//
// A down item's first weight tiles are loaded before its wait; the up and router items copy their weights into shared
// memory (cp.async, evict-first) as soon as they are claimed.  Every output is computed with exactly the operations, in
// exactly the order, of the kernels it replaces: the bf16 read's staged variant (gr_norm_split / gr_down_staged /
// gr_up_multi), the hc8 read (hc8::norm / down / up), the router GEMV (bf16_f32_mmvf_multi_kernel, 256 threads) and
// native_router_top10_multi - the last two are compiled with --use_fast_math in their own files, so their arithmetic is
// spelled here in the ftz PTX they compile to; several xor-butterfly warp sums run as one transposed butterfly
// (s26_tsum.cuh: the same pairs added).  The fused read is therefore BITWISE the unfused one (hcf_bench checks every
// output for T 1..8, both reads, with and without the pending write, inject and router); only launches and overlap
// change.  Measured on the 3090 (hcf_bench, cold weights, T 4 in a graph): 0.80-0.87x the unfused read's speed -
// see BASALT-PORT-ANALYSIS-2026-10-21.md 3.5 for why.
#if !defined(__HIPCC__)
namespace hcf {
constexpr int MT = kFusedGrMaxT;
constexpr int NWARP = THREADS / 32;   // 8
constexpr int NE = 512, KT = 10;
constexpr int UPB = 32;               // bf16 up item: columns (x 4 streams = 128 rows, 16 a warp)
constexpr int UP8 = 32;               // hc8 up item: columns (4 a warp)
constexpr int RTR = 8;                // router item: rows
constexpr int T8 = 1024;              // hc8 down tile: columns (32 groups of 32, one a lane)
static_assert(N % UPB == 0 && N % UP8 == 0 && NE % RTR == 0 && D % T8 == 0 && LR % 32 == 0, "hcf geometry");

struct alignas(256) Ctr { unsigned v; };
struct Sync { Ctr q, nrm, dn, up, rt, ex; };

struct L {
    const float* R[MT];
    float* R_out[MT];
    const float* bo[MT];
    const float* inj_prev[MT];
    float* lo[MT];
    float* rs[MT];
    float* inj_out[MT];
    float* mixed[MT];
    float* xn;
    const float* w_norm;
    const uint16_t* w_down;
    const uint16_t* w_up;
    const uint16_t* w_inject;
    const int8_t* qd;
    const uint16_t* sd;
    const int8_t* qu;
    const uint16_t* su;
    const int8_t* qi;
    const uint16_t* si;
    int sd_row, su_row, si_row;
    const uint16_t* w_rt;   // null: no router items
    float* logits;          // [T][NE]
    int32_t* ids;           // [T][KT]
    float* wts;             // [T][KT]
    Sync* sy;
    int nN, nD, nU, nR, nK;
    int apply, inj;
    float eps;
};

// --- the fast-math router's arithmetic (native_bf16.cu / native_router.cu are built with --use_fast_math) ---
__device__ __forceinline__ float fz_add(float a, float b) { float r; asm("add.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r; }
__device__ __forceinline__ float fz_sub(float a, float b) { float r; asm("sub.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r; }
__device__ __forceinline__ float fz_mul(float a, float b) { float r; asm("mul.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r; }
__device__ __forceinline__ float fz_max(float a, float b) { float r; asm("max.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r; }
__device__ __forceinline__ float fz_fma(float a, float b, float c) { float r; asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(r) : "f"(a), "f"(b), "f"(c)); return r; }
__device__ __forceinline__ float fz_rcp(float a) { float r; asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(a)); return r; }
__device__ __forceinline__ float fz_exp(float a) {   // expf under --use_fast_math: ex2.approx.ftz(x * log2(e))
    float r; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(fz_mul(a, 1.4426950216293335f))); return r;
}
__device__ __forceinline__ float fz_warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fz_add(v, __shfl_xor_sync(0xffffffffu, v, o, 32));
    return v;
}
__device__ __forceinline__ float fz_warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fz_max(v, __shfl_xor_sync(0xffffffffu, v, o, 32));
    return v;
}

template <int NP>
__device__ __forceinline__ void cp_async_wait_n() {
#if defined(STRATA_GR_CP_ASYNC)
    asm volatile("cp.async.wait_group %0;\n" ::"n"(NP) : "memory");
#endif
}
// shared-memory ring depth of the down items' activation tiles (one CTA an SM: up to ~80 KB)
template <int T> __host__ __device__ constexpr int nb_bf() { return T <= 4 ? 4 : 2; }
template <int T> __host__ __device__ constexpr int nb_h8() { return T <= 4 ? 4 : 2; }

// Several xor-butterfly warp sums at once, bitwise each `for (o = 16; o; o >>= 1) v += shfl_xor(v, o)` (s26_tsum.cuh's
// transposed butterfly; FZ: the adds as add.ftz, the fast-math router's): value j's sum ends in lane tsum_lane<CNT>(j).
template <int CNT, bool FZ, int O = 16>
__device__ __forceinline__ float tsum_x(float* v, int lane) {
    if constexpr (CNT > 1) {
        constexpr int H = CNT / 2;
        const bool hi = (lane & O) != 0;
#pragma unroll
        for (int j = 0; j < H; ++j) {
            const float send = hi ? v[j] : v[j + H];
            const float keep = hi ? v[j + H] : v[j];
            const float got = __shfl_xor_sync(0xffffffffu, send, O);
            v[j] = FZ ? fz_add(keep, got) : keep + got;
        }
        return tsum_x<H, FZ, O / 2>(v, lane);
    } else {
        float x = v[0];
#pragma unroll
        for (int o = O; o > 0; o >>= 1) { const float got = __shfl_xor_sync(0xffffffffu, x, o); x = FZ ? fz_add(x, got) : x + got; }
        return x;
    }
}
__host__ __device__ constexpr int p2(int t) { return t <= 1 ? 1 : t <= 2 ? 2 : t <= 4 ? 4 : t <= 8 ? 8 : t <= 16 ? 16 : 32; }
template <int P>
__device__ __forceinline__ int tsum_lane(int k) { return s26ts::tsum_lane<P>(k); }

// The weight streams are read once a read and are larger than the L2 (6 MB on the 3090): load them evict-first, so
// the activations every down item re-reads (xn / R', lo, mixed) stay in L2.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800 && !defined(__HIPCC__)
#define HCF_L2HINT 1
#endif
__device__ __forceinline__ uint64_t pol_ef() {
    uint64_t p = 0;
#if defined(HCF_L2HINT)
    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(p));
#endif
    return p;
}
__device__ __forceinline__ uint4 ldg_ef(const void* ptr, uint64_t pol) {
#if defined(HCF_L2HINT)
    uint4 r;
    asm volatile("ld.global.nc.L2::cache_hint.v4.u32 {%0, %1, %2, %3}, [%4], %5;"
                 : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(ptr), "l"(pol));
    return r;
#else
    (void) pol;
    return __ldg(reinterpret_cast<const uint4*>(ptr));
#endif
}
__device__ __forceinline__ uint32_t ldg_ef_u16(const uint16_t* ptr, uint64_t pol) {
#if defined(HCF_L2HINT)
    unsigned short r;
    asm volatile("ld.global.nc.L2::cache_hint.u16 %0, [%1], %2;" : "=h"(r) : "l"(ptr), "l"(pol));
    return r;
#else
    (void) pol;
    return __ldg(ptr);
#endif
}
__device__ __forceinline__ void cp_async16_ef(void* smem, const void* gmem, uint64_t pol) {
#if defined(STRATA_GR_CP_ASYNC) && defined(HCF_L2HINT)
    const unsigned sa = (unsigned) __cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, %2;\n" ::"r"(sa), "l"(gmem), "l"(pol) : "memory");
#else
    (void) pol;
    cp_async16(smem, gmem);
#endif
}

// --- the queue ---
__device__ __forceinline__ unsigned ld_acquire(const unsigned* p) {
    unsigned v;
    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}
__device__ unsigned long long* g_trace = nullptr;   // hcf_bench only: per item [item, sm, claimed, waited, done] ns
__device__ __forceinline__ unsigned long long gtimer() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}
__device__ __forceinline__ unsigned smid() { unsigned r; asm volatile("mov.u32 %0, %%smid;" : "=r"(r)); return r; }
__shared__ unsigned long long s_tw;
__device__ __forceinline__ void wait_ge(const unsigned* p, unsigned n) {
    if (threadIdx.x == 0) {
        while (ld_acquire(p) < n) __nanosleep(32);
        if (g_trace != nullptr) s_tw = gtimer();
    }
    __syncthreads();
}
// every thread's stores are fenced, then one release of the item
__device__ __forceinline__ void signal(unsigned* p) {
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) atomicAdd(p, 1u);
}

// ---------------- bf16: gr_norm_split_kernel for (token t, stream c) ----------------
// The thread's (at most three) float4 of stream c held in registers (gr_norm_split_fast_kernel's form: the same
// elements, operations and order as gr_norm_split_kernel, so bitwise its rs and xn, without reading xn back).
template <int T>
__device__ __forceinline__ void norm_bf16(const L& l, int t, int c) {
    __shared__ float part[NWARP];
    __shared__ float s_rs;
    float* xn = l.xn + (size_t) t * D;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const float gw = l.apply ? 2.0f * sigmoidf_(l.inj_prev[t][c] / (float) HC) : 0.0f;
    constexpr int KS = 3;
    int k0 = (c * N - 4 * tid + THREADS * 4 - 1) / (THREADS * 4);
    if (k0 < 0) k0 = 0;
    float4 r[KS], g[KS];
    bool ok[KS];
#pragma unroll
    for (int j = 0; j < KS; ++j) {
        const int i = tid * 4 + (k0 + j) * THREADS * 4;
        ok[j] = i < (c + 1) * N;
        if (ok[j]) {
            r[j] = *reinterpret_cast<const float4*>(l.R[t] + i);
            g[j] = *reinterpret_cast<const float4*>(l.w_norm + i);
            if (l.apply) {
                const float4 b = *reinterpret_cast<const float4*>(l.bo[t] + (i - c * N));
                r[j].x = fmaf(b.x, gw, r[j].x); r[j].y = fmaf(b.y, gw, r[j].y);
                r[j].z = fmaf(b.z, gw, r[j].z); r[j].w = fmaf(b.w, gw, r[j].w);
            }
        }
    }
    float ss = 0.0f;
#pragma unroll
    for (int j = 0; j < KS; ++j) {
        if (ok[j]) {
            const float sq = r[j].x * r[j].x + r[j].y * r[j].y + r[j].z * r[j].z + r[j].w * r[j].w;
            ss += sq;
            r[j] = make_float4(r[j].x * g[j].x, r[j].y * g[j].y, r[j].z * g[j].z, r[j].w * g[j].w);
        }
    }
    const float v = warp_sum(ss);
    if (lane == 0) part[warp] = v;
    __syncthreads();
    if (tid == 0) {
        float s = 0.0f;
        for (int w = 0; w < NWARP; ++w) s += part[w];
        s_rs = rsqrtf(s / (float) N + l.eps);
        l.rs[t][c] = s_rs;
    }
    __syncthreads();
    const float rs = s_rs;
#pragma unroll
    for (int j = 0; j < KS; ++j)
        if (ok[j])
            *reinterpret_cast<float4*>(xn + tid * 4 + (k0 + j) * THREADS * 4) =
                make_float4(r[j].x * rs, r[j].y * rs, r[j].z * rs, r[j].w * rs);
}

// ---------------- bf16: gr_down_staged_kernel<T, true> for row group g (g == LR / 8: the inject rows) ----------------
__device__ __forceinline__ void pf_l2(const void* p) { asm volatile("prefetch.global.L2 [%0];" ::"l"(p)); }
template <int T>
__device__ __forceinline__ void stage_xn(const L& l, int h, float4* buf, int k0, int nk) {   // tokens [k0, k0 + nk)
    for (int i = threadIdx.x; i < nk * (H_TILE / 4); i += THREADS) {
        const int k = i / (H_TILE / 4), s4 = i - k * (H_TILE / 4);
        const float* src = l.xn + (size_t) (k0 + k) * D + (size_t) h * H_TILE + (size_t) s4 * 4;
        cp_async16(buf + (size_t) k * (H_TILE / 4) + (s4 & 1) * (H_TILE / 8) + (s4 >> 1), src);
    }
}
constexpr int DEPTH_BF = 4;   // tiles of a row's weights in flight in registers
// A down item: 8 rows (a warp each) for one half of the window's tokens (T >= 2), so two CTAs share a row group's
// weights (the second mostly from L2) and each stages and reads only its own tokens' activations: the 324 rows run on
// ~82 SMs with half the activation traffic each.  A token's sums are the same chain whichever CTA runs it.
template <int T> __host__ __device__ constexpr int dn_nh() { return T >= 2 ? 2 : 1; }
template <int T> __host__ __device__ constexpr int dn_th() { return (T + dn_nh<T>() - 1) / dn_nh<T>(); }
template <int T>
__device__ __forceinline__ void down_bf16(const L& l, int item, float4* hbuf) {
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    constexpr int NH = dn_nh<T>(), TH = dn_th<T>();
    const int g = item / NH, k0 = (item % NH) * TH, nk = T - k0 < TH ? T - k0 : TH;
    const int k = g * NWARP + warp;                      // row of [w_down; w_inject]
    const int nrows = LR + (l.w_inject != nullptr ? HC : 0);
    const bool active = k < nrows;
    const uint16_t* wrow = (k < LR || !active ? l.w_down + (size_t) (active ? k : 0) * D : l.w_inject + (size_t) (k - LR) * D);
    const uint4* w4 = reinterpret_cast<const uint4*>(wrow);
    constexpr size_t buf_f4 = (size_t) TH * (H_TILE / 4);
    float acc[TH];
#pragma unroll
    for (int q = 0; q < TH; ++q) acc[q] = 0.0f;
    uint4 wb[DEPTH_BF][HQ];
    constexpr int PRE = 2;   // tiles requested before the wait (more would queue ahead of the norm items' reads)
    if (active) {
#pragma unroll
        for (int h = 0; h < PRE; ++h)
#pragma unroll
            for (int q = 0; q < HQ; ++q) wb[h][q] = __ldg(w4 + h * (H_TILE / 8) + lane + 32 * q);
    }
    wait_ge(&l.sy->nrm.v, (unsigned) l.nN);
    if (active) {
#pragma unroll
        for (int h = PRE; h < DEPTH_BF; ++h)
#pragma unroll
            for (int q = 0; q < HQ; ++q) wb[h][q] = __ldg(w4 + h * (H_TILE / 8) + lane + 32 * q);
    }
    constexpr int NB = nb_bf<T>();
#pragma unroll
    for (int b = 0; b < NB - 1; ++b) {
        if (b < N_HTILES) stage_xn<T>(l, b, hbuf + b * buf_f4, k0, nk);
        cp_async_commit();
    }
#pragma unroll
    for (int h = 0; h < N_HTILES; ++h) {
        cp_async_wait_n<NB - 2>();
        __syncthreads();   // tile h landed for every thread; every warp is done with tile h - 1's buffer
        if (h + NB - 1 < N_HTILES) stage_xn<T>(l, h + NB - 1, hbuf + ((h + NB - 1) % NB) * buf_f4, k0, nk);
        cp_async_commit();
        const float4* cur = hbuf + (h % NB) * buf_f4;
        if (active) {
#pragma unroll
            for (int q = 0; q < HQ; ++q) {
                const int j = lane + 32 * q;
                const Bf16x8 wvq = unpack8(wb[h % DEPTH_BF][q]);
#pragma unroll
                for (int kk = 0; kk < TH; ++kk) {
                    if (kk < nk) {
                        const float4* pk = cur + (size_t) kk * (H_TILE / 4);
                        acc[kk] += dot8u(wvq, pk[j], pk[H_TILE / 8 + j]);
                    }
                }
            }
            if (h + DEPTH_BF < N_HTILES) {
#pragma unroll
                for (int q = 0; q < HQ; ++q) wb[h % DEPTH_BF][q] = __ldg(w4 + (h + DEPTH_BF) * (H_TILE / 8) + lane + 32 * q);
            }
        }
        if (g_trace != nullptr && t == 0) g_trace[2560 + (item % 96) * 16 + h] = gtimer();
    }
    if (active) {
        float sm[TH];
#pragma unroll
        for (int kk = 0; kk < TH; ++kk) sm[kk] = warp_sum(acc[kk]);
#pragma unroll
        for (int kk = 0; kk < TH; ++kk) {
            if (lane != kk || kk >= nk) continue;
            if (k >= LR) {
                l.inj_out[k0 + kk][k - LR] = sm[kk];
            } else {
                const float x = sm[kk] / (float) HC;
                l.lo[k0 + kk][k] = x / (1.0f + __expf(-x));
            }
        }
    }
}

// ---------------- bf16: gr_up_multi_kernel<T, true> for columns [16 u, 16 u + 16) ----------------
// The item's weights and its epilogue inputs (R, w_norm, bo: none of them written before the up items) are copied into
// shared memory once the norm items are done (so the copies do not queue ahead of the norm's own reads), then it waits
// for the down items.
constexpr int UPB_ROWS = HC * UPB;          // 64
constexpr int UPB_F4 = LR / 8;              // uint4 per row (40)
struct UpB { uint4* sw; float* sR; float* swn; float* sbo; float* sg; };
template <int T>
__device__ __forceinline__ UpB upb_smem(float4* smem4) {
    UpB s;
    s.sw = reinterpret_cast<uint4*>(smem4);
    s.sR = reinterpret_cast<float*>(s.sw + UPB_ROWS * UPB_F4);   // [T][HC][UPB]
    s.swn = s.sR + T * UPB_ROWS;                                 // [HC][UPB]
    s.sbo = s.swn + UPB_ROWS;                                    // [T][UPB]
    s.sg = s.sbo + T * UPB;                                      // [T][HC][UPB]
    return s;
}
constexpr int RU = 4;   // up rows a warp works on at once (independent chains interleaved)
constexpr int UP_GR = NWARP * RU;               // rows a step (32)
constexpr int UP_STEPS = UPB_ROWS / UP_GR;      // 4: the item's rows land and are used in four copy groups
static_assert(UPB_ROWS % UP_GR == 0, "whole row groups");
template <int G>
__device__ __forceinline__ void cp_wait_rt(int pending) {   // cp.async.wait_group with a run-time count (unrolled callers)
    if constexpr (G >= 0) {
        if (pending == G) cp_async_wait_n<G>();
        else cp_wait_rt<G - 1>(pending);
    }
}
template <int T>
__device__ __forceinline__ void up_bf16(const L& l, int u, const UpB& s) {
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int d0 = u * UPB;
    const uint64_t pol = pol_ef();
    wait_ge(&l.sy->nrm.v, (unsigned) l.nN);   // the copies after the norm items' reads, before the down items finish
#pragma unroll
    for (int st = 0; st < UP_STEPS; ++st) {
        for (int i = t; i < UP_GR * UPB_F4; i += THREADS) {
            const int r = st * UP_GR + i / UPB_F4, q = i % UPB_F4, c = r / UPB, dd = r - c * UPB;
            cp_async16_ef(s.sw + r * UPB_F4 + q, l.w_up + (size_t) (c * N + d0 + dd) * LR + (size_t) q * 8, pol);
        }
        if (st == 0) {
            for (int i = t; i < (T + 1) * HC * (UPB / 4) + T * (UPB / 4); i += THREADS) {   // R, w_norm, bo
                if (i < T * HC * (UPB / 4)) {
                    const int k = i / (HC * (UPB / 4)), jj = i - k * (HC * (UPB / 4)), c = jj / (UPB / 4), f = jj - c * (UPB / 4);
                    cp_async16(s.sR + (k * HC + c) * UPB + f * 4, l.R[k] + c * N + d0 + f * 4);
                } else if (i < (T + 1) * HC * (UPB / 4)) {
                    const int jj = i - T * HC * (UPB / 4), c = jj / (UPB / 4), f = jj - c * (UPB / 4);
                    cp_async16(s.swn + c * UPB + f * 4, l.w_norm + c * N + d0 + f * 4);
                } else if (l.apply) {
                    const int jj = i - (T + 1) * HC * (UPB / 4), k = jj / (UPB / 4), f = jj - k * (UPB / 4);
                    cp_async16(s.sbo + k * UPB + f * 4, l.bo[k] + d0 + f * 4);
                }
            }
        }
        cp_async_commit();
    }
    wait_ge(&l.sy->dn.v, (unsigned) l.nD);
    float4 la[T][2], lb[T][2];   // this lane's lo values (gr_up_multi_kernel's chunks lane and 32 + lane), in registers
#pragma unroll
    for (int k = 0; k < T; ++k) {
        const float4* p = reinterpret_cast<const float4*>(l.lo[k]);
        la[k][0] = __ldcg(p + lane * 2); la[k][1] = __ldcg(p + lane * 2 + 1);
        lb[k][0] = lb[k][1] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (lane < LR / 8 - 32) { lb[k][0] = __ldcg(p + (32 + lane) * 2); lb[k][1] = __ldcg(p + (32 + lane) * 2 + 1); }
    }
    __shared__ float s_rsu[MT * HC], s_gwu[MT * HC];   // rs and the pending write's gate input of every (token, stream)
    if (t < T * HC) {
        s_rsu[t] = __ldcg(l.rs[t / HC] + t % HC);
        s_gwu[t] = l.apply ? l.inj_prev[t / HC][t % HC] : 0.0f;
    }
    if (g_trace != nullptr && t == 0) g_trace[4000 + u * 4 + 0] = gtimer();
    constexpr int P = p2(RU * T);
#pragma unroll
    for (int st = 0; st < UP_STEPS; ++st) {
        cp_wait_rt<UP_STEPS - 1>(UP_STEPS - 1 - st);
        __syncthreads();
        if (g_trace != nullptr && t == 0 && st == 0) g_trace[4000 + u * 4 + 1] = gtimer();
        float v[P];
#pragma unroll
        for (int j = 0; j < RU; ++j) {
            const int r = st * UP_GR + j * NWARP + warp;
            const Bf16x8 ua = unpack8(s.sw[r * UPB_F4 + lane]);
            const Bf16x8 ub = unpack8(lane < LR / 8 - 32 ? s.sw[r * UPB_F4 + 32 + lane] : make_uint4(0, 0, 0, 0));
#pragma unroll
            for (int k = 0; k < T; ++k) {
                float acc = dot8u(ua, la[k][0], la[k][1]);
                if (lane < LR / 8 - 32) acc += dot8u(ub, lb[k][0], lb[k][1]);
                v[j * T + k] = acc;
            }
        }
#pragma unroll
        for (int q = RU * T; q < P; ++q) v[q] = 0.0f;
        const float mysum = tsum_x<P, false>(v, lane);   // the sum of value tsum_token<P>(lane)
        // the epilogue of (row, token) j on the one lane that owns sum j (gr_up_multi_kernel's lane k work, spread)
        const int jj = s26ts::tsum_token<P>(lane);
        if (jj < RU * T && lane == tsum_lane<P>(jj)) {
            const int k = jj % T, r = st * UP_GR + (jj / T) * NWARP + warp, c = r / UPB, dd = r - c * UPB;
            float rv = s.sR[(k * HC + c) * UPB + dd];
            const float wn = s.swn[c * UPB + dd];
            const float rsc = s_rsu[k * HC + c];
            if (l.apply) {
                const float bo = s.sbo[k * UPB + dd];
                rv = fmaf(bo, 2.0f * sigmoidf_(s_gwu[k * HC + c] / (float) HC), rv);
                l.R_out[k][c * N + d0 + dd] = rv;
            }
            const float x = rv * wn * rsc;
            s.sg[(k * HC + c) * UPB + dd] = x * sigmoidf_(mysum);
        }
        if (g_trace != nullptr && t == 0 && st == 0) g_trace[4000 + u * 4 + 3] = gtimer();
    }
    __syncthreads();
    if (g_trace != nullptr && t == 0) g_trace[4000 + u * 4 + 2] = gtimer();
    for (int i = t; i < T * UPB; i += THREADS) {
        const int k = i / UPB, col = i - k * UPB;
        float sum = 0.0f;
#pragma unroll
        for (int c = 0; c < HC; ++c) sum += s.sg[(k * HC + c) * UPB + col];
        l.mixed[k][d0 + col] = sum / (float) HC;
    }
}

// ---------------- hc8: hc8::norm_kernel for (stream c, token t) ----------------
__device__ __forceinline__ void norm_h8(const L& l, int c, int t) {
    __shared__ float red[32];
    constexpr int PER = N / THREADS;   // 10
    const float* r = l.R[t] + (size_t) c * N;
    float w = 0.0f;
    if (l.apply) w = 2.0f / (1.0f + expf(-l.inj_prev[t][c] / (float) HC));
    float v[PER];
#pragma unroll
    for (int j = 0; j < PER; ++j) v[j] = r[threadIdx.x + j * THREADS];   // the loads together, then the same chain
    if (l.apply) {
#pragma unroll
        for (int j = 0; j < PER; ++j) {
            const int i = threadIdx.x + j * THREADS;
            v[j] = fmaf(l.bo[t][i], w, v[j]);
            l.R_out[t][(size_t) c * N + i] = v[j];
        }
    }
    float ss = 0.0f;
#pragma unroll
    for (int j = 0; j < PER; ++j) ss = fmaf(v[j], v[j], ss);
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = ss;
    __syncthreads();
    if (threadIdx.x == 0) {
        float s = 0.0f;
        for (int k = 0; k < NWARP; ++k) s += red[k];
        l.rs[t][c] = rsqrtf(s / (float) N + l.eps);
    }
}

// ---------------- hc8: hc8::down_kernel's sums for rows [8 g, 8 g + 8) of [q_down; q_inject] ----------------
// hc8::down_kernel: thread v (0..159) of a 160-thread CTA takes groups v and v + 160 (32 columns each), a 32-thread
// xor tree per warp of five, then the five warp sums in order.  Here a warp is a row and lane l of column tile ti
// (1024 columns) is group 32 ti + l, i.e. thread v = 32 (ti % 5) + l on its (ti / 5)-th group: the same chains.
// The tile's activations sit in shared memory xor-swizzled by float4 (lane l's float4 j at l * 8 + (j ^ (l & 7))).
template <int T>
__device__ __forceinline__ void stage_x8(const L& l, int ti, float4* buf, int k0, int nk) {   // [nk + 1][T8 / 4]: x, w_norm
    for (int i = threadIdx.x; i < (nk + 1) * (T8 / 4); i += THREADS) {
        const int k = i / (T8 / 4), f = i - k * (T8 / 4), ln = f >> 3, j = f & 7;
        const float* src = (k < nk ? (l.apply ? l.R_out[k0 + k] : l.R[k0 + k]) : l.w_norm) + (size_t) ti * T8 + (size_t) f * 4;
        cp_async16(buf + (size_t) k * (T8 / 4) + ln * 8 + (j ^ (ln & 7)), src);
    }
}
template <int T>
__device__ __forceinline__ void down_h8(const L& l, int item, float4* xbuf) {
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    constexpr int NH = dn_nh<T>(), TH = dn_th<T>();
    const int g = item / NH, k0 = (item % NH) * TH, nk = T - k0 < TH ? T - k0 : TH;
    const int nrows = LR + (l.inj ? HC : 0);
    const int k = g * NWARP + warp;
    const bool active = k < nrows;
    const int8_t* q;
    const uint16_t* s;
    if (!active || k < LR) { const int kk = active ? k : 0; q = l.qd + (size_t) kk * D; s = l.sd + (size_t) kk * l.sd_row; }
    else { q = l.qi + (size_t) (k - LR) * D; s = l.si + (size_t) (k - LR) * l.si_row; }
    constexpr int NT8 = D / T8;   // 10 tiles
    // the lane's codes and scales of the whole row (10 groups: 320 bytes) in registers, requested before the wait
    uint4 qa[NT8], qb[NT8];
    uint32_t sc16[NT8];
    const uint64_t pol = pol_ef();
    if (active) {
#pragma unroll
        for (int ti = 0; ti < NT8; ++ti) {
            const int gi = ti * (T8 / 32) + lane;
            const uint4* p = reinterpret_cast<const uint4*>(q + (size_t) gi * 32);
            qa[ti] = ldg_ef(p, pol); qb[ti] = ldg_ef(p + 1, pol);
            sc16[ti] = ldg_ef_u16(s + gi, pol);
        }
    }
    wait_ge(&l.sy->nrm.v, (unsigned) l.nN);
    float acc[5][TH];
#pragma unroll
    for (int v = 0; v < 5; ++v)
#pragma unroll
        for (int t = 0; t < TH; ++t) acc[v][t] = 0.0f;
    constexpr int NB = nb_h8<T>();
    constexpr size_t bstride = (size_t) (TH + 1) * (T8 / 4);
#pragma unroll
    for (int b = 0; b < NB - 1; ++b) {
        if (b < NT8) stage_x8<T>(l, b, xbuf + b * bstride, k0, nk);
        cp_async_commit();
    }
    float rsv[TH][HC];   // rs of this warp's (token, stream)s, once
#pragma unroll
    for (int t = 0; t < TH; ++t)
#pragma unroll
        for (int c = 0; c < HC; ++c) rsv[t][c] = k0 + t < T ? __ldcg(l.rs[k0 + t] + c) : 0.0f;
#pragma unroll
    for (int ti = 0; ti < NT8; ++ti) {
        cp_async_wait_n<NB - 2>();
        __syncthreads();
        if (ti + NB - 1 < NT8) stage_x8<T>(l, ti + NB - 1, xbuf + ((ti + NB - 1) % NB) * bstride, k0, nk);
        cp_async_commit();
        const float4* cur = xbuf + (ti % NB) * bstride;
        if (active) {
            const int col0 = ti * T8 + lane * 32, c = col0 / N;
            float wq[32];
            {
                const float4* wn4 = cur + (size_t) nk * (T8 / 4) + lane * 8;
                const uint32_t w[8] = {qa[ti].x, qa[ti].y, qa[ti].z, qa[ti].w, qb[ti].x, qb[ti].y, qb[ti].z, qb[ti].w};
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const float4 v = wn4[j ^ (lane & 7)];
                    wq[4 * j] = hc8::s8(w[j], 0) * v.x;
                    wq[4 * j + 1] = hc8::s8(w[j], 1) * v.y;
                    wq[4 * j + 2] = hc8::s8(w[j], 2) * v.z;
                    wq[4 * j + 3] = hc8::s8(w[j], 3) * v.w;
                }
            }
            const float sc = hc8::f16f((uint16_t) sc16[ti]);
#pragma unroll
            for (int t = 0; t < TH; ++t) {
                if (t < nk) {
                    const float4* x4 = cur + (size_t) t * (T8 / 4) + lane * 8;
                    float d = 0.0f;
#pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        const float4 v = x4[j ^ (lane & 7)];
                        d = fmaf(wq[4 * j], v.x, d);
                        d = fmaf(wq[4 * j + 1], v.y, d);
                        d = fmaf(wq[4 * j + 2], v.z, d);
                        d = fmaf(wq[4 * j + 3], v.w, d);
                    }
                    float rs = rsv[t][0];
#pragma unroll
                    for (int cc = 1; cc < HC; ++cc) if (c == cc) rs = rsv[t][cc];
                    acc[ti % 5][t] = fmaf(sc * rs, d, acc[ti % 5][t]);
                }
            }
        }
        if (g_trace != nullptr && tid == 0) g_trace[2560 + (item % 96) * 16 + ti] = gtimer();
    }
    if (active) {
#pragma unroll
        for (int t = 0; t < TH; ++t) {
            if (k0 + t >= T) continue;
            float red[5];
#pragma unroll
            for (int vv = 0; vv < 5; ++vv) {
                float v = acc[vv][t];
#pragma unroll
                for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
                red[vv] = v;
            }
            if (lane == 0) {
                float v = 0.0f;
                for (int w = 0; w < 5; ++w) v += red[w];
                if (k < LR) {
                    const float x = v / (float) HC;
                    l.lo[k0 + t][k] = x / (1.0f + expf(-x));
                } else {
                    l.inj_out[k0 + t][k - LR] = v;
                }
            }
        }
    }
}

// ---------------- hc8: hc8::up_kernel for columns [16 u, 16 u + 16) (warp w: columns w, w + 8) ----------------
// As the bf16 up item: the codes, scales, R' and w_norm into shared memory once the norm items are done, then the wait.
constexpr int UP8_CODE_F4 = LR / 16;         // uint4 of codes per row (20)
constexpr int UP8_SROW = 16;                 // f16 scales per row (hc8_srow(320))
struct UpH { uint4* scode; uint16_t* ssc; float* sR; float* swn; };
template <int T>
__device__ __forceinline__ UpH uph_smem(float4* smem4) {
    UpH s;
    s.scode = reinterpret_cast<uint4*>(smem4);                               // [HC][UP8][LR] int8
    s.ssc = reinterpret_cast<uint16_t*>(s.scode + HC * UP8 * UP8_CODE_F4);  // [HC][UP8][16] f16
    s.sR = reinterpret_cast<float*>(s.ssc + HC * UP8 * UP8_SROW);           // [T][HC][UP8]
    s.swn = s.sR + T * HC * UP8;                                             // [HC][UP8]
    return s;
}
template <int T>
__device__ __forceinline__ void up_h8(const L& l, int u, const UpH& s) {
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int d0 = u * UP8;
    const uint64_t pol = pol_ef();
    wait_ge(&l.sy->nrm.v, (unsigned) l.nN);   // the copies after the norm items' reads, before the down items finish
    // four copy groups: step st holds columns [8 st, 8 st + 8) of every stream (codes, scales, R', w_norm)
    constexpr int CS = NWARP;                     // columns a step
    constexpr int NST = UP8 / CS;                 // 4
#pragma unroll
    for (int st = 0; st < NST; ++st) {
        constexpr int NC = HC * CS * UP8_CODE_F4, NS = HC * CS * UP8_SROW / 8, NR = T * HC * (CS / 4), NW = HC * (CS / 4);
        for (int i = tid; i < NC + NS + NR + NW; i += THREADS) {
            if (i < NC) {
                const int c = i / (CS * UP8_CODE_F4), f = i - c * (CS * UP8_CODE_F4);   // f: uint4 within the 8 rows
                const int row = st * CS + f / UP8_CODE_F4, q = f % UP8_CODE_F4;
                cp_async16_ef(s.scode + (c * UP8 + row) * UP8_CODE_F4 + q, l.qu + ((size_t) c * N + d0 + row) * LR + (size_t) q * 16, pol);
            } else if (i < NC + NS) {
                const int jj = i - NC, c = jj / (CS * UP8_SROW / 8), f = jj - c * (CS * UP8_SROW / 8);
                const int row = st * CS + f / (UP8_SROW / 8), q = f % (UP8_SROW / 8);
                cp_async16_ef(reinterpret_cast<uint4*>(s.ssc) + (c * UP8 + row) * (UP8_SROW / 8) + q,
                              l.su + ((size_t) c * N + d0 + row) * UP8_SROW + (size_t) q * 8, pol);
            } else if (i < NC + NS + NR) {
                const int jj = i - NC - NS, k = jj / (HC * (CS / 4)), j2 = jj - k * (HC * (CS / 4)), c = j2 / (CS / 4),
                          f = j2 - c * (CS / 4);
                const float* src = l.apply ? l.R_out[k] : l.R[k];
                cp_async16(s.sR + (k * HC + c) * UP8 + st * CS + f * 4, src + c * N + d0 + st * CS + f * 4);
            } else {
                const int jj = i - NC - NS - NR, c = jj / (CS / 4), f = jj - c * (CS / 4);
                cp_async16(s.swn + c * UP8 + st * CS + f * 4, l.w_norm + c * N + d0 + st * CS + f * 4);
            }
        }
        cp_async_commit();
    }
    wait_ge(&l.sy->dn.v, (unsigned) l.nD);
    constexpr int NJ = hc8::MAX_LR / 128;   // 4 (chunks ch = lane + 32 j < LR / 4)
    float4 lv[T][NJ];   // this lane's lo chunks, in registers
#pragma unroll
    for (int t = 0; t < T; ++t)
#pragma unroll
        for (int j = 0; j < NJ; ++j) {
            const int ch = lane + 32 * j;
            lv[t][j] = ch < LR / 4 ? __ldcg(reinterpret_cast<const float4*>(l.lo[t]) + ch) : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
    __shared__ float s_rsh[MT * HC];
    if (tid < T * HC) s_rsh[tid] = __ldcg(l.rs[tid / HC] + tid % HC);
    const int nch = LR / 4;
    const uint32_t* qc = reinterpret_cast<const uint32_t*>(s.scode);
    constexpr int P = p2(HC * T);
#pragma unroll
    for (int st = 0; st < NST; ++st) {
        cp_wait_rt<NST - 1>(NST - 1 - st);
        __syncthreads();
        const int dl = st * CS + warp;
        const int d = d0 + dl;
        float part[P];
#pragma unroll
        for (int q = 0; q < P; ++q) part[q] = 0.0f;
#pragma unroll
        for (int c = 0; c < HC; ++c) {
            const uint32_t* qr = qc + (size_t) (c * UP8 + dl) * (LR / 4);
            const uint16_t* sr = s.ssc + (size_t) (c * UP8 + dl) * UP8_SROW;
#pragma unroll
            for (int j = 0; j < NJ; ++j) {
                const int ch = lane + 32 * j;
                if (ch < nch) {
                    const uint32_t w = qr[ch];
                    const float sc = hc8::f16f(sr[ch / (hc8::G8 / 4)]);
                    const float q0 = hc8::s8(w, 0), q1 = hc8::s8(w, 1), q2 = hc8::s8(w, 2), q3 = hc8::s8(w, 3);
#pragma unroll
                    for (int t = 0; t < T; ++t) {
                        const float4 lx = lv[t][j];
                        float v = q0 * lx.x;
                        v = fmaf(q1, lx.y, v);
                        v = fmaf(q2, lx.z, v);
                        v = fmaf(q3, lx.w, v);
                        part[c * T + t] = fmaf(sc, v, part[c * T + t]);
                    }
                }
            }
        }
        const float mysum = tsum_x<P, false>(part, lane);
        // the owner of sum (c, t) computes xn and the gate, lane t chains the HC terms in stream order
        float xo = 0.0f, go = 0.0f;
        {
            const int jj = s26ts::tsum_token<P>(lane);
            if (jj < HC * T) {
                const int c = jj / T, tt = jj % T;
                xo = s.sR[(tt * HC + c) * UP8 + dl] * s.swn[c * UP8 + dl] * s_rsh[tt * HC + c];
                go = 1.0f / (1.0f + expf(-mysum));
            }
        }
        float xs[HC], gs[HC];
#pragma unroll
        for (int c = 0; c < HC; ++c) {
            const int src = tsum_lane<P>(c * T + (lane < T ? lane : 0));
            xs[c] = __shfl_sync(0xffffffffu, xo, src);
            gs[c] = __shfl_sync(0xffffffffu, go, src);
        }
        if (lane < T) {
            float m = 0.0f;
#pragma unroll
            for (int c = 0; c < HC; ++c) m = fmaf(xs[c], gs[c], m);
            l.mixed[lane][d] = m / (float) HC;
        }
    }
}

// ---------------- router: bf16_f32_mmvf_multi_kernel<256, T, true> for rows [8 r, 8 r + 8) ----------------
constexpr int RT_PAIRS = N / 2 / THREADS;   // 5 pairs a thread
static_assert(N / 2 == RT_PAIRS * THREADS, "the router GEMV's 256-thread block covers n_embd in whole rounds");
__device__ __forceinline__ void rt_issue(const L& l, int r, uint4* sw) {
    const uint4* src = reinterpret_cast<const uint4*>(l.w_rt + (size_t) r * RTR * N);
    const uint64_t pol = pol_ef();
    for (int i = threadIdx.x; i < RTR * N / 8; i += THREADS) cp_async16_ef(sw + i, src + i, pol);
    cp_async_commit();
}
template <int T>
__device__ __forceinline__ void router_rows(const L& l, int r, const uint4* sw, float* part /* [RTR][T][NWARP] */) {
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    wait_ge(&l.sy->up.v, (unsigned) l.nU);
    float2 x[T][RT_PAIRS];
#pragma unroll
    for (int k = 0; k < T; ++k)
#pragma unroll
        for (int q = 0; q < RT_PAIRS; ++q) x[k][q] = __ldcg(reinterpret_cast<const float2*>(l.mixed[k]) + tid + q * THREADS);
    cp_async_wait0();
    __syncthreads();
    const uint32_t* w32 = reinterpret_cast<const uint32_t*>(sw);
    constexpr int RC = 32 / p2(T) < RTR ? 32 / p2(T) : RTR;   // rows a transposed sum carries (RC x T <= 32 sums)
    constexpr int P = p2(RC * T);
#pragma unroll
    for (int rc = 0; rc < RTR; rc += RC) {
        float acc[P];
#pragma unroll
        for (int q = 0; q < P; ++q) acc[q] = 0.0f;
#pragma unroll
        for (int i = 0; i < RC; ++i) {
            const int row = rc + i;
#pragma unroll
            for (int q = 0; q < RT_PAIRS; ++q) {
                const uint32_t weight = w32[(size_t) row * (N / 2) + tid + q * THREADS];
                const float w0 = __uint_as_float(weight << 16), w1 = __uint_as_float(weight & 0xffff0000u);
#pragma unroll
                for (int k = 0; k < T; ++k) {
                    acc[i * T + k] = fz_fma(w0, x[k][q].x, acc[i * T + k]);
                    acc[i * T + k] = fz_fma(w1, x[k][q].y, acc[i * T + k]);
                }
            }
        }
        // the RC x T warp sums at once (bitwise the per-row xor butterflies); the owning lane stores its sum
        const float mysum = tsum_x<P, true>(acc, lane);
        const int j = s26ts::tsum_token<P>(lane);
        if (j < RC * T && lane == tsum_lane<P>(j)) part[(rc * T + j) * NWARP + warp] = mysum;   // [row][k][warp]
    }
    __syncthreads();
    {   // warp w finishes row w: the 256-thread block's second stage (lanes past the eight warps add zeros)
        const int row = warp;
        constexpr int PT = p2(T);
        float v[PT];
#pragma unroll
        for (int k = 0; k < PT; ++k) v[k] = (k < T && lane < NWARP) ? part[(row * T + k) * NWARP + lane] : 0.0f;
        const float mysum = tsum_x<PT, true>(v, lane);
        const int k = s26ts::tsum_token<PT>(lane);
        if (k < T && lane == tsum_lane<PT>(k)) l.logits[(size_t) k * NE + (size_t) r * RTR + row] = mysum;
    }
}

// ---------------- top-10: native_router.cu's route_multi, warp k = token k ----------------
__device__ __forceinline__ void top10_token(const float* logits, int32_t* ids, float* weights) {
    const int lane = threadIdx.x & 31;
    float values[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) values[i] = __ldcg(logits + lane + i * 32);
    float maximum = -INFINITY;
#pragma unroll
    for (int i = 0; i < 16; ++i) maximum = fz_max(maximum, values[i]);
    maximum = fz_warp_max(maximum);
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        values[i] = fz_exp(fz_sub(values[i], maximum));
        sum = fz_add(sum, values[i]);
    }
    const float reciprocal = fz_rcp(fz_warp_sum(sum));
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        values[i] = fz_mul(values[i], reciprocal);
        if (__isnanf(values[i])) values[i] = -FLT_MAX;
    }
    float selected = 0.0f, selected_sum = 0.0f;
    for (int rank = 0; rank < KT; ++rank) {
        float best = values[0];
        int expert = lane;
#pragma unroll
        for (int i = 1; i < 16; ++i) {
            if (values[i] > best) { best = values[i]; expert = lane + i * 32; }
        }
#pragma unroll
        for (int mask = 16; mask; mask >>= 1) {
            const float other = __shfl_xor_sync(0xffffffffu, best, mask, 32);
            const int other_id = __shfl_xor_sync(0xffffffffu, expert, mask, 32);
            if (other > best || (other == best && other_id < expert)) { best = other; expert = other_id; }
        }
        if ((expert & 31) == lane) {
            values[expert / 32] = -INFINITY;
            ids[rank] = expert;
            selected_sum = fz_add(selected_sum, best);
        }
        if (rank == lane) selected = best;
    }
    selected_sum = fz_max(fz_warp_sum(selected_sum), 6.103515625e-5f);
    const float inverse_selected_sum = fz_rcp(selected_sum);
    if (lane < KT) weights[lane] = fz_mul(selected, inverse_selected_sum);
}

// ---------------- the kernel ----------------
template <int T, bool H8>
__global__ void __launch_bounds__(THREADS) hcf_kernel(const __grid_constant__ L l) {
    extern __shared__ __align__(16) float4 smem4[];
    __shared__ int s_item;
    __shared__ int s_last;
    const int tid = threadIdx.x;
    Sync* sy = l.sy;
    const int eN = l.nN, eD = eN + l.nD, eU = eD + l.nU, eR = eU + l.nR;
    unsigned long long* const tr = g_trace;
    unsigned long long t_claim = 0;
    if (tid == 0) { s_item = (int) atomicAdd(&sy->q.v, 1u); if (tr) { t_claim = gtimer(); s_tw = 0; } }
    __syncthreads();
    int item = s_item;
    while (item < eR) {
        if (item < eN) {
            if constexpr (H8) norm_h8(l, item % HC, item / HC);
            else norm_bf16<T>(l, item / HC, item % HC);
            signal(&sy->nrm.v);
        } else if (item < eD) {
            if constexpr (H8) down_h8<T>(l, item - eN, smem4);
            else down_bf16<T>(l, item - eN, smem4);
            signal(&sy->dn.v);
        } else if (item < eU) {
            if constexpr (H8) up_h8<T>(l, item - eD, uph_smem<T>(smem4));
            else up_bf16<T>(l, item - eD, upb_smem<T>(smem4));
            signal(&sy->up.v);
        } else {
            const int r = item - eU;
            uint4* sw = reinterpret_cast<uint4*>(smem4);
            float* part = reinterpret_cast<float*>(sw + RTR * N / 8);
            rt_issue(l, r, sw);
            router_rows<T>(l, r, sw, part);
            // the last router item to finish runs the top-10 (one warp a token): no item, no wait
            __threadfence();
            __syncthreads();
            if (tid == 0) s_last = atomicAdd(&sy->rt.v, 1u) == (unsigned) (l.nR - 1);
            __syncthreads();
            if (s_last) {
                __threadfence();
                const int k = tid >> 5;
                if (k < T) top10_token(l.logits + (size_t) k * NE, l.ids + (size_t) k * KT, l.wts + (size_t) k * KT);
            }
        }
        __syncthreads();
        if (tid == 0 && tr != nullptr) {
            unsigned long long* e = tr + (size_t) item * 5;
            e[0] = (unsigned long long) item; e[1] = smid(); e[2] = t_claim; e[3] = s_tw; e[4] = gtimer();
            s_tw = 0;
        }
        if (tid == 0) { s_item = (int) atomicAdd(&sy->q.v, 1u); if (tr) t_claim = gtimer(); }
        __syncthreads();
        item = s_item;
    }
    // the counters back to zero once every CTA is through (the next read on this set starts after this kernel)
    if (tid == 0) {
        __threadfence();
        if (atomicAdd(&sy->ex.v, 1u) == gridDim.x - 1) {
            sy->q.v = 0; sy->nrm.v = 0; sy->dn.v = 0; sy->up.v = 0; sy->rt.v = 0; sy->ex.v = 0;
            __threadfence();
        }
    }
}

template <int T, bool H8>
size_t smem_bytes(bool router) {
    size_t s;
    if constexpr (H8) {
        const size_t dn = (size_t) nb_h8<T>() * (dn_th<T>() + 1) * T8 * 4;
        const size_t up = (size_t) HC * UP8 * LR + (size_t) HC * UP8 * UP8_SROW * 2 + ((size_t) T * HC * UP8 + (size_t) HC * UP8) * 4;
        s = dn > up ? dn : up;
    } else {
        const size_t dn = (size_t) nb_bf<T>() * dn_th<T>() * H_TILE * 4;
        const size_t up = (size_t) UPB_ROWS * LR * 2 +
                          ((size_t) T * UPB_ROWS + UPB_ROWS + (size_t) T * UPB + (size_t) T * UPB_ROWS) * 4;
        s = dn > up ? dn : up;
    }
    if (router) {
        const size_t rt = (size_t) RTR * N * 2 + (size_t) RTR * T * NWARP * 4;
        if (rt > s) s = rt;
    }
    return s;
}

int g_last_grid = 0;   // hcf_bench: the last launch's CTAs
inline void hcf_grid_note(int g) { g_last_grid = g; }
// one launch of the read; false when it cannot run on this device (the caller runs the unfused read)
template <int T, bool H8>
bool launch_t(const L& l, bool router, cudaStream_t st) {
    static int occ[64][2] = {};   // per device, without / with the router: CTAs per SM (0 = not decided, -1 = no)
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) return false;
    int& o = occ[dev][router ? 1 : 0];
    static int sms[64] = {};
    const size_t sm = smem_bytes<T, H8>(router);
    if (o == 0) {
        int optin = 0;
        cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
        cudaDeviceGetAttribute(&sms[dev], cudaDevAttrMultiProcessorCount, dev);
        int n = 0;
        if ((int) sm <= optin &&
            cudaFuncSetAttribute(hcf_kernel<T, H8>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem_bytes<T, H8>(true)) == cudaSuccess &&
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, hcf_kernel<T, H8>, THREADS, sm) == cudaSuccess && n > 0)
            o = n;
        else
            o = -1;
        cudaGetLastError();
        static const int cap = [] { const char* e = std::getenv("STRATA_HC_FUSED_CTAS"); return e ? std::atoi(e) : 0; }();
        if (o > 0 && cap > 0 && cap < o) o = cap;
    }
    if (o < 0) return false;
    const int items = l.nN + l.nD + l.nU + l.nR;
    int grid = o * sms[dev];
    if (grid > items) grid = items;
    hcf_kernel<T, H8><<<grid, THREADS, sm, st>>>(l);
    hcf_grid_note(grid);
    return true;
}
template <bool H8>
bool launch_any(int T, const L& l, bool router, cudaStream_t st) {
    switch (T) {
        case 1: return launch_t<1, H8>(l, router, st);
        case 2: return launch_t<2, H8>(l, router, st);
        case 3: return launch_t<3, H8>(l, router, st);
        case 4: return launch_t<4, H8>(l, router, st);
        case 5: return launch_t<5, H8>(l, router, st);
        case 6: return launch_t<6, H8>(l, router, st);
        case 7: return launch_t<7, H8>(l, router, st);
        default: return launch_t<8, H8>(l, router, st);
    }
}
}  // namespace hcf
#endif

}  // namespace

#if STRATA_GR_FAST_BUILD
int g_gr_fast = -1;   // fused_gr_set_fast (the bench); -1 = STRATA_GR_FAST, else the card's default
static bool gr_fast() {
    static const int env = [] { const char* v = std::getenv("STRATA_GR_FAST"); return v ? (std::atoi(v) != 0 ? 1 : 0) : -1; }();
    if (g_gr_fast >= 0) return g_gr_fast != 0;
    if (env >= 0) return env != 0;
#if defined(__HIPCC__)
    return true;
#else
    { const char* v = std::getenv("STRATA_SM70_TABLE"); return cur_dev_volta() && v != nullptr && std::atoi(v) != 0; }   // PR 1401: opt-in   // CUDA: on Volta (V100-SXM2: 50.9 -> 46.7 us a read at T 1, bitwise); elsewhere opt-in
#endif
}
#else
int g_gr_fast = -1;
#endif
void fused_gr_set_fast(int on) { g_gr_fast = on; }

bool fused_gr_read_multi(const FusedGrArgs* a, int n_tok, float* xn_scratch, void* stream, unsigned long long* stamp_buf,
                         int stamp_i0) {
    if (n_tok < 1 || n_tok > kFusedGrMaxT || xn_scratch == nullptr) {
        std::fprintf(stderr, "fused_gr_read_multi: invalid arguments\n");
        std::exit(1);
    }
    GrMulti m;
    for (int t = 0; t < n_tok; ++t) {
        m.a[t] = a[t];
        const FusedGrArgs& x = a[t];
        if (!x.R || !x.w_norm || !((x.w_down && x.w_up) || (x.q8_down && x.q8_up) || (x.hc8_qd && x.hc8_qu)) || !x.lo || !x.rs || !x.mixed || (x.w_inject && !x.inject_out) ||
            (x.apply && (!x.bo_prev || !x.inj_prev || !x.R_out)) || x.w_down != a[0].w_down || x.w_up != a[0].w_up ||
            x.w_inject != a[0].w_inject || x.w_norm != a[0].w_norm || x.q8_down != a[0].q8_down ||
            x.q8_up != a[0].q8_up || x.q8_inject != a[0].q8_inject) {
            std::fprintf(stderr, "fused_gr_read_multi: invalid arguments for token %d\n", t);
            std::exit(1);
        }
    }
    // S26 STRATA_QFUSE: every token needs its q8_1 destination and token 0 the counters; otherwise none is written
    bool q8 = a[0].q8_cnt != nullptr;
    for (int t = 0; t < n_tok; ++t) q8 = q8 && a[t].q8_mixed != nullptr;
    if (!q8)
        for (int t = 0; t < n_tok; ++t) m.a[t].q8_mixed = nullptr;
    m.xn = xn_scratch;
    m.T = n_tok;
    cudaStream_t st = (cudaStream_t) stream;
#if !defined(__HIPCC__)
    if (a[0].hc8_qd != nullptr) {   // STRATA_EXL3_HC8: the hc8 read (Basalt's, on the 8-bit copies)
        const bool wrote = hc8::launch(m, st, stamp_buf, stamp_i0);
        const cudaError_t e8 = cudaGetLastError();
        if (e8 != cudaSuccess) {
            std::fprintf(stderr, "fused_gr_read_multi hc8: %s\n", cudaGetErrorString(e8));
            std::exit(1);
        }
        return wrote;
    }
#endif
    if (a[0].q8_down != nullptr && a[0].q8_up != nullptr) {   // inject: Q8_0 copy, or the BF16 rows
        launch_q8(m, xn_scratch, st, stamp_buf, stamp_i0);   // S23 experiment: STRATA_HC_Q8=1
        const cudaError_t eq = cudaGetLastError();
        if (eq != cudaSuccess) {
            std::fprintf(stderr, "fused_gr_read_multi q8: %s\n", cudaGetErrorString(eq));
            std::exit(1);
        }
        return q8;
    }
    // STRATA_GR_V3=1: the two-kernel read above (another summation order - opt-in)
    static const bool v3 = [] { const char* v = std::getenv("STRATA_GR_V3"); return v != nullptr && std::atoi(v) != 0; }();
    static int split3[64] = {};   // per device: 0 = not decided yet, 1 / 2 = column halves S, -1 = does not fit
    int dev3 = 0;
    if (v3) {
        cudaGetDevice(&dev3);
        if (dev3 >= 0 && dev3 < 64 && split3[dev3] == 0) {
            int optin = 0;
            cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev3);
            const int limit = optin > 0 ? optin : 48 * 1024;
            const int need1 = (int) (kFusedGrMaxT * N * sizeof(float)), need6 = (int) (6 * N * sizeof(float)),
                      need5 = (int) (5 * N * sizeof(float)), need2 = need1 / 2;
            int split = -1;
            if (need1 <= limit &&
                cudaFuncSetAttribute(gr_down_v3_kernel<1, kFusedGrMaxT, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, need1) == cudaSuccess &&
                cudaFuncSetAttribute(gr_down_v3_kernel<1, 6, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, need6) == cudaSuccess &&
                cudaFuncSetAttribute(gr_down_v3_kernel<1, 5, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, need5) == cudaSuccess) {
                cudaFuncSetAttribute(gr_down_v3_kernel<1, 1, true>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
                cudaFuncSetAttribute(gr_down_v3_kernel<1, 2, true>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
                cudaFuncSetAttribute(gr_down_v3_kernel<1, 3, true>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
                cudaFuncSetAttribute(gr_down_v3_kernel<1, 4, true>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
                cudaFuncSetAttribute(gr_down_v3_kernel<1, 5, true>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
                cudaFuncSetAttribute(gr_down_v3_kernel<1, 6, true>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
                cudaFuncSetAttribute(gr_down_v3_kernel<1, 4, false>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
                cudaFuncSetAttribute(gr_down_v3_kernel<1, kFusedGrMaxT, false>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
                split = 1;
            } else if (need2 <= limit)
                // #375 (kenh0u): the S = 2 split (a 64 KB opt-in card: Turing) disagrees with itself in gr_parity
                // (graph replay vs direct call) - such a card keeps the default read until that split is fixed
                std::fprintf(stderr, "strata: STRATA_GR_V3=1 needs the two-half split on this card, which fails its "
                                     "checks (#375): the default read is used\n");
            cudaGetLastError();   // drop any error the attempts left behind
            split3[dev3] = split;
        }
    }
    const int split = v3 && dev3 >= 0 && dev3 < 64 && a[0].s_down == nullptr && a[0].s_up == nullptr ? split3[dev3] : -1;
    if (split > 0) {   // 2 kernels; scratch = partials + sums of squares
        float* part = xn_scratch;
        float* ssg = xn_scratch + (size_t) n_tok * HC * 2 * PR;   // room for S = 2
        const size_t sm = (size_t) n_tok * (N / split) * sizeof(float);
        if (stamp_buf) gpu_stamp(stamp_buf, stamp_i0, stream);
        if (split == 2) gr_down_v3_kernel<2><<<dim3(LR / WARPS + 1, HC * 2), THREADS, sm, st>>>(m, part, ssg);
        else if (n_tok == 1) gr_down_v3_kernel<1, 1, true><<<dim3(LR / WARPS + 1, HC), THREADS, sm, st>>>(m, part, ssg);
        else if (n_tok == 2) gr_down_v3_kernel<1, 2, true><<<dim3(LR / WARPS + 1, HC), THREADS, sm, st>>>(m, part, ssg);
        else if (n_tok == 3) gr_down_v3_kernel<1, 3, true><<<dim3(LR / WARPS + 1, HC), THREADS, sm, st>>>(m, part, ssg);
        else if (n_tok == 4) gr_down_v3_kernel<1, 4, true><<<dim3(LR / WARPS + 1, HC), THREADS, sm, st>>>(m, part, ssg);
        else if (n_tok == 5) gr_down_v3_kernel<1, 5, true><<<dim3(LR / WARPS + 1, HC), THREADS, sm, st>>>(m, part, ssg);
        else if (n_tok == 6) gr_down_v3_kernel<1, 6, true><<<dim3(LR / WARPS + 1, HC), THREADS, sm, st>>>(m, part, ssg);
        else gr_down_v3_kernel<1, kFusedGrMaxT, false><<<dim3(LR / WARPS + 1, HC), THREADS, sm, st>>>(m, part, ssg);
        if (stamp_buf) gpu_stamp(stamp_buf, stamp_i0 + 1, stream);
        if (split == 2) gr_up_v3_kernel<2><<<UPM_BLOCKS, THREADS, 0, st>>>(m, part, ssg);
        else if (n_tok == 1) gr_up_v3_kernel<1, 1, true><<<UPM_BLOCKS, THREADS, 0, st>>>(m, part, ssg);
        else if (n_tok == 2) gr_up_v3_kernel<1, 2, true><<<UPM_BLOCKS, THREADS, 0, st>>>(m, part, ssg);
        else if (n_tok == 3) gr_up_v3_kernel<1, 3, true><<<UPM_BLOCKS, THREADS, 0, st>>>(m, part, ssg);
        else if (n_tok == 4) gr_up_v3_kernel<1, 4, true><<<UPM_BLOCKS, THREADS, 0, st>>>(m, part, ssg);
        else if (n_tok == 5) gr_up_v3_kernel<1, 5, true><<<UPM_BLOCKS, THREADS, 0, st>>>(m, part, ssg);
        else if (n_tok == 6) gr_up_v3_kernel<1, 6, true><<<UPM_BLOCKS, THREADS, 0, st>>>(m, part, ssg);
        else gr_up_v3_kernel<1, kFusedGrMaxT, false><<<UPM_BLOCKS, THREADS, 0, st>>>(m, part, ssg);
        const cudaError_t e3 = cudaGetLastError();
        if (e3 != cudaSuccess) {
            std::fprintf(stderr, "fused_gr_read_multi v3: %s\n", cudaGetErrorString(e3));
            std::exit(1);
        }
        return false;   // the v3 read: no q8_1 (the caller quantizes)
    }
#if defined(STRATA_HIP_GFX906)
    // gfx906, opt-in STRATA_GR_SPLIT=1: the latency-hidden norm/up (STRATA_GR_FAST) with the K-split down (~3% faster
    // per verify window).  Its fixed-order finish sums the K slices in another order than the single-token
    // fused_gr_read, so the multi-token read is then no longer bitwise equal to T single-token calls (gr_parity's
    // contract); off by default for that reason.
    static const bool split_on = std::getenv("STRATA_GR_SPLIT") && std::string(std::getenv("STRATA_GR_SPLIT")) == "1";
    if (split_on && a[0].s_down == nullptr) {   // its K-split down reads bf16 rows (not STRATA_HC_REQ8)
        const bool fast = gr_fast();
        if (fast) gr_norm_fast_kernel<<<n_tok, THREADS, 0, st>>>(m);
        else gr_norm_multi_kernel<<<n_tok, THREADS, 0, st>>>(m);
        if (stamp_buf) gpu_stamp(stamp_buf, stamp_i0, stream);
        gr_down_split_kernel<<<dim3(DOWN_BLOCKS + 1, GS_S), GS_WAVES * 64, (size_t) n_tok * GS_SL * sizeof(float), st>>>(
            m);
        gr_down_finish_kernel<<<(unsigned) ((n_tok * GS_ROWS + 255) / 256), 256, 0, st>>>(m);
        if (stamp_buf) gpu_stamp(stamp_buf, stamp_i0 + 1, stream);
        if (fast) gr_up_fast_kernel<<<UPM_BLOCKS, THREADS, 0, st>>>(m);
        else gr_up_multi_kernel<<<UPM_BLOCKS, THREADS, 0, st>>>(m);
        const cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) {
            std::fprintf(stderr, "fused_gr_read_multi: %s\n", cudaGetErrorString(e));
            std::exit(1);
        }
        return false;   // the gfx906 STRATA_GR_SPLIT read: no q8_1 (the caller quantizes), like the v3 path above
    }
#endif
    // the default read (STRATA_GR_V3 unset): v1, or the bitwise-equal v2 / v3 this card's check accepted
    launch_multi(m, fused_gr_variant(), st, stamp_buf, stamp_i0);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "fused_gr_read_multi: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
    return q8;
}

void hcf_trace_set(void* dev_buf) {
#if !defined(__HIPCC__)
    unsigned long long* p = static_cast<unsigned long long*>(dev_buf);
    cudaMemcpyToSymbol(hcf::g_trace, &p, sizeof(p));
#else
    (void) dev_buf;
#endif
}
int hcf_last_grid() {
#if !defined(__HIPCC__)
    return hcf::g_last_grid;
#else
    return 0;
#endif
}
int hc_fused_mode() {
#if defined(__HIPCC__)
    return 0;
#else
    static const int m = [] { const char* e = std::getenv("STRATA_HC_FUSED"); return e != nullptr ? std::atoi(e) : 0; }();
    return m;
#endif
}

int fused_gr_read_fused(const FusedGrArgs* a, int n_tok, float* xn_scratch, void* ctr, const HcFusedRouter* rt,
                        void* stream, unsigned long long* stamp_buf, int stamp_i0) {
#if defined(__HIPCC__)
    (void) a; (void) n_tok; (void) xn_scratch; (void) ctr; (void) rt; (void) stream; (void) stamp_buf; (void) stamp_i0;
    return 0;
#else
    const int mode = hc_fused_mode();
    if (mode <= 0 || ctr == nullptr || n_tok < 1 || n_tok > kFusedGrMaxT) return 0;
    static int cc_ok[64] = {};   // per device: 0 unknown, 1 sm_80+, -1 older
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) return 0;
    if (cc_ok[dev] == 0) {
        int major = 0;
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
        cc_ok[dev] = major >= 8 ? 1 : -1;
    }
    if (cc_ok[dev] < 0) return 0;
    const FusedGrArgs& a0 = a[0];
    const bool h8 = a0.hc8_qd != nullptr;
    for (int t = 0; t < n_tok; ++t) {
        const FusedGrArgs& x = a[t];
        if (!x.R || !x.w_norm || !x.lo || !x.rs || !x.mixed || (x.apply && (!x.bo_prev || !x.inj_prev || !x.R_out)) ||
            x.R_out != x.R || x.apply != a0.apply || x.w_norm != a0.w_norm || x.w_down != a0.w_down ||
            x.w_up != a0.w_up || x.w_inject != a0.w_inject || x.hc8_qd != a0.hc8_qd || x.hc8_qu != a0.hc8_qu ||
            x.hc8_qi != a0.hc8_qi || x.eps != a0.eps)
            return 0;
    }
    hcf::L l{};
    if (h8) {
        const bool inj = a0.w_inject != nullptr || a0.hc8_qi != nullptr;
        if (a0.hc8_qu == nullptr || a0.hc8_sd == nullptr || a0.hc8_su == nullptr || a0.hc8_su_row != 16 ||
            a0.hc8_sd_row < D / 32 || (inj && (a0.hc8_qi == nullptr || a0.hc8_si == nullptr || a0.hc8_si_row < D / 32)))
            return 0;
        for (int t = 0; t < n_tok; ++t) if (inj && a[t].inject_out == nullptr) return 0;
        l.qd = a0.hc8_qd; l.sd = a0.hc8_sd; l.sd_row = a0.hc8_sd_row;
        l.qu = a0.hc8_qu; l.su = a0.hc8_su; l.su_row = a0.hc8_su_row;
        if (inj) { l.qi = a0.hc8_qi; l.si = a0.hc8_si; l.si_row = a0.hc8_si_row; }
        l.inj = inj ? 1 : 0;
        l.nN = HC * n_tok;
        l.nD = ((LR + (inj ? HC : 0) + 7) / 8) * (n_tok >= 2 ? 2 : 1);   // row groups x token halves
        l.nU = N / hcf::UP8;
    } else {
        // the unfused bf16 read this one equals: split's norm (variant >= split), not STRATA_GR_V3 / the Q8 reads /
        // STRATA_HC_REQ8 / the Volta fast kernels
        static const bool v3 = [] { const char* v = std::getenv("STRATA_GR_V3"); return v != nullptr && std::atoi(v) != 0; }();
        if (v3 || a0.w_down == nullptr || a0.w_up == nullptr || a0.q8_down != nullptr || a0.s_down != nullptr ||
            a0.s_up != nullptr || fused_gr_variant() < kHcSplit || norm_fast_on() || xn_scratch == nullptr)
            return 0;
#if STRATA_GR_FAST_BUILD
        if (gr_fast()) return 0;
#endif
        for (int t = 0; t < n_tok; ++t) if (a0.w_inject != nullptr && a[t].inject_out == nullptr) return 0;
        l.w_down = a0.w_down; l.w_up = a0.w_up; l.w_inject = a0.w_inject;
        l.xn = xn_scratch;
        l.inj = a0.w_inject != nullptr ? 1 : 0;
        l.nN = HC * n_tok;
        l.nD = ((LR + (l.inj ? HC : 0) + 7) / 8) * (n_tok >= 2 ? 2 : 1);   // row groups x token halves
        l.nU = N / hcf::UPB;
    }
    for (int t = 0; t < n_tok; ++t) {
        const FusedGrArgs& x = a[t];
        l.R[t] = x.R; l.R_out[t] = x.R_out; l.bo[t] = x.bo_prev; l.inj_prev[t] = x.inj_prev;
        l.lo[t] = x.lo; l.rs[t] = x.rs; l.inj_out[t] = x.inject_out; l.mixed[t] = x.mixed;
    }
    l.w_norm = a0.w_norm;
    l.apply = a0.apply ? 1 : 0;
    l.eps = a0.eps;
    // the router + top-10 items: only where the unfused path is bf16_f32_mmvf_multi_kernel + route_multi (2+ tokens,
    // not STRATA_MMVF_ROWS / STRATA_TSUM)
    static const bool rows = [] {
        const char* r = std::getenv("STRATA_MMVF_ROWS");
        const char* s = std::getenv("STRATA_TSUM");
        return (r && r[0] == '1') || (s && s[0] == '1');
    }();
    const bool router = mode != 2 && rt != nullptr && rt->w != nullptr && rt->logits != nullptr && rt->ids != nullptr &&
                        rt->weights != nullptr && n_tok >= 2 && !rows;
    if (router) {
        l.w_rt = rt->w; l.logits = rt->logits; l.ids = rt->ids; l.wts = rt->weights;
        l.nR = hcf::NE / hcf::RTR;
    }
    l.sy = static_cast<hcf::Sync*>(ctr);
    cudaStream_t st = (cudaStream_t) stream;
    const bool ran = h8 ? hcf::launch_any<true>(n_tok, l, router, st) : hcf::launch_any<false>(n_tok, l, router, st);
    if (!ran) return 0;
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "fused_gr_read_fused: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
    if (stamp_buf) {   // the profile: the whole read in the norm stage (down and up stamps follow at once)
        gpu_stamp(stamp_buf, stamp_i0, stream);
        gpu_stamp(stamp_buf, stamp_i0 + 1, stream);
    }
    return router ? 2 : 1;
#endif
}

// STRATA_HC_Q8 on a bf16 pack: one warp a Q8_0 block, ggml's reference rounding (d = amax / 127 as fp16, q = round(x * (1 / d)))
__global__ void __launch_bounds__(256) gr_quantize_q8_0_kernel(const uint16_t* __restrict__ w, uint8_t* __restrict__ out, int64_t nb) {
    const int64_t b = (int64_t) blockIdx.x * 8 + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (b >= nb) return;
    const float v = __uint_as_float((uint32_t) w[b * 32 + lane] << 16);
    float amax = fabsf(v);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    const float d = amax / 127.0f, id = d != 0.0f ? 1.0f / d : 0.0f;
    uint8_t* o = out + b * 34;
    if (lane == 0) {
        const unsigned short h = __half_as_ushort(__float2half_rn(d));
        o[0] = (uint8_t) (h & 255u);
        o[1] = (uint8_t) (h >> 8);
    }
    o[2 + lane] = (uint8_t) (int8_t) lrintf(roundf(v * id));
}

__global__ void __launch_bounds__(256) gr_dequant_q8_0_kernel(const uint8_t* __restrict__ q8, uint16_t* __restrict__ out, int64_t nb) {
    const int64_t b = (int64_t) blockIdx.x * 8 + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (b >= nb) return;
    const uint8_t* p = q8 + b * 34;
    const float d = __half2float(__ushort_as_half((unsigned short) (p[0] | (p[1] << 8))));
    out[b * 32 + lane] = bf16_from_f32(d * (float) (int8_t) p[2 + lane]);
}

void fused_gr_dequant_q8_0(const uint8_t* q8, uint16_t* bf16, int64_t n_blocks, void* stream) {
    gr_dequant_q8_0_kernel<<<(unsigned) ((n_blocks + 7) / 8), 256, 0, (cudaStream_t) stream>>>(q8, bf16, n_blocks);
}

void fused_gr_quantize_q8_0(const uint16_t* bf16, uint8_t* out, int64_t n_blocks, void* stream) {
    gr_quantize_q8_0_kernel<<<(unsigned) ((n_blocks + 7) / 8), 256, 0, (cudaStream_t) stream>>>(bf16, out, n_blocks);
}

void hc8_quantize(const uint16_t* w, int64_t rows, int64_t cols, void* out, const int8_t** q, const uint16_t** s,
                  void* stream, float* s32) {
    if (w == nullptr || out == nullptr || rows < 1 || cols < 32 || cols % 32 != 0) {
        std::fprintf(stderr, "hc8_quantize: invalid arguments (%lld x %lld)\n", (long long) rows, (long long) cols);
        std::exit(1);
    }
    int8_t* qp = (int8_t*) out;
    uint16_t* sp = (uint16_t*) ((uint8_t*) out + (((size_t) rows * (size_t) cols + 15) & ~(size_t) 15));
    const int srow = hc8_srow(cols);
    cudaMemsetAsync(sp, 0, (size_t) rows * srow * 2, (cudaStream_t) stream);
#if !defined(__HIPCC__)
    const int64_t groups = rows * cols / 32;
    hc8::quant_kernel<<<(unsigned) ((groups + 255) / 256), 256, 0, (cudaStream_t) stream>>>(w, groups, (int) (cols / 32),
                                                                                           srow, qp, sp, s32);
#else
    std::fprintf(stderr, "hc8_quantize: not built for HIP\n");
    std::exit(1);
#endif
    if (q) *q = qp;
    if (s) *s = sp;
}

bool fused_gr_supported(int64_t n_embd, int64_t hc, int64_t hc_lr) {
    return n_embd == N && hc == HC && hc_lr == LR;
}

void fused_gr_read(const FusedGrArgs& a, void* stream) {
    if (!a.R || !a.w_norm || !a.w_down || !a.w_up || !a.lo || !a.rs || !a.mixed ||
        (a.w_inject && !a.inject_out) || (a.apply && (!a.bo_prev || !a.inj_prev || !a.R_out)) ||
        (a.apply && a.inj_prev == a.inject_out)) {
        std::fprintf(stderr, "fused_gr_read: invalid arguments\n");
        std::exit(1);
    }
    cudaStream_t st = (cudaStream_t) stream;
    if (a.s_down != nullptr) {   // STRATA_HC_REQ8
        gr_down_kernel<true><<<DOWN_BLOCKS + 1, THREADS, 0, st>>>(a);
        gr_up_kernel<true><<<UP_BLOCKS, THREADS, 0, st>>>(a);
    } else {
        gr_down_kernel<<<DOWN_BLOCKS + 1, THREADS, 0, st>>>(a);
        gr_up_kernel<<<UP_BLOCKS, THREADS, 0, st>>>(a);
    }
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "fused_gr_read: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
}


namespace {

/// The plain read, split, staged and (for one token) the single-token read on random bf16 weights and random inputs,
/// 1..8 tokens, with and without the pending write; every output of split and staged compared with the plain read's
/// bit for bit (and the plain read's with the single-token read's).  `why[v]` gets the first difference of variant
/// v; false if the check itself could not run.
bool fused_gr_selftest(bool ok_variant[4], std::string why[4]) {
    constexpr int TM = kFusedGrMaxT;
    constexpr int NV = 4;                             // sets: 0 = plain, 1 = split, 2 = staged, 3 = single-token
    std::mt19937 rng(20260930u);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<uint16_t> h_down((size_t) LR * D), h_up((size_t) D * LR), h_inj((size_t) HC * D);
    std::vector<float> h_norm(D), h_R((size_t) TM * D), h_bo((size_t) TM * N), h_ip((size_t) TM * HC);
    for (auto& w : h_down) w = bf16_from_f32(0.02f * nd(rng));
    for (auto& w : h_up) w = bf16_from_f32(0.05f * nd(rng));
    for (auto& w : h_inj) w = bf16_from_f32(0.02f * nd(rng));
    for (auto& w : h_norm) w = 1.0f + 0.1f * nd(rng);
    for (auto& x : h_R) x = nd(rng);
    for (auto& x : h_bo) x = 0.5f * nd(rng);
    for (auto& x : h_ip) x = 2.0f * nd(rng);
    for (int v = 0; v < 4; ++v) { ok_variant[v] = v == 1; why[v].clear(); }

    const size_t n_set = (size_t) TM * (D + D + LR + HC + HC + N);   // R_out, xn, lo, rs, inject, mixed (floats)
    const size_t bytes = h_down.size() * 2 + h_up.size() * 2 + h_inj.size() * 2 +
                         (h_norm.size() + h_R.size() + h_bo.size() + h_ip.size() + NV * n_set) * 4 + 32 * 256;
    uint8_t* base = nullptr;
    if (cudaMalloc((void**) &base, bytes) != cudaSuccess) {
        cudaGetLastError();
        why[0] = "no room for the check (" + std::to_string(bytes >> 20) + " MiB)";
        return false;
    }
    size_t off = 0;
    auto take = [&](size_t b) { void* p = base + off; off += (b + 255) / 256 * 256; return p; };
    uint16_t* d_down = (uint16_t*) take(h_down.size() * 2);
    uint16_t* d_up = (uint16_t*) take(h_up.size() * 2);
    uint16_t* d_inj = (uint16_t*) take(h_inj.size() * 2);
    float* d_norm = (float*) take(h_norm.size() * 4);
    float* d_R = (float*) take(h_R.size() * 4);
    float* d_bo = (float*) take(h_bo.size() * 4);
    float* d_ip = (float*) take(h_ip.size() * 4);
    struct Set { float *R_out, *xn, *lo, *rs, *inj, *mixed; } set[NV];
    for (Set& x : set) {
        x.R_out = (float*) take((size_t) TM * D * 4); x.xn = (float*) take((size_t) TM * D * 4);
        x.lo = (float*) take((size_t) TM * LR * 4); x.rs = (float*) take((size_t) TM * HC * 4);
        x.inj = (float*) take((size_t) TM * HC * 4); x.mixed = (float*) take((size_t) TM * N * 4);
    }
    cudaStream_t st = nullptr;
    bool ok = off <= bytes && cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking) == cudaSuccess &&
              cudaMemcpyAsync(d_down, h_down.data(), h_down.size() * 2, cudaMemcpyHostToDevice, st) == cudaSuccess &&
              cudaMemcpyAsync(d_up, h_up.data(), h_up.size() * 2, cudaMemcpyHostToDevice, st) == cudaSuccess &&
              cudaMemcpyAsync(d_inj, h_inj.data(), h_inj.size() * 2, cudaMemcpyHostToDevice, st) == cudaSuccess &&
              cudaMemcpyAsync(d_norm, h_norm.data(), h_norm.size() * 4, cudaMemcpyHostToDevice, st) == cudaSuccess &&
              cudaMemcpyAsync(d_R, h_R.data(), h_R.size() * 4, cudaMemcpyHostToDevice, st) == cudaSuccess &&
              cudaMemcpyAsync(d_bo, h_bo.data(), h_bo.size() * 4, cudaMemcpyHostToDevice, st) == cudaSuccess &&
              cudaMemcpyAsync(d_ip, h_ip.data(), h_ip.size() * 4, cudaMemcpyHostToDevice, st) == cudaSuccess;
    if (!ok) why[0] = "setting up the check failed";
    std::vector<float> h1, h2;
    // true when equal; false with the first difference in `w` (or a read-back failure in `ok`)
    auto same = [&](const float* d1, const float* d2, size_t n, const char* what, int T, int apply, std::string& w) {
        h1.resize(n);
        h2.resize(n);
        if (cudaMemcpyAsync(h1.data(), d1, n * 4, cudaMemcpyDeviceToHost, st) != cudaSuccess ||
            cudaMemcpyAsync(h2.data(), d2, n * 4, cudaMemcpyDeviceToHost, st) != cudaSuccess ||
            cudaStreamSynchronize(st) != cudaSuccess) {
            why[0] = "reading back the check failed";
            ok = false;
            return false;
        }
        for (size_t i = 0; i < n; ++i) {
            uint32_t u1, u2;
            std::memcpy(&u1, &h1[i], 4);
            std::memcpy(&u2, &h2[i], 4);
            if (u1 != u2) {
                char buf[192];
                std::snprintf(buf, sizeof buf, "%s differs for %d token(s)%s at %zu (%08x, not %08x)", what, T,
                              apply ? " with the pending write" : "", i, u2, u1);
                w = buf;
                return false;
            }
        }
        return true;
    };
    auto all_same = [&](const Set& s1, const Set& s2, int T, int apply, std::string& w) {
        return same(s1.lo, s2.lo, (size_t) T * LR, "lo", T, apply, w) &&
               same(s1.rs, s2.rs, (size_t) T * HC, "rs", T, apply, w) &&
               same(s1.inj, s2.inj, (size_t) T * HC, "inject", T, apply, w) &&
               same(s1.mixed, s2.mixed, (size_t) T * N, "mixed", T, apply, w) &&
               same(s1.R_out, s2.R_out, (size_t) T * D, "R", T, apply, w);
    };
    ok_variant[2] = ok_variant[3] = ok;
    bool single_ok = ok;
    for (int apply = 0; apply < 2 && ok; ++apply) {
        for (int T = 1; T <= TM && ok; ++T) {
            FusedGrArgs a[NV][TM];
            for (int v = 0; v < NV; ++v) {
                const size_t span = (size_t) ((uint8_t*) (set[v].mixed + (size_t) TM * N) - (uint8_t*) set[v].R_out);
                ok = ok && cudaMemsetAsync(set[v].R_out, 0xFF, span, st) == cudaSuccess;   // the whole set
                for (int k = 0; k < T; ++k) {
                    FusedGrArgs& x = a[v][k];
                    x.R = d_R + (size_t) k * D; x.R_out = set[v].R_out + (size_t) k * D; x.apply = apply != 0;
                    x.bo_prev = d_bo + (size_t) k * N; x.inj_prev = d_ip + (size_t) k * HC;
                    x.w_norm = d_norm; x.w_down = d_down; x.w_up = d_up; x.w_inject = d_inj; x.eps = 1e-6f;
                    x.lo = set[v].lo + (size_t) k * LR; x.rs = set[v].rs + (size_t) k * HC;
                    x.inject_out = set[v].inj + (size_t) k * HC; x.mixed = set[v].mixed + (size_t) k * N;
                }
            }
            for (int v = 0; v < 3; ++v) {
                GrMulti m;
                for (int k = 0; k < T; ++k) m.a[k] = a[v][k];
                m.xn = set[v].xn;
                m.T = T;
                launch_multi(m, v + 1, st, nullptr, 0);
            }
            if (T == 1) fused_gr_read(a[3][0], st);
            if (cudaGetLastError() != cudaSuccess || cudaStreamSynchronize(st) != cudaSuccess) {
                why[0] = "a kernel of the check failed";
                ok = false;
                break;
            }
            if (ok_variant[2] && !all_same(set[0], set[1], T, apply, why[2])) ok_variant[2] = false;
            if (ok && ok_variant[3] && !all_same(set[0], set[2], T, apply, why[3])) ok_variant[3] = false;
            if (ok && T == 1 && single_ok && !all_same(set[3], set[0], T, apply, why[1])) single_ok = false;
        }
    }
    if (!single_ok && ok) {                            // the plain read itself disagrees with the single-token read
        why[2] = why[3] = "the plain read differs from the single-token read: " + why[1];
        ok_variant[2] = ok_variant[3] = false;
    }
    if (st != nullptr) {
        cudaStreamSynchronize(st);
        cudaStreamDestroy(st);
    }
    cudaFree(base);
    cudaGetLastError();
    if (!ok) ok_variant[2] = ok_variant[3] = false;
    return ok;
}

}  // namespace

int fused_gr_variant() {
    int dev = 0;
    cudaGetDevice(&dev);
    const int v = dev >= 0 && dev < 64 ? g_variant[dev].load() : 0;
    if (v > 0) return v;
    // not checked on this card: the plain read, unless STRATA_HC_SPLIT names a variant (a test such as gr_parity)
    const char* e = std::getenv("STRATA_HC_SPLIT");
    return e != nullptr && (e[0] == '1' || e[0] == '2') ? env_variant() : kHcPlain;
}

void fused_gr_check() {
    int dev = 0;
    cudaGetDevice(&dev);
    if (dev < 0 || dev >= 64 || g_variant[dev].load() > 0) return;
    const int want = env_variant();
    if (want == kHcPlain) {
        g_variant[dev].store(kHcPlain);
        std::fprintf(stderr, "strata hc: CUDA%d: the hyper-connection read runs as the plain one (STRATA_HC_SPLIT=0)\n",
                     dev);
        return;
    }
    bool okv[4];
    std::string why[4];
    const bool ran = fused_gr_selftest(okv, why);
    int use = kHcPlain;
    if (want >= kHcStaged && okv[kHcStaged]) use = kHcStaged;
    else if (okv[kHcSplit]) use = kHcSplit;
    g_variant[dev].store(use);
    if (!ran)
        std::fprintf(stderr, "strata hc: CUDA%d: the check of split/staged could not run (%s)\n", dev, why[0].c_str());
    static const char* const name[4] = {"", "plain", "split", "staged"};
    for (int v = kHcStaged; v >= kHcSplit; --v)
        if (v <= want && !okv[v] && ran)
            std::fprintf(stderr, "strata hc: CUDA%d: the %s read differs from the plain read on this card - not used: "
                                 "%s\n", dev, name[v], why[v].c_str());
    static const char* const what[4] = {
        "", "the plain read (the norm per token, the down projection on 41 blocks)",
        "split (the norm per token and stream, then the plain read's down projection)",
        "staged (the norm per token and stream, the down projection's activations staged ahead by cp.async)"};
    std::fprintf(stderr, "strata hc: CUDA%d: the hyper-connection read runs as %s%s\n", dev, what[use],
                 use >= kHcSplit ? "; checked bit for bit against the plain read on this card (STRATA_HC_SPLIT=1 or 0 "
                                   "for the earlier ones)" : "");
}


namespace {
__global__ void hc_q8_bf16_kernel(const int8_t* __restrict__ q, const float* __restrict__ sc, uint16_t* __restrict__ out,
                                  long long n) {
    const long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = sc[i / 32] * (float) q[i];
    uint32_t b = __float_as_uint(v);
    b += 0x7fffu + ((b >> 16) & 1u);                  // round to nearest even (the values are finite)
    out[i] = (uint16_t) (b >> 16);
}
}  // namespace

void hc_q8_to_bf16(const void* codes, const float* scales, uint16_t* out, long long n, void* stream) {
    hc_q8_bf16_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, (cudaStream_t) stream>>>((const int8_t*) codes, scales, out, n);
}

// ===================================== the read as ONE persistent kernel (opt-in STRATA_HC_PERSIST=1, HIP) ==============
// 80 blocks that must all be resident (a start-up run per T decides; larger T keep the default read) and two grid
// barriers on an atomic counter.  Block b: (A) stages R' * w_norm for columns [128 b, 128 b + 128) (one stream) and writes
// that slice's sum of squares and the UNSCALED partial dots of the 320 down + 4 inject rows (the v3 identity
// w . xn = sum_c rs[c] (w[:, c] . x'[c])); barrier; (B) rs per stream, then rows (k, r) = b + 80 j reduced to lo / inject;
// barrier; (C) the up projection read straight from VRAM, a thread pair per row, the epilogue as gr_up_kernel.
// ANOTHER SUMMATION ORDER than the default read (not bitwise; the init check bounds the difference).  Only the verifier's hyper-connection reads use it: they run on one stream per card (both pipeline
// verifiers share it), so two of these kernels never wait on each other; the drafter's reads keep the default kernels.
#if defined(__HIPCC__)
namespace {
namespace hcp {
// Measured on gfx1030 (bench/results/2026-10-12-hc-persist-gfx1030): the gain over the three-kernel read is (1) the down
// projection split by column slices over all 80 blocks (the default read runs 41 blocks staging the activations through
// LDS) and (2) one launch with two grid barriers instead of three launches and the gaps between them.  Prefetching the
// up weights (into registers or by touching them into L2 / Infinity Cache) and the epilogue inputs measured no gain on
// gfx1030 and is gone: the 13 MB are read once from VRAM whichever way.
constexpr int NB = 80, SL = D / NB, SPS = N / SL, UPC = N / NB, NR = LR + HC;
static_assert(UPC * HC * 2 == THREADS, "a thread pair per up row");
struct PArgs {
    FusedGrArgs a[kFusedGrMaxT];
    float* part;        // [T][NR][NB]
    float* ssp;         // [T][NB]
    unsigned* bar;      // [0] count, [32] generation, [48] abort (a barrier that timed out)
    unsigned long long timeout_ticks;   // s_memrealtime ticks (100 MHz) a barrier waits before it aborts the read
    double* partd;      // STRATA_HC_PERSIST=3 (a reference only): the partial dots and sums of squares in double
    double* sspd;
};
// Returns false when the read was aborted: a block waited longer than `ticks` (the 80 blocks were not all resident - the
// start-up check finds that out by running every T with a short timeout) or another block aborted.  Then every block
// leaves the kernel; the outputs are not written completely and bar[48] stays set until the host clears the words.
__device__ __forceinline__ bool grid_sync(unsigned* bar, unsigned long long ticks) {
#if defined(__GFX11__) || defined(__GFX12__)
    // gfx11 and gfx12 have no s_memrealtime (clang: "needs target feature s-memrealtime"), so no barrier clock: this read is not
    // offered there (fused_gr_persist_init refuses it; an R9700 measured it neutral to 1% slower in #2226), and a launch that gets
    // here anyway aborts at once instead of waiting for a barrier it cannot time.
    (void) ticks;
    if (threadIdx.x == 0) __hip_atomic_store(&bar[48], 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    return false;
#else
    __shared__ int s_abort;
    __syncthreads();
    if (threadIdx.x == 0) {
        int ab = 0;
        const unsigned gen = __hip_atomic_load(&bar[32], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        __threadfence();
        const unsigned arrived = __hip_atomic_fetch_add(&bar[0], 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
        if (arrived == NB - 1) {
            __hip_atomic_store(&bar[0], 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
            __hip_atomic_fetch_add(&bar[32], 1u, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
        } else {
            const unsigned long long t0 = __builtin_amdgcn_s_memrealtime();
            while (__hip_atomic_load(&bar[32], __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT) == gen) {
                if (__hip_atomic_load(&bar[48], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) != 0u) { ab = 1; break; }
                if (__builtin_amdgcn_s_memrealtime() - t0 > ticks) {
                    __hip_atomic_store(&bar[48], 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
                    ab = 1;
                    break;
                }
                __builtin_amdgcn_s_sleep(1);
            }
        }
        if (__hip_atomic_load(&bar[48], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) != 0u) ab = 1;
        s_abort = ab;
    }
    __syncthreads();
    __threadfence();
    return s_abort == 0;
#endif
}
// STRATA_HC_PERSIST=3 (a reference only, slow): every dot and sum accumulated in double
__device__ __forceinline__ double dot8d(const uint4 w, const float* x) {
    double acc = 0.0;
    const unsigned u[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        acc += (double) __uint_as_float(u[q] << 16) * (double) x[2 * q];
        acc += (double) __uint_as_float(u[q] & 0xffff0000u) * (double) x[2 * q + 1];
    }
    return acc;
}
template <int T, bool DP = false>
__global__ void __launch_bounds__(THREADS) persist_kernel(PArgs f) {
    using A = typename std::conditional<DP, double, float>::type;
    __shared__ __align__(16) float xs[T][SL];
    __shared__ __align__(16) float lo[T][LR];
    __shared__ float s_rs[T][HC];
    __shared__ float g[T][HC][UPC];
    const int tid = threadIdx.x, lane = tid & 31, b = blockIdx.x;
    const FusedGrArgs& a0 = f.a[0];
    // (A) block b: columns [128 b, 128 b + 128) of D (one stream): R' * w_norm into LDS, the slice's sum of squares, and
    // the UNSCALED partial dots of the 320 down + 4 inject rows (w . xn = sum_c rs[c] (w[:, c] . x'[c]))
    const int c_b = b / SPS, col0 = b * SL;
    if (tid < T * (SL / 4)) {
        const int k = tid / (SL / 4), o4 = tid - k * (SL / 4);
        const FusedGrArgs& a = f.a[k];
        const int i = col0 + o4 * 4, d = i - c_b * N;
        float4 r = *reinterpret_cast<const float4*>(a.R + i);
        if (a.apply) {
            const float gw = 2.0f * sigmoidf_(a.inj_prev[c_b] / (float) HC);
            const float4 bo = *reinterpret_cast<const float4*>(a.bo_prev + d);
            r.x = fmaf(bo.x, gw, r.x); r.y = fmaf(bo.y, gw, r.y); r.z = fmaf(bo.z, gw, r.z); r.w = fmaf(bo.w, gw, r.w);
        }
        const float4 gn = *reinterpret_cast<const float4*>(a.w_norm + i);
        if constexpr (DP) {
            double ss = (double) r.x * r.x + (double) r.y * r.y + (double) r.z * r.z + (double) r.w * r.w;
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor(ss, o);
            if (lane == 0) f.sspd[k * NB + b] = ss;
        } else {
            const float ss = warp_sum(r.x * r.x + r.y * r.y + r.z * r.z + r.w * r.w);   // warp k = token k
            if (lane == 0) f.ssp[k * NB + b] = ss;
        }
        *reinterpret_cast<float4*>(&xs[k][o4 * 4]) = make_float4(r.x * gn.x, r.y * gn.y, r.z * gn.z, r.w * gn.w);
    }
    __syncthreads();
    const int hw = tid >> 4, hl = tid & 15;
    const bool inj = a0.w_inject != nullptr;
    const int nrows = inj ? NR : LR;
#pragma unroll 3
    for (int r = hw; r < nrows; r += 16) {
        const uint16_t* wrow = r < LR ? a0.w_down + (size_t) r * D : a0.w_inject + (size_t) (r - LR) * D;
        const uint4 w = __ldg(reinterpret_cast<const uint4*>(wrow + col0) + hl);
        A acc[T];
#pragma unroll
        for (int k = 0; k < T; ++k) {
            A v;
            if constexpr (DP) v = dot8d(w, &xs[k][hl * 8]); else v = dot8(w, &xs[k][hl * 8]);
#pragma unroll
            for (int o = 8; o > 0; o >>= 1) v += __shfl_xor(v, o);
            acc[k] = v;
        }
#pragma unroll
        for (int k = 0; k < T; ++k)
            if (hl == k) {
                if constexpr (DP) f.partd[((size_t) k * NR + r) * NB + b] = acc[k];
                else f.part[((size_t) k * NR + r) * NB + b] = (float) acc[k];
            }
    }
    if (!grid_sync(f.bar, f.timeout_ticks)) return;
    // (B) rs per stream (every block), then rows (k, r) = b + 80 j reduced to lo / inject by the blocks in turn
    if (tid < T * HC) {
        const int k = tid / HC, c = tid - k * HC;
        float rr;
        if constexpr (DP) {
            double sd = 0.0;
            for (int j = 0; j < SPS; ++j) sd += f.sspd[k * NB + c * SPS + j];
            rr = (float) (1.0 / sqrt(sd / (double) N + (double) f.a[k].eps));
        } else {
            float s = 0.0f;
            for (int j = 0; j < SPS; ++j) s += f.ssp[k * NB + c * SPS + j];
            rr = rsqrtf(s / (float) N + f.a[k].eps);
        }
        s_rs[k][c] = rr;
        if (b == 0) f.a[k].rs[c] = rr;
    }
    __syncthreads();
    for (int idx = b + NB * tid; idx < T * nrows; idx += NB * THREADS) {
        const int k = idx / nrows, r = idx - k * nrows;
        A sum = 0;
        if constexpr (DP) {
            const double* p1 = f.partd + ((size_t) k * NR + r) * NB;
            for (int c = 0; c < HC; ++c) {
                double p = 0.0;
                for (int j = 0; j < SPS; ++j) p += p1[c * SPS + j];
                sum += (double) s_rs[k][c] * p;
            }
        } else {
            const float4* p4 = reinterpret_cast<const float4*>(f.part + ((size_t) k * NR + r) * NB);
#pragma unroll
            for (int c = 0; c < HC; ++c) {
                float p = 0.0f;
#pragma unroll
                for (int j = 0; j < SPS / 4; ++j) {
                    const float4 v = p4[c * (SPS / 4) + j];
                    p += (v.x + v.y) + (v.z + v.w);
                }
                sum = fmaf(s_rs[k][c], p, sum);
            }
        }
        if (r < LR) {
            const A x = sum / (A) HC;
            f.a[k].lo[r] = (float) (x / ((A) 1 + (DP ? (A) exp((double) -x) : (A) __expf((float) -x))));
        } else {
            f.a[k].inject_out[r - LR] = (float) sum;
        }
    }
    if (!grid_sync(f.bar, f.timeout_ticks)) return;
    for (int idx = tid; idx < T * LR; idx += THREADS) {
        const int k = idx / LR, r = idx - k * LR;
        lo[k][r] = f.a[k].lo[r];
    }
    __syncthreads();
    // (C) the up projection, a thread pair per row (block b: columns 32 b .. 32 b + 31 of all four streams), then the
    // epilogue as gr_up_kernel
    {
        const int d0 = b * UPC;
        const int ur = tid >> 1, uh = tid & 1, uc = ur / UPC, udd = ur - uc * UPC, ui = uc * N + d0 + udd;
        const uint4* w4u = reinterpret_cast<const uint4*>(a0.w_up + (size_t) ui * LR) + uh * 20;
        A v[T];
#pragma unroll
        for (int k = 0; k < T; ++k) v[k] = 0;
#pragma unroll 1
        for (int j0 = 0; j0 < 20; j0 += 5) {   // five chunks a batch (all twenty at once costs registers)
            uint4 w[5];
#pragma unroll
            for (int q = 0; q < 5; ++q) w[q] = __ldg(w4u + j0 + q);
#pragma unroll
            for (int q = 0; q < 5; ++q)
#pragma unroll
                for (int k = 0; k < T; ++k) {
                    if constexpr (DP) v[k] += dot8d(w[q], &lo[k][uh * 160] + (j0 + q) * 8);
                    else v[k] += dot8(w[q], &lo[k][uh * 160] + (j0 + q) * 8);
                }
        }
        float acc[T];
#pragma unroll
        for (int k = 0; k < T; ++k) acc[k] = (float) (v[k] + __shfl_xor(v[k], 1));
        if (uh == 0) {
            const float wn = a0.w_norm[ui];
#pragma unroll
            for (int k = 0; k < T; ++k) {
                const FusedGrArgs& a = f.a[k];
                float rv = a.R[ui];
                if (a.apply) {
                    rv = fmaf(a.bo_prev[d0 + udd], 2.0f * sigmoidf_(a.inj_prev[uc] / (float) HC), rv);
                    a.R_out[ui] = rv;
                }
                g[k][uc][udd] = rv * wn * s_rs[k][uc] * sigmoidf_(acc[k]);
            }
        }
    }
    __syncthreads();
    for (int idx = tid; idx < T * UPC; idx += THREADS) {
        const int k = idx / UPC, col = idx - k * UPC;
        float s = 0.0f;
#pragma unroll
        for (int c = 0; c < HC; ++c) s += g[k][c][col];
        f.a[k].mixed[b * UPC + col] = s / (float) HC;
    }
}
struct DevState {
    float* part = nullptr;
    float* ssp = nullptr;
    double* partd = nullptr;
    double* sspd = nullptr;
    unsigned* bar = nullptr;
    int max_t = 0;      // 0: off on this card
    bool tried = false;
};
DevState g_dev[64];
using KFn = void (*)(PArgs);
template <bool DP>
KFn kernel_t(int T) {
    switch (T) {
        case 1: return persist_kernel<1, DP>;
        case 2: return persist_kernel<2, DP>;
        case 3: return persist_kernel<3, DP>;
        case 4: return persist_kernel<4, DP>;
        case 5: return persist_kernel<5, DP>;
        case 6: return persist_kernel<6, DP>;
        case 7: return persist_kernel<7, DP>;
        case 8: return persist_kernel<8, DP>;
    }
    return nullptr;
}
bool dp_on() {
    static const bool dp = [] { const char* v = std::getenv("STRATA_HC_PERSIST"); return v != nullptr && std::atoi(v) == 3; }();
    return dp;
}
KFn kernel_for(int T) { return dp_on() ? kernel_t<true>(T) : kernel_t<false>(T); }
// s_memrealtime runs at 100 MHz on RDNA: a barrier gives up after 2 s in a read (never in a working run; it only turns a
// hang into a wrong read) and after 50 ms in the start-up check (where a timeout means "not resident here")
constexpr unsigned long long kTicksRun = 200000000ULL, kTicksCheck = 5000000ULL;
void launch(const FusedGrArgs* a, int n_tok, const DevState& st, cudaStream_t s, unsigned long long ticks) {
    PArgs f;
    for (int t = 0; t < n_tok; ++t) f.a[t] = a[t];
    f.part = st.part; f.ssp = st.ssp; f.bar = st.bar; f.partd = st.partd; f.sspd = st.sspd; f.timeout_ticks = ticks;
    kernel_for(n_tok)<<<NB, THREADS, 0, s>>>(f);
}
bool env_on() {
    static const bool on = [] { const char* v = std::getenv("STRATA_HC_PERSIST"); return v != nullptr && std::atoi(v) != 0; }();
    return on;
}
}  // namespace hcp
}  // namespace

void fused_gr_persist_init() {
    if (!hcp::env_on()) return;
    int dev = 0;
    cudaGetDevice(&dev);
    if (dev < 0 || dev >= 64 || hcp::g_dev[dev].tried) return;
    hcp::DevState& st = hcp::g_dev[dev];
    st.tried = true;
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, dev);
    if (prop.warpSize != 32) {
        std::fprintf(stderr, "strata hc: CUDA%d: STRATA_HC_PERSIST=1 is off here: the kernel is written for 32-lane waves "
                             "(this card runs %d)\n", dev, prop.warpSize);
        return;
    }
    if (strata::kernels::gfx_arch_is_gfx12(prop.gcnArchName) || strata::kernels::gfx_arch_is_gfx11_wmma(prop.gcnArchName)) {
        std::fprintf(stderr, "strata hc: CUDA%d: STRATA_HC_PERSIST=1 is off here: gfx11 / gfx12 have no s_memrealtime for the barrier "
                             "timeout (written and measured for gfx1030; an R9700 saw no gain from it, #2226)\n", dev);
        return;
    }
    const int mp = prop.multiProcessorCount;   // only for the confirmation line (WGPs on RDNA)
    bool ok = cudaMalloc((void**) &st.part, (size_t) kFusedGrMaxT * hcp::NR * hcp::NB * sizeof(float)) == cudaSuccess &&
              cudaMalloc((void**) &st.ssp, (size_t) kFusedGrMaxT * hcp::NB * sizeof(float)) == cudaSuccess &&
              cudaMalloc((void**) &st.bar, 64 * sizeof(unsigned)) == cudaSuccess &&
              cudaMalloc((void**) &st.partd, (size_t) kFusedGrMaxT * hcp::NR * hcp::NB * sizeof(double)) == cudaSuccess &&
              cudaMalloc((void**) &st.sspd, (size_t) kFusedGrMaxT * hcp::NB * sizeof(double)) == cudaSuccess &&
              cudaMemset(st.bar, 0, 64 * sizeof(unsigned)) == cudaSuccess;
    if (!ok) {
        cudaGetLastError();
        std::fprintf(stderr, "strata hc: CUDA%d: STRATA_HC_PERSIST=1 is off here: workspace allocation failed\n", dev);
        return;
    }
    // the check: random weights and inputs with the pending write and the inject rows, against the default read; the
    // persistent read sums in another order, so it is bounded, not bitwise.  Every T is RUN with a 50 ms barrier timeout; a
    // timeout means "not resident here" and larger T keep the default read.  No occupancy API in front of it: it promised
    // more than the card holds on 30 WGPs under HSA_CU_MASK (a hang without the timeout) and less than it holds with
    // the HIP 7.1 runtime (1 block per WGP on 40 WGPs, which turned the read off although it runs resident).
    std::mt19937 rng(11);
    std::normal_distribution<float> nd(0.f, 1.f);
    auto bf = [](float x) { uint32_t u; std::memcpy(&u, &x, 4); return (uint16_t) ((u + 0x7fff + ((u >> 16) & 1)) >> 16); };
    const int TM = kFusedGrMaxT;
    std::vector<float> R((size_t) TM * D), bo((size_t) TM * N), inj((size_t) TM * HC), wn(D);
    for (auto& x : R) x = nd(rng);
    for (auto& x : bo) x = 0.5f * nd(rng);
    for (auto& x : inj) x = nd(rng);
    for (auto& x : wn) x = 1.0f + 0.3f * nd(rng);
    std::vector<uint16_t> wd((size_t) LR * D), wu((size_t) D * LR), wi((size_t) HC * D);
    for (auto& x : wd) x = bf(0.02f * nd(rng));
    for (auto& x : wu) x = bf(0.05f * nd(rng));
    for (auto& x : wi) x = bf(0.02f * nd(rng));
    float *dR, *dbo, *dinj, *dwn, *dxn, *dlo, *drs, *dio, *dmix;
    uint16_t *dwd, *dwu, *dwi;
    cudaMalloc((void**) &dR, R.size() * 4); cudaMalloc((void**) &dbo, bo.size() * 4);
    cudaMalloc((void**) &dinj, inj.size() * 4); cudaMalloc((void**) &dwn, wn.size() * 4);
    cudaMalloc((void**) &dxn, (size_t) TM * D * 4); cudaMalloc((void**) &dlo, (size_t) TM * LR * 4);
    cudaMalloc((void**) &drs, (size_t) TM * HC * 4); cudaMalloc((void**) &dio, (size_t) TM * HC * 4);
    cudaMalloc((void**) &dmix, (size_t) TM * N * 4);
    cudaMalloc((void**) &dwd, wd.size() * 2); cudaMalloc((void**) &dwu, wu.size() * 2); cudaMalloc((void**) &dwi, wi.size() * 2);
    cudaMemcpy(dbo, bo.data(), bo.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dinj, inj.data(), inj.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dwn, wn.data(), wn.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dwd, wd.data(), wd.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dwu, wu.data(), wu.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dwi, wi.data(), wi.size() * 2, cudaMemcpyHostToDevice);
    double worst = 0.0;
    int bad_t = 0, timeouts = 0;
    // 0: the read agrees with the default read; 1: a barrier timed out (not resident); 2: wrong or a device error
    auto check = [&](int T) -> int {
        FusedGrArgs a[kFusedGrMaxT];
        for (int t = 0; t < T; ++t) {
            a[t] = FusedGrArgs{};
            a[t].R = dR + (size_t) t * D; a[t].R_out = dR + (size_t) t * D; a[t].apply = true;
            a[t].bo_prev = dbo + (size_t) t * N; a[t].inj_prev = dinj + (size_t) t * HC;
            a[t].w_norm = dwn; a[t].w_down = dwd; a[t].w_up = dwu; a[t].w_inject = dwi; a[t].eps = 1e-6f;
            a[t].lo = dlo + (size_t) t * LR; a[t].rs = drs + (size_t) t * HC;
            a[t].inject_out = dio + (size_t) t * HC; a[t].mixed = dmix + (size_t) t * N;
        }
        std::vector<float> out[2];
        for (int which = 0; which < 2; ++which) {
            cudaMemcpy(dR, R.data(), R.size() * 4, cudaMemcpyHostToDevice);
            cudaMemset(dmix, 0xff, (size_t) TM * N * 4);
            cudaMemset(dio, 0xff, (size_t) TM * HC * 4);
            cudaDeviceSynchronize();
            if (which == 0) fused_gr_read_multi(a, T, dxn, nullptr);
            else hcp::launch(a, T, st, nullptr, hcp::kTicksCheck);
            if (cudaDeviceSynchronize() != cudaSuccess) { cudaGetLastError(); return 2; }
            if (which == 1) {
                unsigned abort_word = 0;
                cudaMemcpy(&abort_word, st.bar + 48, sizeof abort_word, cudaMemcpyDeviceToHost);
                if (abort_word != 0u) {
                    cudaMemset(st.bar, 0, 64 * sizeof(unsigned));
                    cudaDeviceSynchronize();
                    return 1;
                }
            }
            out[which].resize((size_t) T * (N + HC + D));
            cudaMemcpy(out[which].data(), dmix, (size_t) T * N * 4, cudaMemcpyDeviceToHost);
            cudaMemcpy(out[which].data() + (size_t) T * N, dio, (size_t) T * HC * 4, cudaMemcpyDeviceToHost);
            cudaMemcpy(out[which].data() + (size_t) T * (N + HC), dR, (size_t) T * D * 4, cudaMemcpyDeviceToHost);
        }
        double mref = 0.0, md = 0.0;
        bool finite = true;
        for (size_t i = 0; i < out[0].size(); ++i) {
            finite = finite && std::isfinite(out[1][i]);
            mref = std::fmax(mref, std::fabs((double) out[0][i]));
            md = std::fmax(md, std::fabs((double) out[1][i] - out[0][i]));
        }
        const double rel = md / (mref > 0 ? mref : 1.0);
        worst = std::fmax(worst, rel);
        return finite && rel < 1e-4 ? 0 : 2;
    };
    int max_t = 0;
    for (int T = 1; T <= kFusedGrMaxT && ok; ++T) {
        const int r = check(T);
        if (r == 1) ++timeouts; else if (r == 2) { bad_t = T; ok = false; }
        if (r != 0) break;
        max_t = T;
    }
    cudaFree(dR); cudaFree(dbo); cudaFree(dinj); cudaFree(dwn); cudaFree(dxn); cudaFree(dlo); cudaFree(drs);
    cudaFree(dio); cudaFree(dmix); cudaFree(dwd); cudaFree(dwu); cudaFree(dwi);
    cudaMemset(st.bar, 0, 64 * sizeof(unsigned));
    cudaDeviceSynchronize();
    cudaGetLastError();
    if (!ok) {
        std::fprintf(stderr, "strata hc: CUDA%d: STRATA_HC_PERSIST=1 is off here: the check failed at %d tokens "
                             "(max relative difference %.3g)\n", dev, bad_t, worst);
        return;
    }
    if (max_t == 0) {
        std::fprintf(stderr, "strata hc: CUDA%d: STRATA_HC_PERSIST=1 is off here: 80 blocks are not resident at once on %d "
                             "multiprocessors (the check timed out)\n", dev, mp);
        return;
    }
    st.max_t = max_t;
    std::fprintf(stderr, "strata hc: CUDA%d: STRATA_HC_PERSIST=1: the verifier's hyper-connection read runs as one "
                         "persistent kernel (80 blocks on %d multiprocessors, up to %d tokens; %d token counts timed out in "
                         "the check; checked against the default read, max relative difference %.2g)\n", dev, mp, max_t,
                 timeouts, worst);
}

bool fused_gr_read_persist(const FusedGrArgs* a, int n_tok, void* stream) {
    if (!hcp::env_on() || n_tok < 1) return false;
    int dev = 0;
    cudaGetDevice(&dev);
    if (dev < 0 || dev >= 64) return false;
    const hcp::DevState& st = hcp::g_dev[dev];
    if (n_tok > st.max_t) return false;
    if (a[0].q8_down != nullptr || a[0].q8_up != nullptr) return false;    // STRATA_HC_Q8: the default read
    for (int t = 0; t < n_tok; ++t)
        if (a[t].q8_mixed != nullptr) return false;                         // STRATA_QFUSE: the default read
    hcp::launch(a, n_tok, st, (cudaStream_t) stream, hcp::kTicksRun);
    return true;
}
#else
void fused_gr_persist_init() {}
bool fused_gr_read_persist(const FusedGrArgs*, int, void*) { return false; }
#endif

}  // namespace strata::kernels
