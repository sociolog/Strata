// src/kernels/cuda/exl3_dev.cu - EXL3 routed experts on the device: the verify window's grouped experts and the prompt
// path's expert step (include/strata/kernels/exl3_dev.hpp).
//
// Ported from project-maya v1.1.15 src/kernels/cuda/exl3.cu (MIT, Peasant Smith and the Project Maya contributors).
// The codebooks and the trellis window arithmetic follow exllamav3 (MIT, turboderp - exllamav3_ext/quant/codebook.cuh,
// exl3_dq.cuh, hadamard_inner.cuh), and so does decoding a tile straight into tensor-core fragments:
//
//   * a block owns 128 outputs (8 tiles - exactly one output rotation block); warp w takes tile column w through every
//     16-row slice, decoding 8 weights per lane (one fragment: 2 columns x 4 rows) - on sm_80 and later straight into
//     one mma.m16n8k16 a tile with up to 8 activations as its columns, F32 accumulation; elsewhere F32 FMAs;
//   * the input rotation runs per block, 128 inputs at a time (F32 in, FP16 in shared memory); the next 128 inputs'
//     tile words load while these decode.
//
// What changed from Maya's file: only what the two Strata paths use (no decode GEMV `mv`, no single-token MoE with its
// own combine, no q8_1 activations, no pre-sm_80 v1.1.8 kernels); the window kernels read Strata's GPU plan (an
// entry's token and output row explicitly, any top-k, instead of GLM's entry = token * 8 + slot) and write the
// UNWEIGHTED rows the window's combine weighs; a fused gate/up epilogue for the prompt path; streams as void*.
#include "strata/kernels/exl3_dev.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <utility>

namespace strata::kernels::exl3 {
namespace {

constexpr int kThreads = 256, kWarps = 8;
constexpr int kCols = 128;                // outputs per block: 8 tiles, one rotation block
constexpr int kMaxKR = 4096;              // inputs a grouped-expert block walks at most
constexpr float kHad = 0.088388347648f;   // 1 / sqrt(128)

std::atomic<int> g_launch_errors{0};
void launch_check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "exl3_dev %s: %s\n", what, cudaGetErrorString(e));
        g_launch_errors.fetch_add(1, std::memory_order_relaxed);
    }
}

// ---------------------------------------------------------------- the codebooks
__device__ __forceinline__ uint32_t bytesum_plus(uint32_t x, uint32_t acc) {
#if defined(STRATA_USE_HIP)
    return acc + (x & 0xffu) + ((x >> 8) & 0xffu) + ((x >> 16) & 0xffu) + (x >> 24);
#else
    return __dp4a(x, 0x01010101u, acc);
#endif
}

// two states -> two values, FP16 arithmetic as exllamav3's decode_3inst_2 (the tensor-core kernels take the half2)
template <int cb>
__device__ __forceinline__ __half2 decode2h(uint32_t s0, uint32_t s1) {
    if constexpr (cb == 2) {
        const uint32_t a = bytesum_plus(s0 * 0x83DCD12Du, 0x6400u);
        const uint32_t b = bytesum_plus(s1 * 0x83DCD12Du, 0x6400u);
        const __half2 h = __halves2half2(__ushort_as_half((unsigned short) a), __ushort_as_half((unsigned short) b));
        return __hfma2(h, __half2half2(__ushort_as_half(0x1eee)), __half2half2(__ushort_as_half(0xc931)));
    } else {
        uint32_t x0 = cb == 1 ? s0 * 0xCBAC1FEDu : s0 * 89226354u + 64248484u;
        uint32_t x1 = cb == 1 ? s1 * 0xCBAC1FEDu : s1 * 89226354u + 64248484u;
        x0 = (x0 & 0x8fff8fffu) ^ 0x3b603b60u;
        x1 = (x1 & 0x8fff8fffu) ^ 0x3b603b60u;
        const __half2 lo = __halves2half2(__ushort_as_half((unsigned short) (x0 & 0xffffu)),
                                          __ushort_as_half((unsigned short) (x1 & 0xffffu)));
        const __half2 hi = __halves2half2(__ushort_as_half((unsigned short) (x0 >> 16)),
                                          __ushort_as_half((unsigned short) (x1 >> 16)));
        return __hadd2(lo, hi);
    }
}

__device__ __forceinline__ uint32_t h2u(__half2 h) { return *reinterpret_cast<uint32_t*>(&h); }
__device__ __forceinline__ __half2 u2h(uint32_t u) { return *reinterpret_cast<__half2*>(&u); }

// (hi : lo) >> s, the low 32 bits
__device__ __forceinline__ uint32_t fsh(uint32_t lo, uint32_t hi, int s) {
    return (uint32_t) ((((uint64_t) hi << 32) | (uint64_t) lo) >> s);
}

// Step p's state is the 16-bit window ending at bit S(p) of the tile's MSB-first bit stream (w: its 4 K2 words, K2 =
// 2 K), wrapping around the tile: S(p) = (p + 1) K for an integer K, floor((p + 1) K2 / 2) for exllamav3 1.6's
// half-integer K = KA + 0.5 (KA- and KA+1-bit steps alternating).  A lane decodes its 8 consecutive steps lane * 8 ..
// + 7 together, the windows of a group of steps cut from one or two words with funnel shifts - exllamav3's grouping
// per K (exl3_dq.cuh: dq8_aligned_*bits, dq8, dq4, dq2x2, dq8_half), which the steps' fixed offsets keep within two
// words.  f[i]: steps 2i, 2i + 1 as FP16 values.
// Step lane * 8 + j sits at tile row (lane & 3) * 2 + (j & 1) + 8 (j >> 1 & 1), column (lane >> 2) + 8 (j >> 2).
template <int cb>
__device__ __forceinline__ void decode_states(const uint32_t (&s)[8], uint32_t (&f)[4]) {
#pragma unroll
    for (int i = 0; i < 4; ++i) f[i] = h2u(decode2h<cb>(s[2 * i] & 0xffffu, s[2 * i + 1] & 0xffffu));
}
// K = 4: the lane's steps end in its own word b (word lane), the first three reach into a (word lane - 1)
__device__ __forceinline__ void states4(uint32_t a, uint32_t b, uint32_t (&s)[8]) {
    const uint32_t x = fsh(b, a, 20);
    s[0] = x >> 8;
    s[1] = x >> 4;
    s[2] = x;
    s[3] = b >> 16;
    s[4] = b >> 12;
    s[5] = b >> 8;
    s[6] = b >> 4;
    s[7] = b;
}
// step p .. p + G - 1's windows (G = 4 or 2) from the two words that hold them
template <int K, int G>
__device__ __forceinline__ void states_group(const uint32_t* w, int p, uint32_t* s) {
    constexpr int NW = 8 * K;
    const int b0 = (p + 257) * K - 16, b2 = b0 + (G - 1) * K + 16;
    const int i0 = b0 >> 5, i2 = (b2 - 1) >> 5, s2 = ((i2 + 1) << 5) - b2;
    const uint32_t a = w[i0 % NW], b = w[i2 % NW];
#pragma unroll
    for (int j = 0; j < G; ++j) s[G - 1 - j] = fsh(b, a, s2 + j * K);
}
// an integer K
template <int K, int cb>
__device__ __forceinline__ void frag8_int(const uint32_t* w, int lane, uint32_t (&f)[4]) {
    const int p = lane * 8;
    uint32_t s[8];
    if constexpr (K == 4) {
        states4(w[(lane + 31) & 31], w[lane], s);
    } else if constexpr (K == 1 || K == 2) {
        // the 8 windows overlap in one 32-bit window: steps p + j end 8 - 1 - j K bits apart
        constexpr int NW = 8 * K, per = 32 / (8 * K);   // lanes per word
        const int i1 = lane / per, i0 = (i1 + NW - 1) % NW;
        const uint32_t b = fsh(w[i1], w[i0], (per - 1 - lane % per) * 8 * K);
#pragma unroll
        for (int j = 0; j < 8; ++j) s[j] = b >> ((7 - j) * K);
    } else if constexpr (K == 3) {
        // all 8 windows within two words: two funnel shifts, the rest by shifting
        constexpr int NW = 24;
        const int b1 = (p + 257) * 3, b0 = b1 - 16, b2 = b1 + 21;
        const int i0 = b0 >> 5, i2 = (b2 - 1) >> 5, s2 = ((i2 + 1) << 5) - b2;
        const uint32_t a = w[i0 % NW], b = w[i2 % NW];
        const uint32_t hi = fsh(b, a, s2), lo = fsh(b, a, s2 + 12);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            s[7 - j] = hi >> (3 * j);
            s[3 - j] = lo >> (3 * j);
        }
    } else if constexpr (K == 7) {
#pragma unroll
        for (int g = 0; g < 4; ++g) states_group<K, 2>(w, p + 2 * g, s + 2 * g);
    } else {   // 5, 6, 8
        states_group<K, 4>(w, p, s);
        states_group<K, 4>(w, p + 4, s + 4);
    }
    decode_states<cb>(s, f);
}
// a half-integer K (K2 = 2 KA + 1, mul1): exllamav3's dq8_half.  The lane's steps start even, so their windows end at
// S(p + 7) = (p / 2 + 4) K2 and then KA + 1, KA, KA + 1, ... bits earlier; four windows span 18 + 3 KA <= 27 bits, one
// funnel shift of two words each (steps 4..7 from window 7's end, 0..3 from window 3's, 2 K2 bits earlier)
template <int K2, int cb>
__device__ __forceinline__ void frag8_half(const uint32_t* w, int lane, uint32_t (&f)[4]) {
    constexpr int KA = K2 / 2, NW = 4 * K2, gspan = 18 + 3 * KA;
    const int e7 = (lane * 4 + 4) * K2 + 128 * K2;   // one ring on: no negative bit index
    uint32_t s[8];
#pragma unroll
    for (int g = 0; g < 2; ++g) {
        const int e = e7 - (1 - g) * 2 * K2;
        const int hi = (e - 1) >> 5, lo = (e - gspan) >> 5, sh = ((hi + 1) << 5) - e;
        const uint32_t x = fsh(w[hi % NW], w[lo % NW], sh);   // (w[lo] : w[hi]) >> sh: the window ending at e lowest
        s[4 * g + 3] = x;
        s[4 * g + 2] = x >> (KA + 1);
        s[4 * g + 1] = x >> (2 * KA + 1);
        s[4 * g + 0] = x >> (3 * KA + 2);
    }
    decode_states<cb>(s, f);
}
template <int K2, int cb>
__device__ __forceinline__ void frag8(const uint32_t* w, int lane, uint32_t (&f)[4]) {
    if constexpr ((K2 & 1) != 0) {
        static_assert(cb == 2 && K2 <= 7, "a half-integer K: 1.5 / 2.5 / 3.5, mul1 only");
        frag8_half<K2, cb>(w, lane, f);
    } else {
        frag8_int<K2 / 2, cb>(w, lane, f);
    }
}

// ---------------------------------------------------------------- the rotation
// the 128-point transform of the 128 values a warp holds 4 per lane (value lane * 4 + i), over sqrt(128)
__device__ __forceinline__ void had128(float* v, int lane) {
    const float s0 = v[0] + v[1], d0 = v[0] - v[1], s1 = v[2] + v[3], d1 = v[2] - v[3];
    v[0] = s0 + s1;
    v[1] = d0 + d1;
    v[2] = s0 - s1;
    v[3] = d0 - d1;
#pragma unroll
    for (int m = 1; m < 32; m <<= 1) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float o = __shfl_xor_sync(0xffffffffu, v[i], m);
            v[i] = (lane & m) ? o - v[i] : v[i] + o;
        }
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) v[i] *= kHad;
}

__device__ __forceinline__ float h2f(uint16_t b) { return __half2float(__ushort_as_half(b)); }

// the activation a warp rotates into its block's FP16 chunks (F32)
struct Xin {
    const float* f = nullptr;
};

// the 128-input chunk of x from k0 on: the lane's 4 values and their suh ...
__device__ __forceinline__ void chunk_load(const Xin& x, const uint16_t* __restrict__ suh, int k0, int lane,
                                           float (&v)[4], float (&s)[4]) {
    const int base = k0 + lane * 4;
#pragma unroll
    for (int i = 0; i < 4; ++i) v[i] = x.f[base + i];
#pragma unroll
    for (int i = 0; i < 4; ++i) s[i] = h2f(suh[base + i]);
}
// ... and H(x * suh) of the chunk, FP16, into dst[0..127].  The scale is its own rounded multiply (__fmul_rn): left to
// the compiler, x * suh fused into the transform's first add (an FMA), so the rotated values were not those of the
// operations as written (1 FP16 value in ~3,000 one unit off a host emulation); now they are, exactly.
__device__ __forceinline__ void chunk_store(float (&v)[4], const float (&s)[4], int lane, __half* dst) {
#pragma unroll
    for (int i = 0; i < 4; ++i) v[i] = __fmul_rn(v[i], s[i]);
    had128(v, lane);
    __half2* d = reinterpret_cast<__half2*>(dst + lane * 4);
    d[0] = __floats2half2_rn(v[0], v[1]);
    d[1] = __floats2half2_rn(v[2], v[3]);
}

// ---------------------------------------------------------------- the block GEMV
// A block owns 128 outputs - 8 tiles, warp w the tile column tc0 + w through all of the block's inputs [k0, k1) - for
// nc <= kTok activations at once.  Per 128-input chunk, warp c (< nc) scales activation c's chunk by suh and rotates
// it into FP16 (xb, two buffers), then each warp decodes its tile of each of the chunk's 8 slices and multiplies it
// with every activation: on sm_80 and later as one mma.m16n8k16 a tile (the decoded tile is the A operand - a lane's 8
// steps are its A fragment - and the activations are the B operand's columns), F32 accumulation; elsewhere on the CUDA
// cores, in F32.  An activation's sums do not depend on the other columns, so a window's token gets the bits it gets
// alone.  The next chunk's tile words are loaded while this one is decoded.  out[c * 128 + i]: activation c's sum for
// output i.  Ends with a __syncthreads (out complete).
constexpr int kTok = 8;   // activations a block takes at once (the MMA's n)

#if !defined(STRATA_USE_HIP) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#define EXL3_TC 1
#endif

#if !defined(EXL3_TC)
// a lane's 4 weights of one column times the activation's 4 inputs, added to a (the CUDA-core product)
__device__ __forceinline__ float dot4_acc(float a, const float* v, float x0, float x1, float x2, float x3) {
    return __fadd_rn(a, __fmaf_rn(v[3], x3, __fmaf_rn(v[2], x2, __fmaf_rn(v[1], x1, __fmul_rn(v[0], x0)))));
}
#else
__device__ __forceinline__ void mma_f32(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                        uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
#endif

// a tile's words, and a lane's share of them (word lane + 32 r)
template <int K2>
struct Words {
    static constexpr int W = 4 * K2, R = (W + 31) / 32;
};

// the warp's tile of each of a chunk's 8 slices (slice s at tile0 + s sstride)
template <int K2>
__device__ __forceinline__ void load_tiles(const uint32_t* __restrict__ tile0, size_t sstride, int lane,
                                           uint32_t (&w)[8][Words<K2>::R]) {
    constexpr int W = Words<K2>::W, R = Words<K2>::R;
#pragma unroll
    for (int s = 0; s < 8; ++s)
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const int i = lane + 32 * r;
            w[s][r] = (W % 32 == 0 || i < W) ? __ldg(tile0 + (size_t) s * sstride + i) : 0u;
        }
}

// one slice's fragment from the lane's words: K = 4 straight from the registers (the lane's steps end in its own word,
// the first three reach into the previous lane's), other K (a half-integer one too) through the warp's stage (16 K words)
template <int K2, int cb>
__device__ __forceinline__ void slice_frags(const uint32_t (&w)[Words<K2>::R], uint32_t* st, int lane, uint32_t (&f)[4]) {
    if constexpr (K2 == 8) {
        uint32_t s[8];
        states4(__shfl_sync(0xffffffffu, w[0], (lane + 31) & 31), w[0], s);
        decode_states<cb>(s, f);
    } else {
        constexpr int W = Words<K2>::W, R = Words<K2>::R;
        __syncwarp();
#pragma unroll
        for (int r = 0; r < R; ++r)
            if (W % 32 == 0 || lane + 32 * r < W) st[lane + 32 * r] = w[r];
        __syncwarp();
        frag8<K2, cb>(st, lane, f);
    }
}

// the blocks an SM keeps of the block_gemv kernels (their registers), as measured on an RTX 3090 and an RTX 5060 Ti
// (Maya): the routed experts four at K <= 4, two at wider codes; before sm_80 (the CUDA cores' product) two; HIP reads
// the second bound differently and keeps its default.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
template <int K2>
constexpr int kMinBlocksX = K2 <= 8 ? 4 : 2;
#else
template <int K2>
constexpr int kMinBlocksX = 2;
#endif
#if defined(STRATA_USE_HIP)
#define EXL3_BOUNDS(n) __launch_bounds__(kThreads)
#else
#define EXL3_BOUNDS(n) __launch_bounds__(kThreads, n)
#endif

// the shared memory of a block_gemv block
template <int K2>
struct BlockSmem {
    __half xb[2 * kTok * kCols];       // two chunk buffers, kTok activations each
    uint32_t stage[kWarps * 4 * K2];   // a tile per warp (K != 4)
    float out[kTok * kCols];
};

template <int K2, int cb>
__device__ void block_gemv(const uint32_t* __restrict__ tr, int ld, int tc0, int k0, int k1, const Xin& x,
                           const uint16_t* __restrict__ suh, int nc, BlockSmem<K2>& sm) {
    constexpr int R = Words<K2>::R;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, t = lane & 3;
    constexpr int W = Words<K2>::W;
    const size_t sstride = (size_t) ld * W;
    const uint32_t* tile = tr + ((size_t) (k0 / 16) * ld + tc0 + warp) * W;
    uint32_t* st = sm.stage + warp * W;
    const bool rot = warp < nc;
    const int nq = (k1 - k0) / kCols;
    uint32_t wc[8][R], wn[8][R];
    float xv[4], xs4[4];
    load_tiles<K2>(tile, sstride, lane, wc);
    if (rot) chunk_load(x, suh, k0, lane, xv, xs4);
#if defined(EXL3_TC)
    float c[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#else
    float acc[kTok][2];
#pragma unroll
    for (int cc = 0; cc < kTok; ++cc) acc[cc][0] = acc[cc][1] = 0.0f;
#endif
    for (int q = 0; q < nq; ++q) {
        __half* xq = sm.xb + (q & 1) * (kTok * kCols);
        if (rot) chunk_store(xv, xs4, lane, xq + warp * kCols);
        __syncthreads();
        if (q + 1 < nq) {
            load_tiles<K2>(tile + (size_t) (q + 1) * 8 * sstride, sstride, lane, wn);
            if (rot) chunk_load(x, suh, k0 + (q + 1) * kCols, lane, xv, xs4);
        }
#pragma unroll
        for (int s = 0; s < 8; ++s) {
            uint32_t f[4];
            slice_frags<K2, cb>(wc[s], st, lane, f);
#if defined(EXL3_TC)
            uint32_t b0 = 0u, b1 = 0u;
            if (g < nc) {
                const __half* xr = xq + g * kCols + s * 16 + 2 * t;
                b0 = *reinterpret_cast<const uint32_t*>(xr);
                b1 = *reinterpret_cast<const uint32_t*>(xr + 8);
            }
            // A: rows (outputs) g / g + 8, columns (inputs) 2t.. / 2t + 8..: steps 0,1 / 4,5 / 2,3 / 6,7
            mma_f32(c, f[0], f[2], f[1], f[3], b0, b1);
#else
            float v[8];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float2 d = __half22float2(u2h(f[i]));
                v[2 * i] = d.x;
                v[2 * i + 1] = d.y;
            }
#pragma unroll
            for (int cc = 0; cc < kTok; ++cc)
                if (cc < nc) {
                    const __half* xr = xq + cc * kCols + s * 16 + 2 * t;
                    const float x0 = __half2float(xr[0]), x1 = __half2float(xr[1]);
                    const float x2 = __half2float(xr[8]), x3 = __half2float(xr[9]);
                    acc[cc][0] = dot4_acc(acc[cc][0], v, x0, x1, x2, x3);
                    acc[cc][1] = dot4_acc(acc[cc][1], v + 4, x0, x1, x2, x3);
                }
#endif
        }
#pragma unroll
        for (int s = 0; s < 8; ++s)
#pragma unroll
            for (int r = 0; r < R; ++r) wc[s][r] = wn[s][r];
    }
    const int o = warp * 16 + g;
#if defined(EXL3_TC)
    // c: the tile's outputs g (c[0], c[1]) and g + 8 (c[2], c[3]) for activations 2t (c[0], c[2]) and 2t + 1
    if (2 * t < nc) {
        sm.out[(2 * t) * kCols + o] = c[0];
        sm.out[(2 * t) * kCols + o + 8] = c[2];
    }
    if (2 * t + 1 < nc) {
        sm.out[(2 * t + 1) * kCols + o] = c[1];
        sm.out[(2 * t + 1) * kCols + o + 8] = c[3];
    }
#else
    // the 4 lanes of one g hold parts of the same two outputs
#pragma unroll
    for (int cc = 0; cc < kTok; ++cc)
        if (cc < nc) {
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                acc[cc][h] += __shfl_xor_sync(0xffffffffu, acc[cc][h], 1);
                acc[cc][h] += __shfl_xor_sync(0xffffffffu, acc[cc][h], 2);
            }
            if (t == 0) {
                sm.out[cc * kCols + o] = acc[cc][0];
                sm.out[cc * kCols + o + 8] = acc[cc][1];
            }
        }
#endif
    __syncthreads();
}

// one warp's output block of 128: H, then svh
__device__ __forceinline__ void block_out(const float* red, const uint16_t* svh, float* dst) {
    const int lane = threadIdx.x & 31;
    float v[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) v[i] = red[lane * 4 + i];
    had128(v, lane);
#pragma unroll
    for (int i = 0; i < 4; ++i) dst[lane * 4 + i] = __fmul_rn(v[i], h2f(svh[lane * 4 + i]));
}
// the swiglu (clamped when limit > 0: GLM's experts; Qwen's take limit 0)
__device__ __forceinline__ float swiglu(float gv, float uv, float limit) {
    if (limit > 0.0f) {
        gv = fminf(gv, limit);
        uv = fminf(fmaxf(uv, -limit), limit);
    }
    return __fmul_rn(__fdiv_rn(gv, __fadd_rn(1.0f, expf(-gv))), uv);
}
// the second block of a (gate, up) pair to finish (its counter back at zero for the next launch)
__device__ __forceinline__ bool pair_last(unsigned* cnt, int& last) {
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) last = atomicAdd(cnt, 1u) == 1u;
    __syncthreads();
    if (!last) return false;
    __threadfence();
    if (threadIdx.x == 0) *cnt = 0u;
    return true;
}

// ---------------------------------------------------------------- grouped experts (the verify windows)
// GroupArgs on the device.  Each group's weights are read and decoded once for up to kTok of its entries at a time (as
// exllamav3's fused MoE runs a run of slots that picked one expert as the rows of one MMA tile), with each entry's own
// arithmetic: every entry's row is what the same kernels give it alone, bit for bit.
struct KGroup {
    const unsigned long long* grp_ptr;
    const int32_t *grp_start, *n_groups, *ent_tok, *ent_dst;
    const float* x;
    size_t x_stride;
    float* h;            // [entries][n_ff]: the gate's values, then the swiglu
    float* u;            // [entries][n_ff]: the up's values
    float* out;
    size_t out_stride;
    unsigned* cnt;       // groups x n_ff / 128: the gate / up pairs
    size_t suh[3], svh[3], tr[3];
    int n_embd, n_ff;
    float limit;
};

// grid (n_ff / 128, max groups, 2): group blockIdx.y's gate (z 0, into h) or up (z 1, into u) for 128 h values of
// each of its entries, kTok entries a pass; the pair's second block to finish turns them into the swiglu, in h
template <int K2, int cb>
__global__ void EXL3_BOUNDS(kMinBlocksX<K2>) group_gu_kernel(const KGroup a) {
    __shared__ __align__(16) BlockSmem<K2> sm;
    __shared__ int last;
    const int grp = blockIdx.y, g = blockIdx.x, r = blockIdx.z;
    if (grp >= *a.n_groups) return;
    const uint8_t* blob = (const uint8_t*) a.grp_ptr[grp];
    const int warp = threadIdx.x >> 5;
    const int m0 = a.grp_start[grp], m1 = a.grp_start[grp + 1];
    float* dst = r == 0 ? a.h : a.u;
    for (int mm = m0; mm < m1; mm += kTok) {
        const int nc = min(kTok, m1 - mm);
        Xin x;
        if (warp < nc) x.f = a.x + (size_t) a.ent_tok[mm + warp] * a.x_stride;
        block_gemv<K2, cb>((const uint32_t*) (blob + a.tr[r]), a.n_ff / 16, g * 8, 0, a.n_embd, x,
                          (const uint16_t*) (blob + a.suh[r]), nc, sm);
        if (warp < nc)
            block_out(sm.out + warp * kCols, (const uint16_t*) (blob + a.svh[r]) + g * kCols,
                      dst + (size_t) (mm + warp) * a.n_ff + g * kCols);
        __syncthreads();
    }
    if (!pair_last(a.cnt + grp * (a.n_ff / kCols) + g, last)) return;
    if (threadIdx.x < kCols)
        for (int m = m0; m < m1; ++m) {
            const size_t row = (size_t) m * a.n_ff + g * kCols + threadIdx.x;
            a.h[row] = swiglu(*(volatile const float*) &a.h[row], *(volatile const float*) &a.u[row], a.limit);
        }
}

// grid (n_embd / 128, max groups): group blockIdx.y's down rows for 128 outputs of each entry, into the entry's row
template <int K2, int cb>
__global__ void EXL3_BOUNDS(kMinBlocksX<K2>) group_down_kernel(const KGroup a) {
    __shared__ __align__(16) BlockSmem<K2> sm;
    const int grp = blockIdx.y, g = blockIdx.x;
    if (grp >= *a.n_groups) return;
    const uint8_t* blob = (const uint8_t*) a.grp_ptr[grp];
    const int warp = threadIdx.x >> 5;
    const int m0 = a.grp_start[grp], m1 = a.grp_start[grp + 1];
    for (int mm = m0; mm < m1; mm += kTok) {
        const int nc = min(kTok, m1 - mm);
        Xin x;
        if (warp < nc) x.f = a.h + (size_t) (mm + warp) * a.n_ff;
        block_gemv<K2, cb>((const uint32_t*) (blob + a.tr[2]), a.n_embd / 16, g * 8, 0, a.n_ff, x,
                          (const uint16_t*) (blob + a.suh[2]), nc, sm);
        if (warp < nc)
            block_out(sm.out + warp * kCols, (const uint16_t*) (blob + a.svh[2]) + g * kCols,
                      a.out + (size_t) a.ent_dst[mm + warp] * a.out_stride + g * kCols);
        __syncthreads();
    }
}

// ---------------------------------------------------------------- the prompt path: decode and rotations
// grid (n / 128, k / 16): W' row-major, one tile per warp (tests)
template <int K2, int cb>
__global__ void __launch_bounds__(kThreads) recon_inner_kernel(const uint32_t* __restrict__ tr, int ld, int n,
                                                               uint16_t* dst) {
    constexpr int W = 4 * K2;
    const int lane = threadIdx.x & 31, t = threadIdx.x >> 5;
    const int tc = blockIdx.x * 8 + t, ks = blockIdx.y;
    uint32_t f[4];
    frag8<K2, cb>(tr + ((size_t) ks * ld + tc) * W, lane, f);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const __half2 h = u2h(f[j >> 1]);
        const __half v = (j & 1) ? __high2half(h) : __low2half(h);
        const int r = ks * 16 + (lane & 3) * 2 + (j & 1) + 8 * ((j >> 1) & 1);
        const int c = tc * 16 + (lane >> 2) + 8 * (j >> 2);
        dst[(size_t) r * n + c] = __half_as_ushort(v);
    }
}

// grid (n / 128, k / 128): W' of the 128 x 128 block, transposed (dst[c][r]) - the decoded values alone.  Each warp
// decodes its slice's 8 tiles into the block's tile (half2 pairs of rows, conflict-free), then writes 256-byte rows.
template <int K2, int cb>
__global__ void __launch_bounds__(kThreads) inner_rows_kernel(const uint32_t* __restrict__ tr, int ld, int k,
                                                              uint16_t* __restrict__ dst) {
    constexpr int W = 4 * K2;
    constexpr int P = kCols + 8;   // a column's 128 rows, padded (halves; 16-byte rows)
    __shared__ __align__(16) __half tile[kCols * P];   // tile[c * P + r]
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, t = lane & 3;
    const int nb = blockIdx.x, kb = blockIdx.y;
    for (int ks = warp; ks < 8; ks += kWarps) {
        const uint32_t* src = tr + ((size_t) (kb * 8 + ks) * ld + nb * 8) * W;
        const int r = ks * 16 + 2 * t;
#pragma unroll
        for (int tc = 0; tc < 8; ++tc) {
            uint32_t f[4];
            frag8<K2, cb>(src + tc * W, lane, f);
            const int c = tc * 16 + g;
            *reinterpret_cast<uint32_t*>(&tile[c * P + r]) = f[0];
            *reinterpret_cast<uint32_t*>(&tile[c * P + r + 8]) = f[1];
            *reinterpret_cast<uint32_t*>(&tile[(c + 8) * P + r]) = f[2];
            *reinterpret_cast<uint32_t*>(&tile[(c + 8) * P + r + 8]) = f[3];
        }
    }
    __syncthreads();
    for (int c = warp; c < kCols; c += kWarps)
        *reinterpret_cast<uint2*>(dst + (size_t) (nb * kCols + c) * k + kb * kCols + lane * 4) =
            *reinterpret_cast<const uint2*>(&tile[c * P + lane * 4]);
}

// one warp per 128 values of a row: the rotated FP16 input of an EXL3 matrix (rows_rot_f16) ...
__global__ void __launch_bounds__(kThreads) rows_rot_f16_kernel(const float* __restrict__ x, int ldx,
                                                                const int32_t* __restrict__ idx, int nrows, int ncols,
                                                                const uint16_t* __restrict__ suh, uint16_t* out,
                                                                int ldo) {
    const int lane = threadIdx.x & 31, nb = ncols / kCols;
    const int w = blockIdx.x * kWarps + (threadIdx.x >> 5);
    if (w >= nrows * nb) return;
    const int r = w / nb, b = w % nb;
    Xin xin;
    xin.f = x + (size_t) (idx != nullptr ? idx[r] : r) * ldx;
    float v[4], sc[4];
    chunk_load(xin, suh, b * kCols, lane, v, sc);
    chunk_store(v, sc, lane, reinterpret_cast<__half*>(out) + (size_t) r * ldo + b * kCols);
}
// ... and its output rotation in place (block_out's arithmetic)
__global__ void __launch_bounds__(kThreads) rows_rot_out_kernel(float* y, int ldy, int nrows, int ncols,
                                                                const uint16_t* __restrict__ svh) {
    const int lane = threadIdx.x & 31, nb = ncols / kCols;
    const int w = blockIdx.x * kWarps + (threadIdx.x >> 5);
    if (w >= nrows * nb) return;
    const int r = w / nb, b = w % nb;
    float* p = y + (size_t) r * ldy + b * kCols;
    float v[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) v[i] = p[lane * 4 + i];
    had128(v, lane);
#pragma unroll
    for (int i = 0; i < 4; ++i) p[lane * 4 + i] = __fmul_rn(v[i], h2f(svh[b * kCols + lane * 4 + i]));
}
// gate/up's epilogue + down's input in one pass, one warp per 128 values of h: rows_rot_out on the gate and the up
// block, the swiglu, then rows_rot_f16 with suh_d - the same operations in the same order, so the same bits
__global__ void __launch_bounds__(kThreads) gu_epilogue_kernel(const float* __restrict__ gu, int ldgu, int nrows,
                                                               int n_ff, const uint16_t* __restrict__ svh_g,
                                                               const uint16_t* __restrict__ svh_u,
                                                               const uint16_t* __restrict__ suh_d, float limit,
                                                               uint16_t* h16, int ldh) {
    const int lane = threadIdx.x & 31, nb = n_ff / kCols;
    const int w = blockIdx.x * kWarps + (threadIdx.x >> 5);
    if (w >= nrows * nb) return;
    const int r = w / nb, b = w % nb;
    const float* pg = gu + (size_t) r * ldgu + b * kCols;
    const float* pu = pg + n_ff;
    float vg[4], vu[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        vg[i] = pg[lane * 4 + i];
        vu[i] = pu[lane * 4 + i];
    }
    had128(vg, lane);
    had128(vu, lane);
    float hv[4], sc[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int c = b * kCols + lane * 4 + i;
        hv[i] = swiglu(__fmul_rn(vg[i], h2f(svh_g[c])), __fmul_rn(vu[i], h2f(svh_u[c])), limit);
        sc[i] = h2f(suh_d[c]);
    }
    chunk_store(hv, sc, lane, reinterpret_cast<__half*>(h16) + (size_t) r * ldh + b * kCols);
}

// ---------------------------------------------------------------- the prompt path's tensor-core GEMMs (sm_80+)
// A block: BM rows x 128 outputs (one output rotation block).  Per 128 inputs the block's rows are scaled by suh,
// rotated and stored FP16 in shared memory (the A operands); per 16-row slice each warp decodes its 16 x 16 tile
// straight into mma.m16n8k16's B fragments - a lane's 8 steps ARE the fragment order (steps 0,1 / 2,3 rows 2t+{0,1} /
// 2t+8+{0,1} of column g, steps 4..7 the same of column g + 8) - and multiplies every m-tile of the block with them.
// The output rotation and svh run on the F32 tile in shared memory.
#if !defined(STRATA_USE_HIP)
namespace tc {

constexpr int BM = 32, MT = BM / 16, AP = 136;   // rows a block, its m-tiles, the A rows' pitch (halves)

__device__ __forceinline__ void mma16816(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#endif
}
// the lane's 4 B registers of tile t (words staged in st): b[0..1] columns 0..7, b[2..3] columns 8..15
template <int K2, int cb>
__device__ __forceinline__ void tile_frags(const uint32_t* st, int lane, uint32_t* b) {
    uint32_t f[4];
    frag8<K2, cb>(st, lane, f);
#pragma unroll
    for (int i = 0; i < 4; ++i) b[i] = f[i];
}

// A[r][j] = FP16(H(x_r * suh)[j]), j < 128, for the block's rows (zero past nrows).  x_r: F32 row idx[r] of xf (ldx)
// when xf is given, else FP16 row r of xh (ldx).  Every warp, BM / 8 rows each.
__device__ __forceinline__ void load_a(const float* __restrict__ xf, const int32_t* __restrict__ idx,
                                       const __half* __restrict__ xh, int ldx, int r0, int nrows, int k0,
                                       const uint16_t* __restrict__ suh, __half* A) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (int rr = warp; rr < BM; rr += kWarps) {
        float v[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        const int r = r0 + rr;
        if (rr < nrows) {
            const int c = k0 + lane * 4;
            if (xf != nullptr) {
                const float* src = xf + (size_t) idx[r] * ldx + c;
#pragma unroll
                for (int i = 0; i < 4; ++i) v[i] = src[i];
            } else {
                const __half* src = xh + (size_t) r * ldx + c;
#pragma unroll
                for (int i = 0; i < 4; ++i) v[i] = __half2float(src[i]);
            }
#pragma unroll
            for (int i = 0; i < 4; ++i) v[i] = __fmul_rn(v[i], h2f(suh[c + i]));   // (chunk_store's note)
            had128(v, lane);
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) A[rr * AP + lane * 4 + i] = __float2half_rn(v[i]);
    }
}

// the m-tile's A registers for the 16-column slice s of A
__device__ __forceinline__ void a_frags(const __half* A, int mt, int s, int lane, uint32_t* a) {
    const int g = lane >> 2, t = lane & 3;
    const __half* p = A + (mt * 16 + g) * AP + s * 16 + 2 * t;
    a[0] = *reinterpret_cast<const uint32_t*>(p);
    a[1] = *reinterpret_cast<const uint32_t*>(p + 8 * AP);
    a[2] = *reinterpret_cast<const uint32_t*>(p + 8);
    a[3] = *reinterpret_cast<const uint32_t*>(p + 8 * AP + 8);
}

// acc (this warp's 16 columns of the block's BM rows) -> T[BM][128] F32
__device__ __forceinline__ void store_acc(const float (&acc)[MT][2][4], float* T) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, t = lane & 3;
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
        for (int ns = 0; ns < 2; ++ns) {
            const int c = warp * 16 + ns * 8 + 2 * t, r = mt * 16 + g;
            T[r * 128 + c] = acc[mt][ns][0];
            T[r * 128 + c + 1] = acc[mt][ns][1];
            T[(r + 8) * 128 + c] = acc[mt][ns][2];
            T[(r + 8) * 128 + c + 1] = acc[mt][ns][3];
        }
}

struct GArgs {
    const uint8_t* wbase;     // expert e's blob at wbase + e * stride
    size_t stride;
    size_t tr[3], suh[3], svh[3];
    int ld[3];                // tiles per 16-row slice of each matrix
    const int32_t* bounds;    // device: expert e's rows [bounds[e], bounds[e + 1])
    int row0;                 // local row of a sorted row: bounds[e] - row0 + r
    int k, n;                 // of the matrix (gate/up: n_embd, n_ff)
    const float* xf;          // F32 input rows by index (idx[local row]) ...
    const int32_t* idx;
    const __half* xh;         // ... or FP16 rows by local row
    int ldx;
    float* y;                 // F32 output rows (ldy)
    int ldy;
    __half* h;                // (gate/up) the swiglu as FP16 rows (ldh)
    int ldh;
    float limit;
};

// one matrix: y = H(H(x * suh) W') * svh.  grid (row blocks, n / 128, experts)
template <int K2, int cb>
__global__ void __launch_bounds__(kThreads) gemm_kernel(const GArgs a) {
    constexpr int W = 4 * K2;
    __shared__ __align__(16) unsigned char sm[BM * 128 * 4];   // A (BM x AP halves), then the F32 tile
    __shared__ uint32_t stage[kWarps][64];
    __half* A = reinterpret_cast<__half*>(sm);
    float* T = reinterpret_cast<float*>(sm);
    const int e = blockIdx.z, rb = blockIdx.x, g = blockIdx.y;
    const int e0 = a.bounds[e] - a.row0, ne = a.bounds[e + 1] - a.bounds[e];
    const int r0 = e0 + rb * BM, nrows = min(BM, ne - rb * BM);
    if (nrows <= 0) return;
    const uint8_t* blob = a.wbase + (size_t) e * a.stride;
    const uint32_t* tr = (const uint32_t*) (blob + a.tr[0]);
    const uint16_t* suh = (const uint16_t*) (blob + a.suh[0]);
    const uint16_t* svh = (const uint16_t*) (blob + a.svh[0]);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    float acc[MT][2][4] = {};
    for (int k0 = 0; k0 < a.k; k0 += 128) {
        __syncthreads();
        load_a(a.xf, a.idx, a.xh, a.ldx, r0, nrows, k0, suh, A);
        __syncthreads();
#pragma unroll 1
        for (int s = 0; s < 8; ++s) {
            const int ks = k0 / 16 + s;
            const uint32_t* src = tr + ((size_t) ks * a.ld[0] + g * 8 + warp) * W;
            __syncwarp();
            for (int i = lane; i < W; i += 32) stage[warp][i] = src[i];
            __syncwarp();
            uint32_t b[4];
            tile_frags<K2, cb>(stage[warp], lane, b);
#pragma unroll
            for (int mt = 0; mt < MT; ++mt) {
                uint32_t af[4];
                a_frags(A, mt, s, lane, af);
                mma16816(acc[mt][0], af, b[0], b[1]);
                mma16816(acc[mt][1], af, b[2], b[3]);
            }
        }
    }
    __syncthreads();
    store_acc(acc, T);
    __syncthreads();
    for (int rr = warp; rr < nrows; rr += kWarps) {
        float v[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) v[i] = T[rr * 128 + lane * 4 + i];
        had128(v, lane);
        float* yr = a.y + (size_t) (r0 + rr) * a.ldy + g * 128 + lane * 4;
#pragma unroll
        for (int i = 0; i < 4; ++i) yr[i] = v[i] * h2f(svh[g * 128 + lane * 4 + i]);
    }
}

// gate and up (matrices 0 and 1) over the same rows, the swiglu, FP16 h.  grid (row blocks, n / 128, experts)
template <int K2, int cb>
__global__ void __launch_bounds__(kThreads) gate_up_kernel(const GArgs a) {
    constexpr int W = 4 * K2;
    __shared__ __align__(16) unsigned char sm[2 * BM * 128 * 4];   // A_g, A_u, then the two F32 tiles
    __shared__ uint32_t stage[kWarps][64];
    __half* A[2] = {reinterpret_cast<__half*>(sm), reinterpret_cast<__half*>(sm) + BM * AP};
    float* T[2] = {reinterpret_cast<float*>(sm), reinterpret_cast<float*>(sm) + BM * 128};
    const int e = blockIdx.z, rb = blockIdx.x, g = blockIdx.y;
    const int e0 = a.bounds[e] - a.row0, ne = a.bounds[e + 1] - a.bounds[e];
    const int r0 = e0 + rb * BM, nrows = min(BM, ne - rb * BM);
    if (nrows <= 0) return;
    const uint8_t* blob = a.wbase + (size_t) e * a.stride;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    float acc[2][MT][2][4] = {};
    for (int k0 = 0; k0 < a.k; k0 += 128) {
        __syncthreads();
        for (int m = 0; m < 2; ++m)
            load_a(a.xf, a.idx, a.xh, a.ldx, r0, nrows, k0, (const uint16_t*) (blob + a.suh[m]), A[m]);
        __syncthreads();
#pragma unroll 1
        for (int s = 0; s < 8; ++s) {
            const int ks = k0 / 16 + s;
#pragma unroll
            for (int m = 0; m < 2; ++m) {
                const uint32_t* src = (const uint32_t*) (blob + a.tr[m]) + ((size_t) ks * a.ld[m] + g * 8 + warp) * W;
                __syncwarp();
                for (int i = lane; i < W; i += 32) stage[warp][i] = src[i];
                __syncwarp();
                uint32_t b[4];
                tile_frags<K2, cb>(stage[warp], lane, b);
#pragma unroll
                for (int mt = 0; mt < MT; ++mt) {
                    uint32_t af[4];
                    a_frags(A[m], mt, s, lane, af);
                    mma16816(acc[m][mt][0], af, b[0], b[1]);
                    mma16816(acc[m][mt][1], af, b[2], b[3]);
                }
            }
        }
    }
    __syncthreads();
    store_acc(acc[0], T[0]);
    store_acc(acc[1], T[1]);
    __syncthreads();
    const uint16_t* svg = (const uint16_t*) (a.wbase + (size_t) e * a.stride + a.svh[0]) + g * 128;
    const uint16_t* svu = (const uint16_t*) (a.wbase + (size_t) e * a.stride + a.svh[1]) + g * 128;
    for (int rr = warp; rr < nrows; rr += kWarps) {
        float vg[4], vu[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            vg[i] = T[0][rr * 128 + lane * 4 + i];
            vu[i] = T[1][rr * 128 + lane * 4 + i];
        }
        had128(vg, lane);
        had128(vu, lane);
        __half* hr = a.h + (size_t) (r0 + rr) * a.ldh + g * 128 + lane * 4;
#pragma unroll
        for (int i = 0; i < 4; ++i)
            hr[i] = __float2half_rn(swiglu(__fmul_rn(vg[i], h2f(svg[lane * 4 + i])),
                                           __fmul_rn(vu[i], h2f(svu[lane * 4 + i])), a.limit));
    }
}

}  // namespace tc
#endif

// ---------------------------------------------------------------- dispatch
// F(K2, cb) for K 1..8 (K2 = 2 K) and the three codebooks, and for the half-integer K 1.5 / 2.5 / 3.5 (K2 3, 5, 7) with
// mul1 - exllamav3 1.6 has no other codebook for them; another pair is refused (false: nothing launched)
template <template <int, int> class F, typename... A>
bool dispatch(int K2, int cb, A&&... args) {
#define EXL3_CASE(KK)                                                    \
    case KK:                                                             \
        if (cb == 2) F<KK, 2>::run(std::forward<A>(args)...);            \
        else if (cb == 1) F<KK, 1>::run(std::forward<A>(args)...);       \
        else F<KK, 0>::run(std::forward<A>(args)...);                    \
        return true;
#define EXL3_HALF(KK)                                                    \
    case KK:                                                             \
        if (cb != 2) return false;                                       \
        F<KK, 2>::run(std::forward<A>(args)...);                         \
        return true;
    if (cb < 0 || cb > 2) return false;
    switch (K2) {
        EXL3_CASE(2) EXL3_CASE(4) EXL3_CASE(6) EXL3_CASE(8) EXL3_CASE(10) EXL3_CASE(12) EXL3_CASE(14) EXL3_CASE(16)
        EXL3_HALF(3) EXL3_HALF(5) EXL3_HALF(7)
        default: return false;
    }
#undef EXL3_HALF
#undef EXL3_CASE
}

template <int K2, int cb> struct RunGroupGU {
    static void run(const KGroup& a, int groups, cudaStream_t s) {
        group_gu_kernel<K2, cb><<<dim3((unsigned) (a.n_ff / kCols), (unsigned) groups, 2u), kThreads, 0, s>>>(a);
    }
};
template <int K2, int cb> struct RunGroupDown {
    static void run(const KGroup& a, int groups, cudaStream_t s) {
        group_down_kernel<K2, cb><<<dim3((unsigned) (a.n_embd / kCols), (unsigned) groups), kThreads, 0, s>>>(a);
    }
};
#if !defined(STRATA_USE_HIP)
template <int K2, int cb> struct RunGemm {
    static void run(const tc::GArgs& a, dim3 grid, cudaStream_t s) { tc::gemm_kernel<K2, cb><<<grid, kThreads, 0, s>>>(a); }
};
template <int K2, int cb> struct RunGateUp {
    static void run(const tc::GArgs& a, dim3 grid, cudaStream_t s) {
        tc::gate_up_kernel<K2, cb><<<grid, kThreads, 0, s>>>(a);
    }
};
#endif
template <int K2, int cb> struct RunInnerRows {
    static void run(const Mat& m, uint16_t* dst, cudaStream_t s) {
        inner_rows_kernel<K2, cb><<<dim3((unsigned) (m.n / kCols), (unsigned) (m.k / kCols)), kThreads, 0, s>>>(
            m.trellis, m.ld > 0 ? m.ld : m.n / 16, m.k, dst);
    }
};
template <int K2, int cb> struct RunInner {
    static void run(const Mat& m, uint16_t* dst, cudaStream_t s) {
        recon_inner_kernel<K2, cb><<<dim3((unsigned) (m.n / kCols), (unsigned) (m.k / 16)), kThreads, 0, s>>>(
            m.trellis, m.ld > 0 ? m.ld : m.n / 16, m.n, dst);
    }
};

void count_error(const char* what) {
    std::fprintf(stderr, "exl3_dev: %s\n", what);
    g_launch_errors.fetch_add(1, std::memory_order_relaxed);
}

}  // namespace

bool k2_supported(int k2, int cb) {
    return ((k2 >= 2 && k2 <= 16 && (k2 & 1) == 0) && cb >= 0 && cb <= 2) || ((k2 == 3 || k2 == 5 || k2 == 7) && cb == 2);
}

bool supported(const Mat& m) {
    return m.trellis != nullptr && m.suh != nullptr && m.svh != nullptr && m.k > 0 && m.n > 0 && m.k % 128 == 0 &&
           m.n % 128 == 0 && k2_supported(m.k2, m.cb);
}

bool device_ok() {
#if defined(STRATA_USE_HIP)
    return false;
#else
    int dev = 0, major = 0;
    if (cudaGetDevice(&dev) != cudaSuccess ||
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    return major >= 8;
#endif
}

int launch_errors() { return g_launch_errors.load(std::memory_order_relaxed); }

size_t group_counters(int max_groups, int n_ff) { return (size_t) std::max(0, max_groups) * (size_t) (n_ff / kCols); }

bool grouped_experts(const GroupArgs& a, void* stream) {
    if (a.max_groups < 1 || a.max_entries < 1 || a.n_embd % kCols != 0 || a.n_ff % kCols != 0 || a.n_embd > kMaxKR ||
        a.n_ff > kMaxKR || a.lay.k2[0] != a.lay.k2[1] || a.grp_ptr == nullptr || a.grp_start == nullptr ||
        a.n_groups == nullptr || a.ent_tok == nullptr || a.ent_dst == nullptr || a.x == nullptr || a.out == nullptr ||
        a.h == nullptr || a.u == nullptr || a.cnt == nullptr) {
        std::fprintf(stderr, "exl3_dev grouped_experts: not covered (n_embd %d n_ff %d, gate k2 %d up k2 %d, %d groups, "
                             "%d entries, or a buffer missing)\n", a.n_embd, a.n_ff, a.lay.k2[0], a.lay.k2[1], a.max_groups,
                     a.max_entries);
        g_launch_errors.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    KGroup k{};
    k.grp_ptr = a.grp_ptr;
    k.grp_start = a.grp_start;
    k.n_groups = a.n_groups;
    k.ent_tok = a.ent_tok;
    k.ent_dst = a.ent_dst;
    k.x = a.x;
    k.x_stride = a.x_stride;
    k.h = a.h;
    k.u = a.u;
    k.out = a.out;
    k.out_stride = a.out_stride;
    k.cnt = a.cnt;
    for (int r = 0; r < 3; ++r) {
        k.suh[r] = a.lay.suh[r];
        k.svh[r] = a.lay.svh[r];
        k.tr[r] = a.lay.trellis[r];
    }
    k.n_embd = a.n_embd;
    k.n_ff = a.n_ff;
    k.limit = a.limit;
    const cudaStream_t s = (cudaStream_t) stream;
    if (!k2_supported(a.lay.k2[0], a.lay.cb) || !k2_supported(a.lay.k2[2], a.lay.cb)) {   // nothing launched
        count_error("grouped_experts: a bitrate or codebook the kernels do not take (K 1..8 any codebook, 1.5 / 2.5 / 3.5 "
                    "mul1)");
        return false;
    }
    if (!dispatch<RunGroupGU>(a.lay.k2[0], a.lay.cb, k, a.max_groups, s) ||
        !dispatch<RunGroupDown>(a.lay.k2[2], a.lay.cb, k, a.max_groups, s)) {
        count_error("grouped_experts: dispatch");
        return false;
    }
    launch_check("grouped_experts");
    return true;
}

void set_gate_up(const SetArgs& sa, const float* x, int ldx, const int32_t* row_idx, uint16_t* h16, int ldh,
                 void* stream) {
#if !defined(STRATA_USE_HIP)
    if (sa.lay.k2[0] != sa.lay.k2[1]) {
        count_error("set_gate_up: gate and up differ in K");
        return;
    }
    if (sa.n_exp <= 0 || sa.max_rows <= 0) return;
    tc::GArgs a{};
    a.wbase = sa.wbase;
    a.stride = sa.stride;
    for (int r = 0; r < 2; ++r) {
        a.tr[r] = sa.lay.trellis[r];
        a.suh[r] = sa.lay.suh[r];
        a.svh[r] = sa.lay.svh[r];
        a.ld[r] = sa.n_ff / 16;
    }
    a.bounds = sa.bounds;
    a.row0 = sa.row0;
    a.k = sa.n_embd;
    a.n = sa.n_ff;
    a.xf = x;
    a.idx = row_idx;
    a.ldx = ldx;
    a.h = (__half*) h16;
    a.ldh = ldh;
    a.limit = sa.limit;
    const dim3 grid((unsigned) ((sa.max_rows + tc::BM - 1) / tc::BM), (unsigned) (sa.n_ff / kCols), (unsigned) sa.n_exp);
    if (!dispatch<RunGateUp>(sa.lay.k2[0], sa.lay.cb, a, grid, (cudaStream_t) stream)) count_error("set_gate_up: K");
    launch_check("set_gate_up");
#else
    (void) sa; (void) x; (void) ldx; (void) row_idx; (void) h16; (void) ldh; (void) stream;
    count_error("set_gate_up: no tensor-core kernels in this build");
#endif
}

void set_down(const SetArgs& sa, const uint16_t* h16, int ldh, float* y, int ldy, void* stream) {
#if !defined(STRATA_USE_HIP)
    if (sa.n_exp <= 0 || sa.max_rows <= 0) return;
    tc::GArgs a{};
    a.wbase = sa.wbase;
    a.stride = sa.stride;
    a.tr[0] = sa.lay.trellis[2];
    a.suh[0] = sa.lay.suh[2];
    a.svh[0] = sa.lay.svh[2];
    a.ld[0] = sa.n_embd / 16;
    a.bounds = sa.bounds;
    a.row0 = sa.row0;
    a.k = sa.n_ff;
    a.n = sa.n_embd;
    a.xh = (const __half*) h16;
    a.ldx = ldh;
    a.y = y;
    a.ldy = ldy;
    const dim3 grid((unsigned) ((sa.max_rows + tc::BM - 1) / tc::BM), (unsigned) (sa.n_embd / kCols), (unsigned) sa.n_exp);
    if (!dispatch<RunGemm>(sa.lay.k2[2], sa.lay.cb, a, grid, (cudaStream_t) stream)) count_error("set_down: K");
    launch_check("set_down");
#else
    (void) sa; (void) h16; (void) ldh; (void) y; (void) ldy; (void) stream;
    count_error("set_down: no tensor-core kernels in this build");
#endif
}

void reconstruct_inner_rows(const Mat& m, uint16_t* dst, void* stream) {
    if (!supported(m) || !dispatch<RunInnerRows>(m.k2, m.cb, m, dst, (cudaStream_t) stream)) {
        count_error("reconstruct_inner_rows: unsupported matrix");
        return;
    }
    launch_check("reconstruct_inner_rows");
}

void rows_rot_f16(const float* x, int ldx, const int32_t* idx, int nrows, int ncols, const uint16_t* suh, uint16_t* out,
                  int ldo, void* stream) {
    if (nrows <= 0) return;
    if (ncols % kCols != 0) {
        count_error("rows_rot_f16: ncols is not a multiple of 128");
        return;
    }
    const long long warps = (long long) nrows * (ncols / kCols);
    rows_rot_f16_kernel<<<(unsigned) ((warps + kWarps - 1) / kWarps), kThreads, 0, (cudaStream_t) stream>>>(
        x, ldx, idx, nrows, ncols, suh, out, ldo);
    launch_check("rows_rot_f16");
}

void rows_rot_out(float* y, int ldy, int nrows, int ncols, const uint16_t* svh, void* stream) {
    if (nrows <= 0) return;
    if (ncols % kCols != 0) {
        count_error("rows_rot_out: ncols is not a multiple of 128");
        return;
    }
    const long long warps = (long long) nrows * (ncols / kCols);
    rows_rot_out_kernel<<<(unsigned) ((warps + kWarps - 1) / kWarps), kThreads, 0, (cudaStream_t) stream>>>(
        y, ldy, nrows, ncols, svh);
    launch_check("rows_rot_out");
}

void gu_epilogue(const float* gu, int ldgu, int nrows, int n_ff, const uint16_t* svh_g, const uint16_t* svh_u,
                 const uint16_t* suh_d, float limit, uint16_t* h16, int ldh, void* stream) {
    if (nrows <= 0) return;
    if (n_ff % kCols != 0) {
        count_error("gu_epilogue: n_ff is not a multiple of 128");
        return;
    }
    const long long warps = (long long) nrows * (n_ff / kCols);
    gu_epilogue_kernel<<<(unsigned) ((warps + kWarps - 1) / kWarps), kThreads, 0, (cudaStream_t) stream>>>(
        gu, ldgu, nrows, n_ff, svh_g, svh_u, suh_d, limit, h16, ldh);
    launch_check("gu_epilogue");
}

void reconstruct_inner(const Mat& m, uint16_t* dst, void* stream) {
    if (!supported(m) || !dispatch<RunInner>(m.k2, m.cb, m, dst, (cudaStream_t) stream)) {
        count_error("reconstruct_inner: unsupported matrix");
        return;
    }
    launch_check("reconstruct_inner");
}

}  // namespace strata::kernels::exl3
