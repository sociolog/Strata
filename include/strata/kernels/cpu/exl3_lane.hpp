// include/strata/kernels/cpu/exl3_lane.hpp - EXL3 routed experts computed on the CPU (the CPU pool's EXL3 kernel).
//
// Ported from project-maya's CPU lane (include/strata/kernels/cpu/exl3_cpu.hpp, MIT), where it runs GLM-5.3-Flash's
// EXL3 experts; here it is the CPU tier of Qwen3.8-Flash-Next's EXL3 experts.  The scalar oracle beside it
// (strata/kernels/cpu/exl3.hpp) materializes the decoded matrix; this one never does.
//
// The mul1 codebook is affine in the byte sum s of state * 0x83DCD12D: w = (1024 + s) * k_inv + k_bias, which is the
// GPU kernel's fp32 fold (src/kernels/cuda/exl3.cu, decode_mul1).  The CPU keeps the affine form, so a 128-output band
// is
//   y' = k_inv * sum_k xh_k s_kc + (1024 k_inv + k_bias) * sum_k xh_k,   xh = H(x * suh)
// and y = H(y') * svh.  The byte sums come from AVX-512 (permutes and funnel shifts pull 16 columns' states at once,
// vpmaddubsw / vpmaddwd add the bytes up, an F32 FMA takes the activation), or from AVX2 + FMA the same way 8 columns
// at a time - the same FMAs in the same order, so the two give the same bits; a CPU with neither runs a scalar loop.
//
// Layout: the trellis is #1764's [k/16][n/16][16 K] int16 array (Exl3Linear::trellis), read as 8 K little-endian
// uint32 words a tile - the same bytes.  Tile (i, j) holds W_hat rows 16 i.. and columns 16 j..; the stored position
// t of (row, column) is the tensor-core order of build_perm (src/kernels/cpu/exl3.cpp), the window of t the 16 bits
// ending at bit S(t) + 256 K of the tile's word ring, S(t) = floor((t + 1) k2 / 2) with k2 = 2 K ((t + 1) K for an
// integer K; exllamav3 1.6's half-integer K 1.5 / 2.5 / 3.5 too - window16 there).  exl3_lane_test proves both
// against the oracle tile by tile.  See docs/EXL3.md for the format.
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::kernels::cpu {

/// One EXL3 matrix (mul1 codebook): k inputs, n outputs (both multiples of 128), the bitrate as k2 = 2 K: 2..16 even
/// (K 1..8) or 3 / 5 / 7 (K 1.5 / 2.5 / 3.5).
struct Exl3Part {
    const uint32_t* trellis = nullptr;   // [k/16][n/16][4 k2] words
    const uint16_t* suh = nullptr;       // FP16 bits [k]
    const uint16_t* svh = nullptr;       // FP16 bits [n]
    int k = 0, n = 0, k2 = 0;
};
/// Whether the lane takes this bitrate (k2 2..16 even, or 3 / 5 / 7).
inline bool exl3_lane_k2_ok(int k2) { return (k2 >= 2 && k2 <= 16 && (k2 & 1) == 0) || k2 == 3 || k2 == 5 || k2 == 7; }
/// Trellis bytes of a matrix: 16 k2 a 16x16 tile.
inline size_t exl3_part_trellis_bytes(const Exl3Part& m) { return (size_t) (m.k / 16) * (size_t) (m.n / 16) * 16u * (size_t) m.k2; }
/// Whether a vector kernel runs here (AVX-512F/BW or AVX2 + FMA); the scalar one otherwise.
bool exl3_cpu_vector();
/// The kernel the functions below run here: 2 AVX-512 (F, BW), 1 AVX2 + FMA (+ F16C), 0 scalar.  STRATA_FORCE_AVX2=1
/// keeps an AVX-512 CPU on AVX2 (to compare the two; they give the same bits).
int exl3_cpu_tier();
/// The bit mask of the tiers this CPU can run (bit t: tier t), whatever exl3_cpu_tier() picked.
int exl3_cpu_tier_mask();
/// xh = H(x * suh) (k values, k % 128 == 0); returns the sum of xh.
float exl3_cpu_prepare(const Exl3Part& m, const float* x, float* xh);
/// One 128-value block of that: xh[i] = H(x * suh)[i] for i < 128 (suh: the block's 128 scales); returns its sum.
float exl3_cpu_prepare_block(const uint16_t* suh, const float* x, float* xh);
/// The 128 outputs from column c0 (a multiple of 128) of y = H(x W') * svh, from exl3_cpu_prepare's xh and sum.
void exl3_cpu_band(const Exl3Part& m, const float* xh, float xsum, int c0, float* y);
/// The same in parts: the byte-sum products of the 16-row slices [ks0, ks1) (xh from slice 0) ...
void exl3_cpu_band_acc(const Exl3Part& m, const float* xh, int c0, int ks0, int ks1, float* acc);
/// exl3_cpu_band_acc for n activations of the same matrix at once (xh[i] -> acc[i], each 128 floats): each tile row is
/// decoded once for up to kExl3CpuMaxN of them, and acc[i] is exl3_cpu_band_acc's for xh[i] alone, bit for bit.
constexpr int kExl3CpuMaxN = 8;
void exl3_cpu_band_acc_n(const Exl3Part& m, const float* const* xh, int n, int c0, int ks0, int ks1,
                         float* const* acc);
/// The byte-sum products of the slices [ks0, ks1) for EVERY column (acc: m.n floats, overwritten) - a split-K piece,
/// whose tile rows lie in memory order where a band's are 1 KB runs a whole tile row apart (from DRAM, ~1.5x the
/// bands' speed).  Each column's sums are exl3_cpu_band_acc's over the same slices, bit for bit.
void exl3_cpu_rows_acc(const Exl3Part& m, const float* xh, int ks0, int ks1, float* acc);
/// ... for n activations at once (xh[i] -> acc[i]), each tile row decoded once; acc[i] is the call for xh[i] alone.
void exl3_cpu_rows_acc_n(const Exl3Part& m, const float* const* xh, int n, int ks0, int ks1, float* const* acc);
/// The byte-sum products of nb bands from c0 (128 nb columns; acc[i]: 128 nb floats) over every input for any n
/// activations, as a GEMM: each 256-input chunk of a band decoded once into F32 in `wc` (exl3_cpu_gemm_scratch()
/// floats, 64-byte aligned, the caller's), then multiplied for all n, the chunk of the activations kept in L2 across
/// the nb bands - for many tokens an expert (the prompt path), where the band kernels decode a tile row again every
/// kExl3CpuMaxN.  The same sums as exl3_cpu_band_acc, in its order.
size_t exl3_cpu_gemm_scratch();
void exl3_cpu_band_gemm(const Exl3Part& m, const float* const* xh, int n, int c0, int nb, float* const* acc,
                        float* wc);
/// exl3_cpu_band_acc_n on a given tier (0, 1, 2 as exl3_cpu_tier) - tests and benchmarks; false (nothing computed)
/// where this CPU cannot run it.
bool exl3_cpu_band_acc_on(int tier, const Exl3Part& m, const float* const* xh, int n, int c0, int ks0, int ks1,
                          float* const* acc);
/// exl3_cpu_rows_acc_n on a given tier, as exl3_cpu_band_acc_on.
bool exl3_cpu_rows_acc_on(int tier, const Exl3Part& m, const float* const* xh, int n, int ks0, int ks1,
                          float* const* acc);
/// ... and, with the parts' acc summed over every slice, the outputs (the codebook's affine form, H, svh).
void exl3_cpu_band_out(const Exl3Part& m, const float* acc, float xsum, int c0, float* y);

}  // namespace strata::kernels::cpu
