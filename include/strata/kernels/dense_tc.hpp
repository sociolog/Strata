// include/strata/kernels/dense_tc.hpp - the dense Q8_0 projections of a verify window on int8 tensor cores
// (opt-in STRATA_DENSE_TC, see src/core/verify.cpp).
//
// Ported from Basalt (github.com/jesdga95/basalt @ fb0944c: include/basalt/kernels/dense_tc.hpp and the ring kernel of
// src/kernels/cuda/dense_tc.cu).  Copyright (c) 2026 Jesús Gulfo, MIT License; the full notice is in
// third_party/basalt/LICENSE.
//
// y[c][r] = sum over row r's 32-value blocks of d_w * d_x(c) * dot, where dot is the block's exact int32 dot of the
// weight codes with column c's q8_1 codes (one mma.sync m16n8k32 per block, 16 rows x 8 columns).  The blocks are
// summed in one order fixed by the shape (n_in, n_out) alone, so every column's value is bitwise the same for every
// column count 1..8 and every position in the window.  Not bitwise native_mmvq (which rounds per 8-value quarter
// block); both are within float rounding of the exact sum.
//
// Weights: GGUF Q8_0 row-major blocks (34 bytes per 32 values), 4-byte aligned.  x_q8_1: ncols columns of n_in / 32
// q8_1 blocks (native_quantize_q8_1's layout), 4-byte aligned.  y: column-major, column c at y + c * n_out.
// n_in must be a multiple of 512.  sm_80+ (cp.async, mma.sync m16n8k32 s8); callers check dense_tc_q8_0_ok.
#pragma once

namespace strata::kernels {

/// The current device runs the kernel (compute capability >= 8.0) and n_in is a multiple of 512.
bool dense_tc_q8_0_ok(int n_in);

void dense_tc_q8_0(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols, void* stream);

/// Benchmark / test only: a fixed launch shape (cfg: 0 = the shape's default; 1..N the variants listed by
/// dense_tc_q8_0_cfg_name).  Returns false when cfg is out of range.
bool dense_tc_q8_0_cfg(int cfg, const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols,
                       void* stream);
const char* dense_tc_q8_0_cfg_name(int cfg);

}  // namespace strata::kernels
