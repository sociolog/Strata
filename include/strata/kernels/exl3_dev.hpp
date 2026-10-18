// include/strata/kernels/exl3_dev.hpp - EXL3 routed experts on the device, for the two batched paths of an --exl3
// model: the verify window's experts held in VRAM (each expert read and decoded once for every window token that
// routes to it) and the prompt path's expert step (the decoded W' for cuBLAS, the rows' rotations, and tensor-core
// kernels for the experts few of a chunk's rows route to).
//
// Ported from project-maya v1.1.15 (src/kernels/cuda/exl3.cu, MIT, Peasant Smith and the Project Maya contributors,
// whose codebook and trellis arithmetic follow exllamav3, MIT, turboderp).  The format (docs/EXL3.md): a weight with k
// inputs and n outputs (y = x W) is stored as
//   trellis  int16 [k/16][n/16][16 K]: each 16x16 tile a tail-biting trellis of 256 K-bit steps (8 K uint32 words);
//            K 1..8, or exllamav3 1.6's half-integer 1.5 / 2.5 / 3.5 (KA- and KA+1-bit steps alternating, mul1 only),
//            carried as k2 = 2 K;
//   suh, svh FP16 [k], [n]: the input and output channel scales.
// With H the 128-point Sylvester Hadamard over sqrt(128), applied to each 128 consecutive values:
//   y = H(H(x * suh) W') * svh,   W' the decoded k x n matrix (FP16 values: exllamav3's own decode, one FP16 FMA).
// The activations are rotated into FP16 and multiplied on the tensor cores (sm_80 and later; elsewhere F32 FMAs on the
// CUDA cores), F32 accumulation.  An activation's sums never depend on the other activations a launch takes, so a
// verify window's token gets, bit for bit, what the same kernel gives that token alone.
//
// The stream arguments are cudaStream_t (void* here, as in exl3.hpp, so CPU-side files can include this header).
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::kernels::exl3 {

/// One EXL3 matrix on the device (or a column range of one: trellis/svh advanced to its first column, ld the full
/// matrix's tiles per 16-row slice).
struct Mat {
    const uint32_t* trellis = nullptr;   // 4-byte aligned
    const uint16_t* suh = nullptr;       // FP16 bits [k]
    const uint16_t* svh = nullptr;       // FP16 bits [n]
    int k = 0, n = 0;
    int k2 = 0;                          // bits per weight times 2: 2..16 even (K 1..8), 3 / 5 / 7 (K 1.5 / 2.5 / 3.5)
    int cb = 2;                          // codebook: 0 3inst, 1 mcg, 2 mul1
    int ld = 0;                          // tiles per 16-row slice in memory; 0: n / 16
};
/// Whether the kernels take this matrix (k, n multiples of 128, k2 2..16 even with cb 0..2, or 3 / 5 / 7 with mul1).
bool supported(const Mat& m);
/// Whether the kernels take this bitrate and codebook.
bool k2_supported(int k2, int cb);
/// Trellis bytes of a k x n matrix at k2 / 2 bits (16 k2 a tile).  (Was trellis_bytes(k, n, K): renamed with the k2
/// argument so a call written for K does not compile - it would size half the matrix.)
inline size_t trellis_bytes_k2(int64_t k, int64_t n, int k2) { return (size_t) (k / 16) * (size_t) (n / 16) * 16u * k2; }

/// One routed expert's blob: gate (0), up (1), down (2), each its trellis, suh and svh at byte offsets from the blob's
/// base (the CPU lane's RAM tier keeps exactly such blobs, every part 64-byte aligned), and each matrix's bitrate.
struct ExpertLayout {
    size_t suh[3] = {0, 0, 0}, svh[3] = {0, 0, 0}, trellis[3] = {0, 0, 0};
    int k2[3] = {0, 0, 0};               // as Mat::k2
    int cb = 2;
};

/// Whether the current device has the tensor-core kernels these paths are built on (sm_80 and later).
bool device_ok();

// ---------------------------------------------------------------- grouped experts (the verify windows, the tier)
/// The routed experts of a verify window that sit in VRAM, grouped by expert as the window's GPU plan lists them
/// (core::GpuPlanSink's VRAM groups): group g's blob at grp_ptr[g], its entries [grp_start[g], grp_start[g + 1]); entry m
/// reads token ent_tok[m]'s input (x + ent_tok[m] * x_stride, F32 [n_embd]) and writes its UNWEIGHTED row
/// down(swiglu(gate(x), up(x))) to out + ent_dst[m] * out_stride.  Each group's weights are read and decoded once for up
/// to 8 of its entries at a time; every entry's row is bit for bit what the same launch gives it alone.  The group
/// count is read from n_groups (device) when the kernels run: a launch inside a captured graph is sized for max_groups
/// and its blocks past the real count return at once.
struct GroupArgs {
    const unsigned long long* grp_ptr = nullptr;
    const int32_t* grp_start = nullptr;
    const int32_t* n_groups = nullptr;
    const int32_t* ent_tok = nullptr;
    const int32_t* ent_dst = nullptr;
    const float* x = nullptr;
    size_t x_stride = 0;                 // floats
    float* out = nullptr;
    size_t out_stride = 0;               // floats
    float* h = nullptr;                  // scratch [max_entries][n_ff]: the gate's values, then the swiglu
    float* u = nullptr;                  // scratch [max_entries][n_ff]: the up's values
    unsigned* cnt = nullptr;             // scratch: group_counters() counters, zero (the kernels leave them zero)
    ExpertLayout lay;
    int max_groups = 0, max_entries = 0;
    int n_embd = 0, n_ff = 0;
    float limit = 0.0f;                  // clamped swiglu (GLM); 0: plain swiglu (Qwen)
};
/// The counters GroupArgs::cnt must hold.
size_t group_counters(int max_groups, int n_ff);
/// Launches gate/up and down for the groups.  false (nothing launched) when the geometry is not covered.
bool grouped_experts(const GroupArgs& a, void* stream);

// ---------------------------------------------------------------- the prompt path
/// A set of n_exp routed experts in VRAM, expert e's blob at wbase + e * stride, owning the sorted rows
/// [bounds[e], bounds[e + 1]) (device, n_exp + 1 entries); a sorted row's LOCAL row is bounds[e] - row0 + r.
struct SetArgs {
    const uint8_t* wbase = nullptr;
    size_t stride = 0;
    ExpertLayout lay;
    int n_exp = 0;
    const int32_t* bounds = nullptr;
    int row0 = 0;
    int max_rows = 0;   // the most rows one expert of the set has
    int n_embd = 0, n_ff = 0;
    float limit = 0.0f;
};
/// h[local row] = FP16 swiglu of gate and up over x[row_idx[local row]] (F32 rows, ldx floats apart): the tensor-core
/// kernels, which decode the trellis once a 32-row block (sm_80+; a no-op elsewhere - check device_ok()).
void set_gate_up(const SetArgs& a, const float* x, int ldx, const int32_t* row_idx, uint16_t* h16, int ldh, void* stream);
/// y[local row] = down(h[local row]) (F32 rows, ldy floats apart).
void set_down(const SetArgs& a, const uint16_t* h16, int ldh, float* y, int ldy, void* stream);

/// The decoded W' alone in the [n][k] row layout a cuBLAS product with W transposed reads: dst[c][r] = W'[r][c], FP16
/// bits (exactly the decoded values).  For a product that rotates its activations and outputs instead of the weight
/// (x W = H(H(x * suh) W') * svh): rows_rot_f16 before the GEMM, rows_rot_out (or gu_epilogue) after it.
void reconstruct_inner_rows(const Mat& m, uint16_t* dst, void* stream);
/// out[r][c] = FP16(H(x_r * suh)[c]) for r < nrows, c < ncols (ncols % 128 == 0), x_r = x[idx ? idx[r] : r] (F32, ldx
/// floats a row; ldo halves a row of out).
void rows_rot_f16(const float* x, int ldx, const int32_t* idx, int nrows, int ncols, const uint16_t* suh, uint16_t* out,
                  int ldo, void* stream);
/// y[r][c] = H(y_r)[c] * svh[c] in place for r < nrows, c < ncols (each 128 values one block; ldy floats a row).
void rows_rot_out(float* y, int ldy, int nrows, int ncols, const uint16_t* svh, void* stream);
/// The gate/up products' epilogue and the down product's input in one pass: for each row r, its gate g = gu[r][0, n_ff)
/// and up u = gu[r][n_ff, 2 n_ff) (the raw products H(x * suh) W'), h = swiglu(H(g) * svh_g, H(u) * svh_u), and
/// h16[r] = FP16(H(h * suh_d)) - rows_rot_out on both halves, the swiglu and rows_rot_f16, the same arithmetic.
void gu_epilogue(const float* gu, int ldgu, int nrows, int n_ff, const uint16_t* svh_g, const uint16_t* svh_u,
                 const uint16_t* suh_d, float limit, uint16_t* h16, int ldh, void* stream);

/// The decoded W' alone, row-major [k][n] FP16 bits (tests).
void reconstruct_inner(const Mat& m, uint16_t* dst, void* stream);

/// Launch failures of this family since the process started (each is also printed).
int launch_errors();

}  // namespace strata::kernels::exl3
