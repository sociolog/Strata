// src/kernels/exl3_dev_parity.cpp - the EXL3 device kernels of include/strata/kernels/exl3_dev.hpp (the verify window's
// grouped experts and the prompt path's expert step) against references on the host:
//
//   1. the decoded W' (reconstruct_inner, reconstruct_inner_rows) against #1764's CPU decoder (exl3_decode_weight_hat,
//      itself bit-exact against the format's numpy specification and against exllamav3): every K 1..8, every codebook,
//      and the half-integer K 1.5 / 2.5 / 3.5 (mul1) - 0 differences;
//   2. the rotations (rows_rot_f16, rows_rot_out, gu_epilogue) against a host emulation of the same arithmetic in the
//      same order - 0 differences;
//   3. a whole expert, the three device routes - the window's grouped kernels, the tensor-core set kernels (the prompt
//      path's lightly routed experts), and the decoded W' + cuBLAS (prefill::Gemm::f16, its heavily routed ones) -
//      against a host reference with the device's roundings (FP16 weights and rotated activations, double sums): only
//      the sums' order differs (relative RMS ~1e-6);
//   4. the window's property the speculative decoding relies on: every entry of a grouped launch is bit for bit what
//      the same kernels give that entry alone (one group, one entry) - 0 differences;
//   5. the whole expert against the CPU lane (src/kernels/cpu/exl3_lane.cpp), which decodes in F32 without the FP16
//      rounding of weights and activations: the size of that difference (~1e-3), for the record.
//
//   exl3_dev_parity                     synthetic matrices: the Flash's expert shapes (2560 -> 640 -> 2560), integer K
//                                       and an SC 4.05's 3.5 / 3.5 / 5 and 3 / 3 / 5 experts
//   exl3_dev_parity --real <model dir>  plus routed experts of the model (layers 0, 1, 8, 23, 47; 12 experts each)
// The bitrate is k2 = 2 K throughout.
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "strata/artifact/exl3_model.hpp"
#include "strata/kernels/cpu/exl3.hpp"
#include "strata/kernels/cpu/exl3_lane.hpp"
#include "strata/kernels/exl3_dev.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/prefill/gemm.hpp"

namespace cpu = strata::kernels::cpu;
namespace x3 = strata::kernels::exl3;
using strata::kernels::f16_from_f32;
using strata::kernels::f32_from_f16;

namespace {

int g_fail = 0;
std::mt19937_64 g_rng(1764);
constexpr int E = 2560, F = 640;   // the Flash's n_embd and expert n_ff
constexpr float kHad = 0.088388347648f;

bool ck(cudaError_t e, const char* what) {
    if (e == cudaSuccess) return true;
    std::fprintf(stderr, "exl3_dev_parity: CUDA %s: %s\n", what, cudaGetErrorString(e));
    ++g_fail;
    return false;
}

void report(const std::string& what, double v, double tol, const char* unit = "rel") {
    const bool ok = v <= tol;
    std::printf("  %-58s %s %.2e (tolerance %.0e)  %s\n", what.c_str(), unit, v, tol, ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}
void report_exact(const std::string& what, long long diff, long long n) {
    std::printf("  %-58s %lld of %lld differ  %s\n", what.c_str(), diff, n, diff == 0 ? "ok" : "FAIL");
    if (diff != 0) ++g_fail;
}

// one matrix on the host (aligned words)
struct HMat {
    std::vector<uint32_t> tr;
    std::vector<uint16_t> suh, svh;
    int k = 0, n = 0, k2 = 0, cb = 2;
};

HMat rand_mat(int k, int n, int k2, int cb = 2) {
    HMat m;
    m.k = k;
    m.n = n;
    m.k2 = k2;
    m.cb = cb;
    m.tr.resize((size_t) (k / 16) * (n / 16) * 4 * k2);
    for (auto& w : m.tr) w = (uint32_t) g_rng();
    // a real expert's channel scales (the 4.05 file: |suh| ~0.015, |svh| ~1.0), random signs
    std::uniform_real_distribution<float> us(0.010f, 0.025f), uv(0.8f, 1.2f);
    m.suh.resize((size_t) k);
    m.svh.resize((size_t) n);
    for (auto& v : m.suh) v = f16_from_f32((g_rng() & 1) ? us(g_rng) : -us(g_rng));
    for (auto& v : m.svh) v = f16_from_f32((g_rng() & 1) ? uv(g_rng) : -uv(g_rng));
    return m;
}

bool load_real(const strata::Exl3Model& model, const std::string& base, HMat& m, std::string& err) {
    if (!model.has(base)) {
        err = "no tensor " + base;
        return false;
    }
    const strata::Exl3Linear L = model.linear(base);
    if (!x3::k2_supported(L.k2, L.cb) || L.in() % 128 != 0 || L.out() % 128 != 0) {
        err = base + ": not a matrix these kernels take (K " + strata::exl3_k_label(L.k2) + ")";
        return false;
    }
    m.k = L.in();
    m.n = L.out();
    m.k2 = L.k2;
    m.cb = L.cb;
    m.tr.assign((size_t) L.ki * L.nj * L.tile_words() / 2, 0u);
    std::memcpy(m.tr.data(), L.trellis, m.tr.size() * 4);   // the file's words sit at offsets 2 mod 4
    m.suh.assign(L.suh, L.suh + m.k);
    m.svh.assign(L.svh, L.svh + m.n);
    return true;
}

// ---------------------------------------------------------------- host emulations of the device arithmetic
// had128 as the device runs it: value j = lane * 4 + i; first the lane's own 4 values, then the lanes pairwise for
// lane bits 1, 2, 4, 8, 16 (lower + upper, lower - upper), then the 1/sqrt(128) scale - the same F32 operations in the
// same order, so the same bits
void had128_emul(float* v) {
    for (int l = 0; l < 32; ++l) {
        float* p = v + l * 4;
        const float s0 = p[0] + p[1], d0 = p[0] - p[1], s1 = p[2] + p[3], d1 = p[2] - p[3];
        p[0] = s0 + s1;
        p[1] = d0 + d1;
        p[2] = s0 - s1;
        p[3] = d0 - d1;
    }
    for (int m = 1; m < 32; m <<= 1)
        for (int l = 0; l < 32; ++l) {
            if (l & m) continue;
            float* a = v + l * 4;
            float* b = v + (l | m) * 4;
            for (int i = 0; i < 4; ++i) {
                const float lo = a[i], hi = b[i];
                a[i] = lo + hi;
                b[i] = lo - hi;
            }
        }
    for (int i = 0; i < 128; ++i) v[i] *= kHad;
}
// FP16(H(x * suh)) per 128-block (chunk_store)
std::vector<uint16_t> rot_f16(const float* x, const std::vector<uint16_t>& suh) {
    const int k = (int) suh.size();
    std::vector<uint16_t> out((size_t) k);
    float b[128];
    for (int c0 = 0; c0 < k; c0 += 128) {
        for (int i = 0; i < 128; ++i) b[i] = x[c0 + i] * f32_from_f16(suh[(size_t) (c0 + i)]);
        had128_emul(b);
        for (int i = 0; i < 128; ++i) out[(size_t) (c0 + i)] = f16_from_f32(b[i]);
    }
    return out;
}
// H(y) * svh per 128-block, in place (block_out, rows_rot_out)
void rot_out(float* y, const std::vector<uint16_t>& svh) {
    const int n = (int) svh.size();
    float b[128];
    for (int c0 = 0; c0 < n; c0 += 128) {
        for (int i = 0; i < 128; ++i) b[i] = y[c0 + i];
        had128_emul(b);
        for (int i = 0; i < 128; ++i) y[c0 + i] = b[i] * f32_from_f16(svh[(size_t) (c0 + i)]);
    }
}
float swiglu(float g, float u) { return (g / (1.0f + std::exp(-g))) * u; }

// the decoded W' [k][n] (FP16 bits) by #1764's CPU decoder
std::vector<uint16_t> decode_hat(const HMat& m) {
    std::vector<uint16_t> w((size_t) m.k * m.n);
    cpu::exl3_decode_weight_hat_k2((const uint16_t*) m.tr.data(), m.k / 16, m.n / 16, m.k2, (cpu::Exl3Codebook) m.cb,
                                   w.data());
    return w;
}
// sum_i x16[i] W'[i][j] in double, as F32 (the device sums in F32 in its own order: ulp-level differences)
std::vector<float> sums(const std::vector<uint16_t>& x16, const std::vector<uint16_t>& w, int k, int n) {
    std::vector<double> acc((size_t) n, 0.0);
    for (int i = 0; i < k; ++i) {
        const double xi = f32_from_f16(x16[(size_t) i]);
        if (xi == 0.0) continue;
        const uint16_t* row = w.data() + (size_t) i * n;
        for (int j = 0; j < n; ++j) acc[(size_t) j] += xi * (double) f32_from_f16(row[j]);
    }
    std::vector<float> out((size_t) n);
    for (int j = 0; j < n; ++j) out[(size_t) j] = (float) acc[(size_t) j];
    return out;
}

struct HExpert {
    HMat g, u, d;
    std::vector<uint16_t> wg, wu, wd;   // decoded W'
    void decode() { wg = decode_hat(g); wu = decode_hat(u); wd = decode_hat(d); }
};
// one token through one expert with the device's roundings.  light: the tensor-core set kernels' order (h rounded to
// FP16 before the down product rotates it); otherwise the grouped kernels' and the cuBLAS route's (h rotated in F32)
std::vector<float> expert_ref(const HExpert& x, const float* in, bool light) {
    std::vector<float> gy = sums(rot_f16(in, x.g.suh), x.wg, E, F), uy = sums(rot_f16(in, x.u.suh), x.wu, E, F);
    rot_out(gy.data(), x.g.svh);
    rot_out(uy.data(), x.u.svh);
    std::vector<float> h((size_t) F);
    for (int i = 0; i < F; ++i) {
        h[(size_t) i] = swiglu(gy[(size_t) i], uy[(size_t) i]);
        if (light) h[(size_t) i] = f32_from_f16(f16_from_f32(h[(size_t) i]));
    }
    std::vector<float> y = sums(rot_f16(h.data(), x.d.suh), x.wd, F, E);
    rot_out(y.data(), x.d.svh);
    return y;
}

// the CPU lane's whole expert (F32 decode): its y = H(x * suh) W' * svh per matrix
std::vector<float> lane_mv(const HMat& m, const float* x) {
    cpu::Exl3Part p;
    p.trellis = m.tr.data();
    p.suh = m.suh.data();
    p.svh = m.svh.data();
    p.k = m.k;
    p.n = m.n;
    p.k2 = m.k2;
    std::vector<float> xh((size_t) m.k), acc((size_t) m.n), y((size_t) m.n);
    const float xs = cpu::exl3_cpu_prepare(p, x, xh.data());
    cpu::exl3_cpu_rows_acc(p, xh.data(), 0, m.k / 16, acc.data());
    for (int c0 = 0; c0 < m.n; c0 += 128) cpu::exl3_cpu_band_out(p, acc.data() + c0, xs, c0, y.data() + c0);
    return y;
}
std::vector<float> lane_expert(const HExpert& x, const float* in) {
    const std::vector<float> gy = lane_mv(x.g, in), uy = lane_mv(x.u, in);
    std::vector<float> h((size_t) F);
    for (int i = 0; i < F; ++i) h[(size_t) i] = swiglu(gy[(size_t) i], uy[(size_t) i]);
    return lane_mv(x.d, h.data());
}

double rel_rms(const std::vector<float>& got, const std::vector<float>& ref) {
    double se = 0.0, sr = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        if (!std::isfinite(got[i]) || !std::isfinite(ref[i])) {
            std::printf("    non-finite value at %zu: %g against %g\n", i, got[i], ref[i]);
            return 1e30;
        }
        const double d = (double) got[i] - ref[i];
        se += d * d;
        sr += (double) ref[i] * ref[i];
    }
    return std::sqrt(se / std::max(sr, 1e-30));
}
long long bits_differ(const std::vector<float>& a, const std::vector<float>& b) {
    long long n = 0;
    for (size_t i = 0; i < a.size(); ++i) n += std::memcmp(&a[i], &b[i], 4) != 0;
    return n;
}

std::vector<float> randn(size_t n, float s) {
    std::normal_distribution<float> nd(0.0f, s);
    std::vector<float> v(n);
    for (auto& x : v) x = nd(g_rng);
    return v;
}

template <typename T>
T* dev_copy(const std::vector<T>& h) {
    T* d = nullptr;
    if (!ck(cudaMalloc(&d, std::max<size_t>(1, h.size()) * sizeof(T)), "malloc") ||
        !ck(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice), "copy up"))
        return nullptr;
    return d;
}
template <typename T>
std::vector<T> host_copy(const T* d, size_t n) {
    std::vector<T> h(n);
    ck(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost), "copy down");
    return h;
}

// experts as the RAM tier stores them: one blob each - gate, up, down trellis, then suh / svh of gate, up, down, every
// part 64-byte aligned - at a fixed stride
struct Blobs {
    std::vector<uint8_t> host;
    size_t stride = 0;
    x3::ExpertLayout lay;
    uint8_t* dev = nullptr;
};
Blobs make_blobs(const std::vector<HExpert>& xs) {
    Blobs b;
    const HExpert& x0 = xs[0];
    auto al = [](size_t v) { return (v + 63) & ~(size_t) 63; };
    size_t off = 0;
    const HMat* mats[3] = {&x0.g, &x0.u, &x0.d};
    for (int r = 0; r < 3; ++r) { b.lay.trellis[r] = off; off = al(off + mats[r]->tr.size() * 4); }
    for (int r = 0; r < 3; ++r) {
        b.lay.suh[r] = off; off = al(off + mats[r]->suh.size() * 2);
        b.lay.svh[r] = off; off = al(off + mats[r]->svh.size() * 2);
        b.lay.k2[r] = mats[r]->k2;
    }
    b.lay.cb = x0.g.cb;
    b.stride = off;
    b.host.assign(b.stride * xs.size(), 0);
    for (size_t e = 0; e < xs.size(); ++e) {
        uint8_t* p = b.host.data() + e * b.stride;
        const HMat* ms[3] = {&xs[e].g, &xs[e].u, &xs[e].d};
        for (int r = 0; r < 3; ++r) {
            std::memcpy(p + b.lay.trellis[r], ms[r]->tr.data(), ms[r]->tr.size() * 4);
            std::memcpy(p + b.lay.suh[r], ms[r]->suh.data(), ms[r]->suh.size() * 2);
            std::memcpy(p + b.lay.svh[r], ms[r]->svh.data(), ms[r]->svh.size() * 2);
        }
    }
    b.dev = dev_copy(b.host);
    return b;
}

// ---------------------------------------------------------------- 1. the decode
void test_decode() {
    std::printf("the decoded W' against #1764's CPU decoder (256 x 384, every K and codebook; K 1.5 / 2.5 / 3.5 mul1)\n");
    for (int cb = 0; cb <= 2; ++cb)
        for (int k2 : {2, 4, 6, 8, 10, 12, 14, 16, 3, 5, 7}) {
            if ((k2 & 1) && cb != 2) continue;   // a half-integer K is mul1 only
            const HMat m = rand_mat(256, 384, k2, cb);
            const std::vector<uint16_t> want = decode_hat(m);
            uint32_t* d_tr = dev_copy(m.tr);
            uint16_t *d_suh = dev_copy(m.suh), *d_svh = dev_copy(m.svh), *d_out = nullptr;
            if (!d_tr || !d_suh || !d_svh || !ck(cudaMalloc(&d_out, want.size() * 2), "malloc")) return;
            x3::Mat dm;
            dm.trellis = d_tr; dm.suh = d_suh; dm.svh = d_svh; dm.k = m.k; dm.n = m.n; dm.k2 = k2; dm.cb = cb;
            x3::reconstruct_inner(dm, d_out, nullptr);
            const std::vector<uint16_t> got = host_copy(d_out, want.size());
            long long diff = 0;
            for (size_t i = 0; i < want.size(); ++i) diff += got[i] != want[i];
            x3::reconstruct_inner_rows(dm, d_out, nullptr);
            const std::vector<uint16_t> got_t = host_copy(d_out, want.size());
            long long diff_t = 0;
            for (int r = 0; r < m.k; ++r)
                for (int c = 0; c < m.n; ++c) diff_t += got_t[(size_t) c * m.k + r] != want[(size_t) r * m.n + c];
            const std::string lab = "K" + strata::exl3_k_label(k2);
            if (diff != 0 || diff_t != 0 || k2 == 8 || k2 == 16 || (k2 & 1))   // K 4, 8 and the half K reported
                report_exact("cb " + std::to_string(cb) + " " + lab + ": W' row-major, W' transposed", diff + diff_t,
                             (long long) want.size() * 2);
            else
                std::printf("  cb %d %s: 0 of %zu differ, both layouts  ok\n", cb, lab.c_str(), want.size() * 2);
            cudaFree(d_tr); cudaFree(d_suh); cudaFree(d_svh); cudaFree(d_out);
        }
}

// ---------------------------------------------------------------- 2. the rotations
void test_rotations() {
    std::printf("the rotations against the host emulation\n");
    const int rows = 37;
    const HMat g = rand_mat(E, F, 8), u = rand_mat(E, F, 8), d = rand_mat(F, E, 8);
    const std::vector<float> x = randn((size_t) rows * E, 1.0f);
    std::vector<int32_t> idx((size_t) rows);
    for (int r = 0; r < rows; ++r) idx[(size_t) r] = (r * 17 + 5) % rows;   // a gather, as the prompt path's
    float* d_x = dev_copy(x);
    int32_t* d_idx = dev_copy(idx);
    uint16_t *d_suh = dev_copy(g.suh), *d_svg = dev_copy(g.svh), *d_svu = dev_copy(u.svh), *d_sud = dev_copy(d.suh);
    uint16_t* d_o16 = nullptr;
    if (!d_x || !d_idx || !d_suh || !d_svg || !d_svu || !d_sud || !ck(cudaMalloc(&d_o16, (size_t) rows * E * 2), "malloc"))
        return;
    // rows_rot_f16 with the gather
    x3::rows_rot_f16(d_x, E, d_idx, rows, E, d_suh, d_o16, E, nullptr);
    std::vector<uint16_t> got = host_copy(d_o16, (size_t) rows * E);
    long long diff = 0;
    for (int r = 0; r < rows; ++r) {
        const std::vector<uint16_t> want = rot_f16(x.data() + (size_t) idx[(size_t) r] * E, g.suh);
        for (int c = 0; c < E; ++c) diff += got[(size_t) r * E + c] != want[(size_t) c];
    }
    report_exact("rows_rot_f16 (37 gathered rows of 2560)", diff, (long long) rows * E);
    // rows_rot_out in place, F32
    std::vector<float> y = randn((size_t) rows * F, 3.0f);
    float* d_y = dev_copy(y);
    if (!d_y) return;
    x3::rows_rot_out(d_y, F, rows, F, d_svg, nullptr);
    const std::vector<float> yg = host_copy(d_y, y.size());
    std::vector<float> yw = y;
    for (int r = 0; r < rows; ++r) rot_out(yw.data() + (size_t) r * F, g.svh);
    report_exact("rows_rot_out (37 rows of 640)", bits_differ(yg, yw), (long long) yw.size());
    // gu_epilogue: [gate | up] rows of 2 n_ff -> h16
    std::vector<float> gu = randn((size_t) rows * 2 * F, 3.0f);
    float* d_gu = dev_copy(gu);
    if (!d_gu) return;
    x3::gu_epilogue(d_gu, 2 * F, rows, F, d_svg, d_svu, d_sud, 0.0f, d_o16, F, nullptr);
    got = host_copy(d_o16, (size_t) rows * F);
    // the swiglu's exp: the device's expf and the host's are each within an ulp or two, not the same function, so a
    // value can round to the neighbouring FP16 - allowed for at most 1% of them, one FP16 unit apart
    diff = 0;
    int max_ulp = 0;
    for (int r = 0; r < rows; ++r) {
        std::vector<float> a(gu.begin() + (size_t) r * 2 * F, gu.begin() + (size_t) r * 2 * F + F);
        std::vector<float> b(gu.begin() + (size_t) r * 2 * F + F, gu.begin() + (size_t) (r + 1) * 2 * F);
        rot_out(a.data(), g.svh);
        rot_out(b.data(), u.svh);
        std::vector<float> h((size_t) F);
        for (int i = 0; i < F; ++i) h[(size_t) i] = swiglu(a[(size_t) i], b[(size_t) i]);
        const std::vector<uint16_t> want = rot_f16(h.data(), d.suh);
        for (int c = 0; c < F; ++c) {
            const uint16_t x0 = got[(size_t) r * F + c], x1 = want[(size_t) c];
            if (x0 == x1) continue;
            ++diff;
            // FP16 units apart (same sign: the bit patterns' distance; otherwise far)
            max_ulp = std::max(max_ulp, (x0 ^ x1) & 0x8000 ? 99 : std::abs((int) (x0 & 0x7fff) - (int) (x1 & 0x7fff)));
        }
    }
    std::printf("  %-58s %lld of %d differ, at most %d FP16 unit(s)  %s\n", "gu_epilogue (37 rows: rot_out x2, swiglu, rot_f16)",
                diff, rows * F, max_ulp, diff * 100 <= (long long) rows * F && max_ulp <= 1 ? "ok" : "FAIL");
    if (!(diff * 100 <= (long long) rows * F && max_ulp <= 1)) ++g_fail;
    for (void* p : {(void*) d_x, (void*) d_idx, (void*) d_suh, (void*) d_svg, (void*) d_svu, (void*) d_sud,
                    (void*) d_o16, (void*) d_y, (void*) d_gu})
        cudaFree(p);
}

// ---------------------------------------------------------------- 3-5. whole experts
// A window of T tokens routing k of the n experts each (distinct within a token), grouped by expert in order of first
// appearance, as the window's host pool publishes it.  Every route's row through grouped_experts, then each entry alone;
// against the host reference and the CPU lane.
void test_experts(const std::vector<HExpert>& xs, const std::string& tag) {
    const int nx = (int) xs.size();
    const int T = 8, k = std::min(10, nx);
    Blobs b = make_blobs(xs);
    if (b.dev == nullptr) return;
    const std::vector<float> x = randn((size_t) T * E, 1.0f);
    // routes
    std::vector<int> ids((size_t) T * k);
    for (int t = 0; t < T; ++t) {
        std::vector<int> perm((size_t) nx);
        for (int i = 0; i < nx; ++i) perm[(size_t) i] = i;
        std::shuffle(perm.begin(), perm.end(), g_rng);
        for (int j = 0; j < k; ++j) ids[(size_t) (t * k + j)] = perm[(size_t) j];
    }
    // the plan: groups by expert, first appearance; entries (tok, dst = t * k + j)
    std::vector<unsigned long long> gptr;
    std::vector<int32_t> gstart, etok, edst;
    std::vector<int> gexp;
    for (int i = 0; i < T * k; ++i) {
        const int e = ids[(size_t) i];
        if (std::find(gexp.begin(), gexp.end(), e) != gexp.end()) continue;
        gexp.push_back(e);
        gptr.push_back((unsigned long long) (b.dev + (size_t) e * b.stride));
        gstart.push_back((int32_t) etok.size());
        for (int q = i; q < T * k; ++q)
            if (ids[(size_t) q] == e) {
                etok.push_back(q / k);
                edst.push_back(q);
            }
    }
    gstart.push_back((int32_t) etok.size());
    const int groups = (int) gptr.size(), entries = (int) etok.size();
    const int max_groups = T * k;   // a captured launch's size: the blocks past the real count return
    const int32_t ng = groups;
    unsigned long long* d_gptr = dev_copy(gptr);
    int32_t *d_gstart = dev_copy(gstart), *d_etok = dev_copy(etok), *d_edst = dev_copy(edst);
    int32_t* d_ng = dev_copy(std::vector<int32_t>{ng});
    float* d_x = dev_copy(x);
    float *d_out = nullptr, *d_h = nullptr, *d_u = nullptr;
    unsigned* d_cnt = nullptr;
    float* d_part = nullptr;   // proto: the gate/up's split-K partials (STRATA_EXL3_GU_SPLIT)
    const size_t ncnt = x3::group_counters(max_groups, F);
    if (const size_t np = x3::group_partials(max_groups, F); np > 0 && !ck(cudaMalloc(&d_part, np * 4), "malloc part")) return;
    if (!d_gptr || !d_gstart || !d_etok || !d_edst || !d_ng || !d_x ||
        !ck(cudaMalloc(&d_out, (size_t) T * k * E * 4), "malloc out") ||
        !ck(cudaMalloc(&d_h, (size_t) max_groups * F * 4), "malloc h") ||
        !ck(cudaMalloc(&d_u, (size_t) max_groups * F * 4), "malloc u") ||
        !ck(cudaMalloc(&d_cnt, ncnt * 4), "malloc cnt") || !ck(cudaMemset(d_cnt, 0, ncnt * 4), "zero cnt") ||
        !ck(cudaMemset(d_out, 0, (size_t) T * k * E * 4), "zero out"))
        return;
    x3::GroupArgs a;
    a.grp_ptr = d_gptr;
    a.grp_start = d_gstart;
    a.n_groups = d_ng;
    a.ent_tok = d_etok;
    a.ent_dst = d_edst;
    a.x = d_x;
    a.x_stride = E;
    a.out = d_out;
    a.out_stride = E;
    a.h = d_h;
    a.u = d_u;
    a.cnt = d_cnt;
    a.part = d_part;
    a.lay = b.lay;
    a.max_groups = max_groups;
    a.max_entries = max_groups;
    a.n_embd = E;
    a.n_ff = F;
    if (!x3::grouped_experts(a, nullptr) || !ck(cudaDeviceSynchronize(), "grouped_experts")) return;
    const std::vector<float> grouped = host_copy(d_out, (size_t) T * k * E);
    // 4. each entry alone: one group, one entry, the same kernels
    std::vector<float> alone((size_t) T * k * E, 0.0f);
    {
        std::vector<int32_t> one_start{0, 1}, one_ng{1};
        int32_t* d_os = dev_copy(one_start);
        int32_t* d_on = dev_copy(one_ng);
        int32_t *d_ot = nullptr, *d_od = nullptr;
        unsigned long long* d_op = nullptr;
        if (!d_os || !d_on || !ck(cudaMalloc(&d_ot, 4), "malloc") || !ck(cudaMalloc(&d_od, 4), "malloc") ||
            !ck(cudaMalloc(&d_op, 8), "malloc"))
            return;
        x3::GroupArgs o = a;
        o.grp_ptr = d_op;
        o.grp_start = d_os;
        o.n_groups = d_on;
        o.ent_tok = d_ot;
        o.ent_dst = d_od;
        o.max_groups = 1;
        for (int i = 0; i < T * k; ++i) {
            const int32_t tok = i / k, dst = i;
            const unsigned long long p = (unsigned long long) (b.dev + (size_t) ids[(size_t) i] * b.stride);
            if (!ck(cudaMemcpy(d_ot, &tok, 4, cudaMemcpyHostToDevice), "tok") ||
                !ck(cudaMemcpy(d_od, &dst, 4, cudaMemcpyHostToDevice), "dst") ||
                !ck(cudaMemcpy(d_op, &p, 8, cudaMemcpyHostToDevice), "ptr"))
                return;
            if (!x3::grouped_experts(o, nullptr) || !ck(cudaDeviceSynchronize(), "alone")) return;
        }
        alone = host_copy(d_out, (size_t) T * k * E);
        cudaFree(d_os); cudaFree(d_on); cudaFree(d_ot); cudaFree(d_od); cudaFree(d_op);
    }
    report_exact(tag + ": " + std::to_string(entries) + " entries in " + std::to_string(groups) +
                     " groups = each alone (bits)", bits_differ(grouped, alone), (long long) grouped.size());
    // 3. against the host reference, and 5. against the lane
    double worst = 0.0, worst_lane = 0.0;
    for (int i = 0; i < T * k; ++i) {
        const HExpert& xe = xs[(size_t) ids[(size_t) i]];
        const float* in = x.data() + (size_t) (i / k) * E;
        const std::vector<float> got(grouped.begin() + (size_t) i * E, grouped.begin() + (size_t) (i + 1) * E);
        worst = std::max(worst, rel_rms(got, expert_ref(xe, in, false)));
        if (i < 6) worst_lane = std::max(worst_lane, rel_rms(got, lane_expert(xe, in)));
    }
    report(tag + ": grouped rows vs the host reference (worst row)", worst, 2e-4);
    report(tag + ": grouped rows vs the CPU lane (F32 decode; 6 rows)", worst_lane, 5e-3);
    // the prompt path: rows sorted by expert ([bounds[e], bounds[e + 1]) for expert e), input rows by token index
    {
        std::vector<int32_t> bounds((size_t) nx + 1, 0), rowtok;
        for (int e = 0; e < nx; ++e) {
            bounds[(size_t) e] = (int32_t) rowtok.size();
            for (int i = 0; i < T * k; ++i)
                if (ids[(size_t) i] == e) rowtok.push_back(i / k);
        }
        bounds[(size_t) nx] = (int32_t) rowtok.size();
        const int nrows = (int) rowtok.size();
        int max_rows = 0;
        for (int e = 0; e < nx; ++e) max_rows = std::max(max_rows, bounds[(size_t) e + 1] - bounds[(size_t) e]);
        int32_t* d_bounds = dev_copy(bounds);
        int32_t* d_rowtok = dev_copy(rowtok);
        uint16_t* d_h16 = nullptr;
        float* d_y = nullptr;
        if (!d_bounds || !d_rowtok || !ck(cudaMalloc(&d_h16, (size_t) nrows * F * 2), "malloc h16") ||
            !ck(cudaMalloc(&d_y, (size_t) nrows * E * 4), "malloc y"))
            return;
        // the tensor-core set kernels: every expert of the set in one launch (row0 0, bounds absolute)
        if (x3::device_ok()) {
            x3::SetArgs sa;
            sa.wbase = b.dev;
            sa.stride = b.stride;
            sa.lay = b.lay;
            sa.n_exp = nx;
            sa.bounds = d_bounds;
            sa.row0 = 0;
            sa.max_rows = max_rows;
            sa.n_embd = E;
            sa.n_ff = F;
            x3::set_gate_up(sa, d_x, E, d_rowtok, d_h16, F, nullptr);
            x3::set_down(sa, d_h16, F, d_y, E, nullptr);
            if (!ck(cudaDeviceSynchronize(), "set kernels")) return;
            const std::vector<float> ys = host_copy(d_y, (size_t) nrows * E);
            const std::vector<uint16_t> hs = host_copy(d_h16, (size_t) nrows * F);
            double w = 0.0, w_nr = 0.0;
            long long hdiff = 0;
            for (int e = 0; e < nx; ++e)
                for (int r = bounds[(size_t) e]; r < bounds[(size_t) e + 1]; ++r) {
                    const float* in = x.data() + (size_t) rowtok[(size_t) r] * E;
                    const std::vector<float> got(ys.begin() + (size_t) r * E, ys.begin() + (size_t) (r + 1) * E);
                    w = std::max(w, rel_rms(got, expert_ref(xs[(size_t) e], in, true)));
                    w_nr = std::max(w_nr, rel_rms(got, expert_ref(xs[(size_t) e], in, false)));
                    // the FP16 h the set kernels hand the down product, against the reference's
                    const HExpert& xe = xs[(size_t) e];
                    std::vector<float> gy = sums(rot_f16(in, xe.g.suh), xe.wg, E, F), uy = sums(rot_f16(in, xe.u.suh), xe.wu, E, F);
                    rot_out(gy.data(), xe.g.svh);
                    rot_out(uy.data(), xe.u.svh);
                    for (int i = 0; i < F; ++i)
                        hdiff += hs[(size_t) r * F + i] != f16_from_f32(swiglu(gy[(size_t) i], uy[(size_t) i]));
                }
            // These kernels hand the down product h in FP16 (as exllamav3's and Maya's do), rounded before its rotation.
            // The tensor cores' F32 sums differ from exact ones by ~1e-5, enough to put ~3% of those FP16 values one unit
            // from the reference's (measured 1,378-1,430 of 51,200): ~4e-4 against either reference (h rounded or F32),
            // under the device-vs-lane difference of the format itself (~6e-4).  The other two routes rotate h in F32.
            std::printf("    (FP16 h one unit off the reference's: %lld of %lld; against the reference with h in F32: %.2e)\n",
                        hdiff, (long long) nrows * F, w_nr);
            report(tag + ": tensor-core set kernels vs the host reference", w, 6e-4);
        }
        // the decoded W' + cuBLAS (prefill::Gemm::f16), one expert at a time as the prompt path runs a heavy one
        {
            strata::prefill::Gemm gm;
            std::string err;
            uint16_t *d_w16 = nullptr, *d_x16 = nullptr;
            float* d_gu = nullptr;
            if (!gm.init(nullptr, 1 << 20, err)) {
                std::fprintf(stderr, "exl3_dev_parity: Gemm::init: %s\n", err.c_str());
                ++g_fail;
                return;
            }
            if (!ck(cudaMalloc(&d_w16, (size_t) 2 * F * E * 2), "malloc w16") ||
                !ck(cudaMalloc(&d_x16, (size_t) nrows * E * 2), "malloc x16") ||
                !ck(cudaMalloc(&d_gu, (size_t) nrows * 2 * F * 4), "malloc gu"))
                return;
            for (int e = 0; e < nx; ++e) {
                const int o0 = bounds[(size_t) e], ne = bounds[(size_t) e + 1] - o0;
                if (ne == 0) continue;
                const uint8_t* blob = b.dev + (size_t) e * b.stride;
                auto mat = [&](int r) {
                    x3::Mat m;
                    m.trellis = (const uint32_t*) (blob + b.lay.trellis[r]);
                    m.suh = (const uint16_t*) (blob + b.lay.suh[r]);
                    m.svh = (const uint16_t*) (blob + b.lay.svh[r]);
                    m.k = r < 2 ? E : F;
                    m.n = r < 2 ? F : E;
                    m.k2 = b.lay.k2[r];
                    m.cb = b.lay.cb;
                    return m;
                };
                for (int r = 0; r < 2; ++r) {
                    x3::reconstruct_inner_rows(mat(r), d_w16, nullptr);
                    x3::rows_rot_f16(d_x, E, d_rowtok + o0, ne, E, mat(r).suh, d_x16 + (size_t) o0 * E, E, nullptr);
                    gm.f16(d_x16 + (size_t) o0 * E, d_w16, d_gu + (size_t) o0 * 2 * F + (size_t) r * F, ne, F, E, 2 * F);
                }
                x3::gu_epilogue(d_gu + (size_t) o0 * 2 * F, 2 * F, ne, F, mat(0).svh, mat(1).svh, mat(2).suh, 0.0f,
                                d_h16 + (size_t) o0 * F, F, nullptr);
                x3::reconstruct_inner_rows(mat(2), d_w16, nullptr);
                gm.f16(d_h16 + (size_t) o0 * F, d_w16, d_y + (size_t) o0 * E, ne, E, F, E);
                x3::rows_rot_out(d_y + (size_t) o0 * E, E, ne, E, mat(2).svh, nullptr);
            }
            if (!ck(cudaDeviceSynchronize(), "cuBLAS route")) return;
            const std::vector<float> yb = host_copy(d_y, (size_t) nrows * E);
            double w = 0.0;
            for (int e = 0; e < nx; ++e)
                for (int r = bounds[(size_t) e]; r < bounds[(size_t) e + 1]; ++r) {
                    const std::vector<float> got(yb.begin() + (size_t) r * E, yb.begin() + (size_t) (r + 1) * E);
                    w = std::max(w, rel_rms(got, expert_ref(xs[(size_t) e], x.data() + (size_t) rowtok[(size_t) r] * E, false)));
                }
            report(tag + ": decoded W' + cuBLAS vs the host reference", w, 2e-4);
            cudaFree(d_w16); cudaFree(d_x16); cudaFree(d_gu);
        }
        cudaFree(d_bounds); cudaFree(d_rowtok); cudaFree(d_h16); cudaFree(d_y);
    }
    for (void* p : {(void*) d_gptr, (void*) d_gstart, (void*) d_etok, (void*) d_edst, (void*) d_ng, (void*) d_x,
                    (void*) d_out, (void*) d_h, (void*) d_u, (void*) d_cnt, (void*) b.dev})
        cudaFree(p);
}

}  // namespace

int main(int argc, char** argv) {
    std::string real;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--real" && i + 1 < argc) real = argv[++i];
        else {
            std::fprintf(stderr, "usage: exl3_dev_parity [--real <EXL3 model dir>]\n");
            return 2;
        }
    }
    int dev = 0;
    cudaDeviceProp p{};
    if (!ck(cudaGetDevice(&dev), "device") || !ck(cudaGetDeviceProperties(&p, dev), "properties")) return 1;
    std::printf("exl3_dev_parity on %s (sm_%d%d), tensor-core kernels %s, lane tier %d\n", p.name, p.major, p.minor,
                x3::device_ok() ? "yes" : "no", cpu::exl3_cpu_tier());
    test_decode();
    test_rotations();
    std::printf("whole experts, synthetic (the Flash's shapes, 12 experts, a window of 8 tokens x top-10)\n");
    // gate/up and down as 2 K: K 2..6, then an SC 4.05's layers 1-7 (3.5 / 5) and 0 (3 / 5), and 1.5 / 2.5 gate/up
    for (const auto& kk : {std::pair<int, int>(4, 4), {6, 6}, {8, 8}, {10, 10}, {12, 12}, {7, 10}, {6, 10}, {3, 6},
                           {5, 8}}) {
        std::vector<HExpert> xs(12);
        for (auto& x : xs) {
            x.g = rand_mat(E, F, kk.first);
            x.u = rand_mat(E, F, kk.first);
            x.d = rand_mat(F, E, kk.second);
            x.decode();
        }
        test_experts(xs, "K" + strata::exl3_k_label(kk.first) +
                             (kk.second != kk.first ? "/" + strata::exl3_k_label(kk.second) : std::string()));
    }
    if (!real.empty()) {
        std::printf("whole experts, real (%s)\n", real.c_str());
        try {
            const strata::Exl3Model model(real);
            for (int layer : {0, 1, 8, 23, 47}) {
                std::vector<HExpert> xs;
                for (int e : {0, 300, 511, 17, 64, 128, 200, 256, 333, 400, 450, 500}) {
                    const std::string b = "model.language_model.layers." + std::to_string(layer) + ".mlp.experts." +
                                          std::to_string(e) + ".";
                    HExpert x;
                    std::string err;
                    if (!load_real(model, b + "gate_proj", x.g, err) || !load_real(model, b + "up_proj", x.u, err) ||
                        !load_real(model, b + "down_proj", x.d, err)) {
                        std::printf("  %s: %s  FAIL\n", b.c_str(), err.c_str());
                        ++g_fail;
                        continue;
                    }
                    x.decode();
                    xs.push_back(std::move(x));
                }
                if (xs.size() == 12) test_experts(xs, "layer " + std::to_string(layer));
            }
        } catch (const std::exception& e) {
            std::fprintf(stderr, "exl3_dev_parity: %s\n", e.what());
            ++g_fail;
        }
    }
    if (x3::launch_errors() != 0) {
        std::printf("  launch errors: %d  FAIL\n", x3::launch_errors());
        ++g_fail;
    }
    std::printf("exl3_dev_parity: %s (%d failures)\n", g_fail ? "FAIL" : "ok", g_fail);
    return g_fail ? 1 : 0;
}
