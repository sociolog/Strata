// src/kernels/exl3_lane_gpu_parity.cpp - the EXL3 CPU lane (src/kernels/cpu/exl3_lane.cpp) against the GPU kernels:
// one matrix through exl3_gemv_f32, one whole expert (gate, up, SwiGLU, down) through exl3_moe_rows.  Both compute
// the codebook's fp32 fold as the lane does; the GPU rounds its activations (and the GEMV its outputs) to fp16, so the
// two agree to that rounding (relative RMS error ~1e-3), not bit for bit.
//
//   exl3_lane_gpu_parity                     synthetic matrices: K 2..6 on the Flash's expert shapes
//   exl3_lane_gpu_parity --real <model dir>  plus 11 routed experts of the model (layers 0, 23, 47, MTP)
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
#include "strata/kernels/cpu/exl3_lane.hpp"
#include "strata/kernels/exl3.hpp"
#include "strata/kernels/f16_bits.hpp"

namespace cpu = strata::kernels::cpu;
using strata::kernels::f16_from_f32;
using strata::kernels::f32_from_f16;

namespace {

int g_fail = 0;
std::mt19937_64 g_rng(4096);

bool ck(cudaError_t e, const char* what) {
    if (e == cudaSuccess) return true;
    std::fprintf(stderr, "exl3_lane_gpu_parity: CUDA %s: %s\n", what, cudaGetErrorString(e));
    ++g_fail;
    return false;
}

// one matrix on the host (aligned words) and on the device
struct Mat {
    std::vector<uint32_t> tr;
    std::vector<uint16_t> suh, svh;
    int k = 0, n = 0, K = 0;
    uint16_t *d_tr = nullptr, *d_suh = nullptr, *d_svh = nullptr;
    cpu::Exl3Part part() const {
        cpu::Exl3Part m;
        m.trellis = tr.data();
        m.suh = suh.data();
        m.svh = svh.data();
        m.k = k;
        m.n = n;
        m.k2 = 2 * K;   // #1764's kernels beside it take integer K only
        return m;
    }
    strata::kernels::Exl3Mat dev() const {
        strata::kernels::Exl3Mat m;
        m.trellis = d_tr;
        m.suh = d_suh;
        m.svh = d_svh;
        m.ki = k / 16;
        m.nj = n / 16;
        m.bits = K;
        m.cb = 2;
        return m;
    }
    bool upload() {
        return ck(cudaMalloc(&d_tr, tr.size() * 4), "malloc trellis") && ck(cudaMalloc(&d_suh, suh.size() * 2), "malloc suh") &&
               ck(cudaMalloc(&d_svh, svh.size() * 2), "malloc svh") &&
               ck(cudaMemcpy(d_tr, tr.data(), tr.size() * 4, cudaMemcpyHostToDevice), "copy trellis") &&
               ck(cudaMemcpy(d_suh, suh.data(), suh.size() * 2, cudaMemcpyHostToDevice), "copy suh") &&
               ck(cudaMemcpy(d_svh, svh.data(), svh.size() * 2, cudaMemcpyHostToDevice), "copy svh");
    }
    void free_dev() {
        (void) cudaFree(d_tr);
        (void) cudaFree(d_suh);
        (void) cudaFree(d_svh);
        d_tr = nullptr;
        d_suh = d_svh = nullptr;
    }
};

Mat rand_mat(int k, int n, int K) {
    Mat m;
    m.k = k;
    m.n = n;
    m.K = K;
    m.tr.resize((size_t) (k / 16) * (n / 16) * 8 * K);
    for (auto& w : m.tr) w = (uint32_t) g_rng();
    // channel scales of a real expert's size (the 4.05 file: |suh| ~0.015, |svh| ~1.0), random signs - the GPU's
    // expert keeps gate, up and SwiGLU in fp16, which larger scales overflow (65504) where the lane's F32 does not
    std::uniform_real_distribution<float> us(0.010f, 0.025f), uv(0.8f, 1.2f);
    m.suh.resize((size_t) k);
    m.svh.resize((size_t) n);
    for (auto& v : m.suh) v = f16_from_f32((g_rng() & 1) ? us(g_rng) : -us(g_rng));
    for (auto& v : m.svh) v = f16_from_f32((g_rng() & 1) ? uv(g_rng) : -uv(g_rng));
    return m;
}

bool load_real(const strata::Exl3Model& model, const std::string& base, Mat& m, std::string& err) {
    if (!model.has(base)) {
        err = "no tensor " + base;
        return false;
    }
    const strata::Exl3Linear L = model.linear(base);
    if (L.cb != 2 || L.half() || L.in() % 128 != 0 || L.out() % 128 != 0) {
        err = base + ": not a mul1 matrix of integer K the lane and #1764's kernels run (K " + strata::exl3_k_label(L.k2) + ")";
        return false;
    }
    m.k = L.in();
    m.n = L.out();
    m.K = L.k2 / 2;
    m.tr.assign((size_t) L.ki * L.nj * L.tile_words() / 2, 0u);
    std::memcpy(m.tr.data(), L.trellis, m.tr.size() * 4);   // the file's words sit at offsets 2 mod 4
    m.suh.assign(L.suh, L.suh + m.k);
    m.svh.assign(L.svh, L.svh + m.n);
    return true;
}

// relative RMS difference; a non-finite value on either side is reported and counts as a failure (1e30)
double rel_rms(const std::vector<float>& got, const std::vector<float>& ref) {
    double se = 0.0, sr = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        if (!std::isfinite(got[i]) || !std::isfinite(ref[i])) {
            std::printf("    non-finite value at %zu: lane %g, GPU %g\n", i, got[i], ref[i]);
            return 1e30;
        }
        const double d = (double) got[i] - ref[i];
        se += d * d;
        sr += (double) ref[i] * ref[i];
    }
    return std::sqrt(se / std::max(sr, 1e-30));
}

// the lane's y = H(x * suh) W' * svh
std::vector<float> lane_mv(const Mat& m, const std::vector<float>& x) {
    const cpu::Exl3Part p = m.part();
    std::vector<float> xh((size_t) m.k), acc((size_t) m.n), y((size_t) m.n);
    const float xs = cpu::exl3_cpu_prepare(p, x.data(), xh.data());
    cpu::exl3_cpu_rows_acc(p, xh.data(), 0, m.k / 16, acc.data());
    for (int c0 = 0; c0 < m.n; c0 += 128) cpu::exl3_cpu_band_out(p, acc.data() + c0, xs, c0, y.data() + c0);
    return y;
}

// one matrix: the lane against exl3_gemv_f32 (the same x, f32 on both sides)
double gemv_vs_gpu(const Mat& m, const std::vector<float>& x) {
    float *d_x = nullptr, *d_y = nullptr;
    std::vector<float> y((size_t) m.n, 0.0f);
    if (ck(cudaMalloc(&d_x, x.size() * 4), "malloc x") && ck(cudaMalloc(&d_y, y.size() * 4), "malloc y") &&
        ck(cudaMemcpy(d_x, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "copy x")) {
        strata::kernels::exl3_gemv_f32(d_x, m.d_suh, m.d_svh, m.d_tr, m.k / 16, m.n / 16, m.K, 2, d_y, nullptr);
        ck(cudaGetLastError(), "exl3_gemv_f32");
        ck(cudaMemcpy(y.data(), d_y, y.size() * 4, cudaMemcpyDeviceToHost), "copy y");
    }
    (void) cudaFree(d_x);
    (void) cudaFree(d_y);
    return rel_rms(lane_mv(m, x), y);
}

float silu(float g) { return g / (1.0f + std::exp(-g)); }

// one expert: the lane's down(silu(gate(x)) * up(x)) against exl3_moe_rows (x in fp16 on both sides)
double expert_vs_gpu(Mat& g, Mat& u, Mat& d, const std::vector<uint16_t>& x16) {
    std::vector<float> x(x16.size());
    for (size_t i = 0; i < x.size(); ++i) x[i] = f32_from_f16(x16[i]);
    const std::vector<float> gy = lane_mv(g, x), uy = lane_mv(u, x);
    std::vector<float> h(gy.size());
    for (size_t i = 0; i < h.size(); ++i) h[i] = silu(gy[i]) * uy[i];
    const std::vector<float> want = lane_mv(d, h);
    const strata::kernels::Exl3Mat mg = g.dev(), mu = u.dev(), md = d.dev();
    uint16_t *d_x = nullptr, *d_out = nullptr;
    int* d_ids = nullptr;
    std::vector<uint16_t> out16((size_t) d.n, 0);
    const int id0 = 0;
    if (ck(cudaMalloc(&d_x, x16.size() * 2), "malloc x") && ck(cudaMalloc(&d_out, out16.size() * 2), "malloc out") &&
        ck(cudaMalloc(&d_ids, sizeof(int)), "malloc ids") &&
        ck(cudaMemcpy(d_x, x16.data(), x16.size() * 2, cudaMemcpyHostToDevice), "copy x") &&
        ck(cudaMemcpy(d_ids, &id0, sizeof(int), cudaMemcpyHostToDevice), "copy ids")) {
        strata::kernels::exl3_moe_rows(&mg, &mu, &md, d_ids, 1, d_x, d_out, nullptr);
        ck(cudaGetLastError(), "exl3_moe_rows");
        ck(cudaMemcpy(out16.data(), d_out, out16.size() * 2, cudaMemcpyDeviceToHost), "copy out");
    }
    (void) cudaFree(d_x);
    (void) cudaFree(d_out);
    (void) cudaFree(d_ids);
    std::vector<float> got(out16.size());
    for (size_t i = 0; i < got.size(); ++i) got[i] = f32_from_f16(out16[i]);
    return rel_rms(want, got);
}

void report(const std::string& what, double rel, double tol) {
    const bool ok = rel <= tol;
    std::printf("  %-50s rel %.2e (tolerance %.0e)  %s\n", what.c_str(), rel, tol, ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}

std::vector<float> randn(int n, float s) {
    std::normal_distribution<float> nd(0.0f, s);
    std::vector<float> v((size_t) n);
    for (auto& x : v) x = nd(g_rng);
    return v;
}

}  // namespace

int main(int argc, char** argv) {
    std::string real;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--real" && i + 1 < argc) real = argv[++i];
        else {
            std::fprintf(stderr, "usage: exl3_lane_gpu_parity [--real <EXL3 model dir>]\n");
            return 2;
        }
    }
    int dev = 0;
    cudaDeviceProp p{};
    if (!ck(cudaGetDevice(&dev), "device") || !ck(cudaGetDeviceProperties(&p, dev), "properties")) return 1;
    std::printf("exl3_lane_gpu_parity on %s, lane tier %d\n", p.name, cpu::exl3_cpu_tier());
    strata::kernels::exl3_gemv_reserve(8192, 262144);
    std::printf("synthetic matrices (the Flash's expert shapes)\n");
    for (int K = 2; K <= 6; ++K) {
        Mat g = rand_mat(2560, 640, K), u = rand_mat(2560, 640, K), d = rand_mat(640, 2560, K);
        if (!g.upload() || !u.upload() || !d.upload()) return 1;
        report("gemv 2560x640 K" + std::to_string(K), gemv_vs_gpu(g, randn(2560, 1.0f)), 2e-3);
        report("gemv 640x2560 K" + std::to_string(K), gemv_vs_gpu(d, randn(640, 1.0f)), 2e-3);
        std::vector<uint16_t> x16(2560);
        for (auto& v : x16) v = f16_from_f32(std::normal_distribution<float>(0.0f, 1.0f)(g_rng));
        report("expert K" + std::to_string(K) + " (exl3_moe_rows)", expert_vs_gpu(g, u, d, x16), 5e-3);
        g.free_dev();
        u.free_dev();
        d.free_dev();
    }
    if (!real.empty()) {
        std::printf("real experts from %s\n", real.c_str());
        try {
            const strata::Exl3Model model(real);
            std::vector<std::string> experts;
            for (int layer : {0, 23, 47})
                for (int e : {0, 300, 511})
                    experts.push_back("model.language_model.layers." + std::to_string(layer) + ".mlp.experts." +
                                      std::to_string(e) + ".");
            for (int e : {0, 511}) experts.push_back("mtp.layers.0.mlp.experts." + std::to_string(e) + ".");
            for (const std::string& b : experts) {
                Mat g, u, d;
                std::string err;
                if (!load_real(model, b + "gate_proj", g, err) || !load_real(model, b + "up_proj", u, err) ||
                    !load_real(model, b + "down_proj", d, err)) {
                    std::printf("  %s: %s  FAIL\n", b.c_str(), err.c_str());
                    ++g_fail;
                    continue;
                }
                if (!g.upload() || !u.upload() || !d.upload()) return 1;
                const std::string name = b.rfind("model.language_model.", 0) == 0 ? b.substr(21) : b;
                report(name + "gate gemv", gemv_vs_gpu(g, randn(g.k, 1.0f)), 2e-3);
                report(name + "down gemv", gemv_vs_gpu(d, randn(d.k, 0.1f)), 2e-3);
                std::vector<uint16_t> x16((size_t) g.k);
                for (auto& v : x16) v = f16_from_f32(std::normal_distribution<float>(0.0f, 1.0f)(g_rng));
                report(name + "expert (exl3_moe_rows)", expert_vs_gpu(g, u, d, x16), 5e-3);
                g.free_dev();
                u.free_dev();
                d.free_dev();
            }
        } catch (const std::exception& e) {
            std::fprintf(stderr, "exl3_lane_gpu_parity: %s\n", e.what());
            ++g_fail;
        }
    }
    std::printf("exl3_lane_gpu_parity: %s (%d failures)\n", g_fail ? "FAIL" : "ok", g_fail);
    return g_fail ? 1 : 0;
}
