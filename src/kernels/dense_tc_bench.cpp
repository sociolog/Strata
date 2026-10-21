// src/kernels/dense_tc_bench.cpp - STRATA_DENSE_TC: the int8 tensor-core Q8_0 kernel (dense_tc_q8_0, ported from
// Basalt) against native_mmvq at the engine's dense shapes and window widths T = 1..8: parity and microseconds.
//
// Parity, per shape and T: (1) every column of a T-column call is bitwise the same as that column alone (T = 1) - the
// kernel is T-invariant; (2) both kernels' error against an exact (f64) sum of the same q8_1 x Q8_0 terms, and the
// tensor-core result's max / RMS difference from native_mmvq relative to the output's RMS.  Exit status 1 when (1)
// fails or (2) exceeds float rounding (rel. RMS > 1e-5).
// Timing: enough copies of each matrix are cycled to miss the GPU's last-level cache, as a layer's call does.
//
//     build/dense_tc_bench [--cols 1,2,3,4,5,6,7,8] [--iters 200] [--cfg all|N] [--shape substring]
#include "strata/kernels/dense_tc.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace K = strata::kernels;

namespace {
struct Shape { const char* what; int n_in, n_out; };
const Shape SHAPES[] = {
    {"GDN qkv 2560->10240", 2560, 10240}, {"GDN z 2560->6144", 2560, 6144},   {"GDN/QSA out 6144->2560", 6144, 2560},
    {"QSA q 2560->12288", 2560, 12288},   {"QSA k/v 2560->512", 2560, 512},   {"shared gate/up 2560->640", 2560, 640},
    {"head 2560->248320", 2560, 248320},
};
void ck(cudaError_t e, const char* w) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", w, cudaGetErrorString(e)); std::exit(1); }
}
std::vector<int> ints(const std::string& s) {
    std::vector<int> v;
    size_t from = 0;
    for (;;) {
        const size_t c = s.find(',', from);
        v.push_back(std::atoi(s.substr(from, c == std::string::npos ? std::string::npos : c - from).c_str()));
        if (c == std::string::npos) break;
        from = c + 1;
    }
    return v;
}
float h2f(const uint8_t* p) {
    __half h;
    std::memcpy(&h, p, 2);
    return __half2float(h);
}
}  // namespace

int main(int argc, char** argv) {
    std::vector<int> cols = {1, 2, 3, 4, 5, 6, 7, 8};
    int iters = 200, cfg_only = -1;
    std::string shape_filter;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--cols" && i + 1 < argc) cols = ints(argv[++i]);
        else if (a == "--iters" && i + 1 < argc) iters = std::atoi(argv[++i]);
        else if (a == "--cfg" && i + 1 < argc) { const std::string v = argv[++i]; cfg_only = v == "all" ? -1 : std::atoi(v.c_str()); }
        else if (a == "--shape" && i + 1 < argc) shape_filter = argv[++i];
    }
    cudaDeviceProp prop;
    ck(cudaGetDeviceProperties(&prop, 0), "props");
    std::printf("dense_tc_bench: %s (sm_%d%d, %d SMs)\n", prop.name, prop.major, prop.minor, prop.multiProcessorCount);
    if (!K::dense_tc_q8_0_ok(2560)) {
        std::printf("dense_tc_q8_0 needs sm_80+\n");
        return 0;
    }
    std::mt19937 rng(7);
    bool fail = false;
    const int maxc = 8;
    for (const Shape& s : SHAPES) {
        if (!shape_filter.empty() && std::string(s.what).find(shape_filter) == std::string::npos) continue;
        const int nb = s.n_in / 32;
        const size_t bytes = (size_t) s.n_out * nb * 34;
        const int copies = (int) std::max<size_t>(1, std::min<size_t>(16, ((size_t) 512 << 20) / bytes));
        std::vector<uint8_t> host(bytes);
        for (size_t b = 0; b < bytes; b += 34) {   // fp16 scales around 0.01, random int8 codes
            const __half d = __float2half(0.004f + 0.012f * (float) (rng() % 1000) / 1000.0f);
            std::memcpy(&host[b], &d, 2);
            for (int k = 0; k < 32; ++k) host[b + 2 + k] = (uint8_t) (rng() & 0xff);
        }
        std::vector<void*> w(copies);
        for (auto& p : w) {
            ck(cudaMalloc(&p, bytes), "malloc w");
            ck(cudaMemcpy(p, host.data(), bytes, cudaMemcpyHostToDevice), "copy w");
        }
        std::vector<float> xh((size_t) s.n_in * maxc);
        for (auto& v : xh) v = std::normal_distribution<float>(0.f, 1.f)(rng);
        float *x = nullptr, *y = nullptr, *y1 = nullptr;
        void *xq = nullptr, *xq1 = nullptr;
        const size_t qb = K::native_q8_1_bytes(s.n_in, 1);
        ck(cudaMalloc(&x, xh.size() * 4), "x");
        ck(cudaMemcpy(x, xh.data(), xh.size() * 4, cudaMemcpyHostToDevice), "x copy");
        ck(cudaMalloc(&xq, K::native_q8_1_bytes(s.n_in, maxc)), "xq");
        ck(cudaMalloc(&xq1, qb), "xq1");
        ck(cudaMalloc(&y, (size_t) s.n_out * maxc * 4), "y");
        ck(cudaMalloc(&y1, (size_t) s.n_out * 4), "y1");
        cudaStream_t st;
        ck(cudaStreamCreate(&st), "stream");
        cudaEvent_t e0, e1;
        cudaEventCreate(&e0);
        cudaEventCreate(&e1);
        K::native_quantize_q8_1(x, xq, s.n_in, maxc, st);
        std::vector<uint8_t> xqh(K::native_q8_1_bytes(s.n_in, maxc));
        ck(cudaMemcpy(xqh.data(), xq, xqh.size(), cudaMemcpyDeviceToHost), "xq host");
        // exact reference of the quantized product, all 8 columns (a row sample for the head)
        const int rstep = s.n_out > 20000 ? 97 : 1;
        std::vector<double> ref((size_t) s.n_out * maxc, 0.0);
        for (int r = 0; r < s.n_out; r += rstep)
            for (int c = 0; c < maxc; ++c) {
                double acc = 0;
                for (int b = 0; b < nb; ++b) {
                    const uint8_t* wb = &host[((size_t) r * nb + b) * 34];
                    const uint8_t* xb = &xqh[((size_t) c * nb + b) * 36];
                    int dot = 0;
                    for (int k = 0; k < 32; ++k) dot += (int) (int8_t) wb[2 + k] * (int) (int8_t) xb[4 + k];
                    acc += (double) h2f(wb) * (double) h2f(xb) * dot;
                }
                ref[(size_t) c * s.n_out + r] = acc;
            }
        std::printf("\n%s: %.2f MB, %d copies\n", s.what, bytes / 1e6, copies);
        std::vector<float> ym((size_t) s.n_out * maxc), yt((size_t) s.n_out * maxc), ya(s.n_out);
        auto err_vs_ref = [&](const std::vector<float>& v, int c, double& rms, double& refrms) {
            double se = 0, sr = 0;
            int cnt = 0;
            for (int r = 0; r < s.n_out; r += rstep) {
                const double d = v[(size_t) c * s.n_out + r] - ref[(size_t) c * s.n_out + r];
                se += d * d;
                sr += ref[(size_t) c * s.n_out + r] * ref[(size_t) c * s.n_out + r];
                ++cnt;
            }
            rms = std::sqrt(se / cnt);
            refrms = std::sqrt(sr / cnt);
        };
        // -------- parity (default launch shape)
        {
            double worst_tc = 0, worst_mm = 0, worst_diff = 0, worst_max = 0;
            bool tinv = true;
            // each column alone
            std::vector<std::vector<float>> alone(maxc, std::vector<float>(s.n_out));
            for (int c = 0; c < maxc; ++c) {
                ck(cudaMemcpy(xq1, (uint8_t*) xq + (size_t) c * qb, qb, cudaMemcpyDeviceToDevice), "col");
                K::dense_tc_q8_0(w[0], xq1, y1, s.n_in, s.n_out, 1, st);
                ck(cudaMemcpyAsync(alone[c].data(), y1, (size_t) s.n_out * 4, cudaMemcpyDeviceToHost, st), "y1");
            }
            ck(cudaStreamSynchronize(st), "sync");
            for (int T = 1; T <= maxc; ++T) {
                K::native_mmvq(8, w[0], xq, y, s.n_in, s.n_out, T, st);
                ck(cudaMemcpyAsync(ym.data(), y, (size_t) s.n_out * T * 4, cudaMemcpyDeviceToHost, st), "ym");
                K::dense_tc_q8_0(w[0], xq, y, s.n_in, s.n_out, T, st);
                ck(cudaMemcpyAsync(yt.data(), y, (size_t) s.n_out * T * 4, cudaMemcpyDeviceToHost, st), "yt");
                ck(cudaStreamSynchronize(st), "sync");
                for (int c = 0; c < T; ++c) {
                    if (std::memcmp(&yt[(size_t) c * s.n_out], alone[c].data(), (size_t) s.n_out * 4) != 0) tinv = false;
                    double rt, rm, rr;
                    err_vs_ref(yt, c, rt, rr);
                    err_vs_ref(ym, c, rm, rr);
                    worst_tc = std::max(worst_tc, rt / rr);
                    worst_mm = std::max(worst_mm, rm / rr);
                    double se = 0, mx = 0;
                    for (int r = 0; r < s.n_out; ++r) {
                        const double d = (double) yt[(size_t) c * s.n_out + r] - ym[(size_t) c * s.n_out + r];
                        se += d * d;
                        mx = std::max(mx, std::fabs(d));
                    }
                    worst_diff = std::max(worst_diff, std::sqrt(se / s.n_out) / rr);
                    worst_max = std::max(worst_max, mx / rr);
                }
            }
            const bool ok = tinv && worst_diff < 1e-5;
            if (!ok) fail = true;
            std::printf("  parity T=1..8: T-invariant %s | rel RMS vs exact: tc %.2e, mmvq %.2e | tc-mmvq rel RMS %.2e, "
                        "max %.2e  %s\n",
                        tinv ? "yes (bitwise)" : "NO", worst_tc, worst_mm, worst_diff, worst_max, ok ? "OK" : "FAIL");
        }
        // -------- timing
        auto time_it = [&](auto&& call, int c) {
            for (int i = 0; i < 5; ++i) call(w[i % copies], c);
            cudaEventRecord(e0, st);
            for (int i = 0; i < iters; ++i) call(w[i % copies], c);
            cudaEventRecord(e1, st);
            ck(cudaEventSynchronize(e1), "sync");
            float ms = 0;
            cudaEventElapsedTime(&ms, e0, e1);
            return 1000.0 * ms / iters;
        };
        std::printf("  %-14s", "us  T=");
        for (int c : cols) std::printf("%8d", c);
        std::printf("\n  %-14s", "mmvq");
        std::vector<double> base;
        for (int c : cols) {
            const double us = time_it([&](void* wp, int n) { K::native_mmvq(8, wp, xq, y, s.n_in, s.n_out, n, st); }, c);
            base.push_back(us);
            std::printf("%8.1f", us);
        }
        std::printf("   (GB/s at T=4: %.0f)\n", cols.size() > 3 ? bytes / (base[3] * 1e3) : 0.0);
        for (int cfg = 0; K::dense_tc_q8_0_cfg_name(cfg) != nullptr; ++cfg) {
            if (cfg_only >= 0 && cfg != cfg_only) continue;
            if (!K::dense_tc_q8_0_cfg(cfg, w[0], xq, y, s.n_in, s.n_out, 1, st)) continue;
            std::printf("  %-14s", K::dense_tc_q8_0_cfg_name(cfg));
            size_t k = 0;
            for (int c : cols) {
                const double us = time_it(
                    [&](void* wp, int n) { K::dense_tc_q8_0_cfg(cfg, wp, xq, y, s.n_in, s.n_out, n, st); }, c);
                std::printf("%8.1f", us);
                (void) base[k++];
            }
            std::printf("\n");
        }
        for (auto p : w) cudaFree(p);
        cudaFree(x);
        cudaFree(xq);
        cudaFree(xq1);
        cudaFree(y);
        cudaFree(y1);
        cudaStreamDestroy(st);
    }
    std::printf("\n%s\n", fail ? "FAIL" : "all parity checks passed");
    return fail ? 1 : 0;
}
