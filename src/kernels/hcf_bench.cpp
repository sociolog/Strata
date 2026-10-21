// src/kernels/hcf_bench.cpp - STRATA_HC_FUSED: the fused HC read (+ router + top-10) against the unfused path it
// replaces (fused_gr_read_multi, then bf16_gemv_fp32_mmvf_multi + native_router_top10_multi), bit for bit on every
// output (R in place, lo, rs, inject, mixed, logits, ids, weights), on the bf16 read and the hc8 read, and the time of
// both with the weights rotating over `cold sets` copies (a layer's weights are cold in the engine).
//
//     STRATA_HC_FUSED=1 build/hcf_bench [iters] [T min] [T max] [cold sets]
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/fused_gr.hpp"
#include "strata/kernels/native_router.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace K = strata::kernels;

static uint16_t bf16(float x) {
    uint32_t u;
    std::memcpy(&u, &x, 4);
    return (uint16_t) ((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    if (K::hc_fused_mode() <= 0) { std::printf("hcf_bench: set STRATA_HC_FUSED=1\n"); return 2; }
    const int iters = argc > 1 ? std::atoi(argv[1]) : 200;
    const int t_lo = argc > 2 ? std::atoi(argv[2]) : 1, t_hi = argc > 3 ? std::atoi(argv[3]) : 8;
    const int cold_sets = argc > 4 ? std::atoi(argv[4]) : 16;
    const int N = 2560, HC = 4, D = N * HC, LR = 320, TM = K::kFusedGrMaxT, NE = 512, KT = 10;
    K::fused_gr_check();
    cudaStream_t s;
    cudaStreamCreate(&s);
    std::mt19937 rng(11);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> R((size_t) TM * D), bo((size_t) TM * N), inj((size_t) TM * HC), wn(D);
    for (auto& x : R) x = nd(rng);
    for (auto& x : bo) x = 0.5f * nd(rng);
    for (auto& x : inj) x = nd(rng);
    for (auto& x : wn) x = 1.0f + 0.3f * nd(rng);
    const int S = std::max(1, cold_sets);
    // weights: S sets of (down, up, inject, router) bf16 and their hc8 copies
    std::vector<uint16_t*> cwd(S), cwu(S), cwi(S), cwr(S);
    std::vector<const int8_t*> qd(S), qu(S), qi(S);
    std::vector<const uint16_t*> sd(S), su(S), si(S);
    {
        std::vector<uint16_t> wd((size_t) LR * D), wu((size_t) D * LR), wi((size_t) HC * D), wr((size_t) NE * N);
        for (int c = 0; c < S; ++c) {
            for (auto& x : wd) x = bf16(0.02f * nd(rng));
            for (auto& x : wu) x = bf16(0.05f * nd(rng));
            for (auto& x : wi) x = bf16(0.02f * nd(rng));
            for (auto& x : wr) x = bf16(0.03f * nd(rng));
            cudaMalloc((void**) &cwd[c], wd.size() * 2); cudaMemcpy(cwd[c], wd.data(), wd.size() * 2, cudaMemcpyHostToDevice);
            cudaMalloc((void**) &cwu[c], wu.size() * 2); cudaMemcpy(cwu[c], wu.data(), wu.size() * 2, cudaMemcpyHostToDevice);
            cudaMalloc((void**) &cwi[c], wi.size() * 2); cudaMemcpy(cwi[c], wi.data(), wi.size() * 2, cudaMemcpyHostToDevice);
            cudaMalloc((void**) &cwr[c], wr.size() * 2); cudaMemcpy(cwr[c], wr.data(), wr.size() * 2, cudaMemcpyHostToDevice);
            void *b8d, *b8u, *b8i;
            cudaMalloc(&b8d, K::hc8_bytes(LR, D)); cudaMalloc(&b8u, K::hc8_bytes(D, LR)); cudaMalloc(&b8i, K::hc8_bytes(HC, D));
            K::hc8_quantize(cwd[c], LR, D, b8d, &qd[c], &sd[c], s);
            K::hc8_quantize(cwu[c], D, LR, b8u, &qu[c], &su[c], s);
            K::hc8_quantize(cwi[c], HC, D, b8i, &qi[c], &si[c], s);
        }
    }
    float *dR, *dbo, *dinj, *dwn, *dxn, *dlo, *drs, *dio, *dmix, *dlog, *dw;
    int32_t* dids;
    void* dctr;
    cudaMalloc((void**) &dR, R.size() * 4);
    cudaMalloc((void**) &dbo, bo.size() * 4);
    cudaMalloc((void**) &dinj, inj.size() * 4);
    cudaMalloc((void**) &dwn, wn.size() * 4);
    cudaMalloc((void**) &dxn, (size_t) TM * D * 4);
    cudaMalloc((void**) &dlo, (size_t) TM * LR * 4);
    cudaMalloc((void**) &drs, (size_t) TM * HC * 4);
    cudaMalloc((void**) &dio, (size_t) TM * HC * 4);
    cudaMalloc((void**) &dmix, (size_t) TM * N * 4);
    cudaMalloc((void**) &dlog, (size_t) TM * NE * 4);
    cudaMalloc((void**) &dw, (size_t) TM * KT * 4);
    cudaMalloc((void**) &dids, (size_t) TM * KT * 4);
    cudaMalloc(&dctr, K::kHcFusedCtrBytes);
    cudaMemset(dctr, 0, K::kHcFusedCtrBytes);
    cudaMemcpy(dbo, bo.data(), bo.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dinj, inj.data(), inj.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dwn, wn.data(), wn.size() * 4, cudaMemcpyHostToDevice);

    auto args = [&](K::FusedGrArgs* a, int T, int c, bool h8, bool apply, bool inject) {
        for (int t = 0; t < T; ++t) {
            a[t] = K::FusedGrArgs();
            a[t].R = dR + (size_t) t * D; a[t].R_out = dR + (size_t) t * D; a[t].apply = apply;
            a[t].bo_prev = dbo + (size_t) t * N; a[t].inj_prev = dinj + (size_t) t * HC;
            a[t].w_norm = dwn; a[t].w_down = cwd[c]; a[t].w_up = cwu[c]; a[t].w_inject = inject ? cwi[c] : nullptr;
            a[t].lo = dlo + (size_t) t * LR; a[t].rs = drs + (size_t) t * HC;
            a[t].inject_out = inject ? dio + (size_t) t * HC : nullptr; a[t].mixed = dmix + (size_t) t * N;
            if (h8) {
                a[t].hc8_qd = qd[c]; a[t].hc8_sd = sd[c]; a[t].hc8_sd_row = K::hc8_srow(D);
                a[t].hc8_qu = qu[c]; a[t].hc8_su = su[c]; a[t].hc8_su_row = K::hc8_srow(LR);
                if (inject) { a[t].hc8_qi = qi[c]; a[t].hc8_si = si[c]; a[t].hc8_si_row = K::hc8_srow(D); }
            }
        }
    };
    auto unfused = [&](const K::FusedGrArgs* a, int T, int c, bool router) {
        K::fused_gr_read_multi(a, T, dxn, s);
        if (router) {
            K::bf16_gemv_fp32_mmvf_multi(dmix, N, cwr[c], dlog, NE, N, NE, T, s);
            K::native_router_top10_multi(dlog, dids, dw, T, s);
        }
    };
    auto fused = [&](const K::FusedGrArgs* a, int T, int c, bool router) {
        K::HcFusedRouter rt;
        rt.w = cwr[c]; rt.logits = dlog; rt.ids = dids; rt.weights = dw;
        return K::fused_gr_read_fused(a, T, dxn, dctr, router ? &rt : nullptr, s);
    };
    auto snap = [&](int T, std::vector<uint32_t>& o) {
        o.clear();
        auto add = [&](const void* d, size_t n4) {
            const size_t at = o.size();
            o.resize(at + n4);
            cudaMemcpy(o.data() + at, d, n4 * 4, cudaMemcpyDeviceToHost);
        };
        add(dR, (size_t) T * D); add(dlo, (size_t) T * LR); add(drs, (size_t) T * HC); add(dio, (size_t) T * HC);
        add(dmix, (size_t) T * N); add(dlog, (size_t) T * NE); add(dids, (size_t) T * KT); add(dw, (size_t) T * KT);
    };
    int failures = 0, checks = 0;
    for (int h8 = 0; h8 < 2; ++h8)
        for (int T = t_lo; T <= t_hi; ++T)
            for (int apply = 0; apply < 2; ++apply)
                for (int inject = 0; inject < 2; ++inject)
                    for (int router = 0; router < 2; ++router) {
                        if (router && T < 2) continue;
                        const int c = (T + apply + inject) % S;
                        K::FusedGrArgs a[TM];
                        args(a, T, c, h8, apply, inject);
                        std::vector<uint32_t> ref, got;
                        for (int f = 0; f < 2; ++f) {
                            cudaMemcpy(dR, R.data(), R.size() * 4, cudaMemcpyHostToDevice);
                            cudaMemsetAsync(dlo, 0, (size_t) TM * LR * 4, s); cudaMemsetAsync(drs, 0, (size_t) TM * HC * 4, s);
                            cudaMemsetAsync(dio, 0, (size_t) TM * HC * 4, s); cudaMemsetAsync(dmix, 0, (size_t) TM * N * 4, s);
                            cudaMemsetAsync(dlog, 0, (size_t) TM * NE * 4, s); cudaMemsetAsync(dw, 0, (size_t) TM * KT * 4, s);
                            cudaMemsetAsync(dids, 0, (size_t) TM * KT * 4, s);
                            int r = 0;
                            if (f == 0) unfused(a, T, c, router);
                            else r = fused(a, T, c, router);
                            cudaStreamSynchronize(s);
                            const cudaError_t e = cudaGetLastError();
                            if (e != cudaSuccess) { std::printf("CUDA error %s\n", cudaGetErrorString(e)); return 3; }
                            if (f == 1 && r != (router ? 2 : 1)) {
                                std::printf("fused read did not run as asked (h8 %d T %d router %d: %d)\n", h8, T, router, r);
                                ++failures;
                            }
                            snap(T, f == 0 ? ref : got);
                        }
                        ++checks;
                        size_t diff = 0, first = (size_t) -1;
                        for (size_t i = 0; i < ref.size(); ++i)
                            if (ref[i] != got[i]) { if (!diff) first = i; ++diff; }
                        if (diff) {
                            ++failures;
                            std::printf("DIFF %s T %d apply %d inject %d router %d: %zu words differ (first at %zu of %zu)\n",
                                        h8 ? "hc8 " : "bf16", T, apply, inject, router, diff, first, ref.size());
                        }
                    }
    std::printf("parity: %d of %d configurations bitwise equal\n", checks - failures, checks);

    if (std::getenv("HCF_TRACE") != nullptr) {   // one cold read per mode at T = HCF_TRACE: every item's timeline
        const int T = std::atoi(std::getenv("HCF_TRACE"));
        unsigned long long* dtr;
        const int maxi = 4000 + 4 * 200;
        cudaMalloc((void**) &dtr, (size_t) maxi * 5 * 8);
        for (int h8 = 0; h8 < 2; ++h8)
            for (int router = 0; router < 2; ++router) {
                K::FusedGrArgs a[TM];
                for (int w = 0; w < 4000; ++w) { args(a, T, w % S, h8, true, true); fused(a, T, w % S, router); }   // clocks up
                cudaStreamSynchronize(s);
                cudaMemset(dtr, 0, (size_t) maxi * 5 * 8);
                K::hcf_trace_set(dtr);
                args(a, T, 0, h8, true, true);
                fused(a, T, 0, router);
                cudaStreamSynchronize(s);
                K::hcf_trace_set(nullptr);
                std::vector<unsigned long long> h((size_t) maxi * 5);
                cudaMemcpy(h.data(), dtr, h.size() * 8, cudaMemcpyDeviceToHost);
                unsigned long long t0 = ~0ull, t1 = 0;
                for (int i = 0; i < 512; ++i) if (h[i * 5 + 4]) { t0 = std::min(t0, h[i * 5 + 2]); t1 = std::max(t1, h[i * 5 + 4]); }
                std::printf("TRACE %s router %d T %d grid %d: total %.1f us\n", h8 ? "hc8" : "bf16", router, T, K::hcf_last_grid(), (t1 - t0) / 1e3);
                for (int g = 0; g < 41; g += 8) {
                    std::printf("  down %2d tiles:", g);
                    for (int x = 0; x < 10; ++x) { const unsigned long long v = h[2560 + g * 16 + x]; std::printf(" %5.1f", v ? (v - t0) / 1e3 : -1.0); }
                    std::printf("\n");
                }
                for (int u = 0; u < 160; u += 10) {
                    const unsigned long long* x = &h[4000 + u * 4];
                    if (x[0]) std::printf("  up %3d lo %5.1f cp %5.1f row0 %5.1f rows %5.1f\n", u, (x[0] - t0) / 1e3, (x[1] - t0) / 1e3, (x[3] - t0) / 1e3, (x[2] - t0) / 1e3);
                }
                for (int i = 0; i < 512; ++i) {
                    if (!h[i * 5 + 4]) continue;
                    const double c = (h[i * 5 + 2] - t0) / 1e3, w = h[i * 5 + 3] ? (h[i * 5 + 3] - t0) / 1e3 : -1, d = (h[i * 5 + 4] - t0) / 1e3;
                    std::printf("  item %3d sm %2llu claim %6.1f waited %6.1f done %6.1f\n", i, h[i * 5 + 1], c, w, d);
                }
            }
    }
    if (cold_sets > 1) {
        cudaEvent_t e0, e1;
        cudaEventCreate(&e0); cudaEventCreate(&e1);
        cudaMemcpy(dR, R.data(), R.size() * 4, cudaMemcpyHostToDevice);
        std::vector<K::FusedGrArgs> as((size_t) S * TM);
        for (int h8 = 0; h8 < 2; ++h8)
            for (int router = 0; router < 2; ++router)
                for (int T = std::max(t_lo, router ? 2 : 1); T <= t_hi; ++T) {
                    for (int c = 0; c < S; ++c) args(&as[(size_t) c * TM], T, c, h8, true, true);
                    float ms[2];
                    for (int f = 0; f < 2; ++f) {
                        for (int i = 0; i < 2000; ++i) {   // warm-up (clocks up)
                            const int c = i % S;
                            if (f == 0) unfused(&as[(size_t) c * TM], T, c, router);
                            else fused(&as[(size_t) c * TM], T, c, router);
                        }
                        cudaEventRecord(e0, s);
                        for (int i = 0; i < iters; ++i) {
                            const int c = i % S;
                            if (f == 0) unfused(&as[(size_t) c * TM], T, c, router);
                            else fused(&as[(size_t) c * TM], T, c, router);
                        }
                        cudaEventRecord(e1, s);
                        cudaEventSynchronize(e1);
                        cudaEventElapsedTime(&ms[f], e0, e1);
                    }
                    std::printf("COLD(%d sets) %s %s T %d | unfused %6.1f us | fused %6.1f us | %.2fx\n", S,
                                h8 ? "hc8 " : "bf16", router ? "read+router" : "read       ", T, 1e3 * ms[0] / iters,
                                1e3 * ms[1] / iters, ms[0] / ms[1]);
                }
        // the same in a CUDA graph (the engine replays graphs: no host launch gaps), T = 4
        for (int h8 = 0; h8 < 2; ++h8)
            for (int router = 0; router < 2; ++router) {
                const int T = std::min(std::max(4, t_lo), t_hi);
                for (int c = 0; c < S; ++c) args(&as[(size_t) c * TM], T, c, h8, true, true);
                float ms[2];
                for (int f = 0; f < 2; ++f) {
                    cudaGraph_t g;
                    cudaGraphExec_t ge;
                    cudaStreamBeginCapture(s, cudaStreamCaptureModeRelaxed);
                    for (int i = 0; i < 48; ++i) {
                        const int c = i % S;
                        if (f == 0) unfused(&as[(size_t) c * TM], T, c, router);
                        else fused(&as[(size_t) c * TM], T, c, router);
                    }
                    cudaStreamEndCapture(s, &g);
                    cudaGraphInstantiate(&ge, g, 0);
                    for (int w = 0; w < 3; ++w) cudaGraphLaunch(ge, s);
                    cudaEventRecord(e0, s);
                    const int reps = std::max(1, iters / 48);
                    for (int w = 0; w < reps; ++w) cudaGraphLaunch(ge, s);
                    cudaEventRecord(e1, s);
                    cudaEventSynchronize(e1);
                    cudaEventElapsedTime(&ms[f], e0, e1);
                    ms[f] /= (float) (reps * 48);
                    cudaGraphExecDestroy(ge);
                    cudaGraphDestroy(g);
                }
                std::printf("GRAPH(%d sets) %s %s T %d | unfused %6.1f us | fused %6.1f us | %.2fx\n", S, h8 ? "hc8 " : "bf16",
                            router ? "read+router" : "read       ", T, 1e3 * ms[0], 1e3 * ms[1], ms[0] / ms[1]);
            }
    }
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { std::printf("CUDA error %s\n", cudaGetErrorString(e)); return 3; }
    std::printf("hcf_bench: %d differing\n", failures);
    return failures ? 1 : 0;
}
