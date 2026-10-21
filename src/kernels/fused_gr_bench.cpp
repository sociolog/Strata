// src/kernels/fused_gr_bench.cpp - fused_gr_read_multi: the AMD fast kernels (STRATA_GR_FAST) against the old
// ones, bit for bit on every output (lo, rs, inject, mixed, the in-place R), and each one's time.
//
//     build/fused_gr_bench [iters] [T min] [T max] [cold sets]
// With "cold sets" > 1 it also times the read with the weights rotating over that many copies (a layer's weights are
// cold in the engine) and prints the per-stage split (norm / down / up) from the GPU stamps.
#include "strata/kernels/fused_gr.hpp"
#include "strata/kernels/verify_kernels.hpp"

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

// Q8_0 (ggml reference rounding: d = amax / 127 as fp16, q = round(x / d)) of a row-major [rows][cols] bf16 matrix
static std::vector<uint8_t> q8_0_of_bf16(const std::vector<uint16_t>& w, size_t cols) {
    std::vector<uint8_t> out(w.size() / 32 * 34);
    for (size_t b = 0; b < w.size() / 32; ++b) {
        float v[32], amax = 0.f;
        for (int i = 0; i < 32; ++i) {
            uint32_t u = (uint32_t) w[b * 32 + i] << 16;
            std::memcpy(&v[i], &u, 4);
            amax = std::fmax(amax, std::fabs(v[i]));
        }
        const float d = amax / 127.f, id = d != 0.f ? 1.f / d : 0.f;
        // fp16 of d
        uint32_t x; std::memcpy(&x, &d, 4);
        const uint32_t sign = (x >> 16) & 0x8000u;
        int32_t e = (int32_t) ((x >> 23) & 0xff) - 127 + 15;
        uint32_t m = x & 0x7fffffu;
        uint16_t h;
        if (e <= 0) h = (uint16_t) sign;   // flush denormals: weights here are far from them
        else if (e >= 31) h = (uint16_t) (sign | 0x7c00u);
        else h = (uint16_t) (sign | ((uint32_t) e << 10) | (m >> 13));
        uint8_t* o = &out[b * 34];
        std::memcpy(o, &h, 2);
        for (int i = 0; i < 32; ++i) o[2 + i] = (uint8_t) (int8_t) std::lround(v[i] * id);
    }
    (void) cols;
    return out;
}

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    const int iters = argc > 1 ? std::atoi(argv[1]) : 500;
    const int t_lo = argc > 2 ? std::atoi(argv[2]) : 1, t_hi = argc > 3 ? std::atoi(argv[3]) : 6;
    const int cold_sets = argc > 4 ? std::atoi(argv[4]) : 0;
    const int N = 2560, HC = 4, D = N * HC, LR = 320, TM = K::kFusedGrMaxT;
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 1.f);
    auto fill = [&](std::vector<float>& v, float sc) { for (auto& x : v) x = sc * nd(rng); };
    std::vector<float> R((size_t) TM * D), bo((size_t) TM * N), inj((size_t) TM * HC), wn(D);
    fill(R, 1.0f); fill(bo, 0.5f); fill(inj, 1.0f); fill(wn, 0.3f);
    for (auto& x : wn) x += 1.0f;
    std::vector<uint16_t> wd((size_t) LR * D), wu((size_t) D * LR), wi((size_t) HC * D);
    for (auto& x : wd) x = bf16(0.02f * nd(rng));
    for (auto& x : wu) x = bf16(0.05f * nd(rng));
    for (auto& x : wi) x = bf16(0.02f * nd(rng));
    float *dR, *dbo, *dinj, *dwn, *dxn, *dlo, *drs, *dio, *dmix;
    uint16_t *dwd, *dwu, *dwi;
    cudaMalloc((void**) &dR, R.size() * 4);
    cudaMalloc((void**) &dbo, bo.size() * 4);
    cudaMalloc((void**) &dinj, inj.size() * 4);
    cudaMalloc((void**) &dwn, wn.size() * 4);
    cudaMalloc((void**) &dxn, (size_t) TM * D * 4);
    cudaMalloc((void**) &dlo, (size_t) TM * LR * 4);
    cudaMalloc((void**) &drs, (size_t) TM * HC * 4);
    cudaMalloc((void**) &dio, (size_t) TM * HC * 4);
    cudaMalloc((void**) &dmix, (size_t) TM * N * 4);
    cudaMalloc((void**) &dwd, wd.size() * 2);
    cudaMalloc((void**) &dwu, wu.size() * 2);
    cudaMalloc((void**) &dwi, wi.size() * 2);
    cudaMemcpy(dbo, bo.data(), bo.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dinj, inj.data(), inj.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dwn, wn.data(), wn.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dwd, wd.data(), wd.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dwu, wu.data(), wu.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dwi, wi.data(), wi.size() * 2, cudaMemcpyHostToDevice);
    cudaStream_t s;
    cudaStreamCreate(&s);
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    int failures = 0;
    for (int T = t_lo; T <= t_hi; ++T)
        for (int apply = 0; apply < 2; ++apply)
            for (int inject = 0; inject < 2; ++inject) {
                K::FusedGrArgs a[TM];
                for (int t = 0; t < T; ++t) {
                    a[t].R = dR + (size_t) t * D; a[t].R_out = dR + (size_t) t * D; a[t].apply = apply;
                    a[t].bo_prev = dbo + (size_t) t * N; a[t].inj_prev = dinj + (size_t) t * HC;
                    a[t].w_norm = dwn; a[t].w_down = dwd; a[t].w_up = dwu; a[t].w_inject = inject ? dwi : nullptr;
                    a[t].lo = dlo + (size_t) t * LR; a[t].rs = drs + (size_t) t * HC;
                    a[t].inject_out = dio + (size_t) t * HC; a[t].mixed = dmix + (size_t) t * N;
                }
                std::vector<float> out[2];
                for (int f = 0; f < 2; ++f) {
                    cudaMemcpy(dR, R.data(), R.size() * 4, cudaMemcpyHostToDevice);
                    cudaMemset(dlo, 0xff, (size_t) TM * LR * 4);
                    cudaMemset(drs, 0xff, (size_t) TM * HC * 4);
                    cudaMemset(dio, 0xff, (size_t) TM * HC * 4);
                    cudaMemset(dmix, 0xff, (size_t) TM * N * 4);
                    K::fused_gr_set_fast(f);
                    K::fused_gr_read_multi(a, T, dxn, s);
                    cudaStreamSynchronize(s);
                    auto& o = out[f];
                    o.resize((size_t) TM * (D + LR + HC + HC + N));
                    float* p = o.data();
                    cudaMemcpy(p, dR, (size_t) TM * D * 4, cudaMemcpyDeviceToHost); p += (size_t) TM * D;
                    cudaMemcpy(p, dlo, (size_t) TM * LR * 4, cudaMemcpyDeviceToHost); p += (size_t) TM * LR;
                    cudaMemcpy(p, drs, (size_t) TM * HC * 4, cudaMemcpyDeviceToHost); p += (size_t) TM * HC;
                    cudaMemcpy(p, dio, (size_t) TM * HC * 4, cudaMemcpyDeviceToHost); p += (size_t) TM * HC;
                    cudaMemcpy(p, dmix, (size_t) TM * N * 4, cudaMemcpyDeviceToHost);
                }
                const bool same = std::memcmp(out[0].data(), out[1].data(), out[0].size() * 4) == 0;
                double us[2] = {1e30, 1e30};
                for (int round = 0; round < 3; ++round)
                    for (int f = 0; f < 2; ++f) {
                        K::fused_gr_set_fast(f);
                        for (int i = 0; i < 20; ++i) K::fused_gr_read_multi(a, T, dxn, s);
                        cudaEventRecord(e0, s);
                        for (int i = 0; i < iters; ++i) K::fused_gr_read_multi(a, T, dxn, s);
                        cudaEventRecord(e1, s);
                        cudaEventSynchronize(e1);
                        float ms = 0;
                        cudaEventElapsedTime(&ms, e0, e1);
                        us[f] = std::fmin(us[f], 1e3 * ms / iters);
                    }
                unsigned long long h = 1469598103934665603ull;   // FNV-1a of the outputs: compare across runs (env switches)
                for (size_t b = 0; b < out[0].size() * 4; ++b) h = (h ^ ((const unsigned char*) out[0].data())[b]) * 1099511628211ull;
                std::printf("T %d apply %d inject %d | old %6.1f us  fast %6.1f us | %s | hash %016llx\n", T, apply, inject,
                            us[0], us[1], same ? "bitwise equal" : "DIFFERS", h);
                if (!same) ++failures;
            }
    {   // the Q8_0 read against the bf16 read: relative RMS error of lo / mixed / R / inject, and the q8_1 image against mixed
        const std::vector<uint8_t> qd = q8_0_of_bf16(wd, D), qu = q8_0_of_bf16(wu, LR);
        uint8_t *dqd, *dqu, *dqm; unsigned* dcnt;
        cudaMalloc((void**) &dqd, qd.size()); cudaMemcpy(dqd, qd.data(), qd.size(), cudaMemcpyHostToDevice);
        cudaMalloc((void**) &dqu, qu.size()); cudaMemcpy(dqu, qu.data(), qu.size(), cudaMemcpyHostToDevice);
        cudaMalloc((void**) &dqm, (size_t) TM * (N / 32) * 36); cudaMalloc((void**) &dcnt, (N / 32) * 4);
        cudaMemset(dcnt, 0, (N / 32) * 4);
        for (int T = t_lo; T <= t_hi; ++T) for (int apply = 0; apply < 2; ++apply) {
            K::FusedGrArgs a[TM];
            for (int t = 0; t < T; ++t) {
                a[t].R = dR + (size_t) t * D; a[t].R_out = dR + (size_t) t * D; a[t].apply = apply;
                a[t].bo_prev = dbo + (size_t) t * N; a[t].inj_prev = dinj + (size_t) t * HC;
                a[t].w_norm = dwn; a[t].w_down = dwd; a[t].w_up = dwu; a[t].w_inject = dwi;
                a[t].lo = dlo + (size_t) t * LR; a[t].rs = drs + (size_t) t * HC;
                a[t].inject_out = dio + (size_t) t * HC; a[t].mixed = dmix + (size_t) t * N;
            }
            std::vector<float> o[2];
            std::vector<uint8_t> img;
            for (int q = 0; q < 2; ++q) {
                for (int t = 0; t < T; ++t) {
                    a[t].q8_down = q ? dqd : nullptr; a[t].q8_up = q ? dqu : nullptr;
                    a[t].q8_mixed = (uint8_t*) dqm + (size_t) t * (N / 32) * 36; a[t].q8_cnt = dcnt;
                }
                cudaMemcpy(dR, R.data(), R.size() * 4, cudaMemcpyHostToDevice);
                cudaMemset(dqm, 0, (size_t) TM * (N / 32) * 36);
                K::fused_gr_read_multi(a, T, dxn, s);
                cudaStreamSynchronize(s);
                o[q].resize((size_t) T * (LR + N + D + HC));
                float* p = o[q].data();
                cudaMemcpy(p, dlo, (size_t) T * LR * 4, cudaMemcpyDeviceToHost); p += (size_t) T * LR;
                cudaMemcpy(p, dmix, (size_t) T * N * 4, cudaMemcpyDeviceToHost); p += (size_t) T * N;
                cudaMemcpy(p, dR, (size_t) T * D * 4, cudaMemcpyDeviceToHost); p += (size_t) T * D;
                cudaMemcpy(p, dio, (size_t) T * HC * 4, cudaMemcpyDeviceToHost);
                if (q) { img.resize((size_t) T * (N / 32) * 36); cudaMemcpy(img.data(), dqm, img.size(), cudaMemcpyDeviceToHost); }
            }
            unsigned long long hq = 1469598103934665603ull;   // FNV-1a of the Q8 read's outputs and q8_1 image
            for (size_t b = 0; b < o[1].size() * 4; ++b) hq = (hq ^ ((const unsigned char*) o[1].data())[b]) * 1099511628211ull;
            for (size_t b = 0; b < img.size(); ++b) hq = (hq ^ img[b]) * 1099511628211ull;
            double num = 0, den = 0, nm = 0, dm = 0;
            for (size_t i = 0; i < (size_t) T * LR; ++i) { num += std::pow(o[1][i] - o[0][i], 2); den += std::pow(o[0][i], 2); }
            for (size_t i = (size_t) T * LR; i < (size_t) T * (LR + N); ++i) { nm += std::pow(o[1][i] - o[0][i], 2); dm += std::pow(o[0][i], 2); }
            double rn = 0;   // R (apply) and inject must be identical or close
            for (size_t i = (size_t) T * (LR + N); i < o[0].size(); ++i) rn = std::fmax(rn, std::fabs(o[1][i] - o[0][i]));
            // q8_1 image: 36 bytes = half2 (d, sum) + 32 int8; dequantized against mixed
            double qn = 0, qd2 = 0;
            for (int t = 0; t < T; ++t) for (int g = 0; g < N / 32; ++g) {
                const uint8_t* b = &img[((size_t) t * (N / 32) + g) * 36];
                uint16_t h; std::memcpy(&h, b, 2);
                const uint32_t sg = (h >> 15) & 1, ex = (h >> 10) & 31, mn = h & 1023;
                const float d = (ex ? std::ldexp(1.0f + mn / 1024.0f, (int) ex - 15) : std::ldexp(mn / 1024.0f, -14)) * (sg ? -1.f : 1.f);
                for (int j = 0; j < 32; ++j) {
                    const float ref = o[1][(size_t) T * LR + (size_t) t * N + g * 32 + j];
                    qn += std::pow(d * (int8_t) b[4 + j] - ref, 2); qd2 += ref * ref;
                }
            }
            std::printf("Q8 vs bf16 T %d apply %d | rel rms lo %.4f mixed %.4f | max |dR,dinject| %.3g | q8_1 image vs mixed rel rms %.4f | hash %016llx\n", T, apply,
                        std::sqrt(num / den), std::sqrt(nm / dm), rn, std::sqrt(qn / qd2), hq);
        }
    }
    {   // the hc8 read (STRATA_EXL3_HC8) against the bf16 read: relative RMS error of lo / mixed / inject, R exact
        void *b_d, *b_u, *b_i;
        cudaMalloc(&b_d, K::hc8_bytes(LR, D)); cudaMalloc(&b_u, K::hc8_bytes(D, LR)); cudaMalloc(&b_i, K::hc8_bytes(HC, D));
        const int8_t *qd8, *qu8, *qi8; const uint16_t *sd8, *su8, *si8;
        K::hc8_quantize(dwd, LR, D, b_d, &qd8, &sd8, s);
        K::hc8_quantize(dwu, D, LR, b_u, &qu8, &su8, s);
        K::hc8_quantize(dwi, HC, D, b_i, &qi8, &si8, s);
        uint8_t* dqm; unsigned* dcnt;
        cudaMalloc((void**) &dqm, (size_t) TM * (N / 32) * 36); cudaMalloc((void**) &dcnt, (N / 32) * 4);
        cudaMemset(dcnt, 0, (N / 32) * 4);
        for (int T = t_lo; T <= t_hi; ++T) for (int apply = 0; apply < 2; ++apply) for (int qm = 0; qm < 2; ++qm) {
            K::FusedGrArgs a[TM];
            std::vector<float> o[2];
            std::vector<uint8_t> img[2];
            for (int q = 0; q < 2; ++q) {
                for (int t = 0; t < T; ++t) {
                    a[t] = K::FusedGrArgs();
                    a[t].R = dR + (size_t) t * D; a[t].R_out = dR + (size_t) t * D; a[t].apply = apply;
                    a[t].bo_prev = dbo + (size_t) t * N; a[t].inj_prev = dinj + (size_t) t * HC;
                    a[t].w_norm = dwn; a[t].w_down = dwd; a[t].w_up = dwu; a[t].w_inject = dwi;
                    a[t].lo = dlo + (size_t) t * LR; a[t].rs = drs + (size_t) t * HC;
                    a[t].inject_out = dio + (size_t) t * HC; a[t].mixed = dmix + (size_t) t * N;
                    if (qm) { a[t].q8_mixed = dqm + (size_t) t * (N / 32) * 36; a[t].q8_cnt = dcnt; }
                    if (q) {
                        a[t].hc8_qd = qd8; a[t].hc8_sd = sd8; a[t].hc8_sd_row = K::hc8_srow(D);
                        a[t].hc8_qu = qu8; a[t].hc8_su = su8; a[t].hc8_su_row = K::hc8_srow(LR);
                        a[t].hc8_qi = qi8; a[t].hc8_si = si8; a[t].hc8_si_row = K::hc8_srow(D);
                    }
                }
                cudaMemcpy(dR, R.data(), R.size() * 4, cudaMemcpyHostToDevice);
                cudaMemset(dqm, 0, (size_t) TM * (N / 32) * 36);
                const bool wrote = K::fused_gr_read_multi(a, T, dxn, s);
                cudaStreamSynchronize(s);
                if (wrote != (qm == 1)) { std::printf("hc8: q8_1 written flag wrong (q %d qm %d)\n", q, qm); ++failures; }
                o[q].resize((size_t) T * (LR + N + D + HC));
                float* p = o[q].data();
                cudaMemcpy(p, dlo, (size_t) T * LR * 4, cudaMemcpyDeviceToHost); p += (size_t) T * LR;
                cudaMemcpy(p, dmix, (size_t) T * N * 4, cudaMemcpyDeviceToHost); p += (size_t) T * N;
                cudaMemcpy(p, dR, (size_t) T * D * 4, cudaMemcpyDeviceToHost); p += (size_t) T * D;
                cudaMemcpy(p, dio, (size_t) T * HC * 4, cudaMemcpyDeviceToHost);
                img[q].resize((size_t) T * (N / 32) * 36);
                cudaMemcpy(img[q].data(), dqm, img[q].size(), cudaMemcpyDeviceToHost);
            }
            double num = 0, den = 0, nm = 0, dm = 0, ni = 0, di = 0, rn = 0;
            for (size_t i = 0; i < (size_t) T * LR; ++i) { num += std::pow(o[1][i] - o[0][i], 2); den += std::pow(o[0][i], 2); }
            for (size_t i = (size_t) T * LR; i < (size_t) T * (LR + N); ++i) { nm += std::pow(o[1][i] - o[0][i], 2); dm += std::pow(o[0][i], 2); }
            for (size_t i = (size_t) T * (LR + N); i < (size_t) T * (LR + N + D); ++i) rn = std::fmax(rn, std::fabs(o[1][i] - o[0][i]));
            for (size_t i = (size_t) T * (LR + N + D); i < o[0].size(); ++i) { ni += std::pow(o[1][i] - o[0][i], 2); di += std::pow(o[0][i], 2); }
            // q8_1 image of the hc8 read: dequantized against its own mixed
            double qn = 0, qd2 = 0;
            if (qm) for (int t = 0; t < T; ++t) for (int g = 0; g < N / 32; ++g) {
                const uint8_t* b = &img[1][((size_t) t * (N / 32) + g) * 36];
                uint16_t h; std::memcpy(&h, b, 2);
                const uint32_t sg = (h >> 15) & 1, ex = (h >> 10) & 31, mn = h & 1023;
                const float d = (ex ? std::ldexp(1.0f + mn / 1024.0f, (int) ex - 15) : std::ldexp(mn / 1024.0f, -14)) * (sg ? -1.f : 1.f);
                for (int j = 0; j < 32; ++j) {
                    const float ref = o[1][(size_t) T * LR + (size_t) t * N + g * 32 + j];
                    qn += std::pow(d * (int8_t) b[4 + j] - ref, 2); qd2 += ref * ref;
                }
            }
            std::printf("hc8 vs bf16 T %d apply %d qfuse %d | rel rms lo %.5f mixed %.5f inject %.5f | max |dR| %.3g | q8_1 image vs mixed rel rms %.4f\n",
                        T, apply, qm, std::sqrt(num / den), std::sqrt(nm / dm), std::sqrt(ni / di), rn, qd2 > 0 ? std::sqrt(qn / qd2) : 0.0);
            if (!(std::sqrt(nm / dm) < 0.02) || rn > 1e-4 || (qm && !(std::sqrt(qn / qd2) < 0.02))) ++failures;
        }
        // T-invariance: token 0's outputs of a T-token read equal a 1-token read's, bit for bit
        {
            std::vector<float> ref, got;
            for (int T : {1, t_hi}) {
                K::FusedGrArgs a[TM];
                for (int t = 0; t < T; ++t) {
                    a[t] = K::FusedGrArgs();
                    a[t].R = dR + (size_t) t * D; a[t].R_out = dR + (size_t) t * D; a[t].apply = 1;
                    a[t].bo_prev = dbo + (size_t) t * N; a[t].inj_prev = dinj + (size_t) t * HC;
                    a[t].w_norm = dwn; a[t].w_inject = dwi;
                    a[t].lo = dlo + (size_t) t * LR; a[t].rs = drs + (size_t) t * HC;
                    a[t].inject_out = dio + (size_t) t * HC; a[t].mixed = dmix + (size_t) t * N;
                    a[t].hc8_qd = qd8; a[t].hc8_sd = sd8; a[t].hc8_sd_row = K::hc8_srow(D);
                    a[t].hc8_qu = qu8; a[t].hc8_su = su8; a[t].hc8_su_row = K::hc8_srow(LR);
                    a[t].hc8_qi = qi8; a[t].hc8_si = si8; a[t].hc8_si_row = K::hc8_srow(D);
                }
                cudaMemcpy(dR, R.data(), R.size() * 4, cudaMemcpyHostToDevice);
                K::fused_gr_read_multi(a, T, dxn, s);
                cudaStreamSynchronize(s);
                std::vector<float>& o = T == 1 ? ref : got;
                o.resize(LR + N + HC);
                cudaMemcpy(o.data(), dlo, LR * 4, cudaMemcpyDeviceToHost);
                cudaMemcpy(o.data() + LR, dmix, N * 4, cudaMemcpyDeviceToHost);
                cudaMemcpy(o.data() + LR + N, dio, HC * 4, cudaMemcpyDeviceToHost);
            }
            const bool same = std::memcmp(ref.data(), got.data(), ref.size() * 4) == 0;
            std::printf("hc8 T-invariance (token 0 of T=1 vs T=%d): %s\n", t_hi, same ? "bitwise equal" : "DIFFERS");
            if (!same) ++failures;
        }
    }
    if (cold_sets > 1) {
        // the engine reads a different layer's weights every time: rotate over `cold_sets` copies (13 MB each)
        std::vector<uint16_t*> cwd(cold_sets), cwu(cold_sets), cwi(cold_sets);
        const std::vector<uint8_t> qd = q8_0_of_bf16(wd, D), qu = q8_0_of_bf16(wu, LR);
        std::vector<uint8_t*> cqd(cold_sets), cqu(cold_sets);
        std::vector<float*> cwn(cold_sets);
        std::vector<const int8_t*> c8d(cold_sets), c8u(cold_sets), c8i(cold_sets);   // the hc8 copies (f 3)
        std::vector<const uint16_t*> s8d(cold_sets), s8u(cold_sets), s8i(cold_sets);
        for (int c = 0; c < cold_sets; ++c) {
            cudaMalloc((void**) &cwd[c], wd.size() * 2); cudaMemcpy(cwd[c], wd.data(), wd.size() * 2, cudaMemcpyHostToDevice);
            cudaMalloc((void**) &cwu[c], wu.size() * 2); cudaMemcpy(cwu[c], wu.data(), wu.size() * 2, cudaMemcpyHostToDevice);
            cudaMalloc((void**) &cwi[c], wi.size() * 2); cudaMemcpy(cwi[c], wi.data(), wi.size() * 2, cudaMemcpyHostToDevice);
            cudaMalloc((void**) &cqd[c], qd.size()); cudaMemcpy(cqd[c], qd.data(), qd.size(), cudaMemcpyHostToDevice);
            cudaMalloc((void**) &cqu[c], qu.size()); cudaMemcpy(cqu[c], qu.data(), qu.size(), cudaMemcpyHostToDevice);
            cudaMalloc((void**) &cwn[c], wn.size() * 4); cudaMemcpy(cwn[c], wn.data(), wn.size() * 4, cudaMemcpyHostToDevice);
            void *b8d, *b8u, *b8i;
            cudaMalloc(&b8d, K::hc8_bytes(LR, D)); cudaMalloc(&b8u, K::hc8_bytes(D, LR)); cudaMalloc(&b8i, K::hc8_bytes(HC, D));
            K::hc8_quantize(cwd[c], LR, D, b8d, &c8d[c], &s8d[c], s);
            K::hc8_quantize(cwu[c], D, LR, b8u, &c8u[c], &s8u[c], s);
            K::hc8_quantize(cwi[c], HC, D, b8i, &c8i[c], &s8i[c], s);
        }
        uint8_t* dqm; unsigned* dcnt;   // the q8_1 image of mixed (STRATA_QFUSE_HC) as the engine asks for it
        cudaMalloc((void**) &dqm, (size_t) TM * (N / 32) * 36); cudaMalloc((void**) &dcnt, (N / 32) * 4);
        cudaMemset(dcnt, 0, (N / 32) * 4);
        const bool with_q = std::getenv("STRATA_BENCH_QMIXED") != nullptr;
        unsigned long long* dst;
        cudaMalloc((void**) &dst, (size_t) iters * 4 * 8);
        std::vector<unsigned long long> hst((size_t) iters * 4);
        std::vector<K::FusedGrArgs> as((size_t) cold_sets * 8);
        for (int T = t_lo; T <= t_hi; ++T) for (int f = 0; f < 4; ++f) {
            K::fused_gr_set_fast(f == 1);
            for (int c = 0; c < cold_sets; ++c)
                for (int t = 0; t < T; ++t) {
                    K::FusedGrArgs& a = as[(size_t) c * 8 + t];
                    a = K::FusedGrArgs();
                    a.R = dR + (size_t) t * D; a.R_out = dR + (size_t) t * D; a.apply = 1;
                    a.bo_prev = dbo + (size_t) t * N; a.inj_prev = dinj + (size_t) t * HC;
                    a.w_norm = cwn[c]; a.w_down = cwd[c]; a.w_up = cwu[c]; a.w_inject = cwi[c];
                    if (with_q) { a.q8_mixed = dqm + (size_t) t * (N / 32) * 36; a.q8_cnt = dcnt; }
                    if (f == 2) { a.q8_down = cqd[c]; a.q8_up = cqu[c]; }
                    if (f == 3) {
                        a.hc8_qd = c8d[c]; a.hc8_sd = s8d[c]; a.hc8_sd_row = K::hc8_srow(D);
                        a.hc8_qu = c8u[c]; a.hc8_su = s8u[c]; a.hc8_su_row = K::hc8_srow(LR);
                        a.hc8_qi = c8i[c]; a.hc8_si = s8i[c]; a.hc8_si_row = K::hc8_srow(D);
                    }
                    a.lo = dlo + (size_t) t * LR; a.rs = drs + (size_t) t * HC;
                    a.inject_out = dio + (size_t) t * HC; a.mixed = dmix + (size_t) t * N;
                }
            for (int rep = 0; rep < 2; ++rep) {
                for (int i = 0; i < iters; ++i) {
                    K::gpu_stamp(dst, i * 4, s);
                    K::fused_gr_read_multi(&as[(size_t) (i % cold_sets) * 8], T, dxn, s, dst, i * 4 + 1);
                    K::gpu_stamp(dst, i * 4 + 3, s);
                }
                cudaStreamSynchronize(s);
            }
            cudaEventRecord(e0, s);
            for (int i = 0; i < iters; ++i) K::fused_gr_read_multi(&as[(size_t) (i % cold_sets) * 8], T, dxn, s);
            cudaEventRecord(e1, s);
            cudaEventSynchronize(e1);
            float ms = 0;
            cudaEventElapsedTime(&ms, e0, e1);
            cudaMemcpy(hst.data(), dst, hst.size() * 8, cudaMemcpyDeviceToHost);
            // stamps: 0 start, 1 after norm, 2 after down, 3 after up (a ~1-2 us stamp kernel between each)
            std::vector<double> n_, d_, u_;
            for (int i = 1; i < iters; ++i) {
                const unsigned long long* p = &hst[(size_t) i * 4];
                n_.push_back((double) (p[1] - p[0])); d_.push_back((double) (p[2] - p[1])); u_.push_back((double) (p[3] - p[2]));
            }
            auto med = [](std::vector<double>& v) { std::sort(v.begin(), v.end()); return v[v.size() / 2] / 1e3; };
            std::printf("COLD(%d sets) %s T %d | read %6.1f us (events) | stamps median: norm %5.1f down %5.1f up %5.1f us (incl. a stamp kernel each)\n",
                        cold_sets, f == 3 ? "hc8 " : f == 2 ? "q8_0" : f == 1 ? "fast" : "bf16", T, 1e3 * ms / iters, med(n_), med(d_), med(u_));
        }
    }
    std::printf("fused_gr_bench: %d differing\n", failures);
    return failures ? 1 : 0;
}
