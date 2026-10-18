// src/kernels/cpu/exl3_lane_test.cpp - the EXL3 CPU lane (exl3_lane.cpp) against the scalar oracle (exl3.cpp) and
// an exact model of its own arithmetic.  No model, no GPU: any trellis bit pattern is a valid encoding, so the
// matrices are random words.
//
//   exl3_lane_test       run everything; exit 0 when every case passes
//
// 1. Layout: the oracle's tile order and windows (exl3_decode_tile through an identity table gives the raw states)
//    against the lane's bit model (the window ending at bit (p + 1 + 256) K of the tile's MSB-first word ring, p the
//    tensor-core step at (row, column)), K 1..8, every position of 64 random tiles each.
// 2. Tile by tile: one-hot activations through every tier's band kernel (1, 2, 3 and 8 at once) and split-K pieces
//    read each weight's byte sum s alone; fp16(fma(1024 + s, k_inv, k_bias)) must be the oracle's decoded weight and s
//    the byte sum of the oracle's state, for every weight of every tile, K 1..8.
// 3. Exactness (ported from project-maya's tests/exl3_test.cu): every tier against the exact model (the vector tiers
//    bit for bit, the scalar tier - no FMA - within rounding), several activations at once against one call each,
//    split-K pieces against bands, the 128-point transform, the prompt GEMM.
// 4. The whole product against the oracle's folded GEMV (fp16 weights there, so equal up to the fp16 rounding of each
//    weight), on the Flash's expert shapes: gate/up 2560 -> 640, down 640 -> 2560.
// 5. With --real <EXL3 model dir> (instead of 1-4): 33 routed-expert matrices of the model (layers 0, 23, 47 and the
//    MTP layer) through the checks above.
#include "strata/artifact/exl3_model.hpp"
#include "strata/kernels/cpu/exl3.hpp"
#include "strata/kernels/cpu/exl3_lane.hpp"
#include "strata/kernels/f16_bits.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace cpu = strata::kernels::cpu;
using strata::kernels::f16_from_f32;
using strata::kernels::f32_from_f16;

namespace {

int g_fail = 0;
std::mt19937_64 g_rng(1764);

void report(const std::string& what, bool ok, const std::string& detail) {
    std::printf("  %-52s %s  %s\n", what.c_str(), detail.c_str(), ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}

std::vector<uint32_t> rand_words(size_t n) {
    std::vector<uint32_t> w(n);
    for (auto& x : w) x = (uint32_t) g_rng();
    return w;
}
// channel scales: random sign, magnitude in [0.25, 2]
std::vector<uint16_t> rand_scales(int n) {
    std::uniform_real_distribution<float> u(0.25f, 2.0f);
    std::vector<uint16_t> v((size_t) n);
    for (auto& x : v) x = f16_from_f32((g_rng() & 1) ? u(g_rng) : -u(g_rng));
    return v;
}
std::vector<float> rand_x(int n, float scale = 1.0f) {
    std::normal_distribution<float> d(0.0f, scale);
    std::vector<float> v((size_t) n);
    for (auto& x : v) x = d(g_rng);
    return v;
}

struct HostMat {
    std::vector<uint32_t> tr;
    std::vector<uint16_t> suh, svh;
    cpu::Exl3Part part() const {
        cpu::Exl3Part m;
        m.trellis = tr.data();
        m.suh = suh.data();
        m.svh = svh.data();
        m.k = k;
        m.n = n;
        m.K = K;
        return m;
    }
    int k = 0, n = 0, K = 0;
};
HostMat make_mat(int k, int n, int K) {
    HostMat h;
    h.tr = rand_words((size_t) (k / 16) * (n / 16) * 8 * K);
    h.suh = rand_scales(k);
    h.svh = rand_scales(n);
    h.k = k;
    h.n = n;
    h.K = K;
    return h;
}

const char* tier_name(int tier) { return tier == 2 ? "avx512" : tier == 1 ? "avx2" : "scalar"; }

// ---------------------------------------------------------------- the lane's bit model (project-maya's exl3_test)
// the tile's step p at (row, column): row (p/8 % 4) 2 + (p & 1) + 8 (p/2 & 1), column p/32 + 8 (p/4 & 1)
struct StepMap {
    int step[16][16];
    StepMap() {
        for (int p = 0; p < 256; ++p) step[(p / 8 % 4) * 2 + (p & 1) + 8 * (p / 2 & 1)][p / 32 + 8 * (p / 4 & 1)] = p;
    }
};
const StepMap& step_map() {
    static const StepMap s;
    return s;
}
// the state at (r, c): the 16-bit window that ends at bit (p + 1 + 256) K of the tile's MSB-first bit ring, bit by bit
uint32_t model_state(const uint32_t* tile, int K, int r, int c) {
    const int bits = 256 * K, end = (step_map().step[r][c] + 1 + 256) * K;
    uint32_t st = 0;
    for (int q = end - 16; q < end; ++q) {
        const int b = q % bits;
        st = (st << 1) | ((tile[b >> 5] >> (31 - (b & 31))) & 1u);
    }
    return st;
}
int byte_sum(uint32_t st) {
    const uint32_t x = st * 0x83DCD12Du;
    return (int) ((x & 0xffu) + ((x >> 8) & 0xffu) + ((x >> 16) & 0xffu) + (x >> 24));
}
// the vector tiers' arithmetic written out: one fma(s, xh, acc) a weight in (slice, row) order
float exact_band_col(const cpu::Exl3Part& m, const float* xh, int col, int ks0, int ks1) {
    const int t = col / 16, c = col % 16;
    float a = 0.0f;
    for (int ks = ks0; ks < ks1; ++ks) {
        const uint32_t* tile = m.trellis + ((size_t) ks * (m.n / 16) + t) * 8 * m.K;
        for (int r = 0; r < 16; ++r) a = std::fma((float) byte_sum(model_state(tile, m.K, r, c)), xh[ks * 16 + r], a);
    }
    return a;
}

// ---------------------------------------------------------------- 1. the layout against the oracle
// exl3_decode_tile with the identity table returns each position's raw 16-bit window, in row-major tile order
void test_layout(int K) {
    std::vector<uint16_t> ident(1u << 16);
    for (uint32_t w = 0; w < (1u << 16); ++w) ident[w] = (uint16_t) w;
    constexpr int kTiles = 64;
    size_t diff = 0;
    for (int it = 0; it < kTiles; ++it) {
        const std::vector<uint32_t> tile = rand_words((size_t) 8 * K);
        uint16_t got[256];
        cpu::exl3_decode_tile((const uint16_t*) tile.data(), K, ident.data(), got);
        for (int r = 0; r < 16; ++r)
            for (int c = 0; c < 16; ++c) diff += got[r * 16 + c] != (uint16_t) model_state(tile.data(), K, r, c);
    }
    report("layout K" + std::to_string(K) + ": oracle windows vs the lane's model", diff == 0,
           std::to_string(diff) + " of " + std::to_string(kTiles * 256) + " states differ");
}

// ---------------------------------------------------------------- 2. every tier against the oracle, tile by tile
// the weight the oracle decodes from a byte sum s (decode_window, mul1: fp16(fma(fp16(1024 + s), k_inv, k_bias)))
uint16_t oracle_weight_from_sum(int s) {
    const float h = f32_from_f16((uint16_t) (0x6400u + (uint32_t) s));   // = 1024 + s exactly (s <= 1020)
    return f16_from_f32(std::fma(h, f32_from_f16(0x1EEEu), f32_from_f16(0xC931u)));
}

void test_tiles_vs_oracle(int K) {
    constexpr int k = 256, n = 256;   // 16 slices, 16 tile columns (2 bands)
    const HostMat h = make_mat(k, n, K);
    const cpu::Exl3Part m = h.part();
    std::vector<uint16_t> lut(1u << 16), ident(1u << 16);
    cpu::exl3_codebook_lut(cpu::Exl3Codebook::Mul1, lut.data());
    for (uint32_t w = 0; w < (1u << 16); ++w) ident[w] = (uint16_t) w;
    // the oracle: every tile's decoded fp16 weights and raw states, row-major [k][n]
    std::vector<uint16_t> ow((size_t) k * n), os((size_t) k * n);
    for (int i = 0; i < k / 16; ++i)
        for (int j = 0; j < n / 16; ++j) {
            const uint16_t* tw = (const uint16_t*) (h.tr.data() + ((size_t) i * (n / 16) + j) * 8 * K);
            uint16_t tw_out[256], ts_out[256];
            cpu::exl3_decode_tile(tw, K, lut.data(), tw_out);
            cpu::exl3_decode_tile(tw, K, ident.data(), ts_out);
            for (int r = 0; r < 16; ++r)
                for (int c = 0; c < 16; ++c) {
                    ow[(size_t) (i * 16 + r) * n + j * 16 + c] = tw_out[r * 16 + c];
                    os[(size_t) (i * 16 + r) * n + j * 16 + c] = ts_out[r * 16 + c];
                }
        }
    // one-hot activations e_row, read back through the kernels: acc[col] = s(row, col) exactly
    std::vector<std::vector<float>> onehot((size_t) k, std::vector<float>((size_t) k, 0.0f));
    for (int r = 0; r < k; ++r) onehot[(size_t) r][(size_t) r] = 1.0f;
    for (int tier = 0; tier <= 2; ++tier) {
        if (((cpu::exl3_cpu_tier_mask() >> tier) & 1) == 0) continue;
        size_t wdiff = 0, sdiff = 0, cmp = 0;
        auto check = [&](int row, int col, float got) {
            ++cmp;
            const int s = (int) got;
            if ((float) s != got || s < 0 || s > 1020) {
                ++wdiff;
                ++sdiff;
                return;
            }
            wdiff += oracle_weight_from_sum(s) != ow[(size_t) row * n + col];
            sdiff += s != byte_sum(os[(size_t) row * n + col]);
        };
        // the band kernels: 1, 2, 3 and 8 rows of a slice at once (the single, both-halves and one-half paths)
        for (int nt : {1, 2, 3, 8})
            for (int c0 = 0; c0 < n; c0 += 128)
                for (int ks = 0; ks < k / 16; ++ks)
                    for (int r0 = 0; r0 < 16; r0 += nt) {
                        const int cnt = std::min(nt, 16 - r0);
                        const float* xp[8];
                        std::vector<float> acc((size_t) cnt * 128);
                        float* ap[8];
                        for (int i = 0; i < cnt; ++i) {
                            xp[i] = onehot[(size_t) (ks * 16 + r0 + i)].data();
                            ap[i] = acc.data() + (size_t) i * 128;
                        }
                        if (!cpu::exl3_cpu_band_acc_on(tier, m, xp, cnt, c0, ks, ks + 1, ap)) continue;
                        for (int i = 0; i < cnt; ++i)
                            for (int j = 0; j < 128; ++j) check(ks * 16 + r0 + i, c0 + j, ap[i][j]);
                    }
        // the split-K pieces: 1 and 3 rows at once, every column
        for (int nt : {1, 3})
            for (int ks = 0; ks < k / 16; ++ks)
                for (int r0 = 0; r0 < 16; r0 += nt) {
                    const int cnt = std::min(nt, 16 - r0);
                    const float* xp[3];
                    std::vector<float> acc((size_t) cnt * n);
                    float* ap[3];
                    for (int i = 0; i < cnt; ++i) {
                        xp[i] = onehot[(size_t) (ks * 16 + r0 + i)].data();
                        ap[i] = acc.data() + (size_t) i * n;
                    }
                    if (!cpu::exl3_cpu_rows_acc_on(tier, m, xp, cnt, ks, ks + 1, ap)) continue;
                    for (int i = 0; i < cnt; ++i)
                        for (int j = 0; j < n; ++j) check(ks * 16 + r0 + i, j, ap[i][j]);
                }
        report(std::string("tiles K") + std::to_string(K) + " " + tier_name(tier) + ": weights vs exl3_decode_tile",
               wdiff == 0 && sdiff == 0 && cmp > 0,
               std::to_string(wdiff) + " weights, " + std::to_string(sdiff) + " sums of " + std::to_string(cmp) +
                   " differ");
    }
}

// ---------------------------------------------------------------- 3. exactness (project-maya's CPU lane tests)
// every tier this CPU runs against the exact model: the vector tiers bit for bit, the scalar tier within rounding;
// every band, the whole slice range and partial ones, 1..11 activations at once (past kExl3CpuMaxN)
void test_cpu_exact(int k, int n, int K, int max_nt = 11) {
    const HostMat h = make_mat(k, n, K);
    const cpu::Exl3Part m = h.part();
    const int NT = max_nt;
    std::vector<std::vector<float>> xh;
    std::vector<const float*> xp((size_t) NT);
    for (int i = 0; i < NT; ++i) xh.push_back(rand_x(k));
    for (int i = 0; i < NT; ++i) xp[(size_t) i] = xh[(size_t) i].data();
    std::vector<float> want((size_t) NT * 128), got((size_t) NT * 128);
    std::vector<float*> gp((size_t) NT);
    for (int i = 0; i < NT; ++i) gp[(size_t) i] = got.data() + (size_t) i * 128;
    const int S = k / 16;
    const int ranges[3][2] = {{0, S}, {3, S - 1}, {S / 2, S / 2 + 1}};
    size_t diff[3] = {0, 0, 0}, cmp[3] = {0, 0, 0};
    double se = 0.0, sr = 0.0;
    for (int c0 = 0; c0 < n; c0 += 128)
        for (const auto& rg : ranges) {
            for (int i = 0; i < NT; ++i)
                for (int j = 0; j < 128; ++j)
                    want[(size_t) i * 128 + j] = exact_band_col(m, xp[(size_t) i], c0 + j, rg[0], rg[1]);
            for (int tier = 0; tier <= 2; ++tier)
                for (int nt : {1, 2, 3, 5, 8, 11}) {
                    if (nt > NT) break;
                    if (!cpu::exl3_cpu_band_acc_on(tier, m, xp.data(), nt, c0, rg[0], rg[1], gp.data())) break;
                    for (size_t i = 0; i < (size_t) nt * 128; ++i) {
                        ++cmp[tier];
                        diff[tier] += std::memcmp(&got[i], &want[i], sizeof(float)) != 0;
                        if (tier == 0) {
                            se += ((double) got[i] - want[i]) * ((double) got[i] - want[i]);
                            sr += (double) want[i] * want[i];
                        }
                    }
                }
        }
    const std::string shape = " k" + std::to_string(k) + " n" + std::to_string(n) + " K" + std::to_string(K);
    for (int tier = 1; tier <= 2; ++tier) {
        if (cmp[tier] == 0) continue;
        report(std::string("exact ") + tier_name(tier) + shape, diff[tier] == 0,
               std::to_string(diff[tier]) + " of " + std::to_string(cmp[tier]) + " differ from the model");
    }
    const double rel = std::sqrt(se / std::max(sr, 1e-30));
    char buf[64];
    std::snprintf(buf, sizeof buf, "rel %.2e to the model", rel);
    report("exact scalar" + shape, cmp[0] > 0 && rel <= 1e-5, buf);
}

// several activations at once against one call each: the same bits
void test_cpu_n(int k, int n, int K) {
    const HostMat h = make_mat(k, n, K);
    const cpu::Exl3Part m = h.part();
    constexpr int NT = 11;   // past kExl3CpuMaxN: two runs
    std::vector<std::vector<float>> xh;
    const float* xp[NT];
    for (int i = 0; i < NT; ++i) xh.push_back(rand_x(k));
    for (int i = 0; i < NT; ++i) xp[i] = xh[(size_t) i].data();
    std::vector<float> one((size_t) NT * 128), all((size_t) NT * 128);
    float* ap[NT];
    for (int i = 0; i < NT; ++i) ap[i] = all.data() + (size_t) i * 128;
    size_t diff = 0;
    for (int c0 = 0; c0 < n; c0 += 128)
        for (int nt = 2; nt <= NT; ++nt) {
            const int ks0 = c0 == 0 ? 0 : 3, ks1 = k / 16 - (nt % 2);
            for (int i = 0; i < nt; ++i) cpu::exl3_cpu_band_acc(m, xp[i], c0, ks0, ks1, one.data() + (size_t) i * 128);
            cpu::exl3_cpu_band_acc_n(m, xp, nt, c0, ks0, ks1, ap);
            for (size_t i = 0; i < (size_t) nt * 128; ++i) diff += std::memcmp(&one[i], &all[i], sizeof(float)) != 0;
        }
    report("x2..11 at once k" + std::to_string(k) + " n" + std::to_string(n) + " K" + std::to_string(K), diff == 0,
           std::to_string(diff) + " differ from one call each");
}

// the split-K pieces (every column, slices [ks0, ks1)) against the bands over the same slices, and several
// activations at once against one call each: the same bits
void test_cpu_rows(int k, int n, int K) {
    const HostMat h = make_mat(k, n, K);
    const cpu::Exl3Part m = h.part();
    constexpr int NT = 11;
    std::vector<std::vector<float>> xh;
    const float* xp[NT];
    for (int i = 0; i < NT; ++i) xh.push_back(rand_x(k));
    for (int i = 0; i < NT; ++i) xp[i] = xh[(size_t) i].data();
    std::vector<float> band(128), one((size_t) NT * n), all((size_t) NT * n);
    float* ap[NT];
    for (int i = 0; i < NT; ++i) ap[i] = all.data() + (size_t) i * n;
    size_t diff = 0;
    for (int nt = 1; nt <= NT; ++nt) {
        const int ks0 = nt % 3, ks1 = k / 16 - (nt % 2);
        for (int i = 0; i < nt; ++i) {
            cpu::exl3_cpu_rows_acc(m, xp[i], ks0, ks1, one.data() + (size_t) i * n);
            for (int c0 = 0; c0 < n; c0 += 128) {
                cpu::exl3_cpu_band_acc(m, xp[i], c0, ks0, ks1, band.data());
                for (int j = 0; j < 128; ++j)
                    diff += std::memcmp(&band[(size_t) j], &one[(size_t) i * n + c0 + j], sizeof(float)) != 0;
            }
        }
        if (nt > 1) {
            cpu::exl3_cpu_rows_acc_n(m, xp, nt, ks0, ks1, ap);
            for (size_t i = 0; i < (size_t) nt * n; ++i) diff += std::memcmp(&one[i], &all[i], sizeof(float)) != 0;
        }
    }
    report("split-K x1..11 k" + std::to_string(k) + " n" + std::to_string(n) + " K" + std::to_string(K), diff == 0,
           std::to_string(diff) + " differ from the bands / one call each");
}

// the input block (x * suh, the 128-point transform) against the scalar transform: the same bits
void test_cpu_had() {
    size_t diff = 0;
    for (int rep = 0; rep < 64; ++rep) {
        std::vector<float> x = rand_x(128, 3.0f), want(128), got(128);
        std::vector<uint16_t> suh(128);
        for (int i = 0; i < 128; ++i) suh[(size_t) i] = (uint16_t) (0x3800 + (rep * 131 + i * 37) % 0x800);   // [0.5, 1)
        for (int i = 0; i < 128; ++i) want[(size_t) i] = x[(size_t) i] * f32_from_f16(suh[(size_t) i]);
        for (int hh = 1; hh < 128; hh <<= 1)
            for (int i = 0; i < 128; i += 2 * hh)
                for (int j = i; j < i + hh; ++j) {
                    const float a = want[(size_t) j], b = want[(size_t) (j + hh)];
                    want[(size_t) j] = a + b;
                    want[(size_t) (j + hh)] = a - b;
                }
        for (int i = 0; i < 128; ++i) want[(size_t) i] *= 0.088388347648f;
        cpu::exl3_cpu_prepare_block(suh.data(), x.data(), got.data());
        for (int i = 0; i < 128; ++i) diff += std::memcmp(&want[(size_t) i], &got[(size_t) i], sizeof(float)) != 0;
    }
    report("input blocks (128-point transform)", diff == 0, std::to_string(diff) + " of 8192 differ from the scalar");
}

// the prompt path's GEMM (a band decoded once for many activations) against the band kernel: the same sums up to
// F32 rounding (another order)
void test_cpu_gemm(int k, int n, int K) {
    const HostMat h = make_mat(k, n, K);
    const cpu::Exl3Part m = h.part();
    constexpr int NT = 21;   // 3 tiles of 6 and 3 left over
    std::vector<std::vector<float>> xh;
    const float* xp[NT];
    for (int i = 0; i < NT; ++i) xh.push_back(rand_x(k));
    for (int i = 0; i < NT; ++i) xp[i] = xh[(size_t) i].data();
    std::vector<float> one(128), all, wc(cpu::exl3_cpu_gemm_scratch() + 16);
    float* const w64 = (float*) (((uintptr_t) wc.data() + 63) & ~(uintptr_t) 63);
    float* ap[NT];
    double worst = 0.0;
    for (int nb : {1, 2}) {
        all.assign((size_t) NT * 128 * nb, 0.0f);
        for (int i = 0; i < NT; ++i) ap[i] = all.data() + (size_t) i * 128 * nb;
        for (int c0 = 0; c0 + 128 * nb <= n; c0 += 128 * nb) {
            cpu::exl3_cpu_band_gemm(m, xp, NT, c0, nb, ap, w64);
            for (int i = 0; i < NT; ++i)
                for (int b = 0; b < nb; ++b) {
                    cpu::exl3_cpu_band_acc(m, xp[i], c0 + 128 * b, 0, k / 16, one.data());
                    double mx = 0.0, d = 0.0;
                    for (int j = 0; j < 128; ++j) {
                        mx = std::max(mx, (double) std::fabs(one[(size_t) j]));
                        d = std::max(d, (double) std::fabs(one[(size_t) j] - ap[i][128 * b + j]));
                    }
                    worst = std::max(worst, d / std::max(mx, 1e-30));
                }
        }
    }
    char buf[64];
    std::snprintf(buf, sizeof buf, "worst %.2e of the band's largest", worst);
    report("prompt GEMM x21 nb1,2 k" + std::to_string(k) + " n" + std::to_string(n) + " K" + std::to_string(K),
           worst < 1e-5, buf);
}

// ---------------------------------------------------------------- 4. the whole product against the oracle
// y = H(x * suh) W_hat H * svh: the lane (bands, and split-K pieces + band_out) against exl3_folded_gemv, whose weights
// are rounded to fp16 (the lane keeps them exact in F32): relative RMS error a few 1e-4, as project-maya measured
void test_product(int k, int n, int K, int tokens) {
    const HostMat h = make_mat(k, n, K);
    const cpu::Exl3Part m = h.part();
    std::vector<uint16_t> x16((size_t) tokens * k);
    for (auto& v : x16) v = f16_from_f32(std::normal_distribution<float>(0.0f, 1.0f)(g_rng));
    std::vector<float> ref((size_t) tokens * n);
    cpu::exl3_folded_gemv((const uint16_t*) h.tr.data(), k / 16, n / 16, K, cpu::Exl3Codebook::Mul1, h.suh.data(),
                          h.svh.data(), x16.data(), tokens, ref.data());
    std::vector<float> x((size_t) k), xh((size_t) k), band((size_t) n), rows((size_t) n), acc((size_t) n);
    double se = 0.0, sr = 0.0;
    size_t rdiff = 0;
    bool finite = true;
    for (int t = 0; t < tokens; ++t) {
        for (int i = 0; i < k; ++i) x[(size_t) i] = f32_from_f16(x16[(size_t) t * k + i]);
        const float xs = cpu::exl3_cpu_prepare(m, x.data(), xh.data());
        for (int c0 = 0; c0 < n; c0 += 128) cpu::exl3_cpu_band(m, xh.data(), xs, c0, band.data() + c0);
        cpu::exl3_cpu_rows_acc(m, xh.data(), 0, k / 16, acc.data());
        for (int c0 = 0; c0 < n; c0 += 128) cpu::exl3_cpu_band_out(m, acc.data() + c0, xs, c0, rows.data() + c0);
        for (int j = 0; j < n; ++j) {
            const double r = ref[(size_t) t * n + j], d = (double) band[(size_t) j] - r;
            if (!std::isfinite(band[(size_t) j])) finite = false;
            se += d * d;
            sr += r * r;
            rdiff += std::memcmp(&band[(size_t) j], &rows[(size_t) j], sizeof(float)) != 0;
        }
    }
    const double rel = std::sqrt(se / std::max(sr, 1e-30));
    char buf[96];
    std::snprintf(buf, sizeof buf, "rel %.2e to the oracle, split-K %zu differ", rel, rdiff);
    report("product " + std::string(tier_name(cpu::exl3_cpu_tier())) + " k" + std::to_string(k) + " n" +
               std::to_string(n) + " K" + std::to_string(K) + " x" + std::to_string(tokens),
           finite && rel <= 2e-3 && rdiff == 0, buf);
}

// ---------------------------------------------------------------- 5. real expert matrices (--real <model dir>)
// Each matrix copied out of the shards into aligned buffers (the file keeps its trellises at offsets 2 mod 4: the
// engine's RAM tier is such a copy too), then: every tile's windows against the oracle, the band of columns 0..127
// weight by weight against exl3_decode_tile on every tier, every tier's bands against the exact model (vector tiers
// bit for bit) for 1 and 3 tokens, the split-K pieces against the bands, and the product against exl3_folded_gemv.
bool load_real(const strata::Exl3Model& model, const std::string& base, HostMat& h, std::string& err) {
    if (!model.has(base)) {
        err = "no tensor " + base + ".trellis";
        return false;
    }
    const strata::Exl3Linear L = model.linear(base);
    if (L.cb != 2) {
        err = base + ": codebook " + std::to_string(L.cb) + ", the lane runs mul1 (2) only";
        return false;
    }
    if (L.half() || L.ki <= 0 || L.nj <= 0 || L.in() % 128 != 0 || L.out() % 128 != 0) {
        err = base + ": unexpected shape ki " + std::to_string(L.ki) + " nj " + std::to_string(L.nj) + " K " +
              strata::exl3_k_label(L.k2);
        return false;
    }
    h.k = L.in();
    h.n = L.out();
    h.K = L.k2 / 2;
    h.tr.assign((size_t) L.ki * L.nj * L.tile_words() / 2, 0u);
    std::memcpy(h.tr.data(), L.trellis, h.tr.size() * sizeof(uint32_t));
    h.suh.assign(L.suh, L.suh + h.k);
    h.svh.assign(L.svh, L.svh + h.n);
    return true;
}

void test_real_matrix(const std::string& name, const HostMat& h, size_t& n_ok, size_t& n_bad) {
    const int fails0 = g_fail;
    const cpu::Exl3Part m = h.part();
    const int k = h.k, n = h.n, K = h.K, S = k / 16;
    // windows of every tile against the oracle's
    std::vector<uint16_t> ident(1u << 16), lut(1u << 16);
    for (uint32_t w = 0; w < (1u << 16); ++w) ident[w] = (uint16_t) w;
    cpu::exl3_codebook_lut(cpu::Exl3Codebook::Mul1, lut.data());
    size_t sdiff = 0;
    for (int i = 0; i < S; ++i)
        for (int j = 0; j < n / 16; ++j) {
            const uint32_t* tile = h.tr.data() + ((size_t) i * (n / 16) + j) * 8 * K;
            uint16_t st[256];
            cpu::exl3_decode_tile((const uint16_t*) tile, K, ident.data(), st);
            for (int r = 0; r < 16; ++r)
                for (int c = 0; c < 16; ++c) sdiff += st[r * 16 + c] != (uint16_t) model_state(tile, K, r, c);
        }
    // band 0 weight by weight on every tier: one-hot rows, 3 at a time
    std::vector<uint16_t> ow((size_t) k * 128);
    for (int i = 0; i < S; ++i)
        for (int j = 0; j < 8; ++j) {
            uint16_t tw[256];
            cpu::exl3_decode_tile((const uint16_t*) (h.tr.data() + ((size_t) i * (n / 16) + j) * 8 * K), K, lut.data(), tw);
            for (int r = 0; r < 16; ++r)
                for (int c = 0; c < 16; ++c) ow[(size_t) (i * 16 + r) * 128 + j * 16 + c] = tw[r * 16 + c];
        }
    size_t wdiff = 0, wcmp = 0;
    {
        std::vector<float> e0((size_t) k, 0.0f), e1((size_t) k, 0.0f), e2((size_t) k, 0.0f), acc(3 * 128);
        for (int tier = 0; tier <= 2; ++tier) {
            if (((cpu::exl3_cpu_tier_mask() >> tier) & 1) == 0) continue;
            for (int row = 0; row < k; row += 3) {
                const int cnt = std::min(3, k - row);
                std::vector<float>* ev[3] = {&e0, &e1, &e2};
                const float* xp[3];
                float* ap[3];
                for (int i = 0; i < cnt; ++i) {
                    (*ev[i])[(size_t) (row + i)] = 1.0f;
                    xp[i] = ev[i]->data();
                    ap[i] = acc.data() + 128 * i;
                }
                const int ks = row / 16;   // rows row..row+2 may straddle two slices: take both
                const int ks1 = std::min(S, (row + cnt - 1) / 16 + 1);
                cpu::exl3_cpu_band_acc_on(tier, m, xp, cnt, 0, ks, ks1, ap);
                for (int i = 0; i < cnt; ++i) {
                    for (int c = 0; c < 128; ++c) {
                        ++wcmp;
                        const float got = ap[i][c];
                        const int s = (int) got;
                        wdiff += (float) s != got || s < 0 || s > 1020 ||
                                 oracle_weight_from_sum(s) != ow[(size_t) (row + i) * 128 + c];
                    }
                    (*ev[i])[(size_t) (row + i)] = 0.0f;
                }
            }
        }
    }
    // every tier against the exact model, 1 and 3 tokens, whole range; split-K pieces against the bands
    std::vector<std::vector<float>> xh;
    const float* xp[3];
    for (int i = 0; i < 3; ++i) xh.push_back(rand_x(k));
    for (int i = 0; i < 3; ++i) xp[i] = xh[(size_t) i].data();
    std::vector<float> want(3 * 128), got(3 * 128), rows((size_t) 3 * n), band(128);
    float* gp[3] = {got.data(), got.data() + 128, got.data() + 256};
    size_t ediff[3] = {0, 0, 0}, ecmp[3] = {0, 0, 0}, rdiff = 0;
    double se = 0.0, sr = 0.0;
    for (int c0 = 0; c0 < n; c0 += 128) {
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 128; ++j) want[(size_t) i * 128 + j] = exact_band_col(m, xp[i], c0 + j, 0, S);
        for (int tier = 0; tier <= 2; ++tier)
            for (int nt : {1, 3}) {
                if (!cpu::exl3_cpu_band_acc_on(tier, m, xp, nt, c0, 0, S, gp)) break;
                for (size_t i = 0; i < (size_t) nt * 128; ++i) {
                    ++ecmp[tier];
                    ediff[tier] += std::memcmp(&got[i], &want[i], sizeof(float)) != 0;
                    if (tier == 0) {
                        se += ((double) got[i] - want[i]) * ((double) got[i] - want[i]);
                        sr += (double) want[i] * want[i];
                    }
                }
            }
    }
    {
        float* rp[3] = {rows.data(), rows.data() + n, rows.data() + 2 * n};
        for (const auto& rg : {std::pair<int, int>(0, S), std::pair<int, int>(S / 4, S / 2)}) {
            cpu::exl3_cpu_rows_acc_n(m, xp, 3, rg.first, rg.second, rp);
            for (int i = 0; i < 3; ++i)
                for (int c0 = 0; c0 < n; c0 += 128) {
                    cpu::exl3_cpu_band_acc(m, xp[i], c0, rg.first, rg.second, band.data());
                    for (int j = 0; j < 128; ++j)
                        rdiff += std::memcmp(&band[(size_t) j], &rp[i][c0 + j], sizeof(float)) != 0;
                }
        }
    }
    // the product against the oracle
    std::vector<uint16_t> x16((size_t) 2 * k);
    for (auto& v : x16) v = f16_from_f32(std::normal_distribution<float>(0.0f, 1.0f)(g_rng));
    std::vector<float> ref((size_t) 2 * n), y((size_t) n), xf((size_t) k), xhh((size_t) k);
    cpu::exl3_folded_gemv((const uint16_t*) h.tr.data(), k / 16, n / 16, K, cpu::Exl3Codebook::Mul1, h.suh.data(),
                          h.svh.data(), x16.data(), 2, ref.data());
    double pse = 0.0, psr = 0.0;
    for (int t = 0; t < 2; ++t) {
        for (int i = 0; i < k; ++i) xf[(size_t) i] = f32_from_f16(x16[(size_t) t * k + i]);
        const float xs = cpu::exl3_cpu_prepare(m, xf.data(), xhh.data());
        for (int c0 = 0; c0 < n; c0 += 128) cpu::exl3_cpu_band(m, xhh.data(), xs, c0, y.data() + c0);
        for (int j = 0; j < n; ++j) {
            const double d = (double) y[(size_t) j] - ref[(size_t) t * n + j];
            pse += d * d;
            psr += (double) ref[(size_t) t * n + j] * ref[(size_t) t * n + j];
        }
    }
    const double erel = std::sqrt(se / std::max(sr, 1e-30)), prel = std::sqrt(pse / std::max(psr, 1e-30));
    const bool vec_ok = ediff[1] == 0 && ediff[2] == 0 && (ecmp[1] + ecmp[2]) > 0;
    const bool ok = sdiff == 0 && wdiff == 0 && wcmp > 0 && vec_ok && erel <= 1e-5 && rdiff == 0 && prel <= 2e-3 &&
                    std::isfinite(prel);
    char buf[200];
    std::snprintf(buf, sizeof buf, "K%d %dx%d: states %zu, weights %zu/%zu, exact %zu/%zu, scalar %.1e, split-K %zu, "
                  "product %.2e", K, k, n, sdiff, wdiff, wcmp, ediff[1] + ediff[2], ecmp[1] + ecmp[2], erel, rdiff, prel);
    report(name, ok, buf);
    if (g_fail == fails0) ++n_ok;
    else ++n_bad;
}

int test_real(const std::string& dir) {
    std::printf("5. real expert matrices from %s\n", dir.c_str());
    try {
        const strata::Exl3Model model(dir);
        size_t n_ok = 0, n_bad = 0;
        std::vector<std::string> bases;
        for (int layer : {0, 23, 47})
            for (int e : {0, 300, 511})
                for (const char* role : {"gate_proj", "up_proj", "down_proj"})
                    bases.push_back("model.language_model.layers." + std::to_string(layer) + ".mlp.experts." +
                                    std::to_string(e) + "." + role);
        for (int e : {0, 511})
            for (const char* role : {"gate_proj", "up_proj", "down_proj"})
                bases.push_back("mtp.layers.0.mlp.experts." + std::to_string(e) + "." + role);
        for (const std::string& b : bases) {
            HostMat h;
            std::string err;
            if (!load_real(model, b, h, err)) {
                report(b, false, err);
                ++n_bad;
                continue;
            }
            test_real_matrix(b.rfind("model.language_model.", 0) == 0 ? b.substr(21) : b, h, n_ok, n_bad);
        }
        std::printf("  real matrices: %zu pass, %zu fail\n", n_ok, n_bad);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "exl3_lane_test: %s\n", e.what());
        ++g_fail;
    }
    return g_fail;
}

}  // namespace

int main(int argc, char** argv) {
    std::string real_dir;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--real" && i + 1 < argc) real_dir = argv[++i];
        else {
            std::fprintf(stderr, "usage: exl3_lane_test [--real <EXL3 model dir>]\n");
            return 2;
        }
    }
    std::printf("exl3_lane_test: tier %s (mask %d)\n", tier_name(cpu::exl3_cpu_tier()), cpu::exl3_cpu_tier_mask());
    if (!real_dir.empty()) {
        test_real(real_dir);
        std::printf("exl3_lane_test: %s (%d failures)\n", g_fail ? "FAIL" : "ok", g_fail);
        return g_fail ? 1 : 0;
    }
    std::printf("1. layout: #1764's windows and tile order against the lane's bit model\n");
    for (int K = 1; K <= 8; ++K) test_layout(K);
    std::printf("2. every tier against exl3_decode_tile, weight by weight\n");
    for (int K = 1; K <= 8; ++K) test_tiles_vs_oracle(K);
    std::printf("3. exactness\n");
    for (int K = 1; K <= 8; ++K) test_cpu_exact(512, 256, K);
    test_cpu_exact(592, 384, 3);   // 37 slices
    test_cpu_exact(592, 256, 4);
    test_cpu_exact(2560, 640, 4, 3);   // the Flash's gate/up
    test_cpu_exact(640, 256, 5, 3);    // the Flash's down (two of its 20 bands)
    for (int K : {1, 3, 4, 5, 6, 8}) test_cpu_n(512, 256, K);
    for (int K : {2, 3, 4, 5, 8}) test_cpu_rows(512, 384, K);
    test_cpu_rows(2560, 640, 4);
    test_cpu_rows(640, 2560, 5);
    for (int K : {3, 4, 6}) test_cpu_gemm(768, 256, K);
    test_cpu_had();
    std::printf("4. the whole product against exl3_folded_gemv (Flash shapes)\n");
    for (int K = 2; K <= 6; ++K) {
        test_product(2560, 640, K, 2);
        test_product(640, 2560, K, 2);
    }
    std::printf("exl3_lane_test: %s (%d failures)\n", g_fail ? "FAIL" : "ok", g_fail);
    return g_fail ? 1 : 0;
}
