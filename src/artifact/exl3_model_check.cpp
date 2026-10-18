// src/artifact/exl3_model_check.cpp - reconstruct EXL3 experts from the multi-shard model and print a
// deterministic checksum to compare against tools/exl3/check_model.py.  CPU-only.
//
// Usage:
//   exl3_model_check <model_dir> <base_name>
//   exl3_model_check <model_dir> --layer <L> --role <down_proj|gate_proj|up_proj> --count <N>
//   exl3_model_check <model_dir> --expect <dir>   each tensor of <dir>/manifest.txt against exllamav3's decode of it
//       (tools/exl3/emit_exllamav3_fixture.py --model): W_hat bit for bit, W (the original basis) within rounding
#include "strata/artifact/exl3_model.hpp"
#include "strata/kernels/cpu/exl3.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace strata;
using namespace strata::kernels::cpu;

static uint64_t fnv1a(uint64_t h, const void* p, size_t n) {
    const uint8_t* b = (const uint8_t*)p;
    for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 1099511628211ull; }
    return h;
}

static uint64_t check_one(const Exl3Model& m, const std::string& base) {
    Exl3Linear L = m.linear(base);
    std::vector<uint16_t> w((size_t)L.in() * L.out());
    exl3_reconstruct_weight_k2(L.trellis, L.ki, L.nj, L.k2, (Exl3Codebook)L.cb, L.suh, L.svh, w.data());
    std::printf("  %s: trellis[%d,%d] K=%s cb=%d -> %dx%d\n", base.c_str(), L.ki, L.nj, exl3_k_label(L.k2).c_str(),
                L.cb, L.in(), L.out());
    return fnv1a(1469598103934665603ull, w.data(), w.size() * 2);
}

static bool read_f16(const std::string& path, size_t n, std::vector<uint16_t>& out) {
    std::ifstream in(path, std::ios::binary);
    out.assign(n, 0);
    in.read(reinterpret_cast<char*>(out.data()), (std::streamsize) (n * 2));
    return (bool) in && in.peek() == std::char_traits<char>::eof();
}

static float f16f(uint16_t h) {
    const uint32_t sign = (uint32_t) (h >> 15) << 31, exp = (h >> 10) & 0x1F;
    uint32_t man = h & 0x3FF, f;
    if (exp == 0) {
        if (man == 0) f = sign;
        else { int e = 127 - 15 + 1; while (!(man & 0x400)) { man <<= 1; --e; } f = sign | ((uint32_t) e << 23) | ((man & 0x3FF) << 13); }
    } else if (exp == 0x1F) f = sign | 0x7F800000u | (man << 13);
    else f = sign | ((exp + 127 - 15) << 23) | (man << 13);
    float o;
    std::memcpy(&o, &f, 4);
    return o;
}

// every tensor of the manifest: our W_hat against exllamav3's reconstruct (bit for bit), our W against its
// reconstruct_had_slice (relative RMS: the two round differently - exllamav3 in its fused kernel, ours in F32)
static int expect(const Exl3Model& m, const std::string& dir) {
    std::ifstream man(dir + "/manifest.txt");
    if (!man) { std::fprintf(stderr, "error: no %s/manifest.txt\n", dir.c_str()); return 1; }
    int bad = 0, n_ok = 0;
    std::string line;
    while (std::getline(man, line)) {
        std::istringstream ss(line);
        std::string base, klabel;
        int k = 0, n = 0;
        if (!(ss >> base >> klabel >> k >> n)) continue;
        const Exl3Linear L = m.linear(base);
        if (L.in() != k || L.out() != n) {
            std::printf("  %s: %dx%d here, %dx%d in the manifest  FAIL\n", base.c_str(), L.in(), L.out(), k, n);
            ++bad;
            continue;
        }
        std::vector<uint16_t> want_hat, want_w;
        if (!read_f16(dir + "/" + base + ".what.f16", (size_t) k * n, want_hat) ||
            !read_f16(dir + "/" + base + ".w.f16", (size_t) k * n, want_w)) {
            std::printf("  %s: the expected files are missing or of another size  FAIL\n", base.c_str());
            ++bad;
            continue;
        }
        std::vector<uint16_t> hat((size_t) k * n), w((size_t) k * n);
        exl3_decode_weight_hat_k2(L.trellis, L.ki, L.nj, L.k2, (Exl3Codebook) L.cb, hat.data());
        exl3_reconstruct_weight_k2(L.trellis, L.ki, L.nj, L.k2, (Exl3Codebook) L.cb, L.suh, L.svh, w.data());
        size_t hdiff = 0, wdiff = 0;
        double se = 0.0, sr = 0.0;
        for (size_t i = 0; i < hat.size(); ++i) {
            hdiff += hat[i] != want_hat[i];
            wdiff += w[i] != want_w[i];
            const double a = f16f(w[i]), b = f16f(want_w[i]);
            se += (a - b) * (a - b);
            sr += b * b;
        }
        const double rel = std::sqrt(se / (sr > 0 ? sr : 1e-30));
        const bool ok = hdiff == 0 && rel < 1e-3 && std::isfinite(rel);
        std::printf("  %-64s K %-3s W_hat %zu/%zu differ, W %zu differ (rel %.2e)  %s\n", base.c_str(),
                    exl3_k_label(L.k2).c_str(), hdiff, hat.size(), wdiff, rel, ok ? "ok" : "FAIL");
        if (ok) ++n_ok;
        else ++bad;
    }
    std::printf("against exllamav3: %d ok, %d failed\n", n_ok, bad);
    return bad == 0 && n_ok > 0 ? 0 : 1;
}

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: %s <dir> <base> | <dir> --layer L --role R --count N\n", argv[0]); return 2; }
    try {
        Exl3Model m(argv[1]);
        std::printf("model: %zu tensors\n", m.tensor_count());
        uint64_t h = 1469598103934665603ull;
        if (std::string(argv[2]) == "--expect" && argc > 3) return expect(m, argv[3]);
        if (std::string(argv[2]) == "--layer") {
            int layer = std::stoi(argv[3]);
            std::string role = argv[5];
            int count = std::stoi(argv[7]);
            for (int e = 0; e < count; ++e) {
                std::string base = "model.language_model.layers." + std::to_string(layer) +
                                   ".mlp.experts." + std::to_string(e) + "." + role;
                Exl3Linear L = m.linear(base);
                std::vector<uint16_t> w((size_t)L.in() * L.out());
                exl3_reconstruct_weight_k2(L.trellis, L.ki, L.nj, L.k2, (Exl3Codebook)L.cb, L.suh, L.svh, w.data());
                h = fnv1a(h, w.data(), w.size() * 2);
            }
            std::printf("layer %d %s x%d combined fnv1a=%016llx\n", layer, role.c_str(), count,
                        (unsigned long long)h);
        } else {
            h = check_one(m, argv[2]);
            std::printf("fnv1a=%016llx\n", (unsigned long long)h);
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
