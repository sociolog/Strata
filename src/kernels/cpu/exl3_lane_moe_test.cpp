// src/kernels/cpu/exl3_lane_moe_test.cpp - the EXL3 CPU lane's MoE on the expert pool (exl3_lane_moe.cpp) and the
// resident RAM tier (exl3_ram.cpp).  No GPU.
//
//   exl3_lane_moe_test                     synthetic experts on the Flash's shapes
//   exl3_lane_moe_test --real <model dir>  plus the RAM tier of layers 0-1 against the file, and the MoE on it
//
// 1. Fixed cuts: the pool's rows are the serial run's bit for bit with 1, 3 and 9 workers (the split never depends on
//    the thread count), for 1..3 tokens an expert and batches past kMaxBatch - in four phases and fused into one.
// 2. The rows against an unsplit reference (each matrix in one split-K pass, exl3_cpu_band_out): equal up to the order
//    of the piece sums - measured 1.1e-5 / 1.4e-5 of the row's largest (K4 / K5, random weights through gate/up,
//    SwiGLU and down); the limit is 1e-4, where a wrong cut or band would be of order 1.
// 3. --real: every byte of 24 experts' nine tensors in the tier equals the file's (via Exl3Model), and their rows
//    through the tier equal the rows through a copy made from the file, bit for bit.
#include "strata/artifact/exl3_model.hpp"
#include "strata/kernels/cpu/exl3_lane_moe.hpp"
#include "strata/kernels/cpu/exl3_ram.hpp"
#include "strata/kernels/f16_bits.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace cpu = strata::kernels::cpu;

namespace {

int g_fail = 0;
std::mt19937_64 g_rng(2026);
constexpr int kH = 2560, kFF = 640;

void report(const std::string& what, bool ok, const std::string& detail) {
    std::printf("  %-50s %s  %s\n", what.c_str(), detail.c_str(), ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}

struct HostMat {
    std::vector<uint32_t> tr;
    std::vector<uint16_t> suh, svh;
    int k = 0, n = 0, K = 0;
    cpu::Exl3Part part() const { return cpu::Exl3Part{tr.data(), suh.data(), svh.data(), k, n, K}; }
};
HostMat rand_mat(int k, int n, int K) {
    HostMat m;
    m.k = k;
    m.n = n;
    m.K = K;
    m.tr.resize((size_t) (k / 16) * (n / 16) * 8 * K);
    for (auto& w : m.tr) w = (uint32_t) g_rng();
    std::uniform_real_distribution<float> u(0.25f, 2.0f);
    m.suh.resize((size_t) k);
    m.svh.resize((size_t) n);
    for (auto& v : m.suh) v = strata::kernels::f16_from_f32((g_rng() & 1) ? u(g_rng) : -u(g_rng));
    for (auto& v : m.svh) v = strata::kernels::f16_from_f32((g_rng() & 1) ? u(g_rng) : -u(g_rng));
    return m;
}
struct HostExpert {
    HostMat g, u, d;
    cpu::Exl3Expert view() const { return cpu::Exl3Expert{g.part(), u.part(), d.part()}; }
};

// the unsplit reference: each matrix in one pass over all its slices
std::vector<float> reference(const cpu::Exl3Expert& e, const float* x) {
    auto mv = [](const cpu::Exl3Part& m, const float* in) {
        std::vector<float> xh((size_t) m.k), acc((size_t) m.n), y((size_t) m.n);
        const float xs = cpu::exl3_cpu_prepare(m, in, xh.data());
        cpu::exl3_cpu_rows_acc(m, xh.data(), 0, m.k / 16, acc.data());
        for (int c0 = 0; c0 < m.n; c0 += 128) cpu::exl3_cpu_band_out(m, acc.data() + c0, xs, c0, y.data() + c0);
        return y;
    };
    const std::vector<float> g = mv(e.gate, x), u = mv(e.up, x);
    std::vector<float> h(g.size());
    for (size_t i = 0; i < h.size(); ++i) h[i] = g[i] / (1.0f + std::exp(-g[i])) * u[i];
    return mv(e.down, h.data());
}

// jobs over `experts`, expert i taking 1 + i % 3 tokens of `xs`; rows in `out` (expert-major)
void make_jobs(const std::vector<cpu::Exl3Expert>& experts, const std::vector<std::vector<float>>& xs,
               std::vector<float>& out, std::vector<cpu::Exl3ExpertJob>& jobs) {
    size_t rows = 0;
    for (size_t i = 0; i < experts.size(); ++i) rows += 1 + i % 3;
    out.assign(rows * kH, -12345.0f);
    jobs.assign(experts.size(), cpu::Exl3ExpertJob{});
    size_t r = 0;
    for (size_t i = 0; i < experts.size(); ++i) {
        jobs[i].e = experts[i];
        jobs[i].nt = 1 + (int) (i % 3);
        for (int t = 0; t < jobs[i].nt; ++t) {
            jobs[i].x[t] = xs[(i + (size_t) t) % xs.size()].data();
            jobs[i].out[t] = out.data() + (r++) * kH;
        }
    }
}

void test_moe(const std::vector<cpu::Exl3Expert>& experts, const std::string& what) {
    std::vector<std::vector<float>> xs(5, std::vector<float>(kH));
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (auto& x : xs)
        for (auto& v : x) v = nd(g_rng);
    std::string err;
    cpu::Exl3LaneMoe moe;
    if (!moe.init(kH, kFF, err)) {
        report(what, false, err);
        return;
    }
    std::vector<float> serial, pooled;
    std::vector<cpu::Exl3ExpertJob> jobs;
    make_jobs(experts, xs, serial, jobs);
    if (!moe.run_serial(jobs.data(), (int) jobs.size(), err)) {
        report(what + " serial", false, err);
        return;
    }
    // against the unsplit reference
    double worst = 0.0;
    bool finite = true;
    size_t r = 0;
    for (const auto& j : jobs)
        for (int t = 0; t < j.nt; ++t, ++r) {
            const std::vector<float> ref = reference(j.e, j.x[t]);
            double mx = 0.0, d = 0.0;
            for (int i = 0; i < kH; ++i) {
                const float got = serial[r * kH + (size_t) i];
                if (!std::isfinite(got)) finite = false;
                mx = std::max(mx, (double) std::fabs(ref[(size_t) i]));
                d = std::max(d, (double) std::fabs((double) got - ref[(size_t) i]));
            }
            worst = std::max(worst, d / std::max(mx, 1e-30));
        }
    char buf[96];
    std::snprintf(buf, sizeof buf, "%zu rows, worst %.2e of the row's largest", r, worst);
    report(what + ": split vs unsplit", finite && worst < 1e-4, buf);
    // the pool, three sizes, the four phases, the fused one and whole experts: the serial rows bit for bit
    for (const int mode : {0, 1, 2})   // 0 four phases, 1 fused, 2 whole experts where the batch allows
        for (int workers : {1, 3, 9}) {
            const bool fused = mode == 1;
            cpu::ExpertPool pool(workers, /*pin=*/false, /*host_works=*/true);
            make_jobs(experts, xs, pooled, jobs);
            moe.set_fused(fused);
            moe.set_whole(mode == 2);
            const bool ran = moe.run(pool, jobs.data(), (int) jobs.size(), err);
            moe.set_fused(false);
            moe.set_whole(false);
            if (!ran) {
                report(what + " pool", false, err);
                return;
            }
            size_t diff = 0;
            for (size_t i = 0; i < serial.size(); ++i) diff += std::memcmp(&serial[i], &pooled[i], sizeof(float)) != 0;
            report(what + ": " + (mode == 2 ? "whole " : fused ? "fused " : "") + "pool of " + std::to_string(workers) +
                       " (+ host) vs serial",
                   diff == 0, std::to_string(diff) + " of " + std::to_string(serial.size()) + " differ");
        }
    // a job that does not match the shape is refused with a message, nothing run
    cpu::Exl3ExpertJob bad = jobs[0];
    bad.e.down.n = 1280;
    const bool refused = !moe.run_serial(&bad, 1, err) && !err.empty();
    report(what + ": a mis-shaped job", refused, refused ? "refused: " + err.substr(0, 40) + "..." : "accepted");
}

void test_real(const std::string& dir) {
    std::printf("RAM tier of %s, layers 0-1\n", dir.c_str());
    cpu::Exl3ExpertRam ram;
    std::string err;
    if (!ram.open(dir, 2, 512, 8, 0, err)) {
        report("RAM tier open", false, err);
        return;
    }
    std::printf("  %s\n", ram.describe().c_str());
    try {
        const strata::Exl3Model model(dir);
        size_t diff = 0, compared = 0;
        std::vector<cpu::Exl3Expert> tier, copies;
        std::vector<std::unique_ptr<HostExpert>> keep;
        for (int layer : {0, 1})
            for (int e : {0, 1, 2, 63, 64, 127, 255, 256, 300, 400, 510, 511}) {
                const cpu::Exl3Expert x = ram.expert(layer, e);
                auto h = std::make_unique<HostExpert>();
                const std::string b =
                    "model.language_model.layers." + std::to_string(layer) + ".mlp.experts." + std::to_string(e) + ".";
                const char* roles[3] = {"gate_proj", "up_proj", "down_proj"};
                HostMat* hm[3] = {&h->g, &h->u, &h->d};
                const cpu::Exl3Part* tp[3] = {&x.gate, &x.up, &x.down};
                for (int r = 0; r < 3; ++r) {
                    const strata::Exl3Linear L = model.linear(b + roles[r]);
                    HostMat& m = *hm[r];
                    m.k = L.in();
                    m.n = L.out();
                    m.K = L.bits;
                    m.tr.resize((size_t) L.ki * L.nj * 8 * L.bits);
                    std::memcpy(m.tr.data(), L.trellis, m.tr.size() * 4);
                    m.suh.assign(L.suh, L.suh + m.k);
                    m.svh.assign(L.svh, L.svh + m.n);
                    const cpu::Exl3Part& t = *tp[r];
                    if (t.k != m.k || t.n != m.n || t.K != m.K) ++diff;
                    diff += std::memcmp(t.trellis, m.tr.data(), m.tr.size() * 4) != 0;
                    diff += std::memcmp(t.suh, m.suh.data(), m.suh.size() * 2) != 0;
                    diff += std::memcmp(t.svh, m.svh.data(), m.svh.size() * 2) != 0;
                    compared += 3;
                }
                tier.push_back(x);
                copies.push_back(h->view());
                keep.push_back(std::move(h));
            }
        report("tier vs file, 24 experts x 9 tensors", diff == 0,
               std::to_string(diff) + " of " + std::to_string(compared) + " tensors differ");
        // the rows through the tier and through the file's copies
        std::vector<std::vector<float>> xs(3, std::vector<float>(kH));
        std::normal_distribution<float> nd(0.0f, 1.0f);
        for (auto& x : xs)
            for (auto& v : x) v = nd(g_rng);
        cpu::Exl3LaneMoe moe;
        if (!moe.init(kH, kFF, err)) {
            report("lane init", false, err);
            return;
        }
        std::vector<float> a, b;
        std::vector<cpu::Exl3ExpertJob> ja, jb;
        make_jobs(tier, xs, a, ja);
        make_jobs(copies, xs, b, jb);
        cpu::ExpertPool pool(9, false, true);
        const bool ok = moe.run(pool, ja.data(), (int) ja.size(), err) && moe.run(pool, jb.data(), (int) jb.size(), err);
        size_t rd = 0;
        for (size_t i = 0; ok && i < a.size(); ++i) rd += std::memcmp(&a[i], &b[i], sizeof(float)) != 0;
        report("rows through the tier vs the file's copies", ok && rd == 0,
               ok ? std::to_string(rd) + " of " + std::to_string(a.size()) + " differ" : err);
        test_moe(tier, "real layer 0-1 experts");
    } catch (const std::exception& e) {
        report("real model", false, e.what());
    }
}

}  // namespace

int main(int argc, char** argv) {
    std::string real;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--real" && i + 1 < argc) real = argv[++i];
        else {
            std::fprintf(stderr, "usage: exl3_lane_moe_test [--real <EXL3 model dir>]\n");
            return 2;
        }
    }
    std::printf("exl3_lane_moe_test: lane tier %d\n", cpu::exl3_cpu_tier());
    for (int K : {4, 5}) {
        std::vector<std::unique_ptr<HostExpert>> keep;
        std::vector<cpu::Exl3Expert> experts;
        for (int i = 0; i < 37; ++i) {   // past kMaxBatch: two batches
            auto h = std::make_unique<HostExpert>();
            h->g = rand_mat(kH, kFF, K);
            h->u = rand_mat(kH, kFF, K);
            h->d = rand_mat(kFF, kH, K);
            experts.push_back(h->view());
            keep.push_back(std::move(h));
        }
        test_moe(experts, "K" + std::to_string(K) + " x37 experts");
    }
    if (!real.empty()) test_real(real);
    std::printf("exl3_lane_moe_test: %s (%d failures)\n", g_fail ? "FAIL" : "ok", g_fail);
    return g_fail ? 1 : 0;
}
