// src/kernels/cpu/exl3_lane_bench.cpp - MODEL-UPDATE: the EXL3 CPU lane (exl3_lane.cpp) on Qwen3.8-Flash-Next's expert
// shapes against the RAM bandwidth, measured the way kq_cpu_bench measures the GGUF pairs.  No GPU, no model: --mb MB
// of expert blobs (random trellis words and scales - every bit pattern is a valid EXL3 encoding - far more than any
// L3) computed by T threads at once, nt tokens per expert, every expert a fresh read from DRAM.  Two ways to cut it:
//
//   own    each thread walks its own experts (kq_cpu_bench's way): the per-expert comparison with the GGUF pairs
//   split  all T threads on the same experts, in the engine's shape, --batch experts at a time (a layer's CPU share):
//            A   gate/up in split-K pieces of 256 inputs (each piece transforms its own two input blocks first)
//            H1  per 128-column band: the pieces summed in order, the outputs' transform, SwiGLU, the down
//                matrix's input block
//            B   down in split-K pieces of 128 inputs (one input block each)
//            H2  per 128-column band: the pieces summed in order, the outputs' transform
//          fixed cuts, so the sums do not depend on the thread count
//   engine the engine's own path: Strata's ExpertPool (T-1 pinned workers + the host, as --pool-workers T-1) running
//          Exl3LaneMoe's four phases, --batch experts at a time
//   fused  the same with Exl3LaneMoe's phases fused into one (per-expert dependencies, no barriers)
//
// Prints GB/s of weights and the pool's time per expert and per expert-token (an expert's wall time on T threads).
//
//     exl3_lane_bench [--K 4,5] [--threads 9,10] [--nt 1,2,3,4,8] [--mb 4096] [--reps 5] [--mode own,split,engine,fused]
//                     [--batch 4] [--kernel rows|bands] [--ram-gbs N]
//
// --K: each entry the bitrate of all three matrices (4, or a half-integer 3.5) or gate/up : down (3.5:5 - an SC 4.05's
// layers 1-7).
#include "strata/kernels/cpu/exl3_lane.hpp"
#include "strata/kernels/cpu/exl3_lane_moe.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/pool.hpp"
#include "strata/kernels/f16_bits.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#define BENCH_PAUSE() _mm_pause()
#else
#define BENCH_PAUSE() std::this_thread::yield()
#endif

namespace cpu = strata::kernels::cpu;

namespace {

constexpr int kH = 2560, kFF = 640, kMaxT = 8;
constexpr int kGuPiece = 16, kDownPiece = 8;   // split-K pieces in 16-row slices: 256 and 128 inputs
constexpr int kGuPieces = kH / 16 / kGuPiece, kDownPieces = kFF / 16 / kDownPiece;   // 10, 5
constexpr int kGuBands = kFF / 128, kDownBands = kH / 128;                           // 5, 20

std::vector<std::string> split(const std::string& s) {
    std::vector<std::string> v;
    size_t from = 0;
    for (;;) {
        const size_t c = s.find(',', from);
        v.push_back(s.substr(from, c == std::string::npos ? std::string::npos : c - from));
        if (c == std::string::npos) break;
        from = c + 1;
    }
    return v;
}
std::vector<int> ints(const std::string& s) {
    std::vector<int> v;
    for (auto& x : split(s)) v.push_back(std::atoi(x.c_str()));
    return v;
}

// one expert blob: the three trellises, then the six scale vectors (every part a multiple of 64 bytes)
struct Layout {
    int k2gu = 0, k2d = 0;   // the bitrates as 2 K: gate and up, down
    size_t tr_bytes = 0, gate = 0, up = 0, down = 0, suh_g = 0, svh_g = 0, suh_u = 0, svh_u = 0, suh_d = 0, svh_d = 0,
           bytes = 0;
    Layout(int gu, int d) : k2gu(gu), k2d(d) {
        tr_bytes = (size_t) (kH / 16) * (kFF / 16) * 16 * k2gu;   // 102,400 k2 (gate and up)
        gate = 0;
        up = gate + tr_bytes;
        down = up + tr_bytes;
        suh_g = down + (size_t) (kFF / 16) * (kH / 16) * 16 * k2d;
        svh_g = suh_g + 2 * kH;
        suh_u = svh_g + 2 * kFF;
        svh_u = suh_u + 2 * kH;
        suh_d = svh_u + 2 * kFF;
        svh_d = suh_d + 2 * kFF;
        bytes = svh_d + 2 * kH;
    }
    cpu::Exl3Part part(const uint8_t* blob, int which) const {   // 0 gate, 1 up, 2 down
        cpu::Exl3Part m;
        m.k2 = which < 2 ? k2gu : k2d;
        if (which == 0) {
            m.trellis = (const uint32_t*) (blob + gate);
            m.suh = (const uint16_t*) (blob + suh_g);
            m.svh = (const uint16_t*) (blob + svh_g);
            m.k = kH;
            m.n = kFF;
        } else if (which == 1) {
            m.trellis = (const uint32_t*) (blob + up);
            m.suh = (const uint16_t*) (blob + suh_u);
            m.svh = (const uint16_t*) (blob + svh_u);
            m.k = kH;
            m.n = kFF;
        } else {
            m.trellis = (const uint32_t*) (blob + down);
            m.suh = (const uint16_t*) (blob + suh_d);
            m.svh = (const uint16_t*) (blob + svh_d);
            m.k = kFF;
            m.n = kH;
        }
        return m;
    }
};

void make_blob(const Layout& L, uint8_t* blob, std::mt19937_64& rng) {
    for (size_t i = 0; i < L.suh_g; i += 8) {
        const uint64_t w = rng();
        std::memcpy(blob + i, &w, 8);
    }
    // channel scales as a real file has them: magnitude ~[0.25, 2], random sign
    std::uniform_real_distribution<float> u(0.25f, 2.0f);
    for (size_t i = L.suh_g; i < L.bytes; i += 2) {
        const uint16_t h = strata::kernels::f16_from_f32((rng() & 1) ? u(rng) : -u(rng));
        std::memcpy(blob + i, &h, 2);
    }
}

float silu(float g) { return g / (1.0f + std::exp(-g)); }

// ---------------------------------------------------------------- own: one thread, one whole expert
struct OwnScratch {
    std::vector<float> xg, xu, ag, au, g, u, h, xd, ad, out;
    OwnScratch()
        : xg((size_t) kMaxT * kH), xu((size_t) kMaxT * kH), ag((size_t) kMaxT * kFF), au((size_t) kMaxT * kFF),
          g((size_t) kMaxT * kFF), u((size_t) kMaxT * kFF), h((size_t) kMaxT * kFF), xd((size_t) kMaxT * kFF),
          ad((size_t) kMaxT * kH), out((size_t) kMaxT * kH) {}
};

void acc_full(const cpu::Exl3Part& m, const float* const* xh, int nt, float* const* acc, bool bands) {
    if (!bands) {
        cpu::exl3_cpu_rows_acc_n(m, xh, nt, 0, m.k / 16, acc);
        return;
    }
    float* ap[kMaxT];
    for (int c0 = 0; c0 < m.n; c0 += 128) {
        for (int t = 0; t < nt; ++t) ap[t] = acc[t] + c0;
        cpu::exl3_cpu_band_acc_n(m, xh, nt, c0, 0, m.k / 16, ap);
    }
}

void own_expert(const Layout& L, const uint8_t* blob, const float* const* x, int nt, OwnScratch& s, bool bands) {
    const cpu::Exl3Part G = L.part(blob, 0), U = L.part(blob, 1), D = L.part(blob, 2);
    const float* xg[kMaxT] = {};
    const float* xu[kMaxT] = {};
    const float* xd[kMaxT] = {};
    float* ag[kMaxT] = {};
    float* au[kMaxT] = {};
    float* ad[kMaxT] = {};
    float sg[kMaxT] = {}, su[kMaxT] = {}, sd[kMaxT] = {};
    for (int t = 0; t < nt; ++t) {
        float* pg = &s.xg[(size_t) t * kH];
        float* pu = &s.xu[(size_t) t * kH];
        sg[t] = cpu::exl3_cpu_prepare(G, x[t], pg);
        su[t] = cpu::exl3_cpu_prepare(U, x[t], pu);
        xg[t] = pg;
        xu[t] = pu;
        ag[t] = &s.ag[(size_t) t * kFF];
        au[t] = &s.au[(size_t) t * kFF];
        ad[t] = &s.ad[(size_t) t * kH];
    }
    acc_full(G, xg, nt, ag, bands);
    acc_full(U, xu, nt, au, bands);
    for (int t = 0; t < nt; ++t) {
        float* g = &s.g[(size_t) t * kFF];
        float* u = &s.u[(size_t) t * kFF];
        float* h = &s.h[(size_t) t * kFF];
        for (int c0 = 0; c0 < kFF; c0 += 128) {
            cpu::exl3_cpu_band_out(G, ag[t] + c0, sg[t], c0, g + c0);
            cpu::exl3_cpu_band_out(U, au[t] + c0, su[t], c0, u + c0);
        }
        for (int i = 0; i < kFF; ++i) h[i] = silu(g[i]) * u[i];
        float* pd = &s.xd[(size_t) t * kFF];
        sd[t] = cpu::exl3_cpu_prepare(D, h, pd);
        xd[t] = pd;
    }
    acc_full(D, xd, nt, ad, bands);
    for (int t = 0; t < nt; ++t)
        for (int c0 = 0; c0 < kH; c0 += 128) cpu::exl3_cpu_band_out(D, ad[t] + c0, sd[t], c0, &s.out[(size_t) t * kH + c0]);
}

// ---------------------------------------------------------------- split: a small spinning pool, phases of tasks
class Pool {
public:
    explicit Pool(int threads) : n_(threads) {
        for (int i = 1; i < n_; ++i) th_.emplace_back([this, i] { worker(i); });
    }
    ~Pool() {
        stop_.store(true, std::memory_order_release);
        epoch_.fetch_add(1, std::memory_order_acq_rel);
        for (auto& t : th_) t.join();
    }
    // runs fn(task, thread) for task in [0, n) on every thread (the caller is thread 0) and returns when all are done
    void run(int n, const std::function<void(int, int)>& fn) {
        fn_ = &fn;
        ntasks_ = n;
        next_.store(0, std::memory_order_relaxed);
        done_.store(0, std::memory_order_relaxed);
        epoch_.fetch_add(1, std::memory_order_acq_rel);
        drain(0);
        while (done_.load(std::memory_order_acquire) < n) BENCH_PAUSE();
        while (active_.load(std::memory_order_acquire) != 0) BENCH_PAUSE();   // every worker back at the park
    }

private:
    void drain(int id) {
        for (;;) {
            const int i = next_.fetch_add(1, std::memory_order_acq_rel);
            if (i >= ntasks_) break;
            (*fn_)(i, id);
            done_.fetch_add(1, std::memory_order_acq_rel);
        }
    }
    void worker(int id) {
        uint32_t seen = 0;
        for (;;) {
            uint32_t e;
            while ((e = epoch_.load(std::memory_order_acquire)) == seen) BENCH_PAUSE();
            if (stop_.load(std::memory_order_acquire)) return;
            seen = e;
            active_.fetch_add(1, std::memory_order_acq_rel);
            drain(id);
            active_.fetch_sub(1, std::memory_order_acq_rel);
        }
    }
    int n_;
    std::vector<std::thread> th_;
    std::atomic<uint32_t> epoch_{0};
    std::atomic<int> next_{0}, done_{0}, active_{0};
    std::atomic<bool> stop_{false};
    const std::function<void(int, int)>* fn_ = nullptr;
    int ntasks_ = 0;
};

// the buffers of a batch of experts in the engine's shape
struct SplitBufs {
    int B = 0;
    std::vector<float> xg, xu, pg, pu, bsg, bsu, xd, bsd, pd, out;
    explicit SplitBufs(int b) : B(b) {
        xg.resize((size_t) B * kMaxT * kH);
        xu.resize(xg.size());
        pg.resize((size_t) B * kGuPieces * kMaxT * kFF);   // per expert, piece, token: 640 partial sums
        pu.resize(pg.size());
        bsg.resize((size_t) B * kMaxT * (kH / 128));       // the input blocks' sums
        bsu.resize(bsg.size());
        xd.resize((size_t) B * kMaxT * kFF);
        bsd.resize((size_t) B * kMaxT * (kFF / 128));
        pd.resize((size_t) B * kDownPieces * kMaxT * kH);
        out.resize((size_t) B * kMaxT * kH);
    }
};

struct PhaseTimes {
    double a = 0, h1 = 0, b = 0, h2 = 0;
};

void split_batch(const Layout& L, const uint8_t* const* blobs, int B, const float* const* x, int nt, SplitBufs& s,
                 Pool& pool, PhaseTimes& pt) {
    using clk = std::chrono::steady_clock;
    const auto t0 = clk::now();
    // A: gate and up, piece p = slices [16 p, 16 p + 16) = inputs [256 p, 256 p + 256) = input blocks 2p, 2p + 1
    pool.run(B * 2 * kGuPieces, [&](int task, int) {
        const int e = task / (2 * kGuPieces), mat = (task / kGuPieces) % 2, p = task % kGuPieces;
        const cpu::Exl3Part M = L.part(blobs[e], mat);
        std::vector<float>& xb = mat == 0 ? s.xg : s.xu;
        std::vector<float>& bs = mat == 0 ? s.bsg : s.bsu;
        std::vector<float>& pp = mat == 0 ? s.pg : s.pu;
        const float* xh[kMaxT];
        float* acc[kMaxT];
        for (int t = 0; t < nt; ++t) {
            float* xe = &xb[((size_t) e * kMaxT + t) * kH];
            for (int b = 2 * p; b < 2 * p + 2; ++b)
                bs[((size_t) e * kMaxT + t) * (kH / 128) + b] =
                    cpu::exl3_cpu_prepare_block(M.suh + 128 * b, x[t] + 128 * b, xe + 128 * b);
            xh[t] = xe;
            acc[t] = &pp[(((size_t) e * kGuPieces + p) * kMaxT + t) * kFF];
        }
        cpu::exl3_cpu_rows_acc_n(M, xh, nt, kGuPiece * p, kGuPiece * (p + 1), acc);
    });
    const auto t1 = clk::now();
    // H1: band c of gate and up (the pieces summed in order), SwiGLU, the down matrix's input block c
    pool.run(B * kGuBands, [&](int task, int) {
        const int e = task / kGuBands, c = task % kGuBands, c0 = 128 * c;
        const cpu::Exl3Part G = L.part(blobs[e], 0), U = L.part(blobs[e], 1), D = L.part(blobs[e], 2);
        for (int t = 0; t < nt; ++t) {
            float ag[128], au[128], g[128], u[128], h[128];
            std::memcpy(ag, &s.pg[(((size_t) e * kGuPieces + 0) * kMaxT + t) * kFF + c0], sizeof ag);
            std::memcpy(au, &s.pu[(((size_t) e * kGuPieces + 0) * kMaxT + t) * kFF + c0], sizeof au);
            for (int p = 1; p < kGuPieces; ++p) {
                const float* qg = &s.pg[(((size_t) e * kGuPieces + p) * kMaxT + t) * kFF + c0];
                const float* qu = &s.pu[(((size_t) e * kGuPieces + p) * kMaxT + t) * kFF + c0];
                for (int j = 0; j < 128; ++j) ag[j] += qg[j], au[j] += qu[j];
            }
            double sg = 0.0, su = 0.0;   // exl3_cpu_prepare's sum: the block sums in order, in double
            for (int b = 0; b < kH / 128; ++b) {
                sg += s.bsg[((size_t) e * kMaxT + t) * (kH / 128) + b];
                su += s.bsu[((size_t) e * kMaxT + t) * (kH / 128) + b];
            }
            cpu::exl3_cpu_band_out(G, ag, (float) sg, c0, g);
            cpu::exl3_cpu_band_out(U, au, (float) su, c0, u);
            for (int j = 0; j < 128; ++j) h[j] = silu(g[j]) * u[j];
            s.bsd[((size_t) e * kMaxT + t) * (kFF / 128) + c] =
                cpu::exl3_cpu_prepare_block(D.suh + c0, h, &s.xd[((size_t) e * kMaxT + t) * kFF + c0]);
        }
    });
    const auto t2 = clk::now();
    // B: down, piece p = slices [8 p, 8 p + 8) = input block p
    pool.run(B * kDownPieces, [&](int task, int) {
        const int e = task / kDownPieces, p = task % kDownPieces;
        const cpu::Exl3Part D = L.part(blobs[e], 2);
        const float* xh[kMaxT];
        float* acc[kMaxT];
        for (int t = 0; t < nt; ++t) {
            xh[t] = &s.xd[((size_t) e * kMaxT + t) * kFF];
            acc[t] = &s.pd[(((size_t) e * kDownPieces + p) * kMaxT + t) * kH];
        }
        cpu::exl3_cpu_rows_acc_n(D, xh, nt, kDownPiece * p, kDownPiece * (p + 1), acc);
    });
    const auto t3 = clk::now();
    // H2: band c of down, the pieces summed in order
    pool.run(B * kDownBands, [&](int task, int) {
        const int e = task / kDownBands, c = task % kDownBands, c0 = 128 * c;
        const cpu::Exl3Part D = L.part(blobs[e], 2);
        for (int t = 0; t < nt; ++t) {
            float a[128];
            std::memcpy(a, &s.pd[(((size_t) e * kDownPieces + 0) * kMaxT + t) * kH + c0], sizeof a);
            for (int p = 1; p < kDownPieces; ++p) {
                const float* q = &s.pd[(((size_t) e * kDownPieces + p) * kMaxT + t) * kH + c0];
                for (int j = 0; j < 128; ++j) a[j] += q[j];
            }
            double sd = 0.0;
            for (int b = 0; b < kFF / 128; ++b) sd += s.bsd[((size_t) e * kMaxT + t) * (kFF / 128) + b];
            cpu::exl3_cpu_band_out(D, a, (float) sd, c0, &s.out[((size_t) e * kMaxT + t) * kH + c0]);
        }
    });
    const auto t4 = clk::now();
    pt.a += std::chrono::duration<double>(t1 - t0).count();
    pt.h1 += std::chrono::duration<double>(t2 - t1).count();
    pt.b += std::chrono::duration<double>(t3 - t2).count();
    pt.h2 += std::chrono::duration<double>(t4 - t3).count();
}

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

// the split path against the one-thread path on the same experts and tokens: the same outputs up to the order of the
// sums (the pieces are added after their own chains) - the worst difference relative to each output vector's largest
bool check_split(const Layout& L, const uint8_t* blobs, int batch, const float* const* x, int nt, int threads) {
    Pool pool(threads);
    SplitBufs sb(batch);
    std::vector<const uint8_t*> bp((size_t) batch);
    for (int b = 0; b < batch; ++b) bp[(size_t) b] = blobs + (size_t) b * L.bytes;
    PhaseTimes pt;
    split_batch(L, bp.data(), batch, x, nt, sb, pool, pt);
    double worst = 0.0;
    bool finite = true;
    OwnScratch s;
    for (int b = 0; b < batch; ++b) {
        own_expert(L, bp[(size_t) b], x, nt, s, false);
        for (int t = 0; t < nt; ++t) {
            const float* want = &s.out[(size_t) t * kH];
            const float* got = &sb.out[((size_t) b * kMaxT + t) * kH];
            double mx = 0.0, d = 0.0;
            for (int i = 0; i < kH; ++i) {
                if (!std::isfinite(got[i]) || !std::isfinite(want[i])) finite = false;
                mx = std::max(mx, (double) std::fabs(want[i]));
                d = std::max(d, (double) std::fabs((double) got[i] - want[i]));
            }
            worst = std::max(worst, d / std::max(mx, 1e-30));
        }
    }
    const bool ok = finite && worst < 1e-4;
    std::printf("  check: split vs one thread, %d experts x %d tokens: worst %.2e of the largest output  %s\n", batch, nt,
                worst, ok ? "ok" : "FAIL");
    return ok;
}

}  // namespace

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    std::vector<std::string> Ks = {"4", "5"};
    std::vector<int> threads = {9, 10}, nts = {1, 2, 3, 4, 8};
    std::vector<std::string> modes = {"own", "split"};
    int mb = 4096, reps = 5, batch = 4;
    bool bands = false;
    double ram = 0;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        const bool more = i + 1 < argc;
        if (a == "--K" && more) Ks = split(argv[++i]);
        else if (a == "--threads" && more) threads = ints(argv[++i]);
        else if (a == "--nt" && more) nts = ints(argv[++i]);
        else if (a == "--mb" && more) mb = std::atoi(argv[++i]);
        else if (a == "--reps" && more) reps = std::max(1, std::atoi(argv[++i]));
        else if (a == "--mode" && more) modes = split(argv[++i]);
        else if (a == "--batch" && more) batch = std::max(1, std::atoi(argv[++i]));
        else if (a == "--kernel" && more) bands = std::string(argv[++i]) == "bands";
        else if (a == "--ram-gbs" && more) ram = std::atof(argv[++i]);
        else {
            std::fprintf(stderr,
                         "usage: exl3_lane_bench [--K 4,5] [--threads 9,10] [--nt 1,2,3,4,8] [--mb 4096] [--reps 5] "
                         "[--mode own,split] [--batch 4] [--kernel rows|bands] [--ram-gbs N]\n");
            return 2;
        }
    }
    std::vector<std::pair<int, int>> k2s;   // gate/up, down as 2 K
    for (const std::string& k : Ks) {
        const size_t c = k.find(':');
        const double a = std::atof(k.c_str()), b = c == std::string::npos ? a : std::atof(k.c_str() + c + 1);
        const int ga = (int) std::lround(2 * a), gd = (int) std::lround(2 * b);
        if (2 * a != ga || 2 * b != gd || !cpu::exl3_lane_k2_ok(ga) || !cpu::exl3_lane_k2_ok(gd)) {
            std::fprintf(stderr, "exl3_lane_bench: K %s: 1..8, 1.5, 2.5 or 3.5 (gate/up:down)\n", k.c_str());
            return 2;
        }
        k2s.emplace_back(ga, gd);
    }
    for (int nt : nts)
        if (nt < 1 || nt > kMaxT) {
            std::fprintf(stderr, "exl3_lane_bench: nt %d is out of range (1..%d)\n", nt, kMaxT);
            return 2;
        }
    static const char* tiers[] = {"scalar", "avx2", "avx512"};
    std::printf("exl3_lane_bench: %s (lane tier %s), %d MB of experts per K, kernel %s, reps %d\n",
                cpu::cpu_name().c_str(), tiers[cpu::exl3_cpu_tier()], mb, bands ? "bands" : "split-K rows", reps);
    // activations: kMaxT tokens of x (n_embd), as the engine hands them over (F32)
    std::mt19937 arng(77);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<std::vector<float>> xs(kMaxT, std::vector<float>(kH));
    const float* x[kMaxT];
    for (int t = 0; t < kMaxT; ++t) {
        for (auto& v : xs[(size_t) t]) v = nd(arng);
        x[t] = xs[(size_t) t].data();
    }
    for (const auto& kk : k2s) {
        const Layout L(kk.first, kk.second);
        const int K = 100 * kk.first + kk.second;   // a seed
        const size_t E = std::max<size_t>(64, ((size_t) mb << 20) / L.bytes);
        std::unique_ptr<uint8_t[]> mem(new uint8_t[E * L.bytes + 4096]);
        uint8_t* blobs = (uint8_t*) (((uintptr_t) mem.get() + 4095) & ~(uintptr_t) 4095);
        {   // 16 distinct experts, copied over the whole buffer (the bytes in DRAM are what is measured)
            std::mt19937_64 rng(1234 + K);
            const size_t U = std::min<size_t>(16, E);
            for (size_t e = 0; e < U; ++e) make_blob(L, blobs + e * L.bytes, rng);
            for (size_t e = U; e < E; ++e) std::memcpy(blobs + e * L.bytes, blobs + (e % U) * L.bytes, L.bytes);
        }
        std::printf("\nEXL3 K%g/%g: %zu experts of %.3f MB (%.0f MB)\n", kk.first / 2.0, kk.second / 2.0, E, L.bytes / 1e6,
                    E * (double) L.bytes / 1e6);
        if (std::find(modes.begin(), modes.end(), "split") != modes.end() &&
            !check_split(L, blobs, std::min<int>(batch, (int) E), x, std::min(3, kMaxT), std::max(1, threads[0])))
            return 1;
        for (const std::string& mode : modes) {
            if (mode != "own" && mode != "split" && mode != "engine" && mode != "fused") {
                std::printf("  mode %s: unknown (own, split, engine, fused)\n", mode.c_str());
                continue;
            }
            for (int nt : nts)
                for (int T : threads) {
                    if (T < 1) continue;
                    std::vector<double> secs;
                    PhaseTimes pt;
                    if (mode == "own") {
                        for (int rep = 0; rep < reps; ++rep) {
                            std::atomic<size_t> next{0};
                            auto work = [&] {
                                OwnScratch s;
                                for (;;) {
                                    const size_t e = next.fetch_add(1);
                                    if (e >= E) break;
                                    own_expert(L, blobs + e * L.bytes, x, nt, s, bands);
                                }
                                volatile float sink = s.out[0];
                                (void) sink;
                            };
                            const auto t0 = std::chrono::steady_clock::now();
                            std::vector<std::thread> th;
                            for (int k = 0; k < T; ++k) th.emplace_back(work);
                            for (auto& t : th) t.join();
                            secs.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
                        }
                    } else if (mode == "engine" || mode == "fused") {
                        // T threads = T-1 pinned pool workers + this thread as the pool's host (pinned where the engine
                        // puts its host loop)
                        cpu::ExpertPool pool(std::max(1, T - 1), /*pin=*/true, /*host_works=*/true);
                        const cpu::ThreadAffinity prev = cpu::pin_current_thread(cpu::planned_host_core());
                        cpu::Exl3LaneMoe moe;
                        std::string err;
                        if (!moe.init(kH, kFF, err)) {
                            std::fprintf(stderr, "exl3_lane_bench: %s\n", err.c_str());
                            return 1;
                        }
                        moe.set_fused(mode == "fused");
                        std::vector<cpu::Exl3ExpertJob> jobs((size_t) batch);
                        std::vector<float> outb((size_t) batch * kMaxT * kH);
                        for (int rep = 0; rep < reps; ++rep) {
                            moe.ms_gu = moe.ms_mid = moe.ms_down = moe.ms_out = 0;
                            const auto t0 = std::chrono::steady_clock::now();
                            for (size_t e0 = 0; e0 + (size_t) batch <= E; e0 += (size_t) batch) {
                                for (int b = 0; b < batch; ++b) {
                                    const uint8_t* blob = blobs + (e0 + (size_t) b) * L.bytes;
                                    cpu::Exl3ExpertJob& j = jobs[(size_t) b];
                                    j.e = cpu::Exl3Expert{L.part(blob, 0), L.part(blob, 1), L.part(blob, 2)};
                                    j.nt = nt;
                                    for (int t = 0; t < nt; ++t) {
                                        j.x[t] = x[t];
                                        j.out[t] = &outb[((size_t) b * kMaxT + t) * kH];
                                    }
                                }
                                if (!moe.run(pool, jobs.data(), batch, err)) {
                                    std::fprintf(stderr, "exl3_lane_bench: %s\n", err.c_str());
                                    return 1;
                                }
                            }
                            secs.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
                            if (rep == 0 || secs.back() <= *std::min_element(secs.begin(), secs.end())) {
                                pt.a = moe.ms_gu / 1e3;
                                pt.h1 = moe.ms_mid / 1e3;
                                pt.b = moe.ms_down / 1e3;
                                pt.h2 = moe.ms_out / 1e3;
                            }
                        }
                        cpu::restore_thread_affinity(prev);
                    } else {
                        Pool pool(T);
                        SplitBufs sb(batch);
                        std::vector<const uint8_t*> bp((size_t) batch);
                        for (int rep = 0; rep < reps; ++rep) {
                            PhaseTimes rp;
                            const auto t0 = std::chrono::steady_clock::now();
                            for (size_t e0 = 0; e0 + (size_t) batch <= E; e0 += (size_t) batch) {
                                for (int b = 0; b < batch; ++b) bp[(size_t) b] = blobs + (e0 + (size_t) b) * L.bytes;
                                split_batch(L, bp.data(), batch, x, nt, sb, pool, rp);
                            }
                            secs.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
                            if (rep == 0 || secs.back() <= *std::min_element(secs.begin(), secs.end())) pt = rp;
                        }
                    }
                    const size_t done = mode == "own" ? E : E / (size_t) batch * (size_t) batch;
                    const double med = median(secs), best = *std::min_element(secs.begin(), secs.end());
                    const double gbs = done * (double) L.bytes / med / 1e9;
                    const double us = med / (double) done * 1e6, us_best = best / (double) done * 1e6;
                    std::printf("  %-5s nt %d threads %2d: %6.1f GB/s of weights, %7.2f us per expert (best %7.2f), "
                                "%7.2f us per expert-token%s\n",
                                mode.c_str(), nt, T, gbs, us, us_best, us / nt,
                                ram > 0 ? (" = " + std::to_string((int) std::lround(100 * gbs / ram)) + "% of RAM").c_str()
                                        : "");
                    if (mode != "own") {
                        const double n = (double) done;
                        std::printf("        phases per expert (fastest round): gate/up %.2f, band+SwiGLU %.2f, down %.2f, "
                                    "out %.2f us\n",
                                    pt.a / n * 1e6, pt.h1 / n * 1e6, pt.b / n * 1e6, pt.h2 / n * 1e6);
                    }
                }
        }
    }
    return 0;
}
