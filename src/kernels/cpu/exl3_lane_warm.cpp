// src/kernels/cpu/exl3_lane_warm.cpp - proto STRATA_EXL3_LANE_WARM (see exl3_lane_warm.hpp).  The router dot products
// and the selection are Basalt's (src/core/verify.cpp warm_logits / warm_select; MIT, third_party/basalt/LICENSE).
#include "strata/kernels/cpu/exl3_lane_warm.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#define WARM_AVX2 __attribute__((target("avx2,fma")))
#define WARM_PAUSE() _mm_pause()
#else
#include <thread>
#define WARM_AVX2
#define WARM_PAUSE() std::this_thread::yield()
#endif

namespace strata::kernels::cpu {

namespace {
float bf16_to_f32(uint16_t b) {
    const uint32_t u = (uint32_t) b << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}
// 32 values -> int8 with one scale (max |v| / 127); returns the scale
float quant32(const float* v, int8_t* q) {
    float m = 0.0f;
    for (int i = 0; i < 32; ++i) m = std::max(m, std::fabs(v[i]));
    const float s = m / 127.0f, inv = s > 0.0f ? 1.0f / s : 0.0f;
    for (int i = 0; i < 32; ++i) q[i] = (int8_t) std::lround(v[i] * inv);
    return s;
}
}  // namespace

bool Exl3LaneWarm::init(const std::vector<std::vector<uint16_t>>& routers, int n_embd, int n_expert, int k, int nsel,
                        LaneBlobFn lane_blob, void* ctx, std::string& err) {
    if (n_embd <= 0 || n_embd % 32 != 0 || n_expert <= 0 || n_expert > 512 || routers.empty() || lane_blob == nullptr) {
        err = "the lane warm-up needs n_embd a multiple of 32, at most 512 experts, the routers and the lane's blobs";
        return false;
    }
    const int L = (int) routers.size(), NB = n_embd / 32;
    rq_.assign((size_t) L * n_expert * n_embd, 0);
    rs_.assign((size_t) L * n_expert * NB, 0.0f);
    for (int l = 0; l < L; ++l) {
        if (routers[(size_t) l].size() != (size_t) n_expert * n_embd) {
            err = "the lane warm-up: layer " + std::to_string(l) + "'s router has another shape";
            return false;
        }
        float v[32];
        for (int e = 0; e < n_expert; ++e)
            for (int b = 0; b < NB; ++b) {
                const size_t o = ((size_t) l * n_expert + e) * n_embd + (size_t) b * 32;
                for (int i = 0; i < 32; ++i) v[i] = bf16_to_f32(routers[(size_t) l][(size_t) e * n_embd + b * 32 + i]);
                rs_[((size_t) l * n_expert + e) * NB + b] = quant32(v, rq_.data() + o);
            }
    }
    n_layers_ = L;
    n_expert_ = n_expert;
    n_embd_ = n_embd;
    k_ = std::max(1, std::min(k, 16));
    nsel_ = std::max(1, std::min(nsel, 16));
    lane_blob_ = lane_blob;
    ctx_ = ctx;
    xq_.assign((size_t) MAXT * n_embd, 0);
    xs_.assign((size_t) MAXT * NB, 0.0f);
    lg_.assign((size_t) MAXT * n_expert, 0.0f);
    pred_.assign((size_t) L, {});
    warmed_.assign((size_t) L, {});
    return true;
}

void Exl3LaneWarm::start(ExpertPool& pool, int layer, const float* x, int n_tok, uint64_t blob_bytes_next) {
    if (n_layers_ == 0 || layer < 0 || layer >= n_layers_ || n_tok < 1 || n_tok > MAXT) return;
    pool_ = &pool;
    pool.stop_background();   // the previous warm-up (normally long stopped by the lane's own phase)
    const int next = (layer + 1) % n_layers_;
    history_ = next == 0;
    layer_ = next;
    n_tok_ = n_tok;
    bytes_ = blob_bytes_next;
    nblobs_ = 0;
    pred_[(size_t) next].clear();
    warmed_[(size_t) next].clear();
    if (history_) {   // layer 0: the last round's lane experts there, as they are placed now
        for (const int32_t e : first_) {
            if (nblobs_ >= nsel_) break;
            if (const uint8_t* b = lane_blob_(ctx_, 0, e)) {
                sel_[nblobs_] = e;
                blobs_[nblobs_++] = b;
            }
        }
        pred_[0].assign(sel_, sel_ + nblobs_);
        warmed_[0] = pred_[0];
        if (nblobs_ == 0) return;
        ready_.store(1, std::memory_order_relaxed);
    } else {
        x_ = x;   // quantized by the first task (off the host's path back to the GPU)
        xready_.store(0, std::memory_order_relaxed);
        ready_.store(0, std::memory_order_relaxed);
    }
    W_ = pool.workers();
    arrive_.store(0, std::memory_order_relaxed);
    left_.store(W_, std::memory_order_relaxed);
    ++st_warm_starts;
    pool.start_background(W_, &Exl3LaneWarm::task, this);
}

WARM_AVX2 void Exl3LaneWarm::logits(int e0, int e1) {
#if defined(__x86_64__) || defined(_M_X64)
    const int NB = n_embd_ / 32, pt = n_tok_;
    const int8_t* R = rq_.data() + (size_t) layer_ * n_expert_ * n_embd_;
    const float* S = rs_.data() + (size_t) layer_ * n_expert_ * NB;
    for (int e = e0; e < e1; ++e) {
        if (pool_->bg_stopping()) return;
        __m256 acc[MAXT];
        for (int t = 0; t < pt; ++t) acc[t] = _mm256_setzero_ps();
        const int8_t* row = R + (size_t) e * n_embd_;
        for (int b = 0; b < NB; ++b) {
            const __m256i w = _mm256_loadu_si256((const __m256i*) (row + b * 32));
            const __m256i wl = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(w));
            const __m256i wh = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(w, 1));
            const float sw = S[(size_t) e * NB + b];
            for (int t = 0; t < pt; ++t) {
                const __m256i xv = _mm256_loadu_si256((const __m256i*) (xq_.data() + (size_t) t * n_embd_ + b * 32));
                const __m256i p = _mm256_add_epi32(_mm256_madd_epi16(wl, _mm256_cvtepi8_epi16(_mm256_castsi256_si128(xv))),
                                                   _mm256_madd_epi16(wh, _mm256_cvtepi8_epi16(_mm256_extracti128_si256(xv, 1))));
                acc[t] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p), _mm256_set1_ps(sw * xs_[(size_t) t * NB + b]), acc[t]);
            }
        }
        for (int t = 0; t < pt; ++t) {
            __m128 v = _mm_add_ps(_mm256_castps256_ps128(acc[t]), _mm256_extractf128_ps(acc[t], 1));
            v = _mm_add_ps(v, _mm_movehl_ps(v, v));
            v = _mm_add_ss(v, _mm_shuffle_ps(v, v, 1));
            lg_[(size_t) t * n_expert_ + e] = _mm_cvtss_f32(v);
        }
    }
#else
    (void) e0;
    (void) e1;
#endif
}

// every token's top k (softmax weights over them), summed per predicted expert the lane runs; the likeliest nsel
void Exl3LaneWarm::select() {
    const int NE = n_expert_, K = k_;
    float score[512] = {};
    std::vector<int32_t>& pred = pred_[(size_t) layer_];
    for (int t = 0; t < n_tok_; ++t) {
        const float* lg = lg_.data() + (size_t) t * NE;
        int top[16];
        float val[16];
        int m = 0;
        for (int e = 0; e < NE; ++e) {
            const float v = lg[e];
            if (m == K && v <= val[K - 1]) continue;
            int j = m < K ? m++ : K - 1;
            while (j > 0 && val[j - 1] < v) { val[j] = val[j - 1]; top[j] = top[j - 1]; --j; }
            val[j] = v;
            top[j] = e;
        }
        float w[16], sum = 0.0f;
        for (int j = 0; j < m; ++j) sum += (w[j] = std::exp(val[j] - val[0]));
        for (int j = 0; j < m; ++j) {
            const int e = top[j];
            if (lane_blob_(ctx_, layer_, e) == nullptr) continue;   // a GPU tier holds it
            if (score[e] == 0.0f) pred.push_back(e);
            score[e] += w[j] / sum;
        }
    }
    std::vector<int32_t> order(pred);
    std::sort(order.begin(), order.end(), [&](int32_t a, int32_t b) { return score[a] > score[b] || (score[a] == score[b] && a < b); });
    nblobs_ = 0;
    for (const int32_t e : order) {
        if (nblobs_ >= nsel_) break;
        sel_[nblobs_] = e;
        blobs_[nblobs_++] = lane_blob_(ctx_, layer_, e);
    }
    warmed_[(size_t) layer_].assign(sel_, sel_ + nblobs_);
}

void Exl3LaneWarm::task(void* ctx, int w, int /*thread*/) {
    Exl3LaneWarm& c = *(Exl3LaneWarm*) ctx;
    const int W = c.W_;
    if (!c.history_) {
        if (w == 0) {
            const int NB = c.n_embd_ / 32;
            for (int t = 0; t < c.n_tok_; ++t)
                for (int b = 0; b < NB; ++b)
                    c.xs_[(size_t) t * NB + b] = quant32(c.x_ + (size_t) t * c.n_embd_ + b * 32,
                                                         c.xq_.data() + (size_t) t * c.n_embd_ + b * 32);
            c.xready_.store(1, std::memory_order_release);
        } else {
            while (c.xready_.load(std::memory_order_acquire) == 0) {
                if (c.pool_->bg_stopping()) break;
                WARM_PAUSE();
            }
        }
        c.logits(c.n_expert_ * w / W, c.n_expert_ * (w + 1) / W);
        if (c.arrive_.fetch_add(1, std::memory_order_acq_rel) + 1 == W) {
            if (!c.pool_->bg_stopping()) c.select();
            c.ready_.store(1, std::memory_order_release);
        } else {
            while (c.ready_.load(std::memory_order_acquire) == 0) {
                if (c.pool_->bg_stopping()) { c.left_.fetch_sub(1, std::memory_order_relaxed); return; }
                WARM_PAUSE();
            }
        }
    }
    // this worker's share of each selected blob in rank order, one load per line, 4 KiB at a time between the checks
    uint64_t x = 0, bytes = 0;
    bool stopped = false;
    const size_t B = (size_t) c.bytes_, per = ((B + (size_t) W - 1) / (size_t) W + 63) & ~(size_t) 63;
    for (int i = 0; i < c.nblobs_ && !stopped; ++i) {
        const uint8_t* blob = c.blobs_[i];
        const size_t b0 = std::min(B, per * (size_t) w), b1 = std::min(B, b0 + per);
        for (size_t o = b0; o < b1; o += 4096) {
            if (c.pool_->bg_stopping()) { stopped = true; break; }
            const size_t n = std::min<size_t>(4096, b1 - o);
            for (size_t q = 0; q + 8 <= n; q += 64) x += *(const volatile uint64_t*) (blob + o + q);
            bytes += n;
        }
    }
    c.st_bytes.fetch_add((int64_t) bytes, std::memory_order_relaxed);
    if (c.left_.fetch_sub(1, std::memory_order_acq_rel) == 1 && !stopped) c.st_done.fetch_add(1, std::memory_order_relaxed);
    if (x == 0x9e3779b97f4a7c15ull) c.sink_.fetch_add(x, std::memory_order_relaxed);   // keeps the loads
}

void Exl3LaneWarm::score(int layer, const int32_t* lane, int n) {
    if (n_layers_ == 0 || layer < 0 || layer >= n_layers_) return;
    if (layer == 0) first_.assign(lane, lane + n);
    if (st_warm_starts == 0) return;
    if (layer_ == layer && ready_.load(std::memory_order_acquire) == 0) {   // the prediction did not finish in time
        ++st_layers;
        ++st_late;
        st_lane += n;
        return;
    }
    const std::vector<int32_t>& p = pred_[(size_t) layer];
    const std::vector<int32_t>& wv = warmed_[(size_t) layer];
    ++st_layers;
    st_lane += n;
    st_pred += (int64_t) p.size();
    for (int i = 0; i < n; ++i) {
        if (std::find(p.begin(), p.end(), lane[i]) != p.end()) ++st_hit;
        if (std::find(wv.begin(), wv.end(), lane[i]) != wv.end()) ++st_warmed_hit;
    }
}

}  // namespace strata::kernels::cpu
