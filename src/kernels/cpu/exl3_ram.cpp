// src/kernels/cpu/exl3_ram.cpp - the resident RAM tier of an EXL3 model's routed experts (see exl3_ram.hpp).
#include "strata/kernels/cpu/exl3_ram.hpp"

#include "strata/artifact/exl3_model.hpp"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <map>
#include <mutex>
#include <thread>

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

namespace strata::kernels::cpu {

namespace {

uint64_t up64(uint64_t v) { return (v + 63) & ~(uint64_t) 63; }

std::string expert_base(int layer, int e) {
    return "model.language_model.layers." + std::to_string(layer) + ".mlp.experts." + std::to_string(e) + ".";
}

struct Read {
    int file = 0;
    uint64_t off = 0, bytes = 0;
    uint8_t* dst = nullptr;
};

bool pread_all(int fd, uint8_t* dst, uint64_t n, uint64_t off) {
    while (n > 0) {
        const ssize_t r = ::pread(fd, dst, (size_t) std::min<uint64_t>(n, (uint64_t) 1 << 30), (off_t) off);
        if (r < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        if (r == 0) return false;   // the file ends early
        dst += r;
        n -= (uint64_t) r;
        off += (uint64_t) r;
    }
    return true;
}

double gib(uint64_t b) { return (double) b / (double) (1ull << 30); }

}  // namespace

Exl3ExpertRam::~Exl3ExpertRam() {
    if (base_ != nullptr) ::munmap(base_, map_bytes_);
}

bool Exl3ExpertRam::open(const std::string& dir, int n_layers, int n_expert, int threads, uint64_t budget,
                         std::string& err) {
    if (base_ != nullptr) {
        err = "EXL3 RAM tier: already loaded";
        return false;
    }
    if (n_layers <= 0 || n_expert <= 0) {
        err = "EXL3 RAM tier: " + std::to_string(n_layers) + " layers x " + std::to_string(n_expert) + " experts";
        return false;
    }
    threads = std::clamp(threads, 1, 32);
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<LayerFmt> lay((size_t) n_layers);
    std::vector<std::string> files;
    std::vector<Read> reads;
    uint8_t* map = nullptr;
    uint64_t total = 0;
    auto fail = [&](const std::string& why) {
        err = "EXL3 RAM tier: " + why;
        if (map != nullptr) ::munmap(map, total);
        return false;
    };
    try {
        const strata::Exl3Model model(dir);
        // 1. every layer's shapes, from its expert 0 (every other expert is checked against them below)
        for (int l = 0; l < n_layers; ++l) {
            const std::string b = expert_base(l, 0);
            for (const char* r : {"gate_proj", "up_proj", "down_proj"})
                if (!model.has(b + r)) return fail("the model has no " + b + r);
            const strata::Exl3Linear g = model.linear(b + "gate_proj"), u = model.linear(b + "up_proj"),
                                     d = model.linear(b + "down_proj");
            if (g.cb != 2 || u.cb != 2 || d.cb != 2)
                return fail("layer " + std::to_string(l) + "'s experts are not mul1 (the only codebook the lane runs)");
            if (g.in() != u.in() || g.out() != u.out() || d.in() != g.out() || d.out() != g.in() || g.in() % 128 != 0 ||
                g.out() % 128 != 0)
                return fail("layer " + std::to_string(l) + "'s expert shapes do not chain: gate " +
                            std::to_string(g.in()) + "x" + std::to_string(g.out()) + ", up " + std::to_string(u.in()) +
                            "x" + std::to_string(u.out()) + ", down " + std::to_string(d.in()) + "x" +
                            std::to_string(d.out()));
            LayerFmt& f = lay[(size_t) l];
            f.kg = g.int_bits(b + "gate_proj (the RAM tier)");
            f.ku = u.int_bits(b + "up_proj (the RAM tier)");
            f.kd = d.int_bits(b + "down_proj (the RAM tier)");
            f.h = g.in();
            f.ff = g.out();
            const uint64_t tile_words = (uint64_t) f.h / 16 * (uint64_t) (f.ff / 16);   // 16x16 tiles a matrix
            f.up = up64(tile_words * 32 * (uint64_t) f.kg);
            f.down = f.up + up64(tile_words * 32 * (uint64_t) f.ku);
            f.suh_g = f.down + up64(tile_words * 32 * (uint64_t) f.kd);
            f.svh_g = f.suh_g + up64(2 * (uint64_t) f.h);
            f.suh_u = f.svh_g + up64(2 * (uint64_t) f.ff);
            f.svh_u = f.suh_u + up64(2 * (uint64_t) f.h);
            f.suh_d = f.svh_u + up64(2 * (uint64_t) f.ff);
            f.svh_d = f.suh_d + up64(2 * (uint64_t) f.ff);
            f.blob = up64(f.svh_d + 2 * (uint64_t) f.h);
            f.base = total;
            total += f.blob * (uint64_t) n_expert;
        }
        if (budget > 0 && total > budget)
            return fail(std::to_string(gib(total)) + " GiB of experts against a budget of " + std::to_string(gib(budget)) +
                        " GiB");
        // 2. the tier, and every tensor's place in it (its size checked against the layer's shape)
        void* m = ::mmap(nullptr, total, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
        if (m == MAP_FAILED) return fail("cannot map " + std::to_string(gib(total)) + " GiB: " + std::strerror(errno));
        map = (uint8_t*) m;
        (void) ::madvise(map, total, MADV_HUGEPAGE);   // fewer TLB misses for the lane's reads; a hint only
        std::map<std::string, int> file_ix;
        reads.reserve((size_t) n_layers * (size_t) n_expert * 9);
        for (int l = 0; l < n_layers; ++l) {
            const LayerFmt& f = lay[(size_t) l];
            const uint64_t tile_words = (uint64_t) f.h / 16 * (uint64_t) (f.ff / 16);
            const struct {
                const char* name;
                uint64_t at, bytes;
            } part[9] = {
                {"gate_proj.trellis", 0, tile_words * 32 * (uint64_t) f.kg},
                {"up_proj.trellis", f.up, tile_words * 32 * (uint64_t) f.ku},
                {"down_proj.trellis", f.down, tile_words * 32 * (uint64_t) f.kd},
                {"gate_proj.suh", f.suh_g, 2 * (uint64_t) f.h},
                {"gate_proj.svh", f.svh_g, 2 * (uint64_t) f.ff},
                {"up_proj.suh", f.suh_u, 2 * (uint64_t) f.h},
                {"up_proj.svh", f.svh_u, 2 * (uint64_t) f.ff},
                {"down_proj.suh", f.suh_d, 2 * (uint64_t) f.ff},
                {"down_proj.svh", f.svh_d, 2 * (uint64_t) f.h},
            };
            for (int e = 0; e < n_expert; ++e) {
                const std::string b = expert_base(l, e);
                uint8_t* blob = map + f.base + (uint64_t) e * f.blob;
                for (const auto& p : part) {
                    std::string path;
                    size_t off = 0, bytes = 0;
                    if (!model.locate(b + p.name, path, off, bytes)) return fail("the model has no " + b + p.name);
                    if ((uint64_t) bytes != p.bytes)
                        return fail(b + p.name + " is " + std::to_string(bytes) + " bytes; its layer's shape says " +
                                    std::to_string(p.bytes));
                    auto it = file_ix.find(path);
                    if (it == file_ix.end()) {
                        it = file_ix.emplace(path, (int) files.size()).first;
                        files.push_back(path);
                    }
                    reads.push_back(Read{it->second, (uint64_t) off, (uint64_t) bytes, blob + p.at});
                }
            }
        }
    } catch (const std::exception& e) {
        return fail(e.what());
    }
    // 3. the reads in file order, `threads` contiguous shares of about equal bytes, the page cache dropped behind them
    std::sort(reads.begin(), reads.end(),
              [](const Read& a, const Read& b) { return a.file != b.file ? a.file < b.file : a.off < b.off; });
    uint64_t sum = 0;
    for (const Read& r : reads) sum += r.bytes;
    std::vector<size_t> cut((size_t) threads + 1, reads.size());
    cut[0] = 0;
    {
        uint64_t acc = 0;
        int t = 1;
        for (size_t i = 0; i < reads.size() && t < threads; ++i) {
            acc += reads[i].bytes;
            if (acc >= sum / (uint64_t) threads * (uint64_t) t) cut[(size_t) t++] = i + 1;
        }
    }
    std::atomic<bool> bad{false};
    std::mutex emu;
    std::string first_error;
    auto worker = [&](size_t i0, size_t i1) {
        std::vector<int> fd(files.size(), -1);
        for (size_t i = i0; i < i1 && !bad.load(std::memory_order_relaxed); ++i) {
            const Read& r = reads[i];
            int& f = fd[(size_t) r.file];
            if (f < 0) f = ::open(files[(size_t) r.file].c_str(), O_RDONLY | O_CLOEXEC);
            if (f < 0 || !pread_all(f, r.dst, r.bytes, r.off)) {
                std::lock_guard<std::mutex> lk(emu);
                if (first_error.empty())
                    first_error = files[(size_t) r.file] + ": " + (f < 0 ? "cannot open: " : "short read at " +
                                  std::to_string(r.off) + ": ") + std::strerror(errno);
                bad.store(true);
                break;
            }
            if (r.bytes >= (64u << 10)) (void) ::posix_fadvise(f, (off_t) r.off, (off_t) r.bytes, POSIX_FADV_DONTNEED);
        }
        for (int f : fd)
            if (f >= 0) ::close(f);
    };
    std::vector<std::thread> th;
    for (int t = 0; t < threads; ++t)
        if (cut[(size_t) t] < cut[(size_t) t + 1]) th.emplace_back(worker, cut[(size_t) t], cut[(size_t) t + 1]);
    for (auto& t : th) t.join();
    if (bad.load()) return fail(first_error);
    lay_ = std::move(lay);
    n_expert_ = n_expert;
    base_ = map;
    bytes_ = map_bytes_ = total;
    load_s_ = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return true;
}

Exl3Expert Exl3ExpertRam::expert(int layer, int e) const {
    const LayerFmt& f = lay_[(size_t) layer];
    const uint8_t* b = base_ + f.base + (uint64_t) e * f.blob;
    Exl3Expert x;
    x.gate = Exl3Part{(const uint32_t*) b, (const uint16_t*) (b + f.suh_g), (const uint16_t*) (b + f.svh_g), f.h, f.ff,
                      2 * f.kg};
    x.up = Exl3Part{(const uint32_t*) (b + f.up), (const uint16_t*) (b + f.suh_u), (const uint16_t*) (b + f.svh_u), f.h,
                    f.ff, 2 * f.ku};
    x.down = Exl3Part{(const uint32_t*) (b + f.down), (const uint16_t*) (b + f.suh_d), (const uint16_t*) (b + f.svh_d),
                      f.ff, f.h, 2 * f.kd};
    return x;
}

const uint8_t* Exl3ExpertRam::blob(int layer, int e) const {
    const LayerFmt& f = lay_[(size_t) layer];
    return base_ + f.base + (uint64_t) e * f.blob;
}

uint64_t Exl3ExpertRam::blob_bytes(int layer) const { return lay_[(size_t) layer].blob; }

std::string Exl3ExpertRam::describe() const {
    if (lay_.empty()) return "EXL3 RAM tier: empty";
    char buf[256];
    const LayerFmt& f = lay_[0];
    bool uniform = true;
    for (const LayerFmt& g : lay_) uniform &= g.kg == f.kg && g.ku == f.ku && g.kd == f.kd;
    std::snprintf(buf, sizeof buf, "%d layers x %d experts, gate/up/down K %d/%d/%d%s (%.2f MB an expert), %.1f GiB, "
                  "loaded in %.1f s", (int) lay_.size(), n_expert_, f.kg, f.ku, f.kd, uniform ? "" : " in layer 0 (mixed)",
                  (double) f.blob / 1e6, gib(bytes_), load_s_);
    return buf;
}

}  // namespace strata::kernels::cpu
