// include/strata/artifact/exl3_model.hpp - multi-shard EXL3 model accessor (docs/EXL3.md).
//
// An EXL3 model is a set of safetensors shards plus `model.safetensors.index.json` (tensor -> shard).
// Every quantized linear is `prefix.trellis` (I16 [ki,nj,16*K]), `prefix.suh`/`prefix.svh` (F16) and a
// `prefix.mul1` / `prefix.mcg` marker.  This resolves a linear across shards and hands back the raw
// pointers the decode kernels take.  Shards are mmap'd lazily and never copied.
//
// The bitrate K comes from the tile width, 16 K uint16 words: an integer 1..8, or (exllamav3 1.6) a half-integer
// 1.5 / 2.5 / 3.5 - 24 / 40 / 56 words, alternating KA- and (KA + 1)-bit trellis steps, mul1 only (docs/EXL3.md).  It
// is carried as k2 = 2 K, exllamav3's own unit (bits_k.cuh: k2_from_K); a path that decodes integer K only takes
// int_bits(), which refuses a half-integer K by name instead of reading 56 words a tile as K = 3.
#pragma once

#include <cstdint>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

#include "strata/artifact/safetensors.hpp"

namespace strata {

/// "4" or "3.5": a bitrate in half-bit units (k2 = 2 K) as text, for messages.
inline std::string exl3_k_label(int k2) {
    return std::to_string(k2 / 2) + ((k2 & 1) ? ".5" : "");
}
/// Whether k2 is a bitrate exllamav3 1.6 writes and Strata decodes: K 1..8 (k2 2..16 even) or K 1.5 / 2.5 / 3.5 (k2 3,
/// 5, 7; the mul1 codebook only - the caller checks that).
inline bool exl3_k2_valid(int k2) { return (k2 >= 2 && k2 <= 16 && (k2 & 1) == 0) || k2 == 3 || k2 == 5 || k2 == 7; }

struct Exl3Linear {
    const uint16_t* trellis = nullptr;
    const uint16_t* suh = nullptr;
    const uint16_t* svh = nullptr;
    int ki = 0, nj = 0;
    int k2 = 0;                        // twice the bitrate K: 2..16 even = K 1..8; 3, 5, 7 = K 1.5, 2.5, 3.5 (mul1 only)
    int cb = 2;                        // 0=3inst, 1=mcg, 2=mul1
    int in() const { return ki * 16; }
    int out() const { return nj * 16; }
    bool half() const { return (k2 & 1) != 0; }
    /// uint16 words a 16x16 tile: 16 K (56 at K 3.5)
    int tile_words() const { return 8 * k2; }
    /// The integer K, for a path that decodes integer K only (#1764's GPU kernels, the dense requantizer); a
    /// half-integer K throws, naming `who` - never read as its integer part.
    int int_bits(const std::string& who) const {
        if (half())
            throw std::runtime_error(who + ": K " + exl3_k_label(k2) + " (a half-integer EXL3 bitrate) is not supported "
                                     "on this path; it decodes integer K only");
        return k2 / 2;
    }
};

namespace exl3_detail {

// Parse the `weight_map` object of a safetensors index: {"tensor.name":"model-0000N-of-0000M.safetensors", ...}.
inline std::map<std::string, std::string> parse_weight_map(const std::string& json) {
    std::map<std::string, std::string> out;
    size_t p = json.find("\"weight_map\"");
    if (p == std::string::npos) throw std::runtime_error("index.json: no weight_map");
    p = json.find('{', p);
    if (p == std::string::npos) throw std::runtime_error("index.json: malformed weight_map");
    ++p;
    auto skip_ws = [&]() { while (p < json.size() && (json[p] == ' ' || json[p] == '\n' || json[p] == '\r' || json[p] == '\t')) ++p; };
    auto str = [&]() -> std::string {
        if (p >= json.size() || json[p] != '"') throw std::runtime_error("index.json: expected string");
        ++p;
        std::string s;
        while (p < json.size() && json[p] != '"') {
            if (json[p] == '\\' && p + 1 < json.size()) { ++p; }
            s.push_back(json[p++]);
        }
        ++p;
        return s;
    };
    skip_ws();
    if (json[p] == '}') return out;
    for (;;) {
        skip_ws();
        std::string key = str();
        skip_ws(); ++p;                        // ':'
        skip_ws();
        std::string val = str();
        out.emplace(std::move(key), std::move(val));
        skip_ws();
        if (json[p] == ',') { ++p; continue; }
        if (json[p] == '}') break;
        throw std::runtime_error("index.json: malformed weight_map entry");
    }
    return out;
}

}  // namespace exl3_detail

class Exl3Model {
public:
    explicit Exl3Model(const std::string& dir) : dir_(dir) {
        std::string path = dir + "/model.safetensors.index.json";
        std::ifstream in(path, std::ios::binary);
        if (!in) throw std::runtime_error("exl3: cannot open " + path);
        std::ostringstream ss;
        ss << in.rdbuf();
        weight_map_ = exl3_detail::parse_weight_map(ss.str());
    }

    bool has(const std::string& base) const { return weight_map_.count(base + ".trellis") != 0; }
    /// The shard path and the FILE byte range of a tensor, for readers that pread instead of touching the mmap.
    bool locate(const std::string& name, std::string& path, size_t& off, size_t& bytes) const {
        auto it = weight_map_.find(name);
        if (it == weight_map_.end()) return false;
        const SafetensorsFile& sf = shard(it->second);
        const StTensor* t = sf.find(name);
        if (t == nullptr) return false;
        path = dir_ + "/" + it->second;
        off = sf.data_offset() + t->begin;
        bytes = t->nbytes();
        return true;
    }
    size_t tensor_count() const { return weight_map_.size(); }

    Exl3Linear linear(const std::string& base) const {
        auto tr = tensor(base + ".trellis");
        auto suh = tensor(base + ".suh");
        auto svh = tensor(base + ".svh");
        if (tr->dtype != "I16" || tr->shape.size() != 3 || tr->shape[0] <= 0 || tr->shape[1] <= 0)
            throw std::runtime_error("exl3: " + base + ".trellis is not an I16 [k/16, n/16, words] tensor");
        Exl3Linear L;
        L.trellis = (const uint16_t*)data(*tr);
        L.suh = (const uint16_t*)data(*suh);
        L.svh = (const uint16_t*)data(*svh);
        L.ki = (int)tr->shape[0];
        L.nj = (int)tr->shape[1];
        L.cb = weight_map_.count(base + ".mul1") ? 2 : weight_map_.count(base + ".mcg") ? 1 : 0;
        // the tile width is 16 K words: 16 | words -> integer K; 8 | words (odd words / 8) -> K = words / 16, a
        // half-integer (exllamav3 1.6's trellis_words; mul1 only)
        const int64_t words = tr->shape[2];
        L.k2 = words % 8 == 0 ? (int)(words / 8) : 0;
        if (!exl3_k2_valid(L.k2))
            throw std::runtime_error("exl3: " + base + ".trellis has " + std::to_string(words) + " words a tile: not a "
                                     "bitrate exllamav3 1.6 writes (16 K words for K 1..8, or 24 / 40 / 56 for K 1.5 / "
                                     "2.5 / 3.5)");
        if (L.half() && L.cb != 2)
            throw std::runtime_error("exl3: " + base + ": K " + exl3_k_label(L.k2) + " without the mul1 codebook (a "
                                     "half-integer K is mul1 only)");
        if (suh->dtype != "F16" || svh->dtype != "F16" || suh->nelem() < (size_t)L.in() || svh->nelem() < (size_t)L.out())
            throw std::runtime_error("exl3: " + base + ": suh / svh are " + suh->dtype + " / " + svh->dtype + " with " +
                                     std::to_string(suh->nelem()) + " / " + std::to_string(svh->nelem()) + " values "
                                     "for a " + std::to_string(L.in()) + " x " + std::to_string(L.out()) + " matrix "
                                     "(F16, at least that many expected)");
        return L;
    }

private:
    const SafetensorsFile& shard(const std::string& name) const {
        auto it = shards_.find(name);
        if (it != shards_.end()) return *it->second;
        auto f = std::make_unique<SafetensorsFile>(dir_ + "/" + name);
        auto* raw = f.get();
        shards_.emplace(name, std::move(f));
        return *raw;
    }
    const StTensor* tensor(const std::string& name) const {
        auto it = weight_map_.find(name);
        if (it == weight_map_.end()) throw std::runtime_error("exl3: no tensor " + name);
        const StTensor* t = shard(it->second).find(name);
        if (!t) throw std::runtime_error("exl3: tensor " + name + " not in " + it->second);
        return t;
    }
    const uint8_t* data(const StTensor& t) const {
        auto it = weight_map_.find(t.name);
        return shard(it->second).data(t);
    }

    std::string dir_;
    std::map<std::string, std::string> weight_map_;
    mutable std::map<std::string, std::unique_ptr<SafetensorsFile>> shards_;
};

}  // namespace strata
