# EXL3 with a CPU expert lane, VRAM tiers, MTP and `--serve`

> **Status: experimental, `--exl3-lane`.** What lets a PC whose RAM holds an EXL3 Qwen3.8-Flash-Next's experts run it
> at about the speed of the GGUF packs. It was built and measured on one Linux machine with NVIDIA cards (below).
> Nothing here is tested on AMD/HIP, Windows or macOS. The format itself is described in [EXL3.md](EXL3.md).

## What it adds to the EXL3 path

The EXL3 path ([EXL3.md](EXL3.md)) keeps the routed experts on the GPU: a per-layer VRAM cache, misses read from the SSD.
That is the right design for a machine whose RAM cannot hold the experts. This path is for the machine whose RAM
can (56.7 GiB of routed experts at 4.05 bpw):

| Piece | What it does | Option |
| --- | --- | --- |
| **RAM tier** | every routed expert copied once from the safetensors into one block of RAM at start (huge pages asked for; each trellis 4-byte aligned) | `--exl3-lane` |
| **CPU expert lane** | project-maya's EXL3 CPU kernel (AVX2 + FMA and AVX-512BW tiers, after exllamav3's `moe_mul1.cpp`) runs the experts the GPU does not hold, on Strata's expert pool | `--exl3-lane` |
| **Verify windows + MTP** | the lane takes all of a window's tokens at once; Strata's MTP drafter (`--mtp`) drafts for the EXL3 model, its head reading the EXL3 pack's output weight | `--spec`, `--mtp` |
| **VRAM tier, helper tier** | the routing profile's hottest experts in the main card's free VRAM, the next hottest on a second card; the windows run them in their graph on grouped device kernels (project-maya's) while the lane computes the rest | `--exl3-lane-profile`, `--exl3-vram-tier`, `--exl3-helper-tier` |
| **Adaptive tier** | each verify round, the experts the lane computed most replace the tier's least used | `--exl3-tier-adapt` (32) |
| **GPU prompt path** | the batched prompt reader runs the trellis experts: tensor-core kernels for experts with few rows, a decoded FP16 weight and cuBLAS above, each expert streamed once per chunk from the page-locked RAM tier | `--prefill` |
| **`--serve`** | the tiers in the server; each batched prompt part borrows the tier's last slots and refills them after, reads the tier's other experts in place, and up to 8,192 tokens has the helper card compute its tier's experts | `--serve` |
| **Half-integer bitrates** | K 1.5 / 2.5 / 3.5 (exllamav3 1.6, e.g. self-calibrated quants) on every path: loader, CPU reference, lane, device kernels, RAM tier | - |

## What it needs

| | |
| --- | --- |
| RAM | the whole routed-expert set plus the dense part and the OS: 56.7 GiB of experts at 4.05 bpw (measured on 125 GiB) |
| CPU | x86-64 with AVX2 + FMA (AVX-512BW is used when present) |
| GPU | an NVIDIA card with CUDA 12; the tier takes what is free after the model (an RTX 3090 here: 6,400-6,800 experts at a 32K context). A second card is optional: the helper tier |
| The model | an EXL3 Qwen3.8-Flash-Next, e.g. [turboderp's](https://huggingface.co/turboderp/Qwen3.8-Flash-Next-exl3) `4.05bpw_h6_ng6` or [Vemloir's SC](https://huggingface.co/Vemloir/Qwen3.8-Flash-Next-sc-exl3) `4.05bpw_sc_h8_ng8` (the model shards; the n-gram table can be skipped) |
| The n-gram table | a GGUF that holds `per_layer_token_embd`, passed with `--ple-gguf` (a Flash-Next GGUF shard, or a BF16 table). The EXL3 table reader (`--ple-exl3`) expects `shard_N.trellis` tensors, which the 4.05bpw_h6_ng6 table does not have |
| The drafter | Strata's MTP runtime directory, made as for the GGUF models (`tools/mtp_fetch.py`, `tools/mtp_pack.py`, `tools/mtp_rt.py`; see [ORCA.md](ORCA.md)) |
| The routing profile | `data/expert-profile.bin` (shipped) |
| The tokenizer | the model's own, reshaped for the server: `python3 tools/exl3/tokenizer_to_pack.py --hf <model dir> --out packs/exl3` |

## Build

The measured build (Ninja, Release; the GGUF options are the daily engine's and do not matter for EXL3):

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 -DGGML_NATIVE=ON \
  -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_MMQ_KQUANTS=ON -DSTRATA_Q6K_EXPERTS=ON -DSTRATA_PREFILL_MMQ=OFF \
  -DSTRATA_BUILD_TESTS=ON
cmake --build build -j
cd build && ctest -R exl3      # 30 tests; the fixture generators use .venv/bin/python (numpy, safetensors)
```

Set `CMAKE_CUDA_ARCHITECTURES` for your card. Without `-DSTRATA_GGML_DIR` CMake fetches the pinned llama.cpp commit
(`third_party/ggml/VERSION.txt`).

## Run

The helper tier goes on the run's device 1, so list the main card first: `CUDA_VISIBLE_DEVICES=<main>,<helper>`
(with `CUDA_DEVICE_ORDER=PCI_BUS_ID` the numbers are nvidia-smi's).

**The CLI** takes token ids (`--tokens-file`, from the tokenizer):

```sh
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1,2 build/strata \
  --exl3 /models/Qwen3.8-Flash-Next-exl3-4.05bpw_h6_ng6 --ple-gguf /models/Qwen3.8-Flash-Next-PLE-BF16.gguf \
  --exl3-lane --pool-workers 9 --exl3-lane-profile data/expert-profile.bin --exl3-helper-tier auto \
  --prefill auto:32768 --max-context 32768 --kv int8 --spec 4 --spec-min-p 0.70 --mtp mtp/rt \
  --tokens-file prompt.tokens --max-new 256 --stats
```

**The server** (OpenAI-compatible, the same `serve/server.py` as the GGUF models), as `strata-exl3.json`:

```json
{
  "exe": "build/strata",
  "args": [
    "--exl3", "/models/Qwen3.8-Flash-Next-exl3-4.05bpw_h6_ng6",
    "--ple-gguf", "/models/Qwen3.8-Flash-Next-PLE-BF16.gguf",
    "--exl3-lane", "--pool-workers", "9",
    "--exl3-lane-profile", "data/expert-profile.bin", "--exl3-helper-tier", "auto",
    "--prefill", "auto:32768", "--max-context", "32768", "--kv", "int8",
    "--spec", "4", "--spec-min-p", "0.70", "--mtp", "mtp/rt",
    "--prompt-cache", "24", "--conversation-cache-mib", "8192", "--conversation-cache-slots", "4"
  ],
  "cwd": ".",
  "tokenizer": "packs/exl3/tokenizer",
  "model_name": "qwen3.8-flash-next-exl3-4.05",
  "env": { "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": "1,2" },
  "host": "127.0.0.1",
  "port": 8080
}
```

```sh
.venv/bin/python -m serve.server --engine strata --config strata-exl3.json --port 8080
```

Pictures are not configured: the EXL3 path was not tested with them.

| Option | Default | What it does |
| --- | --- | --- |
| `--exl3-lane` | off | the RAM tier and the CPU lane; without it the experts take the GPU stream |
| `--pool-workers N` | | the lane's threads (9 of 12 cores measured best here) |
| `--exl3-lane-profile P` | none | the ranking that fills the VRAM tier and the helper tier |
| `--exl3-vram-tier MIB\|auto` | auto | the main card's tier: what is free after the model, minus `STRATA_EXL3_TIER_RESERVE_MIB` (512; 1,664 in `--serve`) |
| `--exl3-helper-tier MIB\|auto` | 0 | the second card's tier: its free VRAM minus `STRATA_EXL3_HELPER_RESERVE_MIB` (512) |
| `--exl3-tier-adapt N` | 32 | swaps a verify round; 0 keeps the tier static (an exact comparison against single-token decode needs that) |
| `--prefill auto:N` | | the GPU prompt path, chunks of up to N tokens |

Switches back to the older paths, each measured slower or kept for checks: `STRATA_EXL3_NATIVE=0` (the first EXL3 dense
kernels), `STRATA_EXL3_WINDOW_PROMPT=0` (the prompt token by token), `STRATA_EXL3_PF_INPLACE=0` (every expert streamed
in `--serve`), `STRATA_EXL3_PF_HELP=0` (no helper card in a served prompt), `STRATA_EXL3_ADAPT_HELPER=1` (the helper's
tier adapts too), `STRATA_EXL3_OLD_SHARE=1` (the GPU stream as the GPU share).

## Measured

Ryzen 9 5900X (12 cores, AVX2 + FMA, no AVX-512), 125 GiB DDR4, an RTX 3090 24 GB on PCIe 4.0 x8 as the main card and
an RTX 3060 12 GB as the helper (no peer access between them), CUDA 12.8, Linux. Greedy, warm, each run twice.

**A 21K prompt, then 256 tokens** (the CLI line above):

| Model and engine | prompt tok/s | decode tok/s |
| --- | ---: | ---: |
| GGUF GSQ IQ3_S, Strata 0.1.44.1 (the same PC, for comparison) | ~1,635 | 94 |
| **EXL3 4.05 (turboderp)** | **2,381** | **89.1** |
| EXL3 SC 4.05 (Vemloir) | 2,375 | 83.9 (on turboderp's text) / 91.1 (on its own) |

- Decode depends on the text through the drafter: per verified window token the two EXL3 models cost the same
  (9.41 vs 9.38 ms). A short English prompt decodes at 94.9 tok/s, a Serbian paragraph at 74.6 (the drafter accepts
  fewer drafts there, and the shipped profile fits it less).
- `--max-context 131072` leaves fewer experts in the tier: 84.9 tok/s instead of 89.
- How it got there, on the same machine: the lane alone 21.2 tok/s (the GPU stream: 10.8, warming from a cold
  cache); + MTP 28.4; + the VRAM tier 52.6; + the helper tier 76.3; + the adaptive tier 88-89.

**`--serve`**, warm, the server's defaults, one batched prompt part each:

| part (tokens) | 512 | 1,024 | 2,048 | 4,096 | 7,030 | 21,025 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| time to read it | 3.02 s | 3.10 s | 3.26 s | 3.60 s | 4.09 s | 9.13 s |

Parts of up to 320 tokens go through the verify windows instead (a 59-token message: 0.46 s). Decode after the 21K
part: ~100 tok/s.

**The CPU lane per expert** (Flash shapes, 9 threads, from RAM, microseconds an expert): K4 57 at one token, 61 at
two, 155 at eight; a UD-Q4 GGUF expert pair (Q4_K/Q5_1) 69 / 72 / 171. At one or two tokens both stream 43-44 GB/s:
the RAM is the limit, not the trellis decode.

**Exactness**

- Every EXL3 decoder here - the CPU reference, the lane (scalar, AVX2) and the device kernels - gives exllamav3 1.6.2's
  bits on its fixtures (K 1.5 / 2.5 / 3.5 / 3 / 5) and on 21 real tensors of the SC model; the lane equals the EXL3
  reference weight by weight (0 of 393,216 per K and tier).
- The verify windows with fixed tiers (`--exl3-tier-adapt 0`) give the single-token text token for token
  (`--spec-oracle`: 192 of 192 drafts accepted).
- Teacher-forced against the GPU stream path: the lane's KL 0.0008 (same top token 100 %); the GPU prompt path against
  the token path: KL 0.0008 (same top token 98.7 %).
- In `--serve`, the helper card's cuBLAS products are not the main card's bits (cuBLAS picks its kernels per card),
  so a served answer is repeatable but not bit-identical to the CLI's; against the token path its KL stays as low as
  without the helper (mean 0.00055 vs 0.0043 on the check's text). `STRATA_EXL3_PF_HELP=0` gives the exact path.
- The GGUF path is unchanged: the daily's config gives identical log-probabilities at 6,120 positions with this
  build and with Strata 0.1.44.1 (with fixed cache sizes; a few MB of free VRAM more or less move an auto-sized cache).

## Not done, not tested

- AMD/HIP, Windows, macOS: not built. The lane is portable C++; the tiers and the device kernels are CUDA paths.
- Pictures (GENI), `--kv-resident`, a layer split: not tested with EXL3.
- AVX-512: that tier is bit-identical to AVX2 by construction, but this CPU has none, so it was not run; the
  half-integer layers run the AVX2 kernels on an AVX-512 CPU.
- The checkpoint's own EXL3 MTP layer is not used (Strata's native drafter is), nor its EXL3 n-gram table (above).
- Without the RAM for the whole expert set, leave out `--exl3-lane`: the stream path is for that machine.

## Credits and license

- **The EXL3 path ([#1764](https://github.com/Niko1221/Strata/pull/1764)) by [@Yxmura](https://github.com/Yxmura):** what
  this builds on - the loader, the CPU reference, the GPU kernels, the n-gram codec and the engine integration.
- **[project-maya](https://github.com/mw00/project-maya)** (MIT License, Copyright (c) 2026 Peasant Smith and the
  Project Maya contributors): the EXL3 CPU expert kernel (by @needmorevram; its AVX2 + FMA tier by @sociolog, merged
  through project-maya #188) and the EXL3 device kernels of the batched paths (by @needmorevram), ported here as
  `src/kernels/cpu/exl3_lane.cpp` and `src/kernels/cuda/exl3_dev.cu` ([license](../third_party/project-maya/LICENSE)).
- **[exllamav3](https://github.com/turboderp-org/exllamav3)** (MIT License, Copyright (c) 2025 Turboderp): the EXL3
  format; the CPU kernel follows its `moe_mul1.cpp`, the half-integer decode its `dq8_half`, and the half-integer test
  fixtures were made with exllamav3 1.6.2 ([license](../third_party/exllamav3/LICENSE)).
- **Strata** (MIT License, Copyright (c) 2026 Niko1221 and the Strata contributors).
- The models: [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) by the Qwen team, in EXL3 by
  turboderp and Vemloir.
