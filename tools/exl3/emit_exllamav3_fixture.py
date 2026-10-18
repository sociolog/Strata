"""Emit EXL3 decode fixtures from exllamav3 itself (1.6.2 or later: the half-integer bitrates), for exl3_parity --halfk.

    <exllamav3 venv>/bin/python -I tools/exl3/emit_exllamav3_fixture.py --out tests/data/exl3_halfk [--k2 3 5 7 6 10]
    <exllamav3 venv>/bin/python -I tools/exl3/emit_exllamav3_fixture.py --model <EXL3 dir> --tensor <base> ... --out <dir>

Needs torch and the exllamav3_ext extension with a CUDA device (CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<n>
picks it).  The bitrate is k2 = 2 K, exllamav3's own unit (bits_k.cuh): odd k2 = the half-integer K 1.5 / 2.5 / 3.5.

Random tiles (default): 2 x 8 random 16x16 tiles a bitrate (reconstruct takes n a multiple of 128) - random 16-bit states packed by exllamav3's own packer
(pack_trellis_frac with KA = k2 / 2 and MASK 0xAAAA for a half-integer K, pack_trellis for an integer one) - then
exllamav3's unpacker (the 256 windows of each tile, ring order) and its reconstruct (W_hat, the rotated basis, mul1).
Writes <out>/exl3_k2_<k2>.bin:
    "EXH1", int32 k2, ki, nj, cb (2), words (16 K uint16 a tile)
    int16  trellis [ki][nj][words]          what pack_trellis(_frac) wrote
    uint16 states  [ki][nj][256]            unpack_trellis(_frac): the window of each ring position
    fp16   w_hat   [ki * 16][nj * 16]       reconstruct(..., K, mcg=False, mul1=True), row-major

Model tensors (--model): each <base>'s W_hat by exllamav3's reconstruct and its original-basis W by reconstruct_had_slice,
as raw little-endian fp16 files <out>/<base>.what.f16 and <out>/<base>.w.f16, plus <out>/manifest.txt
(base K k n fnv1a64(w_hat) fnv1a64(w)) - for strata-exl3-model --expect.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import struct
import sys

import numpy as np
import torch                      # before the extension: it links against torch's libraries
import exllamav3_ext as ext       # noqa: E402

MAGIC = b"EXH1"
MASK = 0xAAAA                     # the one instantiated half-integer pattern: odd positions carry the extra bit


def fnv1a(b: bytes) -> int:
    a = np.frombuffer(b, dtype=np.uint8)
    h = 1469598103934665603
    for x in a.tolist():
        h ^= x
        h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h


def pack(idx: torch.Tensor, k2: int) -> torch.Tensor:
    ki, nj, _ = idx.shape
    packed = torch.zeros((ki, nj, 8 * k2), dtype=torch.int16, device=idx.device)
    if k2 % 2:
        ext.pack_trellis_frac(packed, idx.contiguous(), k2 // 2, MASK)
    else:
        ext.pack_trellis(packed, idx.contiguous(), k2 // 2)
    return packed


def unpack(packed: torch.Tensor, k2: int) -> torch.Tensor:
    ki, nj, _ = packed.shape
    st = torch.zeros((ki, nj, 256), dtype=torch.int16, device=packed.device)
    if k2 % 2:
        ext.unpack_trellis_frac(st, packed.contiguous(), k2 // 2, MASK)
    else:
        ext.unpack_trellis(st, packed.contiguous(), k2 // 2)
    return st


def what(packed: torch.Tensor, k2: int, mcg: bool = False, mul1: bool = True) -> torch.Tensor:
    ki, nj, _ = packed.shape
    w = torch.zeros((ki * 16, nj * 16), dtype=torch.half, device=packed.device)
    ext.reconstruct(w, packed.contiguous(), k2 / 2, mcg, mul1)
    return w


def random_tiles(out: pathlib.Path, k2s: list[int], seed: int, ki: int, nj: int) -> None:
    dev = torch.device("cuda")
    for k2 in k2s:
        if not (2 <= k2 <= 16 and (k2 % 2 == 0 or k2 in (3, 5, 7))):
            raise SystemExit("k2 %d: not a bitrate exllamav3 decodes (2..16 even, or 3 / 5 / 7)" % k2)
        if nj % 8:
            raise SystemExit("--nj %d: exllamav3's reconstruct takes n a multiple of 128 (nj a multiple of 8)" % nj)
        rng = np.random.default_rng(seed + 1000 * k2)
        idx = torch.from_numpy(rng.integers(-(1 << 15), 1 << 15, size=(ki, nj, 256), dtype=np.int16)).to(dev)
        packed = pack(idx, k2)
        states = unpack(packed, k2)
        # exllamav3's own consistency: the windows pack back to the same words
        if not torch.equal(pack(states, k2), packed):
            raise SystemExit("k2 %d: exllamav3's pack(unpack(packed)) != packed" % k2)
        w_hat = what(packed, k2)
        torch.cuda.synchronize()
        path = out / ("exl3_k2_%d.bin" % k2)
        with open(path, "wb") as f:
            f.write(MAGIC)
            f.write(struct.pack("<iiiii", k2, ki, nj, 2, 8 * k2))
            f.write(packed.cpu().numpy().astype("<i2").tobytes())
            f.write(states.cpu().numpy().astype("<i2").tobytes())
            f.write(w_hat.cpu().numpy().astype("<f2").tobytes())
        print("wrote %s (%d bytes): K %g, %d x %d tiles, %d words a tile, exllamav3 %s" %
              (path, path.stat().st_size, k2 / 2, ki, nj, 8 * k2, exllamav3_version()))


def exllamav3_version() -> str:
    try:
        from importlib.metadata import version
        return version("exllamav3")
    except Exception:  # noqa: BLE001 - a label only
        return "?"


def model_tensors(out: pathlib.Path, model: pathlib.Path, bases: list[str]) -> None:
    from safetensors import safe_open
    index = json.loads((model / "model.safetensors.index.json").read_text())["weight_map"]
    dev = torch.device("cuda")
    lines = []
    for base in bases:
        shard = model / index[base + ".trellis"]
        with safe_open(str(shard), framework="pt", device="cpu") as f:
            keys = set(f.keys())
            tr = f.get_tensor(base + ".trellis")
            suh = f.get_tensor(base + ".suh")
            svh = f.get_tensor(base + ".svh")
        mul1, mcg = (base + ".mul1") in keys or (base + ".mul1") in index, (base + ".mcg") in keys or (base + ".mcg") in index
        words = tr.shape[2]
        k2 = words // 8
        if words % 8 or not (k2 % 2 == 0 or k2 in (3, 5, 7)):
            raise SystemExit("%s: %d words a tile is not an EXL3 bitrate" % (base, words))
        trd = tr.to(dev).contiguous()
        w_hat = what(trd, k2, mcg, mul1)
        k, n = w_hat.shape
        w = torch.zeros((k, n), dtype=torch.half, device=dev)
        ext.reconstruct_had_slice(w, trd, suh.to(dev).contiguous(), svh.to(dev).contiguous(), k2 / 2, mcg, mul1, 0)
        torch.cuda.synchronize()
        bw = w_hat.cpu().numpy().astype("<f2").tobytes()
        bf = w.cpu().numpy().astype("<f2").tobytes()
        (out / (base + ".what.f16")).write_bytes(bw)
        (out / (base + ".w.f16")).write_bytes(bf)
        lines.append("%s %g %d %d %016x %016x" % (base, k2 / 2, k, n, fnv1a(bw), fnv1a(bf)))
        print(lines[-1])
    with open(out / "manifest.txt", "a") as f:
        for ln in lines:
            f.write(ln + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--k2", type=int, nargs="*", default=[3, 5, 7, 6, 10])
    ap.add_argument("--seed", type=int, default=20261018)
    ap.add_argument("--ki", type=int, default=2)
    ap.add_argument("--nj", type=int, default=8)
    ap.add_argument("--model")
    ap.add_argument("--tensor", nargs="*", default=[])
    args = ap.parse_args()
    if not torch.cuda.is_available():
        print("no CUDA device: exllamav3's packers and reconstruct are CUDA kernels", file=sys.stderr)
        return 2
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.model:
        model_tensors(out, pathlib.Path(args.model), args.tensor)
    else:
        random_tiles(out, args.k2, args.seed, args.ki, args.nj)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
