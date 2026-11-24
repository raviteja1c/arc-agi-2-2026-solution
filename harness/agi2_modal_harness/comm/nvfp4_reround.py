"""Re-round RadixArk's NVFP4 routed experts of Qwen3.8-Flash-Next at the SAME format (E2M1 codes, one E4M3 block scale
per 16 inputs, RadixArk's FP32 global scale per tensor), from the BF16 originals.

Mode `qs` (this file): for every 16-weight block, the exact weight-MSE-optimal E4M3 scale among the 25 legal values
from 1/4x to 2x RadixArk's own (max/6) scale, codes re-rounded to nearest at each candidate. Mode `qg` (GPTQ on our
harness activations) plugs in through `quantize_block_rows` with a Hessian; see nvfp4_gptq.py.

Fail-closed, per layer and shard:
  1. preflight: max-RTN from our BF16 must reproduce >= 98 % of RadixArk's codes (same source weights and layout);
  2. post-write: decode(our packed bytes, our scales) * global == the values we chose, exactly;
  3. per matrix: our weight error <= RadixArk's (+1e-9).
Outputs: <out>/<same shard names>.safetensors (RadixArk's tensor names, dtypes and metadata; only `weight` and
`weight_scale` change) and <out>/manifest/<shard>.json with per-matrix errors. Progress lines with ETA.

Run on a dev box:
  python -m agi2_modal_harness.comm.nvfp4_reround --mode qs --out /vol/cache/dev/fnxq/qs1 [--layers 0-47]
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch
from safetensors import safe_open
from safetensors.torch import save_file

RADIX = "/vol/models/RadixArk__Qwen3.8-Flash-Next-NVFP4__7b719225242a"
BF16 = "/vol/cache/dev/fnbf16/Qwen__Qwen3.8-Flash-Next__de4b8e4d43b9"
E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
E2M1_MID = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
LO_STEPS, HI_STEPS = 16, 8  # E4M3 has 8 values per octave: 1/4x .. 2x


def e4m3_positive_table() -> torch.Tensor:
    codes = torch.arange(1, 127, dtype=torch.uint8)  # 0x01..0x7E: every positive finite E4M3
    return codes.view(torch.float8_e4m3fn).to(torch.float32).sort().values


E4M3_POS = e4m3_positive_table()


def round_e2m1(x: torch.Tensor):
    """Nearest E2M1 value and its 4-bit code (sign in bit 3). Ties go to the lower magnitude."""
    mag_idx = torch.bucketize(x.abs(), E2M1_MID.to(x.device), right=False)
    val = E2M1.to(x.device)[mag_idx] * torch.sign(x)
    code = mag_idx.to(torch.uint8) | ((x < 0) & (mag_idx > 0)).to(torch.uint8) << 3
    return val, code


def decode_e2m1(code: torch.Tensor) -> torch.Tensor:
    mag = E2M1.to(code.device)[(code & 7).long()]
    return torch.where((code & 8) != 0, -mag, mag)


def pack(code: torch.Tensor) -> torch.Tensor:
    """[..., n] 4-bit codes -> [..., n/2] bytes, low nibble = even column (RadixArk's order, measured)."""
    return (code[..., 0::2] | (code[..., 1::2] << 4)).contiguous()


def unpack(b: torch.Tensor) -> torch.Tensor:
    out = torch.empty(*b.shape[:-1], b.shape[-1] * 2, dtype=torch.uint8, device=b.device)
    out[..., 0::2], out[..., 1::2] = b & 0xF, b >> 4
    return out


def search_scales(w: torch.Tensor, g: torch.Tensor, base: torch.Tensor):
    """w [E, R, C] fp32, g [E] global scales, base [E, R, C/16] RadixArk's E4M3 scale values.
    Returns (best scale values [E, R, C/16], codes [E, R, C] uint8)."""
    E, R, C = w.shape
    wb = w.view(E, R, C // 16, 16)
    table = E4M3_POS.to(w.device)
    base_idx = torch.bucketize(base.contiguous(), table).clamp(0, len(table) - 1)
    gg = g.view(E, 1, 1, 1)
    best_err = torch.full(base.shape, float("inf"), device=w.device)
    best_idx = base_idx.clone()
    for k in range(-LO_STEPS, HI_STEPS + 1):
        idx = (base_idx + k).clamp(0, len(table) - 1)
        s = table[idx].unsqueeze(-1) * gg
        val, _ = round_e2m1(wb / s)
        err = ((wb - val * s) ** 2).sum(-1)
        # strict improvement only, visiting k = 0 first would make ties keep RadixArk's; emulate by tolerance
        better = err < best_err - (1e-12 if k != 0 else -1e-12)
        best_err = torch.where(better, err, best_err)
        best_idx = torch.where(better, idx, best_idx)
    s_best = table[best_idx]
    _, codes = round_e2m1(wb / (s_best.unsqueeze(-1) * gg))
    return s_best, codes.view(E, R, C)


def rtn_codes(w: torch.Tensor, g: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
    E, R, C = w.shape
    s = base.unsqueeze(-1) * g.view(E, 1, 1, 1)
    _, codes = round_e2m1(w.view(E, R, C // 16, 16) / s)
    return codes.view(E, R, C)


def rel_err(w: torch.Tensor, deq: torch.Tensor) -> torch.Tensor:
    return ((w - deq) ** 2).sum((1, 2)) / (w ** 2).sum((1, 2))


def dequant(codes: torch.Tensor, scales: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    E, R, C = codes.shape
    return (decode_e2m1(codes).view(E, R, C // 16, 16) * scales.unsqueeze(-1) * g.view(E, 1, 1, 1)).view(E, R, C)


def parse_layers(spec: str):
    out = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out.extend(range(int(a), int(b or a) + 1))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["qs"], default="qs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", default="0-47")
    ap.add_argument("--radix", default=RADIX)
    ap.add_argument("--bf16", default=BF16)
    a = ap.parse_args()
    dev = torch.device("cuda")
    os.makedirs(os.path.join(a.out, "manifest"), exist_ok=True)
    with open(os.path.join(a.bf16, "model.safetensors.index.json")) as f:
        wmap = json.load(f)["weight_map"]
    layers = parse_layers(a.layers)
    t0 = time.time()
    totals = {"radix": [], "new": []}
    for li, layer in enumerate(layers):
        tl = time.time()
        pre = f"model.language_model.layers.{layer}.mlp.experts"
        with safe_open(os.path.join(a.bf16, wmap[f"{pre}.gate_up_proj"]), "pt") as f:
            gate_up = f.get_tensor(f"{pre}.gate_up_proj")      # [512, 1280, 2560] bf16
        with safe_open(os.path.join(a.bf16, wmap[f"{pre}.down_proj"]), "pt") as f:
            down = f.get_tensor(f"{pre}.down_proj")            # [512, 2560, 640] bf16
        half = gate_up.shape[1] // 2
        for lo in range(0, 512, 128):
            name = f"layer-{layer:05d}-experts-{lo:04d}-{lo + 127:04d}.safetensors"
            out_path = os.path.join(a.out, name)
            man_path = os.path.join(a.out, "manifest", name + ".json")
            if os.path.exists(man_path):
                continue  # resumable
            src = os.path.join(a.radix, name)
            with safe_open(src, "pt") as f:
                meta = f.metadata()
                tensors = {k: f.get_tensor(k) for k in f.keys()}
            man = {"shard": name, "mode": a.mode, "matrices": {}}
            for mat in ("gate_proj", "up_proj", "down_proj"):
                if mat == "gate_proj":
                    w = gate_up[lo:lo + 128, :half]
                elif mat == "up_proj":
                    w = gate_up[lo:lo + 128, half:]
                else:
                    w = down[lo:lo + 128]
                w = w.to(dev, torch.float32)
                keys = [f"{pre}.{e}.{mat}" for e in range(lo, lo + 128)]
                packed = torch.stack([tensors[k + ".weight"] for k in keys]).to(dev)
                sc = torch.stack([tensors[k + ".weight_scale"] for k in keys]).to(dev).to(torch.float32)
                g = torch.stack([tensors[k + ".weight_scale_2"].reshape(()) for k in keys]).to(dev, torch.float32)
                radix_codes = unpack(packed)
                # 1. preflight: our BF16 reproduces RadixArk's max-RTN codes
                # compare decoded values: RadixArk may store -0 as 0x8 where we store 0x0
                repro = (decode_e2m1(rtn_codes(w, g, sc)) == decode_e2m1(radix_codes)).float().mean().item()
                if repro < 0.98:
                    raise SystemExit(f"PREFLIGHT FAIL {name} {mat}: max-RTN reproduces only {repro:.4f} of codes")
                err_radix = rel_err(w, dequant(radix_codes, sc, g))
                s_new, codes = search_scales(w, g, sc)
                new_packed = pack(codes)
                new_scales_f8 = s_new.to(torch.float8_e4m3fn)
                # 2. post-write exactness: stored scales are exactly the chosen values; bytes decode to the codes
                if not torch.equal(new_scales_f8.to(torch.float32), s_new):
                    raise SystemExit(f"SCALE ENCODE FAIL {name} {mat}")
                if not torch.equal(unpack(new_packed), codes):
                    raise SystemExit(f"PACK FAIL {name} {mat}")
                err_new = rel_err(w, dequant(codes, s_new, g))
                # 3. never worse than RadixArk, per matrix
                worse = (err_new > err_radix + 1e-9).nonzero().flatten().tolist()
                if worse:
                    raise SystemExit(f"WORSE FAIL {name} {mat} experts {[lo + i for i in worse][:8]}")
                for i, k in enumerate(keys):
                    tensors[k + ".weight"] = new_packed[i].cpu()
                    tensors[k + ".weight_scale"] = new_scales_f8[i].cpu()
                man["matrices"][mat] = {
                    "repro_rtn": round(repro, 6),
                    "err_radix_mean": err_radix.mean().item(), "err_new_mean": err_new.mean().item(),
                    "gain_mean": (1 - err_new / err_radix).mean().item(),
                    "gain_min": (1 - err_new / err_radix).min().item(),
                    "scales_changed": (s_new != sc).float().mean().item(),
                }
                totals["radix"].append(err_radix.mean().item())
                totals["new"].append(err_new.mean().item())
            save_file(tensors, out_path + ".tmp", metadata=meta)
            os.replace(out_path + ".tmp", out_path)
            with open(man_path, "w") as f:
                json.dump(man, f)
        el = time.time() - t0
        per = el / (li + 1)
        rad = sum(totals["radix"]) / max(len(totals["radix"]), 1)
        new = sum(totals["new"]) / max(len(totals["new"]), 1)
        print(f"[reround] layer {layer} ({li + 1}/{len(layers)}) in {time.time() - tl:.0f} s; "
              f"mean err radix {rad:.5f} -> new {new:.5f} ({100 * (1 - new / rad) if rad else 0:.1f} % less); "
              f"elapsed {el / 60:.1f} min, ETA {per * (len(layers) - li - 1) / 60:.1f} min", flush=True)
        del gate_up, down
        torch.cuda.empty_cache()
    print(f"[PHASE] reround {a.mode} done in {(time.time() - t0) / 60:.1f} min over {len(layers)} layers", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
