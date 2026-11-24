"""Held-out gate for expert re-rounding: the MoE block's routed output against BF16, on tokens of tasks the calibration
never saw (capture phase `gate`).

Per layer, for the captured tokens x_t with served routing (ids -> checkpoint experts through the keep list) and
weights a_t: y_t = sum_k a_tk * down_e(silu(gate_e x_t) * up_e x_t). The reference uses the BF16 experts; each build
(RadixArk, QS, QG, ...) uses its own NVFP4 shards. Reports per layer err = sum_t ||y_t - y_ref,t||^2 / sum_t ||y_ref,t||^2
and the mean over layers, fp32 throughout (TF32 off).

  python -m agi2_modal_harness.comm.nvfp4_blockgate --cap <run dir> --keep <keep.json> \
      --build radix=/vol/models/RadixArk__... --build qs=<dir> --build qg=<dir> [--layers 0-47] [--max-tokens 20000]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time

import torch
import torch.nn.functional as F
from safetensors import safe_open

from agi2_modal_harness.comm.nvfp4_reround import BF16, decode_e2m1, parse_layers, unpack


def gate_rows(plan_path, exclude_prefix, max_tokens):
    """Row indices into the concatenated gate capture: every held-out history contributes, its first
    `exclude_prefix` positions (the harness system prompt every history shares with the calibration) are dropped,
    and the rest is subsampled evenly to max_tokens. The capture prefilled the gate histories one at a time in plan
    order with no cache hits, so the rows are the histories' tokens back to back."""
    plan = json.load(open(plan_path))
    hs = plan["phases"]["gate"]["histories"]
    sizes = [h["tokens"] for h in hs]
    keep, hist, start = [], [], 0
    for i, n in enumerate(sizes):
        keep.extend(range(start + exclude_prefix, start + n))
        hist.extend([i] * (n - exclude_prefix))
        start += n
    keep, hist = torch.tensor(keep), torch.tensor(hist)
    if len(keep) > max_tokens:
        pick = torch.linspace(0, len(keep) - 1, max_tokens).round().long()
        keep, hist = keep[pick], hist[pick]
    return keep, sum(sizes), hist, [h["task"] for h in hs]


def load_gate(cap, layer, dev, max_tokens, rows=None, total=None):
    files = sorted(glob.glob(os.path.join(cap, "gate", f"L{layer:02d}", "*.pt")))
    parts = [torch.load(f, map_location="cpu") for f in files]
    x = torch.cat([p["x"] for p in parts])
    ids = torch.cat([p["ids"] for p in parts])
    w = torch.cat([p["w"] for p in parts])
    if rows is not None:
        if x.shape[0] != total:
            raise SystemExit(f"layer {layer}: gate capture has {x.shape[0]} rows, plan says {total}")
        x, ids, w = x[rows], ids[rows], w[rows]
    else:
        x, ids, w = x[:max_tokens], ids[:max_tokens], w[:max_tokens]
    return x.to(dev).float(), ids.to(dev).long(), w.to(dev).float()


def nvfp4_weights(build_dir, layer, experts, mat, dev):
    out = {}
    for lo in range(0, 512, 128):
        es = [e for e in experts if lo <= e < lo + 128]
        if not es:
            continue
        name = os.path.join(build_dir, f"layer-{layer:05d}-experts-{lo:04d}-{lo + 127:04d}.safetensors")
        pre = f"model.language_model.layers.{layer}.mlp.experts"
        with safe_open(name, "pt") as f:
            for e in es:
                b = f.get_tensor(f"{pre}.{e}.{mat}.weight").to(dev)
                s = f.get_tensor(f"{pre}.{e}.{mat}.weight_scale").to(dev).to(torch.float32)
                g = f.get_tensor(f"{pre}.{e}.{mat}.weight_scale_2").to(dev, torch.float32).reshape(())
                codes = unpack(b)
                R, C = codes.shape
                out[e] = (decode_e2m1(codes).view(R, C // 16, 16) * s.unsqueeze(-1) * g).view(R, C)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", required=True)
    ap.add_argument("--keep", required=True)
    ap.add_argument("--build", action="append", required=True, help="name=dir of NVFP4 expert shards")
    ap.add_argument("--layers", default="0-47")
    ap.add_argument("--bf16", default=BF16)
    ap.add_argument("--max-tokens", type=int, default=20000)
    ap.add_argument("--plan", default="", help="capture plan json: sample rows from every held-out history")
    ap.add_argument("--exclude-prefix", type=int, default=0, help="drop each history's first N positions")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows, total, hist, tasks = (gate_rows(a.plan, a.exclude_prefix, a.max_tokens) if a.plan
                                else (None, None, None, None))
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = torch.device("cuda")
    keep = json.load(open(a.keep))
    keep = keep["layers"] if isinstance(keep, dict) else keep  # {"layers": [[kept ids], ...]}
    builds = dict(b.split("=", 1) for b in a.build)
    with open(os.path.join(a.bf16, "model.safetensors.index.json")) as f:
        wmap = json.load(f)["weight_map"]
    res = {"builds": list(builds), "layers": {}}
    t0 = time.time()
    layers = parse_layers(a.layers)
    for li, layer in enumerate(layers):
        x, sid, sw = load_gate(a.cap, layer, dev, a.max_tokens, rows, total)
        orig = torch.tensor(keep[layer], device=dev)[sid]
        experts = sorted(set(orig.flatten().tolist()))
        pre = f"model.language_model.layers.{layer}.mlp.experts"
        with safe_open(os.path.join(a.bf16, wmap[f"{pre}.gate_up_proj"]), "pt") as f:
            gu = f.get_tensor(f"{pre}.gate_up_proj")
        with safe_open(os.path.join(a.bf16, wmap[f"{pre}.down_proj"]), "pt") as f:
            dn = f.get_tensor(f"{pre}.down_proj")
        half = gu.shape[1] // 2
        ref_w = {e: (gu[e, :half].to(dev, torch.float32), gu[e, half:].to(dev, torch.float32),
                     dn[e].to(dev, torch.float32)) for e in experts}
        bw = {}
        for name, d in builds.items():
            g_ = nvfp4_weights(d, layer, experts, "gate_proj", dev)
            u_ = nvfp4_weights(d, layer, experts, "up_proj", dev)
            d_ = nvfp4_weights(d, layer, experts, "down_proj", dev)
            bw[name] = {e: (g_[e], u_[e], d_[e]) for e in experts}
        y = {n: torch.zeros(x.shape[0], x.shape[1], device=dev) for n in ["ref", *builds]}
        for e in experts:
            hit = orig == e
            sel = hit.any(-1).nonzero().flatten()
            ae = (sw * hit.float()).sum(-1)[sel].unsqueeze(-1)
            xe = x[sel]
            for n, (G, U, D) in [("ref", ref_w[e])] + [(n, bw[n][e]) for n in builds]:
                y[n][sel] += ae * ((F.silu(xe @ G.T) * (xe @ U.T)) @ D.T)
        den = (y["ref"] ** 2).sum().item()
        res["layers"][layer] = {n: ((y[n] - y["ref"]) ** 2).sum().item() / den for n in builds}
        res["layers"][layer]["tokens"] = int(x.shape[0])
        if hist is not None:  # the same error per held-out task, so one task cannot carry the pooled number
            hd = hist.to(dev)
            per = {}
            for i, t in enumerate(tasks):
                m = hd == i
                d_ = (y["ref"][m] ** 2).sum().item()
                per[t] = {n: ((y[n][m] - y["ref"][m]) ** 2).sum().item() / d_ for n in builds}
            res["layers"][layer]["per_task"] = per
        line = " ".join(f"{n} {res['layers'][layer][n]:.5f}" for n in builds)
        print(f"[gate] layer {layer} ({li + 1}/{len(layers)}): {line}; {x.shape[0]} tokens; "
              f"elapsed {(time.time() - t0) / 60:.1f} min", flush=True)
        del gu, dn, ref_w, bw, y
        torch.cuda.empty_cache()
    n = len(res["layers"])
    res["mean"] = {b: sum(v[b] for v in res["layers"].values()) / n for b in builds}
    if "radix" in builds:
        res["reduction_vs_radix"] = {b: 1 - res["mean"][b] / res["mean"]["radix"] for b in builds if b != "radix"}
        res["layers_better_than_radix"] = {b: sum(v[b] < v["radix"] for v in res["layers"].values())
                                           for b in builds if b != "radix"}
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"[PHASE] gate done: mean {res['mean']}; vs radix {res.get('reduction_vs_radix')}; "
          f"layers better {res.get('layers_better_than_radix')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
