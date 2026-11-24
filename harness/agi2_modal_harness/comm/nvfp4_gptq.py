"""QG: GPTQ re-rounding of RadixArk's NVFP4 routed experts at the SAME format, calibrated on our own harness traffic.

Inputs: BF16 experts (Qwen/Qwen3.8-Flash-Next @de4b8e4d43b9), RadixArk's expert shards (names, global/input scales,
metadata), the served keep list (served expert id j = keep[layer][j]), and the capture of phase `calib`
(<cap>/calib/L<ll>/*.pt: x bf16 [n, 2560], served ids int16 [n, 10], routing weights fp16 [n, 10]).

Per layer and served expert e with rows R_e (tokens routed to e) and routing weights a:
  gate, up  [640 x 2560]: H = sum_t a_t^2 x_t x_t^T (+1 % mean-diag damping); GPTQ column by column with error feedback
            (lazy blocks of 128), each 16-column group's E4M3 scales from the exact scale search (nvfp4_reround) on the
            error-updated weights, RadixArk's global scale kept;
  down [2560 x 640]: z_t = silu(G^ x_t) * (U^ x_t) with the QUANTIZED gate/up; H = sum a_t^2 z_t z_t^T; GPTQ as above.
Experts with fewer than --min-rows rows, and experts outside the keep list (never served), get QS (exact best scales).
Per matrix it records the calib output-error proxy tr(dW H dW^T) for RadixArk, QS and QG; the held-out gate is
nvfp4_blockgate.py. Fail-closed: same packing/encode checks as nvfp4_reround; a layer whose QG proxy is worse than
QS's on the layer total aborts.

  python -m agi2_modal_harness.comm.nvfp4_gptq --cap /vol/cache/dev/fncap/<run> --keep <keep.json> --out <dir>
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
from safetensors.torch import save_file

from agi2_modal_harness.comm.nvfp4_reround import (BF16, E4M3_POS, HI_STEPS, LO_STEPS, RADIX, decode_e2m1, dequant,
                                                   pack, parse_layers, round_e2m1, search_scales, unpack)

BLOCK = 128


def load_capture(cap: str, layer: int, dev):
    files = sorted(glob.glob(os.path.join(cap, "calib", f"L{layer:02d}", "*.pt")))
    if not files:
        raise SystemExit(f"no capture for layer {layer} under {cap}")
    parts = [torch.load(f, map_location="cpu") for f in files]
    x = torch.cat([p["x"] for p in parts]).to(dev)
    ids = torch.cat([p["ids"] for p in parts]).to(dev).long()
    w = torch.cat([p["w"] for p in parts]).to(dev).float()
    return x, ids, w


def best_scales_for_group(wg: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    """wg [E, R, 16] current weights of one group; g [E] global scales -> best E4M3 scale values [E, R]."""
    E, R, _ = wg.shape
    table = E4M3_POS.to(wg.device)
    amax = wg.abs().amax(-1)
    base = (amax / 6.0 / g.view(E, 1)).clamp(min=float(table[0]))
    base_idx = torch.bucketize(base, table).clamp(0, len(table) - 1)
    best_err = torch.full(base.shape, float("inf"), device=wg.device)
    best_idx = base_idx.clone()
    gg = g.view(E, 1, 1)
    for k in range(-LO_STEPS, HI_STEPS + 1):
        idx = (base_idx + k).clamp(0, len(table) - 1)
        s = table[idx].unsqueeze(-1) * gg
        val, _ = round_e2m1(wg / s)
        err = ((wg - val * s) ** 2).sum(-1)
        better = err < best_err
        best_err = torch.where(better, err, best_err)
        best_idx = torch.where(better, idx, best_idx)
    return table[best_idx]


def gptq(W: torch.Tensor, H: torch.Tensor, g: torch.Tensor, damp: float = 0.01):
    """Batched GPTQ under NVFP4 (groups of 16 along columns). W [E, R, C] fp32 (modified copy), H [E, C, C] fp32.
    Returns (codes uint8 [E, R, C], scale values [E, R, C/16])."""
    E, R, C = W.shape
    W = W.clone()
    H = H.clone()
    diag = torch.diagonal(H, dim1=1, dim2=2)
    dead = diag <= 0
    diag[dead] = 1.0
    base = diag.mean(-1, keepdim=True)
    diag += damp * base
    for attempt in range(4):  # not positive-definite at this damping: add 10x more, at most 3 times
        L, info = torch.linalg.cholesky_ex(H)
        if int(info.max()) == 0:
            break
        diag += damp * (10 ** (attempt + 1)) * base
    else:
        raise RuntimeError("Hessian not positive-definite after 3 damping increases")
    Hinv = torch.cholesky_inverse(L)
    U = torch.linalg.cholesky(Hinv, upper=True)          # [E, C, C]
    codes = torch.zeros(E, R, C, dtype=torch.uint8, device=W.device)
    scales = torch.zeros(E, R, C // 16, device=W.device)
    gg = g.view(E, 1)
    for b0 in range(0, C, BLOCK):
        b1 = min(b0 + BLOCK, C)
        W1 = W[:, :, b0:b1].clone()
        Err = torch.zeros_like(W1)
        U1 = U[:, b0:b1, b0:b1]
        for c in range(b1 - b0):
            col = b0 + c
            if col % 16 == 0:
                s_grp = best_scales_for_group(W1[:, :, c:c + 16], g)     # [E, R]
                scales[:, :, col // 16] = s_grp
            s = scales[:, :, col // 16] * gg                               # [E, R]
            val, code = round_e2m1(W1[:, :, c] / s)
            q = val * s
            codes[:, :, col] = code
            d = U1[:, c, c].unsqueeze(-1)                                  # [E, 1]
            err = (W1[:, :, c] - q) / d
            W1[:, :, c:] -= err.unsqueeze(-1) * U1[:, c, c:].unsqueeze(1)
            Err[:, :, c] = err
        W[:, :, b1:] -= torch.bmm(Err, U[:, b0:b1, b1:])
    return codes, scales


def proxy(dW: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
    """tr(dW H dW^T) per expert: dW [E, R, C], H [E, C, C]."""
    return torch.einsum("erc,ecd,erd->e", dW, H, dW)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", required=True)
    ap.add_argument("--keep", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", default="0-47")
    ap.add_argument("--radix", default=RADIX)
    ap.add_argument("--bf16", default=BF16)
    ap.add_argument("--batch", type=int, default=128, help="experts per GPTQ batch (the column loop is launch-bound)")
    ap.add_argument("--min-rows", type=int, default=256)
    ap.add_argument("--damp", type=float, default=0.01)
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = torch.device("cuda")
    os.makedirs(os.path.join(a.out, "manifest"), exist_ok=True)
    keep = json.load(open(a.keep))
    keep = keep["layers"] if isinstance(keep, dict) else keep  # {"layers": [[kept ids], ...]}
    with open(os.path.join(a.bf16, "model.safetensors.index.json")) as f:
        wmap = json.load(f)["weight_map"]
    layers = parse_layers(a.layers)
    t0 = time.time()
    for li, layer in enumerate(layers):
        tl = time.time()
        if all(os.path.exists(os.path.join(a.out, "manifest", f"layer-{layer:05d}-experts-{lo:04d}-{lo + 127:04d}"
                                           ".safetensors.json")) for lo in range(0, 512, 128)):
            continue
        pre = f"model.language_model.layers.{layer}.mlp.experts"
        with safe_open(os.path.join(a.bf16, wmap[f"{pre}.gate_up_proj"]), "pt") as f:
            gate_up = f.get_tensor(f"{pre}.gate_up_proj")
        with safe_open(os.path.join(a.bf16, wmap[f"{pre}.down_proj"]), "pt") as f:
            down = f.get_tensor(f"{pre}.down_proj")
        half = gate_up.shape[1] // 2
        x, sid, sw = load_capture(a.cap, layer, dev)
        kl = torch.tensor(keep[layer], device=dev)
        orig = kl[sid]                                                  # served id -> checkpoint expert id
        # RadixArk tensors of all four shards of this layer
        shards = {}
        for lo in range(0, 512, 128):
            name = f"layer-{layer:05d}-experts-{lo:04d}-{lo + 127:04d}.safetensors"
            with safe_open(os.path.join(a.radix, name), "pt") as f:
                shards[name] = (f.metadata(), {k: f.get_tensor(k) for k in f.keys()})
        served = sorted(set(keep[layer]))
        rows_of = {}
        for e in served:
            hit = (orig == e)
            r = hit.any(-1).nonzero().flatten()
            rows_of[e] = (r, (sw * hit.float()).sum(-1)[r])
        man = {"layer": layer, "experts": {}, "qs_only": []}
        newW = {}                                                       # (e, mat) -> (codes, scales)
        qg_experts = [e for e in served if len(rows_of[e][0]) >= a.min_rows]
        qs_experts = [e for e in range(512) if e not in set(qg_experts)]
        man["qs_only"] = qs_experts

        def tens(e, mat, kind):
            lo = (e // 128) * 128
            name = f"layer-{layer:05d}-experts-{lo:04d}-{lo + 127:04d}.safetensors"
            return shards[name][1][f"{pre}.{e}.{mat}.{kind}"]

        def bf16_w(es, mat):
            if mat == "gate_proj":
                return gate_up[es, :half].to(dev, torch.float32)
            if mat == "up_proj":
                return gate_up[es, half:].to(dev, torch.float32)
            return down[es].to(dev, torch.float32)

        tot = {"radix": 0.0, "qs": 0.0, "qg": 0.0}
        for b in range(0, len(qg_experts), a.batch):
            es = qg_experts[b:b + a.batch]
            # gate/up Hessian from routed inputs, weighted by routing weight^2
            Hx = torch.stack([(x[rows_of[e][0]].float() * rows_of[e][1].unsqueeze(-1)).T
                              @ (x[rows_of[e][0]].float() * rows_of[e][1].unsqueeze(-1)) for e in es])
            deq = {}
            for mat in ("gate_proj", "up_proj", "down_proj"):
                Wm = bf16_w(es, mat)
                g = torch.stack([tens(e, mat, "weight_scale_2").reshape(()) for e in es]).to(dev, torch.float32)
                rsc = torch.stack([tens(e, mat, "weight_scale") for e in es]).to(dev).to(torch.float32)
                rcodes = unpack(torch.stack([tens(e, mat, "weight") for e in es]).to(dev))
                if mat == "down_proj":
                    zs = []
                    for i, e in enumerate(es):
                        xe = x[rows_of[e][0]].float()
                        z = F.silu(xe @ deq["gate_proj"][i].T) * (xe @ deq["up_proj"][i].T)
                        zs.append((z * rows_of[e][1].unsqueeze(-1)).T @ (z * rows_of[e][1].unsqueeze(-1)))
                    H = torch.stack(zs)
                else:
                    H = Hx
                codes, sc = gptq(Wm, H, g, a.damp)
                qs_s, qs_codes = search_scales(Wm, g, rsc)
                dq = dequant(codes, sc, g)
                p_r = proxy(Wm - dequant(rcodes, rsc, g), H)
                p_s = proxy(Wm - dequant(qs_codes, qs_s, g), H)
                p_g = proxy(Wm - dq, H)
                for i, e in enumerate(es):
                    use_qg = bool(p_g[i] <= p_s[i])                    # never ship a QG matrix worse than QS's
                    newW[(e, mat)] = (codes[i], sc[i]) if use_qg else (qs_codes[i], qs_s[i])
                    man["experts"].setdefault(str(e), {})[mat] = {
                        "rows": int(len(rows_of[e][0])), "proxy_radix": p_r[i].item(), "proxy_qs": p_s[i].item(),
                        "proxy_qg": p_g[i].item(), "used": "qg" if use_qg else "qs"}
                    tot["radix"] += p_r[i].item()
                    tot["qs"] += p_s[i].item()
                    tot["qg"] += min(p_g[i].item(), p_s[i].item())
                deq[mat] = torch.stack([dequant(newW[(e, mat)][0][None], newW[(e, mat)][1][None], g[i:i + 1])[0]
                                        for i, e in enumerate(es)])
            del Hx
        if tot["qg"] > tot["qs"] * 1.0001:
            raise SystemExit(f"QG WORSE THAN QS on layer {layer}: {tot}")
        # QS for the rest (rare and unserved experts)
        for b in range(0, len(qs_experts), 128):
            es = qs_experts[b:b + 128]
            for mat in ("gate_proj", "up_proj", "down_proj"):
                Wm = bf16_w(es, mat)
                g = torch.stack([tens(e, mat, "weight_scale_2").reshape(()) for e in es]).to(dev, torch.float32)
                rsc = torch.stack([tens(e, mat, "weight_scale") for e in es]).to(dev).to(torch.float32)
                s, c = search_scales(Wm, g, rsc)
                for i, e in enumerate(es):
                    newW[(e, mat)] = (c[i], s[i])
        # write the four shards with exactness checks
        for name, (meta, tensors) in shards.items():
            lo = int(name.split("-experts-")[1][:4])
            for e in range(lo, lo + 128):
                for mat in ("gate_proj", "up_proj", "down_proj"):
                    codes, s = newW[(e, mat)]
                    s8 = s.to(torch.float8_e4m3fn)
                    if not torch.equal(s8.to(torch.float32), s):
                        raise SystemExit(f"SCALE ENCODE FAIL L{layer} e{e} {mat}")
                    p = pack(codes)
                    if not torch.equal(unpack(p), codes):
                        raise SystemExit(f"PACK FAIL L{layer} e{e} {mat}")
                    tensors[f"{pre}.{e}.{mat}.weight"] = p.cpu()
                    tensors[f"{pre}.{e}.{mat}.weight_scale"] = s8.cpu()
            path = os.path.join(a.out, name)
            save_file(tensors, path + ".tmp", metadata=meta)
            os.replace(path + ".tmp", path)
        man["totals"] = tot
        for lo in range(0, 512, 128):
            name = f"layer-{layer:05d}-experts-{lo:04d}-{lo + 127:04d}.safetensors"
            with open(os.path.join(a.out, "manifest", name + ".json"), "w") as f:
                json.dump(man if lo == 0 else {"layer": layer, "see": "experts-0000-0127 manifest"}, f)
        el = time.time() - t0
        print(f"[gptq] layer {layer} ({li + 1}/{len(layers)}) in {time.time() - tl:.0f} s: {len(qg_experts)} QG, "
              f"{len(qs_experts)} QS; calib proxy radix {tot['radix']:.4g} qs {tot['qs']:.4g} qg {tot['qg']:.4g} "
              f"(QG {100 * (1 - tot['qg'] / tot['radix']):.1f} % below RadixArk, QS "
              f"{100 * (1 - tot['qs'] / tot['radix']):.1f} %); elapsed {el / 60:.1f} min, "
              f"ETA {el / (li + 1) * (len(layers) - li - 1) / 60:.1f} min", flush=True)
        del gate_up, down, x, sid, sw, shards, newW
        torch.cuda.empty_cache()
    print(f"[PHASE] gptq done in {(time.time() - t0) / 60:.1f} min over {len(layers)} layers", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
