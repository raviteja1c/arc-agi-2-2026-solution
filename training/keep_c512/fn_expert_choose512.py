"""Re-choose Flash-Next's routed experts from all 512 on captured harness traffic: which 128 (K384) or 160 (K352) to
drop so each MoE layer's output stays closest to the FULL model's (all 512 experts), under the keep loader's serving
semantics. The census-chosen K384 (keep_384_code_bal4.json, every Flash-Next sub since sub 12) costs KL 0.344 against
0.195 at all 512 (10-30 klvar runs); this picks the drops on harness traffic instead.

Inputs (Modal cache Volume): the BF16 checkpoint (all 512 experts' gate_up_proj/down_proj and the full router rows) and
the two harness-traffic captures fn_expert_score.py used:
  --cap16  phases `calib` (27 training-task histories, 350k rows) and `gate` (the 4 held-out KL histories, 52k rows);
  --cap32  phase `late32`, rows at history position >= --min-pos (16 more histories, 256k rows).
The captures hold each MoE layer's input x, recorded from a K384 server; x is close to what the full model computes, so
x itself is the one approximation (second order). Rows are thinned by --stride for the choice; the gate phase is
scored whole and never used to choose.

Serving semantics (qwen4_exp.py:141-170 at sub18-levers): a keep list slices the router; each token takes the top-10
kept experts by router logit, weights renormalized over those 10 (the full model is the same rule over all 512). So
y_K(x) = sum_{k in top10_K} p_k f_k(x) / sum p_k, with p = softmax(x Wr^T) over all 512.

Choice (greedy, in rounds): start with all 512 kept. Each round, for every kept expert e, its marginal cost
  sum_t ( ||y_{K - e}(t) - y_full(t)||^2 - ||y_K(t) - y_full(t)||^2 ) / sum_t ||y_full(t)||^2,
summed over the two calibration datasets (each normalized by its own sum ||y_full||^2), where y_{K - e} takes the 11th
kept expert in and renormalizes (prefix/suffix sums, exact). An expert in no token's kept top-10 costs exactly 0. The
round drops the --drops[i] cheapest. After 128 drops the kept set is K384; after 160, K352.

Per layer it also scores, on every dataset including the held-out gate: the census K384, the new K384 and the new K352
(relative error to the full model), and a routing check: the captured served top-10 (mapped to checkpoint ids through
the census list) against our fp32 top-10 restricted to the census set (expect ~0.84-0.99; fail under 0.6).

Outputs in --out: choose_L<ll>.json per layer; with every layer present (--merge or the last box), keep_384_c512.json
and keep_352_c512.json (EP-balanced: 4 bins of equal size by the new sets' harness counts) and summary.json.

    python -m agi2_modal_harness.comm.fn_expert_choose512 --bf16 /tmp/fnbf16full --census /opt/dev/keep_384_code_bal4.json \
        --out /vol/cache/dev/fnscore/c512 --layers 0-15 --skip-existing
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn.functional as F
from safetensors import safe_open

TOP = 10
E_ALL = 512
torch.backends.cuda.matmul.allow_tf32 = True   # ranking, not serving: TF32's ~1e-3 relative noise cancels in deltas


def parse_layers(s):
    out = []
    for part in s.split(","):
        a, _, b = part.partition("-")
        lo, hi = int(a), int(b or a)
        out += list(range(lo, hi + 1)) if lo <= hi else list(range(lo, hi - 1, -1))
    return out


def vol(cmd):
    import subprocess
    subprocess.run([sys.executable, "-m", "agi2_modal_harness.comm.nvfp4_vol", cmd], check=False,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


LOADERS = ThreadPoolExecutor(16)   # the captures are hundreds of small files a layer: Volume latency, not bandwidth


def load_phase(cap, phase, layer, min_pos=0):
    files = sorted(glob.glob(os.path.join(cap, phase, f"L{layer:02d}", "*.pt")))
    if not files:
        return None
    parts = list(LOADERS.map(lambda f: torch.load(f, map_location="cpu"), files))
    x, ids = torch.cat([p["x"] for p in parts]), torch.cat([p["ids"] for p in parts]).long()
    if min_pos:
        man = json.load(open(os.path.join(cap, phase, "manifest.json")))
        assert man["rows"] == x.shape[0], f"manifest rows {man['rows']} != captured {x.shape[0]}"
        keep = torch.zeros(x.shape[0], dtype=torch.bool)
        for h in man["histories"]:
            if h["tokens"] > min_pos:
                keep[h["start"] + min_pos:h["start"] + h["tokens"]] = True
        x, ids = x[keep], ids[keep]
    return x, ids


def tensor(bf16, wmap, name):
    with safe_open(os.path.join(bf16, wmap[name]), "pt") as f:
        return f.get_tensor(name)


def router_name(wmap, layer):
    cands = [k for k in wmap if k.startswith(f"model.language_model.layers.{layer}.mlp.")
             and k.endswith(".weight") and (".gate." in k or k.endswith("mlp.gate.weight") or ".router." in k)
             and "shared" not in k and "experts" not in k]
    if len(cands) != 1:
        raise SystemExit(f"layer {layer}: router weight not unique: {cands}")
    return cands[0]


def expert_outputs(x, ids, G, U, D):
    """f_e(x) for every (token, slot) in ids [c, W] (checkpoint ids). Returns [c, W, H] fp32."""
    c, W = ids.shape
    H = x.shape[1]
    out = torch.zeros(c * W, H, device=x.device, dtype=torch.float32)
    flat = ids.reshape(-1)
    order = torch.argsort(flat)
    uniq, counts = torch.unique_consecutive(flat[order], return_counts=True)
    starts = torch.cumsum(counts, 0) - counts
    xf = x.float()
    for e, s, n in zip(uniq.tolist(), starts.tolist(), counts.tolist()):
        pos = order[s:s + n]
        xe = xf[pos // W]
        out[pos] = (F.silu(xe @ G[e].T) * (xe @ U[e].T)) @ D[e].T
    return out.view(c, W, H)


def top_kept(p, kept, k):
    """The k highest-probability kept experts per token: (probabilities, ids)."""
    return torch.topk(p.masked_fill(~kept, -1.0), k, -1)


def layer_out(p, kept, x, G, U, D):
    pv, ids = top_kept(p, kept, TOP)
    Fx = expert_outputs(x, ids, G, U, D)
    return (pv[:, :, None] * Fx).sum(1) / pv.sum(-1, keepdim=True), ids


def marginal(p, kept, x, y_full, G, U, D, cost, scale):
    """Add each kept expert's marginal cost (error to y_full when it is also dropped) for this chunk into cost[512]."""
    pv, ids = top_kept(p, kept, TOP + 1)                        # [c, 11]
    Fx = expert_outputs(x, ids, G, U, D)                        # [c, 11, H]
    pF = pv[:, :TOP, None] * Fx[:, :TOP]
    pz = pv[:, :TOP]
    S10, Z10 = pF.sum(1), pz.sum(-1, keepdim=True)
    base = ((S10 / Z10 - y_full) ** 2).sum(-1)                  # [c]
    pre, suf = torch.cumsum(pF, 1), torch.flip(torch.cumsum(torch.flip(pF, [1]), 1), [1])
    prez, sufz = torch.cumsum(pz, 1), torch.flip(torch.cumsum(torch.flip(pz, [1]), 1), [1])
    zero = torch.zeros_like(pF[:, 0])
    p11, F11 = pv[:, TOP:TOP + 1], Fx[:, TOP]
    for k in range(TOP):
        Sk = (pre[:, k - 1] if k > 0 else zero) + (suf[:, k + 1] if k + 1 < TOP else zero) + p11 * F11
        Zk = (prez[:, k - 1:k] if k > 0 else 0.0) + (sufz[:, k + 1:k + 2] if k + 1 < TOP else 0.0) + p11
        d = ((Sk / Zk - y_full) ** 2).sum(-1) - base
        cost.index_add_(0, ids[:, k], d.double() * scale)


def set_error(p, kept, x, y_full, G, U, D):
    y, _ = layer_out(p, kept, x, G, U, D)
    return float(((y - y_full) ** 2).sum())


def counts_for(p, kept):
    c = torch.zeros(E_ALL, dtype=torch.float64, device=p.device)
    _, ids = top_kept(p, kept, TOP)
    c.index_add_(0, ids.reshape(-1), torch.ones(ids.numel(), dtype=torch.float64, device=p.device))
    return c


def balance(ids, counts, ep):
    cap = len(ids) // ep
    bins = [[] for _ in range(ep)]
    load = [0.0] * ep
    for e in sorted(ids, key=lambda i: -counts[i]):
        r = min((r for r in range(ep) if len(bins[r]) < cap), key=lambda r: load[r])
        bins[r].append(e)
        load[r] += counts[e]
    return [e for b in bins for e in sorted(b)]


class Data:
    """One dataset of a layer on the device, in chunks: x (bf16), p (fp32 softmax over 512), y_full (fp32)."""

    def __init__(self, x, Wr, G, U, D, chunk, dev):
        self.chunks = []
        self.y2 = 0.0
        for s in range(0, x.shape[0], chunk):
            xc = x[s:s + chunk].to(dev)
            p = torch.softmax(xc.float() @ Wr.T, -1)
            y, _ = layer_out(p, torch.ones(E_ALL, dtype=torch.bool, device=dev), xc, G, U, D)
            self.y2 += float((y * y).sum())
            self.chunks.append((xc, p, y))
        self.rows = x.shape[0]


def choose(datasets, G, U, D, drops, dev, log=None):
    """Greedy drops in rounds. Returns the drop order [(expert, round, cost)] and kept masks at each cumulative total."""
    kept = torch.ones(E_ALL, dtype=torch.bool, device=dev)
    order, masks, total = [], {}, 0
    for ri, R in enumerate(drops):
        cost = torch.zeros(E_ALL, dtype=torch.float64, device=dev)
        for ds in datasets:
            for xc, p, y in ds.chunks:
                marginal(p, kept, xc, y, G, U, D, cost, 1.0 / max(ds.y2, 1e-30))
        cand = torch.nonzero(kept).flatten()
        pick = cand[torch.argsort(cost[cand])[:R]]
        for e in pick.tolist():
            order.append((e, ri, float(cost[e])))
        kept[pick] = False
        total += R
        masks[total] = kept.clone()
        if log:
            log(f"  round {ri}: dropped {R} (total {total}), cost of the picked {float(cost[pick].sum()):.5f}, "
                f"max {float(cost[pick].max()):.2e}, zero-cost kept left {int(((cost == 0) & kept).sum())}")
    return order, masks


def selftest(dev):
    """Random small layer (E 40): the greedy's marginal cost of each kept expert, given a dropped set, must equal the
    exact change in set error from dropping it too."""
    global E_ALL
    torch.manual_seed(0)
    saved, E_ALL = E_ALL, 40
    tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False   # exactness check in full fp32
    try:
        H, I, n = 64, 16, 2000
        x = torch.randn(n, H, device=dev).bfloat16()
        Wr = torch.randn(E_ALL, H, device=dev)
        G, U, D = (torch.randn(E_ALL, I, H, device=dev) / 8, torch.randn(E_ALL, I, H, device=dev) / 8,
                   torch.randn(E_ALL, H, I, device=dev) / 4)
        ds = Data(x, Wr, G, U, D, 700, dev)
        kept = torch.ones(E_ALL, dtype=torch.bool, device=dev)
        kept[torch.tensor([3, 7, 11], device=dev)] = False
        cost = torch.zeros(E_ALL, dtype=torch.float64, device=dev)
        for xc, p, y in ds.chunks:
            marginal(p, kept, xc, y, G, U, D, cost, 1.0)
        base = sum(set_error(p, kept, xc, y, G, U, D) for xc, p, y in ds.chunks)
        worst = 0.0
        for e in torch.nonzero(kept).flatten().tolist():
            k2 = kept.clone()
            k2[e] = False
            exact = sum(set_error(p, k2, xc, y, G, U, D) for xc, p, y in ds.chunks) - base
            worst = max(worst, abs(exact - float(cost[e])) / max(abs(exact), 1e-6))
        order, masks = choose([ds], G, U, D, [5, 5], dev)
        ok = worst < 2e-3 and len(order) == 10 and int(masks[10].sum()) == E_ALL - 10
        print(f"selftest: marginal vs exact worst rel {worst:.2e}; greedy dropped {len(order)} -> "
              f"{'ok' if ok else 'FAILED'}", flush=True)
        return 0 if ok else 3
    finally:
        E_ALL = saved
        torch.backends.cuda.matmul.allow_tf32 = tf32


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--bf16")
    ap.add_argument("--census", help="the shipped K384 keep list (checkpoint ids; maps the captured served ids)")
    ap.add_argument("--cap16", default="/vol/cache/dev/fncap/xq1")
    ap.add_argument("--cap32", default="/vol/cache/dev/fncap/xq32")
    ap.add_argument("--late-phase", default="late32")
    ap.add_argument("--min-pos", type=int, default=16000)
    ap.add_argument("--out")
    ap.add_argument("--layers", default="0-47")
    ap.add_argument("--stride", type=int, default=3, help="use every n-th row of the calibration datasets")
    ap.add_argument("--drops", default="32,32,32,16,16,16,16", help="drops a round; cumulative 128 = K384, 160 = K352")
    ap.add_argument("--chunk", type=int, default=4096)
    ap.add_argument("--ep", type=int, default=4)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--merge", action="store_true", help="no scoring: write the keep files from the layer files")
    a = ap.parse_args()
    dev = torch.device("cuda")
    t0 = time.time()

    def log(msg):
        print(f"[c512 {time.strftime('%H:%M:%S')} +{time.time() - t0:5.0f}s] {msg}", flush=True)

    rc = selftest(dev)
    if a.selftest or rc:
        return rc
    drops = [int(d) for d in a.drops.split(",")]
    marks = {128: "k384", 160: "k352"}
    os.makedirs(a.out, exist_ok=True)
    census = json.load(open(a.census))
    census = census["layers"] if isinstance(census, dict) else census
    wmap = json.load(open(os.path.join(a.bf16, "model.safetensors.index.json")))["weight_map"]

    def fetch(layer):
        """A layer's weights and captures on the CPU (run one layer ahead, in the background)."""
        pre = f"model.language_model.layers.{layer}.mlp.experts"
        jobs = {"gu": LOADERS.submit(tensor, a.bf16, wmap, f"{pre}.gate_up_proj"),
                "down": LOADERS.submit(tensor, a.bf16, wmap, f"{pre}.down_proj"),
                "wr": LOADERS.submit(tensor, a.bf16, wmap, router_name(wmap, layer))}
        cal = load_phase(a.cap16, "calib", layer)
        late = load_phase(a.cap32, a.late_phase, layer, a.min_pos)
        gate = load_phase(a.cap16, "gate", layer)
        return {k: j.result() for k, j in jobs.items()} | {"cal": cal, "late": late, "gate": gate}

    todo = [] if a.merge else parse_layers(a.layers)
    ahead = ThreadPoolExecutor(1)
    pending = {}
    for i, layer in enumerate(todo):
        tl = time.time()
        if a.skip_existing:
            vol("reload")
            if os.path.exists(os.path.join(a.out, f"choose_L{layer:02d}.json")):
                log(f"L{layer:02d} already chosen (another box): skipped")
                pending.pop(layer, None)
                continue
        got = (pending.pop(layer) if layer in pending else ahead.submit(fetch, layer)).result()
        for nxt in todo[i + 1:i + 2]:   # prefetch the next layer while this one computes
            if nxt not in pending and not os.path.exists(os.path.join(a.out, f"choose_L{nxt:02d}.json")):
                pending[nxt] = ahead.submit(fetch, nxt)
        log(f"L{layer:02d}: loaded in {time.time() - tl:.0f}s")
        gu = got["gu"]
        assert gu.shape[0] == E_ALL, f"expected {E_ALL} experts, got {tuple(gu.shape)}"
        half = gu.shape[1] // 2
        G, U = gu[:, :half].to(dev, torch.float32), gu[:, half:].to(dev, torch.float32)
        D = got["down"].to(dev, torch.float32)
        Wr = got["wr"].to(dev, torch.float32)
        assert Wr.shape[0] == E_ALL
        cal, late, gate = got["cal"], got["late"], got["gate"]
        del got, gu
        # routing check on the calib rows: captured served ids -> checkpoint ids through the census list
        cmap = torch.tensor(census[layer], dtype=torch.long)
        cset = torch.zeros(E_ALL, dtype=torch.bool, device=dev)
        cset[cmap.to(dev)] = True
        xs, ids_cap = cal[0][:20000].to(dev), cmap[cal[1][:20000]].to(dev)
        p = torch.softmax(xs.float() @ Wr.T, -1)
        mine = top_kept(p, cset, TOP)[1]
        agree = float((torch.sort(mine, -1).values == torch.sort(ids_cap, -1).values).all(-1).float().mean())
        if agree < 0.6:
            log(f"L{layer:02d}: routing check FAILED ({agree:.3f} of tokens agree with the capture)")
            return 2
        dsets = {"calib": Data(cal[0][::a.stride], Wr, G, U, D, a.chunk, dev),
                 "late": Data(late[0][::a.stride], Wr, G, U, D, a.chunk, dev)}
        del cal, late
        log(f"L{layer:02d}: routing check {agree:.4f}; rows calib {dsets['calib'].rows}, late {dsets['late'].rows}; "
            f"choosing")
        order, masks = choose([dsets["calib"], dsets["late"]], G, U, D, drops, dev, log)
        dsets["gate"] = Data(gate[0], Wr, G, U, D, a.chunk, dev)
        del gate
        sets = {"census384": cset} | {marks[t]: m for t, m in masks.items() if t in marks}
        errs = {name: {dn: sum(set_error(p, m, xc, y, G, U, D) for xc, p, y in ds.chunks) / max(ds.y2, 1e-30)
                       for dn, ds in dsets.items()} for name, m in sets.items()}
        counts = {name: sum(counts_for(p, m) for ds in (dsets["calib"], dsets["late"]) for _, p, _ in ds.chunks)
                  for name, m in sets.items() if name != "census384"}
        res = {"layer": layer, "routing_check": agree, "drops": drops,
               "rows": {dn: ds.rows for dn, ds in dsets.items()},
               "order": order, "errors": errs,
               "kept": {name: torch.nonzero(m).flatten().tolist() for name, m in sets.items() if name != "census384"},
               "counts": {name: c.cpu().tolist() for name, c in counts.items()},
               "overlap_with_census": {name: int((m & cset).sum()) for name, m in sets.items() if name != "census384"}}
        json.dump(res, open(os.path.join(a.out, f"choose_L{layer:02d}.json"), "w"))
        if a.skip_existing:
            vol("commit")
        log(f"L{layer:02d} {time.time() - tl:4.0f}s | rel err to all-512 (calib / late / gate): " + " | ".join(
            f"{n} {e['calib']:.4f}/{e['late']:.4f}/{e['gate']:.4f}" for n, e in errs.items())
            + f" | new K384 keeps {res['overlap_with_census']['k384']} of the census 384")
        del G, U, D, Wr, dsets
        torch.cuda.empty_cache()
    vol("reload")
    files = [os.path.join(a.out, f"choose_L{layer:02d}.json") for layer in range(48)]
    missing = [layer for layer, f in enumerate(files) if not os.path.exists(f)]
    if missing:
        log(f"keep files wait for layers {missing}")
        return 0
    rs = [json.load(open(f)) for f in files]
    for name, k in (("k384", 384), ("k352", 352)):
        layers_out = [balance(r["kept"][name], r["counts"][name], a.ep) for r in rs]
        assert all(len(x) == k for x in layers_out)
        json.dump({"layers": layers_out, "source": f"fn_expert_choose512 {name}: greedy from all 512 on harness "
                   f"traffic (calib + late32), drops {a.drops}", "cut": 512 - k},
                  open(os.path.join(a.out, f"keep_{k}_c512.json"), "w"))
    summ = {}
    for name in ("census384", "k384", "k352"):
        for dn in ("calib", "late", "gate"):
            summ[f"{name}_{dn}"] = sum(r["errors"][name][dn] for r in rs) / len(rs)
    summ["k384_overlap_with_census_mean"] = sum(r["overlap_with_census"]["k384"] for r in rs) / len(rs)
    json.dump(summ, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    vol("commit")
    log("mean rel err to all-512: " + json.dumps({k: round(v, 5) for k, v in summ.items()}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
