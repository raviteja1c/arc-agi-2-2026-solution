"""Acceptance simulations on the DFlash2 probe's outputs (agi2_dflash_probe.py) against the served target probabilities.

Every policy is simulated on the recorded histories themselves: the emitted tokens ARE the history (a coupling that is
exact in distribution, because the history is a sample of the served target), and only how many of them each verify
step commits depends on the policy:
- sampled chain (what DFLASH serves: the selector samples one path, exact rejection sampling against the target):
  given the written token x at slot j, the slot is accepted with probability min(1, q_j(x) / p_j(x)); an exact DP over
  step starts gives the expected number of steps (q from the probe, p = the served lp_samp);
- greedy chain (argmax path, accept while the target's own sample equals it): deterministic given the history;
- best-N tree (the N most probable nodes of the draft's prefix tree, the target samples down the tree): the history
  path is in the tree to depth d iff its path log-probability at d is >= the N-th best (exact, from the probe's beam);
- confidence-cut chains and a fixed batch-wide row budget across concurrent streams (global top-M nodes).
Tokens per step = generated tokens after each turn's first (prefill-sampled) token / steps, the served definition
(served_and_copy.py's j-column count).

Usage: python sim.py --probe <out dir> --inputs <runs/colab_inputs/dflash2_probe> [--names exp1] [--json out.json]
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time

import numpy as np


def load_inputs(inputs):
    z = np.load(os.path.join(inputs, "histories.npz"))
    reqs = {}
    for f in glob.glob(os.path.join(inputs, "requests", "*.json")):
        r = json.load(open(f))
        reqs[os.path.basename(f)[:-5]] = r
    return z, reqs


class Hist:
    def __init__(self, name, z, req, probe_dir, blocks):
        self.name = name
        self.ids = z[name]
        self.lps = z[name + "__lps"].astype(np.float64)
        self.j = z[name + "__j"]
        self.spans = [(int(s), int(e)) for s, e in req["spans"]]
        self.T = req["temperature"]
        self.N = len(self.ids)
        self.p = {}
        for B in blocks:
            f = os.path.join(probe_dir, f"{name}.B{B}.npz")
            if not os.path.exists(f):
                continue
            a = dict(np.load(f))
            row = np.full(self.N, -1, dtype=np.int64)
            row[a["anchors"]] = np.arange(len(a["anchors"]))
            # Fail closed on a missing anchor (2026-10-28, Sol's draft review): chain_dp / walk / budget_sim index
            # row[t] for every generated-span position, and a missing one (-1) silently reads the LAST row.
            need = [np.arange(s, e - 1) for s, e in self.spans if e - s >= 2]
            need = np.concatenate(need) if need else np.zeros(0, dtype=np.int64)
            miss = need[row[need] < 0] if need.size else need
            if miss.size:
                raise SystemExit(f"fail closed: {name} B{B}: {miss.size} generated-span anchors have no probe row "
                                 f"(first {miss[:5].tolist()}); the simulator would read them as the last row")
            a["row"] = row
            self.p[B] = a

    def gen_tokens(self):
        return sum(e - s - 1 for s, e in self.spans)

    def served_steps(self):
        return int(sum(((self.j[s:e] == 0).sum()) for s, e in self.spans))


def accept_probs(h: Hist, B: int):
    """a[t, j-1] = P(slot j accepted | slot j-1 accepted, the written tokens) for the sampled chain, rows = anchors."""
    a = h.p[B]
    S = B - 1
    anc = a["anchors"].astype(np.int64)
    lq = a["logq"].astype(np.float64)
    pos = anc[:, None] + np.arange(1, B)[None, :]
    lp = h.lps[np.minimum(pos, h.N - 1)]
    valid = np.arange(1, B)[None, :] <= a["vlen"].astype(np.int64)[:, None]
    with np.errstate(invalid="ignore", over="ignore"):
        r = np.exp(np.minimum(lq - lp, 0.0))
    r = np.where(valid & np.isfinite(lq) & np.isfinite(lp), r, 0.0)
    nan_lp = int((valid & ~np.isfinite(lp)).sum())
    return r, nan_lp


def chain_dp(h: Hist, B: int, acc, cap=None):
    """Exact expected steps of the sampled chain (acc rows = per-slot conditional acceptance). cap[t]: draft length
    used at anchor t (<= B-1), for confidence-cut chains. Returns (expected steps, expected rows)."""
    a = h.p[B]
    row = a["row"]
    vlen = a["vlen"].astype(np.int64)
    steps = rows = 0.0
    for s, e in h.spans:
        if e - s < 2:
            continue
        pi = np.zeros(e - s + 1)
        pi[0] = 1.0
        for t in range(s, e - 1):
            w = pi[t - s]
            if w < 1e-12:
                continue
            ri = row[t]
            v = int(vlen[ri])
            ell = B - 1 if cap is None else int(cap[ri])
            ell = min(ell, v)
            steps += w
            rows += w * (ell + 1)
            # P(L >= i) for i = 0..ell
            p_ge = 1.0
            ar = acc[ri]
            for i in range(0, ell + 1):
                # probability of exactly i accepted drafts
                p_next = p_ge * ar[i] if i < ell else 0.0
                p_eq = p_ge - p_next
                emit = min(i + 1, v)
                pi[t - s + emit] += w * p_eq
                p_ge = p_next
                if p_ge < 1e-12:
                    break
    return steps, rows


def walk(h: Hist, B: int, depth, rows_fn=None):
    """Deterministic walk: depth[ri] = accepted drafts at anchor row ri. Returns (steps, rows)."""
    a = h.p[B]
    row = a["row"]
    vlen = a["vlen"].astype(np.int64)
    steps = rows = 0
    for s, e in h.spans:
        t = s
        while t < e - 1:
            ri = row[t]
            v = int(vlen[ri])
            d = int(depth[ri])
            steps += 1
            rows += rows_fn(ri) if rows_fn else B
            t += min(d + 1, v)
    return steps, rows


def tree_depth(a, N):
    """Accepted depth of the best-N tree (the history path's path log-prob >= the N-th best node), per anchor row."""
    lq = a["logq"].astype(np.float64)
    cum = np.cumsum(np.where(np.isfinite(lq), lq, -np.inf), axis=1)
    thr = a["top"].astype(np.float64)[:, N - 1][:, None]
    # fp16 storage of top: a small tolerance; a tree with fewer than N finite nodes takes all of them
    thr = np.where(np.isfinite(thr), thr, -1e30)
    ok = np.isfinite(cum) & (cum >= thr - 1e-3)
    vl = np.arange(1, lq.shape[1] + 1)[None, :] <= a["vlen"].astype(np.int64)[:, None]
    ok &= vl
    return np.cumprod(ok, axis=1).sum(1)


def calib_served(h: Hist, acc4):
    """At the served steps' own anchors (block 4): mean predicted tokens per step against the realized run length."""
    a = h.p[4]
    row = a["row"]
    vlen = a["vlen"].astype(np.int64)
    pred, real = [], []
    for s, e in h.spans:
        jj = h.j[s:e]
        starts = np.flatnonzero(jj == 0) + s
        for k, p in enumerate(starts):
            t = p - 1
            nxt = starts[k + 1] if k + 1 < len(starts) else e
            if t < s or row[t] < 0:
                continue
            ri = row[t]
            v = int(vlen[ri])
            ell = min(3, v)
            ar = acc4[ri]
            p_ge, ex = 1.0, 0.0
            for i in range(ell + 1):
                p_next = p_ge * ar[i] if i < ell else 0.0
                ex += (p_ge - p_next) * min(i + 1, v)
                p_ge = p_next
            pred.append(ex)
            real.append(nxt - p)
    return np.asarray(pred), np.asarray(real)


def budget_sim(hists, B, streams, budget, N_fixed=None):
    """streams concurrent requests, each walking its own queue of histories; every global step verifies `budget` rows
    in all (one anchor row per stream plus the global top-(budget - streams) tree nodes by draft path log-prob), or with
    N_fixed, N_fixed nodes per stream. Returns tokens per global step (summed over streams) and rows per step."""
    queues = [[] for _ in range(streams)]
    order = sorted(hists, key=lambda h: -h.gen_tokens())
    loads = [0] * streams
    for h in order:  # longest first onto the least-loaded stream
        k = int(np.argmin(loads))
        queues[k].append(h)
        loads[k] += h.gen_tokens()
    state = []
    for q in queues:
        it = iter([(h, s, e) for h in q for (s, e) in h.spans if e - s >= 2])
        cur = next(it, None)
        state.append([it, cur, cur[1] if cur else None])
    cums = {}
    for h in hists:
        a = h.p[B]
        lq = a["logq"].astype(np.float64)
        c = np.cumsum(np.where(np.isfinite(lq), lq, -np.inf), axis=1)
        vl = np.arange(1, lq.shape[1] + 1)[None, :] <= a["vlen"].astype(np.int64)[:, None]
        cums[h.name] = np.where(vl, c, -np.inf)
    gsteps = 0
    gtok = 0
    grows = 0
    active_hist = []
    while True:
        act = [k for k in range(streams) if state[k][1] is not None]
        if not act:
            break
        active_hist.append(len(act))
        tops = []
        for k in act:
            h, s, e = state[k][1]
            t = state[k][2]
            ri = h.p[B]["row"][t]
            tops.append(h.p[B]["top"][ri].astype(np.float64))
        if N_fixed is None:
            m = budget - len(act)
            allv = np.concatenate(tops)
            allv = allv[np.isfinite(allv)]
            m = max(0, min(m, len(allv)))
            thr = np.sort(allv)[::-1][m - 1] if m > 0 else np.inf
        rows = len(act)
        for idx, k in enumerate(act):
            h, s, e = state[k][1]
            t = state[k][2]
            a = h.p[B]
            ri = a["row"][t]
            v = int(a["vlen"][ri])
            tp = tops[idx]
            if N_fixed is None:
                th = thr
            else:
                fin = tp[np.isfinite(tp)]
                th = fin[min(N_fixed, len(fin)) - 1] if len(fin) else np.inf
            nodes = int((np.isfinite(tp) & (tp >= th - 1e-3)).sum()) if np.isfinite(th) else 0
            rows += nodes
            c = cums[h.name][ri]
            d = int(np.cumprod(np.isfinite(c) & (c >= th - 1e-3)).sum()) if np.isfinite(th) else 0
            emit = min(d + 1, v)
            gtok += emit
            t += emit
            if t >= e - 1:
                nxt = next(state[k][0], None)
                state[k][1] = nxt
                state[k][2] = nxt[1] if nxt else None
            else:
                state[k][2] = t
        grows += rows
        gsteps += 1
    return dict(tokens_per_step=gtok / max(gsteps, 1), rows_per_step=grows / max(gsteps, 1), steps=gsteps,
                tokens=gtok, mean_active=float(np.mean(active_hist)) if active_hist else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", required=True)
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--names", default="exp1", help="comma list of round prefixes (exp1, t12) or full names")
    ap.add_argument("--blocks", default="4,6,8,12,16,32")
    ap.add_argument("--trees", default="3,4,6,8,12,16,24,32,48,64")
    ap.add_argument("--budget", default="", help="e.g. 12:48,96,192 = 12 streams at these batch row budgets")
    ap.add_argument("--budget-blocks", default="8,16", help="lattices the budget simulations use")
    ap.add_argument("--json", default="")
    ap.add_argument("--require-all", action="store_true",
                    help="fail if any asked history has no probe output (a gate must not shrink silently)")
    ap.add_argument("--allow-nan-lp", action="store_true",
                    help="tolerate non-finite target logprobs at valid slots (default: fail closed)")
    a = ap.parse_args()
    blocks = [int(b) for b in a.blocks.split(",")]
    z, reqs = load_inputs(a.inputs)
    pref = a.names.split(",")
    names = sorted(n for n in reqs if any(n == p or n.startswith(p + "__") for p in pref))
    hists = []
    for n in names:
        if not os.path.exists(os.path.join(a.probe, f"{n}.json")):
            continue
        hists.append(Hist(n, z, reqs[n], a.probe, blocks))
    print(f"{len(hists)} histories with probe output of {len(names)} asked", flush=True)
    if not hists:
        return 1
    if a.require_all and len(hists) != len(names):
        missing = sorted(set(names) - {h.name for h in hists})
        raise SystemExit(f"fail closed: {len(missing)} asked histories have no probe output: {missing[:5]}")
    rep = dict(histories=[h.name for h in hists], gen=sum(h.gen_tokens() for h in hists))
    served_steps = sum(h.served_steps() for h in hists)
    gen = rep["gen"]
    rep["served_tps"] = gen / served_steps
    print(f"served (block 4, j column): {gen:,} tokens / {served_steps:,} steps = {gen / served_steps:.3f} tok/step")
    acc = {}
    nan_lp = 0
    for B in blocks:
        for h in hists:
            if B in h.p:
                acc[(h.name, B)], n = accept_probs(h, B)
                nan_lp += n
    rep["nan_lp_slots"] = nan_lp
    if nan_lp and not a.allow_nan_lp:
        raise SystemExit(f"fail closed: {nan_lp} valid slots have a non-finite target logprob (pass --allow-nan-lp "
                         f"only with an explanation)")
    # calibration at the served steps' own anchors
    if all(4 in h.p for h in hists):
        P, R = [], []
        for h in hists:
            p_, r_ = calib_served(h, acc[(h.name, 4)])
            P.append(p_)
            R.append(r_)
        P, R = np.concatenate(P), np.concatenate(R)
    # generated histories (sim_inputs_from_teacher.py) carry no served steps: no calibration
    if all(4 in h.p for h in hists) and len(P) >= 100:
        bins = np.quantile(P, np.linspace(0, 1, 11))
        cal = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (P >= lo) & (P <= hi)
            cal.append(dict(lo=float(lo), hi=float(hi), n=int(m.sum()), pred=float(P[m].mean()), real=float(R[m].mean())))
        rep["calibration_b4"] = dict(steps=int(len(P)), pred=float(P.mean()), real=float(R.mean()), deciles=cal)
        print(f"calibration at the served anchors (block 4): predicted {P.mean():.3f} tok/step vs realized {R.mean():.3f}"
              f" over {len(P):,} steps")
        for c in cal:
            print(f"   pred {c['lo']:.2f}-{c['hi']:.2f}: n {c['n']:7d}  pred {c['pred']:.3f}  real {c['real']:.3f}")
    rep["chain"] = {}
    rep["greedy"] = {}
    rep["tree"] = {}
    trees = [int(x) for x in a.trees.split(",")]
    for B in blocks:
        hs = [h for h in hists if B in h.p]
        if len(hs) != len(hists):
            continue
        t0 = time.time()
        st = rw = 0.0
        for h in hs:
            s_, r_ = chain_dp(h, B, acc[(h.name, B)])
            st += s_
            rw += r_
        gs = 0
        for h in hs:
            s_, _ = walk(h, B, h.p[B]["glcp"])
            gs += s_
        rep["chain"][B] = dict(tps=gen / st, rows_per_step=rw / st, steps=st)
        rep["greedy"][B] = dict(tps=gen / gs, steps=gs)
        line = f"block {B:2d}: sampled chain {gen / st:.3f} tok/step ({rw / st:.2f} rows/step) | greedy chain {gen / gs:.3f}"
        rep["tree"][B] = {}
        for N in trees:
            if N > h.p[B]["top"].shape[1]:
                continue
            ts = 0
            for h in hs:
                d = tree_depth(h.p[B], N)
                s_, _ = walk(h, B, d)
                ts += s_
            rep["tree"][B][N] = dict(tps=gen / ts, rows=N + 1)
        line += " | trees " + " ".join(f"N{N}:{v['tps']:.2f}" for N, v in rep["tree"][B].items())
        print(line + f"  ({time.time() - t0:.0f}s)", flush=True)
    # per-slot diagnostics along the history path (conditioned on every earlier slot having the written token among
    # the candidates): candidate recall@16 (the written token is among slot j's K candidates), recall@1 (it is the
    # selector's top transition), mean sampled-chain acceptance; and a perfect-selector ceiling (accept while the written
    # token is a candidate: what a better selector on this backbone could reach with target-sample matching)
    rep["slots"] = {}
    for B in blocks:
        if any(B not in h.p for h in hists):
            continue
        S = B - 1
        n_ok = np.zeros(S)
        r16 = np.zeros(S)
        r1 = np.zeros(S)
        am = np.zeros(S)
        orc_steps = 0
        for h in hists:
            a_ = h.p[B]
            idx = a_["idx"].astype(np.int64)
            rank = a_["rank"].astype(np.int64)
            vl = np.arange(1, B)[None, :] <= a_["vlen"].astype(np.int64)[:, None]
            pres = (idx >= 0) & vl
            ok_prev = np.cumprod(np.concatenate([np.ones((len(idx), 1), bool), pres[:, :-1]], 1), 1).astype(bool) & vl
            n_ok += ok_prev.sum(0)
            r16 += (ok_prev & pres).sum(0)
            r1 += (ok_prev & pres & (rank == 0)).sum(0)
            am += np.where(ok_prev, acc[(h.name, B)], 0).sum(0)
            s_, _ = walk(h, B, np.cumprod(pres, 1).sum(1))
            orc_steps += s_
        rep["slots"][B] = dict(recall16=(r16 / np.maximum(n_ok, 1)).tolist(), recall1=(r1 / np.maximum(n_ok, 1)).tolist(),
                               accept=(am / np.maximum(n_ok, 1)).tolist(), oracle_selector_tps=gen / orc_steps)
        print(f"slots B{B}: recall@16 " + " ".join(f"{x:.2f}" for x in rep['slots'][B]['recall16']) +
              f" | recall@1 " + " ".join(f"{x:.2f}" for x in rep['slots'][B]['recall1']) +
              f" | sampled accept " + " ".join(f"{x:.2f}" for x in rep['slots'][B]['accept']) +
              f" | perfect-selector chain {gen / orc_steps:.3f} tok/step", flush=True)
    # confidence-cut sampled chains: draft slot j only while the selector's top transition probability out of the
    # drafted predecessor is >= c (decided before slot j's token is drawn, so exact rejection stays lossless)
    rep["confcut"] = {}
    for B in blocks:
        if B < 8 or any(B not in h.p for h in hists):
            continue
        for c in (0.2, 0.4, 0.6, 0.8, 0.9):
            st = rw = 0.0
            for h in hists:
                conf = h.p[B]["conf"].astype(np.float64)
                below = conf < c
                cap = np.where(below.any(1), below.argmax(1), conf.shape[1])  # slots drafted before the first low one
                s_, r_ = chain_dp(h, B, acc[(h.name, B)], cap=cap)
                st += s_
                rw += r_
            rep["confcut"][f"B{B}_c{c}"] = dict(tps=gen / st, rows_per_step=rw / st)
            print(f"conf-cut sampled chain B{B} c {c}: {gen / st:.3f} tok/step at {rw / st:.2f} rows/step "
                  f"({gen / st / (rw / st):.3f} tokens per row)", flush=True)
    if a.budget:
        streams, budgets = a.budget.split(":")
        streams = int(streams)
        rep["budget"] = {}
        bb = [int(x) for x in a.budget_blocks.split(",")] if a.budget_blocks else blocks
        for B in bb:
            if B < 8 or any(B not in h.p for h in hists):
                continue
            for M in [int(x) for x in budgets.split(",")]:
                g = budget_sim(hists, B, streams, M)
                nf = max(1, M // streams - 1)
                f = budget_sim(hists, B, streams, M, N_fixed=min(nf, 64))
                rep["budget"][f"B{B}_M{M}"] = dict(global_=g, fixed=f)
                print(f"budget B{B} {streams} streams M{M}: global top-M {g['tokens_per_step'] / g['mean_active']:.3f} tok/step/stream "
                      f"({g['rows_per_step']:.0f} rows) | fixed N{nf} {f['tokens_per_step'] / f['mean_active']:.3f} "
                      f"({f['rows_per_step']:.0f} rows)", flush=True)
    if a.json:
        json.dump(rep, open(a.json, "w"), indent=1, default=float)
        print("wrote", a.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
