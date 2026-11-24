"""Choose the token histories to prefill for the expert-activation capture (laptop side).

Source: the 199 on-policy harness histories of ARC-AGI-2 TRAINING tasks (runs/10/25/1025_1926_colab-g4-dflash2-ft/
harness_h1; make_harness_histories.py already failed closed on any evaluation or child task). Its split.json holds 12
validation tasks: their histories form the held-out `gate` phase; the rest form `calib`.
Each history is truncated to --calib-cap / --gate-cap tokens (spreads the budget over more tasks while keeping long
contexts), taken in a seeded random order until the phase budget is met.

Writes <out>/xq_histories.npz (one int32 array per chosen history) and <out>/xq_plan.json.
  python -m agi2_modal_harness.comm.nvfp4_capture_plan --out runs/10/29/xq_capture_inputs
"""
from __future__ import annotations

import argparse
import json
import os
import random

import numpy as np

SRC = "runs/10/25/1025_1926_colab-g4-dflash2-ft/harness_h1"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--out", required=True)
    ap.add_argument("--calib-tokens", type=int, default=300_000)
    ap.add_argument("--gate-tokens", type=int, default=50_000)
    ap.add_argument("--calib-cap", type=int, default=48_000)
    ap.add_argument("--gate-cap", type=int, default=24_000)
    ap.add_argument("--seed", type=int, default=1029)
    a = ap.parse_args()
    idx = json.load(open(os.path.join(a.src, "index.json")))
    split = json.load(open(os.path.join(a.src, "split.json")))
    z = np.load(os.path.join(a.src, "histories.npz"))
    val_tasks = set(split["val_tasks"])
    rng = random.Random(a.seed)
    calib = [r for r in idx if r["task"] not in val_tasks]
    gate = [r for r in idx if r["task"] in val_tasks]
    rng.shuffle(calib)
    rng.shuffle(gate)
    os.makedirs(a.out, exist_ok=True)
    arrays, plan = {}, {"source": a.src, "seed": a.seed, "phases": {}}
    for phase, rows, budget, cap in (("calib", calib, a.calib_tokens, a.calib_cap),
                                     ("gate", gate, a.gate_tokens, a.gate_cap)):
        used, chosen = 0, []
        for r in rows:
            if used >= budget:
                break
            ids = z[r["name"]].astype(np.int32)[:cap]
            arrays[r["name"]] = ids
            chosen.append({"name": r["name"], "task": r["task"], "tokens": int(len(ids))})
            used += len(ids)
        plan["phases"][phase] = {"histories": chosen, "tokens": used,
                                 "tasks": sorted({c["task"] for c in chosen})}
        print(f"{phase}: {len(chosen)} histories, {used} tokens, {len(plan['phases'][phase]['tasks'])} tasks")
    assert not (set(plan["phases"]["calib"]["tasks"]) & set(plan["phases"]["gate"]["tasks"])), "task overlap"
    np.savez(os.path.join(a.out, "xq_histories.npz"), **arrays)
    json.dump(plan, open(os.path.join(a.out, "xq_plan.json"), "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
