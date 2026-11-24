"""The 32k capture plan: harness histories of ARC training tasks (runs/10/25/1025_1926_colab-g4-dflash2-ft/harness_h1),
not the held-out validation tasks, at least 32,000 tokens long, capped at 32,000; 16 chosen in a seeded order.
Positions 0-15,999 repeat what the 10-29 capture (cap 16k) saw; 16,000-31,999 are the late reasoning it missed.

    python runs/plans/1030_sub18_modal/cap32/make_plan32.py  ->  cap32/xq32_histories.npz, cap32/xq32_plan.json
"""
import json
import os
import random

import numpy as np

SRC = "runs/10/25/1025_1926_colab-g4-dflash2-ft/harness_h1"
OUT = os.path.dirname(os.path.abspath(__file__))
CAP, N, SEED = 32_000, 16, 1030
idx = json.load(open(os.path.join(SRC, "index.json")))
val = set(json.load(open(os.path.join(SRC, "split.json")))["val_tasks"])
z = np.load(os.path.join(SRC, "histories.npz"))
cands = [r for r in idx if r["task"] not in val and r["name"] in z.files and len(z[r["name"]]) >= CAP]
by_task = {}
for r in cands:                                   # one history a task
    by_task.setdefault(r["task"], r)
pool = sorted(by_task.values(), key=lambda r: r["name"])
random.Random(SEED).shuffle(pool)
pick = pool[:N]
arrays = {r["name"]: np.asarray(z[r["name"]][:CAP], dtype=np.int32) for r in pick}
np.savez(os.path.join(OUT, "xq32_histories.npz"), **arrays)
plan = {"source": SRC, "seed": SEED, "cap": CAP,
        "phases": {"late32": {"histories": [{"name": r["name"], "task": r["task"], "tokens": CAP} for r in pick]}}}
json.dump(plan, open(os.path.join(OUT, "xq32_plan.json"), "w"), indent=1)
print(f"{len(pick)} histories x {CAP} tokens = {len(pick) * CAP:,} tokens; tasks {sorted(r['task'] for r in pick)}")
