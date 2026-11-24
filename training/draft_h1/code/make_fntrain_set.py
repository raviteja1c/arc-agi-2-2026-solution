"""A harness task set of ARC-AGI-2 TRAINING tasks, for generating draft-model training data (2026-10-25).

The DFlash2 fine-tune (dflash2 session) needs Flash-Next harness conversations: multi-turn, tool calls, the served
sampling. Its direct single-turn data gained +3.8 % accept on held-out direct histories and nothing on harness
histories. Weights may be trained only on public training tasks, never on evaluation tasks or on our hand-made eval
children (the user, 2026-10-25: children never train weights). So this set holds training tasks only, and fails closed if
any id from the evaluation file or from any local bench set's children appears.

A seeded random sample (no difficulty filter: the draft should see the harness's whole range). The files have the
shape of fnhard27's, so the notebook cell `hbfntrain` reads them the way `hbfnhard` reads fnhard27:
  data/harness_bench/fntrain_challenges.json   task -> {train, test}
  data/harness_bench/fntrain_solutions.json    task -> [output grid per test input]
  data/harness_bench/fntrain_families.json     parents (= every task), tasks (queue order), family, kind ('parent')

    python docs/reports/1025_1016_fn_decode_3x/make_fntrain_set.py [--n 200] [--seed 1025]
"""
from __future__ import annotations

import argparse
import glob
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data" / "original_competition_data"
BENCH = ROOT / "data" / "harness_bench"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1025)
    a = ap.parse_args()
    tr_c = json.loads((DATA / "arc-agi_training_challenges.json").read_text(encoding="utf-8"))
    tr_s = json.loads((DATA / "arc-agi_training_solutions.json").read_text(encoding="utf-8"))
    ev = set(json.loads((DATA / "arc-agi_evaluation_challenges.json").read_text(encoding="utf-8")))
    # Every id any local bench set calls a child, or that is not a training id there (eval parents, children).
    banned = set(ev)
    for f in glob.glob(str(BENCH / "*_families.json")):
        if Path(f).name.startswith("fntrain"):
            continue
        fam = json.loads(Path(f).read_text(encoding="utf-8"))
        kinds = fam.get("kind") or {}
        banned |= {t for t, k in kinds.items() if k == "child"}
        for p, kids in (fam.get("children") or {}).items():
            banned |= set(kids)
        banned |= {t for t in (fam.get("family") or {}) if t not in tr_c}
    pool = sorted(t for t in tr_c if t not in banned)
    assert len(pool) >= a.n, f"only {len(pool)} training tasks left"
    rng = random.Random(a.seed)
    tasks = rng.sample(pool, a.n)
    # Fail closed: training ids only, none from the evaluation file or any child list.
    assert all(t in tr_c and t in tr_s for t in tasks), "a task is not a training task"
    assert not (set(tasks) & banned), f"banned ids in the set: {sorted(set(tasks) & banned)[:5]}"
    assert all(len(tr_s[t]) == len(tr_c[t]["test"]) for t in tasks), "a solution count differs from the tests"
    ch = {t: tr_c[t] for t in tasks}
    so = {t: tr_s[t] for t in tasks}
    fam = dict(parents=list(tasks), tasks=list(tasks), family={t: t for t in tasks}, kind={t: "parent" for t in tasks},
               source=f"arc-agi_training_challenges.json, random.Random({a.seed}).sample of {len(pool)} ids, n={a.n}",
               purpose="draft-model training data (harness conversations); training tasks only, never eval or children")
    BENCH.mkdir(parents=True, exist_ok=True)
    for name, obj in (("challenges", ch), ("solutions", so), ("families", fam)):
        p = BENCH / f"fntrain_{name}.json"
        p.write_text(json.dumps(obj), encoding="utf-8", newline="\n")
    n_out = sum(len(ch[t]["test"]) for t in tasks)
    print(f"fntrain: {len(tasks)} training tasks, {n_out} outputs, {len(banned)} ids banned (eval + children); "
          f"first {tasks[:3]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
