"""Training histories for the DFlash2 fine-tune from recorded harness rounds on ARC-AGI-2 TRAINING tasks (the
fntrain set; data rounds 1025_2055 on G4 and the main session's 4 x L4 round). Same reading as make_g4_inputs.py
does for the gate's histories: per attempt npz (logprobs/fn_record_cell.py) the main conversation's exact token
history, its generated turn spans (turns with n > 1), and the served per-token rows placed at their positions:
`__lps` (lp_samp, log q_target of the written token under the verify's own sampling distribution) and `__j` (its
place in its verify step; 0 = a step start).
Fail closed: every attempt's task must be an ARC-AGI-2 training task and in no evaluation file or bench children list
(memory children-never-train-weights), and the recorded rows must equal the history's tokens.
Outputs in <out>: histories.npz (<name>, <name>__spans [k, 2], <name>__lps, <name>__j), requests/<name>.json (spans,
temperature, source), index.json. Usage:
  python make_harness_histories.py <out dir> <prefix> <temperature> <glob of lp/*.npz> [<glob> ...]
"""
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), *[".."] * 4))
DATA = os.path.join(ROOT, "data", "original_competition_data")
BENCH = os.path.join(ROOT, "data", "harness_bench")


def banned_and_training():
    tr = set(json.load(open(os.path.join(DATA, "arc-agi_training_challenges.json"))))
    ban = set(json.load(open(os.path.join(DATA, "arc-agi_evaluation_challenges.json"))))
    for f in glob.glob(os.path.join(BENCH, "*_families.json")):
        if os.path.basename(f).startswith("fntrain"):
            continue
        fam = json.load(open(f))
        ban |= {t for t, k in (fam.get("kind") or {}).items() if k == "child"}
        for kids in (fam.get("children") or {}).values():
            ban |= set(kids)
    return tr, ban


def main():
    out, prefix, temp = sys.argv[1], sys.argv[2], float(sys.argv[3])
    files = sorted({f for pat in sys.argv[4:] for f in glob.glob(pat)})
    if not files:
        raise SystemExit("no recorded attempts")
    tr, ban = banned_and_training()
    os.makedirs(os.path.join(out, "requests"), exist_ok=True)
    arrays, index = {}, []
    skipped = []
    for f in files:
        z = np.load(f, allow_pickle=True)
        meta = json.loads(str(z["meta"]))
        aid = meta["aid"]
        task = aid.split("_")[0]
        if task not in tr or task in ban:
            raise SystemExit(f"fail closed: {f}: task {task} is not a training task or is banned (eval / child)")
        if "main" not in meta["roles"]:
            skipped.append((aid, "no main conversation"))
            continue
        main_c = meta["roles"].index("main")
        hist = z[f"hist_{main_c}"].astype(np.int32)
        spans = [[int(s), int(s + n)] for (ci, ti, s, n, *_r) in z["turns"] if ci == main_c and n > 1]
        if not spans:
            skipped.append((aid, "no generated turn"))
            continue
        lps = np.full(len(hist), np.nan, dtype=np.float32)
        jj = np.full(len(hist), -1, dtype=np.int8)
        val = z["val"]
        for (ci, ti, s, n, row0, has, ok, repl, *_r) in z["turns"]:
            if ci != main_c or not has:
                continue
            if repl or not np.array_equal(hist[s:s + n], z["tok"][row0:row0 + n]):
                raise SystemExit(f"fail closed: {f}: turn {ti} rows do not match the history")
            lps[s:s + n] = val[row0:row0 + n, 1]
            jj[s:s + n] = np.nan_to_num(val[row0:row0 + n, 11], nan=-1).astype(np.int8)
        name = f"{prefix}__{aid}"
        if name in arrays:
            raise SystemExit(f"fail closed: {name} twice")
        arrays[name] = hist
        arrays[name + "__spans"] = np.asarray(spans, dtype=np.int64)
        arrays[name + "__lps"] = lps
        arrays[name + "__j"] = jj
        gen = int(sum(e - s for s, e in spans))
        json.dump(dict(spans=spans, temperature=temp, aid=aid, task=task, tokens=int(len(hist)), gen=gen,
                       src=os.path.relpath(f, ROOT).replace(os.sep, "/") if f.startswith(ROOT) else f),
                  open(os.path.join(out, "requests", f"{name}.json"), "w"))
        index.append(dict(name=name, tokens=int(len(hist)), gen=gen, T=temp, task=task))
    np.savez_compressed(os.path.join(out, "histories.npz"), **arrays)
    json.dump(index, open(os.path.join(out, "index.json"), "w"), indent=1)
    tasks = sorted({r["task"] for r in index})
    # a task-level split (every history of a task on one side), seeded: VAL_TASKS (env, default 10) for validation
    import random

    n_val = int(os.environ.get("VAL_TASKS", "10"))
    val_tasks = set(random.Random(1025).sample(tasks, min(n_val, max(len(tasks) - 1, 0))))
    split = dict(seed=1025, val_tasks=sorted(val_tasks),
                 val=[r["name"] for r in index if r["task"] in val_tasks],
                 train=[r["name"] for r in index if r["task"] not in val_tasks])
    json.dump(split, open(os.path.join(out, "split.json"), "w"), indent=1)
    print(f"split: {len(split['train'])} train / {len(split['val'])} val histories ({len(val_tasks)} val tasks)")
    print(f"{len(index)} histories from {len(files)} attempts ({len(tasks)} training tasks), "
          f"{sum(r['tokens'] for r in index):,} tokens, {sum(r['gen'] for r in index):,} generated; "
          f"skipped {len(skipped)}: {skipped[:5]}")


if __name__ == "__main__":
    main()
