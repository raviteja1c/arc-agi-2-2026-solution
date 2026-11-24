"""The G4 slice of the training-task harness set: the main session's builder (make_fntrain_set.py, the same bans and
seed) at --n 400 into a staging directory, so the repo's 200-task fntrain_*.json stay as they are. The first 200 tasks
equal the repo's set (random.Random(1025).sample's pool method draws the same first 200); the G4 rounds take tasks
200-399 with NIGHT_TASKS (disjoint from the 4 x L4 round's 0-199). Writes <out>/data/harness_bench/fntrain_*.json and
<out>/g4_slices.json ({"dfg": tasks 200-299, "dft": tasks 300-399}).
  python make_fntrain_g4_set.py <out dir>
"""
import importlib.util
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
SRC = ROOT / "docs" / "reports" / "1025_1016_fn_decode_3x" / "make_fntrain_set.py"


def main():
    out = Path(sys.argv[1])
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 400
    spec = importlib.util.spec_from_file_location("make_fntrain_set", SRC)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.BENCH = out / "data" / "harness_bench"
    sys.argv = [str(SRC), "--n", str(n), "--seed", "1025"]
    mod.main()
    fam = json.loads((mod.BENCH / "fntrain_families.json").read_text(encoding="utf-8"))
    tasks = fam["tasks"]
    repo = json.loads((ROOT / "data" / "harness_bench" / "fntrain_families.json").read_text(encoding="utf-8"))["tasks"]
    if tasks[:200] != repo:
        raise SystemExit("fail closed: the first 200 tasks differ from the repo's fntrain set")
    slices = {"dfg": tasks[200:300], "dft": tasks[300:400]}
    # follow-on rounds once a box finishes its first slice: 200 more tasks each (needs n >= 800)
    if len(tasks) >= 800:
        slices.update(dfg2=tasks[400:600], dft2=tasks[600:800])
    (out / "g4_slices.json").write_text(json.dumps(slices, indent=1), encoding="utf-8")
    ev = set(json.loads((ROOT / "data" / "original_competition_data" / "arc-agi_evaluation_challenges.json")
                        .read_text(encoding="utf-8")))
    if ev & set(tasks):
        raise SystemExit("fail closed: an evaluation id in the set")
    print(f"400 tasks; first 200 = the repo's set; G4 slices dfg {len(slices['dfg'])}, dft {len(slices['dft'])} "
          f"(first {slices['dfg'][:2]}, {slices['dft'][:2]}); no evaluation id")


if __name__ == "__main__":
    main()
