"""Logprob recording for Flash-Next harness rounds (docs/reports/1024_2113_fn_logprobs/DESIGN.md).

The recording is one notebook cell, `fn_record_cell.py` beside this file, run between the round scheduler (cell
1b17dcde) and the round cell (e03c3ecd) of notebook 1018-harness-bench. It is off unless LOGPROBS=1; with it, every
harness turn asks the server for its per-token confidence (fork agi2/feat/fn-conf, server env SGLANG_AGI2_FN_CONF=1)
and every attempt writes <batch>/lp/<aid>.npz. The shared notebook is not edited: `insert` writes a copy with the cell
(id CELL_ID) right before the round cell, as kv_sibfix.py writes its copy, and the round runs that copy with CELL_ID in
its --run list.

    python -m agi2_modal_harness.logprobs.fn_record insert SRC.ipynb DST.ipynb
    python -m agi2_modal_harness.logprobs.fn_record cells "<the round's --run list>"   # the list with CELL_ID added
    python -m agi2_modal_harness.logprobs.fn_record read RUN_DIR/<batch>/lp              # summary of a round's npz
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CELL_FILE = HERE / "fn_record_cell.py"
CELL_ID = "fnlprec1"
AFTER, BEFORE = "1b17dcde", "e03c3ecd"   # the scheduler cell, the round cell


def cell_source() -> str:
    return CELL_FILE.read_text(encoding="utf-8")


def insert(src: Path, dst: Path, after: str = AFTER, before: str = BEFORE) -> int:
    """Copy notebook `src` to `dst` with the recording cell right before cell `before` (the round cell), which must come
    after cell `after`. Fails closed when the anchors moved, and replaces an older copy of the cell. The 11-01 probe
    notebooks: after 7a7b1aaa (repl_attempt, SessionConv before it), before a6c36a51 (the solve)."""
    nb = json.loads(src.read_text(encoding="utf-8"))
    cells = [c for c in nb["cells"] if c.get("id") != CELL_ID]
    ids = [c.get("id") for c in cells]
    if after not in ids or before not in ids or ids.index(after) > ids.index(before):
        print(f"[fn_record] {src}: expected cell {after} before cell {before}; not writing", file=sys.stderr)
        return 3
    src_text = cell_source()
    compile(src_text, CELL_ID, "exec")
    cell = {"cell_type": "code", "execution_count": None, "id": CELL_ID, "metadata": {}, "outputs": [],
            "source": src_text.splitlines(keepends=True)}
    cells.insert(ids.index(before), cell)
    nb["cells"] = cells
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(f"[fn_record] wrote {dst}: cell {CELL_ID} ({len(src_text):,} characters) before {before}")
    return 0


def with_cell(run_cells: str, before: str = BEFORE) -> str:
    """A round's --run list with CELL_ID before the round cell (harness_nb runs them in notebook order anyway)."""
    cells = [c for c in run_cells.split(",") if c and c != CELL_ID]
    if before not in cells:
        raise SystemExit(f"[fn_record] the run list has no {before}: {run_cells}")
    cells.insert(cells.index(before), CELL_ID)
    return ",".join(cells)


def read(lp_dir: Path) -> dict:
    """A round's recording at a glance: attempts, tokens, turns without rows, misaligned turns, NaN rows."""
    import numpy as np

    out = dict(files=0, tokens=0, turns=0, turns_without_rows=0, misaligned=0, replaced=0, nan_rows=0, bytes=0)
    for f in sorted(lp_dir.glob("*.npz")):
        z = np.load(f)
        tt = z["turns"]
        out["files"] += 1
        out["bytes"] += f.stat().st_size
        out["tokens"] += int(tt[:, 3].sum()) if len(tt) else 0
        out["turns"] += len(tt)
        out["turns_without_rows"] += int((tt[:, 5] == 0).sum()) if len(tt) else 0
        out["misaligned"] += int(((tt[:, 5] == 1) & (tt[:, 6] == 0)).sum()) if len(tt) else 0
        out["replaced"] += int(tt[:, 7].sum()) if len(tt) else 0
        out["nan_rows"] += int(np.isnan(z["val"][:, 0]).sum())
    return out


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) in (3, 5) and argv[0] == "insert":   # insert SRC DST [AFTER BEFORE]
        return insert(Path(argv[1]), Path(argv[2]), *argv[3:5])
    if len(argv) in (2, 3) and argv[0] == "cells":    # cells "<run list>" [BEFORE]
        print(with_cell(*argv[1:3]))
        return 0
    if len(argv) == 2 and argv[0] == "read":
        print(json.dumps(read(Path(argv[1])), indent=1))
        return 0
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
