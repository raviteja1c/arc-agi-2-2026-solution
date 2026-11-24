"""Summarise an expert re-rounding build's manifests for a run note.

QS manifests (one per shard, from nvfp4_reround): per-matrix weight error vs BF16 for RadixArk and the new rounding,
its per-expert gain (mean, min) and the share of block scales changed. QG manifests (one per layer, on the
experts-0000-0127 shard, from nvfp4_gptq): per expert and matrix the calib output proxy for RadixArk, QS and QG and
which one was written.

  python -m agi2_modal_harness.comm.nvfp4_manifest_sum /vol/cache/dev/fnxq/qs1
"""
from __future__ import annotations

import glob
import json
import os
import sys


def main() -> int:
    d = sys.argv[1]
    ms = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(d, "manifest", "*.json")))]
    print(f"{len(ms)} manifests in {d}")
    qs = [m for m in ms if "matrices" in m]
    qg = [m for m in ms if "experts" in m]
    if qs:
        rad = [v["err_radix_mean"] for m in qs for v in m["matrices"].values()]
        new = [v["err_new_mean"] for m in qs for v in m["matrices"].values()]
        gmin = min(v["gain_min"] for m in qs for v in m["matrices"].values())
        rep = min(v["repro_rtn"] for m in qs for v in m["matrices"].values())
        sch = sum(v["scales_changed"] for m in qs for v in m["matrices"].values()) / len(rad)
        print(f"QS: {len(qs)} shards, {len(rad)} matrix groups; weight error vs BF16 radix {sum(rad) / len(rad):.5f} "
              f"-> {sum(new) / len(new):.5f} ({1 - sum(new) / sum(rad):.1%} lower); worst single expert gain "
              f"{gmin:.1%}; min RTN reproduction {rep:.4f}; block scales changed {sch:.1%}")
    if qg:
        used, rows = {}, []
        for m in qg:
            for e, mats in m["experts"].items():
                for mat, v in mats.items():
                    used[v["used"]] = used.get(v["used"], 0) + 1
                    rows.append(v["rows"])
        tr = sum(m["totals"]["radix"] for m in qg)
        ts = sum(m["totals"]["qs"] for m in qg)
        tq = sum(m["totals"]["qg"] for m in qg)
        rows.sort()
        print(f"QG: {len(qg)} layers; matrices written by method {used}; calib rows per served matrix min "
              f"{rows[0]} median {rows[len(rows) // 2]} max {rows[-1]}; calib proxy summed over layers radix {tr:.4g} "
              f"qs {ts:.4g} ({1 - ts / tr:.1%} lower) qg {tq:.4g} ({1 - tq / tr:.1%} lower)")
        qs_only = sum(len(m.get("qs_only") or []) for m in qg)
        print(f"QG: experts left to QS (unserved or < min rows): {qs_only}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
