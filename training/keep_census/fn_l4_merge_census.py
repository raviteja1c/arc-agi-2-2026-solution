"""Merge Flash-Next's prose and code expert censuses into keep lists that serve both (Flash-Next on 4 x L4).

Round 1023_0833 showed the K = 320 model collapsing in code (`def def def ...`). Its keep list came from boot 8's
census, 20 plain-text training puzzles with no tools or code. This takes that prose census and a code census (P10
`--census-code`: training puzzles answered with a python tool call, thinking off, `comm/fn_l4_census_code.sh`). In each
layer it ranks experts by max(prose share, code share), so an expert that matters to either kind of text survives. It
keeps the top K and balances the list for EP like `fn_l4_balance_keep.py`. Load is estimated from the mean of the two
shares. It prints coverage of each kind for the old prose-only list and the merged one.

    python -m agi2_modal_harness.comm.fn_l4_merge_census PROSE.json CODE.json OUTDIR [--keep 320,384,448] [--ep 4]
        [--old KEEP.json ...]
"""
from __future__ import annotations

import argparse
import json
import os

from agi2_modal_harness.comm.fn_l4_balance_keep import balance


def shares(counts: list[list[float]]) -> list[list[float]]:
    return [[c / (sum(row) or 1.0) for c in row] for row in counts]


def coverage(keep: list[list[int]], share: list[list[float]]) -> tuple[float, float]:
    per = [sum(share[li][e] for e in ids) for li, ids in enumerate(keep)]
    return sum(per) / len(per), min(per)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("prose")
    ap.add_argument("code")
    ap.add_argument("outdir")
    ap.add_argument("--keep", default="320,384,448")
    ap.add_argument("--ep", type=int, default=4)
    ap.add_argument("--old", nargs="*", default=[], help="existing keep lists to score the same way")
    a = ap.parse_args()
    P = json.load(open(a.prose, encoding="utf-8"))
    C = json.load(open(a.code, encoding="utf-8"))
    sp, sc = shares(P["counts"]), shares(C["counts"])
    layers, experts = len(sp), len(sp[0])
    assert len(sc) == layers and len(sc[0]) == experts, "the two censuses differ in shape"
    print(f"{layers} layers x {experts} experts; prose tasks {len(P.get('tasks', []))}, "
          f"code tasks {len(C.get('code_tasks', []) or C.get('tasks', []))}")
    for path in a.old:
        old = json.load(open(path, encoding="utf-8"))["layers"]
        k = len(old[0])
        cp, cc = coverage(old, sp), coverage(old, sc)
        top_code = [sorted(range(experts), key=lambda e: -sc[li][e])[:64] for li in range(layers)]
        cut = [sum(1 for e in top_code[li] if e not in set(old[li])) for li in range(layers)]
        print(f"old {os.path.basename(path)} (K {k}): prose coverage mean {cp[0]:.4f} (worst layer {cp[1]:.4f}), "
              f"code coverage mean {cc[0]:.4f} (worst {cc[1]:.4f}); of each layer's top-64 code experts it cut "
              f"{sum(cut) / layers:.1f} on average (max {max(cut)})")
    os.makedirs(a.outdir, exist_ok=True)
    for k in [int(x) for x in a.keep.split(",")]:
        keep, loads = [], []
        for li in range(layers):
            score = [max(sp[li][e], sc[li][e]) for e in range(experts)]
            ids = sorted(range(experts), key=lambda e: -score[e])[:k]
            mix = [(sp[li][e] + sc[li][e]) / 2 for e in range(experts)]
            order, load = balance(ids, mix, a.ep)
            keep.append(order)
            loads.append(max(load) / (sum(load) / a.ep))
        cp, cc = coverage(keep, sp), coverage(keep, sc)
        out = os.path.join(a.outdir, f"keep_{k}_code_bal{a.ep}.json")
        json.dump({"layers": keep, "keep": k, "source": f"max(prose, code) share: {a.prose} + {a.code}",
                   "tasks": P.get("tasks", []), "code_tasks": C.get("code_tasks", []), "balanced_for_ep": a.ep,
                   "balanced_from": "mean of prose and code shares"}, open(out, "w", encoding="utf-8"))
        print(f"new K {k}: prose coverage mean {cp[0]:.4f} (worst {cp[1]:.4f}), code coverage mean {cc[0]:.4f} "
              f"(worst {cc[1]:.4f}); EP max rank load / mean {sum(loads) / len(loads):.3f} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
