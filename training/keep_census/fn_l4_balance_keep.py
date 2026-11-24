"""Reorder an expert keep list so each EP rank's experts carry equal routing mass (Flash-Next on 4 x L4).

With SGLANG_AGI2_FN_EXPERT_KEEP a kept expert's new id is its position in the layer's list, and EP gives rank r the
contiguous ids [r*K/ep, (r+1)*K/ep). An ascending list therefore places experts by checkpoint id, and the census shows
it unbalanced: in boot 19's profile rank 3's Marlin MoE ran 1.6-1.8 ms a step lighter than rank 0's, and the slowest
rank sets every step's pace (torch second look of 1023_0432, Lead 5: 2.8 ms a step of MoE imbalance). This reorders
each layer's kept ids by longest-processing-time greedy over the census counts into ep equal-size bins, then lists
bin 0, bin 1, ...; each bin ascending. The model is unchanged: the same experts, the router rows in the same order.
Needs SGLANG_AGI2_FN_EXPERT_KEEP_ANY_ORDER=1 (the loader otherwise requires ascending ids).

    python -m agi2_modal_harness.comm.fn_l4_balance_keep CENSUS.json KEEP.json OUT.json [--ep 4]
"""
from __future__ import annotations

import argparse
import json


def balance(ids: list[int], counts: list[float], ep: int) -> tuple[list[int], list[float]]:
    cap = len(ids) // ep
    bins: list[list[int]] = [[] for _ in range(ep)]
    load = [0.0] * ep
    for e in sorted(ids, key=lambda i: -counts[i]):
        r = min((r for r in range(ep) if len(bins[r]) < cap), key=lambda r: load[r])
        bins[r].append(e)
        load[r] += counts[e]
    return [e for b in bins for e in sorted(b)], load


def rank_loads(order: list[int], counts: list[float], ep: int) -> list[float]:
    cap = len(order) // ep
    return [sum(counts[e] for e in order[r * cap:(r + 1) * cap]) for r in range(ep)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("census")
    ap.add_argument("keep")
    ap.add_argument("out")
    ap.add_argument("--ep", type=int, default=4)
    a = ap.parse_args()
    counts = json.load(open(a.census, encoding="utf-8"))["counts"]
    keep = json.load(open(a.keep, encoding="utf-8"))
    layers = keep["layers"]
    if len(layers[0]) % a.ep:
        raise SystemExit(f"keep size {len(layers[0])} does not divide by ep {a.ep}")
    out, before, after = [], [], []
    for li, ids in enumerate(layers):
        c = counts[li]
        tot = sum(c[e] for e in ids) or 1.0
        b = rank_loads(sorted(ids), c, a.ep)
        order, load = balance(ids, c, a.ep)
        assert sorted(order) == sorted(ids) and len(set(order)) == len(ids)
        out.append(order)
        before.append(max(b) / (tot / a.ep))
        after.append(max(load) / (tot / a.ep))
    res = dict(keep, layers=out, balanced_for_ep=a.ep, balanced_from=a.census)
    json.dump(res, open(a.out, "w", encoding="utf-8"))
    print(f"{len(out)} layers, {len(out[0])} experts each, ep {a.ep}")
    print(f"max rank load / mean, ascending order: mean {sum(before) / len(before):.3f}, worst {max(before):.3f}")
    print(f"max rank load / mean, balanced order : mean {sum(after) / len(after):.3f}, worst {max(after):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
