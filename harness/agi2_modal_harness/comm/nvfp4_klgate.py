"""End-to-end gate for expert re-rounding: how close each served build's next-token distributions sit to the BF16
model's, teacher-forced on the held-out harness histories (capture phase `gate`, tasks QG never saw).

  score:   one running server -> <out>.npz with, per position past --start of every gate history, the top-K token ids
           and logprobs and the logprob of the history's own next token.
  compare: KL(BF16 || build) per position over BF16's top-K support (a build's logprob for a token outside its own
           top-K is taken as its K-th logprob, an upper bound, so that term is underestimated; the share of BF16 mass
           this touches is reported), top-1 agreement with BF16, and |delta| of the history token's logprob; per task
           and pooled.

  python -m agi2_modal_harness.comm.nvfp4_klgate score --base http://127.0.0.1:30000 --hist H.npz --plan P.json \
      --out bf16.npz [--start 640] [--topk 20]
  python -m agi2_modal_harness.comm.nvfp4_klgate compare --ref bf16.npz --build radix=radix.npz --build qg=qg.npz \
      --out kl.json
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.request

import numpy as np


def post(base: str, body: dict, timeout: float = 1800) -> dict:
    req = urllib.request.Request(base + "/generate", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def score(a) -> int:
    z = np.load(a.hist)
    plan = json.load(open(a.plan))
    hs = plan["phases"]["gate"]["histories"]
    out = {}
    t0 = time.time()
    for h in hs:
        ids = [int(x) for x in z[h["name"]]]
        r = post(a.base, {"input_ids": ids, "return_logprob": True, "logprob_start_len": a.start,
                          "top_logprobs_num": a.topk, "sampling_params": {"temperature": 0.0, "max_new_tokens": 1}})
        mi = r["meta_info"]
        tok = mi["input_token_logprobs"]
        top = mi["input_top_logprobs"]
        # entry i scores the token at position start + i given the tokens before it; entry 0 has no top list when
        # start is 0, which never happens here (start 640)
        n = len(tok)
        if n != len(ids) - a.start or len(top) != n:
            raise SystemExit(f"{h['task']}: got {n} token / {len(top)} top entries, want {len(ids) - a.start}")
        lp_tok = np.array([x[0] if x[0] is not None else np.nan for x in tok], np.float32)
        ids_top = np.full((n, a.topk), -1, np.int32)
        lp_top = np.full((n, a.topk), -np.inf, np.float32)
        for i, row in enumerate(top):
            for j, x in enumerate(row or []):
                lp_top[i, j], ids_top[i, j] = x[0], x[1]
        out[f"{h['task']}__lp_tok"], out[f"{h['task']}__ids_top"], out[f"{h['task']}__lp_top"] = lp_tok, ids_top, lp_top
        print(f"[kl-score] {h['task']}: {n} positions, mean history-token logprob {np.nanmean(lp_tok):.4f}; "
              f"elapsed {time.time() - t0:.0f} s", flush=True)
    out["tasks"] = np.array([h["task"] for h in hs])
    np.savez_compressed(a.out, **out)
    print(f"[PHASE] score done in {time.time() - t0:.0f} s -> {a.out}", flush=True)
    return 0


def kl_one(ref_ids, ref_lp, ids, lp):
    """Per position: KL over ref's top-K support, the ref mass whose token is missing from the build's top-K, and
    whether the top-1 tokens agree."""
    n, k = ref_ids.shape
    kl = np.zeros(n)
    miss = np.zeros(n)
    for i in range(n):
        ok_q = np.isfinite(lp[i]) & (ids[i] >= 0)  # padded entries are (-1, -inf)
        q = dict(zip(ids[i][ok_q].tolist(), lp[i][ok_q].tolist()))
        floor = float(lp[i][ok_q].min()) if ok_q.any() else -30.0
        ok_p = np.isfinite(ref_lp[i]) & (ref_ids[i] >= 0)
        rid, rlp = ref_ids[i][ok_p], ref_lp[i][ok_p]
        p = np.exp(rlp)
        lq = np.array([q.get(t, floor) for t in rid.tolist()])
        kl[i] = float(np.sum(p * (rlp - lq)))
        miss[i] = float(np.sum(p[[t not in q for t in rid.tolist()]]))
    agree = ref_ids[:, 0] == ids[:, 0]
    return kl, miss, agree


def compare(a) -> int:
    ref = np.load(a.ref)
    tasks = [str(t) for t in ref["tasks"]]
    builds = dict(b.split("=", 1) for b in a.build)
    res = {"ref": a.ref, "builds": list(builds), "per_task": {}, "pooled": {}}
    pooled = {b: {"kl": [], "miss": [], "agree": [], "dtok": []} for b in builds}
    for t in tasks:
        res["per_task"][t] = {}
        for b, path in builds.items():
            z = np.load(path)
            kl, miss, agree = kl_one(ref[f"{t}__ids_top"], ref[f"{t}__lp_top"], z[f"{t}__ids_top"], z[f"{t}__lp_top"])
            dtok = np.abs(ref[f"{t}__lp_tok"] - z[f"{t}__lp_tok"])
            conf = np.exp(ref[f"{t}__lp_top"][:, 0]) > 0.9  # BF16's top token holds > 90 % of the mass
            res["per_task"][t].setdefault("_bf16_confident_share", float(conf.mean()))
            res["per_task"][t][b] = {"positions": int(len(kl)), "kl_mean": float(kl.mean()),
                                     "kl_confident": float(kl[conf].mean()), "kl_uncertain": float(kl[~conf].mean()),
                                     "top1_agree_confident": float(agree[conf].mean()),
                                     "kl_p99": float(np.quantile(kl, 0.99)), "top1_agree": float(agree.mean()),
                                     "missing_ref_mass": float(miss.mean()), "abs_dlogprob_tok": float(np.nanmean(dtok)),
                                     "hist_tok_logprob": float(np.nanmean(z[f"{t}__lp_tok"]))}
            for key, v in (("kl", kl), ("miss", miss), ("agree", agree), ("dtok", dtok)):
                pooled[b][key].append(v)
            print(f"[kl] {t} {b}: KL {kl.mean():.5f} (p99 {np.quantile(kl, 0.99):.4f}), top-1 agree "
                  f"{agree.mean():.4f}, |dlogprob tok| {np.nanmean(dtok):.4f}, missing mass {miss.mean():.5f}",
                  flush=True)
    for b in builds:
        kl = np.concatenate(pooled[b]["kl"])
        res["pooled"][b] = {"positions": int(len(kl)), "kl_mean": float(kl.mean()),
                            "top1_agree": float(np.concatenate(pooled[b]["agree"]).mean()),
                            "missing_ref_mass": float(np.concatenate(pooled[b]["miss"]).mean()),
                            "abs_dlogprob_tok": float(np.nanmean(np.concatenate(pooled[b]["dtok"])))}
    base = res["pooled"].get("radix")
    if base:
        res["kl_vs_radix"] = {b: 1 - v["kl_mean"] / base["kl_mean"] for b, v in res["pooled"].items() if b != "radix"}
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"[PHASE] compare: pooled {json.dumps(res['pooled'])}; KL vs radix {res.get('kl_vs_radix')}", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("--base", default="http://127.0.0.1:30000")
    s.add_argument("--hist", required=True)
    s.add_argument("--plan", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--start", type=int, default=640)
    s.add_argument("--topk", type=int, default=20)
    c = sub.add_parser("compare")
    c.add_argument("--ref", required=True)
    c.add_argument("--build", action="append", required=True)
    c.add_argument("--out", required=True)
    a = ap.parse_args()
    return score(a) if a.cmd == "score" else compare(a)


if __name__ == "__main__":
    raise SystemExit(main())
