"""Capture client for the DFlash2 draft fine-tune (decode-3x campaign, DFlash2 line): sends each generated history
(1025_1506_colab-g4-dflash2-gen, ARC-AGI-2 training tasks) to the probe server as ONE teacher-forced prefill, with
  - custom_params agi2_probe = <name>: the probe hook (SGLANG_AGI2_DFLASH_CAPTURE=1) writes the draft's projected
    context features of every position to $PROBE/out/<name>.feat.npy [N, hidden] (bf16 bits as int16);
  - return_logprob, logprob_start_len 0, top_logprobs_num 20: the served target's top-20 next-token logprobs at every
    input position, saved as $PROBE/out/<name>.teacher.npz: ids [N, 20] int32, lp [N, 20] float16 (row i = the
    distribution that predicts token i; row 0 is unused), tok_lp [N] float16 (logprob of the written token).
Fail-closed per history: the feature rows must equal the history length and every teacher row from 1 on must hold 20
entries whose ids include the written token's when its logprob ranks in the top 20. One plain progress line per
history (tokens, s, rate, ETA). Resumable: histories with both outputs are skipped.
  /tmp/sgl/bin/python capture_client.py --hist gen_histories.npz --probe /content/probe --names all|<a,b,...>
"""
import argparse
import json
import os
import time

import numpy as np
import requests

TOPK = 20


def say(msg):
    print(f"[capture {time.strftime('%H:%M:%S', time.gmtime())} UTC] {msg}", flush=True)


def parse_top(meta, n):
    """meta_info['input_top_logprobs'] -> (ids [n, 20] int32, lp [n, 20] float32). Entries are lists of
    [logprob, token_id, text] (text None); the first position has no prediction (None or empty)."""
    rows = meta.get("input_top_logprobs") or []
    if len(rows) != n:
        raise RuntimeError(f"input_top_logprobs has {len(rows)} rows for {n} tokens")
    ids = np.zeros((n, TOPK), dtype=np.int32)
    lp = np.full((n, TOPK), -np.inf, dtype=np.float32)
    for i, r in enumerate(rows):
        if not r:
            if i == 0:
                continue
            raise RuntimeError(f"empty top-logprob row at position {i}")
        if len(r) != TOPK:
            raise RuntimeError(f"top-logprob row {i} has {len(r)} entries")
        for k, e in enumerate(r):
            lp[i, k] = e[0]
            ids[i, k] = e[1]
    tok = meta.get("input_token_logprobs") or []
    tok_lp = np.array([(e[0] if e and e[0] is not None else np.nan) for e in tok], dtype=np.float32)
    if len(tok_lp) != n:
        raise RuntimeError(f"input_token_logprobs has {len(tok_lp)} rows for {n} tokens")
    return ids, lp, tok_lp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hist", required=True)
    ap.add_argument("--probe", required=True)
    ap.add_argument("--names", default="all")
    ap.add_argument("--server", default="http://127.0.0.1:30000")
    a = ap.parse_args()
    z = np.load(a.hist)
    # direct generations carry <name>__pl; harness histories (make_harness_histories.py) <name>__spans, __lps, __j
    names = sorted(k for k in z.files if not k.endswith(("__pl", "__spans", "__lps", "__j")))
    if a.names != "all":
        want = [n for n in a.names.split(",") if n]
        missing = sorted(set(want) - set(names))
        if missing:
            raise SystemExit(f"unknown histories: {missing}")
        names = want
    out = os.path.join(a.probe, "out")
    req_dir = os.path.join(a.probe, "requests")
    os.makedirs(out, exist_ok=True)
    os.makedirs(req_dir, exist_ok=True)
    total = sum(int(z[n].shape[0]) for n in names)
    done_tok = 0
    t0 = time.time()
    say(f"{len(names)} histories, {total:,} tokens")
    for i, n in enumerate(names):
        ids = z[n].astype(np.int64)
        N = int(ids.shape[0])
        if n + "__spans" in z.files:
            spans = [[int(s), int(e)] for s, e in z[n + "__spans"]]
        else:
            spans = [[int(z[n + "__pl"]) - 1, N]]
        feat_f = os.path.join(out, f"{n}.feat.npy")
        teach_f = os.path.join(out, f"{n}.teacher.npz")
        if os.path.exists(feat_f) and os.path.exists(teach_f):
            done_tok += N
            continue
        with open(os.path.join(req_dir, f"{n}.json"), "w") as f:
            json.dump({"spans": spans, "temperature": 1.0}, f)
        body = {"input_ids": ids.tolist(),
                "sampling_params": {"max_new_tokens": 1, "temperature": 1.0, "top_p": 0.95, "top_k": 20,
                                    "custom_params": {"agi2_probe": n}},
                "return_logprob": True, "logprob_start_len": 0, "top_logprobs_num": TOPK}
        ts = time.time()
        r = requests.post(f"{a.server}/generate", json=body, timeout=7200)
        if r.status_code != 200:
            raise SystemExit(f"{n}: HTTP {r.status_code}: {r.text[:300]}")
        meta = r.json().get("meta_info", {})
        tid, tlp, tok_lp = parse_top(meta, N)
        for _ in range(60):  # the probe writes feat.npy when the last chunk is processed
            if os.path.exists(feat_f):
                break
            time.sleep(1)
        feat = np.load(feat_f, mmap_mode="r")
        if feat.shape[0] != N:
            raise SystemExit(f"{n}: feature rows {feat.shape[0]} != history length {N}")
        # the written token, when its logprob is strictly above the 20th's, must appear among the ids (both come from
        # the same logits); a tie with the 20th may list another tied token (878187ab: 7 such positions stopped the
        # first capture at 215 of 433, run 1025_1900, with a check that counted ties)
        found = (tid[1:] == ids[1:, None]).any(1)
        strict = tok_lp[1:] > tlp[1:, -1] + 1e-3
        tie = (np.abs(tok_lp[1:] - tlp[1:, -1]) <= 1e-3) & ~found
        miss = int((strict & ~found).sum())
        if miss:
            raise SystemExit(f"{n}: {miss} positions whose written token ranks strictly inside the top 20 but is "
                             f"absent from the ids")
        if tie.any():
            say(f"{n}: {int(tie.sum())} positions where the written token ties the 20th and another tied token is "
                f"listed (kept)")
        np.savez(teach_f, ids=tid, lp=tlp.astype(np.float16), tok_lp=tok_lp.astype(np.float16))
        done_tok += N
        el = time.time() - t0
        rate = done_tok / max(el, 1e-6)
        say(f"{i + 1}/{len(names)} {n}: {N:,} tokens in {time.time() - ts:.1f} s, feat {tuple(feat.shape)}, "
            f"written token in top-20 {found.mean():.3f}; {done_tok:,}/{total:,} ({100 * done_tok / total:.1f} %), "
            f"{rate:,.0f} tok/s, ETA {(total - done_tok) / max(rate, 1e-6) / 60:.0f} min")
    say(f"done: {len(names)} histories, {total:,} tokens in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
