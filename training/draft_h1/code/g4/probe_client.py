"""Probe client: sends every recorded history to the probe server as one teacher-forced prefill (max_new_tokens 1,
custom_params agi2_probe), one at a time, and checks the probe wrote its outputs.

Fail-closed preflight: the first CALIB histories (exp1, moderate size) run first; then sim.py compares, at the served
steps' own anchors, the probe's predicted tokens per step at block 4 with what the 4 x L4 server realized on the same
steps. Outside [LO, HI] the client stops (the probe would not be measuring the served draft), unless --no-gate.
Progress lines carry the count, tokens done, rate and ETA.
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
import requests

CALIB = ["exp1__58f5dbd5_0_g1_s0", "exp1__58490d8a_0_g1_s0", "exp1__WITHHELD_0_g1_s0"]  # package edit 2026-11-17: the third calibration history is one of the private hand-made validation puzzles; its id is withheld
LO, HI = 0.90, 1.10


def say(msg):
    print(f"[client {time.strftime('%H:%M:%S', time.gmtime())} UTC] {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--probe", required=True)
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--only", default="", help="comma list of names (default: all)")
    ap.add_argument("--no-gate", action="store_true")
    ap.add_argument("--server", default="http://127.0.0.1:30000")
    a = ap.parse_args()
    z = np.load(os.path.join(a.inputs, "histories.npz"))
    index = json.load(open(os.path.join(a.inputs, "index.json")))
    by = {r["name"]: r for r in index}
    names = [n for n in a.only.split(",") if n] or (
        CALIB + sorted([r["name"] for r in index if r["name"] not in CALIB],
                       key=lambda n: (not n.startswith("exp1"), by[n]["tokens"])))
    total = sum(by[n]["tokens"] for n in names)
    done_tok = 0
    t0 = time.time()
    prog = open(os.path.join(a.run, "client_progress.jsonl"), "a")
    for i, n in enumerate(names):
        out_json = os.path.join(a.probe, "out", f"{n}.json")
        if os.path.exists(out_json):
            done_tok += by[n]["tokens"]
            say(f"{i + 1}/{len(names)} {n}: already done")
            continue
        ids = z[n].astype(int).tolist()
        req = json.load(open(os.path.join(a.probe, "requests", f"{n}.json")))
        body = {"input_ids": ids, "sampling_params": {"max_new_tokens": 1, "temperature": req["temperature"],
                                                      "top_p": 0.95, "top_k": 20, "custom_params": {"agi2_probe": n}}}
        ts = time.time()
        r = requests.post(f"{a.server}/generate", json=body, timeout=7200)
        dt = time.time() - ts
        ok = r.status_code == 200 and os.path.exists(out_json)
        info = json.load(open(out_json)) if os.path.exists(out_json) else {}
        done_tok += len(ids)
        el = time.time() - t0
        rate = done_tok / max(el, 1e-6)
        eta = (total - done_tok) / max(rate, 1e-6)
        say(f"{i + 1}/{len(names)} {n}: {len(ids):,} tokens, {info.get('anchors', 0):,} anchors in {dt:.1f} s "
            f"(probe {info.get('probe_s', '?')} s) {'ok' if ok else 'FAILED http ' + str(r.status_code)}; "
            f"{done_tok:,} of {total:,} tokens ({100 * done_tok / total:.1f} %), {rate:,.0f} tok/s, ETA {eta / 60:.0f} min")
        prog.write(json.dumps(dict(i=i, name=n, tokens=len(ids), s=round(dt, 1), ok=ok, info=info, t=time.time())) + "\n")
        prog.flush()
        if not ok:
            say(f"probe output missing for {n}: {r.text[:300]}")
            return 4
        if i + 1 == len(CALIB) and names[:len(CALIB)] == CALIB:
            cj = os.path.join(a.run, "calib.json")
            cmd = [sys.executable, os.path.join(a.inputs, "sim.py"), "--probe", os.path.join(a.probe, "out"),
                   "--inputs", a.inputs, "--names", ",".join(CALIB), "--blocks", "4,8,16", "--json", cj]
            say("preflight: " + " ".join(cmd[1:]))
            p = subprocess.run(cmd, capture_output=True, text=True)
            print(p.stdout[-4000:], p.stderr[-2000:], flush=True)
            c = json.load(open(cj)).get("calibration_b4") if os.path.exists(cj) else None
            if c is None:
                say("preflight FAILED: no calibration")
                return 5
            ratio = c["pred"] / c["real"]
            say(f"preflight: predicted {c['pred']:.3f} vs served {c['real']:.3f} tok/step at the served anchors, "
                f"ratio {ratio:.3f} (gate {LO}-{HI})")
            if not (LO <= ratio <= HI) and not a.no_gate:
                say("preflight FAILED: the probe does not reproduce the served acceptance; stopping")
                return 6
    say(f"all {len(names)} done in {(time.time() - t0) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
