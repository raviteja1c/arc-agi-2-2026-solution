"""Capture client for the 32k plan, run as harness_nb's --probe once the server (with SGLANG_AGI2_MOE_CAPTURE_DIR set)
is up and idle: for each history, arm the capture with the phase tag, prefill it with max_new_tokens 1, disarm. Fails
closed after the first history unless every layer holds exactly its token count. Writes <cap>/<tag>/manifest.json
with each history's name, task and row range, so the scorer can split rows by position.

    python capture_client.py <cap dir> <histories.npz> <plan.json> [base=http://127.0.0.1:30000]
"""
import glob
import json
import os
import sys
import time
import urllib.request

import numpy as np

cap, hist, plan = sys.argv[1], sys.argv[2], sys.argv[3]
base = sys.argv[4] if len(sys.argv) > 4 else "http://127.0.0.1:30000"
z = np.load(hist)
plan = json.load(open(plan))
armed = os.path.join(cap, "ARMED")
os.makedirs(cap, exist_ok=True)


def rows(tag, layer):
    import torch
    return sum(torch.load(f, map_location="cpu")["x"].shape[0]
               for f in glob.glob(os.path.join(cap, tag, f"L{layer:02d}", "*.pt")))


for tag, ph in plan["phases"].items():
    man, start = [], 0
    for i, h in enumerate(ph["histories"]):
        ids = [int(t) for t in z[h["name"]]]
        open(armed, "w").write(tag)
        t0 = time.time()
        req = urllib.request.Request(base + "/generate", data=json.dumps(
            {"input_ids": ids, "sampling_params": {"max_new_tokens": 1, "temperature": 0.0}}).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=3600) as r:
            r.read()
        os.remove(armed)
        man.append({"name": h["name"], "task": h["task"], "start": start, "tokens": len(ids)})
        start += len(ids)
        print(f"[capture] {tag} {i + 1}/{len(ph['histories'])} {h['name']}: {len(ids)} tokens in "
              f"{time.time() - t0:.1f} s", flush=True)
        if i == 0:
            bad = [(layer, n) for layer in range(48) if (n := rows(tag, layer)) != len(ids)]
            if bad:
                print(f"[capture] FAILED: layer rows != {len(ids)}: {bad[:6]}", flush=True)
                sys.exit(3)
            print(f"[capture] preflight ok: all 48 layers hold {len(ids)} rows", flush=True)
    json.dump({"tag": tag, "histories": man, "rows": start}, open(os.path.join(cap, tag, "manifest.json"), "w"),
              indent=1)
    print(f"[capture] {tag}: {start:,} rows; manifest written", flush=True)
