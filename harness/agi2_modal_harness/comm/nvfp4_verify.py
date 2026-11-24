"""Verify BF16 expert shards against the Hub's LFS SHA-256 (the tree API's lfs.oid), in parallel over CPU cores.
Shards whose hash differs are deleted with --delete-bad (they will be re-fetched). Prints one line per bad shard and a
summary; exit 0 only when every present shard verifies.

  python -m agi2_modal_harness.comm.nvfp4_verify --dir /tmp/fnbf16 [--delete-bad] [--procs 16]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import urllib.request
from multiprocessing import Pool

REPO = "Qwen/Qwen3.8-Flash-Next"
REV = "de4b8e4d43b917e7706784d8bb445c9af86a3540"


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(64 << 20)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def job(args):
    path, want = args
    return path, sha256(path) == want


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--delete-bad", action="store_true")
    ap.add_argument("--procs", type=int, default=16)
    a = ap.parse_args()
    with urllib.request.urlopen(f"https://huggingface.co/api/models/{REPO}/tree/{REV}", timeout=120) as r:
        tree = {x["path"]: x["lfs"]["oid"] for x in json.loads(r.read()) if x.get("lfs")}
    files = sorted(f for f in os.listdir(a.dir) if f.startswith("model-") and f.endswith(".safetensors"))
    t0 = time.time()
    bad = []
    with Pool(a.procs) as p:
        for i, (path, ok) in enumerate(p.imap_unordered(job, [(os.path.join(a.dir, f), tree[f]) for f in files]), 1):
            if not ok:
                bad.append(os.path.basename(path))
                print(f"[verify] BAD {os.path.basename(path)}", flush=True)
            if i % 10 == 0:
                print(f"[verify] {i}/{len(files)} hashed, {len(bad)} bad, {time.time() - t0:.0f} s", flush=True)
    if a.delete_bad:
        for b in bad:
            os.remove(os.path.join(a.dir, b))
    print(f"[PHASE] verify done in {time.time() - t0:.0f} s: {len(files)} shards, {len(bad)} bad "
          f"{'(deleted)' if a.delete_bad and bad else ''}", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
