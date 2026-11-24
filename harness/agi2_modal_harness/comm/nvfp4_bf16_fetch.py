"""Fetch only the BF16 routed-expert shards of Qwen/Qwen3.8-Flash-Next (96 of 131 shards, ~253 GB) onto the cache
Volume, for re-rounding RadixArk's NVFP4 experts (docs/reports/1029_1845_nvfp4_rerounding_scope/PRIORITIES.md).

Resumable: a shard whose local size matches the Hub's is skipped. Prints a progress line every 30 s (files, GB, rate,
ETA) and fails closed if any shard's final size differs from the Hub's.

Run on a dev box:  python -m agi2_modal_harness.comm.nvfp4_bf16_fetch [--dest DIR] [--workers 8]
"""
from __future__ import annotations

import argparse
import json
import os
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

REPO = "Qwen/Qwen3.8-Flash-Next"
REV = "de4b8e4d43b917e7706784d8bb445c9af86a3540"
DEST = "/vol/cache/dev/fnbf16/Qwen__Qwen3.8-Flash-Next__de4b8e4d43b9"


def hub_json(url: str):
    with urllib.request.urlopen(url, timeout=120) as r:
        return json.loads(r.read())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default=DEST)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--slice", default="", help="a:b = only shards [a, b) of the sorted expert-shard list "
                    "(split the fetch across containers: one container is capped near ~210 MB/s by the Hub)")
    ap.add_argument("--rest", action="store_true", help="instead: every repo file that is not an expert shard "
                    "(the other 34 shards, config, tokenizer), completing a servable BF16 checkpoint")
    a = ap.parse_args()
    os.makedirs(a.dest, exist_ok=True)

    idx = hub_json(f"https://huggingface.co/{REPO}/resolve/{REV}/model.safetensors.index.json")
    with open(os.path.join(a.dest, "model.safetensors.index.json"), "w") as f:
        json.dump(idx, f)
    tree = {x["path"]: x.get("size", 0) for x in hub_json(f"https://huggingface.co/api/models/{REPO}/tree/{REV}")
            if x.get("type", "file") == "file"}
    shards = sorted({v for k, v in idx["weight_map"].items() if ".mlp.experts." in k and "mtp" not in k})
    if a.rest:
        shards = sorted(p for p in tree if p not in set(shards) and p != "model.safetensors.index.json")
    if a.slice:
        lo, hi = (int(v) for v in a.slice.split(":"))
        shards = shards[lo:hi]
    total = sum(tree[s] for s in shards)
    todo = [s for s in shards if not (os.path.exists(os.path.join(a.dest, s))
                                      and os.path.getsize(os.path.join(a.dest, s)) == tree[s])]
    print(f"[PHASE] plan: {len(shards)} expert shards, {total / 1e9:.1f} GB; {len(todo)} to fetch", flush=True)
    t0 = time.time()
    done_bytes = [total - sum(tree[s] for s in todo)]
    start_bytes = done_bytes[0]
    done_files = [len(shards) - len(todo)]
    lock = threading.Lock()
    stop = threading.Event()

    def progress():
        while not stop.wait(30):
            with lock:
                # finished files plus the bytes already in .part files; counting finished files alone reads as a
                # stall for minutes while 24 large shards download in parallel (1029_2056 was stopped for that)
                parts = sum(os.path.getsize(os.path.join(a.dest, s + ".part")) for s in todo
                            if os.path.exists(os.path.join(a.dest, s + ".part")))
                got = done_bytes[0] - start_bytes + parts
                rate = got / max(time.time() - t0, 1)
                have = done_bytes[0] + parts
                left = total - have
                eta = left / rate / 60 if rate > 0 else float("nan")
                print(f"[fetch] {done_files[0]}/{len(shards)} files done, {have / 1e9:.1f}/{total / 1e9:.1f} GB "
                      f"incl. partials ({100 * have / total:.0f} %), {rate / 1e6:.0f} MB/s, ETA {eta:.1f} min, "
                      f"elapsed {(time.time() - t0) / 60:.1f} min", flush=True)

    threading.Thread(target=progress, daemon=True).start()

    def one(s):
        # plain HTTPS through curl (resumable): measured 28 MB/s a connection on Modal, while hf_xet stalled at
        # ~1 MB/s in all; HF_TOKEN comes from the environment, never argv (curl reads the header from stdin)
        import subprocess

        part = os.path.join(a.dest, s + ".part")
        url = f"https://huggingface.co/{REPO}/resolve/{REV}/{s}"
        hdr = f"Authorization: Bearer {os.environ.get('HF_TOKEN', '')}\n"
        for attempt in range(5):
            r = subprocess.run(["curl", "-sS", "-L", "--fail", "-C", "-", "-H", "@-", "-o", part,
                                "--retry", "3", "--speed-time", "60", "--speed-limit", "1000000", url],
                               input=hdr.encode(), capture_output=True)
            if r.returncode == 0 and os.path.getsize(part) == tree[s]:
                break
            print(f"[fetch] retry {s} (rc {r.returncode}: {r.stderr.decode()[:200]})", flush=True)
        if not os.path.exists(part) or os.path.getsize(part) != tree[s]:
            raise RuntimeError(f"{s}: incomplete after retries; left as {part}")  # never give a partial file its name
        os.replace(part, os.path.join(a.dest, s))
        size = os.path.getsize(os.path.join(a.dest, s))
        if size != tree[s]:
            raise RuntimeError(f"{s}: size {size} != hub {tree[s]}")
        with lock:
            done_bytes[0] += tree[s]
            done_files[0] += 1
        return s

    with ThreadPoolExecutor(a.workers) as ex:
        for fut in as_completed([ex.submit(one, s) for s in todo]):
            fut.result()
    stop.set()
    bad = [s for s in shards if os.path.getsize(os.path.join(a.dest, s)) != tree[s]]
    if bad:
        print(f"FAILED size check: {bad}", flush=True)
        return 1
    with open(os.path.join(a.dest, "FETCH_OK.json"), "w") as f:
        json.dump({"repo": REPO, "revision": REV, "shards": len(shards), "bytes": total,
                   "minutes": round((time.time() - t0) / 60, 1)}, f)
    print(f"[PHASE] fetch done in {(time.time() - t0) / 60:.1f} min: {len(shards)} shards, {total / 1e9:.1f} GB",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
