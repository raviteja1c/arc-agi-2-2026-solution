"""Boot the shipped Flash-Next server (one arm of a bench config, through the shared command builder) with the expert
capture on (fork agi2/feat/expert-reround: SGLANG_AGI2_MOE_CAPTURE_DIR), prefill the planned training-task histories
one at a time (max_new_tokens=1, so no decode step and no draft MoE), phase by phase, then stop the server.

Fail-closed preflight before the long phase: after the first calib history, every one of the 48 MoE layers must have
written files whose token count equals the history's uncached prefill tokens, with ids in [0, n_experts) and weights
finite; else abort.

  python -m agi2_modal_harness.comm.nvfp4_capture --config <toml> --arm S24QZBRP --out $OUT \
      --cap-dir /tmp/fncap --histories data/xq_histories.npz --plan data/xq_plan.json [--model-dir DIR]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np


def log(msg: str) -> None:
    print(f"[capture {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def post(base: str, payload: dict, timeout: float = 1800) -> dict:
    req = urllib.request.Request(base + "/generate", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def layer_tokens(cap_dir: str, tag: str, layer: int) -> int:
    import torch

    n = 0
    for f in glob.glob(os.path.join(cap_dir, tag, f"L{layer:02d}", "*.pt")):
        n += torch.load(f, map_location="cpu")["x"].shape[0]
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cap-dir", required=True)
    ap.add_argument("--histories", required=True)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--model-dir", default="")
    ap.add_argument("--cache-gpu", default="L4")
    ap.add_argument("--boot-timeout-min", type=float, default=45)
    ap.add_argument("--env", action="append", default=[])
    a = ap.parse_args()

    sys.path.insert(0, os.getcwd())
    from agi2_modal_harness.core import config, models
    from agi2_modal_harness.core.runtime.sglang_server import SGLangServer
    from agi2_modal_harness.workloads.sglang_bench.bench_config import BenchConfig

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    os.makedirs(a.cap_dir, exist_ok=True)
    armed = os.path.join(a.cap_dir, "ARMED")
    if os.path.exists(armed):
        os.remove(armed)
    cfg = BenchConfig.load(Path(a.config), arms=[a.arm])
    arm = cfg.arms[0]
    model_dir = a.model_dir or str(models.model_path(config.MOUNT_MODELS, cfg.model_alias))
    draft = str(models.model_path(config.MOUNT_MODELS, arm.draft_alias)) if arm.draft_alias else None
    cmd = cfg.command(arm, model_dir, draft)
    server_cfg = cfg.arm_server(arm)
    # same cache layout as fn_bringup.py (resolved: the JIT cache misses under a symlink)
    cache = (Path(config.MOUNT_CACHE) / "sglang-0.5.21" / a.cache_gpu / "flashnext").resolve()
    env = {"SGLANG_CACHE_DIR": str(cache / "sglang"), "TRITON_CACHE_DIR": str(cache / "triton"),
           "TORCHINDUCTOR_CACHE_DIR": str(cache / "inductor"), "FLASHINFER_WORKSPACE_BASE": str(cache / "fi"),
           "XDG_CACHE_HOME": str(cache / "xdg"), "SGLANG_JIT_CACHE_DIR": str(cache / "jit"),
           **server_cfg.env, **arm.env, "SGLANG_AGI2_MOE_CAPTURE_DIR": a.cap_dir}
    for kv in a.env:
        k, _, v = kv.partition("=")
        env[k] = v
    for key in ("SGLANG_CACHE_DIR", "TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR", "FLASHINFER_WORKSPACE_BASE",
                "XDG_CACHE_HOME", "SGLANG_JIT_CACHE_DIR"):
        Path(env[key]).mkdir(parents=True, exist_ok=True)
    (out / "server_cmd.json").write_text(json.dumps({"cmd": cmd, "env": env}, indent=2), encoding="utf-8")
    keep_path = env.get("SGLANG_AGI2_FN_EXPERT_KEEP", "")
    log(f"model {model_dir}; keep list {keep_path or 'none (512 experts)'}")

    plan = json.load(open(a.plan))
    hist = np.load(a.histories)
    server = SGLangServer(cmd, port=server_cfg.port, log_path=out / "sglang.log", echo=log, env=env)
    t0 = time.time()
    server.start()
    result = {"arm": a.arm, "model_dir": model_dir, "keep": keep_path, "phases": {}}
    try:
        boot = server.wait_ready(a.boot_timeout_min * 60, on_tick=lambda s: None)
        log(f"[PHASE] boot done in {boot:.0f} s")
        base = server.base_url
        for phase in ("calib", "gate"):
            rows = plan["phases"][phase]["histories"]
            total = sum(r["tokens"] for r in rows)
            with open(armed, "w") as f:
                f.write(phase)
            tp, done_tok = time.time(), 0
            stats = []
            for i, r in enumerate(rows):
                ids = hist[r["name"]].tolist()
                th = time.time()
                resp = post(base, {"input_ids": ids, "sampling_params": {"max_new_tokens": 1, "temperature": 0.0}})
                meta = resp.get("meta_info", {})
                cached = int(meta.get("cached_tokens", 0) or 0)
                stats.append({"name": r["name"], "tokens": len(ids), "cached": cached,
                              "seconds": round(time.time() - th, 1)})
                done_tok += len(ids)
                if phase == "calib" and i == 0:
                    # fail-closed preflight: all 48 layers captured exactly the uncached prefill tokens
                    want = len(ids) - cached
                    got = [layer_tokens(a.cap_dir, phase, L) for L in range(48)]
                    import torch

                    f0 = sorted(glob.glob(os.path.join(a.cap_dir, phase, "L00", "*.pt")))[0]
                    d0 = torch.load(f0, map_location="cpu")
                    ok_ids = int(d0["ids"].min()) >= 0 and int(d0["ids"].max()) < 512
                    ok_w = bool(torch.isfinite(d0["w"].float()).all())
                    log(f"preflight: want {want} tokens a layer; got min {min(got)} max {max(got)}; "
                        f"x {tuple(d0['x'].shape)} {d0['x'].dtype}; ids ok {ok_ids}; weights finite {ok_w}")
                    result["preflight"] = {"want": want, "got_min": min(got), "got_max": max(got),
                                           "ids_ok": ok_ids, "w_ok": ok_w}
                    if not (min(got) == max(got) == want and ok_ids and ok_w):
                        raise RuntimeError(f"PREFLIGHT FAIL: {result['preflight']}")
                el = time.time() - tp
                rate = done_tok / max(el, 1e-6)
                log(f"{phase} {i + 1}/{len(rows)}: {done_tok}/{total} tokens ({100 * done_tok / total:.0f} %), "
                    f"{rate:.0f} tok/s, ETA {(total - done_tok) / max(rate, 1e-6) / 60:.1f} min, cached {cached}")
            with open(armed, "w") as f:
                f.write("")
            time.sleep(2)
            result["phases"][phase] = {"histories": stats, "seconds": round(time.time() - tp, 1),
                                       "layer_tokens": [layer_tokens(a.cap_dir, phase, L) for L in (0, 23, 47)]}
            log(f"[PHASE] {phase} done in {time.time() - tp:.0f} s: {done_tok} tokens; "
                f"layer tokens L0/L23/L47 {result['phases'][phase]['layer_tokens']}")
    except Exception as e:  # noqa: BLE001 - the failure is the result
        result["error"] = f"{type(e).__name__}: {str(e)[:2000]}"
        log(f"FAILED: {result['error']}")
    finally:
        if os.path.exists(armed):
            os.remove(armed)
        (out / "capture.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        server.stop()
    log(f"[PHASE] capture run done in {(time.time() - t0) / 60:.1f} min")
    return 1 if "error" in result else 0


if __name__ == "__main__":
    raise SystemExit(main())
