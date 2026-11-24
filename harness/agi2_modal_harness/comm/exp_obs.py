"""System metrics for EVERY dev-box experiment: `comm.dev run --exp` starts this beside the experiment's command, always
(user, 2026-10-20: System Metrics were missed "5th or 6th time"; like kaggle-nb's 120 s wait cap, it is enforced in
code, not remembered). The NVML/CPU sampler (comm/interact_obs.Sampler, with SGLang's /v1/loads and /metrics when a
server answers on --base) appends on container disk, and the file is copied whole to $OUT/obs/samples.jsonl every
--every-s seconds (Volumes do not honour O_APPEND, README run 1003_1242). The campaign tracker turns it into the run
`<run_id>/system`: Trackio's System Metrics tab (gpu/<i>/utilization, power, memory, clocks; cpu/*) and gpu/* bins.
tests/test_observability_contract.py keeps both ends in place.

  python -m agi2_modal_harness.comm.exp_obs --out $OUT --hours 6 [--base http://127.0.0.1:30000]
"""

from __future__ import annotations

import argparse
import os
import shutil
import signal
import time
from pathlib import Path


def mirror(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    tmp = dst.with_name(dst.name + ".tmp")
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)


def main(argv: list[str] | None = None) -> int:
    from agi2_modal_harness.comm.interact_obs import Sampler

    ap = argparse.ArgumentParser(prog="exp_obs")
    ap.add_argument("--out", required=True, help="the experiment's $OUT on the cache Volume")
    ap.add_argument("--base", default=None, help="an SGLang server to read /v1/loads and /metrics from, if any")
    ap.add_argument("--every", type=float, default=2.0)
    ap.add_argument("--every-s", type=float, default=30.0)
    ap.add_argument("--hours", type=float, default=24.0)
    a = ap.parse_args(argv)
    out = Path(a.out) / "obs"
    out.mkdir(parents=True, exist_ok=True)
    local = Path("/tmp/exp-obs") / Path(a.out).name
    local.mkdir(parents=True, exist_ok=True)
    sampler = Sampler(local / "samples.jsonl", a.base, a.every).start()
    stop = {"now": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(now=True))
    print(f"[exp_obs {time.strftime('%H:%M:%S')}] sampling every {a.every:g} s -> {out}/samples.jsonl", flush=True)
    end = time.time() + a.hours * 3600
    try:
        while time.time() < end and not stop["now"]:
            for _ in range(int(a.every_s)):
                if stop["now"]:
                    break
                time.sleep(1)
            mirror(local / "samples.jsonl", out / "samples.jsonl")
    finally:
        sampler.stop()
        mirror(local / "samples.jsonl", out / "samples.jsonl")
    return 0


def wrap(cmd: str, out_remote: str, log: str, hours: float, base: str | None = None) -> str:
    """`cmd` with the sampler running beside it for its whole life; the command's exit code is kept."""
    obs = f"python -m agi2_modal_harness.comm.exp_obs --out {out_remote} --hours {hours:g}"
    if base:
        obs += f" --base {base}"
    return (f"{obs} > {log} 2>&1 & OBS_PID=$!; ( {cmd} ); rc=$?; kill $OBS_PID 2>/dev/null; wait $OBS_PID 2>/dev/null; "
            f"exit $rc")


if __name__ == "__main__":
    raise SystemExit(main())
