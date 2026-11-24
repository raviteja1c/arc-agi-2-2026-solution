"""Observability for the interact campaign (harness interactivity on 4 x L4, 2026-10-19): a sampler that runs beside
a harness round or replay for its whole length, and a trigger that opens profiler captures at chosen minutes.

The sampler writes one JSON line every `every` seconds to a local file (the runner mirrors it; nothing on the cache
Volume is ever appended to, harness_nb.snapshot):
  t       epoch seconds
  gpu     per GPU [power W, SM MHz, memory MHz, GPU util %, memory util %, memory used MiB, energy mJ, power-capped 0/1]
  cpu     the box's busy share of its CPUs (`total`, of `ncpu`) and per process class the CPUs it used (1.0 = one core):
          sglang ranks by name (TP0..TP3, the HTTP server, detokenizer), the sandbox workers summed, the runner, other
  cg      the cgroup's CPU throttling counters (nr_throttled, throttled_usec), when the container exposes them
  load    the server's /v1/loads (running, queued, used tokens, pool)
Every `metrics_every`-th row also carries `metrics`: the server's Prometheus counters and gauges (sglang:* without
histogram buckets), e.g. scheduler stage seconds, idle seconds, process CPU seconds, generated tokens.

The trigger POSTs /start_profile at each requested minute after `start()`: CUDA_PROFILER opens an nsys capture range
(the server runs under `nsys profile --capture-range=cudaProfilerApi --capture-range-end=repeat:N`); CPU,GPU starts the
torch profiler with an output directory. SGLang stops either after `num_steps` forward passes.

  python -m agi2_modal_harness.comm.interact_obs sample OUT.jsonl [--base URL] [--every 2]   (standalone, for tests)
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
import urllib.request
from pathlib import Path

_METRIC = re.compile(r"^(sglang:[a-zA-Z_:]+)(\{[^}]*\})?\s+([-+0-9.eE]+|NaN)$")


def _get(url: str, timeout: float = 5.0) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read().decode("utf-8", errors="replace")


def _post(url: str, body: dict, timeout: float = 120.0) -> str:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", errors="replace")


def _classify(cmd: str) -> str | None:
    """A process's class by its command line; None for processes not worth a column (shells, nsys's own helpers)."""
    if "sglang::scheduler" in cmd or "sglang::" in cmd:
        m = re.search(r"sglang::(\w+)", cmd)
        return m.group(1) if m else "sglang"
    if "arc_sandbox_worker" in cmd:
        return "sandbox"
    if "sglang.launch_server" in cmd:
        return "http_server"
    if "harness_nb" in cmd or "harness_replay" in cmd or "park_replay" in cmd:
        return "runner"
    if "nsys" in cmd.split(" ")[0] or "QdstrmImporter" in cmd:
        return "nsys"
    return "other"


class Sampler:
    def __init__(self, out: Path, base: str | None = None, every: float = 2.0, metrics_every: int = 5):
        self.out, self.base, self.every, self.metrics_every = Path(out), base, every, metrics_every
        self.out.parent.mkdir(parents=True, exist_ok=True)
        self.done = threading.Event()
        self.n = 0
        self.ticks = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100
        self.ncpu = os.cpu_count() or 1
        self._prev_proc: dict[int, tuple[float, str]] = {}
        self._prev_stat: tuple[float, float] | None = None
        self._prev_t = None
        try:
            import pynvml

            pynvml.nvmlInit()
            self.nv = pynvml
            self.handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(pynvml.nvmlDeviceGetCount())]
            self._reasons = getattr(pynvml, "nvmlDeviceGetCurrentClocksEventReasons", None) or \
                pynvml.nvmlDeviceGetCurrentClocksThrottleReasons
        except Exception:  # noqa: BLE001 - a CPU-only box still samples CPU
            self.nv, self.handles = None, []
        self.thread = threading.Thread(target=self._run, name="interact-sampler", daemon=True)

    def start(self) -> "Sampler":
        self.thread.start()
        return self

    def stop(self) -> None:
        self.done.set()
        self.thread.join(timeout=10)

    # ---- one row
    def _gpus(self) -> list:
        nv, rows = self.nv, []
        for h in self.handles:
            try:
                u = nv.nvmlDeviceGetUtilizationRates(h)
                rows.append([round(nv.nvmlDeviceGetPowerUsage(h) / 1000, 1), nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_SM),
                             nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_MEM), u.gpu, u.memory,
                             nv.nvmlDeviceGetMemoryInfo(h).used // (1 << 20), nv.nvmlDeviceGetTotalEnergyConsumption(h),
                             int(bool(self._reasons(h) & 0x4))])
            except Exception:  # noqa: BLE001
                rows.append(None)
        return rows

    def _cpu(self, now: float) -> dict:
        out: dict = {"ncpu": self.ncpu}
        try:
            f = open("/proc/stat").readline().split()[1:]
            vals = [float(x) for x in f]
            busy, total = sum(vals) - vals[3] - (vals[4] if len(vals) > 4 else 0), sum(vals)
            if self._prev_stat:
                db, dt = busy - self._prev_stat[0], total - self._prev_stat[1]
                out["total"] = round(db / dt, 4) if dt else None
            self._prev_stat = (busy, total)
        except OSError:
            pass
        procs: dict[str, float] = {}
        cur: dict[int, tuple[float, str]] = {}
        for d in os.listdir("/proc"):
            if not d.isdigit():
                continue
            try:
                with open(f"/proc/{d}/stat") as fh:
                    parts = fh.read().rsplit(")", 1)[1].split()
                cpu_s = (float(parts[11]) + float(parts[12])) / self.ticks
                prev = self._prev_proc.get(int(d))
                cls = prev[1] if prev else _classify(open(f"/proc/{d}/cmdline").read().replace("\0", " "))
            except (OSError, IndexError, ValueError):
                continue
            if cls is None:
                continue
            cur[int(d)] = (cpu_s, cls)
            if prev and self._prev_t:
                procs[cls] = procs.get(cls, 0.0) + (cpu_s - prev[0]) / (now - self._prev_t)
        self._prev_proc = cur
        if procs:
            out["procs"] = {k: round(v, 3) for k, v in sorted(procs.items())}
        return out

    @staticmethod
    def _cgroup() -> dict:
        for p in ("/sys/fs/cgroup/cpu.stat", "/sys/fs/cgroup/cpu/cpu.stat", "/sys/fs/cgroup/cpu,cpuacct/cpu.stat"):
            try:
                kv = dict(line.split() for line in open(p) if line.strip())
                return {k: int(v) for k, v in kv.items() if k in ("nr_periods", "nr_throttled", "throttled_usec",
                                                                    "throttled_time", "usage_usec")}
            except (OSError, ValueError):
                continue
        return {}

    @staticmethod
    def _mem() -> dict:
        """Host memory in GiB: /proc/meminfo (used = total - available; shmem holds pinned shared tables such as
        Flash-Next's 47.7 GiB PLE; mlocked) and the container's cgroup (current and peak, what an OOM kill counts).
        Kaggle's L4x4 has ~188 GB, so a Modal box with more must still be read against that ceiling."""
        out: dict = {}
        try:
            kv = {}
            for line in open("/proc/meminfo"):
                k, v = line.split(":", 1)
                kv[k] = int(v.split()[0]) / 2**20  # kB -> GiB
            out = {"total": round(kv["MemTotal"], 2), "used": round(kv["MemTotal"] - kv["MemAvailable"], 2),
                   "shmem": round(kv.get("Shmem", 0.0), 2), "cached": round(kv.get("Cached", 0.0), 2),
                   "mlocked": round(kv.get("Mlocked", 0.0), 2)}
        except (OSError, KeyError, ValueError):
            pass
        for name, key in (("/sys/fs/cgroup/memory.current", "cg_current"), ("/sys/fs/cgroup/memory.peak", "cg_peak"),
                          ("/sys/fs/cgroup/memory.max", "cg_max")):
            try:
                raw = open(name).read().strip()
                if raw != "max":
                    out[key] = round(int(raw) / 2**30, 2)
            except (OSError, ValueError):
                continue
        return out

    def _metrics(self) -> dict:
        out = {}
        for line in _get(f"{self.base}/metrics").splitlines():
            m = _METRIC.match(line.strip())
            if not m or m.group(1).endswith("_bucket"):
                continue
            labels = m.group(2) or ""
            cat = re.search(r'category="([^"]+)"', labels)
            key = m.group(1)[7:] + (f".{cat.group(1)}" if cat else "")
            try:
                out[key] = out.get(key, 0.0) + float(m.group(3))
            except ValueError:
                continue
        return out

    def _run(self) -> None:
        with open(self.out, "a", encoding="utf-8") as f:
            while not self.done.is_set():
                now = time.time()
                row: dict = {"t": round(now, 2)}
                if self.handles:
                    row["gpu"] = self._gpus()
                row["cpu"] = self._cpu(now)
                mem = self._mem()
                if mem:
                    row["mem"] = mem
                cg = self._cgroup()
                if cg:
                    row["cg"] = cg
                if self.base:
                    try:
                        load = json.loads(_get(f"{self.base}/v1/loads"))["loads"][0]
                        row["load"] = {"running": load.get("num_running_reqs"), "queue": load.get("queues"),
                                       "used": load.get("num_used_tokens"), "pool": load.get("max_total_num_tokens")}
                    except Exception:  # noqa: BLE001 - the server may be booting or gone
                        pass
                    if self.n % self.metrics_every == 0:
                        try:
                            row["metrics"] = self._metrics()
                        except Exception:  # noqa: BLE001 - --enable-metrics may be off
                            pass
                self._prev_t = now
                f.write(json.dumps(row) + "\n")
                f.flush()
                self.n += 1
                self.done.wait(max(0.1, self.every - (time.time() - now)))


class ProfileTrigger:
    """Open profiler captures at the given minutes after start(), each for `steps` forward passes."""

    def __init__(self, base: str, at_min: list[float], steps: int, kind: str, out_dir: str, log=print):
        self.base, self.at_min, self.steps, self.kind, self.out_dir, self.log = base, sorted(at_min), steps, kind, \
            out_dir, log
        self.events: list[dict] = []
        self.thread: threading.Thread | None = None
        self.done = threading.Event()

    def start(self) -> "ProfileTrigger":
        t0 = time.time()

        def run():
            for i, m in enumerate(self.at_min):
                if self.done.wait(max(0.0, t0 + m * 60 - time.time())):
                    return
                acts = ["CUDA_PROFILER"] if self.kind == "nsys" else ["CPU", "GPU"]
                body = {"num_steps": self.steps, "activities": acts, "output_dir": self.out_dir,
                        "profile_prefix": f"p{i}_m{m:g}"}
                if self.kind == "torch":
                    body.update(with_stack=False, record_shapes=False)
                ev = {"i": i, "at_min": m, "t": round(time.time(), 2), "kind": self.kind, "steps": self.steps}
                try:
                    ev["reply"] = _post(f"{self.base}/start_profile", body)[:300]
                except Exception as e:  # noqa: BLE001 - a failed capture must not stop the round
                    ev["error"] = f"{type(e).__name__}: {str(e)[:300]}"
                self.events.append(ev)
                self.log(f"[profile] capture {i} ({self.kind}, {self.steps} steps) at {m:g} min: "
                         f"{ev.get('reply') or ev.get('error')}")

        self.thread = threading.Thread(target=run, name="profile-trigger", daemon=True)
        self.thread.start()
        return self

    def stop(self) -> None:
        self.done.set()


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="interact_obs")
    ap.add_argument("cmd", choices=["sample"])
    ap.add_argument("out")
    ap.add_argument("--base", default=None)
    ap.add_argument("--every", type=float, default=2.0)
    ap.add_argument("--seconds", type=float, default=10.0)
    a = ap.parse_args(argv)
    s = Sampler(Path(a.out), a.base, a.every).start()
    time.sleep(a.seconds)
    s.stop()
    print(Path(a.out).read_text(encoding="utf-8")[-2000:])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
