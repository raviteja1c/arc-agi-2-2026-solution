"""Launch an SGLang server subprocess, gate on readiness, and read its metrics.

The server's full log goes to a file; selected lines (errors, decode stats,
startup milestones) are echoed with a prefix so a remote log stream shows
progress during the minutes a 27B model takes to load.
"""

from __future__ import annotations

import os
import re
import signal
import subprocess
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from collections.abc import Callable
from pathlib import Path

# Lines that mean the server will never become healthy; checked while booting.
FATAL_PATTERNS = re.compile(
    r"CUDA out of memory|OutOfMemoryError|RuntimeError: |torch\.OutOfMemoryError|"
    r"Not enough memory|NCCL error|Segmentation fault|core dumped|ValueError: |AssertionError|"
    r"Address already in use|Exception in thread|ModuleNotFoundError|ImportError"
)
# Lines worth echoing to the remote log stream.
ECHO_PATTERNS = re.compile(
    r"Load weight|weight[s]? (begin|end)|Memory pool|max_total_num_tokens|max_running_requests|"
    r"Capture cuda graph|cuda graph|The server is fired up|Uvicorn running|ready to roll|"
    r"Decode batch|Prefill batch|retract|Retract|error|Error|ERROR|WARN|Traceback|"
    r"mamba|Mamba|KV Cache|kv cache|context_len|attention backend|available_gpu_mem"
)
METRIC_NAMES = (
    "num_running_reqs", "num_queue_reqs", "num_retracted_reqs", "token_usage", "full_token_usage", "mamba_usage",
    "gen_throughput", "kv_used_tokens", "kv_available_tokens", "max_total_num_tokens",
    "generation_tokens_total", "prompt_tokens_total", "num_requests_total",
    "num_aborted_requests_total", "num_retracted_requests_total", "utilization",
)
_METRIC_LINE = re.compile(r"^sglang:([a-z_]+)(?:\{[^}]*\})?\s+([-+0-9.eE]+|NaN)$")


class ServerError(RuntimeError):
    """The SGLang server died or will not become healthy."""


_EXC_LINE = re.compile(r"^\s*([\w.]*(?:Error|Exception)):\s*(\S.*)$")


def failure_gist(lines) -> str:
    """The last ``SomethingError: message`` line of a crash log: the line worth reading first. Error strings
    get cut to a prefix downstream, and four TP ranks' tracebacks push the real cause past any short tail."""
    for line in reversed(list(lines)):
        if m := _EXC_LINE.match(line):
            return f"{m.group(1)}: {m.group(2)}"[:400]
    return ""


def failure_summary(error: object) -> str:
    """One alert line for a failure: the error's head (before any log tail) plus its exception line if the head
    lacks it. Accepts an error string or a result/summary dict carrying an ``error`` key."""
    text = str((error.get("error") if isinstance(error, dict) else error) or "no error recorded")
    head = text.split("; last lines")[0].strip()[:300]
    gist = failure_gist(text.splitlines())
    return head + (f" | {gist}" if gist and gist not in head else "")


class SGLangServer:
    def __init__(
        self,
        cmd: list[str],
        *,
        port: int,
        log_path: Path,
        echo: Callable[[str], None] = print,
        env: dict[str, str] | None = None,
        echo_every_decode: int = 20,
    ) -> None:
        self.cmd = cmd
        self.port = port
        self.base_url = f"http://127.0.0.1:{port}"
        self.log_path = log_path
        self.echo = echo
        self.env = env
        self.echo_every_decode = echo_every_decode
        self.proc: subprocess.Popen | None = None
        self.recent = deque(maxlen=200)
        self.fatal_line: str | None = None
        self.started_at = 0.0
        self._decode_lines = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.echo("[SGL] launching: " + " ".join(self.cmd))
        self.started_at = time.monotonic()
        self.proc = subprocess.Popen(
            self.cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env={**os.environ, **(self.env or {})},
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        self._pump_thread = threading.Thread(target=self._pump, name="sglang-log", daemon=True)
        self._pump_thread.start()

    def _drain_log(self, timeout_s: float = 5.0) -> None:
        """After the process exits, let the pump read the last lines (the traceback) before they are quoted."""
        if getattr(self, "_pump_thread", None) is not None:
            self._pump_thread.join(timeout=timeout_s)

    def _pump(self) -> None:
        assert self.proc is not None and self.proc.stdout is not None
        with open(self.log_path, "a", encoding="utf-8") as log:
            for line in self.proc.stdout:
                log.write(line)
                log.flush()
                line = line.rstrip()
                self.recent.append(line)
                if self.fatal_line is None and FATAL_PATTERNS.search(line):
                    self.fatal_line = line
                if "Decode batch" in line:
                    self._decode_lines += 1
                    if self._decode_lines % self.echo_every_decode:
                        continue
                if ECHO_PATTERNS.search(line):
                    self.echo(f"[SGL] {line[:400]}")

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def wait_ready(self, timeout_s: float, on_tick: Callable[[float], None] | None = None) -> float:
        """Block until /health answers 200; fail fast if the process dies or logs a fatal line."""
        deadline = time.monotonic() + timeout_s
        fatal_seen_at: float | None = None
        while time.monotonic() < deadline:
            if not self.alive():
                self._drain_log()
                raise ServerError(f"SGLang exited with code {self.proc.returncode if self.proc else '?'}: "
                                  f"{failure_gist(self.recent) or 'no exception line'}; "
                                  f"last lines:\n" + "\n".join(list(self.recent)[-30:]))
            if self.fatal_line and fatal_seen_at is None:
                fatal_seen_at = time.monotonic()
                self.echo(f"[SGL] fatal-looking line during boot: {self.fatal_line[:300]}")
            # A crashed TP rank can leave the parent alive but hung; don't wait out the full timeout.
            if fatal_seen_at and time.monotonic() - fatal_seen_at > 90 and not self._health():
                raise ServerError(f"SGLang logged a fatal error and never became healthy: {self.fatal_line}\n"
                                  + "\n".join(list(self.recent)[-30:]))
            if self._health():
                return time.monotonic() - self.started_at
            if on_tick:
                on_tick(time.monotonic() - self.started_at)
            time.sleep(5)
        raise ServerError(f"SGLang not ready within {timeout_s:.0f}s: {failure_gist(self.recent) or 'no exception line'}; "
                          f"last lines:\n"
                          + "\n".join(list(self.recent)[-30:]))

    def _health(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self.base_url}/health", timeout=5) as r:
                return r.status == 200
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
            return False

    def stop(self, grace_s: float = 20) -> None:
        if not self.alive():
            return
        assert self.proc is not None
        try:
            os.killpg(self.proc.pid, signal.SIGTERM)
            self.proc.wait(timeout=grace_s)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                os.killpg(self.proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def server_info(self) -> dict:
        import json

        try:
            with urllib.request.urlopen(f"{self.base_url}/get_server_info", timeout=10) as r:
                return json.loads(r.read())
        except Exception as exc:  # noqa: BLE001 - informational only
            return {"error": str(exc)}

    def metrics(self) -> dict[str, float]:
        """Selected Prometheus gauges and counters, summed across label sets."""
        try:
            with urllib.request.urlopen(f"{self.base_url}/metrics", timeout=5) as r:
                text = r.read().decode("utf-8", "replace")
        except Exception:  # noqa: BLE001 - a missed scrape is not fatal
            return {}
        out: dict[str, float] = {}
        for line in text.splitlines():
            m = _METRIC_LINE.match(line)
            if m and m.group(1) in METRIC_NAMES and m.group(2) != "NaN":
                out[m.group(1)] = out.get(m.group(1), 0.0) + float(m.group(2))
        return out
