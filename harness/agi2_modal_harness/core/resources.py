"""Handles to the shared Modal objects: Volumes, per-run Dicts, Secrets, App factory."""

from __future__ import annotations

import modal

from . import config


def models_volume() -> modal.Volume:
    return modal.Volume.from_name(config.VOLUME_MODELS, create_if_missing=True, version=2)


def cache_volume() -> modal.Volume:
    return modal.Volume.from_name(config.VOLUME_CACHE, create_if_missing=True, version=2)


def runs_volume() -> modal.Volume:
    return modal.Volume.from_name(config.VOLUME_RUNS, create_if_missing=True, version=2)


def ckpt_volume() -> modal.Volume:
    return modal.Volume.from_name(config.VOLUME_CKPT, create_if_missing=True, version=2)


class RetryingDict:
    """A modal.Dict whose calls retry transient backend errors with backoff.

    1004_0517's orchestrator died on ``ResourceExhaustedError: Dict backend is overloaded`` raised from
    ``d.keys()``. Every run component talks through this Dict: claims, heartbeats, traces, stop flags and the
    orchestrator loop. One transient error must not kill a run. After ``attempts`` tries (~2 min in all) the
    error propagates as before.
    """

    def __init__(self, d: modal.Dict, attempts: int = 7, base_delay_s: float = 1.0):
        self._d = d
        self._attempts = attempts
        self._base = base_delay_s

    def _retry(self, fn, *args, **kwargs):
        import random
        import time

        delay = self._base
        for i in range(self._attempts):
            try:
                return fn(*args, **kwargs)
            except (modal.exception.ResourceExhaustedError, modal.exception.ConnectionError,
                    modal.exception.InternalFailure, TimeoutError, ConnectionError):
                if i == self._attempts - 1:
                    raise
                time.sleep(delay + random.random() * delay)
                delay = min(delay * 2, 30)

    def get(self, key, default=None):
        return self._retry(self._d.get, key, default)

    def put(self, key, value, **kwargs):
        return self._retry(self._d.put, key, value, **kwargs)

    def keys(self):
        return self._retry(lambda: list(self._d.keys()))

    def items(self):
        return self._retry(lambda: list(self._d.items()))

    def pop(self, key, *default):  # d.pop(key, None) in the trace drain (1004_0648's orchestrator died on this)
        return self._retry(self._d.pop, key, *default)

    def __getitem__(self, key):
        return self._retry(self._d.__getitem__, key)

    def __setitem__(self, key, value):
        return self._retry(self._d.__setitem__, key, value)

    def __contains__(self, key):
        return self._retry(self._d.contains, key)

    def __getattr__(self, name):  # anything else: the raw Dict
        return getattr(self._d, name)


def run_dict(run_id: str) -> RetryingDict:
    """Live state for one run: heartbeats ``hb/<unit>``, job claims, control flags, summary."""
    return RetryingDict(modal.Dict.from_name(f"{config.PREFIX}-run-{run_id}", create_if_missing=True))


def hf_secret() -> modal.Secret:
    return modal.Secret.from_name(config.SECRET_HF)


def make_app(workload: str) -> modal.App:
    """One App per workload; each run is launched as a separately named, detached ephemeral app."""
    return modal.App(f"{config.PREFIX}-{workload}", tags={"project": config.PREFIX, "workload": workload})
