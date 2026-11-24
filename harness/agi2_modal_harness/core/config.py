"""Workspace-wide constants for every workload: names, mount points, prices, limits.

Prices are Modal list rates read from https://modal.com/pricing on 2026-10-03
(USD per second). Re-read the page before trusting a cost estimate after a
price change; the harness uses them only for live estimates, never billing.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

PREFIX = "agi2"
REPO_ROOT = Path(__file__).resolve().parents[2]
LOCAL_RUNS_DIR = REPO_ROOT / "runs"  # local mirror of run artifacts (gitignored), runs/mm/dd/<run_id>/

# Volumes (v2: many concurrent writers, fast commits; contents are rebuildable).
VOLUME_MODELS = f"{PREFIX}-models"  # HF snapshots, one directory per (repo, revision)
VOLUME_CACHE = f"{PREFIX}-cache"  # JIT/compile caches, keyed by software stack and GPU type
VOLUME_RUNS = f"{PREFIX}-runs"  # /mm/dd/<run_id>/ per run: inputs, records, logs, summaries
VOLUME_CKPT = f"{PREFIX}-ckpt"  # draft-training checkpoints (tens of GB each): kept out of the runs Volume, which `sync` mirrors
MOUNT_MODELS = "/vol/models"
MOUNT_CACHE = "/vol/cache"
MOUNT_RUNS = "/vol/runs"
MOUNT_CKPT = "/vol/ckpt"

SECRET_HF = f"{PREFIX}-hf"  # HF_TOKEN; created once from the local HF_API_KEY

# Hugging Face, all private and all named agi2: model repo for run archives and checkpoints,
# Trackio dashboard Space, and the Bucket the Space mounts at /data (Trackio DB + traces).
HF_REPO = "ravitejac/agi2"
TRACKIO_SPACE = "ravitejac/agi2"
TRACKIO_BUCKET = "ravitejac/agi2"
TRACKIO_VERSION = "0.40.0"
# One Trackio project for the whole competition, so every run (eval, bench, later RL) compares on one
# dashboard. Runs are told apart by `group` ("<workload>/<slug>") and config keys (workload, slug,
# config_hash); metrics are namespaced per workload (arc: run/ score/ shard-k/ tokens/, bench: bench/).
TRACKIO_PROJECT = "agi2"

# Starter plan: 10 GPUs in use at once, 100 containers, logs kept 1 day.
PLAN_GPU_CONCURRENCY = 10

GPU_PRICE_PER_S = {
    "B300": 0.001972, "B200": 0.001736, "H200": 0.001261, "H100": 0.001097,
    "RTX-PRO-6000": 0.000842, "A100-80GB": 0.000694, "A100": 0.000583, "A100-40GB": 0.000583,
    "L40S": 0.000542, "A10": 0.000306, "L4": 0.000222, "T4": 0.000164,
}
CPU_CORE_PRICE_PER_S = 0.0000131
MEM_GIB_PRICE_PER_S = 0.00000222
# Sandboxes (and Notebooks) pay 3x for CPU and memory; the GPU price is the same (`modal billing rates`, 2026-10-13:
# CPU (Sandbox) $0.1419/core/h, Memory (Sandbox) $0.024/GiB/h).
SANDBOX_CPU_CORE_PRICE_PER_S = 0.00003942
SANDBOX_MEM_GIB_PRICE_PER_S = 0.00000667


def parse_gpu(spec: str | None) -> tuple[str | None, int]:
    """'L4:4' -> ('L4', 4); 'H100!:4' -> ('H100', 4); None -> (None, 0). A trailing '!' only tells Modal's
    scheduler not to substitute another GPU; the name, price and count are the same."""
    if not spec:
        return None, 0
    name, _, count = spec.partition(":")
    return name.upper().rstrip("!"), int(count or 1)


def gpu_name_ok(spec: str, device_name: str) -> bool:
    """Whether a device ``nvidia-smi`` reports satisfies a ``gpu=`` request. Modal may run ``gpu="H100"`` on an
    H200 at the H100 price ("Automatic upgrades to H200s", docs/modal_docs GPU guide); ``"H100!"`` pins it."""
    want = spec.partition(":")[0].upper()
    exact, name, got = want.endswith("!"), want.rstrip("!"), device_name.upper()
    if name.replace("-", " ") in got:
        return True
    return not exact and name == "H100" and "H200" in got


def gpu_hourly_usd(spec: str | None) -> float:
    name, count = parse_gpu(spec)
    return GPU_PRICE_PER_S.get(name, 0.0) * count * 3600 if name else 0.0


def run_rel(run_id: str) -> str:
    """Month/day folders so a listing stays short over many days: ``1003_1154_x`` -> ``10/03/1003_1154_x``.

    The same relative path is used under local ``runs/``, on the runs Volume and in the HF repo.
    Ids that do not start with ``mmdd_`` (none are created any more) stay flat.
    """
    head = run_id[:4]
    if len(run_id) > 10 and head.isdigit() and run_id[4] == "_":
        return f"{head[:2]}/{head[2:]}/{run_id}"
    return run_id


def local_run_dir(run_id: str) -> Path:
    return LOCAL_RUNS_DIR / run_rel(run_id)


def volume_run_dir(run_id: str) -> str:
    """The run's directory inside the runs Volume (mounted at MOUNT_RUNS in containers)."""
    return "/" + run_rel(run_id)


def new_run_id(slug: str) -> str:
    """Local wall-clock (IST on this machine) + slug, e.g. ``1003_1452_qwen38-low-smoke``.

    Two launches in the same minute with the same slug get ``-2``, ``-3`` ... appended.
    """
    safe = "".join(c if c.isalnum() or c in "-_" else "-" for c in slug.lower()).strip("-")
    base = f"{datetime.now():%m%d_%H%M}_{safe}"
    run_id, n = base, 1
    while local_run_dir(run_id).exists():
        n += 1
        run_id = f"{base}-{n}"
    return run_id
