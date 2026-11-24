"""Bench configuration: a shared base server config, named arms that override it, one concurrency sweep.

Arms are flag variants (speculative algorithm, all-reduce env, kernels) over the same model, prompts and
sampling, so the only thing that differs between two arms' numbers is what the arm changed.
"""

from __future__ import annotations

import copy
import hashlib
import json
import tomllib
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from types import SimpleNamespace

from agi2_modal_harness.workloads.arc_eval.run_config import ComputeCfg, SamplingCfg, ServerCfg, server_command


def _flag_groups(args: list[str]) -> list[list[str]]:
    groups: list[list[str]] = []
    for a in args:
        if a.startswith("--") or not groups:
            groups.append([a])
        else:
            groups[-1].append(a)
    return groups


def merge_flags(base: list[str], arm: list[str]) -> list[str]:
    """Base flags followed by the arm's, minus any base flag the arm sets again, so the logged command holds one
    value per flag (argparse would take the last one anyway, but a command line showing both misleads)."""
    overridden = {g[0] for g in _flag_groups(arm)}
    kept = [g for g in _flag_groups(base) if g[0] not in overridden]
    return [a for g in kept for a in g] + list(arm)


@dataclass
class SweepCfg:
    levels: list[int] = field(default_factory=lambda: [1, 8, 32, 48])
    c1_requests: int = 2  # c=1 runs this many requests back to back
    max_tokens: int = 3072  # cap so nearly every request runs full length and concurrency holds within a level
    level_timeout_s: float = 1500.0
    cooldown_s: float = 90.0  # idle gap between levels on the same GPUs, so heat from one level does not tax the next


@dataclass
class SnapshotCfg:
    """Context-realistic steady-state levels instead of the fresh-prompt sweep (runtime/snapshot.py)."""
    enabled: bool = False
    # "waves": a fresh prefill per level (every level sees the sample's own context mix). "ladder": one prefill at
    # levels[0], abort to step down; cheaper, but late rungs drift as requests age and end (1007_1243).
    mode: str = "waves"
    levels: list[int] = field(default_factory=list)  # concurrencies, descending
    max_new_tokens: int = 65536  # effectively unbounded: requests end by abort, not by budget
    measure_s: float = 90.0  # window per level once every request decodes
    settle_s: float = 8.0  # after stepping down, before the window opens
    growth_tokens: int = 20000  # per-request context growth allowed for in the KV capacity check (prefill + windows)
    prefill_timeout_s: float = 1200.0
    profile_levels: list[int] = field(default_factory=list)  # torch-profile these levels, after their window
    profile_steps: int = 12
    cut_lo: float = 0.02  # reasoning-prefix fraction drawn uniformly from [cut_lo, cut_hi]
    cut_hi: float = 0.9  # below 1 so few requests reach their trace's natural end inside a window
    seed: int = 20261007
    sources: list[str] = field(default_factory=list)  # record globs relative to the repo; [] = snapshot.DEFAULT_SOURCES


@dataclass
class FidelityCfg:
    """Gate F1 instead of a sweep: teacher-forced logprobs of full xhigh traces (runtime/fidelity.py)."""
    enabled: bool = False
    traces: int = 24  # spread over lengths, the same ones in every arm
    top_k: int = 20
    seed: int = 20261007
    concurrency: int = 2  # prefill-only requests in flight (each computes logprobs at up to 131k positions)
    sources: list[str] = field(default_factory=list)  # record globs; [] = snapshot.DEFAULT_SOURCES


@dataclass
class GreedyCfg:
    """Greedy probe after warmup: the same prompts at temperature 0, one at a time, in every arm. Arms whose kernels
    claim bitwise equality must return identical text (a hash per prompt is recorded)."""
    prompts: int = 0  # 0 = off
    max_new_tokens: int = 1024


@dataclass
class ArmCfg:
    name: str
    extra_args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    draft_alias: str = ""  # models registry alias -> --speculative-draft-model-path on the models Volume
    server: dict = field(default_factory=dict)  # ServerCfg field overrides, e.g. {"max_running_requests": 32}
    levels: list[int] = field(default_factory=list)  # replaces sweep.levels, e.g. for an arm capped at fewer slots
    model_alias: str = ""  # replaces the run's model, e.g. a derived variant with calibrated KV scales
    patches: list[str] = field(default_factory=list)  # core/patches/<name>.patch applied to SGLang before boot


@dataclass
class BenchConfig:
    slug: str
    model_alias: str
    served_name: str = "qwen3.8-27b"
    subset: str = "pilot30"  # prompts: every test input of these tasks, cycled to fill the largest level
    budget_usd: float = 6.0
    heartbeat_stale_s: float = 600.0
    # A GPU call with no heartbeat this long after spawn is cancelled (never placed or never booted); a queued call
    # costs only orchestrator CPU, so allow hours when capacity is short (1008_0004: L4:4 queued 50 min).
    placement_timeout_s: float = 3000.0
    max_parallel_arms: int = 2  # 2 x 4xL4 = 8 GPUs, under the plan's 10-GPU cap; later arms start as slots free
    arm_stagger_s: float = 90.0  # gap between arm starts, so two arms never boot (load weights, capture graphs) at once
    same_host: bool = False  # run every arm back to back in ONE container (paired A/B on one host; bench_group)
    sweep: SweepCfg = field(default_factory=SweepCfg)
    snapshot: SnapshotCfg = field(default_factory=SnapshotCfg)
    fidelity: FidelityCfg = field(default_factory=FidelityCfg)
    greedy: GreedyCfg = field(default_factory=GreedyCfg)
    sampling: SamplingCfg = field(default_factory=SamplingCfg)
    server: ServerCfg = field(default_factory=ServerCfg)
    compute: ComputeCfg = field(default_factory=lambda: ComputeCfg(shards=1, timeout_hours=1.5))
    arms: list[ArmCfg] = field(default_factory=list)

    def arm_server(self, arm: ArmCfg) -> ServerCfg:
        s = copy.deepcopy(self.server)
        for k, v in arm.server.items():
            if k not in {f.name for f in fields(ServerCfg)}:
                raise ValueError(f"arm {arm.name}: unknown server field {k!r}")
            setattr(s, k, v)
        s.extra_args = merge_flags(self.server.extra_args, arm.extra_args)
        return s

    def arm_model(self, arm: ArmCfg) -> str:
        return arm.model_alias or self.model_alias

    def arm_levels(self, arm: ArmCfg) -> list[int]:
        if self.snapshot.enabled and self.snapshot.levels and not arm.levels:
            return list(self.snapshot.levels)
        return list(arm.levels or self.sweep.levels)

    def arm_requests(self, arm: ArmCfg) -> int:
        if self.fidelity.enabled:
            return self.fidelity.traces
        if self.snapshot.enabled:
            return sum(self.arm_levels(arm))
        return sum(self.sweep.c1_requests if c == 1 else c for c in self.arm_levels(arm))

    def command(self, arm: ArmCfg, model_dir: str, draft_dir: str | None) -> list[str]:
        cmd = server_command(SimpleNamespace(server=self.arm_server(arm), served_name=self.served_name), model_dir)
        if arm.draft_alias:
            if not draft_dir:
                raise ValueError(f"arm {arm.name} needs draft {arm.draft_alias} but no directory was resolved")
            cmd += ["--speculative-draft-model-path", draft_dir]
        return cmd

    def to_dict(self) -> dict:
        return asdict(self)

    def config_hash(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()[:12]

    @classmethod
    def from_dict(cls, d: dict) -> "BenchConfig":
        d = dict(d)
        return cls(
            sweep=SweepCfg(**d.pop("sweep", {})),
            snapshot=SnapshotCfg(**d.pop("snapshot", {})),
            fidelity=FidelityCfg(**d.pop("fidelity", {})),
            greedy=GreedyCfg(**d.pop("greedy", {})),
            sampling=SamplingCfg(**d.pop("sampling", {})),
            server=ServerCfg(**d.pop("server", {})),
            compute=ComputeCfg(**d.pop("compute", {})),
            arms=[ArmCfg(**a) for a in d.pop("arms", [])],
            **d,
        )

    @classmethod
    def load(cls, path: Path, arms: list[str] | None = None, overrides: dict | None = None) -> "BenchConfig":
        raw = tomllib.loads(Path(path).read_text(encoding="utf-8"))
        for key, value in (overrides or {}).items():
            node = raw
            *parents, leaf = key.split(".")
            for p in parents:
                node = node.setdefault(p, {})
            node[leaf] = value
        cfg = cls.from_dict(raw)
        if arms:
            unknown = set(arms) - {a.name for a in cfg.arms}
            if unknown:
                raise ValueError(f"unknown arms {sorted(unknown)}; config has {[a.name for a in cfg.arms]}")
            cfg.arms = [a for a in cfg.arms if a.name in arms]
        if not cfg.arms:
            raise ValueError("no arms selected")
        if len({a.name for a in cfg.arms}) != len(cfg.arms):
            raise ValueError("arm names must be unique")
        return cfg
