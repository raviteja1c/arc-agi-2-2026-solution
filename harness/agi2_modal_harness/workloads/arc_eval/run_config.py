"""Run configuration for the ARC eval workload: TOML in, server command and plan out."""

from __future__ import annotations

import hashlib
import json
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path

# Qwen3.8-27B geometry (config.json @ 1d4bf0f; SGLang cookbook "Mamba ratio calculator").
GDN_STATE_BYTES = {"float32": 153.9e6, "bfloat16": 78.4e6}  # one request's linear-attention state
KV_BYTES_PER_TOKEN = {"auto": 65.5e3, "bfloat16": 65.5e3, "fp8_e4m3": 32.8e3}  # 16 full-attn layers


@dataclass
class DataCfg:
    challenges: str = "data/original_competition_data/arc-agi_evaluation_challenges.json"
    solutions: str = "data/original_competition_data/arc-agi_evaluation_solutions.json"
    limit: int = 0  # 0 = all selected tasks
    task_ids: list[str] = field(default_factory=list)
    subset: str = ""  # name of a fixed subset in subsets/<name>.json (make_subset.py); task_ids wins if both set
    reference_variant: str = "low"
    num_attempts: int = 2
    # Run id of a finished, synced run: this run then makes only the follow-up attempt 2 of each pair, conditioned on
    # that run's attempt 1 (copied in as this run's attempt 1, so scoring is pass@2 on identical first attempts).
    followup_of: str = ""
    followup_skip_solved: bool = False  # experiment only: no attempt 2 where the source's attempt 1 is already right
    prompt_variant: str = "arc"  # arc_prompt.PROMPT_VARIANTS; "arc" = ARC's prompt byte for byte
    # Several variants as arms of one run (one boot, one host, shared slots); scored per variant. Overrides
    # prompt_variant when set. The first variant is the run's headline score.
    variants: list[str] = field(default_factory=list)

    def variant_list(self) -> list[str]:
        return list(self.variants) or [self.prompt_variant]


@dataclass
class ComputeCfg:
    gpu: str = "L4:4"
    shards: int = 2
    cpu: float = 4.0
    memory_mib: int = 32768
    timeout_hours: float = 8.0
    boot_timeout_min: float = 30.0


@dataclass
class ServerCfg:
    tp: int = 4
    port: int = 30000
    mem_fraction_static: float = 0.82
    context_length: int = 262144
    max_running_requests: int = 32
    chunked_prefill_size: int = 2048
    disable_radix_cache: bool = True
    mamba_ssm_dtype: str = "float32"
    kv_cache_dtype: str = "auto"
    avg_request_tokens: int = 40000  # prompt + output, for the state/KV split
    mamba_full_memory_ratio: float | None = None  # None = derive from avg_request_tokens
    attention_backend: str | None = None
    decode_log_interval: int = 200
    extra_args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)  # server process env, e.g. a SGLANG_* opt-in
    patches: list[str] = field(default_factory=list)  # core/patches/<name>.patch applied to SGLang before boot

    def derived_mamba_ratio(self) -> float:
        if self.mamba_full_memory_ratio is not None:
            return self.mamba_full_memory_ratio
        slots = 1 if self.disable_radix_cache else 5
        ratio = slots * GDN_STATE_BYTES[self.mamba_ssm_dtype] / (
            self.avg_request_tokens * KV_BYTES_PER_TOKEN[self.kv_cache_dtype])
        return round(ratio * 1.25, 4)  # 25% spare state slots so state never clamps before KV


@dataclass
class SamplingCfg:
    reasoning_effort: str | None = "low"
    enable_thinking: bool = True
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 20
    min_p: float = 0.0
    presence_penalty: float = 0.0
    repetition_penalty: float = 1.0
    max_tokens: int = 131072
    seed: int | None = None
    # Keep each try's exact prompt and output token ids (SGLang sglext input_ids / output_ids): what SpecForge
    # trains a draft from (prompts_path), with no re-tokenization drift. ~0.4 MB per 70k-token try.
    return_token_ids: bool = False


@dataclass
class RunnerCfg:
    concurrency: int = 48
    retry_attempts: int = 2
    stall_abort_s: float = 900
    expected_tokens_per_call: int = 31300
    on_cap: str = "retry"  # "conclude": continue a capped try with the thinking block closed before any retry


@dataclass
class GuardCfg:
    budget_usd: float = 40.0
    heartbeat_stale_s: float = 600  # orchestrator cancels a shard whose heartbeat is older than this
    boot_max_s: float = 2400


@dataclass
class RunConfig:
    slug: str
    model_alias: str = "qwen3.8-27b"
    draft_alias: str = ""  # models-registry alias of a separate speculative draft (DFlash2, DSpark); "" = none
    served_name: str = "qwen3.8-27b"
    sglang_image_variant: str = ""
    data: DataCfg = field(default_factory=DataCfg)
    compute: ComputeCfg = field(default_factory=ComputeCfg)
    server: ServerCfg = field(default_factory=ServerCfg)
    sampling: SamplingCfg = field(default_factory=SamplingCfg)
    runner: RunnerCfg = field(default_factory=RunnerCfg)
    guards: GuardCfg = field(default_factory=GuardCfg)

    def to_dict(self) -> dict:
        return asdict(self)

    def config_hash(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()[:12]

    @classmethod
    def from_dict(cls, d: dict) -> "RunConfig":
        sections = {"data": DataCfg, "compute": ComputeCfg, "server": ServerCfg,
                    "sampling": SamplingCfg, "runner": RunnerCfg, "guards": GuardCfg}
        kwargs = {k: v for k, v in d.items() if k not in sections}
        for name, klass in sections.items():
            kwargs[name] = klass(**d.get(name, {}))
        return cls(**kwargs)

    @classmethod
    def load(cls, path: str | Path, overrides: dict | None = None) -> "RunConfig":
        with open(path, "rb") as f:
            d = tomllib.load(f)
        for dotted, value in (overrides or {}).items():
            node = d
            *parents, leaf = dotted.split(".")
            for p in parents:
                node = node.setdefault(p, {})
            node[leaf] = value
        return cls.from_dict(d)


def server_command(cfg: RunConfig, model_dir: str, draft_dir: str | None = None) -> list[str]:
    """``draft_dir`` is the resolved directory of ``cfg.draft_alias`` (the caller checks its manifest)."""
    s = cfg.server
    cmd = [
        "python", "-m", "sglang.launch_server",
        "--model-path", model_dir,
        "--served-model-name", cfg.served_name,
        "--host", "127.0.0.1", "--port", str(s.port),
        "--tp", str(s.tp),
        "--reasoning-parser", "qwen3",
        "--context-length", str(s.context_length),
        "--mem-fraction-static", str(s.mem_fraction_static),
        "--max-running-requests", str(s.max_running_requests),
        "--chunked-prefill-size", str(s.chunked_prefill_size),
        "--mamba-ssm-dtype", s.mamba_ssm_dtype,
        "--kv-cache-dtype", s.kv_cache_dtype,
        "--enable-metrics",
        "--decode-log-interval", str(s.decode_log_interval),
        "--trust-remote-code",
    ]
    if s.disable_radix_cache:
        cmd.append("--disable-radix-cache")
    else:  # with radix off and N set (always, here) SGLang ignores the ratio (1007_1121 slice A, section 3)
        cmd += ["--mamba-full-memory-ratio", str(s.derived_mamba_ratio())]
    if s.attention_backend:
        cmd += ["--attention-backend", s.attention_backend]
    cmd += list(s.extra_args)
    if draft_dir:
        cmd += ["--speculative-draft-model-path", draft_dir]
    return cmd
