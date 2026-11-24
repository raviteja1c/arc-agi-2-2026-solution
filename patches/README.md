# SGLang and SpecForge patches

Both final submissions use patched SGLang v0.5.21 to serve Qwen3.8-Flash-Next on four NVIDIA L4 GPUs. The builds
load only the kept routed experts, keep most of the sparse-attention cache in host memory, add up tensor-parallel
partial sums on the host CPU, keep each conversation on the server between turns, and speed up speculative decoding.
Submission 20 adds grouped INT8 index reads, shared prompt prefill across attempts, and smaller GPU allocations;
its notebook also changes the prefill and all-reduce settings.
[`OPTIMIZATIONS.md`](OPTIMIZATIONS.md) explains each change, what it measured and which switch enables it.

## Inference server

The [submission 20](../inference/sub20/) and [submission 18](../inference/sub18/) notebooks embed the diffs they
apply to the offline runtime's SGLang installation. They run `patch -p2 --fuzz=0 --forward` in `site-packages`,
in the order below. The fork commits name the snapshots of the private development forks that the diffs were made
from; the diffs themselves hold every change.

| Submission | Order | Diff | Fork commit | Purpose |
|---|---|---|---|---|
| 57853136 (sub 20) | 1 | [`1-sglang-v0.5.21-to-sub20-c414512.diff`](sglang/sub20/1-sglang-v0.5.21-to-sub20-c414512.diff) | `c414512aa737` | Base server build (126 files) |
| | 2 | [`2-req_split.diff`](sglang/sub20/2-req_split.diff) | Based on `5d76d788f1eb` | Splits Qwen Sparse Attention (QSA) work by request. `SGLANG_AGI2_QSA_REQ_SPLIT` is off. |
| | 3 | [`3-session_fork.diff`](sglang/sub20/3-session_fork.diff) | `cc16a7a200fb` | Reuses a parked copy of the shared prompt across attempts. Enabled. |
| | 4 | [`4-memlev3.diff`](sglang/sub20/4-memlev3.diff) | `c756b83ee02c` | Allocates the attention mask buffer on demand, keeps an expert location map on the CPU, and shrinks the draft's successor table. Enabled. |
| 57785004 (sub 18) | 1 | [`1-sglang-v0.5.21-to-sub18-5d76d78.diff`](sglang/sub18/1-sglang-v0.5.21-to-sub18-5d76d78.diff) | `5d76d788f1eb` | Base server build (124 files) |

Each base diff reproduces its fork's `python/sglang` tree from a clean v0.5.21 source tree. From the root of
this repository, apply the submission 20 stack as follows. For submission 18, use its base diff alone.

```sh
git clone --config core.autocrlf=false --branch v0.5.21 https://github.com/sgl-project/sglang
cd sglang
git apply ../patches/sglang/sub20/1-sglang-v0.5.21-to-sub20-c414512.diff
git apply ../patches/sglang/sub20/2-req_split.diff
git apply ../patches/sglang/sub20/3-session_fork.diff
git apply ../patches/sglang/sub20/4-memlev3.diff
```

## Server builds and hooks for fitting

The [fitting commands](../entry_points.md) show where the older builds and capture hooks were used. Both
older builds patch SGLang v0.5.21. Apply the listed hook after its server build.

| Fitting stage | Server build | Capture hook |
|---|---|---|
| Draft training conversations and training features for h1 | [`sglang-v0.5.21-to-sub13-e5cb6d9.diff`](sglang/capture/sglang-v0.5.21-to-sub13-e5cb6d9.diff) (`e5cb6d9afb87`) | [`dflash2_probe.diff`](sglang/capture/dflash2_probe.diff) (`110bd2042fbe`) records target features and teacher scores for training. |
| Expert activations for GPTQ re-rounding | [`sglang-v0.5.21-to-sub17-06e9a82.diff`](sglang/capture/sglang-v0.5.21-to-sub17-06e9a82.diff) (`06e9a8204c16`) | [`moe_capture.diff`](sglang/capture/moe_capture.diff) (`ea30d8e7afb7`) records expert inputs and routing. |
| Expert activations at positions 16,000–31,999 for expert selection | Submission 18 base build above | The same `moe_capture.diff` hook. |

## Draft trainer

| Diff | Upstream base | Fork commit | Purpose |
|---|---|---|---|
| [`specforge-53398a8-to-1878d57a.diff`](specforge/specforge-53398a8-to-1878d57a.diff) | SpecForge `53398a8f01ae` | `1878d57a8675` (33 files) | Fine-tunes the h1 draft with `scripts/agi2_dflash2_offline_ft.py`. |

## Licence

The original changes are released under [MIT-0](../LICENSE). A patched copy stays under its upstream licence.
SGLang uses [Apache-2.0](../licenses/sglang/LICENSE) and SpecForge uses the [MIT licence](../licenses/specforge/LICENSE).
