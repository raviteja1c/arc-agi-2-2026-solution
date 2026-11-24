# Third-party components

Original contributions are released under MIT-0 ([LICENSE](LICENSE)). The components below keep their own terms.
The licence texts are in [`licenses/`](licenses/).

## Models and weights

None of the weights are in this repository. The three checkpoints the notebooks load are Kaggle Models, listed with
exact versions in [`inference/MODELS.md`](inference/MODELS.md); the upstream models are linked below.

| Component | Role | Licence |
|---|---|---|
| [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) at `de4b8e4d43b9` | the base model; its BF16 routed experts are the QG build's input | Qwen Community License 1.0 ([text](licenses/qwen3.8-flash-next/LICENSE)) |
| [RadixArk/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4) at `7b719225242a` | the base checkpoint used to assemble the served model (Kaggle: `carriem7/qwen3-8-flash-next-nvfp4-mirror`) | the base model's terms: Qwen Community License 1.0 |
| QG routed experts, fitted for these submissions (Kaggle: `carriem7/qwen3-8-flash-next-nvfp4-qg1`) | the base model's routed experts re-rounded in the same NVFP4 format, with GPTQ or with re-chosen scales alone | a derivative of the base model: Qwen Community License 1.0, whose text ships with the Kaggle variation |
| [incoai/Qwen3.8-Flash-Next-DFlash2](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2) at `a22a1744e85c` | the speculative draft's base | Apache-2.0 ([standard licence text](licenses/sglang/LICENSE)) |
| h1 draft, fitted for these submissions (Kaggle: `carriem7/qwen3-8-flash-next-dflash2`, variation `agi2-h1-s1500-9a064a9068f9`) | DFlash2 with its 35 linear tensors fine-tuned | Apache-2.0, as its base |

The expert keep lists and the 32k draft vocabulary are lists of indices made for these submissions. They are covered
by [LICENSE](LICENSE).

## Code that is modified or copied

| Component | Where it is used here | Licence |
|---|---|---|
| [SGLang](https://github.com/sgl-project/sglang) v0.5.21 | `patches/sglang/`: every change made to the server, as diffs against v0.5.21. Each diff names the files it changes; `patches/README.md` says what each build and overlay changes | Apache-2.0 ([text](licenses/sglang/LICENSE)) |
| [SpecForge](https://github.com/sgl-project/SpecForge) at `53398a8` | `patches/specforge/`: the h1 trainer's changes | MIT, Copyright (c) 2025 sgl-project ([text](licenses/specforge/LICENSE)) |
| [ARC Prize's arc-agi-benchmarking](https://github.com/arcprize/arc-agi-benchmarking) at `9e2828f` | in cell `49f20e6b` of the submission 18 and 20 notebooks: the task prompt (`PROMPT`, from `prompts/system_prompt.txt`), the example format of `build_prompt` (from `prompts/prompt_manager.py`), and the `\boxed{}` reading that `parse_grid` tries first (from `utils/parsing.py`; the rest of `parse_grid` was written for these submissions) | MIT, Copyright (c) 2025 ARC Prize Foundation ([text](licenses/arc-agi-benchmarking/LICENSE.md)) |
| Kaggle notebook [`madarshbb/arc-qwen-3-8-27b-agent`](https://www.kaggle.com/code/madarshbb/arc-qwen-3-8-27b-agent) | the reminder after six `python` calls without a submission, adapted in cell `a5509211` of the submission 18 and 20 notebooks | shared publicly in the competition's notebooks, which the competition rules treat as licensed under an OSI-approved licence; no licence file is attached to the notebook |

## Runtime packages

The notebooks run on an offline runtime built by `runtime/1006-sglang-runtime-cp312`: a standalone CPython 3.12.15
and the 363 packages pinned in [`runtime/requirements.lock`](runtime/requirements.lock). Each package keeps its own
licence. The main ones:

| Component | Version | Licence |
|---|---|---|
| SGLang | 0.5.21 | Apache-2.0 ([text](licenses/sglang/LICENSE)) |
| FlashInfer | 0.6.18 | Apache-2.0, with bundled BSD-3-Clause and MIT components ([texts](licenses/flashinfer/)) |
| Triton | 3.7.1 | MIT ([text](licenses/triton/LICENSE)) |
| TileLang | 0.1.12 | MIT ([text](licenses/tilelang/LICENSE)) |
| NCCL | 2.30.7 | mostly Apache-2.0, with parts under their original BSD licence ([text](licenses/nccl/LICENSE.txt)) |
| PyTorch | 2.13.0+cu130 | BSD-3-Clause |
| NVIDIA CUDA libraries (cuBLAS, cuDNN, NVRTC and others, as wheels) | per the lock | NVIDIA's software licence terms |
| CPython (python-build-standalone) | 3.12.15 | PSF License, with the licences of the libraries the build bundles |

## Data

The notebooks read the [Kaggle competition data](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/data)
from `/kaggle/input` under the competition's access and redistribution rules.

The executed notebook copies keep Kaggle's outputs. The sandbox self-test (cell `aceb4d00`) shows demonstration
grids from [training task `00576224`](https://github.com/arcprize/ARC-AGI-2/blob/main/data/training/00576224.json)
in [submission 10's notebook](harness/notebooks/1022-sub10-qwen38-xhigh-kv4-all-levers/) and evaluation tasks
[`8b7bacbf`](https://github.com/arcprize/ARC-AGI-2/blob/main/data/evaluation/8b7bacbf.json) in submission 18 and
[`0934a4d8`](https://github.com/arcprize/ARC-AGI-2/blob/main/data/evaluation/0934a4d8.json) in submission 20.
ARC Prize publishes these grids under [Apache-2.0](https://github.com/arcprize/ARC-AGI-2/blob/main/LICENSE).
