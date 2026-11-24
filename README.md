# ARC Prize 2026 – ARC-AGI-2: Qwen3.8-Flash-Next as a program-writing agent on 4 × L4

Code, notebooks and server patches behind two final submissions to the
[ARC Prize 2026 – ARC-AGI-2](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2) competition on Kaggle.
The method is described in the
[Kaggle Solution Writeup](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/757546).

**The problem it solves.** One open-weight model has to solve 240 hidden puzzles in 12 hours on Kaggle's offline
4 × L4 machine, which has 96 GB of GPU memory. The model is Qwen3.8-Flash-Next, a mixture of experts with 48 MoE
layers of 512 routed experts each (10 active per token), and its 4-bit NVFP4 export is 135 GB.

## Approach

**The agent.** For each task the model first looks at the puzzle drawn as pictures and describes it. Then it writes a
Python program in a sandbox, using a `python` tool and a `submit` tool.
- A program is **accepted** only when it turns every demonstration input into exactly its output; its output for the
  test input is the answer.
- Each task gets up to 4 attempts. Submission 20 runs 64 conversations at once, and submission 18 runs 48.
- The two answers allowed per test input come from a vote that ranks accepted programs first.

**Making it fit and run fast.** Four fitted pieces and a patched SGLang 0.5.21 server:
- **Expert keep list.** 384 of each layer's 512 routed experts stay routable.
  - Submission 18 keeps the experts the router picks most often on training tasks ([`training/keep_census/`](training/keep_census/)).
  - Submission 20 re-chose them by the error each removal leaves in the layer's output. That lowered the layers'
    held-out output error against the full BF16 model from 0.0335 to 0.0104
    ([`training/keep_c512/`](training/keep_c512/)).
- **Re-rounded experts (QG).** RadixArk's NVFP4 export of the model rounds each weight to the nearest 4-bit value.
  The routed experts are re-rounded in the same format: with GPTQ, calibrated on the model's own runs on training
  tasks, for 18,329 of the 24,576 experts, and with re-chosen scales alone (QS) for the rest. On held-out training
  tasks the MoE output error falls 63.5 %, and the next-token KL to the BF16 model 25 % with all 512 experts served
  ([`training/qg_experts/`](training/qg_experts/)).
- **Fine-tuned speculative draft (h1).** incoai's DFlash2 draft model, fine-tuned on the agent's own conversations on
  training tasks. Simulated over 24 recorded conversations on public evaluation tasks, it raises the tokens per
  verification step by 2.3 % (2.775 → 2.840). The model verifies every drafted token, so answers still follow the
  model's own distribution ([`training/draft_h1/`](training/draft_h1/)).
- **Draft vocabulary.** The draft proposes only from 32,768 candidate tokens. In a microbenchmark on one L4, that cut
  its output layer from 1.65 ms to 0.24 ms a step ([`training/draft_vocab/`](training/draft_vocab/)).
- **Server changes.** Most of the work went into memory. The model's n-gram tables and most of the sparse-attention
  cache live in host RAM, and the number of conversations decoding at once rose from 1 to 64.
  [`patches/OPTIMIZATIONS.md`](patches/OPTIMIZATIONS.md) explains each change, why it helps on 4 × L4 and what it
  measured. The exact diffs are in [`patches/`](patches/README.md).

## The two submissions

| Submission | Kaggle notebook (the scored version) | Public score | Expert keep list | Conversations at once |
|---|---|---|---|---|
| 57853136 (sub 20) | [`1101-sub20-qwen38-flashnext-s64b-par2fix`, v2](https://www.kaggle.com/code/carriem7/1101-sub20-qwen38-flashnext-s64b-par2fix?scriptVersionId=364353181) | 80.56 | re-chosen by output error | 64 |
| 57785004 (sub 18) | [`1030-sub18-qwen38-flashnext-s48-5d76d78-int8`, v3](https://www.kaggle.com/code/carriem7/1030-sub18-qwen38-flashnext-s48-5d76d78-int8?scriptVersionId=363637083) | 78.19 | routing counts | 48 |

Both use the QG experts, h1 and the draft vocabulary. Submission 20's server also:
- reuses each task's prefilled prompt across its attempts;
- has six memory savings that submission 18 lacks;
- reads each request's sparse-attention index keys once a step, not once per verified position;
- prefills in chunks of 2,048 tokens.

The [settings table](patches/OPTIMIZATIONS.md#11-the-two-finals-settings) compares the two builds.

## Kaggle inputs

All of them are public:

| Input | Kaggle | Version |
|---|---|---|
| Offline runtime: Python 3.12 and 363 pinned packages | notebook [`1006-sglang-runtime-cp312`](https://www.kaggle.com/code/carriem7/1006-sglang-runtime-cp312?scriptVersionId=355555024) | 4 |
| Compile cache ("seed") for submission 20's server | notebook [`1101-seed-flashnext-sub20-s64b-t8`](https://www.kaggle.com/code/carriem7/1101-seed-flashnext-sub20-s64b-t8?scriptVersionId=364348853) | 2 |
| Compile cache ("seed") for submission 18's server | notebook [`1030-seed-flashnext-sub18-5d76d78-s48-int8`](https://www.kaggle.com/code/carriem7/1030-seed-flashnext-sub18-5d76d78-s48-int8?scriptVersionId=363632895) | 3 |
| Base model: RadixArk's NVFP4 export of Qwen3.8-Flash-Next | model [`qwen3-8-flash-next-nvfp4-mirror`](https://www.kaggle.com/models/carriem7/qwen3-8-flash-next-nvfp4-mirror/Transformers/hf-7b719225242a/1) | 1 |
| QG experts | model [`qwen3-8-flash-next-nvfp4-qg1`](https://www.kaggle.com/models/carriem7/qwen3-8-flash-next-nvfp4-qg1/Transformers/agi2-qg1-8352943a271b/1) | 1 |
| h1 draft | model [`qwen3-8-flash-next-dflash2`](https://www.kaggle.com/models/carriem7/qwen3-8-flash-next-dflash2/Transformers/agi2-h1-s1500-9a064a9068f9/1) | 1 |

[`inference/INPUTS.md`](inference/INPUTS.md) pins each one, and [`inference/MODELS.md`](inference/MODELS.md) describes
the models.

## Running a submission notebook

1. Copy a notebook from the table above, or upload its source copy:
   [submission 20](inference/sub20/1101-sub20-qwen38-flashnext-s64b-par2fix.v2.source.ipynb) or
   [submission 18](inference/sub18/1030-sub18-qwen38-flashnext-s48-5d76d78-int8.v3.source.ipynb).
2. Attach the competition data, the runtime and the three models above, **at those versions**. A copy of a Kaggle
   version lists the original base model, which is private; attach the public mirror instead. The notebook finds
   every input by name, so the mirror needs no code change.
3. Attach **only the seed made for that notebook**. The notebook uses the first seed it finds, and a seed works only
   with the server build it was made for. With the wrong one it compiles everything again, which adds about 18
   minutes to start-up.
4. Run it with the internet off on 4 × NVIDIA L4. The scored runs used Kaggle's `NvidiaL4` machine. The competition
   has closed, and Kaggle may not offer that machine now, so check the GPUs the settings cell prints.

Started this way, the notebook runs a timed benchmark on public evaluation tasks:
- submission 20: all 120 tasks (172 test outputs) for 105 minutes;
- submission 18: tasks 71–120 (70 test outputs) for 25 minutes.

Its last cell prints the benchmark's score and writes the answers to `answers_final.json`. `submission.json` always
covers the attached test file; with only the competition data attached, that is a 240-task placeholder. To run on your
own tasks, see [`inference/INPUTS.md`](inference/INPUTS.md).

Answers vary from run to run: sampling runs at temperature 1.0, and the multi-GPU arithmetic is not bit-exact. Each
build was scored once.

## Refitting the fitted pieces

- [`entry_points.md`](entry_points.md) gives every command in order: data preparation D1–D4, fitting T1–T9.
- Each [`training/`](training/) folder's README gives the inputs, the method and the results.
- [`harness/README.md`](harness/README.md) says how to stage the development harness the GPU steps ran through.

The fitting ran on Modal (4 × L4, 4 × L40S, 1 × L40S) and Colab RTX PRO 6000 machines; one QG check ran on Modal
4 × H200. The GPU steps were not re-run after the competition.

**Data.** The weights and expert lists were fitted on public training tasks only. The draft vocabulary also counted
the tokens of earlier runs on 30 public evaluation tasks, 3 training tasks and 4 private hand-made validation puzzles.
It only limits which tokens the draft may propose.

## Checks without a GPU

    python -I training/keep_census/rebuild.py                           # submission 18's expert list, byte for byte
    python -I training/keep_c512/remerge.py                             # submission 20's expert list, byte for byte
    python -I training/draft_h1/check_export.py <model.safetensors>     # a downloaded h1 against its SHA-256

These need Python 3.10 or later and its standard library only. The first two run on the repository alone; the third
needs the `model.safetensors` file of the h1 model on Kaggle.

## Repository layout

| Path | What it holds |
|---|---|
| `inference/` | both submitted notebooks (Kaggle's source, the executed copy, the metadata), their seed notebooks, and the input and model notes |
| `runtime/` | the offline runtime notebook and its 363 pins, `requirements.lock` |
| `patches/` | every change to SGLang and to the SpecForge trainer, as diffs; [`OPTIMIZATIONS.md`](patches/OPTIMIZATIONS.md) explains the server changes |
| `training/` | one folder per fitted piece, with its code, records and README |
| `harness/` | the development harness modules and notebooks the GPU fitting steps used |
| `entry_points.md` | every command, in order |
| `LICENSE`, `THIRD_PARTY_NOTICES.md`, `licenses/` | the licence of this work, and the third-party components with their licence texts |

## Terms

| Term | Meaning |
|---|---|
| sub N | the N-th submission to the competition; the finals are sub 20 and sub 18 |
| keep list | the 384 of each layer's 512 routed experts that stay routable |
| census | counting how often the router picks each expert |
| QG, QS | experts re-rounded with GPTQ; experts with re-chosen scales only |
| h1 | the fine-tuned DFlash2 speculative draft |
| seed | a notebook whose output is the compiled kernels and caches for one server build; it cuts start-up to about 9 minutes |
| capture | recording the model's activations for fitting |
| harness | the agent's prompt, tools and attempt loop; "the development harness" is the code that ran the agent and the fitting on cloud machines |
| D1–D4, T1–T9 | the data-preparation and fitting steps in [`entry_points.md`](entry_points.md) |
| eval children | hand-made validation puzzles derived from public evaluation tasks; never published |

## Licence

The work written for these submissions is released under [MIT-0](LICENSE). That covers the code, the notebooks, the
documentation, the server and trainer patches, the expert keep lists and the draft vocabulary.

Third-party components keep their own terms ([`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md),
[`licenses/`](licenses/)):
- the base model and the QG experts are under the Qwen Community License 1.0;
- the h1 draft is under Apache-2.0, as its DFlash2 base;
- the patches modify Apache-2.0 SGLang and MIT SpecForge.
