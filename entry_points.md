# Prediction and fitting entry points

This guide lists the commands used to prepare data (D1–D4), fit and validate the model components (T1–T9), and
rebuild the fitted lists. The Check column says what a reader can compare each step's output with. See the
[README terms](README.md#terms) for the component names.

Run `agi2_modal_harness` modules with `harness/` on `PYTHONPATH` or from that directory. Substitute local paths and
values for angle-bracket placeholders. The scripts are unmodified copies from a development repository and keep its
relative paths and fixed mount points, so several steps below stage files at those paths rather than edit the
scripts. The [GPU staging guide](harness/README.md) gives the launchers' required paths. The original runs used
Kaggle for predictions, Modal for expert captures and fitting, and Colab RTX PRO 6000 runtimes for draft data,
capture and training.

## Predict

Choose one of the two [Kaggle notebooks and its pinned inputs](inference/INPUTS.md). That guide covers public
benchmarks, custom test sets and accelerator requirements. The [patch guide](patches/README.md) lists the server
builds.

## Prepare data

| Step | What | Command | Where it ran | Check |
|---|---|---|---|---|
| D1 | Generate h1 training conversations on tasks 200–399 of the seeded draw | Run the command below on each task slice. | Two Colab RTX PRO 6000 runtimes, ~55 min each | None: sampled. The recordings contain competition tasks and are withheld. |
| D2 | Build token histories and split by task | `VAL_TASKS=12 python training/draft_h1/code/make_harness_histories.py <out> fntrain 1.0 "<dft round>/lp/*.npz" "<dfg round>/lp/*.npz"` | CPU | The shipped `training/draft_h1/data/` index, split and request records; the original recordings reproduced them exactly. |
| D3 | Plan the QG capture | `python -m agi2_modal_harness.comm.nvfp4_capture_plan --src <D2 output> --out <dir> --calib-cap 16000 --gate-cap 16000 --calib-tokens 350000` | CPU | The shipped `xq_plan.json`; the original histories reproduced it exactly. |
| D4 | Plan the late capture for expert selection | Place the D2 output at `runs/10/25/1025_1926_colab-g4-dflash2-ft/harness_h1` relative to the working directory. Run `python <scratch-dir>/make_plan32.py` from that working directory with a copy of the script. It overwrites `xq32_plan.json` and writes `xq32_histories.npz` beside itself. | CPU | The shipped `xq32_plan.json`; the original histories reproduced it exactly. |

### Stage the training files

The task-draw scripts read and write the competition data one level above this repository's root, where it sat in
the development repository. From the repository root, put the Kaggle `arc-agi_*` files in
`../data/original_competition_data/` and link `data` to `../data`.

    mkdir -p ../data/original_competition_data
    ln -s ../data data

On Windows, use `mklink /J data ..\data` from Command Prompt for the link. The first task-draw script writes to
`data/` and the second reads `../data/`.

1. Run `python training/draft_h1/code/make_fntrain_set.py`. It draws 200 tasks with seed 1025 and writes
   `data/harness_bench/fntrain_{challenges,solutions,families}.json`.
2. `make_fntrain_g4_set.py` loads the first script from its development path, so copy it there first:

       mkdir -p ../docs/reports/1025_1016_fn_decode_3x
       cp training/draft_h1/code/make_fntrain_set.py ../docs/reports/1025_1016_fn_decode_3x/

   Then run `python training/draft_h1/code/make_fntrain_g4_set.py <out>`. It draws 400 tasks with the same seed
   and checks the first 200 against the reference. Use slices `dfg` and `dft` from `<out>/g4_slices.json`.
   The reference slices are in [`fntrain_g4_slices.json`](training/draft_h1/code/results/fntrain_g4_slices.json).

After staging [the runtime inputs](harness/README.md#stage-the-gpu-inputs), generate training conversations (D1)
with one slice per run. `fntrain_round.sh` deletes and recreates `/content/agi2`.

    bash /content/inputs/fntrain_round.sh <run-id> 8 150 \
        FN_SCHED=1 FN_TIERS=1 FN_NO_EXTRAS=1 FN_BUDGET=196608 \
        NIGHT_ARMS=g1 NIGHT_SAMPLES=0 NIGHT_TASKS="<comma-separated task IDs>"

D2 sorts all the matched recording paths together, so the order of its arguments does not matter; the paths do. In the
original run the `dft` paths sorted before the `dfg` ones. Keep that order, through the folder names, to reproduce
the history index and the capture plans built from it. The request records store each recording's path. D2 checks
that every task is a training task and refuses any evaluation or private validation task.

## Fit and validate

| Step | Piece | Command | Where | Time | Check |
|---|---|---|---|---|---|
| T1 | Expert routing counts | In the [staged harness root](harness/README.md#restore-paths-for-the-census-modules), with the competition's `arc-agi_training_challenges.json` under `data/original_competition_data/`, run `OUT=<dir> FN_MODEL_SRC=<RadixArk export> bash agi2_modal_harness/comm/fn_l4_census_code.sh 48`. Check that the count file was written: the script can report done after a failed server. For the prose counts, the census driver settings were `--census-tasks 20 --census-code 0`. See the [census guide](training/keep_census/README.md). | Modal, 4 × L4 | — | The recorded count files |
| T2 | Census expert list (sub 18) | For new counts, from the staged harness root: `python -m agi2_modal_harness.comm.fn_l4_merge_census <prose counts> <code counts> <out> --keep 384 --ep 4`, which writes `<out>/keep_384_code_bal4.json`. To replay the shipped list from the recorded counts: `python training/keep_census/rebuild.py` | CPU | seconds | Byte-identical (`a4a75c4e`) |
| T3 | MoE activation capture for QG | Run `nvfp4_capture` with `S24QZBRP`, a server configuration in `l4x4_sub14_b8study.toml` (the census list, FP8 KV cache, h1 and the MoE capture hook), as shown in the [QG guide](training/qg_experts/README.md#fit-the-experts). | Modal, 4 × L40S | 15 min | Its plan is D3's |
| T4 | QG experts | From `harness/`, `CAP=/tmp/fncap OUT=/tmp/fnxq/qg1 bash agi2_modal_harness/comm/nvfp4_qg_4gpu.sh`. Stage the BF16 expert weights first. | Same 4 × L40S machine | 10 min | [Shard hashes](training/qg_experts/README.md#verify-the-shards) |
| T5 | Late activation capture | After [staging](harness/README.md#stage-the-late-capture-machine), `OUT=/tmp/arc-late32-run RUN_ID=arc-late32 bash training/keep_c512/capture_late32/capture32.sh`. Use a fresh output directory and run ID. The script recreates `/content/agi2_cap32` and `/tmp/fncap32`. | Modal, 4 × L40S | — | Its plan is D4's |
| T6 | Output-error expert list (sub 20) | `python training/keep_c512/fn_expert_choose512.py --bf16 <BF16> --census training/keep_census/keep_384_code_bal4.json --cap16 <T3 capture> --cap32 <T5 capture> --out <dir> --layers <range>`. Collect the per-layer outputs in one directory and rerun with `--merge` and the same `--bf16`, `--census` and `--out` arguments, or verify the shipped merge with `python training/keep_c512/remerge.py`. | Three Modal machines, 1 × L40S each | ~40 min | Merge byte-identical (`ccaac87a`) |
| T7 | Draft candidate vocabulary | `python training/draft_vocab/rebuild.py --tokenizer <tokenizer.json> --attempts <attempt-root>`. An exact rebuild needs the original attempt files, which are withheld; the [vocabulary guide](training/draft_vocab/README.md) describes them. Inference uses the shipped list. | CPU | minutes | Byte-identical with the original attempt files (`f72ceca4`) |
| T8 | h1 features and teacher log-probabilities | `bash training/draft_h1/code/g4/harness_capture_chain.sh <run>-hcap1 <D2 output>`. Once the server is ready, the script deletes the staged RadixArk safetensors if free space under `/content` is below its 90 GiB threshold. T9's evaluation server reads them through links, so restore them before T9 if they were deleted. | Colab RTX PRO 6000 | 20 min | `capture_status.py` counts the captured histories |
| T9 | h1 training, checkpoint selection, export and evaluation | `bash training/draft_h1/code/g4/harness_train_gate.sh <run> h1 <D2 output> 1e-4 10`. It waits for T8 and checks only the training split's capture: confirm both splits first (187 of 187 and 12 of 12 in the recorded run) with `python3 /content/inputs/capture_status.py /content/hprobe/out <D2 output>/split.json`. The evaluation needs its own recordings and a stock-draft result in `/content/gate_stock.txt`. Check the evaluation logs: the done marker can appear after a failed evaluation. | Colab RTX PRO 6000 | 29 min training | Step-1500 tensor hashes and [export hash](training/draft_h1/README.md#verify-the-export) |

### Order of a full refit

1. Census list: T1 → T2.
2. Draft data and training: T2 → D1 → D2. T2 + D2 → T8 → T9.
3. Expert re-rounding: D2 → D3. T2 + D3 + T9 → T3. T3 + BF16 expert weights + RadixArk export → T4.
4. Expert selection: D2 → D4. T4 + T9 + D4 → T5. T2 + T3 + T5 + BF16 weights → T6.
5. Draft vocabulary (T7) is independent of the other stages; it reads attempt files of earlier submissions.

The T2 replay and the T6 merge replay need only the repository and Python's standard library. D2, D3, D4 and T7 also
need the withheld recordings or token histories. The 58 GB of draft features and two MoE captures (92.9 GiB and
118.2 GiB) need to be recorded again for a refit.

The GPU fitting stages use different patched SGLang 0.5.21 builds. See the [patch guide](patches/README.md) for
the server builds and the patched SpecForge trainer. The CPU data preparation uses NumPy, and the vocabulary
rebuild uses `tokenizers`. The [runtime lock](runtime/requirements.lock) pins the Python dependencies.
