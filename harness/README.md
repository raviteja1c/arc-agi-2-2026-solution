# GPU fitting harness

The GPU fitting launchers import `agi2_modal_harness` and run cells from two notebooks. The [entry-point guide](../entry_points.md)
defines the step IDs used here. The scored inference notebooks carry their own harness.

| Path | Purpose |
|---|---|
| `agi2_modal_harness/core/` | Model paths, configuration and the SGLang server wrapper |
| `agi2_modal_harness/comm/` | Notebook runner (`harness_nb.py`), GPU monitoring (`exp_obs.py` and `interact_obs.py`), expert fitting (`nvfp4_*`) and the tool-call probe used to check the census list |
| `agi2_modal_harness/logprobs/` | Inserts the cell that records token histories and log-probabilities |
| `agi2_modal_harness/workloads/` | Server configurations and notebook utilities, including the `arc_vision.py` grid renderer |
| `notebooks/1018-harness-bench/` | GPU fitting notebook run by `harness_nb.py` |
| `notebooks/1022-sub10-qwen38-xhigh-kv4-all-levers/` | Earlier submission notebook whose helper cells are loaded by the training-data runner |

Run modules from this folder or put it on `PYTHONPATH`.

    PYTHONPATH=<repo>/harness python -m agi2_modal_harness.comm.nvfp4_capture_plan --help

## Notes on this snapshot

This folder is the final state of the development harness. A few files changed after the runs that used them: the
notebook runner, the token-recording helpers, the capture module, the model registry and `l4x4_sub14_b8study.toml`.
The commands in the [entry-point guide](../entry_points.md) and the staging below work with this snapshot. Use
server configuration `S24QZBRP` for T3, and the runner's `--boot-only` path with the submission 18 notebook for T5.

## Stage the GPU inputs

Create a staging directory for each launcher. Use the layout below at the root of each archive.

### Draft training conversations (D1)

Add `agi2_modal_harness/` and `notebooks/` from `<repo>/harness/` and place the following files under `data/`.

- Under `data/harness_bench/`, put the 400-task `fntrain_{challenges,solutions,families}.json` set produced by
  `make_fntrain_g4_set.py`. The two runs take task slices 200–399 from this set. [Task draw commands](../entry_points.md#stage-the-training-files).
- In the same folder, supply a stand-in `bench24_{challenges,solutions,families}.json`. The harness notebook is an
  unmodified copy, and its setup cell `hbdata` checks for a 24-task bench before cell `hbfntrain` switches to
  `fntrain` and recording starts. Any 24 public tasks pass, given 32 test inputs in total, solutions for all tasks,
  and a `families` file whose `parents` and `children` lists cover all 24.
- Put Kaggle's competition files under `data/original_competition_data/`. Also stage
  `arc-agi_evaluation_challenges.json` under `/tmp/hw/<run-id>/data/original_competition_data/` so that
  `hbfntrain` can check the training subset there. The task draw checks exclusions itself.

    tar -czf /content/inputs/agi2_repo_fntrain.tgz -C <d1-staging-dir> agi2_modal_harness notebooks data

`fntrain_round.sh` recreates `/content/agi2` on launch and passes its own server configuration. When invoking
`harness_nb.py` directly, pass `--config` to name one of the supplied configurations.

### Stage the late-capture machine

For T5, add `agi2_modal_harness/` to the staging directory. Stage the
[submission 18 source notebook](../inference/sub18/1030-sub18-qwen38-flashnext-s48-5d76d78-int8.v3.source.ipynb)
at `notebooks/1030-sub18-qwen38-flashnext-s48-5d76d78-int8/1030-sub18-qwen38-flashnext-s48-5d76d78-int8.ipynb`.
Put Kaggle's challenge files under `data/fresh61/`, the `--data` path fixed in `capture32.sh`.

    tar -czf /content/inputs/agi2_repo_s18.tgz -C <t5-staging-dir> agi2_modal_harness notebooks data

Before running `capture32.sh`, prepare the following.

1. Install the runtime in `/tmp/sgl` from `training/draft_h1/g4_inputs/wheels.txt` and
   `runtime/requirements.lock`. `probe_setup.sh` shows the installation steps.
2. Apply `patches/sglang/sub18/1-sglang-v0.5.21-to-sub18-5d76d78.diff` with `patch -p2 --fuzz=0`
   from the runtime's `site-packages` directory.
3. Place RadixArk, h1 and the QG shards at `/tmp/models/qwen3.8-flash-next-nvfp4-radixark`,
   `/tmp/models/dflash2-h1` and `/tmp/models/qg1`. Write `ok` to `/content/setup_exit.txt`
   and `sub18 5d76d788f1` to `/content/runtime_ok`.
4. Put `capture_client.py`, `xq32_plan.json`, `xq32_histories.npz` and `moe_capture.diff` in
   `/vol/cache/dev/fncap/xq32_inputs/`. Take the plan and the histories from the same D4 run: the client looks up
   every plan entry in the histories. The hook is in `patches/sglang/capture/`.

`capture32.sh` calls the notebook runner with `--boot-only`. The notebook's GPU-name check counts `NVIDIA L4`
substrings, so the original four L40S GPUs pass. For an intentional run on different compatible hardware,
export `KAGGLE_IS_COMPETITION_RERUN=1` in the launch environment to make the notebook's failed setting checks
warn. The variable controls notebook checks and workload selection.

### Draft capture and training (T8–T9)

`probe_setup.sh` extracts `/content/inputs/agi2_lib.tgz` under `/content/lib`. Build that archive from this folder:

    mkdir -p /content/inputs
    tar -czf /content/inputs/agi2_lib.tgz -C <repo>/harness agi2_modal_harness

Stage these files together under `/content/inputs/`.

- `training/draft_h1/g4_inputs/wheels.txt` (363 URLs with SHA-256s, 361 wheels and two source archives),
  `fn_dl_retry.sh`, and `runtime/requirements.lock`.
- `patches/sglang/capture/sglang-v0.5.21-to-sub13-e5cb6d9.diff` and
  `patches/sglang/capture/dflash2_probe.diff`, staged as `dflash2-probe.diff`.
- The files in `training/draft_h1/code/g4/`, `training/draft_h1/code/sim.py`,
  `training/draft_h1/code/results/gate_parents.txt` and `training/keep_census/keep_384_code_bal4.json`.
- The evaluation probe's recorded histories, index and request metadata for T9. See the
  [draft fitting guide](../training/draft_h1/README.md#fit-and-evaluate-the-draft).

For T8, stage a disposable copy of the RadixArk weights before running `harness_capture_chain.sh`. The script
can delete those safetensors when free disk space is below its 90 GiB threshold, and T9's evaluation server reads
them through links, so restore them before T9 if they were deleted.

The trainer runs in `/content/sf` from SpecForge `53398a8` with the [trainer patch](../patches/README.md) applied.

### Restore paths for the census modules

For module-based commands, put `training/keep_census/fn_l4_merge_census.py` and `fn_l4_balance_keep.py` in
`agi2_modal_harness/comm/` along with `training/keep_census/capture/fn_l4_census_code.sh` and `fn_l4_up.sh`.
The server launcher reads `training/keep_census/capture/fn_bringup.py` from `/opt/dev/fn_bringup.py`.
For the module form of the expert chooser, stage `training/keep_c512/fn_expert_choose512.py` in
`agi2_modal_harness/comm/`. The [selection guide](../training/keep_c512/README.md) also gives a standalone command.

Several scripts default to `/vol/cache/dev/` and `/vol/models/`. The fitting guides give the expected inputs and flags.
