# Kaggle inputs and run settings

Use the public base-model mirror [`carriem7/qwen3-8-flash-next-nvfp4-mirror`](https://www.kaggle.com/models/carriem7/qwen3-8-flash-next-nvfp4-mirror/Transformers/hf-7b719225242a/1).
The scored runs mounted an identical copy that is not public. [Model files and fitted lists](MODELS.md) describes
how the notebooks assemble the checkpoint.

## Notebook versions

| Submission | Notebook used for scoring | Compile-cache seed to attach |
|---|---|---|
| 57853136 (sub 20) | [`1101-sub20-qwen38-flashnext-s64b-par2fix` v2](https://www.kaggle.com/code/carriem7/1101-sub20-qwen38-flashnext-s64b-par2fix?scriptVersionId=364353181) | [`1101-seed-flashnext-sub20-s64b-t8` v2](https://www.kaggle.com/code/carriem7/1101-seed-flashnext-sub20-s64b-t8?scriptVersionId=364348853) |
| 57785004 (sub 18) | [`1030-sub18-qwen38-flashnext-s48-5d76d78-int8` v3](https://www.kaggle.com/code/carriem7/1030-sub18-qwen38-flashnext-s48-5d76d78-int8?scriptVersionId=363637083) | [`1030-seed-flashnext-sub18-5d76d78-s48-int8` v3](https://www.kaggle.com/code/carriem7/1030-seed-flashnext-sub18-5d76d78-s48-int8?scriptVersionId=363632895) |

For either notebook, also attach the [competition data](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/data),
the [offline runtime v4](https://www.kaggle.com/code/carriem7/1006-sglang-runtime-cp312?scriptVersionId=355555024)
and all three [model variations at version 1](MODELS.md). Select the runtime and seed versions explicitly.
- **Attach only the seed of the chosen notebook.** The notebook uses the first seed it finds, and a seed is valid
  only for its own server build and runtime. With the wrong one the server compiles everything again, which adds
  about 18 minutes.
- **A copy of a Kaggle version lists the original base model,** which is private. Replace it with the public mirror
  above.

In the archived benchmark runs, the seeded server was ready 511 s (sub 18) and 526 s (sub 20) after the first cell.
The [patch guide](../patches/README.md) maps each notebook to its embedded server build and overlays.

The archived `*.metadata.json` files of submission 18 and its seed were saved before they were made public, so they
still say `"isPrivate": true`; both notebooks are public.

## Run on a custom test set

1. Copy one of the notebooks and attach a dataset containing `arc-agi_test_challenges.json` in the competition's
   format: a mapping from task ID to an object with `train` demonstration pairs and a `test` list of objects with
   `input` grids. Set `DATA` in cell `8d9d5b0a` to that dataset's folder; by default it takes the first file of that
   name under `/kaggle/input`.
2. Set `REHEARSAL = 'off'` for the full workload. Solving stops 11 h 45 min after the first cell; reporting, cleanup
   and the final write follow.
3. Read `/kaggle/working/submission.json` for the two attempts per test input, and compare them with your solutions.

A test ID found in neither the training nor the evaluation challenge file under `DATA`, as any new task is, makes the
notebook run as Kaggle's scored rerun would: the full workload, failed setting checks logged as warnings, and no
scoring against local solutions. The same happens when `KAGGLE_IS_COMPETITION_RERUN` is set (to anything except
empty, `0` or `false`). The last cell (`7e5fa109`) always writes and checks `submission.json`.

## Run a public benchmark

With only the competition data attached, the visible test file is a 240-task placeholder of public training tasks.
For a public evaluation benchmark, use `REHEARSAL = 'bench'` and `BENCH_SET = 'eval'`, as both notebooks ship:
- `BENCH_MINUTES` sets the solving time: 105 minutes in sub 20, 25 in sub 18.
- `BENCH_TASKS` selects a subset in the listed order. Sub 18 lists evaluation tasks 71–120 (70 test outputs); sub 20's
  is empty, which queues all 120 tasks (172 test outputs). For a longer run, empty it and raise `BENCH_MINUTES`.

The last cell prints the benchmark's score against the public solutions and writes the answers to
`/kaggle/working/answers_final.json`. `submission.json` still covers the attached test file.

## Runtime and hardware

- The scored runs used Kaggle's `NvidiaL4` accelerator, four NVIDIA L4 GPUs (24 GB each), about 188 GiB of host RAM
  and 46 usable CPU cores with AVX-512. Run with internet off. The competition has closed and this accelerator may
  be unavailable. Check the devices in `nvidia-smi`. The notebook counts lines containing `NVIDIA L4`, which also
  matches `NVIDIA L40S`.
- Both executions logged NVIDIA driver `580.178.04`. The notebook metadata pins the Linux x86-64 image
  `gcr.io/kaggle-private-byod/python@sha256:9a6bcfda6d03b2cd9aa347ee8c1bae62043328ed0f4afe9bda014e4f8bbf9435`
  (Ubuntu 24.04, glibc 2.39).
- The CPU-side all-reduce builds for AVX-512, AVX2/FMA or a generic CPU. If the host reducer fails to initialize,
  the server logs a warning and uses NCCL.
- The offline runtime is CPython 3.12.15 with 363 packages pinned in [`requirements.lock`](../runtime/requirements.lock).
  It includes PyTorch 2.13.0+cu130, SGLang 0.5.21, FlashInfer 0.6.18, Triton 3.7.1, TileLang 0.1.12 and NCCL 2.30.7.
- The notebooks put harness files, logs and `submission.json` under `/kaggle/working`. They install the local Python
  environment and apply the SGLang patches under `/tmp/sgl`, restore the seed caches under `/tmp/cache`, and write the
  assembled model, expert list and draft vocabulary under `/tmp`.

Sampling at temperature 1.0 and multi-GPU arithmetic can change individual answers. Each build was scored once.
