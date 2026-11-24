"""Harness notebooks' rounds on a Modal comm.dev box: notebook 1017-vision-harness-probe from E8 on (Kaggle's quota is
kept for submissions, the user, 2026-10-17 18:00), and 1018-harness-bench, submission 6's harness alone on any GPU type
(the user, 2026-10-18: harness effectiveness need not be measured on 4 x L4).

The notebook stays the one source of the harness. This runner starts our SGLang server from a bench-config arm (one
config per GPU type, e.g. configs/l4x4_harness.toml), then executes the notebook's own code cells, by id and in notebook
order: the notebook's SETUP (PROFILES below), then the experiment cells named with --run. Only Kaggle paths, Kaggle-only
steps and the request count (--slots) are substituted.

Before the server boots, a preflight checks the cells it will run, and stops with every problem it found: a cell that
is missing, a substitution that no longer matches, a cell that does not compile, a Kaggle path left in a cell, or a name
no run cell defines (a variable only Kaggle's kernel held, like E4's records in 1017_1818). `--check` runs the preflight
alone, on the laptop too. Records, live logs and attempt files land in $OUT/work/<probe|bench>/<batch>/ as on Kaggle;
the server log is $OUT/work/sglang.log.

  python -m agi2_modal_harness.comm.harness_nb --notebook NB.ipynb --data DIR [--config TOML] [--slots N]
      [--run CELL[,CELL...]] [--boot-only|--check]

Interact campaign (2026-10-19), for profiling a round under its real load:
  --round-minutes M       1018-harness-bench's ROUND_MINUTES (the round's stop) instead of the notebook's 300
  --batch-name NAME       round 1's batch name ('r1_l4x4' in cell hbr1l4x4) replaced, so records never mix
  --sample S              comm/interact_obs.Sampler every S seconds into the work directory (samples.jsonl)
  --nsys                  the server under `nsys profile` with cudaProfilerApi capture ranges (repeat), CUDA trace with
                          graph nodes, NVTX, GPU metrics at --nsys-metrics-hz; reports land in work/nsys/
  --profile-at M[,M...]   open a capture (--profile-kind nsys or torch) of --profile-steps forward passes at each
                          minute after the round's server is ready
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import builtins
import glob
import json
import os
import shutil
import subprocess
import symtable
import sys
import threading
import time
from pathlib import Path

CONFIG = "agi2_modal_harness/workloads/sglang_bench/configs/l4x4_harness.toml"
# The cells that build the harness, in notebook order (ids never shift): settings; the data; the server wait and
# restart_server(); the template check; the ARC prompt; pictures; the sandbox worker and the skills (written as files);
# the sandbox check; the harness's prompts, sessions, KV gate and attempt; E0's request helpers; the overfit check
# (written); run_batch; sub 5's history; E5's output lists; the park-with-pictures check.
SETUP = ["6f1359a6", "ee8d4a36", "6f2468cf", "3525222a", "71817d15", "49f20e6b", "78249d71", "74bf130f", "cae4f7b8",
         "aceb4d00", "a5509211", "e8365c62", "ca2b5fbb", "7a7b1aaa", "69c70482", "2da4ff36", "4ee008e5", "9e571b18",
         "8ae13300", "7ec1601a"]
# 1018-harness-bench: settings; the frozen set; the server wait, restart_server() and the template check; the ARC
# prompt; the worker and the skills (written); the sandbox check and its timer; the prompts; the pictures stub; sessions;
# the KV gate; the park check; the attempt; run_batch.
BENCH_SETUP = ["hbsettings", "hbdata", "6f2468cf", "5c3e5ea3", "6880000d", "49f20e6b", "74bf130f", "cae4f7b8",
               "aceb4d00", "hbtimer", "a5509211", "hbnopics", "e8365c62", "ca2b5fbb", "hbparkck", "7a7b1aaa", "hbround"]
RUNNER_NAMES = {"get_ipython", "RELAUNCH", "__name__"}   # what the runner puts in the cells' namespace


def log(msg: str) -> None:
    print(f"[harness-nb {time.strftime('%H:%M:%S')}] {msg}", flush=True)


class IPython:
    """What the cells ask of get_ipython(): `!cmd` lines and getoutput."""

    @staticmethod
    def system(cmd: str) -> None:
        subprocess.run(cmd, shell=True, check=False)

    @staticmethod
    def getoutput(cmd: str) -> list[str]:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.splitlines()


def subs(model_dir: str, data_dir: str) -> dict[str, list[tuple[str, str]]]:
    """Per cell id: (the Kaggle text, the Modal text). Every pair must match its cell."""
    return {
        "ee8d4a36": [("'/kaggle/input/**/arc-agi_evaluation_challenges.json'",
                      repr(f"{data_dir}/arc-agi_evaluation_challenges.json")),
                     ("df -h /tmp /kaggle/working", "df -h /tmp .")],
        "71817d15": [("[p for p in glob.glob('/kaggle/input/models/**/config.json', recursive=True) if 'fp8' in p][0]",
                      repr(f"{model_dir}/config.json"))],
        "3525222a": [("subprocess.run(['bash', 'launch_server.sh'], check=True)", "RELAUNCH()")],
        "4ee008e5": [("OUT = '/kaggle/working/probe'", "OUT = os.path.abspath('probe')")],
        "9e571b18": [("vs_history(e4, 'repl_s')", "pass   # E4's table needs E4's records, which are on Kaggle")],
        "8ae13300": [("e5 = await run_batch('e5_skills_pictures', jobs5, minutes=90)\nsummarize(e5)",
                      "e5 = []   # E5 ran on Kaggle; only its output lists are needed here")],
    }


def bench_subs(model_dir: str, data_dir: str, slots: int, round_minutes: float | None = None,
               batch_name: str | None = None) -> dict[str, list[tuple[str, str]]]:
    """1018-harness-bench: the set's directory, the model, the relaunch, the output directory and the request count;
    optionally the round's stop and round 1's batch name (interact campaign profiling rounds)."""
    settings = [("SLOTS = 18 ", f"SLOTS = {slots} ")]
    if round_minutes:
        settings.append(("ROUND_MINUTES = 300 ", f"ROUND_MINUTES = {round_minutes:g} "))
    extra = {"hbr1l4x4": [("'r1_l4x4'", repr(batch_name))]} if batch_name else {}
    return {
        **extra,
        "hbsettings": settings,
        "hbdata": [("os.path.dirname(glob.glob('/kaggle/input/**/bench24_challenges.json', recursive=True)[0])",
                    repr(data_dir))],
        "5c3e5ea3": [("subprocess.run(['bash', 'launch_server.sh'], check=True)", "RELAUNCH()")],
        "6880000d": [("[p for p in glob.glob('/kaggle/input/models/**/config.json', recursive=True) if 'fp8' in p][0]",
                      repr(f"{model_dir}/config.json"))],
        "hbround": [("OUT = '/kaggle/working/bench'", "OUT = os.path.abspath('bench')")],
    }


# 1019-sub7-qwen38-xhigh-harness-only, as submitted (the interact campaign profiles it, the user 2026-10-19: "do
# profiling on how actually sub 7 was built"): every cell from the settings to the solve, in notebook order, its own
# server launch included (launch_server.sh with sub 7's flags, PATCHES and its CPU split), only the runtime install,
# the build diffs (the box's image already carries .14 and the .15 diff is applied) and the finish's server kill left
# out. The notebook starts and waits for its own server; the runner starts its sampler and profile captures once cell
# 6f2468cf (the server wait) has run.
SUB7_SETUP = ["8b067093", "8d9d5b0a", "eb6a9d4e", "ed04db59", "41aad7bc", "6c30d806", "e143e8ee", "6f2468cf",
              "5c3e5ea3", "6880000d", "49f20e6b", "74bf130f", "cae4f7b8", "aceb4d00", "a5509211", "e8365c62", "ca2b5fbb",
              "7a7b1aaa"]
SUB7_RUN = ["a6c36a51", "7e5fa109"]   # the solve and the finish's summary
NSYS = ["nsys", "profile", "--trace=cuda,nvtx", "--cuda-graph-trace=node", "--sample=none", "--cpuctxsw=none",
        "--trace-fork-before-exec=true", "--capture-range=cudaProfilerApi", "--gpu-metrics-devices=all"]


def sub7_subs(model_dir: str, data_dir: str, slots: int, ctx: dict) -> dict[str, list[tuple[str, str]]]:
    """Sub 7 on a Modal box: Modal paths for Kaggle's, the image's own CUDA toolkit and Python for the offline runtime's,
    warm compile caches on the cache Volume, full strength for `ctx['hours']`, and observability only (one Decode line
    per step, scheduler NVTX, the device timer, and the server under nsys when ctx['nsys'] is set)."""
    cache, draft = ctx["cache"], ctx["draft_dir"]
    nsys = (" ".join(NSYS + [f"--capture-range-end=repeat:{max(1, ctx['captures'])}",
                             f"--gpu-metrics-frequency={ctx['nsys_hz']}", "-o", f"{ctx['work']}/nsys/report"]) + " "
            if ctx.get("nsys") else "")
    modal = "# (Modal: the image's own CUDA toolkit and Python)"
    return {
        "8b067093": [("HOURS = 11 + 35 / 60 ", f"HOURS = {ctx['hours']:g} "), ("SLOTS = 18 ", f"SLOTS = {slots} "),
                     ("REHEARSAL = 'smoke'", f"REHEARSAL = {ctx['rehearsal']!r}")],
        "8d9d5b0a": [("os.path.dirname(glob.glob('/kaggle/input/**/arc-agi_test_challenges.json', recursive=True)[0])",
                      repr(data_dir)),
                     ("SUBMISSION = '/kaggle/working/submission.json'", "SUBMISSION = os.path.abspath('submission.json')")],
        "eb6a9d4e": [("df -h /tmp /kaggle/working", "df -h /tmp .")],
        "6c30d806": [
            ("MODEL=$(dirname \"$(find -L /kaggle/input/models -path '*qwen3-8-27b-fp8*' -name config.json | head -1)\")",
             f"MODEL={model_dir}"),
            ("DRAFT=$(dirname \"$(find -L /kaggle/input/models -path '*dflash2*' -name config.json | head -1)\")",
             f"DRAFT={draft}"),
            ("export CUDA_HOME=/tmp/sgl/lib/python3.12/site-packages/nvidia/cu13", modal),
            ("export PATH=$CUDA_HOME/bin:$PATH:/tmp/sgl/bin NVCC_APPEND_FLAGS=\"-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK\"", modal),
            ("export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib", modal),
            ("ln -sfn lib $CUDA_HOME/lib64 && ln -sf libcudart.so.13 $CUDA_HOME/lib/libcudart.so", modal + " #"),
            ("export TRITON_CACHE_DIR=/tmp/cache/triton TORCHINDUCTOR_CACHE_DIR=/tmp/cache/inductor",
             f"export TRITON_CACHE_DIR={cache}/triton TORCHINDUCTOR_CACHE_DIR={cache}/inductor SGLANG_CACHE_DIR={cache}/sglang"),
            ("export FLASHINFER_WORKSPACE_BASE=/tmp/cache/flashinfer XDG_CACHE_HOME=/tmp/cache/xdg TORCH_EXTENSIONS_DIR=/tmp/cache/torch_extensions",
             f"export FLASHINFER_WORKSPACE_BASE={cache}/fi XDG_CACHE_HOME={cache}/xdg"),
            ("export HF_HUB_OFFLINE=1", "export SGLANG_ENABLE_NVTX_SCHEDULER=1 SGLANG_ENABLE_METRICS_DEVICE_TIMER=1 HF_HUB_OFFLINE=1"),
            ("nohup setsid /tmp/sgl/bin/python -m sglang.launch_server", f"nohup setsid {nsys}python -m sglang.launch_server"),
            ("--decode-log-interval 200", f"--decode-log-interval {ctx['decode_log_interval']}"
             + "".join(f" {x}" for x in ctx.get("server_args", []))),
        ] + ([] if ctx.get("kv_cache_dtype", "fp8_e4m3") == "fp8_e4m3" else [
            # A KV-pool format arm (e.g. agi2_v4: fp8 K + rotated int4 V, build .17 on): the gate reads the larger
            # pool from /v1/loads, so nothing else in the notebook changes. Its own switches come in through --env.
            ("--kv-cache-dtype fp8_e4m3", f"--kv-cache-dtype {ctx['kv_cache_dtype']}")]),
        "6880000d": [("[p for p in glob.glob('/kaggle/input/models/**/config.json', recursive=True) if 'fp8' in p][0]",
                      repr(f"{model_dir}/config.json"))],
        # The CPU reducer's thread count (sub 7: 8 threads on the last 8 CPUs; interact 1019_1421 measured 16 faster).
        "41aad7bc": [] if ctx.get("smr_threads", 8) == 8 else [
            ("KEEP = CPUS[:-8] if PATCHES", f"KEEP = CPUS[:-{ctx['smr_threads']}] if PATCHES"),
            ("SGLANG_SM_REDUCE_THREADS='8'", f"SGLANG_SM_REDUCE_THREADS='{ctx['smr_threads']}'"),
            ("SGLANG_SM_REDUCE_CPUS=','.join(map(str, CPUS[-8:]))",
             f"SGLANG_SM_REDUCE_CPUS=','.join(map(str, CPUS[-{ctx['smr_threads']}:]))")],
        "a6c36a51": [("LIVE = '/kaggle/working/attempts/live'", "LIVE = os.path.abspath('attempts/live')"),
                     ("f\"/kaggle/working/attempts/{rec['task']}", "f\"attempts/{rec['task']}")],
    }


def sub9_subs(model_dir: str, data_dir: str, slots: int, ctx: dict) -> dict[str, list[tuple[str, str]]]:
    """Sub 9 (sub 8's one conversation per task + KV gate 0.90 + exact turn boundaries) keeps sub 7's cell ids, so it
    takes sub 7's substitutions plus one: EXACT_TURNS switches itself off unless dflash_length_cap.py is in the
    offline venv, so the check looks where the image's build put it (build .18 on carries it)."""
    table = sub7_subs(model_dir, data_dir, slots, ctx)
    table["41aad7bc"] = table.get("41aad7bc", []) + [
        ("glob.glob('/tmp/sgl/lib/python3*/site-packages/sglang/srt/speculative/dflash_length_cap.py')",
         "glob.glob('/sgl-workspace/sglang/python/sglang/srt/speculative/dflash_length_cap.py')")]
    return table


def sub11_subs(model_dir: str, data_dir: str, slots: int, ctx: dict) -> dict[str, list[tuple[str, str]]]:
    """Submission 10 and 11 (sub 9's cell ids, plus the DeepInfer, all-levers and 0.5.22-picks switches) take sub 9's
    substitutions, plus the three checks that turn a lever on only where its patch is in place: on the Modal image the
    patched SGLang tree is /sgl-workspace/sglang/python (the box applies the picks diff over build .18)."""
    table = sub9_subs(model_dir, data_dir, slots, ctx)
    site, modal = "/tmp/sgl/lib/python3*/site-packages/sglang/srt/", "/sgl-workspace/sglang/python/sglang/srt/"
    table["41aad7bc"] = table.get("41aad7bc", []) + [
        (f"glob.glob('{site}{p}')", f"glob.glob('{modal}{p}')")
        for p in ("environ.py", "layers/attention/va4_attention.py", "entrypoints/api_contract.py")]
    if ctx.get("bench_minutes"):   # a bench of another length (e.g. the v2b2 comparison's 255 min)
        table["8b067093"] = table.get("8b067093", []) + [
            ("BENCH_MINUTES = 342 ", f"BENCH_MINUTES = {ctx['bench_minutes']:g} ")]
    if ctx.get("quiet_outputs"):   # a set with eval children (a validation split): no per-output verdict lines
        table["7e5fa109"] = table.get("7e5fa109", []) + [
            ("        print(f'  {tid} test {i}: {verdict}  ({how}; {sum(g is not None for g in grids)} grid(s){tail})')",
             "        pass   # per-output verdicts left out: this set's children are scored in aggregate only")]
    return table


def sub11_kaggle_subs(model_dir: str, data_dir: str, slots: int, ctx: dict) -> dict[str, list[tuple[str, str]]]:
    """Submission 11 as submitted (notebook 1022-sub11-qwen38-xhigh-vision-tail, v6), for the 2026-10-27 AREX-2 round:
    the Modal copy's table, plus the bench set read from the data folder (the notebook globs Kaggle's input for
    `{BENCH_SET}_challenges.json`; on the box the set's files sit beside the competition files)."""
    table = sub11_subs(model_dir, data_dir, slots, {**ctx, "bench_minutes": None})
    table["ed04db59"] = [("os.path.dirname(glob.glob(f'/kaggle/input/**/{BENCH_SET}_challenges.json', recursive=True)[0])",
                          repr(data_dir))]
    table["6f2468cf"] = [("# 60 min: /kaggle/input has read ~24 MB/s at times since 10-20",
                          "# 60 min (Kaggle's input mount reads slowly at times; on Modal the models Volume)")]
    if ctx.get("bench_minutes"):   # this notebook's own bench length is 171 (its 60-task v2b12 at Kaggle's pace)
        table["8b067093"] = table.get("8b067093", []) + [
            ("BENCH_MINUTES = 171 ", f"BENCH_MINUTES = {ctx['bench_minutes']:g} ")]
    if ctx.get("bench_set"):
        table["8b067093"] = table.get("8b067093", []) + [("BENCH_SET = 'v2b12'", f"BENCH_SET = {ctx['bench_set']!r}")]
    return table


# Submission 14 (2026-10-26, Flash-Next on 4 x L4: the QSA block tier at 16 slots, the 196k budget, draft h1), for the
# long-run study (docs/reports/1026_1244_longrun_sub14): every cell from the settings to the solve and the finish, in
# notebook order, its own server launch included (launch_server.sh with sub 14's switches, the warm boot when no seed is
# attached, as an unseeded Kaggle boot). Left out: the offline runtime's install (8a6a87dc, 2cf3eee7) and the build
# diff's apply (fnapply): the box carries agi2/feat/sub14 as runs/forks/sglang-v0.5.21-agi2.18-to-sub14.diff. fndiff
# still runs: it only writes the Kaggle diff file and checks its sha256, and its imports serve fnkeep.
SUB14_SETUP = ["8b067093", "8d9d5b0a", "eb6a9d4e", "ed04db59", "fndiff", "fnkeep", "41aad7bc", "6c30d806", "fnwarmreq",
               "seedchk", "hostmem", "e143e8ee", "6f2468cf", "5c3e5ea3", "fnbootck", "seedout", "6880000d",
               "49f20e6b", "74bf130f", "cae4f7b8", "aceb4d00", "a5509211", "e8365c62", "ca2b5fbb", "visionpy",
               "visionlk", "7a7b1aaa"]
SUB14_RUN = ["a6c36a51", "b3e91b1f", "7e5fa109", "benchtok"]   # the solve, the compile count + server stop, the finish


def sub14_subs(model_dir: str, data_dir: str, slots: int, ctx: dict) -> dict[str, list[tuple[str, str]]]:
    """Sub 14 on a Modal box: Modal paths for Kaggle's (the data folder, the model and draft, the submission and attempt
    files), the image's own CUDA toolkit and Python for the offline runtime's, no Kaggle compile-cache seed (so the
    launch runs its warm boot), the bench's set, length and slice, and observability only (a Decode line every
    ctx['decode_log_interval'] steps). The scheduler, the harness and the server's switches are the notebook's own."""
    draft = ctx["draft_dir"]
    modal = "# (Modal: the image's own CUDA toolkit and Python)"
    settings = [("SLOTS = 16 ", f"SLOTS = {slots} "), ("REHEARSAL = 'bench'", f"REHEARSAL = {ctx['rehearsal']!r}")]
    if ctx.get("bench_minutes"):
        settings.append(("BENCH_MINUTES = 90 ", f"BENCH_MINUTES = {ctx['bench_minutes']:g} "))
    if ctx.get("bench_set"):   # a set of ours: its whole file (sub 14's BENCH_TASKS slice is of the evaluation file)
        settings += [("BENCH_SET = 'eval'", f"BENCH_SET = {ctx['bench_set']!r}"),
                     ("BENCH_TASKS = [", "BENCH_TASKS = []   # (Modal profile: the whole bench set; sub 14's slice) [")]
    table = {
        "8b067093": settings,
        "8d9d5b0a": [("os.path.dirname(glob.glob('/kaggle/input/**/arc-agi_test_challenges.json', recursive=True)[0])",
                      repr(data_dir)),
                     ("SUBMISSION = '/kaggle/working/submission.json'", "SUBMISSION = os.path.abspath('submission.json')")],
        "eb6a9d4e": [("df -h /tmp /kaggle/working", "df -h /tmp .")],
        "ed04db59": [("glob.glob(f'/kaggle/input/**/{BENCH_SET}_challenges.json', recursive=True)[0]",
                      f"os.path.join({data_dir!r}, f'{{BENCH_SET}}_challenges.json')")],
        "41aad7bc": [("glob.glob('/tmp/sgl/lib/python3*/site-packages/sglang/srt/speculative/dflash_length_cap.py')",
                      "glob.glob('/sgl-workspace/sglang/python/sglang/srt/speculative/dflash_length_cap.py')")],
        "6c30d806": [
            ("MODEL=$(dirname \"$(find -L /kaggle/input/models -path '*qwen3-8-flash-next-nvfp4*' -name config.json | head -1)\")",
             f"MODEL={model_dir}"),
            ("DRAFT=$(dirname \"$(find -L /kaggle/input/models -path '*qwen3-8-flash-next-dflash2*' -name config.json | head -1)\")",
             f"DRAFT={draft}"),
            ("export CUDA_HOME=/tmp/sgl/lib/python3.12/site-packages/nvidia/cu13", modal),
            ("export PATH=$CUDA_HOME/bin:$PATH:/tmp/sgl/bin NVCC_APPEND_FLAGS=\"-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK\"", modal),
            ("export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib", modal),
            ("ln -sfn lib $CUDA_HOME/lib64 && ln -sf libcudart.so.13 $CUDA_HOME/lib/libcudart.so", modal + " #"),
            ("/tmp/sgl/bin/python", "python"),
            # no Kaggle seed on Modal: "no compile-cache seed attached", then the warm boot, as an unseeded Kaggle boot
            ("find -L /kaggle/input -maxdepth 5 -name jit_seed_manifest.json",
             "find -L /nonexistent_kaggle_input -maxdepth 5 -name jit_seed_manifest.json"),
            ("--decode-log-interval 200", f"--decode-log-interval {ctx['decode_log_interval']}"),
        ],
        "6f2468cf": [("# 60 min: /kaggle/input has read ~24 MB/s at times since 10-20",
                      "# 60 min (Kaggle's input mount reads slowly at times; on Modal the models Volume)")],
        "6880000d": [("[p for p in glob.glob('/kaggle/input/models/**/config.json', recursive=True) if 'flash-next-nvfp4' in p][0]",
                      repr(f"{model_dir}/config.json"))],
        "a6c36a51": [("LIVE = '/kaggle/working/attempts/live'", "LIVE = os.path.abspath('attempts/live')"),
                     ("f\"/kaggle/working/attempts/{rec['task']}", "f\"attempts/{rec['task']}")],
    }
    if ctx.get("quiet_outputs"):   # a set with eval children (a validation split): no per-output verdict lines
        table["7e5fa109"] = [
            ("        print(f'  {tid} test {i}: {verdict}  ({how}; {sum(g is not None for g in grids)} grid(s){tail})')",
             "        pass   # per-output verdicts left out: this set's children are scored in aggregate only")]
    if ctx.get("temperature") is not None:   # a temperature arm (the solve cell copies it into SAMPLING, a6c36a51:29)
        table["8b067093"].append(("TEMPERATURE = 1.0 ", f"TEMPERATURE = {ctx['temperature']:g} "))
    if ctx.get("graph_bs"):   # a slots arm above 16 (1026 24-slot slice): decode CUDA graphs for the larger batches too
        table.setdefault("41aad7bc", []).append(
            ("FLAGS['ARC_GRAPH_BS'] = '1 2 3 4 5 6 7 8 10 12 14 16'", f"FLAGS['ARC_GRAPH_BS'] = '{ctx['graph_bs']}'"))
    if ctx.get("mem_fraction"):   # headroom for larger graphs: DFlash drops its draft graph below 1.0 GB free
        table.setdefault("41aad7bc", []).append(
            ("FLAGS['ARC_MEM_FRACTION'] = '0.90'", f"FLAGS['ARC_MEM_FRACTION'] = '{ctx['mem_fraction']}'"))
    if ctx.get("tier_x"):   # the block tier's index slots per untiered token: at 24 slots the 2.5x pool is 1.66M tokens
        table.setdefault("41aad7bc", []).append(
            ("SGLANG_AGI2_QSA_BLOCK_TIER_X='2.5'", f"SGLANG_AGI2_QSA_BLOCK_TIER_X='{ctx['tier_x']}'"))
    if ctx.get("server_flags"):   # a candidate server's switches, into FLAGS after the tier's (10-28: POT + bundle A/B)
        extra = ", ".join(f"{k}={v!r}" for k, v in (kv.split("=", 1) for kv in ctx["server_flags"]))
        table.setdefault("41aad7bc", []).append(
            ("SGLANG_AGI2_QSA_BLOCK_TIER_HYSTERESIS='0.03',", f"SGLANG_AGI2_QSA_BLOCK_TIER_HYSTERESIS='0.03', {extra},"))
    if ctx.get("g4"):   # the launch, switches and machine cells are replaced; the others get the G4 lines appended
        for cid, subs_ in sub14_g4_overrides(ctx).items():
            if cid in ("eb6a9d4e", "41aad7bc", "6c30d806"):
                table[cid] = subs_
            else:
                table.setdefault(cid, []).extend(subs_)
    return table


# The 4 x L4-only switches of sub 14's server (cell 41aad7bc), dropped by the Colab G4 profile (--g4): the CPU
# all-reduce, the sm_89 Triton QSA decode path and the block tier that needs it (on sm_120 QSA decodes through FlashInfer),
# the TP verify shards, and the L4 memory and layout levers (one 96 GB GPU has the room).
SUB14_L4_ONLY = (
    "SGLANG_ENABLE_SM_REDUCE_ALLREDUCE", "SGLANG_SM_REDUCE_FMT", "SGLANG_SM_REDUCE_DOWNQ", "SGLANG_SM_REDUCE_THREADS",
    "SGLANG_SM_REDUCE_CHUNK_KB", "SGLANG_SM_REDUCE_PUBWARP", "SGLANG_SM_REDUCE_CPUS",
    "SGLANG_AGI2_QSA_TRITON_DECODE", "SGLANG_AGI2_QSA_IDX_GPC", "SGLANG_AGI2_QSA_IDX_STAGES",
    "SGLANG_AGI2_QSA_DIRECT_SPARSE", "SGLANG_AGI2_QSA_BLOCK_TIER_X", "SGLANG_AGI2_QSA_BLOCK_TIER_EVERY",
    "SGLANG_AGI2_QSA_BLOCK_TIER_MAX_GROUPS", "SGLANG_AGI2_QSA_BLOCK_TIER_RESERVE", "SGLANG_AGI2_QSA_BLOCK_TIER_LOW_WATER",
    "SGLANG_AGI2_QSA_BLOCK_TIER_HYSTERESIS",
    "SGLANG_ENABLE_DFLASH_SHARDED_VERIFY", "SGLANG_AGI2_VERIFY_TAIL",
    "SGLANG_AGI2_QSA_PAIR_LAYER_SPLIT", "SGLANG_AGI2_FN_VISION_HOST_BLOCKS", "SGLANG_FLASHINFER_WORKSPACE_SIZE",
    "SGLANG_AGI2_HOST_EMBED", "SGLANG_AGI2_MARLIN_NO_SWIZZLED", "SGLANG_AGI2_MARLIN_INPLACE_REPACK",
    "SGLANG_AGI2_PLE_DIRECT_READ", "SGLANG_AGI2_PREFETCH_EXCLUDE",
)


def sub14_g4_overrides(ctx: dict) -> dict[str, list[tuple[str, str]]]:
    """Sub 14 on one Colab G4 (RTX PRO 6000, sm_120, 96 GB; harness_nb --g4): the 10-26 long run's G4 copy
    (runs/plans/1026_longrun_g4/README.md). The Colab runtime is Kaggle's byte for byte (/tmp/sgl, the wheelhouse's
    CUDA 13), so the notebook's toolkit and Python lines stay; what changes is the server's shape: TP1 without EP, the
    4 x L4-only switches dropped (SUB14_L4_ONLY), mem 0.95 as every native G4 Flash-Next config, and the machine and
    block-tier checks. Marlin W4A16 numerics, the K384 keep list, draft h1, 16 slots, the scheduler and the harness stay
    sub 14's. G4 only: nothing on the Modal or Kaggle path passes --g4 (memory g4-changes-isolated)."""
    model_dir, draft = ctx["model_dir"], ctx["draft_dir"]
    l4_only = list(SUB14_L4_ONLY) + list(ctx.get("l4_only_extra", ()))   # sub 17 adds the block tier's POT switches
    drop = "\n".join([
        "# (Colab G4 profile, harness_nb --g4) one RTX PRO 6000 at TP1: no all-reduce, no EP, QSA decodes through",
        "# FlashInfer on sm_120, so the 4 x L4-only switches go (the CPU reducer, the sm_89 Triton QSA path and the block",
        "# tier it carries, the TP verify shards, the L4 memory and layout levers); mem 0.95 as the native G4 configs",
        f"G4_DROPPED = [k for k in {l4_only!r} if FLAGS.pop(k, None) is not None]",
        "FLAGS['ARC_MEM_FRACTION'] = '0.95'",
        "print('(Colab G4 profile) dropped', len(G4_DROPPED), 'L4-only switches:', G4_DROPPED)",
        ""])
    return {
        "eb6a9d4e": [("df -h /tmp /kaggle/working", "df -h /tmp ."),
                     ("check(n_l4 == 4, f'expected 4 NVIDIA L4, found {n_l4}')",
                      "check(n_l4 == 0 and 'RTX PRO 6000' in ' '.join(get_ipython().getoutput('nvidia-smi -L')), "
                      "'expected one RTX PRO 6000 (the Colab G4 profile)')")],
        "41aad7bc": [("FLAGS['ARC_GRAPH_BS']", drop + "FLAGS['ARC_GRAPH_BS']")],
        "6c30d806": [
            ("MODEL=$(dirname \"$(find -L /kaggle/input/models -path '*qwen3-8-flash-next-nvfp4*' -name config.json | head -1)\")",
             f"MODEL={model_dir}"),
            ("DRAFT=$(dirname \"$(find -L /kaggle/input/models -path '*qwen3-8-flash-next-dflash2*' -name config.json | head -1)\")",
             f"DRAFT={draft}"),
            ("find -L /kaggle/input -maxdepth 5 -name jit_seed_manifest.json",
             "find -L /nonexistent_kaggle_input -maxdepth 5 -name jit_seed_manifest.json"),
            ("--decode-log-interval 200", f"--decode-log-interval {ctx['decode_log_interval']}"),
            ("--port 30000 --tp 4 ", "--port 30000 --tp 1 "),
            ("--disable-radix-cache --ep-size 4", "--disable-radix-cache"),
        ],
        # At TP1 the server's log lines carry no "TP0 EP0" tag ("[time] KV Cache is allocated ..."): the cells that read
        # it take the tag out of their patterns (the status line, the boot facts, the finish's throughput, benchtok).
        "fnbootck": [
            (r"r'TP0[^\]]*\] KV Cache is allocated", r"r'\] KV Cache is allocated"),
            ("check(tier is not None and 'OFF (host-NUMA pages unavailable' not in boot,", "check(tier is None,"),
            ("'the QSA block tier did not switch on (host-NUMA pages refused?): 16 slots would park')",
             "'(Colab G4 profile) the QSA block tier must be off: it needs the sm_89 Triton decode path')")],
        "a6c36a51": [("if 'TP0' in l and 'Decode batch' in l", "if 'Decode batch' in l")],
        "7e5fa109": [(r"r'TP0[^\]]*\] Decode batch, #running-req", r"r'\] Decode batch, #running-req")],
        "benchtok": [(r"r'TP0[^\]]*\] KV Cache is allocated", r"r'\] KV Cache is allocated"),
                     (r"r'\[(\S+ \S+) TP0[^\]]*\] Decode batch", r"r'\[(\S+ \S+)\] Decode batch")],
    }


# Submission 15 (2026-10-27): sub 14's build and cell ids, with 24 slots, the pace-line fixes F1-F3 and the toolless
# look. Sub 14's table applies with three lines re-keyed to sub 15's text; on G4 (16 slots) the decode graphs go back to
# sub 14's list. --first-pass-only (the 10-27 hard-task lever runs, runs/plans/1027_hard_task_g4) turns the idle-end
# tails and the frugal switch off, so every first attempt runs and every acceptance gets its review.
SUB15_REKEY = {
    "SLOTS = 16 ": "SLOTS = 24 ",
    "BENCH_MINUTES = 90 ": "BENCH_MINUTES = 45 ",
    "'the QSA block tier did not switch on (host-NUMA pages refused?): 16 slots would park')":
        "'the QSA block tier did not switch on (host-NUMA pages refused?): 24 slots would park')",
}
SUB15_GRAPH_BS_24 = "FLAGS['ARC_GRAPH_BS'] = '1 2 3 4 5 6 7 8 10 12 14 16 18 20 22 24'"


def sub15_subs(model_dir: str, data_dir: str, slots: int, ctx: dict) -> dict[str, list[tuple[str, str]]]:
    table = {cid: [(SUB15_REKEY.get(old, old), new) for old, new in subs]
             for cid, subs in sub14_subs(model_dir, data_dir, slots, ctx).items()}
    if ctx.get("g4") and slots <= 16:   # the 1026 G4 copies' shape: decode graphs up to 16 (sub 14's list)
        table.setdefault("41aad7bc", []).append(
            (SUB15_GRAPH_BS_24, "FLAGS['ARC_GRAPH_BS'] = '1 2 3 4 5 6 7 8 10 12 14 16'"))
    if ctx.get("first_pass_only"):
        table.setdefault("8b067093", []).extend([
            ("TAIL = True ", "TAIL = False   # (harness_nb --first-pass-only) "),
            ("FRUGAL_AUTO = True ", "FRUGAL_AUTO = False   # (harness_nb --first-pass-only) ")])
    if ctx.get("ornith"):   # after the G4 lines: the Ornith server in place of Flash-Next's (ornith_overrides)
        if not ctx.get("g4"):
            raise SystemExit("--ornith needs --g4 (one RTX PRO 6000 at TP1)")
        for cid, subs_ in ornith_overrides(ctx).items():
            table.setdefault(cid, []).extend(subs_)
    if ctx.get("nanbeige"):   # after the G4 lines: Nanbeige4.3-33B's server in place of Flash-Next's (nanbeige_overrides)
        if not ctx.get("g4") or ctx.get("ornith"):
            raise SystemExit("--nanbeige needs --g4 (one RTX PRO 6000 at TP1) and excludes --ornith")
        for cid, subs_ in nanbeige_overrides(ctx).items():
            table.setdefault(cid, []).extend(subs_)
    return table


# Submission 17 as submitted (2026-10-29, notebook 1028-sub16-harness-decode-server-06e9a82 v2): sub 16's harness
# (transposed tails, tiers R/E, graded frugal) and the agi2/feat/sub17 06e9a8204c server, with sub 15's cell ids. Sub
# 15's table applies with its stop re-keyed to sub 17's 86 min; on one RTX PRO 6000 (--g4) the block tier's sub 17
# switches (promote-on-touch, its audit, per-layer residency) go with the tier, while the no-sync paths, exact top-k and
# the bf16 GDN state stay (runs/plans/1029_fresh61/README.md, the user 10-29: "make sure you running on the sub 17 branch").
SUB17_REKEY = {"BENCH_MINUTES = 45 ": "BENCH_MINUTES = 86 "}
SUB17_L4_ONLY = ("SGLANG_AGI2_QSA_BLOCK_TIER_POT", "SGLANG_AGI2_QSA_BLOCK_TIER_POT_AUDIT",
                 "SGLANG_AGI2_QSA_BLOCK_TIER_PERLAYER")


def sub17_subs(model_dir: str, data_dir: str, slots: int, ctx: dict) -> dict[str, list[tuple[str, str]]]:
    ctx = dict(ctx, l4_only_extra=SUB17_L4_ONLY)
    # --kaggle-runtime (2026-10-31, docs/reports/1031_1206_hard_output_check): sub 17 at TP4 on Kaggle's own offline
    # runtime in /tmp/sgl (the box setup installs it with the sub 17 diff), as sub 18's profile runs: the notebook's
    # CUDA toolkit and Python lines stay, and its own 0.896 mem-fraction line is the one --mem-fraction rewrites
    krt = bool(ctx.get("kaggle_runtime"))
    if krt and ctx.get("g4"):
        raise SystemExit("--kaggle-runtime is the 4-GPU path: --g4 already runs Kaggle's runtime on one GPU")
    table = {cid: [(SUB17_REKEY.get(old, old), new) for old, new in subs
                   if not (krt and old.startswith(SUB18_IMAGE_LINES))]
             for cid, subs in sub15_subs(model_dir, data_dir, slots,
                                         dict(ctx, mem_fraction="") if krt else ctx).items()}
    if krt and ctx.get("mem_fraction"):   # 0.448 on 4 x L40S keeps L4's 0.896 x 23,034 MiB = 20.6 GB static a GPU
        table.setdefault("41aad7bc", []).append(
            ("FLAGS['ARC_MEM_FRACTION'] = '0.896'", f"FLAGS['ARC_MEM_FRACTION'] = '{ctx['mem_fraction']}'"))
    if ctx.get("nccl_allreduce"):   # (10-31 hard-output check) NCCL's exact bf16 all-reduce instead of sm_reduce's int8
        # one (subs 15-18 share the line; sub 18's profile passes it through): the lossy step TP1 never has
        table.setdefault("41aad7bc", []).append(
            ("SGLANG_ENABLE_SM_REDUCE_ALLREDUCE='1',",
             "SGLANG_ENABLE_SM_REDUCE_ALLREDUCE='0',   # (harness_nb --nccl-allreduce) NCCL's exact all-reduce\n   "))
    return table


# Submission 18 as submitted (2026-10-30, notebook 1030-sub18-qwen38-flashnext-s48-5d76d78-int8 v3): 48 slots, up to 4
# attempts a task (MAX_ATTEMPTS), the QG experts overlay, the draft's 32k vocabulary (cell fnvocab) and the
# agi2/feat/sub18-levers 5d76d788f1 server, with sub 15's cell ids. 4 x L4 only, on Kaggle's own offline runtime in
# /tmp/sgl (the box setup installs it and applies the sub 18 diff, runs/plans/1030_sub18_modal/stage/setup.sh), so the
# notebook's CUDA toolkit and Python lines stay as on Kaggle; only Kaggle's input paths change (the data, RadixArk's
# files, draft h1, the QG experts folder). --max-attempts sets MAX_ATTEMPTS (the 2-attempt control arm).
SUB18_REKEY = {"SLOTS = 24 ": "SLOTS = 48 ", "BENCH_MINUTES = 86 ": "BENCH_MINUTES = 25 "}
SUB18_SETUP = [c for cid in SUB14_SETUP for c in ([cid, "fnvocab"] if cid == "fnkeep" else [cid])]
SUB18_IMAGE_LINES = ("export CUDA_HOME=", "export PATH=$CUDA_HOME", "export LD_LIBRARY_PATH=", "ln -sfn lib $CUDA_HOME",
                     "/tmp/sgl/bin/python", "glob.glob('/tmp/sgl/lib/python3*/site-packages/sglang/srt/speculative/")


def sub18_subs(model_dir: str, data_dir: str, slots: int, ctx: dict, rekey: dict | None = None) -> dict[str, list[tuple[str, str]]]:
    if ctx.get("g4") or ctx.get("ornith") or ctx.get("nanbeige"):
        raise SystemExit("sub 18's profile runs on 4 x L4 as submitted: no --g4, --ornith, --nanbeige")
    if not ctx.get("qg_dir") and not ctx.get("radixark_experts"):
        raise SystemExit("sub 18's profile needs --qg-dir (the QG experts' 192 shards on local disk) "
                         "or --radixark-experts")
    if ctx.get("qg_dir") and ctx.get("radixark_experts"):
        raise SystemExit("--qg-dir and --radixark-experts are exclusive")
    rekey = rekey or SUB18_REKEY
    table = {cid: [(rekey.get(old, old), new) for old, new in subs
                   if not old.startswith(SUB18_IMAGE_LINES)]   # Kaggle's runtime in /tmp/sgl: those lines stay
             for cid, subs in sub17_subs(model_dir, data_dir, slots, dict(ctx, mem_fraction="", tier_x="",
                                                                            graph_bs="")).items()}
    if ctx.get("graph_bs"):   # sub 18's own decode-graph line, e.g. '1 2 4 8 12 16 20 24' for a 24-slot arm
        table.setdefault("41aad7bc", []).append(
            ("FLAGS['ARC_GRAPH_BS'] = '1 2 4 8 12 16 24 32 40 48'", f"FLAGS['ARC_GRAPH_BS'] = '{ctx['graph_bs']}'"))
    if ctx.get("tier_x"):   # sub 18's own line (sub 14's '2.5' is not in this notebook): a host with less RAM than
        # Kaggle's 188 GB pins fewer host rows (X 7.5 pins ~13.4 GB a rank; the vast box has 109 GB)
        table.setdefault("41aad7bc", []).append(
            ("SGLANG_AGI2_QSA_BLOCK_TIER_X='7.5'", f"SGLANG_AGI2_QSA_BLOCK_TIER_X='{ctx['tier_x']}'"))
    if ctx.get("mem_fraction"):   # sub 18's own line (sub 14's '0.90' is not in this notebook). On 4 x L40S, 0.44 keeps
        # L4's ~20 GB static a GPU, so the KV pool and the block tier's pinned host rows stay L4's (0.89 there OOM-killed
        # the box: ~5x the pool, ~270 GB of host rows; runs/plans/1030_sub18_modal/README.md)
        table.setdefault("41aad7bc", []).append(
            ("FLAGS['ARC_MEM_FRACTION'] = '0.89'", f"FLAGS['ARC_MEM_FRACTION'] = '{ctx['mem_fraction']}'"))
    qg_line = "QG=$(dirname \"$(find -L /kaggle/input/models -path '*agi2-qg1-8352943a271b*' -name MANIFEST.sha256 | head -1)\")"
    if ctx.get("radixark_experts"):   # (10-31 hard-output check, arm c) RadixArk's own experts as subs 12-17: the overlay
        # cell's QG folder unset, and its fail-closed exit turned into a log line
        table.setdefault("6c30d806", []).extend([
            (qg_line, "QG=.   # (harness_nb --radixark-experts) RadixArk's own experts: no QG overlay"),
            ('    echo "QG experts not attached"; exit 1',
             '    echo "QG experts not attached: RadixArk\'s own experts (harness_nb --radixark-experts)"')])
    else:
        table.setdefault("6c30d806", []).append((qg_line, f"QG={ctx['qg_dir']}"))
    if ctx.get("bf16_index_keys"):   # (10-31 hard-output check, arm c) INT8 off = the QSA index keys in BF16, as subs 12-17
        table.setdefault("41aad7bc", []).append(
            ("    SGLANG_AGI2_QSA_INDEX_INT8='1',",
             "    SGLANG_AGI2_QSA_INDEX_INT8='0',   # (harness_nb --bf16-index-keys) BF16 index keys, as subs 12-17"))
    # --first-pass-only: sub 15's table turns TAIL and FRUGAL_AUTO off, and TAIL = False alone ends every task at its
    # first attempt (consider_tail returns before queueing a later attempt or a retry, a6c36a51:205); MAX_ATTEMPTS stays
    # as submitted, since the machine check refuses anything outside 2-4 (cell eb6a9d4e; 10-31 arm b's first launch)
    if ctx.get("max_attempts"):
        table.setdefault("8b067093", []).append(
            ("MAX_ATTEMPTS = 4 ", f"MAX_ATTEMPTS = {int(ctx['max_attempts'])} "))
    if ctx.get("park_host_gb"):   # the streaming-session park store's host budget a rank (8 as submitted)
        table.setdefault("41aad7bc", []).append(
            ("SGLANG_STREAMING_SESSION_PARK_HOST_GB='8'", f"SGLANG_STREAMING_SESSION_PARK_HOST_GB='{ctx['park_host_gb']}'"))
    if ctx.get("chunked_prefill") or ctx.get("launch_args"):   # the server launch line (cell 6c30d806), e.g. the
        # prefill-pass arm: a larger chunk + SGLang's PrefillDelayer (runs/plans/1030_sub18_modal/README.md)
        chunk = int(ctx.get("chunked_prefill") or 1024)
        extra = " ".join(ctx.get("launch_args") or [])
        table.setdefault("6c30d806", []).append(
            ("--chunked-prefill-size 1024 ", f"--chunked-prefill-size {chunk} {extra + ' ' if extra else ''}"))
    return table


# Nanbeige4.3-33B pilot (2026-10-29, runs/plans/1029_nanbeige_pilot; harness_nb --g4 --nanbeige): the levers notebook's
# harness on one RTX PRO 6000 with Nanbeige's server. Nanbeige4.3-33B is a dense LlamaForCausalLM (all full attention,
# 8 KV heads, no vision), served here from ProCreations' compressed-tensors FP8 build (auto-detected) with
# ProCreations' DFlash2 draft. So every hybrid and MoE switch goes (GDN fold, mamba cache and dtype, ReplaySSM verify,
# modelopt/Marlin MoE, PLE offload, image processor), the picture look is off (VISION_LOOK, submission 10's attempt: a
# text-only model), the warm boot skips its picture request, and the draft's KV is full length (this draft attends to
# the whole context: a 2k window would only cost acceptance). The template has no effort line (the Ornith checks).
# Sampling: the vendor's agentic setting, T 1.0 and top_p 0.95 with no top-k. G4 only: nothing else passes --nanbeige.
NANBEIGE_DROP = ("SGLANG_AGI2_FN_EXPERT_KEEP", "SGLANG_AGI2_FN_EXPERT_KEEP_ANY_ORDER", "SGLANG_AGI2_QSA_SESSION_PREFIX",
                 "SGLANG_AGI2_QSA_RING2", "SGLANG_ENABLE_GDN_REPLAYSSM_FOLD", "SGLANG_AGI2_SWA_ONLY_WRAPPER0",
                 "SGLANG_MM_ENCODE_MAX_TOKENS", "SGLANG_VLM_CACHE_SIZE_MB")
NANBEIGE_WINDOW_KV = ("SGLANG_DFLASH_DRAFT_WINDOW_KV", "SGLANG_AGI2_DRAFT_WINDOW_KV_PAGED")


def nanbeige_overrides(ctx: dict) -> dict[str, list[tuple[str, str]]]:
    drop = list(NANBEIGE_DROP) + ([] if ctx.get("nanbeige_window_kv") else list(NANBEIGE_WINDOW_KV))
    # No host parking (--nanbeige-no-park): a park carries the target KV and the draft's window ring, but not a
    # full-length draft KV pool, and this draft has no window (sliding_window null). So a parked-and-restored
    # conversation would draft from stale slots. Without parking, the KV gate closes the session and later rebuilds it
    # from its history, and that prefill recomputes the draft's KV.
    if ctx.get("nanbeige_no_park"):
        drop += ["SGLANG_ENABLE_STREAMING_SESSION_PARK", "SGLANG_STREAMING_SESSION_PARK_HOST_GB"]
    lines = [
        "# (Nanbeige profile, harness_nb --nanbeige) a dense LlamaForCausalLM: no GDN, no MoE, no QSA, no vision",
        f"NANBEIGE_DROPPED = [k for k in {drop!r} if FLAGS.pop(k, None) is not None]",
        "print('(Nanbeige profile) dropped', len(NANBEIGE_DROPPED), 'switches:', NANBEIGE_DROPPED)"]
    if ctx.get("nanbeige_mem"):
        lines.append(f"FLAGS['ARC_MEM_FRACTION'] = '{ctx['nanbeige_mem']}'   # (Nanbeige profile)")
    kv = ctx.get("nanbeige_kv") or "fp8_e4m3"
    launch = [
        (" --mamba-ssm-dtype float32", ""),
        ("--quantization modelopt_fp4 --fp4-gemm-backend marlin --moe-runner-backend marlin --page-size 64 "
         "--ple-offload-embedding", "--page-size 64"),
        (" --max-mamba-cache-size ${ARC_MAMBA_SLOTS:-18}", ""),
        ("--enable-linear-replayssm-spec --disable-prefill-cuda-graph --linear-attn-verify-backend triton",
         "--disable-prefill-cuda-graph"),
        ("--image-processor-backend pil ", "")]
    if kv != "fp8_e4m3":
        launch.append(("--kv-cache-dtype fp8_e4m3", f"--kv-cache-dtype {kv}"))
    return {
        "8b067093": [("VISION_LOOK = True ", "VISION_LOOK = False   # (Nanbeige profile: a text-only model) ")],
        "41aad7bc": [("FLAGS['ARC_GRAPH_BS']", "\n".join(lines + [""]) + "FLAGS['ARC_GRAPH_BS']")],
        "6c30d806": launch,
        "fnwarmreq": [("img = Image.new(", "raise SystemExit(0)   # (Nanbeige profile) text-only: no warm picture request\n"
                                           "img = Image.new(")],
        "fnbootck": [
            ("check(keep is not None and keep[1] == '384', 'the 384-expert keep list did not load')",
             "check(keep is None, '(Nanbeige profile) no expert keep list may load')"),
            ("check(pool is not None and int(pool[1]) > 600_000,", "check(pool is not None and int(pool[1]) > 200_000,")],
        "6880000d": [
            ("check('Reasoning effort is set to xhigh' in rendered, 'the xhigh effort line is missing')",
             "check('Reasoning effort' not in rendered and rendered.endswith('<|im_start|>assistant\\n<think>\\n'), "
             "'(Nanbeige profile) the render is not the thinking template without an effort line')")],
        "e8365c62": [
            ("and 'Reasoning effort is set to xhigh' in TPREFIX", "and 'Reasoning effort' not in TPREFIX"),
            ("and 'Reasoning effort is set to xhigh' in pre", "and 'Reasoning effort' not in pre")],
        "49f20e6b": [
            ("SAMPLING = dict(temperature=1.0, top_p=0.95, top_k=20, min_p=0.0)",
             "SAMPLING = dict(temperature=1.0, top_p=0.95, top_k=-1, min_p=0.0)   # (Nanbeige profile) the vendor's "
             "agentic setting, no top-k; temperature from TEMPERATURE")],
    }


# Ornith-1.5-122B pilot (2026-10-29, runs/plans/1029_ornith_pilot; harness_nb --g4 --ornith): the levers notebook's
# harness unchanged, on one RTX PRO 6000, with Ornith's server in place of Flash-Next's. Ornith-1.5-122B-NVFP4 is
# Qwen3_5MoeForConditionalGeneration (ModelOpt MIXED_PRECISION: FP8 attention/GDN projections, W4A16 NVFP4 experts,
# shared experts and lm_head), so the Flash-Next-only parts go: the expert keep list, the QSA session switches, the PLE
# offload and modelopt_fp4. Its chat template is Qwen3.5's, with no reasoning-effort line, so the three checks that assert
# Flash-Next's xhigh line assert its absence instead (the harness's own turn rendering is the same text: <think>, the
# qwen3_coder tool calls and the vision tokens are identical). Sampling: the vendor's 122B settings (temperature from
# --temperature, presence_penalty from --presence-penalty). G4 only: nothing on the Kaggle or Modal path passes --ornith.
ORNITH_FN_ONLY = ("SGLANG_AGI2_FN_EXPERT_KEEP", "SGLANG_AGI2_FN_EXPERT_KEEP_ANY_ORDER", "SGLANG_AGI2_QSA_SESSION_PREFIX",
                  "SGLANG_AGI2_QSA_RING2")


def ornith_overrides(ctx: dict) -> dict[str, list[tuple[str, str]]]:
    pres = float(ctx.get("presence_penalty") or 0.0)
    lines = [
        "# (Ornith profile, harness_nb --ornith) Qwen3_5MoeForConditionalGeneration: no expert keep list, no QSA",
        f"ORNITH_DROPPED = [k for k in {list(ORNITH_FN_ONLY)!r} if FLAGS.pop(k, None) is not None]",
        "print('(Ornith profile) dropped', len(ORNITH_DROPPED), 'Flash-Next-only switches:', ORNITH_DROPPED)"]
    # KV pool levers (exact), the 10-29 completion arm: the embedding table back in pinned host RAM (the G4 profile
    # drops it as an L4 lever; qwen3_5._build_embed_tokens serves it bit-identically) and a larger static fraction.
    if ctx.get("ornith_host_embed"):
        lines.append("FLAGS['SGLANG_AGI2_HOST_EMBED'] = '1'   # (Ornith profile) the 1.5 GB embedding table in pinned host RAM")
    if ctx.get("ornith_mem"):
        lines.append(f"FLAGS['ARC_MEM_FRACTION'] = '{ctx['ornith_mem']}'   # (Ornith profile) a larger KV pool")
    drop = "\n".join(lines + [""])
    return {
        "41aad7bc": [("FLAGS['ARC_GRAPH_BS']", drop + "FLAGS['ARC_GRAPH_BS']")],
        "6c30d806": [
            ("--quantization modelopt_fp4 --fp4-gemm-backend marlin --moe-runner-backend marlin --page-size 64 "
             "--ple-offload-embedding",
             "--quantization modelopt_mixed --fp4-gemm-backend marlin --moe-runner-backend marlin --page-size 64")],
        "fnbootck": [
            ("check(keep is not None and keep[1] == '384', 'the 384-expert keep list did not load')",
             "check(keep is None, '(Ornith profile) no expert keep list may load')"),
            ("check(pool is not None and int(pool[1]) > 600_000,", "check(pool is not None and int(pool[1]) > 200_000,")],
        "6880000d": [
            ("check('Reasoning effort is set to xhigh' in rendered, 'the xhigh effort line is missing')",
             "check('Reasoning effort' not in rendered and rendered.endswith('<|im_start|>assistant\\n<think>\\n'), "
             "'(Ornith profile) the render is not the thinking template without an effort line')")],
        "e8365c62": [
            ("and 'Reasoning effort is set to xhigh' in TPREFIX", "and 'Reasoning effort' not in TPREFIX"),
            ("and 'Reasoning effort is set to xhigh' in pre", "and 'Reasoning effort' not in pre")],
        "49f20e6b": [
            ("SAMPLING = dict(temperature=1.0, top_p=0.95, top_k=20, min_p=0.0)",
             f"SAMPLING = dict(temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty={pres:g}, "
             "repetition_penalty=1.0)   # (Ornith profile) the vendor's 122B settings; temperature from TEMPERATURE")],
    }


# The 12-hour run of S64b (2026-10-31, runs/plans/1031_long_s64b; notebook built by its build_nb.py from the x15 probe):
# sub 18's cells plus the notebook's own build cells, which patch a clean v0.5.21 runtime in /tmp/sgl exactly as on
# Kaggle (the sub 20 diff, the request split's increment, the session fork, the three later memory levers; the box
# setup, stage/setup_long.sh, installs that runtime unpatched). Its 64 slots and 675-min stop are the notebook's own
# text, so --slots 64 --bench-minutes 675 rewrite them to themselves.
LONG_SETUP = [c for cid in SUB18_SETUP for c in ([cid, "fnapply", "fnrs", "fnfork", "fnml3"] if cid == "fndiff" else [cid])]
LONG_REKEY = dict(SUB18_REKEY, **{"SLOTS = 24 ": "SLOTS = 64 ", "BENCH_MINUTES = 86 ": "BENCH_MINUTES = 675 "})


# The 11-01 prompt probe on one Colab G4 (RTX PRO 6000, sm_120, 96 GB; harness_nb --g4), on submission 20's stack (the
# user, 11-01: "you should use sub20 stack"): the probe notebook (sub 20 v2 with its job-runner solve cell,
# runs/plans/1101_prompts), its build cells patching a clean runtime as on Kaggle (LONG_SETUP: fnapply, fnrs, fnfork,
# fnml3), the QG experts and the K384 c512 keep list, at TP1 without EP and 16 slots. Kaggle's path substitutions are
# sub 18's (sub18_subs, its --g4 refusal bypassed here only); the G4 lines are sub 14's G4 overrides re-keyed to sub 20's
# text. Dropped on one GPU: sub 17's 4 x L4-only list plus what subs 18-20 added for L4 (the block tier's stage cache,
# INT8 index keys (TileLang kernels built for the sm_89 path), the reducer's split and envelope, the EP-only paths, the
# logits shard buffer). G4 only: nothing on the Modal or Kaggle path uses this profile (memory g4-changes-isolated).
SUB20_G4_L4_ONLY = SUB14_L4_ONLY + SUB17_L4_ONLY + (
    "SGLANG_AGI2_QSA_BLOCK_TIER_STAGE", "SGLANG_AGI2_QSA_BLOCK_TIER_STAGE_MAX", "SGLANG_AGI2_QSA_BLOCK_TIER_STAGE_CACHE",
    "SGLANG_AGI2_QSA_BLOCK_TIER_WINDOW", "SGLANG_AGI2_QSA_INDEX_INT8", "SGLANG_AGI2_QSA_IDX_SHARED_INT8",
    "SGLANG_AGI2_QSA_NO_LOGITS_FILL", "SGLANG_AGI2_SMR_SPLIT", "SGLANG_SM_REDUCE_MAX_KB", "SGLANG_AGI2_SHARD_LOGITS_BUFFER",
    "SGLANG_AGI2_MOE_EP_SUM", "SGLANG_AGI2_EP_MAP_TRITON", "SGLANG_AGI2_EPLB_MAP_HOST",
    # (11-01 06:45) the decode speed levers added after sub 17 (the last build checked at TP1): with them on, one G4
    # wrote every tool call scrambled ("<=python python>", "</</parametertool>"; run 1101_0555_colab-g4-prompts-s1),
    # while 4 x L4 (TP4) writes them cleanly. All are exact-math speed levers, so dropping them keeps the numerics.
    "SGLANG_AGI2_SPARSE_ACCEPT", "SGLANG_AGI2_SPARSE_ACCEPT_CHUNK", "SGLANG_AGI2_DRAFT_SUCC_COMPACT",
    "SGLANG_AGI2_DRAFT_VOCAB", "SGLANG_AGI2_LAZY_GRAPH_MASK", "SGLANG_AGI2_SWA_GRAPH_IDX_CAP", "SGLANG_AGI2_MROPE_ASYNC",
    "SGLANG_AGI2_MOE_EMPTY_CACHE13")
PROBE_REKEY = dict(SUB18_REKEY, **{"SLOTS = 24 ": "SLOTS = 64 ", "BENCH_MINUTES = 86 ": "BENCH_MINUTES = 200 "})


def sub20_g4_subs(model_dir: str, data_dir: str, slots: int, ctx: dict) -> dict[str, list[tuple[str, str]]]:
    if slots > 16:
        raise SystemExit("the G4 sub 20 profile runs 16 slots (decode graphs to 16)")
    table = sub18_subs(model_dir, data_dir, slots, dict(ctx, g4=False, ornith=False, nanbeige=False), rekey=PROBE_REKEY)
    drop = "\n".join([
        "# (Colab G4 profile for sub 20, harness_nb --g4) one RTX PRO 6000 at TP1: no all-reduce, no EP, QSA decodes",
        "# through FlashInfer on sm_120, so the 4 x L4-only switches go; mem 0.95 as the native G4 configs; 16 slots",
        f"G4_DROPPED = [k for k in {SUB20_G4_L4_ONLY!r} if FLAGS.pop(k, None) is not None]",
        "FLAGS['ARC_MEM_FRACTION'] = '0.95'",
        "print('(Colab G4 profile) dropped', len(G4_DROPPED), 'L4-only switches:', G4_DROPPED)",
        ""])
    table.setdefault("eb6a9d4e", []).append(
        ("check(n_l4 == 4, f'expected 4 NVIDIA L4, found {n_l4}')",
         "check(n_l4 == 0 and 'RTX PRO 6000' in ' '.join(get_ipython().getoutput('nvidia-smi -L')), "
         "'expected one RTX PRO 6000 (the Colab G4 profile)')"))
    table.setdefault("41aad7bc", []).append(
        ("FLAGS['ARC_GRAPH_BS'] = '1 2 4 8 12 16 24 32 40 48 56 64'",
         drop + "FLAGS['ARC_GRAPH_BS'] = '1 2 3 4 5 6 7 8 10 12 14 16'"))
    table.setdefault("6c30d806", []).extend([("--port 30000 --tp 4 ", "--port 30000 --tp 1 "),
                                            ("--disable-radix-cache --ep-size 4", "--disable-radix-cache")])
    table.setdefault("fnbootck", []).extend([
        (r"r'TP0[^\]]*\] KV Cache is allocated", r"r'\] KV Cache is allocated"),
        ("check(pool is not None and int(pool[1]) > 600_000,", "check(pool is not None and int(pool[1]) > 100_000,"),
        ("check(tier is not None and 'OFF (host-NUMA pages unavailable' not in boot,", "check(tier is None,"),
        ("'the QSA block tier did not switch on (host-NUMA pages refused?): 64 slots would park')",
         "'(Colab G4 profile) the QSA block tier must be off: it needs the sm_89 Triton decode path')")])
    table.setdefault("a6c36a51", []).append(("if 'TP0' in l and 'Decode batch' in l", "if 'Decode batch' in l"))
    table.setdefault("7e5fa109", []).append((r"r'TP0[^\]]*\] Decode batch, #running-req", r"r'\] Decode batch, #running-req"))
    table.setdefault("benchtok", []).extend([(r"r'TP0[^\]]*\] KV Cache is allocated", r"r'\] KV Cache is allocated"),
                                             (r"r'\[(\S+ \S+) TP0[^\]]*\] Decode batch", r"r'\[(\S+ \S+)\] Decode batch")])
    return table


# notebook stem -> (its SETUP cells, its substitutions given (model_dir, data_dir, slots, round_minutes, batch_name,
# ctx), and options: own_server (the notebook launches its server), obs_after (the cell after which the sampler and the
# captures start), run (the cells --run defaults to))
PROFILES = {
    "1017-vision-harness-probe": (SETUP, lambda model_dir, data_dir, slots, *_: subs(model_dir, data_dir), {}),
    "1018-harness-bench": (BENCH_SETUP, lambda m, d, s, rm, bn, ctx: bench_subs(m, d, s, rm, bn), {}),
    "1019-sub7-qwen38-xhigh-harness-only": (SUB7_SETUP, lambda m, d, s, rm, bn, ctx: sub7_subs(m, d, s, ctx),
                                            {"own_server": True, "obs_after": "6f2468cf", "run": SUB7_RUN}),
    "1020-sub9-qwen38-xhigh-gate90-exact-cuts": (SUB7_SETUP, lambda m, d, s, rm, bn, ctx: sub9_subs(m, d, s, ctx),
                                                 {"own_server": True, "obs_after": "6f2468cf", "run": SUB7_RUN}),
    # submission 11 (2026-10-22): sub 10's cells + the idle-end tails, wrap-up and pace line, and the picture look
    # (its pictures module and the look, before the attempt); `benchtok` reads a bench
    "1022-sub11-qwen38-xhigh-idle-tail": (SUB7_SETUP[:-1] + ["visionpy", "visionlk", "7a7b1aaa"],
                                          lambda m, d, s, rm, bn, ctx: sub11_subs(m, d, s, ctx),
                                          {"own_server": True, "obs_after": "6f2468cf",
                                           "run": SUB7_RUN + ["benchtok"]}),
    # submission 11 as submitted (v6): the same cells as its Modal copy; the AREX-2 round of 2026-10-27 runs it
    "1022-sub11-qwen38-xhigh-vision-tail": (SUB7_SETUP[:-1] + ["visionpy", "visionlk", "7a7b1aaa"],
                                            lambda m, d, s, rm, bn, ctx: sub11_kaggle_subs(m, d, s, ctx),
                                            {"own_server": True, "obs_after": "6f2468cf",
                                             "run": SUB7_RUN + ["benchtok"]}),
    # submission 14 (2026-10-26): Flash-Next, the QSA block tier at 16 slots, draft h1 (the long-run study)
    "1026-sub14-qwen38-flashnext-blocktier16-h1": (SUB14_SETUP, lambda m, d, s, rm, bn, ctx: sub14_subs(m, d, s, ctx),
                                                   {"own_server": True, "obs_after": "6f2468cf", "run": SUB14_RUN}),
    # submission 15 (2026-10-27): sub 14's cell ids and build, 24 slots, F1-F3, the toolless look
    "1027-sub15-qwen38-flashnext-slots24-f123-look": (SUB14_SETUP,
                                                      lambda m, d, s, rm, bn, ctx: sub15_subs(m, d, s, ctx),
                                                      {"own_server": True, "obs_after": "6f2468cf", "run": SUB14_RUN}),
    # levers 1 + 2 (2026-10-27, runs/plans/1027_confirm51): sub 15 + transposed second attempts + graded frugal, the same
    # cell ids and substitutions (built by runs/plans/1027_confirm51/build_nb.py)
    "1027-sub15-transposed-tails-graded-frugal": (SUB14_SETUP, lambda m, d, s, rm, bn, ctx: sub15_subs(m, d, s, ctx),
                                                  {"own_server": True, "obs_after": "6f2468cf", "run": SUB14_RUN}),
    # lever 3 (2026-10-27, runs/plans/1027_lever3): sub 15's setup and substitutions, then branches of recorded
    # attempts at their first acceptance in place of the solve (built by runs/plans/1027_lever3/build_l3_nb.py)
    "1027-lever3-review-replay": (SUB14_SETUP, lambda m, d, s, rm, bn, ctx: sub15_subs(m, d, s, ctx),
                                  {"own_server": True, "obs_after": "6f2468cf",
                                   "run": ["l3flagpy", "l3notepy", "l3help", "l3run", "b3e91b1f"]}),
    # submission 17 as submitted (2026-10-29): sub 16's harness + the sub 17 server, sub 15's cell ids (sub17_subs)
    "1028-sub16-harness-decode-server-06e9a82": (SUB14_SETUP, lambda m, d, s, rm, bn, ctx: sub17_subs(m, d, s, ctx),
                                                 {"own_server": True, "obs_after": "6f2468cf", "run": SUB14_RUN}),
    # the 10-31 hard-output check's arm p (runs/plans/1031_hard_check/build_plain_nb.py): sub 17's notebook + one cell of
    # plain attempts (question in, answer out) run in place of the solve, then the server stop
    # (SETUP without the harness attempt 7a7b1aaa: it needs the solve's names, and a plain attempt never calls it)
    "1031-sub17-plain-hard": ([c for c in SUB14_SETUP if c != "7a7b1aaa"],
                              lambda m, d, s, rm, bn, ctx: sub17_subs(m, d, s, ctx),
                              {"own_server": True, "obs_after": "6f2468cf", "run": ["plainrun", "b3e91b1f"]}),
    # submission 18 as submitted (2026-10-30): 48 slots, up to 4 attempts, QG experts, the sub18-levers server (sub18_subs)
    "1030-sub18-qwen38-flashnext-s48-5d76d78-int8": (SUB18_SETUP, lambda m, d, s, rm, bn, ctx: sub18_subs(m, d, s, ctx),
                                                     {"own_server": True, "obs_after": "6f2468cf", "run": SUB14_RUN}),
    # the same + the KV gate's two wait lists (docs/reports/1030_0326_sub18_build/build_sub18_gatefix_next.py)
    "1030-sub18-gatefix": (SUB18_SETUP, lambda m, d, s, rm, bn, ctx: sub18_subs(m, d, s, ctx),
                           {"own_server": True, "obs_after": "6f2468cf", "run": SUB14_RUN}),
    # the 10-30 optimization notebook (docs/reports/1030_1510_sub19_bundle/build_opt.py): the gate-fix notebook + its own
    # server flags (chunk 2048, PrefillDelayer with a pass cap, SMR_SPLIT) + IDLE_FILL; its bench runs 50 min
    "1030-opt-flashnext-s48-pf-idlefill": (
        SUB18_SETUP, lambda m, d, s, rm, bn, ctx: sub18_subs(m, d, s, ctx, rekey=dict(SUB18_REKEY, **{
            "BENCH_MINUTES = 86 ": "BENCH_MINUTES = 50 "})),
        {"own_server": True, "obs_after": "6f2468cf", "run": SUB14_RUN}),
    # the 12-hour run of S64b (LONG_SETUP above): the long235 set, 64 slots, 675 min of solving
    "1031-long-s64b-235": (LONG_SETUP, lambda m, d, s, rm, bn, ctx: sub18_subs(m, d, s, ctx, rekey=LONG_REKEY),
                           {"own_server": True, "obs_after": "6f2468cf", "run": SUB14_RUN,
                            "strict_bash": ("fnapply", "fnrs", "fnfork", "fnml3")}),
    # the 11-01 prompt probe on one Colab G4, submission 20's stack (sub20_g4_subs above; harness_nb --g4 --slots 16)
    "1101-probe-prompts-g4": (LONG_SETUP, lambda m, d, s, rm, bn, ctx: sub20_g4_subs(m, d, s, ctx),
                              {"own_server": True, "obs_after": "6f2468cf", "run": SUB14_RUN,
                               "strict_bash": ("fnapply", "fnrs", "fnfork", "fnml3")}),
}


def _scopes(t):
    yield t
    for c in t.get_children():
        yield from _scopes(c)


def undefined_names(sources: list[tuple[str, str]], top_only: bool = False) -> list[str]:
    """Names the cells read that no cell defines (nor the runner, nor builtins). A name read at a cell's top level must
    come from an earlier cell or the same cell; one read inside a function may come from any cell (call time).
    top_only (--boot-only): only top-level reads count, since no run cell calls the setup cells' functions."""
    defined, uses = {}, []
    for i, (cid, src) in enumerate(sources):
        st = symtable.symtable(src.replace("await ", ""), f"cell_{cid}", "exec")
        for s in st.get_symbols():
            if s.is_assigned() or s.is_imported() or s.is_namespace():
                defined.setdefault(s.get_name(), i)
            if s.is_referenced():
                uses.append((i, cid, s.get_name(), True))
        for sc in _scopes(st):
            if sc.get_type() != "module":
                uses += [(i, cid, s.get_name(), False) for s in sc.get_symbols()
                         if s.is_referenced() and s.is_global() and not s.is_free()]
    known = set(dir(builtins)) | RUNNER_NAMES
    return sorted({f"cell {cid}: {name!r}" for i, cid, name, top in uses if name not in known and (top or not top_only)
                   and (defined.get(name) is None or (top and defined[name] > i))})


def prepare(notebook: str, want: list[str], table: dict, boot_only: bool = False) -> list[tuple[str, str, str]]:
    """The cells to run, in notebook order, as (id, 'write' or 'code', text): a written file's name and contents
    (joined by a newline), or the code after substitution. Stops with every problem found."""
    cells = [c for c in json.load(open(notebook, encoding="utf-8"))["cells"] if c["cell_type"] == "code"]
    problems = [f"cell {x}: not in the notebook" for x in want if x not in {c.get("id") for c in cells}]
    plan = []
    for c in cells:
        cid = c.get("id")
        if cid not in want:
            continue
        src = "".join(c["source"])
        for old, new in table.get(cid, []):
            if old not in src:
                problems.append(f"cell {cid}: a substitution no longer matches: {old[:90]!r}")
            src = src.replace(old, new)
        if src.startswith("%%writefile"):
            plan.append((cid, "write", src.split(" ", 1)[1]))
            problems += [f"cell {cid}: a Kaggle path is left: {line.strip()[:100]}" for line in src.splitlines()
                         if "/kaggle/" in line and not line.lstrip().startswith("#")
                         and cid in table]   # only the files the profile substitutes (the server launch)
            continue
        if src.startswith("%%bash"):
            plan.append((cid, "bash", src.split("\n", 1)[1] if "\n" in src else ""))
            continue
        src = "\n".join(f"get_ipython().system({line.strip()[1:]!r})" if line.lstrip().startswith("!") else line
                        for line in src.splitlines())
        try:
            compile(src, f"cell_{cid}", "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
        except SyntaxError as e:
            problems.append(f"cell {cid}: does not compile ({e})")
        problems += [f"cell {cid}: a Kaggle path is left: {line.strip()[:100]}" for line in src.splitlines()
                     if "/kaggle/" in line]
        plan.append((cid, "code", src))
    if not problems:
        problems = [f"{u}: no run cell defines it"
                    for u in undefined_names([(i, s) for i, k, s in plan if k == "code"], top_only=boot_only)]
    if problems:
        raise SystemExit("preflight failed:\n  " + "\n  ".join(problems))
    return plan


def snapshot(src: Path, dst: Path, seen: dict) -> int:
    """Copy the files of `src` that changed since the last call into `dst` (the cache Volume), each as a whole new file
    renamed into place. Modal's volumefs2 rewrote appended files from stale copies in 1017_1823 (live logs repeating
    their earlier turns and losing later ones, the server log missing five minutes), so the round works on local disk
    and nothing on the Volume is ever appended to."""
    n = 0
    for p in src.rglob("*"):
        if not p.is_file():
            continue
        st = p.stat()
        if seen.get(str(p)) == (st.st_size, st.st_mtime_ns):
            continue
        target = dst / p.relative_to(src)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(target.name + ".snap")
        try:
            shutil.copyfile(p, tmp)
            os.replace(tmp, target)
        except OSError as e:   # 1022_1011: the Volume lost a .snap between copy and rename; the next call retries
            print(f"[harness-nb] mirror of {p.name} skipped this time: {e}", flush=True)
            continue
        seen[str(p)] = (st.st_size, st.st_mtime_ns)
        n += 1
    return n


OPEN: dict[int, tuple[float, str]] = {}   # requests in flight (trace_requests): seq -> (start, path); the observer reads it
_SEQ = iter(range(1, 1 << 62))


def trace_requests(path: Path) -> None:
    """Log every HTTP request the notebook's cells make through aiohttp (interact campaign): start and end time, path,
    status, the streaming session's id and the number of input ids sent. Pure observation, for each conversation's
    timeline (queued + prefill + decode per turn; the gaps between a session's requests are its tool calls and parks).
    A /generate response arrives when its turn ends, so end - start is the turn's server time. Requests still in
    flight are kept in OPEN, so the scheduler observer can count the attempts that are inside a server request."""
    import aiohttp

    orig = aiohttp.ClientSession._request
    fh = open(path, "a", encoding="utf-8", buffering=1)

    async def _request(self, method, str_or_url, **kwargs):
        t0 = time.time()
        body = kwargs.get("json") if isinstance(kwargs.get("json"), dict) else {}
        url = str(str_or_url)
        seq = next(_SEQ)
        OPEN[seq] = (t0, url.rsplit("/", 1)[-1])
        try:
            resp = await orig(self, method, str_or_url, **kwargs)
        finally:
            OPEN.pop(seq, None)
        if "/v1/loads" not in url and "/health" not in url:
            sp = body.get("session_params") or {}
            sid = sp.get("id") if isinstance(sp, dict) else None
            ids = body.get("input_ids")
            fh.write(json.dumps({"t0": round(t0, 3), "t1": round(time.time(), 3), "path": url.rsplit("/", 1)[-1],
                                 "status": resp.status, "sid": sid or body.get("session_id"),
                                 "n_in": len(ids) if isinstance(ids, list) else None,
                                 "max_new": (body.get("sampling_params") or {}).get("max_new_tokens")}) + "\n")
        return resp

    aiohttp.ClientSession._request = _request


def _copy(obj, kind=dict, tries: int = 5):
    """A copy of a dict or list that the solve's event loop may be changing on the main thread (retried on a resize)."""
    for _ in range(tries):
        try:
            return kind(obj) if obj is not None else kind()
        except RuntimeError:   # "dictionary changed size during iteration"
            time.sleep(0.01)
    return kind()


def _live_tokens(path: str) -> tuple[int, int, int]:
    """Turns, tokens and tool calls of a running attempt, from its live log (the solve cell's live_calls, read-only)."""
    try:
        with open(path, encoding="utf-8") as f:
            turns = [json.loads(line) for line in f if line.strip()]
    except (OSError, ValueError):
        return 0, 0, 0
    return len(turns), sum(t.get("tokens", 0) for t in turns), sum(len(t.get("calls", [])) for t in turns)


class SchedObserver:
    """Read-only snapshots of submission 14's scheduler state (the solve cell a6c36a51's globals) every `every` seconds
    into `path` (JSONL), for the long-run study (docs/reports/1026_1244_longrun_sub14). It never writes into the cells'
    namespace and never calls a cell function: the solve cell runs byte for byte as on Kaggle.

    Each row: the workers (running attempts by kind, idle workers), how many running attempts are inside a server
    request (trace_requests' OPEN; the rest are in tool calls, the harness's own CPU work or parked), the queue (unstarted
    first attempts, waiting siblings, outputs waiting behind their own task), second attempts (waiting by class, by
    state), the frugal switch and the per-stream speed the pacer set, the KV gate, finished attempts and answered outputs,
    the minutes left. Every `detail_every`-th row also lists each running attempt: name, kind, age, turns, tokens, calls
    (task ids: this file stays on the Volume and the laptop; the tracker publishes aggregates only)."""

    def __init__(self, ns_ref: dict, path: Path, every: float = 10.0, detail_every: int = 6, log=print):
        self.ns_ref, self.path, self.every, self.detail_every, self.log = ns_ref, Path(path), every, detail_every, log
        self.stop_ev, self.n, self.thread = threading.Event(), 0, None

    def start(self) -> "SchedObserver":
        self.thread = threading.Thread(target=self._run, name="sched-observer", daemon=True)
        self.thread.start()
        return self

    def stop(self) -> None:
        self.stop_ev.set()
        if self.thread:
            self.thread.join(timeout=10)

    @staticmethod
    def kind_of(name: str) -> str:
        # later attempts are named `_tail` (the second) and `_tail3`, `_tail4` (MAX_ATTEMPTS 4, sub 18 on); GPT
        # pre-run review 10-31 P0: the numbered ones were counted as sibling first attempts
        if name.rsplit("_", 1)[-1].startswith("tail"):
            return "tail"
        return "first" if name.endswith("_0_harness") else "sibling"

    def row(self, ns: dict, now: float, detail: bool) -> dict | None:
        t_solve = ns.get("t_solve")
        if t_solve is None:   # the solve cell has not started
            return None
        slots = ns.get("SLOTS") or 0
        running = _copy(ns.get("running"))
        kinds = {"first": 0, "sibling": 0, "tail": 0}
        for name in running:
            kinds[self.kind_of(name)] += 1
        queue = _copy(ns.get("queue"), list)
        shared = _copy(ns.get("shared"))
        active = _copy(ns.get("active"))
        tails = _copy(ns.get("tails"), list)
        tail_cls = _copy(ns.get("tail_cls"))
        tail_state = _copy(ns.get("tail_state"))
        done = _copy(ns.get("done"), list)
        answers = _copy(ns.get("answers"))
        outputs = _copy(ns.get("outputs"), list)
        gate = ns.get("GATE")
        fin = [r for r in done if r.get("kind") == "harness" and not r.get("unfinished")]
        by_cls: dict[str, int] = {}
        for entry in tails:   # sub 11-17 queue (tier, key, task); sub 18 (tier, attempts so far, key, task)
            t = entry[-1]
            by_cls[tail_cls.get(t, "?")] = by_cls.get(tail_cls.get(t, "?"), 0) + 1
        states: dict[str, int] = {}
        for v in tail_state.values():
            states[v] = states.get(v, 0) + 1
        opened = list(OPEN.values())
        stop_t, wrap_at = ns.get("STOP"), ns.get("WRAP_AT")
        row = {
            "t": round(now, 2), "t_min": round((now - t_solve) / 60, 2), "slots": slots,
            "running": len(running), "running_first": kinds["first"], "running_sibling": kinds["sibling"],
            "running_tail": kinds["tail"], "idle_workers": max(0, slots - len(running)),
            "open_generate": sum(1 for _, p in opened if p == "generate"), "open_requests": len(opened),
            "unstarted_firsts": sum(1 for o in queue if o[1] == 0),
            "waiting_siblings": sum(1 for o in queue if o[1] > 0 and o not in shared),
            "blocked_by_task": sum(1 for o in queue if active.get(o[0])), "shared": len(shared),
            "tails_waiting": len(tails), "tails_waiting_by_cls": by_cls, "tail_state": states,
            "first_running": (ns.get("first_running") or [None])[0],
            "frugal": bool((ns.get("FRUGAL") or [False])[0]), "stream_tps": (ns.get("STREAM_TPS") or [None])[0],
            "frugal_level": int((ns.get("FRUGAL") or [0])[0] or 0),   # graded frugal (10-27): 0 / 1 early / 2 all
            "gate": ({k: getattr(gate, k, None) for k in ("usage", "polled", "parked", "parks", "host_parks", "failed")}
                     if gate is not None else None),
            "done": len(done), "finished_harness": len(fin), "verified": sum(bool(r.get("verified")) for r in fin),
            "finished_tails": sum(1 for r in done if r.get("kind") == "tail" and not r.get("unfinished")),
            "unfinished": sum(1 for r in done if r.get("unfinished")), "errors": sum(1 for r in done if r.get("error")),
            "spent": sum(1 for r in done if r.get("spent")),
            "tokens_done": sum((r.get("tokens") or 0) + (r.get("gestalt_tokens") or 0) for r in done),
            "answered_outputs": sum(1 for o in outputs if (answers.get(o) or [None])[0] is not None),
            "outputs": len(outputs),
            "left_min": round((stop_t - now) / 60, 2) if stop_t else None,
            "wrap": (now >= wrap_at) if wrap_at else None,
        }
        if detail:
            live = ns.get("LIVE") or "attempts/live"
            att = []
            for name, t0 in running.items():
                turns, tok, calls = _live_tokens(f"{live}/{name}.jsonl")
                att.append([name, self.kind_of(name), round(now - t0), turns, tok, calls])
            row["attempts"] = att
            row["tokens_running"] = sum(a[4] for a in att)
        return row

    def _run(self) -> None:
        with self.path.open("a", encoding="utf-8", buffering=1) as fh:
            while not self.stop_ev.wait(self.every):
                ns = self.ns_ref.get("ns")
                if ns is None:
                    continue
                try:
                    r = self.row(ns, time.time(), detail=self.n % self.detail_every == 0)
                except Exception as e:  # noqa: BLE001 - observation must never stop the round
                    self.log(f"sched observer: {type(e).__name__}: {e}")
                    continue
                if r is not None:
                    fh.write(json.dumps(r) + "\n")
                    self.n += 1


def clear_stale_ext_locks() -> None:
    """A server killed while its G5 torch extension was still JIT-compiling leaves torch's file_baton lock behind, and
    the next boot waits on it (18 min, then an ImportError on the half-built .so: the park study, 2026-10-17)."""
    root = os.environ.get("TORCH_EXTENSIONS_DIR") or os.path.expanduser("~/.cache/torch_extensions")
    for lock in glob.glob(f"{root}/*/sglang_fp8_g5/lock") + glob.glob(f"{root}/sglang_fp8_g5/lock"):
        os.remove(lock)
        log(f"removed a stale extension lock: {lock}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="harness_nb")
    ap.add_argument("--notebook", required=True)
    ap.add_argument("--data", required=True, help="the directory of arc-agi_evaluation_{challenges,solutions}.json")
    ap.add_argument("--config", default=CONFIG, help="the GPU type's server config (a bench config with one arm)")
    ap.add_argument("--arm", default="harness")
    ap.add_argument("--slots", type=int, default=18,
                    help="requests in flight, for notebooks that take it (1018-harness-bench's SLOTS)")
    ap.add_argument("--run", default="", help="experiment cell ids, run after SETUP in notebook order")
    ap.add_argument("--boot-only", action="store_true", help="the server and SETUP only (a smoke)")
    ap.add_argument("--check", action="store_true", help="the preflight only: no server, no cell runs")
    ap.add_argument("--round-minutes", type=float, default=None)
    ap.add_argument("--batch-name", default=None)
    ap.add_argument("--sample", type=float, default=0.0, help="sampler period in seconds (0: off)")
    ap.add_argument("--nsys", action="store_true")
    ap.add_argument("--nsys-metrics-hz", type=int, default=10000)
    ap.add_argument("--profile-at", default="", help="minutes after the server is ready, comma-separated")
    ap.add_argument("--profile-steps", type=int, default=60)
    ap.add_argument("--profile-kind", choices=["nsys", "torch"], default="nsys")
    ap.add_argument("--hours", type=float, default=11 + 35 / 60, help="sub 7's HOURS (its solve stops then)")
    ap.add_argument("--rehearsal", default="off", help="sub 7's REHEARSAL: off (full strength), load or smoke")
    ap.add_argument("--bench-minutes", type=float, default=None,
                    help="sub 11's BENCH_MINUTES instead of the notebook's 342 (REHEARSAL bench)")
    ap.add_argument("--quiet-outputs", action="store_true",
                    help="sub 11: no per-output verdict lines in the finish (a set with eval children)")
    ap.add_argument("--bench-set", default=None,
                    help="sub 14: BENCH_SET, a set of ours in --data as <set>_{challenges,solutions,parents}.json, "
                         "solved whole (BENCH_TASKS emptied)")
    ap.add_argument("--observe", type=float, default=0.0,
                    help="sub 14: the scheduler observer's period in seconds (0: off) -> sched.jsonl")
    ap.add_argument("--decode-log-interval", type=int, default=1, help="sub 7's server: one Decode line per N steps")
    ap.add_argument("--trace-requests", action="store_true", help="log the cells' HTTP requests (requests.jsonl)")
    ap.add_argument("--smr-threads", type=int, default=8, help="sub 7's CPU reducer threads, on the last N CPUs")
    ap.add_argument("--kv-cache-dtype", default="fp8_e4m3", choices=["fp8_e4m3", "agi2_v4"],
                    help="sub 7's server KV pool format (agi2_v4: fp8 K + rotated int4 V, build .17 on)")
    ap.add_argument("--env", action="append", default=[], help="K=V set before the cells run (the server inherits it)")
    ap.add_argument("--server-arg", action="append", default=[],
                    help="an extra launch flag for a notebook that launches its own server (sub 7/9), appended to its "
                         "launch line; repeatable (e.g. --server-arg=--load-format=instanttensor)")
    ap.add_argument("--probe", default="",
                    help="a shell command run once the notebook's own server is up, before its experiment cells")
    ap.add_argument("--model-dir", default="",
                    help="serve this directory instead of the config alias's (e.g. a checkpoint converted on the box "
                         "into the cache Volume: /vol/models is read-only there)")
    ap.add_argument("--g4", action="store_true",
                    help="sub 14's profile on one Colab G4 (TP1, the 4 x L4-only switches dropped; sub14_g4_overrides)")
    ap.add_argument("--temperature", type=float, default=None,
                    help="sub 14's TEMPERATURE (settings cell), for a temperature arm; unset, the notebook's own 1.0")
    ap.add_argument("--graph-bs", default="",
                    help="sub 14's ARC_GRAPH_BS (switches cell), for a slots arm above 16, e.g. "
                         "'1 2 3 4 5 6 7 8 10 12 14 16 18 20 22 24'; unset, the notebook's own list (up to 16)")
    ap.add_argument("--mem-fraction", default="",
                    help="sub 14's ARC_MEM_FRACTION (switches cell, '0.90'), e.g. '0.88' for headroom at 24 slots")
    ap.add_argument("--tier-x", default="",
                    help="sub 14's SGLANG_AGI2_QSA_BLOCK_TIER_X (server cell, '2.5'): index slots per untiered token; "
                         "more pool tokens for fewer device rows (S = X T, D = T (kv + qsa - X qsa) / kv)")
    ap.add_argument("--first-pass-only", action="store_true",
                    help="sub 15: TAIL and FRUGAL_AUTO off (first attempts only, every acceptance reviewed)")
    ap.add_argument("--server-flag", action="append", default=[],
                    help="K=V added to the switches cell's FLAGS after the block tier's (41aad7bc; repeatable), e.g. "
                         "SGLANG_AGI2_QSA_BLOCK_TIER_POT=1 for a candidate server; the cell's own clear-and-set keeps it")
    ap.add_argument("--ornith", action="store_true",
                    help="with --g4: Ornith-1.5-122B's server in place of Flash-Next's (ornith_overrides; the 10-29 pilot)")
    ap.add_argument("--presence-penalty", type=float, default=None,
                    help="--ornith: the harness's presence_penalty (the vendor's precise coding 0.0, general 1.5)")
    ap.add_argument("--ornith-host-embed", action="store_true",
                    help="--ornith: keep SGLANG_AGI2_HOST_EMBED (the embedding table in pinned host RAM, exact)")
    ap.add_argument("--ornith-mem", default="",
                    help="--ornith: ARC_MEM_FRACTION after the G4 profile's 0.95, e.g. 0.96")
    ap.add_argument("--nanbeige", action="store_true",
                    help="with --g4: Nanbeige4.3-33B's server in place of Flash-Next's (nanbeige_overrides; the 10-29 pilot)")
    ap.add_argument("--nanbeige-kv", default="fp8_e4m3", help="--nanbeige: the KV cache dtype (fp8_e4m3 or auto)")
    ap.add_argument("--nanbeige-mem", default="", help="--nanbeige: ARC_MEM_FRACTION after the G4 profile's 0.95")
    ap.add_argument("--nanbeige-no-park", action="store_true",
                    help="--nanbeige: no host parking of sessions (the gate closes and rebuilds; the draft KV is rebuilt too)")
    ap.add_argument("--nanbeige-window-kv", action="store_true",
                    help="--nanbeige: keep the DFLASH draft's 2k window KV (off: the draft attends to the whole context)")
    ap.add_argument("--draft-dir", default="",
                    help="serve this draft directory instead of the config arm's alias (a copy on local disk)")
    ap.add_argument("--qg-dir", default="",
                    help="sub 18: the QG experts' folder on local disk (its overlay cell's QG=, 192 expert shards)")
    ap.add_argument("--max-attempts", type=int, default=0,
                    help="sub 18: MAX_ATTEMPTS (settings cell, 4 as submitted), e.g. 2 for the control arm")
    ap.add_argument("--park-host-gb", default="",
                    help="sub 18: SGLANG_STREAMING_SESSION_PARK_HOST_GB (8 as submitted), e.g. 2 on a host with less RAM")
    ap.add_argument("--chunked-prefill", type=int, default=0,
                    help="sub 18: the launch line's --chunked-prefill-size (1024 as submitted)")
    ap.add_argument("--launch-arg", action="append", default=[],
                    help="sub 18: server flags appended to the launch line after --chunked-prefill-size (repeatable)")
    ap.add_argument("--kaggle-runtime", action="store_true",
                    help="sub 17: TP4 on Kaggle's offline runtime in /tmp/sgl (its toolkit and Python lines kept)")
    ap.add_argument("--radixark-experts", action="store_true",
                    help="sub 18: RadixArk's own experts instead of the QG overlay (no --qg-dir)")
    ap.add_argument("--bf16-index-keys", action="store_true",
                    help="sub 18: SGLANG_AGI2_QSA_INDEX_INT8 off, so the QSA index keys are BF16 as in subs 12-17")
    ap.add_argument("--nccl-allreduce", action="store_true",
                    help="subs 17-18: sm_reduce off, so TP4's all-reduce is NCCL's exact one instead of the int8 one")
    a = ap.parse_args(argv)
    for kv in a.env:
        k, _, v = kv.partition("=")
        os.environ[k] = v

    from agi2_modal_harness.core import config, models
    from agi2_modal_harness.workloads.sglang_bench.bench_config import BenchConfig

    notebook, data = os.path.abspath(a.notebook), os.path.abspath(a.data)
    stem = Path(notebook).stem
    if stem not in PROFILES:
        raise SystemExit(f"no profile for notebook {stem!r}: known {sorted(PROFILES)}")
    setup, table, opts = PROFILES[stem]
    own = bool(opts.get("own_server"))
    cfg = BenchConfig.load(Path(a.config), arms=[a.arm])
    arm = cfg.arms[0]
    model_dir = a.model_dir or str(models.model_path(config.MOUNT_MODELS, cfg.model_alias))
    draft = a.draft_dir or (str(models.model_path(config.MOUNT_MODELS, arm.draft_alias)) if arm.draft_alias else None)
    out = Path(os.environ.get("OUT", "/tmp/harness_nb"))
    # Local disk: appends on the cache Volume are not safe (snapshot()). $OUT/work gets whole-file copies every 2 min.
    work = Path(os.environ.get("HARNESS_WORK") or f"/opt/dev/work/{os.environ.get('RUN_ID', 'scratch')}")
    gpu = (cfg.compute.gpu or "L4").split(":")[0].rstrip("!")
    # Compile caches per GPU type: Triton, inductor and FlashInfer builds are architecture-specific.
    # HARNESS_CACHE_TAG gives a box its own folder: two boxes on different builds must never JIT into one (their
    # torch-extension builds and locks collide, and clear_stale_ext_locks would remove the other box's locks).
    tag = os.environ.get("HARNESS_CACHE_TAG", "")
    cache = Path(config.MOUNT_CACHE) / "sglang-0.5.21" / gpu / f"harness-{arm.name}{'-' + tag if tag else ''}"
    at = [float(x) for x in a.profile_at.split(",") if x.strip()]
    ctx = {"hours": a.hours, "rehearsal": a.rehearsal, "cache": str(cache), "draft_dir": draft, "work": str(work),
           "nsys": a.nsys, "captures": len(at), "nsys_hz": a.nsys_metrics_hz, "decode_log_interval": a.decode_log_interval,
           "smr_threads": a.smr_threads, "kv_cache_dtype": a.kv_cache_dtype, "server_args": list(a.server_arg),
           "bench_minutes": a.bench_minutes, "quiet_outputs": a.quiet_outputs, "bench_set": a.bench_set,
           "g4": a.g4, "temperature": a.temperature, "model_dir": model_dir, "graph_bs": a.graph_bs,
           "mem_fraction": a.mem_fraction, "tier_x": a.tier_x, "first_pass_only": a.first_pass_only,
           "server_flags": list(a.server_flag), "ornith": a.ornith, "presence_penalty": a.presence_penalty,
           "ornith_host_embed": a.ornith_host_embed, "ornith_mem": a.ornith_mem,
           "nanbeige": a.nanbeige, "nanbeige_kv": a.nanbeige_kv, "nanbeige_mem": a.nanbeige_mem,
           "nanbeige_window_kv": a.nanbeige_window_kv, "nanbeige_no_park": a.nanbeige_no_park,
           "qg_dir": a.qg_dir, "max_attempts": a.max_attempts, "chunked_prefill": a.chunked_prefill,
           "park_host_gb": a.park_host_gb,
           "launch_args": list(a.launch_arg), "kaggle_runtime": a.kaggle_runtime,
           "radixark_experts": a.radixark_experts, "bf16_index_keys": a.bf16_index_keys,
           "nccl_allreduce": a.nccl_allreduce}
    run_cells = [x for x in a.run.split(",") if x] or list(opts.get("run", []))
    want = setup + ([] if a.boot_only else run_cells)
    plan = prepare(notebook, want, table(model_dir, data, a.slots, a.round_minutes, a.batch_name, ctx),
                   boot_only=a.boot_only)
    log(f"preflight passed: {len(plan)} cells ({sum(k == 'write' for _, k, _ in plan)} write files, "
        f"{sum(k == 'bash' for _, k, _ in plan)} bash){'; the notebook launches its own server' if own else ''}")
    if a.check:
        return 0

    from agi2_modal_harness.core.runtime.sglang_server import SGLangServer

    work.mkdir(parents=True, exist_ok=True)
    log(f"work directory {work} (mirrored to {out / 'work'} every 2 min)")
    seen: dict = {}
    stop_mirror = threading.Event()

    def mirror() -> None:
        while not stop_mirror.wait(120):
            try:
                snapshot(work, out / "work", seen)
            except OSError as e:   # a mirror failure must not stop the round; the next pass retries
                log(f"mirror pass failed: {e}")

    threading.Thread(target=mirror, name="mirror", daemon=True).start()
    server_cfg = cfg.arm_server(arm)
    cmd = cfg.command(arm, model_dir, draft)
    if a.nsys:
        (work / "nsys").mkdir(parents=True, exist_ok=True)
        cmd = NSYS + [f"--capture-range-end=repeat:{max(1, len(at))}", f"--gpu-metrics-frequency={a.nsys_metrics_hz}",
                      "-o", str(work / "nsys" / "report")] + cmd
    env = {"SGLANG_CACHE_DIR": str(cache / "sglang"), "TRITON_CACHE_DIR": str(cache / "triton"),
           "TORCHINDUCTOR_CACHE_DIR": str(cache / "inductor"), "FLASHINFER_WORKSPACE_BASE": str(cache / "fi"),
           "XDG_CACHE_HOME": str(cache / "xdg"), **server_cfg.env, **arm.env}
    for key in ("SGLANG_CACHE_DIR", "TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR", "FLASHINFER_WORKSPACE_BASE",
                "XDG_CACHE_HOME"):
        Path(env[key]).mkdir(parents=True, exist_ok=True)
    os.chdir(work)
    sys.path.insert(0, str(work))   # as a kernel's cwd: the cells import the modules they write (arc_skills)
    if a.trace_requests:
        trace_requests(work / "requests.jsonl")
        log(f"tracing the cells' HTTP requests -> {work / 'requests.jsonl'}")
    servers: list = []
    ready: list[bool] = []

    def relaunch() -> bool:
        """restart_server()'s `bash launch_server.sh`: keep the old log, start the same command again."""
        if ready and not ready[-1]:   # the last server died before it was ready: maybe mid-compile
            clear_stale_ext_locks()
        logp = work / "sglang.log"
        if logp.exists():
            logp.rename(work / f"sglang_{time.strftime('%m%d_%H%M%S')}.log")
        s = SGLangServer(cmd, port=server_cfg.port, log_path=logp, echo=log, env=env)
        s.start()
        servers.append(s)
        ready.append(False)
        return True

    obs: list = []   # interact_obs sampler and profile trigger, stopped with the round
    ns_ref: dict = {}   # the cells' namespace, once it exists (the scheduler observer reads it)

    def start_obs(base: str) -> None:
        if a.observe:
            obs.append(SchedObserver(ns_ref, work / "sched.jsonl", a.observe, log=log).start())
            log(f"scheduler observer every {a.observe:g} s -> {work / 'sched.jsonl'}"
                + ("" if a.trace_requests else " (no --trace-requests: open_generate stays 0)"))
        if not (a.sample or at):
            return
        from agi2_modal_harness.comm import interact_obs

        if a.sample:
            obs.append(interact_obs.Sampler(work / "samples.jsonl", base, a.sample).start())
            log(f"sampler every {a.sample:g} s -> {work / 'samples.jsonl'}")
        if at:
            prof_dir = work / ("nsys" if a.profile_kind == "nsys" else "torch")
            prof_dir.mkdir(parents=True, exist_ok=True)
            obs.append(interact_obs.ProfileTrigger(base, at, a.profile_steps, a.profile_kind, str(prof_dir),
                                                   log).start())
            log(f"{a.profile_kind} captures of {a.profile_steps} steps at {at} min from now")

    if a.nsys:
        (work / "nsys").mkdir(parents=True, exist_ok=True)
    if not own:
        relaunch()
    try:
        if not own:
            boot = servers[-1].wait_ready(cfg.compute.boot_timeout_min * 60)
            ready[-1] = True
            log(f"server ready in {boot:.0f} s; pool {servers[-1].server_info().get('max_total_num_tokens')} tokens")
            start_obs(servers[-1].base_url)
        ns = {"get_ipython": IPython, "__name__": "__main__", "RELAUNCH": relaunch}
        ns_ref["ns"] = ns
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        for cid, kind, text in plan:
            if kind == "write":
                name, contents = text.split("\n", 1)
                Path(name.strip()).write_text(contents, encoding="utf-8")
                log(f"cell {cid}: wrote {name.strip()} ({len(contents):,} characters)")
                continue
            log(f"cell {cid}")
            if kind == "bash":
                rc = subprocess.run(["bash", "-c", text], check=False).returncode
                # a profile's build cells must succeed: no run on a partial build (GPT pre-run review 10-31, P1)
                if rc and cid in opts.get("strict_bash", ()):
                    raise SystemExit(f"cell {cid} exited {rc}: a build cell failed, so nothing runs on a partial build")
            else:
                result = eval(compile(text, f"cell_{cid}", "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT), ns)
                if asyncio.iscoroutine(result):
                    loop.run_until_complete(result)
            if cid == "7ec1601a" and not ns.get("PARK_PICTURES_OK"):
                raise SystemExit("park_picture_check failed on this server: no experiment runs")
            if own and cid == opts.get("obs_after"):
                if not ns.get("SERVER_UP"):
                    raise SystemExit("the notebook's server did not come up: see work/sglang.log")
                start_obs(ns.get("SERVER", "http://127.0.0.1:30000"))
                if a.probe:
                    # Once the server is up and idle, before any experiment cell: probes that need an idle server
                    # (e.g. comm/picks_probe.py). Its result is recorded; a failed probe never stops the round.
                    log(f"probe: {a.probe}")
                    rc = subprocess.run(["bash", "-c", a.probe], check=False).returncode
                    (out / "probe_rc.txt").write_text(f"{rc}\n", encoding="utf-8")
                    log(f"probe exit code {rc}")
        (out / "result.json").write_text(json.dumps({"status": "done", "cells": want}), encoding="utf-8")
        log("done")
        return 0
    finally:
        for o in obs:
            o.stop()
        alive = (servers and servers[-1].alive()) or (own and subprocess.run(
            ["pgrep", "-f", "sglang.launch_server"], capture_output=True).returncode == 0)
        if a.nsys and alive:
            # nsys writes each capture range's report when the range ends; the last one may still be importing
            t_wait = time.time()
            while time.time() - t_wait < 300 and any(p.suffix == ".qdstrm" for p in (work / "nsys").glob("*")):
                time.sleep(5)
            log(f"nsys reports: {sorted(p.name for p in (work / 'nsys').glob('*.nsys-rep'))}")
        for s in servers:
            if s.alive():
                s.stop()
        if own and alive:   # the notebook's own server: SIGTERM first, so nsys can close its session
            subprocess.run(["pkill", "-TERM", "-f", "sglang.launch_server"], check=False)
            t_wait = time.time()
            while time.time() - t_wait < 90 and subprocess.run(["pgrep", "-f", "sglang.launch_server"],
                                                               capture_output=True).returncode == 0:
                time.sleep(3)
            subprocess.run(["pkill", "-9", "-f", "sglang::"], check=False)
            subprocess.run(["pkill", "-9", "-f", "sglang.launch_server"], check=False)
        stop_mirror.set()
        n = snapshot(work, out / 'work', seen)
        n += snapshot(work, out / 'work', seen)   # a second pass copies only what the first skipped
        log(f"mirrored {n} changed files to {out / 'work'}")


if __name__ == "__main__":
    raise SystemExit(main())
