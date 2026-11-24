#!/bin/bash
# A harness DATA round on one G4 (decode-3x campaign, DFlash2 line): Flash-Next's own harness conversations on
# ARC-AGI-2 training tasks (the main session's `fntrain` set and notebook cell `hbfntrain`; no eval parent, no child)
# for the draft fine-tune, with every attempt's exact token history recorded (logprob recording cell `fnlprec1`,
# fork fn-conf, as the 4 x L4 logprob rounds 1024_2211 / 1025_0005 and comm/kv_round.sh's LOGPROBS=1 do).
# The probe/capture runtime's install (submission 13's build at TP1, Marlin numerics, probe_setup.sh) is reused; the
# server is the G4 config rtxpro6000_flashnext_radixark_marlin_k384_sub13_harness.toml on the full VL model.
#   setsid nohup bash /content/inputs/fntrain_round.sh <run_id> <slots> <minutes> [K=V ...] > /content/fntrain.log 2>&1 < /dev/null &
# The K=V list is the round recipe (e.g. FN_SCHED=1 FN_TIERS=1 FN_NO_EXTRAS=1 NIGHT_ARMS=g1 NIGHT_SAMPLES=0
# NIGHT_BATCH=<name> NIGHT_TASKS=<subset>); SGLANG_AGI2_FN_CONF=1 and LOGPROBS=1 are added here (fail closed).
set -u
RUN_ID=$1; SLOTS=$2; MINUTES=$3; shift 3
CONFIG=agi2_modal_harness/workloads/sglang_bench/configs/rtxpro6000_flashnext_radixark_marlin_k384_sub13_harness.toml
TARGET_DIR=/tmp/models/qwen3.8-flash-next-nvfp4-radixark
RUN=/content/run/$RUN_ID
mkdir -p $RUN/obs /vol/models /vol/cache
log() { echo "[fntrain $(date -u +%H:%M:%S) UTC] $*" | tee -a $RUN/round.log; }
if pgrep -f '[s]glang.launch_server' > /dev/null || pgrep -f '[a]gi2_modal_harness.comm.harness_nb' > /dev/null; then
  log "a server or a round is still running"; echo "busy 99" > $RUN/round_exit.txt; exit 1
fi
rm -rf /content/agi2 && mkdir -p /content/agi2 && tar -xzf /content/inputs/agi2_repo_fntrain.tgz -C /content/agi2
ls $TARGET_DIR/*.safetensors > /dev/null 2>&1 || { log "no model weights at $TARGET_DIR"; echo "$RUN_ID no-model" > $RUN/round_exit.txt; exit 1; }
[ -f /content/agi2/$CONFIG ] || { log "no config $CONFIG"; echo "$RUN_ID no-config" > $RUN/round_exit.txt; exit 1; }
[ -f /content/agi2/data/harness_bench/fntrain_challenges.json ] || { log "no fntrain set"; echo "$RUN_ID no-set" > $RUN/round_exit.txt; exit 1; }
ln -sfn $TARGET_DIR /vol/models/RadixArk__Qwen3.8-Flash-Next-NVFP4__7b719225242a
ln -sfn /tmp/models/dflash2-qwen3.8-flash-next /vol/models/incoai__Qwen3.8-Flash-Next-DFlash2__a22a1744e85c
export CUDA_HOME=/tmp/sgl/lib/python3.12/site-packages/nvidia/cu13
export PATH=/tmp/sgl/bin:$CUDA_HOME/bin:$PATH NVCC_APPEND_FLAGS="-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK"
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}:$CUDA_HOME/lib
ln -sfn lib $CUDA_HOME/lib64 && ln -sf libcudart.so.13 $CUDA_HOME/lib/libcudart.so
mkdir -p /tmp/cache/{triton,inductor,flashinfer,xdg,torch_extensions,sglang_jit,sglang,cute_dsl}
export TRITON_CACHE_DIR=/tmp/cache/triton TORCHINDUCTOR_CACHE_DIR=/tmp/cache/inductor SGLANG_JIT_CACHE_DIR=/tmp/cache/sglang_jit
export FLASHINFER_WORKSPACE_BASE=/tmp/cache/flashinfer XDG_CACHE_HOME=/tmp/cache/xdg TORCH_EXTENSIONS_DIR=/tmp/cache/torch_extensions
export SGLANG_CACHE_DIR=/tmp/cache/sglang CUTE_DSL_CACHE_DIR=/tmp/cache/cute_dsl
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
# fn-conf must be in the installed build (submission 13's e5cb6d9afb merges agi2/feat/fn-conf)
SITE=$(/tmp/sgl/bin/python -c "import sglang, os; print(os.path.dirname(sglang.__file__))")
grep -q AGI2_FN_CONF $SITE/srt/speculative/dflash_utils.py || { log "fn-conf is not in the installed SGLang"; echo "$RUN_ID no-fnconf" > $RUN/round_exit.txt; exit 3; }
cd /content/agi2
NOTEBOOK=notebooks/1018-harness-bench/1018-harness-bench.ipynb
/tmp/sgl/bin/python -m agi2_modal_harness.logprobs.fn_record insert "$NOTEBOOK" /content/nb_lp/1018-harness-bench.ipynb || { log "recording cell insert failed"; exit 3; }
NOTEBOOK=/content/nb_lp/1018-harness-bench.ipynb
CELLS=$(/tmp/sgl/bin/python -m agi2_modal_harness.logprobs.fn_record cells "hbfntrain,e8c03140,73e25b42,3d8ea3a5,1b17dcde,e03c3ecd") || { log "cells failed"; exit 3; }
ENVS="--env SGLANG_ENABLE_DFLASH_LENGTH_CAP=1 --env SGLANG_ENABLE_DFLASH_STOP_CAP=1 --env SGLANG_ENABLE_STREAMING_SESSION_FOLDED_CLAIM=1"
ENVS="$ENVS --env SGLANG_AGI2_FN_CONF=1 --env LOGPROBS=1"
for kv in "$@"; do ENVS="$ENVS --env $kv"; done
pkill -f "[c]omm.exp_obs" || true
(cd /content/lib && setsid nohup python3 -m agi2_modal_harness.comm.exp_obs --out $RUN --hours 6 > $RUN/obs/exp_obs.log 2>&1 < /dev/null &)
log "$RUN_ID: cells $CELLS, slots $SLOTS, $MINUTES min, config $CONFIG, model $TARGET_DIR, env $ENVS"
export OUT=$RUN HARNESS_WORK=/tmp/hw/$RUN_ID HARNESS_CACHE_TAG=flashnext
/tmp/sgl/bin/python -m agi2_modal_harness.comm.harness_nb --notebook $NOTEBOOK --data data/harness_bench --config $CONFIG \
  --slots $SLOTS --run $CELLS --round-minutes $MINUTES --sample 2 --trace-requests $ENVS > $RUN/harness_nb.log 2>&1 < /dev/null
RC=$?
echo "$RUN_ID $RC" > $RUN/round_exit.txt
log "$RUN_ID exit $RC; $(ls $RUN/work/bench/*/lp/*.npz 2> /dev/null | wc -l) recorded attempts"
