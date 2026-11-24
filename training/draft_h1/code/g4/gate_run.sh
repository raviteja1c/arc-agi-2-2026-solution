#!/bin/bash
# The fine-tune gate (1025_1926_colab-g4-dflash2-ft): the probe at blocks 4 and 8 over the exp1 / t12 recorded
# histories (eval parents + children), with a given draft directory. Same server as probe_run.sh (submission 13's
# flags at TP1, Marlin W4A16, K384, fp8 KV, BF16 head), the probe at fork 110bd2042f (capture off).
#   DRAFT=<draft dir> PROBE_DIR=<out root> bash /content/inputs/gate_run.sh <run_id> [client args...]
# The stock draft runs with the client's calibration preflight (its block-4 prediction must reproduce the served
# 4 x L4 steps); a fine-tuned draft passes --no-gate (its prediction is meant to differ).
set -u
RUN_ID=$1; shift
RUN=/content/run/$RUN_ID
IN=/content/inputs
PD=${PROBE_DIR:-/content/gate_probe}
PIN=${PROBE_INPUTS:-$IN}  # histories.npz, index.json and requests/ to probe (default: the gate's)
DRAFT=${DRAFT:-/tmp/models/dflash2-qwen3.8-flash-next}
mkdir -p $RUN $PD/out $PD/requests && cp $PIN/requests/*.json $PD/requests/
T0=$(date +%s)
log() { echo "[gate $(date -u +%H:%M:%S) UTC +$(( ($(date +%s) - T0) / 60 )) min] $*" | tee -a $RUN/run.log; }
export CUDA_HOME=/tmp/sgl/lib/python3.12/site-packages/nvidia/cu13
export PATH=/tmp/sgl/bin:$CUDA_HOME/bin:$PATH NVCC_APPEND_FLAGS="-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK"
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}:$CUDA_HOME/lib
ln -sfn lib $CUDA_HOME/lib64 && ln -sf libcudart.so.13 $CUDA_HOME/lib/libcudart.so
mkdir -p /tmp/cache/{triton,inductor,flashinfer,xdg,torch_extensions,sglang_jit,sglang,cute_dsl}
export TRITON_CACHE_DIR=/tmp/cache/triton TORCHINDUCTOR_CACHE_DIR=/tmp/cache/inductor SGLANG_JIT_CACHE_DIR=/tmp/cache/sglang_jit
export FLASHINFER_WORKSPACE_BASE=/tmp/cache/flashinfer XDG_CACHE_HOME=/tmp/cache/xdg TORCH_EXTENSIONS_DIR=/tmp/cache/torch_extensions
export SGLANG_CACHE_DIR=/tmp/cache/sglang CUTE_DSL_CACHE_DIR=/tmp/cache/cute_dsl
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SGLANG_AGI2_FN_EXPERT_KEEP=$IN/keep_384_code_bal4.json SGLANG_AGI2_FN_EXPERT_KEEP_ANY_ORDER=1
export SGLANG_ENABLE_GDN_REPLAYSSM_FOLD=1 SGLANG_DFLASH_DRAFT_WINDOW_KV=1 SGLANG_AGI2_DRAFT_WINDOW_KV_PAGED=1
export SGLANG_AGI2_DFLASH_PROBE_DIR=$PD SGLANG_AGI2_DFLASH_PROBE_BLOCKS=${BLOCKS:-4,8}
export SGLANG_AGI2_DFLASH_PROBE_ROWS=${ROWS:-4096} SGLANG_AGI2_DFLASH_PROBE_DUMP_BLOCK=0
ARGS=(--model-path /tmp/models/flashnext-textonly --speculative-draft-model-path $DRAFT
      --served-model-name qwen3.8-flash-next --host 127.0.0.1 --port 30000 --tp 1 --context-length 262144
      --mem-fraction-static ${MEM:-0.80} --max-running-requests 2 --chunked-prefill-size ${CHUNK:-2048}
      --mamba-ssm-dtype float32 --kv-cache-dtype fp8_e4m3 --trust-remote-code --disable-radix-cache
      --quantization modelopt_fp4 --fp4-gemm-backend marlin --moe-runner-backend marlin --page-size 64 --ple-offload-embedding
      --max-mamba-cache-size 4 --enable-linear-replayssm-spec --disable-prefill-cuda-graph --linear-attn-verify-backend triton
      --speculative-algorithm DFLASH --speculative-num-draft-tokens 4 --speculative-draft-model-quantization unquant
      --disable-overlap-schedule --cuda-graph-max-bs-decode 2 --watchdog-timeout 3600)
env | grep -E '^(SGLANG_|PYTORCH_CUDA_ALLOC)' | sort > $RUN/server_env.txt
printf '%s\n' "${ARGS[@]}" > $RUN/server_args.txt
sha256sum $DRAFT/model.safetensors > $RUN/draft_sha256.txt
log "server: draft $DRAFT ($(cut -c1-16 $RUN/draft_sha256.txt)), blocks $SGLANG_AGI2_DFLASH_PROBE_BLOCKS"
cd $RUN
nohup setsid /tmp/sgl/bin/python -m sglang.launch_server "${ARGS[@]}" > $RUN/sglang.log 2>&1 < /dev/null &
SPID=$!
UP=0
for i in $(seq 480); do
  curl -sf -m 5 http://127.0.0.1:30000/health > /dev/null && { UP=1; break; }
  kill -0 $SPID 2> /dev/null || break
  [ $((i % 6)) -eq 0 ] && log "boot +$(( $(date +%s) - T0 )) s: $(grep -c . $RUN/sglang.log) log lines"
  sleep 5
done
if [ $UP = 0 ]; then
  log "server did not come up: $(grep -E 'Error|error' $RUN/sglang.log | tail -3 | cut -c1-300)"
  echo "boot-failed" > $RUN/run_exit.txt
  pkill -f '^/tmp/sgl/bin/python -m sglang.launch_server'; exit 3
fi
log "server up in $(( $(date +%s) - T0 )) s"
/tmp/sgl/bin/python $IN/probe_client.py --run $RUN --probe $PD --inputs $PIN "$@" >> $RUN/client.log 2>&1
RC=$?
log "client exit $RC: $(tail -1 $RUN/client.log | cut -c1-200)"
pkill -f '^/tmp/sgl/bin/python -m sglang.launch_server'; sleep 5; pkill -9 -f '^/tmp/sgl/bin/python -m sglang.launch_server'
echo "client $RC" > $RUN/run_exit.txt
