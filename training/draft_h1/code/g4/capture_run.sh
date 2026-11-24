#!/bin/bash
# Boot the probe server in capture mode (submission 13's build + the probe at fork 110bd2042f, TP1 on one G4, Marlin
# W4A16 = the 4 x L4's numerics, K384, fp32 GDN state, fp8 KV, BF16 head) and run capture_client.py.
# BLOCKS (env, default empty) = probe block sizes to run as well (parity reference: BLOCKS=4,8 with DUMP_BLOCK=8);
# empty = capture only (features + teacher, no per-anchor draft runs). NAMES (env, default all) = histories to send.
#   bash /content/inputs/capture_run.sh <run_id>
set -u
RUN_ID=$1
RUN=/content/run/$RUN_ID
IN=/content/inputs
PD=${PROBE_DIR:-/content/probe}
mkdir -p $RUN $PD/out $PD/requests
T0=$(date +%s)
log() { echo "[cap-run $(date -u +%H:%M:%S) UTC +$(( ($(date +%s) - T0) / 60 )) min] $*" | tee -a $RUN/run.log; }
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
export SGLANG_AGI2_DFLASH_PROBE_DIR=$PD SGLANG_AGI2_DFLASH_PROBE_BLOCKS="${BLOCKS:-}" SGLANG_AGI2_DFLASH_CAPTURE=1
export SGLANG_AGI2_DFLASH_PROBE_ROWS=${ROWS:-4096}
export SGLANG_AGI2_DFLASH_PROBE_DUMP_BLOCK=${DUMP_BLOCK:-0} SGLANG_AGI2_DFLASH_PROBE_DUMP_FRAC=${DUMP_FRAC:-0.05}
export SGLANG_LOGPROB_CHUNK_SIZE=${LOGPROB_CHUNK:-512}
ARGS=(--model-path /tmp/models/flashnext-textonly --speculative-draft-model-path /tmp/models/dflash2-qwen3.8-flash-next
      --served-model-name qwen3.8-flash-next --host 127.0.0.1 --port 30000 --tp 1 --context-length 262144
      --mem-fraction-static ${MEM:-0.80} --max-running-requests 2 --chunked-prefill-size ${CHUNK:-2048}
      --mamba-ssm-dtype float32 --kv-cache-dtype fp8_e4m3 --trust-remote-code --disable-radix-cache
      --quantization modelopt_fp4 --fp4-gemm-backend marlin --moe-runner-backend marlin --page-size 64 --ple-offload-embedding
      --max-mamba-cache-size 4 --enable-linear-replayssm-spec --disable-prefill-cuda-graph --linear-attn-verify-backend triton
      --speculative-algorithm DFLASH --speculative-num-draft-tokens 4 --speculative-draft-model-quantization unquant
      --disable-overlap-schedule --cuda-graph-max-bs-decode 2 --watchdog-timeout 3600)
env | grep -E '^(SGLANG_|PYTORCH_CUDA_ALLOC)' | sort > $RUN/server_env.txt
printf '%s\n' "${ARGS[@]}" > $RUN/server_args.txt
log "server: capture, blocks '${BLOCKS:-}', dump block ${DUMP_BLOCK:-0}, names ${NAMES:-all}"
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
/tmp/sgl/bin/python $IN/capture_client.py --hist ${HIST:-$IN/gen_histories.npz} --probe $PD --names "${NAMES:-all}" >> $RUN/client.log 2>&1
RC=$?
log "client exit $RC: $(tail -1 $RUN/client.log)"
pkill -f '^/tmp/sgl/bin/python -m sglang.launch_server'; sleep 10
echo "client $RC" > $RUN/run_exit.txt
