#!/usr/bin/env bash
# Box side: the code and tool-call half of the expert census for Flash-Next pruning (round 1023_0833: a prose-only
# census pruned the experts that carry Python). Unpruned, text-only, 4 requests, no draft, context 32768, the expert
# recorder on (boot 8's settings). P10 runs code-only: ARC-AGI-2 TRAINING puzzles asked for a `python` tool call with
# thinking off. Merge with boot 8's prose counts on the laptop (comm/fn_l4_merge_census.py), then rebuild keep lists.
#   comm.dev run --exp <slug> --detach -- bash agi2_modal_harness/comm/fn_l4_census_code.sh [n_puzzles] [wait_log]
set -u
N=${1:-48}
WAIT=${2:-}
if [ -n "$WAIT" ]; then   # chain: start once the previous box job has printed its last line
    echo "[census-code $(date -u +%H:%M:%S) UTC] waiting for 'done' in $WAIT"
    until grep -q 'code-ab .* done' "$WAIT" 2> /dev/null; do sleep 15; done
    pkill -f '[s]glang.launch_server'; sleep 10
fi
export SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR=/tmp/fn_census_code
mkdir -p $SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR
CFG=agi2_modal_harness/workloads/sglang_bench/configs/l4x4_flashnext_nvfp4_tp4_probe.toml
SKIP=P1_chat,P14_code,P3_session,P12_session_lp,P13_pressure,P2_speed,P4_long,P5_fidelity,P6_fixed,P7_image,P9_curve,P11_profile
echo "[census-code $(date -u +%H:%M:%S) UTC] start: $N tool-call puzzles"
bash agi2_modal_harness/comm/fn_l4_up.sh census_code --config $CFG --arm target --skip $SKIP \
  --census-tasks 0 --census-code "$N" --census-max-new 2048 \
  --env SGLANG_AGI2_EXPERT_COUNT_SUM=1 --env SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR=$SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR \
  --env SGLANG_AGI2_HOST_EMBED=1 --env SGLANG_AGI2_MARLIN_NO_SWIZZLED=1 --env SGLANG_AGI2_MARLIN_INPLACE_REPACK=1 \
  --env SGLANG_AGI2_QSA_TRITON_DECODE=1 \
  --server-arg=--ep-size --server-arg=4 --server-arg=--context-length --server-arg=32768 \
  --server-arg=--mem-fraction-static --server-arg=0.95 \
  --server-arg=--max-running-requests --server-arg=4 --server-arg=--max-mamba-cache-size --server-arg=8 \
  --server-arg=--cuda-graph-max-bs-decode --server-arg=4 --server-arg=--chunked-prefill-size --server-arg=1024 \
  --server-arg=--expert-distribution-recorder-mode --server-arg=stat
echo "[census-code $(date -u +%H:%M:%S) UTC] exit $? | $(ls $OUT/census_code/p10_expert_counts.json 2>&1)"
pkill -f '[s]glang.launch_server'
echo "[census-code $(date -u +%H:%M:%S) UTC] done"
