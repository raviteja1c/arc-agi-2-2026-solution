#!/usr/bin/env bash
# On the serving box after the capture: QG (GPTQ-NVFP4) on all four GPUs at once, 12 layers each, reading the BF16
# experts from the cache Volume and the capture from local disk. Fails closed unless all 96 BF16 shards are complete
# on the Volume and the calib capture has all 48 layers.
#   CAP=/tmp/fncap OUT=/tmp/fnxq/qg1 bash agi2_modal_harness/comm/nvfp4_qg_4gpu.sh
set -u
CAP=${CAP:-/tmp/fncap}
OUT=${OUT:-/tmp/fnxq/qg1}
BF16=${BF16:-/vol/cache/dev/fnbf16_parts}
KEEP=${KEEP:-/vol/cache/dev/fn_keep/keep_384_code_bal4.json}
t0=$(date +%s)
say() { echo "[qg4 $(date +%H:%M:%S), +$(( $(date +%s) - t0 )) s] $*"; }
python -m agi2_modal_harness.comm.nvfp4_vol reload || true  # the caller (qg_prep) has reloaded already
ready=$(python -m agi2_modal_harness.comm.nvfp4_ready_layers "$BF16" | tr ',' '\n' | grep -c .)
[ "$ready" -eq 48 ] || { say "FAILED: BF16 complete for $ready/48 layers on the Volume"; exit 2; }
caps=$(ls -d $CAP/calib/L* 2>/dev/null | wc -l)
[ "$caps" -eq 48 ] || { say "FAILED: calib capture has $caps/48 layers"; exit 2; }
mkdir -p "$OUT"
for g in 0 1 2 3; do
    lo=$((g * 12)); hi=$((lo + 11))
    CUDA_VISIBLE_DEVICES=$g python -m agi2_modal_harness.comm.nvfp4_gptq --cap "$CAP" --keep "$KEEP" --out "$OUT" \
        --bf16 "$BF16" --layers "$lo-$hi" ${QG_ARGS:-} > "$OUT/gpu$g.log" 2>&1 &
done
fail=0
for p in $(jobs -p); do wait "$p" || fail=1; done
tail -n 2 $OUT/gpu*.log
n=$(ls $OUT/manifest/*.json 2>/dev/null | wc -l)
say "QG manifests $n/192; failures $fail"
[ "$fail" -eq 0 ] && [ "$n" -ge 192 ] || exit 3
