#!/bin/bash
# The 32k expert-routing capture on a 4 x L40S box set up by setup.sh (Kaggle runtime + sub 18 build, QG experts):
# apply the fork's MoE capture hook (agi2/feat/expert-reround ea30d8e7af: agi2_capture.py + one call in FusedMoE.forward)
# to the installed sglang, boot submission 18's own server (K384, QG, served routing) via harness_nb --boot-only, and run
# capture_client.py as its probe over the 16 histories of xq32_plan.json. Then copy the capture to the Volume
# (/vol/cache/dev/fncap/xq32) and restore the installed files.
#   comm.dev run --exp ... -- bash /vol/cache/dev/fncap/xq32_inputs/capture32.sh
set -u
: "${OUT:?run this under comm.dev run --exp}" "${RUN_ID:?run this under comm.dev run --exp}"
I=/vol/cache/dev/fncap/xq32_inputs
IN=/content/inputs
M=/tmp/models/qwen3.8-flash-next-nvfp4-radixark
D=/tmp/models/dflash2-h1
Q=/tmp/models/qg1
CAP=/tmp/fncap32
CFG=agi2_modal_harness/workloads/sglang_bench/configs/l4x4_sub14_notebook_h1.toml
NB18=notebooks/1030-sub18-qwen38-flashnext-s48-5d76d78-int8/1030-sub18-qwen38-flashnext-s48-5d76d78-int8.ipynb
mkdir -p $OUT
log() { echo "[cap32 $(date -u +%H:%M:%S) UTC] $*" | tee -a $OUT/cap32.log; }
fail() { log "FAILED: $*"; echo "$RUN_ID fail $*" > $OUT/round_exit.txt; exit 2; }
[ "$(cat /content/setup_exit.txt 2> /dev/null)" = ok ] || fail "the box setup has not finished"
grep -q "sub18 5d76d788f1" /content/runtime_ok 2> /dev/null || fail "the runtime is not sub 18's build"
if pgrep -f '[s]glang.launch_server' > /dev/null || pgrep -f '[a]gi2_modal_harness.comm.harness_nb' > /dev/null; then
  fail "a server or a round is still running"
fi
SP=$(/tmp/sgl/bin/python -c "import sglang, os; print(os.path.dirname(os.path.dirname(sglang.__file__)))")
[ -d "$SP/sglang/srt/layers/moe" ] || fail "sglang site-packages not found ($SP)"
BK=$OUT/orig; mkdir -p $BK
cp "$SP/sglang/srt/layers/moe/fused_moe_triton/layer.py" $BK/layer.py
# the diff's paths are python/sglang/...: from site-packages strip 2 components (a/python)
(cd "$SP" && patch -p2 --forward < $I/moe_capture.diff) > $OUT/patch.txt 2>&1 || { cat $OUT/patch.txt; fail "patch"; }
restore() {
  cp $BK/layer.py "$SP/sglang/srt/layers/moe/fused_moe_triton/layer.py"
  rm -f "$SP/sglang/srt/layers/moe/agi2_capture.py"
  log "installed files restored"
}
trap restore EXIT
log "capture hook applied to $SP ($(grep -c agi2_capture "$SP/sglang/srt/layers/moe/fused_moe_triton/layer.py") references)"
W=/content/agi2_cap32
rm -rf $W && mkdir -p $W && tar -xzf $IN/agi2_repo_s18.tgz -C $W || fail "tarball"
cd $W
export PYTHONPATH=$W
rm -rf $CAP && mkdir -p $CAP
PROBE="/tmp/sgl/bin/python -u $I/capture_client.py $CAP $I/xq32_histories.npz $I/xq32_plan.json > $OUT/capture_client.log 2>&1"
ARGS=(--notebook "$NB18" --data data/fresh61 --config "$CFG" --slots 48 --rehearsal bench --bench-minutes 1
      --model-dir "$M" --draft-dir "$D" --qg-dir "$Q" --mem-fraction 0.44 --decode-log-interval 20
      --server-flag "SGLANG_AGI2_MOE_CAPTURE_DIR=$CAP" --boot-only --probe "$PROBE")
export HARNESS_WORK=/tmp/hw/$RUN_ID HARNESS_CACHE_TAG=$RUN_ID
mkdir -p $HARNESS_WORK
log "booting sub 18's server with the capture on; then 16 x 32k histories"
/tmp/sgl/bin/python -u -m agi2_modal_harness.comm.harness_nb "${ARGS[@]}" > $OUT/harness_nb.log 2>&1 < /dev/null
RC=$?
cp $HARNESS_WORK/sglang.log $OUT/sglang.log 2> /dev/null
log "harness_nb exit $RC | $(tail -n 2 $OUT/capture_client.log 2>/dev/null | tr '\n' ' ' | cut -c1-200)"
[ -f $CAP/late32/manifest.json ] || fail "no manifest: the capture did not finish"
du -sh $CAP | tee -a $OUT/cap32.log
mkdir -p /vol/cache/dev/fncap/xq32
t0=$(date +%s)
cp -r $CAP/late32 /vol/cache/dev/fncap/xq32/ || fail "copy to the Volume"
python3 -m agi2_modal_harness.comm.nvfp4_vol commit > /dev/null 2>&1
log "copied to /vol/cache/dev/fncap/xq32 in $(( $(date +%s) - t0 )) s and committed"
echo "$RUN_ID 0" > $OUT/round_exit.txt
