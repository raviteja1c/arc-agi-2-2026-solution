#!/bin/bash
# On dfg after harness_capture_chain.sh (card 1025_1926): train the DFlash2 draft on the harness histories with the
# approved recipe (7 linears, 0.9 TV + 0.1 CE, AUF, cosine; lr from the fit test, blocks 4 / 8 alternating), pick the
# checkpoint with the best validation chain at block 4 (held-out training tasks), export it, and gate it: the probe at
# blocks 4 and 8 over the 24 eval-parent histories on this box's target, then sim.py against the paired stock run.
#   setsid nohup bash /content/inputs/harness_train_gate.sh <ft run_id> <tag> <histories dir> <lr> <epochs> > /content/htg_<tag>.log 2>&1 < /dev/null &
set -u
RUN_ID=$1; TAG=$2; HD=$3; LR=$4; EPOCHS=$5
OUT=/content/ft/$RUN_ID
IN=/content/inputs
mkdir -p $OUT
log() { echo "[htg $(date -u +%H:%M:%S) UTC] $*" | tee -a $OUT/htg_$TAG.log; }
CAP=/content/run/${RUN_ID}-hcap1
until [ -f $CAP/capture.exit ]; do sleep 15; done
python3 $IN/capture_status.py /content/hprobe/out $HD/split.json > $OUT/hcap_status_$TAG.txt
cat $OUT/hcap_status_$TAG.txt | tee -a $OUT/htg_$TAG.log
NT=$(python3 -c "import json; s=json.load(open('$HD/split.json')); print(len(s['train']), len(s['val']))")
grep -q "train: $(echo $NT | cut -d' ' -f1) of $(echo $NT | cut -d' ' -f1) done" $OUT/hcap_status_$TAG.txt || { log "capture incomplete"; exit 2; }
log "train $TAG: lr $LR, $EPOCHS epochs, histories $HD"
bash $IN/ft_run.sh $RUN_ID $TAG train --probe-out /content/hprobe/out --hist $HD/histories.npz --split $HD/split.json \
  --lr $LR --epochs $EPOCHS --ckpt-every 100 --log-every 20 --val-stride 4
[ "$(cat $OUT/$TAG.exit)" = 0 ] || { log "training failed"; exit 3; }
BEST=$(python3 - "$OUT/$TAG/val.jsonl" << 'EOF'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1])]
best = max((r for r in rows if r["step"] > 0), key=lambda r: r["val"]["B4"]["chain"])
print(best["step"], best["val"]["B4"]["chain"], rows[0]["val"]["B4"]["chain"])
EOF
)
STEP=$(echo $BEST | cut -d' ' -f1)
log "best val B4 chain at step $STEP: $(echo $BEST | cut -d' ' -f2) (step 0: $(echo $BEST | cut -d' ' -f3))"
CK=$OUT/$TAG/ckpt/step$(printf %06d $STEP).safetensors
EXP=/content/gate_queue/${TAG}_s$STEP
PYTHONPATH=/content/sf /tmp/sgl/bin/python /content/sf/scripts/agi2_dflash2_offline_ft.py export --weights $CK \
  --export-dir $EXP --out $OUT/${TAG}_export > $OUT/${TAG}_export.log 2>&1 || { log "export failed"; exit 4; }
log "exported $EXP ($(tail -1 $OUT/${TAG}_export.log | cut -c1-120))"
DRAFT=$EXP PROBE_DIR=/content/gate_probe_${TAG}_s$STEP bash $IN/gate_run.sh ${RUN_ID}-gate-${TAG}_s$STEP \
  --only "$(cat $IN/gate_parents.txt)" --no-gate
log "gate probe: $(cat /content/run/${RUN_ID}-gate-${TAG}_s$STEP/run_exit.txt 2> /dev/null)"
cd /content && python3 $IN/sim.py --probe /content/gate_probe_${TAG}_s$STEP/out --inputs $IN --names $(cat $IN/gate_parents.txt) \
  --blocks 4,8 --trees 16 --json /content/gate_${TAG}_s$STEP.json > /content/gate_${TAG}_s$STEP.txt 2>&1
log "gate $TAG s$STEP: $(grep -E 'block  [48]:' /content/gate_${TAG}_s$STEP.txt | cut -c1-70 | tr '\n' ' ')"
log "stock (paired, this box): $(grep -E 'block  [48]:' /content/gate_stock.txt | cut -c1-70 | tr '\n' ' ')"
echo done > $OUT/htg_$TAG.exit
