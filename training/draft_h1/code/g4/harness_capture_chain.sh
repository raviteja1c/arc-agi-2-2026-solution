#!/bin/bash
# After the harness data rounds (card 1025_2055): on the training G4 (dfg), capture features + teacher for the harness
# training histories (inputs built on the laptop by make_harness_histories.py from every round's lp/*.npz, merged into
# one histories.npz with a split), then train with the approved recipe. The target's safetensors are deleted once the
# capture server holds them when the disk is short (as cap_chain.sh).
#   setsid nohup bash /content/inputs/harness_capture_chain.sh <run_id> <histories dir> > /content/hcap.log 2>&1 < /dev/null &
# <histories dir> holds histories.npz (names + __spans/__lps/__j) and split.json.
set -u
RUN_ID=$1; HD=$2
RUN=/content/run/$RUN_ID
IN=/content/inputs
mkdir -p $RUN
log() { echo "[hcap $(date -u +%H:%M:%S) UTC] $*" | tee -a $RUN/chain.log; }
[ -f $HD/histories.npz ] && [ -f $HD/split.json ] || { log "no histories/split in $HD"; exit 2; }
while pgrep -f '[s]glang.launch_server' > /dev/null || pgrep -f '[a]gi2_modal_harness.comm.harness_nb' > /dev/null; do sleep 15; done
if [ ! -f /content/ft/target_embed.pt ]; then
  /tmp/sgl/bin/python $IN/extract_target.py /tmp/models/qwen3.8-flash-next-nvfp4-radixark /tmp/models/dflash2-qwen3.8-flash-next \
    /content/ft > $RUN/extract.log 2>&1 || { log "extract failed"; exit 3; }
fi
log "target tensors: $(tail -1 $RUN/extract.log 2> /dev/null)"
(
  until curl -sf -m 5 http://127.0.0.1:30000/health > /dev/null; do sleep 10; done
  FREE=$(df -BG --output=avail /content | tail -1 | tr -dc 0-9)
  if [ "$FREE" -lt 90 ]; then
    rm -f /tmp/models/qwen3.8-flash-next-nvfp4-radixark/*.safetensors
    log "deleted the target safetensors (free disk ${FREE}G -> $(df -BG --output=avail /content | tail -1 | tr -dc 0-9)G)"
  fi
) &
# capture_run.sh hands HIST to capture_client.py
HIST=$HD/histories.npz BLOCKS= PROBE_DIR=/content/hprobe NAMES=all bash $IN/capture_run.sh ${RUN_ID}-capture
log "capture: $(cat /content/run/${RUN_ID}-capture/run_exit.txt 2> /dev/null)"
python3 $IN/capture_status.py /content/hprobe/out $HD/split.json | tee -a $RUN/chain.log
echo done > $RUN/capture.exit
