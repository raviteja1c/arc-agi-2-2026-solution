#!/usr/bin/env bash
# Flash-Next on 4 x L4 (docs/reports/1023_0106_flashnext_l4x4/): one bring-up boot plus probes on a held dev box.
#   comm.dev --session <s> run --exp <slug> --detach -- bash agi2_modal_harness/comm/fn_l4_up.sh <label> [--full] \
#       [fn_bringup.py args...]
# Text-only by default: a copy of the checkpoint directory under /tmp with every file linked and config.json setting
# language_model_only (the server flag refuses Qwen4-Exp; the vision tower costs memory on every rank), as the G4
# lever boots did (notebooks/1022-colab-g4-flashnext-a/fn_levers.sh). --full serves the vision-language model.
# Writes $OUT/<label>/ (up.log, fn_bringup.log, probe.json, sglang.log, exit.txt); fn_bringup.py from /opt/dev.
set -u
LABEL=$1; shift
MODE=text
if [ "${1:-}" = "--full" ]; then MODE=full; shift; fi
# FN_MODEL_SRC serves another checkpoint in the same layout (e.g. NVIDIA's QAD weights converted by fn_l4_nv_convert.py).
SRC=${FN_MODEL_SRC:-/vol/models/RadixArk__Qwen3.8-Flash-Next-NVFP4__7b719225242a}
MODEL=$SRC
if [ "$MODE" = text ]; then
    TXT=/tmp/fn_textonly
    # Another checkpoint's copy is rebuilt every time (links only): its source may have changed since the last boot
    # (1023_1514 booted a stale copy made before the NVIDIA overlay linked its shards at the top level).
    [ -n "${FN_MODEL_SRC:-}" ] && TXT=/tmp/fn_textonly_$(basename "$SRC") && rm -rf "$TXT"
    if ! grep -q '"language_model_only": true' $TXT/config.json 2> /dev/null; then
        rm -rf $TXT && mkdir -p $TXT
        for f in $SRC/*; do ln -s "$f" $TXT/; done
        rm $TXT/config.json
        python -c "import json, sys; c = json.load(open(sys.argv[1])); c['language_model_only'] = True; json.dump(c, open(sys.argv[2], 'w'), indent=2)" $SRC/config.json $TXT/config.json
    fi
    grep -q '"language_model_only": true' $TXT/config.json || { echo "text-only copy FAILED"; exit 1; }
    MODEL=$TXT
fi
O=$OUT/$LABEL
mkdir -p $O
echo "[fn-l4-up $(date -u +%H:%M:%S) UTC] $LABEL: model $MODEL ($MODE); patch $(cat /opt/dev/patches/CURRENT 2> /dev/null)" | tee -a $O/up.log
ulimit -l unlimited 2> /dev/null
echo "memlock $(ulimit -l); nproc $(nproc); $(free -g | sed -n 2p)" | tee -a $O/up.log
python /opt/dev/fn_bringup.py --out $O --model-dir $MODEL --cache-gpu L4 "$@" > $O/fn_bringup.log 2>&1
RC=$?
echo "$LABEL $RC" > $O/exit.txt
echo "[fn-l4-up $(date -u +%H:%M:%S) UTC] $LABEL exit $RC" | tee -a $O/up.log
exit $RC
