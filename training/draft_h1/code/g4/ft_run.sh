#!/bin/bash
# Run one mode of the DFlash2 draft fine-tune trainer (SpecForge fork agi2/feat/dflash2-offline-ft,
# scripts/agi2_dflash2_offline_ft.py, unpacked at /content/sf) on the G4, logging to /content/ft/<run_id>/<tag>.log.
#   bash /content/inputs/ft_run.sh <run_id> <tag> <mode> [trainer args...]
# e.g. ft_run.sh 1025_1926_colab-g4-dflash2-ft parity parity --probe-out /content/probe_parity/out --names a,b,c
set -u
RUN_ID=$1; TAG=$2; MODE=$3; shift 3
OUT=/content/ft/$RUN_ID
mkdir -p $OUT
export PYTHONPATH=/content/sf PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUDA_HOME=/tmp/sgl/lib/python3.12/site-packages/nvidia/cu13
export PATH=/tmp/sgl/bin:$CUDA_HOME/bin:$PATH
export TRITON_CACHE_DIR=/tmp/cache/triton_ft TORCHINDUCTOR_CACHE_DIR=/tmp/cache/inductor_ft
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "[ft-run $(date -u +%H:%M:%S) UTC] $MODE $*" | tee -a $OUT/$TAG.log
/tmp/sgl/bin/python /content/sf/scripts/agi2_dflash2_offline_ft.py $MODE --out $OUT/$TAG "$@" >> $OUT/$TAG.log 2>&1
RC=$?
echo "[ft-run $(date -u +%H:%M:%S) UTC] exit $RC" | tee -a $OUT/$TAG.log
echo $RC > $OUT/$TAG.exit
