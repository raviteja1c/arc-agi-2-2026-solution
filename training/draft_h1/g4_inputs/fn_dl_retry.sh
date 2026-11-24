#!/bin/bash
# Download a pinned Hugging Face snapshot with retries, resuming each time (finished files are kept), for unauthenticated
# runtimes that draw HF's 429 (2026-10-23: two xet attempts at NVIDIA FP4-Local died at 6 and 11 of 21 shards).
# Odd tries use plain HTTP (HF_HUB_DISABLE_XET=1, a different CDN path), even tries xet at default concurrency; the
# wait between tries grows 60, 120, 180 s. Writes "<alias> DONE" as a whole line on success.
#   bash fn_dl_retry.sh <alias> <repo> <revision> <local dir> <log> [tries]
set -u
ALIAS=$1; REPO=$2; REV=$3; DIR=$4; LOG=$5; TRIES=${6:-8}
for t in $(seq 1 $TRIES); do
  if [ $((t % 2)) = 1 ]; then XET="HF_HUB_DISABLE_XET=1"; else XET="HF_HUB_DISABLE_XET=0"; fi
  echo "[dl $(date -u +%H:%M:%S) UTC] $ALIAS try $t ($XET), $(du -sh $DIR 2> /dev/null | cut -f1) on disk" >> $LOG
  env $XET HF_XET_CHUNK_CACHE_SIZE_BYTES=0 python3 -c "from huggingface_hub import snapshot_download as get; get('$REPO', revision='$REV', local_dir='$DIR')" >> $LOG 2>&1 && { echo "$ALIAS DONE" >> $LOG; exit 0; }
  echo "[dl $(date -u +%H:%M:%S) UTC] $ALIAS try $t failed: $(grep -E '^[A-Za-z]*Error' $LOG | tail -1 | cut -c1-200)" >> $LOG
  sleep $((60 * ((t + 1) / 2)))
done
echo "[dl] $ALIAS FAILED after $TRIES tries" >> $LOG
exit 1
