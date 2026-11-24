#!/bin/bash
# G4 setup for the DFlash2 acceptance probe (decode-3x campaign, DFlash2 line):
# - Kaggle's runtime: the 1006-sglang-runtime-cp312 v4 lock, installed as the G4 notebooks do (the commands of
#   1023-colab-g4-flashnext-nv's cells nvuv / nvdl / nvinstall, verbatim);
# - submission 13's build (sglang-v0.5.21-to-sub13-e5cb6d9.diff, its sha256 checked as the sub 13 notebook checks it)
#   plus the probe (dflash2-probe.diff: fork branch agi2/feat/dflash2-probe over e5cb6d9afb);
# - the RadixArk NVFP4 and DFlash2 checkpoints at their pinned revisions, and a text-only copy of the model.
#   bash /content/inputs/probe_setup.sh <run_id>
set -u
RUN_ID=$1
RUN=/content/run/$RUN_ID
IN=/content/inputs
mkdir -p $RUN/obs /content/lib
T0=$(date +%s)
log() { echo "[setup $(date -u +%H:%M:%S) UTC +$(( ($(date +%s) - T0) / 60 )) min] $*" | tee -a $RUN/setup.log; }
fail() { log "FAILED: $*"; echo "setup-failed $*" > $RUN/setup_exit.txt; exit 2; }
# Some G4 images keep the driver's libcuda.so only in /usr/lib64-nvidia (flashinfer's JIT links -lcuda).
[ -e /usr/lib/x86_64-linux-gnu/libcuda.so ] || ln -sf /usr/lib64-nvidia/libcuda.so /usr/lib/x86_64-linux-gnu/libcuda.so
tar -xzf $IN/agi2_lib.tgz -C /content/lib
(cd /content/lib && setsid nohup python3 -m agi2_modal_harness.comm.exp_obs --out $RUN --hours 5 > $RUN/obs/exp_obs.log 2>&1 < /dev/null &)
log "downloads in the background: RadixArk NVFP4 (~135 GB) and DFlash2 (~1.2 GB), plain HTTP with retries"
setsid nohup bash $IN/fn_dl_retry.sh radixark RadixArk/Qwen3.8-Flash-Next-NVFP4 7b719225242aacd3dbd3f9407468c2ee9a9d2594 /tmp/models/qwen3.8-flash-next-nvfp4-radixark $RUN/weights_radixark.log > /dev/null 2>&1 < /dev/null &
setsid nohup bash $IN/fn_dl_retry.sh dflash2 incoai/Qwen3.8-Flash-Next-DFlash2 a22a1744e85c17364dcc54a6a420af7135d4c7f6 /tmp/models/dflash2-qwen3.8-flash-next $RUN/weights_dflash2.log > /dev/null 2>&1 < /dev/null &
log "CPython 3.12.15 + uv 0.12.21"
mkdir -p /tmp/runtime/dl /tmp/runtime/wheels
wget -q -O /tmp/runtime/python312.tar.gz.bin https://github.com/astral-sh/python-build-standalone/releases/download/20261003/cpython-3.12.15%2B20261003-x86_64-unknown-linux-gnu-install_only.tar.gz || fail "python download"
curl -LsSf https://astral.sh/uv/0.12.21/install.sh | env UV_INSTALL_DIR=/tmp/uv UV_NO_MODIFY_PATH=1 sh > /dev/null || fail "uv install"
log "fetching $(wc -l < $IN/wheels.txt) wheelhouse files (32 in parallel, ~7 GB)"
cd /tmp/runtime
awk '{print $1, $3}' $IN/wheels.txt | xargs -P 32 -n 2 sh -c 'curl -sfSL --retry 3 -o "dl/$2" "$1"' _ || log "some fetches failed; checking every file"
while read url sha name; do
  if ! echo "$sha  dl/$name" | sha256sum -c --quiet 2> /dev/null; then
    log "refetch $name"
    curl -sfSL --retry 5 -o "dl/$name" "$url"
    echo "$sha  dl/$name" | sha256sum -c --quiet || fail "sha256 of $name"
  fi
done < $IN/wheels.txt
mv dl/*.whl wheels/
for s in dl/*.tar.gz; do /tmp/uv/uv build -q --wheel --out-dir wheels "$s" || fail "sdist build $s"; done
log "$(ls wheels | wc -l) wheels ready; installing the venv"
tar -xzf /tmp/runtime/python312.tar.gz.bin -C /tmp
/tmp/python/bin/python3.12 -m venv /tmp/sgl
/tmp/uv/uv pip install --python /tmp/sgl/bin/python --offline --no-index --no-deps --find-links /tmp/runtime/wheels -r $IN/requirements.lock > $RUN/install.log 2>&1 || fail "venv install"
SITE=$(/tmp/sgl/bin/python -c "import os, sglang; print(os.path.dirname(os.path.dirname(sglang.__file__)))")
echo "e588512b09e45eae257ad0d077b99241178d8204ee039a50227d753ba0b168cf  $IN/sglang-v0.5.21-to-sub13-e5cb6d9.diff" | sha256sum -c --quiet || fail "sub13 diff sha256"
patch -p2 --fuzz=0 --forward --no-backup-if-mismatch -d "$SITE" < $IN/sglang-v0.5.21-to-sub13-e5cb6d9.diff > $RUN/patch_sub13.log 2>&1 || fail "sub13 patch"
patch -p2 --fuzz=0 --forward --no-backup-if-mismatch -d "$SITE" < $IN/dflash2-probe.diff > $RUN/patch_probe.log 2>&1 || fail "probe patch"
grep -q "def agi2_expert_keep_lists" $SITE/sglang/srt/configs/qwen4_exp.py || fail "keep-list code missing"
test -f $SITE/sglang/srt/speculative/fn_conf.py || fail "sub13 tree incomplete"
test -f $SITE/sglang/srt/speculative/agi2_dflash_probe.py || fail "probe module missing"
grep -q "_agi2_probe.on_extend" $SITE/sglang/srt/speculative/dflash_worker_v2.py || fail "probe hook missing"
/tmp/sgl/bin/python -c "import torch, sglang, flashinfer; print('torch', torch.__version__, '| sglang', sglang.__version__, '| flashinfer', flashinfer.__version__, '|', torch.cuda.get_device_name(0))" 2>&1 | tail -1 | tee -a $RUN/setup.log
log "build in place: sub 13 ($(grep -c '^patching file' $RUN/patch_sub13.log) files) + probe ($(grep -c '^patching file' $RUN/patch_probe.log)); waiting for the downloads"
for a in radixark dflash2; do
  until grep -qx "$a DONE" $RUN/weights_$a.log 2> /dev/null; do
    grep -q "FAILED after" $RUN/weights_$a.log 2> /dev/null && fail "$a download"
    sleep 15
  done
  log "$a downloaded: $(du -sh /tmp/models/* 2> /dev/null | tr '\n' ' ')"
done
SRC=/tmp/models/qwen3.8-flash-next-nvfp4-radixark; TXT=/tmp/models/flashnext-textonly
rm -rf $TXT && mkdir -p $TXT
for f in $SRC/*; do [ "$(basename $f)" = config.json ] || ln -s "$f" "$TXT/"; done
/tmp/sgl/bin/python -c "import json, sys; c = json.load(open(sys.argv[1])); c['language_model_only'] = True; json.dump(c, open(sys.argv[2], 'w'), indent=2)" $SRC/config.json $TXT/config.json
grep -q '"language_model_only": true' $TXT/config.json || fail "text-only copy"
rm -rf /tmp/runtime/dl /tmp/runtime/wheels
log "ready"
echo "ok" > $RUN/setup_exit.txt
