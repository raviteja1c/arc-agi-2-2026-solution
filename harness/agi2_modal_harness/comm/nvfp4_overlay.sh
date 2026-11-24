#!/usr/bin/env bash
# Build a servable model directory for an expert re-rounding build: every file of RadixArk's checkpoint symlinked,
# except the 192 routed-expert shards, which link to the build's own files (same names). Fails closed unless the
# build has all 192 shards and the overlay ends with exactly RadixArk's file list.
#   BUILD=/tmp/fnxq/qs1 OVL=/tmp/fnxq/model_qs1 bash agi2_modal_harness/comm/nvfp4_overlay.sh
set -eu
RADIX=${RADIX:-/vol/models/RadixArk__Qwen3.8-Flash-Next-NVFP4__7b719225242a}
BUILD=${BUILD:?build dir with layer-*-experts-*.safetensors}
OVL=${OVL:?overlay dir to create}
n=$(ls "$BUILD"/layer-*-experts-*.safetensors 2>/dev/null | wc -l)
[ "$n" -eq 192 ] || { echo "[overlay] FAILED: $BUILD has $n/192 expert shards"; exit 2; }
rm -rf "$OVL"; mkdir -p "$OVL"
for f in "$RADIX"/*; do
    b=$(basename "$f")
    case "$b" in
        layer-*-experts-*.safetensors) ln -s "$BUILD/$b" "$OVL/$b";;
        *) ln -s "$f" "$OVL/$b";;
    esac
done
want=$(ls "$RADIX" | sort | md5sum); got=$(ls "$OVL" | sort | md5sum)
[ "$want" = "$got" ] || { echo "[overlay] FAILED: file list differs from RadixArk's"; exit 2; }
echo "[overlay] $OVL: $(ls "$OVL" | wc -l) entries, 192 expert shards from $BUILD"
