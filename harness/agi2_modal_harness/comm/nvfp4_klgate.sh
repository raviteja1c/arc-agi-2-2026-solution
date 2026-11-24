#!/usr/bin/env bash
# End-to-end KL gate on a 4 x H200 box (pinned build, sm_90; Marlin W4A16 for NVFP4 like the L4): one server per
# build at all 512 experts, identical flags except the NVFP4 quantization ones, prefill only; each scores the 4
# held-out harness histories past position 640 (nvfp4_klgate score, top-20); then KL(BF16 || build) per position.
# BF16 first: it is the reference and the riskiest bring-up.
#   OUT=$OUT bash agi2_modal_harness/comm/nvfp4_klgate.sh
set -u
OUT=${OUT:?out dir}
RADIX=/vol/models/RadixArk__Qwen3.8-Flash-Next-NVFP4__7b719225242a
BUILDS=${BUILDS:-"bf16 radix qg=/vol/cache/dev/fnxq/qg1 qs=/vol/cache/dev/fnxq/qs1"}
HIST=/opt/dev/data/xq_histories.npz
PLAN=/opt/dev/data/xq_plan.json
t0=$(date +%s)
say() { echo "[klgate $(date +%H:%M:%S), +$(( $(date +%s) - t0 )) s] $*"; }
[ -f "$HIST" ] && [ -f "$PLAN" ] || { say "FAILED: histories or plan missing"; exit 2; }
for d in triton inductor jit sglang fi xdg; do mkdir -p /tmp/klc/$d; done
export TRITON_CACHE_DIR=/tmp/klc/triton TORCHINDUCTOR_CACHE_DIR=/tmp/klc/inductor SGLANG_JIT_CACHE_DIR=/tmp/klc/jit \
    SGLANG_CACHE_DIR=/tmp/klc/sglang FLASHINFER_WORKSPACE_BASE=/tmp/klc/fi XDG_CACHE_HOME=/tmp/klc/xdg

# the BF16 checkpoint: 96 verified expert shards + the 47 other files (incl. the index), all 131 shards present;
# the other box fetches the 47 into the Volume, so reload until its FETCH_OK marker shows
for i in $(seq 1 90); do
    python -m agi2_modal_harness.comm.nvfp4_vol reload > /dev/null 2>&1 || true
    [ -f /vol/cache/dev/fnbf16_rest2/FETCH_OK.json ] && ! ls /vol/cache/dev/fnbf16_rest2/*.part > /dev/null 2>&1 && break
    [ $((i % 4)) -eq 1 ] && say "waiting for the BF16 rest on the Volume: $(ls /vol/cache/dev/fnbf16_rest2 2>/dev/null | wc -l) entries"
    sleep 20
done
rm -rf /tmp/fnbf16full; mkdir -p /tmp/fnbf16full
for f in /vol/cache/dev/fnbf16_rest2/*; do ln -s "$f" /tmp/fnbf16full/; done
for f in /vol/cache/dev/fnbf16_parts/model-*.safetensors; do [ -e "/tmp/fnbf16full/$(basename $f)" ] || ln -s "$f" /tmp/fnbf16full/; done
n=$(ls /tmp/fnbf16full/model-*.safetensors | wc -l)
[ "$n" -eq 131 ] && [ -f /tmp/fnbf16full/config.json ] && [ -f /tmp/fnbf16full/model.safetensors.index.json ] || {
    say "FAILED: BF16 dir has $n/131 shards (config/index present: $(ls /tmp/fnbf16full/*.json | wc -l) json)"; exit 2; }
say "BF16 dir ready: $n shards"

COMMON=(--tp 4 --ep-size 4 --trust-remote-code --host 127.0.0.1 --port 30000 --context-length 32768
        --mem-fraction-static 0.85 --max-running-requests 2 --chunked-prefill-size 1024 --kv-cache-dtype fp8_e4m3
        --mamba-ssm-dtype bfloat16 --page-size 64 --disable-radix-cache --ple-offload-embedding --disable-cuda-graph
        --served-model-name fn)
FP4=(--quantization modelopt_fp4 --fp4-gemm-backend marlin --moe-runner-backend marlin)
for b in $BUILDS; do
    name=${b%%=*}
    case "$name" in
        bf16) md=/tmp/fnbf16full; q=();;
        radix) md=$RADIX; q=("${FP4[@]}");;
        *) d=${b#*=}
           if [ -f "$d/config.json" ]; then md=$d   # a complete model dir (e.g. nvfp4_ple_overlay)
           else BUILD=$d OVL=/tmp/fnxq/model_$name bash agi2_modal_harness/comm/nvfp4_overlay.sh || exit 2
                md=/tmp/fnxq/model_$name; fi
           q=("${FP4[@]}");;
    esac
    say "boot $name ($md)"
    python -m sglang.launch_server --model-path "$md" "${COMMON[@]}" "${q[@]}" > "$OUT/$name.server.log" 2>&1 &
    pid=$!
    ok=0
    for i in $(seq 1 270); do
        kill -0 $pid 2>/dev/null || break
        curl -sf -m 5 http://127.0.0.1:30000/health_generate > /dev/null 2>&1 && { ok=1; break; }
        [ $((i % 6)) -eq 0 ] && say "  $name booting, $((i * 10)) s: $(grep -v '^\s*$' "$OUT/$name.server.log" | tail -1 | cut -c1-160)"
        sleep 10
    done
    if [ "$ok" -ne 1 ]; then
        say "FAILED: $name did not come up"; tail -30 "$OUT/$name.server.log"; pkill -9 -f '[s]glang'; exit 3
    fi
    say "$name up; scoring"
    python -m agi2_modal_harness.comm.nvfp4_klgate score --hist "$HIST" --plan "$PLAN" --out "$OUT/$name.npz" \
        > "$OUT/$name.score.log" 2>&1
    rc=$?
    say "$name score exit $rc | $(grep -hE 'kl-score|PHASE|Error' "$OUT/$name.score.log" | tr '\n' ' ' | cut -c1-400)"
    pkill -f '[s]glang.launch_server'; sleep 15; pkill -9 -f '[s]glang' 2>/dev/null; sleep 5
    [ "$rc" -eq 0 ] || exit 4
done
args=()
for b in $BUILDS; do n=${b%%=*}; [ "$n" = bf16 ] || args+=(--build "$n=$OUT/$n.npz"); done
python -m agi2_modal_harness.comm.nvfp4_klgate compare --ref "$OUT/bf16.npz" "${args[@]}" --out "$OUT/kl.json"
say "done"
