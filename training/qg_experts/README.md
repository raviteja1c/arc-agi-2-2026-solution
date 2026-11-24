# GPTQ re-rounding for routed experts (QG)

QG re-rounds the routed experts of RadixArk's NVFP4 Qwen3.8-Flash-Next export. GPTQ quantizes each matrix
column by column with error feedback, using MoE activations captured from the model's runs on ARC-AGI-2 training
tasks. The weights keep the NVFP4 format (4-bit values, one E4M3 scale per 16 values). The 192 shards replace
RadixArk's routed-expert shards in both submissions. [Kaggle model and versions](../../inference/MODELS.md).

## Results

The four held-out training histories contain 52,296 recorded positions. Both tests skip the first 640 positions
of each history, leaving 49,736 scored positions.

| Metric | RadixArk (round-to-nearest) | QS (scale-only) | QG |
|---|---|---|---|
| Routed-MoE relative squared output error against BF16, mean over 48 layers | 0.01525 | 0.01131 | 0.00557 (−63.5%) |
| Top-20 next-token KL estimate against BF16 | 0.258 | 0.238 | 0.194 (−25.0%) |
| Top-1 agreement with BF16 | 85.4% | 86.1% | 87.6% |

The layer test uses the captured census-list routes and weights for all rounding methods and the BF16 reference.
QG reduced error in every layer (52.6–92.6%) and every held-out task. The end-to-end test served all 512 experts
per layer in each build and in the BF16 reference on 4 × H200 to measure rounding at full width. The
[expert-selection test](../keep_c512/README.md#results) compares two 384-expert configurations using QG.

The overlay covers all 24,576 experts. Of these, 18,329 use GPTQ and 6,247 use scale-only rounding (QS).
The QS set includes 6,144 experts outside the calibration server's census list and 103 kept experts with
fewer than 256 calibration rows. For each eligible matrix, the build writes QG when its calibration proxy
is at most the QS proxy error.

## Fit the experts

Run the modules from [`harness/`](../../harness/README.md), or with it on `PYTHONPATH`. Set these paths, in full:

- `BF16`: the 96 BF16 expert shards of Qwen/Qwen3.8-Flash-Next at `de4b8e4d43b9`. The end-to-end test in step 4
  reads them from `/vol/cache/dev/fnbf16_parts`, so use that path. Fetch and check them:

      BF16=/vol/cache/dev/fnbf16_parts
      python -m agi2_modal_harness.comm.nvfp4_bf16_fetch --dest "$BF16"
      python -m agi2_modal_harness.comm.nvfp4_verify --dir "$BF16"

- `KEEP`: the [census list](../keep_census/keep_384_code_bal4.json).
- `RADIX`: the [RadixArk checkpoint](../../inference/MODELS.md#base-checkpoint), for its global weight scales.

1. **Plan the capture (D3).** With the [h1 history directory](../draft_h1/README.md#prepare-the-training-histories)
   containing `index.json`, `split.json` and `histories.npz`, run:

       python -m agi2_modal_harness.comm.nvfp4_capture_plan \
           --src <history-directory> --out <plan-directory> \
           --calib-cap 16000 --gate-cap 16000 --calib-tokens 350000

   The plan has 27 calibration histories (350,445 positions) and four held-out histories (52,296 positions).
   The [recorded plan](capture_inputs/xq_plan.json) includes the source path; the original histories reproduce it
   exactly. Its `xq_histories.npz` is withheld because it contains competition tasks.
2. **Capture MoE inputs (T3).** `S24QZBRP`, a section of `l4x4_sub14_b8study.toml`, is the server configuration with
   the census list, FP8 KV cache and h1.
   Use [the submission 17 capture build and MoE hook](../../patches/README.md). From `harness/`:

       python -m agi2_modal_harness.comm.nvfp4_capture \
           --config agi2_modal_harness/workloads/sglang_bench/configs/l4x4_sub14_b8study.toml \
           --arm S24QZBRP --out "$OUT" --cap-dir /tmp/fncap \
           --histories <plan-directory>/xq_histories.npz --plan <plan-directory>/xq_plan.json

   Set `OUT` to the run output directory. The config reads the model directories from `/vol/models/` and the
   census list from `/vol/cache/dev/fn_keep/keep_384_code_bal4.json`. Use `--model-dir` for a different base-model
   path and `--env SGLANG_AGI2_FN_EXPERT_KEEP=<path>` for the list. The capture records each layer's input and
   the served top-10 expert IDs and weights on rank 0. It prefills with radix caching off and checks each
   layer's token count after the first history. It took 15.2 minutes on 4 × L40S.
3. **Build the shards (T4).** After verifying the BF16 shards, run from `harness/`:

       CAP=/tmp/fncap OUT=/tmp/fnxq/qg1 BF16="$BF16" KEEP="$KEEP" QG_ARGS="--radix $RADIX" \
           bash agi2_modal_harness/comm/nvfp4_qg_4gpu.sh

   `nvfp4_gptq_selftest.py` tests the rounding methods on synthetic data on CUDA. The build ran about 10 minutes
   across four L40S GPUs, 12 layers per GPU.

   For inputs `x` weighted by routing weights `a`, gate and up projections use the Hessian approximation
   `Σ a² x xᵀ` with 1% damping. The down projection uses the quantized gate and up outputs as its inputs.
   GPTQ processes blocks of 128 with an exact scale search for each 16-value group.

   [`shard_manifests/`](shard_manifests/) has 192 records: 48 detailed layer records and 144 pointers to them.
   The detailed records give per-matrix proxies and choices for experts eligible for GPTQ, and a list of QS-only
   experts. Put the records in `<build-directory>/manifest/` and run
   `python -m agi2_modal_harness.comm.nvfp4_manifest_sum <build-directory>` to summarize them.
4. **Evaluate.** The end-to-end launcher reads fixed paths:
   - the plan and histories at `/opt/dev/data/xq_plan.json` and `/opt/dev/data/xq_histories.npz`;
   - RadixArk at `/vol/models/RadixArk__Qwen3.8-Flash-Next-NVFP4__7b719225242a`;
   - the BF16 checkpoint in `/vol/cache/dev/fnbf16_parts/` (the expert shards) and `/vol/cache/dev/fnbf16_rest2/`
     (every other file).

   Stage the plan and histories there, and set `QG`, `QS` and `OUT` to full paths, with `OUT` created. Then, from
   `harness/`:

       python -m agi2_modal_harness.comm.nvfp4_bf16_fetch --rest --dest /vol/cache/dev/fnbf16_rest2
       python -m agi2_modal_harness.comm.nvfp4_reround --mode qs --bf16 "$BF16" --radix "$RADIX" --out "$QS"
       python -m agi2_modal_harness.comm.nvfp4_blockgate --cap /tmp/fncap --keep "$KEEP" --bf16 "$BF16" \
           --build "radix=$RADIX" --build "qs=$QS" --build "qg=$QG" \
           --plan /opt/dev/data/xq_plan.json --exclude-prefix 640 --max-tokens 49736 --out "$OUT/block.json"
       BUILDS="bf16 radix qg=$QG qs=$QS" OUT="$OUT" bash agi2_modal_harness/comm/nvfp4_klgate.sh

   The second line builds the scale-only (QS) comparison, the third measures the layer output error, and the last
   the end-to-end KL. The end-to-end test ran on 4 × H200; on another machine, stage the shards and the D3 files at
   the same paths there.

Rebuilding the shards needs the approximately 236 GiB of BF16 expert weights, the capture (92.9 GiB across both
phases) and RadixArk's global scales. The capture has to be recorded again.

## Verify the shards

[`qg_volume.sha256`](hashes/qg_volume.sha256) records the 192 build hashes. The shards were staged through a private
Hugging Face repository before Kaggle; its [file tree](hashes/hf_tree.json) holds the same shard hashes, and its 196
filenames and sizes match the [Kaggle listing](../../inference/models/qg1.agi2-qg1-8352943a271b.v1.files.json).
The Kaggle model ships `MANIFEST.sha256` to check a download against. All 192 shards differ from the
[original RadixArk shards](hashes/radix_experts.sha256).
