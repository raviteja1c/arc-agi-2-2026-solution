# Fine-tuned DFlash2 draft (h1)

h1 fine-tunes the seven linear layers in each of the five transformer blocks of
[incoai's DFlash2 draft for Qwen3.8-Flash-Next](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2/tree/a22a1744e85c)
(Apache-2.0). That covers 35 tensors and 426M parameters. The projection, norms, convolutions and candidate
selector keep their original weights. Both submissions load [the same draft](../../inference/MODELS.md#speculative-draft)
at draft block size 4. The target verifies the proposed tokens.

## Acceptance measurements

| Metric | Draft block size 4 | Draft block size 8 |
|---|---|---|
| Estimated generated tokens per verify step, public evaluation histories | 2.775 → 2.840 (+2.3%) | 3.421 → 3.516 (+2.8%) |
| Validation chain, held-out training histories | 1.913 → 1.964 (+2.7%) | 2.947 → 3.042 (+3.2%) |

The public evaluation probe compared the stock draft and step 1500 on the same GPU and 24 recorded conversations
from 10 tasks. `sim.py` divides the generated tokens after each turn's first (prefill-sampled) token by the
expected number of verify steps. Its simulation walks the recorded token spans using acceptance probabilities and
counts the target token emitted with the accepted draft prefix, clipped at the end of each span.

The trainer's `validation chain` measures expected accepted **draft** tokens. It sums cumulative products of the
slot acceptance probabilities (`1 − TV`) and averages over fixed validation anchors on 12 held-out training tasks.
This metric excludes the target token and uses a different set of histories and a different way of weighting
anchors. The evaluation simulation estimates verify-step starts from the acceptance probabilities.

## Prepare the training histories

1. Run [D1](../../entry_points.md#prepare-data) on two Colab RTX PRO 6000 runtimes, one per 100-task slice (`dfg`
   and `dft` in [`fntrain_g4_slices.json`](code/results/fntrain_g4_slices.json)). The seeded draw selects
   ARC-AGI-2 training tasks. The server uses the [capture build](../../patches/sglang/capture/sglang-v0.5.21-to-sub13-e5cb6d9.diff),
   temperature 1.0, top-p 0.95, top-k 20 and `xhigh` reasoning effort. `fntrain_round.sh` deletes and recreates
   `/content/agi2`. The recorded conversations are withheld because they contain competition tasks.
2. Convert the recordings (D2). Follow the [data staging layout](../../entry_points.md#stage-the-training-files)
   and run from this directory with the two sets of recordings:

       VAL_TASKS=12 python code/make_harness_histories.py <out> fntrain 1.0 \
           "<dft round>/lp/*.npz" "<dfg round>/lp/*.npz"

   The converter refuses any task that is not a training task. It sorts all the matched paths together, so the order
   follows the paths, not the arguments. In the original run the `dft` paths sorted first; keep that through the
   folder names to reproduce the index order and the QG plan built from it. The train/validation split is sampled
   from the sorted task IDs. The result was 199 conversations on 199 tasks (187 train, 12 validation), with
   4,796,302 tokens including 3,234,748 generated tokens. This folder ships `data/index.json`, `data/split.json`
   and the 199 request records; the original recordings reproduce them exactly. The token-history arrays are
   withheld because they contain competition tasks.

## Fit and evaluate the draft

1. **Capture (T8).** Install the [staged runtime and probe hook](../../harness/README.md#draft-capture-and-training-t8t9).
   On one RTX PRO 6000, run from this folder:

       bash code/g4/harness_capture_chain.sh <run>-hcap1 <data>

   The script deletes the staged RadixArk safetensors after the server is ready if free space under `/content`
   is below its 90 GiB threshold. Stage a disposable copy, and restore it before step 2 if it was deleted: the
   evaluation server in step 4 reads those files through links. The capture prefilled the histories to save draft
   context features and target top-20 next-token log-probabilities. It took about 20 minutes and produced 58 GB
   of features to regenerate for training.
2. **Train (T9).** First check that both splits were captured (187 of 187 and 12 of 12 in the recorded run):

       python3 /content/inputs/capture_status.py /content/hprobe/out <data>/split.json

   The launcher checks only the training split. Stop the capture server before running the trainer's gradient
   self-test, which checks gradients on the 35 trainable tensors and runs out of memory beside it. Then run:

       bash code/g4/harness_train_gate.sh <run> h1 <data> 1e-4 10

   The launcher waits for `/content/run/<run>-hcap1/capture.exit`, checks the capture and invokes the
   [patched SpecForge trainer](../../patches/README.md). The loss is 0.9 × total variation between target and
   draft sampling distributions plus 0.1 × soft cross-entropy, weighted through the first slot where the draft's greedy choice differs from the recorded token.
   Training used AdamW (0.9, 0.95) on FP32 master weights, weight decay 0.01, clip 1.0, learning rate 1e-4,
   5% warm-up and cosine decay to 10% of the initial rate. Ten epochs alternated draft block sizes 4 and 8
   (1,518 steps, 29 minutes on one RTX PRO 6000). [Forward parity record](run/parity.json).
3. **Select and export.** The launcher selects the checkpoint with the highest `val.B4.chain` in `val.jsonl`:
   step 1500. The [export directory](export/) holds what was published with the weights: `config.json`, h1's note
   [`AGI2_PROVENANCE.md`](export/AGI2_PROVENANCE.md), and the model card [`README.md`](export/README.md), whose figure
   and capture-patch links resolve inside the Kaggle model download.
   [Checkpoint delta manifest](run/ckpt_deltas_MANIFEST.txt).
4. **Compare with the stock draft.** `gate_run.sh` and `sim.py` use the histories selected by
   [`gate_parents.txt`](code/results/gate_parents.txt), recorded by two earlier evaluation probes at temperatures 1.0
   and 1.2. Regenerate or provide evaluation recordings and stage the stock-draft simulation at
   `/content/gate_stock.txt`.
   [Recorded stock results](code/results/gate_baseline_stock_parents.txt). Check the probe and simulation logs
   separately. The training launcher writes its done marker even if evaluation fails.

## Verify the export

The public `model.safetensors` has SHA-256
`9a064a9068f9f539d2b1967c53799a5b26bb34aae0847895e7e71279f188541c`. Verify a downloaded file with:

    python check_export.py <model.safetensors>

With the original step-1500 delta as a second argument, the same tool compares the hashes of the 35 trained tensors
with their checkpoint hashes. The delta is not included.
