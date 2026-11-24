# Model weights and fitted lists

Attach the public base-model mirror [`carriem7/qwen3-8-flash-next-nvfp4-mirror`](https://www.kaggle.com/models/carriem7/qwen3-8-flash-next-nvfp4-mirror/Transformers/hf-7b719225242a/1).
The scored runs mounted an identical copy that is not public. The notebooks find the variation by path pattern
under `/kaggle/input`, so the mirror needs no code change.

| Input | Kaggle variation (version 1) | Files and size |
|---|---|---|
| Base model, RadixArk's NVFP4 export of Qwen3.8-Flash-Next | [`carriem7/qwen3-8-flash-next-nvfp4-mirror`, `Transformers/hf-7b719225242a`](https://www.kaggle.com/models/carriem7/qwen3-8-flash-next-nvfp4-mirror/Transformers/hf-7b719225242a/1) | 419 files, 135,253,622,894 B |
| Routed experts re-rounded in NVFP4 (QG) | [`carriem7/qwen3-8-flash-next-nvfp4-qg1`, `Transformers/agi2-qg1-8352943a271b`](https://www.kaggle.com/models/carriem7/qwen3-8-flash-next-nvfp4-qg1/Transformers/agi2-qg1-8352943a271b/1) | 196 files, 67,987,306,461 B |
| Fine-tuned DFlash2 speculative draft (h1) | [`carriem7/qwen3-8-flash-next-dflash2`, `Transformers/agi2-h1-s1500-9a064a9068f9`](https://www.kaggle.com/models/carriem7/qwen3-8-flash-next-dflash2/Transformers/agi2-h1-s1500-9a064a9068f9/1) | 6 files, 1,206,438,408 B |

Each notebook builds `/tmp/model_qg` from the base checkpoint with the 192 routed-expert shards replaced. The
page-cache warm-up skips the original expert shards. The server loads the assembled checkpoint and the draft.

The base model and the QG shards use the Qwen Community License 1.0. The h1 draft uses Apache-2.0, as does its
base model. [Third-party notices](../THIRD_PARTY_NOTICES.md).

## Base checkpoint

The mirror is RadixArk's [Qwen3.8-Flash-Next NVFP4 export](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4/tree/7b719225242aacd3dbd3f9407468c2ee9a9d2594).
Two file inventories are kept here:
- [Hugging Face's manifest](models/base.hf-7b719225242a.hub-manifest.json) of that revision: 419 files, with SHA-256
  hashes for the 208 large ones;
- [the Kaggle listing](models/base.hf-7b719225242a.v1.files.json) of the copy the scored runs mounted: the same 419
  names and sizes. It names that original, non-public model; the public mirror holds the same files.

## Routed experts

Of the 24,576 experts in the distributed overlay, 18,329 use GPTQ and 6,247 use scale-only rounding. Each final
loads 18,432 experts (384 × 48 layers), selected by its own keep list. [Expert fitting guide](../training/qg_experts/README.md).

The shards were staged through a private Hugging Face repository before Kaggle. Three records tie them together:
- the 192 [build hashes](../training/qg_experts/hashes/qg_volume.sha256);
- the staging repository's [file tree](../training/qg_experts/hashes/hf_tree.json), whose shard hashes match them;
- the [Kaggle listing](models/qg1.agi2-qg1-8352943a271b.v1.files.json), whose 196 names and sizes, including
  `MANIFEST.sha256`, match that tree.

The Kaggle model also ships `MANIFEST.sha256`, to check a download against.

## Speculative draft

h1 starts from [incoai's DFlash2 draft](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2/tree/a22a1744e85c)
and fine-tunes 35 tensors. The exported `model.safetensors` has SHA-256
`9a064a9068f9f539d2b1967c53799a5b26bb34aae0847895e7e71279f188541c`. Its 35 trained tensors match step 1500.
Use [`check_export.py`](../training/draft_h1/check_export.py) to verify the hash of a downloaded copy.
[Draft fitting guide](../training/draft_h1/README.md).

## Embedded files

Each notebook contains two gzip- and base64-encoded files, its own keep list (`fnkeep`) and the common draft
candidate vocabulary (`fnvocab`). The cells write them to `/tmp` and check their SHA-256 hashes.

| File | Used by | SHA-256 prefix | Rebuild |
|---|---|---|---|
| `keep_384_code_bal4.json`, routing-count expert list | 57785004 | `a4a75c4eb06b` | [Expert counts](../training/keep_census/README.md) |
| `keep_384_c512.json`, output-error expert list, written to the same `/tmp/keep_384_code_bal4.json` path | 57853136 | `ccaac87ab41f` | [Expert selection](../training/keep_c512/README.md) |
| `draft_vocab_32k.json`, draft candidate vocabulary | both | `f72ceca4e382` | [Draft vocabulary](../training/draft_vocab/README.md) |

## BF16 weights for refitting

The reference is [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/tree/de4b8e4d43b9).
Its 96 routed-expert shards occupy about 236 GiB. The end-to-end BF16 comparison needs the full 131-shard
checkpoint.
