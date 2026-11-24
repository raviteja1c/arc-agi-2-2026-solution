"""Save the two target tensors the DFlash2 draft trainer needs, read straight from the served checkpoint's safetensors:
the token embedding table (the draft's input embeddings: anchor token + mask tokens; DFlash's worker embeds with the
target's get_input_embeddings(), which is Qwen4-Exp's embed_tokens, with no noise scale) and the BF16 lm_head (the
draft's unary logits and the candidate top-k). If the draft repo ships mask_embedding.pt, the worker overwrites the
mask-token row with it (dflash_worker_v2._maybe_merge_trained_mask_embedding); this script applies the same override.
  /tmp/sgl/bin/python extract_target.py <model_dir> <draft_dir> <out_dir>
"""
import json
import os
import sys

import torch
from safetensors import safe_open


def find(index, suffixes, exclude=()):
    keys = [k for k in index["weight_map"] if any(k.endswith(s) for s in suffixes) and not any(e in k for e in exclude)]
    if len(keys) != 1:
        raise SystemExit(f"expected one key ending in {suffixes}, found {keys}")
    return keys[0]


def load(model_dir, index, key):
    with safe_open(os.path.join(model_dir, index["weight_map"][key]), framework="pt") as f:
        return f.get_tensor(key)


def main():
    model_dir, draft_dir, out = sys.argv[1], sys.argv[2], sys.argv[3]
    os.makedirs(out, exist_ok=True)
    index = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))
    ek = find(index, ["embed_tokens.weight"], exclude=("mtp", "visual", "per_layer", "ngram", "ple"))
    hk = find(index, ["lm_head.weight"], exclude=("mtp",))
    emb = load(model_dir, index, ek)
    head = load(model_dir, index, hk)
    dcfg = json.load(open(os.path.join(draft_dir, "config.json")))
    mask_id = int(dcfg["dflash_config"]["mask_token_id"])
    print(f"embed {ek} {tuple(emb.shape)} {emb.dtype}; lm_head {hk} {tuple(head.shape)} {head.dtype}; mask id {mask_id}")
    if emb.dtype != torch.bfloat16 or head.dtype != torch.bfloat16:
        raise SystemExit("expected bf16 embedding and lm_head (the BF16-head checkpoint)")
    me = os.path.join(draft_dir, "mask_embedding.pt")
    if os.path.exists(me):
        saved = torch.load(me, map_location="cpu", weights_only=True)
        if int(saved["mask_token_id"]) != mask_id:
            raise SystemExit("mask_embedding.pt was trained for another mask token id")
        emb[mask_id] = saved["embedding"].to(emb.dtype).view(-1)
        print(f"mask row {mask_id} overridden from mask_embedding.pt")
    else:
        print("no mask_embedding.pt in the draft repo: the target's own mask row is used, as the worker does")
    torch.save(emb.contiguous(), os.path.join(out, "target_embed.pt"))
    torch.save(head.contiguous(), os.path.join(out, "target_lm_head.pt"))
    json.dump({"embed_key": ek, "lm_head_key": hk, "mask_token_id": mask_id, "mask_override": os.path.exists(me),
               "shape": list(emb.shape)}, open(os.path.join(out, "target_tensors.json"), "w"))
    print("saved", out)


if __name__ == "__main__":
    main()
