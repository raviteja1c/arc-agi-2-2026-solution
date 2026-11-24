# agi2 DFlash2 fine-tune of incoai/Qwen3.8-Flash-Next-DFlash2 @ a22a1744: h1, step 1500

- **What it is:** incoai's DFlash2 draft for Qwen3.8-Flash-Next (Apache-2.0) with the 7 linears of each of its 5
  layers (q/k/v/o, gate/up/down; 35 tensors, 426M parameters) fine-tuned. Every other tensor (fc, hidden_norm, the
  grouped convolutions, the candidate selector, every norm) is bit-identical to the original. Same config.json and
  serving files as the original (dflash_config.block_size stays 8; served at --speculative-num-draft-tokens 4); SGLang
  loads it as the original (`--speculative-draft-model-path <this dir>`).
- **model.safetensors sha256:** 9a064a9068f9f539d2b1967c53799a5b26bb34aae0847895e7e71279f188541c.
- **Training data: ARC-AGI-2 TRAINING tasks only.** 199 Flash-Next harness conversations on the `fntrain` set (tasks
  200-399 of its seed-1025 draw; every evaluation id and every eval child banned by construction), generated on two
  Colab G4s at submission 13's numerics (Marlin W4A16, K384, T 1.0 / top-p 0.95 / top-k 20, xhigh, g1 harness).
  4.80M tokens, 3.23M generated. No evaluation task and no eval child modified any weight.
- **Teacher and features:** captured from the served stack: the draft's projected context features and the target's
  top-20 next-token logprobs at every position.
- **Recipe:** 0.9 TV (the selector's distribution against the target's processed sampling distribution) + 0.1 soft CE
  (unary), accept-until-fail, fp8 straight-through on context K/V, AdamW (0.9, 0.95) on fp32 masters, wd 0.01, clip
  1.0, lr 1e-4, warmup 5 %, cosine to 10 %, 10 epochs (1,518 steps), blocks 4 / 8 alternating; selector, convolutions,
  norms, fc and hidden_norm frozen. Checkpoint step 1500 chosen by validation on 12 held-out training tasks (block-4
  chain 1.913 -> 1.964, +2.7 %).
- **Measured** (offline probe + simulation of the served sampled chain, paired stock draft on the same G4): on 24
  eval-parent harness histories, 2.775 -> 2.840 tokens a verify step at block 4 (+2.3 %), 3.421 -> 3.516 at block 8
  (+2.8 %); T 1.0 and T 1.2 alike. Speculative decoding verifies every token, so outputs keep the target's
  distribution; only speed changes.
