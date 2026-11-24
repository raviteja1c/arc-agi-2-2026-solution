# The draft's 32k candidate vocabulary

`draft_vocab_32k.json` lets the speculative draft score 32,768 candidate token IDs instead of all 248,320 rows.
Enable it with `SGLANG_AGI2_DRAFT_VOCAB`. The served selector chooses 16 candidates and the target still verifies
each proposal. Both submission notebooks embed the list in `fnvocab` and check its SHA-256.

A microbenchmark on one L4 timed the draft head's GEMM with a top-8 selection (the served selector takes 16): 1.65 ms
with the full vocabulary and 0.24 ms with the shortlist. Measured acceptance on 4 × L40S was 2.870 and 2.868
generated tokens per step, respectively.

## Build inputs

[`draft_vocab_build.py`](draft_vocab_build.py) re-tokenizes generated reasoning, answers and tool-call arguments,
keeps all observed token IDs plus added and special tokens, then fills to 32,768 with the lowest available base IDs.
A half-task holdout missed 0.35–0.39% of generated token occurrences.
[Coverage script](draft_vocab_coverage.py).

The build used attempt files from 13 Kaggle runs, including submissions 8–16 and intermediate tests. Across 37
tasks it counted 10,033,735 generated-token occurrences and 18,924 distinct IDs. The tasks were:

- 30 public ARC-AGI-2 evaluation tasks
- 3 public ARC-AGI-2 training tasks
- 4 private validation puzzles

The 242 public-task attempt files are withheld because they contain competition tasks. The 12 private validation
attempts are withheld. [`aux_validation_ids.json`](aux_validation_ids.json) contains the 276 vocabulary IDs
contributed only by those validation attempts. They account for 1,456 generated-token occurrences (0.0145%).
Without them, 171 of the 32,768 slots would refill from the lowest base IDs.

## Rebuild

Use `tokenizer.json` from the public base-model mirror
[`carriem7/qwen3-8-flash-next-nvfp4-mirror`](https://www.kaggle.com/models/carriem7/qwen3-8-flash-next-nvfp4-mirror/Transformers/hf-7b719225242a/1).
It is the RadixArk tokenizer at revision `7b719225242a` (12,809,320 bytes, SHA-256
`0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3`).

From the repository root, with Python's `tokenizers` package and the original public-task attempt files:

    python training/draft_vocab/rebuild.py --tokenizer <tokenizer.json> --attempts <attempt-root>

The helper checks the tokenizer hash, seeds the builder with the aggregate validation IDs and reads the public-task
attempts in their original directory layout (`RUNS` in [`rebuild.py`](rebuild.py)). This reproduces
`draft_vocab_32k.json`, SHA-256 `f72ceca4e3822fa7fdfdab6c5989cc0142e4af7fdde197cc22fc083e492de098`.
