# Expert selection by routing counts (submission 57785004)

`keep_384_code_bal4.json` selects 384 of the 512 routed experts in each of 48 MoE layers. The server reads the
list from `SGLANG_AGI2_FN_EXPERT_KEEP` with `SGLANG_AGI2_FN_EXPERT_KEEP_ANY_ORDER=1`. Each expert's position in
the order is its new ID, with 96 consecutive positions assigned to each GPU. Submission 18 embeds this list
in `fnkeep`. [Submission 20's list](../keep_c512/README.md) uses a different selection method.

## Record expert routing counts

The censuses ran on the full 512-expert model on 4 × L4 using only ARC-AGI-2 training tasks. The
[patched server](../../patches/README.md) counts routed expert selections per layer with
`--expert-distribution-recorder-mode stat` and `SGLANG_AGI2_EXPERT_COUNT_SUM=1`.

- **Prose.** One request per task for 20 tasks. The prompt presents the puzzle as text, with thinking on and
  sampling at temperature 1.0, top-p 0.95 and top-k 20. Each request generates up to 2,048 tokens.
  [Prose counts](p10_expert_counts_1023_0223.json).
- **Code.** A disjoint set of 48 tasks. The prompt asks for a `python` call with a `transform(grid)` function,
  with thinking off, four requests at a time and at most 2,048 generated tokens per request.
  [Code counts](p10_code_counts_1023_0916.json).

`p10_census` in [`fn_bringup.py`](capture/fn_bringup.py) selects tasks from the training challenge file. The
outputs are sampled, so a fresh run can give different counts. Use the [staging guide](../../harness/README.md#restore-paths-for-the-census-modules)
and [census command](../../entry_points.md#fit-and-validate) for the code pass. The prose pass used the same driver
with `--census-tasks 20 --census-code 0`; its exact command line was not recorded.

To build a list from new counts rather than replay the recorded ones, run the merge from the staged harness root:
`python -m agi2_modal_harness.comm.fn_l4_merge_census <prose counts> <code counts> <out> --keep 384 --ep 4`. It
writes `<out>/keep_384_code_bal4.json`.

## Merge and check the list

[`fn_l4_merge_census.py`](fn_l4_merge_census.py) ranks each expert by the larger of its prose and code routing
shares, then takes the top 384 per layer. [`fn_l4_balance_keep.py`](fn_l4_balance_keep.py) greedily orders them
into four groups of 96 experts with approximately balanced routing load, using the mean of the two shares.
The resulting list covers 98.5% of prose routing and 98.2% of code routing, averaged over layers.

From this directory:

    python rebuild.py

The standard-library replay uses the two recorded count files and checks the result against the list embedded
in submission 18. It reproduces SHA-256 `a4a75c4eb06b8a593b0c0a7374de17f31c2315d864db6090048ac685396dcd0b`.

On a 24-puzzle tool-call probe
([`fn_l4_code_probe.py`](../../harness/agi2_modal_harness/comm/fn_l4_code_probe.py)), the merged list and the full
model each produced 3 broken answers. The prose-only list produced 16.
