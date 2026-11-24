# Expert selection by output error (submission 57853136)

`keep_384_c512.json` selects 384 of the 512 routed experts per MoE layer by greedily removing experts with
the lowest estimated cost to the layer's output. The order assigns 96 experts to each of four GPUs. Submission
20's `fnkeep` cell writes the list at the server's expected path, `/tmp/keep_384_code_bal4.json`.
[Routing-count baseline](../keep_census/README.md).

## Select the experts

[`fn_expert_choose512.py`](fn_expert_choose512.py) needs BF16 expert and router weights from
[Qwen3.8-Flash-Next](../../inference/MODELS.md#bf16-weights-for-refitting), the census list and two MoE activation
captures of ARC-AGI-2 training-task histories. The captures used a 384-expert server and the
[MoE hook](../../patches/sglang/capture/moe_capture.diff).

| Phase | Histories | Role |
|---|---|---|
| `calib` | 27, 350,445 positions | Early windows from the [QG capture](../qg_experts/README.md) |
| `late32` | 16, scored at positions 16,000–31,999 | Late windows. Two histories also contributed an early calibration window. |
| `gate` | 4 disjoint held-out histories, 52,296 positions | Layer error measurement after selection |

The reference recomputes top-10 routing and output with the full BF16 weights. Candidate sets take the top ten
among their kept experts and renormalize their routing weights. The chooser scores each possible next removal
against the reference on every third row of `calib` and `late32`, normalizing squared error by each dataset's
reference output energy. If a top-ten expert is removed, the scorer includes the next-ranked kept expert.

The first five rounds remove 32, 32, 32, 16 and 16 experts to reach 384. Two further rounds of 16 also produce a
352-expert list. Each `choose/choose_LNN.json` records the two lists, routing counts and errors for one layer.
The merge greedily balances measured routing load across four groups of 96 for the 384-expert list.

From this directory, run one layer range per GPU, covering 0–47:

    python fn_expert_choose512.py --bf16 <bf16-dir> \
        --census ../keep_census/keep_384_code_bal4.json \
        --cap16 <qg-capture-dir> --cap32 <late-capture-dir> \
        --out <choice-dir> --layers 0-15

Three single-L40S machines split the layers (about 40 minutes from launch through selection). Collect all 48
outputs in `<choice-dir>` and rerun with `--merge` and the same `--bf16`, `--census` and `--out` arguments.
The script still runs its CUDA self-test in merge mode, so on a CPU use `remerge.py` (below). The captures (92.9 GiB and 118.2 GiB) must be recorded again. Follow the [late-capture staging guide](../../harness/README.md#stage-the-late-capture-machine)
and [D4](../../entry_points.md#prepare-data). Run `make_plan32.py` from a scratch copy of `capture_late32/`. It
overwrites `xq32_plan.json` and writes `xq32_histories.npz` beside itself. The original token histories are
withheld because they contain competition tasks.

## Results

Mean relative squared output error against all 512 BF16 experts. [Layer summary](summary.json).

| List | Calibration | Late 16k–32k | Held-out layer error |
|---|---|---|---|
| Census, 384 experts | 0.0376 | 0.0404 | 0.0335 |
| Output-error selection, 384 experts | 0.0119 | 0.0072 | 0.0104 |

The census list's held-out layer error was 1.87–12.24 times the selected list's error across the 48 layers.
The lists share 311–359 experts per layer (334 on average).

The end-to-end test compared two 384-expert lists with the same QG shards on 4 × L40S. It scored 49,736
positions from the four held-out histories using the top-20 next-token KL estimate against full BF16. KL fell
from 0.3404 with the census list to 0.2475 with the selected list. The [QG rounding test](../qg_experts/README.md#results)
uses the full 512 experts in the evaluated builds and a separate server configuration.

## Rebuild the list on CPU

    python remerge.py

This standard-library replay reads the 48 choices, reuses the script's balancing function and verifies
`keep_384_c512.json` byte for byte (SHA-256 `ccaac87ab41f3038aff0a983856f3ef44afb3efda1a164d1e7612224539cbaab`).
