"""Code-generation probe for a served Flash-Next: does the server still write valid Python?

Round 1023_0833 (Flash-Next on 4 x L4, K = 320 pruned experts) showed 69 % of the harness's tool calls failing with
SyntaxError, against 0 of 458 on the G4's full model. The pruning census (fn_bringup P10) holds only plain-text puzzles
with no code. This sends ARC-AGI-2 TRAINING puzzles (never evaluation or bench tasks) with a request for a Python
`transform(grid)`, thinking off, at the harness's sampling (T 1.0, top-p 0.95, top-k 20). Each answer's first python
block goes through compile(). It prints the SyntaxError, no-code and truncated shares, so two servers (pruned and
unpruned) compare directly.

    python -m agi2_modal_harness.comm.fn_l4_code_probe [--base http://127.0.0.1:30000] [--n 24] [--conc 8]
        [--max-tokens 1500] [--out probe.json]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

DATA = os.path.join("data", "original_competition_data", "arc-agi_training_challenges.json")
BLOCK = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S)


def grid(g):
    return "\n".join(" ".join(str(c) for c in row) for row in g)


def prompts(n: int) -> list[tuple[str, str]]:
    tasks = json.load(open(DATA, encoding="utf-8"))
    ids = sorted(tasks)
    out = []
    for tid in ids[7::max(1, len(ids) // (n * 2))]:     # an offset from the census's own picks (ids[::step])
        t = tasks[tid]
        text = ("Here is a puzzle. Each example maps an input grid to an output grid.\n" + "".join(
            f"Example {i + 1} input:\n{grid(p['input'])}\nExample {i + 1} output:\n{grid(p['output'])}\n"
            for i, p in enumerate(t["train"])) +
            "Write a Python function `transform(grid)` that takes the input grid as a list of lists of ints and "
            "returns the output grid. Use numpy if it helps. Reply with one ```python code block and nothing else.")
        if len(text) > 12000:
            continue
        out.append((tid, text))
        if len(out) >= n:
            break
    return out


def ask(base: str, model: str, text: str, max_tokens: int) -> dict:
    body = {"model": model, "messages": [{"role": "user", "content": text}], "temperature": 1.0, "top_p": 0.95,
            "top_k": 20, "max_tokens": max_tokens, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=900) as r:
        d = json.loads(r.read())
    ch = d["choices"][0]
    return {"content": ch["message"].get("content") or "", "finish": ch.get("finish_reason"),
            "tokens": d.get("usage", {}).get("completion_tokens"), "s": round(time.time() - t0, 1)}


REPEAT_WORD = re.compile(r"\b([A-Za-z_]\w*)\b(?:\W{1,3}\1\b){7,}")   # `def def def ...`, `class class ...`
GRIDLIKE = re.compile(r"^[\W\d_]*$")                                    # `0 0 0 0`, `# 8 8 8 |` (grid rows)


def collapsed(text: str) -> bool:
    """Degenerate repetition: one word (with a letter) 8+ times in a row, or one non-blank line that is not a grid row
    6+ times in a row. Identical grid rows are normal in a model's comments."""
    if REPEAT_WORD.search(text):
        return True
    run, prev = 0, None
    for line in (x.strip() for x in text.splitlines()):
        if not line or GRIDLIKE.match(line):
            continue
        run = run + 1 if line == prev else 1
        prev = line
        if run >= 6:
            return True
    return False


def judge(res: dict) -> str:
    if collapsed(res["content"]):
        return "collapsed"
    m = BLOCK.search(res["content"])
    if not m:
        return "truncated" if res["finish"] == "length" else "no_code"
    try:
        compile(m.group(1), "<probe>", "exec")
    except SyntaxError:
        return "syntax_error"
    return "ok" if "def transform" in m.group(1) else "no_transform"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:30000")
    ap.add_argument("--model", default="qwen3.8-flash-next")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--conc", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=1500)
    ap.add_argument("--out", default="")
    ap.add_argument("--rescore", nargs="*", default=[], help="judge saved --out files again (no server needed)")
    a = ap.parse_args()
    kinds = ("ok", "collapsed", "syntax_error", "no_code", "truncated", "no_transform")
    if a.rescore:
        for path in a.rescore:
            saved = json.load(open(path, encoding="utf-8"))
            v = [judge(r) for r in saved]
            toks = sorted(r.get("tokens") or 0 for r in saved)
            print(f"[code-probe rescore] {path}: " + ", ".join(f"{k} {v.count(k)}" for k in kinds) +
                  f" | collapsed share {v.count('collapsed') / max(1, len(v)):.0%} | tokens median {toks[len(toks) // 2]}")
        return 0
    ps = prompts(a.n)
    t0 = time.time()
    with ThreadPoolExecutor(a.conc) as ex:
        results = list(ex.map(lambda p: ask(a.base, a.model, p[1], a.max_tokens), ps))
    verdicts = [judge(r) for r in results]
    counts = {k: verdicts.count(k) for k in kinds}
    toks = sorted(r["tokens"] or 0 for r in results)
    print(f"[code-probe] {len(ps)} training puzzles in {time.time() - t0:.0f} s: " +
          ", ".join(f"{k} {v}" for k, v in counts.items()) +
          f" | collapsed share {counts['collapsed'] / max(1, len(ps)):.0%} | tokens median {toks[len(toks) // 2]}")
    for (tid, _), r, v in zip(ps, results, verdicts):
        if v != "ok":
            print(f"  {tid}: {v}, {r['tokens']} tokens, finish {r['finish']}: {r['content'][-160:]!r}")
    if a.out:
        json.dump([dict(task=tid, verdict=v, **r) for (tid, _), r, v in zip(ps, results, verdicts)],
                  open(a.out, "w", encoding="utf-8"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
