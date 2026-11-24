"""How much of the generated-token mass does a frequent-token subset of the vocabulary cover? Sizes the draft's
reduced candidate vocabulary (the draft head reads the whole 248k-row lm_head every step; the target still verifies,
so a subset changes acceptance only).

Reads harness attempt files (public parents only: 8-hex ids), tokenizes every generation's reasoning, content and
tool-call arguments with the served tokenizer, builds the ranking on half the tasks and measures coverage on the
other half (by task, so a task's own tokens never rank themselves).

Run: python docs/reports/1029_0021_upstream_sweep_flashnext/draft_vocab_coverage.py <attempt dirs...>
"""
import collections
import glob
import json
import os
import re
import sys

from tokenizers import Tokenizer

TOK = "runs/plans/1027_thinkingcap/meta/radixark/tokenizer.json"
PARENT = re.compile(r"^([0-9a-f]{8})_\d+_[a-z]+\.json$")


def texts(d):
    for g in d.get("generations") or []:
        for k in ("reasoning", "content"):
            if g.get(k):
                yield g[k]
        for c in g.get("calls") or []:
            f = c.get("function") or {}
            if f.get("arguments"):
                yield f["arguments"] if isinstance(f["arguments"], str) else json.dumps(f["arguments"])


def main():
    tok = Tokenizer.from_file(TOK)
    per_task = collections.defaultdict(collections.Counter)
    files = 0
    for d in sys.argv[1:]:
        for p in glob.glob(os.path.join(d, "*.json")):
            m = PARENT.match(os.path.basename(p))
            if not m:
                continue
            try:
                j = json.load(open(p, encoding="utf-8"))
            except Exception:  # noqa: BLE001
                continue
            files += 1
            c = per_task[m.group(1)]
            for t in texts(j):
                c.update(tok.encode(t, add_special_tokens=False).ids)
    tasks = sorted(per_task)
    total = sum(sum(c.values()) for c in per_task.values())
    distinct = len(set().union(*[set(c) for c in per_task.values()]))
    print(f"{files} attempt files, {len(tasks)} parent tasks, {total:,} generated tokens, {distinct:,} distinct ids")
    for split in (0, 1):
        rank_tasks = [t for i, t in enumerate(tasks) if i % 2 == split]
        eval_tasks = [t for i, t in enumerate(tasks) if i % 2 != split]
        rank = collections.Counter()
        for t in rank_tasks:
            rank.update(per_task[t])
        ev = collections.Counter()
        for t in eval_tasks:
            ev.update(per_task[t])
        order = [i for i, _ in rank.most_common()]
        n_ev = sum(ev.values())
        line = []
        for n in (4096, 8192, 16384, 24576, 32768, 49152, 65536):
            s = set(order[:n])
            miss = sum(v for k, v in ev.items() if k not in s)
            line.append(f"top {n // 1024}k miss {miss / n_ev * 100:.3f} %")
        print(f"split {split}: rank on {len(rank_tasks)} tasks ({len(rank):,} ids), eval on {len(eval_tasks)} tasks "
              f"({n_ev:,} tokens): " + "; ".join(line))


if __name__ == "__main__":
    main()
