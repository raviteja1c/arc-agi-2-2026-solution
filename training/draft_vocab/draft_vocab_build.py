"""Build the draft's reduced candidate vocabulary (SGLANG_AGI2_DRAFT_VOCAB) and report how it falls on the TP4 vocab
shards (the head is vocab-parallel: each rank scores its own contiguous shard).

The set: every token id generated on the public parents' harness attempts, plus every added/special token, plus the
lowest ids of the base vocabulary (byte-level BPE: low ids are the bytes and the most frequent merges) up to SIZE.

Run: python docs/reports/1029_0021_upstream_sweep_flashnext/draft_vocab_build.py <out.json> <size> <attempt dirs...>
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
VOCAB, TP = 248320, 4


def main():
    out, size, dirs = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
    tok = Tokenizer.from_file(TOK)
    seen = collections.Counter()
    for d in dirs:
        for p in glob.glob(os.path.join(d, "*.json")):
            if not PARENT.match(os.path.basename(p)):
                continue
            try:
                j = json.load(open(p, encoding="utf-8"))
            except Exception:  # noqa: BLE001
                continue
            for g in j.get("generations") or []:
                for k in ("reasoning", "content"):
                    if g.get(k):
                        seen.update(tok.encode(g[k], add_special_tokens=False).ids)
                for c in g.get("calls") or []:
                    f = c.get("function") or {}
                    a = f.get("arguments")
                    if a:
                        seen.update(tok.encode(a if isinstance(a, str) else json.dumps(a), add_special_tokens=False).ids)
    added = list(tok.get_added_tokens_decoder().keys()) if hasattr(tok, "get_added_tokens_decoder") else []
    if not added:
        added = [i for i in range(tok.get_vocab_size(with_added_tokens=True))
                 if i >= tok.get_vocab_size(with_added_tokens=False)]
    s = set(seen) | set(added)
    i = 0
    while len(s) < size and i < VOCAB:
        s.add(i)
        i += 1
    ids = sorted(x for x in s if x < VOCAB)
    shard = VOCAB // TP
    per = [sum(1 for x in ids if r * shard <= x < (r + 1) * shard) for r in range(TP)]
    json.dump(ids, open(out, "w"))
    print(f"{len(ids)} ids ({len(seen)} generated, {len(added)} added/special, base ids 0..{i - 1}); per TP4 shard "
          f"{per}; written {out}")


if __name__ == "__main__":
    main()
