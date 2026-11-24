"""Rebuild the draft candidate vocabulary and verify its published SHA-256.

    python training/draft_vocab/rebuild.py --tokenizer <tokenizer.json> --attempts <attempt-root> [--out FILE]

Use the pinned RadixArk tokenizer and the public-task recordings from the
13 original runs. The recordings are withheld because they contain competition
tasks. The attempt root must contain the run paths listed in RUNS below. The
included aux_validation_ids.json contributes 276 vocabulary indices. The
builder replays the original vocabulary selection with those indices and the
public-task recordings. Requires Python with the `tokenizers` package.
"""
import argparse
import collections
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import types

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
EXPECTED = "f72ceca4e3822fa7fdfdab6c5989cc0142e4af7fdde197cc22fc083e492de098"
TOKENIZER_SHA = "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3"
RUNS = ["10/20/1020_0153_1020-sub8-qwen38-xhigh-harness-per-task",
        "10/21/1021_0158_1020-sub9-qwen38-xhigh-gate90-exact-cuts",
        "10/22/1022_0205_1022-sub10-qwen38-xhigh-kv4-all-levers",
        "10/23/1023_0035_1022-sub11-qwen38-xhigh-vision-tail",
        "10/23/1023_0221_1022-sub11-qwen38-xhigh-vision-tail",
        "10/23/1023_0302_1022-sub11-qwen38-xhigh-vision-tail",
        "10/24/1024_0308_1024-sub12-qwen38-flashnext-k384-kvpool",
        "10/24/1024_1651_1024-startup-flashnext-seeded",
        "10/25/1025_0129_1025-sub13-qwen38-flashnext-bf16head-ple",
        "10/26/1026_0202_1026-sub14-qwen38-flashnext-blocktier16-h1",
        "10/27/1027_0220_1027-sub15-qwen38-flashnext-slots24-f123-look",
        "10/28/1028_0028_1028-sub16-qwen38-flashnext-transposed-tails",
        "10/28/1028_1418_1028-sub16-harness-decode-server-06e9a82"]


def sha256(path: str) -> str:
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tokenizer", required=True, help="pinned RadixArk tokenizer.json from the public base mirror")
    ap.add_argument("--attempts", default=os.path.join(ROOT, "training", "draft_vocab", "attempts"),
                    help="root of the 13 run directories holding the original public-task attempt files")
    ap.add_argument("--out", help="output file (default: draft_vocab_32k.json in a new temporary folder)")
    a = ap.parse_args()
    if a.out is None:
        a.out = os.path.join(tempfile.mkdtemp(prefix="draft_vocab_"), "draft_vocab_32k.json")

    if sha256(a.tokenizer) != TOKENIZER_SHA:
        print(f"Tokenizer does not match the pinned RadixArk file (expected SHA-256 prefix {TOKENIZER_SHA[:12]})", file=sys.stderr)
        return 2
    dirs = [os.path.join(a.attempts, r, "output", "attempts") for r in RUNS]
    missing = [d for d in dirs if not os.path.isdir(d)]
    if missing:
        print(f"{len(missing)} attempt directories missing, first: {missing[0]} "
              f"(pass the directory containing the 13 run directories with --attempts)", file=sys.stderr)
        return 2
    with open(os.path.join(HERE, "aux_validation_ids.json"), encoding="utf-8") as fh:
        aux = json.load(fh)

    spec = importlib.util.spec_from_file_location("draft_vocab_build", os.path.join(HERE, "draft_vocab_build.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.TOK = a.tokenizer
    # the builder's only `collections` use is `seen = collections.Counter()`: start it from the aggregate ids
    mod.collections = types.SimpleNamespace(Counter=lambda: collections.Counter(aux))
    sys.argv = ["draft_vocab_build.py", a.out, "32768", *dirs]
    mod.main()

    got, shipped = sha256(a.out), sha256(os.path.join(HERE, "draft_vocab_32k.json"))
    same = got == EXPECTED == shipped
    print(f"draft_vocab_32k.json: rebuilt {got[:16]}  expected {EXPECTED[:16]}  shipped {shipped[:16]}  "
          f"{'IDENTICAL' if same else 'DIFFERENT'}")
    return 0 if same else 1


if __name__ == "__main__":
    sys.exit(main())
