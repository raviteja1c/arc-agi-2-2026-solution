"""Rebuild submission 20's expert list from the 48 per-layer choices.

    python remerge.py [CHOOSE_DIR]

The default input is choose/ beside this script. The training script's greedy
balancing function assigns 96 experts to each of four groups and approximately
balances measured routing load. The replay checks the result against the
published list and requires only the Python standard library.
"""
import ast
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "fn_expert_choose512.py")
EXPECTED = {"k384": ("keep_384_c512.json", "ccaac87ab41f3038aff0a983856f3ef44afb3efda1a164d1e7612224539cbaab")}
DROPS, EP = "32,32,32,16,16,16,16", 4


def balance_from_script():
    tree = ast.parse(open(SCRIPT, encoding="utf-8").read())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "balance")
    ns = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), SCRIPT, "exec"), ns)
    return ns["balance"]


def main() -> int:
    choose_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "choose")
    balance = balance_from_script()
    rs = [json.load(open(os.path.join(choose_dir, f"choose_L{layer:02d}.json"))) for layer in range(48)]
    ok = True
    for name, (fname, want) in EXPECTED.items():
        k = int(name[1:])
        layers_out = [balance(r["kept"][name], r["counts"][name], EP) for r in rs]
        assert all(len(x) == k for x in layers_out)
        data = json.dumps({"layers": layers_out, "source": f"fn_expert_choose512 {name}: greedy from all 512 on harness "
                           f"traffic (calib + late32), drops {DROPS}", "cut": 512 - k}).encode()
        got = hashlib.sha256(data).hexdigest()
        shipped = hashlib.sha256(open(os.path.join(HERE, fname), "rb").read()).hexdigest()
        same = got == want == shipped
        ok &= same
        print(f"{fname}: remerged {got[:16]}  expected {want[:16]}  shipped {shipped[:16]}  "
              f"{'IDENTICAL' if same else 'DIFFERENT'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
