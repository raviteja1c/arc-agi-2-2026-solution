"""Rebuild submission 18's expert list from the two recorded census files.

    python rebuild.py

The replay runs the merge and balance scripts with the recorded input paths
in a temporary directory and checks the published SHA-256. Requires Python
3.10 or later, using only the standard library.
"""
import hashlib
import importlib.util
import os
import shutil
import sys
import tempfile
import types

HERE = os.path.dirname(os.path.abspath(__file__))
REL = "docs/reports/1023_0106_flashnext_l4x4/census"
PROSE, CODE = "p10_expert_counts_1023_0223.json", "p10_code_counts_1023_0916.json"
EXPECTED = {
    "keep_384_code_bal4.json": "a4a75c4eb06b8a593b0c0a7374de17f31c2315d864db6090048ac685396dcd0b",  # submission 57785004
}


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    for pkg in ("agi2_modal_harness", "agi2_modal_harness.comm"):
        sys.modules.setdefault(pkg, types.ModuleType(pkg))
    load("agi2_modal_harness.comm.fn_l4_balance_keep", os.path.join(HERE, "fn_l4_balance_keep.py"))
    merge = load("fn_l4_merge_census", os.path.join(HERE, "fn_l4_merge_census.py"))
    work = tempfile.mkdtemp(prefix="keep_census_")
    try:
        os.makedirs(os.path.join(work, *REL.split("/")))
        for f in (PROSE, CODE):
            shutil.copyfile(os.path.join(HERE, f), os.path.join(work, *REL.split("/"), f))
        out = os.path.join(work, "out")
        cwd = os.getcwd()
        os.chdir(work)
        try:
            sys.argv = ["fn_l4_merge_census", f"{REL}/{PROSE}", f"{REL}/{CODE}", "out", "--keep", "384", "--ep", "4"]
            merge.main()
        finally:
            os.chdir(cwd)
        ok = True
        for name, want in EXPECTED.items():
            got = hashlib.sha256(open(os.path.join(out, name), "rb").read()).hexdigest()
            shipped = hashlib.sha256(open(os.path.join(HERE, name), "rb").read()).hexdigest()
            same = got == want == shipped
            ok &= same
            print(f"{name}: rebuilt {got[:16]}  expected {want[:16]}  shipped {shipped[:16]}  "
                  f"{'IDENTICAL' if same else 'DIFFERENT'}")
        return 0 if ok else 1
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
