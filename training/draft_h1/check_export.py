"""Check a downloaded h1 model against its published SHA-256.

    python check_export.py <model.safetensors> [<ckpt_step001500.delta>]

With the original step-1500 delta, also compare the hashes of the 35 trained
tensors. The delta is not included. Requires only the Python standard library.
"""
import hashlib
import json
import struct
import sys

EXPECTED = "9a064a9068f9f539d2b1967c53799a5b26bb34aae0847895e7e71279f188541c"


def tensors(path):
    out = {}
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        base = 8 + n
        for k, v in header.items():
            if k != "__metadata__":
                s, e = v["data_offsets"]
                f.seek(base + s)
                out[k] = hashlib.sha256(f.read(e - s)).hexdigest()
    return out


def main() -> int:
    if len(sys.argv) not in (2, 3):
        print(__doc__.strip(), file=sys.stderr)
        return 2
    model = sys.argv[1]
    h = hashlib.sha256()
    with open(model, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    ok = h.hexdigest() == EXPECTED
    print(f"model.safetensors sha256 {h.hexdigest()[:16]}  expected {EXPECTED[:16]}  {'OK' if ok else 'DIFFERENT'}")
    if len(sys.argv) > 2:
        with open(sys.argv[2], "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            meta = json.loads(f.read(n))
        t = tensors(model)
        eq = sum(1 for m in meta if t.get(m["name"]) == m["sha"])
        print(f"trained tensors equal to checkpoint step 1500: {eq} of {len(meta)} (file holds {len(t)} tensors)")
        ok &= eq == len(meta)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
