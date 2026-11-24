"""Print the comma-separated layers whose BF16 expert shards (gate_up and down) are complete: the file exists and its
safetensors header covers it exactly (8 + header + the largest tensor end == file size). A partial or truncated file
fails this even if it carries the final name."""
import json
import os
import struct
import sys


def complete(path: str) -> bool:
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            if 8 + n > size:
                return False
            h = json.loads(f.read(n))
        end = max(v["data_offsets"][1] for k, v in h.items() if k != "__metadata__")
        return 8 + n + end == size
    except (OSError, ValueError, struct.error):
        return False


if __name__ == "__main__":
    d = sys.argv[1] if len(sys.argv) > 1 else "/tmp/fnbf16"
    wmap = json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"]
    ok = []
    for L in range(48):
        pre = f"model.language_model.layers.{L}.mlp.experts"
        files = {wmap[f"{pre}.gate_up_proj"], wmap[f"{pre}.down_proj"]}
        if all(complete(os.path.join(d, f)) for f in files):
            ok.append(str(L))
    print(",".join(ok))
