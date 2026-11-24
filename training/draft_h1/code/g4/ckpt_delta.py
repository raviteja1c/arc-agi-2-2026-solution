"""Lossless compact transfer of a fine-tuned DFlash2 checkpoint (the trainer's 35 trained bf16 tensors) between
Colab runtimes through the laptop, whose tunnel runs at ~1.2 MB/s. A fine-tune moves most bf16 weights by zero or a
few ulps, so the XOR of the trained and stock bit patterns is mostly zero bits and compresses well.
  encode: python3 ckpt_delta.py encode <stock model.safetensors> <trained .safetensors> <out.delta>
  decode: python3 ckpt_delta.py decode <stock model.safetensors> <in.delta> <out trained .safetensors>
decode checks every tensor's sha256 against the one recorded at encode (fail closed).
Needs numpy only (reads and writes the safetensors format directly; no torch), so any Python with numpy runs it.
"""
import hashlib
import json
import lzma
import struct
import sys
import time

import numpy as np


def load_raw(path):
    """name -> (shape, dtype string, uint16 array) for bf16 tensors, read from the file's header directly."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        base = 8 + n
        out = {}
        for k, v in header.items():
            if k == "__metadata__":
                continue
            s, e = v["data_offsets"]
            f.seek(base + s)
            buf = f.read(e - s)
            out[k] = (v["shape"], v["dtype"], np.frombuffer(buf, dtype=np.uint16 if v["dtype"] == "BF16" else np.uint8))
    return out


def encode(stock_path, trained_path, out_path):
    t0 = time.time()
    stock = load_raw(stock_path)
    trained = load_raw(trained_path)
    meta, blobs = [], []
    changed = total = 0
    for k in sorted(trained):
        shape, dt, x = trained[k]
        if dt != "BF16" or k not in stock or stock[k][1] != "BF16" or stock[k][0] != shape:
            raise SystemExit(f"fail closed: {k} {dt} {shape} vs stock {stock.get(k, (None, None))[:2]}")
        d = np.bitwise_xor(x, stock[k][2])
        changed += int((d != 0).sum())
        total += d.size
        meta.append(dict(name=k, shape=shape, sha=hashlib.sha256(x.tobytes()).hexdigest(), n=int(d.size)))
        blobs.append(d.tobytes())
    payload = lzma.compress(b"".join(blobs), preset=1)
    head = json.dumps(meta).encode()
    with open(out_path, "wb") as f:
        f.write(struct.pack("<Q", len(head)))
        f.write(head)
        f.write(payload)
    print(f"encoded {len(meta)} tensors, {total:,} values, {changed / total:.4f} changed; "
          f"{total * 2 / 1e6:.0f} MB -> {(len(payload) + len(head)) / 1e6:.1f} MB in {time.time() - t0:.0f} s")


def decode_np(stock_path, delta_path, out_path):
    t0 = time.time()
    stock = load_raw(stock_path)
    with open(delta_path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        meta = json.loads(f.read(n))
        raw = lzma.decompress(f.read())
    off = 0
    header, chunks, pos = {}, [], 0
    for m in meta:
        d = np.frombuffer(raw, dtype=np.uint16, count=m["n"], offset=off)
        off += 2 * m["n"]
        x = np.bitwise_xor(d, stock[m["name"]][2])
        if hashlib.sha256(x.tobytes()).hexdigest() != m["sha"]:
            raise SystemExit(f"fail closed: {m['name']} sha256 mismatch after decode")
        b = x.tobytes()
        header[m["name"]] = dict(dtype="BF16", shape=m["shape"], data_offsets=[pos, pos + len(b)])
        chunks.append(b)
        pos += len(b)
    header["__metadata__"] = {"format": "pt"}
    hb = json.dumps(header).encode()
    hb += b" " * ((8 - len(hb) % 8) % 8)
    with open(out_path, "wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        for b in chunks:
            f.write(b)
    print(f"decoded {len(meta)} tensors, every sha256 matches, in {time.time() - t0:.0f} s -> {out_path}")


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "encode":
        encode(*sys.argv[2:5])
    elif mode == "decode":
        decode_np(*sys.argv[2:5])
    else:
        raise SystemExit(__doc__)
