"""Which histories the capture has finished (both <name>.feat.npy and <name>.teacher.npz), per split.
  python3 capture_status.py /content/probe/out /content/inputs/split.json
"""
import json
import os
import sys

out, split = sys.argv[1], json.load(open(sys.argv[2]))
files = os.listdir(out)
feat = {f[:-len(".feat.npy")] for f in files if f.endswith(".feat.npy")}
teach = {f[:-len(".teacher.npz")] for f in files if f.endswith(".teacher.npz")}
done = feat & teach
print(f"feat {len(feat)}, teacher {len(teach)}, both {len(done)}; feat without teacher: {sorted(feat - teach)}")
for k in ("train", "val"):
    print(f"{k}: {sum(n in done for n in split[k])} of {len(split[k])} done")
