"""Commit or reload the cache Volume from inside a box (box_app mounts it at /vol/cache; Modal's guide
12_data-sharing-and-storage/02_volumes.md:189-203: writes are seen by other containers after a commit and their reload).

  python -m agi2_modal_harness.comm.nvfp4_vol commit|reload
"""
import sys
import time

from agi2_modal_harness.core import resources

op = sys.argv[1]
t = time.time()
vol = resources.cache_volume()
getattr(vol, op)()
print(f"[vol] {op} done in {time.time() - t:.1f} s", flush=True)
