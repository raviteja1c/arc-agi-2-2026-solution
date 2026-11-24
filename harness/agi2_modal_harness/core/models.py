"""Model weights on the models Volume: registry, download with progress, manifest.

Each (repo, revision) lands in its own directory, written to ``<dir>.building``
first and renamed only after every file is present with the size the Hub
reports. ``MANIFEST.json`` beside the weights records what was downloaded, so
a worker can refuse to boot on a partial or mismatched snapshot.
"""

from __future__ import annotations

import json
import os
import shutil
import threading
import time
from pathlib import Path

# Pinned revisions: never serve "main", or a silent upstream push changes the model mid-study.
MODELS: dict[str, dict] = {
    "qwen3.8-27b": {"repo": "Qwen/Qwen3.8-27B", "revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"},
    "qwen3.8-27b-fp8": {"repo": "Qwen/Qwen3.8-27B-FP8", "revision": "017b9c7af6b5689d5dd426a76e0bc077eb5ca20a"},
    # 4-bit weights (mega lever M16, the PTX audit's largest dE): RedHatAI's llm-compressor W4A16, symmetric int4 per
    # 128 along K, compressed-tensors pack-quantized (Marlin on sm_89); Apache-2.0, base Qwen/Qwen3.8-27B. For the
    # drift gate and a served arm first; not a submission model until it passes them and a task-accuracy run.
    "qwen3.8-27b-int4-redhat": {"repo": "RedHatAI/Qwen3.8-27B-INT4", "revision": "91bd022d5b49442a868bc35008f6c21e1860edfa"},
    # BAAI's AREX-2 (2026-10-27 study, docs/reports/1027_2307_arex2_review.md): a full fine-tune of Qwen3.8-27B (same
    # architecture, chat template and tokenizer), Apache-2.0, BF16 only, shipped without the MTP head. Experiment-only.
    "arex-2": {"repo": "BAAI/AREX-2", "revision": "d4e3502f92d9e889031c2d04e387ab2eb3520268"},
    # Its FP8 copy in Qwen3.8-27B-FP8's own format (block 128 x 128 e4m3, the same tensors quantized), so submission
    # 11's 4 x L4 server serves it unchanged. Built on the Volume by comm/fp8_block_quant.py, never downloaded.
    "arex-2-fp8": {"repo": "agi2/AREX-2-FP8", "revision": "d4e3502f92d9e889031c2d04e387ab2eb3520268",
                   "derived_from": "arex-2"},
    # Speculative-decoding drafts (the in-checkpoint MTP head needs no entry: it ships inside both targets).
    # DFlash2: Apache-2.0, trained against Qwen/Qwen3.8-27B. DSpark: license "other" -- research use until
    # its terms are read; trained against the NVFP4 export.
    "dflash2-qwen3.8-27b": {"repo": "incoai/Qwen3.8-27B-DFlash2", "revision": "015e795645c74b1a0eeef3b570031fb62e769bc5"},
    "dspark-qwen3.8-27b": {"repo": "RadixArk/Qwen3.8-27B-DSpark", "revision": "b9a5dbdf03bc999c6c73c426b19c2d9041cea393"},
    # Our own DFlash2 fine-tunes of incoai's draft (workloads/draft_train): written to the models Volume by that
    # workload's export (incoai's serving files + our weights + MANIFEST), never downloaded. Revision = training run.
    "dflash2-agi2-poolb-pilot": {"repo": "agi2/dflash2-qwen3.8-27b-agi2",
                                 "revision": "1007_1654_draft-poolb-pilot-b200x4"},
    # The recipe smoke (LoRA, frozen conv/selector, LK-lambda vs the processed target, AUF): docs/reports/
    # 1007_1922_draft_training_recipe.md. Exists once 1007_2019's export completes (1007_1949 OOMed and 1007_2004
    # failed its checkpoint save; neither exported).
    "dflash2-agi2-e2e-smoke": {"repo": "agi2/dflash2-qwen3.8-27b-agi2",
                               "revision": "1007_2019_draft-e2e-smoke-h100x4"},
    # Derived variant: the FP8 checkpoint plus calibrated static FP8 KV scales (core/kv_scales/*.json), which the
    # FP8 export lacks (SGLang then runs FP8 KV at scale 1.0). Built on the Volume by `build-variant`, not downloaded.
    "qwen3.8-27b-fp8-kvs": {"repo": "Qwen/Qwen3.8-27B-FP8", "revision": "017b9c7af6b5689d5dd426a76e0bc077eb5ca20a",
                            "derived_from": "qwen3.8-27b-fp8", "kv_scales": "qwen3.8-27b-fp8.json"},
    # NVFP4 (Blackwell only) for the B200 study, docs/reports/1007_1200_b200_plan.md. Both are ModelOpt
    # MIXED_PRECISION: FP8 per-tensor attention/GDN projections, NVFP4 W4A4 MLP, static activation scales.
    # NVIDIA's also has an NVFP4 lm_head and declares no KV scheme (pass --kv-cache-dtype fp8_e4m3).
    "qwen3.8-27b-nvfp4-nvidia": {"repo": "nvidia/Qwen3.8-27B-NVFP4",
                                 "revision": "482ca0f3832238542f8f5295dde86b5f22711d80"},
    "qwen3.8-27b-nvfp4-radixark-bf16head": {"repo": "RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead",
                                            "revision": "009632fef96dd349150baa780c984e62e70e91fe"},
    # RadixArk's (the SGLang team's) FP4-head export, SGLang's cookbook default: same ModelOpt MIXED_PRECISION map as
    # NVIDIA's (208 FP8 + 193 NVFP4 incl. lm_head), max calibration on cnn_dailymail, declares FP8 KV without scales.
    # Revision 319f741 restores the tree of 554ebba (the NVFP4 head) after the 08-20 BF16-head detour.
    "qwen3.8-27b-nvfp4-radixark": {"repo": "RadixArk/Qwen3.8-27B-NVFP4",
                                   "revision": "319f741cce68d7914884900c138a1fbb70a42f30"},
    # RedHat's v2 (2026-09-30; Apache-2.0 card): compressed-tensors mixed-precision. MLP layers 0-55 NVFP4 W4A4,
    # attention/GDN projections and MLP 56-63 FP8 per-channel with dynamic per-token activations, BF16 lm_head,
    # calibrated FP8 KV scales (docs/reports/1021_1859_nvfp4_g4_survey/agents/N1_checkpoints/REPORT.txt §1.2).
    "qwen3.8-27b-nvfp4-redhat": {"repo": "RedHatAI/Qwen3.8-27B-NVFP4",
                                 "revision": "2889ddeeda282f17fc9e159a42125e535159a5e5"},
    # ParoQuant / SLQ-EAR experiments on Colab G4 (docs/reports/1022_1220_paroquant_ear_experiments). Experiment-only
    # aliases: nothing on the Kaggle or Modal path names them.
    # z-lab's ParoQuant W4A16 (int4 g128 + zero points, AWQ packing, 8 learned Givens rotations per 128-channel group,
    # applied online to activations; quant_method "paroquant", which stock SGLang does not know). MIT code, Apache-2.0
    # weights. Unquantized tensors are F16 copies, not all lossless (T4, 1022_1220 README).
    "qwen3.8-27b-paro": {"repo": "z-lab/Qwen3.8-27B-PARO", "revision": "252f329a8a75ea36c563a603653e050c1d5268c3"},
    # Its dense equivalent W_eff = inv_rot(dequant(Q)) * channel_scales for the 400 linears, everything else from the
    # official BF16 checkpoint; built on the box by comm/paro_weff.py, never downloaded.
    "qwen3.8-27b-paro-weff": {"repo": "agi2/Qwen3.8-27B-PARO-weff",
                              "revision": "252f329a8a75ea36c563a603653e050c1d5268c3",
                              "derived_from": "qwen3.8-27b-paro"},
    # Qwen3.8-Flash-Next on Colab G4 (docs/reports/1022_1339_flashnext_g4). Experiment-only aliases: nothing on the
    # Kaggle or Modal path names them. RadixArk's (the SGLang team's) NVFP4 export: 176B total / 6B active GDN + QSA
    # MoE, NVFP4 W4A4 on the routed experts only, the 47.7 GiB PLE n-gram table FP8, 135.3 GB; licence "other" =
    # the base model's Qwen Community License 1.0 (not OSI). Its DFlash2 draft: incoai, Apache-2.0.
    "qwen3.8-flash-next-nvfp4-radixark": {"repo": "RadixArk/Qwen3.8-Flash-Next-NVFP4",
                                          "revision": "7b719225242aacd3dbd3f9407468c2ee9a9d2594"},
    "dflash2-qwen3.8-flash-next": {"repo": "incoai/Qwen3.8-Flash-Next-DFlash2",
                                   "revision": "a22a1744e85c17364dcc54a6a420af7135d4c7f6"},
    # Our fine-tune of that draft (dflash2 session, 2026-10-25: run 1025_1926, h1 step 1500; 199 harness conversations on
    # ARC-AGI-2 training tasks only, lr 1e-4, 10 epochs, selector/convs/norms frozen). +2.3 % accepted tokens a step at
    # block 4 on the 24 eval-parent gate. Only model.safetensors differs (35 of 81 tensors). On the models Volume as
    # agi2__dflash2-qwen3.8-flash-next-agi2__9a064a9068f9 (with AGI2_PROVENANCE.md); on Kaggle as variation
    # carriem7/qwen3-8-flash-next-dflash2/transformers/agi2-h1-s1500-9a064a9068f9 (sub 14). Mirrored 2026-10-30 to the
    # private HF repo ravitejac/dflash2-qwen3.8-flash-next-h1 @ bf724c20b8b0 (Hub LFS sha256 = 9a064a9068f9; the common
    # store every box pulls from, memory hf-common-store).
    "dflash2-qwen3.8-flash-next-agi2-h1": {"repo": "agi2/dflash2-qwen3.8-flash-next-agi2",
                                           "revision": "9a064a9068f9f539d2b1967c53799a5b26bb34aae0847895e7e71279f188541c"},
    # NVIDIA's QAD export (experiment-only, 2026-10-23, the user's pick for 4 x L4): ModelOpt 0.47.0 MIXED_PRECISION.
    # Routed experts NVFP4 with block scales in a lossless compressed format (`weight_scale.nvfp4_csf_fixed` +
    # `_exceptions`, not yet in upstream vLLM); attention / GDN / indexer / shared expert MXFP8; the PLE n-gram table
    # NVFP4 (128 shards); vision MXFP8 attention + NVFP4 MLP; lm_head BF16. ~96 GiB. Licence: NVIDIA Open Model
    # Agreement over the base's Qwen Community License 1.0 (both non-OSI; check before any submission).
    "qwen3.8-flash-next-fp4-local-nvidia": {"repo": "nvidia/Qwen3.8-Flash-Next-FP4-Local",
                                            "revision": "b9ce18a6887e18c089fdea8509a33e63429a3300"},
    # The same weights in RadixArk's layout, so the fork serves them unchanged: csf scales decoded (exact), MXFP8
    # dense -> BF16 (exact), NVFP4 vision / PLE -> BF16 (one rounding), config from RadixArk. Built on the box by
    # comm/fn_l4_nv_convert.py (1023_1411), never downloaded.
    "qwen3.8-flash-next-fp4-local-nvidia-conv": {"repo": "agi2/Qwen3.8-Flash-Next-FP4-Local-conv",
                                                 "revision": "1023_1411_conv",
                                                 "derived_from": "qwen3.8-flash-next-fp4-local-nvidia"},
    # BottleCapAI's ThinkingCap fine-tune of Flash-Next (fewer thinking tokens; experiment-only, 2026-10-28,
    # runs/plans/1027_thinkingcap). NVFP4 W4A16 routed experts in compressed-tensors format, 131.6 GB; licence: the
    # base's Qwen Community License 1.0. The router, embeddings, lm_head, n-gram table and vision tower are
    # bit-identical to RadixArk's; the routed and shared experts and (slightly) attention carry the fine-tune.
    "qwen3.8-flash-next-thinkingcap-nvfp4": {"repo": "bottlecapai/ThinkingCap-Qwen3.8-Flash-Next-NVFP4",
                                             "revision": "df3922b21fc28a8604cc9b31451495690d155194"},
    # The same weights in RadixArk's layout, so the fork serves them through the same Marlin path: weight_packed ->
    # weight, weight_scale_2 = 1 / weight_global_scale, RadixArk's input scales added, mtp.* dropped. Built on the box
    # by runs/plans/1027_thinkingcap/convert_tc.py, never downloaded.
    "qwen3.8-flash-next-thinkingcap-nvfp4-conv": {"repo": "agi2/ThinkingCap-Qwen3.8-Flash-Next-NVFP4-conv",
                                                  "revision": "1027_tc_conv",
                                                  "derived_from": "qwen3.8-flash-next-thinkingcap-nvfp4"},
    # Ornith-1.5-122B pilot (experiment-only, 2026-10-29, runs/plans/1029_ornith_pilot; docs/reports/
    # 1028_2255_ornith_15_122b_vs_stack). Qwen3_5MoeForConditionalGeneration (Qwen3.5-122B-A10B geometry), official
    # ModelOpt 0.45.0 MIXED_PRECISION: FP8 attention/GDN projections, W4A16_NVFP4 routed + shared experts and lm_head,
    # BF16 rest (MTP and vision included), 77.6 GB. Licence MIT (card metadata + vendor post; no LICENSE file in the repo).
    "ornith-1.5-122b-nvfp4": {"repo": "ornith-ai/Ornith-1.5-122B-NVFP4",
                              "revision": "f049036f794622db10bb69855bf9819207334626"},
    # Its DFlash2 draft (incoai, Apache-2.0, base_model the 122B): 6 layers, block 16, separate mask_embedding.
    "dflash2-ornith-1.5-122b": {"repo": "incoai/Ornith-1.5-122B-DFlash2",
                                "revision": "9baefc1bec85c08193ab442e0ca5f7057669b61d"},
    # Nanbeige4.3-33B pilot (experiment-only, 2026-10-29, runs/plans/1029_nanbeige_pilot): the official BF16 weights
    # (LlamaForCausalLM, dense 33.1B, Apache-2.0; weights unchanged since the 10-24 upload, c1093c50 == df919e1e).
    "nanbeige4.3-33b": {"repo": "Nanbeige/Nanbeige4.3-33B", "revision": "df919e1e2cf077a22204c78b6c826bf3db18bc08"},
    # ProCreations' DFlash2 draft for it (Apache-2.0, base revision c1093c50): 3 layers, block 8, taps [1, 25, 49],
    # mask token 0, shares the target's embedding and head; measured on stock SGLang 0.5.21 on an RTX PRO 6000.
    # ProCreations' FP8 build (Apache-2.0, base c1093c50; the user's pick 10-29 for one RTX PRO 6000): llm-compressor
    # FP8_BLOCK W8A8 E4M3 128x128 weights, dynamic token/group-128 activations, lm_head and embeddings BF16, 35.5 GB.
    # Its card: native KL 0.0043 vs BF16; coding 82/90 vs BF16 86/90 (inconclusive).
    "nanbeige4.3-33b-fp8-procreations": {"repo": "ProCreations/Nanbeige4.3-33B-FP8",
                                         "revision": "5dac23ce7c768af188e14da1f55dae6440928161"},
    "dflash2-nanbeige4.3-33b-procreations": {"repo": "ProCreations/Nanbeige4.3-33B-DFlash2",
                                             "revision": "22a73c4d099e61a34b6a26251b0a3a98a7b41746"},
}

KV_SCALES_DIR = Path(__file__).parent / "kv_scales"


def model_dir_name(repo: str, revision: str) -> str:
    return f"{repo.replace('/', '__')}__{revision[:12]}"


def _kv_scales_doc(alias: str) -> dict:
    return json.loads((KV_SCALES_DIR / MODELS[alias]["kv_scales"]).read_text(encoding="utf-8"))


def model_path(models_root: str | Path, alias: str) -> Path:
    m = MODELS[alias]
    name = model_dir_name(m["repo"], m["revision"])
    if m.get("kv_scales"):  # a scale-set edit makes a new directory, never a silent overwrite
        import hashlib

        digest = hashlib.sha256(json.dumps(_kv_scales_doc(alias)["scales"], sort_keys=True).encode()).hexdigest()
        name += f"__kvs-{digest[:8]}"
    return Path(models_root) / name


def kv_scales_safetensors(scales: dict[str, float]) -> bytes:
    """A safetensors file of F32 scalars (the k_scale/v_scale layout SGLang's Qwen3.5 loader maps onto attention
    layers), written directly: 8-byte header length, JSON header, data."""
    import struct

    header: dict = {}
    data = b""
    for name, value in scales.items():
        header[name] = {"dtype": "F32", "shape": [], "data_offsets": [len(data), len(data) + 4]}
        data += struct.pack("<f", float(value))
    header["__metadata__"] = {"format": "pt"}
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * ((8 - len(raw) % 8) % 8)
    return struct.pack("<Q", len(raw)) + raw + data


def build_kv_scale_variant(alias: str, volume, log=print) -> dict:
    """Laptop-side: server-side copy of the base checkpoint on the models Volume, then add kv_scales.safetensors
    and an index/manifest that reference it. Idempotent: a complete variant is left alone."""
    import io

    m = MODELS[alias]
    base_dir, dst_dir = model_path("", m["derived_from"]).name, model_path("", alias).name

    def read_json(path: str) -> dict | None:
        try:
            return json.loads(b"".join(volume.read_file(path)))
        except Exception:  # noqa: BLE001 - missing file
            return None

    existing = read_json(f"/{dst_dir}/MANIFEST.json")
    if existing and existing.get("complete") and existing.get("kv_scales"):
        log(f"[VARIANT] {alias} already built at /{dst_dir}")
        return existing
    base_manifest = read_json(f"/{base_dir}/MANIFEST.json")
    if not base_manifest or not base_manifest.get("complete"):
        raise RuntimeError(f"base {m['derived_from']} is not downloaded at /{base_dir}")
    doc = _kv_scales_doc(alias)
    index = read_json(f"/{base_dir}/model.safetensors.index.json")
    if not index:
        raise RuntimeError(f"/{base_dir}/model.safetensors.index.json missing")
    clash = [n for n in doc["scales"] if n in index["weight_map"]]
    if clash:
        raise RuntimeError(f"checkpoint already has KV scale tensors: {clash[:3]}")
    index["weight_map"].update({n: "kv_scales.safetensors" for n in doc["scales"]})
    # copy_files follows cp semantics (the destination directory must exist), so the scale shard goes first.
    with volume.batch_upload(force=True) as batch:
        batch.put_file(io.BytesIO(kv_scales_safetensors(doc["scales"])), f"/{dst_dir}/kv_scales.safetensors")
    rewritten = {"MANIFEST.json", "model.safetensors.index.json"}
    entries = [e for e in volume.listdir(f"/{base_dir}") if e.path.rsplit("/", 1)[-1] not in rewritten]
    t0 = time.monotonic()
    log(f"[VARIANT] copying {len(entries)} entries /{base_dir} -> /{dst_dir} on the Volume (server-side)")
    for i in range(0, len(entries), 16):
        batch_entries = entries[i:i + 16]
        volume.copy_files(["/" + e.path.lstrip("/") for e in batch_entries], f"/{dst_dir}",
                          recursive=any(getattr(e.type, "name", str(e.type)).upper().endswith("DIRECTORY")
                                        for e in batch_entries))
        log(f"[VARIANT]   {min(i + 16, len(entries))}/{len(entries)} entries, {time.monotonic() - t0:.0f}s")
    manifest = {**base_manifest, "alias": alias, "derived_from": m["derived_from"], "complete": True,
                "kv_scales": {k: doc[k] for k in ("method", "sources", "margin")},
                "kv_scales_file": m["kv_scales"], "built_at": time.time()}
    with volume.batch_upload(force=True) as batch:  # the manifest last: only now is the variant complete
        batch.put_file(io.BytesIO(json.dumps(index, indent=2).encode()), f"/{dst_dir}/model.safetensors.index.json")
        batch.put_file(io.BytesIO(json.dumps(manifest, indent=2).encode()), f"/{dst_dir}/MANIFEST.json")
    log(f"[VARIANT] {alias}: {len(doc['scales'])} KV scale tensors added; manifest complete")
    return manifest


def read_manifest(path: Path) -> dict | None:
    f = path / "MANIFEST.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None


def _dir_bytes(path: Path) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return total


def download_snapshot(alias: str, models_root: str | Path, log=print, progress_every_s: float = 15) -> dict:
    """Download one registered model into the models root; idempotent. Runs inside a container."""
    from huggingface_hub import HfApi, snapshot_download

    m = MODELS[alias]
    if m.get("derived_from"):
        raise ValueError(f"{alias} is a derived variant: build it with `build-variant {alias}`, not download-model")
    final = model_path(models_root, alias)
    existing = read_manifest(final)
    if existing and existing.get("complete"):
        log(f"[DL] {alias} already present at {final} ({existing['total_bytes'] / 1e9:.2f} GB); skipping")
        return existing

    token = os.environ.get("HF_TOKEN")
    info = HfApi(token=token).model_info(m["repo"], revision=m["revision"], files_metadata=True)
    files = {s.rfilename: s.size for s in info.siblings}
    expected = sum(v or 0 for v in files.values())
    building = final.with_name(final.name + ".building")
    building.mkdir(parents=True, exist_ok=True)
    log(f"[DL] {m['repo']}@{m['revision'][:12]}: {len(files)} files, {expected / 1e9:.2f} GB -> {final}")

    done = threading.Event()
    t0 = time.monotonic()

    def report() -> None:
        last = 0
        while not done.wait(progress_every_s):
            got = _dir_bytes(building)
            dt = time.monotonic() - t0
            rate = (got - last) / progress_every_s
            last = got
            eta = (expected - got) / rate if rate > 0 else float("nan")
            log(f"[DL] {got / 1e9:6.2f}/{expected / 1e9:.2f} GB  {rate / 1e6:7.1f} MB/s  t={dt:.0f}s  eta={eta:.0f}s")

    threading.Thread(target=report, daemon=True).start()
    try:
        snapshot_download(repo_id=m["repo"], revision=m["revision"], local_dir=building, token=token, max_workers=16)
    finally:
        done.set()

    missing = [f for f, size in files.items() if size is not None and (building / f).stat().st_size != size]
    if missing:
        raise RuntimeError(f"size mismatch or missing after download: {missing[:10]}")
    shutil.rmtree(building / ".cache", ignore_errors=True)
    manifest = {
        "alias": alias, "repo": m["repo"], "revision": m["revision"], "files": files,
        "total_bytes": expected, "complete": True, "downloaded_at": time.time(),
        "seconds": round(time.monotonic() - t0, 1),
    }
    (building / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    if final.exists():
        shutil.rmtree(final)
    building.rename(final)
    log(f"[DL] done in {manifest['seconds']}s ({expected / 1e9 / max(manifest['seconds'], 1):.2f} GB/s avg)")
    return manifest
