"""Flash-Next bring-up on one Colab G4: boot the config's server through the shared builder, then probe it.

Run on the VM from /content/agi2 with the venv's python (the JIT environment of round.sh in the notebook):

    python fn_bringup.py --config <toml> --out /content/run/<run_id>/<arm> [--env K=V ...]

The server command is the bench config's (BenchConfig.command -> arc_eval.run_config.server_command), the caches are
harness_nb's layout, and the server log is <out>/sglang.log. Every probe records its result in <out>/probe.json and
never stops the others; the exit code is 0 once the server booted and the probes ran.

Probes (each with its question):
  P0 boot      did it boot, in how long; KV pool, GPU and host memory; did DFLASH capture the 5 Flash-Next layers?
  P1 chat      does the reasoning parser split thinking from the answer, and is a trivial answer right?
  P2 speed     DFlash2 acceptance (tokens per verify step) and decode tok/s at 1, 4, 8 and 12 requests in flight
  P3 session   does a streaming session's second turn equal a stateless re-prefill of the same tokens (greedy)?
  P4 long      does a ~40k-token prompt prefill, and how fast?
  P5 fidelity  decode against a teacher-forced prefill at a ~12k-token ARC prompt (logprob gap, acceptance)
  P6 fixed     one fixed sequence scored by prefill on every boot (logprob shift between server changes)
  P7 image     does the vision path answer, and does the model read a ruled c32r picture (g1's style) right: the
               grid's rows, columns and cells against the truth?
  P9 curve     acceptance by position: a greedy streamed decode from a short ARC prompt across the draft ring's first
               wrap (2,112 positions), tokens per verify step logged per chunk, mean per 256-token window
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path

ARC_PROMPT = (
    "Here is a puzzle. Each example maps an input grid to an output grid.\n"
    "Example 1 input:\n0 0 1\n0 1 0\n1 0 0\nExample 1 output:\n1 0 0\n0 1 0\n0 0 1\n"
    "Example 2 input:\n2 0 0\n0 0 0\n0 0 3\nExample 2 output:\n0 0 2\n0 0 0\n3 0 0\n"
    "Test input:\n4 0 0 5\n0 6 0 0\n0 0 0 0\n7 0 0 8\n"
    "Find the rule, explain it briefly, and give the test output grid."
)
LONG_PARAGRAPH = (
    "The grid puzzles in this collection follow simple rules about colours, shapes and positions, and each rule is "
    "shown through a few examples before a test input is given. "
)


def log(msg: str) -> None:
    print(f"[fn-bringup {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def meminfo_gb() -> dict:
    out = {}
    try:
        for line in open("/proc/meminfo", encoding="utf-8"):
            key, val = line.split(":", 1)
            if key in ("MemTotal", "MemAvailable", "Mlocked", "Shmem", "Cached"):
                out[key] = round(int(val.split()[0]) / 1e6, 1)
    except OSError as e:
        out["error"] = str(e)
    return out


def gpu_mem() -> str:
    r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader"],
                       capture_output=True, text=True, timeout=20)
    return r.stdout.strip()


def log_facts(path: Path) -> dict:
    """The boot's own lines that answer P0: weights, capture, memory, KV pool, graphs."""
    keep = re.compile(r"Load weight (begin|end)|DFLASH capture|DFLASH draft runner|KV Cache is allocated|"
                      r"max_total_num_tokens|Mamba|mamba|Capture cuda graph|avail mem|PLE|ple_offload|"
                      r"max_running_requests|Traceback|Error|error|WARNING")
    lines = []
    try:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if keep.search(line):
                lines.append(line[:400])
    except OSError as e:
        return {"error": str(e)}
    text = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    # A silent fallback that costs ~14 ms a step: DFlash skips the draft graph when < 1.0 GB is free after the target's
    # graphs (dflash_worker_v2.py:829). Boot 27 (1023_0554) lost it at 0.98 GB and ran its draft eager.
    draft_graph = ("disabled" if "Disable DFLASH draft cuda graph" in text else
                   "captured" if "Capture draft verify CUDA graph end" in text else "unknown")
    return {"lines": lines[-80:], "dflash_capture": [x for x in lines if "DFLASH capture" in x],
            "draft_cuda_graph": draft_graph}


async def post(http, base: str, path: str, body: dict, timeout: float = 1800) -> dict:
    import aiohttp

    async with http.post(f"{base}{path}", json=body, timeout=aiohttp.ClientTimeout(total=timeout)) as r:
        text = await r.text()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {"raw": text[:2000], "status": r.status}


def chat_ids(tok, text: str) -> list[int]:
    rendered = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True,
                                       tokenize=False)
    return tok.encode(rendered, add_special_tokens=False)


async def p1_chat(http, base: str) -> dict:
    body = {"model": "qwen3.8-flash-next", "messages": [{"role": "user", "content": "What is 15% of 240? Reply with "
                                                                                     "the number only."}],
            "max_tokens": 4096, "temperature": 0.6, "reasoning_effort": "low"}
    t = time.time()
    r = await post(http, base, "/v1/chat/completions", body)
    msg = (r.get("choices") or [{}])[0].get("message", {})
    content, reasoning = msg.get("content") or "", msg.get("reasoning_content") or ""
    return {"seconds": round(time.time() - t, 1), "content": content[:200], "reasoning_chars": len(reasoning),
            "answer_ok": "36" in content, "usage": r.get("usage"), "error": r.get("error") or r.get("raw")}


async def _skipped(why: str) -> dict:
    return {"skipped": why}


async def p2_long_decode(http, base: str, tok, concurrency: list[int], max_new: int, prompt_tokens: int) -> list[dict]:
    """Decode speed at a long context, prefill cancelled out: each concurrency runs twice on the same ~prompt_tokens ARC
    context (max_new 128, then max_new), and the step time is the wall difference over the verify-step difference.
    Past 2,051 tokens of context every QSA row selects 2,051, as in a harness round; P2's short prompt selects ~1k."""
    ids = arc_context_ids(tok, prompt_tokens)
    rows = []
    for n in concurrency:
        out = {}
        for new in (128, max_new):
            sp = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "max_new_tokens": new, "ignore_eos": True}
            t = time.time()
            res = await asyncio.gather(*[post(http, base, "/generate", {"input_ids": ids, "sampling_params": sp})
                                         for _ in range(n)])
            out[new] = (time.time() - t, [int(x.get("meta_info", {}).get("spec_verify_ct", 0) or 0) for x in res],
                        [int(x.get("meta_info", {}).get("completion_tokens", 0)) for x in res],
                        [str(x.get("error") or x.get("raw"))[:200] for x in res if "meta_info" not in x])
        (w1, s1, t1, e1), (w2, s2, t2, e2) = out[128], out[max_new]
        dsteps = (max(s2) - max(s1)) if s1 and s2 else 0
        row = {"in_flight": n, "prompt_tokens": len(ids), "wall_s": [round(w1, 1), round(w2, 1)],
               "decode_ms_per_step": round((w2 - w1) * 1000 / dsteps, 2) if dsteps > 0 else None,
               "accept_len": round(sum(t2) / sum(s2), 3) if sum(s2) else None,
               "decode_tok_s": round((sum(t2) - sum(t1)) / (w2 - w1), 1) if w2 > w1 else None,
               "errors": e1 + e2}
        log(f"P2L {n} in flight at {len(ids)} tokens: {row}")
        rows.append(row)
    return rows


async def p2_speed(http, base: str, tok, concurrency: list[int], max_new: int) -> list[dict]:
    ids = chat_ids(tok, ARC_PROMPT)
    rows = []
    for n in concurrency:
        # ignore_eos since boot 20 (1023_0440): every request decodes exactly max_new tokens, so the batch stays full
        # to the end. Before, requests stopped at EOS at different points and agg_tok_s moved with the lengths
        # (boot 14 6,895 tokens vs boot 19 7,535 at 8 in flight: +11 % aggregate, +2.6 % per request).
        sp = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "max_new_tokens": max_new, "ignore_eos": True}
        t = time.time()
        res = await asyncio.gather(*[post(http, base, "/generate", {"input_ids": ids, "sampling_params": sp})
                                     for _ in range(n)])
        wall = time.time() - t
        toks = [int(x.get("meta_info", {}).get("completion_tokens", 0)) for x in res]
        steps = [int(x.get("meta_info", {}).get("spec_verify_ct", 0) or 0) for x in res]
        lat = [float(x.get("meta_info", {}).get("e2e_latency", 0) or 0) for x in res]
        row = {"in_flight": n, "wall_s": round(wall, 1), "tokens": sum(toks),
               "agg_tok_s": round(sum(toks) / wall, 1) if wall else 0.0,
               "per_req_tok_s": round(sum(toks) / sum(lat), 1) if sum(lat) else 0.0,
               "accept_len": round(sum(toks) / sum(steps), 3) if sum(steps) else None,
               # The server's own speed, apart from how many draft tokens the sampled content lets it accept:
               # wall over the longest request's verify steps (prefill included, so a slight overestimate).
               "ms_per_step": round(wall * 1000 / max(steps), 2) if steps and max(steps) else None,
               "errors": [str(x.get("error") or x.get("raw"))[:200] for x in res if "meta_info" not in x]}
        log(f"P2 {n} in flight: {row}")
        rows.append(row)
    return rows


SESSION_TURNS = ("Now do the same for rotations.", "And for transpositions?", "Which of the three is cheapest?")


def first_diff(x, y):
    for i, (p, q) in enumerate(zip(x, y)):
        if p != q:
            return i
    return None if len(x) == len(y) else min(len(x), len(y))


async def p3_session(http, base: str, tok, n_new: int = 48) -> dict:
    """A 4-turn streaming session, greedy, against a stateless re-prefill of the same tokens at every turn. Each
    turn's prefix length mod 4 is recorded: QSA compresses keys in groups of 4, so a misaligned continuation is the
    case under test (SGLANG_AGI2_QSA_SESSION_PREFIX)."""
    greedy = {"temperature": 0.0, "max_new_tokens": n_new, "no_stop_trim": True, "skip_special_tokens": False,
              "spaces_between_special_tokens": False}
    sends = [chat_ids(tok, "Describe, step by step, how you would check whether two grids are mirror images.")]
    sends += [tok.encode(f"<|im_end|>\n<|im_start|>user\n{t}<|im_end|>\n<|im_start|>assistant\n",
                         add_special_tokens=False) for t in SESSION_TURNS]
    sid = await post(http, base, "/open_session", {"capacity_of_str_len": 262144, "streaming": True})
    if not isinstance(sid, str):
        return {"error": f"open_session: {str(sid)[:300]}"}
    hist, turns = [], []
    try:
        for k, send in enumerate(sends):
            prefix = len(hist)
            r = await post(http, base, "/generate", {"input_ids": send, "session_params": {"id": sid},
                                                     "sampling_params": greedy})
            out = r.get("output_ids") or []
            ref = (await post(http, base, "/generate", {"input_ids": hist + send, "sampling_params": greedy})).get(
                "output_ids") or []
            turns.append({"turn": k + 1, "prefix": prefix, "prefix_mod4": prefix % 4, "sent": len(send),
                          "out": len(out), "first_diff_vs_reprefill": first_diff(out, ref), "exact": out == ref,
                          "error": str(r.get("error") or r.get("raw") or "")[:200]})
            hist = hist + send + out
    finally:
        await post(http, base, "/close_session", {"session_id": sid})
    return {"turns": turns, "all_exact": all(t["exact"] for t in turns),
            "misaligned_turns": sum(1 for t in turns if t["prefix_mod4"])}


async def p12_session_lp(http, base: str, tok, prompt_tokens: int, n_turns: int, n_new: int, out: Path) -> dict:
    """A streaming session judged by logprobs rather than exact tokens. Greedy tokens flip at near-ties between two
    identical stateless requests on this server (P3's turn 1, which has no prefix, already differs), so an exact
    match is no test of the carried state. Here every session turn returns its own token logprobs; after the last
    turn one stateless prefill of the whole history scores the same tokens. A wrong carried state (GDN, QSA's
    compressed keys at a misaligned prefix, the pending ring) shows as large gaps and many argmax disagreements;
    a correct one sits at P5's decode-vs-prefill floor."""
    greedy = {"temperature": 0.0, "max_new_tokens": n_new, "ignore_eos": True}
    sends = [arc_context_ids(tok, prompt_tokens)]
    sends += [tok.encode(f"<|im_end|>\n<|im_start|>user\n{SESSION_TURNS[k % len(SESSION_TURNS)]}<|im_end|>\n"
                         "<|im_start|>assistant\n", add_special_tokens=False) for k in range(n_turns - 1)]
    sid = await post(http, base, "/open_session", {"capacity_of_str_len": 1048576, "streaming": True})
    if not isinstance(sid, str):
        return {"error": f"open_session: {str(sid)[:300]}"}
    hist, spans, dec_all = [], [], []
    try:
        for k, send in enumerate(sends):
            r = await post(http, base, "/generate", {"input_ids": send, "session_params": {"id": sid},
                                                     "return_logprob": True, "sampling_params": greedy})
            ids = r.get("output_ids") or []
            dec = [x[0] for x in (r.get("meta_info", {}).get("output_token_logprobs") or [])]
            if not ids or len(dec) != len(ids):
                return {"error": f"turn {k + 1}: {len(ids)} ids / {len(dec)} logprobs: {str(r)[:300]}"}
            start = len(hist) + len(send)
            spans.append((start, len(ids), (len(hist)) % 4))
            dec_all.append(dec)
            hist = hist + send + ids
    finally:
        await post(http, base, "/close_session", {"session_id": sid})
    ref = await post(http, base, "/generate", {"input_ids": hist, "return_logprob": True, "logprob_start_len": 0,
                                               "top_logprobs_num": 1,
                                               "sampling_params": {"temperature": 0.0, "max_new_tokens": 1}})
    rmi = ref.get("meta_info", {})
    pre = [x[0] for x in (rmi.get("input_token_logprobs") or [])]   # pre[i] = logprob of token i given tokens < i
    top = [x[0][1] if x else None for x in (rmi.get("input_top_logprobs") or [])]
    if len(pre) < len(hist):
        return {"error": f"prefill returned {len(pre)} logprobs for {len(hist)} tokens: {str(ref)[:300]}"}
    def stats(dec, pre_, top_, toks):
        gaps = sorted(abs(d - p) for d, p in zip(dec, pre_) if d is not None and p is not None)
        dis = sum(1 for t, x in zip(top_, toks) if t is not None and t != x)
        return gaps, dis

    turns, all_gaps, all_dis, all_n, ctl_gaps, ctl_dis = [], [], 0, 0, [], 0
    for k, ((start, n, mod4), dec) in enumerate(zip(spans, dec_all)):
        gaps, dis = stats(dec, pre[start:start + n], top[start:start + n], hist[start:start + n])
        row = {"turn": k + 1, "prefix_mod4": mod4, "tokens": n,
               "mean_abs_gap": round(sum(gaps) / len(gaps), 5) if gaps else None,
               "p99_abs_gap": round(gaps[int(0.99 * (len(gaps) - 1))], 5) if gaps else None,
               "max_abs_gap": round(gaps[-1], 5) if gaps else None, "argmax_disagrees": dis}
        # Control at the same context: a stateless greedy decode from the same prefix (no session), scored by its own
        # stateless prefill. Its gap is this turn's decode-vs-prefill floor; the session's gap should match it.
        c = await post(http, base, "/generate", {"input_ids": hist[:start], "return_logprob": True,
                                                 "sampling_params": greedy})
        c_ids = c.get("output_ids") or []
        c_dec = [x[0] for x in (c.get("meta_info", {}).get("output_token_logprobs") or [])]
        cr = await post(http, base, "/generate", {"input_ids": hist[:start] + c_ids, "return_logprob": True,
                                                  "logprob_start_len": start - 1, "top_logprobs_num": 1,
                                                  "sampling_params": {"temperature": 0.0, "max_new_tokens": 1}})
        cmi = cr.get("meta_info", {})
        c_pre = [x[0] for x in (cmi.get("input_token_logprobs") or [])][1:]
        c_top = [x[0][1] if x else None for x in (cmi.get("input_top_logprobs") or [])][1:]
        cg, cd = stats(c_dec, c_pre, c_top, c_ids)
        row.update({"ctl_mean_abs_gap": round(sum(cg) / len(cg), 5) if cg else None, "ctl_argmax_disagrees": cd,
                    "ctl_tokens": len(c_ids), "same_tokens_as_session": c_ids == hist[start:start + n]})
        turns.append(row)
        all_gaps += gaps
        all_dis += dis
        all_n += n
        ctl_gaps += cg
        ctl_dis += cd
    (out / "p12_session.json").write_text(json.dumps({"history": hist, "spans": spans, "session_logprobs": dec_all,
                                                      "prefill_logprobs": pre, "prefill_top_ids": top}),
                                          encoding="utf-8")
    return {"history_tokens": len(hist), "turns": turns,
            "mean_abs_gap": round(sum(all_gaps) / len(all_gaps), 5) if all_gaps else None,
            "argmax_disagrees": all_dis, "scored": all_n,
            "ctl_mean_abs_gap": round(sum(ctl_gaps) / len(ctl_gaps), 5) if ctl_gaps else None,
            "ctl_argmax_disagrees": ctl_dis}


async def p13_pressure(http, base: str, tok, n_sessions: int, prompt_tokens: int, n_turns: int, n_new: int,
                       out: Path, think_s: float = 0.0) -> dict:
    """KV-pool pressure, as a harness round brings it: n_sessions streaming sessions at once (more than the slots, so
    some queue), each a long ARC prompt then n_turns - 1 short user turns, n_new greedy tokens a turn, so their KV
    outgrows the pool and the server must retract and evict sessions (the RETRACT / EVICT flags). Counts errors and
    the server's retract / abort lines, then scores every session's tokens against one stateless prefill of its whole
    history: a session resumed from a retract or an eviction with a wrong state shows as large gaps, a right one sits
    at P12's floor."""
    greedy = {"temperature": 0.0, "max_new_tokens": n_new, "ignore_eos": True}
    log_path = out / "sglang.log"
    before = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""

    async def session(s: int) -> dict:
        # distinct prompts: a different length per session moves the cut through the parents
        sends = [arc_context_ids(tok, prompt_tokens + 977 * s)]
        sends += [tok.encode(f"<|im_end|>\n<|im_start|>user\n{SESSION_TURNS[(k + s) % len(SESSION_TURNS)]}<|im_end|>\n"
                             "<|im_start|>assistant\n", add_special_tokens=False) for k in range(n_turns - 1)]
        sid = await post(http, base, "/open_session", {"capacity_of_str_len": 4194304, "streaming": True})
        if not isinstance(sid, str):
            return {"error": f"open_session: {str(sid)[:200]}"}
        hist, spans, decs, errors, t0 = [], [], [], [], time.time()
        try:
            for k, send in enumerate(sends):
                r = await post(http, base, "/generate", {"input_ids": send, "session_params": {"id": sid},
                                                         "return_logprob": True, "sampling_params": greedy})
                ids = r.get("output_ids") or []
                dec = [x[0] for x in (r.get("meta_info", {}).get("output_token_logprobs") or [])]
                if not ids or len(dec) != len(ids):
                    errors.append(f"turn {k + 1}: {len(ids)} ids / {len(dec)} lps: {str(r)[:200]}")
                    break
                spans.append((len(hist) + len(send), len(ids)))
                decs.append(dec)
                hist = hist + send + ids
                if think_s > 0 and k < len(sends) - 1:
                    # a harness's tool time: the session sits idle holding its KV while the others run, so an
                    # overfull pool must evict idle sessions (SGLANG_ENABLE_STREAMING_SESSION_EVICT) and resume them
                    await asyncio.sleep(think_s * (0.5 + random.Random(s * 131 + k).random()))
        finally:
            await post(http, base, "/close_session", {"session_id": sid})
        return {"session": s, "hist": hist, "spans": spans, "decs": decs, "errors": errors,
                "seconds": round(time.time() - t0, 1)}

    t = time.time()
    results = await asyncio.gather(*[session(s) for s in range(n_sessions)])
    wall = time.time() - t
    after = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    new_log = after[len(before):]
    rows, all_gaps, all_dis, all_n = [], [], 0, 0
    for res in results:
        if res.get("error") or not res.get("spans"):
            rows.append({"session": res.get("session"), "error": res.get("error") or res.get("errors")})
            continue
        ref = await post(http, base, "/generate", {"input_ids": res["hist"], "return_logprob": True,
                                                   "logprob_start_len": 0, "top_logprobs_num": 1,
                                                   "sampling_params": {"temperature": 0.0, "max_new_tokens": 1}})
        rmi = ref.get("meta_info", {})
        pre = [x[0] for x in (rmi.get("input_token_logprobs") or [])]
        top = [x[0][1] if x else None for x in (rmi.get("input_top_logprobs") or [])]
        per_turn = []
        for (start, n), dec in zip(res["spans"], res["decs"]):
            g = [abs(d - p) for d, p in zip(dec, pre[start:start + n]) if d is not None and p is not None]
            dis = sum(1 for tk, x in zip(top[start:start + n], res["hist"][start:start + n]) if tk is not None
                      and tk != x)
            per_turn.append((round(sum(g) / len(g), 4) if g else None, dis))
            all_gaps += g
            all_dis += dis
            all_n += n
        rows.append({"session": res["session"], "history_tokens": len(res["hist"]), "turns": len(res["spans"]),
                     "seconds": res["seconds"], "errors": res["errors"], "turn_gap_dis": per_turn})
    (out / "p13_pressure.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    return {"sessions": n_sessions, "wall_s": round(wall, 1),
            "tokens_total": sum(r.get("history_tokens", 0) for r in rows),
            "sessions_with_errors": sum(1 for r in rows if r.get("error") or r.get("errors")),
            "retract_lines": new_log.count("Retract requests"), "abort_lines": new_log.count("aborted request"),
            "evict_lines": new_log.count("Evicting idle session"), "park_lines": new_log.count("Parked idle session"),
            "think_s": think_s,
            "mean_abs_gap": round(sum(all_gaps) / len(all_gaps), 5) if all_gaps else None,
            "argmax_disagrees": all_dis, "scored": all_n,
            "worst_turn_gap": max((g for r in rows for g, _ in r.get("turn_gap_dis", []) if g is not None),
                                  default=None)}


async def p4_long(http, base: str, tok, target_tokens: int) -> dict:
    unit = tok.encode(LONG_PARAGRAPH, add_special_tokens=False)
    body_ids = (unit * (target_tokens // len(unit) + 1))[:target_tokens]
    ids = chat_ids(tok, "Summarise the following in one sentence.\n") + body_ids
    t = time.time()
    r = await post(http, base, "/generate", {"input_ids": ids, "sampling_params": {"temperature": 0.0,
                                                                                    "max_new_tokens": 32}})
    wall = time.time() - t
    return {"prompt_tokens": len(ids), "seconds": round(wall, 1), "prefill_tok_s": round(len(ids) / wall, 1),
            "ok": "meta_info" in r, "error": str(r.get("error") or r.get("raw") or "")[:300]}


def arc_context_ids(tok, target_tokens: int) -> list[int]:
    """A realistic long prompt: v2b1's evaluation parents (#21-#30, never the eval children) as grids in text, until
    ~target_tokens, ending with a request to solve the last one. Sparse selection matters past QSA's 2,048 tokens."""
    import os

    data = os.path.join(os.getcwd(), "data", "harness_bench")
    tasks = json.load(open(os.path.join(data, "v2b1_challenges.json"), encoding="utf-8"))
    parents = json.load(open(os.path.join(data, "v2b1_families.json"), encoding="utf-8"))["parents"]
    parent_ids = list(parents) if isinstance(parents, dict) else [p if isinstance(p, str) else p["id"] for p in parents]

    def grid(g):
        return "\n".join(" ".join(str(c) for c in row) for row in g)

    parts, n = [], 0
    for k, tid in enumerate(parent_ids):
        t = tasks[tid]
        text = f"Puzzle {k + 1}.\n" + "".join(
            f"Example {i + 1} input:\n{grid(p['input'])}\nExample {i + 1} output:\n{grid(p['output'])}\n"
            for i, p in enumerate(t["train"])) + f"Test input:\n{grid(t['test'][0]['input'])}\n\n"
        parts.append(text)
        n += len(tok.encode(text, add_special_tokens=False))
        if n >= target_tokens:
            break
    ask = ("Above are several grid puzzles. Find the rule of the LAST puzzle from its examples, explain your "
           "reasoning step by step, and write its test output grid.\n\n")
    return chat_ids(tok, ask + "".join(parts))


async def p6_fixed(http, base: str, tok, ref_path: Path, out: Path, prompt_tokens: int, gen_tokens: int) -> dict:
    """Teacher-forced logprobs of ONE fixed token sequence, identical on every boot: the first boot generates the
    continuation (greedy) after an ARC prompt and saves prompt + continuation to ref_path; every boot then scores that
    same sequence by one prefill. Comparing the saved per-token logprobs across boots isolates what each server change
    (KV dtype, GDN state dtype) does to the prefill path; P5 covers decode."""
    if ref_path.exists():
        ref = json.loads(ref_path.read_text(encoding="utf-8"))
        ids, cont, made = ref["prompt_ids"], ref["cont_ids"], False
    else:
        ids = arc_context_ids(tok, prompt_tokens)
        gen = await post(http, base, "/generate", {
            "input_ids": ids, "sampling_params": {"temperature": 0.0, "max_new_tokens": gen_tokens,
                                                  "ignore_eos": True}})
        cont, made = gen.get("output_ids") or [], True
        if not cont:
            return {"error": f"no continuation: {str(gen)[:300]}"}
        ref_path.write_text(json.dumps({"prompt_ids": ids, "cont_ids": cont}), encoding="utf-8")
    r = await post(http, base, "/generate", {
        "input_ids": ids + cont, "return_logprob": True, "logprob_start_len": len(ids) - 1, "top_logprobs_num": 1,
        "sampling_params": {"temperature": 0.0, "max_new_tokens": 1}})
    mi = r.get("meta_info", {})
    lps = [x[0] for x in (mi.get("input_token_logprobs") or [])][1:]
    tops = [x[0][1] if x else None for x in (mi.get("input_top_logprobs") or [])][1:]
    if not lps:
        return {"error": f"no logprobs: {str(r)[:300]}"}
    (out / "p6_logprobs.json").write_text(json.dumps({"logprobs": lps, "top_ids": tops}), encoding="utf-8")
    n = min(len(lps), len(cont))
    return {"ref_generated_here": made, "prompt_tokens": len(ids), "scored": n,
            "mean_logprob": round(sum(x for x in lps[:n] if x is not None) / n, 5),
            "argmax_matches_sequence": sum(1 for i in range(min(n, len(tops))) if tops[i] == cont[i])}


async def p5_fidelity(http, base: str, tok, prompt_tokens: int, gen_tokens: int, out: Path) -> dict:
    """Decode against a prefill-built floor at a context past QSA's 2,048-token index budget: greedy DFLASH decode
    of gen_tokens after a prompt_tokens prompt, then one teacher-forced prefill of prompt + output. Prefill compresses
    every group from in-chunk rows, so its index keys are right by construction; decode compresses from the pending
    ring. The gap between the two logprob streams (and argmax disagreements) measures what decode's compression adds
    on top of the numeric floor (SGLANG_AGI2_QSA_RING2 off vs on)."""
    import math

    ids = arc_context_ids(tok, prompt_tokens)
    gen = await post(http, base, "/generate", {
        "input_ids": ids, "return_logprob": True, "top_logprobs_num": 1,
        "sampling_params": {"temperature": 0.0, "max_new_tokens": gen_tokens, "ignore_eos": True}})
    mi = gen.get("meta_info", {})
    out_ids = gen.get("output_ids") or []
    dec = [x[0] for x in (mi.get("output_token_logprobs") or [])]
    if not out_ids or len(dec) != len(out_ids):
        return {"error": f"decode returned {len(out_ids)} ids / {len(dec)} logprobs: {str(gen)[:300]}"}
    ref = await post(http, base, "/generate", {
        "input_ids": ids + out_ids, "return_logprob": True, "logprob_start_len": len(ids) - 1, "top_logprobs_num": 1,
        "sampling_params": {"temperature": 0.0, "max_new_tokens": 1}})
    rmi = ref.get("meta_info", {})
    pre = [x[0] for x in (rmi.get("input_token_logprobs") or [])][1:]   # logprob of output token i given its prefix
    top = [x[0][1] if x else None for x in (rmi.get("input_top_logprobs") or [])][1:]
    n = min(len(dec), len(pre))
    if n == 0:
        return {"error": f"prefill returned no logprobs: {str(ref)[:300]}"}
    (out / "p5_tokens.json").write_text(json.dumps({"output_ids": out_ids, "decode_logprobs": dec,
                                                     "prefill_logprobs": pre, "prefill_top_ids": top}), encoding="utf-8")
    gaps = [abs(dec[i] - pre[i]) for i in range(n) if dec[i] is not None and pre[i] is not None]
    disagree = sum(1 for i in range(min(n, len(top))) if top[i] is not None and top[i] != out_ids[i])
    gaps_sorted = sorted(gaps)
    return {"prompt_tokens": len(ids), "generated": len(out_ids), "compared": n,
            "mean_abs_logprob_gap": round(sum(gaps) / len(gaps), 5) if gaps else None,
            "p99_abs_gap": round(gaps_sorted[int(0.99 * (len(gaps_sorted) - 1))], 5) if gaps else None,
            "max_abs_gap": round(gaps_sorted[-1], 5) if gaps else None,
            "prefill_argmax_disagrees": disagree,
            "decode_mean_logprob": round(sum(x for x in dec if x is not None) / n, 5),
            "spec_accept_len": round(len(out_ids) / mi["spec_verify_ct"], 3) if mi.get("spec_verify_ct") else None,
            "ppl_decode": round(math.exp(-sum(x for x in dec[:n] if x is not None) / n), 4)}


def digit_rows(text: str) -> list[list[int]]:
    """The last block of consecutive lines that are only digits (space-separated or not): a grid written as text."""
    blocks, cur = [], []
    for line in text.splitlines():
        cells = line.replace(",", " ").split()
        if cells and all(c.isdigit() for c in cells):
            row = [int(c) for c in cells] if len(cells) > 1 else [int(ch) for ch in cells[0]]
            cur.append(row)
        elif cur:
            blocks.append(cur)
            cur = []
    if cur:
        blocks.append(cur)
    return blocks[-1] if blocks else []


async def p7_image(http, base: str, n_grids: int = 3) -> dict:
    """g1's pictures through the server's vision path: a ruled c32r picture (arc_vision, the harness's renderer) of
    the first training input of v2b1's first evaluation parents (never the eval children), and a request to read it
    back as digits. Records the prompt tokens against the picture's visual tokens, and how much of the grid comes back
    right (dimensions, cells)."""
    import os

    from agi2_modal_harness.workloads.arc_eval.runtime import arc_vision as av

    data = os.path.join(os.getcwd(), "data", "harness_bench")
    tasks = json.load(open(os.path.join(data, "v2b1_challenges.json"), encoding="utf-8"))
    parents = json.load(open(os.path.join(data, "v2b1_families.json"), encoding="utf-8"))["parents"]
    parent_ids = list(parents) if isinstance(parents, dict) else [p if isinstance(p, str) else p["id"] for p in parents]
    rows = []
    for tid in parent_ids[:n_grids]:
        g = tasks[tid]["train"][0]["input"]
        uri, vis = av.picture(g, style=av.STYLES["c32r"])
        ask = ("The picture shows one grid, drawn with its row and column numbers in the margins and a heavier line "
               "every 5 cells; the colours are " + av.COLOURS + ". How many rows and columns does it have? Then write "
               "the whole grid as digits, one row per line, the digits separated by spaces.")
        body = {"model": "qwen3.8-flash-next", "max_tokens": 16384, "temperature": 0.6, "reasoning_effort": "low",
                "messages": [{"role": "user", "content": [{"type": "text", "text": ask},
                                                          {"type": "image_url", "image_url": {"url": uri}}]}]}
        t = time.time()
        r = await post(http, base, "/v1/chat/completions", body)
        msg = (r.get("choices") or [{}])[0].get("message", {})
        content = msg.get("content") or ""
        got = digit_rows(content)
        h, w = len(g), len(g[0])
        dims_ok = len(got) == h and all(len(x) == w for x in got)
        cells = sum(got[i][j] == g[i][j] for i in range(h) for j in range(w)) if dims_ok else 0
        rows.append({"task": tid, "size": f"{h}x{w}", "visual_tokens": vis, "seconds": round(time.time() - t, 1),
                     "usage": r.get("usage"), "dims_ok": dims_ok, "cells_right": cells, "cells": h * w,
                     "exact": dims_ok and cells == h * w, "reply_head": content[:240],
                     "reasoning_chars": len(msg.get("reasoning_content") or ""),
                     "error": str(r.get("error") or r.get("raw") or "")[:300]})
        log(f"P7 {tid} {h}x{w}: {json.dumps(rows[-1])[:400]}")
    return {"grids": rows, "answered": sum(1 for x in rows if x["usage"]), "exact": sum(1 for x in rows if x["exact"]),
            "cells_right": sum(x["cells_right"] for x in rows), "cells": sum(x["cells"] for x in rows)}


async def p9_curve(http, base: str, tok, prompt_tokens: int, gen_tokens: int, out: Path) -> dict:
    """Acceptance by position across the draft ring's first wrap. One greedy streamed /generate from an ARC prompt of
    ~prompt_tokens; each streamed chunk is one verify step (stream_interval 1), so the growth of completion_tokens
    per chunk is that step's accepted length. Records (position, accepted) per step and the mean per 256-token
    window of absolute position; the ring (2,112 slots) first wraps when the sequence passes 2,112 tokens."""
    import aiohttp

    # the bring-up's small puzzle (~250 tokens), so the decode starts below the ring and crosses its first wrap;
    # arc_context_ids adds whole puzzles and one v2b1 parent alone can pass 2,112 tokens
    ids = chat_ids(tok, ARC_PROMPT + " Then explain, at length, two other rules you considered and why they fail.")
    body = {"input_ids": ids, "stream": True,
            "sampling_params": {"temperature": 0.0, "max_new_tokens": gen_tokens, "ignore_eos": True}}
    steps, last = [], 0
    # the generated ids too, so two boots' sequences can be compared: if the ring moves the target's own decode
    # (1022_2005: decode-vs-prefill disagreements 23 -> 71), the arms part, and acceptance after that point is on
    # different text. SGLang streams output_ids either cumulatively or per chunk; both are handled below.
    out_ids, text = [], ""
    async with http.post(f"{base}/generate", json=body, timeout=aiohttp.ClientTimeout(total=3600)) as r:
        async for raw in r.content:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:") or line.endswith("[DONE]"):
                continue
            try:
                ev = json.loads(line[5:].strip())
            except json.JSONDecodeError:
                continue
            n = int((ev.get("meta_info") or {}).get("completion_tokens", 0))
            got = ev.get("output_ids") or []
            if len(got) == n:
                out_ids = list(got)
            elif n > last:
                out_ids.extend(got)
            text = ev.get("text") or text
            if n > last:
                steps.append((len(ids) + last, n - last))
                last = n
    (out / "p9_steps.json").write_text(json.dumps({"prompt_tokens": len(ids), "steps": steps, "output_ids": out_ids,
                                                   "text": text}), encoding="utf-8")
    windows = {}
    for pos, k in steps:
        windows.setdefault(pos // 256 * 256, []).append(k)
    curve = [(w, round(sum(v) / len(v), 3), len(v)) for w, v in sorted(windows.items())]
    before = [k for pos, k in steps if pos < 2112]
    after = [k for pos, k in steps if pos >= 2112 + 256]
    return {"prompt_tokens": len(ids), "generated": last, "steps": len(steps),
            "accept_before_wrap": round(sum(before) / len(before), 3) if before else None,
            "accept_after_wrap": round(sum(after) / len(after), 3) if after else None,
            "curve": curve}


async def p10_census(http, base: str, tok, n_tasks: int, max_new: int, out: Path, code_n: int = 0) -> dict:
    """Expert-usage census for pruning decisions: ARC-AGI-2 TRAINING tasks only (never the evaluation tasks or the
    bench sets, which are what a pruned model would later be scored on), one puzzle per request, sampled at the model
    card's settings with thinking, while SGLang's expert-distribution recorder counts (layer, expert) selections.
    Needs the server flags --expert-distribution-recorder-mode stat and SGLANG_AGI2_EXPERT_COUNT_SUM=1 (a running
    sum) and SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR; the dump is reduced to [layers, experts] counts in out/."""
    import glob

    if n_tasks <= 0 and code_n <= 0:
        return {"skipped": "census tasks 0"}
    path = os.path.join(os.getcwd(), "data", "original_competition_data", "arc-agi_training_challenges.json")
    tasks = json.load(open(path, encoding="utf-8"))
    ids = sorted(tasks)

    def grid(g):
        return "\n".join(" ".join(str(c) for c in row) for row in g)

    picked, prompts = [], []
    step = max(1, len(ids) // max(1, n_tasks * 3))
    for tid in (ids[::step] if n_tasks > 0 else []):
        t = tasks[tid]
        text = ("Here is a puzzle. Each example maps an input grid to an output grid.\n" + "".join(
            f"Example {i + 1} input:\n{grid(p['input'])}\nExample {i + 1} output:\n{grid(p['output'])}\n"
            for i, p in enumerate(t["train"])) + f"Test input:\n{grid(t['test'][0]['input'])}\n"
            "Find the rule and give the test output grid.")
        ids_ = chat_ids(tok, text)
        if len(ids_) > 6000:
            continue
        picked.append(tid)
        prompts.append(ids_)
        if len(picked) >= n_tasks:
            break
    rec_dir = os.environ.get("SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR", "/tmp")
    before = set(glob.glob(os.path.join(rec_dir, "expert_distribution_recorder_*.pt")))
    await post(http, base, "/start_expert_distribution_record", {})
    t0, prompt_tok, gen_tok = time.time(), 0, 0
    for k, ids_ in enumerate(prompts):
        r = await post(http, base, "/generate", {
            "input_ids": ids_, "sampling_params": {"temperature": 1.0, "top_p": 0.95, "top_k": 20,
                                                   "max_new_tokens": max_new}})
        mi = r.get("meta_info", {})
        prompt_tok += len(ids_)
        gen_tok += int(mi.get("completion_tokens", 0))
        log(f"P10 census {k + 1}/{len(prompts)} {picked[k]}: prompt {len(ids_)}, generated "
            f"{mi.get('completion_tokens')}, {time.time() - t0:.0f} s")
    # Code and tool-call traffic (round 1023_0833: a prose-only census pruned the experts that carry Python): more
    # TRAINING puzzles, disjoint from the prose picks, asked for a `python` tool call with thinking off.
    code_picked = []
    if code_n > 0:
        tool = {"type": "function", "function": {"name": "python", "description": "Run Python code and return its "
                "stdout.", "parameters": {"type": "object", "properties": {"code": {"type": "string"}},
                                          "required": ["code"]}}}
        pool = [t for t in ids[3::max(1, len(ids) // (code_n * 3))] if t not in picked]
        sem = asyncio.Semaphore(4)

        async def code_one(tid):
            t = tasks[tid]
            text = ("Here is a puzzle. Each example maps an input grid to an output grid.\n" + "".join(
                f"Example {i + 1} input:\n{grid(p['input'])}\nExample {i + 1} output:\n{grid(p['output'])}\n"
                for i, p in enumerate(t["train"])) + "Call the python tool with a function `transform(grid)` that "
                "maps each example input to its output, and test it on the examples.")
            async with sem:
                r = await post(http, base, "/v1/chat/completions", {
                    "model": "qwen3.8-flash-next", "messages": [{"role": "user", "content": text}], "tools": [tool],
                    "temperature": 1.0, "top_p": 0.95, "top_k": 20, "max_tokens": max_new,
                    "chat_template_kwargs": {"enable_thinking": False}})
            return int((r.get("usage") or {}).get("completion_tokens") or 0)

        code_picked = [t for t in pool if len(json.dumps(tasks[t]["train"])) < 9000][:code_n]
        gens = await asyncio.gather(*(code_one(t) for t in code_picked))
        gen_tok += sum(gens)
        log(f"P10 census code: {len(code_picked)} tool-call puzzles, {sum(gens)} tokens, {time.time() - t0:.0f} s")
    await post(http, base, "/stop_expert_distribution_record", {})
    await post(http, base, "/dump_expert_distribution_record", {})
    time.sleep(3)
    new = sorted(set(glob.glob(os.path.join(rec_dir, "expert_distribution_recorder_*.pt"))) - before)
    res = {"tasks": picked, "code_tasks": code_picked, "prompt_tokens": prompt_tok, "generated_tokens": gen_tok,
           "seconds": round(time.time() - t0), "dumps": new}
    if new:
        import torch

        d = torch.load(new[-1], map_location="cpu")
        counts = d["logical_count"].sum(dim=0)  # [layers, experts]
        (out / "p10_expert_counts.json").write_text(json.dumps({"tasks": picked, "code_tasks": code_picked,
                                                                "counts": counts.tolist()}),
                                                     encoding="utf-8")
        tot = counts.sum(dim=1, keepdim=True).clamp(min=1)
        share = (counts / tot).sort(dim=1, descending=True).values
        for frac in (0.5, 0.625, 0.75, 0.875):
            k = int(counts.shape[1] * frac)
            res[f"routing_share_top_{k}"] = round(float(share[:, :k].sum(dim=1).mean()), 4)
        res["never_used_per_layer_mean"] = round(float((counts == 0).sum(dim=1).float().mean()), 1)
    return res


async def p11_profile(http, base: str, tok, concurrency: int, max_new: int, num_steps: int, out: Path,
                      with_stack: bool = False) -> dict:
    """A torch-profiler capture of decode steps at `concurrency` requests in flight (SGLang /start_profile, one trace
    per TP rank in out/torch_prof): the requests are long greedy decodes, the profile starts once they are all
    decoding, and covers num_steps forward passes."""
    import glob

    if concurrency <= 0:
        return {"skipped": "profile concurrency 0"}
    prof_dir = out / "torch_prof"
    prof_dir.mkdir(parents=True, exist_ok=True)

    async def one(i: int):
        ids = chat_ids(tok, ARC_PROMPT + f" Then explain, at length, other rules you considered (variant {i}).")
        # the harness's sampling (P2's), so the profiled verify path is the sampled one the rounds run
        return await post(http, base, "/generate", {
            "input_ids": ids, "sampling_params": {"temperature": 1.0, "top_p": 0.95, "top_k": 20,
                                                  "max_new_tokens": max_new, "ignore_eos": True}})

    tasks = [asyncio.create_task(one(i)) for i in range(concurrency)]
    await asyncio.sleep(8)
    r = await post(http, base, "/start_profile", {"output_dir": str(prof_dir), "num_steps": num_steps,
                                                  "activities": ["CPU", "GPU"], "record_shapes": False,
                                                  # Python stacks attribute the host time between kernels
                                                  # (--profile-stack); off by default, they inflate the capture
                                                  "with_stack": with_stack})
    await asyncio.gather(*tasks)
    time.sleep(5)
    files = sorted(glob.glob(str(prof_dir / "**" / "*"), recursive=True))
    return {"start_profile": str(r)[:300], "files": [f for f in files if os.path.isfile(f)][:12]}


async def p14_code(base: str, n: int, out: Path) -> dict:
    """Code generation (agi2_modal_harness/comm/fn_l4_code_probe.py): n ARC-AGI-2 training puzzles, a Python
    `transform(grid)` each, thinking off, the harness's sampling, then compile(). Round 1023_0833 on the K = 320 pruned
    model: 2 of 24 valid, the rest collapsed into repeated keywords."""
    import subprocess

    if n <= 0:
        return {"skipped": "code probe 0"}
    cmd = [sys.executable, "-m", "agi2_modal_harness.comm.fn_l4_code_probe", "--base", base, "--n", str(n),
           "--conc", "4", "--out", str(out / "code_probe.json")]
    r = await asyncio.to_thread(subprocess.run, cmd, capture_output=True, text=True, cwd="/opt/dev")
    lines = [x for x in (r.stdout or "").splitlines() if x.strip()]
    (out / "code_probe.txt").write_text(r.stdout + r.stderr, encoding="utf-8")
    return {"rc": r.returncode, "summary": next((x for x in lines if x.startswith("[code-probe]")), lines[-1:]),
            "stderr": (r.stderr or "")[-300:]}


def _with_fid_lengths(probes, http, base: str, tok, out: Path, a):
    """--fid-prompts n1,n2,...: P5 and P6 once per prompt length (P5_fidelity_<n>k, P6_fixed_<n>k), each with its own
    output folder (out/fid_<n>) and its own P6 reference (<p6-ref>.<n>), in place of the single-length pair. Without
    the flag the list is unchanged."""
    lengths = [int(x) for x in (a.fid_prompts or "").split(",") if x.strip()]
    for name, coro in probes:
        if not lengths or name not in ("P5_fidelity", "P6_fixed"):
            yield name, coro
            continue
        if name == "P6_fixed":
            continue  # emitted beside each P5 below
        for n in lengths:
            sub = out / f"fid_{n}"
            sub.mkdir(parents=True, exist_ok=True)
            tag = f"{n // 1000}k"
            yield f"P5_fidelity_{tag}", (lambda n=n, sub=sub: p5_fidelity(http, base, tok, n, a.fid_gen, sub))
            if "P6_fixed" not in a.skip:
                yield f"P6_fixed_{tag}", (lambda n=n, sub=sub: p6_fixed(http, base, tok, Path(f"{a.p6_ref}.{n}"), sub,
                                                                        n, a.fid_gen))


async def run_probes(base: str, model_dir: str, out: Path, a) -> dict:
    import aiohttp
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir)
    res: dict = {}
    async with aiohttp.ClientSession() as http:
        for name, coro in _with_fid_lengths((("P1_chat", lambda: p1_chat(http, base)),
                           ("P14_code", lambda: p14_code(base, a.code_probe, out)),
                           ("P3_session", lambda: p3_session(http, base, tok)),
                           ("P12_session_lp", lambda: p12_session_lp(http, base, tok, a.sess_prompt, a.sess_turns,
                                                                     a.sess_gen, out)),
                           ("P13_pressure", lambda: p13_pressure(http, base, tok, a.press_sessions, a.press_prompt,
                                                                 a.press_turns, a.press_gen, out, a.press_think)),
                           ("P2_speed", lambda: p2_speed(http, base, tok, a.concurrency, a.max_new)),
                           ("P2L_long_decode", lambda: (p2_long_decode(http, base, tok, a.concurrency, a.max_new,
                                                                       a.p2_long_prompt)
                                                        if a.p2_long_prompt > 0 else _skipped("p2-long-prompt 0"))),
                           ("P4_long", lambda: p4_long(http, base, tok, a.long_tokens)),
                           ("P5_fidelity", lambda: p5_fidelity(http, base, tok, a.fid_prompt, a.fid_gen, out)),
                           ("P6_fixed", lambda: p6_fixed(http, base, tok, Path(a.p6_ref), out, a.fid_prompt,
                                                         a.fid_gen)),
                           ("P7_image", lambda: p7_image(http, base)),
                           ("P9_curve", lambda: p9_curve(http, base, tok, a.curve_prompt, a.curve_gen, out)),
                           ("P10_census", lambda: p10_census(http, base, tok, a.census_tasks, a.census_max_new,
                                                              out, a.census_code)),
                           ("P11_profile", lambda: p11_profile(http, base, tok, a.profile_concurrency,
                                                                a.profile_max_new, a.profile_steps, out,
                                                                a.profile_stack))), http, base, tok, out, a):
            if name in a.skip:
                log(f"{name}: skipped (--skip)")
                continue
            t = time.time()
            try:
                res[name] = await coro()
            except Exception as e:  # noqa: BLE001 - a failed probe is a result, never the end of the others
                res[name] = {"exception": f"{type(e).__name__}: {e}"}
            log(f"{name} ({time.time() - t:.0f} s): {json.dumps(res[name])[:600]}")
            (out / "probe.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--arm", default="harness")
    ap.add_argument("--out", required=True)
    ap.add_argument("--env", action="append", default=[], help="K=V for the server process (repeatable)")
    ap.add_argument("--server-arg", action="append", default=[], help="an extra launch argument (repeatable)")
    ap.add_argument("--model-dir", default="", help="serve this directory instead of the config alias's (e.g. a "
                    "text-only copy whose config.json sets language_model_only)")
    ap.add_argument("--boot-timeout-min", type=float, default=45)
    ap.add_argument("--cache-gpu", default="RTX-PRO-6000", help="GPU name in the JIT cache path (L4 on Modal 4 x L4), "
                    "so kernels compiled for one GPU type are never reused on another")
    ap.add_argument("--concurrency", type=lambda s: [int(x) for x in s.split(",")], default=[1, 4, 8, 12])
    ap.add_argument("--max-new", type=int, default=1536)
    ap.add_argument("--long-tokens", type=int, default=40000)
    ap.add_argument("--sess-prompt", type=int, default=3000, help="P12: first-turn ARC prompt tokens")
    ap.add_argument("--sess-turns", type=int, default=6, help="P12: session turns")
    ap.add_argument("--sess-gen", type=int, default=192, help="P12: greedy tokens per turn")
    ap.add_argument("--press-sessions", type=int, default=10, help="P13: concurrent streaming sessions")
    ap.add_argument("--press-prompt", type=int, default=40000, help="P13: first-turn prompt tokens (+977 per session)")
    ap.add_argument("--press-turns", type=int, default=4, help="P13: turns per session")
    ap.add_argument("--press-gen", type=int, default=1024, help="P13: greedy tokens per turn")
    ap.add_argument("--press-think", type=float, default=0.0, help="P13: mean idle seconds between a session's turns "
                    "(0.5-1.5 x this; a harness's tool time), so an overfull pool must evict idle sessions")
    ap.add_argument("--fid-prompt", type=int, default=6000, help="P5 prompt tokens (past QSA's 2,048-token budget)")
    ap.add_argument("--fid-gen", type=int, default=1500, help="P5/P6 greedy decode tokens")
    ap.add_argument("--fid-prompts", default="", help="comma-separated P5/P6 prompt lengths, one P5 + P6 pair each "
                    "(out/fid_<n>, P6 reference <p6-ref>.<n>); empty = the single --fid-prompt pair")
    ap.add_argument("--curve-prompt", type=int, default=1500, help="P9 prompt tokens (below the ring's 2,112 slots)")
    ap.add_argument("--curve-gen", type=int, default=3000, help="P9 greedy decode tokens (across the first wrap)")
    ap.add_argument("--census-tasks", type=int, default=0, help="P10: ARC-AGI-2 training tasks for the expert census "
                    "(0 = skip); the server needs --expert-distribution-recorder-mode stat")
    ap.add_argument("--census-max-new", type=int, default=2048, help="P10 decode tokens per task")
    ap.add_argument("--census-code", type=int, default=0, help="P10: extra training puzzles asked for a python tool "
                    "call with thinking off, so the census covers code and tool-call tokens")
    ap.add_argument("--profile-concurrency", type=int, default=0, help="P11: requests in flight for the torch "
                    "profile of decode steps (0 = skip)")
    ap.add_argument("--profile-max-new", type=int, default=1024)
    ap.add_argument("--profile-steps", type=int, default=8)
    ap.add_argument("--profile-stack", action="store_true", help="P11 with Python stacks (with_stack)")
    ap.add_argument("--p6-ref", default="/content/run/p6_reference.json",
                    help="P6's fixed sequence: written by the first boot that runs P6, reused by every later one")
    ap.add_argument("--code-probe", type=int, default=0, help="P14: training puzzles for the code-generation probe "
                    "(0 = skip)")
    ap.add_argument("--p2-long-prompt", type=int, default=0, help="P2L: prompt tokens for the long-context decode "
                    "speed probe (0 = skip); ~4,000 already makes every QSA row select its full 2,051")
    ap.add_argument("--skip", type=lambda s: [x for x in s.split(",") if x], default=[],
                    help="probes to leave out, e.g. P3_session (QSA asserts on a misaligned session prefix)")
    a = ap.parse_args(argv)

    sys.path.insert(0, os.getcwd())
    from agi2_modal_harness.core import config, models
    from agi2_modal_harness.core.runtime.sglang_server import SGLangServer
    from agi2_modal_harness.workloads.sglang_bench.bench_config import BenchConfig

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = BenchConfig.load(Path(a.config), arms=[a.arm])
    arm = cfg.arms[0]
    model_dir = a.model_dir or str(models.model_path(config.MOUNT_MODELS, cfg.model_alias))
    draft = str(models.model_path(config.MOUNT_MODELS, arm.draft_alias)) if arm.draft_alias else None
    cmd = cfg.command(arm, model_dir, draft) + list(a.server_arg)
    server_cfg = cfg.arm_server(arm)
    cache = Path(config.MOUNT_CACHE) / "sglang-0.5.21" / a.cache_gpu / "flashnext"
    # SGLANG_JIT_CACHE_DIR too: SGLang's JIT kernels (Marlin MoE, QSA, hyper-connections) default to
    # ~/.cache/sglang/jit on the container disk, and the Marlin MoE module alone compiles for minutes per GPU type.
    # Resolved: on Modal /vol/cache is a symlink to /__modal/volumes/<id>. The JIT cache drops the build's own files
    # from its dependency list with is_relative_to(staging) on resolved paths against the unresolved staging dir, so
    # under a symlink it records .staging-<uuid>/cuda.cu, which vanishes on publish: every lookup then misses and every
    # module rebuilds on every boot (24 leaves of speculative_sampling by boot 18; FP8 Marlin 6.5 min per boot).
    cache = cache.resolve()
    env = {"SGLANG_CACHE_DIR": str(cache / "sglang"), "TRITON_CACHE_DIR": str(cache / "triton"),
           "TORCHINDUCTOR_CACHE_DIR": str(cache / "inductor"), "FLASHINFER_WORKSPACE_BASE": str(cache / "fi"),
           "XDG_CACHE_HOME": str(cache / "xdg"), "SGLANG_JIT_CACHE_DIR": str(cache / "jit"),
           **server_cfg.env, **arm.env}
    for kv in a.env:
        k, _, v = kv.partition("=")
        env[k] = v
    for key in ("SGLANG_CACHE_DIR", "TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR", "FLASHINFER_WORKSPACE_BASE",
                "XDG_CACHE_HOME", "SGLANG_JIT_CACHE_DIR"):
        Path(env[key]).mkdir(parents=True, exist_ok=True)
    (out / "server_cmd.json").write_text(json.dumps({"cmd": cmd, "env": env}, indent=2), encoding="utf-8")
    log(f"host before boot: {meminfo_gb()}; GPU {gpu_mem()}")
    server = SGLangServer(cmd, port=server_cfg.port, log_path=out / "sglang.log", echo=log, env=env)
    result: dict = {"config": a.config, "arm": a.arm, "env": a.env, "server_args": a.server_arg}
    server.start()
    try:
        boot = server.wait_ready(a.boot_timeout_min * 60, on_tick=lambda s: None)
        info = server.server_info()
        internal = (info.get("internal_states") or [{}])[0] if isinstance(info, dict) else {}
        result["P0_boot"] = {"boot_s": round(boot, 1), "max_total_num_tokens": info.get("max_total_num_tokens"),
                             "internal": {k: v for k, v in internal.items()
                                          if re.search(r"token|running|mamba|memory|accept|graph", k)},
                             "host_gb": meminfo_gb(), "gpu_mem": gpu_mem(), "log": log_facts(out / "sglang.log")}
        log(f"P0 boot in {boot:.0f} s: pool {result['P0_boot']['max_total_num_tokens']}, "
            f"draft cuda graph {result['P0_boot']['log'].get('draft_cuda_graph')}, "
            f"capture {result['P0_boot']['log'].get('dflash_capture')}, host {result['P0_boot']['host_gb']}, "
            f"GPU {result['P0_boot']['gpu_mem']}")
        (out / "probe.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        probes = asyncio.run(run_probes(server.base_url, model_dir, out, a))
        result.update(probes)
        result["host_after_gb"] = meminfo_gb()
        result["gpu_after"] = gpu_mem()
        result["metrics"] = server.metrics()
    except Exception as e:  # noqa: BLE001 - the boot failure is the result
        result["P0_boot"] = {"failed": f"{type(e).__name__}: {str(e)[:3000]}", "log": log_facts(out / "sglang.log"),
                             "host_gb": meminfo_gb()}
        log(f"boot failed: {str(e)[:1500]}")
        (out / "probe.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        server.stop()
        return 2
    (out / "probe.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    server.stop()
    log("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
