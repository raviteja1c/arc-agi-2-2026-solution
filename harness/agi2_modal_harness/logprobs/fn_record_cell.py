# Logprob recording for a Flash-Next round (LOGPROBS=1; 2026-10-24, docs/reports/1024_2113_fn_logprobs/DESIGN.md).
# Additive and off by default: without LOGPROBS=1 this cell defines a few names and changes nothing. With it:
# - every /generate of every harness conversation also carries sampling_params.custom_params {"agi2_conf": LP_IDS}.
#   The sampler never reads custom_params; the server (fork agi2/feat/fn-conf, SGLANG_AGI2_FN_CONF=1) answers with
#   meta_info["agi2_conf"]: per output token 12 columns (its logprob, the entropies of the full and of the sampled
#   distribution, DeepConf's confidence, the top-1 margin, its rank, ...) and the logprobs of LP_IDS, as flat base64.
# - each attempt writes {out}/lp/<aid>.npz beside {out}/live/<aid>.jsonl: every conversation's exact token history,
#   each turn's span in it, and the per-token rows, checked against the turn's output ids; {out}/lp/index.jsonl gets
#   one line per attempt.
# Nothing the model sees changes, and no decision reads the rows (the user, 2026-10-24: analysis only).
import base64 as _lp_b64
import contextvars as _lp_cv

import numpy as _lp_np

LOGPROBS = os.environ.get('LOGPROBS') == '1'
# The ids, from Flash-Next's tokenizer (byte-identical to Qwen3.8-27B's: same tokenizer.json SHA-256), checked below
# against the served tokenizer. Self-checking words, revisions, alternatives, hedges, and the structural tokens.
LP_WORDS = {'Wait': 13784, ' Wait': 13428, ' check': 1716, ' verify': 9845, 'Actually': 50821, ' Actually': 32645,
            'Hmm': 77264, ' Alternatively': 37201, 'Alternatively': 88842, ' maybe': 6970, ' perhaps': 8106,
            '</think>': 248069, '<tool_call>': 248058, '<|im_end|>': 248046}
LP_IDS = list(LP_WORDS.values())
LP_ATTEMPT = _lp_cv.ContextVar('LP_ATTEMPT', default=None)   # the running attempt's recording, per asyncio task
LP_STATS = dict(attempts=0, turns=0, tokens=0, missing=0, misaligned=0, nan_rows=0, errors=0)


def lp_decode(meta):
    """meta_info["agi2_conf"] -> (rows [n, F] float32, token ids [n] int32, column names)."""
    n, f = meta['shape']
    val = _lp_np.frombuffer(_lp_b64.b64decode(meta['val_b64']), dtype='<f4').reshape(n, f)
    tok = _lp_np.frombuffer(_lp_b64.b64decode(meta['tok_b64']), dtype='<i4').reshape(n)
    return val, tok, list(meta['cols'])


def lp_turn_record(conv, data, thinking, g):
    """One finished turn of a conversation: its span in the history and its rows (None when the server sent none)."""
    out = data['output_ids']
    mi = data['meta_info']
    rec = dict(start=conv.out_at, n=len(out), tok=_lp_np.asarray(out, dtype=_lp_np.int32), thinking=bool(thinking),
               finish=g['finish'], prompt=g['prompt'], cached=g['cached'], t=time.time(), val=None, cols=None,
               ok=False, nan_rows=-1)
    c = mi.get('agi2_conf')
    if c is not None:
        val, tok, cols = lp_decode(c)
        rec.update(val=val, cols=cols, nan_rows=int(c.get('nan_rows', -1)),
                   ok=bool(tok.shape[0] == len(out) and _lp_np.array_equal(tok, rec['tok'])))
    return rec


_LP_BASE = getattr(SessionConv, '_lp_base', SessionConv)   # submission 10's SessionConv (cell e8365c62), once


class LPConv(_LP_BASE):
    """Submission 10's SessionConv that also asks for every turn's per-token confidence and keeps it (LOGPROBS).
    Only the request's custom_params differ; what is sent, sampled and parsed is the base class's."""

    _lp_base = _LP_BASE

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.lp_turns, self.lp_last = [], None
        att = LP_ATTEMPT.get()
        if att is not None:
            att['convs'].append(self)

    async def _post(self, path, body):
        if path == '/generate':
            sp = dict(body.get('sampling_params') or {})
            sp['custom_params'] = {**(sp.get('custom_params') or {}), 'agi2_conf': LP_IDS}
            body = {**body, 'sampling_params': sp}
            self.lp_last = None
        res = await super()._post(path, body)
        if path == '/generate':
            self.lp_last = res
        return res

    async def turn(self, max_tokens, thinking=True):
        g = await super().turn(max_tokens, thinking)
        try:
            self.lp_turns.append(lp_turn_record(self, self.lp_last, thinking, g))
        except Exception as e:  # noqa: BLE001 - the measurement must never cost the attempt a turn
            LP_STATS['errors'] += 1
            self.lp_turns.append(dict(start=self.out_at, n=len(self.hist) - self.out_at, error=f'{type(e).__name__}: {e}'[:200]))
        return g


def lp_roles(convs, rec):
    """'main' for the conversations whose turns are the attempt's own requests (rec['requests'], in order), 'side'
    for the others (the picture look, a verify pass)."""
    want = [q['tokens'] for q in (rec or {}).get('requests', [])]
    roles = []
    for conv in convs:
        mine = [len(t['tok']) if 'tok' in t else t['n'] for t in conv.lp_turns]
        hit = bool(mine) and any(want[i:i + len(mine)] == mine for i in range(len(want) - len(mine) + 1))
        roles.append('main' if hit else 'side')
    return roles


FINISH_CODE = {'stop': 0, 'tool_calls': 1, 'length': 2}


def lp_save(att, rec):
    """The attempt's npz: per conversation c its history hist_<c>; per turn (all conversations, in order) its
    conversation, span (start, n), row offset, flags and token counts; rows val [N, F] and their token ids tok [N]."""
    convs, live = att['convs'], att['live']
    folder = os.path.dirname(live)
    if os.path.basename(folder) == 'live':   # <batch>/live/<aid>.jsonl -> <batch>/lp/<aid>.npz
        folder = os.path.join(os.path.dirname(folder), 'lp')
    path = os.path.join(folder, os.path.basename(live).removesuffix('.jsonl') + '.npz')
    os.makedirs(folder, exist_ok=True)
    cols = next((t['cols'] for c in convs for t in c.lp_turns if t.get('cols')), None)
    width = len(cols) if cols else 12 + len(LP_IDS)
    arrays, vals, toks, tt = {}, [], [], []
    row0 = 0
    for ci, conv in enumerate(convs):
        hist = _lp_np.asarray(conv.hist, dtype=_lp_np.int32)
        arrays[f'hist_{ci}'] = hist
        for ti, t in enumerate(conv.lp_turns):
            n = int(t['n'])
            tok = t.get('tok')
            if tok is None:
                tok = hist[t['start']:t['start'] + n] if t['start'] + n <= len(hist) else _lp_np.full(n, -1, _lp_np.int32)
            val = t.get('val')
            has = val is not None and val.shape == (n, width)
            vals.append(val if has else _lp_np.full((n, width), _lp_np.nan, dtype=_lp_np.float32))
            toks.append(_lp_np.asarray(tok, dtype=_lp_np.int32))
            kept = t['start'] + n <= len(hist) and _lp_np.array_equal(hist[t['start']:t['start'] + n], tok)
            tt.append((ci, ti, t['start'], n, row0, int(has), int(bool(t.get('ok'))), int(not kept),
                       int(t.get('thinking', True)), FINISH_CODE.get(t.get('finish'), 3), int(t.get('prompt') or 0),
                       int(t.get('cached') or 0), int(t.get('nan_rows', -1))))
            row0 += n
    tt = _lp_np.asarray(tt, dtype=_lp_np.int64).reshape(-1, 13)
    roles = lp_roles(convs, rec)
    meta = dict(version=1, aid=os.path.basename(live).removesuffix('.jsonl'), live=live, ids=LP_IDS, words=LP_WORDS,
                cols=cols, roles=roles, finished=rec is not None, t0=att['t0'], t1=time.time(),
                turn_cols=['conv', 'turn', 'start', 'n', 'row0', 'has_rows', 'ok', 'replaced', 'thinking', 'finish',
                           'prompt', 'cached', 'nan_rows'])
    _lp_np.savez_compressed(
        path, val=_lp_np.concatenate(vals) if vals else _lp_np.zeros((0, width), _lp_np.float32),
        tok=_lp_np.concatenate(toks) if toks else _lp_np.zeros((0,), _lp_np.int32), turns=tt,
        meta=_lp_np.asarray(json.dumps(meta)), **arrays)
    n_tok = int(tt[:, 3].sum()) if len(tt) else 0
    miss = int((tt[:, 5] == 0).sum()) if len(tt) else 0
    bad = int(((tt[:, 5] == 1) & (tt[:, 6] == 0)).sum()) if len(tt) else 0
    LP_STATS.update(attempts=LP_STATS['attempts'] + 1, turns=LP_STATS['turns'] + len(tt), tokens=LP_STATS['tokens'] + n_tok,
                    missing=LP_STATS['missing'] + miss, misaligned=LP_STATS['misaligned'] + bad)
    with open(os.path.join(os.path.dirname(path), 'index.jsonl'), 'a') as f:
        f.write(json.dumps(dict(aid=meta['aid'], file=os.path.basename(path), convs=len(convs), roles=roles,
                                turns=len(tt), tokens=n_tok, turns_without_rows=miss, misaligned=bad,
                                finished=rec is not None, bytes=os.path.getsize(path))) + '\n')
    return path


def lp_wrap(fn):
    """An attempt function (night_attempt, sched_attempt) that records its conversations when LOGPROBS is on."""
    if getattr(fn, '_lp_wrapped', False):
        return fn

    async def run(*a, **kw):
        live = kw.get('live')
        if not LOGPROBS or not live:
            return await fn(*a, **kw)
        att = dict(convs=[], live=live, t0=time.time())
        token = LP_ATTEMPT.set(att)
        rec = None
        try:
            rec = await fn(*a, **kw)
            return rec
        finally:
            LP_ATTEMPT.reset(token)
            try:   # a stopped attempt (the round's stop) keeps the turns it finished
                lp_save(att, rec)
            except Exception as e:  # noqa: BLE001 - never cost the attempt its answer
                LP_STATS['errors'] += 1
                print(f"{time.strftime('%H:%M:%S')} [lp] could not save {live}: {type(e).__name__}: {e}", flush=True)

    run._lp_wrapped, run.__wrapped__ = True, fn
    return run


def lp_probe():
    """Fail closed before a LOGPROBS round: the served tokenizer gives LP_WORDS' ids, and the server answers a short
    request with rows aligned to its output (so it runs fork agi2/feat/fn-conf with SGLANG_AGI2_FN_CONF=1)."""
    bad = {w: ids(w) for w, t in LP_WORDS.items() if ids(w) != [t]}
    check(not bad, f'LOGPROBS: the served tokenizer gives other ids: {bad}')
    r = requests.post(f'{SERVER}/generate', timeout=600, json={
        'input_ids': ids('Count from one to five.'),
        'sampling_params': {**SAMPLING, 'max_new_tokens': 24, 'custom_params': {'agi2_conf': LP_IDS}}}).json()
    c = r.get('meta_info', {}).get('agi2_conf')
    check(c is not None, 'LOGPROBS: the server sent no agi2_conf (needs fork agi2/feat/fn-conf and SGLANG_AGI2_FN_CONF=1)')
    val, tok, cols = lp_decode(c)
    check(tok.tolist() == r['output_ids'] and val.shape == (len(tok), 12 + len(LP_IDS)) and c.get('aligned'),
          f"LOGPROBS: rows do not match the output ({val.shape}, {len(r['output_ids'])} ids)")
    fin = _lp_np.isfinite(val[:, 0])
    check(int((~fin).sum()) == 1 and not fin[0], 'LOGPROBS: expected exactly one NaN row, the prefill-sampled first token')
    print(f"[lp] probe: {len(tok)} tokens, columns {cols[:12]} + {len(LP_IDS)} ids; mean logprob "
          f"{float(val[fin, 0].mean()):.3f}, mean entropy {float(val[fin, 2].mean()):.3f}", flush=True)
    return True


if LOGPROBS:
    lp_probe()
    SessionConv = LPConv
    # 1018-harness-bench's attempt functions; a notebook without them (the 11-01 probe notebooks, whose solve cell calls
    # repl_attempt directly with live=) gets repl_attempt wrapped instead. Never both: night/sched call repl_attempt,
    # and a nested wrap would record every attempt twice.
    _lp_names = [n for n in ('night_attempt', 'sched_attempt') if callable(globals().get(n))]
    if not _lp_names and callable(globals().get('repl_attempt')):
        _lp_names = ['repl_attempt']
    check(_lp_names, 'LOGPROBS: no attempt function to wrap (night_attempt, sched_attempt or repl_attempt)')
    for _lp_name in _lp_names:
        globals()[_lp_name] = lp_wrap(globals()[_lp_name])
    print(f'LOGPROBS: wrapped {_lp_names}', flush=True)
    print('LOGPROBS: every harness turn asks for its per-token confidence; attempts write lp/<aid>.npz', flush=True)
else:
    print('LOGPROBS off: the round runs unchanged', flush=True)
