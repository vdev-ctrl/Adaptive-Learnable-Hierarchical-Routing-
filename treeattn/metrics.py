"""Benchmark metrics: ONE place that measures and prints the nine publication metrics. Measurement only: nothing here changes the
model, the tree search, the budget predictor, the cache or any training code; it calls `model(...)` / `model.new_caches(...)` the same
way `GPT.generate` and the phase scripts do and reads the numbers the model already reports (`stats['keys']`, ...).

 [1] top-1 token accuracy        existing code (`src.final` / `src.quick`: MQAR answer accuracy, lm loss/accuracy by position bucket), only labelled
 [2] needle accuracy by depth    existing code (`needles.run_needle_eval`), table + mean row + control, `needle_table`
 [3] prefill latency             `measure_prefill`: one parallel forward over the whole context through fresh caches (cache allocation not timed)
 [4] decode generation speed     `measure_decode`: prefill, then greedy one-token-at-a-time steps with the KV cache (argmax inside the timing)
 [5] average keys read per token `stats['keys']` = keys the exact softmax attends to (<= max_keys), mean over layers, heads, queries
 [6] KV retrieval % / compression keys read / keys a dense causal model would read. Prefill: total read / (T (T+1) / 2) = 2 * keys / (T + 1).
                                 Decode: keys read / (context length + 1 token). Compression ratio = 1 / retrieval fraction.
 [7] KV-cache top-1 agreement    `decode_sanity`: cached token-by-token decoding vs the parallel forward pass (fp32): max |logit diff| and top-1 agreement
 [8] peak allocated VRAM         torch.cuda.max_memory_allocated during the prefill forward (and during decoding); also the part added by the forward
 [9] KV-cache memory footprint   real tensor sizes of the `LayerCache`s: KV (k, v) + tree index (queries, tree levels, global sum, stored leaves)

Dense checkpoints (phase 1) get the subset that exists for them: accuracy, prefill latency, peak VRAM, and the analytic dense numbers
(reads every key: retrieval 100%, KV cache = k + v). There is no decode / cache path in the dense head.
"""
import json
import os
import time

import numpy as np
import torch

MB = 2.0 ** 20


def _p(s=''):
    print(s, flush=True)


def _is_cuda(dev):
    return str(dev).startswith('cuda') and torch.cuda.is_available()


def _sync(dev):
    if _is_cuda(dev):
        torch.cuda.synchronize()


def _ac(dev, amp):
    return torch.autocast('cuda' if _is_cuda(dev) else 'cpu', dtype=torch.float16, enabled=bool(amp) and _is_cuda(dev))


def _device_name(dev):
    return torch.cuda.get_device_name(0) if _is_cuda(dev) else 'cpu'


def _nbytes(t):
    return t.numel() * t.element_size()


def _n(v, spec):
    """format a number, or 'n/a' when it does not exist."""
    return 'n/a' if v is None else spec % v


def make_fwd(model, dev, amp=True):
    """same as scripts/_common.make_fwd (logits only, fp16 autocast on cuda)."""
    @torch.no_grad()
    def fwd(x):
        with _ac(dev, amp):
            return model(x)[0]
    return fwd


# ------------------------------------------------------------------------------------------------ [9] KV-cache footprint
def cache_footprint(caches):
    """Real sizes of the LayerCaches (all layers). kv = keys + values (what a dense KV cache would also hold);
    index = what the tree adds on top (content queries, tree levels, global sum, stored leaf ids)."""
    kv = sum(_nbytes(c.k) + _nbytes(c.v) for c in caches)
    idx = 0
    for c in caches:
        idx += _nbytes(c.qc) + _nbytes(c.G) + _nbytes(c.tops) + sum(_nbytes(s) for s in c.S if s is not None)
    b, tmax = caches[0].k.shape[0], caches[0].tmax
    return {'kv_mb': kv / MB, 'index_mb': idx / MB, 'total_mb': (kv + idx) / MB, 'bytes_per_token': (kv + idx) / float(b * tmax), 'tmax': tmax,
            'dense_kv_mb': kv / MB}


def dense_cache_mb(model, b, T):
    """analytic KV cache of a dense model of this size at context T (k and v of every layer)."""
    cfg = model.cfg
    el = model.wte.weight.element_size()
    kv = 2 * cfg.n_layer * b * cfg.n_head * T * cfg.head_dim * el
    return {'kv_mb': kv / MB, 'index_mb': 0.0, 'total_mb': kv / MB, 'bytes_per_token': kv / float(b * T), 'tmax': T, 'dense_kv_mb': kv / MB}


# ------------------------------------------------------------------------------------------------ [3] [5] [6] [8] [9] prefill
@torch.no_grad()
def measure_prefill(model, x, amp=True, reps=2):
    """x (b,T). The first pass (untimed) measures peak VRAM and the cache footprint and warms the kernels up; then `reps` timed passes,
    each with fresh caches (allocation is outside the timed region)."""
    dev = x.device.type
    b, T = x.shape
    tree = model.attn_kind == 'tree'
    cuda = _is_cuda(dev)
    r = {'T': T, 'peak_mb': None, 'over_mb': None}
    if cuda:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
    caches = model.new_caches(b, T) if tree else None
    with _ac(dev, amp):
        _, _, st = model(x, caches=caches)
    if cuda:
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()
        r['peak_mb'], r['over_mb'] = peak / MB, (peak - base) / MB
    r['cache'] = cache_footprint(caches) if tree else dense_cache_mb(model, b, T)
    keys = st.get('keys') if tree else (T + 1) / 2.0                       # dense reads every causal key: mean (T + 1) / 2 per query
    r['keys'] = keys
    r['leaves_used'], r['nb_leaves'] = st.get('leaves_used'), st.get('nb_leaves')
    r['retrieval_pct'] = None if keys is None else 100.0 * keys / ((T + 1) / 2.0)
    r['compression'] = None if not keys else ((T + 1) / 2.0) / keys
    del caches
    times = []
    for _ in range(max(1, reps)):
        caches = model.new_caches(b, T) if tree else None
        _sync(dev)
        t0 = time.perf_counter()
        with _ac(dev, amp):
            model(x, caches=caches)
        _sync(dev)
        times.append(time.perf_counter() - t0)
        del caches
    r['ms'] = 1e3 * float(np.mean(times))
    r['us_per_token'] = 1e6 * float(np.mean(times)) / T
    return r


# ------------------------------------------------------------------------------------------------ [4] [5] [6] [8] decode
@torch.no_grad()
def measure_decode(model, prompt, n_decode=64, amp=True, warm=2):
    """Tree model only. Prefill `prompt` (b,T0) in parallel, then `warm` untimed + `n_decode` timed greedy steps, one token at a time through
    the caches (the same loop as GPT.generate). Each timed step = forward of one token + argmax."""
    assert model.attn_kind == 'tree', 'decoding with a KV cache exists for the tree model only'
    dev = prompt.device.type
    b, T0 = prompt.shape
    cuda = _is_cuda(dev)
    caches = model.new_caches(b, T0 + warm + n_decode)
    with _ac(dev, amp):
        logits, _, _ = model(prompt, caches=caches)
    nxt = logits[:, -1].argmax(-1, keepdim=True)
    base = 0
    if cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
    times, keys, ctxs = [], [], []
    for i in range(warm + n_decode):
        _sync(dev)
        t0 = time.perf_counter()
        with _ac(dev, amp):
            logits, _, st = model(nxt, caches=caches)
        nxt = logits[:, -1].argmax(-1, keepdim=True)
        _sync(dev)
        dt = time.perf_counter() - t0
        if i >= warm:
            times.append(dt)
            keys.append(st['keys'])
            ctxs.append(T0 + i + 1)                                        # keys a dense model reads for this token: every position up to itself
    r = {'n': n_decode, 'ms_mean': 1e3 * float(np.mean(times)), 'ms_median': 1e3 * float(np.median(times)), 'tok_s': n_decode / float(np.sum(times)),
         'keys': float(np.mean(keys)), 'retrieval_pct': 100.0 * float(np.sum(keys)) / float(np.sum(ctxs)), 'compression': float(np.sum(ctxs)) / float(np.sum(keys)),
         'dense_keys': float(np.mean(ctxs)), 'peak_mb': None, 'over_mb': None}
    if cuda:
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()
        r['peak_mb'], r['over_mb'] = peak / MB, (peak - base) / MB
    del caches
    return r


# ------------------------------------------------------------------------------------------------ [7] decode sanity
@torch.no_grad()
def decode_sanity(model, x, n_decode=64):
    """Cached token-by-token decoding of the last n_decode tokens of x (1,T) must reproduce the parallel forward pass.
    Call it outside autocast (fp32), like scripts/check_decode.py. Returns max |logit difference| and the top-1 agreement in percent."""
    assert model.attn_kind == 'tree'
    T = x.shape[1]
    n = max(1, min(n_decode, T - 1))
    pre = T - n
    ref, _, _ = model(x)                                                   # parallel forward over the whole window
    caches = model.new_caches(x.shape[0], T)
    model(x[:, :pre], caches=caches)                                       # prefill the prompt in parallel
    outs = []
    for t in range(pre, T):                                                # then one cached token at a time
        lg, _, _ = model(x[:, t:t + 1], caches=caches)
        outs.append(lg[:, 0])
    dec = torch.stack(outs, 1)
    ref = ref[:, pre:]
    return {'T': T, 'n': n, 'max_diff': (dec - ref).abs().max().item(),
            'top1_agree_pct': 100 * (dec.argmax(-1) == ref.argmax(-1)).float().mean().item()}


# ------------------------------------------------------------------------------------------------ [2] needle accuracy by depth
def needle_setup(data_dir, model, dev, min_answer_id=3000, amp=True):
    """tokenizer + validation tokens + answer vocabulary + a logits function for the needle test (same as scripts/needle_eval.py)."""
    from tokenizers import Tokenizer
    from .needles import answer_ids_from_vocab
    tok = Tokenizer.from_file(os.path.join(data_dir, 'tokenizer.json'))
    enc, dec = (lambda s: tok.encode(s).ids), (lambda ids: tok.decode(list(ids)))
    val = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    answer_ids = answer_ids_from_vocab(tok.get_vocab(), min_id=min_answer_id)

    @torch.no_grad()
    def logits_fn(ids):
        x = torch.tensor([ids], device=dev)
        with _ac(dev, amp):
            lg, _, _ = model(x)
        return lg[0, -1].float().cpu().numpy()

    return {'logits_fn': logits_fn, 'val': val, 'enc': enc, 'dec': dec, 'answer_ids': answer_ids}


def needle_table(logits_fn, val, enc, answer_ids, lengths, depths, n, seed=1, log=_p):
    """Accuracy of the hidden word as the next token: rows = prompt length, columns = depth of the queried fact, last column = control
    (fact missing). Adds a mean row over the lengths. Numbers come from needles.run_needle_eval, unchanged. -> (lines, data)."""
    from .needles import run_needle_eval
    lines = ['depth = where the queried fact starts in the haystack (10% = near the start, 90% = just before the question)',
             'length   ' + ' '.join('%5d%%' % int(d * 100) for d in depths) + '  | control']
    for s in lines:
        log(s)
    acc, ctl = {}, {}
    for L in lengths:
        res, c = run_needle_eval(logits_fn, val, enc, answer_ids, L, depths, n, seed=seed)
        acc[L], ctl[L] = res, c
        lines.append('%-8d ' % L + ' '.join('%6.2f' % res[d] for d in depths) + '  | %.2f' % c)
        log(lines[-1])
    mean_d = [float(np.mean([acc[L][d] for L in lengths])) for d in depths]
    mean_c = float(np.mean([ctl[L] for L in lengths]))
    lines.append('%-8s ' % 'mean' + ' '.join('%6.2f' % m for m in mean_d) + '  | %.2f' % mean_c)
    log(lines[-1])
    data = {'lengths': list(lengths), 'depths': list(depths), 'cases_per_cell': n,
            'acc': {str(L): {str(d): acc[L][d] for d in depths} for L in lengths}, 'control': {str(L): ctl[L] for L in lengths},
            'mean_by_depth': {str(d): m for d, m in zip(depths, mean_d)}, 'mean_control': mean_c}
    return lines, data


# ------------------------------------------------------------------------------------------------ the report
def measure_context(model, x, n_decode, reps, amp):
    r = {'prefill': measure_prefill(model, x, amp, reps)}
    if model.attn_kind == 'tree' and n_decode > 0:
        r['decode'] = measure_decode(model, x, n_decode, amp)
    return r


def context_lines(T, r, model):
    tree = model.attn_kind == 'tree'
    pre, dec = r['prefill'], r.get('decode')
    cfg = model.cfg
    L = ['  context %d' % T]
    L.append('    [3] prefill latency      : %.1f ms (%.1f us/token)' % (pre['ms'], pre['us_per_token']))
    if dec:
        L.append('    [4] decode speed         : %.1f tokens/s (%.2f ms/token, median %.2f) over %d tokens after the prefill' % (dec['tok_s'], dec['ms_mean'], dec['ms_median'], dec['n']))
    else:
        L.append('    [4] decode speed         : n/a (%s)' % ('dense head has no KV-cache decode path' if not tree else 'not measured'))
    kd = '    [5] avg keys read/token  : prefill %.1f' % pre['keys']
    if dec:
        kd += ' | decode %.1f' % dec['keys']
    kd += '   (dense reads %.0f | %s)' % ((T + 1) / 2.0, ('%.0f' % dec['dense_keys']) if dec else 'n/a') + ('   cap max_keys=%d' % cfg.max_keys if tree else '')
    L.append(kd)
    rd = '    [6] KV retrieval         : prefill %.2f%% (compression x%.1f)' % (pre['retrieval_pct'], pre['compression'])
    if dec:
        rd += ' | decode %.2f%% (x%.1f)' % (dec['retrieval_pct'], dec['compression'])
    L.append(rd)
    if pre['peak_mb'] is None:
        L.append('    [8] peak VRAM (forward)  : n/a (no cuda device)')
    else:
        v = '    [8] peak VRAM (forward)  : %.0f MB total | the forward adds %.0f MB' % (pre['peak_mb'], pre['over_mb'])
        if dec and dec['peak_mb'] is not None:
            v += ' | decoding: %.0f MB total, adds %.0f MB' % (dec['peak_mb'], dec['over_mb'])
        L.append(v)
    c = pre['cache']
    if tree:
        L.append('    [9] KV-cache footprint   : %.1f MB = KV %.1f + tree index %.1f (%.1f KB/token; a dense KV cache alone: %.1f MB)' % (
            c['total_mb'], c['kv_mb'], c['index_mb'], c['bytes_per_token'] / 1024.0, c['dense_kv_mb']))
    else:
        L.append('    [9] KV-cache footprint   : %.1f MB (analytic dense KV cache: k + v of every layer, %.1f KB/token)' % (c['total_mb'], c['bytes_per_token'] / 1024.0))
    return L


def summary_lines(rows, tree):
    L = ['SUMMARY (per context length)', '  %-7s %10s %8s %7s %7s %8s %8s' % ('T', 'prefill_ms', 'tok/s', 'keys', 'read%', 'peak_MB', 'cache_MB')]
    for T, r in rows:
        pre, dec = r['prefill'], r.get('decode')
        L.append('  %-7d %10.1f %8s %7s %7s %8s %8.1f' % (T, pre['ms'], _n(dec['tok_s'] if dec else None, '%.1f'), _n(dec['keys'] if dec else pre['keys'], '%.1f'),
                                                        _n(dec['retrieval_pct'] if dec else pre['retrieval_pct'], '%.2f'), _n(pre['peak_mb'], '%.0f'), pre['cache']['total_mb']))
    L.append('  (keys / read% are the decode-step numbers; prefill numbers are in the blocks above)' if tree else '  (dense: reads every key)')
    return L


def run_benchmark(model, cfg, src, dev, lens, n_decode=64, reps=2, amp=True, sanity_T=None, accuracy='final', needle=None,
                  title='BENCHMARK', log=_p):
    """Print and return (lines, results) for the nine metrics.
    accuracy: 'final' = the task's full report, 'quick' = its one-line check, None = skip [1] (the caller printed it already).
    needle: None or dict(setup=needle_setup(...), lengths=[...], depths=[...], n=int) for [2] (lm tasks)."""
    was_training = model.training
    model.eval()
    kind = model.attn_kind
    lines = []

    def out(s=''):
        lines.append(s)
        log(s)

    res = {'title': title, 'attention': kind, 'task': src.kind, 'device': _device_name(dev),
           'precision': 'fp16 autocast' if (amp and _is_cuda(dev)) else 'fp32', 'params_m': sum(q.numel() for q in model.parameters()) / 1e6, 'contexts': {}}
    bar = '=' * 78
    out(bar)
    out('%s | %s attention | task %s | %s | %s' % (title, kind, src.kind, res['device'], res['precision']))
    d = 'layers %d, heads %d, d_model %d, params %.2fM' % (cfg.n_layer, cfg.n_head, cfg.n_embd, res['params_m'])
    if kind == 'tree':
        d += ' | wmax %d, budget classes %s, max_keys %d, static keys %d + sink, neighbours %s, rope_cap %d' % (
            cfg.wmax, ','.join(str(v) for v in cfg.beam_classes), cfg.max_keys, cfg.local, 'on' if cfg.neighbors else 'off', cfg.rope_cap)
    out(d)
    out(bar)

    # ---- [1] top-1 token accuracy (existing code, only labelled)
    if accuracy:
        out('[1] TOP-1 TOKEN ACCURACY ' + ('(MQAR answer positions)' if src.kind == 'mqar' else '(next token; loss/accuracy by position bucket)'))
        try:
            fwd = make_fwd(model, dev, amp)
            body = src.final(fwd, dev) if accuracy == 'final' else [src.quick(fwd, dev)]
            for s in body:
                out('    ' + s)
            res['accuracy_lines'] = body
        except Exception as e:
            out('    failed: %s: %s' % (type(e).__name__, e))
        out('')

    # ---- [2] needle accuracy by depth (existing code, tabled)
    if needle is not None:
        out('[2] NEEDLE ACCURACY BY DEPTH (hidden word as the next token; guess rate ~%.2f%%; cases per cell %d)' % (100.0 / max(len(needle['setup']['answer_ids']), 1), needle['n']))
        try:
            s = needle['setup']
            _, res['needle'] = needle_table(s['logits_fn'], s['val'], s['enc'], s['answer_ids'], needle['lengths'], needle['depths'], needle['n'],
                                            log=lambda t: out('    ' + t))
        except Exception as e:
            out('    failed: %s: %s' % (type(e).__name__, e))
    elif src.kind == 'lm':
        out('[2] NEEDLE ACCURACY BY DEPTH: not run here (scripts/benchmark.py --needle 1, or scripts/needle_eval.py)')
    else:
        out('[2] NEEDLE ACCURACY BY DEPTH: n/a for MQAR (no needle text; see [1])')
    out('')

    # ---- [3]-[6], [8], [9] per context length
    out('[3] prefill  [4] decode speed  [5] keys read  [6] KV retrieval  [8] peak VRAM  [9] KV-cache footprint   (batch 1, by context length)')
    rows = []
    for T in lens:
        try:
            x = src.eval_window(T, dev)
            r = measure_context(model, x, n_decode, reps, amp)
            rows.append((T, r))
            res['contexts'][str(T)] = r
            for s in context_lines(T, r, model):
                out(s)
        except Exception as e:
            out('  context %d: skipped (%s: %s)' % (T, type(e).__name__, str(e).split('\n')[0][:120]))
        if _is_cuda(dev):
            torch.cuda.empty_cache()
    out('')

    # ---- [7] KV-cache top-1 agreement
    if kind == 'tree':
        st_ = sanity_T or max([T for T in lens if T <= 2048] or [min(lens)])
        out('[7] KV-CACHE TOP-1 AGREEMENT (cached one-token decoding vs the parallel forward pass, fp32, window %d)' % st_)
        try:
            r7 = decode_sanity(model, src.eval_window(st_, dev), min(64, max(1, st_ // 4)))
            res['decode_sanity'] = r7
            out('    last %d tokens | top-1 token agreement %.1f%% | max |logit difference| %.2e' % (r7['n'], r7['top1_agree_pct'], r7['max_diff']))
        except Exception as e:
            out('    failed: %s: %s' % (type(e).__name__, e))
    else:
        out('[7] KV-CACHE TOP-1 AGREEMENT: n/a (the dense head has no cache)')
    out('')

    if rows:
        for s in summary_lines(rows, kind == 'tree'):
            out(s)
        out(bar)
    model.train(was_training)
    return lines, res


def benchmark_block(model, cfg, src, dev, title, accuracy='quick', lens=None, n_decode=32, reps=1, amp=True):
    """The efficiency block the phase scripts print at their end. Never raises: a failing measurement must not lose a finished run."""
    try:
        return run_benchmark(model, cfg, src, dev, lens or [src.seq_len], n_decode=n_decode, reps=reps, amp=amp, accuracy=accuracy, title=title)
    except Exception as e:
        _p('  [metrics block failed: %s: %s]' % (type(e).__name__, e))
        return None


def save_report(lines, res, prefix):
    os.makedirs(os.path.dirname(os.path.abspath(prefix)), exist_ok=True)
    with open(prefix + '.txt', 'w') as f:
        f.write('\n'.join(lines) + '\n')
    with open(prefix + '.json', 'w') as f:
        json.dump(res, f, indent=1, default=float)
