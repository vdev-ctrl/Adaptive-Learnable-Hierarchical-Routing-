"""Empirical check that tree INFERENCE is sub-quadratic, on the real code path (random weights, eval mode).

For one cached decode step at growing context length T it reports
  * keys read by exact attention per token (capped by max_keys: a constant),
  * tree-node elements gathered per token (descent + neighbour search: O(wmax * log T)),
  * milliseconds per token,
  * KV-cache + tree-level memory (linear in T: the tree is kept up to date in place, never rebuilt).
The context is filled with random K/V and a random tree (the cost of a step depends on shapes, not on values).
Then it times the parallel prefill (us/token must grow only ~log T, not ~T) and exits non-zero if a bound is violated.
  python scripts/check_complexity.py --T_list 1024,4096,16384,65536"""
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import treeattn.attention as att
import treeattn.tree as tree
from treeattn import Config, GPT
from treeattn.tree import build_levels

p = argparse.ArgumentParser()
p.add_argument('--T_list', default='1024,4096,16384,65536')
p.add_argument('--steps', type=int, default=8)
p.add_argument('--prefill', default='1024,4096,16384')
a = p.parse_args()
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
cfg = Config(n_layer=2, n_embd=128, n_head=4, vocab_size=1024, rope_cap=256)
model = GPT(cfg, 'tree').to(dev).eval()
assert not any('dense' in type(m).__name__.lower() for m in model.modules()), 'a dense module is present in the tree model'

counter = {'elems': 0}
orig = tree.gather_nodes


def counting_gather(S, ids):
    counter['elems'] += ids.numel() * S.shape[-1]
    return orig(S, ids)


tree.gather_nodes = att.gather_nodes = counting_gather                # counts every tree-node / key gather

rows = []
with torch.no_grad():
    for T in [int(v) for v in a.T_list.split(',')]:
        caches = model.new_caches(1, T + a.steps + 1)
        for c in caches:
            c.k.normal_(), c.v.normal_(), c.qc.normal_()
            kidx = torch.randn(1, cfg.n_head, c.tpad, cfg.content_dim, device=dev)
            c.S, c.G, c.t = build_levels(kidx, c.tpad), kidx[:, :, :T].sum(2), T
        cache_mb = sum(sum(t.numel() * t.element_size() for t in [c.k, c.v, c.qc, c.tops] + [s for s in c.S if s is not None]) for c in caches) / 2 ** 20
        tok = torch.randint(0, cfg.vocab_size, (1, 1), device=dev)
        model(tok, caches=caches)                                       # warm-up (also advances t by 1)
        counter['elems'] = 0
        keys = 0.0
        t0 = time.time()
        for _ in range(a.steps):
            _, _, st = model(tok, caches=caches)
            keys += st['keys']
        if dev == 'cuda':
            torch.cuda.synchronize()
        ms = 1e3 * (time.time() - t0) / a.steps
        rows.append((T, keys / a.steps, counter['elems'] / a.steps, ms, cache_mb))
        print('decode @ context %-6d | keys read/token %5.1f | gathered elems/token %9.0f | %7.2f ms/token | cache %8.1f MB' % rows[-1], flush=True)

ok = True
if len(rows) > 1:
    (T0, k0, e0, m0, c0), (T1, k1, e1, m1, c1) = rows[0], rows[-1]
    r = T1 / T0
    print('\ncontext grew x%.0f: keys/token x%.2f (cap %d) | gathered elems/token x%.2f (log T grows x%.2f) | cache x%.2f' % (
        r, k1 / k0, cfg.max_keys, e1 / e0, torch.log2(torch.tensor(float(T1))) / torch.log2(torch.tensor(float(T0))), c1 / c0))
    ok &= k1 <= cfg.max_keys and e1 / e0 < 3.0 and c1 / c0 < 1.3 * r                  # constant keys, ~log work, linear memory
    ok &= e1 / e0 < r ** 0.5                                                           # far below linear growth

pre = []
with torch.no_grad(), torch.autocast(dev, dtype=torch.float16, enabled=dev == 'cuda'):
    for T in [int(v) for v in a.prefill.split(',')]:
        x = torch.randint(0, cfg.vocab_size, (1, T), device=dev)
        model(x[:, :512])
        t0 = time.time()
        model(x)
        if dev == 'cuda':
            torch.cuda.synchronize()
        pre.append((T, 1e6 * (time.time() - t0) / T))
        print('prefill  T=%-6d | %.1f us/token' % pre[-1], flush=True)
if len(pre) > 1:
    r = pre[-1][0] / pre[0][0]
    g = pre[-1][1] / pre[0][1]
    print('prefill length x%.0f: us/token x%.2f (quadratic would be ~x%.0f)' % (r, g, r))
    ok &= g < r ** 0.5
print('\nRESULT:', 'sub-quadratic: constant keys per token, ~log T tree work per token, linear cache, n log n prefill' if ok else 'BOUND VIOLATED')
sys.exit(0 if ok else 1)
