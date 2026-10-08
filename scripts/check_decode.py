"""Inference check on a trained TREE checkpoint (no dense code is built, eval mode, fp32):
 1. cached token-by-token decoding must reproduce the parallel forward pass (same logits, same top-1 token);
 2. lm only: greedy text generation through the cached path from a real validation prompt (printed if tokenizer.json is present).
  python scripts/check_decode.py --task lm --ckpt runs/lm/tree/ckpt.pt --data_dir data/wiki
  python scripts/check_decode.py --task mqar --ckpt runs/mqar/tree/ckpt.pt --T 1024"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import load
from treeattn.metrics import decode_sanity
from treeattn.sources import add_source_args, make_source

p = argparse.ArgumentParser()
add_source_args(p)
p.add_argument('--ckpt', required=True)
p.add_argument('--T', type=int, default=2048)
p.add_argument('--n_decode', type=int, default=64)
p.add_argument('--n_gen', type=int, default=60)
a = p.parse_args()
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
model, cfg, _ = load(a.ckpt, dev, want='tree')
src = make_source(a, 0)
x = src.eval_window(a.T, dev)
T = x.shape[1]
r = decode_sanity(model, x, a.n_decode)                             # parallel forward vs cached token-by-token decoding (fp32)
print('DECODE CHECK | window %d, last %d tokens decoded one by one through the cache' % (T, r['n']))
print('  max |logit difference| %.2e | top-1 token agreement %.1f%%' % (r['max_diff'], r['top1_agree_pct']))
if a.task == 'lm':
    tokp = os.path.join(a.data_dir, 'tokenizer.json')
    n0 = min(1024, T)
    with torch.no_grad():
        out = model.generate(x[:, :n0], a.n_gen)
    if os.path.exists(tokp):
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(tokp)
        print('  PROMPT (last 300 chars): ...' + tok.decode(x[0, :n0].tolist())[-300:].replace('\n', ' '))
        print('  GENERATED (cached path):', tok.decode(out[0, n0:].tolist()).replace('\n', ' '))
    else:
        print('  generated token ids:', out[0, -a.n_gen:].tolist())
