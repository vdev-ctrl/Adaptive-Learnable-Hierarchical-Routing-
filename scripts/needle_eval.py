"""[2] NEEDLE ACCURACY BY DEPTH: needle-in-Wikipedia + free-prompt benchmark for a trained TREE checkpoint (evaluation only; nothing here is
used in training). The table now comes from treeattn/metrics.py (same numbers as before, plus a mean row over the lengths); the full nine-metric
report is scripts/benchmark.py.
  python scripts/needle_eval.py --ckpt runs/tree/ckpt.pt --data_dir data/wiki --out runs/demo_tree.txt"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import load
from treeattn.metrics import needle_setup, needle_table
from treeattn.needles import FREE_PROMPTS, needle_demo

p = argparse.ArgumentParser()
p.add_argument('--ckpt', required=True)
p.add_argument('--data_dir', default='data/wiki')
p.add_argument('--lengths', default='512,1024,2048,4096,8192,16384')
p.add_argument('--depths', default='0.1,0.3,0.5,0.7,0.9')
p.add_argument('--min_answer_id', type=int, default=3000)
p.add_argument('--n', type=int, default=20, help='cases per (length, depth)')
p.add_argument('--gen', type=int, default=40)
p.add_argument('--out', default='')
a = p.parse_args()
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
model, cfg, _ = load(a.ckpt, dev, want='tree')
S = needle_setup(a.data_dir, model, dev, a.min_answer_id, amp=dev == 'cuda')
logits_fn, val, enc, dec, answer_ids = S['logits_fn'], S['val'], S['enc'], S['dec'], S['answer_ids']


@torch.no_grad()
def gen_fn(ids, n):
    return model.generate(torch.tensor([ids], device=dev), n)[0, len(ids):].tolist()


out = ['NEEDLE-IN-WIKIPEDIA | model: tree attention | %d answer words (guess rate ~%.2f%%) | cases per cell: %d' % (len(answer_ids), 100.0 / len(answer_ids), a.n)]
depths = [float(v) for v in a.depths.split(',')]
lengths = [int(v) for v in a.lengths.split(',')]
out.append('[2] NEEDLE ACCURACY BY DEPTH: hidden word as the next token, by prompt length (rows) and depth of the queried fact (columns); last column = control without the fact')
lines, data = needle_table(logits_fn, val, enc, answer_ids, lengths, depths, a.n, seed=1, log=lambda s: print(s, flush=True))
out += lines
out.append('')
out.append('=== WHAT THE MODEL SAYS (greedy) ===')
out.append(needle_demo(logits_fn, gen_fn, val, enc, dec, answer_ids, L=1024, depth=0.5))
out.append(needle_demo(logits_fn, gen_fn, val, enc, dec, answer_ids, L=4096, depth=0.2, seed=8))
for pr in FREE_PROMPTS:
    out.append('  PROMPT: %r' % pr)
    out.append('  ANSWER: %r' % dec(gen_fn(enc(pr), a.gen)))
text = '\n'.join(out)
print('\n' + text)
if a.out:
    open(a.out, 'w').write(text)
    json.dump(data, open(os.path.splitext(a.out)[0] + '.json', 'w'), indent=1)
