"""ONE report with the nine publication metrics for a trained checkpoint (tree = the inference path; a dense phase-1 checkpoint gets the
subset that exists for it). Measurement only: it loads the model and runs it, nothing is trained or changed.

  [1] top-1 token accuracy   [2] needle accuracy by depth   [3] prefill latency   [4] decode generation speed   [5] avg keys read per token
  [6] KV retrieval % / compression   [7] KV-cache top-1 agreement   [8] peak allocated VRAM   [9] KV-cache memory footprint
(definitions: treeattn/metrics.py). Printed to the screen and saved as <ckpt folder>/benchmark.txt and benchmark.json.

  MQAR : python scripts/benchmark.py --task mqar --ckpt runs/mqar/tree/ckpt.pt --eval_grid 256:16,1024:16,4096:16,16384:16
  lm   : python scripts/benchmark.py --task lm --data_dir data/wiki --ckpt runs/lm/tree/ckpt.pt --needle 1
  dense comparison: the same command with the phase-1 checkpoint (--ckpt runs/.../dense/ckpt.pt)."""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import load
from treeattn.metrics import needle_setup, run_benchmark, save_report
from treeattn.sources import add_source_args, make_source

p = argparse.ArgumentParser()
add_source_args(p)
p.add_argument('--ckpt', required=True)
p.add_argument('--lens', default='', help='context lengths of metrics 3-6, 8, 9 (default: lm --eval_lens, mqar the --eval_grid lengths)')
p.add_argument('--n_decode', type=int, default=64, help='tokens decoded one by one for the decode speed / decode retrieval')
p.add_argument('--reps', type=int, default=2, help='timed prefill repetitions per length')
p.add_argument('--amp', type=int, default=1, help='1: fp16 autocast on cuda (as in training and evaluation), 0: fp32. The decode sanity check [7] is always fp32')
p.add_argument('--sanity_T', type=int, default=0, help='window of the decode sanity check (0 = the largest length <= 2048)')
p.add_argument('--accuracy', default='final', choices=['final', 'quick', 'none'], help='[1]: the full task report, one line, or skip')
p.add_argument('--needle', type=int, default=0, help='1: also run [2] needle accuracy by depth (lm tasks; needs tokenizer.json and val.bin in --data_dir)')
p.add_argument('--needle_lengths', default='1024,4096,16384')
p.add_argument('--needle_depths', default='0.1,0.3,0.5,0.7,0.9')
p.add_argument('--needle_n', type=int, default=10, help='cases per (length, depth)')
p.add_argument('--needle_min_answer_id', type=int, default=3000)
p.add_argument('--save', type=int, default=1, help='1: write benchmark.txt / benchmark.json next to the checkpoint')
a = p.parse_args()

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
model, cfg, kind = load(a.ckpt, dev)
src = make_source(a, 0)
if a.lens:
    lens = [int(v) for v in a.lens.split(',')]
elif src.kind == 'mqar':
    lens = [T for T, _ in src.grid]
else:
    lens = list(src.eval_lens)

needle = None
if a.needle and src.kind == 'lm':
    try:
        needle = {'setup': needle_setup(a.data_dir, model, dev, a.needle_min_answer_id, amp=bool(a.amp) and dev == 'cuda'),
                  'lengths': [int(v) for v in a.needle_lengths.split(',')], 'depths': [float(v) for v in a.needle_depths.split(',')], 'n': a.needle_n}
    except Exception as e:
        print('needle test not available (%s: %s); continuing without [2]' % (type(e).__name__, e), flush=True)

lines, res = run_benchmark(model, cfg, src, dev, lens, n_decode=a.n_decode, reps=a.reps, amp=bool(a.amp), sanity_T=a.sanity_T or None,
                           accuracy=None if a.accuracy == 'none' else a.accuracy, needle=needle,
                           title='BENCHMARK %s' % os.path.basename(os.path.dirname(os.path.abspath(a.ckpt))))
if a.save:
    prefix = os.path.join(os.path.dirname(os.path.abspath(a.ckpt)), 'benchmark')
    save_report(lines, res, prefix)
    print('saved %s.txt and %s.json' % (prefix, prefix), flush=True)
