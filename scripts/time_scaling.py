"""Measured cost of ONE forward pass (random weights, eval mode, batch 1) at growing lengths: us/token and peak GPU memory.
Quadratic attention makes us/token grow ~linearly with T (x4 per 4x length); n log n grows slowly.
--dense 1 also times the dense head (full causal softmax, flash/SDPA where available) for comparison.
  python scripts/time_scaling.py --T_list 1024,4096,16384,65536 --dense 1"""
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from treeattn import Config, GPT

p = argparse.ArgumentParser()
p.add_argument('--T_list', default='1024,4096,16384,65536')
p.add_argument('--reps', type=int, default=2)
p.add_argument('--dense', type=int, default=0)
a = p.parse_args()
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
cfg = Config(rope_cap=256)
kinds = ['tree'] + (['dense'] if a.dense else [])
for kind in kinds:
    model = GPT(cfg, kind).to(dev).eval()
    print('%s | T | seconds | us/token | peak GPU MB' % kind)
    with torch.no_grad(), torch.autocast(dev, dtype=torch.float16, enabled=dev == 'cuda'):
        for T in [int(v) for v in a.T_list.split(',')]:
            x = torch.randint(0, cfg.vocab_size, (1, T), device=dev)
            try:
                model(x[:, :1024])                                   # warm-up
                if dev == 'cuda':
                    torch.cuda.synchronize(), torch.cuda.reset_peak_memory_stats()
                t0 = time.time()
                for _ in range(a.reps):
                    model(x)
                if dev == 'cuda':
                    torch.cuda.synchronize()
            except torch.cuda.OutOfMemoryError:
                print('%s | %d | out of memory' % (kind, T), flush=True)
                torch.cuda.empty_cache()
                continue
            dt = (time.time() - t0) / a.reps
            print('%s | %d | %.2f | %.1f | %.0f' % (kind, T, dt, 1e6 * dt / T, torch.cuda.max_memory_allocated() / 2 ** 20 if dev == 'cuda' else 0), flush=True)
