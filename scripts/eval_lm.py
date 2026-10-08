"""Evaluate a trained checkpoint (tree = the inference path, or the dense phase-1 model for comparison) on the task's report:
lm: loss / accuracy by position bucket; mqar: accuracy grid over context length and number of pairs.
  python scripts/eval_lm.py --task lm --ckpt runs/lm/tree/ckpt.pt --data_dir data/wiki
  python scripts/eval_lm.py --task mqar --ckpt runs/mqar/tree/ckpt.pt --eval_grid 256:16,1024:16,4096:16,16384:16"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import load, make_fwd
from treeattn.sources import add_source_args, make_source

p = argparse.ArgumentParser()
add_source_args(p)
p.add_argument('--ckpt', required=True)
a = p.parse_args()
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
model, cfg, kind = load(a.ckpt, dev)
src = make_source(a, 0)
print('EVAL | task %s | %s attention | trained at %d' % (a.task, kind, cfg.seq_len))
print('[1] TOP-1 TOKEN ACCURACY')
print('\n'.join(src.final(make_fwd(model, dev), dev)))
