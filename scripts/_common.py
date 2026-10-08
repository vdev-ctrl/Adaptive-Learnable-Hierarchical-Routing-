"""Helpers shared by the phase and evaluation scripts."""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from treeattn import Config, GPT
from treeattn.distill import sample_rows
from treeattn.sources import task_defaults


def load(path, dev, want=None):
    ck = torch.load(path, map_location=dev, weights_only=False)
    kind = ck.get('attn', 'tree')
    assert want is None or kind == want, 'checkpoint holds a %s model, this script needs a %s model' % (kind, want)
    cfg = Config(**ck['cfg'])
    model = GPT(cfg, kind).to(dev)
    model.load_state_dict(ck['model'])
    return model.eval(), cfg, kind


def add_arch_args(p):
    for n in ('n_layer', 'n_head', 'n_embd'):
        p.add_argument('--' + n, type=int, default=None, help='default depends on --task (lm: 8/6/384, mqar: 4/4/256)')


def add_tree_args(p):
    p.add_argument('--wmax', type=int, default=None, help='beam width = most leaf groups (2 keys each) the tree can add per query')
    p.add_argument('--beam_classes', default=None, help='budget classes in leaf groups; the largest must be <= wmax')
    p.add_argument('--max_keys', type=int, default=None, help='hard cap on keys read per query (a constant)')
    p.add_argument('--q_chunk', type=int, default=None)
    p.add_argument('--rope_cap', type=int, default=-1, help='-1 = longest training length, 0 = off')
    p.add_argument('--ref_rows', type=int, default=128, help='dense attention rows extracted per sequence')
    p.add_argument('--ref_mass', type=float, default=0.9, help='budget label = fewest leaf groups holding this share of the dense attention')
    p.add_argument('--gate_mode', default='topk_pos', choices=['topk_pos', 'topk_mix', 'topk', 'soft', 'off'], help='gate on the dense reference targets (phase 2)')
    p.add_argument('--gate_k', type=int, default=8, help='number of top dense-attention keys per query row used as the phase-2 targets (and for the top-k hit metric); 8 default, try 1 for MQAR')
    p.add_argument('--gate_temp0', type=float, default=0.5, help='topk gate: starting temperature (relative to the row peak)')
    p.add_argument('--gate_temp1', type=float, default=0.01, help='topk gate: temperature just before it becomes hard')
    p.add_argument('--gate_tau', type=float, default=0.1, help='soft gate on the dense reference targets: keys below ~this x the row peak fade out (0 = off)')
    p.add_argument('--gate_soft', type=float, default=0.5, help='smooth edge width of the gate, as a fraction of --gate_tau')
    p.add_argument('--split_weight', type=float, default=0.3)
    p.add_argument('--pred_weight', type=float, default=0.1)


def fill_defaults(a):
    for k, v in task_defaults(a.task).items():
        if hasattr(a, k) and getattr(a, k) is None:
            setattr(a, k, v)
    return a


def tree_config(dcfg, a, seq_len):
    """Tree config on top of the dense config of phase 1 (same model size, vocabulary, RoPE layout)."""
    longest = max(dcfg.seq_len, seq_len)
    return Config(**{**dcfg.to_dict(), 'seq_len': longest, 'wmax': a.wmax, 'beam_classes': tuple(int(v) for v in a.beam_classes.split(',')),
                     'max_keys': a.max_keys, 'q_chunk': a.q_chunk, 'ref_rows': a.ref_rows, 'ref_mass': a.ref_mass,
                     'gate_mode': a.gate_mode, 'gate_k': a.gate_k, 'gate_temp0': a.gate_temp0, 'gate_temp1': a.gate_temp1,
                     'gate_tau': a.gate_tau, 'gate_soft': a.gate_soft,
                     'split_weight': a.split_weight, 'pred_weight': a.pred_weight,
                     'rope_cap': longest if a.rope_cap < 0 else a.rope_cap})


def load_teacher(path, dev):
    """The frozen reference: the phase-1 dense model, eval mode, no gradients."""
    ck = torch.load(path, map_location=dev, weights_only=False)
    assert ck.get('attn') == 'dense', 'the reference must be a dense (phase 1) checkpoint'
    cfg = Config(**ck['cfg'])
    teacher = GPT(cfg, 'dense').to(dev)
    teacher.load_state_dict(ck['model'])
    teacher.eval()
    for q in teacher.parameters():
        q.requires_grad_(False)
    return teacher, cfg, ck


def make_fwd(model, dev):
    @torch.no_grad()
    def fwd(x):
        with torch.autocast(dev, dtype=torch.float16, enabled=dev == 'cuda'):
            return model(x)[0]
    return fwd


def cosine_lr(base, step, steps, warmup):
    import math
    return base * min(1.0, (step + 1) / warmup) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / max(steps, 1))))


@torch.no_grad()
def recall(model, teacher, src, dev, cfg, step_base=10 ** 6):
    """Phase-3 monitor: the tree model run on its OWN stream, against the frozen dense model's attention rows of the same
    tokens. Returns (far, mass) averaged over the validation batches (the streams differ a little, so this is an estimate)."""
    model.eval()
    fars, masses = [], []
    for i, (x, y) in enumerate(src.val_batches(dev)):
        rows = sample_rows(x.shape[0], x.shape[1], cfg.ref_rows, cfg.ref_min_pos, step_base + i, 0, dev, focus=(y != -100) if src.focus_rows else None)
        with torch.autocast(dev, dtype=torch.float16, enabled=dev == 'cuda'):
            ref = (rows, teacher.extract_reference(x, rows))
            _, _, st = model(x, ref=ref)
        fars.append(st['far']), masses.append(st['mass'])
    model.train()
    return float(np.mean(fars)), float(np.mean(masses))


def add_bench_args(p):
    p.add_argument('--bench', type=int, default=1, help='1: print the efficiency metrics block (prefill / decode speed, keys read, KV retrieval, VRAM, KV-cache size) at the end of the run, 0: skip it')
    p.add_argument('--bench_lens', default='', help='context lengths of that block, comma separated (default: the training length)')
    p.add_argument('--bench_decode', type=int, default=32, help='tokens decoded one by one for the decode-speed measurement')


def run_bench_block(a, model, cfg, src, dev, title, accuracy):
    """End-of-phase metrics block (measurement only; see treeattn/metrics.py). accuracy: 'quick' / 'final' / None (already printed above)."""
    if not a.bench:
        return
    from treeattn.metrics import benchmark_block
    lens = [int(v) for v in a.bench_lens.split(',')] if a.bench_lens else [src.seq_len]
    benchmark_block(model, cfg, src, dev, title, accuracy=accuracy, lens=lens, n_decode=a.bench_decode, amp=dev == 'cuda')
