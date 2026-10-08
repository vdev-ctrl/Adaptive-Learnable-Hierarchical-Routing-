"""Data sources for the three phases. A source gives training batches (x, y) as a pure function of (seed, step), validation
batches, and a printable accuracy / loss report for ANY forward function `fwd(x) -> logits` (dense or tree), so the phase scripts
are the same for Wikipedia / TinyStories / PG-19 token files (`lm`) and for MQAR (`mqar`).

  lm    : flat uint16 token files (train.bin / val.bin / meta.json from the prepare_* scripts), windows of --seq_len
  mqar  : synthetic multi-query associative recall in STAGES of growing context, `--stages T:n_kv:steps,...`
          (loss only on the answer positions: y == -100 elsewhere)
"""
import json
import os

import numpy as np
import torch

from .lmdata import bucket_line, train_batch, val_windows, window
from .mqar import MQARTask


def add_source_args(p):
    p.add_argument('--task', default='lm', choices=['lm', 'mqar'])
    p.add_argument('--data_dir', default='data/wiki', help='lm: folder with train.bin / val.bin / meta.json')
    p.add_argument('--seq_len', type=int, default=1024, help='lm: training window')
    p.add_argument('--batch', type=int, default=16, help='lm: sequences per batch')
    p.add_argument('--eval_lens', default='1024,4096,16384', help='lm: evaluation lengths of the final report')
    p.add_argument('--val_windows', type=int, default=24)
    p.add_argument('--stages', default='256:16:3000,1024:16:1500', help='mqar: T:n_kv:steps,... (curriculum over context length)')
    p.add_argument('--tokens_per_batch', type=int, default=16384, help='mqar: batch = this // T sequences')
    p.add_argument('--vocab', type=int, default=8192, help='mqar: vocabulary size')
    p.add_argument('--eval_grid', default='256:16,1024:16,4096:16,16384:16', help='mqar: T:n_kv grid of the final report')
    p.add_argument('--eval_n', type=int, default=32, help='mqar: sequences per evaluation cell')


def task_defaults(task):
    """architecture and tree-size defaults per task (the v12 values of each task)."""
    if task == 'mqar':
        return dict(n_layer=4, n_head=4, n_embd=256, wmax=16, beam_classes='2,3,4,6,8', max_keys=40, q_chunk=1024)
    return dict(n_layer=8, n_head=6, n_embd=384, wmax=32, beam_classes='2,4,8,16,32', max_keys=96, q_chunk=256)


def make_source(a, seed):
    if a.task == 'mqar':
        stages = [tuple(int(v) for v in sp.split(':')) for sp in a.stages.split(',')]
        grid = [tuple(int(v) for v in g.split(':')) for g in a.eval_grid.split(',')]
        return MQARSource(a.vocab, stages, a.tokens_per_batch, seed, grid, a.eval_n)
    return LMSource(a.data_dir, a.seq_len, a.batch, seed, [int(v) for v in a.eval_lens.split(',')], a.val_windows)


def _ce_acc(fwd, x, y):
    lg = fwd(x)
    ce = torch.nn.functional.cross_entropy(lg.float().view(-1, lg.size(-1)), y.reshape(-1), reduction='none', ignore_index=-100)
    return ce, (lg.argmax(-1).view(-1) == y.reshape(-1)).float()


# ------------------------------------------------------------------------------------------------------------- language
class LMSource:
    kind = 'lm'
    total_steps = None                                     # the scripts supply their own default
    focus_rows = False

    def __init__(self, data_dir, seq_len, batch, seed, eval_lens, val_windows_n):
        self.meta = json.load(open(os.path.join(data_dir, 'meta.json')))
        self.vocab, self.seq_len, self.batch, self.seed = self.meta['vocab'], seq_len, batch, seed
        self.eval_lens, self.nval = eval_lens, val_windows_n
        self.cross = bool(self.meta.get('cross_docs', False))
        self.tr = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
        self.va = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')

    def train_batch(self, step, dev, mixed=False):
        return train_batch(self.tr, self.batch, self.seq_len, step, self.seed, dev)

    def val_batches(self, dev, n=16, T=None):
        T = T or self.seq_len
        return [window(self.va, s, T, dev) for s in val_windows(self.va, self.meta['eot'], T, n, cross=self.cross)]

    def eval_window(self, T, dev):
        return self.val_batches(dev, 1, T)[0][0]

    @torch.no_grad()
    def _losses(self, fwd, T, n, dev):
        tot = hit = None
        starts = val_windows(self.va, self.meta['eot'], T, n, cross=self.cross)
        for s in starts:
            x, y = window(self.va, s, T, dev)
            ce, ok = _ce_acc(fwd, x, y)
            tot, hit = (ce, ok) if tot is None else (tot + ce, hit + ok)
        return tot / len(starts), hit / len(starts), len(starts)

    def score(self, fwd, dev):
        """the number phase 1 stops on: next-token accuracy on held-out windows of the training length."""
        return self._losses(fwd, self.seq_len, 16, dev)[1].mean().item()

    def quick(self, fwd, dev):
        ce, ac, n = self._losses(fwd, self.seq_len, 16, dev)
        return 'T=%d | loss %.3f acc %.3f (%d windows)' % (self.seq_len, ce.mean().item(), ac.mean().item(), n)

    def final(self, fwd, dev):
        lines = []
        for T in self.eval_lens:
            try:
                ce, ac, n = self._losses(fwd, T, self.nval, dev)
                lines.append('T=%-6d (%d windows) %s' % (T, n, bucket_line(ce, ac, T, self.seq_len)))
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                lines.append('T=%-6d out of memory' % T)
        return lines


# ------------------------------------------------------------------------------------------------------------------ MQAR
class MQARSource:
    kind = 'mqar'
    focus_rows = True                                      # reference rows are drawn at the answer positions (y != -100)

    def __init__(self, vocab, stages, tokens_per_batch, seed, grid, eval_n):
        self.vocab, self.stages, self.tpb, self.seed, self.grid, self.eval_n = vocab, stages, tokens_per_batch, seed, grid, eval_n
        self.seq_len = max(T for T, _, _ in stages)
        self.total_steps = sum(s for _, _, s in stages)

    def stage_at(self, step, mixed=False):
        if mixed:                                          # phase 2: every stage in turn (it may stop long before the last stage)
            T, nkv, _ = self.stages[step % len(self.stages)]
            return T, nkv
        k = step
        for T, nkv, n in self.stages:
            if k < n:
                return T, nkv
            k -= n
        return self.stages[-1][0], self.stages[-1][1]

    def train_batch(self, step, dev, mixed=False):
        T, nkv = self.stage_at(step, mixed)
        x, y, _ = MQARTask(self.vocab, seed=self.seed * 1_000_003 + step).sample(max(1, self.tpb // T), T, nkv)
        return x.to(dev), y.to(dev)

    def val_batches(self, dev, n=2, T=None):
        out = []
        for i, (T_, nkv, _) in enumerate(self.stages):
            x, y, _ = MQARTask(self.vocab, seed=777_000_001 + i).sample(max(1, min(n, self.tpb // T_)), T_, nkv)
            out.append((x.to(dev), y.to(dev)))
        return out

    def eval_window(self, T, dev):
        nkv = self.stages[-1][1]
        x, _, _ = MQARTask(self.vocab, seed=777_000_123).sample(1, T, nkv)
        return x.to(dev)

    @torch.no_grad()
    def accuracy(self, fwd, T, nkv, n, dev, fresh=False):
        # fresh=False: the sequences phase 1 stops on and the phase-3 monitor watches. fresh=True: the final / inference report, sequences
        # (seed 999M+T) that were never used for training, stopping, selection or monitoring.
        task = MQARTask(self.vocab, seed=(999_000_001 if fresh else 888_000_001) + T)
        corr = tot = 0
        bs = 1 if T > 4096 else 2
        for _ in range(0, n, bs):
            x, y, _ = task.sample(bs, T, nkv)
            lg = fwd(x.to(dev))
            m = y != -100
            pred = lg.argmax(-1).cpu()
            corr += ((pred == y) & m).sum().item()
            tot += m.sum().item()
        return corr / max(tot, 1)

    def score(self, fwd, dev):
        """the number phase 1 stops on: answer accuracy at the LAST (longest) stage, so the curriculum has to be completed."""
        T, nkv, _ = self.stages[-1]
        return self.accuracy(fwd, T, nkv, self.eval_n, dev)

    def quick(self, fwd, dev):
        T, nkv, _ = self.stages[-1]
        return 'T=%d n_kv=%d | ACC %.3f' % (T, nkv, self.accuracy(fwd, T, nkv, self.eval_n, dev))

    def final(self, fwd, dev):
        lines = ['(MQAR accuracy at the answer positions, on fresh held-out sequences never used in training or stopping; chance ~ 1/%d)' % (self.vocab // 2)]
        for T, nkv in self.grid:
            try:
                lines.append('T=%6d n_kv=%3d | acc %.3f' % (T, nkv, self.accuracy(fwd, T, nkv, self.eval_n, dev, fresh=True)))
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                lines.append('T=%6d n_kv=%3d | out of memory' % (T, nkv))
        return lines
