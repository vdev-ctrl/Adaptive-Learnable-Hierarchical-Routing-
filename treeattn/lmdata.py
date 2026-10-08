"""Data helpers shared by the dense and the tree training scripts and the evaluation (identical batches and validation windows)."""

import numpy as np
import torch

EDGES = [0, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]


def train_batch(data, b, T, step, seed, dev):
    """The batch for a given global step is a pure function of (seed, step): every model sees the same data in the same order."""
    rng = np.random.default_rng(seed * 1_000_003 + step)
    ix = rng.integers(0, len(data) - T - 1, size=b)
    x = np.stack([data[i:i + T] for i in ix]).astype(np.int64)
    y = np.stack([data[i + 1:i + 1 + T] for i in ix]).astype(np.int64)
    return torch.from_numpy(x).to(dev), torch.from_numpy(y).to(dev)


def val_windows(val, eot, T, n, seed=123, cross=False):
    """n deterministic window starts. cross=False: the T+1 tokens contain no end-of-document token (windows never cross a book).
    cross=True (Wikipedia: articles are short, windows span several): any start is allowed."""
    eots = np.array([], dtype=np.int64) if cross else np.flatnonzero(np.asarray(val) == eot)
    rng = np.random.default_rng(seed + T)
    starts, tries = [], 0
    while len(starts) < n and tries < 200000:
        tries += 1
        s = int(rng.integers(0, len(val) - T - 1))
        j = int(np.searchsorted(eots, s))
        if j < len(eots) and eots[j] <= s + T:
            continue
        starts.append(s)
    return starts


def window(val, s, T, dev):
    x = torch.from_numpy(np.asarray(val[s:s + T]).astype(np.int64))[None].to(dev)
    y = torch.from_numpy(np.asarray(val[s + 1:s + 1 + T]).astype(np.int64))[None].to(dev)
    return x, y


def bucket_line(ce, acc, T, train_len):
    """ce, acc: 1-D tensors, mean loss / next-token top-1 accuracy per position over the windows.
    Prints loss/accuracy per position bucket, the overall numbers (with perplexity) and the share beyond the trained length."""
    ce, acc = ce.cpu(), acc.cpu()
    parts = ['[%d,%d):%.3f/%.3f' % (lo, min(hi, T), ce[lo:hi].mean(), acc[lo:hi].mean()) for lo, hi in zip(EDGES, EDGES[1:]) if lo < T]
    far = ' | beyond train len %d: %.3f/%.3f' % (train_len, ce[train_len:].mean(), acc[train_len:].mean()) if T > train_len else ''
    return '(loss/acc) ' + ' '.join(parts) + ' | all %.3f/%.3f ppl %.0f' % (ce.mean(), acc.mean(), float(ce.mean().exp())) + far
