"""MQAR (multi-query associative recall) generator, Zoology-style. Used as a synthetic task source for the same three-phase pipeline."""
import numpy as np
import torch


class MQARTask:
    """Multi-query associative recall (Arora et al., "Zoology"), generated as in the published recipe:
    n_kv unique key->value pairs first (keys from the lower half of the vocabulary, values from the upper half), then
    every key is queried once at a power-law-distributed gap among random filler tokens; the model must output the
    value right after each queried key. Loss only on those answer positions.
    route = (bidx, qpos, tgt) with one row per query: qpos = position of the queried key, tgt = position of its value."""

    def __init__(self, vocab_size=8192, power_a=0.01, seed=0):
        self.V, self.a = vocab_size, power_a
        self.rng = np.random.default_rng(seed)

    def sample(self, b, T, n_kv):
        V, rng = self.V, self.rng
        ctx = 2 * n_kv
        space = (T - ctx) // 2
        assert T % 2 == 0 and space >= n_kv, 'need T even and (T - 2 n_kv) / 2 >= n_kv'
        p = self.a * np.arange(1, space + 1) ** (self.a - 1.0)
        p = p / p.sum()
        X, Y, bi, qp, tg = [], [], [], [], []
        for j in range(b):
            keys = rng.choice(np.arange(1, V // 2), n_kv, replace=False)
            vals = rng.choice(np.arange(V // 2, V), n_kv, replace=False)
            kvs = np.zeros(ctx, dtype=np.int64)
            kvs[0::2], kvs[1::2] = keys, vals
            gaps = rng.choice(space, size=n_kv, replace=False, p=p)
            queries = np.zeros(T - ctx + 1, dtype=np.int64)
            queries[gaps * 2] = keys
            labels = np.full(T + 1, -100, dtype=np.int64)
            labels[gaps * 2 + ctx + 1] = vals
            ex = np.concatenate([kvs, queries])
            x, y = ex[:-1].copy(), labels[1:]
            zero = x == 0
            x[zero] = rng.integers(1, V, size=int(zero.sum()))
            X.append(x), Y.append(y)
            bi += [j] * n_kv
            qp += list(gaps * 2 + ctx)
            tg += list(2 * np.arange(n_kv) + 1)
        return (torch.from_numpy(np.stack(X)), torch.from_numpy(np.stack(Y)),
                (torch.tensor(bi), torch.tensor(qp), torch.tensor(tg)))
