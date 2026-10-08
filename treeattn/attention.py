"""Sub-quadratic causal tree attention (the inference attention).

Trainable parts (everything else is parameter-free search):
  1. the splitter: one linear map x -> q, k, v
  2. the budget predictor: a small MLP (query + anti-dilution global vector -> how many leaf groups are needed)

Per query, in order: static keys (sink, self, previous local-1) -> neighbour search (borrow from earlier queries)
-> tree descent for whatever the budget still needs. Exact softmax attention runs only over the (<= max_keys) selected keys.
No T x T matrix exists anywhere in this file. Prefill and decoding use the same functions. Decoding keeps a LayerCache
(keys, values, tree levels, global sum, stored leaves): the tree is NOT rebuilt, the new key is added into its log2(T) ancestor
nodes in place, and a token costs O(wmax * log T + max_keys) whatever the context length.

Position handling: the first `rot_dims` dims of every head are rotated (RoPE) and used by exact attention only; the
remaining content dims are what the tree indexes. The tree therefore sums position-free content vectors.

Training against the dense reference (phase 2): `ref` = (rows, probs) is the dense head's attention at sampled query rows
(see distill.py). It adds two losses (aux['split'], aux['pred']) and the similarity numbers aux['mass'] (share of ALL dense
attention mass on the keys this layer read) and aux['far'] (the same for the non-static mass: what the tree has to find).
Without `ref` (inference, phase 3) none of that code runs.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .distill import budget_labels, gate_targets, split_loss, static_zeroed, topk_positions
from .rope import rotary, rotate_extra
from .tree import (build_levels, gather_nodes, neighbor_search, next_pow2, select_unique, split_budget, tree_descent)


class LayerCache:
    """Linear-size state of one layer: everything grows with the context length at most linearly, nothing is ever rebuilt."""

    def __init__(self, cfg, b, tmax, device, dtype):
        H, dh, dc = cfg.n_head, cfg.head_dim, cfg.content_dim
        self.tmax, self.tpad = tmax, next_pow2(tmax)
        self.k = torch.zeros(b, H, tmax, dh, device=device, dtype=dtype)
        self.v = torch.zeros(b, H, tmax, dh, device=device, dtype=dtype)
        self.qc = torch.zeros(b, H, tmax, dc, device=device)
        self.S = build_levels(torch.zeros(b, H, self.tpad, dc, device=device), self.tpad)
        self.G = torch.zeros(b, H, dc, device=device)
        self.tops = torch.full((b, H, tmax, cfg.nb_store), -1, dtype=torch.long, device=device)
        self.t = 0


class TreeAttention(nn.Module):
    def __init__(self, cfg, layer=0):
        super().__init__()
        self.cfg, self.layer = cfg, layer
        C, H = cfg.n_embd, cfg.n_head
        self.H, self.dh, self.dc, self.rot = H, cfg.head_dim, cfg.content_dim, cfg.rot_dims
        assert C % H == 0 and self.dc > 0 and self.rot % 2 == 0
        assert cfg.wmax >= cfg.nb_store and max(cfg.beam_classes) <= cfg.wmax
        self.qkv = nn.Linear(C, 3 * C, bias=False)          # the splitter
        self.proj = nn.Linear(C, C, bias=False)
        if cfg.budget_mode == 'pred':                       # the budget predictor
            fin = self.dc + 1 + H + self.dc + 1
            self.pred = nn.Sequential(nn.Linear(fin, cfg.pred_hidden), nn.ReLU(),
                                      nn.Linear(cfg.pred_hidden, len(cfg.beam_classes)))
            self.register_buffer('classes', torch.tensor(list(cfg.beam_classes)), persistent=False)
            self.register_buffer('head_eye', torch.eye(H), persistent=False)
        self.aux = {}

    # ------------------------------------------------------------------ budget predictor
    def _budget(self, qc, G):
        """qc (b,H,N,dc) content queries, G (b,H,N,dc) anti-dilution global vector (sum of all earlier index keys).
        Returns (leaf-group budget (b,H,N) long, logits or None)."""
        b, H, N, _ = qc.shape
        if self.cfg.budget_mode != 'pred':
            return torch.full((b, H, N), self.cfg.fixed_budget, dtype=torch.long, device=qc.device), None
        qd, gn = qc.detach().float(), G.detach().float().norm(dim=-1, keepdim=True)
        feats = torch.cat([qd, torch.log(qd.norm(dim=-1, keepdim=True) + 1e-6),
                           self.head_eye.view(1, H, 1, H).expand(b, H, N, H),
                           G.detach().float() / (gn + 1e-6), torch.log1p(gn) / 10.0], -1)
        logits = self.pred(feats)
        return self.classes[logits.argmax(-1).detach()], logits

    # ------------------------------------------------------------------ exact attention over selected keys
    def _exact(self, q, kbuf, vbuf, pos, ok, tqc=None):
        p = pos.clamp(0, kbuf.shape[2] - 1)                     # masked slots may point past the buffer
        kg, vg = gather_nodes(kbuf, p), gather_nodes(vbuf, p)
        cap = self.cfg.rope_cap
        if cap > 0 and self.rot > 0 and tqc is not None:         # optional: no key looks farther away than `cap` positions
            kg = rotate_extra(kg, (tqc.view(1, 1, -1, 1) - p - cap).clamp(min=0), self.rot, self.cfg.rope_base)
        s = torch.einsum('bhnd,bhnkd->bhnk', q, kg).float() / math.sqrt(q.shape[-1])
        w = torch.softmax(s.masked_fill(~ok, float('-inf')), -1).to(vg.dtype)
        return torch.einsum('bhnk,bhnkd->bhnd', w, vg)

    # ------------------------------------------------------------------ forward
    def forward(self, x, cache=None, ref=None, select_only=False):
        """select_only=True (phase 2): run the selection and the reference numbers/losses only, skip the exact attention, return None."""
        cfg = self.cfg
        b, T, C = x.shape
        H, dh, dc, rot = self.H, self.dh, self.dc, self.rot
        dev = x.device
        self.aux = {}
        q, k, v = [t.view(b, T, H, dh).transpose(1, 2) for t in self.qkv(x).split(C, 2)]
        t0 = cache.t if cache is not None else 0
        decode = cache is not None and t0 > 0
        assert not decode or T == 1, 'cached decoding takes one token at a time (prefill the prompt first)'
        tq = torch.arange(t0, t0 + T, device=dev)
        q, k = rotary(q, tq, rot, cfg.rope_base), rotary(k, tq, rot, cfg.rope_base)

        # ---- buffers visible to this call
        if cache is not None:
            cache.k[:, :, t0:t0 + T], cache.v[:, :, t0:t0 + T] = k.to(cache.k.dtype), v.to(cache.v.dtype)
            cache.qc[:, :, t0:t0 + T] = q[..., rot:].detach().float()
            kbuf, vbuf, qbuf = cache.k[:, :, :t0 + T], cache.v[:, :, :t0 + T], cache.qc[:, :, :t0 + T]
        else:
            kbuf, vbuf, qbuf = k, v, q[..., rot:].detach().float()

        # ---- index keys: the content dims of k (the sink at position 0 is never indexed)
        kidx = torch.nan_to_num(k[..., rot:].float(), nan=0.0, posinf=1e4, neginf=-1e4)
        kidx = kidx * (tq != 0).view(1, 1, T, 1).to(kidx.dtype)

        # ---- tree levels + anti-dilution global vector
        use_ref = ref is not None and not decode                    # extract the reference-based numbers
        use_loss = use_ref and self.training                         # ... and train on them
        if decode:
            for l in range(1, len(cache.S)):                         # the new key goes into its ancestors in place: no rebuild
                cache.S[l][:, :, t0 >> l] += kidx[:, :, 0]
            S, G = cache.S, cache.G.view(b, H, 1, dc).clone()
            cache.G += kidx[:, :, 0]
        else:
            tpad = cache.tpad if cache is not None else next_pow2(T)
            with torch.set_grad_enabled(use_loss):
                S = build_levels(kidx, tpad)
            kd = kidx.detach()
            G = kd.cumsum(2) - kd
            if cache is not None:
                cache.S = [None] + [s_.detach().clone() for s_ in S[1:]]
                cache.G = kd.sum(2)
        Sd = [None] + [s_.detach() for s_ in S[1:]]

        # ---- budget predictor
        lhat, plog = self._budget(qbuf[:, :, t0:t0 + T], G)
        if plog is not None and not decode:                          # share of queries per budget class (for the log)
            self.aux['chist'] = (plog.argmax(-1).unsqueeze(-1) == torch.arange(plog.shape[-1], device=dev)).float().mean((0, 1, 2)).detach()

        # ---- pass 1: tree descent for every query; each query stores its best leaves for its neighbours
        NS, chunk = cfg.nb_store, cfg.q_chunk
        spans = [(s0, min(s0 + chunk, T)) for s0 in range(0, T, chunk)]
        qf = qbuf[:, :, t0:t0 + T]
        tops = cache.tops if cache is not None else torch.full((b, H, T, NS), -1, dtype=torch.long, device=dev)
        found = []
        for s0, s1 in spans:
            ids, vld, _ = tree_descent(qf[:, :, s0:s1], Sd, tq[s0:s1], cfg.wmax, cfg.node_norm)
            found.append((ids, vld))
            tops[:, :, t0 + s0:t0 + s1] = torch.where(vld[..., :NS], ids[..., :NS], torch.full_like(ids[..., :NS], -1))

        if use_ref:
            rows, rprobs = ref
            R = rows.shape[1]
            m0 = static_zeroed(rprobs, rows, cfg.local)             # (b,H,R,T): reference without the always-read keys
            hot = topk_positions(m0, cfg.gate_k)                    # the dense top-k keys of each row (for the hit metric and the targets)
            cap_num, cap_den, far_num, far_den, hit_num, hit_den = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        # ---- pass 2: neighbour search, budget split, key selection, exact attention
        outs, st_keys, st_tree, st_nb = [], 0.0, 0.0, 0.0
        ar_w = torch.arange(found[0][0].shape[-1], device=dev)
        lvk = 1 << cfg.nb_level
        for (s0, s1), (ids, vld) in zip(spans, found):
            n, tqc = s1 - s0, tq[s0:s1]
            grp, match, nL = neighbor_search(qf[:, :, s0:s1], qbuf, Sd, tqc, tops, cfg)
            n_tree = split_budget(lhat[:, :, s0:s1], nL, cfg.wmax)
            leaf_ok = vld & (ar_w[:ids.shape[-1]] < n_tree.unsqueeze(-1))
            tree_pos = (2 * ids.unsqueeze(-1) + torch.arange(2, device=dev)).flatten(-2)
            nb_pos = (grp.unsqueeze(-1) * lvk + torch.arange(lvk, device=dev)).flatten(-2)
            st = (tqc.view(1, 1, n, 1) - torch.arange(cfg.local, device=dev)).expand(b, H, n, cfg.local)
            st = torch.cat([torch.zeros(b, H, n, 1, dtype=torch.long, device=dev), st], -1)
            st_ok = st >= 0
            cand = torch.cat([st, tree_pos, nb_pos], -1)
            valid = torch.cat([st_ok, leaf_ok.repeat_interleave(2, -1), match.repeat_interleave(lvk, -1)], -1)
            pos, ok = select_unique(cand, valid, cfg.max_keys)
            if use_ref:                                              # share of the dense attention mass on the keys read
                inc = (rows >= s0) & (rows < s1)                     # (b,R) rows that fall into this chunk
                ic = (rows - s0).clamp(0, n - 1)
                K = pos.shape[-1]
                gi = ic.view(b, 1, R, 1).expand(b, H, R, K)
                pr, okr = pos.gather(2, gi), ok.gather(2, gi)
                pc = pr.clamp(0, rprobs.shape[-1] - 1)
                m = rprobs.gather(3, pc).float() * okr
                mf = m0.gather(3, pc) * okr
                cap_num = cap_num + (m.sum(-1) * inc.unsqueeze(1)).sum().detach()
                cap_den = cap_den + inc.sum().detach() * H
                far_num = far_num + (mf.sum(-1) * inc.unsqueeze(1)).sum().detach()
                far_den = far_den + (m0.sum(-1) * inc.unsqueeze(1)).sum().detach()
                hit_num = hit_num + ((hot.gather(3, pc) * okr).sum(-1) * inc.unsqueeze(1)).sum().detach()
                hit_den = hit_den + (hot.sum(-1) * inc.unsqueeze(1)).sum().detach()
            if select_only:
                st_keys = st_keys + ok.sum(-1).float().sum()
                st_tree = st_tree + leaf_ok.sum(-1).float().sum()
                st_nb = st_nb + nL.float().sum()
                continue
            qc_rows = q[:, :, s0:s1]
            if self.training and qc_rows.requires_grad:
                outs.append(checkpoint(self._exact, qc_rows, kbuf, vbuf, pos, ok, tqc, use_reentrant=False))
            else:
                outs.append(self._exact(qc_rows, kbuf, vbuf, pos, ok, tqc))
            st_keys = st_keys + ok.sum(-1).float().sum()
            st_tree = st_tree + leaf_ok.sum(-1).float().sum()
            st_nb = st_nb + nL.float().sum()
        denom = float(b * H * T)
        self.aux['keys'] = (st_keys / denom).detach()
        self.aux['leaves_used'] = (st_tree / denom).detach()
        self.aux['nb_leaves'] = (st_nb / denom).detach()

        # ---- losses against the dense reference attention
        if use_ref:
            self.aux['mass'] = cap_num / (cap_den + 1e-6)           # share of ALL dense attention mass on the keys read
            self.aux['hit'] = hit_num / (hit_den + 1e-6)            # share of the dense top-k keys (non-static) that the tree read
            self.aux['far'] = far_num / (far_den + 1e-6)            # same, for the non-static mass only: what the tree has to find
        if use_loss:
            m0t = gate_targets(m0, cfg, hot)         # gated targets; metrics above stay on the raw m0
            if plog is not None:
                label, share = budget_labels(m0t, cfg.ref_mass, self.classes)
                pl = plog.gather(2, rows.view(b, 1, R, 1).expand(b, H, R, plog.shape[-1]))
                ce = F.cross_entropy(pl.float().reshape(-1, plog.shape[-1]), label.reshape(-1), reduction='none')
                self.aux['pred'] = (ce * share.reshape(-1)).sum() / (share.sum() + 1e-6)
            q_rows = q.gather(2, rows.view(b, 1, R, 1).expand(b, H, R, dh))[..., rot:]
            self.aux['split'] = split_loss(q_rows, S, rows, m0t, cfg.node_norm)
        if cache is not None:
            cache.t = t0 + T
        if select_only:
            return None
        out = torch.cat(outs, 2)
        return self.proj(out.transpose(1, 2).reshape(b, T, C))
