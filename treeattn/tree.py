"""Positional binary (segment) tree over the keys, causal descent, neighbour search.

Level l has Tpad >> l nodes, node m covers positions [m * 2**l, (m + 1) * 2**l). A parent is the SUM of its two
children (the index key vector). Padding with zero vectors goes at the END (future side), so it never touches a
past-only node; for a causal tree this is equivalent to padding "uniformly".

Causality (the "fractal" rule): for a query at position t, the nodes that lie completely in its past are exactly the
left siblings along the root-to-t path: at every level l where bit l of t is 1, node (t >> l) - 1. Together with the
static keys they cover [0, t). No node that contains t or anything later is ever read.
All functions are fully parallel over queries and have bounded work per query.
"""
import math

import torch
import torch.nn.functional as F


def node_scale(l, norm):
    """divisor of a node's score q.S: 'mean' = n (mean key), 'sqrt' = sqrt(n) (constant noise level), 'none' = 1 (raw sum, boost proportional to n)."""
    return {'mean': float(1 << l), 'sqrt': math.sqrt(1 << l), 'none': 1.0}[norm]


def next_pow2(n):
    return 1 << max(2, (n - 1).bit_length())


def n_levels(tpad):
    return int(math.log2(tpad)) - 1


def build_levels(kidx, tpad):
    """kidx (b, H, T, d) -> list S, S[l] (b, H, tpad >> l, d) for l = 1..Lm (S[0] is None). O(T) total."""
    b, H, T, d = kidx.shape
    if T < tpad:
        kidx = F.pad(kidx, (0, 0, 0, tpad - T))
    S, x = [None], kidx
    for _ in range(n_levels(tpad)):
        x = x.reshape(b, H, x.shape[2] // 2, 2, d).sum(3)
        S.append(x)
    return S


def gather_nodes(S, ids):
    """S (b, H, M, d), ids (b, H, N, C) long (>= 0) -> (b, H, N, C, d)."""
    b, H, N, C = ids.shape
    d = S.shape[-1]
    return S.gather(2, ids.reshape(b, H, N * C, 1).expand(b, H, N * C, d)).reshape(b, H, N, C, d)


@torch.no_grad()
def tree_descent(q, S, tq, wmax, norm='mean'):
    """Beam descent over the past-only subtrees of every query.
    q (b, H, Nq, d) float, tq (Nq,) positions. At each level the candidates are the children of the nodes kept so far
    plus the newly available left sibling of that level; the best wmax by (q . mean key) are kept.
    Returns leaf ids (level 1) (b, H, Nq, w), validity, scores; best first."""
    b, H, Nq, _ = q.shape
    ids = vld = sct = None
    for l in range(len(S) - 1, 0, -1):
        bit = ((tq >> l) & 1) == 1
        inj = ((tq >> l) - 1).clamp(min=0).view(1, 1, Nq, 1).expand(b, H, Nq, 1)
        injv = bit.view(1, 1, Nq, 1).expand(b, H, Nq, 1)
        if ids is None:
            cand, cv = inj, injv
        else:
            cand = torch.cat([2 * ids, 2 * ids + 1, inj], -1)
            cv = torch.cat([vld, vld, injv], -1)
        sc = torch.einsum('bhnd,bhncd->bhnc', q, gather_nodes(S[l], cand)) / node_scale(l, norm)
        sc = sc.masked_fill(~cv, float('-inf'))
        sct, order = sc.topk(min(wmax, cand.shape[-1]), dim=-1)
        ids, vld = cand.gather(-1, order), cv.gather(-1, order)
    return ids, vld, sct


@torch.no_grad()
def neighbor_search(q, qbuf, S, tq, tops, cfg):
    """Borrow the best leaves that the nearest earlier queries found; keep the ones this query also scores well on.
    The baseline is the earlier query's own score on the same group. The scan width (cap) grows with the match rate
    of the nearest earlier queries, so far-away matches get a small cap.
    q (b,H,Nq,d); qbuf (b,H,Tbuf,d) content queries of earlier positions; tops (b,H,Tbuf,NS) stored leaf ids (-1 = none).
    Returns group ids, match mask (b,H,Nq,Wn*NS) and the distinct matched groups in leaf units nL (b,H,Nq)."""
    b, H, Nq, d = q.shape
    Wn, NS, lv = cfg.nb_window, cfg.nb_store, cfg.nb_level
    dev = q.device
    jpos = tq.view(Nq, 1) - torch.arange(1, Wn + 1, device=dev).view(1, Wn)
    jval = jpos >= 0
    jc = jpos.clamp(min=0)
    leaf = tops[:, :, jc]                                     # (b,H,Nq,Wn,NS)
    ok = (leaf >= 0) & jval.view(1, 1, Nq, Wn, 1)
    grp = (leaf.clamp(min=0) >> (lv - 1)).reshape(b, H, Nq, Wn * NS)
    ok = ok.reshape(b, H, Nq, Wn * NS)
    ok = ok & (((grp + 1) << lv) <= tq.view(1, 1, Nq, 1))     # group fully in the past of this query
    U = gather_nodes(S[lv], grp.clamp(max=S[lv].shape[2] - 1)) / node_scale(lv, cfg.node_norm)
    s_new = torch.einsum('bhnd,bhncd->bhnc', q, U)
    qj = qbuf[:, :, jc]                                       # (b,H,Nq,Wn,d)
    s_old = torch.einsum('bhnwd,bhnwsd->bhnws', qj, U.reshape(b, H, Nq, Wn, NS, d)).reshape(b, H, Nq, Wn * NS)
    match = ok & (s_new >= s_old - cfg.nb_rho * s_old.abs())
    probe = cfg.nb_probe * NS
    rate = match[..., :probe].float().sum(-1) / ok[..., :probe].float().sum(-1).clamp(min=1)
    cap = cfg.nb_cap_min + torch.round((Wn - cfg.nb_cap_min) * rate).long()
    didx = torch.arange(Wn * NS, device=dev) // NS + 1
    match = (match & (didx <= cap.unsqueeze(-1))) if cfg.neighbors else torch.zeros_like(match)
    gm = torch.where(match, grp, torch.full_like(grp, -1))
    ss, _ = gm.sort(-1)
    vs = ss >= 0
    distinct = vs.sum(-1) - ((ss[..., 1:] == ss[..., :-1]) & vs[..., 1:]).sum(-1)
    return grp, match, distinct * (1 << (lv - 1))


def split_budget(lhat, nL, wmax):
    """lhat = predicted leaf groups needed. Neighbour matches count first; the tree fills the rest (at least 1,
    at least 2 if the neighbours found fewer than half of the budget). Returns the tree leaves to use."""
    need = (lhat - nL).clamp(min=0)
    floor_b = torch.where(nL < (lhat + 1) // 2, torch.full_like(lhat, 2), torch.ones_like(lhat))
    return torch.where(nL >= lhat, torch.zeros_like(lhat), torch.maximum(need, floor_b)).clamp(max=wmax)


def select_unique(cand, valid, kmax):
    """Remove duplicate positions and keep the kmax highest-priority ones (priority = column order).
    cand/valid (b,H,N,C) -> pos (b,H,N,k), ok (b,H,N,k)."""
    big = 1 << 30
    pos = torch.where(valid, cand, torch.full_like(cand, big))
    sp, order = pos.sort(dim=-1, stable=True)
    dup = torch.zeros_like(valid)
    dup[..., 1:] = sp[..., 1:] == sp[..., :-1]
    keep = torch.zeros_like(valid).scatter_(-1, order, (sp < big) & ~dup)
    C = cand.shape[-1]
    pr = torch.where(keep, torch.arange(C, device=cand.device).expand_as(cand), torch.full_like(cand, big))
    prs, o2 = pr.sort(-1)
    k = min(kmax, C)
    return cand.gather(-1, o2[..., :k]), prs[..., :k] < big
