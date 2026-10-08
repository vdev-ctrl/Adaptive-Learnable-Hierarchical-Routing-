"""Losses that use the dense head's attention matrix as the reference (phase 2 of training; never used at inference).

The reference for a window is the frozen dense model's attention: for R sampled query rows per window, the full softmax row
over all keys, for every layer and head (`probs` (b, H, R, T), `rows` (b, R) = the query positions). A complete T x T matrix per
head and layer is far too large to store for a training set, so it is extracted on demand from the frozen phase-1 checkpoint;
the numbers are identical to a stored matrix because that model is frozen and deterministic.

Static keys (the sink at position 0 and the last `local` positions) are always read by the tree, so they are removed from the
reference: the tree only has to FIND the rest. Two losses come out of what is left:

  split_loss   at every tree level, the scores q.S_node/n over the nodes lying fully in the query's past must follow the
               dense attention mass inside each node (cross-entropy to the normalised mass). Trains the QK splitter (q, k).
  budget_labels the fewest leaf groups (2 keys each) that hold `ref_mass` of the non-static dense attention, as a class index.
               Trains the budget predictor.
"""
import math

import torch

from .tree import node_scale


def static_zeroed(probs, rows, local):
    """probs (b,H,R,T), rows (b,R) -> float copy with the always-read keys (sink, t, t-1, ..., t-local+1) set to 0."""
    T = probs.shape[-1]
    j = torch.arange(T, device=probs.device).view(1, 1, T)
    static = (j == 0) | (j > rows.unsqueeze(-1) - local)               # (b,R,T)
    return probs.float().masked_fill(static.unsqueeze(1), 0.0)


def soft_gate(m0, tau, soft=0.5):
    """Soft relative gate on the static-zeroed reference m0 (b,H,R,T). Per row, each key's mass is divided by the row's largest
    non-static mass (r in [0,1]) and multiplied by a smoothstep gate that is 0 below tau*(1-soft), 1 above tau*(1+soft).
    Clearly linked keys (large relative to the row's peak) keep their full mass; the diffuse low-level tail is faded out.
    A row with no peak (flat attention) keeps everything, so only rows that HAVE something to find get sharpened.
    tau <= 0 switches the gate off. Used for the training targets only (split loss, budget labels); the similarity numbers
    are always measured on the raw m0. No gradient flows here (the reference is frozen)."""
    if tau <= 0:
        return m0
    r = m0 / m0.amax(-1, keepdim=True).clamp(min=1e-12)
    lo, hi = tau * (1.0 - soft), tau * (1.0 + soft)
    g = ((r - lo) / max(hi - lo, 1e-9)).clamp(0.0, 1.0)
    return m0 * (g * g * (3.0 - 2.0 * g))


def topk_gate(m0, k, progress, temp0=0.5, temp1=0.01):
    """Annealed top-k gate on the static-zeroed reference m0 (b,H,R,T): keep the k keys with the most mass per row.
    progress in [0,1] is the annealing state. Below 1 the gate is soft: sigmoid((m - v_k) / (temp * row peak)) with v_k the k-th
    largest mass of the row and temp falling geometrically from temp0 to temp1, so at the start (temp0 large) it is close to a
    uniform scale (targets nearly unchanged) and it sharpens towards a hard cut. At progress >= 1 it is the exact hard gate: keys
    outside the row's top k (and keys with no mass) get 0, the rest keep their mass. Used for the training targets only; the
    similarity numbers are measured on the raw m0. The reference is frozen, so no gradient flows here."""
    k = max(1, min(int(k), m0.shape[-1]))
    vk = m0.topk(k, dim=-1).values[..., -1:]                         # (b,H,R,1) the k-th largest mass of each row
    if progress >= 1.0:
        return m0 * ((m0 >= vk) & (m0 > 0)).to(m0.dtype)
    temp = temp0 * (temp1 / temp0) ** max(progress, 0.0)
    peak = m0.amax(-1, keepdim=True).clamp(min=1e-12)
    return m0 * torch.sigmoid((m0 - vk) / (temp * peak))


def topk_positions(m0, k):
    """m0 (b,H,R,T) static-zeroed reference -> (b,H,R,T) float 0/1 map of the k keys with the most dense mass per row (their
    positions only; the mass values are dropped). Keys with no mass are never marked, so a row with fewer than k attended keys
    marks fewer than k. The reference is frozen: no gradient flows here."""
    k = max(1, min(int(k), m0.shape[-1]))
    top = m0.topk(k, dim=-1)
    hot = torch.zeros_like(m0)
    hot.scatter_(-1, top.indices, (top.values > 0).to(m0.dtype))
    return hot


def gate_targets(m0, cfg, hot=None):
    """The reference the phase-2 losses train on: m0 passed through the configured gate ('topk', 'soft' or 'off')."""
    mode = getattr(cfg, 'gate_mode', 'topk')
    if mode == 'topk_mix':                                              # progress p: 0 = raw dense rows, 1 = top-k positions only
        hot = hot if hot is not None else topk_positions(m0, cfg.gate_k)
        p = min(1.0, max(0.0, getattr(cfg, 'gate_progress', 1.0)))
        raw = m0 * (hot.sum(-1, keepdim=True) / m0.sum(-1, keepdim=True).clamp(min=1e-12))   # raw rows rescaled to the top-k row weight
        return (1.0 - p) * raw + p * hot
    if mode == 'topk_pos':                                              # only the top-k keys and their positions, equal weight
        return hot if hot is not None else topk_positions(m0, cfg.gate_k)
    if mode == 'topk':
        return topk_gate(m0, cfg.gate_k, getattr(cfg, 'gate_progress', 1.0), cfg.gate_temp0, cfg.gate_temp1)
    if mode == 'soft':
        return soft_gate(m0, cfg.gate_tau, cfg.gate_soft)
    return m0


def budget_labels(m0, ref_mass, classes):
    """m0 (b,H,R,T) static-zeroed reference. -> (class index (b,H,R), weight (b,H,R) = non-static share of the attention row)."""
    b, H, R, T = m0.shape
    if T % 2:
        m0 = torch.nn.functional.pad(m0, (0, 1))
    g = m0.reshape(b, H, R, -1, 2).sum(-1)                              # mass per leaf group of 2 keys
    total = g.sum(-1)
    gs, _ = g.sort(-1, descending=True)
    need = (gs.cumsum(-1) < ref_mass * total.unsqueeze(-1)).sum(-1) + 1
    label = (need.unsqueeze(-1) > classes).sum(-1).clamp(max=len(classes) - 1)
    return label, total


def split_loss(q_rows, S, rows, m0, norm='mean'):
    """q_rows (b,H,R,dc) content queries WITH grad, S tree levels WITH grad (S[l]: (b,H,tpad>>l,dc)), rows (b,R),
    m0 (b,H,R,T) static-zeroed reference. At level l the candidates are the nodes fully in the past of the row (node m is, iff
    m < t >> l). Loss = cross-entropy(normalised reference mass per node, softmax of the node scores), weighted by the
    reference mass that lies in those nodes, so rows / levels with nothing to find carry no weight."""
    with torch.autocast(q_rows.device.type, enabled=False):          # scores in fp32 (node sums can be large)
        b, H, R, dc = q_rows.shape
        tpad = S[1].shape[2] * 2
        mass = torch.nn.functional.pad(m0, (0, tpad - m0.shape[-1]))        # (b,H,R,tpad)
        tot = q_rows.new_zeros((), dtype=torch.float32)
        den = q_rows.new_zeros((), dtype=torch.float32)
        qf = q_rows.float()
        for l in range(1, len(S)):
            mass = mass.reshape(b, H, R, -1, 2).sum(-1)                     # (b,H,R,tpad>>l)
            M = mass.shape[-1]
            done = torch.arange(M, device=q_rows.device).view(1, 1, 1, M) < (rows >> l).view(b, 1, R, 1)
            mc = mass.masked_fill(~done, 0.0)
            w = mc.sum(-1)                                                  # (b,H,R)
            target = mc / w.clamp(min=1e-9).unsqueeze(-1)
            sc = torch.einsum('bhrd,bhmd->bhrm', qf, S[l].float()) / (node_scale(l, norm) * math.sqrt(dc))
            sc = torch.nan_to_num(sc, nan=0.0, posinf=1e4, neginf=-1e4).clamp(-1e4, 1e4).masked_fill(~done, -1e4)
            ce = -(target * torch.log_softmax(sc, -1)).sum(-1)              # (b,H,R)
            tot = tot + (w * ce).sum()
            den = den + w.sum()
    return tot / den.clamp(min=1e-6)


def sample_rows(b, T, R, min_pos, step, seed, dev, focus=None):
    """(b,R) query positions for the batch at `step`: a pure function of (seed, step), uniform in [min_pos, T).
    focus (b,T) bool, optional: half of the rows are drawn from the positions where it is True (e.g. the MQAR answer positions,
    where retrieval happens); rows of a sequence without any focus position stay uniform."""
    g = torch.Generator().manual_seed(seed * 1_000_003 + step)
    lo = min(min_pos, T - 1)
    rows = torch.randint(lo, T, (b, R), generator=g)
    if focus is not None and R >= 2:
        w = focus.detach().cpu().float().clone()
        w[:, :lo] = 0.0
        none = w.sum(1) == 0
        w[none, lo:] = 1.0
        rows[:, :R // 2] = torch.multinomial(w, R // 2, replacement=True, generator=g)
    return rows.to(dev)
