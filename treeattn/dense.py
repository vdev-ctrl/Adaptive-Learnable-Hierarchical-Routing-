"""The dense head: ordinary full causal softmax attention. It exists for TRAINING only: phase 1 trains it together with the FFN,
phase 2 uses the frozen result as the reference for the tree. It has the same parameters (qkv, proj), the same head layout and the
same RoPE as the tree attention, so the two produce comparable attention matrices and the tree can be warm-started from it.
A model built with attn='tree' contains no DenseAttention at all."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rope import rotary


class DenseAttention(nn.Module):
    def __init__(self, cfg, layer=0):
        super().__init__()
        self.cfg, self.layer = cfg, layer
        C, H = cfg.n_embd, cfg.n_head
        self.H, self.dh, self.rot = H, cfg.head_dim, cfg.rot_dims
        assert C % H == 0 and cfg.content_dim > 0 and self.rot % 2 == 0
        self.qkv = nn.Linear(C, 3 * C, bias=False)
        self.proj = nn.Linear(C, C, bias=False)
        self.aux = {}

    def forward(self, x, cache=None, ref=None, rows=None):
        """rows (b,R) long: also return the softmax attention of those query rows, (b,H,R,T) float16 (the reference)."""
        assert cache is None and ref is None, 'the dense head has no cache and takes no reference'
        b, T, C = x.shape
        H, dh = self.H, self.dh
        q, k, v = [t.view(b, T, H, dh).transpose(1, 2) for t in self.qkv(x).split(C, 2)]
        pos = torch.arange(T, device=x.device)
        q, k = rotary(q, pos, self.rot, self.cfg.rope_base), rotary(k, pos, self.rot, self.cfg.rope_base)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = self.proj(y.transpose(1, 2).reshape(b, T, C))
        if rows is None:
            return out
        R = rows.shape[1]
        qr = q.gather(2, rows.view(b, 1, R, 1).expand(b, H, R, dh))
        sc = (qr @ k.transpose(-1, -2)).float() / math.sqrt(dh)         # (b,H,R,T)
        allowed = pos.view(1, 1, 1, T) <= rows.view(b, 1, R, 1)
        return out, torch.softmax(sc.masked_fill(~allowed, float('-inf')), -1).to(torch.float16)
