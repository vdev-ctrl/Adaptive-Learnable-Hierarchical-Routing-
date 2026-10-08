"""Rotary position embedding, shared by the dense head and the tree attention (so both produce comparable attention matrices)."""
import torch


def rotary(x, pos, rot, base):
    """Rotate the first `rot` dims of x (..., T, dh) by the positions `pos` (T,). The remaining dims are position-free."""
    if rot == 0:
        return x
    inv = 1.0 / (base ** (torch.arange(0, rot, 2, device=x.device).float() / rot))
    ang = pos.float().view(-1, 1) * inv.view(1, -1)
    cos, sin = ang.cos().to(x.dtype), ang.sin().to(x.dtype)
    x1, x2, rest = x[..., :rot // 2], x[..., rot // 2:rot], x[..., rot:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos, rest], -1)


def rotate_extra(x, extra, rot, base):
    """Rotate the first `rot` dims of x (..., dh) by an additional per-element position `extra` (x.shape[:-1]).
    Keys are stored rotated by their absolute position j and queries by t, so the score sees R((j - t) * theta).
    Adding extra = max(t - j - cap, 0) to the key makes the score see R(-min(t - j, cap) * theta): relative distance capped."""
    inv = 1.0 / (base ** (torch.arange(0, rot, 2, device=x.device).float() / rot))
    ang = extra.float().unsqueeze(-1) * inv
    cos, sin = ang.cos().to(x.dtype), ang.sin().to(x.dtype)
    x1, x2, rest = x[..., :rot // 2], x[..., rot // 2:rot], x[..., rot:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos, rest], -1)
