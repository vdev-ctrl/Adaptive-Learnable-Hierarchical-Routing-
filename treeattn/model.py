"""GPT with a switchable attention: 'dense' (training only) or 'tree' (training and inference).

Everything except the attention is common to both: token embedding (tied output head), LayerNorms and the FFN have the same
names and shapes, so one phase-1 checkpoint initialises the tree model directly (`load_dense`). The tree model contains no dense
attention module; its `generate` is prefill + cached one-token steps through the tree only."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import LayerCache, TreeAttention
from .dense import DenseAttention


class LayerNorm(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n))

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, None, 1e-5)


class MLP(nn.Module):
    """The FFN, common to the dense and the tree model: expand 4x, GELU, compress."""

    def __init__(self, cfg):
        super().__init__()
        C = cfg.n_embd
        self.fc = nn.Linear(C, 4 * C, bias=False)
        self.proj = nn.Linear(4 * C, C, bias=False)

    def forward(self, x):
        return self.proj(F.gelu(self.fc(x)))


class Block(nn.Module):
    def __init__(self, cfg, layer, attn):
        super().__init__()
        self.ln1 = LayerNorm(cfg.n_embd)
        self.attn = TreeAttention(cfg, layer) if attn == 'tree' else DenseAttention(cfg, layer)
        self.ln2, self.mlp = LayerNorm(cfg.n_embd), MLP(cfg)

    def forward(self, x, cache=None, ref=None):
        x = x + self.attn(self.ln1(x), cache, ref)
        return x + self.mlp(self.ln2(x))


def _f(x):
    return float(x.detach()) if torch.is_tensor(x) else float(x)


class GPT(nn.Module):
    def __init__(self, cfg, attn='tree'):
        super().__init__()
        assert attn in ('tree', 'dense')
        self.cfg, self.attn_kind = cfg, attn
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.blocks = nn.ModuleList([Block(cfg, i, attn) for i in range(cfg.n_layer)])
        self.ln_f = LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight
        self.apply(self._init)
        for n, p in self.named_parameters():
            if n.endswith('proj.weight'):
                nn.init.normal_(p, 0.0, 0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, 0.0, 0.02)

    # ------------------------------------------------------------------ phase 1 -> phase 2
    def load_dense(self, state, copy_attn=True):
        """Initialise this (tree) model from a dense model's state dict. The shared parts (embedding, norms, FFN) are always
        copied. copy_attn=True also copies the dense qkv / proj into the tree's splitter / output projection (warm start: the tree
        starts as 'dense attention restricted to the keys it selects'). Only the budget predictor stays freshly initialised.
        Returns the names of the parameters that were not loaded."""
        sd = {k: v for k, v in state.items() if copy_attn or '.attn.' not in k}
        missing, unexpected = self.load_state_dict(sd, strict=False)
        assert not unexpected, unexpected
        return missing

    @torch.no_grad()
    def extract(self, idx, rows):
        """Dense model only. Run the forward pass and return (hs, probs): per layer, the attention input hs[l] (b,T,C) (= ln1 of the
        layer's residual stream) and the attention rows probs[l] (b,H,R,T) float16 of the query positions `rows` (b,R).
        probs is the 'extracted attention matrix' the tree is trained against; hs lets phase 2 run each tree layer on exactly the
        input the dense layer saw (no FFN and no labels needed on the tree side)."""
        assert self.attn_kind == 'dense'
        x = self.wte(idx)
        hs, refs = [], []
        for blk in self.blocks:
            h = blk.ln1(x)
            a, probs = blk.attn(h, rows=rows)
            x = x + a
            x = x + blk.mlp(blk.ln2(x))
            hs.append(h)
            refs.append(probs)
        return hs, refs

    def extract_reference(self, idx, rows):
        return self.extract(idx, rows)[1]

    def distill_forward(self, hs, rows, probs):
        """PHASE 2 (tree model). Teacher-forced and layer by layer: tree attention l runs on the dense layer-l input hs[l] and is
        scored against the dense attention rows probs[l]. No FFN, no residual stream, no labels, no exact attention: only the
        selection (splitter + budget predictor). Returns (loss, stats); stats['far'] is the similarity (share of the dense
        attention mass outside the static keys that sits on the keys the tree read), averaged over layers, stats['far_layers'] per layer."""
        assert self.attn_kind == 'tree'
        n = len(self.blocks)
        split = pred = 0.0
        far, mass, keys, hit = [], [], [], []
        for l, blk in enumerate(self.blocks):
            blk.attn(hs[l], ref=(rows, probs[l]), select_only=True)
            aux = blk.attn.aux
            split = split + aux.get('split', 0.0)
            pred = pred + aux.get('pred', 0.0)
            far.append(_f(aux['far']))
            mass.append(_f(aux['mass']))
            hit.append(_f(aux['hit']))
            keys.append(_f(aux['keys']))
        loss = (self.cfg.split_weight * split + self.cfg.pred_weight * pred) / n if self.training else None
        stats = dict(split=_f(split) / n, pred=_f(pred) / n, far=sum(far) / n, mass=sum(mass) / n, keys=sum(keys) / n, far_layers=far, hit=sum(hit) / n)
        return loss, stats

    # ------------------------------------------------------------------ forward / generate
    def new_caches(self, b, tmax, dtype=None):
        assert self.attn_kind == 'tree'
        dev = self.wte.weight.device
        dtype = dtype or self.wte.weight.dtype
        return [LayerCache(self.cfg, b, tmax, dev, dtype) for _ in self.blocks]

    def forward(self, idx, targets=None, caches=None, ref=None):
        """targets: (b,T) next tokens. ref = (rows, [probs per layer]) from a dense model's extract_reference (tree model,
        training). Returns (logits, loss or None, stats)."""
        assert ref is None or self.attn_kind == 'tree'
        x = self.wte(idx)
        acc, chists = {}, []
        for i, blk in enumerate(self.blocks):
            x = blk(x, caches[i] if caches else None, (ref[0], ref[1][i]) if ref is not None else None)
            for name, val in blk.attn.aux.items():
                if name == 'chist':
                    chists.append([float(v) for v in val.tolist()])
                else:
                    acc[name] = acc.get(name, 0.0) + val
        logits = self.lm_head(self.ln_f(x))
        n = len(self.blocks)
        pred, split = acc.pop('pred', 0.0), acc.pop('split', 0.0)
        stats = {k: _f(v) / n for k, v in acc.items()}
        if chists:
            stats['chist'] = chists
        if targets is None:
            return logits, None, stats
        lm = F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100)
        loss = lm + self.cfg.pred_weight * pred / n + self.cfg.split_weight * split / n
        stats.update(lm=_f(lm), pred=_f(pred) / n, split=_f(split) / n)
        return logits, loss, stats

    @torch.no_grad()
    def generate(self, prompt, max_new, temperature=0.0):
        """prompt (b,T0) long. Prefill in parallel, then one cached token at a time (sub-quadratic per token)."""
        self.eval()
        b, T0 = prompt.shape
        caches = self.new_caches(b, T0 + max_new)
        logits, _, _ = self(prompt, caches=caches)
        out = [prompt]
        for _ in range(max_new):
            lg = logits[:, -1]
            nxt = lg.argmax(-1, keepdim=True) if temperature <= 0 else torch.multinomial(F.softmax(lg / temperature, -1), 1)
            out.append(nxt)
            logits, _, _ = self(nxt, caches=caches)
        return torch.cat(out, 1)
