from dataclasses import dataclass, asdict


@dataclass
class Config:
    # ---- model
    vocab_size: int = 8192
    seq_len: int = 1024            # training window length
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 256
    rot_dims: int = 16             # head dims that carry rotary position (exact attention only, NOT indexed by the tree)
    rope_base: float = 10000.0
    rope_cap: int = 0              # 0 = off. > 0: exact attention sees every key at relative distance <= rope_cap

    # ---- tree search
    wmax: int = 32                 # beam width: max tree nodes kept per level (= most leaf groups of 2 keys added per query)
    local: int = 3                 # static recent keys: t, t-1, t-2 (the sink at position 0 is always added too)
    max_keys: int = 96             # hard cap on keys read per query (a constant: attention cost stays O(T * max_keys))
    q_chunk: int = 256             # queries processed at once (bounds memory, any sequence length)
    node_norm: str = 'mean'        # node score = q.S / n ('mean'), q.S / sqrt(n) ('sqrt'), or q.S ('none')

    # ---- budget predictor
    budget_mode: str = 'pred'      # 'pred' or 'fixed'
    fixed_budget: int = 4          # leaf groups (2 keys each) when budget_mode == 'fixed'
    beam_classes: tuple = (2, 4, 8, 16, 32)
    pred_hidden: int = 64

    # ---- neighbour search
    neighbors: bool = True
    nb_window: int = 8             # how many earlier queries a query may borrow from
    nb_store: int = 2              # best tree leaves each query stores for its neighbours
    nb_level: int = 1              # 1 = groups of 2 keys, 2 = groups of 4 keys, ...
    nb_rho: float = 0.35           # a borrowed group matches if new score >= old score - rho * |old score|
    nb_probe: int = 3              # nearest queries used to estimate the match rate (sets the variable cap)
    nb_cap_min: int = 2

    # ---- training against the dense reference attention (phase 2 only; unused at inference)
    ref_rows: int = 128            # query rows per window for which the dense attention row is extracted
    ref_min_pos: int = 16          # rows are sampled from positions >= this
    ref_mass: float = 0.9          # budget label = fewest leaf groups holding this share of the (non-static) dense attention
    gate_mode: str = 'topk_pos'    # phase-2 TARGETS from the dense reference: 'topk_pos' (only the top-k keys and their positions), 'topk' (annealed soft -> hard, keeps mass), 'soft', 'off'
    gate_k: int = 8                # topk: keys kept per query row (by dense mass, always-read keys excluded)
    gate_temp0: float = 0.5        # topk: starting temperature (relative to the row peak); large = almost no gating
    gate_temp1: float = 0.01       # topk: temperature just before the gate becomes exactly hard
    gate_tau: float = 0.1          # soft gate on the dense reference for the phase-2 TARGETS: keys below ~gate_tau x the row's peak mass fade out (0 = off)
    gate_soft: float = 0.5         # width of the gate's smooth edge, as a fraction of gate_tau
    split_weight: float = 0.3      # weight of the splitter loss (tree-node scores vs dense attention mass per node)
    pred_weight: float = 0.1       # weight of the budget-predictor loss

    def to_dict(self):
        return asdict(self)

    @property
    def head_dim(self):
        return self.n_embd // self.n_head

    @property
    def content_dim(self):
        return self.head_dim - self.rot_dims
