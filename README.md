# treeattn (final): sub-quadratic tree attention, trained in three phases against a dense reference
(ALHR: adaptive learnable hierarchical routing = the QK splitter + budget predictor + tree descent below.)

One architecture: **FFN + tree attention (QK splitter + budget predictor) + a dense head used for training only.**
Works on language data (Wikipedia / TinyStories / PG-19 token files) and on MQAR, with the same three scripts.

## The three phases
| phase | script | what trains | data used for | ends when |
|---|---|---|---|---|
| 1 | `phase1_dense.py` | dense head + FFN (+ embeddings, norms) | next-token loss (MQAR: answers) | held-out accuracy >= `--target_acc` (0.90), or plateau, or `--steps` |
| 2 | `phase2_distill.py` | **tree attention only** (splitter + budget predictor) | only as *input* to the frozen dense model: no labels, no FFN, no next-word loss | similarity >= `--target_sim` (0.90), or plateau, or `--max_steps` |
| 3 | `phase3_finetune.py` | tree + FFN, everything at 1x | next-token loss (MQAR: answers), standard full training | `--steps` |

**Phase 1** is deliberately not trained to convergence: every `--val_every` steps it measures accuracy on held-out data and stops at the target. The FFN gets its main training here.
Caveat: 0.90 next-token accuracy is far above what a small model reaches on natural text (typically 0.3-0.7), so for `--task lm` the plateau rule or `--steps` ends the phase unless you set `--target_acc` to a reachable value. On MQAR 0.90 is reachable (the metric is answer accuracy at the longest stage, so the curriculum has to be completed; the plateau rule only starts after the earlier stages).

**Phase 2** loads the phase-1 checkpoint as a **frozen reference** and as the initialisation of the tree model (embeddings, norms, FFN identical; with `--copy_attn 1` the dense qkv/proj also warm-start the splitter/projection). Every step the dense model is run on a batch and gives, per layer, its attention input and its attention rows for sampled query positions (`GPT.extract`; for MQAR half the rows are the answer positions). Tree attention layer *l* is run on the dense layer-*l* input (teacher forcing; `GPT.distill_forward`) and trained with
* **splitter loss** (`distill.split_loss`): at every tree level the node scores `q.S/n` over the nodes lying fully in the query's past must follow the dense attention mass inside each node;
* **budget loss** (`distill.budget_labels`): label = fewest leaf groups (2 keys) holding `ref_mass` (0.9) of the dense attention outside the always-read keys.

Similarity = the share of the dense attention mass (outside the always-read keys: sink and last 3 positions) that sits on the keys the tree read. It is measured on fixed validation batches every `--eval_every` steps (per-layer values are printed); `--sim_metric all` counts all mass including the sink. The checkpoint saved is the best state. The full T x T matrix is not stored (about 100 MB per window at T=1024, 8 layers, 6 heads): it is extracted live from the frozen model, which gives the same numbers.
In this phase `v` and the output projection receive no gradient (there is no output loss), so with `--copy_attn 0` they stay random until phase 3.

**Phase 3** starts from the phase-2 checkpoint and is plain training of the tree model on the external data. No dense model is needed. Optional: `--dense_ckpt` only monitors how much of the dense attention the tree still reads; `--keep_ref 1` (needs `--dense_ckpt`) keeps the phase-2 selection losses on during phase 3 to prevent drift of the index (with the next-word loss alone the budget predictor and the tree index get no gradient).

**Inference** is tree + FFN only. `GPT(cfg, 'tree')` contains no dense module (a test checks this, and that `generate` never calls one).

## Layout
- `treeattn/tree.py`       positional binary tree, causal beam descent, neighbour search, budget split, unique key selection
- `treeattn/attention.py`  `TreeAttention` (splitter, budget predictor, exact softmax over <= max_keys keys, `LayerCache`)
- `treeattn/dense.py`      `DenseAttention` (training only; same parameters, head layout and RoPE as the tree attention)
- `treeattn/distill.py`    reference losses, similarity inputs, row sampling
- `treeattn/model.py`      shared FFN/blocks, `GPT(cfg, 'dense'|'tree')`, `load_dense`, `extract`, `distill_forward`, cached `generate`
- `treeattn/sources.py`    task sources: `lm` (token files) and `mqar` (staged curriculum), batches + reports for any model
- `treeattn/mqar.py`, `rope.py`, `config.py`, `lmdata.py`, `needles.py` (needle-in-Wikipedia *evaluation*; unused in training)
- `treeattn/metrics.py` the nine benchmark metrics (measurement only)
- `scripts/` `prepare_wiki.py`, `prepare_tinystories.py`, `prepare_pg19.py`, `phase1_dense.py`, `phase2_distill.py`, `phase3_finetune.py`, `eval_lm.py`, `benchmark.py`, `check_decode.py`, `check_complexity.py`, `time_scaling.py`, `needle_eval.py`
- `tests/test_core.py`

## Run: language
```
python -m pytest tests -q
python scripts/prepare_tinystories.py --out data/tinystories --train_tokens 50000000 --vocab 4096     # or prepare_wiki.py / prepare_pg19.py
python scripts/phase1_dense.py   --task lm --data_dir data/tinystories --out runs/ts/dense --steps 4000 --target_acc 0.55
python scripts/phase2_distill.py --task lm --data_dir data/tinystories --dense_ckpt runs/ts/dense/ckpt.pt --out runs/ts/distill --target_sim 0.90
python scripts/phase3_finetune.py --task lm --data_dir data/tinystories --ckpt runs/ts/distill/ckpt.pt --out runs/ts/tree --steps 2000 --dense_ckpt runs/ts/dense/ckpt.pt
python scripts/eval_lm.py     --task lm --data_dir data/tinystories --ckpt runs/ts/tree/ckpt.pt
python scripts/check_decode.py --task lm --data_dir data/tinystories --ckpt runs/ts/tree/ckpt.pt
```
## Run: MQAR
```
S=256:16:3000,1024:16:1500,4096:16:2500      # T:n_kv:steps, curriculum over context length
python scripts/phase1_dense.py   --task mqar --stages $S --out runs/mqar/dense
python scripts/phase2_distill.py --task mqar --stages $S --dense_ckpt runs/mqar/dense/ckpt.pt --out runs/mqar/distill
python scripts/phase3_finetune.py --task mqar --stages $S --ckpt runs/mqar/distill/ckpt.pt --out runs/mqar/tree --dense_ckpt runs/mqar/dense/ckpt.pt
python scripts/eval_lm.py --task mqar --ckpt runs/mqar/tree/ckpt.pt --eval_grid 256:16,1024:16,4096:16,16384:16
```
(`eval_lm.py` works on the dense checkpoint too, for comparison. MQAR defaults: 4 layers, d=256, wmax 16, max_keys 40; lm defaults: 8 layers, d=384, wmax 32, max_keys 96.)

Complexity and timing: `python scripts/check_complexity.py --T_list 1024,4096,16384,65536`, `python scripts/time_scaling.py --dense 1`.

## Benchmark metrics (v13.1: printing only, no model / training change)
`python scripts/benchmark.py --ckpt runs/.../tree/ckpt.pt --task mqar|lm ...` prints all nine metrics in one report and saves `benchmark.txt` / `benchmark.json` next to the checkpoint. The same numbers appear as a `PHASE n METRICS` block at the end of `phase1_dense.py` / `phase2_distill.py` / `phase3_finetune.py` (`--bench 0` switches it off, `--bench_lens 1024,4096` sets the lengths, default = training length). Definitions live in `treeattn/metrics.py`.

| # | metric | how it is measured | where |
|---|---|---|---|
| 1 | top-1 token accuracy | existing `src.final` (MQAR: answer accuracy per T / n_kv; lm: loss, accuracy by position bucket) | all |
| 2 | needle accuracy by depth | existing `needles.run_needle_eval`: rows = length, columns = depth of the queried fact, control column, mean row | `benchmark.py --needle 1`, `needle_eval.py` (lm) |
| 3 | prefill latency | one parallel forward over the context through fresh caches (cache allocation not timed), mean of `--reps` | benchmark, phase blocks |
| 4 | decode generation speed | prefill, then greedy one-token steps through the KV cache (forward + argmax timed): tokens/s, ms/token, median | tree only |
| 5 | average keys read per token | `stats['keys']`: keys the exact softmax attends to (<= `max_keys`), mean over layers, heads, queries; prefill and decode | tree (dense: analytic) |
| 6 | KV retrieval % / compression | keys read / keys a dense causal model reads. Prefill: `2 * keys / (T + 1)`; decode: keys / (context + 1). Compression = 1 / fraction | tree |
| 7 | KV-cache top-1 agreement | cached token-by-token decoding vs the parallel forward (fp32): top-1 agreement, max abs logit difference (same check as `check_decode.py`) | tree |
| 8 | peak allocated VRAM | `torch.cuda.max_memory_allocated` during the prefill forward (and during decoding), plus how much the forward added over what was already resident | cuda only |
| 9 | KV-cache memory footprint | real tensor sizes of the caches: KV (k, v) + tree index (queries, tree levels, global sum, leaf ids), per token, next to a dense KV cache | tree (dense: analytic) |

Reading the numbers: the tree reduces *compute* (keys read per token), not KV *storage*: the cache holds the full K/V plus the tree index, so [9] is larger than a dense KV cache. Inside a phase script the VRAM total includes everything else resident on the GPU (optimizer state, the dense teacher); `benchmark.py` on a saved checkpoint is the clean number. A dense phase-1 checkpoint gets [1], [3], [8] and the analytic dense numbers for [5], [6], [9] (no cache path, so no [4], [7]); run it next to the tree checkpoint for the comparison table.

## Inference cost (sub-quadratic)
- Prefill: `build_levels` O(T); descent O(T * wmax * log T) in chunks of `q_chunk`; exact attention O(T * max_keys). No T x T tensor exists in `attention.py` / `tree.py`.
- Decode, per token: the new key is added **in place** into its log2(T) ancestor nodes (the tree is never rebuilt); descent O(wmax * log T); neighbour search O(nb_window * nb_store); exact attention over <= `max_keys` keys.
- Memory: KV cache + stored queries + tree levels are linear in T (levels up to ~2x because of power-of-two padding).
- `scripts/check_complexity.py` measures this on the real code path and exits non-zero if a bound is violated.

## Not validated in the authoring environment
The sandbox that produced this version had no PyTorch and no GPU: the code compiles and was desk-checked, but **no test or script has been executed**. First run: `python -m pytest tests -q`. The v13.1 metrics code (`treeattn/metrics.py`, `scripts/benchmark.py`, the `PHASE n METRICS` blocks and their tests) was likewise only compiled and desk-checked, not executed.
Untuned defaults: phase 1 `--target_acc 0.90` (see caveat), phase 2 `lr 3e-4`, `split_weight 0.3`, `pred_weight 0.1`, plateau patience/delta of phases 1 and 2, phase 3 `lr 3e-4`.
