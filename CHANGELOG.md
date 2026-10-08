# Changelog

## v13.1 (metrics, ALHR)
- Core mechanism, search, budget predictor, cache and training code unchanged. Only printing / measurement was added.
- `treeattn/metrics.py`: the nine benchmark metrics (top-1 accuracy, needle accuracy by depth, prefill latency, decode speed, keys read per token, KV retrieval % / compression, KV-cache top-1 agreement, peak VRAM, KV-cache footprint).
- `scripts/benchmark.py`: one report (screen + `benchmark.txt` / `benchmark.json`); works on tree and dense checkpoints.
- `PHASE n METRICS` block at the end of phases 1-3 (`--bench`, `--bench_lens`, `--bench_decode`); phase 2 measures its best saved checkpoint. Training logs of phases 2 and 3 show keys read as a share of the context.
- MQAR final / inference accuracy (`src.final`, so also `benchmark.py [1]`) now uses fresh held-out sequences (seed 999M+T). Before, at the last-stage T it used the same sequences as the phase-1 stopping check and the phase-3 monitor (evaluation-only change).
- `needle_eval.py` table now comes from `metrics.needle_table` (same numbers, plus a mean row and a json); `check_decode.py` uses `metrics.decode_sanity` (same printed line); `[1] TOP-1 TOKEN ACCURACY` label above the existing accuracy reports.

## v13 (final)
- Three phases. 1: dense + FFN, stopped at a held-out accuracy target (default 0.90) / plateau / budget. 2: tree attention only (splitter + budget predictor) trained against the frozen dense model's extracted attention, teacher-forced layer by layer, no FFN and no labels, stopped at a similarity target (default 0.90) / plateau / budget. 3: tree + FFN, standard training at 1x.
- Needle labels, needle extractor/detector and external label files are gone. The dense head lives in the repo (`treeattn/dense.py`) for training only; the tree model has no dense module; FFN, embeddings and norms are common (`GPT.load_dense`).
- Task sources (`treeattn/sources.py`): language token files and MQAR run through the same phase scripts. `scripts/prepare_tinystories.py` added next to the Wikipedia and PG-19 preparation scripts.
- Similarity metric: share of the dense attention mass outside the always-read keys that the tree read (`far`), also `mass` (all keys).
- Removed (kept in `past/`): spike detector and gate, early-token boost, anti-needle routing loss, needle/MQAR training scripts of v12, `tree_off`, `abs_pos`, diagnostic hooks (`route`, `hit`), external dense scripts.
- Kept deliberately: optional `rope_cap` (v12 runs trained with it = longest training length), the needle-in-Wikipedia evaluation.
- `scripts/check_complexity.py` added.
