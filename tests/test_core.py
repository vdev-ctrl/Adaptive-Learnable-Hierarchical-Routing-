"""Run with: python -m pytest tests -q   (CPU is fine, everything is tiny)."""
import json
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from treeattn import Config, GPT
from treeattn.dense import DenseAttention
from treeattn.distill import budget_labels, sample_rows, soft_gate, split_loss, topk_gate, topk_positions, static_zeroed
from treeattn.rope import rotary, rotate_extra
from treeattn.sources import MQARSource
from treeattn.tree import build_levels, next_pow2, tree_descent


def small_cfg(**kw):
    base = dict(vocab_size=64, n_layer=2, n_head=2, n_embd=64, rot_dims=8, wmax=8, beam_classes=(2, 3, 4, 6, 8), q_chunk=32)
    base.update(kw)
    return Config(**base)


def read_everything_cfg():
    """A budget that covers every past key: the tree must then reproduce dense attention exactly (T <= 32)."""
    return small_cfg(wmax=16, beam_classes=(16,), budget_mode='fixed', fixed_budget=16, max_keys=40, neighbors=False)


def _run(args):
    r = subprocess.run([sys.executable] + args, cwd=ROOT, capture_output=True, text=True)
    return r.returncode, r.stdout, r.stderr


def _run_ok(args):
    rc, out, err = _run(args)
    assert rc == 0, out[-1500:] + err[-1500:]
    return out


# ---------------------------------------------------------------------------------------------------- tree machinery
def test_descent_covers_exactly_the_past():
    torch.manual_seed(0)
    T = 64
    k = torch.randn(1, 1, T, 8)
    S = build_levels(k, next_pow2(T))
    ids, vld, _ = tree_descent(torch.randn(1, 1, T, 8), S, torch.arange(T), wmax=64)
    for t in range(T):
        got = sorted(ids[0, 0, t][vld[0, 0, t]].tolist())
        want = [m for m in range(T // 2) if 2 * m + 1 < t]
        assert got == want, (t, got, want)


def test_no_future_leak():
    torch.manual_seed(1)
    model = GPT(small_cfg(), 'tree').eval()
    x = torch.randint(0, 64, (1, 128))
    base, _, _ = model(x)
    j = 90
    x2 = x.clone()
    x2[0, j:] = torch.randint(0, 64, (128 - j,))
    alt, _, _ = model(x2)
    assert torch.allclose(base[:, :j], alt[:, :j], atol=1e-5), (base[:, :j] - alt[:, :j]).abs().max()


def test_decode_matches_parallel():
    torch.manual_seed(2)
    model = GPT(small_cfg(), 'tree').eval()
    x = torch.randint(0, 64, (1, 96))
    full, _, _ = model(x)
    caches = model.new_caches(1, 96)
    lg, _, _ = model(x[:, :48], caches=caches)
    outs = [lg]
    for t in range(48, 96):
        lg, _, _ = model(x[:, t:t + 1], caches=caches)
        outs.append(lg)
    assert torch.allclose(full, torch.cat(outs, 1), atol=1e-3), (full - torch.cat(outs, 1)).abs().max()


def test_rope_cap_equals_distance_cap_and_is_off_by_default():
    torch.manual_seed(9)
    q, k = torch.randn(1, 1, 1, 16), torch.randn(1, 1, 1, 16)
    t, j, cap = torch.tensor([300]), torch.tensor([100]), 64
    far = (rotary(q, t, 8, 1e4) * rotary(k, j, 8, 1e4)).sum()
    k_cap = rotate_extra(rotary(k, j, 8, 1e4), (t - j - cap).clamp(min=0).view(1, 1, 1), 8, 1e4)
    got = (rotary(q, t, 8, 1e4) * k_cap).sum()
    want = (rotary(q, torch.tensor([cap]), 8, 1e4) * rotary(k, torch.tensor([0]), 8, 1e4)).sum()
    assert torch.allclose(got, want, atol=1e-4) and not torch.allclose(far, want, atol=1e-4)
    model = GPT(small_cfg(), 'tree').eval()
    x = torch.randint(0, 64, (1, 96))
    base, _, _ = model(x)
    model.cfg.rope_cap = 10 ** 6
    assert torch.allclose(base, model(x)[0], atol=1e-5)


# ---------------------------------------------------------------------------------------------------- dense <-> tree
def test_tree_equals_dense_when_the_budget_reads_every_key():
    """Same weights, budget large enough to read all past keys -> the tree head IS the dense head (same RoPE, same softmax)."""
    torch.manual_seed(4)
    cfg = read_everything_cfg()
    dense, tree = GPT(cfg, 'dense').eval(), GPT(cfg, 'tree').eval()
    assert tree.load_dense(dense.state_dict()) == []
    x = torch.randint(0, 64, (2, 32))
    ld, _, _ = dense(x)
    lt, _, _ = tree(x)
    assert torch.allclose(ld, lt, atol=1e-4), (ld - lt).abs().max()


def test_load_dense_shares_the_ffn_and_leaves_only_the_predictor_fresh():
    torch.manual_seed(5)
    cfg = small_cfg()
    dense, tree = GPT(cfg, 'dense'), GPT(cfg, 'tree')
    missing = tree.load_dense(dense.state_dict(), copy_attn=True)
    assert missing and all('.attn.pred.' in m for m in missing)
    for bd, bt in zip(dense.blocks, tree.blocks):
        assert torch.equal(bd.mlp.fc.weight, bt.mlp.fc.weight) and torch.equal(bd.attn.qkv.weight, bt.attn.qkv.weight)
    assert torch.equal(dense.wte.weight, tree.wte.weight)
    tree2 = GPT(cfg, 'tree')
    tree2.load_dense(dense.state_dict(), copy_attn=False)
    assert torch.equal(dense.blocks[0].mlp.fc.weight, tree2.blocks[0].mlp.fc.weight)
    assert not torch.equal(dense.blocks[0].attn.qkv.weight, tree2.blocks[0].attn.qkv.weight)


def test_inference_never_touches_the_dense_head(monkeypatch):
    model = GPT(small_cfg(), 'tree').eval()
    assert not any(isinstance(m, DenseAttention) for m in model.modules())

    def boom(*a, **k):
        raise AssertionError('dense attention was called at inference')
    monkeypatch.setattr(DenseAttention, 'forward', boom)
    out = model.generate(torch.randint(0, 64, (1, 40)), 8)
    assert out.shape == (1, 48)
    for fn in ('attention.py', 'tree.py'):                                   # the inference modules have no dense code at all
        src = open(os.path.join(ROOT, 'treeattn', fn)).read()
        assert 'scaled_dot_product_attention' not in src and 'DenseAttention' not in src


def test_extracted_reference_rows_are_causal_distributions():
    torch.manual_seed(6)
    dense = GPT(small_cfg(), 'dense').eval()
    x = torch.randint(0, 64, (2, 64))
    rows = sample_rows(2, 64, 12, 8, 0, 0, 'cpu')
    assert torch.equal(rows, sample_rows(2, 64, 12, 8, 0, 0, 'cpu')) and rows.min() >= 8 and rows.max() < 64
    refs = dense.extract_reference(x, rows)
    assert len(refs) == 2 and refs[0].shape == (2, 2, 12, 64) and refs[0].dtype == torch.float16
    p = refs[0].float()
    assert torch.allclose(p.sum(-1), torch.ones(2, 2, 12), atol=5e-3)
    future = torch.arange(64).view(1, 1, 1, 64) > rows.view(2, 1, 12, 1)
    assert (p.masked_fill(~future, 0.0) == 0).all()


# ---------------------------------------------------------------------------------------------------- reference losses
def test_split_loss_rewards_ranking_the_ancestors_of_the_attended_key_first():
    T = 16
    kidx = torch.eye(T).view(1, 1, T, T)                                     # key j = unit vector e_j -> node sums are indicators
    S = build_levels(kidx, T)
    rows = torch.tensor([[12]])
    probs = torch.zeros(1, 1, 1, T)
    probs[..., 5] = 1.0                                                      # the reference attends only to key 5 (not a static key)
    m0 = static_zeroed(probs, rows, 3)
    assert m0[..., 5].item() == 1.0
    e5 = torch.zeros(1, 1, 1, T)
    e5[..., 5] = 20.0
    good = split_loss(e5, S, rows, m0, norm='none')
    bad = split_loss(-e5, S, rows, m0, norm='none')
    assert good < 0.5 and bad > 2.0, (good.item(), bad.item())
    m_static = static_zeroed(torch.nn.functional.one_hot(torch.tensor([[[[12]]]]).view(1, 1, 1), T).float().view(1, 1, 1, T), rows, 3)
    assert split_loss(e5, S, rows, m_static, norm='none') == 0.0             # attention on an always-read key: nothing to learn


def test_budget_label_is_the_fewest_groups_holding_ref_mass():
    T = 64
    probs = torch.zeros(1, 1, 1, T)
    probs[..., 10], probs[..., 20], probs[..., 30], probs[..., 0] = 0.5, 0.3, 0.1, 0.1   # key 0 is the sink: static, removed
    m0 = static_zeroed(probs, torch.tensor([[40]]), 3)
    label, total = budget_labels(m0, 0.9, torch.tensor([2, 4, 8]))
    assert abs(total.item() - 0.9) < 1e-6
    assert label.item() == 1                                                  # 3 groups needed (0.5 + 0.3 < 0.81) -> class 4


def test_reference_training_step_trains_splitter_predictor_and_the_common_ffn():
    torch.manual_seed(3)
    cfg = small_cfg()
    dense = GPT(cfg, 'dense').eval()
    tree = GPT(cfg, 'tree')
    tree.load_dense(dense.state_dict())
    tree.train()
    x, y = torch.randint(0, 64, (2, 128)), torch.randint(0, 64, (2, 128))
    rows = sample_rows(2, 128, 16, 8, 0, 0, 'cpu')
    ref = (rows, dense.extract_reference(x, rows))
    _, loss, st = tree(x, y, ref=ref)
    loss.backward()
    a = tree.blocks[1].attn
    assert torch.isfinite(loss) and st['split'] > 0 and st['pred'] > 0 and 0.0 <= st['mass'] <= 1.0001
    assert a.qkv.weight.grad.abs().sum() > 0                                   # splitter
    assert a.pred[0].weight.grad is not None and a.pred[0].weight.grad.abs().sum() > 0   # budget predictor
    assert tree.blocks[1].mlp.fc.weight.grad.abs().sum() > 0                   # common FFN (next-token loss)


def test_mass_read_is_one_when_every_key_is_read():
    torch.manual_seed(7)
    cfg = read_everything_cfg()
    dense, tree = GPT(cfg, 'dense').eval(), GPT(cfg, 'tree').eval()
    tree.load_dense(dense.state_dict())
    x = torch.randint(0, 64, (2, 32))
    rows = sample_rows(2, 32, 8, 4, 0, 0, 'cpu')
    _, _, st = tree(x, ref=(rows, dense.extract_reference(x, rows)))
    assert st["mass"] > 0.99, st["mass"]


# ---------------------------------------------------------------------------------------------------- data + pipeline
def test_val_windows_never_cross_a_book():
    from treeattn.lmdata import val_windows
    val = np.random.randint(1, 50, 20000).astype(np.uint16)
    val[[3000, 9000, 15000]] = 0
    for T in (64, 512, 2048):
        starts = val_windows(val, 0, T, 12)
        assert len(starts) == 12 and starts == val_windows(val, 0, T, 12)
        assert all(not (val[s:s + T + 1] == 0).any() for s in starts)


def test_prepare_wiki_from_local_text(tmp_path):
    pytest.importorskip('tokenizers')
    src = tmp_path / 'wiki'
    src.mkdir()
    rng = np.random.default_rng(2)
    words = ['alpha', 'beta', 'gamma', 'delta', 'omega', 'river', 'stone', 'light', 'shadow', 'winter']
    for i in range(10):
        (src / ('a%d.txt' % i)).write_text(' '.join(rng.choice(words, 400)))
    out = tmp_path / 'w'
    _run_ok([os.path.join(ROOT, 'scripts', 'prepare_wiki.py'), '--out', str(out), '--local_txt_dir', str(src), '--val_articles', '2',
             '--tok_articles', '3', '--vocab', '300', '--train_tokens', '100'])
    meta = json.load(open(out / 'meta.json'))
    assert meta['train_tokens'] > 100 and meta['val_tokens'] > 0 and meta['cross_docs'] is True


def _synthetic_lm(tmp_path):
    d = tmp_path / 'data'
    d.mkdir()
    rng = np.random.default_rng(0)
    for name, n in (('train', 6000), ('val', 3000)):
        arr = rng.integers(1, 64, n).astype(np.uint16)
        arr[::300] = 0
        arr.tofile(d / (name + '.bin'))
    json.dump({'vocab': 64, 'eot': 0, 'cross_docs': True}, open(d / 'meta.json', 'w'))
    return d


TREE = ['--wmax', '16', '--beam_classes', '2,3,4,6,8', '--max_keys', '40', '--q_chunk', '64', '--ref_rows', '16']


def S(n):
    return os.path.join(ROOT, 'scripts', n)


def test_three_phase_pipeline_lm(tmp_path):
    d = _synthetic_lm(tmp_path)
    lm = ['--task', 'lm', '--data_dir', str(d), '--seq_len', '64', '--eval_lens', '64,128', '--val_windows', '2', '--batch', '2']
    arch = ['--n_layer', '2', '--n_head', '2', '--n_embd', '64']
    out = _run_ok([S('phase1_dense.py'), '--out', str(tmp_path / 'dense'), '--steps', '3', '--warmup', '2', '--save_every', '2'] + lm + arch)
    assert 'FINAL (phase 1, dense attention)' in out
    assert 'PHASE 1 METRICS' in out and '[3] prefill latency' in out and '[4] decode speed         : n/a' in out
    dense = str(tmp_path / 'dense' / 'ckpt.pt')
    # phase 2: unreachable target (similarity <= 1) -> it must end by plateau / max_steps, with a checkpoint
    out = _run_ok([S('phase2_distill.py'), '--dense_ckpt', dense, '--out', str(tmp_path / 'distill'), '--max_steps', '4', '--eval_every', '2',
                   '--patience', '1', '--target_sim', '1.5', '--log_every', '1'] + lm + TREE)
    assert 'PHASE 2 distill' in out and 'tree attention only' in out and 'PHASE 2 finished' in out and 'similarity' in out
    assert 'PHASE 2 METRICS' in out and '[6] KV retrieval' in out and '[7] KV-CACHE TOP-1 AGREEMENT' in out
    assert os.path.exists(tmp_path / 'distill' / 'ckpt.pt')
    # phase 2 with a target the warm start already meets: stops at once
    out = _run_ok([S('phase2_distill.py'), '--dense_ckpt', dense, '--out', str(tmp_path / 'distill0'), '--target_sim', '0.0'] + lm + TREE)
    assert 'already reached' in out
    # phase 3 (with the dense monitor and with the selection losses kept on)
    out = _run_ok([S('phase3_finetune.py'), '--ckpt', str(tmp_path / 'distill' / 'ckpt.pt'), '--out', str(tmp_path / 'tree'), '--steps', '3',
                   '--warmup', '2', '--log_every', '1', '--val_every', '2', '--save_every', '2', '--dense_ckpt', dense, '--keep_ref', '1'] + lm)
    assert 'PHASE 3 tree + FFN' in out and 'dense attention read' in out and 'FINAL (phase 3' in out and 'out of memory' not in out
    assert 'PHASE 3 METRICS' in out and '[9] KV-cache footprint' in out and 'SUMMARY' in out and '% of context' in out and 'failed' not in out
    out = _run_ok([S('phase3_finetune.py'), '--ckpt', str(tmp_path / 'distill' / 'ckpt.pt'), '--out', str(tmp_path / 'tree2'), '--steps', '2',
                   '--warmup', '2', '--save_every', '2'] + lm)
    assert 'dense monitor False' in out
    tree = str(tmp_path / 'tree' / 'ckpt.pt')
    assert 'tree attention' in _run_ok([S('eval_lm.py'), '--ckpt', tree] + lm)
    assert 'dense attention' in _run_ok([S('eval_lm.py'), '--ckpt', dense] + lm)
    out = _run_ok([S('benchmark.py'), '--ckpt', tree, '--lens', '64,128', '--n_decode', '4', '--reps', '1', '--amp', '0'] + lm)
    assert '[1] TOP-1 TOKEN ACCURACY' in out and '[2] NEEDLE' in out and '[8] peak VRAM' in out and 'SUMMARY' in out and 'skipped' not in out
    assert os.path.exists(tmp_path / 'tree' / 'benchmark.json') and os.path.exists(tmp_path / 'tree' / 'benchmark.txt')
    out = _run_ok([S('benchmark.py'), '--ckpt', dense, '--lens', '64', '--reps', '1', '--amp', '0', '--accuracy', 'quick'] + lm)
    assert 'dense attention' in out and '[7] KV-CACHE TOP-1 AGREEMENT: n/a' in out
    out = _run_ok([S('check_decode.py'), '--ckpt', tree, '--task', 'lm', '--data_dir', str(d), '--T', '256', '--n_decode', '16', '--n_gen', '8'])
    assert 'DECODE CHECK' in out
    rc, _, _ = _run([S('check_decode.py'), '--ckpt', dense, '--task', 'lm', '--data_dir', str(d), '--T', '256'])
    assert rc != 0                                                           # inference scripts refuse a dense checkpoint


def test_phase1_stops_on_accuracy_target_or_plateau(tmp_path):
    d = _synthetic_lm(tmp_path)
    base = [S('phase1_dense.py'), '--task', 'lm', '--data_dir', str(d), '--seq_len', '64', '--eval_lens', '64', '--val_windows', '2', '--batch', '2',
            '--n_layer', '2', '--n_head', '2', '--n_embd', '64', '--warmup', '2', '--val_every', '1', '--steps', '50']
    out = _run_ok(base + ['--out', str(tmp_path / 'a'), '--target_acc', '0.0'])
    assert 'target accuracy 0.00 reached' in out and 'finished at step 1:' in out
    out = _run_ok(base + ['--out', str(tmp_path / 'b'), '--target_acc', '1.5', '--patience', '1', '--plateau_delta', '0.5'])
    assert 'plateau' in out and 'finished at step 2:' in out
    out = _run_ok(base + ['--out', str(tmp_path / 'c'), '--target_acc', '1.5', '--patience', '99', '--steps', '3'])
    assert 'step budget 3 reached' in out
    out = _run_ok(base + ['--out', str(tmp_path / 'a'), '--target_acc', '0.0'])          # resuming a finished run does not train further
    assert 'already finished' in out


def test_three_phase_pipeline_mqar(tmp_path):
    mq = ['--task', 'mqar', '--stages', '64:8:3', '--vocab', '128', '--tokens_per_batch', '256', '--eval_grid', '64:8', '--eval_n', '2']
    arch = ['--n_layer', '2', '--n_head', '2', '--n_embd', '64']
    out = _run_ok([S('phase1_dense.py'), '--out', str(tmp_path / 'dense'), '--warmup', '2', '--save_every', '2'] + mq + arch)
    assert 'FINAL (phase 1' in out and 'MQAR accuracy' in out
    dense = str(tmp_path / 'dense' / 'ckpt.pt')
    out = _run_ok([S('phase2_distill.py'), '--dense_ckpt', dense, '--out', str(tmp_path / 'distill'), '--max_steps', '4', '--eval_every', '2',
                   '--patience', '1', '--target_sim', '1.5'] + mq + TREE)
    assert 'PHASE 2 finished' in out
    out = _run_ok([S('phase3_finetune.py'), '--ckpt', str(tmp_path / 'distill' / 'ckpt.pt'), '--out', str(tmp_path / 'tree'), '--warmup', '2',
                   '--save_every', '2', '--dense_ckpt', dense] + mq)
    assert 'FINAL (phase 3' in out and 'ACC' in out and 'dense attention read' in out
    tree = str(tmp_path / 'tree' / 'ckpt.pt')
    assert 'MQAR accuracy' in _run_ok([S('eval_lm.py'), '--ckpt', tree] + mq)
    assert 'DECODE CHECK' in _run_ok([S('check_decode.py'), '--ckpt', tree, '--T', '64', '--n_decode', '8'] + mq)


def test_prepare_tinystories_from_local_text(tmp_path):
    pytest.importorskip('tokenizers')
    src = tmp_path / 'stories'
    src.mkdir()
    rng = np.random.default_rng(5)
    words = ['once', 'upon', 'a', 'time', 'there', 'was', 'little', 'girl', 'named', 'lily', 'dog', 'happy']
    for i in range(12):
        (src / ('s%d.txt' % i)).write_text(' '.join(rng.choice(words, 120)))
    out = tmp_path / 'ts'
    _run_ok([S('prepare_tinystories.py'), '--out', str(out), '--local_txt_dir', str(src), '--val_stories', '2', '--tok_stories', '3',
             '--vocab', '300', '--train_tokens', '200'])
    meta = json.load(open(out / 'meta.json'))
    assert meta['train_tokens'] > 200 and meta['val_tokens'] > 0 and meta['cross_docs'] is True and os.path.exists(out / 'tokenizer.json')
    assert np.fromfile(out / 'train.bin', dtype=np.uint16).max() < meta['vocab']


def test_phase2_distill_forward_trains_only_the_selection():
    torch.manual_seed(11)
    cfg = small_cfg()
    dense = GPT(cfg, 'dense').eval()
    tree = GPT(cfg, 'tree')
    tree.load_dense(dense.state_dict())
    tree.train()
    x = torch.randint(0, 64, (2, 96))
    rows = sample_rows(2, 96, 16, 8, 0, 0, 'cpu')
    hs, probs = dense.extract(x, rows)
    assert len(hs) == 2 and hs[0].shape == (2, 96, 64)
    loss, st = tree.distill_forward(hs, rows, probs)
    loss.backward()
    a0 = tree.blocks[0].attn
    assert torch.isfinite(loss) and 0.0 <= st['far'] <= 1.0001 and 0.0 <= st['mass'] <= 1.0001 and len(st['far_layers']) == 2
    assert a0.qkv.weight.grad.abs().sum() > 0 and a0.pred[0].weight.grad.abs().sum() > 0
    assert tree.blocks[0].mlp.fc.weight.grad is None and tree.wte.weight.grad is None      # no FFN, no embeddings, no labels
    assert a0.proj.weight.grad is None                                                      # no exact attention in this phase
    tree.eval()
    loss2, st2 = tree.distill_forward(hs, rows, probs)
    assert loss2 is None and 'far' in st2


def test_far_similarity_is_one_when_every_key_is_read():
    torch.manual_seed(12)
    cfg = read_everything_cfg()
    dense, tree = GPT(cfg, 'dense').eval(), GPT(cfg, 'tree').eval()
    tree.load_dense(dense.state_dict())
    x = torch.randint(0, 64, (2, 32))
    rows = sample_rows(2, 32, 8, 4, 0, 0, 'cpu')
    hs, probs = dense.extract(x, rows)
    _, st = tree.distill_forward(hs, rows, probs)
    assert st['far'] > 0.99 and st['mass'] > 0.99, st


def test_sample_rows_focus_and_mqar_source():
    focus = torch.zeros(2, 64, dtype=torch.bool)
    focus[0, 40] = True
    rows = sample_rows(2, 64, 10, 8, 3, 0, 'cpu', focus=focus)
    assert (rows[0, :5] == 40).all() and rows.min() >= 8 and rows.max() < 64             # half of the rows sit on the focus position
    assert torch.equal(rows, sample_rows(2, 64, 10, 8, 3, 0, 'cpu', focus=focus))
    src = MQARSource(128, [(64, 8, 5), (128, 8, 3)], 256, 0, [(64, 8)], 2)
    assert src.seq_len == 128 and src.total_steps == 8
    assert src.stage_at(0) == (64, 8) and src.stage_at(5) == (128, 8) and src.stage_at(99) == (128, 8)
    assert src.stage_at(0, mixed=True) == (64, 8) and src.stage_at(1, mixed=True) == (128, 8)
    x, y = src.train_batch(7, 'cpu')
    x2, y2 = src.train_batch(7, 'cpu')
    assert x.shape == (2, 128) and torch.equal(x, x2) and torch.equal(y, y2) and (y != -100).sum(1).tolist() == [8, 8]


def test_task_defaults_and_source_args():
    import argparse
    from treeattn.sources import add_source_args, task_defaults
    p = argparse.ArgumentParser()
    add_source_args(p)
    a = p.parse_args(['--task', 'mqar'])
    assert a.task == 'mqar' and task_defaults('mqar')['wmax'] == 16 and task_defaults('lm')['wmax'] == 32


def test_complexity_script_runs_and_reports():
    rc, out, err = _run([os.path.join(ROOT, 'scripts', 'check_complexity.py'), '--T_list', '256,1024', '--steps', '2', '--prefill', '256,512'])
    assert rc in (0, 1) and 'RESULT' in out, out[-800:] + err[-800:]       # tiny sizes are timing-noisy; the real run uses larger T
    assert 'keys read/token' in out


def test_soft_gate_keeps_peaks_fades_the_tail_and_leaves_flat_rows_alone():
    m0 = torch.zeros(1, 1, 3, 8)
    m0[0, 0, 0, 2], m0[0, 0, 0, 5] = 0.5, 0.004                     # a peak and a faint tail (0.8% of the peak)
    m0[0, 0, 1] = 0.1                                               # flat row
    g = soft_gate(m0, 0.1, 0.5)
    assert g[0, 0, 0, 2] == m0[0, 0, 0, 2] and g[0, 0, 0, 5] == 0.0
    assert torch.equal(g[0, 0, 1], m0[0, 0, 1])
    assert torch.equal(soft_gate(m0, 0.0), m0) and g.sum() > 0 and (g <= m0).all()
    assert g[0, 0, 2].sum() == 0                                    # empty row stays empty, no NaN


def test_topk_gate_is_soft_at_the_start_and_exactly_hard_at_the_end():
    m0 = torch.zeros(1, 1, 2, 10)
    m0[0, 0, 0, :5] = torch.tensor([0.4, 0.3, 0.2, 0.06, 0.04])
    hard = topk_gate(m0, 2, 1.0)
    assert torch.equal(hard[0, 0, 0], torch.tensor([0.4, 0.3, 0, 0, 0, 0, 0, 0, 0, 0]))
    assert hard[0, 0, 1].sum() == 0                                  # empty row stays empty, no NaN
    early = topk_gate(m0, 2, 0.0)
    assert (early[0, 0, 0, :5] > 0).all() and (early <= m0).all()    # soft start keeps everything, scaled down
    mid, late = topk_gate(m0, 2, 0.5), topk_gate(m0, 2, 0.99)
    assert mid[0, 0, 0, 2] < early[0, 0, 0, 2]                       # the third key fades as the gate hardens
    assert late[0, 0, 0, 2] < 0.2 * 0.05 and late[0, 0, 0, 0] > 0.39  # nearly hard just before the end


def test_topk_positions_marks_only_the_k_heaviest_keys_with_equal_weight():
    m0 = torch.zeros(1, 1, 2, 10)
    m0[0, 0, 0, :5] = torch.tensor([0.1, 0.4, 0.05, 0.3, 0.15])
    m0[0, 0, 1, 7] = 0.9                                             # a row with a single attended key
    hot = topk_positions(m0, 3)
    assert hot[0, 0, 0].sum() == 3 and hot[0, 0, 0, 1] == 1 and hot[0, 0, 0, 3] == 1 and hot[0, 0, 0, 4] == 1
    assert hot[0, 0, 1].sum() == 1 and hot[0, 0, 1, 7] == 1          # fewer than k attended keys: only those are marked


def test_topk_mix_endpoints_are_raw_rows_and_top_k_positions():
    from types import SimpleNamespace
    m0 = torch.zeros(1, 1, 2, 10)
    m0[0, 0, 0, :5] = torch.tensor([0.1, 0.4, 0.05, 0.3, 0.15])
    cfg = SimpleNamespace(gate_mode='topk_mix', gate_k=3, gate_progress=1.0)
    from treeattn.distill import gate_targets
    assert torch.equal(gate_targets(m0, cfg), topk_positions(m0, 3))
    cfg.gate_progress = 0.0
    raw = gate_targets(m0, cfg)
    assert torch.allclose(raw[0, 0, 0] / raw[0, 0, 0].sum(), m0[0, 0, 0] / m0[0, 0, 0].sum()) and raw[0, 0, 1].sum() == 0
    cfg.gate_progress = 0.5
    mid = gate_targets(m0, cfg)
    assert torch.allclose(mid[0, 0, 0].sum(), topk_positions(m0, 3)[0, 0, 0].sum())


def test_metrics_cache_footprint_and_dense_analytic():
    from treeattn.metrics import cache_footprint, dense_cache_mb
    cfg = small_cfg()
    caches = GPT(cfg, 'tree').eval().new_caches(1, 64)
    f = cache_footprint(caches)
    kv = 2 * cfg.n_layer * 1 * cfg.n_head * 64 * cfg.head_dim * 4 / 2 ** 20           # k and v of every layer, fp32
    assert abs(f['kv_mb'] - kv) < 1e-9 and f['index_mb'] > 0 and abs(f['total_mb'] - f['kv_mb'] - f['index_mb']) < 1e-9
    assert abs(dense_cache_mb(GPT(cfg, 'dense'), 1, 64)['kv_mb'] - kv) < 1e-9


def test_metrics_prefill_decode_retrieval_and_sanity():
    from treeattn.metrics import decode_sanity, measure_decode, measure_prefill
    torch.manual_seed(5)
    model = GPT(small_cfg(), 'tree').eval()
    x = torch.randint(0, 64, (1, 96))
    pre = measure_prefill(model, x, amp=False, reps=1)
    assert pre['ms'] > 0 and 0 < pre['keys'] <= model.cfg.max_keys and 0 < pre['retrieval_pct'] <= 100.0001
    assert abs(pre['compression'] * pre['retrieval_pct'] / 100 - 1) < 1e-6 and pre['peak_mb'] is None      # cpu: no VRAM number
    dec = measure_decode(model, x, n_decode=4, amp=False)
    assert dec['n'] == 4 and dec['tok_s'] > 0 and 0 < dec['keys'] <= model.cfg.max_keys and 0 < dec['retrieval_pct'] <= 100.0001
    s = decode_sanity(model, x, 8)
    assert s['n'] == 8 and 0.0 <= s['top1_agree_pct'] <= 100.0 and s['max_diff'] < 1e-3      # a random model has near-tied logits: agreement itself is not asserted


def test_metrics_run_benchmark_report_tree_and_dense():
    from treeattn.metrics import run_benchmark
    src = MQARSource(128, [(64, 8, 5)], 256, 0, [(64, 8)], 2)
    cfg = small_cfg(vocab_size=128)
    for kind in ('tree', 'dense'):
        model = GPT(cfg, kind)
        lines, res = run_benchmark(model, cfg, src, 'cpu', [64], n_decode=4, reps=1, amp=False, accuracy='final', log=lambda s: None, title='T')
        text = '\n'.join(lines)
        for tag in ('[1] TOP-1', '[2] NEEDLE', '[3] prefill latency', '[4] decode speed', '[5] avg keys read', '[6] KV retrieval', '[7] KV-CACHE', '[8] peak VRAM', '[9] KV-cache footprint', 'SUMMARY'):
            assert tag in text, tag
        assert 'failed' not in text and 'skipped' not in text and '64' in res['contexts'] and model.training is True
        assert ('decode_sanity' in res) == (kind == 'tree')


def test_needle_table_has_a_mean_row():
    from treeattn.metrics import needle_table
    rng = np.random.default_rng(0)
    val = rng.integers(1, 50, 4000)
    enc = lambda s: [ord(c) % 50 + 1 for c in s]
    lines, data = needle_table(lambda ids: rng.standard_normal(200), val, enc, list(range(100, 140)), [512, 1024], [0.1, 0.5, 0.9], 2, log=lambda s: None)
    assert len(lines) == 5 and lines[-1].startswith('mean') and '| control' in lines[1]
    assert set(data['mean_by_depth']) == {'0.1', '0.5', '0.9'} and set(data['acc']) == {'512', '1024'}


def test_mqar_final_report_uses_fresh_sequences():
    src = MQARSource(128, [(64, 8, 5)], 256, 0, [(64, 8)], 2)
    seen = []

    def fwd(x):
        seen.append(x.clone())
        return torch.zeros(*x.shape, 128)

    src.score(fwd, 'cpu')
    stop = torch.cat(seen)
    seen.clear()
    src.final(fwd, 'cpu')
    fin = torch.cat(seen)
    seen.clear()
    src.final(fwd, 'cpu')
    assert not torch.equal(stop, fin) and torch.equal(fin, torch.cat(seen))      # final: never the stopping sequences, but reproducible
