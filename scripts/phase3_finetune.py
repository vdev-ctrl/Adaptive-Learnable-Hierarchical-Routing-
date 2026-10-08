"""PHASE 3: standard training of the tree model + FFN on the external data (next word; MQAR: the answers), everything at 1x.
Starts from the phase-2 checkpoint (tree attention aligned to the dense model, FFN / embeddings / norms and v / proj from phase 1).
No dense model is needed. Inference uses exactly this model.
  --dense_ckpt PATH (optional): monitor only, prints how much of the dense model's attention (non-static part) the tree still reads.
  --keep_ref 1 (needs --dense_ckpt): keep the phase-2 splitter / budget losses switched on next to the next-word loss, to stop the
      selection from drifting while the exact-attention weights are trained (the next-word loss alone gives the index sums and the
      budget predictor no gradient).
  python scripts/phase3_finetune.py --task lm --data_dir data/wiki --ckpt runs/lm/distill/ckpt.pt --out runs/lm/tree --steps 2000"""
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import add_bench_args, cosine_lr, fill_defaults, load_teacher, make_fwd, recall, run_bench_block
from treeattn import Config, GPT
from treeattn.distill import sample_rows
from treeattn.sources import add_source_args, make_source

p = argparse.ArgumentParser()
add_source_args(p)
add_bench_args(p)
p.add_argument('--ckpt', required=True, help='phase-2 checkpoint')
p.add_argument('--out', default='runs/tree')
p.add_argument('--seed', type=int, default=2)
p.add_argument('--steps', type=int, default=0, help='0 = lm: 2000, mqar: sum of the stage steps')
p.add_argument('--lr', type=float, default=3e-4)
p.add_argument('--wd', type=float, default=None, help='default 0.1 (lm) / 0.01 (mqar)')
p.add_argument('--warmup', type=int, default=100)
p.add_argument('--dense_ckpt', default='')
p.add_argument('--keep_ref', type=int, default=0)
p.add_argument('--rope_cap', type=int, default=-1, help='-1 = keep the value of phase 2 (raised to the longest training length if needed), 0 = off')
p.add_argument('--log_every', type=int, default=50)
p.add_argument('--val_every', type=int, default=500)
p.add_argument('--save_every', type=int, default=250)
a = fill_defaults(p.parse_args())
assert not a.keep_ref or a.dense_ckpt, '--keep_ref 1 needs --dense_ckpt'

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
src = make_source(a, a.seed)
pck = torch.load(a.ckpt, map_location=dev, weights_only=False)
assert pck.get('attn') == 'tree', 'phase 3 starts from the phase-2 (tree) checkpoint'
cfg = Config(**pck['cfg'])
cfg.seq_len = max(cfg.seq_len, src.seq_len)
if a.rope_cap == 0:
    cfg.rope_cap = 0
elif a.rope_cap > 0:
    cfg.rope_cap = a.rope_cap
elif cfg.rope_cap > 0:
    cfg.rope_cap = max(cfg.rope_cap, src.seq_len)
steps = a.steps or src.total_steps or 2000
wd = a.wd if a.wd is not None else (0.01 if a.task == 'mqar' else 0.1)
torch.manual_seed(a.seed)
model = GPT(cfg, 'tree').to(dev)
model.load_state_dict(pck['model'])
teacher = load_teacher(a.dense_ckpt, dev)[0] if a.dense_ckpt else None
print('PHASE 3 tree + FFN | task %s | device %s | params %.2fM | all parameters at 1x lr %.1e | steps %d | rope_cap %d | '
      'dense monitor %s | keep_ref %s' % (a.task, dev, sum(q.numel() for q in model.parameters()) / 1e6, a.lr, steps, cfg.rope_cap,
                                          bool(teacher), bool(a.keep_ref)), flush=True)
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=wd)
use_amp = dev == 'cuda'
scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
os.makedirs(a.out, exist_ok=True)
ckpt_path = os.path.join(a.out, 'ckpt.pt')
step = 0
if os.path.exists(ckpt_path):
    ck = torch.load(ckpt_path, map_location=dev, weights_only=False)
    model.load_state_dict(ck['model'])
    opt.load_state_dict(ck['opt'])
    step = ck['step']
    print('resumed at step', step, flush=True)


def save():
    torch.save({'cfg': cfg.to_dict(), 'attn': 'tree', 'phase': 3, 'model': model.state_dict(), 'opt': opt.state_dict(), 'step': step}, ckpt_path)


def monitor(tag):
    msg = '  %s step %d | %s' % (tag, step, src.quick(fwd, dev))
    if teacher is not None:
        far, mass = recall(model, teacher, src, dev, cfg)
        msg += ' | dense attention read: non-static %.3f, all %.3f' % (far, mass)
    print(msg, flush=True)


fwd = make_fwd(model, dev)
model.eval()
monitor('START')
model.train()
t_start, n_done, ema, bad = time.time(), 0, None, 0
while step < steps:
    for g in opt.param_groups:
        g['lr'] = cosine_lr(a.lr, step, steps, a.warmup)
    x, y = src.train_batch(step, dev)
    ref = None
    with torch.autocast(dev, dtype=torch.float16, enabled=use_amp):
        if a.keep_ref:
            rows = sample_rows(x.shape[0], x.shape[1], cfg.ref_rows, cfg.ref_min_pos, step, a.seed, dev, focus=(y != -100) if src.focus_rows else None)
            ref = (rows, teacher.extract_reference(x, rows))
        _, loss, st = model(x, y, ref=ref)
    if not torch.isfinite(loss):
        bad += 1
        step += 1
        if bad >= 20:
            raise SystemExit('training diverged: 20 non-finite steps')
        continue
    bad = 0
    opt.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(opt)
    scaler.update()
    step += 1
    n_done += 1
    ema = st['lm'] if ema is None else 0.98 * ema + 0.02 * st['lm']
    if step % a.log_every == 0 or step == 1:
        print('step %d | lm %.3f (ema %.3f) | keys %.1f (%.1f%% of context) | lr %.1e | %.2fs/step' % (
            step, st['lm'], ema, st.get('keys', 0), 200.0 * st.get('keys', 0) / (x.shape[1] + 1), opt.param_groups[0]['lr'], (time.time() - t_start) / n_done), flush=True)
    if step % a.val_every == 0:
        model.eval()
        monitor('VAL')
        model.train()
    if step % a.save_every == 0 or step >= steps:
        save()
model.eval()
print('\n=== FINAL (phase 3, tree attention only: the inference path) ===')
print('[1] TOP-1 TOKEN ACCURACY')
print('\n'.join(src.final(fwd, dev)))
print()
run_bench_block(a, model, cfg, src, dev, 'PHASE 3 METRICS (tree + FFN: the inference model)', accuracy=None)
