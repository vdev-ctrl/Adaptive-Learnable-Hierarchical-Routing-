"""PHASE 1: dense training (full causal softmax attention + FFN, next-token loss; MQAR: loss on the answers), NOT to convergence:
it trains until the accuracy on held-out external data reaches --target_acc (default 0.90), exactly like phase 2 stops at its
similarity target. Stops at the first of
  * validation accuracy >= --target_acc,
  * plateau: no gain above --plateau_delta for --patience evaluations (only counted after --plateau_after steps; default: lm 0,
    mqar = all stages but the last, so the curriculum cannot be cut short by the flat part before the recall circuit forms),
  * --steps (the budget; the cosine learning-rate schedule is laid out over it).
The accuracy is next-token accuracy on held-out windows of the training length (lm) or answer accuracy at the longest stage (mqar).
Note for lm: 0.90 next-token accuracy is far above what a small model reaches on natural text (typically 0.3-0.7), so on lm the
plateau / --steps rule is what ends this phase unless you set --target_acc to a reachable value. On mqar 0.90 is reachable.
The result is the initialisation and the frozen reference of phases 2 and 3.
  lm   : python scripts/phase1_dense.py --task lm --data_dir data/wiki --out runs/lm/dense --steps 3000 --target_acc 0.45
  mqar : python scripts/phase1_dense.py --task mqar --stages 256:16:3000,1024:16:1500 --out runs/mqar/dense"""
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import add_arch_args, add_bench_args, cosine_lr, fill_defaults, make_fwd, run_bench_block
from treeattn import Config, GPT
from treeattn.sources import add_source_args, make_source

p = argparse.ArgumentParser()
add_source_args(p)
add_arch_args(p)
add_bench_args(p)
p.add_argument('--out', default='runs/dense')
p.add_argument('--seed', type=int, default=0)
p.add_argument('--steps', type=int, default=0, help='budget. 0 = lm: 3000, mqar: sum of the stage steps')
p.add_argument('--lr', type=float, default=1e-3)
p.add_argument('--wd', type=float, default=None, help='default 0.1 (lm) / 0.01 (mqar)')
p.add_argument('--warmup', type=int, default=100)
p.add_argument('--target_acc', type=float, default=0.90, help='stop when held-out accuracy reaches this')
p.add_argument('--val_every', type=int, default=250, help='evaluation interval (steps)')
p.add_argument('--patience', type=int, default=5, help='stop after this many evaluations without a gain of --plateau_delta')
p.add_argument('--plateau_delta', type=float, default=0.002)
p.add_argument('--plateau_after', type=int, default=-1, help='plateau rule only from this step on; -1 = auto (see above)')
p.add_argument('--log_every', type=int, default=50)
p.add_argument('--save_every', type=int, default=250)
a = fill_defaults(p.parse_args())

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
src = make_source(a, a.seed)
steps = a.steps or src.total_steps or 3000
plateau_after = a.plateau_after if a.plateau_after >= 0 else (sum(n for _, _, n in src.stages[:-1]) if a.task == 'mqar' else 0)
wd = a.wd if a.wd is not None else (0.01 if a.task == 'mqar' else 0.1)
cfg = Config(vocab_size=src.vocab, seq_len=src.seq_len, n_layer=a.n_layer, n_head=a.n_head, n_embd=a.n_embd)
torch.manual_seed(a.seed)
model = GPT(cfg, 'dense').to(dev)
print('PHASE 1 dense | task %s | device %s | params %.2fM | seq_len %d | budget %d steps | target accuracy %.2f | plateau: patience %d, delta %.3f, from step %d' % (
    a.task, dev, sum(q.numel() for q in model.parameters()) / 1e6, cfg.seq_len, steps, a.target_acc, a.patience, a.plateau_delta, plateau_after), flush=True)
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=wd)
use_amp = dev == 'cuda'
scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
os.makedirs(a.out, exist_ok=True)
ckpt_path = os.path.join(a.out, 'ckpt.pt')
step, best, bad, reason, acc = 0, -1.0, 0, None, float('nan')
if os.path.exists(ckpt_path):
    ck = torch.load(ckpt_path, map_location=dev, weights_only=False)
    model.load_state_dict(ck['model'])
    opt.load_state_dict(ck['opt'])
    step, best, bad, reason, acc = ck['step'], ck.get('best', -1.0), ck.get('bad', 0), ck.get('reason'), ck.get('acc', float('nan'))
    print('resumed at step %d%s' % (step, ' (already finished: %s)' % reason if reason else ''), flush=True)


def save():
    torch.save({'cfg': cfg.to_dict(), 'attn': 'dense', 'phase': 1, 'model': model.state_dict(), 'opt': opt.state_dict(), 'step': step,
                'best': best, 'bad': bad, 'reason': reason, 'acc': acc}, ckpt_path)


fwd = make_fwd(model, dev)
t_start, n_done, ema, nonfinite = time.time(), 0, None, 0
model.train()
while reason is None:
    if step >= steps:
        reason = 'step budget %d reached (accuracy %.3f)' % (steps, acc)
        break
    for g in opt.param_groups:
        g['lr'] = cosine_lr(a.lr, step, steps, a.warmup)
    x, y = src.train_batch(step, dev)
    with torch.autocast(dev, dtype=torch.float16, enabled=use_amp):
        _, loss, st = model(x, y)
    if not torch.isfinite(loss):
        nonfinite += 1
        step += 1
        if nonfinite > 100:
            raise SystemExit('too many non-finite steps')
        continue
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
        print('step %d | lm %.3f (ema %.3f) | lr %.1e | %.2fs/step' % (step, st['lm'], ema, opt.param_groups[0]['lr'], (time.time() - t_start) / n_done), flush=True)
    if step % a.val_every == 0:
        model.eval()
        acc = src.score(fwd, dev)
        print('  VAL step %d | accuracy %.3f (target %.2f) | %s' % (step, acc, a.target_acc, src.quick(fwd, dev)), flush=True)
        model.train()
        if acc > best + a.plateau_delta:
            bad = 0
        elif step >= plateau_after:
            bad += 1
        best = max(best, acc)
        if acc >= a.target_acc:
            reason = 'target accuracy %.2f reached (%.3f)' % (a.target_acc, acc)
        elif bad >= a.patience:
            reason = 'plateau: no gain above %.3f for %d evaluations (best %.3f)' % (a.plateau_delta, a.patience, best)
    if reason is None and step % a.save_every == 0:
        save()
save()
model.eval()
print('\n=== PHASE 1 finished at step %d: %s ===' % (step, reason))
print('=== FINAL (phase 1, dense attention) ===')
print('[1] TOP-1 TOKEN ACCURACY')
print('\n'.join(src.final(fwd, dev)))
print()
run_bench_block(a, model, cfg, src, dev, 'PHASE 1 METRICS', accuracy=None)
