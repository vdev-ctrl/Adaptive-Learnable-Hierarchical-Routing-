"""PHASE 2: train the tree code (QK splitter + budget predictor) against the dense model's attention. Nothing else.
  * no FFN, no residual stream, no next-word loss, no labels: the data is only used as INPUT to the frozen dense model;
  * the frozen phase-1 model runs on a batch and, per layer, gives (a) its attention input and (b) its attention rows (the extracted
    attention matrix, rows sampled per sequence); tree attention l is run on (a) and trained so that what it READS covers (b);
  * only the tree attention parameters are trained (splitter qkv + budget predictor); embeddings, norms and FFN stay as phase 1 left them;
  * similarity = share of the dense attention mass OUTSIDE the always-read keys that sits on the keys the tree read
    (--sim_metric all: of all the mass, sink included). Measured on validation batches every --eval_every steps.
  * stops when similarity >= --target_sim (default 0.90), or when it stops improving by more than --plateau_delta for --patience
    evaluations, or at --max_steps. The checkpoint holds the best state seen.
  python scripts/phase2_distill.py --task lm --data_dir data/wiki --dense_ckpt runs/lm/dense/ckpt.pt --out runs/lm/distill
Note: with --copy_attn 1 (default) v and the output projection are the dense head's and are not touched here (this phase has no output
loss); with --copy_attn 0 they stay random until phase 3."""
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import add_bench_args, add_tree_args, fill_defaults, load_teacher, run_bench_block, tree_config
from treeattn import GPT
from treeattn.distill import sample_rows
from treeattn.sources import add_source_args, make_source

p = argparse.ArgumentParser()
add_source_args(p)
add_tree_args(p)
add_bench_args(p)
p.add_argument('--dense_ckpt', required=True)
p.add_argument('--out', default='runs/distill')
p.add_argument('--seed', type=int, default=1)
p.add_argument('--max_steps', type=int, default=3000)
p.add_argument('--lr', type=float, default=3e-4)
p.add_argument('--warmup', type=int, default=50)
p.add_argument('--copy_attn', type=int, default=1, help='1: warm-start splitter / projection from the dense head')
p.add_argument('--target_sim', type=float, default=0.90, help='stop when the similarity reaches this')
p.add_argument('--sim_metric', default='far', choices=['far', 'all', 'hit'], help='far/all: share of dense mass read; hit: share of the dense top-k keys read')
p.add_argument('--eval_every', type=int, default=100)
p.add_argument('--patience', type=int, default=3, help='stop after this many evaluations without a gain of --plateau_delta')
p.add_argument('--plateau_delta', type=float, default=0.003)
p.add_argument('--gate_anneal', type=int, default=800, help='steps over which the top-k gate hardens from soft to hard (the plateau stop waits for this)')
p.add_argument('--gate_bounce', type=int, default=0, help='1: the gate moves one way, and on every plateau it REVERSES instead of stopping (modes topk_mix / topk); the run ends at --max_steps')
p.add_argument('--gate_start', type=float, default=0.0, help='gate value the bouncing sweep starts from')
p.add_argument('--log_every', type=int, default=25)
a = fill_defaults(p.parse_args())

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
src = make_source(a, a.seed)
teacher, dcfg, dck = load_teacher(a.dense_ckpt, dev)
cfg = tree_config(dcfg, a, src.seq_len)
torch.manual_seed(a.seed)
model = GPT(cfg, 'tree').to(dev)
fresh = model.load_dense(dck['model'], copy_attn=bool(a.copy_attn))
for n_, q_ in model.named_parameters():
    q_.requires_grad_('.attn.' in n_)
train_params = [q_ for q_ in model.parameters() if q_.requires_grad]
if not a.copy_attn:
    print('WARNING: --copy_attn 0: v and the output projection are random and get no gradient in this phase (they are learned in phase 3)', flush=True)
print('PHASE 2 distill | task %s | device %s | trainable %.2fM of %.2fM params (tree attention only) | warm start %s | '
      'target similarity %.2f (%s) | wmax %d classes %s max_keys %d rope_cap %d' % (
          a.task, dev, sum(q.numel() for q in train_params) / 1e6, sum(q.numel() for q in model.parameters()) / 1e6, bool(a.copy_attn),
          a.target_sim, a.sim_metric, cfg.wmax, a.beam_classes, cfg.max_keys, cfg.rope_cap), flush=True)
opt = torch.optim.AdamW(train_params, lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0)
use_amp = dev == 'cuda'
scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
os.makedirs(a.out, exist_ok=True)
ckpt_path = os.path.join(a.out, 'ckpt.pt')
step, best, bad = 0, -1.0, 0
last_hit = 0.0
gate_p, gdir, ref, trace = a.gate_start, 1, None, []   # bouncing gate state (only used with --gate_bounce 1)
if os.path.exists(ckpt_path):
    ck = torch.load(ckpt_path, map_location=dev, weights_only=False)
    model.load_state_dict(ck['model'])
    opt.load_state_dict(ck['opt'])
    step, best, bad = ck['step'], ck['best'], ck['bad']
    print('resumed at step %d (best similarity %.3f)' % (step, best), flush=True)


def save(sim):
    torch.save({'cfg': cfg.to_dict(), 'attn': 'tree', 'phase': 2, 'model': model.state_dict(), 'opt': opt.state_dict(), 'step': step,
                'best': best, 'bad': bad, 'sim': sim}, ckpt_path)


def focus_of(y):
    return (y != -100) if src.focus_rows else None


@torch.no_grad()
def similarity():
    """(similarity used for stopping, mass-all, per-layer far) on fixed validation batches and fixed rows."""
    model.eval()
    global last_hit
    fars, masses, layers, hits = [], [], [], []
    for i, (x, y) in enumerate(src.val_batches(dev)):
        rows = sample_rows(x.shape[0], x.shape[1], cfg.ref_rows, cfg.ref_min_pos, i, 0, dev, focus=focus_of(y))
        with torch.autocast(dev, dtype=torch.float16, enabled=use_amp):
            hs, probs = teacher.extract(x, rows)
            _, st = model.distill_forward(hs, rows, probs)
        fars.append(st['far']), masses.append(st['mass']), layers.append(st['far_layers']), hits.append(st['hit'])
    model.train()
    n = len(fars)
    far, mass, last_hit = sum(fars) / n, sum(masses) / n, sum(hits) / n
    return {'far': far, 'all': mass, 'hit': last_hit}[a.sim_metric], mass, [sum(l[j] for l in layers) / n for j in range(len(layers[0]))]


def report(tag, sim, mass, layers):
    print('  %s step %d | similarity %.3f (%s) | all-mass %.3f | top-%d hit %.3f | per layer %s' % (
        tag, step, sim, a.sim_metric, mass, cfg.gate_k, last_hit, ' '.join('%.2f' % v for v in layers)), flush=True)


model.train()
sim, mass, layers = similarity()
report('SIM', sim, mass, layers)
best = max(best, sim)
if not os.path.exists(ckpt_path):
    save(sim)                                                  # baseline: the warm start itself
reason = None
if sim >= a.target_sim:
    reason = 'target similarity %.2f already reached by the warm start' % a.target_sim
t_start, n_done = time.time(), 0
ref = sim                                                      # similarity level the bouncing plateau test measures gains against
while reason is None:
    if step >= a.max_steps:
        reason = 'max_steps %d reached (best similarity %.3f)' % (a.max_steps, best)
        break
    for g in opt.param_groups:
        g['lr'] = a.lr * min(1.0, (step + 1) / a.warmup)
    cfg.gate_progress = gate_p if a.gate_bounce else min(1.0, step / max(a.gate_anneal, 1))   # shared by all layers; 0 = raw/soft, 1 = top-k
    x, y = src.train_batch(step, dev, mixed=True)
    rows = sample_rows(x.shape[0], x.shape[1], cfg.ref_rows, cfg.ref_min_pos, step, a.seed, dev, focus=focus_of(y))
    with torch.autocast(dev, dtype=torch.float16, enabled=use_amp):
        with torch.no_grad():
            hs, probs = teacher.extract(x, rows)              # the extracted dense attention (rows) and the layer inputs
        loss, st = model.distill_forward(hs, rows, probs)
    if not torch.isfinite(loss):
        step += 1
        continue
    opt.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(train_params, 1.0)
    scaler.step(opt)
    scaler.update()
    step += 1
    n_done += 1
    if a.gate_bounce:
        gate_p = min(1.0, max(0.0, gate_p + gdir / max(a.gate_anneal, 1)))
    if step % a.log_every == 0 or step == 1:
        gate_txt = 'gate %.2f | ' % cfg.gate_progress if a.gate_mode in ('topk', 'topk_mix') else ''   # the gate value only means something in these modes
        print('step %d | %ssplit %.3f pred %.3f | train similarity %.3f | keys %.1f (%.1f%% of context) | lr %.1e | %.2fs/step' % (
            step, gate_txt, st['split'], st['pred'], st['far'], st['keys'], 200.0 * st['keys'] / (x.shape[1] + 1), opt.param_groups[0]['lr'], (time.time() - t_start) / n_done), flush=True)
    if step % a.eval_every == 0:
        sim, mass, layers = similarity()
        report('SIM', sim, mass, layers)
        if a.gate_bounce:
            trace.append((step, cfg.gate_progress, gdir, sim, last_hit))
            print('  GATE step %d | gate %.2f moving %s | similarity %.3f | top-%d hit %.3f' % (
                step, cfg.gate_progress, 'UP' if gdir > 0 else 'DOWN', sim, cfg.gate_k, last_hit), flush=True)
            if sim > ref + a.plateau_delta:
                ref, bad = sim, 0
            else:
                bad += 1
            if bad >= a.patience:                              # plateau: reverse the gate instead of stopping
                gdir, ref, bad = -gdir, sim, 0
                print('  GATE FLIP at step %d: plateau at gate %.2f, now moving %s' % (step, cfg.gate_progress, 'UP' if gdir > 0 else 'DOWN'), flush=True)
        elif sim > best + a.plateau_delta or (a.gate_mode == 'topk' and step < a.gate_anneal):
            bad = 0                                            # no plateau stop while the gate is still hardening
        else:
            bad += 1
        if sim > best:
            best = sim
            save(sim)
        if sim >= a.target_sim:
            reason = 'target similarity %.2f reached (%.3f)' % (a.target_sim, sim)
        elif bad >= a.patience and not a.gate_bounce:
            reason = 'plateau: no gain above %.3f for %d evaluations (best %.3f)' % (a.plateau_delta, a.patience, best)
print('\n=== PHASE 2 finished at step %d: %s ===' % (step, reason))
print('similarity of the saved checkpoint: %.3f' % best)

if a.gate_bounce and len(trace) > 1:
    print('\n=== GATE SWEEP: mean change in similarity per evaluation, by gate range and direction of travel ===')
    print('(gate = weight of the top-%d positions in the targets; 0 = raw dense rows, 1 = top-k only)' % cfg.gate_k)
    edges = [0.0, 0.25, 0.5, 0.75, 1.0001]
    for d, name in ((1, 'moving UP  '), (-1, 'moving DOWN')):
        for lo, hi in zip(edges[:-1], edges[1:]):
            ds = [trace[i][3] - trace[i - 1][3] for i in range(1, len(trace)) if trace[i - 1][2] == d and lo <= 0.5 * (trace[i][1] + trace[i - 1][1]) < hi]
            if ds:
                print('  %s | gate %.2f-%.2f | evals %2d | mean d(sim) %+.4f | total %+.4f' % (name, lo, min(hi, 1.0), len(ds), sum(ds) / len(ds), sum(ds)))
    top = max(trace, key=lambda t: t[3])
    print('best similarity %.3f at gate %.2f (step %d, moving %s)' % (top[3], top[1], top[0], 'UP' if top[2] > 0 else 'DOWN'))

# ---- metrics block on the SAVED (best) checkpoint
if os.path.exists(ckpt_path):
    model.load_state_dict(torch.load(ckpt_path, map_location=dev, weights_only=False)['model'])
print()
run_bench_block(a, model, cfg, src, dev, 'PHASE 2 METRICS (best checkpoint: tree attention aligned to the dense model)', accuracy='quick')
