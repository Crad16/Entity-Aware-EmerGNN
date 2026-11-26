"""Predictions on the validation pairs, for a temperature fitted without the test set. Run from DrugBank/:

    python ../analysis/valid_probs.py --train_seed 1 --ckpt saved_models/S1_1_learned_shuf_seed1_best.pt \
        saved_models/S1_1_learned_shuf_gate-llm-v2_seed1_best.pt

Every checkpoint predicts all validation pairs over the validation-time graph (training DDIs + the KG of
the training and validation drugs, as evaluate.py does), with its gate ('original') and, for a gated model,
without it ('off'). Probabilities go to runs/<tag>/valid_probs/<key>-v2_<best|last>.npz
(baseline_<best|last>.npz for the ungated model); existing files are skipped. valid_calibration.py reads them.
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

sys.path.insert(0, os.getcwd())
from load_data import DataLoader  # noqa: E402
from models import EmerGNN  # noqa: E402
from ablation_common import EMB_PREFIX, parse_ckpt, run_dir, setup  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument('--ckpt', nargs='+', required=True)
p.add_argument('--batch', type=int, default=32)
p.add_argument('--limit', type=int, default=None, help='N pairs spread over the validation set (quick check)')
p.add_argument('--out_dir', default=None, help='default: runs/<tag>/valid_probs')
cli = setup(p)
cli.out_dir = cli.out_dir or run_dir('valid_probs')
os.makedirs(cli.out_dir, exist_ok=True)

torch.manual_seed(0)
np.random.seed(0)
args = argparse.Namespace(task_dir='./', dataset=cli.dataset, length=3, feat='M', attn_mode='learned', n_dim=64,
                          gate_dim=32, emb_prefix=EMB_PREFIX, lamb=0, lr=1e-3)
data = DataLoader(args)
args.all_ent, args.all_rel, args.eval_rel = data.all_ent, data.all_rel, data.eval_rel
del data.KG, data.tKG
torch.cuda.empty_cache()

valid = data.triplets['valid']
rows = np.arange(len(valid))
if cli.limit:
    rows = np.unique(np.linspace(0, len(valid) - 1, cli.limit).astype(int))
labels = valid[rows, 2]
heads = torch.LongTensor(valid[rows, 0]).cuda()
tails = torch.LongTensor(valid[rows, 1]).cuda()
print(f'{len(rows)} validation pairs', flush=True)


@torch.no_grad()
def predict(model, gate_on):
    gate = model.entity_gate
    model.entity_gate = gate if gate_on else None
    try:
        probs = [F.softmax(model.enc_r(model.enc_ht(heads[s:s + cli.batch], tails[s:s + cli.batch], data.vKG)), dim=-1)
                 .cpu().numpy() for s in range(0, len(rows), cli.batch)]
    finally:
        model.entity_gate = gate
    return np.concatenate(probs)


for path in cli.ckpt:
    feat, version, which = parse_ckpt(path)
    key = f'{feat}-v2_{which}' if feat else f'baseline_{which}'
    out = os.path.join(cli.out_dir, f'{key}.npz')
    if os.path.exists(out) and not cli.limit:
        print(f'{key}: exists, skipped', flush=True)
        continue
    a = argparse.Namespace(**vars(args), entity_feat=feat or 'none', gate_version='v2')
    model = EmerGNN(data.eval_ent, data.eval_rel, a).cuda().eval()
    model.load_state_dict(torch.load(path, map_location='cuda'))
    probs = {}
    for cond in (['original', 'off'] if feat else ['original']):
        t0 = time.time()
        pr = probs[cond] = predict(model, cond == 'original')
        print(f'  {key:14s} {cond:8s} {time.time() - t0:5.0f}s  valid acc {(pr.argmax(1) == labels).mean():.4f} '
              f'macro-F1 {f1_score(labels, pr.argmax(1), average="macro"):.4f}  mean top prob {pr.max(1).mean():.3f}', flush=True)
    if not cli.limit:
        np.savez_compressed(out, rows=rows, labels=labels, **{c: x.astype(np.float16) for c, x in probs.items()})
    del model
    torch.cuda.empty_cache()
print(f'peak GPU memory {torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GB; saved to {cli.out_dir}', flush=True)
