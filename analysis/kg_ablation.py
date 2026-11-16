"""Test-time KG ablations F4 (CYP path removal) and M1 augmentation (KG-poor emerging drugs). Run from
DrugBank/:

    python ../analysis/kg_ablation.py --ckpt saved_models/S1_1_learned_shuf_gate-llm-v2_seed0_best.pt --variants cyp hub
    python ../analysis/kg_ablation.py --ckpt saved_models/S1_1_learned_shuf_seed0_best.pt --variants full cyp hub ...

The trained models stay fixed; only the graph they propagate over changes (kg_variants.py defines the
variants). Every checkpoint is evaluated with its gate ('original') and without it ('off', every
gate = 1, which computes exactly the EmerGNN forward pass) on the 8,195 type-stratified test pairs of
gate_interventions.py, whose saved full-KG predictions are the reference. Probabilities go to
<out_dir>/<key>-v2_<best|last>__<variant>.npz; files that already exist are skipped, so an
interrupted run resumes. An ungated checkpoint (S1_1_learned_shuf_seed0_*.pt, the EmerGNN baseline
trained without a gate) has only 'original' and is saved as baseline_<best|last>__<variant>.npz.
Variant 'full' removes nothing and checks the match with the reference.
analyze_kg_ablation.py reads the output.
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.getcwd())
from load_data import DataLoader  # noqa: E402
from models import EmerGNN  # noqa: E402
from ablation_common import EMB_PREFIX, mechanism_class, parse_ckpt, run_dir, setup  # noqa: E402
from kg_variants import build  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument('--ckpt', nargs='+', required=True)
p.add_argument('--variants', nargs='+', required=True, help="names from kg_variants.py, or 'full'")
p.add_argument('--sample', default=None, help='npz with rows and weight of the test sample; default: the sample\n'
               'stored in the gate_interventions.py output of this split')
p.add_argument('--reference_dir', default=None, help='default: runs/<tag>/gate_interventions')
p.add_argument('--batch', type=int, default=32)
p.add_argument('--limit', type=int, default=None, help='N pairs spread over the sample (quick check)')
p.add_argument('--out_dir', default=None, help='default: runs/<tag>/kg_ablation')
cli = setup(p)
cli.reference_dir = cli.reference_dir or run_dir('gate_interventions')
cli.out_dir = cli.out_dir or run_dir('kg_ablation')
os.makedirs(cli.out_dir, exist_ok=True)

torch.manual_seed(0)
np.random.seed(0)
args = argparse.Namespace(task_dir='./', dataset=cli.dataset, length=3, feat='M', attn_mode='learned', n_dim=64,
                          gate_dim=32, emb_prefix=EMB_PREFIX, lamb=0, lr=1e-3)
data = DataLoader(args)
args.all_ent, args.all_rel, args.eval_rel = data.all_ent, data.all_rel, data.eval_rel
del data.KG, data.vKG, data.tKG                              # rebuilt per variant below
torch.cuda.empty_cache()

kg, keep, info = build([v for v in cli.variants if v != 'full'])
assert np.array_equal(kg, data.test_kg), 'kg_variants.py must index the same KG as load_data.py'
for v, d in info.items():
    json.dump(d, open(os.path.join(cli.out_dir, f'variant_{v}.json'), 'w'), indent=1)

if cli.sample is None:
    found = sorted(glob.glob(os.path.join(cli.reference_dir, '*_probs.npz')))
    assert found, f'no *_probs.npz in {cli.reference_dir}: run gate_interventions.py first, or pass --sample'
    cli.sample = found[0]
sample = np.load(cli.sample)
rows, w = sample['rows'], sample['weight'].astype(float)
if cli.limit:
    pick = np.unique(np.linspace(0, len(rows) - 1, cli.limit).astype(int))
    rows, w = rows[pick], w[pick]
test = data.triplets['test']
labels = test[rows, 2]
mech = mechanism_class()
cls = np.array([mech[y] for y in labels])
heads = torch.LongTensor(test[rows, 0]).cuda()
tails = torch.LongTensor(test[rows, 1]).cuda()
print(f'{len(rows)} pairs; variants {cli.variants}', flush=True)

models = {}
for path in cli.ckpt:
    feat, version, which = parse_ckpt(path)
    if feat:
        assert version == 'v2', path
        a = argparse.Namespace(**vars(args), entity_feat=feat, gate_version='v2')
        key = f'{feat}-v2_{which}'
    else:                                                    # ungated EmerGNN baseline
        a = argparse.Namespace(**vars(args), entity_feat='none')
        key = f'baseline_{which}'
    model = EmerGNN(data.eval_ent, data.eval_rel, a).cuda().eval()
    model.load_state_dict(torch.load(path, map_location='cuda'))
    models[key] = model


def out_path(key, v):
    return os.path.join(cli.out_dir, f'{key}__{v}.npz')


@torch.no_grad()
def predict(model, KG, gate_on):
    gate = model.entity_gate
    model.entity_gate = gate if gate_on else None              # None: plain EmerGNN forward pass
    try:
        probs = [F.softmax(model.enc_r(model.enc_ht(heads[s:s + cli.batch], tails[s:s + cli.batch], KG)), dim=-1)
                 .cpu().numpy() for s in range(0, len(rows), cli.batch)]
    finally:
        model.entity_gate = gate
    return np.concatenate(probs)


def reference(key):
    f = os.path.join(cli.reference_dir, f'{key}_probs.npz')
    if not os.path.exists(f):
        return None
    z = np.load(f)
    idx = np.searchsorted(z['rows'], rows)
    assert (z['rows'][idx] == rows).all()
    return {c: z[c][idx].astype(np.float64) for c in ['original', 'off'] if c in z}


for v in cli.variants:
    todo = [k for k in models if not os.path.exists(out_path(k, v))]
    if not todo:
        print(f'== {v}: all files exist, skipped', flush=True)
        continue
    t0 = time.time()
    mask = np.ones(len(kg), bool) if v == 'full' else keep[v]
    KG = data.load_graph(np.concatenate([data.triplets['train'], data.triplets['valid'], data.test_kg[mask]], axis=0))
    print(f'\n== {v}: {(~mask).sum()} KG edges removed; graph built in {time.time() - t0:.0f}s', flush=True)
    for key in todo:
        probs, ref = {}, reference(key)
        for cond in (['original'] if models[key].entity_gate is None else ['original', 'off']):
            t0 = time.time()
            probs[cond] = predict(models[key], KG, cond == 'original')
            sec = time.time() - t0
            pr = probs[cond]
            line = f'  {key:14s} {cond:8s} {sec:6.0f}s  acc {np.average(pr.argmax(1) == labels, weights=w):.3f}'
            if cli.limit:
                line += f'  (~{sec * len(sample["rows"]) / len(rows) / 60:.0f} min for the whole sample)'
            if ref is not None and cond in ref:
                lp = np.log(np.clip(pr[np.arange(len(rows)), labels], 1e-7, 1))
                lr = np.log(np.clip(ref[cond][np.arange(len(rows)), labels], 1e-7, 1))
                line += '  dlogp vs full KG: ' + ' '.join(
                    f'{c} {np.average((lp - lr)[cls == c], weights=w[cls == c]):+.3f}'
                    for c in ['metabolism', 'pk_other', 'pd', 'unspecified'] if (cls == c).any())
                if v == 'full':
                    line += f'  max |p - saved p| {np.abs(pr - ref[cond]).max():.1e}'
            print(line, flush=True)
        np.savez_compressed(out_path(key, v), rows=rows, weight=w, **{c: x.astype(np.float16) for c, x in probs.items()})
    del KG
    torch.cuda.empty_cache()
if torch.cuda.is_available():
    print(f'peak GPU memory {torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GB')
print('saved to', cli.out_dir, flush=True)
