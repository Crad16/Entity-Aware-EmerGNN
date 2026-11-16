"""Gate input interventions (ablations F1, F2). Run from DrugBank/:

    python ../analysis/gate_interventions.py --ckpt saved_models/S1_1_learned_shuf_gate-llm-v2_seed0_best.pt

The gate has two inputs: the drug pair (query) and the entity (key). Each condition replaces one of
them at test time and leaves everything else, including EmerGNN's relation attention, unchanged.

    original       the trained model
    off            every gate = 1 (the trained model without its gate)
    mean           each entity's gate fixed to its mean over training pairs (no pair dependence)
    shuffle_all    query taken from another test pair drawn from the whole set (test_ddi.txt is
                   sorted by interaction type, so a within-batch shuffle would stay within a type)
    shuffle_type   query taken from another test pair of the same interaction type
    key_perm       keys permuted among non-drug entities of the same kind (Gene<->Gene, ...)

Under the shuffles and the mean gate the gate's metabolism-vs-rest contrast disappears by
construction, so only the change in predictions is informative: the flip rate against the original
prediction and the change in log-probability of the true type, for metabolism-type pairs, non-PK
pairs and all pairs. Pairs used: a type-stratified sample of the test set (default 8,192 of 32,322),
weighted back to the type frequencies of the full set. Intervals are 95% bootstrap intervals over
emerging drugs (pairs of one drug are not independent).
"""
import argparse
import collections
import csv
import json
import os
import re
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score

sys.path.insert(0, os.getcwd())
from load_data import DataLoader  # noqa: E402
from models import EmerGNN  # noqa: E402
from ablation_common import EMB_PREFIX, ENT_TSV, parse_ckpt, res_path, run_dir, setup  # noqa: E402

METAB, PK_OTHER = {46, 3}, {72, 74, 76, 10}

p = argparse.ArgumentParser()
p.add_argument('--ckpt', nargs='+', required=True)
p.add_argument('--dataset', default='S1_1')
p.add_argument('--n_pairs', type=int, default=8192, help='size of the type-stratified test sample')
p.add_argument('--reps', type=int, default=3, help='random draws per shuffle / permutation condition')
p.add_argument('--n_mean_pairs', type=int, default=4096, help='training pairs used for the mean gate')
p.add_argument('--batch', type=int, default=32)
p.add_argument('--seed', type=int, default=0)
p.add_argument('--limit', type=int, default=None, help='use only the first N sampled pairs (quick check)')
p.add_argument('--out_dir', default=None, help='probabilities; default: runs/<tag>/gate_interventions')
cli = setup(p)
cli.out_dir = cli.out_dir or run_dir('gate_interventions')
os.makedirs(cli.out_dir, exist_ok=True)


class InterventionGate(nn.Module):
    """Wraps a trained EntityGate; enc_ht calls it exactly like the original."""

    def __init__(self, gate):
        super().__init__()
        self.gate, self.mode = gate, 'original'
        self.query_ht = self.mean_gates = self.perm_keys = None

    def forward(self, ht_embed, keys=None, center=None):
        if self.mode == 'off':
            return [None] * self.gate.n_layer
        if self.mode == 'mean':
            return [None if m is None else m.unsqueeze(1).expand(-1, ht_embed.size(0)) for m in self.mean_gates]
        if self.mode == 'query':
            return self.gate(self.query_ht, center=center)
        if self.mode == 'key':
            return self.gate(ht_embed, keys=self.perm_keys, center=center)
        return self.gate(ht_embed, center=center)


def drug_embed(model, idx):
    return model.Went(model.ent_kg[idx])


# ------------------------------------------------------------------ data and pair sample
np.random.seed(cli.seed)
torch.manual_seed(cli.seed)
args = argparse.Namespace(task_dir='./', dataset=cli.dataset, length=3, feat='M', attn_mode='learned', n_dim=64,
                          gate_dim=32, emb_prefix=EMB_PREFIX, lamb=0, lr=1e-3)
data = DataLoader(args)
args.all_ent, args.all_rel, args.eval_rel = data.all_ent, data.all_rel, data.eval_rel

test = data.triplets['test']
group = np.array(['metabolism' if r in METAB else 'pk_other' if r in PK_OTHER else 'non_pk' for r in test[:, 2]])
rng = np.random.RandomState(cli.seed)
frac = min(1.0, cli.n_pairs / len(test))
rows, weight = [], {}
for r in np.unique(test[:, 2]):
    idx = np.where(test[:, 2] == r)[0]
    take = rng.choice(idx, max(1, int(round(len(idx) * frac))), replace=False)
    rows += list(take)
    weight.update({i: len(idx) / len(take) for i in take})   # rare types keep at least one pair
rows = np.array(sorted(rows))
if cli.limit:
    rows = rows[rng.permutation(len(rows))[:cli.limit]]
w = np.array([weight[i] for i in rows])
labels, groups = test[rows, 2], group[rows]
new_drug = np.array([h if h not in data.train_ent else t for h, t, _ in test[rows]])
heads = torch.LongTensor(test[rows, 0]).cuda()
tails = torch.LongTensor(test[rows, 1]).cuda()
print(f'{len(rows)} pairs: ' + ', '.join(f'{g} {(groups == g).sum()}' for g in ['metabolism', 'non_pk', 'pk_other'])
      + f'; {len(set(new_drug))} emerging drugs')

kind = {int(r['ent_id']): r['kind'] for r in csv.DictReader(open(ENT_TSV), delimiter='\t')}
train = data.triplets['train']
mean_rows = rng.choice(len(train), min(cli.n_mean_pairs, len(train)), replace=False)


def donors(within_type, rep):
    """Index (into rows) of the pair whose query each pair receives."""
    r = np.random.RandomState(1000 * cli.seed + 10 * rep + within_type)
    out = np.arange(len(rows))
    keys = labels if within_type else np.zeros(len(rows), dtype=int)
    for k in np.unique(keys):
        idx = np.where(keys == k)[0]
        out[idx] = idx[r.permutation(len(idx))]
    return out


def key_permutation(keys, rep):
    r = np.random.RandomState(2000 * cli.seed + rep)
    on = gate.mask.squeeze(1).bool().cpu().numpy()
    perm = np.arange(len(keys))
    by_kind = collections.defaultdict(list)
    for e in np.where(on)[0]:
        by_kind[kind.get(int(e), '?')].append(e)
    for ids in by_kind.values():
        ids = np.array(ids)
        perm[ids] = ids[r.permutation(len(ids))]
    return keys[torch.as_tensor(perm, device=keys.device)]


@torch.no_grad()
def predict(model, wrapper, donor=None):
    probs = []
    for s in range(0, len(rows), cli.batch):
        h, t = heads[s:s + cli.batch], tails[s:s + cli.batch]
        if donor is not None:
            d = torch.as_tensor(donor[s:s + cli.batch], device='cuda')
            wrapper.query_ht = torch.cat([drug_embed(model, heads[d]), drug_embed(model, tails[d])], dim=-1)
        scores = model.enc_r(model.enc_ht(h, t, data.tKG))
        probs.append(F.softmax(scores, dim=-1).cpu().numpy())
    return np.concatenate(probs)


# ------------------------------------------------------------------ metrics
def boot_ci(values, weights, clusters, n=1000):
    """95% interval of the weighted mean, resampling emerging drugs."""
    ids, inv = np.unique(clusters, return_inverse=True)
    sums, counts = np.bincount(inv, weights=values * weights), np.bincount(inv, weights=weights)
    pick = np.random.RandomState(0).randint(0, len(ids), size=(n, len(ids)))
    means = sums[pick].sum(1) / counts[pick].sum(1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def dlogp_of(prob, ref, eps=1e-12):
    return np.log(prob[np.arange(len(rows)), labels] + eps) - np.log(ref[np.arange(len(rows)), labels] + eps)


def metrics(prob, ref):
    pred, pred_ref = prob.argmax(1), ref.argmax(1)
    eps = 1e-12
    dlogp = dlogp_of(prob, ref)
    kl = (ref * (np.log(ref + eps) - np.log(prob + eps))).sum(1)
    out = {'macro_f1': float(f1_score(labels, pred, average='macro', sample_weight=w)),
           'acc': float(np.average(pred == labels, weights=w))}
    for g in ['metabolism', 'non_pk', 'all']:
        m = np.ones(len(rows), bool) if g == 'all' else groups == g
        ww = w[m]
        out[g] = {'n': int(m.sum()),
                  'acc': float(np.average(pred[m] == labels[m], weights=ww)),
                  'flip': float(np.average(pred[m] != pred_ref[m], weights=ww)),
                  'dlogp': float(np.average(dlogp[m], weights=ww)),
                  'kl': float(np.average(kl[m], weights=ww))}
    return out


# ------------------------------------------------------------------ run
for path in cli.ckpt:
    feat, version, which = parse_ckpt(path)
    assert feat, f'{path} has no gate, so there is nothing to intervene on (use kg_ablation.py)'
    args.entity_feat, args.gate_version = feat, version
    model = EmerGNN(data.eval_ent, data.eval_rel, args).cuda().eval()
    model.load_state_dict(torch.load(path, map_location='cuda'))
    gate = model.entity_gate
    wrapper = model.entity_gate = InterventionGate(gate)
    name = f'{feat}-{version}_{which}'
    print(f'\n== {name}  ({path})', flush=True)

    with torch.no_grad():                                       # mean gate over training pairs
        center, sums = model.pair_center(), None
        for s in range(0, len(mean_rows), 512):
            tr = torch.LongTensor(train[mean_rows[s:s + 512]]).cuda()
            ht = torch.cat([drug_embed(model, tr[:, 0]), drug_embed(model, tr[:, 1])], dim=-1)
            g = [None if x is None else x.sum(1) for x in gate(ht, center=center)]
            sums = g if sums is None else [None if a is None else a + b for a, b in zip(sums, g)]
        wrapper.mean_gates = [None if x is None else x / len(mean_rows) for x in sums]
        keys = gate.keys()

    conditions = [('original', None), ('off', None), ('mean', None)]
    conditions += [(f'shuffle_all_{k}', k) for k in range(cli.reps)]
    conditions += [(f'shuffle_type_{k}', k) for k in range(cli.reps)]
    conditions += [(f'key_perm_{k}', k) for k in range(cli.reps)]
    probs = {}
    for cond, rep in conditions:
        t0 = time.time()
        donor = None
        if cond.startswith('shuffle'):
            wrapper.mode, donor = 'query', donors(cond.startswith('shuffle_type'), rep)
        elif cond.startswith('key_perm'):
            wrapper.mode, wrapper.perm_keys = 'key', key_permutation(keys, rep)
        else:
            wrapper.mode = cond
        probs[cond] = predict(model, wrapper, donor)
        print(f'  {cond:16s} {time.time() - t0:6.0f}s', flush=True)

    res = {c: metrics(pr, probs['original']) for c, pr in probs.items()}
    summary = {}
    for c in ['original', 'off', 'mean', 'shuffle_all', 'shuffle_type', 'key_perm']:
        keys_c = [k for k in res if k == c or re.fullmatch(rf'{c}_\d+', k)]
        reps = [res[k] for k in keys_c]
        dl = np.mean([dlogp_of(probs[k], probs['original']) for k in keys_c], axis=0)
        summary[c] = {'macro_f1': float(np.mean([r['macro_f1'] for r in reps])),
                      'acc': float(np.mean([r['acc'] for r in reps]))}
        for g in ['metabolism', 'non_pk', 'all']:
            summary[c][g] = {k: float(np.mean([r[g][k] for r in reps])) for k in ['acc', 'flip', 'dlogp', 'kl']}
            if g != 'all':
                m = groups == g
                summary[c][g]['dlogp_ci'] = boot_ci(dl[m], w[m], new_drug[m])
            summary[c][g]['dlogp_rep_sd'] = float(np.std([r[g]['dlogp'] for r in reps]))

    print(f'\n{"condition":13s} {"F1":>6s} {"acc":>6s} | {"metab flip":>10s} {"dlogp":>7s} {"95% CI":>16s} | '
          f'{"non-PK flip":>11s} {"dlogp":>7s} {"95% CI":>16s}')
    for c, s in summary.items():
        mt, npk = s['metabolism'], s['non_pk']
        print(f'{c:13s} {s["macro_f1"]:6.3f} {s["acc"]:6.3f} | {mt["flip"]:10.3f} {mt["dlogp"]:+7.3f} '
              f'[{mt["dlogp_ci"][0]:+.3f},{mt["dlogp_ci"][1]:+.3f}] | {npk["flip"]:11.3f} {npk["dlogp"]:+7.3f} '
              f'[{npk["dlogp_ci"][0]:+.3f},{npk["dlogp_ci"][1]:+.3f}]')

    # a --limit run is a quick check: keep its table out of results/
    table = os.path.join(cli.out_dir, f'{name}.json') if cli.limit else res_path(f'gate_interventions/{name}.json')
    json.dump({'checkpoint': path, 'n_pairs': len(rows), 'summary': summary, 'per_draw': res}, open(table, 'w'), indent=1)
    np.savez_compressed(os.path.join(cli.out_dir, f'{name}_probs.npz'), rows=rows, weight=w,
                        **{c: pr.astype(np.float16) for c, pr in probs.items()})
    if torch.cuda.is_available():
        print(f'peak GPU memory {torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GB')
    print('saved', table, 'and', os.path.join(cli.out_dir, f'{name}_probs.npz'), flush=True)
