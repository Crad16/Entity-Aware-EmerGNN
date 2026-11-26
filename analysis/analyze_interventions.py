"""Follow-up analysis of gate_interventions.py outputs (CPU only). Run from DrugBank/:

    python ../analysis/analyze_interventions.py

Reads runs/<tag>/gate_interventions/<key>-v2_best_probs.npz (<key>-v2_last_probs.npz for a key given
as '<key>@last') and reports, per checkpoint:
  1. paired contrasts between conditions (change in log-prob of the true type), with 95% bootstrap
     intervals over emerging drugs: key_perm - off, shuffle_all - off, mean - off
  2. whether a same-type donor query moves predictions less than a random donor (KL to the original),
     overall and only over recipients whose donor shares no drug with them in either shuffle
  3. the key_perm effect split by whether the two drugs bind a common CYP in the KG
"""
import argparse
import collections
import csv
import json
import re

import numpy as np

from ablation_common import ENT_TSV, probs_stem, res_path, run_dir, setup

METAB, PK_OTHER = {46, 3}, {72, 74, 76, 10}
N_DRUG = 1710
EPS = 1e-6

p = argparse.ArgumentParser()
p.add_argument('--dataset', default='S1_1')
p.add_argument('--dir', default=None, help='default: runs/<tag>/gate_interventions')
p.add_argument('--keys', nargs='+', default=['llm', 'id', 'random', 'name'])
p.add_argument('--reps', type=int, default=3)
p.add_argument('--seed', type=int, default=0)
p.add_argument('--out', default=None, help='default: results/<tag>/gate_interventions/followup.json')
cli = setup(p)
cli.dir = cli.dir or run_dir('gate_interventions')
cli.out = cli.out or res_path('gate_interventions/followup.json')

load = lambda s: np.array([l.split() for l in open(f'data/{cli.dataset}/{s}_ddi.txt')], dtype=int)  # noqa: E731
train, test = load('train'), load('test')
train_ent = set(train[:, :2].ravel())
group = np.array(['metabolism' if r in METAB else 'pk_other' if r in PK_OTHER else 'non_pk' for r in test[:, 2]])

rel = json.load(open('data/relation2id.json'))
cbg = [int(k) for k, v in rel.items() if v == 'CbG'][0]
ents = {int(r['ent_id']): r for r in csv.DictReader(open(ENT_TSV), delimiter='\t')}
cyp = {e for e, r in ents.items() if r['kind'] == 'Gene' and re.match(r'CYP\d', r['name'])}
binds = collections.defaultdict(set)
for split in ['train', 'valid', 'test']:
    for line in open(f'data/{cli.dataset}/{split}_KG.txt'):
        h, t, r = map(int, line.split())
        if r == cbg and h < N_DRUG <= t:
            binds[h].add(t)


def boot_ci(values, weights, clusters, n=1000):
    ids, inv = np.unique(clusters, return_inverse=True)
    sums, counts = np.bincount(inv, weights=values * weights), np.bincount(inv, weights=weights)
    pick = np.random.RandomState(0).randint(0, len(ids), size=(n, len(ids)))
    means = sums[pick].sum(1) / counts[pick].sum(1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def stat(values, weights, clusters):
    return {'mean': float(np.average(values, weights=weights)), 'ci': boot_ci(values, weights, clusters),
            'n': int(len(values))}


res = {}
for key in cli.keys:
    d = np.load(f'{cli.dir}/{probs_stem(key)}_probs.npz')
    rows, w = d['rows'], d['weight'].astype(float)
    labels, groups = test[rows, 2], group[rows]
    heads, tails = test[rows, 0], test[rows, 1]
    new_drug = np.where(np.isin(heads, list(train_ent)), tails, heads)
    P = {c: np.clip(d[c].astype(np.float64), EPS, 1) for c in d.files if c not in ('rows', 'weight')}
    P = {c: p / p.sum(1, keepdims=True) for c, p in P.items()}
    ref = P['original']
    logp = {c: np.log(p[np.arange(len(rows)), labels]) for c, p in P.items()}
    kl = {c: (ref * (np.log(ref) - np.log(p))).sum(1) for c, p in P.items()}
    avg = lambda f, c: np.mean([f[f'{c}_{k}'] for k in range(cli.reps)], axis=0)  # noqa: E731
    lp = {c: logp[c] for c in ['original', 'off', 'mean']}
    lp.update({c: avg(logp, c) for c in ['shuffle_all', 'shuffle_type', 'key_perm']})
    shared_cyp = np.array([bool(binds[h] & binds[t] & cyp) for h, t in zip(heads, tails)])

    # donors exactly as in gate_interventions.py
    def donors(within_type, rep):
        r = np.random.RandomState(1000 * cli.seed + 10 * rep + within_type)
        out = np.arange(len(rows))
        ks = labels if within_type else np.zeros(len(rows), dtype=int)
        for k in np.unique(ks):
            idx = np.where(ks == k)[0]
            out[idx] = idx[r.permutation(len(idx))]
        return out

    def shares_drug(dn):
        return (heads[dn] == heads) | (heads[dn] == tails) | (tails[dn] == heads) | (tails[dn] == tails)

    out = {}
    for g in ['metabolism', 'non_pk']:
        m = groups == g
        o = {}
        for c in ['key_perm', 'shuffle_all', 'shuffle_type', 'mean', 'original']:
            o[f'{c}-off'] = stat((lp[c] - lp['off'])[m], w[m], new_drug[m])
        o['flip_mean_vs_off'] = float(np.average((P['mean'].argmax(1) != P['off'].argmax(1))[m], weights=w[m]))

        # same-type vs random donor
        share = {}
        kl_diff, clean = [], []
        for k in range(cli.reps):
            sa, st = shares_drug(donors(0, k)), shares_drug(donors(1, k))
            share.setdefault('all', []).append(np.average(sa[m], weights=w[m]))
            share.setdefault('type', []).append(np.average(st[m], weights=w[m]))
            kl_diff.append(kl[f'shuffle_type_{k}'] - kl[f'shuffle_all_{k}'])
            clean.append(~sa & ~st)
        kl_diff = np.array(kl_diff)
        o['donor_shares_a_drug'] = {s: float(np.mean(v)) for s, v in share.items()}
        o['kl_type-all'] = stat(kl_diff.mean(0)[m], w[m], new_drug[m])
        cl = np.array(clean)
        mc = m & cl.any(0)
        vals = np.array([np.average(kl_diff[k][mc & cl[k]], weights=w[mc & cl[k]]) for k in range(cli.reps)])
        o['kl_type-all_no_shared_drug'] = {'mean_over_reps': float(vals.mean()), 'per_rep': vals.tolist()}

        # key_perm by shared CYP
        o['shares_a_cyp'] = float(np.average(shared_cyp[m], weights=w[m]))
        for name, s in [('with_shared_cyp', shared_cyp), ('without_shared_cyp', ~shared_cyp)]:
            mm = m & s
            if mm.sum() >= 20:
                o[f'key_perm-original_{name}'] = stat((lp['key_perm'] - lp['original'])[mm], w[mm], new_drug[mm])
                o[f'off-original_{name}'] = stat((lp['off'] - lp['original'])[mm], w[mm], new_drug[mm])
        out[g] = o
    for t in sorted(METAB):
        m = labels == t
        out[f'type_{t}'] = {c: stat((lp[c] - lp['original'])[m], w[m], new_drug[m]) for c in ['off', 'key_perm']}
    res[key] = out

fmt = lambda s: f'{s["mean"]:+.3f} [{s["ci"][0]:+.3f},{s["ci"][1]:+.3f}]'  # noqa: E731
for key, out in res.items():
    print(f'\n== {key}')
    for g in ['metabolism', 'non_pk']:
        o = out[g]
        print(f'  {g}: key_perm-off {fmt(o["key_perm-off"])}  shuffle_all-off {fmt(o["shuffle_all-off"])}  '
              f'shuffle_type-off {fmt(o["shuffle_type-off"])}  mean-off {fmt(o["mean-off"])}  '
              f'flip(mean vs off) {o["flip_mean_vs_off"]:.3f}')
        print(f'    donor shares a drug: all {o["donor_shares_a_drug"]["all"]:.3f} type {o["donor_shares_a_drug"]["type"]:.3f}; '
              f'KL type-all {fmt(o["kl_type-all"])}; no shared drug {o["kl_type-all_no_shared_drug"]["mean_over_reps"]:+.4f} '
              f'{np.round(o["kl_type-all_no_shared_drug"]["per_rep"], 4).tolist()}')
        print(f'    shares a CYP {o["shares_a_cyp"]:.3f}; ' + '; '.join(
            f'{k.replace("-original_", " ")} {fmt(v)} n={v["n"]}' for k, v in o.items() if '-original_' in k))
    for t in sorted(METAB):
        print(f'  type {t}: off {fmt(out[f"type_{t}"]["off"])}  key_perm {fmt(out[f"type_{t}"]["key_perm"])}  '
              f'n={out[f"type_{t}"]["off"]["n"]}')
json.dump(res, open(cli.out, 'w'), indent=1)
print('saved', cli.out)
