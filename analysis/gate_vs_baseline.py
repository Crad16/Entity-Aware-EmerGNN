"""Gated models against the ungated baseline, per training seed and pooled over seeds. CPU only; run from DrugBank/.

    python ../analysis/gate_vs_baseline.py --train_seeds 0 1 2 --keys llm name

Reads the full-KG probabilities (runs/<tag>/gate_interventions/<key>-v2_best_probs.npz, baseline:
runs/<tag>/kg_ablation/baseline_best__full.npz) and the KG-deletion variants (runs/<tag>/kg_ablation). All
seeds use the same 8,195 test pairs. Reports sample-weighted accuracy
  - with the full KG, by stratum (pairs sharing a CYP / sharing no CYP / sharing no gene; metabolism / other types),
  - its change after the CYP edges are removed (cyp - full), by the same strata,
  - with 50 / 90 / 100 % of the emerging drug's KG edges removed,
and the difference gate - baseline with a 95% bootstrap interval over emerging drugs (paired: same pairs, same
resamples). 'pooled' averages each pair's value over the seeds before resampling the drugs, so its interval
covers the choice of test drugs but not the variation between training runs; read it next to the per-seed signs.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ablation_common as ac  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument('--dataset', default='S1_1')
p.add_argument('--train_seeds', type=int, nargs='+', default=[0, 1, 2])
p.add_argument('--keys', nargs='+', default=['llm', 'name'])
p.add_argument('--n_boot', type=int, default=1000)
p.add_argument('--out', default=None, help='default: results/<dataset>/gate_vs_baseline.json')
cli = p.parse_args()
ac.DATASET = cli.dataset
MODELS = ['baseline'] + cli.keys
VARIANTS = ['cyp', 'hub', 'rand0', 'new50', 'new90', 'new100']


def tag(s):
    return cli.dataset if s == 0 else f'{cli.dataset}_seed{s}'


def norm(q):
    q = np.clip(q.astype(np.float64), 1e-7, 1)
    return q / q.sum(1, keepdims=True)


def load(s, model, v):
    d = os.path.join(ac.ROOT, 'runs', tag(s))
    if model == 'baseline':
        f = f'{d}/kg_ablation/baseline_best__{v}.npz'
    elif v == 'full':
        f = f'{d}/gate_interventions/{model}-v2_best_probs.npz'
    else:
        f = f'{d}/kg_ablation/{model}-v2_best__{v}.npz'
    z = np.load(f)
    return z['rows'], z['weight'].astype(np.float64), norm(z['original'])


train, test = ac.load_ddi('train'), ac.load_ddi('test')
ents = ac.entities()
cyp = ac.cyp_genes(ents)
binds = ac.drug_binds(ac.kg_triples(), ents)
mech = ac.mechanism_class()

rows, w, _ = load(cli.train_seeds[0], 'baseline', 'full')
pairs = test[rows]
y = pairs[:, 2]
shared = [binds[u] & binds[v] for u, v, _ in pairs]
has_cyp = np.array([bool(s & cyp) for s in shared])
has_gene = np.array([bool(s) for s in shared])
cls = np.array([mech[t] for t in y])
STRATA = {'all': np.ones(len(y), bool), 'shared_cyp': has_cyp, 'no_shared_cyp': ~has_cyp,
          'no_shared_gene': ~has_gene, 'metabolism': cls == 'metabolism', 'not_metabolism': cls != 'metabolism',
          'metabolism&shared_cyp': (cls == 'metabolism') & has_cyp, 'pd': cls == 'pd', 'pd&shared_cyp': (cls == 'pd') & has_cyp,
          'pd&no_shared_cyp': (cls == 'pd') & ~has_cyp}
M = ac.boot_multiplicity(ac.emerging_drug(pairs, train), cli.n_boot) * w.astype(np.float32)

# correct[seed][model][variant] -> 0/1 per pair
correct = {}
for s in cli.train_seeds:
    correct[s] = {}
    for m in MODELS:
        correct[s][m] = {}
        for v in ['full'] + VARIANTS:
            r, _, pr = load(s, m, v)
            idx = np.searchsorted(r, rows)
            assert (r[idx] == rows).all(), (s, m, v)
            correct[s][m][v] = (pr[idx].argmax(1) == y).astype(np.float64)


def cell(x, mask):
    xm = np.where(mask, x, 0.0)
    draws = (M @ xm) / (M @ mask.astype(np.float64))
    return {'mean': float(np.average(x[mask], weights=w[mask])), 'ci': [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]}


def value(s, m, q):
    """per-pair quantity: 'full', '<variant>' (accuracy) or 'd<variant>' (variant - full)."""
    c = correct[s][m]
    return c[q[1:]] - c['full'] if q.startswith('d') else c[q]


QUANT = [('full', list(STRATA)), ('dcyp', list(STRATA)), ('dhub', ['all', 'shared_cyp']), ('drand0', ['all', 'shared_cyp']),
         ('new50', ['all']), ('new90', ['all', 'shared_cyp', 'no_shared_cyp']), ('new100', ['all', 'shared_cyp', 'no_shared_cyp'])]
res = {'n_pairs': int(len(y)), 'strata_pairs': {k: int(m.sum()) for k, m in STRATA.items()}, 'seeds': cli.train_seeds, 'rows': {}}
scopes = [str(s) for s in cli.train_seeds] + (['pooled'] if len(cli.train_seeds) > 1 else [])
for q, strata in QUANT:
    for st in strata:
        mask = STRATA[st]
        for m in MODELS:
            per_seed = [value(s, m, q) for s in cli.train_seeds]
            base = [value(s, 'baseline', q) for s in cli.train_seeds]
            for name, x, b in [(str(s), per_seed[i], base[i]) for i, s in enumerate(cli.train_seeds)] + \
                              ([('pooled', np.mean(per_seed, 0), np.mean(base, 0))] if len(cli.train_seeds) > 1 else []):
                res['rows'][f'{q}|{st}|{m}|{name}'] = cell(x, mask)
                if m != 'baseline':
                    res['rows'][f'{q}|{st}|{m}-baseline|{name}'] = cell(x - b, mask)


def f(c, signed):
    star = '*' if signed and (c['ci'][0] > 0 or c['ci'][1] < 0) else ' '
    return (f'{c["mean"]:+.3f} [{c["ci"][0]:+.3f},{c["ci"][1]:+.3f}]{star}' if signed else f'{c["mean"]:.3f}').rjust(24)


print('pairs per stratum:', res['strata_pairs'])
print(f'{"quantity | stratum | model":52s}' + ''.join(f'seed {s}'.rjust(24) if s != 'pooled' else s.rjust(24) for s in scopes))
for q, strata in QUANT:
    for st in strata:
        for m in MODELS + [f'{k}-baseline' for k in cli.keys]:
            signed = q.startswith('d') or '-' in m
            print(f'{q + " | " + st + " | " + m:52s}' + ''.join(f(res['rows'][f'{q}|{st}|{m}|{s}'], signed) for s in scopes))
        print()
out = cli.out or os.path.join(ac.ROOT, 'results', cli.dataset, 'gate_vs_baseline.json')
json.dump(res, open(out, 'w'), indent=1)
print('saved', out)
