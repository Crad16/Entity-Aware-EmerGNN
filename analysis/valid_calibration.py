"""Calibration of the test predictions with a temperature fitted on the validation pairs. CPU only; run from
DrugBank/ after valid_probs.py:

    python ../analysis/valid_calibration.py --train_seeds 0 1 2 --keys llm name

kg_calibration.py and analyze_kg_ablation.py rescale with a temperature fitted on the evaluated test pairs (an
oracle). Here the temperature of every model is fitted once on its validation predictions
(runs/<tag>/valid_probs) and then held fixed. Reported on the 8,195 sampled test pairs: accuracy, confidence
(mean top probability), gap = confidence - accuracy and ECE (15 bins), raw / with the validation temperature /
with the oracle temperature, for
  - all pairs and thirds of the emerging drug's KG degree (cut points as in kg_calibration.py),
  - PD pairs that share a CYP and metabolism pairs that share a CYP,
  - the graph with 50 / 90 / 100 % of the emerging drug's KG edges removed (kg_ablation.py).
Intervals are 95% bootstrap intervals over emerging drugs with the temperature held fixed.
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
p.add_argument('--out', default=None, help='default: results/<dataset>/valid_calibration.json')
cli = p.parse_args()
ac.DATASET = cli.dataset
MODELS = ['baseline'] + cli.keys
LEVELS = ['new50', 'new90', 'new100']


def run(s, *parts):
    return os.path.join(ac.ROOT, 'runs', cli.dataset if s == 0 else f'{cli.dataset}_seed{s}', *parts)


def norm(q):
    q = np.clip(q.astype(np.float64), 1e-7, 1)
    return q / q.sum(1, keepdims=True)


def stem(m):
    return 'baseline_best' if m == 'baseline' else f'{m}-v2_best'


def test_probs(s, m, v, cond='original'):
    if v == 'full' and m != 'baseline':
        z = np.load(run(s, 'gate_interventions', f'{stem(m)}_probs.npz'))
    else:
        z = np.load(run(s, 'kg_ablation', f'{stem(m)}__{v}.npz'))
    return z['rows'], z['weight'].astype(np.float64), norm(z[cond])


train, test = ac.load_ddi('train'), ac.load_ddi('test')
ents = ac.entities()
cyp = ac.cyp_genes(ents)
kg = ac.kg_triples()
binds = ac.drug_binds(kg, ents)
mech = ac.mechanism_class()
degree = np.bincount(kg[:, :2].ravel(), minlength=len(ents))
new_all = ac.emerging_drug(test, train)
cuts = np.percentile(degree[new_all], [100 / 3, 200 / 3])

rows, w, _ = test_probs(cli.train_seeds[0], 'baseline', 'full')
y, new_drug = test[rows, 2], new_all[rows]
third = np.digitize(degree[new_drug], cuts)
has_cyp = np.array([bool(binds[u] & binds[v] & cyp) for u, v, _ in test[rows]])
cls = np.array([mech[t] for t in y])
STRATA = {'all': np.ones(len(y), bool), 'kg_low': third == 0, 'kg_mid': third == 1, 'kg_high': third == 2,
          'pd&shared_cyp': (cls == 'pd') & has_cyp, 'metabolism&shared_cyp': (cls == 'metabolism') & has_cyp}
M = ac.boot_multiplicity(new_drug, cli.n_boot)


def summarize(prob, mask):
    pp = ac.per_pair(prob, y)
    W = M[:, mask] * w[mask]
    d = {k: W @ v[mask] / W.sum(1) for k, v in pp.items() if k in ('acc', 'conf', 'nll')}
    d['gap'] = d['conf'] - d['acc']
    d['ece'] = ac.ece_boot(pp['conf'][mask], pp['acc'][mask], W)
    pt = {k: float(np.average(pp[k][mask], weights=w[mask])) for k in ('acc', 'conf', 'nll')}
    pt['gap'] = pt['conf'] - pt['acc']
    pt['ece'] = float(ac.ece_boot(pp['conf'][mask], pp['acc'][mask], w[mask][None])[0])
    return pt, d


def ci(x):
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


res = {'cut_points': cuts.tolist(), 'strata_pairs': {k: int(m.sum()) for k, m in STRATA.items()}, 'models': {}}
for s in cli.train_seeds:
    for m in MODELS:
        for cond in (['original'] if m == 'baseline' else ['original', 'off']):
            z = np.load(run(s, 'valid_probs', f'{stem(m)}.npz'))
            pv, yv = norm(z[cond]), z['labels']
            ones = np.ones(len(yv))
            T_valid = ac.fit_T(pv, yv, ones)
            vp_raw, vp_T = ac.per_pair(pv, yv), ac.per_pair(ac.scale(pv, T_valid), yv)
            o = {'T_valid': T_valid, 'n_valid': int(len(yv)),
                 'valid': {t: {'acc': float(v['acc'].mean()), 'conf': float(v['conf'].mean()), 'nll': float(v['nll'].mean()),
                               'ece': float(ac.ece_boot(v['conf'], v['acc'], ones[None])[0])}
                           for t, v in [('raw', vp_raw), ('valid_T', vp_T)]}}
            r, _, full = test_probs(s, m, 'full', cond)
            assert (r == rows).all()
            T_oracle = o['T_oracle'] = ac.fit_T(full, y, w)
            cells = {}
            for name, T in [('raw', 1.0), ('valid_T', T_valid), ('oracle_T', T_oracle)]:
                draws = {}
                for st, mask in STRATA.items():
                    pt, draws[st] = summarize(ac.scale(full, T), mask)
                    cells[f'{name}|{st}'] = dict(pt, gap_ci=ci(draws[st]['gap']), ece_ci=ci(draws[st]['ece']))
                cells[f'{name}|kg_low-kg_high'] = {k: {'mean': cells[f'{name}|kg_low'][k] - cells[f'{name}|kg_high'][k],
                                                      'ci': ci(draws['kg_low'][k] - draws['kg_high'][k])} for k in ('acc', 'conf', 'gap')}
                for v in LEVELS:
                    r, _, pr = test_probs(s, m, v, cond)
                    assert (r == rows).all()
                    pt, d = summarize(ac.scale(pr, T), STRATA['all'])
                    cells[f'{name}|{v}'] = dict(pt, gap_ci=ci(d['gap']), ece_ci=ci(d['ece']))
            o['test'] = cells
            res['models'][f'{s}|{m}|{cond}'] = o

print('pairs per stratum:', res['strata_pairs'], '| KG-degree cut points', np.round(cuts, 1).tolist())
print('\n### temperatures and validation calibration')
for k, o in res['models'].items():
    v = o['valid']
    print(f'{k:22s} T_valid {o["T_valid"]:.2f}  T_oracle(test) {o["T_oracle"]:.2f} | validation: acc {v["raw"]["acc"]:.3f} '
          f'conf {v["raw"]["conf"]:.3f} -> {v["valid_T"]["conf"]:.3f}  ECE {v["raw"]["ece"]:.3f} -> {v["valid_T"]["ece"]:.3f}')
cols = list(STRATA) + LEVELS
for metric in ['gap', 'ece']:
    print(f'\n### test sample, {metric} ({"confidence - accuracy" if metric == "gap" else "15 bins"}); rows: raw / validation T / oracle T')
    print(f'{"seed|model|condition":22s} {"":9s}' + ''.join(c[:14].rjust(15) for c in cols))
    for k, o in res['models'].items():
        if k.endswith('off'):
            continue
        for name in ['raw', 'valid_T', 'oracle_T']:
            print(f'{k:22s} {name:9s}' + ''.join(f'{o["test"][f"{name}|{c}"][metric]:+15.3f}' for c in cols))
print('\n### test sample with the validation temperature: accuracy / confidence, and the gap with its 95% interval')
for k, o in res['models'].items():
    if k.endswith('off'):
        continue
    print(k)
    for c in cols:
        x = o['test'][f'valid_T|{c}']
        print(f'    {c:22s} acc {x["acc"]:.3f} conf {x["conf"]:.3f} gap {x["gap"]:+.3f} [{x["gap_ci"][0]:+.3f},{x["gap_ci"][1]:+.3f}]')
    lh = o['test']['valid_T|kg_low-kg_high']
    print('    kg_low - kg_high       ' + '  '.join(f'{q} {lh[q]["mean"]:+.3f} [{lh[q]["ci"][0]:+.3f},{lh[q]["ci"][1]:+.3f}]' for q in ('acc', 'conf', 'gap')))
out = cli.out or os.path.join(ac.ROOT, 'results', cli.dataset, 'valid_calibration.json')
json.dump(res, open(out, 'w'), indent=1)
print('saved', out)
