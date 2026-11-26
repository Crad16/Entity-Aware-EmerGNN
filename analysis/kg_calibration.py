"""KG amount vs calibration (ablation M1, with P3). CPU only; run from DrugBank/ after
gate_interventions.py:

    python ../analysis/kg_calibration.py

Does the model know when the knowledge graph says little about a new drug? Test pairs are split
into thirds by the KG degree of their emerging drug (its edges in the KG available at test time;
cut points are the tertiles over all 32,322 test pairs). Per third we compare accuracy with
confidence (mean top probability), ECE (15 equal-width bins), Brier score and NLL. If accuracy
drops in the KG-poor third and confidence does not, the model is overconfident exactly where its
evidence is thin. A single temperature fitted on all evaluated pairs (an oracle that favours the
model) checks whether a global rescaling removes the gap. A second axis splits pairs by the number
of non-drug entities adjacent to both drugs (pair-level KG evidence).

No ungated EmerGNN checkpoint exists. 'off' (the gated model with every gate = 1, which computes
exactly the EmerGNN forward pass) stands in for it, and the four key types give four such models.
P3: per third, the gate's gain NLL(off) - NLL(original), and NLL differences between key types.

Probabilities come from runs/<tag>/gate_interventions/<key>-v2_best_probs.npz (8,195 type-stratified test
pairs, weighted back to the full test set; '<key>@last' reads <key>-v2_last_probs.npz). Intervals are 95% bootstrap intervals over emerging
drugs; differences between thirds use the same bootstrap draws.
"""
import argparse
import collections
import json

import numpy as np

from ablation_common import KEYS, N_DRUG, boot_multiplicity, ece_boot, emerging_drug, fit_T, fmt, kg_triples, \
    load_ddi, load_probs, per_pair, res_path, scale, setup, stat

p = argparse.ArgumentParser()
p.add_argument('--n_boot', type=int, default=1000)
p.add_argument('--out', default=None, help='default: results/<tag>/kg_calibration.json')
cli = setup(p)
cli.out = cli.out or res_path('kg_calibration.json')

train, test = load_ddi('train'), load_ddi('test')
kg = kg_triples()
degree = np.bincount(kg[:, :2].ravel(), minlength=kg[:, :2].max() + 1)
nbrs = collections.defaultdict(set)
for h, t, _ in kg:
    if h < N_DRUG <= t:
        nbrs[h].add(t)
    elif t < N_DRUG <= h:
        nbrs[t].add(h)
new_all = emerging_drug(test, train)
deg_all = degree[new_all]
shared_all = np.array([len(nbrs[u] & nbrs[v]) for u, v, _ in test])
AXES = {'drug_degree': (deg_all, np.percentile(deg_all, [100 / 3, 200 / 3])),
        'shared_neighbours': (shared_all, np.percentile(shared_all, [100 / 3, 200 / 3]))}
BINS = ['low', 'mid', 'high']
print('cut points: ' + '; '.join(f'{a} {np.round(c, 1).tolist()}' for a, (_, c) in AXES.items()))


rows, w, _ = load_probs(KEYS[0])
y, new_drug = test[rows, 2], new_all[rows]
M = boot_multiplicity(new_drug, cli.n_boot)
bins = {a: np.digitize(v[rows], cuts) for a, (v, cuts) in AXES.items()}
very_low = deg_all[rows] <= 10


def summarize(prob, mask):
    """Metrics on the pairs in mask, with bootstrap draws for differences."""
    pp = per_pair(prob, y)
    W = M[:, mask] * w[mask]
    draws = {k: W @ v[mask] / W.sum(1) for k, v in pp.items()}
    draws['gap'] = draws['conf'] - draws['acc']
    draws['ece'] = ece_boot(pp['conf'][mask], pp['acc'][mask], W)
    point = {k: float(np.average(v[mask], weights=w[mask])) for k, v in pp.items()}
    point['gap'] = point['conf'] - point['acc']
    point['ece'] = float(ece_boot(pp['conf'][mask], pp['acc'][mask], w[mask][None])[0])
    return point, draws


def ci(d):
    return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


res = {'cut_points': {a: c.tolist() for a, (_, c) in AXES.items()}, 'models': {}}
nll_by = {}
for key in KEYS:
    rows_k, _, probs = load_probs(key)
    assert (rows_k == rows).all()
    out = {}
    for cond in ['original', 'off']:
        prob = probs[cond]
        T = fit_T(prob, y, w)
        o = {'T': T}
        for tag, pr in [('raw', prob), ('temp', scale(prob, T))]:
            for axis, b in bins.items():
                cells, draws = {}, {}
                for i, name in enumerate(BINS):
                    m = b == i
                    cells[name], draws[name] = summarize(pr, m)
                    cells[name].update(n_pairs=int(m.sum()), n_drugs=int(len(set(new_drug[m]))),
                                       **{f'{k}_ci': ci(v) for k, v in draws[name].items()})
                cells['low-high'] = {k: {'mean': cells['low'][k] - cells['high'][k],
                                         'ci': ci(draws['low'][k] - draws['high'][k])}
                                     for k in ['acc', 'conf', 'gap', 'ece', 'nll']}
                o[f'{tag}_{axis}'] = cells
            (lo, lo_d), (rest, rest_d) = summarize(pr, very_low), summarize(pr, ~very_low)
            o[f'{tag}_degree<=10'] = dict(lo, n_pairs=int(very_low.sum()), n_drugs=int(len(set(new_drug[very_low]))),
                                          **{f'{k}_ci': ci(v) for k, v in lo_d.items()},
                                          minus_rest={k: {'mean': lo[k] - rest[k], 'ci': ci(lo_d[k] - rest_d[k])}
                                                      for k in ['acc', 'conf', 'gap']})
        out[cond] = o
        nll_by[key, cond] = per_pair(prob, y)['nll']
    # P3: gain of the gate over its own backbone, per third of drug degree
    gain = nll_by[key, 'off'] - nll_by[key, 'original']
    out['gate_gain_nll'] = {name: stat(gain[bins['drug_degree'] == i], w[bins['drug_degree'] == i],
                                       new_drug[bins['drug_degree'] == i]) for i, name in enumerate(BINS)}
    res['models'][key] = out

# P3: key types against each other (different training runs, one seed each)
res['key_vs_key_nll'] = {}
for a, b_ in [('id', 'llm'), ('random', 'llm'), ('name', 'llm'), ('random', 'id')]:
    if a not in KEYS or b_ not in KEYS:
        continue
    d = nll_by[a, 'original'] - nll_by[b_, 'original']        # > 0: second key better
    res['key_vs_key_nll'][f'{a}-{b_}'] = {name: stat(d[bins['drug_degree'] == i], w[bins['drug_degree'] == i],
                                                     new_drug[bins['drug_degree'] == i]) for i, name in enumerate(BINS)}

# ------------------------------------------------------------------ report
for axis in AXES:
    print(f'\n### axis: {axis}   (low / mid / high thirds; low-high with 95% CI)')
    for key in KEYS:
        for cond in ['original', 'off']:
            o = res['models'][key][cond]
            for tag in ['raw', 'temp']:
                c = o[f'{tag}_{axis}']
                line = '  '.join(f'{n}: acc {c[n]["acc"]:.3f} conf {c[n]["conf"]:.3f} ece {c[n]["ece"]:.3f}'
                                 for n in BINS)
                lh = c['low-high']
                print(f'{key:6s} {cond:8s} {tag:4s} | {line} | low-high acc {fmt(lh["acc"])} conf {fmt(lh["conf"])} '
                      f'gap {fmt(lh["gap"])}' + (f' (T={o["T"]:.2f})' if tag == 'temp' else ''))
v = res['models'][KEYS[0]]['original']['raw_degree<=10']
print(f'\nKG-poor drugs (degree <= 10; {v["n_pairs"]} pairs, {v["n_drugs"]} drugs) minus all other pairs:')
for key in KEYS:
    for cond in ['original', 'off']:
        v = res['models'][key][cond]['raw_degree<=10']
        print(f'  {key:6s} {cond:8s} acc {v["acc"]:.3f} conf {v["conf"]:.3f} | minus rest: '
              + '  '.join(f'{k} {fmt(v["minus_rest"][k])}' for k in ['acc', 'conf', 'gap']))
print('\nP3 gate gain NLL(off) - NLL(original) by drug-degree third (> 0: gate helps)')
for key in KEYS:
    print(f'  {key:6s} ' + '  '.join(f'{n} {fmt(res["models"][key]["gate_gain_nll"][n])}' for n in BINS))
print('P3 key vs key NLL difference (> 0: second key better)')
for k, v in res['key_vs_key_nll'].items():
    print(f'  {k:10s} ' + '  '.join(f'{n} {fmt(v[n])}' for n in BINS))
json.dump(res, open(cli.out, 'w'), indent=1)
print('saved', cli.out)
