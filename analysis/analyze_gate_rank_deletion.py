"""Tables for gate_rank_deletion.py. CPU only; run from DrugBank/:

    python ../analysis/analyze_gate_rank_deletion.py --train_seeds 0 1 2 --keys llm name

Per pair, the change in log p(true type) and in accuracy when a deletion set is removed, against the same model
on the full KG (gate_interventions.py for 'original' and 'off', kg_ablation.py's baseline_best__full for the
ungated baseline). Only pairs with at least two shared non-drug neighbours are used. Reported: the mean change
per set, and the contrasts
    top - bottom      high- against low-gated quarter (unmatched)
    top - rand        high-gated quarter against a random quarter (mean of two draws; one for the baseline)
    bottom - rand
    mH - mL           higher- against lower-gated member of degree-matched pairs
A negative contrast means that deleting the entities the gate ranks higher hurts more. The last table repeats
top - bottom and mH - mL on the pairs whose two sets hold the same number of CYP genes (per seed; the sets
differ between seeds, so there is no pooled column). 'original' minus 'off'
of a contrast is what the gate's own multiplier adds beyond the backbone. 95% bootstrap intervals over emerging
drugs; 'pooled' averages each pair over the seeds before resampling (covers test drugs, not training runs).
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
p.add_argument('--out', default=None, help='default: results/<dataset>/gate_rank_deletion.json')
cli = p.parse_args()
ac.DATASET = cli.dataset
SETS = ['top', 'bottom', 'rand0', 'rand1', 'mH', 'mL']
CONDS = ['original', 'off', 'baseline']
CONTRASTS = ['top-bottom', 'top-rand', 'bottom-rand', 'mH-mL']


def run(s, *parts):
    return os.path.join(ac.ROOT, 'runs', cli.dataset if s == 0 else f'{cli.dataset}_seed{s}', *parts)


def norm(q):
    q = np.clip(q.astype(np.float64), 1e-7, 1)
    return q / q.sum(1, keepdims=True)


train, test = ac.load_ddi('train'), ac.load_ddi('test')
mech = ac.mechanism_class()
z0 = np.load(run(cli.train_seeds[0], 'gate_rank_deletion', f'{cli.keys[0]}-v2_best.npz'))
rows, w, eligible = z0['rows'], z0['weight'].astype(np.float64), z0['eligible']
y = test[rows, 2]
cls = np.array([mech[t] for t in y])
CLASSES = {'all': eligible, 'metabolism': eligible & (cls == 'metabolism'), 'pk_other': eligible & (cls == 'pk_other'),
           'pd': eligible & (cls == 'pd'), 'unspecified': eligible & (cls == 'unspecified')}
M = ac.boot_multiplicity(ac.emerging_drug(test[rows], train), cli.n_boot) * w.astype(np.float32)
idx = np.arange(len(y))


def measures(pr, ref):
    return {'dlogp': np.log(pr[idx, y]) - np.log(ref[idx, y]), 'dacc': (pr.argmax(1) == y).astype(float) - (ref.argmax(1) == y)}


def cell(x, mask):
    draws = (M @ np.where(mask, x, 0.0)) / (M @ mask.astype(np.float64))
    return {'mean': float(np.average(x[mask], weights=w[mask])), 'ci': [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]}


# per[(seed, key, cond)][set][measure] -> per-pair array
per, stats, ncyp = {}, {}, {}
for s in cli.train_seeds:
    zb = np.load(run(s, 'kg_ablation', 'baseline_best__full.npz'))
    assert (zb['rows'] == rows).all()
    base_full = norm(zb['original'])
    for k in cli.keys:
        z = np.load(run(s, 'gate_rank_deletion', f'{k}-v2_best.npz'))
        assert (z['rows'] == rows).all() and (z['eligible'] == eligible).all()
        zr = np.load(run(s, 'gate_interventions', f'{k}-v2_best_probs.npz'))
        assert (zr['rows'] == rows).all()
        for cond in ['original', 'off']:
            ref = norm(zr[cond])
            per[s, k, cond] = {st: measures(norm(z[f'{cond}__{st}']), ref) for st in SETS}
        f = run(s, 'gate_rank_deletion', f'baseline_best__sets-{k}-v2_best.npz')
        if os.path.exists(f):
            zz = np.load(f)
            per[s, k, 'baseline'] = {st: measures(norm(zz[f'baseline__{st}']), base_full) for st in SETS if f'baseline__{st}' in zz}
        stats[s, k] = {q: {st: float(np.nanmean(z[f'stat_{st}_{q}'][eligible])) for st in SETS} for q in ('gate', 'logdeg', 'cyp', 'n')}
        ncyp[s, k] = {st: np.rint(np.nan_to_num(z[f'stat_{st}_cyp']) * np.nan_to_num(z[f'stat_{st}_n'])).astype(int) for st in SETS}
        stats[s, k]['tied_matched_pairs'] = float(np.nanmean(z['stat_tied_matched_pairs'][eligible]))


def contrast(d, name, meas):
    a, b = name.split('-')

    def get(q):
        if q == 'rand':
            return np.mean([d[r][meas] for r in ('rand0', 'rand1') if r in d], 0)
        return d[q][meas]
    return get(a) - get(b)


res = {'n_pairs': int(len(y)), 'eligible_pairs': {c: int(m.sum()) for c, m in CLASSES.items()},
       'eligible_weighted_share': float(w[eligible].sum() / w.sum()), 'set_stats': {f'{s}|{k}': v for (s, k), v in stats.items()}, 'rows': {}}
scopes = [str(s) for s in cli.train_seeds] + (['pooled'] if len(cli.train_seeds) > 1 else [])
for k in cli.keys:
    for cond in CONDS + ['original-off']:
        for meas in ('dlogp', 'dacc'):
            for q in SETS + CONTRASTS:
                if cond == 'original-off' and q not in CONTRASTS:
                    continue
                arrs = []
                for s in cli.train_seeds:
                    if cond == 'original-off':
                        arrs.append(contrast(per[s, k, 'original'], q, meas) - contrast(per[s, k, 'off'], q, meas))
                    elif (s, k, cond) not in per or (q in SETS and q not in per[s, k, cond]):
                        arrs.append(None)
                    else:
                        arrs.append(contrast(per[s, k, cond], q, meas) if q in CONTRASTS else per[s, k, cond][q][meas])
                if any(a is None for a in arrs):
                    continue
                for c, mask in CLASSES.items():
                    for name, x in list(zip(scopes, arrs)) + ([('pooled', np.mean(arrs, 0))] if len(arrs) > 1 else []):
                        res['rows'][f'{k}|{cond}|{meas}|{q}|{c}|{name}'] = cell(x, mask)


def f(c):
    star = '*' if c['ci'][0] > 0 or c['ci'][1] < 0 else ' '
    return f'{c["mean"]:+.3f} [{c["ci"][0]:+.3f},{c["ci"][1]:+.3f}]{star}'.rjust(25)


print(f'eligible pairs: {res["eligible_pairs"]} (weighted share of the test set {res["eligible_weighted_share"]:.3f})')
print('\n### deletion sets (means over eligible pairs): gate / log KG degree / share of CYP genes / size')
for (s, k), v in stats.items():
    print(f'seed {s} {k:5s} ' + '  '.join(f'{st} {v["gate"][st]:.2f}/{v["logdeg"][st]:.2f}/{v["cyp"][st]:.2f}/{v["n"][st]:.1f}' for st in SETS)
          + f'  | tied matched pairs {v["tied_matched_pairs"]:.2f}')
for meas, title in [('dlogp', 'change in log p(true type)'), ('dacc', 'change in accuracy')]:
    for c in (['all', 'metabolism', 'pd'] if meas == 'dlogp' else ['all']):
        print(f'\n### {title}, {c} pairs ({res["eligible_pairs"][c]})')
        print(f'{"key | model | quantity":40s}' + ''.join((f'seed {s}' if s != 'pooled' else s).rjust(25) for s in scopes))
        for k in cli.keys:
            for cond in CONDS + ['original-off']:
                for q in (['top', 'bottom', 'rand0', 'mH', 'mL'] if cond != 'original-off' else []) + CONTRASTS:
                    key = f'{k}|{cond}|{meas}|{q}|{c}|'
                    if key + scopes[0] in res['rows']:
                        print(f'{k + " | " + cond + " | " + q:40s}' + ''.join(f(res['rows'][key + s]) for s in scopes))
            print()
print('\n### same number of CYP genes in both sets: change in log p(true type); n = pairs used')
res['cyp_balanced'] = {}
for c in ('metabolism', 'pd'):
    for k in cli.keys:
        for q in ('top-bottom', 'mH-mL'):
            a, b = q.split('-')
            for cond in CONDS + ['original-off']:
                line = f'{c + " | " + k + " | " + cond + " | " + q:44s}'
                for s in cli.train_seeds:
                    mask = CLASSES[c] & (ncyp[s, k][a] == ncyp[s, k][b])
                    if cond == 'original-off':
                        x = contrast(per[s, k, 'original'], q, 'dlogp') - contrast(per[s, k, 'off'], q, 'dlogp')
                    else:
                        x = contrast(per[s, k, cond], q, 'dlogp')
                    r = cell(x, mask)
                    r['n'] = int(mask.sum())
                    res['cyp_balanced'][f'{k}|{cond}|{q}|{c}|{s}'] = r
                    line += f(r) + f' n={r["n"]:4d}'
                print(line)
        print()
out = cli.out or os.path.join(ac.ROOT, 'results', cli.dataset, 'gate_rank_deletion.json')
json.dump(res, open(out, 'w'), indent=1)
print('saved', out)
