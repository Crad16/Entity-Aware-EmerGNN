"""Analysis of kg_ablation.py outputs: F4 (CYP path removal) and M1 augmentation (KG-poor emerging
drugs). CPU only; run from DrugBank/ after copying runs/<tag>/kg_ablation from the GPU machine:

    python ../analysis/analyze_kg_ablation.py

Full-KG predictions on the same 8,195 pairs come from runs/<tag>/gate_interventions. Every model is read
with its gate ('original') and without it ('off' = the EmerGNN forward pass); 'emergnn_x4' averages
the per-pair quantity over the four 'off' models (four backbones co-trained with different keys).
Intervals are 95% bootstrap intervals over emerging drugs; differences use the same draws.

F4, per mechanism class (metabolism / other PK / PD / unspecified, as in stratify.py)
  dlogp        change in log-probability of the true type when the variant's edges are removed
  specificity  dlogp(metabolism) - dlogp(PD); also among pairs that lost a shared bound gene only,
               where both classes lost the same kind of evidence
  vs control   dlogp(cyp) - dlogp(hub), dlogp(cyp) - dlogp(rand, mean of draws)
  gate         dlogp(cyp, original) - dlogp(cyp, off)
  pull         share of pairs predicted as a metabolism type (46, 3) on the full KG and after the
               cyp cut, by mechanism class x shared CYP (S1: a shared CYP pulls PD pairs to metabolism);
               among shared-CYP pairs also the precision and recall of the metabolism predictions
  lost gene    dlogp of metabolism and PD pairs that did / did not lose a shared bound gene, per
               variant: the symmetric check whether losing a shared non-CYP gene (mostly receptors,
               hub) hurts PD pairs the way losing a shared CYP hurts metabolism pairs
M1, per removal level (0 / 50 / 90 / 100% of the emerging drug's KG edges)
  accuracy, confidence, gap, ECE, NLL, raw and with the temperature fitted on the full KG, and their
  changes against the full KG. If accuracy falls and confidence does not, the model does not notice
  that its evidence is gone. Reference: the existing drug's most frequent training type, a prediction
  that needs no KG at all. Key vs key: difference between two models in the accuracy drop (paired).
"""
import argparse
import collections
import json
import os

import numpy as np

from ablation_common import KEYS, METAB, boot_multiplicity, cyp_genes, drug_binds, ece_boot, emerging_drug, \
    entities, fit_T, load_ddi, load_probs, mechanism_class, per_pair, res_path, run_dir, scale, setup
from kg_variants import F4, M1, build

p = argparse.ArgumentParser()
p.add_argument('--dir', default=None, help='default: runs/<tag>/kg_ablation')
p.add_argument('--n_boot', type=int, default=1000)
p.add_argument('--out', default=None, help='default: results/<tag>/kg_ablation.json')
cli = setup(p)
cli.dir = cli.dir or run_dir('kg_ablation')
cli.out = cli.out or res_path('kg_ablation.json')
np.seterr(invalid='ignore', divide='ignore')
CLASSES = ['metabolism', 'pk_other', 'pd', 'unspecified']
CONDS = ['original', 'off']


def load_variant(key, v):
    f = f'{cli.dir}/{key}-v2_best__{v}.npz'
    if not os.path.exists(f):
        return None, None
    z = np.load(f)
    out = {}
    for c in CONDS:
        q = np.clip(z[c].astype(np.float64), 1e-7, 1)
        out[c] = q / q.sum(1, keepdims=True)
    return z['rows'], out


variants = [v for v in F4 + M1 if any(os.path.exists(f'{cli.dir}/{k}-v2_best__{v}.npz') for k in KEYS)]
rows = load_variant(*next((k, v) for v in variants for k in KEYS if load_variant(k, v)[0] is not None))[0]
print('variants found:', variants, f'({len(rows)} pairs)')

# ------------------------------------------------------------------ pair covariates
train, test = load_ddi('train'), load_ddi('test')
mech = mechanism_class()
pairs = test[rows]
y = pairs[:, 2]
cls = np.array([mech[r] for r in y])
new_drug = emerging_drug(test, train)[rows]
ents = entities()
cyp = cyp_genes(ents)
kg, keep, info = build(F4 + M1)
binds = drug_binds(kg, ents)
shared = [binds[u] & binds[v] for u, v, _ in pairs]
shares_cyp = np.array([bool(s & cyp) for s in shared])
lost = {}
for v in F4:
    b = drug_binds(kg[keep[v]], ents)
    lost[v] = np.array([len(s) > len(b[u] & b[w]) for s, (u, w, _) in zip(shared, pairs)])

ref, probs = {}, {}
for key in KEYS:
    r, w_ref, pr = load_probs(key)
    idx = np.searchsorted(r, rows)
    assert (r[idx] == rows).all()
    ref[key] = {c: pr[c][idx] for c in CONDS}
    w = w_ref[idx]
    for v in variants:
        rv, pv = load_variant(key, v)
        if pv is not None:
            assert (rv == rows).all()
            probs[key, v] = pv
M = boot_multiplicity(new_drug, cli.n_boot)


def ci(d):
    return [float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))]


def wmean(x, m):
    W = M[:, m] * w[m]
    return float(np.average(x[m], weights=w[m])), W @ x[m] / W.sum(1)


def cell(x, m):
    pt, d = wmean(x, m)
    return {'mean': pt, 'ci': ci(d), 'n': int(m.sum())}


def diff(x1, m1, x2, m2):
    (a, da), (b, db) = wmean(x1, m1), wmean(x2, m2)
    return {'mean': a - b, 'ci': ci(da - db)}


def fmt(s):
    star = '*' if s['ci'][0] > 0 or s['ci'][1] < 0 else ' '
    return f'{s["mean"]:+.3f} [{s["ci"][0]:+.3f},{s["ci"][1]:+.3f}]{star}'


def logp(q):
    return np.log(q[np.arange(len(y)), y])


res = {'n_pairs': int(len(rows)), 'variants': info, 'f4': {}, 'm1': {}}
cm = {c: cls == c for c in CLASSES}

res['accuracy'] = {f'{key}/{c}': {v: float(np.average((pr[c].argmax(1) == y), weights=w))
                                  for v, pr in [('full', ref[key])] + [(v, probs[key, v]) for v in variants
                                                                       if (key, v) in probs]}
                   for key in KEYS for c in CONDS}
print('\n### accuracy per variant')
for name, a in res['accuracy'].items():
    print(f'{name:16s} ' + '  '.join(f'{v} {x:.3f}' for v, x in a.items()))

# ------------------------------------------------------------------ F4
f4 = [v for v in F4 if v in variants]
if 'cyp' in f4:
    q = {}                                        # (model, cond) -> per-pair arrays
    for key in KEYS:
        for c in CONDS:
            d = {'metab_pred_full': np.isin(ref[key][c].argmax(1), list(METAB)).astype(float)}
            for v in f4:
                if (key, v) in probs:
                    d[f'dlogp_{v}'] = logp(probs[key, v][c]) - logp(ref[key][c])
                    d[f'metab_pred_{v}'] = np.isin(probs[key, v][c].argmax(1), list(METAB)).astype(float)
            q[key, c] = d
    common = set.intersection(*(set(q[k, 'off']) for k in KEYS))
    q['emergnn_x4', 'off'] = {n: np.mean([q[k, 'off'][n] for k in KEYS], 0) for n in common}
    rand = [v for v in f4 if v.startswith('rand')]
    for (key, c), d in q.items():
        if 'dlogp_cyp' not in d:
            continue
        o = {f'dlogp_{v}': {cl: cell(d[f'dlogp_{v}'], cm[cl]) for cl in CLASSES} for v in f4 if f'dlogp_{v}' in d}
        x = d['dlogp_cyp']
        o['specificity'] = {'metab-pd': diff(x, cm['metabolism'], x, cm['pd']),
                            'metab-pd_lost_shared': diff(x, cm['metabolism'] & lost['cyp'], x, cm['pd'] & lost['cyp']),
                            'metab-pk_other': diff(x, cm['metabolism'], x, cm['pk_other'])}
        for ctrl, xc in [('hub', d.get('dlogp_hub')),
                         ('rand', np.mean([d[f'dlogp_{v}'] for v in rand], 0) if all(f'dlogp_{v}' in d for v in rand) and rand else None)]:
            if xc is not None:
                o[f'cyp-{ctrl}'] = {cl: cell(x - xc, cm[cl]) for cl in CLASSES}
                o[f'cyp-{ctrl}_metab-pd'] = diff(x - xc, cm['metabolism'], x - xc, cm['pd'])
        if c == 'original' and 'dlogp_cyp' in q[key, 'off']:
            g = x - q[key, 'off']['dlogp_cyp']
            o['gate_original-off'] = {cl: cell(g, cm[cl]) for cl in CLASSES}
        o['pull'] = {}
        for cl in CLASSES:
            for s_name, s in [('shared_cyp', shares_cyp), ('no_shared_cyp', ~shares_cyp)]:
                m = cm[cl] & s
                if m.sum() >= 20:
                    o['pull'][f'{cl}&{s_name}'] = {'full': cell(d['metab_pred_full'], m), 'cyp': cell(d['metab_pred_cyp'], m),
                                                   'change': cell(d['metab_pred_cyp'] - d['metab_pred_full'], m)}
        if key != 'emergnn_x4':                   # needs hard predictions
            o['shared_cyp_metab_pred'] = {
                v: {'rate': cell(d[f'metab_pred_{v}'], shares_cyp),
                    'precision': cell(cm['metabolism'].astype(float), shares_cyp & (d[f'metab_pred_{v}'] > 0)),
                    'recall': cell(d[f'metab_pred_{v}'], shares_cyp & cm['metabolism'])} for v in ['full', 'cyp']}
        o['by_lost_gene'] = {f'{v}/{cl}/{s}': cell(d[f'dlogp_{v}'], cm[cl] & (lost[v] if s == 'lost' else ~lost[v]))
                             for v in f4 if f'dlogp_{v}' in d for cl in ['metabolism', 'pd'] for s in ['lost', 'kept']}
        res['f4'][f'{key}/{c}'] = o
    res['f4_lost_shared_gene'] = {v: {cl: float(np.average(lost[v][cm[cl]], weights=w[cm[cl]])) for cl in CLASSES}
                                  for v in F4}

    print('\n### F4 share of pairs that lose a shared bound gene')
    for v, r_ in res['f4_lost_shared_gene'].items():
        print(f'  {v:6s} ' + '  '.join(f'{cl} {r_[cl]:.2f}' for cl in CLASSES))
    print('\n### F4 dlogp of the true type under the cyp cut (< 0: prediction got worse)')
    print(f'{"model":20s} ' + ' '.join(f'{cl:>24s}' for cl in CLASSES) + f' {"metab - PD":>24s} {"metab - PD (lost gene)":>24s}')
    for name, o in res['f4'].items():
        print(f'{name:20s} ' + ' '.join(f'{fmt(o["dlogp_cyp"][cl]):>24s}' for cl in CLASSES)
              + f' {fmt(o["specificity"]["metab-pd"]):>24s} {fmt(o["specificity"]["metab-pd_lost_shared"]):>24s}')
    for ctrl in ['hub', 'rand']:
        print(f'\n### F4 dlogp under the {ctrl} control, and cyp - {ctrl}')
        for name, o in res['f4'].items():
            if f'cyp-{ctrl}' not in o:
                continue
            ctrl_d = o['dlogp_hub'] if ctrl == 'hub' else {cl: {'mean': np.mean([o[f'dlogp_{v}'][cl]['mean'] for v in rand])}
                                                           for cl in CLASSES}
            print(f'{name:20s} {ctrl}: ' + ' '.join(f'{cl[:5]} {ctrl_d[cl]["mean"]:+.3f}' for cl in CLASSES)
                  + ' | cyp-ctrl: ' + ' '.join(f'{cl[:5]} {fmt(o[f"cyp-{ctrl}"][cl])}' for cl in ['metabolism', 'pd'])
                  + f' | metab-PD {fmt(o[f"cyp-{ctrl}_metab-pd"])}')
    print('\n### F4 gate: dlogp(cyp, original) - dlogp(cyp, off)  (< 0: the gate makes the model rely more on CYP edges)')
    for name, o in res['f4'].items():
        if 'gate_original-off' in o:
            print(f'{name:20s} ' + ' '.join(f'{cl[:5]} {fmt(o["gate_original-off"][cl])}' for cl in CLASSES))
    print('\n### F4 pull: share predicted as a metabolism type, full KG -> cyp cut')
    for name, o in res['f4'].items():
        print(f'{name:20s} ' + '  '.join(f'{k} {o["pull"][k]["full"]["mean"]:.3f}->{o["pull"][k]["cyp"]["mean"]:.3f}'
                                         for k in ['metabolism&shared_cyp', 'pk_other&shared_cyp', 'pd&shared_cyp',
                                                   'pd&no_shared_cyp'] if k in o['pull']))
    print('\n### F4 shared-CYP pairs: metabolism predictions, full KG -> cyp cut')
    for name, o in res['f4'].items():
        if 'shared_cyp_metab_pred' in o:
            s = o['shared_cyp_metab_pred']
            print(f'{name:20s} ' + '  '.join(f'{k} {s["full"][k]["mean"]:.3f}->{s["cyp"][k]["mean"]:.3f}'
                                             for k in ['rate', 'precision', 'recall']))
    print('\n### F4 dlogp of pairs that lost a shared bound gene (lost) or not (kept), per variant')
    for name, o in res['f4'].items():
        print(f'{name:20s} ' + '  '.join(f'{k} {fmt(o["by_lost_gene"][k])}' for k in
                                         ['cyp/metabolism/lost', 'cyp/pd/lost', 'hub/pd/lost', 'hub/pd/kept',
                                          'hub/metabolism/lost', 'rand0/pd/lost'] if k in o['by_lost_gene']))
    print('  n pairs: ' + '  '.join(f'{v}/{cl} {int((cm[cl] & lost[v]).sum())}' for v in f4 for cl in ['metabolism', 'pd']))

# ------------------------------------------------------------------ M1
levels = ['full'] + [v for v in M1 if v in variants]
if len(levels) > 1:
    def summarize(prob, mask):
        pp = per_pair(prob, y)
        W = M[:, mask] * w[mask]
        draws = {k: W @ v[mask] / W.sum(1) for k, v in pp.items()}
        draws['gap'] = draws['conf'] - draws['acc']
        draws['ece'] = ece_boot(pp['conf'][mask], pp['acc'][mask], W)
        point = {k: float(np.average(v[mask], weights=w[mask])) for k, v in pp.items()}
        point['gap'] = point['conf'] - point['acc']
        point['ece'] = float(ece_boot(pp['conf'][mask], pp['acc'][mask], w[mask][None])[0])
        return point, draws

    every = np.ones(len(rows), bool)
    for key in KEYS:
        for c in CONDS:
            if not all((key, v) in probs for v in levels[1:]):
                continue
            full = ref[key][c]
            T = fit_T(full, y, w)
            o = {'T_full_kg': T}
            for tag in ['raw', 'temp']:
                tf = (lambda x: x) if tag == 'raw' else (lambda x: scale(x, T))  # noqa: E731
                base, base_d = summarize(tf(full), every)
                cells = {}
                for lv in levels:
                    pr = tf(full if lv == 'full' else probs[key, lv][c])
                    pt, dr = summarize(pr, every)
                    cells[lv] = dict(pt, minus_full={k: {'mean': pt[k] - base[k], 'ci': ci(dr[k] - base_d[k])}
                                                     for k in ['acc', 'conf', 'gap', 'ece', 'nll']})
                    if tag == 'raw':
                        cells[lv]['by_class'] = {cl: {k: float(np.average(per_pair(pr, y)[k][cm[cl]], weights=w[cm[cl]]))
                                                      for k in ['acc', 'conf']} for cl in CLASSES}
                        cells[lv]['top_type_share'] = float(np.average(pr.argmax(1) == np.bincount(y).argmax(), weights=w))
                o[tag] = cells
            res['m1'][f'{key}/{c}'] = o

    print('\n### M1 emerging-drug KG removal: acc / conf / gap / ECE per level (raw)')
    for name, o in res['m1'].items():
        print(f'{name:16s} ' + '  '.join(f'{lv}: {o["raw"][lv]["acc"]:.3f}/{o["raw"][lv]["conf"]:.3f}/'
                                         f'{o["raw"][lv]["gap"]:+.3f}/{o["raw"][lv]["ece"]:.3f}' for lv in levels))
    print('\n### M1 change against the full KG (acc, conf, gap); temp = temperature fitted on the full KG')
    for name, o in res['m1'].items():
        for tag in ['raw', 'temp']:
            print(f'{name:16s} {tag:4s} ' + '  '.join(
                f'{lv}: acc {fmt(o[tag][lv]["minus_full"]["acc"])} conf {fmt(o[tag][lv]["minus_full"]["conf"])} '
                f'gap {fmt(o[tag][lv]["minus_full"]["gap"])}' for lv in levels[1:]))
    print('\n### M1 by mechanism class, acc/conf at each level')
    for name, o in res['m1'].items():
        print(f'{name:16s} ' + '  '.join(f'{cl[:5]} ' + ' '.join(f'{o["raw"][lv]["by_class"][cl]["acc"]:.2f}/'
                                                                 f'{o["raw"][lv]["by_class"][cl]["conf"]:.2f}'
                                                                 for lv in levels) for cl in CLASSES))
    print('\n### M1 share predicted as the most frequent type (48)')
    for name, o in res['m1'].items():
        print(f'{name:16s} ' + '  '.join(f'{lv} {o["raw"][lv]["top_type_share"]:.3f}' for lv in levels))

    counts = collections.defaultdict(collections.Counter)
    for h, t, r in train:
        counts[h][r] += 1
        counts[t][r] += 1
    partner = np.where(pairs[:, 0] == new_drug, pairs[:, 1], pairs[:, 0])
    prior = np.array([counts[d].most_common(1)[0][0] if counts[d] else np.bincount(y).argmax() for d in partner])
    res['m1_reference'] = {'existing_drug_top_type': cell((prior == y).astype(float), every),
                           'most_frequent_type': cell((y == np.bincount(y).argmax()).astype(float), every)}
    print('\n### M1 references without any KG: ' + '  '.join(f'{k} acc {fmt(v)}' for k, v in res['m1_reference'].items()))

    res['m1_key_vs_key'] = {}
    acc = {(key, lv): per_pair(ref[key]['original'] if lv == 'full' else probs[key, lv]['original'], y)['acc']
           for key in KEYS for lv in levels}
    print('### M1 key vs key, gate on: accuracy drop (level - full) of the first model minus that of the second')
    for lv in levels[1:]:
        for a, b in [('llm', 'id'), ('llm', 'random'), ('name', 'id'), ('name', 'random'), ('llm', 'name')]:
            if a not in KEYS or b not in KEYS:
                continue
            s = cell((acc[a, lv] - acc[a, 'full']) - (acc[b, lv] - acc[b, 'full']), every)
            res['m1_key_vs_key'][f'{lv}/{a}-{b}'] = s
            print(f'  {lv:6s} {a}-{b:7s} {fmt(s)}')

json.dump(res, open(cli.out, 'w'), indent=1)
print('saved', cli.out)
