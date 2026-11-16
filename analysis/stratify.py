"""Stratified view of the ablation results (S1). CPU only; run from DrugBank/ after
gate_interventions.py:

    python ../analysis/stratify.py

Averages over all test pairs let a few CYP hubs dominate. This re-reads the saved predictions of the
gate interventions (8,195 type-stratified pairs, weighted to the full test set) and splits them by
  mechanism   metabolism (types 46, 3) / other PK (types naming serum concentration, absorption,
              bioavailability, excretion or protein binding) / PD (types naming an effect) /
              unspecified (type 48, "risk or severity of adverse effects")
  mediator    the two drugs bind a common CYP / a common non-CYP gene only / no common gene
  KG amount   third of the emerging drug's KG degree (cut points as in kg_calibration.py)
plus mechanism x mediator. Per stratum:
  descriptives  share of test pairs, accuracy and confidence of the ungated forward pass ('off',
                mean of the four models), M2 tie size of a random 2-hop path
  F2            key_perm - original, change in log-prob of the true type
  F1            KL(original || same-type donor) - KL(original || random donor), < 0 = query carries type
  gate gain     NLL(off) - NLL(original), > 0 = the gate helps
  key vs key    NLL(other key) - NLL(llm), > 0 = llm better (different training runs, one seed each)
with 95% bootstrap intervals over emerging drugs. Type classes are written to
results/<tag>/ddi_type_classes.tsv for review.
"""
import argparse
import collections
import json

import numpy as np

from ablation_common import KEYS, N_DRUG, cyp_genes, drug_binds, emerging_drug, entities, fmt, kg_triples, \
    load_ddi, load_probs, mechanism_class, res_path, setup, stat

p = argparse.ArgumentParser()
p.add_argument('--reps', type=int, default=3)
p.add_argument('--out', default=None, help='default: results/<tag>/stratify.json')
cli = setup(p)
cli.out = cli.out or res_path('stratify.json')
REF = 'llm' if 'llm' in KEYS else KEYS[0]                  # reference model of the key-vs-key rows

train, test = load_ddi('train'), load_ddi('test')
mech = mechanism_class()
rel = json.load(open('data/relation2id.json'))
counts = collections.Counter(test[:, 2])
with open(res_path('ddi_type_classes.tsv'), 'w') as f:
    f.write('type\tclass\ttest_pairs\tdescription\n')
    for k in sorted(mech):
        f.write(f'{k}\t{mech[k]}\t{counts[k]}\t{rel[str(k)]}\n')

ents = entities()
cyp = cyp_genes(ents)
kg = kg_triples()
binds = drug_binds(kg, ents)
degree = np.bincount(kg[:, :2].ravel(), minlength=len(ents))
new_all = emerging_drug(test, train)
cuts = np.percentile(degree[new_all], [100 / 3, 200 / 3])

# M2 tie size of a random 2-hop path through a non-drug entity (labels as in load_data.double_triple)
n_rel = max(map(int, rel)) + 1
out_nb = collections.defaultdict(lambda: collections.defaultdict(list))
in_nb = collections.defaultdict(lambda: collections.defaultdict(list))
for h, t, r in kg:
    for a, b, l in ((h, t, r), (t, h, r + n_rel)):
        if a < N_DRUG <= b:
            out_nb[a][b].append(l)
        elif b < N_DRUG <= a:
            in_nb[b][a].append(l)


def tie_size(u, v):
    c = collections.Counter()
    for e in out_nb[u].keys() & in_nb[v].keys():
        for l1 in out_nb[u][e]:
            for l2 in in_nb[v][e]:
                c[l1, l2] += 1
    c = np.array(list(c.values()), float)
    return (c ** 2).sum() / c.sum() if len(c) else np.nan


rows, w, _ = load_probs(KEYS[0])
pairs = test[rows]
y, new_drug = pairs[:, 2], new_all[rows]
mech_s = np.array([mech[r] for r in y])
shared = [binds[u] & binds[v] for u, v, _ in pairs]
mediator = np.array(['cyp' if s & cyp else 'other_gene' if s else 'none' for s in shared])
kg_third = np.array(['low', 'mid', 'high'])[np.digitize(degree[new_drug], cuts)]
ties = np.array([tie_size(u, v) for u, v, _ in pairs])

STRATA = {'all': np.ones(len(rows), bool)}
for name, lab, vals in [('mech', mech_s, ['metabolism', 'pk_other', 'pd', 'unspecified']),
                        ('mediator', mediator, ['cyp', 'other_gene', 'none']),
                        ('kg', kg_third, ['low', 'mid', 'high'])]:
    STRATA.update({f'{name}={v}': lab == v for v in vals})
for m_ in ['metabolism', 'pd', 'unspecified']:
    for d_ in ['cyp', 'other_gene', 'none']:
        STRATA[f'{m_} & {d_}'] = (mech_s == m_) & (mediator == d_)

# per-pair quantities per model
q = {}
for key in KEYS:
    _, _, P = load_probs(key)
    idx = np.arange(len(rows))
    logp = {c: np.log(p[idx, y]) for c, p in P.items()}
    kl = {c: (P['original'] * (np.log(P['original']) - np.log(p))).sum(1) for c, p in P.items()}
    reps = range(cli.reps)
    q[key] = {
        'f2_key_perm': np.mean([logp[f'key_perm_{k}'] for k in reps], 0) - logp['original'],
        'f1_kl_type_minus_all': np.mean([kl[f'shuffle_type_{k}'] - kl[f'shuffle_all_{k}'] for k in reps], 0),
        'gate_gain': logp['original'] - logp['off'],                      # = NLL(off) - NLL(original)
        'nll': -logp['original'],
        'off_acc': (P['off'].argmax(1) == y).astype(float),
        'off_conf': P['off'].max(1),
    }

res = {}
for s, m in STRATA.items():
    if m.sum() < 30:
        continue
    ww, cl = w[m], new_drug[m]
    r = {'n_pairs': int(m.sum()), 'n_drugs': int(len(set(cl))), 'share_of_test': float(w[m].sum() / w.sum()),
         'off_acc': float(np.mean([np.average(q[k]['off_acc'][m], weights=ww) for k in KEYS])),
         'off_conf': float(np.mean([np.average(q[k]['off_conf'][m], weights=ww) for k in KEYS])),
         'no_2hop_path': float(np.average(np.isnan(ties[m]), weights=ww)),
         'tie_size_median': float(np.nanmedian(ties[m])) if (~np.isnan(ties[m])).any() else None}
    for k in KEYS:
        for name in ['f2_key_perm', 'f1_kl_type_minus_all', 'gate_gain']:
            r[f'{name}/{k}'] = stat(q[k][name][m], ww, cl)
        if k != REF:
            r[f'nll_{k}-{REF}'] = stat(q[k]['nll'][m] - q[REF]['nll'][m], ww, cl)
    res[s] = r

# ------------------------------------------------------------------ report
print(f'{"stratum":24s} {"pairs":>5s} {"drugs":>5s} {"share":>6s} {"acc":>6s} {"conf":>6s} {"no path":>7s} {"tie":>6s}')
for s, r in res.items():
    tie = f'{r["tie_size_median"]:6.1f}' if r['tie_size_median'] is not None else '     -'
    print(f'{s:24s} {r["n_pairs"]:5d} {r["n_drugs"]:5d} {r["share_of_test"]:6.3f} {r["off_acc"]:6.3f} '
          f'{r["off_conf"]:6.3f} {r["no_2hop_path"]:7.3f} {tie}')
for title, name, keys in [('F2 key_perm - original (log-prob of true type)', 'f2_key_perm', KEYS),
                          ('F1 KL same-type donor - random donor (< 0: query carries type)', 'f1_kl_type_minus_all', KEYS),
                          ('gate gain NLL(off) - NLL(original) (> 0: gate helps)', 'gate_gain', KEYS),
                          (f'NLL(key) - NLL({REF}) (> 0: {REF} better)', 'nll', [k for k in KEYS if k != REF])]:
    print(f'\n{title}')
    print(f'{"stratum":24s} ' + ' '.join(f'{k:>22s}' for k in keys))
    for s, r in res.items():
        cells = [r[f'nll_{k}-{REF}'] if name == 'nll' else r[f'{name}/{k}'] for k in keys]
        star = lambda c: '*' if c['ci'][0] > 0 or c['ci'][1] < 0 else ' '  # noqa: E731
        print(f'{s:24s} ' + ' '.join(f'{fmt(c):>21s}{star(c)}' for c in cells))
json.dump(res, open(cli.out, 'w'), indent=1)
print('saved', cli.out, 'and', res_path('ddi_type_classes.tsv'))
