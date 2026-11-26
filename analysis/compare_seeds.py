"""Side-by-side table of the report's key numbers for several training seeds (CPU; reads results/<tag>/*.json).
Run from DrugBank/:   python ../analysis/compare_seeds.py --train_seeds 0 1 [--keys llm name]
"""
import argparse
import json
import os

p = argparse.ArgumentParser()
p.add_argument('--dataset', default='S1_1')
p.add_argument('--train_seeds', type=int, nargs='+', default=[0, 1])
p.add_argument('--keys', nargs='+', default=['llm', 'name'])
cli = p.parse_args()
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')


def load(seed, name):
    tag = cli.dataset if seed == 0 else f'{cli.dataset}_seed{seed}'
    path = os.path.join(ROOT, tag, name)
    return json.load(open(path)) if os.path.exists(path) else None


def get(d, *path):
    for k in path:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def fmt(x, null=0.):
    if x is None:
        return '      -'
    if isinstance(x, dict):
        star = '*' if x['ci'][0] > null or x['ci'][1] < null else ' '
        return f"{x['mean']:+.3f} [{x['ci'][0]:+.3f},{x['ci'][1]:+.3f}]{star}"
    return f'{x:7.3f}'


def row(label, values, null=0.):    # '*': the 95% interval excludes `null`
    print(f'{label:58s}' + '  '.join(f'{fmt(v, null):>26s}' for v in values))


S = cli.train_seeds
print(f'{"":58s}' + '  '.join(f'{"seed " + str(s):>26s}' for s in S))
kg = {s: load(s, 'kg_ablation.json') for s in S}
st = {s: load(s, 'stratify.json') for s in S}
fu = {s: load(s, 'gate_interventions/followup.json') for s in S}
gd = {s: load(s, 'gate_degree.json') for s in S}

print('# accuracy on the sample after KG deletions (original condition; baseline = ungated)')
for m in [f'{k}/original' for k in cli.keys] + ['baseline/original']:
    for v in ['full', 'cyp', 'hub', 'rand0', 'rand1', 'new50', 'new90', 'new100']:
        row(f'acc {m} {v}', [get(kg[s], 'accuracy', m, v) for s in S])
print('# F4: change in log p(true) after CYP removal, by class')
for m in [f'{k}/original' for k in cli.keys] + [f'{k}/off' for k in cli.keys] + ['baseline/original']:
    for c in ['metabolism', 'pk_other', 'pd']:
        row(f'dlogp_cyp {m} {c}', [get(kg[s], 'f4', m, 'dlogp_cyp', c) for s in S])
    for v in ['hub', 'rand0']:
        row(f'dlogp_{v} {m} metabolism', [get(kg[s], 'f4', m, f'dlogp_{v}', 'metabolism') for s in S])
print('# F4: metabolism predictions among shared-CYP pairs (rate, precision), full -> cyp')
for m in [f'{k}/original' for k in cli.keys] + ['baseline/original']:
    for v in ['full', 'cyp']:
        for f in ['rate', 'precision']:
            row(f'shared_cyp_metab_pred {m} {v} {f}', [get(kg[s], 'f4', m, 'shared_cyp_metab_pred', v, f) for s in S])
print('# stratify: PD pairs sharing a CYP (off condition, mean over the keys stratify.py was run with)')
for f in ['off_acc', 'off_conf']:
    row(f'pd & cyp {f}', [get(st[s], 'pd & cyp', f) for s in S])
print('# gate interventions: key permutation, change in log p(true)')
for k in cli.keys:
    for g in ['metabolism', 'non_pk']:
        for f in ['key_perm-off', 'key_perm-original_with_shared_cyp', 'key_perm-original_without_shared_cyp', 'kl_type-all']:
            row(f'{k} {g} {f}', [get(fu[s], k, g, f) for s in S])
print('# shared-CYP ranking AUC gap (metabolism - non-PK)')
for k in cli.keys:
    for w in ['best', 'last']:
        for g in ['metabolism', 'non_pk', 'gap']:
            row(f'{k}-v2 [{w}] e2 {g} auc', [get(gd[s], f'{k}-v2 [{w}]', 'e2', g, 'auc') for s in S], 0. if g == 'gap' else .5)
