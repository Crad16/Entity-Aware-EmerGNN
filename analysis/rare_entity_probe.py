"""Do gates learned on frequent CYPs carry over to rarely seen ones? (DrugBank)

Genes are split by training-KG degree (hub: more than 20 binding drugs, rare: 1-20). Per stratum,
delta = mean gate(CYPs) - mean gate(non-DME genes of the same stratum), compared between
metabolism and non-PK test pairs. Only finished runs (with a *_last.pt checkpoint) are used.

    cd DrugBank && python ../analysis/rare_entity_probe.py
"""
import argparse
import collections
import csv
import glob
import json
import os
import re
import sys

import numpy as np
import torch

sys.path.insert(0, os.getcwd())
from load_data import DataLoader  # noqa: E402
from models import EmerGNN  # noqa: E402

METAB, PK_OTHER = {46, 3}, {72, 74, 76, 10}
N_DRUG, RARE_MAX = 1710, 20
DME = (r'^(CYP\d|UGT\d|SULT\d|FMO\d|NAT[12]$|GST[AMPTOZ]\d|EPHX\d|CES\d|ADH\d|ALDH\d|AOX1|XDH|MAO[AB]$|TPMT$|COMT$'
       r'|ABC[BCG]\d|SLCO\d|SLC22A\d|SLC47A\d)')

p = argparse.ArgumentParser()
p.add_argument('--dataset', default='S1_1')
p.add_argument('--out', default='../results/rare_entity_probe.json')
cli = p.parse_args()

# ------------------------------------------------------------------ entity strata from the training KG
rel = json.load(open('data/relation2id.json'))
cbg = [int(k) for k, v in rel.items() if v == 'CbG'][0]
ents = {int(r['ent_id']): r for r in csv.DictReader(open('../entity_text/drugbank_entities.tsv'), delimiter='\t')}
degree = collections.Counter()
for line in open(f'data/{cli.dataset}/train_KG.txt'):
    h, t, r = map(int, line.split())
    if r == cbg:
        d, g = (h, t) if h < N_DRUG else (t, h)
        if d < N_DRUG and ents[g]['kind'] == 'Gene':
            degree[g] += 1
name = lambda g: ents[g]['name']  # noqa: E731
groups = {}
for stratum, ok in [('hub', lambda d: d > RARE_MAX), ('rare', lambda d: 1 <= d <= RARE_MAX)]:
    genes = [g for g, d in degree.items() if ok(d)]
    groups[f'cyp_{stratum}'] = sorted(g for g in genes if re.match(r'CYP\d', name(g)))
    groups[f'other_{stratum}'] = sorted(g for g in genes if not re.match(DME, name(g)))
for k, v in groups.items():
    print(f'{k:10s}: {len(v):5d} genes' + (f'  e.g. {[name(g) for g in v[:6]]}' if k.startswith('cyp') else ''))

test = np.array([tuple(map(int, l.split())) for l in open(f'data/{cli.dataset}/test_ddi.txt')])
is_metab = np.isin(test[:, 2], list(METAB))
is_nonpk = ~is_metab & ~np.isin(test[:, 2], list(PK_OTHER))
keep = is_metab | is_nonpk
test, is_metab = test[keep], is_metab[keep]

# ------------------------------------------------------------------ probe each finished run
args = argparse.Namespace(task_dir='./', dataset=cli.dataset, length=3, feat='M', attn_mode='learned', n_dim=64,
                          gate_dim=32, emb_prefix='../entity_text/emb/drugbank', lamb=0, lr=1e-3)
data = DataLoader(args)
args.all_ent, args.all_rel, args.eval_rel = data.all_ent, data.all_rel, data.eval_rel
heads, tails = (torch.LongTensor(test[:, i]).cuda() for i in range(2))
idx = {k: torch.LongTensor(v).cuda() for k, v in groups.items()}


def cohen_d(a, b):
    return float((a.mean() - b.mean()) / np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2))


results = {}
for path in sorted(glob.glob(f'saved_models/{cli.dataset}_learned_shuf_gate-*_seed0_*.pt')):
    if not os.path.exists(re.sub(r'_(best|last)\.pt$', '_last.pt', path)):
        continue                                            # run still training
    m = re.search(r'gate-(\w+?)(-v2)?_seed0_(best|last)\.pt', path)
    feat, version, ckpt = m.group(1), 'v2' if m.group(2) else 'v1', m.group(3)
    args.entity_feat, args.gate_version = feat, version
    model = EmerGNN(data.eval_ent, data.eval_rel, args).cuda().eval()
    model.load_state_dict(torch.load(path, map_location='cuda'))
    gate = model.entity_gate
    means = collections.defaultdict(list)
    with torch.no_grad():
        keys, center = gate.keys(), model.pair_center()
        for s in range(0, len(test), 1024):
            h, t = heads[s:s + 1024], tails[s:s + 1024]
            ht = torch.cat([model.Went(model.ent_kg[h]), model.Went(model.ent_kg[t])], dim=-1)
            g = torch.stack([x for x in gate(ht, keys, center) if x is not None]).mean(0)   # (n_ent, batch)
            for k, ix in idx.items():
                means[k].append(g[ix].mean(0).cpu())
    means = {k: torch.cat(v).numpy() for k, v in means.items()}
    res = {}
    for stratum in ['hub', 'rare']:
        delta = means[f'cyp_{stratum}'] - means[f'other_{stratum}']
        a, b = delta[is_metab], delta[~is_metab]
        res[stratum] = {'diff': float(a.mean() - b.mean()), 'cohen_d': cohen_d(a, b)}
    res['corr_rare_cyp_vs_hub_cyp'] = float(np.corrcoef(means['cyp_rare'], means['cyp_hub'])[0, 1])
    res['corr_rare_other_vs_hub_cyp'] = float(np.corrcoef(means['other_rare'], means['cyp_hub'])[0, 1])
    results[f'{feat}-{version} [{ckpt}]'] = res

print(f'\n{"model":18s} | {"hub CYP: diff":>13s} {"d":>6s} | {"rare CYP: diff":>14s} {"d":>6s} | '
      f'{"corr(rareCYP,hubCYP)":>20s} {"corr(rareOther,hubCYP)":>22s}')
for k, r in results.items():
    print(f'{k:18s} | {r["hub"]["diff"]:+13.4f} {r["hub"]["cohen_d"]:+6.2f} | {r["rare"]["diff"]:+14.4f} '
          f'{r["rare"]["cohen_d"]:+6.2f} | {r["corr_rare_cyp_vs_hub_cyp"]:20.3f} {r["corr_rare_other_vs_hub_cyp"]:22.3f}')
json.dump({'groups': {k: [name(g) for g in v] if k.startswith('cyp') else len(v) for k, v in groups.items()},
           'results': results}, open(cli.out, 'w'), indent=1)
print('saved', cli.out)
