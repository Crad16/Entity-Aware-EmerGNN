"""Entities whose names do not reveal their function (DrugBank).

A: non-CYP drug-metabolizing enzymes compared with CYPs.
B: clinically relevant drug transporters (ITC list) from different name families compared with
   same-prefix transporters of endogenous substrates.
Compares cosine similarity of the input vectors and correlation of trained v2 gate profiles.

    cd DrugBank && python ../analysis/name_vs_function.py
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
N_DRUG, HUB_MIN = 1710, 21
DME = r'^(UGT\d+[AB]\d+|SULT\d[AB]?\d*|FMO\d|NAT[12]|CES[12]|AOX1|TPMT|EPHX1)$'
DRUG_TR = ['ABCB1', 'ABCG2', 'ABCC2', 'ABCB11', 'SLCO1B1', 'SLCO1B3',
           'SLC22A1', 'SLC22A2', 'SLC22A6', 'SLC22A8', 'SLC47A1', 'SLC47A2']
LOOKALIKE = ['ABCA1', 'ABCA4', 'ABCD1', 'ABCB7', 'SLC2A1', 'SLC2A4',
             'SLC4A1', 'SLC26A4', 'SLC34A1', 'SLC39A4', 'SLC40A1', 'SLC1A2']
family = lambda n: 'ABC' if n.startswith('ABC') else 'SLC'  # noqa: E731

p = argparse.ArgumentParser()
p.add_argument('--dataset', default='S1_1')
p.add_argument('--pairs', type=int, default=4000, help='test pairs sampled for gate profiles')
p.add_argument('--out', default='../results/name_vs_function.json')
cli = p.parse_args()

genes = {r['name']: int(r['ent_id']) for r in csv.DictReader(open('../entity_text/drugbank_entities.tsv'), delimiter='\t')
         if r['kind'] == 'Gene'}
cyp = [i for n, i in genes.items() if re.match(r'CYP\d', n)]
dme = [i for n, i in genes.items() if re.match(DME, n)]
tr = [genes[n] for n in DRUG_TR]
look = [genes[n] for n in LOOKALIKE]
special = set(cyp) | set(dme) | set(tr) | set(look)
rng = np.random.default_rng(0)
background = sorted(rng.choice([i for i in genes.values() if i not in special], 2000, replace=False))

rel = json.load(open('data/relation2id.json'))
cbg = [int(k) for k, v in rel.items() if v == 'CbG'][0]
degree = collections.Counter()
for line in open(f'data/{cli.dataset}/train_KG.txt'):
    h, t, r = map(int, line.split())
    if r == cbg:
        degree[t if h < N_DRUG else h] += 1
hub_cyp = [i for i in cyp if degree[i] >= HUB_MIN]

# pairs within a group across name families vs pairs across groups within a name family
cross_name = [(a, b) for i, a in enumerate(DRUG_TR) for b in DRUG_TR[i + 1:] if family(a) != family(b)]
same_name = [(a, b) for a in DRUG_TR for b in LOOKALIKE if family(a) == family(b)]


def structure(sim):
    """sim(i, j) for entity ids -> the two test statistics."""
    a = np.mean([sim(d, c) for d in dme for c in cyp]) - np.mean([sim(g, c) for g in background[:500] for c in cyp])
    b_func = np.mean([sim(genes[x], genes[y]) for x, y in cross_name])
    b_name = np.mean([sim(genes[x], genes[y]) for x, y in same_name])
    return {'A_dme_to_cyp_minus_background': float(a), 'B_function_cross_name': float(b_func),
            'B_name_cross_function': float(b_name), 'B_function_minus_name': float(b_func - b_name)}


results = {'vectors': {}, 'gates': {}}
for feat in ['random', 'name', 'llm']:
    v = np.load(f'../entity_text/emb/drugbank_{feat}.npy')
    results['vectors'][feat] = structure(lambda i, j: float(v[i] @ v[j]))

# ------------------------------------------------------------------ trained v2 gates
args = argparse.Namespace(task_dir='./', dataset=cli.dataset, length=3, feat='M', attn_mode='learned', n_dim=64,
                          gate_dim=32, emb_prefix='../entity_text/emb/drugbank', lamb=0, lr=1e-3)
data = DataLoader(args)
args.all_ent, args.all_rel, args.eval_rel = data.all_ent, data.all_rel, data.eval_rel
test = np.array([tuple(map(int, l.split())) for l in open(f'data/{cli.dataset}/test_ddi.txt')])
keep = np.isin(test[:, 2], list(METAB)) | ~np.isin(test[:, 2], list(METAB | PK_OTHER))
test = test[keep][rng.choice(keep.sum(), cli.pairs, replace=False)]
is_metab = np.isin(test[:, 2], list(METAB))
heads, tails = (torch.LongTensor(test[:, i]).cuda() for i in range(2))
probe = sorted(special | set(background))
col = {e: k for k, e in enumerate(probe)}
probe_t = torch.LongTensor(probe).cuda()

for path in sorted(glob.glob(f'saved_models/{cli.dataset}_learned_shuf_gate-*-v2_seed0_*.pt')):
    feat, ckpt = re.search(r'gate-(\w+)-v2_seed0_(best|last)\.pt', path).groups()
    args.entity_feat, args.gate_version = feat, 'v2'
    model = EmerGNN(data.eval_ent, data.eval_rel, args).cuda().eval()
    model.load_state_dict(torch.load(path, map_location='cuda'))
    gate = model.entity_gate
    with torch.no_grad():
        keys, center = gate.keys(), model.pair_center()
        ht = torch.cat([model.Went(model.ent_kg[heads]), model.Went(model.ent_kg[tails])], dim=-1)
        prof = torch.stack([x[probe_t] for x in gate(ht, keys, center) if x is not None]).mean(0).cpu().numpy()
    prof = prof - prof.mean(1, keepdims=True)                       # per-entity gate profile over pairs
    prof = prof / (np.linalg.norm(prof, axis=1, keepdims=True) + 1e-12)
    res = structure(lambda i, j: float(prof[col[i]] @ prof[col[j]]))
    # does the model treat non-CYP DMEs like hub CYPs, specifically for metabolism-type pairs?
    g = torch.stack([x[probe_t] for x in gate(ht, keys, center) if x is not None]).mean(0).detach().cpu().numpy()
    mean_of = lambda ids: g[[col[i] for i in ids]].mean(0)          # noqa: E731
    delta = mean_of(dme) - mean_of(background)
    a, b = delta[is_metab], delta[~is_metab]
    res['A_dme_metab_contrast_d'] = float((a.mean() - b.mean()) / np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2))
    res['A_corr_dme_hubcyp'] = float(np.corrcoef(mean_of(dme), mean_of(hub_cyp))[0, 1])
    results['gates'][f'{feat}-v2 [{ckpt}]'] = res

print(f'groups: {len(cyp)} CYPs ({len(hub_cyp)} hubs), {len(dme)} non-CYP DMEs, {len(tr)} drug transporters, '
      f'{len(look)} lookalikes, {len(background)} background genes; {cli.pairs} test pairs\n')
print('Part 1, input vectors (cosine similarity)')
print(f'{"vectors":10s} | {"A: DME~CYP vs bg":>16s} | {"B: same function":>16s} {"same name":>10s} {"function-name":>13s}')
for k, r in results['vectors'].items():
    print(f'{k:10s} | {r["A_dme_to_cyp_minus_background"]:+16.3f} | {r["B_function_cross_name"]:16.3f} '
          f'{r["B_name_cross_function"]:10.3f} {r["B_function_minus_name"]:+13.3f}')
print('\nPart 2, trained v2 gates (correlation of gate profiles over test pairs)')
print(f'{"model":18s} | {"A: DME metab d":>14s} {"corr(DME,hubCYP)":>16s} | {"B: same function":>16s} '
      f'{"same name":>10s} {"function-name":>13s}')
for k, r in results['gates'].items():
    print(f'{k:18s} | {r["A_dme_metab_contrast_d"]:+14.2f} {r["A_corr_dme_hubcyp"]:16.3f} | '
          f'{r["B_function_cross_name"]:16.3f} {r["B_name_cross_function"]:10.3f} {r["B_function_minus_name"]:+13.3f}')
json.dump(results, open(cli.out, 'w'), indent=1)
print('saved', cli.out)
