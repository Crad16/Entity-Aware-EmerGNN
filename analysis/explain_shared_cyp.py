"""Shared-CYP explanation score on DrugBank.

Candidates for a test pair are the genes bound by both drugs and the CYP450 genes among them are
the positives. Reports the within-pair AUC and hit@1 on metabolism pairs (types 46, 3) and on
non-pharmacokinetic pairs. EmerGNN weighs all candidates equally (AUC 0.5); a degree heuristic is
included for reference.

    cd DrugBank && python ../analysis/explain_shared_cyp.py
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
N_DRUG = 1710

p = argparse.ArgumentParser()
p.add_argument('--dataset', default='S1_1')
p.add_argument('--out', default='../results/shared_cyp_scores.json')
cli = p.parse_args()

# ------------------------------------------------------------------ candidates from the KG
rel = json.load(open('data/relation2id.json'))
cbg = [int(k) for k, v in rel.items() if v == 'CbG'][0]
ents = {int(r['ent_id']): r for r in csv.DictReader(open('../entity_text/drugbank_entities.tsv'), delimiter='\t')}
cyp = {e for e, r in ents.items() if r['kind'] == 'Gene' and re.match(r'CYP\d', r['name'])}
binds = collections.defaultdict(set)
for split in ['train', 'valid', 'test']:
    for line in open(f'data/{cli.dataset}/{split}_KG.txt'):
        h, t, r = map(int, line.split())
        if r == cbg:
            d, g = (h, t) if h < N_DRUG else (t, h)
            if d < N_DRUG and ents[g]['kind'] == 'Gene':
                binds[d].add(g)
degree = collections.Counter(g for genes in binds.values() for g in genes)

test = [tuple(map(int, l.split())) for l in open(f'data/{cli.dataset}/test_ddi.txt')]
items = []  # (row in test, label, candidates, positive mask)
for i, (u, v, r) in enumerate(test):
    if r in METAB:
        group = 'metabolism'
    elif r not in PK_OTHER:
        group = 'non_pk'
    else:
        continue
    cand = sorted(binds[u] & binds[v])
    pos = np.array([g in cyp for g in cand])
    if pos.any() and not pos.all():
        items.append((i, group, np.array(cand), pos))
stats = {g: [it for it in items if it[1] == g] for g in ['metabolism', 'non_pk']}
for g, its in stats.items():
    print(f'{g:10s}: {len(its)} scorable pairs, mean candidates {np.mean([len(c) for _, _, c, _ in its]):.1f}, '
          f'mean shared CYPs {np.mean([p.sum() for _, _, _, p in its]):.1f}')


def pair_scores(score, pos):
    s_pos, s_neg = score[pos], score[~pos]
    auc = ((s_pos[:, None] > s_neg[None, :]).mean() + 0.5 * (s_pos[:, None] == s_neg[None, :]).mean())
    top = score == score.max()
    return auc, pos[top].mean()


def evaluate(score_fn):
    out = {}
    for g, its in stats.items():
        res = np.array([pair_scores(score_fn(i, cand), pos) for i, _, cand, pos in its])
        out[g] = {'auc': float(res[:, 0].mean()), 'hit1': float(res[:, 1].mean()), 'n': len(its)}
    out['auc_gap'] = out['metabolism']['auc'] - out['non_pk']['auc']
    return out


results = {
    'EmerGNN (relation-level, ties)': evaluate(lambda i, cand: np.zeros(len(cand))),
    'degree heuristic': evaluate(lambda i, cand: np.array([degree[g] for g in cand], dtype=float)),
}

# ------------------------------------------------------------------ gate models
args = argparse.Namespace(task_dir='./', dataset=cli.dataset, length=3, feat='M', attn_mode='learned', n_dim=64,
                          gate_dim=32, emb_prefix='../entity_text/emb/drugbank', lamb=0, lr=1e-3)
data = DataLoader(args)
args.all_ent, args.all_rel, args.eval_rel = data.all_ent, data.all_rel, data.eval_rel
rows = np.array([i for i, *_ in items])
heads = torch.LongTensor([test[i][0] for i in rows]).cuda()
tails = torch.LongTensor([test[i][1] for i in rows]).cuda()

for path in sorted(glob.glob(f'saved_models/{cli.dataset}_learned_shuf_gate-*_seed0_*.pt')):
    m = re.search(r'gate-(\w+?)(-v2)?_seed0_(best|last)\.pt', path)
    feat, version, ckpt = m.group(1), 'v2' if m.group(2) else 'v1', m.group(3)
    args.entity_feat, args.gate_version = feat, version
    model = EmerGNN(data.eval_ent, data.eval_rel, args).cuda().eval()
    model.load_state_dict(torch.load(path, map_location='cuda'))
    gate = model.entity_gate
    scores = {}
    with torch.no_grad():
        keys, center = gate.keys(), model.pair_center()
        for s in range(0, len(rows), 512):
            h, t = heads[s:s + 512], tails[s:s + 512]
            ht = torch.cat([model.Went(model.ent_kg[h]), model.Went(model.ent_kg[t])], dim=-1)
            g = torch.stack([x for x in gate(ht, keys, center) if x is not None]).mean(0)  # (n_ent, batch)
            for j in range(len(h)):
                i = rows[s + j]
                cand = items[s + j][2]
                scores[i] = g[torch.as_tensor(cand, device='cuda'), j].cpu().numpy()
    name = f'{feat}-{version} [{ckpt}]'
    results[name] = evaluate(lambda i, cand: scores[i])

print(f'\n{"ranker":32s} | {"metabolism AUC":>14s} {"hit@1":>6s} | {"non-PK AUC":>10s} {"hit@1":>6s} | {"AUC gap":>7s}')
for name, r in results.items():
    print(f'{name:32s} | {r["metabolism"]["auc"]:14.3f} {r["metabolism"]["hit1"]:6.3f} | '
          f'{r["non_pk"]["auc"]:10.3f} {r["non_pk"]["hit1"]:6.3f} | {r["auc_gap"]:+7.3f}')
json.dump(results, open(cli.out, 'w'), indent=1)
print('saved', cli.out)
