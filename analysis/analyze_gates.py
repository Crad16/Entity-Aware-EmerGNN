"""CYP450 gate contrast on DrugBank test pairs.

For each pair, delta = mean gate over CYP450 genes - mean gate over the other genes. Compares
metabolism pairs (types 46, 3) with all other pairs; also reports the serum-concentration types
and KCNH2 on QT-prolonging pairs (type 19).

    cd DrugBank && python ../analysis/analyze_gates.py --gate_version v2 --ckpt best
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
import torch
from scipy.stats import mannwhitneyu

sys.path.insert(0, os.getcwd())
from load_data import DataLoader  # noqa: E402
from models import EmerGNN  # noqa: E402

METAB, SERUM, QTC = [46, 3], [72, 74, 76, 10], [19]

p = argparse.ArgumentParser()
p.add_argument('--dataset', default='S1_1')
p.add_argument('--ckpt', default='best', choices=['best', 'last'])
p.add_argument('--arms', default='id,random,name,llm')
p.add_argument('--gate_version', default='v1', choices=['v1', 'v2'])
p.add_argument('--out', default='../results/h2_gates_%s%s.json')
cli = p.parse_args()

args = argparse.Namespace(task_dir='./', dataset=cli.dataset, length=3, feat='M', attn_mode='learned', n_dim=64,
                          gate_dim=32, emb_prefix='../entity_text/emb/drugbank', lamb=0, lr=1e-3)
data = DataLoader(args)
args.all_ent, args.all_rel, args.eval_rel = data.all_ent, data.all_rel, data.eval_rel
test = np.asarray(data.triplets['test'])
heads, tails, labels = (torch.LongTensor(test[:, i]).cuda() for i in range(3))
labels = test[:, 2]
kcnh2 = [int(r['ent_id']) for r in csv.DictReader(open('../entity_text/drugbank_entities.tsv'), delimiter='\t')
         if r['name'] == 'KCNH2' and r['kind'] == 'Gene'][0]


@torch.no_grad()
def pair_gates(model, batch=512):
    """Per test pair: mean gate over CYP genes, other genes, KCNH2 (each averaged over layers)."""
    gate = model.entity_gate
    keys, center = gate.keys(), model.pair_center()
    out = {'cyp': [], 'gene': [], 'kcnh2': []}
    for i in range(0, len(heads), batch):
        h, t = heads[i:i + batch], tails[i:i + batch]
        ht = torch.cat([model.Went(model.ent_kg[h]), model.Went(model.ent_kg[t])], dim=-1)
        g = torch.stack([x for x in gate(ht, keys, center) if x is not None]).mean(0)  # (n_ent, batch), gated layers
        out['cyp'].append(g[gate.cyp_idx].mean(0).cpu())
        out['gene'].append(g[gate.gene_idx].mean(0).cpu())
        out['kcnh2'].append(g[kcnh2].cpu())
    return {k: torch.cat(v).numpy() for k, v in out.items()}


def contrast(delta, in_group, rest):
    a, b = delta[in_group], delta[rest]
    pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return {'n_group': int(in_group.sum()), 'n_rest': int(rest.sum()),
            'mean_group': float(a.mean()), 'mean_rest': float(b.mean()), 'diff': float(a.mean() - b.mean()),
            'cohen_d': float((a.mean() - b.mean()) / pooled), 'mannwhitney_p': float(mannwhitneyu(a, b).pvalue)}


results = {}
is_metab, is_serum, is_qtc = (np.isin(labels, g) for g in (METAB, SERUM, QTC))
for arm in cli.arms.split(','):
    args.entity_feat, args.gate_version = arm, cli.gate_version
    model = EmerGNN(data.eval_ent, data.eval_rel, args).cuda().eval()
    tag = arm if cli.gate_version == 'v1' else f'{arm}-{cli.gate_version}'
    path = f'saved_models/{cli.dataset}_learned_shuf_gate-{tag}_seed0_{cli.ckpt}.pt'
    model.load_state_dict(torch.load(path, map_location='cuda'))
    g = pair_gates(model)
    d_cyp, d_kcnh2 = g['cyp'] - g['gene'], g['kcnh2'] - g['gene']
    # rank of the metabolism types among all types (types with >= 30 test pairs), by mean delta_cyp
    types = [r for r in np.unique(labels) if (labels == r).sum() >= 30]
    order = sorted(types, key=lambda r: -d_cyp[labels == r].mean())
    results[arm] = {
        'static': {'cyp': float(g['cyp'].mean()), 'other_gene': float(g['gene'].mean()), 'delta_cyp': float(d_cyp.mean())},
        'H2_metab_vs_rest': contrast(d_cyp, is_metab, ~is_metab),
        'serum_vs_rest': contrast(d_cyp, is_serum, ~(is_serum | is_metab)),
        'qtc_kcnh2_vs_rest': contrast(d_kcnh2, is_qtc, ~is_qtc),
        'qtc_cyp_vs_rest': contrast(d_cyp, is_qtc, ~is_qtc),
        'metab_type_ranks': {int(r): order.index(r) + 1 for r in METAB if r in order},
        'n_types_ranked': len(order),
    }
    h = results[arm]['H2_metab_vs_rest']
    print(f'{arm:6s} [{cli.ckpt}] static CYP-other {d_cyp.mean():+.3f} | H2: delta_cyp metab {h["mean_group"]:+.3f} vs rest '
          f'{h["mean_rest"]:+.3f} -> diff {h["diff"]:+.3f}, d={h["cohen_d"]:+.2f}, p={h["mannwhitney_p"]:.1e} | '
          f'ranks of types 46,3: {results[arm]["metab_type_ranks"]} / {len(order)}')
    q = results[arm]['qtc_kcnh2_vs_rest']
    print(f'       QTc: delta_kcnh2 {q["mean_group"]:+.3f} vs rest {q["mean_rest"]:+.3f} -> diff {q["diff"]:+.3f}, '
          f'd={q["cohen_d"]:+.2f} | QTc delta_cyp diff {results[arm]["qtc_cyp_vs_rest"]["diff"]:+.3f}')

suffix = '' if cli.gate_version == 'v1' else '_' + cli.gate_version
json.dump(results, open(cli.out % (cli.ckpt, suffix), 'w'), indent=1)
print('saved', cli.out % (cli.ckpt, suffix))
