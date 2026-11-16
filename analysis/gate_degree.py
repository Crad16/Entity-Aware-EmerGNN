"""Gate vs entity degree (ablation E4) and the shared-gene explanation score with intervals (E2).
CPU only, no GNN pass (the gate depends only on the drug-pair embedding and the entity key).
Run from DrugBank/:

    python ../analysis/gate_degree.py

E4. Is a high gate just a hub? CYP3A4 and friends bind hundreds of drugs. Over all 32,322 test pairs
we average every entity's gate (mean of the two gated layers) and correlate it with the entity's
degree: its edges in the KG available at test time, and for genes also the number of DDI drugs
binding it. Three readings:
  static    mean gate over all pairs vs degree (non-drug entities; genes)
  metab     mean gate over metabolism-type pairs vs degree (genes)
  shift     mean gate on metabolism pairs minus on non-PK pairs, per gene: the pair-dependent part
            that the H2 analysis used. Regressed on log degree plus a CYP indicator, the CYP
            coefficient says whether CYPs shift more than genes of the same degree.
|Spearman| above 0.8 would mean the gate is mostly degree.

E2. As explain_shared_cyp.py (candidates = genes both drugs bind, positives = CYPs, pairs with at
least one CYP and one non-CYP candidate), now with 95% bootstrap intervals over emerging drugs and a
degree-adverse AUC: only the (CYP, non-CYP) comparisons in which the non-CYP gene binds at least
as many drugs as the CYP, where ranking by degree cannot pick the CYP. EmerGNN ties every
candidate (AUC 0.5; its hit@1 is the CYP share among candidates, i.e. a random pick).
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))    # entity_gate.py
from ablation_common import KEYS, N_DRUG, boot_multiplicity, cyp_genes, drug_binds, emerging_drug, entities, \
    kg_triples, load_ddi, old_group, EMB_PREFIX, ckpt_path, res_path, setup  # noqa: E402
from entity_gate import EntityGate  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument('--ckpts', nargs='+', default=['best', 'last'])
p.add_argument('--batch', type=int, default=1024)
p.add_argument('--n_boot', type=int, default=1000)
p.add_argument('--n_hub', type=int, default=10, help='CYPs bound by the most drugs, reported apart from the rest')
p.add_argument('--out', default=None, help='default: results/<tag>/gate_degree.json')
cli = setup(p)
cli.out = cli.out or res_path('gate_degree.json')
torch.set_grad_enabled(False)
N_HUB = cli.n_hub

train, test = load_ddi('train'), load_ddi('test')
group = old_group(test[:, 2])
new_drug = emerging_drug(test, train)
ents = entities()
n_ent = len(ents)
cyp = cyp_genes(ents)
kg = kg_triples()
kg_deg = np.bincount(kg[:, :2].ravel(), minlength=n_ent)
binds = drug_binds(kg, ents)
bind_deg = np.zeros(n_ent)
for genes in binds.values():
    bind_deg[list(genes)] += 1
mask = np.load(EMB_PREFIX + '_mask.npy').astype(bool)       # entities that have a gate
is_gene = np.array([ents[e]['kind'] == 'Gene' for e in range(n_ent)])
is_cyp = np.zeros(n_ent, bool)
is_cyp[list(cyp)] = True
genes = np.where(mask & is_gene)[0]
nondrug = np.where(mask)[0]

# E2 candidates
items = []                                                     # (test row, candidates, positive mask)
for i, (u, v, r) in enumerate(test):
    if group[i] == 'pk_other':
        continue
    cand = np.array(sorted(binds[u] & binds[v]))
    pos = is_cyp[cand] if len(cand) else np.zeros(0, bool)
    if pos.any() and not pos.all():
        items.append((i, cand, pos))
item_rows = np.array([i for i, _, _ in items])
item_group = group[item_rows]
M = boot_multiplicity(new_drug[item_rows], cli.n_boot)


def pair_scores(score, pos, deg):
    sp, sn = score[pos][:, None], score[~pos][None, :]
    auc = (sp > sn).mean() + 0.5 * (sp == sn).mean()
    top = score == score.max()
    adverse = deg[~pos][None, :] >= deg[pos][:, None]           # non-CYP binds at least as many drugs
    adv = ((sp > sn) + 0.5 * (sp == sn))[adverse].mean() if adverse.any() else np.nan
    return auc, pos[top].mean(), adv


def evaluate(score_of):
    per = np.array([pair_scores(score_of(j), pos, bind_deg[cand]) for j, (_, cand, pos) in enumerate(items)])
    out = {}
    draws = {}
    for g in ['metabolism', 'non_pk']:
        out[g], draws[g] = {}, {}
        for k, name in enumerate(['auc', 'hit1', 'adverse_auc']):
            m = (item_group == g) & ~np.isnan(per[:, k])
            vals = np.nan_to_num(per[:, k])
            W = M * m
            draws[g][name] = (W @ vals) / W.sum(1)
            out[g][name] = {'mean': float(vals[m].mean()), 'ci': np.percentile(draws[g][name], [2.5, 97.5]).tolist(),
                            'n': int(m.sum())}
    out['gap'] = {name: {'mean': out['metabolism'][name]['mean'] - out['non_pk'][name]['mean'],
                         'ci': np.percentile(draws['metabolism'][name] - draws['non_pk'][name], [2.5, 97.5]).tolist()}
                  for name in ['auc', 'hit1', 'adverse_auc']}
    return out


def corr(x, idx, deg):
    return float(spearmanr(x[idx], deg[idx]).correlation)


def cyp_coef(shift, idx, n_boot=1000):
    """OLS shift ~ 1 + log(1 + binding degree) + CYP over genes; CYP coefficient with a gene bootstrap."""
    X = np.stack([np.ones(len(idx)), np.log1p(bind_deg[idx]), is_cyp[idx].astype(float)], 1)
    y = shift[idx]
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    r2_deg = 1 - np.var(y - X[:, :2] @ np.linalg.lstsq(X[:, :2], y, rcond=None)[0]) / np.var(y)
    rng = np.random.RandomState(0)
    bs = []
    for _ in range(n_boot):
        s = rng.randint(0, len(idx), len(idx))
        bs.append(np.linalg.lstsq(X[s], y[s], rcond=None)[0][2])
    # residual shift (after log degree) of the CYPs that many drugs share vs the remaining CYPs
    resid = y - X[:, :2] @ np.linalg.lstsq(X[:, :2], y, rcond=None)[0]
    c = is_cyp[idx]
    hub = c & (bind_deg[idx] >= np.sort(bind_deg[idx][c])[-N_HUB])
    by_hub = {}
    for name, m in [('hub_cyps', hub), ('other_cyps', c & ~hub)]:
        v = resid[m]
        b = [v[rng.randint(0, len(v), len(v))].mean() for _ in range(n_boot)]
        by_hub[name] = {'mean': float(v.mean()), 'ci': np.percentile(b, [2.5, 97.5]).tolist(), 'n': int(m.sum())}
    return {'cyp_coef': float(beta[2]), 'cyp_coef_ci': np.percentile(bs, [2.5, 97.5]).tolist(),
            'logdeg_coef': float(beta[1]), 'r2_logdeg_only': float(r2_deg),
            'shift_sd_genes': float(np.std(y)), 'cyp_residual': by_hub}


results = {'EmerGNN (ties)': {'e2': evaluate(lambda j: np.zeros(len(items[j][1])))},
           'degree heuristic': {'e2': evaluate(lambda j: bind_deg[items[j][1]])}}
groups = ['all', 'metabolism', 'non_pk']
gmat = np.stack([np.ones(len(test)), group == 'metabolism', group == 'non_pk'], 1).astype(np.float32)
pos_in_items = {i: j for j, (i, _, _) in enumerate(items)}
for key in KEYS:
    for which in cli.ckpts:
        path = ckpt_path(key, which)
        sd = torch.load(path, map_location='cpu')
        gate = EntityGate(key, EMB_PREFIX, n_ent, 64, 3, 32, 'v2')
        gate.load_state_dict({k[len('entity_gate.'):]: v for k, v in sd.items() if k.startswith('entity_gate.')})
        drug = sd['ent_kg'] @ sd['Went.weight'].t() + sd['Went.bias']         # (n_drug, 64)
        m_ = drug.mean(0)
        center, keys = torch.cat([m_, m_]), gate.keys()
        sums = np.zeros((n_ent, 3))
        cand_scores = {}
        for s in range(0, len(test), cli.batch):
            b = slice(s, s + cli.batch)
            h, t = torch.as_tensor(test[b, 0]), torch.as_tensor(test[b, 1])
            g = torch.stack([x for x in gate(torch.cat([drug[h], drug[t]], -1), keys, center) if x is not None]).mean(0)
            sums += (g @ torch.as_tensor(gmat[b])).numpy()
            for j, i in enumerate(range(s, min(s + cli.batch, len(test)))):
                if i in pos_in_items:
                    cand_scores[pos_in_items[i]] = g[torch.as_tensor(items[pos_in_items[i]][1]), j].numpy()
        mean_gate = sums / gmat.sum(0)
        shift = mean_gate[:, 1] - mean_gate[:, 2]
        e4 = {'static_vs_kgdeg_nondrug': corr(mean_gate[:, 0], nondrug, kg_deg),
              'static_vs_kgdeg_genes': corr(mean_gate[:, 0], genes, kg_deg),
              'static_vs_binddeg_genes': corr(mean_gate[:, 0], genes, bind_deg),
              'metab_vs_binddeg_genes': corr(mean_gate[:, 1], genes, bind_deg),
              'shift_vs_binddeg_genes': corr(shift, genes, bind_deg),
              'shift_vs_kgdeg_genes': corr(shift, genes, kg_deg),
              'shift_regression': cyp_coef(shift, genes),
              'cyp_minus_gene_static': float(mean_gate[is_cyp & mask, 0].mean() - mean_gate[genes, 0].mean()),
              'cyp_minus_gene_shift': float(shift[is_cyp & mask].mean() - shift[genes].mean())}
        # within-pair: does the gate order the shared genes like their degree does?
        rho = [spearmanr(cand_scores[j], bind_deg[c]).correlation for j, (_, c, _) in enumerate(items)
               if len(c) >= 3 and np.ptp(cand_scores[j]) > 0 and np.ptp(bind_deg[c]) > 0]
        e4['within_pair_rho_mean'] = float(np.mean(rho))
        e4['within_pair_n'] = len(rho)
        results[f'{key}-v2 [{which}]'] = {'e4': e4, 'e2': evaluate(lambda j: cand_scores[j])}
        print(f'{key}-v2 [{which}] done', flush=True)

# ------------------------------------------------------------------ report
f = lambda s: f'{s["mean"]:.3f} [{s["ci"][0]:.3f},{s["ci"][1]:.3f}]'  # noqa: E731
fg = lambda s: f'{s["mean"]:+.3f} [{s["ci"][0]:+.3f},{s["ci"][1]:+.3f}]'  # noqa: E731
print(f'\nE4  Spearman with degree (|rho| > 0.8 = mostly degree)')
print(f'{"model":20s} {"static/kg":>9s} {"static/bind":>11s} {"metab/bind":>10s} {"shift/bind":>10s} '
      f'{"within-pair":>11s} | shift ~ logdeg + CYP: CYP coef [CI] (R2 of logdeg alone)')
for name, r in results.items():
    if 'e4' not in r:
        continue
    e, reg = r['e4'], r['e4']['shift_regression']
    print(f'{name:20s} {e["static_vs_kgdeg_genes"]:+9.3f} {e["static_vs_binddeg_genes"]:+11.3f} '
          f'{e["metab_vs_binddeg_genes"]:+10.3f} {e["shift_vs_binddeg_genes"]:+10.3f} {e["within_pair_rho_mean"]:+11.3f} | '
          f'{reg["cyp_coef"]:+.4f} [{reg["cyp_coef_ci"][0]:+.4f},{reg["cyp_coef_ci"][1]:+.4f}] '
          f'(R2 {reg["r2_logdeg_only"]:.3f}; shift sd {reg["shift_sd_genes"]:.4f})')
print(f'residual metabolism shift of CYPs after log degree: {N_HUB} most-shared CYPs vs the rest')
for name, r in results.items():
    if 'e4' in r:
        h = r['e4']['shift_regression']['cyp_residual']
        print(f'{name:20s} hub {fg(h["hub_cyps"])} (n={h["hub_cyps"]["n"]})  other {fg(h["other_cyps"])} '
              f'(n={h["other_cyps"]["n"]})')
n_m = results['EmerGNN (ties)']['e2']['metabolism']['auc']['n']
n_n = results['EmerGNN (ties)']['e2']['non_pk']['auc']['n']
print(f'\nE2  shared-gene explanation (metabolism {n_m} pairs, non-PK {n_n} pairs)')
for name, r in results.items():
    e = r['e2']
    print(f'{name:20s} AUC metab {f(e["metabolism"]["auc"])} nonPK {f(e["non_pk"]["auc"])} gap {fg(e["gap"]["auc"])} | '
          f'hit@1 metab {f(e["metabolism"]["hit1"])} nonPK {f(e["non_pk"]["hit1"])} gap {fg(e["gap"]["hit1"])} | '
          f'deg-adverse AUC metab {f(e["metabolism"]["adverse_auc"])} nonPK {f(e["non_pk"]["adverse_auc"])} '
          f'gap {fg(e["gap"]["adverse_auc"])}')
json.dump(results, open(cli.out, 'w'), indent=1)
print('saved', cli.out)
