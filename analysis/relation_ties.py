"""How many KG paths does EmerGNN tie? (ablation M2). Run from DrugBank/, no GPU needed:

    python ../analysis/relation_ties.py

EmerGNN weighs an edge by its relation only (alpha_r(u, v) per layer), and every message is the
source hidden state times a relation vector followed by entity-agnostic layers. So for a drug pair,
all paths with the same relation sequence get the same weight, and 2-hop paths
u -r1-> e -r2-> v also carry identical messages whenever u reaches e by r1 alone: the model cannot
tell which intermediate entity e a path runs through. A path-based explanation can at best name the relation sequence and
then has k tied candidates; the chance that it singles out a given entity is 1/k.

For every test pair this counts, over the KG available at test time, the paths whose intermediate
entities are not DDI drugs:
    2-hop  u -> e -> v          exact counts per relation sequence
    3-hop  u -> e1 -> e2 -> v   counts per relation sequence (sparse matrix products)
and reports the tie-group size of a random path (sum c^2 / sum c), the share of paths that share
their sequence with another path, and, for the mechanism-relevant sequence
u -binds-> gene <-binds- v, the number k of genes both drugs bind and the share of CYPs among them.
Edge directions follow load_data.double_triple: (h, t, r) carries h -> t as r and t -> h as r + n_rel.
"""
import argparse
import collections
import csv
import json
import re
import time

import numpy as np
import scipy.sparse as sp

from ablation_common import ENT_TSV, res_path, setup

METAB, PK_OTHER = {46, 3}, {72, 74, 76, 10}
N_DRUG = 1710

p = argparse.ArgumentParser()
p.add_argument('--dataset', default='S1_1')
p.add_argument('--out', default=None, help='default: results/<tag>/relation_ties.json')
p.add_argument('--no_3hop', action='store_true')
cli = setup(p)
cli.out = cli.out or res_path('relation_ties.json')

rel = json.load(open('data/relation2id.json'))
n_rel = max(map(int, rel)) + 1
name = {int(k): v for k, v in rel.items()}
label_name = lambda l: name[l] if l < n_rel else name[l - n_rel] + '^-1'  # noqa: E731
cbg = [int(k) for k, v in rel.items() if v == 'CbG'][0]
ents = {int(r['ent_id']): r for r in csv.DictReader(open(ENT_TSV), delimiter='\t')}
cyp = {e for e, r in ents.items() if r['kind'] == 'Gene' and re.match(r'CYP\d', r['name'])}

# ------------------------------------------------------------------ KG at test time (directed, labelled)
edges = []
for split in ['train', 'valid', 'test']:
    for line in open(f'data/{cli.dataset}/{split}_KG.txt'):
        h, t, r = map(int, line.split())
        edges.append((h, t, r))
        edges.append((t, h, r + n_rel))
edges = np.array(edges)
n_ent = edges[:, :2].max() + 1
print(f'{len(edges)} directed KG edges, {n_ent} entities')

out_nb = collections.defaultdict(lambda: collections.defaultdict(list))   # drug -> e -> labels (drug -> e)
in_nb = collections.defaultdict(lambda: collections.defaultdict(list))    # drug -> e -> labels (e -> drug)
for a, b, l in edges:
    if a < N_DRUG <= b:
        out_nb[a][b].append(l)
    elif b < N_DRUG <= a:
        in_nb[b][a].append(l)

test = np.array([l.split() for l in open(f'data/{cli.dataset}/test_ddi.txt')], dtype=int)
group = np.array(['metabolism' if r in METAB else 'pk_other' if r in PK_OTHER else 'non_pk' for r in test[:, 2]])


def tie_stats(counts):
    """counts: paths per relation sequence for one pair."""
    c = np.asarray(counts, dtype=float)
    if c.sum() == 0:
        return dict(paths=0, sequences=0, tie_size=np.nan, tied_share=np.nan, max_tie=0)
    return dict(paths=c.sum(), sequences=len(c), tie_size=(c ** 2).sum() / c.sum(),
                tied_share=c[c >= 2].sum() / c.sum(), max_tie=c.max())


# ------------------------------------------------------------------ 2-hop, exact
t0 = time.time()
two, bind, per_pair = [], [], []
seq_total = collections.Counter()
for u, v, _ in test:
    cnt = collections.Counter()
    for e in out_nb[u].keys() & in_nb[v].keys():
        for l1 in out_nb[u][e]:
            for l2 in in_nb[v][e]:
                cnt[(l1, l2)] += 1
    seq_total.update(cnt)
    per_pair.append(cnt)
    two.append(tie_stats(list(cnt.values())))
    shared = [e for e in out_nb[u].keys() & in_nb[v].keys() if cbg in out_nb[u][e] and cbg + n_rel in in_nb[v][e]]
    bind.append((len(shared), sum(e in cyp for e in shared)))
bind = np.array(bind)
print(f'2-hop done in {time.time() - t0:.0f}s')

# ------------------------------------------------------------------ 3-hop, counts per sequence
three = None
if not cli.no_3hop:
    t0 = time.time()
    nd = np.arange(N_DRUG, n_ent)
    pos = -np.ones(n_ent, dtype=int)
    pos[nd] = np.arange(len(nd))
    mats = collections.defaultdict(dict)
    for l in np.unique(edges[:, 2]):
        e = edges[edges[:, 2] == l]
        a_drug, b_drug = e[:, 0] < N_DRUG, e[:, 1] < N_DRUG
        for kind, m, shape, rows_, cols_ in [
                ('P', a_drug & ~b_drug, (N_DRUG, len(nd)), lambda x: x[:, 0], lambda x: pos[x[:, 1]]),
                ('Q', ~a_drug & ~b_drug, (len(nd), len(nd)), lambda x: pos[x[:, 0]], lambda x: pos[x[:, 1]]),
                ('R', ~a_drug & b_drug, (len(nd), N_DRUG), lambda x: pos[x[:, 0]], lambda x: x[:, 1])]:
            if m.any():
                x = e[m]
                mats[kind][l] = sp.csr_matrix((np.ones(len(x)), (rows_(x), cols_(x))), shape=shape)
    u_idx, v_idx = test[:, 0], test[:, 1]
    cols = []                                                    # per relation sequence: counts over test pairs
    for l1, P in mats['P'].items():
        for l2, Q in mats['Q'].items():
            T = P @ Q
            if T.nnz == 0:
                continue
            for l3, R in mats['R'].items():
                M = (T @ R).tocsr()
                if M.nnz == 0:
                    continue
                vals = np.asarray(M[u_idx, v_idx]).ravel()
                if vals.any():
                    cols.append(((l1, l2, l3), vals))
    C = np.stack([c for _, c in cols], axis=1) if cols else np.zeros((len(test), 0))
    three = [tie_stats(row[row > 0]) for row in C]
    seq3_total = {'→'.join(map(label_name, s)): float(c.sum()) for s, c in cols}
    print(f'3-hop done in {time.time() - t0:.0f}s ({len(cols)} relation sequences reach test pairs)')


# ------------------------------------------------------------------ summary
def summarize(stats, mask):
    s = [x for x, m in zip(stats, mask) if m and x['paths'] > 0]
    q = lambda k: [float(np.percentile([x[k] for x in s], z)) for z in (25, 50, 75)]  # noqa: E731
    return {'pairs': int(mask.sum()), 'pairs_with_paths': len(s),
            'paths_q': q('paths'), 'sequences_q': q('sequences'), 'tie_size_q': q('tie_size'),
            'tied_share_mean': float(np.mean([x['tied_share'] for x in s])),
            'single_out_chance_mean': float(np.mean([1 / x['tie_size'] for x in s]))}


res = {'2hop': {}, '3hop': {}, 'binds_binds': {}}
for g in ['metabolism', 'non_pk', 'all']:
    m = np.ones(len(test), bool) if g == 'all' else group == g
    res['2hop'][g] = summarize(two, m)
    if three is not None:
        res['3hop'][g] = summarize(three, m)
    k, ncyp = bind[m, 0], bind[m, 1]
    has, has_cyp = k > 0, ncyp > 0
    res['binds_binds'][g] = {
        'pairs_sharing_a_bound_gene': float(has.mean()),
        'k_q': [float(np.percentile(k[has], z)) for z in (25, 50, 75)],
        'single_out_chance_mean': float(np.mean(1 / k[has])),
        'pairs_sharing_a_cyp': float(has_cyp.mean()),
        'cyp_share_of_tied_genes_mean': float(np.mean(ncyp[has_cyp] / k[has_cyp])),   # EmerGNN hit@1 by tie-break
        'chance_to_name_one_specific_cyp_mean': float(np.mean(1 / k[has_cyp])),
    }


def k_stats(k):
    """Tie size k of one relation sequence over the pairs where it occurs."""
    k = k[k > 0]
    return {'pairs': int(len(k)), 'k_median': float(np.median(k)), 'k_p90': float(np.percentile(k, 90)),
            'share_k_ge_2': float((k >= 2).mean())}


res['top_2hop_sequences'] = {'→'.join(map(label_name, s)): {'paths': c, **k_stats(np.array([pc[s] for pc in per_pair]))}
                             for s, c in seq_total.most_common(8)}
if three is not None:
    top3 = sorted(cols, key=lambda x: -x[1].sum())[:8]
    res['top_3hop_sequences'] = {'→'.join(map(label_name, s)): {'paths': float(c.sum()), **k_stats(c)} for s, c in top3}
    res['3hop_max_tie_per_pair_q'] = [float(np.percentile([x['max_tie'] for x in three if x['paths'] > 0], z))
                                      for z in (50, 90)]

for hop in ['2hop', '3hop']:
    for g, r in res[hop].items():
        print(f'{hop} {g:10s}: paths/pair median {r["paths_q"][1]:.0f}, sequences median {r["sequences_q"][1]:.0f}, '
              f'tie size of a random path median {r["tie_size_q"][1]:.1f} [{r["tie_size_q"][0]:.1f}, '
              f'{r["tie_size_q"][2]:.1f}], tied share {r["tied_share_mean"]:.3f}, '
              f'chance to single out {r["single_out_chance_mean"]:.3f}')
for g, r in res['binds_binds'].items():
    print(f'binds-binds {g:10s}: share with a shared bound gene {r["pairs_sharing_a_bound_gene"]:.3f}, '
          f'k median {r["k_q"][1]:.0f} [{r["k_q"][0]:.0f}, {r["k_q"][2]:.0f}], 1/k {r["single_out_chance_mean"]:.3f}; '
          f'share with a shared CYP {r["pairs_sharing_a_cyp"]:.3f}, CYP share of tied genes '
          f'{r["cyp_share_of_tied_genes_mean"]:.3f}, chance to name one CYP {r["chance_to_name_one_specific_cyp_mean"]:.3f}')
for hop in ['top_2hop_sequences', 'top_3hop_sequences']:
    for s, r in res.get(hop, {}).items():
        print(f'{hop[4:8]} {s:24s} pairs {r["pairs"]:6d}  k median {r["k_median"]:5.0f}  p90 {r["k_p90"]:6.0f}  '
              f'share k>=2 {r["share_k_ge_2"]:.2f}')
json.dump(res, open(cli.out, 'w'), indent=1)
print('saved', cli.out)
