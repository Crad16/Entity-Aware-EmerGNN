"""Test-time KG variants for ablations F4 (CYP path removal) and M1 augmentation (KG-poor emerging
drugs). numpy only; kg_ablation.py builds GPU graphs from these masks and analyze_kg_ablation.py
reuses them to describe what each variant removed. Run from DrugBank/:

    python ../analysis/kg_variants.py        # prints what every variant removes

Each variant is a boolean keep-mask over kg_triples() (train + valid + test KG, the order in which
load_data.DataLoader builds the test-time graph). DDI edges are never touched.

  cyp      every Compound-Gene edge (binds, up- or downregulates) whose gene is a CYP, so no drug
           reaches a CYP in one hop
  hub      the same cut for the non-CYP genes bound by the most drugs, taken in order until at least
           as many edges are gone as under 'cyp'. Drug-metabolism and transport (ADME) genes are
           excluded, so these are mostly receptor and channel targets. Gene-level control: CYP3A4
           binds 516 drugs and no non-CYP gene more than 272, so genes cannot be matched one by one
  rand<k>  every drug loses as many edges to non-CYP, non-ADME genes as it lost CYP edges, chosen at
           random (draw k): binds edges first, then up/downregulation edges when it has too few
           binds edges. Drug-level control
  new<p>   every emerging test drug loses p% of its KG edges (any relation), chosen at random. The
           removed sets are nested: new50 within new90 within new100
"""
import json
import re

import numpy as np

from ablation_common import cyp_genes, emerging_drug, entities, kg_triples, load_ddi

ADME = re.compile(r'^(ABC[A-G]\d|SLC|UGT|SULT|NAT[12]$|FMO\d|GST|EPHX|CES\d|ALB$|ORM[12]$|AOX1$|POR$|BCHE$'
                  r'|MAO[AB]$|COMT$|TPMT$|ADH\d|ALDH|XDH$|NQO\d|AKR\d|DPYD$)')
F4 = ['cyp', 'hub', 'rand0', 'rand1']
M1 = ['new50', 'new90', 'new100']


def build(names=F4 + M1, seed=0):
    """name -> keep mask over kg_triples(), plus a description of what was removed."""
    kg, ents = kg_triples(), entities()
    cbg = [int(k) for k, v in json.load(open('data/relation2id.json')).items() if v == 'CbG'][0]
    n = max(ents) + 1
    kind = np.array([ents[e]['kind'] if e in ents else '' for e in range(n)])
    name = np.array([ents[e]['name'] if e in ents else '' for e in range(n)])
    h, t, r = kg.T
    is_drug, is_gene = np.isin(kind, ['Compound', 'Drug(DDI node)']), kind == 'Gene'
    cg = (is_drug[h] & is_gene[t]) | (is_drug[t] & is_gene[h])         # compound-gene edges
    gene, drug = np.where(is_gene[t], t, h), np.where(is_gene[t], h, t)  # read only where cg
    cyp_g = np.zeros(n, bool)
    cyp_g[list(cyp_genes(ents))] = True
    ok_g = is_gene & ~cyp_g & np.array([not ADME.match(s) for s in name])
    cyp_e = cg & cyp_g[gene]

    out, info = {}, {}
    for v in names:
        drop = np.zeros(len(kg), bool)
        if v == 'cyp':
            drop = cyp_e
        elif v == 'hub':
            binds = np.bincount(gene[cg & (r == cbg)], minlength=n)
            edges = np.bincount(gene[cg], minlength=n)
            order = [g for g in np.lexsort((np.arange(n), -binds)) if ok_g[g] and binds[g] > 0]
            k = np.searchsorted(np.cumsum(edges[order]), cyp_e.sum()) + 1
            drop = cg & np.isin(gene, order[:k])
            info[v] = {'genes': [str(name[g]) for g in order[:k]]}
        elif v.startswith('rand'):
            rng = np.random.RandomState(1000 * seed + int(v[4:]) + 1)
            want = np.bincount(drug[cyp_e], minlength=n)

            def by_drug(mask):
                c = np.where(mask)[0]
                return np.split(c[np.argsort(drug[c], kind='stable')], np.cumsum(np.bincount(drug[c], minlength=n))[:-1])
            binds_e, other_e = by_drug(cg & (r == cbg) & ok_g[gene]), by_drug(cg & (r != cbg) & ok_g[gene])
            filled = short = 0
            for d in np.where(want > 0)[0]:
                a = rng.choice(binds_e[d], min(want[d], len(binds_e[d])), replace=False)
                b = rng.choice(other_e[d], min(want[d] - len(a), len(other_e[d])), replace=False)
                drop[a] = drop[b] = True
                filled, short = filled + len(b), short + want[d] - len(a) - len(b)
            info[v] = {'filled_with_regulation_edges': int(filled), 'shortfall_edges': int(short)}
        elif v.startswith('new'):
            frac = int(v[3:]) / 100
            new = np.unique(emerging_drug(load_ddi('test'), load_ddi('train')))
            rng = np.random.RandomState(seed)                       # same draw for every level: nested
            for d in new:
                idx = np.where((h == d) | (t == d))[0]
                drop[rng.permutation(idx)[:int(round(frac * len(idx)))]] = True
        else:
            raise ValueError(v)
        out[v] = ~drop
        info.setdefault(v, {})['removed_edges'] = int(drop.sum())
    return kg, out, info


if __name__ == '__main__':
    import argparse
    from ablation_common import drug_binds, mechanism_class, setup
    setup(argparse.ArgumentParser())
    kg, keep, info = build()
    ents = entities()
    train, test = load_ddi('train'), load_ddi('test')
    mech = mechanism_class()
    cls = np.array([mech[y] for y in test[:, 2]])
    new = emerging_drug(test, train)
    binds_all = drug_binds(kg, ents)
    for v, m in keep.items():
        line = f'{v:7s} removes {(~m).sum():6d} edges'
        if v.startswith('new'):
            deg = np.bincount(kg[:, :2].ravel(), minlength=len(ents))
            deg_v = np.bincount(kg[m][:, :2].ravel(), minlength=len(ents))
            u = np.unique(new)
            line += f'; emerging-drug degree median {np.median(deg[u]):.0f} -> {np.median(deg_v[u]):.0f}'
        else:
            b = drug_binds(kg[m], ents)
            lost = np.array([len(binds_all[u] & binds_all[w]) > len(b[u] & b[w]) for u, w, _ in test])
            line += '; test pairs losing a shared bound gene: ' + ', '.join(
                f'{c} {lost[cls == c].mean():.2f}' for c in ['metabolism', 'pk_other', 'pd', 'unspecified'])
        print(line, {k: x for k, x in info[v].items() if k != 'removed_edges'} or '')
