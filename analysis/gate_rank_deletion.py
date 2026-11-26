"""Deletion of the entities a gate ranks high or low for each pair (faithfulness of the gate ranking). Run from
DrugBank/ after gate_interventions.py:

    python ../analysis/gate_rank_deletion.py --train_seed 1 \
        --ckpt saved_models/S1_1_learned_shuf_gate-llm-v2_seed1_best.pt saved_models/S1_1_learned_shuf_gate-name-v2_seed1_best.pt \
        --baseline saved_models/S1_1_learned_shuf_seed1_best.pt

For every sampled test pair (u, v) the candidates are the non-drug entities adjacent to both drugs in the
test-time KG (the mediators of its 2-hop paths). A gated model scores each candidate by its gate for that pair
(mean over the gated layers). Pairs with at least two candidates get six deletion sets:

  top / bottom   the k = max(1, n // 4) candidates with the highest / lowest gate (ties broken at random)
  rand0 / rand1  k candidates drawn at random
  mH / mL        degree-matched halves: candidates are sorted by KG degree and paired with their neighbour in that
                 order; in each pair the entity with the higher gate goes to mH, the other to mL (an unpaired last
                 entity stays in the graph). The two sets have the same size and almost the same degrees.

An entity is deleted for one pair only, by zeroing its hidden state before every propagation step after the
first (before the first step a non-drug entity carries nothing). Messages leave an entity only through that
state, so this equals removing all of its edges from the graph for that pair; --selftest compares the two on
the CYP genes. Each set is evaluated with the gate ('original'), without it ('off', the EmerGNN forward pass
of the same weights) and, with --baseline, on the ungated baseline of the same seed ('baseline'). Probabilities
go to runs/<tag>/gate_rank_deletion/<key>-v2_<best|last>.npz as <condition>__<set>; existing files are skipped.
analyze_gate_rank_deletion.py reads them; full-KG references come from gate_interventions.py and kg_ablation.py.
"""
import argparse
import collections
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.getcwd())
from load_data import DataLoader  # noqa: E402
from models import EmerGNN  # noqa: E402
from ablation_common import EMB_PREFIX, N_DRUG, cyp_genes, entities, parse_ckpt, run_dir, setup  # noqa: E402

SETS = ['top', 'bottom', 'rand0', 'rand1', 'mH', 'mL']
BASE_SETS = ['top', 'bottom', 'rand0', 'mH', 'mL']

p = argparse.ArgumentParser()
p.add_argument('--ckpt', nargs='+', required=True, help='gated checkpoints')
p.add_argument('--baseline', default=None, help='ungated checkpoint of the same seed (optional)')
p.add_argument('--batch', type=int, default=32)
p.add_argument('--limit', type=int, default=None, help='N pairs spread over the sample (quick check)')
p.add_argument('--selftest', action='store_true', help='compare state zeroing with edge removal on the CYP genes, then exit')
p.add_argument('--reference_dir', default=None, help='default: runs/<tag>/gate_interventions')
p.add_argument('--out_dir', default=None, help='default: runs/<tag>/gate_rank_deletion')
cli = setup(p)
cli.reference_dir = cli.reference_dir or run_dir('gate_interventions')
cli.out_dir = cli.out_dir or run_dir('gate_rank_deletion')
os.makedirs(cli.out_dir, exist_ok=True)

torch.manual_seed(0)
np.random.seed(0)
args = argparse.Namespace(task_dir='./', dataset=cli.dataset, length=3, feat='M', attn_mode='learned', n_dim=64,
                          gate_dim=32, emb_prefix=EMB_PREFIX, lamb=0, lr=1e-3)
data = DataLoader(args)
args.all_ent, args.all_rel, args.eval_rel = data.all_ent, data.all_rel, data.eval_rel
del data.KG, data.vKG
torch.cuda.empty_cache()
n_ent = data.all_ent


class DeletionGate(nn.Module):
    """Stands in for model.entity_gate: the trained gate (or none) times a 0/1 mask (n_ent, batch) on layers >= 1."""

    def __init__(self, gate, n_layer):
        super().__init__()
        self.gate, self.n_layer, self.gate_on, self.mask = gate, n_layer, True, None

    def forward(self, ht_embed, keys=None, center=None):
        g = self.gate(ht_embed, center=center) if self.gate is not None and self.gate_on else [None] * self.n_layer
        if self.mask is None:
            return g
        return [x if l == 0 else (self.mask if x is None else x * self.mask) for l, x in enumerate(g)]


def load_model(path):
    feat, version, which = parse_ckpt(path)
    a = argparse.Namespace(**vars(args), entity_feat=feat or 'none', gate_version='v2')
    model = EmerGNN(data.eval_ent, data.eval_rel, a).cuda().eval()
    model.load_state_dict(torch.load(path, map_location='cuda'))
    wrap = DeletionGate(model.entity_gate, model.L)
    model.entity_gate = wrap
    return model, wrap, (f'{feat}-v2_{which}' if feat else f'baseline_{which}')


# ------------------------------------------------------------------ sample and candidates
ref_file = os.path.join(cli.reference_dir, os.path.basename(parse_ckpt(cli.ckpt[0])[0]) + f'-v2_{parse_ckpt(cli.ckpt[0])[2]}_probs.npz')
sample = np.load(ref_file)
rows, w = sample['rows'], sample['weight'].astype(float)
if cli.limit:
    pick = np.unique(np.linspace(0, len(rows) - 1, cli.limit).astype(int))
    rows, w = rows[pick], w[pick]
test = data.triplets['test']
labels = test[rows, 2]
heads = torch.LongTensor(test[rows, 0]).cuda()
tails = torch.LongTensor(test[rows, 1]).cuda()

kg = data.test_kg
degree = np.bincount(kg[:, :2].ravel(), minlength=n_ent)
nbrs = collections.defaultdict(set)
for h, t, _ in kg:
    if h < N_DRUG <= t:
        nbrs[h].add(t)
    elif t < N_DRUG <= h:
        nbrs[t].add(h)
cand = [np.array(sorted(nbrs[u] & nbrs[v]), dtype=np.int64) for u, v, _ in test[rows]]
ncand = np.array([len(c) for c in cand])
eligible = ncand >= 2
is_cyp = np.zeros(n_ent, bool)
is_cyp[list(cyp_genes(entities()))] = True
print(f'{len(rows)} pairs; {eligible.sum()} with >= 2 shared non-drug neighbours (median {int(np.median(ncand[eligible]))}, '
      f'mean {ncand[eligible].mean():.1f}); weighted share {w[eligible].sum() / w.sum():.3f}', flush=True)


@torch.no_grad()
def predict(model, wrap, KG, gate_on, sets=None):
    """sets: per pair an array of entities to delete (None or empty: nothing)."""
    wrap.gate_on = gate_on
    probs = []
    for s in range(0, len(rows), cli.batch):
        h, t = heads[s:s + cli.batch], tails[s:s + cli.batch]
        wrap.mask = None
        if sets is not None:
            ent = np.concatenate([sets[i] for i in range(s, s + len(h))] or [np.zeros(0, np.int64)])
            col = np.concatenate([np.full(len(sets[i]), i - s) for i in range(s, s + len(h))] or [np.zeros(0, np.int64)])
            mask = torch.ones(n_ent, len(h), device='cuda')
            mask[torch.as_tensor(ent, device='cuda'), torch.as_tensor(col, device='cuda')] = 0
            wrap.mask = mask
        probs.append(F.softmax(model.enc_r(model.enc_ht(h, t, KG)), dim=-1).cpu().numpy())
    wrap.mask = None
    return np.concatenate(probs)


@torch.no_grad()
def gate_scores(model, wrap):
    """Per pair: gate of each candidate, mean over the gated layers."""
    out = []
    for s in range(0, len(rows), cli.batch):
        h, t = heads[s:s + cli.batch], tails[s:s + cli.batch]
        ht = torch.cat([model.Went(model.ent_kg[h]), model.Went(model.ent_kg[t])], dim=-1)
        g = torch.stack([x for x in wrap.gate(ht, center=model.pair_center()) if x is not None]).mean(0)   # (n_ent, batch)
        for j in range(len(h)):
            out.append(g[torch.as_tensor(cand[s + j], device='cuda'), j].cpu().numpy().astype(np.float64))
    return out


def build_sets(score, seed):
    """-> {set name: list of entity arrays per pair}, per-pair statistics."""
    rng = np.random.RandomState(seed)
    sets = {s: [np.zeros(0, np.int64)] * len(rows) for s in SETS}
    tied = np.full(len(rows), np.nan)
    for i in np.where(eligible)[0]:
        c, g = cand[i], score[i]
        n, k = len(c), max(1, len(c) // 4)
        order = np.lexsort((rng.rand(n), g))                       # ascending gate, random within ties
        sets['bottom'][i], sets['top'][i] = c[order[:k]], c[order[-k:]]
        sets['rand0'][i], sets['rand1'][i] = c[rng.permutation(n)[:k]], c[rng.permutation(n)[:k]]
        by_deg = np.lexsort((rng.rand(n), degree[c]))
        a, b = by_deg[0:n - n % 2:2], by_deg[1:n - n % 2:2]
        coin = rng.rand(len(a)) < 0.5
        a_high = np.where(g[a] == g[b], coin, g[a] > g[b])
        sets['mH'][i], sets['mL'][i] = c[np.where(a_high, a, b)], c[np.where(a_high, b, a)]
        tied[i] = np.mean(g[a] == g[b])
    stats = {'tied_matched_pairs': tied}
    for s in SETS:
        stats[f'{s}_n'] = np.array([len(x) for x in sets[s]])
        with np.errstate(invalid='ignore'), __import__('warnings').catch_warnings():
            __import__('warnings').simplefilter('ignore')
            stats[f'{s}_logdeg'] = np.array([np.log(degree[x]).mean() if len(x) else np.nan for x in sets[s]])
            stats[f'{s}_cyp'] = np.array([is_cyp[x].mean() if len(x) else np.nan for x in sets[s]])
            stats[f'{s}_gate'] = np.array([score[i][np.searchsorted(cand[i], x)].mean() if len(x) else np.nan
                                           for i, x in enumerate(sets[s])])
    return sets, stats


def acc(pr):
    return np.average((pr.argmax(1) == labels)[eligible], weights=w[eligible])


# ------------------------------------------------------------------ self-test
if cli.selftest:
    model, wrap, key = load_model(cli.ckpt[0])
    S = np.where(is_cyp)[0]
    full = predict(model, wrap, data.tKG, True)
    by_mask = predict(model, wrap, data.tKG, True, [S] * len(rows))
    keep = ~(is_cyp[kg[:, 0]] | is_cyp[kg[:, 1]])
    KG = data.load_graph(np.concatenate([data.triplets['train'], data.triplets['valid'], kg[keep]], axis=0))
    by_edges = predict(model, wrap, KG, True)
    off_mask, off_edges = predict(model, wrap, data.tKG, False, [S] * len(rows)), predict(model, wrap, KG, False)
    print(f'selftest {key}: {(~keep).sum()} edges touch a CYP gene; max |p_mask - p_edges| gate on '
          f'{np.abs(by_mask - by_edges).max():.2e}, gate off {np.abs(off_mask - off_edges).max():.2e}; '
          f'max |p_mask - p_full| {np.abs(by_mask - full).max():.2e} (must be large)')
    sys.exit(0)

# ------------------------------------------------------------------ run
all_sets = {}
for path in cli.ckpt:
    model, wrap, key = load_model(path)
    out = os.path.join(cli.out_dir, f'{key}.npz')
    score = gate_scores(model, wrap)
    sets, stats = build_sets(score, seed=0)
    all_sets[key] = sets
    if os.path.exists(out) and not cli.limit:
        print(f'{key}: exists, skipped', flush=True)
        del model
        continue
    el = eligible
    print(f'\n== {key}: mean gate top {np.nanmean(stats["top_gate"][el]):.3f} bottom {np.nanmean(stats["bottom_gate"][el]):.3f} '
          f'rand {np.nanmean(stats["rand0_gate"][el]):.3f} | mH {np.nanmean(stats["mH_gate"][el]):.3f} mL {np.nanmean(stats["mL_gate"][el]):.3f}; '
          f'mean log degree top {np.nanmean(stats["top_logdeg"][el]):.2f} bottom {np.nanmean(stats["bottom_logdeg"][el]):.2f} '
          f'mH {np.nanmean(stats["mH_logdeg"][el]):.2f} mL {np.nanmean(stats["mL_logdeg"][el]):.2f}; '
          f'tied matched pairs {np.nanmean(stats["tied_matched_pairs"][el]):.3f}', flush=True)
    probs = {}
    for cond in ['original', 'off']:
        for s in SETS:
            t0 = time.time()
            pr = probs[f'{cond}__{s}'] = predict(model, wrap, data.tKG, cond == 'original', sets[s])
            sec = time.time() - t0
            print(f'  {key:14s} {cond:8s} {s:7s} {sec:5.0f}s  acc (eligible pairs) {acc(pr):.3f}'
                  + (f'  (~{sec * len(sample["rows"]) / len(rows) / 60:.1f} min for the whole sample)' if cli.limit else ''), flush=True)
    if not cli.limit:
        np.savez_compressed(out, rows=rows, weight=w, eligible=eligible, ncand=ncand,
                            **{f'stat_{k}': v.astype(np.float32) for k, v in stats.items()},
                            **{c: x.astype(np.float16) for c, x in probs.items()})
    del model
    torch.cuda.empty_cache()

if cli.baseline:
    model, wrap, bkey = load_model(cli.baseline)
    for key, sets in all_sets.items():
        out = os.path.join(cli.out_dir, f'{bkey}__sets-{key}.npz')
        if os.path.exists(out) and not cli.limit:
            print(f'{bkey} with the sets of {key}: exists, skipped', flush=True)
            continue
        probs = {}
        for s in BASE_SETS:
            t0 = time.time()
            pr = probs[f'baseline__{s}'] = predict(model, wrap, data.tKG, False, sets[s])
            print(f'  {bkey:14s} sets of {key:12s} {s:7s} {time.time() - t0:5.0f}s  acc (eligible pairs) {acc(pr):.3f}', flush=True)
        if not cli.limit:
            np.savez_compressed(out, rows=rows, weight=w, eligible=eligible, **{c: x.astype(np.float16) for c, x in probs.items()})
print(f'peak GPU memory {torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GB; saved to {cli.out_dir}', flush=True)
