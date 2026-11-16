"""Shared pieces of the intervention analyses (gate_interventions.py, kg_ablation.py, stratify.py, ...).

Scripts run from DrugBank/ (data/ and saved_models/ are read relative to it). Everything else is
resolved from the repository root, so the split, the training seed and the key types are the only
things to set:

    --dataset S1_12 --train_seed 0 --keys llm name

  large files (probabilities)  runs/<tag>/...       (git-ignored)
  small tables                 results/<tag>/...    <tag> = <dataset>, or <dataset>_seed<k> for k > 0

Checkpoints are named saved_models/<dataset>_learned_shuf[_gate-<key>-v2]_seed<k>_<best|last>.pt;
parse_ckpt() refuses a checkpoint whose split or seed differs from the command line.
"""
import collections
import csv
import json
import os
import re

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENT_TSV = os.path.join(ROOT, 'entity_text/drugbank_entities.tsv')
EMB_PREFIX = os.path.join(ROOT, 'entity_text/emb/drugbank')
DATASET, TRAIN_SEED = 'S1_1', 0

METAB, PK_OTHER = {46, 3}, {72, 74, 76, 10}         # groups used by the earlier analyses
N_DRUG = 1710
KEYS = ['llm', 'id', 'random', 'name']              # changed in place by setup()
# mechanism classes for stratification: PK types name a pharmacokinetic process; type 48 ("risk or
# severity of adverse effects") names no mechanism; everything else names a pharmacodynamic effect
PK_WORDS = re.compile(r'metabolism|serum concentration|bioavailability|absorption|excretion|protein binding')
UNSPECIFIED = {48}
CKPT_RE = re.compile(r'(?P<dataset>S\d_\d+)_learned_shuf(?:_gate-(?P<feat>\w+?)(?P<v2>-v2)?)?'
                     r'_seed(?P<seed>\d+)_(?P<which>best|last)\.pt$')


# ------------------------------------------------------------------ split / seed / keys
def setup(p):
    """Add --dataset, --train_seed and --keys (unless the script defines them), parse, and apply."""
    global DATASET, TRAIN_SEED
    have = p._option_string_actions
    if '--dataset' not in have:
        p.add_argument('--dataset', default='S1_1', help='DrugBank split: S1_1, S1_12, S1_123, ...')
    if '--train_seed' not in have:
        p.add_argument('--train_seed', type=int, default=0, help='seed<k> in the checkpoint names')
    if '--keys' not in have:
        p.add_argument('--keys', nargs='+', default=list(KEYS), help='gate key types to analyse')
    cli = p.parse_args()
    DATASET, TRAIN_SEED = cli.dataset, cli.train_seed
    KEYS[:] = cli.keys
    return cli


def tag():
    return DATASET if TRAIN_SEED == 0 else f'{DATASET}_seed{TRAIN_SEED}'


def run_dir(name):
    return os.path.join(ROOT, 'runs', tag(), name)


def res_path(name):
    path = os.path.join(ROOT, 'results', tag(), name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def ckpt_path(key, which='best'):
    gate = '' if key == 'baseline' else f'_gate-{key}-v2'
    return f'saved_models/{DATASET}_learned_shuf{gate}_seed{TRAIN_SEED}_{which}.pt'


def parse_ckpt(path):
    """-> (key or None for the ungated baseline, 'v1' | 'v2', 'best' | 'last')."""
    m = CKPT_RE.search(os.path.basename(path))
    assert m, f'unrecognised checkpoint name: {path}'
    assert m['dataset'] == DATASET and int(m['seed']) == TRAIN_SEED, \
        f'{path} is {m["dataset"]} seed {m["seed"]}, but --dataset {DATASET} --train_seed {TRAIN_SEED} was given'
    return m['feat'], 'v2' if m['v2'] else 'v1', m['which']


# ------------------------------------------------------------------ data
def load_ddi(split, dataset=None):
    return np.array([l.split() for l in open(f'data/{dataset or DATASET}/{split}_ddi.txt')], dtype=int)


def mechanism_class():
    """DDI type -> 'metabolism' | 'pk_other' | 'pd' | 'unspecified'."""
    out = {}
    for k, desc in json.load(open('data/relation2id.json')).items():
        k = int(k)
        if k >= 86:
            continue
        out[k] = ('metabolism' if k in METAB else 'unspecified' if k in UNSPECIFIED
                  else 'pk_other' if PK_WORDS.search(desc) else 'pd')
    return out


def old_group(labels):
    """metabolism / pk_other / non_pk as in gate_interventions.py and explain_shared_cyp.py."""
    return np.array(['metabolism' if r in METAB else 'pk_other' if r in PK_OTHER else 'non_pk' for r in labels])


def kg_triples(dataset=None):
    """KG available at test time, one row per triple (h, t, r)."""
    return np.concatenate([np.loadtxt(f'data/{dataset or DATASET}/{s}_KG.txt', dtype=int)
                           for s in ['train', 'valid', 'test']])


def entities():
    return {int(r['ent_id']): r for r in csv.DictReader(open(ENT_TSV), delimiter='\t')}


def cyp_genes(ents):
    return {e for e, r in ents.items() if r['kind'] == 'Gene' and re.match(r'CYP\d', r['name'])}


def drug_binds(kg, ents):
    """DDI drug -> set of genes it binds (Compound-binds-Gene)."""
    cbg = [int(k) for k, v in json.load(open('data/relation2id.json')).items() if v == 'CbG'][0]
    binds = collections.defaultdict(set)
    for h, t, r in kg[kg[:, 2] == cbg]:
        d, g = (h, t) if h < N_DRUG else (t, h)
        if d < N_DRUG and ents[g]['kind'] == 'Gene':
            binds[d].add(g)
    return binds


def emerging_drug(pairs, train):
    """The drug of each test pair that has no training DDI (S1: exactly one per pair)."""
    seen = np.zeros(max(pairs[:, :2].max(), train[:, :2].max()) + 1, bool)
    seen[train[:, :2].ravel()] = True
    return np.where(seen[pairs[:, 0]], pairs[:, 1], pairs[:, 0])


# ------------------------------------------------------------------ statistics
def boot_ci(values, weights, clusters, n=1000):
    """95% interval of the weighted mean, resampling clusters (emerging drugs). Fixed seed 0."""
    ids, inv = np.unique(clusters, return_inverse=True)
    sums, counts = np.bincount(inv, weights=values * weights), np.bincount(inv, weights=weights)
    pick = np.random.RandomState(0).randint(0, len(ids), size=(n, len(ids)))
    means = sums[pick].sum(1) / counts[pick].sum(1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def boot_multiplicity(clusters, n=1000):
    """(n, len(clusters)) matrix: how often each pair's cluster is drawn in each bootstrap sample.
    Multiply it into the pair weights to bootstrap any weighted statistic."""
    ids, inv = np.unique(clusters, return_inverse=True)
    pick = np.random.RandomState(0).randint(0, len(ids), size=(n, len(ids)))
    times = np.stack([np.bincount(p, minlength=len(ids)) for p in pick])
    return times[:, inv].astype(np.float32)


def stat(values, weights, clusters):
    return {'mean': float(np.average(values, weights=weights)), 'ci': boot_ci(values, weights, clusters)}


def fmt(s, d=3):
    return f'{s["mean"]:+.{d}f} [{s["ci"][0]:+.{d}f},{s["ci"][1]:+.{d}f}]'


def nll(prob, y):
    return -np.log(prob[np.arange(len(y)), y])


def scale(prob, T):
    q = np.exp(np.log(prob) / T)
    return q / q.sum(1, keepdims=True)


def fit_T(prob, y, w):
    """Single temperature minimising the weighted NLL."""
    from scipy.optimize import minimize_scalar
    r = minimize_scalar(lambda lt: np.average(nll(scale(prob, np.exp(lt)), y), weights=w), bounds=(-3, 3),
                        method='bounded')
    return float(np.exp(r.x))


def per_pair(prob, y):
    conf, pred = prob.max(1), prob.argmax(1)
    onehot = np.zeros_like(prob)
    onehot[np.arange(len(y)), y] = 1
    return {'acc': (pred == y).astype(float), 'conf': conf, 'brier': ((prob - onehot) ** 2).sum(1), 'nll': nll(prob, y)}


def ece_boot(conf, correct, W, n_bins=15):
    """ECE for every row of the weight matrix W (n, pairs)."""
    b = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    onehot = np.eye(n_bins)[b] * (conf - correct)[:, None]
    return np.abs(W @ onehot).sum(1) / W.sum(1)


def load_probs(key, which='best', d=None):
    """Saved probabilities of gate_interventions.py, renormalised in float64."""
    z = np.load(f'{d or run_dir("gate_interventions")}/{key}-v2_{which}_probs.npz')
    probs = {}
    for c in z.files:
        if c in ('rows', 'weight'):
            continue
        p = np.clip(z[c].astype(np.float64), 1e-7, 1)
        probs[c] = p / p.sum(1, keepdims=True)
    return z['rows'], z['weight'].astype(np.float64), probs
