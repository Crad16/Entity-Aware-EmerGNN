"""Side-by-side key numbers of the report for several splits (folds).

Reads the tables written by the pipeline in analysis/REPRODUCE.md (results/<split>/...) and the
training curves, and writes
    results/fold_summary.md    one markdown table per report chapter, one column per split
    results/fold_summary.csv   the same numbers, one row per (chapter, item, split)
    results/fold_summary.json
Cells are 'mean [95% CI]*'; * marks an interval that excludes 0 (for AUC rows: excludes 0.5).
For S1_1 it prefers stratify_llm_name.json (strata averaged over the llm and name models, as in the
report) and kg_ablation_llm_name_baseline.json (adds the ungated baseline) when they exist.

    python analysis/fold_summary.py --splits S1_1 S1_12 S1_123 \
        --curves <dir with drugbank_S1_1_*.csv> --eval_logs <DrugBank/results with <split>_*_eval.txt>
"""
import argparse
import csv
import json
import os
import re

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
p = argparse.ArgumentParser()
p.add_argument('--splits', nargs='+', default=['S1_1', 'S1_12', 'S1_123'])
p.add_argument('--curves', default=os.path.join(ROOT, 'results/curves'))
p.add_argument('--eval_logs', default=os.path.join(ROOT, 'DrugBank/results'))
p.add_argument('--out', default=os.path.join(ROOT, 'results/fold_summary'))
cli = p.parse_args()
MODELS = ['baseline', 'llm', 'name']


def load(split, *names):
    for n in names:
        f = os.path.join(ROOT, 'results', split, n)
        if os.path.exists(f):
            return json.load(open(f))
    return None


def curve(split, model):
    """Test macro-F1 averaged over evaluations at epochs 10-20, and the validation-best epoch."""
    rows = []
    name = {'baseline': 'emergnn', 'llm': 'gate-v2_llm', 'name': 'gate-v2_name'}[model]
    f = os.path.join(cli.curves, f'drugbank_{split}_{name}.csv')
    if os.path.exists(f):
        for r in csv.DictReader(open(f)):
            rows.append((int(r['epoch']), float(r['valid_f1']), float(r['test_f1']), float(r['test_acc'])))
    else:
        tag = '' if model == 'baseline' else f'_gate-{model}-v2'
        f = os.path.join(cli.eval_logs, f'{split}_learned_shuf{tag}_seed0_eval.txt')
        if not os.path.exists(f):
            return None
        for line in open(f):
            # re.match: skip the trailing "Best results: epoch:..." line, which repeats the best epoch
            m = re.match(r'epoch:(\d+).*\[Valid\] f1:([\d.]+).*\[Test\] f1:([\d.]+) acc:([\d.]+)', line)
            if m:
                rows.append((int(m[1]), float(m[2]), float(m[3]), float(m[4])))
    rows = [r for r in rows if r[0] <= 20]
    if not rows:
        return None
    best = max(rows, key=lambda r: r[1])
    late = [r[2] for r in rows if 10 <= r[0] <= 20]
    return {'f1_ep10_20': float(np.mean(late)) if late else None, 'best_epoch': best[0],
            'best_test_f1': best[2], 'best_test_acc': best[3], 'last_epoch': rows[-1][0]}


def g(d, *path):
    for k in path:
        if d is None:
            return None
        d = d.get(k) if isinstance(d, dict) else None
    return d


def cell(v, null=0.0):
    """-> (text, mean, lo, hi)"""
    if v is None:
        return 'n/a', None, None, None
    if isinstance(v, dict):
        m = v.get('mean', v.get('dlogp'))
        ci = v.get('ci', v.get('dlogp_ci'))
        if m is None:
            return 'n/a', None, None, None
        if ci:
            star = '*' if ci[0] > null or ci[1] < null else ''
            f = '+.3f' if null == 0 else '.3f'              # AUC rows (null 0.5) without a sign
            return f'{m:{f}} [{ci[0]:{f}}, {ci[1]:{f}}]{star}', m, ci[0], ci[1]
        return f'{m:.3f}', m, None, None
    if isinstance(v, (int, np.integer)):
        return f'{v:,}', v, None, None
    return f'{v:.3f}', v, None, None


D = {}
for s in cli.splits:
    D[s] = {'rt': load(s, 'relation_ties.json'), 'st': load(s, 'stratify_llm_name.json', 'stratify.json'),
            'fu': load(s, 'gate_interventions/followup.json'),
            'gi': {k: load(s, f'gate_interventions/{k}-v2_best.json') for k in ('llm', 'name')},
            'kg': load(s, 'kg_ablation_llm_name_baseline.json', 'kg_ablation.json'),
            'gd': load(s, 'gate_degree.json'), 'kc': load(s, 'kg_calibration.json'),
            'cv': {m: curve(s, m) for m in MODELS}}


def kgm(d, model):            # model key inside kg_ablation.json
    return model


CH = []      # (chapter title, [(label, fn(split data) -> value, null)])

CH.append(('0. Performance (test macro-F1, mean over epochs 10-20) / validation-best epoch', [
    (f'{m} F1 (mean, epochs 10-20)', (lambda m: lambda d: g(d['cv'][m], 'f1_ep10_20'))(m), 0) for m in MODELS] + [
    (f'{m} best epoch / test acc', (lambda m: lambda d: None if d['cv'][m] is None else
                                  f"{d['cv'][m]['best_epoch']} / {d['cv'][m]['best_test_acc']:.3f}")(m), 0) for m in MODELS]))

CH.append(('1. The original attention cannot single out an entity (relation_ties.json)', [
    ('test pairs', lambda d: g(d['rt'], '2hop', 'all', 'pairs'), 0),
    ('pairs with a 2-hop path', lambda d: g(d['rt'], '2hop', 'all', 'pairs_with_paths'), 0),
    ('2-hop tied share (all)', lambda d: g(d['rt'], '2hop', 'all', 'tied_share_mean'), 0),
    ('2-hop tied share (metabolism)', lambda d: g(d['rt'], '2hop', 'metabolism', 'tied_share_mean'), 0),
    ('2-hop tied share (non-PK)', lambda d: g(d['rt'], '2hop', 'non_pk', 'tied_share_mean'), 0),
    ('2-hop tie size, median', lambda d: g(d['rt'], '2hop', 'all', 'tie_size_q', 1) if False else
        (None if g(d['rt'], '2hop', 'all', 'tie_size_q') is None else g(d['rt'], '2hop', 'all', 'tie_size_q')[1]), 0),
    ('2-hop chance to single out one path', lambda d: g(d['rt'], '2hop', 'all', 'single_out_chance_mean'), 0),
    ('3-hop tied share', lambda d: g(d['rt'], '3hop', 'all', 'tied_share_mean'), 0),
    ('3-hop chance to single out one path', lambda d: g(d['rt'], '3hop', 'all', 'single_out_chance_mean'), 0),
    ('metabolism pairs sharing a bound gene', lambda d: g(d['rt'], 'binds_binds', 'metabolism', 'pairs_sharing_a_bound_gene'), 0),
    ('metabolism pairs sharing a CYP', lambda d: g(d['rt'], 'binds_binds', 'metabolism', 'pairs_sharing_a_cyp'), 0),
    ('non-PK pairs sharing a CYP', lambda d: g(d['rt'], 'binds_binds', 'non_pk', 'pairs_sharing_a_cyp'), 0),
    ('CYP share of tied genes (metabolism)', lambda d: g(d['rt'], 'binds_binds', 'metabolism', 'cyp_share_of_tied_genes_mean'), 0),
    ('chance to name one specific CYP (metabolism)', lambda d: g(d['rt'], 'binds_binds', 'metabolism', 'chance_to_name_one_specific_cyp_mean'), 0),
]))

STRATA = [('all', 'all'), ('metabolism', 'mech=metabolism'), ('other PK', 'mech=pk_other'), ('PD', 'mech=pd'),
          ('unspecified', 'mech=unspecified'), ('shared CYP', 'mediator=cyp'), ('no shared gene', 'mediator=none'),
          ('KG low', 'kg=low'), ('KG mid', 'kg=mid'), ('KG high', 'kg=high'), ('metabolism & CYP', 'metabolism & cyp'),
          ('PD & CYP', 'pd & cyp'), ('PD & no shared gene', 'pd & none')]
CH.append(('2-2. Accuracy / confidence by stratum (mean of llm and name with the gate off, test sample)', [
    (f'{lab}: accuracy / confidence', (lambda k: lambda d: None if g(d['st'], k) is None else
                          f"{d['st'][k]['off_acc']:.3f} / {d['st'][k]['off_conf']:.3f} (n={d['st'][k]['n_pairs']})")(k), 0)
    for lab, k in STRATA]))

PULL = [('metabolism & CYP', 'metabolism&shared_cyp'), ('other PK & CYP', 'pk_other&shared_cyp'),
        ('other PK & no shared CYP', 'pk_other&no_shared_cyp'), ('PD & CYP', 'pd&shared_cyp'), ('PD & no shared CYP', 'pd&no_shared_cyp')]
CH.append(('2-3. Share predicted as a metabolism type (full KG)', [
    (f'{lab}: {m}', (lambda k, m: lambda d: g(d['kg'], 'f4', f'{m}/original', 'pull', k, 'full', 'mean'))(k, m), 0)
    for lab, k in PULL for m in MODELS]))

rows = []
for m in MODELS:
    rows += [(f'{m}: accuracy, full KG', (lambda m: lambda d: g(d['kg'], 'accuracy', f'{m}/original', 'full'))(m), 0),
             (f'{m}: accuracy, CYP edges removed', (lambda m: lambda d: g(d['kg'], 'accuracy', f'{m}/original', 'cyp'))(m), 0),
             (f'{m}: accuracy, controls removed (hub / rand0 / rand1)', (lambda m: lambda d: None if g(d['kg'], 'accuracy', f'{m}/original') is None
                else ' / '.join(f"{d['kg']['accuracy'][f'{m}/original'][v]:.3f}" for v in ('hub', 'rand0', 'rand1')))(m), 0),
             (f'{m}: metabolism dlogp (CYP edges removed)', (lambda m: lambda d: g(d['kg'], 'f4', f'{m}/original', 'dlogp_cyp', 'metabolism'))(m), 0),
             (f'{m}: PD dlogp', (lambda m: lambda d: g(d['kg'], 'f4', f'{m}/original', 'dlogp_cyp', 'pd'))(m), 0),
             (f'{m}: other PK dlogp', (lambda m: lambda d: g(d['kg'], 'f4', f'{m}/original', 'dlogp_cyp', 'pk_other'))(m), 0),
             (f'{m}: shared-CYP pairs predicted as metabolism, before -> after', (lambda m: lambda d: None if g(d['kg'], 'f4', f'{m}/original', 'shared_cyp_metab_pred') is None
                else '{:.2f} -> {:.2f}'.format(*(d['kg']['f4'][f'{m}/original']['shared_cyp_metab_pred'][v]['rate']['mean'] for v in ('full', 'cyp'))))(m), 0),
             (f'{m}: precision of those predictions, before -> after', (lambda m: lambda d: None if g(d['kg'], 'f4', f'{m}/original', 'shared_cyp_metab_pred') is None
                else '{:.2f} -> {:.2f}'.format(*(d['kg']['f4'][f'{m}/original']['shared_cyp_metab_pred'][v]['precision']['mean'] for v in ('full', 'cyp'))))(m), 0),
             (f'{m}: dlogp of PD pairs that lost a shared receptor (hub)', (lambda m: lambda d: g(d['kg'], 'f4', f'{m}/original', 'by_lost_gene', 'hub/pd/lost'))(m), 0)]
for m in ('llm', 'name'):
    rows += [(f'{m}: metabolism dlogp, gate off', (lambda m: lambda d: g(d['kg'], 'f4', f'{m}/off', 'dlogp_cyp', 'metabolism'))(m), 0),
             (f'{m}: metabolism dlogp, gate on - off', (lambda m: lambda d: g(d['kg'], 'f4', f'{m}/original', 'gate_original-off', 'metabolism'))(m), 0)]
CH.append(('2-4. Removing the CYP edges (F4)', rows))

CH.append(("2-5. Removing the emerging drug's KG edges (M1): accuracy", [
    (f'{m}: 0% / 50% / 90% / 100%', (lambda m: lambda d: None if g(d['kg'], 'm1', f'{m}/original', 'raw') is None else
        ' / '.join(f"{d['kg']['m1'][f'{m}/original']['raw'][lv]['acc']:.3f}" for lv in ('full', 'new50', 'new90', 'new100')))(m), 0)
    for m in MODELS] + [
    ("reference without KG (known drug's most frequent type)", lambda d: g(d['kg'], 'm1_reference', 'existing_drug_top_type', 'mean'), 0),
    ('llm - name difference in accuracy drop (90%)', lambda d: g(d['kg'], 'm1_key_vs_key', 'new90/llm-name'), 0)]))


def tert(d, m, kind, stat):
    t = g(d['kc'], 'models', m, 'original', f'{kind}_shared_neighbours')
    return None if t is None else ' / '.join(f"{t[b][stat]:.3f}" for b in ('low', 'mid', 'high'))


rows = [('shared-neighbour cut points a, b (np.digitize: low < a <= mid < b <= high)', lambda d: None if g(d['kc'], 'cut_points') is None else
         '{:.0f}, {:.0f}'.format(*d['kc']['cut_points']['shared_neighbours']), 0)]
for m in ('llm', 'name'):
    rows += [(f'{m}: accuracy low / mid / high', (lambda m: lambda d: tert(d, m, 'raw', 'acc'))(m), 0),
             (f'{m}: confidence low / mid / high', (lambda m: lambda d: tert(d, m, 'raw', 'conf'))(m), 0),
             (f'{m}: accuracy difference (low - high)', (lambda m: lambda d: g(d['kc'], 'models', m, 'original', 'raw_shared_neighbours', 'low-high', 'acc'))(m), 0),
             (f'{m}: gap (confidence - accuracy) difference (low - high), raw', (lambda m: lambda d: g(d['kc'], 'models', m, 'original', 'raw_shared_neighbours', 'low-high', 'gap'))(m), 0),
             (f'{m}: same, after temperature scaling', (lambda m: lambda d: g(d['kc'], 'models', m, 'original', 'temp_shared_neighbours', 'low-high', 'gap'))(m), 0)]
CH.append(('2-6. Over-confidence by number of shared neighbours (kg_calibration.json, gated models)', rows))

rows = []
for m in ('llm', 'name'):
    rows.append((f'{m}: accuracy original / off / mean / shuffle_all / shuffle_type / key_perm',
                 (lambda m: lambda d: None if d['gi'][m] is None else ' / '.join(
                     f"{d['gi'][m]['summary'][c]['acc']:.3f}" for c in ('original', 'off', 'mean', 'shuffle_all', 'shuffle_type', 'key_perm')))(m), 0))
for m in ('llm', 'name'):
    rows += [(f'{m}: F2 key_perm dlogp (metabolism)', (lambda m: lambda d: g(d['gi'][m], 'summary', 'key_perm', 'metabolism'))(m), 0),
             (f'{m}: F2 key_perm dlogp (non-PK)', (lambda m: lambda d: g(d['gi'][m], 'summary', 'key_perm', 'non_pk'))(m), 0),
             (f'{m}: share of predictions changed by key_perm / off (metabolism)', (lambda m: lambda d: None if d['gi'][m] is None else
                 '{:.1%} / {:.1%}'.format(d['gi'][m]['summary']['key_perm']['metabolism']['flip'], d['gi'][m]['summary']['off']['metabolism']['flip']))(m), 0),
             (f'{m}: key_perm - off (metabolism)', (lambda m: lambda d: g(d['fu'], m, 'metabolism', 'key_perm-off'))(m), 0),
             (f'{m}: shuffle_all - off (metabolism)', (lambda m: lambda d: g(d['fu'], m, 'metabolism', 'shuffle_all-off'))(m), 0),
             (f'{m}: mean - off (metabolism)', (lambda m: lambda d: g(d['fu'], m, 'metabolism', 'mean-off'))(m), 0),
             (f'{m}: F1 KL(same type) - KL(any pair) (metabolism)', (lambda m: lambda d: g(d['fu'], m, 'metabolism', 'kl_type-all'))(m), 0),
             (f'{m}: F1 (non-PK)', (lambda m: lambda d: g(d['fu'], m, 'non_pk', 'kl_type-all'))(m), 0),
             (f'{m}: key_perm, metabolism & shared CYP', (lambda m: lambda d: g(d['fu'], m, 'metabolism', 'key_perm-original_with_shared_cyp'))(m), 0),
             (f'{m}: key_perm, non-PK & shared CYP', (lambda m: lambda d: g(d['fu'], m, 'non_pk', 'key_perm-original_with_shared_cyp'))(m), 0),
             (f'{m}: key_perm, type 46', (lambda m: lambda d: g(d['fu'], m, 'type_46', 'key_perm'))(m), 0),
             (f'{m}: gate gain NLL(off)-NLL(original), metabolism & CYP', (lambda m: lambda d: g(d['st'], 'metabolism & cyp', f'gate_gain/{m}'))(m), 0),
             (f'{m}: gate gain, all', (lambda m: lambda d: g(d['st'], 'all', f'gate_gain/{m}'))(m), 0)]


def n_sig_gain(d, m):
    if d['st'] is None:
        return None
    vals = [v.get(f'gate_gain/{m}') for v in d['st'].values() if isinstance(v, dict) and v.get(f'gate_gain/{m}')]
    up = sum(1 for v in vals if v['ci'][0] > 0)
    down = sum(1 for v in vals if v['ci'][1] < 0)
    return f'{up + down} / {len(vals)} (gain {up}, loss {down})'


rows += [(f'{m}: strata with a significant gate gain', (lambda m: lambda d: n_sig_gain(d, m))(m), 0) for m in ('llm', 'name')]
CH.append(('3. Gate interventions: query replacement and key permutation (F1 + F2)', rows))

rows = []
for name, k in [('EmerGNN (ties)', 'EmerGNN (ties)'), ('degree order', 'degree heuristic'),
                ('llm best', 'llm-v2 [best]'), ('name best', 'name-v2 [best]')]:
    for grp, lab in [('metabolism', 'metabolism'), ('non_pk', 'non-PK'), ('gap', 'gap')]:
        rows.append((f'{name}: AUC {lab}', (lambda k, grp: lambda d: g(d['gd'], k, 'e2', grp, 'auc'))(k, grp), 0 if grp == 'gap' else 0.5))
    rows.append((f'{name}: AUC metabolism, no degree advantage', (lambda k: lambda d: g(d['gd'], k, 'e2', 'metabolism', 'adverse_auc'))(k), 0.5))
for m in ('llm', 'name'):
    k = f'{m}-v2 [best]'
    rows += [(f'{m}: rank correlation of gate and degree, largest absolute value (genes)', (lambda k: lambda d: None if g(d['gd'], k, 'e4') is None else max(
                abs(v) for n, v in d['gd'][k]['e4'].items() if n.endswith('_genes') and isinstance(v, (int, float))))(k), 0),
             (f'{m}: CYP gate shift after controlling for degree', (lambda k: lambda d: None if g(d['gd'], k, 'e4', 'shift_regression') is None else
                {'mean': d['gd'][k]['e4']['shift_regression']['cyp_coef'], 'ci': d['gd'][k]['e4']['shift_regression']['cyp_coef_ci']})(k), 0)]
CH.append(('4. Does the gate rank the shared CYP first? (E2, E4)', rows))

CH.append(('5. llm against name', [
    ('AUC metabolism, llm - name', lambda d: None if d['gd'] is None else
        d['gd']['llm-v2 [best]']['e2']['metabolism']['auc']['mean'] - d['gd']['name-v2 [best]']['e2']['metabolism']['auc']['mean'], 0),
    ('AUC non-PK, llm - name', lambda d: None if d['gd'] is None else
        d['gd']['llm-v2 [best]']['e2']['non_pk']['auc']['mean'] - d['gd']['name-v2 [best]']['e2']['non_pk']['auc']['mean'], 0)] + [
    (f'NLL(name) - NLL(llm), {lab}', (lambda k: lambda d: g(d['st'], k, 'nll_name-llm'))(k), 0)
    for lab, k in [('all', 'all'), ('metabolism', 'mech=metabolism'), ('other PK', 'mech=pk_other'), ('PD', 'mech=pd'),
                   ('unspecified', 'mech=unspecified')]]))

md, flat = [], []
for title, rows in CH:
    md += [f'### {title}', '', '| item | ' + ' | '.join(cli.splits) + ' |', '|---|' + '---|' * len(cli.splits)]
    for label, fn, null in rows:
        cells = []
        for s in cli.splits:
            try:
                v = fn(D[s])
            except (KeyError, TypeError, IndexError):
                v = None
            if isinstance(v, str):
                txt, m, lo, hi = v, None, None, None
            else:
                txt, m, lo, hi = cell(v, null)
            cells.append(txt)
            flat.append({'chapter': title, 'item': label, 'split': s, 'text': txt, 'mean': m, 'ci_lo': lo, 'ci_hi': hi})
        md.append(f'| {label} | ' + ' | '.join(cells) + ' |')
    md.append('')
open(cli.out + '.md', 'w').write('\n'.join(md))
with open(cli.out + '.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(flat[0]))
    w.writeheader()
    w.writerows(flat)
json.dump(flat, open(cli.out + '.json', 'w'), indent=1, ensure_ascii=False)
print('\n'.join(md))
print('saved', cli.out + '.{md,csv,json}')
