"""Side-by-side key numbers of the progress report for several splits (folds).

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

CH.append(('0. 성능 (test macro-F1, epoch 10-20 평균) / 검증 최적 epoch', [
    (f'{m} F1 (ep10-20 평균)', (lambda m: lambda d: g(d['cv'][m], 'f1_ep10_20'))(m), 0) for m in MODELS] + [
    (f'{m} 최적 epoch / test acc', (lambda m: lambda d: None if d['cv'][m] is None else
                                  f"{d['cv'][m]['best_epoch']} / {d['cv'][m]['best_test_acc']:.3f}")(m), 0) for m in MODELS]))

CH.append(('1. 원 구조는 엔티티를 고를 수 없다 (relation_ties.json)', [
    ('테스트 쌍', lambda d: g(d['rt'], '2hop', 'all', 'pairs'), 0),
    ('2-hop 경로가 있는 쌍', lambda d: g(d['rt'], '2hop', 'all', 'pairs_with_paths'), 0),
    ('2-hop 동점 비율 (전체)', lambda d: g(d['rt'], '2hop', 'all', 'tied_share_mean'), 0),
    ('2-hop 동점 비율 (대사형)', lambda d: g(d['rt'], '2hop', 'metabolism', 'tied_share_mean'), 0),
    ('2-hop 동점 비율 (비PK)', lambda d: g(d['rt'], '2hop', 'non_pk', 'tied_share_mean'), 0),
    ('2-hop 동점 크기 중앙값', lambda d: g(d['rt'], '2hop', 'all', 'tie_size_q', 1) if False else
        (None if g(d['rt'], '2hop', 'all', 'tie_size_q') is None else g(d['rt'], '2hop', 'all', 'tie_size_q')[1]), 0),
    ('2-hop 집어낼 확률', lambda d: g(d['rt'], '2hop', 'all', 'single_out_chance_mean'), 0),
    ('3-hop 동점 비율', lambda d: g(d['rt'], '3hop', 'all', 'tied_share_mean'), 0),
    ('3-hop 집어낼 확률', lambda d: g(d['rt'], '3hop', 'all', 'single_out_chance_mean'), 0),
    ('대사형 중 결합 유전자 공유', lambda d: g(d['rt'], 'binds_binds', 'metabolism', 'pairs_sharing_a_bound_gene'), 0),
    ('대사형 중 CYP 공유', lambda d: g(d['rt'], 'binds_binds', 'metabolism', 'pairs_sharing_a_cyp'), 0),
    ('비PK 중 CYP 공유', lambda d: g(d['rt'], 'binds_binds', 'non_pk', 'pairs_sharing_a_cyp'), 0),
    ('동점 유전자 중 CYP 비율 (대사형)', lambda d: g(d['rt'], 'binds_binds', 'metabolism', 'cyp_share_of_tied_genes_mean'), 0),
    ('특정 CYP를 집어낼 확률 (대사형)', lambda d: g(d['rt'], 'binds_binds', 'metabolism', 'chance_to_name_one_specific_cyp_mean'), 0),
]))

STRATA = [('전체', 'all'), ('대사', 'mech=metabolism'), ('기타 PK', 'mech=pk_other'), ('PD', 'mech=pd'),
          ('불특정', 'mech=unspecified'), ('CYP 공유', 'mediator=cyp'), ('공유 없음', 'mediator=none'),
          ('KG 적음', 'kg=low'), ('KG 중간', 'kg=mid'), ('KG 많음', 'kg=high'), ('대사 & CYP', 'metabolism & cyp'),
          ('PD & CYP', 'pd & cyp'), ('PD & 없음', 'pd & none')]
CH.append(('2-2. 층별 정확도 / 확신도 (게이트 끈 llm·name 평균, 표본)', [
    (f'{lab} 정확도 / 확신도', (lambda k: lambda d: None if g(d['st'], k) is None else
                          f"{d['st'][k]['off_acc']:.3f} / {d['st'][k]['off_conf']:.3f} (n={d['st'][k]['n_pairs']})")(k), 0)
    for lab, k in STRATA]))

PULL = [('대사 & CYP', 'metabolism&shared_cyp'), ('기타 PK & CYP', 'pk_other&shared_cyp'),
        ('기타 PK & 없음', 'pk_other&no_shared_cyp'), ('PD & CYP', 'pd&shared_cyp'), ('PD & 없음', 'pd&no_shared_cyp')]
CH.append(('2-3. 대사형으로 예측되는 비율 (전체 KG)', [
    (f'{lab}: {m}', (lambda k, m: lambda d: g(d['kg'], 'f4', f'{m}/original', 'pull', k, 'full', 'mean'))(k, m), 0)
    for lab, k in PULL for m in MODELS]))

rows = []
for m in MODELS:
    rows += [(f'{m}: 정확도 전체 KG', (lambda m: lambda d: g(d['kg'], 'accuracy', f'{m}/original', 'full'))(m), 0),
             (f'{m}: 정확도 CYP 지움', (lambda m: lambda d: g(d['kg'], 'accuracy', f'{m}/original', 'cyp'))(m), 0),
             (f'{m}: 정확도 대조 지움 (hub / rand0 / rand1)', (lambda m: lambda d: None if g(d['kg'], 'accuracy', f'{m}/original') is None
                else ' / '.join(f"{d['kg']['accuracy'][f'{m}/original'][v]:.3f}" for v in ('hub', 'rand0', 'rand1')))(m), 0),
             (f'{m}: 대사형 dlogp (CYP 지움)', (lambda m: lambda d: g(d['kg'], 'f4', f'{m}/original', 'dlogp_cyp', 'metabolism'))(m), 0),
             (f'{m}: PD dlogp', (lambda m: lambda d: g(d['kg'], 'f4', f'{m}/original', 'dlogp_cyp', 'pd'))(m), 0),
             (f'{m}: 기타 PK dlogp', (lambda m: lambda d: g(d['kg'], 'f4', f'{m}/original', 'dlogp_cyp', 'pk_other'))(m), 0),
             (f'{m}: CYP 공유 쌍 대사형 예측 비율 전 -> 후', (lambda m: lambda d: None if g(d['kg'], 'f4', f'{m}/original', 'shared_cyp_metab_pred') is None
                else '{:.2f} -> {:.2f}'.format(*(d['kg']['f4'][f'{m}/original']['shared_cyp_metab_pred'][v]['rate']['mean'] for v in ('full', 'cyp'))))(m), 0),
             (f'{m}: 그 예측의 정밀도 전 -> 후', (lambda m: lambda d: None if g(d['kg'], 'f4', f'{m}/original', 'shared_cyp_metab_pred') is None
                else '{:.2f} -> {:.2f}'.format(*(d['kg']['f4'][f'{m}/original']['shared_cyp_metab_pred'][v]['precision']['mean'] for v in ('full', 'cyp'))))(m), 0),
             (f'{m}: 공유 수용체를 잃은 PD 쌍 dlogp (hub)', (lambda m: lambda d: g(d['kg'], 'f4', f'{m}/original', 'by_lost_gene', 'hub/pd/lost'))(m), 0)]
for m in ('llm', 'name'):
    rows += [(f'{m}: 대사형 dlogp, 게이트 끈 모델', (lambda m: lambda d: g(d['kg'], 'f4', f'{m}/off', 'dlogp_cyp', 'metabolism'))(m), 0),
             (f'{m}: 대사형 dlogp 게이트 on - off', (lambda m: lambda d: g(d['kg'], 'f4', f'{m}/original', 'gate_original-off', 'metabolism'))(m), 0)]
CH.append(('2-4. CYP 엣지를 지우면 (F4)', rows))

CH.append(('2-5. 새 약 KG를 지우면 (M1): 정확도', [
    (f'{m}: 0% / 50% / 90% / 100%', (lambda m: lambda d: None if g(d['kg'], 'm1', f'{m}/original', 'raw') is None else
        ' / '.join(f"{d['kg']['m1'][f'{m}/original']['raw'][lv]['acc']:.3f}" for lv in ('full', 'new50', 'new90', 'new100')))(m), 0)
    for m in MODELS] + [
    ('KG 없는 기준선 (기존 약의 최빈 타입)', lambda d: g(d['kg'], 'm1_reference', 'existing_drug_top_type', 'mean'), 0),
    ('llm - name 정확도 하락 차이 (90%)', lambda d: g(d['kg'], 'm1_key_vs_key', 'new90/llm-name'), 0)]))


def tert(d, m, kind, stat):
    t = g(d['kc'], 'models', m, 'original', f'{kind}_shared_neighbours')
    return None if t is None else ' / '.join(f"{t[b][stat]:.3f}" for b in ('low', 'mid', 'high'))


rows = [('공유 이웃 구간 경계 a, b (np.digitize: 적음 < a <= 중간 < b <= 많음)', lambda d: None if g(d['kc'], 'cut_points') is None else
         '{:.0f}, {:.0f}'.format(*d['kc']['cut_points']['shared_neighbours']), 0)]
for m in ('llm', 'name'):
    rows += [(f'{m}: 정확도 적음 / 중간 / 많음', (lambda m: lambda d: tert(d, m, 'raw', 'acc'))(m), 0),
             (f'{m}: 확신도 적음 / 중간 / 많음', (lambda m: lambda d: tert(d, m, 'raw', 'conf'))(m), 0),
             (f'{m}: 정확도 차이 (적음 - 많음)', (lambda m: lambda d: g(d['kc'], 'models', m, 'original', 'raw_shared_neighbours', 'low-high', 'acc'))(m), 0),
             (f'{m}: 과신(확신도-정확도) 차이 (적음 - 많음), 보정 전', (lambda m: lambda d: g(d['kc'], 'models', m, 'original', 'raw_shared_neighbours', 'low-high', 'gap'))(m), 0),
             (f'{m}: 같은 값, temperature scaling 후', (lambda m: lambda d: g(d['kc'], 'models', m, 'original', 'temp_shared_neighbours', 'low-high', 'gap'))(m), 0)]
CH.append(('2-6. 공유 이웃이 많을수록 과신하는가 (kg_calibration.json, 게이트 모델)', rows))

rows = []
for m in ('llm', 'name'):
    rows.append((f'{m}: 정확도 original / off / mean / shuffle_all / shuffle_type / key_perm',
                 (lambda m: lambda d: None if d['gi'][m] is None else ' / '.join(
                     f"{d['gi'][m]['summary'][c]['acc']:.3f}" for c in ('original', 'off', 'mean', 'shuffle_all', 'shuffle_type', 'key_perm')))(m), 0))
for m in ('llm', 'name'):
    rows += [(f'{m}: F2 key_perm dlogp (대사형)', (lambda m: lambda d: g(d['gi'][m], 'summary', 'key_perm', 'metabolism'))(m), 0),
             (f'{m}: F2 key_perm dlogp (비PK)', (lambda m: lambda d: g(d['gi'][m], 'summary', 'key_perm', 'non_pk'))(m), 0),
             (f'{m}: key_perm / off 때 예측이 바뀐 비율 (대사형)', (lambda m: lambda d: None if d['gi'][m] is None else
                 '{:.1%} / {:.1%}'.format(d['gi'][m]['summary']['key_perm']['metabolism']['flip'], d['gi'][m]['summary']['off']['metabolism']['flip']))(m), 0),
             (f'{m}: key_perm - off (대사형)', (lambda m: lambda d: g(d['fu'], m, 'metabolism', 'key_perm-off'))(m), 0),
             (f'{m}: shuffle_all - off (대사형)', (lambda m: lambda d: g(d['fu'], m, 'metabolism', 'shuffle_all-off'))(m), 0),
             (f'{m}: mean - off (대사형)', (lambda m: lambda d: g(d['fu'], m, 'metabolism', 'mean-off'))(m), 0),
             (f'{m}: F1 KL(같은 타입) - KL(무작위) (대사형)', (lambda m: lambda d: g(d['fu'], m, 'metabolism', 'kl_type-all'))(m), 0),
             (f'{m}: F1 (비PK)', (lambda m: lambda d: g(d['fu'], m, 'non_pk', 'kl_type-all'))(m), 0),
             (f'{m}: key_perm, 대사형 & CYP 공유', (lambda m: lambda d: g(d['fu'], m, 'metabolism', 'key_perm-original_with_shared_cyp'))(m), 0),
             (f'{m}: key_perm, 비PK & CYP 공유', (lambda m: lambda d: g(d['fu'], m, 'non_pk', 'key_perm-original_with_shared_cyp'))(m), 0),
             (f'{m}: key_perm, 타입 46', (lambda m: lambda d: g(d['fu'], m, 'type_46', 'key_perm'))(m), 0),
             (f'{m}: 게이트 이득 NLL(off)-NLL(original), 대사 & CYP', (lambda m: lambda d: g(d['st'], 'metabolism & cyp', f'gate_gain/{m}'))(m), 0),
             (f'{m}: 게이트 이득, 전체', (lambda m: lambda d: g(d['st'], 'all', f'gate_gain/{m}'))(m), 0)]


def n_sig_gain(d, m):
    if d['st'] is None:
        return None
    vals = [v.get(f'gate_gain/{m}') for v in d['st'].values() if isinstance(v, dict) and v.get(f'gate_gain/{m}')]
    up = sum(1 for v in vals if v['ci'][0] > 0)
    down = sum(1 for v in vals if v['ci'][1] < 0)
    return f'{up + down} / {len(vals)} (이득 {up}, 손해 {down})'


rows += [(f'{m}: 게이트 이득이 유의한 층 수', (lambda m: lambda d: n_sig_gain(d, m))(m), 0) for m in ('llm', 'name')]
CH.append(('3. 게이트는 그 자리에서만 엔티티 정보를 쓴다 (F1 + F2)', rows))

rows = []
for name, k in [('원 EmerGNN (동점)', 'EmerGNN (ties)'), ('연결 수 순', 'degree heuristic'),
                ('llm best', 'llm-v2 [best]'), ('name best', 'name-v2 [best]')]:
    for grp, lab in [('metabolism', '대사형'), ('non_pk', '비PK'), ('gap', '차이')]:
        rows.append((f'{name}: AUC {lab}', (lambda k, grp: lambda d: g(d['gd'], k, 'e2', grp, 'auc'))(k, grp), 0 if grp == 'gap' else 0.5))
    rows.append((f'{name}: 연결 수가 불리한 AUC 대사형', (lambda k: lambda d: g(d['gd'], k, 'e2', 'metabolism', 'adverse_auc'))(k), 0.5))
for m in ('llm', 'name'):
    k = f'{m}-v2 [best]'
    rows += [(f'{m}: 게이트와 연결 수의 순위 상관, 절댓값 최대 (유전자)', (lambda k: lambda d: None if g(d['gd'], k, 'e4') is None else max(
                abs(v) for n, v in d['gd'][k]['e4'].items() if n.endswith('_genes') and isinstance(v, (int, float))))(k), 0),
             (f'{m}: 연결 수 통제 후 CYP 게이트 상승', (lambda k: lambda d: None if g(d['gd'], k, 'e4', 'shift_regression') is None else
                {'mean': d['gd'][k]['e4']['shift_regression']['cyp_coef'], 'ci': d['gd'][k]['e4']['shift_regression']['cyp_coef_ci']})(k), 0)]
CH.append(('4. 게이트는 공유 CYP를 짚는가 (E2, E4)', rows))

CH.append(('5. llm과 name의 차이', [
    ('AUC 대사형 llm - name', lambda d: None if d['gd'] is None else
        d['gd']['llm-v2 [best]']['e2']['metabolism']['auc']['mean'] - d['gd']['name-v2 [best]']['e2']['metabolism']['auc']['mean'], 0),
    ('AUC 비PK llm - name', lambda d: None if d['gd'] is None else
        d['gd']['llm-v2 [best]']['e2']['non_pk']['auc']['mean'] - d['gd']['name-v2 [best]']['e2']['non_pk']['auc']['mean'], 0)] + [
    (f'NLL(name) - NLL(llm), {lab}', (lambda k: lambda d: g(d['st'], k, 'nll_name-llm'))(k), 0)
    for lab, k in [('전체', 'all'), ('대사', 'mech=metabolism'), ('기타 PK', 'mech=pk_other'), ('PD', 'mech=pd'),
                   ('불특정', 'mech=unspecified')]]))

md, flat = [], []
for title, rows in CH:
    md += [f'### {title}', '', '| 항목 | ' + ' | '.join(cli.splits) + ' |', '|---|' + '---|' * len(cli.splits)]
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
