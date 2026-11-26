"""Sparse-KG accuracy of the seed-0 control keys against the ungated baseline and the text keys. CPU only; run from DrugBank/:

    python ../analysis/control_keys_sparse_kg.py

Accuracy on the test sample with the full KG, without the CYP edges, and with 50 / 90 / 100% of the emerging
drug's KG edges removed, for baseline, id, random, name, llm, desc and llmperm (best; llmperm also last),
gate on and off. Paired differences (95% bootstrap over emerging drugs) between every control and the
baseline, and between the text keys and every control. Writes results/S1_1/control_keys_sparse_kg.json.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ablation_common as ac  # noqa: E402

ac.DATASET = 'S1_1'
RUN = os.path.join(ac.ROOT, 'runs', 'S1_1')
VARIANTS = ['full', 'cyp', 'new50', 'new90', 'new100']
MODELS = {'baseline': 'baseline_best', 'id': 'id-v2_best', 'random': 'random-v2_best', 'name': 'name-v2_best', 'llm': 'llm-v2_best',
          'desc': 'desc-v2_best', 'llmperm': 'llmperm-v2_best', 'llmperm_last': 'llmperm-v2_last'}


def norm(q):
    q = np.clip(q.astype(np.float64), 1e-7, 1)
    return q / q.sum(1, keepdims=True)


def load(model, v, cond):
    stem = MODELS[model]
    f = f'{RUN}/gate_interventions/{stem}_probs.npz' if (v == 'full' and model != 'baseline') else f'{RUN}/kg_ablation/{stem}__{v}.npz'
    if not os.path.exists(f):
        return None
    z = np.load(f)
    return (z['rows'], z['weight'].astype(np.float64), norm(z[cond])) if cond in z else None


train, test = ac.load_ddi('train'), ac.load_ddi('test')
rows, w, _ = load('baseline', 'full', 'original')
y = test[rows, 2]
M = ac.boot_multiplicity(ac.emerging_drug(test[rows], train), 1000) * w.astype(np.float32)
hit = {}
for m in MODELS:
    for v in VARIANTS:
        for cond in (['original'] if m == 'baseline' else ['original', 'off']):
            r = load(m, v, cond)
            if r is not None:
                assert (r[0] == rows).all()
                hit[m, v, cond] = (r[2].argmax(1) == y).astype(np.float64)


def cell(x):
    d = (M @ x) / M.sum(1)
    return {'mean': float(np.average(x, weights=w)), 'ci': [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}


res = {'accuracy': {}, 'diff': {}}
print('accuracy (gate on | off)'.ljust(26) + ''.join(v.rjust(15) for v in VARIANTS))
for m in MODELS:
    line = m.ljust(26)
    for v in VARIANTS:
        a = [f'{np.average(hit[m, v, c], weights=w):.3f}' if (m, v, c) in hit else '  -  ' for c in ('original', 'off')]
        for c in ('original', 'off'):
            if (m, v, c) in hit:
                res['accuracy'][f'{m}|{v}|{c}'] = float(np.average(hit[m, v, c], weights=w))
        line += f'{a[0]}|{a[1]}'.rjust(15)
    print(line)
PAIRS = [(a, 'baseline') for a in MODELS if a != 'baseline'] + [(t, c) for t in ('llm', 'name', 'desc') for c in ('id', 'random', 'llmperm', 'llmperm_last')]
print('\npaired accuracy difference, gate on (95% interval; * excludes 0)')
print('a - b'.ljust(26) + ''.join(v.rjust(28) for v in ('cyp', 'new90', 'new100')))
for a, b in PAIRS:
    line = f'{a} - {b}'.ljust(26)
    for v in ('cyp', 'new90', 'new100'):
        if (a, v, 'original') not in hit or (b, v, 'original') not in hit:
            line += '-'.rjust(28)
            continue
        x = hit[a, v, 'original'] - hit[b, v, 'original']
        if v == 'cyp':   # change caused by the deletion, not the level
            x = x - (hit[a, 'full', 'original'] - hit[b, 'full', 'original'])
        c = cell(x)
        res['diff'][f'{a}-{b}|{v}'] = c
        line += (f'{c["mean"]:+.3f} [{c["ci"][0]:+.3f},{c["ci"][1]:+.3f}]' + ('*' if c['ci'][0] > 0 or c['ci'][1] < 0 else ' ')).rjust(28)
    print(line)
print("\n('cyp' column: difference in the accuracy change caused by removing the CYP edges; positive = a drops less than b)")
out = os.path.join(ac.ROOT, 'results', 'S1_1', 'control_keys_sparse_kg.json')
json.dump(res, open(out, 'w'), indent=1)
print('saved', out)
