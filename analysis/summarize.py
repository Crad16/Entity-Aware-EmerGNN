#!/usr/bin/env python3
"""Summarize EmerGNN run logs: best epoch picked by the primary validation metric (never by test).

Primary metric: PR-AUC for TWOSIDES, macro F1 for DrugBank.

    python3 scripts/summarize.py runs/ts_S1_1_*.log
    python3 scripts/summarize.py --curves runs/ts_S1_1_*.log
"""
import argparse
import os
import re

# rows are (epoch, v1, v2, v3, t1, t2, t3); the primary metric is v2/t2 for TWOSIDES and v1/t1 for DrugBank
TWOSIDES = re.compile(r'^epoch:(\d+)\t.*\[Valid\] ROC-AUC:([\d.]+) PR-AUC:([\d.]+) AP:([\d.]+)\t \[Test\] ROC-AUC:([\d.]+) PR-AUC:([\d.]+) AP:([\d.]+)')
DRUGBANK = re.compile(r'^epoch:(\d+)\t.*\[Valid\] f1:([\d.]+) acc:([\d.]+) kap:([\d.]+)\t\[Test\] f1:([\d.]+) acc:([\d.]+) kap:([\d.]+)')


def parse(path, max_epoch=float('inf')):
    text = open(path, errors='ignore').read().replace('\r', '\n')
    status = 'done' if 'Best results' in text else ('CRASHED' if 'Traceback' in text else 'running')
    for pattern, key in [(TWOSIDES, 2), (DRUGBANK, 1)]:
        # with --save_model an evaluation line is printed twice; keep one row per epoch
        by_epoch = {}
        for m in filter(None, map(pattern.match, text.splitlines())):
            row = tuple(float(x) for x in m.groups())
            by_epoch[row[0]] = row
        rows = [by_epoch[e] for e in sorted(by_epoch) if e <= max_epoch]
        if rows:
            return rows, status, key
    return [], status, 2


def main():
    p = argparse.ArgumentParser()
    p.add_argument('logs', nargs='+')
    p.add_argument('--curves', action='store_true')
    p.add_argument('--max_epoch', type=int, default=10 ** 9, help='only use evaluations up to this epoch')
    a = p.parse_args()

    print(f'{"run":32s} {"status":8s} {"#eval":>5s} {"best_ep":>7s} {"V-main":>6s} | test metrics')
    for path in a.logs:
        rows, status, key = parse(path, a.max_epoch)
        name = os.path.basename(path).removesuffix('.log')
        if not rows:
            print(f'{name:32s} {status:8s} {0:5d}')
            continue
        names = ('ROC', 'PR', 'AP') if key == 2 else ('F1', 'Acc', 'Kappa')
        # first epoch reaching the max primary valid metric, same rule as evaluate.py (strict >)
        best = max(rows, key=lambda r: (r[key], -r[0]))
        tests = '  '.join(f'{n} {v:.4f}' for n, v in zip(names, best[4:]))
        print(f'{name:32s} {status:8s} {len(rows):5d} {int(best[0]):7d} {best[key]:.4f} | {tests}')
        if a.curves:
            for r in rows:
                print(f'    ep {int(r[0]):3d}  valid {names[key - 1]} {r[key]:.4f}  test {names[key - 1]} {r[key + 3]:.4f}')


if __name__ == '__main__':
    main()
