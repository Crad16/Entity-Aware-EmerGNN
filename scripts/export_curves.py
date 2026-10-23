#!/usr/bin/env python3
"""Export evaluation curves from run logs to CSV (one row per evaluated epoch).

    python scripts/export_curves.py runs/*.log --out results/curves
"""
import argparse
import csv
import os
import re

DRUGBANK = re.compile(r'^epoch:(\d+)\t.*\[Valid\] f1:([\d.]+) acc:([\d.]+) kap:([\d.]+)\t\[Test\] f1:([\d.]+) acc:([\d.]+) kap:([\d.]+)')
TWOSIDES = re.compile(r'^epoch:(\d+)\t.*\[Valid\] ROC-AUC:([\d.]+) PR-AUC:([\d.]+) AP:([\d.]+)\t \[Test\] ROC-AUC:([\d.]+) PR-AUC:([\d.]+) AP:([\d.]+)')
GATE = re.compile(r'gate: mean ([\d.]+) std ([\d.]+) min ([\d.]+) max ([\d.]+) cyp ([\d.]+) other_gene ([\d.]+)'
                  r'(?: saturated ([\d.]+) pair_share ([\d.]+))?')
HEADER = {DRUGBANK: ['valid_f1', 'valid_acc', 'valid_kappa', 'test_f1', 'test_acc', 'test_kappa'],
          TWOSIDES: ['valid_roc', 'valid_pr', 'valid_ap', 'test_roc', 'test_pr', 'test_ap']}
GATE_COLS = ['gate_mean', 'gate_std', 'gate_min', 'gate_max', 'gate_cyp', 'gate_other_gene', 'gate_saturated', 'gate_pair_share']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('logs', nargs='+')
    p.add_argument('--out', default='results/curves')
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for path in a.logs:
        lines = open(path, errors='ignore').read().replace('\r', '\n').splitlines()
        for pattern, cols in HEADER.items():
            rows = {}
            for line in lines:
                m = pattern.match(line)
                if m:
                    g = GATE.search(line)
                    rows[int(m.group(1))] = list(m.groups()[1:]) + (list(g.groups()) if g else [None] * 8)
            if rows:
                name = os.path.basename(path).removesuffix('.log')
                with open(os.path.join(a.out, name + '.csv'), 'w', newline='') as f:
                    w = csv.writer(f)
                    w.writerow(['epoch'] + cols + GATE_COLS)
                    w.writerows([e] + r for e, r in sorted(rows.items()))
                print(f'{name}: {len(rows)} evaluations')
                break


if __name__ == '__main__':
    main()
