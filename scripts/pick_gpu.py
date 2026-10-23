#!/usr/bin/env python3
"""Pick an idle GPU on the shared server.

Samples nvidia-smi several times and prints the index of the least-loaded GPU
whose average utilization and memory use are under the thresholds.
Exits with code 1 (and prints nothing on stdout) when no GPU qualifies.

    python3 scripts/pick_gpu.py                 # best idle GPU
    python3 scripts/pick_gpu.py --show          # table of all GPUs
    python3 scripts/pick_gpu.py --max-util 50 --min-free-gb 20
"""
import argparse
import subprocess
import sys
import time


def sample():
    out = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used,memory.total',
         '--format=csv,noheader,nounits'], text=True)
    rows = {}
    for line in out.strip().splitlines():
        idx, util, used, total = [int(x) for x in line.split(',')]
        rows[idx] = (util, used, total)
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--max-util', type=float, default=20, help='max average utilization (%%)')
    p.add_argument('--min-free-gb', type=float, default=30, help='min free memory (GB)')
    p.add_argument('--samples', type=int, default=5)
    p.add_argument('--show', action='store_true')
    p.add_argument('--exclude', default='', help='comma-separated GPU indices to skip (e.g. ones our runs already use)')
    a = p.parse_args()

    acc = {}
    for i in range(a.samples):
        for idx, (util, used, total) in sample().items():
            u, m, _ = acc.get(idx, (0, 0, total))
            acc[idx] = (u + util, max(m, used), total)
        if i < a.samples - 1:
            time.sleep(1)

    stats = {idx: (u / a.samples, m / 1024, t / 1024) for idx, (u, m, t) in acc.items()}
    excluded = {int(x) for x in a.exclude.split(',') if x.strip()}
    ok = [idx for idx, (u, m, t) in stats.items()
          if u <= a.max_util and t - m >= a.min_free_gb and idx not in excluded]
    if a.show:
        for idx, (u, m, t) in sorted(stats.items()):
            mark = '*' if idx in ok else ' '
            print(f'{mark} GPU {idx}: util {u:5.1f}%  mem {m:5.1f}/{t:.0f} GB', file=sys.stderr)
    if not ok:
        sys.exit(1)
    print(min(ok, key=lambda idx: (stats[idx][0], stats[idx][1])))


if __name__ == '__main__':
    main()
