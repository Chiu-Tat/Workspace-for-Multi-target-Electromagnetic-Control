#!/usr/bin/env python3
# Run from the repository root:
#   .venv/bin/python -B -m revised_case_studies.cs1_cylinder_interaction.pilot_tfw3d_refine \
#       --layout single --variant V1 --workers 14
"""Pilot: which refinement sharpens the 3-D (x, y, beta) TFW?

Classifies one heading band, beta in [90, 95.625] deg (one level of the full
3-D run), with the same affine-hybrid classifier as ``workspace_hybrid`` and
four stopping widths:

  V0  3.125 mm, 5.625 deg   (the full 3-D run; should reproduce it)
  V1  3.125 mm, 1.406 deg   (finer heading only)
  V2  0.781 mm, 5.625 deg   (finer position only, as the 2-D maps)
  V3  0.781 mm, 1.406 deg   (both)

Every certified box holds for all headings in its own beta interval, so V1 and
V3 certify the band level by level (4 sub-levels).  The TFW-in share is
reported per sub-level and as the share certified on the whole band.
"""
from __future__ import annotations

import os
for _v in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_v] = '1'

import argparse
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from cases.case_6_interaction_obstacle_avoidance import workspace_hybrid as W
from emns.numerics.interval import interval
from revised_case_studies.shared.paths import out_dir

OUT = out_dir('cs1') / 'tfw3d_pilot'
LIM = 0.05
B0, B1 = math.pi / 2, math.pi / 2 + 2 * math.pi / 64
VARIANTS = {'V0': (0.1 / 32, (B1 - B0)), 'V1': (0.1 / 32, (B1 - B0) / 4),
            'V2': (0.1 / 128, (B1 - B0)), 'V3': (0.1 / 128, (B1 - B0) / 4)}


def subtree(task):
    """Branch and prune one initial box; returns classified leaf boxes."""
    box0, layout, exy, eb = task
    I, b, v, lo, hi = W._task_setup()
    fixed = W.LAYOUTS[layout]
    out = {'inside': [], 'outside': [], 'undetermined': []}
    stack = [box0]
    n = 0
    while stack:
        x0, x1, y0, y1, b0, b1 = stack.pop()
        n += 1
        q = [[interval[x0, x1]], [interval[y0, y1]], [interval[b0, b1]]]
        try:
            verdict, _ = W.classify_box_hybrid(q, b, I, v, lo, hi, fixed=fixed)
        except (ValueError, ZeroDivisionError, OverflowError, FloatingPointError):
            verdict = 'undetermined'
        split_xy = x1 - x0 > exy * (1 + 1e-9)
        split_b = b1 - b0 > eb * (1 + 1e-9)
        if verdict != 'undetermined' or not (split_xy or split_b):
            out[verdict].append([x0, x1, y0, y1, b0, b1])
            continue
        xs = [(x0, (x0 + x1) / 2), ((x0 + x1) / 2, x1)] if split_xy else [(x0, x1)]
        ys = [(y0, (y0 + y1) / 2), ((y0 + y1) / 2, y1)] if split_xy else [(y0, y1)]
        bs = [(b0, (b0 + b1) / 2), ((b0 + b1) / 2, b1)] if split_b else [(b0, b1)]
        stack += [(a, c, d, e, f, g) for a, c in xs for d, e in ys for f, g in bs]
    return out, n


def shares(boxes, eb):
    """TFW-in share of the 100 x 100 mm square for each beta sub-level."""
    n = max(1, round((B1 - B0) / eb))
    res = []
    for k in range(n):
        mid = B0 + (k + .5) * (B1 - B0) / n
        a = sum((q[1] - q[0]) * (q[3] - q[2]) for q in boxes if q[4] <= mid < q[5])
        res.append(100 * a / (2 * LIM) ** 2)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--layout', default='single', choices=sorted(W.LAYOUTS))
    ap.add_argument('--variant', default='V1', choices=sorted(VARIANTS))
    ap.add_argument('--workers', type=int, default=14)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    exy, eb = VARIANTS[args.variant]
    edges = [-LIM + k * 2 * LIM / 16 for k in range(17)]
    tasks = [((xa, xb, ya, yb, B0, B1), args.layout, exy, eb)
             for xa, xb in zip(edges[:-1], edges[1:]) for ya, yb in zip(edges[:-1], edges[1:])]
    t0 = time.monotonic()
    boxes = {'inside': [], 'outside': [], 'undetermined': []}
    total = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(subtree, t) for t in tasks]
        for k, f in enumerate(as_completed(futs), 1):
            out, n = f.result()
            total += n
            for key in boxes:
                boxes[key] += out[key]
            if k % 32 == 0:
                print(f'{k}/{len(tasks)} subtrees  {total} boxes  {time.monotonic() - t0:.0f} s',
                      flush=True)
    sh = {k: shares(boxes[k], eb) for k in boxes}
    band_in = 100 * sum((q[1] - q[0]) * (q[3] - q[2]) * (q[5] - q[4]) for q in boxes['inside']) \
        / ((2 * LIM) ** 2 * (B1 - B0))
    res = {'layout': args.layout, 'variant': args.variant, 'stop_xy_m': exy, 'stop_beta_rad': eb,
           'band_rad': [B0, B1], 'boxes_tested': total, 'elapsed_s': time.monotonic() - t0,
           'workers': args.workers, 'inside_share_per_sublevel_pct': sh['inside'],
           'outside_share_per_sublevel_pct': sh['outside'], 'inside_band_mean_pct': band_in,
           'boxes': boxes}
    path = OUT / f'pilot_{args.layout}_{args.variant}.json'
    path.write_text(json.dumps(res))
    print(f'{args.variant}: in {band_in:.1f}% (sub-levels '
          + ', '.join(f'{v:.1f}' for v in sh['inside']) + f'),  {total} boxes, '
          f'{res["elapsed_s"]:.0f} s  -> {path}', flush=True)


if __name__ == '__main__':
    main()
