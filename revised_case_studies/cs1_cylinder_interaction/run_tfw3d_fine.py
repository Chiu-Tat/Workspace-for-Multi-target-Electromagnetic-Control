#!/usr/bin/env python3
# Run from the repository root:
#   .venv/bin/python -B -m revised_case_studies.cs1_cylinder_interaction.run_tfw3d_fine \
#       --layout single --workers 14
"""Full 3-D (x, y, beta) TFW at the resolution of the 2-D maps.

Same classifier, task and layouts as ``workspace_hybrid`` (corrected dBy/dy),
with stopping widths 0.781 mm in x and y and 5.625 deg in beta (setting V2 of
``pilot_tfw3d_refine``).  The domain is split into 16 x 16 x 64 initial boxes,
one per heading level; each is refined independently.  Finished subtrees are
appended to a JSON-lines checkpoint, so an interrupted run resumes with
``--resume``.  The result has the same format as ``tfw3d_corrected``.
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
from revised_case_studies.cs1_cylinder_interaction.pilot_tfw3d_refine import subtree
from revised_case_studies.shared.paths import out_dir

OUT = out_dir('cs1') / 'tfw3d_fine'
LIM = 0.05
N_XY, N_B = 16, 64
STOP_XY, STOP_B = 0.1 / 128, 2 * math.pi / N_B


def tasks(layout):
    xe = [-LIM + k * 2 * LIM / N_XY for k in range(N_XY + 1)]
    be = [k * 2 * math.pi / N_B for k in range(N_B + 1)]
    out = []
    for kb in range(N_B):
        for i in range(N_XY):
            for j in range(N_XY):
                box = (xe[i], xe[i + 1], xe[j], xe[j + 1], be[kb], be[kb + 1])
                out.append((f'{kb}_{i}_{j}', (box, layout, STOP_XY, STOP_B)))
    return out


def run_task(item):
    key, task = item
    out, n = subtree(task)
    return key, out, n


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--layout', default='single', choices=sorted(W.LAYOUTS))
    ap.add_argument('--workers', type=int, default=14)
    ap.add_argument('--resume', action='store_true')
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    ckpt = OUT / f'tfw3d_{args.layout}.checkpoint.jsonl'
    done = {}
    if ckpt.exists():
        if not args.resume:
            raise SystemExit(f'{ckpt} exists; use --resume')
        for line in ckpt.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                done[rec['key']] = rec
    todo = [t for t in tasks(args.layout) if t[0] not in done]
    print(f'{args.layout}: {len(done)} subtrees done, {len(todo)} to go', flush=True)
    t0 = time.monotonic()
    prior = sum(r['elapsed_s'] for r in done.values()) if done else 0.0
    with ProcessPoolExecutor(max_workers=args.workers) as pool, open(ckpt, 'a') as fh:
        futs = [pool.submit(run_task, t) for t in todo]
        for k, f in enumerate(as_completed(futs), 1):
            key, out, n = f.result()
            rec = {'key': key, 'boxes': out, 'n': n, 'elapsed_s': 0.0}
            fh.write(json.dumps(rec) + '\n')
            fh.flush()
            done[key] = rec
            if k % 256 == 0 or k == len(todo):
                el = time.monotonic() - t0
                print(f'{len(done)}/{N_B * N_XY * N_XY} subtrees  {el:.0f} s  '
                      f'eta {el / k * (len(todo) - k) / 60:.0f} min', flush=True)
    keys = {'inside': 'in_workspace', 'outside': 'out_of_workspace',
            'undetermined': 'undetermined'}
    res = {v: [] for v in keys.values()}
    total = 0
    for rec in done.values():
        total += rec['n']
        for k, v in keys.items():
            res[v] += rec['boxes'][k]
    vol = lambda bs: sum((b[1] - b[0]) * (b[3] - b[2]) * (b[5] - b[4]) for b in bs) \
        / ((2 * LIM) ** 2 * 2 * math.pi)
    data = {'mode': 'affine_hybrid', 'eps': [STOP_XY, STOP_XY, STOP_B],
            'elapsed_s': prior + time.monotonic() - t0, 'iterations': total,
            'workers': args.workers, **res,
            'volume_fraction': {k: vol(v) for k, v in res.items()},
            'protocol': {'z_m': W.Z_PLANE, 'free_m': W.FREE_M, 'layout': args.layout,
                         'task': ['By=0', f'Bz={W.B_HOLD}', 'fy=0', f'fz=±{W.F_MAX}'],
                         'current_A': [W.I_MIN, W.I_MAX],
                         'note': 'Corrected dBy/dy; 0.781 mm x 5.625 deg stop; '
                                 'initial grid 16 x 16 x 64.'}}
    path = OUT / f'tfw3d_{args.layout}.json'
    path.write_text(json.dumps(data))
    vf = data['volume_fraction']
    print(f'{args.layout}: in {vf["in_workspace"] * 100:.1f}%  out '
          f'{vf["out_of_workspace"] * 100:.1f}%  unresolved {vf["undetermined"] * 100:.1f}%  '
          f'{total} boxes  -> {path}', flush=True)


if __name__ == '__main__':
    main()
