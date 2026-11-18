#!/usr/bin/env python3
"""Decision 1, Stage A: matched comparison of L = 1, 3, 5 multipole offsets.

Run from the repository root:
    .venv/bin/python -B dipole_error_comparison/stageA_compare.py [--n 31] [--workers 8]

For each admissible pose (centre separation above the non-overlap radius, which
equals the sum of the two bounding-sphere radii), the script evaluates

  * the point-dipole offset of the Case 6 model (sanity check for L = 1),
  * the compact multipole offsets for L = 1, 3, 5, 7 (multipole_model.py),
  * the reference: the exact surface-charge model of both uniformly magnetized
    cylinders with Gauss quadrature (charge_reference.py, 8 x 32 x 8 nodes;
    refinement to 16 x 64 x 16 changes outputs by < 1e-6 mT or mN),
  * the earlier point-element finite-body quadrature (4 x 16 x 5 and 6 x 24 x 8),

and the pointwise TFW containment test of run_comparison.contains_box for
L = 1, 3, 5, 7 and the reference, with the same coil map, task, and
current limit.  Results: results/stageA_compare.json (+ a printed summary).
These are sampled numerical comparisons, not continuous bounds.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(HERE))
from cases.case_6_interaction_obstacle_avoidance.simulation import model  # noqa: E402
from run_comparison import finite_interaction, dipole_interaction, contains_box  # noqa: E402
import multipole_model as mp  # noqa: E402
import charge_reference as cr  # noqa: E402

CFG = json.loads((HERE / 'config.json').read_text())
FINE = json.loads(json.dumps(CFG)); FINE['reference_quadrature'] = {'radial': 6, 'azimuthal': 24, 'axial': 8}
KW = dict(free_m=model.FREE_M, fixed_m=model.FIXED_M, fixed_xy=(model.FIXED_X, model.FIXED_Y),
          fixed_beta=model.FIXED_BETA, moving_cyl=CFG['moving_cylinder'], fixed_cyl=CFG['fixed_cylinder'])
GM, GF = CFG['moving_cylinder'], CFG['fixed_cylinder']
R_CONV = mp.bounding_radius(GM['length_m'], GM['diameter_m']) + mp.bounding_radius(GF['length_m'], GF['diameter_m'])
CENTER = np.array([0.0, model.B_HOLD, 0.0, 0.0])
HALF = np.array([0.0, 0.0, 0.0, model.F_MAX])
BINS = [(R_CONV, 0.015), (0.015, 0.020), (0.020, 0.030), (0.030, 0.050), (0.050, 1.0)]
HEADINGS = (0, 45, 90, 135)


def pose(task):
    x, y, deg = task
    beta = math.radians(deg)
    A = model.actuation_matrix_body(x, y, beta)
    Acoil, cdip = A[:, :model.NUM_COILS], A[:, model.NUM_COILS]
    t0 = time.perf_counter(); c = {L: mp.multipole_interaction(x, y, beta, L, **KW) for L in (1, 3, 5, 7)}
    t_mp = time.perf_counter() - t0
    t0 = time.perf_counter(); coarse = finite_interaction(x, y, beta, CFG); t_fb = time.perf_counter() - t0
    fine = finite_interaction(x, y, beta, FINE)
    t0 = time.perf_counter()
    ref = cr.charge_interaction(x, y, beta, z_plane=model.Z_PLANE, nr=8, nphi=32, nz=8, **KW)
    t_ref = time.perf_counter() - t0
    lim = CFG['current_limit_A']
    cls = {f'L{L}': contains_box(Acoil, CENTER - c[L], HALF, lim) for L in (1, 3, 5, 7)}
    cls['REF'] = contains_box(Acoil, CENTER - ref, HALF, lim)
    return {'x': x, 'y': y, 'deg': deg, 'sep': math.hypot(x - model.FIXED_X, y - model.FIXED_Y),
            'dip': cdip.tolist(), **{f'L{L}': c[L].tolist() for L in c},
            'fb_coarse': coarse.tolist(), 'fb_fine': fine.tolist(), 'ref': ref.tolist(), 'cls': cls,
            't_multipole_s': t_mp, 't_fb_coarse_s': t_fb, 't_ref_s': t_ref}


def summary(rows):
    lab = ['b_y (mT)', 'b_z (mT)', 'f_y (mN)', 'f_z (mN)']
    out = {}
    for lo, hi in BINS:
        sel = [r for r in rows if lo <= r['sep'] < hi]
        if not sel:
            continue
        ref = np.array([r['ref'] for r in sel])
        key = f'{lo*1e3:.1f}-{min(hi,0.2)*1e3:.0f} mm'
        d = {'samples': len(sel)}
        for name in ('L1', 'L3', 'L5', 'L7', 'fb_coarse', 'fb_fine'):
            e = np.abs(np.array([r[name] for r in sel]) - ref) * 1e3     # T->mT, N->mN
            d[name] = {'max': dict(zip(lab, e.max(0).round(4).tolist())),
                       'p95': dict(zip(lab, np.percentile(e, 95, axis=0).round(4).tolist()))}
        d['class_disagree_vs_REF'] = {f'L{L}': sum(r['cls'][f'L{L}'] != r['cls']['REF'] for r in sel) for L in (1, 3, 5, 7)}
        d['ref_inside'] = sum(r['cls']['REF'] for r in sel)
        out[key] = d
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=31)
    ap.add_argument('--workers', type=int, default=8)
    args = ap.parse_args()
    xs = np.linspace(*CFG['x_limits_m'], args.n); ys = np.linspace(*CFG['y_limits_m'], args.n)
    tasks = [(float(x), float(y), d) for d in HEADINGS for y in ys for x in xs
             if math.hypot(x - model.FIXED_X, y - model.FIXED_Y) >= R_CONV]
    print(f'{len(tasks)} admissible poses, convergence radius {R_CONV*1e3:.2f} mm', flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(pose, tasks, chunksize=8))
    l1_vs_dip = max(float(np.max(np.abs(np.array(r['L1']) - np.array(r['dip'])))) for r in rows)
    res = {'scope': 'Sampled numerical comparison on a grid; not continuous bounds.',
           'grid_n': args.n, 'headings_deg': HEADINGS, 'convergence_radius_m': R_CONV,
           'reference': 'exact surface-charge model, Gauss 8x32x8 (charge_reference.py); point-element quadrature 4x16x5 (fb_coarse) and 6x24x8 (fb_fine) shown for comparison',
           'max_abs_L1_minus_dipole': l1_vs_dip,
           'mean_time_s': {'multipole_L1_L3_L5_L7': float(np.mean([r['t_multipole_s'] for r in rows])),
                           'finite_body_4x16x5': float(np.mean([r['t_fb_coarse_s'] for r in rows])),
                           'charge_reference_8x32x8': float(np.mean([r['t_ref_s'] for r in rows]))},
           'elapsed_s': time.time() - t0,
           'by_heading': {d: summary([r for r in rows if r['deg'] == d]) for d in HEADINGS},
           'all_headings': summary(rows), 'rows': rows}
    (HERE / 'results' / 'stageA_compare.json').write_text(json.dumps(res))
    print(json.dumps({k: res[k] for k in ('max_abs_L1_minus_dipole', 'mean_time_s', 'elapsed_s')}, indent=1))
    print(json.dumps(res['all_headings'], indent=1))


if __name__ == '__main__':
    main()
