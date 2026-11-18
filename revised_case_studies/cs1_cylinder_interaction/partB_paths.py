#!/usr/bin/env python3
# Run from the repository root:
#   python -m revised_case_studies.cs1_cylinder_interaction.partB_paths
"""Case Study 1, Part B (analysis) -- what the interaction does along a path.

Compares the two references that share start and goal (``common.PLANNED`` and
``common.NAIVE``, written by ``presentation.py``):

  planned   planned in the sharp 3-D TFW (``nonholonomic.plan``)
  straight  constant-heading straight line (interaction ignored)

For every sample the configuration is labelled inside / near boundary /
uncertified / infeasible (sharp 3-D TFW + exact pointwise test), and
the reachable total-force ranges are computed by LP at that configuration.
No TFW is computed.  This is the reviewer's "inside vs. near the boundary vs.
outside" comparison, expressed in actuation terms.

Figures (``revised_case_studies/results/cs1``):

  B1_paths_labels     both paths over the TFW slice, coloured by label
  B2_force_ranges     reachable thrust fz and lateral fy along both paths,
                      against the +-0.1 N / zero requirements (paper e, f, j, k)
  B3_interaction      fixed-magnet force on the moving magnet along both paths
  B_paths_summary.json
"""

from __future__ import annotations

import argparse
import json
import math

import matplotlib.pyplot as plt
import numpy as np

from revised_case_studies.cs1_cylinder_interaction import common as c
from revised_case_studies.shared import paths as P

CASE = 'cs1'
LABELS = ('inside', 'boundary', 'uncertified', 'infeasible')
LABEL_COLOR = {'inside': P.C_INSIDE, 'boundary': P.C_BOUNDARY,
               'uncertified': '#9a8fb5', 'infeasible': P.C_OUTSIDE}
PATHS = {'planned (in TFW)': c.PLANNED, 'straight (interaction ignored)': c.NAIVE}


def retime(traj, speed):
    """Arc-length re-timing at a common speed so both paths compare fairly."""
    xyb = traj[:, :3].copy()
    xyb[:, 2] = np.unwrap(xyb[:, 2])
    keep = np.concatenate([[True], np.linalg.norm(np.diff(xyb[:, :2], axis=0), axis=1) > 1e-9])
    xyb = xyb[keep]
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(xyb[:, :2], axis=0), axis=1))])
    t = s / speed
    return np.column_stack([xyb, t])


def analyse(traj, stride=1):
    rows = []
    for x, y, b, t in traj[::stride]:
        fr = c.force_ranges(x, y, b)
        rows.append({
            't': float(t), 'x': float(x), 'y': float(y), 'beta': float(b),
            'label': c.label(x, y, b),
            'dist_fixed': math.dist((x, y), c.FIXED_XY),
            **{k: (list(v) if isinstance(v, tuple) else v) for k, v in fr.items()},
        })
    return rows


def fig_paths(results, beta=math.pi / 2):
    t = c.tfw()
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.4), constrained_layout=True)
    for ax, (name, rows) in zip(axes, results.items()):
        ax.imshow(t.slice_at(beta), origin='lower', extent=t.extent,
                  cmap=plt.matplotlib.colors.ListedColormap(['#f3f1ec', '#cfe3d2']),
                  interpolation='nearest')
        c.draw_magnet(ax, *c.FIXED_XY, c.FIXED_BETA, P.C_FIXED)
        xs = np.array([r['x'] for r in rows])
        ys = np.array([r['y'] for r in rows])
        ax.plot(xs, ys, '-', color='k', lw=0.6, alpha=0.5)
        for lab in LABELS:
            sel = [r['label'] == lab for r in rows]
            if any(sel):
                ax.scatter(xs[sel], ys[sel], s=12, color=LABEL_COLOR[lab],
                           label=f'{lab} ({sum(sel)})', zorder=3)
        ax.set_aspect('equal')
        ax.set_xlim(-0.05, 0.05)
        ax.set_ylim(-0.055, 0.055)
        ax.set_xticks([-0.04, 0, 0.04], ['-40', '0', '40'])
        ax.set_yticks([-0.04, 0, 0.04], ['-40', '0', '40'])
        ax.set_xlabel('x (mm)')
        ax.set_ylabel('y (mm)')
        ax.set_title(name)
        ax.legend(loc='lower left', frameon=False)
    fig.suptitle('Configurations along each path (background: certified slice at β = π/2)')
    P.save(fig, CASE, 'B1_paths_labels')


def fig_ranges(results):
    fig, axes = plt.subplots(2, 2, figsize=(10, 5.6), sharex='col',
                             constrained_layout=True)
    for col, (name, rows) in enumerate(results.items()):
        t = np.array([r['t'] for r in rows])
        for row, key, req in ((0, 'fz_range', 'fz'), (1, 'fy_range', 'fy')):
            ax = axes[row, col]
            lo = np.array([r[key][0] for r in rows])
            hi = np.array([r[key][1] for r in rows])
            ax.fill_between(t, lo, hi, color=P.C_MOVING, alpha=0.3, lw=0,
                            label='reachable total force')
            if req == 'fz':
                ax.axhline(c.F_MAX, color='k', ls='--', lw=0.8, label='requirement ±0.1 N')
                ax.axhline(-c.F_MAX, color='k', ls='--', lw=0.8)
                bad = ~((lo <= -c.F_MAX + 1e-9) & (hi >= c.F_MAX - 1e-9))
                ax.set_ylabel('$^Hf_{z,1}$ (N)')
            else:
                ax.axhline(0.0, color='k', ls='--', lw=0.8, label='requirement 0 N')
                bad = ~((lo <= 1e-9) & (hi >= -1e-9))
                ax.set_ylabel('$^Hf_{y,1}$ (N)')
            bad |= np.isnan(lo)
            for tt in t[bad]:
                ax.axvspan(tt - 0.05, tt + 0.05, color=P.C_OUTSIDE, alpha=0.25, lw=0)
            ax.set_ylim(-0.6, 0.6)
            if row == 0:
                ax.set_title(f'{name}: {int(bad.sum())} samples fail')
            ax.legend(loc='upper right', frameon=False)
        axes[1, col].set_xlabel('time (s)')
    fig.suptitle('Reachable total force (coils + interaction), red = requirement not met')
    P.save(fig, CASE, 'B2_force_ranges')


def fig_interaction(results):
    fig, ax = plt.subplots(figsize=(5.2, 3.4), constrained_layout=True)
    for name, rows in results.items():
        d = np.array([r['dist_fixed'] for r in rows]) * 1e3
        f = np.hypot([r['fz_interaction'] for r in rows], [r['fy_interaction'] for r in rows])
        ax.semilogy(d, f, '.', ms=4, label=name)
    ax.axhline(c.F_MAX, color='k', ls='--', lw=0.8, label='required thrust 0.1 N')
    ax.set_xlabel('distance to fixed magnet (mm)')
    ax.set_ylabel('|interaction force| (N)')
    ax.set_title('The interaction grows as 1/r⁴ and exhausts the coils')
    ax.legend(frameon=False)
    P.save(fig, CASE, 'B3_interaction')


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--speed', type=float, default=0.005, help='common path speed, m/s')
    ap.add_argument('--stride', type=int, default=1)
    ap.add_argument('--show', action='store_true')
    args = ap.parse_args()
    P.apply_style()

    results = {}
    for name, fn in PATHS.items():
        traj = retime(c.load_trajectory(fn), args.speed)
        results[name] = analyse(traj, args.stride)
        counts = {k: sum(r['label'] == k for r in results[name]) for k in LABELS}
        print(f'{name}: {counts}')

    fig_paths(results)
    fig_ranges(results)
    fig_interaction(results)
    out = P.out_dir(CASE) / 'B_paths_summary.json'
    out.write_text(json.dumps(results, indent=1, default=float))
    print(f'saved {out}')
    if args.show:
        plt.show()


if __name__ == '__main__':
    main()
