#!/usr/bin/env python3
# Run from the repository root:
#   python -m revised_case_studies.cs1_cylinder_interaction.partB_simulation
"""Case Study 1, Part B (simulation) -- following a path inside vs. outside the TFW.

Runs the same closed loop (``closed_loop.py``) on the planned (sharp 3-D TFW)
and straight references of ``common``, re-timed at a common speed.  Reports tracking
error, allocator slack, coil saturation, minimum separation from the fixed
magnet and whether contact occurs.  Uses the TFW only for labels.

Figures (``revised_case_studies/results/cs1``):

  B4_sim_paths        reference vs. simulated path for both trajectories
  B5_sim_timeseries   position error, allocator slack, max |current|, separation
  B_sim_summary.json
"""

from __future__ import annotations

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np

from revised_case_studies.cs1_cylinder_interaction import common as c
from revised_case_studies.cs1_cylinder_interaction.closed_loop import simulate
from revised_case_studies.cs1_cylinder_interaction.partB_paths import PATHS, retime
from revised_case_studies.shared import paths as P
from revised_case_studies.shared.metrics import position_stats

CASE = 'cs1'


def fig_paths(runs):
    t = c.tfw()
    fig, axes = plt.subplots(1, len(runs), figsize=(9, 4.4), constrained_layout=True)
    for ax, (name, r) in zip(axes, runs.items()):
        ax.imshow(t.slice_at(np.pi / 2), origin='lower', extent=t.extent,
                  cmap=plt.matplotlib.colors.ListedColormap(['#f3f1ec', '#cfe3d2']),
                  interpolation='nearest')
        c.draw_magnet(ax, *c.FIXED_XY, c.FIXED_BETA, P.C_FIXED)
        ax.plot(r['xr'], r['yr'], '--', color=P.C_REF, lw=1, label='reference')
        ax.plot(r['x'], r['y'], '-', color=P.C_MOVING, lw=1.8, label='simulated')
        for k in np.linspace(0, len(r['t']) - 1, 7).astype(int):
            c.draw_magnet(ax, r['x'][k], r['y'][k], r['beta'][k], P.C_MOVING, alpha=0.8)
        if r['contact_any']:
            k = int(np.argmax(r['contact']))
            ax.plot(r['x'][k], r['y'][k], 'x', color=P.C_OUTSIDE, ms=12, mew=2,
                    label=f'contact at t = {r["t"][k]:.1f} s')
        ax.set_aspect('equal')
        ax.set_xlim(-0.05, 0.05)
        ax.set_ylim(-0.055, 0.055)
        ax.set_xticks([-0.04, 0, 0.04], ['-40', '0', '40'])
        ax.set_yticks([-0.04, 0, 0.04], ['-40', '0', '40'])
        ax.set_xlabel('x (mm)')
        ax.set_ylabel('y (mm)')
        ax.set_title(name)
        ax.legend(loc='lower left', frameon=False)
    P.save(fig, CASE, 'B4_sim_paths')


def fig_series(runs):
    fig, axes = plt.subplots(4, 1, figsize=(7, 7.5), sharex=True, constrained_layout=True)
    for name, r in runs.items():
        axes[0].plot(r['t'], r['pos_err'] * 1e3, label=name)
        axes[1].plot(r['t'], r['slack'], label=name)
        axes[2].plot(r['t'], np.max(np.abs(r['currents']), axis=1), label=name)
        axes[3].plot(r['t'], r['sep'] * 1e3, label=name)
    axes[0].set_ylabel('position error (mm)')
    axes[1].set_ylabel('allocator slack\n(relative)')
    axes[2].set_ylabel('max |I| (A)')
    axes[2].axhline(c.I_MAX, color='k', ls='--', lw=0.8)
    axes[3].set_ylabel('distance to\nfixed magnet (mm)')
    axes[3].axhline(c.CONTACT_DIST * 1e3, color=P.C_OUTSIDE, ls='--', lw=0.8, label='contact')
    axes[3].set_xlabel('time (s)')
    axes[0].legend(frameon=False)
    axes[3].legend(frameon=False)
    P.save(fig, CASE, 'B5_sim_timeseries')


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--speed', type=float, default=0.005, help='common path speed, m/s')
    ap.add_argument('--slew', type=float, default=None, help='per-step current slew limit, A')
    ap.add_argument('--show', action='store_true')
    args = ap.parse_args()
    P.apply_style()

    runs, summary = {}, {}
    for name, fn in PATHS.items():
        traj = retime(c.load_trajectory(fn), args.speed)
        r = simulate(traj, di_max=args.slew)
        runs[name] = r
        active = r['t'] <= traj[-1, 3]
        summary[name] = {
            'position_error_m': position_stats(
                np.column_stack([r['x'], r['y']])[active],
                np.column_stack([r['xr'], r['yr']])[active]),
            'final_goal_error_m': r['final_err'],
            'max_slack': float(np.nanmax(r['slack'])),
            'saturated_fraction': float(np.mean(np.max(np.abs(r['currents']), axis=1) > c.I_MAX - 1e-3)),
            'min_separation_m': float(r['sep'].min()),
            'contact': r['contact_any'],
        }
        print(name, json.dumps(summary[name], indent=1))

    fig_paths(runs)
    fig_series(runs)
    out = P.out_dir(CASE) / 'B_sim_summary.json'
    out.write_text(json.dumps(summary, indent=1))
    print(f'saved {out}')
    if args.show:
        plt.show()


if __name__ == '__main__':
    main()
