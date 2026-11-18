#!/usr/bin/env python3
# Run from the repository root:
#   python -m revised_case_studies.cs1_cylinder_interaction.partB_probe_points
"""Case Study 1, Part B (probe points) -- the same demanding motion at three poses.

Reviewer comment 5: compare actuation performance inside the TFW, near its
boundary and outside it.  Three configurations with the same heading
(beta = pi/2) are picked on the line x = 10 mm approaching the fixed magnet:

  inside     certified, >= 3 mm from the certified edge
  boundary   certified, < 3 mm from the edge
  outside    exact pointwise test fails, with the reachable thrust or the
             zero-lateral-force condition missed by a clear margin

At each, the closed loop holds for 1 s, then moves 8 mm forward and 8 mm back
along the heading at 20 mm/s.  With the drag law this needs the full
+-0.1 N of thrust in both directions -- exactly the task that defines the TFW.
Only the sharp 3-D TFW and pointwise checks are used.

Figure: B6_probe_points (+ B_probe_summary.json)
"""

from __future__ import annotations

import argparse
import json
import math

import matplotlib.pyplot as plt
import numpy as np

from revised_case_studies.cs1_cylinder_interaction import common as c
from revised_case_studies.cs1_cylinder_interaction.closed_loop import simulate
from revised_case_studies.shared import paths as P

CASE = 'cs1'
BETA = math.pi / 2
X_LINE = 0.01
STEP = 0.008        # m, commanded forward-and-back motion
HOLD, MOVE = 1.0, 0.4   # s; 8 mm / 0.4 s = 20 mm/s -> 0.1 N with DRAG_C = 5
MARGIN = 0.2        # outside probe: requirement missed by >= 20 % of F_MAX


def pick_probes():
    """Walk towards the fixed magnet and keep the first pose of each class."""
    found = {}
    for y in np.arange(-0.045, 0.0, 0.0005):
        lab = c.label(X_LINE, y, BETA)
        if lab == 'uncertified':
            continue
        if lab == 'infeasible':
            fr = c.force_ranges(X_LINE, y, BETA)
            fz_lo, fz_hi = fr['fz_range']
            fy_lo, fy_hi = fr['fy_range']
            short = max(fz_lo + c.F_MAX, c.F_MAX - fz_hi, fy_lo, -fy_hi) \
                if not math.isnan(fz_lo) else math.inf
            if short < MARGIN * c.F_MAX:
                continue
            lab = 'outside'
        found.setdefault(lab, float(y))
    return {k: found[k] for k in ('inside', 'boundary', 'outside') if k in found}


def step_reference(y0):
    return np.array([[X_LINE, y0, BETA, 0.0],
                     [X_LINE, y0, BETA, HOLD],
                     [X_LINE, y0 + STEP, BETA, HOLD + MOVE],
                     [X_LINE, y0 + STEP, BETA, HOLD + MOVE + 0.5],
                     [X_LINE, y0, BETA, HOLD + 2 * MOVE + 0.5]])


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--show', action='store_true')
    args = ap.parse_args()
    P.apply_style()
    probes = pick_probes()
    print('probe y (m):', probes)

    fig, axes = plt.subplots(2, len(probes), figsize=(3.3 * len(probes), 5.6),
                             constrained_layout=True)
    summary = {}
    for col, (name, y0) in enumerate(probes.items()):
        r = simulate(step_reference(y0), t_extra=1.0)
        fr = c.force_ranges(X_LINE, y0, BETA)
        summary[name] = {
            'y0_m': y0, 'fz_range_N': list(fr['fz_range']), 'fy_range_N': list(fr['fy_range']),
            'final_error_m': r['final_err'], 'max_slack': float(np.nanmax(r['slack'])),
            'contact': r['contact_any'],
        }
        ax = axes[0, col]
        ax.plot(r['t'], (r['yr'] - y0) * 1e3, '--', color=P.C_REF, label='reference')
        ax.plot(r['t'], (r['y'] - y0) * 1e3, color=P.C_MOVING, label='y')
        ax.plot(r['t'], (r['x'] - X_LINE) * 1e3, color=P.C_BOUNDARY, label='x (lateral)')
        ax.set_ylim(-12, 12)
        ax.set_title(f'{name}: y0 = {y0 * 1e3:.1f} mm\n'
                     f'final error {r["final_err"] * 1e3:.1f} mm'
                     + ('  (contact)' if r['contact_any'] else ''))
        ax.set_ylabel('displacement (mm)')
        ax.legend(frameon=False, loc='lower left')
        ax2 = axes[1, col]
        ax2.plot(r['t'], r['slack'], color=P.C_OUTSIDE)
        ax2.set_ylabel('allocator slack')
        ax2.set_xlabel('time (s)')
        ax2.set_yscale('log')
        ax2.set_ylim(1e-6, 1e2)
        print(name, summary[name])
    fig.suptitle('Same commanded ±8 mm move (needs ±0.1 N) at three configurations (β = π/2)')
    P.save(fig, CASE, 'B6_probe_points')
    out = P.out_dir(CASE) / 'B_probe_summary.json'
    out.write_text(json.dumps(summary, indent=1))
    if args.show:
        plt.show()


if __name__ == '__main__':
    main()
