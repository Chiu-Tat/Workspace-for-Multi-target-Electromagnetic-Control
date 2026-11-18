#!/usr/bin/env python3
# Run from the repository root:
#   python -m revised_case_studies.cs3_helical_robots.field_check
"""Case Study 3 -- rotating-field feasibility inside vs. outside the certified set.

For three configurations (a1, a2) -- deep inside the archived Case 3 TFW,
near its boundary, and in the uncertified band -- sweep both field phases
(phi1, phi2) over a full rotation and allocate currents with the bounded
least-squares fallback.  The relative field error over the phase torus shows
whether both robots can be rotated independently at that configuration.
No TFW is computed; this is a pointwise check at three configurations.

Figure: D5_field_check (+ D_field_check.json)
"""

from __future__ import annotations

import argparse
import json
import math

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import distance_transform_edt

from revised_case_studies.cs3_helical_robots import config as K
from revised_case_studies.cs3_helical_robots import geometry as g
from revised_case_studies.cs3_helical_robots.dynamics import field_matrix
from revised_case_studies.shared import paths as P
from revised_case_studies.shared.allocator import fallback_allocate

CASE = 'cs3'


def pick_configs():
    t = g.tfw()
    # pad with 'blocked' so the search-box edge counts as a boundary too
    d_in = distance_transform_edt(np.pad(t.free, 1))[1:-1, 1:-1]
    d_out = distance_transform_edt(~t.free)
    inside = np.unravel_index(np.argmax(d_in), d_in.shape)
    d_band = distance_transform_edt(t.free)      # distance to the uncertified band only
    boundary = np.argwhere((d_band > 0) & (d_band <= 2))
    boundary = tuple(boundary[len(boundary) // 2])
    outside = np.unravel_index(np.argmax(d_out), d_out.shape)
    return {'inside': t.angles(inside), 'boundary': t.angles(boundary),
            'outside': t.angles(outside)}


def error_map(a, n=24):
    phis = np.linspace(0, 2 * math.pi, n, endpoint=False)
    err = np.zeros((n, n))
    w = np.full(6, 1.0 / K.B_ROT)
    for i, p1 in enumerate(phis):
        for j, p2 in enumerate(phis):
            phi = (p1, p2)
            G, c = field_matrix(a, phi)
            h = np.concatenate([K.B_ROT * g.field_direction(a[k], phi[k]) for k in range(2)])
            sol = fallback_allocate(G, c, h, w, -K.I_LIMIT, K.I_LIMIT)
            err[j, i] = np.max(np.abs(sol['slack'])) / K.B_ROT
    return phis, err


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--n', type=int, default=24, help='phase samples per robot')
    ap.add_argument('--show', action='store_true')
    args = ap.parse_args()
    P.apply_style()
    configs = pick_configs()
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.4), constrained_layout=True,
                             gridspec_kw={'width_ratios': [1.1, 1, 1, 1]})
    t = g.tfw()
    ext = [math.degrees(v) for v in (*K.A1_RANGE, *K.A2_RANGE)]
    axes[0].imshow(t.free, origin='lower', extent=ext, aspect='auto',
                   cmap=plt.matplotlib.colors.ListedColormap(['#e6ded3', '#cfe3d2']))
    summary = {}
    for k, (name, a) in enumerate(configs.items()):
        phis, err = error_map(np.array(a), args.n)
        summary[name] = {'a_deg': [math.degrees(v) for v in a],
                         'max_rel_field_error': float(err.max()),
                         'fraction_of_phases_feasible': float(np.mean(err < 1e-6))}
        axes[0].plot(*np.degrees(a), 'o', ms=7, mec='k', clip_on=False,
                     color=(P.C_INSIDE, P.C_BOUNDARY, P.C_OUTSIDE)[k])
        axes[0].annotate(name, np.degrees(a), xytext=(5, 4), textcoords='offset points')
        ax = axes[k + 1]
        im = ax.imshow(err * 100, origin='lower', cmap='magma_r', vmin=0,
                       vmax=max(1.0, err.max() * 100), extent=[0, 360, 0, 360])
        ax.set_title(f'{name}: max error {err.max() * 100:.1f}%\n'
                     f'{summary[name]["fraction_of_phases_feasible"] * 100:.0f}% of phase pairs exact')
        ax.set_xlabel('φ₁ (deg)')
        ax.set_ylabel('φ₂ (deg)')
        fig.colorbar(im, ax=ax, shrink=0.8, label='field error (%)')
        print(name, summary[name])
    axes[0].set_xlabel('a₁ (deg)')
    axes[0].set_ylabel('a₂ (deg)')
    axes[0].set_title('certified set (green)')
    P.save(fig, CASE, 'D5_field_check')
    (P.out_dir(CASE) / 'D_field_check.json').write_text(json.dumps(summary, indent=1))
    if args.show:
        plt.show()


if __name__ == '__main__':
    main()
