#!/usr/bin/env python3
"""Section 6.2 figure: finite-target model accuracy, truncation bounds, and TFW impact.

Run from the repository root after run_multipole_stage_a.py:
    .venv/bin/python -B dipole_error_comparison/plot_validation_figure.py

(a) largest force-component error and (b) largest field-component error of the
L = 1, 3, 5 models against the refined finite-body reference, as the upper
envelope over all sampled poses (four headings) at centre separation >= s, with
the analytical truncation bounds (dashed); (c) the 90 deg heading-equilibrium task with the L = 5
truncation reserve.  Writes images/multipole_validation.pdf/.png in the paper.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Patch, Rectangle
from matplotlib.collections import PatchCollection

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(HERE))
from cases.case_6_interaction_obstacle_avoidance.simulation import model  # noqa: E402
import multipole_stage_a as M  # noqa: E402
from multipole_model import bounding_radius  # noqa: E402

OUTPUT = ROOT / 'dipole_error_comparison' / 'results'
RES = HERE / 'results' / 'multipole_stage_a'
COL = {1: '#9b3b3b', 3: '#bd7b31', 5: '#215f98'}
IN, OUT, UNK, NOM = '#8fd19e', '#f4b6b6', '#d9d9d9', '#cfe3f3'
INK = '#252a30'


def style():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7.5, 'axes.titlesize': 7.8,
                         'axes.labelsize': 7.5, 'legend.fontsize': 6.6, 'xtick.labelsize': 6.8,
                         'ytick.labelsize': 6.8, 'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.edgecolor': '#777b80', 'text.color': INK, 'axes.labelcolor': INK,
                         'pdf.fonttype': 42, 'savefig.dpi': 250})


def main():
    style()
    cfg = json.loads((HERE / 'config.json').read_text())
    gm, gf = cfg['moving_cylinder'], cfg['fixed_cylinder']
    a_s, a_r = bounding_radius(gf['length_m'], gf['diameter_m']), bounding_radius(gm['length_m'], gm['diameter_m'])
    data = {h: np.load(RES / f'heading_{h}.npz') for h in (0, 45, 90, 135)}
    r = np.concatenate([d['radius'] for d in data.values()]) * 1e3
    ref = np.concatenate([d['reference'] for d in data.values()])
    s_line = np.linspace((a_s + a_r) * 1e3 + 0.15, 60, 400)

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.35), gridspec_kw={'width_ratios': [1, 1, 0.92], 'wspace': 0.42})
    for ax, cols, key, scale, unit, task, title in (
            (axes[0], [2, 3], 'force', 1e3, 'mN', 100, '(a) Force error vs separation'),
            (axes[1], [0, 1], 'field', 1e3, 'mT', None, '(b) Field error vs separation')):
        for L in (1, 3, 5):
            off = np.concatenate([d[f'L{L}_offset'] for d in data.values()])
            err = np.max(np.abs(off - ref)[:, cols], axis=1) * scale
            # Upper envelope: largest sampled error over all poses at separation >= s.
            order = np.argsort(r)
            suffix = np.maximum.accumulate(err[order][::-1])[::-1]
            k = np.searchsorted(r[order], s_line)
            env = np.where(k < len(r), suffix[np.minimum(k, len(r) - 1)], np.nan)
            ax.semilogy(s_line, env, color=COL[L], lw=1.2, label=f'$L={L}$ error')
            b = M.truncation_bounds(s_line / 1e3, L, a_s, a_r, model.FIXED_M, model.FREE_M,
                                    gf['length_m'], gm['length_m'])[key] * scale
            ax.semilogy(s_line, b, color=COL[L], lw=0.9, ls='--')
        if task:
            ax.axhline(task, color=INK, lw=0.6, ls=':')
            ax.text(59, task * 1.25, 'task half-width', ha='right', va='bottom', fontsize=6.2)
        ax.axvspan(10, (a_s + a_r) * 1e3, color=UNK, alpha=0.6, lw=0)
        ax.set(xlim=(10, 60), xlabel='Centre separation (mm)', ylabel=f'Largest component error ({unit})')
        ax.set_ylim(1e-4 if key == 'force' else 1e-6, 1e4 if key == 'force' else 1e2)
        ax.set_title(title, loc='left', pad=3)
    h = [plt.Line2D([], [], color=COL[L], lw=1.2, label=f'$L={L}$') for L in (1, 3, 5)]
    h += [plt.Line2D([], [], color=INK, lw=1.2, label=r'largest sampled error at $\geq s$'),
          plt.Line2D([], [], color=INK, lw=0.9, ls='--', label='analytical bound')]
    fig.legend(handles=h, loc='lower center', bbox_to_anchor=(0.36, -0.15), frameon=False, ncol=5,
               handlelength=1.8, columnspacing=1.2)

    d = data[90]
    pts = d['points'] * 1e3
    res = d['L5_reserved_mask']; nom = d['L5_physical_mask']; refm = d['reference_physical_mask']
    ax = axes[2]
    step = pts[1, 0] - pts[0, 0] if pts[1, 0] != pts[0, 0] else np.diff(np.unique(pts[:, 0]))[0]
    for mask, color in ((~nom, UNK), (nom & ~res, NOM), (res, IN)):
        cells = [Rectangle((x - step / 2, y - step / 2), step, step) for x, y in pts[mask]]
        ax.add_collection(PatchCollection(cells, facecolor=color, edgecolor='white',
                                          linewidth=0.18, antialiased=True))
    ax.add_patch(Circle((model.FIXED_X * 1e3, model.FIXED_Y * 1e3), (a_s + a_r) * 1e3,
                        fc='white', ec='#66707b', lw=0.5, ls=':', zorder=3))
    ax.plot(model.FIXED_X * 1e3, model.FIXED_Y * 1e3, 's', ms=3, color='#69747e', zorder=4)
    ax.set(xlim=(-50, 50), ylim=(-50, 50), xlabel='$x$ (mm)', ylabel='$y$ (mm)')
    ax.set_xticks([-50, 0, 50]); ax.set_yticks([-50, 0, 50]); ax.set_aspect('equal')
    ax.set_title(r'(c) $L=5$ with reserve, $\beta=90^\circ$', loc='left', pad=3)
    ax.legend(handles=[Patch(fc=IN, label=f'feasible with truncation reserve ({res.sum()})'),
                       Patch(fc=NOM, label=f'feasible without reserve only ({(nom & ~res).sum()})'),
                       Patch(fc=UNK, label=f'nominally infeasible ({(~nom).sum()})')],
              loc='lower left', bbox_to_anchor=(-0.13, -0.68), frameon=True, framealpha=0.9,
              edgecolor='none', fontsize=6.0, handlelength=1.0, borderpad=0.3)
    for ext in ('pdf', 'png'):
        fig.savefig(OUTPUT / f'multipole_validation.{ext}', bbox_inches='tight', pad_inches=0.03, facecolor='white')
    print('saved', OUTPUT / 'multipole_validation.pdf', '| feasible with truncation reserve', int(res.sum()), 'nominal', int(nom.sum()),
          'reference', int(refm.sum()), 'of', len(res))


if __name__ == '__main__':
    main()
