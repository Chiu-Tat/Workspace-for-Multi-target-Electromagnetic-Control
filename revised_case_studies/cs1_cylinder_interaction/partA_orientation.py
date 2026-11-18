#!/usr/bin/env python3
# Run from the repository root:
#   python -m revised_case_studies.cs1_cylinder_interaction.partA_orientation [--animate]
"""Case Study 1, Part A -- the TFW is a set of configurations, not positions.

Uses the sharp certified 3-D TFW ``results/cs1/tfw3d_fine/tfw3d_single.json``
(configuration x, y, beta; via ``common.tfw``).  No workspace is recomputed.

Figures (``revised_case_studies/results/cs1``):

  A1_slices          feasible (x, y) region at eight headings, each with a
                     magnet drawn at that heading
  A2_heading_rings   fraction of certified headings per position, and at
                     three probe positions which headings are certified
  A3_area_vs_beta    certified area fraction versus heading
  A4_tfw_3d          the 3-D TFW with one slicing plane
  A5_beta_sweep.gif  (--animate) the region deforming as beta sweeps 0..2pi
"""

from __future__ import annotations

import argparse
import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from revised_case_studies.cs1_cylinder_interaction import common as c
from revised_case_studies.shared import paths as P

CASE = 'cs1'
HEADINGS = [k * math.pi / 4 for k in range(8)]
# Probe positions are chosen from the certified heading freedom: one point
# where almost every heading is certified, one near 50 % and one near 25 %.
PROBE_FRACTIONS = (1.0, 0.5, 0.25)
_CMAP = ListedColormap(['#f3f1ec', P.C_INSIDE])


def _beta_label(b):
    k = round(b / (math.pi / 4))
    return {0: '0', 1: 'π/4', 2: 'π/2', 3: '3π/4', 4: 'π', 5: '5π/4',
            6: '3π/2', 7: '7π/4'}.get(k, f'{b:.2f}')


def _draw_slice(ax, t, beta, show_glyphs=True):
    ax.imshow(t.slice_at(beta), origin='lower', extent=t.extent, cmap=_CMAP,
              vmin=0, vmax=1, interpolation='nearest')
    c.draw_magnet(ax, *c.FIXED_XY, c.FIXED_BETA, P.C_FIXED)
    if show_glyphs:
        # A sparse lattice of moving-magnet glyphs at this heading: solid where
        # the configuration is certified, faint where it is not.
        for x in np.linspace(-0.04, 0.04, 5):
            for y in np.linspace(-0.04, 0.04, 5):
                if math.hypot(x - c.FIXED_XY[0], y - c.FIXED_XY[1]) < 0.012:
                    continue
                ok = t.contains(x, y, t.slice_at(beta))
                c.draw_magnet(ax, x, y, beta, P.C_MOVING, alpha=1.0 if ok else 0.18)
    ax.set_xlim(t.x_min, t.x_max)
    ax.set_ylim(t.y_min, t.y_max)
    ax.set_aspect('equal')
    ax.set_xticks([-0.04, 0, 0.04], ['-40', '0', '40'])
    ax.set_yticks([-0.04, 0, 0.04], ['-40', '0', '40'])


def fig_slices(t):
    fig, axes = plt.subplots(2, 4, figsize=(10, 5.4), constrained_layout=True)
    for ax, beta in zip(axes.flat, HEADINGS):
        _draw_slice(ax, t, beta)
        frac = t.slice_at(beta).mean()
        ax.set_title(f'β = {_beta_label(beta)}   ({100 * frac:.0f}% certified)')
    for ax in axes[1]:
        ax.set_xlabel('x (mm)')
    for ax in axes[:, 0]:
        ax.set_ylabel('y (mm)')
    fig.suptitle('Same positions, different headings: the certified region '
                 'changes with orientation (grey = fixed magnet)')
    P.save(fig, CASE, 'A1_slices')


def pick_probes(t, fractions=PROBE_FRACTIONS, min_gap=0.015):
    """Positions whose certified-heading fraction is closest to each target."""
    n = t.raster
    step = max(1, n // 40)
    freedom = t.heading_freedom / t.n_levels
    cells = [(j, i) for j in range(step // 2, n, step) for i in range(step // 2, n, step)]
    xy = lambda j, i: (t.x_min + (i + 0.5) / n * (t.x_max - t.x_min),
                       t.y_min + (j + 0.5) / n * (t.y_max - t.y_min))
    probes = []
    for f in fractions:
        best = min((abs(freedom[j, i] - f), xy(j, i)) for j, i in cells
                   if all(math.dist(xy(j, i), q) >= min_gap for q in probes)
                   and abs(xy(j, i)[0]) < 0.042 and abs(xy(j, i)[1]) < 0.042)
        probes.append(best[1])
    return probes


def fig_rings(t):
    probes = pick_probes(t)
    fig = plt.figure(figsize=(10, 3.6), constrained_layout=True)
    gs = fig.add_gridspec(1, 4, width_ratios=[1.3, 1, 1, 1])
    ax0 = fig.add_subplot(gs[0])
    freedom = t.heading_freedom / t.n_levels
    im = ax0.imshow(freedom, origin='lower', extent=t.extent, cmap='Greens',
                    vmin=0, vmax=1)
    c.draw_magnet(ax0, *c.FIXED_XY, c.FIXED_BETA, P.C_FIXED)
    for k, (x, y) in enumerate(probes):
        ax0.plot(x, y, 'o', mfc='white', mec='k', ms=7)
        ax0.annotate(f'P{k + 1}', (x, y), xytext=(5, 5), textcoords='offset points')
    ax0.set_aspect('equal')
    ax0.set_xticks([-0.04, 0, 0.04], ['-40', '0', '40'])
    ax0.set_yticks([-0.04, 0, 0.04], ['-40', '0', '40'])
    ax0.set_xlabel('x (mm)')
    ax0.set_ylabel('y (mm)')
    ax0.set_title('fraction of certified headings')
    fig.colorbar(im, ax=ax0, shrink=0.8)

    for k, (x, y) in enumerate(probes):
        ax = fig.add_subplot(gs[k + 1], projection='polar')
        ok = t.headings_at(x, y)
        th = (np.arange(t.n_levels) + 0.5) * t.beta_step
        ax.bar(th, np.ones_like(th), width=t.beta_step, bottom=0.6,
               color=np.where(ok, P.C_INSIDE, '#e6e1d8'), edgecolor='white', lw=0.3)
        ax.set_ylim(0, 1.6)
        ax.set_yticks([])
        ax.set_title(f'P{k + 1} ({x * 1e3:.0f}, {y * 1e3:.0f}) mm\n'
                     f'{ok.mean() * 100:.0f}% of headings certified')
    P.save(fig, CASE, 'A2_heading_rings')


def fig_area(t):
    fig, ax = plt.subplots(figsize=(5, 3), constrained_layout=True)
    th = (np.arange(t.n_levels) + 0.5) * t.beta_step
    area = t.stack.reshape(t.n_levels, -1).mean(axis=1)
    ax.plot(th, 100 * area, color=P.C_INSIDE, lw=1.8)
    ax.set_xticks([0, math.pi / 2, math.pi, 1.5 * math.pi, 2 * math.pi],
                  ['0', 'π/2', 'π', '3π/2', '2π'])
    ax.set_xlabel('heading β (rad)')
    ax.set_ylabel('certified area of (x, y) domain (%)')
    ax.set_title('Area of the TFW slice versus heading')
    P.save(fig, CASE, 'A3_area_vs_beta')


def fig_3d(t, beta_cut=math.pi / 2, max_boxes=6000):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: F401
    fig = plt.figure(figsize=(9, 4.2), constrained_layout=True)
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    boxes = t.boxes
    rng = np.random.default_rng(0)
    if len(boxes) > max_boxes:
        boxes = boxes[rng.choice(len(boxes), max_boxes, replace=False)]
    cx = (boxes[:, 0] + boxes[:, 1]) / 2
    cy = (boxes[:, 2] + boxes[:, 3]) / 2
    cb = (boxes[:, 4] + boxes[:, 5]) / 2
    ax.scatter(cx * 1e3, cy * 1e3, cb, c=cb, cmap='viridis', s=2, alpha=0.35)
    xx, yy = np.meshgrid([-50, 50], [-50, 50])
    ax.plot_surface(xx, yy, np.full_like(xx, beta_cut, dtype=float),
                    color='#c8553d', alpha=0.25)
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    ax.set_zlabel('β (rad)')
    ax.set_title('3-D TFW (certified boxes)')
    ax2 = fig.add_subplot(1, 2, 2)
    _draw_slice(ax2, t, beta_cut)
    ax2.set_title(f'slice at β = {_beta_label(beta_cut)}')
    P.save(fig, CASE, 'A4_tfw_3d')


def animate(t, frames=64):
    from matplotlib.animation import FuncAnimation, PillowWriter
    fig, ax = plt.subplots(figsize=(4.6, 4.6), constrained_layout=True)

    def draw(k):
        ax.clear()
        beta = (k + 0.5) * 2 * math.pi / frames
        _draw_slice(ax, t, beta)
        ax.set_title(f'β = {math.degrees(beta):5.1f}°')

    anim = FuncAnimation(fig, draw, frames=frames)
    out = P.out_dir(CASE) / 'A5_beta_sweep.gif'
    anim.save(out, writer=PillowWriter(fps=8))
    print(f'saved {out}')


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--animate', action='store_true', help='also write the beta-sweep GIF')
    ap.add_argument('--show', action='store_true')
    args = ap.parse_args()
    P.apply_style()
    t = c.tfw()
    fig_slices(t)
    fig_rings(t)
    fig_area(t)
    fig_3d(t)
    if args.animate:
        animate(t)
    if args.show:
        plt.show()


if __name__ == '__main__':
    main()
