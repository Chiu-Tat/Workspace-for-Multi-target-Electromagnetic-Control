"""Case Study 1 figures, second design: workspace figure and hardware figure.

Run from the repository root after ``hardware`` has written trials.json:
  .venv/bin/python -B -m revised_case_studies.cs1_cylinder_interaction.figures_v2

Writes cs1_workspace.{pdf,png} and cs1_hardware.{pdf,png} to
results/cs1/publication and copies them to the paper's images/case_study_1.

Heading coverage of a position (x, y) is the share of the 64 heading levels
whose TFW-in boxes contain it; it is exact because every TFW-in box spans
whole heading levels.  It summarizes the 3-D TFW and shows where a position is
usable at every heading (100 %) or at none (0 %).
"""
from __future__ import annotations

import json
import math
import os
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Ellipse, Patch
import numpy as np
from functools import lru_cache
from scipy.ndimage import distance_transform_edt
from matplotlib import patheffects as pe

# White outline so lines stay readable on the dark coverage background.
HALO = [pe.Stroke(linewidth=2.6, foreground='white'), pe.Normal()]
MEAS = '#ffd23f'             # measured paths, same yellow as the tracked path in the camera frames
REF_INK = '#15191e'          # reference paths: thin black dashes; measured paths: thick solid colour

from cases.case_6_interaction_obstacle_avoidance import workspace_hybrid as W
from revised_case_studies.cs1_cylinder_interaction import common as C
from revised_case_studies.cs1_cylinder_interaction import publication as P
from revised_case_studies.shared.paths import out_dir

OUT = P.OUT
HW = out_dir('cs1') / 'hardware'
FRAMES = HW / 'frames'
INNER, OUTER, UNKNOWN, INK, BLUE, ORANGE = P.INNER, P.OUTER, P.UNKNOWN, P.INK, P.BLUE, P.ORANGE
AMBER, RED = '#d99a2b', '#b2302b'
GUARD_MM = 16.0
LAYOUTS = ('single', 'gate', 'triangle')
LAYOUT_NAME = {'single': 'Single', 'gate': 'Gate', 'triangle': 'Triangle'}
# One opaque coverage scale for every coverage panel in Figs. 14 and 15.
COVER = LinearSegmentedColormap.from_list('cover', ['#ffffff', '#e3eef8', '#b3d0ea', '#6fa3d2', '#3c78b4'])
HEADING = plt.get_cmap('twilight_shifted')
ZOOM = (-12, 48, -30, 30)        # section window, mm: centred on the single fixed magnet

# Camera crop of ui.avi (608 x 608 px at x=20, y=68) and its map to EM millimetres,
# fitted to the fixed-magnet centres visible in the frames.
# The control panel (ROS_CAN .../scripts/control_panel.py, CameraView.world_transform) warps each
# camera frame with the session's pixel-to-EM calibration onto a square spanning x, y in [-50, 50] mm,
# +y up.  In every CS1 ui.avi that square fills the 607 x 607 px box inside the 1 px panel border
# at (21, 69); the edges were measured from the video pixels.  Hence the crop spans exactly +-50 mm.
CROP = 'crop=607:607:21:69'
EXTENT = (-50.0, 50.0, -50.0, 50.0)


# --------------------------------------------------------------------------- helpers
def coverage(layout):
    t = C.tfw(layout)
    return t.heading_freedom / t.n_levels * 100.0, t.extent


def fixed_mm(layout):
    return [(x * 1e3, y * 1e3, b) for x, y, b, _ in W.LAYOUTS[layout]]


def draw_fixed(ax, layout, *, guard=True, scale=1.0):
    for x, y, b in fixed_mm(layout):
        if guard:
            ax.add_patch(Circle((x, y), GUARD_MM, fc='none', ec='#4b545e', lw=.5,
                                ls=(0, (2.2, 1.6)), zorder=6))
        P.magnet_symbol(ax, x, y, b)


def axes_mm(ax, lim=(-50, 50, -50, 50), *, xlabel=True, ylabel=True, ticks=(-40, 0, 40)):
    ax.set(xlim=lim[:2], ylim=lim[2:]); ax.set_aspect('equal')
    ax.set_xticks([v for v in ticks if lim[0] <= v <= lim[1]])
    ax.set_yticks([v for v in ticks if lim[2] <= v <= lim[3]])
    ax.tick_params(length=2, pad=1.5, labelsize=6.2)
    if xlabel: ax.set_xlabel('$x$ (mm)', labelpad=1, fontsize=6.8)
    else: ax.set_xticklabels([])
    if ylabel: ax.set_ylabel('$y$ (mm)', labelpad=0, fontsize=6.8)
    else: ax.set_yticklabels([])


def title(ax, letter, text, **kw):
    ax.set_title(f'({letter}) {text}', loc='left', pad=kw.pop('pad', 3), fontsize=kw.pop('fontsize', 7.2), **kw)


def coverage_map(ax, layout, *, alpha=1.0, contour=True):
    cov, ext = coverage(layout)
    ext_mm = [v * 1e3 for v in ext]
    im = ax.imshow(cov, origin='lower', extent=ext_mm, cmap=COVER, vmin=0, vmax=100,
                   interpolation='nearest', alpha=alpha, rasterized=True, zorder=0)
    if contour:
        n = cov.shape[0]
        g = np.linspace(ext_mm[0], ext_mm[1], n + 1); g = .5 * (g[1:] + g[:-1])
        ax.contour(g, g, cov, levels=[99.9], colors=['#173f22'], linewidths=.55, zorder=2)
    return im


def section_boxes(ax, layout, deg):
    data = P.cut3d(P.load_tfw3d(layout), layout, deg)
    edges = {'undetermined': '#a9adb2', 'inside': '#5fa874', 'outside': '#d98a8a'}
    for kind, color in (('undetermined', UNKNOWN), ('inside', INNER), ('outside', OUTER)):
        polys = [[(b[0]*1e3, b[2]*1e3), (b[1]*1e3, b[2]*1e3), (b[1]*1e3, b[3]*1e3), (b[0]*1e3, b[3]*1e3)]
                 for b in data['boxes'][kind]]
        if polys:
            # Box outlines show the adaptive subdivision of the TFW.
            ax.add_collection(PolyCollection(polys, facecolors=color, edgecolors=edges[kind],
                                             linewidths=.12, rasterized=True, zorder=1))
    return data['area_fraction_full_square']['inside'] * 100


def load_trials():
    d = json.loads((HW / 'trials.json').read_text())
    for r in d.values():
        for k in ('x', 'y', 'beta', 'lam', 'd_fixed', 'i_cmd', 'i_meas', 't'):
            r[k] = np.array([np.nan if v is None else v for v in r[k]], float)
        r['valid'] = np.array(r['valid'], bool)
        r['tfw'] = np.array([bool(v) for v in r['tfw']], bool)
        r['reason'] = np.array(r['reason']); r['phase'] = np.array(r['phase'])
        if 'reference' in r:
            r['reference'] = {k: np.array(v, float) for k, v in r['reference'].items()}
    return d


# --------------------------------------------------------------------------- workspace figure
def heading_planes(ax):
    """Mark the four section headings on the 3-D TFW."""
    xx, yy = np.meshgrid([-50, 50], [-50, 50])
    for deg in P.HEADINGS:
        ax.plot_surface(xx, yy, np.full_like(xx, deg, float), color='#215f98', alpha=.13,
                        linewidth=0, shade=False, zorder=30)
        ax.plot([-50, 50, 50], [50, 50, -50], [deg]*3, color='#215f98', lw=.6, zorder=31)


# Probe positions (mm): P1 away from the fixed magnets, P2 and P3 about 12 mm from one of them.
PROBES = {'gate': [(0.0, 0.0), (-10.0, 22.0), (-11.0, 28.0)],
          'triangle': [(0.0, 0.0), (-10.0, 24.0), (-11.0, 30.0)]}
PROBE_OFFSETS = [(3, 2), (-10, -7), (-10, 3)]
# Where the moving-magnet symbol sits in each section, clear of the fixed magnets (mm).
GLYPH_AT = {'single': (-32, -32), 'gate': (-30, -40), 'triangle': (0, -40)}
ROW_NAME = {'single': 'Single', 'gate': 'Gate', 'triangle': 'Triangle'}
COL_HEAD = ['3-D TFW', 'Coverage'] + [rf'$\beta={d}^\circ$' for d in P.HEADINGS] + ['Point tests']


def probe_details(layout):
    """Pointwise endpoint tests at P1-P3 over 60 headings (cached)."""
    if layout == 'single':
        return json.loads((OUT / 'orientation_probes.json').read_text())
    path = OUT / f'orientation_probes_{layout}.json'
    if path.exists():
        return json.loads(path.read_text())
    angles = list(range(0, 360, 6))
    probes = [(x / 1e3, y / 1e3) for x, y in PROBES[layout]]
    d = {'layout': layout, 'probes_m': probes, 'probe_headings_deg': angles,
         'probe_classes': [[P.point_test(x, y, math.radians(b), layout) for b in angles]
                           for x, y in probes]}
    path.write_text(json.dumps(d, indent=1))
    return d


def mark_probes(ax, probes, layout, deg=None, ms=2.6):
    for i, (x, y) in enumerate(probes):
        if deg is None:
            face = 'white'
        else:
            face = INNER if P.point_test(x / 1e3, y / 1e3, math.radians(deg), layout) == 'inside' else OUTER
        ax.plot(x, y, 'o', ms=ms, mfc=face, mec=INK, mew=.55, zorder=10)
        ax.annotate(f'P{i+1}', (x, y), xytext=PROBE_OFFSETS[i], textcoords='offset points',
                    fontsize=4.9, zorder=11)


def small_axes(ax, *, left, bottom):
    axes_mm(ax, xlabel=bottom, ylabel=left)
    ax.tick_params(labelsize=5.2, length=1.6, pad=1)
    ax.xaxis.label.set_size(5.8); ax.yaxis.label.set_size(5.8)
    ax.xaxis.labelpad = .5; ax.yaxis.labelpad = -1


def fig_workspace():
    E_IN = 1.5
    top_in, row_in, gap_in = .57, 1.117, .19
    grid_in = len(LAYOUTS) * row_in + (len(LAYOUTS) - 1) * gap_in
    H = grid_in + 4.08
    fig = plt.figure(figsize=(7.16, H))
    gs = fig.add_gridspec(len(LAYOUTS), 1, left=.085, right=.99, top=1 - top_in / H,
                          bottom=1 - (top_in + grid_in) / H, hspace=gap_in / row_in)
    for row, layout in enumerate(LAYOUTS):
        tag = 'abc'[row]
        last = row == len(LAYOUTS) - 1
        details = probe_details(layout)
        probes = [(x * 1e3, y * 1e3) for x, y in details['probes_m']]
        sub = gs[row].subgridspec(1, 9, wspace=.12, width_ratios=[1.3, .2, 1, .12, 1, 1, 1, 1, 1.05])
        heads = [f'({tag}{j + 1}) {COL_HEAD[j]}' if row == 0 else f'({tag}{j + 1})' for j in range(7)]
        # 3-D TFW with the section headings.
        ax3 = fig.add_subplot(sub[0], projection='3d')
        tfw = P.load_tfw3d(layout)
        P.voxel_panel(ax3, tfw, layout, '')
        heading_planes(ax3)
        ax3.set_rasterized(True)   # voxels as an image keep the PDF small
        for t in list(ax3.texts):
            if t.get_text() == r'$\beta$': t.remove()
        ax3.tick_params(labelsize=5, pad=-3)
        ax3.set_zlabel(r'$\beta$', fontsize=6.5, labelpad=-9, rotation=0)
        ax3.zaxis.set_rotate_label(False)
        if not last:
            ax3.set_xlabel(''); ax3.set_ylabel('')
        b = ax3.get_position(); ax3.set_position([b.x0 - .035, b.y0 - .01, b.width * 1.06, b.height * 1.06])
        ax3.set_title(heads[0], loc='left', fontsize=6.6, pad=-4, y=1.02)
        vol = P.volume_fractions(tfw)['inside'] * 100
        b = ax3.get_position()
        fig.text(.012, b.y0 + b.height / 2, f'{ROW_NAME[layout]}\n{vol:.1f}% TFW-in', rotation=90,
                 ha='center', va='center', fontsize=7, fontweight='bold', color=INK, linespacing=1.3)
        # Heading coverage with the probe points.
        ax = fig.add_subplot(sub[2])
        im = coverage_map(ax, layout, contour=False)
        draw_fixed(ax, layout, guard=False)
        mark_probes(ax, probes, layout)
        small_axes(ax, left=True, bottom=True)
        if row == 0:
            q = ax.get_position()
            cax = fig.add_axes([q.x0, q.y1 + .04 / H, q.width, .045 / H])
            cb = fig.colorbar(im, cax=cax, orientation='horizontal', ticks=[0, 50, 100])
            cb.ax.xaxis.set_ticks_position('top')
            cb.ax.tick_params(labelsize=4.8, length=1.4, pad=.8)
            cb.outline.set_linewidth(.4)
            ax.set_title('(a2) Coverage (%)', loc='left', fontsize=6.6, pad=17)
        else:
            ax.set_title(heads[1], loc='left', fontsize=6.6, pad=2.5)
        # Sections at four headings.
        for k, deg in enumerate(P.HEADINGS):
            ax = fig.add_subplot(sub[4 + k])
            sq = ax
            frac = section_boxes(ax, layout, deg)
            draw_fixed(ax, layout, guard=False)
            P.magnet_symbol(ax, *GLYPH_AT[layout], math.radians(deg), moving=True)
            mark_probes(ax, probes, layout, deg)
            small_axes(ax, left=k == 0, bottom=True)    # sections share y: ticks on the first only
            ax.set_ylabel('')                   # one y label per row, on the coverage panel
            ax.set_title(heads[2 + k], loc='left', fontsize=6.6, pad=2.5)
            ax.text(.97, .03, f'{frac:.0f}%', transform=ax.transAxes, ha='right', fontsize=5,
                    bbox={'fc': 'white', 'ec': 'none', 'alpha': .85, 'pad': .8}, zorder=12)
        # Pointwise tests over 60 headings.
        ax = fig.add_subplot(sub[8], projection='polar')
        P.heading_rings(ax, details)
        q, c = sq.get_position(), ax.get_position()
        h = q.height * 1.02
        ax.set_position([c.x0 + (c.width - h * H / 7.16) / 2 + .006, q.y0 + (q.height - h) / 2, h * H / 7.16, h])
        for t in ax.texts:
            t.set_fontsize(min(t.get_fontsize(), 5.2))
            if '%' in t.get_text():
                t.set_position((.5, -.03))
        ax.set_xticklabels(['0°', '90°', '', ''], fontsize=5.2)
        ax.set_title(heads[6], loc='center', fontsize=6.6, pad=6)
        for deg in P.HEADINGS:
            ax.annotate('', xy=(math.radians(deg), 1.64), xytext=(math.radians(deg), 1.86),
                        arrowprops={'arrowstyle': '-|>', 'color': BLUE, 'lw': .5, 'mutation_scale': 4},
                        annotation_clip=False)
    handles = [Patch(fc=INNER, label='TFW-in'), Patch(fc=OUTER, label='TFW-out'),
               Patch(fc=UNKNOWN, label='Unresolved'),
               Patch(fc='#b9bfc6', ec='#8a929b', lw=.4, label='3-D: not TFW-in'),
               Patch(fc='#215f98', alpha=.25, label='Section headings'),
               Patch(fc=BLUE, ec='#29313a', lw=.5, label='Moving magnet (N red)')]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.53, 1.0), ncol=6, frameon=False,
               fontsize=5.8, handlelength=1.3, columnspacing=.9)
    keyboard_section(fig, H, E_IN)
    comparison_section(fig, H, E_IN)
    export(fig, 'cs1_workspace')


FRAME_FRACTIONS = (.12, .38, .64, .9)        # frame times as fractions of each keyboard run
FREE_RING = '#e8452c'                        # ellipse marking the moving magnet in frames
FRAME_HALF = 50.0                            # half-width of each camera view, mm (50 = full workspace)


def comparison_section(fig, H, height_in):
    """Row (e): keyboard comparison trials near the fixed magnets (camera frames)."""
    d = load_trials()
    kbs, kbg = d['kb_single_fail'], d['kb_gate_fail']
    bottom = .028
    row = fig.add_gridspec(1, 3, left=.045, right=.995, bottom=bottom,
                          top=bottom + (height_in - .42) / H, wspace=.06)
    camera_panel(fig.add_subplot(row[0]), kbs, 36, 'single_attach', 'e1', 'Single: attachment', 'single',
                 upto=36, note='tracking lost at 32 s')
    camera_panel(fig.add_subplot(row[1]), kbg, 46, 'gate_stall', 'e2', 'Gate: approach stalls', 'gate',
                 upto=46, note='34–49 s: drive in 61%\nof samples, moved < 2 mm')
    camera_panel(fig.add_subplot(row[2]), kbg, 80, 'gate_attach', 'e3', 'Gate: attachment', 'gate', upto=80)
    fig.text(.012, bottom + (height_in - .2) / H, '(e) Keyboard comparison trials near the fixed magnets',
             fontsize=7, fontweight='bold', color=INK, va='bottom')


def keyboard_section(fig, H, below_in=0.0):
    """Row (d): keyboard path over heading coverage, with four numbered camera frames."""
    d = load_trials()
    b0 = .03 + below_in / H
    kb = fig.add_gridspec(1, 3, left=.045, right=.995, bottom=b0, top=b0 + 1.26 / H, wspace=.17)
    for k, layout in enumerate(LAYOUTS):
        r = d[f'kb_{layout}']
        v = r['valid'] & (r['phase'] == 'holding')
        idx = np.flatnonzero(v); t = r['t'][idx]
        picks = [idx[np.argmin(abs(t - (t[0] + f * (t[-1] - t[0]))))] for f in FRAME_FRACTIONS]
        blk = kb[k].subgridspec(2, 3, width_ratios=[2, 1, 1], wspace=.05, hspace=.05)
        ax = fig.add_subplot(blk[:, 0])
        coverage_map(ax, layout, contour=False)
        draw_fixed(ax, layout, guard=False)
        lc = heading_path(ax, r['x'][v] * 1e3, r['y'][v] * 1e3, r['beta'][v], lw=1.0)
        ax.plot(r['x'][idx[0]] * 1e3, r['y'][idx[0]] * 1e3, 'o', ms=3, mfc='white', mec=INK, mew=.6, zorder=9)
        ax.plot(r['x'][idx[-1]] * 1e3, r['y'][idx[-1]] * 1e3, 's', ms=2.8, mfc='white', mec=INK, mew=.6, zorder=9)
        for n, i in enumerate(picks, start=1):
            ax.text(r['x'][i] * 1e3, r['y'][i] * 1e3, str(n), fontsize=4.6, ha='center', va='center',
                    color='white', fontweight='bold', zorder=12,
                    bbox={'boxstyle': 'circle,pad=.18', 'fc': INK, 'ec': 'white', 'lw': .4})
        small_axes(ax, left=True, bottom=True)
        dur = t[-1] - t[0]
        ax.set_title(f'(d{k + 1}) {LAYOUT_NAME[layout]}, {dur:.0f} s', loc='left', fontsize=6.6, pad=2.5)
        path_top = ax.get_position().y1
        for n, i in enumerate(picks, start=1):
            fa = fig.add_subplot(blk[(n - 1) // 2, 1 + (n - 1) % 2])
            img = grab_frame(r['folder'], r['t'][i], f'kb_{layout}_{n}')
            fa.imshow(img, extent=EXTENT, interpolation='bilinear', zorder=0)
            cx = min(max(r['x'][i] * 1e3, -50 + FRAME_HALF), 50 - FRAME_HALF)
            cy = min(max(r['y'][i] * 1e3, -50 + FRAME_HALF), 50 - FRAME_HALF)
            fa.set(xlim=(cx - FRAME_HALF, cx + FRAME_HALF), ylim=(cy - FRAME_HALF, cy + FRAME_HALF))
            fa.set_aspect('equal')
            fa.set_xticks([]); fa.set_yticks([])
            # Dotted ellipse on the moving magnet, along its measured heading.
            fa.add_patch(Ellipse((r['x'][i] * 1e3, r['y'][i] * 1e3), 24, 14, angle=math.degrees(r['beta'][i]),
                                 fc='none', ec=FREE_RING, lw=.8, ls=(0, (1.2, 1.2)), zorder=5))
            for sp in fa.spines.values(): sp.set_visible(True); sp.set_edgecolor('#9aa3ab'); sp.set_linewidth(.4)
            fa.text(.05, .95, str(n), transform=fa.transAxes, fontsize=4.8, ha='left', va='top', color='white',
                    fontweight='bold', bbox={'boxstyle': 'circle,pad=.18', 'fc': INK, 'ec': 'white', 'lw': .4})
            fa.text(.96, .04, f'{r["t"][i]:.0f} s, {math.degrees(r["beta"][i]):.0f}°', transform=fa.transAxes,
                    fontsize=4.4, ha='right', va='bottom', color='white', zorder=8,
                    bbox={'fc': '#2a3038', 'ec': 'none', 'pad': .8})
    top = path_top + .2 / H
    fig.text(.012, top, '(d) Keyboard steering: measured path and camera frames', fontsize=7,
             fontweight='bold', color=INK, va='bottom')
    cax = fig.add_axes([.80, top + .07 / H, .17, .05 / H])
    cb = fig.colorbar(lc, cax=cax, orientation='horizontal', ticks=[0, 90, 180, 270, 360])
    cb.ax.tick_params(labelsize=5.2, length=1.5, pad=1)
    fig.text(.79, top + .095 / H, r'Measured heading $\beta$ (°)', fontsize=5.6, ha='right', va='center',
             color=INK)


# --------------------------------------------------------------------------- hardware figure
def heading_path(ax, x, y, b, lw=1.6):
    pts = np.column_stack([x, y]).reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    keep = np.hypot(np.diff(x), np.diff(y)) < 4.0       # no lines across tracking gaps
    ax.add_collection(LineCollection(segs[keep], colors='#1d2329', linewidths=lw + .9, zorder=4,
                                     capstyle='round'))
    lc = LineCollection(segs[keep], cmap=HEADING, norm=Normalize(0, 360), linewidths=lw, zorder=5,
                        capstyle='round')
    lc.set_array(np.degrees(b[:-1][keep])); ax.add_collection(lc)
    return lc


def heading_arrows(ax, x, y, b, every_mm=8.0, color=INK):
    s = np.concatenate([[0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
    marks = np.arange(every_mm / 2, s[-1], every_mm)
    for m in marks:
        i = int(np.searchsorted(s, m))
        u = np.array([math.cos(b[i]), math.sin(b[i])])
        ax.annotate('', xy=(x[i] + 5.5 * u[0], y[i] + 5.5 * u[1]), xytext=(x[i] - 1.5 * u[0], y[i] - 1.5 * u[1]),
                    arrowprops={'arrowstyle': '-|>', 'color': color, 'lw': .9, 'mutation_scale': 7.5,
                                'path_effects': [pe.withStroke(linewidth=2.2, foreground='white')]},
                    zorder=8)


def segments(ax, x, y, mask, **kw):
    """Plot only the runs of consecutive True entries."""
    idx = np.flatnonzero(mask)
    if not len(idx): return
    splits = np.flatnonzero(np.diff(idx) > 1) + 1
    for run in np.split(idx, splits):
        if len(run) > 1: ax.plot(x[run], y[run], **kw)
        else: ax.plot(x[run], y[run], marker='o', ms=kw.get('lw', 1) * 1.2, color=kw.get('color'),
                      ls='none', zorder=kw.get('zorder', 5))


def keyboard_panel(ax, r, letter, layout, *, ylabel):
    coverage_map(ax, layout, contour=False)
    draw_fixed(ax, layout, guard=False)
    v = r['valid'] & (r['phase'] == 'holding')
    x, y, b = r['x'][v] * 1e3, r['y'][v] * 1e3, r['beta'][v]
    lc = heading_path(ax, x, y, b)
    blocked = v & np.char.startswith(r['reason'].astype(str), 'boundary blocked')
    ax.plot(x[0], y[0], 'o', ms=4, mfc='white', mec=INK, mew=.8, zorder=9)
    ax.plot(x[-1], y[-1], 's', ms=3.6, mfc='white', mec=INK, mew=.8, zorder=9)
    axes_mm(ax, xlabel=False, ylabel=ylabel)
    dt = np.median(np.diff(r['t'])); hold_s = blocked.sum() * dt; dur = r['t'][v][-1] - r['t'][v][0]
    title(ax, letter, f'{LAYOUT_NAME[layout]}, keyboard')
    ax.text(.97, .03, f'{dur:.0f} s run, guard held {hold_s:.0f} s', transform=ax.transAxes, ha='right',
            fontsize=5.5, bbox={'fc': 'white', 'ec': 'none', 'alpha': .85, 'pad': 1}, zorder=12)
    return lc


LAPSE_SPACING_MM = 14.0      # arc length between magnet snapshots in the time-lapse composites
LAPSE_RADIUS_MM = 7.0        # radius of the image patch copied around the logged magnet centre


def _crop_xy(n):
    c = (np.arange(n) + .5) / n * 100 - 50
    return np.meshgrid(c, -c)                 # x right, y up (row 0 is y = +50 mm)


def timelapse_image(r, name):
    """Camera time-lapse from experiment frames only.

    Background: per-pixel median of 15 frames spread over the tracked interval (the moving magnet
    is at a different place in each, so the median removes it).  Snapshots: discs of radius
    LAPSE_RADIUS_MM copied from the frame at each selected time, centred on the logged pose.
    """
    v = np.flatnonzero(r['valid'] & (r['phase'] == 'tracking'))
    tt = r['t'][v]
    bg_times = np.linspace(tt[0], tt[-1], 15)
    stack = np.stack([grab_frame(r['folder'], t, f'{name}_bg')[..., :3] for t in bg_times])
    img = np.median(stack, axis=0)
    x, y = r['x'][v] * 1e3, r['y'][v] * 1e3
    arc = np.concatenate([[0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
    marks = np.arange(0, arc[-1], LAPSE_SPACING_MM).tolist() + [arc[-1]]
    picks = sorted({int(np.searchsorted(arc, m)) for m in marks})
    X, Y = _crop_xy(img.shape[0])
    for k in picks:
        i = v[min(k, len(v) - 1)]
        f = grab_frame(r['folder'], r['t'][i], f'{name}_snap')[..., :3]
        w = np.clip((LAPSE_RADIUS_MM - np.hypot(X - r['x'][i] * 1e3, Y - r['y'][i] * 1e3)) / 1.5, 0, 1)
        img = img * (1 - w[..., None]) + f * w[..., None]       # feathered edge
    return np.clip(img, 0, 1), [r['t'][v[min(k, len(v) - 1)]] for k in picks]


def auto_panel(ax, r, letter, layout, *, ylabel, direct=None):
    img, _ = timelapse_image(r, f'lapse_{layout}')
    ax.imshow(img, extent=EXTENT, interpolation='bilinear', zorder=0)
    ref = r['reference']
    ax.plot(ref['x'] * 1e3, ref['y'] * 1e3, color=REF_INK, lw=.8, ls=(0, (2.6, 1.6)), zorder=5,
            path_effects=[pe.Stroke(linewidth=2.0, foreground='white'), pe.Normal()])
    v = r['valid'] & (r['phase'] == 'tracking')
    x, y, b = r['x'][v] * 1e3, r['y'][v] * 1e3, r['beta'][v]
    # The snapshot chain is the measured motion; only the planned reference is drawn over it.
    gx, gy = r['goal'][0] * 1e3, r['goal'][1] * 1e3
    ax.plot(ref['x'][0] * 1e3, ref['y'][0] * 1e3, 'o', ms=4, mfc='white', mec=INK, mew=.8, zorder=9)
    ax.plot(gx, gy, '*', ms=7, color=INK, zorder=9)
    if direct is not None:
        d = direct['reference']
        dx, dy = d['x'] * 1e3, d['y'] * 1e3
        tin = d['tfw'].astype(bool); ok = d['lam'] >= 1 - 1e-9
        segments(ax, dx, dy, tin, color=REF_INK, lw=.8, ls=(0, (2.6, 1.6)), zorder=7,
                 path_effects=[pe.Stroke(linewidth=2.0, foreground='white'), pe.Normal()])
        segments(ax, dx, dy, ~tin, color=AMBER, lw=2.2, zorder=5, solid_capstyle='butt')
        dv = direct['valid'] & (direct['phase'] == 'tracking')
        mx, my = direct['x'][dv] * 1e3, direct['y'][dv] * 1e3
        ax.plot(mx, my, color=ORANGE, lw=1.3, zorder=6, path_effects=HALO)
        ax.plot(mx[-1], my[-1], 'x', ms=5.5, mew=1.3, color=ORANGE, zorder=10)
        loss_inset(ax, direct, layout)
    axes_mm(ax, ylabel=ylabel)
    title(ax, letter, f'{LAYOUT_NAME[layout]}: camera time-lapse')


def plan3d_panel(ax, r, layout, letter, direct=None):
    """3-D TFW (grey = not TFW-in) with the planned reference in (x, y, beta)."""
    P.voxel_panel(ax, P.load_tfw3d(layout), layout, '')
    ax.computed_zorder = False
    ax.set_rasterized(True)
    for t in list(ax.texts):
        if t.get_text() == r'$\beta$': t.remove()
    ax.set_zlabel(r'$\beta$ (°)', fontsize=6, labelpad=-2, rotation=0); ax.zaxis.set_rotate_label(False)
    def path3d(ref, mask, **kw):
        bx = np.degrees(ref['beta']) % 360
        cut = np.abs(np.diff(bx)) > 180          # break the line where beta wraps
        idx = np.flatnonzero(mask)
        for run in np.split(idx, np.flatnonzero((np.diff(idx) > 1) | cut[idx[:-1]]) + 1):
            if len(run) > 1:
                ax.plot(ref['x'][run] * 1e3, ref['y'][run] * 1e3, bx[run], zorder=40, **kw)
    ref = r['reference']
    path3d(ref, np.ones(len(ref['t']), bool), color=BLUE, lw=1.6,
           path_effects=[pe.Stroke(linewidth=2.8, foreground='white'), pe.Normal()])
    b0, b1 = np.degrees(ref['beta'][[0, -1]]) % 360
    ax.scatter([ref['x'][0] * 1e3], [ref['y'][0] * 1e3], [b0], s=10, c='white', edgecolors=INK,
               linewidths=.7, zorder=41, depthshade=False)
    ax.scatter([ref['x'][-1] * 1e3], [ref['y'][-1] * 1e3], [b1], s=26, c=INK, marker='*', zorder=41,
               depthshade=False)
    if direct is not None:
        dref = direct['reference']; tin = dref['tfw'].astype(bool)
        path3d(dref, tin, color=ORANGE, lw=1.1, ls=(0, (2.5, 1.4)))
        path3d(dref, ~tin, color=AMBER, lw=2.2)
    ax.set_title(f'({letter}) {LAYOUT_NAME[layout]}: planned path in 3-D TFW', loc='left', fontsize=6.8,
                 pad=-2, y=1.0)


def loss_inset(ax, r, layout):
    """TFW section at the heading of the last valid direct pose, around the loss point.

    The coverage background counts all headings; this shows the TFW at the magnet's own heading.
    """
    t = C.tfw(layout)
    v = np.flatnonzero(r['valid'] & (r['phase'] == 'tracking'))
    i = v[-1]
    x0, y0, b0 = r['x'][i] * 1e3, r['y'][i] * 1e3, r['beta'][i]
    level = t.beta_level(b0)
    lo, hi = level * 360 / t.n_levels, (level + 1) * 360 / t.n_levels
    ext = [e * 1e3 for e in t.extent]
    n = t.stack.shape[1]
    g = np.linspace(ext[0], ext[1], n + 1); g = .5 * (g[1:] + g[:-1])
    h = 4.0
    ax.plot([x0 - h, x0 + h, x0 + h, x0 - h, x0 - h], [y0 - h, y0 - h, y0 + h, y0 + h, y0 - h],
            color=INK, lw=.5, zorder=9)
    ins = ax.inset_axes([.6, .03, .37, .37])
    ins.set_zorder(20)
    ins.imshow(t.stack[level], origin='lower', extent=ext, interpolation='nearest', zorder=0,
               cmap=ListedColormap([UNKNOWN, INNER]), vmin=0, vmax=1)
    ins.contour(g, g, t.stack[level].astype(float), levels=[.5], colors=[INK], linewidths=.7,
                linestyles=[(0, (3, 1.2))], zorder=2)
    ref = r['reference']
    ins.plot(ref['x'] * 1e3, ref['y'] * 1e3, color=REF_INK, lw=.7, ls=(0, (2.6, 1.6)), zorder=3)
    tail = v[-8:]
    ins.plot(r['x'][tail] * 1e3, r['y'][tail] * 1e3, color=ORANGE, lw=1.1, marker='o', ms=2, zorder=4)
    ins.plot(x0, y0, 'x', ms=6, mew=1.4, color=ORANGE, zorder=5)
    ins.set(xlim=(x0 - h, x0 + h), ylim=(y0 - h, y0 + h)); ins.set_aspect('equal')
    ins.set_xticks([]); ins.set_yticks([])
    for sp in ins.spines.values(): sp.set_visible(True); sp.set_edgecolor(INK); sp.set_linewidth(.6)
    ins.text(.04, .96, f'TFW at $\\beta$ = {lo:.0f}–{hi:.0f}°', transform=ins.transAxes, fontsize=4.8,
             ha='left', va='top', color=INK, bbox={'fc': 'white', 'ec': 'none', 'alpha': .85, 'pad': .6})
    ins.text(.96, .04, 'tracking lost', transform=ins.transAxes, fontsize=4.8, ha='right', va='bottom',
             color=ORANGE, bbox={'fc': 'white', 'ec': 'none', 'alpha': .85, 'pad': .6})
    ax.indicate_inset_zoom(ins, edgecolor=INK, alpha=.6, lw=.5)


def ref_time_of_samples(r):
    """Reference time of each logged controller target (nearest reference sample)."""
    ref = r['reference']
    tg = np.array(r['target'], float)
    rx, ry, rb = ref['x'], ref['y'], ref['beta']
    out = np.full(len(tg), np.nan)
    for i, (x, y, b) in enumerate(tg):
        if not np.isfinite(x): continue
        db = np.angle(np.exp(1j * (rb - b)))
        j = np.argmin(((rx - x) / .001) ** 2 + ((ry - y) / .001) ** 2 + (db / .1) ** 2)
        out[i] = ref['t'][j]
    return out


def ref_time_of_poses(r):
    """Reference time of the reference pose nearest to each measured pose.

    Places every measured sample by where the magnet actually was, not by its controller target
    (the magnet can lead or lag the target).
    """
    ref = r['reference']
    rx, ry, rb = ref['x'], ref['y'], ref['beta']
    out = np.full(len(r['x']), np.nan)
    for i, (x, y, b) in enumerate(zip(r['x'], r['y'], r['beta'])):
        if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(b)): continue
        db = np.angle(np.exp(1j * (rb - b)))
        j = np.argmin(((rx - x) / .001) ** 2 + ((ry - y) / .001) ** 2 + (db / .1) ** 2)
        out[i] = ref['t'][j]
    return out


@lru_cache(maxsize=None)
def _tfw_distance(layout, level):
    """Distance (m) to the TFW boundary at one heading level: inside cells and outside cells."""
    t = C.tfw(layout)
    cell = (t.x_max - t.x_min) / t.raster
    m = t.stack[level]
    return distance_transform_edt(m) * cell, distance_transform_edt(~m) * cell


def tfw_margin(x, y, beta, layout):
    """Signed distance (mm) from (x, y) to the TFW boundary at heading beta: + inside, - outside."""
    t = C.tfw(layout)
    c = t.cell_of(x, y) if np.isfinite(x) and np.isfinite(y) and np.isfinite(beta) else None
    if c is None:
        return np.nan
    level = t.beta_level(beta)
    i, j = c
    din, dout = _tfw_distance(layout, level)
    return (din[j, i] if t.stack[level][j, i] else -dout[j, i]) * 1e3


def margin_panels(fig, cell, d):
    gs = cell.subgridspec(1, 3, wspace=.38)
    axm, axd, axi = (fig.add_subplot(gs[k]) for k in range(3))
    runs = [('auto_single', 'Single', BLUE), ('auto_gate', 'Gate', '#3f8f9f'),
            ('auto_triangle', 'Triangle', '#6c5aa0'), ('auto_direct', 'Direct', ORANGE)]
    for key, name, color in runs:
        r = d[key]; ref = r['reference']; layout = r['layout']
        v = r['valid'] & (r['phase'] == 'tracking')
        tr = ref_time_of_poses(r)
        ref_m = np.array([tfw_margin(x, y, b, layout) for x, y, b in zip(ref['x'], ref['y'], ref['beta'])])
        ref_c = ref['d_fixed'] * 1e3
        order = np.argsort(tr[v], kind='stable')
        tv, iv = tr[v][order], r['i_cmd'][v][order]
        if key != 'auto_direct':
            axm.plot(ref['t'], ref_m, color=color, lw=.9, label=name, zorder=4)
            axd.plot(ref['t'], ref_c, color=color, lw=.9, zorder=4)
            axi.plot(tv, iv, color=color, lw=.9, zorder=4)
            continue
        # Direct run: solid up to the loss of tracking, grey dashed where the magnet never got.
        lost = tr[v][-1]
        reached = ref['t'] <= lost
        for ax, yv in ((axm, ref_m), (axd, ref_c)):
            ax.plot(ref['t'][reached], yv[reached], color=color, lw=1.5, zorder=4,
                    label=name if ax is axm else None)
            ax.plot(ref['t'][~reached], yv[~reached], color='#8a929b', lw=1.1, ls=(0, (2.5, 1.5)), zorder=4,
                    label='Not reached' if ax is axm else None)
        axi.plot(tv, iv, color=color, lw=1.5, zorder=4)
        # Measured direct poses: margin and clearance where the magnet actually was.
        meas_m = np.array([tfw_margin(x, y, b, layout) for x, y, b in zip(r['x'], r['y'], r['beta'])])
        tail = v & (tr > lost - 1.2)
        out_m = tail & (meas_m < 0)
        for ax, yv in ((axm, meas_m), (axd, r['d_fixed'] * 1e3)):
            ax.plot(tr[tail], yv[tail], color='#7a3a18', lw=.8, marker='o', ms=1.8, zorder=6,
                    label='Measured' if ax is axm else None)
            ax.plot(tr[out_m], yv[out_m], 'o', ms=3.6, mfc='white', mec='#7a3a18', mew=.9, zorder=7)
        # Outside the TFW: from the first measured pose outside it (the magnet left before the
        # reference did) to the end of the reference segment outside it.
        out = ref['t'][ref_m < 0]
        start = min(out.min(), tr[out_m].min()) if out_m.any() else out.min()
        for ax in (axm, axd, axi):
            ax.axvspan(start, out.max(), color=AMBER, alpha=.18, lw=0, zorder=1)
            ax.axvline(lost, color=ORANGE, lw=.8, ls=(0, (2, 1.5)), zorder=2)
        last_m = meas_m[np.flatnonzero(v)[-1]]
        axm.annotate(f'{last_m:.2f} mm'.replace('-', '\u2212'), (lost, last_m), xytext=(4, -6), textcoords='offset points',
                     ha='left', va='top', fontsize=5.2, color='#7a3a18', zorder=8,
                     bbox={'fc': 'white', 'ec': 'none', 'alpha': .85, 'pad': .5})
        axm.annotate('tracking lost', (lost, -6.5), xytext=(-3, 0), textcoords='offset points',
                     ha='right', va='bottom', fontsize=5.4, color=ORANGE)
    axm.axhline(0, color=INK, lw=.7, zorder=3)
    axm.text(.25, .4, 'inside', fontsize=5.2, ha='left', va='bottom', color='#5c656e')
    axm.text(.25, -.5, 'outside', fontsize=5.2, ha='left', va='top', color='#5c656e')
    axm.set(ylim=(-7, 15), ylabel='TFW margin (mm)')
    axd.set(ylim=(10, 50), ylabel='Clearance (mm)')
    axi.axhline(15, color=INK, lw=.7, zorder=3)
    axi.text(11.8, 14.6, 'limit 15 A', fontsize=5.4, va='top', ha='right')
    axi.set(ylim=(0, 23), yticks=[0, 5, 10, 15], ylabel=r'max$_a|i_a|$ commanded (A)')
    for ax, letter, text in ((axm, 'g', 'TFW margin'),
                             (axd, 'h', 'Fixed-magnet clearance'),
                             (axi, 'i', 'Peak coil current')):
        ax.set_xlim(0, 12); ax.set_xlabel('Reference time (s)', labelpad=1, fontsize=6.8)
        ax.tick_params(length=2, pad=1.5, labelsize=6.2); ax.yaxis.label.set_size(6.8)
        title(ax, letter, text)
    axi.legend(*axm.get_legend_handles_labels(), loc='upper left', fontsize=5.1, frameon=False,
               handlelength=1.2, ncol=3, columnspacing=.6, borderaxespad=.15, handletextpad=.4)
    return axm


def grab_frame(folder, t, name):
    FRAMES.mkdir(parents=True, exist_ok=True)
    out = FRAMES / f'{name}_{t:.2f}s.png'     # the time is part of the key: no stale frames
    if not out.exists():
        root = Path(os.environ.get('ROS_CAN_SRC', 'ROS_CAN/src'))
        folder = Path(folder)
        if not folder.is_absolute() and folder.parts[:2] == ('ROS_CAN', 'src'):
            folder = root.joinpath(*folder.parts[2:])
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-ss', f'{t:.2f}', '-i', str(folder / 'ui.avi'),
                        '-vf', CROP, '-frames:v', '1', str(out)], check=True)
    return plt.imread(out)


def detect_moving_magnet(img):
    """Largest green blob in a cropped frame: (x mm, y mm, axis deg), from image pixels only.

    The moving magnet carries the green marker the OpenMV tracker follows; the fixed magnets are
    not green.  Used only where the controller log has no valid pose (after tracking was lost).
    """
    from scipy import ndimage
    a = img[..., :3].astype(float)
    if a.max() > 1.5: a /= 255.0
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    green = (g > r + .06) & (g > b + .02) & (a.max(2) - a.min(2) > .12)
    lab, n = ndimage.label(green)
    if n == 0:
        return None
    k = int(np.argmax(ndimage.sum(green, lab, range(1, n + 1)))) + 1
    ys, xs = np.nonzero(lab == k)
    h, w = green.shape
    x = (xs + .5) / w * 100 - 50; y = 50 - (ys + .5) / h * 100
    ev, vec = np.linalg.eigh(np.cov(np.vstack([x, y])))
    return float(x.mean()), float(y.mean()), math.degrees(math.atan2(vec[1, 1], vec[0, 1]))


def camera_panel(ax, r, t_frame, name, letter, text, layout, *, upto=None, note=None, arrow=None):
    img = grab_frame(r['folder'], t_frame, name)
    ax.imshow(img, extent=EXTENT, interpolation='bilinear', zorder=0)
    v = r['valid'] & (r['t'] <= (upto if upto is not None else t_frame))
    v &= np.isin(r['phase'], ['holding', 'tracking'])
    x, y = r['x'][v] * 1e3, r['y'][v] * 1e3
    keep = np.concatenate([[True], np.hypot(np.diff(x), np.diff(y)) < 4.0])
    ax.plot(np.where(keep, x, np.nan), np.where(keep, y, np.nan), color='#ffd23f', lw=.9, zorder=3, alpha=.95)
    ax.plot(x[-1], y[-1], 'x', ms=5, mew=1.2, color='#ff5a36', zorder=4)
    # Dotted ellipse on the moving magnet when its pose is still tracked at the frame time.
    j = np.flatnonzero(r['valid'])
    j = j[np.argmin(abs(r['t'][j] - t_frame))]
    # Logged pose when tracked at the frame time; otherwise the green blob detected in the frame.
    pose = ((r['x'][j] * 1e3, r['y'][j] * 1e3, math.degrees(r['beta'][j]))
            if abs(r['t'][j] - t_frame) < .3 else detect_moving_magnet(img))
    if pose is not None:
        ax.add_patch(Ellipse(pose[:2], 24, 14, angle=pose[2],
                             fc='none', ec=FREE_RING, lw=1.0, ls=(0, (1.2, 1.2)), zorder=5))
    if arrow is not None:
        ax.annotate('', xy=arrow[1], xytext=arrow[0],
                    arrowprops={'arrowstyle': '-|>', 'color': 'white', 'lw': 1.0, 'mutation_scale': 7}, zorder=5)
    ax.set(xlim=(-40, 40), ylim=(-38, 42)); ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(True); sp.set_edgecolor('#9aa3ab'); sp.set_linewidth(.5)
    title(ax, letter, text, fontsize=6.8)
    ax.text(.97, .03, f'{t_frame:.0f} s', transform=ax.transAxes, ha='right', va='bottom', fontsize=6,
            color='white', bbox={'fc': '#2a3038', 'ec': 'none', 'pad': 1.2}, zorder=6)
    if note:
        ax.text(.03, .97, note, transform=ax.transAxes, ha='left', va='top', fontsize=5.4, color=INK,
                bbox={'fc': 'white', 'ec': 'none', 'alpha': .85, 'pad': 1}, zorder=6)


def fig_hardware():
    d = load_trials()
    fig = plt.figure(figsize=(7.16, 6.7))
    gs = fig.add_gridspec(3, 1, left=.07, right=.985, top=.94, bottom=.03, hspace=.3,
                          height_ratios=[.86, 1, .62])
    row1 = gs[0].subgridspec(1, 3, wspace=.02)
    for k, layout in enumerate(LAYOUTS):
        ax = fig.add_subplot(row1[k], projection='3d')
        plan3d_panel(ax, d[f'auto_{layout}'], layout, 'abc'[k],
                     direct=d['auto_direct'] if layout == 'single' else None)
    row2 = gs[1].subgridspec(1, 3, wspace=.12)
    for k, layout in enumerate(LAYOUTS):
        ax = fig.add_subplot(row2[k])
        auto_panel(ax, d[f'auto_{layout}'], 'def'[k], layout, ylabel=k == 0,
                   direct=d['auto_direct'] if layout == 'single' else None)
    row3 = gs[2].subgridspec(1, 2, wspace=.12, width_ratios=[3.25, 1])
    margin_panels(fig, row3[0], d)
    camera_panel(fig.add_subplot(row3[1]), d['auto_direct'], 13, 'direct_attach', 'j', 'Direct: attachment',
                 'single', upto=13)
    handles = [Line2D([], [], color=BLUE, lw=1.6, label='Planned (3-D)'),
               Line2D([], [], color=REF_INK, lw=.8, ls=(0, (2.6, 1.6)), label='Reference (2-D)'),
               Line2D([], [], color=ORANGE, lw=1.3, label='Direct: measured'),
               Line2D([], [], color=AMBER, lw=2.2, label='Direct ref.: outside TFW'),
               Patch(fc='#b9bfc6', ec='#8a929b', lw=.4, label='3-D: not TFW-in')]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .999), ncol=6, frameon=False,
               fontsize=5.8, handlelength=1.5, columnspacing=.8)
    export(fig, 'cs1_hardware')


def export(fig, stem):
    for ext in ('pdf', 'png'):
        fig.savefig(OUT / f'{stem}.{ext}', facecolor='white', bbox_inches='tight', pad_inches=.03)
    plt.close(fig)
    print('Exported', stem, flush=True)


def main():
    P.science_style()
    fig_workspace()
    fig_hardware()


if __name__ == '__main__':
    main()
