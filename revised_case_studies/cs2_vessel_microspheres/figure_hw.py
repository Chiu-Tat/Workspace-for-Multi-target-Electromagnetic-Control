"""Case Study 2 hardware figure: camera time-lapse, joint TFW, and telemetry for three trials.

Run from the repository root after ``hardware`` has written trials.json:
  .venv/bin/python -B -m revised_case_studies.cs2_vessel_microspheres.figure_hw

Every element is from the trial records or computed by Python:
  camera background  per-pixel median of 15 ui.avi frames (removes the moving spheres and overlays)
  sphere markers     logged positions, mapped to pixels by the fitted registration (hardware.register)
  centrelines        registered trunk centrelines (publication.Geometry), mapped the same way
  TFW, lambda = 1    saved nominal TFW boxes and the cached force-margin grid
  traces             telemetry.csv; times are logged host times
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import patheffects as pe
from matplotlib.collections import PolyCollection
from matplotlib.image import BboxImage
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch
from matplotlib.transforms import Affine2D, Bbox, TransformedBbox
import numpy as np

from analysis.sections_67.common import plot_pole_sphere
from revised_case_studies.cs2_vessel_microspheres import hardware as H, publication as P

OUT = P.SAVED / 'hardware'
INK = '#252a30'
GREEN, RED = '#1f8a4c', '#c0392b'          # camera colours: green RA sphere, red RV sphere
INNER, UNKNOWN, OUTER = P.INNER, P.UNKNOWN, P.CLASS_COLORS['outside']
REF, MEAS, EXIT = '#252a30', '#215f98', '#d99a2b'
LAMC = '#a64242'
N_SNAP = 6                                  # sphere snapshots per time-lapse
HALO = [pe.Stroke(linewidth=1.8, foreground='white'), pe.Normal()]


def load():
    d = json.loads((OUT / 'trials.json').read_text())
    for r in d.values():
        for k in ('t', 'sep_mm', 'i_cmd', 'i_meas', 'lam'):
            r[k] = np.array([np.nan if v is None else v for v in r[k]], float)
        for k in ('green_xyz', 'red_xyz', 'arc', 'target_arc', 's'):
            r[k] = np.array([[np.nan if v is None else v for v in row] for row in r[k]], float)
        for k in ('valid', 'follow', 'model_in'):
            r[k] = np.array(r[k], bool)
        if 'reference' in r:
            for k in r['reference']:
                r['reference'][k] = np.array(r['reference'][k], float if k != 'model_in' else bool)
    return d


# Manual trial: only the segment in the edited video "cs2 over under failure.mov" (2x speed, 9.24 s).
# Its frames were matched to ui.avi frames by image correlation (r = 0.86-0.90); a linear fit gives
# trial time = 76.72 s + 2.026 x edited time (RMS 0.25 s), i.e. the window below.
MANUAL_WINDOW_S = (76.72, 95.45)


def active(r):
    """Samples shown: route following (guided) or the edited-video window (manual)."""
    if r['kind'] == 'guided':
        return r['valid'] & r['follow']
    return r['valid'] & (r['t'] >= MANUAL_WINDOW_S[0]) & (r['t'] <= MANUAL_WINDOW_S[1])


def time_axis(r):
    t = r['t']
    if r['kind'] == 'guided':
        return t - t[np.flatnonzero(r['follow'])[0]]
    return t - MANUAL_WINDOW_S[0]                 # manual: time from the start of the video segment


def passing_index(r):
    """First sample where the two spheres swap their longitudinal (y) order."""
    m = np.flatnonzero(active(r))
    dy = r['green_xyz'][m, 1] - r['red_xyz'][m, 1]
    k = np.flatnonzero(np.sign(dy[1:]) != np.sign(dy[:-1]))
    return int(m[k[0] + 1]) if len(k) else None


def to_px(reg, xy_mm):
    A = np.asarray(reg['affine_mm_to_px'])
    xy = np.atleast_2d(xy_mm)
    return np.column_stack([xy, np.ones(len(xy))]) @ A


# ------------------------------------------------------------------ panels
def timelapse_panel(ax, r, letter, title, background):
    reg = r['registration']
    m = np.flatnonzero(active(r))
    ax.imshow(background, interpolation='bilinear', zorder=0)
    g = H.geometry(r['variant'])
    ss = np.linspace(0, 1, 200)
    for k, col in enumerate((GREEN, RED)):
        line = np.array([g.lines[k].point(s)[:2] for s in ss]) * 1e3
        uv = to_px(reg, line)
        ax.plot(uv[:, 0], uv[:, 1], color=col, lw=.6, ls=(0, (3, 2)), alpha=.75, zorder=2)
        for s, endpoint in zip((0, 1), uv[[0, -1]]):
            ax.plot(*endpoint, marker='s', ms=2.6, mfc=col, mec='white', mew=.55, zorder=10)
            ax.annotate(rf'$s_{k + 1}={s}$', endpoint, xytext=(2.7, 0),
                        textcoords='offset points', ha='left', va='center',
                        fontsize=5.1, color=col, path_effects=HALO, zorder=11)
    for xyz, col in ((r['green_xyz'], GREEN), (r['red_xyz'], RED)):
        uv = to_px(reg, xyz[m, :2] * 1e3)
        ax.plot(uv[:, 0], uv[:, 1], color=col, lw=.9, alpha=.85, zorder=3, path_effects=HALO)
    px_mm = math.sqrt(abs(np.linalg.det(np.asarray(reg['affine_mm_to_px'])[:2])))
    snaps = m[np.linspace(0, len(m) - 1, N_SNAP).astype(int)]
    for n, i in enumerate(snaps, 1):
        for xyz, col in ((r['green_xyz'], GREEN), (r['red_xyz'], RED)):
            u, v = to_px(reg, xyz[i, :2] * 1e3)[0]
            ax.add_patch(plt.Circle((u, v), 1.5 * px_mm, fc=col, ec='white', lw=.5, zorder=5))
            ax.text(u + 2.6 * px_mm, v, str(n), fontsize=4.6, color=col, ha='left', va='center', zorder=6,
                    fontweight='bold', path_effects=HALO)
    ip = passing_index(r)
    if ip is not None:
        for xyz in (r['green_xyz'], r['red_xyz']):
            u, v = to_px(reg, xyz[ip, :2] * 1e3)[0]
            ax.add_patch(plt.Circle((u, v), 3.2 * px_mm, fc='none', ec=INK, lw=.8, ls=(0, (1.5, 1)), zorder=6))
    h, w = background.shape[:2]
    ax.set_xlim(0, w); ax.set_ylim(h, 0)
    xl, yl = ax.get_xlim(), ax.get_ylim()          # 10 mm scale bar, bottom left of the view
    bx, by = xl[0] + .06 * (xl[1] - xl[0]), yl[0] - .05 * (yl[0] - yl[1])
    ax.plot([bx, bx + 10 * px_mm], [by, by], color='white', lw=2, solid_capstyle='butt', zorder=7)
    ax.text(bx + 5 * px_mm, by - .02 * (yl[0] - yl[1]), '10 mm', color='white', fontsize=5, ha='center',
            va='bottom', zorder=7)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(True); sp.set_edgecolor('#9aa3ab'); sp.set_linewidth(.5)
    t = time_axis(r)
    span = t[snaps[-1]] - t[snaps[0]]
    ax.set_title(f'({letter}) {title}', loc='left', fontsize=6.8, pad=2.5)
    ax.text(.03, .918, f'1–{N_SNAP}: every {span / (N_SNAP - 1):.0f} s', transform=ax.transAxes, fontsize=5,
            ha='left', va='top', color=INK, bbox={'fc': 'white', 'ec': 'none', 'alpha': .85, 'pad': .8}, zorder=8)


def microsphere_callout(fig, ax, trial):
    """Place a dimensioned sphere schematic in the left margin of panel (a)."""
    width_pt, height_pt = fig.get_size_inches() * 72
    points = Affine2D().scale(1 / width_pt, 1 / height_pt) + fig.transFigure
    left_pt = ax.get_position().x0 * width_pt
    artery = H.geometry(trial['variant']).lines[0]
    tip_px = to_px(trial['registration'], artery.point(0.45)[:2] * 1e3)[0]
    tip_x, cy = points.inverted().transform(ax.transData.transform(tip_px))
    radius = 9
    cx = left_pt - radius - 3.5
    left, right = cx - radius, cx + radius

    render = plt.figure(figsize=(1, 1), dpi=400)
    sphere = render.add_axes([0, 0, 1, 1], projection='3d', proj_type='ortho')
    plot_pole_sphere(sphere, np.zeros(3), np.array([0., 1., 0.]), 1.0)
    sphere.set(xlim=(-1.05, 1.05), ylim=(-1.05, 1.05), zlim=(-1.05, 1.05))
    sphere.set_box_aspect((1, 1, 1))
    sphere.view_init(elev=15, azim=-30)
    sphere.set_axis_off()
    bounds = Bbox.from_extents(left, cy - radius, right, cy + radius)
    image = BboxImage(TransformedBbox(bounds, points), interpolation='lanczos', zorder=11)
    image.set_data(crop_figure_rgba(render))
    fig.add_artist(image)

    arrow = FancyArrowPatch((cx + radius + 1.3, cy), (tip_x, cy), arrowstyle='-|>',
                            shrinkA=0, shrinkB=0, mutation_scale=7, linewidth=.65, color=INK,
                            transform=points, zorder=10, path_effects=HALO)
    fig.add_artist(arrow)

    dim_y = cy - radius - 4
    for x in (left, right):
        fig.add_artist(Line2D([x, x], [dim_y - 1.8, dim_y + 2.5], linewidth=.45,
                              color=INK, transform=points, zorder=11))
    fig.add_artist(FancyArrowPatch((left, dim_y), (right, dim_y), arrowstyle='<->',
                                  shrinkA=0, shrinkB=0, mutation_scale=5, linewidth=.5,
                                  color=INK, transform=points, zorder=11))
    fig.text(cx, dim_y - 2.7, '3 mm', transform=points, ha='center', va='top',
             fontsize=5.5, color=INK, zorder=11)


def s_limits(r):
    """Full joint space for every run."""
    return dict(xlim=(0, 1), ylim=(0, 1), xticks=[0, .5, 1], yticks=[0, .5, 1])


TRUNK_OUTER_R_MM = 4.0     # main trunks: 8 mm outer-diameter pipes (geometry JSON, main_trunk_pipe)


def _tube(ves, branch, radius_mm=None, n=48, m=14):
    """Tube surface (mm) around a branch centreline of the vessel geometry."""
    ss = np.linspace(0, 1, n)
    c = np.array([ves.centerline(branch, s_) for s_ in ss]) * 1e3
    rad = np.array([radius_mm if radius_mm else ves.radius(branch, s_) * 1e3 for s_ in ss])
    t = np.gradient(c, axis=0); t /= np.linalg.norm(t, axis=1)[:, None]
    ref = np.array([0, 0, 1.0])
    u = np.cross(t, ref); u /= np.maximum(np.linalg.norm(u, axis=1), 1e-9)[:, None]
    w = np.cross(t, u)
    th = np.linspace(0, 2 * np.pi, m)
    return (c[:, None, :] + rad[:, None, None] * (np.cos(th)[None, :, None] * u[:, None, :]
                                                 + np.sin(th)[None, :, None] * w[:, None, :]))


def vessel3d_image(r):
    """Render vessel3d_panel off-screen and crop it to its content (RGBA array)."""
    f3 = plt.figure(figsize=(3.2, 2.4), dpi=400)
    a = f3.add_axes([0, 0, 1, 1], projection='3d')
    vessel3d_panel(a, r, '')
    for t in list(a.texts): t.remove()
    return crop_figure_rgba(f3)


def crop_figure_rgba(fig):
    from io import BytesIO
    from PIL import Image

    with BytesIO() as buffer:
        fig.savefig(buffer, format='png', transparent=True)
        plt.close(fig)
        with Image.open(buffer) as png:
            image = png.convert('RGBA')
            return np.asarray(image.crop(image.getbbox()))


def vessel3d_panel(ax, r, letter):
    """3-D vessel networks from the vessel geometry (styled after the earlier CAD view).

    Solid tubes around the geometry centrelines (8 mm outer diameter on the trunks, branch radii from the
    geometry), the spheres at their logged closest approach, ortho projection with a mm grid.
    """
    from matplotlib.colors import LightSource
    ves = H.geometry(r['variant']).pair.vessel
    ls = LightSource(azdeg=315, altdeg=45)
    for b in ves.branches:
        col = GREEN if b.startswith('RA') else RED
        X = _tube(ves, b, TRUNK_OUTER_R_MM if b in ('RA', 'RV') else None, n=60, m=18)
        ax.plot_surface(X[..., 0], X[..., 1], X[..., 2], color=col, linewidth=0, shade=True, lightsource=ls,
                        antialiased=False, rasterized=True)
    m = active(r); idx = np.flatnonzero(m); ic = idx[np.nanargmin(r['sep_mm'][idx])]
    for xyz in (r['green_xyz'], r['red_xyz']):
        x, y, z = xyz[ic] * 1e3
        ax.scatter([x], [y], [z + 4.2], s=10, color='#ffd24b', edgecolors='#5b4a12', linewidths=.4,
                   depthshade=False, zorder=20)
    allc = np.concatenate([np.array([ves.centerline(b, s_) for s_ in np.linspace(0, 1, 20)]) for b in ves.branches]) * 1e3
    lo, hi = allc.min(0) - 5, allc.max(0) + 5
    ax.set(xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]), zlim=(lo[2], hi[2]))
    ax.set_box_aspect(hi - lo, zoom=1.0)                  # z (depth) points up
    ax.set_proj_type('ortho'); ax.view_init(elev=28, azim=-65)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((1, 1, 1, .85)); axis.pane.set_edgecolor('#c9cdd2')
        axis._axinfo['grid'].update(color='#d5d8dc', linewidth=.3)
        axis.line.set_linewidth(.4); axis.line.set_color('#9a9fa5')
    ax.set_xticks([-40, 0, 40]); ax.set_yticks([-40, 0, 40]); ax.set_zticks([])
    ax.set_xticklabels([]); ax.set_yticklabels([])
    ax.tick_params(length=0)
    ax.patch.set_alpha(0)
    ax.text2D(.5, 1.18, '3-D phantom', transform=ax.transAxes, fontsize=4.8, color=INK, va='top', ha='center')
    ax.text2D(.06, 1.04, 'Artery', transform=ax.transAxes, fontsize=4.2, color=GREEN, va='top', fontweight='bold')
    ax.text2D(.94, 1.04, 'Vein', transform=ax.transAxes, fontsize=4.2, color=RED, va='top', ha='right',
              fontweight='bold')


def tfw_panel(ax, r, letter):
    g = H.geometry(r['variant'])
    for cls, col in (('undetermined', UNKNOWN), ('inside', INNER), ('outside', OUTER)):
        verts = [[(a, c), (b, c), (b, d), (a, d)] for a, b, c, d in g.map[cls]]
        if verts:
            ax.add_collection(PolyCollection(verts, fc=col, ec='none', rasterized=True, zorder=0))
    fm = json.loads((P.OUT / 'force_margin.json').read_text())['maps'][r['variant']]
    grid = np.array(fm['grid']); lam = np.array(fm['lambda_s1_s2'])
    ax.contour(grid, grid, lam.T, levels=[1], colors=[LAMC], linewidths=.7, linestyles=[(0, (1.5, 1))], zorder=2)
    m = active(r); s = r['s']
    if 'reference' in r:
        rs = r['reference']['s']
        ax.plot(rs[:, 0], rs[:, 1], color=REF, lw=.8, ls=(0, (3, 1.6)), zorder=3, path_effects=HALO)
        ax.plot(*rs[0], 'o', ms=3.4, mfc='white', mec=INK, mew=.7, zorder=6)
        ax.plot(*rs[-1], 's', ms=3.2, mfc='white', mec=INK, mew=.7, zorder=6)
    idx = np.flatnonzero(m)
    if r['kind'] == 'guided':
        ax.plot(s[idx, 0], s[idx, 1], color=MEAS, lw=1.1, zorder=4, path_effects=HALO)
    else:
        inn = m & r['model_in']; out = m & ~r['model_in']
        ax.plot(s[idx, 0], s[idx, 1], color=MEAS, lw=.9, zorder=4, path_effects=HALO)
        ax.plot(s[out, 0], s[out, 1], '.', ms=2.4, color=EXIT, zorder=5)
        bad = m & (r['lam'] < 1)
        ax.plot(s[bad, 0], s[bad, 1], 'x', ms=3, mew=.8, color=LAMC, zorder=6)
        ax.plot(*s[idx[0]], 'o', ms=3.4, mfc='white', mec=INK, mew=.7, zorder=6)
        ax.plot(*s[idx[-1]], 's', ms=3.2, mfc='white', mec=INK, mew=.7, zorder=6)
    ip = passing_index(r)
    if ip is not None:
        ax.plot(*s[ip], 'D', ms=3.6, mfc='white', mec=INK, mew=.8, zorder=7)
    ic = idx[np.nanargmin(r['sep_mm'][idx])]
    ax.plot(*s[ic], '*', ms=6, color=INK, zorder=7)
    ax.set(**s_limits(r))
    ax.set_xlabel('$s_1$ (artery)', fontsize=6.2, labelpad=1); ax.set_ylabel('$s_2$ (vein)', fontsize=6.2, labelpad=0)
    ax.tick_params(labelsize=5.6, length=1.6, pad=1)
    ax.set_aspect('equal')
    ax.set_title(f'({letter}) Joint TFW', loc='left', fontsize=6.8, pad=2.5)


def margin_panel(ax, r, letter):
    """Nominal force-margin map lambda(s1, s2) with the hardware paths."""
    fm = json.loads((P.OUT / 'force_margin.json').read_text())
    mp = fm['maps'][r['variant']]
    grid = np.array(mp['grid']); lam = np.array(mp['lambda_s1_s2'])
    mesh = ax.pcolormesh(grid, grid, lam.T, cmap=P.MARGIN_CMAP, vmin=0, vmax=3, shading='nearest',
                         rasterized=True, zorder=0)
    ax.contour(grid, grid, lam.T, levels=[1], colors=[LAMC], linewidths=.9, zorder=2)
    m = active(r); s = r['s']; idx = np.flatnonzero(m)
    white = [pe.Stroke(linewidth=2.4, foreground='white'), pe.Normal()]
    if 'reference' in r:
        rs = r['reference']['s']
        ax.plot(rs[:, 0], rs[:, 1], color=REF, lw=.8, ls=(0, (3, 1.6)), zorder=3, path_effects=white)
        ax.plot(*rs[0], 'o', ms=3.4, mfc='white', mec=INK, mew=.7, zorder=6)
        ax.plot(*rs[-1], 's', ms=3.2, mfc='white', mec=INK, mew=.7, zorder=6)
    else:
        ax.plot(*s[idx[0]], 'o', ms=3.4, mfc='white', mec=INK, mew=.7, zorder=6)
        ax.plot(*s[idx[-1]], 's', ms=3.2, mfc='white', mec=INK, mew=.7, zorder=6)
    ax.plot(s[idx, 0], s[idx, 1], color=INK, lw=1.0, zorder=4, path_effects=white)
    frac = fm['consistency'][r['variant']]['fraction_below_one'] * 100
    ax.text(.03, .03, rf'$\lambda<1$ on {frac:.1f}%', transform=ax.transAxes, fontsize=5, zorder=8,
            bbox={'fc': 'white', 'ec': 'none', 'alpha': .85, 'pad': .8})
    ax.set(**s_limits(r))
    ax.set_xlabel('$s_1$ (artery)', fontsize=6.2, labelpad=1); ax.set_yticklabels([])
    ax.tick_params(labelsize=5.6, length=1.6, pad=1)
    ax.set_aspect('equal')
    ax.set_title(f'({letter}) Force margin', loc='left', fontsize=6.8, pad=2.5)
    return mesh


def trace_panels(fig, cell, r, letter):
    # Reserve room above the telemetry title and below the time axis.
    stack = cell.subgridspec(3, 1, hspace=0, height_ratios=[.075, .85, .075])[1]
    gs = stack.subgridspec(3, 1, hspace=.16, height_ratios=[1.15, 1, .8])
    axs = [fig.add_subplot(gs[k]) for k in range(3)]
    t = time_axis(r); m = active(r); idx = np.flatnonzero(m)
    ip = passing_index(r); ic = idx[np.nanargmin(r['sep_mm'][idx])]
    a = r['arc'] * 1e3
    ax = axs[0]
    for k, col, name in ((0, GREEN, 'Artery'), (1, RED, 'Vein')):
        ax.plot(t[idx], a[idx, k], color=col, lw=1, label=name)
        if r['kind'] == 'guided':
            ax.plot(t[idx], r['target_arc'][idx, k] * 1e3, color=col, lw=.6, ls=(0, (2, 1.5)), alpha=.8)
    ax.set_ylabel('Arc (mm)', fontsize=6, labelpad=1)
    ax.legend(loc='lower right', bbox_to_anchor=(1, 1.0), fontsize=5, frameon=False,
              ncol=2, handlelength=1.1, columnspacing=.6, borderaxespad=0)
    ax = axs[1]
    if r['kind'] == 'guided':
        ax.plot(t[idx], r['sep_mm'][idx], color=INK, lw=.9)
        ax.plot(t[ic], r['sep_mm'][ic], '*', ms=5.5, color=INK, zorder=5)
        ax.annotate(f'{r["sep_mm"][ic]:.1f} mm', (t[ic], r['sep_mm'][ic]), xytext=(0, 9),
                    textcoords='offset points', fontsize=5, ha='center', va='bottom',
                    bbox={'fc': 'white', 'ec': 'none', 'alpha': .9, 'pad': .5})
        ax.set_ylabel('Separation\n(mm)', fontsize=6, labelpad=1)
    else:
        lam = r['lam']
        ax.plot(t[idx], lam[idx], color=INK, lw=.9)
        ax.axhline(1, color=LAMC, lw=.7, ls=(0, (1.5, 1)))
        ax.set(ylim=(0, max(5.5, np.nanmax(lam[idx]) * 1.05)), yticks=(0, 2, 4))
        ax.set_ylabel(r'Margin $\lambda$', fontsize=6, labelpad=1)
    ax = axs[2]
    ax.plot(t[idx], r['i_cmd'][idx], color='#4b5d73', lw=.7, label='Command')
    ax.set_ylim(0, 16); ax.set_yticks([0, 7, 14]); ax.set_ylabel('Peak (A)', fontsize=6, labelpad=1)
    for axx in axs:
        axx.tick_params(labelsize=5.6, length=1.6, pad=1)
        axx.set_xlim(0 if r['kind'] == 'manual' else t[idx[0]], t[idx[-1]])
        if ip is not None:
            axx.axvline(t[ip], color=INK, lw=.6, ls=(0, (2, 1.5)), zorder=1)
    if r['kind'] == 'manual':
        pause = np.array([x.startswith('Tracking pause') for x in r['reason']])
        out = m & ~r['model_in']
        t_exit = t[np.flatnonzero(out)[0]] if out.any() else None
        bad = m & (r['lam'] < 1)
        t_bad = t[np.flatnonzero(bad)[0]] if bad.any() else None
        for axx in axs:
            if t_exit is not None: axx.axvline(t_exit, color=EXIT, lw=.9, zorder=2)
            if t_bad is not None: axx.axvline(t_bad, color=LAMC, lw=.9, zorder=2)
        axs[1].text(t_exit - .8, 2.6, 'leaves\nTFW', fontsize=5, color='#9a6a12', ha='right', va='center',
                    bbox={'fc': 'white', 'ec': 'none', 'alpha': .85, 'pad': .5})
        axs[1].text(t_bad - .3, .35, r'$\lambda<1$', fontsize=5, color=LAMC, ha='right', va='bottom',
                    bbox={'fc': 'white', 'ec': 'none', 'alpha': .85, 'pad': .5})
    if ip is not None:
        axs[0].text(t[ip], .04, 'pass', transform=axs[0].get_xaxis_transform(), fontsize=5, ha='center',
                    va='bottom', color=INK, zorder=6, bbox={'fc': 'white', 'ec': 'none', 'alpha': .9, 'pad': .5})
    for axx in axs[:2]: axx.set_xticklabels([])
    axs[2].set_xlabel('Time following reference (s)' if r['kind'] == 'guided' else 'Time in video segment (s)',
                      fontsize=6.2, labelpad=1)
    axs[0].set_title(f'({letter}) Telemetry', loc='left', fontsize=6.8, pad=2.5)
    return axs


def figure():
    P.style()
    d = load()
    row_in, gap_in = 1.696, .34
    top_in, bottom_in = .555, .333
    height = 3 * row_in + 2 * gap_in + top_in + bottom_in
    fig = plt.figure(figsize=(7.16, height))
    rows = fig.add_gridspec(3, 1, left=.06, right=.985, top=1 - top_in / height,
                           bottom=bottom_in / height, hspace=gap_in / row_in)
    heads = ['A  Side-by-side vessels, guided', 'B  Over-under vessels, guided',
             'C  Over-under vessels, manual control without the TFW guard']
    for k, key in enumerate(('side_by_side', 'over_under', 'manual')):
        r = d[key]
        sub = rows[k].subgridspec(1, 6, wspace=.2, width_ratios=[.78, .82, .82, .04, .1, 1.25])
        with np.load(OUT / f'background_{key}.npz') as images:
            background = images['background']
        ax = fig.add_subplot(sub[0]); timelapse_panel(ax, r, 'aei'[k], 'Camera time-lapse', background)
        if k == 0:
            microsphere_callout(fig, ax, r)
        ax.add_patch(FancyBboxPatch((.5, .108), .485, .277, boxstyle='round,pad=0,rounding_size=.02',
                                    transform=ax.transAxes, fc='white', ec='#9aa3ab', lw=.5, alpha=.94,
                                    zorder=9))
        img3 = vessel3d_image(r)
        ia = ax.inset_axes([.515, .112, .455, .178]); ia.set_zorder(10)
        ia.imshow(img3, interpolation='lanczos'); ia.set_axis_off()
        ax.text(.515, .375, '3-D phantom', transform=ax.transAxes, fontsize=4.6, ha='left', va='top',
                color=INK, zorder=11)
        ax.text(.515, .318, 'Artery', transform=ax.transAxes, fontsize=4.2, color=GREEN, va='top',
                fontweight='bold', zorder=11)
        ax.text(.72, .318, 'Vein', transform=ax.transAxes, fontsize=4.2, color=RED, va='top',
                fontweight='bold', zorder=11)
        ax2 = fig.add_subplot(sub[1]); tfw_panel(ax2, r, 'bfj'[k])
        ax3 = fig.add_subplot(sub[2]); mesh = margin_panel(ax3, r, 'cgk'[k])
        cax = fig.add_subplot(sub[3])
        q, c = ax3.get_position(), cax.get_position()
        cax.set_position([c.x0 - .008, q.y0, c.width, q.height])
        cb = fig.colorbar(mesh, cax=cax, ticks=[0, 1, 2, 3])
        cb.ax.tick_params(labelsize=4.8, length=1.4, pad=.6); cb.outline.set_linewidth(.4)
        cb.ax.axhline(1, color=LAMC, lw=1)
        cb.set_label(r'$\lambda$', fontsize=6, labelpad=1, rotation=0)
        trace_panels(fig, sub[5], r, 'dhl'[k])
        b = ax.get_position()
        fig.text(.012, b.y1 + .16 / height, heads[k], fontsize=7.6, fontweight='bold',
                 color=INK, va='bottom')
    handles = [Patch(fc=INNER, label='TFW-in'), Patch(fc=OUTER, label='TFW-out'), Patch(fc=UNKNOWN, label='Unresolved'),
               Line2D([], [], color=LAMC, lw=.8, ls=(0, (1.5, 1)), label=r'$\lambda=1$'),
               Line2D([], [], color=REF, lw=.8, ls=(0, (3, 1.6)), label='Reference'),
               Line2D([], [], color=MEAS, lw=1.1, label='Measured pair'),
               Line2D([], [], ls='none', marker='.', ms=5, color=EXIT, label='Outside TFW'),
               Line2D([], [], ls='none', marker='x', ms=4, color=LAMC, label=r'$\lambda<1$'),
               Line2D([], [], ls='none', marker='D', ms=4, mfc='white', mec=INK, label='Passing'),
               Line2D([], [], ls='none', marker='*', ms=6, color=INK, label='Closest approach')]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, 1.0), ncol=10, frameon=False,
               fontsize=5.4, handlelength=1.2, columnspacing=.55)
    for ext in ('pdf', 'png'):
        f = OUT / f'cs2_hardware.{ext}'
        fig.savefig(f, facecolor='white', bbox_inches='tight', pad_inches=.03, dpi=300)
    plt.close(fig)
    print('Exported cs2_hardware', flush=True)


if __name__ == '__main__':
    figure()
