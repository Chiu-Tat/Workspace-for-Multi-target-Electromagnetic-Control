#!/usr/bin/env python3
# Run from the repository root:
#   .venv/bin/python -B -m revised_case_studies.cs1_cylinder_interaction.presentation [--only NAME ...]
"""Case Study 1 figures and animations for the slide deck.

All maps are the fresh 3-D TFWs with the corrected dBy/dy coefficient
(``common.tfw(layout)``, results/cs1/tfw3d_corrected) for one, two (gate) and
three (triangle) fixed magnets; this script computes no TFW.  The
free magnet moves only forward, backward or turns in place (``nonholonomic``).

Outputs in ``results/cs1/presentation``:

  tfw3d_single.png, tfw3d_gate.png, tfw3d_triangle.png
                       3-D TFW (x, y, beta) with four heading slices
  demo_single.gif      scripted drive (forward, turn, reverse) in closed loop
  plans.png            planned paths for the three layouts (3-D and top view)
  plans.gif            the three planned runs in closed loop
  crossing.gif         planned path vs straight crossing (one fixed magnet)
  summary.json         numbers quoted on the slides
  trajectory_<layout>_planned.json, trajectory_<layout>_straight.json
                       planned and straight references, shared with the paper
                       figure (publication.py) and the simulators
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LightSource, to_rgb  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Ellipse, FancyBboxPatch, Patch, Rectangle  # noqa: E402
from matplotlib.transforms import Affine2D  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

from revised_case_studies.cs1_cylinder_interaction import common as c
from revised_case_studies.cs1_cylinder_interaction import nonholonomic as nh
from revised_case_studies.shared.paths import out_dir

OUT = out_dir('cs1') / 'presentation'
INK, MUTED, FAINT = '#172d40', '#536677', '#9aa6b1'
BLUE, ORANGE, PURPLE, RED = '#215f98', '#bd5c31', '#755a9c', '#b3261e'
CERT, NOCERT = '#8fd19e', '#dcdcdc'
VOX = '#5fae75'
LAYOUT_NAME = {'single': 'One fixed magnet', 'gate': 'Two fixed magnets',
               'triangle': 'Three fixed magnets'}
HEADINGS = (0, 45, 90, 135)
MOVING_SIZE, FIXED_SIZE = (9.0, 5.0), (11.0, 5.5)   # mm, as in the simulation
P = math.pi / 2
# Start and goal are certified with the planner margin (2 mm) in the sharp TFW.
# Reversing is penalised so the planner uses it only when it is needed.
D = math.radians
MISSIONS = {   # start, goal (x m, y m, beta rad)
    'single': ((0.010, -0.045, P), (0.010, 0.045, P)),
    'gate': ((-0.040, -0.042, P), (0.040, 0.042, P)),
    'triangle': ((-0.042, -0.040, P), (0.040, 0.040, P)),
}
SMOOTH = dict(reverse=20, switch=.05, edge=.001, curve=.0005, turn=.002)
PLAN_OPTS = {layout: {'costs': SMOOTH} for layout in MISSIONS}
# The full turn is at (-20, 0) mm, certified at every heading.
DEMO_START = (-0.020, -0.030, P)
DEMO_CMDS = [('Forward', 30), ('Turn', 360), ('Turn', -90), ('Forward', 10),
             ('Reverse', 10), ('Turn', 90), ('Forward', 20), ('Turn', -45),
             ('Forward', 8)]
MODE_COLOR = {'Forward': BLUE, 'Reverse': PURPLE, 'Turn': ORANGE, 'Hold': MUTED}


def style():
    plt.rcParams.update({
        'font.family': 'Helvetica Neue', 'font.size': 11, 'axes.titlesize': 12,
        'axes.labelsize': 11, 'xtick.labelsize': 9.5, 'ytick.labelsize': 9.5,
        'legend.fontsize': 10, 'axes.edgecolor': '#8a949e', 'axes.labelcolor': INK,
        'text.color': INK, 'xtick.color': MUTED, 'ytick.color': MUTED,
        'axes.spines.top': False, 'axes.spines.right': False, 'savefig.dpi': 200,
        'mathtext.fontset': 'custom', 'mathtext.rm': 'Helvetica Neue',
        'mathtext.it': 'Helvetica Neue:italic'})


def to_palette(img, pal):
    """Exact nearest-colour mapping (Pillow's palette quantizer is approximate)."""
    from PIL import Image
    from scipy.spatial import cKDTree
    entries = np.array(pal.getpalette()[:768], float).reshape(-1, 3)
    a = np.asarray(img.convert('RGB')).reshape(-1, 3)
    uniq, inv = np.unique(a, axis=0, return_inverse=True)
    idx = cKDTree(entries).query(uniq)[1].astype(np.uint8)
    out = Image.fromarray(idx[inv.ravel()].reshape(img.size[1], img.size[0]), 'P')
    out.putpalette(pal.getpalette()[:768])
    return out


def save_gif(fig, update, frames, path, fps=12, dpi=100):
    """Render frames and write one GIF with a shared, undithered palette (small files)."""
    import io
    from PIL import Image
    imgs = []
    for f in frames:
        update(f)
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=dpi, facecolor='white')
        buf.seek(0)
        imgs.append(Image.open(buf).convert('RGB'))
    w, h = imgs[0].size
    sample = Image.new('RGB', (w, h * 4))
    for i, k in enumerate(np.linspace(0, len(imgs) - 1, 4).astype(int)):
        sample.paste(imgs[k], (0, h * i))
    pal = sample.quantize(colors=255, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    entries = np.array(pal.getpalette()[:768], dtype=int).reshape(-1, 3)
    entries[np.argmin(((entries - 255) ** 2).sum(axis=1))] = 255   # keep the page pure white
    pal.putpalette(entries.astype(np.uint8).ravel().tolist())
    q = [to_palette(im, pal) for im in imgs]
    q[0].save(path, save_all=True, append_images=q[1:], duration=int(1000 / fps), loop=0,
              optimize=False, disposal=1)
    print('saved', path, len(frames), 'frames', f'{Path(path).stat().st_size / 1e6:.1f} MB')


# ---------------------------------------------------------------- drawing ---

def magnet(ax, x, y, beta, *, moving=True, alpha=1.0, z=8):
    """Two-tone cylinder seen from above (red half = north / heading), in mm."""
    L, W = MOVING_SIZE if moving else FIXED_SIZE
    tr = Affine2D().rotate(beta).translate(x, y) + ax.transData
    body = FancyBboxPatch((-L / 2, -W / 2), L, W, boxstyle=f'round,pad=0,rounding_size={W * .2}',
                          fc=BLUE if moving else '#69747e', ec='#29313a', lw=.8,
                          transform=tr, zorder=z, alpha=alpha)
    north = Rectangle((0, -W / 2), L / 2, W, fc='#c0554b', ec='none', transform=tr,
                      zorder=z + .1, alpha=alpha)
    cap = Ellipse((L / 2 - W * .09, 0), W * .24, W * .8, fc='#d9948b', ec='#29313a',
                  lw=.4, transform=tr, zorder=z + .2, alpha=alpha)
    ax.add_patch(body)
    ax.add_patch(north)
    north.set_clip_path(body)
    ax.add_patch(cap)
    return [body, north, cap]


def move_magnet(patches, ax, x, y, beta):
    tr = Affine2D().rotate(beta).translate(x, y) + ax.transData
    for p in patches:
        p.set_transform(tr)
    patches[1].set_clip_path(patches[0])


def slice_rgb(layout, level):
    s = c.tfw(layout).stack[level]
    img = np.empty(s.shape + (3,))
    img[s] = to_rgb(CERT)
    img[~s] = to_rgb(NOCERT)
    return img


def workspace(ax, layout, *, level=None, labels=True):
    """Top view: slice at ``level`` (or nothing), fixed magnets, square frame."""
    im = None
    if level is not None:
        im = ax.imshow(slice_rgb(layout, level), origin='lower', extent=(-50, 50, -50, 50),
                       interpolation='nearest', zorder=0)
    for fx, fy, fb, _ in c.fixed_magnets(layout):
        magnet(ax, fx * 1e3, fy * 1e3, fb, moving=False)
    ax.set_xlim(-50, 50)
    ax.set_ylim(-50, 50)
    ax.set_aspect('equal')
    ax.set_xticks([-50, -25, 0, 25, 50])
    ax.set_yticks([-50, -25, 0, 25, 50])
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(True)
    if labels:
        ax.set_xlabel('x (mm)', labelpad=1)
        ax.set_ylabel('y (mm)', labelpad=0)
    return im


def level_of(deg, layout='single'):
    return c.tfw(layout).beta_level(math.radians(deg) + 1e-9)


def coarse(layout, n=32):
    """(n, n, 64) boolean TFW sampled at display-cell centres, indexed [x, y, beta]."""
    t = c.tfw(layout)
    idx = (np.arange(n) * t.raster / n + t.raster / (2 * n)).astype(int)
    v = t.stack[:, idx][:, :, idx]
    return np.transpose(v, (2, 1, 0))


def axes3d(ax):
    ax.set_xlim(-50, 50)
    ax.set_ylim(-50, 50)
    ax.set_zlim(0, 360)
    ax.set_box_aspect((1, 1, 1.15), zoom=.92)
    ax.set_xticks([-40, 0, 40])
    ax.set_yticks([-40, 0, 40])
    ax.set_zticks([0, 90, 180, 270, 360], ['0°', '90°', '180°', '270°', '360°'])
    ax.set_xlabel('x (mm)', labelpad=-2)
    ax.set_ylabel('y (mm)', labelpad=-2)
    ax.set_zlabel('β', labelpad=-1)
    ax.tick_params(pad=-1)
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.pane.set_facecolor((1, 1, 1, 0))
        a.pane.set_edgecolor('#c5ccd3')
        a._axinfo['grid']['color'] = (0.85, 0.87, 0.9, 1)
        a._axinfo['grid']['linewidth'] = .5


def voxels(ax, mask, color, alpha=1.0):
    e = np.linspace(-50, 50, mask.shape[0] + 1)
    b = np.linspace(0, 360, mask.shape[2] + 1)
    X, Y, Z = np.meshgrid(e, e, b, indexing='ij')
    ax.voxels(X, Y, Z, mask, facecolors=color, edgecolor=None, alpha=alpha, shade=True,
              lightsource=LightSource(azdeg=225, altdeg=40))


def fixed_posts(ax, layout):
    """Vertical lines through the fixed magnets: they are blocked at every heading."""
    for fx, fy, _, _ in c.fixed_magnets(layout):
        ax.plot([fx * 1e3] * 2, [fy * 1e3] * 2, [0, 360], color='#3b4652', lw=2.2, zorder=20)


# ------------------------------------------------------------ 3-D TFW fig ---

def exact_fraction(layout, deg=None):
    """TFW-in share from the boxes themselves (no raster rounding), in %."""
    b = c.tfw(layout).boxes
    if deg is None:
        v = (b[:, 1] - b[:, 0]) * (b[:, 3] - b[:, 2]) * (b[:, 5] - b[:, 4])
        return 100 * v.sum() / (0.1 * 0.1 * 2 * math.pi)
    lvl = level_of(deg, layout)
    beta = (lvl + .5) * c.tfw(layout).beta_step
    sel = (b[:, 4] <= beta) & (beta < b[:, 5])
    return 100 * ((b[sel, 1] - b[sel, 0]) * (b[sel, 3] - b[sel, 2])).sum() / 0.01


def fig_tfw3d(layout):
    t = c.tfw(layout)
    fig = plt.figure(figsize=(12.6, 5.0))
    ax = fig.add_axes([0.0, 0.0, 0.46, 1.0], projection='3d')
    axes3d(ax)
    ax.set_zticks([0, 180, 360], ['0°', '180°', '360°'])
    # The TFW fills most of the box, so draw its complement: the uncertified
    # tubes around the fixed magnets twist with heading.
    voxels(ax, ~coarse(layout, 64), '#a9b2bc')
    fixed_posts(ax, layout)
    for deg in HEADINGS:
        z = (level_of(deg, layout) + .5) * math.degrees(t.beta_step)
        sq = [(-50, -50, z), (50, -50, z), (50, 50, z), (-50, 50, z), (-50, -50, z)]
        ax.plot(*zip(*sq), color=BLUE, lw=1.1, zorder=30)
        ax.text(-50, -62, z, f'{deg}°', color=BLUE, fontsize=10.5, fontweight='bold',
                ha='right', va='center')
    ax.view_init(elev=24, azim=-58)
    ax.set_title(f'3-D TFW, {LAYOUT_NAME[layout].lower()}', loc='left', pad=0, y=.97,
                 fontweight='bold')
    frac = exact_fraction(layout)
    ax.text2D(0.02, 0.90, f'{frac:.0f}% of (x, y, β) certified · grey: not certified · blue: slices shown',
              transform=ax.transAxes, color=MUTED, fontsize=10.5)
    h = 0.41
    w = h * 5.0 / 12.6
    xs = [0.455, 0.455 + w + 0.025]
    ys = [0.53, 0.07]
    for k, deg in enumerate(HEADINGS):
        a = fig.add_axes([xs[k % 2], ys[k // 2], w, h])
        lvl = level_of(deg, layout)
        workspace(a, layout, level=lvl, labels=False)
        a.set_xticks([-50, 0, 50])
        a.set_yticks([-50, 0, 50])
        if k % 2:
            a.set_yticklabels([])
        if k < 2:
            a.set_xticklabels([])
        a.set_title(f'β = {deg}°   {exact_fraction(layout, deg):.0f}% certified', loc='left',
                    fontsize=11, pad=3)
        ins = a.inset_axes([0.79, 0.79, 0.19, 0.19])
        ins.set_xlim(-8, 8)
        ins.set_ylim(-8, 8)
        ins.set_aspect('equal')
        ins.set_xticks([])
        ins.set_yticks([])
        ins.set_facecolor('white')
        for sp in ins.spines.values():
            sp.set_visible(True)
            sp.set_color('#9aa6b1')
        magnet(ins, 0, 0, math.radians(deg))
    x_leg = xs[1] + w + 0.03
    leg = [Patch(fc=CERT, label='Certified (TFW)'), Patch(fc=NOCERT, label='Not certified'),
           Patch(fc='#69747e', label='Fixed magnet'), Patch(fc=BLUE, label='Free magnet\n(inset: heading)')]
    fig.legend(handles=leg, loc='center left', bbox_to_anchor=(x_leg, 0.5), ncol=1,
               frameon=False, fontsize=10.5, labelspacing=1.1)
    fig.text(x_leg + 0.005, 0.10, 'axes: x, y (mm)', color=MUTED, fontsize=9.5)
    p = OUT / f'tfw3d_{layout}.png'
    fig.savefig(p, facecolor='white')
    plt.close(fig)
    print('saved', p)
    return {'certified_volume_pct': round(exact_fraction(layout), 1),
            'slice_pct': {d: round(exact_fraction(layout, d), 1) for d in HEADINGS}}


# ---------------------------------------------------------- trajectories ---

def demo_trajectory():
    """Scripted forward / turn / reverse reference, every pose checked in the TFW."""
    x, y, b = DEMO_START
    rows, modes = [(x, y, b)], ['Hold']
    for kind, val in DEMO_CMDS:
        if kind == 'Turn':
            n = max(1, int(abs(val) / 1.0))
            for _ in range(n):
                b += math.radians(val) / n
                rows.append((x, y, b))
                modes.append(kind)
        else:
            d = val if kind == 'Forward' else -val
            n = max(1, int(abs(d) / 0.25))
            for _ in range(n):
                x += d / n * 1e-3 * math.cos(b)
                y += d / n * 1e-3 * math.sin(b)
                rows.append((x, y, b))
                modes.append(kind)
    rows = np.array(rows)
    assert all(nh.pose_ok(*p, 'single') for p in rows), 'demo leaves the TFW'
    traj = nh.timed(rows)
    # mode at each reference sample (by the pose index reached)
    return traj, rows, modes


def mode_series(sim):
    """Forward / Reverse / Turn / Hold from the reference motion at each log sample."""
    xr, yr, br = sim['xr'], sim['yr'], np.unwrap(sim['betar'])
    out = []
    for k in range(len(xr)):
        j = min(k + 1, len(xr) - 1)
        i = max(j - 1, 0)
        d = np.array([xr[j] - xr[i], yr[j] - yr[i]])
        db = br[j] - br[i]
        u = np.array([math.cos(br[i]), math.sin(br[i])])
        if np.linalg.norm(d) > 1e-5:
            out.append('Forward' if d @ u > 0 else 'Reverse')
        elif abs(db) > 1e-4:
            out.append('Turn')
        else:
            out.append('Hold')
    return out


def body_velocity(sim):
    t = sim['t']
    vx, vy = np.gradient(sim['x'], t), np.gradient(sim['y'], t)
    b = sim['beta']
    fwd = vx * np.cos(b) + vy * np.sin(b)
    lat = -vx * np.sin(b) + vy * np.cos(b)
    om = np.gradient(np.unwrap(b), t)
    return fwd * 1e3, lat * 1e3, np.degrees(om)


def path3d(x, y, beta):
    """(x, y, beta in degrees) with NaN breaks where beta wraps through 360."""
    bd = np.degrees(np.asarray(beta)) % 360
    X, Y, B, pos = [], [], [], []
    for k in range(len(bd)):
        if k and abs(bd[k] - bd[k - 1]) > 180:
            X.append(np.nan)
            Y.append(np.nan)
            B.append(np.nan)
        pos.append(len(X))
        X.append(x[k] * 1e3)
        Y.append(y[k] * 1e3)
        B.append(bd[k])
    return np.array(X), np.array(Y), np.array(B), np.array(pos)


# ---------------------------------------------------------------- demo GIF ---

def gif_demo(sim, fps=12, step=2):
    layout = 'single'
    t = c.tfw(layout)
    modes = mode_series(sim)
    fwd, lat, om = body_velocity(sim)
    n = len(sim['t'])
    fig = plt.figure(figsize=(12.6, 5.0), facecolor='white')
    ax = fig.add_axes([0.045, 0.11, 0.33, 0.76])
    im = workspace(ax, layout, level=t.beta_level(sim['beta'][0]))
    ax.plot(sim['xr'] * 1e3, sim['yr'] * 1e3, color=INK, lw=.9, ls=(0, (3, 3)), alpha=.45, zorder=3)
    trail, = ax.plot([], [], color=BLUE, lw=2.2, zorder=5)
    mag = magnet(ax, sim['x'][0] * 1e3, sim['y'][0] * 1e3, sim['beta'][0])
    title = ax.text(0.02, 0.975, '', transform=ax.transAxes, fontsize=11.5, va='top', zorder=20,
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='none', alpha=.85))
    chip = fig.text(0.045, 0.945, '', fontsize=14, fontweight='bold', color='white',
                    va='center', bbox=dict(boxstyle='round,pad=0.35', fc=BLUE, ec='none'))
    fig.text(0.16, 0.945, 'Drives forward or backward along its axis, or turns in place.', fontsize=11.5, color=MUTED, va='center')

    a3 = fig.add_axes([0.37, 0.0, 0.34, 0.86], projection='3d')
    axes3d(a3)
    voxels(a3, ~coarse(layout), '#b8bec5', alpha=.16)
    fixed_posts(a3, layout)
    X, Y, B, pos = path3d(sim['x'], sim['y'], sim['beta'])
    a3.plot(X, Y, B, color=INK, lw=.8, alpha=.35)
    tr3, = a3.plot([], [], [], color=BLUE, lw=2.4, zorder=40)
    pt3, = a3.plot([], [], [], 'o', ms=8, color=BLUE, mec='white', mew=1.2, zorder=41)
    plane = Poly3DCollection([[(-50, -50, 0)] * 4], facecolors=(0.37, 0.68, 0.46, 0.18),
                             edgecolors='#3f8a55', linewidths=1.0)
    a3.add_collection3d(plane)
    a3.view_init(elev=22, azim=-60)
    a3.text2D(0.04, 0.98, 'Configuration space (x, y, β)', transform=a3.transAxes,
              fontsize=12, va='top')
    a3.text2D(0.04, 0.92, 'grey: not certified · green plane: current β', transform=a3.transAxes,
              fontsize=10, color=MUTED, va='top')

    tt = sim['t']
    ab = fig.add_axes([0.765, 0.56, 0.215, 0.30])
    bd = np.degrees(sim['beta']) % 360
    bd_plot = np.where(np.abs(np.diff(bd, prepend=bd[0])) > 180, np.nan, bd)
    ab.plot(tt, bd_plot, color=ORANGE, lw=1.8)
    ab.set_ylim(0, 360)
    ab.set_yticks([0, 90, 180, 270, 360])
    ab.set_ylabel('β (°)')
    ab.set_xlim(0, tt[-1])
    ab.set_xticklabels([])
    ab.set_title('Heading', loc='left', fontsize=12, pad=4)
    av = fig.add_axes([0.765, 0.13, 0.215, 0.30])
    av.plot(tt, fwd, color=BLUE, lw=1.6, label='forward')
    av.plot(tt, lat, color=RED, lw=1.6, label='sideways')
    av.axhline(0, color=FAINT, lw=.6)
    av.set_xlim(0, tt[-1])
    av.set_ylim(-8, 8)
    av.set_xlabel('time (s)')
    av.set_ylabel('speed (mm/s)')
    av.set_title('Body-frame velocity', loc='left', fontsize=12, pad=4)
    av.legend(loc='lower left', frameon=False, ncol=2, fontsize=9.5, borderaxespad=.1)
    cur = [a.axvline(0, color=INK, lw=.9) for a in (ab, av)]
    leg = [Patch(fc=CERT, label='Certified at current β'), Patch(fc=NOCERT, label='Not certified')]
    fig.legend(handles=leg, loc='center right', bbox_to_anchor=(0.995, 0.945), ncol=2,
               frameon=False, fontsize=9.5, handlelength=1.2)

    def update(k):
        x, y, b = sim['x'][k], sim['y'][k], sim['beta'][k]
        lvl = t.beta_level(b)
        im.set_data(slice_rgb(layout, lvl))
        trail.set_data(sim['x'][:k + 1] * 1e3, sim['y'][:k + 1] * 1e3)
        move_magnet(mag, ax, x * 1e3, y * 1e3, b)
        lab = 'inside the TFW' if sim['label'][k] in ('inside', 'boundary') else 'outside the TFW'
        title.set_text(f'β = {math.degrees(b) % 360:3.0f}°  ·  {lab}')
        m = modes[k]
        chip.set_text({'Turn': 'Turn in place', 'Hold': 'Hold'}.get(m, m))
        chip.get_bbox_patch().set_facecolor(MODE_COLOR[m])
        e = pos[k] + 1
        tr3.set_data_3d(X[:e], Y[:e], B[:e])
        z = math.degrees(b) % 360
        pt3.set_data_3d([x * 1e3], [y * 1e3], [z])
        plane.set_verts([[(-50, -50, z), (50, -50, z), (50, 50, z), (-50, 50, z)]])
        for line in cur:
            line.set_xdata([tt[k]] * 2)
        return []

    frames = list(range(0, n, step))
    frames += [n - 1] * fps
    p = OUT / 'demo_single.gif'
    save_gif(fig, update, frames, p, fps)
    update(frames[len(frames) // 2])
    fig.savefig(OUT / 'demo_single_frame.png', dpi=100, facecolor='white')
    plt.close(fig)
    return {'duration_s': round(float(tt[-1]), 1),
            'max_sideways_mm_s': round(float(np.abs(lat).max()), 2),
            'max_forward_mm_s': round(float(np.abs(fwd).max()), 2),
            'samples_inside_pct': round(100 * float(np.isin(sim['label'], ['inside', 'boundary']).mean()), 1),
            'max_track_err_mm': round(float(sim['err'].max() * 1e3), 2)}


# ----------------------------------------------------------- planning fig ---

def freedom_rgb(layout):
    """All headings / some headings / no heading certified at each position."""
    f = c.tfw(layout).stack.mean(axis=0)
    img = np.empty(f.shape + (3,))
    img[:] = to_rgb(NOCERT)
    img[f > 0] = to_rgb('#cfeed6')
    img[f >= 1] = to_rgb(CERT)
    return img


def fig_plans(runs):
    fig = plt.figure(figsize=(12.6, 5.6))
    for j, layout in enumerate(MISSIONS):
        r = runs[layout]
        sim = r['sim']
        x0 = 0.06 + j * 0.322
        ax = fig.add_axes([x0, 0.43, 0.255, 0.47])
        ax.imshow(freedom_rgb(layout), origin='lower', extent=(-50, 50, -50, 50),
                  interpolation='nearest', zorder=0)
        workspace(ax, layout, labels=False)
        ax.set_xticks([-50, 0, 50])
        ax.set_yticks([-50, 0, 50])
        if j:
            ax.set_yticklabels([])
        ax.plot(sim['x'] * 1e3, sim['y'] * 1e3, color=BLUE, lw=2, zorder=5)
        d = np.r_[0, np.cumsum(np.hypot(np.diff(sim['x']), np.diff(sim['y'])))]
        idx = [int(np.searchsorted(d, v)) for v in np.linspace(0, d[-1], 8)]
        for i in idx:
            magnet(ax, sim['x'][i] * 1e3, sim['y'][i] * 1e3, sim['beta'][i], alpha=.9, z=9)
        g = r['goal']
        ax.plot(g[0] * 1e3, g[1] * 1e3, marker='*', ms=15, color='#f2c14e', mec=INK, zorder=12)
        fig.text(x0, 0.95, LAYOUT_NAME[layout], fontsize=13.5, fontweight='bold')
        tl = fig.add_axes([x0, 0.14, 0.255, 0.20])
        tt = sim['t']
        modes = mode_series(sim)
        for m, col in MODE_COLOR.items():
            if m == 'Hold':
                continue
            on = np.array([mm == m for mm in modes])
            tl.fill_between(tt, 0, 1, where=on, color=col, alpha=.18, lw=0,
                            transform=tl.get_xaxis_transform(), step='mid')
        bd = np.degrees(sim['beta']) % 360
        tl.plot(tt, np.where(np.abs(np.diff(bd, prepend=bd[0])) > 180, np.nan, bd),
                color=ORANGE, lw=1.6)
        tl.set_ylim(0, 360)
        tl.set_yticks([0, 180, 360])
        tl.set_ylabel('β (°)' if j == 0 else '')
        tl.set_xlim(0, tt[-1])
        tl.set_xlabel('time (s)', labelpad=1)
        tl2 = tl.twinx()
        tl2.plot(tt, sim['sep'] * 1e3, color=INK, lw=1.4)
        tl2.set_ylim(0, 60)
        tl2.set_yticks([0, 30, 60])
        tl2.spines['right'].set_visible(True)
        tl2.set_ylabel('distance (mm)' if j == 2 else '')
        ins = np.isin(sim['label'], ['inside', 'boundary']).mean() * 100
        ax.set_title(f'{tt[-1]:.0f} s · {ins:.0f}% of samples certified · '
                     f'closest {sim["sep"].min() * 1e3:.0f} mm', loc='left', fontsize=10.5,
                     color=MUTED, pad=4)
    leg = [Patch(fc=CERT, label='certified at every heading'),
           Patch(fc='#cfeed6', label='at some headings'),
           Patch(fc=NOCERT, label='at no heading'),
           Patch(fc=MODE_COLOR['Forward'], alpha=.3, label='forward'),
           Patch(fc=MODE_COLOR['Reverse'], alpha=.3, label='reverse'),
           Patch(fc=MODE_COLOR['Turn'], alpha=.3, label='turn in place'),
           Line2D([], [], color=ORANGE, lw=1.6, label='heading β'),
           Line2D([], [], color=INK, lw=1.4, label='distance to nearest fixed magnet')]
    fig.legend(handles=leg, loc='lower center', bbox_to_anchor=(.5, -0.01), ncol=8,
               frameon=False, fontsize=9.5, handlelength=1.2, columnspacing=1.0)
    p = OUT / 'plans.png'
    fig.savefig(p, facecolor='white')
    plt.close(fig)
    print('saved', p)


def gif_plans(runs, fps=12, dt_frame=0.25):
    cols = list(MISSIONS)
    T = max(runs[k]['sim']['t'][-1] for k in cols)
    times = np.arange(0, T + 1e-9, dt_frame)
    fig = plt.figure(figsize=(12.6, 4.9), facecolor='white')
    arts = []
    for j, layout in enumerate(cols):
        sim = runs[layout]['sim']
        ax = fig.add_axes([0.04 + j * 0.325, 0.19, 0.28, 0.66])
        im = workspace(ax, layout, level=c.tfw(layout).beta_level(sim['beta'][0]), labels=(j == 0))
        if j:
            ax.set_xlabel('x (mm)')
        ax.plot(sim['xr'] * 1e3, sim['yr'] * 1e3, color=INK, lw=.9, ls=(0, (3, 3)), alpha=.45, zorder=3)
        g = runs[layout]['goal']
        ax.plot(g[0] * 1e3, g[1] * 1e3, marker='*', ms=14, color='#f2c14e', mec=INK, zorder=12)
        trail, = ax.plot([], [], color=BLUE, lw=2.2, zorder=5)
        mag = magnet(ax, sim['x'][0] * 1e3, sim['y'][0] * 1e3, sim['beta'][0])
        fig.text(0.04 + j * 0.325, 0.93, LAYOUT_NAME[layout], fontsize=13.5, fontweight='bold')
        st = ax.set_title('', loc='left', fontsize=11, pad=4, color=MUTED)
        arts.append((sim, ax, im, trail, mag, st, layout))
    leg = [Patch(fc=CERT, label='Certified slice at the current heading'),
           Patch(fc=NOCERT, label='Not certified'),
           Line2D([], [], color=INK, ls=(0, (3, 3)), alpha=.6, label='planned path'),
           Line2D([], [], color=BLUE, lw=2.2, label='simulated motion')]
    fig.legend(handles=leg, loc='lower center', bbox_to_anchor=(.5, 0.0), ncol=4, frameon=False)
    clock = fig.text(0.985, 0.93, '', ha='right', fontsize=12, color=MUTED)

    def update(tk):
        for sim, ax, im, trail, mag, st, layout in arts:
            k = min(int(round(tk / nh.DT_CTRL)), len(sim['t']) - 1)
            im.set_data(slice_rgb(layout, c.tfw(layout).beta_level(sim['beta'][k])))
            trail.set_data(sim['x'][:k + 1] * 1e3, sim['y'][:k + 1] * 1e3)
            move_magnet(mag, ax, sim['x'][k] * 1e3, sim['y'][k] * 1e3, sim['beta'][k])
            done = k == len(sim['t']) - 1
            st.set_text(f'β = {math.degrees(sim["beta"][k]) % 360:5.1f}°'
                        + ('  ·  goal reached' if done else ''))
        clock.set_text(f't = {tk:4.1f} s   (×{dt_frame * fps:.0f} speed)')
        return []

    frames = list(times) + [times[-1]] * fps * 2
    p = OUT / 'plans.gif'
    save_gif(fig, update, frames, p, fps)
    update(times[len(times) // 3])
    fig.savefig(OUT / 'plans_frame.png', dpi=100, facecolor='white')
    plt.close(fig)


# -------------------------------------------------------------- crossing ---

crossing_times = {}


def gif_crossing(run, fps=12, dt_frame=0.2):
    layout = 'single'
    planned, cross = run['sim'], run['sim_straight']
    T = planned['t'][-1]
    tc = cross['contact_t']
    times = np.arange(0, T + 1e-9, dt_frame)
    fig = plt.figure(figsize=(12.6, 5.0), facecolor='white')
    arts = []
    for j, (sim, name, col) in enumerate([(planned, 'Planned in the TFW', BLUE),
                                          (cross, 'Straight crossing', RED)]):
        ax = fig.add_axes([0.04 + j * 0.27, 0.12, 0.25, 0.72])
        im = workspace(ax, layout, level=c.tfw(layout).beta_level(sim['beta'][0]), labels=(j == 0))
        if j:
            ax.set_xlabel('x (mm)')
            ax.set_yticklabels([])
        ax.plot(sim['xr'] * 1e3, sim['yr'] * 1e3, color=INK, lw=.9, ls=(0, (3, 3)), alpha=.5, zorder=3)
        trail, = ax.plot([], [], color=col, lw=2.2, zorder=5)
        mag = magnet(ax, sim['x'][0] * 1e3, sim['y'][0] * 1e3, sim['beta'][0])
        fig.text(0.04 + j * 0.27, 0.93, name, fontsize=13.5, fontweight='bold', color=col)
        st = ax.set_title('', loc='left', fontsize=11, pad=4)
        boom = ax.text(0, 0, '', fontsize=13, fontweight='bold', color='white', zorder=30,
                       ha='center', va='center',
                       bbox=dict(boxstyle='round,pad=0.3', fc=RED, ec='none'))
        arts.append((sim, ax, im, trail, mag, st, boom))
    a1 = fig.add_axes([0.62, 0.56, 0.355, 0.30])
    a2 = fig.add_axes([0.62, 0.12, 0.355, 0.30])
    for sim, col in ((planned, BLUE), (cross, RED)):
        a1.plot(sim['t'], sim['sep'] * 1e3, color=col, lw=1.8)
        a2.plot(sim['t'], np.abs(sim['fy_res']) * 1e3, color=col, lw=1.8)
    a1.axhline(c.CONTACT_DIST * 1e3, color=RED, lw=.9, ls='--')
    a1.text(T, c.CONTACT_DIST * 1e3 + 1.5, 'contact', color=RED, ha='right', fontsize=10)
    a1.set_ylim(0, 60)
    a1.set_xlim(0, T)
    a1.set_xticklabels([])
    a1.set_ylabel('mm')
    a1.set_title('Distance to the fixed magnet', loc='left', fontsize=12, pad=4)
    a2.set_xlim(0, T)
    a2.set_ylim(0, 100)
    a2.set_xlabel('time (s)')
    a2.set_ylabel('mN')
    a2.set_title('Lateral force the coils cannot cancel', loc='left', fontsize=12, pad=4)
    out = ~np.isin(cross['label'], ['inside', 'boundary'])
    t_exit = float(cross['t'][np.argmax(out)]) if out.any() else None
    bad = np.abs(cross['fy_res']) > 1e-3
    t_bad = float(cross['t'][np.argmax(bad)]) if bad.any() else None
    for a in (a1, a2):
        if t_exit is not None:
            a.axvspan(t_exit, cross['t'][-1], color=NOCERT, alpha=.6, lw=0, zorder=0)
    if t_exit is not None:
        a1.text(t_exit + .2, 52, f'leaves the TFW\nat {t_exit:.1f} s', color=MUTED, fontsize=9.5, va='top')
    if t_bad is not None:
        a2.annotate(f'task infeasible\nfrom {t_bad:.1f} s', xy=(t_bad, 60), xytext=(t_bad + 3, 70),
                    color=RED, fontsize=9.5, arrowprops=dict(arrowstyle='-', color=RED, lw=.8))
    cur = [a.axvline(0, color=INK, lw=.9) for a in (a1, a2)]
    leg = [Patch(fc=CERT, label='Certified slice at the current heading'),
           Patch(fc=NOCERT, label='Not certified'),
           Line2D([], [], color=INK, ls=(0, (3, 3)), alpha=.6, label='reference')]
    crossing_times.update(exit=t_exit, infeasible=t_bad, contact=tc)
    fig.legend(handles=leg, loc='lower left', bbox_to_anchor=(0.04, 0.0), ncol=3, frameon=False)

    def update(tk):
        for sim, ax, im, trail, mag, st, boom in arts:
            k = min(int(round(tk / nh.DT_CTRL)), len(sim['t']) - 1)
            b = sim['beta'][k]
            im.set_data(slice_rgb(layout, c.tfw(layout).beta_level(b)))
            trail.set_data(sim['x'][:k + 1] * 1e3, sim['y'][:k + 1] * 1e3)
            move_magnet(mag, ax, sim['x'][k] * 1e3, sim['y'][k] * 1e3, b)
            inside = sim['label'][k] in ('inside', 'boundary')
            hit = sim['contact_t'] is not None and k == len(sim['t']) - 1
            if hit:
                st.set_text(f'collision at t = {sim["contact_t"]:.1f} s')
                st.set_color(RED)
                boom.set_position((sim['x'][k] * 1e3 - 16, sim['y'][k] * 1e3 - 14))
                boom.set_text('CONTACT')
            else:
                st.set_text('inside the TFW' if inside else 'outside the TFW')
                st.set_color('#2e7d43' if inside else RED)
        for line in cur:
            line.set_xdata([tk] * 2)
        return []

    frames = list(times) + [times[-1]] * fps * 2
    p = OUT / 'crossing.gif'
    save_gif(fig, update, frames, p, fps)
    update(min(tc + 1.0, T) if tc else T / 2)
    fig.savefig(OUT / 'crossing_frame.png', dpi=100, facecolor='white')
    plt.close(fig)


# ------------------------------------------------------------------ main ---

def run_missions():
    runs = {}
    for layout, (s, g) in MISSIONS.items():
        opt = PLAN_OPTS.get(layout, {})
        pts = [s, *opt.get('via', []), g]
        legs = [nh.plan(layout, a, b, margin=opt.get('margin', nh.MARGIN), costs=opt.get('costs'))
                for a, b in zip(pts[:-1], pts[1:])]
        path = np.vstack([legs[0]] + [leg[1:] for leg in legs[1:]])
        traj = nh.timed(path)
        sim = nh.simulate(traj, layout)
        hd = math.atan2(g[1] - s[1], g[0] - s[0])
        st = nh.straight((s[0], s[1], hd), g)
        sim2 = nh.simulate(st, layout)
        runs[layout] = dict(start=s, goal=g, path=path, traj=traj, sim=sim,
                            straight=st, sim_straight=sim2)
        for kind, arr in (('planned', traj), ('straight', st)):
            (OUT / f'trajectory_{layout}_{kind}.json').write_text(json.dumps({
                'layout': layout, 'kind': kind, 'start': list(s), 'goal': list(g),
                'source': 'nonholonomic.plan on the sharp 3-D TFW' if kind == 'planned'
                          else 'constant-heading straight reference (interaction ignored)',
                'columns': ['x_m', 'y_m', 'beta_rad', 't_s'],
                'trajectory': np.asarray(arr).tolist()}))
        print(layout, 'planned', f'{traj[-1, 3]:.1f} s', 'contact', sim['contact_t'],
              '| straight contact', sim2['contact_t'])
    return runs


def run_summary(sim, goal):
    lab = sim['label']
    modes = mode_series(sim)
    return {'duration_s': round(float(sim['t'][-1]), 1),
            'samples_certified_pct': round(100 * float(np.isin(lab, ['inside', 'boundary']).mean()), 1),
            'contact_t_s': None if sim['contact_t'] is None else round(sim['contact_t'], 2),
            'min_distance_mm': round(float(sim['sep'].min() * 1e3), 1),
            'max_track_err_mm': round(float(sim['err'].max() * 1e3), 2),
            'final_goal_err_mm': round(float(math.dist((sim['x'][-1], sim['y'][-1]), goal[:2]) * 1e3), 2),
            'peak_current_A': round(float(np.abs(sim['currents']).max()), 1),
            'max_lateral_residual_mN': round(float(np.abs(sim['fy_res']).max() * 1e3), 2),
            'samples': {m: int(sum(mm == m for mm in modes)) for m in MODE_COLOR}}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--only', nargs='*', default=None,
                    help='subset of: tfw3d demo plans plans_gif crossing')
    args = ap.parse_args()
    want = set(args.only or ['tfw3d', 'demo', 'plans', 'plans_gif', 'crossing'])
    OUT.mkdir(parents=True, exist_ok=True)
    style()
    sp = OUT / 'summary.json'
    summary = json.loads(sp.read_text()) if sp.exists() else {}
    summary['source'] = ('fresh 3-D TFWs with the corrected dBy/dy coefficient '
                         '(results/cs1/tfw3d_corrected/tfw3d_*.json); simulations only')
    if 'tfw3d' in want:
        summary['tfw'] = {l: fig_tfw3d(l) for l in MISSIONS}
    if 'demo' in want:
        traj, _, _ = demo_trajectory()
        summary['demo'] = gif_demo(nh.simulate(traj, 'single'))
    if want & {'plans', 'plans_gif', 'crossing'}:
        runs = run_missions()
        summary['planned'] = {l: run_summary(r['sim'], r['goal']) for l, r in runs.items()}
        summary['straight'] = {l: run_summary(r['sim_straight'], r['goal']) for l, r in runs.items()}
        summary['missions_mm_deg'] = {l: {'start': [round(s[0] * 1e3, 1), round(s[1] * 1e3, 1), round(math.degrees(s[2]))],
                                          'goal': [round(g[0] * 1e3, 1), round(g[1] * 1e3, 1), round(math.degrees(g[2]))]}
                                      for l, (s, g) in MISSIONS.items()}
        if 'plans' in want:
            fig_plans(runs)
        if 'plans_gif' in want:
            gif_plans(runs)
        if 'crossing' in want:
            gif_crossing(runs['single'])
            summary['crossing_single_s'] = {k: None if v is None else round(v, 2)
                                            for k, v in crossing_times.items()}
    sp.write_text(json.dumps(summary, indent=2) + '\n')
    print('saved', sp)


if __name__ == '__main__':
    main()
