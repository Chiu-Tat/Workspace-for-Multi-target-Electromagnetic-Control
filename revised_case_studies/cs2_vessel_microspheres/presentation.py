#!/usr/bin/env python3
# Run from the repository root:
#   .venv/bin/python -B -m revised_case_studies.cs2_vessel_microspheres.presentation
"""Case Study 2 figures and animation for the slide deck.

Reads the saved publication records (``results/cs2/publication``); runs no new
simulation.  Outputs in ``results/cs2/presentation``:

  maps.png        vessel geometry and joint TFW with both paths, both arrangements
  passing.gif     side-by-side passing: aware and direct runs in the vessels and
                  in the joint configuration space, with the full-task LP status
  margin.png      force-margin maps and margin along the four runs
"""

from __future__ import annotations

import io
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from revised_case_studies.cs2_vessel_microspheres import publication as B

OUT = B.SAVED / 'presentation'
INK, MUTED, GREEN, RED = '#172d40', '#536677', '#267347', '#a64242'


def style():
    B.style()
    plt.rcParams.update({'font.family': 'Helvetica Neue'})


def export(fig, name):
    fig.savefig(OUT / name, dpi=330, facecolor='white', bbox_inches='tight', pad_inches=.04)
    plt.close(fig)
    print('saved', OUT / name)


def fig_maps(geoms, data):
    fig, axs = plt.subplots(1, 4, figsize=(7.4, 2.3),
                            gridspec_kw={'wspace': .42, 'left': .06, 'right': .99, 'bottom': .2, 'top': .82})
    for k, v in enumerate(B.VARIANTS):
        g = geoms[v]
        name = ['Side-by-side', 'Over-under'][k]
        B.geometry_panel(axs[2 * k], g, '', f'{name} vessels')
        B.workspace_panel(axs[2 * k + 1], g, data[v], '')
        axs[2 * k].set_title(f'{name} vessels', loc='left', pad=4, fontsize=8.5)
        axs[2 * k + 1].set_title(f'{name}: joint TFW', loc='left', pad=4, fontsize=8.5)
    handles = [B.Patch(fc=B.INNER, label='TFW-in'), B.Patch(fc=B.UNKNOWN, label='Unresolved'),
               B.Line2D([], [], marker='x', color=RED, ls='', ms=4, label='LP-infeasible sample'),
               B.Line2D([], [], color=B.INK, lw=1.1, label='TFW-aware path'),
               B.Line2D([], [], color=B.ORANGE, lw=1.1, label='Direct path')]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, -.05), ncol=5, frameon=False, fontsize=7)
    export(fig, 'maps.png')


def fig_margin(geoms, data, margins):
    fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.45),
                            gridspec_kw={'wspace': .5, 'left': .06, 'right': .99, 'bottom': .18, 'top': .86})
    B.margin_map_panel(axs[0], geoms['side_by_side'], data['side_by_side'], margins, 'side_by_side', '', '')
    mesh = B.margin_map_panel(axs[1], geoms['over_under'], data['over_under'], margins, 'over_under', '', '')
    cax = axs[1].inset_axes([1.05, 0, .05, 1])
    cb = fig.colorbar(mesh, cax=cax, ticks=[0, 1, 2, 3])
    cb.ax.tick_params(labelsize=6)
    cb.set_label('$\\lambda$', fontsize=7, labelpad=1, rotation=0)
    cb.ax.axhline(1, color=RED, lw=1)
    B.margin_time_panel(axs[2], margins, '')
    for ax, t in zip(axs, ['Side-by-side: force margin', 'Over-under: force margin', 'Margin along the four runs']):
        ax.set_title(t, loc='left', pad=4, fontsize=8.5)
    handles = [B.Line2D([], [], color=B.INK, lw=1.1, label='TFW-aware path'),
               B.Line2D([], [], color=B.ORANGE, lw=1.1, label='Direct path'),
               B.Line2D([], [], color=RED, lw=1, label='$\\lambda=1$: full task just attainable')]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.38, -.06), ncol=3, frameon=False, fontsize=7)
    export(fig, 'margin.png')


def gif_passing(geoms, data, margins, fps=12, dt=0.25):
    v = 'side_by_side'
    g = geoms[v]
    runs = data[v]
    plt.rcParams.update({'font.size': 13, 'axes.labelsize': 12, 'xtick.labelsize': 11, 'ytick.labelsize': 11})
    fig = plt.figure(figsize=(12.6, 5.0), dpi=100, facecolor='white')
    axes = [fig.add_axes([0.06, 0.16, 0.24, 0.64]), fig.add_axes([0.355, 0.16, 0.24, 0.64]),
            fig.add_axes([0.665, 0.16, 0.245, 0.64])]
    arms = [('coordinated', 'TFW-aware reference', B.INK), ('direct', 'Direct reference', B.ORANGE)]
    objs = []
    for ax, (arm, title, color) in zip(axes[:2], arms):
        B.geometry_panel(ax, g, '', '')
        for p in list(ax.patches):
            p.remove()
        for tx in list(ax.texts):
            if tx.get_text().startswith('Sphere') or tx.get_text() == '':
                tx.remove()
            else:
                tx.set_fontsize(10)
        ax.set_title('')
        ax.set_title('', loc='left')
        fig.text(ax.get_position().x0, 0.93, title, fontsize=15, weight='bold', color=color)
        status = fig.text(ax.get_position().x0, 0.86, '', fontsize=12)
        dots = [(ax.plot([], [], 'o', ms=13, color=B.SPHERE_GOLD, mec='white', mew=1.4, zorder=10)[0],
                 ax.text(0, 0, str(j + 1), fontsize=10, weight='bold', color=c, zorder=11,
                         bbox={'fc': 'white', 'ec': 'none', 'alpha': .85, 'pad': .8}))
                for j, c in enumerate(B.SPHERE_COLORS)]
        objs.append((runs[arm]['records'], dots, status))
    ax = axes[2]
    B.workspace_panel(ax, g, runs, '')
    ax.set_title('', loc='left')
    for line in list(ax.lines):
        if line.get_linestyle() == '-' and len(line.get_xdata()) > 50:
            line.remove()
    for t in ax.texts:
        t.set_fontsize(10)
    fig.text(ax.get_position().x0, 0.93, 'Joint configuration space', fontsize=15, weight='bold')
    traces = []
    for arm, _, color in arms:
        tr, = ax.plot([], [], color=color, lw=2.6, zorder=8)
        dot, = ax.plot([], [], 'o' if arm == 'coordinated' else 's', color=color, ms=9, mec='white', mew=1.1, zorder=10)
        traces.append((runs[arm]['records'], tr, dot))
    lam = margins['runs'][v]
    lax = fig.add_axes([0.935, 0.16, 0.012, 0.64])
    lax.set_xlim(0, 1)
    lax.set_ylim(0, 3)
    lax.axhspan(0, 1, color=RED, alpha=.12, lw=0)
    lax.axhline(1, color=RED, lw=1)
    lax.set_xticks([])
    lax.set_yticks([0, 1, 2, 3])
    lax.tick_params(labelsize=10)
    lax.yaxis.tick_right()
    lax.set_title('λ', fontsize=13)
    bars = {arm: lax.plot([], [], marker=m, color=c, ms=8, mec='white', ls='')[0]
            for (arm, _, c), m in zip(arms, ('o', 's'))}
    clock = fig.text(0.985, 0.93, '', ha='right', fontsize=13, color=MUTED)
    fig.text(0.06, 0.015, 'Gold: microspheres (1 in the artery, 2 in the vein)   '
             'λ: force margin, the full task needs λ ≥ 1   Simulation, 3× speed',
             fontsize=11, color=MUTED)
    T = runs['coordinated']['summary']['reference_duration_s'] + 1.0

    def at(rr, t):
        ix = max(1, np.searchsorted([r['t'] for r in rr], t, side='right'))
        return ix - 1

    def update(t):
        for rr, dots, st in objs:
            k = at(rr, t)
            r = rr[k]
            for j, (d, lab) in enumerate(dots):
                x, y = r['positions'][j][0] * 1e3, r['positions'][j][1] * 1e3
                d.set_data([x], [y])
                lab.set_position((x + (-4.2 if j == 0 else 2.4), y + 1.6))
            ok = r['full_task_class'] == 'feasible'
            st.set_text('Full task feasible' if ok else 'Full task infeasible')
            st.set_color(GREEN if ok else RED)
        for (arm, _, _), (rr, tr, dot) in zip(arms, traces):
            k = at(rr, t)
            tr.set_data([r['s'][0] for r in rr[:k + 1]], [r['s'][1] for r in rr[:k + 1]])
            dot.set_data([rr[k]['s'][0]], [rr[k]['s'][1]])
            lk = min(k, len(lam[arm]['lambda']) - 1)
            bars[arm].set_data([.5], [min(lam[arm]['lambda'][lk], 2.95)])
        clock.set_text(f't = {t:4.1f} s')

    times = list(np.arange(0, T + 1e-9, dt)) + [T] * fps * 2
    frames = []
    for t in times:
        update(t)
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, facecolor='white')
        buf.seek(0)
        frames.append(Image.open(buf).convert('RGB'))
    plt.close(fig)
    w, h = frames[0].size
    sample = Image.new('RGB', (w, h * 4))
    for i, k in enumerate(np.linspace(0, len(frames) - 1, 4).astype(int)):
        sample.paste(frames[k], (0, h * i))
    pal = sample.quantize(colors=255, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    e = np.array(pal.getpalette()[:768], int).reshape(-1, 3)
    e[np.argmin(((e - 255) ** 2).sum(1))] = 255
    pal.putpalette(e.astype(np.uint8).ravel().tolist())
    from scipy.spatial import cKDTree
    tree = cKDTree(e.astype(float))
    q = []
    for f in frames:
        a = np.asarray(f).reshape(-1, 3)
        u, inv = np.unique(a, axis=0, return_inverse=True)
        idx = tree.query(u)[1].astype(np.uint8)
        im = Image.fromarray(idx[inv.ravel()].reshape(h, w), 'P')
        im.putpalette(pal.getpalette()[:768])
        q.append(im)
    path = OUT / 'passing.gif'
    q[0].save(path, save_all=True, append_images=q[1:], duration=int(1000 / fps), loop=0, disposal=1)
    q[len(q) // 2].convert('RGB').save(OUT / 'passing_frame.png')
    print('saved', path, len(q), 'frames', f'{path.stat().st_size / 1e6:.1f} MB')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    style()
    geoms = {v: B.Geometry(v) for v in B.VARIANTS}
    data = B.load(B.OUT / 'simulation_records.json')
    margins = B.margin_data(data, geoms)
    fig_maps(geoms, data)
    fig_margin(geoms, data, margins)
    gif_passing(geoms, data, margins)


if __name__ == '__main__':
    main()
