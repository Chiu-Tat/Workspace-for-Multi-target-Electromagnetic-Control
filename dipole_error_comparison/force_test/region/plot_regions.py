"""Figure 12: force-test setup (a), measured error (b), check with measured force (c), and region classifications at beta=90 deg
with the calibrated moment: point dipole (d), L=5 (e), L=5 + measured reserve (f).
Region data: boxes.npy / metrics.json from run_regions.py (MIN_WIDTH_MM=0.125)."""
from pathlib import Path
import json, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle, Circle, Patch, Annulus, ConnectionPatch
from matplotlib.lines import Line2D
O = Path(__file__).resolve().parent
J = json.loads((O / 'metrics.json').read_text())
D = np.load(O.parent / 'analysis.npz')
K = np.load(O.parent / 'kernels.npz')
GUARD_HW = 16.0; DATA_MM = 14.5; CONV = J['convergence_mm']; NOM_CUT = J['guard_mm']
colors = {1: '#6fb680', -1: '#d87670', 0: '#cfcfcf', 2: 'white'}
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.titlesize': 8.5,
                     'pdf.fonttype': 42, 'hatch.linewidth': 0.35})
fig = plt.figure(figsize=(7.16, 3.7))
# Top row: photo, error, check. Bottom row: three maps sharing one y-axis.
top = fig.add_gridspec(1, 3, left=.065, right=.995, bottom=.665, top=.93, wspace=.36,
                       width_ratios=(1.12, 1, 1))
bottom = fig.add_gridspec(1, 3, left=.072, right=.74, bottom=.105, top=.535, wspace=.05)
photo_ax = fig.add_subplot(top[0, 0])
photo_ax.imshow(plt.imread(O / 'force_test_setup.jpg'), aspect='auto')
photo_ax.axis('off')
photo_ax.set_title('(a) Force-test setup', pad=3)
error_ax = fig.add_subplot(top[0, 1])
check_ax = fig.add_subplot(top[0, 2])
map_axes = [fig.add_subplot(bottom[0, 0])]
map_axes += [fig.add_subplot(bottom[0, i], sharey=map_axes[0]) for i in (1, 2)]


def draw_magnet(ax, xc=20.0, yc=0.0, dia=4.0, length=10.0):
    """Cylinder seen from above: 4 mm wide, 10 mm long, moment along +y.
    Shading across the width suggests the round surface; north half red, south half grey."""
    import matplotlib.colors as mc
    n = 24; w = dia / n
    for k in range(n):
        u = (k + 0.5) / n * 2 - 1                      # -1..1 across the diameter
        light = 0.55 + 0.45 * np.sqrt(max(0.0, 1 - u * u)) * (1 - 0.35 * u)   # light from the left
        for y0, base in [(yc, '#c0392b'), (yc - length / 2, '#7f8c8d')]:
            rgb = np.clip(np.array(mc.to_rgb(base)) * light + (1 - light) * 0.08, 0, 1)
            ax.add_patch(Rectangle((xc - dia / 2 + k * w, y0), w * 1.02, length / 2, facecolor=rgb, edgecolor='none', zorder=5))
    ax.add_patch(Rectangle((xc - dia / 2, yc - length / 2), dia, length, facecolor='none', edgecolor='black', lw=.6, zorder=6))
    ax.plot([xc - dia / 2, xc + dia / 2], [yc, yc], color='black', lw=.4, zorder=6)
    ax.annotate('', (xc, yc + 3.6), (xc, yc - 3.6), arrowprops=dict(arrowstyle='-|>', color='white', lw=.9, mutation_scale=7), zorder=7)


def draw_sphere(ax, xc=20.0, yc=0.0, volume_mm3=np.pi * 2.0 ** 2 * 10.0):
    """Sphere with the same volume as the 4 x 10 mm cylinder, seen from above; shaded, north half red."""
    import matplotlib.colors as mc
    r = (3 * volume_mm3 / (4 * np.pi)) ** (1 / 3)
    n = 200; g = np.linspace(-1, 1, n); X, Y = np.meshgrid(g, g)
    rho2 = X ** 2 + Y ** 2; z = np.sqrt(np.clip(1 - rho2, 0, 1))
    light = np.clip(0.35 + 0.65 * (z * 0.8 + (-0.45 * X + 0.35 * Y) * 0.6), 0.2, 1.15)
    img = np.zeros((n, n, 4))
    for half, base in [(Y >= 0, '#c0392b'), (Y < 0, '#7f8c8d')]:
        rgb = np.array(mc.to_rgb(base))
        img[half, :3] = np.clip(rgb[None, :] * light[half][:, None], 0, 1)
    img[..., 3] = (rho2 <= 1).astype(float)
    im = ax.imshow(img, extent=(xc - r, xc + r, yc - r, yc + r), origin='lower', zorder=5, interpolation='bilinear')
    edge = Circle((xc, yc), r, facecolor='none', edgecolor='black', lw=.6, zorder=6); ax.add_patch(edge)
    im.set_clip_path(edge)
    ax.annotate('', (xc, yc + r * 0.72), (xc, yc - r * 0.72), arrowprops=dict(arrowstyle='-|>', color='white', lw=.9, mutation_scale=7), zorder=7)
    return r

# (b) measured force error vs separation
ax = error_ax; s = D['sep']
nom = np.linalg.norm(0.12 * 0.145 * K['dip'][:, :2] - D['F'], axis=1) * 1e3
sty = [('dip_loo', 'L = 1', '#c0392b', 'o'), ('L3_loo', 'L = 3', '#e0a020', '^'), ('L5_loo', 'L = 5', '#2471a3', 's')]
ax.scatter(s, nom, s=13, marker='x', c='0.35', lw=0.8, label='dipole, nominal')
for k, lab, c, m in sty:
    ax.scatter(s, np.linalg.norm(D[k], axis=1) * 1e3, s=13, marker=m, facecolors='none', edgecolors=c, lw=0.8, label=lab + ' fitted')
e5 = np.linalg.norm(D['L5_loo'], axis=1) * 1e3; ss = np.sort(np.unique(s))
env = [e5[s >= v - 1e-9].max() for v in ss]
ax.step(np.r_[ss, 62], np.r_[env, env[-1]], where='post', c='#2471a3', lw=1.0, label=r'reserve $\hat\epsilon_{F,5}$')
ax.axhline(100, c='0.5', ls=':', lw=0.8); ax.text(61, 82, 'task half-width', color='0.4', fontsize=6.5, ha='right', va='top')
ax.set(yscale='log', ylim=(0.03, 1e4), xlim=(12, 62), xlabel='centre separation (mm)',
       ylabel='force error norm (mN)')
ax.set_title('(b) Leave-one-pose-out force error', pad=3)
ax.set_yticks((1e-1, 1e1, 1e3))
ax.xaxis.labelpad = 1; ax.yaxis.labelpad = 1
ax.legend(loc='upper right', frameon=False, fontsize=5.6, handletextpad=0.1, ncol=2, columnspacing=0.6, borderaxespad=0.2)

# (c)-(d) region maps at beta = 90 deg, both with the calibrated L=5 moment product:
# (c) point dipole (run_dipole_calibrated.py, column 'L5_p'), (d) L=5 + measured reserve (run_regions.py).
A90 = np.load(O / 'boxes_b90.npy'); J90 = json.loads((O / 'metrics_b90.json').read_text())
C90 = np.load(O / 'boxes_dipole_calibrated_b90.npy'); K90 = json.loads((O / 'metrics_dipole_calibrated_b90.json').read_text())
P_TXT = f"{K90['p']['L5_p'] * 1e3:.2f}"
maps_spec = [
    (C90, C90[:, 7], NOM_CUT, '(d) Dipole, calibrated', K90['L5_p']),
    (A90, A90[:, 10], CONV, '(e) L = 5', {c: J90['modes']['L5_corrected'][c]['area_mm2'] for c in ('in', 'out', 'unresolved')}),
    (A90, np.where((A90[:, 11] == 2) & (A90[:, 10] != 2), A90[:, 10], A90[:, 11]), CONV,
     '(f) L = 5 + reserve', {c: J90['modes']['L5_empirical_reserve'][c]['area_mm2'] for c in ('in', 'out', 'unresolved')}),
]
NEW_OUT = '#8b1a1a'   # TFW-out only after adding the measured reserve, panel (f)
for k, (ax, (B, shown, inner, title, m)) in enumerate(zip(map_axes, maps_spec)):
    rects = [Rectangle((r[0] * 1e3, r[2] * 1e3), (r[1] - r[0]) * 1e3, (r[3] - r[2]) * 1e3) for r in B]
    fc = [colors[int(v)] for v in shown]
    if k == 2:
        extra = (B[:, 11] == -1) & (B[:, 10] != -1)
        fc = [NEW_OUT if e else c for e, c in zip(extra, fc)]
    ax.add_collection(PatchCollection(rects, facecolors=fc, edgecolors='face',
                                      linewidths=0, antialiaseds=False, rasterized=True))
    ax.add_patch(Annulus((20, 0), DATA_MM, DATA_MM - inner, facecolor='none', edgecolor='#7d5a8c', hatch='////', lw=0, zorder=3))
    ax.add_patch(Circle((20, 0), DATA_MM, fill=False, color='0.3', ls=':', lw=0.8, zorder=4))
    (draw_sphere(ax) if k == 0 else draw_magnet(ax))
    ax.set(xlim=(-2, 42), ylim=(-22, 22), aspect='equal', xticks=(0, 10, 20, 30, 40),
           yticks=(-20, -10, 0, 10, 20), title=title)
    ax.set_xlabel('x (mm)', labelpad=1)
    ax.set_title(title, pad=3)
    if k == 0:
        ax.set_ylabel('y (mm)', labelpad=1)
    else:
        ax.tick_params(labelleft=False)
    ax.text(.03, .97, f"TFW-in {m['in']:.0f}   TFW-out {m['out']:.0f}\nunresolved {m['unresolved']:.0f} mm$^2$",
            va='top', transform=ax.transAxes, fontsize=5.6, bbox=dict(facecolor='white', alpha=.9, edgecolor='none', pad=1.5))
    if k == 2:
        # Magnify the lower pole boundary where the reserve changes the classification.
        xlim = (17, 19)
        ylim = (-17.625, -16.625)
        width, height = xlim[1] - xlim[0], ylim[1] - ylim[0]
        x_min, x_max, y_min, y_max = B[:, :4].T * 1e3
        visible = ((x_min < xlim[1]) & (x_max > xlim[0]) &
                   (y_min < ylim[1]) & (y_max > ylim[0]))
        inset_width = .34
        inset = ax.inset_axes([.025, .025, inset_width, inset_width * height / width])
        inset.add_collection(PatchCollection(
            [rects[i] for i in np.flatnonzero(visible)],
            facecolors=np.asarray(fc)[visible], edgecolors='face',
            linewidths=0, antialiaseds=False, rasterized=True,
        ))
        inset.set(xlim=xlim, ylim=ylim, aspect='equal', xticks=[], yticks=[])
        box_style = dict(edgecolor='0.2', linewidth=.65, zorder=4.5)
        ax.add_patch(Rectangle(
            (xlim[0], ylim[0]), width, height,
            fill=False, **box_style,
        ))
        for corner, y in enumerate(ylim):
            ax.add_artist(ConnectionPatch(
                xyA=(1, corner), coordsA=inset.transAxes,
                xyB=(xlim[0], y), coordsB=ax.transData, **box_style,
            ))

# (e) verification against the measured interaction force at the 48 in-domain test poses
V = [r for r in json.loads((O / 'verification.json').read_text()) if r['in_domain']]
au = np.array([[np.nan if a is None else a for a in r['authority_mN']] for r in V])
meas, dip, l5 = au[:, 2], au[:, 0], au[:, 3]
ax = check_ax; lim = (0, 380)
ax.fill_between([100, lim[1]], 0, 100, color='#d87670', alpha=.15, lw=0)      # model infeasible, measured feasible
ax.fill_between([0, 100], 100, lim[1], color='#d87670', alpha=.15, lw=0)      # model feasible, measured infeasible
ax.plot(lim, lim, c='0.5', lw=.6); ax.axhline(100, c='0.3', ls=':', lw=.7); ax.axvline(100, c='0.3', ls=':', lw=.7)
ok = ~np.isnan(dip)
ax.scatter(meas[ok], dip[ok], s=16, marker='x', c='#c0392b', lw=.9, label='dipole, nominal', zorder=3)
ax.scatter(meas, l5, s=14, marker='s', facecolors='none', edgecolors='#2471a3', lw=.8, label='L = 5, held-out', zorder=4)
for r, m_, d_ in zip(V, meas, dip):
    if r['pose'] == 15: ax.annotate('20.5 mm', (m_, d_), (140, 58), fontsize=5.8, arrowprops=dict(arrowstyle='->', lw=.5))
ax.set(xlim=lim, ylim=lim, xlabel='authority, measured force (mN)', ylabel='predicted authority (mN)')
ax.set_title('(c) Check with measured force', pad=3)
ax.xaxis.labelpad = 1; ax.yaxis.labelpad = 1
ax.legend(loc='upper left', bbox_to_anchor=(0.27, 1.0), frameon=False, fontsize=6.0, handletextpad=0.2)
handles = [Patch(facecolor=colors[1], label='TFW-in'), Patch(facecolor=colors[-1], label='TFW-out'),
           Patch(facecolor=NEW_OUT, label='TFW-out only with\nreserve (f)'),
           Patch(facecolor=colors[0], label='Unresolved'), Patch(facecolor='white', edgecolor='0.6', label='Not evaluated'),
           Patch(facecolor='white', edgecolor='#7d5a8c', hatch='////', label='Below measured range\n(no reserve)'),
           Line2D([], [], color='0.3', ls=':', label='14.5 mm data limit')]
fig.legend(handles=handles, loc='center left', bbox_to_anchor=(.745, .32), ncol=1, frameon=False, fontsize=6.5,
           handlelength=1.6, labelspacing=0.45, borderaxespad=0)
for directory, name in [(O, 'force_test_validation')]:
    directory.mkdir(exist_ok=True, parents=True)
    fig.savefig(directory / (name + '.pdf'), dpi=600); fig.savefig(directory / (name + '.png'), dpi=240)
print('saved')
