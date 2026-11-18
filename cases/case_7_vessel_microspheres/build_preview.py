"""Generate geometry only: PNG, offline rotatable HTML, OBJ and centerline CSV.

python -m cases.case_7_vessel_microspheres.build_preview
Requires numpy, matplotlib and scikit-image. Never runs control or flow.
"""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
try:
    from skimage.measure import marching_cubes
except ImportError:  # scikit-image < 0.18
    from skimage.measure import marching_cubes_lewiner as marching_cubes
from .vessel import Vessel

HERE = Path(__file__).parent


def main():
    vessel = Vessel()
    out = HERE / 'preview'
    out.mkdir(exist_ok=True)
    centers, radii = vessel.samples()
    step = 0.00065
    lower = centers.min(axis=0) - radii.max() - 2*step
    upper = centers.max(axis=0) + radii.max() + 2*step
    axes = [np.arange(lo, hi+step, step) for lo, hi in zip(lower, upper)]
    grid = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1)
    values = vessel.clearance(grid)
    verts, faces, _, _ = marching_cubes(values, level=0, spacing=(step,)*3)
    verts += lower
    mm = (verts - vessel.origin)*1000
    with (out / 'carotid_lumen.obj').open('w') as f:
        f.write('# Idealized lumen, EMNS world coordinates in metres. Closed end caps.\n')
        for v in verts:
            f.write('v {:.8f} {:.8f} {:.8f}\n'.format(*v))
        for face in faces+1:
            f.write('f {} {} {}\n'.format(*face))
    curves = {}
    with (out / 'centerlines.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['branch', 's', 'x_m', 'y_m', 'z_m', 'radius_m'])
        for b in vessel.branches:
            s = np.linspace(0, 1, 101)
            curve = vessel.centerline(b, s)
            curves[b] = ((curve-vessel.origin)*1000).round(4).tolist()
            for u, p, r in zip(s, curve, vessel.radius(b, s)):
                writer.writerow([b, u, *p, r])
    particles = []
    for p in vessel.config['illustrative_particles']:
        xyz = (vessel.centerline(p['branch'], p['s']) - vessel.origin)*1000
        particles.append(dict(p, xyz=xyz.tolist()))

    fig = plt.figure(figsize=(13, 8), facecolor='#f4f7fb')
    ax = fig.add_subplot(121, projection='3d', facecolor='#f4f7fb')
    poly = Poly3DCollection(mm[faces], alpha=0.16, linewidths=0, facecolor='#d46e78')
    ax.add_collection3d(poly)
    colors = {'CCA': '#716079', 'ICA': '#007f92', 'ECA': '#bd6929'}
    for b, points in curves.items():
        c = np.asarray(points)
        ax.plot(*c.T, color=colors[b], linewidth=2, label=b)
    for p, color in zip(particles, ['#006fcc', '#dd9b00']):
        ax.scatter(*p['xyz'], s=65, color=color, depthshade=False)
    ax.set(xlim=(-17, 24), ylim=(-34, 36), zlim=(-13, 13), xlabel='x (mm)', ylabel='y (mm)', zlabel='z - 85 mm')
    if hasattr(ax, 'set_box_aspect'):
        ax.set_box_aspect((41, 70, 26))
    ax.view_init(elev=55, azim=-76)
    ax.set_title('3D lumen and branch centerlines', pad=18)
    ax.legend(loc='lower left')
    plan = fig.add_subplot(122, facecolor='#f4f7fb')
    # True projection of the same merged mesh, not disconnected branch tubes.
    from matplotlib.collections import PolyCollection
    plan.add_collection(PolyCollection(mm[faces][:, :, :2], facecolors='#eed0d3', edgecolors='none'))
    for b, points in curves.items():
        c = np.asarray(points)
        plan.plot(c[:, 0], c[:, 1], color=colors[b], linewidth=2)
    for p, color in zip(particles, ['#006fcc', '#dd9b00']):
        x, y, _ = p['xyz']
        plan.scatter(x, y, s=90, c=color, zorder=5)
        plan.annotate(p['name']+' → '+p['target_branch'], (x, y), xytext=(5, y), fontsize=10,
                      color=color, arrowprops={'arrowstyle': '-', 'color': color})
    plan.annotate('ICA / outlet A', (-10, 30), xytext=(-17, 35), color=colors['ICA'])
    plan.annotate('ECA / outlet B', (17, 24), xytext=(9, 29), color=colors['ECA'])
    plan.annotate('Carotid bulb', (-3, 6), xytext=(-20, 10), arrowprops={'arrowstyle': '->'}, fontsize=10)
    plan.text(-9, -33, 'CCA / inlet', color=colors['CCA'])
    plan.set(xlim=(-22, 28), ylim=(-36, 39), xlabel='x (mm)', ylabel='y (mm)')
    plan.set_aspect('equal')
    for side in ('top', 'right'):
        plan.spines[side].set_visible(False)
    plan.set_title('Branch assignment concept', pad=18)
    fig.suptitle('Two magnetic microspheres in a carotid bifurcation', fontsize=18, y=0.97)
    fig.text(0.5, 0.025, 'Idealized anatomy • editable dimensions • sphere markers enlarged • geometry only, no control simulation',
             ha='center', fontsize=10, color='#526174')
    fig.subplots_adjust(top=0.88, bottom=0.12, wspace=0.16)
    fig.savefig(out / 'carotid_preview.png', dpi=170)
    plt.close(fig)
    data = {'vertices': mm.round(3).tolist(), 'faces': faces.tolist(), 'curves': curves, 'particles': particles}
    template = (HERE / 'viewer_template.html').read_text(encoding='utf-8')
    (out / 'carotid_viewer.html').write_text(template.replace('__GEOMETRY_DATA__', json.dumps(data, separators=(',', ':'))), encoding='utf-8')
    print('Geometry exported: {} vertices, {} triangles; no control simulation.'.format(len(verts), len(faces)))
    print(out.resolve())


if __name__ == '__main__':
    main()
