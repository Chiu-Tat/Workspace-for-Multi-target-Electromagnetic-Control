"""Export separate vessel pairs and a switchable offline gallery; no dynamics.

python -m cases.case_7_vessel_microspheres.build_variants
Only NumPy and Matplotlib are needed; exact straight/radial tube display meshes
are sampled around cross-section rings. The analytical domain remains the
swept-ball union in vessel.py; display rings are explicitly an approximation.
"""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from .vessel import Vessel

HERE = Path(__file__).parent


def coordinate_range(control_points, axis):
    """Exact scalar cubic Bezier extrema via its quadratic derivative."""
    q = np.asarray(control_points, dtype=float)[:, axis]
    coef = [-q[0]+3*q[1]-3*q[2]+q[3], 3*q[0]-6*q[1]+3*q[2], -3*q[0]+3*q[1], q[0]]
    roots = np.roots(np.trim_zeros([3*coef[0], 2*coef[1], coef[2]], 'f'))
    s = [0., 1.] + [float(r.real) for r in roots if abs(r.imag) < 1e-10 and 0 <= r.real <= 1]
    values = np.polyval(coef, s)
    return float(min(values)), float(max(values))


def separation_certificate(vessel):
    """Continuous separating-axis bound for the constant-radius paired presets.

    Positive bound proves disjoint swept-ball lumens, independent of mesh
    sampling. The midpoint distance attains it in both supplied presets.
    """
    config = vessel.config
    axis = config['separation_axis']
    keys = list(vessel.branches)
    if len(keys) != 2:
        raise ValueError('Certificate is for exactly two branches')
    b = [vessel.branches[k] for k in keys]
    for branch in b:
        if branch['radius_start_mm'] != branch['radius_end_mm'] or branch['bulb_amplitude_mm'] != 0:
            raise ValueError('Separating-axis bound currently requires constant radii')
    ranges = [coordinate_range(branch['control_points_mm'], axis) for branch in b]
    center_bound = max(ranges[1][0]-ranges[0][1], ranges[0][0]-ranges[1][1])
    radius_sum = sum(branch['radius_start_mm'] for branch in b)
    bound = center_bound - radius_sum
    mid = np.linalg.norm(vessel.centerline(keys[0], .5)-vessel.centerline(keys[1], .5))*1000-radius_sum
    return {'continuous_lumen_gap_lower_bound_mm': bound,
            'midpoint_lumen_gap_mm': float(mid),
            'global_minimum_attained_at_midpoints': bool(np.isclose(bound, mid)),
            'separate_lumens': bool(bound > 0),
            'method': 'Analytic extrema of cubic centerlines on a separating axis, minus constant radii'}


def tube_mesh(vessel, branch, ns=151, nt=32):
    """Sample disk-boundary rings; capped visualization, not a CFD mesh."""
    s = np.linspace(0, 1, ns)
    theta = np.arange(nt)*2*np.pi/nt
    rings = []
    for u in s:
        t, _, _ = vessel.frame(branch, u)
        # A fixed reference avoids ring-index jumps when an adaptive frame
        # changes its reference axis at the exact point of closest approach.
        n = np.cross(t, vessel.rotation[:, 2])
        if np.linalg.norm(n) < 1e-6:
            raise ValueError('Display tube requires a transported frame for a near-vertical tangent')
        n /= np.linalg.norm(n)
        b = np.cross(t, n)
        ring = vessel.centerline(branch, u) + vessel.radius(branch, u)*(np.cos(theta)[:, None]*n + np.sin(theta)[:, None]*b)
        rings.append(ring)
    vertices = np.concatenate(rings)
    faces = []
    for j in range(ns-1):
        for k in range(nt):
            a, b = j*nt+k, j*nt+(k+1)%nt
            faces.extend([[a, b, b+nt], [a, b+nt, a+nt]])
    vertices = np.vstack([vertices, vessel.centerline(branch, 0), vessel.centerline(branch, 1)])
    for k in range(nt):
        faces.extend([[ns*nt, (k+1)%nt, k], [ns*nt+1, (ns-1)*nt+k, (ns-1)*nt+(k+1)%nt]])
    return vertices, np.asarray(faces)


def build_variant(name):
    v = Vessel(HERE / 'geometries' / (name+'.json'))
    out = HERE / 'preview' / name
    out.mkdir(parents=True, exist_ok=True)
    certificate = separation_certificate(v)
    if not certificate['separate_lumens']:
        raise ValueError('Preset lumens intersect or separation has not been certified')
    (out / 'geometry_checks.json').write_text(json.dumps(certificate, indent=2), encoding='utf-8')
    data = {'title': v.config['title'], 'description': v.config['description'],
            'gap': certificate['continuous_lumen_gap_lower_bound_mm'], 'curves': {}, 'meshes': [], 'particles': []}
    colors = ['#c25362', '#347da5']
    with (out / 'centerlines.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['vessel', 's', 'x_m', 'y_m', 'z_m', 'radius_m'])
        for branch, color in zip(v.branches, colors):
            verts, faces = tube_mesh(v, branch)
            with (out / (branch+'_lumen.obj')).open('w') as obj:
                obj.write('# Separate vessel visualization; EMNS metres; artificial closed caps\n')
                for vertex in verts:
                    obj.write('v {:.8f} {:.8f} {:.8f}\n'.format(*vertex))
                for face in faces+1:
                    obj.write('f {} {} {}\n'.format(*face))
            local = (verts-v.origin)*1000
            data['meshes'].append({'vertices': local.round(4).tolist(), 'faces': faces.tolist(), 'color': color})
            ss = np.linspace(0, 1, 151)
            cc = v.centerline(branch, ss)
            data['curves'][branch] = ((cc-v.origin)*1000).round(4).tolist()
            for s, p, r in zip(ss, cc, v.radius(branch, ss)):
                writer.writerow([branch, s, *p, r])
    for particle in v.config['illustrative_particles']:
        p = (v.centerline(particle['branch'], particle['s'])-v.origin)*1000
        data['particles'].append(dict(particle, xyz=p.tolist()))
    template = (HERE / 'pair_viewer_template.html').read_text(encoding='utf-8')
    (out / 'viewer.html').write_text(template.replace('__DATA__', json.dumps(data, separators=(',', ':'))), encoding='utf-8')
    return v, data


def main():
    variants = [build_variant(n) for n in ('close_parallel', 'close_crossing')]
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), gridspec_kw={'width_ratios':[1, 1.2]})
    fig.patch.set_facecolor('#f4f7fb')
    for row, (v, data) in enumerate(variants):
        front, detail = axes[row]
        for b, color in zip(v.branches, ['#c25362', '#347da5']):
            c = np.asarray(data['curves'][b])
            front.plot(c[:, 0], c[:, 1], color=color, lw=9, alpha=.35)
            front.plot(c[:, 0], c[:, 1], color=color, lw=1.5, label=v.branches[b]['label'])
        front.set(xlim=(-15,15), ylim=(-32,32), xlabel='x (mm)', ylabel='y (mm)')
        front.set_aspect('equal')
        front.legend(fontsize=9, loc='lower center')
        front.set_title(data['title'], fontsize=12)
        separation_axis = v.config['separation_axis']
        for b, color, particle in zip(v.branches, ['#c25362','#347da5'], data['particles']):
            c = np.asarray(particle['xyz'])
            horizontal, vertical = (c[0], c[2]) if row == 0 else (c[2], c[0])
            r = float(v.radius(b,.5))*1000
            detail.add_patch(Circle((horizontal,vertical), r, fc=color, ec=color, alpha=.25))
            detail.add_patch(Circle((horizontal,vertical), .05, fc=color))
            detail.text(horizontal, vertical-r-.28, particle['name']+' / '+b, ha='center', fontsize=10, color=color)
        c1, c2 = [np.array(p['xyz'])[separation_axis] for p in data['particles']]
        keys = list(v.branches)
        lo = c1+float(v.radius(keys[0],.5))*1000
        hi = c2-float(v.radius(keys[1],.5))*1000
        detail.annotate('', (lo,0), (hi,0), arrowprops={'arrowstyle':'<->','color':'#263f53'})
        detail.text((lo+hi)/2, 1.8, '{:.2f} mm lumen gap'.format(data['gap']), ha='center', fontsize=12)
        detail.set(xlim=(-3.1,3.1), ylim=(-2.5,2.5), xlabel=('x (mm)' if row==0 else 'z - 85 mm'),
                   ylabel=('z - 85 mm' if row==0 else 'x (mm)'))
        detail.set_aspect('equal')
        detail.set_title('Closest generating balls: spheres at true size (100 µm)', fontsize=11)
        for ax in (front,detail):
            ax.set_facecolor('#f4f7fb')
            for spine in ('top','right'):
                ax.spines[spine].set_visible(False)
    fig.suptitle('Separate vessels • close magnetic targets', fontsize=20)
    fig.text(.5,.02,'Synthetic forearm-inspired phantoms • positive gaps certified for continuous lumens • no control simulation',ha='center',fontsize=10)
    fig.subplots_adjust(top=.91,bottom=.09,hspace=.35,wspace=.18)
    fig.savefig(HERE/'preview'/'vessel_variants.png',dpi=170)
    plt.close(fig)
    for _, data in variants:
        print('{}: minimum lumen gap {:.3f} mm'.format(data['title'],data['gap']))


if __name__ == '__main__':
    main()
