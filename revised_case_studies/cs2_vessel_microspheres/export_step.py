"""Export the two branched vessel models as CAD pipes with open main-trunk bores.

Run from the repository root with CadQuery, NumPy and Matplotlib installed:
    python -B -m revised_case_studies.cs2_vessel_microspheres.export_step

All CAD lengths are millimetres. Main trunks have 8 mm outer diameter and
4 mm open inner diameter (2 mm nominal wall), with flat annular ends.
Side branches retain their previous exterior sizes and rounded terminal caps.
The canal-surface envelope of each continuous swept-ball branch is approximated
by a smooth circular-section loft. Branches are fused within each network, while
artery and vein are separate named, coloured bodies in the STEP assembly.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import time

import cadquery as cq
import numpy as np
from OCP.BRepOffsetAPI import BRepOffsetAPI_ThruSections
from OCP.BRep import BRep_Tool
from OCP.BRepClass3d import BRepClass3d_SolidClassifier
from OCP.TopAbs import TopAbs_IN
from OCP.gp import gp_Pnt

from cases.case_7_vessel_microspheres.vessel import Vessel

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'cases/case_7_vessel_microspheres/geometries'
DEFAULT_OUT = Path(__file__).resolve().parent / 'cad'
VARIANTS = {'side_by_side': 'branched_parallel', 'over_under': 'branched_crossing'}
COLORS = {'artery': (0.76, 0.32, 0.38), 'vein': (0.20, 0.49, 0.65)}


def registered_vessel(source):
    """Repeat VesselPair's closest-trunk registration, without loading planners."""
    vessel = Vessel(source)
    trunks = []
    for name in ('RA', 'RV'):
        dense = vessel.centerline(name, np.linspace(0, 1, 200))
        length = np.linalg.norm(np.diff(dense, axis=0), axis=1).sum()
        pts = vessel.centerline(name, np.linspace(0, 1, max(3, int(length / 2e-4))))
        trunks.append(pts[::4])
    d = np.linalg.norm(trunks[0][:, None] - trunks[1][None, :], axis=-1)
    i, j = np.unravel_index(np.argmin(d), d.shape)
    midpoint = (trunks[0][i] + trunks[1][j]) / 2
    translation = np.array([0.0, 0.0, 0.0702]) - midpoint
    vessel.origin += translation
    return vessel, translation * 1000


def envelope(vessel, name, parameters):
    """Envelope circles of the union of balls, including variable-radius offset."""
    s = np.asarray(parameters)
    branch = vessel.branches[name]
    cp = np.asarray(branch['control_points_mm'])
    d = 3 * ((1-s[:, None])**2 * (cp[1]-cp[0])
             + 2*(1-s[:, None])*s[:, None]*(cp[2]-cp[1])
             + s[:, None]**2*(cp[3]-cp[2]))
    phase = branch.get('waviness_phase', 0)
    wave_d = (np.pi*np.sin(2*np.pi*s)*np.sin(2*np.pi*s+phase)
              + 2*np.pi*np.sin(np.pi*s)**2*np.cos(2*np.pi*s+phase))
    d += wave_d[:, None] * np.asarray(branch.get('waviness_mm', [0, 0, 0]))
    d = d @ vessel.rotation.T
    speed = np.linalg.norm(d, axis=1)
    if np.any(speed < 1e-10):
        raise ValueError(f'{name}: singular centreline tangent')
    tangent = d / speed[:, None]
    dr = np.full(s.shape, branch['radius_end_mm'] - branch['radius_start_mm'])
    if branch.get('smooth_taper', False):
        dr *= 6*s*(1-s)
    bulb = branch['bulb_amplitude_mm'] * np.exp(-((s-branch['bulb_center_s'])/branch['bulb_width_s'])**2)
    dr -= 2*(s-branch['bulb_center_s']) / branch['bulb_width_s']**2 * bulb
    slope = dr / speed
    if np.any(np.abs(slope) >= 1):
        raise ValueError(f'{name}: radius changes too quickly for a regular canal surface')
    radius = vessel.radius(name, s) * 1000
    centers = vessel.centerline(name, s) * 1000 - (radius*slope)[:, None]*tangent
    return centers, radius*np.sqrt(1-slope**2), tangent


def make_branch(vessel, name, stations, *, flat_ends=False):
    s = np.linspace(0, 1, stations)
    centers, radii, tangents = envelope(vessel, name, s)
    # A fixed reference avoids changing the seam of successive circles.
    ref = np.eye(3)[np.argmin(np.max(np.abs(tangents), axis=0))]
    wires = []
    for center, radius, tangent in zip(centers, radii, tangents):
        x = ref - np.dot(ref, tangent)*tangent
        x /= np.linalg.norm(x)
        y = np.cross(tangent, x)
        def point(angle):
            return cq.Vector(*(center+radius*(np.cos(angle)*x+np.sin(angle)*y)))
        # Four exact circular arcs avoid a periodic B-spline face at the loft
        # seam, which can change trim orientation when translated through STEP.
        edges = [cq.Edge.makeThreePointArc(point(a), point(a+np.pi/4), point(a+np.pi/2))
                 for a in np.arange(4)*np.pi/2]
        wires.append(cq.Wire.assembleEdges(edges))
    # Degree-three interpolation avoids high-degree oscillations and preserves
    # the intentionally aligned circular seams during STEP conversion.
    builder = BRepOffsetAPI_ThruSections(True, False, 1e-7)
    builder.CheckCompatibility(False)
    builder.SetMaxDegree(3)
    for wire in wires:
        builder.AddWire(wire.wrapped)
    builder.Build()
    body = cq.Shape.cast(builder.Shape())
    if flat_ends:
        if not body.isValid() or len(body.Solids()) != 1:
            raise RuntimeError(f'{name}: invalid flat-ended branch')
        return body
    # Sew the analytic spherical caps directly to the envelope loft. Boolean
    # unions of full balls and the loft are numerically fragile at tangency.
    faces = [face for face in body.Faces() if face.geomType() != 'PLANE']
    for index, s0, sign in ((0, 0.0, -1), (-1, 1.0, 1)):
        center = vessel.centerline(name, s0)*1000
        radius = float(vessel.radius(name, s0)*1000)
        outward = sign*tangents[index]
        latitude = np.degrees(np.arcsin(np.clip(np.dot(centers[index]-center, outward)/radius, -1, 1)))
        cap = cq.Solid.makeSphere(radius, cq.Vector(*center), cq.Vector(*outward),
                                 angleDegrees1=float(latitude), angleDegrees2=90)
        faces.extend(face for face in cap.Faces() if face.geomType() == 'SPHERE')
    body = cq.Solid.makeSolid(cq.Shell.makeShell(faces))
    if not body.isValid() or len(body.Solids()) != 1:
        raise RuntimeError(f'{name}: invalid or disconnected branch')
    return body



def outer_vessel(vessel):
    exterior = copy.deepcopy(vessel)
    diameter = vessel.config['main_trunk_pipe']['outer_diameter_mm']
    for name in ('RA', 'RV'):
        exterior.branches[name].update(radius_start_mm=diameter/2,
                                      radius_end_mm=diameter/2, bulb_amplitude_mm=0)
    return exterior


def bore_cutter(vessel, name, stations):
    """Curved 4 mm bore, extended through both flat terminal faces."""
    cutter = make_branch(vessel, name, stations, flat_ends=True)
    radius = vessel.config['main_trunk_pipe']['inner_diameter_mm']/2
    for s, sign in ((0., -1), (1., 1)):
        point = vessel.centerline(name, s)*1000
        outward = sign*vessel.frame(name, s)[0]
        # The 1 mm inward overlap makes the Boolean union robust at the port.
        extension = cq.Solid.makeCylinder(radius, 6., cq.Vector(*(point-outward)),
                                           cq.Vector(*outward))
        cutter = cutter.fuse(extension, tol=1e-6)
    if not cutter.isValid() or len(cutter.Solids()) != 1:
        raise RuntimeError(f'{name}: invalid continuous bore cutter')
    return cutter


def pipe_checks(vessel, name, body):
    """Check a clear bore, retained wall and two open annular terminal faces."""
    cfg = vessel.config['main_trunk_pipe']
    inner = cfg['inner_diameter_mm']/2
    outer = cfg['outer_diameter_mm']/2
    classifier = BRepClass3d_SolidClassifier(body.wrapped)
    def inside(point):
        classifier.Perform(gp_Pnt(*point), 1e-6)
        return classifier.State() == TopAbs_IN
    for s in np.linspace(0.02, 0.98, 49):
        center = vessel.centerline(name, s)*1000
        _, n, b = vessel.frame(name, s)
        for angle in np.linspace(0, 2*np.pi, 8, endpoint=False):
            direction = np.cos(angle)*n+np.sin(angle)*b
            if inside(center+(inner-.1)*direction):
                raise RuntimeError(f'{name}: blocked or undersized bore')
            if not inside(center+(inner+.1)*direction):
                raise RuntimeError(f'{name}: missing wall beside bore')
    ports = []
    expected = np.pi*(outer**2-inner**2)
    for s in (0., 1.):
        center = vessel.centerline(name, s)*1000
        faces = [face for face in body.Faces() if face.geomType() == 'PLANE'
                 and np.linalg.norm(np.asarray(face.Center().toTuple())-center) < 1e-3]
        if len(faces) != 1 or abs(faces[0].Area()-expected) > .02:
            raise RuntimeError(f'{name}: end is not the expected open annular face')
        diameters = sorted(wire.Length()/np.pi for wire in faces[0].Wires())
        if len(diameters) != 2 or not np.allclose(diameters, [2*inner, 2*outer], atol=1e-5, rtol=0):
            raise RuntimeError(f'{name}: incorrect inner or outer port diameter')
        ports.append({'center_mm': center.tolist(), 'annular_face_area_mm2': faces[0].Area(),
                      'measured_inner_diameter_mm': diameters[0], 'measured_outer_diameter_mm': diameters[1],
                      'boundary_wires': len(faces[0].Wires())})
        if len(faces[0].Wires()) != 2:
            raise RuntimeError(f'{name}: port does not contain a hole')
    return {'outer_diameter_mm': 2*outer, 'inner_diameter_mm': 2*inner,
            'nominal_wall_thickness_mm': outer-inner,
            'bore_wall_samples': 49*8, 'bore_clear': True, 'ends_open': True, 'ports': ports}


def branch_accuracy(vessel, name, body, stations):
    """Measure surface error at envelope circles between loft stations."""
    parameters = (np.arange(stations-1) + 0.5) / (stations-1)
    centers, radii, tangents = envelope(vessel, name, parameters)
    boundary = cq.Compound.makeCompound(body.Faces())
    errors = []
    for center, radius, tangent in zip(centers, radii, tangents):
        ref = np.eye(3)[np.argmin(np.abs(tangent))]
        x = np.cross(tangent, ref)
        x /= np.linalg.norm(x)
        y = np.cross(tangent, x)
        for angle in np.linspace(0, 2*np.pi, 8, endpoint=False):
            p = center + radius*(np.cos(angle)*x + np.sin(angle)*y)
            errors.append(cq.Vertex.makeVertex(*p).distance(boundary))
    maximum = float(max(errors))
    if maximum > 0.01:
        raise RuntimeError(f'{name}: sampled loft surface error {maximum:.6f} mm exceeds 0.01 mm')
    return maximum


def shape_record(shape):
    box = shape.BoundingBox()
    return {'valid': shape.isValid(), 'solid_count': len(shape.Solids()),
            'closed_shells': all(BRep_Tool.IsClosed_s(shell.wrapped) for shell in shape.Shells()),
            'volume_mm3': shape.Volume(1e-9), 'face_count': len(shape.Faces()),
            'bounds_mm': {'min': [box.xmin, box.ymin, box.zmin],
                          'max': [box.xmax, box.ymax, box.zmax]}}


def preview(all_models, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=(12, 7), layout='constrained')
    for index, (variant, networks) in enumerate(all_models.items(), 1):
        ax = fig.add_subplot(1, 2, index, projection='3d')
        points = []
        for network, shape in networks.items():
            vertices, faces = shape.tessellate(0.04, 0.15)
            verts = np.array([v.toTuple() for v in vertices])
            points.append(verts)
            ax.add_collection3d(Poly3DCollection(verts[np.asarray(faces)],
                                                facecolors=COLORS[network],
                                                linewidths=0, alpha=1.0,
                                                shade=True, lightsource=matplotlib.colors.LightSource(azdeg=315, altdeg=45)))
        pts = np.concatenate(points)
        low, high = pts.min(axis=0), pts.max(axis=0)
        span = high-low
        ax.set(xlim=(low[0]-2, high[0]+2), ylim=(low[1]-2, high[1]+2),
               zlim=(low[2]-2, high[2]+2), xlabel='x (mm)', ylabel='y (mm)', zlabel='z (mm)',
               title=variant.replace('_', ' ').capitalize())
        ax.set_box_aspect(span+4)
        ax.view_init(elev=28, azim=-65)
    fig.suptitle('Case Study 2 — hollow vessel CAD models\nMain trunks: 8 mm OD / 4 mm ID; artery (red), vein (blue); dimensions in mm')
    fig.savefig(out/'vessel_models_preview.png', dpi=180)
    plt.close(fig)
    # Dimensioned nominal cross section, shared by all four main trunks.
    from matplotlib.patches import Circle
    fig, ax = plt.subplots(figsize=(5, 4.5), layout='constrained')
    ax.add_patch(Circle((0, 0), 4, facecolor='#c25362', edgecolor='#333333', lw=1.2))
    ax.add_patch(Circle((0, 0), 2, facecolor='white', edgecolor='#333333', lw=1.2))
    ax.annotate('', xy=(-4, 4.7), xytext=(4, 4.7),
                arrowprops={'arrowstyle': '<->', 'color': '#252a30'})
    ax.text(0, 4.9, '8 mm outer diameter', ha='center', va='bottom', fontsize=11)
    ax.annotate('', xy=(-2, 0), xytext=(2, 0),
                arrowprops={'arrowstyle': '<->', 'color': '#252a30'})
    ax.text(0, .3, '4 mm bore', ha='center', va='bottom', fontsize=11)
    ax.annotate('2 mm wall', xy=(3, 0), xytext=(4.5, 1.5), fontsize=10,
                arrowprops={'arrowstyle': '->', 'color': '#252a30'})
    ax.set(xlim=(-5.5, 7.5), ylim=(-5, 6), aspect='equal')
    ax.set_axis_off()
    fig.suptitle('Main-trunk pipe cross section\nSame dimensions for artery and vein in both models', fontsize=12)
    for extension in ('png', 'pdf'):
        fig.savefig(out/f'pipe_cross_section.{extension}', dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--stations', type=int, default=65)
    args = parser.parse_args()
    if args.stations < 17:
        parser.error('--stations must be at least 17')
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = {'units': 'mm', 'cadquery_version': cq.__version__,
                'loft_stations_per_branch': args.stations,
                'representation': 'Main-trunk pipes: 8 mm OD / 4 mm ID, flat open ends, 2 mm nominal wall; solid side appendages with rounded tips.',
                'coordinate_frame': 'Publication eMNS frame; closest sampled trunk pair centred at (0, 0, 70.2) mm.',
                'accuracy_note': 'Sampled surface errors are diagnostics, not rigorous global bounds. Volumes use 1e-9 numerical integration tolerance.',
                'models': {}}
    models = {}
    for variant, source_name in VARIANTS.items():
        source = SOURCE / f'{source_name}.json'
        vessel, translation = registered_vessel(source)
        exterior = outer_vessel(vessel)
        pipe_records = {}
        networks, branch_records = {}, {}
        start = time.monotonic()
        for network in ('artery', 'vein'):
            branches = []
            for name, specification in vessel.branches.items():
                if specification['network'] != network:
                    continue
                body = make_branch(exterior, name, args.stations, flat_ends=name in ('RA', 'RV'))
                error = branch_accuracy(exterior, name, body, args.stations)
                branch_records[name] = {'sampled_envelope_error_mm': error, **shape_record(body)}
                branches.append(body)
                print(f'{variant}/{name}: solid valid, sampled surface error {error:.6f} mm', flush=True)
            body = branches[0]
            for branch in branches[1:]:
                body = body.fuse(branch, tol=1e-6)
            if (not body.isValid() or len(body.Solids()) != 1
                    or not all(BRep_Tool.IsClosed_s(shell.wrapped) for shell in body.Shells())):
                raise RuntimeError(f'{variant}/{network}: invalid or disconnected network')
            trunk = vessel.config['networks'][network]['trunk']
            body = body.cut(bore_cutter(vessel, trunk, args.stations), tol=1e-6)
            if not body.isValid() or len(body.Solids()) != 1:
                raise RuntimeError(f'{variant}/{network}: invalid hollow pipe')
            pipe_records[network] = pipe_checks(vessel, trunk, body)
            networks[network] = body
            print(f'{variant}/{network}: fused {len(branches)} branches', flush=True)
        overlap = networks['artery'].intersect(networks['vein']).Volume()
        if overlap > 1e-8:
            raise RuntimeError(f'{variant}: artery and vein overlap by {overlap} mm3')
        separation = networks['artery'].distance(networks['vein'])
        if separation <= 0:
            raise RuntimeError(f'{variant}: artery and vein touch')
        assembly = cq.Assembly(name=f'cs2_branched_{variant}')
        for network, body in networks.items():
            assembly.add(body, name=f'{network}_pipe', color=cq.Color(*COLORS[network]))
        destination = args.out / f'cs2_branched_{variant}.step'
        assembly.export(str(destination))
        imported = cq.importers.importStep(str(destination)).val()
        expected_volume = sum(body.Volume(1e-9) for body in networks.values())
        if not imported.isValid() or len(imported.Solids()) != 2:
            raise RuntimeError(f'{variant}: STEP round-trip did not retain two valid solids')
        imported_pipes = imported.Solids()
        round_trip_pipes = {network: pipe_checks(vessel, vessel.config['networks'][network]['trunk'], body)
                            for network, body in zip(('artery', 'vein'), imported_pipes)}
        relative_error = abs(imported.Volume(1e-9)-expected_volume)/expected_volume
        print(f'{variant}: source volume {expected_volume:.12f} mm3; STEP volume {imported.Volume(1e-9):.12f} mm3; relative error {relative_error:.9g}', flush=True)
        if relative_error > 1e-6:
            raise RuntimeError(f'{variant}: STEP round-trip changed volume')
        manifest['models'][variant] = {
            'main_trunk_pipe': vessel.config['main_trunk_pipe'],
            'pipe_checks': pipe_records,
            'uniform_scaling': json.loads(source.read_text()).get('uniform_scaling'),
            'source': str(source.relative_to(ROOT)),
            'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
            'translation_from_source_emns_mm': translation.tolist(),
            'step_file': destination.name,
            'step_sha256': hashlib.sha256(destination.read_bytes()).hexdigest(),
            'branches': branch_records, 'networks': {n: shape_record(s) for n, s in networks.items()},
            'minimum_network_gap_mm': separation, 'overlap_volume_mm3': overlap,
            'round_trip': {**shape_record(imported), 'relative_volume_error': relative_error,
                           'pipe_checks': round_trip_pipes}}
        models[variant] = networks
        print(f'Exported {destination.name} in {time.monotonic()-start:.1f}s; gap {separation:.6f} mm', flush=True)
        (args.out/'validation.json').write_text(json.dumps(manifest, indent=2)+'\n')
    preview(models, args.out)
    write_readme(args.out)


def write_readme(out):
    (out/'README.md').write_text(
        '# Case Study 2 — hollow STEP vessel models\n\n'
        'Two publication layouts: `cs2_branched_side_by_side.step` and `cs2_branched_over_under.step`. '
        'Each is a coloured assembly with two separate valid B-rep solids: artery and vein. '
        'All 14 source branch segments are included and fused at their intersections within each network.\n\n'
        'Main trunks are **8 mm outer diameter** and **4 mm inner diameter**, '
        'with a **2 mm nominal wall** and flat annular ports open at both ends. '
        'Each continuous bore follows its original curved main-trunk centreline. '
        'Side branches retain their previous exterior sizes and rounded tips; '
        'no separate side-branch bores were requested. Units are **millimetres**.\n\n'
        'The approximately 100 mm y length is retained. The networks were separated '
        'slightly along x (side-by-side) or z (over-under) to accommodate the 8 mm '
        'outer trunks, with a conservative 1 mm separating-slab gap. The closest '
        'sampled trunk midpoint is registered at (0, 0, 70.2) mm. Source radius '
        'fields describe the 4 mm internal lumen; `main_trunk_pipe` describes the '
        'external pipe dimensions. Particle dimensions and magnetic parameters '
        'are separate simulation settings.\n\n'
        '`validation.json` records scaling factors, source/output hashes, per-branch sampled surface errors, '
        'solid validity, dimensions, network separation, bore/wall samples, open annular ports and STEP re-import checks. '
        'Surface sampling is a diagnostic and does not prove a global approximation bound.\n\n'
        'Regenerate from the repository root with CadQuery '+cq.__version__+' and Matplotlib installed:\n\n'
        '```bash\npython -B -m revised_case_studies.cs2_vessel_microspheres.export_step\n```\n')


if __name__ == '__main__':
    main()
