"""Create separate underside support walls without rewriting the pure vessel CAD.

Run from the repository root:
    python -B -m revised_case_studies.cs2_vessel_microspheres.export_supports

Units and coordinates match the existing vessel STEP files (millimetres).
The support starts at z=40; a 3 mm plate carries one wall under every branch.
Each wall follows its branch centreline, is as wide as the branch but never
narrower than 2 mm or wider than 8 mm, and ends in a cradle cut by the original
vessel surface. Everything above any vessel is removed, so both networks can be
lowered onto the support from above.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
import zipfile

import cadquery as cq
import numpy as np
from OCP.BRepBndLib import BRepBndLib
from OCP.BRepOffsetAPI import BRepOffsetAPI_ThruSections
from OCP.Bnd import Bnd_Box
from OCP.IntCurvesFace import IntCurvesFace_ShapeIntersector
from OCP.gp import gp_Dir, gp_Lin, gp_Pnt

from .export_step import (
    COLORS, DEFAULT_OUT, ROOT, SOURCE, VARIANTS, outer_vessel,
    pipe_checks, registered_vessel, shape_record,
)

BASE_Z = 40.0
BASE_THICKNESS = 3.0
BASE_MARGIN = 3.0
MIN_WALL_WIDTH = 2.0
MAX_WALL_WIDTH = 8.0
# The wall top is this fraction of the outer radius below the centreline. The
# vessel then cuts a 120 degree cradle that stays below its widest section.
CRADLE_DEPTH = 0.5
# Overhead clearance starts inside each vessel, above the cradle and safely
# below the widest section, where the vessel surface begins to face upward.
OVERHEAD_DEPTH = 0.4
SHADOW_PADDING = 0.05
TRUNK_END_SETBACK = 1.0
TRUNKS = ('RA', 'RV')
SUPPORT_COLOR = (.72, .74, .76)
OUT = DEFAULT_OUT / 'supports'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def branch_frames(exterior, name, s):
    """Centreline stations with horizontal section directions that never cross.

    Sections perpendicular to the centreline fold over each other on the inside
    of twig bends tighter than the wall half-width. Smoothing the section
    direction along the branch removes the fold; the section is lengthened by
    1/cos so that the width measured across the branch is unchanged.
    """
    points = exterior.centerline(name, s)*1000
    radius = exterior.radius(name, s)*1000
    tangent = np.gradient(points[:, :2], axis=0)
    tangent /= np.linalg.norm(tangent, axis=1)[:, None]
    normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))
    half = np.clip(radius, MIN_WALL_WIDTH/2, MAX_WALL_WIDTH/2)
    reach = np.maximum(half, radius+SHADOW_PADDING)
    arc = np.concatenate(([0], np.cumsum(np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1))))
    for sigma in (0, .5, 1, 1.5, 2, 3, 4):
        lateral = normal
        if sigma:
            lateral = np.exp(-.5*((arc[:, None]-arc[None, :])/sigma)**2) @ normal
            lateral /= np.linalg.norm(lateral, axis=1)[:, None]
        cosine = np.sum(lateral*normal, axis=1)
        forward = np.column_stack((lateral[:, 1], -lateral[:, 0]))
        # Both ends of each section, with a 50% margin, lie ahead of the previous one.
        extent = (1.5*reach/cosine)[:, None]*lateral
        if all(np.all(np.sum(np.diff(points[:, :2]+side*extent, axis=0)*forward[:-1], axis=1) > 0)
               for side in (-1, 1)):
            break
    else:
        raise RuntimeError(f'{name}: support sections cross at every smoothing length')
    if name in TRUNKS and sigma:
        raise RuntimeError(f'{name}: trunk wall sections must stay perpendicular')
    return {'points': points, 'radius': radius, 'lateral': lateral, 'forward': forward,
            'cosine': cosine, 'half_width': half, 'smoothing_mm': sigma}


def extended(frame, start, end):
    """Add one station before and after the branch, along its end directions."""
    points = np.vstack([frame['points'][0], frame['points'], frame['points'][-1]])
    points[0, :2] -= start*frame['forward'][0]
    points[-1, :2] += end*frame['forward'][-1]
    keep = slice(0 if start else 1, len(points) if end else -1)
    pad = lambda values: np.concatenate([values[:1], values, values[-1:]])[keep]
    return points[keep], pad(frame['lateral']), pad(frame['cosine']), pad(frame['radius'])


def loft(name, profiles):
    builder = BRepOffsetAPI_ThruSections(True, False, 1e-7)
    builder.CheckCompatibility(False)
    builder.SetMaxDegree(3)
    for profile in profiles:
        builder.AddWire(cq.Wire.makePolygon([cq.Vector(*p) for p in profile], close=True).wrapped)
    builder.Build()
    body = cq.Shape.cast(builder.Shape())
    if not body.isValid() or len(body.Solids()) != 1:
        raise RuntimeError(f'{name}: invalid support loft')
    return body


def section(point, lateral, corners):
    """Polygon in the vertical plane through a station; corners are (offset, z)."""
    return [(*(point[:2]+offset*lateral), z) for offset, z in corners]


def wall(name, frame, bottom):
    """Vertical wall under one branch, up to its cradle height."""
    tip = 0 if name in TRUNKS else frame['radius'][-1]  # under the rounded tip
    points, lateral, cosine, radius = extended(frame, 0, tip)
    half = np.clip(radius, MIN_WALL_WIDTH/2, MAX_WALL_WIDTH/2)/cosine
    tops = points[:, 2]-CRADLE_DEPTH*radius
    return loft(name, [section(p, n, ((-h, bottom), (h, bottom), (h, top), (-h, top)))
                       for p, n, h, top in zip(points, lateral, half, tops)])


def overhead(name, frame, top, pipe):
    """Everything above one branch, from just below its widest section upward.

    Under a main trunk the cutter also reaches down into the pipe, which clears
    its bore; the cutter stays inside the outer pipe surface there.
    """
    reach = frame['radius'][[0, -1]]+SHADOW_PADDING
    points, lateral, cosine, radius = extended(frame, *reach)
    half = (radius+SHADOW_PADDING)/cosine
    profiles = []
    for p, n, h, r in zip(points, lateral, half, radius):
        low = p[2]-OVERHEAD_DEPTH*r
        corners = [(h, low), (h, top), (-h, top), (-h, low)]
        if name in TRUNKS:
            plug = 1.1*pipe['inner_diameter_mm']/2
            depth = .9*np.sqrt((pipe['outer_diameter_mm']/2)**2-plug**2)
            corners += [(-plug, low), (-plug, p[2]-depth), (plug, p[2]-depth), (plug, low)]
        profiles.append(section(p, n, corners))
    return loft(name, profiles)


def single_solid(shape, step):
    solids = shape.Solids()
    if len(solids) != 1 or not solids[0].isValid():
        raise RuntimeError(f'Support is invalid or disconnected after {step}')
    return solids[0]


def build_support(vessel, exterior, networks, base_z, stations):
    box = cq.Compound.makeCompound(list(networks.values())).BoundingBox()
    if base_z+BASE_THICKNESS >= box.zmin:
        raise RuntimeError('The base plate must be below the lowest vessel')
    low = cq.Vector(box.xmin-BASE_MARGIN, box.ymin-BASE_MARGIN, base_z)
    support = cq.Solid.makeBox(box.xlen+2*BASE_MARGIN, box.ylen+2*BASE_MARGIN, BASE_THICKNESS, low)
    walls, cutters, frames = {}, {}, {}
    for name in vessel.branches:
        s = np.linspace(0, 1, stations)
        frames[name] = branch_frames(exterior, name, s)
        wall_frame = frames[name]
        if name in TRUNKS:
            # Keep the wall clear of both open ports.
            dense = exterior.centerline(name, np.linspace(0, 1, 400))*1000
            inset = TRUNK_END_SETBACK/np.linalg.norm(np.diff(dense, axis=0), axis=1).sum()
            wall_frame = branch_frames(exterior, name, np.linspace(inset, 1-inset, stations))
        # Wall bottoms overlap the plate by 1 mm, so no faces coincide in the union.
        walls[name] = wall(name, wall_frame, base_z+BASE_THICKNESS-1)
        cutters[name] = overhead(name, frames[name], box.zmax+5, vessel.config['main_trunk_pipe'])
    for name, body in walls.items():
        support = single_solid(support.fuse(body, tol=1e-6), f'adding the {name} wall')
    print('Walls fused to base', flush=True)
    for name, body in {**cutters, **networks}.items():
        support = single_solid(support.cut(body, tol=1e-6), f'clearing {name}')
        print(f'{name}: cleared', flush=True)
    return support, frames


def vertical_hits(shape):
    intersector = IntCurvesFace_ShapeIntersector()
    intersector.Load(shape.wrapped, 1e-7)
    def hits(xy, base_z):
        intersector.Perform(gp_Lin(gp_Pnt(xy[0], xy[1], base_z-1), gp_Dir(0, 0, 1)), 0, 200)
        return [intersector.Pnt(i).Z() for i in range(1, intersector.NbPnt()+1)]
    return hits


def support_checks(exterior, networks, support, frames, base_z):
    """Audit the finished support against the imported vessel solids."""
    box = support.BoundingBox()
    if abs(box.zmin-base_z) > 1e-5 or not support.isValid() or len(support.Solids()) != 1:
        raise RuntimeError('Support validity, connectivity or base-plane check failed')
    overlap = {name: support.intersect(shape).Volume() for name, shape in networks.items()}
    if max(overlap.values()) > 1e-6:
        raise RuntimeError(f'Support overlaps a vessel: {overlap}')
    vessel_hits = [vertical_hits(shape) for shape in networks.values()]
    support_hits = vertical_hits(support)

    def clearance(xy):
        """Gap from the support top to the first vessel surface above the base."""
        first = min((z for hits in vessel_hits for z in hits(xy, base_z)), default=None)
        return None if first is None else first-max(support_hits(xy, base_z))

    # Nothing above any vessel: rays across every branch and through mesh vertices.
    samples = []
    for name, frame in frames.items():
        for fraction in (-.99, -.9, -.6, -.3, 0, .3, .6, .9, .99):
            samples.extend(frame['points'][:, :2]+fraction*(frame['radius']/frame['cosine'])[:, None]*frame['lateral'])
    random = np.random.default_rng(0)
    for shape in networks.values():
        vertices = np.array([v.toTuple() for v in shape.tessellate(.05, .2)[0]])
        samples.extend(random.choice(vertices, 4000, replace=False)[:, :2])
    gaps = np.array([gap for gap in map(clearance, samples) if gap is not None])
    if gaps.min() < -1e-4:
        raise RuntimeError(f'Support reaches {-gaps.min()} mm above a vessel underside')

    # Contact under each centreline. A gap is expected only where another
    # vessel passes below, or beside a trunk port.
    dense = {name: (exterior.centerline(name, np.linspace(0, 1, 801))*1000,
                    exterior.radius(name, np.linspace(0, 1, 801))*1000) for name in frames}
    records = {}
    for name, frame in frames.items():
        centre = np.array([clearance(xy) for xy in frame['points'][:, :2]])
        open_stations = np.flatnonzero(centre > 1e-3)
        unexplained = 0
        for index in open_stations:
            xy = frame['points'][index, :2]
            shadowed = any(np.min(np.linalg.norm(points[:, :2]-xy, axis=1)-radius) < 2*SHADOW_PADDING
                           for other, (points, radius) in dense.items() if other != name)
            port = name in TRUNKS and min(index, len(centre)-1-index) < 3
            unexplained += not (shadowed or port)
        if unexplained or len(open_stations) > len(centre)/2:
            raise RuntimeError(f'{name}: underside is not supported along its wall')
        width = 2*np.clip(frame['radius'], MIN_WALL_WIDTH/2, MAX_WALL_WIDTH/2)
        records[name] = {'wall_width_mm': [float(width.min()), float(width.max())],
                         'section_smoothing_mm': frame['smoothing_mm'],
                         'centreline_stations': len(centre),
                         'stations_in_contact': int(len(centre)-len(open_stations)),
                         'stations_open_over_another_vessel_or_port': int(len(open_stations))}

    # Walls follow the vessels: no wall point is farther than 4 mm from a centreline.
    vertices = np.array([v.toTuple() for v in support.tessellate(.05, .2)[0]])
    vertices = vertices[vertices[:, 2] > base_z+BASE_THICKNESS+1e-3]
    centres = np.concatenate([points[:, :2] for points, _ in dense.values()])
    distance = np.concatenate([np.linalg.norm(part[:, None, :2]-centres[None], axis=2).min(axis=1)
                               for part in np.array_split(vertices, len(vertices)//2000+1)])
    if distance.max() > MAX_WALL_WIDTH/2+1e-3:
        raise RuntimeError(f'A wall extends {distance.max()} mm from the nearest centreline')
    print(f'Checked {len(gaps)} vertical rays; highest support is {gaps.min():.6f} mm below a vessel', flush=True)
    return {'branches': records, 'vessel_overlap_volume_mm3': overlap,
            'vertical_rays_through_vessels': len(gaps),
            'min_gap_support_top_to_first_vessel_surface_mm': float(gaps.min()),
            'no_support_above_any_vessel': True,
            'max_wall_distance_from_nearest_centreline_mm': float(distance.max())}


def vessel_round_trip_checks(original, imported):
    records = {}
    for name, source in original.items():
        shape = imported[name]
        volume_error = abs(shape.Volume(1e-9)-source.Volume(1e-9))/source.Volume(1e-9)
        # Tessellation changes the default box through mesh deflection margins.
        # Compare the underlying CAD surfaces on both sides of the export.
        bounds = []
        for body in (source, shape):
            box = Bnd_Box()
            BRepBndLib.AddOptimal_s(body.wrapped, box, False, False)
            bounds.append(np.array(box.Get()))
        bounds_error = float(np.max(np.abs(bounds[0]-bounds[1])))
        if volume_error > 1e-6 or bounds_error > 1e-4:
            raise RuntimeError(f'{name}: assembly export changed vessel volume or position')
        records[name] = {'relative_volume_error': volume_error,
                         'max_bounding_coordinate_error_mm': bounds_error}
    return records


def preview(models, out, base_z):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from matplotlib.collections import PolyCollection

    fig = plt.figure(figsize=(16, 10), layout='constrained')
    for row, (variant, (networks, support)) in enumerate(models.items()):
        meshes = {}
        for name, shape in {**networks, 'support': support}.items():
            vertices, faces = shape.tessellate(.08, .2)
            vertices = np.array([v.toTuple() for v in vertices])
            meshes[name] = vertices[np.asarray(faces)]
        box = support.BoundingBox()
        high = max(shape.BoundingBox().zmax for shape in networks.values())
        title = variant.replace('_', ' ').capitalize()
        for column in (0, 1):
            ax = fig.add_subplot(2, 3, row*3+column+1, projection='3d')
            names = ['support', *networks] if column == 0 else ['support']
            triangles = np.concatenate([meshes[name] for name in names])
            colors = np.concatenate([np.tile(COLORS.get(name, SUPPORT_COLOR),
                                             (len(meshes[name]), 1)) for name in names])
            # One collection sorts individual triangles across all bodies;
            # separate collections incorrectly draw entire supports in front
            # of an artery even where only their contact surfaces coincide.
            ax.add_collection3d(Poly3DCollection(
                triangles, facecolors=colors, linewidths=0, antialiased=False, shade=True,
                lightsource=matplotlib.colors.LightSource(azdeg=315, altdeg=45)))
            ax.set(xlim=(box.xmin-2, box.xmax+2), ylim=(box.ymin-2, box.ymax+2),
                   zlim=(base_z-1, high+3), xlabel='x (mm)', ylabel='y (mm)', zlabel='z (mm)',
                   title=f'{title}: '+('assembly' if column == 0 else 'support only'))
            ax.set_box_aspect((box.xlen+4, box.ylen+4, high-base_z+4))
            ax.view_init(elev=30, azim=-65)
            ax.set_proj_type('ortho')
        ax = fig.add_subplot(2, 3, row*3+3)
        # Draw top-facing surfaces in increasing height for an opaque top view.
        triangles = np.concatenate(list(meshes.values()))
        colors = np.concatenate([np.tile(COLORS.get(name, SUPPORT_COLOR), (len(mesh), 1))
                                 for name, mesh in meshes.items()])
        order = np.argsort(triangles[:, :, 2].mean(axis=1))
        normals = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
        order = order[normals[order, 2] > 1e-10]
        ax.add_collection(PolyCollection(triangles[order, :, :2], facecolors=colors[order],
                                         edgecolors='none', linewidths=0, antialiased=False))
        ax.set(xlim=(box.xmin-2, box.xmax+2), ylim=(box.ymin-2, box.ymax+2),
               xlabel='x (mm)', ylabel='y (mm)', title=f'{title}: top view', aspect='equal')
        ax.grid(alpha=.15)
    fig.suptitle('Underside support walls for CS2 vessels — base plane z = '+f'{base_z:g} mm\n'
                 'Existing vessels: artery (red), vein (blue). New walls and base: grey. '
                 'No support above any vessel.', fontsize=14)
    fig.savefig(out/'supports_preview.png', dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--base-z', type=float, default=BASE_Z)
    parser.add_argument('--stations', type=int, default=161)
    parser.add_argument('--variant', choices=tuple(VARIANTS))
    args = parser.parse_args()
    if args.stations < 17:
        parser.error('--stations must be at least 17')
    args.out.mkdir(parents=True, exist_ok=True)
    originals = {p: digest(p) for p in DEFAULT_OUT.iterdir() if p.is_file()}
    originals.update({SOURCE/f'{s}.json': digest(SOURCE/f'{s}.json') for s in VARIANTS.values()})
    vessel_manifest = json.loads((DEFAULT_OUT/'validation.json').read_text())
    manifest = {'units': 'mm', 'base_z_mm': args.base_z,
                'base_plate_thickness_mm': BASE_THICKNESS, 'base_xy_margin_mm': BASE_MARGIN,
                'wall_width_limits_mm': [MIN_WALL_WIDTH, MAX_WALL_WIDTH],
                'wall_top_below_centreline_outer_radii': CRADLE_DEPTH,
                'overhead_clearance_xy_padding_mm': SHADOW_PADDING,
                'overhead_clearance_starts_below_centreline_outer_radii': OVERHEAD_DEPTH,
                'trunk_wall_setback_from_ports_mm': TRUNK_END_SETBACK,
                'contact_clearance_mm': 0, 'stations_per_branch': args.stations,
                'wall_representation': 'One lofted wall under each branch centreline, cut by the original STEP vessel surfaces; all material above any vessel removed.',
                'vessel_files_modified': False, 'cadquery_version': cq.__version__, 'models': {}}
    models = {}
    for variant, source_name in VARIANTS.items():
        if args.variant and args.variant != variant:
            continue
        start = time.monotonic()
        source = SOURCE/f'{source_name}.json'
        step = DEFAULT_OUT/f'cs2_branched_{variant}.step'
        original_record = vessel_manifest['models'][variant]
        if digest(source) != original_record['source_sha256'] or digest(step) != original_record['step_sha256']:
            raise RuntimeError(f'{variant}: vessel CAD provenance does not match its source')
        vessel, _ = registered_vessel(source)
        exterior = outer_vessel(vessel)
        shapes = cq.importers.importStep(str(step)).val().Solids()
        if len(shapes) != 2 or not all(shape.isValid() for shape in shapes):
            raise RuntimeError(f'{variant}: invalid source vessel assembly')
        networks = dict(zip(('artery', 'vein'), shapes))
        support, frames = build_support(vessel, exterior, networks, args.base_z, args.stations)
        checks = support_checks(exterior, networks, support, frames, args.base_z)
        support_path = args.out/f'cs2_branched_{variant}_support.step'
        support_assembly = cq.Assembly(name=f'cs2_{variant}_support')
        support_assembly.add(support, name='support', color=cq.Color(*SUPPORT_COLOR))
        support_assembly.export(str(support_path))
        imported = cq.importers.importStep(str(support_path)).val()
        if not imported.isValid() or len(imported.Solids()) != 1:
            raise RuntimeError(f'{variant}: support STEP re-import failed')
        round_trip_checks = support_checks(exterior, networks, imported, frames, args.base_z)
        volume_error = abs(imported.Volume(1e-9)-support.Volume(1e-9))/support.Volume(1e-9)
        if volume_error > 1e-6:
            raise RuntimeError(f'{variant}: support STEP volume changed')
        assembly = cq.Assembly(name=f'cs2_{variant}_with_support')
        for network, shape in networks.items():
            assembly.add(shape, name=f'{network}_pipe', color=cq.Color(*COLORS[network]))
        assembly.add(support, name='support', color=cq.Color(*SUPPORT_COLOR))
        assembly_path = args.out/f'cs2_branched_{variant}_with_support.step'
        assembly.export(str(assembly_path))
        imported_assembly = cq.importers.importStep(str(assembly_path)).val()
        if not imported_assembly.isValid() or len(imported_assembly.Solids()) != 3:
            raise RuntimeError(f'{variant}: assembly STEP re-import failed')
        assembly_solids = imported_assembly.Solids()
        assembly_networks = dict(zip(('artery', 'vein'), assembly_solids[:2]))
        for network, shape in assembly_networks.items():
            pipe_checks(vessel, vessel.config['networks'][network]['trunk'], shape)
        assembly_vessel_checks = vessel_round_trip_checks(networks, assembly_networks)
        manifest['models'][variant] = {
            'source_vessel_step': str(step.relative_to(ROOT)), 'source_vessel_sha256': digest(step),
            'source_geometry_sha256': digest(source), 'support_step': support_path.name,
            'support_sha256': digest(support_path), 'assembly_step': assembly_path.name,
            'assembly_sha256': digest(assembly_path), 'support': shape_record(support),
            'checks': checks,
            'round_trip': {**shape_record(imported), 'relative_volume_error': volume_error,
                           'checks': round_trip_checks},
            'assembly_round_trip': {**shape_record(imported_assembly),
                                    'vessel_geometry_checks': assembly_vessel_checks}}
        models[variant] = networks, support
        (args.out/'validation.json').write_text(json.dumps(manifest, indent=2)+'\n')
        print(f'{variant}: support and assembly exported in {time.monotonic()-start:.1f}s', flush=True)
    if any(digest(path) != expected for path, expected in originals.items()):
        raise RuntimeError('An original vessel source or CAD file changed during export')
    preview(models, args.out, args.base_z)
    (args.out/'README.md').write_text(
        '# Underside support walls for the CS2 vessels\n\n'
        'Each layout has a support-only STEP and an assembly STEP containing the '
        'original artery, original vein and new support as three separate solids. '
        'The two pure vessel STEP files and their source geometries are preserved byte for byte.\n\n'
        f'All dimensions are millimetres in the existing eMNS coordinate frame. '
        f'The support underside is the plane **z = {args.base_z:g} mm**. A {BASE_THICKNESS:g} mm plate, '
        f'extending {BASE_MARGIN:g} mm beyond the vessel XY bounds, carries one vertical wall under '
        'each of the 14 branch segments. Every wall follows its branch centreline and is as wide as '
        f'the branch, but never narrower than **{MIN_WALL_WIDTH:g} mm** or wider than '
        f'**{MAX_WALL_WIDTH:g} mm**: {MAX_WALL_WIDTH:g} mm under the main trunks and '
        f'{MIN_WALL_WIDTH:g} to about 2.6 mm under the side branches.\n\n'
        f'A wall rises to {CRADLE_DEPTH:g} outer radii below its centreline and is then cut by the '
        'original vessel surface, which leaves a cradle of about 120 degrees with nominal '
        'zero-clearance contact on the underside. **No support material is above any vessel**: '
        'where one vessel passes over another, the wall of the upper vessel is interrupted above '
        f'the lower one (with {SHADOW_PADDING:g} mm lateral padding), so both networks can be lowered '
        'onto the support from above. The upper vessel is carried on either side of such a crossing. '
        f'Trunk walls stop {TRUNK_END_SETBACK:g} mm short of the ports, and both 4 mm bores are clear.\n\n'
        'These are geometry models; no manufacturing fit allowance, fastening or structural load '
        'rating is specified.\n\n'
        '`validation.json` records source hashes, solid validity, the base plane, zero interference '
        'volumes, vertical rays showing that the support top never exceeds the first vessel surface, '
        'contact stations under every branch, the largest wall distance from a centreline and STEP '
        're-import checks. Ray sampling is a diagnostic, not a global bound.\n\n'
        'Regenerate from the repository root using CadQuery, NumPy and Matplotlib:\n\n'
        '```bash\npython -B -m revised_case_studies.cs2_vessel_microspheres.export_supports\n```\n')
    if not args.variant:
        with zipfile.ZipFile(args.out/'cs2_vessel_supports.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(args.out.iterdir()):
                if path.suffix != '.zip':
                    archive.write(path, path.name)


if __name__ == '__main__':
    main()
