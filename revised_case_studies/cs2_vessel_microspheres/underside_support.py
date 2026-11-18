"""Conservative, vertically monotone supports below the lowest vessel surface.

Raster cells are capped by the minimum vertex height of EVERY downward vessel
triangle whose expanded XY bounding box meets that cell. Taking the minimum
over adjacent cells at each terrain vertex ensures the entire terrain triangle
stays below its cell's cap. Thus no material can remain above a lower vessel at
a crossing. Absolute meshing, padding and a downward allowance accommodate
the CAD-to-mesh approximation. Dense ray audits separately verify clearance.
"""

from __future__ import annotations

import cadquery as cq
import manifold3d as manifold
import numpy as np
import trimesh
from trimesh.ray.ray_pyembree import RayMeshIntersector
from OCP.BRep import BRep_Builder
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.BRepBuilderAPI import (BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeFace,
                               BRepBuilderAPI_MakeVertex, BRepBuilderAPI_MakeWire)
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.BRepTools import BRepTools
from OCP.IntCurvesFace import IntCurvesFace_ShapeIntersector
from OCP.TopoDS import TopoDS, TopoDS_Shell
from OCP.gp import gp_Dir, gp_Lin, gp_Pnt

GRID_MM = .2
MESH_DEFLECTION_MM = .01
XY_PADDING_MM = .03
DOWNWARD_ALLOWANCE_MM = .04
HEIGHT_QUANTUM_MM = .05
SIMPLIFY_MM = .003


def triangulate(shape, tolerance=MESH_DEFLECTION_MM):
    if not BRepTools.Triangulation_s(shape.wrapped, tolerance):
        BRepTools.Clean_s(shape.wrapped)
        BRepMesh_IncrementalMesh(shape.wrapped, tolerance, False, .12, True)
    vertices, faces = shape.tessellate(tolerance, .12)
    return np.array([p.toTuple() for p in vertices]), np.array(faces)


def as_ray_mesh(shape):
    vertices, faces = triangulate(shape)
    return trimesh.Trimesh(vertices, faces, process=False)


def mesh_to_cad(vertices, triangles):
    """Share mesh vertices/edges directly; avoid expensive global sewing."""
    vertices = np.asarray(vertices)
    points = [BRepBuilderAPI_MakeVertex(gp_Pnt(*p)).Vertex() for p in vertices]
    edges = {}
    builder, shell = BRep_Builder(), TopoDS_Shell()
    builder.MakeShell(shell)
    for indices in triangles:
        wire = BRepBuilderAPI_MakeWire()
        for a, b in zip(indices, np.roll(indices, -1)):
            key = tuple(sorted((int(a), int(b))))
            if key not in edges:
                edges[key] = BRepBuilderAPI_MakeEdge(points[key[0]], points[key[1]]).Edge()
            edge = edges[key]
            wire.Add(edge if a < b else TopoDS.Edge_s(edge.Reversed()))
        face = BRepBuilderAPI_MakeFace(wire.Wire(), True).Face()
        normal = np.array(BRepAdaptor_Surface(face).Plane().Axis().Direction().Coord())
        p = vertices[indices]
        if np.dot(normal, np.cross(p[1]-p[0], p[2]-p[0])) < 0:
            face = TopoDS.Face_s(face.Reversed())
        builder.Add(shell, face)
    solid = cq.Solid.makeSolid(cq.Shell(shell)).clean()
    if not solid.isValid() or len(solid.Solids()) != 1:
        raise RuntimeError('Height-field CAD is not one valid solid')
    return solid


def bottom_only_support(template, networks, base_z, plate_thickness=3):
    box = template.BoundingBox()
    nx, ny = int(np.ceil(box.xlen/GRID_MM)), int(np.ceil(box.ylen/GRID_MM))
    x = np.linspace(box.xmin, box.xmax, nx+1)
    y = np.linspace(box.ymin, box.ymax, ny+1)
    spacing = np.array([x[1]-x[0], y[1]-y[0]])
    height = np.full((ny, nx), base_z+plate_thickness)
    limits = np.full_like(height, np.inf)
    source_triangles = 0

    def raster(triangles, target, minimum, padding):
        low = np.floor((triangles[:, :, :2].min(1)-padding-[x[0], y[0]])/spacing).astype(int)
        high = np.floor((triangles[:, :, :2].max(1)+padding-[x[0], y[0]])/spacing).astype(int)
        low = np.maximum(low, 0)
        high = np.minimum(high, [nx-1, ny-1])
        values = (triangles[:, :, 2].min(1)-DOWNWARD_ALLOWANCE_MM if minimum
                  else triangles[:, :, 2].max(1))
        for lo, hi, z in zip(low, high, values):
            if (hi < lo).any():
                continue
            part = target[lo[1]:hi[1]+1, lo[0]:hi[0]+1]
            if minimum:
                np.minimum(part, z, out=part)
            else:
                np.maximum(part, z, out=part)

    vertices, faces = triangulate(template, .02)
    triangles = vertices[faces]
    normals = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
    raster(triangles[normals[:, 2] > 1e-10], height, False, 0)
    for shape in networks.values():
        vertices, faces = triangulate(shape)
        triangles = vertices[faces]
        normals = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
        downward = triangles[normals[:, 2] < -1e-10]
        source_triangles += len(downward)
        raster(downward, limits, True, XY_PADDING_MM)
    height = np.floor(np.minimum(height, limits)/HEIGHT_QUANTUM_MM)*HEIGHT_QUANTUM_MM
    corners = np.full((ny+1, nx+1), np.inf)
    for yy, xx in ((0, 0), (0, 1), (1, 0), (1, 1)):
        part = corners[yy:yy+ny, xx:xx+nx]
        np.minimum(part, height, out=part)
    ceiling = np.maximum.reduce([corners[:-1, :-1], corners[:-1, 1:],
                                 corners[1:, :-1], corners[1:, 1:]])
    finite = np.isfinite(limits)
    bound_error = float(np.max((ceiling-limits)[finite]))
    if bound_error > 1e-9 or corners.min() < base_z+plate_thickness-1e-8:
        raise RuntimeError('Terrain violates the vessel cap or base-plate height')
    xx, yy = np.meshgrid(x, y)
    vertices = np.column_stack([xx.ravel(), yy.ravel(), corners.ravel()])
    row, column = np.arange(ny)[:, None], np.arange(nx)[None, :]
    a = (row*(nx+1)+column).ravel()
    b, c = a+1, a+nx+1
    d = c+1
    faces = np.vstack([np.column_stack([a, b, c]), np.column_stack([b, d, c])])
    perimeter = np.concatenate([np.arange(nx+1), np.arange(1, ny+1)*(nx+1)+nx,
                                ny*(nx+1)+np.arange(nx-1, -1, -1),
                                np.arange(ny-1, 0, -1)*(nx+1)])
    count = len(vertices)
    bottom = vertices[perimeter].copy()
    bottom[:, 2] = base_z
    center = np.array([(x[0]+x[-1])/2, (y[0]+y[-1])/2, base_z])
    vertices = np.vstack([vertices, bottom, center])
    q = count+np.arange(len(perimeter))
    q_next, p_next = np.roll(q, -1), np.roll(perimeter, -1)
    faces = np.vstack([faces, np.column_stack([perimeter, q, q_next]),
                       np.column_stack([perimeter, q_next, p_next]),
                       np.column_stack([np.full(len(q), len(vertices)-1), q_next, q])])
    mesh = manifold.Manifold(manifold.Mesh64(vertices, faces.astype(np.uint64))).simplify(SIMPLIFY_MM)
    if mesh.status() != manifold.Error.NoError or len(mesh.decompose()) != 1:
        raise RuntimeError('Height field is not a connected manifold')
    mesh = mesh.to_mesh64()
    print(f'Bottom-only terrain: {len(mesh.tri_verts)} triangles; converting to CAD', flush=True)
    solid = mesh_to_cad(mesh.vert_properties[:, :3], mesh.tri_verts)
    return solid, {
        'construction': 'Monotone height field capped below the lowest vessel across both networks.',
        'grid_spacing_mm': GRID_MM, 'absolute_vessel_mesh_deflection_mm': MESH_DEFLECTION_MM,
        'xy_shadow_padding_mm': XY_PADDING_MM, 'downward_allowance_mm': DOWNWARD_ALLOWANCE_MM,
        'downward_height_quantization_mm': HEIGHT_QUANTUM_MM, 'mesh_simplification_mm': SIMPLIFY_MM,
        'source_downward_triangles': source_triangles, 'capped_raster_cells': int(finite.sum()),
        'max_pre_simplification_cell_ceiling_minus_vessel_cap_mm': bound_error,
        'clearance_note': 'Conservative approximated underside, with variable additional clearance from cell minima and downward quantization.'}


def support_tops(ray, xy, base_z):
    origins = np.column_stack([xy, np.full(len(xy), base_z-1)])
    directions = np.tile([0., 0., 1.], (len(xy), 1))
    locations, indices, _ = ray.intersects_location(origins, directions, multiple_hits=True)
    tops = np.full(len(xy), -np.inf)
    np.maximum.at(tops, indices, locations[:, 2])
    if not np.isfinite(tops).all():
        raise RuntimeError('Support does not cover every audit ray with its base')
    return tops


def clearance_checks(vessel, networks, support, base_z, dense=False):
    support_ray = RayMeshIntersector(as_ray_mesh(support))
    source_rays = {}
    for name, shape in networks.items():
        ray = IntCurvesFace_ShapeIntersector()
        ray.Load(shape.wrapped, 1e-7)
        source_rays[name] = ray
    xy, labels, branches = [], [], {}
    for name in vessel.branches:
        begin = len(xy)
        for s in np.linspace(.02, .98, 25):
            center = vessel.centerline(name, s)*1000
            tangent = vessel.frame(name, s)[0]
            lateral = np.array([-tangent[1], tangent[0], 0.])
            lateral /= np.linalg.norm(lateral)
            radius = 4. if name in ('RA', 'RV') else float(vessel.radius(name, s)*1000)
            for fraction in (-.8, -.5, 0, .5, .8):
                xy.append((center+fraction*radius*lateral)[:2])
                labels.append(name)
        branches[name] = (begin, len(xy))
    xy = np.array(xy)
    first = np.full(len(xy), np.inf)
    first_network = np.full(len(xy), '', dtype=object)
    for index, p in enumerate(xy):
        for name, ray in source_rays.items():
            ray.Perform(gp_Lin(gp_Pnt(p[0], p[1], base_z-1), gp_Dir(0, 0, 1)), 0, 100)
            for hit in range(1, ray.NbPnt()+1):
                z = ray.Pnt(hit).Z()
                if z < first[index]:
                    first[index], first_network[index] = z, name
    if not np.isfinite(first).all():
        raise RuntimeError('An underside audit ray did not intersect a vessel')
    tops = support_tops(support_ray, xy, base_z)
    clearance = first-tops
    if clearance.min() < -1e-5:
        index = int(np.argmin(clearance))
        raise RuntimeError(f'Support above lowest vessel at {xy[index]} by {-clearance[index]} mm')
    records = {}
    for name, (lo, hi) in branches.items():
        gaps = clearance[lo:hi]
        if np.min(tops[lo:hi]) <= base_z+3+.01:
            raise RuntimeError(f'{name}: missing elevated support below an audit ray')
        records[name] = {'vertical_samples': hi-lo, 'min_clearance_to_lowest_vessel_mm': float(gaps.min()),
                         'max_clearance_to_lowest_vessel_mm': float(gaps.max()),
                         'lower_other_network_shadow_samples': int(np.sum(first_network[lo:hi] != vessel.branches[name]['network']))}
    result = {'branch_vertical_clearance': records, 'exact_cad_vessel_rays': len(xy),
              'maximum_support_above_first_vessel_mm': float(np.max(tops-first)),
              'no_support_at_or_above_lowest_vessel': True}
    if dense:
        meshes = [as_ray_mesh(shape) for shape in networks.values()]
        source_mesh = trimesh.util.concatenate(meshes)
        source_ray = RayMeshIntersector(source_mesh)
        triangles = source_mesh.triangles
        normal = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
        sample = triangles[normal[:, 2] < -1e-10].mean(axis=1)[:, :2]
        origins = np.column_stack([sample, np.full(len(sample), base_z-1)])
        directions = np.tile([0., 0., 1.], (len(sample), 1))
        hits, indices, _ = source_ray.intersects_location(origins, directions, multiple_hits=False)
        source_z = np.full(len(sample), np.inf)
        source_z[indices] = hits[:, 2]
        valid = np.isfinite(source_z)
        overhead = support_tops(support_ray, sample[valid], base_z)-source_z[valid]
        if overhead.max() > 1e-5:
            raise RuntimeError('Dense shadow audit found overhead support')
        result['dense_mesh_shadow_audit'] = {'tested_rays': int(valid.sum()),
                                            'maximum_support_above_first_vessel_mm': float(overhead.max()),
                                            'overhead_ray_count': int(np.sum(overhead > 1e-5))}
    box = support.BoundingBox()
    if abs(box.zmin-base_z) > 1e-5 or not support.isValid() or len(support.Solids()) != 1:
        raise RuntimeError('Support validity, connectivity or base-plane check failed')
    print(f'Clearance checked: {len(xy)} exact CAD rays; maximum overhead {np.max(tops-first):.6f} mm', flush=True)
    return result
