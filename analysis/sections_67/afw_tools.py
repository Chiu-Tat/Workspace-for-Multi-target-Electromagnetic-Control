"""Pointwise AFW helpers that do not reuse the hyperplane-construction code."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog

from emns.workspace.fast import Map_I2H, ModifiedHyperplaneShiftingMethod
from analysis.sections_67.common import PAPER_Z_OFFSET_M


FIELD_SCALE = 1.0
FORCE_SCALE = 1.0
DEFAULT_FIELD_TOL_T = 1e-7
DEFAULT_FORCE_TOL_N = 1e-6


def actuation_map(target_points: list[dict], active_coil_indices=None):
    """Return ``G``, ``c`` for the selected outputs of ``target_points``."""

    augmented = np.asarray(Map_I2H(target_points), dtype=float)
    generators = augmented[:, :-1]
    offset = augmented[:, -1]
    if active_coil_indices is not None:
        generators = generators[:, list(active_coil_indices)]
    return generators, offset


def current_center_and_radii(n_coils: int, current_limit_a: float = 15.0):
    center = np.zeros(n_coils)
    radii = np.full(n_coils, float(current_limit_a))
    return center, radii


def support_function(normal: np.ndarray, generators: np.ndarray, offset: np.ndarray, radii: np.ndarray) -> float:
    normal = np.asarray(normal, dtype=float).reshape(-1)
    return float(normal @ offset + np.sum(radii * np.abs(generators.T @ normal)))


def maximize_support_lp(
    normal: np.ndarray,
    generators: np.ndarray,
    offset: np.ndarray,
    current_limit_a: float,
):
    normal = np.asarray(normal, dtype=float).reshape(-1)
    n = generators.shape[1]
    result = linprog(
        -generators.T @ normal,
        bounds=[(-current_limit_a, current_limit_a)] * n,
        method="highs",
    )
    if not result.success:
        return {
            "success": False,
            "status": str(result.message),
            "support": None,
        }
    current = result.x
    value = float(normal @ (generators @ current + offset))
    return {"success": True, "status": "optimal", "support": value, "current": current}


def current_feasibility(
    output: np.ndarray,
    generators: np.ndarray,
    offset: np.ndarray,
    current_limit_a: float,
    residual_tol: float,
):
    """Independent linear feasibility of ``G i + c = y`` with box currents."""

    target = np.asarray(output, dtype=float).reshape(-1) - offset
    n = generators.shape[1]
    lp = linprog(
        np.zeros(n), A_eq=generators, b_eq=target,
        bounds=[(-current_limit_a, current_limit_a)] * n,
        method="highs",
        options={"primal_feasibility_tolerance": 1e-9,
                 "dual_feasibility_tolerance": 1e-9},
    )
    residual = None if lp.x is None else float(np.max(np.abs(generators @ lp.x - target)))
    current_max = None if lp.x is None else float(np.max(np.abs(lp.x)))
    feasible = None
    if lp.success and residual <= residual_tol and current_max <= current_limit_a + 1e-9:
        feasible = True
    elif lp.status == 2:
        feasible = False  # Explicit solver infeasibility, never failed convergence.
    return {
        "feasible": feasible, "method": "linprog", "status": int(lp.status),
        "message": str(lp.message), "residual_inf": residual,
        "current": lp.x, "max_abs_current": current_max,
    }


def zonotope_hrep(generators: np.ndarray, offset: np.ndarray, radii: np.ndarray):
    """Exercise the production modified hyperplane method.

    Full-rank cases call the existing implementation directly. Degenerate
    cases are represented in their affine hull, with explicit hull equalities.
    All arithmetic here is nominal floating point.
    """
    generators = np.asarray(generators, dtype=float)
    offset = np.asarray(offset, dtype=float).reshape(-1)
    radii = np.asarray(radii, dtype=float)
    if np.any(radii < 0):
        raise ValueError("current radii must be nonnegative")
    d = generators.shape[0]
    weighted = generators * radii
    u, singular, _ = np.linalg.svd(weighted, full_matrices=True)
    tolerance = (singular[0] if singular.size else 0) * max(weighted.shape) * np.finfo(float).eps
    rank = int(np.sum(singular > tolerance))
    if rank == 0:
        normals = np.vstack((np.eye(d), -np.eye(d)))
        return normals, normals @ offset
    if rank == d and np.all(radii == radii[0]):
        normals, supports = ModifiedHyperplaneShiftingMethod(
            np.column_stack((generators, offset)), -float(radii[0]), float(radii[0])
        )
        return normals, np.asarray(supports).reshape(-1)
    basis = u[:, :rank]
    intrinsic = basis.T @ weighted
    normals_r, supports_r = ModifiedHyperplaneShiftingMethod(
        np.column_stack((intrinsic, np.zeros(rank))), -1.0, 1.0
    )
    normals = normals_r @ basis.T
    supports = np.asarray(supports_r).reshape(-1) + normals @ offset
    hull = u[:, rank:].T
    if len(hull):
        normals = np.vstack((normals, hull, -hull))
        supports = np.concatenate((supports, hull @ offset, -hull @ offset))
    return normals, supports


def hyperplane_membership(
    output: np.ndarray,
    generators: np.ndarray,
    offset: np.ndarray,
    current_limit_a: float,
    atol: float,
    hrep=None,
):
    """AFW membership from the production zonotope H-representation."""

    output = np.asarray(output, dtype=float).reshape(-1)
    if hrep is None:
        radii = np.full(generators.shape[1], float(current_limit_a))
        hrep = zonotope_hrep(generators, offset, radii)
    normals, supports = hrep
    if len(supports) == 0:
        return {"feasible": False, "min_slack": None, "n_planes": 0}
    slacks = supports - normals @ output
    return {
        "feasible": bool(np.all(slacks >= -atol)),
        "min_slack": float(np.min(slacks)),
        "n_planes": int(len(slacks)),
    }


def zonotope_vertices_2d(generators: np.ndarray, offset: np.ndarray, radii: np.ndarray) -> np.ndarray:
    if generators.shape[0] != 2:
        raise ValueError("zonotope_vertices_2d requires a 2-output map")
    from itertools import product

    from scipy.spatial import ConvexHull

    columns = generators * np.asarray(radii, dtype=float)
    signs = np.array(list(product((-1.0, 1.0), repeat=columns.shape[1])))
    points = (columns @ signs.T) + np.asarray(offset, dtype=float).reshape(2, 1)
    rank = np.linalg.matrix_rank(columns)
    if rank == 0:
        return np.asarray(offset, dtype=float).reshape(1, 2)
    if rank == 1:
        direction = np.linalg.svd(columns, full_matrices=False)[0][:, 0]
        projections = direction @ points
        return points[:, [np.argmin(projections), np.argmax(projections), np.argmin(projections)]].T
    hull = ConvexHull(points.T)
    vertices = points.T[hull.vertices]
    return np.vstack((vertices, vertices[:1]))


def interaction_offset_only(target_points: list[dict], active_coil_indices=None):
    generators, offset = actuation_map(target_points, active_coil_indices)
    zeroed = [dict(point) for point in target_points]
    for point in zeroed:
        point["m"] = 0.0
    _, zero_offset = actuation_map(zeroed, active_coil_indices)
    return generators, offset, zero_offset


@dataclass(frozen=True)
class TargetSpec:
    name: str
    points_paper: tuple[dict, ...]
    description: str

    def points_coil(self) -> list[dict]:
        converted = []
        for point in self.points_paper:
            item = dict(point)
            item["Z"] = float(point["Z"]) + PAPER_Z_OFFSET_M
            converted.append(item)
        return converted
