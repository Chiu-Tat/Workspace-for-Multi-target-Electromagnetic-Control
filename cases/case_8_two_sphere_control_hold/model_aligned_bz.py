"""Aligned-field model with optimized nonzero Bz for Case 8.

For one chosen pair ``(B1z, B2z)``, both permanent moments align with their
complete prescribed local fields.  The pair is held fixed across all force-box
vertices, so the current-to-output map is affine and vertex convexity remains
valid.
"""

from __future__ import annotations

import math
from itertools import product

import numpy as np
from scipy.optimize import minimize

from cases.case_7_vessel_microspheres.magnetics import coil_basis
from emns.numerics import affine as ca
from emns.numerics.interval import interval
from emns.workspace.affine import params_list

from .model import (
    B_HOLD,
    HELD_M,
    HELD_X,
    HELD_Y,
    I_MAX,
    MOVING_M,
    MU0_OVER_4PI,
    NUM_COILS,
    SeparationError,
    XY_LIMIT,
    Z_PLANE,
    _affine,
    _negative_power,
    _product,
    interval_map_bounds,
    separation_bounds,
    stable_alignment_radius,
)


BZ_LIMIT = 0.010
F_HOLD = 0.001
F_MOVE = 0.005
OUTPUT_LABELS = (
    "B1x", "B1y", "B1z", "F1x", "F1y",
    "B2x", "B2y", "B2z", "F2x", "F2y",
)
CONTROLLED_ROWS = np.asarray([0, 1, 3, 4, 5, 6, 8, 9])


def _as_affine(value):
    if hasattr(value, "center") and hasattr(value, "deviations"):
        return value
    if hasattr(value, "lower") and hasattr(value, "upper"):
        return ca.Affine.from_interval(value)
    return ca.Affine(float(value))


def _dipole_field_and_gradients(target, source, moment):
    """Return B plus x/y derivatives of By and Bz as affine forms."""

    dx = target[0] - source[0]
    dy = target[1] - source[1]
    dz = target[2] - source[2]
    mx, my, mz = (_as_affine(value) for value in moment)
    r2 = _product(dx, dx) + _product(dy, dy) + _product(dz, dz)
    if r2.to_interval().lower <= 0.0:
        raise SeparationError("dipole source and target boxes overlap")
    inv_r3 = _negative_power(r2, -1.5)
    inv_r5 = _negative_power(r2, -2.5)
    inv_r7 = _negative_power(r2, -3.5)
    dot = _product(mx, dx) + _product(my, dy) + _product(mz, dz)

    bx = MU0_OVER_4PI * (3.0 * _product(dx, dot, inv_r5) - _product(mx, inv_r3))
    by = MU0_OVER_4PI * (3.0 * _product(dy, dot, inv_r5) - _product(my, inv_r3))
    bz = MU0_OVER_4PI * (3.0 * _product(dz, dot, inv_r5) - _product(mz, inv_r3))
    dby_dx = MU0_OVER_4PI * (
        3.0 * _product(dy, mx, inv_r5)
        + 3.0 * _product(my, dx, inv_r5)
        - 15.0 * _product(dy, dx, dot, inv_r7)
    )
    dby_dy = MU0_OVER_4PI * (
        3.0 * _product(dot, inv_r5)
        + 6.0 * _product(dy, my, inv_r5)
        - 15.0 * _product(dy, dy, dot, inv_r7)
    )
    dbz_dx = MU0_OVER_4PI * (
        3.0 * _product(dz, mx, inv_r5)
        + 3.0 * _product(mz, dx, inv_r5)
        - 15.0 * _product(dz, dx, dot, inv_r7)
    )
    dbz_dy = MU0_OVER_4PI * (
        3.0 * _product(dz, my, inv_r5)
        + 3.0 * _product(mz, dy, inv_r5)
        - 15.0 * _product(dz, dy, dot, inv_r7)
    )
    return bx, by, bz, dby_dx, dby_dy, dbz_dx, dbz_dy


def _moment_cone(moment, bz_limit):
    angle = math.atan2(bz_limit, B_HOLD)
    return (
        _as_affine(0.0),
        ca.Affine.from_interval(interval[moment * math.cos(angle), moment]),
        ca.Affine.from_interval(
            interval[-moment * math.sin(angle), moment * math.sin(angle)]
        ),
    )


def aligned_moment(moment, bz, bias_field=B_HOLD):
    if not all(math.isfinite(value) for value in (moment, bz, bias_field)):
        raise ValueError("moment and field values must be finite")
    if moment <= 0.0 or bias_field <= 0.0:
        raise ValueError("moment and bias field must be positive")
    norm = math.hypot(bias_field, bz)
    return np.asarray([0.0, moment * bias_field / norm, moment * bz / norm])


def _numeric_dipole_field_gradient(target, source, moment):
    r = np.asarray(target, dtype=float) - np.asarray(source, dtype=float)
    moment = np.asarray(moment, dtype=float)
    norm = np.linalg.norm(r)
    if norm <= 0.0:
        raise ValueError("dipole source and target overlap")
    dot = float(moment @ r)
    field = 1e-7 * (3.0 * r * dot / norm**5 - moment / norm**3)
    gradient = np.empty((3, 3))
    for a in range(3):
        for b in range(3):
            gradient[a, b] = 1e-7 * (
                3.0 * ((a == b) * dot + r[a] * moment[b] + moment[a] * r[b]) / norm**5
                - 15.0 * r[a] * r[b] * dot / norm**7
            )
    return field, gradient


def point_actuation_map(x2, y2, bz1, bz2, *, held_x=HELD_X, held_y=HELD_Y):
    """Return numerical ``G,c`` for fixed aligned field directions."""

    p1 = np.asarray([held_x, held_y, Z_PLANE])
    p2 = np.asarray([float(x2), float(y2), Z_PLANE])
    moment1 = aligned_moment(HELD_M, bz1)
    moment2 = aligned_moment(MOVING_M, bz2)
    H1, D1 = coil_basis(p1)
    H2, D2 = coil_basis(p2)
    rows1 = np.vstack((
        H1,
        moment1 @ D1[:, 0, :],
        moment1 @ D1[:, 1, :],
    ))
    rows2 = np.vstack((
        H2,
        moment2 @ D2[:, 0, :],
        moment2 @ D2[:, 1, :],
    ))
    field_2_on_1, _ = _numeric_dipole_field_gradient(p1, p2, moment2)
    field_1_on_2, gradient_1_on_2 = _numeric_dipole_field_gradient(p2, p1, moment1)
    force2 = gradient_1_on_2.T @ moment2
    force1 = -force2
    offset = np.concatenate((field_2_on_1, force1[:2], field_1_on_2, force2[:2]))
    return np.vstack((rows1, rows2)), offset


def _coil_rows_at(position, target_moment):
    rows = [[] for _ in range(5)]
    for coil in params_list:
        source = tuple(_affine(value) for value in coil[3:])
        effect = _dipole_field_and_gradients(position, source, coil[:3])
        fx = sum(target_moment[index] * effect[3 + 2 * (index == 2)]
                 for index in (1, 2))
        fy = sum(target_moment[index] * effect[4 + 2 * (index == 2)]
                 for index in (1, 2))
        for row, value in zip(rows, effect[:3] + (fx, fy)):
            row.append(value)
    return rows


def affine_actuation_map(
    x2, y2, bz1, bz2, *, held_x=HELD_X, held_y=HELD_Y,
    return_affine=False,
):
    moment1 = aligned_moment(HELD_M, bz1)
    moment2 = aligned_moment(MOVING_M, bz2)
    p1 = (_affine(held_x), _affine(held_y), _affine(Z_PLANE))
    p2 = (_affine(x2), _affine(y2), _affine(Z_PLANE))
    rows = _coil_rows_at(p1, moment1) + _coil_rows_at(p2, moment2)
    field_2_on_1 = _dipole_field_and_gradients(p1, p2, moment2)
    field_1_on_2 = _dipole_field_and_gradients(p2, p1, moment1)
    force2 = (
        moment2[1] * field_1_on_2[3] + moment2[2] * field_1_on_2[5],
        moment2[1] * field_1_on_2[4] + moment2[2] * field_1_on_2[6],
    )
    force1 = (-force2[0], -force2[1])
    offset = np.asarray(
        field_2_on_1[:3] + force1 + field_1_on_2[:3] + force2,
        dtype=object,
    )
    full = np.column_stack((np.asarray(rows, dtype=object), offset))
    if return_affine:
        return full
    return np.asarray(
        [[entry.to_interval() for entry in row] for row in full], dtype=object
    )


def cone_affine_actuation_map(
    x2, y2, bz_limit, *, held_x=HELD_X, held_y=HELD_Y,
    return_affine=False,
):
    """Over-enclose all aligned directions allowed by the Bz bound.

    This map is used only by the one-sided outside certificate. Its independent
    moment-component intervals enlarge the attainable set, so separation from
    that set remains a valid proof of infeasibility for every aligned direction.
    """

    p1 = (_as_affine(held_x), _as_affine(held_y), _as_affine(Z_PLANE))
    p2 = (_as_affine(x2), _as_affine(y2), _as_affine(Z_PLANE))
    moment1 = _moment_cone(HELD_M, bz_limit)
    moment2 = _moment_cone(MOVING_M, bz_limit)
    rows = _coil_rows_at(p1, moment1) + _coil_rows_at(p2, moment2)
    field_2_on_1 = _dipole_field_and_gradients(p1, p2, moment2)
    field_1_on_2 = _dipole_field_and_gradients(p2, p1, moment1)
    force2 = (
        _product(moment2[1], field_1_on_2[3])
        + _product(moment2[2], field_1_on_2[5]),
        _product(moment2[1], field_1_on_2[4])
        + _product(moment2[2], field_1_on_2[6]),
    )
    force1 = (-force2[0], -force2[1])
    offset = np.asarray(
        field_2_on_1[:3] + force1 + field_1_on_2[:3] + force2,
        dtype=object,
    )
    full = np.column_stack((np.asarray(rows, dtype=object), offset))
    if return_affine:
        return full
    return np.asarray(
        [[entry.to_interval() for entry in row] for row in full], dtype=object
    )


def task_vertices(bz1, bz2, f_hold=F_HOLD, f_move=F_MOVE):
    if not all(math.isfinite(value) and value >= 0.0 for value in (f_hold, f_move)):
        raise ValueError("force bounds must be finite and nonnegative")
    vertices = []
    ranges = [(-f, f) if f else (0.0,) for f in (f_hold, f_hold, f_move, f_move)]
    for f1x, f1y, f2x, f2y in product(*ranges):
        vertices.append([
            0.0, B_HOLD, bz1, f1x, f1y,
            0.0, B_HOLD, bz2, f2x, f2y,
        ])
    return np.asarray(vertices, dtype=float)


def planar_task_vertices(f_hold=F_HOLD, f_move=F_MOVE):
    """Eight controlled planar outputs used by the relaxed outside proof."""

    if not all(math.isfinite(value) and value >= 0.0 for value in (f_hold, f_move)):
        raise ValueError("force bounds must be finite and nonnegative")
    vertices = []
    ranges = [(-f, f) if f else (0.0,) for f in (f_hold, f_hold, f_move, f_move)]
    for f1x, f1y, f2x, f2y in product(*ranges):
        vertices.append([
            0.0, B_HOLD, f1x, f1y,
            0.0, B_HOLD, f2x, f2y,
        ])
    return np.asarray(vertices, dtype=float)


def peak_current(
    x2, y2, bz1, bz2, f_hold=F_HOLD, f_move=F_MOVE, *,
    held_x=HELD_X, held_y=HELD_Y,
):
    try:
        G, c = point_actuation_map(
            x2, y2, bz1, bz2, held_x=held_x, held_y=held_y
        )
        currents = np.linalg.solve(G, (task_vertices(bz1, bz2, f_hold, f_move) - c).T)
    except (ValueError, np.linalg.LinAlgError, FloatingPointError):
        return float("inf")
    return float(np.max(np.abs(currents)))


def optimize_bz_pair(
    x2,
    y2,
    *,
    bz_limit=BZ_LIMIT,
    f_hold=F_HOLD,
    f_move=F_MOVE,
    grid_size=5,
    held_x=HELD_X,
    held_y=HELD_Y,
):
    """Continuously minimize worst current over one common aligned Bz pair."""

    if not math.isfinite(bz_limit) or bz_limit < 0.0:
        raise ValueError("Bz limit must be finite and nonnegative")
    if grid_size < 2:
        raise ValueError("grid_size must be at least two")
    grid = np.linspace(-bz_limit, bz_limit, grid_size)
    best_pair = np.zeros(2)
    best_value = float("inf")
    for bz1 in grid:
        for bz2 in grid:
            value = peak_current(
                x2, y2, bz1, bz2, f_hold, f_move,
                held_x=held_x, held_y=held_y,
            )
            if value < best_value:
                best_value = value
                best_pair[:] = (bz1, bz2)
    if bz_limit == 0.0:
        return best_pair, best_value

    result = minimize(
        lambda values: peak_current(
            x2, y2, values[0], values[1], f_hold, f_move,
            held_x=held_x, held_y=held_y,
        ),
        best_pair,
        method="Powell",
        bounds=[(-bz_limit, bz_limit), (-bz_limit, bz_limit)],
        options={"xtol": 1e-6, "ftol": 1e-5, "maxiter": 60},
    )
    if result.success and result.fun < best_value:
        best_pair = np.asarray(result.x, dtype=float)
        best_value = float(result.fun)
    return best_pair, best_value
