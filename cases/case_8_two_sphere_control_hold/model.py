"""Affine global-frame model for the Case 8 two-sphere TFW.

The model is deliberately reduced: both permanent-magnet spheres are assumed
to remain on the stable quasistatic branch aligned with the prescribed net
field ``(0, B_HOLD, 0)``.  Magnet 1 is control-held at ``(0.02, 0)`` while
magnet 2 is the positional workspace variable.  Mutual fields and forces are
included as the pinned affine offset.
"""

from __future__ import annotations

import math
from itertools import product

import numpy as np

from emns.numerics import affine as ca
from emns.numerics.interval import interval
from emns.workspace.affine import params_list
from hybrid_region.certificates import interval_object_bounds


MU0_OVER_4PI = 1e-7
NUM_COILS = 10
Z_PLANE = 0.082
XY_LIMIT = 0.05

HELD_X, HELD_Y = 0.02, 0.0
SPHERE_M = 0.12
HELD_M = SPHERE_M
MOVING_M = SPHERE_M
B_HOLD = 0.02
F_HOLD = 0.02
F_MOVE = 0.10
I_MIN, I_MAX = -15.0, 15.0

OUTPUT_LABELS = (
    "B1x", "B1y", "B1z", "F1x", "F1y",
    "B2x", "B2y", "B2z", "F2x", "F2y",
)


class SeparationError(ValueError):
    """Raised before evaluating a dipole expression across zero separation."""


def _affine(value):
    if hasattr(value, "lower") and hasattr(value, "upper"):
        return ca.Affine.from_interval(value)
    return ca.Affine(float(value))


def _product(*forms):
    """Multiply affine forms without dropping small nonlinear remainders.

    Dipole denominators involve very small powers of metre-valued distances.
    The shared backend's absolute error cutoffs are unsuitable at that scale.
    Keep this correction local so other case studies retain their behavior.
    """

    result = ca.Affine(1.0)
    for other in forms:
        deviations = {}
        for symbol, value in result.deviations.items():
            deviations[symbol] = value * other.center
        for symbol, value in other.deviations.items():
            deviations[symbol] = deviations.get(symbol, 0.0) + value * result.center
        remainder = (
            sum(abs(v) for v in result.deviations.values())
            * sum(abs(v) for v in other.deviations.values())
        )
        if remainder:
            deviations[ca.get_next_symbol_index()] = remainder
        result = ca.Affine(result.center * other.center, deviations)
    return result


def _negative_power(value, exponent):
    """Enclose a negative power on a positive domain using a Taylor remainder.

    For f(x)=x**p, p<0, f'' is positive and largest at the lower endpoint.
    The remainder lies in [0, 0.5*max(f'')*radius**2]. This preserves shared
    position symbols, unlike replacing the reciprocal by an independent box.
    Bounds are in real arithmetic; the backend is still not outward-rounded.
    """

    radius = sum(abs(v) for v in value.deviations.values())
    lower = value.center - radius
    if lower <= 0.0:
        raise SeparationError("distance enclosure reaches zero")
    center = value.center ** exponent
    slope = exponent * value.center ** (exponent - 1.0)
    deviations = {k: slope * v for k, v in value.deviations.items()}
    if radius:
        remainder = (
            0.5 * exponent * (exponent - 1.0)
            * lower ** (exponent - 2.0) * radius ** 2
        )
        center += 0.5 * remainder
        deviations[ca.get_next_symbol_index()] = 0.5 * remainder
    return ca.Affine(center, deviations)


def _dipole_field_and_y_gradient(target, source, moment):
    """Return ``(Bx, By, Bz, dBy/dx, dBy/dy)`` as affine forms.

    ``moment`` is a three-vector in A m^2 (or A m^2 per ampere for a coil).
    The analytic derivative uses the correct ``6 * y * m_y`` term in
    ``dBy/dy``.
    """

    dx = target[0] - source[0]
    dy = target[1] - source[1]
    dz = target[2] - source[2]
    mx, my, mz = (_affine(value) for value in moment)
    r2 = _product(dx, dx) + _product(dy, dy) + _product(dz, dz)
    r2_interval = r2.to_interval()
    if r2_interval.lower <= 0.0:
        raise SeparationError("dipole source and target boxes overlap")
    inv_r3 = _negative_power(r2, -1.5)
    inv_r5 = _negative_power(r2, -2.5)
    inv_r7 = _negative_power(r2, -3.5)
    dot = _product(mx, dx) + _product(my, dy) + _product(mz, dz)

    bx = MU0_OVER_4PI * (3.0 * _product(dx, dot, inv_r5) - _product(mx, inv_r3))
    by = MU0_OVER_4PI * (3.0 * _product(dy, dot, inv_r5) - _product(my, inv_r3))
    bz = MU0_OVER_4PI * (3.0 * _product(dz, dot, inv_r5) - _product(mz, inv_r3))
    dby_dx = MU0_OVER_4PI * (
        3.0 * _product(dy, mx, inv_r5) + 3.0 * _product(my, dx, inv_r5)
        - 15.0 * _product(dy, dx, dot, inv_r7)
    )
    dby_dy = MU0_OVER_4PI * (
        3.0 * _product(dot, inv_r5) + 6.0 * _product(dy, my, inv_r5)
        - 15.0 * _product(dy, dy, dot, inv_r7)
    )
    return bx, by, bz, dby_dx, dby_dy


def _coil_rows_at(position, target_moment):
    rows = [[] for _ in range(5)]
    for coil in params_list:
        source = tuple(_affine(value) for value in coil[3:])
        effect = _dipole_field_and_y_gradient(position, source, coil[:3])
        values = effect[:3] + (
            effect[3] * target_moment,
            effect[4] * target_moment,
        )
        for row, value in zip(rows, values):
            row.append(value)
    return rows


def affine_actuation_map(x2, y2, *, return_affine=False):
    """Return the global-frame ``[G c]`` map for a magnet-2 position box.

    Rows follow :data:`OUTPUT_LABELS`.  The first ten columns multiply coil
    currents; the final column is the mutual-interaction offset and must be
    pinned to one by callers using the legacy interval checks.
    """

    p1 = (_affine(HELD_X), _affine(HELD_Y), _affine(Z_PLANE))
    p2 = (_affine(x2), _affine(y2), _affine(Z_PLANE))

    G_rows = _coil_rows_at(p1, HELD_M) + _coil_rows_at(p2, MOVING_M)

    moment1 = (0.0, HELD_M, 0.0)
    moment2 = (0.0, MOVING_M, 0.0)
    field_2_on_1 = _dipole_field_and_y_gradient(p1, p2, moment2)
    field_1_on_2 = _dipole_field_and_y_gradient(p2, p1, moment1)

    # Evaluate one mutual force and impose Newton's third law exactly.  This
    # also preserves the shared affine dependency between the two force rows.
    force_2 = (MOVING_M * field_1_on_2[3], MOVING_M * field_1_on_2[4])
    force_1 = (-force_2[0], -force_2[1])
    c = np.asarray(
        field_2_on_1[:3] + force_1 + field_1_on_2[:3] + force_2,
        dtype=object,
    )
    G = np.asarray(G_rows, dtype=object)
    full = np.column_stack((G, c))
    if return_affine:
        return full
    return np.asarray(
        [[entry.to_interval() for entry in row] for row in full],
        dtype=object,
    )


def point_actuation_map(x2, y2):
    """Return numerical ``G, c`` at one position for diagnostics/tests."""

    affine_map = affine_actuation_map(
        interval[float(x2), float(x2)],
        interval[float(y2), float(y2)],
        return_affine=True,
    )
    G = np.asarray(
        [[float(value.center) for value in row[:NUM_COILS]] for row in affine_map]
    )
    c = np.asarray([float(row[NUM_COILS].center) for row in affine_map])
    return G, c


def task_bounds(f_hold=F_HOLD, f_move=F_MOVE):
    """Lower/upper bounds for the ten-output task box."""

    if not all(math.isfinite(v) and v >= 0.0 for v in (f_hold, f_move)):
        raise ValueError("force bounds must be finite and nonnegative")
    lower = np.array([
        0.0, B_HOLD, 0.0, -f_hold, -f_hold,
        0.0, B_HOLD, 0.0, -f_move, -f_move,
    ])
    upper = np.array([
        0.0, B_HOLD, 0.0, f_hold, f_hold,
        0.0, B_HOLD, 0.0, f_move, f_move,
    ])
    return lower, upper


def task_vertices(f_hold=F_HOLD, f_move=F_MOVE):
    """The distinct force vertices (16 unless a force range is zero)."""

    task_bounds(f_hold, f_move)  # Validate before building vertices.
    vertices = []
    for f1x, f1y, f2x, f2y in product(
        *[(-f, f) if f else (0.0,) for f in (f_hold, f_hold, f_move, f_move)]
    ):
        vertices.append([
            0.0, B_HOLD, 0.0, f1x, f1y,
            0.0, B_HOLD, 0.0, f2x, f2y,
        ])
    return np.asarray(vertices, dtype=float)


def interval_map_bounds(full_interval_map):
    """Split an interval ``[G c]`` map into numerical inclusion bounds."""

    G_lower, G_upper = interval_object_bounds(
        np.asarray(full_interval_map, dtype=object)[:, :NUM_COILS]
    )
    c_lower, c_upper = interval_object_bounds(
        np.asarray(full_interval_map, dtype=object)[:, NUM_COILS]
    )
    return G_lower, G_upper, c_lower, c_upper


def stable_alignment_radius(bias_field=B_HOLD):
    """Worst-direction in-plane small-angle stability separation.

    The sufficient condition is
    ``bias_field > 2*mu0/(4*pi)*sqrt(m1*m2)/r**3``.
    """

    return (
        2.0 * MU0_OVER_4PI * math.sqrt(HELD_M * MOVING_M) / bias_field
    ) ** (1.0 / 3.0)


def separation_bounds(x_interval, y_interval, held_x=HELD_X, held_y=HELD_Y):
    """Exact minimum/maximum planar distance from a rectangle to the held sphere."""

    xmin, xmax = float(x_interval.lower), float(x_interval.upper)
    ymin, ymax = float(y_interval.lower), float(y_interval.upper)
    dx_min = 0.0 if xmin <= held_x <= xmax else min(abs(xmin - held_x), abs(xmax - held_x))
    dy_min = 0.0 if ymin <= held_y <= ymax else min(abs(ymin - held_y), abs(ymax - held_y))
    minimum = math.hypot(dx_min, dy_min)
    maximum = max(
        math.hypot(x - held_x, y - held_y)
        for x in (xmin, xmax) for y in (ymin, ymax)
    )
    return minimum, maximum
