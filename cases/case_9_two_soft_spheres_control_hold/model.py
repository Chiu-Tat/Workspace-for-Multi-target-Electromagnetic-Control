"""Fixed-field affine model for two isotropic soft-magnetic spheres.

The local total field at both sphere centres is prescribed exactly.  This
fixes each induced moment at its material-law operating point, so the field
and force requirements remain affine in the coil currents.  Varying the field
magnitude as an independent command would make the force-current relation
quadratic and is deliberately outside this case study.
"""

from __future__ import annotations

import math
from itertools import product

import numpy as np

from cases.case_8_two_sphere_control_hold.model import (
    MU0_OVER_4PI,
    NUM_COILS,
    Z_PLANE,
    XY_LIMIT,
    _affine,
    _coil_rows_at,
    _dipole_field_and_y_gradient,
    interval_map_bounds,
    separation_bounds,
)


MU0 = 4.0 * math.pi * 1e-7
HELD_X, HELD_Y = 0.02, 0.0
B_HOLD = 0.02
F_HOLD = 0.001
F_MOVE = 0.005
I_MIN, I_MAX = -15.0, 15.0

# Illustrative bulk soft-iron operating parameters, not a fitted material
# model.  Users should replace these values with a measured M(B) curve when
# comparing against hardware.
HELD_RADIUS = 0.005
MOVING_RADIUS = 0.005
SUSCEPTIBILITY = 5000.0
SATURATION_MAGNETIZATION = 1.7e6

OUTPUT_LABELS = (
    "B1x", "B1y", "B1z", "F1x", "F1y",
    "B2x", "B2y", "B2z", "F2x", "F2y",
)


def effective_sphere_susceptibility(susceptibility: float) -> float:
    """Return the demagnetization-corrected susceptibility of a sphere."""

    if not math.isfinite(susceptibility) or susceptibility < 0.0:
        raise ValueError("susceptibility must be finite and nonnegative")
    return 3.0 * susceptibility / (susceptibility + 3.0)


def soft_sphere_moment(
    radius: float,
    *,
    bias_field: float = B_HOLD,
    susceptibility: float = SUSCEPTIBILITY,
    saturation_magnetization: float = SATURATION_MAGNETIZATION,
) -> float:
    """Moment magnitude at a prescribed local field for an isotropic sphere.

    The constitutive approximation is linear with the spherical
    demagnetization correction, clipped at ``M_s``::

        M = min(3 chi/(chi+3) * |B|/mu0, M_s),  m = V M.

    It is reversible and rate-independent; remanence and hysteresis are not
    represented.
    """

    values = (radius, bias_field, saturation_magnetization)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("sphere and material parameters must be finite")
    if radius <= 0.0:
        raise ValueError("sphere radius must be positive")
    if saturation_magnetization <= 0.0:
        raise ValueError("saturation magnetization must be positive")
    chi_eff = effective_sphere_susceptibility(susceptibility)
    magnetization = min(
        chi_eff * abs(bias_field) / MU0,
        saturation_magnetization,
    )
    volume = 4.0 * math.pi * radius ** 3 / 3.0
    return volume * magnetization


def operating_moments(
    held_radius: float = HELD_RADIUS,
    moving_radius: float = MOVING_RADIUS,
    *,
    bias_field: float = B_HOLD,
    susceptibility: float = SUSCEPTIBILITY,
    saturation_magnetization: float = SATURATION_MAGNETIZATION,
):
    kwargs = {
        "bias_field": bias_field,
        "susceptibility": susceptibility,
        "saturation_magnetization": saturation_magnetization,
    }
    return (
        soft_sphere_moment(held_radius, **kwargs),
        soft_sphere_moment(moving_radius, **kwargs),
    )


def affine_actuation_map(
    x2,
    y2,
    *,
    held_radius: float = HELD_RADIUS,
    moving_radius: float = MOVING_RADIUS,
    bias_field: float = B_HOLD,
    susceptibility: float = SUSCEPTIBILITY,
    saturation_magnetization: float = SATURATION_MAGNETIZATION,
    return_affine: bool = False,
):
    """Return the fixed-local-field ``[G c]`` map for a position box.

    The final column contains mutual field and force offsets.  It must be
    pinned to one by callers using the legacy interval checks.
    """

    held_m, moving_m = operating_moments(
        held_radius,
        moving_radius,
        bias_field=bias_field,
        susceptibility=susceptibility,
        saturation_magnetization=saturation_magnetization,
    )
    p1 = (_affine(HELD_X), _affine(HELD_Y), _affine(Z_PLANE))
    p2 = (_affine(x2), _affine(y2), _affine(Z_PLANE))
    G_rows = _coil_rows_at(p1, held_m) + _coil_rows_at(p2, moving_m)

    sign = 1.0 if bias_field >= 0.0 else -1.0
    moment1 = (0.0, sign * held_m, 0.0)
    moment2 = (0.0, sign * moving_m, 0.0)
    field_2_on_1 = _dipole_field_and_y_gradient(p1, p2, moment2)
    field_1_on_2 = _dipole_field_and_y_gradient(p2, p1, moment1)

    force_2 = (moving_m * field_1_on_2[3], moving_m * field_1_on_2[4])
    force_1 = (-force_2[0], -force_2[1])
    c = np.asarray(
        field_2_on_1[:3] + force_1 + field_1_on_2[:3] + force_2,
        dtype=object,
    )
    full = np.column_stack((np.asarray(G_rows, dtype=object), c))
    if return_affine:
        return full
    return np.asarray(
        [[entry.to_interval() for entry in row] for row in full],
        dtype=object,
    )


def point_actuation_map(x2, y2, **material):
    from emns.numerics.interval import interval

    affine_map = affine_actuation_map(
        interval[float(x2), float(x2)],
        interval[float(y2), float(y2)],
        return_affine=True,
        **material,
    )
    G = np.asarray([
        [float(value.center) for value in row[:NUM_COILS]]
        for row in affine_map
    ])
    c = np.asarray([float(row[NUM_COILS].center) for row in affine_map])
    return G, c


def task_bounds(f_hold=F_HOLD, f_move=F_MOVE, bias_field=B_HOLD):
    if not all(math.isfinite(value) and value >= 0.0 for value in (f_hold, f_move)):
        raise ValueError("force bounds must be finite and nonnegative")
    if not math.isfinite(bias_field) or bias_field <= 0.0:
        raise ValueError("bias field must be finite and positive")
    lower = np.array([
        0.0, bias_field, 0.0, -f_hold, -f_hold,
        0.0, bias_field, 0.0, -f_move, -f_move,
    ])
    upper = np.array([
        0.0, bias_field, 0.0, f_hold, f_hold,
        0.0, bias_field, 0.0, f_move, f_move,
    ])
    return lower, upper


def task_vertices(f_hold=F_HOLD, f_move=F_MOVE, bias_field=B_HOLD):
    task_bounds(f_hold, f_move, bias_field)
    vertices = []
    ranges = [(-f, f) if f else (0.0,) for f in (f_hold, f_hold, f_move, f_move)]
    for f1x, f1y, f2x, f2y in product(*ranges):
        vertices.append([
            0.0, bias_field, 0.0, f1x, f1y,
            0.0, bias_field, 0.0, f2x, f2y,
        ])
    return np.asarray(vertices, dtype=float)
