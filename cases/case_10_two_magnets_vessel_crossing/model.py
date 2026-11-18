"""Affine actuation model for two route-confined permanent magnets.

Each magnet is reduced to one route coordinate ``s`` on its own centerline.
Both moments are assumed to sit on the stable quasistatic branch aligned with
the complete prescribed net field ``B_HOLD * u_i``; the direction ``u_i`` is a
free control decision, fixed per configuration box so that the current-to-
output map stays affine.

Controlled rows (eight, against ten coil currents)::

    [B1x, B1y, B1z, F1.t1, B2x, B2y, B2z, F2.t2]

The two remaining degrees of freedom are spent on four transverse side rows,
``F_i.n_i`` and ``F_i.z``, which are inequality constraints rather than
commanded outputs: a tube wall reacts them, and their magnitude sets the
Coulomb friction that opposes the commanded tangential drive.
"""

from __future__ import annotations

import math
from itertools import product

import numpy as np
from scipy.optimize import linprog, minimize

from cases.case_7_vessel_microspheres.magnetics import coil_basis
from emns.numerics import affine as ca
from emns.numerics.interval import interval
from emns.workspace.affine import params_list

from .route import Route, load, magnet_properties

MU0_OVER_4PI = 1e-7
NUM_COILS = len(params_list)
I_MIN, I_MAX = -15.0, 15.0

B_HOLD = 0.020
F_DRIVE = 0.5e-3
F_SIDE = 2.0e-3
FRICTION_COEFFICIENT = 0.15

OUTPUT_LABELS = ("B1x", "B1y", "B1z", "F1t", "B2x", "B2y", "B2z", "F2t")
SIDE_LABELS = ("F1n", "F1z", "F2n", "F2z")

_VESSEL, CONFIGURATION = load()
BRANCHES = tuple(entry["branch"] for entry in CONFIGURATION["magnets"])
ROUTES = tuple(Route(_VESSEL, name) for name in BRANCHES)
_PROPERTIES = tuple(magnet_properties(entry) for entry in CONFIGURATION["magnets"])
MAGNET_RADIUS = tuple(value[0] for value in _PROPERTIES)
MOMENT = tuple(value[1] for value in _PROPERTIES)
NET_WEIGHT = tuple(value[2] for value in _PROPERTIES)
LUMEN_SLACK = tuple(
    route.radius_m - radius for route, radius in zip(ROUTES, MAGNET_RADIUS)
)


class SeparationError(ValueError):
    """Raised before evaluating a dipole expression across zero separation."""


# ---------------------------------------------------------------------------
# Affine enclosures
# ---------------------------------------------------------------------------

def _affine(value):
    if hasattr(value, "center") and hasattr(value, "deviations"):
        return value
    if hasattr(value, "lower") and hasattr(value, "upper"):
        return ca.Affine.from_interval(value)
    return ca.Affine(float(value))


def _product(*forms):
    """Multiply affine forms while keeping the nonlinear remainder.

    Mirrors the local product of Case 8: the shared backend drops remainders
    below an absolute cutoff, which is unsafe for the metre-scale powers that
    appear in the dipole denominators.
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
    """Enclose ``x**exponent`` for ``exponent < 0`` on a strictly positive box.

    ``f''`` is positive and largest at the lower endpoint, so the Taylor
    remainder lies in ``[0, 0.5*max(f'')*radius**2]``.  Keeping the linear part
    preserves the shared position symbols instead of collapsing the power to an
    independent interval.
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


def _dipole(target, source, moment):
    """Return ``(B, D)`` as affine forms, with ``D[a][b] = dB_a/dx_b``."""

    r = tuple(_affine(target[axis]) - _affine(source[axis]) for axis in range(3))
    m = tuple(_affine(value) for value in moment)
    r2 = _product(r[0], r[0]) + _product(r[1], r[1]) + _product(r[2], r[2])
    if r2.to_interval().lower <= 0.0:
        raise SeparationError("dipole source and target boxes overlap")
    inv_r3 = _negative_power(r2, -1.5)
    inv_r5 = _negative_power(r2, -2.5)
    inv_r7 = _negative_power(r2, -3.5)
    dot = _product(m[0], r[0]) + _product(m[1], r[1]) + _product(m[2], r[2])

    field = [
        MU0_OVER_4PI * (3.0 * _product(r[a], dot, inv_r5) - _product(m[a], inv_r3))
        for a in range(3)
    ]
    gradient = [[None] * 3 for _ in range(3)]
    for a in range(3):
        for b in range(a, 3):
            term = 3.0 * (_product(r[a], m[b], inv_r5) + _product(m[a], r[b], inv_r5))
            if a == b:
                term = term + 3.0 * _product(dot, inv_r5)
            entry = MU0_OVER_4PI * (term - 15.0 * _product(r[a], r[b], dot, inv_r7))
            # A dipole field is curl-free, so its gradient is symmetric.
            gradient[a][b] = entry
            gradient[b][a] = entry
    return field, gradient


def _force_component(gradient, moment, direction):
    """``sum_ab direction_a moment_b dB_b/dx_a`` as one affine form.

    The moment components may be plain numbers or affine forms; the latter is
    how a whole cone of admissible field directions is enclosed at once.
    """

    total = ca.Affine(0.0)
    for a in range(3):
        if not direction[a]:
            continue
        for b in range(3):
            entry = moment[b]
            if isinstance(entry, ca.Affine):
                total = total + float(direction[a]) * _product(entry, gradient[b][a])
                continue
            weight = float(direction[a]) * float(entry)
            if weight:
                total = total + weight * gradient[b][a]
    return total


def _project(vector, direction):
    total = ca.Affine(0.0)
    for axis in range(3):
        if direction[axis]:
            total = total + float(direction[axis]) * vector[axis]
    return total


def consolidate(matrix):
    """Merge per-row private noise symbols without loosening any row.

    The certificates only exploit dependency between the columns of one row:
    the Neumann majorant contracts each row against a right-inverse column, and
    the robust-current program majorizes one auxiliary variable per
    ``(row, symbol)`` pair.  Symbols that reach a single column of a row can
    therefore be summed into one private symbol for that entry, leaving that
    row's attainable range unchanged while removing thousands of auxiliary
    variables.  Symbols spanning two or more columns of the same row are kept,
    so every cancellation the certificates can use survives.  Dependency
    between different rows is dropped, which only enlarges the enclosed set and
    keeps both one-sided certificates valid.
    """

    matrix = np.asarray(matrix, dtype=object)
    reduced = np.empty(matrix.shape, dtype=object)
    for row in range(matrix.shape[0]):
        occurrences = {}
        for column in range(matrix.shape[1]):
            for symbol in matrix[row, column].deviations:
                occurrences[symbol] = occurrences.get(symbol, 0) + 1
        shared = {symbol for symbol, count in occurrences.items() if count > 1}
        for column in range(matrix.shape[1]):
            form = matrix[row, column]
            deviations = {}
            private = 0.0
            for symbol, value in form.deviations.items():
                if symbol in shared:
                    deviations[symbol] = value
                else:
                    private += abs(value)
            if private:
                deviations[ca.get_next_symbol_index()] = private
            reduced[row, column] = ca.Affine(form.center, deviations)
    return reduced


def frames(s1, s2):
    """Task frames ``(t, n, z)`` of both routes at scalar route coordinates."""

    return ROUTES[0].frame(float(s1)), ROUTES[1].frame(float(s2))


def aligned_moment(index, direction):
    """Moment vector of magnet ``index`` aligned with a field direction."""

    direction = np.asarray(direction, dtype=float)
    norm = float(np.linalg.norm(direction))
    if not math.isfinite(norm) or norm <= 0.0:
        raise ValueError("field direction must be a nonzero finite vector")
    return MOMENT[index] * direction / norm


def _bounds(value):
    if hasattr(value, "lower"):
        return float(value.lower), float(value.upper)
    return float(value[0][0]), float(value[0][1])


def _route_positions(s1_interval, s2_interval, slack):
    first, second = _bounds(s1_interval), _bounds(s2_interval)
    return (
        ROUTES[0].affine_point(interval[first[0], first[1]], slack[0]),
        ROUTES[1].affine_point(interval[second[0], second[1]], slack[1]),
    ), first, second


def _assemble(positions, moments, task_frames, field_targets=None):
    """Build the eight controlled and four side rows plus their offsets.

    ``field_targets`` moves a direction-dependent field command into the
    offset column, which is what lets a whole cone of admissible directions
    share one fixed task vertex.
    """

    rows, side_rows = [], []
    for index in (0, 1):
        tangent, normal, depth = task_frames[index]
        field_rows = [[] for _ in range(3)]
        force_rows = [[], [], []]
        for coil in params_list:
            field, gradient = _dipole(
                positions[index], tuple(coil[3:]), tuple(coil[:3])
            )
            for axis in range(3):
                field_rows[axis].append(field[axis])
            for slot, direction in enumerate((tangent, normal, depth)):
                force_rows[slot].append(
                    _force_component(gradient, moments[index], direction)
                )
        rows.extend(field_rows)
        rows.append(force_rows[0])
        side_rows.append(force_rows[1])
        side_rows.append(force_rows[2])

    # Mutual field and force are current independent here, so they enter as the
    # pinned offset column.  Only one of the two forces is evaluated; Newton's
    # third law gives the other exactly and keeps their noise symbols shared.
    field_on_1, _ = _dipole(positions[0], positions[1], moments[1])
    field_on_2, gradient_on_2 = _dipole(positions[1], positions[0], moments[0])
    force_2 = tuple(
        _force_component(gradient_on_2, moments[1], axis)
        for axis in (np.eye(3)[0], np.eye(3)[1], np.eye(3)[2])
    )
    force_1 = tuple(-component for component in force_2)
    tangent_1, normal_1, depth_1 = task_frames[0]
    tangent_2, normal_2, depth_2 = task_frames[1]

    offsets = [
        field_on_1[0], field_on_1[1], field_on_1[2], _project(force_1, tangent_1),
        field_on_2[0], field_on_2[1], field_on_2[2], _project(force_2, tangent_2),
    ]
    if field_targets is not None:
        for index in (0, 1):
            for axis in range(3):
                offsets[4 * index + axis] = (
                    offsets[4 * index + axis] - field_targets[index][axis]
                )
    side_offsets = [
        _project(force_1, normal_1), _project(force_1, depth_1),
        _project(force_2, normal_2), _project(force_2, depth_2),
    ]
    return rows, offsets, side_rows, side_offsets


def affine_actuation_map(s1_interval, s2_interval, u1, u2, *,
                         task_frames=None, slack=(0.0, 0.0)):
    """Return the affine ``([G c], [G_side c_side])`` maps for one route box.

    ``u1``/``u2`` are the prescribed field directions, held fixed inside the
    box.  ``task_frames`` fixes the commanded force frame, by default the
    centerline frame at the box centre.
    """

    positions, first, second = _route_positions(s1_interval, s2_interval, slack)
    if task_frames is None:
        task_frames = frames(0.5 * sum(first), 0.5 * sum(second))
    moments = (aligned_moment(0, u1), aligned_moment(1, u2))
    rows, offsets, side_rows, side_offsets = _assemble(
        positions, moments, task_frames
    )
    full = np.column_stack((
        np.asarray(rows, dtype=object), np.asarray(offsets, dtype=object)
    ))
    side_full = np.column_stack((
        np.asarray(side_rows, dtype=object), np.asarray(side_offsets, dtype=object)
    ))
    return consolidate(full), consolidate(side_full)


# ---------------------------------------------------------------------------
# Direction cells and the relaxed map used by the outside certificate
# ---------------------------------------------------------------------------

def direction_cells(divisions=2, half_sphere=False):
    """Tile the direction sphere with gnomonic cube-face patches.

    A unit vector belongs to the face of its largest component, so the six
    faces partition the sphere exactly and a cell is described by
    ``(axis, sign, x range, y range)`` with ``u = (sign*e_axis + x*e_b +
    y*e_c)`` normalized and ``x, y in [-1, 1]``.  Unlike a latitude band this
    chart is smooth and keeps all three components as functions of the same
    two noise symbols, which is what makes the relaxed enclosure tight enough
    to be useful.

    ``half_sphere`` keeps only the three positive faces.  Their antipodes are
    the three negative faces, and reversing both moments reverses every
    current while leaving all forces unchanged, so one magnet may be
    restricted to that half without losing a solution.
    """

    if divisions < 1:
        raise ValueError("cell divisions must be positive")
    edges = np.linspace(-1.0, 1.0, divisions + 1)
    signs = (1.0,) if half_sphere else (1.0, -1.0)
    return [
        (axis, sign,
         (float(edges[i]), float(edges[i + 1])),
         (float(edges[j]), float(edges[j + 1])))
        for axis in range(3) for sign in signs
        for i in range(divisions) for j in range(divisions)
    ]


def cell_direction_forms(cell):
    """Affine enclosure of every unit vector in one cell.

    The three components share the two chart symbols, so products such as the
    mutual force keep the constraint that they come from a single direction.
    """

    axis, sign, x_range, y_range = cell
    x = ca.Affine.from_interval(interval[x_range[0], x_range[1]])
    y = ca.Affine.from_interval(interval[y_range[0], y_range[1]])
    vector = [None, None, None]
    vector[axis] = ca.Affine(float(sign))
    vector[(axis + 1) % 3] = x
    vector[(axis + 2) % 3] = y
    squared = ca.Affine(1.0) + _square(x) + _square(y)
    inverse_norm = _negative_power(squared, -0.5)
    return tuple(_product(component, inverse_norm) for component in vector)


def _square(form):
    """Enclose ``x**2``; the neglected remainder is nonnegative."""

    radius = sum(abs(value) for value in form.deviations.values())
    deviations = {
        symbol: 2.0 * form.center * value
        for symbol, value in form.deviations.items()
    }
    if radius:
        deviations[ca.get_next_symbol_index()] = 0.5 * radius**2
    return ca.Affine(form.center**2 + 0.5 * radius**2, deviations)


def cone_actuation_map(s1_interval, s2_interval, cell1, cell2, *,
                       task_frames=None, slack=(0.0, 0.0), f_side=F_SIDE):
    """Relaxed map enclosing every field direction pair in two cells.

    Two changes make the one-sided outside test applicable to a continuum of
    admissible moment directions.  The direction-dependent field command is
    moved into the offset column, so all cells share the task vertex zero on
    the field rows.  The four transverse inequality rows are turned into
    equality rows by four bounded virtual actuators, which reproduces the side
    constraints exactly inside a certificate that only understands a current
    box.  Dropping no constraint keeps the relaxation valid: the enclosed set
    contains every attainable output of the original problem.
    """

    positions, first, second = _route_positions(s1_interval, s2_interval, slack)
    if task_frames is None:
        task_frames = frames(0.5 * sum(first), 0.5 * sum(second))
    moments, targets = [], []
    for index, cell in enumerate((cell1, cell2)):
        forms = cell_direction_forms(cell)
        # The same direction forms scale both the moment and the commanded
        # field, so the relaxation cannot pick one direction for the force and
        # a different one for the field.
        moments.append(tuple(MOMENT[index] * form for form in forms))
        targets.append(tuple(B_HOLD * form for form in forms))
    rows, offsets, side_rows, side_offsets = _assemble(
        positions, moments, task_frames, field_targets=targets
    )
    zero = ca.Affine(0.0)
    augmented = []
    for row in rows:
        augmented.append(list(row) + [zero] * 4)
    for index, row in enumerate(side_rows):
        slack_columns = [zero] * 4
        slack_columns[index] = ca.Affine(-1.0)
        augmented.append(list(row) + slack_columns)
    full = np.column_stack((
        np.asarray(augmented, dtype=object),
        np.asarray(list(offsets) + list(side_offsets), dtype=object),
    ))
    lower = np.concatenate((np.full(NUM_COILS, I_MIN), np.full(4, -f_side)))
    upper = np.concatenate((np.full(NUM_COILS, I_MAX), np.full(4, f_side)))
    return consolidate(full), lower, upper


def cone_task_vertices(f_drive=F_DRIVE):
    """Task vertices of the relaxed twelve-row system."""

    drives = (-f_drive, f_drive) if f_drive else (0.0,)
    return np.asarray([
        [0.0, 0.0, 0.0, first, 0.0, 0.0, 0.0, second,
         0.0, NET_WEIGHT[0], 0.0, NET_WEIGHT[1]]
        for first, second in product(drives, drives)
    ], dtype=float)


# ---------------------------------------------------------------------------
# Numeric point model, used by the per-box direction search
# ---------------------------------------------------------------------------

def numeric_basis(s1, s2):
    """Direction-independent coil data and frames at one configuration."""

    positions = (ROUTES[0].point(s1), ROUTES[1].point(s2))
    basis = tuple(coil_basis(position) for position in positions)
    return positions, basis, frames(s1, s2)


def _numeric_dipole(target, source, moment):
    r = np.asarray(target, dtype=float) - np.asarray(source, dtype=float)
    norm = float(np.linalg.norm(r))
    if norm <= 0.0:
        raise SeparationError("dipole source and target overlap")
    dot = float(np.asarray(moment) @ r)
    field = MU0_OVER_4PI * (3.0 * r * dot / norm**5 - np.asarray(moment) / norm**3)
    gradient = np.empty((3, 3))
    for a in range(3):
        for b in range(3):
            gradient[a, b] = MU0_OVER_4PI * (
                3.0 * ((a == b) * dot + r[a] * moment[b] + moment[a] * r[b]) / norm**5
                - 15.0 * r[a] * r[b] * dot / norm**7
            )
    return field, gradient


def point_maps(basis, u1, u2):
    """Numeric ``(G, c, G_side, c_side)`` for one field-direction pair."""

    positions, coil_data, task_frames = basis
    moments = (aligned_moment(0, u1), aligned_moment(1, u2))
    rows, side_rows = [], []
    for index in (0, 1):
        H, D = coil_data[index]
        tangent, normal, depth = task_frames[index]
        rows.extend(H)
        projected = [
            np.einsum("b,bak,a->k", moments[index], D, direction)
            for direction in (tangent, normal, depth)
        ]
        rows.append(projected[0])
        side_rows.extend(projected[1:])

    field_on_1, _ = _numeric_dipole(positions[0], positions[1], moments[1])
    field_on_2, gradient_on_2 = _numeric_dipole(
        positions[1], positions[0], moments[0]
    )
    force_2 = gradient_on_2.T @ moments[1]
    force_1 = -force_2
    offsets = np.concatenate((
        field_on_1, [force_1 @ task_frames[0][0]],
        field_on_2, [force_2 @ task_frames[1][0]],
    ))
    side_offsets = np.array([
        force_1 @ task_frames[0][1], force_1 @ task_frames[0][2],
        force_2 @ task_frames[1][1], force_2 @ task_frames[1][2],
    ])
    return (np.asarray(rows), offsets, np.asarray(side_rows), side_offsets)


def task_vertices(u1, u2, f_drive=F_DRIVE):
    """The four controlled task vertices; only the two drive forces vary."""

    if not (math.isfinite(f_drive) and f_drive >= 0.0):
        raise ValueError("drive force must be finite and nonnegative")
    field_1 = B_HOLD * np.asarray(u1, dtype=float) / np.linalg.norm(u1)
    field_2 = B_HOLD * np.asarray(u2, dtype=float) / np.linalg.norm(u2)
    drives = (-f_drive, f_drive) if f_drive else (0.0,)
    return np.asarray([
        [field_1[0], field_1[1], field_1[2], first,
         field_2[0], field_2[1], field_2[2], second]
        for first, second in product(drives, drives)
    ], dtype=float)


def side_bounds(f_side=F_SIDE):
    """Inequality bounds on the four transverse force rows."""

    if not (math.isfinite(f_side) and f_side >= 0.0):
        raise ValueError("transverse bound must be finite and nonnegative")
    lower = np.array([
        -f_side, NET_WEIGHT[0] - f_side, -f_side, NET_WEIGHT[1] - f_side,
    ])
    upper = np.array([
        f_side, NET_WEIGHT[0] + f_side, f_side, NET_WEIGHT[1] + f_side,
    ])
    return lower, upper


def peak_current(basis, u1, u2, *, f_drive=F_DRIVE, f_side=F_SIDE):
    """Smallest achievable ``max|I|`` over the four task vertices, or inf."""

    G, c, G_side, c_side = point_maps(basis, u1, u2)
    lower, upper = side_bounds(f_side)
    n = NUM_COILS
    objective = np.zeros(n + 1)
    objective[-1] = 1.0
    A_ub = np.vstack((
        np.hstack((np.eye(n), -np.ones((n, 1)))),
        np.hstack((-np.eye(n), -np.ones((n, 1)))),
        np.hstack((G_side, np.zeros((G_side.shape[0], 1)))),
        np.hstack((-G_side, np.zeros((G_side.shape[0], 1)))),
    ))
    b_ub = np.concatenate((np.zeros(2 * n), upper - c_side, c_side - lower))
    A_eq = np.hstack((G, np.zeros((G.shape[0], 1))))
    worst = 0.0
    for task in task_vertices(u1, u2, f_drive):
        result = linprog(
            objective, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=task - c,
            bounds=[(None, None)] * n + [(0.0, None)], method="highs",
        )
        if not result.success:
            return float("inf")
        worst = max(worst, float(result.fun))
    return worst


def unit_vector(theta, phi):
    return np.array([
        math.sin(theta) * math.cos(phi),
        math.sin(theta) * math.sin(phi),
        math.cos(theta),
    ])


def _to_angles(vector):
    vector = np.asarray(vector, dtype=float)
    vector = vector / np.linalg.norm(vector)
    return np.array([
        math.acos(float(np.clip(vector[2], -1.0, 1.0))),
        math.atan2(float(vector[1]), float(vector[0])),
    ])


def half_sphere_directions(count):
    """Roughly uniform unit vectors on a half sphere.

    Reversing both prescribed field directions reverses every current and
    leaves the mutual force unchanged, so half the sphere covers the search.
    """

    if count < 1:
        raise ValueError("direction count must be positive")
    golden = math.pi * (3.0 - math.sqrt(5.0))
    vectors = []
    for index in range(count):
        z = 1.0 - (index + 0.5) / count
        radius = math.sqrt(max(0.0, 1.0 - z * z))
        angle = golden * index
        vectors.append(np.array([
            radius * math.cos(angle), radius * math.sin(angle), z
        ]))
    return vectors


def optimize_directions(basis, *, f_drive=F_DRIVE, f_side=F_SIDE,
                        shared_count=14, refine=2, hint=None,
                        easy_threshold=0.5 * I_MAX):
    """Search the field-direction pair minimizing the worst peak current.

    A shared direction is tried first: away from the crossing the cheapest
    solution keeps both prescribed fields nearly parallel, and near the
    crossing a shared direction is the only affordable option because the coil
    field cannot rotate quickly over a few millimetres.  ``easy_threshold``
    stops the search as soon as a comfortable allocation is in hand.
    """

    def evaluate(u1, u2):
        try:
            return peak_current(basis, u1, u2, f_drive=f_drive, f_side=f_side)
        except (SeparationError, ArithmeticError, ValueError,
                np.linalg.LinAlgError):
            return float("inf")

    best_value = float("inf")
    best_pair = None
    if hint is not None:
        value = evaluate(hint[0], hint[1])
        if math.isfinite(value):
            best_value = value
            best_pair = (np.asarray(hint[0], dtype=float),
                         np.asarray(hint[1], dtype=float))
    candidates = half_sphere_directions(shared_count)
    scored = []
    for direction in candidates:
        value = evaluate(direction, direction)
        scored.append((value, direction))
        if value < best_value:
            best_value, best_pair = value, (direction, direction)
    # Independent directions only pay off where a shared field is expensive,
    # which is the neighbourhood of the crossing.
    if best_value > easy_threshold:
        scored.sort(key=lambda item: item[0])
        for _, direction in scored[:refine]:
            for other in candidates:
                value = evaluate(direction, other)
                if value < best_value:
                    best_value, best_pair = value, (direction, other)
    if best_pair is None:
        return None, float("inf")
    if best_value <= easy_threshold:
        return best_pair, best_value

    angles = np.concatenate([_to_angles(best_pair[0]), _to_angles(best_pair[1])])
    result = minimize(
        lambda x: evaluate(unit_vector(x[0], x[1]), unit_vector(x[2], x[3])),
        angles, method="Powell",
        options={"xtol": 1e-3, "ftol": 1e-3, "maxiter": 60},
    )
    if float(result.fun) < best_value:
        best_value = float(result.fun)
        best_pair = (unit_vector(result.x[0], result.x[1]),
                     unit_vector(result.x[2], result.x[3]))
    return best_pair, best_value


# ---------------------------------------------------------------------------
# Geometry-derived filters
# ---------------------------------------------------------------------------

def _max_speed(route, samples=101):
    return max(
        float(np.linalg.norm(route.derivative(value)))
        for value in np.linspace(0.0, 1.0, samples)
    )


_MAX_SPEED = tuple(_max_speed(route) for route in ROUTES)


def separation_bounds(s1_interval, s2_interval, samples=9):
    """Conservative min/max centre distance between two route boxes.

    Sampled distances are inflated by the maximum centerline speed times half a
    sample step, so the bound covers the whole continuum.
    """

    first, second = _bounds(s1_interval), _bounds(s2_interval)
    a = np.linspace(first[0], first[1], samples)
    b = np.linspace(second[0], second[1], samples)
    points_a = np.asarray([ROUTES[0].point(value) for value in a])
    points_b = np.asarray([ROUTES[1].point(value) for value in b])
    distances = np.linalg.norm(points_a[:, None, :] - points_b[None, :, :], axis=-1)
    step_a = (first[1] - first[0]) / (samples - 1)
    step_b = (second[1] - second[0]) / (samples - 1)
    slack = 0.5 * (_MAX_SPEED[0] * step_a + _MAX_SPEED[1] * step_b)
    return max(0.0, float(distances.min()) - slack), float(distances.max()) + slack


def stable_alignment_radius(bias_field=B_HOLD):
    """Sufficient centre distance for worst-direction alignment stability."""

    return (
        2.0 * MU0_OVER_4PI * math.sqrt(MOMENT[0] * MOMENT[1]) / bias_field
    ) ** (1.0 / 3.0)
