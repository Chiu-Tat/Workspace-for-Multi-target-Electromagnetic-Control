"""Tighter sufficient certificates for interval workspace classification.

The routines in this module address a specific source of indeterminacy in the
original region checker.  The original robust-feasibility LP treats every entry
of an interval actuation matrix as independently variable.  Physical matrices
lie on a much lower-dimensional, correlated manifold, so that relaxation can be
very conservative even when affine arithmetic produced a tight enclosure.

This module supplies two complementary tests for

    h = c(q) + G(q) i,  q in Q,  i in [i_lower, i_upper].

``constructive_inside_certificate`` proves that every requested task vertex has
an admissible current for every q in Q.  It constructs a parameter-dependent
current correction around a nominal matrix and bounds that correction with a
Neumann majorant.  ``dual_outside_certificate`` proves that one task vertex is
separated from every attainable-output zonotope in Q by a fixed hyperplane.

Both tests are one-sided: ``True`` is a mathematical certificate, while
``False`` only means that the test was inconclusive.  Their guarantees require
``G_lower/G_upper`` and ``c_lower/c_upper`` to be genuine inclusion bounds.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable

import numpy as np
from scipy.optimize import linprog


_DEFAULT_TOL = 1e-10


@dataclass(frozen=True)
class InsideCertificate:
    """Diagnostic output of the constructive inside test."""

    certified: bool
    reason: str
    spectral_radius: float = float("inf")
    minimum_reserve: float = float("-inf")
    maximum_correction: float = float("inf")
    vertex_currents: tuple[np.ndarray, ...] = ()
    correction_bounds: tuple[np.ndarray, ...] = ()


@dataclass(frozen=True)
class OutsideCertificate:
    """Diagnostic output of the dual outside test."""

    certified: bool
    reason: str
    separation_margin: float = float("-inf")
    task_vertex: np.ndarray | None = None
    normal: np.ndarray | None = None


def _affine_linear_combination(forms: Iterable[object], weights: np.ndarray) -> float:
    """Bound the absolute value of a linear combination of affine forms."""

    center = 0.0
    coefficients: dict[int, float] = {}
    for form, weight in zip(forms, weights):
        center += float(weight) * float(form.center)
        for symbol, coefficient in form.deviations.items():
            coefficients[symbol] = coefficients.get(symbol, 0.0) + float(weight) * float(coefficient)
    return abs(center) + sum(abs(value) for value in coefficients.values())


def _affine_robust_current(
    G_affine: np.ndarray,
    c_affine: np.ndarray,
    G_nominal: np.ndarray,
    c_nominal: np.ndarray,
    task: np.ndarray,
    current_lower: np.ndarray,
    current_upper: np.ndarray,
    correction_gain: np.ndarray,
    tol: float,
    side_G_affine: np.ndarray | None = None,
    side_c_affine: np.ndarray | None = None,
    side_lower: np.ndarray | None = None,
    side_upper: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, float] | None:
    """Jointly choose a nominal current and its certified uncertainty reserve.

    For every affine noise symbol ``s`` and task row ``r``, an auxiliary
    variable majorizes

        |c_rs + sum_j G_rjs i_j|.

    The resulting row sums bound ``|Delta c + Delta G i|``.  Current-box
    constraints include the induced correction ``correction_gain @ row_sum``
    directly, so redundancy is allocated to robustness rather than merely to
    nominal distance from the current limits.
    """

    d, n = G_affine.shape
    row_symbols: list[list[int]] = []
    coefficient_rows: list[tuple[int, int]] = []
    for row in range(d):
        symbols = set(c_affine[row].deviations)
        for form in G_affine[row, :]:
            symbols.update(form.deviations)
        ordered = sorted(symbols)
        row_symbols.append(ordered)
        coefficient_rows.extend((row, symbol) for symbol in ordered)

    side_coefficient_rows: list[tuple[int, int]] = []
    if side_G_affine is not None:
        for row in range(side_G_affine.shape[0]):
            symbols = set(side_c_affine[row].deviations)
            for form in side_G_affine[row, :]:
                symbols.update(form.deviations)
            side_coefficient_rows.extend((row, symbol) for symbol in sorted(symbols))

    auxiliary_count = len(coefficient_rows) + len(side_coefficient_rows)
    t_index = n + auxiliary_count
    variable_count = t_index + 1
    objective = np.zeros(variable_count)
    objective[t_index] = -1.0
    constraints: list[np.ndarray] = []
    rhs: list[float] = []

    # Map each (row, symbol) pair to its absolute-value auxiliary variable.
    auxiliary_index = {
        pair: n + offset for offset, pair in enumerate(coefficient_rows)
    }
    for row, symbol in coefficient_rows:
        expression = np.zeros(variable_count)
        for col in range(n):
            expression[col] = float(G_affine[row, col].deviations.get(symbol, 0.0))
        constant = float(c_affine[row].deviations.get(symbol, 0.0))
        expression[auxiliary_index[(row, symbol)]] = -1.0
        constraints.append(expression)
        rhs.append(-constant)
        constraints.append(-expression)
        # Negating also changes the -u coefficient; restore it to -1.
        constraints[-1][auxiliary_index[(row, symbol)]] = -1.0
        rhs.append(constant)

    side_auxiliary_index = {
        pair: n + len(coefficient_rows) + offset
        for offset, pair in enumerate(side_coefficient_rows)
    }
    for row, symbol in side_coefficient_rows:
        expression = np.zeros(variable_count)
        for col in range(n):
            expression[col] = float(
                side_G_affine[row, col].deviations.get(symbol, 0.0)
            )
        constant = float(side_c_affine[row].deviations.get(symbol, 0.0))
        expression[side_auxiliary_index[(row, symbol)]] = -1.0
        constraints.append(expression)
        rhs.append(-constant)
        constraints.append(-expression)
        constraints[-1][side_auxiliary_index[(row, symbol)]] = -1.0
        rhs.append(constant)

    # The correction for current j is sum_r gain[j,r] sum_s u[r,s].
    for col in range(n):
        upper_constraint = np.zeros(variable_count)
        upper_constraint[col] = 1.0
        lower_constraint = np.zeros(variable_count)
        lower_constraint[col] = -1.0
        for row, symbol in coefficient_rows:
            index = auxiliary_index[(row, symbol)]
            upper_constraint[index] += correction_gain[col, row]
            lower_constraint[index] += correction_gain[col, row]
        upper_constraint[t_index] = 1.0
        lower_constraint[t_index] = 1.0
        constraints.append(upper_constraint)
        rhs.append(float(current_upper[col]))
        constraints.append(lower_constraint)
        rhs.append(float(-current_lower[col]))

    # Optional affine side outputs are inequalities, not controlled task rows.
    # Their uncertainty includes both direct affine variation and the effect of
    # the certified current correction used to satisfy the controlled rows.
    if side_G_affine is not None:
        side_count = side_G_affine.shape[0]
        side_abs_G = np.empty((side_count, n))
        for row in range(side_count):
            for col in range(n):
                form = side_G_affine[row, col]
                side_abs_G[row, col] = abs(float(form.center)) + sum(
                    abs(float(value)) for value in form.deviations.values()
                )
        for side_row in range(side_count):
            upper_constraint = np.zeros(variable_count)
            lower_constraint = np.zeros(variable_count)
            for col in range(n):
                center = float(side_G_affine[side_row, col].center)
                upper_constraint[col] = center
                lower_constraint[col] = -center
            for row, symbol in side_coefficient_rows:
                if row == side_row:
                    index = side_auxiliary_index[(row, symbol)]
                    upper_constraint[index] = 1.0
                    lower_constraint[index] = 1.0
            for controlled_row, symbol in coefficient_rows:
                index = auxiliary_index[(controlled_row, symbol)]
                correction_effect = sum(
                    side_abs_G[side_row, col]
                    * correction_gain[col, controlled_row]
                    for col in range(n)
                )
                upper_constraint[index] += correction_effect
                lower_constraint[index] += correction_effect
            side_center = float(side_c_affine[side_row].center)
            constraints.append(upper_constraint)
            rhs.append(float(side_upper[side_row]) - side_center)
            constraints.append(lower_constraint)
            rhs.append(-float(side_lower[side_row]) + side_center)

    A_eq = np.zeros((d, variable_count))
    A_eq[:, :n] = G_nominal
    b_eq = task - c_nominal
    maximum_possible_reserve = 0.5 * float(np.min(current_upper - current_lower))
    bounds = [(float(lo), float(hi)) for lo, hi in zip(current_lower, current_upper)]
    bounds.extend([(0.0, None)] * auxiliary_count)
    bounds.append((0.0, maximum_possible_reserve))
    result = linprog(
        objective,
        A_ub=np.asarray(constraints),
        b_ub=np.asarray(rhs),
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )
    if not result.success:
        return None
    current = result.x[:n]
    forcing = np.zeros(d)
    for row, symbol in coefficient_rows:
        forcing[row] += result.x[auxiliary_index[(row, symbol)]]
    correction = correction_gain @ forcing
    verified_margin = float(min(
        np.min(current - correction - current_lower),
        np.min(current_upper - current - correction),
    ))
    if verified_margin < -tol:
        return None
    return current, correction, min(float(result.x[t_index]), verified_margin)


def box_vertices(lower: Iterable[float], upper: Iterable[float]) -> np.ndarray:
    """Return the vertices of an axis-aligned box as a ``(2**d, d)`` array."""

    lower_array = np.asarray(lower, dtype=float)
    upper_array = np.asarray(upper, dtype=float)
    if lower_array.shape != upper_array.shape or lower_array.ndim != 1:
        raise ValueError("lower and upper must be one-dimensional arrays of equal shape")
    if np.any(lower_array > upper_array):
        raise ValueError("lower must not exceed upper")
    return np.asarray(list(product(*zip(lower_array, upper_array))), dtype=float)


def interval_object_bounds(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Extract outward-inflated bounds from the project's interval objects."""

    array = np.asarray(matrix, dtype=object)
    lower = np.empty(array.shape, dtype=float)
    upper = np.empty(array.shape, dtype=float)
    for index in np.ndindex(array.shape):
        value = array[index]
        if hasattr(value, "lower") and hasattr(value, "upper"):
            lower[index] = float(value.lower)
            upper[index] = float(value.upper)
        else:
            lower[index] = float(value[0][0])
            upper[index] = float(value[0][1])
    # One ulp of inflation protects the certificate from endpoint conversion.
    return np.nextafter(lower, -np.inf), np.nextafter(upper, np.inf)


def _validate_problem(
    G_lower: np.ndarray,
    G_upper: np.ndarray,
    c_lower: np.ndarray,
    c_upper: np.ndarray,
    task_vertices: np.ndarray,
    current_lower: np.ndarray,
    current_upper: np.ndarray,
) -> tuple[int, int]:
    d, n = G_lower.shape
    if G_upper.shape != (d, n):
        raise ValueError("G bounds must have equal shape")
    if c_lower.shape != (d,) or c_upper.shape != (d,):
        raise ValueError("c bounds must have shape (number of task components,)")
    if task_vertices.ndim != 2 or task_vertices.shape[1] != d:
        raise ValueError("task_vertices must have shape (number of vertices, d)")
    if current_lower.shape != (n,) or current_upper.shape != (n,):
        raise ValueError("current bounds must have shape (number of actuators,)")
    arrays = (G_lower, G_upper, c_lower, c_upper, task_vertices,
              current_lower, current_upper)
    if not all(np.all(np.isfinite(value)) for value in arrays):
        raise ValueError("all certificate inputs must be finite")
    if np.any(G_lower > G_upper) or np.any(c_lower > c_upper):
        raise ValueError("invalid interval bounds")
    if np.any(current_lower > current_upper):
        raise ValueError("invalid current bounds")
    return d, n


def _maximum_reserve_current(
    G_nominal: np.ndarray,
    c_nominal: np.ndarray,
    task: np.ndarray,
    current_lower: np.ndarray,
    current_upper: np.ndarray,
    tol: float,
) -> tuple[np.ndarray, float] | None:
    """Find a nominal current that maximizes its componentwise box reserve."""

    n = G_nominal.shape[1]
    objective = np.zeros(n + 1)
    objective[-1] = -1.0

    # i_j + t <= upper_j and -i_j + t <= -lower_j.
    upper_constraints = np.hstack((np.eye(n), np.ones((n, 1))))
    lower_constraints = np.hstack((-np.eye(n), np.ones((n, 1))))
    A_ub = np.vstack((upper_constraints, lower_constraints))
    b_ub = np.concatenate((current_upper, -current_lower))
    A_eq = np.hstack((G_nominal, np.zeros((G_nominal.shape[0], 1))))
    b_eq = task - c_nominal
    maximum_possible_reserve = 0.5 * float(np.min(current_upper - current_lower))
    bounds = [(float(lo), float(hi)) for lo, hi in zip(current_lower, current_upper)]
    bounds.append((0.0, maximum_possible_reserve))

    result = linprog(
        objective,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )
    if not result.success:
        return None
    current = result.x[:n]
    reserve = float(result.x[-1])
    residual = np.linalg.norm(G_nominal @ current + c_nominal - task, ord=np.inf)
    if residual > max(tol, 1e-8 * max(1.0, np.linalg.norm(task, ord=np.inf))):
        return None
    return current, reserve


def constructive_inside_certificate(
    G_lower: np.ndarray,
    G_upper: np.ndarray,
    task_vertices: np.ndarray,
    current_lower: np.ndarray,
    current_upper: np.ndarray,
    *,
    c_lower: np.ndarray | None = None,
    c_upper: np.ndarray | None = None,
    G_nominal: np.ndarray | None = None,
    c_nominal: np.ndarray | None = None,
    tol: float = _DEFAULT_TOL,
) -> InsideCertificate:
    """Certify a whole configuration box by constructive current recourse.

    Let ``R`` be a right inverse of ``G_nominal``.  For a nominal current
    ``i_v`` associated with task vertex ``h_v``, define

        i_v(q) = i_v - R (I + (G(q)-G0)R)^(-1)
                         ((c(q)-c0) + (G(q)-G0)i_v).

    This current satisfies the task equation exactly.  If ``E_bar`` bounds the
    absolute value of ``(G-G0)R`` and ``rho(E_bar) < 1``, the Neumann series
    gives a componentwise bound on the correction.  The box is certified when
    that bound fits within the current reserve for every task vertex.  Convexity
    then extends the result from vertices to the entire task polytope.
    """

    G_lower = np.asarray(G_lower, dtype=float)
    G_upper = np.asarray(G_upper, dtype=float)
    task_vertices = np.asarray(task_vertices, dtype=float)
    current_lower = np.asarray(current_lower, dtype=float)
    current_upper = np.asarray(current_upper, dtype=float)
    d, _ = G_lower.shape
    if c_lower is None:
        c_lower = np.zeros(d)
    if c_upper is None:
        c_upper = np.zeros(d)
    c_lower = np.asarray(c_lower, dtype=float)
    c_upper = np.asarray(c_upper, dtype=float)
    _validate_problem(
        G_lower, G_upper, c_lower, c_upper, task_vertices,
        current_lower, current_upper,
    )

    if G_nominal is None:
        G_nominal = 0.5 * (G_lower + G_upper)
    if c_nominal is None:
        c_nominal = 0.5 * (c_lower + c_upper)
    G_nominal = np.asarray(G_nominal, dtype=float)
    c_nominal = np.asarray(c_nominal, dtype=float)
    if G_nominal.shape != G_lower.shape or c_nominal.shape != (d,):
        raise ValueError("nominal values have incompatible shape")
    if np.linalg.matrix_rank(G_nominal, tol=tol) < d:
        return InsideCertificate(False, "nominal actuation matrix lacks full row rank")

    # Minimum-norm right inverse.  The proof accepts any R satisfying G0 R = I.
    gram = G_nominal @ G_nominal.T
    try:
        right_inverse = G_nominal.T @ np.linalg.solve(gram, np.eye(d))
    except np.linalg.LinAlgError:
        return InsideCertificate(False, "right-inverse construction failed")
    right_inverse_residual = np.linalg.norm(
        G_nominal @ right_inverse - np.eye(d), ord=np.inf
    )
    if right_inverse_residual > max(tol, 1e-8):
        return InsideCertificate(False, "right-inverse residual is too large")

    delta_G_abs = np.maximum(
        np.abs(G_lower - G_nominal), np.abs(G_upper - G_nominal)
    )
    delta_c_abs = np.maximum(
        np.abs(c_lower - c_nominal), np.abs(c_upper - c_nominal)
    )
    E_bar = delta_G_abs @ np.abs(right_inverse)
    spectral_radius = float(np.max(np.abs(np.linalg.eigvals(E_bar))))
    if not np.isfinite(spectral_radius) or spectral_radius >= 1.0 - tol:
        return InsideCertificate(
            False,
            "Neumann contraction condition failed",
            spectral_radius=spectral_radius,
        )

    try:
        inverse_majorant = np.linalg.solve(np.eye(d) - E_bar, np.eye(d))
    except np.linalg.LinAlgError:
        return InsideCertificate(
            False,
            "Neumann majorant solve failed",
            spectral_radius=spectral_radius,
        )
    # For a nonnegative E_bar with rho < 1 this matrix is nonnegative.  Reject
    # numerically inconsistent cases rather than weakening the certificate.
    if np.min(inverse_majorant) < -100.0 * tol:
        return InsideCertificate(
            False,
            "computed Neumann majorant is not nonnegative",
            spectral_radius=spectral_radius,
        )
    inverse_majorant = np.maximum(inverse_majorant, 0.0)

    correction_gain = np.abs(right_inverse) @ inverse_majorant
    currents: list[np.ndarray] = []
    corrections: list[np.ndarray] = []
    minimum_reserve = float("inf")
    maximum_correction = 0.0

    for task in task_vertices:
        nominal_solution = _maximum_reserve_current(
            G_nominal,
            c_nominal,
            task,
            current_lower,
            current_upper,
            tol,
        )
        if nominal_solution is None:
            return InsideCertificate(
                False,
                "a task vertex is infeasible at the nominal matrix",
                spectral_radius=spectral_radius,
                minimum_reserve=minimum_reserve,
                maximum_correction=maximum_correction,
                vertex_currents=tuple(currents),
                correction_bounds=tuple(corrections),
            )
        current, reserve = nominal_solution
        forcing_bound = delta_c_abs + delta_G_abs @ np.abs(current)
        correction = np.abs(right_inverse) @ (inverse_majorant @ forcing_bound)
        lower_margin = current - correction - current_lower
        upper_margin = current_upper - current - correction
        certified_margin = float(min(np.min(lower_margin), np.min(upper_margin)))
        minimum_reserve = min(minimum_reserve, reserve, certified_margin)
        maximum_correction = max(maximum_correction, float(np.max(correction)))
        currents.append(current)
        corrections.append(correction)
        if certified_margin < -tol:
            return InsideCertificate(
                False,
                "bounded current correction exceeds the available reserve",
                spectral_radius=spectral_radius,
                minimum_reserve=minimum_reserve,
                maximum_correction=maximum_correction,
                vertex_currents=tuple(currents),
                correction_bounds=tuple(corrections),
            )

    return InsideCertificate(
        True,
        "constructive right-inverse certificate succeeded",
        spectral_radius=spectral_radius,
        minimum_reserve=minimum_reserve,
        maximum_correction=maximum_correction,
        vertex_currents=tuple(currents),
        correction_bounds=tuple(corrections),
    )


def affine_constructive_inside_certificate(
    G_affine: np.ndarray,
    task_vertices: np.ndarray,
    current_lower: np.ndarray,
    current_upper: np.ndarray,
    *,
    c_affine: np.ndarray | None = None,
    side_G_affine: np.ndarray | None = None,
    side_c_affine: np.ndarray | None = None,
    side_lower: np.ndarray | None = None,
    side_upper: np.ndarray | None = None,
    tol: float = _DEFAULT_TOL,
) -> InsideCertificate:
    """Dependency-aware form of the constructive right-inverse certificate.

    Unlike :func:`constructive_inside_certificate`, this routine retains the
    common noise symbols while bounding ``(G-G0)R`` and ``(G-G0)i_v``.  It is
    therefore strictly no worse than first collapsing every matrix entry to an
    independent interval, subject to roundoff.  The guarantee requires the
    input affine forms themselves to be inclusion-preserving.
    """

    G_affine = np.asarray(G_affine, dtype=object)
    if G_affine.ndim != 2:
        raise ValueError("G_affine must be a two-dimensional array")
    d, n = G_affine.shape
    task_vertices = np.asarray(task_vertices, dtype=float)
    current_lower = np.asarray(current_lower, dtype=float)
    current_upper = np.asarray(current_upper, dtype=float)
    if task_vertices.ndim != 2 or task_vertices.shape[1] != d:
        raise ValueError("task_vertices must have shape (number of vertices, d)")
    if current_lower.shape != (n,) or current_upper.shape != (n,):
        raise ValueError("current bounds have incompatible shape")
    if c_affine is None:
        # A tiny local class would obscure diagnostics; reuse the first form's
        # type to construct exact affine zeros.
        form_type = type(G_affine.flat[0])
        c_affine = np.asarray([form_type(0.0) for _ in range(d)], dtype=object)
    else:
        c_affine = np.asarray(c_affine, dtype=object)
    if c_affine.shape != (d,):
        raise ValueError("c_affine must have shape (d,)")

    side_arguments = (side_G_affine, side_c_affine, side_lower, side_upper)
    if any(value is not None for value in side_arguments):
        if not all(value is not None for value in side_arguments):
            raise ValueError("all affine side-constraint arguments are required")
        side_G_affine = np.asarray(side_G_affine, dtype=object)
        side_c_affine = np.asarray(side_c_affine, dtype=object)
        side_lower = np.asarray(side_lower, dtype=float)
        side_upper = np.asarray(side_upper, dtype=float)
        if side_G_affine.ndim != 2 or side_G_affine.shape[1] != n:
            raise ValueError("side_G_affine must have shape (side outputs, actuators)")
        side_count = side_G_affine.shape[0]
        if side_c_affine.shape != (side_count,):
            raise ValueError("side_c_affine has incompatible shape")
        if side_lower.shape != (side_count,) or side_upper.shape != (side_count,):
            raise ValueError("side bounds have incompatible shape")
        if np.any(side_lower > side_upper):
            raise ValueError("invalid affine side bounds")

    G_nominal = np.asarray(
        [[float(G_affine[row, col].center) for col in range(n)] for row in range(d)]
    )
    c_nominal = np.asarray([float(form.center) for form in c_affine])
    if np.linalg.matrix_rank(G_nominal, tol=tol) < d:
        return InsideCertificate(False, "nominal actuation matrix lacks full row rank")
    try:
        right_inverse = G_nominal.T @ np.linalg.solve(
            G_nominal @ G_nominal.T, np.eye(d)
        )
    except np.linalg.LinAlgError:
        return InsideCertificate(False, "right-inverse construction failed")

    E_bar = np.zeros((d, d))
    for row in range(d):
        delta_row = []
        for col in range(n):
            form = G_affine[row, col]
            # Copy the form representation without importing the project class.
            delta = type(form)(
                float(form.center) - G_nominal[row, col],
                dict(form.deviations),
            )
            delta_row.append(delta)
        for output_col in range(d):
            E_bar[row, output_col] = _affine_linear_combination(
                delta_row, right_inverse[:, output_col]
            )
    spectral_radius = float(np.max(np.abs(np.linalg.eigvals(E_bar))))
    if not np.isfinite(spectral_radius) or spectral_radius >= 1.0 - tol:
        return InsideCertificate(
            False,
            "affine Neumann contraction condition failed",
            spectral_radius=spectral_radius,
        )
    try:
        inverse_majorant = np.linalg.solve(np.eye(d) - E_bar, np.eye(d))
    except np.linalg.LinAlgError:
        return InsideCertificate(
            False,
            "affine Neumann majorant solve failed",
            spectral_radius=spectral_radius,
        )
    if np.min(inverse_majorant) < -100.0 * tol:
        return InsideCertificate(
            False,
            "computed affine Neumann majorant is not nonnegative",
            spectral_radius=spectral_radius,
        )
    inverse_majorant = np.maximum(inverse_majorant, 0.0)

    correction_gain = np.abs(right_inverse) @ inverse_majorant
    currents: list[np.ndarray] = []
    corrections: list[np.ndarray] = []
    minimum_reserve = float("inf")
    maximum_correction = 0.0
    for task in task_vertices:
        robust_solution = _affine_robust_current(
            G_affine,
            c_affine,
            G_nominal,
            c_nominal,
            task,
            current_lower,
            current_upper,
            correction_gain,
            tol,
            side_G_affine=side_G_affine,
            side_c_affine=side_c_affine,
            side_lower=side_lower,
            side_upper=side_upper,
        )
        if robust_solution is None:
            return InsideCertificate(
                False,
                "joint robust-current allocation failed for a task vertex",
                spectral_radius=spectral_radius,
                minimum_reserve=minimum_reserve,
                maximum_correction=maximum_correction,
                vertex_currents=tuple(currents),
                correction_bounds=tuple(corrections),
            )
        current, correction, reserve = robust_solution
        lower_margin = current - correction - current_lower
        upper_margin = current_upper - current - correction
        certified_margin = float(min(np.min(lower_margin), np.min(upper_margin)))
        minimum_reserve = min(minimum_reserve, reserve, certified_margin)
        maximum_correction = max(maximum_correction, float(np.max(correction)))
        currents.append(current)
        corrections.append(correction)
        if certified_margin < -tol:
            return InsideCertificate(
                False,
                "affine current correction exceeds the available reserve",
                spectral_radius=spectral_radius,
                minimum_reserve=minimum_reserve,
                maximum_correction=maximum_correction,
                vertex_currents=tuple(currents),
                correction_bounds=tuple(corrections),
            )

    return InsideCertificate(
        True,
        "dependency-aware affine right-inverse certificate succeeded",
        spectral_radius=spectral_radius,
        minimum_reserve=minimum_reserve,
        maximum_correction=maximum_correction,
        vertex_currents=tuple(currents),
        correction_bounds=tuple(corrections),
    )


def _center_separating_normal(
    G_nominal: np.ndarray,
    c_nominal: np.ndarray,
    task: np.ndarray,
    current_center: np.ndarray,
    current_radius: np.ndarray,
) -> tuple[np.ndarray, float] | None:
    """Find an L1-normalized hyperplane separating a point and a zonotope."""

    d, n = G_nominal.shape
    residual = task - c_nominal - G_nominal @ current_center
    # Variables are lambda_plus, lambda_minus, and t_j >= |g_j^T lambda|.
    objective = np.concatenate((-residual, residual, current_radius))
    A_ub = np.zeros((2 * n + 1, 2 * d + n))
    b_ub = np.zeros(2 * n + 1)
    for j in range(n):
        column = G_nominal[:, j]
        A_ub[2 * j, :d] = column
        A_ub[2 * j, d:2 * d] = -column
        A_ub[2 * j, 2 * d + j] = -1.0
        A_ub[2 * j + 1, :d] = -column
        A_ub[2 * j + 1, d:2 * d] = column
        A_ub[2 * j + 1, 2 * d + j] = -1.0
    A_ub[-1, :2 * d] = 1.0
    b_ub[-1] = 1.0
    result = linprog(
        objective,
        A_ub=A_ub,
        b_ub=b_ub,
        bounds=[(0.0, None)] * (2 * d + n),
        method="highs",
    )
    if not result.success:
        return None
    normal = result.x[:d] - result.x[d:2 * d]
    margin = -float(result.fun)
    return normal, margin


def _interval_matrix_vector(
    lower: np.ndarray, upper: np.ndarray, vector: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    low_terms = np.minimum(lower * vector[np.newaxis, :], upper * vector[np.newaxis, :])
    high_terms = np.maximum(lower * vector[np.newaxis, :], upper * vector[np.newaxis, :])
    return np.sum(low_terms, axis=1), np.sum(high_terms, axis=1)


def dual_outside_certificate(
    G_lower: np.ndarray,
    G_upper: np.ndarray,
    task_vertices: np.ndarray,
    current_lower: np.ndarray,
    current_upper: np.ndarray,
    *,
    c_lower: np.ndarray | None = None,
    c_upper: np.ndarray | None = None,
    G_nominal: np.ndarray | None = None,
    c_nominal: np.ndarray | None = None,
    tol: float = _DEFAULT_TOL,
) -> OutsideCertificate:
    """Certify that a configuration box fails at least one requested task.

    For each task vertex, a center LP searches for a fixed separating normal
    ``lambda``.  The candidate is then verified over the full interval matrix:

        lambda^T h > sup_q [lambda^T(c(q)+G(q)i_c)
                            + sum_j r_j |lambda^T g_j(q)|].

    The expression on the right is an upper bound on the support function of
    every attainable-output zonotope in the configuration box.  A positive
    verified margin therefore proves that the same task is infeasible for every
    configuration in the box.
    """

    G_lower = np.asarray(G_lower, dtype=float)
    G_upper = np.asarray(G_upper, dtype=float)
    task_vertices = np.asarray(task_vertices, dtype=float)
    current_lower = np.asarray(current_lower, dtype=float)
    current_upper = np.asarray(current_upper, dtype=float)
    d, n = G_lower.shape
    if c_lower is None:
        c_lower = np.zeros(d)
    if c_upper is None:
        c_upper = np.zeros(d)
    c_lower = np.asarray(c_lower, dtype=float)
    c_upper = np.asarray(c_upper, dtype=float)
    _validate_problem(
        G_lower, G_upper, c_lower, c_upper, task_vertices,
        current_lower, current_upper,
    )

    if G_nominal is None:
        G_nominal = 0.5 * (G_lower + G_upper)
    if c_nominal is None:
        c_nominal = 0.5 * (c_lower + c_upper)
    G_nominal = np.asarray(G_nominal, dtype=float)
    c_nominal = np.asarray(c_nominal, dtype=float)
    current_center = 0.5 * (current_lower + current_upper)
    current_radius = 0.5 * (current_upper - current_lower)

    Gic_lower, Gic_upper = _interval_matrix_vector(
        G_lower, G_upper, current_center
    )
    output_center_lower = c_lower + Gic_lower
    output_center_upper = c_upper + Gic_upper

    best_margin = float("-inf")
    best_task = None
    best_normal = None
    for task in task_vertices:
        candidate = _center_separating_normal(
            G_nominal,
            c_nominal,
            task,
            current_center,
            current_radius,
        )
        if candidate is None:
            continue
        normal, center_margin = candidate
        if center_margin <= tol or np.linalg.norm(normal, ord=1) <= tol:
            continue

        positive = np.maximum(normal, 0.0)
        negative = np.minimum(normal, 0.0)
        center_support_upper = float(
            positive @ output_center_upper + negative @ output_center_lower
        )

        projected_lower = np.empty(n)
        projected_upper = np.empty(n)
        for j in range(n):
            terms_low = np.minimum(normal * G_lower[:, j], normal * G_upper[:, j])
            terms_high = np.maximum(normal * G_lower[:, j], normal * G_upper[:, j])
            projected_lower[j] = np.sum(terms_low)
            projected_upper[j] = np.sum(terms_high)
        projected_abs_upper = np.maximum(
            np.abs(projected_lower), np.abs(projected_upper)
        )
        support_upper = center_support_upper + float(current_radius @ projected_abs_upper)
        verified_margin = float(normal @ task - support_upper)
        if verified_margin > best_margin:
            best_margin = verified_margin
            best_task = task.copy()
            best_normal = normal.copy()
        if verified_margin > tol:
            return OutsideCertificate(
                True,
                "dual support-function separation succeeded",
                separation_margin=verified_margin,
                task_vertex=task.copy(),
                normal=normal.copy(),
            )

    return OutsideCertificate(
        False,
        "no fixed separating hyperplane survived interval verification",
        separation_margin=best_margin,
        task_vertex=best_task,
        normal=best_normal,
    )


def classify_interval_system(
    G_lower: np.ndarray,
    G_upper: np.ndarray,
    task_vertices: np.ndarray,
    current_lower: np.ndarray,
    current_upper: np.ndarray,
    **kwargs,
) -> tuple[str, InsideCertificate | OutsideCertificate]:
    """Return ``inside``, ``outside``, or ``undetermined`` with diagnostics."""

    inside = constructive_inside_certificate(
        G_lower,
        G_upper,
        task_vertices,
        current_lower,
        current_upper,
        **kwargs,
    )
    if inside.certified:
        return "inside", inside
    outside = dual_outside_certificate(
        G_lower,
        G_upper,
        task_vertices,
        current_lower,
        current_upper,
        **kwargs,
    )
    if outside.certified:
        return "outside", outside
    return "undetermined", inside
