"""Problem specification and actuation-matrix enclosures.

``RegionProblem`` collects everything the cascade needs: the configuration box
to search, the coil model plane, the current limits, and the requested task
box.  ``interval_matrix``/``affine_matrix`` wrap the project's field model and
reproduce the conventions used by ``Affine_10c.calculate_interval_A`` and
``calculate_affine_A`` exactly, including the trailing-column drop the affine
backend requires.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from ._backend import Map_I2H_Affine, Map_I2H_Interval_Fast, interval
from .certificates import box_vertices

__all__ = ["RegionProblem", "interval_matrix", "affine_matrix"]


# The field model is queried for the planar field components only.  Every other
# output row is disabled, matching the archived Fig. 12 configuration.
_TARGET_TEMPLATE = {
    "m": interval[0.0, 0.0],
    "alpha": interval[0.0, 0.0],
    "beta": interval[0.0, 0.0],
    "Bx": True,
    "By": True,
    "Bz": None,
    "Bx_dx": None,
    "Bx_dy": None,
    "Bx_dz": None,
    "By_dy": None,
    "By_dz": None,
    "fx": None,
    "fy": None,
    "fz": None,
    "tx": None,
    "ty": None,
    "tz": None,
}


def _target_points(x_interval, y_interval, z_offset: float):
    point = dict(_TARGET_TEMPLATE)
    point["X"] = x_interval
    point["Y"] = y_interval
    point["Z"] = interval[z_offset, z_offset]
    return [point]


def interval_matrix(x_interval, y_interval, z_offset: float) -> np.ndarray:
    """Standard interval enclosure of the actuation matrix ``G(q)``."""

    matrix, _row = Map_I2H_Interval_Fast(_target_points(x_interval, y_interval, z_offset))
    return matrix


def affine_matrix(x_interval, y_interval, z_offset: float, *, return_affine: bool = False):
    """Affine enclosure of ``G(q)``.

    With ``return_affine=True`` the entries are affine forms carrying shared
    noise symbols, which is what the dependency-preserving inside certificate
    needs.  With ``return_affine=False`` they are collapsed to intervals.

    The affine backend appends a constant column that the original code drops;
    that behaviour is preserved here.
    """

    matrix = Map_I2H_Affine(
        _target_points(x_interval, y_interval, z_offset), return_affine=return_affine
    )
    return matrix[:, :-1]


@dataclass(frozen=True)
class RegionProblem:
    """A certified-workspace query.

    Parameters
    ----------
    initial_box:
        ``(lower, upper)`` pair per configuration axis, in metres.
    z_offset:
        Coil-model plane in the model frame, in metres.
    task_box:
        ``(lower, upper)`` pair per requested field component, in tesla.  The
        default is the anisotropic stress case used in Fig. 12.
    current_bounds:
        ``(lower, upper)`` coil current limits in amperes, applied to every
        *active* coil.
    n_coils:
        Number of independently driven coils in the underlying model.
    active_coil_indices:
        Zero-based indices of coils that may carry current.  ``None`` keeps
        every coil.  Inactive columns are removed from the actuation map,
        which is equivalent to fixing those currents at zero.
    """

    initial_box: tuple[tuple[float, float], ...] = ((-0.05, 0.05), (-0.05, 0.05))
    z_offset: float = 0.075
    task_box: tuple[tuple[float, float], ...] = ((-0.12, 0.12), (-0.04, 0.04))
    current_bounds: tuple[float, float] = (-15.0, 15.0)
    n_coils: int = 10
    active_coil_indices: tuple[int, ...] | None = None

    # Derived, cached once because the cascade reuses them on every box.
    _derived: dict = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        for name, pairs in (("initial_box", self.initial_box), ("task_box", self.task_box)):
            for axis, pair in enumerate(pairs):
                if len(pair) != 2 or pair[0] > pair[1]:
                    raise ValueError(f"{name} axis {axis} is not a valid (lower, upper) pair")
        if self.current_bounds[0] > self.current_bounds[1]:
            raise ValueError("current_bounds lower exceeds upper")
        if self.n_coils < 1:
            raise ValueError("n_coils must be positive")

        if self.active_coil_indices is None:
            indices = tuple(range(self.n_coils))
        else:
            from numbers import Integral
            if any(not isinstance(index, Integral) or isinstance(index, bool)
                   for index in self.active_coil_indices):
                raise ValueError("active_coil_indices must contain integer indices")
            indices = tuple(int(index) for index in self.active_coil_indices)
            if not indices:
                raise ValueError("active_coil_indices must be non-empty")
            if len(set(indices)) != len(indices):
                raise ValueError("active_coil_indices contains duplicates")
            if min(indices) < 0 or max(indices) >= self.n_coils:
                raise ValueError("active_coil_indices out of range")
        object.__setattr__(self, "active_coil_indices", indices)

        lower, upper = self.current_bounds
        n_active = len(indices)
        task_lower = np.array([pair[0] for pair in self.task_box], dtype=float)
        task_upper = np.array([pair[1] for pair in self.task_box], dtype=float)
        self._derived.update(
            current_lower=np.full(n_active, float(lower)),
            current_upper=np.full(n_active, float(upper)),
            current_intervals=[interval[lower, upper]] * n_active,
            task_intervals=[interval[pair[0], pair[1]] for pair in self.task_box],
            task_vertices=box_vertices(task_lower, task_upper),
        )

    @property
    def current_lower(self) -> np.ndarray:
        return self._derived["current_lower"]

    @property
    def current_upper(self) -> np.ndarray:
        return self._derived["current_upper"]

    @property
    def current_intervals(self) -> list:
        """Coil limits as interval objects, for the ``Check_lib`` routines."""
        return self._derived["current_intervals"]

    @property
    def task_intervals(self) -> list:
        """Task box as interval objects, for the ``Check_lib`` routines."""
        return self._derived["task_intervals"]

    @property
    def task_vertices(self) -> np.ndarray:
        """Task box vertices, for the certificate routines."""
        return self._derived["task_vertices"]

    @property
    def n_active(self) -> int:
        return len(self.active_coil_indices)

    def select_columns(self, matrix):
        """Keep only active-coil columns of an actuation matrix."""

        if matrix.shape[1] != self.n_coils:
            raise ValueError(
                f"expected {self.n_coils} actuator columns, got {matrix.shape[1]}"
            )
        if self.active_coil_indices == tuple(range(self.n_coils)):
            return matrix
        return matrix[:, list(self.active_coil_indices)]

    def protocol(self) -> dict:
        """JSON-serialisable record of the query, for result provenance."""

        return {
            "initial_box_m": [list(pair) for pair in self.initial_box],
            "coil_model_z_m": self.z_offset,
            "task_box_T": [list(pair) for pair in self.task_box],
            "current_A": list(self.current_bounds),
            "n_coils": self.n_coils,
            "n_active": self.n_active,
            "active_coil_indices": list(self.active_coil_indices),
        }
