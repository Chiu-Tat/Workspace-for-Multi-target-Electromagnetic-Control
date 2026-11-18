"""Axis-aligned configuration boxes for the branch-and-bound search.

A box is the project's existing nested-list form, ``Box[i][0]`` holding the
``custom_interval`` object for axis ``i``.  That layout is kept verbatim so
boxes produced here can be handed to ``Check_lib`` and ``Affine_10c`` routines
without conversion.  The helpers below generalise the project's 2-D-only
``Create2DBox``/``BisectBox`` pair to arbitrary dimension.
"""

from __future__ import annotations

from itertools import product
from typing import Iterable, Sequence

import numpy as np

from ._backend import interval

__all__ = [
    "make_box",
    "bisect_box",
    "box_area",
    "box_bounds",
    "box_widths",
    "box_vertices",
]


def make_box(*ranges: Sequence[float]):
    """Build a box from ``(lower, upper)`` pairs or existing interval objects.

    >>> box = make_box((-0.05, 0.05), (-0.05, 0.05))
    >>> box_widths(box)
    (0.1, 0.1)
    """

    if not ranges:
        raise ValueError("a box needs at least one axis")

    box = []
    for axis, item in enumerate(ranges):
        if hasattr(item, "lower") and hasattr(item, "upper"):
            lower, upper = float(item.lower), float(item.upper)
        else:
            pair = tuple(item)
            if len(pair) != 2:
                raise ValueError(
                    f"axis {axis} must be a (lower, upper) pair, got {pair!r}"
                )
            lower, upper = float(pair[0]), float(pair[1])
        if lower > upper:
            raise ValueError(f"axis {axis}: lower {lower} exceeds upper {upper}")
        box.append([interval[lower, upper]])
    return box


def box_bounds(box) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(lower, upper)`` float arrays for every axis of ``box``."""

    lower = np.array([float(box[i][0].lower) for i in range(len(box))])
    upper = np.array([float(box[i][0].upper) for i in range(len(box))])
    return lower, upper


def box_widths(box) -> tuple[float, ...]:
    """Return the per-axis width of ``box``."""

    lower, upper = box_bounds(box)
    return tuple(float(value) for value in (upper - lower))


def box_area(box) -> float:
    """Return the Lebesgue measure (area in 2-D, volume in 3-D) of ``box``."""

    lower, upper = box_bounds(box)
    return float(np.prod(upper - lower))


def bisect_box(box) -> list:
    """Split ``box`` at its midpoint along every axis.

    Returns ``2**d`` sub-boxes.  This reproduces the project's ``BisectBox``
    for ``d == 2`` while working for any dimension.
    """

    dimension = len(box)
    halves = []
    for axis in range(dimension):
        lower = float(box[axis][0].lower)
        upper = float(box[axis][0].upper)
        middle = (lower + upper) / 2.0
        halves.append((interval[lower, middle], interval[middle, upper]))

    children = []
    for selector in product((0, 1), repeat=dimension):
        children.append([[halves[axis][selector[axis]]] for axis in range(dimension)])
    return children


def box_vertices(lower: Iterable[float], upper: Iterable[float]) -> np.ndarray:
    """Return the ``(2**d, d)`` vertex array of an axis-aligned box.

    Thin re-export of the certificate module's helper so callers building a
    task box do not need to import two modules.
    """

    from .certificates import box_vertices as _box_vertices

    return _box_vertices(lower, upper)
