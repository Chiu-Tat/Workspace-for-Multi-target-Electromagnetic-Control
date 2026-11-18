"""Branch-and-bound driver over configuration space.

``certify_region`` runs the requested cascade breadth-first, bisecting every
box no test could close until the box is narrower than ``eps`` on some axis.
The returned :class:`RegionReport` carries areas as well as box counts, because
terminal boxes need not be the same size once a non-uniform stopping rule or a
non-square initial box is used.
"""

from __future__ import annotations

import time
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Callable

from .boxes import bisect_box, box_area, box_widths, make_box
from .cascade import classify_box
from .model import RegionProblem

__all__ = ["RegionReport", "certify_region"]


@dataclass
class RegionReport:
    """Outcome of a certified-region search."""

    mode: str
    eps_m: float
    elapsed_s: float
    iterations: int
    areas: dict
    area_fractions: dict
    counts: dict
    certificate_counts: dict
    protocol: dict
    boxes: dict = field(repr=False, default_factory=dict)

    @property
    def unresolved_fraction(self) -> float:
        """Fraction of the searched measure left undetermined."""

        return self.area_fractions["undetermined"]

    def summary(self) -> dict:
        """JSON-serialisable view, dropping the retained box geometry."""

        return {
            "mode": self.mode,
            "eps_m": self.eps_m,
            "elapsed_s": self.elapsed_s,
            "iterations": self.iterations,
            "counts": dict(self.counts),
            "areas_m2": dict(self.areas),
            "area_fractions": dict(self.area_fractions),
            "certificate_counts": dict(self.certificate_counts),
            "protocol": dict(self.protocol),
        }


def certify_region(
    problem: RegionProblem,
    mode: str = "affine_hybrid",
    eps: float = 0.002,
    *,
    keep_boxes: bool = True,
    progress: Callable[[int, int], None] | None = None,
) -> RegionReport:
    """Classify the configuration space of ``problem`` under ``mode``.

    Parameters
    ----------
    eps:
        Bisection tolerance in metres.  A box is only split while *every* axis
        is wider than ``eps``, matching the archived benchmark's stopping rule.
    keep_boxes:
        Retain terminal box geometry on the report.  Set ``False`` for long
        runs where only the aggregate numbers are needed.
    progress:
        Optional callback receiving ``(iterations, pending)`` after each box.
    """

    if eps <= 0:
        raise ValueError("eps must be positive")

    pending = deque([make_box(*problem.initial_box)])
    terminal = {"inside": [], "outside": [], "undetermined": []}
    reason_counts: Counter = Counter()
    iterations = 0

    start = time.perf_counter()
    while pending:
        box = pending.popleft()
        verdict = classify_box(box, problem, mode)
        iterations += 1

        if verdict.is_terminal:
            terminal[verdict.classification].append(box)
            reason_counts[verdict.reason] += 1
        elif all(width > eps for width in box_widths(box)):
            pending.extend(bisect_box(box))
        else:
            terminal["undetermined"].append(box)

        if progress is not None:
            progress(iterations, len(pending))
    elapsed = time.perf_counter() - start

    areas = {
        key: float(sum(box_area(box) for box in boxes))
        for key, boxes in terminal.items()
    }
    total_area = sum(areas.values())
    if total_area <= 0:
        raise RuntimeError("search produced zero total area")

    return RegionReport(
        mode=mode,
        eps_m=eps,
        elapsed_s=elapsed,
        iterations=iterations,
        areas=areas,
        area_fractions={key: value / total_area for key, value in areas.items()},
        counts={key: len(value) for key, value in terminal.items()},
        certificate_counts=dict(reason_counts),
        protocol=problem.protocol(),
        boxes=terminal if keep_boxes else {},
    )
