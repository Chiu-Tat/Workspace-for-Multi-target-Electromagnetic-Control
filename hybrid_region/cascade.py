"""Decision cascades that classify a single configuration box.

Three cascades are provided so the enclosure and the certificate contributions
can be attributed separately:

``interval_original``
    standard interval enclosure -> Rohn -> contractor -> split
``affine_original``
    affine enclosure -> Rohn -> contractor -> split
``affine_hybrid``
    the affine enclosure of ``affine_original``, plus the constructive recourse
    and dual-separation certificates -> split

``affine_original`` and ``affine_hybrid`` share an identical enclosure, so any
difference between them is attributable to the added certificates alone.

Each cascade returns a :class:`BoxVerdict`.  ``"inside"`` and ``"outside"`` are
mathematical certificates for the whole box; ``"undetermined"`` only means no
test succeeded, and the caller should bisect.
"""

from __future__ import annotations

from dataclasses import dataclass

from ._backend import Out, RobustFeasible
from .certificates import (
    affine_constructive_inside_certificate,
    constructive_inside_certificate,
    dual_outside_certificate,
    interval_object_bounds,
)
from .model import RegionProblem, affine_matrix, interval_matrix

__all__ = ["MODES", "BoxVerdict", "classify_box"]

MODES = ("interval_original", "affine_original", "affine_hybrid")


@dataclass(frozen=True)
class BoxVerdict:
    """Outcome of one box classification."""

    classification: str  # "inside" | "outside" | "undetermined"
    reason: str  # which test closed the box, or "none"

    @property
    def is_terminal(self) -> bool:
        return self.classification != "undetermined"


def classify_box(box, problem: RegionProblem, mode: str) -> BoxVerdict:
    """Classify one configuration box under the requested cascade."""

    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")

    x_interval = box[0][0]
    y_interval = box[1][0]

    if mode == "interval_original":
        enclosure = interval_matrix(x_interval, y_interval, problem.z_offset)
    else:
        enclosure = affine_matrix(x_interval, y_interval, problem.z_offset)
    enclosure = problem.select_columns(enclosure)

    # --- Original inside test -------------------------------------------
    if RobustFeasible(enclosure, problem.current_intervals, problem.task_intervals):
        return BoxVerdict("inside", "rohn")

    G_lower = None
    G_upper = None

    # --- Added inside tests (hybrid only) --------------------------------
    if mode == "affine_hybrid":
        G_affine = problem.select_columns(
            affine_matrix(
                x_interval, y_interval, problem.z_offset, return_affine=True
            )
        )
        certificate = affine_constructive_inside_certificate(
            G_affine,
            problem.task_vertices,
            problem.current_lower,
            problem.current_upper,
        )
        if certificate.certified:
            return BoxVerdict("inside", "affine_right_inverse")

        G_lower, G_upper = interval_object_bounds(enclosure)
        certificate = constructive_inside_certificate(
            G_lower,
            G_upper,
            problem.task_vertices,
            problem.current_lower,
            problem.current_upper,
        )
        if certificate.certified:
            return BoxVerdict("inside", "right_inverse")

    # --- Original outside test -------------------------------------------
    if Out(enclosure, problem.current_intervals, problem.task_intervals):
        return BoxVerdict("outside", "contractor")

    # --- Added outside test (hybrid only) --------------------------------
    if mode == "affine_hybrid":
        if G_lower is None:
            G_lower, G_upper = interval_object_bounds(enclosure)
        certificate = dual_outside_certificate(
            G_lower,
            G_upper,
            problem.task_vertices,
            problem.current_lower,
            problem.current_upper,
        )
        if certificate.certified:
            return BoxVerdict("outside", "dual_separation")

    return BoxVerdict("undetermined", "none")
