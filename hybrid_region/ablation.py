"""Three-way ablation used for the manuscript's Fig. 12 comparison.

Running the cascades in the order ``interval_original -> affine_original ->
affine_hybrid`` isolates two separate effects, because consecutive runs differ
in exactly one component:

* ``interval_original`` vs ``affine_original``: the enclosure only.
* ``affine_original`` vs ``affine_hybrid``: the added certificates only.
"""

from __future__ import annotations

from .cascade import MODES
from .model import RegionProblem
from .search import certify_region

__all__ = ["run_ablation"]


def run_ablation(
    problem: RegionProblem | None = None,
    eps: float = 0.002,
    *,
    modes: tuple[str, ...] = MODES,
    keep_boxes: bool = False,
) -> dict:
    """Run every cascade on one problem and report the relative reductions.

    Returns a JSON-serialisable dict holding each mode's summary plus the
    pairwise relative reduction in unresolved area and the hybrid runtime
    ratio.
    """

    problem = problem or RegionProblem()
    reports = {
        mode: certify_region(problem, mode=mode, eps=eps, keep_boxes=keep_boxes)
        for mode in modes
    }

    result = {
        "protocol": {**problem.protocol(), "eps_m": eps},
        **{mode: report.summary() for mode, report in reports.items()},
    }

    def reduction(from_mode: str, to_mode: str) -> float | None:
        if from_mode not in reports or to_mode not in reports:
            return None
        before = reports[from_mode].unresolved_fraction
        after = reports[to_mode].unresolved_fraction
        return 1.0 - after / before if before else None

    result["relative_unresolved_reduction"] = {
        "standard_to_affine": reduction("interval_original", "affine_original"),
        "affine_to_hybrid": reduction("affine_original", "affine_hybrid"),
        "standard_to_hybrid": reduction("interval_original", "affine_hybrid"),
    }

    if "affine_original" in reports and "affine_hybrid" in reports:
        baseline = reports["affine_original"]
        hybrid = reports["affine_hybrid"]
        result["affine_to_hybrid_runtime_ratio"] = (
            hybrid.elapsed_s / baseline.elapsed_s if baseline.elapsed_s else None
        )
        result["affine_to_hybrid_box_evaluation_relative_reduction"] = (
            1.0 - hybrid.iterations / baseline.iterations if baseline.iterations else None
        )

    return result
