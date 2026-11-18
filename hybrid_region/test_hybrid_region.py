"""Self-tests for the hybrid_region package.
# Run from the repository root: python -m hybrid_region.test_hybrid_region

Matches the project convention in ``tests/test_tight_region_certificates.py``:
plain ``def test_*`` functions with bare ``assert``, so ``pytest`` collects
them when it is installed.  Because the bundled virtual environment currently
has no pytest, the module also runs standalone::

    python -m hybrid_region.test_hybrid_region

These cover the package's own logic -- geometry, problem setup, cascade
ordering, and the coil-model synchronisation the enclosure comparison depends
on.  The certificate proofs themselves are covered by
``tests/test_tight_region_certificates.py``.
"""

from __future__ import annotations

import numpy as np

from hybrid_region import (
    MODES,
    RegionProblem,
    bisect_box,
    box_area,
    box_bounds,
    box_widths,
    certify_region,
    classify_box,
    make_box,
)
from hybrid_region._backend import COIL_MODEL_ORIGIN_OFFSET_M, Map_I2H_Affine, interval
from hybrid_region.model import _target_points, affine_matrix, interval_matrix


def _close(a, b, tol=1e-12):
    return abs(float(a) - float(b)) <= tol


def _raises(exception, message_fragment, callable_, *args, **kwargs):
    try:
        callable_(*args, **kwargs)
    except exception as error:
        assert message_fragment in str(error), f"unexpected message: {error}"
        return
    raise AssertionError(
        f"expected {exception.__name__} mentioning {message_fragment!r}"
    )


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------


def test_make_box_roundtrip():
    box = make_box((-0.05, 0.05), (-0.02, 0.02))
    lower, upper = box_bounds(box)
    assert lower.tolist() == [-0.05, -0.02]
    assert upper.tolist() == [0.05, 0.02]
    assert all(_close(a, b) for a, b in zip(box_widths(box), (0.1, 0.04)))
    assert _close(box_area(box), 0.004)


def test_make_box_accepts_interval_objects():
    box = make_box(interval[-0.05, 0.05], interval[-0.05, 0.05])
    lower, upper = box_bounds(box)
    assert lower.tolist() == [-0.05, -0.05]
    assert upper.tolist() == [0.05, 0.05]


def test_make_box_rejects_inverted_axis():
    _raises(ValueError, "exceeds upper", make_box, (0.05, -0.05), (0.0, 1.0))


def test_bisect_box_partitions_area_exactly():
    box = make_box((-0.05, 0.05), (-0.05, 0.05))
    children = bisect_box(box)
    assert len(children) == 4
    assert _close(sum(box_area(child) for child in children), box_area(box))


def test_bisect_box_generalises_to_three_dimensions():
    box = make_box((0.0, 1.0), (0.0, 1.0), (0.0, 1.0))
    children = bisect_box(box)
    assert len(children) == 8
    assert _close(sum(box_area(child) for child in children), 1.0)


# --------------------------------------------------------------------------
# Coil model
# --------------------------------------------------------------------------


def test_coil_models_are_synchronised():
    """The two backends must agree on the coil geometry.

    Without the synchronisation performed in ``_backend``, the interval and
    affine enclosures describe systems whose coils differ by 0.085 m in z, and
    the enclosure ablation then compares two different machines.  The failure
    is silent rather than conservative: the unsynchronised interval cascade
    certifies the entire search region as infeasible.
    """

    from emns.workspace import affine as WS_lib_Affine
    from emns.workspace import interval_fast as WS_lib_Interval_fast

    affine = np.asarray(WS_lib_Affine.params_list, dtype=float)
    interval_params = np.asarray(WS_lib_Interval_fast.params_list, dtype=float)
    assert np.array_equal(affine, interval_params)
    assert _close(COIL_MODEL_ORIGIN_OFFSET_M, 0.085)


def test_affine_backend_trailing_column_is_dropped():
    """The affine backend returns an extra column the interval one does not."""

    x, y = interval[-0.05, 0.05], interval[-0.05, 0.05]
    raw = Map_I2H_Affine(_target_points(x, y, 0.075))
    assert np.shape(raw) == (2, 11)
    assert np.shape(affine_matrix(x, y, 0.075)) == (2, 10)
    assert np.shape(interval_matrix(x, y, 0.075)) == (2, 10)


# --------------------------------------------------------------------------
# Problem specification
# --------------------------------------------------------------------------


def test_problem_derived_quantities():
    problem = RegionProblem()
    assert problem.current_lower.shape == (10,)
    assert problem.current_upper.tolist() == [15.0] * 10
    assert len(problem.current_intervals) == 10
    assert problem.active_coil_indices == tuple(range(10))
    # 2 field components -> 4 task vertices
    assert problem.task_vertices.shape == (4, 2)
    assert problem.protocol()["task_box_T"] == [[-0.12, 0.12], [-0.04, 0.04]]


def test_active_coil_indices_reduce_current_dimension():
    problem = RegionProblem(active_coil_indices=(0, 1, 2, 3, 4, 5))
    assert problem.n_active == 6
    assert problem.current_lower.shape == (6,)
    x, y = interval[-0.01, 0.01], interval[-0.01, 0.01]
    selected = problem.select_columns(affine_matrix(x, y, 0.075))
    assert np.shape(selected) == (2, 6)


def test_problem_rejects_bad_task_box():
    _raises(
        ValueError,
        "task_box",
        RegionProblem,
        task_box=((0.12, -0.12), (-0.04, 0.04)),
    )


def test_problem_rejects_bad_current_bounds():
    _raises(
        ValueError, "current_bounds", RegionProblem, current_bounds=(15.0, -15.0)
    )


# --------------------------------------------------------------------------
# Cascade
# --------------------------------------------------------------------------


def test_modes_registered():
    assert MODES == ("interval_original", "affine_original", "affine_hybrid")


def test_unknown_mode_rejected():
    problem = RegionProblem()
    box = make_box(*problem.initial_box)
    _raises(ValueError, "unknown mode", classify_box, box, problem, "not_a_mode")


def test_hybrid_never_weaker_than_affine_baseline():
    """The hybrid cascade only adds tests, so it can never lose a box.

    Its added certificates are consulted only after the original Rohn and
    contractor tests decline, so any verdict the affine baseline reaches must
    be reproduced identically by the hybrid cascade.
    """

    problem = RegionProblem()
    boxes = [make_box(*problem.initial_box)]
    boxes += bisect_box(boxes[0])
    for box in boxes:
        baseline = classify_box(box, problem, "affine_original")
        hybrid = classify_box(box, problem, "affine_hybrid")
        if baseline.is_terminal:
            assert hybrid.classification == baseline.classification
        if hybrid.classification == "undetermined":
            assert not baseline.is_terminal


def test_baseline_modes_only_use_original_certificates():
    """Only the hybrid cascade may report the added certificate names."""

    added = {"affine_right_inverse", "right_inverse", "dual_separation"}
    problem = RegionProblem()
    for mode in ("interval_original", "affine_original"):
        report = certify_region(problem, mode=mode, eps=0.02, keep_boxes=False)
        assert not (set(report.certificate_counts) & added)


# --------------------------------------------------------------------------
# Search
# --------------------------------------------------------------------------


def test_report_is_consistent():
    """Areas, counts, and fractions must describe the same partition."""

    problem = RegionProblem()
    for mode in MODES:
        report = certify_region(problem, mode=mode, eps=0.02)
        assert sum(report.counts.values()) == sum(
            len(v) for v in report.boxes.values()
        )
        assert _close(sum(report.area_fractions.values()), 1.0)
        # The terminal boxes must exactly tile the 0.1 x 0.1 m initial box.
        assert _close(sum(report.areas.values()), 0.01)
        # Every closed box was closed by exactly one named test.
        assert sum(report.certificate_counts.values()) == (
            report.counts["inside"] + report.counts["outside"]
        )
        assert set(report.summary()) >= {"mode", "eps_m", "area_fractions", "protocol"}


def test_eps_must_be_positive():
    _raises(
        ValueError, "eps must be positive", certify_region, RegionProblem(), eps=0.0
    )


def test_keep_boxes_false_drops_geometry():
    report = certify_region(
        RegionProblem(), mode="affine_original", eps=0.02, keep_boxes=False
    )
    assert report.boxes == {}
    assert report.counts["undetermined"] >= 0


def test_hybrid_resolves_at_least_as_much_area():
    """End-to-end monotonicity of the ablation."""

    problem = RegionProblem()
    baseline = certify_region(
        problem, mode="affine_original", eps=0.0125, keep_boxes=False
    )
    hybrid = certify_region(
        problem, mode="affine_hybrid", eps=0.0125, keep_boxes=False
    )
    assert hybrid.unresolved_fraction <= baseline.unresolved_fraction + 1e-12


if __name__ == "__main__":
    import traceback

    tests = [
        (name, value)
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    failures = 0
    for name, test in tests:
        try:
            test()
        except Exception:
            failures += 1
            print(f"FAIL {name}")
            traceback.print_exc()
        else:
            print(f"ok   {name}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    raise SystemExit(1 if failures else 0)
