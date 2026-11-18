"""Physics and certificate checks for Case 8."""

import math
import unittest
from unittest.mock import patch

import numpy as np

from emns.numerics.interval import interval
from emns.numerics.affine import Affine
from cases.case_7_vessel_microspheres.magnetics import coil_basis
from cases.case_8_two_sphere_control_hold.model import (
    B_HOLD,
    F_HOLD,
    F_MOVE,
    HELD_M,
    HELD_X,
    HELD_Y,
    MOVING_M,
    MU0_OVER_4PI,
    NUM_COILS,
    OUTPUT_LABELS,
    Z_PLANE,
    _negative_power,
    _product,
    affine_actuation_map,
    interval_map_bounds,
    point_actuation_map,
    separation_bounds,
    stable_alignment_radius,
    task_bounds,
    task_vertices,
)
from cases.case_8_two_sphere_control_hold.workspace import classify_box, determine_workspace
from cases.case_8_two_sphere_control_hold.model_aligned_bz import (
    affine_actuation_map as aligned_affine_actuation_map,
    aligned_moment,
    optimize_bz_pair,
    point_actuation_map as aligned_point_actuation_map,
)
from cases.case_8_two_sphere_control_hold.workspace_aligned_bz import (
    classify_box as classify_aligned_bz_box,
)
from emns.workspace.checks import Create2DBox


class Case8TwoSphereTests(unittest.TestCase):
    def test_small_distance_products_keep_nonlinear_error(self):
        distance = Affine.from_interval(interval[1e-8, 2e-8])
        bounds = _product(distance, distance).to_interval()
        self.assertLessEqual(bounds.lower, 1e-16)
        self.assertGreaterEqual(bounds.upper, 4e-16)

    def test_inverse_powers_preserve_dependency_enclosures(self):
        for scale in (1.0, 1e-2, 1e-6):
            value = Affine.from_interval(interval[scale, 2 * scale])
            symbol, coefficient = next(iter(value.deviations.items()))
            for exponent in (-1.0, -1.5, -2.5, -3.5):
                power = _negative_power(value, exponent)
                error = sum(abs(v) for k, v in power.deviations.items() if k != symbol)
                for noise in np.linspace(-1, 1, 21):
                    actual = (value.center + coefficient * noise) ** exponent
                    approximation = power.center + power.deviations[symbol] * noise
                    self.assertLessEqual(abs(actual - approximation), error * (1 + 1e-12))

    def test_task_has_ten_outputs_and_sixteen_force_vertices(self):
        vertices = task_vertices()
        self.assertEqual(len(OUTPUT_LABELS), 10)
        self.assertEqual(vertices.shape, (16, 10))
        np.testing.assert_array_equal(vertices[:, [0, 2, 5, 7]], 0.0)
        np.testing.assert_array_equal(vertices[:, [1, 6]], B_HOLD)
        self.assertEqual(len(np.unique(vertices, axis=0)), 16)
        lower, upper = task_bounds()
        np.testing.assert_allclose(lower[[3, 4, 8, 9]], [-F_HOLD, -F_HOLD, -F_MOVE, -F_MOVE])
        np.testing.assert_allclose(upper[[3, 4, 8, 9]], [F_HOLD, F_HOLD, F_MOVE, F_MOVE])

    def test_coil_rows_match_independent_full_gradient(self):
        p2 = np.array([-0.031, 0.027, Z_PLANE])
        G, _ = point_actuation_map(p2[0], p2[1])
        for offset, position, moment in (
            (0, np.array([HELD_X, HELD_Y, Z_PLANE]), HELD_M),
            (5, p2, MOVING_M),
        ):
            H, D = coil_basis(position)
            np.testing.assert_allclose(G[offset:offset + 3], H, rtol=2e-12, atol=1e-14)
            np.testing.assert_allclose(G[offset + 3], moment * D[1, 0], rtol=2e-12, atol=1e-13)
            np.testing.assert_allclose(G[offset + 4], moment * D[1, 1], rtol=2e-12, atol=1e-13)

    def test_mutual_offset_field_and_action_reaction(self):
        x2, y2 = -0.026, 0.019
        _, c = point_actuation_map(x2, y2)
        r = np.array([x2 - HELD_X, y2 - HELD_Y, 0.0])
        moment1 = np.array([0.0, HELD_M, 0.0])
        expected_B2 = MU0_OVER_4PI * (
            3.0 * r * np.dot(moment1, r) / np.linalg.norm(r) ** 5
            - moment1 / np.linalg.norm(r) ** 3
        )
        np.testing.assert_allclose(c[5:8], expected_B2, rtol=2e-12, atol=1e-14)
        np.testing.assert_allclose(c[3:5], -c[8:10], rtol=0, atol=1e-14)
        self.assertEqual(c[2], 0.0)
        self.assertEqual(c[7], 0.0)

    def test_affine_box_encloses_sampled_point_maps(self):
        x_box, y_box = interval[-0.031, -0.027], interval[0.021, 0.025]
        interval_map = affine_actuation_map(x_box, y_box)
        G_lower, G_upper, c_lower, c_upper = interval_map_bounds(interval_map)
        for x in np.linspace(x_box.lower, x_box.upper, 3):
            for y in np.linspace(y_box.lower, y_box.upper, 3):
                G, c = point_actuation_map(x, y)
                self.assertTrue(np.all(G >= G_lower))
                self.assertTrue(np.all(G <= G_upper))
                self.assertTrue(np.all(c >= c_lower))
                self.assertTrue(np.all(c <= c_upper))

    def test_alignment_radius_and_separation_geometry(self):
        expected = (
            2 * MU0_OVER_4PI * math.sqrt(HELD_M * MOVING_M) / B_HOLD
        ) ** (1 / 3)
        self.assertAlmostEqual(stable_alignment_radius(), expected, places=14)
        self.assertAlmostEqual(stable_alignment_radius(), 0.010626585691826113, places=14)
        minimum, maximum = separation_bounds(
            interval[0.03, 0.04], interval[0.01, 0.02]
        )
        self.assertAlmostEqual(minimum, math.sqrt(2) * 0.01)
        self.assertAlmostEqual(maximum, math.sqrt(8) * 0.01)

    def test_excluded_point_is_detected_before_dipole_evaluation(self):
        box = Create2DBox(
            interval[HELD_X, HELD_X], interval[HELD_Y, HELD_Y]
        )
        action, _, reason = classify_box(
            box,
            f_hold=F_HOLD,
            f_move=F_MOVE,
            physical_min_separation=0.0,
            eps=0.005,
        )
        self.assertEqual(action, "excluded")
        self.assertEqual(reason, "alignment_or_separation")

    def test_boundary_leaf_retains_admissible_positions_as_undetermined(self):
        radius = stable_alignment_radius()
        box = Create2DBox(
            interval[HELD_X + radius - 0.001, HELD_X + radius + 0.001],
            interval[HELD_Y, HELD_Y],
        )
        with patch(
            "cases.case_8_two_sphere_control_hold.workspace.affine_actuation_map",
            side_effect=AssertionError("mixed separation box must not be evaluated"),
        ):
            action, _, reason = classify_box(
                box, f_hold=F_HOLD, f_move=F_MOVE,
                physical_min_separation=0.0, eps=0.005,
            )
        self.assertEqual((action, reason), ("undetermined", "separation_boundary_leaf"))

    def test_small_inside_box_has_feasible_currents_at_all_sampled_vertices(self):
        x_box, y_box = interval[-0.04001, -0.03999], interval[-0.02001, -0.01999]
        action, _, _ = classify_box(
            Create2DBox(x_box, y_box), f_hold=0.002, f_move=0.01,
            physical_min_separation=0.0, eps=0.005,
        )
        self.assertEqual(action, "inside")
        vertices = task_vertices(0.002, 0.01)
        for x in (x_box.lower, x_box.upper):
            for y in (y_box.lower, y_box.upper):
                G, c = point_actuation_map(x, y)
                currents = np.linalg.solve(G, (vertices - c).T)
                self.assertLessEqual(np.max(np.abs(currents)), 15.0)
                np.testing.assert_allclose(G @ currents + c[:, None], vertices.T, atol=1e-12)

    def test_zero_force_ranges_do_not_duplicate_vertices(self):
        self.assertEqual(task_vertices(0, 0).shape, (1, 10))
        self.assertEqual(task_vertices(0, F_MOVE).shape, (4, 10))

    def test_invalid_options_fail_before_starting_workers(self):
        for options in (
            {"eps": 0}, {"eps": float("nan")}, {"f_hold": -1},
            {"f_move": float("inf")}, {"physical_min_separation": float("nan")},
            {"processes": 0}, {"processes": 1.5}, {"max_iter": -2},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                determine_workspace(**options)

    def test_zero_budget_leaves_entire_domain_pending(self):
        result = determine_workspace(max_iter=0)
        self.assertFalse(result["complete"])
        self.assertEqual(result["iterations"], 0)
        self.assertEqual(len(result["pending"]), 1)

    def test_serial_partial_run_preserves_domain_area(self):
        with patch(
            "cases.case_8_two_sphere_control_hold.workspace.mp.Pool",
            side_effect=AssertionError("serial runs should not spawn workers"),
        ):
            result = determine_workspace(processes=1, max_iter=2)
        self.assertEqual(result["iterations"], 2)
        self.assertFalse(result["complete"])
        area = sum(
            (box[0][0].upper - box[0][0].lower) * (box[1][0].upper - box[1][0].lower)
            for category in ("inside", "outside", "undetermined", "excluded", "pending")
            for box in result[category]
        )
        self.assertAlmostEqual(area, 0.1 ** 2)

    def test_representative_point_is_certified_outside_at_default_forces(self):
        box = Create2DBox(interval[-0.03, -0.03], interval[0.02, 0.02])
        action, _, reason = classify_box(
            box,
            f_hold=F_HOLD,
            f_move=F_MOVE,
            physical_min_separation=0.0,
            eps=0.005,
        )
        self.assertEqual(action, "outside")
        self.assertEqual(reason, "dual_separation")

    def test_inside_certificate_path_with_reduced_trial_forces(self):
        box = Create2DBox(interval[-0.04, -0.04], interval[-0.02, -0.02])
        action, _, reason = classify_box(
            box,
            f_hold=0.002,
            f_move=0.01,
            physical_min_separation=0.0,
            eps=0.005,
        )
        self.assertEqual(action, "inside")
        self.assertEqual(reason, "affine_right_inverse")

    def test_aligned_bz_numeric_and_affine_maps_match(self):
        x, y, bz1, bz2 = -0.031, 0.019, 0.006, -0.004
        G, c = aligned_point_actuation_map(x, y, bz1, bz2)
        full = aligned_affine_actuation_map(
            interval[x, x], interval[y, y], bz1, bz2, return_affine=True
        )
        affine_G = np.asarray([
            [value.center for value in row[:NUM_COILS]] for row in full
        ])
        affine_c = np.asarray([row[NUM_COILS].center for row in full])
        np.testing.assert_allclose(G, affine_G, rtol=2e-12, atol=1e-13)
        np.testing.assert_allclose(c, affine_c, rtol=2e-12, atol=1e-13)
        moment = aligned_moment(MOVING_M, bz2)
        self.assertAlmostEqual(moment[2] / moment[1], bz2 / B_HOLD)

    def test_aligned_bz_optimization_adds_a_feasible_point(self):
        pair, peak = optimize_bz_pair(-0.04, 0.005, bz_limit=0.01)
        self.assertLessEqual(np.max(np.abs(pair)), 0.01)
        self.assertLess(peak, 15.0)
        box = Create2DBox(interval[-0.04, -0.04], interval[0.005, 0.005])
        action, _, reason, witness = classify_aligned_bz_box(
            box,
            f_hold=0.001,
            f_move=0.005,
            bz_limit=0.01,
            physical_min_separation=0.0,
            eps=0.001,
        )
        self.assertEqual((action, reason), ("inside", "affine_aligned_bz_witness"))
        self.assertLessEqual(max(abs(value) for value in witness), 0.01)


if __name__ == "__main__":
    unittest.main()
