"""Physics, geometry, and certificate checks for Case 9."""

import math
import unittest
from unittest.mock import patch

import numpy as np

from cases.case_7_vessel_microspheres.magnetics import coil_basis
from cases.case_8_two_sphere_control_hold.model import stable_alignment_radius
from cases.case_9_two_soft_spheres_control_hold.model import (
    B_HOLD,
    F_HOLD,
    F_MOVE,
    HELD_RADIUS,
    HELD_X,
    HELD_Y,
    MOVING_RADIUS,
    MU0,
    SATURATION_MAGNETIZATION,
    SUSCEPTIBILITY,
    Z_PLANE,
    affine_actuation_map,
    effective_sphere_susceptibility,
    interval_map_bounds,
    operating_moments,
    point_actuation_map,
    soft_sphere_moment,
    task_vertices,
)
from cases.case_9_two_soft_spheres_control_hold.workspace import (
    classify_box,
    determine_workspace,
)
from emns.numerics.interval import interval
from emns.workspace.checks import Create2DBox


class Case9TwoSoftSphereTests(unittest.TestCase):
    def test_demagnetization_corrected_moment(self):
        chi_eff = 3.0 * SUSCEPTIBILITY / (SUSCEPTIBILITY + 3.0)
        self.assertAlmostEqual(
            effective_sphere_susceptibility(SUSCEPTIBILITY), chi_eff
        )
        volume = 4.0 * math.pi * HELD_RADIUS ** 3 / 3.0
        expected = volume * chi_eff * B_HOLD / MU0
        self.assertAlmostEqual(soft_sphere_moment(HELD_RADIUS), expected)
        self.assertAlmostEqual(expected, 0.02498500899460324)

    def test_material_law_clips_at_saturation(self):
        radius = 0.001
        volume = 4.0 * math.pi * radius ** 3 / 3.0
        moment = soft_sphere_moment(
            radius,
            bias_field=10.0,
            susceptibility=SUSCEPTIBILITY,
            saturation_magnetization=SATURATION_MAGNETIZATION,
        )
        self.assertAlmostEqual(moment, volume * SATURATION_MAGNETIZATION)

    def test_force_rows_use_operating_moments(self):
        p2 = np.array([-0.031, 0.027, Z_PLANE])
        held_m, moving_m = operating_moments()
        G, _ = point_actuation_map(p2[0], p2[1])
        for offset, position, moment in (
            (0, np.array([HELD_X, HELD_Y, Z_PLANE]), held_m),
            (5, p2, moving_m),
        ):
            H, D = coil_basis(position)
            np.testing.assert_allclose(G[offset:offset + 3], H, rtol=2e-12, atol=1e-14)
            np.testing.assert_allclose(G[offset + 3], moment * D[1, 0], rtol=2e-12, atol=1e-13)
            np.testing.assert_allclose(G[offset + 4], moment * D[1, 1], rtol=2e-12, atol=1e-13)

    def test_mutual_force_obeys_action_reaction(self):
        _, c = point_actuation_map(-0.026, 0.019)
        np.testing.assert_allclose(c[3:5], -c[8:10], rtol=0, atol=1e-14)
        self.assertEqual(c[2], 0.0)
        self.assertEqual(c[7], 0.0)

    def test_affine_box_encloses_point_maps(self):
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

    def test_task_has_sixteen_exact_force_vertices(self):
        vertices = task_vertices()
        self.assertEqual(vertices.shape, (16, 10))
        np.testing.assert_array_equal(vertices[:, [0, 2, 5, 7]], 0.0)
        np.testing.assert_array_equal(vertices[:, [1, 6]], B_HOLD)

    def test_overlap_is_excluded(self):
        box = Create2DBox(interval[HELD_X, HELD_X], interval[HELD_Y, HELD_Y])
        action, _, reason = classify_box(
            box,
            f_hold=F_HOLD,
            f_move=F_MOVE,
            held_radius=HELD_RADIUS,
            moving_radius=MOVING_RADIUS,
            susceptibility=SUSCEPTIBILITY,
            saturation_magnetization=SATURATION_MAGNETIZATION,
            clearance=0.0,
            eps=0.005,
        )
        self.assertEqual((action, reason), ("excluded", "physical_separation"))

    def test_no_permanent_magnet_alignment_exclusion(self):
        distance = (HELD_RADIUS + MOVING_RADIUS + stable_alignment_radius()) / 2.0
        self.assertLess(distance, stable_alignment_radius())
        box = Create2DBox(
            interval[HELD_X + distance, HELD_X + distance],
            interval[HELD_Y, HELD_Y],
        )
        action, _, _ = classify_box(
            box,
            f_hold=F_HOLD,
            f_move=F_MOVE,
            held_radius=HELD_RADIUS,
            moving_radius=MOVING_RADIUS,
            susceptibility=SUSCEPTIBILITY,
            saturation_magnetization=SATURATION_MAGNETIZATION,
            clearance=0.0,
            eps=0.005,
        )
        self.assertNotEqual(action, "excluded")

    def test_serial_partial_run_preserves_domain_area(self):
        with patch(
            "cases.case_9_two_soft_spheres_control_hold.workspace.mp.Pool",
            side_effect=AssertionError("serial runs should not spawn workers"),
        ):
            result = determine_workspace(processes=1, max_iter=2)
        area = sum(
            (box[0][0].upper - box[0][0].lower)
            * (box[1][0].upper - box[1][0].lower)
            for category in ("inside", "outside", "undetermined", "excluded", "pending")
            for box in result[category]
        )
        self.assertAlmostEqual(area, 0.1 ** 2)

    def test_invalid_material_parameters_fail_before_workers(self):
        for options in (
            {"held_radius": 0.0},
            {"moving_radius": -1.0},
            {"susceptibility": -1.0},
            {"saturation_magnetization": 0.0},
            {"clearance": -1.0},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                determine_workspace(**options)


if __name__ == "__main__":
    unittest.main()
