"""Quick checks for the revised case-study code.

Run from the repository root:
    python -B -m unittest revised_case_studies.tests.test_revised_case_studies
"""

import math
import unittest

import numpy as np


class AllocatorTest(unittest.TestCase):
    def test_zero_slack_when_reachable(self):
        from revised_case_studies.shared.allocator import QPAllocator
        rng = np.random.default_rng(0)
        G = rng.normal(size=(4, 10))
        i_true = rng.uniform(-5, 5, 10)
        c = rng.normal(size=4)
        sol = QPAllocator(4).solve(G, c, G @ i_true + c)
        self.assertLess(np.max(np.abs(sol['slack'])), 1e-6)

    def test_slack_reported_when_unreachable(self):
        from revised_case_studies.shared.allocator import QPAllocator
        G = np.ones((1, 10))
        sol = QPAllocator(1).solve(G, None, np.array([1000.0]))
        self.assertGreater(sol['slack'][0], 800)
        self.assertTrue(sol['saturated'])


class CaseStudy1Test(unittest.TestCase):
    def test_labels_on_naive_path(self):
        from revised_case_studies.cs1_cylinder_interaction import common as c
        self.assertEqual(c.label(0.01, -0.05, math.pi / 2), 'inside')
        self.assertEqual(c.label(0.01, 0.0, math.pi / 2), 'infeasible')

    def test_force_range_meets_task_inside(self):
        from revised_case_studies.cs1_cylinder_interaction import common as c
        lo, hi = c.force_ranges(0.01, -0.05, math.pi / 2)['fz_range']
        self.assertLessEqual(lo, -c.F_MAX)
        self.assertGreaterEqual(hi, c.F_MAX)


class CaseStudy2Test(unittest.TestCase):
    def test_allocation_rows_match_plant(self):
        """With moments along the prescribed directions, G I + c equals the plant."""
        from revised_case_studies.cs2_vessel_microspheres import model as m
        p = np.array([[0.0, -0.01, 0.085], [0.0, -0.02, 0.085]])
        u = np.array([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
        frames = [tuple(np.eye(3))] * 2
        G, c = m.allocation_rows(p, u, frames)
        I = np.linalg.pinv(G[:6]) @ (np.array([0.005, 0, 0, 0.005, 0, 0]) - c[:6])
        mom, B, F = m.plant(p, I)
        B_frame = np.concatenate([m.field_frame(u[i]) @ B[i] for i in range(2)])
        np.testing.assert_allclose(B_frame, (G @ I + c)[:6], rtol=1e-3, atol=1e-7)
        np.testing.assert_allclose(F.ravel(), (G @ I + c)[6:], rtol=1e-3, atol=1e-9)

    def test_dipole_gradient_finite_difference(self):
        from revised_case_studies.cs2_vessel_microspheres.model import dipole_field, dipole_gradient
        r = np.array([0.004, -0.002, 0.003])
        mvec = np.array([1e-4, 3e-4, -2e-4])
        G = dipole_gradient(r, mvec)
        h = 1e-7
        for b in range(3):
            e = np.zeros(3)
            e[b] = h
            fd = (dipole_field(r + e, mvec) - dipole_field(r - e, mvec)) / (2 * h)
            np.testing.assert_allclose(G[:, b], fd, rtol=1e-5, atol=1e-12)


class AllocatorScalingTest(unittest.TestCase):
    def test_tiny_output_scale_is_resolved(self):
        """Sub-micronewton rows must be met, not lost below solver tolerances."""
        from revised_case_studies.shared.allocator import QPAllocator
        rng = np.random.default_rng(1)
        G = rng.normal(size=(4, 10)) * 1e-6
        i_true = rng.uniform(-5, 5, 10)
        h = G @ i_true
        sol = QPAllocator(4).solve(G, None, h, w=np.full(4, 1e7))
        np.testing.assert_allclose(sol['achieved'], h, rtol=1e-3, atol=1e-10)


class CaseStudy2TFWTest(unittest.TestCase):
    def test_route_enclosure_contains_centerline(self):
        """Affine route enclosure (incl. the Case 7 winding term) contains the curve."""
        from revised_case_studies.cs2_vessel_microspheres import tfw
        tfw._init('branched_over_under')
        for line in tfw._LINES:
            for lo, hi in ((0.05, 0.12), (0.4, 0.47), (0.8, 0.9)):
                iv = [f.to_interval() for f in line.affine_point(lo, hi)]
                pts = np.array([line.point(s) for s in np.linspace(lo, hi, 60)])
                for k in range(3):
                    self.assertGreaterEqual(pts[:, k].min(), iv[k].lower - 1e-12)
                    self.assertLessEqual(pts[:, k].max(), iv[k].upper + 1e-12)

    def test_certified_box_is_pointwise_feasible(self):
        from revised_case_studies.cs2_vessel_microspheres import tfw
        # 3 mm N35 spheres: a box far from the close pass, in the publication map.
        tfw._init('branched_side_by_side')
        box = ((0.0, 1 / 64), (0.0, 1 / 64))
        self.assertEqual(tfw.classify(box, 1 / 64)[0], 'inside')
        self.assertTrue(tfw.point_feasible(tfw._LINES, 0.008, 0.008))


class CorridorPlannerTest(unittest.TestCase):
    def test_bezier_stays_in_corridor_and_limits(self):
        from revised_case_studies.shared import corridor_planner as cpl
        free = np.ones((40, 40), bool)
        free[10:30, 12:28] = False                       # an obstacle to go around
        res = cpl.plan(free, (2, 2), (37, 37), (1.0, 1.0), (0.0, 0.0),
                       vmax=(1.0, 1.0), amax=(1.0, 1.0), dt=0.02)
        self.assertIsNotNone(res)
        xy = res['xy']
        inside_any = [any(b[0] - 1e-6 <= x <= b[1] + 1e-6 and b[2] - 1e-6 <= y <= b[3] + 1e-6
                          for b in res['boxes_xy']) for x, y in xy]
        self.assertTrue(all(inside_any))
        for x, y in xy:                                  # never inside the obstacle
            self.assertFalse(12 < x < 28 and 10 < y < 30)
        v = np.abs(np.diff(xy, axis=0) / np.diff(res['t'])[:, None])
        self.assertLessEqual(v.max(), 1.0 + 1e-3)


class CaseStudy3Test(unittest.TestCase):
    def test_field_matrix_matches_emns(self):
        from emns.workspace.fast import Map_I2H
        from revised_case_studies.cs3_helical_robots import dynamics as d, geometry as g
        a, roll = np.radians([170.0, 10.0]), [0.3, 1.2]
        G, c = d.field_matrix(a, roll)
        pts = [{'X': g.position(i, a[i])[0], 'Y': g.position(i, a[i])[1],
                'Z': g.position(i, a[i])[2], 'm': 0.023, 'alpha': roll[i],
                'beta': a[i] + math.pi / 2, 'Bx': True, 'By': True, 'Bz': True}
               for i in range(2)]
        A = Map_I2H(pts)
        np.testing.assert_allclose(G, A[:, :10], rtol=1e-9, atol=1e-12)
        np.testing.assert_allclose(c, A[:, 10], rtol=1e-5, atol=1e-10)

    def test_certified_configuration_is_exact_for_all_phases(self):
        from revised_case_studies.cs3_helical_robots.field_check import error_map, pick_configs
        _, err = error_map(np.array(pick_configs()['inside']), n=6)
        self.assertLess(err.max(), 1e-9)


if __name__ == '__main__':
    unittest.main()
