import unittest

import numpy as np

from cases.case_8_two_sphere_control_hold.simulation.model import (
    MAGNET1_HOME,
    MAGNET2_HOME,
    MAGNET2_START,
    STAGE1_JSON,
    STAGE2_JSON,
    CertifiedSlice,
    solve_control,
)


class Case8SequentialSimulationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.stage1 = CertifiedSlice(STAGE1_JSON)
        cls.stage2 = CertifiedSlice(STAGE2_JSON)

    def test_handover_endpoints_are_certified(self):
        self.assertTrue(self.stage1.contains(MAGNET2_HOME))
        self.assertTrue(self.stage2.contains(MAGNET1_HOME))
        np.testing.assert_allclose(self.stage1.held, MAGNET1_HOME)
        np.testing.assert_allclose(self.stage2.held, MAGNET2_HOME)

    def test_return_paths_remain_in_certified_rectangles(self):
        path2 = self.stage1.plan(MAGNET2_START, MAGNET2_HOME)
        path1 = self.stage2.plan(np.array([0.03, 0.0]), MAGNET1_HOME)
        self.assertTrue(self.stage1.path_is_certified(path2))
        self.assertTrue(self.stage2.path_is_certified(path1))

    def test_taut_paths_are_certified_and_no_longer_than_midpoint_paths(self):
        rng = np.random.default_rng(8)
        for stage in (self.stage1, self.stage2):
            boxes = stage.boxes[rng.integers(0, len(stage.boxes), 40)]
            points = np.column_stack((
                rng.uniform(boxes[:, 0], boxes[:, 1]),
                rng.uniform(boxes[:, 2], boxes[:, 3]),
            ))
            for start, goal in zip(points[:20], points[20:]):
                try:
                    midpoint = stage.plan(start, goal)
                except ValueError:
                    continue  # different certified components
                taut = stage.plan_taut(start, goal)
                np.testing.assert_allclose(taut[0], start)
                np.testing.assert_allclose(taut[-1], goal)
                for a, b in zip(taut[:-1], taut[1:]):
                    self.assertEqual(stage.reach(a, b), 1.0)
                length = lambda p: np.linalg.norm(np.diff(p, axis=0), axis=1).sum()
                self.assertLessEqual(length(taut), length(midpoint) + 1e-12)

    def test_reach_stops_at_the_certified_edge(self):
        # Rightward from magnet 2's start, the stage-1 set ends a little past
        # x = -8 mm; the reach must stop there and never overshoot.
        start = MAGNET2_START
        end = np.array([0.03, MAGNET2_START[1]])
        t = self.stage1.reach(start, end)
        self.assertGreater(t, 0.0)
        self.assertLess(t, 1.0)
        edge = start + t * (end - start)
        self.assertTrue(self.stage1.contains(edge))
        self.assertFalse(self.stage1.contains(edge + np.array([1e-6, 0.0])))
        self.assertEqual(self.stage1.reach(np.array([0.0, 0.0]), end), 0.0)

    def test_certified_witness_allocates_bounded_currents(self):
        result = solve_control(self.stage1, MAGNET2_START, (0.004, -0.004))
        self.assertTrue(result["within_limits"])
        self.assertLess(np.max(np.abs(result["residual"])), 1e-10)


if __name__ == "__main__":
    unittest.main()
