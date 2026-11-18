"""Focused regression checks for the Section 6/7 scientific review."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from analysis.sections_67 import afw_tools
from hybrid_region import RegionProblem


class ReviewRegressions(unittest.TestCase):
    def test_solver_failure_is_unknown(self):
        failed = SimpleNamespace(success=False, status=4, message='numerical failure', x=None)
        with patch.object(afw_tools, 'linprog', return_value=failed):
            result = afw_tools.current_feasibility(np.zeros(2), np.eye(2), np.zeros(2), 1, 1e-7)
        self.assertIsNone(result['feasible'])

    def test_invalid_success_is_unknown(self):
        failed = SimpleNamespace(success=True, status=0, message='bad residual', x=np.ones(2))
        with patch.object(afw_tools, 'linprog', return_value=failed):
            result = afw_tools.current_feasibility(np.zeros(2), np.eye(2), np.zeros(2), 1, 1e-7)
        self.assertIsNone(result['feasible'])

    def test_hrep_uses_production_implementation(self):
        with patch.object(afw_tools, 'ModifiedHyperplaneShiftingMethod', wraps=afw_tools.ModifiedHyperplaneShiftingMethod) as method:
            afw_tools.zonotope_hrep(np.eye(2), np.array([2., 3.]), np.ones(2))
        self.assertEqual(method.call_count, 1)

    def test_translated_and_degenerate_sets(self):
        examples = [
            (np.eye(2), np.array([2., 3.]), [([2, 3], True), ([3.1, 3], False)]),
            (np.array([[1., 2.], [0., 0.]]), np.array([2., 3.]),
             [([2, 3], True), ([5, 3], True), ([5.1, 3], False), ([2, 3.01], False)]),
            (np.zeros((2, 2)), np.array([2., 3.]), [([2, 3], True), ([2.01, 3], False)]),
        ]
        for g, c, queries in examples:
            for point, expected in queries:
                with self.subTest(g=g, point=point):
                    h = afw_tools.hyperplane_membership(point, g, c, 1, 1e-8)
                    lp = afw_tools.current_feasibility(point, g, c, 1, 1e-7)
                    self.assertEqual(h['feasible'], expected)
                    self.assertEqual(lp['feasible'], expected)
            self.assertTrue(np.isfinite(afw_tools.zonotope_vertices_2d(g, c, np.ones(2))).all())

    def test_disabled_coils_equal_column_removal(self):
        g = np.array([[1., 0., 2.], [0., 1., -1.]])
        c = np.array([0.3, -0.2])
        for y in ([0, 0], [4, 4], [1.3, 0.8]):
            reduced = afw_tools.current_feasibility(y, g[:, :2], c, 1, 1e-7)
            zeroed = g.copy(); zeroed[:, 2] = 0
            full = afw_tools.current_feasibility(y, zeroed, c, 1, 1e-7)
            self.assertEqual(reduced['feasible'], full['feasible'])

    def test_active_indices_keep_order_and_reject_fractional_indices(self):
        order = tuple(reversed(range(10)))
        matrix = np.arange(20).reshape(2, 10)
        self.assertTrue(np.array_equal(RegionProblem(active_coil_indices=order).select_columns(matrix), matrix[:, order]))
        with self.assertRaises(ValueError):
            RegionProblem(active_coil_indices=(0, 1.5))


if __name__ == '__main__':
    unittest.main()
