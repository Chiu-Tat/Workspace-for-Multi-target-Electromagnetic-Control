"""Constrained current allocator of the revised manuscript.

At sample k the allocator solves

    minimize    1/2 ||W_out s||^2 + lam_i/2 ||i||^2 + lam_d/2 ||i - i_prev||^2
    subject to  G i + c + s = h_des
                i_min <= i <= i_max
                |i - i_prev| <= di_max          (optional slew limit)
                G i + c >= out_min              (optional output lower bounds)

``G`` (d x n) maps coil currents to the selected outputs, ``c`` (d) is the
current-independent offset (for example a fixed magnet's interaction), and
``s`` is an explicit slack.  A nonzero slack means the requested outputs are
not reachable under the current limits: it is reported, never hidden.

The problem is built once per (d, n) with cvxpy parameters, so repeated calls
inside a simulation loop only re-solve.  It is solved in weighted units
(rows multiplied by W_out), so rows whose physical scale is tiny -- e.g.
sub-micronewton forces in a low-viscosity fluid -- are not lost below the
solver's absolute tolerances.
"""

from __future__ import annotations

import numpy as np

try:
    import cvxpy as cp
except ImportError:  # pragma: no cover - cvxpy ships in the project .venv
    cp = None

from scipy.optimize import lsq_linear

I_LIMIT = 15.0      # A, per coil, shared by all case studies
DT = 0.1            # s, archived 10 Hz allocation/update rate


class QPAllocator:
    """Reusable slack QP for a fixed output dimension ``d`` and ``n`` coils."""

    def __init__(self, d, n=10, i_min=-I_LIMIT, i_max=I_LIMIT, di_max=None,
                 lam_i=1e-6, lam_d=0.0, solver=None, out_min=None):
        if cp is None:
            raise ImportError('cvxpy is required for QPAllocator')
        self.d, self.n = d, n
        self.solver = solver
        self.Gw = cp.Parameter((d, n))          # W_out G
        self.rhs = cp.Parameter(d)              # W_out (h_des - c)
        self.i_prev = cp.Parameter(n)
        self.i = cp.Variable(n)
        self.sw = cp.Variable(d)                # weighted slack W_out s

        cost = 0.5 * cp.sum_squares(self.sw)
        cost += 0.5 * lam_i * cp.sum_squares(self.i)
        if lam_d > 0:
            cost += 0.5 * lam_d * cp.sum_squares(self.i - self.i_prev)
        cons = [self.Gw @ self.i + self.sw == self.rhs,
                self.i >= i_min, self.i <= i_max]
        if di_max is not None:
            cons.append(cp.abs(self.i - self.i_prev) <= di_max)
        self.lo_rows = None
        if out_min is not None:
            lo = np.asarray(out_min, float)
            self.lo_rows = np.flatnonzero(np.isfinite(lo))
            self.lo = lo
            self.lo_rhs = cp.Parameter(len(self.lo_rows))   # W_out (lo - c)
            cons.append(self.Gw[self.lo_rows] @ self.i >= self.lo_rhs)
        self.problem = cp.Problem(cp.Minimize(cost), cons)
        self.i_prev.value = np.zeros(n)

    def solve(self, G, c, h_des, w=None, i_prev=None):
        G = np.asarray(G, float)
        c = np.zeros(self.d) if c is None else np.asarray(c, float)
        h_des = np.asarray(h_des, float)
        w = np.ones(self.d) if w is None else np.asarray(w, float)
        self.Gw.value = w[:, None] * G
        self.rhs.value = w * (h_des - c)
        if self.lo_rows is not None:
            r = self.lo_rows
            self.lo_rhs.value = w[r] * (self.lo[r] - c[r])
        if i_prev is not None:
            self.i_prev.value = np.asarray(i_prev, float)
        try:
            self.problem.solve(solver=self.solver, warm_start=True)
            ok = self.problem.status in ('optimal', 'optimal_inaccurate')
        except Exception:  # solver failure is reported, not raised
            ok = False
        if not ok or self.i.value is None:
            return fallback_allocate(G, c, h_des, w)
        currents = np.asarray(self.i.value, float)
        return _report(G, c, h_des, currents, w, self.problem.status)


def fallback_allocate(G, c, h_des, w=None, i_min=-I_LIMIT, i_max=I_LIMIT):
    """Bounded weighted least squares; used when the QP cannot be solved."""
    G = np.asarray(G, float)
    w = np.ones(G.shape[0]) if w is None else np.asarray(w, float)
    sol = lsq_linear(w[:, None] * G, w * (np.asarray(h_des) - c),
                     bounds=(i_min, i_max))
    return _report(G, c, h_des, sol.x, w, 'fallback_lsq')


def _report(G, c, h_des, currents, w, status):
    achieved = G @ currents + c
    slack = h_des - achieved
    w = np.ones(len(h_des)) if w is None else np.asarray(w, float)
    return {
        'currents': currents,
        'achieved': achieved,
        'slack': slack,
        'weighted_slack': float(np.linalg.norm(w * slack)),
        'saturated': bool(np.max(np.abs(currents)) > I_LIMIT - 1e-3),
        'status': status,
    }


def row_weights(scales):
    """W_out diagonal that turns mixed T / N rows into relative errors."""
    return 1.0 / np.asarray(scales, float)
