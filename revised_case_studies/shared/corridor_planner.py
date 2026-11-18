"""A* + safe corridor + piecewise-Bezier trajectory optimisation in a 2-D TFW.

The planning pipeline of the paper (Sec. "Planning, feedback, and current
allocation", after gao2018online), as used by the Case 3 and Case 6 planners:

  1. A* on a grid of free (certified) cells, with an extra cost near the
     boundary of the free set (so the path runs through wide free areas),
     then a line-of-sight shortcut of the path;
  2. a corridor of overlapping axis-aligned free boxes covering the path
     (grow a box along the path, then inflate it);
  3. one Bezier segment of degree N per box, minimising squared jerk, with
       - every position control point inside its box  (convex-hull property:
         the whole curve stays in the corridor, hence in the free set),
       - velocity / acceleration control points inside per-axis limits
         (hodograph property: the limits hold along the whole curve),
       - rest-to-rest boundary conditions and C2 continuity at junctions.
     Segment durations start from the per-axis (Chebyshev) distance at the
     speed limit and are then adapted per segment: a relaxed QP with slack
     on each segment's limits shows which segments must be lengthened, and
     segments far below the limits are shortened; the fastest allocation
     that satisfies every limit (checked by the hard QP) is kept and finally
     scaled down as a whole by bisection while the hard QP stays feasible.

Coordinates are generic: ``x0 + (index + 0.5) * cell`` per axis.  Limits are
per axis, so each axis may be a different robot (Case Study 2: the arc
positions of the two spheres).
"""

from __future__ import annotations

import heapq
import math

import cvxpy as cp
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.special import comb

from revised_case_studies.shared import grid_planner as gp

DEGREE = 5
CLEARANCE_WEIGHT = 4.0        # extra step cost at the boundary of the free set
CLEARANCE_SCALE = 6.0         # cells over which that extra cost decays


def astar_clearance(free, start, goal, weight=CLEARANCE_WEIGHT, scale=CLEARANCE_SCALE):
    """8-connected A*; step cost = length * (1 + weight * exp(-clearance / scale))."""
    rows, cols = free.shape
    if not (free[start] and free[goal]):
        return None
    clear = distance_transform_edt(free)
    penalty = 1.0 + weight * np.exp(-clear / scale)
    g = {start: 0.0}
    parent = {start: None}
    heap = [(0.0, start)]
    moves = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    while heap:
        _, cur = heapq.heappop(heap)
        if cur == goal:
            path = []
            while cur is not None:
                path.append(cur)
                cur = parent[cur]
            return path[::-1]
        for dr, dc in moves:
            nxt = (cur[0] + dr, cur[1] + dc)
            if not (0 <= nxt[0] < rows and 0 <= nxt[1] < cols) or not free[nxt]:
                continue
            cost = g[cur] + math.hypot(dr, dc) * 0.5 * (penalty[cur] + penalty[nxt])
            if cost < g.get(nxt, math.inf):
                g[nxt] = cost
                parent[nxt] = cur
                heapq.heappush(heap, (cost + math.hypot(goal[0] - nxt[0], goal[1] - nxt[1]), nxt))
    return None


def densify(path):
    """Cells along a shortcut polyline (so the corridor can follow it)."""
    out = [path[0]]
    for a, b in zip(path[:-1], path[1:]):
        n = int(max(abs(b[0] - a[0]), abs(b[1] - a[1])))
        for t in np.linspace(0, 1, n + 1)[1:]:
            rc = (int(round(a[0] + t * (b[0] - a[0]))), int(round(a[1] + t * (b[1] - a[1]))))
            if rc != out[-1]:
                out.append(rc)
    return out


def corridor(free, path):
    """Overlapping free boxes (row0, row1, col0, col1; half-open) covering ``path``."""
    rows, cols = free.shape
    boxes, i = [], 0
    while i < len(path):
        r0, c0 = path[i]
        r1, c1 = r0 + 1, c0 + 1
        end = i
        for j in range(i + 1, len(path)):
            r, c = path[j]
            tr0, tc0, tr1, tc1 = min(r0, r), min(c0, c), max(r1, r + 1), max(c1, c + 1)
            if not free[tr0:tr1, tc0:tc1].all():
                break
            r0, c0, r1, c1, end = tr0, tc0, tr1, tc1, j
        grown = True
        while grown:                                    # inflate
            grown = False
            if r0 > 0 and free[r0 - 1, c0:c1].all():
                r0 -= 1; grown = True
            if r1 < rows and free[r1, c0:c1].all():
                r1 += 1; grown = True
            if c0 > 0 and free[r0:r1, c0 - 1].all():
                c0 -= 1; grown = True
            if c1 < cols and free[r0:r1, c1].all():
                c1 += 1; grown = True
        boxes.append((r0, r1, c0, c1))
        i = end if end > i and end < len(path) - 1 else end + 1
    return boxes


def _bezier_basis(n, u):
    return np.array([comb(n, k) * u ** k * (1 - u) ** (n - k) for k in range(n + 1)])


def _solve(start, goal, boxes_xy, vmax, amax, T, n, soft):
    """One QP for fixed durations ``T``; with ``soft`` the speed/acceleration
    limits get per-segment slack (returned) instead of being hard."""
    m = len(boxes_xy)
    C = [cp.Variable((n + 1, m)) for _ in range(2)]
    sv = cp.Variable(m, nonneg=True) if soft else None
    sa = cp.Variable(m, nonneg=True) if soft else None
    cons, cost = [], 0
    for ax in range(2):
        X = C[ax]
        cons += [X[0, 0] == start[ax], X[1, 0] == X[0, 0], X[2, 0] - 2 * X[1, 0] + X[0, 0] == 0,
                 X[n, m - 1] == goal[ax], X[n, m - 1] == X[n - 1, m - 1],
                 X[n, m - 1] - 2 * X[n - 1, m - 1] + X[n - 2, m - 1] == 0]
        for i in range(m - 1):
            T1, T2 = T[i], T[i + 1]
            cons += [X[n, i] == X[0, i + 1],
                     (X[n, i] - X[n - 1, i]) / T1 == (X[1, i + 1] - X[0, i + 1]) / T2,
                     (X[n, i] - 2 * X[n - 1, i] + X[n - 2, i]) / T1 ** 2
                     == (X[2, i + 1] - 2 * X[1, i + 1] + X[0, i + 1]) / T2 ** 2]
        for i in range(m):
            lo, hi = (boxes_xy[i][0], boxes_xy[i][1]) if ax == 0 else (boxes_xy[i][2], boxes_xy[i][3])
            cons += [X[:, i] >= lo, X[:, i] <= hi]
            V = (n / T[i]) * (X[1:, i] - X[:-1, i])
            A = ((n - 1) / T[i]) * (V[1:] - V[:-1])
            J = ((n - 2) / T[i]) * (A[1:] - A[:-1])
            if soft:
                cons += [cp.abs(V) <= vmax[ax] * (1 + sv[i]), cp.abs(A) <= amax[ax] * (1 + sa[i])]
            else:
                cons += [cp.abs(V) <= vmax[ax], cp.abs(A) <= amax[ax]]
            cost += cp.sum_squares(J) * T[i]
    scale = max(float(np.max(vmax)), 1e-12)
    if soft:
        cost = cost / scale ** 2 + 1e3 * cp.sum(sv + sa)
    prob = cp.Problem(cp.Minimize(cost), cons)
    try:
        prob.solve(solver=cp.CLARABEL)
    except cp.SolverError:
        return None, None, None
    if prob.status not in ('optimal', 'optimal_inaccurate') or C[0].value is None:
        return None, None, None
    ctrl = np.stack([C[0].value, C[1].value])
    return ctrl, (sv.value if soft else None), (sa.value if soft else None)


def _peak_ratio(ctrl, T, vmax, amax, n):
    """Per segment: max over axes of (peak speed / vmax, peak accel / amax)."""
    out = []
    for i, Ti in enumerate(T):
        V = (n / Ti) * np.diff(ctrl[:, :, i], axis=1)
        A = ((n - 1) / Ti) * np.diff(V, axis=1)
        out.append(max(float(np.max(np.abs(V[k]) / vmax[k])) for k in range(2)) if V.size else 0.0)
        out[-1] = max(out[-1], max(float(np.max(np.abs(A[k]) / amax[k])) for k in range(2)) if A.size else 0.0)
    return np.array(out)


def optimise(start, goal, boxes_xy, vmax, amax, n=DEGREE, iters=40, tol=1e-3):
    """Piecewise Bezier through ``boxes_xy`` [(x0, x1, y0, y1)], rest to rest.

    Per-segment time allocation: start from the per-axis distance at the
    speed limit; solve with per-segment slack on the limits; lengthen the
    segments that need slack, shorten the ones well below the limits; keep
    the fastest solution that satisfies all limits (verified by a hard QP).
    Returns (control points (2, n+1, m), durations) or (None, None).
    """
    start, goal = np.asarray(start, float), np.asarray(goal, float)
    vmax, amax = np.asarray(vmax, float), np.asarray(amax, float)
    m = len(boxes_xy)
    centres = [np.array([(b[0] + b[1]) / 2, (b[2] + b[3]) / 2]) for b in boxes_xy]
    way = [start] + centres[1:] + [goal] if m > 1 else [start, goal]
    T = np.array([max(float(np.max(np.abs(way[min(i + 1, len(way) - 1)] - way[i]) / vmax)), 0.3)
                  for i in range(m)])
    T[0] += float(np.max(vmax / amax))          # time to accelerate from rest
    T[-1] += float(np.max(vmax / amax))         # and to stop
    best = (None, None, math.inf)
    for _ in range(iters):
        ctrl, sv, sa = _solve(start, goal, boxes_xy, vmax, amax, T, n, soft=True)
        if ctrl is None:
            T = T * 1.3
            continue
        slack = np.maximum(sv, sa)
        if slack.max() <= tol:
            hard, _, _ = _solve(start, goal, boxes_xy, vmax, amax, T, n, soft=False)
            if hard is not None and T.sum() < best[2]:
                best = (hard, T.copy(), T.sum())
            ratio = _peak_ratio(hard if hard is not None else ctrl, T, vmax, amax, n)
            shrink = ratio < 0.8
            if not shrink.any():
                break
            T = np.where(shrink, T * np.maximum(0.85, ratio / 0.9), T)
        else:
            T = np.where(slack > tol, T * (1.0 + np.minimum(slack, 1.0) + 0.1), T)
    if best[0] is None:
        return None, None
    # final global time scaling: shrink all durations together while the hard QP stays feasible
    ctrl, Tb = best[0], best[1]
    lo, hi = 0.2, 1.0
    for _ in range(8):
        mid = 0.5 * (lo + hi)
        trial, _, _ = _solve(start, goal, boxes_xy, vmax, amax, Tb * mid, n, soft=False)
        if trial is not None:
            hi, ctrl = mid, trial
        else:
            lo = mid
    return ctrl, list(Tb * hi)


def sample(ctrl, T, dt):
    """(t, xy) sampled from the piecewise Bezier."""
    n = ctrl.shape[1] - 1
    ts, xy, t0 = [0.0], [ctrl[:, 0, 0]], 0.0
    for i, Ti in enumerate(T):
        k = max(2, int(math.ceil(Ti / dt)) + 1)
        u = np.linspace(0, 1, k)[1:]
        B = _bezier_basis(n, u)
        for j in range(len(u)):
            ts.append(t0 + u[j] * Ti)
            xy.append(ctrl[:, :, i] @ B[:, j])
        t0 += Ti
    return np.array(ts), np.array(xy)


def plan(free, start_rc, goal_rc, cell, origin, vmax, amax, dt=0.05):
    """Full pipeline on a free grid [row = axis 1, col = axis 0].

    ``cell``/``origin`` map indices to coordinates (per axis: x = origin +
    (index + 0.5) * cell for centres; box edges at origin + index * cell).
    Returns dict(t, xy, boxes_xy, path_xy) or None.
    """
    path = astar_clearance(free, start_rc, goal_rc)
    if path is None:
        return None
    path = densify(gp.shortcut(free, path))
    to_xy = lambda rc: (origin[0] + (rc[1] + 0.5) * cell[0], origin[1] + (rc[0] + 0.5) * cell[1])
    start, goal = to_xy(start_rc), to_xy(goal_rc)
    if start_rc == goal_rc:
        return {'t': np.array([0.0]), 'xy': np.array([start]), 'boxes_xy': [], 'path_xy': [start]}
    boxes = corridor(free, path)
    boxes_xy = [(origin[0] + c0 * cell[0], origin[0] + c1 * cell[0],
                 origin[1] + r0 * cell[1], origin[1] + r1 * cell[1]) for r0, r1, c0, c1 in boxes]
    ctrl, T = optimise(start, goal, boxes_xy, np.asarray(vmax, float), np.asarray(amax, float))
    if ctrl is None:
        return None
    t, xy = sample(ctrl, T, dt)
    return {'t': t, 'xy': xy, 'boxes_xy': boxes_xy, 'path_xy': [to_xy(rc) for rc in path],
            'ctrl': ctrl, 'T': T}
