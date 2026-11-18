"""Forward/backward-and-turn motion of the free magnet inside the 3-D TFW.

The task holds zero lateral body force (fy = 0) and commands only the thrust
along the magnet axis (fz), so a magnet that stays in the certified set can only
drive forward, drive backward, or turn (heading follows the field).  This module
plans and simulates exactly that motion on the corrected 3-D TFWs
(``common.tfw(layout)``); it computes no new TFW.

  plan(layout, start, goal)   hybrid A* over (x, y, beta) with straight,
                              arc and turn-in-place primitives, forward and
                              backward; every sampled pose must be certified
                              with a margin at its own heading and the two
                              neighbouring heading levels
  straight(start, goal)       constant-heading reference (no planning)
  simulate(traj, layout)      10 Hz QP allocation + 100 Hz plant of
                              ``closed_loop`` for any fixed-magnet layout; stops
                              on contact with a fixed magnet
"""

from __future__ import annotations

import heapq
import math
from functools import lru_cache

import numpy as np
from scipy.ndimage import distance_transform_edt

from revised_case_studies.cs1_cylinder_interaction import common as c
from revised_case_studies.cs1_cylinder_interaction.closed_loop import (
    DT_CTRL, DT_PLANT, ROW_SCALE, control_command, plant_step)
from revised_case_studies.shared.allocator import QPAllocator, row_weights

V_REF = 0.005                 # m/s, driving speed
W_REF = math.radians(30)      # rad/s, turning rate
STEP = 0.002                  # m, primitive length
R_TURN = 0.015                # m, arc radius of the curved primitives
MARGIN = 0.002                # m, certified-edge clearance required in the slice
CLEAR = 0.016                 # m, centre distance kept from every fixed magnet
GOAL_TOL = 0.0025             # m
REVERSE_COST, SWITCH_COST, TURN_COST, EDGE_COST = 1.6, 0.01, 0.006, 0.004


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


@lru_cache(maxsize=None)
def edge_distance(layout):
    """(levels, n, n) distance (m) from each cell to the nearest non-certified cell."""
    t = c.tfw(layout)
    cell = (t.x_max - t.x_min) / t.raster
    return np.stack([distance_transform_edt(np.pad(s, 1)) [1:-1, 1:-1] * cell
                     for s in t.stack])


def clearance(x, y, beta, layout):
    """Smallest certified-edge distance over this heading level and its neighbours."""
    t = c.tfw(layout)
    cell = t.cell_of(x, y)
    if cell is None:
        return 0.0
    i, j = cell
    lvl = t.beta_level(beta)
    d = edge_distance(layout)
    return min(d[(lvl + k) % t.n_levels, j, i] for k in (-1, 0, 1))


def pose_ok(x, y, beta, layout, margin=MARGIN):
    if c.nearest_fixed(x, y, layout)[0] < CLEAR:
        return False
    return clearance(x, y, beta, layout) >= margin


def _primitives():
    """(kind, direction, steering) triples; kind 'turn' rotates in place."""
    prims = []
    for d in (1, -1):
        for s in (-1, 0, 1):
            prims.append(('drive', d, s))
    prims += [('turn', 0, 1), ('turn', 0, -1)]
    return prims


def _apply(pose, prim, n=4):
    """Poses sampled along one primitive, endpoint last."""
    x, y, b = pose
    kind, d, s = prim
    out = []
    if kind == 'turn':
        db = s * c.tfw('single').beta_step
        for k in range(1, n + 1):
            out.append((x, y, b + db * k / n))
        return out
    for k in range(1, n + 1):
        L = d * STEP * k / n
        if s == 0:
            out.append((x + L * math.cos(b), y + L * math.sin(b), b))
        else:
            db = s * L / R_TURN
            xc, yc = x - s * R_TURN * math.sin(b), y + s * R_TURN * math.cos(b)
            out.append((xc + s * R_TURN * math.sin(b + db),
                        yc - s * R_TURN * math.cos(b + db), b + db))
    return out


def plan(layout, start, goal, max_expand=400000, margin=MARGIN, costs=None):
    """Hybrid A* from ``start`` to ``goal`` (x, y, beta).  Returns (N, 3) poses.

    ``costs`` overrides the turn, reverse, switch, edge and curvature weights.
    """
    w = dict(turn=TURN_COST, reverse=REVERSE_COST, switch=SWITCH_COST, edge=EDGE_COST,
             curve=0.0003)
    w.update(costs or {})
    t = c.tfw(layout)
    nb = t.n_levels
    res = 0.001

    def key(p):
        return (int(round(p[0] / res)), int(round(p[1] / res)),
                int(round((p[2] % (2 * math.pi)) / t.beta_step)) % nb)

    def h(p):
        return math.dist(p[:2], goal[:2]) + w['turn'] / t.beta_step * abs(wrap(p[2] - goal[2]))

    start = tuple(start)
    prims = _primitives()
    openq = [(h(start), 0.0, start, None)]
    parent, gbest = {}, {key(start): 0.0}
    closed = set()
    n = 0
    while openq and n < max_expand:
        f, g, p, came = heapq.heappop(openq)
        k = key(p)
        if k in closed:
            continue
        closed.add(k)
        parent[k] = came
        n += 1
        if math.dist(p[:2], goal[:2]) < GOAL_TOL and abs(wrap(p[2] - goal[2])) < t.beta_step:
            path = [p]
            while parent[key(path[-1])] is not None:
                path.append(parent[key(path[-1])][0])
            path = path[::-1]
            return _finish(np.array(path), goal, layout)
        last = came[1] if came else None
        for prim in prims:
            seg = _apply(p, prim)
            if not all(pose_ok(*q, layout, margin) for q in seg):
                continue
            q = seg[-1]
            kind, d, s = prim
            if kind == 'turn':
                cost = w['turn']
            else:
                cost = STEP * (w['reverse'] if d < 0 else 1.0) + (w['curve'] if s else 0.0)
            if last is not None and last[1] != 0 and d != 0 and d != last[1]:
                cost += w['switch']
            cost += w['edge'] * max(0.0, 1 - clearance(*q, layout) / 0.008)
            g2 = g + cost
            kq = key(q)
            if kq in closed or g2 >= gbest.get(kq, math.inf):
                continue
            gbest[kq] = g2
            heapq.heappush(openq, (g2 + h(q), g2, q, (p, prim)))
    raise RuntimeError(f'no path for {layout} after {n} expansions')


def _finish(path, goal, layout):
    """Close the last sub-level heading gap in place (checked by the goal test)."""
    x, y, b = path[-1]
    db = wrap(goal[2] - b)
    m = max(1, int(abs(db) / math.radians(2)))
    tail = [(x, y, b + db * k / m) for k in range(1, m + 1)]
    return np.vstack([path, tail]) if tail else path


def segments(path):
    """Split a pose list into runs of 'drive' (+1/-1) and 'turn' (0)."""
    out = []
    for a, b in zip(path[:-1], path[1:]):
        dxy = b[:2] - a[:2]
        if np.linalg.norm(dxy) < 1e-9:
            mode = 0
        else:
            mode = 1 if dxy @ (math.cos(a[2]), math.sin(a[2])) > 0 else -1
        out.append(mode)
    return np.array(out)


def timed(path, v=V_REF, w=W_REF, dt=0.05, dwell=1.0):
    """(N, 4) [x, y, beta, t] at constant speed / turning rate."""
    path = np.asarray(path, float)
    path[:, 2] = np.unwrap(path[:, 2])
    tt = [0.0]
    for a, b in zip(path[:-1], path[1:]):
        L = np.linalg.norm(b[:2] - a[:2])
        tt.append(tt[-1] + max(L / v, abs(b[2] - a[2]) / w, 1e-6))
    tt = np.array(tt) + dwell
    path = np.vstack([path[:1], path])
    tt = np.concatenate([[0.0], tt])
    T = np.arange(0, tt[-1] + dwell + 1e-9, dt)
    cols = [np.interp(T, tt, path[:, k]) for k in range(3)]
    return np.column_stack(cols + [T])


def straight(start, goal, v=V_REF, dt=0.05, dwell=1.0):
    """Constant-heading reference from start to goal (heading of start)."""
    L = math.dist(start[:2], goal[:2])
    T = np.arange(0, L / v + 2 * dwell + 1e-9, dt)
    s = np.clip((T - dwell) * v / L, 0, 1)
    x = start[0] + s * (goal[0] - start[0])
    y = start[1] + s * (goal[1] - start[1])
    return np.column_stack([x, y, np.full_like(T, start[2]), T])


def simulate(traj, layout='single', t_extra=2.0):
    """Closed-loop tracking of ``traj`` with the slack QP; stops at contact."""
    alloc = QPAllocator(4)
    w = row_weights(ROW_SCALE)
    x, y, beta = traj[0, :3]
    i_prev = np.zeros(c.NUM_COILS)
    T = traj[-1, 3] + t_extra
    keys = ('t', 'x', 'y', 'beta', 'xr', 'yr', 'betar', 'slack', 'fy_res',
            'currents', 'label', 'sep', 'f_lat', 'f_int')
    log = {k: [] for k in keys}
    n_sub = int(round(DT_CTRL / DT_PLANT))
    contact_t = None
    for k in range(int(T / DT_CTRL) + 1):
        t = k * DT_CTRL
        beta_c, fz_c, (xr, yr, br) = control_command(traj, t, x, y)
        A_hat = c.body_matrix(x, y, beta_c, layout)
        sol = alloc.solve(A_hat[:, :c.NUM_COILS], A_hat[:, c.NUM_COILS],
                          np.array([0.0, c.B_HOLD, 0.0, fz_c]), w, i_prev)
        cur = sol['currents']
        i_prev = cur
        Aw = c.world_matrix(x, y, beta, layout)
        F = Aw[3:5] @ np.append(cur, 1.0)
        n_hat = np.array([-math.sin(beta), math.cos(beta)])
        sep = c.nearest_fixed(x, y, layout)[0]
        for name, val in (('t', t), ('x', x), ('y', y), ('beta', beta), ('xr', xr),
                          ('yr', yr), ('betar', br), ('slack', sol['weighted_slack']),
                          ('fy_res', float(sol['slack'][2])), ('currents', cur),
                          ('label', c.label(x, y, beta, check_outside=False, layout=layout)),
                          ('sep', sep), ('f_lat', float(F @ n_hat)),
                          ('f_int', float(np.linalg.norm(Aw[3:5, c.NUM_COILS])))):
            log[name].append(val)
        if contact_t is not None:
            break
        for _ in range(n_sub):
            x, y, beta, _, _ = plant_step(x, y, beta, cur, layout=layout)
            if c.nearest_fixed(x, y, layout)[0] < c.CONTACT_DIST:
                contact_t = t + DT_PLANT
                break
    out = {k: np.asarray(v) for k, v in log.items()}
    out['contact_t'] = contact_t
    out['err'] = np.hypot(out['x'] - out['xr'], out['y'] - out['yr'])
    return out
