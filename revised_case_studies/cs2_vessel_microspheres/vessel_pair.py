"""Two microspheres in two thin vessel networks, moving only along their routes.

Geometry: the Case 7 separate-vessel presets
(``cases/case_7_vessel_microspheres/geometries``), re-registered so that the
point where the two trunks pass closest sits at the centre of the coil array,
``WORKSPACE_CENTER`` = (0, 0, 70.2) mm in the eMNS frame (array axis at
x = y = 0; array centre plane from ``shared/coils.py``).  Case 7 itself places
them at z = 85 mm; only this rigid translation differs:

  side_by_side           close_parallel.json     artery + companion vein, 0.50 mm gap
  over_under             close_crossing.json     crossing in projection, 0.35 mm gap in z
  branched_side_by_side  branched_parallel.json  arterial and venous trees (14 segments)
  branched_over_under    branched_crossing.json  the same trees crossing at two depths

The lumens are thin compared with the travel, so each sphere is reduced to
one coordinate: the arc length a along its own route.  Each sphere stays in
its main vessel (the trunk RA or RV); in the branched presets the side
branches are anatomical context only and are drawn, not travelled.  The wall
reacts every transverse load; only the tangential force moves a sphere:
da/dt = (F . t) / drag.

Actuation (same model and allocator as ``model.py``): the net field at both
spheres is held along one common direction (this fixes the aligned moments),
its magnitude above ``B_MIN``; the tangential force of each sphere is
controlled; the transverse (wall) forces are only weakly penalised.  The
sphere-sphere dipole interaction is included in the allocation offset and
in the plant.

Mission (autonomous, both spheres at the same time, all four presets): each
sphere moves from its start end to the opposite end of its trunk (sphere 1
forward, sphere 2 backward), so they pass each other, then stop.

Two ways of turning that mission into a reference for (a1, a2):
  plan='tfw'       the paper's planner inside the certified TFW of ``tfw.py``:
                   on a uniform grid in (a1, a2) whose free cells lie wholly in
                   certified boxes, A* -> safe corridor of free boxes ->
                   jerk-minimising piecewise Bezier (degree 5) with every
                   control point in its box (so the whole curve is certified)
                   and per-sphere speed <= SPEED, acceleration <= ACCEL,
                   rest to rest (``shared/corridor_planner.py``).  A waypoint
                   outside the free set is moved to the nearest free cell in
                   the same connected region as the previous waypoint.  Where
                   the corridor detours around an uncertified region, one
                   sphere slows while the other keeps going.
  plan='straight'  both spheres move at constant speed straight to each
                   waypoint (a straight segment in (a1, a2)), ignoring the TFW.
"""

from __future__ import annotations

from pathlib import Path
import hashlib

import numpy as np

from cases.case_7_vessel_microspheres.vessel import Vessel
from revised_case_studies.cs2_vessel_microspheres import config as K
from revised_case_studies.cs2_vessel_microspheres.model import allocation_rows, plant
from revised_case_studies.shared import coils
from revised_case_studies.shared.allocator import QPAllocator

GEOM_DIR = Path(__file__).resolve().parents[2] / 'cases' / 'case_7_vessel_microspheres' / 'geometries'
WORKSPACE_CENTER = np.array([0.0, 0.0, coils.POSE_TO_EMNS_MM[2]]) * 1e-3   # m, coil-array centre
# name: (file, title, route of sphere 1, route of sphere 2)
# a route is a list of (branch, s_from, s_to); every sphere stays in its trunk
VARIANTS = {
    'side_by_side': ('close_parallel.json', 'Side by side (radial artery + companion vein)',
                     [('RA', 0.0, 1.0)], [('RV', 0.0, 1.0)]),
    'over_under': ('close_crossing.json', 'Over-under (crossing at different depths)',
                   [('RA', 0.0, 1.0)], [('RV', 0.0, 1.0)]),
    'branched_side_by_side': ('branched_parallel.json', 'Branched networks, side by side',
                              [('RA', 0.0, 1.0)], [('RV', 0.0, 1.0)]),
    'branched_over_under': ('branched_crossing.json', 'Branched networks, over-under',
                            [('RA', 0.0, 1.0)], [('RV', 0.0, 1.0)]),
}
SPEED = 0.003                  # m/s along the route
K_P = 2.0                      # 1/s
A_LO, A_HI = 0.08, 0.92        # route fractions where the mission starts and turns
# Sphere 1 (3 mm N35) starts and ends further from its trunk ends: near the far
# ends the coils cannot supply the 0.5 mN task in the over-under geometry, and
# 0.28 -> 0.82 is connected inside both certified maps with > 2 mm clearance.
A1_LO, A1_HI = 0.28, 0.82
W_TRANSVERSE = 1e-2            # relative weight of wall-load rows
DT_CTRL = 0.1                  # s, 10 Hz allocation
DT_PLANT = 0.002               # s


class Route:
    """Chain of centerline segments, parameterised by arc length (m)."""

    def __init__(self, vessel, segments, step=2e-4):
        pts, rad, names, s_all = [], [], [], []
        for branch, s0, s1 in segments:
            L = np.linalg.norm(np.diff(vessel.centerline(branch, np.linspace(s0, s1, 200)),
                                       axis=0), axis=1).sum()
            s = np.linspace(s0, s1, max(3, int(L / step)))
            p = vessel.centerline(branch, s)
            r = vessel.radius(branch, s)
            if pts:
                p, r, s = p[1:], r[1:], s[1:]
            pts.append(p)
            s_all.append(s)
            rad.append(r)
            names += [branch] * len(p)
        self.pts = np.concatenate(pts)
        self.s = np.concatenate(s_all)
        self.radius = np.concatenate(rad)
        self.branches = names
        self.arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(self.pts, axis=0), axis=1))])
        self.length = float(self.arc[-1])
        d = np.gradient(self.pts, self.arc, axis=0)
        # light smoothing of the tangent (~1 mm window)
        w = max(1, int(1e-3 / step))
        kern = np.ones(2 * w + 1) / (2 * w + 1)
        d = np.column_stack([np.convolve(np.pad(d[:, k], w, mode='edge'), kern, 'valid')
                             for k in range(3)])
        self.tan = d / np.linalg.norm(d, axis=1, keepdims=True)

    def _i(self, a):
        return int(np.clip(np.searchsorted(self.arc, a), 0, len(self.arc) - 1))

    def point(self, a):
        a = float(np.clip(a, 0, self.length))
        return np.array([np.interp(a, self.arc, self.pts[:, k]) for k in range(3)])

    def frame(self, a):
        t = self.tan[self._i(a)]
        ref = np.eye(3)[np.argmin(np.abs(t))]
        n = np.cross(t, ref)
        n /= np.linalg.norm(n)
        return t, n, np.cross(t, n)

    def branch_at(self, a):
        return self.branches[self._i(a)]

    def a_of_s(self, s):
        return float(np.interp(s, self.s, self.arc))

    def s_of_a(self, a):
        return float(np.interp(a, self.arc, self.s))


TFW_RASTER = 256                   # cells per axis of the (s1, s2) certified raster
PLAN_CELL = 0.25e-3                # m, planning-grid cell in (a1, a2)
ACCEL = 1.5e-3                     # m/s^2, per-sphere acceleration limit of the planner
SNAP_CLEARANCE = 1.0e-3            # m, preferred depth of waypoints inside the free set


class VesselPair:
    def __init__(self, variant='side_by_side', plan='tfw'):
        fname, self.title, r1, r2 = VARIANTS[variant]
        self.variant = variant
        self.branched = variant.startswith('branched')
        self.vessel = Vessel(GEOM_DIR / fname)
        self.labels = [self.vessel.branches[r[0][0]].get('label', r[0][0]) for r in (r1, r2)]
        # centre the trunks' closest point on the coil array (rigid translation)
        self._build_routes(r1, r2)
        mid = (self.routes[0].point(self.a_close[0]) + self.routes[1].point(self.a_close[1])) / 2
        self.vessel.origin = self.vessel.origin + (WORKSPACE_CENTER - mid)
        self._build_routes(r1, r2)
        self.center = WORKSPACE_CENTER
        self.tfw = self._load_tfw()
        self.plan = plan if self.tfw is not None else 'straight'
        self.mission = self._mission()

    def _build_routes(self, r1, r2):
        self.routes = [Route(self.vessel, r1), Route(self.vessel, r2)]
        # closest pair of route points (where the two vessels pass nearest)
        P, Q = self.routes[0].pts, self.routes[1].pts
        d = np.linalg.norm(P[::4, None, :] - Q[None, ::4, :], axis=2)
        i, j = np.unravel_index(np.argmin(d), d.shape)
        self.a_close = (float(self.routes[0].arc[4 * i]), float(self.routes[1].arc[4 * j]))
        self.d_close = float(d[i, j])

    # ------------------------------------------------------------ TFW
    def _load_tfw(self):
        from revised_case_studies.shared.paths import RESULTS_OUT
        f = RESULTS_OUT / 'cs2' / f'tfw_{self.variant}.json'
        if not f.exists():
            return None
        import json
        d = json.loads(f.read_text())
        source = GEOM_DIR / VARIANTS[self.variant][0]
        if d.get('geometry_sha256') != hashlib.sha256(source.read_bytes()).hexdigest():
            # Historical unscaled presets predate geometry provenance.
            if self.branched or 'geometry_sha256' in d:
                return None
        n = TFW_RASTER
        free = np.zeros((n, n), bool)                  # [row = s2, col = s1]
        for b in d['inside']:
            c0, c1 = int(round(b[0] * n)), int(round(b[1] * n))
            r0, r1 = int(round(b[2] * n)), int(round(b[3] * n))
            free[r0:r1, c0:c1] = True
        d['raster'] = free
        return d

    def certified(self, a):
        if self.tfw is None:
            return None
        n = TFW_RASTER
        s = [self.routes[k].s_of_a(a[k]) for k in range(2)]
        i = [min(n - 1, max(0, int(v * n))) for v in s]
        return bool(self.tfw['raster'][i[1], i[0]])

    # ------------------------------------------------------------ planning grid
    def _build_arc_grid(self):
        """Uniform grid in (a1, a2); a cell is free only if its whole footprint
        in (s1, s2) lies inside certified boxes (so corridors built from free
        cells, and Bezier curves inside them, stay certified)."""
        n = TFW_RASTER
        h = PLAN_CELL
        blocked = (~self.tfw['raster']).astype(int)                 # [s2, s1]
        P = np.zeros((n + 1, n + 1), int)
        P[1:, 1:] = blocked.cumsum(0).cumsum(1)

        def span(route, count):
            edges = np.arange(count + 1) * h
            s = np.array([route.s_of_a(min(e, route.length)) for e in edges])
            lo = np.clip(np.floor(s[:-1] * n).astype(int), 0, n - 1)
            hi = np.clip(np.ceil(s[1:] * n).astype(int), lo + 1, n)
            return lo, hi

        n1 = int(self.routes[0].length / h)
        n2 = int(self.routes[1].length / h)
        c0, c1 = span(self.routes[0], n1)
        r0, r1 = span(self.routes[1], n2)
        R0, C0 = np.meshgrid(r0, c0, indexing='ij')
        R1, C1 = np.meshgrid(r1, c1, indexing='ij')
        count = P[R1, C1] - P[R0, C1] - P[R1, C0] + P[R0, C0]
        free = count == 0                                            # [a2 cell, a1 cell]
        from scipy.ndimage import label
        self.grid = free
        self.grid_comp, _ = label(free, structure=np.ones((3, 3), int))

    def _rc(self, a):
        return (min(self.grid.shape[0] - 1, max(0, int(a[1] / PLAN_CELL))),
                min(self.grid.shape[1] - 1, max(0, int(a[0] / PLAN_CELL))))

    def _centre(self, rc):
        return ((rc[1] + 0.5) * PLAN_CELL, (rc[0] + 0.5) * PLAN_CELL)

    def _snap(self, a, reach_from=None):
        """Nearest free planning cell (arc-length metric), as (cell centre, distance).

        With ``reach_from``, only cells in the same connected free region as
        ``reach_from`` are candidates, so a certified path exists.  Cells at
        least SNAP_CLEARANCE inside the free set are preferred, so no leg
        starts or ends in a sliver of the certified set; a waypoint whose own
        cell is free is kept (moved only to that cell's centre), and one within
        SNAP_CLEARANCE of a free cell goes to that cell.
        """
        region = self.grid if reach_from is None else self.grid_comp == self.grid_comp[self._rc(reach_from)]
        if region[self._rc(a)]:
            return self._centre(self._rc(a)), float(np.hypot(*(np.array(self._centre(self._rc(a))) - a)))
        from scipy.ndimage import distance_transform_edt

        def nearest(mask):
            rows, cols = np.nonzero(mask)
            c1 = (cols + 0.5) * PLAN_CELL
            c2 = (rows + 0.5) * PLAN_CELL
            d = np.hypot(c1 - a[0], c2 - a[1])
            k = int(np.argmin(d))
            return (float(c1[k]), float(c2[k])), float(d[k])

        near = nearest(region)
        if near[1] <= SNAP_CLEARANCE:
            return near
        deep = region & (distance_transform_edt(self.grid) * PLAN_CELL >= SNAP_CLEARANCE)
        return nearest(deep) if deep.any() else near

    def _leg(self, a0, a1):
        """A* -> corridor -> jerk-optimal Bezier inside the certified set."""
        from revised_case_studies.shared import corridor_planner as cpl
        res = cpl.plan(self.grid, self._rc(a0), self._rc(a1), (PLAN_CELL, PLAN_CELL), (0.0, 0.0),
                       vmax=(SPEED, SPEED), amax=(ACCEL, ACCEL), dt=0.05)
        if res is not None:
            self.corridors.append(res['boxes_xy'])
        return res

    # ------------------------------------------------------------ mission
    def _mission(self):
        L = [r.length for r in self.routes]
        a1_lo, a1_hi = (A1_LO, A1_HI) if self.branched else (A_LO, A_HI)
        lo = (a1_lo * L[0], (1 - A_LO) * L[1])
        hi = (a1_hi * L[0], (1 - A_HI) * L[1])
        way = [(lo, 1.0, 'start'),
               (hi, None, 'move to the end')]
        self.snapped, self.corridors = {}, []
        cached = self._cache_load() if self.plan == 'tfw' else None
        if cached is not None:
            return cached
        if self.plan == 'tfw':
            self._build_arc_grid()
            fixed, prev = [], None
            for target, hold, name in way:
                snapped, dist = self._snap(target, prev)
                if dist > PLAN_CELL:
                    self.snapped[name] = dist
                fixed.append((snapped, hold, name))
                prev = snapped
            way = fixed
        t, a1, a2 = [0.0], [way[0][0][0]], [way[0][0][1]]
        self.phases = []
        for target, hold, name in way[1:]:
            start = (a1[-1], a2[-1])
            t0 = t[-1]
            if hold is not None:
                seg = (np.linspace(0, hold, max(2, int(hold / 0.05) + 1))[1:], None)
                for tt in seg[0]:
                    t.append(t0 + tt)
                    a1.append(start[0])
                    a2.append(start[1])
            elif self.plan == 'tfw':
                res = self._leg(start, target)
                if res is None:
                    raise RuntimeError(f'{self.variant}: no certified trajectory for "{name}"')
                for tt, (x, y) in zip(res['t'][1:], res['xy'][1:]):
                    t.append(t0 + tt)
                    a1.append(x)
                    a2.append(y)
            else:
                dur = max(abs(target[k] - start[k]) for k in range(2)) / SPEED
                for f in np.linspace(0, 1, max(2, int(dur / 0.05) + 1))[1:]:
                    t.append(t0 + f * dur)
                    # each sphere moves at SPEED, then waits for the other
                    for k, arr in ((0, a1), (1, a2)):
                        span = target[k] - start[k]
                        frac = 1.0 if span == 0 else min(1.0, f * dur * SPEED / abs(span))
                        arr.append(start[k] + frac * span)
            self.phases.append((t0, t[-1], name))
        self.phases.insert(0, (0.0, 0.0, 'start'))
        mission = np.column_stack([t, a1, a2])
        if self.plan == 'tfw':
            self._cache_save(mission)
        return mission

    # ------------------------------------------------------------ plan cache
    def _cache_key(self):
        from revised_case_studies.shared.paths import RESULTS_OUT
        f = RESULTS_OUT / 'cs2' / f'tfw_{self.variant}.json'
        return {'tfw_mtime': f.stat().st_mtime, 'speed': SPEED, 'accel': ACCEL,
                'cell': PLAN_CELL, 'a_lo': A_LO, 'a_hi': A_HI, 'a1': [A1_LO, A1_HI] if self.branched else None,
                'snap': SNAP_CLEARANCE, 'version': 9,
                'geometry_sha256': hashlib.sha256((GEOM_DIR / VARIANTS[self.variant][0]).read_bytes()).hexdigest()}

    def _cache_path(self):
        from revised_case_studies.shared.paths import RESULTS_OUT
        return RESULTS_OUT / 'cs2' / f'plan_{self.variant}.json'

    def _cache_load(self):
        import json
        f = self._cache_path()
        if not f.exists():
            return None
        d = json.loads(f.read_text())
        if d.get('key') != self._cache_key():
            return None
        self.phases = [tuple(p) for p in d['phases']]
        self.snapped = d['snapped']
        self.corridors = d['corridors']
        return np.asarray(d['mission'])

    def _cache_save(self, mission):
        import json
        self._cache_path().write_text(json.dumps({
            'key': self._cache_key(), 'mission': mission.tolist(), 'phases': self.phases,
            'snapped': self.snapped, 'corridors': self.corridors}))

    def reference(self, t):
        m = self.mission
        a = np.array([np.interp(t, m[:, 0], m[:, 1]), np.interp(t, m[:, 0], m[:, 2])])
        h = 0.05
        da = np.array([np.interp(t + h, m[:, 0], m[:, k]) - np.interp(t - h, m[:, 0], m[:, k])
                       for k in (1, 2)]) / (2 * h)
        return a, da

    def phase(self, t):
        for lo, hi, name in self.phases:
            if lo <= t < hi:
                return name
        return 'done'

    @property
    def duration(self):
        return float(self.mission[-1, 0]) + 1.0


class PairController:
    """10 Hz allocation + 500 Hz plant for the two 1-DOF spheres."""

    def __init__(self, pair, interaction=True):
        self.pair = pair
        self.interaction = interaction
        lo = np.full(12, -np.inf)
        lo[[0, 3]] = K.B_MIN
        self.alloc = QPAllocator(12, i_min=-K.I_LIMIT, i_max=K.I_LIMIT, out_min=lo)
        wt = W_TRANSVERSE / K.F_SCALE
        self.w = np.array([K.W_B_MAG, K.W_B_DIR, K.W_B_DIR] * 2 + [1.0 / K.F_SCALE, wt, wt] * 2)
        u = np.asarray(K.FIELD_DIRECTION, float)
        self.u = np.tile(u / np.linalg.norm(u), (2, 1))
        self.reset()

    def reset(self):
        self.t = 0.0
        a0, _ = self.pair.reference(0.0)
        self.a = a0.copy()
        self.currents = np.zeros(10)
        self.sol = None
        self.F = np.zeros((2, 3))
        self.B = np.zeros((2, 3))
        self.ft_req = np.zeros(2)
        self.ft_alloc = np.zeros(2)
        self.ft_plant = np.zeros(2)
        self.v = np.zeros(2)

    def positions(self, a=None):
        a = self.a if a is None else a
        return np.array([self.pair.routes[k].point(a[k]) for k in range(2)])

    def allocate(self):
        a_ref, da_ref = self.pair.reference(self.t)
        p = self.positions()
        frames, hdes = [], []
        for k in range(2):
            fr = self.pair.routes[k].frame(self.a[k])
            frames.append(fr)
            v = da_ref[k] + K_P * (a_ref[k] - self.a[k])          # m/s along t
            hdes.append(np.array([K.DRAG * v, 0.0, 0.0]))
        G, c = allocation_rows(p, self.u, frames, interaction=self.interaction)
        h = np.concatenate([[K.B_HOLD, 0, 0], [K.B_HOLD, 0, 0], hdes[0], hdes[1]])
        self.sol = self.alloc.solve(G, c, h, self.w, self.currents)
        self.currents = self.sol['currents']
        self.frames = frames
        self.ft_req = np.array([hdes[0][0], hdes[1][0]])
        self.ft_alloc = self.sol['achieved'][[6, 9]]
        return a_ref

    def advance(self, dt_total):
        n = max(1, int(round(dt_total / DT_PLANT)))
        dt = dt_total / n
        for _ in range(n):
            p = self.positions()
            _, B, F = plant(p, self.currents, interaction=self.interaction)
            for k in range(2):
                rt = self.pair.routes[k]
                ft = float(F[k] @ rt.frame(self.a[k])[0])
                self.ft_plant[k] = ft
                self.v[k] = ft / K.DRAG
                self.a[k] = float(np.clip(self.a[k] + self.v[k] * dt, 0.0, rt.length))
            self.F, self.B = F, B
        self.t += dt_total

    def wall_load(self):
        """Transverse (wall-reacted) magnetic force per sphere, N."""
        out = []
        for k in range(2):
            t_hat = self.pair.routes[k].frame(self.a[k])[0]
            out.append(float(np.linalg.norm(self.F[k] - (self.F[k] @ t_hat) * t_hat)))
        return np.array(out)


def run(variant, interaction=True, plan='tfw'):
    """Headless mission run; returns time series."""
    pair = VesselPair(variant, plan)
    ctl = PairController(pair, interaction)
    log = {k: [] for k in ('t', 'a', 'a_ref', 'err', 'sep', 'I', 'wall', 'cert')}
    while ctl.t < pair.duration:
        a_ref = ctl.allocate()
        ctl.advance(DT_CTRL)
        p = ctl.positions()
        log['t'].append(ctl.t)
        log['a'].append(ctl.a.copy())
        log['a_ref'].append(a_ref)
        log['err'].append(ctl.a - a_ref)
        log['sep'].append(float(np.linalg.norm(p[0] - p[1])))
        log['I'].append(float(np.abs(ctl.currents).max()))
        log['wall'].append(ctl.wall_load())
        log['cert'].append(pair.certified(ctl.a))
    return pair, {k: np.asarray(v) for k, v in log.items()}


if __name__ == '__main__':
    for v in VARIANTS:
        for plan in ('straight', 'tfw'):
            pair, r = run(v, plan=plan)
            e = r['err'] * 1e3
            cert = np.mean([c for c in r['cert'] if c is not None]) if pair.tfw else float('nan')
            print(f'{v:22s} {plan:8s} T {pair.duration:5.1f} s  RMSE {np.sqrt((e ** 2).mean(0)).round(2)} mm  '
                  f'max {np.abs(e).max(0).round(2)} mm  min sep {r["sep"].min() * 1e3:.2f} mm  '
                  f'max|I| {r["I"].max():4.1f} A  saturated {np.mean(r["I"] > 14.999):.2f}  '
                  f'in TFW {cert:.2f}  snapped {({k: round(d * 1e3, 1) for k, d in pair.snapped.items()})}')
