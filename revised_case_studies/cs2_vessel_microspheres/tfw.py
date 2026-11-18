#!/usr/bin/env python3
# Run from the repository root:
#   python -m revised_case_studies.cs2_vessel_microspheres.tfw [--variant NAME|all] [--eps 0.015625]
"""Certified TFW of Case Study 2 in the configuration space of the two spheres.

Configuration: q = (s1, s2), the Bezier parameter of each sphere on its own
trunk (RA for sphere 1, RV for sphere 2) of a Case 7 vessel preset, placed as
in ``vessel_pair.py`` (trunks' closest point at the coil-array centre).

Actuation requirement (the task set of the paper, per configuration):
  * net field at both spheres  B_i = B_TFW * y_hat   (fixes the aligned moments)
  * tangential force           F_i . t_i in [-F_REQ, +F_REQ], independently for
                               both spheres (4 task vertices), F_REQ = 0.5 mN
  * transverse (wall) forces   |F_i . n_i|, |F_i . b_i| <= F_SIDE
  * coil currents              |I_k| <= 15 A
The mutual dipole field and force of the two spheres are included (moments
along y_hat), as in the allocator and the plant.

Method: the affine-hybrid route cascade of Case 10
(``cases/case_10_two_magnets_vessel_crossing``), reusing its dipole and row
assembly (``_assemble``) and the certificates of ``hybrid_region``:
  inside   dependency-preserving constructive right-inverse certificate
           (``affine_constructive_inside_certificate``) with the transverse
           rows as bounded side constraints;
  outside  dual support-function separation (``dual_outside_certificate``)
           with the side rows as bounded virtual actuators.
Both are one-sided; boxes that neither certifies at the resolution ``eps``
stay undetermined.  The centerlines are enclosed with one shared affine
symbol per sphere, including the Case 7 winding term (remainder bound
|w''| <= 6 pi^2 per unit amplitude).  The commanded force frame is the frame
at the box centre; boxes are refined until the tangent varies by at most
MAX_TANGENT_DEG inside them (as in Case 10).  All rows are scaled to order
one (fields by 1/B_TFW, forces by 1/F_REQ) before certification.

Also stored: an exact pointwise feasibility grid (one LP per task vertex),
which shows which undetermined boxes are truly infeasible.  It is a
diagnosis, not a certificate.

Output: ``revised_case_studies/results/cs2/tfw_<variant>.json``.
"""

from __future__ import annotations

import argparse
import json
import hashlib
import math
import multiprocessing as mp
import time
from collections import Counter, deque
from functools import partial

import numpy as np
from scipy.optimize import linprog

from cases.case_10_two_magnets_vessel_crossing.model import _assemble, consolidate
from cases.case_10_two_magnets_vessel_crossing.route import _multiply, _scale, _square
from emns.certificates.tight_region import (
    affine_constructive_inside_certificate, dual_outside_certificate, interval_object_bounds)
from emns.numerics import affine as ca
from revised_case_studies.cs2_vessel_microspheres import config as K
from revised_case_studies.cs2_vessel_microspheres.vessel_pair import GEOM_DIR, VARIANTS, VesselPair
from revised_case_studies.shared.paths import out_dir

B_TFW = K.B_HOLD                 # T, field magnitude along +y at both spheres
# Tangential force budget: worst-case sliding friction mu*(W + sqrt(2) F_SIDE)
# = 0.46 mN plus wall-corrected drag at 3 mm/s (0.014 mN), rounded up.
F_REQ = 0.5e-3                   # N, tangential force each sphere must be able to receive
F_SIDE = 1e-3                    # N, per transverse component (~ buoyant weight) (PLACEHOLDER)
I_LIM = K.I_LIMIT
N_COILS = 10
MAX_TANGENT_DEG = 5.0
DEFAULT_EPS = 1 / 64             # box width in s (~1.6 mm along a 100 mm trunk)
POINT_GRID = 65                  # pointwise check resolution (per axis)
Y_HAT = np.array([0.0, 1.0, 0.0])
WAVE_D2 = 6 * math.pi ** 2       # bound on |d^2/ds^2 sin^2(pi s) sin(2 pi s + phi)|


# ---------------------------------------------------------------- geometry
class AffineCenterline:
    """Case 7 branch centerline with numeric and affine evaluation (metres)."""

    def __init__(self, vessel, branch):
        cfg = vessel.branches[branch]
        self.v, self.b = vessel, branch
        p = np.asarray(cfg['control_points_mm'], float) * 1e-3
        self.R = np.asarray(vessel.rotation, float)
        self.o = np.asarray(vessel.origin, float)
        self.coef = np.stack((p[0], 3 * (p[1] - p[0]), 3 * (p[0] - 2 * p[1] + p[2]),
                              -p[0] + 3 * p[1] - 3 * p[2] + p[3]))       # local power basis
        self.amp = np.asarray(cfg.get('waviness_mm', [0, 0, 0]), float) * 1e-3
        self.phase = float(cfg.get('waviness_phase', 0.0))
        self.d2_cubic = 6 * max(np.linalg.norm(p[2] - 2 * p[1] + p[0]),
                                np.linalg.norm(p[3] - 2 * p[2] + p[1]))

    def point(self, s):
        return self.v.centerline(self.b, float(s))

    def frame(self, s):
        """(t, n, b): tangent, horizontal normal, and b = t x n."""
        t = self.v.frame(self.b, float(s))[0]
        n = np.cross([0.0, 0.0, 1.0], t)
        n /= np.linalg.norm(n)
        return t, n, np.cross(t, n)

    def _wave(self, s):
        return math.sin(math.pi * s) ** 2 * math.sin(2 * math.pi * s + self.phase)

    def _dwave(self, s):
        ph = self.phase
        return (math.pi * math.sin(2 * math.pi * s) * math.sin(2 * math.pi * s + ph)
                + 2 * math.pi * math.sin(math.pi * s) ** 2 * math.cos(2 * math.pi * s + ph))

    def affine_point(self, lo, hi):
        """Three affine forms enclosing the centerline for s in [lo, hi]."""
        c, r = 0.5 * (lo + hi), 0.5 * (hi - lo)
        sym = ca.get_next_symbol_index()
        s = ca.Affine(c, {sym: r} if r else {})
        powers = (ca.Affine(1.0), s, _square(s), _multiply(_square(s), s))
        local = []
        for axis in range(3):
            form = ca.Affine(0.0)
            for pw, coef in zip(powers, self.coef):
                if coef[axis]:
                    form = form + _scale(pw, float(coef[axis]))
            a = self.amp[axis]
            if a:
                # w(s) = w(c) + w'(c)(s - c) + R,  |R| <= 0.5 max|w''| r^2
                dev = {sym: a * self._dwave(c) * r} if r else {}
                rem = abs(a) * 0.5 * WAVE_D2 * r * r
                if rem:
                    dev[ca.get_next_symbol_index()] = rem
                form = form + ca.Affine(a * self._wave(c), dev)
            local.append(form)
        world = []
        for i in range(3):
            form = ca.Affine(float(self.o[i]))
            for j in range(3):
                if self.R[i, j]:
                    form = form + _scale(local[j], float(self.R[i, j]))
            world.append(form)
        return tuple(world)

    def tangent_spread_deg(self, lo, hi, samples=9):
        ref = self.frame(0.5 * (lo + hi))[0]
        ss = np.linspace(lo, hi, samples)
        worst = max(math.acos(float(np.clip(ref @ self.frame(s)[0], -1, 1))) for s in ss)
        d = [np.linalg.norm(self.v.centerline(self.b, min(1.0, s + 1e-6))
                            - self.v.centerline(self.b, max(0.0, s - 1e-6))) / 2e-6 for s in ss]
        second = self.d2_cubic + np.linalg.norm(self.amp) * WAVE_D2
        step = (hi - lo) / (samples - 1)
        return math.degrees(worst + second * step / max(min(d), 1e-12))


# ---------------------------------------------------------------- maps
SCALE_CTRL = np.array([1 / B_TFW] * 3 + [1 / F_REQ] + [1 / B_TFW] * 3 + [1 / F_REQ])
SCALE_SIDE = np.full(4, 1 / F_REQ)
VERTICES = np.array([[0, 1, 0, f1, 0, 1, 0, f2] for f1 in (-1.0, 1.0) for f2 in (-1.0, 1.0)])
SIDE_LO, SIDE_HI = np.full(4, -F_SIDE / F_REQ), np.full(4, F_SIDE / F_REQ)


def _scaled(rows, offsets, scale):
    return [[_scale(e, float(k)) for e in row] + [_scale(off, float(k))]
            for row, off, k in zip(rows, offsets, scale)]


def affine_maps(lines, box):
    """Scaled affine ([G c], [G_side c_side]) for one (s1, s2) box."""
    (a0, a1), (b0, b1) = box
    positions = (lines[0].affine_point(a0, a1), lines[1].affine_point(b0, b1))
    frames = (lines[0].frame(0.5 * (a0 + a1)), lines[1].frame(0.5 * (b0 + b1)))
    moments = (tuple(K.MOMENT * Y_HAT), tuple(K.MOMENT * Y_HAT))
    rows, offsets, side_rows, side_offsets = _assemble(positions, moments, frames)
    full = consolidate(np.asarray(_scaled(rows, offsets, SCALE_CTRL), dtype=object))
    side = consolidate(np.asarray(_scaled(side_rows, side_offsets, SCALE_SIDE), dtype=object))
    return full, side


def inside(full, side):
    cert = affine_constructive_inside_certificate(
        full[:, :N_COILS], VERTICES, np.full(N_COILS, -I_LIM), np.full(N_COILS, I_LIM),
        c_affine=full[:, N_COILS], side_G_affine=side[:, :N_COILS],
        side_c_affine=side[:, N_COILS], side_lower=SIDE_LO, side_upper=SIDE_HI)
    return bool(cert.certified)


def outside(full, side):
    zero = ca.Affine(0.0)
    aug = [list(r[:N_COILS]) + [zero] * 4 + [r[N_COILS]] for r in full]
    for k, r in enumerate(side):
        cols = [zero] * 4
        cols[k] = ca.Affine(-1.0)
        aug.append(list(r[:N_COILS]) + cols + [r[N_COILS]])
    bounds = np.asarray([[e.to_interval() for e in row] for row in aug], dtype=object)
    G_lo, G_hi = interval_object_bounds(bounds[:, :N_COILS + 4])
    c_lo, c_hi = interval_object_bounds(bounds[:, N_COILS + 4])
    verts = np.hstack([VERTICES, np.zeros((len(VERTICES), 4))])
    lo = np.concatenate([np.full(N_COILS, -I_LIM), SIDE_LO])
    hi = np.concatenate([np.full(N_COILS, I_LIM), SIDE_HI])
    cert = dual_outside_certificate(G_lo, G_hi, verts, lo, hi, c_lower=c_lo, c_upper=c_hi)
    return bool(cert.certified)


def point_feasible(lines, s1, s2):
    """Exact pointwise test at one configuration (LP per task vertex)."""
    full, side = affine_maps(lines, ((s1, s1), (s2, s2)))
    G = np.array([[e.center for e in r] for r in full])
    Gs = np.array([[e.center for e in r] for r in side])
    A, c, As, cs = G[:, :N_COILS], G[:, N_COILS], Gs[:, :N_COILS], Gs[:, N_COILS]
    A_ub = np.vstack([As, -As])
    b_ub = np.concatenate([SIDE_HI - cs, cs - SIDE_LO])
    for v in VERTICES:
        r = linprog(np.zeros(N_COILS), A_eq=A, b_eq=v - c, A_ub=A_ub, b_ub=b_ub,
                    bounds=[(-I_LIM, I_LIM)] * N_COILS, method='highs')
        if r.status != 0:
            return False
    return True


def _point_row(s1, grid):
    return [point_feasible(_LINES, s1, s2) for s2 in grid]


# ---------------------------------------------------------------- search
_LINES = None


def _init(variant):
    global _LINES
    pair = VesselPair(variant)
    _LINES = [AffineCenterline(pair.vessel, 'RA'), AffineCenterline(pair.vessel, 'RV')]


def classify(box, eps):
    lines = _LINES
    (a0, a1), (b0, b1) = box
    leaf = max(a1 - a0, b1 - b0) <= eps
    spread = max(lines[0].tangent_spread_deg(a0, a1), lines[1].tangent_spread_deg(b0, b1))
    if spread > MAX_TANGENT_DEG and not leaf:
        return 'bisect', box, 'tangent_refinement'
    try:
        full, side = affine_maps(lines, box)
        if inside(full, side):
            return 'inside', box, 'affine_right_inverse'
        if leaf and outside(full, side):
            return 'outside', box, 'dual_separation'
    except (ArithmeticError, ValueError, np.linalg.LinAlgError):
        pass
    if leaf:
        return 'undetermined', box, 'no_certificate'
    return 'bisect', box, 'unresolved'


def _bisect(box):
    (a0, a1), (b0, b1) = box
    if a1 - a0 >= b1 - b0:
        m = 0.5 * (a0 + a1)
        return [((a0, m), (b0, b1)), ((m, a1), (b0, b1))]
    m = 0.5 * (b0 + b1)
    return [((a0, a1), (b0, m)), ((a0, a1), (m, b1))]


def determine(variant, eps=DEFAULT_EPS, processes=None):
    processes = processes or min(mp.cpu_count(), 12)
    queue = deque([((0.0, 1.0), (0.0, 1.0))])
    res = {'inside': [], 'outside': [], 'undetermined': []}
    reasons = Counter()
    t0 = time.time()
    with mp.Pool(processes, initializer=_init, initargs=(variant,)) as pool:
        while queue:
            batch = [queue.popleft() for _ in range(min(len(queue), 8 * processes))]
            for action, box, why in pool.map(partial(classify, eps=eps), batch):
                reasons[why] += 1
                if action == 'bisect':
                    queue.extend(_bisect(box))
                else:
                    res[action].append([box[0][0], box[0][1], box[1][0], box[1][1]])
        # exact pointwise check on a grid (display / diagnosis only, not a certificate)
        grid = np.linspace(0.0, 1.0, POINT_GRID)
        feasible = pool.map(partial(_point_row, grid=grid), list(grid))
    area = {k: sum((b[1] - b[0]) * (b[3] - b[2]) for b in v) for k, v in res.items()}
    return {
        'pointwise': {'grid': grid.tolist(), 'feasible_s1_s2': feasible},
        'geometry_sha256': hashlib.sha256((GEOM_DIR / VARIANTS[variant][0]).read_bytes()).hexdigest(),
        'variant': variant, 'method': 'affine_hybrid_route (Case 10 cascade)',
        'config': 's1 on RA, s2 on RV (Bezier parameters)', 'eps_s': eps,
        'requirement': {'B_T_along_y': B_TFW, 'F_req_N': F_REQ,
                        'F_side_N': F_SIDE, 'I_limit_A': I_LIM, 'moment_A_m2': K.MOMENT,
                        'viscosity_Pa_s': K.VISCOSITY, 'interaction': True},
        'area_fraction': area, 'certificate_counts': dict(reasons),
        'elapsed_s': time.time() - t0, **res,
    }


def path(variant):
    return out_dir('cs2') / f'tfw_{variant}.json'


def load(variant):
    with open(path(variant)) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--variant', default='all', choices=['all', *VARIANTS])
    ap.add_argument('--eps', type=float, default=DEFAULT_EPS)
    ap.add_argument('--processes', type=int, default=None)
    args = ap.parse_args()
    for v in (VARIANTS if args.variant == 'all' else [args.variant]):
        r = determine(v, args.eps, args.processes)
        path(v).write_text(json.dumps(r))
        a = r['area_fraction']
        print(f'{v}: inside {a["inside"]:.3f}  outside {a["outside"]:.3f}  '
              f'undetermined {a["undetermined"]:.3f}  ({r["elapsed_s"]:.0f} s)  -> {path(v)}')


if __name__ == '__main__':
    main()
