"""Robot positions, frames, the archived Case 3 TFW, and phantom rendering."""

from __future__ import annotations

import json
import math
from functools import lru_cache

import numpy as np

from revised_case_studies.cs3_helical_robots import config as K
from revised_case_studies.shared.paths import RESULTS_IN

SIGN = (+1, -1)   # robot 1 path centre at +OFFSET, robot 2 at -OFFSET


def position(i, a):
    return np.array([SIGN[i] * K.OFFSET + K.R_PATH * math.cos(a),
                     K.R_PATH * math.sin(a), K.Z_PLANE])


def frame(a):
    """(tangent, normal, up) of a path point; the rotation axis is ``normal``.

    Matches ``emns.workspace`` with beta = a + pi/2: body y = (-sin b, cos b, 0).
    """
    b = a + math.pi / 2
    t = np.array([math.cos(b), math.sin(b), 0.0])
    n = np.array([-math.sin(b), math.cos(b), 0.0])
    return t, n, np.array([0.0, 0.0, 1.0])


def field_direction(a, phi):
    """Body z of the Case 3 task at roll angle ``phi`` (world frame)."""
    b = a + math.pi / 2
    return np.array([math.sin(phi) * math.cos(b), math.sin(phi) * math.sin(b), math.cos(phi)])


# ---------------------------------------------------------------- TFW raster
class Case3TFW:
    """Certified set of the archived Case 3 run in the (a1, a2) plane.

    ``results_case3.json`` lists the undetermined and out-of-workspace boxes;
    the certified-inside set is the rest of the search box (the archived
    script did not save ``in_workspace``).
    """

    def __init__(self, n=240):
        with open(RESULTS_IN / K.TFW_JSON) as f:
            data = json.load(f)
        self.blocked = np.array(data.get('undetermined', []) + data.get('out_of_workspace', []))
        self.n = n
        self.a1 = np.linspace(*K.A1_RANGE, n)
        self.a2 = np.linspace(*K.A2_RANGE, n)
        A1, A2 = np.meshgrid(self.a1, self.a2)      # rows = a2, cols = a1
        free = np.ones_like(A1, dtype=bool)
        for x0, x1, y0, y1 in self.blocked:
            free &= ~((A1 >= x0) & (A1 <= x1) & (A2 >= y0) & (A2 <= y1))
        self.free = free

    def index(self, a1, a2):
        c = int(round((a1 - K.A1_RANGE[0]) / (K.A1_RANGE[1] - K.A1_RANGE[0]) * (self.n - 1)))
        r = int(round((a2 - K.A2_RANGE[0]) / (K.A2_RANGE[1] - K.A2_RANGE[0]) * (self.n - 1)))
        return min(max(r, 0), self.n - 1), min(max(c, 0), self.n - 1)

    def angles(self, rc):
        return self.a1[rc[1]], self.a2[rc[0]]

    def contains(self, a1, a2):
        return bool(self.free[self.index(a1, a2)])


@lru_cache(maxsize=1)
def tfw():
    return Case3TFW()


# ---------------------------------------------------------------- rendering
def lumen_centerline(i, n=200, margin=0.06):
    rng = K.A1_RANGE if i == 0 else K.A2_RANGE
    a = np.linspace(rng[0] - margin, rng[1] + margin, n)
    return np.array([position(i, v) for v in a]), a


def tube_mesh(i, radius=K.LUMEN_RADIUS, n_u=160, n_v=28):
    """Surface of a slightly wavy lumen around robot ``i``'s path (metres)."""
    pts, a = lumen_centerline(i, n_u)
    v = np.linspace(0, 2 * math.pi, n_v)
    wav = 1 + 0.12 * np.sin(5 * a + i)[:, None] * np.cos(2 * v)[None, :]
    X = np.empty((n_u, n_v))
    Y = np.empty_like(X)
    Z = np.empty_like(X)
    for k, ak in enumerate(a):
        _, n, up = frame(ak)
        ring = pts[k] + radius * wav[k, :, None] * (np.cos(v)[:, None] * n + np.sin(v)[:, None] * up)
        X[k], Y[k], Z[k] = ring.T
    return X, Y, Z


def helix_curve(center, axis, phase, turns=None):
    """Points of a helix of the configured size, axis ``axis``, rotated by ``phase``."""
    turns = turns or K.HELIX_LENGTH / K.HELIX_PITCH
    s = np.linspace(-0.5, 0.5, 120)
    th = 2 * math.pi * turns * s + phase
    ref = np.array([0.0, 0.0, 1.0]) if abs(axis[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = np.cross(axis, ref)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(axis, e1)
    r = K.HELIX_RADIUS * 0.8
    return (center[None] + s[:, None] * K.HELIX_LENGTH * axis[None]
            + r * (np.cos(th)[:, None] * e1 + np.sin(th)[:, None] * e2))
