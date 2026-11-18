"""Vessel phantom for Case Study 2, built on the Case 7 geometry API.

Provides the sphere routes as dense polylines, wall-contact projection, and
realistic renderings (top view with a tissue-like background, and the 3-D
lumen mesh).  Geometry only; no magnetics here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from cases.case_7_vessel_microspheres.vessel import Vessel
from revised_case_studies.cs2_vessel_microspheres import config as K
from revised_case_studies.shared.paths import ROOT


class Phantom:
    def __init__(self, geometry=K.GEOMETRY, samples=241):
        self.vessel = Vessel(geometry)
        self.samples = samples
        s = np.linspace(0, 1, samples)
        self.center = {b: self.vessel.centerline(b, s) for b in self.vessel.branches}
        self.radius = {b: self.vessel.radius(b, s) for b in self.vessel.branches}

    # -- routes --------------------------------------------------------------
    def route(self, segments, step=2e-4):
        """Dense polyline (N, 3), branch name per point, and arc length."""
        pts, names = [], []
        for branch, s0, s1 in segments:
            n = max(2, int(abs(s1 - s0) * 400))
            s = np.linspace(s0, s1, n)
            p = self.vessel.centerline(branch, s)
            if pts:
                p, s = p[1:], s[1:]
            pts.append(p)
            names += [branch] * len(p)
        p = np.concatenate(pts)
        arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])
        return p, np.array(names), arc

    # -- wall contact --------------------------------------------------------
    def nearest(self, p, branches=None):
        best = (np.inf, None, None, None)
        for b in (branches or self.center):
            d = np.linalg.norm(self.center[b] - p, axis=1)
            k = int(np.argmin(d))
            if d[k] < best[0]:
                best = (d[k], b, k, self.center[b][k])
        return best

    def project_inside(self, p, particle_radius, branches=None):
        """Push ``p`` back inside the lumen if the sphere would cross the wall.

        Uses the nearest generating ball of the swept-ball model, which is the
        conservative interior certificate of Case 7.
        """
        clear = self.vessel.clearance(p[None], count=self.samples)[0]
        if clear >= particle_radius:
            return p, False
        d, b, k, c = self.nearest(p, branches)
        r = self.radius[b][k] - particle_radius
        v = p - c
        n = np.linalg.norm(v)
        if n < 1e-12:
            return c.copy(), True
        return c + v / n * max(r, 0.0), True

    def constrain_velocity(self, p, v, particle_radius, branches=None, tol=1e-6):
        """Remove the outward wall-normal velocity of a sphere touching the wall.

        The wall reacts normal load only (frictionless contact); motion along
        the wall and away from it is unchanged.  Returns (v, in_contact).
        """
        d, b, k, c = self.nearest(p, branches)
        r = self.radius[b][k] - particle_radius
        t, _, _ = self.vessel.frame(b, k / (self.samples - 1))
        radial = (p - c) - ((p - c) @ t) * t
        n = np.linalg.norm(radial)
        if n < 1e-12 or n < r - tol:
            return v, False
        n_out = radial / n
        vn = v @ n_out
        return (v - vn * n_out if vn > 0 else v), True

    def flow_velocity(self, p, mean_speed, branches=None):
        """Provisional branchwise Poiseuille velocity (Case 7 THEORY.md, Sec. 3).

        Directed along increasing s, which is downstream in every branch of
        the carotid preset.  Not a CFD solution at the bifurcation.
        """
        if mean_speed == 0.0:
            return np.zeros(3)
        d, b, k, c = self.nearest(p, branches)
        r = self.radius[b][k]
        t, _, _ = self.vessel.frame(b, k / (self.samples - 1))
        return 2.0 * mean_speed * max(0.0, 1.0 - (d / r) ** 2) * t

    def tangent_frame(self, p, branches=None):
        """(t, n, b) of the nearest centerline sample."""
        _, b, k, _ = self.nearest(p, branches)
        s = k / (self.samples - 1)
        return self.vessel.frame(b, s)

    # -- rendering -------------------------------------------------------------
    def draw_top(self, ax, tissue=True):
        """Top view (x, y) in mm with a tissue-like background."""
        if tissue:
            xs = np.linspace(-0.03, 0.03, 300)
            ys = np.linspace(-0.035, 0.035, 350)
            X, Y = np.meshgrid(xs, ys)
            rng = np.random.default_rng(3)
            noise = rng.normal(size=X.shape)
            from scipy.ndimage import gaussian_filter
            tex = gaussian_filter(noise, 6) * 4 + gaussian_filter(noise, 1.5) * 0.6
            ax.imshow(tex, extent=(xs[0] * 1e3, xs[-1] * 1e3, ys[0] * 1e3, ys[-1] * 1e3),
                      origin='lower', cmap='pink', alpha=0.55, vmin=-3, vmax=3,
                      interpolation='bilinear', zorder=0)
        for b in self.center:
            c, r = self.center[b], self.radius[b]
            for k in range(0, len(c), 2):
                ax.add_patch(_disk(c[k, :2] * 1e3, (r[k] + 0.35e-3) * 1e3, '#8c2d2d', 1))
        for b in self.center:
            c, r = self.center[b], self.radius[b]
            for k in range(0, len(c), 2):
                ax.add_patch(_disk(c[k, :2] * 1e3, r[k] * 1e3, '#e7a5a0', 2))
        for b in self.center:
            c = self.center[b]
            ax.plot(c[:, 0] * 1e3, c[:, 1] * 1e3, ':', color='#8c2d2d', lw=0.6, zorder=3)
            lab = self.vessel.branches[b].get('label', b).split('|')[0].strip()
            ax.annotate(lab, c[-1, :2] * 1e3, xytext=(4, 4), textcoords='offset points',
                        fontsize=7, color='#5a1e1e')
        ax.set_aspect('equal')
        ax.set_xlabel('x (mm)')
        ax.set_ylabel('y (mm)')

    def draw_3d(self, ax, mesh=K.LUMEN_MESH, max_faces=12000):
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        V, F = load_obj(ROOT / mesh)
        if len(F) > max_faces:
            F = F[np.random.default_rng(0).choice(len(F), max_faces, replace=False)]
        tri = V[F] * 1e3
        n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
        light = np.clip(n @ np.array([0.3, -0.4, 0.85]), 0, 1)
        col = np.column_stack([0.75 + 0.2 * light, 0.35 + 0.3 * light,
                               0.33 + 0.3 * light, np.full_like(light, 0.25)])
        ax.add_collection3d(Poly3DCollection(tri, facecolors=col, linewidths=0))
        lo, hi = tri.reshape(-1, 3).min(0), tri.reshape(-1, 3).max(0)
        mid, span = (lo + hi) / 2, (hi - lo).max() / 2
        for setter, m in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), mid):
            setter(m - span, m + span)
        ax.set_xlabel('x (mm)')
        ax.set_ylabel('y (mm)')
        ax.set_zlabel('z (mm)')


def _disk(center, r, color, z):
    from matplotlib.patches import Circle
    return Circle(center, r, color=color, lw=0, zorder=z)


def load_obj(path):
    V, F = [], []
    for line in Path(path).read_text().splitlines():
        if line.startswith('v '):
            V.append([float(v) for v in line.split()[1:4]])
        elif line.startswith('f '):
            F.append([int(v.split('/')[0]) - 1 for v in line.split()[1:4]])
    return np.asarray(V), np.asarray(F)
