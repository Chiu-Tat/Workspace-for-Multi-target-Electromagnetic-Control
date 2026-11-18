"""Real geometry of the ten-coil array, for drawing it around the robots.

Source: ``Modeling/Consistency`` (``coil pose.txt``, ``plot_coil_array.py``).
Each coil is an iron core (diameter 45 mm, length 180 mm) with a winding
(inner diameter 45 mm, outer diameter 105 mm, length 120 mm) that starts
20 mm behind the core tip.  The pose file gives, per coil, the unit axis
pointing toward the workspace and the core-tip position in mm.

Frame registration.  The pose file uses the array centre as origin; the
calibrated dipole model (``emns.workspace.fast.params_list``) and all case
studies use the eMNS frame, whose x-y origin is the array axis.  Fitting the
calibrated dipole positions to ``tip - d_i * axis + t`` gives a robust
vertical offset, t_z = 70.2 mm (69.4-72.8 mm over coil subsets; dipoles
25-42 mm behind the tips).  The fitted horizontal offset is not robust
(x from -4.7 to -1.8 mm, y from -2.1 to +0.9 mm depending on which coils are
used; the calibrated dipole directions differ from the pose axes by 5-31 deg
and the vertical coil's dipole sits 7 mm off its axis), i.e. it is calibration
noise, so it is set to zero: t = (0, 0, 70.2) mm.  Drawn coils and simulated
robots therefore share one frame with the array axis at x = y = 0.
"""

from __future__ import annotations

import numpy as np

AXES = np.array([
    [-1, 0, 0], [-0.5, 0.866025, 0], [0.5, 0.866025, 0], [1, 0, 0],
    [0.5, -0.866025, 0], [-0.5, -0.866025, 0], [0, 0.766045, 0.642787],
    [0.663414, -0.383022, 0.642787], [-0.663414, -0.383022, 0.642787], [0, 0, 1]], float)
TIPS_POSE_MM = np.array([
    [85, 0, 0], [42.5, -73.612, 0], [-42.5, -73.612, 0], [-85, 0, 0],
    [-42.5, 73.612, 0], [42.5, 73.612, 0], [0, -65.1138, -54.6369],
    [-56.3902, 32.5569, -54.6369], [56.3902, 32.5569, -54.6369], [0, 0, -85]], float)
POSE_TO_EMNS_MM = np.array([0.0, 0.0, 70.2])

CORE_R, CORE_LEN = 22.5, 180.0
WIND_R_IN, WIND_R_OUT = 22.5, 52.5
WIND_START, WIND_LEN = 20.0, 120.0

C_CORE = '#b9b4ab'
C_CORE_EDGE = '#6f6a62'
C_WIND = '#c8773f'
C_WIND_EDGE = '#7a3f1b'

TIPS_MM = TIPS_POSE_MM + POSE_TO_EMNS_MM          # eMNS frame, mm
_VIEW = {'xy': (0, 1, 2), 'xz': (0, 2, 1), 'yz': (1, 2, 0)}   # (u, v, depth)


def _ring(center, axis, r, n=48):
    ref = np.array([0.0, 0.0, 1.0]) if abs(axis[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = np.cross(axis, ref)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(axis, e1)
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return center + r * (np.cos(th)[:, None] * e1 + np.sin(th)[:, None] * e2)


def _silhouette(p0, p1, axis, r, view):
    from scipy.spatial import ConvexHull
    u, v, d = _VIEW[view]
    pts = np.vstack([_ring(p0, axis, r), _ring(p1, axis, r)])
    uv = pts[:, [u, v]]
    hull = ConvexHull(uv)
    return uv[hull.vertices], float(pts[:, d].mean())


def cylinders(view):
    """(polygon_mm, depth_mm, kind, coil_index) for every core and winding."""
    out = []
    for k in range(10):
        a, tip = AXES[k] / np.linalg.norm(AXES[k]), TIPS_MM[k]
        core = _silhouette(tip, tip - CORE_LEN * a, a, CORE_R, view)
        wind = _silhouette(tip - WIND_START * a, tip - (WIND_START + WIND_LEN) * a, a,
                           WIND_R_OUT, view)
        out.append((wind[0], wind[1], 'winding', k))
        out.append((core[0], core[1] + 1e-3, 'core', k))
    return out


def draw(ax, view='xy', scale=1.0, alpha=0.9, labels=True, zorder=0.5, fade_below=None):
    """Draw the coil silhouettes projected on ``view`` (axes units = mm / scale).

    Silhouettes are painted far-to-near (depth = the dropped coordinate:
    z for the top view, y for 'xz', x for 'yz').  Anything outside the axes
    limits is clipped, so a zoomed view shows only the coil ends that reach
    into it.  ``fade_below`` (mm, depth coordinate) lightens coils lying
    below the robots' plane in a top view.
    """
    from matplotlib.patches import Polygon
    items = cylinders(view)
    sign = 1.0 if view == 'xy' else -1.0            # top view: larger z is nearer
    items.sort(key=lambda it: sign * it[1])
    for poly, depth, kind, k in items:
        a = alpha
        if fade_below is not None and depth < fade_below:
            a *= 0.22
        fc, ec = (C_WIND, C_WIND_EDGE) if kind == 'winding' else (C_CORE, C_CORE_EDGE)
        ax.add_patch(Polygon(poly / scale, closed=True, fc=fc, ec=ec, lw=0.6, alpha=a,
                             zorder=zorder))
    if labels:
        u, v, _ = _VIEW[view]
        for k in range(10):
            p = (TIPS_MM[k] - 8 * AXES[k])[[u, v]] / scale
            ax.text(p[0], p[1], str(k + 1), fontsize=7, color='white', ha='center',
                    va='center', zorder=zorder + 0.01, fontweight='bold', clip_on=True)
