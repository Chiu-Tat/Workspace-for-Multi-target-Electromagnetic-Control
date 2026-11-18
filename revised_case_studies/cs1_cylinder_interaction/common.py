"""Case Study 1 helpers built on the archived Case 6 model and TFW.

Nothing here computes a new TFW.  The certified sets are the sharp 3-D
affine-hybrid results (corrected dBy/dy, 0.781 mm x 5.625 deg) for the three
Case 6 fixed-magnet layouts (``results/cs1/tfw3d_fine/tfw3d_<layout>.json``,
via ``TFWSlice``);
single configurations are checked pointwise with the exact Case 6 test.
Every helper defaults to the original ``single`` layout.
"""

from __future__ import annotations

import json
import math
from functools import lru_cache

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.optimize import linprog

from cases.case_6_interaction_obstacle_avoidance.simulation import model as m6
from revised_case_studies.shared.paths import RESULTS_IN, out_dir

# Re-exported so the scripts read like the paper.
TFWSlice = m6.TFWSlice
actuation_matrix_body = m6.actuation_matrix_body
actuation_matrix_world = m6.actuation_matrix_world
feasible_point = m6.feasible_point
B_HOLD, F_MAX = m6.B_HOLD, m6.F_MAX
FIXED_XY = (m6.FIXED_X, m6.FIXED_Y)
FIXED_BETA = m6.FIXED_BETA
I_MIN, I_MAX = m6.I_MIN, m6.I_MAX
NUM_COILS = m6.NUM_COILS

# Fixed-magnet layouts of archive Case 6 (x m, y m, beta rad, m A m^2), all in
# the plane z = 82 mm, each with its own existing affine-hybrid TFW result.
from cases.case_6_interaction_obstacle_avoidance.workspace_hybrid import LAYOUTS  # noqa: E402
# Archived results (earlier dBy/dy coefficient), kept for the historical record.
ARCHIVED_TFW = {'single': 'results_case6_3D_hybrid.json',
                'gate': 'results_case6_3D_hybrid_gate.json',
                'triangle': 'results_case6_3D_hybrid_triangle.json'}
# Coarse 3-D TFWs with the corrected derivative (run_tfw3d_corrected.py): same
# task, layouts and resolution (3.125 mm, 5.625 deg) as the archived runs.
COARSE_TFW_DIR = out_dir('cs1') / 'tfw3d_corrected'
# Sharp 3-D TFWs (run_tfw3d_fine.py): 0.781 mm x 5.625 deg.  Used by every helper.
TFW_DIR = out_dir('cs1') / 'tfw3d_fine'
LAYOUT_TFW = {k: f'tfw3d_{k}.json' for k in ARCHIVED_TFW}

# Single-layout references shared by the deck, the paper figure and the
# simulators; written by presentation.py (run_missions).  The archived Case 6
# trajectories were planned on the archived TFW and are kept for the record.
PLANNED = out_dir('cs1') / 'presentation' / 'trajectory_single_planned.json'
NAIVE = out_dir('cs1') / 'presentation' / 'trajectory_single_straight.json'
ARCHIVED_PLANNED = 'trajectory_case6_3D_02.json'
ARCHIVED_NAIVE = 'trajectory_case6_3D_no_obstacles.json'

NEAR_BOUNDARY_M = 0.003    # "near boundary" = certified, within 3 mm of the edge
MAG_LEN, MAG_WID = 0.009, 0.005                  # drawn magnet size, m
CONTACT_DIST = 0.009       # centre distance treated as contact with the fixed magnet


def load_trajectory(name):
    """(N, 4) array of [x, y, beta, t]; a bare name is an archived Case 6 file."""
    from pathlib import Path
    path = Path(name) if Path(name).is_absolute() else RESULTS_IN / name
    with open(path) as f:
        data = json.load(f)
    return np.asarray(data['trajectory'], float)


@lru_cache(maxsize=None)
def tfw(layout='single'):
    return TFWSlice(str(TFW_DIR / LAYOUT_TFW[layout]))


def fixed_magnets(layout='single'):
    """[(x, y, beta, m)] of the fixed magnets of ``layout``."""
    return LAYOUTS[layout]


def _points(x, y, beta, comps, layout):
    free = m6._blank_point(x, y, m6.Z_PLANE, m6.FREE_M, m6.ALPHA, beta)
    for k in comps:
        free[k] = True
    fixed = [m6._blank_point(fx, fy, m6.Z_PLANE, fm, m6.ALPHA, fb)
             for fx, fy, fb, fm in LAYOUTS[layout]]
    return [free] + fixed


def body_matrix(x, y, beta, layout='single'):
    """4 x 11 [By, Bz, fy, fz]; column 10 = summed effect of all fixed magnets."""
    if layout == 'single':
        return actuation_matrix_body(x, y, beta)
    return m6.Map_I2H_body(_points(x, y, beta, ('By', 'Bz', 'fy', 'fz'), layout))


def world_matrix(x, y, beta, layout='single'):
    """6 x 11 [Bx, By, Bz, fx, fy, fz] in the world frame."""
    if layout == 'single':
        return actuation_matrix_world(x, y, beta)
    return m6.Map_I2H(_points(x, y, beta, ('Bx', 'By', 'Bz', 'fx', 'fy', 'fz'), layout))


def feasible_layout(x, y, beta, layout='single', tol=1e-6):
    """Exact pointwise test (both thrust extremes) for any layout."""
    if layout == 'single':
        return feasible_point(x, y, beta)
    A_hat = body_matrix(x, y, beta, layout)
    return all(m6.solve_currents(A_hat, m6.task_vector(fz), feasibility_only=True)['rel_error'] < tol
               for fz in (-F_MAX, F_MAX))


def nearest_fixed(x, y, layout='single'):
    """(distance, index) of the closest fixed magnet."""
    d = [math.dist((x, y), (fx, fy)) for fx, fy, _, _ in LAYOUTS[layout]]
    k = int(np.argmin(d))
    return d[k], k


@lru_cache(maxsize=None)
def _edge_distance(level, layout='single'):
    """Distance (m) from each certified raster cell to the nearest non-certified cell."""
    t = tfw(layout)
    cell = (t.x_max - t.x_min) / t.raster
    return distance_transform_edt(t.stack[level]) * cell


def label(x, y, beta, near=NEAR_BOUNDARY_M, check_outside=True, layout='single'):
    """Classify one configuration.

    inside       certified, at least ``near`` from the certified edge
    boundary     certified, closer than ``near`` to the edge
    uncertified  not certified, but the exact pointwise test passes
    infeasible   the exact pointwise test fails (task not reachable)
    """
    t = tfw(layout)
    cell = t.cell_of(x, y)
    lvl = t.beta_level(beta)
    if cell is not None and t.stack[lvl][cell[1], cell[0]]:
        d = _edge_distance(lvl, layout)[cell[1], cell[0]]
        return 'inside' if d >= near else 'boundary'
    if not check_outside:
        return 'uncertified'
    return 'uncertified' if feasible_layout(x, y, beta, layout) else 'infeasible'


def _lp_range(c_obj, A_eq, b_eq):
    bounds = [(I_MIN, I_MAX)] * NUM_COILS
    lo = linprog(c_obj, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    hi = linprog(-c_obj, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    if lo.status != 0 or hi.status != 0:
        return (math.nan, math.nan)
    return (float(lo.fun), float(-hi.fun))


def force_ranges(x, y, beta, layout='single'):
    """Total body-frame force ranges reachable while holding the field task.

    Returns a dict with
      fz_range        [min, max] thrust with By = 0, Bz = B_HOLD, fy = 0
      fy_range        [min, max] lateral force with By = 0, Bz = B_HOLD
      fz_interaction  thrust contributed by the fixed magnet alone
      fy_interaction  lateral force contributed by the fixed magnet alone
    Ranges are NaN when the field/lateral constraints themselves cannot be met.
    These are the quantities of Fig. Case_6 panels (e), (f), (j), (k).
    """
    A_hat = body_matrix(x, y, beta, layout)
    A, c = A_hat[:, :NUM_COILS], A_hat[:, NUM_COILS]
    target = np.array([0.0, B_HOLD, 0.0])
    fz = _lp_range(A[3], A[:3], target - c[:3])
    fy = _lp_range(A[2], A[:2], target[:2] - c[:2])
    return {
        'fz_range': (fz[0] + c[3], fz[1] + c[3]),
        'fy_range': (fy[0] + c[2], fy[1] + c[2]),
        'fz_interaction': float(c[3]),
        'fy_interaction': float(c[2]),
    }


def magnet_polygon(x, y, beta, length=MAG_LEN, width=MAG_WID):
    """Rectangle outline of a cylindrical magnet seen from above."""
    u = np.array([math.cos(beta), math.sin(beta)])
    v = np.array([-u[1], u[0]])
    c = np.array([x, y])
    return np.array([c + a * u * length / 2 + b * v * width / 2
                     for a, b in ((1, 1), (-1, 1), (-1, -1), (1, -1))])


def draw_magnet(ax, x, y, beta, color, north='#c8553d', **kw):
    """Two-tone magnet (north half red) so the heading is readable."""
    from matplotlib.patches import Polygon
    u = np.array([math.cos(beta), math.sin(beta)])
    back = magnet_polygon(x - u[0] * MAG_LEN / 4, y - u[1] * MAG_LEN / 4, beta,
                          MAG_LEN / 2)
    front = magnet_polygon(x + u[0] * MAG_LEN / 4, y + u[1] * MAG_LEN / 4, beta,
                           MAG_LEN / 2)
    ax.add_patch(Polygon(back, closed=True, fc=color, ec='k', lw=0.6, **kw))
    ax.add_patch(Polygon(front, closed=True, fc=north, ec='k', lw=0.6, **kw))
