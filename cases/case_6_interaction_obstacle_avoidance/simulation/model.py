#!/usr/bin/env python3
# Run from the repository root: python -m cases.case_6_interaction_obstacle_avoidance.simulation.model
"""
case6_model.py -- Physics and workspace model behind the Case 6 interactive simulation.

Case 6 setup (identical to ``workspace_hybrid.py``):

  * A *fixed* magnet at (0.02, 0.0, 0.082) m, alpha = pi/2, beta = pi/2,
    magnetic moment m = 0.145 A m^2.  It never moves.
  * A *free* magnet in the same plane Z = 0.082 m, alpha = pi/2, moment
    m = 0.12 A m^2, with in-plane heading beta.  Its state (X, Y, beta) is the
    3-D configuration space whose certified feasible workspace (TFW) is stored
    in ``Results/results_case6_3D_hybrid.json``.
  * Ten electromagnetic coils, currents bounded to +-15 A.

With alpha = pi/2 the body frame of the free magnet is

      body z  =  ( cos beta,  sin beta,  0 )   <-- along the dipole = heading
      body y  =  (-sin beta,  cos beta,  0 )   <-- in-plane, lateral
      body x  =  ( 0,         0,        -1 )   <-- out of plane

so the four constrained rows of the actuation matrix mean

      By = 0        no in-plane torque -> the heading beta is in equilibrium
      Bz = 0.02 T   holding field along the dipole
      fy = 0        no lateral force
      fz in [-0.1, 0.1] N   thrust along the heading  <-- this is the W/S axis

That is why W/S maps onto fz and A/D maps onto beta: the certified workspace is
exactly the set of (X, Y, beta) where the full +-0.1 N of thrust is deliverable
while the other three constraints are held.
"""

import json
import math
import os
from pathlib import Path

import numpy as np
from scipy.optimize import lsq_linear

PROJECT_ROOT = Path(__file__).resolve().parents[3]

from emns.workspace.fast import Map_I2H, Map_I2H_body

# ============================================================================
# Constants -- kept consistent with workspace_hybrid.py
# ============================================================================

NUM_COILS = 10
I_MIN, I_MAX = -15.0, 15.0          # A, per coil

Z_PLANE = 0.082                     # m, both magnets live in this plane
ALPHA = math.pi / 2                 # rad, both magnets lie in the plane

FREE_M = 0.12                       # A m^2, moment of the moving magnet
FIXED_X, FIXED_Y = 0.02, 0.0        # m, position of the fixed magnet
FIXED_BETA = math.pi / 2            # rad, heading of the fixed magnet
FIXED_M = 0.145                     # A m^2, moment of the fixed magnet

# Task vector, in the free magnet's body frame: [By, Bz, fy, fz]
B_HOLD = 0.02                       # T,  required field along the dipole
F_MAX = 0.1                         # N,  required thrust authority, +-F_MAX

# Domain of the certified workspace study
XY_MIN, XY_MAX = -0.05, 0.05        # m

# Row scaling used when measuring how well a current vector meets the task.
# Turns the mixed T / N residual into a dimensionless relative error.
_ROW_SCALE = np.array([B_HOLD, B_HOLD, F_MAX, F_MAX])

ROW_LABELS = ("By", "Bz", "fy", "fz")

# ============================================================================
# Actuation matrices
# ============================================================================


def _blank_point(X, Y, Z, m, alpha, beta):
    """A target-point dict with every output component switched off."""
    return {
        'X': X, 'Y': Y, 'Z': Z, 'm': m, 'alpha': alpha, 'beta': beta,
        'Bx': None, 'By': None, 'Bz': None,
        'Bx_dx': None, 'Bx_dy': None, 'Bx_dz': None,
        'By_dy': None, 'By_dz': None,
        'fx': None, 'fy': None, 'fz': None,
        'tx': None, 'ty': None, 'tz': None,
    }


def fixed_magnet_point():
    """The stationary magnet.  No constrained outputs -- it contributes only
    through its dipole interaction with the free magnet."""
    return _blank_point(FIXED_X, FIXED_Y, Z_PLANE, FIXED_M, ALPHA, FIXED_BETA)


def _target_points(X, Y, beta, components):
    """Case 6 target-point list with ``components`` enabled on the free magnet."""
    free = _blank_point(X, Y, Z_PLANE, FREE_M, ALPHA, beta)
    for c in components:
        free[c] = True
    return [free, fixed_magnet_point()]


def actuation_matrix_body(X, Y, beta):
    """4 x 11 body-frame matrix A_hat with rows [By, Bz, fy, fz].

    Columns 0..9 are the per-ampere contributions of the ten coils; column 10
    is the contribution of the fixed magnet (its "current" is pinned to 1, as
    in ``create_I_intervals(11, custom_bounds={10: (1, 1)})``).
    """
    return Map_I2H_body(_target_points(X, Y, beta, ('By', 'Bz', 'fy', 'fz')))


def actuation_matrix_world(X, Y, beta):
    """6 x 11 world-frame matrix with rows [Bx, By, Bz, fx, fy, fz].

    Used to report the *actual* field and force produced by a current vector.
    """
    return Map_I2H(_target_points(X, Y, beta,
                                  ('Bx', 'By', 'Bz', 'fx', 'fy', 'fz')))


def task_vector(fz):
    """Body-frame task [By, Bz, fy, fz] for a commanded thrust ``fz`` (N)."""
    return np.array([0.0, B_HOLD, 0.0, float(fz)])


def body_axes(beta):
    """Unit body axes expressed in world coordinates, as (x_b, y_b, z_b)."""
    cb, sb = math.cos(beta), math.sin(beta)
    return (np.array([0.0, 0.0, -1.0]),      # body x
            np.array([-sb, cb, 0.0]),        # body y
            np.array([cb, sb, 0.0]))         # body z = dipole = heading


# ============================================================================
# Current allocation
# ============================================================================


def solve_currents(A_hat, b, i_min=I_MIN, i_max=I_MAX, feasibility_only=False):
    """Allocate coil currents that realise the body-frame task ``b``.

    Solves ``A_hat[:, :10] @ I = b - A_hat[:, 10]`` subject to
    ``i_min <= I <= i_max``.

    Strategy: a bounded least-squares solve settles feasibility and always
    returns something usable; when the exact minimum-norm solution also
    respects the bounds it is preferred, because that is the current vector an
    operator would actually command.  ``feasibility_only`` skips that second
    refinement, which roughly halves the cost when only the yes/no answer is
    wanted -- as in the grid sweep of :func:`compute_region`.

    Returns
    -------
    dict with keys
        currents   : (10,) allocated currents, A
        achieved   : (4,)  body-frame task actually produced
        residual   : (4,)  achieved - requested
        rel_error  : float infinity-norm of the residual, scaled by the task
        feasible   : bool  True when the task is met to numerical tolerance
        saturated  : bool  True when any coil sits on its current limit
        min_norm   : bool  True when the exact minimum-norm solution was used
    """
    A_hat = np.asarray(A_hat, float)
    b = np.asarray(b, float)
    A = A_hat[:, :NUM_COILS]
    rhs = b - A_hat[:, NUM_COILS]

    # Row-scale so the T-valued and N-valued rows carry comparable weight.
    W = np.diag(1.0 / _ROW_SCALE)
    sol = lsq_linear(W @ A, W @ rhs, bounds=(i_min, i_max))
    currents = sol.x
    used_min_norm = False

    # Prefer the exact minimum-norm allocation when it is admissible.
    if not feasibility_only:
        i_mn = np.linalg.pinv(A) @ rhs
        if np.max(np.abs(i_mn)) <= i_max + 1e-9 and np.min(i_mn) >= i_min - 1e-9:
            if np.max(np.abs((A @ i_mn - rhs) / _ROW_SCALE)) < 1e-9:
                currents = i_mn
                used_min_norm = True

    achieved = A_hat @ np.append(currents, 1.0)
    residual = achieved - b
    rel_error = float(np.max(np.abs(residual / _ROW_SCALE)))

    return {
        'currents': currents,
        'achieved': achieved,
        'residual': residual,
        'rel_error': rel_error,
        'feasible': rel_error < 1e-6,
        'saturated': bool(np.max(np.abs(currents)) > i_max - 1e-3),
        'min_norm': used_min_norm,
    }


def field_and_force(X, Y, beta, currents):
    """World-frame B (T) and F (N) at the free magnet for a current vector.

    Includes the coils *and* the fixed magnet's dipole contribution.
    """
    A_w = actuation_matrix_world(X, Y, beta)
    h = A_w @ np.append(np.asarray(currents, float), 1.0)
    return h[:3], h[3:]


# ============================================================================
# Point feasibility (exact, for the live recomputation)
# ============================================================================


def feasible_point(X, Y, beta, i_min=I_MIN, i_max=I_MAX, tol=1e-6):
    """Is (X, Y, beta) in the feasible workspace?

    The reachable task set is the image of a box of currents under a linear
    map, hence convex, and the required task set is the segment
    ``[By, Bz, fy, fz] = [0, B_HOLD, 0, t]`` for ``t`` in ``[-F_MAX, F_MAX]``.
    A convex set contains a segment iff it contains both endpoints, so testing
    the two extreme thrusts is exact.
    """
    A_hat = actuation_matrix_body(X, Y, beta)
    for fz in (-F_MAX, F_MAX):
        r = solve_currents(A_hat, task_vector(fz), i_min, i_max,
                           feasibility_only=True)
        if r['rel_error'] >= tol:
            return False
    return True


def compute_region(beta, n=61, x_min=XY_MIN, x_max=XY_MAX,
                   y_min=XY_MIN, y_max=XY_MAX, progress=None):
    """Sweep an ``n`` x ``n`` grid and return (mask, extent) for one heading.

    ``mask`` is a boolean array indexed [row=y, col=x] suitable for imshow with
    ``origin='lower'`` and the returned ``extent``.
    """
    xs = np.linspace(x_min, x_max, n)
    ys = np.linspace(y_min, y_max, n)
    mask = np.zeros((n, n), dtype=bool)
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            mask[j, i] = feasible_point(x, y, beta)
        if progress is not None:
            progress((j + 1) / n)
    dx = (x_max - x_min) / (n - 1) / 2.0
    dy = (y_max - y_min) / (n - 1) / 2.0
    return mask, (x_min - dx, x_max + dx, y_min - dy, y_max + dy)


# ============================================================================
# Certified workspace (the precomputed 3-D TFW)
# ============================================================================

DEFAULT_TFW_JSON = str(PROJECT_ROOT / 'Results' / 'results_case6_3D_hybrid.json')


class TFWSlice:
    """Slices the affine-hybrid certified 3-D TFW at a given heading.

    The JSON stores axis-aligned boxes ``[xmin, xmax, ymin, ymax, bmin, bmax]``
    in the ``in_workspace`` list.  Slicing at a heading keeps every box whose
    beta-span contains it; the surviving rectangles are rasterised onto a fine
    grid so that membership queries during the simulation loop are O(1).
    """

    def __init__(self, path=DEFAULT_TFW_JSON, raster=400):
        with open(path, 'r') as f:
            data = json.load(f)
        self.path = path
        self.boxes = np.array(data.get('in_workspace', []), dtype=float)
        self.undetermined = np.array(data.get('undetermined', []), dtype=float)
        if self.boxes.size == 0:
            raise ValueError(f'No in_workspace boxes found in {path}')

        self.raster = raster
        self.x_min, self.x_max = XY_MIN, XY_MAX
        self.y_min, self.y_max = XY_MIN, XY_MAX
        self.extent = (self.x_min, self.x_max, self.y_min, self.y_max)

        # Every certified box begins and ends on a uniform beta grid whose step
        # is the narrowest box (2*pi/64 for the shipped result), so quantising
        # beta onto that grid selects exactly the same boxes as beta itself.
        # That makes the per-level cache exact and bounded to n_levels entries.
        widths = np.round(self.boxes[:, 5] - self.boxes[:, 4], 9)
        self.beta_step = float(widths.min())
        self.n_levels = int(round(2.0 * math.pi / self.beta_step))

        # Rasterising all levels up front costs a fraction of a second and
        # turns every later query -- slicing, membership, and "which headings
        # are feasible right here" -- into plain array indexing.
        self.stack = np.stack([self._rasterise_level(k)
                               for k in range(self.n_levels)])
        # Per-cell count of feasible headings, out of n_levels.
        self.heading_freedom = self.stack.sum(axis=0)
        self._mask = None

    # -- slicing ------------------------------------------------------------

    def _rasterise(self, rects):
        """Burn a set of [x0, x1, y0, y1] rectangles into a boolean grid."""
        n = self.raster
        mask = np.zeros((n, n), dtype=bool)
        sx = n / (self.x_max - self.x_min)
        sy = n / (self.y_max - self.y_min)
        for x0, x1, y0, y1 in rects:
            i0 = max(0, int(math.floor((x0 - self.x_min) * sx)))
            i1 = min(n, int(math.ceil((x1 - self.x_min) * sx)))
            j0 = max(0, int(math.floor((y0 - self.y_min) * sy)))
            j1 = min(n, int(math.ceil((y1 - self.y_min) * sy)))
            if i1 > i0 and j1 > j0:
                mask[j0:j1, i0:i1] = True
        return mask

    def beta_level(self, beta):
        """Index of the beta grid cell containing ``beta``."""
        lvl = int(math.floor((beta % (2.0 * math.pi)) / self.beta_step))
        return min(max(lvl, 0), self.n_levels - 1)

    def _rasterise_level(self, k):
        # Sample at the centre of the level: box edges coincide with the grid,
        # so any interior point selects the same set of boxes.
        b = (k + 0.5) * self.beta_step
        sel = (self.boxes[:, 4] <= b) & (b <= self.boxes[:, 5])
        return self._rasterise(self.boxes[sel][:, :4])

    def slice_at(self, beta):
        """Boolean raster of the feasible (x, y) set at heading ``beta``."""
        return self.stack[self.beta_level(beta)]

    def cell_of(self, x, y):
        """Raster indices (i, j) for a point, or None if outside the domain."""
        if not (self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max):
            return None
        n = self.raster
        i = min(max(int((x - self.x_min) / (self.x_max - self.x_min) * n), 0), n - 1)
        j = min(max(int((y - self.y_min) / (self.y_max - self.y_min) * n), 0), n - 1)
        return i, j

    def headings_at(self, x, y):
        """Boolean array over the beta levels: which headings are certified
        feasible at this position.  All False outside the domain."""
        cell = self.cell_of(x, y)
        if cell is None:
            return np.zeros(self.n_levels, dtype=bool)
        i, j = cell
        return self.stack[:, j, i]

    def beta_levels(self):
        """Sorted unique beta breakpoints present in the certified boxes."""
        return np.unique(np.round(self.boxes[:, 4], 9))

    # -- queries ------------------------------------------------------------

    def set_beta(self, beta):
        self._mask = self.slice_at(beta)
        return self._mask

    def contains(self, x, y, mask=None):
        """Is (x, y) inside the sliced feasible set?"""
        m = self._mask if mask is None else mask
        if m is None:
            return False
        cell = self.cell_of(x, y)
        if cell is None:
            return False
        i, j = cell
        return bool(m[j, i])

    def any_feasible(self, beta):
        return bool(self.slice_at(beta).any())

    def find_seed(self, beta):
        """A comfortable starting pose at heading ``beta``.  None if empty.

        Picks the feasible cell with the most feasible headings, so the magnet
        does not start pinned against the boundary in orientation.
        """
        mask = self.slice_at(beta)
        if not mask.any():
            return None
        js, is_ = np.nonzero(mask)
        n = self.raster
        xs = self.x_min + (is_ + 0.5) / n * (self.x_max - self.x_min)
        ys = self.y_min + (js + 0.5) / n * (self.y_max - self.y_min)
        free = self.heading_freedom[js, is_].astype(float)
        # Prefer heading freedom, then break ties toward the domain centre.
        score = free - 0.02 * np.hypot(xs, ys) / (self.x_max - self.x_min)
        k = int(np.argmax(score))
        return float(xs[k]), float(ys[k])


if __name__ == '__main__':
    # Smoke test: report the model at one pose and cross-check the certified
    # slice against the exact point test.
    np.set_printoptions(precision=4, suppress=True)
    beta = math.pi / 4
    tfw = TFWSlice()
    seed = tfw.find_seed(beta)
    print(f'certified boxes: {len(tfw.boxes)}   seed at beta=pi/4: {seed}')

    x, y = seed
    A_hat = actuation_matrix_body(x, y, beta)
    res = solve_currents(A_hat, task_vector(F_MAX))
    B, F = field_and_force(x, y, beta, res['currents'])
    print(f'currents (A): {res["currents"]}')
    print(f'achieved [By, Bz, fy, fz] = {res["achieved"]}  rel err {res["rel_error"]:.2e}')
    print(f'B world (mT): {B * 1e3}    F world (N): {F}')
    print(f'exact point feasibility: {feasible_point(x, y, beta)}')
