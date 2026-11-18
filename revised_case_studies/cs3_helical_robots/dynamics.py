"""Field allocation and rotation/rolling dynamics for the two helical robots.

Allocation at the field-update rate (= plant rate, 500 Hz; a rotating field
cannot be synthesised at the 10 Hz feedback rate): world rows [B_1; B_2]
(6 rows, 10 coils) by the minimum-norm solution, falling back to bounded
weighted least squares (``shared.allocator.fallback_allocate``) when a coil
would exceed its limit; the residual is the reported slack.  The
requested field at robot i is B_ROT * field_direction(a_i, phi_i).  As in the
archived Case 3 task, the other robot's dipole field is an offset column
(its moment assumed synchronous with its commanded phase).

Plant (500 Hz), per robot, synchronous-rotor model:

    ROT_DRAG * dtheta/dt = m |B_plane| sin(psi - theta)
    da/dt = ROLL_EFFICIENCY * HELIX_RADIUS * dtheta/dt / R_PATH   (free mode)

where psi is the angle of the actual field in the robot's rotation plane
(tangent, up).  When the commanded rate exceeds m B / ROT_DRAG the robot
steps out: theta lags, slips and the mean rotation rate collapses.
"""

from __future__ import annotations

import math

import numpy as np

from cases.case_7_vessel_microspheres.magnetics import coil_basis
from revised_case_studies.cs2_vessel_microspheres.model import dipole_field
from revised_case_studies.cs3_helical_robots import config as K
from revised_case_studies.cs3_helical_robots.geometry import field_direction, frame, position
from revised_case_studies.shared.allocator import fallback_allocate


def field_matrix(a, roll):
    """World field rows at both robots: G (6 x 10) per ampere and c (6), the
    field of the other robot whose moment points along its roll angle.

    Same coil dipole parameters as ``emns.workspace`` (via the vectorised
    Case 7 ``coil_basis``); the offset matches the pinned interaction column
    of the archived Case 3 task.
    """
    p = [position(i, a[i]) for i in range(2)]
    G = np.vstack([coil_basis(p[i])[0] for i in range(2)])
    m = [K.MOMENT * field_direction(a[i], roll[i]) for i in range(2)]
    c = np.concatenate([dipole_field(p[i] - p[1 - i], m[1 - i]) for i in range(2)])
    return G, c


class FieldController:
    """Minimum-norm field allocation with a bounded fallback."""

    def __init__(self):
        self.w = np.full(6, 1.0 / K.B_ROT)
        self._a = None

    def currents(self, a, phi):
        if self._a is None or np.max(np.abs(a - self._a)) > 1e-4:
            self.G, _ = field_matrix(a, phi)
            self.G_pinv = np.linalg.pinv(self.G)
            self._a = np.array(a, float)
        _, c = field_matrix(a, phi)
        h = np.concatenate([K.B_ROT * field_direction(a[i], phi[i]) for i in range(2)])
        I = self.G_pinv @ (h - c)
        if np.max(np.abs(I)) <= K.I_LIMIT:
            return I, {'slack': np.zeros(6), 'saturated': False, 'currents': I}
        sol = fallback_allocate(self.G, c, h, self.w, -K.I_LIMIT, K.I_LIMIT)
        return sol['currents'], sol


def plane_angle(a, B):
    t, _, up = frame(a)
    return math.atan2(B @ t, B @ up), math.hypot(B @ t, B @ up)


def step_plant(a, theta, currents, free, dt=K.DT_PLANT, G=None):
    """Advance both robots by ``dt``; returns new (a, theta, dtheta/dt).

    ``G`` (coil rows at the current positions) may be passed to skip its
    recomputation when the robots are captured.
    """
    Gc, c = field_matrix(a, theta)
    B = (Gc if G is None else G) @ currents + c
    a, theta = a.copy(), theta.copy()
    rate = np.zeros(2)
    for i in range(2):
        psi, mag = plane_angle(a[i], B[3 * i:3 * i + 3])
        rate[i] = K.MOMENT * mag * math.sin(psi - theta[i]) / K.ROT_DRAG
        theta[i] += rate[i] * dt
        if free[i]:
            a[i] += K.ROLL_EFFICIENCY * K.HELIX_RADIUS * rate[i] / K.R_PATH * dt
    return a, theta, rate


def roll_rate_for(a_dot):
    """Roll rate (rad/s) that produces path-angle rate ``a_dot`` in free mode."""
    return a_dot * K.R_PATH / (K.ROLL_EFFICIENCY * K.HELIX_RADIUS)
