"""Closed-loop simulation of the moving cylindrical magnet (Case Study 1).

Control loop at 10 Hz (paper, Sec. "Planning, feedback, and current allocation"):

  1. reference pose (x_r, y_r, beta_r) from the trajectory at time t;
  2. outer loop: commanded heading beta_c = beta_r + k_h * lateral error,
     commanded thrust fz_c = drag feed-forward + k_p * along-track error;
  3. current allocation with the slack QP of ``shared.allocator`` on the
     body-frame rows [By, Bz, fy, fz] = [0, B_HOLD, 0, fz_c] at (x, y, beta_c).

Plant (integrated at 100 Hz, same drag law as the Case 6 interactive
simulator): the planar magnetic force, including the fixed magnet's
interaction, moves the magnet with v = F / DRAG_C; the heading relaxes toward
the in-plane field direction with time constant TAU_ROT.  Contact with the
fixed magnet (centre distance < CONTACT_DIST) latches the magnet in place.
"""

from __future__ import annotations

import math

import numpy as np

from revised_case_studies.cs1_cylinder_interaction import common as c
from revised_case_studies.shared.allocator import QPAllocator, row_weights

DRAG_C = 5.0          # N s / m, from the Case 6 interactive simulator
TAU_ROT = 0.15        # s, heading alignment time constant
DT_CTRL = 0.1         # s, 10 Hz allocation
DT_PLANT = 0.01       # s
K_P = 4.0             # N / m, along-track gain
K_H = 20.0            # rad / m, lateral-error-to-heading gain
MAX_HEADING_CORR = 0.6
ROW_SCALE = (c.B_HOLD, c.B_HOLD, c.F_MAX, c.F_MAX)


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def _interp_ref(traj, t):
    t = min(max(t, traj[0, 3]), traj[-1, 3])
    x = np.interp(t, traj[:, 3], traj[:, 0])
    y = np.interp(t, traj[:, 3], traj[:, 1])
    b = np.interp(t, traj[:, 3], np.unwrap(traj[:, 2]))
    return x, y, b


def _ref_velocity(traj, t, h=0.05):
    a = np.array(_interp_ref(traj, t - h)[:2])
    b = np.array(_interp_ref(traj, t + h)[:2])
    return (b - a) / (2 * h)


def control_command(traj, t, x, y):
    """Outer loop: commanded heading and thrust for reference ``traj`` at ``t``."""
    xr, yr, br = _interp_ref(traj, t)
    u = np.array([math.cos(br), math.sin(br)])
    n = np.array([-u[1], u[0]])
    e = np.array([xr - x, yr - y])
    v_ref = _ref_velocity(traj, t)
    beta_c = br + float(np.clip(K_H * (e @ n), -MAX_HEADING_CORR, MAX_HEADING_CORR))
    fz_c = float(np.clip(DRAG_C * (v_ref @ u) + K_P * (e @ u), -c.F_MAX, c.F_MAX))
    return beta_c, fz_c, (xr, yr, br)


def plant_step(x, y, beta, currents, dt=DT_PLANT, layout='single'):
    """One plant step; returns (x, y, beta, B_world, F_world)."""
    A_w = c.world_matrix(x, y, beta, layout)
    h = A_w @ np.append(currents, 1.0)
    B, F = h[:3], h[3:]
    x += F[0] / DRAG_C * dt
    y += F[1] / DRAG_C * dt
    beta += _wrap(math.atan2(B[1], B[0]) - beta) * dt / TAU_ROT
    return x, y, beta, B, F


def simulate(traj, t_extra=3.0, di_max=None, allocator=None):
    """Track ``traj`` ([x, y, beta, t] rows).  Returns a dict of time series."""
    alloc = allocator or QPAllocator(4, di_max=di_max)
    w = row_weights(ROW_SCALE)
    x, y, beta = traj[0, :3]
    i_prev = np.zeros(c.NUM_COILS)
    T = traj[-1, 3] + t_extra
    log = {k: [] for k in ('t', 'x', 'y', 'beta', 'xr', 'yr', 'betar', 'slack',
                           'currents', 'fz_cmd', 'label', 'contact')}
    contact = False
    currents = i_prev
    beta_c = beta
    n_sub = int(round(DT_CTRL / DT_PLANT))
    for k in range(int(T / DT_CTRL) + 1):
        t = k * DT_CTRL
        beta_c_new, fz_new, (xr, yr, br) = control_command(traj, t, x, y)
        if not contact:
            beta_c, fz_c = beta_c_new, fz_new
            A_hat = c.actuation_matrix_body(x, y, beta_c)
            sol = alloc.solve(A_hat[:, :c.NUM_COILS], A_hat[:, c.NUM_COILS],
                              np.array([0.0, c.B_HOLD, 0.0, fz_c]), w, i_prev)
            currents = sol['currents']
            i_prev = currents
            slack = sol['weighted_slack']
        else:
            fz_c, slack = 0.0, float('nan')
        log['t'].append(t); log['x'].append(x); log['y'].append(y)
        log['beta'].append(beta); log['xr'].append(xr); log['yr'].append(yr)
        log['betar'].append(br); log['slack'].append(slack)
        log['currents'].append(np.array(currents)); log['fz_cmd'].append(fz_c)
        log['label'].append(c.label(x, y, beta, check_outside=False))
        log['contact'].append(contact)

        for _ in range(n_sub):
            if contact:
                break
            x, y, beta, _, _ = plant_step(x, y, beta, currents)
            if math.dist((x, y), c.FIXED_XY) < c.CONTACT_DIST:
                contact = True
    out = {k: np.asarray(v) for k, v in log.items()}
    err = np.hypot(out['x'] - out['xr'], out['y'] - out['yr'])
    out['pos_err'] = err
    out['sep'] = np.hypot(out['x'] - c.FIXED_XY[0], out['y'] - c.FIXED_XY[1])
    out['final_err'] = float(math.dist((out['x'][-1], out['y'][-1]), traj[-1, :2]))
    out['contact_any'] = bool(out['contact'].any())
    return out
