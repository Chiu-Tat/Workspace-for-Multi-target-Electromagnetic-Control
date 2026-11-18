#!/usr/bin/env python3
# Run from the repository root:
#   python -m revised_case_studies.cs3_helical_robots.simulate [--captured] [--animate]
"""Case Study 3 -- two helical robots driven by independent rotating fields.

Scene: two curved lumens in a gel phantom that follow the archived Case 3
circular paths.  Each helical robot lies across its lumen and rolls along it
when its local field rotates in the vertical plane of the path tangent
(the rotation the archived Case 3 TFW certifies for every phase).

Mission in the configuration space (a1, a2) of the two path angles:
waypoints are joined by A* paths on the certified set of
``Results/results_case3.json`` (no TFW is computed), then the last leg
reverses robot 1 while robot 2 holds.  The resulting commands contain
simultaneous motion, stop/start of one robot while the other moves, and a
reversal of the rotation direction.

Loop: path-angle reference -> roll-rate command per robot (feed-forward +
position feedback, capped at MAX_FREQ_HZ) -> phase integration
and field allocation at 500 Hz on the six field rows -> synchronous-rotor
and rolling plant.
``--captured`` fixes both positions (captured-rotation experiment) and
tracks only rotation.

Outputs (``revised_case_studies/results/cs3``):
  D1_phantom      3-D phantom with lumens, robots and simulated paths
  D2_tfw_plan     certified (a1, a2) set with the planned and simulated path
  D3_timeseries   commanded vs. actual rotation frequency, phase lag, path
                  angle, field slack and currents
  D4_run.gif      (--animate)
  D_summary.json
"""

from __future__ import annotations

import argparse
import json
import math

import matplotlib.pyplot as plt
import numpy as np

from revised_case_studies.cs3_helical_robots import config as K
from revised_case_studies.cs3_helical_robots import geometry as g
from revised_case_studies.cs3_helical_robots.dynamics import (
    FieldController, roll_rate_for, step_plant)
from revised_case_studies.shared import grid_planner as gp
from revised_case_studies.shared import paths as P
from revised_case_studies.shared.metrics import error_stats

CASE = 'cs3'
COLORS = ('#1f5fa8', '#d98a1c')
K_POS = 1.5                       # 1/s, path-angle feedback gain
# (a1, a2) waypoints in degrees; the last leg moves robot 1 back only.
WAYPOINTS_DEG = [(150, -30), (208, 28), (208, 28), (180, 28)]
HOLD_S = 2.0                      # s, pause at the repeated waypoint


def config_speed():
    """Configuration-space speed (rad/s) so each robot stays under MAX_FREQ_HZ."""
    max_roll = 2 * math.pi * K.MAX_FREQ_HZ * 0.75
    return max_roll * K.ROLL_EFFICIENCY * K.HELIX_RADIUS / K.R_PATH


def plan():
    tfw = g.tfw()
    speed = config_speed()
    t_all, q_all, t0 = [], [], 0.0
    wps = [tuple(math.radians(v) for v in w) for w in WAYPOINTS_DEG]
    for a, b in zip(wps[:-1], wps[1:]):
        if a == b:
            t = np.arange(0, HOLD_S + 1e-9, K.DT)
            q = np.tile(a, (len(t), 1))
        else:
            path = gp.astar(tfw.free, tfw.index(*a), tfw.index(*b))
            if path is None:
                raise RuntimeError(f'no certified path between {a} and {b}')
            path = gp.shortcut(tfw.free, path)
            pts = np.array([tfw.angles(rc) for rc in path])
            pts[0], pts[-1] = a, b
            t, q = gp.resample(pts, speed, K.DT)
        if t_all:
            t, q = t[1:], q[1:]
        t_all.append(t + t0)
        q_all.append(q)
        t0 = t_all[-1][-1]
    return np.concatenate(t_all), np.concatenate(q_all)


def run(captured=False, t_extra=2.0):
    t_ref, q_ref = plan()
    qdot = np.gradient(q_ref, t_ref, axis=0)
    ctrl = FieldController()
    free = (not captured, not captured)
    a = q_ref[0].copy()
    theta = np.zeros(2)
    phi = np.zeros(2)
    n_sub = int(round(K.DT / K.DT_PLANT))
    w_max = 2 * math.pi * K.MAX_FREQ_HZ
    log = {k: [] for k in ('t', 'a', 'aref', 'f_cmd', 'f_act', 'lag', 'slack', 'currents', 'sat')}
    T = t_ref[-1] + t_extra
    for k in range(int(T / K.DT) + 1):
        # 10 Hz outer loop: roll-rate command from the path-angle reference
        t = k * K.DT
        aref = np.array([np.interp(t, t_ref, q_ref[:, i]) for i in range(2)])
        adot = np.array([np.interp(t, t_ref, qdot[:, i], right=0.0) for i in range(2)])
        if captured:
            w_cmd = roll_rate_for(adot)          # rotation as if free, position fixed
        else:
            w_cmd = roll_rate_for(adot + K_POS * (aref - a))
        w_cmd = np.clip(w_cmd, -w_max, w_max)
        th0 = theta.copy()
        slack, sat = np.zeros(2), False
        for _ in range(n_sub):
            # field-update rate: phase advance, allocation, plant step
            phi = phi + w_cmd * K.DT_PLANT
            currents, sol = ctrl.currents(a, phi)
            slack = np.maximum(slack, np.linalg.norm(np.reshape(sol['slack'], (2, 3)), axis=1) / K.B_ROT)
            sat |= sol['saturated']
            a, theta, _ = step_plant(a, theta, currents, free)
        log['t'].append(t)
        log['a'].append(a.copy())
        log['aref'].append(aref)
        log['f_cmd'].append(w_cmd / (2 * math.pi))
        log['f_act'].append((theta - th0) / K.DT / (2 * math.pi))
        log['lag'].append((phi - theta + math.pi) % (2 * math.pi) - math.pi)
        log['slack'].append(slack)
        log['currents'].append(currents)
        log['sat'].append(sat)
    out = {k: np.asarray(v) for k, v in log.items()}
    out['t_ref'], out['q_ref'] = t_ref, q_ref
    out['captured'] = captured
    return out


def summarize(r):
    s = {}
    for i in range(2):
        s[f'robot {i + 1}'] = {
            'path_angle_error_rad': error_stats(r['a'][:, i] - r['aref'][:, i]),
            'arc_error_m': error_stats((r['a'][:, i] - r['aref'][:, i]) * K.R_PATH),
            'frequency_error_hz': error_stats(r['f_act'][:, i] - r['f_cmd'][:, i]),
            'max_phase_lag_rad': float(np.abs(r['lag'][:, i]).max()),
            'stepped_out': bool(np.abs(r['lag'][:, i]).max() > math.pi / 2),
        }
    s['max_field_slack_rel'] = float(r['slack'].max())
    s['saturated_fraction'] = float(np.mean(r['sat']))
    s['all_in_certified_set'] = bool(all(g.tfw().contains(*q) for q in r['a']))
    s['captured'] = r['captured']
    return s


# ---------------------------------------------------------------- figures
def _draw_phantom_top(ax):
    ax.add_patch(plt.Rectangle((-45, -75), 90, 150, color='#d9cbb5', alpha=0.6, zorder=0))
    for i in range(2):
        c, _ = g.lumen_centerline(i)
        ax.plot(c[:, 0] * 1e3, c[:, 1] * 1e3, color='#b98f7a',
                lw=K.LUMEN_RADIUS * 2e3 * 2.8, solid_capstyle='round', alpha=0.9, zorder=1)
        ax.plot(c[:, 0] * 1e3, c[:, 1] * 1e3, color='#f1d9cc',
                lw=K.LUMEN_RADIUS * 2e3 * 2.2, solid_capstyle='round', zorder=2)


def fig_phantom(r):
    fig = plt.figure(figsize=(10, 5), constrained_layout=True)
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    for i in range(2):
        X, Y, Z = g.tube_mesh(i)
        ax.plot_surface(X * 1e3, Y * 1e3, Z * 1e3, color='#d59a8a', alpha=0.28,
                        linewidth=0, shade=True)
        pts = np.array([g.position(i, a) for a in r['a'][:, i]]) * 1e3
        ax.plot(*pts.T, color=COLORS[i], lw=2)
        for k in (0, len(r['t']) - 1):
            a = r['a'][k, i]
            _, n, _ = g.frame(a)
            h = g.helix_curve(g.position(i, a), n, 0.0) * 1e3
            ax.plot(*h.T, color=COLORS[i], lw=1.4)
    ax.set_box_aspect((1, 2.2, 0.35))
    ax.set_zticks([50, 60])
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    ax.set_zlabel('z (mm)')
    ax.view_init(elev=28, azim=-55)
    ax.set_title('Two curved lumens with helical robots')
    ax2 = fig.add_subplot(1, 2, 2)
    _draw_phantom_top(ax2)
    for i in range(2):
        pts = np.array([g.position(i, a) for a in r['a'][:, i]]) * 1e3
        ax2.plot(pts[:, 0], pts[:, 1], color=COLORS[i], lw=2, zorder=4, label=f'robot {i + 1}')
        for k in np.linspace(0, len(r['t']) - 1, 5).astype(int):
            a = r['a'][k, i]
            _, n, _ = g.frame(a)
            h = g.helix_curve(g.position(i, a), n, 0.0) * 1e3
            ax2.plot(h[:, 0], h[:, 1], color=COLORS[i], lw=1, zorder=5)
    ax2.set_aspect('equal')
    ax2.set_xlim(-45, 45)
    ax2.set_ylim(-70, 70)
    ax2.set_xlabel('x (mm)')
    ax2.set_ylabel('y (mm)')
    ax2.legend(frameon=False, loc='upper center')
    ax2.set_title('Top view: robot paths along the lumens')
    P.save(fig, CASE, 'D1_phantom')


def fig_plan(r):
    tfw = g.tfw()
    fig, ax = plt.subplots(figsize=(5, 4.6), constrained_layout=True)
    ext = [math.degrees(v) for v in (*K.A1_RANGE, *K.A2_RANGE)]
    ax.imshow(tfw.free, origin='lower', extent=ext, aspect='auto',
              cmap=plt.matplotlib.colors.ListedColormap(['#e6ded3', '#cfe3d2']))
    ax.plot(*np.degrees(r['q_ref']).T, '--', color='k', lw=1, label='planned')
    ax.plot(*np.degrees(r['a']).T, color=P.C_MOVING, lw=1.8, label='simulated')
    for k, (x, y) in enumerate(WAYPOINTS_DEG):
        ax.plot(x, y, 'o', color='k', ms=5)
    ax.set_xlabel('robot 1 path angle a₁ (deg)')
    ax.set_ylabel('robot 2 path angle a₂ (deg)')
    ax.set_title('Certified TFW of archive Case 3 (green)')
    ax.legend(frameon=False, loc='lower right')
    P.save(fig, CASE, 'D2_tfw_plan')


def fig_series(r):
    fig, axes = plt.subplots(5, 1, figsize=(7.5, 9), sharex=True, constrained_layout=True)
    t = r['t']
    for i in range(2):
        axes[0].plot(t, r['f_cmd'][:, i], '--', color=COLORS[i], label=f'robot {i + 1} command')
        axes[0].plot(t, r['f_act'][:, i], color=COLORS[i], lw=1, label=f'robot {i + 1} actual')
        axes[1].plot(t, np.degrees(r['lag'][:, i]), color=COLORS[i])
        axes[2].plot(t, np.degrees(r['aref'][:, i]), '--', color=COLORS[i])
        axes[2].plot(t, np.degrees(r['a'][:, i]), color=COLORS[i])
        axes[3].plot(t, np.maximum(r['slack'][:, i], 1e-9), color=COLORS[i])
    axes[4].plot(t, np.max(np.abs(r['currents']), axis=1), color='k')
    axes[4].axhline(K.I_LIMIT, ls='--', color='k', lw=0.8)
    axes[0].axhline(K.STEP_OUT_HZ, ls=':', color=P.C_OUTSIDE, lw=0.8)
    axes[0].axhline(-K.STEP_OUT_HZ, ls=':', color=P.C_OUTSIDE, lw=0.8)
    axes[0].set_ylabel('rotation freq. (Hz)')
    axes[1].set_ylabel('phase lag (deg)')
    axes[2].set_ylabel('path angle (deg)')
    axes[3].set_ylabel('field slack\n(relative)')
    axes[3].set_yscale('log')
    axes[4].set_ylabel('max |I| (A)')
    axes[4].set_xlabel('time (s)')
    axes[0].legend(frameon=False, fontsize=6, ncol=2)
    P.save(fig, CASE, 'D3_timeseries')


def animate(r, every=2):
    from matplotlib.animation import FuncAnimation, PillowWriter
    fig, ax = plt.subplots(figsize=(4.2, 6), constrained_layout=True)
    _draw_phantom_top(ax)
    ax.set_aspect('equal')
    ax.set_xlim(-45, 45)
    ax.set_ylim(-70, 70)
    lines = [ax.plot([], [], color=COLORS[i], lw=1.6, zorder=5)[0] for i in range(2)]
    title = ax.set_title('')

    def draw(k):
        for i in range(2):
            a = r['a'][k, i]
            _, n, _ = g.frame(a)
            phase = 2 * math.pi * np.sum(r['f_act'][:k + 1, i]) * K.DT
            h = g.helix_curve(g.position(i, a), n, phase) * 1e3
            lines[i].set_data(h[:, 0], h[:, 1])
        title.set_text(f't = {r["t"][k]:.1f} s   f = {r["f_act"][k, 0]:+.1f}, '
                       f'{r["f_act"][k, 1]:+.1f} Hz')
        return lines + [title]

    anim = FuncAnimation(fig, draw, frames=range(0, len(r['t']), every))
    out = P.out_dir(CASE) / 'D4_run.gif'
    anim.save(out, writer=PillowWriter(fps=10))
    print(f'saved {out}')


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--captured', action='store_true', help='fix both positions')
    ap.add_argument('--animate', action='store_true')
    ap.add_argument('--show', action='store_true')
    args = ap.parse_args()
    P.apply_style()
    r = run(captured=args.captured)
    s = summarize(r)
    print(json.dumps(s, indent=1))
    fig_phantom(r)
    fig_plan(r)
    fig_series(r)
    if args.animate:
        animate(r)
    name = 'D_summary_captured.json' if args.captured else 'D_summary.json'
    (P.out_dir(CASE) / name).write_text(json.dumps(s, indent=1))
    if args.show:
        plt.show()


if __name__ == '__main__':
    main()
