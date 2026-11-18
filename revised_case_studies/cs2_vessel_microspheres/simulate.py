#!/usr/bin/env python3
# Run from the repository root:
#   python -m revised_case_studies.cs2_vessel_microspheres.simulate [--no-interaction] [--animate]
"""Case Study 2 -- route two magnetic microspheres to different carotid branches.

Scene: the Case 7 idealized carotid bifurcation (common, internal and
external carotid).  Sphere 1 starts in the common carotid and goes to the
internal carotid; sphere 2 starts behind it and goes to the external carotid.
Both are driven at the same time by the shared ten-coil current vector.

Loop (10 Hz): reference along each route -> desired velocity with position
feedback -> desired force F* = drag * v* -> slack QP allocation on
[B1, B2, F1, F2]: the net field (moment) direction is held along a common
direction at both spheres, its magnitude is kept above B_MIN, and all three
force components are controlled in each local vessel frame.
Plant (100 Hz): aligned moments with mutual interaction, Stokes drag,
optional branchwise Poiseuille flow (``config.FLOW_MEAN``), frictionless
wall contact (outward normal velocity removed, then a safety projection).  No TFW exists for this case and none is computed;
the allocator slack along the run is the pointwise feasibility record.

Outputs (``revised_case_studies/results/cs2``):
  C1_phantom       realistic top view with routes + 3-D lumen mesh
  C2_snapshots     sphere positions over time on the phantom
  C3_timeseries    tracking error, slack, currents, separation
  C4_run.gif       (--animate)
  C_summary.json
"""

from __future__ import annotations

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np

from revised_case_studies.cs2_vessel_microspheres import config as K
from revised_case_studies.cs2_vessel_microspheres.model import allocation_rows, plant
from revised_case_studies.cs2_vessel_microspheres.phantom import Phantom
from revised_case_studies.shared import paths as P
from revised_case_studies.shared.allocator import QPAllocator
from revised_case_studies.shared.metrics import error_stats, position_stats

CASE = 'cs2'
COLORS = ('#1f5fa8', '#d98a1c')
DT_PLANT = 0.002   # s; fast at 5.4 cP (drag-dominated response)
DRAW_R_MM = max(K.RADIUS * 1e3, 0.8)   # drawn sphere radius (enlarged for visibility)


class Route:
    def __init__(self, phantom, spec):
        self.points, self.branches, self.arc = phantom.route(spec['route'])
        self.allowed = sorted(set(self.branches))
        self.target = spec['target']
        self.length = self.arc[-1]

    def progress(self, p):
        """Arc length of the nearest route point, and distance to it."""
        d = np.linalg.norm(self.points - p, axis=1)
        k = int(np.argmin(d))
        return self.arc[k], d[k]

    def at(self, s):
        s = np.clip(s, 0, self.length)
        p = np.array([np.interp(s, self.arc, self.points[:, k]) for k in range(3)])
        k = min(int(np.searchsorted(self.arc, s)), len(self.arc) - 2)
        t = self.points[k + 1] - self.points[k]
        return p, t / np.linalg.norm(t)


def _frame_along(phantom, p, allowed, t_dir):
    t, n, b = phantom.tangent_frame(p, allowed)
    if t @ t_dir < 0:
        t, b = -t, -b
    return t, n, b


def run(interaction=True, t_extra=4.0):
    ph = Phantom()
    routes = [Route(ph, spec) for spec in K.SPHERES.values()]
    p = np.array([r.at(0.0)[0] for r in routes])
    lo = np.full(12, -np.inf)
    lo[[0, 3]] = K.B_MIN
    alloc = QPAllocator(12, i_min=-K.I_LIMIT, i_max=K.I_LIMIT, out_min=lo)
    w = np.array([K.W_B_MAG, K.W_B_DIR, K.W_B_DIR] * 2
                 + [K.W_F_TANGENT, K.W_F_TRANSVERSE, K.W_F_TRANSVERSE] * 2)
    T = max(r.length for r in routes) / K.SPEED + t_extra
    i_prev = np.zeros(10)
    log = {k: [] for k in ('t', 'p', 'pref', 'slack', 'currents', 'force', 'contact')}
    n_sub = int(round(K.DT / DT_PLANT))
    for k in range(int(T / K.DT) + 1):
        t = k * K.DT
        refs = [r.at(K.SPEED * t) for r in routes]
        frames, u, hdes = [], [], []
        for i, r in enumerate(routes):
            pref, tref = refs[i]
            moving = K.SPEED * t < r.length
            fr = _frame_along(ph, p[i], r.allowed, tref)
            frames.append(fr)
            u.append(np.asarray(K.FIELD_DIRECTION, float) / np.linalg.norm(K.FIELD_DIRECTION))
            v_des = (K.SPEED * tref if moving else 0.0) + K.K_P * (pref - p[i])
            u_flow = ph.flow_velocity(p[i], K.FLOW_MEAN, r.allowed)
            hdes.append(np.array(fr) @ (K.DRAG * (v_des - u_flow)))
        u = np.array(u)
        G, c = allocation_rows(p, u, frames, interaction=interaction)
        h = np.concatenate([[K.B_HOLD, 0, 0], [K.B_HOLD, 0, 0], hdes[0], hdes[1]])
        sol = alloc.solve(G, c, h, w, i_prev)
        I = sol['currents']
        i_prev = I

        contact = [False, False]
        for _ in range(n_sub):
            _, _, F = plant(p, I, interaction=interaction)
            for i, r in enumerate(routes):
                u_flow = ph.flow_velocity(p[i], K.FLOW_MEAN, r.allowed)
                v, touching = ph.constrain_velocity(p[i], u_flow + F[i] / K.DRAG,
                                                    K.RADIUS, r.allowed)
                q, hit = ph.project_inside(p[i] + v * DT_PLANT, K.RADIUS, r.allowed)
                hit |= touching
                p[i] = q
                contact[i] |= hit
        log['t'].append(t)
        log['p'].append(p.copy())
        log['pref'].append(np.array([rf[0] for rf in refs]))
        log['slack'].append(np.abs(w * sol['slack']).reshape(2, 2, 3))
        log['currents'].append(I)
        log['force'].append(F)
        log['contact'].append(contact)
    out = {k: np.asarray(v) for k, v in log.items()}
    out['routes'] = routes
    out['phantom'] = ph
    return out


def summarize(r):
    ph, routes = r['phantom'], r['routes']
    summary = {}
    for i, (name, spec) in enumerate(K.SPHERES.items()):
        _, b, _, _ = ph.nearest(r['p'][-1, i])
        prog = np.array([routes[i].progress(q) for q in r['p'][:, i]])
        s_ref = np.minimum(K.SPEED * r['t'], routes[i].length)
        summary[name] = {
            'along_route_error_m': error_stats(prog[:, 0] - s_ref),
            'off_centerline_m': error_stats(prog[:, 1]),
            'position_error_m': position_stats(r['p'][:, i], r['pref'][:, i]),
            'final_branch': b, 'target_branch': spec['target'],
            'reached_target_branch': b == spec['target'],
            'final_goal_error_m': float(np.linalg.norm(r['p'][-1, i] - routes[i].points[-1])),
            'wall_contact_fraction': float(np.mean(r['contact'][:, i])),
        }
    sep = np.linalg.norm(r['p'][:, 0] - r['p'][:, 1], axis=1)
    summary['min_separation_m'] = float(sep.min())
    summary['saturated_fraction'] = float(np.mean(np.max(np.abs(r['currents']), 1) > K.I_LIMIT - 1e-3))
    summary['max_field_direction_slack_rel'] = float(r['slack'][:, :, 0, 1:].max())
    summary['max_tangent_force_slack_rel'] = float(r['slack'][:, :, 1, 0].max())
    return summary


def fig_phantom(r):
    ph = r['phantom']
    fig = plt.figure(figsize=(10, 5), constrained_layout=True)
    ax = fig.add_subplot(1, 2, 1)
    ph.draw_top(ax)
    for i, route in enumerate(r['routes']):
        q = route.points * 1e3
        ax.plot(q[:, 0], q[:, 1], '--', color=COLORS[i], lw=1.2, zorder=4,
                label=f'{list(K.SPHERES)[i]} → {route.target}')
        ax.plot(*q[0, :2], 'o', color=COLORS[i], ms=7, mec='k', zorder=5)
        ax.plot(*q[-1, :2], '*', color=COLORS[i], ms=12, mec='k', zorder=5)
    ax.set_xlim(-22, 24)
    ax.set_ylim(-30, 33)
    ax.legend(loc='lower right', fontsize=7)
    ax.set_title('Carotid bifurcation phantom: routes')
    ax3 = fig.add_subplot(1, 2, 2, projection='3d')
    ph.draw_3d(ax3)
    for i in range(2):
        ax3.plot(*(r['p'][:, i] * 1e3).T, color=COLORS[i], lw=2)
    ax3.view_init(elev=35, azim=-60)
    ax3.set_title('3-D lumen with simulated sphere paths')
    P.save(fig, CASE, 'C1_phantom')


def fig_snapshots(r, n=5):
    ph = r['phantom']
    idx = np.linspace(0, len(r['t']) - 1, n).astype(int)
    fig, axes = plt.subplots(1, n, figsize=(2.4 * n, 3.6), constrained_layout=True)
    for ax, k in zip(axes, idx):
        ph.draw_top(ax)
        for i in range(2):
            trail = r['p'][:k + 1, i] * 1e3
            ax.plot(trail[:, 0], trail[:, 1], color=COLORS[i], lw=1.2, zorder=6)
            ax.add_patch(plt.Circle(trail[-1, :2], DRAW_R_MM, color=COLORS[i],
                                    ec='k', lw=0.5, zorder=7))
        ax.set_xlim(-18, 22)
        ax.set_ylim(-28, 32)
        ax.set_title(f't = {r["t"][k]:.1f} s')
        ax.set_xlabel('')
        ax.set_ylabel('')
    P.save(fig, CASE, 'C2_snapshots')


def fig_series(r):
    fig, axes = plt.subplots(4, 1, figsize=(7, 7.5), sharex=True, constrained_layout=True)
    t = r['t']
    for i, name in enumerate(K.SPHERES):
        route = r['routes'][i]
        prog = np.array([route.progress(q) for q in r['p'][:, i]])
        s_ref = np.minimum(K.SPEED * t, route.length)
        axes[0].plot(t, (prog[:, 0] - s_ref) * 1e3, color=COLORS[i], label=f'{name} along route')
        axes[0].plot(t, prog[:, 1] * 1e3, ':', color=COLORS[i], label=f'{name} off centreline')
        floor = 1e-6
        axes[1].plot(t, np.maximum(r['slack'][:, i, 1, 0], floor), color=COLORS[i],
                     label=f'{name} tangential force')
        axes[1].plot(t, np.maximum(np.linalg.norm(r['slack'][:, i, 0, 1:], axis=1), floor), ':',
                     color=COLORS[i], label=f'{name} field direction')
    axes[2].plot(t, np.max(np.abs(r['currents']), axis=1), color='k')
    axes[2].axhline(K.I_LIMIT, ls='--', color='k', lw=0.8)
    sep = np.linalg.norm(r['p'][:, 0] - r['p'][:, 1], axis=1) * 1e3
    axes[3].plot(t, sep, color='k')
    axes[3].axhline(K.MIN_SEPARATION * 1e3, ls='--', color=P.C_OUTSIDE, lw=0.8)
    axes[0].set_ylabel('error (mm)')
    axes[0].axhline(0, color='k', lw=0.5)
    axes[1].set_ylabel('allocator slack\n(relative)')
    axes[1].set_yscale('log')
    axes[2].set_ylabel('max |I| (A)')
    axes[3].set_ylabel('sphere separation (mm)')
    axes[3].set_xlabel('time (s)')
    axes[0].legend(frameon=False, fontsize=6, ncol=2)
    axes[1].legend(frameon=False, fontsize=6, ncol=2)
    P.save(fig, CASE, 'C3_timeseries')


def animate(r, every=3):
    from matplotlib.animation import FuncAnimation, PillowWriter
    ph = r['phantom']
    fig, ax = plt.subplots(figsize=(4.4, 5.6), constrained_layout=True)
    ph.draw_top(ax)
    ax.set_xlim(-18, 22)
    ax.set_ylim(-28, 32)
    trails = [ax.plot([], [], color=COLORS[i], lw=1.2, zorder=6)[0] for i in range(2)]
    dots = [plt.Circle((0, 0), DRAW_R_MM, color=COLORS[i], ec='k', zorder=7)
            for i in range(2)]
    for d in dots:
        ax.add_patch(d)
    title = ax.set_title('')

    def draw(k):
        for i in range(2):
            q = r['p'][:k + 1, i] * 1e3
            trails[i].set_data(q[:, 0], q[:, 1])
            dots[i].center = q[-1, :2]
        title.set_text(f't = {r["t"][k]:.1f} s')
        return trails + dots + [title]

    anim = FuncAnimation(fig, draw, frames=range(0, len(r['t']), every))
    out = P.out_dir(CASE) / 'C4_run.gif'
    anim.save(out, writer=PillowWriter(fps=10))
    print(f'saved {out}')


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--no-interaction', action='store_true',
                    help='drop the sphere-sphere dipole terms in both model and plant')
    ap.add_argument('--animate', action='store_true')
    ap.add_argument('--show', action='store_true')
    args = ap.parse_args()
    P.apply_style()
    r = run(interaction=not args.no_interaction)
    summary = summarize(r)
    print(json.dumps(summary, indent=1))
    fig_phantom(r)
    fig_snapshots(r)
    fig_series(r)
    if args.animate:
        animate(r)
    (P.out_dir(CASE) / 'C_summary.json').write_text(json.dumps(summary, indent=1))
    if args.show:
        plt.show()


if __name__ == '__main__':
    main()
