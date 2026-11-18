#!/usr/bin/env python3
# Run from the repository root:
#   python -m revised_case_studies.cs3_helical_robots.interactive
"""
Case Study 3 -- interactive: two helical robots, two independent rotating fields.

Each helical robot lies across its curved lumen (gel phantom, 5.4 cP
water-glycerin) and rolls along it when its local field rotates in the
vertical plane of the path tangent -- the rotation certified by the archived
Case 3 TFW for every phase.  Set each robot's rotation frequency and
direction independently; one shared ten-coil current vector produces both
rotating fields.

The map on the right is the certified TFW in the configuration space
(a1, a2) of the two path angles (``Results/results_case3.json``; nothing is
recomputed).  With the guard on (G) a robot stops rolling rather than take
the pair out of the certified set.  Switch the guard off and roll into the
uncertified band: the field error grows for some phase pairs, and the
robots lose synchronisation.  Above the step-out frequency (~6 Hz) a robot
also slips although its field is exact -- a dynamic limit, not an
electromagnetic one.

Field update and plant at 500 Hz: requested fields B_ROT * (sin phi t + cos
phi z) at both robots (other robot's dipole included) -> minimum-norm /
bounded allocation -> synchronous-rotor model with rolling propulsion.

Controls
--------
  Q / A     robot 1 frequency +0.5 / -0.5 Hz     P / L   robot 2 frequency +0.5 / -0.5 Hz
  1 / 2     stop robot 1 / robot 2               SPACE   stop both
  K         captured rotation on / off           G       guard on / off
  M         mission autopilot (Case 3 plan)      X       reset        ESC quit
"""

from __future__ import annotations

import argparse
import math

import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D

from revised_case_studies.shared import ui
from revised_case_studies.cs3_helical_robots import config as K
from revised_case_studies.cs3_helical_robots import geometry as g
from revised_case_studies.cs3_helical_robots.dynamics import (
    FieldController, roll_rate_for, step_plant)
from revised_case_studies.cs3_helical_robots.simulate import K_POS, plan

FPS = 20.0
DT = 1.0 / FPS
N_SUB = int(round(DT / K.DT_PLANT))
F_STEP = 0.5                       # Hz per key press
F_LIMIT = 10.0                     # Hz, command range (step-out at ~6 Hz)
START_DEG = (150.0, -30.0)
KEY_HINTS = ((('Q', 'A'), 'robot 1 ±f'), (('P', 'L'), 'robot 2 ±f'), (('1', '2'), 'stop one'),
             (('Space',), 'stop'), (('K',), 'captured'), (('G',), 'guard'), (('M',), 'mission'),
             (('X',), 'reset'), (('Esc',), 'quit'))


class CS3Sim:
    def __init__(self):
        self.tfw = g.tfw()
        self.ctrl = FieldController()
        self.guard = True
        self.captured = False
        self._reset()
        self._build()

    def _reset(self):
        self.a = np.radians(START_DEG)
        self.theta = np.zeros(2)
        self.phi = np.zeros(2)
        self.f_cmd = np.zeros(2)
        self.f_act = np.zeros(2)
        self.err = np.zeros(2)
        self.currents = np.zeros(10)
        self.mission = None
        self.t = 0.0
        self.trail = [self.a.copy()]
        self.blocked = [False, False]

    # --------------------------------------------------------------- layout
    def _build(self):
        ui.apply_style()
        fig = self.fig = ui.plt.figure(figsize=(ui.FIG_W, ui.FIG_H))
        if fig.canvas.manager is not None:
            fig.canvas.manager.set_window_title('eMNS · Case Study 3 · interactive')
        ui.chrome(fig, 'Case Study 3  ·  Two helical robots, independent rotating fields',
                  f'5.4 cP water-glycerin   ·   |B| = {K.B_ROT * 1e3:.0f} mT rotating   ·   '
                  f'm = {K.MOMENT:g} A·m²   ·   10 coils, |I| ≤ {K.I_LIMIT:.0f} A', KEY_HINTS)
        left, bottom = 0.95, 0.92
        top = ui.FIG_H - 0.80
        side = top - bottom
        wscene = side * 0.62
        self.ax = fig.add_axes(ui.fig_rect(left, bottom, wscene, side))
        self._build_scene()

        x = left + wscene + 0.7
        w = ui.FIG_W - ui.MARGIN - x
        half = (w - ui.CARD_GAP) / 2
        heights = np.array([0.80, 3.35, 1.25, 1.45])
        bottoms = top - np.cumsum(heights) - ui.CARD_GAP * np.arange(len(heights))
        self.status = ui.StatusCard(ui.Card(fig, x, bottoms[0], w, heights[0]),
                                    (('t', 't', 0.0), ('a1', 'a1', 0.95), ('a2', 'a2', 1.95),
                                     ('mode', '', 2.95)))
        self.map_card = ui.Card(fig, x, bottoms[1], half, heights[1], 'Certified TFW (a1, a2)')
        self.freq_card = ui.Card(fig, x + half + ui.CARD_GAP, bottoms[1], half, heights[1],
                                 'Rotation per robot')
        self.field_card = ui.Card(fig, x, bottoms[2], w, heights[2], 'Field at each robot')
        self.coil_card = ui.Card(fig, x, bottoms[3], w, heights[3], 'Coil currents')
        self._build_map()
        self._build_freq()
        self._build_field()
        self.bars = ui.coil_axes(self.coil_card, i_max=K.I_LIMIT)
        fig.canvas.mpl_connect('key_press_event', self._press)

    def _build_scene(self):
        ax = self.ax
        ui.style_scene(ax)
        ax.grid(False)
        ax.add_patch(ui.plt.Rectangle((-45, -75), 90, 150, color='#d9cbb5', alpha=0.6, zorder=0))
        for i in range(2):
            cl, _ = g.lumen_centerline(i)
            ax.plot(cl[:, 0] * 1e3, cl[:, 1] * 1e3, color='#b98f7a',
                    lw=K.LUMEN_RADIUS * 2e3 * 2.2, solid_capstyle='round', alpha=0.9, zorder=1)
            ax.plot(cl[:, 0] * 1e3, cl[:, 1] * 1e3, color='#f1d9cc',
                    lw=K.LUMEN_RADIUS * 2e3 * 1.7, solid_capstyle='round', zorder=2)
            rng = K.A1_RANGE if i == 0 else K.A2_RANGE
            ends = np.array([g.position(i, v) for v in rng]) * 1e3
            ax.plot(ends[:, 0], ends[:, 1], '|', ms=14, mew=1.2, color=ui.C_MUTED, zorder=3)
        ax.set_xlim(-45, 45)
        ax.set_ylim(-72, 72)
        ax.set_aspect('equal', anchor='W')
        ax.set_xlabel('x (mm)', fontsize=9, color=ui.C_MUTED)
        ax.set_ylabel('y (mm)', fontsize=9, color=ui.C_MUTED)
        self.helix = [ax.plot([], [], color=ui.C_ROBOT[i], lw=1.8, zorder=6)[0] for i in range(2)]
        self.core = [ax.plot([], [], color=ui.C_INK, lw=0.8, zorder=7)[0] for i in range(2)]
        self.dial = [ax.plot([], [], color=ui.C_F, lw=1.6, zorder=8)[0] for i in range(2)]
        self.dial_r = [ax.plot([], [], color=ui.C_ROBOT[i], lw=1.6, zorder=8)[0] for i in range(2)]
        self.labels = [ax.text(0, 0, f'robot {i + 1}', fontsize=8, color=ui.C_ROBOT[i],
                               ha='center', zorder=9) for i in range(2)]
        leg = ax.legend([Line2D([], [], color=ui.C_F, lw=1.6), Line2D([], [], color=ui.C_ROBOT[0], lw=1.6)],
                        ['field phase', 'robot phase'], loc='lower left', bbox_to_anchor=(0, 1),
                        ncol=2, fontsize=8, frameon=False)
        for t in leg.get_texts():
            t.set_color(ui.C_MUTED)

    def _build_map(self):
        card = self.map_card
        ax = card.inset(left=0.62, bottom=0.55, right=0.2, top=0.5)
        ext = [math.degrees(v) for v in (*K.A1_RANGE, *K.A2_RANGE)]
        ax.imshow(self.tfw.free, origin='lower', extent=ext, aspect='auto',
                  cmap=ListedColormap(['#eadfd6', '#cfe3d2']), interpolation='nearest')
        ax.set_xlabel('a1 (deg)', fontsize=8, color=ui.C_MUTED)
        ax.set_ylabel('a2 (deg)', fontsize=8, color=ui.C_MUTED)
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_color(ui.C_LINE)
        self.map_trail = LineCollection([], linewidths=1.3, colors=ui.C_TRAIL, zorder=3)
        ax.add_collection(self.map_trail)
        self.map_dot, = ax.plot([], [], 'o', ms=7, mec=ui.C_INK, zorder=4)
        self.map_plan, = ax.plot([], [], ls=(0, (3, 2)), color=ui.C_INK, lw=0.9, zorder=2)
        self.map_txt = card.ax.text(ui.CARD_PAD, 0.18, '', fontsize=8, va='center')

    def _build_freq(self):
        card = self.freq_card
        ax = self.freq_ax = card.inset(left=0.45, bottom=0.55, right=0.2, top=0.5)
        ax.set_xlim(-0.6, 1.6)
        ax.set_ylim(-F_LIMIT * 1.05, F_LIMIT * 1.05)
        ax.set_xticks([0, 1], ['robot 1', 'robot 2'])
        ax.set_yticks([-10, -6, 0, 6, 10], ['−10', '−6', '0', '6', '10 Hz'])
        for s in (-K.STEP_OUT_HZ, K.STEP_OUT_HZ):
            ax.axhline(s, color=ui.C_BAD, lw=0.8, ls=(0, (3, 2.5)), alpha=0.6)
        ax.axhline(0, color=ui.C_INK, lw=0.6, alpha=0.25)
        self.fb_cmd = ax.bar([-0.17, 0.83], [0, 0], width=0.3, color=ui.C_FAINT, zorder=2)
        self.fb_act = ax.bar([0.17, 1.17], [0, 0], width=0.3, color=list(ui.C_ROBOT), zorder=2)
        self.freq_txt = card.ax.text(ui.CARD_PAD, 0.18, '', fontsize=7.8, color=ui.C_MUTED, va='center')

    def _build_field(self):
        ax = self.field_card.ax
        self.field_txt = [ax.text(ui.CARD_PAD, 0.78 - 0.34 * i, '', fontsize=8.4,
                                  color=ui.C_ROBOT[i], va='center') for i in range(2)]

    # ------------------------------------------------------------- controls
    def _press(self, e):
        k = (e.key or '').lower()
        if k in ('q', 'a', 'p', 'l'):
            self.mission = None
            i = 0 if k in ('q', 'a') else 1
            self.f_cmd[i] = float(np.clip(self.f_cmd[i] + (F_STEP if k in ('q', 'p') else -F_STEP),
                                          -F_LIMIT, F_LIMIT))
        elif k in ('1', '2'):
            self.f_cmd[int(k) - 1] = 0.0
        elif k == ' ':
            self.f_cmd[:] = 0.0
            self.mission = None
        elif k == 'k':
            self.captured = not self.captured
        elif k == 'g':
            self.guard = not self.guard
        elif k == 'm':
            self._reset()
            self.mission = plan()
            self.a = self.mission[1][0].copy()
            self.captured = False
            self.trail = [self.a.copy()]
        elif k == 'x':
            self._reset()
        elif k == 'escape':
            ui.plt.close(self.fig)

    # ------------------------------------------------------------ one frame
    def step(self):
        w_max = 2 * math.pi * F_LIMIT
        if self.mission is not None:
            t_ref, q_ref = self.mission
            aref = np.array([np.interp(self.t, t_ref, q_ref[:, i]) for i in range(2)])
            qdot = np.gradient(q_ref, t_ref, axis=0)
            adot = np.array([np.interp(self.t, t_ref, qdot[:, i], right=0.0) for i in range(2)])
            w_cmd = np.clip(roll_rate_for(adot + K_POS * (aref - self.a)), -w_max, w_max)
            self.f_cmd = w_cmd / (2 * math.pi)
        w_cmd = 2 * math.pi * self.f_cmd
        th0 = self.theta.copy()
        self.err[:] = 0.0
        self.blocked = [False, False]
        free = [not self.captured, not self.captured]
        for _ in range(N_SUB):
            self.phi = self.phi + w_cmd * K.DT_PLANT
            I, sol = self.ctrl.currents(self.a, self.phi)
            self.err = np.maximum(self.err, np.linalg.norm(np.reshape(sol['slack'], (2, 3)), axis=1) / K.B_ROT)
            a_new, th_new, _ = step_plant(self.a, self.theta, I, free)
            if self.guard and not self.captured and not self.tfw.contains(*a_new):
                # roll only the robot whose motion keeps the pair certified
                for i in range(2):
                    trial = self.a.copy()
                    trial[i] = a_new[i]
                    if self.tfw.contains(*trial):
                        self.a[i] = a_new[i]
                    else:
                        self.blocked[i] = True
                self.theta = th_new
            else:
                self.a, self.theta = a_new, th_new
            rng = (K.A1_RANGE, K.A2_RANGE)
            self.a = np.array([np.clip(self.a[i], *rng[i]) for i in range(2)])
            self.currents = I
        self.f_act = (self.theta - th0) / DT / (2 * math.pi)
        self.t += DT
        if np.linalg.norm(self.trail[-1] - self.a) > 1e-4:
            self.trail = (self.trail + [self.a.copy()])[-2000:]

    def update(self, _frame):
        self.step()
        for i in range(2):
            p = g.position(i, self.a[i]) * 1e3
            _, n, _ = g.frame(self.a[i])
            h = g.helix_curve(g.position(i, self.a[i]), n, self.theta[i]) * 1e3
            self.helix[i].set_data(h[:, 0], h[:, 1])
            ends = np.array([p - n[:3] * 2.8, p + n[:3] * 2.8])
            self.core[i].set_data(ends[:, 0], ends[:, 1])
            # phase dial beside the robot: field phase (orange) vs robot phase
            c0 = p + np.array([10 if i == 0 else -10, 0, 0])
            r = 4.0
            self.dial[i].set_data([c0[0], c0[0] + r * math.sin(self.phi[i])],
                                  [c0[1], c0[1] + r * math.cos(self.phi[i])])
            self.dial_r[i].set_data([c0[0], c0[0] + 0.8 * r * math.sin(self.theta[i])],
                                    [c0[1], c0[1] + 0.8 * r * math.cos(self.theta[i])])
            self.labels[i].set_position((c0[0], c0[1] + 6))

        deg = np.degrees(np.asarray(self.trail))
        if len(deg) > 1:
            self.map_trail.set_segments(np.stack([deg[:-1], deg[1:]], axis=1))
        inside = self.tfw.contains(*self.a)
        self.map_dot.set_data([math.degrees(self.a[0])], [math.degrees(self.a[1])])
        self.map_dot.set_color(ui.C_CERT if inside else ui.C_BAD)
        if self.mission is not None:
            self.map_plan.set_data(*np.degrees(self.mission[1]).T)
        else:
            self.map_plan.set_data([], [])
        self.map_txt.set_text('pair inside the certified set' if inside else 'pair OUTSIDE the certified set')
        self.map_txt.set_color(ui.C_OK if inside else ui.C_BAD)
        self.map_card.meta.set_text(f'guard {"on" if self.guard else "off"}')
        self.map_card.meta.set_color(ui.C_CERT if self.guard else ui.C_BAD)

        for i in range(2):
            self.fb_cmd[i].set_height(self.f_cmd[i])
            self.fb_act[i].set_height(float(np.clip(self.f_act[i], -F_LIMIT, F_LIMIT)))
        lag = (self.phi - self.theta + math.pi) % (2 * math.pi) - math.pi
        slipping = [abs(self.f_act[i] - self.f_cmd[i]) > 0.2 * max(abs(self.f_cmd[i]), 1.0)
                    for i in range(2)]
        self.freq_txt.set_text(f'grey = command   ·   lag {math.degrees(lag[0]):+.0f}°, '
                               f'{math.degrees(lag[1]):+.0f}°')
        self.freq_card.meta.set_text('step-out!' if any(slipping) else 'synchronous')
        self.freq_card.meta.set_color(ui.C_BAD if any(slipping) else ui.C_OK)

        for i in range(2):
            self.field_txt[i].set_text(
                f'robot {i + 1}:  f = {self.f_cmd[i]:+.1f} Hz commanded, {self.f_act[i]:+.2f} Hz actual   ·   '
                f'field error {self.err[i] * 100:.2f}% of {K.B_ROT * 1e3:.0f} mT'
                + ('   ·   rolling blocked by guard' if self.blocked[i] else ''))
        self.field_card.meta.set_text('exact' if self.err.max() < 1e-6 else 'NOT exact for some phases')
        self.field_card.meta.set_color(ui.C_OK if self.err.max() < 1e-6 else ui.C_BAD)

        sat = ui.update_coils(self.coil_card, self.bars, self.currents, K.I_LIMIT, 'field update 500 Hz')

        if self.err.max() > 1e-3:
            st = ('bad', 'Field', 'Rotating fields not reachable at this pair')
        elif not inside:
            st = ('warn', 'Uncertified', 'Outside the certified set: fields exact for these phases, not guaranteed')
        elif any(slipping):
            st = ('warn', 'Step-out', 'Robot slips: frequency above its step-out, fields exact')
        elif any(self.blocked):
            st = ('warn', 'Guard', 'Rolling stopped at the edge of the certified set')
        elif self.mission is not None:
            st = ('ok', 'Mission', 'Following the Case 3 plan (stop/start, reversal)')
        elif np.any(self.f_cmd):
            st = ('ok', 'Rotate', 'Both rotations synchronous')
        else:
            st = ('ok', 'Hold', 'Fields held, no rotation')
        flags = [f for f, on in (('coil limit', sat), ('captured', self.captured)) if on]
        self.status.set(*st, '  ·  '.join(flags))
        self.status.vals['t'].set_text(f'{self.t:.1f} s')
        self.status.vals['a1'].set_text(f'{math.degrees(self.a[0]):.1f}°')
        self.status.vals['a2'].set_text(f'{math.degrees(self.a[1]):.1f}°')
        self.status.vals['mode'].set_text('captured' if self.captured else 'rolling')
        return ()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--screenshot', default=None, help='save a still and exit')
    ap.add_argument('--frames', type=int, default=1)
    ap.add_argument('--mission', action='store_true')
    ap.add_argument('--f', type=float, nargs=2, default=None, metavar=('F1', 'F2'),
                    help='initial frequencies in Hz')
    ap.add_argument('--no-guard', action='store_true')
    ap.add_argument('--start', type=float, nargs=2, default=None, metavar=('A1', 'A2'),
                    help='starting path angles in degrees')
    args = ap.parse_args()
    print(__doc__)
    sim = CS3Sim()
    if args.start:
        sim.a = np.radians(args.start)
        sim.trail = [sim.a.copy()]
    sim.guard = not args.no_guard
    if args.mission:
        sim._press(type('E', (), {'key': 'm'})())
    if args.f:
        sim.f_cmd[:] = args.f
    ui.screenshot_or_run(sim, args.screenshot, args.frames, int(DT * 1000))


if __name__ == '__main__':
    main()
