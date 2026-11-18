#!/usr/bin/env python3
# Run from the repository root:
#   python -m revised_case_studies.cs1_cylinder_interaction.interactive
"""
Case Study 1 -- interactive: what the TFW means, and why a magnet leaves it.

One simulator, four scenarios (buttons at the top right, or keys 1-4):

  1  Free drive      drive the moving magnet yourself (W/S thrust, A/D turn);
                     choose 1, 2 or 3 fixed magnets (the Case 6 single, gate and
                     triangle layouts, each with its own sharp 3-D TFW)
  2  Planned path    closed-loop replay of the path planned in the sharp TFW
  3  Naive path      the straight reference that ignores the interaction
  4  Drive outside   as 1, but the guard is off: steer the magnet out of the
     the TFW         certified set (e.g. toward the fixed magnet) and watch the
                     consequences -- allocator slack, the reachable force no
                     longer covering the task, the interaction pull, contact.
                     The chart shows the cause against distance: the force the
                     coils must supply to cancel the interaction versus what
                     they can produce while holding the field.

What you see
  * the real ten-coil array around the workspace (Z: system / workspace view);
    the robots are at their true position, plane z = 82 mm;
  * Part A -- the certified slice at the current heading and the ring of
    certified headings at the current position;
  * Part B -- the reachable total thrust / lateral force at the magnet
    (coils + interaction) against the task [By, Bz, fy, fz] = [0, 20 mT, 0,
    +-0.1 N], the allocator slack, and the coil currents.

The guard (G) blocks moves that would leave the certified TFW.  It is on in
scenarios 1-2 and off in 3-4 (toggle any time).  W/A/S/D during scenario 2 or
3 take over manual control at the current pose.  Uses the sharp 3-D TFW only.

Controls
--------
  1 2 3 4  scenario          W / S  thrust            A / D  turn
  M        fixed magnets 1 -> 2 -> 3 (free drive only)
  SPACE    stop              G      guard on / off    Z      zoom
  C        clear trail       X      restart scenario  ESC    quit
"""

from __future__ import annotations

import argparse
import math

import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon, Rectangle
from matplotlib.widgets import Button

from revised_case_studies.shared import coils, ui
from revised_case_studies.cs1_cylinder_interaction import common as c
from revised_case_studies.cs1_cylinder_interaction.closed_loop import (
    ROW_SCALE, control_command, plant_step)
from revised_case_studies.cs1_cylinder_interaction.partB_paths import retime
from revised_case_studies.shared.allocator import QPAllocator, row_weights

FPS = 20.0
DT = 1.0 / FPS
N_SUB = 5
TURN_RATE = math.radians(60.0)
PATH_SPEED = 0.005                 # m/s, autopilot reference speed
MATCH_TOL = 1e-3                   # relative slack shown as 'matched'
TRAIL_LEN = 800
MAG_LEN = c.MAG_LEN
COMPASS_R = 0.0086
F_ARROW_SCALE = 0.20               # m per newton
F_ARROW_MAX = 0.025                # m, longest drawn arrow
VIEWS = {'system': 0.108, 'workspace': 0.058}
Z_PLANE_MM = 82.0
SCENARIOS = ('Free drive', 'Planned path', 'Naive path', 'Drive outside TFW')
MANUAL = (0, 3)                    # scenarios driven with W/S/A/D
LAYOUT_NAMES = (('single', '1 fixed magnet'), ('gate', '2 fixed magnets'),
                ('triangle', '3 fixed magnets'))
START = (0.01, -0.05, math.pi / 2)
GUARD_DEFAULT = (True, True, False, False)
LABEL_KIND = {'inside': 'ok', 'boundary': 'warn', 'uncertified': 'warn',
              'infeasible': 'bad', 'contact': 'bad'}
LABEL_TEXT = {'inside': 'certified: task deliverable',
              'boundary': 'certified, < 3 mm from edge',
              'uncertified': 'not certified (pointwise ok)',
              'infeasible': 'task not reachable here',
              'contact': 'in contact with fixed magnet'}
KEY_HINTS = ((('1', '2', '3', '4'), 'scenario'), (('M',), 'magnets'), (('W', 'S'), 'thrust'), (('A', 'D'), 'turn'),
             (('Space',), 'stop'), (('G',), 'guard'), (('Z',), 'zoom'), (('C',), 'clear'),
             (('X',), 'restart'), (('Esc',), 'quit'))


def _cap(v, vmax=F_ARROW_MAX):
    n = np.linalg.norm(v)
    return v if n <= vmax else v * (vmax / n)


class CS1Sim:
    def __init__(self, scenario=0):
        self.layout = self.free_layout = 'single'
        self.tfw = c.tfw(self.layout)
        self.alloc = QPAllocator(4)
        self.w = row_weights(ROW_SCALE)
        self.paths = {1: retime(c.load_trajectory(c.PLANNED), PATH_SPEED),
                      2: retime(c.load_trajectory(c.NAIVE), PATH_SPEED)}
        self.view = 'system'
        self.scenario = scenario
        self._build()
        self.select(scenario)

    # ---------------------------------------------------------------- state
    def select(self, k, layout=None):
        self.scenario = k
        self.guard = GUARD_DEFAULT[k]
        if k != 0:
            layout = 'single'
        else:
            layout = layout or self.free_layout
            self.free_layout = layout
        self._set_layout(layout)
        if k in MANUAL:
            self.x, self.y, self.beta = START
            if self.guard and not self._certified(*START):
                seed = self.tfw.find_seed(START[2])
                self.x, self.y = seed
        else:
            self.x, self.y, self.beta = self.paths[k][0, :3]
        self.beta_cmd = self.beta
        self.thrust = 0.0
        self.held = set()
        self.t_auto = 0.0
        self.contact = False
        self.blocked = False
        self.currents = np.zeros(c.NUM_COILS)
        self.trail = [(self.x, self.y)]
        self.hist = []
        self.first_out = None
        self.first_infeasible = None
        for i, b in enumerate(self.buttons):
            b.ax.set_facecolor(ui.C_CERT_RGB if i == k else ui.C_PANEL)
            b.color = ui.C_CERT_RGB if i == k else ui.C_PANEL
            b.label.set_fontweight('bold' if i == k else 'normal')
        for key, line in self.path_lines.items():
            line.set_alpha(0.8 if key == k else 0.15)
        for b in self.layout_buttons:
            b.ax.set_visible(k == 0)

    def _set_layout(self, layout):
        if layout == getattr(self, '_drawn_layout', None):
            return
        self.layout = layout
        self.tfw = c.tfw(layout)
        self._last_level = None
        for art in self.fixed_art:
            art.remove()
        self.fixed_art = []
        for fx, fy, fb, _ in c.fixed_magnets(layout):
            s, n = ui.magnet_polygons(fx, fy, fb, 0.011, 0.0055)
            self.fixed_art.append(self.ax.add_patch(Polygon(s, fc=ui.C_SOUTH, ec=ui.C_INK, lw=0.6, zorder=8)))
            self.fixed_art.append(self.ax.add_patch(Polygon(n, fc=ui.C_NORTH, ec=ui.C_INK, lw=0.6, zorder=8)))
            self.fixed_art.append(self.ax.annotate('fixed', (fx, fy), xytext=(10, -12),
                                                   textcoords='offset points', fontsize=7,
                                                   color=ui.C_MUTED, zorder=13))
        for b, (name, _) in zip(self.layout_buttons, LAYOUT_NAMES):
            on = name == layout
            b.ax.set_facecolor(ui.C_CERT_RGB if on else ui.C_PANEL)
            b.color = ui.C_CERT_RGB if on else ui.C_PANEL
            b.label.set_fontweight('bold' if on else 'normal')
        self._drawn_layout = layout

    # --------------------------------------------------------------- layout
    def _build(self):
        ui.apply_style()
        fig = self.fig = ui.plt.figure(figsize=(ui.FIG_W, ui.FIG_H))
        if fig.canvas.manager is not None:
            fig.canvas.manager.set_window_title('eMNS · Case Study 1 · interactive')
        ui.chrome(fig, 'Case Study 1  ·  TFW, orientation and interaction',
                  f'plane z = {Z_PLANE_MM:.0f} mm   ·   10 coils, |I| ≤ {c.I_MAX:.0f} A   ·   '
                  f'task [B_y, B_z, f_y, f_z] = [0, 20 mT, 0, ±0.1 N]', KEY_HINTS)
        left, bottom = 0.95, 0.92
        top = ui.FIG_H - 0.80
        side = top - bottom
        self.ax = fig.add_axes(ui.fig_rect(left, bottom, side, side))
        self._build_scene()

        x = left + side + 0.62
        w = ui.FIG_W - ui.MARGIN - x
        g = ui.CARD_GAP
        heights = np.array([0.40, 0.75, 1.35, 1.15, 1.80, 1.22])
        bottoms = top - np.cumsum(heights) - g * np.arange(len(heights))
        bw = (w - 3 * 0.08) / 4
        self.buttons = []
        for i, name in enumerate(SCENARIOS):
            bax = fig.add_axes(ui.fig_rect(x + i * (bw + 0.08), bottoms[0], bw, heights[0]))
            b = Button(bax, f'{i + 1}  {name}', color=ui.C_PANEL, hovercolor=ui.C_TRACK)
            b.label.set_fontsize(8.5)
            b.on_clicked(lambda _e, k=i: self.select(k))
            for sp in bax.spines.values():
                sp.set_color(ui.C_LINE)
            self.buttons.append(b)
        self.status = ui.StatusCard(ui.Card(fig, x, bottoms[1], w, heights[1]),
                                    (('x', 'x', 0.0), ('y', 'y', 0.9),
                                     ('beta', 'β', 1.8), ('d', 'd', 2.55)))
        self.task_card = ui.Card(fig, x, bottoms[2], w, heights[2], 'Body-frame task')
        self.budget_card = ui.Card(fig, x, bottoms[3], w, heights[3],
                                   'Actuation budget here (coils + interaction)')
        self.why_card = ui.Card(fig, x, bottoms[4], w, heights[4],
                                'Why it leaves the TFW')
        w1, w2 = w * 0.46, (w * 0.54 - 2 * g) / 2
        self.coil_card = ui.Card(fig, x, bottoms[5], w1, heights[5], 'Coil currents')
        self.head_card = ui.Card(fig, x + w1 + g, bottoms[5], w2, heights[5], 'Headings here')
        self.label_card = ui.Card(fig, x + w1 + w2 + 2 * g, bottoms[5], w2, heights[5],
                                  'This configuration')
        self._build_task()
        self._build_budget()
        self._build_why()
        self.bars = ui.coil_axes(self.coil_card)
        self._build_headings()
        self._build_label()
        fig.canvas.mpl_connect('key_press_event', self._press)
        fig.canvas.mpl_connect('key_release_event',
                               lambda e: self.held.discard((e.key or '').lower()))

    def _build_scene(self):
        ax = self.ax
        ui.style_scene(ax)
        ax.grid(False)
        coils.draw(ax, 'xy', scale=1e3, alpha=0.85, fade_below=20.0, zorder=0.3)
        self._set_view()
        ax.set_xlabel('x (mm)', fontsize=9, color=ui.C_MUTED)
        ax.set_ylabel('y (mm)', fontsize=9, color=ui.C_MUTED)
        ax.add_patch(Rectangle((-0.05, -0.05), 0.1, 0.1, fc=ui.C_SCENE, ec='#cfc7ba', lw=0.7,
                               alpha=0.93, zorder=0.9))
        ax.annotate('workspace plane z = 82 mm (coils 7-10 lie below)', (-0.05, 0.05),
                    xytext=(2, 3), textcoords='offset points', fontsize=7, color=ui.C_MUTED,
                    zorder=13)
        self.im = ax.imshow(np.zeros((2, 2, 4)), origin='lower', extent=self.tfw.extent,
                            interpolation='nearest', zorder=1)
        self.contour = None
        self._last_level = None
        self.path_lines = {}
        for key, col in ((1, ui.C_CERT), (2, ui.C_BAD)):
            p = self.paths[key]
            self.path_lines[key], = ax.plot(p[:, 0], p[:, 1], ls=(0, (4, 3)), color=col, lw=1.1,
                                            zorder=3)
        self.fixed_art = []
        self.layout_buttons = []
        bb = ax.get_position()
        for i, (name, text) in enumerate(LAYOUT_NAMES):
            bax = self.fig.add_axes([bb.x0 + 0.006 + i * 0.066, bb.y1 - 0.034, 0.062, 0.026],
                                    zorder=20)
            b = Button(bax, text, color=ui.C_PANEL, hovercolor=ui.C_TRACK)
            b.label.set_fontsize(7.5)
            b.on_clicked(lambda _e, n=name: self._choose_layout(n))
            for sp in bax.spines.values():
                sp.set_color(ui.C_LINE)
            self.layout_buttons.append(b)
        self.trail_lc = LineCollection([], linewidths=1.3, zorder=5)
        ax.add_collection(self.trail_lc)
        self.pg_s = Polygon(np.zeros((4, 2)), fc=ui.C_SOUTH, ec=ui.C_INK, lw=0.7, zorder=11)
        self.pg_n = Polygon(np.zeros((4, 2)), fc=ui.C_NORTH, ec=ui.C_INK, lw=0.7, zorder=11)
        ax.add_patch(self.pg_s)
        ax.add_patch(self.pg_n)
        self.compass = LineCollection([], linewidths=2.2, zorder=9)
        ax.add_collection(self.compass)
        self.arr_F = ui.arrow(ui.C_F)
        self.arr_I = ui.arrow(ui.C_BAD, lw=1.3, size=8, zorder=10)
        ax.add_patch(self.arr_F)
        ax.add_patch(self.arr_I)
        self.ref_dot, = ax.plot([], [], 'o', ms=5, mfc='none', mec=ui.C_INK, zorder=12)
        leg = ax.legend([Patch(fc=ui.C_CERT_RGB, alpha=0.55, ec=ui.C_CERT),
                         Patch(fc=coils.C_WIND, ec=coils.C_WIND_EDGE),
                         Line2D([], [], color=ui.C_F, lw=1.6),
                         Line2D([], [], color=ui.C_BAD, lw=1.3)],
                        ['certified slice (current β)', 'coil array (real pose)',
                         'total force', 'interaction force'],
                        loc='lower left', bbox_to_anchor=(0, 1), ncol=4, fontsize=7.8,
                        frameon=False, columnspacing=1.2, handlelength=1.4)
        for t in leg.get_texts():
            t.set_color(ui.C_MUTED)
        self.caption = ax.annotate('', xy=(1, 0), xycoords='axes fraction', xytext=(-6, 6),
                                   textcoords='offset points', ha='right', va='bottom',
                                   fontsize=7.5, color=ui.C_MUTED, zorder=14)

    def _set_view(self):
        lim = VIEWS[self.view]
        self.ax.set_xlim(-lim, lim)
        self.ax.set_ylim(-lim, lim)
        self.ax.set_aspect('equal', anchor='W')
        step = 0.05 if self.view == 'system' else 0.025
        ticks = np.arange(-math.floor(lim / step) * step, lim + 1e-9, step)
        self.ax.set_xticks(ticks, [ui.qty(t, 1e3, decimals=0) for t in ticks])
        self.ax.set_yticks(ticks, [ui.qty(t, 1e3, decimals=0) for t in ticks])

    def _build_task(self):
        card = self.task_card
        ax = card.inset(left=0.62, bottom=0.26, right=1.85, top=0.42)
        rows = np.arange(4)
        ax.set_xlim(-1.08, 1.08)
        ax.set_ylim(3.55, -0.55)
        ax.set_yticks(rows, [r'$B_y$', r'$B_z$', r'$f_y$', r'$f_z$'])
        ax.tick_params(axis='y', labelsize=9.5, labelcolor=ui.C_INK, pad=10)
        ax.set_xticks([-1, 0, 1], ['−full', '0', '+full'])
        ax.barh(rows, 2.0, left=-1.0, height=0.45, color=ui.C_TRACK, zorder=1)
        ax.axvline(0, color=ui.C_INK, lw=0.6, alpha=0.22, zorder=3)
        self.task_bars = ax.barh(rows, np.zeros(4), height=0.45, color=ui.C_BAR, zorder=2)
        self.task_cmd, = ax.plot(np.zeros(4), rows, ls='none', marker='|', ms=10, mew=1.6,
                                 color=ui.C_INK, zorder=4, clip_on=False)
        from matplotlib import transforms
        tr = transforms.blended_transform_factory(card.ax.transData, ax.transData)
        self.task_vals = [card.ax.text(card.width - 1.05, r, '', transform=tr, ha='right',
                                       va='center', fontsize=8) for r in rows]

    def _build_budget(self):
        card = self.budget_card
        ax = card.inset(left=0.9, bottom=0.40, right=0.25, top=0.44)
        ax.set_xlim(-0.35, 0.35)
        ax.set_ylim(1.7, -0.7)
        ax.set_yticks([0, 1], [r'$f_z$ range', r'$f_y$ range'])
        ax.tick_params(axis='y', labelsize=8.5, labelcolor=ui.C_INK, pad=8)
        ax.set_xticks([-0.3, -0.1, 0, 0.1, 0.3], ['−0.3', '−0.1', '0', '0.1', '0.3 N'])
        ax.barh([0, 1], 0.7, left=-0.35, height=0.36, color=ui.C_TRACK, zorder=1)
        self.rng_bars = ax.barh([0, 1], [0, 0], height=0.36, color=ui.C_CERT, zorder=2)
        ax.plot([-c.F_MAX, c.F_MAX], [0, 0], ls='none', marker='|', ms=15, mew=2,
                color=ui.C_INK, zorder=4)
        ax.plot([0], [1], ls='none', marker='|', ms=15, mew=2, color=ui.C_INK, zorder=4)
        self.inter_mark, = ax.plot([0, 0], [0, 1], ls='none', marker='D', ms=4.5,
                                   color=ui.C_BAD, zorder=5)
        self.budget_txt = card.ax.text(ui.CARD_PAD, 0.16, '', fontsize=7.6, color=ui.C_MUTED,
                                       va='center')

    def _build_why(self):
        card = self.why_card
        axes = []
        for k in range(2):
            b0 = 0.40 + k * 0.56
            ax = card.inset(left=0.9, bottom=b0, right=0.25, top=card.height - b0 - 0.48)
            ax.axhline(0, color=ui.C_INK, lw=0.5, alpha=0.3)
            ax.tick_params(labelsize=6.8)
            for sp in ax.spines.values():
                sp.set_visible(True)
                sp.set_color(ui.C_LINE)
            axes.append(ax)
            card.ax.text(ui.CARD_PAD, b0 + 0.24, ('thrust\nf_z (N)', 'lateral\nf_y (N)')[k],
                         fontsize=7.2, color=ui.C_INK, va='center')
        self.why_fz, self.why_fy = axes
        self.why_fy.set_xticklabels([])
        self.why_art = {}
        card.ax.text(ui.CARD_PAD, 0.13,
                     'x: distance (mm)  ·  green: coil force available  ·  red: coil force '
                     'needed  ·  amber: left TFW  ·  purple: infeasible',
                     fontsize=6.6, color=ui.C_MUTED, va='center')

    def _build_headings(self):
        card = self.head_card
        size = card.height - 0.46 - 0.12
        ax = card.inset(left=0.12, bottom=0.12, right=card.width - 0.12 - size, top=0.46)
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        self.head_ring = LineCollection([], linewidths=5.0, zorder=2)
        ax.add_collection(self.head_ring)
        self.head_needle, = ax.plot([0, 0], [0, 0], color=ui.C_INK, lw=1.3, zorder=3)
        self.head_big = card.ax.text(0.12 + size + 0.12, 0.62, '', fontsize=13, color=ui.C_INK,
                                     va='center')

    def _build_label(self):
        ax = self.label_card.ax
        self.label_chip = ax.text(0.12 + 0.55, 0.78, '', fontsize=8.5, fontweight='medium',
                                  ha='center', va='center',
                                  bbox=dict(boxstyle='round,pad=0.4,rounding_size=0.4',
                                            fc=ui.STATUS_FACE['ok'], ec=ui.STATUS_EDGE['ok'], lw=0.8))
        self.label_msg = ax.text(0.12, 0.44, '', fontsize=7.2, color=ui.C_MUTED, va='center')
        self.guard_txt = ax.text(0.12, 0.2, '', fontsize=7.6, va='center')

    # ------------------------------------------------------------- controls
    def _press(self, e):
        k = (e.key or '').lower()
        if k in ('1', '2', '3', '4'):
            self.select(int(k) - 1)
        elif k in ('w', 's', 'a', 'd'):
            if self.scenario not in MANUAL:
                keep = (self.x, self.y, self.beta, self.guard)
                self.select(0 if self.guard else 3)
                self.x, self.y, self.beta, self.guard = keep
                self.beta_cmd = self.beta
                self.trail = [(self.x, self.y)]
            self.held.add(k)
        elif k == ' ':
            self.held.clear()
            self.thrust = 0.0
        elif k == 'm' and self.scenario == 0:
            names = [n for n, _ in LAYOUT_NAMES]
            self._choose_layout(names[(names.index(self.layout) + 1) % len(names)])
        elif k == 'g':
            self.guard = not self.guard
        elif k == 'z':
            self.view = 'workspace' if self.view == 'system' else 'system'
            self._set_view()
        elif k == 'c':
            self.trail = [(self.x, self.y)]
            self.hist = []
        elif k == 'x':
            self.select(self.scenario)
        elif k == 'escape':
            ui.plt.close(self.fig)

    def _choose_layout(self, name):
        if self.scenario == 0:
            self.select(0, name)

    def _certified(self, x, y, beta):
        return (abs(x) <= 0.05 and abs(y) <= 0.05
                and self.tfw.contains(x, y, self.tfw.slice_at(beta)))

    # ------------------------------------------------------------ one frame
    def step(self):
        ref = None
        if self.scenario not in MANUAL:
            self.t_auto += DT
            self.beta_cmd, self.thrust, ref = control_command(self.paths[self.scenario],
                                                              self.t_auto, self.x, self.y)
        else:
            turn = (TURN_RATE if 'a' in self.held else 0) - (TURN_RATE if 'd' in self.held else 0)
            self.thrust = (c.F_MAX if 'w' in self.held else 0) - (c.F_MAX if 's' in self.held else 0)
            if turn:
                cand = self.beta_cmd + turn * DT
                if not self.guard or self._certified(self.x, self.y, cand):
                    self.beta_cmd = cand
        A_hat = c.body_matrix(self.x, self.y, self.beta_cmd, self.layout)
        task = np.array([0.0, c.B_HOLD, 0.0, self.thrust])
        sol = self.alloc.solve(A_hat[:, :c.NUM_COILS], A_hat[:, c.NUM_COILS], task, self.w,
                               self.currents)
        self.currents = sol['currents'] if not self.contact else np.zeros(c.NUM_COILS)
        F = np.zeros(3)
        self.blocked = False
        for _ in range(N_SUB):
            if self.contact:
                break
            x, y, beta, _, F = plant_step(self.x, self.y, self.beta, self.currents, DT / N_SUB,
                                          layout=self.layout)
            if self.guard and not self._certified(x, y, beta):
                self.blocked = True
                break
            self.x, self.y, self.beta = x, y, beta
            if c.nearest_fixed(x, y, self.layout)[0] < c.CONTACT_DIST:
                self.contact = True
        if not self.contact and math.dist(self.trail[-1], (self.x, self.y)) > 1e-5:
            self.trail = (self.trail + [(self.x, self.y)])[-TRAIL_LEN:]
        return sol, F, task, ref

    def _interaction_world(self):
        """In-plane force of the fixed magnet on the moving one (world frame)."""
        A_w = c.world_matrix(self.x, self.y, self.beta, self.layout)
        return A_w[3:5, c.NUM_COILS]

    def update(self, _frame):
        sol, F, task, ref = self.step()
        self._update_scene(F, ref)
        rel = self._update_task(sol, task)
        fr = c.force_ranges(self.x, self.y, self.beta, self.layout)
        self._update_budget(fr)
        lab = 'contact' if self.contact else c.label(self.x, self.y, self.beta, layout=self.layout)
        self._update_why(fr, lab)
        sat = ui.update_coils(self.coil_card, self.bars, self.currents, c.I_MAX, compact=True)
        self._update_heading_label(lab)
        self._update_status(rel, sat, lab)
        return ()

    # ------------------------------------------------------------ rendering
    def _update_scene(self, F, ref):
        x, y, beta = self.x, self.y, self.beta
        lvl = self.tfw.beta_level(beta)
        if lvl != self._last_level:
            self._last_level = lvl
            mask = self.tfw.stack[lvl]
            rgba = np.zeros(mask.shape + (4,))
            rgba[..., :3] = ui.C_CERT_RGB
            rgba[..., 3] = np.where(mask, 0.32, 0.0)
            self.im.set_data(rgba)
            if self.contour is not None:
                self.contour.remove()
            gx = np.linspace(-0.05, 0.05, mask.shape[0])
            self.contour = self.ax.contour(gx, gx, mask.astype(float), levels=[0.5],
                                           colors=[ui.C_CERT], linewidths=0.9, zorder=3)
        self.caption.set_text(f'{SCENARIOS[self.scenario]}   ·   '
                              f'{dict(LAYOUT_NAMES)[self.layout]}   ·   {self.view} view   ·   '
                              f'certified slice β {np.degrees(lvl * self.tfw.beta_step):.1f}°–'
                              f'{np.degrees((lvl + 1) * self.tfw.beta_step):.1f}°')
        s, nrt = ui.magnet_polygons(x, y, beta)
        self.pg_s.set_xy(s)
        self.pg_n.set_xy(nrt)
        ok = self.tfw.headings_at(x, y)
        self._heading_ok = ok
        segs, flags = ui.heading_arcs(ok, self.tfw.beta_step, COMPASS_R, x, y)
        self.compass.set_segments(segs)
        self.compass.set_color([ui.C_CERT if f else ui.C_BLOCK for f in flags])
        f = _cap(F[:2] * F_ARROW_SCALE)
        self.arr_F.set_positions((x, y), (x + f[0], y + f[1]))
        self.arr_F.set_visible(np.linalg.norm(f) > MAG_LEN / 2 and not self.contact)
        fi = _cap(self._interaction_world() * F_ARROW_SCALE)
        self.arr_I.set_positions((x, y), (x + fi[0], y + fi[1]))
        self.arr_I.set_visible(np.linalg.norm(fi) > MAG_LEN / 2 and not self.contact)
        self.ref_dot.set_data(([ref[0]], [ref[1]]) if ref else ([], []))
        pts = np.asarray(self.trail)
        if len(pts) > 1:
            segs = np.stack([pts[:-1], pts[1:]], axis=1)
            rgba = np.zeros((len(segs), 4))
            rgba[:, :3] = ui.C_TRAIL
            rgba[:, 3] = np.linspace(0.05, 0.8, len(segs))
            self.trail_lc.set_segments(segs)
            self.trail_lc.set_color(rgba)
        else:
            self.trail_lc.set_segments([])

    def _update_task(self, sol, task):
        scales = np.array(ROW_SCALE)
        ach = sol['achieved']
        good = np.abs(ach - task) / scales < MATCH_TOL
        self.task_cmd.set_xdata(np.clip(task / scales, -1, 1))
        for bar, v, gd, txt, a, unit in zip(self.task_bars, ach / scales, good, self.task_vals,
                                            ach, ('mT', 'mT', 'mN', 'mN')):
            bar.set_width(float(np.clip(v, -1, 1)))
            bar.set_color(ui.C_BAR if gd else ui.C_BAD)
            txt.set_text(f'{ui.qty(a, 1e3)} {unit}')
            txt.set_color(ui.C_INK if gd else ui.C_BAD)
        rel = float(np.max(np.abs(ach - task) / scales))
        self.task_card.meta.set_text('matched' if rel < MATCH_TOL
                                     else f'slack {rel * 100:.1f}% of full scale')
        self.task_card.meta.set_color(ui.C_OK if rel < MATCH_TOL else ui.C_BAD)
        return rel

    def _update_budget(self, fr):
        checks = (lambda lo, hi: lo <= -c.F_MAX and hi >= c.F_MAX,
                  lambda lo, hi: lo <= 0 <= hi)
        oks = []
        for bar, (lo, hi), fn in zip(self.rng_bars, (fr['fz_range'], fr['fy_range']), checks):
            if math.isnan(lo):
                bar.set_width(0)
                oks.append(False)
                continue
            lo_c, hi_c = max(lo, -0.35), min(hi, 0.35)
            bar.set_x(lo_c)
            bar.set_width(max(hi_c - lo_c, 0))
            oks.append(fn(lo, hi))
            bar.set_color(ui.C_CERT if oks[-1] else ui.C_BAD)
        self.inter_mark.set_xdata(np.clip([fr['fz_interaction'], fr['fy_interaction']], -0.35, 0.35))
        self.budget_txt.set_text(
            f'diamond = interaction alone: f_z {ui.qty(fr["fz_interaction"], 1e3, signed=True)} mN, '
            f'f_y {ui.qty(fr["fy_interaction"], 1e3, signed=True)} mN   ·   bars = required')
        good = all(oks)
        self.budget_card.meta.set_text('task covered' if good else 'task NOT covered')
        self.budget_card.meta.set_color(ui.C_OK if good else ui.C_BAD)

    def _update_why(self, fr, lab):
        d = c.nearest_fixed(self.x, self.y, self.layout)[0] * 1e3
        fy_i, fz_i = fr['fy_interaction'], fr['fz_interaction']
        # coil-only capability = total range minus the interaction offset
        fy_lo, fy_hi = (v - fy_i for v in fr['fy_range'])
        fz_lo, fz_hi = (v - fz_i for v in fr['fz_range'])
        if not self.contact:
            self.hist.append((d, fy_i, fy_lo, fy_hi, fz_i, fz_lo, fz_hi))
            if self.first_out is None and lab not in ('inside', 'boundary'):
                self.first_out = d
            if self.first_infeasible is None and lab == 'infeasible':
                self.first_infeasible = d
        if len(self.hist) < 2:
            return
        h = np.array(self.hist[-600:])
        h = h[np.argsort(-h[:, 0])]
        for ax, key, cols, need in ((self.why_fy, 'fy', (2, 3), [-h[:, 1]]),
                                    (self.why_fz, 'fz', (5, 6),
                                     [c.F_MAX - h[:, 4], -c.F_MAX - h[:, 4]])):
            for art in self.why_art.get(key, []):
                art.remove()
            lo, hi = h[:, cols[0]], h[:, cols[1]]
            valid = np.isfinite(lo) & np.isfinite(hi)
            arts = [ax.fill_between(h[:, 0], np.where(valid, lo, np.nan),
                                    np.where(valid, hi, np.nan), color=ui.C_CERT, alpha=0.30, lw=0)]
            for n_ in need:
                arts += ax.plot(h[:, 0], n_, color=ui.C_BAD, lw=1.2)
            if self.first_out is not None:
                arts.append(ax.axvline(self.first_out, color=ui.C_WARN, lw=0.9, ls=(0, (3, 2))))
            if self.first_infeasible is not None:
                arts.append(ax.axvline(self.first_infeasible, color='#7b4bb3', lw=0.9,
                                       ls=(0, (3, 2))))
            arts.append(ax.axvline(d, color=ui.C_INK, lw=0.8))
            vals = np.concatenate([lo[valid], hi[valid], *need])
            ymax = float(np.clip(np.nanmax(np.abs(vals)) * 1.1 if vals.size else 0.4, 0.4, 2.5))
            ax.set_ylim(-ymax, ymax)
            self.why_art[key] = arts
        x0, x1 = max(55, h[:, 0].max() + 1), max(2.0, h[:, 0].min() - 2)
        self.why_fy.set_xlim(x0, x1)
        self.why_fz.set_xlim(x0, x1)
        needed = -h[-1, 1]
        short_y = not (h[-1, 2] <= needed <= h[-1, 3])
        short_z = not (h[-1, 5] <= -c.F_MAX - h[-1, 4] and c.F_MAX - h[-1, 4] <= h[-1, 6])
        msg = ('lateral pull exceeds coils' if short_y else
               'thrust budget exhausted' if short_z else '')
        tail = ''
        if self.first_out is not None:
            tail = f'out of TFW {self.first_out:.0f} mm'
        if self.first_infeasible is not None:
            tail += f', infeasible {self.first_infeasible:.0f} mm'
        self.why_card.meta.set_text('   ·   '.join(m for m in (msg, tail) if m))
        self.why_card.meta.set_color(ui.C_BAD if msg else ui.C_WARN)

    def _update_heading_label(self, lab):
        ok = self._heading_ok
        segs, flags = ui.heading_arcs(ok, self.tfw.beta_step, 1.0)
        self.head_ring.set_segments(segs)
        self.head_ring.set_color([ui.C_CERT if f else ui.C_BLOCK for f in flags])
        self.head_needle.set_data([0, 0.6 * math.cos(self.beta)], [0, 0.6 * math.sin(self.beta)])
        self.head_big.set_text(f'{ok.mean() * 360:.0f}°')
        kind = LABEL_KIND[lab]
        p = self.label_chip.get_bbox_patch()
        p.set_facecolor(ui.STATUS_FACE[kind])
        p.set_edgecolor(ui.STATUS_EDGE[kind])
        self.label_chip.set_text(lab)
        self.label_chip.set_color(ui.STATUS_EDGE[kind])
        self.label_msg.set_text(LABEL_TEXT[lab])
        self.guard_txt.set_text(f'guard {"ON" if self.guard else "OFF"} (G)')
        self.guard_txt.set_color(ui.C_CERT if self.guard else ui.C_BAD)

    def _update_status(self, rel, sat, lab):
        x, y = self.x, self.y
        if self.contact:
            st = ('bad', 'Contact', 'Pulled onto the fixed magnet — X restarts')
        elif self.blocked:
            st = ('warn', 'Blocked', 'Guard stopped a step leaving the certified TFW')
        elif rel >= MATCH_TOL:
            st = ('bad', 'Slack', 'Allocator cannot meet the task here')
        elif self.scenario not in MANUAL:
            st = ('ok', 'Auto', f'{SCENARIOS[self.scenario]}: closed-loop tracking')
        elif self.thrust:
            st = ('ok', 'Drive', 'Thrust along heading')
        else:
            st = ('ok', 'Hold', 'Holding field, zero thrust')
        flags = [f for f, on in (('coil limit', sat),
                                 ('outside certified', lab not in ('inside', 'boundary')))
                 if on]
        self.status.set(*st, '  ·  '.join(flags))
        self.status.vals['x'].set_text(f'{ui.qty(x, 1e3, 1, signed=True)} mm')
        self.status.vals['y'].set_text(f'{ui.qty(y, 1e3, 1, signed=True)} mm')
        self.status.vals['beta'].set_text(f'{math.degrees(self.beta) % 360:.0f}°')
        self.status.vals['d'].set_text(f'{c.nearest_fixed(x, y, self.layout)[0] * 1e3:.1f} mm')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--scenario', type=int, default=1, choices=[1, 2, 3, 4])
    ap.add_argument('--view', choices=list(VIEWS), default='system')
    ap.add_argument('--screenshot', default=None, help='save a still and exit')
    ap.add_argument('--frames', type=int, default=1, help='frames before the screenshot')
    args = ap.parse_args()
    print(__doc__)
    sim = CS1Sim(args.scenario - 1)
    if args.view != sim.view:
        sim.view = args.view
        sim._set_view()
    ui.screenshot_or_run(sim, args.screenshot, args.frames, int(DT * 1000))


if __name__ == '__main__':
    main()
