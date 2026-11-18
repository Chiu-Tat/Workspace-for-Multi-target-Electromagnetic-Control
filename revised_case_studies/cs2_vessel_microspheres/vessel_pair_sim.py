"""Autonomous interactive simulation of two microspheres in two thin vessels.

Shared by the ``interactive_*.py`` launchers (simple pairs and branched
networks).  See ``vessel_pair.py`` for the model and the missions.  The coil
array is drawn from its real pose (``shared/coils.py``); vessels and spheres
are at their true positions in the eMNS frame.  Spheres are drawn enlarged.

The TFW card shows the certified TFW of ``tfw.py`` in the configuration
(position of sphere 1 along its vessel, position of sphere 2 along its
vessel): certified boxes, undetermined boxes, points where the exact
pointwise test fails, the mission path and the live configuration.

Controls
--------
  SPACE  pause / resume      X  restart      I  interaction on / off
  P      mission planned in the TFW <-> straight mission (restarts)
  +/-    playback speed      ESC quit
"""

from __future__ import annotations

import argparse
import math

import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.patches import Circle, Polygon

from revised_case_studies.shared import coils, ui
from revised_case_studies.cs2_vessel_microspheres import config as K
from revised_case_studies.cs2_vessel_microspheres.vessel_pair import DT_CTRL, PairController, VesselPair

FPS = 20.0
DT = 1.0 / FPS
NET = {'artery': 0, 'vein': 1}
C_LUMEN = ('#e3a39b', '#a9b8d6')           # artery / vein fill
C_WALL = ('#8c2d2d', '#3d4f7a')
DRAW_R_SYS = 2.2                           # mm, sphere marker radius in system views
DRAW_R_ZOOM = 0.9                          # mm, in the vessel zoom (true radius 0.25 mm)
C_FIELD = '#6a3fa0'                        # net-field arrows
B_ARROW = 1.0                              # mm of arrow per mT (vessel zoom)
B_ARROW_SEC = 0.35                         # mm of arrow per mT (section view)
KEY_HINTS = ((('Space',), 'pause'), (('X',), 'restart'), (('P',), 'TFW / straight plan'),
             (('I',), 'interaction'),
             (('+', '−'), 'speed'), (('Esc',), 'quit'))
SHORT = {'RA': 'artery trunk', 'RV': 'vein trunk'}


def _band(pts_mm, r_mm, u=0, v=1):
    """Outline of a tube around a polyline (radius scalar or per point), projected on (u, v)."""
    p = pts_mm[:, [u, v]]
    d = np.gradient(p, axis=0)
    d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-12
    n = np.column_stack([-d[:, 1], d[:, 0]])
    r = np.broadcast_to(np.asarray(r_mm, float), (len(p),))[:, None]
    return np.vstack([p + r * n, (p - r * n)[::-1]])


class PairSim:
    def __init__(self, variant, plan='tfw'):
        self.variant = variant
        self.pair = VesselPair(variant, plan)
        self.zc = float(self.pair.center[2] * 1e3)             # mm
        self.ctl = PairController(self.pair)
        self.paused = False
        self.speed = 1.0
        self.frame = 0
        self.hist = []
        v = self.pair.vessel
        s = np.linspace(0, 1, 241)
        self.branch_pts = {b: v.centerline(b, s) * 1e3 for b in v.branches}
        self.branch_rad = {b: v.radius(b, s) * 1e3 for b in v.branches}
        self.branch_net = {b: NET.get(cfg.get('network'), 0 if b.startswith('RA') else 1)
                           for b, cfg in v.branches.items()}
        self._build()

    # --------------------------------------------------------------- layout
    def _build(self):
        ui.apply_style()
        fig = self.fig = ui.plt.figure(figsize=(ui.FIG_W, ui.FIG_H))
        if fig.canvas.manager is not None:
            fig.canvas.manager.set_window_title(f'eMNS · Case Study 2 · {self.pair.variant}')
        ui.chrome(fig, f'Case Study 2  ·  {self.pair.title}',
                  f'5.4 cP water-glycerin   ·   r = {K.RADIUS * 1e6:.0f} µm   ·   '
                  f'autonomous, both spheres at once   ·   10 coils, |I| ≤ {K.I_LIMIT:.0f} A',
                  KEY_HINTS)
        top = ui.FIG_H - 0.80
        bottom = 0.92
        ax_w = 4.9
        self.ax_top = fig.add_axes(ui.fig_rect(0.85, bottom + 2.55, ax_w, top - bottom - 2.55))
        self.ax_side = fig.add_axes(ui.fig_rect(0.85, bottom, ax_w, 2.05))
        xb = 0.85 + ax_w + 0.75
        wb = 3.35
        self.ax_zoom = fig.add_axes(ui.fig_rect(xb, bottom + 2.05, wb, top - bottom - 2.05))
        self.ax_sec = fig.add_axes(ui.fig_rect(xb, bottom, wb, 1.55))
        xc = xb + wb + 0.55
        wc = ui.FIG_W - ui.MARGIN - xc
        heights = np.array([0.80, 2.35, 1.30, 1.25, 1.13])
        bots = top - np.cumsum(heights) - ui.CARD_GAP * np.arange(len(heights))
        self.status = ui.StatusCard(ui.Card(fig, xc, bots[0], wc, heights[0]),
                                    (('t', 't', 0.0), ('d', 'd', 0.95), ('B', '|B|', 2.05)))
        self.map_card = ui.Card(fig, xc, bots[1], wc, heights[1],
                                'Certified TFW (positions along the two vessels, mm)')
        self.track_card = ui.Card(fig, xc, bots[2], wc, heights[2],
                                  'Position along each route (mm)')
        self.force_card = ui.Card(fig, xc, bots[3], wc, heights[3],
                                  'Tangential force (µN): request · allocator · plant')
        self.coil_card = ui.Card(fig, xc, bots[4], wc, heights[4], 'Coil currents')
        self._build_system()
        self._build_zoom()
        self._build_section()
        self._build_map()
        self._build_track()
        self._build_force()
        self.bars = ui.coil_axes(self.coil_card, i_max=K.I_LIMIT)
        fig.canvas.mpl_connect('key_press_event', self._press)

    def _style(self, ax, title):
        ui.style_scene(ax)
        ax.grid(False)
        ax.set_title(title, fontsize=8.5, color=ui.C_MUTED, loc='left', pad=4)

    def _build_system(self):
        for ax, view, (u, v), lims, title in (
                (self.ax_top, 'xy', (0, 1), (-105, 105, -105, 105), 'System, top view (x–y, mm)'),
                (self.ax_side, 'xz', (0, 2), (-105, 105, -35, 140), 'System, side view (x–z, mm)')):
            self._style(ax, title)
            coils.draw(ax, view, alpha=0.85, fade_below=20.0 if view == 'xy' else None, zorder=0.3)
            for b, pts in self.branch_pts.items():
                k = self.branch_net[b]
                ax.plot(pts[:, u], pts[:, v], color=C_WALL[k], lw=1.4 if b in ('RA', 'RV') else 0.8,
                        zorder=3)
            ax.set_xlim(*lims[:2])
            ax.set_ylim(*lims[2:])
            ax.set_aspect('equal')
        self.sys_dots = {view: [Circle((0, 0), DRAW_R_SYS, fc=ui.C_ROBOT[k], ec='k', lw=0.5, zorder=6)
                                for k in range(2)] for view in ('xy', 'xz')}
        for view, ax in (('xy', self.ax_top), ('xz', self.ax_side)):
            for d in self.sys_dots[view]:
                ax.add_patch(d)
        self.ax_top.annotate(f'vessels centred at the array centre, z = {self.zc:.1f} mm', (0, -38),
                             fontsize=7, color=ui.C_MUTED, ha='center', zorder=7)

    def _build_zoom(self):
        ax = self.ax_zoom
        self._style(ax, 'Vessels, top view (mm) — spheres enlarged, purple = net field B')
        ax.set_facecolor('#efe3d6')
        order = sorted(self.branch_pts, key=lambda b: self.branch_pts[b][:, 2].mean())  # deeper first
        for zi, b in enumerate(order):
            k = self.branch_net[b]
            ax.add_patch(Polygon(_band(self.branch_pts[b], self.branch_rad[b]), closed=True,
                                 fc=C_LUMEN[k], ec=C_WALL[k], lw=0.8, alpha=0.95, zorder=2 + 0.01 * zi))
        for k, rt in enumerate(self.pair.routes):
            q = rt.pts * 1e3
            ax.plot(q[:, 0], q[:, 1], ls=(0, (2, 2)), color=ui.C_ROBOT[k], lw=0.9, zorder=4)
        for b in ('RA', 'RV'):
            end = self.branch_pts[b][-1]
            ax.annotate(self.pair.vessel.branches[b].get('label', b), end[:2], xytext=(0, 6),
                        textcoords='offset points', fontsize=7, color=C_WALL[self.branch_net[b]],
                        ha='center', zorder=9)
        pc = [self.pair.routes[k].point(self.pair.a_close[k]) * 1e3 for k in range(2)]
        mid = (pc[0] + pc[1]) / 2
        ax.plot(*mid[:2], marker='+', ms=9, color=ui.C_INK, zorder=8)
        ax.annotate(f'closest: {self.pair.d_close * 1e3:.2f} mm', mid[:2], xytext=(-62, -16),
                    textcoords='offset points', fontsize=6.8, color=ui.C_INK, zorder=9)
        allp = np.vstack(list(self.branch_pts.values()))
        cx = (allp[:, 0].min() + allp[:, 0].max()) / 2
        half_w = max(17, (allp[:, 0].max() - allp[:, 0].min()) / 2 + 2)
        ax.set_xlim(cx - half_w, cx + half_w)
        ax.set_ylim(-31, 31)
        ax.set_aspect('equal', adjustable='datalim')
        self.zoom_dots = [Circle((0, 0), DRAW_R_ZOOM, fc=ui.C_ROBOT[k], ec='k', lw=0.6, zorder=10)
                          for k in range(2)]
        self.zoom_ref = [ax.plot([], [], 'o', ms=6, mfc='none', mec=ui.C_ROBOT[k], mew=1.2,
                                 zorder=11)[0] for k in range(2)]
        self.zoom_trail = [ax.plot([], [], color=ui.C_ROBOT[k], lw=1.4, alpha=0.8, zorder=9)[0]
                           for k in range(2)]
        self.zoom_F = [ui.arrow(ui.C_F, lw=1.2, size=7, zorder=12) for _ in range(2)]
        self.zoom_B = [ui.arrow(C_FIELD, lw=1.6, size=9, zorder=13) for _ in range(2)]
        self.zoom_Btxt = [ax.text(0, 0, '', fontsize=7, color=C_FIELD, zorder=14, ha='left',
                                  va='center', fontweight='medium') for _ in range(2)]
        for d in self.zoom_dots:
            ax.add_patch(d)
        for a in self.zoom_F + self.zoom_B:
            ax.add_patch(a)
        # scale bar for the field arrows (limits read after the aspect is applied)
        ax.apply_aspect()
        x0, x1 = ax.get_xlim()
        y0 = ax.get_ylim()[0] + 0.02 * (ax.get_ylim()[1] - ax.get_ylim()[0])
        ax.add_patch(ui.arrow(C_FIELD, lw=1.6, size=9, zorder=13))
        ax.patches[-1].set_positions((x0 + 1.5, y0 + 2.0), (x0 + 1.5 + 5 * B_ARROW, y0 + 2.0))
        ax.text(x0 + 2.0 + 5 * B_ARROW, y0 + 2.0, '5 mT', fontsize=6.8, color=C_FIELD, va='center')

    def _build_section(self):
        ax = self.ax_sec
        pc = [self.pair.routes[k].point(self.pair.a_close[k]) for k in range(2)]
        self.y_sec = float(np.mean([p[1] for p in pc]))
        self._style(ax, f'Section at the closest point, y = {self.y_sec * 1e3:.1f} mm '
                        f'(x–z, true scale)')
        ax.set_facecolor('#efe3d6')
        for k in range(2):
            rt = self.pair.routes[k]
            r = float(np.interp(self.pair.a_close[k], rt.arc, rt.radius)) * 1e3
            p = pc[k] * 1e3
            ax.add_patch(Circle((p[0], p[2] - self.zc), r, fc=C_LUMEN[k], ec=C_WALL[k], lw=1.0,
                                zorder=2))
        xm = float(np.mean([p[0] for p in pc])) * 1e3
        ax.set_xlim(xm - 4.5, xm + 4.5)
        ax.set_ylim(-4.2, 4.2) if self.pair.branched else ax.set_ylim(-2.4, 2.4)
        ax.set_aspect('equal', adjustable='datalim')
        ax.set_xlabel('x (mm)', fontsize=7.5, color=ui.C_MUTED, labelpad=1)
        ax.set_ylabel(f'z − {self.zc:.1f} mm', fontsize=7.5, color=ui.C_MUTED, labelpad=1)
        self.sec_dots = [Circle((0, 0), K.RADIUS * 1e3, fc=ui.C_ROBOT[k], ec='k', lw=0.5, zorder=5)
                         for k in range(2)]
        for d in self.sec_dots:
            ax.add_patch(d)
        # net field in the section plane (x, z) as an arrow; its y part as a dot (out of the page)
        self.sec_B = [ui.arrow(C_FIELD, lw=1.3, size=7, zorder=7) for _ in range(2)]
        for a in self.sec_B:
            ax.add_patch(a)
        self.sec_Bdot = [ax.plot([], [], marker='o', ms=2.5, color=C_FIELD, ls='none', zorder=8)[0]
                         for _ in range(2)]
        ax.text(0.99, 0.96, 'purple: B in x–z · dot: B along +y (out of page)', transform=ax.transAxes,
                ha='right', va='top', fontsize=6.5, color=C_FIELD)

    def _build_map(self):
        card, pair = self.map_card, self.pair
        ax = self.map_ax = card.inset(left=0.62, bottom=0.42, right=2.05, top=0.46)
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_color(ui.C_LINE)
        ax.tick_params(labelsize=6.8)
        L = [r.length * 1e3 for r in pair.routes]
        ax.set_xlim(0, L[0])
        ax.set_ylim(0, L[1])
        ax.set_xlabel('sphere 1 (artery)', fontsize=7, color=ui.C_MUTED, labelpad=1)
        ax.set_ylabel('sphere 2 (vein)', fontsize=7, color=ui.C_MUTED, labelpad=1)
        d = pair.tfw
        ra, rv = pair.routes

        def rect(b):
            x0, x1 = ra.a_of_s(b[0]) * 1e3, ra.a_of_s(b[1]) * 1e3
            y0, y1 = rv.a_of_s(b[2]) * 1e3, rv.a_of_s(b[3]) * 1e3
            return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]

        if d is not None:
            for key, col in (('inside', '#9fcfa8'), ('undetermined', '#ddd5c7'), ('outside', '#e3b3a8')):
                if d[key]:
                    ax.add_collection(PolyCollection([rect(b) for b in d[key]], facecolors=col,
                                                     edgecolors='none', zorder=1))
            g = np.asarray(d['pointwise']['grid'])
            ok = np.asarray(d['pointwise']['feasible_s1_s2'], bool)     # [s1, s2]
            bad = np.argwhere(~ok)
            if bad.size:
                ax.plot([ra.a_of_s(g[i]) * 1e3 for i, _ in bad], [rv.a_of_s(g[j]) * 1e3 for _, j in bad],
                        'x', ms=2.2, mew=0.6, color=ui.C_BAD, zorder=2)
        self.map_ref, = ax.plot([], [], ls=(0, (3, 2)), color=ui.C_INK, lw=1.0, zorder=3)
        self.map_trail, = ax.plot([], [], color=ui.C_EXACT, lw=1.6, zorder=4)
        self.map_dot, = ax.plot([], [], 'o', ms=6, mec=ui.C_INK, zorder=5)
        self._update_map_path()
        x0 = card.width - 1.9
        for i, (col, txt, mk) in enumerate((('#9fcfa8', 'certified', 's'),
                                            ('#ddd5c7', 'undetermined', 's'),
                                            (ui.C_BAD, 'pointwise infeasible', 'x'),
                                            (ui.C_CERT, 'safe corridor', 's'),
                                            (ui.C_INK, 'mission path', '_'),
                                            (ui.C_EXACT, 'simulated', '_'))):
            y = card.height - 0.62 - i * 0.2
            card.ax.plot([x0], [y], marker=mk, ms=7, color=col, mew=1.4, ls='none')
            card.ax.text(x0 + 0.16, y, txt, fontsize=7, color=ui.C_MUTED, va='center')
        self.map_txt = card.ax.text(x0 - 0.06, card.height - 0.62 - 6 * 0.2 + 0.05, '', fontsize=6.8,
                                    color=ui.C_INK, va='top', linespacing=1.3)
        self._update_map_path()

    def _update_map_path(self):
        m = self.pair.mission
        self.map_ref.set_data(m[:, 1] * 1e3, m[:, 2] * 1e3)
        for art in getattr(self, 'corridor_art', []):
            art.remove()
        boxes = [b for leg in self.pair.corridors for b in leg]
        self.corridor_art = []
        if boxes:
            self.corridor_art.append(self.map_ax.add_collection(PolyCollection(
                [[(b[0] * 1e3, b[2] * 1e3), (b[1] * 1e3, b[2] * 1e3), (b[1] * 1e3, b[3] * 1e3),
                  (b[0] * 1e3, b[3] * 1e3)] for b in boxes],
                facecolors='none', edgecolors=ui.C_CERT, linewidths=0.5, alpha=0.8, zorder=2.5)))
        pair = self.pair
        if pair.tfw is None:
            txt = 'no TFW file:\nrun tfw.py'
        else:
            a = pair.tfw['area_fraction']
            txt = (f'plan: {"A* + corridor + Bezier" if pair.plan == "tfw" else "straight"}\n'
                   f'certified {a["inside"] * 100:.0f}% of q-space')
            if pair.snapped:
                txt += f'\nhold moved {max(pair.snapped.values()) * 1e3:.1f} mm (not certified)'
        if hasattr(self, 'map_txt'):
            self.map_txt.set_text(txt)

    def _build_track(self):
        card = self.track_card
        ax = self.track_ax = card.inset(left=0.62, bottom=0.42, right=0.22, top=0.52)
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_color(ui.C_LINE)
        m = self.pair.mission
        self.track_ref = [ax.plot(m[:, 0], m[:, k + 1] * 1e3, ls=(0, (3, 2)), color=ui.C_ROBOT[k],
                                  lw=1.0)[0] for k in range(2)]
        self._draw_phases()
        ax.set_xlim(0, self.pair.duration)
        ax.tick_params(labelsize=7)
        self.track_lines = [ax.plot([], [], color=ui.C_ROBOT[k], lw=1.8)[0] for k in range(2)]
        self.track_now = ax.axvline(0, color=ui.C_INK, lw=0.8)
        self.track_txt = card.ax.text(ui.CARD_PAD, 0.16, '', fontsize=7.2, color=ui.C_MUTED,
                                      va='center')

    def _build_force(self):
        card = self.force_card
        ax = self.force_ax = card.inset(left=0.75, bottom=0.40, right=0.22, top=0.42)
        ax.set_ylim(1.6, -0.6)
        ax.set_yticks([0, 1], ['sphere 1', 'sphere 2'])
        ax.tick_params(axis='y', labelsize=8, labelcolor=ui.C_INK)
        ax.tick_params(axis='x', labelsize=7)
        ax.axvline(0, color=ui.C_INK, lw=0.6, alpha=0.3)
        self.fb = [ax.barh(np.arange(2) + off, np.zeros(2), height=0.24, color=col, zorder=2)
                   for off, col in ((-0.26, ui.C_FAINT), (0.0, ui.C_BAR), (0.26, ui.C_F))]
        self.f_lim = 0.2
        self.force_txt = card.ax.text(ui.CARD_PAD, 0.16, '', fontsize=7.4, color=ui.C_MUTED,
                                      va='center')

    # ------------------------------------------------------------- controls
    def _press(self, e):
        k = (e.key or '').lower()
        if k == ' ':
            self.paused = not self.paused
        elif k == 'x':
            self.ctl.reset()
            self.hist = []
        elif k == 'p':
            plan = 'straight' if self.pair.plan == 'tfw' else 'tfw'
            self.pair = VesselPair(self.variant, plan)
            self.ctl.pair = self.pair
            self.ctl.reset()
            self.hist = []
            self._redraw_mission()
        elif k == 'i':
            self.ctl.interaction = not self.ctl.interaction
        elif k in ('+', '='):
            self.speed = min(self.speed * 2, 8.0)
        elif k in ('-', '_'):
            self.speed = max(self.speed / 2, 0.25)
        elif k == 'escape':
            ui.plt.close(self.fig)

    def _redraw_mission(self):
        m = self.pair.mission
        self._update_map_path()
        for line in self.track_ref:
            line.remove()
        self.track_ref = [self.track_ax.plot(m[:, 0], m[:, k + 1] * 1e3, ls=(0, (3, 2)),
                                             color=ui.C_ROBOT[k], lw=1.0)[0] for k in range(2)]
        for art in self.track_phase_art:
            art.remove()
        self._draw_phases()
        self.track_ax.set_xlim(0, self.pair.duration)

    def _draw_phases(self):
        ax = self.track_ax
        self.track_phase_art = []
        shade = ('#f3efe7', '#ffffff')
        for i, (a, b, name) in enumerate(p for p in self.pair.phases if p[1] > p[0]):
            self.track_phase_art.append(ax.axvspan(a, b, color=shade[i % 2], zorder=0))
            self.track_phase_art.append(ax.text((a + b) / 2, 1.02, name.split(' ')[0],
                                                transform=ax.get_xaxis_transform(), fontsize=6.3,
                                                color=ui.C_MUTED, ha='center', va='bottom'))

    # ------------------------------------------------------------ one frame
    def step(self):
        if self.paused or self.ctl.t >= self.pair.duration:
            return
        t_end = self.ctl.t + DT * self.speed
        while self.ctl.t < t_end - 1e-9:
            if self.ctl.sol is None or self.frame % max(1, int(round(DT_CTRL / DT))) == 0 \
                    or self.speed > 1:
                self.ctl.allocate()
            self.ctl.advance(min(DT_CTRL / 2, t_end - self.ctl.t))
        self.frame += 1
        self.hist.append((self.ctl.t, *self.ctl.a))

    def update(self, _frame):
        self.step()
        ctl, pair = self.ctl, self.pair
        p = ctl.positions() * 1e3
        for view, (u, v) in (('xy', (0, 1)), ('xz', (0, 2))):
            for k in range(2):
                self.sys_dots[view][k].center = (p[k, u], p[k, v])
        a_ref, _ = pair.reference(ctl.t)
        for k in range(2):
            rt = pair.routes[k]
            self.zoom_dots[k].center = p[k, :2]
            pr = rt.point(a_ref[k]) * 1e3
            self.zoom_ref[k].set_data([pr[0]], [pr[1]])
            t_hat = rt.frame(ctl.a[k])[0]
            f = ctl.ft_plant[k] * t_hat[:2] * 2e7          # mm per N (display)
            self.zoom_F[k].set_positions(tuple(p[k, :2]), tuple(p[k, :2] + np.clip(f, -8, 8)))
            self.zoom_F[k].set_visible(np.linalg.norm(f) > 1.2)
            dy = abs(p[k, 1] - self.y_sec * 1e3)
            self.sec_dots[k].center = (p[k, 0], p[k, 2] - self.zc)
            alpha = float(np.clip(1 - dy / 4, 0.0, 1.0))
            self.sec_dots[k].set_alpha(alpha)
            # net field at the sphere (plant: coils + the other sphere's dipole)
            B = ctl.B[k] * 1e3                                     # mT
            ang = math.degrees(math.acos(np.clip(B[1] / max(np.linalg.norm(B), 1e-12), -1, 1)))
            tip = p[k, :2] + B_ARROW * B[:2]
            self.zoom_B[k].set_positions(tuple(p[k, :2]), tuple(tip))
            self.zoom_B[k].set_visible(np.linalg.norm(B[:2]) > 0.05)
            self.zoom_Btxt[k].set_position((tip[0] + 0.6, tip[1]))
            self.zoom_Btxt[k].set_text(f'{np.linalg.norm(B):.1f} mT, {ang:.1f}°')
            c = np.array([p[k, 0], p[k, 2] - self.zc])
            self.sec_B[k].set_positions(tuple(c), tuple(c + B_ARROW_SEC * np.array([B[0], B[2]])))
            self.sec_B[k].set_visible(alpha > 0.05 and np.hypot(B[0], B[2]) > 0.05)
            self.sec_Bdot[k].set_data(([c[0]], [c[1]]) if alpha > 0.05 and B[1] > 0 else ([], []))
        if self.hist:
            h = np.array(self.hist)
            for k in range(2):
                pts = np.array([pair.routes[k].point(a) for a in h[-300:, k + 1]]) * 1e3
                self.zoom_trail[k].set_data(pts[:, 0], pts[:, 1])
                self.track_lines[k].set_data(h[:, 0], h[:, k + 1] * 1e3)
        self.track_now.set_xdata([ctl.t, ctl.t])
        if self.hist:
            self.map_trail.set_data(h[:, 1] * 1e3, h[:, 2] * 1e3)
        inside = pair.certified(ctl.a)
        self.map_dot.set_data([ctl.a[0] * 1e3], [ctl.a[1] * 1e3])
        self.map_dot.set_color(ui.C_CERT if inside else ui.C_BAD)
        self.map_card.meta.set_text('in the certified TFW' if inside else
                                    ('outside the certified TFW' if inside is not None else ''))
        self.map_card.meta.set_color(ui.C_OK if inside else ui.C_BAD)
        err = (ctl.a - a_ref) * 1e3
        self.track_txt.set_text(f't (s)   ·   dashed: reference, solid: simulated   ·   '
                                f'error {err[0]:+.2f} / {err[1]:+.2f} mm')

        req, alloc, plant_ = ctl.ft_req * 1e6, ctl.ft_alloc * 1e6, ctl.ft_plant * 1e6
        lim = max(0.1, np.abs(np.concatenate([req, alloc, plant_])).max() * 1.15)
        self.f_lim = max(lim, 0.95 * self.f_lim)
        self.force_ax.set_xlim(-self.f_lim, self.f_lim)
        for bars, vals in zip(self.fb, (req, alloc, plant_)):
            for bar, v in zip(bars, vals):
                bar.set_width(v)
        wall = ctl.wall_load() * 1e6
        self.force_txt.set_text(f'grey request · blue allocator · orange plant   ·   '
                                f'wall load {wall[0]:.2f} / {wall[1]:.2f} µN')

        sat = ui.update_coils(self.coil_card, self.bars, ctl.currents, K.I_LIMIT, 'slack QP · 10 Hz')
        d = float(np.linalg.norm(p[0] - p[1]))
        Bmag = np.linalg.norm(ctl.B, axis=1) * 1e3
        phase = pair.phase(ctl.t)
        if self.paused:
            st = ('warn', 'Paused', f'{phase} — Space resumes')
        elif ctl.t >= pair.duration:
            st = ('ok', 'Done', 'Mission complete — X restarts')
        elif sat:
            st = ('warn', 'Limit', f'{phase} — a coil is at its current limit')
        else:
            st = ('ok', 'Auto', phase[0].upper() + phase[1:])
        self.status.set(*st, f'{"TFW plan" if pair.plan == "tfw" else "straight plan"}  ·  '
                             f'interaction {"on" if ctl.interaction else "OFF"}  ·  ×{self.speed:g}')
        self.status.vals['t'].set_text(f'{ctl.t:.1f} s')
        self.status.vals['d'].set_text(f'{d:.2f} mm')
        self.status.vals['B'].set_text(f'{Bmag[0]:.1f} / {Bmag[1]:.1f} mT')
        return ()


def main(variant):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--plan', choices=['tfw', 'straight'], default='tfw')
    ap.add_argument('--screenshot', default=None, help='save a still and exit')
    ap.add_argument('--frames', type=int, default=1, help='frames before the screenshot')
    ap.add_argument('--speed', type=float, default=1.0, help='playback speed')
    args = ap.parse_args()
    sim = PairSim(variant, args.plan)
    sim.speed = args.speed
    print(__doc__)
    ui.screenshot_or_run(sim, args.screenshot, args.frames, int(DT * 1000))
