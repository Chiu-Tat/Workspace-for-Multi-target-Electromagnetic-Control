#!/usr/bin/env python3
# Run from the repository root:
#   python -m revised_case_studies.cs1_cylinder_interaction.demo_gif
"""
Case Study 1 -- scripted demo of the interactive simulator, saved as a GIF.

Uses the same window as ``interactive.py`` (free drive, one fixed magnet) but
replaces keyboard input with a fixed pose sequence for the free magnet:

  1  Rotate   the magnet turns a full circle in place at (-25, 0) mm, a point
              certified at every heading.  The certified slice drawn under it
              is the slice at the current heading, so the TFW changes.
  2  Move     the magnet keeps beta = 90 deg and translates around a loop that
              stays at least 3 mm inside that slice.  Position is not a slice
              coordinate, so the TFW drawn does not change.

The poses are prescribed (kinematic), not integrated from the plant; currents,
force ranges and labels are evaluated at each pose exactly as in the
interactive simulator.  Uses the sharp 3-D TFW only.
"""

from __future__ import annotations

import argparse
import math

import matplotlib
matplotlib.use('Agg')
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter

from revised_case_studies.shared import ui
from revised_case_studies.shared.paths import out_dir
from revised_case_studies.cs1_cylinder_interaction import common as c
from revised_case_studies.cs1_cylinder_interaction import interactive as sim_mod
from revised_case_studies.cs1_cylinder_interaction.interactive import CS1Sim

FPS = 12
ROT_CENTRE = (-0.025, 0.0)          # certified at all 64 heading levels
BETA_MOVE = math.pi / 2
LOOP_MM = ((-25, 0), (-25, -38), (30, -38), (-6, -30), (-6, 30), (-30, 35), (-25, 0))
ROT_STEP = math.radians(3.0)        # per frame
MOVE_STEP = 0.0025                  # m per frame
HOLD = 8                            # frames held at each phase change
PHASES = {
    'rotate': ('Rotate', '1  Rotate in place: the certified slice changes with heading β'),
    'move': ('Move', '2  Move at fixed β = 90°: the certified slice does not change'),
}


def script():
    """(N, 4) rows [x, y, beta, phase] with phase 0 = rotate, 1 = move."""
    rows = [(*ROT_CENTRE, BETA_MOVE, 0)] * HOLD
    n = int(round(2 * math.pi / ROT_STEP))
    rows += [(*ROT_CENTRE, BETA_MOVE + k * ROT_STEP, 0) for k in range(1, n + 1)]
    rows += [(*ROT_CENTRE, BETA_MOVE, 1)] * HOLD
    wp = np.asarray(LOOP_MM, float) / 1e3
    for a, b in zip(wp[:-1], wp[1:]):
        m = max(1, int(math.ceil(np.linalg.norm(b - a) / MOVE_STEP)))
        rows += [(*(a + (b - a) * s), BETA_MOVE, 1) for s in np.arange(1, m + 1) / m]
    rows += [rows[-1]] * HOLD
    return np.asarray(rows, float)


class CS1Demo(CS1Sim):
    def __init__(self):
        self.script = script()
        self.k = 0
        super().__init__(0)
        self.view = 'workspace'
        self._set_view()
        for b in self.buttons + self.layout_buttons:
            b.ax.set_visible(False)
        self.path_lines[1].set_visible(False)
        self.path_lines[2].set_visible(False)
        self.phase_txt = self.fig.text(
            self.buttons[0].ax.get_position().x0, self.buttons[0].ax.get_position().y0 + 0.012,
            '', fontsize=11.5, fontweight='medium', color=ui.C_INK, va='bottom',
            bbox=dict(boxstyle='round,pad=0.5,rounding_size=0.4', fc=ui.C_PANEL, ec=ui.C_LINE,
                      lw=0.8))
        self.x, self.y, self.beta = self.script[0, :3]

    def step(self):
        x, y, beta, ph = self.script[min(self.k, len(self.script) - 1)]
        if self.k > 0 and ph != self.script[self.k - 1, 3]:
            self.trail = [(x, y)]
            self.hist = []
        self.k += 1
        self.x, self.y, self.beta = x, y, beta
        self.beta_cmd = beta
        self.phase = 'rotate' if ph == 0 else 'move'
        A_hat = c.body_matrix(x, y, beta, self.layout)
        task = np.array([0.0, c.B_HOLD, 0.0, 0.0])
        sol = self.alloc.solve(A_hat[:, :c.NUM_COILS], A_hat[:, c.NUM_COILS], task, self.w,
                               self.currents)
        self.currents = sol['currents']
        F = (c.world_matrix(x, y, beta, self.layout) @ np.append(self.currents, 1.0))[3:]
        if math.dist(self.trail[-1], (x, y)) > 1e-5:
            self.trail = (self.trail + [(x, y)])[-sim_mod.TRAIL_LEN:]
        return sol, F, task, None

    def _build_why(self):
        """Replace the distance chart with certified slice area against heading."""
        card = self.why_card
        for t in card.ax.texts:
            if t.get_text() == 'Why it leaves the TFW':
                t.set_text('Certified slice area vs heading β')
        t = self.tfw
        self.area = t.stack.reshape(len(t.stack), -1).mean(axis=1) * 100
        deg = np.arange(len(self.area) + 1) * math.degrees(t.beta_step)
        ax = self.area_ax = card.inset(left=0.75, bottom=0.42, right=0.25, top=0.42)
        ax.step(deg, np.append(self.area, self.area[0]), where='post', color=ui.C_CERT, lw=1.4)
        ax.fill_between(deg, 0, np.append(self.area, self.area[0]), step='post',
                        color=ui.C_CERT_RGB, alpha=0.25, lw=0)
        ax.set_xlim(0, 360)
        ax.set_ylim(0, 100)
        ax.set_xticks(range(0, 361, 90), [f'{d}°' for d in range(0, 361, 90)])
        ax.set_yticks([0, 50, 100], ['0', '50', '100 %'])
        ax.tick_params(labelsize=7.2)
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_color(ui.C_LINE)
        self.area_vline = ax.axvline(0, color=ui.C_INK, lw=0.8)
        self.area_dot, = ax.plot([], [], 'o', ms=6, color=ui.C_INK, zorder=5)
        card.ax.text(ui.CARD_PAD, 0.15, 'fraction of the 100 × 100 mm square certified, '
                     'one level per 5.6° of heading', fontsize=6.8, color=ui.C_MUTED,
                     va='center')

    def _update_why(self, fr, lab):
        lvl = self.tfw.beta_level(self.beta)
        b = math.degrees(self.beta) % 360
        self.area_vline.set_xdata([b, b])
        self.area_dot.set_data([b], [self.area[lvl]])
        self.why_card.meta.set_text(f'{self.area[lvl]:.1f} % certified at β = {b:.0f}°')
        self.why_card.meta.set_color(ui.C_OK)

    def update(self, frame):
        out = super().update(frame)
        self.caption.set_text(self.caption.get_text().replace('Free drive', 'Scripted demo'))
        self.guard_txt.set_text('')
        return out

    def _update_status(self, rel, sat, lab):
        super()._update_status(rel, sat, lab)
        chip, text = PHASES[self.phase]
        self.status.set('ok', chip, 'Heading sweep, position fixed' if self.phase == 'rotate'
                        else 'Translation, heading fixed', self.status.flags.get_text())
        self.phase_txt.set_text(text)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default=str(out_dir('cs1') / 'demo_rotate_move.gif'))
    ap.add_argument('--dpi', type=int, default=80)
    ap.add_argument('--frames', type=int, default=None, help='limit frames (testing)')
    args = ap.parse_args()
    sim_mod.KEY_HINTS = ()
    sim = CS1Demo()
    n = args.frames or len(sim.script)
    anim = FuncAnimation(sim.fig, sim.update, frames=n, blit=False, cache_frame_data=False)
    anim.save(args.out, writer=PillowWriter(fps=FPS), dpi=args.dpi,
              savefig_kwargs={'facecolor': sim.fig.get_facecolor()})
    print(f'saved {args.out}  ({n} frames, {n / FPS:.1f} s)')


if __name__ == '__main__':
    main()
