#!/usr/bin/env python3
# Run from the repository root: python -m cases.case_8_two_sphere_control_hold.simulation.interactive
"""
interactive.py -- Interactive Case 8 sequential-control simulator.

Two identical permanent-magnet spheres share the plane z = 0.082 m.  One moves
while the other is control-held; the certified TFW for the current stage is
drawn underneath, so you can see where the moving sphere is allowed to go.

  stage 1  magnet 2 moves, magnet 1 is held at (+0.02, 0) m
  stage 2  magnet 1 moves, magnet 2 is held at (-0.02, 0) m

Returning magnet 2 home hands control over to magnet 1.

Everything is driven from coil currents.  Each frame:

  1. The keys (or a route) set a planar force on the moving sphere.  The force
     eases toward the keys, so the sphere starts and stops smoothly.
  2. The ten-output task -- field and force at both spheres, with Bz taken
     from the certified witness of the rectangle under the moving sphere -- is
     solved for the ten coil currents.
  3. The sphere moves with a drag-dominated law  v = F / c.

Motion is confined to the certified TFW.  A step is followed rectangle by
rectangle, so the whole step stays certified; one that would leave the set
stops at its edge and slides along it instead.

A click plans a route through adjacent certified rectangles and pulls it taut,
so the sphere travels in straight runs rather than a staircase.  It slows for
corners and settles onto the destination.

Controls
--------
  W A S D   drive the moving sphere (arrow keys also work; combine for diagonals)
  click     travel to a point in the certified TFW
  H         return the moving sphere home (stage 1 then hands over)
  SPACE     stop
  C         clear the trails
  X         reset to the start of stage 1
  ESC       quit

Run:
    python -m cases.case_8_two_sphere_control_hold.simulation.interactive
"""

from __future__ import annotations

import argparse
import math
import sys
import time

import numpy as np
import matplotlib
if any(arg == "--screenshot" or arg.startswith("--screenshot=") for arg in sys.argv):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import (
    ArrowStyle, Circle, FancyArrowPatch, Patch, Rectangle, Wedge,
)
from matplotlib.widgets import Button

from ..model import B_HOLD, I_MAX, I_MIN, NUM_COILS, XY_LIMIT, Z_PLANE
from ..model_aligned_bz import F_MOVE
from .model import (
    MAGNET1_HOME,
    MAGNET2_HOME,
    MAGNET2_START,
    STAGE1_JSON,
    STAGE2_JSON,
    CertifiedSlice,
    solve_control,
)

# ============================================================================
# Simulation tuning
# ============================================================================

FPS = 30.0
DT = 1.0 / FPS                      # s, nominal frame step
MAX_DT = 0.1                        # s, longest step taken after a slow frame

DRAG = 0.25                         # N s / m -- drag-dominated law v = F / c
DRIVE_FORCE = 0.004                 # N, commanded force: 16 mm/s
V_MAX = DRIVE_FORCE / DRAG          # m/s
FORCE_TAU = 0.12                    # s, time constant of the force easing
ACCEL = 0.04                        # m/s^2, speed ramp on a route
CORNER_MIN = 0.2                    # fraction of full speed kept at a U-turn
SLIDE_ANGLES = np.radians([15, -15, 30, -30, 45, -45, 60, -60, 75, -75])

TRAIL_LEN = 900                     # positions of history per sphere

DRIVE_KEYS = {
    "w": (0.0, 1.0), "up": (0.0, 1.0),
    "s": (0.0, -1.0), "down": (0.0, -1.0),
    "a": (-1.0, 0.0), "left": (-1.0, 0.0),
    "d": (1.0, 0.0), "right": (1.0, 0.0),
}

# Drawing scales
SPHERE_R = 0.0023                   # m, display radius only
HOLD_RING_R = 1.7 * SPHERE_R        # m, dashed ring around the held sphere
F_ARROW_SCALE = 2.5                 # m per newton: full drive is 10 mm
NOTICE_S = 2.5                      # s, how long a one-off message stays

# ============================================================================
# Layout -- all in inches, converted to figure fractions when placed
# ============================================================================

FIG_W, FIG_H = 15.8, 9.15
MARGIN = 0.45                       # outer side margin
CARD_PAD = 0.18                     # horizontal padding inside a card
CARD_GAP = 0.15                     # space between cards
TITLE_Y = 0.25                      # card top edge to the title's centre line

KEY_HINTS = (
    (("W", "A", "S", "D"), "drive"),
    (("Click",), "go to point"),
    (("H",), "return home"),
    (("Space",), "stop"),
    (("C",), "clear trails"),
    (("X",), "reset"),
    (("Esc",), "quit"),
)

# ============================================================================
# Palette
# ============================================================================

C_BG = "#f4f1ea"
C_PANEL = "#fcfbf7"
C_SCENE = "#faf8f3"
C_INK = "#23201b"
C_MUTED = "#6f685e"
C_FAINT = "#a29a8e"
C_LINE = "#ddd6ca"
C_GRID = "#ece6db"
C_TRACK = "#eee9df"

C_CERT = "#3d8a58"
C_CERT_RGB = (0.55, 0.73, 0.58)
C_OUT = "#d9aaa0"
C_OUT_RGB = (0.85, 0.62, 0.57)
C_SOUTH = "#3a5478"
C_NORTH = "#c45c4a"
C_F = "#c36b24"
C_TRAIL = (0.40, 0.36, 0.32)
C_BAR = "#4f6b8c"
C_OK = "#2c7a4b"
C_WARN = "#9a5b12"
C_BAD = "#a4332c"
C_BUSY = "#2a5a9a"

_STATUS_FACE = {
    "ok": "#e8f1ea",
    "warn": "#f5ecdc",
    "bad": "#f5e4e0",
    "busy": "#e7edf5",
}
_STATUS_EDGE = {
    "ok": C_OK,
    "warn": C_WARN,
    "bad": C_BAD,
    "busy": C_BUSY,
}


def _pick_font():
    wanted = ("Avenir Next", "Avenir", "Helvetica Neue", "SF Pro Text",
              "Arial", "DejaVu Sans")
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in wanted:
        if name in available:
            return name
    return "DejaVu Sans"


FONT = _pick_font()


def _apply_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [FONT, "DejaVu Sans"],
        # Set subscripted labels in the body font, not Computer Modern.
        "mathtext.fontset": "custom",
        "mathtext.rm": FONT,
        "mathtext.it": f"{FONT}:italic",
        "mathtext.bf": f"{FONT}:bold",
        "axes.unicode_minus": True,
        "axes.linewidth": 0.6,
        "axes.edgecolor": C_LINE,
        "axes.labelcolor": C_INK,
        "text.color": C_INK,
        "xtick.color": C_MUTED,
        "ytick.color": C_MUTED,
        "figure.facecolor": C_BG,
        "savefig.facecolor": C_BG,
    })
    # W/A/S/D and the arrows must reach the simulator, not the toolbar.
    for k in list(matplotlib.rcParams):
        if k.startswith("keymap."):
            matplotlib.rcParams[k] = []


def _fig_rect(left, bottom, width, height):
    """Inches from the figure's lower-left corner -> figure-fraction rect."""
    return [left / FIG_W, bottom / FIG_H, width / FIG_W, height / FIG_H]


class _Card:
    """A flat panel on the right-hand column.

    ``ax`` is the background; its data units are inches from the card's
    lower-left corner, so text inside a card is placed by measurement.
    ``inset`` adds a transparent plotting axes a given distance in from each
    edge.
    """

    def __init__(self, fig, left, bottom, width, height, title=None):
        self.fig = fig
        self.left, self.bottom = left, bottom
        self.width, self.height = width, height

        ax = self.ax = fig.add_axes(_fig_rect(left, bottom, width, height),
                                    zorder=0)
        ax.set_xlim(0, width)
        ax.set_ylim(0, height)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_navigate(False)
        ax.set_facecolor(C_PANEL)
        for sp in ax.spines.values():
            sp.set_color(C_LINE)
            sp.set_linewidth(0.7)

        if title:
            ax.text(CARD_PAD, height - TITLE_Y, title, fontsize=9.5,
                    fontweight="medium", color=C_INK, va="center")
        self.meta = ax.text(width - CARD_PAD, height - TITLE_Y, "",
                            fontsize=8, color=C_MUTED, ha="right", va="center")

    def inset(self, left, bottom, right, top):
        ax = self.fig.add_axes(
            _fig_rect(self.left + left, self.bottom + bottom,
                      self.width - left - right, self.height - bottom - top),
            zorder=1,
        )
        ax.patch.set_visible(False)
        ax.set_navigate(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.tick_params(length=0, labelsize=7.5, labelcolor=C_FAINT, pad=4)
        return ax


def _qty(value, scale=1.0, decimals=2, signed=False):
    """Round display values so solver noise does not show as -0.00."""
    v = value * scale
    if abs(v) < 0.5 * 10 ** (-decimals):
        v = 0.0
    s = f"{v:+.{decimals}f}" if signed else f"{v:.{decimals}f}"
    return s.replace("-", "−")


def _sci(value):
    """A magnitude in one-significant-figure scientific notation."""
    if value == 0.0:
        return "0"
    return f"{value:.0e}".replace("-", "−")


def _point_mm(p):
    return f"({_qty(p[0], 1e3, 0, signed=True)}, {_qty(p[1], 1e3, 0)}) mm"


def _arrow(color, lw=1.5, size=10, zorder=10):
    return FancyArrowPatch(
        (0, 0), (0, 0),
        arrowstyle=ArrowStyle("-|>", head_length=0.6, head_width=0.3),
        mutation_scale=size, lw=lw, color=color, zorder=zorder,
        shrinkA=0, shrinkB=0, joinstyle="miter", capstyle="butt",
    )


def _sentence(exc):
    text = str(exc)
    return text[:1].upper() + text[1:]


class _SphereKey:
    """Legend key: a small two-tone sphere, north half up.  Serves as both
    the legend handle and its handler."""

    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = -handlebox.xdescent, -handlebox.ydescent
        center = (x0 + handlebox.width / 2.0, y0 + handlebox.height / 2.0)
        r = 0.62 * handlebox.height
        parts = [
            Wedge(center, r, 0, 180, fc=C_NORTH, ec="none"),
            Wedge(center, r, 180, 360, fc=C_SOUTH, ec="none"),
            Circle(center, r, fill=False, ec=C_INK, lw=0.6),
        ]
        for p in parts:
            p.set_transform(handlebox.get_transform())
            handlebox.add_artist(p)
        return parts[0]


class _Route:
    """A certified polyline being followed, parameterised by arc length."""

    def __init__(self, points, action=None):
        self.points = np.asarray(points, dtype=float)
        seg = np.diff(self.points, axis=0)
        self.lengths = np.linalg.norm(seg, axis=1)
        self.dirs = seg / np.maximum(self.lengths, 1e-300)[:, None]
        self.cum = np.concatenate(([0.0], np.cumsum(self.lengths)))
        self.total = float(self.cum[-1])
        self.s = 0.0
        self.action = action

    @property
    def goal(self):
        return self.points[-1]

    def segment(self, s):
        k = int(np.searchsorted(self.cum, s, side="right")) - 1
        return min(max(k, 0), len(self.lengths) - 1)

    def point(self, s):
        k = self.segment(s)
        return self.points[k] + self.dirs[k] * (s - self.cum[k])

    def speed_limit(self, s):
        """Fastest speed from which the sphere can still slow, at ``ACCEL``,
        for every corner ahead and come to rest on the goal.  A corner allows
        less speed the sharper it turns."""
        limit = math.sqrt(2.0 * ACCEL * max(self.total - s, 0.0))
        horizon = V_MAX ** 2 / (2.0 * ACCEL)
        for j in range(self.segment(s) + 1, len(self.lengths)):
            gap = self.cum[j] - s
            if gap > horizon:
                break
            turn = float(np.dot(self.dirs[j - 1], self.dirs[j]))
            v_turn = V_MAX * max(CORNER_MIN, 0.5 * (1.0 + turn))
            limit = min(limit, math.sqrt(v_turn ** 2 + 2.0 * ACCEL * gap))
        return min(V_MAX, limit)


class Case8Sim:
    def __init__(self, stage1_path=STAGE1_JSON, stage2_path=STAGE2_JSON):
        self.slices = {
            1: CertifiedSlice(stage1_path),
            2: CertifiedSlice(stage2_path),
        }
        if not np.allclose(self.slices[1].held, MAGNET1_HOME):
            raise ValueError("stage 1 must hold magnet 1 at (0.02, 0)")
        if not np.allclose(self.slices[2].held, MAGNET2_HOME):
            raise ValueError("stage 2 must hold magnet 2 at (-0.02, 0)")
        self.rasters = {stage: self._rasterise(s) for stage, s in self.slices.items()}

        self.held_keys = set()
        self._reset_state()
        self._last_time = None
        self._shown_stage = None
        self.anim = None
        self._build_figure()

    def _reset_state(self):
        self.stage = 1
        self.positions = [MAGNET1_HOME.copy(), MAGNET2_START.copy()]
        self.trails = [[self.positions[0].copy()], [self.positions[1].copy()]]
        self.held_keys.clear()
        self.force = np.zeros(2)
        self.applied_force = np.zeros(2)
        self.speed = 0.0
        self.route = None
        self.m1_returned = False
        self.sliding = False
        self.blocked = False
        self.notice = None                  # (text, kind, seconds left)
        self.solution = solve_control(self.active_slice, self.active_position,
                                      (0.0, 0.0))

    # ------------------------------------------------------------------
    # Stage bookkeeping
    # ------------------------------------------------------------------

    @property
    def active_index(self):
        return 1 if self.stage == 1 else 0

    @property
    def held_index(self):
        return 1 - self.active_index

    @property
    def active_position(self):
        return self.positions[self.active_index]

    @property
    def active_slice(self):
        return self.slices[self.stage]

    def _sequence_step(self):
        """Index into the four-step sequence; 4 once it is complete."""
        action = self.route.action if self.route is not None else None
        if self.stage == 1:
            return 1 if action == "handover" else 0
        if action == "returned_m1":
            return 3
        return 4 if self.m1_returned else 2

    @staticmethod
    def _rasterise(slice_data):
        """RGBA image of the certified and proven-outside rectangles, plus the
        certified mask.  Every rectangle sits on a uniform grid whose cell is
        the smallest rectangle, so the image is exact."""
        every = np.vstack([b for b in (slice_data.boxes, slice_data.undetermined,
                                       slice_data.outside) if len(b)])
        cell = float(np.min(np.concatenate((every[:, 1] - every[:, 0],
                                            every[:, 3] - every[:, 2]))))
        n = int(round(2.0 * XY_LIMIT / cell))

        def burn(boxes):
            mask = np.zeros((n, n), dtype=bool)
            idx = np.rint((boxes + XY_LIMIT) / (2.0 * XY_LIMIT) * n).astype(int)
            for i0, i1, j0, j1 in np.clip(idx, 0, n):
                mask[j0:j1, i0:i1] = True
            return mask

        cert = burn(slice_data.boxes)
        out = burn(slice_data.outside) if len(slice_data.outside) else np.zeros_like(cert)
        rgba = np.zeros((n, n, 4))
        rgba[cert, :3] = C_CERT_RGB
        rgba[cert, 3] = 0.28
        rgba[out & ~cert, :3] = C_OUT_RGB
        rgba[out & ~cert, 3] = 0.30

        domain = (2.0 * XY_LIMIT) ** 2

        def area(boxes):
            if not len(boxes):
                return 0.0
            return float(np.sum((boxes[:, 1] - boxes[:, 0])
                                * (boxes[:, 3] - boxes[:, 2]))) / domain

        fractions = (area(slice_data.boxes), area(slice_data.undetermined),
                     area(slice_data.outside))
        return rgba, cert, fractions

    # ------------------------------------------------------------------
    # Figure construction
    # ------------------------------------------------------------------

    def _build_figure(self):
        _apply_style()

        fig = self.fig = plt.figure(figsize=(FIG_W, FIG_H))
        if fig.canvas.manager is not None:
            fig.canvas.manager.set_window_title("eMNS · Case 8 · control hold")

        # Header and footer live on one background axes measured in inches.
        chrome = fig.add_axes([0, 0, 1, 1], zorder=-1)
        chrome.set_xlim(0, FIG_W)
        chrome.set_ylim(0, FIG_H)
        chrome.set_axis_off()
        chrome.set_navigate(False)
        self._build_header(chrome)
        self._build_footer(chrome)

        plot_left, plot_bottom = 0.95, 0.92
        plot_top = FIG_H - 0.80
        side = plot_top - plot_bottom
        self.ax = fig.add_axes(_fig_rect(plot_left, plot_bottom, side, side))
        self._build_main_axes()

        # Right-hand column: cards stacked from the top of the plot down to
        # its bottom edge, so both columns share the same frame.
        x = plot_left + side + 0.62
        w = FIG_W - MARGIN - x
        half = (w - CARD_GAP) / 2.0
        heights = np.array([0.80, 1.42, 1.56, 1.55, 1.50])
        bottoms = plot_top - np.cumsum(heights) - CARD_GAP * np.arange(len(heights))

        self.status_card = _Card(fig, x, bottoms[0], w, heights[0])
        self.seq_card = _Card(fig, x, bottoms[1], w, heights[1], "Sequence")
        self.coil_card = _Card(fig, x, bottoms[2], w, heights[2], "Coil currents")
        self.sphere_card = _Card(fig, x, bottoms[3], half, heights[3], "Spheres")
        self.slice_card = _Card(fig, x + half + CARD_GAP, bottoms[3], half,
                                heights[3], "Certified slice")
        self.task_card = _Card(fig, x, bottoms[4], w, heights[4],
                               "Task at the spheres")

        self._build_status()
        self._build_sequence()
        self._build_coils()
        self._build_spheres()
        self._build_slice()
        self._build_task()

        fig.canvas.mpl_connect("key_press_event", self._on_press)
        fig.canvas.mpl_connect("key_release_event", self._on_release)
        fig.canvas.mpl_connect("button_press_event", self._on_click)

    def _build_header(self, ax):
        y = FIG_H - 0.32
        brand = ax.text(MARGIN, y, "eMNS", fontsize=13, fontweight="bold",
                        color=C_INK, va="center")
        ax.annotate("Case 8  ·  Two spheres, sequential control hold",
                    xy=(1, 0.5), xycoords=brand, xytext=(12, 0),
                    textcoords="offset points", va="center", fontsize=12,
                    color=C_MUTED)
        m = self.slices[1].moments[0]
        ax.text(FIG_W - MARGIN, y,
                f"plane z = {Z_PLANE * 1e3:.0f} mm   ·   "
                f"{NUM_COILS} coils, |I| ≤ {I_MAX:.0f} A   ·   "
                f"m = {m:g} A·m² each",
                ha="right", va="center", fontsize=8.5, color=C_FAINT)

    def _build_footer(self, ax):
        """Key hints as quiet key caps, each placed after the one before."""
        cap = dict(boxstyle="round,pad=0.35,rounding_size=0.3",
                   fc=C_PANEL, ec=C_LINE, lw=0.7)

        def chain(prev, s, gap, **kw):
            if prev is None:
                return ax.text(MARGIN + 0.04, 0.26, s, va="center", **kw)
            return ax.annotate(s, xy=(1, 0.5), xycoords=prev,
                               xytext=(gap, 0), textcoords="offset points",
                               va="center", **kw)

        prev, gap = None, 0
        for keys, action in KEY_HINTS:
            for key in keys:
                prev = chain(prev, key, gap, fontsize=7.5, color=C_INK, bbox=cap)
                gap = 8
            prev = chain(prev, action, 8, fontsize=8, color=C_MUTED)
            gap = 20

    def _build_main_axes(self):
        ax = self.ax
        ax.set_facecolor(C_SCENE)
        pad = 0.008
        ax.set_xlim(-XY_LIMIT - pad, XY_LIMIT + pad)
        ax.set_ylim(-XY_LIMIT - pad, XY_LIMIT + pad)
        ax.set_aspect("equal", anchor="W")
        ticks = np.linspace(-XY_LIMIT, XY_LIMIT, 5)
        labels = [_qty(t, 1e3, decimals=0) for t in ticks]
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        ax.set_xlabel("x (mm)", fontsize=9, color=C_MUTED, labelpad=6)
        ax.set_ylabel("y (mm)", fontsize=9, color=C_MUTED, labelpad=6)
        ax.tick_params(labelsize=8.5, labelcolor=C_MUTED, length=0, pad=6)
        ax.grid(True, color=C_GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for sp in ax.spines.values():
            sp.set_color(C_LINE)
            sp.set_linewidth(0.8)

        extent = (-XY_LIMIT, XY_LIMIT, -XY_LIMIT, XY_LIMIT)
        self.im_slice = ax.imshow(np.zeros((2, 2, 4)), origin="lower",
                                  extent=extent, interpolation="nearest", zorder=1)
        self.slice_contour = None
        ax.add_patch(Rectangle(
            (-XY_LIMIT, -XY_LIMIT), 2 * XY_LIMIT, 2 * XY_LIMIT, fill=False,
            ec="#cfc7ba", lw=0.7, zorder=4,
        ))

        for home in (MAGNET1_HOME, MAGNET2_HOME):
            ax.plot(*home, marker="+", ms=8, mew=1.0, color=C_MUTED, zorder=7)

        self.trail_lcs = []
        for _ in range(2):
            # Butt caps: round caps overlap at every joint of the fading
            # segments and make the trail look dotted, like the route line.
            lc = LineCollection([], linewidths=1.3, zorder=5, capstyle="butt")
            ax.add_collection(lc)
            self.trail_lcs.append(lc)

        self.route_line, = ax.plot([], [], color=C_INK, lw=1.0, alpha=0.6,
                                   ls=(0, (3.0, 2.5)), zorder=6)
        self.route_goal, = ax.plot([], [], marker="o", ms=6, mfc="none",
                                   mec=C_INK, mew=1.0, ls="none", alpha=0.8,
                                   zorder=6)

        self.arr_F = _arrow(C_F, zorder=9)
        ax.add_patch(self.arr_F)

        self.hold_ring = Circle((0, 0), HOLD_RING_R, fill=False, ec=C_MUTED,
                                lw=0.8, ls=(0, (2.0, 2.0)), zorder=10)
        ax.add_patch(self.hold_ring)

        self.spheres, self.sphere_labels, self.sphere_roles = [], [], []
        for index in (0, 1):
            north = Wedge(self.positions[index], SPHERE_R, 0, 180, fc=C_NORTH,
                          ec="none", zorder=11)
            south = Wedge(self.positions[index], SPHERE_R, 180, 360, fc=C_SOUTH,
                          ec="none", zorder=11)
            outline = Circle(self.positions[index], SPHERE_R, fill=False,
                             ec=C_INK, lw=0.7, zorder=12)
            for patch in (north, south, outline):
                ax.add_patch(patch)
            self.spheres.append((north, south, outline))
            label = ax.annotate(
                str(index + 1), xy=tuple(self.positions[index]), xytext=(19, 15),
                textcoords="offset points", ha="center", va="center",
                fontsize=8.5, fontweight="medium", color=C_INK, zorder=14,
            )
            role = ax.annotate(
                "", xy=(1, 0.5), xycoords=label, xytext=(4, 0),
                textcoords="offset points", va="center", fontsize=7.5,
                color=C_MUTED, zorder=14,
            )
            self.sphere_labels.append(label)
            self.sphere_roles.append(role)

        sphere_key = _SphereKey()
        handles = [
            Patch(facecolor=C_CERT_RGB, alpha=0.55, edgecolor=C_CERT, lw=0.8),
            Patch(facecolor=C_OUT_RGB, alpha=0.5, edgecolor="none"),
            Line2D([], [], color=C_INK, lw=1.0, alpha=0.6, ls=(0, (3.0, 2.5))),
            Line2D([], [], color=C_F, lw=1.6),
            sphere_key,
            Line2D([], [], marker="+", ms=7, mew=1.0, color=C_MUTED, ls="none"),
        ]
        labels = ["certified TFW", "proven outside", "route", r"$F$ drive",
                  "sphere, north up", "home"]
        leg = ax.legend(
            handles, labels, handler_map={_SphereKey: sphere_key},
            loc="lower left", bbox_to_anchor=(0.0, 1.0),
            ncol=len(handles), fontsize=8, frameon=False, borderaxespad=0.5,
            borderpad=0.0, handlelength=1.6, handleheight=0.6,
            handletextpad=0.6, columnspacing=1.5,
        )
        for txt in leg.get_texts():
            txt.set_color(C_MUTED)

    def _build_status(self):
        card = self.status_card
        ax, h = card.ax, card.height
        self.status_chip = ax.text(
            CARD_PAD + 0.36, h / 2, "", ha="center", va="center",
            fontsize=8.8, fontweight="medium", color=C_OK, zorder=3,
            bbox=dict(boxstyle="round,pad=0.55,rounding_size=0.55",
                      fc=_STATUS_FACE["ok"], ec=_STATUS_EDGE["ok"], lw=0.8),
        )
        x_msg = CARD_PAD + 0.86
        self.status_msg = ax.text(x_msg, h * 0.66, "", fontsize=9.5,
                                  color=C_INK, va="center")
        self.status_flags = ax.text(card.width - CARD_PAD, h * 0.66, "",
                                    fontsize=8, ha="right", va="center")
        self.status_vals = {}
        for key, dx in (("x", 0.0), ("y", 0.86), ("v", 1.72), ("r", 2.62)):
            ax.text(x_msg + dx, h * 0.31, key, fontsize=8, color=C_FAINT,
                    va="center")
            self.status_vals[key] = ax.text(x_msg + dx + 0.15, h * 0.31, "",
                                            fontsize=8.2, color=C_INK,
                                            va="center")

    def _build_sequence(self):
        card = self.seq_card
        ax, h = card.ax, card.height
        steps = (
            f"Drive magnet 2 while magnet 1 is held at {_point_mm(MAGNET1_HOME)}",
            f"Return magnet 2 home to {_point_mm(MAGNET2_HOME)} and hand over",
            f"Drive magnet 1 while magnet 2 is held at {_point_mm(MAGNET2_HOME)}",
            f"Return magnet 1 home to {_point_mm(MAGNET1_HOME)}",
        )
        ys = h - 0.55 - 0.21 * np.arange(len(steps))
        mx = CARD_PAD + 0.06
        ax.plot([mx, mx], [ys[0], ys[-1]], color=C_LINE, lw=1.0, zorder=1)
        self.seq_marks, self.seq_texts = [], []
        for y, text in zip(ys, steps):
            mark, = ax.plot([mx], [y], marker="o", ms=6.5, mew=1.0,
                            ls="none", zorder=3)
            self.seq_marks.append(mark)
            self.seq_texts.append(ax.text(CARD_PAD + 0.24, y, text,
                                          fontsize=8.4, va="center"))

        bw, bh = 2.05, 0.34
        bx = card.left + card.width - CARD_PAD - bw
        self.home_button = self._button(
            _fig_rect(bx, card.bottom + 0.62, bw, bh), "",
            face="#edf3ee", hover="#e0ebe2", edge="#b9d3c0",
        )
        self.home_button.on_clicked(lambda _event: self._return_home())
        self.reset_button = self._button(
            _fig_rect(bx, card.bottom + 0.18, bw, bh), "Reset",
            face=C_PANEL, hover=C_TRACK, edge=C_LINE,
        )
        self.reset_button.on_clicked(lambda _event: self._reset())

    def _button(self, rect, label, face, hover, edge):
        ax = self.fig.add_axes(rect, zorder=2)
        ax.set_navigate(False)
        button = Button(ax, label, color=face, hovercolor=hover)
        for sp in ax.spines.values():
            sp.set_color(edge)
            sp.set_linewidth(0.8)
        button.label.set_fontsize(8.5)
        button.label.set_color(C_INK)
        return button

    def _build_coils(self):
        ax = self.coil_card.inset(left=0.66, bottom=0.30, right=CARD_PAD, top=0.50)
        idx = np.arange(1, NUM_COILS + 1)
        ax.set_xlim(0.4, NUM_COILS + 0.6)
        ax.set_ylim(I_MIN * 1.22, I_MAX * 1.22)
        ax.set_xticks(idx)
        ax.set_yticks([I_MIN, 0.0, I_MAX])
        ax.set_yticklabels([f"{_qty(I_MIN, decimals=0)} A", "0",
                            f"+{I_MAX:.0f} A"])
        for lim in (I_MIN, I_MAX):
            ax.axhline(lim, color=C_BAD, lw=0.8, ls=(0, (3.0, 2.5)),
                       alpha=0.5, zorder=1)
        ax.axhline(0.0, color=C_INK, lw=0.6, alpha=0.25, zorder=3)
        self.bars = ax.bar(idx, np.zeros(NUM_COILS), width=0.56, color=C_BAR,
                           edgecolor="none", zorder=2)

    def _build_spheres(self):
        card = self.sphere_card
        ax, h = card.ax, card.height
        cols = (1.45, 2.15, card.width - CARD_PAD)
        for label, cx in zip(("x (mm)", "y (mm)", "to home"), cols):
            ax.text(cx, h - 0.52, label, fontsize=7.8, color=C_FAINT,
                    ha="right", va="center")
        self.sphere_rows = []
        for index, y in ((0, 0.78), (1, 0.53)):
            ax.text(CARD_PAD, y, str(index + 1), fontsize=9,
                    fontweight="medium", color=C_INK, va="center")
            role = ax.text(CARD_PAD + 0.16, y, "", fontsize=8, color=C_MUTED,
                           va="center")
            values = [ax.text(cx, y, "", fontsize=8.6, color=C_INK,
                              ha="right", va="center") for cx in cols]
            self.sphere_rows.append((role, values))
        ax.text(CARD_PAD, 0.25, "separation", fontsize=8, color=C_MUTED,
                va="center")
        self.sphere_sep = ax.text(cols[1], 0.25, "", fontsize=8.6, color=C_INK,
                                  ha="right", va="center")

    def _build_slice(self):
        card = self.slice_card
        ax = card.ax
        x0 = 1.02
        self.slice_track_w = card.width - 0.62 - x0
        self.slice_fill, self.slice_vals = [], []
        for label, y, color in (("certified", 1.00, C_CERT),
                                ("undetermined", 0.77, C_FAINT),
                                ("outside", 0.54, C_OUT)):
            ax.text(CARD_PAD, y, label, fontsize=8.2, color=C_INK, va="center")
            ax.add_patch(Rectangle((x0, y - 0.04), self.slice_track_w, 0.08,
                                   fc=C_TRACK, ec="none", zorder=1))
            fill = Rectangle((x0, y - 0.04), 0.0, 0.08, fc=color, ec="none",
                             zorder=2)
            ax.add_patch(fill)
            self.slice_fill.append(fill)
            self.slice_vals.append(ax.text(card.width - CARD_PAD, y, "",
                                           fontsize=8.2, color=C_INK,
                                           ha="right", va="center"))
        self.slice_dot, = ax.plot([CARD_PAD + 0.035], [0.25], marker="o",
                                  ms=4.2, color=C_OK, ls="none")
        self.slice_inside = ax.text(CARD_PAD + 0.14, 0.25, "", fontsize=8,
                                    color=C_MUTED, va="center")

    def _build_task(self):
        card = self.task_card
        ax, h = card.ax, card.height
        b_cols, f_cols, res_col = (2.10, 2.80, 3.50), (4.50, 5.20), card.width - CARD_PAD
        y_head = h - 0.52

        for label, cx in (("field (mT)", 2.62), ("force (mN)", 4.72)):
            ax.text(cx, h - TITLE_Y, label, fontsize=7.5, color=C_FAINT,
                    ha="center", va="center")
        for label, cx in zip(("x", "y", "z", "x", "y", "residual"),
                             b_cols + f_cols + (res_col,)):
            ax.text(cx, y_head, label, fontsize=7.8, color=C_FAINT,
                    ha="right", va="center")
        for sx in (3.80, 5.50):
            ax.plot([sx, sx], [0.40, h - 0.40], color=C_LINE, lw=0.7)

        self.task_rows = []
        for index, y in ((0, h - 0.76), (1, h - 1.00)):
            ax.text(CARD_PAD, y, str(index + 1), fontsize=9,
                    fontweight="medium", color=C_INK, va="center")
            role = ax.text(CARD_PAD + 0.16, y, "", fontsize=8, color=C_MUTED,
                           va="center")
            values = [ax.text(cx, y, "", fontsize=8.6, color=C_INK,
                              ha="right", va="center")
                      for cx in b_cols + f_cols + (res_col,)]
            self.task_rows.append((role, values))
        ax.text(CARD_PAD, 0.22,
                "Bz is the certified witness of the rectangle under the moving "
                "sphere.  The held sphere is commanded zero force.",
                fontsize=7.5, color=C_FAINT, va="center")

    # ------------------------------------------------------------------
    # Commands
    # ------------------------------------------------------------------

    def _on_press(self, event):
        key = (event.key or "").lower()
        if key in DRIVE_KEYS:
            self.held_keys.add(key)
            if self.route is not None:
                self.route = None
                self._notify("Route cancelled for manual drive", "warn")
        elif key == " ":
            self._stop()
        elif key == "h":
            self._return_home()
        elif key == "c":
            self.trails = [[self.positions[0].copy()], [self.positions[1].copy()]]
        elif key == "x":
            self._reset()
        elif key == "escape":
            plt.close(self.fig)

    def _on_release(self, event):
        self.held_keys.discard((event.key or "").lower())

    def _on_click(self, event):
        if event.inaxes is not self.ax or event.button != 1 or event.xdata is None:
            return
        toolbar = getattr(self.fig.canvas, "toolbar", None)
        if toolbar is not None and getattr(toolbar, "mode", ""):
            return  # zoom or pan is active
        self._start_route(np.array([event.xdata, event.ydata]))

    def _notify(self, text, kind):
        self.notice = (text, kind, NOTICE_S)

    def _stop(self):
        self.held_keys.clear()
        self.route = None
        self.force[:] = 0.0
        self.speed = 0.0

    def _reset(self):
        self._reset_state()
        self._notify("Reset to the start of stage 1", "ok")

    def _return_home(self):
        if self.stage == 1:
            self._start_route(MAGNET2_HOME, "handover")
        else:
            self._start_route(MAGNET1_HOME, "returned_m1")

    def _start_route(self, goal, action=None):
        try:
            points = self.active_slice.plan_taut(self.active_position, goal)
        except ValueError as exc:
            self._notify(_sentence(exc), "bad")
            return False
        route = _Route(points, action)
        # Carry on at the present speed so a new command does not jerk.
        self.speed = min(float(np.linalg.norm(self.force)) / DRAG, V_MAX)
        self.route = route
        if route.total < 1e-9:
            self._finish_route()
        return True

    def _finish_route(self):
        route = self.route
        self.positions[self.active_index] = route.goal.copy()
        self.route = None
        self.speed = 0.0
        self.force[:] = 0.0
        if route.action == "handover":
            self.positions[1] = MAGNET2_HOME.copy()
            self.stage = 2
            self.m1_returned = False
            self._notify("Handover complete: magnet 1 now moves", "ok")
        elif route.action == "returned_m1":
            self.positions[0] = MAGNET1_HOME.copy()
            self.m1_returned = True
            self._notify("Magnet 1 is home: sequence complete", "ok")

    # ------------------------------------------------------------------
    # One simulation step
    # ------------------------------------------------------------------

    def _certified_move(self, p, displacement):
        """Move toward ``p + displacement`` without leaving the certified TFW.

        A step that would leave it stops at the edge and spends the rest of
        the displacement sliding.  The certified edge is a staircase of small
        rectangles, so sliding tries a fan of directions around the commanded
        one -- each shortened to its projection -- and keeps whichever makes
        the most progress along the command.
        """
        self.sliding = self.blocked = False
        if np.linalg.norm(displacement) < 1e-12:
            return p
        slice_data = self.active_slice
        t = slice_data.reach(p, p + displacement)
        if t >= 1.0:
            return p + displacement
        edge = p + t * displacement
        rest = (1.0 - t) * displacement
        length = float(np.linalg.norm(rest))
        heading = math.atan2(rest[1], rest[0])
        best, best_gain = edge, 0.0
        for angle in SLIDE_ANGLES:
            c = math.cos(angle)
            step = length * c * np.array([math.cos(heading + angle),
                                          math.sin(heading + angle)])
            u = slice_data.reach(edge, edge + step)
            gain = u * length * c * c
            if gain > best_gain + 1e-12:
                best, best_gain = edge + u * step, gain
        self.sliding = best_gain > 1e-9
        self.blocked = float(np.linalg.norm(best - p)) < 1e-9
        return best

    def _drive_step(self, p, dt):
        direction = np.zeros(2)
        for key in self.held_keys:
            direction += DRIVE_KEYS[key]
        norm = float(np.linalg.norm(direction))
        target = DRIVE_FORCE * direction / norm if norm > 0 else np.zeros(2)
        self.force += (target - self.force) * (1.0 - math.exp(-dt / FORCE_TAU))
        if norm == 0 and np.linalg.norm(self.force) < 1e-7:
            self.force[:] = 0.0
        return self._certified_move(p, self.force / DRAG * dt), self.force.copy(), False

    def _route_step(self, p, dt):
        route = self.route
        self.sliding = self.blocked = False
        self.speed = min(route.speed_limit(route.s), self.speed + ACCEL * dt)
        s_new = min(route.s + self.speed * dt, route.total)
        new_p = route.point(s_new)

        # The route was certified when planned; re-check this step, through
        # any corners it passes, before committing to it.
        corners = [route.points[j] for j in range(1, len(route.points) - 1)
                   if route.s < route.cum[j] <= s_new]
        pieces = [p, *corners, new_p]
        if any(self.active_slice.reach(a, b) < 1.0
               for a, b in zip(pieces[:-1], pieces[1:])):
            self.route = None
            self.speed = 0.0
            self._notify("Stopped: route step left the certified TFW", "bad")
            return p, np.zeros(2), False

        route.s = s_new
        self.force = DRAG * self.speed * route.dirs[route.segment(s_new)]
        arrived = route.total - s_new <= 1e-9
        return new_p, self.force.copy(), arrived

    def step(self, dt=DT):
        index = self.active_index
        p = self.active_position.copy()
        if self.route is not None:
            new_p, force, arrived = self._route_step(p, dt)
        else:
            new_p, force, arrived = self._drive_step(p, dt)
        force = np.clip(force, -F_MOVE, F_MOVE)

        solution = solve_control(self.active_slice, new_p, force)
        if not solution["within_limits"]:
            self._stop()
            self._notify(f"Stopped: allocation exceeded ±{I_MAX:.0f} A", "bad")
            new_p, force, arrived = p, np.zeros(2), False
            solution = solve_control(self.active_slice, p, force)

        if np.linalg.norm(new_p - p) > 1e-12:
            self.positions[index] = new_p
            self.trails[index].append(new_p.copy())
            del self.trails[index][:-TRAIL_LEN]
            if index == 0:
                self.m1_returned = False
        self.applied_force = force
        self.solution = solution
        if arrived:
            self._finish_route()

        if self.notice is not None:
            text, kind, left = self.notice
            self.notice = (text, kind, left - dt) if left > dt else None

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def update(self, _frame):
        now = time.perf_counter()
        dt = DT if self._last_time is None else min(now - self._last_time, MAX_DT)
        self._last_time = now
        self.step(dt)
        self.refresh()
        return ()

    def refresh(self):
        self._refresh_slice()
        self._refresh_scene()
        self._refresh_status()
        self._refresh_sequence()
        self._refresh_coils()
        self._refresh_spheres()
        self._refresh_membership()
        self._refresh_task()

    def _refresh_slice(self):
        if self._shown_stage == self.stage:
            return
        self._shown_stage = self.stage
        rgba, cert, fractions = self.rasters[self.stage]
        self.im_slice.set_data(rgba)
        n = cert.shape[0]
        centers = -XY_LIMIT + (np.arange(n) + 0.5) * (2.0 * XY_LIMIT / n)
        if self.slice_contour is not None:
            self.slice_contour.remove()
        self.slice_contour = self.ax.contour(
            centers, centers, cert.astype(float), levels=[0.5],
            colors=[C_CERT], linewidths=0.9, alpha=0.8, zorder=3,
        )
        for fill, value, fraction in zip(self.slice_fill, self.slice_vals, fractions):
            fill.set_width(fraction * self.slice_track_w)
            value.set_text(f"{fraction * 100:.0f}%")
        self.slice_card.meta.set_text(
            f"stage {self.stage}  ·  {len(self.active_slice.boxes):,} rectangles")
        self.home_button.label.set_text(
            "Return magnet 2 home, hand over" if self.stage == 1
            else "Return magnet 1 home")

    def _refresh_scene(self):
        active, held = self.active_index, self.held_index
        for index, (north, south, outline) in enumerate(self.spheres):
            center = tuple(self.positions[index])
            north.set_center(center)
            south.set_center(center)
            outline.set_center(center)
            self.sphere_labels[index].xy = center
            self.sphere_roles[index].set_text("held" if index == held else "")
        self.hold_ring.set_center(tuple(self.positions[held]))

        p = self.positions[active]
        f = self.applied_force * F_ARROW_SCALE
        self.arr_F.set_positions(tuple(p), tuple(p + f))
        self.arr_F.set_visible(np.linalg.norm(f) > 1.2 * SPHERE_R)

        if self.route is not None:
            route = self.route
            rest = route.points[route.segment(route.s) + 1:]
            line = np.vstack((p, rest))
            self.route_line.set_data(line[:, 0], line[:, 1])
            self.route_goal.set_data([route.goal[0]], [route.goal[1]])
        else:
            self.route_line.set_data([], [])
            self.route_goal.set_data([], [])

        for lc, trail in zip(self.trail_lcs, self.trails):
            if len(trail) < 2:
                lc.set_segments([])
                continue
            pts = np.asarray(trail)
            segs = np.stack([pts[:-1], pts[1:]], axis=1)
            rgba = np.zeros((len(segs), 4))
            rgba[:, :3] = C_TRAIL
            rgba[:, 3] = np.linspace(0.04, 0.70, len(segs))
            lc.set_segments(segs)
            lc.set_color(rgba)

    def _refresh_status(self):
        active, held = self.active_index + 1, self.held_index + 1
        driving = bool(self.held_keys) or np.linalg.norm(self.force) > 1e-7
        action = self.route.action if self.route is not None else None

        if self.blocked:
            kind, chip = "bad", "Blocked"
        elif self.sliding:
            kind, chip = "warn", "Limit"
        elif self.route is not None:
            kind, chip = "busy", "Route"
        elif driving:
            kind, chip = "ok", "Drive"
        else:
            kind, chip = "ok", "Hold"

        if action == "handover":
            message = f"Returning magnet {active} home for handover"
        elif action == "returned_m1":
            message = f"Returning magnet {active} home"
        elif self.route is not None:
            message = f"Moving magnet {active} along a certified route"
        elif driving:
            message = f"Driving magnet {active}"
        else:
            message = f"Magnet {active} at rest, magnet {held} held"

        flags, flag_kind = [], kind
        if self.sliding:
            flags.append("sliding on the edge")
        if self.blocked:
            flags.append("move blocked")
        if self.notice is not None:
            text, flag_kind, _ = self.notice
            flags = [text]
        flag_text = "  ·  ".join(flags)

        patch = self.status_chip.get_bbox_patch()
        patch.set_facecolor(_STATUS_FACE[kind])
        patch.set_edgecolor(_STATUS_EDGE[kind])
        self.status_chip.set_text(chip)
        self.status_chip.set_color(_STATUS_EDGE[kind])
        self.status_msg.set_text(message)
        self.status_flags.set_text(flag_text)
        self.status_flags.set_color(_STATUS_EDGE[flag_kind])

        p = self.active_position
        speed = np.linalg.norm(self.applied_force) / DRAG * 1e3
        separation = np.linalg.norm(self.positions[0] - self.positions[1])
        self.status_vals["x"].set_text(f"{_qty(p[0], 1e3, 1, signed=True)} mm")
        self.status_vals["y"].set_text(f"{_qty(p[1], 1e3, 1, signed=True)} mm")
        self.status_vals["v"].set_text(f"{speed:.1f} mm/s")
        self.status_vals["r"].set_text(f"{separation * 1e3:.1f} mm")

    def _refresh_sequence(self):
        current = self._sequence_step()
        for i, (mark, text) in enumerate(zip(self.seq_marks, self.seq_texts)):
            if i < current:
                mark.set_markerfacecolor(C_FAINT)
                mark.set_markeredgecolor(C_FAINT)
                text.set_color(C_MUTED)
            elif i == current:
                mark.set_markerfacecolor(C_CERT)
                mark.set_markeredgecolor(C_CERT)
                text.set_color(C_INK)
            else:
                mark.set_markerfacecolor(C_PANEL)
                mark.set_markeredgecolor(C_FAINT)
                text.set_color(C_FAINT)
        self.seq_card.meta.set_text("complete" if current >= 4
                                    else f"stage {self.stage} of 2")
        self.seq_card.meta.set_color(C_OK if current >= 4 else C_MUTED)

    def _refresh_coils(self):
        cur = self.solution["currents"]
        for bar, c in zip(self.bars, cur):
            bar.set_height(c)
            bar.set_color(C_BAD if abs(c) > 0.97 * I_MAX else C_BAR)
        self.coil_card.meta.set_text(
            f"peak {np.max(np.abs(cur)):.2f} A   ·   "
            f"norm {np.linalg.norm(cur):.2f} A   ·   10 × 10 solve")

    def _refresh_spheres(self):
        homes = (MAGNET1_HOME, MAGNET2_HOME)
        for index, (role, values) in enumerate(self.sphere_rows):
            p = self.positions[index]
            role.set_text("held" if index == self.held_index else "moving")
            values[0].set_text(_qty(p[0], 1e3, 1, signed=True))
            values[1].set_text(_qty(p[1], 1e3, 1, signed=True))
            values[2].set_text(f"{np.linalg.norm(p - homes[index]) * 1e3:.1f} mm")
        separation = np.linalg.norm(self.positions[0] - self.positions[1])
        self.sphere_sep.set_text(f"{separation * 1e3:.1f} mm")
        self.sphere_card.meta.set_text(
            f"m = {self.slices[1].moments[0]:g} A·m² each")

    def _refresh_membership(self):
        try:
            box, _ = self.active_slice.box_and_witness(self.active_position)
        except ValueError:
            box = None
        if box is None:
            self.slice_dot.set_color(C_BAD)
            self.slice_inside.set_text(
                f"magnet {self.active_index + 1} outside the certified set")
        else:
            x0, x1, y0, y1 = self.active_slice.boxes[box]
            self.slice_dot.set_color(C_OK)
            self.slice_inside.set_text(
                f"magnet {self.active_index + 1} in a "
                f"{(x1 - x0) * 1e3:.2f} × {(y1 - y0) * 1e3:.2f} mm rectangle")

    def _refresh_task(self):
        sol = self.solution
        bz_held, bz_moving = sol["bz"]
        fx, fy = self.applied_force
        task = np.array([
            0.0, B_HOLD, bz_held, 0.0, 0.0,
            0.0, B_HOLD, bz_moving, fx, fy,
        ])
        achieved = task + sol["residual"]
        halves = {self.held_index: slice(0, 5), self.active_index: slice(5, 10)}
        for index, (role, values) in enumerate(self.task_rows):
            part = halves[index]
            out = achieved[part]
            res = float(np.max(np.abs(sol["residual"][part])))
            role.set_text("held" if index == self.held_index else "moving")
            for txt, v in zip(values[:5], out):
                txt.set_text(_qty(v, 1e3, signed=True))
            values[5].set_text(_sci(res))

    # ------------------------------------------------------------------

    def run(self):
        self.anim = FuncAnimation(self.fig, self.update,
                                  interval=int(DT * 1000),
                                  blit=False, cache_frame_data=False)
        plt.show()


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage1", default=str(STAGE1_JSON),
                        help="certified TFW JSON for stage 1")
    parser.add_argument("--stage2", default=str(STAGE2_JSON),
                        help="certified TFW JSON for stage 2")
    parser.add_argument("--screenshot", default=None,
                        help="save a still to this path and exit (uses the Agg backend)")
    args = parser.parse_args(argv)

    sim = Case8Sim(args.stage1, args.stage2)
    if args.screenshot:
        sim.update(0)
        sim.fig.savefig(args.screenshot, dpi=140,
                        facecolor=sim.fig.get_facecolor(), bbox_inches="tight")
        print(f"saved {args.screenshot}")
        return 0
    print("Click the plot window once so it has keyboard focus, "
          "then drive with W/A/S/D or click a green point.")
    sim.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
