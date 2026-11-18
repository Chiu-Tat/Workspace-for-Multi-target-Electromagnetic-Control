#!/usr/bin/env python3
# Run from the repository root: python -m cases.case_6_interaction_obstacle_avoidance.simulation.interactive
"""
interactive.py -- Interactive Case 6 driving simulator.

Drive the free magnet around the plane Z = 0.082 m with W/S/A/D while the
certified feasible workspace for its *current heading* is drawn underneath it,
so you can see at a glance where the magnet is allowed to go.

Everything is driven from coil currents.  Each frame:

  1. W/S set the commanded thrust  fz = +-0.1 N  along the magnet's heading.
  2. The body-frame task  [By, Bz, fy, fz] = [0, 0.02 T, 0, fz]  is allocated
     to the ten coils, subject to |I| <= 15 A.
  3. The field and force actually produced by that current vector are evaluated
     at the magnet (coils + the fixed magnet's dipole).
  4. The magnet moves with a drag-dominated law  v = F_inplane / c.

The magnet is blocked at the boundary of the certified workspace: a step that
would leave the feasible set is rejected, and the magnet slides along the
boundary instead.  Turning is blocked the same way when the new heading would
make the present position infeasible -- the ring drawn around the magnet shows
which headings are still admissible from where it stands.

Two feasible sets are available, and the difference between them is the point:

  certified  the sliced affine-hybrid 3-D TFW from
             cases.case_6_interaction_obstacle_avoidance.workspace_hybrid.
             Guaranteed and conservative: it is an inner approximation, so
             everything inside it is provably controllable, but controllable
             poses can remain uncertified.
  exact      a pointwise test at the true (x, y, beta), evaluated live.  It can
             be larger than the certified set, but it carries no box-wide
             robustness guarantee.

Controls
--------
  W / S    thrust forward / backward along the heading
  A / D    turn left / right
  SPACE    all stop (zero thrust, zero turn)
  T        switch the blocking rule: certified <-> exact
  R        draw the exact feasible region at the current heading (grid sweep,
           a few seconds; only for display -- T does not need it)
  H        show / hide that overlay
  C        clear the trail
  X        reset the magnet to the starting pose
  ESC      quit

Run:
    python -m cases.case_6_interaction_obstacle_avoidance.simulation.interactive
    python -m cases.case_6_interaction_obstacle_avoidance.simulation.interactive --grid 81
"""

import argparse
import math
import sys
import time

import numpy as np
import matplotlib
if any(a == '--screenshot' or a.startswith('--screenshot=') for a in sys.argv):
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager, transforms
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import (
    ArrowStyle, FancyArrowPatch, Patch, Polygon, Rectangle,
)

from .model import (
    B_HOLD, FIXED_BETA, FIXED_M, FIXED_X, FIXED_Y, FREE_M, F_MAX, I_MAX,
    I_MIN, NUM_COILS, XY_MAX, XY_MIN, Z_PLANE,
    TFWSlice, actuation_matrix_body, compute_region, feasible_point,
    field_and_force, solve_currents, task_vector,
)

# ============================================================================
# Simulation tuning
# ============================================================================

FPS = 20.0
DT = 1.0 / FPS                      # s, integration step

DRAG_C = 5.0                        # N s / m -- drag-dominated law v = F / c
                                    # gives 20 mm/s at the full 0.1 N of thrust
TURN_RATE = math.radians(60.0)      # rad/s of heading change under A / D

TRAIL_LEN = 600                     # frames of position history to draw

# Starting pose of the free magnet.  X resets to it.
START_XY = (0.0, -0.05)             # m
START_BETA = math.pi / 2            # rad

# Drawing scales
# B and F in the plane are always parallel to the heading -- By = 0 and fy = 0
# are enforced.  F, which moves the magnet, is drawn to scale on its axis,
# beneath the body, so it leaves from the face it points out of.  |B| is held
# constant, so only its direction is drawn: a short arrow on the body itself.
F_ARROW_SCALE = 0.20                # m per newton
MAG_LEN, MAG_WID = 0.009, 0.005     # m, drawn size of a magnet
COMPASS_R = 0.0086                  # m, radius of the heading-freedom ring

# Residuals below this fraction of the task scale are numerical noise, not
# tracking error.  1e-6 is the solver feasibility gate; display is stricter.
REL_OK = 1e-6
REL_NOISE = 1e-9

# ============================================================================
# Layout -- all in inches, converted to figure fractions when placed
# ============================================================================

FIG_W, FIG_H = 15.8, 9.15
MARGIN = 0.45                       # outer side margin
CARD_PAD = 0.18                     # horizontal padding inside a card
CARD_GAP = 0.15                     # space between cards
TITLE_Y = 0.25                      # card top edge to the title's centre line

KEY_HINTS = (
    (('W', 'S'), 'thrust'),
    (('A', 'D'), 'turn'),
    (('Space',), 'stop'),
    (('T',), 'blocking rule'),
    (('R',), 'exact region'),
    (('H',), 'overlay'),
    (('C',), 'clear trail'),
    (('X',), 'reset'),
    (('Esc',), 'quit'),
)

# ============================================================================
# Palette
# ============================================================================

C_BG = '#f4f1ea'
C_PANEL = '#fcfbf7'
C_SCENE = '#faf8f3'
C_INK = '#23201b'
C_MUTED = '#6f685e'
C_FAINT = '#a29a8e'
C_LINE = '#ddd6ca'
C_GRID = '#ece6db'
C_TRACK = '#eee9df'

C_CERT = '#3d8a58'
C_CERT_RGB = (0.55, 0.73, 0.58)
C_EXACT = '#2a5a9a'
C_EXACT_RGB = (0.16, 0.35, 0.60)
C_SOUTH = '#3a5478'
C_NORTH = '#c45c4a'
C_F = '#c36b24'
C_TRAIL = (0.40, 0.36, 0.32)
C_BAR = '#4f6b8c'
C_OK = '#2c7a4b'
C_WARN = '#9a5b12'
C_BAD = '#a4332c'
C_BLOCK = '#e6cbc4'

_STATUS_FACE = {
    'ok': '#e8f1ea',
    'warn': '#f5ecdc',
    'bad': '#f5e4e0',
    'busy': '#e7edf5',
}
_STATUS_EDGE = {
    'ok': C_OK,
    'warn': C_WARN,
    'bad': C_BAD,
    'busy': C_EXACT,
}


def _pick_font():
    wanted = ('Avenir Next', 'Avenir', 'Helvetica Neue', 'SF Pro Text',
              'Arial', 'DejaVu Sans')
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in wanted:
        if name in available:
            return name
    return 'DejaVu Sans'


FONT = _pick_font()


def _apply_style():
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': [FONT, 'DejaVu Sans'],
        # Set subscripted labels such as B_y in the body font, not Computer Modern.
        'mathtext.fontset': 'custom',
        'mathtext.rm': FONT,
        'mathtext.it': f'{FONT}:italic',
        'mathtext.bf': f'{FONT}:bold',
        'axes.unicode_minus': True,
        'axes.linewidth': 0.6,
        'axes.edgecolor': C_LINE,
        'axes.labelcolor': C_INK,
        'text.color': C_INK,
        'xtick.color': C_MUTED,
        'ytick.color': C_MUTED,
        'figure.facecolor': C_BG,
        'savefig.facecolor': C_BG,
    })
    for k in list(matplotlib.rcParams):
        if k.startswith('keymap.'):
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
                    fontweight='medium', color=C_INK, va='center')
        self.meta = ax.text(width - CARD_PAD, height - TITLE_Y, '',
                            fontsize=8, color=C_MUTED, ha='right', va='center')

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
    s = f'{v:+.{decimals}f}' if signed else f'{v:.{decimals}f}'
    return s.replace('-', '−')


def _fmt_residual(value, scale, base):
    """Tracking residual in µ- or m-units of ``base``; noise shows as zero."""
    if scale and abs(value) / scale < REL_NOISE:
        value = 0.0
    if abs(value) >= 1e-4:
        return f'Δ {_qty(value, 1e3, signed=True)} m{base}'
    return f'Δ {_qty(value, 1e6, signed=True)} µ{base}'


def _rel_error_label(rel_error):
    if rel_error < REL_OK:
        return 'matched', 'ok'
    if rel_error < 1e-3:
        return f'{rel_error:.1e} of full scale', 'warn'
    return 'task not met', 'bad'


def _heading_arcs(ok, step, r, cx=0.0, cy=0.0):
    """Arcs of a heading ring, one per run of equal admissibility.

    Merging runs keeps the ring free of hairline seams between the
    individual beta levels.  Returns (segments, admissible flags).
    """
    ok = np.asarray(ok, dtype=bool)
    n = len(ok)
    starts = np.nonzero(ok != np.roll(ok, 1))[0]
    if starts.size == 0:
        starts = np.array([0])
    segs, flags = [], []
    for i, s in enumerate(starts):
        e = starts[i + 1] if i + 1 < len(starts) else starts[0] + n
        a = np.linspace(s * step, e * step, max(4, 2 * (e - s) + 1))
        segs.append(np.column_stack([cx + r * np.cos(a), cy + r * np.sin(a)]))
        flags.append(bool(ok[s]))
    return segs, flags


def _magnet_polygons(x, y, beta, length=MAG_LEN, width=MAG_WID):
    """Two half-rectangles for a dipole at (x, y) pointing along ``beta``.

    Returns (south_half, north_half) vertex arrays; the north half is the end
    the magnetic moment points toward.
    """
    u = np.array([math.cos(beta), math.sin(beta)])
    v = np.array([-u[1], u[0]])
    c = np.array([x, y])
    hl, hw = length / 2.0, width / 2.0
    south = np.array([c - hl * u - hw * v, c - hl * u + hw * v,
                      c + hw * v, c - hw * v])
    north = np.array([c - hw * v, c + hw * v,
                      c + hl * u + hw * v, c + hl * u - hw * v])
    return south, north


class _MagnetKey:
    """Legend key for the field arrow: a small two-tone magnet with the arrow
    drawn on it.  Serves as both the legend handle and its handler."""

    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        w, h = handlebox.width, handlebox.height
        x0, y0 = -handlebox.xdescent, -handlebox.ydescent
        bh = 1.4 * h
        yb = y0 + (h - bh) / 2.0
        parts = [
            Rectangle((x0, yb), w / 2.0, bh, fc=C_SOUTH, ec='none'),
            Rectangle((x0 + w / 2.0, yb), w / 2.0, bh, fc=C_NORTH, ec='none'),
            Rectangle((x0, yb), w, bh, fill=False, ec=C_INK, lw=0.6),
            FancyArrowPatch(
                (x0 + 0.20 * w, y0 + h / 2.0), (x0 + 0.84 * w, y0 + h / 2.0),
                arrowstyle=ArrowStyle('-|>', head_length=0.6, head_width=0.3),
                mutation_scale=6, lw=0.9, color=C_PANEL, shrinkA=0, shrinkB=0,
            ),
        ]
        for p in parts:
            p.set_transform(handlebox.get_transform())
            handlebox.add_artist(p)
        return parts[1]


def _arrow(color, lw=1.5, size=10, zorder=10):
    return FancyArrowPatch(
        (0, 0), (0, 0),
        arrowstyle=ArrowStyle('-|>', head_length=0.6, head_width=0.3),
        mutation_scale=size, lw=lw, color=color, zorder=zorder,
        shrinkA=0, shrinkB=0, joinstyle='miter', capstyle='butt',
    )


class Case6Sim:
    def __init__(self, grid=61, tfw_path=None, raster=400, start=START_XY):
        self.grid = grid
        self.tfw = TFWSlice(tfw_path) if tfw_path else TFWSlice(raster=raster)

        self.beta0 = START_BETA
        self.pos0 = np.array(start, dtype=float)
        if not self.tfw.contains(self.pos0[0], self.pos0[1],
                                 self.tfw.slice_at(self.beta0)):
            seed = self.tfw.find_seed(self.beta0)
            if seed is None:
                raise RuntimeError(
                    'The certified workspace is empty at the start heading.')
            print(f'warning: start ({self.pos0[0]:+.4f}, {self.pos0[1]:+.4f}) m is '
                  f'not in the certified workspace at '
                  f'beta = {math.degrees(self.beta0):.1f} deg;\n'
                  f'         falling back to ({seed[0]:+.4f}, {seed[1]:+.4f}) m.')
            self.pos0 = np.array(seed, dtype=float)

        self.pos = self.pos0.copy()
        self.beta = self.beta0
        self.tfw.set_beta(self.beta)

        self.held = set()
        self.thrust_cmd = 0.0
        self.blocked_move = False
        self.blocked_turn = False
        self.sliding = False
        self.trail = [self.pos.copy()]

        self.exact_mask = None
        self.exact_extent = None
        self.exact_beta = None
        self.exact_elapsed = None
        self.show_exact = True
        self.block_on = 'certified'

        self._last_slice = None
        self._build_figure()

    # ------------------------------------------------------------------
    # Figure construction
    # ------------------------------------------------------------------

    def _build_figure(self):
        _apply_style()

        fig = self.fig = plt.figure(figsize=(FIG_W, FIG_H))
        if fig.canvas.manager is not None:
            fig.canvas.manager.set_window_title('eMNS · Case 6 · interaction')

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
        heights = np.array([0.80, 1.66, 1.56, 1.63, 1.18])
        bottoms = plot_top - np.cumsum(heights) - CARD_GAP * np.arange(len(heights))

        self.status_card = _Card(fig, x, bottoms[0], w, heights[0])
        self.task_card = _Card(fig, x, bottoms[1], w, heights[1], 'Body-frame task')
        self.coil_card = _Card(fig, x, bottoms[2], w, heights[2], 'Coil currents')
        self.head_card = _Card(fig, x, bottoms[3], half, heights[3], 'Headings here')
        self.ws_card = _Card(fig, x + half + CARD_GAP, bottoms[3], half,
                             heights[3], 'Workspace slice')
        self.ff_card = _Card(fig, x, bottoms[4], w, heights[4], 'At the magnet')

        self._build_status()
        self._build_task()
        self._build_coils()
        self._build_headings()
        self._build_workspace()
        self._build_field_force()

        fig.canvas.mpl_connect('key_press_event', self._on_press)
        fig.canvas.mpl_connect('key_release_event', self._on_release)

    def _build_header(self, ax):
        y = FIG_H - 0.32
        brand = ax.text(MARGIN, y, 'eMNS', fontsize=13, fontweight='bold',
                        color=C_INK, va='center')
        ax.annotate('Case 6  ·  Interaction and obstacle avoidance',
                    xy=(1, 0.5), xycoords=brand, xytext=(12, 0),
                    textcoords='offset points', va='center', fontsize=12,
                    color=C_MUTED)
        ax.text(FIG_W - MARGIN, y,
                f'plane z = {Z_PLANE * 1e3:.0f} mm   ·   '
                f'{NUM_COILS} coils, |I| ≤ {I_MAX:.0f} A   ·   '
                f'm = {FREE_M:g} / {FIXED_M:g} A·m²',
                ha='right', va='center', fontsize=8.5, color=C_FAINT)

    def _build_footer(self, ax):
        """Key hints as quiet key caps, each placed after the one before."""
        cap = dict(boxstyle='round,pad=0.35,rounding_size=0.3',
                   fc=C_PANEL, ec=C_LINE, lw=0.7)

        def chain(prev, s, gap, **kw):
            if prev is None:
                return ax.text(MARGIN + 0.04, 0.26, s, va='center', **kw)
            return ax.annotate(s, xy=(1, 0.5), xycoords=prev,
                               xytext=(gap, 0), textcoords='offset points',
                               va='center', **kw)

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
        ax.set_xlim(XY_MIN - pad, XY_MAX + pad)
        ax.set_ylim(XY_MIN - pad, XY_MAX + pad)
        ax.set_aspect('equal', anchor='W')
        ticks = np.linspace(XY_MIN, XY_MAX, 5)
        labels = [_qty(t, 1e3, decimals=0) for t in ticks]
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        ax.set_xlabel('x (mm)', fontsize=9, color=C_MUTED, labelpad=6)
        ax.set_ylabel('y (mm)', fontsize=9, color=C_MUTED, labelpad=6)
        ax.tick_params(labelsize=8.5, labelcolor=C_MUTED, length=0, pad=6)
        ax.grid(True, color=C_GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for sp in ax.spines.values():
            sp.set_color(C_LINE)
            sp.set_linewidth(0.8)

        self.im_cert = ax.imshow(
            np.zeros((2, 2, 4)), origin='lower', extent=self.tfw.extent,
            interpolation='nearest', zorder=1,
        )
        self.im_exact = ax.imshow(
            np.zeros((2, 2, 4)), origin='lower', extent=self.tfw.extent,
            interpolation='nearest', zorder=2,
        )
        self.im_exact.set_visible(False)
        self.exact_contour = None
        self.cert_contour = None

        ax.add_patch(Rectangle(
            (XY_MIN, XY_MIN), XY_MAX - XY_MIN, XY_MAX - XY_MIN, fill=False,
            ec='#cfc7ba', lw=0.7, zorder=4,
        ))

        s, n = _magnet_polygons(FIXED_X, FIXED_Y, FIXED_BETA, 0.011, 0.0055)
        ax.add_patch(Polygon(s, closed=True, facecolor=C_SOUTH,
                             edgecolor=C_INK, linewidth=0.6, zorder=8))
        ax.add_patch(Polygon(n, closed=True, facecolor=C_NORTH,
                             edgecolor=C_INK, linewidth=0.6, zorder=8))
        ax.annotate('fixed', (FIXED_X, FIXED_Y), xytext=(18, 0),
                    textcoords='offset points', va='center', fontsize=7.5,
                    color=C_MUTED, zorder=13)

        self.trail_lc = LineCollection(
            [], linewidths=1.3, zorder=5, capstyle='round', joinstyle='round',
        )
        ax.add_collection(self.trail_lc)

        self.pg_south = Polygon(
            np.zeros((4, 2)), closed=True, facecolor=C_SOUTH,
            edgecolor=C_INK, linewidth=0.7, zorder=11,
        )
        self.pg_north = Polygon(
            np.zeros((4, 2)), closed=True, facecolor=C_NORTH,
            edgecolor=C_INK, linewidth=0.7, zorder=11,
        )
        ax.add_patch(self.pg_south)
        ax.add_patch(self.pg_north)

        self.compass = LineCollection([], linewidths=2.2, zorder=9,
                                      capstyle='butt')
        ax.add_collection(self.compass)

        self.arr_F = _arrow(C_F)
        self.arr_B = _arrow(C_PANEL, lw=1.1, size=7, zorder=12)
        ax.add_patch(self.arr_F)
        ax.add_patch(self.arr_B)

        magnet_key = _MagnetKey()
        handles = [
            Patch(facecolor=C_CERT_RGB, alpha=0.55, edgecolor=C_CERT, lw=0.8),
            Line2D([], [], color=C_EXACT, lw=1.4),
            magnet_key,
            Line2D([], [], color=C_F, lw=1.6),
        ]
        labels = ['certified workspace', 'exact region (R)',
                  r'$B$ direction', r'$F$ thrust']
        leg = ax.legend(
            handles, labels, handler_map={_MagnetKey: magnet_key},
            loc='lower left', bbox_to_anchor=(0.0, 1.0),
            ncol=4, fontsize=8, frameon=False, borderaxespad=0.5,
            borderpad=0.0, handlelength=1.6, handleheight=0.6,
            handletextpad=0.6, columnspacing=1.8,
        )
        for txt in leg.get_texts():
            txt.set_color(C_MUTED)

        self.plot_caption = ax.annotate(
            '', xy=(1, 1), xycoords='axes fraction', xytext=(0, 6),
            textcoords='offset points', ha='right', va='bottom',
            fontsize=8, color=C_FAINT,
        )

    def _build_status(self):
        card = self.status_card
        ax, h = card.ax, card.height
        self.status_chip = ax.text(
            CARD_PAD + 0.36, h / 2, '', ha='center', va='center',
            fontsize=8.8, fontweight='medium', color=C_OK, zorder=3,
            bbox=dict(boxstyle='round,pad=0.55,rounding_size=0.55',
                      fc=_STATUS_FACE['ok'], ec=_STATUS_EDGE['ok'], lw=0.8),
        )
        x_msg = CARD_PAD + 0.86
        self.status_msg = ax.text(x_msg, h * 0.66, '', fontsize=9.5,
                                  color=C_INK, va='center')
        self.status_flags = ax.text(card.width - CARD_PAD, h * 0.66, '',
                                    fontsize=8, ha='right', va='center')
        self.status_vals = {}
        for key, label, dx in (('x', 'x', 0.0), ('y', 'y', 0.86),
                               ('beta', 'β', 1.72), ('v', 'v', 2.52)):
            ax.text(x_msg + dx, h * 0.31, label, fontsize=8, color=C_FAINT,
                    va='center')
            self.status_vals[key] = ax.text(x_msg + dx + 0.15, h * 0.31, '',
                                            fontsize=8.2, color=C_INK,
                                            va='center')

    def _build_task(self):
        card = self.task_card
        right = 1.85
        ax = card.inset(left=0.62, bottom=0.30, right=right, top=0.48)
        rows = np.arange(4)
        ax.set_xlim(-1.08, 1.08)
        ax.set_ylim(3.55, -0.55)
        ax.set_yticks(rows)
        ax.set_yticklabels([r'$B_y$', r'$B_z$', r'$f_y$', r'$f_z$'])
        ax.tick_params(axis='y', labelsize=10.5, labelcolor=C_INK, pad=12)
        ax.set_xticks([-1, 0, 1])
        ax.set_xticklabels(['−full', '0', '+full'])

        ax.barh(rows, 2.0, left=-1.0, height=0.40, color=C_TRACK, zorder=1)
        ax.axvline(0.0, color=C_INK, lw=0.6, alpha=0.22, zorder=3)
        self.task_bars = ax.barh(rows, np.zeros(4), height=0.40, color=C_BAR,
                                 zorder=2)
        self.task_cmd, = ax.plot(
            np.zeros(4), rows, linestyle='none', marker='|', markersize=12,
            markeredgewidth=1.6, color=C_INK, zorder=4, clip_on=False,
        )

        # Values sit to the right of the bars, in the card's inch coordinates.
        tr = transforms.blended_transform_factory(card.ax.transData, ax.transData)
        x_val = card.width - right + 0.80
        self.task_vals = [
            card.ax.text(x_val, r, '', transform=tr, ha='right', va='center',
                         fontsize=8.4, color=C_INK)
            for r in rows
        ]
        self.task_res = [
            card.ax.text(x_val + 0.10, r, '', transform=tr, ha='left',
                         va='center', fontsize=7.8, color=C_BAD)
            for r in rows
        ]

    def _build_coils(self):
        ax = self.coil_card.inset(left=0.66, bottom=0.30, right=CARD_PAD, top=0.50)
        idx = np.arange(1, NUM_COILS + 1)
        ax.set_xlim(0.4, NUM_COILS + 0.6)
        ax.set_ylim(I_MIN * 1.22, I_MAX * 1.22)
        ax.set_xticks(idx)
        ax.set_yticks([I_MIN, 0.0, I_MAX])
        ax.set_yticklabels([f'{_qty(I_MIN, decimals=0)} A', '0',
                            f'+{I_MAX:.0f} A'])
        for lim in (I_MIN, I_MAX):
            ax.axhline(lim, color=C_BAD, lw=0.8, ls=(0, (3.0, 2.5)),
                       alpha=0.5, zorder=1)
        ax.axhline(0.0, color=C_INK, lw=0.6, alpha=0.25, zorder=3)
        self.bars = ax.bar(idx, np.zeros(NUM_COILS), width=0.56, color=C_BAR,
                           edgecolor='none', zorder=2)

    def _build_headings(self):
        card = self.head_card
        size = card.height - 0.48 - 0.14
        ax = card.inset(left=CARD_PAD, bottom=0.14,
                        right=card.width - CARD_PAD - size, top=0.48)
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        for a in np.arange(4) * math.pi / 2:
            ax.plot([0.70 * math.cos(a), 0.80 * math.cos(a)],
                    [0.70 * math.sin(a), 0.80 * math.sin(a)],
                    color=C_FAINT, lw=0.8, zorder=1)
        self.head_ring = LineCollection([], linewidths=6.0, capstyle='butt',
                                        zorder=2)
        ax.add_collection(self.head_ring)
        self.head_needle, = ax.plot([0, 0], [0, 0], color=C_INK, lw=1.4,
                                    solid_capstyle='round', zorder=3)
        ax.plot([0], [0], marker='o', ms=3.2, color=C_INK, zorder=4)

        tx = CARD_PAD + size + 0.30
        self.head_big = card.ax.text(tx, 0.88, '', fontsize=18, color=C_INK,
                                     va='center')
        card.ax.text(tx, 0.56, 'of 360° admissible', fontsize=8,
                     color=C_MUTED, va='center')
        self.head_levels = card.ax.text(tx, 0.35, '', fontsize=7.8,
                                        color=C_FAINT, va='center')

    def _build_workspace(self):
        card = self.ws_card
        ax = card.ax
        self.ws_x0 = 0.86
        self.ws_track_w = card.width - 0.62 - self.ws_x0
        self.ws_track, self.ws_fill, self.ws_vals = [], [], []
        for label, y, color in (('certified', 0.96, C_CERT),
                                ('exact', 0.66, C_EXACT)):
            ax.text(CARD_PAD, y, label, fontsize=8.2, color=C_INK, va='center')
            track = Rectangle((self.ws_x0, y - 0.04), self.ws_track_w, 0.08,
                              fc=C_TRACK, ec='none', zorder=1)
            fill = Rectangle((self.ws_x0, y - 0.04), 0.0, 0.08,
                             fc=color, ec='none', zorder=2)
            ax.add_patch(track)
            ax.add_patch(fill)
            self.ws_track.append(track)
            self.ws_fill.append(fill)
            self.ws_vals.append(ax.text(card.width - CARD_PAD, y, '',
                                        fontsize=8.2, color=C_INK,
                                        ha='right', va='center'))
        self.ws_hint = ax.text(self.ws_x0, 0.66, '', fontsize=8,
                               color=C_FAINT, va='center')
        self.ws_dot, = ax.plot([CARD_PAD + 0.035], [0.30], marker='o', ms=4.2,
                               color=C_OK, linestyle='none')
        self.ws_inside = ax.text(CARD_PAD + 0.14, 0.30, '', fontsize=8,
                                 color=C_MUTED, va='center')

    def _build_field_force(self):
        card = self.ff_card
        ax, h = card.ax, card.height
        cols = (1.85, 2.75, 3.65, 5.05, 6.05)
        y_head, y_b, y_f = h - 0.52, h - 0.76, h - 1.00

        for label, cx in (('world frame', 2.55), ('heading frame', 5.35)):
            ax.text(cx, h - TITLE_Y, label, fontsize=7.5, color=C_FAINT,
                    ha='center', va='center')
        for label, cx in zip(('x', 'y', 'z', 'along', 'lateral'), cols):
            ax.text(cx, y_head, label, fontsize=7.8, color=C_FAINT,
                    ha='right', va='center')
        ax.plot([4.15, 4.15], [0.08, h - 0.40], color=C_LINE, lw=0.7)

        self.ff_vals = {}
        for key, sym, unit, color, y in (('B', 'B', 'mT', C_INK, y_b),
                                         ('F', 'F', 'mN', C_F, y_f)):
            ax.text(CARD_PAD, y, sym, fontsize=9, fontweight='bold',
                    color=color, va='center')
            ax.text(CARD_PAD + 0.20, y, unit, fontsize=8, color=C_MUTED,
                    va='center')
            self.ff_vals[key] = [
                ax.text(cx, y, '', fontsize=8.6, color=C_INK, ha='right',
                        va='center')
                for cx in cols
            ]

    # ------------------------------------------------------------------
    # Keyboard
    # ------------------------------------------------------------------

    def _on_press(self, event):
        k = (event.key or '').lower()
        if k in ('w', 's', 'a', 'd'):
            self.held.add(k)
        elif k == ' ':
            self.held.clear()
        elif k == 'r':
            self._recompute_exact()
        elif k == 't':
            self.block_on = 'exact' if self.block_on == 'certified' else 'certified'
        elif k == 'h':
            self.show_exact = not self.show_exact
            vis = self.show_exact and self.exact_mask is not None
            self.im_exact.set_visible(vis)
            if self.exact_contour is not None:
                self.exact_contour.set_visible(vis)
        elif k == 'c':
            self.trail = [self.pos.copy()]
        elif k == 'x':
            self.pos = self.pos0.copy()
            self.beta = self.beta0
            self.trail = [self.pos.copy()]
            self.held.clear()
        elif k == 'escape':
            plt.close(self.fig)

    def _on_release(self, event):
        self.held.discard((event.key or '').lower())

    # ------------------------------------------------------------------
    # Feasibility, with the chosen region doing the blocking
    # ------------------------------------------------------------------

    def _exact_contains(self, x, y):
        if self.exact_mask is None:
            return False
        x0, x1, y0, y1 = self.exact_extent
        if not (x0 <= x <= x1 and y0 <= y <= y1):
            return False
        ny, nx = self.exact_mask.shape
        i = min(max(int((x - x0) / (x1 - x0) * nx), 0), nx - 1)
        j = min(max(int((y - y0) / (y1 - y0) * ny), 0), ny - 1)
        return bool(self.exact_mask[j, i])

    def _allowed(self, x, y, beta):
        """Does the blocking region admit the pose (x, y, beta)?"""
        if not (XY_MIN <= x <= XY_MAX and XY_MIN <= y <= XY_MAX):
            return False
        if self.block_on == 'exact':
            return feasible_point(x, y, beta)
        return self.tfw.contains(x, y, self.tfw.slice_at(beta))

    def _recompute_exact(self):
        self._set_status(
            'busy', 'Busy', 'Computing the exact feasible region',
            f'β = {math.degrees(self.beta) % 360:.1f}°,  '
            f'{self.grid} × {self.grid} grid',
        )
        self.fig.canvas.draw_idle()
        self.fig.canvas.start_event_loop(0.001)

        t0 = time.time()
        mask, extent = compute_region(self.beta, n=self.grid)
        self.exact_mask, self.exact_extent, self.exact_beta = mask, extent, self.beta
        self.exact_elapsed = time.time() - t0
        print(f'exact region at beta = {math.degrees(self.beta) % 360:.1f} deg: '
              f'{mask.mean() * 100:.0f}% feasible  ({self.exact_elapsed:.1f} s)')

        rgba = np.zeros(mask.shape + (4,))
        rgba[..., 0], rgba[..., 1], rgba[..., 2] = C_EXACT_RGB
        rgba[..., 3] = np.where(mask, 0.07, 0.0)
        self.im_exact.set_data(rgba)
        self.im_exact.set_extent(extent)
        self.im_exact.set_visible(self.show_exact)

        if self.exact_contour is not None:
            self.exact_contour.remove()
        xs = np.linspace(extent[0], extent[1], mask.shape[1])
        ys = np.linspace(extent[2], extent[3], mask.shape[0])
        self.exact_contour = self.ax.contour(
            xs, ys, mask.astype(float), levels=[0.5],
            colors=[C_EXACT], linewidths=1.3, zorder=6)
        self.exact_contour.set_visible(self.show_exact)

    # ------------------------------------------------------------------
    # One simulation step
    # ------------------------------------------------------------------

    def step(self):
        turn = (TURN_RATE if 'a' in self.held else 0.0) - \
               (TURN_RATE if 'd' in self.held else 0.0)
        self.thrust_cmd = (F_MAX if 'w' in self.held else 0.0) - \
                          (F_MAX if 's' in self.held else 0.0)

        self.blocked_turn = False
        if turn != 0.0:
            beta_new = (self.beta + turn * DT) % (2 * math.pi)
            if self._allowed(self.pos[0], self.pos[1], beta_new):
                self.beta = beta_new
            else:
                self.blocked_turn = True

        A_hat = actuation_matrix_body(self.pos[0], self.pos[1], self.beta)
        b_cmd = task_vector(self.thrust_cmd)
        sol = solve_currents(A_hat, b_cmd)
        currents = sol['currents']

        B_world, F_world = field_and_force(self.pos[0], self.pos[1],
                                           self.beta, currents)

        v = F_world[:2] / DRAG_C
        target = self.pos + v * DT
        self.blocked_move = False
        self.sliding = False
        moving = np.linalg.norm(v) * DT > 1e-9
        if self._allowed(target[0], target[1], self.beta):
            self.pos = target
        elif moving:
            for cand in (np.array([target[0], self.pos[1]]),
                         np.array([self.pos[0], target[1]])):
                if self._allowed(cand[0], cand[1], self.beta):
                    self.pos = cand
                    self.sliding = True
                    break
            self.blocked_move = not self.sliding

        if np.linalg.norm(v) > 1e-9:
            self.trail.append(self.pos.copy())
            if len(self.trail) > TRAIL_LEN:
                self.trail.pop(0)

        return sol, B_world, F_world

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _refresh_certified(self):
        mask = self.tfw.slice_at(self.beta)
        if mask is self._last_slice:
            return
        self._last_slice = mask
        self.tfw.set_beta(self.beta)
        rgba = np.zeros(mask.shape + (4,))
        rgba[..., 0], rgba[..., 1], rgba[..., 2] = C_CERT_RGB
        rgba[..., 3] = np.where(mask, 0.28, 0.0)
        self.im_cert.set_data(rgba)
        self.im_cert.set_extent(self.tfw.extent)

        n = mask.shape[0]
        xs = np.linspace(self.tfw.x_min, self.tfw.x_max, n)
        ys = np.linspace(self.tfw.y_min, self.tfw.y_max, n)
        if self.cert_contour is not None:
            self.cert_contour.remove()
        self.cert_contour = self.ax.contour(
            xs, ys, mask.astype(float), levels=[0.5],
            colors=[C_CERT], linewidths=0.9, alpha=0.80, zorder=3,
        )

    def update(self, _frame):
        sol, B_world, F_world = self.step()
        x, y = self.pos
        self._refresh_certified()
        ok = self.tfw.headings_at(x, y)
        inside = self.tfw.contains(x, y, self.tfw.slice_at(self.beta))

        self._update_magnet(x, y, B_world, F_world, ok)
        self._update_trail()
        self._update_caption()
        self._update_coils(sol)
        self._update_task(sol)
        self._update_headings(ok)
        self._update_workspace(inside)
        self._update_field_force(B_world, F_world)
        self._update_status(sol, F_world, inside)
        return ()

    def _update_magnet(self, x, y, B_world, F_world, ok):
        s, n = _magnet_polygons(x, y, self.beta)
        self.pg_south.set_xy(s)
        self.pg_north.set_xy(n)

        segs, flags = _heading_arcs(ok, self.tfw.beta_step, COMPASS_R, x, y)
        self.compass.set_segments(segs)
        self.compass.set_color([C_CERT if f else C_BLOCK for f in flags])

        # A thrust arrow shorter than half the body would be hidden under it,
        # and a zero-length one still draws a stray head, so hide both cases.
        f = F_world[:2] * F_ARROW_SCALE
        self.arr_F.set_positions((x, y), (x + f[0], y + f[1]))
        self.arr_F.set_visible(np.linalg.norm(f) > MAG_LEN / 2)

        b = B_world[:2]
        b_norm = np.linalg.norm(b)
        if b_norm > 0.05 * B_HOLD:
            u = b / b_norm
            self.arr_B.set_positions(tuple(self.pos - 0.30 * MAG_LEN * u),
                                     tuple(self.pos + 0.36 * MAG_LEN * u))
        self.arr_B.set_visible(b_norm > 0.05 * B_HOLD)

    def _update_trail(self):
        if len(self.trail) < 2:
            self.trail_lc.set_segments([])
            return
        pts = np.asarray(self.trail)
        segs = np.stack([pts[:-1], pts[1:]], axis=1)
        n = len(segs)
        rgba = np.zeros((n, 4))
        rgba[:, :3] = C_TRAIL
        rgba[:, 3] = np.linspace(0.04, 0.70, n)
        self.trail_lc.set_segments(segs)
        self.trail_lc.set_color(rgba)

    def _update_caption(self):
        step = self.tfw.beta_step
        k = self.tfw.beta_level(self.beta)
        text = (f'certified slice  β {math.degrees(k * step):.1f}° to '
                f'{math.degrees((k + 1) * step):.1f}°')
        if self.exact_mask is not None and self.show_exact:
            text += f'     exact at β = {math.degrees(self.exact_beta) % 360:.1f}°'
        self.plot_caption.set_text(text)

    def _update_coils(self, sol):
        cur = sol['currents']
        for bar, c in zip(self.bars, cur):
            bar.set_height(c)
            bar.set_color(C_BAD if abs(c) > 0.97 * I_MAX else C_BAR)
        kind = 'minimum-norm' if sol['min_norm'] else 'bounded least squares'
        meta = (f'peak {np.max(np.abs(cur)):.2f} A   ·   '
                f'norm {np.linalg.norm(cur):.2f} A   ·   {kind}')
        if sol['saturated']:
            meta = f'at limit   ·   {meta}'
        self.coil_card.meta.set_text(meta)
        self.coil_card.meta.set_color(C_BAD if sol['saturated'] else C_MUTED)

    def _update_task(self, sol):
        cmd = np.array([0.0, B_HOLD, 0.0, self.thrust_cmd])
        ach = np.asarray(sol['achieved'], float)
        scales = np.array([B_HOLD, B_HOLD, F_MAX, F_MAX])
        units = ('mT', 'mT', 'mN', 'mN')
        is_ok = np.abs(ach - cmd) / scales < REL_OK

        self.task_cmd.set_xdata(np.clip(cmd / scales, -1.0, 1.0))
        for bar, val, ok in zip(self.task_bars, ach / scales, is_ok):
            bar.set_width(float(np.clip(val, -1.0, 1.0)))
            bar.set_color(C_BAR if ok else C_BAD)

        for i in range(4):
            self.task_vals[i].set_text(f'{_qty(ach[i], 1e3)} {units[i]}')
            self.task_vals[i].set_color(C_INK if is_ok[i] else C_BAD)
            self.task_res[i].set_text(
                '' if is_ok[i]
                else _fmt_residual(ach[i] - cmd[i], scales[i], units[i][1]))

        label, kind = _rel_error_label(sol['rel_error'])
        self.task_card.meta.set_text(label)
        self.task_card.meta.set_color(_STATUS_EDGE[kind])

    def _update_headings(self, ok):
        n_ok = int(ok.sum())
        n = self.tfw.n_levels
        segs, flags = _heading_arcs(ok, self.tfw.beta_step, 1.0)
        self.head_ring.set_segments(segs)
        self.head_ring.set_color([C_CERT if f else C_BLOCK for f in flags])
        self.head_needle.set_data([0, 0.58 * math.cos(self.beta)],
                                  [0, 0.58 * math.sin(self.beta)])
        self.head_big.set_text(f'{n_ok / n * 360:.0f}°')
        self.head_levels.set_text(f'{n_ok} of {n} heading levels')

    def _update_workspace(self, inside):
        cert = float(self.tfw.slice_at(self.beta).mean())
        self.ws_fill[0].set_width(cert * self.ws_track_w)
        self.ws_vals[0].set_text(f'{cert * 100:.0f}%')

        exact_now = (self.exact_mask is not None
                     and abs(self.beta - self.exact_beta) < 1e-9)
        if exact_now:
            ex = float(self.exact_mask.mean())
            self.ws_fill[1].set_width(ex * self.ws_track_w)
            self.ws_vals[1].set_text(f'{ex * 100:.0f}%')
            self.ws_hint.set_text('')
        else:
            self.ws_fill[1].set_width(0.0)
            self.ws_vals[1].set_text('')
            self.ws_hint.set_text('stale, press R' if self.exact_mask is not None
                                  else 'press R to compute')
        self.ws_track[1].set_visible(exact_now)

        self.ws_card.meta.set_text(f'blocking on {self.block_on}')
        self.ws_card.meta.set_color(C_EXACT if self.block_on == 'exact' else C_CERT)
        self.ws_dot.set_color(C_OK if inside else C_BAD)
        self.ws_inside.set_text('magnet inside the certified set' if inside
                                else 'magnet outside the certified set')

    def _update_field_force(self, B_world, F_world):
        cb, sb = math.cos(self.beta), math.sin(self.beta)
        for key, vec in (('B', B_world), ('F', F_world)):
            along = vec[0] * cb + vec[1] * sb
            lateral = -vec[0] * sb + vec[1] * cb
            for txt, v in zip(self.ff_vals[key],
                              (vec[0], vec[1], vec[2], along, lateral)):
                txt.set_text(_qty(v, 1e3, signed=True))

    def _set_status(self, kind, chip, message, flags=''):
        patch = self.status_chip.get_bbox_patch()
        patch.set_facecolor(_STATUS_FACE[kind])
        patch.set_edgecolor(_STATUS_EDGE[kind])
        self.status_chip.set_text(chip)
        self.status_chip.set_color(_STATUS_EDGE[kind])
        self.status_msg.set_text(message)
        self.status_flags.set_text(flags)
        self.status_flags.set_color(_STATUS_EDGE[kind])

    def _update_status(self, sol, F_world, inside):
        if self.blocked_move or not inside:
            kind, chip = 'bad', 'Blocked'
        elif self.blocked_turn or self.sliding or sol['saturated']:
            kind, chip = 'warn', 'Limit'
        elif self.thrust_cmd != 0.0:
            kind, chip = 'ok', 'Drive'
        else:
            kind, chip = 'ok', 'Hold'

        if self.thrust_cmd > 0:
            motion = 'Thrust forward along heading'
        elif self.thrust_cmd < 0:
            motion = 'Thrust backward along heading'
        else:
            motion = 'Holding field, zero thrust'

        flags = []
        if self.sliding:
            flags.append('sliding')
        if self.blocked_move:
            flags.append('move blocked')
        if self.blocked_turn:
            flags.append('turn blocked')
        if sol['saturated']:
            flags.append('coil limit')
        if not inside:
            flags.append('outside certified')
        self._set_status(kind, chip, motion, '  ·  '.join(flags))

        x, y = self.pos
        speed = np.linalg.norm(F_world[:2]) / DRAG_C * 1e3
        self.status_vals['x'].set_text(f'{_qty(x, 1e3, 1, signed=True)} mm')
        self.status_vals['y'].set_text(f'{_qty(y, 1e3, 1, signed=True)} mm')
        self.status_vals['beta'].set_text(f'{math.degrees(self.beta) % 360:.1f}°')
        self.status_vals['v'].set_text(f'{speed:.1f} mm/s')

    # ------------------------------------------------------------------

    def run(self):
        self.anim = FuncAnimation(self.fig, self.update,
                                  interval=int(DT * 1000),
                                  blit=False, cache_frame_data=False)
        plt.show()


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--grid', type=int, default=61,
                   help='resolution of the exact region computed by R (default 61)')
    p.add_argument('--tfw', default=None,
                   help='path to the certified 3-D TFW JSON')
    p.add_argument('--raster', type=int, default=400,
                   help='raster resolution of the certified slice (default 400)')
    p.add_argument('--start', type=float, nargs=2, metavar=('X', 'Y'),
                   default=list(START_XY),
                   help=f'starting position of the free magnet, in metres '
                        f'(default {START_XY[0]} {START_XY[1]})')
    p.add_argument('--screenshot', default=None,
                   help='save a still to this path and exit (uses the Agg backend)')
    args = p.parse_args()

    print(__doc__)
    sim = Case6Sim(grid=args.grid, tfw_path=args.tfw, raster=args.raster,
                   start=tuple(args.start))
    print(f'certified boxes loaded: {len(sim.tfw.boxes)}'
          f'   beta levels: {len(sim.tfw.beta_levels())}')
    print(f'start pose: x = {sim.pos[0] * 1e3:.2f} mm, y = {sim.pos[1] * 1e3:.2f} mm, '
          f'beta = {math.degrees(sim.beta):.1f} deg')
    if args.screenshot:
        sim.update(0)
        sim.fig.savefig(args.screenshot, dpi=140,
                        facecolor=sim.fig.get_facecolor(), bbox_inches='tight')
        print(f'saved {args.screenshot}')
        return
    print('\nClick the plot window once so it has keyboard focus, then drive with W/S/A/D.\n')
    sim.run()


if __name__ == '__main__':
    main()
