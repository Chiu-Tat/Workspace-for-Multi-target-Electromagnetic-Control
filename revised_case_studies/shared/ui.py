"""Look and layout shared by the three interactive simulators.

Re-uses the palette, fonts and card helpers of the Case 6 interactive
simulator (``cases/case_6_interaction_obstacle_avoidance/simulation/
interactive.py``) so all four simulators look the same, and adds the header,
footer and status-chip builders they have in common.
"""

from __future__ import annotations

import sys

import matplotlib

if any(a == '--screenshot' or a.startswith('--screenshot=') for a in sys.argv):
    matplotlib.use('Agg')

import matplotlib.pyplot as plt  # noqa: E402

from cases.case_6_interaction_obstacle_avoidance.simulation import interactive as _c6  # noqa: E402

# Palette and helpers (names kept as in the Case 6 simulator).
C_BG, C_PANEL, C_SCENE = _c6.C_BG, _c6.C_PANEL, _c6.C_SCENE
C_INK, C_MUTED, C_FAINT = _c6.C_INK, _c6.C_MUTED, _c6.C_FAINT
C_LINE, C_GRID, C_TRACK = _c6.C_LINE, _c6.C_GRID, _c6.C_TRACK
C_CERT, C_CERT_RGB = _c6.C_CERT, _c6.C_CERT_RGB
C_EXACT, C_EXACT_RGB = _c6.C_EXACT, _c6.C_EXACT_RGB
C_SOUTH, C_NORTH, C_F = _c6.C_SOUTH, _c6.C_NORTH, _c6.C_F
C_TRAIL, C_BAR = _c6.C_TRAIL, _c6.C_BAR
C_OK, C_WARN, C_BAD, C_BLOCK = _c6.C_OK, _c6.C_WARN, _c6.C_BAD, _c6.C_BLOCK
STATUS_FACE, STATUS_EDGE = _c6._STATUS_FACE, _c6._STATUS_EDGE
FIG_W, FIG_H = _c6.FIG_W, _c6.FIG_H
MARGIN, CARD_PAD, CARD_GAP, TITLE_Y = _c6.MARGIN, _c6.CARD_PAD, _c6.CARD_GAP, _c6.TITLE_Y
C_ROBOT = ('#2f6db3', '#d98a1c')

Card = _c6._Card
apply_style = _c6._apply_style
fig_rect = _c6._fig_rect
qty = _c6._qty
arrow = _c6._arrow
heading_arcs = _c6._heading_arcs
magnet_polygons = _c6._magnet_polygons


def chrome(fig, case_title, info, key_hints):
    """Header (brand, case title, info line) and footer (key caps)."""
    ax = fig.add_axes([0, 0, 1, 1], zorder=-1)
    ax.set_xlim(0, FIG_W)
    ax.set_ylim(0, FIG_H)
    ax.set_axis_off()
    ax.set_navigate(False)
    y = FIG_H - 0.32
    brand = ax.text(MARGIN, y, 'eMNS', fontsize=13, fontweight='bold',
                    color=C_INK, va='center')
    ax.annotate(case_title, xy=(1, 0.5), xycoords=brand, xytext=(12, 0),
                textcoords='offset points', va='center', fontsize=12, color=C_MUTED)
    ax.text(FIG_W - MARGIN, y, info, ha='right', va='center', fontsize=8.5,
            color=C_FAINT)

    cap = dict(boxstyle='round,pad=0.35,rounding_size=0.3', fc=C_PANEL, ec=C_LINE, lw=0.7)

    def chain(prev, s, gap, **kw):
        if prev is None:
            return ax.text(MARGIN + 0.04, 0.26, s, va='center', **kw)
        return ax.annotate(s, xy=(1, 0.5), xycoords=prev, xytext=(gap, 0),
                           textcoords='offset points', va='center', **kw)

    prev, gap = None, 0
    for keys, action in key_hints:
        for key in keys:
            prev = chain(prev, key, gap, fontsize=7.5, color=C_INK, bbox=cap)
            gap = 8
        prev = chain(prev, action, 8, fontsize=8, color=C_MUTED)
        gap = 18
    return ax


class StatusCard:
    """Chip + message + flags + a row of labelled values (Case 6 style)."""

    def __init__(self, card, fields):
        self.card = card
        ax, h = card.ax, card.height
        self.chip = ax.text(
            CARD_PAD + 0.36, h / 2, '', ha='center', va='center', fontsize=8.8,
            fontweight='medium', color=C_OK, zorder=3,
            bbox=dict(boxstyle='round,pad=0.55,rounding_size=0.55',
                      fc=STATUS_FACE['ok'], ec=STATUS_EDGE['ok'], lw=0.8))
        x = CARD_PAD + 0.86
        self.msg = ax.text(x, h * 0.66, '', fontsize=9.5, color=C_INK, va='center')
        self.flags = ax.text(card.width - CARD_PAD, h * 0.66, '', fontsize=8,
                             ha='right', va='center')
        self.vals = {}
        for key, label, dx in fields:
            ax.text(x + dx, h * 0.31, label, fontsize=8, color=C_FAINT, va='center')
            self.vals[key] = ax.text(x + dx + 0.17, h * 0.31, '', fontsize=8.2,
                                     color=C_INK, va='center')

    def set(self, kind, chip, message, flags=''):
        patch = self.chip.get_bbox_patch()
        patch.set_facecolor(STATUS_FACE[kind])
        patch.set_edgecolor(STATUS_EDGE[kind])
        self.chip.set_text(chip)
        self.chip.set_color(STATUS_EDGE[kind])
        self.msg.set_text(message)
        self.flags.set_text(flags)
        self.flags.set_color(STATUS_EDGE[kind])


def coil_axes(card, n=10, i_max=15.0):
    """Bar chart of coil currents inside ``card``; returns the bar container."""
    import numpy as np
    ax = card.inset(left=0.66, bottom=0.30, right=CARD_PAD, top=0.50)
    idx = np.arange(1, n + 1)
    ax.set_xlim(0.4, n + 0.6)
    ax.set_ylim(-i_max * 1.22, i_max * 1.22)
    ax.set_xticks(idx)
    ax.set_yticks([-i_max, 0.0, i_max])
    ax.set_yticklabels([f'{qty(-i_max, decimals=0)} A', '0', f'+{i_max:.0f} A'])
    for lim in (-i_max, i_max):
        ax.axhline(lim, color=C_BAD, lw=0.8, ls=(0, (3.0, 2.5)), alpha=0.5, zorder=1)
    ax.axhline(0.0, color=C_INK, lw=0.6, alpha=0.25, zorder=3)
    return ax.bar(idx, np.zeros(n), width=0.56, color=C_BAR, edgecolor='none', zorder=2)


def update_coils(card, bars, currents, i_max=15.0, note='', compact=False):
    import numpy as np
    for bar, c in zip(bars, currents):
        bar.set_height(c)
        bar.set_color(C_BAD if abs(c) > 0.97 * i_max else C_BAR)
    sat = np.max(np.abs(currents)) > i_max - 1e-3
    meta = f'peak {np.max(np.abs(currents)):.2f} A'
    if not compact:
        meta += f'   ·   norm {np.linalg.norm(currents):.2f} A'
    if note and not compact:
        meta += f'   ·   {note}'
    card.meta.set_text(('at limit   ·   ' if sat else '') + meta)
    card.meta.set_color(C_BAD if sat else C_MUTED)
    return sat


def style_scene(ax):
    ax.set_facecolor(C_SCENE)
    ax.tick_params(labelsize=8.5, labelcolor=C_MUTED, length=0, pad=6)
    ax.grid(True, color=C_GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_color(C_LINE)
        sp.set_linewidth(0.8)


def screenshot_or_run(sim, path, frames=1, dt_ms=50):
    """Save a still after ``frames`` updates (headless), or run interactively."""
    from matplotlib.animation import FuncAnimation
    if path:
        for k in range(frames):
            sim.update(k)
        sim.fig.savefig(path, dpi=140, facecolor=sim.fig.get_facecolor(),
                        bbox_inches='tight')
        print(f'saved {path}')
        return
    print('\nClick the plot window once so it has keyboard focus.\n')
    sim.anim = FuncAnimation(sim.fig, sim.update, interval=dt_ms, blit=False,
                             cache_frame_data=False)
    plt.show()
