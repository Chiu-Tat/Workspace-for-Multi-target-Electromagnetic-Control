"""Case Study 1 repeated-speed trial: metrics and figure.

Run from the repository root:
  .venv/bin/python -B -m revised_case_studies.cs1_cylinder_interaction.speed_sweep

Reads the ROS_CAN trial telemetry, writes speed_sweep.json and cs1_speed.{pdf,png}
to results/cs1/publication, and copies the figure to the paper's images/case_study_1.

One pass is the closed loop A -> B -> C -> A at a fixed time scaling. Errors are taken
on the moving segments (AB, BC, CA) only. The along-track error is the measured
position minus the reference position, projected on the reference direction of motion,
so a positive value means the magnet is ahead of the reference.
"""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from revised_case_studies.cs1_cylinder_interaction import figures_v2 as F
from revised_case_studies.cs1_cylinder_interaction import publication as P

TRIAL = 'trial_20261107_173319_59f597cb'
DATA = Path(__file__).with_name('data')
MOVING = ('AB', 'BC', 'CA')
TASK_FORCE_N = 0.1
CURRENT_LIMIT_A = 15.0
SPEED_COLORS = {1: '#9cc3e4', 2: '#5b9bd0', 3: '#2f6fae', 4: '#163f6e'}


def num(row, key):
    try:
        return float(row[key])
    except (TypeError, ValueError):
        return None


def load():
    session = json.loads((DATA / 'speed_trial.json').read_text())
    cruise = session['configuration']['repeated_speed']['cruise_speed_m_s']
    passes = defaultdict(list)
    with (DATA / 'speed_telemetry.csv').open() as source:
        for row in csv.DictReader(source):
            if row['stage'] in MOVING and row['pass_number'] not in ('', '0'):
                passes[int(row['pass_number'])].append(row)
    return session, cruise, passes


def pass_metrics(rows, cruise):
    valid = [r for r in rows if r['pose_valid'] == 'True' and num(r, 'along_track_error_m') is not None]
    along = np.array([num(r, 'along_track_error_m') for r in valid]) * 1e3
    cross = np.array([num(r, 'cross_track_error_m') for r in valid]) * 1e3
    head = np.degrees([num(r, 'heading_error_rad') for r in valid])
    pos = np.hypot(along, cross)
    force = np.abs([num(r, 'h_des_F_axial_N') for r in rows if num(r, 'h_des_F_axial_N') is not None])
    lateral = np.abs([num(r, 'h_des_F_lateral_N') for r in rows if num(r, 'h_des_F_lateral_N') is not None])
    current = max(max(abs(num(r, f'command_{i:02d}_A')) for i in range(1, 11)) for r in rows)
    margin = [num(r, 'measured_tfw_margin_m') for r in valid if num(r, 'measured_tfw_margin_m') is not None]
    fixed = [num(r, 'fixed_distance_m') for r in valid if num(r, 'fixed_distance_m') is not None]
    rms = lambda a: float(np.sqrt(np.mean(np.square(a))))
    return dict(
        speed_mm_s=round(float(rows[0]['speed_scale']) * cruise * 1e3, 6),
        drift_check=rows[0]['drift_check'] == 'True',
        samples=len(rows), valid_fraction=len(valid) / len(rows),
        position_rmse_mm=rms(pos), position_p95_mm=float(np.percentile(pos, 95)),
        position_max_mm=float(pos.max()),
        along_mean_mm=float(along.mean()), along_rmse_mm=rms(along), cross_rmse_mm=rms(cross),
        heading_rmse_deg=rms(head),
        axial_force_p95_N=float(np.percentile(force, 95)), axial_force_max_N=float(force.max()),
        lateral_force_max_N=float(lateral.max()),
        peak_current_A=float(current),
        slew_limited_fraction=sum(r['slew_limit_active'] == 'True' for r in rows) / len(rows),
        current_limited_fraction=sum(r['current_limit_active'] == 'True' for r in rows) / len(rows),
        min_tfw_margin_mm=min(margin) * 1e3, min_fixed_distance_mm=min(fixed) * 1e3,
    )


def summarize(per_pass):
    groups = defaultdict(list)
    for m in per_pass.values():
        if not m['drift_check']:
            groups[m['speed_mm_s']].append(m)
    out = {}
    for speed, ms in sorted(groups.items()):
        keys = [k for k in ms[0] if isinstance(ms[0][k], float)]
        out[f'{speed:g}'] = {'passes': len(ms), **{k: float(np.mean([m[k] for m in ms])) for k in keys},
                             **{k + '_sd': float(np.std([m[k] for m in ms], ddof=1)) for k in
                                ('position_rmse_mm', 'along_mean_mm', 'axial_force_max_N')}}
    return out


def speed_color(speed):
    return SPEED_COLORS[int(round(speed))]


def scatter(ax, per_pass, key, scale=1.0):
    for m in per_pass.values():
        c = speed_color(m['speed_mm_s'])
        ax.plot(m['speed_mm_s'] + (0.12 if m['drift_check'] else 0), m[key] * scale,
                'o', ms=3.2, mfc='white' if m['drift_check'] else c, mec=c, mew=.8, zorder=5)


def mean_line(ax, summary, key, scale=1.0, **kw):
    s = sorted(float(k) for k in summary)
    ax.plot(s, [summary[f'{v:g}'][key] * scale for v in s], **{'color': P.INK, 'lw': .9, **kw})


def style(ax, ylabel):
    ax.set_xlim(.5, 4.6); ax.set_xticks([1, 2, 3, 4])
    ax.set_xlabel('Reference speed (mm/s)', labelpad=1, fontsize=6.8)
    ax.set_ylabel(ylabel, labelpad=1, fontsize=6.8)
    ax.tick_params(length=2, pad=1.5, labelsize=6.2)
    ax.spines[['top', 'right']].set_visible(False)


def figure(session, passes, per_pass, summary):
    P.style() if hasattr(P, 'style') else None
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7, 'pdf.fonttype': 42,
                         'ps.fonttype': 42, 'savefig.dpi': 250, 'axes.edgecolor': '#777b80',
                         'text.color': P.INK, 'axes.labelcolor': P.INK})
    fig = plt.figure(figsize=(3.45, 3.45))
    gs = fig.add_gridspec(2, 2, left=.13, right=.98, bottom=.1, top=.95, hspace=.48, wspace=.5)

    # (a) loop over the heading coverage of the single layout
    ax = fig.add_subplot(gs[0, 0])
    F.coverage_map(ax, 'single')
    F.draw_fixed(ax, 'single', guard=False)
    ref = [r for r in passes[10] if num(r, 'ref_x_m') is not None]
    ax.plot([num(r, 'ref_x_m') * 1e3 for r in ref], [num(r, 'ref_y_m') * 1e3 for r in ref],
            color=F.REF_INK, lw=.7, ls=(0, (3, 2)), zorder=7)
    for p, color in ((1, F.MEAS), (10, P.ORANGE)):
        rows = [r for r in passes[p] if r['pose_valid'] == 'True']
        ax.plot([num(r, 'x_m') * 1e3 for r in rows], [num(r, 'y_m') * 1e3 for r in rows],
                color=color, lw=.9, zorder=8)
    wp = session['configuration']['repeated_speed']
    for name, off in (('A', (-9, -7)), ('B', (-9, 2)), ('C', (2, 3))):
        x, y = wp[name][0] * 1e3, wp[name][1] * 1e3
        ax.plot(x, y, 'o', ms=3, mfc='white', mec=P.INK, mew=.7, zorder=9)
        ax.annotate(name, (x, y), xytext=off, textcoords='offset points', fontsize=6.5, zorder=10)
    F.axes_mm(ax)
    F.title(ax, 'a', 'Loop A–B–C–A')
    ax.legend(handles=[Line2D([], [], color=F.REF_INK, lw=.7, ls=(0, (3, 2)), label='Reference'),
                       Line2D([], [], color=F.MEAS, lw=1, label='Measured, 1 mm/s'),
                       Line2D([], [], color=P.ORANGE, lw=1, label='Measured, 4 mm/s')],
              loc='lower right', fontsize=4.9, frameon=True, framealpha=.85, edgecolor='none',
              handlelength=1.1, borderpad=.25, handletextpad=.4, labelspacing=.25)

    # (b) tracking error
    ax = fig.add_subplot(gs[0, 1])
    scatter(ax, per_pass, 'position_rmse_mm'); mean_line(ax, summary, 'position_rmse_mm')
    scatter(ax, per_pass, 'along_mean_mm'); mean_line(ax, summary, 'along_mean_mm', ls='--')
    ax.axhline(0, color='#adb0b4', lw=.5, zorder=0)
    ax.text(.6, .74, 'Position RMSE', fontsize=5.8)
    ax.text(2.2, -.12, 'Mean along-track', fontsize=5.8)
    ax.set_ylim(-.2, 1.0)
    style(ax, 'Error (mm)'); F.title(ax, 'b', 'Tracking error')

    # (c) commanded axial force against the task bound
    ax = fig.add_subplot(gs[1, 0])
    scatter(ax, per_pass, 'axial_force_max_N', 1e3); mean_line(ax, summary, 'axial_force_max_N', 1e3)
    scatter(ax, per_pass, 'axial_force_p95_N', 1e3); mean_line(ax, summary, 'axial_force_p95_N', 1e3, ls='--')
    ax.axhline(TASK_FORCE_N * 1e3, color=P.ORANGE, lw=.9)
    ax.text(.6, TASK_FORCE_N * 1e3 + 3, 'TFW task bound, 100 mN', fontsize=5.8, color=P.ORANGE)
    ax.text(2.4, 58, 'Maximum', fontsize=5.8); ax.text(2.6, 22, '95th pct.', fontsize=5.8)
    ax.set_ylim(0, 115)
    style(ax, '|Axial force request| (mN)'); F.title(ax, 'c', 'Force demand')

    # (d) peak commanded current against the coil limit
    ax = fig.add_subplot(gs[1, 1])
    scatter(ax, per_pass, 'peak_current_A'); mean_line(ax, summary, 'peak_current_A')
    ax.axhline(CURRENT_LIMIT_A, color=P.ORANGE, lw=.9)
    ax.text(.6, CURRENT_LIMIT_A + .35, 'Current limit, 15 A', fontsize=5.8, color=P.ORANGE)
    ax.set_ylim(0, 17)
    style(ax, 'Peak |current| (A)'); F.title(ax, 'd', 'Coil current')

    P.OUT.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(P.OUT / f'cs1_speed.{ext}', facecolor='white', bbox_inches='tight', pad_inches=.04)
    plt.close(fig)


def main():
    session, cruise, passes = load()
    per_pass = {p: pass_metrics(rows, cruise) for p, rows in sorted(passes.items())}
    summary = summarize(per_pass)
    P.OUT.mkdir(parents=True, exist_ok=True)
    json.dump({'trial': TRIAL, 'per_pass': per_pass, 'per_speed': summary},
              open(P.OUT / 'speed_sweep.json', 'w'), indent=1)
    figure(session, passes, per_pass, summary)
    for k, v in summary.items():
        print(k, {a: round(b, 4) for a, b in v.items()})
    d = per_pass[max(per_pass)]
    print('drift', {a: (round(b, 4) if isinstance(b, float) else b) for a, b in d.items()})


if __name__ == '__main__':
    main()
