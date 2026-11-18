#!/usr/bin/env python3
# Run from the repository root:
#   python -m revised_case_studies.cs3_helical_robots.step_out
"""Case Study 3 -- step-out versus electromagnetic feasibility.

Captured rotation at one certified configuration: robot 1's commanded
frequency ramps 0 -> 2 * STEP_OUT_HZ while robot 2 holds a constant 2 Hz.
The field allocation stays exact throughout (zero slack), so the loss of
synchronisation of robot 1 above its step-out frequency is a dynamic limit,
not electromagnetic infeasibility, and robot 2 is unaffected.

Figure: D6_step_out (+ D_step_out.json)
"""

from __future__ import annotations

import argparse
import json
import math

import matplotlib.pyplot as plt
import numpy as np

from revised_case_studies.cs3_helical_robots import config as K
from revised_case_studies.cs3_helical_robots.dynamics import FieldController, step_plant
from revised_case_studies.cs3_helical_robots.field_check import pick_configs
from revised_case_studies.shared import paths as P

CASE = 'cs3'
F2 = 2.0            # Hz, robot 2 constant command
RAMP_S = 12.0       # s


def run():
    a = np.array(pick_configs()['inside'])
    ctrl = FieldController()
    theta, phi = np.zeros(2), np.zeros(2)
    dt = K.DT_PLANT
    win = int(round(0.25 / dt))
    t_log, f_cmd, f_act, slack = [], [], [], []
    th_hist = []
    for k in range(int(RAMP_S / dt)):
        t = k * dt
        f = np.array([2 * K.STEP_OUT_HZ * t / RAMP_S, F2])
        phi = phi + 2 * math.pi * f * dt
        I, sol = ctrl.currents(a, phi)
        a, theta, _ = step_plant(a, theta, I, (False, False))
        th_hist.append(theta.copy())
        if k % win == 0 and k >= win:
            t_log.append(t)
            f_cmd.append(f.copy())
            f_act.append((th_hist[-1] - th_hist[-win]) / (win * dt) / (2 * math.pi))
            slack.append(float(np.max(np.abs(sol['slack']))) / K.B_ROT)
    return np.array(t_log), np.array(f_cmd), np.array(f_act), np.array(slack)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--show', action='store_true')
    args = ap.parse_args()
    P.apply_style()
    t, fc, fa, slack = run()
    lost = (fc[:, 0] > 1.0) & (fa[:, 0] < 0.8 * fc[:, 0])
    f_so = float(fc[np.argmax(lost), 0]) if lost.any() else float('nan')
    summary = {'measured_step_out_hz': f_so, 'model_step_out_hz': K.STEP_OUT_HZ,
               'robot2_max_freq_error_hz': float(np.abs(fa[:, 1] - F2).max()),
               'max_field_slack_rel': float(slack.max())}
    print(summary)
    fig, axes = plt.subplots(2, 1, figsize=(6.5, 5), sharex=True, constrained_layout=True)
    axes[0].plot(t, fc[:, 0], '--', color='#1f5fa8', label='robot 1 command')
    axes[0].plot(t, fa[:, 0], color='#1f5fa8', label='robot 1 actual')
    axes[0].plot(t, fc[:, 1], '--', color='#d98a1c', label='robot 2 command')
    axes[0].plot(t, fa[:, 1], color='#d98a1c', label='robot 2 actual')
    axes[0].axhline(K.STEP_OUT_HZ, ls=':', color=P.C_OUTSIDE, label='step-out (model)')
    axes[0].set_ylabel('rotation freq. (Hz)')
    axes[0].legend(frameon=False, fontsize=7)
    axes[1].plot(t, np.maximum(slack, 1e-12), color='k')
    axes[1].set_yscale('log')
    axes[1].set_ylim(1e-12, 1)
    axes[1].set_ylabel('field slack (relative)')
    if slack.max() == 0:
        axes[1].text(0.5, 0.5, 'slack = 0 at every sample: fields exact throughout',
                     transform=axes[1].transAxes, ha='center')
    axes[1].set_xlabel('time (s)')
    axes[0].set_title('Step-out of robot 1 with exact fields; robot 2 unaffected')
    P.save(fig, CASE, 'D6_step_out')
    (P.out_dir(CASE) / 'D_step_out.json').write_text(json.dumps(summary, indent=1))
    if args.show:
        plt.show()


if __name__ == '__main__':
    main()
