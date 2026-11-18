#!/usr/bin/env python3
# Run from the repository root (after presentation --only plans):
#   .venv/bin/python -B -m revised_case_studies.cs1_cylinder_interaction.experiment_refs
"""Reference trajectories for the Case Study 1 hardware experiments.

Writes ``results/cs1/experiments/*.json``, each with columns [x_m, y_m, beta_rad, t_s]:

  exp1_rotate_A.json, exp1_rotate_B.json   hold position, turn 360 deg at 10 deg/s
  exp2_crossing.json                       straight crossing (10,-45) -> (10,45) mm, 5 mm/s
  exp3_<layout>_planned.json / _straight.json
                                           planned path in the sharp TFW and straight baseline
  exp4_speed_<v>mmps.json                  the one-magnet planned path at 5, 10, 20, 30 mm/s
"""
from __future__ import annotations

import json
import math

import numpy as np

from revised_case_studies.cs1_cylinder_interaction import common as c
from revised_case_studies.cs1_cylinder_interaction import nonholonomic as nh
from revised_case_studies.cs1_cylinder_interaction import presentation as pr
from revised_case_studies.shared.paths import out_dir

OUT = out_dir('cs1') / 'experiments'
ROTATE = {'A': (-0.025, 0.020), 'B': (0.028, 0.014)}   # all headings / two infeasible sectors
B0 = math.radians(90)                                    # start heading (feasible at A and B)
W_ROT = math.radians(10)                                 # rad/s
SPEEDS = (0.005, 0.010, 0.020, 0.030)


def dump(name, traj, **meta):
    traj = np.asarray(traj, float)
    (OUT / name).write_text(json.dumps({**meta, 'columns': ['x_m', 'y_m', 'beta_rad', 't_s'],
                                        'trajectory': traj.tolist()}))
    print(f'{name}: {len(traj)} samples, {traj[-1, 3]:.1f} s')


def rotation(x, y, dt=0.05, dwell=2.0):
    T = np.arange(0, 2 * math.pi / W_ROT + 2 * dwell + 1e-9, dt)
    b = B0 + np.clip(T - dwell, 0, 2 * math.pi / W_ROT) * W_ROT
    return np.column_stack([np.full_like(T, x), np.full_like(T, y), b, T])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for k, (x, y) in ROTATE.items():
        dump(f'exp1_rotate_{k}.json', rotation(x, y), position_m=[x, y],
             note='Turn in place at 10 deg/s from 90 deg; one full turn.')
    for layout in pr.MISSIONS:
        for kind in ('planned', 'straight'):
            traj = c.load_trajectory(c.out_dir('cs1') / 'presentation' / f'trajectory_{layout}_{kind}.json')
            dump(f'exp3_{layout}_{kind}.json', traj, layout=layout, kind=kind)
    traj = c.load_trajectory(c.NAIVE)
    dump('exp2_crossing.json', traj, layout='single', note='Straight crossing at 90 deg, 5 mm/s.')
    s, g = pr.MISSIONS['single']
    path = nh.plan('single', s, g, costs=pr.SMOOTH)
    for v in SPEEDS:
        dump(f'exp4_speed_{round(v * 1e3)}mmps.json', nh.timed(path, v=v), layout='single',
             speed_m_s=v, note='Same planned path; driving speed v, turning rate 30 deg/s.')


if __name__ == '__main__':
    main()
