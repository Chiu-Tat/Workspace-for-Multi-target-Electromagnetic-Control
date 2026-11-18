"""Case Study 1 hardware records: per-sample TFW membership and force margin.

Run from the repository root:
  .venv/bin/python -B -m revised_case_studies.cs1_cylinder_interaction.hardware

Reads the archived ROS_CAN trials (telemetry.csv, session.json) and writes
results/cs1/hardware/trials.json.  For every logged pose it records:

  tfw       TFW-in membership in the sharp 3-D TFW at the measured heading
  lam       nominal full-task force margin: the largest scale s such that both
            thrust endpoints +-s * 0.1 N are attainable together with
            By = 0, Bz = 20 mT, fy = 0 and |I| <= 15 A (0 if the field rows fail).
            The full task of Eq. (case3_constraint) holds iff lam >= 1.
  d_fixed   centre distance to the nearest fixed magnet
  i_cmd     largest absolute commanded coil current
Reference poses of the autonomous runs get the same tfw / lam fields.
"""
from __future__ import annotations

import csv
import json
import os
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from revised_case_studies.cs1_cylinder_interaction import common as C
from revised_case_studies.shared.paths import out_dir

ROS = Path(os.environ.get('ROS_CAN_SRC', 'ROS_CAN/src'))
MANUAL = ROS / 'case_study1_single_fixed_magnet/results'
AUTO = ROS / 'case_study1_auto_planning_control/results'
TRIALS = {
    # key: (layout, folder, kind)
    'kb_single': ('single', MANUAL / 'trial_20261022_orientation_aware', 'keyboard'),
    'kb_gate': ('gate', MANUAL / 'gate/trial_20261026_Orientation aware', 'keyboard'),
    'kb_triangle': ('triangle', MANUAL / 'triangle/trial_20261026_165732_c97a97f2', 'keyboard'),
    'kb_single_fail': ('single', MANUAL / 'trial_20261022_outside_TFW', 'keyboard'),
    'kb_gate_fail': ('gate', MANUAL / 'gate/trial_20261026_outside_TFW', 'keyboard'),
    'auto_single': ('single', AUTO / 'single/trial_20261024_success_planned_path', 'auto'),
    'auto_gate': ('gate', AUTO / 'gate/trial_20261026_162606_5d8cf52e', 'auto'),
    'auto_triangle': ('triangle', AUTO / 'triangle/trial_20261026_173216_da359149', 'auto'),
    'auto_direct': ('single', AUTO / 'single/trial_20261026_failure_direct_path', 'auto'),
}
OUT = out_dir('cs1') / 'hardware'


def margin(x, y, beta, layout):
    """Nominal full-task force margin (see module docstring)."""
    r = C.force_ranges(x, y, beta, layout)['fz_range']
    if not all(math.isfinite(v) for v in r):
        return 0.0
    return max(0.0, min(r[1], -r[0]) / C.F_MAX)


def tfw_in(x, y, beta, layout):
    """Exact membership in a TFW-in box [x0, x1) x [y0, y1) x [b0, b1)."""
    b = C.tfw(layout).boxes
    beta = beta % (2 * math.pi)
    return bool(np.any((b[:, 0] <= x) & (x < b[:, 1]) & (b[:, 2] <= y) & (y < b[:, 3])
                       & (b[:, 4] <= beta) & (beta < b[:, 5])))


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return math.nan


def _flag(v):
    return str(v).strip().lower() in ('1', 'true', 'yes')


def _eval(args):
    x, y, b, layout = args
    return tfw_in(x, y, b, layout), margin(x, y, b, layout), C.nearest_fixed(x, y, layout)[0]


def load_trial(key, pool):
    layout, folder, kind = TRIALS[key]
    rows = list(csv.DictReader(open(folder / 'telemetry.csv')))
    session = json.loads((folder / 'session.json').read_text())
    t = np.array([_num(r['elapsed_s']) for r in rows])
    x = np.array([_num(r['x_m']) for r in rows])
    y = np.array([_num(r['y_m']) for r in rows])
    b = np.array([_num(r['heading_wrapped_rad']) for r in rows]) % (2 * math.pi)
    armed = np.array([_flag(r['armed']) for r in rows])
    valid = armed & np.array([_flag(r['tracking_valid']) and _flag(r['pose_fresh'])
                              for r in rows]) & np.isfinite(x) & np.isfinite(y) & np.isfinite(b)
    cmd = np.array([[_num(r[f'command_{k:02d}_A']) for k in range(1, 11)] for r in rows])
    meas = np.array([[_num(r[f'measured_{k:02d}_A']) for k in range(1, 11)] for r in rows])
    reason = [r['reason'] for r in rows]
    phase = [r['hold_phase'] for r in rows]
    idx = np.flatnonzero(valid)
    res = list(pool.map(_eval, [(x[i], y[i], b[i], layout) for i in idx], chunksize=16))
    n = len(rows)
    tfw = np.full(n, np.nan); lam = np.full(n, np.nan); dfx = np.full(n, np.nan)
    for i, (a, l, d) in zip(idx, res):
        tfw[i], lam[i], dfx[i] = a, l, d
    rec = {'layout': layout, 'kind': kind, 'folder': str(folder),
           'stop_reason': session.get('control_stop_reason') or session.get('stop_reason'),
           't': t.tolist(), 'x': x.tolist(), 'y': y.tolist(), 'beta': b.tolist(),
           'valid': valid.tolist(), 'reason': reason, 'phase': phase,
           'tfw': tfw.tolist(), 'lam': lam.tolist(), 'd_fixed': dfx.tolist(),
           'i_cmd': np.nanmax(np.abs(cmd), axis=1).tolist(),
           'i_meas': [float(np.nanmax(np.abs(r))) if np.isfinite(r).any() else math.nan
                      for r in meas],
           'target': [[_num(r['target_x_m']), _num(r['target_y_m']), _num(r['target_heading_rad'])]
                      for r in rows]}
    auto = session.get('autonomy')
    if auto:
        ref = np.asarray(auto['samples'], float)  # x, y, beta, t
        rr = list(pool.map(_eval, [(p[0], p[1], p[2] % (2 * math.pi), layout) for p in ref],
                           chunksize=16))
        rec['reference'] = {'x': ref[:, 0].tolist(), 'y': ref[:, 1].tolist(),
                            'beta': (ref[:, 2] % (2 * math.pi)).tolist(), 't': ref[:, 3].tolist(),
                            'tfw': [a for a, _, _ in rr], 'lam': [l for _, l, _ in rr],
                            'd_fixed': [d for _, _, d in rr]}
        rec['goal'] = auto['nominal_goal']
    return rec


def _clean(o):
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    return o


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = {}
    with ProcessPoolExecutor() as pool:
        for key in TRIALS:
            data[key] = load_trial(key, pool)
            r = data[key]
            v = np.array(r['valid'])
            lam = np.array(r['lam'], float)[v]
            line = f'{key}: {v.sum()} valid, min lam {np.nanmin(lam):.2f}' if v.any() else key
            if 'reference' in r:
                ref = r['reference']; tf = np.array(ref['tfw']); lm = np.array(ref['lam'])
                line += (f' | ref {len(tf)}: in {tf.sum()}, out-but-feasible '
                         f'{((~tf) & (lm >= 1 - 1e-9)).sum()}, infeasible {(lm < 1 - 1e-9).sum()}')
            print(line, flush=True)
    (OUT / 'trials.json').write_text(json.dumps(_clean(data)))
    print('Wrote', OUT / 'trials.json')


if __name__ == '__main__':
    main()
