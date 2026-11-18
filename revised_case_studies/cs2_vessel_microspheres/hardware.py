"""Case Study 2 hardware records: per-sample TFW membership, force margin, separation, camera registration.

Run from the repository root:
  .venv/bin/python -B -m revised_case_studies.cs2_vessel_microspheres.hardware

Reads the archived ROS_CAN trials (telemetry.csv, session.json, ui.avi, video_timestamps.csv) and
writes results/cs2/hardware/trials.json.  All times are the logged host times (telemetry elapsed_s).

Per valid sample (armed, tracking valid, pose fresh):
  s          centreline parameters (s1 on RA, s2 on RV) of the logged arc positions
  model_in   membership in a nominal TFW-in box of the joint TFW
  lam        nominal full-task force margin (publication.force_margin): lam >= 1 iff the full task holds
  sep_mm     3-D distance between the two sphere centres (camera x, y; centreline-inferred z)
Camera registration: least-squares affine map from EM (x, y) to UI-video pixels, fitted to the tracker's
own ring overlays, which the UI draws at the logged sphere positions; the RMS residual is stored.
"""
from __future__ import annotations

import csv
import io
import json
import os
import math
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from revised_case_studies.cs2_vessel_microspheres import publication as P

ROS = Path(os.environ.get('ROS_CAN_SRC', 'ROS_CAN/src'))
TRIALS = {
    'side_by_side': ('side_by_side', ROS / 'case_study2_side_by_side/results/trial_20261028_204005_d2c3ef93', 'guided'),
    'over_under': ('over_under', ROS / 'case_study2_over_under/results/trial_20261030_175851_211b5918', 'guided'),
    'manual': ('over_under', ROS / 'case_study2_over_under_manual/results/trial_20261105_184756_be724799', 'manual'),
}
OUT = P.SAVED / 'hardware'
_G = {}


def geometry(variant):
    if variant not in _G:
        _G[variant] = P.Geometry(variant)
    return _G[variant]


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return math.nan


def _flag(v):
    return str(v).strip().lower() in ('1', 'true', 'yes')


def model_in(g, s):
    b = np.asarray(g.map['inside'])
    return bool(np.any((b[:, 0] <= s[0]) & (s[0] <= b[:, 1]) & (b[:, 2] <= s[1]) & (s[1] <= b[:, 3])))


def _eval(args):
    variant, a = args
    g = geometry(variant)
    s = g.s(np.asarray(a))
    return s.tolist(), model_in(g, s), P.force_margin(*g.rows(np.asarray(a))[2:])


# ------------------------------------------------------------------ camera registration
_VT = {}


def video_time(folder, t):
    """Video time of the captured UI frame closest to telemetry time t (video_timestamps.csv)."""
    folder = str(folder)
    if folder not in _VT:
        rows = [q for q in csv.DictReader(open(Path(folder) / 'video_timestamps.csv')) if q['duplicated'] == 'False']
        _VT[folder] = (np.array([_f(q['capture_elapsed_s']) for q in rows]),
                       np.array([_f(q['video_elapsed_s']) for q in rows]))
    cap, vid = _VT[folder]
    return float(vid[np.argmin(np.abs(cap - t))])


def ui_frame(folder, t):
    """UI-video frame captured closest to telemetry time t."""
    t = video_time(folder, t)
    out = subprocess.run(['ffmpeg', '-v', 'error', '-ss', f'{t:.2f}', '-i', str(Path(folder) / 'ui.avi'),
                          '-frames:v', '1', '-f', 'image2pipe', '-vcodec', 'png', '-'],
                         capture_output=True, check=True).stdout
    from PIL import Image
    return np.asarray(Image.open(io.BytesIO(out)).convert('RGB')).astype(float) / 255.0


def panel_box(img):
    """Camera panel of the UI: longest run of non-background pixels (row/column scans)."""
    nb = np.abs(img - 239 / 255).sum(2) > 12 / 255

    def runs(m):
        idx = np.flatnonzero(m)
        sp = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
        return [(int(r[0]), int(r[-1])) for r in sp if len(r)]
    r0, r1 = max(runs(nb[:900, 320]), key=lambda q: q[1] - q[0])
    c0, c1 = max(runs(nb[(r0 + r1) // 2, :700]), key=lambda q: q[1] - q[0])
    return r0, r1, c0, c1


def ring_centres(crop):
    """Centres (u, v) px of the UI's green and red tracking rings, or None.

    The rings are drawn at the logged sphere positions; long direction arrows start at them.  A ring is
    located as the maximum of the colour mask filtered with an annulus kernel, which favours the ring
    over the thin arrow.
    """
    from scipy.signal import fftconvolve
    r, g, b = crop[..., 0], crop[..., 1], crop[..., 2]
    masks = ((g > r + .18) & (g > b + .05), (r > g + .25) & (r > b + .2))
    yy, xx = np.mgrid[-10:11, -10:11]
    rad = np.hypot(xx, yy)
    kernel = ((rad >= 4) & (rad <= 9)).astype(float)
    out = []
    for m in masks:
        if m.sum() < 8:
            out.append(None); continue
        score = fftconvolve(m.astype(float), kernel, mode='same')
        v, u = np.unravel_index(np.argmax(score), score.shape)
        out.append((float(u), float(v)) if score[v, u] > .35 * kernel.sum() else None)
    return out


def register(folder, t, valid, xy_g, xy_r, n=70):
    idx = np.flatnonzero(valid)
    picks = idx[np.linspace(0, len(idx) - 1, n).astype(int)]
    box = panel_box(ui_frame(folder, t[picks[len(picks) // 2]]))
    r0, r1, c0, c1 = box
    src, dst = [], []
    for i in picks:
        crop = ui_frame(folder, t[i])[r0:r1 + 1, c0:c1 + 1]
        for c, xy in zip(ring_centres(crop), (xy_g[i], xy_r[i])):
            if c is not None and np.all(np.isfinite(xy)):
                src.append([xy[0] * 1e3, xy[1] * 1e3, 1.0]); dst.append(c)
    src, dst = np.array(src), np.array(dst)
    A, *_ = np.linalg.lstsq(src, dst, rcond=None)          # (3, 2): [x, y, 1] mm -> (u, v) px
    res = np.linalg.norm(src @ A - dst, axis=1)
    keep = res < 3 * np.median(res) + 1                   # one robust refit (ring partly hidden)
    A, *_ = np.linalg.lstsq(src[keep], dst[keep], rcond=None)
    res = np.linalg.norm(src[keep] @ A - dst[keep], axis=1)
    px_per_mm = float(np.sqrt(abs(np.linalg.det(A[:2]))))
    return {'box_rows_cols': [r0, r1, c0, c1], 'affine_mm_to_px': A.tolist(), 'points': int(keep.sum()),
            'rms_px': float(np.sqrt(np.mean(res ** 2))), 'rms_mm': float(np.sqrt(np.mean(res ** 2)) / px_per_mm)}


# ------------------------------------------------------------------ trials
def load_trial(key, pool):
    variant, folder, kind = TRIALS[key]
    rows = list(csv.DictReader(open(folder / 'telemetry.csv')))
    sess = json.loads((folder / 'session.json').read_text())
    t = np.array([_f(r['elapsed_s']) for r in rows])
    valid = np.array([_flag(r['armed']) and _flag(r['tracking_valid']) and _flag(r['pose_fresh']) for r in rows])
    reason = [r['reason'] for r in rows]
    follow = np.array([x.startswith('Follow reference') for x in reason])
    xyz = {c: np.array([[_f(r[f'{c}_x_m']), _f(r[f'{c}_y_m']), _f(r[f'{c}_z_inferred_m'])] for r in rows])
           for c in ('green', 'red')}
    arc = np.array([[_f(r['green_arc_m']), _f(r['red_arc_m'])] for r in rows])
    tgt = np.array([[_f(r['green_target_arc_m']), _f(r['red_target_arc_m'])] for r in rows])
    cmd = np.array([[_f(r[f'coil_{k:02d}_command_A']) for k in range(1, 11)] for r in rows])
    meas = np.array([[_f(r[f'coil_{k:02d}_measured_A']) for k in range(1, 11)] for r in rows])
    valid &= np.all(np.isfinite(arc), axis=1)
    sel = np.flatnonzero(valid)
    res = list(pool.map(_eval, [(variant, arc[i]) for i in sel], chunksize=8))
    n = len(rows)
    s = np.full((n, 2), np.nan); mi = np.zeros(n, bool); lam = np.full(n, np.nan)
    for i, (si, m, l) in zip(sel, res):
        s[i], mi[i], lam[i] = si, m, l
    sep = np.linalg.norm(xyz['green'] - xyz['red'], axis=1) * 1e3
    rec = {'variant': variant, 'kind': kind, 'folder': str(folder), 'stop_reason': sess.get('stop_reason'),
           't': t.tolist(), 'valid': valid.tolist(), 'follow': follow.tolist(), 'reason': reason,
           'green_xyz': xyz['green'].tolist(), 'red_xyz': xyz['red'].tolist(), 'arc': arc.tolist(),
           'target_arc': tgt.tolist(), 's': s.tolist(), 'model_in': mi.tolist(), 'lam': lam.tolist(),
           'sep_mm': sep.tolist(),
           'i_cmd': np.nanmax(np.abs(cmd), axis=1).tolist(),
           'i_meas': [float(np.nanmax(np.abs(m))) if np.isfinite(m).any() else math.nan for m in meas]}
    ref = np.asarray(sess['reference'], float) if sess.get('reference') else None
    if ref is not None and kind == 'guided':
        g = geometry(variant)
        rres = list(pool.map(_eval, [(variant, a) for a in ref[:, 1:]], chunksize=8))
        rec['reference'] = {'t': ref[:, 0].tolist(), 'arc': ref[:, 1:].tolist(),
                            's': [r[0] for r in rres], 'model_in': [r[1] for r in rres],
                            'lam': [r[2] for r in rres]}
    rec['registration'] = register(folder, t, valid, xyz['green'], xyz['red'])
    vt = list(csv.DictReader(open(folder / 'video_timestamps.csv')))
    rec['video'] = {'video_s': [_f(q['video_elapsed_s']) for q in vt], 'capture_s': [_f(q['capture_elapsed_s']) for q in vt]}
    return rec


def _clean(o):
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating,)):
        return _clean(float(o))
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = {}
    with ProcessPoolExecutor(max_workers=8) as pool:
        for key in TRIALS:
            r = data[key] = load_trial(key, pool)
            v = np.array(r['valid']); mi = np.array(r['model_in']); lam = np.array(r['lam'], float)
            print(f"{key}: {v.sum()} valid, TFW-in {mi[v].sum()}, min lam {np.nanmin(lam[v]):.2f}, "
                  f"min sep {np.nanmin(np.array(r['sep_mm'])[v]):.2f} mm, registration rms "
                  f"{r['registration']['rms_mm']:.2f} mm ({r['registration']['points']} pts)", flush=True)
    (OUT / 'trials.json').write_text(json.dumps(_clean(data)))
    print('Wrote', OUT / 'trials.json')


if __name__ == '__main__':
    main()
