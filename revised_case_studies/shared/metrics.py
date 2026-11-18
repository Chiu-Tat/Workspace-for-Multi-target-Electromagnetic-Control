"""Tracking-error statistics reported in the revised manuscript table.

Same definitions as ``revision_analysis/quantitative_validation.py``:
RMSE, MAE and the 95th-percentile absolute error, with angles wrapped.
"""

from __future__ import annotations

import numpy as np


def wrap_angle(a):
    return (np.asarray(a, float) + np.pi) % (2 * np.pi) - np.pi


def error_stats(err):
    e = np.abs(np.asarray(err, float))
    if e.size == 0:
        return {'rmse': float('nan'), 'mae': float('nan'), 'p95': float('nan')}
    return {'rmse': float(np.sqrt(np.mean(e ** 2))),
            'mae': float(np.mean(e)),
            'p95': float(np.percentile(e, 95))}


def position_stats(actual_xy, ref_xy):
    d = np.linalg.norm(np.asarray(actual_xy) - np.asarray(ref_xy), axis=1)
    return error_stats(d)


def angle_stats(actual, ref):
    return error_stats(wrap_angle(np.asarray(actual) - np.asarray(ref)))
