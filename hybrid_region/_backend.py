"""Expose the shared numerical backends used by the hybrid-region package.

This package deliberately does not fork the interval/affine arithmetic, the
magnetic field model, or the certificate implementations.  Those live in the
``emns`` package and remain the single source of truth.
"""

from __future__ import annotations

from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parent.parent

# Re-exported so the rest of the package has one backend import surface.
from emns.numerics.interval import interval
from emns.workspace import affine as WS_lib_Affine
from emns.workspace import interval_fast as WS_lib_Interval_fast
from emns.workspace.affine import Map_I2H_Affine
from emns.workspace.interval_fast import Map_I2H_Interval_Fast
from emns.workspace.checks import Out, RobustFeasible


# ---------------------------------------------------------------------------
# Coil-model synchronisation.  REQUIRED -- do not remove.
#
# ``WS_lib_Interval_fast`` and ``WS_lib_Affine`` ship different coil parameter
# tables: their coil z-coordinates differ by exactly 0.085 m, because the two
# modules were written against different coordinate origins.  Comparing an
# interval enclosure against an affine one without reconciling them compares
# two different physical systems.  The failure is silent rather than
# conservative: the unsynchronised interval cascade certifies the whole search
# region as infeasible instead of leaving it unresolved.
#
# ``Affine_10c`` performs this same assignment at module scope, which is why
# the archived benchmark is correct only because it imports that module.  It is
# repeated here so ``hybrid_region`` does not depend on that import side effect.
# ---------------------------------------------------------------------------
WS_lib_Interval_fast.params_list = WS_lib_Affine.params_list

COIL_MODEL_ORIGIN_OFFSET_M = 0.085

__all__ = [
    "CODE_ROOT",
    "COIL_MODEL_ORIGIN_OFFSET_M",
    "interval",
    "Map_I2H_Affine",
    "Map_I2H_Interval_Fast",
    "Out",
    "RobustFeasible",
]
