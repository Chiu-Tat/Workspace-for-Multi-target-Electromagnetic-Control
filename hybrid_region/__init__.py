"""Dependency-preserving certified-region checking for electromagnetic actuation.

This package packages the hybrid method evaluated in ``revision_analysis`` as a
reusable library.  It answers, for a box ``Q`` of microrobot configurations,
whether *every* configuration in ``Q`` admits an admissible coil current for
*every* requested field in the task box:

    h = c(q) + G(q) i,   q in Q,   i in [i_lower, i_upper].

The method layers two dependency-preserving certificates on top of the original
Rohn/contractor cascade: a constructive right-inverse recourse test that
retains shared affine noise symbols, and a fixed-normal dual separation test.

Typical use::

    from hybrid_region import RegionProblem, certify_region

    problem = RegionProblem(
        initial_box=((-0.05, 0.05), (-0.05, 0.05)),
        z_offset=0.075,
        task_box=((-0.12, 0.12), (-0.04, 0.04)),
        current_bounds=(-15.0, 15.0),
        n_coils=10,
    )
    report = certify_region(problem, mode="affine_hybrid", eps=0.002)
    print(report.unresolved_fraction, report.certificate_counts)

The arithmetic backends, the magnetic field model, and the certificate proofs
are imported from the project root rather than duplicated here; see
``_backend.py``.  All certificates are one-sided, and their guarantees require
inclusion-preserving enclosures.
"""

from __future__ import annotations

from .ablation import run_ablation
from .boxes import bisect_box, box_area, box_bounds, box_vertices, box_widths, make_box
from .cascade import MODES, BoxVerdict, classify_box
from .certificates import (
    InsideCertificate,
    OutsideCertificate,
    affine_constructive_inside_certificate,
    constructive_inside_certificate,
    dual_outside_certificate,
    interval_object_bounds,
)
from .model import RegionProblem, affine_matrix, interval_matrix
from .search import RegionReport, certify_region

__version__ = "0.1.0"

__all__ = [
    # problem definition and search
    "RegionProblem",
    "RegionReport",
    "certify_region",
    "run_ablation",
    # cascade
    "MODES",
    "BoxVerdict",
    "classify_box",
    # certificates
    "InsideCertificate",
    "OutsideCertificate",
    "affine_constructive_inside_certificate",
    "constructive_inside_certificate",
    "dual_outside_certificate",
    "interval_object_bounds",
    # model
    "interval_matrix",
    "affine_matrix",
    # geometry
    "make_box",
    "bisect_box",
    "box_area",
    "box_bounds",
    "box_widths",
    "box_vertices",
]
