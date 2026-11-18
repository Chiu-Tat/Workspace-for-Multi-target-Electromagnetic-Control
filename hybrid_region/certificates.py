"""Certificate layer of the hybrid method.

The mathematics lives in ``emns.certificates.tight_region``
and is *not* duplicated here.  This module re-exports the parts the hybrid
cascade uses so that callers have one import surface, and documents which test
contributes what.

Three tests are added on top of the original Rohn/contractor cascade:

``affine_constructive_inside_certificate``
    Dependency-preserving constructive right inverse.  Retains the common
    affine noise symbols while bounding the current correction, so it is no
    worse than collapsing each matrix entry to an independent interval.
``constructive_inside_certificate``
    The same recourse argument driven by plain interval bounds.  Kept as a
    fallback for enclosures that carry no affine forms.
``dual_outside_certificate``
    Fixed-normal support-function separation, proving a task vertex is
    unreachable for every configuration in the box.

Every test is one-sided: ``certified=True`` is a proof, ``certified=False``
only means the test was inconclusive.  The guarantees hold when the supplied
``G_lower``/``G_upper`` or affine forms are genuine inclusion bounds.
"""

from __future__ import annotations

from emns.certificates.tight_region import (
    InsideCertificate,
    OutsideCertificate,
    affine_constructive_inside_certificate,
    box_vertices,
    classify_interval_system,
    constructive_inside_certificate,
    dual_outside_certificate,
    interval_object_bounds,
)

__all__ = [
    "InsideCertificate",
    "OutsideCertificate",
    "affine_constructive_inside_certificate",
    "box_vertices",
    "classify_interval_system",
    "constructive_inside_certificate",
    "dual_outside_certificate",
    "interval_object_bounds",
]
