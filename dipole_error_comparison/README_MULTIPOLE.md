# Compact multipole revision: reproducible Stage A

Date: 25 October 2026. This study uses illustrative uniform axial cylinders, not
measured specimens. Physical calibration and validation remain open.

## Run

From the workspace root, with the existing `.venv`:

```sh
.venv/bin/python dipole_error_comparison/test_multipole_stage_a.py
OPENBLAS_NUM_THREADS=1 .venv/bin/python dipole_error_comparison/run_multipole_stage_a.py
OPENBLAS_NUM_THREADS=1 .venv/bin/python dipole_error_comparison/run_multipole_box_pilot.py --xy 128 --heading 128
.venv/bin/python dipole_error_comparison/verify_stage_a_extended.py
OPENBLAS_NUM_THREADS=1 .venv/bin/python dipole_error_comparison/summarize_multipole_stage_a.py
```

The full four-heading sweep takes about 220 s on the current machine. The fine
3-D inner-set pilot takes about 193 s. Timing depends on hardware. The offline
Gauss reference uses a small C loop, compiled with the system `cc` into `/tmp`.
No additional Python package was installed. Operational multipole evaluations use
only finite derivative formulas; no volume integration is performed at runtime.
The summary script writes the paper figure and supplementary table in the sibling
manuscript project; pass `--paper-root` if it is stored elsewhere.

## Files and conventions

- `multipole_model.py`: corrected initial planar evaluator; retained for regression.
- `multipole_stage_a.py`: batch analytical L=1,3,5 model, energy-derived force and
  torque, averaged-field gradient, finite-receiver coil coupling, analytic tails.
- `reference_gauss.py`, `reference_kernel.c`: independent offline reference and
  conservative analytical quadrature remainder majorants.
- `results/multipole_stage_a/metrics.json`: per-heading comparisons, refinement,
  pointwise masks and reserve statistics. NPZ files preserve the raw arrays.
- `pooled_metrics.json`: field, gradient, force and torque errors by separation;
  sampled sensitivity to candidate minimum separations.
- `box_pilot.json` / `.npz`: whole-box sufficient inner-set pilot. Failed boxes are
  unresolved. This is not the original affine-hybrid method or its replacement.
- `stage_b/`: empty real-data template, acquisition protocol and weighted axial
  coefficient fitter. Synthetic recovery tests do not count as physical evidence.

The source and receiver are independently truncated through L (a rectangular
truncation). Q1=m, Q3=m(h²-3R²)/4, Q5=m(h⁴-10h²R²+10R⁴)/16.
Body origins are cylinder centres; magnetization follows the body z-axis. Case 1
axes lie in the laboratory xy plane. Reference and model vectors use SI units.

Two tasks are intentionally distinct:

1. **Fixed-G comparison:** original dipole coil map and averaged-field task
   `[By,Bz,Fy,Fz]`, to isolate interaction changes.
2. **Physical heading task:** finite-receiver coil map and
   `[Tz/m,Bz,Fy,Fz]`. For finite bodies, By=0 does not generally imply Tz=0.

For both tasks, target `[0,0.02,0,[-0.1,0.1]]` uses T,N units and current bounds
are ±15 A. Source and receiver moments are 0.145 and 0.12 A m². Source position is
(20,0) mm, heading 90°. The model domain requires separation >11.2970 mm.

## What the numbers support

There are 14,304 configurations (four full 61x61 grids, 3576 admissible each).
Fixed-G disagreements with the refined reference: 114,25,1 for L=1,3,5.
Physical-task disagreements: 116,22,0. L5 is a promising validation candidate.
It is not automatically accepted: below 20 mm, its sampled maximum force-component
error is still 100.26 mN near the convergence cutoff. Above 20 mm the sampled
maximum is 0.131 mN. Set the operating domain from an actual task error budget.

The analytical truncation bounds (`truncation_bounds`) sum exactly the terms that the
rectangular truncation omits, and take the smaller of a distributed-dipole and an
end-face-charge bound; the earlier total-degree majorant is kept as
`truncation_bounds_total_degree` for comparison only. Tail-reserved L5 counts rose from
3036-3106 to 3339-3395 of 3418-3492 reference-feasible positions per heading.
The analytical truncation bounds are independent of reference quadrature. They
apply to the stated prescribed uniform magnetization and symmetric geometries.
Observed quadrature refinement is an error estimate; a separate positive-rule
Taylor majorant bounds ideal integration error but can be very loose. Neither
includes floating-point roundoff. No physical residual rho has been measured.

Pointwise reserves cover interaction and finite-receiver coil truncation. The
3-D pilot additionally bounds pose variation across complete boxes, but uses
ordinary float64 and conservative coil/rotation bounds. It retains only 5.74% (5.34% with the superseded total-degree tails) of
the search volume. Do not label the remaining boxes physically infeasible or
replace the existing sharper figures with this pilot. Tighter enclosures and
outward-rounded evaluation remain work for a full continuous certificate.

## Manuscript integration

The new main-text subsection and figure are supplementary to the historical
midpoint-reference comparison; its 39 disagreements at 90° are not the new
refined-reference count (41). The two studies must not be silently mixed.
New text is green, replaced text is struck through, and prior dark-blue text is
preserved. The supplement includes harmonic normalization, general energy
operators, cylinder coefficients, explicit tail proof, quadrature remainder proof,
whole-box derivative bounds and the numerical tables. R1-M2 and AE-4 describe
completed numerical work and pending physical evidence separately.
