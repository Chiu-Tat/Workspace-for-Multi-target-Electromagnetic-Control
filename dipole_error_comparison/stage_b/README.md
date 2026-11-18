# Measurements needed to finish physical validation

No measurements have been generated or inferred. The CSV is an empty acquisition template.

1. Record specimen IDs, measured length/diameter and uncertainty, magnetization axis,
   origin and sensor-to-body transform, probe calibration, temperature and repeat IDs.
2. Measure three-axis fields outside the bounding sphere, at multiple radii and
   non-collinear directions. Allocate complete shells/runs to train, validation and
   test before fitting. Do not randomly split neighboring readings from one run.
3. Supply calibrated standard uncertainties (tesla). The fit uses column-scaled
   weighted least squares, includes a constant background field, selects L on the
   validation set, and reports the untouched test residuals. The covariance covers
   specified independent sensor noise only; separately assess pose and systematic
   errors. Do not interpret fit residuals as a uniform physical error envelope.
4. Run from the workspace root:
   `.venv/bin/python dipole_error_comparison/stage_b/fit_axial_field.py measurements.csv --bounding-radius-m 0.00515 --output fit.json`
   Replace radius with actual metrology; the example is the illustrative small magnet.
5. Compare fitted Q1/Q3/Q5 against the measured geometry formulas. An axial fit is
   inappropriate if the specimen has material transverse/non-axisymmetric terms;
   expand the coefficient basis rather than hiding them in a noise estimate.
6. With coils off, measure force at closest intended separation and several relative
   headings, plus torque if the manuscript claims heading equilibrium. Use held-out
   poses. Record load-cell calibration, offsets, gravity, fixture friction and repeats.
7. Declare acceptable field/wrench residuals from the task budget before accepting a
   model/order/domain. The numerical sweep reports errors; it cannot set this physical
   tolerance or certify an unmeasured specimen.

A uniform additive residual rho needs justified coverage over the intended domain,
not merely a maximum over these measurements. Keep coil calibration error separate.
