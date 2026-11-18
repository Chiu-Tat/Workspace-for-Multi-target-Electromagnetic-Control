# Actuation-requirement comparison

This directory compares three region-decision pipelines on the fixed ten-coil
Fig. 12 geometry.  Only the required symmetric field box in `(Bx, By)` changes.
The search domain is `[-0.05, 0.05]^2 m`, the paper plane is `z=-0.01 m`
(`z=0.075 m` in the coil-model coordinate), currents are limited to `+/-15 A`,
and the bisection tolerance is `0.002 m`.

Cases are reported as the positive half-widths `Bx/By` in mT:

- `60/60`: easy symmetric requirement;
- `80/80`: nominal Fig. 12 requirement;
- `100/100`: mixed-capacity symmetric requirement;
- `120/120`: beyond-capacity symmetric requirement;
- `120/40`: Bx-dominant anisotropic requirement;
- `40/120`: By-dominant anisotropic requirement.

Pipelines:

1. standard interval enclosure with the original Rohn/contractor tests;
2. affine enclosure with the same tests;
3. affine enclosure with the dependency-preserving recourse and dual tests.

Files:

- `actuation_requirement_spatial_grid.*`: full six-by-three spatial comparison;
- `actuation_requirement_summary.*`: unresolved area, model-in area, and box
  evaluation summaries;
- `actuation_requirement_metrics.json`: full protocol, terminal-box coordinates,
  metrics, certificate counts, and finite regression checks;
- `actuation_requirement_metrics.csv`: tidy source-data table.

The archived custom affine backend is not outward-rounded throughout, so these
are nominal model-based classifications.  The finite point checks are regression
tests only and do not replace the continuous region propositions.
