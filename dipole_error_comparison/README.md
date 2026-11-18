# Point-dipole/finite-body TFW comparison

This folder reproduces a matched, pointwise comparison for two prescribed
permanent magnets. Both runs use the unchanged Case-6 coil model, body-frame
task `[By,Bz,fy,fz] = [0,0.02,0,[-0.1,0.1]]`, `+-15 A` current limits,
planar domain, and grid.

Run from the repository root:

```bash
.venv/bin/python dipole_error_comparison/run_comparison.py
.venv/bin/python dipole_error_comparison/run_orientation_study.py
.venv/bin/python dipole_error_comparison/plot_worked_example.py
.venv/bin/python dipole_error_comparison/check_derivatives.py
.venv/bin/python dipole_error_comparison/check_reference_convergence.py
```

The first model inserts the existing point-dipole interaction offset into the
TFW containment test. The second inserts a distributed finite-body offset from
a cylindrical midpoint quadrature that includes both source and receiver
extent. Comparing the two task classifications directly preserves the signed
model correction and reveals both point-dipole-only and finite-body-only
samples. No empirical safety multiplier is used.

The cylinder dimensions in `config.json` are illustrative values inherited
from simulator drawings, not measured magnet metrology. The finite-body model
is a numerical comparison model, not experimental truth, and the grid is
pointwise. The outputs must therefore not be described as continuous or
hardware-certified TFWs.

The point-dipole limiting check exposed and this work corrected a coefficient
in `emns/workspace/fast.py`: `dBy/dy` requires `6*y*my/r^5`, not
`3*y*my/r^5`. Both comparison masks use the corrected map. A derivative check,
a shrinking-body check, and selected reported-versus-finer quadrature checks
provide numerical diagnostics but not a rigorous physical-error enclosure.

Outputs are written to `results/`: PDF/PNG figures, JSON metrics, the selected
safety-relevant boundary example, and compressed masks and discrepancies. The
orientation study adds separate matched 31x31 slices at headings 0, 45, 90,
and 135 degrees and separation-binned absolute force discrepancies. Their
union is not an all-heading feasibility result.

## Optional physical validation

No pair-interaction experiment is required to reproduce the numerical study.
If physical robustness is later claimed, `measurements_template.csv` and
`validate_measurements.py` provide a Case-6-specific input and validation path
for coils-off observations. Blank components are allowed, so force-only rows
are valid. Measurement standard deviations are retained but are not
automatically promoted to coverage bounds. The synthetic validator tests check
the input contract only; they are not physical evidence.
