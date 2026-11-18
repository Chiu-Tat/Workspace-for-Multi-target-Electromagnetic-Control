# Measured cylinder-force validation and region comparison

The inputs contain 54 coil-off force/torque readings at 15 positions. The sensor magnet is 4 mm in diameter and 11 mm long, with its moment along +y. The source magnet is 4 mm in diameter and 10 mm long. Force readings in `data_original.csv` are in gram-force and the torque readings are in mN·mm. The torque conversion follows the verified units used in the revised analysis.

## Fit the force models

Run from the repository root with the [Python environment](../../README.md#setup) and a C compiler available as `cc`.

```sh
python -m dipole_error_comparison.force_test.analyze
python -m dipole_error_comparison.force_test.plot_multipole
```

The fit uses one moment product and two constant force biases. The results include leave-one-pose-out and position-holdout errors. The fitted dipole and L=5 models have pose-holdout force-vector RMS errors of 9.02 mN and 2.06 mN. The archived [force-fit figure](multipole_validation.pdf) accompanies these results. The solver keeps geometry-dependent predictions in `kernels.npz`. Remove this file before changing the geometry, pose list, or calculation of the prediction kernels.

## Reproduce the region comparison

The [region figure](region/force_test_validation.pdf) can be redrawn from the included arrays and measured-force check.

```sh
python -m dipole_error_comparison.force_test.region.plot_regions
```

To regenerate the arrays, run the following commands. The full search uses six workers and can take several minutes. Use a fresh working copy to retain the bundled results.

```sh
python -m dipole_error_comparison.force_test.region.run_regions
BAND_CENTER_DEG=90 RUN_TAG=_b90 python -m dipole_error_comparison.force_test.region.run_regions
RUN_TAG=_b90 python -m dipole_error_comparison.force_test.region.run_dipole_calibrated
python -m dipole_error_comparison.force_test.region.verify_measured
python -m dipole_error_comparison.force_test.region.validate_regions
python -m dipole_error_comparison.force_test.region.check_classifications
python -m dipole_error_comparison.force_test.region.plot_regions
```

The default heading range is 45.5°–46.5°. The 89.5°–90.5° run supports the comparison at matched moment. The classifier subdivides the planar domain to 0.125 mm where needed. Array columns are x low/high, y low/high, heading low/high, minimum and maximum separation in mm, force reserve in N, and the nominal, multipole, and reserved classes. Class values are 1 (inside), −1 (outside), 0 (unresolved), and 2 (outside the model's evaluation domain).

The local enclosures use analytical Taylor remainders in float64. The empirical residual reserve applies within the measured separation range. The reference-integration refinement and sampled checks characterize numerical agreement.
