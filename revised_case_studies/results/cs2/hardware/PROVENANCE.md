# Case Study 2 hardware figure: provenance

Reproduce from `Workspace_eMNS-main`:

```bash
.venv/bin/python -B -m revised_case_studies.cs2_vessel_microspheres.hardware     # trials.json
.venv/bin/python -B -m revised_case_studies.cs2_vessel_microspheres.figure_hw    # cs2_hardware.{pdf,png}
```

| Element | Source |
|---|---|
| Trials | `ROS_CAN/src/case_study2_side_by_side/.../trial_20261028_204005_d2c3ef93`, `case_study2_over_under/.../trial_20261030_175851_211b5918`, `case_study2_over_under_manual/.../trial_20261105_184756_be724799` |
| Positions, arcs, targets, currents, pauses | `telemetry.csv`; times are logged host times (`elapsed_s`), not rescaled |
| References | `session.json` → `reference` (t, a1, a2) |
| Camera background | Per-pixel median of 15 `ui.avi` frames (camera panel of the UI), each the frame captured closest to the telemetry time via `video_timestamps.csv` |
| Image registration | Affine EM (x, y) mm → UI pixels, least squares on the tracker ring overlays (annulus matched filter), one robust refit. RMS 1.32 / 1.20 / 1.07 mm. Check: the fit is best with unscaled telemetry times (side-by-side 1.21 mm at x1.0 vs 5.45 mm at x0.5) |
| Centrelines | `publication.Geometry(variant).lines` (registered model; z identical to logged inferred z, x/y within 1.3 mm of camera poses) |
| TFW boxes, lambda = 1 contour | `results/cs2/tfw_branched_<variant>.json`, `results/cs2/publication/force_margin.json` |
| Force-margin maps (c, g, k) | `results/cs2/publication/force_margin.json` (65 x 65 grid, `publication.margin_data`); cache key equals the SHA-256 of the current TFW files; lambda >= 1 agrees with the pointwise full-task LP at all 4225 grid points per layout |
| Per-sample Model-in, lambda | `hardware.py`: box membership and `publication.force_margin` at each logged arc pair |
| Separation | 3-D distance of logged sphere centres (camera x, y; centreline-inferred z) |
| Manual row window | Only the segment in the edited video `cs2 over under failure.mov` (2x speed, 9.24 s): its frames were matched to `ui.avi` frames by normalized image correlation (r = 0.86-0.90); linear fit trial time = 76.72 s + 2.026 x edited time (RMS 0.25 s), window 76.72-95.45 s (`figure_hw.MANUAL_WINDOW_S`) |
| Passing | First sample where the logged y order of the two spheres changes sign |

Audit (5 Nov 2026): every number in the CS2 text was recomputed from these records and reproduces (area fractions, centreline distances, reference endpoints and maxima, buffers, field targets, current limits, sample counts, RMSE, goal errors, current peaks, passing and closest-approach times and separations, manual start, hold-interval drift 8.25 mm over 5.45-27.17 s, 18 pause events, 96.15 s stop, TFW exit 85.51 s, lambda < 1 at 94.12 s).

Schematic only: none. Sphere markers are drawn at logged positions (3 mm diameter to scale).
