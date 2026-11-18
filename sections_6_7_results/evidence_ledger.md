# Evidence ledger

| Study | Status | Type | Notes |
|---|---|---|---|
| 6.1 coil field | existing | archived model–measurement residuals | 477 samples, 117 locations/coil from `Modeling/10 coil data 3D/coil1_to_10_model_accuracy_summary.csv`. Mean 0.746 mT, pooled RMSE 1.055 mT, worst 9.152 mT. No held-out fitting/validation split is documented in the summary; probe/registration/gradient uncertainties are unavailable. |
| 6.2 AFW check | new | nominal numerical | Production `ModifiedHyperplaneShiftingMethod` vs independent HiGHS LP after the NumPy scalar/index compatibility fixes. Weak/strong cases: 501 queries each; six-coil case: 477. Zero disagreements and zero failures. All query records and residuals are exported. Earlier replacement-helper results are superseded. |
| 6.3 resolution | new | nominal classification | Affine/hybrid on anisotropic benchmark. Widths 6.25, 3.12, 1.56, 0.78 mm. Model-in 10.55→44.13%; unresolved 89.45→13.54%. |
| 6.4 ablation | archived | nominal classification | 1 mm unresolved 69.568 / 46.094 / 13.538%. Times 81.72 / 70.78 / 151.96 s. Backend not outward-rounded. |
| 7.1 AFW geometry | new + existing figure | nominal numerical | Full-set translation error 3e-17 T. Origin feasible with `c=0`, infeasible with strong-interaction `c`. Original AFW.pdf retained for MMFW/MWFW/MHFW. |
| 7.2 tasks | reproduced | nominal classification | 60/60 100% in; 80/80 98.07%; 100/100 13.77%; 120/120 0%; 120/40 40.55%; 40/120 69.48%. |
| 7.3 coils/currents | new | numerical capacity | Nested 8: coils 1–8, 9.55% in. Nested 6: 1–6, 1.49% in. Alternate 8: drop 1 and 4, 0.56% in. ±5 and ±10 A: 100% model-out. 121-point inclusion: 0 violations. Not hardware relocation. |
| 7.4 dipole reserve | existing | numerical sensitivity | 3495→3442 samples, 98.48% count / 98.43% area. Factor 1.25 illustrative. Simulator dimensions, not metrology. |
| Tracking table | existing | single-run hardware | Moved to Section 8. n=1 per archive. |
| Pair force/torque | missing | — | No coils-off pair measurements found. |
| Verified arithmetic | missing | — | Labels remain nominal. |

## Review on 16 September 2026

- Regenerated 6.1–6.3 and 7.1–7.3; the task/coil area fractions reproduce the earlier outputs. Timings in 6.3 are new single-run measurements and are generated into the manuscript from JSON.
- Exported terminal-box geometry, source hashes, software/environment records, and AFW query/current data.
- `validation/review_export_checks.json`: 1,008 finite pose checks, zero contradictions and zero unknown solves; area accounting agrees with every checked report.
- Coil/current nesting: 121 poses, zero inclusion violations and zero unknown solves.
- `validation/review_tests.txt`: focused regression tests and the hybrid-region suite.
- These checks are nominal finite regressions. They do not establish outward-rounded continuous certification or physical robustness.
- Sections 6.4 and 7.4 retain historical ablation and dipole sensitivity records; the present review does not describe copied records as fresh numerical reruns.
