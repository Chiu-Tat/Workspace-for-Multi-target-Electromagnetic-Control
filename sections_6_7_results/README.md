# Sections 6 and 7 results

Run from the repository root with the [Python environment](../README.md#setup). The coil calibration summary used by study 6.1 is in [inputs](inputs/).

```bash
export MPLCONFIGDIR="$PWD/sections_6_7_results/.mplconfig"
python -m hybrid_region.test_hybrid_region
python -m analysis.sections_67.run --studies 6.1,6.2,6.3,6.4,7.1,7.2,7.3,7.4
```

Code: `analysis/sections_67/`. Shared coil-model origin offset is 0.085 m.
The hybrid-region library now accepts `active_coil_indices` by column removal.

Archived 1 mm / 2 mm ablation numbers in `revision_analysis/tight_region_benchmark*.json` are reused for Table 6.4. The resolution, task, and coil-subset studies were recomputed with `hybrid_region.certify_region`.
Dipole counts are reproduced from `dipole_error_comparison/results/`.


## Review updates (16 September 2026)

The AFW verification now calls the production `emns.workspace.fast.ModifiedHyperplaneShiftingMethod` and compares it with independent HiGHS feasibility solves. Solver failures remain unknown. The former hyperplane helper was not a test of that production routine; its verification counts are superseded by the edge-probe study.

The production routine needed explicit scalar extraction for NumPy 2.5 and integer combination indices for one-dimensional intrinsic sets. The helper handles affine-hull equalities for lower-dimensional AFWs. Active-coil selection preserves index order and rejects non-integer indices.

After running the selected studies, validate/export the paper tables with:

```bash
python -m unittest analysis.sections_67.test_review_regressions
python -m hybrid_region.test_hybrid_region
python -m analysis.sections_67.validate_exports
python -m analysis.sections_67.paper_tables
```

The final command writes the AFW verification table, resolution table and timing macros, and coil macros and table to [writing](writing/). Figures and statistics use the retained numeric results; AFW query records and region terminal boxes are now saved. `validation/run_*.json` records run-start source hashes and environment; `validation/review_source_manifest.json` records the delivered source state, including reporting/checking utilities added after the computation began.

Fresh resolution timings and historical ablation timings come from separate runs; compare performance within the matched ablation only. Studies 6.4 and 7.4 import historical data and do not claim to re-execute those computations.

The coil/task studies are single-location field benchmarks. The production-code verification and interaction-on/off figures use the selected `(Bx1, Bx2)` field projection.
