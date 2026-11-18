# Case Study 2 figure provenance

Maps, plans and simulation records updated 20 October 2026 for 3 mm N35 spheres.
The vessel networks retain approximately 100 mm y extent. Both main trunks
are manufactured as 8 mm outer / 4 mm inner diameter pipes with 2 mm walls
and open annular ends. Separate side-branch bores are not part of this CAD
request. Networks are moved apart along x or z to keep the outer pipes separate.
Simulation radius fields describe the 4 mm inner lumen; particles stay on
the trunk centrelines. Side branches remain geometric context.
All motion traces are simulations; no new hardware data.

Run from Workspace_eMNS-main:

```bash
.venv/bin/python -B -m revised_case_studies.cs2_vessel_microspheres.publication
.venv/bin/python -B -m revised_case_studies.cs2_vessel_microspheres.publication --render
```

The nine-panel figure uses branched_parallel.json and branched_crossing.json,
with the matching branched TFW files and corridor plans. Geometry panels (a)
and (d) pair shaded 3D and top (+z) views of the validated hollow STEP pipe assemblies, including
all 14 segments: red arterial and blue venous networks. CadQuery is required
to render these panels. An orthographic camera and equal physical scaling
preserve the 3D proportions without depth exaggeration. Each view has a
calibrated 10 mm scale bar. The 3D view includes an x/y/z orientation triad;
the top view has equal-scale x/y axes in millimetres, showing projected length
without perspective foreshortening. Full trunk arc lengths are reported
separately beneath each pair and are not inferred from the top projection.
Scene coordinates are relative to the coil-array centre. Surface shading and the
workspace/force maps are rasterized (450 dpi); labels and trajectory curves
remain vector PDF content.
Spheres are 3 mm N35; top-view markers are to scale, 3D markers and arrows are schematic.
Both spheres travel only on the main trunks; side-branch routing is not tested.
The earlier carotid-bifurcation task and unbranched results are not pooled.
Publication rendering reads the recomputed TFW files and corridor plans without rewriting them. They are
nominal floating-point box classifications, with zero Model-out area. The sampled
LP-infeasible points in unresolved boxes are crosses, never red region fills.
Stored region force frames are frozen at box centres with tangent refinement;
the figure does not claim rigorous varying-frame or outward-rounded certification.
Point and affine row consistency and sampled inside-box checks are in manifest.json.
Row C (panels g-i): pointwise force margin lambda, the largest scale of the +-F_max
tangential task with all four corners attainable (fields exact, transverse and current
bounds enforced), on the stored 65 x 65 grid and along every logged run; cached in
force_margin.json. lambda < 1 agrees with the stored pointwise LP at every grid point.
It is a diagnostic, not a region certificate.
allocator_validation.json compares the recorded motion command with an
unattainable full-task corner at one recorded direct-arm configuration (with
3 mm N35 spheres the recorded command there is itself LP-infeasible). The QP
keeps the hard field/side constraints in both cases and exposes the force residual.

New simulations enforce the map's 5 mT total field and transverse-force-component
bounds as hard QP constraints. Only tangential motion force has residual. There
is no unconstrained fallback. All four task corners and each individual command
are tested by LP at every logged configuration, using exact vessel frames.
The nonlinear aligned-moment plant includes mutual interaction at every step.
Hard field/side constraints are enforced at the 10 Hz allocation updates;
inter-update field, transverse-load, and alignment deviations are logged, not
claimed to be continuously constrained. Parameters and centreline confinement
remain illustrative. The plant slides each 3 mm N35 sphere on the wall with Coulomb
friction (mu = 0.2) under buoyant weight and magnetic side load. Transverse load is a model prediction, not a measurement.

The TFW-aware arm reuses each saved corridor mission. Its physical endpoints
are shared by a direct, quintic-retimed reference with the same total duration.
Both obey speed <= 3 mm/s and acceleration <= 1.5 mm/s^2; force commands are
bounded by the full task's 0.5 mN tangential limit (friction and weight feedforward included). No waypoint is silently
moved between comparison arms. Per-arm durations, endpoint errors, sample counts,
LP infeasibility, currents, loads, and reference-limit checks are saved in
simulation_records.json. The separate advance/hold demonstration uses the
side-by-side geometry near its starting configuration, not near the close pass.

Replace simulated actual paths/progress/diagnostics in panels (b),(c),(e)-(h)
with synchronized repeated hardware trials. Recompute feasibility at measured
poses. Keep numerical workspace classes, pointwise tests, and measured motion
separate. The 3D sphere markers are schematic. The geometry is engineered,
not patient-specific, and the parameter-area fractions are not anatomical areas.

Source hashes and all settings: manifest.json.
