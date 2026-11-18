# Case Study 1 figure provenance

Updated on 17 October 2026. No new hardware data.

## Reproduction

From `Workspace_eMNS-main`:

```bash
# Sharp 3-D TFWs (about 12.5 h on 14 cores in total):
.venv/bin/python -B -m revised_case_studies.cs1_cylinder_interaction.run_tfw3d_fine --layout single
.venv/bin/python -B -m revised_case_studies.cs1_cylinder_interaction.run_tfw3d_fine --layout gate
.venv/bin/python -B -m revised_case_studies.cs1_cylinder_interaction.run_tfw3d_fine --layout triangle
# Planned and straight references (also used by the slide deck):
.venv/bin/python -B -m revised_case_studies.cs1_cylinder_interaction.presentation --only plans
# Analysis, simulations, macros and figure:
.venv/bin/python -B -m revised_case_studies.cs1_cylinder_interaction.publication
# Redraw from saved records:
.venv/bin/python -B -m revised_case_studies.cs1_cylinder_interaction.publication --render
```

## TFW data

Every TFW panel is the sharp 3-D (x, y, beta) TFW of `results/cs1/tfw3d_fine`, or a
section of it at one heading level: affine-hybrid cascade, corrected dBy/dy,
stopping widths 0.781 mm in x and y and 5.625 deg in beta.
A Model-in box holds for every heading in its beta interval. The 3-D runs have
no numerical cutout; boxes that contain a fixed-dipole centre stay unresolved.
Section shares divide by the full 100 x 100 mm square; volume shares by the full
(x, y, beta) box. The 3-D panels draw the complement of Model-in on a 64 x 64 x 64
display grid sampled at cell centres. Labels are nominal Model-in/Model-out/
unresolved from the floating-point backend, without outward rounding.
`manifest.json` records numerical consistency checks on sampled 3-D boxes.

## Paths, force, and simulation

The planned path is `presentation/trajectory_single_planned.json`: hybrid A* in
(x, y, beta) on the sharp TFW, 2 mm margin at each heading level and its
neighbours, reversing penalised. The straight crossing shares its endpoints at
heading 90 deg and 5 mm/s; the auxiliary passage is the crossing shifted to
x = 8 mm. Reference samples are tested by endpoint LPs at their own headings
and against the 3-D TFW. Panel (k) draws the paths over the 90-deg section only.
The force interval jointly enforces holding-field and zero-lateral-force tasks.
`simulation_records.json` and `auxiliary_weaker_task_crossing.json` retain
references, poses, currents, residuals, feasibility tests and abort records.
The simulation stops at 11.30 mm centre separation for
illustrative 9x5 mm moving and 11x5.5 mm fixed cylinders. No contact or
post-abort motion is fabricated. `passage_force_capacity.json` contains the
force-interval LPs of panel (m), checked against command and full-task LPs.

## Hardware replacement

Replace the simulated paths in (k) and the traces in (n) with synchronized
measurements; update the metrics and the caption together. The archived
`images/Case_6.pdf` remains untouched for the historical supplement.

## Final source hashes

- `revised_case_studies/cs1_cylinder_interaction/publication.py`: `f3f9b98d1452f29b372db8c3ffba5c1ffa9dec6c6eeb4a1210eb90232da437a8`
- `emns/workspace/affine.py`: `6d21518e6072e669126a60c10fa22865caad818059c6c9761fd5b600e3a21a2e`
- `emns/workspace/fast.py`: `ffbbe427c5c32cf1d0f4c0f1df79166dfe3d4cb054c8bd61cab826eec1b314a8`
- `cases/case_6_interaction_obstacle_avoidance/workspace_hybrid.py`: `19f0b289214296d84a8afb83ea72981a76267941d878b72fb1efad1c05614542`
