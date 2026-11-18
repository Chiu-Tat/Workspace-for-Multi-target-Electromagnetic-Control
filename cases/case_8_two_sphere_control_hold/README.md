# Case 8 — two permanent-magnet spheres, one control-held

This case computes a two-dimensional slice of the reduced four-dimensional
positional TFW for two freely rotating permanent-magnet spheres. Both centers
lie in the Case 6 plane, `z = 0.082 m`. Sphere 1 is **control-held**, not
physically fixed, at `(0.02, 0)`; sphere 2 is scanned over
`[-0.05, 0.05]² m`.

The two identical spheres each have magnetic moment `0.12 A m²`.

The reduced model assumes both moments remain on a stable quasistatic branch
aligned with the complete prescribed net field

```text
B1 = B2 = (0, 0.02, 0) T.
```

The zero out-of-plane component is essential: planar position confinement does
not prevent a freely rotating permanent moment from tilting. Orientation
dynamics and closed-loop feedback stability still require separate validation.

## Task

The global-frame output order is

```text
[B1x, B1y, B1z, F1x, F1y, B2x, B2y, B2z, F2x, F2y].
```

Default force component ranges are

```text
F1x, F1y in [-0.02, 0.02] N   control-held feedback authority
F2x, F2y in [-0.10, 0.10] N   moving-sphere authority
```

These independent component bounds form a square; its moving-force corners
have magnitude `sqrt(2) * 0.10 N`. There are four varying force coordinates,
so the task box has 16 distinct vertices. No direction sampling is used.

Both mutual dipole fields and mutual forces are included. With the moment
directions fixed along global `+y`, they are current-independent and form the
pinned offset in

```text
b = G(p1, p2) I + c_mutual(p1, p2).
```

The ten coil currents are bounded to `[-15, 15] A`. The global field-gradient
implementation is local to this case and uses the corrected `6*y*m_y` term in
`dBy/dy`.

## Alignment and separation

The default calculation applies a conservative, worst-direction small-angle
stability filter. For the current moments and 20-mT field, the sufficient separation is
approximately 10.97 mm:

```text
r > [2 (mu0/4pi) sqrt(m1 m2) / B_hold]^(1/3).
```

`--min-separation` can impose a larger physical collision/clearance limit, but
cannot weaken this alignment filter. Boxes crossing the effective boundary are
subdivided before any mutual-field expression is evaluated. At the requested
resolution, boundary-straddling leaves remain **undetermined**, because part of
each rectangle is admissible. Only wholly filtered rectangles are excluded.
Exclusion by this sufficient condition does not prove physical instability at
every excluded position. Set `--min-separation` from the actual sphere radii and
clearance; the default zero does not account for their physical sizes.

## Calculation

The dependency-preserving affine right-inverse certificate tests inside boxes
using the 16 task vertices. An interval right-inverse is retained as a fallback.
The dual support-function certificate proves outside boxes. Failed certificates
produce subdivision or an undetermined leaf; they are never treated as proof of
infeasibility.

The project's custom affine/interval backend is useful for conservative region
analysis but is not a machine-verified outward-rounded interval package.
This case uses local product and inverse-power enclosures that retain nonlinear
remainders, including at the small distance scales in the dipole formulas.
It does not use the shared backend's truncated products or approximate square
root/reciprocal operations for those calculations.

From the repository root:

```bash
# Three representative point classifications.
python -m cases.case_8_two_sphere_control_hold.workspace --smoke

# Full default 2D slice.
python -m cases.case_8_two_sphere_control_hold.workspace

# Example with a larger physical separation and coarser preliminary run.
python -m cases.case_8_two_sphere_control_hold.workspace \
  --min-separation 0.015 --eps 0.01

# Short run with an exact box budget; unevaluated boxes are saved as pending.
python -m cases.case_8_two_sphere_control_hold.workspace \
  --processes 1 --max-iter 100

# Focused model and certificate tests.
python -B -m unittest tests.test_case8_two_sphere_control_hold
```

Outputs are written to:

```text
Results/results_case8_two_sphere_control_hold.json
Results/results_case8_two_sphere_control_hold.png
```

The JSON records the task protocol, effective exclusion radius, certificate
counts, completion flag, and separate inside/outside/undetermined/excluded/pending
rectangle lists. Partial plots show pending regions in yellow. `--max-iter 0`
saves the initial pending domain; `-1` runs to completion. A single-worker run
does not spawn subprocesses. Plots are saved without blocking; add `--show`
to open an interactive window.

## Optimized aligned-field reconstruction

`workspace_aligned_bz.py` allows a wider default `|Bz| <= 10 mT`. For every
position box it continuously optimizes one `(B1z, B2z)` pair, aligns each
permanent moment with its complete prescribed field, and uses the same pair for
all 16 force vertices. The planar force rows therefore include both gradient
components:

```text
Fx = my dBy/dx + mz dBz/dx
Fy = my dBy/dy + mz dBz/dy
```

Using one fixed aligned direction across the force box preserves affine
current dependence and vertex convexity. Each certified box stores its Bz
witness in the result JSON. This formulation assumes moment/field alignment
directly and therefore does not apply the older worst-case alignment-radius
filter. `--min-separation` is reserved for a physical centre-distance limit.

```bash
python -m cases.case_8_two_sphere_control_hold.workspace_aligned_bz

# Sequential stage 2: sphere 2 is control-held at (-0.02, 0), while the
# identical other sphere is scanned. Separate output names preserve stage 1.
python -m cases.case_8_two_sphere_control_hold.workspace_aligned_bz \
  --held-x -0.02 --held-y 0 \
  --output-json Results/results_case8_aligned_bz_stage2.json \
  --output-plot Results/results_case8_aligned_bz_stage2.png
```

Outputs are written to:

```text
Results/results_case8_aligned_bz.json
Results/results_case8_aligned_bz.png
```
