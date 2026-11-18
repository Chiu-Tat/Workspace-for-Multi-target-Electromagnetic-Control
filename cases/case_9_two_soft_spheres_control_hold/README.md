# Case 9 — two soft-magnetic spheres, one control-held

This case computes a two-dimensional slice of the reduced four-dimensional
positional TFW for two freely translating, isotropic soft-magnetic spheres.
Both centres lie in the Case 6 plane, `z = 0.082 m`. Sphere 1 is held by force
control at `(0.02, 0)`; sphere 2 is scanned over `[-0.05, 0.05]^2 m`.

This is a deliberately restricted **fixed-local-field** study. The complete
field at both sphere centres is prescribed as

```text
B1 = B2 = (0, 0.02, 0) T.
```

At that operating point, each induced moment is fixed by the material law.
The force is consequently affine in the coil-generated gradient, allowing the
same affine-hybrid certificates as Case 8. If field magnitude is allowed to
vary as a command, force becomes quadratic in coil current and this solver is
not applicable.

## Soft-sphere model

The default illustrative parameters are

```text
held radius                5 mm
moving radius              5 mm
intrinsic susceptibility   5000
saturation magnetization   1.7e6 A/m
surface clearance          0
spatial resolution         1 mm
```

For a sphere, the demagnetization-corrected reversible model is

```text
chi_eff = 3 chi / (chi + 3)
M = min(chi_eff |B| / mu0, M_s)
m = (4 pi r^3 / 3) M.
```

At 20 mT the default moment of each sphere is approximately
`0.024985 A m^2`. These are material-model values, not fitted measurements.
Replace the defaults with data for the intended particles before drawing an
experimental conclusion. The model excludes hysteresis, remanence, eddy
currents, nonuniform magnetization and multipole effects near contact.

Mutual dipole fields and forces are included. Because the specified **total**
local fields determine the induced moments, the required coil field equals
the target field minus the mutual field. Mutual terms therefore form the
configuration-dependent pinned offset in the affine map.

## Task and exclusion

The output order is

```text
[B1x, B1y, B1z, F1x, F1y, B2x, B2y, B2z, F2x, F2y].
```

Defaults match the successful Option 2 protocol from Case 8:

```text
F1x, F1y in [-0.001, 0.001] N
F2x, F2y in [-0.005, 0.005] N
coil currents in [-15, 15] A
```

The four independent force components give 16 exact task-box vertices; no
direction sampling is used. Unlike Case 8, there is no permanent-moment
alignment-stability exclusion. The only default exclusion is physical overlap:

```text
centre distance <= held radius + moving radius + clearance.
```

Boundary-straddling leaf boxes remain undetermined rather than being entirely
excluded.

## Run

From the repository root:

```bash
python -m cases.case_9_two_soft_spheres_control_hold.workspace --smoke
python -m cases.case_9_two_soft_spheres_control_hold.workspace
python -B -m unittest tests.test_case9_two_soft_spheres_control_hold
```

Material and geometry can be changed explicitly:

```bash
python -m cases.case_9_two_soft_spheres_control_hold.workspace \
  --held-radius 0.003 --moving-radius 0.003 \
  --bias-field 0.02 \
  --susceptibility 5000 --saturation-magnetization 1.7e6 \
  --clearance 0.001 --eps 0.001
```

Results are saved to

```text
Results/results_case9_two_soft_spheres_control_hold.json
Results/results_case9_two_soft_spheres_control_hold.png
```

The JSON records all constitutive parameters and calculated operating moments.
Green and red boxes are one-sided inside/outside certificates. Gray means the
certificates were inconclusive, not that the point is infeasible.
