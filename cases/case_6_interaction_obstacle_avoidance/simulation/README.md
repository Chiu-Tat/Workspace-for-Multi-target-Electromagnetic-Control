# Case 6 — interactive driving simulation

Drive the free magnet around with **W/S/A/D** while the certified feasible
workspace *for its current heading* is drawn underneath it, so you can see
where the magnet is allowed to go.

```bash
python -m cases.case_6_interaction_obstacle_avoidance.simulation.interactive
```

Click the plot window once so it takes keyboard focus, then drive.

| file | what it is |
| --- | --- |
| `interactive.py` | the interactive simulator (run this) |
| `model.py` | physics, current allocation, workspace slicing — importable on its own |

The live pointwise model reuses `emns.workspace.fast`, while the default
certified set is loaded from the affine-hybrid workspace result; nothing is
re-derived by the simulator.

## Controls

| key | action |
| --- | --- |
| `W` / `S` | thrust forward / backward along the heading |
| `A` / `D` | turn left / right |
| `SPACE` | all stop |
| `T` | switch the blocking rule: certified ↔ exact |
| `R` | draw the exact feasible region at the current heading (grid sweep, a few seconds) |
| `H` | show / hide that overlay |
| `C` | clear the trail |
| `X` | reset the pose |
| `ESC` | quit |

## Why W/S/A/D map onto this problem

The scene is the one `cases.case_6_interaction_obstacle_avoidance.workspace` analyses: a **fixed** magnet at
(0.02, 0, 0.082) m with heading π/2 and m = 0.145 A·m², a **free** magnet in the
same plane with m = 0.12 A·m² and heading β, and ten coils limited to ±15 A.

With α = π/2 the free magnet's body frame is

```
body z = ( cos β,  sin β, 0)   along the dipole  = the heading
body y = (-sin β,  cos β, 0)   in-plane, lateral
body x = ( 0,      0,    -1)   out of plane
```

so the four constrained rows of the actuation matrix read

| row | value | meaning |
| --- | --- | --- |
| `By` | 0 | no in-plane torque → the heading β is in equilibrium |
| `Bz` | 0.02 T | holding field along the dipole |
| `fy` | 0 | no lateral force |
| `fz` | ±0.1 N | **thrust along the heading** |

The certified workspace is exactly the set of (x, y, β) where all ±0.1 N of
thrust is deliverable while the other three are held. So `fz` is the W/S axis
and β is the A/D axis — the controls are the task vector, not an overlay on it.

## What happens each frame

1. W/S set the commanded thrust `fz = ±0.1 N`.
2. The body-frame task `[By, Bz, fy, fz] = [0, 0.02, 0, fz]` is allocated to the
   ten coils: bounded least squares on `A[:, :10] I = b − A[:, 10]`, refined to
   the exact minimum-norm solution whenever that respects ±15 A. The last column
   of `A` is the fixed magnet's dipole contribution, at its pinned "current" 1 —
   matching `create_I_intervals(11, custom_bounds={10: (1, 1)})` in
   `cases.case_6_interaction_obstacle_avoidance.workspace`.
3. The field and force those currents actually produce are evaluated at the
   magnet, coils plus fixed magnet, via the world-frame matrix.
4. The magnet moves under a drag-dominated law `v = F_in-plane / c`, with
   `c = 5 N·s/m` — 20 mm/s at full thrust.

Everything on the panel is read back from the solved currents, not from the
command: the coil bar chart, the achieved-vs-commanded task, and B and F at the
magnet. Inside the workspace the relative error sits at ~1e-15.

Because `By = 0` and `fy = 0` are enforced, the in-plane **B** and **F** are
always parallel to the heading, so the two arrows are drawn offset to either
side to keep both readable.

## The two feasible sets

`T` switches which one blocks motion, and the gap between them is worth seeing.

- **Certified** — the affine-hybrid 3-D TFW from
  `Results/results_case6_3D_hybrid.json`, sliced at the current β. Guaranteed
  but conservative: an inner approximation, quantised to 3.125 mm in x/y and
  5.625° in β. Use `--tfw` to load a different result explicitly.
- **Exact** — a pointwise feasibility test at the true (x, y, β), run live
  (~0.7 ms). Since the reachable task set is convex and the required set is the
  segment `fz ∈ [−0.1, 0.1]`, testing the two extreme thrusts is exact.
  Typically ~93% of the domain.

Every certified pose passes the exact test, as it must. The reverse can fail,
because the hybrid run finds **zero** `out_of_workspace` boxes — everything not
certified is *undetermined*, not proven infeasible. So the blue region is not a
correction to the green one; it is the pointwise result where the certificate
can conservatively under-report.

Two consequences are visible immediately:

- The **hole around the fixed magnet** is real infeasibility, not conservatism —
  its dipole field dominates there and cannot be cancelled.
- The hybrid certificates reduce the conservative gap substantially compared
  with the original TFW, but do not eliminate the undetermined region.

## Notes

- The magnet starts at **(0, −0.05) m** heading +y, which is certified feasible
  with all 64 sampled headings available in the hybrid TFW. `X` resets to it.
  Change it with `--start X Y`, or edit `START_XY` / `START_BETA` in
  `interactive.py`. A start outside the certified set would leave the magnet
  unable to move under the certified blocking rule, so that case warns and
  falls back.
- The ring around the magnet shows certified-feasible headings from the present
  position; the telemetry gives the count out of 64 sampled headings.
- `--grid N` sets the resolution of the `R` sweep (default 61, ~4 s). `--tfw`
  points at a different results JSON; `--raster` sets the slice raster.
- Tuning constants — `DRAG_C`, `TURN_RATE`, `FPS`, arrow scales — are at the top
  of `interactive.py`.
- `python -m cases.case_6_interaction_obstacle_avoidance.simulation.model` runs a self-check of the model alone,
  no GUI.
