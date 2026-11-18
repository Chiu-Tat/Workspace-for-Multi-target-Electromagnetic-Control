# Case-study commands

Run every command below from the repository root after activating `.venv`.

## Case 1 - Rotor rotation

```bash
python -m cases.case_1_rotor_rotation.workspace
python -m cases.case_1_rotor_rotation.workspace_fast
python -m cases.case_1_rotor_rotation.workspace_interval
python -m cases.case_1_rotor_rotation.workspace_3d
python -m cases.case_1_rotor_rotation.point_discretization
python -m cases.case_1_rotor_rotation.planner_3d
```

## Case 2 - Magnet translation

```bash
python -m cases.case_2_magnet_translation.workspace_vertical
python -m cases.case_2_magnet_translation.workspace_horizontal
python -m cases.case_2_magnet_translation.workspace_3d
python -m cases.case_2_magnet_translation.planner_2d
python -m cases.case_2_magnet_translation.planner_3d
python -m cases.case_2_magnet_translation.planner_3d_modified
```

## Case 3 - Magnets on circular paths

```bash
python -m cases.case_3_circular_path_magnets.workspace
python -m cases.case_3_circular_path_magnets.planner
```

## Case 4 - Three-DOF manipulator

```bash
python -m cases.case_4_three_dof_manipulator.workspace
python -m cases.case_4_three_dof_manipulator.kinematics
python -m cases.case_4_three_dof_manipulator.planner
python -m cases.case_4_three_dof_manipulator.plot_end_effector
```

## Case 5 - Two-DOF manipulator

```bash
python -m cases.case_5_two_dof_manipulator.workspace
python -m cases.case_5_two_dof_manipulator.planner
python -m cases.case_5_two_dof_manipulator.plot_end_effector
```

## Case 6 - Interaction-based obstacle avoidance

```bash
python -m cases.case_6_interaction_obstacle_avoidance.workspace
python -m cases.case_6_interaction_obstacle_avoidance.workspace_hybrid
python -m cases.case_6_interaction_obstacle_avoidance.workspace_hybrid --layout gate
python -m cases.case_6_interaction_obstacle_avoidance.workspace_hybrid --layout triangle
python -m cases.case_6_interaction_obstacle_avoidance.workspace_2d
python -m cases.case_6_interaction_obstacle_avoidance.planner
python -m cases.case_6_interaction_obstacle_avoidance.planner_failure_case
python -m cases.case_6_interaction_obstacle_avoidance.plot_results_2d
python -m cases.case_6_interaction_obstacle_avoidance.simulation.model
python -m cases.case_6_interaction_obstacle_avoidance.simulation.interactive
```

## Case 7 - Two microspheres in branching or separate nearby vessels

Geometry and theory only; no control simulation. See
[`case_7_vessel_microspheres/README.md`](case_7_vessel_microspheres/README.md)
for the interactive vessel preview and control formulation.

```bash
python -m cases.case_7_vessel_microspheres.build_preview
python -m cases.case_7_vessel_microspheres.build_variants
python -m cases.case_7_vessel_microspheres.build_networks
```

Preview regeneration additionally requires `scikit-image`.

## Case 8 - Two permanent-magnet spheres, one control-held

Reduced positional TFW slice with sphere 1 control-held at `(0.02, 0)` and
sphere 2 moving over the Case 6 plane. The model prescribes the full 3-D net
field at both spheres and includes mutual field/force offsets and a stable-
alignment separation filter. See
[`case_8_two_sphere_control_hold/README.md`](case_8_two_sphere_control_hold/README.md).

```bash
python -m cases.case_8_two_sphere_control_hold.workspace --smoke
python -m cases.case_8_two_sphere_control_hold.workspace
python -m cases.case_8_two_sphere_control_hold.workspace_aligned_bz
python -B -m unittest tests.test_case8_two_sphere_control_hold
```

## Case 9 - Two soft-magnetic spheres, one control-held

Fixed-local-field positional TFW slice for two isotropic induced-moment
spheres. Material parameters and physical radii are explicit, mutual fields
and forces are included, and only physical overlap is excluded. See
[`case_9_two_soft_spheres_control_hold/README.md`](case_9_two_soft_spheres_control_hold/README.md).

```bash
python -m cases.case_9_two_soft_spheres_control_hold.workspace --smoke
python -m cases.case_9_two_soft_spheres_control_hold.workspace
python -B -m unittest tests.test_case9_two_soft_spheres_control_hold
```
