# Robotic case studies

Run these modules from the repository root after the [Python setup](../README.md#setup). Each module writes its outputs under [results](results/). The hardware plots use processed telemetry and selected camera images. The repository includes the inputs required for the commands below.

## 1. Cylindrical magnets

The workspace and autonomous-trial figures use saved three-dimensional TFW maps for the single, gate, and triangle layouts, together with measured poses and camera frames. The speed analysis uses the moving stages of the recorded loop trial.

```sh
python -m revised_case_studies.cs1_cylinder_interaction.figures_v2
python -m revised_case_studies.cs1_cylinder_interaction.speed_sweep
```

To recompute a fine TFW map, use `python -m revised_case_studies.cs1_cylinder_interaction.run_tfw3d_fine --layout single`. The fine search can take several hours. See the [classifier](../hybrid_region/) and [saved maps](results/cs1/tfw3d_fine/).

## 2. Magnetic spheres in vessels

The TFW analysis uses the two arc positions along the branched vessel paths. The hardware figure uses the saved force-margin maps, trial records, and median camera backgrounds for the two guided trials and the manual over-under trial.

```sh
python -m revised_case_studies.cs2_vessel_microspheres.figure_hw
```

To recompute the archived branched maps at their recorded resolution, run the following searches. Each can take several minutes.

```sh
python -m revised_case_studies.cs2_vessel_microspheres.tfw --variant branched_side_by_side --eps 0.00390625
python -m revised_case_studies.cs2_vessel_microspheres.tfw --variant branched_over_under --eps 0.00390625
```

## 3. Magnetic growing rods

The calibration workflow uses the retained push-force readings and the ten-coil configuration from the bench trial. It computes spacing, height, and offset slices and checks the commanded phase pairs at the recorded placement. The phantom workflow plots the measured marker traces and field targets with the three extracted camera frames.

```sh
python -m revised_case_studies.cs3_growing_rods.plot_calibration_tfw
python -m revised_case_studies.cs3_growing_rods.plot_phantom_trial
```

The [input files](cs3_growing_rods/data/) contain the recorded coil and magnet parameters, force-gauge readings, plot columns from the phantom trial, and the selected camera frames. The calibration figure uses the [supplied CAD render](cs3_growing_rods/assets/cs3_blender_model.png).

## Supporting simulations

Each case's `publication` module can redraw the earlier simulation figures from saved simulation records with `--render`. The commands above produce the revised manuscript's experimental figures. The CS1 and CS2 `hardware` utilities can reprocess the original recordings if `ROS_CAN_SRC` points to their source directory and FFmpeg is installed. The repository includes processed outputs for those utilities. The earlier helical-robot preparation is available in [cs3_helical_robots](cs3_helical_robots/).
