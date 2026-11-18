# Case Study 1 figures 14 and 15: provenance

Reproduce from `Workspace_eMNS-main`:

```bash
.venv/bin/python -B -m revised_case_studies.cs1_cylinder_interaction.hardware     # trials.json
.venv/bin/python -B -m revised_case_studies.cs1_cylinder_interaction.figures_v2   # both figures
```

Every element comes from a computed result or an experiment record.

| Element | Source |
|---|---|
| 3-D TFWs, sections, box outlines, area and volume shares | `results/cs1/tfw3d_fine/tfw3d_<layout>.json` (affine-hybrid cascade, `run_tfw3d_fine.py`) |
| Heading coverage | Same TFW boxes: share of the 64 heading levels with a Model-in box at each raster cell |
| P1–P3 point tests | Endpoint LPs (`publication.point_test`); single layout `results/cs1/publication/orientation_probes.json`, gate and triangle `orientation_probes_<layout>.json`. Gate and triangle positions were picked from a computed candidate scan (`P1`: all headings feasible; `P2`, `P3`: partial and none) |
| Fixed-magnet positions and headings | `cases/case_6_.../workspace_hybrid.LAYOUTS`, identical to each trial's `session.json` |
| Paths, headings, guard holds, RMSE, goal distances, currents | `ROS_CAN/src/case_study1_*/results/<trial>/telemetry.csv` and `session.json` |
| Force margin, clearance, TFW membership per pose | `hardware.py` (LP on the nominal model), stored in `trials.json` |
| 16 mm guard | `session.json`: `configuration.control.min_separation_m = 0.016` |
| Camera frames | `ui.avi` of each trial, extracted with ffmpeg at the stated time; cached as `frames/<name>_<t>s.png` |
| Camera-to-mm mapping | `control_panel.py` (`CameraView.world_transform`) draws the calibrated +-50 mm square; its 607 x 607 px box at (21, 69) was measured from the video pixels. Check: green-marker centroids vs logged poses, mean offset (0.7, 0.3) mm over 39 frames |
| Fig. 15 (a)–(c) planned paths in 3-D | `session.json` → `autonomy.samples` (x, y, beta) drawn in the `tfw3d_<layout>.json` voxels |
| Fig. 15 (d)–(f) time-lapse | Per-pixel median of 15 `ui.avi` frames per run (background); discs of radius 7 mm copied from the frame at each logged pose, every 14 mm of travel (`timelapse_image`) |
| Fig. 15 (g) TFW margin | Signed planar distance to the Model-in boundary at the pose's heading level (`tfw_margin`, distance transform of the TFW raster) |
| Ellipses on the moving magnet | Logged pose at the frame time; after tracking was lost, the largest green blob in the frame (`detect_moving_magnet`) |

Schematic only (stated in the captions): magnet glyphs and the place where the moving-magnet glyph is drawn in the section panels.
