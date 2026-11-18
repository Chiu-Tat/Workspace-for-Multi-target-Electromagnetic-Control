# Case Study 2 — hollow STEP vessel models

Two publication layouts: `cs2_branched_side_by_side.step` and `cs2_branched_over_under.step`. Each is a coloured assembly with two separate valid B-rep solids: artery and vein. All 14 source branch segments are included and fused at their intersections within each network.

Main trunks are **8 mm outer diameter** and **4 mm inner diameter**, with a **2 mm nominal wall** and flat annular ports open at both ends. Each continuous bore follows its original curved main-trunk centreline. Side branches retain their previous exterior sizes and rounded tips; no separate side-branch bores were requested. Units are **millimetres**.

The approximately 100 mm y length is retained. The networks were separated slightly along x (side-by-side) or z (over-under) to accommodate the 8 mm outer trunks, with a conservative 1 mm separating-slab gap. The closest sampled trunk midpoint is registered at (0, 0, 70.2) mm. Source radius fields describe the 4 mm internal lumen; `main_trunk_pipe` describes the external pipe dimensions. Particle dimensions and magnetic parameters are separate simulation settings.

`validation.json` records scaling factors, source/output hashes, per-branch sampled surface errors, solid validity, dimensions, network separation, bore/wall samples, open annular ports and STEP re-import checks. Surface sampling is a diagnostic and does not prove a global approximation bound.

Regenerate from the repository root with CadQuery 2.8.0 and Matplotlib installed:

```bash
python -B -m revised_case_studies.cs2_vessel_microspheres.export_step
```
