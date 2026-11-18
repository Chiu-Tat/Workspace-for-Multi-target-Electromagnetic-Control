# Parameter and evidence provenance

Search performed 15 September 2026 across the manuscript project,
`Workspace_eMNS-main`, and `Modeling`. Sources inspected included manuscript and
review text; Python/YAML/JSON configuration; filenames and headers of CSV/TSV;
and workbook sheets/cells in XLS/XLSX files for magnet dimensions, material,
moment calibration, pair force/torque, and target-interaction measurements.

Found:

- Case-6 code constants: moving and fixed moments 0.12 and 0.145 A m2; planar
  poses and headings; ten-coil map and current limits.
- Simulator drawing dimensions: moving cylinder 9 x 5 mm and fixed cylinder
  11 x 5.5 mm. These govern only drawing geometry and have no metrology record.
- Archived ten-coil field-validation files and manuscript statistics. They
  concern coil field, not target-target interaction or coil gradient accuracy.
- Trajectory, current, position, and field-record CSV/JSON files for case-study
  operation. None provides independent target dimensions/moment uncertainty or
  coils-off pair-interaction force/torque measurements.

Not found:

- measured Case-6 target length/diameter, material/grade, magnetization
  distribution, mass, or moment-calibration uncertainty;
- a separation-orientation pair-interaction dataset;
- independent force/torque sensor data suitable for fitting or validating a
  physical discrepancy envelope;
- a validated gradient-error envelope for the ten-coil model.

Therefore `config.json` preserves simulator-derived dimensions with
`geometry_scope: illustrative_not_measured`. No current numerical discrepancy
or 1.25 multiplier may be described as a physical worst-case bound.
