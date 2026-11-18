# Vessel variants: connected branch versus nearby separate channels

The gallery now starts with two richer [branched-network versions](BRANCHED_NETWORKS.md).
The three configurations below remain as simpler comparison models.

Open [the interactive gallery](preview/index.html) to switch between all three.
For the two new variants, one sphere is already in each vessel. They are not
expected to cross a vessel wall or transfer between the two channels.

| Preset | Anatomical inspiration | Topology | Minimum lumen gap |
|---|---|---|---|
| `geometry.json` | Carotid bifurcation in the neck | One connected Y lumen | Not applicable at the junction |
| `geometries/close_parallel.json` | Radial artery and one companion vein in the forearm | Two separate, curved lumens approaching side by side | 0.50 mm |
| `geometries/close_crossing.json` | Engineered forearm-inspired artery/vein crossing | Two separate lumens crossing in projection, separated in z | 0.35 mm |

The radial artery's accompanying venous anatomy provides a specific bodily
context for a nearby two-vessel experiment. The real arrangement includes
paired companion veins and communicating branches; only an artery and one
companion vein are retained here. A forearm anatomical study measured these
vessels [An anatomical study on the forearm vascular system](https://pubmed.ncbi.nlm.nih.gov/8938512/).
The selected centerline shapes, gaps, crossing pattern and constant diameters
(artery 2.3 mm, vein 1.6 mm) are design assumptions, not a reconstruction of
that study or a patient. The crossing is an engineered geometry variation,
not a claim that a specific patient has that exact vessel path.

![Two nearby vessel pairs](preview/vessel_variants.png)

The close-up panels show sections of the closest **generating balls**, with
100-µm-diameter spheres at true size. They are not exact oblique lumen sections.
The 3D viewer enlarges the particle markers to make their locations visible.
The gap is between lumen boundaries; no wall thickness is included. To treat
the remaining space as tissue, require gap > wall_thickness_1 + wall_thickness_2
plus a tissue margin, or explicitly add wall surfaces.

## Continuous separation, not just a visual gap

All new radii are constant. In the parallel preset, the artery's centerline
satisfies x≤-1.225 mm and the vein's x≥1.225 mm. Thus any two generating ball
centers are separated by at least 2.45 mm; subtracting 1.15+0.80 mm gives a
0.50-mm lower bound on lumen separation. Both centerlines attain this at
s=0.5, y=z_local=0, so this is the global minimum for the continuous model.

In the crossing preset, the centerlines lie in z_local=-1.15 mm and +1.15 mm.
Their projected crossing occurs at s=0.5 with x=y=0. Their minimum separation
is therefore 2.30 mm, giving 2.30-1.15-0.80=0.35 mm between lumen boundaries.
The apparent X in a front view is not a vessel junction.

`build_variants.separation_certificate` computes scalar cubic extrema from
derivative roots and checks this separating-axis bound independently of mesh
resolution. The exported `geometry_checks.json` files record the bound and
whether it is attained. If you change the geometry enough to invalidate the
separating-axis proof, the exporter stops; that does not necessarily mean
intersection, only that this particular certificate no longer proves separation.

## Change the spacing

Edit the centerline control points in the JSON and regenerate. For the parallel
pair with vein endpoint x=+7 mm and inner control-point x=c (and the artery
coordinates negated), the minimum centerline separation is 3.5+1.5c mm when
-7/3<c≤7. Therefore set c=(desired_gap_mm+1.95-3.5)/1.5.
The default c=-0.70 gives a 0.50-mm gap. Bezier control points are handles,
not points the vessel necessarily passes through; the lumens stay separate
even though these inner handles cross the x=0 plane.
For the crossing pair set all artery z values to -h/2 and vein z values to +h/2,
where h=desired_gap_mm+1.95. These recipes assume the default other coordinates
and radii remain unchanged. `designed_lumen_gap_mm` is descriptive metadata;
update it alongside control points. The validation checks recompute the gap
from geometry, not from that metadata.

```bash
python -m cases.case_7_vessel_microspheres.build_variants
python -B -m unittest discover -s tests -p test_vessel_microspheres.py
```

Each separate vessel has its own OBJ and centerline samples. The display mesh
uses 151 cross-section rings and 32 angular samples, with artificial closed
end caps; it approximates a smooth tube, not an exact wall-distance oracle or
CFD domain. The analytic swept-ball geometry and conservative clearance API
remain the source for containment. These new previews require NumPy and
Matplotlib; regenerating the original merged carotid mesh also uses scikit-image.

```python
from pathlib import Path
from cases.case_7_vessel_microspheres.vessel import Vessel
folder = Path('cases/case_7_vessel_microspheres')
vessel = Vessel(folder / 'geometries/close_parallel.json')
p1 = vessel.centerline('RA', 0.5)
p2 = vessel.centerline('RV', 0.5)
# For separate lumens, query each sphere against its assigned vessel.
safe_1 = vessel.clearance(p1, branch='RA') >= 50e-6 + 100e-6
```

## Control implications

The ten-coil aligned-moment force theory in `THEORY.md` applies unchanged, but
the topology changes the task: p1∈Ω_RA and p2∈Ω_RV, rather than allowing each
sphere anywhere in the union. Maintain the vessel assignment in localization,
planning and containment checks. In a 2D projection the crossing may be
ambiguous; depth-resolved localization is needed to distinguish the channels.

For an artery/vein model, nominal blood-flow directions can oppose each other.
Choose separate u_RA and u_RV and separate boundary data; there is no junction
flow-conservation constraint Q_RA=Q_RV for these isolated modeled segments.
Venous valves/compliance and pulsatile arterial flow are not modeled here.

Near the closest approach the spheres see similar coil fields, which can make
independent force allocation poorly conditioned. Vessel separation does not
shield magnetic interaction. Compare the full pair Jacobian's singular values,
dipole interaction forces, achievable lateral drift and current limits as a
function of sphere separation before claiming independent control. Centered
spheres are 2.45 mm or 2.30 mm apart in the displayed examples; near-wall spheres
could approach substantially closer, so lumen gap is not particle center distance.
All of these are proposed subsequent analyses; no controller or motion
simulation has been run.
