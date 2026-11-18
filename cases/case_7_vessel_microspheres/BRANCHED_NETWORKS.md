# Branched nearby vessels

The two richer presets extend the side-by-side and over-under concepts into
separate arterial and venous trees. Each contains **14 parameterized segments**:
one trunk, four primary side branches, and two secondary branches per network.
These add branch-choice tasks while preserving the central close-approach region.

The current revision uses uneven branch origins and lengths, curved take-offs,
gentle winding centerlines, smooth taper and finer surface extraction. The
arterial and venous trees have different silhouettes rather than mirrored forks.

![Branched networks](preview/branched_networks.png)

Open [the gallery](preview/index.html), or open the individual
[side-by-side network](preview/branched_parallel/viewer.html) and
[crossing network](preview/branched_crossing/viewer.html).

## Changes from the simple pairs

* Four asymmetric side branches arise along each arterial trunk; the vein has
  four tributaries with different attachment positions. Two branches on each
  network divide again, giving secondary targets.
* Radii taper along every segment. The arterial trunk radius decreases from
  1.15 to 0.94 mm, the venous trunk from 0.80 to 0.66 mm; terminal radii are
  about 0.24–0.37 mm. These are chosen phantom dimensions, not measurements.
* The side-by-side trunks bend out of the original plane. In the crossing
  preset, the trunks move closer in depth near the projected crossing and
  separate farther away; their branches extend into different depth ranges.
* Each network's surface is extracted from the **union** of its generating
  balls. Branch mouths are open into the parent lumen; there are no internal
  branch caps. Artificial rounded caps occur only at exposed ends for viewing.
* The viewer can isolate either network, show centerlines, zoom into the close
  approach and show two illustrative branch targets. Gold sphere markers are
  enlarged; target stars are geometric assignments, not verified destinations.

## Anatomical grounding and scope

The context remains a radial-forearm vascular segment. Actual radial arteries
give off cutaneous perforators, and venous drainage includes companion veins
and smaller communicating vessels. These motivate the side branches and
tributaries, rather than an ornamental branching pattern. See the primary
anatomical studies [Vascular Analysis of Radial Artery Perforator Flaps](https://pmc.ncbi.nlm.nih.gov/articles/PMC6018282/)
and [The vascular basis of the radial forearm flap](https://pubmed.ncbi.nlm.nih.gov/3941853/).

The exact trees, branch counts, crossing, radii and spacing here are synthetic.
Only one companion venous network is represented, and arterial branches are
not connected to veins by artificial shunts. Capillary beds, the second
companion vein, valves, compliant walls and other surrounding tissue are not
included. Taper is prescribed geometrically; Murray's law, pressure loss and
flow conservation have not been used to calibrate these shapes. This is a
more structured research phantom, not a complete or patient-specific forearm.

## Editable geometry and graph

The new files are `geometries/branched_parallel.json` and
`geometries/branched_crossing.json`. They use the existing cubic Bezier
centerlines with the following optional, local-coordinate winding term:

\[
c(s)=c_{Bezier}(s)+a\sin^2(\pi s)\sin(2\pi s+\phi).
\]

`waviness_mm` is the vector a, and `waviness_phase` is φ in radians. Both the
offset and its derivative vanish at the endpoints. The branch's first point
must still match its parent's full winding centerline. With `smooth_taper=true`,
the radius interpolates between its endpoints using 3s²−2s³ instead of s.
This preserves the radius bounds while giving a gradual taper near each end.
These are explicit design curves, not a statistical model of vascular anatomy.

Topology uses three fields:

| Field | Meaning |
|---|---|
| `network` | `artery` or `vein`; the sphere's permissible connected domain |
| `parent` | Parent segment of a side/secondary branch |
| `parent_s` | Parameter on the parent where the new segment's s=0 attaches |

The child's first control point must equal the parent centerline evaluated at
`parent_s`. The exporter checks this within 1 nm in world coordinates. When
moving a parent, update attached control points and descendants accordingly.
To make a graph for routing, split each parent centerline at its attachment
parameters: these interior points are graph junctions, not just the parent
segment's endpoints. Attachment graph edges describe topology, not blood-flow
direction; physiological venous flow generally joins the trunk.

```python
from pathlib import Path
from cases.case_7_vessel_microspheres.vessel import Vessel

root = Path('cases/case_7_vessel_microspheres')
v = Vessel(root / 'geometries/branched_parallel.json')
p = v.centerline('RA_B3_twig', 0.7)  # position in EMNS metres
safe = v.clearance(p, network='artery') >= 50e-6 + 100e-6
```

Use `network=` for a sphere allowed to move between the trunk and its branches.
Using a single `branch=` would incorrectly exclude the other branches;
using no selector would incorrectly admit both arterial and venous networks.
Sampled clearance remains a sufficient, potentially conservative sphere
containment test, not exact distance to the wall.

## Separation and mesh checks

The exporter evaluates extrema of each cubic centerline along a separating
axis, expands by the absolute winding amplitude on that axis, then expands
that coordinate range by an upper bound on its radius.
Taking the outer union of these intervals over all segments gives a bound on
each network. The intervals are disjoint along local x for the parallel case
and local z for the crossing case. The natural revision retains conservative
reported lower bounds of **0.99 mm** and
**0.35 mm**, respectively, for continuous lumen separation. Unlike the simple
constant-radius presets, these are not claimed to be exact minima. Calculations
use ordinary floating-point arithmetic, not an outward-rounded proof backend.
The gap excludes vessel-wall thickness; add wall and tissue thickness when
designing a physical phantom.

Each mesh uses 301 samples per segment and a 0.20-mm extraction grid. Local
ball rasterization merges branches without an expensive full grid-by-center
distance matrix. Mesh checks require every edge to belong to two triangles,
one connected component per network, and Euler characteristic 2 for each
closed tree surface. These detect disconnected pieces and unintended handles;
they are not a certificate of mesh approximation error or CFD quality.
The attachment and separation checks concern the analytic model independently
of the displayed mesh. The smallest branches are only a few voxels across;
refine and perform convergence/geometry-error checks before using them in CFD.

Five bounded mesh-fairing passes reduce faceting and soften surface junctions.
Display vertices can move at most 0.06 mm from the extracted surface. This is
display-only: the analytic lumen and clearance API are not altered by fairing.
The exporter also checks that the final display networks have disjoint slabs
on the separating axis, and records their separate gap bound. Neither the
grid resolution nor the displacement limit is a full geometry-error bound.

Outputs in each preset's preview folder include two OBJ network surfaces in
EMNS metres, centerline/radius CSV, `geometry_checks.json` and an offline viewer.

```bash
# Read the editable JSON, rebuild both networks, validate and update previews.
python -m cases.case_7_vessel_microspheres.build_networks

# All Case 7 model tests, without a control simulation.
python -B -m unittest discover -s tests -p test_vessel_microspheres.py
```

Regeneration needs NumPy, Matplotlib and scikit-image. The separate
`create_network_presets` module is an explicit reset tool: it overwrites the
two branched JSON presets with their original designed values. Do not run it
to regenerate previews after editing geometry.

## More useful next steps

The next physical improvement is a scan-derived lumen or measured phantom
dimensions, followed by branch-specific pulsatile flow and pressure boundary
conditions. For the control study, choose target branches and test whether
the two spheres can make those branch decisions while close together, including
interaction and current limits. A geometric branch target does not imply
upstream or cross-flow reachability. Network-specific position tracking is
especially important where the two trees overlap in a camera image.

Wall thickness/compliance and a measured flow field would add more physical
value than adding increasingly tiny decorative branches. No flow solve,
trajectory optimization or control simulation has been run for these presets.
