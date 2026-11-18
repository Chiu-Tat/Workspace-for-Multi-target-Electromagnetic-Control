# Case 7 — two magnetic microspheres in blood vessels

**New: two branched network models**, each with tapered trunks, four primary
branches and two secondary branches per network. See
[branched geometry and validation](BRANCHED_NETWORKS.md) or
[open the gallery](preview/index.html).

![Branched vessel networks](preview/branched_networks.png)

**Three simpler baseline options:** a carotid Y bifurcation, two separate vessels
approaching side by side, and two separate vessels crossing at different depths.
Open [the interactive gallery](preview/index.html), or read
[the vessel variants and gap parameterization](VESSEL_VARIANTS.md).

![Separate close-proximity vessel pairs](preview/vessel_variants.png)

In the Y configuration, route sphere 1 into the internal carotid branch and
sphere 2 into the external carotid branch. In the separate-vessel configurations,
each sphere stays in its own lumen. Their dipoles align with the local field;
only translation is controlled. This case supplies geometry and theory,
without a control simulation.

![Idealized carotid bifurcation](preview/carotid_preview.png)

Open [the offline 3D viewer](preview/carotid_viewer.html) in a browser: drag to
rotate, scroll to zoom, and adjust wall opacity. Sphere markers are enlarged;
their illustrative physical diameter is 100 µm.

| File | Purpose |
|---|---|
| `geometry.json` | Editable branch control points, radii, bulb, coil-frame placement |
| `vessel.py` | Bezier centerlines, interior coordinates, conservative clearance |
| `magnetics.py` | Ten-coil field/gradient and two aligned-moment force laws |
| [THEORY.md](THEORY.md) | Geometry, flow, interaction, simultaneous control and feasibility |
| `build_preview.py` | Geometry-only exports; no control or flow calculation |
| `preview/carotid_lumen.obj` | Merged lumen surface in EMNS metres, with closed viewing caps |
| `preview/centerlines.csv` | Branch parameter, world coordinates and radii in SI units |

The shape is inspired by the carotid bifurcation in the neck, not reconstructed
from a scan. Read the theory note before treating the preview mesh as a CFD or
collision model. The initial sphere positions and outlet assignments are a task
concept, not computed trajectories or verified reachable states.

From the repository root:

```bash
# Regenerate geometry outputs (requires numpy, matplotlib, scikit-image).
python -m cases.case_7_vessel_microspheres.build_preview

# Regenerate the two separate-vessel previews (numpy, matplotlib).
python -m cases.case_7_vessel_microspheres.build_variants

# Regenerate the richer branched networks (also requires scikit-image).
python -m cases.case_7_vessel_microspheres.build_networks

# Geometry and field-model checks; no control simulation.
python -B -m unittest discover -s tests -p test_vessel_microspheres.py
```

The existing environment's NumPy/SciPy/Matplotlib suffice for the geometry and
magnetics APIs. `scikit-image` is only needed to regenerate the surface/preview;
the prebuilt viewer is self-contained and requires no Python or network access.

```python
from cases.case_7_vessel_microspheres.vessel import Vessel
vessel = Vessel()
p = vessel.point('ICA', s=0.6, rho=0.3, theta=1.2)  # EMNS metres
safe = vessel.clearance(p) >= 50e-6 + 100e-6       # radius + assumed margin
```

Sphere magnetization and flow properties remain to be selected before running
a controller. The magnetic helper uses the existing coil dipole parameters but
evaluates its own full analytic field gradient; see the derivative note in
`THEORY.md`.
