# Case 8 sequential two-sphere simulation

This interactive demonstration controls one identical spherical permanent
magnet at a time while the other magnet is control-held by force feedback.

```bash
.venv/bin/python -m cases.case_8_two_sphere_control_hold.simulation.interactive
```

The initial state is:

- magnet 1 control-held at `(0.02, 0)`;
- magnet 2 active at `(-0.04, -0.03)`.

Hold W/A/S/D (or the arrow keys) to drive the active magnet continuously, in
any direction; combine keys for diagonals. The commanded force eases in and
out, and at the edge of the certified TFW the magnet slides along it. Click
anywhere in the green certified TFW to travel there. Press **H**, or click
**Return magnet 2 home, hand over**, to plan a certified path back to
`(-0.02, 0)`. Only after it arrives does the simulation switch to stage 2 and
make magnet 1 active. The same command then returns magnet 1 to `(0.02, 0)`.
Space stops, C clears the trails, X resets, and Esc quits.

## Return-path guarantee

The planner does not connect the current point to home with an unchecked
straight line. It constructs an adjacency graph from the certified rectangles
in the corresponding TFW JSON. Adjacent rectangles must share a boundary edge
of positive length. A route is a chain of such rectangles.

`CertifiedSlice.plan` crosses each shared edge at its midpoint, so every route
segment is contained in one convex certified rectangle. Because the rectangles
are small, that route zig-zags. The simulator uses `CertifiedSlice.plan_taut`
instead: it pulls the same chain taut (the funnel algorithm), giving the
shortest polyline that crosses each shared edge within the edge. The piece
between consecutive crossings still lies in one convex rectangle. Each taut
segment is then re-checked with `CertifiedSlice.reach`, which follows a segment
rectangle by rectangle. If the check fails, the midpoint route is used. The
magnet follows the route with bounded acceleration and slows for corners.
Uncertified regions are never used for movement or return paths.

At each animation step, the simulator uses the stored `(B1z,B2z)` witness of
the rectangle under the moving magnet, requests planar force components within
the certified bounds, and solves the complete 10-output/10-current actuation
system. A keyboard step is accepted only as far as `reach` certifies it; a
route step is re-checked through any corner it passes. Motion stops if a
current exceeds `±15 A`.

The simulation uses:

```text
Results/results_case8_aligned_bz.json         stage 1
Results/results_case8_aligned_bz_stage2.json  stage 2
```

This demonstrates sequential role exchange. It is not a certificate for
simultaneous movement of both magnets.
