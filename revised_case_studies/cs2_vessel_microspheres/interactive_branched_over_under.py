#!/usr/bin/env python3
# Run from the repository root:
#   python -m revised_case_studies.cs2_vessel_microspheres.interactive_branched_over_under
"""Case Study 2 -- autonomous simulation, branched arterial and venous networks, over-under.

Case 7 preset with 14 vessel segments (trunk, four side branches, two
secondary branches per network).  Each microsphere stays in its main vessel
(artery / vein trunk); the side branches are anatomical context only.  Both
spheres are driven simultaneously by one ten-coil current vector, each moving
only forward / backward along its trunk.
See ``vessel_pair.py`` (model, mission) and ``vessel_pair_sim.py`` (display).
"""

from revised_case_studies.cs2_vessel_microspheres.vessel_pair_sim import main

if __name__ == '__main__':
    main('branched_over_under')
