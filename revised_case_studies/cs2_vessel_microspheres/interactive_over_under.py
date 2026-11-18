#!/usr/bin/env python3
# Run from the repository root:
#   python -m revised_case_studies.cs2_vessel_microspheres.interactive_over_under
"""Case Study 2 -- autonomous simulation, over-under vessel pair (Case 7 preset).

Two microspheres, one per thin vessel, each moving only forward / backward
along its centerline, driven simultaneously by one ten-coil current vector.
See vessel_pair.py (model, mission) and vessel_pair_sim.py (display).
"""

from revised_case_studies.cs2_vessel_microspheres.vessel_pair_sim import main

if __name__ == '__main__':
    main('over_under')
