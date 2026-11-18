"""Case Study 3 parameters.

The geometry and the actuation task are those of archive Case 3
(``cases/case_3_circular_path_magnets/workspace.py``), whose certified TFW
is ``Results/results_case3.json``:

  robot i centre   (X, Y, Z) = (+-OFFSET + R cos a_i, R sin a_i, Z_PLANE)
  a_1 in [144, 216] deg,  a_2 in [-36, 36] deg
  task             body field (Bx, By, Bz) = (0, 0, B_ROT) for EVERY roll
                   angle alpha in [0, 2 pi], at both robots simultaneously,
                   with beta_i = a_i + pi/2 (path tangent).

In the body convention of ``emns.workspace`` the body z axis is
(sin(alpha) cos(beta), sin(alpha) sin(beta), cos(alpha)); sweeping alpha
therefore turns the field in the vertical plane that contains the path
tangent.  The certified rotation axis of each robot is the horizontal path
normal.  A helical robot lying across the channel with its axis along that
normal rolls along the channel (surface rolling) while its screw thrust
presses it against the outer wall.

Values marked PLACEHOLDER are illustrative and must be replaced by measured
robot and medium data (paper AuthorAction box for Case Study 3).
"""

from __future__ import annotations

import math

# ---- archive Case 3 geometry and task (do not change: the TFW depends on it)
OFFSET = 0.10625          # m
R_PATH = 0.1              # m
Z_PLANE = 0.055           # m
MOMENT = 0.023            # A m^2
B_ROT = 0.01              # T, rotating-field magnitude
A1_RANGE = (144 / 180 * math.pi, 216 / 180 * math.pi)
A2_RANGE = (-math.pi / 5, math.pi / 5)
TFW_JSON = 'results_case3.json'

# ---- helical robots (PLACEHOLDER)
HELIX_LENGTH = 5e-3       # m, lies across the lumen
HELIX_RADIUS = 1.5e-3     # m, outer radius (rolling radius)
HELIX_PITCH = 2.5e-3      # m
ROLL_EFFICIENCY = 0.6     # fraction of no-slip rolling speed realised

# ---- medium: water-glycerin solution, 5.4 cP, to mimic blood
VISCOSITY = 5.4e-3        # Pa s
# Rotational resistance = fluid drag (slender cylinder about its axis,
# 4 pi eta r^2 L) + wall-contact rolling resistance.  With the archived
# Case 3 moment, fluid drag alone would put step-out far above any command,
# so the observed step-out is set by contact; STEP_OUT_HZ is a PLACEHOLDER
# to be measured in the 5.4 cP phantom.
FLUID_ROT_DRAG = 4 * math.pi * VISCOSITY * HELIX_RADIUS ** 2 * HELIX_LENGTH
STEP_OUT_HZ = 6.0         # Hz, measured step-out in the phantom (PLACEHOLDER)
ROT_DRAG = MOMENT * B_ROT / (2 * math.pi * STEP_OUT_HZ)   # N m s / rad, total
CONTACT_ROT_DRAG = ROT_DRAG - FLUID_ROT_DRAG

# ---- phantom (PLACEHOLDER): two curved lumens in a gel block
LUMEN_RADIUS = 4e-3       # m (diameter > helix length)

# ---- control
DT = 0.1                  # s, allocation rate
DT_PLANT = 0.002          # s
I_LIMIT = 15.0
MAX_FREQ_HZ = 4.0         # commanded rotation frequency limit (< step-out)
