"""Case Study 2 parameters.

Values marked PLACEHOLDER are illustrative and must be replaced by measured
values before any result is reported as experimental (see the paper's
AuthorAction box for Case Study 2).
"""

from __future__ import annotations

import math

from cases.case_7_vessel_microspheres.vessel import CONFIG as CAROTID_GEOMETRY

# ---- vessel phantom (Case 7 idealized carotid bifurcation, EMNS frame) ----
GEOMETRY = CAROTID_GEOMETRY
LUMEN_MESH = 'cases/case_7_vessel_microspheres/preview/carotid_lumen.obj'

# Each sphere: start (branch, s) and its route as an ordered list of
# (branch, s_start, s_end) segments along Case 7 centerlines.
SPHERES = {
    'sphere 1': {'route': [('CCA', 0.52, 1.0), ('ICA', 0.0, 0.92)], 'target': 'ICA'},
    'sphere 2': {'route': [('CCA', 0.20, 1.0), ('ECA', 0.0, 0.92)], 'target': 'ECA'},
}

# ---- spheres: 3 mm diameter N35 NdFeB spheres, freely rotating, constant |m|
# (a permanent magnet, so the aligned constant-moment model applies directly).
# Remanence is the nominal N35 value (1.17-1.21 T); measure before reporting.
RADIUS = 1.5e-3                     # m
REMANENCE = 1.19                    # T, nominal N35 (PLACEHOLDER until measured)
MAGNETIZATION = REMANENCE / (4e-7 * math.pi)                 # A/m
MOMENT = MAGNETIZATION * 4.0 / 3.0 * math.pi * RADIUS ** 3   # A m^2
MIN_SEPARATION = 4 * RADIUS         # m, centre distance kept by the planner

# ---- fluid: water-glycerin solution, 5.4 cP, to mimic blood (no imposed flow) ----
VISCOSITY = 5.4e-3                  # Pa s (5.4 cP)
FLOW_MEAN = 0.0                     # m/s, mean axial flow (Poiseuille), 0 = static
DRAG = 6 * math.pi * VISCOSITY * RADIUS                      # N s / m, unbounded Stokes

# ---- wall contact (Case Study 2 publication plant) ----
# The sphere (106 mg) rests on the 4 mm lumen wall.  Its moment is locked to
# the +y field, so it slides rather than rolls: Coulomb friction mu * N, with
# N the magnitude of the transverse load (buoyant weight + magnetic side force).
DENSITY = 7500.0                    # kg/m^3, sintered NdFeB
FLUID_DENSITY = 1115.0              # kg/m^3, ~45 wt% glycerin (5.4 cP)
GRAVITY = 9.81                      # m/s^2, acting along -z of the eMNS frame
WEIGHT = (DENSITY - FLUID_DENSITY) * 4.0 / 3.0 * math.pi * RADIUS ** 3 * GRAVITY   # N, buoyant
WEIGHT_VECTOR = (0.0, 0.0, -WEIGHT)
MU_WALL = 0.2                       # sliding friction coefficient (PLACEHOLDER)
LUMEN_DIAMETER = 4e-3               # m, main-trunk bore
_K = 2 * RADIUS / LUMEN_DIAMETER    # Haberman-Sayre wall correction, sphere on tube axis
WALL_DRAG_FACTOR = (1 - 0.75857 * _K ** 5) / (
    1 - 2.1050 * _K + 2.0865 * _K ** 3 - 1.7068 * _K ** 5 + 0.72603 * _K ** 6)
DRAG_WALL = WALL_DRAG_FACTOR * DRAG                          # N s / m

# ---- actuation task ----
B_HOLD = 0.005                      # T, preferred net field magnitude at each sphere
B_MIN = 0.002                       # T, alignment threshold (PLACEHOLDER)
FIELD_DIRECTION = (0.0, 1.0, 0.0)   # common net-field (moment) direction at both spheres
SPEED = 0.002                       # m/s, reference speed along each route
K_P = 2.0                           # 1/s, position feedback gain
# Allocator weights (relative errors).  The field direction (moment
# direction) is held tightly; its magnitude may float above B_MIN.
F_SCALE = DRAG * SPEED              # N, force needed for the reference speed
W_B_DIR = 1e3 / B_HOLD
W_B_MAG = 1e-2 / B_HOLD
W_F_TANGENT = 1.0 / F_SCALE
W_F_TRANSVERSE = 1.0 / F_SCALE
I_LIMIT = 15.0
DT = 0.1                            # s, 10 Hz allocation
