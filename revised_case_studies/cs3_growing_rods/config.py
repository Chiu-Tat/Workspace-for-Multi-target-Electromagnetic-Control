"""Illustrative bench parameters, not measured implant specifications."""
MOMENT = 0.04                  # A m^2; diametrically magnetized rotor
ROTOR_RADIUS = 0.003           # m; schematic/non-overlap check only
I_LIMIT = 15.0                 # A; thermal and slew limits not imposed
B_ROT = 0.015                  # T; design threshold, not measured B_req
HOLD_EPS = 0.0003              # T; illustrative field diagnostic threshold
FORCE_LIMIT = 0.15             # N; separate conservative sensitivity study
GEAR_RATIO = 64.0
GEAR_EFFICIENCY = 0.5
SCREW_EFFICIENCY = 0.3
LEAD_PITCH = 0.0005            # m per output revolution
LOAD = 2.0                    # N; constant bidirectional resisting bench load
FRICTION = 1e-5               # N m; Coulomb/static friction
ROT_DRAG = 2e-5               # N m s/rad; overdamped resistance
FREQ = 0.5                    # Hz; illustrative drive rate
DT_CTRL = 0.02                # s; 50 Hz allocation
DT_PLANT = 0.001              # s; 1 kHz mechanism integration
PHASE_SEQUENCE = [(32.,(.5,0.)),(32.,(0.,.5)),(32.,(.5,.25)),(16.,(-.5,0.))]
