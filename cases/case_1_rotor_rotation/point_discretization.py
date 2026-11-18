#!/usr/bin/env python3
# Run from the repository root: python -m cases.case_1_rotor_rotation.point_discretization
"""
Discretization method for the determination of the workspace of an electromagnetic navigation system (10 coils)
"""

# Standard library imports
import math
from itertools import product
import time

# Scientific computing imports
import numpy as np
# Plotting imports
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
# Interval analysis imports - use our custom module
from emns.numerics.interval import interval, imath, fpu
from emns.workspace.checks import Feasible, Out, RobustFeasible, create_I_intervals, create_b_intervals
from emns.numerics import affine as ca
from emns.workspace.fast import ModifiedHyperplaneShiftingMethod, Map_I2H_body

# ============================================================================

x = np.linspace(0, 2 * math.pi, 50)
y = np.linspace(0, 2 * math.pi, 50)

# Create a meshgrid from x and y
X, Y = np.meshgrid(y,x)

I_min = -15
I_max = 15

# Define the vertices of the B field box (rectangular)
box_vertices = np.array([
    [0, 0.045, 0, 0.045],
])

# Evaluate the function at each point in the meshgrid
Z = np.zeros_like(X)
start_time = time.time()
for i in range(len(x)):
    for j in range(len(y)):
        target_points = [
            {'X': 0.02, 'Y': 0, 'Z': 0.085, 'm': 0.145,
            'alpha': math.pi/2, 'beta': X[i,j],
            'Bx': None, 'By': True, 'Bz': True, 'Bx_dx': None, 'Bx_dy': None,
            'Bx_dz': None, 'By_dy': None, 'By_dz': None, 'fx': None, 'fy': None,
            'fz': None, 'tx': None, 'ty': None, 'tz': None},
            {'X': -0.02, 'Y': 0, 'Z': 0.085, 'm': 0.145,
            'alpha': math.pi/2, 'beta': Y[i,j],
            'Bx': None, 'By': True, 'Bz': True, 'Bx_dx': None, 'Bx_dy': None,
            'Bx_dz': None, 'By_dy': None, 'By_dz': None, 'fx': None, 'fy': None,
            'fz': None, 'tx': None, 'ty': None, 'tz': None},
            ]
        A = Map_I2H_body(target_points)
        G, K = ModifiedHyperplaneShiftingMethod(A, I_min, I_max)
        # Check if G*box_vertices <= K
        box_constraints = G @ box_vertices.T
        d = K - (box_constraints.max(axis=1)).reshape(-1, 1)
        # print("K:", K)
        # print("box_constraints:", (box_constraints.max(axis=1)).reshape(-1, 1))

        Z[i, j] = 1 if (d >= 0).all() else -1

end_time = time.time()
print(f"Core algorithm runtime: {end_time - start_time:.4f} seconds")

# Create a 2D plot
# Get coordinates for feasible (Z=1) and infeasible (Z=-1) points
feasible_x = X[Z == 1]
feasible_y = Y[Z == 1]
infeasible_x = X[Z == -1]
infeasible_y = Y[Z == -1]

# Create a scatter plot
plt.scatter(infeasible_x, infeasible_y, color='lightgray', s=10, label='Out workspace')
plt.scatter(feasible_x, feasible_y, color='lightgreen', s=10, label='In workspace')

# Set the aspect ratio of the plot to be equal
plt.gca().set_aspect('equal')

# Add labels and title
plt.xlabel('x')
plt.ylabel('y')

# Set the x and y axis tick labels
plt.xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi], ['0', 'π/2', 'π', '3π/2', '2π'])
plt.yticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi], ['0', 'π/2', 'π', '3π/2', '2π'])

# Add a legend
plt.legend()

# Show the plot
plt.grid()
plt.show()
