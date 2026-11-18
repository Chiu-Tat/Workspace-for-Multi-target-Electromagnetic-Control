#!/usr/bin/env python3
# Run from the repository root: python -m analysis.affine.ten_coil_3d
"""
Affine analysis for the determination of the workspace of an electromagnetic navigation system (10 coils)
"""

# Standard library imports
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
from emns.workspace.interval_fast import Map_I2H_Interval_Fast
from emns.workspace.checks import Feasible, Out, RobustFeasible
from emns.workspace.affine import Map_I2H_Affine

# ============================================================================
# Affine version of WS_lib_Interval_fast functions (Original Working Version)
# ============================================================================

def calculate_affine_A(xi, yi, zi):
    """Affine version of calculate_interval_A"""
    target_points = [
        {'X': xi, 'Y': yi, 'Z': zi, 'm': interval[0.00, 0.00],
         'alpha': interval[0.00, 0.00], 'beta': interval[0.00, 0.00],
         'Bx': True, 'By': True, 'Bz': True, 'Bx_dx': None, 'Bx_dy': None,
         'Bx_dz': None, 'By_dy': None, 'By_dz': None, 'fx': None, 'fy': None,
         'fz': None, 'tx': None, 'ty': None, 'tz': None}
    ]

    # Use affine calculation for the first target point
    A = Map_I2H_Affine(target_points)
    A = A[:, :-1]
    # A = affine_target_point_calculation(target_points[0])

    return A

# ============================================================================
# Utility Functions for Interval Analysis (Keep original functions)
# ============================================================================

def Create3DBox(xi, yi, zi):
    """Create a 3D box for interval analysis"""
    # Initialize
    Box = [[0 for c in range(1)] for r in range(3)]

    xmin = xi[0][0]
    xmax = xi[0][1]
    ymin = yi[0][0]
    ymax = yi[0][1]
    zmin = zi[0][0]
    zmax = zi[0][1]

    Box[0][0] = interval[xmin, xmax]
    Box[1][0] = interval[ymin, ymax]
    Box[2][0] = interval[zmin, zmax]

    return Box

def BisectBox(Box):
    """Bisection of a box and return the list of bisected boxes"""
    d = np.shape(Box)[0]

    # Initialize
    Bb = [[0 for c in range(2)] for r in range(d)]
    comb = product(np.arange(2), repeat=d)
    M = np.asarray(list(comb))

    # Divide each 1D-intervals
    for i in range(d):
        inter = Box[i][0]
        xmin = inter[0][0]
        xmax = inter[0][1]
        Bb[i][0] = interval[xmin, (xmin+xmax)/2]
        Bb[i][1] = interval[(xmin+xmax)/2, xmax]

    # Create list of bisected boxes
    Box_list = []
    for i in range(2**d):
        B_temp = [[0 for c in range(1)] for r in range(d)]
        for j in range(d):
            B_temp[j][0] = Bb[j][M[i,j]]
        Box_list.append(B_temp)

    return Box_list

def Ws3DDetermination(InitBox, bmin, bmax, Imin, Imax, eps, max_iter=-1):
    """Determination of the 3D workspace of a eMNS using affine analysis"""
    L = []
    Lin = []
    Lout = []
    Lneg = []
    L.append(InitBox)
    iteration = 0
    I_intervals = [interval[Imin, Imax]] * 10
    b_intervals = [interval[bmin, bmax], interval[bmin, bmax], interval[bmin, bmax]]

    while(len(L) > 0):
        # Extract intervals
        CurrentBox = L[0]
        del L[0]  # remove the element from the list

        # Build interval matrix using affine method
        xi = CurrentBox[0][0]
        yi = CurrentBox[1][0]
        zi = CurrentBox[2][0]

        Ainter = calculate_affine_A(xi, yi, zi)

        # Check feasibility
        isFeasible = Feasible(Ainter, I_intervals, b_intervals)

        if isFeasible:
            Lin.append(CurrentBox)
        else:
            isOut = Out(Ainter, I_intervals, b_intervals)
            if isOut:
                Lout.append(CurrentBox)
            else:
                if ((xi[0][1] - xi[0][0]) > eps and
                   (yi[0][1] - yi[0][0]) > eps and
                   (zi[0][1] - zi[0][0]) > eps):
                    Bisected_box = BisectBox(CurrentBox)
                    L.extend(Bisected_box)
                else:
                    Lneg.append(CurrentBox)

        iteration = iteration + 1

        if iteration >= max_iter and max_iter >= 1:
            print('Max iter reached')
            break

    return L, Lin, Lout, Lneg, iteration

# ============================================================================
# Plotting Helper
# ============================================================================

def plot_cuboid(ax, origin, size, color_face):
    """
    Plots a 3D cuboid.
    """
    ox, oy, oz = origin
    dx, dy, dz = size
    v = np.array([
        (ox, oy, oz), (ox + dx, oy, oz), (ox + dx, oy + dy, oz), (ox, oy + dy, oz),
        (ox, oy, oz + dz), (ox + dx, oy, oz + dz), (ox + dx, oy + dy, oz + dz), (ox, oy + dy, oz + dz)
    ])
    faces = [
        [v[0], v[1], v[2], v[3]], [v[4], v[5], v[6], v[7]], [v[0], v[1], v[5], v[4]],
        [v[2], v[3], v[7], v[6]], [v[0], v[3], v[7], v[4]], [v[1], v[2], v[6], v[5]]
    ]
    poly3d = Poly3DCollection(faces, facecolors=color_face, linewidths=0.5, edgecolors='k', alpha=0.1)
    # poly3d = Poly3DCollection(faces, facecolors=color_face, linewidths=0.5, alpha=0.25)
    ax.add_collection3d(poly3d)

# ============================================================================
# Main Application
# ============================================================================

def main():
    """Main function for affine analysis of 10-coil system"""

    print("Affine analysis for the determination of the workspace of an electromagnetic navigation system (10 coils)")
    print("=" * 80)

    # Build initial box
    pmin = -0.05
    pmax = 0.05
    xi = interval[pmin, pmax]
    yi = interval[pmin, pmax]
    zi = interval[pmin + 0.085, pmax + 0.085]  # Adjusted z-offset to match the original context
    InitBox = Create3DBox(xi, yi, zi)

    # Define inputs interval and task set
    Imin = -15
    Imax = 15
    bmin = -0.05
    bmax = 0.05

    # Define precision of bisection and run the determination algorithm
    eps = 0.005
    print(f"Running 3D workspace determination with precision eps = {eps}")
    print("This may take some time...")

    start_time = time.time()
    L, Lin, Lout, Lneg, iteration = Ws3DDetermination(
        InitBox, bmin, bmax, Imin, Imax, eps
    )
    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.2f} seconds")

    print(f"Completed after {iteration} iterations")
    print(f"Boxes in workspace: {len(Lin)}")
    print(f"Boxes out of workspace: {len(Lout)}")
    print(f"Undetermined boxes: {len(Lneg)}")

    # Plotting
    print("Creating 3D plot...")

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot boxes in workspace (green)
    for box in Lin:
        origin = (box[0][0][0][0], box[1][0][0][0], box[2][0][0][0])
        size = (box[0][0][0][1] - origin[0], box[1][0][0][1] - origin[1], box[2][0][0][1] - origin[2])
        plot_cuboid(ax, origin, size, 'green')

    # # Plot undetermined boxes (gray)
    # for box in Lneg:
    #     origin = (box[0][0][0][0], box[1][0][0][0], box[2][0][0][0])
    #     size = (box[0][0][0][1] - origin[0], box[1][0][0][1] - origin[1], box[2][0][0][1] - origin[2])
    #     plot_cuboid(ax, origin, size, 'gray')

    # # Plot boxes out of workspace (red)
    # for box in Lout:
    #     origin = (box[0][0][0][0], box[1][0][0][0], box[2][0][0][0])
    #     size = (box[0][0][0][1] - origin[0], box[1][0][0][1] - origin[1], box[2][0][0][1] - origin[2])
    #     plot_cuboid(ax, origin, size, 'red')

    # Format plot
    ax.axis('equal')
    ax.set_title('Workspace determination by affine analysis (10 coils)')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('z (m)')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='green', alpha=0.5, label='In workspace'),
                    #   Patch(facecolor='gray', alpha=0.2, label='Undetermined'),
                    #   Patch(facecolor='red', alpha=0.2, label='Out of workspace')
                      ]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    plt.show()

    print("3D Analysis complete!")

if __name__ == "__main__":
    main()
