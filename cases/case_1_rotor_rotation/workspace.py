#!/usr/bin/env python3
# Run from the repository root: python -m cases.case_1_rotor_rotation.workspace
"""
Affine analysis for the determination of the workspace of an electromagnetic navigation system (10 coils)
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
from emns.workspace.affine import Map_I2H_Affine_body

# ============================================================================
# Affine version of WS_lib_Affine functions
# ============================================================================

def calculate_affine_A(beta1, beta2):
    """Affine version of calculate_interval_A"""
    target_points = [
        {'X': interval[0.02, 0.02], 'Y': interval[0, 0], 'Z': interval[0.085, 0.085], 'm': interval[0.145, 0.145],
         'alpha': interval[math.pi/2, math.pi/2], 'beta': beta1,
         'Bx': None, 'By': True, 'Bz': True, 'Bx_dx': None, 'Bx_dy': None,
         'Bx_dz': None, 'By_dy': None, 'By_dz': None, 'fx': None, 'fy': None,
         'fz': None, 'tx': None, 'ty': None, 'tz': None},
        {'X': interval[-0.02, -0.02], 'Y': interval[0, 0], 'Z': interval[0.085, 0.085], 'm': interval[0.145, 0.145],
         'alpha': interval[math.pi/2, math.pi/2], 'beta': beta2,
         'Bx': None, 'By': True, 'Bz': True, 'Bx_dx': None, 'Bx_dy': None,
         'Bx_dz': None, 'By_dy': None, 'By_dz': None, 'fx': None, 'fy': None,
         'fz': None, 'tx': None, 'ty': None, 'tz': None},
    ]

    # Use affine calculation for the first target point
    A = Map_I2H_Affine_body(target_points)

    return A

# ============================================================================
# Utility Functions for Interval Analysis (Keep original functions)
# ============================================================================

def Create2DBox(xi, yi):
    """Create a 2D box for interval analysis"""
    # Initialize
    Box = [[0 for c in range(1)] for r in range(2)]

    xmin = xi[0][0]
    xmax = xi[0][1]
    ymin = yi[0][0]
    ymax = yi[0][1]
    Box[0][0] = interval[xmin, xmax]
    Box[1][0] = interval[ymin, ymax]

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

def Ws2DDetermination(InitBox, b_intervals, I_intervals, eps, max_iter=-1):
    """Determination of the 2D workspace of a eMNS using affine analysis"""
    L = []
    Lin = []
    Lout = []
    Lneg = []
    L.append(InitBox)
    iteration = 0

    while(len(L) > 0):
        # Extract intervals
        CurrentBox = L[0]
        del L[0]  # remove the element from the list

        # Build interval matrix using affine method
        beta1 = CurrentBox[0][0]
        beta2 = CurrentBox[1][0]
        Ainter = calculate_affine_A(beta1, beta2)

        # Check feasibility
        isFeasible = RobustFeasible(Ainter, I_intervals, b_intervals)

        if isFeasible:
            Lin.append(CurrentBox)
        else:
            isOut = Out(Ainter, I_intervals, b_intervals)
            if isOut:
                Lout.append(CurrentBox)
            else:
                if (beta1[0][1] - beta1[0][0]) > eps and (beta2[0][1] - beta2[0][0]) > eps:
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
# Main Application
# ============================================================================

def main():
    """Main function for affine analysis of 10-coil system"""

    print("Affine analysis for the determination of the workspace of an electromagnetic navigation system (10 coils)")
    print("=" * 80)

    # Build initial box
    pmin = 0
    pmax = 2 * math.pi
    beta1 = interval[pmin, pmax]
    beta2 = interval[pmin, pmax]
    InitBox = Create2DBox(beta1, beta2)

    # Define inputs interval and task set
    I_intervals = create_I_intervals(11, custom_bounds={10: (1, 1)})
    # I_intervals = [interval[-15, 15]] * 10

    b_intervals = [
        interval[0, 0],
        interval[0.045, 0.045],
        interval[0, 0],
        interval[0.045, 0.045],
    ]

    # Define precision of bisection and run the determination algorithm
    eps = 0.1
    print(f"Running workspace determination with precision eps = {eps}")
    print("This may take some time...")

    start_time = time.time()
    L, Lin, Lout, Lneg, iteration = Ws2DDetermination(
        InitBox, b_intervals, I_intervals, eps
    )
    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    print(f"Completed after {iteration} iterations")
    print(f"Boxes in workspace: {len(Lin)}")
    print(f"Boxes out of workspace: {len(Lout)}")
    print(f"Undetermined boxes: {len(Lneg)}")

    # Plotting
    print("Creating plot...")

    fig = plt.figure(figsize=(6, 6))
    ax = fig.gca()

    # Plot rectangle in workspace (green)
    for i in range(len(Lin)):
        Box = Lin[i]
        xo = Box[0][0][0][0]
        yo = Box[1][0][0][0]
        dx = Box[0][0][0][1] - xo
        dy = Box[1][0][0][1] - yo
        rect = patches.Rectangle((xo, yo), dx, dy, linewidth=0.1, edgecolor='black',
                               facecolor='green', alpha=0.5)
        ax.add_patch(rect)

    # Plot rectangle undetermined (gray)
    for i in range(len(Lneg)):
        Box = Lneg[i]
        xo = Box[0][0][0][0]
        yo = Box[1][0][0][0]
        dx = Box[0][0][0][1] - xo
        dy = Box[1][0][0][1] - yo
        rect = patches.Rectangle((xo, yo), dx, dy, linewidth=0.1, edgecolor='black',
                               facecolor='gray', alpha=0.2)
        ax.add_patch(rect)

    # Plot rectangle out of workspace (red)
    for i in range(len(Lout)):
        Box = Lout[i]
        xo = Box[0][0][0][0]
        yo = Box[1][0][0][0]
        dx = Box[0][0][0][1] - xo
        dy = Box[1][0][0][1] - yo
        rect = patches.Rectangle((xo, yo), dx, dy, linewidth=0.1, edgecolor='black',
                               facecolor='red', alpha=0.2)
        ax.add_patch(rect)

    # Format plot
    ax.axis('equal')
    ax.set_title('Workspace determination by affine analysis (10 coils)')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')

    # Set tick label font size
    ax.tick_params(axis='both', which='major', labelsize=12)  # Change 12 to your desired size

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='green', alpha=0.5, label='In workspace'),
                      Patch(facecolor='gray', alpha=0.2, label='Undetermined'),
                      Patch(facecolor='red', alpha=0.2, label='Out of workspace')]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi], ['0', 'π/2', 'π', '3π/2', '2π'])
    plt.yticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi], ['0', 'π/2', 'π', '3π/2', '2π'])

    plt.tight_layout()
    plt.grid()
    plt.show()

    print("Analysis complete!")

if __name__ == "__main__":
    main()
