#!/usr/bin/env python3
# Run from the repository root: python -m analysis.affine.two_coil
"""
Affine analysis for the determination of the workspace of an electromagnetic navigation system
"""
# Standard library imports
import os
import sys
import math
import yaml
from itertools import combinations
from itertools import product
import time

# Scientific computing imports
import numpy as np
from numpy.linalg import lstsq
import pandas as pd
from scipy.spatial.transform import Rotation as R
from scipy.linalg import null_space
from scipy.spatial import ConvexHull
from scipy.interpolate import griddata
from scipy.optimize import linprog

# Plotting imports
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib import cm
import matplotlib as mpl
from matplotlib.patches import Circle
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)
from matplotlib.colors import LightSource

# Interval analysis imports - use our custom module
from emns.numerics.interval import interval, imath, fpu
from emns.workspace.checks import Feasible, Out, RobustFeasible
from emns.numerics import affine as ca

# ============================================================================
# Affine Arithmetic Engine is now in custom_affine.py
# ============================================================================

# ============================================================================
# Utility Functions for Affine Analysis are now in custom_affine.py
# ============================================================================

def AffineEvaluateActuationMatrix(xi, yi, zi, parsed_yaml_file):
    """Affine evaluation of the actuation matrix of a system"""
    # Reset symbol index for each calculation to avoid symbol explosion
    ca.reset_symbol_index()

    coil_list = parsed_yaml_file.get("Coil_List")
    n_coil = len(coil_list)

    # Initialize affine actuation matrix
    Blist_affine = [[ca.Affine(0) for c in range(n_coil)] for r in range(2)]

    # Convert inputs to affine forms
    xi_affine = ca.Affine.from_interval(xi)
    yi_affine = ca.Affine.from_interval(yi)
    zi_affine = ca.Affine.from_interval(zi)

    # Iterate on coils
    for i in range(0, n_coil):
        # Get actual coil
        coil_i = parsed_yaml_file.get("Coil_"+str(i))
        n_sources = len(coil_i.get("Source_List"))

        # Initialize coil contrib terms
        Bcoil_x = ca.Affine(0.)
        Bcoil_y = ca.Affine(0.)

        # Iterate on sources for the coil
        for j in range(0, n_sources):
            # Get source parameters
            src_j = coil_i.get("Src_"+str(j))
            A = src_j.get("A_Coeff")
            B = src_j.get("B_Coeff")
            src_dir = src_j.get("Source_Direction")
            src_pos = src_j.get("Source_Position")

            # Transform position in source frame using affine arithmetic
            xs = xi_affine - ca.Affine(src_pos[0])
            ys = yi_affine - ca.Affine(src_pos[1])
            zs = zi_affine - ca.Affine(src_pos[2])

            # Get norm from position vector for affine forms
            rnorm = ca.AffineEvaluateVectorNorm(xs, ys, zs)
            xni, yni, zni = ca.AffineEvaluateNormalizedVector(xs, ys, zs)

            # Affine evaluation of cos from inclination angle
            gamma = (ca.Affine(src_dir[0]) * xni) + (ca.Affine(src_dir[1]) * yni) + (ca.Affine(src_dir[2]) * zni)

            # Consider only the dipole term
            P, Pp = ca.AffineEvaluateLegendrePolyNomial(gamma, 1)

            # Simplified expression using affine arithmetic
            rnorm_cubed = rnorm * rnorm * rnorm

            # Check for division by very small numbers
            rnorm_cubed_interval = rnorm_cubed.to_interval()
            if rnorm_cubed_interval[0][0] <= 1e-9:
                rnorm_cubed = ca.Affine(max(rnorm_cubed.center, 1e-9))

            Bsrc_x = ca.Affine(B[0]) * (ca.Affine(3) * gamma * xni - ca.Affine(src_dir[0])) / rnorm_cubed
            Bsrc_y = ca.Affine(B[0]) * (ca.Affine(3) * gamma * yni - ca.Affine(src_dir[1])) / rnorm_cubed

            # Add source contribution to coil
            Bcoil_x = Bcoil_x + Bsrc_x
            Bcoil_y = Bcoil_y + Bsrc_y

        Blist_affine[0][i] = Bcoil_x
        Blist_affine[1][i] = Bcoil_y

    # Convert affine results back to intervals for compatibility with existing checking functions
    Blist = [[0 for c in range(n_coil)] for r in range(2)]
    for i in range(2):
        for j in range(n_coil):
            Blist[i][j] = Blist_affine[i][j].to_interval()

    return Blist

# ============================================================================
# Utility Functions for Interval Analysis (Keep original functions)
# ============================================================================

def IntervalEvaluateVectorNorm(xi, yi, zi):
    """Interval evaluation of the norm of an interval vector [xi, yi, zi]"""
    inorm = imath.sqrt(xi**2 + yi**2 + zi**2)
    return inorm

def IntervalEvaluateNormalizedVector(xi, yi, zi):
    """Interval evaluation of the normalize vector from a vector [xi, yi, zi]"""
    inorm = IntervalEvaluateVectorNorm(xi, yi, zi)
    xni = (xi / inorm) & interval[-1., 1.]
    yni = (yi / inorm) & interval[-1., 1.]
    zni = (zi / inorm) & interval[-1., 1.]
    return xni, yni, zni

def IntervalEvaluateLegendrePolyNomial(x, n):
    """Interval evaluation of the Legendre polynomial P and its derivative Pp of order n at x"""
    if n == 0:
        P = 1
        Pp = 0
    elif n == 1:
        P = x
        Pp = 1
    elif n == 2:
        P = 0.5 * (3*x**2 - 1)
        Pp = 3*x
    return P, Pp

def IntervalEvaluateActuationMatrix(xi, yi, zi, parsed_yaml_file):
    """Interval evaluation of the actuation matrix of a system"""
    coil_list = parsed_yaml_file.get("Coil_List")
    n_coil = len(coil_list)

    # Initialize interval actuation matrix
    Blist = [[0 for c in range(n_coil)] for r in range(2)]

    # Iterate on coils
    for i in range(0, n_coil):
        # Get actual coil
        coil_i = parsed_yaml_file.get("Coil_"+str(i))
        n_sources = len(coil_i.get("Source_List"))

        # Initialize coil contrib terms
        Bcoil_x = 0.
        Bcoil_y = 0.

        # Iterate on sources for the coil
        for j in range(0, n_sources):
            # Get source parameters
            src_j = coil_i.get("Src_"+str(j))
            A = src_j.get("A_Coeff")
            B = src_j.get("B_Coeff")
            src_dir = src_j.get("Source_Direction")
            src_pos = src_j.get("Source_Position")

            # Transform position in source frame
            xs = (xi - src_pos[0])
            ys = (yi - src_pos[1])
            zs = (zi - src_pos[2])

            # Get norm from position vector for intervals
            rnorm = IntervalEvaluateVectorNorm(xs, ys, zs)
            xni, yni, zni = IntervalEvaluateNormalizedVector(xs, ys, zs)

            # Interval evaluation of cos from inclination angle
            gamma = (src_dir[0] * xni) + (src_dir[1] * yni) + (src_dir[2] * zni)

            # Consider only the dipole term
            P, Pp = IntervalEvaluateLegendrePolyNomial(gamma, 1)

            # Simplified expression
            Bsrc_x = B[0] * (3 * gamma * xni - src_dir[0]) / (rnorm ** 3)
            Bsrc_y = B[0] * (3 * gamma * yni - src_dir[1]) / (rnorm ** 3)

            # Add source contribution to coil
            Bcoil_x = Bcoil_x + Bsrc_x
            Bcoil_y = Bcoil_y + Bsrc_y

        Blist[0][i] = Bcoil_x
        Blist[1][i] = Bcoil_y

    return Blist

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

def Ws2DDetermination(InitBox, bmin, bmax, Imin, Imax, eps, parsed_yaml_file, zoff, max_iter=-1, use_affine=True):
    """Determination of the 2D workspace of a eMNS using affine or interval analysis"""
    L = []
    Lin = []
    Lout = []
    Lneg = []
    L.append(InitBox)
    iteration = 0
    I_intervals = [interval[Imin, Imax]] * 8
    b_intervals = [
        interval[bmin, bmax],
        interval[bmin, bmax],
    ]

    # Choose evaluation function based on method
    evaluate_matrix = AffineEvaluateActuationMatrix if use_affine else IntervalEvaluateActuationMatrix

    while(len(L) > 0):
        # Extract intervals
        CurrentBox = L[0]
        del L[0]  # remove the element from the list

        # Build interval matrix using chosen method
        xi = CurrentBox[0][0]
        yi = CurrentBox[1][0]
        Ainter = evaluate_matrix(xi, yi, zoff, parsed_yaml_file)

        # Check feasibility using auto-accelerated methods
        isFeasible = Feasible(Ainter, I_intervals, b_intervals)

        if isFeasible:
            Lin.append(CurrentBox)
        else:
            isOut = Out(Ainter, I_intervals, b_intervals)
            if isOut:
                Lout.append(CurrentBox)
            else:
                if (xi[0][1] - xi[0][0]) > eps and (yi[0][1] - yi[0][0]) > eps:
                    Bisected_box = BisectBox(CurrentBox)
                    L.extend(Bisected_box)
                else:
                    Lneg.append(CurrentBox)

        iteration = iteration + 1

        if iteration >= max_iter and max_iter >= 1:
            print('Max iter reached')
            break

    return L, Lin, Lout, Lneg, iteration

def Coils2DPatch(coil_length, coil_width, cal_path_mns):
    """Plot 2D coils"""
    with open(cal_path_mns, 'r') as yaml_file:
        parsed_yaml_file = yaml.load(yaml_file, Loader=yaml.FullLoader)

    patches_list = []

    coil_list = parsed_yaml_file.get("Coil_List")
    n_coil = len(coil_list)

    # Iterate on coils
    for i in range(0, n_coil):
        # Get actual coil
        coil_i = parsed_yaml_file.get("Coil_"+str(i))
        n_sources = len(coil_i.get("Source_List"))

        # Iterate on sources for the coil
        for j in range(0, n_sources):
            # Get source parameters
            src_j = coil_i.get("Src_"+str(j))
            src_dir = src_j.get("Source_Direction")
            src_pos = src_j.get("Source_Position")

            # Transform position in source frame
            xpos = src_pos[0]
            ypos = src_pos[1]
            zpos = src_pos[2]

            angle = np.arctan2(src_dir[1], src_dir[0])

            # Coil
            coil = patches.Rectangle((0, 0), coil_length, coil_width, color="black", alpha=0.80)
            toff = mpl.transforms.Affine2D().translate(-coil_length/2, -coil_width/2)
            r = mpl.transforms.Affine2D().rotate_deg(angle*180/math.pi)
            t = mpl.transforms.Affine2D().translate(xpos, ypos)
            coil.set_transform(toff + r + t)

            # Center
            center = patches.Circle((xpos, ypos), coil_width/10, color="blue", alpha=1.)

            patches_list.append(coil)
            patches_list.append(center)

    p = PatchCollection(patches_list, alpha=0.4)
    return p

# ============================================================================
# Main Application
# ============================================================================

def main():
    """Main function reproducing the affine analysis"""

    print("Affine analysis for the determination of the workspace of an electromagnetic navigation system")
    print("=" * 80)

    # Load systems calibration
    # cal_path_mns = 'models/mns_2coils_90deg.yaml'
    cal_path_mns = 'models/mns_4coils_1.yaml'

    # Check if the calibration file exists
    if not os.path.exists(cal_path_mns):
        print(f"Error: Calibration file {cal_path_mns} not found!")
        return

    print(f"Loading system calibration from: {cal_path_mns}")

    # Build initial box
    pmin = -0.1
    pmax = 0.1
    xi = interval[pmin, pmax]
    yi = interval[pmin, pmax]
    z = 0.
    InitBox = Create2DBox(xi, yi)

    # Define inputs interval and task set
    Imin = -1
    Imax = 1
    bmin = -0.08
    bmax = 0.08

    # Load system model
    print("Loading system model...")
    with open(cal_path_mns, 'r') as yaml_file:
        parsed_yaml_file = yaml.load(yaml_file, Loader=yaml.FullLoader)

    # Define precision of bisection and run the determination algorithm
    eps = 0.009
    print(f"Running workspace determination with precision eps = {eps}")
    print("This may take some time...")

    # Run both methods for comparison
    print("\n--- Running Affine Analysis ---")
    start_time = time.time()
    z_interval = interval[z, z]  # Convert scalar z to interval
    L_affine, Lin_affine, Lout_affine, Lneg_affine, iteration_affine = Ws2DDetermination(
        InitBox, bmin, bmax, Imin, Imax, eps, parsed_yaml_file, z_interval, use_affine=True
    )
    end_time = time.time()
    time_affine = end_time - start_time
    print(f"Affine method took: {time_affine:.2f} seconds")
    print(f"Completed after {iteration_affine} iterations")
    print(f"Boxes in workspace: {len(Lin_affine)}")
    print(f"Boxes out of workspace: {len(Lout_affine)}")
    print(f"Undetermined boxes: {len(Lneg_affine)}")

    print("\n--- Running Interval Analysis for Comparison ---")
    start_time = time.time()
    L_interval, Lin_interval, Lout_interval, Lneg_interval, iteration_interval = Ws2DDetermination(
        InitBox, bmin, bmax, Imin, Imax, eps, parsed_yaml_file, z_interval, use_affine=False
    )
    end_time = time.time()
    time_interval = end_time - start_time
    print(f"Interval method took: {time_interval:.2f} seconds")
    print(f"Completed after {iteration_interval} iterations")
    print(f"Boxes in workspace: {len(Lin_interval)}")
    print(f"Boxes out of workspace: {len(Lout_interval)}")
    print(f"Undetermined boxes: {len(Lneg_interval)}")

    # Plotting comparison
    print("Creating comparison plot...")

    # Coil dimensions for visualization
    coil_length = 0.05
    coil_width = 0.03

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle('Workspace Determination Comparison', fontsize=16)

    # Plot Affine results
    for i in range(len(Lin_affine)):
        Box = Lin_affine[i]
        xo = Box[0][0][0][0]
        yo = Box[1][0][0][0]
        dx = Box[0][0][0][1] - xo
        dy = Box[1][0][0][1] - yo
        rect = patches.Rectangle((xo, yo), dx, dy, linewidth=1, edgecolor='black',
                               facecolor='green', alpha=0.5)
        ax1.add_patch(rect)

    for i in range(len(Lneg_affine)):
        Box = Lneg_affine[i]
        xo = Box[0][0][0][0]
        yo = Box[1][0][0][0]
        dx = Box[0][0][0][1] - xo
        dy = Box[1][0][0][1] - yo
        rect = patches.Rectangle((xo, yo), dx, dy, linewidth=1, edgecolor='black',
                               facecolor='gray', alpha=0.2)
        ax1.add_patch(rect)

    for i in range(len(Lout_affine)):
        Box = Lout_affine[i]
        xo = Box[0][0][0][0]
        yo = Box[1][0][0][0]
        dx = Box[0][0][0][1] - xo
        dy = Box[1][0][0][1] - yo
        rect = patches.Rectangle((xo, yo), dx, dy, linewidth=1, edgecolor='black',
                               facecolor='red', alpha=0.2)
        ax1.add_patch(rect)

    # Plot Interval results
    for i in range(len(Lin_interval)):
        Box = Lin_interval[i]
        xo = Box[0][0][0][0]
        yo = Box[1][0][0][0]
        dx = Box[0][0][0][1] - xo
        dy = Box[1][0][0][1] - yo
        rect = patches.Rectangle((xo, yo), dx, dy, linewidth=1, edgecolor='black',
                               facecolor='green', alpha=0.5)
        ax2.add_patch(rect)

    for i in range(len(Lneg_interval)):
        Box = Lneg_interval[i]
        xo = Box[0][0][0][0]
        yo = Box[1][0][0][0]
        dx = Box[0][0][0][1] - xo
        dy = Box[1][0][0][1] - yo
        rect = patches.Rectangle((xo, yo), dx, dy, linewidth=1, edgecolor='black',
                               facecolor='gray', alpha=0.2)
        ax2.add_patch(rect)

    for i in range(len(Lout_interval)):
        Box = Lout_interval[i]
        xo = Box[0][0][0][0]
        yo = Box[1][0][0][0]
        dx = Box[0][0][0][1] - xo
        dy = Box[1][0][0][1] - yo
        rect = patches.Rectangle((xo, yo), dx, dy, linewidth=1, edgecolor='black',
                               facecolor='red', alpha=0.2)
        ax2.add_patch(rect)

    # Plot coils on both plots
    try:
        coll1 = Coils2DPatch(coil_length, coil_width, cal_path_mns)
        coll2 = Coils2DPatch(coil_length, coil_width, cal_path_mns)
        ax1.add_collection(coll1)
        ax2.add_collection(coll2)
    except Exception as e:
        print(f"Warning: Could not plot coils: {e}")

    # Format plots
    for ax in [ax1, ax2]:
        ax.axis('equal')
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.grid(True, linestyle='--', alpha=0.3)

    ax1.set_title(f'Affine Arithmetic\n({time_affine:.2f}s, {len(Lneg_affine)} boundary boxes)')
    ax2.set_title(f'Standard Interval Arithmetic\n({time_interval:.2f}s, {len(Lneg_interval)} boundary boxes)')

    # Add legend
    legend_elements = [patches.Patch(facecolor='green', alpha=0.5, label='In workspace'),
                      patches.Patch(facecolor='gray', alpha=0.2, label='Undetermined'),
                      patches.Patch(facecolor='red', alpha=0.2, label='Out of workspace')]
    ax1.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    plt.show()

    print("Analysis complete!")
    print(f"\nComparison Summary:")
    print(f"Affine method: {len(Lneg_affine)} undetermined boxes in {time_affine:.2f}s")
    print(f"Interval method: {len(Lneg_interval)} undetermined boxes in {time_interval:.2f}s")
    print(f"Improvement: {len(Lneg_interval) - len(Lneg_affine)} fewer undetermined boxes")



if __name__ == "__main__":
    # Run the standard affine analysis
    main()
