#!/usr/bin/env python3
# Run from the repository root: python -m cases.case_2_magnet_translation.workspace_vertical
"""
Affine analysis for the determination of the workspace of an electromagnetic navigation system (10 coils)
"""

# Standard library imports
import os
import json
import sys
import math
import yaml
from itertools import combinations
from itertools import product
import time
import multiprocessing as mp
from functools import partial

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
from emns.workspace.checks import Feasible, Out, RobustFeasible, create_I_intervals, create_b_intervals, Create2DBox, BisectBox
from emns.numerics import affine as ca
from emns.workspace.affine import Map_I2H_Affine

# ============================================================================
# Affine version of WS_lib_Affine functions
# ============================================================================

def calculate_affine_A(xi, yi, zoff):
    """Affine version of calculate_interval_A"""
    target_points = [
        {'X': interval[-0.014,-0.014], 'Y': interval[-0.003, -0.003], 'Z': xi, 'm': interval[0.145, 0.145],
         'alpha': interval[0, 0], 'beta': interval[0.00, 0.00],
         'Bx': None, 'By': None, 'Bz': None, 'Bx_dx': None, 'Bx_dy': None,
         'Bx_dz': None, 'By_dy': None, 'By_dz': None, 'fx': None, 'fy': None,
         'fz': True, 'tx': None, 'ty': None, 'tz': None},
         {'X': interval[0.014,0.014], 'Y': interval[-0.003, -0.003], 'Z': yi, 'm': interval[0.145, 0.145],
         'alpha': interval[0,0], 'beta': interval[0.00, 0.00],
         'Bx': None, 'By': None, 'Bz': None, 'Bx_dx': None, 'Bx_dy': None,
         'Bx_dz': None, 'By_dy': None, 'By_dz': None, 'fx': None, 'fy': None,
         'fz': True, 'tx': None, 'ty': None, 'tz': None}
    ]

    # Use affine calculation for the first target point
    A = Map_I2H_Affine(target_points)

    # A = A[:,:-1]

    return A

# ============================================================================
# Parallel Processing Functions
# ============================================================================

def process_single_box_case2(box, b_intervals, I_intervals, eps, zoff):
    """Worker function to process a single box for Case 2"""
    try:
        xi = box[0][0]
        yi = box[1][0]
        Ainter = calculate_affine_A(xi, yi, zoff)

        isFeasible = RobustFeasible(Ainter, I_intervals, b_intervals)

        if isFeasible:
            return 'feasible', box
        else:
            isOut = Out(Ainter, I_intervals, b_intervals)
            if isOut:
                return 'out', box
            else:
                if (xi[0][1] - xi[0][0]) > eps and (yi[0][1] - yi[0][0]) > eps:
                    return 'bisect', box
                else:
                    return 'undetermined', box
    except Exception as e:
        return 'undetermined', box

def Ws2DDetermination_Parallel(InitBox, b_intervals, I_intervals, eps, zoff, max_iter=-1, num_processes=None):
    """Parallel version of 2D workspace determination for Case 2"""
    if num_processes is None:
        num_processes = min(mp.cpu_count(), 8)

    print(f"Using {num_processes} parallel processes")

    L = []
    Lin = []
    Lout = []
    Lneg = []
    L.append(InitBox)
    iteration = 0

    worker_func = partial(process_single_box_case2,
                         b_intervals=b_intervals,
                         I_intervals=I_intervals,
                         eps=eps,
                         zoff=zoff)

    with mp.Pool(processes=num_processes) as pool:
        while(len(L) > 0):
            if len(L) >= num_processes:
                current_boxes = L[:num_processes]
                L = L[num_processes:]
                results = pool.map(worker_func, current_boxes)
            else:
                current_boxes = L
                L = []
                results = pool.map(worker_func, current_boxes)

            boxes_to_bisect = []
            for result_type, box in results:
                if result_type == 'feasible':
                    Lin.append(box)
                elif result_type == 'out':
                    Lout.append(box)
                elif result_type == 'bisect':
                    boxes_to_bisect.append(box)
                else:
                    Lneg.append(box)

            for box in boxes_to_bisect:
                bisected_boxes = BisectBox(box)
                L.extend(bisected_boxes)

            iteration += len(current_boxes)

            if iteration >= max_iter and max_iter >= 1:
                print('Max iter reached')
                break

    return L, Lin, Lout, Lneg, iteration

# ============================================================================
# Main Application
# ============================================================================

def main():
    """Main function for affine analysis of 10-coil system"""

    print("Parallel Affine analysis for the determination of the workspace of an electromagnetic navigation system (10 coils)")
    print("=" * 80)

    # Build initial box
    pmin = -0.036 + 0.085
    pmax = 0.0268 + 0.085
    xi = interval[pmin, pmax]
    yi = interval[pmin, pmax]
    z = -0.01
    InitBox = Create2DBox(xi, yi)

    # Define inputs interval and task set
    I_intervals = create_I_intervals(11, custom_bounds={10: (1, 1)})
    # I_intervals = [interval[-15, 15]] * 10

    b_intervals = [
        interval[-0.1, 0.1],
        interval[-0.1, 0.1],
    ]

    # Define precision of bisection and run the determination algorithm
    eps = 0.001
    num_processes = min(mp.cpu_count(), 16)
    print(f"core number: {mp.cpu_count()}")
    print(f"Running parallel workspace determination with precision eps = {eps}")
    print(f"Using {num_processes} processes")
    print("This may take some time...")

    start_time = time.time()
    L, Lin, Lout, Lneg, iteration = Ws2DDetermination_Parallel(
        InitBox, b_intervals, I_intervals, eps, z, num_processes=num_processes
    )
    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.2f} seconds")

    print(f"Completed after {iteration} iterations")
    print(f"Boxes in workspace: {len(Lin)}")
    print(f"Boxes out of workspace: {len(Lout)}")
    print(f"Undetermined boxes: {len(Lneg)}")

    # Save results
    print("Saving results...")
    if not os.path.exists('Results'):
        os.makedirs('Results')
    def box_to_list(box):
        # Converts a box from its nested interval structure to a simple list [xmin, xmax, ymin, ymax]
        return [box[0][0][0][0], box[0][0][0][1], box[1][0][0][0], box[1][0][0][1]]
    results_data = {
        'undetermined': [box_to_list(box) for box in Lneg],
        'out_of_workspace': [box_to_list(box) for box in Lout]
    }
    with open('Results/results_case2_V01028.json', 'w') as f:
        json.dump(results_data, f, indent=4)
    print("Results saved to Results/results_case2_V01028.json")

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
                               facecolor='gray', alpha=0.2)
        ax.add_patch(rect)

    # Format plot
    ax.axis('equal')
    # ax.set_title('Workspace determination by affine analysis (10 coils)')
    ax.set_xlim(pmin, pmax)
    ax.set_ylim(pmin, pmax)
    ticksx = [0.05, 0.06,0.07, 0.08, 0.09, 0.1, 0.11]
    labelsx = ['0.05', '0.06', '0.07', '0.08', '0.09', '0.1', '0.11']
    ax.set_xticks(ticksx, labelsx)
    ticksy = [0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11]
    labelsy = ['0.05', '0.06', '0.07', '0.08', '0.09', '0.1', '0.11']
    ax.set_yticks(ticksy, labelsy)
    ax.set_xlabel(r'$z_1$ (m)', fontsize=20)
    ax.set_ylabel(r'$z_2$ (m)', fontsize=20)

    # Set tick label font size
    ax.tick_params(axis='both', which='major', labelsize=18)  # Change 12 to your desired size

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='green', alpha=0.5, label='In workspace'),
                    #   Patch(facecolor='gray', alpha=0.2, label='Undetermined'),
                    #   Patch(facecolor='red', alpha=0.2, label='Out of workspace')
                    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=20)

    plt.tight_layout()
    plt.show()

    print("Analysis complete!")

if __name__ == "__main__":
    mp.freeze_support()
    main()
