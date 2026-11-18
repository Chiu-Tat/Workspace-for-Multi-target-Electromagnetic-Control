#!/usr/bin/env python3
# Run from the repository root: python -m cases.case_4_three_dof_manipulator.workspace
"""
Affine analysis for the determination of the workspace of an electromagnetic navigation system (10 coils)
"""

# Standard library imports
import math
from itertools import product
import time
import multiprocessing as mp
from functools import partial
import os
import json
# Scientific computing imports
import numpy as np
# Plotting imports
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
# Interval analysis imports - use our custom module
from emns.numerics.interval import interval, imath, fpu
from emns.workspace.checks import Feasible, Out, RobustFeasible, create_I_intervals, create_b_intervals, BisectBox, Create2DBox
from emns.numerics import affine as ca
from emns.workspace.affine import Map_I2H_Affine_body, robot_arm_kinematics

# ============================================================================
# Affine version of WS_lib_Affine functions
# ============================================================================

def calculate_affine_A(alpha1, beta1, beta2):
    """Affine version of calculate_interval_A"""
    X1, Y1, Z1, X2, Y2, Z2, alpha2w, beta2w = robot_arm_kinematics(alpha1, beta1, beta2)
    target_points = [
        {'X': X1, 'Y': Y1, 'Z': Z1, 'm': interval[0.145, 0.145],
         'alpha': alpha1, 'beta': beta1,
         'Bx': True, 'By': True, 'Bz': True, 'Bx_dx': None, 'Bx_dy': None,
         'Bx_dz': None, 'By_dy': None, 'By_dz': None, 'fx': None, 'fy': None,
         'fz': None, 'tx': None, 'ty': None, 'tz': None},
        {'X': X2, 'Y': Y2, 'Z': Z2, 'm': interval[0.05, 0.05],
         'alpha': alpha2w, 'beta': beta2w,
         'Bx': True, 'By': True, 'Bz': True, 'Bx_dx': None, 'Bx_dy': None,
         'Bx_dz': None, 'By_dy': None, 'By_dz': None, 'fx': None, 'fy': None,
         'fz': None, 'tx': None, 'ty': None, 'tz': None},
    ]

    # Use affine calculation for the first target point
    A = Map_I2H_Affine_body(target_points)

    return A

# ============================================================================
# Parallel Worker Functions (must be top-level)
# ============================================================================

def _is_valid_for_all_orientations_worker(pos_box, full_angle_range, local_memo, I_intervals, b_intervals, eps_b3):
    """Check if position box is valid for ALL orientations in the given range.
    full_angle_range is a single interval object (not a list)."""
    from collections import deque

    xi, yi = pos_box[0][0], pos_box[1][0]
    # Create memoization key
    pos_key = (xi[0][0], xi[0][1], yi[0][0], yi[0][1], full_angle_range[0][0], full_angle_range[0][1])
    if pos_key in local_memo:
        return local_memo[pos_key]

    inner_queue = deque([full_angle_range])

    while inner_queue:
        current_angle_range = inner_queue.popleft()

        # Calculate interval matrix A for current position and orientation box
        Ainter = calculate_affine_A(xi, yi, current_angle_range)

        # Check if this orientation range is completely outside workspace
        isOut = Out(Ainter, I_intervals, b_intervals)
        if isOut:
            local_memo[pos_key] = 'OUT'
            return 'OUT'

        # Check if this orientation range is completely inside workspace
        isFeasible = Feasible(Ainter, I_intervals, b_intervals)
        if isFeasible:
            # This angle range is feasible, continue checking other ranges
            continue

        # If orientation range is too small, mark as unknown
        if (current_angle_range[0][1] - current_angle_range[0][0]) < eps_b3:
            local_memo[pos_key] = 'UNKNOWN'
            return 'UNKNOWN'

        # Bisect the orientation interval
        angle_min = current_angle_range[0][0]
        angle_max = current_angle_range[0][1]
        angle_mid = (angle_min + angle_max) / 2

        # Add bisected orientation intervals to queue
        inner_queue.append(interval[angle_min, angle_mid])
        inner_queue.append(interval[angle_mid, angle_max])

    # If we've checked all orientation subdivisions without finding any outside regions
    local_memo[pos_key] = 'IN'
    return 'IN'

def _process_single_pos_box_worker(pos_box, angle_interval_p, memo_p, I_intervals, b_intervals, eps_b3):
    """Worker function to process a single position box."""
    # angle_interval_p is a list containing one interval object
    angle_range = angle_interval_p[0]  # Get the interval object from the list
    status = _is_valid_for_all_orientations_worker(pos_box, angle_range, memo_p, I_intervals, b_intervals, eps_b3)
    return status, pos_box

# ============================================================================
# Utility Functions for Interval Analysis (Keep original functions)
# ============================================================================

def Ws2DDetermination(InitBox, b_intervals, I_intervals, eps_b12, eps_b3, angle_interval, max_iter=-1):
    """
    Determination of the 2D workspace of a eMNS using interval analysis with independent constraints
    Modified to handle orientation intervals - finds boxes where ALL orientations in Angle_interval satisfy b_intervals
    """
    from collections import deque

    Lin = []
    Lout = []
    Lneg = []
    outer_queue = deque([InitBox])
    iteration = 0
    memo = {}  # Memoization to avoid recomputing same position boxes

    def is_valid_for_all_orientations(pos_box, full_angle_range):
        """Check if position box is valid for ALL orientations in the given range"""
        # Create memoization key
        xi, yi = pos_box[0][0], pos_box[1][0]
        # full_angle_range is an interval object, access bounds directly
        pos_key = (xi[0][0], xi[0][1], yi[0][0], yi[0][1], full_angle_range[0][0], full_angle_range[0][1])
        if pos_key in memo:
            return memo[pos_key]

        inner_queue = deque([full_angle_range])

        while inner_queue:
            current_angle_range = inner_queue.popleft()

            # Calculate interval matrix A for current position and orientation box
            Ainter = calculate_affine_A(xi, yi, current_angle_range)

            # Check if this orientation range is completely outside workspace
            isOut = Out(Ainter, I_intervals, b_intervals)
            if isOut:
                memo[pos_key] = 'OUT'
                return 'OUT'

            # Check if this orientation range is completely inside workspace
            isFeasible = Feasible(Ainter, I_intervals, b_intervals)
            if isFeasible:
                # This angle range is feasible, continue checking other ranges
                continue

            # If orientation range is too small, mark as unknown
            if (current_angle_range[0][1] - current_angle_range[0][0]) < eps_b3:
                memo[pos_key] = 'UNKNOWN'
                return 'UNKNOWN'

            # Bisect the orientation interval
            angle_min = current_angle_range[0][0]
            angle_max = current_angle_range[0][1]
            angle_mid = (angle_min + angle_max) / 2

            # Add bisected orientation intervals to queue
            inner_queue.append(interval[angle_min, angle_mid])
            inner_queue.append(interval[angle_mid, angle_max])

        # If we've checked all orientation subdivisions without finding any outside regions
        memo[pos_key] = 'IN'
        return 'IN'

    # Main outer loop for position boxes
    while outer_queue:
        if iteration >= max_iter and max_iter >= 1:
            print('Max iter reached')
            break

        pos_box = outer_queue.popleft()
        iteration += 1

        # Check if this position box is valid for all orientations in angle_interval
        # angle_interval is a list containing one interval object
        angle_range = angle_interval[0]  # Get the interval object from the list
        status = is_valid_for_all_orientations(pos_box, angle_range)

        if status == 'IN':
            Lin.append(pos_box)
        elif status == 'OUT':
            Lout.append(pos_box)
        else:  # UNKNOWN
            # Extract position intervals
            xi = pos_box[0][0]
            yi = pos_box[1][0]

            # Check if position box is small enough to stop subdivision
            if (xi[0][1] - xi[0][0]) < eps_b12 and (yi[0][1] - yi[0][0]) < eps_b12:
                Lneg.append(pos_box)
            else:
                # Bisect position box
                Bisected_box = BisectBox(pos_box)
                outer_queue.extend(Bisected_box)

    return [], Lin, Lout, Lneg, iteration

def Ws2DDetermination_Parallel(InitBox, b_intervals, I_intervals, eps_b12, eps_b3, angle_interval, max_iter=-1, num_processes=None):
    """
    Parallel determination of the 2D workspace of a eMNS using interval analysis.
    """
    if num_processes is None:
        num_processes = min(mp.cpu_count(), 8)

    print(f"Using {num_processes} parallel processes")

    Lin = []
    Lout = []
    Lneg = []
    L = [InitBox]
    iteration = 0
    manager = mp.Manager()
    memo = manager.dict()  # Use a thread-safe dictionary for memoization

    worker_func = partial(_process_single_pos_box_worker,
                          angle_interval_p=angle_interval,
                          memo_p=memo,
                          I_intervals=I_intervals,
                          b_intervals=b_intervals,
                          eps_b3=eps_b3)

    with mp.Pool(processes=num_processes) as pool:
        while L:
            if max_iter >= 1 and iteration >= max_iter:
                print('Max iter reached')
                Lneg.extend(L)
                L = []
                break

            batch_size = min(len(L), num_processes * 4)
            current_boxes = L[:batch_size]
            L = L[batch_size:]

            results = pool.map(worker_func, current_boxes)

            boxes_to_bisect = []
            for status, pos_box in results:
                if status == 'IN':
                    Lin.append(pos_box)
                elif status == 'OUT':
                    Lout.append(pos_box)
                else:  # UNKNOWN
                    xi = pos_box[0][0]
                    yi = pos_box[1][0]
                    if (xi[0][1] - xi[0][0]) < eps_b12 and (yi[0][1] - yi[0][0]) < eps_b12:
                        Lneg.append(pos_box)
                    else:
                        boxes_to_bisect.append(pos_box)

            for box in boxes_to_bisect:
                bisected_boxes = BisectBox(box)
                L.extend(bisected_boxes)

            iteration += len(current_boxes)

    return [], Lin, Lout, Lneg, iteration

# ============================================================================
# Main Application
# ============================================================================

def main():
    """Main function for affine analysis of 10-coil system"""

    print("Affine analysis for the determination of the workspace of an electromagnetic navigation system (10 coils)")
    print("=" * 80)

    # Build initial box
    alpha1 = interval[1/6 * math.pi, 5/6 * math.pi]
    beta1 = interval[-5/6 * math.pi, -1/6 * math.pi]
    InitBox = Create2DBox(alpha1, beta1)

    beta2 = [interval[0, 30/180 * math.pi]]

    # Define inputs interval and task set
    I_intervals = create_I_intervals(11, custom_bounds={10: (1, 1)})
    # I_intervals = [interval[-15, 15]] * 10

    b_intervals = [
        interval[0, 0],
        interval[0, 0],
        interval[0.05, 0.05],
        interval[0, 0],
        interval[0, 0],
        interval[0.03, 0.03],
    ]

    # Define precision of bisection and run the determination algorithm
    eps_b12 = 0.1
    eps_b3 = 0.1

    num_processes = min(mp.cpu_count(), 16)
    print(f"core number: {mp.cpu_count()}")
    print(f"Running parallel workspace determination with precision eps_b12 = {eps_b12}, eps_b3 = {eps_b3}")
    print(f"Using {num_processes} processes")
    print("This may take some time...")

    start_time = time.time()
    L, Lin, Lout, Lneg, iteration = Ws2DDetermination_Parallel(
        InitBox, b_intervals, I_intervals, eps_b12, eps_b3, beta2, max_iter=-1, num_processes=num_processes
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
    with open('Results/results_case4_0503.json', 'w') as f:
        json.dump(results_data, f, indent=4)
    print("Results saved to Results/results_case4_0503.json")

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
                               facecolor='salmon', alpha=0.2)
        ax.add_patch(rect)

    # Format plot
    ax.axis('equal')
    ax.set_title('Workspace determination by affine analysis (10 coils)')
    ax.set_xlabel(r'$\alpha_1$ (rad)', fontsize=20)
    ax.set_ylabel(r'$\beta_1$ (rad)', fontsize=20)
    # ax.set_zlabel('z (m)')

    # Set tick label font size
    ax.tick_params(axis='both', which='major', labelsize=18)  # Change 12 to your desired size

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='green', alpha=0.5, label='In workspace'),
                    #   Patch(facecolor='gray', alpha=0.2, label='Undetermined'),
                    #   Patch(facecolor='red', alpha=0.2, label='Out of workspace')
                      ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=20)

    ticksx = [1/6 * math.pi, 1/3 * math.pi, 1/2 * math.pi, 2/3 * math.pi, 5/6 * math.pi]
    labelsx = ['π/6', 'π/3', 'π/2', '2π/3', '5π/6']

    ticksy = [-5/6 * math.pi, -2/3 * math.pi, -1/2 * math.pi, -1/3 * math.pi, -1/6 * math.pi]
    labelsy = ['-5π/6', '-2π/3', '-π/2', '-π/3', '-π/6']

    ax.set_xticks(ticksx, labelsx)
    ax.set_yticks(ticksy, labelsy)
    # ax.set_zticks(ticks, labels)

    plt.tight_layout()
    plt.show()

    print("Analysis complete!")

if __name__ == "__main__":
    mp.freeze_support()
    main()
