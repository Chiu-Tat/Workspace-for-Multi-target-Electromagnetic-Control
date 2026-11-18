#!/usr/bin/env python3
# Run from the repository root: python -m cases.case_3_circular_path_magnets.workspace
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
# Interval analysis imports - use our custom module
from emns.numerics.interval import interval, imath
from emns.workspace.checks import Feasible, Out, RobustFeasible, create_I_intervals, create_b_intervals, BisectBox, Create2DBox
from emns.numerics import affine as ca
from emns.workspace.affine import Map_I2H_Affine_body, robot_arm_kinematics

# ============================================================================
# Affine version of WS_lib_Affine functions
# ============================================================================

def calculate_affine_A(angle1, angle2, angle1_range):
    """Affine version of calculate_interval_A"""
    Offset = interval[0.10625, 0.10625]
    Radius = interval[0.1, 0.1]

    X1 = Offset + Radius * imath.cos(angle1)
    Y1 = Radius * imath.sin(angle1)
    Z1 = interval[0.055, 0.055]
    alpha1 = angle1_range[0]
    beta1 = interval[math.pi / 2, math.pi / 2] + angle1

    X2 = -Offset + Radius * imath.cos(angle2)
    Y2 = Radius * imath.sin(angle2)
    Z2 = interval[0.055, 0.055]
    alpha2 = angle1_range[1]
    beta2 = interval[math.pi / 2, math.pi / 2] + angle2

    target_points = [
        {'X': X1, 'Y': Y1, 'Z': Z1, 'm': interval[0.023, 0.023],
         'alpha': alpha1, 'beta': beta1,
         'Bx': True, 'By': True, 'Bz': True, 'Bx_dx': None, 'Bx_dy': None,
         'Bx_dz': None, 'By_dy': None, 'By_dz': None, 'fx': None, 'fy': None,
         'fz': None, 'tx': None, 'ty': None, 'tz': None},
        {'X': X2, 'Y': Y2, 'Z': Z2, 'm': interval[0.023, 0.023],
         'alpha': alpha2, 'beta': beta2,
         'Bx': True, 'By': True, 'Bz': True, 'Bx_dx': None, 'Bx_dy': None,
         'Bx_dz': None, 'By_dy': None, 'By_dz': None, 'fx': None, 'fy': None,
         'fz': None, 'tx': None, 'ty': None, 'tz': None},
    ]

    # Use affine calculation for the first target point
    A = Map_I2H_Affine_body(target_points)

    return A

# ============================================================================
# Parallel Worker Functions (must be top-level)
#============================================================================

def _is_valid_for_all_orientations_worker(pos_box, full_angle_range, local_memo, I_intervals, b_intervals, eps_b3):
    """Check if position box is valid for ALL orientations in the given range.
    full_angle_range is now a list of two interval objects."""
    from collections import deque

    xi, yi = pos_box[0][0], pos_box[1][0]
    # The key should now include both angle ranges to be unique
    pos_key = (xi[0][0], xi[0][1], yi[0][0], yi[0][1],
               full_angle_range[0][0][0], full_angle_range[0][0][1],
               full_angle_range[1][0][0], full_angle_range[1][0][1])
    if pos_key in local_memo:
        return local_memo[pos_key]

    # Create two queues, one for each angle dimension
    # inner_queue1 = deque([full_angle_range[0]])
    # inner_queue2 = deque([full_angle_range[1]])

    # We need to check all combinations of bisections from both queues.
    # This can be complex. A simpler approach is to bisect them together.
    # Let's use a single queue with tuples of angle intervals.

    master_queue = deque([(full_angle_range[0], full_angle_range[1])])

    while master_queue:
        angle1_range, angle2_range = master_queue.popleft()

        # The `calculate_affine_A` expects a list of intervals for beta2
        Ainter = calculate_affine_A(xi, yi, [angle1_range, angle2_range])

        isOut = Out(Ainter, I_intervals, b_intervals)
        if isOut:
            # If any orientation sub-box is OUT, the whole position box is OUT for this orientation range.
            # However, the problem is to check if it's valid for *any* orientation.
            # The logic seems to be checking if it's valid for *all* orientations.
            # If for one orientation sub-box it's out, we can't conclude for the whole pos_box.
            # The current logic seems to be: if any sub-box is OUT, the whole check for this pos_box returns OUT.
            # This implies the goal is to find pos_boxes that are valid for ALL orientations.
            local_memo[pos_key] = 'OUT'
            return 'OUT'

        isFeasible = Feasible(Ainter, I_intervals, b_intervals)
        if isFeasible:
            continue

        # If not feasible, bisect the angle ranges
        width1 = angle1_range[0][1] - angle1_range[0][0]
        width2 = angle2_range[0][1] - angle2_range[0][0]

        if width1 < eps_b3 and width2 < eps_b3:
            local_memo[pos_key] = 'UNKNOWN'
            return 'UNKNOWN'

        angle_box = Create2DBox(angle1_range, angle2_range)
        bisected_boxes = BisectBox(angle_box)
        for box in bisected_boxes:
            master_queue.append((box[0][0], box[1][0]))

    local_memo[pos_key] = 'IN'
    return 'IN'

def _process_single_pos_box_worker(pos_box, angle_interval_p, memo_p, I_intervals, b_intervals, eps_b3):
    # angle_interval_p is the full list of angle intervals (beta2)
    status = _is_valid_for_all_orientations_worker(pos_box, angle_interval_p, memo_p, I_intervals, b_intervals, eps_b3)
    return status, pos_box

# ============================================================================
# Utility Functions for Interval Analysis (Keep original functions)
# ============================================================================

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
    angle1 = interval[144/180 * math.pi, 216/180 * math.pi]
    # angle1 = interval[216/180 * math.pi, 144/180 * math.pi]
    angle2 = interval[-1/5 * math.pi, 1/5 * math.pi]
    InitBox = Create2DBox(angle1, angle2)

    beta2 = [interval[0, 2 * math.pi], interval[0, 2 * math.pi]]

    # Define inputs interval and task set
    I_intervals = create_I_intervals(11, custom_bounds={10: (1, 1)})
    # I_intervals = [interval[-15, 15]] * 10

    b_intervals = [
        interval[0, 0],
        interval[0, 0],
        interval[0.01, 0.01],
        interval[0, 0],
        interval[0, 0],
        interval[0.01, 0.01],
    ]

    # Define precision of bisection and run the determination algorithm
    eps_b12 = 0.03 # 1.2
    eps_b3 = 0.2 # 6.28

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

    with open('Results/results_case3.json', 'w') as f:
        json.dump(results_data, f, indent=4)
    print("Results saved to Results/results_case3.json")

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
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    # ax.set_zticks(ticks, labels)

    # Set tick label font size
    ax.tick_params(axis='both', which='major', labelsize=12)  # Change 12 to your desired size

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='green', alpha=0.5, label='In workspace'),
                      Patch(facecolor='gray', alpha=0.2, label='Undetermined'),
                      Patch(facecolor='red', alpha=0.2, label='Out of workspace')
                      ]
    ax.legend(handles=legend_elements, loc='upper right')

    ticksx = [144/180 * math.pi, 1 * math.pi, 216/180 * math.pi] # interval[144/180 * math.pi, 216/180 * math.pi]
    labelsx = ['0', '1/2', '1']

    ticksy = [-1/5 * math.pi, 0, 1/5 * math.pi]
    labelsy = ['0', '1/2', '1']

    ax.set_xticks(ticksx, labelsx)
    ax.set_yticks(ticksy, labelsy)
    # ax.set_zticks(ticks, labels)

    plt.tight_layout()
    plt.show()

    print("Analysis complete!")

if __name__ == "__main__":
    mp.freeze_support()
    main()
