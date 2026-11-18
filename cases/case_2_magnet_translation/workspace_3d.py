#!/usr/bin/env python3
# Run from the repository root: python -m cases.case_2_magnet_translation.workspace_3d
"""
Affine analysis for the determination of the workspace of an electromagnetic navigation system (10 coils)
"""
import os
import json
# Standard library imports
import math
from itertools import product
import time
import multiprocessing as mp
from functools import partial

# Scientific computing imports
import numpy as np
# Plotting imports
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
# Interval analysis imports - use our custom module
from emns.numerics.interval import interval, imath, fpu
from emns.workspace.checks import Feasible, Out, RobustFeasible, create_I_intervals, create_b_intervals, Create3DBox, BisectBox
from emns.numerics import affine as ca
from emns.workspace.affine import Map_I2H_Affine

# ============================================================================
# Affine version of WS_lib_Affine functions
# ============================================================================

def calculate_affine_A(z1, z2, z3):
    """Affine version of calculate_interval_A"""
    # target_points = [
    #     {'X': interval[-0.014, -0.014], 'Y': interval[-0.003, -0.003], 'Z': z1, 'm': interval[0.145, 0.145],
    #      'alpha': interval[0,0], 'beta': interval[0.00, 0.00],
    #      'Bx': None, 'By': None, 'Bz': None, 'Bx_dx': None, 'Bx_dy': None,
    #      'Bx_dz': None, 'By_dy': None, 'By_dz': None, 'fx': None, 'fy': None,
    #      'fz': True, 'tx': None, 'ty': None, 'tz': None},
    #      {'X': interval[0, 0], 'Y': interval[-0.003, -0.003], 'Z': z2, 'm': interval[0.145, 0.145],
    #      'alpha': interval[0.00, 0.00], 'beta': interval[0.00, 0.00],
    #      'Bx': None, 'By': None, 'Bz': None, 'Bx_dx': None, 'Bx_dy': None,
    #      'Bx_dz': None, 'By_dy': None, 'By_dz': None, 'fx': None, 'fy': None,
    #      'fz': True, 'tx': None, 'ty': None, 'tz': None},
    #         {'X': interval[0.014, 0.014], 'Y': interval[-0.003, -0.003], 'Z': z3, 'm': interval[0.145, 0.145],
    #      'alpha': interval[0.00, 0.00], 'beta': interval[0.00, 0.00],
    #      'Bx': None, 'By': None, 'Bz': None, 'Bx_dx': None, 'Bx_dy': None,
    #      'Bx_dz': None, 'By_dy': None, 'By_dz': None, 'fx': None, 'fy': None,
    #      'fz': True, 'tx': None, 'ty': None, 'tz': None}
    # ]
    target_points = [
        {'X': interval[-0.014, -0.014], 'Y': z1, 'Z': interval[0.0835, 0.0835], 'm': interval[0.145, 0.145],
         'alpha': interval[math.pi/2,math.pi/2], 'beta': interval[math.pi/2,math.pi/2],
         'Bx': None, 'By': None, 'Bz': None, 'Bx_dx': None, 'Bx_dy': None,
         'Bx_dz': None, 'By_dy': None, 'By_dz': None, 'fx': None, 'fy': True,
         'fz': None, 'tx': None, 'ty': None, 'tz': None},
         {'X': interval[0, 0], 'Y': z2, 'Z': interval[0.0835, 0.0835], 'm': interval[0.145, 0.145],
         'alpha': interval[0.00, 0.00], 'beta': interval[math.pi/2,math.pi/2],
         'Bx': None, 'By': None, 'Bz': None, 'Bx_dx': None, 'Bx_dy': None,
         'Bx_dz': None, 'By_dy': None, 'By_dz': None, 'fx': None, 'fy': True,
         'fz': None, 'tx': None, 'ty': None, 'tz': None},
            {'X': interval[0.014, 0.014], 'Y': z3, 'Z': interval[0.0835, 0.0835], 'm': interval[0.145, 0.145],
         'alpha': interval[0.00, 0.00], 'beta': interval[math.pi/2,math.pi/2],
         'Bx': None, 'By': None, 'Bz': None, 'Bx_dx': None, 'Bx_dy': None,
         'Bx_dz': None, 'By_dy': None, 'By_dz': None, 'fx': None, 'fy': True,
         'fz': None, 'tx': None, 'ty': None, 'tz': None}
    ]

    # Use affine calculation for the first target point
    A = Map_I2H_Affine(target_points)

    return A

# ============================================================================
# Parallel Processing Functions
# ============================================================================

def process_single_box_case2_3d(box, b_intervals, I_intervals, eps):
    """Worker function to process a single box for Case 2 3D"""
    try:
        z1 = box[0][0]
        z2 = box[1][0]
        z3 = box[2][0]

        Ainter = calculate_affine_A(z1, z2, z3)

        isFeasible = RobustFeasible(Ainter, I_intervals, b_intervals)

        if isFeasible:
            return 'feasible', box
        else:
            isOut = Out(Ainter, I_intervals, b_intervals)
            if isOut:
                return 'out', box
            else:
                if ((z1[0][1] - z1[0][0]) > eps and
                   (z2[0][1] - z2[0][0]) > eps and
                   (z3[0][1] - z3[0][0]) > eps):
                    return 'bisect', box
                else:
                    return 'undetermined', box
    except Exception as e:
        return 'undetermined', box

def Ws3DDetermination_Parallel(InitBox, b_intervals, I_intervals, eps, max_iter=-1, num_processes=None):
    """Parallel version of 3D workspace determination for Case 2"""
    if num_processes is None:
        num_processes = min(mp.cpu_count(), 8)

    print(f"Using {num_processes} parallel processes")

    L = []
    Lin = []
    Lout = []
    Lneg = []
    L.append(InitBox)
    iteration = 0

    worker_func = partial(process_single_box_case2_3d,
                         b_intervals=b_intervals,
                         I_intervals=I_intervals,
                         eps=eps)

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
    # poly3d = Poly3DCollection(faces, facecolors=color_face, linewidths=0.1, edgecolors='k', alpha=0.1)
    poly3d = Poly3DCollection(faces, facecolors=color_face, linewidths=0.1, alpha=0.1)
    ax.add_collection3d(poly3d)

# ============================================================================
# Main Application
# ============================================================================

def main():
    """Main function for affine analysis of 10-coil system"""

    print("Parallel Affine analysis for the determination of the workspace of an electromagnetic navigation system (10 coils)")
    print("=" * 80)

    # Build initial box
    # pmin = -0.036 + 0.085 # 0.049
    # pmax = 0.0268 + 0.085 # 0.1138
    pmin = -0.0317
    pmax = 0.0317
    z1 = interval[pmin, pmax]
    z2 = interval[pmin, pmax]
    z3 = interval[pmin, pmax]
    InitBox = Create3DBox(z1, z2, z3)

    # Define inputs interval and task set
    I_intervals = create_I_intervals(11, custom_bounds={10: (1, 1)})
    # I_intervals = [interval[-15, 15]] * 10

    # Define forces intervals
    F_intervals = [
        interval[-0.01, 0.01],
        interval[-0.01, 0.01],
        interval[-0.01, 0.01]
    ]

    # Define precision of bisection and run the determination algorithm
    eps = 0.003
    num_processes = min(mp.cpu_count(), 16)
    print(f"core number: {mp.cpu_count()}")
    print(f"Running parallel workspace determination with precision eps = {eps}")
    print(f"Using {num_processes} processes")
    print("This may take some time...")

    start_time = time.time()
    L, Lin, Lout, Lneg, iteration = Ws3DDetermination_Parallel(
        InitBox, F_intervals, I_intervals, eps, num_processes=num_processes
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
        # Converts a 3D box from its nested interval structure to a simple list [xmin, xmax, ymin, ymax, zmin, zmax]
        return [box[0][0][0][0], box[0][0][0][1], box[1][0][0][0], box[1][0][0][1], box[2][0][0][0], box[2][0][0][1]]

    results_data = {
        'undetermined': [box_to_list(box) for box in Lneg],
        'out_of_workspace': [box_to_list(box) for box in Lout]
    }

    with open('Results/results_case2_3D_0101.json', 'w') as f:
        json.dump(results_data, f, indent=4)
    print("Results saved to Results/results_case2_3D_0101.json")

    # Plotting
    print("Creating 3D plot...")

    fig = plt.figure(figsize=(7, 6))
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

        # Draw the edges of the maximum box
    # p_max = 2 * np.pi
    r = [pmin, pmax]
    for s, e in product(r, r):
        ax.plot([s, e], [r[0], r[0]], [r[0], r[0]], 'k-', linewidth=1)
        ax.plot([s, e], [r[1], r[1]], [r[0], r[0]], 'k-', linewidth=1)
        ax.plot([s, e], [r[0], r[0]], [r[1], r[1]], 'k-', linewidth=1)
        ax.plot([s, e], [r[1], r[1]], [r[1], r[1]], 'k-', linewidth=1)
    for s, e in product(r, r):
        ax.plot([r[0], r[0]], [s, e], [r[0], r[0]], 'k-', linewidth=1)
        ax.plot([r[1], r[1]], [s, e], [r[0], r[0]], 'k-', linewidth=1)
        ax.plot([r[0], r[0]], [s, e], [r[1], r[1]], 'k-', linewidth=1)
        ax.plot([r[1], r[1]], [s, e], [r[1], r[1]], 'k-', linewidth=1)
    for s, e in product(r, r):
        ax.plot([r[0], r[0]], [r[0], r[0]], [s, e], 'k-', linewidth=1)
        ax.plot([r[1], r[1]], [r[0], r[0]], [s, e], 'k-', linewidth=1)
        ax.plot([r[0], r[0]], [r[1], r[1]], [s, e], 'k-', linewidth=1)
        ax.plot([r[1], r[1]], [r[1], r[1]], [s, e], 'k-', linewidth=1)

    # Format plot
    ax.axis('equal')
    # ax.set_title('Workspace determination by affine analysis (10 coils)')
    ax.set_xlabel(r'$z_1$ (m)', fontsize=20, labelpad=10)
    ax.set_ylabel(r'$z_2$ (m)', fontsize=20, labelpad=10)
    ax.set_zlabel(r'$z_3$ (m)', fontsize=20, labelpad=10)

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
