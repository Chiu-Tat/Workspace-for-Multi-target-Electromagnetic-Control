    #!/usr/bin/env python3
# Run from the repository root: python -m cases.case_6_interaction_obstacle_avoidance.workspace
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
from emns.workspace.checks import Feasible, Out, RobustFeasible, create_I_intervals, create_b_intervals, Create3DBox, BisectBox
from emns.numerics import affine as ca
from emns.workspace.affine import Map_I2H_Affine_body

# ============================================================================
# Affine version of WS_lib_Affine functions
# ============================================================================

def calculate_affine_A(X, Y, beta):
    """Affine version of calculate_interval_A"""
    target_points = [
        {'X': X, 'Y': Y, 'Z': interval[0.082, 0.082], 'm': interval[0.12, 0.12],
         'alpha': interval[math.pi/2, math.pi/2], 'beta': beta,
         'Bx': None, 'By': True, 'Bz': True, 'Bx_dx': None, 'Bx_dy': None,
         'Bx_dz': None, 'By_dy': None, 'By_dz': None, 'fx': None, 'fy': True,
         'fz': True, 'tx': None, 'ty': None, 'tz': None},
        {'X': interval[0.02, 0.02], 'Y': interval[0.0, 0.0], 'Z': interval[0.082, 0.082], 'm': interval[0.145, 0.145],
         'alpha': interval[math.pi/2, math.pi/2], 'beta': interval[math.pi/2, math.pi/2],
         'Bx': None, 'By': None, 'Bz': None, 'Bx_dx': None, 'Bx_dy': None,
         'Bx_dz': None, 'By_dy': None, 'By_dz': None, 'fx': None, 'fy': None,
         'fz': None, 'tx': None, 'ty': None, 'tz': None},
        # {'X': interval[0.02, 0.02], 'Y': interval[-0.01155, -0.01155], 'Z': interval[0.085, 0.085], 'm': interval[0.145, 0.145],
        #  'alpha': interval[math.pi/2, math.pi/2], 'beta': beta3,
        #  'Bx': None, 'By': True, 'Bz': True, 'Bx_dx': None, 'Bx_dy': None,
        #  'Bx_dz': None, 'By_dy': None, 'By_dz': None, 'fx': None, 'fy': None,
        #  'fz': None, 'tx': None, 'ty': None, 'tz': None},
    ]

    # Use affine calculation for the first target point
    A = Map_I2H_Affine_body(target_points)

    return A

# ============================================================================
# Parallel Processing Functions
# ============================================================================

def process_single_box_3d(box, b_intervals, I_intervals, eps):
    """Worker function to process a single 3D box"""
    try:
        # Build interval matrix using affine method
        beta1 = box[0][0]
        beta2 = box[1][0]
        beta3 = box[2][0]
        Ainter = calculate_affine_A(beta1, beta2, beta3)

        # Check feasibility
        isFeasible = Feasible(Ainter, I_intervals, b_intervals)

        if isFeasible:
            return 'feasible', box
        else:
            isOut = Out(Ainter, I_intervals, b_intervals)
            if isOut:
                return 'out', box
            else:
                # Handle eps as list or scalar
                if isinstance(eps, (list, tuple, np.ndarray)):
                    eps_list = eps
                else:
                    eps_list = [eps] * 3

                # Check if ANY dimension needs refinement
                needs_bisection = False
                if (beta1[0][1] - beta1[0][0]) > eps_list[0]:
                    needs_bisection = True
                elif (beta2[0][1] - beta2[0][0]) > eps_list[1]:
                    needs_bisection = True
                elif (beta3[0][1] - beta3[0][0]) > eps_list[2]:
                    needs_bisection = True

                if needs_bisection:
                    return 'bisect', box
                else:
                    return 'undetermined', box
    except Exception as e:
        return 'undetermined', box

def BisectBox_Flexible(Box, eps):
    """
    Bisection of a box that only splits dimensions larger than their respective epsilon.
    eps can be a scalar or a list of values for each dimension.
    """
    d = len(Box)

    # Handle eps as list or scalar
    if isinstance(eps, (list, tuple, np.ndarray)):
        eps_list = eps
    else:
        eps_list = [eps] * d

    # Determine which dimensions to split
    to_split = []
    for i in range(d):
        # Box[i][0] is the Interval object
        # Use .width() method or access via [0][1] - [0][0]
        width = Box[i][0].width()
        if width > eps_list[i]:
            to_split.append(True)
        else:
            to_split.append(False)

    # If no splitting needed (should be handled by caller, but safe check)
    if not any(to_split):
        return [Box]

    # Prepare intervals
    splits = []
    for i in range(d):
        inter = Box[i][0]
        if to_split[i]:
            # inter[0] returns [lower, upper] for compatibility
            lower = inter[0][0]
            upper = inter[0][1]
            mid = (lower + upper) / 2.0
            splits.append([interval[lower, mid], interval[mid, upper]])
        else:
            splits.append([inter])

    # Create new boxes
    new_boxes = []
    for p in product(*splits):
        new_box = [[p[i]] for i in range(d)]
        new_boxes.append(new_box)

    return new_boxes

def Ws3DDetermination_Parallel(InitBox, b_intervals, I_intervals, eps, max_iter=-1, num_processes=None):
    """Parallel version of 3D workspace determination"""
    if num_processes is None:
        num_processes = min(mp.cpu_count(), 8)

    print(f"Using {num_processes} parallel processes")

    L = []
    Lin = []
    Lout = []
    Lneg = []
    L.append(InitBox)
    iteration = 0

    worker_func = partial(process_single_box_3d,
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
                bisected_boxes = BisectBox_Flexible(box, eps)
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
    poly3d = Poly3DCollection(faces, facecolors=color_face, linewidths=0.03, edgecolors='k', alpha=0.1)
    # poly3d = Poly3DCollection(faces, facecolors=color_face, linewidths=0.1, alpha=0.1)
    ax.add_collection3d(poly3d)

# ============================================================================
# Main Application
# ============================================================================

def main():
    """Main function for affine analysis of 10-coil system"""

    print("Parallel Affine analysis for the determination of the workspace of an electromagnetic navigation system (10 coils)")
    print("=" * 80)

    # Build initial box
    # pmin = 0
    # pmax = 2 * math.pi
    X = interval[-0.05, 0.05]
    Y = interval[-0.05, 0.05]
    beta1 = interval[0, 2 * math.pi]
    InitBox = Create3DBox(X, Y, beta1)

    # Define inputs interval and task set
    I_intervals = create_I_intervals(11, custom_bounds={10: (1, 1)})
    # I_intervals = [interval[-15, 15]] * 10

    bf_intervals = [
        interval[0, 0],
        interval[0.02, 0.02], # T
        interval[0, 0],
        interval[-0.1, 0.1] # N
    ]

    # Define precision of bisection and run the determination algorithm
    # eps can be scalar or list [eps_x, eps_y, eps_beta]
    # Using larger eps for beta (range 2pi) compared to X,Y (range 0.1)
    eps = [0.005, 0.005, 0.1]
    num_processes = min(mp.cpu_count(), 16)
    print(f"core number: {mp.cpu_count()}")
    print(f"Running parallel workspace determination with precision eps = {eps}")
    print(f"Using {num_processes} processes")
    print("This may take some time...")

    start_time = time.time()
    L, Lin, Lout, Lneg, iteration = Ws3DDetermination_Parallel(
        InitBox, bf_intervals, I_intervals, eps, num_processes=num_processes
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
        'in_workspace': [box_to_list(box) for box in Lin],
        'undetermined': [box_to_list(box) for box in Lneg],
        'out_of_workspace': [box_to_list(box) for box in Lout]
    }

    with open('Results/results_case6_3D_02.json', 'w') as f:
        json.dump(results_data, f, indent=4)
    print("Results saved to Results/results_case6_3D_02.json")

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
    p_max = 2 * np.pi
    r = [0, p_max]
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
    ax.set_title('Workspace determination by affine analysis (10 coils)')
    ax.set_xlabel(r'$x$ (m)', fontsize=20, labelpad=10)
    ax.set_ylabel(r'$y$ (m)', fontsize=20, labelpad=10)
    ax.set_zlabel(r'$\beta$ (rad)', fontsize=20, labelpad=10)

    # Set tick label font size
    ax.tick_params(axis='both', which='major', labelsize=18)  # Change 12 to your desired size

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='green', alpha=0.4, label='In workspace'),
                    #   Patch(facecolor='gray', alpha=0.2, label='Undetermined'),
                    #   Patch(facecolor='red', alpha=0.2, label='Out of workspace')
                      ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=20)

    xticks = [-0.05, 0, 0.05]
    xlabels = ['-0.05', '0', '0.05']
    yticks = [-0.05, 0, 0.05]
    ylabels = ['-0.05', '0', '0.05']
    zticks = [0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi]
    zlabels = ['0', 'π/2', 'π', '3π/2', '2π']
    ax.set_xticks(xticks, xlabels)
    ax.set_yticks(yticks, ylabels)
    ax.set_zticks(zticks, zlabels)
    ax.set_xlim(-0.05, 0.05)
    ax.set_ylim(-0.05, 0.05)
    ax.set_zlim(0, 2*math.pi)
    # plt.tight_layout()

    # Save figure as SVG
    # svg_path = 'Results/plot_case1_3D_boxes002.svg'
    # plt.savefig(svg_path, format='svg', bbox_inches='tight')
    # print(f"Figure saved to {svg_path}")

    plt.show()

    print("Analysis complete!")

if __name__ == "__main__":
    mp.freeze_support()
    main()
