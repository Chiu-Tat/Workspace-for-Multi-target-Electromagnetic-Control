#!/usr/bin/env python3
# Run from the repository root: python -m analysis.affine.ten_coil_fast
"""
Affine analysis for the determination of the workspace of an electromagnetic navigation system (10 coils)
Parallel version based on Case_1 fast.py
"""

# Standard library imports
from itertools import product
import time
import multiprocessing as mp
from functools import partial

# Scientific computing imports
import numpy as np
# Plotting imports
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Interval analysis imports
from emns.numerics.interval import interval
from emns.workspace.interval_fast import Map_I2H_Interval_Fast
from emns.workspace.checks import Out, RobustFeasible
from emns.workspace.affine import Map_I2H_Affine
from emns.workspace import interval_fast as WS_lib_Interval_fast
from emns.workspace import affine as WS_lib_Affine

# Synchronize coil parameters
WS_lib_Interval_fast.params_list = WS_lib_Affine.params_list

# ============================================================================
# Functions from Affine_10c.py
# ============================================================================

def calculate_affine_A(xi, yi, zoff):
    """Affine version of calculate_interval_A"""
    target_points = [
        {'X': xi, 'Y': yi, 'Z': interval[zoff, zoff], 'm': interval[0.00, 0.00],
         'alpha': interval[0.00, 0.00], 'beta': interval[0.00, 0.00],
         'Bx': True, 'By': True, 'Bz': None, 'Bx_dx': None, 'Bx_dy': None,
         'Bx_dz': None, 'By_dy': None, 'By_dz': None, 'fx': None, 'fy': None,
         'fz': None, 'tx': None, 'ty': None, 'tz': None}
    ]

    A = Map_I2H_Affine(target_points)
    A = A[:, :-1]
    return A

def calculate_interval_A(xi, yi, zoff):
    """Calculate the interval actuation matrix of a system"""
    target_points = [
    {'X': xi, 'Y': yi, 'Z': interval[zoff, zoff], 'm': interval[0.00, 0.00], 'alpha': interval[0.00, 0.00], 'beta': interval[0.00, 0.00], 'Bx': True, 'By': True, 'Bz': None, 'Bx_dx': None, 'Bx_dy': None, 'Bx_dz': None, 'By_dy': None, 'By_dz': None, 'fx': None, 'fy': None, 'fz': None, 'tx': None, 'ty': None, 'tz': None}
    ]
    A, row = Map_I2H_Interval_Fast(target_points)
    return A

def Create2DBox(xi, yi):
    """Create a 2D box for interval analysis"""
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
    Bb = [[0 for c in range(2)] for r in range(d)]
    comb = product(np.arange(2), repeat=d)
    M = np.asarray(list(comb))

    for i in range(d):
        inter = Box[i][0]
        xmin = inter[0][0]
        xmax = inter[0][1]
        Bb[i][0] = interval[xmin, (xmin+xmax)/2]
        Bb[i][1] = interval[(xmin+xmax)/2, xmax]

    Box_list = []
    for i in range(2**d):
        B_temp = [[0 for c in range(1)] for r in range(d)]
        for j in range(d):
            B_temp[j][0] = Bb[j][M[i,j]]
        Box_list.append(B_temp)
    return Box_list

# ============================================================================
# Parallel Processing Functions
# ============================================================================

def process_single_box(box, b_intervals, I_intervals, eps, zoff, use_affine):
    """Worker function to process a single box"""
    try:
        xi = box[0][0]
        yi = box[1][0]

        if use_affine:
            Ainter = calculate_affine_A(xi, yi, zoff)
        else:
            Ainter = calculate_interval_A(xi, yi, zoff)

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
        # In case of any error, treat as undetermined
        return 'undetermined', box

def Ws2DDetermination_Parallel(InitBox, bmin, bmax, Imin, Imax, eps, zoff, max_iter=-1, use_affine=True, num_processes=None):
    """Parallel version of workspace determination"""
    if num_processes is None:
        num_processes = min(mp.cpu_count(), 8)

    print(f"Using {num_processes} parallel processes")

    L = []
    Lin = []
    Lout = []
    Lneg = []
    L.append(InitBox)
    iteration = 0

    I_intervals = [interval[Imin, Imax]] * 10
    b_intervals = [interval[bmin, bmax], interval[bmin, bmax]]

    worker_func = partial(process_single_box,
                         b_intervals=b_intervals,
                         I_intervals=I_intervals,
                         eps=eps,
                         zoff=zoff,
                         use_affine=use_affine)

    with mp.Pool(processes=num_processes) as pool:
        while(len(L) > 0):
            # Process boxes in parallel
            if len(L) >= num_processes:
                # Process multiple boxes in parallel
                current_boxes = L[:num_processes]
                L = L[num_processes:]

                results = pool.map(worker_func, current_boxes)
            else:
                # Process remaining boxes
                current_boxes = L
                L = []

                results = pool.map(worker_func, current_boxes)

            # Process results and update lists
            boxes_to_bisect = []
            for result_type, box in results:
                if result_type == 'feasible':
                    Lin.append(box)
                elif result_type == 'out':
                    Lout.append(box)
                elif result_type == 'bisect':
                    boxes_to_bisect.append(box)
                else:  # undetermined
                    Lneg.append(box)

            # Bisect boxes that need bisection
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
    """Main function for affine vs interval analysis of 10-coil system (Parallel)"""

    print("Affine vs Interval analysis (Parallel) for the determination of the workspace of an electromagnetic navigation system (10 coils)")
    print("=" * 80)

    # Build initial box
    pmin = -0.05
    pmax = 0.05
    xi = interval[pmin, pmax]
    yi = interval[pmin, pmax]
    z = -0.01 + 0.085  # Adjusted z-offset to match the original context
    InitBox = Create2DBox(xi, yi)

    # Define inputs interval and task set
    Imin = -15
    Imax = 15
    bmin = -0.08
    bmax = 0.08

    # Define precision of bisection and run the determination algorithm
    eps = 0.002

    num_processes = min(mp.cpu_count(), 16)
    print(f"Running parallel workspace determination with {num_processes} processes")
    print(f"Precision eps = {eps}")
    print("This may take some time...")

    # --- Run Affine Analysis ---
    print("\n--- Running Affine Analysis (Parallel) ---")
    start_time = time.time()
    L_affine, Lin_affine, Lout_affine, Lneg_affine, iteration_affine = Ws2DDetermination_Parallel(
        InitBox, bmin, bmax, Imin, Imax, eps, z, use_affine=True, num_processes=num_processes
    )
    end_time = time.time()
    time_affine = end_time - start_time
    print(f"Affine method took: {time_affine:.2f} seconds")
    print(f"Completed after {iteration_affine} iterations")
    print(f"Boxes in workspace: {len(Lin_affine)}")
    print(f"Boxes out of workspace: {len(Lout_affine)}")
    print(f"Undetermined boxes: {len(Lneg_affine)}")

    # --- Run Interval Analysis ---
    print("\n--- Running Interval Analysis (Parallel) ---")
    start_time = time.time()
    L_interval, Lin_interval, Lout_interval, Lneg_interval, iteration_interval = Ws2DDetermination_Parallel(
        InitBox, bmin, bmax, Imin, Imax, eps, z, use_affine=False, num_processes=num_processes
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

    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(13, 6))

    # Function to plot boxes
    def plot_boxes(ax, Lin, Lout, Lneg, title):
        # Plot rectangle in workspace (green)
        for i in range(len(Lin)):
            Box = Lin[i]
            xo = Box[0][0][0][0]
            yo = Box[1][0][0][0]
            dx = Box[0][0][0][1] - xo
            dy = Box[1][0][0][1] - yo
            rect = patches.Rectangle((xo, yo), dx, dy, linewidth=0.5, edgecolor='black',
                                   facecolor='green', alpha=0.5)
            ax.add_patch(rect)

        # Plot rectangle undetermined (gray)
        for i in range(len(Lneg)):
            Box = Lneg[i]
            xo = Box[0][0][0][0]
            yo = Box[1][0][0][0]
            dx = Box[0][0][0][1] - xo
            dy = Box[1][0][0][1] - yo
            rect = patches.Rectangle((xo, yo), dx, dy, linewidth=0.5, edgecolor='black',
                                   facecolor='gray', alpha=0.2)
            ax.add_patch(rect)

        # Plot rectangle out of workspace (red)
        for i in range(len(Lout)):
            Box = Lout[i]
            xo = Box[0][0][0][0]
            yo = Box[1][0][0][0]
            dx = Box[0][0][0][1] - xo
            dy = Box[1][0][0][1] - yo
            rect = patches.Rectangle((xo, yo), dx, dy, linewidth=0.5, edgecolor='black',
                                   facecolor='red', alpha=0.2)
            ax.add_patch(rect)

        ax.axis('equal')
        ax.set_title(title, fontsize=20)
        # ax.set_xlabel('x (m)', fontsize=20)
        # ax.set_ylabel('y (m)', fontsize=20)
        ax.grid(True, linestyle='--', alpha=0.3)
        # ax.tick_params(axis='both', which='major', labelsize=18)
        # hide the ticks
        ax.set_xticks([])
        ax.set_yticks([])

        # Add legend
        legend_elements = [patches.Patch(facecolor='green', alpha=0.5, label='In workspace'),
                          patches.Patch(facecolor='gray', alpha=0.2, label='Undetermined'),
                          patches.Patch(facecolor='red', alpha=0.2, label='Out of workspace')]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=18, framealpha=0.5)

    plot_boxes(ax1, Lin_affine, Lout_affine, Lneg_affine,
              f'Affine Arithmetic\n({time_affine:.2f}s, {len(Lneg_affine)} boundary boxes)')
    plot_boxes(ax2, Lin_interval, Lout_interval, Lneg_interval,
              f'Interval Arithmetic\n({time_interval:.2f}s, {len(Lneg_interval)} boundary boxes)')

    plt.tight_layout()
    plt.show()

    print("Analysis complete!")
    print(f"\nComparison Summary:")
    print(f"Affine method: {len(Lneg_affine)} undetermined boxes in {time_affine:.2f}s")
    print(f"Interval method: {len(Lneg_interval)} undetermined boxes in {time_interval:.2f}s")
    improvement = len(Lneg_interval) - len(Lneg_affine)
    print(f"Improvement: {improvement} fewer undetermined boxes")

if __name__ == "__main__":
    mp.freeze_support()
    main()
