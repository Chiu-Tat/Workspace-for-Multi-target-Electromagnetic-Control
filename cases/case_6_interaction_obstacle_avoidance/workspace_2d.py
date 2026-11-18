#!/usr/bin/env python3
# Run from the repository root: python -m cases.case_6_interaction_obstacle_avoidance.workspace_2d
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
from emns.workspace.affine import Map_I2H_Affine_body

# Shared with cases.case_6_interaction_obstacle_avoidance.workspace
Z_PLANE = 0.082
FREE_M = 0.12
B_HOLD = 0.02
F_MAX = 0.1
XY_LIM = 0.05
EPS = 0.005

# ============================================================================
# Affine version of WS_lib_Affine functions
# ============================================================================

def calculate_affine_A(Y1, X2):
    """Affine version of calculate_interval_A"""
    target_points = [
        {'X': interval[0.0, 0.0], 'Y': Y1, 'Z': interval[Z_PLANE, Z_PLANE],
         'm': interval[FREE_M, FREE_M],
         'alpha': interval[math.pi/2,math.pi/2], 'beta': interval[math.pi/2,math.pi/2],
         'Bx': None, 'By': True, 'Bz': True, 'Bx_dx': None, 'Bx_dy': None,
         'Bx_dz': None, 'By_dy': None, 'By_dz': None, 'fx': None, 'fy': True,
         'fz': True, 'tx': None, 'ty': None, 'tz': None},
    ]
    target_points.append(
        {'X': X2, 'Y': interval[0.0, 0.0], 'Z': interval[Z_PLANE, Z_PLANE],
         'm': interval[FREE_M, FREE_M],
         'alpha': interval[math.pi/2,math.pi/2], 'beta': interval[0,0],
         'Bx': None, 'By': True, 'Bz': True, 'Bx_dx': None, 'Bx_dy': None,
         'Bx_dz': None, 'By_dy': None, 'By_dz': None, 'fx': None, 'fy': True,
         'fz': True, 'tx': None, 'ty': None, 'tz': None})
    # Use affine calculation for the first target point
    A = Map_I2H_Affine_body(target_points)

    # A = A[:,:-1]

    return A

# ============================================================================
# Parallel Processing Functions
# ============================================================================

def process_single_box_case2h(box, b_intervals, I_intervals, eps):
    """Worker function to process a single box for Case 2_H"""
    try:
        y1 = box[0][0]
        x2 = box[1][0]
        Ainter = calculate_affine_A(y1, x2)

        isFeasible = RobustFeasible(Ainter, I_intervals, b_intervals)

        if isFeasible:
            return 'feasible', box
        else:
            isOut = Out(Ainter, I_intervals, b_intervals)
            if isOut:
                return 'out', box
            else:
                if (y1[0][1] - y1[0][0]) > eps and (x2[0][1] - x2[0][0]) > eps:
                    return 'bisect', box
                else:
                    return 'undetermined', box
    except Exception as e:
        return 'undetermined', box

def _progress_bar(done, total, pending, n_in, n_out, n_und, t0):
    elapsed = time.time() - t0
    frac = min(1.0, done / total) if total else 0.0
    width = 28
    filled = int(frac * width)
    bar = '#' * filled + '-' * (width - filled)
    rate = done / elapsed if elapsed > 0 else 0.0
    remain = max(total - done, 0)
    eta = remain / rate if rate > 0 else float('inf')
    eta_txt = f'{eta:.0f}s' if math.isfinite(eta) else '?'
    print(
        f'|{bar}| {100.0 * frac:5.1f}%  '
        f'boxes {done}/{total}  queue {pending}  '
        f'in {n_in}  out {n_out}  undetermined {n_und}  '
        f'{elapsed:.0f}s  eta {eta_txt}',
        flush=True,
    )


def Ws2DDetermination_Parallel(InitBox, b_intervals, I_intervals, eps, max_iter=-1, num_processes=None):
    """Parallel version of 2D workspace determination for Case 6_2"""
    if num_processes is None:
        num_processes = min(mp.cpu_count(), 8)

    print(f"Using {num_processes} parallel processes")
    # Uniform 2-D bisection of [-XY_LIM, XY_LIM]^2 down to EPS yields this many
    # boxes if every node is split to the leaves.
    n_levels = 0
    width = 2.0 * XY_LIM
    while width > eps + 1e-15:
        width *= 0.5
        n_levels += 1
    est_total = (4 ** (n_levels + 1) - 1) // 3

    L = []
    Lin = []
    Lout = []
    Lneg = []
    L.append(InitBox)
    iteration = 0
    t0 = time.time()

    worker_func = partial(process_single_box_case2h,
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
            _progress_bar(iteration, est_total, len(L), len(Lin), len(Lout),
                          len(Lneg), t0)

            if iteration >= max_iter and max_iter >= 1:
                print('Max iter reached')
                break

    _progress_bar(iteration, max(est_total, iteration), 0, len(Lin), len(Lout),
                  len(Lneg), t0)
    print()
    return L, Lin, Lout, Lneg, iteration

# ============================================================================
# Physical-plane magnet paths
# ============================================================================

# Geometry from calculate_affine_A: magnet 1 slides on x = 0 (heading +y),
# magnet 2 slides on y = 0 (heading +x). Both live in the plane Z = Z_PLANE.
Y1_LIM = XY_LIM
X2_LIM = XY_LIM
MAG_LEN, MAG_WID = 0.009, 0.005


def _dipole_halves(x, y, beta, length=MAG_LEN, width=MAG_WID):
    u = np.array([math.cos(beta), math.sin(beta)])
    v = np.array([-u[1], u[0]])
    c = np.array([x, y])
    hl, hw = length / 2.0, width / 2.0
    south = np.array([c - hl * u - hw * v, c - hl * u + hw * v,
                      c + hw * v, c - hw * v])
    north = np.array([c - hw * v, c + hw * v,
                      c + hl * u + hw * v, c + hl * u - hw * v])
    return south, north


def plot_magnet_paths(ax=None, y1=0.03, x2=0.03):
    """Draw the two magnets in the physical (x, y) plane.

    The (y1, x2) snapshot is an example pose inside the study domain, not a
    recorded trajectory. There is no saved 2-D path for this case.
    """
    created = ax is None
    if created:
        _, ax = plt.subplots(figsize=(6, 6))

    ax.plot([0, 0], [-Y1_LIM, Y1_LIM], color='#c36b24', lw=2.0, zorder=2,
            label=r'magnet 1  ($x=0$, $\beta=\pi/2$)')
    ax.plot([-X2_LIM, X2_LIM], [0, 0], color='#2a5a9a', lw=2.0, zorder=2,
            label=r'magnet 2  ($y=0$, $\beta=0$)')

    s1, n1 = _dipole_halves(0.0, y1, math.pi / 2)
    s2, n2 = _dipole_halves(x2, 0.0, 0.0)
    ax.add_patch(patches.Polygon(s1, closed=True, facecolor='#c8c2b8',
                                 edgecolor='#1e1b16', lw=0.8, zorder=4))
    ax.add_patch(patches.Polygon(n1, closed=True, facecolor='#c36b24',
                                 edgecolor='#1e1b16', lw=0.8, zorder=4))
    ax.add_patch(patches.Polygon(s2, closed=True, facecolor='#c8c2b8',
                                 edgecolor='#1e1b16', lw=0.8, zorder=4))
    ax.add_patch(patches.Polygon(n2, closed=True, facecolor='#2a5a9a',
                                 edgecolor='#1e1b16', lw=0.8, zorder=4))
    ax.annotate('', xy=(0.0, y1 + 0.012), xytext=(0.0, y1),
                arrowprops=dict(arrowstyle='-|>', color='#1e1b16', lw=1.1),
                zorder=5)
    ax.annotate('', xy=(x2 + 0.012, 0.0), xytext=(x2, 0.0),
                arrowprops=dict(arrowstyle='-|>', color='#1e1b16', lw=1.1),
                zorder=5)
    ax.plot(0, y1, '+', color='#1e1b16', ms=6, mew=1.0, zorder=5)
    ax.plot(x2, 0, '+', color='#1e1b16', ms=6, mew=1.0, zorder=5)
    ax.text(0.004, y1 + 0.006, 'magnet 1', fontsize=9, color='#c36b24')
    ax.text(x2 + 0.004, 0.006, 'magnet 2', fontsize=9, color='#2a5a9a')

    ax.set_aspect('equal')
    pad = 0.008
    ax.set_xlim(-XY_LIM - pad, XY_LIM + pad)
    ax.set_ylim(-XY_LIM - pad, XY_LIM + pad)
    ticks = [-0.05, -0.025, 0.0, 0.025, 0.05]
    labels = ['−50', '−25', '0', '25', '50']
    ax.set_xticks(ticks, labels)
    ax.set_yticks(ticks, labels)
    ax.set_xlabel(r'$x$ (mm)')
    ax.set_ylabel(r'$y$ (mm)')
    ax.set_title(rf'Two-magnet travel axes  ($Z={Z_PLANE * 1e3:.0f}$ mm)')
    ax.grid(True, color='#e7e0d4', lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc='upper right', fontsize=8, frameon=True)
    if created:
        plt.tight_layout()
    return ax

# ============================================================================
# Main Application
# ============================================================================

def main():
    """Main function for affine analysis of 10-coil system"""

    print("Parallel Affine analysis for the determination of the workspace of an electromagnetic navigation system (10 coils)")
    print("=" * 80)

    # Build initial box (same xy domain as workspace.py)
    y1 = interval[-XY_LIM, XY_LIM]
    x2 = interval[-XY_LIM, XY_LIM]
    InitBox = Create2DBox(y1, x2)

    # Define inputs interval and task set
    I_intervals = create_I_intervals(11, custom_bounds={10: (1, 1)})
    # I_intervals = [interval[-15, 15]] * 10

    # Same body-frame task as workspace.py, applied to both magnets:
    # By = 0, Bz = B_HOLD, fy = 0, fz in [-F_MAX, F_MAX]
    b_intervals = [
        interval[0, 0],
        interval[B_HOLD, B_HOLD],
        interval[0, 0],
        interval[-F_MAX, F_MAX],
        interval[0, 0],
        interval[B_HOLD, B_HOLD],
        interval[0, 0],
        interval[-F_MAX, F_MAX],
    ]

    # Define precision of bisection and run the determination algorithm
    eps = EPS
    num_processes = min(mp.cpu_count(), 16)
    print(f"core number: {mp.cpu_count()}")
    print(f"Running parallel workspace determination with precision eps = {eps}")
    print(f"Using {num_processes} processes")
    print("This may take some time...")

    start_time = time.time()
    L, Lin, Lout, Lneg, iteration = Ws2DDetermination_Parallel(
        InitBox, b_intervals, I_intervals, eps, num_processes=num_processes
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
        'in_workspace': [box_to_list(box) for box in Lin],
        'undetermined': [box_to_list(box) for box in Lneg],
        'out_of_workspace': [box_to_list(box) for box in Lout]
    }
    results_path = os.path.join('Results', 'results_case6_2d.json')
    with open(results_path, 'w') as f:
        json.dump(results_data, f, indent=4)
    print(f"Results saved to {results_path}")

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
    ax.set_title('Workspace determination by affine analysis (10 coils) 2D')
    ax.set_xlim(-XY_LIM, XY_LIM)
    ax.set_ylim(-XY_LIM, XY_LIM)
    ticksx = [-0.05, 0, 0.05]
    labelsx = ['-0.05', '0', '0.05']
    ax.set_xticks(ticksx, labelsx)
    ticksy = [-0.05, 0, 0.05]
    labelsy = ['-0.05', '0', '0.05']
    ax.set_yticks(ticksy, labelsy)
    ax.set_xlabel(r'$y_1$ (m)', fontsize=20)
    ax.set_ylabel(r'$x_2$ (m)', fontsize=20)

    # Set tick label font size
    ax.tick_params(axis='both', which='major', labelsize=18)  # Change 12 to your desired size

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='green', alpha=0.5, label='In workspace'),
                      Patch(facecolor='gray', alpha=0.2, label='Undetermined'),
                      Patch(facecolor='red', alpha=0.2, label='Out of workspace')
                    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=14)

    plt.tight_layout()
    fig_path = os.path.join('Results', 'results_case6_2d.png')
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"Figure saved to {fig_path}")

    fig_paths, ax_paths = plt.subplots(figsize=(6, 6))
    plot_magnet_paths(ax_paths)
    fig_paths.tight_layout()
    paths_path = os.path.join('Results', 'results_case6_2d_paths.png')
    fig_paths.savefig(paths_path, dpi=150, bbox_inches='tight')

    if mpl.get_backend().lower() not in ('agg', 'pdf', 'svg', 'template'):
        plt.show()

    print("Analysis complete!")

if __name__ == "__main__":
    mp.freeze_support()
    if '--paths-only' in sys.argv:
        plot_magnet_paths()
        plt.show()
    else:
        main()
