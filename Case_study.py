#!/usr/bin/env python3
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
from custom_interval import interval, imath, fpu
from Check_lib import Feasible, Out, RobustFeasible, create_I_intervals, create_b_intervals, Create2DBox, BisectBox
import custom_affine as ca
from WS_lib_Affine import Map_I2H_Affine_body, arm_end_positions_affine

def arm_end_positions(beta1, alpha2):
    # Forward kinematics calculations
    # x_base = 0
    # y_base = 0
    # z_base = 0
    # L1_m = 0.5
    # L1 = 1
    # L2_m = 0.5
    # L2 = 1
    x_base = 0
    y_base = 0.023
    z_base = 0.079
    L1_m = 0.011
    L1 = 0.023
    L2_m = 0.011
    L2 = 0.029
    
    beta1 = beta1 + np.pi/2
    alpha2 = alpha2 - np.pi/2

    # position of the first magnet
    p1_x = L1_m * np.sin(beta1)
    p1_y = -L1_m * np.cos(beta1)
    p1_z = 0.0
    p1_x += x_base
    p1_y += y_base
    p1_z += z_base
    
    # position of the second magnet
    p2_x = (L1 + L2_m * np.cos(alpha2)) * np.sin(beta1)
    p2_y = -(L1 + L2_m * np.cos(alpha2)) * np.cos(beta1)
    p2_z = -L2_m * np.sin(alpha2)
    p2_x += x_base
    p2_y += y_base
    p2_z += z_base
    
    return p1_x, p1_y, p1_z, p2_x, p2_y, p2_z

def calculate_affine_A(beta1, alpha2):
    """Affine version of calculate_interval_A"""
    p1_x, p1_y, p1_z, p2_x, p2_y, p2_z = arm_end_positions_affine(beta1, alpha2)
    target_points = [
        {'X': p1_x, 'Y': p1_y, 'Z': p1_z, 'm': interval[0.145, 0.145], 
         'alpha': interval[math.pi/2, math.pi/2], 'beta': beta1, 
         'Bx': True, 'By': True, 'Bz': True, 'Bx_dx': None, 'Bx_dy': None, 
         'Bx_dz': None, 'By_dy': None, 'By_dz': None, 'fx': None, 'fy': None, 
         'fz': None, 'tx': None, 'ty': None, 'tz': None},
        {'X': p2_x, 'Y': p2_y, 'Z': p2_z, 'm': interval[0.145, 0.145], 
         'alpha': alpha2, 'beta': beta1, 
         'Bx': True, 'By': True, 'Bz': True, 'Bx_dx': None, 'Bx_dy': None, 
         'Bx_dz': None, 'By_dy': None, 'By_dz': None, 'fx': None, 'fy': None, 
         'fz': None, 'tx': None, 'ty': None, 'tz': None},
    ]
    
    # Use affine calculation for the first target point
    A = Map_I2H_Affine_body(target_points)
    
    return A

# ============================================================================
# Parallel Processing Functions
# ============================================================================

def process_single_box(box, b_intervals, I_intervals, eps):
    """Worker function to process a single box"""
    try:
        # Build interval matrix using affine method
        beta1 = box[0][0]
        beta2 = box[1][0]
        Ainter = calculate_affine_A(beta1, beta2)

        # Check feasibility
        isFeasible = RobustFeasible(Ainter, I_intervals, b_intervals)

        if isFeasible:
            return 'feasible', box
        else:
            isOut = Out(Ainter, I_intervals, b_intervals)
            if isOut:
                return 'out', box
            else: 
                if (beta1[0][1] - beta1[0][0]) > eps and (beta2[0][1] - beta2[0][0]) > eps:
                    return 'bisect', box
                else:
                    return 'undetermined', box
    except Exception as e:
        # In case of any error, treat as undetermined
        return 'undetermined', box

def Ws2DDetermination_Parallel(InitBox, b_intervals, I_intervals, eps, max_iter=-1, num_processes=None):
    """Parallel version of workspace determination"""
    if num_processes is None:
        num_processes = min(mp.cpu_count(), 8)  # Use up to 8 cores by default
    
    print(f"Using {num_processes} parallel processes")
    
    L = []
    Lin = []
    Lout = []
    Lneg = []
    L.append(InitBox)
    iteration = 0
    
    # Create worker function with fixed parameters
    worker_func = partial(process_single_box, 
                         b_intervals=b_intervals, 
                         I_intervals=I_intervals, 
                         eps=eps)

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
    """Main function for affine analysis of 10-coil system"""
    
    print("Parallel Affine analysis for the determination of the workspace of an electromagnetic navigation system (10 coils)")
    print("=" * 80)
    
    # Build initial box
    beta1 = interval[-5/6 * math.pi, -1/6 * math.pi]
    alpha2 = interval[1/6 * math.pi, 5/6 * math.pi]
    InitBox = Create2DBox(beta1, alpha2)

    # Define inputs interval and task set
    I_intervals = create_I_intervals(11, custom_bounds={10: (1, 1)})
    # I_intervals = [interval[-15, 15]] * 10

    b_intervals = [
        interval[0, 0],
        interval[0, 0],
        interval[0.02, 0.02],
        interval[0, 0],
        interval[0, 0],
        interval[0.03, 0.03],
    ]

    # Define precision of bisection and run the determination algorithm
    eps = 0.05
    num_processes = min(mp.cpu_count(), 16)  # Adjust based on your system
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
    # print("Saving results...")
    # if not os.path.exists('Results'):
    #     os.makedirs('Results')
    # def box_to_list(box):
    #     # Converts a box from its nested interval structure to a simple list [xmin, xmax, ymin, ymax]
    #     return [box[0][0][0][0], box[0][0][0][1], box[1][0][0][0], box[1][0][0][1]]
    # results_data = {
    #     'undetermined': [box_to_list(box) for box in Lneg],
    #     'out_of_workspace': [box_to_list(box) for box in Lout]
    # }
    # with open('Results/results_case5_0203.json', 'w') as f:
    #     json.dump(results_data, f, indent=4)

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
    # ax.set_title('Workspace determination by affine analysis (10 coils)')
    ax.set_xlabel(r'$\beta_1$ (rad)', fontsize=20)
    ax.set_ylabel(r'$\alpha_2$ (rad)', fontsize=20)

    # Set tick label font size
    ax.tick_params(axis='both', which='major', labelsize=18)  # Change 12 to your desired size
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='green', alpha=0.5, label='In workspace'),
                    #   Patch(facecolor='gray', alpha=0.2, label='Undetermined'),
                    #   Patch(facecolor='red', alpha=0.2, label='Out of workspace')
                      ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=20)

    ticksx = [-5/6 * math.pi, -2/3 * math.pi, -1/2 * math.pi, -1/3 * math.pi, -1/6 * math.pi]
    labelsx = ['-5π/6', '-2π/3', '-π/2', '-π/3', '-π/6']
    
    ticksy = [1/6 * math.pi, 1/3 * math.pi, 1/2 * math.pi, 2/3 * math.pi, 5/6 * math.pi]
    labelsy = ['π/6', 'π/3', 'π/2', '2π/3', '5π/6']
    
    ax.set_xticks(ticksx, labelsx)
    ax.set_yticks(ticksy, labelsy)

    plt.tight_layout()
    plt.grid()
    plt.show()
    
    print("Analysis complete!")

if __name__ == "__main__":
    # Required for multiprocessing on Windows
    mp.freeze_support()
    main()