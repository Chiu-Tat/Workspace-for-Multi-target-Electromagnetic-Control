import numpy as np
from scipy.spatial import ConvexHull
from pypoman import compute_polytope_vertices
from WS_lib import *
from scipy.optimize import minimize
import sympy as sp
from mpl_toolkits.mplot3d import Axes3D

def Plot_Multiple_Hulls(target_points_list, colors=['lightblue', 'lightgreen', 'lightcoral'], alpha=0.6, labels=None):
    """
    Plot multiple hulls in one figure with different colors
    
    Parameters:
    target_points_list: list of target_points configurations
    colors: list of colors for each hull
    alpha: transparency level
    labels: list of labels for each hull
    """
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Pre-compute vertices and hulls to determine axis limits for equal aspect ratio
    all_vertices_list = []
    hulls = []
    for target_points in target_points_list:
        A = Extract_Map_I2H(target_points) @ Map_I2H(target_points)
        G, k = ModifiedHyperplaneShiftingMethod(A, -15, 15)
        vertices = np.array(compute_polytope_vertices(G, k))
        all_vertices_list.append(vertices)
        hulls.append(ConvexHull(vertices))
        
    # Determine plot limits for equal aspect ratio
    all_vertices = np.concatenate(all_vertices_list)
    x_min, y_min = np.min(all_vertices, axis=0)
    x_max, y_max = np.max(all_vertices, axis=0)
    
    max_range = np.array([x_max - x_min, y_max - y_min]).max()
    mid_x = (x_min + x_max) / 2
    mid_y = (y_min + y_max) / 2
    
    # Set plot limits with a little padding
    padding = 1.3
    ax.set_xlim(mid_x - max_range / 2 * padding, mid_x + max_range / 2 * padding)
    ax.set_ylim(mid_y - max_range / 2 * padding, mid_y + max_range / 2 * padding)
    
    ax.set_aspect('equal', adjustable='box')
    
    for i, (vertices, hull) in enumerate(zip(all_vertices_list, hulls)):
        # Calculate center point (centroid) of the hull
        center_x = np.mean(vertices[hull.vertices, 0])
        center_y = np.mean(vertices[hull.vertices, 1])
        
        # Fill the hull region with color
        hull_vertices = vertices[hull.vertices]
        hull_vertices = hull_vertices[ConvexHull(hull_vertices).vertices]  # Ensure proper ordering
        
        color = colors[i % len(colors)]
        label = labels[i] if labels else f'Case {i+1}'
        ax.fill(hull_vertices[:, 0], hull_vertices[:, 1], color=color, alpha=alpha, label=label)
        
        # Plot the edges
        for simplex in hull.simplices:
            ax.plot(vertices[simplex, 0], vertices[simplex, 1], color='k', linewidth=1.0)
        
        # Plot and label the center point
        ax.scatter(center_x, center_y, color=color, s=100, zorder=10, marker='*')
        # ax.annotate(f'Center {i+1}\n({center_x:.3f}, {center_y:.3f})', 
        #            xy=(center_x, center_y), xytext=(center_x+0.001, center_y+0.001),
        #            fontsize=8, ha='left', va='bottom',
        #            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.legend(fontsize=20)
    ax.set_xlabel('Dimension 1', fontsize=20)
    ax.set_ylabel('Dimension 2', fontsize=20)
    # ax.set_title('Multi-Target Workspaces Comparison')
    plt.show()

# Example usage
if __name__ == "__main__":

    # Case 1: Original target points    
    target_points_case1 = [
        {'X': 0.02, 'Y': 0.02, 'Z': -0.03, 'm': 0.2, 'alpha': np.pi/2, 'beta': np.pi/2, 
         'Bx': True, 'By': None, 'Bz': None, 'Bx_dx': None, 'Bx_dy': None, 
         'Bx_dz': None, 'By_dy': None, 'By_dz': None, 
         'fx': None, 'fy': None, 'fz': None, 'tx': None, 'ty': None, 'tz': None},
        {'X': -0.01, 'Y': -0.01, 'Z': 0, 'm': 0.5, 'alpha': np.pi/2, 'beta': np.pi/2, 
         'Bx': True, 'By': None, 'Bz': None, 'Bx_dx': None, 'Bx_dy': None, 
         'Bx_dz': None, 'By_dy': None, 'By_dz': None, 
         'fx': None, 'fy': None, 'fz': None, 'tx': None, 'ty': None, 'tz': None},
    ]
    
    # Case 2: Different positions and moments
    target_points_case2 = [
        {'X': 0.0, 'Y': 0.01, 'Z': 0.00, 'm': 0.2, 'alpha': -np.pi/3, 'beta': np.pi/4, 
         'Bx': True, 'By': None, 'Bz': None, 'Bx_dx': None, 'Bx_dy': None, 
         'Bx_dz': None, 'By_dy': None, 'By_dz': None, 
         'fx': None, 'fy': None, 'fz': None, 'tx': None, 'ty': None, 'tz': None},
        {'X': 0.0, 'Y': 0.01, 'Z': -0.01, 'm': 0.3, 'alpha': np.pi/3, 'beta': np.pi/6, 
         'Bx': True, 'By': None, 'Bz': None, 'Bx_dx': None, 'Bx_dy': None, 
         'Bx_dz': None, 'By_dy': None, 'By_dz': None, 
         'fx': None, 'fy': None, 'fz': None, 'tx': None, 'ty': None, 'tz': None},
    ]
    
    # Case 3: More different configurations
    # target_points_case3 = [
    #     {'X': -0.005, 'Y': -0.025, 'Z': -0.01, 'm': 0.15, 'alpha': np.pi/2, 'beta': np.pi/3, 
    #      'Bx': None, 'By': None, 'Bz': None, 'Bx_dx': None, 'Bx_dy': True, 
    #      'Bx_dz': None, 'By_dy': None, 'By_dz': None, 
    #      'fx': None, 'fy': None, 'fz': None, 'tx': None, 'ty': None, 'tz': None},
    #     {'X': 0.002, 'Y': -0.015, 'Z': -0.005, 'm': 0.9, 'alpha': -np.pi/2, 'beta': -2*np.pi/3, 
    #      'Bx': None, 'By': None, 'Bz': None, 'Bx_dx': None, 'Bx_dy': None, 
    #      'Bx_dz': None, 'By_dy': None, 'By_dz': None, 
    #      'fx': None, 'fy': None, 'fz': True, 'tx': None, 'ty': None, 'tz': None},
    # ]
    
    # Plot all three cases in one figure
    target_points_list = [target_points_case1, target_points_case2]
    colors = ['#9DC3E6', "#FD7A00"]
    labels = ['Case 1', 'Case 2']
    
    Plot_Multiple_Hulls(target_points_list, colors=colors, alpha=0.4, labels=labels)
    
