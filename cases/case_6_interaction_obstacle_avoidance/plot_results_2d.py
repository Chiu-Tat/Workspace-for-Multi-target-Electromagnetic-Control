import json
# Run from the repository root: python -m cases.case_6_interaction_obstacle_avoidance.plot_results_2d
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import numpy as np
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def plot_cuboid(ax, origin, size, color_face, alpha=0.2):
    """
    Plots a 3D cuboid.

    Parameters:
    -----------
    ax : matplotlib 3D axes
        The axes to plot on
    origin : tuple
        (x, y, z) coordinates of the origin (minimum corner)
    size : tuple
        (dx, dy, dz) dimensions of the cuboid
    color_face : str or color
        Face color of the cuboid
    alpha : float
        Transparency (0-1)
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
    poly3d = Poly3DCollection(faces, facecolors=color_face, linewidths=0.1, edgecolors='k', alpha=alpha)
    ax.add_collection3d(poly3d)

def detect_box_dimension(boxes):
    """
    Detects if boxes are 2D or 3D by checking the first non-empty box.

    Parameters:
    -----------
    boxes : list
        List of boxes

    Returns:
    --------
    int : 2 or 3 for 2D or 3D boxes, None if no boxes found
    """
    for box in boxes:
        if len(box) == 4:
            return 2
        elif len(box) == 6:
            return 3
    return None

def plot_results_from_file(filepath='Results/results_case6_3D_02.json', z_slice=None):
    """
    Reads workspace analysis results from a JSON file and plots them.
    Supports both 2D and 3D boxes.

    Parameters:
    -----------
    filepath : str
        Path to the JSON results file.
    z_slice : float, optional
        If provided and data is 3D, plots a 2D slice at this z-coordinate.
        In this mode, ONLY 'in_workspace' boxes are plotted.
    """
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file {filepath} was not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from the file {filepath}.")
        return

    undetermined_boxes = data.get('undetermined', [])
    out_of_workspace_boxes = data.get('out_of_workspace', [])
    in_workspace_boxes = data.get('in_workspace', [])

    print(f"Found {len(undetermined_boxes)} undetermined boxes.")
    print(f"Found {len(out_of_workspace_boxes)} out-of-workspace boxes.")
    print(f"Found {len(in_workspace_boxes)} in-workspace boxes.")

    # Detect box dimension
    dimension = None
    if undetermined_boxes:
        dimension = detect_box_dimension(undetermined_boxes)
    elif out_of_workspace_boxes:
        dimension = detect_box_dimension(out_of_workspace_boxes)
    elif in_workspace_boxes:
        dimension = detect_box_dimension(in_workspace_boxes)

    if dimension is None:
        print("Error: No valid boxes found in the file.")
        return

    print(f"Detected {dimension}D boxes.")

    # Handle z_slice 2D plotting
    if z_slice is not None:
        if dimension != 3:
            print("Warning: z_slice provided but data is not 3D. Ignoring z_slice and using standard plotting.")
        else:
            print(f"Plotting 2D slice at z = {z_slice}")
            fig, ax = plt.subplots(figsize=(6, 6))

            count_plotted = 0

            # Plot only in_workspace boxes that intersect with z_slice
            for box in in_workspace_boxes:
                if len(box) != 6:
                    continue
                # box: [x1, x2, y1, y2, z1, z2]
                z_min, z_max = box[4], box[5]

                if z_min <= z_slice <= z_max:
                    xo = box[0]
                    yo = box[2]
                    dx = box[1] - box[0]
                    dy = box[3] - box[2]

                    # Using a slightly larger linewidth or specific color
                    rect = patches.Rectangle((xo, yo), dx, dy, linewidth=0.1, edgecolor='black',
                                           facecolor='green', alpha=0.6)
                    ax.add_patch(rect)
                    count_plotted += 1

            print(f"Plotted {count_plotted} boxes intersecting z={z_slice}.")

            # Add Rectangle magnet at (0.02, 0) with direction toward y axis
            mag_x, mag_y = 0.02, 0.0
            mag_w, mag_h = 0.007, 0.015

            # Rectangle centered at (mag_x, mag_y)
            rect_magnet_blue = patches.Rectangle((mag_x - mag_w/2, mag_y - mag_h/2), mag_w, mag_h/2,
                                          linewidth=1.5, edgecolor='blue', facecolor='blue', zorder=10)
            rect_magnet_red = patches.Rectangle((mag_x - mag_w/2, mag_y), mag_w, mag_h/2,
                                          linewidth=1.5, edgecolor='red', facecolor='red', zorder=10)
            ax.add_patch(rect_magnet_blue)
            ax.add_patch(rect_magnet_red)


            # Setup plot limits automatically
            ax.autoscale()
            ax.axis('equal')
            ax.set_xlim(-0.05, 0.05)
            ticksx = [-0.05, 0, 0.05]
            labelsx = ['-0.05', '0', '0.05']
            ax.set_xticks(ticksx, labelsx)
            ax.set_ylim(-0.05, 0.05)
            ticksy = [-0.05, 0, 0.05]
            labelsy = ['-0.05', '0', '0.05']
            ax.set_yticks(ticksy, labelsy)
            # ax.set_title('Workspace determination by affine analysis (10 coils)')
            ax.set_xlabel(r'$x$ (m)', fontsize=20)
            ax.set_ylabel(r'$y$ (m)', fontsize=20)
            ax.tick_params(axis='both', which='major', labelsize=18)

            legend_elements = [Patch(facecolor='green', alpha=0.6, label='In workspace')]
            ax.legend(handles=legend_elements, loc='upper right', fontsize=20)

            plt.tight_layout()
            plt.show()
            return

    # Standard plotting (original behavior)
    if dimension == 2:
        # 2D plotting
        fig = plt.figure(figsize=(6, 6))
        ax = fig.gca()

        # Plot rectangle undetermined (gray)
        for box in undetermined_boxes:
            if len(box) != 4:
                print(f"Skipping malformed box data: {box}")
                continue
            xo = box[0]
            yo = box[2]
            dx = box[1] - box[0]
            dy = box[3] - box[2]
            rect = patches.Rectangle((xo, yo), dx, dy, linewidth=0.1, edgecolor='black',
                                   facecolor='gray', alpha=0.2)
            ax.add_patch(rect)

        # Plot rectangle out of workspace (red)
        for box in out_of_workspace_boxes:
            if len(box) != 4:
                print(f"Skipping malformed box data: {box}")
                continue
            xo = box[0]
            yo = box[2]
            dx = box[1] - box[0]
            dy = box[3] - box[2]
            rect = patches.Rectangle((xo, yo), dx, dy, linewidth=0.1, edgecolor='black',
                                   facecolor='salmon', alpha=0.2)
            ax.add_patch(rect)

        # Format plot
        ax.axis('equal')
        ax.set_title('Workspace determination by affine analysis (10 coils)')
        ax.set_xlabel(r'$x$ (m)', fontsize=20, labelpad=10)
        ax.set_ylabel(r'$y$ (m)', fontsize=20, labelpad=10)

        # Set tick label font size
        ax.tick_params(axis='both', which='major', labelsize=18)

        # Add legend
        legend_elements = [
                          Patch(facecolor='gray', alpha=0.2, label='Undetermined'),
                          Patch(facecolor='salmon', alpha=0.2, label='Out of workspace')
                          ]
        legend_elements.insert(0, Patch(facecolor='green', alpha=0.5, label='In workspace (not plotted)'))
        ax.legend(handles=legend_elements, loc='upper right', fontsize=20)

    elif dimension == 3:
        # 3D plotting
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, projection='3d')

        # Plot cuboid undetermined (gray)
        for box in undetermined_boxes:
            if len(box) != 6:
                print(f"Skipping malformed box data: {box}")
                continue
            origin = (box[0], box[2], box[4])
            size = (box[1] - box[0], box[3] - box[2], box[5] - box[4])
            plot_cuboid(ax, origin, size, 'gray', alpha=0.2)

        # Plot cuboid out of workspace (salmon/red)
        for box in out_of_workspace_boxes:
            if len(box) != 6:
                print(f"Skipping malformed box data: {box}")
                continue
            origin = (box[0], box[2], box[4])
            size = (box[1] - box[0], box[3] - box[2], box[5] - box[4])
            plot_cuboid(ax, origin, size, 'salmon', alpha=0.2)

        # Format plot
        ax.set_title('Workspace determination by affine analysis (10 coils)')
        ax.set_xlabel(r'$x$ (m)', fontsize=20, labelpad=10)
        ax.set_ylabel(r'$y$ (m)', fontsize=20, labelpad=10)
        ax.set_zlabel('Orientation (rad)', fontsize=20, labelpad=10)

        # Set tick label font size
        ax.tick_params(axis='both', which='major', labelsize=18)

        # Add legend
        legend_elements = [
                          Patch(facecolor='gray', alpha=0.1, label='Undetermined'),
                          Patch(facecolor='salmon', alpha=0.1, label='Out of workspace')
                          ]
        legend_elements.insert(0, Patch(facecolor='green', alpha=0.5, label='In workspace (not plotted)'))
        ax.legend(handles=legend_elements, loc='upper right', fontsize=20)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Example usage with a fixed z-value
    # Change z_slice to None to view the original 3D plot
    plot_results_from_file(z_slice=math.pi* 3/4)
