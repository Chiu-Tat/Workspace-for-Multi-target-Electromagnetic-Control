import json
# Run from the repository root: python -m visualization.plot_results
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

def plot_results_from_file(filepath='Results/results_case6_3D_02.json'):
    """
    Reads workspace analysis results from a JSON file and plots them.
    Supports both 2D and 3D boxes.
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

    print(f"Found {len(undetermined_boxes)} undetermined boxes.")
    print(f"Found {len(out_of_workspace_boxes)} out-of-workspace boxes.")

    # Detect box dimension
    dimension = None
    if undetermined_boxes:
        dimension = detect_box_dimension(undetermined_boxes)
    elif out_of_workspace_boxes:
        dimension = detect_box_dimension(out_of_workspace_boxes)

    if dimension is None:
        print("Error: No valid boxes found in the file.")
        return

    print(f"Detected {dimension}D boxes.")

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
        ax.set_title('Workspace Plot from Saved Results')
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')

        # Set tick label font size
        ax.tick_params(axis='both', which='major', labelsize=12)

        # Add legend
        legend_elements = [
                          Patch(facecolor='gray', alpha=0.2, label='Undetermined'),
                          Patch(facecolor='salmon', alpha=0.2, label='Out of workspace')
                          ]
        # Add a green patch for context, as "in workspace" boxes are not saved in the file.
        legend_elements.insert(0, Patch(facecolor='green', alpha=0.5, label='In workspace (not plotted)'))
        ax.legend(handles=legend_elements, loc='upper right')

        # Set plot limits based on the original InitBox (for 2D case)
        # ax.set_xlim(144/180 * math.pi, 216/180 * math.pi)
        # ax.set_ylim(-1/5 * math.pi, 1/5 * math.pi)
        # ax.set_xlim(1/6 * math.pi, 5/6 * math.pi)
        # ax.set_ylim(-5/6 * math.pi, -1/6 * math.pi)

    elif dimension == 3:
        # 3D plotting
        fig = plt.figure(figsize=(10, 10))
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
        ax.set_title('Workspace Plot from Saved Results (3D)')
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_zlabel('x3')

        # Set tick label font size
        ax.tick_params(axis='both', which='major', labelsize=12)

        # Add legend
        legend_elements = [
                          Patch(facecolor='gray', alpha=0.1, label='Undetermined'),
                          Patch(facecolor='salmon', alpha=0.1, label='Out of workspace')
                          ]
        # Add a green patch for context, as "in workspace" boxes are not saved in the file.
        legend_elements.insert(0, Patch(facecolor='green', alpha=0.5, label='In workspace (not plotted)'))
        ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_results_from_file()
