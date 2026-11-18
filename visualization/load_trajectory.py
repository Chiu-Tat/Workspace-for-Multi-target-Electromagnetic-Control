"""
# Run from the repository root: python -m visualization.load_trajectory
Example script showing how to load and use trajectory results saved from planner scripts
for both 2D (alpha-beta) and 3D trajectories.

This script demonstrates:
1. Loading trajectory data from JSON file
2. Reconstructing Bezier curves from control points
3. Evaluating trajectory at specific time points
4. Visualizing the loaded trajectory in 2D or 3D
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.special import comb
import math
import os

def load_trajectory_results(filepath):
    """
    Loads trajectory optimization results from a JSON file.

    Parameters:
    -----------
    filepath : str
        Path to the trajectory JSON file

    Returns:
    --------
    dict : Dictionary containing all trajectory data
        - control_points: dict with 'Cx', 'Cy', and optional 'Cz' (numpy arrays or None)
        - time_segments: list of time durations
        - corridor: list of corridor boxes (dimension-dependent)
        - start_pos: tuple of joint angles (length equals trajectory dimension)
        - end_pos: tuple of joint angles (length equals trajectory dimension)
        - config: dict with N_DEGREE, V_MAX, A_MAX
        - num_segments: int
        - dimension: 2 for planar trajectories, 3 for spatial trajectories
    """
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)

        # Convert lists back to numpy arrays
        control_points_data = data["control_points"]
        Cx = np.array(control_points_data["Cx"])
        Cy = np.array(control_points_data["Cy"])
        Cz = (
            np.array(control_points_data["Cz"])
            if "Cz" in control_points_data and control_points_data["Cz"] is not None
            else None
        )

        start_pos = tuple(data["start_pos"])
        end_pos = tuple(data["end_pos"])
        dimension = len(start_pos)

        trajectory_data = {
            "control_points": {
                "Cx": Cx,
                "Cy": Cy,
                "Cz": Cz,
            },
            "time_segments": data["time_segments"],
            "corridor": [np.array(box) for box in data["corridor"]],
            "start_pos": start_pos,
            "end_pos": end_pos,
            "config": data["config"],
            "num_segments": data["num_segments"],
            "dimension": dimension,
        }

        print(f"Successfully loaded trajectory from: {filepath}")
        print(f"Number of segments: {trajectory_data['num_segments']}")
        print(f"Total trajectory time: {sum(trajectory_data['time_segments']):.3f} seconds")

        return trajectory_data
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse JSON file: {e}")
        return None
    except KeyError as e:
        print(f"Error: Missing key in JSON file: {e}")
        return None


def evaluate_bezier_segment(Cx, Cy, Cz, n_degree, t_normalized):
    """
    Evaluates a Bezier curve segment at normalized time t (0 to 1).

    Parameters:
    -----------
    Cx, Cy : numpy.ndarray
        Control points for one segment (shape: (n+1,))
    Cz : numpy.ndarray or None
        Control points for z-axis (shape: (n+1,)) or None for 2D trajectories
    n_degree : int
        Degree of Bezier curve
    t_normalized : float or numpy.ndarray
        Normalized time parameter (0 to 1)

    Returns:
    --------
    tuple : Positions at time t. Returns (x, y) for 2D or (x, y, z) for 3D.
    """
    if isinstance(t_normalized, (int, float)):
        t_normalized = np.array([t_normalized])
        single_value = True
    else:
        single_value = False

    # Compute Bernstein basis
    bernstein_basis = np.zeros((n_degree + 1, len(t_normalized)))
    for i in range(n_degree + 1):
        bernstein_basis[i, :] = comb(n_degree, i) * (t_normalized**i) * ((1 - t_normalized)**(n_degree - i))

    # Evaluate Bezier curve
    px = bernstein_basis.T @ Cx
    py = bernstein_basis.T @ Cy
    pz = bernstein_basis.T @ Cz if Cz is not None else None

    if single_value:
        if pz is not None:
            return (px[0], py[0], pz[0])
        return (px[0], py[0])
    else:
        if pz is not None:
            return (px, py, pz)
        return (px, py)


def evaluate_trajectory_at_time(trajectory_data, time):
    """
    Evaluates the full trajectory at a specific absolute time.

    Parameters:
    -----------
    trajectory_data : dict
        Loaded trajectory data
    time : float
        Absolute time from start (seconds)

    Returns:
    --------
    tuple : Position at time t (length matches trajectory dimension), or None if time is out of range
    """
    Cx = trajectory_data["control_points"]["Cx"]
    Cy = trajectory_data["control_points"]["Cy"]
    Cz = trajectory_data["control_points"]["Cz"]
    time_segments = trajectory_data["time_segments"]
    n_degree = trajectory_data["config"]["N_DEGREE"]

    # Find which segment this time belongs to
    cumulative_time = 0
    for seg_idx, T in enumerate(time_segments):
        if cumulative_time <= time <= cumulative_time + T:
            # Normalize time to [0, 1] for this segment
            t_normalized = (time - cumulative_time) / T
            # Evaluate the segment
            segment_Cx = Cx[:, seg_idx]
            segment_Cy = Cy[:, seg_idx]
            segment_Cz = Cz[:, seg_idx] if Cz is not None else None
            return evaluate_bezier_segment(segment_Cx, segment_Cy, segment_Cz, n_degree, t_normalized)
        cumulative_time += T

    # Time is out of range
    return None


def generate_trajectory_points(trajectory_data, num_points_per_segment=100):
    """
    Generates a dense set of points along the entire trajectory.

    Parameters:
    -----------
    trajectory_data : dict
        Loaded trajectory data
    num_points_per_segment : int
        Number of points to generate per segment

    Returns:
    --------
    tuple : (times, positions)
        times : numpy.ndarray of shape (total_points,)
        positions : dict mapping axis label to numpy.ndarray of shape (total_points,)
    """
    Cx = trajectory_data["control_points"]["Cx"]
    Cy = trajectory_data["control_points"]["Cy"]
    Cz = trajectory_data["control_points"]["Cz"]
    time_segments = trajectory_data["time_segments"]
    n_degree = trajectory_data["config"]["N_DEGREE"]
    dimension = trajectory_data["dimension"]

    axis_labels = ['x', 'y'] + (['z'] if dimension == 3 else [])
    axis_data = [[] for _ in axis_labels]
    all_times = []

    cumulative_time = 0
    for seg_idx in range(len(time_segments)):
        T = time_segments[seg_idx]
        t_normalized = np.linspace(0, 1, num_points_per_segment)

        segment_Cx = Cx[:, seg_idx]
        segment_Cy = Cy[:, seg_idx]
        segment_Cz = Cz[:, seg_idx] if Cz is not None else None

        segment_values = evaluate_bezier_segment(
            segment_Cx,
            segment_Cy,
            segment_Cz,
            n_degree,
            t_normalized,
        )

        # Convert normalized time to absolute time
        times = cumulative_time + t_normalized * T
        all_times.extend(times)

        for axis_idx, values in enumerate(segment_values):
            axis_data[axis_idx].extend(values)

        cumulative_time += T

    positions = {
        axis_labels[idx]: np.array(axis_data[idx]) for idx in range(len(axis_labels))
    }

    return np.array(all_times), positions


def plot_loaded_trajectory(trajectory_data, num_points_per_segment=100):
    """
    Visualizes the loaded trajectory for both 2D and 3D cases.
    """
    times, positions = generate_trajectory_points(trajectory_data, num_points_per_segment)
    dimension = trajectory_data["dimension"]
    corridor = trajectory_data["corridor"]
    start_pos = trajectory_data["start_pos"]
    end_pos = trajectory_data["end_pos"]
    Cx = trajectory_data["control_points"]["Cx"]
    Cy = trajectory_data["control_points"]["Cy"]
    Cz = trajectory_data["control_points"]["Cz"]

    if dimension == 3:
        x = positions['x']
        y = positions['y']
        z = positions['z']

        fig = plt.figure(figsize=(12, 8))

        ax1 = fig.add_subplot(221, projection='3d')
        ax1.plot(x, y, z, 'b-', linewidth=2, label='Trajectory')
        ax1.scatter([start_pos[0]], [start_pos[1]], [start_pos[2]],
                    c='green', s=100, marker='o', label='Start')
        ax1.scatter([end_pos[0]], [end_pos[1]], [end_pos[2]],
                    c='red', s=100, marker='o', label='End')
        ax1.set_xlabel('X (radians)')
        ax1.set_ylabel('Y (radians)')
        ax1.set_zlabel('Z (radians)')
        ax1.set_title('3D Trajectory')
        ax1.legend()
        ax1.grid(True)

        ax2 = fig.add_subplot(222, projection='3d')
        for box in corridor:
            x_min, x_max, y_min, y_max, z_min, z_max = box
            origin = (x_min, y_min, z_min)
            size = (x_max - x_min, y_max - y_min, z_max - z_min)
            plot_cuboid(ax2, origin, size, 'green', alpha=0.2)
        ax2.plot(x, y, z, 'b-', linewidth=2)
        ax2.scatter([start_pos[0]], [start_pos[1]], [start_pos[2]],
                    c='green', s=100, marker='o')
        ax2.scatter([end_pos[0]], [end_pos[1]], [end_pos[2]],
                    c='red', s=100, marker='o')
        ax2.set_xlabel('X (radians)')
        ax2.set_ylabel('Y (radians)')
        ax2.set_zlabel('Z (radians)')
        ax2.set_title('Trajectory with Corridor')
        ax2.grid(True)

        ax3 = fig.add_subplot(223)
        ax3.plot(times, x, 'r-', label='X', linewidth=2)
        ax3.plot(times, y, 'g-', label='Y', linewidth=2)
        ax3.plot(times, z, 'b-', label='Z', linewidth=2)
        ax3.set_xlabel('Time (seconds)')
        ax3.set_ylabel('Position (radians)')
        ax3.set_title('Position vs Time')
        ax3.legend()
        ax3.grid(True)

        ax4 = fig.add_subplot(224, projection='3d')
        for seg_idx in range(Cx.shape[1]):
            ax4.scatter(
                Cx[:, seg_idx],
                Cy[:, seg_idx],
                Cz[:, seg_idx],
                s=50,
                alpha=0.6,
                label=f'Segment {seg_idx+1}' if seg_idx < 3 else ''
            )
        ax4.plot(x, y, z, 'b-', linewidth=2, alpha=0.5)
        ax4.set_xlabel('X (radians)')
        ax4.set_ylabel('Y (radians)')
        ax4.set_zlabel('Z (radians)')
        ax4.set_title('Control Points')
        handles, labels = ax4.get_legend_handles_labels()
        if any(label for label in labels):
            ax4.legend()
        ax4.grid(True)

        plt.tight_layout()
        plt.show()
        return

    # 2D visualization fallback
    x = positions['x']
    y = positions['y']

    fig = plt.figure(figsize=(12, 8))

    ax1 = fig.add_subplot(221)
    ax1.plot(x, y, 'b-', linewidth=2, label='Trajectory')
    ax1.scatter(start_pos[0], start_pos[1], c='green', s=100, marker='o', label='Start')
    ax1.scatter(end_pos[0], end_pos[1], c='red', s=100, marker='o', label='End')
    ax1.set_xlabel('X (radians)')
    ax1.set_ylabel('Y (radians)')
    ax1.set_title('2D Trajectory')
    ax1.legend()
    ax1.grid(True)

    ax2 = fig.add_subplot(222)
    for box in corridor:
        x_min, y_min, x_max, y_max = box
        rect = patches.Rectangle(
            (x_min, y_min),
            x_max - x_min,
            y_max - y_min,
            linewidth=1.0,
            edgecolor='black',
            facecolor='green',
            alpha=0.2,
        )
        ax2.add_patch(rect)
    ax2.plot(x, y, 'b-', linewidth=2)
    ax2.scatter(start_pos[0], start_pos[1], c='green', s=100, marker='o')
    ax2.scatter(end_pos[0], end_pos[1], c='red', s=100, marker='o')
    ax2.set_xlabel('X (radians)')
    ax2.set_ylabel('Y (radians)')
    ax2.set_title('Trajectory with Corridor')
    ax2.set_aspect('equal', adjustable='box')
    ax2.grid(True)

    ax3 = fig.add_subplot(223)
    ax3.plot(times, x, 'r-', label='X', linewidth=2)
    ax3.plot(times, y, 'b-', label='Y', linewidth=2)
    ax3.set_xlabel('Time (seconds)')
    ax3.set_ylabel('Position (radians)')
    ax3.set_title('Position vs Time')
    ax3.legend()
    ax3.grid(True)

    ax4 = fig.add_subplot(224)
    for seg_idx in range(Cx.shape[1]):
        label = f'Segment {seg_idx+1}' if seg_idx < 3 else ''
        ax4.plot(
            Cx[:, seg_idx],
            Cy[:, seg_idx],
            'o-',
            alpha=0.6,
            label=label,
        )
    ax4.set_xlabel('Control X')
    ax4.set_ylabel('Control Y')
    ax4.set_title('Control Points')
    handles, labels = ax4.get_legend_handles_labels()
    handles_labels = [(h, l) for h, l in zip(handles, labels) if l]
    if handles_labels:
        ax4.legend(*zip(*handles_labels))
    ax4.grid(True)

    plt.tight_layout()
    plt.show()


def plot_cuboid(ax, origin, size, color_face, alpha=0.2):
    """Helper function to plot a 3D cuboid."""
    x, y, z = origin
    dx, dy, dz = size
    vertices = np.array([
        [x, y, z], [x+dx, y, z], [x+dx, y+dy, z], [x, y+dy, z],
        [x, y, z+dz], [x+dx, y, z+dz], [x+dx, y+dy, z+dz], [x, y+dy, z+dz]
    ])
    faces = [
        [vertices[0], vertices[1], vertices[2], vertices[3]],
        [vertices[4], vertices[5], vertices[6], vertices[7]],
        [vertices[0], vertices[1], vertices[5], vertices[4]],
        [vertices[2], vertices[3], vertices[7], vertices[6]],
        [vertices[1], vertices[2], vertices[6], vertices[5]],
        [vertices[0], vertices[3], vertices[7], vertices[4]]
    ]
    poly3d = Poly3DCollection(faces, alpha=alpha, facecolor=color_face,
                              edgecolor='black', linewidth=0.5)
    ax.add_collection3d(poly3d)


# --- EXAMPLE USAGE ---
if __name__ == "__main__":
    # Load trajectory results
    filepath = "Results/trajectory_case2_3D_with_stops.json"
    trajectory_data = load_trajectory_results(filepath)

    if trajectory_data is not None:
        print("\n" + "="*60)
        print("TRAJECTORY DATA SUMMARY")
        print("="*60)
        print(f"Start Position: {trajectory_data['start_pos']}")
        print(f"End Position: {trajectory_data['end_pos']}")
        print(f"Number of Segments: {trajectory_data['num_segments']}")
        print(f"Time Segments: {trajectory_data['time_segments']}")
        print(f"Total Time: {sum(trajectory_data['time_segments']):.3f} seconds")
        print(f"Bezier Degree: {trajectory_data['config']['N_DEGREE']}")
        print(f"Max Velocity: {trajectory_data['config']['V_MAX']}")
        print(f"Max Acceleration: {trajectory_data['config']['A_MAX']}")
        print("="*60 + "\n")

        # Example 1: Evaluate trajectory at specific time
        total_time = sum(trajectory_data["time_segments"])
        query_time = min(1.0, total_time)
        print(f"Example 1: Evaluating trajectory at t={query_time:.2f} seconds")
        result = evaluate_trajectory_at_time(trajectory_data, time=query_time)
        if result:
            coord_str = ", ".join(f"{value:.3f}" for value in result)
            print(f"  Position at t={query_time:.2f}s: ({coord_str})")
        else:
            print("  Time is out of range")

        # Example 2: Generate dense trajectory points
        print("\nExample 2: Generating dense trajectory points")
        times, positions = generate_trajectory_points(trajectory_data, num_points_per_segment=50)
        print(f"  Generated {len(times)} points")
        axes = list(positions.keys())
        first_coords = ", ".join(f"{positions[axis][0]:.3f}" for axis in axes)
        last_coords = ", ".join(f"{positions[axis][-1]:.3f}" for axis in axes)
        print(f"  First point: ({first_coords}) at t={times[0]:.3f}s")
        print(f"  Last point: ({last_coords}) at t={times[-1]:.3f}s")

        # Example 3: Access control points directly
        print("\nExample 3: Accessing control points")
        control_points = trajectory_data["control_points"]
        for key, array in control_points.items():
            if array is None:
                continue
            print(f"  {key} shape: {array.shape}")
        print("  First segment control points:")
        for key, array in control_points.items():
            if array is None:
                continue
            print(f"    {key}: {array[:, 0]}")

        # Example 4: Visualize the trajectory
        print("\nExample 4: Plotting trajectory visualization")
        plot_loaded_trajectory(trajectory_data, num_points_per_segment=100)
