import numpy as np
# Run from the repository root: python -m cases.case_4_three_dof_manipulator.planner
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import heapq
import cvxpy as cp
from scipy.special import comb
import json
import math
import os

# --- CONFIGURATION ---
# Coordinate ranges (in radians)
X_MIN = 1/6 * math.pi # x axis minimum (radians)
X_MAX = 5/6 * math.pi  # x axis maximum (radians)
Y_MIN = -5/6 * math.pi      # y axis minimum (radians)
Y_MAX = -1/6 * math.pi       # y axis maximum (radians)

# Grid resolution: number of grid cells per radian
# Higher values = finer grid, more precise but slower
GRID_RESOLUTION = 50  # grid cells per radian

# Compute MAP_SIZE from coordinate ranges and resolution
MAP_SIZE = (
    int((X_MAX - X_MIN) * GRID_RESOLUTION),  # width in grid cells
    int((Y_MAX - Y_MIN) * GRID_RESOLUTION)   # height in grid cells
)

# Results JSON file path
RESULTS_JSON_PATH = os.path.join("Results", "results_case4_0503.json")

# Output JSON file paths
TRAJECTORY_ALPHA_BETA_JSON_PATH = os.path.join("Results", "trajectory_case4_alpha_beta.json")
TRAJECTORY_END_EFFECTOR_JSON_PATH = os.path.join("Results", "trajectory_case4_end_effector.json")

START_POS = (4/8 * math.pi, -3/4 * math.pi)  # Start position in radians (real coordinates)
END_POS = (3/8 * math.pi, -7/24 * math.pi)  # End position in radians (real coordinates)

V_MAX = 2  # Max velocity
A_MAX = 10   # Max acceleration
N_DEGREE = 5   # Degree of Bezier curve

# Visualization switches
SHOW_ASTAR_PATH = False   # Set to True to display A* path
SHOW_CORRIDOR = False      # Set to True to display corridor boxes
SHOW_CONTROL_POINTS = False  # Set to True to display Bezier curve control points

# Corridor generation sampling
CORRIDOR_SAMPLE_STEP = 5 # Sample path every CORRIDOR_SAMPLE_STEP points (1 = use all points)

# --- STEP 0: MAP GENERATION ---
def load_obstacles_from_json(json_path):
    """
    Loads obstacles from JSON file.
    Returns list of boxes as [x_min, x_max, y_min, y_max] in radians.
    """
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)

        obstacles = []
        # Add undetermined boxes
        if 'undetermined' in data:
            obstacles.extend(data['undetermined'])

        # Add out_of_workspace boxes
        if 'out_of_workspace' in data:
            obstacles.extend(data['out_of_workspace'])

        print(f"Loaded {len(obstacles)} obstacles from JSON file.")
        return obstacles
    except FileNotFoundError:
        print(f"Error: JSON file not found at {json_path}")
        return []
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse JSON file: {e}")
        return []

def rad_to_grid_idx(x_rad, y_rad):
    """
    Converts radian coordinates to grid indices for array access.

    Parameters:
    - x_rad, y_rad: coordinates in radians

    Returns:
    - grid_x, grid_y: grid indices (clamped to valid range)
    """
    grid_x = int((x_rad - X_MIN) / (X_MAX - X_MIN) * MAP_SIZE[0])
    grid_y = int((y_rad - Y_MIN) / (Y_MAX - Y_MIN) * MAP_SIZE[1])
    return max(0, min(grid_x, MAP_SIZE[0] - 1)), max(0, min(grid_y, MAP_SIZE[1] - 1))

def grid_idx_to_rad(grid_x, grid_y):
    """
    Converts grid indices back to radian coordinates.

    Parameters:
    - grid_x, grid_y: grid indices

    Returns:
    - x_rad, y_rad: coordinates in radians
    """
    x_rad = X_MIN + (grid_x / MAP_SIZE[0]) * (X_MAX - X_MIN)
    y_rad = Y_MIN + (grid_y / MAP_SIZE[1]) * (Y_MAX - Y_MIN)
    return x_rad, y_rad

def generate_map():
    """Generates a grid with obstacles loaded from JSON file."""
    grid = np.zeros(MAP_SIZE, dtype=int)

    # Load obstacles from JSON file (already in radians)
    obstacles = load_obstacles_from_json(RESULTS_JSON_PATH)

    # Place obstacles in grid
    for box in obstacles:
        if len(box) >= 4:
            x_min_rad, x_max_rad, y_min_rad, y_max_rad = box[0], box[1], box[2], box[3]
            # Convert to grid indices
            grid_x_min = int((x_min_rad - X_MIN) / (X_MAX - X_MIN) * MAP_SIZE[0])
            grid_x_max = int((x_max_rad - X_MIN) / (X_MAX - X_MIN) * MAP_SIZE[0])
            grid_y_min = int((y_min_rad - Y_MIN) / (Y_MAX - Y_MIN) * MAP_SIZE[1])
            grid_y_max = int((y_max_rad - Y_MIN) / (Y_MAX - Y_MIN) * MAP_SIZE[1])

            # Clamp and place obstacles
            grid_x_min = max(0, min(grid_x_min, MAP_SIZE[0] - 1))
            grid_x_max = max(0, min(grid_x_max, MAP_SIZE[0]))
            grid_y_min = max(0, min(grid_y_min, MAP_SIZE[1] - 1))
            grid_y_max = max(0, min(grid_y_max, MAP_SIZE[1]))

            if grid_x_max > grid_x_min and grid_y_max > grid_y_min:
                grid[grid_x_min:grid_x_max, grid_y_min:grid_y_max] = 1

    # Ensure start and end positions are clear
    start_grid_x, start_grid_y = rad_to_grid_idx(START_POS[0], START_POS[1])
    end_grid_x, end_grid_y = rad_to_grid_idx(END_POS[0], END_POS[1])

    for gx, gy in [(start_grid_x, start_grid_y), (end_grid_x, end_grid_y)]:
        x_start, y_start = max(0, gx-1), max(0, gy-1)
        x_end, y_end = min(MAP_SIZE[0], gx+2), min(MAP_SIZE[1], gy+2)
        grid[x_start:x_end, y_start:y_end] = 0

    return grid

# --- STEP 1: A* PATH FINDING ---
def find_path_astar(grid, start, end):
    """Custom A* pathfinding implementation."""

    def heuristic(a, b):
        return np.sqrt((b[0] - a[0])**2 + (b[1] - a[1])**2)

    open_list = []
    # (f_score, g_score, position)
    heapq.heappush(open_list, (0, 0, start))

    came_from = {}
    g_score = { (r,c): float('inf') for r in range(grid.shape[0]) for c in range(grid.shape[1]) }
    g_score[start] = 0

    f_score = { (r,c): float('inf') for r in range(grid.shape[0]) for c in range(grid.shape[1]) }
    f_score[start] = heuristic(start, end)

    open_list_hash = {start}

    while open_list:
        _, _, current = heapq.heappop(open_list)
        open_list_hash.remove(current)

        if current == end:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return np.array(path[::-1])

        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
            neighbor = (current[0] + dx, current[1] + dy)

            if not (0 <= neighbor[0] < grid.shape[0] and 0 <= neighbor[1] < grid.shape[1]):
                continue

            if grid[neighbor[0], neighbor[1]] == 1:
                continue

            # distance between current and neighbor is sqrt(dx^2 + dy^2)
            move_cost = np.sqrt(dx**2 + dy**2)
            tentative_g_score = g_score[current] + move_cost

            if tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, end)
                if neighbor not in open_list_hash:
                    heapq.heappush(open_list, (f_score[neighbor], g_score[neighbor], neighbor))
                    open_list_hash.add(neighbor)

    print("A* failed to find a path.")
    return None

# --- STEP 2: CORRIDOR GENERATION (IMPROVED) ---
def generate_corridor(grid, path):
    """
    Generates a corridor of overlapping axis-aligned rectangles.
    This method iteratively finds the largest rectangle that can cover a segment of the path.
    Samples the path every CORRIDOR_SAMPLE_STEP points before generating corridors.
    Path is in grid indices, returns corridor boxes in radians.
    """
    # Sample the path - always include first and last points
    if CORRIDOR_SAMPLE_STEP > 1:
        sampled_indices = list(range(0, len(path), CORRIDOR_SAMPLE_STEP))
        if sampled_indices[-1] != len(path) - 1:
            sampled_indices.append(len(path) - 1)
        sampled_path = path[sampled_indices]
    else:
        sampled_path = path

    corridor = []
    path_idx = 0

    while path_idx < len(sampled_path):
        start_point = sampled_path[path_idx]

        # 1. Find the maximum path segment that can be enclosed in a growing box
        x_min, y_min = start_point
        x_max, y_max = min(grid.shape[0], start_point[0] + 1), min(grid.shape[1], start_point[1] + 1)

        end_path_idx = path_idx
        for i in range(path_idx + 1, len(sampled_path)):
            point = sampled_path[i]

            temp_x_min = min(x_min, point[0])
            temp_y_min = min(y_min, point[1])
            temp_x_max = max(x_max, point[0] + 1)
            temp_y_max = max(y_max, point[1] + 1)

            temp_x_min = max(0, temp_x_min)
            temp_y_min = max(0, temp_y_min)
            temp_x_max = min(grid.shape[0], temp_x_max)
            temp_y_max = min(grid.shape[1], temp_y_max)

            if np.any(grid[temp_x_min:temp_x_max, temp_y_min:temp_y_max] == 1):
                break

            x_min, y_min, x_max, y_max = temp_x_min, temp_y_min, temp_x_max, temp_y_max
            end_path_idx = i

        # 2. Inflate the bounding box as much as possible
        if x_max > x_min and y_max > y_min:
            while x_min > 0 and y_max > y_min and np.all(grid[x_min - 1, y_min:y_max] == 0):
                x_min -= 1
            while x_max < grid.shape[0] and y_max > y_min and np.all(grid[x_max, y_min:y_max] == 0):
                x_max += 1
            while y_min > 0 and x_max > x_min and np.all(grid[x_min:x_max, y_min - 1] == 0):
                y_min -= 1
            while y_max < grid.shape[1] and x_max > x_min and np.all(grid[x_min:x_max, y_max] == 0):
                y_max += 1

        # Convert corridor box from grid indices to radians
        x_min_rad, y_min_rad = grid_idx_to_rad(x_min, y_min)
        x_max_rad, y_max_rad = grid_idx_to_rad(x_max, y_max)
        box = np.array([x_min_rad, y_min_rad, x_max_rad, y_max_rad])
        corridor.append(box)

        # 3. Move to the next path point
        if end_path_idx > path_idx and end_path_idx < len(sampled_path) - 1:
             path_idx = end_path_idx
        else:
             path_idx = end_path_idx + 1

    return corridor

# --- Print Segment Information ---
def print_segment_info(start_pos, end_pos, corridor, time_segments, Cx=None, Cy=None):
    """
    Prints the start point, end point, and time for each trajectory segment.

    Parameters:
    - start_pos: Starting position (x, y)
    - end_pos: Ending position (x, y)
    - corridor: List of corridor boxes
    - time_segments: List of time durations for each segment
    - Cx, Cy: Optional optimized control points (if available, uses them for endpoints)
    """
    print("\n" + "="*60)
    print("TRAJECTORY SEGMENT INFORMATION")
    print("="*60)

    m = len(corridor)

    for i in range(m):
        print(f"\nSegment {i+1}/{m}:")
        print("-" * 40)

        # Determine start point
        if i == 0:
            start_point = start_pos
            start_label = "START"
        else:
            # Use junction point between segments
            if Cx is not None and Cy is not None:
                # Use optimized endpoint from previous segment
                start_point = (Cx[N_DEGREE, i-1], Cy[N_DEGREE, i-1])
            else:
                # Use corridor center as approximation
                prev_box = corridor[i-1]
                start_point = ((prev_box[0] + prev_box[2]) / 2,
                              (prev_box[1] + prev_box[3]) / 2)
            start_label = "JUNCTION"

        # Determine end point
        if i == m - 1:
            end_point = end_pos
            end_label = "END"
        else:
            # Use junction point
            if Cx is not None and Cy is not None:
                # Use optimized endpoint
                end_point = (Cx[N_DEGREE, i], Cy[N_DEGREE, i])
            else:
                # Use corridor center as approximation
                box = corridor[i]
                end_point = ((box[0] + box[2]) / 2,
                            (box[1] + box[3]) / 2)
            end_label = "JUNCTION"

        # Get time duration
        T = time_segments[i]

        # Print information
        print(f"  Start Point ({start_label}): ({start_point[0]:.3f}, {start_point[1]:.3f})")
        print(f"  End Point   ({end_label}):   ({end_point[0]:.3f}, {end_point[1]:.3f})")
        print(f"  Duration:   {T:.3f} seconds")

        # Calculate distance for reference
        dist = np.linalg.norm(np.array(end_point) - np.array(start_point))
        avg_velocity = dist / T if T > 0 else 0
        print(f"  Distance:   {dist:.3f} units")
        print(f"  Avg Speed: {avg_velocity:.3f} units/s")

    print("\n" + "="*60)
    total_time = sum(time_segments)
    print(f"Total Trajectory Time: {total_time:.3f} seconds")
    print("="*60 + "\n")



def compute_bernstein_basis(n, t_vec):
    """
    Computes the Bernstein basis matrix for Bezier curves of degree n.

    Parameters:
    - n: Degree of the Bezier curve.
    - t_vec: 1D array of parameter values in [0, 1].

    Returns:
    - basis: Array of shape ((n + 1), len(t_vec)).
    """
    basis = np.zeros((n + 1, len(t_vec)))
    for i in range(n + 1):
        basis[i, :] = comb(n, i) * (t_vec**i) * ((1 - t_vec)**(n - i))
    return basis


# --- STEP 3: TRAJECTORY OPTIMIZATION (GAO ET AL.) ---
def optimize_trajectory(start_pos, end_pos, corridor):
    if len(corridor) == 0:
        print("Error: Empty corridor, cannot optimize trajectory.")
        return None, None, []

    n = N_DEGREE
    m = len(corridor)

    # Control points for x and y
    Cx = cp.Variable((n + 1, m))
    Cy = cp.Variable((n + 1, m))

    # --- Objective Function: Minimize Jerk ---1
    cost = 0
    # Time allocation (heuristic)
    time_segments = []

    current_pos = np.array(start_pos)

    for i in range(m):
        if i == m - 1:
            target_pos = np.array(end_pos)
        else:
            # Find intersection with next box
            curr_box = corridor[i]
            next_box = corridor[i+1]

            x_min = max(curr_box[0], next_box[0])
            y_min = max(curr_box[1], next_box[1])
            x_max = min(curr_box[2], next_box[2])
            y_max = min(curr_box[3], next_box[3])

            target_pos = np.array([(x_min + x_max)/2, (y_min + y_max)/2])

        dist = np.linalg.norm(target_pos - current_pos)
        time_segments.append(max(dist / V_MAX * 11.0, 0.5))
        current_pos = target_pos

    for i in range(m):
        T = time_segments[i]
        # Velocity control points
        Vx = (n / T) * (Cx[1:, i] - Cx[:-1, i])
        Vy = (n / T) * (Cy[1:, i] - Cy[:-1, i])
        # Acceleration control points
        Ax = (n - 1) / T * (Vx[1:] - Vx[:-1])
        Ay = (n - 1) / T * (Vy[1:] - Vy[:-1])
        # Jerk control points
        Jx = (n - 2) / T * (Ax[1:] - Ax[:-1])
        Jy = (n - 2) / T * (Ay[1:] - Ay[:-1])
        cost += cp.sum_squares(Jx) + cp.sum_squares(Jy)

    # --- Constraints ---
    constraints = []

    # 1. Waypoint Constraints (Start and End)
    # Start position, velocity, acceleration
    constraints += [Cx[0, 0] == start_pos[0], Cy[0, 0] == start_pos[1]]
    constraints += [Cx[1, 0] - Cx[0, 0] == 0, Cy[1, 0] - Cy[0, 0] == 0] # v_start = 0
    constraints += [Cx[2, 0] - 2*Cx[1, 0] + Cx[0, 0] == 0, Cy[2, 0] - 2*Cy[1, 0] + Cy[0, 0] == 0] # a_start = 0

    # End position, velocity, acceleration
    constraints += [Cx[n, m-1] == end_pos[0], Cy[n, m-1] == end_pos[1]]
    constraints += [Cx[n, m-1] - Cx[n-1, m-1] == 0, Cy[n, m-1] - Cy[n-1, m-1] == 0] # v_end = 0
    constraints += [Cx[n, m-1] - 2*Cx[n-1, m-1] + Cx[n-2, m-1] == 0,
                    Cy[n, m-1] - 2*Cy[n-1, m-1] + Cy[n-2, m-1] == 0] # a_end = 0

    # 2. Continuity Constraints (at segment junctions)
    for i in range(m - 1):
        T1, T2 = time_segments[i], time_segments[i+1]
        # Position
        constraints += [Cx[n, i] == Cx[0, i+1], Cy[n, i] == Cy[0, i+1]]
        # Velocity
        constraints += [(1/T1)*(Cx[n, i] - Cx[n-1, i]) == (1/T2)*(Cx[1, i+1] - Cx[0, i+1])]
        constraints += [(1/T1)*(Cy[n, i] - Cy[n-1, i]) == (1/T2)*(Cy[1, i+1] - Cy[0, i+1])]
        # Acceleration
        constraints += [(1/T1**2)*(Cx[n, i] - 2*Cx[n-1, i] + Cx[n-2, i]) ==
                        (1/T2**2)*(Cx[2, i+1] - 2*Cx[1, i+1] + Cx[0, i+1])]
        constraints += [(1/T1**2)*(Cy[n, i] - 2*Cy[n-1, i] + Cy[n-2, i]) ==
                        (1/T2**2)*(Cy[2, i+1] - 2*Cy[1, i+1] + Cy[0, i+1])]

    # 3. Safety & Dynamic Constraints
    for i in range(m):
        box = corridor[i]
        T = time_segments[i]

        # Safety (Convex Hull Property)
        constraints += [Cx[:, i] >= box[0], Cx[:, i] <= box[2]]
        constraints += [Cy[:, i] >= box[1], Cy[:, i] <= box[3]]

        # Dynamics (Hodograph Property)
        # Velocity control points
        Vx = (n / T) * (Cx[1:, i] - Cx[:-1, i])
        Vy = (n / T) * (Cy[1:, i] - Cy[:-1, i])
        # Constrain velocity magnitude (not just components)
        # Component-wise constraints allow sqrt(vx^2 + vy^2) > V_MAX
        # Use squared magnitude constraint: vx^2 + vy^2 <= V_MAX^2 (OSQP compatible)
        # Vx has n elements (from Cx[1:] which has n elements)
        for j in range(n):
            constraints += [cp.square(Vx[j]) + cp.square(Vy[j]) <= V_MAX**2]

        # Acceleration control points
        Ax = (n - 1) / T * (Vx[1:] - Vx[:-1])
        Ay = (n - 1) / T * (Vy[1:] - Vy[:-1])
        # Constrain acceleration magnitude using squared constraint
        # Ax has n-1 elements (from Vx[1:] which has n-1 elements)
        for j in range(n - 1):
            constraints += [cp.square(Ax[j]) + cp.square(Ay[j]) <= A_MAX**2]

    # --- Solve QP ---
    prob = cp.Problem(cp.Minimize(cost), constraints)
    # OSQP doesn't support quadratic constraints, use ECOS or let cvxpy choose automatically
    # Try ECOS first (supports SOCP), then fallback to automatic solver selection
    try:
        prob.solve(solver=cp.ECOS, verbose=False)
    except cp.SolverError:
        # Fallback to automatic solver selection (will use available conic solver)
        prob.solve(verbose=False)

    if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        print(f"Optimization failed with status: {prob.status}")
        return None, None, time_segments

    if prob.status == cp.OPTIMAL_INACCURATE:
        print("Warning: Optimization was inaccurate.")

    return Cx.value, Cy.value, time_segments

# --- STEP 4: VISUALIZATION ---
def robot_arm_kinematics(alpha, beta):
    """Compute end-effector position for the first magnet."""
    alpha = np.asarray(alpha)
    beta = np.asarray(beta)

    x_base = 0.0
    y_base = 0.023
    z_base = 0.079
    L1 = 0.02773

    x1 = L1 * np.sin(alpha) * np.cos(beta) + x_base
    y1 = L1 * np.sin(alpha) * np.sin(beta) + y_base
    z1 = L1 * np.cos(alpha) + z_base

    return x1, y1, z1


def plot_results(grid, path, corridor, Cx, Cy, show_astar_path=True, show_corridor=True, show_control_points=True, time_segments=None):
    # Figure 1: 2D alpha-beta space
    fig, ax = plt.subplots(figsize=(6, 6))

    # Map and obstacles - use radian extent for axes
    ax.imshow(grid.T, cmap='Greys', origin='lower',
              extent=[X_MIN, X_MAX, Y_MIN, Y_MAX])

    # Corridor - already in radians
    if show_corridor:
        for box in corridor:
            x_min_rad, y_min_rad, x_max_rad, y_max_rad = box[0], box[1], box[2], box[3]
            rect = patches.Rectangle(
                (x_min_rad, y_min_rad),
                x_max_rad - x_min_rad,
                y_max_rad - y_min_rad,
                linewidth=1, edgecolor='g', facecolor='g', alpha=0.2
            )
            ax.add_patch(rect)

    # A* Path - convert from grid indices to radians
    if show_astar_path and path is not None:
        path_rad = np.zeros_like(path, dtype=float)
        for i, (grid_x, grid_y) in enumerate(path):
            x_rad, y_rad = grid_idx_to_rad(grid_x + 0.5, grid_y + 0.5)
            path_rad[i] = [x_rad, y_rad]
        ax.plot(path_rad[:, 0], path_rad[:, 1], 'b-o', label='A* Path')

    # Start and End - already in radians
    ax.plot(START_POS[0], START_POS[1], 'go', markersize=10, label='Start')
    ax.plot(END_POS[0], END_POS[1], 'ro', markersize=10, label='End')

    # Bezier Trajectory - already in radians
    custom_traj_handle = None
    if Cx is not None and Cy is not None:
        t_vec = np.linspace(0, 1, 100)
        n = N_DEGREE
        bernstein_basis = compute_bernstein_basis(n, t_vec)

        use_time_coloring = (
            time_segments is not None
            and len(time_segments) == Cx.shape[1]
            and len(time_segments) > 0
        )
        total_time = float(sum(time_segments)) if use_time_coloring else 0.0
        if use_time_coloring and total_time <= 0:
            use_time_coloring = False

        if use_time_coloring:
            cmap = plt.get_cmap('viridis')
            norm = Normalize(vmin=0.0, vmax=total_time if total_time > 0 else 1.0)
            trajectory_segments = []
            segment_time_values = []
            cumulative_time = 0.0
            custom_traj_handle = plt.Line2D([0], [0], color=cmap(0.8), linewidth=2.5, label='Trajectory')

        for i in range(Cx.shape[1]):
            # Control points are already in radians
            px = bernstein_basis.T @ Cx[:, i]
            py = bernstein_basis.T @ Cy[:, i]

            if use_time_coloring:
                segment_duration = time_segments[i]
                seg_times = cumulative_time + t_vec * segment_duration
                segment_points = np.column_stack((px, py))
                for j in range(len(segment_points) - 1):
                    trajectory_segments.append(
                        np.array([
                            segment_points[j],
                            segment_points[j + 1]
                        ])
                    )
                    segment_time_values.append(float((seg_times[j] + seg_times[j + 1]) * 0.5))
                cumulative_time += segment_duration
            else:
                ax.plot(px, py, 'r-', linewidth=2.5, label='Trajectory' if i == 0 else '')

            if show_control_points:
                ax.plot(
                    Cx[:, i],
                    Cy[:, i],
                    'c--o',
                    markersize=4,
                    alpha=0.7,
                    label='Control Points' if i == 0 else ''
                )

        if use_time_coloring and trajectory_segments:
            lc = LineCollection(trajectory_segments, cmap=cmap, norm=norm, linewidth=2.5)
            lc.set_array(np.array(segment_time_values))
            lc.set_label('Trajectory')
            ax.add_collection(lc)
            cbar = fig.colorbar(lc, ax=ax, pad=0.02, fraction=0.046)
            cbar.set_label('Time (s)', fontsize=20)
            cbar.ax.tick_params(labelsize=18)
        else:
            custom_traj_handle = None

    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel(r'$\alpha_1$ (rad)', fontsize=20)
    ax.set_ylabel(r'$\beta_1$ (rad)', fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=18)

    legend_handles, legend_labels = ax.get_legend_handles_labels()
    if custom_traj_handle is not None and 'Trajectory' not in legend_labels:
        legend_handles.append(custom_traj_handle)
        legend_labels.append('Trajectory')
    if legend_handles:
        ax.legend(legend_handles, legend_labels, fontsize=15, loc='upper right')
    # ax.set_title("Quadrotor Motion Planning")
    ticksx = [1/6 * math.pi, 1/3 * math.pi, 1/2 * math.pi, 2/3 * math.pi, 5/6 * math.pi]
    labelsx = ['π/6', 'π/3', 'π/2', '2π/3', '5π/6']

    ticksy = [-5/6 * math.pi, -2/3 * math.pi, -1/2 * math.pi, -1/3 * math.pi, -1/6 * math.pi]
    labelsy = ['-5π/6', '-2π/3', '-π/2', '-π/3', '-π/6']

    ax.set_xticks(ticksx, labelsx)
    ax.set_yticks(ticksy, labelsy)
    # ax.set_zticks(ticks, labels)

    plt.tight_layout()
    plt.grid(True)
    plt.show()

    # Figure 2: 3D end-effector space
    fig2 = plt.figure(figsize=(8, 6))
    ax2 = fig2.add_subplot(111, projection='3d')

    # Conversion factor: meters to centimeters
    M_TO_CM = 100.0

    # Set view angle
    ax2.view_init(elev=20, azim=250)

    # Generate workspace boundary (similar to Plot_case4_end.py)
    alpha_range = (X_MIN, X_MAX)
    beta_range = (Y_MIN, Y_MAX)

    # Generate boundary curves
    samples = 200
    beta_edge = np.linspace(beta_range[0], beta_range[1], samples)
    alpha_edge = np.linspace(alpha_range[0], alpha_range[1], samples)

    boundary_params = [
        (np.full(samples, alpha_range[0]), beta_edge),  # alpha = alpha_min
        (np.full(samples, alpha_range[1]), beta_edge),  # alpha = alpha_max
        (alpha_edge, np.full(samples, beta_range[0])),  # beta = beta_min
        (alpha_edge, np.full(samples, beta_range[1])),  # beta = beta_max
    ]

    for alpha_vals, beta_vals in boundary_params:
        x, y, z = robot_arm_kinematics(alpha_vals, beta_vals)
        ax2.plot(x * M_TO_CM, y * M_TO_CM, z * M_TO_CM, color="black", linewidth=2.0)

    # Plot obstacles as black regions
    # Load obstacles from JSON and convert to 3D
    obstacles = load_obstacles_from_json(RESULTS_JSON_PATH)
    obstacle_sample_density = 10
    for box in obstacles:
        if len(box) >= 4:
            # Obstacle box format: [x_min, x_max, y_min, y_max] in radians
            x_min, x_max, y_min, y_max = box[0], box[1], box[2], box[3]
            alpha_obs = np.linspace(x_min, x_max, obstacle_sample_density)
            beta_obs = np.linspace(y_min, y_max, obstacle_sample_density)
            alpha_mesh, beta_mesh = np.meshgrid(alpha_obs, beta_obs, indexing='ij')
            x_obs, y_obs, z_obs = robot_arm_kinematics(alpha_mesh.ravel(), beta_mesh.ravel())
            ax2.scatter(x_obs * M_TO_CM, y_obs * M_TO_CM, z_obs * M_TO_CM,
                       c='black', s=5, alpha=0.3)

    # Initialize trajectory lists
    trajectory_x_all = []
    trajectory_y_all = []
    trajectory_z_all = []
    trajectory_time_all = []

    # Convert trajectory from alpha-beta to 3D end-effector space
    if Cx is not None and Cy is not None:
        t_vec = np.linspace(0, 1, 100)
        n = N_DEGREE
        bernstein_basis = compute_bernstein_basis(n, t_vec)

        use_time_coloring = (
            time_segments is not None
            and len(time_segments) == Cx.shape[1]
            and len(time_segments) > 0
        )
        total_time = float(sum(time_segments)) if use_time_coloring else 0.0
        if use_time_coloring and total_time <= 0:
            use_time_coloring = False

        cumulative_time = 0.0
        for i in range(Cx.shape[1]):
            px = bernstein_basis.T @ Cx[:, i]
            py = bernstein_basis.T @ Cy[:, i]

            # Convert to 3D end-effector positions
            x_ee, y_ee, z_ee = robot_arm_kinematics(px, py)

            if use_time_coloring:
                segment_duration = time_segments[i]
                seg_times = cumulative_time + t_vec * segment_duration
                trajectory_x_all.extend(x_ee * M_TO_CM)
                trajectory_y_all.extend(y_ee * M_TO_CM)
                trajectory_z_all.extend(z_ee * M_TO_CM)
                trajectory_time_all.extend(seg_times)
                cumulative_time += segment_duration
            else:
                # Collect for aspect ratio even if not using time coloring
                trajectory_x_all.extend(x_ee * M_TO_CM)
                trajectory_y_all.extend(y_ee * M_TO_CM)
                trajectory_z_all.extend(z_ee * M_TO_CM)
                ax2.plot(x_ee * M_TO_CM, y_ee * M_TO_CM, z_ee * M_TO_CM,
                        'r-', linewidth=2.5, label='Trajectory' if i == 0 else '', zorder=1)

        if use_time_coloring and trajectory_x_all:
            scatter = ax2.scatter(trajectory_x_all, trajectory_y_all, trajectory_z_all,
                                 c=trajectory_time_all, cmap='viridis', s=20, alpha=0.7,
                                 zorder=1, label='Trajectory')
            cbar = fig2.colorbar(scatter, ax=ax2, pad=0.02, fraction=0.046)
            cbar.set_label('Time (s)', fontsize=20)
            cbar.ax.tick_params(labelsize=18)

    # Plot start and end points in 3D (make them more visible)
    x_start, y_start, z_start = robot_arm_kinematics(START_POS[0], START_POS[1])
    x_end, y_end, z_end = robot_arm_kinematics(END_POS[0], END_POS[1])

    # Plot start point with larger size, edge color, and higher zorder
    ax2.scatter([x_start * M_TO_CM], [y_start * M_TO_CM], [z_start * M_TO_CM],
               c='green', s=300, marker='o', linewidths=2.5,
               label='Start', zorder=100, alpha=0.9)
    # Plot end point with larger size, edge color, and higher zorder
    ax2.scatter([x_end * M_TO_CM], [y_end * M_TO_CM], [z_end * M_TO_CM],
               c='red', s=300, marker='o', linewidths=2.5,
               label='End', zorder=100, alpha=0.9)

    # Plot base point
    x_base = 0.0
    y_base = 0.023
    z_base = 0.079
    # ax2.scatter([x_base * M_TO_CM], [y_base * M_TO_CM], [z_base * M_TO_CM],
    #            marker='x', s=200, color='red', linewidths=3, label='Base point')

    ax2.set_xlabel("X (cm)", fontsize=20, labelpad=10)
    ax2.set_ylabel("Y (cm)", fontsize=20, labelpad=10)
    ax2.set_zlabel("Z (cm)", fontsize=20, labelpad=10)
    ax2.tick_params(axis='both', which='major', labelsize=18)

    # Set equal aspect ratio
    all_x_coords = []
    all_y_coords = []
    all_z_coords = []

    # Collect boundary points
    for alpha_vals, beta_vals in boundary_params:
        x, y, z = robot_arm_kinematics(alpha_vals, beta_vals)
        all_x_coords.extend(x * M_TO_CM)
        all_y_coords.extend(y * M_TO_CM)
        all_z_coords.extend(z * M_TO_CM)

    # Collect trajectory points
    if Cx is not None and Cy is not None and trajectory_x_all:
        all_x_coords.extend(trajectory_x_all)
        all_y_coords.extend(trajectory_y_all)
        all_z_coords.extend(trajectory_z_all)

    # Add start/end/base points
    all_x_coords.extend([x_start * M_TO_CM, x_end * M_TO_CM, x_base * M_TO_CM])
    all_y_coords.extend([y_start * M_TO_CM, y_end * M_TO_CM, y_base * M_TO_CM])
    all_z_coords.extend([z_start * M_TO_CM, z_end * M_TO_CM, z_base * M_TO_CM])

    if all_x_coords:
        all_x_coords = np.array(all_x_coords)
        all_y_coords = np.array(all_y_coords)
        all_z_coords = np.array(all_z_coords)

        ranges = np.array([
            all_x_coords.max() - all_x_coords.min(),
            all_y_coords.max() - all_y_coords.min(),
            all_z_coords.max() - all_z_coords.min()
        ])
        max_range = ranges.max()
        mid_x = (all_x_coords.max() + all_x_coords.min()) / 2.0
        mid_y = (all_y_coords.max() + all_y_coords.min()) / 2.0
        mid_z = (all_z_coords.max() + all_z_coords.min()) / 2.0

        ax2.set_xlim(mid_x - max_range / 2.0, mid_x + max_range / 2.0)
        ax2.set_ylim(mid_y - max_range / 2.0, mid_y + max_range / 2.0)
        ax2.set_zlim(mid_z - max_range / 2.0, mid_z + max_range / 2.0)

    if ax2.get_legend_handles_labels()[0]:
        ax2.legend(fontsize=20, loc='upper right')

    plt.tight_layout()
    plt.show()

def save_alpha_beta_trajectory_results(Cx, Cy, time_segments, corridor, start_pos, end_pos, filepath=None):
    """
    Saves trajectory results in the alpha-beta space to a JSON file.
    """
    if filepath is None:
        filepath = TRAJECTORY_ALPHA_BETA_JSON_PATH

    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    corridor_serialized = [
        box.tolist() if hasattr(box, "tolist") else list(box) for box in corridor
    ]

    trajectory_data = {
        "control_points": {
            "Cx": Cx.tolist() if hasattr(Cx, "tolist") else Cx,
            "Cy": Cy.tolist() if hasattr(Cy, "tolist") else Cy,
        },
        "time_segments": [float(t) for t in time_segments],
        "corridor": corridor_serialized,
        "start_pos": [float(start_pos[0]), float(start_pos[1])],
        "end_pos": [float(end_pos[0]), float(end_pos[1])],
        "config": {
            "N_DEGREE": N_DEGREE,
            "V_MAX": V_MAX,
            "A_MAX": A_MAX,
        },
        "num_segments": len(time_segments),
        "total_time": float(sum(time_segments)),
    }

    with open(filepath, "w") as f:
        json.dump(trajectory_data, f, indent=2)

    print(f"\nAlpha-beta trajectory results saved to: {filepath}")
    return filepath


def save_end_effector_trajectory_results(
    Cx,
    Cy,
    time_segments,
    start_pos,
    end_pos,
    filepath=None,
    samples_per_segment=100,
):
    """
    Saves end-effector trajectory samples (in centimeters) derived from alpha-beta control
    points to a JSON file.
    """
    if filepath is None:
        filepath = TRAJECTORY_END_EFFECTOR_JSON_PATH

    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    if Cx is None or Cy is None:
        raise ValueError("Control points cannot be None when saving trajectory results.")

    if time_segments is None or len(time_segments) == 0:
        raise ValueError("time_segments must be a non-empty list.")

    if Cx.shape[1] != len(time_segments) or Cy.shape[1] != len(time_segments):
        raise ValueError("Control points and time_segments must have matching segment counts.")

    t_vec = np.linspace(0, 1, samples_per_segment)
    bernstein_basis = compute_bernstein_basis(N_DEGREE, t_vec)

    samples = []
    cumulative_time = 0.0
    M_TO_CM = 100.0

    for segment_idx in range(Cx.shape[1]):
        px = bernstein_basis.T @ Cx[:, segment_idx]
        py = bernstein_basis.T @ Cy[:, segment_idx]

        segment_duration = time_segments[segment_idx]
        seg_times = cumulative_time + t_vec * segment_duration

        x_ee, y_ee, z_ee = robot_arm_kinematics(px, py)
        x_cm = x_ee * M_TO_CM
        y_cm = y_ee * M_TO_CM
        z_cm = z_ee * M_TO_CM

        for k in range(len(t_vec)):
            samples.append(
                {
                    "time": float(seg_times[k]),
                    "position_cm": [
                        float(x_cm[k]),
                        float(y_cm[k]),
                        float(z_cm[k]),
                    ],
                    "alpha_beta_rad": [
                        float(px[k]),
                        float(py[k]),
                    ],
                    "segment_index": int(segment_idx),
                }
            )

        cumulative_time += segment_duration

    x_start, y_start, z_start = robot_arm_kinematics(start_pos[0], start_pos[1])
    x_end, y_end, z_end = robot_arm_kinematics(end_pos[0], end_pos[1])

    trajectory_data = {
        "samples": samples,
        "time_segments": [float(t) for t in time_segments],
        "start": {
            "alpha_beta_rad": [float(start_pos[0]), float(start_pos[1])],
            "position_cm": [
                float(x_start * M_TO_CM),
                float(y_start * M_TO_CM),
                float(z_start * M_TO_CM),
            ],
        },
        "end": {
            "alpha_beta_rad": [float(end_pos[0]), float(end_pos[1])],
            "position_cm": [
                float(x_end * M_TO_CM),
                float(y_end * M_TO_CM),
                float(z_end * M_TO_CM),
            ],
        },
        "config": {
            "N_DEGREE": N_DEGREE,
            "samples_per_segment": samples_per_segment,
        },
        "num_segments": len(time_segments),
        "total_time": float(sum(time_segments)),
    }

    with open(filepath, "w") as f:
        json.dump(trajectory_data, f, indent=2)

    print(f"End-effector trajectory results saved to: {filepath}")
    return filepath


# --- MAIN EXECUTION ---
if __name__ == '__main__':
    grid = generate_map()

    # Convert start/end positions to grid indices for pathfinding
    start_grid = rad_to_grid_idx(START_POS[0], START_POS[1])
    end_grid = rad_to_grid_idx(END_POS[0], END_POS[1])

    path = find_path_astar(grid, start_grid, end_grid)

    if path is not None:
        # Generate corridor (returns boxes in radians)
        corridor = generate_corridor(grid, path)
        # Optimize trajectory (works in radians)
        Cx, Cy, time_segments = optimize_trajectory(START_POS, END_POS, corridor)

        if Cx is not None and Cy is not None:
            # Print segment information (all in radians)
            print_segment_info(START_POS, END_POS, corridor, time_segments, Cx, Cy)

            # Save trajectory results to JSON files
            save_alpha_beta_trajectory_results(Cx, Cy, time_segments, corridor, START_POS, END_POS)
            save_end_effector_trajectory_results(Cx, Cy, time_segments, START_POS, END_POS)

            # Plot with visualization switches
            plot_results(
                grid,
                path,
                corridor,
                Cx,
                Cy,
                show_astar_path=SHOW_ASTAR_PATH,
                show_corridor=SHOW_CORRIDOR,
                show_control_points=SHOW_CONTROL_POINTS,
                time_segments=time_segments,
            )
        else:
            print("Optimization failed, cannot print segment information.")
    else:
        print("Pathfinding failed, cannot proceed with optimization.")
