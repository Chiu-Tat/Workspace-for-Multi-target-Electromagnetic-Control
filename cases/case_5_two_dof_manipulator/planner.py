import numpy as np
# Run from the repository root: python -m cases.case_5_two_dof_manipulator.planner
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
import heapq
import cvxpy as cp
from scipy.special import comb
import json
import math
import os

# --- CONFIGURATION ---
# Coordinate ranges (in radians)
X_MIN = -5/6 * np.pi  # x axis minimum (radians)
X_MAX = -1/6 * np.pi  # x axis maximum (radians)
Y_MIN = 1/6 * np.pi    # y axis minimum (radians)
Y_MAX = 5/6 * np.pi      # y axis maximum (radians)

# Grid resolution: number of grid cells per radian
# Higher values = finer grid, more precise but slower
GRID_RESOLUTION = 50  # grid cells per radian

# Compute MAP_SIZE from coordinate ranges and resolution
MAP_SIZE = (
    int(80),  # width in grid cells
    int(80)   # height in grid cells
)

# Results JSON file path
RESULTS_JSON_PATH = os.path.join("Results", "results_case5_0203.json")

# Define all waypoints: first is start, last is end
WAYPOINTS = [
    (-5/6 * math.pi, 1/2 * math.pi),      # Start position (waypoint 0)
    (-13/24 * math.pi, 1/2 * math.pi),       # Intermediate waypoint 1
    (-1/2 * math.pi, 3/4 * math.pi),       # Intermediate waypoint 1
    (-1/4 * math.pi, 1/2 * math.pi),       # Intermediate waypoint 1
    (-1/2 * math.pi, 5/12 * math.pi),       # Intermediate waypoint 1
    (-2/3 * math.pi, 5/12 * math.pi)      # Intermediate waypoint 1
]

# For backward compatibility
START_POS = WAYPOINTS[0]
END_POS = WAYPOINTS[-1]

V_MAX = 5 # Max velocity (m/s)
A_MAX = 5 # Max acceleration (m/s^2)
N_DEGREE = 5   # Degree of Bezier curve

# Visualization switches
SHOW_ASTAR_PATH = False   # Set to True to display A* path
SHOW_CORRIDOR = False      # Set to True to display corridor boxes
SHOW_CONTROL_POINTS = False  # Set to True to display Bezier curve control points

# Corridor generation sampling
CORRIDOR_SAMPLE_STEP = 5 # Sample path every CORRIDOR_SAMPLE_STEP points (1 = use all points)

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
    p2_x = (L1 + L2 * np.cos(alpha2)) * np.sin(beta1)
    p2_y = -(L1 + L2 * np.cos(alpha2)) * np.cos(beta1)
    p2_z = -L2 * np.sin(alpha2)
    p2_x += x_base
    p2_y += y_base
    p2_z += z_base

    return p2_x, p2_y, p2_z

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

    # Ensure all waypoints are clear
    for waypoint in WAYPOINTS:
        wp_grid_x, wp_grid_y = rad_to_grid_idx(waypoint[0], waypoint[1])
        x_start, y_start = max(0, wp_grid_x-1), max(0, wp_grid_y-1)
        x_end, y_end = min(MAP_SIZE[0], wp_grid_x+2), min(MAP_SIZE[1], wp_grid_y+2)
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

def find_multi_waypoint_path(grid, waypoints_rad):
    """
    Finds paths between consecutive waypoints using A*.

    Parameters:
    - grid: The obstacle grid
    - waypoints_rad: List of waypoints in radians [(x, y), ...]

    Returns:
    - path_segments: List of path segments, one for each waypoint-to-waypoint connection
    """
    if len(waypoints_rad) < 2:
        print("Error: Need at least 2 waypoints (start and end)")
        return None

    path_segments = []

    for i in range(len(waypoints_rad) - 1):
        start_pos = waypoints_rad[i]
        end_pos = waypoints_rad[i + 1]

        # Convert to grid indices
        start_grid = rad_to_grid_idx(start_pos[0], start_pos[1])
        end_grid = rad_to_grid_idx(end_pos[0], end_pos[1])

        print(f"Finding path from waypoint {i} to waypoint {i+1}...")
        path_segment = find_path_astar(grid, start_grid, end_grid)

        if path_segment is None:
            print(f"Failed to find path between waypoint {i} and waypoint {i+1}")
            return None

        path_segments.append(path_segment)
        print(f"  Path segment {i}: {len(path_segment)} points")

    print(f"All path segments found through {len(waypoints_rad)} waypoints")
    return path_segments

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

# --- Waypoint Planning and Verification ---
def verify_waypoint_passage(waypoints, corridor_segments, Cx, Cy):
    """
    Verifies that the trajectory passes through all waypoints.

    Parameters:
    - waypoints: List of waypoint positions
    - corridor_segments: List of corridor lists per segment
    - Cx, Cy: Control points

    Returns:
    - verification_results: List of (waypoint_idx, actual_pos, error) tuples
    """
    n = N_DEGREE

    # Flatten corridors and find waypoint indices
    waypoint_segment_indices = [0]
    for corridor_list in corridor_segments:
        waypoint_segment_indices.append(waypoint_segment_indices[-1] + len(corridor_list))

    print("\n" + "="*60)
    print("WAYPOINT PASSAGE VERIFICATION")
    print("="*60)

    verification_results = []

    for wp_idx, waypoint in enumerate(waypoints):
        corridor_idx = waypoint_segment_indices[wp_idx]

        if wp_idx == 0:
            # Start waypoint
            actual_pos = (Cx[0, corridor_idx], Cy[0, corridor_idx])
        elif wp_idx == len(waypoints) - 1:
            # End waypoint
            actual_pos = (Cx[n, -1], Cy[n, -1])
        else:
            # Intermediate waypoint (end of previous segment)
            actual_pos = (Cx[n, corridor_idx], Cy[n, corridor_idx])

        # Calculate error
        error = np.linalg.norm(np.array(waypoint) - np.array(actual_pos))
        verification_results.append((wp_idx, actual_pos, error))

        status = "✓ PASS" if error < 1e-4 else "✗ FAIL"
        print(f"Waypoint {wp_idx}: {status}")
        print(f"  Target:  ({waypoint[0]:.6f}, {waypoint[1]:.6f})")
        print(f"  Actual:  ({actual_pos[0]:.6f}, {actual_pos[1]:.6f})")
        print(f"  Error:   {error:.2e}")

    print("="*60 + "\n")

    return verification_results

def plan_waypoint_trajectory(grid, waypoints):
    """
    Plans trajectory through multiple waypoints.

    Parameters:
    - grid: The obstacle grid
    - waypoints: List of waypoints in radians

    Returns:
    - all_corridors: List of corridor segments
    - Cx, Cy: Optimized control points
    - time_segments: Time durations
    - path_segments: A* path segments
    """
    # Step 1: Find A* paths between waypoints
    path_segments = find_multi_waypoint_path(grid, waypoints)
    if path_segments is None:
        return None, None, None, None, None

    # Step 2: Generate corridors for each path segment
    print("\nGenerating corridors for each segment...")
    all_corridors = []
    for i, path in enumerate(path_segments):
        print(f"  Generating corridor for segment {i}...")
        corridor = generate_corridor(grid, path)
        all_corridors.append(corridor)
        print(f"    Segment {i}: {len(corridor)} corridor boxes")

    # Step 3: Optimize trajectory through all waypoints
    print("\nOptimizing trajectory through waypoints...")
    Cx, Cy, time_segments = optimize_trajectory_waypoints(waypoints, all_corridors)

    return all_corridors, Cx, Cy, time_segments, path_segments

def optimize_trajectory_waypoints(waypoints, corridor_segments):
    """
    Optimizes trajectory through multiple waypoints.

    Parameters:
    - waypoints: List of waypoint positions [(x,y), ...]
    - corridor_segments: List of corridor lists, one per waypoint segment

    Returns:
    - Cx, Cy: Control points
    - time_segments: Time allocations
    """
    n = N_DEGREE

    # Flatten all corridors into one list and track segment boundaries
    all_corridors = []
    segment_corridor_counts = []
    waypoint_segment_indices = [0]  # First waypoint is at start of segment 0

    for corridor_list in corridor_segments:
        all_corridors.extend(corridor_list)
        segment_corridor_counts.append(len(corridor_list))
        # Intermediate/end waypoints are at the END of each segment
        waypoint_segment_indices.append(len(all_corridors) - 1)

    m = len(all_corridors)

    if m == 0:
        print("Error: No corridors generated")
        return None, None, []

    print(f"  Total corridor boxes: {m}")
    print(f"  Waypoint segment boundaries: {waypoint_segment_indices}")

    # Control points for x and y
    Cx = cp.Variable((n + 1, m))
    Cy = cp.Variable((n + 1, m))

    # --- Objective Function: Minimize Jerk ---
    cost = 0
    time_segments = []

    # Improved time allocation
    current_pos = np.array(waypoints[0])
    wp_target_map = {}
    for k in range(1, len(waypoints)):
        idx = waypoint_segment_indices[k]
        wp_target_map[idx] = np.array(waypoints[k])

    for i in range(m):
        if i in wp_target_map:
            target_pos = wp_target_map[i]
        else:
            # Find intersection with next box
            curr_box = all_corridors[i]
            next_box = all_corridors[i+1]

            x_min = max(curr_box[0], next_box[0])
            y_min = max(curr_box[1], next_box[1])
            x_max = min(curr_box[2], next_box[2])
            y_max = min(curr_box[3], next_box[3])

            # Use center of intersection as target
            target_pos = np.array([(x_min + x_max)/2, (y_min + y_max)/2])

        dist = np.linalg.norm(target_pos - current_pos)

        # Allocate time based on distance and max velocity
        # Factor 11.0 provides significant slack for dynamics
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

    # 1. Waypoint Constraints - enforce trajectory passes through each waypoint
    for wp_idx, waypoint in enumerate(waypoints):
        corridor_idx = waypoint_segment_indices[wp_idx]

        if wp_idx == 0:
            # Start waypoint: position, zero velocity, zero acceleration
            constraints += [Cx[0, corridor_idx] == waypoint[0],
                          Cy[0, corridor_idx] == waypoint[1]]
            constraints += [Cx[1, corridor_idx] - Cx[0, corridor_idx] == 0,
                          Cy[1, corridor_idx] - Cy[0, corridor_idx] == 0]
            constraints += [Cx[2, corridor_idx] - 2*Cx[1, corridor_idx] + Cx[0, corridor_idx] == 0,
                          Cy[2, corridor_idx] - 2*Cy[1, corridor_idx] + Cy[0, corridor_idx] == 0]

        elif wp_idx == len(waypoints) - 1:
            # End waypoint: position, zero velocity, zero acceleration
            constraints += [Cx[n, m-1] == waypoint[0],
                          Cy[n, m-1] == waypoint[1]]
            constraints += [Cx[n, m-1] - Cx[n-1, m-1] == 0,
                          Cy[n, m-1] - Cy[n-1, m-1] == 0]
            constraints += [Cx[n, m-1] - 2*Cx[n-1, m-1] + Cx[n-2, m-1] == 0,
                          Cy[n, m-1] - 2*Cy[n-1, m-1] + Cy[n-2, m-1] == 0]

        else:
            # Intermediate waypoint: position constraint at end of previous segment
            # corridor_idx points to the end of the segment leading to this waypoint
            constraints += [Cx[n, corridor_idx] == waypoint[0],
                          Cy[n, corridor_idx] == waypoint[1]]

    # 2. Continuity Constraints (at all segment junctions)
    for i in range(m - 1):
        T1, T2 = time_segments[i], time_segments[i+1]
        # Position continuity
        constraints += [Cx[n, i] == Cx[0, i+1], Cy[n, i] == Cy[0, i+1]]
        # Velocity continuity
        constraints += [(1/T1)*(Cx[n, i] - Cx[n-1, i]) == (1/T2)*(Cx[1, i+1] - Cx[0, i+1])]
        constraints += [(1/T1)*(Cy[n, i] - Cy[n-1, i]) == (1/T2)*(Cy[1, i+1] - Cy[0, i+1])]
        # Acceleration continuity
        constraints += [(1/T1**2)*(Cx[n, i] - 2*Cx[n-1, i] + Cx[n-2, i]) ==
                       (1/T2**2)*(Cx[2, i+1] - 2*Cx[1, i+1] + Cx[0, i+1])]
        constraints += [(1/T1**2)*(Cy[n, i] - 2*Cy[n-1, i] + Cy[n-2, i]) ==
                       (1/T2**2)*(Cy[2, i+1] - 2*Cy[1, i+1] + Cy[0, i+1])]

    # 3. Safety & Dynamic Constraints
    for i in range(m):
        box = all_corridors[i]
        T = time_segments[i]

        # Safety (Convex Hull Property)
        constraints += [Cx[:, i] >= box[0], Cx[:, i] <= box[2]]
        constraints += [Cy[:, i] >= box[1], Cy[:, i] <= box[3]]

        # Dynamics (Hodograph Property)
        Vx = (n / T) * (Cx[1:, i] - Cx[:-1, i])
        Vy = (n / T) * (Cy[1:, i] - Cy[:-1, i])

        for j in range(n):
            constraints += [cp.square(Vx[j]) + cp.square(Vy[j]) <= V_MAX**2]

        Ax = (n - 1) / T * (Vx[1:] - Vx[:-1])
        Ay = (n - 1) / T * (Vy[1:] - Vy[:-1])

        for j in range(n - 1):
            constraints += [cp.square(Ax[j]) + cp.square(Ay[j]) <= A_MAX**2]

    # --- Solve QP ---
    prob = cp.Problem(cp.Minimize(cost), constraints)
    try:
        prob.solve(solver=cp.ECOS, verbose=False)
    except cp.SolverError:
        prob.solve(verbose=False)

    if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        print(f"Optimization failed with status: {prob.status}")
        return None, None, time_segments

    if prob.status == cp.OPTIMAL_INACCURATE:
        print("Warning: Optimization was inaccurate.")

    print(f"  Optimization successful!")
    return Cx.value, Cy.value, time_segments

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
    for i in range(m):
        box = corridor[i]
        center = np.array([(box[0]+box[2])/2, (box[1]+box[3])/2])
        if i < m - 1:
            next_box = corridor[i+1]
            next_center = np.array([(next_box[0]+next_box[2])/2, (next_box[1]+next_box[3])/2])
            dist = np.linalg.norm(center - next_center)
        else: # Last segment
            dist = np.linalg.norm(center - end_pos)

        # Give more time to relax constraints, e.g., by a factor of 5.0
        time_segments.append(max(dist / V_MAX * 11.0, 0.5)) # Add a minimum time

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


def plot_results(grid, path, corridor, Cx, Cy, show_astar_path=True, show_corridor=True, show_control_points=True, time_segments=None):
    # Figure 1: 2D beta1-alpha2 space
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

    # Plot all waypoints
    for i, wp in enumerate(WAYPOINTS):
        if i == 0:
            # Start waypoint (green)
            ax.plot(wp[0], wp[1], 'go', markersize=12, label='Start',
                   markeredgecolor='black', markeredgewidth=2, zorder=10)
        elif i == len(WAYPOINTS) - 1:
            # End waypoint (red)
            ax.plot(wp[0], wp[1], 'ro', markersize=12, label='End',
                   markeredgecolor='black', markeredgewidth=2, zorder=10)
        else:
            # Intermediate waypoints (blue)
            ax.plot(wp[0], wp[1], 'bs', markersize=10,
                   label='Waypoint' if i == 1 else '',
                   markeredgecolor='black', markeredgewidth=1.5, zorder=10)

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
            custom_traj_handle = Line2D([0], [0], color=cmap(0.8), linewidth=2.5, label='Trajectory')

        for i in range(Cx.shape[1]):
            # Control points are already in radians
            px = bernstein_basis.T @ Cx[:, i]
            py = bernstein_basis.T @ Cy[:, i]

            if use_time_coloring:
                segment_duration = time_segments[i]
                seg_times = cumulative_time + t_vec * segment_duration
                for j in range(len(px) - 1):
                    trajectory_segments.append(
                        np.array([
                            [px[j], py[j]],
                            [px[j + 1], py[j + 1]]
                        ])
                    )
                    segment_time_values.append(float((seg_times[j] + seg_times[j + 1]) * 0.5))
                cumulative_time += segment_duration
            else:
                ax.plot(px, py, 'r-', linewidth=2.5, label='Trajectory' if i == 0 else '')

            if show_control_points:
                ax.plot(Cx[:, i], Cy[:, i], 'c--o', markersize=4, alpha=0.7,
                       label='Control Points' if i == 0 else '')

        if use_time_coloring and trajectory_segments:
            lc = LineCollection(trajectory_segments, cmap=cmap, norm=norm, linewidth=2.5)
            lc.set_array(np.array(segment_time_values))
            lc.set_label('Trajectory')
            ax.add_collection(lc)
            cbar = fig.colorbar(lc, ax=ax, pad=0.02, fraction=0.05)
            cbar.set_label('Time (s)', fontsize=20)
            cbar.ax.tick_params(labelsize=18)
        else:
            custom_traj_handle = None

    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel(r'$\beta_1$ (rad)', fontsize=20)
    ax.set_ylabel(r'$\alpha_2$ (rad)', fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=18)

    legend_handles, legend_labels = ax.get_legend_handles_labels()
    if custom_traj_handle is not None and 'Trajectory' not in legend_labels:
        legend_handles.append(custom_traj_handle)
        legend_labels.append('Trajectory')
    if legend_handles:
        ax.legend(legend_handles, legend_labels, fontsize=15, loc='lower right')
    # ax.set_title("Quadrotor Motion Planning")
    ticksx = [-5/6 * math.pi, -2/3 * math.pi, -1/2 * math.pi, -1/3 * math.pi, -1/6 * math.pi]
    labelsx = ['-5π/6', '-2π/3', '-π/2', '-π/3', '-π/6']

    ticksy = [1/6 * math.pi, 1/3 * math.pi, 1/2 * math.pi, 2/3 * math.pi, 5/6 * math.pi]
    labelsy = ['π/6', 'π/3', 'π/2', '2π/3', '5π/6']

    ax.set_xticks(ticksx, labelsx)
    ax.set_yticks(ticksy, labelsy)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Figure 2: 3D end-effector space
    fig2 = plt.figure(figsize=(8, 6))
    ax2 = fig2.add_subplot(111, projection='3d')

    # Conversion factor: meters to centimeters
    M_TO_CM = 100.0

    # Set view angle
    ax2.view_init(elev=20, azim=250)

    # Generate workspace boundary
    beta1_range = (X_MIN, X_MAX)
    alpha2_range = (Y_MIN, Y_MAX)

    # Generate boundary curves
    samples = 200
    alpha2_edge = np.linspace(alpha2_range[0], alpha2_range[1], samples)
    beta1_edge = np.linspace(beta1_range[0], beta1_range[1], samples)

    boundary_params = [
        (np.full(samples, beta1_range[0]), alpha2_edge),  # beta1 = beta1_min
        (np.full(samples, beta1_range[1]), alpha2_edge),  # beta1 = beta1_max
        (beta1_edge, np.full(samples, alpha2_range[0])),  # alpha2 = alpha2_min
        (beta1_edge, np.full(samples, alpha2_range[1])),  # alpha2 = alpha2_max
    ]

    for beta1_vals, alpha2_vals in boundary_params:
        x, y, z = arm_end_positions(beta1_vals, alpha2_vals)
        ax2.plot(x * M_TO_CM, y * M_TO_CM, z * M_TO_CM, color="black", linewidth=2.0)

    # Plot obstacles as black regions
    obstacles = load_obstacles_from_json(RESULTS_JSON_PATH)
    obstacle_sample_density = 10
    for box in obstacles:
        if len(box) >= 4:
            # Obstacle box format: [x_min, x_max, y_min, y_max] in radians
            x_min, x_max, y_min, y_max = box[0], box[1], box[2], box[3]
            beta1_obs = np.linspace(x_min, x_max, obstacle_sample_density)
            alpha2_obs = np.linspace(y_min, y_max, obstacle_sample_density)
            beta1_mesh, alpha2_mesh = np.meshgrid(beta1_obs, alpha2_obs, indexing='ij')
            x_obs, y_obs, z_obs = arm_end_positions(beta1_mesh.ravel(), alpha2_mesh.ravel())
            ax2.scatter(x_obs * M_TO_CM, y_obs * M_TO_CM, z_obs * M_TO_CM,
                       c='black', s=5, alpha=0.3)

    # Initialize trajectory lists
    trajectory_x_all = []
    trajectory_y_all = []
    trajectory_z_all = []
    trajectory_time_all = []

    # Convert trajectory from beta1-alpha2 to 3D end-effector space
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

            # Convert to 3D end-effector positions using arm_end_positions
            x_ee, y_ee, z_ee = arm_end_positions(px, py)

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
    x_start, y_start, z_start = arm_end_positions(START_POS[0], START_POS[1])
    x_end, y_end, z_end = arm_end_positions(END_POS[0], END_POS[1])

    # Plot start point with larger size, edge color, and higher zorder
    ax2.scatter([x_start * M_TO_CM], [y_start * M_TO_CM], [z_start * M_TO_CM],
               c='green', s=300, marker='o', linewidths=2.5,
               label='Start', zorder=100, alpha=0.9)
    # Plot end point with larger size, edge color, and higher zorder
    ax2.scatter([x_end * M_TO_CM], [y_end * M_TO_CM], [z_end * M_TO_CM],
               c='red', s=300, marker='o', linewidths=2.5,
               label='End', zorder=100, alpha=0.9)

    ax2.set_xlabel("X (cm)", fontsize=20, labelpad=10)
    ax2.set_ylabel("Y (cm)", fontsize=20, labelpad=10)
    ax2.set_zlabel("Z (cm)", fontsize=20, labelpad=10)
    ax2.tick_params(axis='both', which='major', labelsize=18)

    # Set equal aspect ratio
    all_x_coords = []
    all_y_coords = []
    all_z_coords = []

    # Collect boundary points
    for beta1_vals, alpha2_vals in boundary_params:
        x, y, z = arm_end_positions(beta1_vals, alpha2_vals)
        all_x_coords.extend(x * M_TO_CM)
        all_y_coords.extend(y * M_TO_CM)
        all_z_coords.extend(z * M_TO_CM)

    # Collect trajectory points
    if Cx is not None and Cy is not None and trajectory_x_all:
        all_x_coords.extend(trajectory_x_all)
        all_y_coords.extend(trajectory_y_all)
        all_z_coords.extend(trajectory_z_all)

    # Add start/end points
    all_x_coords.extend([x_start * M_TO_CM, x_end * M_TO_CM])
    all_y_coords.extend([y_start * M_TO_CM, y_end * M_TO_CM])
    all_z_coords.extend([z_start * M_TO_CM, z_end * M_TO_CM])

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

# --- SAVE TRAJECTORY RESULTS ---
def save_trajectory_results(Cx, Cy, time_segments, corridor, start_pos, end_pos,
                            filepath=None):
    """
    Saves trajectory optimization results to a JSON file.

    Parameters:
    -----------
    Cx, Cy : numpy.ndarray
        Control points for Bezier curves (shape: (n+1, m))
    time_segments : list
        Time durations for each segment
    corridor : list
        List of corridor boxes [x_min, y_min, x_max, y_max]
    start_pos : tuple
        Start position (x, y)
    end_pos : tuple
        End position (x, y)
    filepath : str, optional
        Path to save the JSON file. If None, uses default path in Results folder.
    """
    if filepath is None:
        filepath = os.path.join("Results", "trajectory_case5_0203.json")

    # Ensure Results directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # Convert numpy arrays to lists for JSON serialization
    trajectory_data = {
        "control_points": {
            "Cx": Cx.tolist() if hasattr(Cx, 'tolist') else Cx,
            "Cy": Cy.tolist() if hasattr(Cy, 'tolist') else Cy
        },
        "time_segments": time_segments,
        "corridor": [box.tolist() if hasattr(box, 'tolist') else list(box) for box in corridor],
        "start_pos": list(start_pos),
        "end_pos": list(end_pos),
        "config": {
            "N_DEGREE": N_DEGREE,
            "V_MAX": V_MAX,
            "A_MAX": A_MAX
        },
        "num_segments": len(time_segments)
    }

    # Save to JSON file
    with open(filepath, 'w') as f:
        json.dump(trajectory_data, f, indent=2)

    print(f"\nTrajectory results saved to: {filepath}")
    return filepath

# --- MAIN EXECUTION ---
if __name__ == '__main__':
    grid = generate_map()

    # Find path through all waypoints
    print(f"\n{'='*60}")
    print(f"MULTI-WAYPOINT PATH PLANNING")
    print(f"{'='*60}")
    print(f"Number of waypoints: {len(WAYPOINTS)}")
    for i, wp in enumerate(WAYPOINTS):
        print(f"  Waypoint {i}: ({wp[0]:.4f}, {wp[1]:.4f})")
    print(f"{'='*60}\n")

    # Plan trajectory through waypoints
    corridor_segments, Cx, Cy, time_segments, path_segments = plan_waypoint_trajectory(grid, WAYPOINTS)

    if Cx is not None and Cy is not None:
        # Flatten corridors for visualization and saving
        all_corridors = []
        for corridor_list in corridor_segments:
            all_corridors.extend(corridor_list)

        # Concatenate path segments for visualization
        complete_path = None
        for i, path_seg in enumerate(path_segments):
            if complete_path is None:
                complete_path = path_seg
            else:
                complete_path = np.vstack([complete_path, path_seg[1:]])

        # Verify waypoint passage
        verify_waypoint_passage(WAYPOINTS, corridor_segments, Cx, Cy)

        # Print segment information (all in radians)
        print_segment_info(START_POS, END_POS, all_corridors, time_segments, Cx, Cy)

        # Save trajectory results to JSON file
        save_trajectory_results(Cx, Cy, time_segments, all_corridors, START_POS, END_POS)

        # Plot with visualization switches
        plot_results(grid, complete_path, all_corridors, Cx, Cy,
                    show_astar_path=SHOW_ASTAR_PATH,
                    show_corridor=SHOW_CORRIDOR,
                    show_control_points=SHOW_CONTROL_POINTS,
                    time_segments=time_segments)
    else:
        print("Trajectory optimization failed.")
