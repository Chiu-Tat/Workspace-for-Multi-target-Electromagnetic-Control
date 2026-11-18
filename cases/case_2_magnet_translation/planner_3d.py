import numpy as np
# Run from the repository root: python -m cases.case_2_magnet_translation.planner_3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from mpl_toolkits.mplot3d import Axes3D
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
X_MIN = -0.0317  # x axis minimum (radians)
X_MAX = 0.0317  # x axis maximum (radians)
Y_MIN = -0.0317      # y axis minimum (radians)
Y_MAX = 0.0317       # y axis maximum (radians)
Z_MIN = -0.0317     # z axis minimum (radians)
Z_MAX = 0.0317       # z axis maximum (radians)

# Grid resolution: number of grid cells per radian
# Higher values = finer grid, more precise but slower
GRID_RESOLUTION = 50  # grid cells per radian

# Compute MAP_SIZE from coordinate ranges and resolution
MAP_SIZE = (
    int(80),  # width in grid cells
    int(80),  # height in grid cells
    int(80)   # depth in grid cells
)

# Results JSON file path
RESULTS_JSON_PATH = os.path.join("Results", "results_case2_3D_0101.json")

# Define all waypoints: first is start, last is end
WAYPOINTS = [
    (-0.0317, 0.0317, -0.0317),     # Start position (waypoint 0)
    (0.0317, 0.0317, -0.0317),            # Intermediate waypoint 1
    # (-0.0317, 0.0317, -0.0317)        # Intermediate waypoint 2
    # (-0.0317, 0.00, -0.0317)        # End position (last waypoint)
]

# For backward compatibility
START_POS = WAYPOINTS[0]
END_POS = WAYPOINTS[-1]

V_MAX = 0.1  # Max velocity
A_MAX = 0.1   # Max acceleration
N_DEGREE = 5   # Degree of Bezier curve

# Visualization switches
SHOW_ASTAR_PATH = False   # Set to True to display A* path
SHOW_CORRIDOR = False      # Set to True to display corridor boxes
SHOW_CONTROL_POINTS = False  # Set to True to display Bezier curve control points
SHOW_OBSTACLES = False     # Set to True to display obstacles from JSON file

# Corridor generation sampling
CORRIDOR_SAMPLE_STEP = 1 # Sample path every CORRIDOR_SAMPLE_STEP points (1 = use all points)

# --- STEP 0: MAP GENERATION ---
def load_obstacles_from_json(json_path):
    """
    Loads obstacles from JSON file.
    Returns list of boxes as [x_min, x_max, y_min, y_max, z_min, z_max] in radians.
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

def rad_to_grid_idx(x_rad, y_rad, z_rad):
    """
    Converts radian coordinates to grid indices for array access.

    Parameters:
    - x_rad, y_rad, z_rad: coordinates in radians

    Returns:
    - grid_x, grid_y, grid_z: grid indices (clamped to valid range)
    """
    grid_x = int((x_rad - X_MIN) / (X_MAX - X_MIN) * MAP_SIZE[0])
    grid_y = int((y_rad - Y_MIN) / (Y_MAX - Y_MIN) * MAP_SIZE[1])
    grid_z = int((z_rad - Z_MIN) / (Z_MAX - Z_MIN) * MAP_SIZE[2])
    return (max(0, min(grid_x, MAP_SIZE[0] - 1)),
            max(0, min(grid_y, MAP_SIZE[1] - 1)),
            max(0, min(grid_z, MAP_SIZE[2] - 1)))

def grid_idx_to_rad(grid_x, grid_y, grid_z):
    """
    Converts grid indices back to radian coordinates.

    Parameters:
    - grid_x, grid_y, grid_z: grid indices

    Returns:
    - x_rad, y_rad, z_rad: coordinates in radians
    """
    x_rad = X_MIN + (grid_x / MAP_SIZE[0]) * (X_MAX - X_MIN)
    y_rad = Y_MIN + (grid_y / MAP_SIZE[1]) * (Y_MAX - Y_MIN)
    z_rad = Z_MIN + (grid_z / MAP_SIZE[2]) * (Z_MAX - Z_MIN)
    return x_rad, y_rad, z_rad

def generate_map():
    """Generates a 3D grid with obstacles loaded from JSON file."""
    grid = np.zeros(MAP_SIZE, dtype=int)

    # Load obstacles from JSON file (already in radians)
    obstacles = load_obstacles_from_json(RESULTS_JSON_PATH)

    # Place obstacles in grid
    for box in obstacles:
        if len(box) >= 6:
            x_min_rad, x_max_rad, y_min_rad, y_max_rad, z_min_rad, z_max_rad = box[0], box[1], box[2], box[3], box[4], box[5]
            # Convert to grid indices
            grid_x_min = int((x_min_rad - X_MIN) / (X_MAX - X_MIN) * MAP_SIZE[0])
            grid_x_max = int((x_max_rad - X_MIN) / (X_MAX - X_MIN) * MAP_SIZE[0])
            grid_y_min = int((y_min_rad - Y_MIN) / (Y_MAX - Y_MIN) * MAP_SIZE[1])
            grid_y_max = int((y_max_rad - Y_MIN) / (Y_MAX - Y_MIN) * MAP_SIZE[1])
            grid_z_min = int((z_min_rad - Z_MIN) / (Z_MAX - Z_MIN) * MAP_SIZE[2])
            grid_z_max = int((z_max_rad - Z_MIN) / (Z_MAX - Z_MIN) * MAP_SIZE[2])

            # Clamp and place obstacles
            grid_x_min = max(0, min(grid_x_min, MAP_SIZE[0] - 1))
            grid_x_max = max(0, min(grid_x_max, MAP_SIZE[0]))
            grid_y_min = max(0, min(grid_y_min, MAP_SIZE[1] - 1))
            grid_y_max = max(0, min(grid_y_max, MAP_SIZE[1]))
            grid_z_min = max(0, min(grid_z_min, MAP_SIZE[2] - 1))
            grid_z_max = max(0, min(grid_z_max, MAP_SIZE[2]))

            if grid_x_max > grid_x_min and grid_y_max > grid_y_min and grid_z_max > grid_z_min:
                grid[grid_x_min:grid_x_max, grid_y_min:grid_y_max, grid_z_min:grid_z_max] = 1

    # Ensure all waypoints are clear
    for waypoint in WAYPOINTS:
        wp_grid_x, wp_grid_y, wp_grid_z = rad_to_grid_idx(waypoint[0], waypoint[1], waypoint[2])
        x_start, y_start, z_start = max(0, wp_grid_x-1), max(0, wp_grid_y-1), max(0, wp_grid_z-1)
        x_end, y_end, z_end = min(MAP_SIZE[0], wp_grid_x+2), min(MAP_SIZE[1], wp_grid_y+2), min(MAP_SIZE[2], wp_grid_z+2)
        grid[x_start:x_end, y_start:y_end, z_start:z_end] = 0

    return grid, obstacles

# --- STEP 1: A* PATH FINDING ---
def find_path_astar(grid, start, end):
    """Custom A* pathfinding implementation for 3D."""

    def heuristic(a, b):
        return np.sqrt((b[0] - a[0])**2 + (b[1] - a[1])**2 + (b[2] - a[2])**2)

    open_list = []
    # (f_score, g_score, position)
    heapq.heappush(open_list, (0, 0, start))

    came_from = {}
    g_score = { (r,c,z): float('inf') for r in range(grid.shape[0]) for c in range(grid.shape[1]) for z in range(grid.shape[2]) }
    g_score[start] = 0

    f_score = { (r,c,z): float('inf') for r in range(grid.shape[0]) for c in range(grid.shape[1]) for z in range(grid.shape[2]) }
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

        # 26-connected neighbors in 3D (all adjacent cells)
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    neighbor = (current[0] + dx, current[1] + dy, current[2] + dz)

                    if not (0 <= neighbor[0] < grid.shape[0] and 0 <= neighbor[1] < grid.shape[1] and 0 <= neighbor[2] < grid.shape[2]):
                        continue

                    if grid[neighbor[0], neighbor[1], neighbor[2]] == 1:
                        continue

                    # distance between current and neighbor is sqrt(dx^2 + dy^2 + dz^2)
                    move_cost = np.sqrt(dx**2 + dy**2 + dz**2)
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
    - waypoints_rad: List of waypoints in radians [(x, y, z), ...]

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
        start_grid = rad_to_grid_idx(start_pos[0], start_pos[1], start_pos[2])
        end_grid = rad_to_grid_idx(end_pos[0], end_pos[1], end_pos[2])

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
    Generates a corridor of overlapping axis-aligned 3D boxes.
    This method iteratively finds the largest box that can cover a segment of the path.
    Samples the path every CORRIDOR_SAMPLE_STEP points before generating corridors.
    Path is in grid indices, returns corridor boxes in radians [x_min, x_max, y_min, y_max, z_min, z_max].
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
        x_min, y_min, z_min = start_point
        x_max, y_max, z_max = min(grid.shape[0], start_point[0] + 1), min(grid.shape[1], start_point[1] + 1), min(grid.shape[2], start_point[2] + 1)

        end_path_idx = path_idx
        for i in range(path_idx + 1, len(sampled_path)):
            point = sampled_path[i]

            temp_x_min = min(x_min, point[0])
            temp_y_min = min(y_min, point[1])
            temp_z_min = min(z_min, point[2])
            temp_x_max = max(x_max, point[0] + 1)
            temp_y_max = max(y_max, point[1] + 1)
            temp_z_max = max(z_max, point[2] + 1)

            temp_x_min = max(0, temp_x_min)
            temp_y_min = max(0, temp_y_min)
            temp_z_min = max(0, temp_z_min)
            temp_x_max = min(grid.shape[0], temp_x_max)
            temp_y_max = min(grid.shape[1], temp_y_max)
            temp_z_max = min(grid.shape[2], temp_z_max)

            if np.any(grid[temp_x_min:temp_x_max, temp_y_min:temp_y_max, temp_z_min:temp_z_max] == 1):
                break

            x_min, y_min, z_min, x_max, y_max, z_max = temp_x_min, temp_y_min, temp_z_min, temp_x_max, temp_y_max, temp_z_max
            end_path_idx = i

        # 2. Inflate the bounding box as much as possible
        if x_max > x_min and y_max > y_min and z_max > z_min:
            # Expand in x direction
            while x_min > 0 and y_max > y_min and z_max > z_min and np.all(grid[x_min - 1, y_min:y_max, z_min:z_max] == 0):
                x_min -= 1
            while x_max < grid.shape[0] and y_max > y_min and z_max > z_min and np.all(grid[x_max, y_min:y_max, z_min:z_max] == 0):
                x_max += 1
            # Expand in y direction
            while y_min > 0 and x_max > x_min and z_max > z_min and np.all(grid[x_min:x_max, y_min - 1, z_min:z_max] == 0):
                y_min -= 1
            while y_max < grid.shape[1] and x_max > x_min and z_max > z_min and np.all(grid[x_min:x_max, y_max, z_min:z_max] == 0):
                y_max += 1
            # Expand in z direction
            while z_min > 0 and x_max > x_min and y_max > y_min and np.all(grid[x_min:x_max, y_min:y_max, z_min - 1] == 0):
                z_min -= 1
            while z_max < grid.shape[2] and x_max > x_min and y_max > y_min and np.all(grid[x_min:x_max, y_min:y_max, z_max] == 0):
                z_max += 1

        # Convert corridor box from grid indices to radians
        x_min_rad, y_min_rad, z_min_rad = grid_idx_to_rad(x_min, y_min, z_min)
        x_max_rad, y_max_rad, z_max_rad = grid_idx_to_rad(x_max, y_max, z_max)
        box = np.array([x_min_rad, x_max_rad, y_min_rad, y_max_rad, z_min_rad, z_max_rad])
        corridor.append(box)

        # 3. Move to the next path point
        if end_path_idx > path_idx and end_path_idx < len(sampled_path) - 1:
             path_idx = end_path_idx
        else:
             path_idx = end_path_idx + 1

    return corridor

# --- Print Segment Information ---
def verify_waypoint_passage(waypoints, corridor_segments, Cx, Cy, Cz):
    """
    Verifies that the trajectory passes through all waypoints.

    Parameters:
    - waypoints: List of waypoint positions
    - corridor_segments: List of corridor lists per segment
    - Cx, Cy, Cz: Control points

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
            actual_pos = (Cx[0, corridor_idx], Cy[0, corridor_idx], Cz[0, corridor_idx])
        elif wp_idx == len(waypoints) - 1:
            # End waypoint
            actual_pos = (Cx[n, -1], Cy[n, -1], Cz[n, -1])
        else:
            # Intermediate waypoint (end of previous segment)
            seg_end_idx = corridor_idx - 1
            actual_pos = (Cx[n, seg_end_idx], Cy[n, seg_end_idx], Cz[n, seg_end_idx])

        # Calculate error
        error = np.linalg.norm(np.array(waypoint) - np.array(actual_pos))
        verification_results.append((wp_idx, actual_pos, error))

        status = "✓ PASS" if error < 1e-4 else "✗ FAIL"
        print(f"Waypoint {wp_idx}: {status}")
        print(f"  Target:  ({waypoint[0]:.6f}, {waypoint[1]:.6f}, {waypoint[2]:.6f})")
        print(f"  Actual:  ({actual_pos[0]:.6f}, {actual_pos[1]:.6f}, {actual_pos[2]:.6f})")
        print(f"  Error:   {error:.2e}")

    print("="*60 + "\n")

    return verification_results

def print_segment_info(start_pos, end_pos, corridor, time_segments, Cx=None, Cy=None, Cz=None):
    """
    Prints the start point, end point, and time for each trajectory segment.

    Parameters:
    - start_pos: Starting position (x, y, z)
    - end_pos: Ending position (x, y, z)
    - corridor: List of corridor boxes
    - time_segments: List of time durations for each segment
    - Cx, Cy, Cz: Optional optimized control points (if available, uses them for endpoints)
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
            if Cx is not None and Cy is not None and Cz is not None:
                # Use optimized endpoint from previous segment
                start_point = (Cx[N_DEGREE, i-1], Cy[N_DEGREE, i-1], Cz[N_DEGREE, i-1])
            else:
                # Use corridor center as approximation
                prev_box = corridor[i-1]
                start_point = ((prev_box[0] + prev_box[1]) / 2,
                              (prev_box[2] + prev_box[3]) / 2,
                              (prev_box[4] + prev_box[5]) / 2)
            start_label = "JUNCTION"

        # Determine end point
        if i == m - 1:
            end_point = end_pos
            end_label = "END"
        else:
            # Use junction point
            if Cx is not None and Cy is not None and Cz is not None:
                # Use optimized endpoint
                end_point = (Cx[N_DEGREE, i], Cy[N_DEGREE, i], Cz[N_DEGREE, i])
            else:
                # Use corridor center as approximation
                box = corridor[i]
                end_point = ((box[0] + box[1]) / 2,
                            (box[2] + box[3]) / 2,
                            (box[4] + box[5]) / 2)
            end_label = "JUNCTION"

        # Get time duration
        T = time_segments[i]

        # Print information
        print(f"  Start Point ({start_label}): ({start_point[0]:.3f}, {start_point[1]:.3f}, {start_point[2]:.3f})")
        print(f"  End Point   ({end_label}):   ({end_point[0]:.3f}, {end_point[1]:.3f}, {end_point[2]:.3f})")
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



def plan_waypoint_trajectory(grid, waypoints):
    """
    Plans trajectory through multiple waypoints.

    Parameters:
    - grid: The obstacle grid
    - waypoints: List of waypoints in radians

    Returns:
    - all_corridors: List of corridor segments
    - Cx, Cy, Cz: Optimized control points
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
    Cx, Cy, Cz, time_segments = optimize_trajectory_waypoints(waypoints, all_corridors)

    return all_corridors, Cx, Cy, Cz, time_segments, path_segments

# --- STEP 3: TRAJECTORY OPTIMIZATION (GAO ET AL.) ---
def optimize_trajectory_waypoints(waypoints, corridor_segments):
    """
    Optimizes trajectory through multiple waypoints.

    Parameters:
    - waypoints: List of waypoint positions [(x,y,z), ...]
    - corridor_segments: List of corridor lists, one per waypoint segment

    Returns:
    - Cx, Cy, Cz: Control points
    - time_segments: Time allocations
    """
    n = N_DEGREE

    # Flatten all corridors into one list and track segment boundaries
    all_corridors = []
    segment_corridor_counts = []
    waypoint_segment_indices = [0]  # First waypoint is at segment 0

    for corridor_list in corridor_segments:
        all_corridors.extend(corridor_list)
        segment_corridor_counts.append(len(corridor_list))
        waypoint_segment_indices.append(len(all_corridors))

    m = len(all_corridors)

    if m == 0:
        print("Error: No corridors generated")
        return None, None, None, []

    print(f"  Total corridor boxes: {m}")
    print(f"  Waypoint segment boundaries: {waypoint_segment_indices}")

    # Control points for x, y, and z
    Cx = cp.Variable((n + 1, m))
    Cy = cp.Variable((n + 1, m))
    Cz = cp.Variable((n + 1, m))

    # --- Objective Function: Minimize Jerk ---
    cost = 0
    time_segments = []
    for i in range(m):
        box = all_corridors[i]
        center = np.array([(box[0]+box[1])/2, (box[2]+box[3])/2, (box[4]+box[5])/2])
        if i < m - 1:
            next_box = all_corridors[i+1]
            next_center = np.array([(next_box[0]+next_box[1])/2, (next_box[2]+next_box[3])/2, (next_box[4]+next_box[5])/2])
            dist = np.linalg.norm(center - next_center)
        else:
            dist = np.linalg.norm(center - waypoints[-1])

        time_segments.append(max(dist / V_MAX * 11.0, 0.5))

    for i in range(m):
        T = time_segments[i]
        # Velocity control points
        Vx = (n / T) * (Cx[1:, i] - Cx[:-1, i])
        Vy = (n / T) * (Cy[1:, i] - Cy[:-1, i])
        Vz = (n / T) * (Cz[1:, i] - Cz[:-1, i])
        # Acceleration control points
        Ax = (n - 1) / T * (Vx[1:] - Vx[:-1])
        Ay = (n - 1) / T * (Vy[1:] - Vy[:-1])
        Az = (n - 1) / T * (Vz[1:] - Vz[:-1])
        # Jerk control points
        Jx = (n - 2) / T * (Ax[1:] - Ax[:-1])
        Jy = (n - 2) / T * (Ay[1:] - Ay[:-1])
        Jz = (n - 2) / T * (Az[1:] - Az[:-1])
        cost += cp.sum_squares(Jx) + cp.sum_squares(Jy) + cp.sum_squares(Jz)

    # --- Constraints ---
    constraints = []

    # 1. Waypoint Constraints - enforce trajectory passes through each waypoint
    for wp_idx, waypoint in enumerate(waypoints):
        corridor_idx = waypoint_segment_indices[wp_idx]

        if wp_idx == 0:
            # Start waypoint: position, zero velocity, zero acceleration
            constraints += [Cx[0, corridor_idx] == waypoint[0],
                          Cy[0, corridor_idx] == waypoint[1],
                          Cz[0, corridor_idx] == waypoint[2]]
            constraints += [Cx[1, corridor_idx] - Cx[0, corridor_idx] == 0,
                          Cy[1, corridor_idx] - Cy[0, corridor_idx] == 0,
                          Cz[1, corridor_idx] - Cz[0, corridor_idx] == 0]
            constraints += [Cx[2, corridor_idx] - 2*Cx[1, corridor_idx] + Cx[0, corridor_idx] == 0,
                          Cy[2, corridor_idx] - 2*Cy[1, corridor_idx] + Cy[0, corridor_idx] == 0,
                          Cz[2, corridor_idx] - 2*Cz[1, corridor_idx] + Cz[0, corridor_idx] == 0]

        elif wp_idx == len(waypoints) - 1:
            # End waypoint: position, zero velocity, zero acceleration
            constraints += [Cx[n, m-1] == waypoint[0],
                          Cy[n, m-1] == waypoint[1],
                          Cz[n, m-1] == waypoint[2]]
            constraints += [Cx[n, m-1] - Cx[n-1, m-1] == 0,
                          Cy[n, m-1] - Cy[n-1, m-1] == 0,
                          Cz[n, m-1] - Cz[n-1, m-1] == 0]
            constraints += [Cx[n, m-1] - 2*Cx[n-1, m-1] + Cx[n-2, m-1] == 0,
                          Cy[n, m-1] - 2*Cy[n-1, m-1] + Cy[n-2, m-1] == 0,
                          Cz[n, m-1] - 2*Cz[n-1, m-1] + Cz[n-2, m-1] == 0]

        else:
            # Intermediate waypoint: position constraint
            # The segment ending at this waypoint
            seg_end_idx = corridor_idx - 1
            if seg_end_idx >= 0:
                constraints += [Cx[n, seg_end_idx] == waypoint[0],
                              Cy[n, seg_end_idx] == waypoint[1],
                              Cz[n, seg_end_idx] == waypoint[2]]

    # 2. Continuity Constraints (at all segment junctions)
    for i in range(m - 1):
        T1, T2 = time_segments[i], time_segments[i+1]
        # Position continuity
        constraints += [Cx[n, i] == Cx[0, i+1], Cy[n, i] == Cy[0, i+1], Cz[n, i] == Cz[0, i+1]]
        # Velocity continuity
        constraints += [(1/T1)*(Cx[n, i] - Cx[n-1, i]) == (1/T2)*(Cx[1, i+1] - Cx[0, i+1])]
        constraints += [(1/T1)*(Cy[n, i] - Cy[n-1, i]) == (1/T2)*(Cy[1, i+1] - Cy[0, i+1])]
        constraints += [(1/T1)*(Cz[n, i] - Cz[n-1, i]) == (1/T2)*(Cz[1, i+1] - Cz[0, i+1])]
        # Acceleration continuity
        constraints += [(1/T1**2)*(Cx[n, i] - 2*Cx[n-1, i] + Cx[n-2, i]) ==
                       (1/T2**2)*(Cx[2, i+1] - 2*Cx[1, i+1] + Cx[0, i+1])]
        constraints += [(1/T1**2)*(Cy[n, i] - 2*Cy[n-1, i] + Cy[n-2, i]) ==
                       (1/T2**2)*(Cy[2, i+1] - 2*Cy[1, i+1] + Cy[0, i+1])]
        constraints += [(1/T1**2)*(Cz[n, i] - 2*Cz[n-1, i] + Cz[n-2, i]) ==
                       (1/T2**2)*(Cz[2, i+1] - 2*Cz[1, i+1] + Cz[0, i+1])]

    # 3. Safety & Dynamic Constraints
    for i in range(m):
        box = all_corridors[i]
        T = time_segments[i]

        # Safety (Convex Hull Property)
        constraints += [Cx[:, i] >= box[0], Cx[:, i] <= box[1]]
        constraints += [Cy[:, i] >= box[2], Cy[:, i] <= box[3]]
        constraints += [Cz[:, i] >= box[4], Cz[:, i] <= box[5]]

        # Dynamics (Hodograph Property)
        Vx = (n / T) * (Cx[1:, i] - Cx[:-1, i])
        Vy = (n / T) * (Cy[1:, i] - Cy[:-1, i])
        Vz = (n / T) * (Cz[1:, i] - Cz[:-1, i])

        for j in range(n):
            constraints += [cp.square(Vx[j]) + cp.square(Vy[j]) + cp.square(Vz[j]) <= V_MAX**2]

        Ax = (n - 1) / T * (Vx[1:] - Vx[:-1])
        Ay = (n - 1) / T * (Vy[1:] - Vy[:-1])
        Az = (n - 1) / T * (Vz[1:] - Vz[:-1])

        for j in range(n - 1):
            constraints += [cp.square(Ax[j]) + cp.square(Ay[j]) + cp.square(Az[j]) <= A_MAX**2]

    # --- Solve QP ---
    prob = cp.Problem(cp.Minimize(cost), constraints)
    try:
        prob.solve(solver=cp.ECOS, verbose=False)
    except cp.SolverError:
        prob.solve(verbose=False)

    if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        print(f"Optimization failed with status: {prob.status}")
        return None, None, None, time_segments

    if prob.status == cp.OPTIMAL_INACCURATE:
        print("Warning: Optimization was inaccurate.")

    print(f"  Optimization successful!")
    return Cx.value, Cy.value, Cz.value, time_segments

def optimize_trajectory(start_pos, end_pos, corridor):
    if len(corridor) == 0:
        print("Error: Empty corridor, cannot optimize trajectory.")
        return None, None, None, []

    n = N_DEGREE
    m = len(corridor)

    # Control points for x, y, and z
    Cx = cp.Variable((n + 1, m))
    Cy = cp.Variable((n + 1, m))
    Cz = cp.Variable((n + 1, m))

    # --- Objective Function: Minimize Jerk ---
    cost = 0
    # Time allocation (heuristic)
    time_segments = []
    for i in range(m):
        box = corridor[i]
        center = np.array([(box[0]+box[1])/2, (box[2]+box[3])/2, (box[4]+box[5])/2])
        if i < m - 1:
            next_box = corridor[i+1]
            next_center = np.array([(next_box[0]+next_box[1])/2, (next_box[2]+next_box[3])/2, (next_box[4]+next_box[5])/2])
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
        Vz = (n / T) * (Cz[1:, i] - Cz[:-1, i])
        # Acceleration control points
        Ax = (n - 1) / T * (Vx[1:] - Vx[:-1])
        Ay = (n - 1) / T * (Vy[1:] - Vy[:-1])
        Az = (n - 1) / T * (Vz[1:] - Vz[:-1])
        # Jerk control points
        Jx = (n - 2) / T * (Ax[1:] - Ax[:-1])
        Jy = (n - 2) / T * (Ay[1:] - Ay[:-1])
        Jz = (n - 2) / T * (Az[1:] - Az[:-1])
        cost += cp.sum_squares(Jx) + cp.sum_squares(Jy) + cp.sum_squares(Jz)

    # --- Constraints ---
    constraints = []

    # 1. Waypoint Constraints (Start and End)
    # Start position, velocity, acceleration
    constraints += [Cx[0, 0] == start_pos[0], Cy[0, 0] == start_pos[1], Cz[0, 0] == start_pos[2]]
    constraints += [Cx[1, 0] - Cx[0, 0] == 0, Cy[1, 0] - Cy[0, 0] == 0, Cz[1, 0] - Cz[0, 0] == 0] # v_start = 0
    constraints += [Cx[2, 0] - 2*Cx[1, 0] + Cx[0, 0] == 0,
                    Cy[2, 0] - 2*Cy[1, 0] + Cy[0, 0] == 0,
                    Cz[2, 0] - 2*Cz[1, 0] + Cz[0, 0] == 0] # a_start = 0

    # End position, velocity, acceleration
    constraints += [Cx[n, m-1] == end_pos[0], Cy[n, m-1] == end_pos[1], Cz[n, m-1] == end_pos[2]]
    constraints += [Cx[n, m-1] - Cx[n-1, m-1] == 0, Cy[n, m-1] - Cy[n-1, m-1] == 0, Cz[n, m-1] - Cz[n-1, m-1] == 0] # v_end = 0
    constraints += [Cx[n, m-1] - 2*Cx[n-1, m-1] + Cx[n-2, m-1] == 0,
                    Cy[n, m-1] - 2*Cy[n-1, m-1] + Cy[n-2, m-1] == 0,
                    Cz[n, m-1] - 2*Cz[n-1, m-1] + Cz[n-2, m-1] == 0] # a_end = 0

    # 2. Continuity Constraints (at segment junctions)
    for i in range(m - 1):
        T1, T2 = time_segments[i], time_segments[i+1]
        # Position
        constraints += [Cx[n, i] == Cx[0, i+1], Cy[n, i] == Cy[0, i+1], Cz[n, i] == Cz[0, i+1]]
        # Velocity
        constraints += [(1/T1)*(Cx[n, i] - Cx[n-1, i]) == (1/T2)*(Cx[1, i+1] - Cx[0, i+1])]
        constraints += [(1/T1)*(Cy[n, i] - Cy[n-1, i]) == (1/T2)*(Cy[1, i+1] - Cy[0, i+1])]
        constraints += [(1/T1)*(Cz[n, i] - Cz[n-1, i]) == (1/T2)*(Cz[1, i+1] - Cz[0, i+1])]
        # Acceleration
        constraints += [(1/T1**2)*(Cx[n, i] - 2*Cx[n-1, i] + Cx[n-2, i]) ==
                        (1/T2**2)*(Cx[2, i+1] - 2*Cx[1, i+1] + Cx[0, i+1])]
        constraints += [(1/T1**2)*(Cy[n, i] - 2*Cy[n-1, i] + Cy[n-2, i]) ==
                        (1/T2**2)*(Cy[2, i+1] - 2*Cy[1, i+1] + Cy[0, i+1])]
        constraints += [(1/T1**2)*(Cz[n, i] - 2*Cz[n-1, i] + Cz[n-2, i]) ==
                        (1/T2**2)*(Cz[2, i+1] - 2*Cz[1, i+1] + Cz[0, i+1])]

    # 3. Safety & Dynamic Constraints
    for i in range(m):
        box = corridor[i]
        T = time_segments[i]

        # Safety (Convex Hull Property)
        constraints += [Cx[:, i] >= box[0], Cx[:, i] <= box[1]]
        constraints += [Cy[:, i] >= box[2], Cy[:, i] <= box[3]]
        constraints += [Cz[:, i] >= box[4], Cz[:, i] <= box[5]]

        # Dynamics (Hodograph Property)
        # Velocity control points
        Vx = (n / T) * (Cx[1:, i] - Cx[:-1, i])
        Vy = (n / T) * (Cy[1:, i] - Cy[:-1, i])
        Vz = (n / T) * (Cz[1:, i] - Cz[:-1, i])
        # Constrain velocity magnitude (not just components)
        # Use squared magnitude constraint: vx^2 + vy^2 + vz^2 <= V_MAX^2
        # Vx has n elements (from Cx[1:] which has n elements)
        for j in range(n):
            constraints += [cp.square(Vx[j]) + cp.square(Vy[j]) + cp.square(Vz[j]) <= V_MAX**2]

        # Acceleration control points
        Ax = (n - 1) / T * (Vx[1:] - Vx[:-1])
        Ay = (n - 1) / T * (Vy[1:] - Vy[:-1])
        Az = (n - 1) / T * (Vz[1:] - Vz[:-1])
        # Constrain acceleration magnitude using squared constraint
        # Ax has n-1 elements (from Vx[1:] which has n-1 elements)
        for j in range(n - 1):
            constraints += [cp.square(Ax[j]) + cp.square(Ay[j]) + cp.square(Az[j]) <= A_MAX**2]

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
        return None, None, None, time_segments

    if prob.status == cp.OPTIMAL_INACCURATE:
        print("Warning: Optimization was inaccurate.")

    return Cx.value, Cy.value, Cz.value, time_segments

# --- STEP 4: VISUALIZATION ---
def plot_cuboid(ax, origin, size, color_face, edge_color=None):
    """Helper function to plot a 3D cuboid."""
    x, y, z = origin
    dx, dy, dz = size

    # Define the 8 vertices of the cuboid
    vertices = np.array([
        [x, y, z], [x+dx, y, z], [x+dx, y+dy, z], [x, y+dy, z],  # bottom face
        [x, y, z+dz], [x+dx, y, z+dz], [x+dx, y+dy, z+dz], [x, y+dy, z+dz]  # top face
    ])

    # Define the 6 faces of the cuboid
    faces = [
        [vertices[0], vertices[1], vertices[2], vertices[3]],  # bottom
        [vertices[4], vertices[5], vertices[6], vertices[7]],  # top
        [vertices[0], vertices[1], vertices[5], vertices[4]],  # front
        [vertices[2], vertices[3], vertices[7], vertices[6]],  # back
        [vertices[1], vertices[2], vertices[6], vertices[5]],  # right
        [vertices[0], vertices[3], vertices[7], vertices[4]]   # left
    ]

    edge_color = edge_color if edge_color is not None else color_face
    poly3d = Poly3DCollection(faces, alpha=0.3, facecolor=color_face, edgecolor=edge_color, linewidth=0.5)
    ax.add_collection3d(poly3d)

def plot_results(grid, path, corridor, Cx, Cy, Cz, obstacles, show_astar_path=True, show_corridor=True, show_control_points=True, show_obstacles=True, time_segments=None):
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Plot obstacles from JSON file as 3D boxes
    if show_obstacles and obstacles:
        # Sample obstacles for visualization if there are too many
        max_obstacles_to_plot = 1000  # Limit number of obstacles to plot for performance
        if len(obstacles) > max_obstacles_to_plot:
            import random
            obstacles_to_plot = random.sample(obstacles, max_obstacles_to_plot)
            print(f"Plotting {max_obstacles_to_plot} randomly sampled obstacles out of {len(obstacles)} total obstacles.")
        else:
            obstacles_to_plot = obstacles

        for box in obstacles_to_plot:
            if len(box) >= 6:
                x_min_rad, x_max_rad, y_min_rad, y_max_rad, z_min_rad, z_max_rad = box[0], box[1], box[2], box[3], box[4], box[5]
                origin = (x_min_rad, y_min_rad, z_min_rad)
                size = (x_max_rad - x_min_rad, y_max_rad - y_min_rad, z_max_rad - z_min_rad)
                # Only plot if size is valid
                if size[0] > 0 and size[1] > 0 and size[2] > 0:
                    plot_cuboid(ax, origin, size, 'red')

    # Corridor - already in radians [x_min, x_max, y_min, y_max, z_min, z_max]
    if show_corridor:
        for box in corridor:
            x_min_rad, x_max_rad, y_min_rad, y_max_rad, z_min_rad, z_max_rad = box[0], box[1], box[2], box[3], box[4], box[5]
            origin = (x_min_rad, y_min_rad, z_min_rad)
            size = (x_max_rad - x_min_rad, y_max_rad - y_min_rad, z_max_rad - z_min_rad)
            plot_cuboid(ax, origin, size, 'green')

    # A* Path - convert from grid indices to radians
    if show_astar_path and path is not None:
        path_rad = np.zeros((len(path), 3), dtype=float)
        for i, (grid_x, grid_y, grid_z) in enumerate(path):
            x_rad, y_rad, z_rad = grid_idx_to_rad(grid_x + 0.5, grid_y + 0.5, grid_z + 0.5)
            path_rad[i] = [x_rad, y_rad, z_rad]
        ax.plot(path_rad[:, 0], path_rad[:, 1], path_rad[:, 2], 'b-o', label='A* Path', markersize=3, linewidth=1)

    # Plot all waypoints
    for i, wp in enumerate(WAYPOINTS):
        if i == 0:
            # Start waypoint (green)
            ax.scatter([wp[0]], [wp[1]], [wp[2]], c='green', s=150, marker='o',
                      label='Start', edgecolors='black', linewidths=2, zorder=10)
        elif i == len(WAYPOINTS) - 1:
            # End waypoint (red)
            ax.scatter([wp[0]], [wp[1]], [wp[2]], c='red', s=150, marker='o',
                      label='End', edgecolors='black', linewidths=2, zorder=10)
        else:
            # Intermediate waypoints (blue)
            ax.scatter([wp[0]], [wp[1]], [wp[2]], c='blue', s=100, marker='s',
                      label='Waypoint' if i == 1 else '', edgecolors='black', linewidths=1.5, zorder=10)

    # Bezier Trajectory - already in radians
    custom_traj_handle = None
    if Cx is not None and Cy is not None and Cz is not None:
        t_vec = np.linspace(0, 1, 100)
        n = N_DEGREE
        bernstein_basis = np.zeros((n + 1, len(t_vec)))
        for i in range(n + 1):
            bernstein_basis[i, :] = comb(n, i) * (t_vec**i) * ((1 - t_vec)**(n - i))

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
            pz = bernstein_basis.T @ Cz[:, i]

            if use_time_coloring:
                segment_duration = time_segments[i]
                seg_times = cumulative_time + t_vec * segment_duration
                for j in range(len(px) - 1):
                    trajectory_segments.append(
                        np.array([
                            [px[j], py[j], pz[j]],
                            [px[j + 1], py[j + 1], pz[j + 1]]
                        ])
                    )
                    segment_time_values.append(float((seg_times[j] + seg_times[j + 1]) * 0.5))
                cumulative_time += segment_duration
            else:
                ax.plot(px, py, pz, 'r-', linewidth=2.5, label='Trajectory' if i == 0 else '')

            if show_control_points:
                ax.scatter(
                    Cx[:, i],
                    Cy[:, i],
                    Cz[:, i],
                    c='cyan',
                    s=30,
                    alpha=0.7,
                    label='Control Points' if i == 0 else ''
                )

        if use_time_coloring and trajectory_segments:
            lc = Line3DCollection(trajectory_segments, cmap=cmap, norm=norm, linewidth=2.5)
            lc.set_array(np.array(segment_time_values))
            lc.set_label('Trajectory')
            ax.add_collection3d(lc)
            cbar = fig.colorbar(lc, ax=ax, pad=0.12, fraction=0.05)
            cbar.set_label('Time (s)', fontsize=20)
            cbar.ax.tick_params(labelsize=18)
        else:
            custom_traj_handle = None

    ax.set_xlim(-0.04, 0.04)
    ax.set_ylim(-0.04, 0.04)
    ax.set_zlim(-0.04, 0.04)
    ax.set_xlabel(r'$y_1$ (m)', fontsize=20, labelpad=10)
    ax.set_ylabel(r'$y_2$ (m)', fontsize=20, labelpad=10)
    ax.set_zlabel(r'$y_3$ (m)', fontsize=20, labelpad=10)
    ax.tick_params(axis='both', which='major', labelsize=18)

    # Set custom tick labels for x, y, z axes
    tick_positions = [-0.032, 0, 0.032]
    tick_labels = ['-0.032', '0', '0.032']
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels)
    ax.set_zticks(tick_positions)
    ax.set_zticklabels(tick_labels)

    legend_handles, legend_labels = ax.get_legend_handles_labels()
    if custom_traj_handle is not None and 'Trajectory' not in legend_labels:
        legend_handles.append(custom_traj_handle)
        legend_labels.append('Trajectory')
    if legend_handles:
        ax.legend(legend_handles, legend_labels, fontsize=20)
    plt.grid(True)
    fig.tight_layout()

    plt.show()

# --- SAVE TRAJECTORY RESULTS ---
def save_trajectory_results(Cx, Cy, Cz, time_segments, corridor, start_pos, end_pos,
                            filepath=None):
    """
    Saves trajectory optimization results to a JSON file.

    Parameters:
    -----------
    Cx, Cy, Cz : numpy.ndarray
        Control points for Bezier curves (shape: (n+1, m))
    time_segments : list
        Time durations for each segment
    corridor : list
        List of corridor boxes [x_min, x_max, y_min, y_max, z_min, z_max]
    start_pos : tuple
        Start position (x, y, z)
    end_pos : tuple
        End position (x, y, z)
    filepath : str, optional
        Path to save the JSON file. If None, uses default path in Results folder.
    """
    if filepath is None:
        filepath = os.path.join("Results", "trajectory_case2_3D.json")

    # Ensure Results directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # Convert numpy arrays to lists for JSON serialization
    trajectory_data = {
        "control_points": {
            "Cx": Cx.tolist() if hasattr(Cx, 'tolist') else Cx,
            "Cy": Cy.tolist() if hasattr(Cy, 'tolist') else Cy,
            "Cz": Cz.tolist() if hasattr(Cz, 'tolist') else Cz
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
    grid, obstacles = generate_map()

    # Find path through all waypoints
    print(f"\n{'='*60}")
    print(f"MULTI-WAYPOINT PATH PLANNING")
    print(f"{'='*60}")
    print(f"Number of waypoints: {len(WAYPOINTS)}")
    for i, wp in enumerate(WAYPOINTS):
        print(f"  Waypoint {i}: ({wp[0]:.4f}, {wp[1]:.4f}, {wp[2]:.4f})")
    print(f"{'='*60}\n")

    # Plan trajectory through waypoints
    corridor_segments, Cx, Cy, Cz, time_segments, path_segments = plan_waypoint_trajectory(grid, WAYPOINTS)

    if Cx is not None and Cy is not None and Cz is not None:
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
        verify_waypoint_passage(WAYPOINTS, corridor_segments, Cx, Cy, Cz)

        # Print segment information (all in radians)
        print_segment_info(START_POS, END_POS, all_corridors, time_segments, Cx, Cy, Cz)

        # Save trajectory results to JSON file
        save_trajectory_results(Cx, Cy, Cz, time_segments, all_corridors, START_POS, END_POS)

        # Plot with visualization switches
        plot_results(grid, complete_path, all_corridors, Cx, Cy, Cz, obstacles,
                    show_astar_path=SHOW_ASTAR_PATH,
                    show_corridor=SHOW_CORRIDOR,
                    show_control_points=SHOW_CONTROL_POINTS,
                    show_obstacles=SHOW_OBSTACLES,
                    time_segments=time_segments)
    else:
        print("Trajectory optimization failed.")
