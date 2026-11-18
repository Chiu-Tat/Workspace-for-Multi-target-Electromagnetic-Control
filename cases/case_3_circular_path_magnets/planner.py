import numpy as np
# Run from the repository root: python -m cases.case_3_circular_path_magnets.planner
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import heapq
import cvxpy as cp
from scipy.special import comb
import json
import math
import os

# --- CONFIGURATION ---
# Coordinate ranges (in radians)
X_MIN = 144/180 * math.pi  # x axis minimum (radians)
X_MAX = 216/180 * math.pi  # x axis maximum (radians)
Y_MIN = -1/5 * math.pi      # y axis minimum (radians)
Y_MAX = 1/5 * math.pi       # y axis maximum (radians)

# Grid resolution: number of grid cells per radian
# Higher values = finer grid, more precise but slower
GRID_RESOLUTION = 50  # grid cells per radian

# Compute MAP_SIZE from coordinate ranges and resolution
MAP_SIZE = (
    int((X_MAX - X_MIN) * GRID_RESOLUTION),  # width in grid cells
    int((Y_MAX - Y_MIN) * GRID_RESOLUTION)   # height in grid cells
)

# Results JSON file path
RESULTS_JSON_PATH = os.path.join("Results", "results_case3.json")

START_POS = (144/180 * math.pi, -1/5 * math.pi)  # Start position in radians (real coordinates)
END_POS = (216/180 * math.pi, 1/5 * math.pi)  # End position in radians (real coordinates)

V_MAX = 0.2  # Max velocity (m/s)
A_MAX = 0.1   # Max acceleration (m/s^2)
N_DEGREE = 5   # Degree of Bezier curve

# Visualization switches
SHOW_ASTAR_PATH = False   # Set to True to display A* path
SHOW_CORRIDOR = False      # Set to True to display corridor boxes
SHOW_CONTROL_POINTS = False  # Set to True to display Bezier curve control points

# Corridor generation sampling
CORRIDOR_SAMPLE_STEP = 20 # Sample path every CORRIDOR_SAMPLE_STEP points (1 = use all points)

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
def plot_results(grid, path, corridor, Cx, Cy, show_astar_path=True, show_corridor=True, show_control_points=True):
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
    if Cx is not None and Cy is not None:
        t_vec = np.linspace(0, 1, 100)
        n = N_DEGREE
        bernstein_basis = np.zeros((n + 1, len(t_vec)))
        for i in range(n + 1):
            bernstein_basis[i, :] = comb(n, i) * (t_vec**i) * ((1 - t_vec)**(n - i))

        for i in range(Cx.shape[1]):
            # Control points are already in radians
            px = bernstein_basis.T @ Cx[:, i]
            py = bernstein_basis.T @ Cy[:, i]
            ax.plot(px, py, 'r-', linewidth=2.5, label='Bezier Trajectory' if i == 0 else '')
            if show_control_points:
                ax.plot(Cx[:, i], Cy[:, i], 'c--o', markersize=4, alpha=0.7,
                       label='Control Points' if i == 0 else '')

    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('X (radians)')
    ax.set_ylabel('Y (radians)')
    ax.legend()
    # ax.set_title("Quadrotor Motion Planning")
    plt.grid(True)
    plt.show()

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
            # Plot with visualization switches
            plot_results(grid, path, corridor, Cx, Cy,
                        show_astar_path=SHOW_ASTAR_PATH,
                        show_corridor=SHOW_CORRIDOR,
                        show_control_points=SHOW_CONTROL_POINTS)
        else:
            print("Optimization failed, cannot print segment information.")
    else:
        print("Pathfinding failed, cannot proceed with optimization.")
