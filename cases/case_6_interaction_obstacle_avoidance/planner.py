import numpy as np
# Run from the repository root: python -m cases.case_6_interaction_obstacle_avoidance.planner
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
import heapq
import cvxpy as cp
from scipy.special import comb
import json
import math
import os

# --- CONFIGURATION ---
# Coordinate ranges (in radians)
X_MIN = -0.05  # x axis minimum (radians)
X_MAX = 0.05 # x axis maximum (radians)
Y_MIN = -0.05      # y axis minimum (radians)
Y_MAX = 0.05       # y axis maximum (radians)
Z_MIN = 0    # z axis minimum (radians) - orientation
Z_MAX = 2 * math.pi       # z axis maximum (radians) - orientation

# Grid resolution
GRID_RESOLUTION = 50  # grid cells per radian

# Compute MAP_SIZE from coordinate ranges and resolution
MAP_SIZE = (
    int(80),  # width
    int(80),  # height
    int(80)   # depth (theta)
)

# Results JSON file path
RESULTS_JSON_PATH = os.path.join("Results", "results_case6_3D_02.json")

START_POS = (0.01, -0.05, 1/2 * math.pi)  # Start position (x, y, theta)
END_POS = (0.01, 0.05, 1/2 * math.pi)  # End position (x, y, theta)

# --- ROBOT DYNAMICS PARAMETERS ---
V_MAX = 0.05 # Max linear velocity
A_MAX = 0.05  # Max linear acceleration

# --- TRAJECTORY PARAMETERS ---
N_DEGREE = 5   # Degree of Bezier curve

# Visualization switches
SHOW_ASTAR_PATH = False
SHOW_CORRIDOR = False
SHOW_CONTROL_POINTS = False
SHOW_OBSTACLES = False
CORRIDOR_SAMPLE_STEP = 5

# --- STEP 0: MAP GENERATION ---
def load_obstacles_from_json(json_path):
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        obstacles = []
        if 'undetermined' in data:
            obstacles.extend(data['undetermined'])
        if 'out_of_workspace' in data:
            obstacles.extend(data['out_of_workspace'])
        print(f"Loaded {len(obstacles)} obstacles from JSON file.")
        return obstacles
    except Exception as e:
        print(f"Error loading obstacles: {e}")
        return []

def rad_to_grid_idx(x_rad, y_rad, z_rad):
    grid_x = int((x_rad - X_MIN) / (X_MAX - X_MIN) * MAP_SIZE[0])
    grid_y = int((y_rad - Y_MIN) / (Y_MAX - Y_MIN) * MAP_SIZE[1])
    grid_z = int((z_rad - Z_MIN) / (Z_MAX - Z_MIN) * MAP_SIZE[2])
    return (max(0, min(grid_x, MAP_SIZE[0] - 1)),
            max(0, min(grid_y, MAP_SIZE[1] - 1)),
            max(0, min(grid_z, MAP_SIZE[2] - 1)))

def grid_idx_to_rad(grid_x, grid_y, grid_z):
    x_rad = X_MIN + (grid_x / MAP_SIZE[0]) * (X_MAX - X_MIN)
    y_rad = Y_MIN + (grid_y / MAP_SIZE[1]) * (Y_MAX - Y_MIN)
    z_rad = Z_MIN + (grid_z / MAP_SIZE[2]) * (Z_MAX - Z_MIN)
    return x_rad, y_rad, z_rad

def generate_map():
    grid = np.zeros(MAP_SIZE, dtype=int)
    obstacles = load_obstacles_from_json(RESULTS_JSON_PATH)
    for box in obstacles:
        if len(box) >= 6:
            x_min_rad, x_max_rad, y_min_rad, y_max_rad, z_min_rad, z_max_rad = box[:6]
            grid_x_min = int((x_min_rad - X_MIN) / (X_MAX - X_MIN) * MAP_SIZE[0])
            grid_x_max = int((x_max_rad - X_MIN) / (X_MAX - X_MIN) * MAP_SIZE[0])
            grid_y_min = int((y_min_rad - Y_MIN) / (Y_MAX - Y_MIN) * MAP_SIZE[1])
            grid_y_max = int((y_max_rad - Y_MIN) / (Y_MAX - Y_MIN) * MAP_SIZE[1])
            grid_z_min = int((z_min_rad - Z_MIN) / (Z_MAX - Z_MIN) * MAP_SIZE[2])
            grid_z_max = int((z_max_rad - Z_MIN) / (Z_MAX - Z_MIN) * MAP_SIZE[2])

            grid_x_min = max(0, min(grid_x_min, MAP_SIZE[0] - 1))
            grid_x_max = max(0, min(grid_x_max, MAP_SIZE[0]))
            grid_y_min = max(0, min(grid_y_min, MAP_SIZE[1] - 1))
            grid_y_max = max(0, min(grid_y_max, MAP_SIZE[1]))
            grid_z_min = max(0, min(grid_z_min, MAP_SIZE[2] - 1))
            grid_z_max = max(0, min(grid_z_max, MAP_SIZE[2]))

            if grid_x_max > grid_x_min and grid_y_max > grid_y_min and grid_z_max > grid_z_min:
                grid[grid_x_min:grid_x_max, grid_y_min:grid_y_max, grid_z_min:grid_z_max] = 1

    # Clear start/end
    s_idx = rad_to_grid_idx(*START_POS)
    e_idx = rad_to_grid_idx(*END_POS)
    for idx in [s_idx, e_idx]:
        x, y, z = idx
        grid[max(0, x-1):min(MAP_SIZE[0], x+2),
             max(0, y-1):min(MAP_SIZE[1], y+2),
             max(0, z-1):min(MAP_SIZE[2], z+2)] = 0
    return grid, obstacles

# --- STEP 1: A* PATH FINDING ---
def find_path_astar(grid, start, end):
    def heuristic(a, b):
        return np.sqrt((b[0]-a[0])**2 + (b[1]-a[1])**2) + 0.1 * abs(b[2]-a[2])

    def get_valid_neighbors(current):
        neighbors = []
        curr_x, curr_y, curr_theta_idx = current
        _, _, curr_theta_rad = grid_idx_to_rad(curr_x, curr_y, curr_theta_idx)

        # Forward moves only to simplify orientation
        forward_steps = [1, 2, 3]
        rotation_steps = [-2, -1, 0, 1, 2]

        for f_step in forward_steps:
            for r_step in rotation_steps:
                dx = f_step * np.cos(curr_theta_rad)
                dy = f_step * np.sin(curr_theta_rad)
                nx = int(curr_x + dx)
                ny = int(curr_y + dy)
                nt = (curr_theta_idx + r_step) % MAP_SIZE[2]

                if 0 <= nx < MAP_SIZE[0] and 0 <= ny < MAP_SIZE[1]:
                    if grid[nx, ny, nt] == 0:
                        dist = np.sqrt(dx**2 + dy**2)
                        cost = dist + abs(r_step)*0.5 + 1.0
                        neighbors.append(((nx, ny, nt), cost))
        return neighbors

    open_list = []
    heapq.heappush(open_list, (0, 0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, end)}

    while open_list:
        _, _, current = heapq.heappop(open_list)

        if (current[0] == end[0] and current[1] == end[1]):
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return np.array(path[::-1])

        for neighbor, cost in get_valid_neighbors(current):
            tent_g = g_score[current] + cost
            if tent_g < g_score.get(neighbor, float('inf')):
                came_from[neighbor] = current
                g_score[neighbor] = tent_g
                f_score[neighbor] = tent_g + heuristic(neighbor, end)
                heapq.heappush(open_list, (f_score[neighbor], tent_g, neighbor))

    print("A* failed.")
    return None

# --- STEP 2: CORRIDOR GENERATION ---
def generate_corridor(grid, path):
    # Same logic as before but simpler
    if CORRIDOR_SAMPLE_STEP > 1:
        indices = list(range(0, len(path), CORRIDOR_SAMPLE_STEP))
        if indices[-1] != len(path)-1: indices.append(len(path)-1)
        path = path[indices]

    corridor = []
    path_idx = 0
    while path_idx < len(path):
        start = path[path_idx]
        x_min, y_min, z_min = start
        x_max, y_max, z_max = start[0]+1, start[1]+1, start[2]+1

        end_idx = path_idx
        for i in range(path_idx+1, len(path)):
            p = path[i]
            tx_min, ty_min, tz_min = min(x_min, p[0]), min(y_min, p[1]), min(z_min, p[2])
            tx_max, ty_max, tz_max = max(x_max, p[0]+1), max(y_max, p[1]+1), max(z_max, p[2]+1)

            # Check collision in box
            subgrid = grid[max(0,tx_min):min(MAP_SIZE[0],tx_max),
                           max(0,ty_min):min(MAP_SIZE[1],ty_max),
                           max(0,tz_min):min(MAP_SIZE[2],tz_max)]
            if np.any(subgrid == 1):
                break
            x_min, y_min, z_min = tx_min, ty_min, tz_min
            x_max, y_max, z_max = tx_max, ty_max, tz_max
            end_idx = i

        # Expand box
        while x_min > 0 and np.all(grid[x_min-1, y_min:y_max, z_min:z_max]==0): x_min-=1
        while x_max < MAP_SIZE[0] and np.all(grid[x_max, y_min:y_max, z_min:z_max]==0): x_max+=1
        while y_min > 0 and np.all(grid[x_min:x_max, y_min-1, z_min:z_max]==0): y_min-=1
        while y_max < MAP_SIZE[1] and np.all(grid[x_min:x_max, y_max, z_min:z_max]==0): y_max+=1
        # Keep Z constrained to path to avoid huge rotation boxes

        x_min_rad, y_min_rad, z_min_rad = grid_idx_to_rad(x_min, y_min, z_min)
        x_max_rad, y_max_rad, z_max_rad = grid_idx_to_rad(x_max, y_max, z_max)
        corridor.append([x_min_rad, x_max_rad, y_min_rad, y_max_rad, z_min_rad, z_max_rad])

        path_idx = end_idx + 1 if end_idx < len(path)-1 else len(path) # Fix loop
        if end_idx == len(path)-1: break # Done

    return corridor

# --- STEP 3: OPTIMIZATION ---
def optimize_trajectory(start_pos, end_pos, corridor):
    n = N_DEGREE
    m = len(corridor)

    Cx = cp.Variable((n + 1, m))
    Cy = cp.Variable((n + 1, m))
    # No Cz variable! Orientation is derived.

    time_segments = []
    for i in range(m):
        box = corridor[i]
        center = np.array([(box[0]+box[1])/2, (box[2]+box[3])/2])
        if i < m-1:
            next_box = corridor[i+1]
            next_center = np.array([(next_box[0]+next_box[1])/2, (next_box[2]+next_box[3])/2])
            dist = np.linalg.norm(center - next_center)
        else:
            dist = np.linalg.norm(center - np.array(end_pos[:2]))
        time_segments.append(max(dist / V_MAX * 10.0, 1.0)) # Conservative time

    cost = 0
    constraints = []

    # Start/End Constraints
    constraints += [Cx[0,0] == start_pos[0], Cy[0,0] == start_pos[1]]
    constraints += [Cx[n,m-1] == end_pos[0], Cy[n,m-1] == end_pos[1]]

    # Initial Orientation constraint: Velocity must align with start_theta
    # (P1 - P0) is tangent vector
    # dy * cos(theta) - dx * sin(theta) = 0
    constraints += [(Cy[1,0] - Cy[0,0]) * np.cos(start_pos[2]) == (Cx[1,0] - Cx[0,0]) * np.sin(start_pos[2])]
    # Ensure non-zero velocity in correct direction (approximate)
    constraints += [(Cy[1,0] - Cy[0,0]) * np.sin(start_pos[2]) + (Cx[1,0] - Cx[0,0]) * np.cos(start_pos[2]) >= 0.001]

    # Final Orientation constraint
    constraints += [(Cy[n,m-1] - Cy[n-1,m-1]) * np.cos(end_pos[2]) == (Cx[n,m-1] - Cx[n-1,m-1]) * np.sin(end_pos[2])]

    for i in range(m):
        T = time_segments[i]
        # Velocity
        Vx = (n/T) * (Cx[1:,i] - Cx[:-1,i])
        Vy = (n/T) * (Cy[1:,i] - Cy[:-1,i])

        # Acceleration
        Ax = (n-1)/T * (Vx[1:] - Vx[:-1])
        Ay = (n-1)/T * (Vy[1:] - Vy[:-1])

        # Jerk
        Jx = (n-2)/T * (Ax[1:] - Ax[:-1])
        Jy = (n-2)/T * (Ay[1:] - Ay[:-1])

        cost += cp.sum_squares(Jx) + cp.sum_squares(Jy)

        # Constraints
        box = corridor[i]
        constraints += [Cx[:,i] >= box[0], Cx[:,i] <= box[1]]
        constraints += [Cy[:,i] >= box[2], Cy[:,i] <= box[3]]

        # Velocity Limit
        for j in range(n):
            constraints += [cp.square(Vx[j]) + cp.square(Vy[j]) <= V_MAX**2]

        # Continuity
        if i < m-1:
            T2 = time_segments[i+1]
            constraints += [Cx[n,i] == Cx[0,i+1], Cy[n,i] == Cy[0,i+1]]
            constraints += [(1/T)*(Cx[n,i]-Cx[n-1,i]) == (1/T2)*(Cx[1,i+1]-Cx[0,i+1])]
            constraints += [(1/T)*(Cy[n,i]-Cy[n-1,i]) == (1/T2)*(Cy[1,i+1]-Cy[0,i+1])]
            constraints += [(1/T**2)*(Cx[n,i]-2*Cx[n-1,i]+Cx[n-2,i]) == (1/T2**2)*(Cx[2,i+1]-2*Cx[1,i+1]+Cx[0,i+1])]
            constraints += [(1/T**2)*(Cy[n,i]-2*Cy[n-1,i]+Cy[n-2,i]) == (1/T2**2)*(Cy[2,i+1]-2*Cy[1,i+1]+Cy[0,i+1])]

    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve(solver=cp.ECOS, verbose=False)

    if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        print(f"Optimization failed: {prob.status}")
        return None, None, None

    return Cx.value, Cy.value, time_segments

# --- PLOTTING & EXPORT ---
def analyze_and_plot(Cx, Cy, time_segments):
    # Sample trajectory
    t_vec = np.linspace(0, 1, 100)
    n = N_DEGREE
    basis = np.zeros((n+1, len(t_vec)))
    for i in range(n+1):
        basis[i,:] = comb(n, i) * (t_vec**i) * ((1-t_vec)**(n-i))

    all_x, all_y, all_vx, all_vy, all_theta = [], [], [], [], []

    for i in range(Cx.shape[1]):
        px = basis.T @ Cx[:,i]
        py = basis.T @ Cy[:,i]

        # Derivative for velocity
        # d/dt B(t) = n * (P_{i+1} - P_i) * B_{n-1}(t)
        basis_vel = np.zeros((n, len(t_vec)))
        for j in range(n):
            basis_vel[j,:] = comb(n-1, j) * (t_vec**j) * ((1-t_vec)**(n-1-j))

        # Control points for velocity
        T = time_segments[i]
        CP_vx = (n/T) * (Cx[1:,i] - Cx[:-1,i])
        CP_vy = (n/T) * (Cy[1:,i] - Cy[:-1,i])

        vx = basis_vel.T @ CP_vx
        vy = basis_vel.T @ CP_vy

        # Calculate theta from velocity vector
        theta = np.arctan2(vy, vx)

        all_x.extend(px)
        all_y.extend(py)
        all_vx.extend(vx)
        all_vy.extend(vy)
        all_theta.extend(theta)

    # Save
    # Time vector
    # Calculate cumulative time from time_segments
    # time_segments contains the duration T for each segment.
    # We sample 100 points per segment.
    total_duration = sum(time_segments)
    cum_time = [0]
    current_t = 0
    for T in time_segments:
        current_t += T
        cum_time.append(current_t)

    all_time = []
    for i in range(len(time_segments)):
        # Local time t_vec goes from 0 to 1. Actual time is t_vec * T + start_time_of_segment
        segment_start_time = cum_time[i]
        T = time_segments[i]
        # t_vec is 0..1
        # real_time = t_vec * T + segment_start_time
        real_t = t_vec * T + segment_start_time
        all_time.extend(real_t)

    # Ensure all_time matches length of other lists
    all_time = all_time[:len(all_x)]

    data = {
        "trajectory": list(zip(all_x, all_y, all_theta, all_time)),
        "start": START_POS,
        "end": END_POS
    }

    # Save trajectory to a SEPARATE file, NOT overwriting the workspace results
    traj_file_path = os.path.join("Results", "trajectory_case6_3D_02.json")
    with open(traj_file_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Trajectory saved to: {traj_file_path}")

    all_x_arr = np.array(all_x)
    all_y_arr = np.array(all_y)
    all_theta_arr = np.array(all_theta)

    # Use the actual time vector instead of normalized 0-1
    t = np.array(all_time)

    # Use actual time for normalization
    norm = plt.Normalize(t.min(), t.max())

    # Figure 1: 2D
    fig1 = plt.figure(figsize=(7,7))
    ax1 = fig1.add_subplot(111)

    points = np.array([all_x_arr, all_y_arr]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    lc = LineCollection(segments, cmap='viridis', norm=norm)
    lc.set_array(t)
    lc.set_linewidth(4)
    line = ax1.add_collection(lc)

    ax1.plot(START_POS[0], START_POS[1], 'go', label='Start')
    ax1.plot(END_POS[0], END_POS[1], 'ro', label='End')

    # Plot arrows to show orientation, let it above the trajectory
    # Sample every Nth point to avoid clutter
    step = max(1, len(all_x_arr) // 20)
    for i in range(0, len(all_x_arr), step):
        ax1.arrow(all_x_arr[i], all_y_arr[i], 0.003 * np.cos(all_theta_arr[i]), 0.003 * np.sin(all_theta_arr[i]), head_width=0.003, head_length=0.005, fc='r', ec='r', alpha=0.6)

    ax1.set_xlim(-0.05, 0.05)
    ax1.set_ylim(-0.05, 0.05)
    ax1.set_xlabel(r'$x$ (m)', fontsize=20)
    ax1.set_ylabel(r'$y$ (m)', fontsize=20)
    # ax1.set_title(f'2D Trajectory (Color = Time, Total Time = {total_duration:.2f}s)')
    ax1.tick_params(axis='both', which='major', labelsize=18)
    ax1.grid(True)
    ax1.legend(fontsize=20)
    cbar = fig1.colorbar(line, ax=ax1, fraction=0.05)
    cbar.set_label('Time (s)', fontsize=20)
    cbar.ax.tick_params(labelsize=18)

    # Figure 2: 3D
    fig2 = plt.figure(figsize=(8,8))
    ax2 = fig2.add_subplot(111, projection='3d')

    points_3d = np.array([all_x_arr, all_y_arr, all_theta_arr]).T.reshape(-1, 1, 3)
    segments_3d = np.concatenate([points_3d[:-1], points_3d[1:]], axis=1)

    lc_3d = Line3DCollection(segments_3d, cmap='viridis', norm=norm)
    lc_3d.set_array(t)
    lc_3d.set_linewidth(4)
    ax2.add_collection3d(lc_3d)
    ax2.plot(START_POS[0], START_POS[1], START_POS[2], 'go', label='Start')
    ax2.plot(END_POS[0], END_POS[1], END_POS[2], 'ro', label='End')
    ax2.set_xlim(-0.05, 0.05)
    ax2.set_ylim(-0.05, 0.05)
    ax2.set_zlim(0, 2*math.pi)

    zticks = [0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi]
    zlabels = ['0', 'π/2', 'π', '3π/2', '2π']
    ax2.set_zticks(zticks, zlabels)

    ax2.set_xlabel(r'$x$ (m)', fontsize=20, labelpad=10)
    ax2.set_ylabel(r'$y$ (m)', fontsize=20, labelpad=13)
    ax2.set_zlabel(r'$\beta$ (rad)', fontsize=20, labelpad=10)
    # ax2.set_title(f'3D Trajectory (Z = Orientation, Color = Time, Total Time = {total_duration:.2f}s)')
    ax2.tick_params(axis='both', which='major', labelsize=18)
    cbar = fig2.colorbar(lc_3d, ax=ax2, fraction=0.05)
    cbar.set_label('Time (s)', fontsize=20)
    cbar.ax.tick_params(labelsize=18)
    ax2.legend(fontsize=20)
    # Save figures
    fig1.savefig(os.path.join("Results", "Figure_trajectory_2D_time.png"))
    fig2.savefig(os.path.join("Results", "Figure_trajectory_3D_time.png"))
    print("Figures saved to Results/Figure_trajectory_2D_time.png and Results/Figure_trajectory_3D_time.png")

    plt.show()

if __name__ == "__main__":
    grid, _ = generate_map()
    s_grid = rad_to_grid_idx(*START_POS)
    e_grid = rad_to_grid_idx(*END_POS)

    path = find_path_astar(grid, s_grid, e_grid)
    if path is not None:
        corridor = generate_corridor(grid, path)
        Cx, Cy, times = optimize_trajectory(START_POS, END_POS, corridor)
        if Cx is not None:
            analyze_and_plot(Cx, Cy, times)
