"""
Real-time rotating magnetic field simulation for multiple target points.
- Right stick: Control position (X, Y) of the currently selected target point
- Button A / Tab: Switch the target point currently being controlled
- D-pad Up/Down or Arrow Up/Down: Increase/decrease the selected point rotation speed
- All target points share one rotating-field amplitude radius computed from
  WS_lib.multi_point_rotating_radius()
"""

import time

import matplotlib.pyplot as plt
import numpy as np
import pygame
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
from scipy.optimize import minimize

from WS_lib import multi_point_rotating_radius

# ===================== Coil Parameters =====================
params_list = [
    np.array([-13.04945069, -4.41557229, 6.47376799, 0.12129096, 0.00466922, -0.0174842]),
    np.array([-5.10083416, 13.54294901, 7.85474539, 0.05834654, -0.11165548, -0.01850546]),
    np.array([4.05088788, 14.23365818, 6.44760956, -0.05903076, -0.11020417, -0.01488244]),
    np.array([13.89011305, -0.06092074, 4.77365608, -0.12306086, -0.00085745, -0.01378161]),
    np.array([11.44363813, -9.40543896, 4.46367162, -0.06806179, 0.1024875, -0.01397152]),
    np.array([-9.00577939, -12.78905365, 5.98650851, 0.06473315, 0.10618968, -0.0151172]),
    np.array([0.92820081, 8.54965337, 8.72298349, -0.00381254, -0.08845466, -0.08874662]),
    np.array([8.7302819, -4.90773115, 7.00109937, -0.07977306, 0.04481733, -0.08536032]),
    np.array([-7.68962762, -6.83258326, 8.12112247, 0.07498008, 0.04542436, -0.08696975]),
    np.array([2.35614001, -1.11370036, 14.00304846, -0.00722183, 0.00029277, -0.12482979]),
]

num_coils = len(params_list)
mu0 = 4 * np.pi * 1e-7

# Current limits (Amperes)
CURRENT_MAX = 15
CURRENT_MIN = -15

# Workspace and rotation settings
CONTROL_RADIUS = 0.06
DISPLAY_RADIUS = 0.08
MIN_POINT_SPACING = 0.02
INITIAL_ROTATION_FREQUENCY_HZ = 0.5
INITIAL_FREQUENCY_SPREAD_HZ = 0.2
ROTATION_FREQUENCY_STEP_HZ = 0.1
MIN_ROTATION_FREQUENCY_HZ = 0.0
MAX_ROTATION_FREQUENCY_HZ = 5.0

# Arrow display settings
USE_UNIFORM_ARROW_LENGTH = False
ARROW_DISPLAY_LENGTH = 0.02
AMPLITUDE_ARROW_MAX_LENGTH = 0.06
AMPLITUDE_ARROW_MIN_LENGTH = 0.001
AMPLITUDE_ARROW_EXPONENT = 0.25
SHOW_ARROW_MAGNITUDE_LABELS = True

# Field colormap settings
FIELD_COLORBAR_MAX_TESLA = 0.3

# Target overlay settings
FIELD_CIRCLE_MIN_RADIUS = 0.004
FIELD_CIRCLE_MAX_RADIUS = 0.06
FIELD_CIRCLE_EXPONENT = 1
FIELD_CIRCLE_REFERENCE_TESLA = FIELD_COLORBAR_MAX_TESLA


# ===================== Magnetic Field Calculation =====================
def dipole_field_contribution(params, x_pos, y_pos, z_pos):
    """Return the field contribution per unit current from one coil."""
    m0, m1, m2, r0_0, r0_1, r0_2 = params

    dx = x_pos - r0_0
    dy = y_pos - r0_1
    dz = z_pos - r0_2

    radius = np.sqrt(dx**2 + dy**2 + dz**2) + 1e-9
    dot_product = m0 * dx + m1 * dy + m2 * dz

    coeff = mu0 / (4 * np.pi)
    bx = coeff * (3 * dx * dot_product / radius**5 - m0 / radius**3)
    by = coeff * (3 * dy * dot_product / radius**5 - m1 / radius**3)
    bz = coeff * (3 * dz * dot_product / radius**5 - m2 / radius**3)

    return np.array([bx, by, bz])


def build_field_matrix(x_pos, y_pos, z_pos):
    """Build the actuation matrix A such that B = A @ currents."""
    matrix = np.zeros((3, num_coils))
    for i in range(num_coils):
        matrix[:, i] = dipole_field_contribution(params_list[i], x_pos, y_pos, z_pos)
    return matrix


def calculate_b(currents, x_pos, y_pos, z_pos):
    """Calculate total magnetic field at the given position."""
    matrix = build_field_matrix(x_pos, y_pos, z_pos)
    field = matrix @ currents
    return field[0], field[1], field[2]


def build_inplane_target_system(target_specs):
    """Build the stacked in-plane field map and desired target vector."""
    matrix_rows = []
    target_values = []

    for spec in target_specs:
        full_matrix = build_field_matrix(*spec["position"])
        matrix_rows.append(full_matrix[:2, :])
        target_values.extend(spec["target_field"][:2])

    if not matrix_rows:
        return np.zeros((0, num_coils)), np.zeros(0)

    return np.vstack(matrix_rows), np.array(target_values, dtype=float)


def optimize_currents_for_targets(target_specs):
    """Match the in-plane rotating field targets for all points at once."""
    target_matrix, target_vector = build_inplane_target_system(target_specs)

    def objective(x_var):
        currents = CURRENT_MAX * np.tanh(x_var)
        field_error = target_matrix @ currents - target_vector
        return np.dot(field_error, field_error)

    def gradient(x_var):
        currents = CURRENT_MAX * np.tanh(x_var)
        field_error = target_matrix @ currents - target_vector
        gradient_currents = 2.0 * target_matrix.T @ field_error
        tanh_x = np.tanh(x_var)
        sech2_x = 1 - tanh_x**2
        return gradient_currents * CURRENT_MAX * sech2_x

    result = minimize(
        objective,
        np.zeros(num_coils),
        method="L-BFGS-B",
        jac=gradient,
        options={"maxiter": 100, "ftol": 1e-8},
    )

    optimal_currents = CURRENT_MAX * np.tanh(result.x)
    return np.clip(optimal_currents, CURRENT_MIN, CURRENT_MAX)


# ===================== Visualization Grid =====================
def create_visualization_grid(z_plane=0.0, radius_max=DISPLAY_RADIUS, num_r=50, num_theta=100):
    """Create a polar grid for visualization."""
    radius = np.linspace(0.001, radius_max, num_r)
    theta = np.linspace(0, 2 * np.pi, num_theta)
    grid_r, grid_theta = np.meshgrid(radius, theta)
    x_grid = grid_r * np.cos(grid_theta)
    y_grid = grid_r * np.sin(grid_theta)
    z_grid = np.full_like(x_grid, z_plane)
    return x_grid, y_grid, z_grid


def calculate_field_magnitude(currents, x_grid, y_grid, z_grid):
    """Calculate field magnitude and in-plane components over a grid."""
    shape = x_grid.shape
    bx = np.zeros(shape)
    by = np.zeros(shape)
    bz = np.zeros(shape)

    for i in range(num_coils):
        params = params_list[i]
        m0, m1, m2, r0_0, r0_1, r0_2 = params

        dx = x_grid - r0_0
        dy = y_grid - r0_1
        dz = z_grid - r0_2

        radius = np.sqrt(dx**2 + dy**2 + dz**2) + 1e-9
        dot_product = m0 * dx + m1 * dy + m2 * dz

        coeff = mu0 / (4 * np.pi)
        bx += coeff * (3 * dx * dot_product / radius**5 - m0 / radius**3) * currents[i]
        by += coeff * (3 * dy * dot_product / radius**5 - m1 / radius**3) * currents[i]
        bz += coeff * (3 * dz * dot_product / radius**5 - m2 / radius**3) * currents[i]

    magnitude = np.sqrt(bx**2 + by**2 + bz**2)
    return magnitude, bx, by


# ===================== Target Initialization =====================
def is_far_enough(candidate, existing_points, min_spacing):
    """Check whether a new point respects the minimum spacing."""
    for point in existing_points:
        if np.linalg.norm(candidate[:2] - point[:2]) < min_spacing:
            return False
    return True


def generate_initial_positions(num_points, max_radius=CONTROL_RADIUS, min_spacing=MIN_POINT_SPACING):
    """Generate evenly spread initial positions inside the control workspace."""
    if num_points <= 0:
        return []

    if num_points == 1:
        return [np.array([0.0, 0.0, 0.0])]

    positions = []
    radial_step = max(min_spacing, max_radius / max(2, int(np.ceil(np.sqrt(num_points)))))
    ring_radii = np.arange(0.0, max_radius + 0.5 * radial_step, radial_step)

    for radius in ring_radii:
        if len(positions) >= num_points:
            break

        if radius < 1e-9:
            candidate = np.array([0.0, 0.0, 0.0])
            if is_far_enough(candidate, positions, min_spacing):
                positions.append(candidate)
            continue

        circumference = 2 * np.pi * radius
        count_on_ring = max(6, int(np.floor(circumference / min_spacing)))
        angles = np.linspace(0.0, 2.0 * np.pi, count_on_ring, endpoint=False)

        for angle in angles:
            candidate = np.array([radius * np.cos(angle), radius * np.sin(angle), 0.0])
            if is_far_enough(candidate, positions, min_spacing):
                positions.append(candidate)
                if len(positions) >= num_points:
                    break

    if len(positions) < num_points:
        raise ValueError(
            f"Cannot place {num_points} points inside the workspace while keeping "
            f"{min_spacing * 1000:.1f} mm spacing."
        )

    return positions[:num_points]


def get_num_points():
    """Prompt the user to manually set the number of target points."""
    while True:
        try:
            raw_value = input("Enter the number of target points to control: ").strip()
            num_points = int(raw_value)
            if num_points <= 0:
                print("Please enter a positive integer.")
                continue

            max_points_estimate = int(np.pi * CONTROL_RADIUS**2 / (np.pi * (MIN_POINT_SPACING / 2) ** 2))
            if num_points > max_points_estimate:
                print(
                    f"Please choose {max_points_estimate} points or fewer so the initial spacing stays safe."
                )
                continue

            return num_points
        except ValueError:
            print("Invalid input. Please enter an integer.")


def build_radius_target_points(targets):
    """Build the target-point spec expected by WS_lib.multi_point_rotating_radius."""
    radius_targets = []
    for target in targets:
        x_pos, y_pos, z_pos = target["position"]
        radius_targets.append(
            {
                "X": x_pos,
                "Y": y_pos,
                "Z": z_pos,
                "m": 1.0,
                "alpha": np.pi / 2,
                "beta": 0.0,
                "Bx": True,
                "By": True,
                "Bz": None,
                "Bx_dx": None,
                "Bx_dy": None,
                "Bx_dz": None,
                "By_dy": None,
                "By_dz": None,
                "fx": None,
                "fy": None,
                "fz": None,
                "tx": None,
                "ty": None,
                "tz": None,
            }
        )
    return radius_targets


# ===================== Xbox Controller Handler =====================
class XboxController:
    def __init__(self):
        pygame.init()
        pygame.joystick.init()

        self.joystick = None
        self.connected = False

        self.right_stick_x = 0.0
        self.right_stick_y = 0.0
        self.move_speed = 0.002
        self.deadzone = 0.15

        self.switch_pressed = False
        self.speed_up_pressed = False
        self.speed_down_pressed = False

        self._previous_switch_state = False
        self._previous_speed_up_state = False
        self._previous_speed_down_state = False

        self._connect()

    def _connect(self):
        """Try to connect to the first available controller."""
        if pygame.joystick.get_count() > 0:
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()
            self.connected = True
            print(f"Controller connected: {self.joystick.get_name()}")
        else:
            print("No controller found. Using keyboard fallback (WASD, Tab, Up/Down).")

    def _apply_deadzone(self, value):
        """Apply joystick deadzone."""
        if abs(value) < self.deadzone:
            return 0.0
        return value

    def update(self):
        """Update controller and keyboard state."""
        pygame.event.pump()

        self.right_stick_x = 0.0
        self.right_stick_y = 0.0

        raw_switch_state = False
        raw_speed_up_state = False
        raw_speed_down_state = False

        if self.connected and self.joystick:
            self.right_stick_x = self._apply_deadzone(self.joystick.get_axis(2))
            self.right_stick_y = self._apply_deadzone(-self.joystick.get_axis(3))
            raw_switch_state = bool(self.joystick.get_button(0))

            hat_x, hat_y = self.joystick.get_hat(0)
            raw_speed_up_state = hat_y > 0
            raw_speed_down_state = hat_y < 0

        keys = pygame.key.get_pressed()
        if keys[pygame.K_d]:
            self.right_stick_x = 1.0
        elif keys[pygame.K_a]:
            self.right_stick_x = -1.0

        if keys[pygame.K_w]:
            self.right_stick_y = 1.0
        elif keys[pygame.K_s]:
            self.right_stick_y = -1.0

        raw_switch_state = raw_switch_state or bool(keys[pygame.K_TAB])
        raw_speed_up_state = raw_speed_up_state or bool(keys[pygame.K_UP])
        raw_speed_down_state = raw_speed_down_state or bool(keys[pygame.K_DOWN])

        self.switch_pressed = raw_switch_state and not self._previous_switch_state
        self.speed_up_pressed = raw_speed_up_state and not self._previous_speed_up_state
        self.speed_down_pressed = raw_speed_down_state and not self._previous_speed_down_state

        self._previous_switch_state = raw_switch_state
        self._previous_speed_up_state = raw_speed_up_state
        self._previous_speed_down_state = raw_speed_down_state

    def get_position_delta(self):
        """Return position delta for the selected point."""
        return self.right_stick_x * self.move_speed, self.right_stick_y * self.move_speed


# ===================== Real-Time Simulation =====================
class MagneticFieldSimulator:
    def __init__(self, num_points):
        self.controller = XboxController()
        self.num_points = num_points

        initial_positions = generate_initial_positions(num_points)
        self.targets = []
        for index, position in enumerate(initial_positions):
            self.targets.append(
                {
                    "position": position.copy(),
                    "phase": 0.0,
                    "frequency_hz": INITIAL_ROTATION_FREQUENCY_HZ + index * INITIAL_FREQUENCY_SPREAD_HZ,
                    "target_field": np.array([0.0, 0.0, 0.0]),
                    "actual_field": np.array([0.0, 0.0, 0.0]),
                }
            )

        self.active_target_index = 0
        self.shared_radius = 0.0
        self.currents = np.zeros(num_coils)
        self.animation = None

        self.x_grid, self.y_grid, self.z_grid = create_visualization_grid()

        self.num_arrows = 20
        x_arrow = np.linspace(-DISPLAY_RADIUS, DISPLAY_RADIUS, self.num_arrows)
        y_arrow = np.linspace(-DISPLAY_RADIUS, DISPLAY_RADIUS, self.num_arrows)
        self.x_arrow, self.y_arrow = np.meshgrid(x_arrow, y_arrow)
        self.z_arrow = np.zeros_like(self.x_arrow)

        self.last_update_time = time.perf_counter()

        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.setup_plot()

    def setup_plot(self):
        """Initialize the plot."""
        self.ax.set_xlim(-DISPLAY_RADIUS, DISPLAY_RADIUS)
        self.ax.set_ylim(-DISPLAY_RADIUS, DISPLAY_RADIUS)
        self.ax.set_aspect("equal")
        self.ax.set_xlabel("X (m)")
        self.ax.set_ylabel("Y (m)")
        self.ax.set_title("Rotating Magnetic Field Simulation (Multiple Target Points)")

        self.magnitude_plot = self.ax.pcolormesh(
            self.x_grid,
            self.y_grid,
            np.zeros_like(self.x_grid),
            cmap="jet",
            shading="gouraud",
            vmin=0,
            vmax=FIELD_COLORBAR_MAX_TESLA,
        )
        self.colorbar = self.fig.colorbar(self.magnitude_plot, ax=self.ax, label="|B| (T)")

        self.quiver_arrows = self.ax.quiver(
            self.x_arrow,
            self.y_arrow,
            np.zeros_like(self.x_arrow),
            np.zeros_like(self.y_arrow),
            color="black",
            scale=30,
            width=0.003,
        )

        self.point_markers = []
        self.field_circles = []
        self.direction_arrows = []
        self.magnitude_labels = []
        for _ in range(self.num_points):
            marker, = self.ax.plot([], [], "o", markersize=4, markerfacecolor="none", markeredgewidth=1.5)
            self.point_markers.append(marker)
            circle = Circle((0, 0), FIELD_CIRCLE_MIN_RADIUS, fill=False, edgecolor="gray", linewidth=2)
            self.ax.add_patch(circle)
            self.field_circles.append(circle)
            arrow = self.ax.quiver(0, 0, 0, 0, color="gray", scale=1, scale_units="xy", angles="xy", width=0.006)
            self.direction_arrows.append(arrow)
            label = self.ax.text(
                0,
                0,
                "",
                color="white",
                fontsize=8,
                ha="left",
                va="bottom",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="black", alpha=0.45),
            )
            self.magnitude_labels.append(label)

        self.info_text = self.ax.text(
            0.02,
            0.98,
            "",
            transform=self.ax.transAxes,
            verticalalignment="top",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

    def move_active_target(self):
        """Update the currently selected target point from controller input."""
        dx, dy = self.controller.get_position_delta()
        target = self.targets[self.active_target_index]

        new_position = target["position"].copy()
        new_position[0] += dx
        new_position[1] += dy

        distance = np.sqrt(new_position[0] ** 2 + new_position[1] ** 2)
        if distance > CONTROL_RADIUS:
            new_position[0] = new_position[0] / distance * CONTROL_RADIUS
            new_position[1] = new_position[1] / distance * CONTROL_RADIUS

        target["position"] = new_position

    def update_active_target_speed(self):
        """Adjust the rotation speed of the selected target point."""
        target = self.targets[self.active_target_index]

        if self.controller.speed_up_pressed:
            target["frequency_hz"] = min(
                MAX_ROTATION_FREQUENCY_HZ,
                target["frequency_hz"] + ROTATION_FREQUENCY_STEP_HZ,
            )
        if self.controller.speed_down_pressed:
            target["frequency_hz"] = max(
                MIN_ROTATION_FREQUENCY_HZ,
                target["frequency_hz"] - ROTATION_FREQUENCY_STEP_HZ,
            )

    def update_target_phases(self, dt_seconds):
        """Advance each target phase with its own rotation speed."""
        for target in self.targets:
            target["phase"] = (target["phase"] + 2 * np.pi * target["frequency_hz"] * dt_seconds) % (
                2 * np.pi
            )

    def update_marker_styles(self):
        """Highlight the active target while showing all target points."""
        for index, marker in enumerate(self.point_markers):
            if index == self.active_target_index:
                marker.set_color("red")
                marker.set_markersize(5)
            else:
                marker.set_color("gray")
                marker.set_markersize(4)

    def build_target_specs(self):
        """Build current rotating target fields for all points."""
        target_specs = []
        for target in self.targets:
            direction = np.array([np.cos(target["phase"]), np.sin(target["phase"]), 0.0])
            target["target_field"] = self.shared_radius * direction
            target_specs.append(
                {
                    "position": target["position"],
                    "target_field": target["target_field"],
                }
            )
        return target_specs

    def update(self, frame):
        """Animation callback."""
        self.controller.update()

        if self.controller.switch_pressed:
            self.active_target_index = (self.active_target_index + 1) % self.num_points

        self.update_active_target_speed()
        self.move_active_target()

        current_time = time.perf_counter()
        dt_seconds = current_time - self.last_update_time
        self.last_update_time = current_time
        self.update_target_phases(dt_seconds)

        self.shared_radius = multi_point_rotating_radius(build_radius_target_points(self.targets))
        self.currents = optimize_currents_for_targets(self.build_target_specs())

        active_target = self.targets[self.active_target_index]
        active_bx, active_by, active_bz = calculate_b(self.currents, *active_target["position"])
        active_target["actual_field"] = np.array([active_bx, active_by, active_bz])
        active_inplane_error = np.linalg.norm(
            active_target["actual_field"][:2] - active_target["target_field"][:2]
        )

        for target in self.targets:
            bx, by, bz = calculate_b(self.currents, *target["position"])
            target["actual_field"] = np.array([bx, by, bz])

        magnitude, _, _ = calculate_field_magnitude(
            self.currents, self.x_grid, self.y_grid, self.z_grid
        )
        _, bx_arrow, by_arrow = calculate_field_magnitude(
            self.currents, self.x_arrow, self.y_arrow, self.z_arrow
        )

        arrow_magnitude = np.sqrt(bx_arrow**2 + by_arrow**2)
        nonzero_mask = arrow_magnitude > 1e-9
        u_arrow = np.zeros_like(bx_arrow)
        v_arrow = np.zeros_like(by_arrow)
        u_arrow[nonzero_mask] = bx_arrow[nonzero_mask] / arrow_magnitude[nonzero_mask]
        v_arrow[nonzero_mask] = by_arrow[nonzero_mask] / arrow_magnitude[nonzero_mask]

        self.magnitude_plot.set_array(magnitude.ravel())
        self.magnitude_plot.set_clim(0, FIELD_COLORBAR_MAX_TESLA)
        self.quiver_arrows.set_UVC(u_arrow, v_arrow)

        self.update_marker_styles()

        for index, target in enumerate(self.targets):
            position = target["position"]
            actual_field = target["actual_field"]
            actual_inplane_norm = np.linalg.norm(actual_field[:2])
            actual_field_norm = np.linalg.norm(actual_field)
            overlay_color = "red" if index == self.active_target_index else "gray"

            self.point_markers[index].set_data([position[0]], [position[1]])
            self.point_markers[index].set_color(overlay_color)
            self.field_circles[index].center = (position[0], position[1])
            self.direction_arrows[index].set_offsets([position[0], position[1]])
            self.direction_arrows[index].set_color(overlay_color)

            normalized_amplitude = np.clip(
                actual_field_norm / max(FIELD_CIRCLE_REFERENCE_TESLA, 1e-9),
                0.0,
                1.0,
            )

            circle_radius = (
                FIELD_CIRCLE_MIN_RADIUS
                + (FIELD_CIRCLE_MAX_RADIUS - FIELD_CIRCLE_MIN_RADIUS)
                * (normalized_amplitude**FIELD_CIRCLE_EXPONENT)
            )
            self.field_circles[index].set_radius(circle_radius)
            self.field_circles[index].set_edgecolor(overlay_color)

            if actual_inplane_norm > 1e-9:
                arrow_length = circle_radius
                arrow_scale = arrow_length / actual_inplane_norm

                self.direction_arrows[index].set_UVC(
                    actual_field[0] * arrow_scale,
                    actual_field[1] * arrow_scale,
                )
            else:
                self.direction_arrows[index].set_UVC(0.0, 0.0)

            if SHOW_ARROW_MAGNITUDE_LABELS:
                self.magnitude_labels[index].set_position(
                    (position[0] + circle_radius + 0.002, position[1] + circle_radius + 0.002)
                )
                self.magnitude_labels[index].set_text(f"{actual_inplane_norm * 1000:.2f}")
                self.magnitude_labels[index].set_color(overlay_color)
                self.magnitude_labels[index].set_visible(True)
            else:
                self.magnitude_labels[index].set_visible(False)

        active_field_magnitude = np.linalg.norm(active_target["actual_field"])
        active_pos_mm = active_target["position"][:2] * 1000
        speed_summary = ", ".join(
            f"P{index + 1}:{target['frequency_hz']:.1f}Hz" for index, target in enumerate(self.targets)
        )
        info = (
            f"Active point: {self.active_target_index + 1}/{self.num_points}\n"
            f"Position: ({active_pos_mm[0]:.1f}, {active_pos_mm[1]:.1f}) mm\n"
            f"Shared rotating radius: {self.shared_radius * 1000:.3f} mT\n"
            f"Active speed: {active_target['frequency_hz']:.2f} Hz\n"
            f"Active field: ({active_bx * 1000:.3f}, {active_by * 1000:.3f}, {active_bz * 1000:.3f}) mT\n"
            f"Active |B|: {active_field_magnitude * 1000:.3f} mT\n"
            f"Active in-plane error: {active_inplane_error * 1000:.3f} mT\n"
            f"Speeds: {speed_summary}\n"
            f"Max |I|: {np.max(np.abs(self.currents)):.2f} A"
        )
        self.info_text.set_text(info)

        artists = [self.magnitude_plot, self.info_text, self.quiver_arrows]
        artists.extend(self.point_markers)
        artists.extend(self.field_circles)
        artists.extend(self.direction_arrows)
        artists.extend(self.magnitude_labels)
        return artists

    def run(self):
        """Run the simulation."""
        self.animation = FuncAnimation(
            self.fig, self.update, interval=50, blit=False, cache_frame_data=False
        )
        plt.show()
        pygame.quit()


# ===================== Main =====================
if __name__ == "__main__":
    print("=" * 60)
    print("Rotating Magnetic Field Simulation with Multiple Target Points")
    print("=" * 60)
    print("Controls:")
    print("  Right Stick: Move the selected target point")
    print("  Button A:    Switch to the next target point")
    print("  D-pad Up:    Increase the selected point rotation speed")
    print("  D-pad Down:  Decrease the selected point rotation speed")
    print("Keyboard fallback:")
    print("  WASD:        Move the selected target point")
    print("  Tab:         Switch to the next target point")
    print("  Up/Down:     Change the selected point rotation speed")
    print("=" * 60)

    number_of_points = get_num_points()
    simulator = MagneticFieldSimulator(number_of_points)
    simulator.run()
