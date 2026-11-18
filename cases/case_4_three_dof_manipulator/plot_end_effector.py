import argparse
# Run from the repository root: python -m cases.case_4_three_dof_manipulator.plot_end_effector
import json
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


DEFAULT_ALPHA_RANGE = (math.pi / 6.0, 5.0 * math.pi / 6.0)
DEFAULT_BETA_RANGE = (-5.0 * math.pi / 6.0, -math.pi / 6.0)


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


def load_nonfeasible_boxes(result_path):
    with open(result_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data.get("undetermined", [])


def _mask_nonfeasible(alpha_values, beta_values, nonfeasible_boxes):
    if not nonfeasible_boxes:
        return np.ones_like(alpha_values, dtype=bool)

    boxes = np.asarray(nonfeasible_boxes)
    if boxes.size == 0:
        return np.ones_like(alpha_values, dtype=bool)

    alpha_values = np.asarray(alpha_values)
    beta_values = np.asarray(beta_values)
    in_box = (
        (alpha_values[:, None] >= boxes[None, :, 0])
        & (alpha_values[:, None] <= boxes[None, :, 1])
        & (beta_values[:, None] >= boxes[None, :, 2])
        & (beta_values[:, None] <= boxes[None, :, 3])
    )
    return ~np.any(in_box, axis=1)


def generate_feasible_points(alpha_range, beta_range, nonfeasible_boxes, samples_per_axis=50):
    alpha_values = np.linspace(alpha_range[0], alpha_range[1], samples_per_axis)
    beta_values = np.linspace(beta_range[0], beta_range[1], samples_per_axis)
    alpha_mesh, beta_mesh = np.meshgrid(alpha_values, beta_values, indexing="ij")

    alpha_flat = alpha_mesh.ravel()
    beta_flat = beta_mesh.ravel()

    mask = _mask_nonfeasible(alpha_flat, beta_flat, nonfeasible_boxes)
    alpha_flat = alpha_flat[mask]
    beta_flat = beta_flat[mask]

    return robot_arm_kinematics(alpha_flat, beta_flat)


def generate_boundary_curves(alpha_range, beta_range, nonfeasible_boxes, samples=200):
    a_min, a_max = alpha_range
    b_min, b_max = beta_range

    beta_edge = np.linspace(b_min, b_max, samples)
    alpha_edge = np.linspace(a_min, a_max, samples)

    boundary_params = [
        (np.full(samples, a_min), beta_edge),
        (np.full(samples, a_max), beta_edge),
        (alpha_edge, np.full(samples, b_min)),
        (alpha_edge, np.full(samples, b_max)),
    ]

    curves = []
    for alpha_vals, beta_vals in boundary_params:
        mask = _mask_nonfeasible(alpha_vals, beta_vals, nonfeasible_boxes)
        if not np.any(mask):
            continue
        x, y, z = robot_arm_kinematics(alpha_vals[mask], beta_vals[mask])
        curves.append((x, y, z))

    return curves


def generate_full_boundary_curves(alpha_range, beta_range, samples=200):
    """Generate the complete workspace boundary without filtering non-feasible boxes."""
    a_min, a_max = alpha_range
    b_min, b_max = beta_range

    beta_edge = np.linspace(b_min, b_max, samples)
    alpha_edge = np.linspace(a_min, a_max, samples)

    boundary_params = [
        (np.full(samples, a_min), beta_edge),  # alpha = alpha_min
        (np.full(samples, a_max), beta_edge),  # alpha = alpha_max
        (alpha_edge, np.full(samples, b_min)),  # beta = beta_min
        (alpha_edge, np.full(samples, b_max)),  # beta = beta_max
    ]

    curves = []
    for alpha_vals, beta_vals in boundary_params:
        x, y, z = robot_arm_kinematics(alpha_vals, beta_vals)
        curves.append((x, y, z))

    return curves


def generate_meridians(alpha_range, beta_range, num_meridians=8, samples=200):
    """Generate meridian curves (lines of constant beta/azimuthal angle)."""
    a_min, a_max = alpha_range
    b_min, b_max = beta_range

    # Create evenly spaced beta values for meridians
    beta_values = np.linspace(b_min, b_max, num_meridians)
    alpha_values = np.linspace(a_min, a_max, samples)

    curves = []
    for beta in beta_values:
        alpha_vals = alpha_values
        beta_vals = np.full(samples, beta)
        x, y, z = robot_arm_kinematics(alpha_vals, beta_vals)
        curves.append((x, y, z))

    return curves


def generate_parallels(alpha_range, beta_range, num_parallels=8, samples=200):
    """Generate parallel curves (lines of constant alpha/polar angle)."""
    a_min, a_max = alpha_range
    b_min, b_max = beta_range

    # Create evenly spaced alpha values for parallels
    alpha_values = np.linspace(a_min, a_max, num_parallels)
    beta_values = np.linspace(b_min, b_max, samples)

    curves = []
    for alpha in alpha_values:
        alpha_vals = np.full(samples, alpha)
        beta_vals = beta_values
        x, y, z = robot_arm_kinematics(alpha_vals, beta_vals)
        curves.append((x, y, z))

    return curves


def plot_feasible_endpoints(
    result_path,
    alpha_range,
    beta_range,
    samples_per_axis=50,
    plot_interior=True,
    boundary_samples=200,
    feasible_color="tab:green",
    num_meridians=8,
    num_parallels=8,
    plot_meridians=True,
    plot_parallels=True,
    view_elev=30,
    view_azim=45,
):
    boxes = load_nonfeasible_boxes(result_path)

    x_interior = y_interior = z_interior = np.array([])
    if plot_interior:
        x_interior, y_interior, z_interior = generate_feasible_points(
            alpha_range, beta_range, boxes, samples_per_axis
        )

    # Generate full workspace boundary (without filtering non-feasible boxes)
    boundary_curves = generate_full_boundary_curves(alpha_range, beta_range, boundary_samples)

    # Generate meridians and parallels
    meridian_curves = []
    parallel_curves = []
    if plot_meridians:
        meridian_curves = generate_meridians(alpha_range, beta_range, num_meridians, boundary_samples)
    if plot_parallels:
        parallel_curves = generate_parallels(alpha_range, beta_range, num_parallels, boundary_samples)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    # Set the default view angle
    ax.view_init(elev=view_elev, azim=view_azim)

    # Conversion factor: meters to centimeters
    M_TO_CM = 100.0

    plotted_any = False
    legend_elements = []

    if x_interior.size:
        # Convert to cm
        ax.scatter(x_interior * M_TO_CM, y_interior * M_TO_CM, z_interior * M_TO_CM, s=10, color=feasible_color, alpha=0.1, label='Feasible workspace')
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=feasible_color, markersize=10, alpha=0.7, label='Feasible workspace'))
        plotted_any = True

    if boundary_curves:
        for curve in boundary_curves:
            # Convert to cm
            ax.plot(curve[0] * M_TO_CM, curve[1] * M_TO_CM, curve[2] * M_TO_CM, color="black", linewidth=2.0)
        if not legend_elements or legend_elements[-1].get_label() != 'Boundary':
            legend_elements.append(plt.Line2D([0], [0], color='black', linewidth=2.0, label='Boundary'))
        plotted_any = True

    # Plot meridians (lines of constant beta)
    if meridian_curves:
        for curve in meridian_curves:
            # Convert to cm
            ax.plot(curve[0] * M_TO_CM, curve[1] * M_TO_CM, curve[2] * M_TO_CM, color="gray", linewidth=1.0, alpha=1, linestyle='--')
        # if not any(elem.get_label() == 'Meridians' for elem in legend_elements):
        #     legend_elements.append(plt.Line2D([0], [0], color='gray', linewidth=1.0, linestyle='--', label='Meridians'))
        plotted_any = True

    # Plot parallels (lines of constant alpha)
    if parallel_curves:
        for curve in parallel_curves:
            # Convert to cm
            ax.plot(curve[0] * M_TO_CM, curve[1] * M_TO_CM, curve[2] * M_TO_CM, color="gray", linewidth=1.0, alpha=1, linestyle=':')
        # if not any(elem.get_label() == 'Parallels' for elem in legend_elements):
        #     legend_elements.append(plt.Line2D([0], [0], color='gray', linewidth=1.0, linestyle=':', label='Parallels'))
        plotted_any = True

    # Plot base point with cross marker
    x_base = 0.0
    y_base = 0.023
    z_base = 0.079
    # Convert to cm
    ax.scatter([x_base * M_TO_CM], [y_base * M_TO_CM], [z_base * M_TO_CM], marker='x', s=200, color='red', linewidths=3, label='Base point')
    legend_elements.append(plt.Line2D([0], [0], marker='x', color='red', linewidth=3, markersize=12, label='Base point'))
    plotted_any = True

    if not plotted_any:
        ax.text2D(0.5, 0.5, "No feasible points found", transform=ax.transAxes, ha="center")

    ax.set_xlabel("X (cm)", fontsize=20, labelpad=10)
    ax.set_ylabel("Y (cm)", fontsize=20, labelpad=10)
    ax.set_zlabel("Z (cm)", fontsize=20, labelpad=10)
    ax.tick_params(axis="both", which="major", labelsize=18)

    combined_x = np.concatenate([curve[0] for curve in boundary_curves]) if boundary_curves else np.array([])
    combined_y = np.concatenate([curve[1] for curve in boundary_curves]) if boundary_curves else np.array([])
    combined_z = np.concatenate([curve[2] for curve in boundary_curves]) if boundary_curves else np.array([])

    # Include meridians and parallels in aspect ratio calculation
    meridian_x = np.concatenate([curve[0] for curve in meridian_curves]) if meridian_curves else np.array([])
    meridian_y = np.concatenate([curve[1] for curve in meridian_curves]) if meridian_curves else np.array([])
    meridian_z = np.concatenate([curve[2] for curve in meridian_curves]) if meridian_curves else np.array([])

    parallel_x = np.concatenate([curve[0] for curve in parallel_curves]) if parallel_curves else np.array([])
    parallel_y = np.concatenate([curve[1] for curve in parallel_curves]) if parallel_curves else np.array([])
    parallel_z = np.concatenate([curve[2] for curve in parallel_curves]) if parallel_curves else np.array([])

    # Collect all points for aspect ratio calculation (convert to cm)
    arrays_to_concat_x = []
    arrays_to_concat_y = []
    arrays_to_concat_z = []

    arrays_to_concat_x.append(np.array([x_base * M_TO_CM]))
    arrays_to_concat_y.append(np.array([y_base * M_TO_CM]))
    arrays_to_concat_z.append(np.array([z_base * M_TO_CM]))

    if x_interior.size > 0:
        arrays_to_concat_x.append(x_interior * M_TO_CM)
        arrays_to_concat_y.append(y_interior * M_TO_CM)
        arrays_to_concat_z.append(z_interior * M_TO_CM)

    if combined_x.size > 0:
        arrays_to_concat_x.append(combined_x * M_TO_CM)
        arrays_to_concat_y.append(combined_y * M_TO_CM)
        arrays_to_concat_z.append(combined_z * M_TO_CM)

    if meridian_x.size > 0:
        arrays_to_concat_x.append(meridian_x * M_TO_CM)
        arrays_to_concat_y.append(meridian_y * M_TO_CM)
        arrays_to_concat_z.append(meridian_z * M_TO_CM)

    if parallel_x.size > 0:
        arrays_to_concat_x.append(parallel_x * M_TO_CM)
        arrays_to_concat_y.append(parallel_y * M_TO_CM)
        arrays_to_concat_z.append(parallel_z * M_TO_CM)

    all_x = np.concatenate(arrays_to_concat_x) if arrays_to_concat_x else np.array([])
    all_y = np.concatenate(arrays_to_concat_y) if arrays_to_concat_y else np.array([])
    all_z = np.concatenate(arrays_to_concat_z) if arrays_to_concat_z else np.array([])

    _set_equal_aspect(ax, all_x, all_y, all_z)

    # Add legend
    if legend_elements:
        ax.legend(handles=legend_elements, loc='upper right', fontsize=16)
    plt.tight_layout()
    plt.show()


def _set_equal_aspect(ax, x, y, z):
    if x.size == 0:
        return

    ranges = np.array([x.max() - x.min(), y.max() - y.min(), z.max() - z.min()])
    max_range = ranges.max()
    mid_x = (x.max() + x.min()) / 2.0
    mid_y = (y.max() + y.min()) / 2.0
    mid_z = (z.max() + z.min()) / 2.0

    ax.set_xlim(mid_x - max_range / 2.0, mid_x + max_range / 2.0)
    ax.set_ylim(mid_y - max_range / 2.0, mid_y + max_range / 2.0)
    ax.set_zlim(mid_z - max_range / 2.0, mid_z + max_range / 2.0)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot feasible end-effector positions excluding non-feasible boxes."
    )
    parser.add_argument(
        "--result",
        type=Path,
        default=Path("Results") / "results_case4_0503.json",
        help="Path to JSON file containing non-feasible boxes.",
    )
    parser.add_argument("--alpha-min", type=float, default=None, help="Minimum alpha (radians).")
    parser.add_argument("--alpha-max", type=float, default=None, help="Maximum alpha (radians).")
    parser.add_argument("--beta-min", type=float, default=None, help="Minimum beta (radians).")
    parser.add_argument("--beta-max", type=float, default=None, help="Maximum beta (radians).")
    parser.add_argument(
        "--grid-samples",
        type=int,
        default=200,
        help="Samples per axis for the interior grid (default: 200).",
    )
    parser.add_argument(
        "--boundary-samples",
        type=int,
        default=400,
        help="Samples per edge for drawing the boundary (default: 400).",
    )
    parser.add_argument(
        "--no-interior", action="store_true", help="Do not plot interior feasible points."
    )
    parser.add_argument(
        "--color",
        type=str,
        default="tab:green",
        help="Matplotlib color for interior feasible points (default: tab:green).",
    )
    parser.add_argument(
        "--num-meridians",
        type=int,
        default=8,
        help="Number of meridian curves to plot (default: 8).",
    )
    parser.add_argument(
        "--num-parallels",
        type=int,
        default=8,
        help="Number of parallel curves to plot (default: 8).",
    )
    parser.add_argument(
        "--no-meridians",
        action="store_true",
        help="Do not plot meridian curves.",
    )
    parser.add_argument(
        "--no-parallels",
        action="store_true",
        help="Do not plot parallel curves.",
    )
    parser.add_argument(
        "--view-elev",
        type=float,
        default=20,
        help="Elevation angle for 3D view in degrees (default: 30).",
    )
    parser.add_argument(
        "--view-azim",
        type=float,
        default=250,
        help="Azimuth angle for 3D view in degrees (default: 45).",
    )
    return parser.parse_args()


def validate_range(range_tuple, name):
    if range_tuple[0] >= range_tuple[1]:
        raise ValueError(f"{name} minimum must be less than maximum.")


if __name__ == "__main__":
    args = parse_args()
    boxes = load_nonfeasible_boxes(args.result)

    alpha_range = (
        args.alpha_min if args.alpha_min is not None else DEFAULT_ALPHA_RANGE[0],
        args.alpha_max if args.alpha_max is not None else DEFAULT_ALPHA_RANGE[1],
    )
    beta_range = (
        args.beta_min if args.beta_min is not None else DEFAULT_BETA_RANGE[0],
        args.beta_max if args.beta_max is not None else DEFAULT_BETA_RANGE[1],
    )

    validate_range(alpha_range, "Alpha")
    validate_range(beta_range, "Beta")

    plot_feasible_endpoints(
        args.result,
        alpha_range,
        beta_range,
        samples_per_axis=args.grid_samples,
        plot_interior=not args.no_interior,
        boundary_samples=args.boundary_samples,
        feasible_color=args.color,
        num_meridians=args.num_meridians,
        num_parallels=args.num_parallels,
        plot_meridians=not args.no_meridians,
        plot_parallels=not args.no_parallels,
        view_elev=args.view_elev,
        view_azim=args.view_azim,
    )
