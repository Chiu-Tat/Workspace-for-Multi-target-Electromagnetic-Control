import argparse
# Run from the repository root: python -m cases.case_5_two_dof_manipulator.plot_end_effector
import json
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

DEFAULT_BETA1_RANGE = (-5*math.pi / 6.0, -math.pi / 6.0)
DEFAULT_ALPHA2_RANGE = (math.pi / 6.0, 5.0 * math.pi / 6.0)

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

def load_nonfeasible_boxes(result_path):
    with open(result_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data.get("undetermined", [])

def _mask_nonfeasible(beta1_values, alpha2_values, nonfeasible_boxes):
    if not nonfeasible_boxes:
        return np.ones_like(beta1_values, dtype=bool)

    boxes = np.asarray(nonfeasible_boxes)
    if boxes.size == 0:
        return np.ones_like(beta1_values, dtype=bool)

    beta1_values = np.asarray(beta1_values)
    alpha2_values = np.asarray(alpha2_values)
    in_box = (
        (beta1_values[:, None] >= boxes[None, :, 0])
        & (beta1_values[:, None] <= boxes[None, :, 1])
        & (alpha2_values[:, None] >= boxes[None, :, 2])
        & (alpha2_values[:, None] <= boxes[None, :, 3])
    )
    return ~np.any(in_box, axis=1)


def generate_feasible_points(beta1_range, alpha2_range, nonfeasible_boxes, samples_per_axis=50):
    beta1_values = np.linspace(beta1_range[0], beta1_range[1], samples_per_axis)
    alpha2_values = np.linspace(alpha2_range[0], alpha2_range[1], samples_per_axis)
    beta1_mesh, alpha2_mesh = np.meshgrid(beta1_values, alpha2_values, indexing="ij")

    beta1_flat = beta1_mesh.ravel()
    alpha2_flat = alpha2_mesh.ravel()

    mask = _mask_nonfeasible(beta1_flat, alpha2_flat, nonfeasible_boxes)
    beta1_flat = beta1_flat[mask]
    alpha2_flat = alpha2_flat[mask]

    # Return p2 positions only
    x, y, z = arm_end_positions(beta1_flat, alpha2_flat)
    return x, y, z


def generate_boundary_curves(beta1_range, alpha2_range, nonfeasible_boxes, samples=200):
    b1_min, b1_max = beta1_range
    a2_min, a2_max = alpha2_range

    beta1_edge = np.linspace(b1_min, b1_max, samples)
    alpha2_edge = np.linspace(a2_min, a2_max, samples)

    boundary_params = [
        (np.full(samples, b1_min), alpha2_edge),
        (np.full(samples, b1_max), alpha2_edge),
        (beta1_edge, np.full(samples, a2_min)),
        (beta1_edge, np.full(samples, a2_max)),
    ]

    curves = []
    for beta1_vals, alpha2_vals in boundary_params:
        mask = _mask_nonfeasible(beta1_vals, alpha2_vals, nonfeasible_boxes)
        if not np.any(mask):
            continue
        x, y, z = arm_end_positions(beta1_vals[mask], alpha2_vals[mask])
        curves.append((x, y, z))

    return curves


def generate_full_boundary_curves(beta1_range, alpha2_range, samples=200):
    """Generate the complete workspace boundary without filtering non-feasible boxes."""
    b1_min, b1_max = beta1_range
    a2_min, a2_max = alpha2_range

    beta1_edge = np.linspace(b1_min, b1_max, samples)
    alpha2_edge = np.linspace(a2_min, a2_max, samples)

    boundary_params = [
        (np.full(samples, b1_min), alpha2_edge),  # beta1 = beta1_min
        (np.full(samples, b1_max), alpha2_edge),  # beta1 = beta1_max
        (beta1_edge, np.full(samples, a2_min)),   # alpha2 = alpha2_min
        (beta1_edge, np.full(samples, a2_max)),   # alpha2 = alpha2_max
    ]

    curves = []
    for beta1_vals, alpha2_vals in boundary_params:
        x, y, z = arm_end_positions(beta1_vals, alpha2_vals)
        curves.append((x, y, z))

    return curves


def generate_meridians(beta1_range, alpha2_range, num_meridians=8, samples=200):
    """Generate meridian curves (lines of constant alpha2)."""
    b1_min, b1_max = beta1_range
    a2_min, a2_max = alpha2_range

    # Create evenly spaced alpha2 values for meridians
    alpha2_values = np.linspace(a2_min, a2_max, num_meridians)
    beta1_values = np.linspace(b1_min, b1_max, samples)

    curves = []
    for alpha2 in alpha2_values:
        beta1_vals = beta1_values
        alpha2_vals = np.full(samples, alpha2)
        x, y, z = arm_end_positions(beta1_vals, alpha2_vals)
        curves.append((x, y, z))

    return curves


def generate_parallels(beta1_range, alpha2_range, num_parallels=8, samples=200):
    """Generate parallel curves (lines of constant beta1)."""
    b1_min, b1_max = beta1_range
    a2_min, a2_max = alpha2_range

    # Create evenly spaced beta1 values for parallels
    beta1_values = np.linspace(b1_min, b1_max, num_parallels)
    alpha2_values = np.linspace(a2_min, a2_max, samples)

    curves = []
    for beta1 in beta1_values:
        beta1_vals = np.full(samples, beta1)
        alpha2_vals = alpha2_values
        x, y, z = arm_end_positions(beta1_vals, alpha2_vals)
        curves.append((x, y, z))

    return curves


def plot_feasible_endpoints(
    result_path,
    beta1_range,
    alpha2_range,
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
            beta1_range, alpha2_range, boxes, samples_per_axis
        )

    # Generate full workspace boundary (without filtering non-feasible boxes)
    boundary_curves = generate_full_boundary_curves(beta1_range, alpha2_range, boundary_samples)

    # Generate meridians and parallels
    meridian_curves = []
    parallel_curves = []
    if plot_meridians:
        meridian_curves = generate_meridians(beta1_range, alpha2_range, num_meridians, boundary_samples)
    if plot_parallels:
        parallel_curves = generate_parallels(beta1_range, alpha2_range, num_parallels, boundary_samples)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    # Set the default view angle
    ax.view_init(elev=view_elev, azim=view_azim)

    # Conversion factor: meters to centimeters
    M_TO_CM = 100.0

    plotted_any = False
    legend_elements = []

    # Plot interior feasible points
    if x_interior.size:
        # Convert to cm
        ax.scatter(x_interior * M_TO_CM, y_interior * M_TO_CM, z_interior * M_TO_CM,
                   s=10, color=feasible_color, alpha=0.1, label='Feasible workspace')
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=feasible_color,
                                         markersize=10, alpha=0.7, label='Feasible workspace'))
        plotted_any = True

    # Plot boundaries
    if boundary_curves:
        for curve in boundary_curves:
            # Convert to cm
            ax.plot(curve[0] * M_TO_CM, curve[1] * M_TO_CM, curve[2] * M_TO_CM, color="black", linewidth=2.0)
        if not any(elem.get_label() == 'Boundary' for elem in legend_elements):
            legend_elements.append(plt.Line2D([0], [0], color='black', linewidth=2.0, label='Boundary'))
        plotted_any = True

    # Plot meridians (lines of constant alpha2)
    if meridian_curves:
        for curve in meridian_curves:
            # Convert to cm
            ax.plot(curve[0] * M_TO_CM, curve[1] * M_TO_CM, curve[2] * M_TO_CM,
                    color="gray", linewidth=1.0, alpha=1, linestyle='--')
        plotted_any = True

    # Plot parallels (lines of constant beta1)
    if parallel_curves:
        for curve in parallel_curves:
            # Convert to cm
            ax.plot(curve[0] * M_TO_CM, curve[1] * M_TO_CM, curve[2] * M_TO_CM,
                    color="gray", linewidth=1.0, alpha=1, linestyle=':')
        plotted_any = True

    # Plot base point with cross marker
    x_base = 0.0
    y_base = 0.023
    z_base = 0.079
    # Convert to cm
    # ax.scatter([x_base * M_TO_CM], [y_base * M_TO_CM], [z_base * M_TO_CM],
    #            marker='x', s=200, color='red', linewidths=3, label='Base point')
    # legend_elements.append(plt.Line2D([0], [0], marker='x', color='red', linewidth=3,
    #                                  markersize=12, label='Base point'))
    plotted_any = True

    if not plotted_any:
        ax.text2D(0.5, 0.5, "No feasible points found", transform=ax.transAxes, ha="center")

    ax.set_xlabel("X (cm)", fontsize=20, labelpad=10)
    ax.set_ylabel("Y (cm)", fontsize=20, labelpad=10)
    ax.set_zlabel("Z (cm)", fontsize=20, labelpad=10)
    ax.tick_params(axis="both", which="major", labelsize=18)

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

    if boundary_curves:
        combined_x = np.concatenate([curve[0] for curve in boundary_curves])
        combined_y = np.concatenate([curve[1] for curve in boundary_curves])
        combined_z = np.concatenate([curve[2] for curve in boundary_curves])
        arrays_to_concat_x.append(combined_x * M_TO_CM)
        arrays_to_concat_y.append(combined_y * M_TO_CM)
        arrays_to_concat_z.append(combined_z * M_TO_CM)

    if meridian_curves:
        meridian_x = np.concatenate([curve[0] for curve in meridian_curves])
        meridian_y = np.concatenate([curve[1] for curve in meridian_curves])
        meridian_z = np.concatenate([curve[2] for curve in meridian_curves])
        arrays_to_concat_x.append(meridian_x * M_TO_CM)
        arrays_to_concat_y.append(meridian_y * M_TO_CM)
        arrays_to_concat_z.append(meridian_z * M_TO_CM)

    if parallel_curves:
        parallel_x = np.concatenate([curve[0] for curve in parallel_curves])
        parallel_y = np.concatenate([curve[1] for curve in parallel_curves])
        parallel_z = np.concatenate([curve[2] for curve in parallel_curves])
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
    import argparse
    parser = argparse.ArgumentParser(
        description="Plot feasible end-effector positions for Case 5 excluding non-feasible boxes."
    )
    parser.add_argument(
        "--result",
        type=Path,
        default=Path("Results") / "results_case5_0203.json",
        help="Path to JSON file containing non-feasible boxes.",
    )
    parser.add_argument("--beta1-min", type=float, default=None, help="Minimum beta1 (radians).")
    parser.add_argument("--beta1-max", type=float, default=None, help="Maximum beta1 (radians).")
    parser.add_argument("--alpha2-min", type=float, default=None, help="Minimum alpha2 (radians).")
    parser.add_argument("--alpha2-max", type=float, default=None, help="Maximum alpha2 (radians).")
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
        help="Matplotlib color for feasible points (default: tab:green).",
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
        help="Elevation angle for 3D view in degrees (default: 20).",
    )
    parser.add_argument(
        "--view-azim",
        type=float,
        default=250,
        help="Azimuth angle for 3D view in degrees (default: 250).",
    )
    return parser.parse_args()


def validate_range(range_tuple, name):
    if range_tuple[0] >= range_tuple[1]:
        raise ValueError(f"{name} minimum must be less than maximum.")


if __name__ == "__main__":
    args = parse_args()
    boxes = load_nonfeasible_boxes(args.result)

    beta1_range = (
        args.beta1_min if args.beta1_min is not None else DEFAULT_BETA1_RANGE[0],
        args.beta1_max if args.beta1_max is not None else DEFAULT_BETA1_RANGE[1],
    )
    alpha2_range = (
        args.alpha2_min if args.alpha2_min is not None else DEFAULT_ALPHA2_RANGE[0],
        args.alpha2_max if args.alpha2_max is not None else DEFAULT_ALPHA2_RANGE[1],
    )

    validate_range(beta1_range, "Beta1")
    validate_range(alpha2_range, "Alpha2")

    plot_feasible_endpoints(
        args.result,
        beta1_range,
        alpha2_range,
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
