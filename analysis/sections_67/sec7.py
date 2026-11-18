"""Section 7 studies: AFW geometry, task sweeps, coil subsets, dipole reuse."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from matplotlib import patheffects
from matplotlib import pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import to_rgb
from matplotlib.patches import Rectangle

from analysis.sections_67 import afw_tools
from analysis.sections_67.common import (
    add_region_legend,
    ANISOTROPIC_TASK_T,
    CLASS_COLORS,
    CLASS_HATCHES,
    COIL_MODEL_Z_M,
    CURRENT_LIMIT_A,
    DIPOLE_RESULTS,
    SEARCH_BOX_M,
    TASK_SWEEP_CSV,
    coil_positions_m,
    configure_matplotlib,
    copy_to_paper,
    dump_json,
    report_to_metrics,
)
from analysis.sections_67.sec6 import AFW_CASES, _anisotropic_problem, _draw_region
from hybrid_region import RegionProblem, certify_region
from hybrid_region.boxes import box_bounds


TASK_CASES = (
    ("symmetric_60mT", 0.060, 0.060),
    ("symmetric_80mT", 0.080, 0.080),
    ("symmetric_100mT", 0.100, 0.100),
    ("symmetric_120mT", 0.120, 0.120),
    ("anisotropic_120_40mT", 0.120, 0.040),
    ("anisotropic_40_120mT", 0.040, 0.120),
)


def _coil_geometry_subsets() -> dict:
    positions = coil_positions_m()
    z = positions[:, 2]
    xy_radius = np.hypot(positions[:, 0], positions[:, 1])
    upper = tuple(int(i) for i in np.where(z > 0.05)[0])
    mid = tuple(int(i) for i in np.where(np.abs(z) < 0.01)[0])
    lower = tuple(int(i) for i in np.where(z < -0.02)[0])
    mid_closest = int(mid[int(np.argmin(xy_radius[list(mid)]))])
    nested_8 = tuple(i for i in range(10) if i not in lower + (mid_closest,))
    nested_6 = tuple(i for i in nested_8 if i not in mid)
    upper_x = np.abs(positions[list(upper), 0])
    opposite_upper = tuple(sorted(int(upper[i]) for i in np.argsort(-upper_x)[:2]))
    alt_8 = tuple(i for i in range(10) if i not in opposite_upper)
    return {
        "rule": (
            "Upper ring: z>0.05 m; mid plane: |z|<0.01 m; lower: z<-0.02 m. "
            "Nested 8-coil set drops the unique lower coil and the mid-plane coil "
            "closest to the z-axis. Nested 6-coil set additionally drops remaining "
            "mid-plane coils. Alternate 8-coil set drops the two upper-ring coils "
            "with largest |x|."
        ),
        "positions_m": positions.tolist(),
        "clusters": {"upper": upper, "mid": mid, "lower": lower},
        "subsets": {
            "all_10": tuple(range(10)),
            "nested_8": nested_8,
            "nested_6": nested_6,
            "alternate_8": alt_8,
        },
        "dropped": {
            "nested_8": lower + (mid_closest,),
            "nested_6": tuple(i for i in range(10) if i not in nested_6),
            "alternate_8": opposite_upper,
        },
    }


def run_7_1(paths: dict[str, Path]) -> dict:
    configure_matplotlib(paths["root"] / ".mplconfig")
    payloads = []
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2))

    for index, spec in enumerate(AFW_CASES):
        generators, offset, zero_offset = afw_tools.interaction_offset_only(spec.points_coil())
        radii = np.full(generators.shape[1], CURRENT_LIMIT_A)
        verts_on = afw_tools.zonotope_vertices_2d(generators, offset, radii)
        verts_off = afw_tools.zonotope_vertices_2d(generators, zero_offset, radii)
        translation = offset - zero_offset
        reconstructed = verts_off + translation
        translation_error = float(np.max(np.linalg.norm(verts_on - reconstructed, axis=1)))
        payloads.append(
            {
                "name": spec.name,
                "points_coil": spec.points_coil(),
                "offset_on_T": offset.tolist(),
                "offset_off_T": zero_offset.tolist(),
                "translation_T": translation.tolist(),
                "full_set_translation_max_error_T": translation_error,
            }
        )
        axis = axes[index]
        axis.fill(verts_off[:, 0], verts_off[:, 1], color="#bbbbbb", alpha=0.45, label="interaction off")
        axis.fill(verts_on[:, 0], verts_on[:, 1], color="#FD7A00" if index else "#9DC3E6", alpha=0.45, label="interaction on")
        axis.plot([zero_offset[0], offset[0]], [zero_offset[1], offset[1]], "k--", lw=0.8)
        axis.set_aspect("equal")
        axis.set_xlabel(r"$b_{x,1}$ (T)")
        axis.set_ylabel(r"$b_{x,2}$ (T)")
        axis.set_title(spec.name.replace("_", " "))
        axis.legend(fontsize=6)

        axis.plot(0, 0, "k+", ms=7, mew=1.5)

    commands = np.array([[0.0, 0.0], [0.05, 0.05], [-0.04, 0.02]])
    spec = AFW_CASES[1]
    generators, offset, zero_offset = afw_tools.interaction_offset_only(spec.points_coil())
    command_rows = []
    for command in commands:
        on = afw_tools.current_feasibility(command, generators, offset, CURRENT_LIMIT_A, 1e-7)
        off = afw_tools.current_feasibility(command, generators, zero_offset, CURRENT_LIMIT_A, 1e-7)
        command_rows.append(
            {
                "command_T": command.tolist(),
                "feasible_on": on["feasible"],
                "feasible_off": off["feasible"],
                "residual_on": on["residual_inf"],
                "residual_off": off["residual_inf"],
            }
        )
    for axis in axes:
        axis.set_xlim(-0.18, 0.18)
        axis.set_ylim(-0.16, 0.16)
    fig.tight_layout()
    figure_path = paths["figures"] / "afw_geometry_interaction.pdf"
    fig.savefig(figure_path)
    fig.savefig(figure_path.with_suffix(".png"), dpi=300)
    plt.close(fig)
    copy_to_paper(figure_path)
    dump_json(
        paths["data"] / "afw_geometry.json",
        {"cases": payloads, "commands": command_rows},
    )
    return {"cases": payloads, "commands": command_rows}


def run_7_2(paths: dict[str, Path]) -> dict:
    configure_matplotlib(paths["root"] / ".mplconfig")
    metrics = []
    fig, axes = plt.subplots(2, 3, figsize=(7.4, 4.8))
    for axis, (name, bx, by) in zip(axes.ravel(), TASK_CASES):
        problem = RegionProblem(
            initial_box=SEARCH_BOX_M,
            z_offset=COIL_MODEL_Z_M,
            task_box=((-bx, bx), (-by, by)),
            current_bounds=(-CURRENT_LIMIT_A, CURRENT_LIMIT_A),
        )
        report = certify_region(problem, mode="affine_hybrid", eps=0.002, keep_boxes=True)
        payload = report_to_metrics(report, {"case": name, "bx": bx, "by": by})
        metrics.append(payload)
        _draw_region(axis, report, f"$B_x$: ±{bx*1000:.0f} mT; $B_y$: ±{by*1000:.0f} mT\nTFW-in {100*payload['area_fractions']['inside']:.2f}%")
    add_region_legend(fig)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    figure_path = paths["figures"] / "task_requirement_tfw.pdf"
    fig.savefig(figure_path)
    fig.savefig(figure_path.with_suffix(".png"), dpi=300)
    plt.close(fig)
    copy_to_paper(figure_path)
    copy_to_paper(figure_path.with_suffix(".png"))
    dump_json(paths["data"] / "task_requirement_hybrid.json", {"metrics": metrics})

    if TASK_SWEEP_CSV.exists():
        with TASK_SWEEP_CSV.open(newline="", encoding="utf-8") as handle:
            archived = list(csv.DictReader(handle))
        dump_json(paths["data"] / "task_requirement_archived_all_pipelines.json", {"rows": archived})
    return {"metrics": metrics}


def _pointwise_tfw_inclusion(problem_small, problem_large, n=21) -> dict:
    return {"n": n, "note": "nested inclusion is reported by _nested_pointwise_checks"}


def run_7_3(paths: dict[str, Path]) -> dict:
    configure_matplotlib(paths["root"] / ".mplconfig")
    geometry = _coil_geometry_subsets()
    dump_json(paths["configs"] / "coil_subsets.json", geometry)
    subsets = geometry["subsets"]
    tfw_metrics = []
    fig, axes = plt.subplots(2, 4, figsize=(7.4, 4.1))

    subset_items = [
        ("all_10", subsets["all_10"]),
        ("nested_8", subsets["nested_8"]),
        ("alternate_8", subsets["alternate_8"]),
        ("nested_6", subsets["nested_6"]),
    ]
    for axis, (name, active) in zip(axes[0], subset_items):
        _plot_coils(axis, active, name)

    for axis, (name, active) in zip(axes[1], subset_items):
        problem = _anisotropic_problem(active=active)
        report = certify_region(problem, mode="affine_hybrid", eps=0.002, keep_boxes=True)
        payload = report_to_metrics(report, {"subset": name, "active_1based": [i + 1 for i in active]})
        tfw_metrics.append(payload)
        _draw_region(axis, report, f"{name.replace('_', ' ')}:\nTFW-in {100*payload['area_fractions']['inside']:.2f}%")

    current_metrics = []
    current_fig, current_axes = plt.subplots(1, 3, figsize=(7.2, 2.5))
    for axis, limit in zip(current_axes, (5.0, 10.0, 15.0)):
        problem = _anisotropic_problem(current_limit=limit)
        report = certify_region(problem, mode="affine_hybrid", eps=0.002, keep_boxes=True)
        payload = report_to_metrics(report, {"current_limit_A": limit})
        current_metrics.append(payload)
        _draw_region(axis, report, rf"$\pm{limit:.0f}$ A; in {100*payload['area_fractions']['inside']:.2f}%")
    add_region_legend(current_fig)
    current_fig.tight_layout(rect=(0, 0.12, 1, 1))
    current_path = paths["figures"] / "current_limit_tfw.pdf"
    current_fig.savefig(current_path)
    current_fig.savefig(current_path.with_suffix(".png"), dpi=300)
    plt.close(current_fig)

    add_region_legend(fig)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    figure_path = paths["figures"] / "coil_subset_tfw.pdf"
    fig.savefig(figure_path)
    fig.savefig(figure_path.with_suffix(".png"), dpi=300)
    plt.close(fig)
    copy_to_paper(figure_path)
    copy_to_paper(figure_path.with_suffix(".png"))
    copy_to_paper(current_path)
    copy_to_paper(current_path.with_suffix(".png"))

    # Pointwise nested-set checks at a coarse grid.
    inclusion = _nested_pointwise_checks(subsets)
    dump_json(
        paths["data"] / "coil_configuration.json",
        {
            "geometry": geometry,
            "tfw": tfw_metrics,
            "current_limits": current_metrics,
            "inclusion_checks": inclusion,
        },
    )
    return {"tfw": tfw_metrics, "current_limits": current_metrics, "inclusion": inclusion}


# Nominal coil poses (unit axis pointing toward the workspace, core-top position
# in mm), from Modeling/Consistency/coil pose.txt. Numbering matches the model.
COIL_POSES_MM = (
    ((-1.0, 0.0, 0.0), (85.0, 0.0, 0.0)),
    ((-0.5, 0.866025, 0.0), (42.5, -73.612, 0.0)),
    ((0.5, 0.866025, 0.0), (-42.5, -73.612, 0.0)),
    ((1.0, 0.0, 0.0), (-85.0, 0.0, 0.0)),
    ((0.5, -0.866025, 0.0), (-42.5, 73.612, 0.0)),
    ((-0.5, -0.866025, 0.0), (42.5, 73.612, 0.0)),
    ((0.0, 0.766045, 0.642787), (0.0, -65.1138, -54.6369)),
    ((0.663414, -0.383022, 0.642787), (-56.3902, 32.5569, -54.6369)),
    ((-0.663414, -0.383022, 0.642787), (56.3902, 32.5569, -54.6369)),
    ((0.0, 0.0, 1.0), (0.0, 0.0, -85.0)),
)
# Shape parameters from Modeling/Consistency/plot_coil_array.py (mm).
CORE_RADIUS_MM = 22.5
COIL_OUTER_RADIUS_MM = 52.5
CORE_LENGTH_MM = 180.0
COIL_LENGTH_MM = 120.0
COIL_OFFSET_FROM_TOP_MM = 20.0


def _cylinder_faces(start, end, radius, n_theta=40):
    axis = (end - start) / np.linalg.norm(end - start)
    trial = np.array([0.0, 0.0, 1.0]) if abs(axis[2]) < 0.95 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(axis, trial)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(axis, e1)
    theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    radial = np.cos(theta)[:, None] * e1 + np.sin(theta)[:, None] * e2
    faces = []
    for i in range(n_theta):
        j = (i + 1) % n_theta
        quad = np.array([start + radius * radial[i], start + radius * radial[j],
                         end + radius * radial[j], end + radius * radial[i]])
        normal = radial[i] + radial[j]
        faces.append((quad, normal / np.linalg.norm(normal)))
    faces.append((start + radius * radial, -axis))
    faces.append((end + radius * radial, axis))
    return faces


def _plot_coils(axis, active, title: str) -> None:
    """Top view (along -z) of the coil bodies; disabled coils are grey."""
    view = np.array([0.0, 0.0, 1.0])
    light = np.array([-0.3, 0.6, 1.0])
    light /= np.linalg.norm(light)
    active_set = set(active)
    polygons, colours, depths, labels = [], [], [], []
    for index, (direction, top) in enumerate(COIL_POSES_MM):
        direction = np.asarray(direction) / np.linalg.norm(direction)
        top = np.asarray(top)
        head = top - COIL_OFFSET_FROM_TOP_MM * direction
        back = head - COIL_LENGTH_MM * direction
        tail = top - CORE_LENGTH_MM * direction
        on = index in active_set
        core_rgb = np.array(to_rgb("#c9c6bb" if on else "#ececec"))
        coil_rgb = np.array(to_rgb("#b86b45" if on else "#dcdcdc"))
        segments = ((top, head, CORE_RADIUS_MM, core_rgb),
                    (back, tail, CORE_RADIUS_MM, core_rgb),
                    (head, back, COIL_OUTER_RADIUS_MM, coil_rgb))
        for start, end, radius, rgb in segments:
            for points, normal in _cylinder_faces(start, end, radius):
                if normal @ view <= 0.0:
                    continue
                polygons.append(points[:, :2])
                depths.append(points[:, 2].mean())
                colours.append(np.clip(rgb * (0.55 + 0.45 * max(normal @ light, 0.0)), 0.0, 1.0))
        labels.append((index, tail - 18.0 * direction, on))
    order = np.argsort(depths)
    axis.add_collection(PolyCollection([polygons[i] for i in order],
                                       facecolors=[colours[i] for i in order],
                                       edgecolors="face", linewidths=0.2))
    for index, position, on in labels:
        axis.text(position[0], position[1], str(index + 1), fontsize=6, ha="center", va="center",
                  color="#202124" if on else "#9a9a9a", zorder=5,
                  path_effects=[patheffects.withStroke(linewidth=1.5, foreground="white")])
    axis.set_aspect("equal")
    axis.set_xlabel("$x$ (mm)")
    axis.set_ylabel("$y$ (mm)")
    axis.set_title(title.replace("_", " "), fontsize=8)
    axis.set_xlim(-300, 300)
    axis.set_ylim(-300, 300)


def _nested_pointwise_checks(subsets: dict, n: int = 11) -> dict:
    xs = np.linspace(-0.05, 0.05, n)
    ys = np.linspace(-0.05, 0.05, n)
    task_vertices = RegionProblem().task_vertices
    counts = {"checked": 0, "violations_8_in_10": 0, "violations_6_in_8": 0, "violations_current": 0, "unresolved_pose_checks": 0, "records": []}

    def pose_feasible(x, y, active, current_limit):
        target = [
            {
                "X": float(x), "Y": float(y), "Z": COIL_MODEL_Z_M, "m": 0.0,
                "alpha": 0.0, "beta": 0.0,
                "Bx": True, "By": True, "Bz": None,
                "Bx_dx": None, "Bx_dy": None, "Bx_dz": None, "By_dy": None, "By_dz": None,
                "fx": None, "fy": None, "fz": None, "tx": None, "ty": None, "tz": None,
            }
        ]
        generators, offset = afw_tools.actuation_map(target, active)
        for vertex in task_vertices:
            result = afw_tools.current_feasibility(vertex, generators, offset, current_limit, 1e-6)
            if result["feasible"] is None:
                return None
            if result["feasible"] is False:
                return False
        return True

    for x in xs:
        for y in ys:
            counts["checked"] += 1
            ok10 = pose_feasible(x, y, subsets["all_10"], 15.0)
            ok8 = pose_feasible(x, y, subsets["nested_8"], 15.0)
            ok6 = pose_feasible(x, y, subsets["nested_6"], 15.0)
            ok10_small = pose_feasible(x, y, subsets["all_10"], 10.0)
            statuses = [ok10, ok8, ok6, ok10_small]
            counts["unresolved_pose_checks"] += sum(value is None for value in statuses)
            counts["records"].append({"x_m": float(x), "y_m": float(y),
                                      "all10_15A": ok10, "nested8_15A": ok8,
                                      "nested6_15A": ok6, "all10_10A": ok10_small})
            if ok8 is True and ok10 is False:
                counts["violations_8_in_10"] += 1
            if ok6 is True and ok8 is False:
                counts["violations_6_in_8"] += 1
            if ok10_small is True and ok10 is False:
                counts["violations_current"] += 1
    return counts


def run_7_4(paths: dict[str, Path]) -> dict:
    metrics = json.loads((DIPOLE_RESULTS / "metrics.json").read_text(encoding="utf-8"))
    orientation = json.loads((DIPOLE_RESULTS / "orientation_metrics.json").read_text(encoding="utf-8"))
    derivatives = json.loads((DIPOLE_RESULTS / "derivative_checks.json").read_text(encoding="utf-8"))
    convergence = json.loads((DIPOLE_RESULTS / "reference_convergence.json").read_text(encoding="utf-8"))
    dump_json(paths["data"] / "dipole_metrics.json", metrics)
    dump_json(paths["data"] / "dipole_orientation.json", orientation)
    dump_json(paths["validation"] / "dipole_derivatives.json", derivatives)
    dump_json(paths["validation"] / "dipole_convergence.json", convergence)
    for name in (
        "dipole_error_tfw_comparison.pdf",
        "dipole_error_tfw_comparison.png",
        "nominal_vs_robust_tfw.pdf",
        "nominal_vs_robust_tfw.png",
    ):
        source = DIPOLE_RESULTS / name
        if source.exists():
            dest = paths["figures"] / name
            dest.write_bytes(source.read_bytes())
            copy_to_paper(dest)
    return {
        "valid_points": metrics["grid"]["valid_points"],
        "nominal_feasible": metrics["nominal_feasible_points"],
        "robust_feasible": metrics["robust_feasible_points"],
        "retained_fraction": metrics["retained_fraction"],
        "sanity": metrics["sanity"],
        "headings": [
            {
                "deg": row["heading_deg"],
                "retained_fraction": row["retained_fraction"],
                "nominal": row["nominal"],
                "robust": row["robust"],
            }
            for row in orientation["summaries"]
        ],
        "max_derivative_relative_error": max(item["relative_error"] for item in derivatives["checks"]),
    }
