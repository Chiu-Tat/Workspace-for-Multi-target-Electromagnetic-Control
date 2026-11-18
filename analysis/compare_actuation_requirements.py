#!/usr/bin/env python3
# Run from the repository root: python -m analysis.compare_actuation_requirements
"""Compare region classifiers under several magnetic-field requirements.

The geometry and numerical protocol match the ten-coil planar example used for
Fig. 12 of the associated manuscript.  Only the required (Bx, By) field box is
changed.  The script produces a spatial comparison grid, compact metric plots,
and JSON/CSV source data under ``Results/actuation_requirement_comparison``.

The labels are deliberately ``TFW-in``, ``TFW-out``, and ``unresolved``.
They become mathematical certificates only when the actuation enclosure is
inclusion-preserving and outward-rounded.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

from analysis.affine.ten_coil import (
    BisectBox,
    Create2DBox,
    calculate_affine_A,
    calculate_interval_A,
)
from emns.workspace.checks import Out, RobustFeasible
from emns.numerics.interval import interval
from emns.certificates.tight_region import (
    affine_constructive_inside_certificate,
    box_vertices,
    constructive_inside_certificate,
    dual_outside_certificate,
    interval_object_bounds,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = PROJECT_ROOT / "Results" / "actuation_requirement_comparison"
MODES = ("interval_original", "affine_original", "affine_recourse")
MODE_LABELS = {
    "interval_original": "Standard interval",
    "affine_original": "Affine",
    "affine_recourse": "Affine + recourse",
}
CLASSIFICATIONS = ("inside", "outside", "unresolved")
CLASS_LABELS = {
    "inside": "TFW-in",
    "outside": "TFW-out",
    "unresolved": "Unresolved",
}
CLASS_COLORS = {
    "inside": "#9bcfa8",
    "outside": "#e6a2a2",
    "unresolved": "#e5e5e5",
}
CLASS_HATCHES = {"inside": "", "outside": "xx", "unresolved": "...."}
MODE_COLORS = {
    "interval_original": "#b8b8b8",
    "affine_original": "#86a8cf",
    "affine_recourse": "#4d8f71",
}


@dataclass(frozen=True)
class RequirementCase:
    key: str
    bx_half_width_t: float
    by_half_width_t: float
    description: str

    @property
    def lower(self) -> np.ndarray:
        return np.array([-self.bx_half_width_t, -self.by_half_width_t])

    @property
    def upper(self) -> np.ndarray:
        return np.array([self.bx_half_width_t, self.by_half_width_t])

    @property
    def compact_label(self) -> str:
        return f"{self.bx_half_width_t * 1000:.0f}/{self.by_half_width_t * 1000:.0f}"

    @property
    def row_label(self) -> str:
        return (
            rf"$|B_x|\leq {self.bx_half_width_t * 1000:.0f}$ mT"
            "\n"
            rf"$|B_y|\leq {self.by_half_width_t * 1000:.0f}$ mT"
        )


CASES = (
    RequirementCase("symmetric_60mT", 0.060, 0.060, "easy symmetric task"),
    RequirementCase("symmetric_80mT", 0.080, 0.080, "nominal Fig. 12 task"),
    RequirementCase("symmetric_100mT", 0.100, 0.100, "mixed-capacity symmetric task"),
    RequirementCase("symmetric_120mT", 0.120, 0.120, "beyond-capacity symmetric task"),
    RequirementCase("anisotropic_120_40mT", 0.120, 0.040, "Bx-dominant task"),
    RequirementCase("anisotropic_40_120mT", 0.040, 0.120, "By-dominant task"),
)


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "axes.linewidth": 0.75,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def box_area(box) -> float:
    return float(
        (box[0][0].upper - box[0][0].lower)
        * (box[1][0].upper - box[1][0].lower)
    )


def classify_box(
    box,
    mode: str,
    z_offset: float,
    task_vertices: np.ndarray,
    current_lower: np.ndarray,
    current_upper: np.ndarray,
    current_intervals,
    task_intervals,
) -> tuple[str, str]:
    if mode == "interval_original":
        actuation_interval = calculate_interval_A(
            box[0][0], box[1][0], z_offset
        )
    else:
        actuation_interval = calculate_affine_A(
            box[0][0], box[1][0], z_offset
        )

    if RobustFeasible(actuation_interval, current_intervals, task_intervals):
        return "inside", "rohn"

    lower = None
    upper = None
    if mode == "affine_recourse":
        actuation_affine = calculate_affine_A(
            box[0][0], box[1][0], z_offset, return_affine=True
        )
        certificate = affine_constructive_inside_certificate(
            actuation_affine,
            task_vertices,
            current_lower,
            current_upper,
        )
        if certificate.certified:
            return "inside", "affine_right_inverse"

        lower, upper = interval_object_bounds(actuation_interval)
        certificate = constructive_inside_certificate(
            lower,
            upper,
            task_vertices,
            current_lower,
            current_upper,
        )
        if certificate.certified:
            return "inside", "interval_right_inverse"

    if Out(actuation_interval, current_intervals, task_intervals):
        return "outside", "contractor"

    if mode == "affine_recourse":
        if lower is None:
            lower, upper = interval_object_bounds(actuation_interval)
        certificate = dual_outside_certificate(
            lower,
            upper,
            task_vertices,
            current_lower,
            current_upper,
        )
        if certificate.certified:
            return "outside", "dual_separation"

    return "unresolved", "none"


def run_case(case: RequirementCase, mode: str, eps: float) -> dict:
    initial = Create2DBox(interval[-0.05, 0.05], interval[-0.05, 0.05])
    z_offset = 0.075
    current_lower = np.full(10, -15.0)
    current_upper = np.full(10, 15.0)
    current_intervals = [interval[-15.0, 15.0]] * 10
    task_intervals = [
        interval[case.lower[0], case.upper[0]],
        interval[case.lower[1], case.upper[1]],
    ]
    task_vertices = box_vertices(case.lower, case.upper)

    pending = deque([initial])
    terminal = {classification: [] for classification in CLASSIFICATIONS}
    reasons: Counter[str] = Counter()
    iterations = 0
    start = time.perf_counter()

    while pending:
        box = pending.popleft()
        classification, reason = classify_box(
            box,
            mode,
            z_offset,
            task_vertices,
            current_lower,
            current_upper,
            current_intervals,
            task_intervals,
        )
        iterations += 1
        if classification != "unresolved":
            terminal[classification].append(box)
            reasons[reason] += 1
            continue

        width_x = box[0][0].upper - box[0][0].lower
        width_y = box[1][0].upper - box[1][0].lower
        if width_x > eps and width_y > eps:
            pending.extend(BisectBox(box))
        else:
            terminal["unresolved"].append(box)

    areas = {
        key: float(sum(box_area(box) for box in boxes))
        for key, boxes in terminal.items()
    }
    total_area = sum(areas.values())
    return {
        "case_key": case.key,
        "mode": mode,
        "eps_m": eps,
        "elapsed_s": time.perf_counter() - start,
        "iterations": iterations,
        "counts": {key: len(value) for key, value in terminal.items()},
        "areas_m2": areas,
        "area_fractions": {key: value / total_area for key, value in areas.items()},
        "certificate_counts": dict(reasons),
        "boxes": terminal,
    }


def regression_cross_check(
    case: RequirementCase, result: dict, nodes_per_axis: int
) -> dict:
    """Finite implementation check of hybrid terminal boxes, not a proof."""

    current_intervals = [interval[-15.0, 15.0]] * 10
    task_intervals = [
        interval[case.lower[0], case.upper[0]],
        interval[case.lower[1], case.upper[1]],
    ]
    failures = {"inside": [], "outside": []}
    failure_counts = {"inside": 0, "outside": 0}
    samples = {"inside": 0, "outside": 0}
    for classification in ("inside", "outside"):
        for box_index, box in enumerate(result["boxes"][classification]):
            xs = np.linspace(box[0][0].lower, box[0][0].upper, nodes_per_axis)
            ys = np.linspace(box[1][0].lower, box[1][0].upper, nodes_per_axis)
            for x in xs:
                for y in ys:
                    samples[classification] += 1
                    point_matrix = calculate_affine_A(
                        interval[float(x), float(x)],
                        interval[float(y), float(y)],
                        0.075,
                    )
                    feasible = RobustFeasible(
                        point_matrix, current_intervals, task_intervals
                    )
                    failed = (
                        classification == "inside" and not feasible
                    ) or (
                        classification == "outside" and feasible
                    )
                    if failed:
                        failure_counts[classification] += 1
                        if len(failures[classification]) < 20:
                            failures[classification].append(
                                {
                                    "box_index": box_index,
                                    "x_m": float(x),
                                    "y_m": float(y),
                                }
                            )

    return {
        "nodes_per_axis": nodes_per_axis,
        "sampled_points": samples,
        "inside_failures": failure_counts["inside"],
        "outside_failures": failure_counts["outside"],
        "first_failures": failures,
        "interpretation": "finite regression check only; not a continuous-region proof",
    }


def draw_terminal_boxes(axis, result: dict) -> None:
    for classification in ("unresolved", "outside", "inside"):
        for box in result["boxes"][classification]:
            x0, x1 = box[0][0].lower, box[0][0].upper
            y0, y1 = box[1][0].lower, box[1][0].upper
            axis.add_patch(
                patches.Rectangle(
                    (x0 * 1000.0, y0 * 1000.0),
                    (x1 - x0) * 1000.0,
                    (y1 - y0) * 1000.0,
                    facecolor=CLASS_COLORS[classification],
                    edgecolor="#454545",
                    hatch=CLASS_HATCHES[classification],
                    linewidth=0.13,
                )
            )


def save_figure(fig, stem: Path, png_dpi: int = 300) -> None:
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=png_dpi, bbox_inches="tight")


def plot_spatial_grid(results: dict, output_dir: Path) -> None:
    figure, axes = plt.subplots(
        len(CASES),
        len(MODES),
        figsize=(7.2, 11.5),
        sharex=True,
        sharey=True,
    )
    for row, case in enumerate(CASES):
        for column, mode in enumerate(MODES):
            axis = axes[row, column]
            result = results[case.key][mode]
            draw_terminal_boxes(axis, result)
            inside = 100.0 * result["area_fractions"]["inside"]
            outside = 100.0 * result["area_fractions"]["outside"]
            unresolved = 100.0 * result["area_fractions"]["unresolved"]
            axis.set_title(
                f"in {inside:.1f}% | out {outside:.1f}% | unresolved {unresolved:.1f}%",
                fontsize=6.2,
                pad=2.0,
            )
            axis.set_xlim(-50, 50)
            axis.set_ylim(-50, 50)
            axis.set_aspect("equal")
            axis.set_xticks((-50, -25, 0, 25, 50))
            axis.set_yticks((-50, -25, 0, 25, 50))
            axis.tick_params(labelsize=6, length=2.5)
            axis.grid(alpha=0.18, linewidth=0.35, linestyle=":")
            if row == 0:
                axis.text(
                    0.5,
                    1.10,
                    MODE_LABELS[mode],
                    transform=axis.transAxes,
                    ha="center",
                    va="bottom",
                    fontsize=8.2,
                    fontweight="bold",
                )
            if column == 0:
                axis.set_ylabel(case.row_label, fontsize=7.2, labelpad=8)
            if row == len(CASES) - 1:
                axis.set_xlabel("$x$ (mm)", fontsize=7)

    handles = [
        patches.Patch(
            facecolor=CLASS_COLORS[key],
            edgecolor="#454545",
            hatch=CLASS_HATCHES[key],
            label=CLASS_LABELS[key],
        )
        for key in CLASSIFICATIONS
    ]
    figure.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.006),
        fontsize=7.2,
    )
    figure.text(
        0.012,
        0.5,
        "$y$ position (mm)",
        rotation=90,
        va="center",
        fontsize=7,
    )
    figure.subplots_adjust(
        left=0.15,
        right=0.992,
        bottom=0.070,
        top=0.975,
        wspace=0.12,
        hspace=0.32,
    )
    save_figure(figure, output_dir / "actuation_requirement_spatial_grid")
    plt.close(figure)


def grouped_metric(
    axis,
    results: dict,
    metric,
    ylabel: str,
    panel_label: str,
    percentage: bool = False,
) -> None:
    x = np.arange(len(CASES), dtype=float)
    width = 0.25
    for index, mode in enumerate(MODES):
        values = np.array([metric(results[case.key][mode]) for case in CASES])
        if percentage:
            values *= 100.0
        axis.bar(
            x + (index - 1) * width,
            values,
            width=width,
            color=MODE_COLORS[mode],
            edgecolor="#333333",
            linewidth=0.45,
            label=MODE_LABELS[mode],
        )
    axis.set_xticks(x)
    axis.set_xticklabels([case.compact_label for case in CASES], rotation=35, ha="right")
    axis.set_ylabel(ylabel)
    axis.grid(axis="y", alpha=0.25, linewidth=0.45, linestyle=":")
    axis.text(
        -0.15,
        1.04,
        panel_label,
        transform=axis.transAxes,
        fontsize=8,
        fontweight="bold",
    )


def plot_summary(results: dict, output_dir: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(7.2, 2.75))
    grouped_metric(
        axes[0],
        results,
        lambda item: item["area_fractions"]["unresolved"],
        "Unresolved area (%)",
        "a",
        percentage=True,
    )
    grouped_metric(
        axes[1],
        results,
        lambda item: item["area_fractions"]["inside"],
        "TFW-in area (%)",
        "b",
        percentage=True,
    )
    grouped_metric(
        axes[2],
        results,
        lambda item: item["iterations"],
        "Evaluated boxes",
        "c",
    )
    axes[0].set_ylim(0, 105)
    axes[1].set_ylim(0, 105)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=3,
        fontsize=7,
    )
    figure.text(
        0.5,
        0.07,
        "Actuation half-widths $B_x/B_y$ (mT)",
        ha="center",
        fontsize=7,
    )
    figure.subplots_adjust(left=0.075, right=0.995, bottom=0.31, top=0.94, wspace=0.34)
    save_figure(figure, output_dir / "actuation_requirement_summary")
    plt.close(figure)


def serializable_result(result: dict) -> dict:
    serializable = {key: value for key, value in result.items() if key != "boxes"}
    serializable["terminal_boxes_m"] = {
        classification: [
            [
                float(box[0][0].lower),
                float(box[0][0].upper),
                float(box[1][0].lower),
                float(box[1][0].upper),
            ]
            for box in result["boxes"][classification]
        ]
        for classification in CLASSIFICATIONS
    }
    return serializable


def write_outputs(
    results: dict,
    cross_checks: dict,
    eps: float,
    output_dir: Path,
) -> None:
    payload = {
        "protocol": {
            "search_xy_m": [-0.05, 0.05],
            "paper_plane_z_m": -0.01,
            "coil_model_z_m": 0.075,
            "coordinate_origin_offset_m": 0.085,
            "current_limits_A": [-15.0, 15.0],
            "eps_m": eps,
            "changed_variable": "task-box half-widths for Bx and By",
            "classification_note": (
                "nominal model-based labels pending an inclusion-preserving, "
                "outward-rounded affine backend"
            ),
        },
        "cases": [
            {
                "key": case.key,
                "bx_T": [-case.bx_half_width_t, case.bx_half_width_t],
                "by_T": [-case.by_half_width_t, case.by_half_width_t],
                "description": case.description,
                "results": {
                    mode: serializable_result(results[case.key][mode])
                    for mode in MODES
                },
                "affine_recourse_cross_check": cross_checks[case.key],
            }
            for case in CASES
        ],
    }
    (output_dir / "actuation_requirement_metrics.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )

    with (output_dir / "actuation_requirement_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        fieldnames = [
            "case",
            "description",
            "bx_half_width_T",
            "by_half_width_T",
            "mode",
            "inside_area_percent",
            "outside_area_percent",
            "unresolved_area_percent",
            "inside_boxes",
            "outside_boxes",
            "unresolved_boxes",
            "evaluated_boxes",
            "runtime_s",
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for case in CASES:
            for mode in MODES:
                result = results[case.key][mode]
                writer.writerow(
                    {
                        "case": case.key,
                        "description": case.description,
                        "bx_half_width_T": case.bx_half_width_t,
                        "by_half_width_T": case.by_half_width_t,
                        "mode": mode,
                        "inside_area_percent": 100.0
                        * result["area_fractions"]["inside"],
                        "outside_area_percent": 100.0
                        * result["area_fractions"]["outside"],
                        "unresolved_area_percent": 100.0
                        * result["area_fractions"]["unresolved"],
                        "inside_boxes": result["counts"]["inside"],
                        "outside_boxes": result["counts"]["outside"],
                        "unresolved_boxes": result["counts"]["unresolved"],
                        "evaluated_boxes": result["iterations"],
                        "runtime_s": result["elapsed_s"],
                    }
                )


def write_readme(output_dir: Path, eps: float) -> None:
    text = f"""# Actuation-requirement comparison

This directory compares three region-decision pipelines on the fixed ten-coil
Fig. 12 geometry.  Only the required symmetric field box in `(Bx, By)` changes.
The search domain is `[-0.05, 0.05]^2 m`, the paper plane is `z=-0.01 m`
(`z=0.075 m` in the coil-model coordinate), currents are limited to `+/-15 A`,
and the bisection tolerance is `{eps:g} m`.

Cases are reported as the positive half-widths `Bx/By` in mT:

- `60/60`: easy symmetric requirement;
- `80/80`: nominal Fig. 12 requirement;
- `100/100`: mixed-capacity symmetric requirement;
- `120/120`: beyond-capacity symmetric requirement;
- `120/40`: Bx-dominant anisotropic requirement;
- `40/120`: By-dominant anisotropic requirement.

Pipelines:

1. standard interval enclosure with the original Rohn/contractor tests;
2. affine enclosure with the same tests;
3. affine enclosure with the dependency-preserving recourse and dual tests.

Files:

- `actuation_requirement_spatial_grid.*`: full six-by-three spatial comparison;
- `actuation_requirement_summary.*`: unresolved area, TFW-in area, and box
  evaluation summaries;
- `actuation_requirement_metrics.json`: full protocol, terminal-box coordinates,
  metrics, certificate counts, and finite regression checks;
- `actuation_requirement_metrics.csv`: tidy source-data table.

The archived custom affine backend is not outward-rounded throughout, so these
are nominal model-based classifications.  The finite point checks are regression
tests only and do not replace the continuous region propositions.
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def validate_results(results: dict, cross_checks: dict) -> None:
    expected_area = 0.01
    for case in CASES:
        interval_result = results[case.key]["interval_original"]
        affine_result = results[case.key]["affine_original"]
        recourse_result = results[case.key]["affine_recourse"]
        for result in (interval_result, affine_result, recourse_result):
            total = sum(result["areas_m2"].values())
            if not np.isclose(total, expected_area, atol=1e-12):
                raise RuntimeError(
                    f"area partition failed for {case.key}/{result['mode']}: {total}"
                )
        if affine_result["area_fractions"]["inside"] + 1e-12 < interval_result[
            "area_fractions"
        ]["inside"]:
            raise RuntimeError(f"affine enclosure regressed for {case.key}")
        if recourse_result["area_fractions"]["inside"] + 1e-12 < affine_result[
            "area_fractions"
        ]["inside"]:
            raise RuntimeError(f"recourse inside set regressed for {case.key}")
        check = cross_checks[case.key]
        if check["inside_failures"] or check["outside_failures"]:
            raise RuntimeError(f"finite regression check failed for {case.key}")


def print_summary(results: dict) -> None:
    header = (
        "case".ljust(26)
        + "method".ljust(24)
        + "in %".rjust(9)
        + "out %".rjust(9)
        + "unknown %".rjust(12)
        + "boxes".rjust(9)
        + "time s".rjust(10)
    )
    print(header)
    print("-" * len(header))
    for case in CASES:
        for mode in MODES:
            result = results[case.key][mode]
            fractions = result["area_fractions"]
            print(
                case.key.ljust(26)
                + MODE_LABELS[mode].ljust(24)
                + f"{100 * fractions['inside']:9.3f}"
                + f"{100 * fractions['outside']:9.3f}"
                + f"{100 * fractions['unresolved']:12.3f}"
                + f"{result['iterations']:9d}"
                + f"{result['elapsed_s']:10.2f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eps", type=float, default=0.002)
    parser.add_argument("--cross-check-nodes", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.eps <= 0:
        parser.error("--eps must be positive")
    if args.cross_check_nodes < 2:
        parser.error("--cross-check-nodes must be at least 2")

    configure_matplotlib()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for case in CASES:
        results[case.key] = {}
        for mode in MODES:
            print(f"Running {case.key}: {MODE_LABELS[mode]}", flush=True)
            results[case.key][mode] = run_case(case, mode, args.eps)

    cross_checks = {}
    for case in CASES:
        print(f"Cross-checking {case.key}: affine + recourse", flush=True)
        cross_checks[case.key] = regression_cross_check(
            case,
            results[case.key]["affine_recourse"],
            args.cross_check_nodes,
        )

    validate_results(results, cross_checks)
    write_outputs(results, cross_checks, args.eps, args.output_dir)
    write_readme(args.output_dir, args.eps)
    plot_spatial_grid(results, args.output_dir)
    plot_summary(results, args.output_dir)
    print_summary(results)
    print(f"\nOutputs: {args.output_dir}")


if __name__ == "__main__":
    main()
