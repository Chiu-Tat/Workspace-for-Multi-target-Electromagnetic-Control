#!/usr/bin/env python3
# Run from the repository root: python -m revision_analysis.benchmark_tight_region_check
"""Benchmark the hybrid region checker on the manuscript's Fig. 12 stress case.

The benchmark preserves the original model, initial box, current limits, task
box, bisection tolerance, and coordinate transform.  The actuation requirement
is the anisotropic field box Bx in [-0.12, 0.12] T and By in [-0.04, 0.04] T.
Three decision cascades isolate the enclosure and certificate contributions:

interval_original: standard interval enclosure -> Rohn/contractor -> split
affine_original:   affine enclosure -> Rohn/contractor -> split
affine_hybrid:     affine_original -> constructive recourse/dual tests -> split

Results are reported by area as well as box count because terminal boxes need
not have the same size in more general runs.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, deque
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


HERE = Path(__file__).resolve().parent
CODE_ROOT = HERE.parent
sys.path.insert(0, str(CODE_ROOT))

from analysis.affine.ten_coil import (  # noqa: E402
    BisectBox,
    Create2DBox,
    calculate_affine_A,
    calculate_interval_A,
)
from emns.workspace.checks import Out, RobustFeasible  # noqa: E402
from emns.numerics.interval import interval  # noqa: E402
from emns.certificates.tight_region import (  # noqa: E402
    affine_constructive_inside_certificate,
    box_vertices,
    constructive_inside_certificate,
    dual_outside_certificate,
    interval_object_bounds,
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
):
    if mode == "interval_original":
        A_interval = calculate_interval_A(box[0][0], box[1][0], z_offset)
    else:
        A_interval = calculate_affine_A(box[0][0], box[1][0], z_offset)
    if RobustFeasible(A_interval, current_intervals, task_intervals):
        return "inside", "rohn"

    G_lower = None
    G_upper = None
    if mode == "affine_hybrid":
        G_affine = calculate_affine_A(
            box[0][0], box[1][0], z_offset, return_affine=True
        )
        certificate = affine_constructive_inside_certificate(
            G_affine,
            task_vertices,
            current_lower,
            current_upper,
        )
        if certificate.certified:
            return "inside", "affine_right_inverse"

        G_lower, G_upper = interval_object_bounds(A_interval)
        certificate = constructive_inside_certificate(
            G_lower,
            G_upper,
            task_vertices,
            current_lower,
            current_upper,
        )
        if certificate.certified:
            return "inside", "right_inverse"

    if Out(A_interval, current_intervals, task_intervals):
        return "outside", "contractor"

    if mode == "affine_hybrid":
        if G_lower is None:
            G_lower, G_upper = interval_object_bounds(A_interval)
        certificate = dual_outside_certificate(
            G_lower,
            G_upper,
            task_vertices,
            current_lower,
            current_upper,
        )
        if certificate.certified:
            return "outside", "dual_separation"

    return "undetermined", "none"


def run(mode: str, eps: float) -> dict:
    initial = Create2DBox(interval[-0.05, 0.05], interval[-0.05, 0.05])
    z_offset = 0.075
    current_lower = np.full(10, -15.0)
    current_upper = np.full(10, 15.0)
    task_lower = np.array([-0.12, -0.04])
    task_upper = np.array([0.12, 0.04])
    task_vertices = box_vertices(task_lower, task_upper)
    current_intervals = [interval[-15.0, 15.0]] * 10
    task_intervals = [interval[-0.12, 0.12], interval[-0.04, 0.04]]

    pending = deque([initial])
    terminal = {"inside": [], "outside": [], "undetermined": []}
    reason_counts: Counter[str] = Counter()
    iteration = 0
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
        iteration += 1
        if classification != "undetermined":
            terminal[classification].append(box)
            reason_counts[reason] += 1
            continue
        width_x = box[0][0].upper - box[0][0].lower
        width_y = box[1][0].upper - box[1][0].lower
        if width_x > eps and width_y > eps:
            pending.extend(BisectBox(box))
        else:
            terminal["undetermined"].append(box)
    elapsed = time.perf_counter() - start

    areas = {
        key: float(sum(box_area(box) for box in boxes))
        for key, boxes in terminal.items()
    }
    total_area = sum(areas.values())
    return {
        "mode": mode,
        "eps_m": eps,
        "elapsed_s": elapsed,
        "iterations": iteration,
        "counts": {key: len(value) for key, value in terminal.items()},
        "areas_m2": areas,
        "area_fractions": {key: value / total_area for key, value in areas.items()},
        "certificate_counts": dict(reason_counts),
        "boxes": terminal,
    }


def serializable(result: dict) -> dict:
    return {key: value for key, value in result.items() if key != "boxes"}


def pointwise_cross_check(result: dict, nodes_per_axis: int) -> dict:
    """Sample returned inside/outside boxes with exact-point feasibility LPs.

    This is a regression check, not a region proof.  It catches implementation
    errors by testing a Cartesian grid that includes every box corner and centre
    when ``nodes_per_axis=3``.
    """

    current_intervals = [interval[-15.0, 15.0]] * 10
    task_intervals = [interval[-0.12, 0.12], interval[-0.04, 0.04]]
    failures = {"inside": [], "outside": []}
    failure_counts = {"inside": 0, "outside": 0}
    sampled_points = {"inside": 0, "outside": 0}
    for classification in ("inside", "outside"):
        for box_index, box in enumerate(result["boxes"][classification]):
            xs = np.linspace(box[0][0].lower, box[0][0].upper, nodes_per_axis)
            ys = np.linspace(box[1][0].lower, box[1][0].upper, nodes_per_axis)
            for x in xs:
                for y in ys:
                    sampled_points[classification] += 1
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
        "terminal_boxes": {
            key: len(result["boxes"][key]) for key in ("inside", "outside")
        },
        "sampled_points": sampled_points,
        "pointwise_failures": failure_counts,
        "first_failures": failures,
        "interpretation": "regression cross-check only; not a continuous-region proof",
    }


def plot_two_pipelines(results: list[dict], outputs: list[Path]) -> None:
    """Pipelines A and C only, drawn at the paper column width (84.9 mm)."""
    plt.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
                         "mathtext.fontset": "dejavusans"})
    colors = {"inside": "#8fd19e", "outside": "#f4b6b6", "undetermined": "#d9d9d9"}
    labels = {"inside": "TFW-in", "outside": "TFW-out", "undetermined": "Unresolved"}
    panels = ((results[0], "(a) Interval + baseline"), (results[2], "(b) Affine + recourse/dual"))
    fig, axes = plt.subplots(1, 2, figsize=(3.34, 2.15), sharex=True, sharey=True)
    for ax, (result, label) in zip(axes, panels):
        for classification in ("undetermined", "outside", "inside"):
            for box in result["boxes"][classification]:
                x0, x1 = box[0][0].lower, box[0][0].upper
                y0, y1 = box[1][0].lower, box[1][0].upper
                ax.add_patch(patches.Rectangle((x0 * 1000.0, y0 * 1000.0), (x1 - x0) * 1000.0,
                                               (y1 - y0) * 1000.0, facecolor=colors[classification],
                                               linewidth=0))
        fr = {k: 100.0 * v for k, v in result["area_fractions"].items()}
        ax.set_title(f"{label}\nunresolved {fr['undetermined']:.1f}%", fontsize=8, pad=3)
        ax.set_xlim(-50, 50)
        ax.set_ylim(-50, 50)
        ax.set_aspect("equal")
        ax.set_xlabel("$x$ (mm)", fontsize=8, labelpad=1)
        ax.set_xticks((-40, 0, 40), ("\u221240", "0", "40"))
        ax.set_yticks((-40, 0, 40), ("\u221240", "0", "40"))
        ax.tick_params(labelsize=7, pad=1.5)
    axes[0].set_ylabel("$y$ (mm)", fontsize=8, labelpad=1)
    fig.legend(handles=[patches.Patch(facecolor=colors[k], label=labels[k]) for k in colors],
               loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=3, frameon=False, fontsize=7.5,
               handlelength=1.4, columnspacing=1.0)
    fig.subplots_adjust(left=0.13, right=0.98, top=0.83, bottom=0.27, wspace=0.12)
    for output in outputs:
        for suffix in ("pdf", "png"):
            fig.savefig(output.with_suffix(f".{suffix}"), dpi=300)
    plt.close(fig)


def plot_results(results: list[dict], outputs: list[Path]) -> None:
    plt.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
                         "mathtext.fontset": "dejavusans"})
    colors = {  # same palette as the Section 6.3 resolution figure
        "inside": "#8fd19e",
        "outside": "#f4b6b6",
        "undetermined": "#d9d9d9",
    }
    hatches = {"inside": "", "outside": "", "undetermined": ""}
    labels = {
        "inside": "TFW-in",
        "outside": "TFW-out",
        "undetermined": "Unresolved",
    }
    panel_labels = (
        "(a) Interval + baseline tests",
        "(b) Affine + baseline tests",
        "(c) Affine + recourse/dual tests",
    )
    # Drawn at the paper text width (174 mm) so the fonts print at their nominal size.
    fig, axes = plt.subplots(1, 3, figsize=(6.85, 2.55), sharex=True, sharey=True)
    for ax, result, label in zip(axes, results, panel_labels):
        for classification in ("undetermined", "outside", "inside"):
            for box in result["boxes"][classification]:
                x0, x1 = box[0][0].lower, box[0][0].upper
                y0, y1 = box[1][0].lower, box[1][0].upper
                ax.add_patch(
                    patches.Rectangle(
                        (x0 * 1000.0, y0 * 1000.0),
                        (x1 - x0) * 1000.0,
                        (y1 - y0) * 1000.0,
                        facecolor=colors[classification],
                        edgecolor="#3d3d3d",
                        hatch=hatches[classification],
                        linewidth=0,
                    )
                )
        inside = 100.0 * result["area_fractions"]["inside"]
        outside = 100.0 * result["area_fractions"]["outside"]
        unknown = 100.0 * result["area_fractions"]["undetermined"]
        ax.set_title(
            f"{label}\nin {inside:.1f}% | out {outside:.1f}% | unres. {unknown:.1f}%",
            fontsize=8,
            pad=3,
        )
        ax.set_xlim(-50, 50)
        ax.set_ylim(-50, 50)
        ax.set_aspect("equal")
        ax.set_xlabel("$x$ (mm)", fontsize=8)
        ax.set_xticks((-40, 0, 40), ("\u221240", "0", "40"))
        ax.set_yticks((-40, 0, 40), ("\u221240", "0", "40"))
        ax.tick_params(labelsize=7)
        ax.grid(False)
    axes[0].set_ylabel("$y$ (mm)", fontsize=8)
    handles = [
        patches.Patch(
            facecolor=colors[key],
            edgecolor="#3d3d3d",
            hatch=hatches[key],
            label=labels[key],
        )
        for key in ("inside", "outside", "undetermined")
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=3,
        frameon=False,
        fontsize=8,
    )
    fig.subplots_adjust(left=0.07, right=0.99, top=0.84, bottom=0.25, wspace=0.12)
    for output in outputs:
        for suffix in ("pdf", "png"):
            fig.savefig(output.with_suffix(f".{suffix}"), dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eps", type=float, default=0.002)
    parser.add_argument(
        "--output",
        type=Path,
        default=HERE / "tight_region_benchmark.json",
    )
    parser.add_argument("--cross-check-nodes", type=int, default=3)
    parser.add_argument("--figure-only", action="store_true",
                        help="recompute the boxes and redraw the figure without rewriting the JSON")
    args = parser.parse_args()
    if args.figure_only:
        import pickle
        cache = args.output.with_suffix(".boxes.pkl")
        if cache.exists():
            results = pickle.loads(cache.read_bytes())
        else:
            results = [run(mode, args.eps) for mode in ("interval_original", "affine_original", "affine_hybrid")]
            try:
                cache.write_bytes(pickle.dumps(results))
            except Exception as exc:  # cache is only a convenience
                print(f"box cache not written: {exc}")
        plot_two_pipelines(results, [args.output.with_suffix("")])
        return

    interval_original = run("interval_original", args.eps)
    affine_original = run("affine_original", args.eps)
    affine_hybrid = run("affine_hybrid", args.eps)
    interval_unknown = interval_original["area_fractions"]["undetermined"]
    baseline_unknown = affine_original["area_fractions"]["undetermined"]
    hybrid_unknown = affine_hybrid["area_fractions"]["undetermined"]
    reduction = 1.0 - hybrid_unknown / baseline_unknown if baseline_unknown else 0.0
    result = {
        "protocol": {
            "initial_xy_m": [-0.05, 0.05],
            "paper_plane_z_m": -0.01,
            "coil_model_z_m": 0.075,
            "coordinate_origin_offset_m": 0.085,
            "current_A": [-15.0, 15.0],
            "task_box_T": {"Bx": [-0.12, 0.12], "By": [-0.04, 0.04]},
            "eps_m": args.eps,
            "case": "anisotropic Fig. 12 stress case",
        },
        "interval_original": serializable(interval_original),
        "affine_original": serializable(affine_original),
        "affine_hybrid": serializable(affine_hybrid),
        "standard_to_affine_unresolved_area_relative_reduction": (
            1.0 - baseline_unknown / interval_unknown if interval_unknown else 0.0
        ),
        "affine_to_hybrid_unresolved_area_relative_reduction": reduction,
        "standard_to_hybrid_unresolved_area_relative_reduction": (
            1.0 - hybrid_unknown / interval_unknown if interval_unknown else 0.0
        ),
        "affine_to_hybrid_box_evaluation_relative_reduction": (
            1.0 - affine_hybrid["iterations"] / affine_original["iterations"]
            if affine_original["iterations"]
            else 0.0
        ),
        "affine_to_hybrid_runtime_ratio": (
            affine_hybrid["elapsed_s"] / affine_original["elapsed_s"]
            if affine_original["elapsed_s"]
            else 0.0
        ),
    }
    if args.cross_check_nodes >= 2:
        result["affine_hybrid_pointwise_cross_check"] = pointwise_cross_check(
            affine_hybrid, args.cross_check_nodes
        )
        if any(
            result["affine_hybrid_pointwise_cross_check"]["pointwise_failures"].values()
        ):
            raise RuntimeError("hybrid terminal boxes failed the pointwise regression check")
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    figure_name = args.output.stem
    plot_results(
        [interval_original, affine_original, affine_hybrid],
        [HERE / figure_name],
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
