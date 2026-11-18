#!/usr/bin/env python3
# Run from the repository root: python -m revision_analysis.quantitative_validation
"""Reproducible, dependency-free audit for the IJRR revision.

The script reads existing experimental records without modifying them.  It reports
single-run tracking metrics separately from repeated-trial success rates (which
cannot be inferred from the available files), aggregates the ten-coil field-model
validation, and checks three algebraic identities used in the revised manuscript.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from pathlib import Path
from typing import Iterable, Sequence


def read_numeric_csv(path: Path) -> list[dict[str, float]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = []
        for raw in csv.DictReader(handle):
            rows.append({key: float(value) for key, value in raw.items()})
    if not rows:
        raise ValueError(f"No data rows in {path}")
    return rows


def percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    location = probability * (len(ordered) - 1)
    lower = math.floor(location)
    upper = math.ceil(location)
    weight = location - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def error_summary(errors: Sequence[float], scale: float = 1.0) -> dict[str, float | int]:
    absolute = [abs(value) for value in errors]
    return {
        "n_samples": len(errors),
        "mae": statistics.fmean(absolute) * scale,
        "rmse": math.sqrt(statistics.fmean(value * value for value in errors)) * scale,
        "p95_absolute": percentile(absolute, 0.95) * scale,
        "max_absolute": max(absolute) * scale,
    }


def sampling_summary(rows: Sequence[dict[str, float]]) -> dict[str, float | int]:
    times = [row["Time(s)"] for row in rows]
    steps = [right - left for left, right in zip(times, times[1:]) if right > left]
    median_step = statistics.median(steps)
    return {
        "n_samples": len(rows),
        "duration_s": times[-1] - times[0],
        "median_period_s": median_step,
        "median_rate_hz": 1.0 / median_step,
    }


def analyze_case_2(model_root: Path) -> dict:
    source = model_root / "Case_2" / "magnet_positions_recorded.csv"
    rows = read_numeric_csv(source)
    error_1 = [row["Actual_Magnet1_Y(m)"] - row["Desired_Magnet1_Y(m)"] for row in rows]
    error_2 = [row["Actual_Magnet3_Y(m)"] - row["Desired_Magnet3_Y(m)"] for row in rows]
    joint = [math.hypot(first, second) for first, second in zip(error_1, error_2)]
    tolerance_m = 0.005
    return {
        "source": str(source),
        "scope": "one recorded run; sample statistics are not a repeated-trial success rate",
        "sampling": sampling_summary(rows),
        "magnet_1_y_mm": error_summary(error_1, 1000.0),
        "magnet_2_y_mm": error_summary(error_2, 1000.0),
        "joint_two_axis_error_mm": error_summary(joint, 1000.0),
        "samples_both_within_5_mm_percent": 100.0 * sum(
            abs(first) <= tolerance_m and abs(second) <= tolerance_m
            for first, second in zip(error_1, error_2)
        ) / len(rows),
    }


def case_6_metrics(rows: Sequence[dict[str, float]]) -> dict:
    error_x = [row["Actual_X(m)"] - row["Desired_X(m)"] for row in rows]
    error_y = [row["Actual_Y(m)"] - row["Desired_Y(m)"] for row in rows]
    error_position = [math.hypot(x_value, y_value) for x_value, y_value in zip(error_x, error_y)]
    error_theta = [
        math.atan2(
            math.sin(row["Actual_Theta(rad)"] - row["Desired_Theta(rad)"]),
            math.cos(row["Actual_Theta(rad)"] - row["Desired_Theta(rad)"]),
        )
        for row in rows
    ]
    return {
        "sampling": sampling_summary(rows),
        "x_error_mm": error_summary(error_x, 1000.0),
        "y_error_mm": error_summary(error_y, 1000.0),
        "position_error_mm": error_summary(error_position, 1000.0),
        "orientation_error_deg": error_summary(error_theta, 180.0 / math.pi),
        "samples_within_5_mm_and_15_deg_percent": 100.0 * sum(
            position <= 0.005 and abs(theta) <= math.radians(15.0)
            for position, theta in zip(error_position, error_theta)
        ) / len(rows),
    }


def analyze_case_6(model_root: Path) -> dict:
    source = model_root / "Case_6" / "closed_loop_results.csv"
    rows = read_numeric_csv(source)
    initial = rows[0]
    movement_threshold = 1.0e-6
    first_active_index = next(
        index
        for index, row in enumerate(rows)
        if math.hypot(row["Desired_X(m)"] - initial["Desired_X(m)"], row["Desired_Y(m)"] - initial["Desired_Y(m)"])
        > movement_threshold
        or abs(row["Desired_Theta(rad)"] - initial["Desired_Theta(rad)"]) > movement_threshold
    )
    active_rows = rows[first_active_index:]
    legacy_rows = [
        row for row in rows
        if row["Time(s)"] >= rows[0]["Time(s)"] + 22.2
        and row["Time(s)"] <= rows[-1]["Time(s)"] - 2.4
    ]
    return {
        "source": str(source),
        "scope": "one recorded run; sample statistics are not a repeated-trial success rate",
        "full_record": case_6_metrics(rows),
        "active_reference_interval": {
            "definition": "from the first recorded change in desired pose through the end of the record",
            "start_time_s": active_rows[0]["Time(s)"],
            **case_6_metrics(active_rows),
        },
        "legacy_figure_window": {
            "definition": "the manually cropped interval used by the existing Case_6/plot_final_results.py",
            "warning": "reported for auditability only; it must not replace the full or pre-declared active-interval result",
            **case_6_metrics(legacy_rows),
        },
    }


def analyze_coil_model(model_root: Path) -> dict:
    source = model_root / "10 coil data 3D" / "coil1_to_10_model_accuracy_summary.csv"
    rows = read_numeric_csv(source)
    pooled = lambda key: math.sqrt(statistics.fmean(row[key] ** 2 for row in rows))
    return {
        "source": str(source),
        "n_coils": len(rows),
        "samples_per_coil": int(rows[0]["sample_count"]),
        "test_points_per_coil": int(rows[0]["test_point_count"]),
        "mean_vector_error_mT_across_coils": statistics.fmean(row["mean_vector_error_mT"] for row in rows),
        "pooled_vector_rmse_mT": pooled("rms_vector_error_mT"),
        "worst_sample_vector_error_mT": max(row["max_vector_error_mT"] for row in rows),
        "pooled_bx_rmse_mT": pooled("bx_rmse_mT"),
        "pooled_by_rmse_mT": pooled("by_rmse_mT"),
        "pooled_bz_rmse_mT": pooled("bz_rmse_mT"),
        "limitation": "coil-field validation does not validate target-target dipole forces at close separation",
    }


def dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def matvec(matrix: Sequence[Sequence[float]], vector: Sequence[float]) -> list[float]:
    return [dot(row, vector) for row in matrix]


def algebraic_checks() -> dict:
    rng = random.Random(7175)
    force_max_residual = 0.0
    support_max_gap = 0.0
    for _ in range(1000):
        gxx, gxy, gxz, gyy, gyz = [rng.uniform(-3.0, 3.0) for _ in range(5)]
        mx, my, mz = [rng.uniform(-2.0, 2.0) for _ in range(3)]
        gradient = [[gxx, gxy, gxz], [gxy, gyy, gyz], [gxz, gyz, -(gxx + gyy)]]
        reduced = [
            [mx, my, mz, 0.0, 0.0],
            [0.0, mx, 0.0, my, mz],
            [-mz, 0.0, mx, -mz, my],
        ]
        direct = matvec(gradient, [mx, my, mz])
        compact = matvec(reduced, [gxx, gxy, gxz, gyy, gyz])
        force_max_residual = max(force_max_residual, max(abs(a - b) for a, b in zip(direct, compact)))

        normal = [rng.uniform(-1.0, 1.0) for _ in range(6)]
        radii = [rng.uniform(0.0, 2.0) for _ in range(3)]
        analytic = sum(radii[k] * math.hypot(normal[2 * k], normal[2 * k + 1]) for k in range(3))
        maximizer = []
        for k, radius in enumerate(radii):
            block = normal[2 * k:2 * k + 2]
            block_norm = math.hypot(*block)
            maximizer.extend([0.0, 0.0] if block_norm == 0.0 else [radius * value / block_norm for value in block])
        support_max_gap = max(support_max_gap, abs(analytic - dot(normal, maximizer)))

    nominal_denominators = [10, 15, 20, 30, 40, 60, 90, 120, 180]
    depths = [math.ceil(math.log2(value)) for value in nominal_denominators]
    reported_inner_percent = [12.50, 12.50, 32.42, 32.42, 46.87, 46.87, 55.98, 61.17, 61.17]
    return {
        "force_gradient_identity_max_abs_residual": force_max_residual,
        "product_of_balls_support_identity_max_abs_residual": support_max_gap,
        "resolution_nominal_denominators": nominal_denominators,
        "resolution_effective_dyadic_depths": depths,
        "resolution_reported_inner_percent": reported_inner_percent,
        "resolution_inner_fraction_monotone": all(
            right >= left for left, right in zip(reported_inner_percent, reported_inner_percent[1:])
        ),
        "rohn_lp_count_note": "for d selected output equations, the row-sign test has 2^d systems (not 2^number_of_currents)",
    }


def tex_number(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}"


def write_tex(path: Path, report: dict) -> None:
    case_2 = report["case_2"]
    case_6 = report["case_6"]["active_reference_interval"]
    coil = report["coil_model"]
    tracking = f"""% Generated by revision_analysis/quantitative_validation.py; tracking table only.
\\begin{{table*}}[t]
\\centering
\\caption{{Quantitative results recoverable from the archived data. Each motion result is from one recorded run ($n=1$); therefore, sample-level percentages are not repeated-trial success rates.}}
\\label{{tab:archived_quantitative_results}}
\\begin{{tabular}}{{@{{}}lrrr@{{}}}}
\\toprule
Record & MAE & RMSE & 95th percentile \\\\
\\midrule
Case 2, magnet 1 position (mm) & {tex_number(case_2['magnet_1_y_mm']['mae'])} & {tex_number(case_2['magnet_1_y_mm']['rmse'])} & {tex_number(case_2['magnet_1_y_mm']['p95_absolute'])} \\\\
Case 2, magnet 2 position (mm) & {tex_number(case_2['magnet_2_y_mm']['mae'])} & {tex_number(case_2['magnet_2_y_mm']['rmse'])} & {tex_number(case_2['magnet_2_y_mm']['p95_absolute'])} \\\\
Case 6, planar position (mm) & {tex_number(case_6['position_error_mm']['mae'])} & {tex_number(case_6['position_error_mm']['rmse'])} & {tex_number(case_6['position_error_mm']['p95_absolute'])} \\\\
Case 6, orientation (deg) & {tex_number(case_6['orientation_error_deg']['mae'])} & {tex_number(case_6['orientation_error_deg']['rmse'])} & {tex_number(case_6['orientation_error_deg']['p95_absolute'])} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table*}}
"""
    macros = f"""% Generated by revision_analysis/quantitative_validation.py; coil-field macros only.
\\newcommand{{\\CoilMeanVectorError}}{{{tex_number(coil['mean_vector_error_mT_across_coils'], 3)}~mT}}
\\newcommand{{\\CoilPooledVectorRMSE}}{{{tex_number(coil['pooled_vector_rmse_mT'], 3)}~mT}}
\\newcommand{{\\CoilWorstVectorError}}{{{tex_number(coil['worst_sample_vector_error_mT'], 3)}~mT}}
"""
    path.write_text(tracking + "\n" + macros, encoding="utf-8")
    path.with_name("tracking_metrics.tex").write_text(tracking, encoding="utf-8")
    path.with_name("coil_field_macros.tex").write_text(macros, encoding="utf-8")


def main() -> None:
    workspace_root = Path(__file__).resolve().parent.parent
    research_root = workspace_root.parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-root",
        type=Path,
        default=research_root / "Reconfigurable surgical robot document" / "Modeling",
    )
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "case_2": analyze_case_2(args.model_root),
        "case_6": analyze_case_6(args.model_root),
        "coil_model": analyze_coil_model(args.model_root),
        "algebraic_checks": algebraic_checks(),
        "missing_evidence": [
            "repeated physical trials with a pre-declared trial-level success criterion",
            "physical tests at configurations certified inside, near, and outside the TFW boundary",
            "close-range target-target dipole field/force measurements over separation and orientation",
            "unconstrained simultaneous position-and-orientation control",
            "step/chirp transient measurements for rise time, settling time, overshoot, and bandwidth",
        ],
    }
    json_path = args.output_dir / "quantitative_validation.json"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_tex(args.output_dir / "quantitative_metrics.tex", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
