#!/usr/bin/env python3
"""Compute a positional TFW slice for two control-actuated soft spheres.

Run from the repository root::

    python -m cases.case_9_two_soft_spheres_control_hold.workspace --smoke
    python -m cases.case_9_two_soft_spheres_control_hold.workspace
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import time
from collections import Counter, deque
from contextlib import nullcontext
from functools import partial

import numpy as np
from matplotlib.patches import Circle, Patch, Rectangle

from emns.numerics.interval import interval
from emns.workspace.checks import BisectBox, Create2DBox
from hybrid_region.certificates import (
    affine_constructive_inside_certificate,
    constructive_inside_certificate,
    dual_outside_certificate,
    interval_object_bounds,
)

from .model import (
    B_HOLD,
    F_HOLD,
    F_MOVE,
    HELD_RADIUS,
    HELD_X,
    HELD_Y,
    I_MAX,
    I_MIN,
    MOVING_RADIUS,
    NUM_COILS,
    OUTPUT_LABELS,
    SATURATION_MAGNETIZATION,
    SUSCEPTIBILITY,
    XY_LIMIT,
    Z_PLANE,
    affine_actuation_map,
    operating_moments,
    separation_bounds,
    task_bounds,
    task_vertices,
)
from cases.case_8_two_sphere_control_hold.model import SeparationError


RESULTS_JSON = "Results/results_case9_two_soft_spheres_control_hold.json"
RESULTS_PNG = "Results/results_case9_two_soft_spheres_control_hold.png"
DEFAULT_EPS = 0.001


def _validate_options(
    f_hold,
    f_move,
    bias_field,
    held_radius,
    moving_radius,
    susceptibility,
    saturation_magnetization,
    clearance,
    eps,
    processes=None,
    max_iter=-1,
):
    task_bounds(f_hold, f_move, bias_field)
    operating_moments(
        held_radius,
        moving_radius,
        bias_field=bias_field,
        susceptibility=susceptibility,
        saturation_magnetization=saturation_magnetization,
    )
    if not math.isfinite(clearance) or clearance < 0.0:
        raise ValueError("clearance must be finite and nonnegative")
    if not math.isfinite(eps) or eps <= 0.0:
        raise ValueError("eps must be finite and positive")
    if processes is not None and (
        not isinstance(processes, int) or isinstance(processes, bool) or processes < 1
    ):
        raise ValueError("processes must be a positive integer")
    if not isinstance(max_iter, int) or isinstance(max_iter, bool) or max_iter < -1:
        raise ValueError("max_iter must be -1 or a nonnegative integer")


def _width(box, axis):
    return float(box[axis][0].upper - box[axis][0].lower)


def _needs_bisection(box, eps):
    return _width(box, 0) > eps or _width(box, 1) > eps


def _box_to_list(box):
    return [
        float(box[0][0].lower), float(box[0][0].upper),
        float(box[1][0].lower), float(box[1][0].upper),
    ]


def classify_box(
    box,
    *,
    f_hold,
    f_move,
    held_radius,
    moving_radius,
    susceptibility,
    saturation_magnetization,
    clearance,
    eps,
    bias_field=B_HOLD,
):
    """Return a one-sided certificate verdict for one position rectangle."""

    x_interval, y_interval = box[0][0], box[1][0]
    required_separation = held_radius + moving_radius + clearance
    minimum, maximum = separation_bounds(x_interval, y_interval)
    if maximum <= required_separation:
        return "excluded", box, "physical_separation"
    if minimum <= required_separation:
        if _needs_bisection(box, eps):
            return "bisect", box, "separation_boundary"
        return "undetermined", box, "separation_boundary_leaf"

    material = {
        "held_radius": held_radius,
        "moving_radius": moving_radius,
        "bias_field": bias_field,
        "susceptibility": susceptibility,
        "saturation_magnetization": saturation_magnetization,
    }
    try:
        full_affine = affine_actuation_map(
            x_interval, y_interval, return_affine=True, **material
        )
    except (SeparationError, ArithmeticError, ValueError, ZeroDivisionError):
        if _needs_bisection(box, eps):
            return "bisect", box, "singular_enclosure"
        return "undetermined", box, "singular_enclosure"

    G_affine = full_affine[:, :NUM_COILS]
    c_affine = full_affine[:, NUM_COILS]
    vertices = task_vertices(f_hold, f_move, bias_field)
    current_lower = np.full(NUM_COILS, I_MIN)
    current_upper = np.full(NUM_COILS, I_MAX)

    inside = affine_constructive_inside_certificate(
        G_affine,
        vertices,
        current_lower,
        current_upper,
        c_affine=c_affine,
    )
    if inside.certified:
        return "inside", box, "affine_right_inverse"

    full_interval = np.asarray(
        [[entry.to_interval() for entry in row] for row in full_affine],
        dtype=object,
    )
    G_lower, G_upper = interval_object_bounds(full_interval[:, :NUM_COILS])
    c_lower, c_upper = interval_object_bounds(full_interval[:, NUM_COILS])
    interval_inside = constructive_inside_certificate(
        G_lower,
        G_upper,
        vertices,
        current_lower,
        current_upper,
        c_lower=c_lower,
        c_upper=c_upper,
    )
    if interval_inside.certified:
        return "inside", box, "interval_right_inverse"

    outside = dual_outside_certificate(
        G_lower,
        G_upper,
        vertices,
        current_lower,
        current_upper,
        c_lower=c_lower,
        c_upper=c_upper,
    )
    if outside.certified:
        return "outside", box, "dual_separation"
    if _needs_bisection(box, eps):
        return "bisect", box, "none"
    return "undetermined", box, "none"


def determine_workspace(
    *,
    f_hold=F_HOLD,
    f_move=F_MOVE,
    bias_field=B_HOLD,
    held_radius=HELD_RADIUS,
    moving_radius=MOVING_RADIUS,
    susceptibility=SUSCEPTIBILITY,
    saturation_magnetization=SATURATION_MAGNETIZATION,
    clearance=0.0,
    eps=DEFAULT_EPS,
    processes=None,
    max_iter=-1,
):
    _validate_options(
        f_hold, f_move, bias_field, held_radius, moving_radius, susceptibility,
        saturation_magnetization, clearance, eps, processes, max_iter,
    )
    if processes is None:
        processes = min(mp.cpu_count(), 16)
    pending = deque([Create2DBox(
        interval[-XY_LIMIT, XY_LIMIT], interval[-XY_LIMIT, XY_LIMIT]
    )])
    inside, outside, undetermined, excluded = [], [], [], []
    reasons = Counter()
    iterations = 0
    worker = partial(
        classify_box,
        f_hold=f_hold,
        f_move=f_move,
        bias_field=bias_field,
        held_radius=held_radius,
        moving_radius=moving_radius,
        susceptibility=susceptibility,
        saturation_magnetization=saturation_magnetization,
        clearance=clearance,
        eps=eps,
    )
    started = time.time()
    last_progress = 0
    context = mp.Pool(processes=processes) if processes > 1 and max_iter != 0 else nullcontext()
    with context as pool:
        while pending and (max_iter < 0 or iterations < max_iter):
            batch_size = min(processes, len(pending))
            if max_iter >= 0:
                batch_size = min(batch_size, max_iter - iterations)
            batch = [pending.popleft() for _ in range(batch_size)]
            verdicts = pool.map(worker, batch) if pool is not None else map(worker, batch)
            for action, box, reason in verdicts:
                if action == "inside":
                    inside.append(box)
                    reasons[reason] += 1
                elif action == "outside":
                    outside.append(box)
                    reasons[reason] += 1
                elif action == "excluded":
                    excluded.append(box)
                    reasons[reason] += 1
                elif action == "bisect":
                    pending.extend(BisectBox(box))
                else:
                    undetermined.append(box)
                    reasons[reason] += 1
            iterations += len(batch)
            if iterations - last_progress >= 100 or not pending:
                print(
                    f"boxes {iterations} queue {len(pending)} in {len(inside)} "
                    f"out {len(outside)} und {len(undetermined)} "
                    f"excluded {len(excluded)} {time.time() - started:.1f}s",
                    flush=True,
                )
                last_progress = iterations
    return {
        "complete": not pending,
        "pending": list(pending),
        "inside": inside,
        "outside": outside,
        "undetermined": undetermined,
        "excluded": excluded,
        "reasons": reasons,
        "iterations": iterations,
        "elapsed_s": time.time() - started,
    }


def _plot_rectangles(ax, boxes, color, alpha):
    for box in boxes:
        xmin, xmax, ymin, ymax = _box_to_list(box)
        ax.add_patch(Rectangle(
            (xmin, ymin), xmax - xmin, ymax - ymin,
            facecolor=color, edgecolor="none", alpha=alpha,
        ))


def save_plot(result, path, required_separation, *, show=False):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 6.5))
    _plot_rectangles(ax, result["pending"], "#e6bc53", 0.55)
    _plot_rectangles(ax, result["outside"], "#d95f5f", 0.62)
    _plot_rectangles(ax, result["undetermined"], "#9e9e9e", 0.55)
    _plot_rectangles(ax, result["inside"], "#3d995f", 0.72)
    _plot_rectangles(ax, result["excluded"], "#2f2f2f", 0.75)
    ax.add_patch(Circle(
        (HELD_X, HELD_Y), required_separation, fill=False,
        color="#171717", linestyle="--", linewidth=1.2,
    ))
    ax.plot(HELD_X, HELD_Y, marker="o", color="#171717", markersize=5)
    ax.annotate(
        "control-held soft sphere", (HELD_X, HELD_Y), xytext=(7, 8),
        textcoords="offset points", fontsize=9,
    )
    ax.set_aspect("equal")
    ax.set_xlim(-XY_LIMIT, XY_LIMIT)
    ax.set_ylim(-XY_LIMIT, XY_LIMIT)
    ax.set_xlabel(r"moving sphere $x_2$ (m)")
    ax.set_ylabel(r"moving sphere $y_2$ (m)")
    suffix = " (partial run)" if result["pending"] else ""
    ax.set_title("Case 9: two soft spheres, fixed-field TFW" + suffix)
    ax.grid(color="#e1ddd4", linewidth=0.5)
    ax.legend(handles=[
        Patch(facecolor="#3d995f", label="Certified inside"),
        Patch(facecolor="#d95f5f", label="Certified outside"),
        Patch(facecolor="#9e9e9e", label="Undetermined"),
        Patch(facecolor="#2f2f2f", label="Excluded by physical separation"),
        Patch(facecolor="#e6bc53", label="Pending (not evaluated)"),
    ], loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def _material_args(args):
    return {
        "held_radius": args.held_radius,
        "moving_radius": args.moving_radius,
        "bias_field": args.bias_field,
        "susceptibility": args.susceptibility,
        "saturation_magnetization": args.saturation_magnetization,
    }


def _smoke(args):
    for x, y in [(-0.03, 0.02), (0.0, 0.03), (0.02, 0.0)]:
        box = Create2DBox(interval[x, x], interval[y, y])
        action, _, reason = classify_box(
            box,
            f_hold=args.f_hold,
            f_move=args.f_move,
            clearance=args.clearance,
            eps=args.eps,
            **_material_args(args),
        )
        print(f"({x:+.3f}, {y:+.3f}) -> {action:12s} {reason}")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--f-hold", type=float, default=F_HOLD)
    parser.add_argument("--f-move", type=float, default=F_MOVE)
    parser.add_argument("--bias-field", type=float, default=B_HOLD)
    parser.add_argument("--held-radius", type=float, default=HELD_RADIUS)
    parser.add_argument("--moving-radius", type=float, default=MOVING_RADIUS)
    parser.add_argument("--susceptibility", type=float, default=SUSCEPTIBILITY)
    parser.add_argument(
        "--saturation-magnetization", type=float,
        default=SATURATION_MAGNETIZATION,
    )
    parser.add_argument("--clearance", type=float, default=0.0)
    parser.add_argument("--eps", type=float, default=DEFAULT_EPS)
    parser.add_argument("--processes", type=int, default=None)
    parser.add_argument("--max-iter", type=int, default=-1)
    parser.add_argument("--output-json", default=RESULTS_JSON)
    parser.add_argument("--output-plot", default=RESULTS_PNG)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args(argv)
    try:
        _validate_options(
            args.f_hold, args.f_move, args.bias_field,
            args.held_radius, args.moving_radius,
            args.susceptibility, args.saturation_magnetization,
            args.clearance, args.eps, args.processes, args.max_iter,
        )
    except ValueError as exc:
        parser.error(str(exc))
    if args.smoke:
        return _smoke(args)

    held_m, moving_m = operating_moments(**_material_args(args))
    required_separation = args.held_radius + args.moving_radius + args.clearance
    print("Case 9: two soft-magnetic spheres, one control-held")
    print(f"outputs: {', '.join(OUTPUT_LABELS)}")
    print(f"operating moments: held {held_m:.6g}, moving {moving_m:.6g} A m^2")
    print(f"physical centre separation: {required_separation:.6g} m")

    result = determine_workspace(
        f_hold=args.f_hold,
        f_move=args.f_move,
        clearance=args.clearance,
        eps=args.eps,
        processes=args.processes,
        max_iter=args.max_iter,
        **_material_args(args),
    )
    payload = {
        "mode": "soft_spheres_fixed_local_field_control_hold",
        "inside_certificate": "affine/interval constructive right inverse",
        "outside_certificate": "dual support-function separation",
        "warning": (
            "Ideal isotropic, reversible operating-point model; hysteresis, "
            "remanence, eddy-current dynamics, nonuniform magnetization, and "
            "closed-loop stability are not certified. Floating-point "
            "enclosures are not machine-verified."
        ),
        "complete": result["complete"],
        "eps_m": args.eps,
        "iterations": result["iterations"],
        "elapsed_s": result["elapsed_s"],
        "certificate_counts": dict(result["reasons"]),
        "in_workspace": [_box_to_list(box) for box in result["inside"]],
        "out_of_workspace": [_box_to_list(box) for box in result["outside"]],
        "undetermined": [_box_to_list(box) for box in result["undetermined"]],
        "excluded": [_box_to_list(box) for box in result["excluded"]],
        "pending": [_box_to_list(box) for box in result["pending"]],
        "protocol": {
            "plane_z_m": Z_PLANE,
            "xy_domain_m": [-XY_LIMIT, XY_LIMIT],
            "held_position_m": [HELD_X, HELD_Y, Z_PLANE],
            "sphere_radii_m": [args.held_radius, args.moving_radius],
            "susceptibility": args.susceptibility,
            "saturation_magnetization_A_m": args.saturation_magnetization,
            "operating_moments_A_m2": [held_m, moving_m],
            "net_local_field_T_each": [0.0, args.bias_field, 0.0],
            "held_force_component_N": [-args.f_hold, args.f_hold],
            "moving_force_component_N": [-args.f_move, args.f_move],
            "coil_current_A": [I_MIN, I_MAX],
            "surface_clearance_m": args.clearance,
            "physical_center_separation_m": required_separation,
        },
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.output_plot)), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    save_plot(result, args.output_plot, required_separation, show=args.show)
    print(f"saved {args.output_json}")
    print(f"saved {args.output_plot}")
    return 0


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
