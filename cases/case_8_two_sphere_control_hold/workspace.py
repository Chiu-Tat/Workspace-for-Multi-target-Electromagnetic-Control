#!/usr/bin/env python3
"""Compute the Case 8 control-held two-sphere positional TFW slice.

Run from the repository root::

    python -m cases.case_8_two_sphere_control_hold.workspace --smoke
    python -m cases.case_8_two_sphere_control_hold.workspace

This is a reduced positional calculation under quasistatic, stable alignment
with the prescribed full 3-D net fields.  It is not a certification of the
orientation dynamics or closed-loop feedback stability.
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
    HELD_M,
    HELD_X,
    HELD_Y,
    I_MAX,
    I_MIN,
    MOVING_M,
    NUM_COILS,
    OUTPUT_LABELS,
    XY_LIMIT,
    Z_PLANE,
    SeparationError,
    affine_actuation_map,
    separation_bounds,
    stable_alignment_radius,
    task_bounds,
    task_vertices,
)


RESULTS_JSON = os.path.join("Results", "results_case8_two_sphere_control_hold.json")
RESULTS_PNG = os.path.join("Results", "results_case8_two_sphere_control_hold.png")
DEFAULT_EPS = 0.005


def _validate_options(f_hold, f_move, physical_min_separation, eps,
                      processes=None, max_iter=-1):
    task_bounds(f_hold, f_move)
    if not math.isfinite(physical_min_separation) or physical_min_separation < 0:
        raise ValueError("separation must be finite and nonnegative")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    if processes is not None and (
        not isinstance(processes, int) or isinstance(processes, bool) or processes < 1
    ):
        raise ValueError("processes must be a positive integer")
    if not isinstance(max_iter, int) or isinstance(max_iter, bool) or max_iter < -1:
        raise ValueError("max_iter must be -1 (unlimited) or a nonnegative integer")


def _width(box, axis):
    return float(box[axis][0].upper - box[axis][0].lower)


def _needs_bisection(box, eps):
    return _width(box, 0) > eps or _width(box, 1) > eps


def _box_to_list(box):
    return [
        float(box[0][0].lower), float(box[0][0].upper),
        float(box[1][0].lower), float(box[1][0].upper),
    ]


def classify_box(box, *, f_hold, f_move, physical_min_separation, eps):
    """Return ``(action, box, reason)`` for one magnet-2 position box."""

    x_interval, y_interval = box[0][0], box[1][0]
    stable_minimum = stable_alignment_radius(B_HOLD)
    required_separation = max(float(physical_min_separation), stable_minimum)
    minimum, maximum = separation_bounds(x_interval, y_interval)

    # Never evaluate a mutual dipole interval across the singular/excluded
    # region.  Boundary-straddling boxes are refined first.
    if maximum <= required_separation:
        return "excluded", box, "alignment_or_separation"
    if minimum <= required_separation:
        if _needs_bisection(box, eps):
            return "bisect", box, "separation_boundary"
        # A mixed leaf still contains admissible positions. Excluding the
        # whole rectangle would silently remove them from the workspace.
        return "undetermined", box, "separation_boundary_leaf"

    try:
        full_affine = affine_actuation_map(
            x_interval, y_interval, return_affine=True
        )
    except (SeparationError, ArithmeticError, ValueError, ZeroDivisionError):
        if _needs_bisection(box, eps):
            return "bisect", box, "singular_enclosure"
        return "undetermined", box, "singular_enclosure"

    G_affine = full_affine[:, :NUM_COILS]
    c_affine = full_affine[:, NUM_COILS]
    vertices = task_vertices(f_hold, f_move)
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


def determine_workspace(*, f_hold=F_HOLD, f_move=F_MOVE,
                        physical_min_separation=0.0, eps=DEFAULT_EPS,
                        processes=None, max_iter=-1):
    _validate_options(f_hold, f_move, physical_min_separation, eps, processes, max_iter)
    if processes is None:
        processes = min(mp.cpu_count(), 16)
    initial = Create2DBox(
        interval[-XY_LIMIT, XY_LIMIT], interval[-XY_LIMIT, XY_LIMIT]
    )
    pending = deque([initial])
    inside, outside, undetermined, excluded = [], [], [], []
    reasons = Counter()
    iterations = 0
    worker = partial(
        classify_box,
        f_hold=f_hold,
        f_move=f_move,
        physical_min_separation=physical_min_separation,
        eps=eps,
    )
    started = time.time()
    last_progress = 0

    # A single-worker run also works from notebooks and ordinary Python callers
    # without requiring a multiprocessing __main__ guard.
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
            elapsed = time.time() - started
            if iterations - last_progress >= 100 or not pending:
                print(
                    f"boxes {iterations} queue {len(pending)} "
                    f"in {len(inside)} out {len(outside)} "
                    f"und {len(undetermined)} excluded {len(excluded)} "
                    f"{elapsed:.1f}s",
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
        "control-held sphere", (HELD_X, HELD_Y), xytext=(7, 8),
        textcoords="offset points", fontsize=9,
    )
    ax.set_aspect("equal")
    ax.set_xlim(-XY_LIMIT, XY_LIMIT)
    ax.set_ylim(-XY_LIMIT, XY_LIMIT)
    ax.set_xlabel(r"moving sphere $x_2$ (m)")
    ax.set_ylabel(r"moving sphere $y_2$ (m)")
    suffix = " (partial run)" if result["pending"] else ""
    ax.set_title("Case 8: reduced positional TFW slice" + suffix)
    ax.grid(color="#e1ddd4", linewidth=0.5)
    ax.legend(handles=[
        Patch(facecolor="#3d995f", label="Certified inside"),
        Patch(facecolor="#d95f5f", label="Certified outside"),
        Patch(facecolor="#9e9e9e", label="Undetermined"),
        Patch(facecolor="#2f2f2f", label="Excluded by separation filter"),
        Patch(facecolor="#e6bc53", label="Pending (not evaluated)"),
    ], loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def _smoke(args):
    samples = [(-0.03, 0.02), (0.0, 0.03), (0.02, 0.0)]
    for x, y in samples:
        box = Create2DBox(interval[x, x], interval[y, y])
        action, _, reason = classify_box(
            box,
            f_hold=args.f_hold,
            f_move=args.f_move,
            physical_min_separation=args.min_separation,
            eps=args.eps,
        )
        print(f"({x:+.3f}, {y:+.3f}) -> {action:12s} {reason}")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--f-hold", type=float, default=F_HOLD)
    parser.add_argument("--f-move", type=float, default=F_MOVE)
    parser.add_argument(
        "--min-separation", type=float, default=0.0,
        help="physical center-separation limit in metres; alignment stability may impose a larger limit",
    )
    parser.add_argument("--eps", type=float, default=DEFAULT_EPS)
    parser.add_argument("--processes", type=int, default=None)
    parser.add_argument("--max-iter", type=int, default=-1)
    parser.add_argument("--output-json", default=RESULTS_JSON)
    parser.add_argument("--output-plot", default=RESULTS_PNG)
    parser.add_argument("--show", action="store_true", help="open the plot after saving")
    args = parser.parse_args(argv)
    try:
        _validate_options(args.f_hold, args.f_move, args.min_separation,
                          args.eps, args.processes, args.max_iter)
    except ValueError as exc:
        parser.error(str(exc))
    if args.smoke:
        return _smoke(args)

    stable_minimum = stable_alignment_radius(B_HOLD)
    required_separation = max(args.min_separation, stable_minimum)
    print("Case 8: two permanent-magnet spheres, one control-held")
    print(f"outputs: {', '.join(OUTPUT_LABELS)}")
    print(f"task force bounds: hold +/-{args.f_hold:g} N, move +/-{args.f_move:g} N")
    print(f"stable-alignment radius: {stable_minimum:.6f} m")
    print(f"effective exclusion radius: {required_separation:.6f} m")

    result = determine_workspace(
        f_hold=args.f_hold,
        f_move=args.f_move,
        physical_min_separation=args.min_separation,
        eps=args.eps,
        processes=args.processes,
        max_iter=args.max_iter,
    )
    payload = {
        "mode": "reduced_positional_tfw_control_hold",
        "inside_certificate": "affine/interval constructive right inverse",
        "outside_certificate": "dual support-function separation",
        "warning": (
            "Reduced positional result under quasistatic stable alignment; "
            "orientation dynamics and closed-loop stability require separate validation. "
            "Floating-point enclosures and certificates are not machine-verified."
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
            "moments_A_m2": [HELD_M, MOVING_M],
            "net_field_T_each": [0.0, B_HOLD, 0.0],
            "held_force_component_N": [-args.f_hold, args.f_hold],
            "moving_force_component_N": [-args.f_move, args.f_move],
            "coil_current_A": [I_MIN, I_MAX],
            "physical_min_separation_m": args.min_separation,
            "stable_alignment_radius_m": stable_minimum,
            "effective_exclusion_radius_m": required_separation,
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
