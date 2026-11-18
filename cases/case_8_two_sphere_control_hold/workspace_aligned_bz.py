#!/usr/bin/env python3
"""Case 8 TFW with continuously optimized aligned nonzero Bz fields."""

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
from matplotlib.patches import Patch

from emns.numerics.interval import interval
from emns.workspace.checks import BisectBox, Create2DBox
from hybrid_region.certificates import (
    affine_constructive_inside_certificate,
    dual_outside_certificate,
    interval_object_bounds,
)

from .model import (
    B_HOLD,
    HELD_M,
    HELD_X,
    HELD_Y,
    I_MAX,
    I_MIN,
    MOVING_M,
    NUM_COILS,
    XY_LIMIT,
    Z_PLANE,
    separation_bounds,
)
from .model_aligned_bz import (
    BZ_LIMIT,
    CONTROLLED_ROWS,
    F_HOLD,
    F_MOVE,
    affine_actuation_map,
    cone_affine_actuation_map,
    optimize_bz_pair,
    planar_task_vertices,
    task_vertices,
)
from .workspace import _box_to_list, _needs_bisection, _plot_rectangles


RESULTS_JSON = "Results/results_case8_aligned_bz.json"
RESULTS_PNG = "Results/results_case8_aligned_bz.png"
DEFAULT_EPS = 0.001
OPTIMIZE_WIDTH = 0.001


def _width(box, axis):
    return float(box[axis][0].upper - box[axis][0].lower)


def _validate(f_hold, f_move, bz_limit, physical_min_separation, eps, processes, max_iter):
    task_vertices(0.0, 0.0, f_hold, f_move)
    if not math.isfinite(bz_limit) or bz_limit < 0.0:
        raise ValueError("Bz limit must be finite and nonnegative")
    if not math.isfinite(physical_min_separation) or physical_min_separation < 0.0:
        raise ValueError("minimum separation must be finite and nonnegative")
    if not math.isfinite(eps) or eps <= 0.0:
        raise ValueError("eps must be finite and positive")
    if processes is not None and (
        not isinstance(processes, int) or isinstance(processes, bool) or processes < 1
    ):
        raise ValueError("processes must be a positive integer")
    if not isinstance(max_iter, int) or isinstance(max_iter, bool) or max_iter < -1:
        raise ValueError("max_iter must be -1 or a nonnegative integer")


def _inside_certificate(full_affine, vertices):
    return affine_constructive_inside_certificate(
        full_affine[:, :NUM_COILS],
        vertices,
        np.full(NUM_COILS, I_MIN),
        np.full(NUM_COILS, I_MAX),
        c_affine=full_affine[:, NUM_COILS],
    )


def classify_box(
    box, *, f_hold, f_move, bz_limit, physical_min_separation, eps,
    held_x=HELD_X, held_y=HELD_Y,
):
    x_interval, y_interval = box[0][0], box[1][0]
    # In this formulation moment/field alignment is a model assumption, so no
    # separate worst-case alignment-stability radius is imposed. Only a
    # user-specified physical centre separation is filtered.
    required_separation = physical_min_separation
    minimum, maximum = separation_bounds(x_interval, y_interval, held_x, held_y)
    if maximum <= required_separation:
        return "excluded", box, "physical_separation", None
    if minimum <= required_separation:
        if _needs_bisection(box, eps):
            return "bisect", box, "separation_boundary", None
        return "undetermined", box, "separation_boundary_leaf", None

    # Retain large boxes already certified in the original exact-Bz study.
    try:
        exact_map = affine_actuation_map(
            x_interval, y_interval, 0.0, 0.0,
            held_x=held_x, held_y=held_y, return_affine=True,
        )
        exact = _inside_certificate(
            exact_map, task_vertices(0.0, 0.0, f_hold, f_move)
        )
        if exact.certified:
            return "inside", box, "affine_exact_bz_fallback", (0.0, 0.0)
    except (ArithmeticError, ValueError, ZeroDivisionError, np.linalg.LinAlgError):
        pass

    # Delay nonlinear witness optimization until the spatial box is fine
    # enough for one fixed aligned field direction to remain useful throughout.
    if max(_width(box, 0), _width(box, 1)) > max(eps, OPTIMIZE_WIDTH):
        return "bisect", box, "defer_aligned_bz_optimization", None

    x_center = 0.5 * (float(x_interval.lower) + float(x_interval.upper))
    y_center = 0.5 * (float(y_interval.lower) + float(y_interval.upper))
    pair, point_peak = optimize_bz_pair(
        x_center,
        y_center,
        bz_limit=bz_limit,
        f_hold=f_hold,
        f_move=f_move,
        held_x=held_x,
        held_y=held_y,
    )
    if point_peak <= I_MAX * (1.0 + 1e-9):
        try:
            aligned_map = affine_actuation_map(
                x_interval, y_interval, pair[0], pair[1],
                held_x=held_x, held_y=held_y, return_affine=True,
            )
            aligned = _inside_certificate(
                aligned_map,
                task_vertices(pair[0], pair[1], f_hold, f_move),
            )
            if aligned.certified:
                return (
                    "inside", box, "affine_aligned_bz_witness",
                    (float(pair[0]), float(pair[1])),
                )
        except (ArithmeticError, ValueError, ZeroDivisionError, np.linalg.LinAlgError):
            pass

    # A robust cone over-encloses every permitted aligned direction. Proving a
    # planar task vertex unreachable in that relaxation is a valid outside proof.
    try:
        cone_map = cone_affine_actuation_map(
            x_interval, y_interval, bz_limit=bz_limit,
            held_x=held_x, held_y=held_y, return_affine=True,
        )
        cone_interval = np.asarray(
            [[entry.to_interval() for entry in row] for row in cone_map],
            dtype=object,
        )
        G_lower, G_upper = interval_object_bounds(
            cone_interval[CONTROLLED_ROWS, :NUM_COILS]
        )
        c_lower, c_upper = interval_object_bounds(
            cone_interval[CONTROLLED_ROWS, NUM_COILS]
        )
        outside = dual_outside_certificate(
            G_lower,
            G_upper,
            planar_task_vertices(f_hold, f_move),
            np.full(NUM_COILS, I_MIN),
            np.full(NUM_COILS, I_MAX),
            c_lower=c_lower,
            c_upper=c_upper,
        )
        if outside.certified:
            return "outside", box, "dual_separation_all_aligned_bz", None
    except (ArithmeticError, ValueError, ZeroDivisionError, np.linalg.LinAlgError):
        pass
    if _needs_bisection(box, eps):
        return "bisect", box, "none", None
    return "undetermined", box, "none", None


def determine_workspace(
    *,
    f_hold=F_HOLD,
    f_move=F_MOVE,
    bz_limit=BZ_LIMIT,
    physical_min_separation=0.0,
    eps=DEFAULT_EPS,
    processes=None,
    max_iter=-1,
    held_x=HELD_X,
    held_y=HELD_Y,
):
    _validate(
        f_hold, f_move, bz_limit, physical_min_separation,
        eps, processes, max_iter,
    )
    if processes is None:
        processes = min(mp.cpu_count(), 16)
    pending = deque([Create2DBox(
        interval[-XY_LIMIT, XY_LIMIT], interval[-XY_LIMIT, XY_LIMIT]
    )])
    inside, outside, undetermined, excluded, witnesses = [], [], [], [], []
    reasons = Counter()
    iterations = 0
    worker = partial(
        classify_box,
        f_hold=f_hold,
        f_move=f_move,
        bz_limit=bz_limit,
        physical_min_separation=physical_min_separation,
        eps=eps,
        held_x=held_x,
        held_y=held_y,
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
            for action, box, reason, witness in verdicts:
                if action == "inside":
                    inside.append(box)
                    witnesses.append((_box_to_list(box), witness))
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
                    f"excluded {len(excluded)} {time.time()-started:.1f}s",
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
        "witnesses": witnesses,
        "reasons": reasons,
        "iterations": iterations,
        "elapsed_s": time.time() - started,
    }


def save_plot(
    result, path, required_separation, bz_limit, *, held_x=HELD_X,
    held_y=HELD_Y, show=False,
):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 6.5))
    _plot_rectangles(ax, result["pending"], "#e6bc53", 0.55)
    _plot_rectangles(ax, result["outside"], "#d95f5f", 0.62)
    _plot_rectangles(ax, result["undetermined"], "#9e9e9e", 0.55)
    # Boxes excluded only by an optional physical separation are shown with
    # the same neutral color as unresolved leaves, not as a third TFW class.
    _plot_rectangles(ax, result["excluded"], "#9e9e9e", 0.55)
    _plot_rectangles(ax, result["inside"], "#3d995f", 0.72)
    ax.plot(held_x, held_y, "o", color="#171717", markersize=5)
    ax.annotate(
        "control-held sphere", (held_x, held_y), xytext=(7, 8),
        textcoords="offset points", fontsize=9,
    )
    ax.set_aspect("equal")
    ax.set_xlim(-XY_LIMIT, XY_LIMIT)
    ax.set_ylim(-XY_LIMIT, XY_LIMIT)
    ax.set_xlabel(r"moving sphere $x$ (m)")
    ax.set_ylabel(r"moving sphere $y$ (m)")
    suffix = " (partial run)" if result["pending"] else ""
    ax.set_title(
        f"Case 8: held at ({held_x:.2f}, {held_y:.2f}) m; "
        f"optimized $|B_z|\\leq${bz_limit*1e3:g} mT" + suffix
    )
    ax.grid(color="#e1ddd4", linewidth=0.5)
    ax.legend(handles=[
        Patch(facecolor="#3d995f", label="Certified inside"),
        Patch(facecolor="#d95f5f", label="Certified outside"),
        Patch(facecolor="#9e9e9e", label="Undetermined"),
        Patch(facecolor="#e6bc53", label="Pending (not evaluated)"),
    ], loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--f-hold", type=float, default=F_HOLD)
    parser.add_argument("--f-move", type=float, default=F_MOVE)
    parser.add_argument("--bz-limit", type=float, default=BZ_LIMIT)
    parser.add_argument("--min-separation", type=float, default=0.0)
    parser.add_argument("--held-x", type=float, default=HELD_X)
    parser.add_argument("--held-y", type=float, default=HELD_Y)
    parser.add_argument("--eps", type=float, default=DEFAULT_EPS)
    parser.add_argument("--processes", type=int, default=None)
    parser.add_argument("--max-iter", type=int, default=-1)
    parser.add_argument("--output-json", default=RESULTS_JSON)
    parser.add_argument("--output-plot", default=RESULTS_PNG)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args(argv)
    try:
        _validate(
            args.f_hold, args.f_move, args.bz_limit, args.min_separation,
            args.eps, args.processes, args.max_iter,
        )
    except ValueError as exc:
        parser.error(str(exc))

    required_separation = args.min_separation
    print("Case 8: optimized aligned nonzero Bz")
    print(f"Bz decision bounds: +/-{args.bz_limit:g} T")
    print(f"maximum field tilt: {math.degrees(math.atan2(args.bz_limit, B_HOLD)):.3f} deg")
    result = determine_workspace(
        f_hold=args.f_hold,
        f_move=args.f_move,
        bz_limit=args.bz_limit,
        physical_min_separation=args.min_separation,
        eps=args.eps,
        processes=args.processes,
        max_iter=args.max_iter,
        held_x=args.held_x,
        held_y=args.held_y,
    )
    payload = {
        "mode": "case8_optimized_aligned_bz",
        "inside_certificate": "fixed aligned-field affine right inverse per position box",
        "outside_certificate": "dual separation over all moment directions in the Bz cone",
        "warning": (
            "Each certified box uses one fixed optimized Bz pair for all force vertices. "
            "Orientation dynamics and feedback stability require separate validation. "
            "Floating-point enclosures are not machine-verified."
        ),
        "complete": result["complete"],
        "eps_m": args.eps,
        "iterations": result["iterations"],
        "elapsed_s": result["elapsed_s"],
        "certificate_counts": dict(result["reasons"]),
        "in_workspace": [_box_to_list(box) for box in result["inside"]],
        "inside_witnesses": [
            {"box": box, "Bz_T": list(witness)}
            for box, witness in result["witnesses"]
        ],
        "out_of_workspace": [_box_to_list(box) for box in result["outside"]],
        "undetermined": [_box_to_list(box) for box in result["undetermined"]],
        "excluded": [_box_to_list(box) for box in result["excluded"]],
        "pending": [_box_to_list(box) for box in result["pending"]],
        "protocol": {
            "plane_z_m": Z_PLANE,
            "xy_domain_m": [-XY_LIMIT, XY_LIMIT],
            "held_position_m": [args.held_x, args.held_y, Z_PLANE],
            "moments_A_m2": [HELD_M, MOVING_M],
            "By_T_each": B_HOLD,
            "Bz_decision_bounds_T_each": [-args.bz_limit, args.bz_limit],
            "maximum_tilt_deg": math.degrees(math.atan2(args.bz_limit, B_HOLD)),
            "held_force_component_N": [-args.f_hold, args.f_hold],
            "moving_force_component_N": [-args.f_move, args.f_move],
            "coil_current_A": [I_MIN, I_MAX],
            "effective_exclusion_radius_m": required_separation,
            "alignment_filter_applied": False,
        },
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.output_plot)), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    save_plot(
        result, args.output_plot, required_separation, args.bz_limit,
        held_x=args.held_x, held_y=args.held_y, show=args.show,
    )
    print(f"saved {args.output_json}")
    print(f"saved {args.output_plot}")
    return 0


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
