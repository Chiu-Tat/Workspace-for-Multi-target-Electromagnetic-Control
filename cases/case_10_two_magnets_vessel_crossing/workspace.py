#!/usr/bin/env python3
"""Two-dimensional route TFW for two permanent magnets in crossing vessels.

The configuration is the pair of route coordinates ``(s1, s2)``, one Bezier
parameter per magnet.  A box is certified *inside* when the ten coils can hold
both prescribed 20-mT net fields and deliver every commanded tangential drive
force while the transverse wall loads stay inside their budget, for every
configuration in the box.

Run from the repository root::

    python -m cases.case_10_two_magnets_vessel_crossing.workspace --smoke
    python -m cases.case_10_two_magnets_vessel_crossing.workspace
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import multiprocessing as mp
import os
import time
from collections import Counter, deque
from contextlib import nullcontext
from functools import partial

import numpy as np

from emns.numerics.interval import interval
from emns.workspace.checks import BisectBox, Create2DBox
from hybrid_region.certificates import (
    affine_constructive_inside_certificate,
    dual_outside_certificate,
    interval_object_bounds,
)

from .model import (
    B_HOLD,
    F_DRIVE,
    F_SIDE,
    FRICTION_COEFFICIENT,
    I_MAX,
    I_MIN,
    LUMEN_SLACK,
    MAGNET_RADIUS,
    MOMENT,
    NET_WEIGHT,
    NUM_COILS,
    OUTPUT_LABELS,
    ROUTES,
    SIDE_LABELS,
    SeparationError,
    affine_actuation_map,
    cone_actuation_map,
    cone_task_vertices,
    direction_cells,
    numeric_basis,
    optimize_directions,
    peak_current,
    separation_bounds,
    side_bounds,
    stable_alignment_radius,
    task_vertices,
)

RESULTS_JSON = os.path.join("Results", "results_case10_two_magnets_vessel_crossing.json")
RESULTS_PNG = os.path.join("Results", "results_case10_two_magnets_vessel_crossing.png")
DEFAULT_EPS = 0.01
OPTIMIZE_WIDTH = 0.0625
MAX_TANGENT_DEG = 5.0
HINT_REUSE_FRACTION = 0.75
OUTSIDE_SCREEN_FACTOR = 1.5


def _width(box, axis):
    return float(box[axis][0].upper - box[axis][0].lower)


def _centre(box, axis):
    return 0.5 * float(box[axis][0].upper + box[axis][0].lower)


def _box_to_list(box):
    return [
        float(box[0][0].lower), float(box[0][0].upper),
        float(box[1][0].lower), float(box[1][0].upper),
    ]


def _needs_bisection(box, eps):
    return _width(box, 0) > eps or _width(box, 1) > eps


def _inside_certificate(full, side, vertices, f_side):
    lower, upper = side_bounds(f_side)
    return affine_constructive_inside_certificate(
        full[:, :NUM_COILS],
        vertices,
        np.full(NUM_COILS, I_MIN),
        np.full(NUM_COILS, I_MAX),
        c_affine=full[:, NUM_COILS],
        side_G_affine=side[:, :NUM_COILS],
        side_c_affine=side[:, NUM_COILS],
        side_lower=lower,
        side_upper=upper,
    )


def _try_inside(box, pair, *, f_drive, f_side, slack):
    try:
        full, side = affine_actuation_map(
            box[0][0], box[1][0], pair[0], pair[1], slack=slack
        )
    except (SeparationError, ArithmeticError, ValueError, ZeroDivisionError,
            np.linalg.LinAlgError):
        return None
    try:
        return _inside_certificate(
            full, side, task_vertices(pair[0], pair[1], f_drive), f_side
        )
    except (ArithmeticError, ValueError, np.linalg.LinAlgError):
        return None


def _outside_over_cells(box, cell_pairs, *, f_drive, f_side, slack):
    """Certify every supplied direction-cell pair as infeasible."""

    vertices = cone_task_vertices(f_drive)
    for cell_1, cell_2 in cell_pairs:
        try:
            full, lower, upper = cone_actuation_map(
                box[0][0], box[1][0], cell_1, cell_2,
                slack=slack, f_side=f_side,
            )
        except (SeparationError, ArithmeticError, ValueError, ZeroDivisionError):
            return False
        bounds = np.asarray(
            [[entry.to_interval() for entry in row] for row in full], dtype=object
        )
        columns = full.shape[1] - 1
        G_lower, G_upper = interval_object_bounds(bounds[:, :columns])
        c_lower, c_upper = interval_object_bounds(bounds[:, columns])
        try:
            certificate = dual_outside_certificate(
                G_lower, G_upper, vertices, lower, upper,
                c_lower=c_lower, c_upper=c_upper,
            )
        except (ArithmeticError, ValueError, np.linalg.LinAlgError):
            return False
        if not certificate.certified:
            return False
    return True


def _cell_centre(cell):
    axis, sign, x_range, y_range = cell
    return (axis, sign,
            (0.5 * sum(x_range),) * 2,
            (0.5 * sum(y_range),) * 2)


def classify_box(item, *, f_drive, f_side, eps, min_separation, slack,
                 max_tangent_deg, cover_divisions, sample_divisions):
    """Return ``(action, box, reason, witness, diagnostic)`` for one route box."""

    box, hint = item
    s1_interval, s2_interval = box[0][0], box[1][0]
    required = max(min_separation, stable_alignment_radius(B_HOLD))
    minimum, maximum = separation_bounds(s1_interval, s2_interval)
    if maximum <= required:
        return "excluded", box, "separation_or_alignment", None, None
    if minimum <= required:
        if _needs_bisection(box, eps):
            return "bisect", box, "separation_boundary", hint, None
        return "undetermined", box, "separation_boundary_leaf", None, None

    # The commanded force frame is the centerline frame at the box centre, so
    # the box is refined until every tangent inside it is nearly that frame.
    spread = max(
        ROUTES[0].tangent_spread_deg(s1_interval),
        ROUTES[1].tangent_spread_deg(s2_interval),
    )
    if spread > max_tangent_deg:
        return "bisect", box, "tangent_frame_refinement", hint, None

    leaf = not _needs_bisection(box, eps)
    basis = None
    best = None
    if hint is not None:
        certificate = _try_inside(box, hint, f_drive=f_drive, f_side=f_side,
                                  slack=slack)
        if certificate is not None and certificate.certified:
            return "inside", box, "inherited_direction_witness", hint, None
        # An inherited witness that is still comfortably inside the current
        # limit is worth carrying further instead of paying for a new search:
        # the certificate then failed on box width, not on the direction.
        if not leaf:
            basis = numeric_basis(_centre(box, 0), _centre(box, 1))
            best = peak_current(basis, hint[0], hint[1],
                                f_drive=f_drive, f_side=f_side)
            if best <= HINT_REUSE_FRACTION * I_MAX:
                return "bisect", box, "reuse_direction_witness", hint, best

    if not leaf and max(_width(box, 0), _width(box, 1)) > OPTIMIZE_WIDTH:
        return "bisect", box, "defer_direction_search", hint, None

    if basis is None:
        basis = numeric_basis(_centre(box, 0), _centre(box, 1))
    pair, best = optimize_directions(
        basis, f_drive=f_drive, f_side=f_side, hint=hint
    )
    if pair is not None and best <= I_MAX * (1.0 + 1e-9):
        certificate = _try_inside(box, pair, f_drive=f_drive, f_side=f_side,
                                  slack=slack)
        if certificate is not None and certificate.certified:
            return "inside", box, "optimized_direction_witness", pair, best
    if pair is None:
        pair = hint

    if not leaf:
        return "bisect", box, "unresolved", pair, best

    # Leaf tests, most expensive last, and only where the point-wise allocation
    # is far outside the current limit.  The cover is a complete tiling of both
    # direction spheres, so success there is a proof for every admissible
    # orientation; the sampled variant only proves it for the listed pairs.
    if best <= OUTSIDE_SCREEN_FACTOR * I_MAX:
        return "undetermined", box, "near_boundary_no_certificate", None, best
    if cover_divisions:
        cells_1 = direction_cells(cover_divisions, half_sphere=True)
        cells_2 = direction_cells(cover_divisions)
        if _outside_over_cells(box, itertools.product(cells_1, cells_2),
                               f_drive=f_drive, f_side=f_side, slack=slack):
            return "outside", box, "dual_separation_all_directions", None, best
    if sample_divisions:
        cells_1 = [_cell_centre(cell)
                   for cell in direction_cells(sample_divisions, half_sphere=True)]
        cells_2 = [_cell_centre(cell)
                   for cell in direction_cells(sample_divisions)]
        if _outside_over_cells(box, itertools.product(cells_1, cells_2),
                               f_drive=f_drive, f_side=f_side, slack=slack):
            return "sampled_outside", box, "dual_separation_sampled_directions", \
                None, best
    return "undetermined", box, "no_certificate", None, best


def determine_workspace(*, f_drive=F_DRIVE, f_side=F_SIDE, eps=DEFAULT_EPS,
                        min_separation=0.0, slack=(0.0, 0.0),
                        max_tangent_deg=MAX_TANGENT_DEG, cover_divisions=1,
                        sample_divisions=2, processes=None, max_iter=-1):
    _validate(f_drive, f_side, eps, min_separation, slack, max_tangent_deg,
              processes, max_iter)
    if processes is None:
        processes = min(mp.cpu_count(), 16)
    pending = deque([(Create2DBox(interval[0.0, 1.0], interval[0.0, 1.0]), None)])
    results = {"inside": [], "outside": [], "sampled_outside": [],
               "undetermined": [], "excluded": []}
    witnesses, diagnostics = [], []
    reasons = Counter()
    worker = partial(
        classify_box,
        f_drive=f_drive, f_side=f_side, eps=eps, min_separation=min_separation,
        slack=tuple(slack), max_tangent_deg=max_tangent_deg,
        cover_divisions=cover_divisions, sample_divisions=sample_divisions,
    )
    started = time.time()
    iterations = 0
    last_report = 0
    context = (mp.Pool(processes=processes)
               if processes > 1 and max_iter != 0 else nullcontext())
    with context as pool:
        while pending and (max_iter < 0 or iterations < max_iter):
            batch_size = min(4 * processes, len(pending))
            if max_iter >= 0:
                batch_size = min(batch_size, max_iter - iterations)
            batch = [pending.popleft() for _ in range(batch_size)]
            verdicts = pool.map(worker, batch) if pool is not None else map(worker, batch)
            for action, box, reason, witness, diagnostic in verdicts:
                reasons[reason] += 1
                if action == "bisect":
                    pending.extend((child, witness) for child in BisectBox(box))
                    continue
                results[action].append(box)
                if diagnostic is not None and math.isfinite(diagnostic):
                    diagnostics.append((_box_to_list(box), float(diagnostic)))
                if action == "inside":
                    witnesses.append((
                        _box_to_list(box),
                        [list(map(float, witness[0])), list(map(float, witness[1]))],
                    ))
            iterations += len(batch)
            if iterations - last_report >= 200 or not pending:
                print(
                    f"boxes {iterations} queue {len(pending)} "
                    f"in {len(results['inside'])} out {len(results['outside'])} "
                    f"sampled {len(results['sampled_outside'])} "
                    f"und {len(results['undetermined'])} "
                    f"{time.time() - started:.1f}s",
                    flush=True,
                )
                last_report = iterations
    results["pending"] = [box for box, _ in pending]
    results["complete"] = not pending
    results["witnesses"] = witnesses
    results["diagnostics"] = diagnostics
    results["reasons"] = reasons
    results["iterations"] = iterations
    results["elapsed_s"] = time.time() - started
    return results


def _validate(f_drive, f_side, eps, min_separation, slack, max_tangent_deg,
              processes, max_iter):
    task_vertices([0.0, 0.0, 1.0], [0.0, 0.0, 1.0], f_drive)
    side_bounds(f_side)
    if not math.isfinite(eps) or not 0.0 < eps <= 1.0:
        raise ValueError("eps must lie in (0, 1]")
    if not math.isfinite(min_separation) or min_separation < 0.0:
        raise ValueError("minimum separation must be finite and nonnegative")
    if len(slack) != 2 or any(
        not math.isfinite(value) or value < 0.0 for value in slack
    ):
        raise ValueError("radial slack must be two finite nonnegative metres")
    for index, value in enumerate(slack):
        if value > LUMEN_SLACK[index] + 1e-12:
            raise ValueError(
                f"radial slack {value} exceeds the lumen clearance "
                f"{LUMEN_SLACK[index]} of magnet {index + 1}"
            )
    if not math.isfinite(max_tangent_deg) or max_tangent_deg <= 0.0:
        raise ValueError("tangent spread limit must be positive")
    if processes is not None and (
        not isinstance(processes, int) or isinstance(processes, bool) or processes < 1
    ):
        raise ValueError("processes must be a positive integer")
    if not isinstance(max_iter, int) or isinstance(max_iter, bool) or max_iter < -1:
        raise ValueError("max_iter must be -1 or a nonnegative integer")


def _plot_boxes(axis, boxes, color, alpha):
    from matplotlib.patches import Rectangle

    for box in boxes:
        lower_1, upper_1 = float(box[0][0].lower), float(box[0][0].upper)
        lower_2, upper_2 = float(box[1][0].lower), float(box[1][0].upper)
        axis.add_patch(Rectangle(
            (lower_1, lower_2), upper_1 - lower_1, upper_2 - lower_2,
            facecolor=color, edgecolor="none", alpha=alpha,
        ))


def save_plot(result, path, *, f_drive, f_side, show=False):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    figure, axis = plt.subplots(figsize=(7.4, 6.6))
    _plot_boxes(axis, result.get("pending", []), "#e6bc53", 0.55)
    _plot_boxes(axis, result["undetermined"], "#9e9e9e", 0.5)
    _plot_boxes(axis, result["excluded"], "#9e9e9e", 0.5)
    _plot_boxes(axis, result["sampled_outside"], "#e08a3c", 0.75)
    _plot_boxes(axis, result["outside"], "#d95f5f", 0.75)
    _plot_boxes(axis, result["inside"], "#3d995f", 0.72)

    grid = np.linspace(0.0, 1.0, 161)
    points_1 = np.asarray([ROUTES[0].point(value) for value in grid])
    points_2 = np.asarray([ROUTES[1].point(value) for value in grid])
    distance = np.linalg.norm(
        points_1[:, None, :] - points_2[None, :, :], axis=-1
    ) * 1e3
    contours = axis.contour(
        grid, grid, distance.T, levels=[5, 6, 8, 12, 20],
        colors="#2b2b2b", linewidths=0.6, alpha=0.5,
    )
    axis.clabel(contours, inline=True, fontsize=7, fmt="%g mm")

    axis.set_aspect("equal")
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.set_xlabel("magnet 1 route coordinate $s_1$ (artery-side lumen)")
    axis.set_ylabel("magnet 2 route coordinate $s_2$ (vein-side lumen)")
    suffix = " (partial run)" if result.get("pending") else ""
    axis.set_title(
        f"Case 10 route TFW: drive $\\pm${f_drive*1e3:g} mN, "
        f"transverse budget $\\pm${f_side*1e3:g} mN, "
        f"$|B|$ = {B_HOLD*1e3:g} mT" + suffix
    )
    axis.legend(handles=[
        Patch(facecolor="#3d995f", label="Certified inside"),
        Patch(facecolor="#d95f5f", label="Certified outside (all directions)"),
        Patch(facecolor="#e08a3c", label="Outside for every sampled direction"),
        Patch(facecolor="#9e9e9e", label="Undetermined"),
        Patch(facecolor="#e6bc53", label="Pending"),
    ], loc="upper left", fontsize=8, framealpha=0.92)
    axis.grid(color="#e1ddd4", linewidth=0.4)
    figure.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(figure)


def _area(boxes):
    return float(sum(
        (float(box[0][0].upper) - float(box[0][0].lower))
        * (float(box[1][0].upper) - float(box[1][0].lower))
        for box in boxes
    ))


def smoke(f_drive=F_DRIVE, f_side=F_SIDE):
    """Three representative point classifications, without branch and bound."""

    print("route coordinate   distance   best peak current   worst-case use")
    for s1, s2 in ((0.1, 0.1), (0.2, 0.8), (0.5, 0.5)):
        basis = numeric_basis(s1, s2)
        pair, best = optimize_directions(basis, f_drive=f_drive, f_side=f_side)
        distance = np.linalg.norm(ROUTES[0].point(s1) - ROUTES[1].point(s2))
        verdict = "within +/-15 A" if best <= I_MAX else "exceeds +/-15 A"
        angle = math.degrees(math.acos(float(np.clip(pair[0] @ pair[1], -1.0, 1.0))))
        print(f"  ({s1:.2f}, {s2:.2f})     {distance*1e3:6.2f} mm     "
              f"{best:9.2f} A        {verdict}; field directions {angle:.1f} deg apart")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--f-drive", type=float, default=F_DRIVE)
    parser.add_argument("--f-side", type=float, default=F_SIDE)
    parser.add_argument("--eps", type=float, default=DEFAULT_EPS)
    parser.add_argument("--min-separation", type=float, default=0.0)
    parser.add_argument("--radial-slack", type=float, nargs=2, default=(0.0, 0.0),
                        help="metres of cross-section offset covered per magnet")
    parser.add_argument("--max-tangent-deg", type=float, default=MAX_TANGENT_DEG)
    parser.add_argument("--cover-divisions", type=int, default=1,
                        help="cube-face divisions of the complete direction cover")
    parser.add_argument("--sample-divisions", type=int, default=2,
                        help="cube-face divisions of the sampled direction menu")
    parser.add_argument("--processes", type=int, default=None)
    parser.add_argument("--max-iter", type=int, default=-1)
    parser.add_argument("--output-json", default=RESULTS_JSON)
    parser.add_argument("--output-plot", default=RESULTS_PNG)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args(argv)
    if args.smoke:
        return smoke(args.f_drive, args.f_side)
    try:
        _validate(args.f_drive, args.f_side, args.eps, args.min_separation,
                  args.radial_slack, args.max_tangent_deg, args.processes,
                  args.max_iter)
    except ValueError as error:
        parser.error(str(error))

    friction = FRICTION_COEFFICIENT * math.sqrt(2.0) * args.f_side
    print("Case 10: two permanent magnets on crossing vessel routes")
    print(f"moments {MOMENT[0]*1e3:.2f} / {MOMENT[1]*1e3:.2f} mA m^2, "
          f"net weights {NET_WEIGHT[0]*1e3:.3f} / {NET_WEIGHT[1]*1e3:.3f} mN")
    print(f"alignment-stability separation {stable_alignment_radius()*1e3:.2f} mm")
    print(f"worst-case Coulomb friction {friction*1e3:.3f} mN against a "
          f"{args.f_drive*1e3:.2f} mN drive")
    result = determine_workspace(
        f_drive=args.f_drive, f_side=args.f_side, eps=args.eps,
        min_separation=args.min_separation, slack=tuple(args.radial_slack),
        max_tangent_deg=args.max_tangent_deg,
        cover_divisions=args.cover_divisions,
        sample_divisions=args.sample_divisions,
        processes=args.processes, max_iter=args.max_iter,
    )
    payload = {
        "mode": "case10_route_tfw",
        "complete": result["complete"],
        "iterations": result["iterations"],
        "elapsed_s": result["elapsed_s"],
        "certificate_counts": dict(result["reasons"]),
        "area_fraction": {
            key: _area(result[key])
            for key in ("inside", "outside", "sampled_outside",
                        "undetermined", "excluded")
        },
        "in_workspace": [_box_to_list(box) for box in result["inside"]],
        "out_of_workspace": [_box_to_list(box) for box in result["outside"]],
        "sampled_out_of_workspace": [
            _box_to_list(box) for box in result["sampled_outside"]
        ],
        "undetermined": [_box_to_list(box) for box in result["undetermined"]],
        "excluded": [_box_to_list(box) for box in result["excluded"]],
        "pending": [_box_to_list(box) for box in result["pending"]],
        "inside_witnesses": [
            {"box": box, "field_directions": directions}
            for box, directions in result["witnesses"]
        ],
        "peak_current_diagnostics": [
            {"box": box, "peak_current_A": value}
            for box, value in result["diagnostics"]
        ],
        "protocol": {
            "configuration": "route coordinates (s1, s2) of the two centerlines",
            "controlled_rows": list(OUTPUT_LABELS),
            "side_rows": list(SIDE_LABELS),
            "field_magnitude_T": B_HOLD,
            "field_direction": "free unit vector, one witness per certified box",
            "drive_force_N": [-args.f_drive, args.f_drive],
            "transverse_budget_N": args.f_side,
            "friction_coefficient": FRICTION_COEFFICIENT,
            "coil_current_A": [I_MIN, I_MAX],
            "moments_A_m2": list(MOMENT),
            "magnet_radius_m": list(MAGNET_RADIUS),
            "net_weight_N": list(NET_WEIGHT),
            "radial_slack_m": list(args.radial_slack),
            "alignment_radius_m": stable_alignment_radius(),
            "eps_route": args.eps,
            "max_tangent_spread_deg": args.max_tangent_deg,
            "geometry": os.path.join(
                "cases", "case_10_two_magnets_vessel_crossing", "geometry.json"
            ),
        },
        "warning": (
            "Inside boxes are proofs for the stated task with one fixed aligned "
            "field direction per box. Outside boxes are proofs over a complete "
            "direction cover. Sampled-outside boxes are proofs only for the "
            "listed direction pairs, not for the direction continuum. "
            "Orientation dynamics, flow and feedback stability are not covered, "
            "and the floating-point enclosures are not machine verified."
        ),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    save_plot(result, args.output_plot, f_drive=args.f_drive,
              f_side=args.f_side, show=args.show)
    print(f"saved {args.output_json}")
    print(f"saved {args.output_plot}")
    return 0


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
