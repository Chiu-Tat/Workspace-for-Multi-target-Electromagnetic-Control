#!/usr/bin/env python3
# Run from the repository root:
#   python -m cases.case_6_interaction_obstacle_avoidance.workspace_hybrid [--layout gate|triangle]
"""Case 6 TFW with the affine-hybrid certificate cascade.

Same configuration space and task as ``workspace.py``:

  * free magnet in the plane ``Z = 0.082`` m, ``m = 0.12`` A m^2, heading
    ``beta``; search box ``(X, Y, beta)``
  * fixed coupling magnets from ``--layout`` (all in the same plane):

      single    (20, 0) mm, beta = pi/2, m = 0.145 A m^2   (original case)
      gate      (0, ±28) mm, beta = 0 / pi, m = 0.10 A m^2 each
      triangle  radius 30 mm at 90/210/330 deg, beta = 0 / 2pi/3 / 4pi/3,
                m = 0.075 A m^2 each

  * body-frame task ``[By, Bz, fy, fz] = [0, 0.02 T, 0, ±0.1 N]``
  * ten coils at ``±15`` A; dipole column pinned at 1.  The column is the sum
    of every fixed magnet's contribution, so the layouts share one column.

``workspace.py`` stops at Rohn (``Feasible``) and the contractor (``Out``).
This file uses the ``affine_hybrid`` cascade from ``hybrid_region``:

    RobustFeasible                 -> inside  / rohn
    affine right-inverse           -> inside  / affine_right_inverse
    interval right-inverse         -> inside  / right_inverse
    Out                            -> outside / contractor
    dual separation                -> outside / dual_separation
    otherwise                      -> undetermined, bisect
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import multiprocessing as mp
from collections import Counter
from functools import partial
from itertools import product

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from emns.numerics.interval import interval
from emns.workspace.affine import Map_I2H_Affine_body
from emns.workspace.checks import (
    Create3DBox,
    Out,
    RobustFeasible,
    create_I_intervals,
)
from hybrid_region.certificates import (
    affine_constructive_inside_certificate,
    box_vertices,
    constructive_inside_certificate,
    dual_outside_certificate,
    interval_object_bounds,
)

from cases.case_6_interaction_obstacle_avoidance.workspace import (
    BisectBox_Flexible,
    plot_cuboid,
)

Z_PLANE = 0.082
FREE_M = 0.12
ALPHA = math.pi / 2
B_HOLD = 0.02
F_MAX = 0.1
XY_LIM = 0.05
NUM_COILS = 10
EPS = [0.005, 0.005, 0.1]
I_MIN, I_MAX = -15.0, 15.0

_TRI_R = 0.030
# Fixed magnets per layout: (x m, y m, beta rad, m A m^2), all at Z_PLANE.
LAYOUTS = {
    "single": ((0.02, 0.0, math.pi / 2, 0.145),),
    "gate": (
        (0.0, 0.028, 0.0, 0.10),
        (0.0, -0.028, math.pi, 0.10),
    ),
    "triangle": (
        (0.0, _TRI_R, 0.0, 0.075),
        (-_TRI_R * math.sqrt(3) / 2, -_TRI_R / 2, 2 * math.pi / 3, 0.075),
        (_TRI_R * math.sqrt(3) / 2, -_TRI_R / 2, 4 * math.pi / 3, 0.075),
    ),
}
DEFAULT_LAYOUT = "single"

CHECKPOINT_VERSION = 1


def result_paths(layout=DEFAULT_LAYOUT):
    """(results JSON, figure PNG, checkpoint JSON); ``single`` keeps the original names."""

    suffix = "" if layout == DEFAULT_LAYOUT else f"_{layout}"
    stem = os.path.join("Results", f"results_case6_3D_hybrid{suffix}")
    return stem + ".json", stem + ".png", stem + ".checkpoint.json"



def _blank_point(X, Y, Z, m, alpha, beta):
    return {
        "X": X, "Y": Y, "Z": Z, "m": m, "alpha": alpha, "beta": beta,
        "Bx": None, "By": None, "Bz": None,
        "Bx_dx": None, "Bx_dy": None, "Bx_dz": None,
        "By_dy": None, "By_dz": None,
        "fx": None, "fy": None, "fz": None,
        "tx": None, "ty": None, "tz": None,
    }


def calculate_affine_A(X, Y, beta, *, return_affine=False,
                       fixed=LAYOUTS[DEFAULT_LAYOUT]):
    """Body-frame actuation map, same targets as ``workspace.calculate_affine_A``."""

    free = _blank_point(
        X, Y, interval[Z_PLANE, Z_PLANE],
        interval[FREE_M, FREE_M],
        interval[ALPHA, ALPHA], beta,
    )
    for component in ("By", "Bz", "fy", "fz"):
        free[component] = True
    magnets = [
        _blank_point(
            interval[x, x], interval[y, y],
            interval[Z_PLANE, Z_PLANE], interval[m, m],
            interval[ALPHA, ALPHA], interval[b, b],
        )
        for x, y, b, m in fixed
    ]
    return Map_I2H_Affine_body([free, *magnets], return_affine=return_affine)


def _as_interval_matrix(A_affine):
    return np.array(
        [[entry.to_interval() for entry in row] for row in A_affine],
        dtype=object,
    )


def _eps_list(eps, dimension=3):
    if isinstance(eps, (list, tuple, np.ndarray)):
        return list(eps)
    return [eps] * dimension


def _needs_bisection(box, eps):
    eps_list = _eps_list(eps, len(box))
    for axis, limit in enumerate(eps_list):
        width = box[axis][0][0][1] - box[axis][0][0][0]
        if width > limit:
            return True
    return False


def classify_box_hybrid(box, b_intervals, I_intervals, task_vertices,
                        current_lower, current_upper,
                        fixed=LAYOUTS[DEFAULT_LAYOUT]):
    """One ``affine_hybrid`` verdict for a Case 6 ``(X, Y, beta)`` box."""

    X, Y, beta = box[0][0], box[1][0], box[2][0]
    A_affine = calculate_affine_A(X, Y, beta, return_affine=True, fixed=fixed)
    A_interval = _as_interval_matrix(A_affine)

    if RobustFeasible(A_interval, I_intervals, b_intervals):
        return "inside", "rohn"

    G_affine = A_affine[:, :NUM_COILS]
    c_affine = A_affine[:, NUM_COILS]
    G_lower, G_upper = interval_object_bounds(A_interval[:, :NUM_COILS])
    c_lower, c_upper = interval_object_bounds(A_interval[:, NUM_COILS])

    affine_inside = affine_constructive_inside_certificate(
        G_affine, task_vertices, current_lower, current_upper,
        c_affine=c_affine,
    )
    if affine_inside.certified:
        return "inside", "affine_right_inverse"

    interval_inside = constructive_inside_certificate(
        G_lower, G_upper, task_vertices, current_lower, current_upper,
        c_lower=c_lower, c_upper=c_upper,
    )
    if interval_inside.certified:
        return "inside", "right_inverse"

    if Out(A_interval, I_intervals, b_intervals):
        return "outside", "contractor"

    dual_outside = dual_outside_certificate(
        G_lower, G_upper, task_vertices, current_lower, current_upper,
        c_lower=c_lower, c_upper=c_upper,
    )
    if dual_outside.certified:
        return "outside", "dual_separation"

    return "undetermined", "none"


def process_single_box(box, b_intervals, I_intervals, task_vertices,
                       current_lower, current_upper, eps,
                       fixed=LAYOUTS[DEFAULT_LAYOUT]):
    try:
        classification, reason = classify_box_hybrid(
            box, b_intervals, I_intervals, task_vertices,
            current_lower, current_upper, fixed,
        )
        if classification == "inside":
            return "feasible", box, reason
        if classification == "outside":
            return "out", box, reason
        if _needs_bisection(box, eps):
            return "bisect", box, reason
        return "undetermined", box, reason
    except Exception as exc:
        print(f"box failed: {exc}", flush=True)
        return "undetermined", box, "error"


def _progress_bar(done, pending, n_in, n_out, n_und, reasons, t0,
                  elapsed_offset=0.0):
    elapsed = elapsed_offset + time.time() - t0
    rate = done / elapsed if elapsed > 0 else 0.0
    reason_txt = " ".join(f"{name}:{count}" for name, count in sorted(reasons.items()))
    print(
        f"boxes {done}  queue {pending}  "
        f"in {n_in}  out {n_out}  undetermined {n_und}  "
        f"{elapsed:.0f}s  {rate:.2f}/s  {reason_txt}",
        flush=True,
    )


def Ws3DDetermination_Hybrid(
    InitBox, b_intervals, I_intervals, task_vertices,
    current_lower, current_upper, eps,
    max_iter=-1, num_processes=None, initial_state=None,
    checkpoint_path=None, checkpoint_every_s=30.0, layout=DEFAULT_LAYOUT,
):
    if num_processes is None:
        num_processes = min(mp.cpu_count(), 16)
    print(f"Using {num_processes} parallel processes")

    if initial_state is None:
        L = [InitBox]
        Lin, Lout, Lneg = [], [], []
        reasons = Counter()
        iteration = 0
        elapsed_offset = 0.0
    else:
        L = initial_state["pending"]
        Lin = initial_state["inside"]
        Lout = initial_state["outside"]
        Lneg = initial_state["undetermined"]
        reasons = initial_state["reasons"]
        iteration = initial_state["iteration"]
        elapsed_offset = initial_state["elapsed_s"]
        print(
            f"Resumed checkpoint at iteration {iteration} with "
            f"{len(L)} queued boxes"
        )
    t0 = time.time()
    last_checkpoint = time.monotonic()

    def save_checkpoint(force=False):
        nonlocal last_checkpoint
        if checkpoint_path is None:
            return
        if not force and time.monotonic() - last_checkpoint < checkpoint_every_s:
            return
        _save_checkpoint(
            checkpoint_path,
            pending=L,
            inside=Lin,
            outside=Lout,
            undetermined=Lneg,
            iteration=iteration,
            reasons=reasons,
            elapsed_s=elapsed_offset + time.time() - t0,
            eps=eps,
            layout=layout,
        )
        last_checkpoint = time.monotonic()
        print(f"Checkpoint saved to {checkpoint_path}", flush=True)

    worker = partial(
        process_single_box,
        b_intervals=b_intervals,
        I_intervals=I_intervals,
        task_vertices=task_vertices,
        current_lower=current_lower,
        current_upper=current_upper,
        eps=eps,
        fixed=LAYOUTS[layout],
    )

    with mp.Pool(processes=num_processes) as pool:
        while L:
            if len(L) >= num_processes:
                current_boxes = L[:num_processes]
                L = L[num_processes:]
            else:
                current_boxes = L
                L = []
            try:
                results = pool.map(worker, current_boxes)
            except BaseException:
                L = current_boxes + L
                save_checkpoint(force=True)
                raise

            to_bisect = []
            for result_type, box, reason in results:
                if result_type == "feasible":
                    Lin.append(box)
                    reasons[reason] += 1
                elif result_type == "out":
                    Lout.append(box)
                    reasons[reason] += 1
                elif result_type == "bisect":
                    to_bisect.append(box)
                else:
                    Lneg.append(box)
                    reasons[reason] += 1
            for box in to_bisect:
                L.extend(BisectBox_Flexible(box, eps))

            iteration += len(current_boxes)
            _progress_bar(iteration, len(L), len(Lin), len(Lout), len(Lneg),
                          reasons, t0, elapsed_offset)
            save_checkpoint()
            if max_iter >= 1 and iteration >= max_iter:
                print("Max iter reached")
                break

    save_checkpoint(force=True)
    elapsed = elapsed_offset + time.time() - t0
    return L, Lin, Lout, Lneg, iteration, reasons, elapsed


def _box_to_list(box):
    return [
        box[0][0][0][0], box[0][0][0][1],
        box[1][0][0][0], box[1][0][0][1],
        box[2][0][0][0], box[2][0][0][1],
    ]


def _list_to_box(values):
    if len(values) != 6:
        raise ValueError(f"Invalid checkpoint box with {len(values)} values")
    return Create3DBox(
        interval[values[0], values[1]],
        interval[values[2], values[3]],
        interval[values[4], values[5]],
    )


def _fixed_magnets_protocol(layout):
    """Fixed magnets as ``[x, y, z, m, beta]`` rows, the original field order."""

    return [[x, y, Z_PLANE, m, b] for x, y, b, m in LAYOUTS[layout]]


def _checkpoint_protocol(layout=DEFAULT_LAYOUT):
    return {
        "z_m": Z_PLANE,
        "free_m": FREE_M,
        "layout": layout,
        "fixed_magnets": _fixed_magnets_protocol(layout),
        "b_hold": B_HOLD,
        "f_max": F_MAX,
        "xy_lim": XY_LIM,
        "num_coils": NUM_COILS,
        "current_A": [I_MIN, I_MAX],
    }


def _save_checkpoint(path, *, pending, inside, outside, undetermined,
                     iteration, reasons, elapsed_s, eps, layout=DEFAULT_LAYOUT):
    """Atomically persist all state needed to continue the box search."""

    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    payload = {
        "version": CHECKPOINT_VERSION,
        "eps": list(eps),
        "protocol": _checkpoint_protocol(layout),
        "iteration": iteration,
        "elapsed_s": elapsed_s,
        "reasons": dict(reasons),
        "pending": [_box_to_list(box) for box in pending],
        "inside": [_box_to_list(box) for box in inside],
        "outside": [_box_to_list(box) for box in outside],
        "undetermined": [_box_to_list(box) for box in undetermined],
    }
    temporary_path = path + ".tmp"
    with open(temporary_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, separators=(",", ":"))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_path, path)


def _load_checkpoint(path, eps, layout=DEFAULT_LAYOUT):
    """Load and validate a checkpoint before reconstructing interval boxes."""

    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("version") != CHECKPOINT_VERSION:
        raise ValueError("Checkpoint version is incompatible")
    if payload.get("eps") != list(eps):
        raise ValueError("Checkpoint precision does not match this run")
    if payload.get("protocol") != _checkpoint_protocol(layout):
        raise ValueError("Checkpoint protocol does not match this run")
    return {
        "pending": [_list_to_box(box) for box in payload["pending"]],
        "inside": [_list_to_box(box) for box in payload["inside"]],
        "outside": [_list_to_box(box) for box in payload["outside"]],
        "undetermined": [
            _list_to_box(box) for box in payload["undetermined"]
        ],
        "iteration": int(payload["iteration"]),
        "elapsed_s": float(payload["elapsed_s"]),
        "reasons": Counter(payload["reasons"]),
    }


def _task_setup():
    I_intervals = create_I_intervals(11, custom_bounds={10: (1, 1)})
    b_intervals = [
        interval[0, 0],
        interval[B_HOLD, B_HOLD],
        interval[0, 0],
        interval[-F_MAX, F_MAX],
    ]
    task_lower = np.array([0.0, B_HOLD, 0.0, -F_MAX])
    task_upper = np.array([0.0, B_HOLD, 0.0, F_MAX])
    task_vertices = box_vertices(task_lower, task_upper)
    current_lower = np.full(NUM_COILS, I_MIN, dtype=float)
    current_upper = np.full(NUM_COILS, I_MAX, dtype=float)
    return I_intervals, b_intervals, task_vertices, current_lower, current_upper


def _save_plot(Lin, path, layout=DEFAULT_LAYOUT):
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    for box in Lin:
        origin = (box[0][0][0][0], box[1][0][0][0], box[2][0][0][0])
        size = (
            box[0][0][0][1] - origin[0],
            box[1][0][0][1] - origin[1],
            box[2][0][0][1] - origin[2],
        )
        plot_cuboid(ax, origin, size, "green")

    p_max = 2 * np.pi
    r = [0, p_max]
    for s, e in product(r, r):
        ax.plot([s, e], [r[0], r[0]], [r[0], r[0]], "k-", linewidth=1)
        ax.plot([s, e], [r[1], r[1]], [r[0], r[0]], "k-", linewidth=1)
        ax.plot([s, e], [r[0], r[0]], [r[1], r[1]], "k-", linewidth=1)
        ax.plot([s, e], [r[1], r[1]], [r[1], r[1]], "k-", linewidth=1)
    for s, e in product(r, r):
        ax.plot([r[0], r[0]], [s, e], [r[0], r[0]], "k-", linewidth=1)
        ax.plot([r[1], r[1]], [s, e], [r[0], r[0]], "k-", linewidth=1)
        ax.plot([r[0], r[0]], [s, e], [r[1], r[1]], "k-", linewidth=1)
        ax.plot([r[1], r[1]], [s, e], [r[1], r[1]], "k-", linewidth=1)
    for s, e in product(r, r):
        ax.plot([r[0], r[0]], [r[0], r[0]], [s, e], "k-", linewidth=1)
        ax.plot([r[1], r[1]], [r[0], r[0]], [s, e], "k-", linewidth=1)
        ax.plot([r[0], r[0]], [r[1], r[1]], [s, e], "k-", linewidth=1)
        ax.plot([r[1], r[1]], [r[1], r[1]], [s, e], "k-", linewidth=1)

    ax.axis("equal")
    ax.set_title(f"Case 6 TFW ({layout}), affine-hybrid cascade")
    ax.set_xlabel(r"$x$ (m)", fontsize=20, labelpad=10)
    ax.set_ylabel(r"$y$ (m)", fontsize=20, labelpad=10)
    ax.set_zlabel(r"$\beta$ (rad)", fontsize=20, labelpad=10)
    ax.tick_params(axis="both", which="major", labelsize=18)
    ax.legend(
        handles=[Patch(facecolor="green", alpha=0.4, label="In workspace")],
        loc="upper right", fontsize=16,
    )
    ax.set_xticks([-0.05, 0, 0.05], ["-0.05", "0", "0.05"])
    ax.set_yticks([-0.05, 0, 0.05], ["-0.05", "0", "0.05"])
    ax.set_zticks(
        [0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi],
        ["0", "π/2", "π", "3π/2", "2π"],
    )
    ax.set_xlim(-XY_LIM, XY_LIM)
    ax.set_ylim(-XY_LIM, XY_LIM)
    ax.set_zlim(0, 2 * math.pi)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Figure saved to {path}")
    if mpl.get_backend().lower() not in ("agg", "pdf", "svg", "template"):
        plt.show()
    plt.close(fig)


def _smoke(I_intervals, b_intervals, task_vertices, current_lower, current_upper,
           layout=DEFAULT_LAYOUT):
    """Classify a few representative boxes; do not run the full tree."""

    fixed = LAYOUTS[layout]
    fixed_x, fixed_y, fixed_beta, _ = fixed[0]
    # (0, 30) mm sits on a fixed magnet in gate/triangle; their centre is clear.
    free_y = 0.03 if layout == DEFAULT_LAYOUT else 0.0
    samples = {
        "point away from fixed magnets": Create3DBox(
            interval[0.0, 0.0], interval[free_y, free_y],
            interval[math.pi / 2, math.pi / 2],
        ),
        "3.125 mm leaf, same pose": Create3DBox(
            interval[0.0, 0.003125], interval[free_y, free_y + 0.003125],
            interval[math.pi / 2, math.pi / 2 + 0.05],
        ),
        "point on the fixed magnet": Create3DBox(
            interval[fixed_x, fixed_x], interval[fixed_y, fixed_y],
            interval[fixed_beta, fixed_beta],
        ),
    }
    print(f"Smoke classifications (affine-hybrid cascade, layout {layout})")
    print("=" * 80)
    for name, box in samples.items():
        classification, reason = classify_box_hybrid(
            box, b_intervals, I_intervals, task_vertices,
            current_lower, current_upper, fixed,
        )
        print(f"  {name:32s}  {classification:13s}  {reason}")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Case 6 workspace determination with the affine-hybrid cascade.",
    )
    parser.add_argument(
        "--layout", choices=sorted(LAYOUTS), default=DEFAULT_LAYOUT,
        help="Fixed-magnet placement (default: single, the original case).",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Classify a few boxes and exit; do not search the full domain.",
    )
    parser.add_argument("--max-iter", type=int, default=-1,
                        help="Stop after this many classified boxes.")
    parser.add_argument("--processes", type=int, default=None)
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from the checkpoint if it exists.",
    )
    parser.add_argument(
        "--checkpoint", default=None,
        help="Durable search checkpoint path (default: per layout under Results/).",
    )
    parser.add_argument(
        "--checkpoint-every", type=float, default=30.0,
        help="Seconds between atomic checkpoints (default: 30).",
    )
    args = parser.parse_args(argv)
    results_json, results_png, default_checkpoint = result_paths(args.layout)
    if args.checkpoint is None:
        args.checkpoint = default_checkpoint

    I_intervals, b_intervals, task_vertices, current_lower, current_upper = _task_setup()

    if args.smoke:
        return _smoke(
            I_intervals, b_intervals, task_vertices, current_lower, current_upper,
            args.layout,
        )

    print(f"Affine-hybrid workspace determination (Case 6 {args.layout}, 10 coils)")
    for x, y, b, m in LAYOUTS[args.layout]:
        print(f"  fixed magnet ({x * 1e3:.2f}, {y * 1e3:.2f}) mm  "
              f"beta {b:.4f} rad  m {m:g} A m^2")
    print("=" * 80)

    InitBox = Create3DBox(
        interval[-XY_LIM, XY_LIM],
        interval[-XY_LIM, XY_LIM],
        interval[0, 2 * math.pi],
    )
    eps = EPS
    initial_state = None
    if args.resume and os.path.exists(args.checkpoint):
        initial_state = _load_checkpoint(args.checkpoint, eps, args.layout)
        print(f"Loading checkpoint from {args.checkpoint}")
    elif args.resume:
        print(f"No checkpoint found at {args.checkpoint}; starting a new search")
    elif os.path.exists(args.checkpoint):
        parser.error(
            f"checkpoint exists at {args.checkpoint}; use --resume to continue it"
        )
    num_processes = args.processes
    if num_processes is None:
        num_processes = min(mp.cpu_count(), 16)

    print(f"core number: {mp.cpu_count()}")
    print(f"Running affine-hybrid determination with precision eps = {eps}")
    print(f"Using {num_processes} processes")
    print("This may take some time...")

    remaining, Lin, Lout, Lneg, iteration, reasons, elapsed = Ws3DDetermination_Hybrid(
        InitBox, b_intervals, I_intervals, task_vertices,
        current_lower, current_upper, eps,
        max_iter=args.max_iter, num_processes=num_processes,
        initial_state=initial_state,
        checkpoint_path=args.checkpoint,
        checkpoint_every_s=args.checkpoint_every,
        layout=args.layout,
    )
    print(f"Time taken: {elapsed:.2f} seconds")
    print(f"Completed after {iteration} iterations")
    print(f"Boxes in workspace: {len(Lin)}")
    print(f"Boxes out of workspace: {len(Lout)}")
    print(f"Undetermined boxes: {len(Lneg)}")
    print(f"Certificate counts: {dict(reasons)}")
    if remaining:
        print(f"Unclassified remaining boxes (max-iter): {len(remaining)}")

    print("Saving results...")
    os.makedirs("Results", exist_ok=True)
    results_data = {
        "mode": "affine_hybrid",
        "eps": eps,
        "elapsed_s": elapsed,
        "iterations": iteration,
        "certificate_counts": dict(reasons),
        "in_workspace": [_box_to_list(box) for box in Lin],
        "undetermined": [_box_to_list(box) for box in Lneg],
        "out_of_workspace": [_box_to_list(box) for box in Lout],
        "protocol": {
            "z_m": Z_PLANE,
            "free_m": FREE_M,
            "layout": args.layout,
            "fixed_magnets": _fixed_magnets_protocol(args.layout),
            "task": ["By=0", f"Bz={B_HOLD}", "fy=0", f"fz=±{F_MAX}"],
            "current_A": [I_MIN, I_MAX],
            "dipole_column_pinned": 1,
        },
    }
    with open(results_json, "w") as handle:
        json.dump(results_data, handle, indent=4)
    print(f"Results saved to {results_json}")

    print("Creating 3D plot...")
    _save_plot(Lin, results_png, args.layout)
    if not remaining and os.path.exists(args.checkpoint):
        os.remove(args.checkpoint)
        print(f"Removed completed checkpoint {args.checkpoint}")
    elif remaining:
        print(f"Checkpoint retained at {args.checkpoint}")
    print("Analysis complete!")
    return 0


if __name__ == "__main__":
    mp.freeze_support()
    sys.exit(main())
