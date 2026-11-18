#!/usr/bin/env python3
"""Reproducible point-dipole versus finite-body Case-6 TFW cross-section.

The finite-body reference distributes each prescribed permanent magnetic moment
over a cylindrical tensor quadrature.  Pairwise dipole-element interactions are
summed, including both source and receiver extent.  This is a convergent
numerical reference, not a closed-form or experimentally validated truth model.
The two interaction offsets are inserted directly into the same coil map and
task-containment test. This exposes both optimistic and conservative dipole
classifications without turning their difference into an arbitrary reserve.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from functools import lru_cache
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linprog

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "results"
sys.path.insert(0, str(ROOT))

from cases.case_6_interaction_obstacle_avoidance.simulation import model

MU04PI = 1e-7


@lru_cache(maxsize=32)
def cylinder_points(length, diameter, nr, nphi, nz):
    """Equal-volume midpoint quadrature in a cylinder aligned with local z."""
    pts, weights = [], []
    radius = diameter / 2
    for iz in range(nz):
        z = length * ((iz + 0.5) / nz - 0.5)
        for ir in range(nr):
            # Equal-area annuli; midpoint in squared radius.
            rho = radius * math.sqrt((ir + 0.5) / nr)
            for ip in range(nphi):
                phi = 2 * math.pi * (ip + 0.5) / nphi
                pts.append([rho * math.cos(phi), rho * math.sin(phi), z])
                weights.append(1.0 / (nr * nphi * nz))
    return np.asarray(pts), np.asarray(weights)


def rotation_from_z(beta):
    # Local cylinder/magnetization z-axis maps to in-plane heading.
    z = np.array([math.cos(beta), math.sin(beta), 0.0])
    y = np.array([-math.sin(beta), math.cos(beta), 0.0])
    x = np.cross(y, z)
    return np.column_stack((x, y, z))


def finite_interaction(x, y, beta, cfg):
    q = cfg["reference_quadrature"]
    gm, gf = cfg["moving_cylinder"], cfg["fixed_cylinder"]
    pm, wm = cylinder_points(gm["length_m"], gm["diameter_m"], q["radial"], q["azimuthal"], q["axial"])
    pf, wf = cylinder_points(gf["length_m"], gf["diameter_m"], q["radial"], q["azimuthal"], q["axial"])
    Rm, Rf = rotation_from_z(beta), rotation_from_z(model.FIXED_BETA)
    pm = pm @ Rm.T + np.array([x, y, model.Z_PLANE])
    pf = pf @ Rf.T + np.array([model.FIXED_X, model.FIXED_Y, model.Z_PLANE])
    um, uf = Rm[:, 2], Rf[:, 2]
    mm = model.FREE_M * wm[:, None] * um
    mf = model.FIXED_M * wf[:, None] * uf

    # Receiver-element x source-element x xyz.
    r = pm[:, None, :] - pf[None, :, :]
    rn = np.linalg.norm(r, axis=2)
    rhat = r / rn[:, :, None]
    dotf = np.einsum("mnj,nj->mn", rhat, mf)
    B = MU04PI * (3 * rhat * dotf[:, :, None] - mf[None, :, :]) / rn[:, :, None] ** 3
    B_at_receiver = B.sum(axis=1)
    B_avg = np.sum(wm[:, None] * B_at_receiver, axis=0)

    dotm = np.einsum("mnj,mj->mn", rhat, mm)
    mdot = np.einsum("mj,nj->mn", mm, mf)
    force_pairs = (3 * MU04PI / rn[:, :, None] ** 4) * (
        dotm[:, :, None] * mf[None, :, :] + dotf[:, :, None] * mm[:, None, :]
        + mdot[:, :, None] * rhat - 5 * (dotm * dotf)[:, :, None] * rhat
    )
    F = force_pairs.sum(axis=(0, 1))
    by, bz = Rm[:, 1], Rm[:, 2]
    return np.array([B_avg @ by, B_avg @ bz, F @ by, F @ bz])


def dipole_interaction(x, y, beta):
    A = model.actuation_matrix_body(x, y, beta)
    return A[:, model.NUM_COILS]


def contains_box(Acoil, center_rhs, halfwidth, current_limit):
    """For every vertex of an output box, test existence of bounded currents."""
    for signs in product((-1.0, 1.0), repeat=4):
        rhs = center_rhs + halfwidth * np.asarray(signs)
        res = linprog(np.zeros(model.NUM_COILS), A_eq=Acoil, b_eq=rhs,
                      bounds=[(-current_limit, current_limit)] * model.NUM_COILS,
                      method="highs")
        if not res.success:
            return False
    return True


def run(cfg):
    n = cfg["grid_points_per_axis"]
    xs = np.linspace(*cfg["x_limits_m"], n)
    ys = np.linspace(*cfg["y_limits_m"], n)
    beta = cfg["moving_heading_rad"]
    dipole_mask = np.zeros((n, n), bool)
    finite_mask = np.zeros_like(dipole_mask)
    valid = np.zeros_like(dipole_mask)
    signed_discrepancy = np.full((n, n, 4), np.nan)
    radii = []
    gm, gf = cfg["moving_cylinder"], cfg["fixed_cylinder"]
    exclusion = 0.5 * math.hypot(gm["length_m"], gm["diameter_m"]) + 0.5 * math.hypot(gf["length_m"], gf["diameter_m"])

    task_center = np.array([0.0, model.B_HOLD, 0.0, 0.0])
    task_half = np.array([0.0, 0.0, 0.0, model.F_MAX])
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            sep = math.hypot(x - model.FIXED_X, y - model.FIXED_Y)
            radii.append(sep)
            if sep < exclusion:
                continue
            valid[j, i] = True
            Ahat = model.actuation_matrix_body(x, y, beta)
            Acoil, cdip = Ahat[:, :model.NUM_COILS], Ahat[:, model.NUM_COILS]
            dipole_mask[j, i] = contains_box(Acoil, task_center - cdip, task_half, cfg["current_limit_A"])
            cfinite = finite_interaction(x, y, beta, cfg)
            signed_discrepancy[j, i] = cfinite - cdip
            finite_mask[j, i] = contains_box(Acoil, task_center - cfinite, task_half, cfg["current_limit_A"])

    dx, dy = (xs[-1] - xs[0]) / (n - 1), (ys[-1] - ys[0]) / (n - 1)
    wx, wy = np.ones(n), np.ones(n)
    wx[[0, -1]] = 0.5; wy[[0, -1]] = 0.5
    area_weights = np.outer(wy, wx) * dx * dy
    common = valid.sum()
    # A limited convergence diagnostic: this measures quadrature sensitivity,
    # but is not promoted to a rigorous remainder bound.
    convergence = []
    fine_cfg = json.loads(json.dumps(cfg))
    fine_cfg["reference_quadrature"] = {"radial": 5, "azimuthal": 24, "axial": 7}
    disagreement_idx = np.argwhere((dipole_mask != finite_mask) & valid)
    check_poses = [(-0.03, -0.03)]
    if len(disagreement_idx):
        # Include the closest disagreement and up to two other boundary samples.
        ordered = sorted(disagreement_idx, key=lambda ji: math.hypot(xs[ji[1]] - model.FIXED_X, ys[ji[0]] - model.FIXED_Y))
        for ji in (ordered[0], ordered[len(ordered)//2], ordered[-1]):
            check_poses.append((float(xs[ji[1]]), float(ys[ji[0]])))
    for x, y in check_poses:
        coarse = finite_interaction(x, y, beta, cfg)
        fine = finite_interaction(x, y, beta, fine_cfg)
        convergence.append({"x_m": x, "y_m": y, "absolute_change": dict(zip(model.ROW_LABELS, np.abs(fine - coarse).tolist()))})

    # Point-dipole limiting sanity test with both cylinders shrunk by 100x.
    tiny_cfg = json.loads(json.dumps(cfg))
    for name in ("moving_cylinder", "fixed_cylinder"):
        tiny_cfg[name]["length_m"] *= 0.01
        tiny_cfg[name]["diameter_m"] *= 0.01
    sx, sy = -0.03, -0.03
    tiny = finite_interaction(sx, sy, beta, tiny_cfg)
    point = dipole_interaction(sx, sy, beta)
    tiny_abs = np.abs(tiny - point)
    tiny_scaled = tiny_abs / np.maximum(np.abs(point), np.array([1e-9, 1e-9, 1e-6, 1e-6]))

    dipole_only = dipole_mask & ~finite_mask & valid
    finite_only = finite_mask & ~dipole_mask & valid
    both = dipole_mask & finite_mask & valid
    either = (dipole_mask | finite_mask) & valid
    worked_example = None
    dipole_only_idx = np.argwhere(dipole_only)
    if len(dipole_only_idx):
        # Use the closest safety-relevant disagreement: the point-dipole model
        # admits the task while the finite-body model does not.
        ji = min(dipole_only_idx, key=lambda idx: math.hypot(xs[idx[1]] - model.FIXED_X,
                                                             ys[idx[0]] - model.FIXED_Y))
        xw, yw = float(xs[ji[1]]), float(ys[ji[0]])
        Ahat = model.actuation_matrix_body(xw, yw, beta)
        Acoil, cdip = Ahat[:, :model.NUM_COILS], Ahat[:, model.NUM_COILS]
        cfinite = finite_interaction(xw, yw, beta, cfg)
        signed = cfinite - cdip
        worked_example = {
            "pose": {"x_m": xw, "y_m": yw, "beta_rad": beta,
                     "center_separation_m": math.hypot(xw - model.FIXED_X, yw - model.FIXED_Y)},
            "component_order": list(model.ROW_LABELS),
            "c_dip": cdip.tolist(),
            "c_finite": cfinite.tolist(),
            "signed_discrepancy": signed.tolist(),
            "absolute_discrepancy": np.abs(signed).tolist(),
            "task_lower": (task_center - task_half).tolist(),
            "task_upper": (task_center + task_half).tolist(),
            "dipole_feasible": bool(contains_box(Acoil, task_center - cdip, task_half,
                                                  cfg["current_limit_A"])),
            "finite_body_feasible": bool(contains_box(Acoil, task_center - cfinite, task_half,
                                                       cfg["current_limit_A"])),
        }
    metrics = {
        "scope": "direct pointwise comparison of point-dipole and distributed finite-body interaction models under configurable illustrative cylinder geometry; not continuous or hardware certification",
        "grid": {"n": n, "cell_spacing_m": float(xs[1] - xs[0]), "valid_points": int(common)},
        "nonoverlap_center_separation_m": exclusion,
        "dipole_feasible_points": int((dipole_mask & valid).sum()),
        "finite_body_feasible_points": int((finite_mask & valid).sum()),
        "both_feasible_points": int(both.sum()),
        "dipole_only_points": int(dipole_only.sum()),
        "finite_body_only_points": int(finite_only.sum()),
        "dipole_area_m2_trapezoidal_grid_estimate": float(area_weights[dipole_mask & valid].sum()),
        "finite_body_area_m2_trapezoidal_grid_estimate": float(area_weights[finite_mask & valid].sum()),
        "symmetric_difference_area_m2_trapezoidal_grid_estimate": float(area_weights[(dipole_mask != finite_mask) & valid].sum()),
        "dipole_only_area_m2_trapezoidal_grid_estimate": float(area_weights[dipole_only].sum()),
        "finite_body_only_area_m2_trapezoidal_grid_estimate": float(area_weights[finite_only].sum()),
        "finite_to_dipole_area_ratio": float(area_weights[finite_mask & valid].sum() / max(np.finfo(float).eps, area_weights[dipole_mask & valid].sum())),
        "feasible_set_jaccard": float(both.sum() / max(1, either.sum())),
        "max_absolute_model_discrepancy": dict(zip(model.ROW_LABELS, np.nanmax(np.abs(signed_discrepancy), axis=(0, 1)).tolist())),
        "quadrature_sensitivity_checks": convergence,
        "worked_boundary_example": worked_example,
        "point_dipole_limit_check": {"pose_m": [sx, sy], "absolute_difference": dict(zip(model.ROW_LABELS, tiny_abs.tolist())), "scaled_difference": dict(zip(model.ROW_LABELS, tiny_scaled.tolist()))},
        "sanity": {
            "finite_reference_tends_to_point_dipole": bool(np.max(tiny_scaled) < 5e-3)
        },
        "config": cfg
    }
    assert all(metrics["sanity"].values()), metrics["sanity"]
    return xs, ys, dipole_mask, finite_mask, valid, signed_discrepancy, metrics


def plot(xs, ys, dipole_mask, finite_mask, valid, signed_discrepancy, metrics):
    OUT.mkdir(exist_ok=True)
    extent = [xs[0] * 1e3, xs[-1] * 1e3, ys[0] * 1e3, ys[-1] * 1e3]
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.65), constrained_layout=True)
    for ax, mask, title in zip(axes[:2], (dipole_mask, finite_mask), ("Point-dipole TFW", "Finite-body TFW")):
        shown = np.where(valid, mask.astype(float), np.nan)
        ax.imshow(shown, origin="lower", extent=extent, vmin=0, vmax=1, cmap="RdYlGn", interpolation="nearest")
        ax.plot(model.FIXED_X * 1e3, model.FIXED_Y * 1e3, "ko", ms=4)
        ax.set_title(title); ax.set_xlabel("moving magnet x (mm)"); ax.set_ylabel("y (mm)"); ax.set_aspect("equal")
    force_delta = np.max(np.where(np.isnan(signed_discrepancy[:, :, 2:]), -np.inf,
                                  np.abs(signed_discrepancy[:, :, 2:])), axis=2) * 1e3
    force_delta[~valid] = np.nan
    im = axes[2].imshow(force_delta, origin="lower", extent=extent, cmap="magma", interpolation="nearest")
    axes[2].set_title("Maximum force discrepancy (mN)"); axes[2].set_xlabel("moving magnet x (mm)"); axes[2].set_ylabel("y (mm)"); axes[2].set_aspect("equal")
    fig.colorbar(im, ax=axes[2], shrink=.82)
    fig.suptitle(f"Case-6 cross-section, feasible-set Jaccard = {100*metrics['feasible_set_jaccard']:.1f}%")
    fig.savefig(OUT / "dipole_vs_finite_tfw.pdf")
    fig.savefig(OUT / "dipole_vs_finite_tfw.png", dpi=240)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    args = ap.parse_args()
    cfg = json.loads(args.config.read_text())
    xs, ys, dipole_mask, finite_mask, valid, signed_discrepancy, metrics = run(cfg)
    OUT.mkdir(exist_ok=True)
    np.savez_compressed(OUT / "masks_and_errors.npz", xs=xs, ys=ys,
                        dipole=dipole_mask, finite_body=finite_mask, valid=valid,
                        signed_discrepancy=signed_discrepancy,
                        absolute_discrepancy=np.abs(signed_discrepancy))
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    plot(xs, ys, dipole_mask, finite_mask, valid, signed_discrepancy, metrics)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
