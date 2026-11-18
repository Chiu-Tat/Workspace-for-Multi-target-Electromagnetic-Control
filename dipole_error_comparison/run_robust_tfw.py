#!/usr/bin/env python3
"""Pointwise model-robust TFW on the matched beta = 90 deg sample grid (diagnostic).

Manuscript Fig. 16 is produced by ``region_tfw.py`` with the affine-hybrid region method;
this script keeps the pointwise counterpart as a cross-check.

Applies Corollary 1 pointwise: the nominal (point-dipole) task box is widened
componentwise by eps_k = |d_FB,k| + rho_k, with d_FB = c_FB - c_dip taken from
``results/masks_and_errors.npz`` (written by ``run_comparison.py``), and the same
vertex-LP containment test is repeated.  rho is unknown; the main result uses
rho = 0 and the sweep uses rho_k = f |c_dip,k| as explicitly *assumed* scenarios.

Because the symmetric box [c_dip - eps, c_dip + eps] contains both c_dip and
c_FB, every robust-feasible sample is feasible under both models.  The result is
a sampled, model-robust classification; it is not a continuous certificate and
not a hardware-robust TFW.

Run after ``run_comparison.py`` and ``run_orientation_study.py``:
    python dipole_error_comparison/run_robust_tfw.py            # compute + figure
    python dipole_error_comparison/run_robust_tfw.py --figure   # diagnostic figure only
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "results"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
from cases.case_6_interaction_obstacle_avoidance.simulation import model  # noqa: E402
from run_comparison import contains_box  # noqa: E402

RHO_FRACTIONS = (0.0, 0.05, 0.10, 0.20)
TRAJECTORY = ROOT / "Results" / "trajectory_case6_3D_02.json"


def compute(cfg):
    base = np.load(OUT / "masks_and_errors.npz")
    xs, ys, valid = base["xs"], base["ys"], base["valid"]
    dip, signed = base["dipole"] & valid, base["signed_discrepancy"]
    beta = cfg["moving_heading_rad"]
    center = np.array([0.0, model.B_HOLD, 0.0, 0.0])
    half = np.array([0.0, 0.0, 0.0, model.F_MAX])
    masks = {f: np.zeros_like(dip) for f in RHO_FRACTIONS}
    zero_check = np.zeros_like(dip)
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            if not valid[j, i]:
                continue
            A = model.actuation_matrix_body(x, y, beta)
            Acoil, cdip = A[:, :model.NUM_COILS], A[:, model.NUM_COILS]
            zero_check[j, i] = contains_box(Acoil, center - cdip, half, cfg["current_limit_A"])
            for f in RHO_FRACTIONS:
                eps = np.abs(signed[j, i]) + f * np.abs(cdip)
                masks[f][j, i] = contains_box(Acoil, center - cdip, half + eps, cfg["current_limit_A"])
    np.savez_compressed(OUT / "robust_tfw.npz", xs=xs, ys=ys, valid=valid, nominal=dip,
                        **{f"robust_rho{int(round(100 * f))}": m for f, m in masks.items()})
    return metrics(base, masks, zero_check)


def metrics(base, masks, zero_check):
    xs, ys, valid = base["xs"], base["ys"], base["valid"]
    dip, fb = base["dipole"] & valid, base["finite_body"] & valid
    X, Y = np.meshgrid(xs, ys)
    sep = np.hypot(X - model.FIXED_X, Y - model.FIXED_Y) * 1e3
    rows = []
    for f, m in masks.items():
        m = m & valid
        rows.append({
            "rho_fraction_of_abs_c_dip": f,
            "assumed": f > 0,
            "robust_feasible": int(m.sum()),
            "lost_vs_nominal": int((dip & ~m).sum()),
            "robust_below_20mm": int((m & (sep < 20)).sum()),
            "robust_at_or_beyond_20mm": int((m & (sep >= 20)).sum()),
            "closest_robust_separation_mm": float(sep[m].min()),
        })
    keys = list(masks)
    path = np.array(json.loads(TRAJECTORY.read_text())["trajectory"])
    path_sep = np.hypot(path[:, 0] - model.FIXED_X, path[:, 1] - model.FIXED_Y) * 1e3
    result = {
        "scope": "pointwise Corollary-1 classification on the 61x61, beta=90 deg slice; "
                 "rho>0 rows are assumed scenarios; not a continuous or hardware-robust certificate",
        "valid": int(valid.sum()),
        "nominal_feasible": int(dip.sum()),
        "finite_body_feasible": int(fb.sum()),
        "feasible_under_both": int((dip & fb).sum()),
        "nominal_feasible_below_20mm": int((dip & (sep < 20)).sum()),
        "valid_at_or_beyond_20mm": int((valid & (sep >= 20)).sum()),
        "rows": rows,
        "checks": {
            "zero_reserve_recovers_nominal": bool(np.array_equal(zero_check & valid, dip)),
            "nested_shrinkage": all(bool(np.all(masks[keys[n + 1]] <= masks[keys[n]])) for n in range(len(keys) - 1)),
            "robust_subset_of_both_models": bool(np.all(masks[0.0] & valid <= dip & fb)),
        },
        "planned_path": {
            "file": str(TRAJECTORY.relative_to(ROOT)),
            "min_separation_mm": float(path_sep.min()),
            "heading_range_deg": [float(np.degrees(path[:, 2].min())), float(np.degrees(path[:, 2].max()))],
        },
    }
    (OUT / "robust_tfw_metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def make_figure():
    """Pointwise nominal/model-robust diagnostic figure (not the manuscript figure)."""
    data = np.load(OUT / "robust_tfw.npz")
    orientation = json.loads((OUT / "orientation_metrics.json").read_text())
    xs, ys = data["xs"] * 1e3, data["ys"] * 1e3
    valid, nominal, robust = data["valid"], data["nominal"], data["robust_rho0"] & data["valid"]
    X, Y = np.meshgrid(xs, ys)

    state = np.full(valid.shape, -1.0)          # excluded (overlap)
    state[valid] = 0.0                          # nominally infeasible
    state[nominal & ~robust] = 1.0              # nominal only: lost under the reserve
    state[robust] = 2.0                         # model-robust feasible
    colors = ["#d9d9d9", "#f2d7d5", "#f2c66d", "#ccebd8"]
    # Contours only between admissible samples, so the overlap edge is not drawn as a boundary.
    nom_field = np.ma.masked_where(~valid, nominal.astype(float))
    rob_field = np.ma.masked_where(~valid, robust.astype(float))

    fig, ax = plt.subplots(1, 3, figsize=(13.2, 4.3), constrained_layout=True,
                           gridspec_kw={"width_ratios": [1, 1, 1.15]})
    for a, title in [(ax[0], "(a) Full $\\beta=90^\\circ$ slice"), (ax[1], "(b) Near the fixed target")]:
        a.contourf(X, Y, state, levels=[-1.5, -0.5, 0.5, 1.5, 2.5], colors=colors)
        a.contour(X, Y, nom_field, levels=[0.5], colors=["#1976d2"], linewidths=2)
        a.contour(X, Y, rob_field, levels=[0.5], colors=["k"], linewidths=1.8, linestyles="--")
        a.plot(model.FIXED_X * 1e3, model.FIXED_Y * 1e3, "ko", ms=5)
        a.set_title(title)
        a.set_xlabel("$x$ (mm)")
        a.set_ylabel("$y$ (mm)")
        a.set_aspect("equal")
    lost = nominal & ~robust
    pad = 8
    ax[1].set_xlim(X[lost | (valid & ~nominal)].min() - pad, X[lost | (valid & ~nominal)].max() + pad)
    ax[1].set_ylim(Y[lost | (valid & ~nominal)].min() - pad, Y[lost | (valid & ~nominal)].max() + pad)
    handles = [
        Patch(color=colors[3], label="model-robust feasible"),
        Patch(color=colors[2], label="nominal only"),
        Patch(color=colors[1], label="nominally infeasible"),
        Patch(color=colors[0], label="excluded (overlap)"),
        Line2D([], [], color="#1976d2", lw=2, label="nominal boundary"),
        Line2D([], [], color="k", lw=1.8, ls="--", label="model-robust boundary"),
    ]
    ax[0].legend(handles=handles, fontsize=7.5, loc="lower left", framealpha=0.9)

    # 45 and 135 deg give identical curves; distinct markers/dashes keep both visible.
    styles = {0: ("o", "-", 6), 45: ("s", "-", 8), 90: ("^", "-", 6), 135: ("x", "--", 7)}
    for s in orientation["summaries"]:
        rr = np.array([v["r_mid_m"] for v in s["radial_force_discrepancy"]]) * 1e3
        ee = np.array([v["force_error_p95_N"] for v in s["radial_force_discrepancy"]]) * 1e3
        marker, ls, ms = styles.get(s["heading_deg"], ("o", "-", 6))
        ax[2].plot(rr, ee, marker=marker, ls=ls, ms=ms, mfc="none" if marker == "s" else None,
                   label=f"$\\beta={s['heading_deg']}^\\circ$")
    ax[2].axhline(model.F_MAX * 1e3, color="#8a5a00", ls=":", lw=1.5)
    ax[2].text(88, model.F_MAX * 1e3 * 1.25, "$f_z$ task half-width", ha="right", fontsize=8, color="#8a5a00")
    ax[2].set_yscale("log")
    ax[2].set_xlabel("centre separation (mm)")
    ax[2].set_ylabel("sampled 95th-percentile\nmax force discrepancy (mN)")
    ax[2].set_title("(c) Separation and orientation sensitivity")
    ax[2].grid(True, which="both", alpha=0.25)
    ax[2].legend(fontsize=8)
    fig.savefig(OUT / "pointwise_robust_tfw.pdf")
    fig.savefig(OUT / "pointwise_robust_tfw.png", dpi=240)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--figure", action="store_true", help="Only redraw the pointwise diagnostic figure.")
    args = parser.parse_args()
    if not args.figure:
        cfg = json.loads((HERE / "config.json").read_text())
        print(json.dumps(compute(cfg), indent=2))
    make_figure()


if __name__ == "__main__":
    main()
