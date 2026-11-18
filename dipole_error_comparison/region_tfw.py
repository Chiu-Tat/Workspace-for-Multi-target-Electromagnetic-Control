#!/usr/bin/env python3
"""Region-based (affine-hybrid) nominal and model-robust TFW on the beta = 90 deg slice.

Nominal:  the Case-6 affine-hybrid cascade of ``workspace_hybrid.classify_box_hybrid``
          with the point-dipole interaction offset c_dip.
Robust:   Corollary 1 at box level.  For each pose box Q, an upper bound
              eps_k(Q) >= sup_{q in Q} |c_FB,k(q) - c_dip,k(q)| (+ rho_k(Q))
          is added to the affine dipole offset as a new, independent noise symbol,
          and the same cascade is run.  An inside verdict therefore holds for every
          pose in Q and every offset in [c_dip(q) - eps, c_dip(q) + eps], which
          contains c_FB(q).

eps(Q) comes from a vectorized interval evaluation of the finite-body sum
(``run_comparison.finite_interaction``) over all source/receiver quadrature pairs,
minus the interval enclosure of c_dip(Q).  The subtraction decouples the two
offsets, so eps is conservative.  Like the rest of the archived backend, the
floating-point evaluation does not use outward rounding; the finite-body
quadrature itself is a numerical reference model, not ground truth.

Run from the repository root:
    python dipole_error_comparison/region_tfw.py --check        # enclosure test
    python dipole_error_comparison/region_tfw.py                # both runs + JSON
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "results"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
from emns.numerics.affine import Affine, get_next_symbol_index  # noqa: E402
from emns.numerics.interval import interval  # noqa: E402
import cases.case_6_interaction_obstacle_avoidance.workspace_hybrid as W  # noqa: E402
from cases.case_6_interaction_obstacle_avoidance.simulation import model  # noqa: E402
from hybrid_region.certificates import (  # noqa: E402
    affine_constructive_inside_certificate,
    constructive_inside_certificate,
    dual_outside_certificate,
    interval_object_bounds,
)
from emns.workspace.checks import Out, RobustFeasible  # noqa: E402
from run_comparison import cylinder_points, finite_interaction, rotation_from_z, MU04PI  # noqa: E402

BETA = math.pi / 2


# ----------------------------------------------------------------------------
# Vectorized interval arithmetic on (lo, hi) numpy pairs
# ----------------------------------------------------------------------------
def i_mul(a, b):
    p = np.stack([a[0] * b[0], a[0] * b[1], a[1] * b[0], a[1] * b[1]])
    return p.min(axis=0), p.max(axis=0)


def i_scale(a, s):
    """Interval times a real array s (broadcast)."""
    lo, hi = a[0] * s, a[1] * s
    return np.minimum(lo, hi), np.maximum(lo, hi)


def i_add(*terms):
    return sum(t[0] for t in terms), sum(t[1] for t in terms)


def i_sq(a):
    lo, hi = a
    m = np.maximum(lo * lo, hi * hi)
    low = np.where((lo <= 0) & (hi >= 0), 0.0, np.minimum(lo * lo, hi * hi))
    return low, m


def i_pow_neg(n2, p):
    """n2 ** (-p) for a positive interval n2 (monotone decreasing)."""
    return n2[1] ** (-p), n2[0] ** (-p)


def fb_offset_enclosure(x_iv, y_iv, cfg, beta=BETA):
    """Interval enclosure of c_FB = [b_y, b_z, f_y, f_z] over the pose box x_iv x y_iv."""
    q = cfg["reference_quadrature"]
    gm, gf = cfg["moving_cylinder"], cfg["fixed_cylinder"]
    pm, wm = cylinder_points(gm["length_m"], gm["diameter_m"], q["radial"], q["azimuthal"], q["axial"])
    pf, wf = cylinder_points(gf["length_m"], gf["diameter_m"], q["radial"], q["azimuthal"], q["axial"])
    Rm, Rf = rotation_from_z(beta), rotation_from_z(model.FIXED_BETA)
    om = pm @ Rm.T                                    # receiver offsets from the moving centre
    pfw = pf @ Rf.T + np.array([model.FIXED_X, model.FIXED_Y, model.Z_PLANE])
    mm = model.FREE_M * wm[:, None] * Rm[:, 2]        # receiver element moments
    mf = model.FIXED_M * wf[:, None] * Rf[:, 2]       # source element moments
    # r = p_m - p_f for every (receiver m, source n) pair; only x and y are uncertain.
    dx = om[:, None, 0] - pfw[None, :, 0]
    dy = om[:, None, 1] - pfw[None, :, 1]
    rz = om[:, None, 2] + model.Z_PLANE - pfw[None, :, 2]
    rx = (x_iv[0] + dx, x_iv[1] + dx)
    ry = (y_iv[0] + dy, y_iv[1] + dy)
    rzi = (rz, rz)
    n2 = i_add(i_sq(rx), i_sq(ry), (rz * rz, rz * rz))
    if np.any(n2[0] <= 0):
        raise ValueError("pose box reaches a quadrature-pair singularity")
    inv3, inv5, inv7 = i_pow_neg(n2, 1.5), i_pow_neg(n2, 2.5), i_pow_neg(n2, 3.5)
    r = (rx, ry, rzi)
    mfx, mfy, mfz = (np.broadcast_to(mf[None, :, k], dx.shape) for k in range(3))
    mmx, mmy, mmz = (np.broadcast_to(mm[:, None, k], dx.shape) for k in range(3))
    sf = i_add(i_scale(rx, mfx), i_scale(ry, mfy), i_scale(rzi, mfz))   # r . m_f
    sm = i_add(i_scale(rx, mmx), i_scale(ry, mmy), i_scale(rzi, mmz))   # r . m_m
    mdot = mm @ mf.T                                                     # m_m . m_f (exact)
    by, bz = Rm[:, 1], Rm[:, 2]
    out = []
    # Field: B = k (3 r (r.mf) / |r|^5 - mf / |r|^3), averaged over receiver weights.
    sf_inv5 = i_mul(sf, inv5)
    for e in (by, bz):
        re = i_add(*(i_scale(r[k], e[k]) for k in range(3)))           # r . e
        mfe = mf @ e
        term = i_add(i_scale(i_mul(re, sf_inv5), 3.0), i_scale(inv3, -np.broadcast_to(mfe[None, :], dx.shape)))
        term = i_scale(term, MU04PI * wm[:, None])
        out.append((term[0].sum(), term[1].sum()))
    # Force: 3k[(r.mm) mf + (r.mf) mm + (mm.mf) r] / |r|^5 - 15k (r.mm)(r.mf) r / |r|^7
    smsf_inv7 = i_mul(i_mul(sm, sf), inv7)
    for e in (by, bz):
        re = i_add(*(i_scale(r[k], e[k]) for k in range(3)))
        mfe = np.broadcast_to((mf @ e)[None, :], dx.shape)
        mme = np.broadcast_to((mm @ e)[:, None], dx.shape)
        lin = i_add(i_scale(sm, mfe), i_scale(sf, mme), i_scale(re, mdot))
        term = i_add(i_scale(i_mul(lin, inv5), 3 * MU04PI), i_scale(i_mul(smsf_inv7, re), -15 * MU04PI))
        out.append((term[0].sum(), term[1].sum()))
    return np.array([o[0] for o in out]), np.array([o[1] for o in out])


# ----------------------------------------------------------------------------
# Box classification
# ----------------------------------------------------------------------------
_TASK = W._task_setup()


def classify(x0, x1, y0, y1, cfg, robust=False, rho_frac=0.0):
    I_iv, b_iv, task_vertices, current_lower, current_upper = _TASK
    A_affine = W.calculate_affine_A(interval[x0, x1], interval[y0, y1], interval[BETA, BETA], return_affine=True)
    eps = np.zeros(4)
    if robust:
        dip_iv = [A_affine[k, W.NUM_COILS].to_interval() for k in range(4)]
        dip_lo = np.array([iv[0][0] for iv in dip_iv]); dip_hi = np.array([iv[0][1] for iv in dip_iv])
        fb_lo, fb_hi = fb_offset_enclosure((x0, x1), (y0, y1), cfg)
        eps = np.maximum(fb_hi - dip_lo, dip_hi - fb_lo)                 # >= sup |c_FB - c_dip|
        eps = eps + rho_frac * np.maximum(np.abs(dip_lo), np.abs(dip_hi))
        A_affine = A_affine.copy()
        for k in range(4):
            if eps[k] > 0:
                A_affine[k, W.NUM_COILS] = A_affine[k, W.NUM_COILS] + Affine(0.0, {get_next_symbol_index(): float(eps[k])})
    A_interval = W._as_interval_matrix(A_affine)
    if RobustFeasible(A_interval, I_iv, b_iv):
        return "inside", "rohn", eps
    G_affine, c_affine = A_affine[:, :W.NUM_COILS], A_affine[:, W.NUM_COILS]
    G_lower, G_upper = interval_object_bounds(A_interval[:, :W.NUM_COILS])
    c_lower, c_upper = interval_object_bounds(A_interval[:, W.NUM_COILS])
    if affine_constructive_inside_certificate(G_affine, task_vertices, current_lower, current_upper, c_affine=c_affine).certified:
        return "inside", "affine_right_inverse", eps
    if constructive_inside_certificate(G_lower, G_upper, task_vertices, current_lower, current_upper,
                                       c_lower=c_lower, c_upper=c_upper).certified:
        return "inside", "right_inverse", eps
    if Out(A_interval, I_iv, b_iv):
        return "outside", "contractor", eps
    if dual_outside_certificate(G_lower, G_upper, task_vertices, current_lower, current_upper,
                                c_lower=c_lower, c_upper=c_upper).certified:
        return "outside", "dual_separation", eps
    return "undetermined", "none", eps


def exclusion_radius(cfg):
    gm, gf = cfg["moving_cylinder"], cfg["fixed_cylinder"]
    return 0.5 * math.hypot(gm["length_m"], gm["diameter_m"]) + 0.5 * math.hypot(gf["length_m"], gf["diameter_m"])


def sep_range(x0, x1, y0, y1):
    fx, fy = model.FIXED_X, model.FIXED_Y
    near = math.hypot(max(x0 - fx, 0, fx - x1), max(y0 - fy, 0, fy - y1))
    far = max(math.hypot(px - fx, py - fy) for px in (x0, x1) for py in (y0, y1))
    return near, far


def run(cfg, robust, stop, rho_frac=0.0, n0=16, excl_radius=None):
    """Branch-and-prune over the slice.

    Poses closer than ``excl_radius`` to the fixed-target centre are left
    unclassified (default: the physical non-overlap radius of the two illustrative
    bodies).  A smaller radius classifies poses at which the bodies would overlap;
    the verdicts there describe the model only.  A box whose evaluation fails is
    reported as undetermined.
    """
    excl = exclusion_radius(cfg) if excl_radius is None else excl_radius
    (xa, xb), (ya, yb) = cfg["x_limits_m"], cfg["y_limits_m"]
    w0x, w0y = (xb - xa) / n0, (yb - ya) / n0
    queue = [(xa + i * w0x, xa + (i + 1) * w0x, ya + j * w0y, ya + (j + 1) * w0y) for i in range(n0) for j in range(n0)]
    boxes = {"inside": [], "outside": [], "undetermined": [], "excluded": []}
    reasons = {}
    t0 = time.time()
    while queue:
        x0, x1, y0, y1 = queue.pop()
        near, far = sep_range(x0, x1, y0, y1)
        last = (x1 - x0) <= stop * (1 + 1e-9)
        if far < excl:
            boxes["excluded"].append((x0, x1, y0, y1)); continue
        if near < excl:                                   # touches the excluded disc: never classified
            if last:
                boxes["excluded"].append((x0, x1, y0, y1)); continue
            verdict = "undetermined"
        else:
            try:
                verdict, why, _ = classify(x0, x1, y0, y1, cfg, robust=robust, rho_frac=rho_frac)
            except (ValueError, ZeroDivisionError, OverflowError, FloatingPointError):
                verdict, why = "undetermined", "evaluation_failed"
            if verdict != "undetermined" or last:
                reasons[why] = reasons.get(why, 0) + 1
        if verdict != "undetermined" or last:
            boxes[verdict].append((x0, x1, y0, y1)); continue
        xm, ym = (x0 + x1) / 2, (y0 + y1) / 2
        queue += [(x0, xm, y0, ym), (xm, x1, y0, ym), (x0, xm, ym, y1), (xm, x1, ym, y1)]
    total = (xb - xa) * (yb - ya)
    area = {k: sum((b[1] - b[0]) * (b[3] - b[2]) for b in v) / total for k, v in boxes.items()}
    return {"robust": robust, "rho_fraction": rho_frac, "stop_width_m": stop, "exclusion_radius_m": excl,
            "elapsed_s": time.time() - t0, "counts": {k: len(v) for k, v in boxes.items()}, "area_fraction": area,
            "reasons": reasons, "boxes": boxes}


def check_enclosure(cfg, n_boxes=40, n_pts=12, seed=1):
    """Every sampled finite-body offset must lie inside its box enclosure."""
    rng = np.random.default_rng(seed)
    excl = exclusion_radius(cfg)
    worst, done = 0.0, 0
    while done < n_boxes:
        w = rng.choice([0.0005, 0.001, 0.002, 0.004])
        x0, y0 = rng.uniform(-0.05, 0.05 - w), rng.uniform(-0.05, 0.05 - w)
        if sep_range(x0, x0 + w, y0, y0 + w)[0] < excl:
            continue
        lo, hi = fb_offset_enclosure((x0, x0 + w), (y0, y0 + w), cfg)
        for _ in range(n_pts):
            x, y = rng.uniform(x0, x0 + w), rng.uniform(y0, y0 + w)
            c = finite_interaction(x, y, BETA, cfg)
            scale = np.maximum(np.abs(c), 1e-12)
            viol = np.maximum(lo - c, c - hi) / scale
            worst = max(worst, float(viol.max()))
        done += 1
    # A degenerate box must reproduce the point value.
    lo, hi = fb_offset_enclosure((-0.03, -0.03), (0.02, 0.02), cfg)
    c = finite_interaction(-0.03, 0.02, BETA, cfg)
    point_err = float(np.max(np.maximum(np.abs(lo - c), np.abs(hi - c)) / np.maximum(np.abs(c), 1e-12)))
    return {"boxes": n_boxes, "points_per_box": n_pts, "worst_relative_violation": worst,
            "degenerate_box_relative_error": point_err, "passed": worst <= 1e-9 and point_err <= 1e-9}


# ----------------------------------------------------------------------------
# Manuscript Fig. 16
# ----------------------------------------------------------------------------
def rasterize(run_result, cfg, n=256):
    """Map box verdicts onto an n x n raster aligned with the bisection grid."""
    (xa, xb), (ya, yb) = cfg["x_limits_m"], cfg["y_limits_m"]
    code = {"excluded": -1, "outside": 0, "undetermined": 1, "inside": 2}
    grid = np.full((n, n), -2, dtype=int)
    for name, boxes in run_result["boxes"].items():
        for x0, x1, y0, y1 in boxes:
            i0, i1 = round((x0 - xa) / (xb - xa) * n), round((x1 - xa) / (xb - xa) * n)
            j0, j1 = round((y0 - ya) / (yb - ya) * n), round((y1 - ya) / (yb - ya) * n)
            grid[j0:j1, i0:i1] = code[name]
    assert (grid != -2).all(), "raster not fully covered"
    return grid


def make_figure(cfg):
    """Manuscript Fig. 16 in the shared Section 6-7 style (analysis/sections_67/common.py)."""
    from analysis.sections_67.common import CLASS_COLORS, configure_matplotlib
    configure_matplotlib()
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection, PatchCollection
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch, Rectangle
    import matplotlib as mpl
    res = json.loads((OUT / "region_tfw.json").read_text())
    orientation = json.loads((OUT / "orientation_metrics.json").read_text())
    nom, rob = rasterize(res["nominal"], cfg), rasterize(res["robust"], cfg)
    excl_mm = res["nominal"].get("exclusion_radius_m", exclusion_radius(cfg)) * 1e3
    n = nom.shape[0]
    (xa, xb), (ya, yb) = cfg["x_limits_m"], cfg["y_limits_m"]
    extent = [xa * 1e3, xb * 1e3, ya * 1e3, yb * 1e3]
    h = (xb - xa) * 1e3 / n
    # state: -1 excluded (blank), 0 TFW-out, 1 unresolved, 2 TFW-in (nominal only), 3 TFW-in (model-robust)
    state = np.full(nom.shape, -1)
    state[nom == 0] = 0
    state[nom == 1] = 1
    state[nom == 2] = 2
    state[rob == 2] = 3
    green, red, grey = CLASS_COLORS["inside"], CLASS_COLORS["outside"], CLASS_COLORS["undetermined"]
    cmap = ListedColormap(["#ffffff", red, grey, green, green])
    norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5, 2.5, 3.5], 5)
    hatch_color = "#3c7a4c"

    def lost_patches():
        """Run-length rectangles for nominal-only TFW-in cells (hatched)."""
        rects = []
        lost = state == 2
        for j in range(n):
            i = 0
            while i < n:
                if lost[j, i]:
                    k = i
                    while k < n and lost[j, k]:
                        k += 1
                    rects.append(Rectangle((extent[0] + i * h, extent[2] + j * h), (k - i) * h, h))
                    i = k
                else:
                    i += 1
        return rects

    def edges(mask):
        """Staircase boundary between classified cells of different membership."""
        valid = nom != -1
        segs = []
        diff = valid[:, :-1] & valid[:, 1:] & (mask[:, :-1] != mask[:, 1:])
        for j, i in zip(*np.nonzero(diff)):
            x = extent[0] + (i + 1) * h; segs.append([(x, extent[2] + j * h), (x, extent[2] + (j + 1) * h)])
        diff = valid[:-1, :] & valid[1:, :] & (mask[:-1, :] != mask[1:, :])
        for j, i in zip(*np.nonzero(diff)):
            y = extent[2] + (j + 1) * h; segs.append([(extent[0] + i * h, y), (extent[0] + (i + 1) * h, y)])
        return segs

    def chain(segs):
        """Join staircase segments into continuous polylines so that dash patterns render."""
        key = lambda p: (round(p[0], 6), round(p[1], 6))
        adj = {}
        for k, (p, q) in enumerate(segs):
            adj.setdefault(key(p), []).append(k); adj.setdefault(key(q), []).append(k)
        used, lines = set(), []
        for k0 in range(len(segs)):
            if k0 in used:
                continue
            used.add(k0)
            line = [key(segs[k0][0]), key(segs[k0][1])]
            for end in (1, 0):                     # grow forward, then backward
                while True:
                    tip = line[-1] if end else line[0]
                    nxt = [k for k in adj.get(tip, []) if k not in used]
                    if not nxt:
                        break
                    k = nxt[0]; used.add(k)
                    p, q = key(segs[k][0]), key(segs[k][1])
                    other = q if p == tip else p
                    if end:
                        line.append(other)
                    else:
                        line.insert(0, other)
            lines.append(line)
        return lines

    def outlines(*runs):
        """Edges of every classified or unresolved box of the given runs (deduplicated)."""
        segs = set()
        for run_result in runs:
            for name, boxes in run_result["boxes"].items():
                if name == "excluded":
                    continue
                for x0, x1, y0, y1 in boxes:
                    x0, x1, y0, y1 = (round(v * 1e3, 6) for v in (x0, x1, y0, y1))
                    segs.update({((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)), ((x0, y1), (x1, y1)), ((x0, y0), (x0, y1))})
        return [list(s) for s in segs]

    box_edges = outlines(res["nominal"], res["robust"])
    mpl.rcParams["hatch.color"] = hatch_color
    mpl.rcParams["hatch.linewidth"] = 0.5
    fig, ax = plt.subplots(1, 3, figsize=(7.1, 2.75), gridspec_kw={"width_ratios": [1, 1, 1.15]})
    for a, title, lw_box in [(ax[0], "(a) Full $\\beta=90^\\circ$ slice", 0.15), (ax[1], "(b) Near the fixed target", 0.3)]:
        a.imshow(state, origin="lower", extent=extent, cmap=cmap, norm=norm, interpolation="nearest")
        a.add_collection(PatchCollection(lost_patches(), facecolor="none", edgecolor=hatch_color, linewidth=0, hatch="////"))
        a.add_collection(LineCollection(box_edges, colors="white", linewidths=lw_box, alpha=0.9))
        a.add_collection(LineCollection(edges(nom == 2), colors="#1f5fa8", linewidths=1.0))
        a.add_collection(LineCollection(chain(edges(rob == 2)), colors="#202124", linewidths=1.1, linestyles=(0, (1, 1.2))))
        a.add_patch(plt.Circle((model.FIXED_X * 1e3, model.FIXED_Y * 1e3), excl_mm, fill=False, ec="#7a7a7a", lw=0.5))
        a.plot(model.FIXED_X * 1e3, model.FIXED_Y * 1e3, "o", color="#202124", ms=2.5)
        a.set_title(title, fontsize=8); a.set_xlabel("$x$ (mm)"); a.set_ylabel("$y$ (mm)"); a.set_aspect("equal")
    ax[0].set_xlim(extent[:2]); ax[0].set_ylim(extent[2:])
    ax[1].set_xlim(-1, 41); ax[1].set_ylim(-23, 23)
    styles = {0: ("o", "-", 3.5), 45: ("s", "-", 4.5), 90: ("^", "-", 3.5), 135: ("x", "--", 4)}
    for s in orientation["summaries"]:
        rr = np.array([v["r_mid_m"] for v in s["radial_force_discrepancy"]]) * 1e3
        ee = np.array([v["force_error_p95_N"] for v in s["radial_force_discrepancy"]]) * 1e3
        marker, ls, ms = styles.get(s["heading_deg"], ("o", "-", 3.5))
        ax[2].plot(rr, ee, marker=marker, ls=ls, ms=ms, lw=0.9, mfc="none" if marker == "s" else None,
                   label=f"$\\beta={s['heading_deg']}^\\circ$")
    ax[2].axhline(model.F_MAX * 1e3, color="#8a5a00", ls=":", lw=1.0)
    ax[2].text(88, model.F_MAX * 1e3 * 1.3, "$f_z$ task half-width", ha="right", fontsize=7, color="#8a5a00")
    ax[2].set_yscale("log"); ax[2].set_xlabel("centre separation (mm)")
    ax[2].set_ylabel("sampled 95th-percentile\nmax force discrepancy (mN)")
    ax[2].set_title("(c) Separation and orientation sensitivity", fontsize=8); ax[2].grid(True, which="major", alpha=0.25, lw=0.5)
    ax[2].legend(fontsize=7, loc="upper right", bbox_to_anchor=(1.0, 0.92))
    handles = [
        Patch(facecolor=green, label="TFW-in (model-robust)"),
        Patch(facecolor=green, edgecolor=hatch_color, hatch="////", linewidth=0, label="TFW-in (nominal only)"),
        Patch(facecolor=red, label="TFW-out"),
        Patch(facecolor=grey, label="Unresolved"),
        Line2D([], [], color="#1f5fa8", lw=1.0, label="nominal TFW-in boundary"),
        Line2D([], [], color="#202124", lw=1.1, ls=(0, (1, 1.2)), label="model-robust TFW-in boundary"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=7, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.savefig(OUT / "dipole_error_tfw_comparison.pdf")
    fig.savefig(OUT / "dipole_error_tfw_comparison.png", dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="Only run the enclosure test.")
    parser.add_argument("--figure", action="store_true", help="Only redraw Fig. 16 from saved results.")
    parser.add_argument("--stop", type=float, default=0.0005, help="Stopping box width (m).")
    parser.add_argument("--exclusion-radius", type=float, default=None,
                        help="Unclassified disc radius around the fixed target (m); default: physical non-overlap radius.")
    args = parser.parse_args()
    cfg = json.loads((HERE / "config.json").read_text())
    if args.figure:
        make_figure(cfg)
        return
    check = check_enclosure(cfg)
    print(json.dumps(check, indent=2))
    if args.check:
        return
    if not check["passed"]:
        raise SystemExit("enclosure check failed")
    r = args.exclusion_radius
    nominal = run(cfg, robust=False, stop=args.stop, excl_radius=r)
    robust = run(cfg, robust=True, stop=args.stop, excl_radius=r)
    robust20 = run(cfg, robust=True, stop=args.stop, rho_frac=0.20, excl_radius=r)
    for r in (nominal, robust, robust20):
        print({k: r[k] for k in ("robust", "rho_fraction", "elapsed_s", "counts", "area_fraction", "reasons")})
    (OUT / "region_tfw.json").write_text(json.dumps(
        {"scope": "affine-hybrid region classification of the beta=90 deg slice; robust runs apply Corollary 1 with "
                  "eps(Q) >= sup |c_FB - c_dip| over each box (rho>0 is assumed); floating-point, no outward rounding",
         "enclosure_check": check, "nominal": nominal, "robust": robust, "robust_rho20": robust20}) + "\n")
    make_figure(cfg)


if __name__ == "__main__":
    main()
