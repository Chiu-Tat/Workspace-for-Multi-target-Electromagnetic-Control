#!/usr/bin/env python3
"""Geometry-only preview of the resized over-under crossing.

No control, flow or feasibility calculation is performed here.

Run from the repository root::

    python -m cases.case_10_two_magnets_vessel_crossing.build_preview
"""

from __future__ import annotations

import argparse
import os

import matplotlib
import numpy as np

from .route import Route, load, magnet_properties

PREVIEW = os.path.join(os.path.dirname(os.path.abspath(__file__)), "preview")
COLORS = {"RA": "#c0504d", "RV": "#4a7ebb"}


def _tube_outline(route, samples=241):
    """Projected lumen boundary in the x-y plane."""

    s = np.linspace(0.0, 1.0, samples)
    centers = np.asarray([route.point(value) for value in s])
    normals = np.asarray([route.frame(value)[1] for value in s])
    left = centers + route.radius_m * normals
    right = centers - route.radius_m * normals
    return centers, left, right


def build(config=None, path=None, show=False):
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle

    vessel, raw = load(config) if config else load()
    routes = {name: Route(vessel, name) for name in ("RA", "RV")}
    design = raw["design"]
    magnets = {entry["branch"]: entry for entry in raw["magnets"]}

    fig = plt.figure(figsize=(11.0, 5.0))
    grid = fig.add_gridspec(2, 2, width_ratios=(1.0, 1.35), height_ratios=(1.0, 1.0))
    top = fig.add_subplot(grid[:, 0])
    side = fig.add_subplot(grid[0, 1])
    cross = fig.add_subplot(grid[1, 1])

    for name, route in routes.items():
        centers, left, right = _tube_outline(route)
        patch = np.vstack((left, right[::-1]))
        top.fill(patch[:, 0] * 1e3, patch[:, 1] * 1e3, color=COLORS[name],
                 alpha=0.28, linewidth=0)
        top.plot(centers[:, 0] * 1e3, centers[:, 1] * 1e3, color=COLORS[name],
                 linewidth=1.2, label=f"{name} ({route.radius_m*2e3:.1f} mm lumen)")
        side.fill_between(
            centers[:, 1] * 1e3,
            (centers[:, 2] - route.radius_m) * 1e3,
            (centers[:, 2] + route.radius_m) * 1e3,
            color=COLORS[name], alpha=0.28, linewidth=0,
        )
        side.plot(centers[:, 1] * 1e3, centers[:, 2] * 1e3, color=COLORS[name],
                  linewidth=1.2)
        for label, s in (("start", magnets[name]["start_s"]),
                         ("target", magnets[name]["target_s"])):
            p = route.point(s)
            radius = 0.5 * magnets[name]["diameter_mm"]
            top.add_patch(Circle((p[0] * 1e3, p[1] * 1e3), radius,
                                 facecolor=COLORS[name], edgecolor="#222222",
                                 linewidth=0.6, zorder=4))
            top.annotate(f"{name} {label}", (p[0] * 1e3, p[1] * 1e3),
                         xytext=(6, -3), textcoords="offset points", fontsize=7)

    top.set_title("Top view: two routes crossing in projection")
    top.set_xlabel("x (mm)")
    top.set_ylabel("y (mm)")
    top.set_aspect("equal")
    top.grid(color="#e4e0d8", linewidth=0.5)
    top.legend(fontsize=7, loc="upper left")

    side.set_title(f"Side view: depth gap {design['depth_gap_mm']:.1f} mm")
    side.set_xlabel("y (mm)")
    side.set_ylabel("z (mm)")
    side.set_aspect("equal")
    side.grid(color="#e4e0d8", linewidth=0.5)

    # True-scale section through the crossing, normal to y.
    for name, route in routes.items():
        p = route.point(0.5)
        cross.add_patch(Circle((p[0] * 1e3, p[2] * 1e3), route.radius_m * 1e3,
                               facecolor=COLORS[name], alpha=0.3,
                               edgecolor=COLORS[name], linewidth=1.2))
        cross.add_patch(Circle((p[0] * 1e3, p[2] * 1e3),
                               0.5 * magnets[name]["diameter_mm"],
                               facecolor="#333333", edgecolor="none", zorder=4))
        cross.annotate(name, (p[0] * 1e3, p[2] * 1e3), xytext=(8, 6),
                       textcoords="offset points", fontsize=8)
    gap = raw["designed_lumen_gap_mm"]
    z_lower = (routes["RA"].point(0.5)[2] + routes["RA"].radius_m) * 1e3
    cross.annotate(
        f"wall-to-wall gap {gap:.2f} mm",
        (0.0, z_lower + 0.5 * gap), xytext=(14, 0), textcoords="offset points",
        fontsize=8, arrowprops={"arrowstyle": "->", "linewidth": 0.7},
    )
    cross.set_title("Section at the crossing (true scale)")
    cross.set_xlabel("x (mm)")
    cross.set_ylabel("z (mm)")
    cross.set_aspect("equal")
    depth = max(route.point(0.5)[2] for route in routes.values()) * 1e3
    span = 0.5 * design["depth_gap_mm"] + 2.5
    cross.set_xlim(-2.2 * span, 2.2 * span)
    cross.set_ylim(depth - 0.5 * design["depth_gap_mm"] - span,
                   depth - 0.5 * design["depth_gap_mm"] + span)
    cross.grid(color="#e4e0d8", linewidth=0.5)

    radius, moment, weight = magnet_properties(raw["magnets"][0])
    fig.suptitle(
        f"Case 10 phantom: route length {design['route_length_mm']:.0f} mm, "
        f"lumens {design['lumen_diameter_mm']['RA']:.1f}/"
        f"{design['lumen_diameter_mm']['RV']:.1f} mm, "
        f"{2e3*radius:.1f} mm magnets (m = {moment*1e3:.2f} mA m$^2$), "
        f"registered at z = {vessel.origin[2]*1e3:.0f} mm in the coil frame",
        fontsize=10,
    )
    fig.tight_layout()
    os.makedirs(PREVIEW, exist_ok=True)
    path = path or os.path.join(PREVIEW, "geometry.png")
    fig.savefig(path, dpi=170, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args(argv)
    print(f"saved {build(args.config, args.output, args.show)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
