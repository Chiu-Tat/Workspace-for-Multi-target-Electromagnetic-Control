#!/usr/bin/env python3
"""Regenerate the parameterized over-under crossing geometry.

The lumen shape is inherited unchanged from
``case_7_vessel_microspheres/geometries/close_crossing.json``: two cubic Bezier
centerlines that cross in projection and are separated along the depth axis.
Only four scalars set the size, so the phantom can be resized without editing
control points by hand::

    route_length_mm        axial (y) extent of both routes
    lateral_half_spread_mm x offset of each inlet/outlet from the crossing
    depth_gap_mm           centerline separation along z
    lumen_diameter_mm      one constant diameter per branch

Run from the repository root::

    python -m cases.case_10_two_magnets_vessel_crossing.build_geometry
    python -m cases.case_10_two_magnets_vessel_crossing.build_geometry \
        --route-length 40 --depth-gap 6
"""

from __future__ import annotations

import argparse
import json
import math
import os

# Normalized control-point shape of the Case 7 crossing preset. Dividing the
# published points by their end values leaves these dimensionless handles, so
# scaling reproduces the original curve shape at any size.
SHAPE_Y = (-1.0, -6.0 / 25.0, 6.0 / 25.0, 1.0)
SHAPE_X = {
    "RA": (-1.0, -5.0 / 9.0, 5.0 / 9.0, 1.0),
    "RV": (1.0, 4.0 / 9.0, -4.0 / 9.0, -1.0),
}
DEPTH_SIGN = {"RA": -1.0, "RV": 1.0}
LABELS = {"RA": "Artery-side route, deeper", "RV": "Vein-side route, shallower"}

DEFAULTS = {
    "route_length_mm": 30.0,
    "lateral_half_spread_mm": 6.0,
    "depth_gap_mm": 4.0,
    "lumen_diameter_mm": {"RA": 3.2, "RV": 2.8},
    "origin_emns_mm": [0.0, 0.0, 85.0],
}
MAGNETS = [
    {
        "name": "magnet 1",
        "branch": "RA",
        "diameter_mm": 2.0,
        "remanence_T": 1.2,
        "density_kg_m3": 7500.0,
        "start_s": 0.05,
        "target_s": 0.95,
    },
    {
        "name": "magnet 2",
        "branch": "RV",
        "diameter_mm": 2.0,
        "remanence_T": 1.2,
        "density_kg_m3": 7500.0,
        "start_s": 0.05,
        "target_s": 0.95,
    },
]
CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "geometry.json")


def control_points(branch, route_length_mm, lateral_half_spread_mm, depth_gap_mm):
    """Return the four scaled Bezier control points of one branch, in mm."""

    if route_length_mm <= 0.0 or lateral_half_spread_mm < 0.0:
        raise ValueError("route length must be positive and spread nonnegative")
    if depth_gap_mm <= 0.0:
        raise ValueError("depth gap must be positive")
    z = DEPTH_SIGN[branch] * 0.5 * depth_gap_mm
    return [
        [
            SHAPE_X[branch][k] * lateral_half_spread_mm,
            SHAPE_Y[k] * 0.5 * route_length_mm,
            z,
        ]
        for k in range(4)
    ]


def build(
    route_length_mm=DEFAULTS["route_length_mm"],
    lateral_half_spread_mm=DEFAULTS["lateral_half_spread_mm"],
    depth_gap_mm=DEFAULTS["depth_gap_mm"],
    lumen_diameter_mm=None,
    origin_emns_mm=None,
):
    """Return the geometry dictionary for one parameter set."""

    lumen_diameter_mm = dict(lumen_diameter_mm or DEFAULTS["lumen_diameter_mm"])
    origin = list(origin_emns_mm or DEFAULTS["origin_emns_mm"])
    radii = {name: 0.5 * value for name, value in lumen_diameter_mm.items()}
    for name, radius in radii.items():
        if radius <= 0.0:
            raise ValueError(f"lumen diameter of {name} must be positive")
    gap = depth_gap_mm - radii["RA"] - radii["RV"]
    if gap <= 0.0:
        raise ValueError(
            "the two lumens overlap: depth gap must exceed the summed radii"
        )
    branches = {}
    for name in ("RA", "RV"):
        branches[name] = {
            "label": LABELS[name],
            "control_points_mm": control_points(
                name, route_length_mm, lateral_half_spread_mm, depth_gap_mm
            ),
            "radius_start_mm": radii[name],
            "radius_end_mm": radii[name],
            "bulb_amplitude_mm": 0.0,
            "bulb_center_s": 0.5,
            "bulb_width_s": 0.2,
        }
    return {
        "description": (
            "Bench-scale over-under vessel crossing for two permanent-magnet "
            "robots. The centerline shape is the Case 7 close_crossing preset "
            "rescaled by the design block; it is not a patient reconstruction."
        ),
        "title": "Two vessels crossing at different depths (resized)",
        "units": "mm",
        "origin_emns_mm": origin,
        "rotation_local_to_emns": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "separation_axis": 2,
        "design": {
            "route_length_mm": route_length_mm,
            "lateral_half_spread_mm": lateral_half_spread_mm,
            "depth_gap_mm": depth_gap_mm,
            "lumen_diameter_mm": lumen_diameter_mm,
            "shape_y": list(SHAPE_Y),
            "shape_x": {name: list(value) for name, value in SHAPE_X.items()},
            "source_preset": (
                "cases/case_7_vessel_microspheres/geometries/close_crossing.json"
            ),
        },
        "designed_lumen_gap_mm": gap,
        "crossing_center_distance_mm": depth_gap_mm,
        "branches": branches,
        "magnets": MAGNETS,
    }


def crossing_angle_deg(route_length_mm, lateral_half_spread_mm):
    """Projected angle between the two tangents at the crossing, in degrees."""

    # At s = 0.5 the cubic derivative is 0.75 * (P3 - P0) + 0.75 * (P2 - P1).
    dy = 0.75 * (
        (SHAPE_Y[3] - SHAPE_Y[0]) + (SHAPE_Y[2] - SHAPE_Y[1])
    ) * 0.5 * route_length_mm
    total = 0.0
    for name in ("RA", "RV"):
        shape = SHAPE_X[name]
        dx = 0.75 * (
            (shape[3] - shape[0]) + (shape[2] - shape[1])
        ) * lateral_half_spread_mm
        total += math.degrees(math.atan2(abs(dx), abs(dy)))
    return total


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route-length", type=float,
                        default=DEFAULTS["route_length_mm"])
    parser.add_argument("--lateral-spread", type=float,
                        default=DEFAULTS["lateral_half_spread_mm"])
    parser.add_argument("--depth-gap", type=float, default=DEFAULTS["depth_gap_mm"])
    parser.add_argument("--artery-diameter", type=float,
                        default=DEFAULTS["lumen_diameter_mm"]["RA"])
    parser.add_argument("--vein-diameter", type=float,
                        default=DEFAULTS["lumen_diameter_mm"]["RV"])
    parser.add_argument("--output", default=CONFIG)
    args = parser.parse_args(argv)
    try:
        geometry = build(
            route_length_mm=args.route_length,
            lateral_half_spread_mm=args.lateral_spread,
            depth_gap_mm=args.depth_gap,
            lumen_diameter_mm={"RA": args.artery_diameter, "RV": args.vein_diameter},
        )
    except ValueError as exc:
        parser.error(str(exc))
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(geometry, handle, indent=2)
        handle.write("\n")
    print(f"saved {args.output}")
    print(f"lumen wall-to-wall gap at the crossing: "
          f"{geometry['designed_lumen_gap_mm']:.2f} mm")
    print(f"projected crossing angle: "
          f"{crossing_angle_deg(args.route_length, args.lateral_spread):.1f} deg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
