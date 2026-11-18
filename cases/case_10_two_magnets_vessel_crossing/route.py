"""Route geometry for the two crossing lumens, in SI units.

Each magnet is reduced to one scalar route coordinate ``s in [0, 1]``, the
Bezier parameter of its assigned centerline.  ``s`` is not arc length.  The
module supplies the numeric centerline and its transport frame, plus affine
enclosures of the centerline over an ``s`` interval so the workspace solver can
propagate one shared noise symbol per magnet.
"""

from __future__ import annotations

import json
import math
import os

import numpy as np

from emns.numerics import affine as ca

from cases.case_7_vessel_microspheres.vessel import Vessel

CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "geometry.json")
MU0 = 4.0e-7 * math.pi
GRAVITY = 9.80665


def load(config=CONFIG):
    """Return ``(Vessel, raw configuration dictionary)``."""

    with open(config, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return Vessel(config), raw


def magnet_properties(entry, fluid_density=1000.0):
    """Return moment (A m^2) and net weight (N) of one described magnet."""

    radius = 0.5 * float(entry["diameter_mm"]) * 1e-3
    if radius <= 0.0:
        raise ValueError("magnet diameter must be positive")
    volume = 4.0 / 3.0 * math.pi * radius**3
    magnetization = float(entry["remanence_T"]) / MU0
    moment = volume * magnetization
    weight = volume * (float(entry["density_kg_m3"]) - fluid_density) * GRAVITY
    return radius, moment, weight


class Route:
    """One centerline with numeric and affine evaluation in EMNS metres."""

    def __init__(self, vessel, branch):
        self.branch = branch
        self.radius_m = float(vessel.branches[branch]["radius_start_mm"]) * 1e-3
        if not math.isclose(
            self.radius_m,
            float(vessel.branches[branch]["radius_end_mm"]) * 1e-3,
        ):
            raise ValueError("this case assumes one constant radius per branch")
        if float(vessel.branches[branch]["bulb_amplitude_mm"]) != 0.0:
            raise ValueError("this case assumes no bulb term")
        points = np.asarray(
            vessel.branches[branch]["control_points_mm"], dtype=float
        ) * 1e-3
        # Fold the rigid registration into the control points once, so the
        # Bezier polynomial can be evaluated directly in the coil frame.
        self.points = points @ vessel.rotation.T + vessel.origin
        p = self.points
        # Power-basis coefficients. Writing the cubic as a0 + a1 s + a2 s^2 +
        # a3 s^3 keeps a constant coordinate exactly constant in the affine
        # enclosure; the Bernstein form leaves four independent remainders
        # that do not cancel.
        self.coefficients = np.stack((
            p[0],
            3.0 * (p[1] - p[0]),
            3.0 * (p[0] - 2.0 * p[1] + p[2]),
            -p[0] + 3.0 * p[1] - 3.0 * p[2] + p[3],
        ))

    def point(self, s):
        s = float(s)
        w = np.array([(1 - s) ** 3, 3 * (1 - s) ** 2 * s, 3 * (1 - s) * s * s, s**3])
        return w @ self.points

    def derivative(self, s):
        s = float(s)
        p = self.points
        return 3.0 * (
            (1 - s) ** 2 * (p[1] - p[0])
            + 2 * (1 - s) * s * (p[2] - p[1])
            + s * s * (p[3] - p[2])
        )

    def frame(self, s):
        """Return ``(t, n, b)``: tangent, in-plane normal and the depth axis.

        Both centerlines lie in a plane of constant ``z``, so the depth axis is
        the global ``+z`` and ``n = b x t`` completes a right-handed frame.
        """

        t = self.derivative(s)
        norm = np.linalg.norm(t)
        if norm <= 0.0:
            raise ValueError("degenerate centerline tangent")
        t = t / norm
        b = np.array([0.0, 0.0, 1.0])
        n = np.cross(b, t)
        n /= np.linalg.norm(n)
        return t, n, b

    def arc_length(self, samples=2001):
        s = np.linspace(0.0, 1.0, samples)
        w = np.stack(((1 - s) ** 3, 3 * (1 - s) ** 2 * s, 3 * (1 - s) * s * s, s**3),
                     axis=-1)
        curve = w @ self.points
        return float(np.sum(np.linalg.norm(np.diff(curve, axis=0), axis=1)))

    def affine_point(self, s_interval, slack=0.0):
        """Enclose the centerline over an ``s`` interval as three affine forms.

        One shared noise symbol represents ``s``, so the enclosure keeps the
        dependency between the three coordinates.  The cubic is expanded with
        exact affine products; ``slack`` adds an independent symbol per axis to
        cover a cross-section offset of at most ``slack`` metres from the
        centerline.
        """

        lower = float(s_interval.lower)
        upper = float(s_interval.upper)
        if not (0.0 <= lower <= upper <= 1.0):
            raise ValueError("s interval must lie inside [0, 1]")
        center = 0.5 * (lower + upper)
        radius = 0.5 * (upper - lower)
        s = ca.Affine(center, {ca.get_next_symbol_index(): radius} if radius else {})
        # Powers are formed by explicit products so the affine backend keeps
        # the shared symbol instead of widening each factor to an interval.
        powers = (ca.Affine(1.0), s, _square(s), _multiply(_square(s), s))
        coordinates = []
        for axis in range(3):
            form = ca.Affine(0.0)
            for power, coefficient in zip(powers, self.coefficients):
                if coefficient[axis]:
                    form = form + _scale(power, float(coefficient[axis]))
            if slack:
                form = form + ca.Affine(
                    0.0, {ca.get_next_symbol_index(): float(slack)}
                )
            coordinates.append(form)
        return tuple(coordinates)

    def tangent_spread_deg(self, s_interval, samples=9):
        """Maximum angle between the box-centre tangent and any tangent inside.

        The cubic tangent is sampled and inflated by a derivative bound, giving
        a conservative angular spread for the fixed task frame.
        """

        lower = float(s_interval.lower)
        upper = float(s_interval.upper)
        reference = self.frame(0.5 * (lower + upper))[0]
        worst = 0.0
        for s in np.linspace(lower, upper, samples):
            t = self.derivative(float(s))
            t = t / np.linalg.norm(t)
            worst = max(worst, math.acos(float(np.clip(reference @ t, -1.0, 1.0))))
        # Second derivative bound of a cubic Bezier over the whole parameter
        # range, converted into an angular slack for the sampled maximum.
        p = self.points
        second = 6.0 * max(
            float(np.linalg.norm(p[2] - 2 * p[1] + p[0])),
            float(np.linalg.norm(p[3] - 2 * p[2] + p[1])),
        )
        speed = min(
            float(np.linalg.norm(self.derivative(float(s))))
            for s in np.linspace(lower, upper, samples)
        )
        step = (upper - lower) / (samples - 1)
        return math.degrees(worst + second * step / max(speed, 1e-12))


def _scale(form, factor):
    return ca.Affine(
        form.center * factor,
        {symbol: value * factor for symbol, value in form.deviations.items()},
    )


def _multiply(left, right):
    """Affine product that keeps the nonlinear remainder as a new symbol."""

    deviations = {}
    for symbol, value in left.deviations.items():
        deviations[symbol] = value * right.center
    for symbol, value in right.deviations.items():
        deviations[symbol] = deviations.get(symbol, 0.0) + value * left.center
    remainder = (
        sum(abs(value) for value in left.deviations.values())
        * sum(abs(value) for value in right.deviations.values())
    )
    if remainder:
        deviations[ca.get_next_symbol_index()] = remainder
    return ca.Affine(left.center * right.center, deviations)


def _square(form):
    """Tighter enclosure of ``x**2``: the remainder is nonnegative."""

    radius = sum(abs(value) for value in form.deviations.values())
    deviations = {
        symbol: 2.0 * form.center * value
        for symbol, value in form.deviations.items()
    }
    if radius:
        deviations[ca.get_next_symbol_index()] = 0.5 * radius**2
    return ca.Affine(form.center**2 + 0.5 * radius**2, deviations)
