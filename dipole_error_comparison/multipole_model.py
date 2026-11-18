#!/usr/bin/env python3
"""Compact axial multipole interaction for two uniformly magnetized cylinders.

Decision 1, Stage A.  Each cylinder is an exterior axial multipole expansion
truncated at order L (odd orders only, by symmetry).  For a uniformly magnetized
cylinder of radius R and length Lc with moment m (axial Legendre convention,
Phi = (1/4pi) sum Q_l P_l(cos th) / r^(l+1)):

    Q_1 = m,   Q_3 = m (Lc^2 - 3 R^2) / 4,
    Q_5 = m (3 Lc^4 - 30 Lc^2 R^2 + 30 R^4) / 48.

The outputs match ``run_comparison.finite_interaction``:

    b_avg = source field averaged over the receiver volume,
    F     = total force on the receiver = m grad(u . b_avg)    (U = -m u . b_avg).

With the axial identity  d^l/dz^l (1/r) = (-1)^l l! P_l(cos th) / r^(l+1),

    B_s   = -(mu0/4pi) sum_l Q_l (-1)^l / l! grad (w.grad)^l (1/r),
    b_avg = sum_k V_k / k! (u.grad)^k B_s,   V_k = Q_(k+1) / (m (k+1)),

where w and u are the source and receiver axes and V_k are the receiver's effective harmonic
volume moments (not raw axial moments) (divergence theorem: Q_n = m n V_(n-1)).  The source order is
truncated at L and the receiver order k+1 at L.  L = 1 is the point dipole.
All vectors lie in the support plane, so only in-plane derivatives are needed.
They are computed exactly from the Taylor expansion of 1/r about each point.
"""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

MU04PI = 1e-7


def cylinder_Q(m, length, diameter, order):
    """Axial multipole coefficient Q_order of a uniformly magnetized cylinder."""
    R, L = diameter / 2, length
    if order == 1:
        return m
    if order == 3:
        return m * (L ** 2 - 3 * R ** 2) / 4
    if order == 5:
        return m * (3 * L ** 4 - 30 * L ** 2 * R ** 2 + 30 * R ** 4) / 48
    if order % 2 == 0:
        return 0.0
    return m * _cylinder_Q_numeric(L, R, order)


@lru_cache(maxsize=None)
def _cylinder_Q_numeric(length, radius, order):
    """Q_order / m for odd order >= 7, by Gauss quadrature of the end-face charges."""
    g, w = np.polynomial.legendre.leggauss(40)
    rho = radius * (g + 1) / 2
    h = length / 2
    r = np.sqrt(h * h + rho * rho)
    P = np.polynomial.legendre.Legendre.basis(order)(h / r)
    face = np.sum(w * radius / 2 * 2 * math.pi * rho * r ** order * P)
    M_over_m = 1.0 / (math.pi * radius ** 2 * length)
    return 2 * face * M_over_m             # faces at +h (charge +M) and -h (charge -M, odd P)


def bounding_radius(length, diameter):
    return math.hypot(diameter / 2, length / 2)


# ----------------------------------------------------- polynomial helpers
def pmul(a, b, n):
    """Product of two bivariate coefficient arrays, truncated to total degree n."""
    out = np.zeros((n + 1, n + 1))
    ia, ja = np.nonzero(a)
    ib, jb = np.nonzero(b)
    for i, j in zip(ia, ja):
        for k, l in zip(ib, jb):
            if i + j + k + l <= n:
                out[i + k, j + l] += a[i, j] * b[k, l]
    return out


def derivative_table(X, Y, n):
    """d^(a+b) (1/r) / dx^a dy^b at (X, Y, 0) for a + b <= n, r = sqrt(x^2+y^2+z^2)."""
    r2 = X * X + Y * Y
    delta = np.zeros((n + 1, n + 1))   # (X+s)^2 + (Y+t)^2 - r2 = 2Xs + 2Yt + s^2 + t^2
    delta[1, 0], delta[0, 1], delta[2, 0], delta[0, 2] = 2 * X, 2 * Y, 1.0, 1.0
    series = np.zeros((n + 1, n + 1))
    power = np.zeros((n + 1, n + 1)); power[0, 0] = 1.0
    coef = 1.0                                           # binom(-1/2, k)
    for k in range(n + 1):
        series += coef * r2 ** (-0.5 - k) * power
        power = pmul(power, delta, n)
        coef *= (-0.5 - k) / (k + 1)
    fact = np.array([math.factorial(i) for i in range(n + 1)])
    return series * fact[:, None] * fact[None, :]


def direction_op(v, k, n):
    """Coefficient array of (v.grad)^k as a polynomial in (d/dx, d/dy)."""
    op = np.zeros((n + 1, n + 1)); op[0, 0] = 1.0
    lin = np.zeros((n + 1, n + 1)); lin[1, 0], lin[0, 1] = v
    for _ in range(k):
        op = pmul(op, lin, n)
    return op


def apply(op, table):
    return float(np.sum(op * table))


@lru_cache(maxsize=None)
def _ops(L, w, u):
    """Operators that map the derivative table of 1/r to [b_avg_x, b_avg_y, F_x/m, F_y/m]."""
    n = 2 * L + 1
    ex = np.zeros((n + 1, n + 1)); ex[1, 0] = 1.0
    ey = np.zeros((n + 1, n + 1)); ey[0, 1] = 1.0
    ops = {}
    for l in range(1, L + 1, 2):
        src = direction_op(w, l, n) * ((-1) ** l / math.factorial(l)) * -1.0   # B = -grad phi
        for k in range(0, L, 2):
            rec = direction_op(u, k, n) / math.factorial(k)
            base = pmul(src, rec, n)
            bx, by = pmul(base, ex, n), pmul(base, ey, n)
            ub = u[0] * bx + u[1] * by
            ops[(l, k)] = (bx, by, pmul(ub, ex, n), pmul(ub, ey, n))
    return n, ops


def multipole_interaction(x, y, beta, L, *, free_m, fixed_m, fixed_xy, fixed_beta,
                          moving_cyl, fixed_cyl):
    """Body-frame [b_y, b_z, f_y, f_z] interaction offset of the order-L model."""
    w = (round(math.cos(fixed_beta), 15), round(math.sin(fixed_beta), 15))
    u = (round(math.cos(beta), 15), round(math.sin(beta), 15))
    n, ops = _ops(L, w, u)
    table = derivative_table(x - fixed_xy[0], y - fixed_xy[1], n)
    out = np.zeros(4)
    for (l, k), (bx, by, fx, fy) in ops.items():
        Q = cylinder_Q(fixed_m, fixed_cyl['length_m'], fixed_cyl['diameter_m'], l)
        V = cylinder_Q(free_m, moving_cyl['length_m'], moving_cyl['diameter_m'], k + 1) / (free_m * (k + 1))
        out += Q * V * np.array([apply(bx, table), apply(by, table),
                                 free_m * apply(fx, table), free_m * apply(fy, table)])
    out *= MU04PI
    ey = np.array([-math.sin(beta), math.cos(beta)])
    ez = np.array([math.cos(beta), math.sin(beta)])
    b, F = out[:2], out[2:]
    return np.array([b @ ey, b @ ez, F @ ey, F @ ez])
