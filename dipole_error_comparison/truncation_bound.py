#!/usr/bin/env python3
"""Explicit truncation bound tau_{L,r}(s) for two uniformly magnetized cylinders.

Charge picture: each cylinder is surface charge q on its end faces, with total
absolute charge  Qbar = sum |q| = 2 m / L_c  and all charges within its
bounding sphere of radius a.  With d = p_r - p_s (|d| = s), receiver offsets xi
(|xi| <= a_r) and source offsets eta (|eta| <= a_s),

    1/|d + xi - eta| = sum_n T_n(xi - eta),   T_n = D^n(1/r)(d)[.,...,.] / n!.

The n-th derivative tensor of 1/r has norm n!/s^(n+1); for symmetric
multilinear forms on a Hilbert space the multilinear and polynomial norms agree
(Banach).  The bihomogeneous part of T_n of degree l in eta and l' in xi is
therefore bounded by  C(n, l) a_s^l a_r^l' / s^(n+1),  n = l + l'.
A gradient adds one derivative:  (n+1) C(n, l) a_s^l a_r^l' / s^(n+2).

Parts with even l (source) vanish because the source charge is odd under
inversion; for the force the same holds for even l' (receiver charge), and for
the receiver-volume average the odd-degree volume moments vanish.  The order-L
model keeps

    force:  odd l <= L and odd l' <= L,
    b_avg:  odd l <= L and even k <= L - 1   (k = receiver volume-moment degree).

The bounds below sum every other (nonvanishing) part.  They bound the Euclidean
norm of the in-plane vector, hence every body-frame component.
"""
from __future__ import annotations

import math

MU04PI = 1e-7


def _g(t, parity):
    """sum over n of the given parity of (n+1) t^n, for |t| < 1."""
    a, b = 1 / (1 - t) ** 2, 1 / (1 + t) ** 2
    return (a + b) / 2 if parity == 0 else (a - b) / 2


def tau(s, L, *, m_s, len_s, a_s, m_r, len_r, a_r):
    """(tau_b [T], tau_f [N]) at centre separation s for the order-L model.

    With x = a_s/s and y = a_r/s, the odd-source parts of degree n sum to
    S(n) = ((y+x)^n - (y-x)^n)/2.  The force keeps odd receiver degrees (n even)
    and the field average even ones (n odd), so the full sums have the closed
    forms below; the tail is the full sum minus the finite kept part.
    """
    if s <= a_s + a_r:
        return math.inf, math.inf
    Qs, Qr = 2 * m_s / len_s, 2 * m_r / len_r
    x, y = a_s / s, a_r / s
    full_f = (_g(x + y, 0) - _g(y - x, 0)) / 2
    full_b = (_g(x + y, 1) - _g(y - x, 1)) / 2
    kept_f = sum(math.comb(l + lp, l) * x ** l * y ** lp * (l + lp + 1)
                 for l in range(1, L + 1, 2) for lp in range(1, L + 1, 2))
    kept_b = sum(math.comb(l + k, l) * x ** l * y ** k * (l + k + 1)
                 for l in range(1, L + 1, 2) for k in range(0, L, 2))
    tf = max(full_f - kept_f, 0.0) / s ** 2
    tb = max(full_b - kept_b, 0.0) / s ** 2
    return MU04PI * Qs * tb, MU04PI * Qs * Qr * tf
