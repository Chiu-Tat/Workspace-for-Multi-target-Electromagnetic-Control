"""Two aligned-moment spheres in the ten-coil field, with mutual interaction.

Built on ``cases/case_7_vessel_microspheres/magnetics.py`` (coil field and
analytic gradient) and the Case 7 THEORY.md section "Mutual interaction":

    B_i^ext = H_i I + D(p_i - p_j) m_j,       m_i = |m| B_i^ext / |B_i^ext|
    F_i     = (grad B_i^ext)^T m_i

Allocation model.  The controller prescribes the *direction* of the net
field at each sphere, u_i, which fixes the moment direction; the magnitude
only has to exceed an alignment threshold.  In a low-viscosity medium the
same direction at both spheres (a near-uniform bias field) is used: different
directions a few centimetres apart need gradients whose forces exceed the
drag budget by orders of magnitude.  With the
moment directions fixed, the field and force rows are affine in the currents:

    [R_1 B_1; R_2 B_2; R_1 F_1; R_2 F_2] = G I + c

where R_i rotates fields into a frame (u_i, e1, e2) and forces into the local
vessel frame (tangent, normal, binormal), and ``c`` holds the other sphere's
dipole field and gradient.  Rows R_i B_i = [|B|, 0, 0] leave 4 direction
equalities and 6 force rows for the 10 coils, so all force components,
including the transverse ones, remain controllable.
"""

from __future__ import annotations

import numpy as np

from cases.case_7_vessel_microspheres.magnetics import coil_basis
from revised_case_studies.cs2_vessel_microspheres import config as K

MU0_4PI = 1e-7


def dipole_field(r, m):
    r = np.asarray(r, float)
    n = np.linalg.norm(r)
    return MU0_4PI * (3 * r * (m @ r) / n ** 5 - m / n ** 3)


def dipole_gradient(r, m):
    """G[a, b] = dB_a / dx_b of a point dipole ``m`` at displacement ``r``."""
    r = np.asarray(r, float)
    n = np.linalg.norm(r)
    d = m @ r
    I3 = np.eye(3)
    return MU0_4PI * (3 * (I3 * d + np.outer(r, m) + np.outer(m, r)) / n ** 5
                      - 15 * np.outer(r, r) * d / n ** 7)


def field_frame(u):
    """Orthonormal rows (u, e1, e2); field rows are expressed in this frame."""
    u = np.asarray(u, float) / np.linalg.norm(u)
    ref = np.eye(3)[np.argmin(np.abs(u))]
    e1 = np.cross(u, ref)
    e1 /= np.linalg.norm(e1)
    return np.array([u, e1, np.cross(u, e1)])


def allocation_rows(p, u, frames, moment=K.MOMENT, interaction=True):
    """G (12 x 10) and c (12) for prescribed moment directions ``u``.

    ``p``, ``u``: (2, 3) positions and unit moment (= net field) directions.
    ``frames``: two vessel frames (t, n, b); force rows are expressed in them.
    Row order: B1 (u, e1, e2), B2 (u, e1, e2), F1 (t, n, b), F2 (t, n, b);
    field rows use ``field_frame(u[i])`` so rows e1, e2 = 0 hold the direction.
    """
    G = np.zeros((12, 10))
    c = np.zeros(12)
    for i in range(2):
        j = 1 - i
        H, D = coil_basis(p[i])
        m_i, m_j = moment * u[i], moment * u[j]
        R = np.array(frames[i])
        Rb = field_frame(u[i])
        G[3 * i:3 * i + 3] = Rb @ H
        G[6 + 3 * i:9 + 3 * i] = R @ np.einsum('abk,a->bk', D, m_i)
        if interaction:
            r = p[i] - p[j]
            c[3 * i:3 * i + 3] = Rb @ dipole_field(r, m_j)
            c[6 + 3 * i:9 + 3 * i] = R @ (dipole_gradient(r, m_j).T @ m_i)
    return G, c


def plant(p, currents, moment=K.MOMENT, interaction=True, iters=6):
    """Actual moments, net fields and forces for a current vector."""
    H = [None, None]
    D = [None, None]
    for i in range(2):
        H[i], D[i] = coil_basis(p[i])
    Bc = [H[i] @ currents for i in range(2)]
    Gc = [np.einsum('abk,k->ab', D[i], currents) for i in range(2)]
    m = [moment * Bc[i] / max(np.linalg.norm(Bc[i]), 1e-12) for i in range(2)]
    B = list(Bc)
    for _ in range(iters if interaction else 1):
        for i in range(2):
            j = 1 - i
            B[i] = Bc[i] + (dipole_field(p[i] - p[j], m[j]) if interaction else 0.0)
        m = [moment * B[i] / max(np.linalg.norm(B[i]), 1e-12) for i in range(2)]
    F = []
    for i in range(2):
        j = 1 - i
        G = Gc[i] + (dipole_gradient(p[i] - p[j], m[j]) if interaction else 0.0)
        F.append(G.T @ m[i])
    return np.array(m), np.array(B), np.array(F)
