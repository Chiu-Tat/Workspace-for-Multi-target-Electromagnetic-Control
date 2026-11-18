"""Dipole coil field and rotation-plane basis used in the phantom trials."""

import numpy as np


def coil_basis(point, parameters):
    """Field per ampere (3x10, T/A) and gradient dB_a/dx_b per ampere (3x3x10)."""
    parameters = np.asarray(parameters, float)
    r = np.asarray(point, float)-parameters[:, 3:]
    m = parameters[:, :3]
    n = np.linalg.norm(r, axis=1)
    if not np.isfinite(r).all() or np.any(n < 1e-6):
        raise ValueError('Invalid coil evaluation point')
    dot = np.sum(m*r, axis=1)
    field = 1e-7*(3*r*dot[:, None]/n[:, None]**5-m/n[:, None]**3)
    gradient = np.empty((3, 3, len(m)))
    for a in range(3):
        for b in range(3):
            gradient[a, b] = 1e-7*(3*((a == b)*dot+r[:, a]*m[:, b]+m[:, a]*r[:, b])/n**5
                                   - 15*r[:, a]*r[:, b]*dot/n**7)
    return field.T, gradient


def rotation_plane(axis):
    n = np.asarray(axis, float)
    norm = np.linalg.norm(n)
    if n.shape != (3,) or not np.isfinite(n).all() or norm < 1e-9:
        raise ValueError('Rotation axis must be a finite non-zero 3-vector')
    n = n/norm
    reference = np.array([0., 0., 1.]) if abs(n[2]) < .9 else np.array([1., 0., 0.])
    u = np.cross(n, reference)
    u /= np.linalg.norm(u)
    return n, u, np.cross(n, u)
