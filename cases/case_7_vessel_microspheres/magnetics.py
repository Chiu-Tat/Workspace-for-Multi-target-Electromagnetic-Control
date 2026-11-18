"""Field and aligned-sphere force primitives; no controller or time simulation.

Uses the existing ten-coil dipole parameters, with a full analytic derivative
of that field. All inputs and outputs use SI units.
"""
import numpy as np
from emns.workspace.fast import params_list


def coil_basis(position):
    """Return H[3,10] (T/A), D[3,3,10] (T/m/A), D[a,b]=dB_a/dx_b."""
    p = np.asarray(position, dtype=float)
    if p.shape != (3,) or not np.all(np.isfinite(p)):
        raise ValueError('position must be a finite 3-vector in metres')
    params = np.asarray(params_list)
    m, r = params[:, :3], p - params[:, 3:]
    norm = np.linalg.norm(r, axis=1)
    if np.any(norm < 1e-9):
        raise ValueError('Point is at a coil dipole singularity')
    dot = np.sum(m*r, axis=1)
    H = 1e-7*(3*r*dot[:, None]/norm[:, None]**5 - m/norm[:, None]**3)
    D = np.empty((3, 3, len(params)))
    for a in range(3):
        for b in range(3):
            D[a, b] = 1e-7*(3*((a == b)*dot + r[:, a]*m[:, b] + m[:, a]*r[:, b])/norm**5
                              - 15*r[:, a]*r[:, b]*dot/norm**7)
    return H.T, D


def field_gradient(position, currents):
    H, D = coil_basis(position)
    currents = np.asarray(currents, dtype=float)
    if currents.shape != (H.shape[1],) or not np.all(np.isfinite(currents)):
        raise ValueError('One finite current per coil is required')
    return H @ currents, np.einsum('abk,k->ab', D, currents)


def aligned_force(position, currents, *, moment=None, polarizability=None, minimum_field=1e-9):
    """F = G.T @ m; select constant |m| (A m^2) OR m=kappa B (A m^2/T).

    Coil-only approximation: mutual sphere fields and flow are not included.
    A zero-field induced sphere has zero force; a constant-moment orientation
    is undefined there and is rejected. minimum_field is a numerical threshold.
    """
    if (moment is None) == (polarizability is None):
        raise ValueError('Select exactly one magnetization law')
    value = moment if moment is not None else polarizability
    if not np.isfinite(value) or value <= 0:
        raise ValueError('Magnetic coefficient must be finite and positive')
    B, G = field_gradient(position, currents)
    if moment is not None:
        norm = np.linalg.norm(B)
        if norm <= minimum_field:
            raise ValueError('Aligned constant moment requires nonzero field')
        m = moment * B / norm
    else:
        m = polarizability * B
    return G.T @ m


def induced_force_matrices(position, polarizability):
    """Q[axis,10,10] so F_axis=I.T Q_axis I for an induced dipole."""
    if polarizability <= 0:
        raise ValueError('polarizability must be positive')
    H, D = coil_basis(position)
    return np.stack([polarizability * (H.T @ D[:, a, :] + D[:, a, :].T @ H)/2
                     for a in range(3)])
