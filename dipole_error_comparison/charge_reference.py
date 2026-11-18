#!/usr/bin/env python3
"""High-accuracy reference for two uniformly magnetized cylinders (charge model).

A uniformly magnetized cylinder is exactly equivalent to surface magnetic charge
sigma = +/-M on its two end faces (M = m / (pi R^2 L)).  In free space:

  * source field:  B(x) = (mu0/4pi) sum_s q_s (x - x_s) / |x - x_s|^3,
  * force on the receiver: F = sum_r q_r B(x_r)  (receiver end-face charges),
  * b_avg: source field averaged over the receiver volume,

with q = sigma dA.  The face integrals use Gauss-Legendre nodes in radius (with
the rho weight) and the trapezoidal rule in azimuth (spectrally accurate for a
periodic, smooth integrand); the receiver volume adds Gauss-Legendre nodes in
the axial direction.  Outputs match run_comparison.finite_interaction:
body-frame [b_y, b_z, f_y, f_z].  Accuracy is estimated by refinement; it is a
numerical reference, not a certified bound.
"""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

MU04PI = 1e-7


@lru_cache(maxsize=16)
def face_nodes(radius, nr, nphi):
    g, w = np.polynomial.legendre.leggauss(nr)
    rho = radius * (g + 1) / 2
    wr = w * radius / 2 * rho                      # includes the rho area weight
    phi = 2 * math.pi * np.arange(nphi) / nphi
    P = np.array([[r * math.cos(p), r * math.sin(p)] for r in rho for p in phi])
    W = np.array([wr_i * 2 * math.pi / nphi for wr_i in wr for _ in phi])
    return P, W                                    # sum W = pi R^2


@lru_cache(maxsize=16)
def volume_nodes(radius, length, nr, nphi, nz):
    P2, W2 = face_nodes(radius, nr, nphi)
    g, w = np.polynomial.legendre.leggauss(nz)
    z = length / 2 * g
    wz = w * length / 2
    P = np.array([[p[0], p[1], zi] for zi in z for p in P2])
    W = np.array([wzi * wi for wzi in wz for wi in W2])
    return P, W / W.sum()                          # averaging weights


def frame(beta):
    ez = np.array([math.cos(beta), math.sin(beta), 0.0])
    ey = np.array([-math.sin(beta), math.cos(beta), 0.0])
    ex = np.cross(ey, ez)
    return np.column_stack((ex, ey, ez))


def face_charges(center, beta, m, length, diameter, nr, nphi):
    R = diameter / 2
    M = m / (math.pi * R * R * length)
    P2, W = face_nodes(R, nr, nphi)
    Rm = frame(beta)
    pts, q = [], []
    for s in (+1, -1):
        local = np.column_stack([P2, np.full(len(P2), s * length / 2)])
        pts.append(local @ Rm.T + center)
        q.append(s * M * W)
    return np.vstack(pts), np.concatenate(q)


def field(points, src_pts, src_q):
    r = points[:, None, :] - src_pts[None, :, :]
    d3 = np.linalg.norm(r, axis=2) ** 3
    return MU04PI * np.einsum('ij,ijk->ik', src_q[None, :] / d3, r)


def charge_interaction(x, y, beta, *, free_m, fixed_m, fixed_xy, fixed_beta, z_plane,
                       moving_cyl, fixed_cyl, nr=10, nphi=40, nz=10):
    src_pts, src_q = face_charges(np.array([fixed_xy[0], fixed_xy[1], z_plane]), fixed_beta,
                                  fixed_m, fixed_cyl['length_m'], fixed_cyl['diameter_m'], nr, nphi)
    c = np.array([x, y, z_plane])
    rec_pts, rec_q = face_charges(c, beta, free_m, moving_cyl['length_m'], moving_cyl['diameter_m'], nr, nphi)
    F = rec_q @ field(rec_pts, src_pts, src_q)
    Vp, Vw = volume_nodes(moving_cyl['diameter_m'] / 2, moving_cyl['length_m'], nr, nphi, nz)
    b_avg = Vw @ field(Vp @ frame(beta).T + c, src_pts, src_q)
    Rm = frame(beta)
    return np.array([b_avg @ Rm[:, 1], b_avg @ Rm[:, 2], F @ Rm[:, 1], F @ Rm[:, 2]])
