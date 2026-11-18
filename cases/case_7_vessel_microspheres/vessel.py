"""SI-unit geometry API for swept-ball vessel branches and assigned networks.

The sampled clearance is a conservative interior certificate, not exact
distance to the wall of the continuous union. See THEORY.md.
"""
import json
from pathlib import Path
import numpy as np

CONFIG = Path(__file__).with_name('geometry.json')


class Vessel:
    def __init__(self, config=CONFIG):
        self.config = json.loads(Path(config).read_text(encoding='utf-8'))
        if self.config['units'] != 'mm':
            raise ValueError('Geometry input must use mm')
        self.origin = np.asarray(self.config['origin_emns_mm'], dtype=float) * 1e-3
        self.rotation = np.asarray(self.config['rotation_local_to_emns'], dtype=float)
        if not (np.allclose(self.rotation.T @ self.rotation, np.eye(3))
                and np.isclose(np.linalg.det(self.rotation), 1)):
            raise ValueError('Registration rotation must be a proper orthogonal matrix')
        self.branches = self.config['branches']

    def centerline(self, branch, s):
        """Cubic Bezier c(s), 0 <= s <= 1; output in EMNS metres."""
        s = np.asarray(s, dtype=float)
        if np.any((s < 0) | (s > 1)):
            raise ValueError('s must lie in [0, 1]')
        p = np.asarray(self.branches[branch]['control_points_mm']) * 1e-3
        weights = np.stack(((1-s)**3, 3*(1-s)**2*s, 3*(1-s)*s*s, s**3), axis=-1)
        local = weights @ p
        b = self.branches[branch]
        amplitude = np.asarray(b.get('waviness_mm', [0, 0, 0])) * 1e-3
        phase = b.get('waviness_phase', 0.0)
        wave = np.sin(np.pi*s)**2 * np.sin(2*np.pi*s+phase)
        local = local + wave[..., None]*amplitude
        return local @ self.rotation.T + self.origin

    def radius(self, branch, s):
        """Radius of the generating ball, in metres."""
        b = self.branches[branch]
        s = np.asarray(s, dtype=float)
        base = b['radius_start_mm'] * (1-s) + b['radius_end_mm'] * s
        if b.get('smooth_taper', False):
            taper = s*s*(3-2*s)
            base = b['radius_start_mm']*(1-taper)+b['radius_end_mm']*taper
        bulb = b['bulb_amplitude_mm'] * np.exp(-((s-b['bulb_center_s']) / b['bulb_width_s'])**2)
        return (base + bulb) * 1e-3

    def frame(self, branch, s):
        """Tangent and orthonormal cross-section basis at a scalar s."""
        p = np.asarray(self.branches[branch]['control_points_mm']) * 1e-3
        d = 3*((1-s)**2*(p[1]-p[0]) + 2*(1-s)*s*(p[2]-p[1]) + s*s*(p[3]-p[2]))
        b = self.branches[branch]
        phase = b.get('waviness_phase', 0.0)
        wave_d = np.pi*np.sin(2*np.pi*s)*np.sin(2*np.pi*s+phase) + 2*np.pi*np.sin(np.pi*s)**2*np.cos(2*np.pi*s+phase)
        d += wave_d*np.asarray(b.get('waviness_mm',[0,0,0]))*1e-3
        t = self.rotation @ d
        t /= np.linalg.norm(t)
        ref = np.eye(3)[np.argmin(np.abs(t))]
        n = np.cross(t, ref)
        n /= np.linalg.norm(n)
        return t, n, np.cross(t, n)

    def point(self, branch, s, rho=0.0, theta=0.0):
        """Interior disk point: c + rho*r*(cos(theta)*n + sin(theta)*b)."""
        if not 0 <= rho <= 1:
            raise ValueError('rho must lie in [0, 1]')
        _, n, b = self.frame(branch, s)
        return self.centerline(branch, s) + rho*self.radius(branch, s)*(np.cos(theta)*n + np.sin(theta)*b)

    def samples(self, count=201, branch=None, network=None):
        if count < 2:
            raise ValueError('At least two samples per branch are required')
        s = np.linspace(0, 1, count)
        if branch is not None and network is not None:
            raise ValueError('Select a branch or a network, not both')
        branches = list(self.branches) if branch is None else [branch]
        if network is not None:
            branches = [b for b in branches if self.branches[b].get('network') == network]
            if not branches:
                raise ValueError('Unknown or empty network: '+str(network))
        return (np.concatenate([self.centerline(b, s) for b in branches]),
                np.concatenate([self.radius(b, s) for b in branches]))

    def clearance(self, points, count=201, branch=None, network=None):
        """max(r_k - |p-c_k|); positive means inside a sampled generating ball.

        clearance >= particle_radius + margin is sufficient for wall clearance.
        Negative values do NOT certify outside the continuous vessel.
        Supply branch for assigned-vessel containment in separate-channel tasks.
        Supply network to include a trunk and its connected side branches.
        """
        p = np.asarray(points, dtype=float)
        shape = p.shape[:-1]
        p = p.reshape(-1, 3)
        centers, radii = self.samples(count, branch=branch, network=network)
        result = np.empty(len(p))
        for start in range(0, len(p), 2048):
            distances = np.linalg.norm(p[start:start+2048, None, :] - centers[None, :, :], axis=-1)
            result[start:start+2048] = np.max(radii - distances, axis=1)
        return result.reshape(shape)
