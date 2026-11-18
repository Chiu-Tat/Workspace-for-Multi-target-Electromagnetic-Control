"""Geometry/constitutive checks, not a control simulation."""
import unittest
from pathlib import Path
import numpy as np
from cases.case_7_vessel_microspheres.vessel import Vessel
from cases.case_7_vessel_microspheres.magnetics import (
    coil_basis, field_gradient, aligned_force, induced_force_matrices,
)
from emns.workspace.fast import calculate_B_selective


class VesselMicrosphereTests(unittest.TestCase):
    def test_winding_curve_tangent_matches_geometry(self):
        folder=Path(__file__).resolve().parents[1]/'cases'/'case_7_vessel_microspheres'/'geometries'
        for name in ('branched_parallel','branched_crossing'):
            v=Vessel(folder/(name+'.json'))
            for branch in v.branches:
                for s in (.2,.5,.8):
                    d=(v.centerline(branch,s+1e-6)-v.centerline(branch,s-1e-6))/2e-6
                    np.testing.assert_allclose(v.frame(branch,s)[0],d/np.linalg.norm(d),atol=1e-8)
                    radii=v.radius(branch,np.linspace(0,1,51))
                    self.assertTrue(np.all(np.diff(radii)<=1e-12))

    def test_branched_network_attachments_and_separation(self):
        from cases.case_7_vessel_microspheres.build_networks import network_checks
        folder=Path(__file__).resolve().parents[1]/'cases'/'case_7_vessel_microspheres'/'geometries'
        for name,bound in [('branched_parallel',.5),('branched_crossing',.35)]:
            v=Vessel(folder/(name+'.json'))
            checks=network_checks(v)
            self.assertEqual(checks['segments_per_network'],{'artery':7,'vein':7})
            self.assertGreaterEqual(checks['continuous_gap_lower_bound_mm'],bound-1e-10)
            for branch,metadata in v.branches.items():
                p=v.centerline(branch,.6)
                self.assertGreater(float(v.clearance(p,network=metadata['network'])),.0002)
                other='vein' if metadata['network']=='artery' else 'artery'
                self.assertLess(float(v.clearance(p,network=other)),0)
                if branch in ('RA','RV') and 'main_trunk_pipe' in v.config:
                    self.assertAlmostEqual(v.radius(branch,0),.002)
                    self.assertAlmostEqual(v.radius(branch,1),.002)
                    self.assertEqual(v.config['main_trunk_pipe']['outer_diameter_mm'],8.)
                    self.assertEqual(v.config['main_trunk_pipe']['inner_diameter_mm'],4.)
                else:
                    self.assertLess(v.radius(branch,1),v.radius(branch,0))
            v.branches['RA_B1']['control_points_mm'][0][0]+=1
            with self.assertRaisesRegex(ValueError,'Detached branch'):
                network_checks(v)

    def test_network_selector_rejects_ambiguous_or_unknown_selection(self):
        v=Vessel()
        with self.assertRaises(ValueError):
            v.samples(branch='CCA',network='artery')
        with self.assertRaises(ValueError):
            v.samples(network='unknown')

    def test_separate_variants_and_assigned_lumen(self):
        from cases.case_7_vessel_microspheres.build_variants import separation_certificate
        folder = Path(__file__).resolve().parents[1] / 'cases' / 'case_7_vessel_microspheres' / 'geometries'
        for name, gap in [('close_parallel', .5), ('close_crossing', .35)]:
            v = Vessel(folder / (name+'.json'))
            cert = separation_certificate(v)
            self.assertTrue(cert['separate_lumens'])
            self.assertTrue(cert['global_minimum_attained_at_midpoints'])
            self.assertAlmostEqual(cert['continuous_lumen_gap_lower_bound_mm'], gap, places=10)
            p = v.centerline('RA', .5)
            self.assertGreater(float(v.clearance(p, branch='RA')), .001)
            self.assertLess(float(v.clearance(p, branch='RV')), 0)
            # A deliberate crossing of the separating planes must lose the certificate.
            axis = v.config['separation_axis']
            for point in v.branches['RV']['control_points_mm']:
                point[axis] -= 4
            self.assertFalse(separation_certificate(v)['separate_lumens'])

    def test_connected_branches_and_distinct_outlets(self):
        v = Vessel()
        junction = v.centerline('CCA', 1)
        for b in ('ICA', 'ECA'):
            np.testing.assert_allclose(v.centerline(b, 0), junction)
            self.assertGreater(float(v.clearance(junction)), 0.002)
        self.assertGreater(np.linalg.norm(v.centerline('ICA', 1)-v.centerline('ECA', 1)), 0.02)
        for b in v.branches:
            for s in np.linspace(0, 1, 21):
                self.assertGreater(float(v.clearance(v.point(b, s, 0.7, 1.1))), 0.0004)

    def test_frame_and_units(self):
        v = Vessel()
        np.testing.assert_allclose(v.centerline('CCA', 1), [0, 0, 0.085])
        for b in v.branches:
            frame = np.array(v.frame(b, 0.4))
            np.testing.assert_allclose(frame @ frame.T, np.eye(3), atol=1e-12)
        with self.assertRaises(ValueError):
            v.centerline('CCA', 1.1)

    def test_field_matches_existing_coils_and_derivatives(self):
        rng = np.random.RandomState(17)
        for b in ('CCA', 'ICA', 'ECA'):
            p = Vessel().centerline(b, 0.6)
            current = rng.uniform(-2, 2, 10)
            B, G = field_gradient(p, current)
            legacy = calculate_B_selective(current, *p, {'Bx': True, 'By': True, 'Bz': True})
            np.testing.assert_allclose(B, [legacy[k] for k in ('Bx', 'By', 'Bz')], atol=1e-14)
            h = 1e-7
            fd = np.column_stack([(field_gradient(p+h*e, current)[0]-field_gradient(p-h*e, current)[0])/(2*h)
                                  for e in np.eye(3)])
            np.testing.assert_allclose(G, fd, rtol=1e-7, atol=1e-10)
            np.testing.assert_allclose(G, G.T, atol=1e-13)
            self.assertAlmostEqual(float(np.trace(G)), 0, places=12)

    def test_force_energy_quadratic_and_current_reversal(self):
        p = Vessel().centerline('ICA', 0.5)
        I = np.arange(10)*0.2-0.5
        kappa = 1e-8  # arbitrary test coefficient, not a calibrated sphere
        Q = induced_force_matrices(p, kappa)
        force = aligned_force(p, I, polarizability=kappa)
        np.testing.assert_allclose(force, np.einsum('i,aij,j->a', I, Q, I), rtol=1e-12, atol=1e-24)
        h = 1e-7
        for law in ({'polarizability': kappa}, {'moment': 1e-9}):
            F = aligned_force(p, I, **law)
            np.testing.assert_allclose(F, aligned_force(p, -I, **law), rtol=1e-12, atol=1e-24)
            def energy(x):
                B = field_gradient(x, I)[0]
                return -kappa*np.dot(B, B)/2 if 'polarizability' in law else -law['moment']*np.linalg.norm(B)
            fd = np.array([-(energy(p+h*e)-energy(p-h*e))/(2*h) for e in np.eye(3)])
            np.testing.assert_allclose(F, fd, rtol=1e-6, atol=1e-20)
        with self.assertRaises(ValueError):
            aligned_force(p, np.zeros(10), moment=1e-9)
        np.testing.assert_array_equal(aligned_force(p, np.zeros(10), polarizability=kappa), np.zeros(3))


if __name__ == '__main__':
    unittest.main()
