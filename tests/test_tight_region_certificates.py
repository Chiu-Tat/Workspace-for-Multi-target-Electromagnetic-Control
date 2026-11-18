import numpy as np

from emns.certificates.tight_region import (
    affine_constructive_inside_certificate,
    box_vertices,
    constructive_inside_certificate,
    dual_outside_certificate,
)
from emns.numerics.affine import Affine


def test_constant_redundant_system_is_certified_inside():
    G = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
    tasks = box_vertices([-0.5, -0.5], [0.5, 0.5])
    certificate = constructive_inside_certificate(
        G,
        G,
        tasks,
        np.full(3, -2.0),
        np.full(3, 2.0),
    )
    assert certificate.certified
    assert certificate.spectral_radius < 1.0


def test_varying_scalar_system_uses_bounded_current_recourse():
    certificate = constructive_inside_certificate(
        np.array([[0.9]]),
        np.array([[1.1]]),
        box_vertices([-1.0], [1.0]),
        np.array([-2.0]),
        np.array([2.0]),
    )
    assert certificate.certified
    assert certificate.maximum_correction < 0.2


def test_dual_certificate_proves_uniform_exclusion():
    certificate = dual_outside_certificate(
        np.array([[0.9]]),
        np.array([[1.1]]),
        np.array([[1.0]]),
        np.array([-0.5]),
        np.array([0.5]),
    )
    assert certificate.certified
    assert certificate.separation_margin > 0.4


def test_dual_certificate_does_not_exclude_attainable_task():
    certificate = dual_outside_certificate(
        np.array([[0.9]]),
        np.array([[1.1]]),
        np.array([[0.4]]),
        np.array([-0.5]),
        np.array([0.5]),
    )
    assert not certificate.certified


def test_large_uncertainty_remains_inconclusive_instead_of_false_positive():
    certificate = constructive_inside_certificate(
        np.array([[-1.0]]),
        np.array([[1.0]]),
        np.array([[0.5]]),
        np.array([-2.0]),
        np.array([2.0]),
    )
    assert not certificate.certified


def test_affine_dependency_cancels_shared_uncertainty():
    shared = 12345
    G_affine = np.array(
        [[Affine(1.0, {shared: 0.4}), Affine(1.0, {shared: -0.4})]],
        dtype=object,
    )
    certificate = affine_constructive_inside_certificate(
        G_affine,
        np.array([[-1.0], [1.0]]),
        np.array([-2.0, -2.0]),
        np.array([2.0, 2.0]),
    )
    assert certificate.certified


def test_affine_side_constraint_uses_redundant_actuator():
    certificate = affine_constructive_inside_certificate(
        np.array([[Affine(1.0), Affine(0.0)]], dtype=object),
        np.array([[-1.0], [1.0]]),
        np.array([-2.0, -2.0]),
        np.array([2.0, 2.0]),
        side_G_affine=np.array([[Affine(0.0), Affine(1.0)]], dtype=object),
        side_c_affine=np.array([Affine(0.0)], dtype=object),
        side_lower=np.array([-0.1]),
        side_upper=np.array([0.1]),
    )
    assert certificate.certified
    assert all(abs(current[1]) <= 0.1 for current in certificate.vertex_currents)


def test_affine_side_constraint_rejects_incompatible_task():
    certificate = affine_constructive_inside_certificate(
        np.array([[Affine(1.0)]], dtype=object),
        np.array([[1.0]]),
        np.array([-2.0]),
        np.array([2.0]),
        side_G_affine=np.array([[Affine(1.0)]], dtype=object),
        side_c_affine=np.array([Affine(0.0)], dtype=object),
        side_lower=np.array([-0.1]),
        side_upper=np.array([0.1]),
    )
    assert not certificate.certified
