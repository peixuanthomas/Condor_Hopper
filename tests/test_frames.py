import math

import numpy as np

from condor_hopper.frames import kerbin_gravity, local_rotation_vector, rotating_frame_matrices


def test_stock_kerbin_surface_gravity_and_altitude_change():
    surface = kerbin_gravity(0.0, 3.5316e12, 600_000.0)
    high = kerbin_gravity(15_000.0, 3.5316e12, 600_000.0)

    assert math.isclose(surface, 9.81, rel_tol=2e-3)
    assert high < surface


def test_rotating_frame_coriolis_matrix_is_linear_and_finite():
    omega = local_rotation_vector(0.0, 21_549.425)
    rotation_r, rotation_v = rotating_frame_matrices(omega)
    velocity = np.array([100.0, 0.0, 20.0])

    assert rotation_r.shape == (3, 3)
    assert rotation_v.shape == (3, 3)
    assert np.all(np.isfinite(rotation_v @ velocity))

