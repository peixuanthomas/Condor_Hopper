import numpy as np

from condor_hopper.math3d import angle_deg, limit_tilt, manual_attitude_controls, normalize


def test_limit_tilt_enforces_requested_cone():
    direction = limit_tilt(np.array([0.5, 0.5, 0.0]), 8.0)

    assert abs(np.linalg.norm(direction) - 1.0) < 1e-12
    assert angle_deg(np.array([1.0, 0.0, 0.0]), direction) <= 8.0 + 1e-9


def test_manual_attitude_damps_roll_when_direction_is_aligned():
    command = manual_attitude_controls(
        current_direction=np.array([1.0, 0.0, 0.0]),
        vessel_up_direction=np.array([0.0, 1.0, 0.0]),
        angular_velocity=np.array([0.1, 0.0, 0.0]),
        target_direction=np.array([1.0, 0.0, 0.0]),
        max_angular_acc=np.array([1.0, 1.0, 1.0]),
    )

    assert command.roll < 0.0
    assert abs(command.pitch) < 1e-12
    assert abs(command.yaw) < 1e-12


def test_normalize_uses_vertical_fallback_for_zero_vector():
    assert np.array_equal(normalize(np.zeros(3)), np.array([1.0, 0.0, 0.0]))

