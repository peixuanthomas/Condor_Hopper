from __future__ import annotations

import math

import numpy as np

from .models import AttitudeCommand


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def normalize(vector: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    value = np.asarray(vector, dtype=float)
    magnitude = float(np.linalg.norm(value))
    if not math.isfinite(magnitude) or magnitude <= 1e-12:
        return np.array([1.0, 0.0, 0.0] if fallback is None else fallback, dtype=float)
    return value / magnitude


def angle_rad(a: np.ndarray, b: np.ndarray) -> float:
    a_n = normalize(a)
    b_n = normalize(b)
    return math.acos(clamp(float(np.dot(a_n, b_n)), -1.0, 1.0))


def angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    return math.degrees(angle_rad(a, b))


def rotate(axis: np.ndarray, vector: np.ndarray, angle: float) -> np.ndarray:
    axis_n = normalize(axis, np.zeros(3))
    if float(np.linalg.norm(axis_n)) <= 1e-12:
        return np.asarray(vector, dtype=float)
    vector = np.asarray(vector, dtype=float)
    return (
        vector * math.cos(angle)
        + np.cross(axis_n, vector) * math.sin(angle)
        + axis_n * float(np.dot(axis_n, vector)) * (1.0 - math.cos(angle))
    )


def limit_tilt(direction: np.ndarray, max_tilt_deg: float) -> np.ndarray:
    up = np.array([1.0, 0.0, 0.0])
    direction = normalize(direction)
    angle = angle_rad(up, direction)
    limit = math.radians(max_tilt_deg)
    if angle <= limit:
        return direction
    axis = np.cross(up, direction)
    return normalize(rotate(axis, up, limit))


def manual_attitude_controls(
    current_direction: np.ndarray,
    vessel_up_direction: np.ndarray,
    angular_velocity: np.ndarray,
    target_direction: np.ndarray,
    target_roll_rad: float | None = None,
    settling_time: float = 0.45,
    max_angular_acc: np.ndarray | tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> AttitudeCommand:
    """Return kRPC roll/yaw/pitch inputs using the project's verified axis convention."""
    current_direction = normalize(current_direction)
    target_direction = normalize(target_direction)
    vessel_up_direction = normalize(vessel_up_direction, np.array([0.0, 1.0, 0.0]))
    angular_velocity = np.asarray(angular_velocity, dtype=float)
    max_acc = np.maximum(np.asarray(max_angular_acc, dtype=float), 1e-4)
    settling_time = max(settling_time, 0.05)

    global_up = np.array([1.0, 0.0, 0.0])
    roll_axis = np.cross(current_direction, global_up)
    aligned_up = rotate(roll_axis, vessel_up_direction, angle_rad(current_direction, global_up))
    current_roll = angle_rad(aligned_up, np.array([0.0, 1.0, 0.0]))
    if angle_rad(aligned_up, np.array([0.0, 0.0, 1.0])) > math.pi / 2.0:
        current_roll = -current_roll

    body_x = current_direction
    body_y = np.array([0.0, math.cos(current_roll), math.sin(current_roll)])
    body_y = rotate(np.cross(global_up, body_x), body_y, angle_rad(global_up, body_x))
    body_y = normalize(body_y, np.array([0.0, 1.0, 0.0]))
    body_z = normalize(np.cross(body_x, body_y), np.array([0.0, 0.0, 1.0]))

    direction_angle = angle_rad(current_direction, target_direction)
    direction_axis = -np.cross(current_direction, target_direction)
    if float(np.linalg.norm(direction_axis)) <= 1e-9:
        direction_axis = body_y
    direction_error = normalize(direction_axis) * direction_angle

    roll_velocity = body_x * float(np.dot(angular_velocity, body_x))
    direction_velocity = angular_velocity - roll_velocity
    direction_accel = 4.0 * direction_error / (settling_time**2) - 4.0 * direction_velocity / settling_time

    if target_roll_rad is None:
        roll_accel = -roll_velocity * 4.0 / settling_time
    else:
        roll_error = target_roll_rad - current_roll
        roll_error = (roll_error + math.pi) % math.tau - math.pi
        roll_error *= -1.0
        roll_accel_scalar = 4.0 * roll_error / (settling_time**2) - 4.0 * float(np.dot(roll_velocity, body_x)) / settling_time
        roll_accel = body_x * roll_accel_scalar

    return AttitudeCommand(
        roll=clamp(float(np.dot(roll_accel, body_x)) / max_acc[0], -1.0, 1.0),
        yaw=clamp(float(np.dot(direction_accel, body_y)) / max_acc[1], -1.0, 1.0),
        pitch=clamp(float(np.dot(direction_accel, body_z)) / max_acc[2], -1.0, 1.0),
    )


def finite_vector(value: object, fallback: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=float)
        if result.shape != (3,) or not np.all(np.isfinite(result)):
            raise ValueError
        return result
    except Exception:
        return np.asarray(fallback, dtype=float)

