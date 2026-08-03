from __future__ import annotations

import math
from typing import Any

import numpy as np

from .math3d import angle_rad, normalize


def safe_read(reader, fallback):
    try:
        return reader()
    except Exception:
        return fallback


def surface_position(body: Any, latitude: float, longitude: float, reference_frame: Any | None = None):
    reference_frame = body.reference_frame if reference_frame is None else reference_frame
    biome = safe_read(lambda: body.biome_at(latitude, longitude), "")
    if biome == "Water":
        return body.msl_position(latitude, longitude, reference_frame)
    return body.surface_position(latitude, longitude, reference_frame)


def create_launch_reference_frame(space_center: Any, body: Any, latitude: float, longitude: float):
    """Create a body-fixed frame whose axes are up, north-like, east-like."""
    body_frame = body.reference_frame
    position = np.asarray(surface_position(body, latitude, longitude, body_frame), dtype=float)
    target_up = normalize(position)
    x_axis = np.array([1.0, 0.0, 0.0])
    axis = np.cross(target_up, x_axis)
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm <= 1e-12:
        rotation = (1.0, 0.0, 0.0, 0.0)
    else:
        half_angle = angle_rad(target_up, x_axis) / 2.0
        rotation = (math.cos(half_angle), *(math.sin(half_angle) * axis / axis_norm))
    return space_center.ReferenceFrame.create_relative(body_frame, tuple(position), rotation=tuple(rotation))


def kerbin_gravity(altitude_m: float, mu_m3s2: float, radius_m: float) -> float:
    radius = max(radius_m + altitude_m, radius_m * 0.5)
    return mu_m3s2 / (radius * radius)


def local_rotation_vector(latitude_deg: float, rotation_period_s: float) -> np.ndarray:
    omega = math.tau / rotation_period_s
    latitude = math.radians(latitude_deg)
    return np.array([omega * math.sin(latitude), omega * math.cos(latitude), 0.0])


def rotating_frame_matrices(rotation_vector_rads: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    wx, wy, wz = np.asarray(rotation_vector_rads, dtype=float)
    cross = np.array([[0.0, -wz, wy], [wz, 0.0, -wx], [-wy, wx, 0.0]])
    return -(cross @ cross), -2.0 * cross

