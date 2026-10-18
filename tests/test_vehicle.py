from types import SimpleNamespace

import numpy as np
import pytest

from condor_hopper.vehicle import KrpcVehicle
from condor_hopper.mission import Command


def fake_vehicle():
    vehicle = KrpcVehicle.__new__(KrpcVehicle)
    body_frame, surface_frame = object(), object()
    calls = []

    def velocity(frame):
        calls.append(frame)
        return (100.0, 3.0, 4.0) if frame is body_frame else (0.0, 0.0, 0.0)

    vessel = SimpleNamespace(parts=SimpleNamespace(all=[1]), mass=1000.0,
                             velocity=velocity, position=lambda frame: (0.0, 0.0, 0.0),
                             direction=lambda frame: (1.0, 0.0, 0.0))
    vehicle.sc = SimpleNamespace(active_vessel=vessel, ut=0.0,
                                 transform_direction=lambda value, source, target: value)
    vehicle.vessel = vessel
    vehicle.body = SimpleNamespace(reference_frame=body_frame)
    vehicle.frame = surface_frame
    vehicle.flight = SimpleNamespace(mean_altitude=100.0, surface_altitude=20.0,
                                     aerodynamic_force=(0.0, 0.0, 0.0))
    vehicle.target_body_position = np.zeros(3)
    vehicle.legs_deployed = False
    vehicle.touchdown_height = 15.0
    vehicle.target_terrain_altitude = 0.0
    vehicle.part_count = 1
    vehicle.engines = [SimpleNamespace(max_thrust=30_000.0, thrust_limit=1.0)]
    vehicle.legs = [SimpleNamespace(is_grounded=False)]
    vehicle.mu, vehicle.radius = 3.5316e12, 600_000.0
    vehicle.velocity_check_state = None
    vehicle.streams = dict(ut=lambda: vehicle.sc.ut, altitude=lambda: vehicle.flight.mean_altitude,
                           surface_altitude=lambda: vehicle.flight.surface_altitude,
                           mass=lambda: vessel.mass, velocity=lambda: vessel.velocity(body_frame),
                           position=lambda: vessel.position(body_frame), direction=lambda: vessel.direction(surface_frame),
                           aero=lambda: vehicle.flight.aerodynamic_force)
    return vehicle, body_frame, calls


def test_velocity_is_read_in_ground_frame_then_rotated_to_surface_axes():
    vehicle, body_frame, calls = fake_vehicle()
    state = vehicle.read()
    assert calls == [body_frame]
    assert np.allclose(state.velocity, [100.0, 3.0, 4.0])
    assert state.height_m == 5.0
    assert state.landing_height_m == 85.0


def test_velocity_inconsistent_with_altitude_aborts_flight():
    vehicle, _, _ = fake_vehicle()
    vehicle.vessel.velocity = lambda frame: (0.0, 0.0, 0.0)
    vehicle.read()
    vehicle.sc.ut = 0.5
    vehicle.flight.mean_altitude = 150.0
    with pytest.raises(RuntimeError, match="高度变化"):
        vehicle.read()


def test_altitude_timestamp_is_captured_before_later_rpc_delay():
    vehicle, _, _ = fake_vehicle()
    vehicle.read()
    vehicle.sc.ut = 1.0
    vehicle.flight.mean_altitude = 200.0

    def delayed_transform(value, source, target):
        vehicle.sc.ut = 2.5
        return value

    vehicle.sc.transform_direction = delayed_transform
    state = vehicle.read()
    assert state.ut == 1.0
    assert state.altitude_m == 200.0


def test_descending_fin_deployment_keeps_measured_sign_and_opens_once():
    vehicle = KrpcVehicle.__new__(KrpcVehicle)
    events = []
    vehicle.fins = [SimpleNamespace(authority_limiter=25.0) for _ in range(4)]
    vehicle.fin_modules = [SimpleNamespace(trigger_event=lambda event: events.append(event)) for _ in range(4)]
    vehicle.fins_deployed = vehicle.legs_deployed = vehicle.engine_active = vehicle.attitude_active = False
    vehicle.control = SimpleNamespace(throttle=0.0, rcs=False, pitch=0.0, yaw=0.0, roll=0.0)
    vehicle.pilot = SimpleNamespace(disengage=lambda: None)
    command = Command(fins=True, hold_attitude=False)
    vehicle.apply(command, None)
    vehicle.apply(command, None)
    assert [fin.authority_limiter for fin in vehicle.fins] == [25.0] * 4
    assert events == ["Extend Fins"] * 4


def test_engine_restart_clears_fin_attitude_bias():
    vehicle = KrpcVehicle.__new__(KrpcVehicle)
    vehicle.fins_deployed = vehicle.legs_deployed = vehicle.attitude_active = False
    vehicle.engine_active = False
    vehicle.manual_descent = True
    vehicle.fins = [SimpleNamespace(pitch_enabled=True, yaw_enabled=True) for _ in range(4)]
    vehicle.engines = [SimpleNamespace(active=False)]
    vehicle.attitude_bias = np.array([0.6, 0.0, -0.8])
    vehicle.control = SimpleNamespace(throttle=0.0, rcs=False)
    vehicle.apply(Command(engine=True, hold_attitude=False), None)
    assert np.allclose(vehicle.attitude_bias, 0.0)
    assert all(not fin.pitch_enabled and not fin.yaw_enabled for fin in vehicle.fins)
