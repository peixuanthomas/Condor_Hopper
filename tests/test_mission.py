import time

import numpy as np

from condor_hopper.math3d import angle_deg
from condor_hopper.mission import FlightControlLaw
from condor_hopper.models import MissionConfig, NavigationState
from condor_hopper.vehicle import validate_snapshot
from condor_hopper.simulation import default_condor_model


class FakeVehicle:
    latitude_deg = 0.0

    def available_angular_acceleration(self):
        return np.ones(3)


def make_state(velocity=None):
    model = default_condor_model()
    return NavigationState(
        ut=0.0,
        monotonic_s=time.monotonic(),
        position_m=np.array([15_013.44, 20.0, -10.0]),
        velocity_ms=np.array([0.0, 1.0, -1.0]) if velocity is None else np.asarray(velocity, dtype=float),
        direction=np.array([1.0, 0.0, 0.0]),
        vessel_up=np.array([0.0, 1.0, 0.0]),
        angular_velocity_rads=np.zeros(3),
        mass_kg=model.current_mass_kg,
        available_thrust_n=model.max_thrust_n,
        specific_impulse_s=model.specific_impulse_s,
        aerodynamic_accel_ms2=np.zeros(3),
        pitch_deg=90.0,
        heading_deg=90.0,
        roll_deg=0.0,
        situation="flying",
        legs_grounded=0,
        resource_fraction=0.5,
    )


def test_ascent_completion_keeps_grid_fins_and_legs_stowed():
    config = MissionConfig()
    law = FlightControlLaw(config, FakeVehicle(), 13.44)
    command, _ = law.complete_ascent(make_state(), np.array([15_113.44, 0.0, 0.0]))

    assert command.rcs
    assert not command.brakes
    assert not command.legs
    assert not command.sas_stability
    assert command.direction_control
    assert command.throttle > 0.0
    assert angle_deg(np.array([1.0, 0.0, 0.0]), command.target_direction) <= config.ascent_max_tilt_deg


def test_ascent_plan_command_uses_sas_without_direction_controller():
    config = MissionConfig()
    law = FlightControlLaw(config, FakeVehicle(), 13.44)
    command, _ = law.complete_ascent(
        make_state(),
        np.array([15_113.44, 0.0, 0.0]),
        sas_stability=True,
    )

    assert command.sas_stability
    assert not command.direction_control


def test_coast_holds_up_while_rising_then_tracks_retrograde_while_falling():
    config = MissionConfig()
    law = FlightControlLaw(config, FakeVehicle(), 13.44)

    rising, _ = law.coast(make_state([40.0, 4.0, 0.0]), brakes=False)
    falling, _ = law.coast(make_state([-40.0, 4.0, 0.0]), brakes=True)

    assert np.allclose(rising.target_direction, [1.0, 0.0, 0.0])
    assert not rising.sas_stability and rising.direction_control
    assert not rising.brakes
    assert falling.target_direction[0] > 0.0
    assert falling.target_direction[1] < 0.0
    assert falling.brakes


def test_vehicle_snapshot_validation_checks_expected_hardware():
    config = MissionConfig()
    model = default_condor_model()
    snapshot = {"counts": {"engines": 1, "rcs": 4, "grid_fins": 4, "legs": 4}}

    assert validate_snapshot(snapshot, model, config) == []
    snapshot["counts"]["grid_fins"] = 3
    assert "expected 4 grid_fins, got 3" in validate_snapshot(snapshot, model, config)
