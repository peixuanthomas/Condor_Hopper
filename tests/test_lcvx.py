import time

import numpy as np

from condor_hopper.lcvx import LCvxPlanner, predict_coast_state, stopping_distance_m
from condor_hopper.models import FlightPhase, MissionConfig, NavigationState, VehicleModel


def state(position=(13.4, 0.0, 0.0), velocity=(0.0, 0.0, 0.0), mass=10_000.0):
    return NavigationState(
        ut=0.0,
        monotonic_s=time.monotonic(),
        position_m=np.array(position, dtype=float),
        velocity_ms=np.array(velocity, dtype=float),
        direction=np.array([1.0, 0.0, 0.0]),
        vessel_up=np.array([0.0, 1.0, 0.0]),
        angular_velocity_rads=np.zeros(3),
        mass_kg=mass,
        available_thrust_n=300_000.0,
        specific_impulse_s=320.0,
        aerodynamic_accel_ms2=np.zeros(3),
        pitch_deg=90.0,
        heading_deg=90.0,
        roll_deg=0.0,
        situation="simulation",
        legs_grounded=0,
        resource_fraction=1.0,
    )


def model():
    return VehicleModel("test", 10_000.0, 4_000.0, 10_000.0, 300_000.0, 320.0, 1, 4, 4, 4)


def test_fixed_duration_lcvx_hits_terminal_state():
    config = MissionConfig(node_count=21, fuel_reserve_fraction=0.05, glideslope_deg=45.0)
    planner = LCvxPlanner(config)
    plan = planner.solve_fixed_duration(
        state(),
        model(),
        np.array([513.4, 0.0, 0.0]),
        np.zeros(3),
        20.0,
        FlightPhase.ASCENT,
        13.4,
        8.0,
    )

    assert plan is not None
    assert plan.diagnostics.accepted
    assert np.linalg.norm(plan.positions_m[:, -1] - np.array([513.4, 0.0, 0.0])) < 1e-3
    assert np.linalg.norm(plan.velocities_ms[:, -1]) < 1e-3
    assert np.all(np.linalg.norm(plan.thrust_accel_ms2, axis=0) <= plan.thrust_slack_ms2 + 1e-4)


def test_coast_prediction_falls_and_stopping_distance_is_positive():
    config = MissionConfig()
    falling = state(position=(10_013.4, 0.0, 0.0), velocity=(-50.0, 0.0, 0.0))
    predicted = predict_coast_state(falling, 2.0, config, 13.4)

    assert predicted.position_m[0] < falling.position_m[0]
    assert predicted.velocity_ms[0] < falling.velocity_ms[0]
    assert stopping_distance_m(-100.0, 300_000.0, 10_000.0, 9.81) > 0.0

