from concurrent.futures import Future

import numpy as np
import pytest

from condor_hopper.landing import LandingSolver, LandingState, external_acceleration, steering_force
from condor_hopper.mission import FlightController, Sample, unit


def initial(height=2200.0, speed=-300.0, offset=20.0):
    return LandingState(0.0, np.array([height, offset, 0.0]), np.array([speed, 1.0, 0.0]),
                        46000.0, 20000.0, 1770000.0, 3238.0, 9.81, 0.0001, np.array([1.0, 0.0, 0.0]))


def test_plan_respects_fuel_thrust_slew_and_terminal_conditions():
    state = initial()
    plan = LandingSolver().plan_ignition(state)
    assert plan is not None
    assert plan.touchdown_ut - plan.ignition_ut < 20.0
    assert plan.mass[-1] > state.dry_mass
    assert plan.fuel_kg < 6000.0
    assert np.max(np.linalg.norm(plan.acceleration, axis=1) * plan.mass[:-1]) <= state.thrust
    assert np.all(plan.velocity[:, 0] <= -0.7 + 1e-5)
    assert np.min(np.diff(plan.velocity[:, 0])) >= -1e-5
    assert np.linalg.norm(plan.position[-1]) < 1e-4
    assert np.allclose(plan.velocity[-1], [-0.7, 0.0, 0.0], atol=1e-5)
    assert plan.slack_ms2 < 0.01 and plan.rollout_error_m < 2.0


def test_unreachable_position_is_rejected_instead_of_low_altitude_return():
    assert LandingSolver().plan_ignition(initial(1000.0, -300.0, 1000.0)) is None


def test_late_ignition_relying_on_extra_drag_is_rejected():
    assert LandingSolver().plan_ignition(initial(1600.0)) is None


def test_insufficient_fuel_is_rejected():
    state = initial()
    state.dry_mass = 45000.0
    assert LandingSolver().plan_ignition(state) is None


def test_large_high_altitude_offset_still_has_a_return_trajectory():
    state = LandingState(0.0, np.array([10000.0, 1000.0, -500.0]), np.array([-500.0, -25.0, 12.5]),
                         41100.0, 19834.0, 1850000.0, 3238.0, 9.5, 0.00004,
                         unit([1.0, 0.05, -0.025]), normal=0.02, aero_speed=500.0)
    plan = LandingSolver().plan_ignition(state)
    assert plan is not None
    assert plan.rollout_error_m < 2.0
    assert plan.fuel_kg < state.mass - state.dry_mass


def test_rotation_deflects_equatorial_descent_eastward():
    state = initial()
    baseline = external_acceleration(state.position, state.velocity, state)
    state.rotation = np.array([0.0, 0.00029157, 0.0])
    rotating = external_acceleration(state.position, state.velocity, state)
    assert rotating[2] - baseline[2] == pytest.approx(2 * 0.00029157 * 300.0)


@pytest.mark.parametrize("normal,expected_sign", [(0.01, 1.0), (0.15, -1.0)])
def test_combined_air_and_engine_steering_changes_sign(normal, expected_sign):
    velocity, desired = np.array([-300.0, 0.0, 0.0]), np.array([35.0, 2.0, 0.0])
    force, direction = steering_force(desired, velocity, 0.03, normal, np.array([1.0, 0.0, 0.0]))
    air = -0.03 * velocity - normal * (velocity - np.dot(velocity, direction) * direction)
    assert direction[1] * expected_sign > 0.0
    assert np.allclose(force + air, desired)


@pytest.mark.parametrize("aero_case", ["drag", "transonic", "normal", "entry_error"])
def test_closed_loop_lands_without_hover_with_attitude_delay(aero_case):
    transonic_drop = aero_case == "transonic"
    normal_air = aero_case in {"normal", "entry_error"}
    controller = FlightController()
    controller.phase = "descent"

    def synchronous(function, *args):
        future = Future()
        future.set_result(function(*args))
        return future

    controller.worker.submit = synchronous
    position, velocity = np.array([6000.0, 100.0, -80.0]), np.array([-300.0, 3.0, -2.0])
    if transonic_drop:
        position, velocity = np.array([4000.0, 30.0, 90.0]), np.array([-462.0, -1.0, -15.0])
    if normal_air:
        position, velocity = np.array([8000.0, 350.0, -250.0]), np.array([-450.0, -25.0, 20.0])
    angle, rate = np.zeros(2), np.zeros(2)
    previous_target = np.zeros(2)
    mass, ut, minimum_low_altitude_speed = 46000.0, 0.0, 1000.0
    deadline = None
    entry_error_applied = False
    try:
        while ut < 60.0:
            direction = unit([1.0, *np.tan(angle)])
            coefficient = 0.0001 * np.exp((6000.0 - position[0]) / 5500.0) * 46000.0 / mass
            if transonic_drop:
                coefficient = 0.00008 * np.exp(-position[0] / 5500.0) * 46000.0 / mass
                coefficient *= 1.0 + 1.5 / (1.0 + np.exp((330.0 - np.linalg.norm(velocity)) / 12.0))
            aero = -coefficient * np.linalg.norm(velocity) * velocity
            if normal_air:
                speed = np.linalg.norm(velocity)
                density = np.exp((8000.0 - position[0]) / 5500.0) * 46000.0 / mass
                aero = -0.00005 * density * speed * velocity
                curve = lambda value: 1.0 + 4.0 / (1.0 + np.exp(np.clip((value - 350.0) / 30.0, -50, 50)))
                normal = 0.04 * density * speed / 450.0 * curve(speed) / curve(450.0)
                aero -= normal * (velocity - np.dot(velocity, direction) * direction)
            state = Sample(ut, position[0] + 85.0, position[0], velocity.copy(),
                           -position[1:].copy(), direction, aero, mass, 1770000.0, 9.81, 0)
            command = controller.update(state)
            if aero_case == "entry_error" and command.engine and not entry_error_applied:
                velocity += [-25.0, -5.0, 5.0]
                entry_error_applied = True
            if controller.touchdown_ut is not None:
                if deadline is None:
                    deadline = controller.touchdown_ut
                assert controller.touchdown_ut == deadline
            target = np.arctan2(command.direction[1:], command.direction[0])
            reference_rate = (target - previous_target) / 0.05
            maximum_rate = np.radians(5.0) if command.engine else 0.05
            desired_rate = np.clip(0.8 * (target - angle) + np.clip(reference_rate, -maximum_rate, maximum_rate),
                                   -maximum_rate, maximum_rate)
            previous_target = target.copy()
            maximum_acceleration = 0.12 if command.engine else 0.08
            rate += np.clip(2.5 * (desired_rate - rate), -maximum_acceleration, maximum_acceleration) * 0.05
            angle += rate * 0.05
            actual_direction = unit([1.0, *np.tan(angle)])
            acceleration = command.throttle * 1770000.0 / mass * actual_direction + aero - [9.81, 0.0, 0.0]
            position += velocity * 0.05 + 0.5 * acceleration * 0.05**2
            velocity += acceleration * 0.05
            mass -= command.throttle * 1770000.0 / 3238.0 * 0.05
            ut += 0.05
            if 3.0 < position[0] < 200.0:
                minimum_low_altitude_speed = min(minimum_low_altitude_speed, -velocity[0])
            if position[0] <= 0.0:
                break
        assert position[0] <= 0.0
        assert -1.2 < velocity[0] < 0.0
        assert np.linalg.norm(position[1:]) <= 5.0
        assert np.linalg.norm(velocity[1:]) < 0.5
        assert minimum_low_altitude_speed > 3.0
        assert ut - controller.ignition_ut < (45.0 if normal_air else 35.0)
        assert 46000.0 - mass < (10000.0 if normal_air else 8000.0)
    finally:
        controller.close()
