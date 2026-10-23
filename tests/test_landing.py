from concurrent.futures import Future

import numpy as np
import pytest

from condor_hopper.landing import LandingSolver, LandingState, external_acceleration, steering_force, propagate, shifted, coast_step, stopping_burn
from condor_hopper.mission import FlightController, Sample, unit


def initial(height=2200.0, speed=-300.0, offset=20.0):
    return LandingState(0.0, np.array([height, offset, 0.0]), np.array([speed, 1.0, 0.0]),
                        46000.0, 20000.0, 1770000.0, 3238.0, 9.81, 0.0001, np.array([1.0, 0.0, 0.0]))


def attitude_step(command, previous_target, angle, rate):
    target = np.arctan2(command.direction[1:], command.direction[0])
    change = target - previous_target
    turn_rate = np.radians(4.0 if command.engine else 2.5)
    target = previous_target + change * min(1.0, turn_rate * 0.05 / max(np.linalg.norm(change), 1e-9))
    reference_rate = (target - previous_target) / 0.05
    maximum_rate = np.radians(5.0) if command.engine else 0.05
    desired_rate = np.clip(0.8 * (target - angle) + np.clip(reference_rate, -maximum_rate, maximum_rate),
                           -maximum_rate, maximum_rate)
    maximum_acceleration = 0.12 if command.engine else 0.08
    angular_accel = np.clip(4.0 * (desired_rate - rate), -maximum_acceleration, maximum_acceleration)
    rate = rate + angular_accel * 0.05
    angle = angle + rate * 0.05
    return unit([1.0, *np.tan(angle)]), target, angle, rate, angular_accel


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
    assert abs(plan.position[-1, 0]) < 1e-4
    assert np.linalg.norm(plan.position[-1, 1:]) <= 10.0 + 1e-4
    assert np.allclose(plan.velocity[-1], [-0.7, 0.0, 0.0], atol=1e-5)
    assert plan.slack_ms2 < 0.01 and plan.rollout_error_m < 2.0
    for k in range(len(plan.times)):
        model = shifted(state, plan.times[k], plan.position[k], plan.velocity[k], plan.mass[k])
        distance, _ = stopping_burn(model)
        assert distance <= plan.position[k, 0] + 2.0


def test_planned_burn_reaches_ground_with_the_measured_air_forces():
    state = initial()
    plan = LandingSolver().solve(state, 15.0)
    assert plan is not None
    position, velocity = state.position.copy(), state.velocity.copy()
    dt = plan.times[1]
    for k, thrust in enumerate(plan.acceleration):
        segment = shifted(state, state.ut + k * dt, position, velocity,
                          0.5 * (plan.mass[k] + plan.mass[k + 1]))
        beginning = thrust if k == 0 else 0.5 * (plan.acceleration[k - 1] + thrust)
        ending = thrust if k == len(plan.acceleration) - 1 else 0.5 * (thrust + plan.acceleration[k + 1])
        position, velocity = propagate(segment, dt / 2.0, beginning, final_acceleration=thrust)
        segment = shifted(state, segment.ut + dt / 2.0, position, velocity, segment.mass)
        position, velocity = propagate(segment, dt / 2.0, thrust, final_acceleration=ending)
    assert np.linalg.norm(position - plan.position[-1]) < 2.0
    assert np.linalg.norm(velocity - [-0.7, 0.0, 0.0]) < 0.5


def test_coast_prediction_includes_vertical_air_braking():
    state = initial(height=8000.0, speed=-300.0, offset=0.0)
    state.velocity[1] = 0.0
    current = state
    for _ in range(40):
        current = coast_step(current, 0.2, target_direction=np.array([1.0, 0.0, 0.0]))
    height, speed = state.position[0], state.velocity[0]
    for _ in range(1600):
        gravity = state.gravity * ((state.radius + state.position[0]) / (state.radius + height))**2
        drag = state.drag * np.exp((state.position[0] - height) / 5500.0)
        acceleration = -gravity - drag * abs(speed) * speed
        height += speed * 0.005 + 0.5 * acceleration * 0.005**2
        speed += acceleration * 0.005
    assert abs(current.position[0] - height) < 1.0
    assert abs(current.velocity[0] - speed) < 0.2


def test_accepted_ignition_entry_matches_the_fin_commands_used_before_it():
    state = initial(height=6000.0, speed=-300.0, offset=100.0)
    plan = LandingSolver().plan_ignition(state)
    assert plan is not None and plan.ignition_ut > state.ut
    current = state
    while current.ut < plan.ignition_ut - 1e-6:
        current = coast_step(current, min(0.5, plan.ignition_ut - current.ut), plan)
    assert np.linalg.norm(current.position - plan.position[0]) < 5.0
    assert np.linalg.norm(current.velocity - plan.velocity[0]) < 1.0


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


def test_similar_air_and_engine_forces_cancel_velocity_without_chasing_position():
    velocity = np.array([-300.0, 10.0, -5.0])
    desired = np.array([35.0, 20.0, -10.0])
    force, direction = steering_force(desired, velocity, 0.03, 0.1, np.array([1.0, 0.0, 0.0]))
    assert np.allclose(direction[1:] / direction[0], -velocity[1:] / 320.0)
    air = -0.03 * velocity - 0.1 * (velocity - np.dot(velocity, direction) * direction)
    assert np.dot(velocity[1:], (force + air)[1:]) < 0.0
    assert (force + air)[0] == pytest.approx(desired[0])


@pytest.mark.parametrize("normal", [0.0, 0.1, 0.5])
@pytest.mark.parametrize("vertical_speed", [-150.0, -30.0, -0.7])
def test_terminal_braking_reduces_horizontal_speed_with_air_or_thrust_dominant(normal, vertical_speed):
    velocity = np.array([vertical_speed, 5.0, -3.0])
    force, direction = steering_force([30.0, 0.0, 0.0], velocity, 0.01, normal,
                                      np.array([1.0, 0.0, 0.0]), brake_horizontal=True)
    air = -0.01 * velocity - normal * (velocity - np.dot(velocity, direction) * direction)
    acceleration = force + air
    assert np.dot(velocity[1:], acceleration[1:]) < 0.0
    assert acceleration[0] == pytest.approx(30.0)


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
    actual_ignition = None
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
            if controller.ignition_ut is not None:
                if actual_ignition is None:
                    actual_ignition = controller.ignition_ut
                assert controller.ignition_ut == actual_ignition
            actual_direction, previous_target, angle, rate, _ = attitude_step(command, previous_target, angle, rate)
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
        assert np.linalg.norm(position[1:]) <= 20.0
        assert np.linalg.norm(velocity[1:]) < 0.5
        assert minimum_low_altitude_speed > 3.0
        assert ut - controller.ignition_ut < (45.0 if normal_air else 35.0)
        assert 46000.0 - mass < (10000.0 if normal_air else 8000.0)
    finally:
        controller.close()


@pytest.mark.parametrize("air_scale", [0.8, 1.0, 1.2])
def test_last_kilometre_cancels_horizontal_speed_before_contact(air_scale):
    controller = FlightController()
    controller.phase = "landing"
    # 取侧翻试飞在约一千米处的实测入口，加入喷口转向时的反向侧推。
    position, velocity = np.array([993.44, -10.517, 24.879]), np.array([-205.081, -0.668, -3.998])
    direction = unit([1.0, -0.045, 0.080])
    angle = np.arctan2(direction[1:], direction[0])
    previous_target, rate = angle.copy(), np.zeros(2)
    mass, ut, minimum_speed = 38315.0, 0.0, 1000.0
    try:
        while ut < 15.0 and position[0] > 0.0:
            speed = np.linalg.norm(velocity)
            density = np.exp((1000.0 - position[0]) / 5500.0) * 38315.0 / mass
            drag, normal = 0.0359 * density * speed / 205.0, air_scale * 0.0839 * density * speed / 205.0
            aero = -drag * velocity - normal * (velocity - np.dot(velocity, direction) * direction)
            command = controller.update(Sample(ut, position[0] + 85.0, position[0], velocity.copy(),
                -position[1:].copy(), direction, aero, mass, 1783000.0, 9.81, 0))
            direction, previous_target, angle, rate, angular_accel = attitude_step(command, previous_target, angle, rate)
            acceleration = command.throttle * 1783000.0 / mass * direction + aero - [9.81, 0.0, 0.0]
            acceleration[1:] -= 4.0 * angular_accel
            position += velocity * 0.05 + 0.5 * acceleration * 0.05**2
            velocity += acceleration * 0.05
            mass -= command.throttle * 1783000.0 / 3238.0 * 0.05
            ut += 0.05
            if 3.0 < position[0] < 200.0:
                minimum_speed = min(minimum_speed, -velocity[0])
        assert position[0] <= 0.0
        assert np.linalg.norm(position[1:]) <= 20.0
        assert np.linalg.norm(velocity[1:]) < 0.5
        assert -1.2 < velocity[0] < 0.0
        assert np.degrees(np.arccos(direction[0])) < 3.0
        assert minimum_speed > 3.0
    finally:
        controller.close()


@pytest.mark.parametrize("air_scale", [0.8, 1.0, 1.2])
def test_latest_flight_entry_brakes_before_horizontal_velocity_reverses(air_scale):
    controller = FlightController()
    controller.phase = "landing"
    # 20261023_151818_568456 的一千米入口，复现接地前横移反向增大的问题。
    position = np.array([989.0886, -73.1160, 189.2073])
    velocity = np.array([-176.0055, -7.3157, 16.8770])
    direction = unit([0.97447043, 0.12068729, -0.18932018])
    angle = np.arctan2(direction[1:], direction[0])
    target, rate = angle.copy(), np.zeros(2)
    mass, ut, previous_speed = 38122.55, 0.0, None
    try:
        while ut < 25.0 and position[0] > 0.0:
            speed = np.linalg.norm(velocity)
            density = np.exp((989.0886 - position[0]) / 5500.0) * 38122.55 / mass
            drag = 0.0326539 * density * speed / 176.9661
            normal = air_scale * 0.1090533 * density * speed / 176.9661
            aero = -drag * velocity - normal * (velocity - np.dot(velocity, direction) * direction)
            command = controller.update(Sample(ut, position[0] + 85.0, position[0], velocity.copy(),
                -position[1:].copy(), direction, aero, mass, 1782861.125, 9.81, 0))
            direction, target, angle, rate, angular_accel = attitude_step(command, target, angle, rate)
            acceleration = command.throttle * 1782861.125 / mass * direction + aero - [9.81, 0.0, 0.0]
            acceleration[1:] -= 4.0 * angular_accel
            position += velocity * 0.05 + 0.5 * acceleration * 0.05**2
            velocity += acceleration * 0.05
            mass -= command.throttle * 1782861.125 / 3238.0 * 0.05
            ut += 0.05
            horizontal_speed = np.linalg.norm(velocity[1:])
            if position[0] < 200.0:
                if previous_speed is not None and horizontal_speed > 0.5:
                    assert horizontal_speed <= previous_speed + 1e-5
                previous_speed = horizontal_speed
        assert controller.horizontal_braking
        assert position[0] <= 0.0
        assert -1.2 < velocity[0] < 0.0
        assert np.linalg.norm(velocity[1:]) < 0.5
        assert np.degrees(np.arccos(direction[0])) < 3.0
    finally:
        controller.close()
