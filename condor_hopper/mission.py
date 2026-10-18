from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace

import numpy as np

from .landing import LandingSolver, LandingState, external_acceleration, steering_force


def unit(vector):
    vector = np.asarray(vector, dtype=float)
    length = np.linalg.norm(vector)
    return vector / length if length > 1e-9 else np.array([1.0, 0.0, 0.0])


def bounded(vector, maximum):
    vector = np.asarray(vector, dtype=float)
    return vector * min(1.0, maximum / max(float(np.linalg.norm(vector)), 1e-9))


def attitude_error(target_local):
    target_local = unit(target_local)
    axis = -np.cross([0.0, 1.0, 0.0], target_local)
    angle = math.acos(float(np.clip(target_local[1], -1.0, 1.0)))
    if np.linalg.norm(axis) < 1e-9:
        axis = np.array([1.0, 0.0, 0.0])
    return unit(axis) * max(0.0, angle - math.radians(0.1))


def descent_attitude(target_local, angular_rate, angular_accel, bias=None, reference_rate=None):
    error = attitude_error(target_local)
    angular_accel = np.clip(np.asarray(angular_accel), 0.02, 0.12)
    target_rate = np.clip(0.8 * error, -0.05, 0.05)
    if reference_rate is not None:
        target_rate = np.clip(target_rate + reference_rate, -0.05, 0.05)
    target_rate[1] = 0.0
    command = 2.5 * (target_rate - angular_rate) / angular_accel
    return np.clip(command + (np.zeros(3) if bias is None else bias), -1.0, 1.0)


@dataclass
class Sample:
    ut: float
    altitude_m: float
    height_m: float
    velocity: np.ndarray
    error: np.ndarray
    direction: np.ndarray
    aero: np.ndarray
    mass_kg: float
    thrust_n: float
    gravity: float
    grounded: int
    target_height_m: float | None = None

    @property
    def landing_height_m(self):
        return self.height_m if self.target_height_m is None else self.target_height_m


@dataclass
class Command:
    throttle: float = 0.0
    direction: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0]))
    engine: bool = False
    fins: bool = False
    legs: bool = False
    hold_attitude: bool = True


class FlightController:
    def __init__(self, cutoff_altitude_m=10000.0, dry_mass=19834.0, exhaust_velocity=3238.0,
                 rotation=None, radius=600_000.0):
        self.cutoff_target_m = cutoff_altitude_m
        self.phase = "ascent"
        self.cutoff_altitude_m = None
        self.cutoff_speed_ms = None
        self.apogee_altitude_m = 0.0
        self.descent_started = None
        self.contact_started = None
        self.success = False
        self.dry_mass = dry_mass
        self.exhaust_velocity = exhaust_velocity
        self.rotation = np.zeros(3) if rotation is None else np.asarray(rotation)
        self.radius = radius
        self.solver = LandingSolver()
        self.worker = ThreadPoolExecutor(max_workers=1)
        self.pending = None
        self.plan = None
        self.plan_number = 0
        self.last_plan_ut = -math.inf
        self.touchdown_ut = None
        self.ignition_ut = None
        self.terminal_acceleration = None
        self.normal_coefficient = 0.025
        self.identified_normal = self.normal_coefficient
        self.capture_missed = False
        self.last_desired_direction = np.array([1.0, 0.0, 0.0])

    def close(self):
        self.worker.shutdown(wait=True, cancel_futures=True)

    def landing_state(self, state):
        speed = max(float(np.linalg.norm(state.velocity)), 1.0)
        along = float(np.dot(state.velocity, state.direction))
        axial = max(0.0, -float(np.dot(state.aero, state.direction)) / min(along, -1.0))
        transverse = state.velocity - along * state.direction
        normal = self.identified_normal
        if np.linalg.norm(transverse) > 1.0:
            estimate = -float(np.dot(state.aero + axial * state.velocity, transverse)) / np.dot(transverse, transverse)
            if 0.0 <= estimate < 0.5:
                normal = estimate
                self.identified_normal = normal
        return LandingState(state.ut, np.array([state.landing_height_m, *(-state.error)]),
                            state.velocity.copy(), state.mass_kg, self.dry_mass,
                            state.thrust_n, self.exhaust_velocity, state.gravity, axial / speed, state.direction.copy(),
                            self.rotation, self.radius, normal, speed)

    def update_plan(self, state):
        if self.pending is not None and self.pending.done():
            result = self.pending.result()
            self.pending = None
            valid = result is not None and result.touchdown_ut > state.ut
            if valid and self.touchdown_ut is not None:
                valid = abs(result.touchdown_ut - self.touchdown_ut) < 0.01
            if valid and self.phase == "descent":
                valid = result.ignition_ut > state.ut - 0.5
            if valid:
                self.plan = result
                self.plan_number += 1
        if self.pending is None and state.ut - self.last_plan_ut >= 1.0 and state.landing_height_m > 200.0:
            if self.phase == "landing":
                position, velocity, _, _ = self.plan.at(state.ut)
                error = position - np.array([state.landing_height_m, *(-state.error)])
                if np.linalg.norm(error) < 10.0 and np.linalg.norm(velocity - state.velocity) < 3.0:
                    return
            initial = self.landing_state(state)
            if self.phase == "landing":
                initial.direction = self.last_desired_direction.copy()
            if self.phase == "descent":
                self.pending = self.worker.submit(self.solver.plan_ignition, initial)
            elif self.touchdown_ut - state.ut > 1.0:
                self.pending = self.worker.submit(self.solver.solve, initial, self.touchdown_ut - state.ut)
            self.last_plan_ut = state.ut

    def fin_direction(self, state):
        velocity = state.velocity
        speed = np.linalg.norm(velocity)
        retrograde = unit(-velocity + np.array([1.0, 0.0, 0.0]))
        effective_gravity = max(2.0, state.gravity - state.aero[0])
        time_to_ground = (velocity[0] + math.sqrt(velocity[0]**2 + 2 * effective_gravity * state.landing_height_m)) / effective_gravity
        burn_time = min(0.5 * time_to_ground, -velocity[0] / max(state.thrust_n / state.mass_kg - state.gravity, 1.0))
        coast_time = max(2.0, time_to_ground - burn_time)
        desired = 2 * (state.error - velocity[1:] * (coast_time + 0.5 * burn_time)) / (coast_time * (coast_time + burn_time))
        drag = max(0.0, -np.dot(state.aero, velocity) / max(speed**2, 1.0))
        normal = velocity - np.dot(velocity, state.direction) * state.direction
        if speed > 80 and np.linalg.norm(normal) > 5.0:
            axis_speed = np.dot(velocity, state.direction)
            axial_drag = max(0.0, -np.dot(state.aero, state.direction) / axis_speed)
            estimate = -np.dot(state.aero + axial_drag * velocity, normal) / np.dot(normal, normal)
            if 0.002 < estimate < 1.0:
                self.normal_coefficient = 0.98 * self.normal_coefficient + 0.02 * estimate
        slope = (desired + (drag + self.normal_coefficient) * velocity[1:]) / (
                 self.normal_coefficient * min(velocity[0], -20.0))
        retro_slope = retrograde[1:] / max(retrograde[0], 0.5)
        slope = retro_slope + bounded(slope - retro_slope, math.tan(math.radians(20.0)))
        return unit([1.0, *bounded(slope, math.tan(math.radians(30.0)))])

    def update(self, state: Sample) -> Command:
        dt = 0.05 if not hasattr(self, "previous_ut") else max(0.02, min(0.2, state.ut - self.previous_ut))
        self.previous_ut = state.ut
        self.apogee_altitude_m = max(self.apogee_altitude_m, state.altitude_m)
        if self.phase == "ascent":
            if state.altitude_m < self.cutoff_target_m:
                return Command(throttle=1.0, engine=True)
            self.cutoff_altitude_m = state.altitude_m
            self.cutoff_speed_ms = float(state.velocity[0])
            self.phase = "coast"

        if self.phase == "coast":
            if state.velocity[0] < -0.5:
                if self.descent_started is None:
                    self.descent_started = state.ut
                if state.ut - self.descent_started >= 0.5:
                    self.phase = "descent"
            else:
                self.descent_started = None
            if self.phase == "coast":
                return Command()

        if self.phase == "descent":
            if state.landing_height_m < 8000.0:
                self.update_plan(state)
            if self.plan is not None and state.ut >= self.plan.ignition_ut:
                self.phase = "landing"
                self.touchdown_ut = self.plan.touchdown_ut
                self.ignition_ut = state.ut
            else:
                if self.plan is None and state.landing_height_m < 600.0 and state.direction[0] > 0.7:
                    raise RuntimeError("未找到能在剩余燃料内到达指定着陆点的轨迹")
                direction = self.fin_direction(state)
                if self.plan is not None and self.plan.ignition_ut - state.ut < 10.0:
                    direction = unit(self.plan.acceleration[0])
                return Command(direction=direction, fins=True, legs=state.landing_height_m < 3000.0)

        if self.phase == "landing" and state.grounded >= 1:
            if abs(float(state.velocity[0])) > 3.0:
                raise RuntimeError("支腿接地时下降速度过大")
            self.phase = "contact"

        if self.phase in {"contact", "landed"}:
            stable = (state.grounded >= 3 and abs(float(state.velocity[0])) <= 1.0
                      and np.linalg.norm(state.velocity[1:]) <= 0.5
                      and state.direction[0] >= math.cos(math.radians(3.0)))
            if not stable:
                self.contact_started = None
            elif self.contact_started is None:
                self.contact_started = state.ut
            elif state.ut - self.contact_started >= 1.0:
                self.phase = "landed"
                self.success = bool(np.linalg.norm(state.error) <= 3.0)
            return Command(fins=True, legs=True, hold_attitude=False)

        if state.direction[0] < math.cos(math.radians(45.0)):
            return Command(fins=True, legs=True)

        if (state.landing_height_m > 200.0 and self.terminal_acceleration is None
                and self.plan is not None):
            self.update_plan(state)
            position, velocity, feedforward, environment = self.plan.at_height(state.landing_height_m)
            actual_position = np.array([state.landing_height_m, *(-state.error)])
            model = self.landing_state(state)
            gravity_rotation = external_acceleration(actual_position, state.velocity, replace(model, drag=0.0, normal=0.0))
            desired = feedforward + environment - gravity_rotation
            desired += bounded(0.15 * (position - actual_position) + 0.7 * (velocity - state.velocity), 3.0)
            acceleration, direction = steering_force(desired, state.velocity, model.drag * model.aero_speed,
                                                       model.normal, feedforward)
        else:
            height = min(state.height_m, state.landing_height_m)
            if self.terminal_acceleration is None:
                self.terminal_acceleration = max(0.5, float((state.velocity[0]**2 - 0.7**2) / max(2 * height, 1.0)))
                self.capture_missed = bool(np.linalg.norm(state.error) > 10.0)
            braking = self.terminal_acceleration
            target_speed = -math.sqrt(0.7**2 + 2 * braking * max(height, 0.0))
            target_accel = braking if height > 0.0 else 0.0
            vertical = max(0.1, state.gravity + target_accel + 1.5 * (target_speed - state.velocity[0]))
            if height < 3.0:
                vertical = max(0.0, min(vertical, state.gravity + (-0.7 - state.velocity[0]) / dt))
            remaining = max(1.0, 2 * max(height, 0.0) / max(-state.velocity[0] + 0.7, 1.0))
            horizontal = -state.velocity[1:] / remaining
            model = self.landing_state(state)
            desired = np.array([vertical, *horizontal])
            acceleration, direction = steering_force(desired, state.velocity, model.drag * model.aero_speed,
                                                       model.normal, np.array([1.0, 0.0, 0.0]), 5.0)
        throttle = float(np.clip(np.linalg.norm(acceleration) * state.mass_kg / state.thrust_n, 0.0, 1.0))
        self.last_desired_direction = direction.copy()
        return Command(throttle=throttle, direction=direction, engine=True, fins=True, legs=True)
