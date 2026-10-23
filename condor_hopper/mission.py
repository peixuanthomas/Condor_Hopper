from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace

import numpy as np

from .landing import LANDING_DISTANCE_TOLERANCE_M, LandingSolver, LandingState, steering_force, fin_target, external_acceleration


DEFAULT_CUTOFF_ALTITUDE_M = 10000.0


def unit(vector):
    vector = np.asarray(vector, dtype=float)
    length = np.linalg.norm(vector)
    return vector / length if length > 1e-9 else np.array([1.0, 0.0, 0.0])


def attitude_error(target_local):
    target_local = unit(target_local)
    axis = -np.cross([0.0, 1.0, 0.0], target_local)
    angle = math.acos(float(np.clip(target_local[1], -1.0, 1.0)))
    if np.linalg.norm(axis) < 1e-9:
        axis = np.array([1.0, 0.0, 0.0])
    return unit(axis) * max(0.0, angle - math.radians(0.1))


def descent_attitude(target_local, angular_rate, angular_accel, bias=None, reference_rate=None, max_rate=0.05):
    error = attitude_error(target_local)
    angular_accel = np.maximum(np.asarray(angular_accel), 0.02)
    target_rate = 0.8 * error
    if reference_rate is not None:
        target_rate += np.clip(reference_rate, -max_rate, max_rate)
    target_rate = np.clip(target_rate, -max_rate, max_rate)
    target_rate[1] = 0.0
    command = 4.0 * (target_rate - angular_rate) / angular_accel
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
    def __init__(self, cutoff_altitude_m=DEFAULT_CUTOFF_ALTITUDE_M, dry_mass=19834.0, exhaust_velocity=3238.0,
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
        self.pending_phase = "descent"
        self.plan = None
        self.plan_number = 0
        self.last_plan_ut = -math.inf
        self.touchdown_ut = None
        self.ignition_ut = None
        self.terminal_acceleration = None
        self.identified_normal = 0.025
        self.capture_missed = False
        self.last_desired_direction = np.array([1.0, 0.0, 0.0])
        self.previous_velocity = None
        self.horizontal_acceleration = np.zeros(2)
        self.horizontal_braking = False

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
            valid = result is not None and result.touchdown_ut > state.ut and self.pending_phase == self.phase
            if valid and self.phase == "descent":
                valid = result.ignition_ut > state.ut - 0.5
                if self.plan is not None:
                    valid = valid and result.ignition_ut <= self.plan.ignition_ut + 0.01
            if valid:
                self.plan = result
                self.plan_number += 1
                if self.phase == "landing":
                    self.touchdown_ut = result.touchdown_ut
        if self.pending is not None or state.ut - self.last_plan_ut < 1.0 or state.landing_height_m <= 200.0:
            return
        if self.phase == "landing":
            if self.plan is None or self.horizontal_braking or self.terminal_acceleration is not None:
                return
            position, velocity, _, _ = self.plan.at(state.ut)
            error = position - np.array([state.landing_height_m, *(-state.error)])
            if np.linalg.norm(error) <= 20.0 and np.linalg.norm(velocity - state.velocity) <= 5.0:
                return
            self.pending = self.worker.submit(self.solver.plan_burn, self.landing_state(state))
        else:
            initial = self.landing_state(state)
            self.pending = self.worker.submit(self.solver.plan_ignition, initial, self.plan)
        self.pending_phase = self.phase
        self.last_plan_ut = state.ut

    def fin_direction(self, state):
        model = self.landing_state(state)
        return fin_target(state.landing_height_m, state.velocity, state.error, state.aero,
                          state.gravity, max(0.002, model.normal), self.plan, state.ut)

    def update(self, state: Sample) -> Command:
        elapsed = 0.05 if not hasattr(self, "previous_ut") else state.ut - self.previous_ut
        dt = max(0.02, min(0.2, elapsed))
        if self.previous_velocity is not None and elapsed > 0.001:
            measured = (state.velocity[1:] - self.previous_velocity[1:]) / elapsed
            self.horizontal_acceleration += elapsed / (0.2 + elapsed) * (measured - self.horizontal_acceleration)
        self.previous_velocity = state.velocity.copy()
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

        deploy_legs = state.landing_height_m <= 800.0
        if self.phase == "descent":
            if state.landing_height_m < 12000.0:
                self.update_plan(state)
            if self.plan is not None and state.ut >= self.plan.ignition_ut:
                self.phase = "landing"
                self.touchdown_ut = self.plan.touchdown_ut
                self.ignition_ut = state.ut
            else:
                if self.plan is None and state.landing_height_m < 600.0 and state.direction[0] > 0.7:
                    raise RuntimeError("未找到能在剩余燃料内到达指定着陆点的轨迹")
                direction = self.fin_direction(state)
                return Command(direction=direction, fins=True, legs=deploy_legs)

        if self.phase == "landing" and state.grounded >= 1:
            if abs(float(state.velocity[0])) > 3.0:
                raise RuntimeError("支腿接地时下降速度过大")
            self.phase = "contact"

        if self.phase in {"contact", "landed"}:
            if state.direction[0] < math.cos(math.radians(30.0)):
                raise RuntimeError("接地后侧翻，停止本次试飞")
            stable = (state.grounded >= 3 and abs(float(state.velocity[0])) <= 1.0
                      and np.linalg.norm(state.velocity[1:]) <= 0.5
                      and state.direction[0] >= math.cos(math.radians(3.0)))
            if not stable:
                self.contact_started = None
            elif self.contact_started is None:
                self.contact_started = state.ut
            elif state.ut - self.contact_started >= 1.0:
                self.phase = "landed"
                self.success = bool(np.linalg.norm(state.error) <= LANDING_DISTANCE_TOLERANCE_M)
            return Command(fins=True, legs=True, hold_attitude=False)

        if state.direction[0] < math.cos(math.radians(45.0)):
            return Command(fins=True, legs=deploy_legs)

        self.update_plan(state)
        if (state.landing_height_m > 200.0 and self.terminal_acceleration is None
                and self.plan is not None):
            actual_position = np.array([state.landing_height_m, *(-state.error)])
            position, velocity, thrust, external = self.plan.at(state.ut)
            model = self.landing_state(state)
            horizon = max(3.0, min(8.0, self.plan.touchdown_ut - state.ut))
            gravity_rotation = external_acceleration(model.position, model.velocity, replace(model, drag=0.0, normal=0.0))
            desired = thrust + external - gravity_rotation
            desired += 6.0 * (position - actual_position) / horizon**2 + 4.0 * (velocity - state.velocity) / horizon
            braking = (state.velocity[0]**2 - 0.7**2) / max(2.0 * state.height_m, 1.0)
            desired[0] = max(desired[0], state.gravity + braking)
            acceleration, direction = steering_force(desired, state.velocity, model.drag * model.aero_speed,
                                                       model.normal, unit(thrust), brake_horizontal=self.horizontal_braking)
        else:
            height = state.height_m
            self.horizontal_braking = True
            if self.terminal_acceleration is None:
                self.terminal_acceleration = max(0.5, float((state.velocity[0]**2 - 0.7**2) / max(2 * height, 1.0)))
                self.capture_missed = bool(np.linalg.norm(state.error) > LANDING_DISTANCE_TOLERANCE_M)
            braking = self.terminal_acceleration
            target_speed = -math.sqrt(0.7**2 + 2 * braking * max(height, 0.0))
            target_accel = braking if height > 0.0 else 0.0
            vertical = max(0.1, state.gravity + target_accel + 1.5 * (target_speed - state.velocity[0]))
            if height < 3.0:
                vertical = max(0.0, min(vertical, state.gravity + (-0.7 - state.velocity[0]) / dt))
            model = self.landing_state(state)
            desired = np.array([vertical, 0.0, 0.0])
            acceleration, direction = steering_force(desired, state.velocity, model.drag * model.aero_speed,
                                                       model.normal, np.array([1.0, 0.0, 0.0]),
                                                       brake_horizontal=True)
        throttle = float(np.clip(np.linalg.norm(acceleration) * state.mass_kg / state.thrust_n, 0.0, 1.0))
        self.last_desired_direction = direction.copy()
        return Command(throttle=throttle, direction=direction, engine=True, fins=True, legs=deploy_legs)
