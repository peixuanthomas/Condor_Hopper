from __future__ import annotations

from dataclasses import dataclass, field, replace
import math

import cvxpy as cp
import numpy as np


@dataclass
class LandingState:
    ut: float
    position: np.ndarray
    velocity: np.ndarray
    mass: float
    dry_mass: float
    thrust: float
    exhaust_velocity: float
    gravity: float
    drag: float
    direction: np.ndarray
    rotation: np.ndarray = field(default_factory=lambda: np.zeros(3))
    radius: float = 600_000.0
    normal: float = 0.0
    aero_speed: float = 1.0


@dataclass
class LandingPlan:
    ignition_ut: float
    touchdown_ut: float
    times: np.ndarray
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    mass: np.ndarray
    external: np.ndarray
    fuel_kg: float
    slack_ms2: float
    rollout_error_m: float


def air_normal(state, speed):
    curve = lambda value: 1.0 + 4.0 / (1.0 + math.exp(np.clip((value - 350.0) / 30.0, -50, 50)))
    return state.normal * speed / max(state.aero_speed, 1.0) * curve(speed) / curve(state.aero_speed)


def external_acceleration(position, velocity, state, thrust=None, vertical_air=True):
    gravity = state.gravity * ((state.radius + state.position[0]) / (state.radius + position[0]))**2
    coefficient = state.drag * math.exp(np.clip((state.position[0] - position[0]) / 5500.0, -8.0, 8.0))
    # 上、北、东是左手坐标；星球自转造成的侧向偏移也必须进入落点预测。
    wu, wn, we = state.rotation
    vu, vn, ve = velocity
    rotation = 2.0 * np.array([wn * ve - we * vn, we * vu - wu * ve, wu * vn - wn * vu])
    rotation += (state.radius + position[0]) * np.array([wn**2, -wu * wn, 0.0])
    speed = np.linalg.norm(velocity)
    direction = state.direction if thrust is None or np.linalg.norm(thrust) < 0.1 else thrust / np.linalg.norm(thrust)
    normal = air_normal(state, speed)
    normal *= math.exp(np.clip((state.position[0] - position[0]) / 5500.0, -8.0, 8.0))
    air = -coefficient * speed * velocity - normal * (velocity - np.dot(velocity, direction) * direction)
    if not vertical_air:
        air[0] = 0.0
    return np.array([-gravity, 0.0, 0.0]) + rotation + air


def steering_force(desired, velocity, axial, normal, reference, tilt_degrees=20.0):
    combined = np.asarray(desired) + (axial + normal) * np.asarray(velocity)
    if abs(combined[0]) < 1.0:
        slope = reference[1:] / max(reference[0], 0.1)
    else:
        slope = combined[1:] / combined[0]
    length = np.linalg.norm(slope)
    slope *= min(1.0, math.tan(math.radians(tilt_degrees)) / max(length, 1e-9))
    direction = np.array([1.0, *slope])
    direction /= np.linalg.norm(direction)
    magnitude = max(0.0, combined[0] / direction[0] - normal * np.dot(velocity, direction))
    return magnitude * direction, direction


def fin_target(height, velocity, error, aero, gravity, normal):
    effective_gravity = max(2.0, gravity - aero[0])
    horizon = max(2.0, (velocity[0] + math.sqrt(velocity[0]**2 + 2 * effective_gravity * height)) / effective_gravity)
    desired = 6.0 * error / horizon**2 - 4.0 * velocity[1:] / horizon
    drag = max(0.0, -np.dot(aero, velocity) / max(np.dot(velocity, velocity), 1.0))
    slope = (desired + (drag + normal) * velocity[1:]) / (normal * min(velocity[0], -20.0))
    retro_slope = -velocity[1:] / max(-velocity[0] + 1.0, 1.0)
    difference = slope - retro_slope
    slope = retro_slope + difference * min(1.0, math.tan(math.radians(20.0)) / max(np.linalg.norm(difference), 1e-9))
    slope *= min(1.0, math.tan(math.radians(30.0)) / max(np.linalg.norm(slope), 1e-9))
    direction = np.array([1.0, *slope])
    return direction / np.linalg.norm(direction)


def propagate(state, duration, acceleration=None, step=0.1, final_acceleration=None, vertical_air=True):
    position, velocity = state.position.copy(), state.velocity.copy()
    elapsed = 0.0
    thrust = np.zeros(3) if acceleration is None else np.asarray(acceleration)
    change = np.zeros(3) if final_acceleration is None else (np.asarray(final_acceleration) - thrust) / duration
    while elapsed < duration - 1e-9:
        dt = min(step, duration - elapsed)
        initial_thrust = thrust + elapsed * change
        a0 = external_acceleration(position, velocity, state, initial_thrust if acceleration is not None else None, vertical_air) + initial_thrust
        vmid = velocity + 0.5 * dt * a0
        pmid = position + 0.5 * dt * velocity
        midpoint_thrust = thrust + (elapsed + 0.5 * dt) * change
        a = external_acceleration(pmid, vmid, state, midpoint_thrust if acceleration is not None else None, vertical_air) + midpoint_thrust
        position += dt * vmid
        velocity += dt * a
        elapsed += dt
    return position, velocity


def shifted(state, ut, position, velocity, mass=None):
    mass = state.mass if mass is None else mass
    gravity = state.gravity * ((state.radius + state.position[0]) / (state.radius + position[0]))**2
    drag = state.drag * state.mass / mass * math.exp(np.clip((state.position[0] - position[0]) / 5500.0, -8, 8))
    return LandingState(ut, position, velocity, mass, state.dry_mass, state.thrust,
                        state.exhaust_velocity, gravity, drag, state.direction.copy(), state.rotation, state.radius,
                        state.normal * state.mass / mass * math.exp(np.clip((state.position[0] - position[0]) / 5500.0, -8, 8)),
                        state.aero_speed)


def stopping_burn(state):
    state = replace(state, drag=0.0, normal=0.0)
    height, speed, mass, elapsed, distance = state.position[0], state.velocity[0], state.mass, 0.0, 0.0
    while speed < -0.7 and elapsed < 90.0:
        dt = 0.05
        position = np.array([height, 0.0, 0.0])
        velocity = np.array([speed, 0.0, 0.0])
        # 点火距离不依赖空气额外制动，并保留油门余量。
        thrust = 0.90 * state.thrust
        accel = thrust / mass + external_acceleration(position, velocity, state)[0]
        travel = speed * dt + 0.5 * accel * dt**2
        height += travel
        distance -= travel
        speed += accel * dt
        mass -= thrust / state.exhaust_velocity * dt
        elapsed += dt
        if mass <= state.dry_mass:
            return math.inf, math.inf
    return distance, elapsed


class LandingSolver:
    """Blackmore 等公开的推力/质量凸化，配合逐次更新的空气阻力。"""

    def __init__(self, nodes=40):
        self.nodes = nodes
        n = nodes
        self.r = cp.Variable((n + 1, 3))
        self.v = cp.Variable((n + 1, 3))
        self.z = cp.Variable(n + 1)
        self.u = cp.Variable((n, 3))
        self.sigma = cp.Variable(n, nonneg=True)
        self.virtual = cp.Variable((n, 3))
        self.r0, self.v0 = cp.Parameter(3), cp.Parameter(3)
        self.dt, self.mass_dt = cp.Parameter(nonneg=True), cp.Parameter(nonneg=True)
        self.ar = [cp.Parameter(3) for _ in range(n)]
        self.av = [cp.Parameter((3, 3)) for _ in range(n)]
        self.au = [cp.Parameter((3, 3)) for _ in range(n)]
        self.bias = cp.Parameter((n, 3))
        self.position_bias = cp.Parameter((n, 3))
        self.reference_thrust = cp.Parameter((n, 3))
        self.thrust_trust = cp.Parameter(n, nonneg=True)
        self.ratio_jacobian = [cp.Parameter((2, 3)) for _ in range(n)]
        self.ratio_offset = cp.Parameter((n, 2))
        self.capture_time = cp.Parameter(n + 1, nonneg=True)
        self.capture_radius = cp.Parameter(n + 1, nonneg=True)
        self.upper, self.upper_z = cp.Parameter(n), cp.Parameter(n, nonneg=True)
        self.minimum_mass = cp.Parameter()
        self.minimum_acceleration = cp.Parameter(nonneg=True)
        self.initial_slope = cp.Parameter(2)
        self.turn_step = cp.Parameter(nonneg=True)
        self.turn_acceleration_step = cp.Parameter(nonneg=True)
        constraints = [self.r[0] == self.r0, self.v[0] == self.v0, self.z[0] == 0,
                       self.r[-1] == 0, self.v[-1] == [-0.7, 0.0, 0.0],
                       self.r[:, 0] >= 0, self.v[:, 0] <= -0.04 * self.r[:, 0] - 0.7, self.z >= self.minimum_mass,
                       self.v[1:, 0] >= self.v[:-1, 0],
                       self.r[1:] == self.r[:-1] + 0.5 * self.dt * (self.v[:-1] + self.v[1:]) + self.position_bias,
                       self.z[1:] == self.z[:-1] - self.mass_dt * self.sigma,
                       cp.norm(self.u, axis=1) <= self.sigma,
                       self.sigma <= self.upper - cp.multiply(self.upper_z, self.z[:-1]),
                       self.u[:, 0] >= math.cos(math.radians(20.0)) * self.sigma,
                       self.u[:, 0] >= self.minimum_acceleration]
        constraints += [cp.norm(self.u - self.reference_thrust, axis=1) <= 25.0,
                        cp.abs(self.u[:, 0] - self.reference_thrust[:, 0]) <= self.thrust_trust,
                        cp.norm(self.r[:, 1:], axis=1) <= math.tan(math.radians(20.0)) * self.r[:, 0] + 1.0,
                        cp.norm(self.v[:, 1:], axis=1) <= 0.025 * self.r[:, 0] + 0.5,
                        cp.norm(self.r[:, 1:] + cp.multiply(0.5 * self.capture_time[:, None], self.v[:, 1:]), axis=1)
                        <= self.capture_radius]
        for k in range(n):
            constraints += [self.v[k + 1] - self.v[k] ==
                            self.ar[k] * 0.5 * (self.r[k, 0] + self.r[k + 1, 0])
                            + self.av[k] @ (0.5 * (self.v[k] + self.v[k + 1]))
                            + self.au[k] @ self.u[k] + self.bias[k] + self.virtual[k]]
        # 飞行中不能瞬间改变推力方向；最后四分之一段只允许小角度刹车。
        slopes = [self.ratio_jacobian[k] @ self.u[k] + self.ratio_offset[k] for k in range(n)]
        constraints += [slopes[0] == self.initial_slope]
        constraints += [cp.norm(slopes[k] - slopes[k - 1]) <= self.turn_step for k in range(1, n)]
        constraints += [cp.norm(slopes[k] - 2 * slopes[k - 1] + slopes[k - 2]) <= self.turn_acceleration_step
                        for k in range(2, n)]
        for k in range(n * 3 // 4, n):
            constraints += [cp.norm(self.r[k, 1:]) <= 0.05 * self.r[k, 0] + 1.0,
                            cp.norm(self.r[k, 1:] + 0.5 * (n - k) * self.dt * self.v[k, 1:]) <= 1.0,
                            cp.norm(self.u[k, 1:]) <= math.tan(math.radians(5.0)) * self.u[k, 0]]
        self.problem = cp.Problem(cp.Minimize(cp.sum(self.sigma) + 10000.0 * cp.norm1(self.virtual)), constraints)

    def solve(self, state, duration):
        if duration < 0.5 or duration > 45.0 or state.position[0] <= 0.0:
            return None
        # 点火后必须由发动机完成制导，不能把预计的空气作用当成可控推力。
        state = replace(state, drag=0.0, normal=0.0)
        n, dt = self.nodes, duration / self.nodes
        times = np.linspace(0.0, duration, n + 1)
        rref = np.linspace(state.position, np.zeros(3), n + 1)
        vref = np.linspace(state.velocity, [-0.7, 0.0, 0.0], n + 1)
        zref = -np.linspace(0.0, min(0.3, state.thrust * duration / state.mass / state.exhaust_velocity), n + 1)
        upref = np.full(n, max(state.gravity + abs(state.velocity[0]) / duration, 1.0))
        uref = np.column_stack((upref, np.zeros((n, 2))))
        self.r0.value, self.v0.value = state.position, state.velocity
        self.dt.value = dt
        self.mass_dt.value = dt / state.exhaust_velocity
        self.minimum_mass.value = math.log(state.dry_mass / state.mass)
        self.minimum_acceleration.value = state.gravity
        self.initial_slope.value = state.direction[1:] / max(state.direction[0], 0.5)
        self.turn_step.value = math.radians(3.0) * dt
        self.turn_acceleration_step.value = 0.03 * dt**2
        for _ in range(12):
            external = []
            biases = []
            position_biases = []
            for k in range(n):
                model = shifted(state, state.ut, state.position, state.velocity,
                                state.mass * math.exp(0.5 * (zref[k] + zref[k + 1])))
                point, speed, thrust = (rref[k] + rref[k + 1]) / 2, (vref[k] + vref[k + 1]) / 2, uref[k]
                force = external_acceleration(point, speed, model, thrust)
                ar = external_acceleration(point + [1.0, 0, 0], speed, model, thrust) - force
                av, au = np.empty((3, 3)), np.empty((3, 3))
                for axis in range(3):
                    dv = np.eye(3)[axis] * 0.1
                    av[:, axis] = (external_acceleration(point, speed + dv, model, thrust) - force) / 0.1
                    du = np.eye(3)[axis] * 0.01
                    au[:, axis] = (external_acceleration(point, speed, model, thrust + du) - force) / 0.01
                self.ar[k].value, self.av[k].value, self.au[k].value = dt * ar, dt * av, dt * (np.eye(3) + au)
                segment = shifted(state, state.ut + k * dt, rref[k], vref[k], model.mass)
                beginning = thrust if k == 0 else 0.5 * (uref[k - 1] + thrust)
                ending = thrust if k == n - 1 else 0.5 * (thrust + uref[k + 1])
                local_r, local_v = propagate(segment, dt / 2, beginning, final_acceleration=thrust)
                segment = shifted(state, state.ut + (k + 0.5) * dt, local_r, local_v, model.mass)
                local_r, local_v = propagate(segment, dt / 2, thrust, final_acceleration=ending)
                velocity_bias = local_v - vref[k] - dt * (force + thrust)
                position_biases.append(local_r - rref[k] - 0.5 * dt * (vref[k] + local_v))
                biases.append(dt * (force - ar * point[0] - av @ speed - au @ thrust) + velocity_bias)
                external.append(force)
            external = np.array(external)
            bound = 0.90 * state.thrust / state.mass * np.exp(-zref[:-1])
            self.upper.value, self.upper_z.value = bound * (1.0 + zref[:-1]), bound
            self.bias.value = np.array(biases)
            self.position_bias.value = np.array(position_biases)
            self.reference_thrust.value = uref
            self.capture_time.value = 2.0 * np.maximum(rref[:, 0], 0.0) / np.maximum(-vref[:, 0] + 0.7, 1.0)
            self.capture_radius.value = 0.5 + 0.1 * np.maximum(rref[:, 0] - 250.0, 0.0)
            self.thrust_trust.value = np.maximum(3.0, 0.5 * upref)
            self.ratio_offset.value = uref[:, 1:] / upref[:, None]
            for k in range(n):
                self.ratio_jacobian[k].value = np.array([[-uref[k, 1] / upref[k]**2, 1.0 / upref[k], 0.0],
                                                        [-uref[k, 2] / upref[k]**2, 0.0, 1.0 / upref[k]]])
            try:
                self.problem.solve(solver="CLARABEL", warm_start=True, max_iter=100)
            except cp.SolverError:
                return None
            if self.problem.status != cp.OPTIMAL:
                return None
            rref, vref, zref = self.r.value.copy(), self.v.value.copy(), self.z.value.copy()
            acceleration = self.u.value.copy()
            uref = acceleration.copy()
            upref = np.maximum(acceleration[:, 0], 1.0)
            # 用原来的非线性运动公式重算，不能只相信凸问题的求解状态。
            position, velocity = state.position.copy(), state.velocity.copy()
            for k in range(n):
                segment = shifted(state, state.ut + k * dt, position, velocity,
                                  state.mass * math.exp(0.5 * (zref[k] + zref[k + 1])))
                beginning = acceleration[k] if k == 0 else 0.5 * (acceleration[k - 1] + acceleration[k])
                ending = acceleration[k] if k == n - 1 else 0.5 * (acceleration[k] + acceleration[k + 1])
                position, velocity = propagate(segment, dt / 2, beginning, final_acceleration=acceleration[k])
                segment = shifted(state, state.ut + (k + 0.5) * dt, position, velocity,
                                  state.mass * math.exp(0.5 * (zref[k] + zref[k + 1])))
                position, velocity = propagate(segment, dt / 2, acceleration[k], final_acceleration=ending)
            error = float(np.linalg.norm(position))
            slack = float(np.max(np.abs(self.sigma.value - np.linalg.norm(acceleration, axis=1))))
            mass = state.mass * np.exp(zref)
            actual_thrust = np.linalg.norm(acceleration, axis=1) * mass[:-1]
            slopes = acceleration[:, 1:] / acceleration[:, :1]
            slew = np.max(np.linalg.norm(np.diff(slopes, axis=0), axis=1)) / dt
            turn_accel = np.max(np.linalg.norm(np.diff(slopes, n=2, axis=0), axis=1)) / dt**2
            remaining = 2.0 * rref[:, 0] / np.maximum(-vref[:, 0] + 0.7, 1.0)
            footprint = np.linalg.norm(rref[:, 1:] + 0.5 * remaining[:, None] * vref[:, 1:], axis=1)
            capture_error = np.max(footprint[rref[:, 0] <= 200.0])
            virtual = np.max(np.linalg.norm(self.virtual.value, axis=1)) / dt
            if (error < 2.0 and np.linalg.norm(velocity - [-0.7, 0, 0]) < 0.5 and slack < 0.01 and virtual < 0.001
                    and np.max(actual_thrust) <= state.thrust and mass[-1] >= state.dry_mass
                    and slew < math.radians(3.5) and turn_accel < 0.04 and capture_error < 0.8):
                external = []
                for k in range(n):
                    model = shifted(state, state.ut, state.position, state.velocity,
                                    state.mass * math.exp(0.5 * (zref[k] + zref[k + 1])))
                    external.append(external_acceleration((rref[k] + rref[k + 1]) / 2,
                                    (vref[k] + vref[k + 1]) / 2, model, acceleration[k]))
                return LandingPlan(state.ut, state.ut + duration, times, rref, vref,
                                   acceleration, mass, np.array(external), float(state.mass - mass[-1]), slack, error)
        return None

    def plan_ignition(self, state):
        coast, current = 0.0, state
        coast_states = [state]
        while coast < 120.0:
            stopping_distance, _ = stopping_burn(current)
            if current.position[0] <= stopping_distance + max(20.0, -0.5 * current.velocity[0]):
                break
            gravity_rotation = external_acceleration(current.position, current.velocity, replace(current, drag=0.0, normal=0.0))
            air = external_acceleration(current.position, current.velocity, current) - gravity_rotation
            speed = np.linalg.norm(current.velocity)
            normal = max(0.002, air_normal(current, speed))
            requested = fin_target(current.position[0], current.velocity, -current.position[1:], air,
                                   current.gravity, normal)
            angle = math.acos(float(np.clip(np.dot(current.direction, requested), -1.0, 1.0)))
            fraction = min(1.0, math.radians(2.5) * 0.5 / max(angle, 1e-9))
            direction = (1.0 - fraction) * current.direction + fraction * requested
            direction /= np.linalg.norm(direction)
            segment = replace(current, direction=(current.direction + direction) / np.linalg.norm(current.direction + direction))
            position, velocity = propagate(segment, 0.5, vertical_air=False)
            coast += 0.5
            current = shifted(state, state.ut + coast, position, velocity)
            current.direction = direction
            coast_states.append(current)
        delays = {max(0.0, coast - offset) for offset in (0.0, 2.0, 5.0, 10.0)}
        delays.add(min(coast, 4.0))
        for delay in sorted(delays, reverse=True):
            candidates = []
            start = coast_states[round(max(0.0, delay - 4.0) / 0.5)]
            while start.ut < state.ut + delay - 1e-6:
                dt = min(0.5, state.ut + delay - start.ut)
                angle = math.acos(float(np.clip(start.direction[0], -1.0, 1.0)))
                fraction = min(1.0, math.radians(2.5) * dt / max(angle, 1e-9))
                direction = (1.0 - fraction) * start.direction + fraction * np.array([1.0, 0.0, 0.0])
                direction /= np.linalg.norm(direction)
                segment = replace(start, direction=(start.direction + direction) / np.linalg.norm(start.direction + direction))
                position, velocity = propagate(segment, dt, vertical_air=False)
                start = shifted(state, start.ut + dt, position, velocity)
                start.direction = direction
            nominal_time = 2.0 * start.position[0] / max(-start.velocity[0], 1.0)
            for factor in (1.0, 1.15, 1.35):
                plan = self.solve(start, nominal_time * factor)
                if plan is not None:
                    candidates.append(plan)
            if candidates:
                return min(candidates, key=lambda plan: plan.fuel_kg)
        return None
