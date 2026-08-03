from __future__ import annotations

import math
import time
import warnings
from dataclasses import dataclass
from typing import Iterable

import cvxpy as cp
import numpy as np

from .frames import kerbin_gravity, local_rotation_vector, rotating_frame_matrices
from .models import FlightPhase, G0, GuidancePlan, MissionConfig, NavigationState, SolverDiagnostics, VehicleModel


OPTIMAL_STATUSES = {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}


@dataclass(slots=True)
class _Template:
    duration_s: float
    node_count: int
    problem: cp.Problem
    r: cp.Variable
    v: cp.Variable
    u: cp.Variable
    sigma: cp.Variable
    dz: cp.Variable
    r_initial: cp.Parameter
    v_initial: cp.Parameter
    r_final: cp.Parameter
    v_final: cp.Parameter
    gravity: cp.Parameter
    aero: cp.Parameter
    rotation_r: cp.Parameter
    rotation_v: cp.Parameter
    z_reference: cp.Parameter
    z_initial_delta: cp.Parameter
    z_upper_delta: cp.Parameter
    log_minimum_mass: cp.Parameter
    mu_lower: cp.Parameter
    mu_upper: cp.Parameter
    alpha: cp.Parameter
    cos_tilt: cp.Parameter
    ground_height: cp.Parameter
    glideslope_inverse_tan: cp.Parameter


class LCvxPlanner:
    """Parameterized three-degree-of-freedom lossless-convexification planner."""

    def __init__(self, config: MissionConfig) -> None:
        self.config = config
        self._templates: dict[tuple[float, int], _Template] = {}
        self.last_diagnostics: dict[float, SolverDiagnostics] = {}

    @property
    def cached_template_count(self) -> int:
        return len(self._templates)

    def solve_fixed_duration(
        self,
        state: NavigationState,
        vehicle: VehicleModel,
        target_position_m: np.ndarray,
        target_velocity_ms: np.ndarray,
        duration_s: float,
        phase: FlightPhase,
        ground_height_m: float,
        max_tilt_deg: float,
        min_throttle: float | None = None,
        max_throttle: float | None = None,
        warm_start: bool = True,
    ) -> GuidancePlan | None:
        duration_s = float(duration_s)
        template = self._template(duration_s, self.config.node_count)
        min_throttle = self.config.min_throttle if min_throttle is None else min_throttle
        max_throttle = self.config.max_plan_throttle if max_throttle is None else max_throttle
        if not 0.0 <= min_throttle < max_throttle <= 1.0:
            raise ValueError("invalid thrust bounds")

        mass0 = max(float(state.mass_kg), vehicle.dry_mass_kg + 1.0)
        thrust = max(float(state.available_thrust_n), vehicle.max_thrust_n)
        isp = max(float(state.specific_impulse_s), vehicle.specific_impulse_s, 1.0)
        rho_lower = min_throttle * thrust
        rho_upper = max_throttle * thrust
        alpha = 1.0 / (isp * G0)
        times = np.linspace(0.0, duration_s, self.config.node_count)
        max_burn_mass = mass0 - alpha * rho_upper * times
        min_burn_mass = mass0 - alpha * rho_lower * times
        if float(np.min(max_burn_mass)) <= 1.0 or float(np.min(min_burn_mass)) <= 1.0:
            return None
        z_reference = np.log(max_burn_mass)
        z_upper = np.log(min_burn_mass)
        z_upper_delta = np.maximum(z_upper - z_reference, 0.0)
        mu_lower = rho_lower * np.exp(-z_reference)
        mu_upper = rho_upper * np.exp(-z_reference)

        altitude = float(state.position_m[0] - ground_height_m)
        gravity_magnitude = kerbin_gravity(altitude, self.config.kerbin_mu_m3s2, self.config.kerbin_radius_m)
        omega = local_rotation_vector(0.0, self.config.kerbin_rotation_period_s)
        rotation_r, rotation_v = rotating_frame_matrices(omega)
        minimum_mass = vehicle.minimum_landing_mass(self.config.fuel_reserve_fraction)

        template.r_initial.value = np.asarray(state.position_m, dtype=float)
        template.v_initial.value = np.asarray(state.velocity_ms, dtype=float)
        template.r_final.value = np.asarray(target_position_m, dtype=float)
        template.v_final.value = np.asarray(target_velocity_ms, dtype=float)
        template.gravity.value = np.array([-gravity_magnitude, 0.0, 0.0])
        template.aero.value = np.asarray(state.aerodynamic_accel_ms2, dtype=float)
        template.rotation_r.value = rotation_r
        template.rotation_v.value = rotation_v
        template.z_reference.value = z_reference
        template.z_initial_delta.value = math.log(mass0) - z_reference[0]
        template.z_upper_delta.value = z_upper_delta
        template.log_minimum_mass.value = math.log(max(minimum_mass, vehicle.dry_mass_kg + 1.0))
        template.mu_lower.value = mu_lower
        template.mu_upper.value = mu_upper
        template.alpha.value = alpha
        template.cos_tilt.value = math.cos(math.radians(max_tilt_deg))
        template.ground_height.value = float(ground_height_m)
        template.glideslope_inverse_tan.value = 1.0 / math.tan(math.radians(self.config.glideslope_deg))

        started = time.perf_counter()
        diagnostics = SolverDiagnostics(duration_s=duration_s, solver="CLARABEL")
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="You are solving a parameterized problem that is not DPP.*")
                warnings.filterwarnings("ignore", message="Solution may be inaccurate.*")
                objective = template.problem.solve(
                    solver="CLARABEL",
                    warm_start=warm_start,
                    tol_gap_abs=1e-7,
                    tol_feas=1e-7,
                    tol_gap_rel=1e-7,
                    max_iter=250,
                    verbose=False,
                )
            diagnostics.status = str(template.problem.status)
            diagnostics.objective = float(objective) if objective is not None else float("nan")
            diagnostics.solve_time_s = float(template.problem.solver_stats.solve_time or 0.0)
            diagnostics.iterations = int(template.problem.solver_stats.num_iters or 0)
        except Exception as exc:
            diagnostics.status = "error"
            diagnostics.message = f"{type(exc).__name__}: {exc}"
        diagnostics.wall_time_s = time.perf_counter() - started
        self.last_diagnostics[duration_s] = diagnostics

        if template.problem.status not in OPTIMAL_STATUSES or any(
            variable.value is None for variable in (template.r, template.v, template.u, template.sigma, template.dz)
        ):
            return None

        positions = np.asarray(template.r.value, dtype=float)
        velocities = np.asarray(template.v.value, dtype=float)
        accelerations = np.asarray(template.u.value, dtype=float)
        slack = np.asarray(template.sigma.value, dtype=float)
        dz = np.asarray(template.dz.value, dtype=float)
        masses = np.exp(dz + z_reference)
        self._populate_residuals(
            diagnostics,
            positions,
            velocities,
            accelerations,
            slack,
            duration_s,
            template.gravity.value,
            template.aero.value,
            rotation_r,
            rotation_v,
            np.asarray(target_position_m, dtype=float),
            np.asarray(target_velocity_ms, dtype=float),
        )
        if not diagnostics.accepted:
            diagnostics.message = diagnostics.message or "solution residuals exceeded flight acceptance limits"
            return None

        return GuidancePlan(
            phase=phase,
            created_monotonic_s=state.monotonic_s,
            duration_s=duration_s,
            times_s=times,
            positions_m=positions,
            velocities_ms=velocities,
            thrust_accel_ms2=accelerations,
            thrust_slack_ms2=slack,
            masses_kg=masses,
            diagnostics=diagnostics,
        )

    def solve_best_duration(
        self,
        state: NavigationState,
        vehicle: VehicleModel,
        target_position_m: np.ndarray,
        target_velocity_ms: np.ndarray,
        durations_s: Iterable[float],
        phase: FlightPhase,
        ground_height_m: float,
        max_tilt_deg: float,
        min_throttle: float | None = None,
    ) -> GuidancePlan | None:
        feasible: list[GuidancePlan] = []
        for duration in durations_s:
            plan = self.solve_fixed_duration(
                state=state,
                vehicle=vehicle,
                target_position_m=target_position_m,
                target_velocity_ms=target_velocity_ms,
                duration_s=duration,
                phase=phase,
                ground_height_m=ground_height_m,
                max_tilt_deg=max_tilt_deg,
                min_throttle=min_throttle,
            )
            if plan is not None:
                feasible.append(plan)
        if not feasible:
            return None
        return max(feasible, key=lambda plan: float(plan.masses_kg[-1]))

    def find_latest_ignition(
        self,
        state: NavigationState,
        vehicle: VehicleModel,
        target_position_m: np.ndarray,
        target_velocity_ms: np.ndarray,
        ground_height_m: float,
    ) -> GuidancePlan | None:
        """Outer-search coast delay; each candidate remains a convex powered-flight problem."""
        for delay in self.config.coast_delays_s:
            predicted = predict_coast_state(state, delay, self.config, ground_height_m)
            # Landing ignition is never scheduled while the vehicle is still
            # rising. Grid fins and the powered descent both begin only after
            # a confirmed descending state.
            if predicted.velocity_ms[0] > self.config.descent_confirm_speed_ms:
                continue
            height = max(predicted.position_m[0] - ground_height_m, 0.0)
            vertical_speed = predicted.velocity_ms[0]
            estimate = max(20.0, min(80.0, 2.0 * height / max(abs(vertical_speed) + 30.0, 30.0)))
            ordered = sorted(self.config.landing_durations_s, key=lambda item: abs(item - estimate))[:3]
            plan = self.solve_best_duration(
                state=predicted,
                vehicle=vehicle,
                target_position_m=target_position_m,
                target_velocity_ms=target_velocity_ms,
                durations_s=ordered,
                phase=FlightPhase.LANDING_BURN,
                ground_height_m=ground_height_m,
                max_tilt_deg=self.config.landing_max_tilt_deg,
            )
            if plan is not None:
                plan.ignition_delay_s = delay
                return plan
        return None

    def _template(self, duration_s: float, node_count: int) -> _Template:
        key = (round(duration_s, 6), node_count)
        if key not in self._templates:
            self._templates[key] = self._build_template(duration_s, node_count)
        return self._templates[key]

    @staticmethod
    def _build_template(duration_s: float, node_count: int) -> _Template:
        dt = duration_s / (node_count - 1)
        r = cp.Variable((3, node_count), name="position")
        v = cp.Variable((3, node_count), name="velocity")
        u = cp.Variable((3, node_count), name="thrust_acceleration")
        sigma = cp.Variable(node_count, nonneg=True, name="thrust_slack")
        dz = cp.Variable(node_count, name="log_mass_delta")

        r_initial = cp.Parameter(3)
        v_initial = cp.Parameter(3)
        r_final = cp.Parameter(3)
        v_final = cp.Parameter(3)
        gravity = cp.Parameter(3)
        aero = cp.Parameter(3)
        rotation_r = cp.Parameter((3, 3))
        rotation_v = cp.Parameter((3, 3))
        z_reference = cp.Parameter(node_count)
        z_initial_delta = cp.Parameter()
        z_upper_delta = cp.Parameter(node_count, nonneg=True)
        log_minimum_mass = cp.Parameter()
        mu_lower = cp.Parameter(node_count, nonneg=True)
        mu_upper = cp.Parameter(node_count, nonneg=True)
        alpha = cp.Parameter(nonneg=True)
        cos_tilt = cp.Parameter(nonneg=True)
        ground_height = cp.Parameter()
        glideslope_inverse_tan = cp.Parameter(nonneg=True)

        constraints = [
            r[:, 0] == r_initial,
            v[:, 0] == v_initial,
            r[:, -1] == r_final,
            v[:, -1] == v_final,
            dz[0] == z_initial_delta,
            dz >= 0.0,
            dz <= z_upper_delta,
            dz + z_reference >= log_minimum_mass,
            r[0, :] >= ground_height,
            u[1:, -1] == 0.0,
        ]
        for index in range(node_count):
            constraints.extend(
                [
                    cp.norm(u[:, index], 2) <= sigma[index],
                    u[0, index] >= cos_tilt * sigma[index],
                    mu_lower[index] * (1.0 - dz[index] + 0.5 * cp.square(dz[index])) <= sigma[index],
                    sigma[index] <= mu_upper[index] * (1.0 - dz[index]),
                    cp.norm(r[1:, index] - r_final[1:], 2)
                    <= glideslope_inverse_tan * (r[0, index] - ground_height),
                ]
            )
        for index in range(node_count - 1):
            acceleration0 = gravity + aero + rotation_r @ r[:, index] + rotation_v @ v[:, index] + u[:, index]
            acceleration1 = gravity + aero + rotation_r @ r[:, index + 1] + rotation_v @ v[:, index + 1] + u[:, index + 1]
            constraints.extend(
                [
                    r[:, index + 1] == r[:, index] + 0.5 * dt * (v[:, index] + v[:, index + 1]),
                    v[:, index + 1] == v[:, index] + 0.5 * dt * (acceleration0 + acceleration1),
                    dz[index + 1] + z_reference[index + 1]
                    == dz[index] + z_reference[index] - 0.5 * alpha * dt * (sigma[index] + sigma[index + 1]),
                ]
            )

        slew = cp.sum(cp.norm(u[:, 1:] - u[:, :-1], 2, axis=0))
        objective = cp.Minimize(-(dz[-1] + z_reference[-1]) + 1e-5 * slew)
        problem = cp.Problem(objective, constraints)
        return _Template(
            duration_s=duration_s,
            node_count=node_count,
            problem=problem,
            r=r,
            v=v,
            u=u,
            sigma=sigma,
            dz=dz,
            r_initial=r_initial,
            v_initial=v_initial,
            r_final=r_final,
            v_final=v_final,
            gravity=gravity,
            aero=aero,
            rotation_r=rotation_r,
            rotation_v=rotation_v,
            z_reference=z_reference,
            z_initial_delta=z_initial_delta,
            z_upper_delta=z_upper_delta,
            log_minimum_mass=log_minimum_mass,
            mu_lower=mu_lower,
            mu_upper=mu_upper,
            alpha=alpha,
            cos_tilt=cos_tilt,
            ground_height=ground_height,
            glideslope_inverse_tan=glideslope_inverse_tan,
        )

    @staticmethod
    def _populate_residuals(
        diagnostics: SolverDiagnostics,
        r: np.ndarray,
        v: np.ndarray,
        u: np.ndarray,
        sigma: np.ndarray,
        duration_s: float,
        gravity: np.ndarray,
        aero: np.ndarray,
        rotation_r: np.ndarray,
        rotation_v: np.ndarray,
        r_target: np.ndarray,
        v_target: np.ndarray,
    ) -> None:
        dt = duration_s / (r.shape[1] - 1)
        residual = 0.0
        for index in range(r.shape[1] - 1):
            a0 = gravity + aero + rotation_r @ r[:, index] + rotation_v @ v[:, index] + u[:, index]
            a1 = gravity + aero + rotation_r @ r[:, index + 1] + rotation_v @ v[:, index + 1] + u[:, index + 1]
            r_res = r[:, index + 1] - r[:, index] - 0.5 * dt * (v[:, index] + v[:, index + 1])
            v_res = v[:, index + 1] - v[:, index] - 0.5 * dt * (a0 + a1)
            residual = max(residual, float(np.max(np.abs(r_res))), float(np.max(np.abs(v_res))))
        diagnostics.max_dynamics_residual = residual
        diagnostics.max_cone_residual = float(max(0.0, np.max(np.linalg.norm(u, axis=0) - sigma)))
        diagnostics.terminal_position_error_m = float(np.linalg.norm(r[:, -1] - r_target))
        diagnostics.terminal_velocity_error_ms = float(np.linalg.norm(v[:, -1] - v_target))


def predict_coast_state(
    state: NavigationState,
    delay_s: float,
    config: MissionConfig,
    ground_height_m: float,
) -> NavigationState:
    if delay_s <= 0.0:
        return state
    r = np.asarray(state.position_m, dtype=float).copy()
    v = np.asarray(state.velocity_ms, dtype=float).copy()
    omega = local_rotation_vector(0.0, config.kerbin_rotation_period_s)
    rotation_r, rotation_v = rotating_frame_matrices(omega)
    steps = max(1, int(math.ceil(delay_s / 0.25)))
    dt = delay_s / steps
    for _ in range(steps):
        altitude = r[0] - ground_height_m
        gravity = np.array([-kerbin_gravity(altitude, config.kerbin_mu_m3s2, config.kerbin_radius_m), 0.0, 0.0])
        acceleration = gravity + state.aerodynamic_accel_ms2 + rotation_r @ r + rotation_v @ v
        r = r + v * dt + 0.5 * acceleration * dt * dt
        v = v + acceleration * dt
    return NavigationState(
        ut=state.ut + delay_s,
        monotonic_s=state.monotonic_s + delay_s,
        position_m=r,
        velocity_ms=v,
        direction=state.direction.copy(),
        vessel_up=state.vessel_up.copy(),
        angular_velocity_rads=state.angular_velocity_rads.copy(),
        mass_kg=state.mass_kg,
        available_thrust_n=state.available_thrust_n,
        specific_impulse_s=state.specific_impulse_s,
        aerodynamic_accel_ms2=state.aerodynamic_accel_ms2.copy(),
        pitch_deg=state.pitch_deg,
        heading_deg=state.heading_deg,
        roll_deg=state.roll_deg,
        situation=state.situation,
        legs_grounded=state.legs_grounded,
        resource_fraction=state.resource_fraction,
    )


def stopping_distance_m(vertical_speed_ms: float, available_thrust_n: float, mass_kg: float, gravity_ms2: float) -> float:
    if vertical_speed_ms >= 0.0:
        return 0.0
    net_accel = available_thrust_n / max(mass_kg, 1.0) - gravity_ms2
    if net_accel <= 0.0:
        return float("inf")
    return vertical_speed_ms * vertical_speed_ms / (2.0 * net_accel)
