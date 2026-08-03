from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .frames import kerbin_gravity, local_rotation_vector, rotating_frame_matrices
from .lcvx import LCvxPlanner, predict_coast_state
from .models import FlightPhase, G0, GuidancePlan, MissionConfig, NavigationState, VehicleModel


@dataclass(slots=True)
class SimulationResult:
    success: bool
    message: str
    cutoff_altitude_error_m: float
    cutoff_vertical_speed_ms: float
    apogee_altitude_m: float
    landing_horizontal_error_m: float
    landing_vertical_speed_ms: float
    landing_horizontal_speed_ms: float
    final_mass_kg: float
    ascent_plan_duration_s: float
    coast_duration_s: float
    landing_plan_duration_s: float


def default_condor_model() -> VehicleModel:
    return VehicleModel(
        name="Condor-Hopper",
        wet_mass_kg=57_494.63671875,
        dry_mass_kg=19_833.66015625,
        current_mass_kg=57_494.63671875,
        max_thrust_n=1_770_593.75,
        specific_impulse_s=330.22165,
        engine_count=1,
        rcs_count=4,
        grid_fin_count=4,
        leg_count=4,
        engine_restartable=True,
        engine_gimbal_deg=3.0,
    )


def make_state(position: np.ndarray, velocity: np.ndarray, mass: float, vehicle: VehicleModel, t: float) -> NavigationState:
    return NavigationState(
        ut=t,
        monotonic_s=t,
        position_m=np.asarray(position, dtype=float),
        velocity_ms=np.asarray(velocity, dtype=float),
        direction=np.array([1.0, 0.0, 0.0]),
        vessel_up=np.array([0.0, 1.0, 0.0]),
        angular_velocity_rads=np.zeros(3),
        mass_kg=mass,
        available_thrust_n=vehicle.max_thrust_n,
        specific_impulse_s=vehicle.specific_impulse_s,
        aerodynamic_accel_ms2=np.zeros(3),
        pitch_deg=90.0,
        heading_deg=90.0,
        roll_deg=0.0,
        situation="simulation",
        legs_grounded=0,
        resource_fraction=max(0.0, (mass - vehicle.dry_mass_kg) / max(vehicle.propellant_mass_kg, 1.0)),
    )


def simulate_mission(
    config: MissionConfig | None = None,
    vehicle: VehicleModel | None = None,
    horizontal_offset_m: float = 0.0,
) -> SimulationResult:
    config = MissionConfig() if config is None else config
    vehicle = default_condor_model() if vehicle is None else vehicle
    config.validate()
    pad_height = 13.44
    initial = make_state(np.array([pad_height, horizontal_offset_m, 0.0]), np.zeros(3), vehicle.current_mass_kg, vehicle, 0.0)
    planner = LCvxPlanner(config)
    ascent_aim = np.array(
        [pad_height + config.target_altitude_m + config.ascent_aim_overshoot_m, 0.0, 0.0]
    )
    ascent = planner.solve_best_duration(
        initial,
        vehicle,
        ascent_aim,
        np.zeros(3),
        config.ascent_durations_s,
        FlightPhase.ASCENT,
        pad_height,
        config.ascent_max_tilt_deg,
    )
    if ascent is None:
        return _failed("ascent LCvx problem is infeasible", vehicle.current_mass_kg)

    state = _track_until_cutoff(initial, ascent, vehicle, config, pad_height, dt=0.05)
    cutoff_height = float(state.position_m[0] - pad_height)
    cutoff_error = abs(cutoff_height - config.target_altitude_m)
    if cutoff_height < config.target_altitude_m:
        return _failed("15 km ascent cutoff plane was not crossed", state.mass_kg, ascent.duration_s)

    landing = planner.find_latest_ignition(
        state,
        vehicle,
        np.array([pad_height, 0.0, 0.0]),
        np.array([config.target_vertical_touchdown_ms, 0.0, 0.0]),
        pad_height,
    )
    if landing is None:
        return _failed("landing LCvx ignition search is infeasible after cutoff", state.mass_kg, ascent.duration_s)
    coasted, apogee = _coast_to_ignition(
        state, landing.ignition_delay_s, config, pad_height, dt=0.05
    )
    if coasted.velocity_ms[0] > config.descent_confirm_speed_ms:
        return _failed(
            "landing ignition was scheduled before confirmed descent",
            coasted.mass_kg,
            ascent.duration_s,
            landing.ignition_delay_s,
        )
    landing.created_monotonic_s = coasted.monotonic_s
    landed = _track_plan(coasted, landing, vehicle, config, pad_height, dt=0.05)
    horizontal_error = float(np.linalg.norm(landed.position_m[1:]))
    horizontal_speed = float(np.linalg.norm(landed.velocity_ms[1:]))
    success = (
        cutoff_error <= config.cutoff_altitude_tolerance_m
        and state.velocity_ms[0] > 0.0
        and horizontal_error <= config.touchdown_horizontal_tolerance_m
        and abs(float(landed.velocity_ms[0])) <= config.touchdown_vertical_speed_tolerance_ms
        and horizontal_speed <= config.touchdown_horizontal_speed_tolerance_ms
        and landed.mass_kg >= vehicle.minimum_landing_mass(config.fuel_reserve_fraction)
    )
    return SimulationResult(
        success=success,
        message="simulation met cutoff/coast/landing gates" if success else "simulation ended outside acceptance gates",
        cutoff_altitude_error_m=cutoff_error,
        cutoff_vertical_speed_ms=float(state.velocity_ms[0]),
        apogee_altitude_m=apogee,
        landing_horizontal_error_m=horizontal_error,
        landing_vertical_speed_ms=float(landed.velocity_ms[0]),
        landing_horizontal_speed_ms=horizontal_speed,
        final_mass_kg=landed.mass_kg,
        ascent_plan_duration_s=ascent.duration_s,
        coast_duration_s=landing.ignition_delay_s,
        landing_plan_duration_s=landing.duration_s,
    )


def _track_plan(
    state: NavigationState,
    plan: GuidancePlan,
    vehicle: VehicleModel,
    config: MissionConfig,
    pad_height: float,
    dt: float,
) -> NavigationState:
    t = 0.0
    while t < plan.duration_s - 1e-9:
        step = min(dt, plan.duration_s - t)
        target_r, target_v, target_u, _ = plan.sample(t)
        feedback = np.asarray(config.position_kp) * (target_r - state.position_m) + np.asarray(config.velocity_kd) * (target_v - state.velocity_ms)
        magnitude = float(np.linalg.norm(feedback))
        if magnitude > config.max_feedback_accel_ms2:
            feedback *= config.max_feedback_accel_ms2 / magnitude
        thrust_accel = target_u + feedback
        state = _integrate(state, thrust_accel, step, vehicle, config, pad_height)
        t += step
    return state


def _track_until_cutoff(
    state: NavigationState,
    plan: GuidancePlan,
    vehicle: VehicleModel,
    config: MissionConfig,
    pad_height: float,
    dt: float,
) -> NavigationState:
    t = 0.0
    while t < plan.duration_s - 1e-9:
        step = min(dt, plan.duration_s - t)
        target_r, target_v, target_u, _ = plan.sample(t)
        feedback = np.asarray(config.position_kp) * (target_r - state.position_m) + np.asarray(config.velocity_kd) * (target_v - state.velocity_ms)
        magnitude = float(np.linalg.norm(feedback))
        if magnitude > config.max_feedback_accel_ms2:
            feedback *= config.max_feedback_accel_ms2 / magnitude
        state = _integrate(state, target_u + feedback, step, vehicle, config, pad_height)
        t += step
        if state.position_m[0] - pad_height >= config.target_altitude_m:
            return state
    return state


def _coast_to_ignition(
    state: NavigationState,
    duration_s: float,
    config: MissionConfig,
    pad_height: float,
    dt: float,
) -> tuple[NavigationState, float]:
    elapsed = 0.0
    apogee = float(state.position_m[0] - pad_height)
    while elapsed < duration_s - 1e-9:
        step = min(dt, duration_s - elapsed)
        state = predict_coast_state(state, step, config, pad_height)
        elapsed += step
        apogee = max(apogee, float(state.position_m[0] - pad_height))
    return state, apogee


def _integrate(
    state: NavigationState,
    thrust_accel: np.ndarray,
    dt: float,
    vehicle: VehicleModel,
    config: MissionConfig,
    pad_height: float,
) -> NavigationState:
    r = state.position_m.copy()
    v = state.velocity_ms.copy()
    altitude = r[0] - pad_height
    gravity = np.array([-kerbin_gravity(altitude, config.kerbin_mu_m3s2, config.kerbin_radius_m), 0.0, 0.0])
    rotation_r, rotation_v = rotating_frame_matrices(local_rotation_vector(0.0, config.kerbin_rotation_period_s))
    acceleration = gravity + rotation_r @ r + rotation_v @ v + thrust_accel
    r = r + v * dt + 0.5 * acceleration * dt * dt
    v = v + acceleration * dt
    mass_flow = float(np.linalg.norm(thrust_accel)) * state.mass_kg / max(vehicle.specific_impulse_s * G0, 1.0)
    mass = max(vehicle.dry_mass_kg, state.mass_kg - mass_flow * dt)
    return make_state(r, v, mass, vehicle, state.monotonic_s + dt)


def _failed(
    message: str,
    mass: float,
    ascent_duration: float = 0.0,
    coast_duration: float = 0.0,
) -> SimulationResult:
    return SimulationResult(
        success=False,
        message=message,
        cutoff_altitude_error_m=float("inf"),
        cutoff_vertical_speed_ms=float("nan"),
        apogee_altitude_m=float("nan"),
        landing_horizontal_error_m=float("inf"),
        landing_vertical_speed_ms=float("inf"),
        landing_horizontal_speed_ms=float("inf"),
        final_mass_kg=mass,
        ascent_plan_duration_s=ascent_duration,
        coast_duration_s=coast_duration,
        landing_plan_duration_s=0.0,
    )
