from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np


G0 = 9.80665


class FlightPhase(str, Enum):
    PREFLIGHT = "preflight"
    ASCENT = "ascent"
    COAST = "coast"
    LANDING_BURN = "landing-burn"
    TERMINAL = "terminal"
    LANDED = "landed"
    ABORT = "abort"


@dataclass(slots=True)
class MissionConfig:
    target_altitude_m: float = 15_000.0
    # The powered-ascent optimizer aims above the cutoff plane.  The flight
    # state machine shuts the engine down on the actual 15 km crossing, so the
    # vehicle reaches the cutoff with positive vertical velocity and coasts
    # cleanly through apogee instead of attempting an unstable hover.
    ascent_aim_overshoot_m: float = 100.0
    control_rate_hz: float = 20.0
    planner_rate_hz: float = 2.0
    node_count: int = 81
    ascent_durations_s: tuple[float, ...] = (45.0, 55.0, 65.0, 75.0, 85.0, 95.0)
    landing_durations_s: tuple[float, ...] = (20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0)
    coast_delays_s: tuple[float, ...] = (30.0, 25.0, 20.0, 15.0, 10.0, 7.5, 5.0, 3.0, 2.0, 1.0, 0.0)
    min_throttle: float = 0.05
    max_plan_throttle: float = 0.95
    fuel_reserve_fraction: float = 0.15
    ascent_max_tilt_deg: float = 8.0
    landing_max_tilt_deg: float = 15.0
    terminal_max_tilt_deg: float = 3.0
    terminal_altitude_m: float = 100.0
    leg_deploy_altitude_m: float = 300.0
    leg_deploy_tgo_s: float = 12.0
    cutoff_altitude_tolerance_m: float = 10.0
    touchdown_horizontal_tolerance_m: float = 3.0
    touchdown_vertical_speed_tolerance_ms: float = 1.0
    touchdown_horizontal_speed_tolerance_ms: float = 0.5
    touchdown_tilt_tolerance_deg: float = 3.0
    ignition_margin_s: float = 2.0
    last_plan_timeout_s: float = 1.0
    preflight_settle_s: float = 2.0
    preflight_timeout_s: float = 15.0
    preflight_max_tilt_deg: float = 1.0
    preflight_max_speed_ms: float = 0.25
    liftoff_height_m: float = 1.0
    liftoff_speed_ms: float = 1.0
    descent_confirm_speed_ms: float = -5.0
    descent_confirm_time_s: float = 0.5
    target_vertical_touchdown_ms: float = -0.35
    max_ascent_speed_ms: float = 475.0
    max_descent_speed_ms: float = 450.0
    glideslope_deg: float = 70.0
    max_attitude_error_deg: float = 25.0
    telemetry_stale_s: float = 0.75
    position_kp: tuple[float, float, float] = (0.018, 0.025, 0.025)
    velocity_kd: tuple[float, float, float] = (0.38, 0.48, 0.48)
    max_feedback_accel_ms2: float = 5.0
    attitude_settling_s: float = 0.45
    kerbin_radius_m: float = 600_000.0
    kerbin_mu_m3s2: float = 3.5316e12
    kerbin_rotation_period_s: float = 21_549.425
    log_root: Path = Path("flight_logs")
    vessel_name: str = "Condor-Hopper"
    expected_engines: int = 1
    expected_rcs: int = 4
    expected_grid_fins: int = 4
    expected_legs: int = 4
    address: str = "127.0.0.1"
    rpc_port: int = 50_000
    stream_port: int = 50_001
    client_name: str = "Condor Hopper LCvx"

    def validate(self) -> None:
        if self.target_altitude_m <= 0.0:
            raise ValueError("target altitude must be positive")
        if self.ascent_aim_overshoot_m <= 0.0:
            raise ValueError("ascent aim overshoot must be positive")
        if self.control_rate_hz <= 0.0 or self.planner_rate_hz <= 0.0:
            raise ValueError("control and planner rates must be positive")
        if self.node_count < 11:
            raise ValueError("LCvx node count must be at least 11")
        if not 0.0 <= self.min_throttle < self.max_plan_throttle <= 1.0:
            raise ValueError("invalid throttle bounds")
        if not 0.0 <= self.fuel_reserve_fraction < 1.0:
            raise ValueError("fuel reserve fraction must be in [0, 1)")

    def json_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["log_root"] = str(self.log_root)
        return result


@dataclass(slots=True)
class VehicleModel:
    name: str
    wet_mass_kg: float
    dry_mass_kg: float
    current_mass_kg: float
    max_thrust_n: float
    specific_impulse_s: float
    engine_count: int
    rcs_count: int
    grid_fin_count: int
    leg_count: int
    engine_restartable: bool = True
    engine_gimbal_deg: float = 0.0

    @property
    def propellant_mass_kg(self) -> float:
        return max(0.0, self.wet_mass_kg - self.dry_mass_kg)

    @property
    def sea_level_twr(self) -> float:
        return self.max_thrust_n / max(self.current_mass_kg * G0, 1.0)

    def minimum_landing_mass(self, reserve_fraction: float) -> float:
        return self.dry_mass_kg + reserve_fraction * self.propellant_mass_kg


@dataclass(slots=True)
class NavigationState:
    ut: float
    monotonic_s: float
    position_m: np.ndarray
    velocity_ms: np.ndarray
    direction: np.ndarray
    vessel_up: np.ndarray
    angular_velocity_rads: np.ndarray
    mass_kg: float
    available_thrust_n: float
    specific_impulse_s: float
    aerodynamic_accel_ms2: np.ndarray
    pitch_deg: float
    heading_deg: float
    roll_deg: float
    situation: str
    legs_grounded: int
    resource_fraction: float

    @property
    def horizontal_error_m(self) -> float:
        return float(np.linalg.norm(self.position_m[1:]))

    @property
    def horizontal_speed_ms(self) -> float:
        return float(np.linalg.norm(self.velocity_ms[1:]))


@dataclass(slots=True)
class SolverDiagnostics:
    status: str = "not-run"
    solver: str = ""
    solve_time_s: float = 0.0
    wall_time_s: float = 0.0
    iterations: int = 0
    objective: float = float("nan")
    duration_s: float = 0.0
    max_dynamics_residual: float = float("nan")
    max_cone_residual: float = float("nan")
    terminal_position_error_m: float = float("nan")
    terminal_velocity_error_ms: float = float("nan")
    message: str = ""

    @property
    def accepted(self) -> bool:
        return (
            self.status in {"optimal", "optimal_inaccurate"}
            and self.max_dynamics_residual <= 0.25
            and self.max_cone_residual <= 1e-3
            and self.terminal_position_error_m <= 0.5
            and self.terminal_velocity_error_ms <= 0.25
        )


@dataclass(slots=True)
class GuidancePlan:
    phase: FlightPhase
    created_monotonic_s: float
    duration_s: float
    times_s: np.ndarray
    positions_m: np.ndarray
    velocities_ms: np.ndarray
    thrust_accel_ms2: np.ndarray
    thrust_slack_ms2: np.ndarray
    masses_kg: np.ndarray
    diagnostics: SolverDiagnostics
    ignition_delay_s: float = 0.0

    def sample(self, elapsed_s: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        t = float(np.clip(elapsed_s, 0.0, self.duration_s))
        r = np.array([np.interp(t, self.times_s, self.positions_m[i]) for i in range(3)])
        v = np.array([np.interp(t, self.times_s, self.velocities_ms[i]) for i in range(3)])
        u = np.array([np.interp(t, self.times_s, self.thrust_accel_ms2[i]) for i in range(3)])
        sigma = float(np.interp(t, self.times_s, self.thrust_slack_ms2))
        return r, v, u, sigma


@dataclass(slots=True)
class AttitudeCommand:
    roll: float = 0.0
    yaw: float = 0.0
    pitch: float = 0.0


@dataclass(slots=True)
class ControlCommand:
    throttle: float = 0.0
    target_direction: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0]))
    attitude: AttitudeCommand = field(default_factory=AttitudeCommand)
    rcs: bool = False
    brakes: bool = False
    legs: bool = False
    sas_stability: bool = False
    direction_control: bool = True
