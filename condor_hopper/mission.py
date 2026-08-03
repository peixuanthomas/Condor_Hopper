from __future__ import annotations

import math
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime
from typing import Callable

import numpy as np

from .flight_logging import FlightLogger
from .frames import kerbin_gravity, local_rotation_vector, rotating_frame_matrices
from .lcvx import LCvxPlanner, stopping_distance_m
from .math3d import angle_deg, limit_tilt, normalize
from .models import ControlCommand, FlightPhase, GuidancePlan, MissionConfig, NavigationState, VehicleModel
from .vehicle import KrpcVehicle, validate_snapshot


class MissionAbort(RuntimeError):
    pass


class StateFilter:
    def __init__(self, alpha: float = 0.45) -> None:
        self.alpha = alpha
        self._position: np.ndarray | None = None
        self._velocity: np.ndarray | None = None
        self._aero: np.ndarray | None = None

    def update(self, state: NavigationState) -> NavigationState:
        if self._position is None:
            self._position = state.position_m.copy()
            self._velocity = state.velocity_ms.copy()
            self._aero = state.aerodynamic_accel_ms2.copy()
        else:
            self._position = self.alpha * state.position_m + (1.0 - self.alpha) * self._position
            self._velocity = self.alpha * state.velocity_ms + (1.0 - self.alpha) * self._velocity
            self._aero = self.alpha * state.aerodynamic_accel_ms2 + (1.0 - self.alpha) * self._aero
        return NavigationState(
            ut=state.ut,
            monotonic_s=state.monotonic_s,
            position_m=self._position.copy(),
            velocity_ms=self._velocity.copy(),
            direction=state.direction,
            vessel_up=state.vessel_up,
            angular_velocity_rads=state.angular_velocity_rads,
            mass_kg=state.mass_kg,
            available_thrust_n=state.available_thrust_n,
            specific_impulse_s=state.specific_impulse_s,
            aerodynamic_accel_ms2=self._aero.copy(),
            pitch_deg=state.pitch_deg,
            heading_deg=state.heading_deg,
            roll_deg=state.roll_deg,
            situation=state.situation,
            legs_grounded=state.legs_grounded,
            resource_fraction=state.resource_fraction,
        )


class AsyncPlanWorker:
    def __init__(self) -> None:
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="condor-lcvx")
        self.future: Future[GuidancePlan | None] | None = None
        self.label = ""

    def request(self, label: str, function: Callable[[], GuidancePlan | None]) -> bool:
        if self.future is not None and not self.future.done():
            return False
        self.label = label
        self.future = self.executor.submit(function)
        return True

    def poll(self) -> tuple[str, GuidancePlan | None] | None:
        if self.future is None or not self.future.done():
            return None
        label = self.label
        future = self.future
        self.future = None
        try:
            return label, future.result()
        except Exception:
            return label, None

    def close(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=True)


class FlightControlLaw:
    def __init__(
        self,
        config: MissionConfig,
        vehicle: KrpcVehicle,
        pad_height_m: float,
        landing_target_m: np.ndarray | None = None,
    ) -> None:
        self.config = config
        self.vehicle = vehicle
        self.pad_height_m = pad_height_m
        self.landing_target_m = (
            np.array([pad_height_m, 0.0, 0.0])
            if landing_target_m is None
            else np.asarray(landing_target_m, dtype=float)
        )

    def track_plan(
        self,
        state: NavigationState,
        plan: GuidancePlan,
        max_tilt_deg: float,
        rcs: bool,
        brakes: bool,
        legs: bool,
        sas_stability: bool = False,
    ) -> tuple[ControlCommand, dict[str, np.ndarray | float]]:
        elapsed = state.monotonic_s - plan.created_monotonic_s
        target_r, target_v, feedforward_u, _ = plan.sample(elapsed)
        position_error = target_r - state.position_m
        velocity_error = target_v - state.velocity_ms
        feedback = np.asarray(self.config.position_kp) * position_error + np.asarray(self.config.velocity_kd) * velocity_error
        magnitude = float(np.linalg.norm(feedback))
        if magnitude > self.config.max_feedback_accel_ms2:
            feedback *= self.config.max_feedback_accel_ms2 / magnitude
        thrust_accel = feedforward_u + feedback
        return self._command_from_thrust(
            state, thrust_accel, max_tilt_deg, rcs, brakes, legs, sas_stability
        ), {
            "target_r": target_r,
            "target_v": target_v,
            "target_u": thrust_accel,
            "plan_age": elapsed,
        }

    def complete_ascent(
        self,
        state: NavigationState,
        target_position: np.ndarray,
        rcs: bool = True,
        brakes: bool = False,
        legs: bool = False,
        sas_stability: bool = False,
    ) -> tuple[ControlCommand, dict[str, np.ndarray | float]]:
        target_v = np.zeros(3)
        position_error = target_position - state.position_m
        velocity_error = target_v - state.velocity_ms
        desired_inertial = np.asarray(self.config.position_kp) * position_error + np.asarray(self.config.velocity_kd) * velocity_error
        altitude = state.position_m[0] - self.pad_height_m
        gravity = np.array([-kerbin_gravity(altitude, self.config.kerbin_mu_m3s2, self.config.kerbin_radius_m), 0.0, 0.0])
        omega = local_rotation_vector(self.vehicle.latitude_deg, self.config.kerbin_rotation_period_s)
        rotation_r, rotation_v = rotating_frame_matrices(omega)
        thrust_accel = desired_inertial - gravity - state.aerodynamic_accel_ms2 - rotation_r @ state.position_m - rotation_v @ state.velocity_ms
        return self._command_from_thrust(
            state, thrust_accel, self.config.ascent_max_tilt_deg, rcs, brakes, legs, sas_stability
        ), {"target_r": target_position, "target_v": target_v, "target_u": thrust_accel, "plan_age": 0.0}

    def coast(self, state: NavigationState, brakes: bool) -> tuple[ControlCommand, dict[str, np.ndarray | float]]:
        if state.velocity_ms[0] >= 0.0:
            # Maintain the launch attitude during the rising half of the
            # ballistic coast. Pointing retrograde here would request a
            # physically undesirable turn toward the ground.
            target_direction = np.array([1.0, 0.0, 0.0])
        else:
            retrograde = normalize(-state.velocity_ms, np.array([1.0, 0.0, 0.0]))
            target_direction = limit_tilt(retrograde, self.config.landing_max_tilt_deg)
        command = ControlCommand(
            throttle=0.0,
            target_direction=target_direction,
            rcs=True,
            brakes=brakes,
            legs=False,
        )
        return command, {
            "target_r": self.landing_target_m.copy(),
            "target_v": np.zeros(3),
            "target_u": np.zeros(3),
            "plan_age": 0.0,
        }

    def fallback_landing(self, state: NavigationState, terminal: bool) -> tuple[ControlCommand, dict[str, np.ndarray | float]]:
        height = max(state.position_m[0] - self.pad_height_m, 0.0)
        target_down_speed = -min(85.0, max(0.35, math.sqrt(max(0.0, 1.2 * height))))
        if terminal:
            target_down_speed = -min(2.0, max(0.35, 0.12 * math.sqrt(height)))
        target_position = self.landing_target_m.copy()
        target_velocity = np.array([target_down_speed, 0.0, 0.0])
        position_error = target_position - state.position_m
        velocity_error = target_velocity - state.velocity_ms
        desired_inertial = np.array(
            [0.45 * velocity_error[0], 0.04 * position_error[1] + 0.65 * velocity_error[1], 0.04 * position_error[2] + 0.65 * velocity_error[2]]
        )
        gravity = np.array(
            [-kerbin_gravity(height, self.config.kerbin_mu_m3s2, self.config.kerbin_radius_m), 0.0, 0.0]
        )
        thrust_accel = desired_inertial - gravity - state.aerodynamic_accel_ms2
        tilt = self.config.terminal_max_tilt_deg if terminal else self.config.landing_max_tilt_deg
        return self._command_from_thrust(state, thrust_accel, tilt, True, True, terminal), {
            "target_r": target_position,
            "target_v": target_velocity,
            "target_u": thrust_accel,
            "plan_age": float("inf"),
        }

    def _command_from_thrust(
        self,
        state: NavigationState,
        thrust_accel: np.ndarray,
        max_tilt_deg: float,
        rcs: bool,
        brakes: bool,
        legs: bool,
        sas_stability: bool = False,
    ) -> ControlCommand:
        target_direction = limit_tilt(normalize(thrust_accel), max_tilt_deg)
        requested_magnitude = max(0.0, float(np.dot(thrust_accel, target_direction)))
        throttle = requested_magnitude * state.mass_kg / max(state.available_thrust_n, 1.0)
        if throttle > 0.0:
            throttle = max(self.config.min_throttle, min(1.0, throttle))
        return ControlCommand(
            throttle=throttle,
            target_direction=target_direction,
            rcs=rcs,
            brakes=brakes,
            legs=legs,
            sas_stability=sas_stability,
            direction_control=not sas_stability,
        )


class MissionRunner:
    def __init__(
        self,
        vehicle: KrpcVehicle,
        config: MissionConfig,
        logger: FlightLogger,
        planner: LCvxPlanner | None = None,
    ) -> None:
        self.vehicle = vehicle
        self.config = config
        self.logger = logger
        self.planner = LCvxPlanner(config) if planner is None else planner
        self.phase = FlightPhase.PREFLIGHT
        self.filter = StateFilter()
        self.worker = AsyncPlanWorker()
        self.vehicle_model: VehicleModel | None = None
        self.pad_height_m = 0.0
        self.landing_target = np.zeros(3)
        self.ascent_aim = np.zeros(3)
        self.current_plan: GuidancePlan | None = None
        self.descent_seed_plan: GuidancePlan | None = None
        self.control_law: FlightControlLaw | None = None
        self.started_monotonic = 0.0
        self.coast_started: float | None = None
        self.descent_confirmed_at: float | None = None
        self.last_replan_request = 0.0
        self.last_event = ""
        self.failure_reason = ""
        self.has_lifted = False
        self.landed = False
        self.max_horizontal_error_m = 0.0
        self.max_altitude_m = 0.0
        self.cutoff_altitude_m = float("nan")
        self.cutoff_vertical_speed_ms = float("nan")
        self.last_console_update = 0.0

    def run(self) -> dict[str, object]:
        self.config.validate()
        summary: dict[str, object] = {}
        try:
            initial_state = self.preflight()
            assert self.vehicle_model is not None
            assert self.current_plan is not None
            assert self.control_law is not None
            self.started_monotonic = time.monotonic()
            self.current_plan.created_monotonic_s = self.started_monotonic
            self.vehicle.set_sas_stability(True)
            initial_throttle = max(
                self.config.min_throttle,
                min(1.0, self.current_plan.thrust_slack_ms2[0] * initial_state.mass_kg / self.vehicle_model.max_thrust_n),
            )
            self.vehicle.ignite_initial(initial_throttle)
            self.transition(FlightPhase.ASCENT, "engine ignited")
            self._flight_loop()
        except KeyboardInterrupt:
            self.failure_reason = "operator interrupt"
            self.transition(FlightPhase.ABORT, self.failure_reason)
        except Exception as exc:
            self.failure_reason = f"{type(exc).__name__}: {exc}"
            self.transition(FlightPhase.ABORT, self.failure_reason)
        finally:
            self.worker.close()
            self.vehicle.secure_after_flight(self.landed, deploy_legs=self.landed)
            state = None
            try:
                state = self.vehicle.read_state()
            except Exception:
                pass
            summary = self._summary(state)
            self.logger.summary(summary)
        return summary

    def preflight(self) -> NavigationState:
        self.vehicle.prepare_controls()
        snapshot = self.vehicle.static_snapshot()
        model = self.vehicle.vehicle_model()
        self.vehicle_model = model
        self.logger.write_vehicle_snapshot(snapshot)
        failures = validate_snapshot(snapshot, model, self.config)
        if failures:
            raise MissionAbort("; ".join(failures))

        stable_started: float | None = None
        heights: list[float] = []
        latitudes: list[float] = []
        longitudes: list[float] = []
        deadline = time.monotonic() + self.config.preflight_timeout_s
        state = self.vehicle.read_state()
        while time.monotonic() < deadline:
            state = self.vehicle.read_state()
            tilt = angle_deg(np.array([1.0, 0.0, 0.0]), state.direction)
            speed = float(np.linalg.norm(state.velocity_ms))
            stable = (
                ("pre_launch" in state.situation.lower() or "landed" in state.situation.lower())
                and self.vehicle.legs_retracted_count() >= self.config.expected_legs
                and tilt <= self.config.preflight_max_tilt_deg
                and speed <= self.config.preflight_max_speed_ms
            )
            if stable:
                if stable_started is None:
                    stable_started = time.monotonic()
                    heights.clear()
                    latitudes.clear()
                    longitudes.clear()
                heights.append(float(state.position_m[0]))
                flight = self.vehicle.vessel.flight(self.vehicle.body.reference_frame)
                latitudes.append(float(flight.latitude))
                longitudes.append(float(flight.longitude))
                if time.monotonic() - stable_started >= self.config.preflight_settle_s:
                    break
            else:
                stable_started = None
            time.sleep(0.05)
        else:
            raise MissionAbort("preflight stability/ground-contact window was not achieved")

        self.vehicle.rebuild_launch_frame(float(np.mean(latitudes)), float(np.mean(longitudes)))
        state = self.filter.update(self.vehicle.read_state())
        self.pad_height_m = float(state.position_m[0])
        self.landing_target = state.position_m.copy()
        self.landing_target[0] = self.pad_height_m
        self.ascent_aim = self.landing_target.copy()
        self.ascent_aim[0] += self.config.target_altitude_m + self.config.ascent_aim_overshoot_m
        self.control_law = FlightControlLaw(self.config, self.vehicle, self.pad_height_m, self.landing_target)
        target_velocity = np.zeros(3)
        self.logger.event(
            "preflight-stable",
            pad_height_m=self.pad_height_m,
            latitude_deg=self.vehicle.latitude_deg,
            longitude_deg=self.vehicle.longitude_deg,
            twr=model.sea_level_twr,
            resource_fraction=state.resource_fraction,
        )
        self.current_plan = self.planner.solve_best_duration(
            state=state,
            vehicle=model,
            target_position_m=self.ascent_aim,
            target_velocity_ms=target_velocity,
            durations_s=self.config.ascent_durations_s,
            phase=FlightPhase.ASCENT,
            ground_height_m=self.pad_height_m,
            max_tilt_deg=self.config.ascent_max_tilt_deg,
        )
        if self.current_plan is None:
            raise MissionAbort("LCvx could not find a feasible ascent trajectory with required fuel reserve")
        self.logger.event("ascent-plan-ready", diagnostics=asdict(self.current_plan.diagnostics), final_mass_kg=self.current_plan.masses_kg[-1])
        print(
            f"Preflight passed: pad={self.pad_height_m:.2f} m, TWR={model.sea_level_twr:.2f}, "
            f"ascent_plan={self.current_plan.duration_s:.1f} s, predicted_mass={self.current_plan.masses_kg[-1]:.1f} kg",
            flush=True,
        )
        return state

    def _flight_loop(self) -> None:
        period = 1.0 / self.config.control_rate_hz
        deadline = self.started_monotonic + 600.0
        while self.phase not in {FlightPhase.LANDED, FlightPhase.ABORT}:
            tick_started = time.monotonic()
            if tick_started > deadline:
                raise MissionAbort("mission exceeded 600 second runtime limit")
            raw_state = self.vehicle.read_state()
            state = self.filter.update(raw_state)
            horizontal_error = float(np.linalg.norm(state.position_m[1:] - self.landing_target[1:]))
            self.max_horizontal_error_m = max(self.max_horizontal_error_m, horizontal_error)
            self.max_altitude_m = max(self.max_altitude_m, state.position_m[0] - self.pad_height_m)
            command, target = self._update_phase(state)
            attitude_error = angle_deg(state.direction, command.target_direction)
            if not math.isfinite(attitude_error):
                raise MissionAbort("non-finite attitude error")
            if attitude_error > 60.0 and command.throttle > 0.2:
                command.throttle = 0.2
                self.last_event = "throttle-limited-for-attitude"
            self.vehicle.apply_command(command)
            self._write_telemetry(state, command, target, attitude_error)
            if tick_started - self.last_console_update >= 1.0:
                self.last_console_update = tick_started
                print(
                    f"{self.phase.value:13s} t={state.monotonic_s-self.started_monotonic:6.1f}s "
                    f"h={state.position_m[0]-self.pad_height_m:8.1f}m vu={state.velocity_ms[0]:7.1f}m/s "
                    f"xy={horizontal_error:6.1f}m vh={state.horizontal_speed_ms:5.1f}m/s "
                    f"thr={command.throttle:4.2f} tilt={attitude_error:5.1f}deg",
                    flush=True,
                )
            elapsed = time.monotonic() - tick_started
            if elapsed < period:
                time.sleep(period - elapsed)

    def _update_phase(self, state: NavigationState) -> tuple[ControlCommand, dict[str, np.ndarray | float]]:
        assert self.control_law is not None
        assert self.vehicle_model is not None
        worker_result = self.worker.poll()
        if worker_result is not None:
            label, plan = worker_result
            if plan is not None:
                if label == "coast-seed" and self.phase is FlightPhase.COAST:
                    self.descent_seed_plan = plan
                    self.logger.event(
                        "descent-seed-ready",
                        ignition_delay_s=plan.ignition_delay_s,
                        ignition_monotonic_s=plan.created_monotonic_s,
                        diagnostics=asdict(plan.diagnostics),
                    )
                elif label == "landing-replan" and self.phase in {FlightPhase.LANDING_BURN, FlightPhase.TERMINAL}:
                    plan.created_monotonic_s = state.monotonic_s
                    self.current_plan = plan
                    self.logger.event("landing-plan-updated", diagnostics=asdict(plan.diagnostics))
            else:
                self.logger.event("planner-attempt-failed", label=label, phase=self.phase.value)

        height = state.position_m[0] - self.pad_height_m
        if self.phase is FlightPhase.ASCENT:
            assert self.current_plan is not None
            if not self.has_lifted and height >= self.config.liftoff_height_m and state.velocity_ms[0] >= self.config.liftoff_speed_ms:
                self.has_lifted = True
                self.vehicle.set_rcs(True)
                self.logger.event("liftoff-confirmed", height_m=height, vertical_speed_ms=state.velocity_ms[0])
            if height >= self.config.target_altitude_m:
                self.cutoff_altitude_m = height
                self.cutoff_vertical_speed_ms = float(state.velocity_ms[0])
                self.vehicle.shutdown_engines()
                self.coast_started = state.monotonic_s
                self.descent_confirmed_at = None
                self.descent_seed_plan = None
                self.last_replan_request = 0.0
                self.transition(FlightPhase.COAST, "15 km cutoff crossed; engines shut down")
                self.logger.event(
                    "ascent-cutoff",
                    altitude_m=height,
                    vertical_speed_ms=state.velocity_ms[0],
                    horizontal_error_m=float(np.linalg.norm(state.position_m[1:] - self.landing_target[1:])),
                )
                return self.control_law.coast(state, brakes=False)
            plan_age = state.monotonic_s - self.current_plan.created_monotonic_s
            if plan_age <= self.current_plan.duration_s + self.config.last_plan_timeout_s:
                return self.control_law.track_plan(
                    state,
                    self.current_plan,
                    self.config.ascent_max_tilt_deg,
                    self.has_lifted,
                    False,
                    False,
                    sas_stability=True,
                )
            # A stale plan must never leave the vehicle using its terminal
            # feed-forward acceleration. Gravity-compensated completion drives
            # toward the optimizer's aim point, which is above the cutoff plane.
            return self.control_law.complete_ascent(
                state, self.ascent_aim, self.has_lifted, False, False, sas_stability=True
            )

        if self.phase is FlightPhase.COAST:
            if state.velocity_ms[0] <= self.config.descent_confirm_speed_ms:
                if self.descent_confirmed_at is None:
                    self.descent_confirmed_at = state.monotonic_s
            else:
                self.descent_confirmed_at = None
            fins = self.descent_confirmed_at is not None and state.monotonic_s - self.descent_confirmed_at >= self.config.descent_confirm_time_s
            assert self.coast_started is not None
            if self.descent_seed_plan is None and state.monotonic_s - self.last_replan_request >= 1.0 / self.config.planner_rate_hz:
                state_copy = state
                if self.worker.request(
                    "coast-seed",
                    lambda: self.planner.find_latest_ignition(
                        state_copy,
                        self.vehicle_model,
                        self.landing_target.copy(),
                        np.array([self.config.target_vertical_touchdown_ms, 0.0, 0.0]),
                        self.pad_height_m,
                    ),
                ):
                    self.last_replan_request = state.monotonic_s
            gravity = kerbin_gravity(height, self.config.kerbin_mu_m3s2, self.config.kerbin_radius_m)
            stop_distance = stopping_distance_m(state.velocity_ms[0], state.available_thrust_n, state.mass_kg, gravity)
            safety_trigger = height <= stop_distance + max(150.0, abs(state.velocity_ms[0]) * self.config.ignition_margin_s)
            scheduled_trigger = (
                self.descent_seed_plan is not None
                and state.monotonic_s >= self.descent_seed_plan.created_monotonic_s - self.config.ignition_margin_s
            )
            if fins and (scheduled_trigger or safety_trigger):
                self.vehicle.relight()
                if self.descent_seed_plan is not None:
                    self.descent_seed_plan.created_monotonic_s = state.monotonic_s
                    self.current_plan = self.descent_seed_plan
                else:
                    self.current_plan = None
                self.transition(FlightPhase.LANDING_BURN, "landing ignition trigger")
                self.last_replan_request = 0.0
            else:
                return self.control_law.coast(state, fins)

        if self.phase in {FlightPhase.LANDING_BURN, FlightPhase.TERMINAL}:
            tgo = self.current_plan.duration_s - (state.monotonic_s - self.current_plan.created_monotonic_s) if self.current_plan else float("nan")
            deploy_legs = height <= self.config.leg_deploy_altitude_m or (math.isfinite(tgo) and tgo <= self.config.leg_deploy_tgo_s)
            if height <= self.config.terminal_altitude_m and self.phase is not FlightPhase.TERMINAL:
                self.transition(FlightPhase.TERMINAL, "entered final 100 m")
            if state.monotonic_s - self.last_replan_request >= 1.0 / self.config.planner_rate_hz:
                current = state
                if self.current_plan is None:
                    duration = 30.0
                else:
                    remaining = max(20.0, self.current_plan.duration_s - (state.monotonic_s - self.current_plan.created_monotonic_s))
                    duration = min(self.config.landing_durations_s, key=lambda item: abs(item - remaining))
                phase = self.phase
                max_tilt = self.config.terminal_max_tilt_deg if phase is FlightPhase.TERMINAL else self.config.landing_max_tilt_deg
                requested = self.worker.request(
                    "landing-replan",
                    lambda: self.planner.solve_fixed_duration(
                        current,
                        self.vehicle_model,
                        self.landing_target.copy(),
                        np.array([self.config.target_vertical_touchdown_ms, 0.0, 0.0]),
                        duration,
                        phase,
                        self.pad_height_m,
                        max_tilt,
                    ),
                )
                if requested:
                    self.last_replan_request = state.monotonic_s

            terminal = self.phase is FlightPhase.TERMINAL
            plan_is_fresh = self.current_plan is not None and state.monotonic_s - self.current_plan.created_monotonic_s <= self.current_plan.duration_s + self.config.last_plan_timeout_s
            if plan_is_fresh:
                command, target = self.control_law.track_plan(
                    state,
                    self.current_plan,
                    self.config.terminal_max_tilt_deg if terminal else self.config.landing_max_tilt_deg,
                    True,
                    True,
                    deploy_legs,
                )
            else:
                command, target = self.control_law.fallback_landing(state, terminal)
                command.legs = deploy_legs

            grounded = state.legs_grounded >= 3
            if grounded:
                if (
                    abs(float(state.velocity_ms[0])) <= self.config.touchdown_vertical_speed_tolerance_ms
                    and state.horizontal_speed_ms <= self.config.touchdown_horizontal_speed_tolerance_ms
                ):
                    self.landed = True
                    self.transition(FlightPhase.LANDED, "touchdown confirmed")
                    safe = ControlCommand(
                        throttle=0.0,
                        target_direction=np.array([1.0, 0.0, 0.0]),
                        rcs=False,
                        brakes=False,
                        legs=True,
                        sas_stability=False,
                        direction_control=False,
                    )
                    return safe, {
                        "target_r": self.landing_target.copy(),
                        "target_v": np.zeros(3),
                        "target_u": np.zeros(3),
                        "plan_age": 0.0,
                    }
                elif abs(float(state.velocity_ms[0])) > 5.0:
                    raise MissionAbort(f"hard contact at {state.velocity_ms[0]:.2f} m/s")
            return command, target

        return self.control_law.fallback_landing(state, terminal=False)

    def transition(self, phase: FlightPhase, reason: str) -> None:
        if self.phase is phase and phase is not FlightPhase.ABORT:
            return
        old = self.phase
        self.phase = phase
        self.last_event = f"{old.value}->{phase.value}: {reason}"
        self.logger.event("phase-transition", old=old.value, new=phase.value, reason=reason)
        print(f"PHASE {old.value} -> {phase.value}: {reason}", flush=True)

    def _write_telemetry(
        self,
        state: NavigationState,
        command: ControlCommand,
        target: dict[str, np.ndarray | float],
        attitude_error: float,
    ) -> None:
        target_r = np.asarray(target["target_r"], dtype=float)
        target_v = np.asarray(target["target_v"], dtype=float)
        target_u = np.asarray(target["target_u"], dtype=float)
        plan = self.current_plan
        diagnostics = None if plan is None else plan.diagnostics
        tgo = "" if plan is None else max(0.0, plan.duration_s - (state.monotonic_s - plan.created_monotonic_s))
        self.logger.telemetry(
            {
                "timestamp": datetime.now().astimezone().isoformat(timespec="milliseconds"),
                "ut": state.ut,
                "mission_time_s": state.monotonic_s - self.started_monotonic,
                "phase": self.phase.value,
                "event": self.last_event,
                "failure_reason": self.failure_reason,
                "position_up_m": state.position_m[0],
                "position_north_m": state.position_m[1],
                "position_east_m": state.position_m[2],
                "altitude_above_pad_m": state.position_m[0] - self.pad_height_m,
                "velocity_up_ms": state.velocity_ms[0],
                "velocity_north_ms": state.velocity_ms[1],
                "velocity_east_ms": state.velocity_ms[2],
                "horizontal_error_m": np.linalg.norm(state.position_m[1:] - self.landing_target[1:]),
                "horizontal_speed_ms": state.horizontal_speed_ms,
                "target_up_m": target_r[0],
                "target_north_m": target_r[1],
                "target_east_m": target_r[2],
                "target_velocity_up_ms": target_v[0],
                "target_velocity_north_ms": target_v[1],
                "target_velocity_east_ms": target_v[2],
                "position_error_up_m": target_r[0] - state.position_m[0],
                "position_error_north_m": target_r[1] - state.position_m[1],
                "position_error_east_m": target_r[2] - state.position_m[2],
                "target_accel_up_ms2": target_u[0],
                "target_accel_north_ms2": target_u[1],
                "target_accel_east_ms2": target_u[2],
                "target_accel_norm_ms2": np.linalg.norm(target_u),
                "target_direction_up": command.target_direction[0],
                "target_direction_north": command.target_direction[1],
                "target_direction_east": command.target_direction[2],
                "attitude_error_deg": attitude_error,
                "pitch_deg": state.pitch_deg,
                "heading_deg": state.heading_deg,
                "roll_deg": state.roll_deg,
                "angular_rate_rads": np.linalg.norm(state.angular_velocity_rads),
                "control_pitch": command.attitude.pitch,
                "control_yaw": command.attitude.yaw,
                "control_roll": command.attitude.roll,
                "throttle": command.throttle,
                "current_thrust_n": self.vehicle.current_thrust(),
                "available_thrust_n": state.available_thrust_n,
                "mass_kg": state.mass_kg,
                "resource_fraction": state.resource_fraction,
                "rcs": command.rcs,
                "brakes": command.brakes,
                "gear": command.legs,
                "legs": command.legs,
                "sas_stability": command.sas_stability,
                "direction_control": command.direction_control,
                "legs_grounded": state.legs_grounded,
                "aero_accel_up_ms2": state.aerodynamic_accel_ms2[0],
                "aero_accel_north_ms2": state.aerodynamic_accel_ms2[1],
                "aero_accel_east_ms2": state.aerodynamic_accel_ms2[2],
                "solver_status": "" if diagnostics is None else diagnostics.status,
                "solver_name": "" if diagnostics is None else diagnostics.solver,
                "solver_duration_s": "" if diagnostics is None else diagnostics.duration_s,
                "solver_wall_ms": "" if diagnostics is None else 1000.0 * diagnostics.wall_time_s,
                "solver_core_ms": "" if diagnostics is None else 1000.0 * diagnostics.solve_time_s,
                "solver_iterations": "" if diagnostics is None else diagnostics.iterations,
                "solver_objective": "" if diagnostics is None else diagnostics.objective,
                "solver_dynamics_residual": "" if diagnostics is None else diagnostics.max_dynamics_residual,
                "solver_cone_residual": "" if diagnostics is None else diagnostics.max_cone_residual,
                "solver_terminal_position_error_m": "" if diagnostics is None else diagnostics.terminal_position_error_m,
                "solver_terminal_velocity_error_ms": "" if diagnostics is None else diagnostics.terminal_velocity_error_ms,
                "predicted_tgo_s": tgo,
                "ignition_delay_s": "" if plan is None else plan.ignition_delay_s,
                "plan_age_s": target["plan_age"],
            }
        )
        self.last_event = ""

    def _summary(self, state: NavigationState | None) -> dict[str, object]:
        if state is None:
            return {"success": False, "phase": self.phase.value, "failure_reason": self.failure_reason}
        tilt = angle_deg(np.array([1.0, 0.0, 0.0]), state.direction)
        horizontal = float(np.linalg.norm(state.position_m[1:] - self.landing_target[1:]))
        success = (
            self.landed
            and horizontal <= self.config.touchdown_horizontal_tolerance_m
            and abs(float(state.velocity_ms[0])) <= self.config.touchdown_vertical_speed_tolerance_ms
            and state.horizontal_speed_ms <= self.config.touchdown_horizontal_speed_tolerance_ms
            and tilt <= self.config.touchdown_tilt_tolerance_deg
        )
        return {
            "success": success,
            "phase": self.phase.value,
            "failure_reason": self.failure_reason,
            "landing_horizontal_error_m": horizontal,
            "landing_vertical_speed_ms": float(state.velocity_ms[0]),
            "landing_horizontal_speed_ms": state.horizontal_speed_ms,
            "landing_tilt_deg": tilt,
            "final_mass_kg": state.mass_kg,
            "final_resource_fraction": state.resource_fraction,
            "max_horizontal_error_m": self.max_horizontal_error_m,
            "cutoff_altitude_m": self.cutoff_altitude_m,
            "cutoff_vertical_speed_ms": self.cutoff_vertical_speed_ms,
            "apogee_altitude_m": self.max_altitude_m,
            "telemetry_csv": str(self.logger.telemetry_path),
            "events_jsonl": str(self.logger.events_path),
        }
