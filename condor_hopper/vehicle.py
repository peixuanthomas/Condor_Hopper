from __future__ import annotations

import math
import time
from typing import Any

import numpy as np

from .frames import create_launch_reference_frame, safe_read
from .math3d import finite_vector
from .models import ControlCommand, MissionConfig, NavigationState, VehicleModel


def safe_float(reader, fallback: float = float("nan")) -> float:
    try:
        value = float(reader())
    except Exception:
        return fallback
    return value if math.isfinite(value) else fallback


def connect_krpc(config: MissionConfig):
    import krpc

    return krpc.connect(
        name=config.client_name,
        address=config.address,
        rpc_port=config.rpc_port,
        stream_port=config.stream_port,
    )


class KrpcVehicle:
    def __init__(self, connection: Any, config: MissionConfig) -> None:
        self.connection = connection
        self.config = config
        self.space_center = connection.space_center
        self.vessel = self.space_center.active_vessel
        self.body = self.vessel.orbit.body
        self.control = self.vessel.control
        self.engines = list(self.vessel.parts.engines)
        self.latitude_deg = safe_float(lambda: self.vessel.flight(self.body.reference_frame).latitude, 0.0)
        self.longitude_deg = safe_float(lambda: self.vessel.flight(self.body.reference_frame).longitude, 0.0)
        self.reference_frame = create_launch_reference_frame(
            self.space_center, self.body, self.latitude_deg, self.longitude_deg
        )
        self.flight = self.vessel.flight(self.reference_frame)
        self._last_resource_fraction = 1.0
        self._direction_control_engaged = False
        self._sas_stability_enabled = False

    def rebuild_launch_frame(self, latitude_deg: float, longitude_deg: float) -> None:
        self.latitude_deg = latitude_deg
        self.longitude_deg = longitude_deg
        self.reference_frame = create_launch_reference_frame(
            self.space_center, self.body, latitude_deg, longitude_deg
        )
        self.flight = self.vessel.flight(self.reference_frame)
        self._direction_control_engaged = False
        self._sas_stability_enabled = False

    def static_snapshot(self) -> dict[str, Any]:
        grid_fins = [surface for surface in self.vessel.parts.control_surfaces if "grid fin" in surface.part.name.lower()]
        legs = list(self.vessel.parts.legs)
        rcs = list(self.vessel.parts.rcs)
        resources = []
        for resource in safe_read(lambda: list(self.vessel.resources.all), []):
            resources.append(
                {
                    "name": safe_read(lambda r=resource: r.name, ""),
                    "amount": safe_float(lambda r=resource: r.amount, 0.0),
                    "max": safe_float(lambda r=resource: r.max, 0.0),
                    "enabled": safe_read(lambda r=resource: r.enabled, None),
                }
            )
        engines = []
        for engine in self.engines:
            engines.append(
                {
                    "part": safe_read(lambda e=engine: e.part.name, ""),
                    "title": safe_read(lambda e=engine: e.part.title, ""),
                    "stage": safe_read(lambda e=engine: e.part.stage, -1),
                    "active": safe_read(lambda e=engine: e.active, False),
                    "can_restart": safe_read(lambda e=engine: e.can_restart, False),
                    "available_thrust_n": safe_float(lambda e=engine: e.available_thrust, 0.0),
                    "max_thrust_n": safe_float(lambda e=engine: e.max_thrust, 0.0),
                    "specific_impulse_s": safe_float(lambda e=engine: e.specific_impulse, 0.0),
                    "gimbal_range_deg": safe_float(lambda e=engine: e.gimbal_range, 0.0),
                    "propellants": list(safe_read(lambda e=engine: e.propellants, [])),
                }
            )
        return {
            "server_version": safe_read(lambda: self.connection.krpc.get_status().version, "unknown"),
            "vessel": {
                "name": safe_read(lambda: self.vessel.name, ""),
                "situation": str(safe_read(lambda: self.vessel.situation, "")),
                "mass_kg": safe_float(lambda: self.vessel.mass, 0.0),
                "dry_mass_kg": safe_float(lambda: self.vessel.dry_mass, 0.0),
                "current_stage": safe_read(lambda: self.control.current_stage, -1),
                "position_local_m": safe_read(lambda: self.vessel.position(self.reference_frame), ()),
                "direction_local": safe_read(lambda: self.vessel.direction(self.reference_frame), ()),
            },
            "control": {
                "throttle": safe_float(lambda: self.control.throttle, 0.0),
                "rcs": safe_read(lambda: self.control.rcs, False),
                "brakes": safe_read(lambda: self.control.brakes, False),
                "gear": safe_read(lambda: self.control.gear, False),
                "legs": safe_read(lambda: self.control.legs, False),
            },
            "engines": engines,
            "counts": {
                "engines": len(self.engines),
                "rcs": len(rcs),
                "grid_fins": len(grid_fins),
                "legs": len(legs),
            },
            "rcs": [
                {
                    "part": safe_read(lambda unit=unit: unit.part.name, ""),
                    "enabled": safe_read(lambda unit=unit: unit.enabled, False),
                    "max_thrust_n": safe_float(lambda unit=unit: unit.max_thrust, 0.0),
                    "propellants": list(safe_read(lambda unit=unit: unit.propellants, [])),
                }
                for unit in rcs
            ],
            "grid_fins": [
                {
                    "part": safe_read(lambda surface=surface: surface.part.name, ""),
                    "authority": safe_float(lambda surface=surface: surface.authority_limiter, 0.0),
                    "pitch": safe_read(lambda surface=surface: surface.pitch_enabled, False),
                    "yaw": safe_read(lambda surface=surface: surface.yaw_enabled, False),
                    "roll": safe_read(lambda surface=surface: surface.roll_enabled, False),
                }
                for surface in grid_fins
            ],
            "legs": [
                {
                    "part": safe_read(lambda leg=leg: leg.part.name, ""),
                    "deployed": safe_read(lambda leg=leg: leg.deployed, False),
                    "grounded": safe_read(lambda leg=leg: leg.is_grounded, False),
                    "state": str(safe_read(lambda leg=leg: leg.state, "")),
                }
                for leg in legs
            ],
            "resources": resources,
        }

    def vehicle_model(self) -> VehicleModel:
        max_thrust = sum(safe_float(lambda e=engine: e.max_thrust, 0.0) for engine in self.engines)
        available = sum(safe_float(lambda e=engine: e.available_thrust, 0.0) for engine in self.engines)
        max_thrust = max(max_thrust, available)
        isp_values = [safe_float(lambda e=engine: e.specific_impulse, 0.0) for engine in self.engines]
        isp = max(isp_values or [0.0])
        grid_count = sum(1 for surface in self.vessel.parts.control_surfaces if "grid fin" in surface.part.name.lower())
        return VehicleModel(
            name=str(safe_read(lambda: self.vessel.name, "")),
            wet_mass_kg=safe_float(lambda: self.vessel.mass, 0.0),
            dry_mass_kg=safe_float(lambda: self.vessel.dry_mass, 0.0),
            current_mass_kg=safe_float(lambda: self.vessel.mass, 0.0),
            max_thrust_n=max_thrust,
            specific_impulse_s=isp,
            engine_count=len(self.engines),
            rcs_count=len(self.vessel.parts.rcs),
            grid_fin_count=grid_count,
            leg_count=len(self.vessel.parts.legs),
            engine_restartable=all(safe_read(lambda e=engine: e.can_restart, False) for engine in self.engines),
            engine_gimbal_deg=max([safe_float(lambda e=engine: e.gimbal_range, 0.0) for engine in self.engines] or [0.0]),
        )

    def read_state(self) -> NavigationState:
        mass = safe_float(lambda: self.vessel.mass, 1.0)
        aerodynamic_force = finite_vector(
            safe_read(lambda: self.flight.aerodynamic_force, (0.0, 0.0, 0.0))
        )
        available_thrust = safe_float(lambda: self.vessel.available_thrust, 0.0)
        if available_thrust <= 1.0:
            available_thrust = sum(safe_float(lambda e=engine: e.available_thrust, 0.0) for engine in self.engines)
        isp = safe_float(lambda: self.vessel.specific_impulse, 0.0)
        if isp <= 1.0:
            isp = max([safe_float(lambda e=engine: e.specific_impulse, 0.0) for engine in self.engines] or [1.0])
        legs_grounded = sum(bool(safe_read(lambda leg=leg: leg.is_grounded, False)) for leg in self.vessel.parts.legs)
        resource_fraction = self.resource_fraction()
        return NavigationState(
            ut=safe_float(lambda: self.space_center.ut, 0.0),
            monotonic_s=time.monotonic(),
            position_m=finite_vector(safe_read(lambda: self.vessel.position(self.reference_frame), ())),
            velocity_ms=finite_vector(safe_read(lambda: self.vessel.velocity(self.reference_frame), ())),
            direction=finite_vector(safe_read(lambda: self.vessel.direction(self.reference_frame), (1.0, 0.0, 0.0))),
            vessel_up=finite_vector(
                safe_read(
                    lambda: self.space_center.transform_direction(
                        (0.0, 0.0, 1.0), self.vessel.reference_frame, self.reference_frame
                    ),
                    (0.0, 1.0, 0.0),
                )
            ),
            angular_velocity_rads=-finite_vector(
                safe_read(lambda: self.vessel.angular_velocity(self.reference_frame), (0.0, 0.0, 0.0))
            ),
            mass_kg=mass,
            available_thrust_n=available_thrust,
            specific_impulse_s=isp,
            aerodynamic_accel_ms2=aerodynamic_force / max(mass, 1.0),
            pitch_deg=safe_float(lambda: self.flight.pitch, float("nan")),
            heading_deg=safe_float(lambda: self.flight.heading, float("nan")),
            roll_deg=safe_float(lambda: self.flight.roll, float("nan")),
            situation=str(safe_read(lambda: self.vessel.situation, "")),
            legs_grounded=legs_grounded,
            resource_fraction=resource_fraction,
        )

    def resource_fraction(self) -> float:
        amounts = 0.0
        maxima = 0.0
        for name in ("LqdMethane", "Oxidizer"):
            amounts += safe_float(lambda n=name: self.vessel.resources.amount(n), 0.0)
            maxima += safe_float(lambda n=name: self.vessel.resources.max(n), 0.0)
        if maxima > 0.0:
            self._last_resource_fraction = amounts / maxima
        return self._last_resource_fraction

    def current_thrust(self) -> float:
        return safe_float(lambda: self.vessel.thrust, 0.0)

    def legs_retracted_count(self) -> int:
        count = 0
        for leg in self.vessel.parts.legs:
            state = str(safe_read(lambda current=leg: current.state, "")).lower()
            if "retracted" in state:
                count += 1
        return count

    def available_angular_acceleration(self) -> np.ndarray:
        torque = np.zeros(3)
        for name in (
            "available_reaction_wheel_torque",
            "available_rcs_torque",
            "available_engine_torque",
            "available_control_surface_torque",
        ):
            pair = safe_read(lambda n=name: getattr(self.vessel, n), None)
            if pair is None:
                continue
            try:
                positive = np.abs(np.asarray(pair[0], dtype=float))
                negative = np.abs(np.asarray(pair[1], dtype=float))
                torque += np.maximum(positive, negative)
            except Exception:
                continue
        inertia = finite_vector(safe_read(lambda: self.vessel.moment_of_inertia, (1.0, 1.0, 1.0)), (1.0, 1.0, 1.0))
        acceleration = torque / np.maximum(inertia, 1.0)
        return np.maximum(np.array([acceleration[1], acceleration[2], acceleration[0]]), 0.05)

    def prepare_controls(self) -> None:
        self.control.throttle = 0.0
        self.set_rcs(False)
        self.set_grid_fins(False)
        self.set_legs(False)
        for name in ("pitch", "yaw", "roll", "forward", "up", "right"):
            try:
                setattr(self.control, name, 0.0)
            except Exception:
                pass
        try:
            self.control.sas = False
            self._sas_stability_enabled = False
        except Exception:
            pass
        try:
            self.vessel.auto_pilot.disengage()
            self._direction_control_engaged = False
        except Exception:
            pass
        try:
            self.space_center.rails_warp_factor = 0
            self.space_center.physics_warp_factor = 0
        except Exception:
            pass

    def engage_direction_control(self, target_direction: np.ndarray | tuple[float, float, float]) -> None:
        """Let KSP's attitude autopilot close the body-axis loop.

        LCvx and the feedback layer still choose the thrust vector.  Delegating
        only the inner attitude loop avoids depending on craft-specific manual
        pitch/yaw signs while RCS, gimbal, reaction wheels and grid fins remain
        available to KSP's actuator controller.
        """
        direction = finite_vector(target_direction, (1.0, 0.0, 0.0))
        direction /= max(float(np.linalg.norm(direction)), 1e-12)
        autopilot = self.vessel.auto_pilot
        autopilot.reference_frame = self.reference_frame
        autopilot.target_direction = tuple(float(value) for value in direction)
        try:
            autopilot.target_roll = 0.0
        except Exception:
            pass
        if not self._direction_control_engaged:
            try:
                self.control.sas = False
                self._sas_stability_enabled = False
            except Exception:
                pass
            autopilot.engage()
            self._direction_control_engaged = True

    def disengage_direction_control(self) -> None:
        if not self._direction_control_engaged:
            return
        try:
            self.vessel.auto_pilot.disengage()
        finally:
            self._direction_control_engaged = False

    def set_sas_stability(self, enabled: bool) -> None:
        """Enable SAS stability assist without competing attitude controllers."""
        enabled = bool(enabled)
        if enabled == self._sas_stability_enabled:
            return
        if enabled:
            self.disengage_direction_control()
        try:
            self.control.sas = enabled
            if enabled:
                self.control.sas_mode = self.space_center.SASMode.stability_assist
            self._sas_stability_enabled = enabled
        except Exception:
            if enabled:
                raise

    def ignite_initial(self, initial_throttle: float) -> None:
        self.control.throttle = max(0.0, min(1.0, initial_throttle))
        if not any(bool(safe_read(lambda e=engine: e.active, False)) for engine in self.engines):
            self.control.activate_next_stage()
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            if any(bool(safe_read(lambda e=engine: e.active, False)) for engine in self.engines):
                return
            time.sleep(0.05)
        raise RuntimeError("engine did not activate after staging")

    def relight(self) -> None:
        for engine in self.engines:
            try:
                engine.active = True
            except Exception:
                pass
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            if any(bool(safe_read(lambda e=engine: e.active, False)) for engine in self.engines):
                return
            time.sleep(0.05)
        raise RuntimeError("engine relight failed")

    def shutdown_engines(self) -> None:
        self.control.throttle = 0.0
        for engine in self.engines:
            try:
                engine.active = False
            except Exception:
                pass

    def set_rcs(self, enabled: bool) -> None:
        self.control.rcs = bool(enabled)

    def set_grid_fins(self, deployed: bool) -> None:
        self.control.brakes = bool(deployed)

    def set_legs(self, deployed: bool) -> None:
        for name in ("gear", "legs"):
            try:
                setattr(self.control, name, bool(deployed))
            except Exception:
                pass

    def apply_command(self, command: ControlCommand) -> None:
        self.control.throttle = max(0.0, min(1.0, float(command.throttle)))
        if command.sas_stability:
            self.set_sas_stability(True)
        elif command.direction_control:
            self.set_sas_stability(False)
            self.engage_direction_control(command.target_direction)
        else:
            self.set_sas_stability(False)
            self.disengage_direction_control()
        self.control.roll = 0.0
        self.control.yaw = 0.0
        self.control.pitch = 0.0
        self.set_rcs(command.rcs)
        self.set_grid_fins(command.brakes)
        self.set_legs(command.legs)

    def secure_after_flight(self, landed: bool, deploy_legs: bool | None = None) -> None:
        self.control.throttle = 0.0
        for name in ("pitch", "yaw", "roll", "forward", "up", "right"):
            try:
                setattr(self.control, name, 0.0)
            except Exception:
                pass
        self.set_rcs(False)
        self.set_grid_fins(False)
        self.set_legs(landed if deploy_legs is None else deploy_legs)
        self.disengage_direction_control()
        try:
            self.control.sas = False
            self._sas_stability_enabled = False
        except Exception:
            pass
        if landed:
            self.shutdown_engines()


def validate_snapshot(snapshot: dict[str, Any], model: VehicleModel, config: MissionConfig) -> list[str]:
    failures: list[str] = []
    if model.name != config.vessel_name:
        failures.append(f"expected vessel {config.vessel_name!r}, got {model.name!r}")
    expected = {
        "engines": config.expected_engines,
        "rcs": config.expected_rcs,
        "grid_fins": config.expected_grid_fins,
        "legs": config.expected_legs,
    }
    for key, count in expected.items():
        actual = int(snapshot.get("counts", {}).get(key, -1))
        if actual != count:
            failures.append(f"expected {count} {key}, got {actual}")
    if not model.engine_restartable:
        failures.append("main engine is not restartable")
    if model.sea_level_twr < 1.2:
        failures.append(f"initial TWR {model.sea_level_twr:.2f} is below 1.20")
    if model.current_mass_kg <= model.dry_mass_kg:
        failures.append("no usable propellant mass detected")
    return failures
