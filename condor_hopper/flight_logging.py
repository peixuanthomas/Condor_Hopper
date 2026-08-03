from __future__ import annotations

import csv
import json
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


TELEMETRY_FIELDS = [
    "timestamp", "ut", "mission_time_s", "phase", "event", "failure_reason",
    "position_up_m", "position_north_m", "position_east_m", "altitude_above_pad_m",
    "velocity_up_ms", "velocity_north_ms", "velocity_east_ms", "horizontal_error_m", "horizontal_speed_ms",
    "target_up_m", "target_north_m", "target_east_m", "target_velocity_up_ms", "target_velocity_north_ms", "target_velocity_east_ms",
    "position_error_up_m", "position_error_north_m", "position_error_east_m",
    "target_accel_up_ms2", "target_accel_north_ms2", "target_accel_east_ms2", "target_accel_norm_ms2",
    "target_direction_up", "target_direction_north", "target_direction_east", "attitude_error_deg",
    "pitch_deg", "heading_deg", "roll_deg", "angular_rate_rads",
    "control_pitch", "control_yaw", "control_roll", "throttle", "current_thrust_n", "available_thrust_n",
    "mass_kg", "resource_fraction", "rcs", "brakes", "gear", "legs", "sas_stability", "direction_control", "legs_grounded",
    "aero_accel_up_ms2", "aero_accel_north_ms2", "aero_accel_east_ms2",
    "solver_status", "solver_name", "solver_duration_s", "solver_wall_ms", "solver_core_ms", "solver_iterations",
    "solver_objective", "solver_dynamics_residual", "solver_cone_residual", "solver_terminal_position_error_m",
    "solver_terminal_velocity_error_ms", "predicted_tgo_s", "ignition_delay_s", "plan_age_s",
]


def _json_default(value: Any):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "value"):
        return value.value
    return str(value)


class FlightLogger:
    def __init__(self, root: Path, project_name: str = "Condor_Hopper", now: datetime | None = None) -> None:
        now = datetime.now().astimezone() if now is None else now
        self.run_id = now.strftime("%H%M%S")
        self.directory = root / f"{now.strftime('%Y-%m-%d')}_{project_name}"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.telemetry_path = self.directory / f"telemetry_{self.run_id}.csv"
        self.events_path = self.directory / f"events_{self.run_id}.jsonl"
        self.config_path = self.directory / f"config_{self.run_id}.json"
        self.snapshot_path = self.directory / f"vehicle_{self.run_id}.json"
        self.summary_path = self.directory / f"summary_{self.run_id}.json"
        self._telemetry_file = self.telemetry_path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._telemetry_file, fieldnames=TELEMETRY_FIELDS, extrasaction="ignore")
        self._writer.writeheader()
        self._telemetry_file.flush()

    def write_config(self, config: Any) -> None:
        self._write_json(self.config_path, config)

    def write_vehicle_snapshot(self, snapshot: Any) -> None:
        self._write_json(self.snapshot_path, snapshot)

    def telemetry(self, row: dict[str, Any]) -> None:
        self._writer.writerow({name: row.get(name, "") for name in TELEMETRY_FIELDS})
        self._telemetry_file.flush()

    def event(self, event: str, **payload: Any) -> None:
        record = {"timestamp": datetime.now().astimezone().isoformat(timespec="milliseconds"), "event": event, **payload}
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")

    def summary(self, payload: Any) -> None:
        self._write_json(self.summary_path, payload)

    def close(self) -> None:
        if not self._telemetry_file.closed:
            self._telemetry_file.flush()
            self._telemetry_file.close()

    def _write_json(self, path: Path, payload: Any) -> None:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
