#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def load_columns(path: Path) -> dict[str, list[float | str]]:
    columns: dict[str, list[float | str]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            for key, value in row.items():
                try:
                    item: float | str = float(value)
                except (TypeError, ValueError):
                    item = value or ""
                columns.setdefault(key, []).append(item)
    return columns


def numeric(columns: dict[str, list[float | str]], name: str):
    import numpy as np

    size = max((len(values) for values in columns.values()), default=0)
    values = columns.get(name, [""] * size)
    converted = []
    for value in values:
        if isinstance(value, str) and value.lower() in {"true", "false"}:
            converted.append(1.0 if value.lower() == "true" else 0.0)
        else:
            converted.append(float(value) if value != "" else float("nan"))
    return np.array(converted, dtype=float)


def analyze(path: Path, output_dir: Path | None = None) -> list[Path]:
    import matplotlib.pyplot as plt

    output_dir = path.with_name(path.stem + "_plots") if output_dir is None else output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    columns = load_columns(path)
    time_s = numeric(columns, "mission_time_s")
    plots: list[tuple[str, list[tuple[str, str]]]] = [
        ("altitude_velocity", [("altitude_above_pad_m", "Altitude (m)"), ("velocity_up_ms", "Vertical speed (m/s)")]),
        ("horizontal", [("horizontal_error_m", "Horizontal error (m)"), ("horizontal_speed_ms", "Horizontal speed (m/s)")]),
        ("propulsion", [("throttle", "Throttle"), ("resource_fraction", "Propellant fraction")]),
        ("attitude", [("attitude_error_deg", "Attitude error (deg)"), ("control_pitch", "Pitch input"), ("control_yaw", "Yaw input")]),
        ("actuators", [("sas_stability", "SAS stability"), ("direction_control", "Direction autopilot"), ("rcs", "RCS"), ("brakes", "Grid fins"), ("legs", "Landing legs")]),
        ("solver", [("solver_wall_ms", "LCvx wall time (ms)"), ("solver_dynamics_residual", "Dynamics residual")]),
    ]
    generated: list[Path] = []
    for filename, series in plots:
        figure, axis = plt.subplots(figsize=(11, 5))
        for column, label in series:
            axis.plot(time_s, numeric(columns, column), label=label)
        axis.set_xlabel("Mission time (s)")
        axis.grid(True, alpha=0.3)
        axis.legend()
        figure.tight_layout()
        target = output_dir / f"{filename}.png"
        figure.savefig(target, dpi=150)
        plt.close(figure)
        generated.append(target)

    figure, axis = plt.subplots(figsize=(7, 7))
    axis.plot(numeric(columns, "position_east_m"), numeric(columns, "position_north_m"))
    axis.scatter([0.0], [0.0], marker="x", color="red", label="Launch target")
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("East (m)")
    axis.set_ylabel("North (m)")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    target = output_dir / "ground_track.png"
    figure.savefig(target, dpi=150)
    plt.close(figure)
    generated.append(target)
    return generated


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot Condor Hopper telemetry")
    parser.add_argument("csv", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    for path in analyze(args.csv, args.output_dir):
        print(path.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
