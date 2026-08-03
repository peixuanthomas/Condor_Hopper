#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from condor_hopper.flight_logging import FlightLogger
from condor_hopper.mission import MissionRunner
from condor_hopper.models import MissionConfig
from condor_hopper.simulation import simulate_mission
from condor_hopper.vehicle import KrpcVehicle, connect_krpc


PROJECT_DIR = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Condor Hopper 15 km LCvx guidance and control")
    subparsers = parser.add_subparsers(dest="command", required=True)

    probe = subparsers.add_parser("probe", help="read the active vessel without changing controls")
    add_connection_args(probe)

    simulate = subparsers.add_parser("simulate", help="run the offline 3-DoF closed-loop simulation")
    add_mission_args(simulate)
    simulate.add_argument("--nodes", type=int, default=61, help="LCvx temporal nodes for simulation")
    simulate.add_argument("--horizontal-offset", type=float, default=0.0, help="initial north offset in meters")

    fly = subparsers.add_parser("fly", help="control the active vessel")
    add_connection_args(fly)
    add_mission_args(fly)
    fly.add_argument("--nodes", type=int, default=81, help="LCvx temporal nodes")
    fly.add_argument("--execute", action="store_true", help="required acknowledgement that staging and flight are intended")
    return parser


def add_connection_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--address", default="127.0.0.1")
    parser.add_argument("--rpc-port", type=int, default=50_000)
    parser.add_argument("--stream-port", type=int, default=50_001)


def add_mission_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--target-altitude", type=float, default=15_000.0)


def config_from_args(args: argparse.Namespace) -> MissionConfig:
    config = MissionConfig()
    if hasattr(args, "target_altitude"):
        config.target_altitude_m = args.target_altitude
    if hasattr(args, "nodes"):
        config.node_count = args.nodes
    for name in ("address", "rpc_port", "stream_port"):
        if hasattr(args, name):
            setattr(config, name, getattr(args, name))
    config.log_root = PROJECT_DIR / "flight_logs"
    config.validate()
    return config


def run_probe(args: argparse.Namespace) -> int:
    config = config_from_args(args)
    connection = connect_krpc(config)
    try:
        adapter = KrpcVehicle(connection, config)
        print(json.dumps(adapter.static_snapshot(), ensure_ascii=False, indent=2, default=str))
    finally:
        connection.close()
    return 0


def run_simulation(args: argparse.Namespace) -> int:
    config = config_from_args(args)
    with FlightLogger(config.log_root) as logger:
        logger.write_config(config.json_dict())
        result = simulate_mission(config=config, horizontal_offset_m=args.horizontal_offset)
        payload = asdict(result)
        logger.summary(payload)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        print(f"Simulation log directory: {logger.directory.resolve()}")
    return 0 if result.success else 2


def run_flight(args: argparse.Namespace) -> int:
    if not args.execute:
        raise SystemExit("live flight refused: add --execute to acknowledge engine ignition and staging")
    config = config_from_args(args)
    with FlightLogger(config.log_root) as logger:
        logger.write_config(config.json_dict())
        connection = connect_krpc(config)
        try:
            adapter = KrpcVehicle(connection, config)
            runner = MissionRunner(adapter, config, logger)
            summary = runner.run()
        finally:
            connection.close()
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
        print(f"Flight log directory: {logger.directory.resolve()}")
    return 0 if summary.get("success") else 2


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "probe":
        return run_probe(args)
    if args.command == "simulate":
        return run_simulation(args)
    if args.command == "fly":
        return run_flight(args)
    raise SystemExit(f"unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
