#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
import traceback
from pathlib import Path

import numpy as np

from condor_hopper.flight_logging import FlightLogger
from condor_hopper.landing import LandingState
from condor_hopper.mission import FlightController
from condor_hopper.vehicle import KrpcVehicle


PROJECT_DIR = Path(__file__).resolve().parent


def build_parser():
    parser = argparse.ArgumentParser(description="Condor Hopper 10000 米垂直起飞与定点着陆")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("probe", "fly"):
        command = commands.add_parser(name)
        command.add_argument("--address", default="127.0.0.1")
        command.add_argument("--rpc-port", type=int, default=50000)
        command.add_argument("--stream-port", type=int, default=50001)
        if name == "fly":
            command.add_argument("--target-altitude", type=float, default=10000.0, help="熄火海拔，单位米")
            command.add_argument("--execute", action="store_true", help="实际点火并执行一次飞行")
    return parser


def fly(vehicle, target_altitude):
    snapshot = vehicle.validate(target_altitude)
    logger = FlightLogger(PROJECT_DIR / "flight_logs")
    controller = FlightController(target_altitude, snapshot["dry_mass_kg"], snapshot["exhaust_velocity_ms"],
                                  snapshot["rotation_up_north_east"], snapshot["body_radius_m"])
    summary = dict(success=False, failure_reason="")
    state = None
    try:
        logger.json("preflight.json", snapshot)
        logger.json("config.json", dict(cutoff_sea_level_altitude_m=target_altitude,
                    landing_latitude=snapshot["target_latitude"], landing_longitude=snapshot["target_longitude"],
                    landing_surface_altitude_m=snapshot["target_surface_altitude_m"],
                    control_rate_hz=20, landing_distance_tolerance_m=5.0))
        initial = vehicle.read()
        controller.solver.solve(LandingState(initial.ut, np.array([1500.0, 0.0, 0.0]),
                                np.array([-200.0, 0.0, 0.0]), initial.mass_kg,
                                snapshot["dry_mass_kg"], initial.thrust_n, snapshot["exhaust_velocity_ms"],
                                initial.gravity, 0.0, np.array([1.0, 0.0, 0.0])), 15.0)
        vehicle.prepare()
        print(f"Flight logs: {logger.directory}", flush=True)
        started = time.monotonic()
        last_progress = started
        previous_ut = None
        last_print = 0.0
        saved_plan = 0
        while controller.phase != "landed":
            tick = time.monotonic()
            if tick - started > 600.0:
                raise RuntimeError("飞行超过 600 秒仍未完成")
            state = vehicle.read()
            if previous_ut is not None and state.ut < previous_ut:
                raise RuntimeError("游戏时间回退，请手动重置后重新试飞")
            if previous_ut is None or state.ut > previous_ut:
                last_progress = tick
            if tick - last_progress > 3.0:
                raise RuntimeError("游戏数据停止更新，请准备好试飞")
            previous_ut = state.ut
            command = controller.update(state)
            vehicle.apply(command, state)
            if controller.plan_number != saved_plan:
                plan = controller.plan
                logger.json(f"plan_{controller.plan_number:04d}.json", dict(
                    ignition_ut=plan.ignition_ut, touchdown_ut=plan.touchdown_ut,
                    times_s=plan.times.tolist(), position_m=plan.position.tolist(),
                    velocity_ms=plan.velocity.tolist(), acceleration_ms2=plan.acceleration.tolist(),
                    external_acceleration_ms2=plan.external.tolist(),
                    mass_kg=plan.mass.tolist(), fuel_kg=plan.fuel_kg,
                    slack_ms2=plan.slack_ms2, rollout_error_m=plan.rollout_error_m))
                saved_plan = controller.plan_number
            tilt = math.degrees(math.acos(float(np.clip(state.direction[0], -1.0, 1.0))))
            logger.row(dict(ut=state.ut, elapsed_s=tick - started, phase=controller.phase,
                       altitude_m=state.altitude_m, height_m=state.height_m, target_height_m=state.landing_height_m,
                       surface_altitude_m=vehicle.flight.surface_altitude,
                       touchdown_com_height_m=vehicle.touchdown_height,
                       vertical_speed_ms=state.velocity[0], north_speed_ms=state.velocity[1], east_speed_ms=state.velocity[2],
                       target_north_error_m=state.error[0], target_east_error_m=state.error[1],
                       distance_to_target_m=np.linalg.norm(state.error), tilt_deg=tilt,
                       direction_up=state.direction[0], direction_north=state.direction[1], direction_east=state.direction[2],
                       mass_kg=state.mass_kg, max_thrust_n=state.thrust_n, actual_thrust_n=vehicle.vessel.thrust,
                       throttle=command.throttle, engine_commanded=command.engine,
                       fins_commanded=vehicle.fins_deployed, legs_commanded=vehicle.legs_deployed,
                       fin_authority_percent=vehicle.fins[0].authority_limiter,
                       fin_pitch_enabled=vehicle.fins[0].pitch_enabled,
                       fin_yaw_enabled=vehicle.fins[0].yaw_enabled,
                       grounded_legs=state.grounded,
                       target_up=vehicle.last_direction[0], target_north=vehicle.last_direction[1], target_east=vehicle.last_direction[2],
                       pitch_input=vehicle.control.pitch, yaw_input=vehicle.control.yaw, roll_input=vehicle.control.roll,
                       attitude_error_deg=math.degrees(math.acos(float(np.clip(np.dot(state.direction, vehicle.last_direction), -1.0, 1.0)))),
                       angular_accel_pitch=vehicle.angular_accel[0], angular_accel_yaw=vehicle.angular_accel[2],
                       angular_rate_pitch=vehicle.angular_rate[0], angular_rate_yaw=vehicle.angular_rate[2],
                       thrust_direction_up=vehicle.thrust_direction[0],
                       thrust_direction_north=vehicle.thrust_direction[1], thrust_direction_east=vehicle.thrust_direction[2],
                       aero_up_ms2=state.aero[0], aero_north_ms2=state.aero[1], aero_east_ms2=state.aero[2],
                       plan_number=controller.plan_number, touchdown_ut=controller.touchdown_ut,
                       planned_ignition_ut=controller.plan.ignition_ut if controller.plan else None,
                       predicted_burn_fuel_kg=controller.plan.fuel_kg if controller.plan else None,
                       capture_missed=controller.capture_missed,
                       part_count=vehicle.part_count))
            if tick - last_print >= 1.0:
                last_print = tick
                print(f"{controller.phase:8} alt={state.altitude_m:8.1f}m vz={state.velocity[0]:7.1f}m/s "
                      f"range={np.linalg.norm(state.error):6.1f}m tilt={tilt:5.1f} thr={command.throttle:.2f}", flush=True)
            time.sleep(max(0.0, 0.05 - (time.monotonic() - tick)))
        summary["success"] = controller.success
        if not controller.success:
            summary["failure_reason"] = "已接地，但距指定着陆点超过 5 米"
    except (Exception, KeyboardInterrupt) as exc:
        summary["failure_reason"] = str(exc) or type(exc).__name__
        (logger.directory / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
    finally:
        try:
            vehicle.stop()
        except Exception as exc:
            summary["success"] = False
            summary["shutdown_error"] = str(exc)
        controller.close()
        summary.update(phase=controller.phase, cutoff_altitude_m=controller.cutoff_altitude_m,
                       cutoff_vertical_speed_ms=controller.cutoff_speed_ms,
                       apogee_altitude_m=controller.apogee_altitude_m)
        summary.update(ignition_ut=controller.ignition_ut, planned_touchdown_ut=controller.touchdown_ut,
                       capture_missed=controller.capture_missed)
        if state is not None:
            summary.update(distance_to_target_m=float(np.linalg.norm(state.error)),
                           vertical_speed_ms=float(state.velocity[0]),
                           horizontal_speed_ms=float(np.linalg.norm(state.velocity[1:])),
                           grounded_legs=state.grounded)
        logger.json("summary.json", summary)
        logger.close()
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0 if summary["success"] else 2


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "fly":
        if not args.execute:
            raise SystemExit("实际飞行需要 --execute")
        if not math.isfinite(args.target_altitude) or args.target_altitude <= 0:
            raise SystemExit("熄火海拔必须是大于零的有限数值")
    import krpc

    connection = krpc.connect(name="Condor Hopper", address=args.address,
                              rpc_port=args.rpc_port, stream_port=args.stream_port,
                              use_pregenerated_stubs=False)
    try:
        vehicle = KrpcVehicle(connection)
        if args.command == "probe":
            print(json.dumps(vehicle.snapshot(), ensure_ascii=False, indent=2))
            return 0
        return fly(vehicle, args.target_altitude)
    finally:
        connection.close()


if __name__ == "__main__":
    raise SystemExit(main())
