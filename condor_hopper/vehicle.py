from __future__ import annotations

import math

import numpy as np

from .mission import Command, Sample, attitude_error, descent_attitude, unit


class KrpcVehicle:
    def __init__(self, connection):
        self.sc = connection.space_center
        self.vessel = self.sc.active_vessel
        self.body = self.vessel.orbit.body
        self.frame = self.vessel.surface_reference_frame
        self.flight = self.vessel.flight(self.frame)
        self.control = self.vessel.control
        self.pilot = self.vessel.auto_pilot
        self.engines = list(self.vessel.parts.engines)
        self.legs = list(self.vessel.parts.legs)
        self.fins = [s for s in self.vessel.parts.control_surfaces if "grid fin" in s.part.name.lower()]
        self.fin_modules = [m for s in self.fins for m in s.part.modules if m.name == "ModuleAnimateGeneric"]
        self.launch_latitude = self.flight.latitude
        self.launch_longitude = self.flight.longitude
        self.touchdown_height = self.flight.surface_altitude
        self.target_latitude = -(12.0 / 60.0 + 21.0 / 3600.0)
        self.target_longitude = -(74.0 + 28.0 / 60.0 + 23.0 / 3600.0)
        self.target_terrain_altitude = self.body.surface_height(self.target_latitude, self.target_longitude)
        self.target_body_position = np.array(self.body.position_at_altitude(
            self.target_latitude, self.target_longitude,
            self.target_terrain_altitude + self.touchdown_height, self.body.reference_frame))
        self.part_count = len(self.vessel.parts.all)
        self.mu = self.body.gravitational_parameter
        self.radius = self.body.equatorial_radius
        self.fins_deployed = False
        self.legs_deployed = False
        self.engine_active = bool(self.engines and self.engines[0].active)
        self.last_direction = np.array([1.0, 0.0, 0.0])
        self.last_ut = None
        self.velocity_check_state = None
        self.attitude_active = False
        self.manual_descent = False
        self.angular_accel = np.ones(3)
        self.angular_rate = np.zeros(3)
        self.attitude_bias = np.zeros(3)
        self.last_torque_update = float("-inf")
        self.max_thrust = sum(e.max_thrust * e.thrust_limit for e in self.engines)
        self.streams = dict(
            ut=connection.add_stream(getattr, self.sc, "ut"),
            altitude=connection.add_stream(getattr, self.flight, "mean_altitude"),
            surface_altitude=connection.add_stream(getattr, self.flight, "surface_altitude"),
            mass=connection.add_stream(getattr, self.vessel, "mass"),
            velocity=connection.add_stream(self.vessel.velocity, self.body.reference_frame),
            position=connection.add_stream(self.vessel.position, self.body.reference_frame),
            direction=connection.add_stream(self.vessel.direction, self.frame),
            angular_velocity=connection.add_stream(self.vessel.angular_velocity, self.body.reference_frame),
            aero=connection.add_stream(getattr, self.flight, "aerodynamic_force"))
        for stream in self.streams.values():
            stream.rate = 50.0
            stream.start()

    def snapshot(self):
        state = self.read()
        return dict(name=self.vessel.name, situation=str(self.vessel.situation),
                    latitude=self.launch_latitude, longitude=self.launch_longitude,
                    target_latitude=self.target_latitude, target_longitude=self.target_longitude,
                    target_surface_altitude_m=self.target_terrain_altitude,
                    distance_to_target_m=float(np.linalg.norm(state.error)),
                    altitude_m=state.altitude_m, surface_altitude_m=self.flight.surface_altitude,
                    mass_kg=state.mass_kg, dry_mass_kg=self.vessel.dry_mass,
                    speed_ms=float(np.linalg.norm(state.velocity)), direction=state.direction.tolist(),
                    parts=self.part_count, engines=len(self.engines), grid_fins=len(self.fins), legs=len(self.legs),
                    restartable=all(e.can_restart for e in self.engines),
                    legs_retracted=all(not leg.deployed for leg in self.legs),
                    fins_retracted=all("Extend Fins" in m.events for m in self.fin_modules),
                    max_thrust_n=self.max_thrust,
                    exhaust_velocity_ms=self.engines[0].specific_impulse * 9.80665,
                    body_radius_m=self.radius,
                    rotation_up_north_east=(self.body.rotational_speed * np.array([
                        math.sin(math.radians(self.launch_latitude)), math.cos(math.radians(self.launch_latitude)), 0.0])).tolist(),
                    resources=[dict(name=r.name, amount=r.amount, capacity=r.max) for r in self.vessel.resources.all])

    def validate(self, cutoff_altitude):
        snapshot = self.snapshot()
        situation = snapshot["situation"].lower()
        if "pre_launch" not in situation and "landed" not in situation:
            raise RuntimeError("飞行器不在地面，请准备好试飞")
        if (snapshot["engines"], snapshot["grid_fins"], snapshot["legs"]) != (1, 4, 4):
            raise RuntimeError("当前飞行器应有一台发动机、四片栅格舵和四条支腿")
        if not snapshot["restartable"] or self.max_thrust <= 1.2 * self.vessel.mass * self.body.surface_gravity:
            raise RuntimeError("发动机不能重新点火，或起飞推力不足")
        if len(self.fin_modules) != 4 or not snapshot["fins_retracted"] or not snapshot["legs_retracted"]:
            raise RuntimeError("请先收起四片栅格舵和四条支腿")
        if snapshot["speed_ms"] > 0.5 or snapshot["direction"][0] < math.cos(math.radians(3.0)):
            raise RuntimeError("飞行器尚未静止竖直放好，请准备好试飞")
        if snapshot["altitude_m"] >= cutoff_altitude:
            raise RuntimeError("起飞海拔已经达到熄火高度")
        return snapshot

    def prepare(self):
        self.control.throttle = 0.0
        self.control.sas = False
        self.control.rcs = False
        for fin in self.fins:
            fin.pitch_enabled = True
            fin.yaw_enabled = True
            fin.roll_enabled = False
            fin.authority_limiter = 100.0
        self.pilot.reference_frame = self.frame
        self.pilot.target_direction = (1.0, 0.0, 0.0)
        self.pilot.target_roll = float("nan")
        self.pilot.stopping_time = (1.0, 1.0, 1.0)
        self.pilot.deceleration_time = (3.0, 3.0, 3.0)
        self.pilot.auto_tune = True
        self.pilot.engage()
        self.attitude_active = True

    def read(self):
        if self.sc.active_vessel != self.vessel or len(self.vessel.parts.all) != self.part_count:
            raise RuntimeError("飞行器已切换或部件损坏，停止本次试飞")
        # 时间和高度连续读取缓存，避免后台求解耗时被误算成高度采样时间。
        ut, altitude = self.streams["ut"](), self.streams["altitude"]()
        mass = self.streams["mass"]()
        if mass <= 0.0:
            raise RuntimeError("载具已损坏或消失，请手动重置飞行器")
        # surface_reference_frame 跟随载具移动，在其中直接读取载具速度会得到零。
        ground_velocity = self.streams["velocity"]()
        velocity = np.array(self.sc.transform_direction(ground_velocity, self.body.reference_frame, self.frame))
        offset = self.target_body_position - np.array(self.streams["position"]())
        error = np.array(self.sc.transform_direction(tuple(offset), self.body.reference_frame, self.frame))[1:]
        if self.legs_deployed:
            self.touchdown_height = max(-leg.part.bounding_box(self.frame)[0][0] for leg in self.legs) + 0.5
        state = Sample(ut=ut, altitude_m=altitude,
                       height_m=self.streams["surface_altitude"]() - self.touchdown_height,
                       target_height_m=altitude - self.target_terrain_altitude - self.touchdown_height,
                       velocity=velocity, error=error, direction=np.array(self.streams["direction"]()),
                       aero=np.array(self.streams["aero"]()) / mass, mass_kg=mass,
                       thrust_n=sum(e.max_thrust * e.thrust_limit for e in self.engines),
                       gravity=self.mu / (self.radius + altitude)**2,
                       grounded=sum(leg.is_grounded for leg in self.legs))
        if not np.all(np.isfinite([state.ut, altitude, state.height_m, mass, state.thrust_n,
                                   *velocity, *error, *state.direction, *state.aero])):
            raise RuntimeError("kRPC 飞行数据异常，请准备好试飞")
        previous = self.velocity_check_state
        if previous is None or state.ut - previous.ut >= 0.5:
            if previous is not None:
                measured_speed = (altitude - previous.altitude_m) / (state.ut - previous.ut)
                reported_speed = (velocity[0] + previous.velocity[0]) / 2.0
                if abs(measured_speed - reported_speed) > max(20.0, 0.3 * abs(measured_speed)):
                    raise RuntimeError(f"高度变化与升降速度不符：高度变化 {measured_speed:.1f}m/s，"
                                       f"速度读数 {reported_speed:.1f}m/s，间隔 {state.ut - previous.ut:.3f}s")
            self.velocity_check_state = state
        return state

    def apply(self, command: Command, state: Sample):
        if command.fins and not self.fins_deployed:
            for module in self.fin_modules:
                module.trigger_event("Extend Fins")
            self.fins_deployed = True
            self.pilot.disengage()
            self.attitude_active = False
            self.manual_descent = True
        if command.legs and not self.legs_deployed:
            self.control.legs = True
            self.legs_deployed = True
        if command.engine != self.engine_active:
            self.control.throttle = 0.0
            for engine in self.engines:
                engine.active = command.engine
            self.engine_active = command.engine
            self.attitude_bias[:] = 0.0
            if command.engine and self.manual_descent:
                for fin in self.fins:
                    fin.pitch_enabled = fin.yaw_enabled = False
                self.last_torque_update = float("-inf")
        if not command.hold_attitude:
            self.control.throttle = 0.0
            if self.attitude_active:
                self.pilot.disengage()
                self.attitude_active = False
            self.control.pitch = self.control.yaw = self.control.roll = 0.0
            self.control.rcs = False
            return
        if state.height_m > 1.0 and not self.control.rcs:
            self.control.rcs = True
        dt = 0.05 if self.last_ut is None else max(0.0, state.ut - self.last_ut)
        self.last_ut = state.ut
        angle = math.acos(float(np.clip(np.dot(self.last_direction, command.direction), -1.0, 1.0)))
        turn_rate = (4.0 if self.engine_active else 2.5) if self.manual_descent else 5.0
        fraction = min(1.0, math.radians(turn_rate) * dt / max(angle, 1e-9))
        old_direction = self.last_direction.copy()
        self.last_direction = unit((1.0 - fraction) * self.last_direction + fraction * command.direction)
        if self.manual_descent:
            if state.ut - self.last_torque_update >= 0.25:
                torque = np.zeros(3)
                # 本机服务端的舵面转动力带正负号，先取幅值再相加，避免互相抵消。
                sources = [self.vessel.available_reaction_wheel_torque, self.vessel.available_rcs_torque,
                           self.vessel.available_engine_torque]
                if not self.engine_active:
                    sources += [fin.available_torque for fin in self.fins]
                for positive, negative in sources:
                    torque += np.maximum(np.abs(positive), np.abs(negative))
                self.angular_accel = torque / np.asarray(self.vessel.moment_of_inertia)
                self.last_torque_update = state.ut
            local_target = self.sc.transform_direction(tuple(self.last_direction), self.frame, self.vessel.reference_frame)
            angular_velocity = self.streams["angular_velocity"]()
            rate = -np.array(self.sc.transform_direction(angular_velocity, self.body.reference_frame, self.vessel.reference_frame))
            self.angular_rate = rate
            requested_rate = -np.cross(old_direction, self.last_direction) / max(dt, 0.02)
            requested_rate = np.array(self.sc.transform_direction(tuple(requested_rate), self.frame, self.vessel.reference_frame))
            maximum_rate = math.radians(5.0) if self.engine_active else 0.05
            command_input = descent_attitude(local_target, rate, self.angular_accel, self.attitude_bias, requested_rate, maximum_rate)
            error = attitude_error(local_target)
            correcting = (np.abs(command_input) < 0.99) | (command_input * error < 0.0)
            # 抵消持续把机体推回迎风方向的空气作用，避免有偏角却一直转不过去。
            self.attitude_bias = np.clip(self.attitude_bias + 0.3 * dt * error * correcting, -0.8, 0.8)
            self.attitude_bias[1] = 0.0
            pitch, roll, yaw = command_input
            self.control.pitch, self.control.roll, self.control.yaw = float(pitch), float(roll), float(yaw)
        else:
            self.pilot.target_direction = tuple(self.last_direction)
        self.control.throttle = command.throttle

    def stop(self):
        self.control.throttle = 0.0
        for engine in self.engines:
            engine.active = False
        self.pilot.disengage()
        self.control.pitch = self.control.yaw = self.control.roll = 0.0
        self.control.rcs = False
