import numpy as np
import pytest
from types import SimpleNamespace
from concurrent.futures import Future

from condor_hopper.mission import FlightController, Sample, descent_attitude, attitude_error, unit


def sample(**values):
    defaults = dict(ut=0.0, altitude_m=85.0, height_m=0.0,
                    velocity=np.zeros(3), error=np.zeros(2), direction=np.array([1.0, 0.0, 0.0]),
                    aero=np.zeros(3), mass_kg=57_500.0, thrust_n=1_770_000.0,
                    gravity=9.81, grounded=0)
    defaults.update(values)
    return Sample(**defaults)


def test_ascent_full_throttle_until_sea_level_10000m():
    controller = FlightController()
    for altitude in (85.0, 5000.0, 9999.9):
        command = controller.update(sample(altitude_m=altitude, height_m=altitude - 85.0))
        assert command.throttle == 1.0 and command.engine
        assert not command.fins and not command.legs
        assert np.allclose(command.direction, [1.0, 0.0, 0.0])
    command = controller.update(sample(altitude_m=10000.0, height_m=9915.0, velocity=np.array([400.0, 0.0, 0.0])))
    assert controller.phase == "coast"
    assert command.throttle == 0.0 and not command.engine
    assert controller.cutoff_altitude_m == 10000.0


def test_descent_attitude_matches_body_pitch_and_yaw_signs():
    acceleration = np.ones(3) * 0.1
    right = descent_attitude([0.1, 1.0, 0.0], np.zeros(3), acceleration)
    down = descent_attitude([0.0, 1.0, 0.1], np.zeros(3), acceleration)
    assert right[2] > 0.0 and down[0] < 0.0


def test_descent_attitude_stops_rotation_when_already_aligned():
    rates = np.array([0.02, -0.01, 0.03])
    command = descent_attitude([0.0, 1.0, 0.0], rates, [0.1, 0.1, 0.1])
    assert np.all(command * rates < 0.0)


def test_moving_target_cannot_reverse_correction_of_a_large_attitude_error():
    command = descent_attitude([0.0, np.cos(0.25), np.sin(0.25)], np.zeros(3), [0.1, 0.1, 0.1],
                               reference_rate=np.array([0.087, 0.0, 0.0]))
    assert command[0] < 0.0


def test_descent_attitude_converges_in_both_pitch_directions():
    for initial in (-0.15, 0.15):
        angle, rate = initial, 0.0
        for _ in range(2000):
            local_target = np.array([0.0, np.cos(angle), np.sin(angle)])
            command = descent_attitude(local_target, np.array([rate, 0.0, 0.0]), [0.1, 0.1, 0.1])
            rate += command[0] * 0.1 * 0.02
            angle += rate * 0.02
        assert abs(angle) < np.radians(0.2)
        assert abs(rate) < 0.001


def test_fins_stowed_through_apogee_then_open_after_descent():
    controller = FlightController()
    controller.phase = "coast"
    for ut, speed in ((0.0, 100.0), (1.0, 0.0), (2.0, -1.0)):
        command = controller.update(sample(ut=ut, height_m=10_000.0, velocity=np.array([speed, 0.0, 0.0])))
        assert not command.fins and command.throttle == 0.0
        assert np.allclose(command.direction, [1.0, 0.0, 0.0])
    command = controller.update(sample(ut=2.6, height_m=10_000.0, velocity=np.array([-6.0, 0.0, 0.0])))
    assert controller.phase == "descent" and command.fins and not command.engine


def test_descent_confirmation_resets_when_vessel_rises():
    controller = FlightController()
    controller.phase = "coast"
    for ut, speed in ((0.0, -1.0), (0.3, 0.1), (0.6, -1.0)):
        assert not controller.update(sample(ut=ut, height_m=5000.0, velocity=np.array([speed, 0.0, 0.0]))).fins


def test_replanning_cannot_keep_delaying_an_accepted_ignition():
    controller = FlightController()
    controller.phase = "descent"
    initial = SimpleNamespace(ignition_ut=8.0, touchdown_ut=30.0)
    controller.plan = initial
    controller.last_plan_ut = 5.0
    controller.pending = Future()
    controller.pending.set_result(SimpleNamespace(ignition_ut=10.0, touchdown_ut=30.0))
    controller.update_plan(sample(ut=5.0, height_m=6000.0))
    assert controller.plan is initial
    controller.close()


def test_fins_aim_to_finish_horizontal_return_before_powered_touchdown():
    controller = FlightController()
    state = sample(height_m=5000.0, mass_kg=40000.0, thrust_n=1800000.0,
                   velocity=np.array([-300.0, 10.0, 0.0]), error=np.array([140.55, 0.0]))
    direction = controller.fin_direction(state)
    # 无动力落地时间更短；应先向目标加速，不能把位移留到燃烧末段。
    assert direction[1] < unit([300.0, -10.0, 0.0])[1]
    controller.close()


def test_contact_cuts_engine_and_bounce_cannot_restart_it():
    controller = FlightController()
    controller.phase = "landing"
    assert controller.update(sample(ut=100.0, grounded=3)).throttle == 0.0
    assert not controller.update(sample(ut=100.1, grounded=3)).hold_attitude
    bounce = sample(ut=100.2, height_m=0.5, velocity=np.array([1.0, 0.0, 0.0]))
    assert not controller.update(bounce).engine and controller.phase == "contact"
    controller.update(sample(ut=101.0, grounded=4))
    controller.update(sample(ut=102.1, grounded=4))
    assert controller.phase == "landed" and controller.success


def test_tip_after_contact_aborts_instead_of_waiting_for_stability():
    controller = FlightController()
    controller.phase = "contact"
    with pytest.raises(RuntimeError, match="侧翻"):
        controller.update(sample(direction=np.array([0.5, 0.0, 0.866])))
    controller.close()


@pytest.mark.parametrize("distance,success", [(4.9, True), (5.1, False)])
def test_landing_accepts_only_positions_within_five_metres(distance, success):
    controller = FlightController()
    controller.phase = "landing"
    for ut in (100.0, 101.1):
        controller.update(sample(ut=ut, grounded=4, error=np.array([distance, 0.0])))
    assert controller.phase == "landed" and controller.success == success




def test_near_ground_never_chases_position_or_hovers_for_it():
    near, far = FlightController(), FlightController()
    near.phase = far.phase = "landing"
    state = sample(height_m=80.0, velocity=np.array([-30.0, 3.0, -2.0]))
    command = near.update(state)
    state.error = np.array([1000.0, -500.0])
    distant_command = far.update(state)
    assert command.throttle == distant_command.throttle
    assert np.allclose(command.direction, distant_command.direction)
    assert far.capture_missed
    near.close()
    far.close()


def test_inverted_vehicle_never_ignites_toward_ground():
    controller = FlightController()
    controller.phase = "landing"
    command = controller.update(sample(height_m=100.0, velocity=np.array([-100.0, 0.0, 0.0]),
                                       direction=np.array([-1.0, 0.0, 0.0])))
    assert not command.engine and command.throttle == 0.0
    controller.close()


def test_terminal_command_does_not_accelerate_up_through_touchdown():
    controller = FlightController()
    controller.phase = "landing"
    controller.terminal_acceleration = 30.0
    state = sample(height_m=0.03, velocity=np.array([-0.8, 0.0, 0.0]))
    command = controller.update(state)
    acceleration = command.throttle * state.thrust_n / state.mass_kg * command.direction[0] - state.gravity
    assert state.velocity[0] + 0.05 * acceleration <= -0.7 + 1e-6
    controller.close()


def test_terminal_tilt_shrinks_before_contact():
    for height in (20.0, 5.0):
        controller = FlightController()
        controller.phase = "landing"
        command = controller.update(sample(height_m=height, velocity=np.array([-30.0, 5.0, 0.0])))
        angle = np.degrees(np.arccos(command.direction[0]))
        assert angle <= 5.0 * height / 40.0 + 1e-6
        controller.close()


def test_zero_thrust_keeps_a_finite_upright_direction():
    assert np.allclose(unit(np.zeros(3)), [1.0, 0.0, 0.0])


def test_powered_descent_can_correct_a_25ms_entry_speed_error():
    controller = FlightController()
    controller.phase = "landing"
    controller.touchdown_ut = 20.0
    controller.plan = object()
    controller.update_plan = lambda state: None
    state = sample(height_m=2500.0, velocity=np.array([-375.0, 0.0, 0.0]), aero=np.array([10.0, 0.0, 0.0]),
                   mass_kg=40000.0, thrust_n=1800000.0)
    command = controller.update(state)
    net_braking = command.throttle * state.thrust_n / state.mass_kg + state.aero[0] - state.gravity
    assert net_braking == pytest.approx((375.0**2 - 0.7**2) / (2.0 * 2500.0))
    controller.close()


def test_attitude_tracks_even_when_reported_torque_is_too_large():
    angle, rate = 0.15, 0.0
    bias = np.zeros(3)
    for _ in range(1500):
        target = [0.0, np.cos(angle), np.sin(angle)]
        command = descent_attitude(target, np.array([rate, 0.0, 0.0]), [1.0, 1.0, 1.0], bias)
        error = attitude_error(target)
        correcting = (np.abs(command) < 0.99) | (command * error < 0.0)
        bias = np.clip(bias + 0.3 * 0.02 * error * correcting, -0.8, 0.8)
        rate += (0.1 * command[0] + 0.02) * 0.02
        angle += rate * 0.02
    assert abs(angle) < np.radians(0.5)
    assert abs(rate) < 0.005
