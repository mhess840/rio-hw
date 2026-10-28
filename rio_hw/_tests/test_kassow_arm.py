"""Kassow motion-stop and feedback contracts without a controller connection."""

from queue import Queue
from types import SimpleNamespace

import numpy as np
import pytest

from rio_hw.robots import kassow_arm


def make_arm():
    arm = object.__new__(kassow_arm.KassowArm)
    arm.dtype = np.float64
    arm.num_joints = 7
    arm.request_queue = Queue()
    arm.example_request = {"type": 1, "target_eef_pose": np.zeros(6), "target_time": 0.0}
    return arm


def test_stop_motion_preserves_request_schema_without_a_new_target():
    arm = make_arm()
    arm.stop_motion()
    request = arm.request_queue.get_nowait()
    assert request["type"] == kassow_arm.RequestType.STOP_MOTION.value
    assert set(request) == set(arm.example_request)
    np.testing.assert_array_equal(request["target_eef_pose"], np.zeros(6))


def test_cancel_motion_discards_velocity_tracking_history():
    arm = make_arm()
    cancelled = []
    arm._bridge = SimpleNamespace(cancel_motion=lambda: cancelled.append(True))
    arm._vel_mode_active = True
    arm._qd_filter = object()
    arm._last_qd = np.ones(7)
    arm._last_qd_t = 1.0
    arm._cancel_motion()
    assert cancelled == [True]
    assert arm._vel_mode_active is False
    assert arm._qd_filter is None
    assert arm._last_qd is None
    assert arm._last_qd_t is None


@pytest.mark.parametrize("method,size", [("moveL", 6), ("moveJ", 7), ("speedJ", 7)])
def test_nonfinite_motion_requests_never_enter_the_queue(method, size):
    arm = make_arm()
    with pytest.raises(ValueError, match="finite"):
        getattr(arm, method)(np.full(size, np.nan), 1.0)
    assert arm.request_queue.empty()


def test_default_arm_does_not_clear_controller_safety():
    arm = make_arm()
    kassow_arm.KassowArm.__init__(arm, robot_controller="joint_vel")
    assert arm.clear_alarms_on_start is False


def test_outdated_native_bridge_is_rejected_before_hardware_connection(monkeypatch):
    connected = []
    bridge = SimpleNamespace(connect=lambda: connected.append(True))
    monkeypatch.setattr(kassow_arm, "_kord_bridge", SimpleNamespace(KordBridge=lambda *args: bridge))
    arm = make_arm()
    kassow_arm.KassowArm.__init__(arm, robot_controller="joint_vel")
    with pytest.raises(ImportError, match="watchdog API"):
        arm.__post_init__()
    assert not connected


def test_only_latest_streaming_goal_is_processed_after_a_scheduling_gap():
    arm = make_arm()
    arm.robot_controller = kassow_arm.RobotController.TASK_POS_IK
    requests = [{"type": kassow_arm.RequestType.MOVEL.value, "target": value} for value in range(10)]
    assert arm._coalesce_stream_requests(requests) == requests[-1:]


def test_discrete_and_stop_requests_are_not_discarded_by_stream_coalescing():
    arm = make_arm()
    arm.robot_controller = kassow_arm.RobotController.TASK_POS_IK
    requests = [
        {"type": kassow_arm.RequestType.MOVEL.value},
        {"type": kassow_arm.RequestType.STOP_MOTION.value},
        {"type": kassow_arm.RequestType.MOVEJ_DISCRETE.value},
    ]
    assert arm._coalesce_stream_requests(requests) == requests


def test_feedback_is_only_published_when_controller_tick_advances(monkeypatch):
    arm = make_arm()
    ticks = iter([0, 0, 1, 1, 2])
    checks = iter([False, False, False, False, True])
    published = []
    stopped = []

    def state():
        return SimpleNamespace(
            tick=next(ticks),
            stream_j_sends=0,
            stream_l_sends=0,
            tcp_pose=np.zeros(6),
            joint_q=np.zeros(7),
            joint_qd=np.zeros(7),
            joint_tau=np.zeros(7),
            alarm_code=0,
        )

    arm._bridge = SimpleNamespace(start=lambda: None, stop=lambda: stopped.append(True), get_state=state)
    arm.exit_event = SimpleNamespace(is_set=lambda: next(checks))
    arm.pub_ready_event = SimpleNamespace(set=lambda: None)
    arm.ring_buffer = SimpleNamespace(put=published.append)
    arm.robot_ip = "unused"
    arm.freq = 250
    arm.log_diagnostics = False
    monkeypatch.setattr(kassow_arm.time, "Rate", lambda *args: SimpleNamespace(precise_sleep=lambda: None))
    arm.pub()
    assert [int(value["controller_tick"]) for value in published] == [1, 2]
    assert stopped == [True]
