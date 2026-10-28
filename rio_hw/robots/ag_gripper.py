import queue
from enum import Enum, auto
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

from .. import time
from ..interpolators import PoseTrajectoryInterpolator
from ..middleware import ClientFactory, ServerFactory
from ..node import Node
from ..request import Request

try:
    from .utils.ag_driver import AgGripperDriver
except ImportError as e:
    if TYPE_CHECKING:
        raise e
    else:
        AgGripperDriver = None


class RobotController(Enum):
    TASK_POS = auto()


class RequestType(Enum):
    MOVEG = auto()


class AgGripper(Node):
    __api__ = [
        "get_state",
        "get_all_state",
        "moveG",
    ]
    __pub__ = True
    __req__ = True

    def __init__(
        self,
        robot_ip: str = "192.168.0.15",
        tcp_port: int = 54321,
        robot_port: str | None = None,
        connection_type: str = "TCPIP",
        baudrate: int = 115200,
        member_id: int = 1,
        timeout: float = 5.0,
        robot_controller: str = "task_pos",
        max_gripper_speed: float | None = 3.0,
        force: int = 50,
        calibrate: bool = True,
        startup_position: float | None = 0.0,
        feedback_freq: float = 10.0,
        dtype=np.float32,
        *,
        freq: int = 30,
        max_buffer_size: int | None = None,
        max_queue_size: int = 1,
        **kwargs,
    ):
        controller = RobotController[robot_controller.upper()]
        if max_buffer_size is None:
            max_buffer_size = int(freq * 10)
        self.robot_ip = robot_ip
        self.tcp_port = tcp_port
        self.robot_port = robot_port
        self.connection_type = connection_type.upper()
        self.baudrate = baudrate
        self.member_id = member_id
        self.timeout = timeout
        self.robot_controller = controller
        self.max_gripper_speed = max_gripper_speed
        self.force = force
        self.calibrate = calibrate
        if startup_position is not None and not 0.0 <= startup_position <= 1.0:
            raise ValueError(f"startup_position must be in [0, 1] or None, got {startup_position}")
        if feedback_freq <= 0:
            raise ValueError(f"feedback_freq must be positive, got {feedback_freq}")
        self.startup_position = startup_position
        self.feedback_freq = feedback_freq
        self.dtype = dtype
        # ThreadClient.__init__ also uses `timeout` (wait for first publish). Keep
        # the serial I/O timeout on this instance after super() returns.
        ready_timeout = kwargs.pop("ready_timeout", 30.0)
        super().__init__(
            freq=freq,
            max_buffer_size=max_buffer_size,
            max_queue_size=max_queue_size,
            timeout=ready_timeout,
            **kwargs,
        )
        self.timeout = timeout

    def __post_init__(self):
        example_request_params = {
            "target_pos": np.zeros((1,), dtype=self.dtype),
        }
        request_params_keys = {
            RobotController.TASK_POS: (RequestType.MOVEG, ("target_pos",)),
        }[self.robot_controller][1]
        example_request_params = {k: example_request_params[k] for k in request_params_keys}
        example_request_params["target_time"] = time.now()

        example_robot_state = {
            "gripper_position": 0.0,
            "gripper_commanded_position": 0.0,
        }

        self.example_request = {
            "type": next(iter(RequestType)).value,
            **example_request_params,
        }
        self.example_data = {
            **example_robot_state,
            "timestamp": time.now(),
        }
        self.worker = None
        self.run = self.pubreq
        return super().__post_init__()

    def pubreq(self):
        gripper = AgGripperDriver(
            self.robot_ip,
            self.tcp_port,
            self.member_id,
            self.timeout,
            self.freq,
            connection_type=self.connection_type,
            robot_port=self.robot_port,
            baudrate=self.baudrate,
        )
        gripper.connect()
        startup_position = None if self.startup_position is None else round(self.startup_position * 1000)
        gripper.setup(
            force=self.force,
            do_calibration=self.calibrate,
            startup_position=startup_position,
        )
        logger.info(
            "AG gripper ready (calibrate={}, position={:.3f})",
            self.calibrate,
            gripper.state / 1000.0,
        )

        try:
            if self.robot_controller == RobotController.TASK_POS:
                # Get current position (normalized 0-1 from per-mille 0-1000)
                curr_pos = gripper.state / 1000.0
                if self.max_gripper_speed is not None:
                    # pose interpolation
                    curr_time = time.now()
                    last_waypoint_time = curr_time
                    pose_interp = PoseTrajectoryInterpolator(times=[curr_time], poses=[[curr_pos, 0, 0, 0, 0, 0]])
                else:
                    target_pos = np.copy(curr_pos)
                    pose_interp = None
            else:
                raise ValueError(self.robot_controller)
            last_feedback_time = time.now()
            feedback_failures = 0
            have_target = startup_position is not None

            # Main loop
            dt = 1.0 / self.freq
            rate = time.Rate(self.freq)
            self.req_ready_event.set()
            not_pub_ready = True
            while not self.exit_event.is_set():
                t_now = time.now()
                # send command to robot
                if self.robot_controller == RobotController.TASK_POS and have_target:
                    if pose_interp is not None:
                        target_pos = pose_interp(t_now)[0]
                        # Convert normalized (0-1) to per-mille (0-1000)
                        target_permil = int(np.clip(target_pos, 0.0, 1.0) * 1000)
                    else:
                        target_permil = int(np.clip(target_pos, 0.0, 1.0) * 1000)
                    gripper.set_position(target_permil)
                elif self.robot_controller != RobotController.TASK_POS:
                    raise ValueError(self.robot_controller)

                # get state from robot
                if self.connection_type == "SERIAL" and t_now - last_feedback_time >= 1.0 / self.feedback_freq:
                    try:
                        gripper.read_position()
                    except RuntimeError as error:
                        feedback_failures += 1
                        if feedback_failures >= 10:
                            raise
                        if feedback_failures == 1:
                            logger.warning("AG gripper missed a feedback response: {}", error)
                    else:
                        feedback_failures = 0
                    last_feedback_time = t_now
                robot_state = {
                    "gripper_position": gripper.state / 1000.0,
                    "gripper_commanded_position": gripper.commanded_position / 1000.0,
                }

                # Store current state in ring buffer
                data = {
                    **robot_state,
                    "timestamp": time.now(),
                }
                self.ring_buffer.put(data)
                if not_pub_ready:
                    self.pub_ready_event.set()
                    not_pub_ready = False

                # Fetch requests from queue
                try:
                    reqs = self.request_queue.get_all()
                    if isinstance(reqs, dict):
                        reqs = [{k: reqs[k][i] for k in reqs.keys()} for i in range(len(reqs["type"]))]
                except queue.Empty:
                    reqs = []
                for r in reqs:
                    req = Request(RequestType(r.pop("type")), r)
                    if req.type == RequestType.MOVEG:
                        if not have_target and pose_interp is not None:
                            pose_interp = PoseTrajectoryInterpolator(
                                times=[t_now], poses=[[gripper.state / 1000.0, 0, 0, 0, 0, 0]]
                            )
                            last_waypoint_time = t_now
                        have_target = True
                        target_pos = np.array(req.params["target_pos"], dtype=self.dtype)[0]
                        target_time = float(req.params["target_time"])
                        if pose_interp is not None:
                            curr_time = t_now + dt
                            pose_interp = pose_interp.schedule_waypoint(
                                pose=[target_pos, 0, 0, 0, 0, 0],
                                time=target_time,
                                max_pos_speed=self.max_gripper_speed,
                                max_rot_speed=self.max_gripper_speed,
                                curr_time=curr_time,
                                last_waypoint_time=last_waypoint_time,
                            )
                            last_waypoint_time = target_time
                        else:
                            target_pos = np.clip(target_pos, 0.0, 1.0)
                    else:
                        raise ValueError(req.type)
                rate.precise_sleep()
        except KeyboardInterrupt:
            pass
        finally:
            gripper.close_conn()

    def get_state(self, k=None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)
        else:
            return self.ring_buffer.get_last_k(k=k, out=out)

    def get_all_state(self):
        return self.ring_buffer.get_all()

    def moveG(self, target_pos, target_time):
        target_pos = np.array(target_pos, dtype=self.dtype)
        assert target_pos.shape == (1,)
        now = time.now()
        if target_time <= now:
            target_time = now + 1.0 / max(int(self.freq), 1)
        req = {
            "type": RequestType.MOVEG.value,
            "target_pos": target_pos,
            "target_time": target_time,
        }
        self.request_queue.put(req)


def AgGripperServer(mw, *args, **kwargs):
    return ServerFactory(mw, AgGripper, *args, **kwargs)


def AgGripperClient(mw, *args, **kwargs):
    return ClientFactory(mw, AgGripper, *args, **kwargs)
