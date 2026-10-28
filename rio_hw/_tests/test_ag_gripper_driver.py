"""AG95 protocol tests using an in-memory transport (no hardware access)."""

import struct
from types import SimpleNamespace

import numpy as np
import pytest

from rio_hw.robots.utils.ag_driver import AgGripperDriver


class FakeTransport:
    def __init__(self, registers: dict[int, int] | None = None):
        self.registers = registers or {}
        self.writes: list[bytes] = []
        self.response = bytearray()
        self.closed = False

    def write(self, data: bytes) -> None:
        self.writes.append(data)
        member_id, function = data[:2]
        if function == 0x06:
            self.response.extend(data)
        elif function == 0x03:
            register = struct.unpack(">H", data[2:4])[0]
            payload = bytes([member_id, function, 2]) + struct.pack(">H", self.registers[register])
            self.response.extend(payload + AgGripperDriver._crc16(payload))
        else:
            raise AssertionError(f"Unexpected function {function}")

    def read(self, size: int) -> bytes:
        data = bytes(self.response[:size])
        del self.response[:size]
        return data

    def drain(self) -> None:
        self.response.clear()

    def close(self) -> None:
        self.closed = True


def make_serial_driver(registers: dict[int, int] | None = None) -> tuple[AgGripperDriver, FakeTransport]:
    driver = AgGripperDriver(connection_type="SERIAL", robot_port="/dev/fake", timeout=0.01)
    transport = FakeTransport(registers)
    driver._transport = transport
    return driver, transport


def test_serial_position_write_uses_modbus_rtu_frame():
    driver, transport = make_serial_driver()

    driver.set_position(500)

    frame = transport.writes[-1]
    assert frame[:6] == bytes([1, 0x06, 0x01, 0x03, 0x01, 0xF4])
    assert frame[-2:] == driver._crc16(frame[:-2])
    assert driver.commanded_position == 500
    assert driver.state == 0


def test_tcp_mode_preserves_fire_and_forget_position_write():
    driver = AgGripperDriver()
    transport = FakeTransport()
    driver._transport = transport

    driver.set_position(750)

    frame = transport.writes[-1]
    assert frame[:6] == bytes([1, 0x06, 0x01, 0x03, 0x02, 0xEE])
    assert frame[-2:] == driver._crc16(frame[:-2])


def test_read_position_returns_measured_register():
    driver, transport = make_serial_driver({0x0202: 437})

    assert driver.read_position() == 437
    assert driver.state == 437
    assert transport.writes[-1][:6] == bytes([1, 0x03, 0x02, 0x02, 0x00, 0x01])


def test_repeated_commands_do_not_overwrite_measured_position():
    driver, transport = make_serial_driver({0x0202: 271})
    driver.set_position(0)
    driver.read_position()
    writes = len(transport.writes)
    driver.set_position(0)
    assert len(transport.writes) == writes
    assert driver.state == 271
    assert driver.commanded_position == 0


def test_failed_serial_write_is_retried_without_changing_measurement(monkeypatch):
    driver, transport = make_serial_driver({0x0202: 437})
    driver.read_position()
    write = driver._write_register_fast
    monkeypatch.setattr(driver, "_write_register_fast", lambda *args: (_ for _ in ()).throw(RuntimeError("write failed")))
    with pytest.raises(RuntimeError, match="write failed"):
        driver.set_position(700)
    assert driver._last_position_sent is None
    assert driver.state == 437
    monkeypatch.setattr(driver, "_write_register_fast", write)
    driver.set_position(700)
    assert driver.commanded_position == 700
    assert driver.state == 437
    assert transport.writes[-1][1] == 0x06


def test_full_tcp_buffer_does_not_cache_an_unsent_target(monkeypatch):
    driver = AgGripperDriver()
    transport = FakeTransport()
    driver._transport = transport
    monkeypatch.setattr(transport, "write", lambda *args: (_ for _ in ()).throw(BlockingIOError()))
    driver.set_position(700)
    assert driver._last_position_sent is None
    assert driver.state == 0


def test_setup_can_initialize_without_startup_motion(monkeypatch):
    driver, transport = make_serial_driver({0x0202: 610})
    monkeypatch.setattr("rio_hw.robots.utils.ag_driver.time.sleep", lambda _seconds: None)

    driver.setup(force=25, do_calibration=False, startup_position=None)

    written_registers = [struct.unpack(">H", frame[2:4])[0] for frame in transport.writes]
    assert written_registers == [0x0101, 0x0202]
    assert driver.state == 610
    assert driver.is_ready


def test_setup_calibration_writes_full_init(monkeypatch):
    driver, transport = make_serial_driver({0x0202: 0})
    sleeps: list[float] = []
    monkeypatch.setattr("rio_hw.robots.utils.ag_driver.time.sleep", sleeps.append)

    driver.setup(force=50, do_calibration=True, startup_position=None)

    init_frame = transport.writes[0]
    assert init_frame[:6] == bytes([1, 0x06, 0x01, 0x00, 0x00, 0xA5])
    assert sleeps[0] == 8.0


def test_invalid_crc_is_rejected():
    driver, transport = make_serial_driver({0x0202: 500})
    original_write = transport.write

    def corrupting_write(data: bytes) -> None:
        original_write(data)
        transport.response[-1] ^= 0xFF

    transport.write = corrupting_write

    with pytest.raises(RuntimeError, match="CRC"):
        driver.read_position()


def test_serial_requires_port():
    with pytest.raises(ValueError, match="robot_port"):
        AgGripperDriver(connection_type="SERIAL")


def test_publisher_reports_measurement_and_command_independently(monkeypatch):
    from rio_hw.robots import ag_gripper

    driver, transport = make_serial_driver({0x0202: 437})
    monkeypatch.setattr(driver, "connect", lambda: None)
    monkeypatch.setattr(ag_gripper, "AgGripperDriver", lambda *args, **kwargs: driver)
    node = object.__new__(ag_gripper.AgGripper)
    node.robot_ip = "unused"
    node.tcp_port = 0
    node.member_id = 1
    node.timeout = 0.01
    node.freq = 30
    node.dtype = np.float32
    node.connection_type = "SERIAL"
    node.robot_port = "/dev/fake"
    node.baudrate = 115200
    node.startup_position = None
    node.force = 50
    node.calibrate = False
    node.max_gripper_speed = None
    node.robot_controller = ag_gripper.RobotController.TASK_POS
    node.feedback_freq = 2.0
    checks = iter([False, False, True])
    requests = iter([[{"type": ag_gripper.RequestType.MOVEG.value, "target_pos": np.array([0.0]), "target_time": 1.0}], []])
    published = []
    node.exit_event = SimpleNamespace(is_set=lambda: next(checks))
    node.req_ready_event = SimpleNamespace(set=lambda: None)
    node.pub_ready_event = SimpleNamespace(set=lambda: None)
    node.request_queue = SimpleNamespace(get_all=lambda: next(requests))
    node.ring_buffer = SimpleNamespace(put=published.append)
    monkeypatch.setattr(ag_gripper.time, "Rate", lambda *args: SimpleNamespace(precise_sleep=lambda: None))
    node.pubreq()
    assert published[-1]["gripper_position"] == pytest.approx(0.437)
    assert published[-1]["gripper_commanded_position"] == 0.0
    position_writes = [frame for frame in transport.writes if frame[1] == 0x06 and frame[2:4] == b"\x01\x03"]
    assert len(position_writes) == 1
    assert transport.closed
