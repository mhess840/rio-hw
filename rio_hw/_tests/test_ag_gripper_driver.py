"""AG95 protocol tests using an in-memory transport (no hardware access)."""

import struct

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
    assert driver.state == 500


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
