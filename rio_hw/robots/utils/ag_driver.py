"""AG95 Modbus RTU driver for serial and raw RTU-over-TCP transports."""

import socket
import struct
import time
from enum import IntEnum
from typing import TYPE_CHECKING, Protocol

try:
    import serial  # ty: ignore[unresolved-import]
except ImportError as e:
    if TYPE_CHECKING:
        raise e
    serial = None


class GripperState(IntEnum):
    CLOSED = 0
    OPENED = 1


class _Transport(Protocol):
    """Byte-stream transport used by the Modbus RTU protocol."""

    def write(self, data: bytes) -> None: ...

    def read(self, size: int) -> bytes: ...

    def drain(self) -> None: ...

    def close(self) -> None: ...


class _TcpTransport:
    def __init__(self, host: str, port: int, timeout: float):
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._socket.settimeout(timeout)
        self._socket.connect((host, port))

    def write(self, data: bytes) -> None:
        self._socket.sendall(data)

    def read(self, size: int) -> bytes:
        return self._socket.recv(size)

    def drain(self) -> None:
        self._socket.setblocking(False)
        try:
            while self._socket.recv(4096):
                pass
        except BlockingIOError:
            pass
        finally:
            self._socket.setblocking(True)

    def close(self) -> None:
        self._socket.close()


class _SerialTransport:
    def __init__(self, port: str, baudrate: int, timeout: float):
        if serial is None:
            raise ImportError("Serial AG95 control requires pyserial")
        self._serial = serial.Serial(
            port=port,
            baudrate=baudrate,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=timeout,
            write_timeout=timeout,
        )

    def write(self, data: bytes) -> None:
        written = self._serial.write(data)
        if written != len(data):
            raise RuntimeError(f"Incomplete serial write: {written}/{len(data)} bytes")

    def read(self, size: int) -> bytes:
        return self._serial.read(size)

    def drain(self) -> None:
        self._serial.reset_input_buffer()

    def close(self) -> None:
        self._serial.close()


class AgGripperDriver:
    """Control a DH Robotics AG95 over Modbus RTU.

    Args:
        robot_ip: Host for raw Modbus RTU-over-TCP.
        tcp_port: TCP port used by the raw RTU bridge.
        member_id: Modbus slave ID.
        timeout: Read/write timeout in seconds.
        freq: Expected command frequency.
        connection_type: ``"TCPIP"`` or ``"SERIAL"``.
        robot_port: Serial device path, required for ``"SERIAL"``.
        baudrate: Serial baud rate.
    """

    def __init__(
        self,
        robot_ip: str = "192.168.0.15",
        tcp_port: int = 54321,
        member_id: int = 1,
        timeout: float = 1.0,
        freq: int = 30,
        *,
        connection_type: str = "TCPIP",
        robot_port: str | None = None,
        baudrate: int = 115200,
    ):
        self.robot_ip = robot_ip
        self.tcp_port = tcp_port
        self.member_id = member_id
        self.timeout = timeout
        self.freq = freq
        self.connection_type = connection_type.upper()
        if self.connection_type not in ("TCPIP", "SERIAL"):
            raise ValueError(f"connection_type must be TCPIP or SERIAL, got {connection_type!r}")
        if self.connection_type == "SERIAL" and not robot_port:
            raise ValueError("robot_port is required for SERIAL")
        self.robot_port = robot_port
        self.baudrate = baudrate
        self._transport: _Transport | None = None

        # Calculate drain interval to support at least 50Hz response
        # Drain at least twice the control frequency or 50Hz, whichever is higher
        drain_freq = max(freq * 2, 50)
        self.drain_interval = 1.0 / drain_freq

        # State tracking
        self.is_ready = False
        self.binary_state = GripperState.CLOSED
        self.state = 0

        # Avoid redundant writes
        self._last_position_sent = None
        self._last_drain_time = 0

    @staticmethod
    def _crc16(data: bytes | bytearray) -> bytes:
        crc = 0xFFFF
        for pos in data:
            crc ^= pos
            for _ in range(8):
                if crc & 0x0001:
                    crc >>= 1
                    crc ^= 0xA001
                else:
                    crc >>= 1
        return struct.pack("<H", crc)

    def connect(self):
        if self.connection_type == "SERIAL":
            assert self.robot_port is not None
            self._transport = _SerialTransport(self.robot_port, self.baudrate, self.timeout)
        else:
            self._transport = _TcpTransport(self.robot_ip, self.tcp_port, self.timeout)
        time.sleep(1.0)

    def close_conn(self):
        if self._transport is not None:
            self._transport.close()
            self._transport = None

    def _require_transport(self) -> _Transport:
        if self._transport is None:
            raise RuntimeError("Gripper is not connected")
        return self._transport

    def _drain_buffer(self):
        """Periodically drain receive buffer at calculated frequency."""
        now = time.time()
        if now - self._last_drain_time > self.drain_interval:
            self._require_transport().drain()
            self._last_drain_time = now

    def _read_exact(self, size: int) -> bytes:
        transport = self._require_transport()
        response = bytearray()
        deadline = time.monotonic() + self.timeout
        while len(response) < size and time.monotonic() < deadline:
            chunk = transport.read(size - len(response))
            if chunk:
                response.extend(chunk)
        if len(response) != size:
            raise RuntimeError(f"Incomplete response from gripper: {len(response)}/{size} bytes")
        return bytes(response)

    def _validate_response(self, response: bytes, function: int) -> None:
        if response[0] != self.member_id:
            raise RuntimeError(f"Unexpected Modbus member ID {response[0]}, expected {self.member_id}")
        if response[1] == function | 0x80:
            raise RuntimeError(f"Gripper returned Modbus exception {response[2]}")
        if response[1] != function:
            raise RuntimeError(f"Unexpected Modbus function {response[1]}, expected {function}")
        if response[-2:] != self._crc16(response[:-2]):
            raise RuntimeError("Invalid Modbus CRC from gripper")

    def _write_register_blocking(self, reg_addr: int, value: int):
        """Blocking write - use only for setup."""
        frame = bytearray()
        frame.append(self.member_id)
        frame.append(0x06)
        frame += struct.pack(">H", reg_addr)
        frame += struct.pack(">H", value)
        frame += self._crc16(frame)

        transport = self._require_transport()
        transport.write(bytes(frame))
        response = self._read_exact(8)
        self._validate_response(response, 0x06)
        if response != bytes(frame):
            raise RuntimeError("Gripper write response does not echo the request")
        return response

    def _write_register_fast(self, reg_addr: int, value: int):
        """Fast non-blocking write."""
        if self.connection_type == "SERIAL":
            # Serial writes are cheap and their echo must be consumed. Blocking
            # here also prevents stale write echoes from corrupting later reads.
            return self._write_register_blocking(reg_addr, value)

        frame = bytearray()
        frame.append(self.member_id)
        frame.append(0x06)
        frame += struct.pack(">H", reg_addr)
        frame += struct.pack(">H", value)
        frame += self._crc16(frame)

        # Fire and forget
        try:
            self._require_transport().write(bytes(frame))
        except BlockingIOError:
            return False

        # Drain buffer at calculated frequency
        self._drain_buffer()
        return True

    def _read_register_blocking(self, reg_addr: int) -> int:
        frame = bytearray([self.member_id, 0x03])
        frame += struct.pack(">H", reg_addr)
        frame += struct.pack(">H", 1)
        frame += self._crc16(frame)

        transport = self._require_transport()
        transport.write(bytes(frame))
        response = self._read_exact(7)
        self._validate_response(response, 0x03)
        if response[2] != 2:
            raise RuntimeError(f"Unexpected Modbus byte count {response[2]}")
        return struct.unpack(">H", response[3:5])[0]

    def initialize(self, full=False):
        """Initialize gripper. full=True => full init (find min & max)."""
        val = 0xA5 if full else 0x01
        self._write_register_blocking(0x0100, val)
        # 0xA5 runs the jaws through the full stroke; Modbus writes during that
        # motion time out. 0x01 only homes toward open and is shorter.
        time.sleep(8.0 if full else 2.0)
        self._require_transport().drain()

    def set_force(self, percent: int):
        """Set grip force: 20-100%."""
        pct = max(20, min(100, percent))
        self._write_register_blocking(0x0101, pct)

    @property
    def commanded_position(self) -> int:
        """Return the last written target, or initial measured opening before a target is known."""
        return self.state if self._last_position_sent is None else self._last_position_sent

    def set_position(self, permil: int):
        """Set target position: 0-1000‰ (0=closed, 1000=open)."""
        pos = max(0, min(1000, permil))

        # Skip if position hasn't changed
        if pos == self._last_position_sent:
            return

        if self._write_register_fast(0x0103, pos) is not False:
            self._last_position_sent = pos

    def read_position(self) -> int:
        """Read and cache the measured opening in per-mille."""
        self.state = self._read_register_blocking(0x0202)
        return self.state

    def open(self):
        """Fully open gripper."""
        self.set_position(1000)
        self.binary_state = GripperState.OPENED

    def close(self):
        """Fully close gripper."""
        self.set_position(0)
        self.binary_state = GripperState.CLOSED

    def grip_width(self, width_permille):
        """Grip to a custom opening."""
        self.set_position(width_permille)

    def setup(self, force=50, do_calibration=True, startup_position: int | None = 0):
        """Set force and optional startup pose.

        Skip init ``0x01`` unless calibrating. That command fully opens the
        jaws; a later Modbus write while they are still moving times out.
        """
        self._require_transport().drain()
        if do_calibration:
            self.initialize(full=True)
        last_error: Exception | None = None
        for _ in range(5):
            try:
                self._require_transport().drain()
                self.set_force(force)
                if startup_position is not None:
                    startup_position = max(0, min(1000, startup_position))
                    self._write_register_blocking(0x0103, startup_position)
                    self._last_position_sent = startup_position
                    time.sleep(1.0)
                self.read_position()
                self.is_ready = True
                return
            except RuntimeError as exc:
                last_error = exc
                time.sleep(0.2)
        raise last_error
