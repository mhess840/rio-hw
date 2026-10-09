"""Read a spare mouse as gripper buttons, without moving the desktop cursor.

The node opens one evdev mouse (default: Logitech Unifying Receiver), exclusive-
grabs it so motion never reaches X11/Wayland, and publishes the same keyboard-
shaped state as `Keyboard`: left click = close, right click = open.
"""

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

from .. import time
from ..middleware import ClientFactory, ServerFactory
from ..node import Node

try:
    import evdev
    from evdev import ecodes
except ImportError as e:
    if TYPE_CHECKING:
        raise e
    else:
        evdev = None  # type: ignore
        ecodes = None  # type: ignore

# USB HID usage codes matching rio.envs.poll / SshKeyboard presenter mapping.
BTN_LEFT = 0x110
BTN_RIGHT = 0x111
_EVDEV_TO_HID = {
    BTN_LEFT: 0x4E,  # PageDown → close
    BTN_RIGHT: 0x4B,  # PageUp → open
}

DEFAULT_DEVICE_PATH = "/dev/input/by-id/usb-Logitech_USB_Receiver-if02-event-mouse"


def find_clicker(device_path: str | None = None) -> str:
    """Return the evdev path of the spare Logitech mouse, never the HP desk mouse.

    Args:
        device_path: Optional explicit ``/dev/input/...`` path.

    Returns:
        Path to an event node that reports mouse buttons.

    Raises:
        RuntimeError: If no Logitech Unifying mouse is found.
    """
    if device_path:
        return device_path
    default = Path(DEFAULT_DEVICE_PATH)
    if default.exists():
        return str(default)
    by_id = Path("/dev/input/by-id")
    if by_id.is_dir():
        matches = sorted(
            path
            for path in by_id.glob("*event-mouse")
            if "Logitech" in path.name and "PixArt" not in path.name and "HP" not in path.name
        )
        if matches:
            return str(matches[0])
    raise RuntimeError(
        "No Logitech Unifying mouse found. Plug in the spare receiver "
        f"(expected {DEFAULT_DEVICE_PATH})."
    )


class Clicker(Node):
    __api__ = [
        "get_state",
        "get_all_state",
    ]
    __pub__ = True
    __req__ = False

    def __init__(
        self,
        n_key_rollover: int = 6,
        device_path: str | None = DEFAULT_DEVICE_PATH,
        required: bool = False,
        dtype=np.float32,
        *,
        freq: int = 100,
        max_buffer_size: int = 30,
        **kwargs,
    ):
        self.n_key_rollover = n_key_rollover
        self.device_path = device_path
        self.required = required
        self.dtype = dtype
        super().__init__(freq=freq, max_buffer_size=max_buffer_size, **kwargs)

    def __post_init__(self):
        self.example_request = None
        self.example_data = {
            "alphanumeric_state": np.zeros((self.n_key_rollover,), dtype=np.byte),
            "special_state": np.zeros((self.n_key_rollover,), dtype=int),
            "timestamp": time.now(),
        }
        self.worker = None
        self.run = self.pub
        super().__post_init__()

    def _empty_data(self):
        return {
            "alphanumeric_state": np.zeros((self.n_key_rollover,), dtype=np.byte),
            "special_state": np.zeros((self.n_key_rollover,), dtype=int),
            "timestamp": time.now(),
        }

    def _idle(self):
        rate = time.Rate(self.freq)
        not_pub_ready = True
        while not self.exit_event.is_set():
            self.ring_buffer.put(self._empty_data())
            if not_pub_ready:
                self.pub_ready_event.set()
                not_pub_ready = False
            rate.precise_sleep()

    def pub(self):
        device = None
        grabbed = False
        try:
            path = find_clicker(self.device_path)
            device = evdev.InputDevice(path)
        except Exception as exc:
            if self.required:
                raise
            logger.warning(
                "Gripper clicker mouse unavailable ({}). Left click / right click disabled. "
                "If this is Permission denied, run `sudo usermod -aG input $USER` and re-login.",
                exc,
            )
            self._idle()
            return

        logger.info("Gripper clicker mouse: {} ({})", device.name, device.path)
        try:
            device.grab()
            grabbed = True
        except OSError as exc:
            logger.warning(
                "Could not exclusive-grab {} ({}); this mouse may still move the desktop cursor.",
                device.path,
                exc,
            )

        held: set[int] = set()
        try:
            rate = time.Rate(self.freq)
            not_pub_ready = True
            while not self.exit_event.is_set():
                while True:
                    try:
                        event = device.read_one()
                    except (BlockingIOError, OSError):
                        break
                    if event is None:
                        break
                    if event.type != ecodes.EV_KEY:
                        continue
                    hid = _EVDEV_TO_HID.get(event.code)
                    if hid is None:
                        continue
                    if event.value:
                        held.add(hid)
                    else:
                        held.discard(hid)

                special_state = np.zeros((self.n_key_rollover,), dtype=int)
                for i, code in enumerate(held):
                    if i >= self.n_key_rollover:
                        break
                    special_state[i] = code
                self.ring_buffer.put(
                    {
                        "alphanumeric_state": np.zeros((self.n_key_rollover,), dtype=np.byte),
                        "special_state": special_state,
                        "timestamp": time.now(),
                    }
                )
                if not_pub_ready:
                    self.pub_ready_event.set()
                    not_pub_ready = False
                rate.precise_sleep()
        except KeyboardInterrupt:
            pass
        finally:
            if device is not None:
                if grabbed:
                    try:
                        device.ungrab()
                    except OSError:
                        pass
                device.close()

    def get_state(self, k=None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)
        return self.ring_buffer.get_last_k(k=k, out=out)

    def get_all_state(self):
        return self.ring_buffer.get_all()


def ClickerServer(mw, *args, **kwargs):
    return ServerFactory(mw, Clicker, *args, **kwargs)


def ClickerClient(mw, *args, **kwargs):
    return ClientFactory(mw, Clicker, *args, **kwargs)
