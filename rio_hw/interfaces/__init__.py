"""Teleoperation interfaces, loaded on first use.

Each interface module is imported only when one of its names is accessed, e.g.
``rio_hw.interfaces.SshKeyboardClient``. Importing every module up front pulled
in all optional drivers, and some of them change process-wide state at import
time (``avp_stream`` installs its own SIGINT/SIGTERM handlers), so a station
that never uses them was still affected.
"""

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .avp_stream import AvpStreamClient, AvpStreamServer
    from .clicker import ClickerClient, ClickerServer
    from .gamepad import GamepadClient, GamepadServer
    from .gello import GelloClient, GelloServer
    from .joycon import JoyconClient, JoyconServer
    from .keyboard import KeyboardClient, KeyboardServer
    from .manus_glove import ManusGloveClient, ManusGloveServer
    from .oculus_reader import OculusReaderClient, OculusReaderServer
    from .spacemouse import SpacemouseClient, SpacemouseServer
    from .ssh_keyboard import SshKeyboardClient, SshKeyboardServer
    from .vive_tracker import ViveTrackerClient, ViveTrackerServer
    from .vuer import VuerClient, VuerServer
    from .x_robotoolkit import XRobotoolkitClient, XRobotoolkitServer

# Interface name -> submodule that defines ``<name>Client`` and ``<name>Server``.
_MODULES = {
    "AvpStream": "avp_stream",
    "Clicker": "clicker",
    "Gamepad": "gamepad",
    "Gello": "gello",
    "Joycon": "joycon",
    "Keyboard": "keyboard",
    "ManusGlove": "manus_glove",
    "OculusReader": "oculus_reader",
    "Spacemouse": "spacemouse",
    "SshKeyboard": "ssh_keyboard",
    "ViveTracker": "vive_tracker",
    "Vuer": "vuer",
    "XRobotoolkit": "x_robotoolkit",
}

__all__ = list(_MODULES)


def __getattr__(name: str) -> Any:
    """Import the defining submodule on first access to ``<Interface>Client`` or ``<Interface>Server``."""
    for suffix in ("Client", "Server"):
        interface = name.removesuffix(suffix)
        if interface != name and interface in _MODULES:
            value = getattr(import_module(f".{_MODULES[interface]}", __name__), name)
            globals()[name] = value
            return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted([*globals(), *(f"{name}{suffix}" for name in _MODULES for suffix in ("Client", "Server"))])
