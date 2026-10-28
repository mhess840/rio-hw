"""Clicker mouse mapping, without opening hardware."""

from rio_hw.interfaces.clicker import _EVDEV_TO_HID, find_clicker


def test_left_click_maps_to_page_down_close():
    assert _EVDEV_TO_HID[0x110] == 0x4E


def test_right_click_maps_to_page_up_open():
    assert _EVDEV_TO_HID[0x111] == 0x4B


def test_side_and_back_map_to_scale_down():
    assert _EVDEV_TO_HID[0x113] == 0x224
    assert _EVDEV_TO_HID[0x116] == 0x224


def test_extra_and_forward_map_to_scale_up():
    assert _EVDEV_TO_HID[0x114] == 0x225
    assert _EVDEV_TO_HID[0x115] == 0x225


def test_find_clicker_uses_explicit_path():
    assert find_clicker("/dev/input/event29") == "/dev/input/event29"
