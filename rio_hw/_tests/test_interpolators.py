"""Pure action interpolation tests, including the rotation-vector branch cut."""

from typing import cast

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from rio_hw.interpolators import ActionInterpolator


def _take_actions(interpolator, count):
    actions = []
    for _ in range(count):
        action = interpolator.get()
        assert action is not None
        actions.append(action)
    return np.stack(actions)


@pytest.mark.parametrize("axis", [[1.0, 0.0, 0.0], [1.0, 2.0, -0.5]])
def test_cartesian_actions_take_short_rotation_across_pi(axis):
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    previous = np.zeros(7)
    previous[3:6] = axis * (np.pi - 0.01)
    target = np.ones(7)
    target[3:6] = axis * (-np.pi + 0.01)
    interpolator = ActionInterpolator(multiplier=5, rotvec=True)
    interpolator.reset(previous)
    interpolator.add(target)
    actions = _take_actions(interpolator, 5)
    angles = cast(Rotation, Rotation.from_rotvec(previous[3:6]).inv() * Rotation.from_rotvec(actions[:, 3:6])).magnitude()
    np.testing.assert_allclose(angles, np.arange(1, 6) * 0.004, atol=2e-6)
    np.testing.assert_allclose(actions[:, 0], np.arange(1, 6) / 5, atol=1e-6)
    np.testing.assert_allclose(actions[:, -1], np.arange(1, 6) / 5, atol=1e-6)
    assert interpolator.needs_new_action()
    assert interpolator.get() is None


def test_joint_actions_do_not_fold_rotation_slots():
    previous = np.full(8, 4.0)
    target = np.full(8, 5.0)
    interpolator = ActionInterpolator(multiplier=5)
    interpolator.reset(previous)
    interpolator.add(target)
    actions = _take_actions(interpolator, 5)
    np.testing.assert_allclose(actions[:, 3], np.linspace(4.2, 5.0, 5), atol=1e-6)


def test_first_action_and_reset_do_not_reuse_buffered_commands():
    interpolator = ActionInterpolator(multiplier=5, rotvec=True)
    assert not interpolator.has_previous
    interpolator.add(np.zeros(7))
    assert interpolator.has_previous
    np.testing.assert_array_equal(interpolator.get(), np.zeros(7))
    assert interpolator.needs_new_action()
    interpolator.add(np.ones(7))
    interpolator.reset()
    assert interpolator.get() is None
    assert not interpolator.has_previous


@pytest.mark.parametrize("multiplier", [0, -1, 1.5])
def test_invalid_interpolation_multiplier_is_rejected(multiplier):
    with pytest.raises(ValueError, match="positive integer"):
        ActionInterpolator(multiplier=multiplier)


@pytest.mark.parametrize("action", [np.full(7, np.nan), np.zeros((2, 7)), np.zeros(5)])
def test_invalid_cartesian_action_is_rejected(action):
    with pytest.raises(ValueError):
        ActionInterpolator(rotvec=True).add(action)


def test_action_shape_cannot_change_mid_interpolation():
    interpolator = ActionInterpolator(multiplier=5)
    interpolator.add(np.zeros(7))
    with pytest.raises(ValueError, match="shape changed"):
        interpolator.add(np.zeros(8))


def test_returned_action_cannot_modify_the_interpolation_anchor():
    interpolator = ActionInterpolator(multiplier=2)
    interpolator.add(np.zeros(7))
    first = interpolator.get()
    assert first is not None
    first[:] = 100.0
    interpolator.add(np.ones(7))
    action = interpolator.get()
    assert action is not None
    np.testing.assert_allclose(action, np.full(7, 0.5))


def test_binary_gripper_commands_are_not_interpolated_through_opening():
    interpolator = ActionInterpolator(multiplier=5, rotvec=True, discrete_indices=(-1,))
    previous = np.zeros(7)
    previous[-1] = 0.7
    target = np.zeros(7)
    target[0] = 0.01
    interpolator.reset(previous)
    interpolator.add(target)
    actions = _take_actions(interpolator, 5)
    np.testing.assert_array_equal(actions[:, -1], 0.0)
    np.testing.assert_allclose(actions[:, 0], np.arange(1, 6) * 0.002, atol=1e-6)
