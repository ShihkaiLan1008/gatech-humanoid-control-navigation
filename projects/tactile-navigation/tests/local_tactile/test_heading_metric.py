import math

import pytest

from local_tactile.pipeline import leg_heading_change_rad


def row(mode, yaw):
    return {"mode": mode, "base_pose": [0.0, 0.0, yaw]}


def test_measures_between_the_parked_runs_that_bracket_the_turn():
    trace = (
        [row("LOCOMOTION", 0.02)] * 3
        + [row("STABILIZE", 0.01), row("PROBE", -0.01), row("STABILIZE", 0.00)]
        + [row("TURN", 0.4), row("TURN", 1.0)]
        + [row("LOCOMOTION", 1.4)] * 3
        + [row("STABILIZE", math.pi / 2 + 0.01), row("ARRIVED", math.pi / 2 - 0.01)]
    )
    assert leg_heading_change_rad(trace) == pytest.approx(math.pi / 2, abs=1e-3)


def test_ignores_the_moving_legs_between_the_two_parked_runs():
    """Heading excursions while walking must not enter the measurement."""
    trace = (
        [row("STABILIZE", 0.0), row("STABILIZE", 0.0)]
        + [row("TURN", -0.5)]
        + [row("LOCOMOTION", -3.0)] * 5  # wild heading while walking
        + [row("STABILIZE", -math.pi / 2), row("ARRIVED", -math.pi / 2)]
    )
    assert leg_heading_change_rad(trace) == pytest.approx(-math.pi / 2, abs=1e-6)


def test_wraps_across_pi():
    trace = (
        [row("STABILIZE", 3.0)]
        + [row("TURN", 3.1)]
        + [row("STABILIZE", -3.0)]
    )
    change = leg_heading_change_rad(trace)
    assert change == pytest.approx(2 * math.pi - 6.0, abs=1e-6)
    assert -math.pi <= change <= math.pi


def test_returns_none_without_a_turn():
    assert leg_heading_change_rad([row("LOCOMOTION", 0.0), row("STABILIZE", 0.0)]) is None


def test_returns_none_when_a_bracketing_parked_run_is_missing():
    trace = [row("LOCOMOTION", 0.0), row("TURN", 0.5), row("STABILIZE", 1.5)]
    assert leg_heading_change_rad(trace) is None
