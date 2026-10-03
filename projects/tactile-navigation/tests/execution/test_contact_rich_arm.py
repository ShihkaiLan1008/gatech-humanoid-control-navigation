"""The contact-rich arm cycle: the invariants a tuning pass could break.

The loop runs only in MuJoCo (docs/HW_ROBUSTNESS.md 16), so what is checked here
is that the phases exist, that the base is not part of any of them, and that the
configuration keeps the properties those measurements bought.
"""

from __future__ import annotations

import inspect
import re

import pytest

from execution.mujoco_backend import MujocoBackend, MujocoConfig

SRC = inspect.getsource(MujocoBackend.wall_parallel_touch)


def _set_goal_targets(src: str) -> list[str]:
    """First argument of every `_set_goal` call in the primitive."""
    return [m.group(1) for m in re.finditer(r"self\._set_goal\(\s*([A-Za-z_][\w\.]*)", src)]


def test_the_arm_cycle_has_a_carry_a_reach_a_touch_and_a_retract():
    for phase in ("FREE", "REACH", "HOLD", "PREPOSITION", "TAP", "RELEASE",
                  "RETRACT"):
        assert f'"{phase}"' in SRC, phase


def test_every_base_command_is_the_lane_a_hold_or_the_declared_stance():
    """The base-command contract, amended once by owner decision: the crab
    style's station steps to a DECLARED press stance and back (the demo_v3
    cycle).  Lateral taps and turn-press keep the original invariant -- their
    paths never target the stance."""
    targets = set(_set_goal_targets(SRC))
    assert targets <= {"target_w", "hold_xy", "hold_pt", "stance_w",
                       "stand_hold_w"}, targets      # = stand-to-settle hold
    lane = SRC.index("self._set_goal(")
    arm = SRC.index("# ── hand:")
    assert lane < arm, "the lane command must be issued before the arm machine"
    # and the only thing that varies it is the cross-track term on the lane
    assert "seg.lane_point(carrot, cmd_off + cross)" in SRC
    for phase in ("REACH", "TAP", "RELEASE", "RETRACT"):
        seg = SRC.split(f'elif state == "{phase}":')[1][:900]
        assert "_set_goal" not in seg, f"{phase} moves the base"


def test_the_gesture_is_large_on_the_axes_that_cannot_touch_the_wall():
    """HW_ROBUSTNESS 17: demo_v3's readability was shoulder pitch (69-80 deg) and
    elbow (119-131), which come from FORWARD and VERTICAL hand motion; the
    lateral axis is the one that makes contact and stays small."""
    mj = MujocoConfig()
    gesture_x = abs(mj.wp_touch_x_m - mj.wp_carry_x_m)
    gesture_z = abs(mj.probe_hand_z_m - mj.wp_carry_z_m)
    lateral = mj.wp_reach_y_m - mj.wp_carry_y_m
    assert gesture_x >= 0.15, "the forward swing is what extends the elbow"
    assert gesture_z >= 0.08, "the vertical swing is what pitches the shoulder"
    assert lateral < gesture_x, "the wall-normal axis must not carry the gesture"


def test_the_touch_is_taken_from_the_compliant_arm_pose():
    """A near-straight arm doubles the contact force for the same penetration:
    x = 0.45 gave 28-38 N where x = 0.34 gives 10-24 N."""
    mj = MujocoConfig()
    assert mj.wp_touch_x_m == pytest.approx(mj.side_x_m)
    assert mj.wp_touch_x_m <= 0.40
    # and it is held from the reach right through the unload
    for phase in ("REACH", "HOLD", "PREPOSITION", "TAP", "RELEASE"):
        # the POSE block, which precedes the transition machine
        seg = SRC.split(f'elif state == "{phase}":')[1][:400]
        assert "mj.wp_touch_x_m" in seg, phase


def test_the_carry_is_a_tucked_pose_the_touch_pose_is_not():
    mj = MujocoConfig()
    assert mj.wp_carry_x_m < mj.wp_touch_x_m
    assert mj.wp_carry_z_m > mj.probe_hand_z_m
    assert mj.side_y_min_m <= mj.wp_carry_y_m <= 0.12
    assert mj.wp_reach_y_m > mj.wp_carry_y_m + 0.03


def test_the_visible_phases_are_slower_than_the_unload_and_faster_than_the_creep():
    mj = MujocoConfig()
    assert mj.wp_tap_rate_m_s < mj.wp_tap_fast_m_s < mj.wp_reach_rate_m_s
    assert mj.wp_retract_rate_m_s < mj.wp_release_rate_m_s
    # the reach is staged: swing out, hold, then close the gap
    assert mj.wp_gap_lead_m < mj.wp_prep_lead_m


def test_the_gait_phase_hold_never_moves_the_hand_backwards():
    """It is a HOLD, not a retreat: rate 0 at the measured pose."""
    assert "if v_n <= -mj.wp_tap_hold_rate_m_s:" in SRC
    held = SRC.split("if v_n <= -mj.wp_tap_hold_rate_m_s:")[1][:200]
    assert "y_want, rate = y_meas, 0.0" in held
    assert MujocoConfig().wp_tap_hold_rate_m_s > 0.0


def test_the_approach_reference_tracks_the_measured_hand():
    """Referencing the COMMAND lets it run ahead by the arm's lag, and the arm
    then clears that debt at whatever speed it likes -- measured 0.52 m/s of gap
    closure and a 41.5 N contact."""
    assert "y_meas = abs(self._ee_base_y(term))" in SRC
    for phase, expr in (("REACH", "y_meas + max(d_ee, 0.0)"),
                        ("PREPOSITION", "y_meas + (d_ee - mj.wp_prep_gap_m)"),
                        ("TAP", "y_meas + max(d_ee, 0.0)")):
        assert expr in SRC, phase


def test_the_unload_ends_on_a_measured_gap_not_a_command_value():
    assert "d_ee >= mj.wp_prep_gap_m + mj.wp_release_m" in SRC
    mj = MujocoConfig()
    assert mj.wp_release_m > 0.0


def test_safety_survives_the_visible_cycle():
    mj = MujocoConfig()
    assert mj.wp_tap_abort_n == 25.0 == mj.force_limit_n
    assert mj.wp_tap_trigger_n < mj.wp_tap_abort_n
    assert "over = f > mj.wp_tap_abort_n" in SRC
    # the abort is armed in every phase where the hand can be near the wall
    assert 'state in ("REACH", "HOLD", "PREPOSITION", "TAP")' in SRC
    # and the 200 Hz sample is what the trigger reads
    assert "self.shield.state(term).peak_total_n" in SRC
