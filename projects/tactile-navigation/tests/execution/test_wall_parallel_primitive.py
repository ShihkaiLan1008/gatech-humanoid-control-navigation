"""WALL_PARALLEL_INTERMITTENT_TOUCH: the parts that need no simulator.

The loop itself is measured on hardware (docs/HW_ROBUSTNESS.md 15).  What is
checked here is the contract around it -- the tap record's arithmetic, and the
configuration invariants a tuning pass could silently break.
"""

from __future__ import annotations

import inspect
import re

import pytest

from execution.mujoco_backend import MujocoBackend, MujocoConfig


def _tap(**kw):
    base = {"station_m": 1.0, "s_start_m": 0.5, "standoff_before_m": 0.28,
            "peak_force_n": 0.0, "contact_ticks": 0, "first_contact_s": None,
            "last_contact_s": None, "pairs": [], "y_contact_m": None,
            "s_contact_m": None, "standoff_at_contact_m": None,
            "max_standoff_during_m": 0.28, "status": "miss"}
    base.update(kw)
    return base


def test_contact_duration_counts_loaded_ticks_not_the_span():
    """A hand that brushes again while the base sways must not read as a second
    of contact: the tap is 2 ticks long, 0.8 s apart."""
    tap = _tap(contact_ticks=2, first_contact_s=10.00, last_contact_s=10.80,
               standoff_at_contact_m=0.27, t_release_s=10.82)
    MujocoBackend._close_tap_record(tap, d_base=0.275, s=1.2, t=11.0, dt=0.02)
    assert tap["contact_s"] == pytest.approx(0.04)
    assert tap["contact_span_s"] == pytest.approx(0.82)


def test_a_tap_that_never_touched_reports_zero_contact_and_no_displacement():
    tap = _tap(status="miss")
    MujocoBackend._close_tap_record(tap, d_base=0.29, s=1.4, t=12.0, dt=0.02)
    assert tap["contact_s"] == 0.0
    assert tap["contact_span_s"] == 0.0
    assert tap["outward_contact_m"] is None      # no contact to measure from
    assert tap["outward_m"] == pytest.approx(0.01)
    assert tap["forward_m"] == pytest.approx(0.9)


def test_outward_displacement_is_measured_from_first_contact():
    """The PREPOSITION..RELEASE window is ~0.5 m of ordinary lane noise wide;
    what the TAP did is the change from the moment it touched."""
    tap = _tap(contact_ticks=1, first_contact_s=9.0, last_contact_s=9.0,
               standoff_before_m=0.250, standoff_at_contact_m=0.270,
               t_release_s=9.02)
    MujocoBackend._close_tap_record(tap, d_base=0.272, s=1.0, t=9.2, dt=0.02)
    assert tap["outward_m"] == pytest.approx(0.022)      # includes the drift
    assert tap["outward_contact_m"] == pytest.approx(0.002)   # the tap's own


def test_an_unfinished_tap_still_closes():
    """The run can end mid-release; the record must still be complete."""
    tap = _tap(contact_ticks=3, first_contact_s=5.0, last_contact_s=5.04,
               standoff_at_contact_m=0.26)
    MujocoBackend._close_tap_record(tap, d_base=0.27, s=2.0, t=5.2, dt=0.02)
    for key in ("contact_s", "release_s", "standoff_after_m", "forward_m",
                "outward_m", "outward_contact_m"):
        assert key in tap


def test_the_tap_never_targets_a_force_above_the_hard_limit():
    mj = MujocoConfig()
    assert mj.wp_tap_trigger_n < mj.wp_tap_abort_n <= mj.force_limit_n == 25.0
    # the trigger sits below the 10-25 N band on purpose: the peak lands above it
    assert mj.wp_tap_trigger_n <= 10.0
    assert mj.wp_tap_contact_n < mj.wp_tap_trigger_n


def test_the_hand_geometry_stays_inside_the_trained_command_box():
    """pos_y (0.05, 0.40) and cmd_x_clamp (0.33, 0.45) for the wall task."""
    mj = MujocoConfig()
    assert 0.05 <= mj.side_y_min_m <= mj.wp_carry_y_m <= mj.side_y_max_m <= 0.40
    assert 0.33 <= mj.side_x_m <= 0.45
    assert 0.10 <= mj.probe_hand_z_m <= 0.20


def test_prepositioning_leads_the_creep_and_the_creep_leads_the_release():
    mj = MujocoConfig()
    assert mj.wp_prep_lead_m > 0.0
    assert mj.wp_tap_rate_m_s < mj.wp_tap_fast_m_s < mj.wp_prep_rate_m_s
    assert mj.wp_release_rate_m_s > mj.wp_tap_fast_m_s   # release beats approach
    assert mj.wp_release_m > 0.0


def test_the_lateral_lane_does_not_inherit_the_facing_approach_bias():
    """`standoff_bias_m` (+0.078 m) was measured arriving at a stance the robot
    FACES.  Applying it to a wall-parallel lane walks 8 cm too close."""
    mj = MujocoConfig()
    assert mj.standoff_bias_m > 0.05
    assert mj.wp_standoff_bias_m == 0.0
    src = inspect.getsource(MujocoBackend.wall_parallel_touch)
    assert "self.standoff_command_m(" not in src     # the facing-approach inverse
    assert "float(standoff_m) - mj.wp_standoff_bias_m" in src


def test_the_primitive_takes_a_segment_a_standoff_and_a_spacing():
    sig = inspect.signature(MujocoBackend.wall_parallel_touch)
    names = list(sig.parameters)
    assert names[1:4] == ["segment", "standoff_m", "spacing_m"]
    for opt in ("stations", "s_from", "s_to", "speed_scale", "max_taps"):
        assert opt in sig.parameters


def test_base_commands_are_the_lane_the_hold_or_the_declared_stance():
    """The base-command contract, as amended by owner decision (crab mode):
    every command is the lane carrot, the frozen hold, or the DECLARED press
    stance of a crab step-in -- the demo_v3 cycle reinstated.  Lateral and
    turn-press styles still never command a wall-normal move; the stance
    target exists only inside the crab station block."""
    src = inspect.getsource(MujocoBackend.wall_parallel_touch)
    targets = {m.group(1) for m in
               re.finditer(r"self\._set_goal\(\s*([A-Za-z_][\w\.]*)", src)}
    assert targets <= {"target_w", "hold_xy", "hold_pt", "stance_w",
                       "stand_hold_w"}, targets      # = stand-to-settle hold
    # the stance is a named lane point at the declared press standoff, not an
    # arbitrary offset (its arc may sit veer-feedforward behind the station)
    assert "seg.lane_point(stance_s, stance_off)" in src
    # and the lane carrot carries the cross-track term, nothing else
    assert "seg.lane_point(carrot, cmd_off + cross)" in src
