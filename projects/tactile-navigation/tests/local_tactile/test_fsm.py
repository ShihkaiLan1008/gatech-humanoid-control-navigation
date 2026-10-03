import pytest

from local_tactile.fsm import (
    CONTACT_MODES,
    MOVING_MODES,
    OVERLAP_MODES,
    ExclusionAudit,
    Mode,
    StabilityGate,
    allows_hand_contact,
    allows_locomotion_command,
    is_overlap_mode,
    modes_are_mutually_exclusive,
)


def test_no_mode_may_both_walk_and_touch_except_the_declared_overlap():
    assert modes_are_mutually_exclusive()
    assert (MOVING_MODES & CONTACT_MODES) == OVERLAP_MODES
    for mode in Mode:
        if mode in OVERLAP_MODES:
            continue
        assert not (allows_locomotion_command(mode) and allows_hand_contact(mode))


def test_the_overlap_is_exactly_touch_walk():
    assert OVERLAP_MODES == {Mode.TOUCH_WALK}
    assert is_overlap_mode(Mode.TOUCH_WALK)
    assert allows_locomotion_command(Mode.TOUCH_WALK)
    assert allows_hand_contact(Mode.TOUCH_WALK)
    assert not any(is_overlap_mode(m) for m in Mode if m is not Mode.TOUCH_WALK)


def test_only_probe_or_touch_walk_may_touch_and_only_moving_modes_may_walk():
    assert allows_hand_contact(Mode.PROBE)
    assert not any(allows_hand_contact(m) for m in Mode
                   if m not in (Mode.PROBE, Mode.TOUCH_WALK))
    assert allows_locomotion_command(Mode.LOCOMOTION)
    assert allows_locomotion_command(Mode.TURN)
    assert not allows_locomotion_command(Mode.STABILIZE)
    assert not allows_locomotion_command(Mode.PROBE)


def test_overlap_mode_forces_land_in_their_own_bucket():
    audit = ExclusionAudit()
    audit.observe(time_s=0.0, mode=Mode.TOUCH_WALK, commanded_twist_norm=0.3,
                  hand_force_n=14.0, commanded_contact=0.0)
    assert audit.clean                       # the overlap is legal, not a violation
    assert audit.max_hand_force_in_overlap_n == pytest.approx(14.0)
    assert audit.max_twist_in_overlap == pytest.approx(0.3)
    assert audit.overlap_samples == 1
    # and it does NOT leak into the exclusion counters those reports quote
    assert audit.max_hand_force_while_moving_n == 0.0
    assert audit.max_commanded_twist_while_parked == 0.0
    assert audit.peak_hand_force_n == pytest.approx(14.0)


def _steady(gate: StabilityGate, *, start: float, duration: float, dt: float = 0.02, **kwargs):
    out = False
    steps = int(round(duration / dt))
    for i in range(steps + 1):
        out = gate.update(now_s=start + i * dt, **kwargs)
    return out


def test_gate_requires_both_feet_low_speed_and_a_continuous_hold():
    gate = StabilityGate(max_speed_m_s=0.06, max_yaw_rate_rad_s=0.20,
                         max_yaw_error_rad=0.25, hold_s=0.60)
    calm = dict(speed_m_s=0.01, yaw_rate_rad_s=0.02, yaw_error_rad=0.05, feet_in_contact=2)

    assert not gate.update(now_s=0.0, **calm)  # hold not yet elapsed
    assert not _steady(gate, start=0.0, duration=0.40, **calm)
    assert _steady(gate, start=0.0, duration=0.70, **calm)

    gate.reset()
    assert not gate.update(now_s=0.0, **{**calm, "feet_in_contact": 1})
    assert gate.last_reason == "feet_not_both_loaded"
    assert not gate.update(now_s=0.0, **{**calm, "speed_m_s": 0.2})
    assert gate.last_reason == "base_speed_above_tolerance"
    assert not gate.update(now_s=0.0, **{**calm, "yaw_rate_rad_s": 0.9})
    assert gate.last_reason == "base_yaw_rate_above_tolerance"
    assert not gate.update(now_s=0.0, **{**calm, "yaw_error_rad": 0.9})
    assert gate.last_reason == "base_yaw_error_above_tolerance"


def test_a_disturbance_restarts_the_hold_clock():
    gate = StabilityGate(hold_s=0.60)
    calm = dict(speed_m_s=0.01, yaw_rate_rad_s=0.02, yaw_error_rad=0.05, feet_in_contact=2)
    _steady(gate, start=0.0, duration=0.50, **calm)
    gate.update(now_s=0.52, **{**calm, "feet_in_contact": 1})  # one foot lifts
    assert gate.held_for_s(0.54) == pytest.approx(0.0)
    assert not gate.update(now_s=0.54, **calm)
    assert not _steady(gate, start=0.54, duration=0.40, **calm)
    assert _steady(gate, start=0.54, duration=0.70, **calm)


def test_audit_flags_a_twist_command_issued_while_parked():
    audit = ExclusionAudit()
    audit.observe(time_s=1.0, mode=Mode.PROBE, commanded_twist_norm=0.4,
                  hand_force_n=8.0, commanded_contact=1.0)
    assert not audit.clean
    assert audit.violations[0]["reason"] == "locomotion_command_issued_while_parked"
    assert audit.max_commanded_twist_while_parked == pytest.approx(0.4)


def test_audit_absorbs_the_slew_drain_but_not_a_lasting_command():
    """Entering a parked mode leaves the previous leg's twist draining through
    the command term's slew filter; only a command that outlasts the grace is a
    violation."""
    audit = ExclusionAudit(twist_settle_grace_s=0.40)
    audit.observe(time_s=1.00, mode=Mode.STABILIZE, commanded_twist_norm=0.13,
                  hand_force_n=0.0, commanded_contact=0.0, time_in_mode_s=0.02)
    audit.observe(time_s=1.10, mode=Mode.STABILIZE, commanded_twist_norm=0.0,
                  hand_force_n=0.0, commanded_contact=0.0, time_in_mode_s=0.12)
    assert audit.clean
    assert audit.max_commanded_twist_while_parked == pytest.approx(0.13)
    assert audit.max_commanded_twist_after_settle == pytest.approx(0.0)
    audit.observe(time_s=2.00, mode=Mode.STABILIZE, commanded_twist_norm=0.09,
                  hand_force_n=0.0, commanded_contact=0.0, time_in_mode_s=1.02)
    assert not audit.clean
    assert audit.max_commanded_twist_after_settle == pytest.approx(0.09)


def test_audit_flags_hand_contact_outside_the_contact_mode():
    audit = ExclusionAudit(contact_force_tolerance_n=1.0)
    audit.observe(time_s=2.0, mode=Mode.LOCOMOTION, commanded_twist_norm=0.6,
                  hand_force_n=6.5, commanded_contact=0.0)
    reasons = {row["reason"] for row in audit.violations}
    assert reasons == {"hand_contact_force_outside_contact_mode"}
    audit.observe(time_s=2.5, mode=Mode.TURN, commanded_twist_norm=0.3,
                  hand_force_n=0.1, commanded_contact=1.0)
    reasons = {row["reason"] for row in audit.violations}
    assert "hand_contact_commanded_outside_contact_mode" in reasons


def test_a_correct_episode_leaves_the_audit_clean():
    audit = ExclusionAudit()
    audit.observe(time_s=0.0, mode=Mode.LOCOMOTION, commanded_twist_norm=0.55,
                  hand_force_n=0.0, commanded_contact=0.0)
    audit.observe(time_s=1.0, mode=Mode.STABILIZE, commanded_twist_norm=0.0,
                  hand_force_n=0.0, commanded_contact=0.0, time_in_mode_s=2.0)
    audit.observe(time_s=2.0, mode=Mode.PROBE, commanded_twist_norm=0.0,
                  hand_force_n=9.4, commanded_contact=1.0)
    audit.observe(time_s=3.0, mode=Mode.TURN, commanded_twist_norm=0.4,
                  hand_force_n=0.02, commanded_contact=0.0)
    assert audit.clean
    assert audit.to_dict()["max_hand_force_while_moving_n"] == pytest.approx(0.02)


def test_repeated_violations_collapse_to_one_row_with_the_peak():
    audit = ExclusionAudit()
    for index, force in enumerate((2.0, 9.0, 4.0)):
        audit.observe(time_s=float(index), mode=Mode.LOCOMOTION, commanded_twist_norm=0.5,
                      hand_force_n=force, commanded_contact=0.0)
    assert len(audit.violations) == 1
    assert audit.violations[0]["peak_value"] == pytest.approx(9.0)
    assert audit.violations[0]["samples"] == 3
    assert audit.violations[0]["first_time_s"] == pytest.approx(0.0)
