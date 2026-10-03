"""The free touch is the DEPLOYED touch with the wall taken out of it.

Pinned as source guards against `wall_parallel_touch`, the same style
`tests/execution/test_vforce_backend_contract.py` uses: if the deployed press
changes its intent, its rate, its debounce or its back-off, a collector that
kept the old ones would be logging a different controller than the demo runs.
"""

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PROBE = (ROOT / "src" / "data_collect" / "probe.py").read_text()
VF = (ROOT / "src" / "execution" / "vforce_backend.py").read_text()


def _src_of(src: str, name: str) -> str:
    i = src.index(f"def {name}")
    j = src.find("\n    def ", i + 1)
    return src[i:j if j > 0 else len(src)]


def test_the_press_writes_the_deployed_contact_intent():
    """`touch_primitive.py`'s `intent_value = 5.0`, the floor of the trained
    force_range -- taken from the config, never restated as a literal."""
    touch = _src_of(PROBE, "touch")
    assert "contact_mask[:] = 1.0" in touch
    assert "float(mj.vf_press_intent_n)" in touch
    deployed = _src_of(VF, "wall_parallel_touch")
    assert "contact_mask[:] = 1.0" in deployed
    assert "float(mj.vf_press_intent_n)" in deployed


def test_the_creep_is_rate_limited_and_stops_on_contact():
    """The EE channel JUMPS by design; a jump onto a surface is a lunge that
    stiff position tracking turns into a 240-435 N impact."""
    touch = _src_of(PROBE, "touch")
    assert "self.rate * be.dt" in touch
    assert "if not contacted and u < L:" in touch


def test_the_creep_rate_defaults_to_the_controllers_own():
    src = _src_of(PROBE, "__init__")
    assert "backend.mj.vf_press_rate_m_s" in src


def test_contact_latches_on_the_debounced_sensed_force():
    """One sample over 2 N on a 1 N-noise channel happens 2.3 % of the time
    in open air, so the latch needs `contact_debounce_samples` of them --
    the same rule the deployed press uses."""
    touch = _src_of(PROBE, "touch")
    assert "be._probe_force_n()" in touch
    assert "mj.wp_tap_contact_n" in touch
    assert "mj.contact_debounce_samples" in touch


def test_the_tip_comes_off_the_surface_the_moment_it_latches():
    touch = _src_of(PROBE, "touch")
    assert "backed_off = True" in touch
    after = touch[touch.index("backed_off = True"):]
    assert "mj.vf_contact_backoff_m" in after[:300]


def test_the_guard_is_checked_after_every_step():
    """`_guard_tripped` runs the safety stop and returns True; every blocking
    loop in this stack bails on it so the reaction latency is one tick."""
    loop = _src_of(PROBE, "touch")
    loop = loop[loop.index("while True:"):]
    assert loop.index("be._step()") < loop.index("be._guard_tripped()")
    assert 'status = "guard"' in loop
    # ...and a trip pending on ENTRY is consumed before anything is commanded
    head = _src_of(PROBE, "touch")
    head = head[:head.index("while True:")]
    assert "be.force_abort_pending" in head and "be._guard_tripped()" in head


def test_a_guard_trip_is_an_invalid_attempt_and_not_a_no_contact():
    """Section 4 lists safety abort, IK failure, controller failure and
    invalid target as things that must NOT be labelled no-contact."""
    touch = _src_of(PROBE, "touch")
    assert '"safety_abort"' in touch
    assert '"controller_timeout"' in touch
    assert '"target_outside_workspace"' in touch
    tail = touch[touch.index('res.label = "invalid_attempt"'):]
    assert "guard" in tail and "press_timeout" in tail
    # ...and no_contact is reached ONLY by arriving at the target
    body = touch[touch.index('elif status == "target_reached"'):]
    assert 'res.label = "no_contact"' in body[:120]


def test_a_no_contact_keeps_the_contact_fields_empty():
    """Section 5: "keep the contact-specific fields empty rather than
    inventing values"."""
    src = PROBE
    i = src.index("class TouchResult")
    j = src.index("class FreeTouchProbe")
    fields = src[i:j]
    for f in ("contact_force_n", "contact_point_body", "contact_normal_body",
              "contact_time_s", "time_to_contact_s"):
        assert f"{f}: " in fields and "None" in fields.split(f)[1][:60], f


def test_the_press_timeout_is_sized_by_the_travel_it_has_to_cover():
    """A sweep that finds nothing must be ended by the absence of a surface,
    not by the stopwatch -- and a free target can sit 0.5 m from the carry,
    further than any wall press ever creeps."""
    touch = _src_of(PROBE, "touch")
    assert "L / max(self.rate, 1e-6)" in touch


def test_the_base_is_placed_by_moving_the_room_not_the_robot():
    """The arithmetic moved to `MujocoBackend.place_map_frame`, which is where
    it belongs -- the backend owns the frame, and a mid-route start needs the
    same primitive the collector does.  The contract is unchanged and is
    asserted at its new home."""
    backend = (ROOT / "src" / "execution" / "mujoco_backend.py").read_text()
    place = _src_of(backend, "place_map_frame")
    for field in ("self.yaw0", "self.v, self.u", "self.base0"):
        assert field in place, field
    assert "self._install_world()" in place
    assert "self.reset_estimator_to(" in place
    assert "_hold_bias" in place, "a stale frozen hold pose survives the move"
    # nothing writes a root state, at either layer
    for leak in ("write_root", "root_state", "qpos["):
        assert leak not in place, leak
        assert leak not in _src_of(PROBE, "place_base"), leak
    # ...and the probe still goes through it rather than round it
    assert "place_map_frame(" in _src_of(PROBE, "place_base")


def test_a_pose_is_rejected_before_the_walls_are_placed():
    """Section 1: reject initial poses that produce wall penetration.  The
    check has to run BEFORE `_install_world`, or the rejection happens by
    dropping a wall on the robot."""
    src = ast.parse(PROBE)
    cls = next(n for n in ast.walk(src) if isinstance(n, ast.ClassDef)
               and n.name == "FreeTouchProbe")
    order = [f.name for f in cls.body if isinstance(f, ast.FunctionDef)]
    assert order.index("validate_pose") < order.index("place_base")
    v = _src_of(PROBE, "validate_pose")
    assert "_install_world" not in v
    assert "PARKED_HAND_MARGIN_M" in v and "base_radius_m" in v


def test_the_carried_hand_margin_is_the_deployed_lanes_own():
    from execution.vforce_backend import VForceConfig

    from data_collect.probe import PARKED_HAND_MARGIN_M
    mj = VForceConfig()
    assert PARKED_HAND_MARGIN_M == pytest.approx(0.045, abs=1e-9)
    assert mj.vf_press_stance_m - mj.park_x_m >= PARKED_HAND_MARGIN_M


def test_the_contact_pair_sensors_are_looked_up_by_this_familys_own_name():
    """`_inject_force_family` names them `..._vfobst{k}_...`; the inherited
    lookup used the V2 infix `v2obst`.

    A wrong infix does not raise -- the scan breaks at `k = 0` and reports
    "nothing is touching", which is indistinguishable from a real miss.
    Measured before the fix: every recorded contact came back with an empty
    pair list while the hand was reading 33 N, so no transition could name
    the body it had touched."""
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from execution.mujoco_backend import MujocoBackend
    from execution.vforce_backend import VForceBackend

    assert MujocoBackend.OBST_SENSOR_TAG == "v2obst"
    assert VForceBackend.OBST_SENSOR_TAG == "vfobst"
    assert 'f"{prefix}_vfobst{o.k}_contact"' in VF, (
        "the injector's naming moved -- the lookup tag has to follow it")
    look = _src_of(
        (ROOT / "src" / "execution" / "mujoco_backend.py").read_text(),
        "hand_obstacle_contacts")
    assert "self.OBST_SENSOR_TAG" in look and "v2obst" not in look


def test_the_arms_reach_the_carry_before_the_first_press():
    """The one reset leaves the arms wherever the policy starts them.  The
    demo hides that by walking twenty seconds to its first station; a
    collector presses immediately, and the first presses of the first episode
    measured 20-33 N at a time-to-contact of 0.38-0.58 s -- the tip swinging
    forward past its command, not the creep reaching a wall."""
    warm = _src_of(PROBE, "warm_up")
    assert "be.park_hands()" in warm and "be._step()" in warm
    assert "be._guard_tripped()" in warm
    col = (ROOT / "src" / "data_collect" / "collector.py").read_text()
    run = col[col.index("def run(self)"):col.index("def _episode(self)")]
    assert "self.probe.warm_up()" in run
    assert run.index("warm_up") < run.index("_episode()")


def test_a_miss_is_only_declared_after_the_tip_has_held_at_the_target():
    """Section 4's valid no-contact is "the controller successfully executed
    the requested reach but no environment contact occurred".  The COMMAND
    arriving is not the reach completing -- the vforce wrist tracks its
    command with a steady-state offset measured at up to 0.03 m -- so a miss
    declared on the arrival tick can be a touch that was still closing."""
    from data_collect.probe import TARGET_DWELL_S
    assert TARGET_DWELL_S > 0.0
    touch = _src_of(PROBE, "touch")
    assert "t_arrive" in touch and "TARGET_DWELL_S" in touch
    body = touch[touch.index("elif u >= L - 1e-9:"):]
    assert 'status = "target_reached"' in body[:300]
    assert body.index("t_arrive") < body.index('status = "target_reached"')
