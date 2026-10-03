"""The vforce backend's executor contract, testable without MuJoCo.

The backend itself needs mjlab + the controller repo; what is pinned here is
the part the EXECUTOR depends on -- the tap vocabulary, the trained-band
constants, and the run scripts' controller wiring -- as source guards, the
same style `test_scene_demo` uses for the corner gates.
"""

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC = (ROOT / "src" / "execution" / "vforce_backend.py").read_text()


def test_the_carry_is_the_deployed_one_verbatim():
    """`wognl0402/locomanip_deploy` @ `tactile_navigation` is the package that
    ran this policy on room2, and its
    `g1_decoupled_velocity_override.yaml -> command_vector_default` holds both
    hands at (0.3323, +-0.1516, 0.0952) with an identity quaternion while
    walking.  That is the carry, copied, not derived.

    It is also the posture asked for: FK on the deploy's `IDLE_ANGLES` puts
    the upper arm 19.2 deg off the torso's down axis with the elbow bent
    80.9 deg, forearm forward, elbow 0.052 m IN FRONT of the shoulder.

    Three carries were shipped here before it, and the trap is that
    `park_x/y/z` is SHIELD space in `MujocoBackend` and an ABSOLUTE tip pose
    here -- so inheriting the base class's numbers commands a different arm
    (measured: 52.3 deg off the torso, elbow 0.117 m BEHIND the shoulder)."""
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from execution.mujoco_backend import MujocoConfig

    ns = _config_literals()
    assert (ns["park_x_m"], ns["park_y_m"], ns["park_z_m"]) == \
        (0.3323, 0.1516, 0.0952)

    c = MujocoConfig()
    assert (ns["park_x_m"], ns["park_y_m"], ns["park_z_m"]) != \
        (c.park_x_m, c.park_y_m, c.park_z_m), (
            "the base class's park is shield space -- copying it commands a "
            "different arm, which is the bug this pins")

    # identity orientation, as the deployed command vector has it
    assert "dyaw=0.0" in _src_of("_park_hand")


def test_the_lane_is_the_press_stance_and_it_is_the_deployed_one():
    """`locomanip_deploy`'s room2 mission: `standoff_m: 0.38`, `spacing_m: 0.2`.

    0.38 is `ACQUISITION_GAP` -- the centre of the band between the carry
    (`PARK_FORWARD` 0.33 + 0.01) and the delivered reach (0.43) -- and the
    base touches FROM the lane.  Our 0.49 lane plus a 0.36 step-in is the
    manoeuvre that kept putting the parked hand on the wall: at 0.36 it sat
    0.028 m off the surface and rode it for the whole leg."""
    import json
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from demo_map.executor import DEPLOY_STANDOFF_M
    from demo_map.deploy_mission import compile_steps, load

    ns = _config_literals()
    mission = load()
    assert DEPLOY_STANDOFF_M == mission["standoff_m"] == 0.38
    assert ns["vf_press_stance_m"] == DEPLOY_STANDOFF_M
    assert DEPLOY_STANDOFF_M - ns["park_x_m"] >= 0.045   # parked hand clear

    # NO STEP-IN, and it must reach the press: the mission's own follow steps
    # carry `press_standoff` equal to the lane.  It travels through the step,
    # not through `CRAB_PRESS_STANCE` -- that constant belongs to the demo
    # room, whose scan bulges into its lane and which needs a step-in.
    # Sharing one global put demo_room's carried hands on the wall: five
    # aborts in 30 s, `stuck:R1:out` before its first landmark.
    from demo_map.executor import DemoExecCfg
    from demo_map.scene_room2 import build_room2_scene, plan_steps_room2
    scene = build_room2_scene(False)
    cfg = DemoExecCfg(standoff_m=mission["standoff_m"],
                      press_spacing_m=mission["spacing_m"],
                      monotone_route=False)
    for st in compile_steps(mission, plan_steps_room2(cfg, False, scene=scene)):
        if st["kind"] == "follow" and st["label"] != "goal_post":
            assert st["press_standoff"] == st["standoff"] == 0.38, st["label"]


def test_the_press_uses_the_deployed_contact_intent():
    """The DEPLOYED touch never asks the policy to push.

    `tacnav_bringup/touch_state_machine.py` runs PARK -> APPROACH ->
    CONTACT_CANDIDATE -> CONTACT_CONFIRMED -> RECORD -> RETRACT and mentions
    neither `contact_mask` nor `force_cmd`; the 8 force slots stay at
    `g1_decoupled_vel_force.yaml`'s `command_vector_default` -- mask 0,
    force 0.  Raising them tells the policy to lean on a tip that is already
    creeping into the surface, which is load nothing accounted for.

    Contact still registers either way: it is a physical touch felt through
    the same SensX channel."""
    ns = _config_literals()
    # the deployed intent, not a band midpoint: touch_primitive.py declares
    # `intent_value = 5.0`, the FLOOR of the trained force_range (5, 25)
    assert ns["vf_press_intent_n"] == 5.0
    # touch_primitive's approach_speed is 0.02; we run 1.5x by OWNER RULING
    # (d4 config, 2026-08-27): the speed matrix measured 0.03 keeping every
    # contact <= 21 N with over_limit 0 -- the parity kept is the INTENT
    # (5.0), not the creep rate.
    assert ns["vf_press_rate_m_s"] == 0.03
    touch = _src_of("wall_parallel_touch")
    assert "float(mj.vf_press_intent_n)" in touch
    assert "_press_target_n()" not in touch, "14 N was ~3x the deployed intent"


def test_a_press_retracts_on_contact_instead_of_dwelling_on_it():
    """P0 "exposes only an image-confirmed probe followed by immediate
    retract"; the contact IS the measurement and every tick after it is load
    with nothing bought."""
    ns = _config_literals()
    assert ns["vf_press_hold_s"] <= 0.20              # the deploy's cell_hold_s
    assert ns["vf_contact_backoff_m"] >= 0.05         # ...and it comes OFF
    touch = _src_of("wall_parallel_touch")
    assert "backed_off = True" in touch
    body = touch[touch.index("backed_off = True"):]
    assert "x - mj.vf_contact_backoff_m" in body[:400]


def test_the_reach_is_capped_by_travel_not_only_by_belief():
    """A bound the belief cannot argue with: a press that thinks the wall is
    far away must still not be able to lean a long way into one that is
    not (the deploy's `touch_travel_cap_m`)."""
    ns = _config_literals()
    assert ns["vf_touch_travel_cap_m"] <= 0.24
    assert "x0 + mj.vf_touch_travel_cap_m" in _src_of("wall_parallel_touch")


def test_the_creep_is_given_the_seconds_its_travel_needs():
    """A sweep that finds nothing must be stopped by the WALL's absence, not
    by the stopwatch -- and the budget must be re-derived whenever the
    approach speed or the travel cap moves, which is what this catches.

    Travel is bounded by BOTH the trained band's far edge (0.48) and the
    deployed `travel_cap_m` (0.24 past the carry), whichever binds first.
    Reference: the deploy allows `approach_timeout_s = 10.0`."""
    ns = _config_literals()
    reach = min(0.48, ns["park_x_m"] + ns["vf_touch_travel_cap_m"])
    travel = reach - ns["park_x_m"]
    assert ns["vf_press_timeout_s"] >= travel / ns["vf_press_rate_m_s"]


def test_press_knobs_sit_inside_the_trained_bands():
    """force_maze trained: kiss-x in (0.30, 0.48), hand y in (0.08, 0.35),
    z in (0.10, 0.25) -- commands outside them are out-of-distribution for
    the checkpoint."""
    ns = _config_literals()
    assert 0.30 <= ns["vf_press_stance_m"] <= 0.48
    assert 0.08 <= ns["vf_hand_y_m"] <= 0.35
    assert 0.10 <= ns["vf_hand_z_m"] <= 0.25
    # acquisition margin only -- not a way to generate force
    assert 0.0 < ns["vf_press_overlap_m"] <= 0.03


def test_the_press_creeps_instead_of_jumping_onto_the_face():
    """The EE channel jumps by design, and a jump from the carry onto the
    face is a quarter-metre lunge that stiff position tracking turns into an
    impact: measured 240-435 N true transients.  Creeping the command at a
    bounded rate and stopping on the first debounced contact measured 33 N."""
    ns = _config_literals()
    assert 0.0 < ns["vf_press_rate_m_s"] <= 0.06
    touch = _src_of("wall_parallel_touch")
    assert "vf_press_rate_m_s * self.dt" in touch
    # the creep must stop once contact is felt, or it is just a slow lunge
    assert 'if not tap["contacted"] and x < x_end:' in touch


def test_the_guard_is_the_one_from_four_days_ago():
    """The force a hand may carry is `MujocoConfig`'s, unchanged: 12 N on one
    sample, and the 3-20 N window around it (contact latches at
    `contact_threshold_n` 2 N, the SensX channel saturates at 20 N).  A 15 N
    / 3.0 s version was sized against a carry that rested its parked tip on
    the wall; the tucked carry removes that load instead."""
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from execution.backends import BackendConfig
    from execution.mujoco_backend import MujocoConfig

    ns = _config_literals()
    for knob in ("guard_abort_force_n", "safety_settle_timeout_s",
                 "safety_clear_force_n"):
        assert knob not in ns, f"VForceConfig must not override {knob}"

    c = MujocoConfig()
    assert c.guard_abort_force_n == 12.0 and c.guard_abort_hold_s == 0.0
    assert BackendConfig().contact_threshold_n <= 3.0
    assert c.guard_abort_force_n <= c.force_sat_n == 20.0

    # the ONE thing overridden, and it does not raise what a hand may press
    # with: the over-force has to LAST, because every abort measured with the
    # tuck restored was `loaded_s` 0.02 -- one control tick of the arm's own
    # swing, before the base had left spawn.
    assert 0.02 < ns["guard_abort_hold_s"] <= 0.5


def _config_literals() -> dict:
    ns = {}
    for node in ast.walk(ast.parse(SRC)):
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            try:
                ns[node.target.id] = ast.literal_eval(node.value)
            except Exception:
                pass
    return ns


def _src_of(name: str) -> str:
    i = SRC.index(f"def {name}")
    j = SRC.find("\n    def ", i + 1)
    return SRC[i:j if j > 0 else len(SRC)]


def test_the_press_normal_is_the_measured_force_direction():
    """The vforce wrist tracks its command stiffly, so an encoder 'normal'
    is the commanded axis in disguise -- the documented yaw-confirmation
    spiral (measured here: belief error 0.4 -> 1.15 m along one wall).  The
    only surface measurement this family has is the contact-force vector."""
    assert "measured_force_w()[0]" in _src_of("_measured_normal_body")


def test_the_free_sweep_actually_sweeps_before_reporting_free():
    """`_tap_obs`'s free gate consumes `free_body` sample points; a free sweep
    without them is silently no evidence at all -- and that is what a
    BELIEF-LIMITED reach produces.

    Stopping the creep at `d_bel + overlap` means a short belief travels
    almost nothing and then reports FREE SPACE for a ray never travelled.
    Measured: every miss in a full room2 run carried `free_body` = 0 points,
    so "a miss where the map promised a wall is evidence" was evidence of
    nothing.  `touch_primitive.py` reaches for a FIXED `target_xy[0] = 0.43`
    and stops on contact, wherever the wall is believed to be."""
    import numpy as np

    assert '"bounded_free_sweep"' in SRC
    assert '"free_body"' in SRC

    touch = _src_of("wall_parallel_touch")
    assert "mj.vf_touch_target_m" in touch
    assert "d_bel + mj.vf_press_overlap_m" not in touch, "belief-limited reach"

    ns = _config_literals()
    assert ns["vf_touch_target_m"] == 0.43          # the delivered reach
    x0 = ns["park_x_m"]
    x_end = min(ns["vf_touch_target_m"], 0.48, x0 + ns["vf_touch_travel_cap_m"])
    assert x_end > x0, "the sweep must travel"
    assert len(np.arange(x0, x_end, 0.05)) >= 2, "and leave evidence behind"


def test_run_scripts_default_to_the_vforce_controller():
    for name in ("demo_room_run.py", "room2_run.py"):
        s = (ROOT / "scripts" / name).read_text()
        assert '"--controller"' in s, name
        assert '("vforce", "hand_force")' in s, name
        assert 'default="vforce"' in s, name


def test_the_video_replay_rebuilds_the_recorded_controller():
    s = (ROOT / "scripts" / "demo_room_video.py").read_text()
    assert 'doc.get("controller")' in s
    # runs recorded before the field existed were all hand_force
    assert '"hand_force"' in s


@pytest.mark.skipif(
    not (ROOT / "tactile_controller" / "src" / "rl").exists(),
    reason="controller repo not checked out beside src/")
def test_backend_reuses_the_honest_sensing_machinery():
    """The estimator, the guard and the frames must come from MujocoBackend
    (inherited), not re-implementations that could drift."""

    assert "class VForceBackend(MujocoBackend)" in SRC
    # goto_map IS overridden here, deliberately: same displacement
    # composition, plus the measured brush allowance its docstring carries.
    for must_not_override in ("def working_pose", "def odometry_increment",
                              "def reset_estimator_to", "def estimator_bias",
                              "def turn_to_map",
                              "def _integrate_estimator", "def _step"):
        assert must_not_override not in SRC, (
            f"{must_not_override} re-implemented -- the honest-sensing "
            "contract lives in the base class")


def test_a_completed_safety_stop_is_not_a_dead_end():
    """`_safety_stop` does not return until the robot is clear and still, so
    the walk and the settle carry on from there instead of returning
    "guard"/False on the first trip and ending the run."""
    assert "vf_guard_retries" in SRC
    for name in ("_walk_to_lane", "_settle"):
        assert "vf_guard_retries" in _src_of(name), name


def test_no_decision_in_this_backend_reads_the_simulator_s_answer():
    """Truth may be read for exactly four things, none of them a decision:
    defining the map frame at spawn (a robot knows its own start pose --
    that IS the origin), initialising and marking the drifting estimator,
    converting a MEASURED world quantity into the body frame (what the
    robot's own FK and its wrist sensor report), and eval-only diagnostics.
    Anything else is the simulator telling the planner where it is."""
    import ast

    allowed = {
        "__init__", "_reset",              # map frame + estimator init/marks
        "_tip_body", "_measured_normal_body",   # world -> body, i.e. FK
        "wall_parallel_touch",             # `true_standoff_eval_m` only
    }
    tree = ast.parse(SRC)
    for fn in ast.walk(tree):
        if not isinstance(fn, ast.FunctionDef) or fn.name in allowed:
            continue
        body = ast.get_source_segment(SRC, fn) or ""
        body = body.replace('"""', "\x00").split("\x00")
        code = "".join(body[::2]) if len(body) > 1 else body[0]
        for leak in ("truth_pose(", "map_point_to_world(", "heading_w"):
            assert leak not in code, f"{fn.name} reads {leak}"

    # ...and the one eval read is named as evaluation where it is recorded
    assert "true_standoff_eval_m" in SRC


def test_one_hand_touches_everything():
    """A pin on `set_touch_hand` binds EVERY caller -- the geometric chooser
    in the press, `_set_probe_side`, the tick path -- so a one-hand rhythm
    cannot be undone by whichever code path happens to pick next.

    The deploy's touch executor drives a single arm (one `left_ikfk_topic`,
    every cell delegated to one `touch_primitive`).  Letting the second hand
    press was mine, and it is what put both hands on the wall at once."""
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from execution.vforce_backend import VForceBackend

    b = VForceBackend.__new__(VForceBackend)
    b.hands = {"left": object(), "right": object()}

    b.pinned_probe_hand = "right"
    for requested in (+1, -1):
        b.set_touch_hand(requested)
        assert b.probe_name == "right", requested

    b.pinned_probe_hand = None                  # unpinned still works
    b.set_touch_hand(+1)
    assert b.probe_name == "left"

    # the lateral command follows the hand chosen, not the geometry that
    # suggested it -- otherwise a pinned right hand goes to the left offset
    assert ("math.copysign(mj.vf_hand_y_m, self.probe_side)"
            in _src_of("wall_parallel_touch"))


def test_the_executor_pins_the_same_hand_for_rhythm_and_landmarks():
    """A body is bracketed by moving the BASE along the face, not by swapping
    arms -- and the ratchet already allows that inside a visit."""
    src = (ROOT / "src" / "demo_map" / "executor.py").read_text()
    assert 'TOUCH_HAND = "right"' in src
    # rhythm, pillar (VisitCtx since the P4 carve pins via `ex.`), corner
    assert (src.count("self._pin_hand()")
            + src.count("ex._pin_hand()")) >= 3
    assert "._pin_hand(None)" not in src


def test_a_hand_target_carries_the_arm_posture_that_reaches_it():
    """`anchor_q` is the policy's ARM-JOINT mimic reference and it defaults to
    the home posture -- arms hanging down.  `_resample_command` normally sets
    it from the reach pool so it always matches the tip; writing the tip
    directly skips that, so the policy is told to hold the arms down AND to
    put the tip 0.33 m forward, and it fights itself.

    Measured, room2: tip 0.082 m short of the command, arm joints at
    125-149 deg/s p95 against 29-34 in the run that arrived, and 223 N read in
    STABILIZE with nothing within 0.24 m of the hand."""
    import math
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from execution.vforce_backend import CARRY_ANCHOR_Q

    jump = _src_of("_jump_hand")
    for field in ("anchor_q", "_anchor_start_q", "_anchor_goal_q"):
        assert field in jump, field

    ns = _config_literals()
    for side, sign in (("left", +1.0), ("right", -1.0)):
        q = CARRY_ANCHOR_Q[side]
        assert len(q) == 7, side
        # mirrored: roll/yaw/wrist flip sign, pitch and elbow do not
        assert math.copysign(1.0, q[1]) == sign
        assert q[3] < 0.0, "elbow bent, forearm forward"
    assert CARRY_ANCHOR_Q["left"][0] == CARRY_ANCHOR_Q["right"][0]
    assert CARRY_ANCHOR_Q["left"][3] == CARRY_ANCHOR_Q["right"][3]

    # the stubbed pool must agree with it, or a draw reintroduces the conflict
    assert "CARRY_ANCHOR_Q[side]" in SRC


def test_an_advance_is_bounded_in_cross_track_not_only_in_arrival():
    """`run_room2_deploy.advance_along` measures lateral deviation and bails:

        lateral = abs(-s * dx + c * dy)
        if progress < -0.025 or lateral > 0.08: ... return False

    because "relative odometry cannot observe cross-track drift while
    travelling along a wall".  Nothing in ours watched it, so a leg could
    drift sideways and only the next press would find out -- measured on
    room2, the base wandered 0.13 m out between two stations while the belief
    held, and every press after the corner then swept air."""
    ns = _config_literals()
    assert ns["vf_lateral_abort_m"] == 0.08          # the deploy's bound
    walk = _src_of("_walk_to_lane")
    assert "-axis[1] * dv[0] + axis[0] * dv[1]" in walk, "cross-track term"
    assert "vf_lateral_abort_m" in walk
    assert 'return "lateral"' in walk

    # ...and a leg that could not hold its lane must not then press from
    # wherever the drift left it
    touch = _src_of("wall_parallel_touch")
    assert 'if st in ("timeout", "lateral"):' in touch
    assert '"status": "not_on_lane"' in touch


def test_no_method_reads_a_bare_mj_it_never_bound():
    """Most methods here open with `mj = self.mj`; `_walk_to_lane` does not.
    Copying a `mj.<knob>` line into one of those is a NameError that only
    fires on the tick that reaches it -- measured: a run died 10.5 s in, with
    the whole route still ahead of it, on `mj.vf_lateral_retries`."""
    import ast

    tree = ast.parse(SRC)
    bad = []
    for fn in ast.walk(tree):
        if not isinstance(fn, ast.FunctionDef):
            continue
        seg = ast.get_source_segment(SRC, fn) or ""
        if "mj = " in seg or any(a.arg == "mj" for a in fn.args.args):
            continue
        for n in ast.walk(fn):
            if (isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
                    and n.value.id == "mj"):
                bad.append(f"{fn.name}:{n.lineno} mj.{n.attr}")
    assert not bad, bad


def test_the_safety_stop_gives_the_caller_its_mode_back():
    """It used to exit in STABILIZE with no caller restoring it, so a walk
    that resumed after a stop commanded twist while labelled STABILIZE.  That
    mislabelled later abort events and raised
    `locomotion_command_issued_while_parked` in the exclusion audit -- a
    finding that was an artefact of this bug, not of the walk."""
    import inspect
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from execution.mujoco_backend import MujocoBackend

    src = inspect.getsource(MujocoBackend._safety_stop)
    assert "mode_was = self.mode" in src
    assert "self._enter(mode_was)" in src
    # ...in the finally, so a timeout or an early return restores it too
    tail = src[src.index("finally:"):]
    assert "self._enter(mode_was)" in tail


def test_the_over_limit_counter_reads_a_channel_that_can_exceed_the_limit():
    """`force_limit_n` is 25 N and the SENSED channel saturates at
    `force_sat_n` 20 N, so the old test could never fire: two runs reported
    `over_limit_s = 0.0` while one peaked at 26.2 N true."""
    import inspect
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from execution.mujoco_backend import MujocoBackend, MujocoConfig

    c = MujocoConfig()
    assert c.force_limit_n > c.force_sat_n, "the premise of this bug"
    src = inspect.getsource(MujocoBackend._step)
    i = src.index("ticks_over_limit")
    assert "true_peak_force_n" in src[max(0, i - 400):i]


def test_a_goto_survives_a_completed_safety_stop():
    """`_walk_to_lane` already treats a finished stop as a place to carry on
    from; `goto_map` returned `force_abort` on the first trip.  The demo
    room's `R1:out` aborted on both of the executor's two attempts and the
    route ended `stuck:R1:out` 30 s in -- before its first landmark, with the
    belief still good at 0.10 m -- because a carried hand brushed a scanned
    bulge in passing."""
    import inspect
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from execution.mujoco_backend import MujocoBackend

    src = inspect.getsource(MujocoBackend.goto_map)
    i = src.index("_guard_tripped()")
    after = src[i:i + 1200]
    assert "vf_guard_retries" in after, "the trip must be counted, not fatal"
    assert "continue" in after
    assert 'return "force_abort"' in after   # ...still fatal once exhausted


def test_every_mj_attribute_the_backend_reads_exists_on_the_config():
    """A partial patch once shipped code reading `mj.vf_transit_margin_m`
    with the field never written -- both rooms crashed at the first station.
    Static check: every `mj.<name>` / `self.mj.<name>` read in the backend
    must be a real field of VForceConfig."""
    import ast
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from execution.vforce_backend import VForceConfig

    cfg = VForceConfig()
    bad = []
    for node in ast.walk(ast.parse(SRC)):
        if isinstance(node, ast.Attribute) and isinstance(node.value,
                                                          (ast.Name,
                                                           ast.Attribute)):
            base = node.value
            name = (base.id if isinstance(base, ast.Name) else base.attr)
            if name == "mj" and not node.attr.startswith("__"):
                if not hasattr(cfg, node.attr):
                    bad.append(f"line {node.lineno}: mj.{node.attr}")
    assert not bad, bad
