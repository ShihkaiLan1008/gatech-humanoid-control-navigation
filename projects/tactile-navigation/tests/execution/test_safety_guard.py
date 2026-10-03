"""Batch 3A.2 Phase 1: the all-mode force guard, checked structurally and
behaviourally.

Two unvalidated safety fixes were shipped before this batch and the first one
measurably did nothing, so these tests are written the other way round: the
STRUCTURE that makes the guard unbypassable is asserted (every loop that steps
physics consumes the trip), and the executive's handling is exercised against a
scripted backend.  The physical demonstration lives in
``batch3a.injected_contact_check`` and its result is quoted in the report.
"""

import ast
import inspect
import textwrap

import numpy as np
import pytest

from execution.backends import BackendConfig, KinematicBackend
from execution.executive import ExecConfig, Executive
from execution.scenes import build_scene
from local_tactile.fsm import ExclusionAudit, Mode


# ── structure: the guard cannot be bypassed by any mode or future primitive ──────


def _mujoco_source_tree():
    from execution import mujoco_backend
    return ast.parse(textwrap.dedent(inspect.getsource(mujoco_backend)))


def _methods_calling(tree, callee: str):
    out = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            calls = {n.func.attr for n in ast.walk(node)
                     if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
            if callee in calls:
                out.append((node.name, calls))
    return out


def test_every_loop_that_steps_physics_consumes_the_guard_trip():
    """A blocking primitive advances physics only through ``_step``; every
    method that calls ``_step`` must also consume ``_guard_tripped`` -- so a
    FUTURE primitive that forgets the guard fails here, not on the robot.
    ``_safety_stop`` is the response itself and is the one exemption."""
    steppers = _methods_calling(_mujoco_source_tree(), "_step")
    assert {n for n, _ in steppers} >= {"goto_map", "_turn_once", "_stabilize_world",
                                        "press", "_retract"}
    for name, calls in steppers:
        if name == "_safety_stop":
            continue
        assert "_guard_tripped" in calls, (
            f"{name} steps physics but never consumes a force-abort trip")


def test_the_guard_lives_in_the_single_per_tick_path():
    """_step samples EVERY hand and applies the threshold there -- the one
    place every mode passes through (measured: a guard placed in goto_map
    never fired while STABILIZE and TURN accumulated 43 N)."""
    src = inspect.getsource(__import__("execution.mujoco_backend",
                                       fromlist=["MujocoBackend"]).MujocoBackend._step)
    assert "for n, h in self.hands.items()" in src
    assert "guard_abort_force_n" in src
    assert "force_abort_events" in src
    assert "allows_hand_contact" in src


def test_the_parked_hand_is_guarded_even_in_probe_mode():
    """The shield regulates the PROBING hand during a press; the parked hand
    has no business touching anything in any mode, so contact-allowed modes
    exempt only the probe hand."""
    src = inspect.getsource(__import__("execution.mujoco_backend",
                                       fromlist=["MujocoBackend"]).MujocoBackend._step)
    assert "self.park_name" in src, "PROBE mode must still guard the parked hand"


def test_guard_threshold_sits_well_below_the_safety_limit():
    from execution.mujoco_backend import MujocoConfig
    cfg = MujocoConfig()
    assert cfg.guard_abort_force_n <= 0.5 * cfg.force_limit_n
    assert cfg.guard_abort_force_n >= 10.0, "must stay clear of the ~8 N a press produces"


def test_the_safety_stop_halts_and_retracts():
    """The response is in the backend, before control returns to the caller:
    hold the current pose standing, park both hands, wait for the force to
    drain."""
    from execution.mujoco_backend import MujocoBackend
    src = inspect.getsource(MujocoBackend._safety_stop)
    assert "standing=True" in src and "hold=True" in src
    assert 'self._park_hand(self.hands["left"])' in src
    assert 'self._park_hand(self.hands["right"])' in src
    assert "safety_clear_force_n" in src


# ── metrics: one headline number, asserted against the audit ─────────────────────


def test_audit_peak_covers_all_modes_and_substep_samples():
    audit = ExclusionAudit()
    audit.observe(time_s=0.0, mode=Mode.LOCOMOTION, commanded_twist_norm=0.2,
                  hand_force_n=5.0, commanded_contact=0.0)
    audit.observe(time_s=0.1, mode=Mode.PROBE, commanded_twist_norm=0.0,
                  hand_force_n=30.0, commanded_contact=1.0)
    assert audit.peak_hand_force_n == 30.0
    assert audit.max_hand_force_while_moving_n == 5.0, \
        "the while-moving figure must stay a separate diagnostic"
    audit.note_force(41.5)
    assert audit.peak_hand_force_n == 41.5
    assert audit.to_dict()["peak_hand_force_n"] == 41.5


def test_summary_peak_is_asserted_equal_to_audit_peak():
    from execution.mujoco_backend import MujocoBackend
    src = inspect.getsource(MujocoBackend.physical_summary)
    assert "assert" in src and "peak_hand_force_n" in src, \
        "physical_summary must refuse to report a peak the audit did not see"
    src_sub = inspect.getsource(MujocoBackend._on_substep)
    assert "note_force" in src_sub, "substep samples must reach the audit too"


# ── the executive's handling, against a scripted backend ─────────────────────────


class TrippingBackend(KinematicBackend):
    """Kinematic stand-in that raises a force abort on selected goto calls,
    exactly as the physical backend does: an event appended, 'force_abort'
    returned, motion NOT performed."""

    def __init__(self, *a, trip_on=(1,), **kw):
        super().__init__(*a, **kw)
        self.force_abort_events = []
        self.force_aborts = 0
        self._goto_calls = 0
        self._trip_on = set(trip_on)

    def goto_map(self, *a, **kw):
        self._goto_calls += 1
        if self._goto_calls in self._trip_on:
            self.force_aborts += 1
            self.force_abort_events.append(
                {"t": self.time_s, "mode": "LOCOMOTION", "hand": "left",
                 "force_n": 14.2})
            return "force_abort"
        return super().goto_map(*a, **kw)


def _tripping_exec(trip_on, **cfg_kw):
    sc = build_scene("two_turns")
    be = TrippingBackend(sc, BackendConfig(), np.random.default_rng(0),
                         trip_on=trip_on)
    e = Executive(sc, be, ExecConfig(seed=0, policy="belief_aware",
                                     n_particles=400, **cfg_kw))
    e.verbose = False
    return e


def test_a_force_abort_is_a_safety_owned_failure_and_grows_the_margin():
    e = _tripping_exec(trip_on=(1,))
    r = e.run()
    safety = [f for f in r["failures"] if f["owner"] == "safety"]
    assert len(safety) == 1
    assert "force_abort" in safety[0]["reason"]
    assert not safety[0]["terminal"]
    assert r["safety_aborts"] == 1
    assert r["retreat_margin_m"] == pytest.approx(
        ExecConfig().safety_retreat_step_m)
    assert r["outcome"] != "safety_abort", "one abort must not end the episode"


def test_repeated_force_aborts_terminate_with_a_safety_outcome():
    e = _tripping_exec(trip_on=(1, 2, 3))
    r = e.run()
    assert r["outcome"] == "safety_abort"
    safety = [f for f in r["failures"] if f["owner"] == "safety"]
    assert len(safety) == ExecConfig().max_safety_aborts
    assert safety[-1]["terminal"]


def test_the_retreat_margin_moves_every_later_approach_off_the_wall():
    e = _tripping_exec(trip_on=())
    b0 = e._bias_vector(0).copy()
    e._retreat_margin_m = 0.08
    b1 = e._bias_vector(0)
    v = np.asarray(e.scene.legs[0]["v"], float)
    assert float((b0 - b1) @ v) == pytest.approx(0.08, abs=1e-9)


# ── preventive clearance, belief-frame only ──────────────────────────────────────


def _plain_exec(scene="two_turns"):
    sc = build_scene(scene)
    be = KinematicBackend(sc, BackendConfig(), np.random.default_rng(0))
    e = Executive(sc, be, ExecConfig(seed=0, policy="belief_aware", n_particles=400))
    e.verbose = False
    return e


def test_translation_clearance_shifts_a_target_that_hugs_the_wall():
    e = _plain_exec()
    wall_face_y = e.scene.standoff_m                      # 0.38
    hugging = np.array([0.8, wall_face_y - 0.16])         # 0.16 m off the face
    adjusted, clearance = e._clear_translation_target(hugging, 0)
    assert e._hyp_min_clearance_m(adjusted) >= ExecConfig().hand_clearance_m - 1e-6
    assert adjusted[1] < hugging[1], "the shift must move AWAY from the wall"


def test_translation_clearance_shift_is_capped():
    """A target so deep that restoring clearance would mean walking a very
    different route is shifted only up to the cap; past that the force guard is
    the protection, not a silent reroute."""
    e = _plain_exec()
    deep = np.array([0.8, e.scene.standoff_m - 0.05])
    adjusted, _ = e._clear_translation_target(deep, 0)
    moved = float(deep[1] - adjusted[1])
    assert moved <= ExecConfig().max_clearance_shift_m + 0.03 + 1e-9


def test_translation_clearance_leaves_a_safe_target_alone():
    e = _plain_exec()
    safe = np.array([0.8, 0.0])                           # on the route line
    adjusted, clearance = e._clear_translation_target(safe, 0)
    assert np.allclose(adjusted, safe)
    assert e._clearance_shifts == 0


def test_clearance_is_worst_case_over_live_hypotheses():
    """With a bimodal belief the same commanded displacement starts from two
    different believed positions; the check must take the worst one."""
    e = _plain_exec("ambiguous_near_edge")
    hyps = [h for h in e.mb.hypotheses() if h["mass"] > 0.05]
    assert len(hyps) == 2, "the prior really is bimodal"
    tracked = e._believed_xy()
    per_hyp = []
    target = np.array([1.2, 0.05])
    for h in hyps:
        a = np.asarray(h["xy"], float)
        rel = target - tracked
        pts = a[None, :] + np.linspace(0, 1, 40)[:, None] * rel[None, :]
        per_hyp.append(min(float(np.min(body.signed_distance(pts, None)[0]))
                           for body in e.scene.plan.bodies))
    assert e._hyp_min_clearance_m(target) == pytest.approx(min(per_hyp), abs=1e-6)


def test_turn_clearance_steps_away_before_turning():
    e = _plain_exec()
    moved = {}
    e._hyp_min_clearance_m = lambda target=None: 0.20     # pretend we are close

    def fake_goto(xy, yaw, *, tol, label):
        moved["to"] = np.asarray(xy, float)
        return "arrived"

    e._goto_map = fake_goto
    e._ensure_turn_clearance()
    assert "to" in moved, "a deficit must produce a retreat move"
    deficit = ExecConfig().turn_clearance_m - 0.20
    got = float(np.linalg.norm(moved["to"] - e._believed_xy()))
    assert got == pytest.approx(deficit, abs=0.02)


def test_turn_clearance_is_a_noop_when_there_is_room():
    e = _plain_exec()
    called = []
    e._goto_map = lambda *a, **k: called.append(1) or "arrived"
    e._ensure_turn_clearance()                            # start pose: 0.38 m clear
    assert not called


def test_clearance_reads_only_belief_and_map():
    """No truth in the prevention path: the check must not call truth_pose or
    world_point_to_map."""
    import textwrap as _tw
    for fn in (Executive._hyp_min_clearance_m, Executive._clear_translation_target,
               Executive._ensure_turn_clearance):
        src = _tw.dedent(inspect.getsource(fn))
        assert "truth_pose" not in src, fn.__name__
        assert "world_point_to_map" not in src, fn.__name__


# ── the locomotion speed governor ────────────────────────────────────────────────


def test_governor_scales_speed_with_believed_clearance():
    """A wrist snag was measured at 44.6 N on the guard's FIRST over-threshold
    sample -- reaction cannot cap that, only approach energy can.  Speed must
    fall monotonically with believed clearance, floored, and reach full speed
    only with real room."""
    from execution.mujoco_backend import MujocoConfig

    class _FakePhysical(KinematicBackend):
        mj = MujocoConfig()

    sc = build_scene("two_turns")
    be = _FakePhysical(sc, BackendConfig(), np.random.default_rng(0))
    e = Executive(sc, be, ExecConfig(seed=0, policy="belief_aware", n_particles=400))
    e.verbose = False
    mj = MujocoConfig()
    lo = e._governed_speed(0.20)
    mid = e._governed_speed(0.38)
    hi = e._governed_speed(0.80)
    assert lo == pytest.approx(mj.speed_scale_min)
    assert hi == pytest.approx(mj.speed_scale_locomotion)
    assert lo < mid < hi
    assert mid <= 0.5 * mj.speed_scale_locomotion, \
        "at the scenes' own standoff the walk must be materially slower"


def test_governor_is_inert_on_the_kinematic_stand_in():
    e = _plain_exec()
    assert e._governed_speed(0.38) is None
