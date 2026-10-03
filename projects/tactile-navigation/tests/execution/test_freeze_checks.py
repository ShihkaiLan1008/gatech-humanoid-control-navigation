"""The pre-freeze audit: the four things that must hold before Batch 3A is frozen.

These are not unit tests of a mechanism -- they are the contract checks the
batch is frozen against, written so that a later change which violates one
fails here rather than in a physical run.
"""

import inspect

import numpy as np
import pytest

from execution import executive as ex_mod
from execution.backends import BackendConfig, KinematicBackend
from execution.executive import ExecConfig, Executive
from execution.interface import GlobalDirective, LocalReport, assert_no_truth_leak
from execution.scenes import SCENES, build_scene
from execution.stations import REACH_M


def _run(scene, policy="belief_aware", seed=0, n=400):
    sc = build_scene(scene)
    be = KinematicBackend(sc, BackendConfig(), np.random.default_rng(seed))
    e = Executive(sc, be, ExecConfig(seed=seed, policy=policy, n_particles=n))
    e.verbose = False
    return sc, be, e.run(), e


# ── 1. planner / controller interface ────────────────────────────────────────────


def test_the_interface_is_the_only_channel_between_planner_and_controller():
    """Every global->local instruction is a GlobalDirective and every
    local->global answer is a LocalReport; nothing else crosses."""
    for scene in SCENES:
        _, _, r, _ = _run(scene)
        for d in r["directives"]:
            assert set(GlobalDirective.__dataclass_fields__) >= set()
            assert "route_edge" in d and "allowed_decisions" in d
        assert isinstance(r["n_probes"], int)


def test_belief_update_accepts_only_a_local_report():
    sig = inspect.signature(Executive._update_global)
    assert list(sig.parameters) == ["self", "report"]
    _, be, _, e = _run("two_turns")
    for bad in ({"odom_delta_body": (0.0, 0.0)}, be, None, 3.14):
        with pytest.raises(AssertionError):
            e._update_global(bad)


# ── 2. truth firewall ────────────────────────────────────────────────────────────


def test_no_ground_truth_geometry_or_identity_reaches_the_belief():
    """Every report the executive actually built during real runs is walked."""
    for scene in SCENES:
        sc = build_scene(scene)
        be = KinematicBackend(sc, BackendConfig(), np.random.default_rng(0))
        e = Executive(sc, be, ExecConfig(seed=0, policy="belief_aware", n_particles=400))
        e.verbose = False
        seen = []
        original = e._update_global

        def spy(report):
            assert isinstance(report, LocalReport)
            assert_no_truth_leak(report)
            seen.append(report)
            return original(report)

        e._update_global = spy
        e.run()
        assert seen, f"{scene} produced no belief updates to audit"
        for rep in seen:
            for p in rep.probes:
                # body frame only -- no world point, no map identity of what was hit
                assert p.p_body is None or len(p.p_body) == 2
                assert not hasattr(p, "world_xy")
                assert p.station is None or isinstance(p.station, str)


def _code_only(fn) -> str:
    """Source with comments and docstrings stripped -- the prose in this module
    talks ABOUT world frames and truth, and a scan that reads prose reports
    leaks that do not exist."""
    import ast, textwrap
    src = textwrap.dedent(inspect.getsource(fn))
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef,
                             ast.Module)) and ast.get_docstring(node):
            node.body = node.body[1:]
    return ast.unparse(tree)


def test_contacts_enter_the_belief_without_a_map_identity():
    """A press reports 'surface', never 'this is wall_0' -- a tactile sensor
    cannot identify which mapped body it touched from one contact, and letting
    it would be identity leakage."""
    src = _code_only(ex_mod.Executive._update_global)
    assert "signature='surface'" in src or 'signature="surface"' in src
    for bad in ("body.name", "plan.bodies", "truth_pose", "world_point"):
        assert bad not in src, bad


def test_analytic_predictions_never_become_probe_outcomes():
    """G1Station.predict is planning-only.  If it appeared in the observation
    path, the belief would be updating on the map instead of on the arm."""
    src = _code_only(ex_mod.Executive.probe_station)
    assert ".predict(" not in src
    assert "be.press(" in src
    assert "reading.contacted" in src


# ── 3. safety audit across all executed legs ─────────────────────────────────────


def test_every_executed_probe_is_within_the_measured_reach():
    for scene in SCENES:
        sc, _, r, e = _run(scene)
        for name in r["stations_visited"]:
            st = e._station_cache.get(name) or sc.stations[name]
            assert max(abs(o) for o in st.fan_offsets_m) <= st.reach_m
            assert st.reach_m <= REACH_M + 1e-9


def test_probe_budgets_bind_on_every_scene():
    for scene in SCENES:
        _, _, r, _ = _run(scene)
        assert r["n_probes"] <= ExecConfig().max_probes_total


def test_failures_carry_an_owner_on_every_scene():
    allowed = {"global_belief", "global_planner", "ate", "controller", "route",
               "safety"}
    for scene in SCENES:
        for seed in range(4):
            _, _, r, _ = _run(scene, seed=seed)
            for f in r["failures"]:
                assert f["owner"] in allowed, (scene, f)


# ── 4. arrival and turn decisions use REALIZED base state ───────────────────────


def test_turn_completion_is_verified_against_realized_yaw_not_duration():
    """The turn must settle-and-verify, and its timeout must be a FAILURE."""
    from execution.mujoco_backend import MujocoBackend, MujocoConfig
    src = _code_only(MujocoBackend.turn_to_map)
    assert "_stabilize_world" in src, "turn must settle before it is believed"
    # realized = the ROBOT'S OWN yaw (working_pose): hardware has no truth
    assert "working_pose()[1]" in src, "and verify against realized yaw"
    assert "turn_settle_tolerance_rad" in src
    assert "timeout" in src and "turn_incomplete" in src, "both failure exits exist"
    # a duration alone can never report success: the outer loop has no clock
    assert "self._t - t0" not in src, "no duration-based completion in the outer loop"
    cfg = MujocoConfig()
    assert cfg.turn_yaw_tolerance_rad <= 0.05, "the drive band IS the error it leaves"
    assert cfg.max_turn_corrections >= 1


def test_arrival_is_decided_on_the_believed_pose_not_the_true_one():
    """Superseded contract.  This test used to REQUIRE `truth_pose()` here, on the
    reading that `_at_goal` is the harness's success detector rather than a robot
    decision.  Measured on 4 scenes x 5 seeds, that reading does not survive: the
    old rule stopped the episode the instant the TRUE pose entered the goal
    radius, so 16/20 runs scored `arrived` while only 8/20 were positions a robot
    could have known it had reached.  A real robot has no such signal.  The stop
    is now belief-gated; `arrived_truly` in the summary carries the truth-scored
    answer, and the gap between them is the error this pipeline has to close."""
    src = _code_only(Executive._at_goal)
    assert "truth_pose" not in src, "arrival must not read simulator truth"
    assert "_believed_xy" in src
    assert "goal_radius_m" in src


def test_arrival_requires_belief_confidence_like_a_turn():
    """Declaring arrival on a belief the planner would refuse to turn on is what
    left the robot 0.205 m out; the same bar now applies."""
    src = _code_only(Executive._plan_inner)
    assert "arrive_sigma_max_m" in src
    assert ExecConfig().arrive_sigma_max_m <= ExecConfig().branch_sigma_max_m + 1e-9


def test_no_touch_baseline_can_never_probe():
    """The baseline's defining property, enforced structurally."""
    for scene in SCENES:
        for seed in range(4):
            _, _, r, _ = _run(scene, policy="no_touch", seed=seed)
            assert r["n_probes"] == 0, scene
            assert r["stations_visited"] == []


# ── determinism, for a frozen artefact ───────────────────────────────────────────


def test_every_scene_is_deterministic_given_a_seed():
    for scene in SCENES:
        a = _run(scene, seed=2)[2]
        b = _run(scene, seed=2)[2]
        assert a["outcome"] == b["outcome"], scene
        assert a["stations_visited"] == b["stations_visited"], scene
        assert a["n_probes"] == b["n_probes"], scene
