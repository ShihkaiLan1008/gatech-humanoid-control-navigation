"""Batch 3A.2 Phase 2: no planner livelock.

The measured failure (ambiguous_near_edge / no_touch / s2, MuJoCo): 120 actions
burned in 5.2 s of simulated time on an identical zero-motion CONTINUE.  The
mechanism was a MEAN-versus-MODE split: the planner gated distance-to-corner on
the weighted mean over a still-bimodal belief while locomotion walked the
TRACKED mode -- 0.28 m apart at a 0.55 m ambiguity -- so the tracked mode stood
"arrived" at a waypoint the mean said was still ahead, no motion was commanded,
no odometry was absorbed, and the identical directive re-issued forever.

Two independent fixes, each tested here: geometric decisions now use the same
tracked-hypothesis position locomotion uses, and a no-forward-progress detector
backstops ANY future zero-progress cycle with a forced anchor (belief-aware) or
an explicit global-planner failure (no-touch, which structurally cannot probe).
"""

import numpy as np

from execution.backends import BackendConfig, KinematicBackend
from execution.executive import ExecConfig, Executive
from execution.scenes import build_scene


class StalledBackend(KinematicBackend):
    """Reports 'arrived' without ever moving and returns zero odometry: the
    distilled livelock substrate -- no realized displacement, no belief change."""

    def goto_map(self, *a, **kw):
        return "arrived"

    def turn_to_map(self, *a, **kw):
        return "arrived"

    def odometry_increment(self):
        return (0.0, 0.0), 0.0


def _exec(scene="ambiguous_near_edge", policy="no_touch", backend_cls=KinematicBackend,
          seed=0, **kw):
    sc = build_scene(scene)
    be = backend_cls(sc, BackendConfig(), np.random.default_rng(seed))
    e = Executive(sc, be, ExecConfig(seed=seed, policy=policy, n_particles=400, **kw))
    e.verbose = False
    return e


# ── the reproduced livelock is removed ───────────────────────────────────────────


def test_a_stalled_no_touch_episode_terminates_with_an_explicit_failure():
    """Zero motion + zero belief change + identical directives must not burn
    the action budget: no_touch cannot probe its way out, so the only honest
    exit is a terminal global-planner failure, quickly and deterministically."""
    e = _exec(policy="no_touch", backend_cls=StalledBackend)
    r = e.run()
    assert r["outcome"] == "no_progress"
    assert r["no_progress_events"] >= 1
    owners = [(f["owner"], f["terminal"]) for f in r["failures"]]
    assert ("global_planner", True) in owners
    assert len(r["directives"]) <= 10, \
        f"a livelock must be cut short, not ridden: {len(r['directives'])} directives"


def test_a_stalled_belief_aware_episode_escalates_then_fails_explicitly():
    """belief_aware first forces an anchor probe; when even that cannot move
    anything (backend fully stalled), the detector must still end the episode
    rather than spin."""
    e = _exec(policy="belief_aware", backend_cls=StalledBackend)
    r = e.run()
    assert r["outcome"] in ("no_progress", "planner_abort", "safety_abort")
    assert len(r["directives"]) < ExecConfig().max_actions, \
        "the action budget must never be consumed by zero-progress repetition"


def test_stalled_episodes_are_deterministic():
    a = _exec(policy="no_touch", backend_cls=StalledBackend).run()
    b = _exec(policy="no_touch", backend_cls=StalledBackend).run()
    assert a["outcome"] == b["outcome"]
    assert a["no_progress_events"] == b["no_progress_events"]
    assert len(a["directives"]) == len(b["directives"])


# ── the detector itself ──────────────────────────────────────────────────────────


def test_detector_needs_three_identical_zero_progress_iterations():
    e = _exec()
    d1 = e.plan()
    assert not e._no_progress_check(d1)
    assert not e._no_progress_check(d1)
    assert e._no_progress_check(d1), "third identical zero-progress plan must trip"


def test_detector_resets_on_realized_motion():
    e = _exec()
    d1 = e.plan()
    assert not e._no_progress_check(d1)
    e._odom_travel += 0.5                      # the robot really moved
    assert not e._no_progress_check(d1)
    assert not e._no_progress_check(d1)
    assert e._no_progress_check(d1)


def test_detector_resets_when_the_directive_changes():
    e = _exec()
    d1 = e.plan()
    assert not e._no_progress_check(d1)
    assert not e._no_progress_check(d1)
    e.mb.predict(np.array([0.3, 0.0]), 0.0)   # belief moved -> different state
    d2 = e.plan()
    assert not e._no_progress_check(d2)


# ── mean-versus-mode coherence ───────────────────────────────────────────────────


def test_geometric_decisions_use_the_tracked_hypothesis_not_the_bimodal_mean():
    """The planner must gate distance-to-corner on the SAME position estimate
    locomotion navigates by.  A bimodal mean is a place no particle believes
    in; gating on it while walking on the mode is what produced the livelock."""
    import inspect, textwrap
    src = textwrap.dedent(inspect.getsource(Executive._plan_inner))
    assert "_tracked_s()" in src


def test_no_touch_on_the_ambiguous_scene_never_burns_the_action_budget():
    """ACCEPTANCE: across 12 seeds, the baseline either finishes or fails
    explicitly -- no seed may repeat a zero-progress directive indefinitely."""
    for seed in range(12):
        e = _exec(policy="no_touch", seed=seed)
        r = e.run()
        assert r["outcome"] != "action_cap", (seed, r["outcome"])
        assert len(r["directives"]) <= 60, (seed, len(r["directives"]))


def test_the_detector_is_silent_on_success_and_explicit_on_stalls():
    """An episode that arrives never tripped the detector; an episode the
    detector DOES trip ends with an explicit failure instead of burning the
    action budget.  (A trip on a non-arriving episode is the feature working:
    measured case -- after two turns the goal leg's along-axis inherits the
    accumulated LATERAL drift, which nothing on that leg can observe, so a
    belief standing 'at the goal' 0.3 m from it is genuinely stuck, and the
    only honest exits are an anchor or an explicit abort.)"""
    for scene in ("two_turns", "ambiguous_near_edge"):
        for seed in range(4):
            e = _exec(scene=scene, policy="belief_aware", seed=seed)
            r = e.run()
            if r["arrived"]:
                assert r["no_progress_events"] == 0, (scene, seed)
            elif r["no_progress_events"]:
                assert r["outcome"] in ("no_progress", "planner_abort"), (scene, seed)
                assert len(r["directives"]) < ExecConfig().max_actions
