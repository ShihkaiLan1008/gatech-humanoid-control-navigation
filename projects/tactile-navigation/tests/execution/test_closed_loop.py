"""The closed loop, on the kinematic stand-in.

These assert the LOGIC the physical runs then have to survive: that touch moves
the global hypothesis weights, that the branch gate refuses to commit on an
unresolved belief, that probe budgets and reach limits bind, and that nothing
in the belief path can see the simulator.  What they cannot check is tracking,
drift and force -- that is what the MuJoCo runs in the report are for.
"""


import numpy as np
import pytest

from execution.backends import BackendConfig, KinematicBackend
from execution.executive import BRANCH_DELTA, ExecConfig, Executive
from execution.interface import RouteDecision
from execution.scenes import build_scene

N = 400


def _exec(scene_name, policy="belief_aware", seed=0, **kw):
    sc = build_scene(scene_name)
    be = KinematicBackend(sc, BackendConfig(), np.random.default_rng(seed))
    return Executive(sc, be, ExecConfig(seed=seed, policy=policy, n_particles=N, **kw))


def _runs(scene_name, policy, seeds=range(6)):
    return [_exec(scene_name, policy, s).run() for s in seeds]


# ── the persistent loop completes route decisions ────────────────────────────────


def test_two_turns_completes_both_route_decisions():
    """ACCEPTANCE (logic half): one episode, at least two route decisions."""
    rows = _runs("two_turns", "belief_aware")
    assert max(r["n_route_decisions"] for r in rows) >= 2
    assert np.mean([r["n_route_decisions"] for r in rows]) >= 1.4


def test_the_ambiguous_scene_is_navigated_by_touch_and_not_by_guessing():
    """The headline comparison the batch asks for."""
    ba = _runs("ambiguous_near_edge", "belief_aware")
    nt = _runs("ambiguous_near_edge", "no_touch")
    assert np.mean([r["arrived"] for r in ba]) >= 0.8
    assert np.mean([r["arrived"] for r in ba]) > np.mean([r["arrived"] for r in nt])
    assert np.mean([r["global_pos_err_m"] for r in ba]) < \
        0.5 * np.mean([r["global_pos_err_m"] for r in nt])


# ── realized touch changes the belief AND the plan ───────────────────────────────


def test_realized_touch_moves_the_global_hypothesis_weights():
    """ACCEPTANCE: realized touch changes the global hypothesis weights.

    Not "a belief update ran" -- the MASS has to move, and it has to move
    because of what the arm felt.
    """
    ex = _exec("ambiguous_near_edge")
    before = [h["mass"] for h in ex._hypotheses()]
    assert len(before) == 2 and min(before) > 0.3, "the prior really is bimodal"
    ex.run()
    after = [h["mass"] for h in ex._hypotheses()]
    assert ex.n_probes > 0
    assert max(after) > 0.9, "touch must resolve it, not merely perturb it"
    assert abs(max(after) - max(before)) > 0.3


def test_the_branch_gate_refuses_to_commit_while_the_belief_is_split():
    """A TURN is never offered while the minor hypothesis holds real mass."""
    ex = _exec("ambiguous_near_edge")
    result = ex.run()
    committed_at = [d for d in result["directives"]
                    if "TURN" in d["allowed_decisions"]]
    assert committed_at, "the episode did commit eventually"
    for d in committed_at:
        minor = 1.0 - max(d["hypothesis_masses"])
        assert minor <= BRANCH_DELTA + 1e-9, d


def test_an_unresolved_belief_produces_a_probe_directive_not_a_turn():
    ex = _exec("ambiguous_near_edge")
    first = ex.plan()
    assert RouteDecision.PROBE in first.allowed_decisions
    assert RouteDecision.TURN not in first.allowed_decisions
    assert first.station is not None
    assert first.probe_budget > 0


def test_the_no_touch_policy_commits_on_the_prior_and_pays_for_it():
    """The baseline must actually be a baseline: it turns without touching."""
    rows = _runs("ambiguous_near_edge", "no_touch")
    assert all(r["n_probes"] == 0 for r in rows)
    assert np.mean([r["arrived"] for r in rows]) < 0.5
    assert np.mean([r["global_pos_err_m"] for r in rows]) > 0.15


# ── conservative execution ───────────────────────────────────────────────────────


def test_probe_budgets_bind_per_station_and_per_episode():
    cfg = ExecConfig(seed=0, policy="belief_aware", n_particles=N,
                     max_probes_per_station=2, max_probes_total=5)
    sc = build_scene("ambiguous_near_edge")
    be = KinematicBackend(sc, BackendConfig(), np.random.default_rng(0))
    r = Executive(sc, be, cfg).run()
    assert r["n_probes"] <= 5
    from collections import Counter
    per_station = Counter(p["station"] for p in r["probes"])
    for station, n in per_station.items():
        assert n <= 2 * 3, (station, n)     # budget per visit, visits capped at 3


def test_every_probe_the_executive_actually_ran_was_reachable():
    """ACCEPTANCE: every selected station and probe is reachable -- checked on
    the stations the run REALLY visited, not just on the catalogue."""
    for scene_name in ("two_turns", "ambiguous_near_edge"):
        r = _runs(scene_name, "belief_aware", seeds=range(3))
        sc = build_scene(scene_name)
        for row in r:
            for name in row["stations_visited"]:
                st = sc.stations[name]
                assert st.reachable(sc.plan, 0.0), name
                assert max(st.fan_offsets_m) <= st.reach_m


def test_failures_are_attributed_to_an_owner():
    """ACCEPTANCE: attribute failures to global belief, planner, ATE, controller,
    route or safety."""
    allowed = {"global_belief", "global_planner", "ate", "controller", "route",
               "safety"}
    for scene_name in ("two_turns", "ambiguous_near_edge"):
        for row in _runs(scene_name, "belief_aware", seeds=range(4)):
            for f in row["failures"]:
                assert f["owner"] in allowed, f


# ── the firewall, exercised by a real run ────────────────────────────────────────


def test_the_belief_update_entry_point_rejects_anything_but_a_report():
    ex = _exec("two_turns")
    with pytest.raises(AssertionError):
        ex._update_global({"odom_delta_body": (0.0, 0.0)})       # a bare dict
    with pytest.raises(AssertionError):
        ex._update_global(ex.be)                                  # the backend


def test_no_truth_reaches_the_belief_even_when_the_belief_is_wrong():
    """The strongest available check: run the ambiguous scene, where the belief
    starts 0.55 m wrong half the time, and confirm the global position error is
    NOT driven to zero -- a leak would show up as an impossibly good belief."""
    rows = _runs("ambiguous_near_edge", "belief_aware", seeds=range(5))
    errs = [r["global_pos_err_m"] for r in rows]
    assert min(errs) > 1e-6, "a belief with zero error is reading the answer"


def test_hypothesis_trace_records_what_the_belief_knew_when():
    r = _exec("ambiguous_near_edge").run()
    tr = r["hypothesis_trace"]
    assert len(tr) > 3
    for row in tr:
        assert {"t", "leg", "s_hat", "s_sigma", "masses"} <= set(row)
        assert abs(sum(row["masses"]) - 1.0) < 0.02


# ── determinism ──────────────────────────────────────────────────────────────────


def test_kinematic_episodes_are_deterministic_given_a_seed():
    a = _exec("ambiguous_near_edge", seed=3).run()
    b = _exec("ambiguous_near_edge", seed=3).run()
    assert a["outcome"] == b["outcome"]
    assert a["stations_visited"] == b["stations_visited"]
    assert a["n_probes"] == b["n_probes"]
