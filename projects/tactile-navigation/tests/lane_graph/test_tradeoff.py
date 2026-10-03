"""Offline route/probe tradeoff selection (`lane_graph.tradeoff`).

Pins the contracts the bounds-sweep report relies on: feasibility is strict
and never relaxed, pruning only ever returns a feasible schedule, the fastest
selection is a lexicographic minimum over feasible rows, and the timing
sensitivity is a pure re-weighting that leaves probe counts and distances
untouched.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

from lane_graph.constrained import PlannerBelief
from lane_graph.graph3 import Graph3
from lane_graph.tradeoff import (GraphBackend, Timing, evaluate_route, feasible, pareto_indices,
                                 prune_schedule, select_candidate, timing_sensitivity)
from lane_graph.uncertainty import DEFAULT


def _row(cid, distance, time, probes, pos=0.05, yaw=2.0, resolved=True, **kw):
    """A candidate whose belief sits at (pos, yaw) from the start onward."""
    return dict(candidate_id=cid, distance_m=distance, estimated_time_s=time,
                probe_count=probes, peak_position_sigma_m=pos, peak_yaw_sigma_deg=yaw,
                initial_position_sigma_m=pos, initial_yaw_sigma_deg=yaw,
                grace_position_m=0.03, grace_yaw_deg=1.5,
                profile=dict(pre_pos=[pos], post_pos=[pos], pre_yaw_deg=[yaw], post_yaw_deg=[yaw],
                             distance=[distance]),
                association_resolved=resolved, commanded_turn_rad=kw.pop("turn", 0.0), **kw)


def _profile_row(cid, s0, pre_pos, post_pos, grace=0.03):
    n = len(pre_pos)
    return dict(candidate_id=cid, distance_m=float(n), estimated_time_s=1.0, probe_count=0,
                peak_position_sigma_m=max([s0] + pre_pos + post_pos), peak_yaw_sigma_deg=1.0,
                initial_position_sigma_m=s0, initial_yaw_sigma_deg=1.0,
                grace_position_m=grace, grace_yaw_deg=1.5,
                profile=dict(pre_pos=pre_pos, post_pos=post_pos, pre_yaw_deg=[1.0] * n,
                             post_yaw_deg=[1.0] * n, distance=[float(i + 1) for i in range(n)]),
                association_resolved=True, commanded_turn_rad=0.0)


def test_timing_rejects_zero_speeds_but_allows_free_probes():
    Timing(0.5, 1.0, 0.0)
    with pytest.raises(ValueError):
        Timing(0.0, 1.0, 5.0)
    with pytest.raises(ValueError):
        Timing(0.5, 1.0, -1.0)
    parts = Timing(0.5, 1.0, 5.0).components(10.0, 2.0, 3)
    assert parts == dict(walk_time_s=20.0, turn_time_s=2.0, probe_time_s=15.0)


def test_feasible_is_strict_and_requires_resolution():
    assert feasible(_row("a", 1, 1, 0, pos=0.10, yaw=4.0), 0.10, 4.0)
    assert not feasible(_row("a", 1, 1, 0, pos=0.1001, yaw=4.0), 0.10, 4.0)
    assert not feasible(_row("a", 1, 1, 0, pos=0.05, yaw=4.01), 0.10, 4.0)
    assert not feasible(_row("a", 1, 1, 0, resolved=False), 0.10, 4.0)


def test_localize_first_start_outside_the_bound():
    from lane_graph.tradeoff import localization
    # Start at 0.15, drift to 0.17, a contact brings it to 0.09, then it stays inside 0.12.
    good = _profile_row("good", 0.15, pre_pos=[0.16, 0.17, 0.10, 0.11], post_pos=[0.16, 0.09, 0.10, 0.11])
    ok, at = localization(good, 0.12, 4.0)
    assert ok and at == 2.0
    assert feasible(good, 0.12, 4.0)
    # Same route but drifting past initial + grace before the first contact: infeasible.
    late = _profile_row("late", 0.15, pre_pos=[0.16, 0.19, 0.10, 0.11], post_pos=[0.16, 0.09, 0.10, 0.11])
    assert not feasible(late, 0.12, 4.0)
    # Localized, then leaves the bound again: infeasible.
    leaks = _profile_row("leaks", 0.15, pre_pos=[0.16, 0.17, 0.10, 0.13], post_pos=[0.16, 0.09, 0.10, 0.13])
    assert not feasible(leaks, 0.12, 4.0)
    # Never reaches the bound: infeasible.
    never = _profile_row("never", 0.15, pre_pos=[0.16, 0.17], post_pos=[0.16, 0.14])
    assert not feasible(never, 0.12, 4.0)
    # A start inside the bound is localized by its first step and may not leave afterwards.
    inside = _profile_row("inside", 0.10, pre_pos=[0.11, 0.13], post_pos=[0.11, 0.09])
    assert not feasible(inside, 0.12, 4.0)
    assert localization(inside, 0.15, 4.0) == (True, 0.0)
    # A start exactly at the bound gets the grace for its first steps rather than failing at once.
    at_bound = _profile_row("at_bound", 0.15, pre_pos=[0.16, 0.17, 0.12], post_pos=[0.16, 0.17, 0.12])
    ok, at = localization(at_bound, 0.15, 4.0)
    assert ok and at == 3.0


def test_select_candidate_is_lexicographic_over_feasible_rows_only():
    rows = [_row("fast_but_infeasible", 5.0, 10.0, 1, pos=0.5),
            _row("slow_short", 5.0, 30.0, 2),
            _row("fast_long", 9.0, 20.0, 0),
            _row("fast_long_more_probes", 9.0, 20.0, 3)]
    assert select_candidate(rows, 0.1, 4.0, "time")["candidate_id"] == "fast_long"
    assert select_candidate(rows, 0.1, 4.0, "distance")["candidate_id"] == "slow_short"
    assert select_candidate(rows, 0.1, 4.0, "probes")["candidate_id"] == "fast_long"
    assert select_candidate([rows[0]], 0.1, 4.0, "time") is None
    with pytest.raises(ValueError):
        select_candidate(rows, 0.1, 4.0, "bogus")


def test_pareto_keeps_ties_and_drops_dominated():
    rows = [_row("a", 1.0, 1.0, 1), _row("b", 1.0, 1.0, 1), _row("c", 2.0, 1.0, 1),
            _row("d", 0.5, 5.0, 1)]
    keep = {rows[i]["candidate_id"] for i in pareto_indices(rows, ("distance_m", "estimated_time_s"))}
    assert keep == {"a", "b", "d"}


def test_prune_never_returns_an_infeasible_schedule():
    """A synthetic model: each probe removes one unit of sigma, the bound needs
    at least two of the four probes. Whatever order is used, the result is
    feasible and no smaller feasible subset exists by single deletion."""
    dense = [dict(edge_index=0, step=s, contact="wall") for s in (1, 2, 3, 4)]

    def evaluate(schedule):
        sigma = 0.20 - 0.03 * len(schedule)
        return _profile_row("x", sigma, pre_pos=[sigma], post_pos=[sigma]), []

    for reverse in (False, True):
        schedule = prune_schedule(evaluate, dense, 0.15, 4.0, reverse)
        assert feasible(evaluate(schedule)[0], 0.15, 4.0)
        assert len(schedule) == 2
        for action in schedule:
            assert not feasible(evaluate(schedule - {action})[0], 0.15, 4.0)


def test_evaluate_route_is_json_serialisable_and_consistent():
    g = Graph3(None)
    a, b, c = g.add_node((0, 0), "start"), g.add_node((2, 0), "free"), g.add_node((2, 2), "goal")
    e1, e2 = g.add_edge(a, b, "FREE_TRANSIT"), g.add_edge(b, c, "FREE_TRANSIT")
    belief = PlannerBelief(minor=0, cov=np.eye(2) * 0.05**2, yaw_sigma=math.radians(2))
    cfg = DEFAULT.merged(require_landmark_for_mode_resolution=False, max_relaxations=0)
    timing = Timing(0.5, 1.0, 5.0)
    metrics, trace = evaluate_route(GraphBackend(g, a, c), [e1.eid, e2.eid], belief, cfg, 0.0, timing)
    json.dumps(dict(metrics=metrics, trace=trace), allow_nan=False)
    assert metrics["distance_m"] == pytest.approx(4.0)
    assert metrics["probe_count"] == 0            # no scene: no conceptual contact
    assert metrics["commanded_turn_rad"] == pytest.approx(math.pi / 2)
    assert metrics["estimated_time_s"] == pytest.approx(8.0 + math.pi / 2)
    assert metrics["association_resolved"]
    # Growth is monotone with no observations, so the peak is the final row.
    assert metrics["peak_position_sigma_m"] == pytest.approx(trace[-1]["max_position_sigma_m"])
    assert trace[-1]["edge_index"] == 1


def test_timing_sensitivity_only_reweights():
    rows = [_row("few_probes_long", 12.0, 0.0, 1, turn=0.0),
            _row("many_probes_short", 8.0, 0.0, 6, turn=0.0)]
    timing = Timing(1.0, 1.0, 5.0)
    out = timing_sensitivity(rows, [0.1], [4.0], timing, probe_seconds_values=(0.0, 1.0, 10.0))
    by_probe = {r["probe_seconds"]: r for r in out}
    assert by_probe[0.0]["fastest_candidate_id"] == "many_probes_short"
    assert by_probe[0.0]["fastest_is_shortest"]
    assert by_probe[10.0]["fastest_candidate_id"] == "few_probes_long"
    assert by_probe[10.0]["fastest_is_fewest_probes"]
    assert by_probe[1.0]["fastest_candidate_id"] == "few_probes_long"
    assert by_probe[1.0]["fastest_time_s"] == pytest.approx(12.0 + 1.0)
    # inputs are untouched
    assert [r["estimated_time_s"] for r in rows] == [0.0, 0.0]
