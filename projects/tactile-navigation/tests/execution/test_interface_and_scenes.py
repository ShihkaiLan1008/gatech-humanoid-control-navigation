"""The interface contract, the firewall, and the scenes' physical legality.

Every test here runs without a GPU or a simulator.  The ones that matter most
are the firewall tests: "no planner input leaks simulator ground truth" is an
acceptance criterion, and a criterion that is only asserted in prose is not
checked at all.
"""

import math

import numpy as np
import pytest

from execution.interface import (
    TRUTH_VOCABULARY,
    ExecStatus,
    GlobalDirective,
    LocalReport,
    ProbeOutcome,
    RouteDecision,
    TruthLog,
    assert_no_truth_leak,
)
from execution.scenes import HALF_LENGTHS, THICKNESS, SCENES, build_scene
from execution.stations import REACH_M


# ── the ground-truth firewall ────────────────────────────────────────────────────


def _report(**kw) -> LocalReport:
    base = dict(odom_delta_body=(0.1, 0.0), odom_delta_yaw=0.0,
                probes=(ProbeOutcome("contact", (0.34, 0.0), (-1.0, 0.0),
                                     ((0.1, 0.0),), 6.0, 0.12, "L0s1"),),
                ate_belief={"terminus_mean_m": 1.0}, status=ExecStatus(True))
    base.update(kw)
    return LocalReport(**base)


def test_a_clean_report_passes_the_firewall():
    assert_no_truth_leak(_report())


def test_the_firewall_catches_a_leaking_field():
    """The guard has to actually fire, or it is decoration."""
    leaky = _report(ate_belief={"true_terminus_m": 1.0})
    with pytest.raises(AssertionError, match="ground-truth leak"):
        assert_no_truth_leak(leaky)


@pytest.mark.parametrize("bad", ["true_xy", "world_xy", "sim_pose", "gt_contact",
                                 "oracle_s", "wall_pose"])
def test_every_truth_word_is_caught(bad):
    with pytest.raises(AssertionError):
        assert_no_truth_leak(_report(ate_belief={bad: 1.0}))


def test_local_report_has_no_field_that_could_carry_a_pose():
    """Structural, not incidental: a realized contact is a BODY-frame point plus
    an odometry increment.  Turning that into a world pose needs something the
    local side is never given."""
    fields = set(LocalReport.__dataclass_fields__)
    assert fields == {"odom_delta_body", "odom_delta_yaw", "probes",
                      "ate_belief", "status", "station"}
    for f in fields:
        assert not any(w in f for w in TRUTH_VOCABULARY)


def test_probe_outcome_is_body_frame_only():
    fields = set(ProbeOutcome.__dataclass_fields__)
    assert "p_body" in fields and "swept_free_body" in fields
    assert not any("world" in f or "_w" in f for f in fields)


def test_truth_log_is_never_a_belief_input():
    """TruthLog exists and holds world state -- so the test is that the belief
    entry point will not take it."""
    from execution.executive import Executive
    import inspect
    sig = inspect.signature(Executive._update_global)
    assert list(sig.parameters)[1] == "report"
    log = TruthLog()
    log.sample(t=0.0, xy=(1.0, 2.0), yaw=0.0, mode="LOCOMOTION")
    assert log.as_arrays()["xy"].shape == (1, 2)


def test_invalid_probes_are_not_informative():
    """A sweep that stopped short is a controller event, not a measurement."""
    bad = ProbeOutcome("invalid", None, None, (), 0.0, 0.12, "L0s1", "short")
    assert not bad.informative
    assert ProbeOutcome("contact", (0.3, 0.0), (-1.0, 0.0), (), 6.0, 0.1).informative
    assert ProbeOutcome("swept_free", None, None, ((0.2, 0.0),), 0.0, 0.1).informative


# ── the interface carries what the batch spec says it must ──────────────────────


def test_global_directive_carries_the_specified_outputs():
    d = GlobalDirective(route_edge=1, waypoint_map_xy=(1.0, 0.0), waypoint_yaw_map=0.0,
                        station="L0s1", station_map_xy=(1.0, 0.0),
                        hypotheses=({"mass": 1.0, "s": 0.0, "xy": (0.0, 0.0)},),
                        probe_budget=4, allowed_decisions=(RouteDecision.PROBE,))
    for key in ("route_edge", "waypoint_map_xy", "station", "n_hypotheses",
                "probe_budget", "allowed_decisions"):
        assert key in d.to_dict()


def test_local_report_carries_the_specified_returns():
    r = _report().to_dict()
    for key in ("odom_delta_body", "odom_delta_yaw", "n_contacts", "n_swept_free",
                "ate_belief", "ok", "owner"):
        assert key in r


# ── scenes: physical legality ────────────────────────────────────────────────────


@pytest.mark.parametrize("name", sorted(SCENES))
def test_scene_matches_the_shipped_wall_asset(name):
    """A scene the asset cannot build is not a scene."""
    sc = build_scene(name)
    assert len(sc.walls) <= 6
    assert len({w.index for w in sc.walls}) == len(sc.walls), "one box per index"
    for w in sc.walls:
        assert 0 <= w.index < len(HALF_LENGTHS)
        assert w.half_length == HALF_LENGTHS[w.index]


def test_the_asset_constants_are_what_the_scenes_assume():
    """Runs only where mjlab is importable; the scene constants are a COPY of
    the asset's, so a drift between them must fail somewhere."""
    pytest.importorskip("mjlab")
    from rl.g1_assets.maze_wall import MAZE_WALL_HALF_LENGTHS, MAZE_WALL_THICKNESS
    assert tuple(MAZE_WALL_HALF_LENGTHS) == HALF_LENGTHS
    assert MAZE_WALL_THICKNESS == THICKNESS


@pytest.mark.parametrize("name", sorted(SCENES))
def test_every_station_is_within_the_measured_reach(name):
    """ACCEPTANCE: every selected station and probe is reachable.

    Checked ray by ray against the map, not asserted: a station whose every ray
    is out of reach is a probe the robot cannot execute and the planner must
    never be able to bank on.
    """
    sc = build_scene(name)
    for node, st in sc.stations.items():
        assert st.reach_m <= REACH_M + 1e-9, node
        assert st.reachable(sc.plan, 0.0), node
        for ray in st.predict(sc.plan, 0.0):
            if ray["hit"]:
                assert ray["range"] <= REACH_M, (node, ray)


@pytest.mark.parametrize("name", sorted(SCENES))
def test_stations_stand_clear_of_the_walls(name):
    """A station the base cannot occupy is not a stance."""
    sc = build_scene(name)
    for node, st in sc.stations.items():
        for body in sc.plan.bodies:
            sd, _ = body.signed_distance(st.xy.reshape(1, 2), None)
            assert float(sd[0]) > 0.17, (node, body.name, float(sd[0]))


def test_two_turns_really_has_two_consecutive_turns():
    """ACCEPTANCE: at least two route decisions in one episode."""
    sc = build_scene("two_turns")
    assert sc.n_route_decisions >= 2
    assert len(sc.plan.edges) == 3
    for a, b in zip(sc.legs, sc.legs[1:]):
        turn = abs(math.degrees(math.atan2(
            float(np.cross(a["u"], b["u"])), float(np.dot(a["u"], b["u"])))))
        assert turn == pytest.approx(90.0, abs=1.0), "each decision is a real corner"


def test_every_leg_terminus_is_bracketed_by_some_station():
    """The fan spans 0.06-0.30 m ahead, so a terminus is only observable from a
    0.24 m window.  If no station sits in it the belief can never be anchored and
    the branch gate cannot open -- measured, before terminus-relative stations
    were added."""
    for name in SCENES:
        sc = build_scene(name)
        if sc.withholds_terminus_evidence:
            continue                 # declared property -- see test_detour_* below
        for k, leg in enumerate(sc.legs):
            if k == len(sc.legs) - 1:
                continue          # the last leg ends at the goal, not at a turn
            term = float(leg["terminus"])
            ok = any(st.s_m + min(st.fan_offsets_m) <= term <= st.s_m + max(st.fan_offsets_m)
                     for st in sc.stations.values() if st.leg == k)
            assert ok, f"{name} leg{k} terminus {term} unbracketed"


# ── the ambiguous scene is actually ambiguous ───────────────────────────────────


def test_the_approach_is_aliased_and_the_near_edge_is_not():
    """The scene's whole claim, station by station: on the way to the corner the
    two registrations predict identical presses; behind the start they do not."""
    sc = build_scene("ambiguous_near_edge")
    sep = sc.prior_modes[1][0] - sc.prior_modes[0][0]
    assert sep == pytest.approx(0.55)

    forward = [st for st in sc.stations.values()
               if st.leg == 0 and 0.0 <= st.s_m <= 0.75]
    assert forward, "there is an approach to be aliased"
    for st in forward:
        assert st.discrimination(sc.plan, 0.0, sep) < 0.6, st.name

    back = sc.stations[sc.disambiguating_stations[0]]
    assert back.s_m < 0.0, "the disambiguator is BEHIND the start"
    assert back.discrimination(sc.plan, 0.0, sep) >= 0.9


def test_the_ambiguity_is_bigger_than_the_commit_tolerance():
    """If the two hypotheses were closer than the turn tolerance the scene would
    not punish guessing, and the benchmark would prove nothing."""
    from execution.executive import ExecConfig
    sc = build_scene("ambiguous_near_edge")
    sep = sc.prior_modes[1][0] - sc.prior_modes[0][0]
    assert sep > 2 * ExecConfig().branch_sigma_max_m


def test_the_unambiguous_scene_has_no_second_hypothesis():
    assert build_scene("two_turns").prior_modes is None


# ── the landmark-detour scene ────────────────────────────────────────────────────


def test_detour_scene_has_no_on_route_evidence_at_all():
    """ACCEPTANCE: all touches along the shortest route are aliased.

    Checked exhaustively -- every on-route station, every ray, both hypotheses.
    If any on-route stance discriminated, the planner could resolve without
    leaving the route and the scene would not test what it claims to.
    """
    sc = build_scene("detour_box")
    sep = sc.prior_modes[1][0] - sc.prior_modes[0][0]
    on_route = [st for st in sc.stations.values() if st.leg == 0 and st.s_m >= 0.0]
    assert len(on_route) >= 4
    for st in on_route:
        assert st.discrimination(sc.plan, 0.0, sep) == 0.0, st.name
        assert all(r["hit"] for r in st.predict(sc.plan, 0.0)), st.name


def test_detour_scene_is_resolvable_only_off_route():
    """ACCEPTANCE: only an off-route box disambiguates, and reaching it is a
    detour AWAY from the goal."""
    sc = build_scene("detour_box")
    sep = sc.prior_modes[1][0] - sc.prior_modes[0][0]
    box = sc.stations[sc.disambiguating_stations[0]]
    assert box.discrimination(sc.plan, 0.0, sep) >= 0.9
    assert box.s_m < 0.0, "the landmark is behind the start"
    assert abs(box.s_m) > 1.0, "and reaching it is a real detour"
    # The thing it touches is the LANDMARK, not the followed wall: the box's
    # face stands proud of the wall plane, so the contact range is measurably
    # SHORTER than any flat-wall touch.  (Identified geometrically, not by
    # signature: every body deliberately carries signature "surface" -- an
    # anonymous touch must be explainable by any of them, and a "box"-signed
    # body was measured invisible to the association, which resolved the
    # belief 0.47 m wrong on 8 of 12 seeds.)
    wall_range = sc.standoff_m
    hits = [r for r in box.predict(sc.plan, 0.0) if r["hit"]]
    assert hits and min(r["range"] for r in hits) <= wall_range - 0.05
    assert all(r["signature"] == "surface" for r in hits)
    assert box.reachable(sc.plan, 0.0)


def test_detour_scene_punishes_guessing_physically():
    """Committing on the wrong hypothesis turns while the wall is still there."""
    sc = build_scene("detour_box")
    sep = sc.prior_modes[1][0] - sc.prior_modes[0][0]
    term = float(sc.legs[0]["terminus"])
    assert sep > 0.4, "the error must be bigger than the arrival band"
    # under the far hypothesis the commit lands short, alongside wall 0
    commit_short = term + sc.standoff_m - sep
    body = sc.plan.bodies[0]
    lo, hi = body.aabb
    assert lo[0] <= commit_short <= hi[0], "an early commit is still beside wall 0"
