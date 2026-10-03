"""Reach compliance and the analytic station signatures.

The first test IS an acceptance criterion: every probe station on every stressed
map respects the measured 0.45 m reach, checked ray by ray against the map.
"""

import math

import numpy as np
import pytest

from global_nav.affordance import (
    FANS,
    LANE_OFFSET_M,
    REACH_M,
    g1_fan,
    predict_rays,
    ray_disagreement,
)
from global_nav.stressmaps import build_scenario


@pytest.fixture(scope="module", params=["twin_junctions", "drift_gauntlet"])
def scn(request):
    return build_scenario(request.param)


# ── acceptance: reach ────────────────────────────────────────────────────────────


def test_every_fan_respects_the_measured_reach():
    for fan in FANS.values():
        assert fan.max_reach_m <= REACH_M + 1e-9
    with pytest.raises(ValueError):
        g1_fan("too_long", (0.0,), reach=REACH_M + 0.05)


def test_every_station_probe_is_reachable(scn):
    """Every ray of every station either CONTACTS inside 95% of reach or is an
    intended swept-free ray.  No station banks on a touch it cannot make."""
    for name, st in scn.graph.stations.items():
        for ray in st.predictions(scn.plan):
            if ray["hit"]:
                assert ray["range"] <= 0.95 * REACH_M, (scn.name, name, ray)


def test_feature_stations_touch_their_feature(scn):
    feature_stations = [s for s in scn.graph.stations.values() if s.purpose == "feature"]
    assert feature_stations, "each stressed map carries a deliberate disambiguator"
    for st in feature_stations:
        nominal = st.nominal(scn.plan)
        assert any(r["hit"] for r in nominal["rays"])
        sigs = set(nominal["reachable_surfaces"])
        assert sigs & {"pillar", "pillar_corner", "box", "box_corner"}, sigs


def test_wall_lane_offset_keeps_oblique_rays_in_reach():
    # the geometric contract behind every wall station
    assert LANE_OFFSET_M / math.sin(math.radians(50.0)) <= REACH_M


# ── the analytic signatures ──────────────────────────────────────────────────────


def test_twin_approach_is_aliased_and_the_pillar_is_not():
    """The benchmark's central claim, checked station by station: along the
    eastward approach NO station distinguishes the two hypotheses; the pillar
    distinguishes them from both sides."""
    sc = build_scenario("twin_junctions")
    D = sc.junction_spacing_m
    for name in ("n30", "n40", "n50", "n62", "n70", "n78", "n85"):
        st = sc.graph.stations[name]
        assert st.discrimination(sc.plan, (+D, 0.0)) < 0.6, name
    for name in ("n62", "n70", "n78"):                    # the approach under MAP=far
        st = sc.graph.stations[name]
        assert st.discrimination(sc.plan, (-D, 0.0)) < 0.6, name
    p = sc.graph.stations["p55"]
    assert p.discrimination(sc.plan, (+D, 0.0)) >= 0.6
    assert p.discrimination(sc.plan, (-D, 0.0)) >= 0.6


def test_discrimination_is_jitter_robust():
    """A knife-edge raycast difference at a slab boundary must not count: the
    planner once bet a branch on disagreement that existed at delta = 3.01 and
    not at 3.00."""
    sc = build_scenario("twin_junctions")
    st = sc.graph.stations["n62"]
    assert st.discrimination(sc.plan, (3.01, 0.0)) < 0.6
    assert st.discrimination(sc.plan, (2.99, 0.0)) < 0.6


def test_anchor_flags_match_rank_one_physics():
    """A station on a featureless wall cannot fix along-route position; a
    station bracketing a terminus or facing the box can."""
    sc = build_scenario("drift_gauntlet")
    g = sc.graph
    assert not g.stations["n90"].anchors_s(sc.plan, (1.0, 0.0))     # plain wall
    assert g.stations["n100"].anchors_s(sc.plan, (1.0, 0.0))        # gap edge
    assert g.stations["n114"].anchors_s(sc.plan, (1.0, 0.0))        # divider
    assert g.stations["p65"].anchors_s(sc.plan, (1.0, 0.0))         # the box


def test_gauntlet_dealiasing_station_breaks_the_junction_alias():
    """The junction furniture is aliased at the gap spacing BY DESIGN; n135 east
    of the cluster is the lane evidence that breaks it."""
    sc = build_scenario("drift_gauntlet")
    spacing = sc.junction_spacing_m                                  # 1.6
    aliased = sc.graph.stations["n106"]
    assert aliased.discrimination(sc.plan, (-spacing, 0.0)) < 0.6 or \
        aliased.discrimination(sc.plan, (spacing, 0.0)) < 0.6
    dealias = sc.graph.stations["n135"]
    assert dealias.discrimination(sc.plan, (-spacing, 0.0)) >= 0.6


def test_ray_disagreement_grades_contact_free_signature_range():
    hit = {"hit": True, "range": 0.30, "normal": np.array([0.0, -1.0]), "signature": "pillar"}
    free = {"hit": False, "range": 0.45, "normal": None, "signature": None}
    other_sig = dict(hit, signature="surface")
    far = dict(hit, range=0.44)
    assert ray_disagreement(hit, free) == 1.0
    assert ray_disagreement(hit, other_sig) == 1.0
    assert ray_disagreement(hit, far) == 0.8
    assert ray_disagreement(free, free) == 0.0
    assert ray_disagreement(hit, dict(hit)) == 0.0


def test_predictions_relabel_signatures_near_features_like_the_sensor_does():
    sc = build_scenario("twin_junctions")
    rays = predict_rays(sc.plan, (6.2, 0.52), 0.0, FANS["left_g1"])
    sigs = {r["signature"] for r in rays if r["hit"]}
    assert "gap_edge" in sigs, "a touch at a terminus reports the terminus"


# ── graph sanity ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("name", ["twin_junctions", "drift_gauntlet"])
def test_goal_is_reachable_only_through_the_route_branch(name):
    sc = build_scenario(name)
    path, dist = sc.graph.shortest(
        sc.graph.nearest_node(sc.prior_mean), sc.goal_node)
    assert path and math.isfinite(dist)
    branch_edges = [sc.graph.edge_between(a, b) for a, b in zip(path, path[1:])]
    branches = {e.branch_junction for e in branch_edges if e and e.branch_junction}
    assert branches == set(sc.route_junctions), \
        "every path to the goal must commit exactly the route junction"
