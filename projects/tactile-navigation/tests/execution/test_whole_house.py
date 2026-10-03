"""Whole-house demo scenes: structure verified analytically, behaviour on the
frozen loop verified on the stand-in.

The scenes are built strictly from the six-box asset and the frozen
batch3a2 machinery -- no new planning method, no new evidence type the
stance window is not already validated on.  What IS new is scale (10-14 m,
3-4 turns) and the chain geometry, and the chain was measured biting twice:
a four-leg spiral put its last wall across the detour corridor, and the
five-leg spiral wrapped its last wall 0.12 m from the START.  The clearance
test here is what keeps that class of bug out.
"""

import numpy as np
import pytest

from execution.backends import BackendConfig, KinematicBackend
from execution.executive import ExecConfig, Executive
from execution.scenes import HALF_LENGTHS, build_scene
from execution.stations import REACH_M

HOUSES = ("nominal_multiroom", "ambiguous_landmark_house", "multi_decision_house")


def _run(scene, policy="belief_aware", seed=0):
    sc = build_scene(scene)
    be = KinematicBackend(sc, BackendConfig(), np.random.default_rng(seed))
    e = Executive(sc, be, ExecConfig(seed=seed, policy=policy, n_particles=900,
                                     episode_budget_s=1400, max_probes_total=40))
    e.verbose = False
    return e, e.run()


# ── structure ────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("name", HOUSES)
def test_house_fits_the_six_box_asset(name):
    sc = build_scene(name)
    assert len(sc.walls) <= 6
    assert len({w.index for w in sc.walls}) == len(sc.walls), "each box used once"
    for w in sc.walls:
        assert w.half_length == HALF_LENGTHS[w.index]


@pytest.mark.parametrize("name,legs,turns,min_route", [
    ("nominal_multiroom", 4, 3, 10.0),
    ("ambiguous_landmark_house", 4, 3, 9.0),
    ("multi_decision_house", 5, 4, 12.0),
])
def test_house_scale(name, legs, turns, min_route):
    sc = build_scene(name)
    assert len(sc.legs) == legs
    assert sc.n_route_decisions == turns
    route = sum(float(np.linalg.norm(np.asarray(e.b) - np.asarray(e.a)))
                for e in sc.plan.edges)
    assert route >= min_route


@pytest.mark.parametrize("name", HOUSES)
def test_every_station_is_within_the_measured_reach(name):
    sc = build_scene(name)
    for st in sc.stations.values():
        assert max(abs(o) for o in st.fan_offsets_m) <= st.reach_m
        assert st.reach_m <= REACH_M + 1e-9


@pytest.mark.parametrize("name", HOUSES)
def test_the_chain_does_not_wrap_onto_the_route(name):
    """Start pose, every leg's lane, and every station approach stand clear of
    every body.  This is the self-intersection guard: measured failures were a
    wall slab at y = -0.42 across the detour corridor and a wall 0.12 m from
    the start."""
    sc = build_scene(name)
    def clearance(p):
        p = np.asarray(p, float).reshape(1, 2)
        return min(float(body.signed_distance(p, None)[0][0])
                   for body in sc.plan.bodies)
    # 0.38 IS the design: the followed wall's face stands one standoff away
    assert clearance((0.0, 0.0)) >= 0.35, "the start must be in open floor"
    for k, leg in enumerate(sc.legs):
        a = np.asarray(leg["origin"], float)
        u, v = np.asarray(leg["u"], float), np.asarray(leg["v"], float)
        # the ROUTE's span: the robot never walks past the goal on the last leg
        e = sc.plan.edges[k]
        span = float(np.linalg.norm(np.asarray(e.b) - np.asarray(e.a)))
        for s in np.arange(0.0, max(span, 0.3), 0.15):
            for lane in (0.0, -0.18):
                c = clearance(a + s * u + lane * v)
                assert c >= 0.27, (leg, round(s, 2), lane, round(c, 3))
    for n, st in sc.stations.items():
        approach = st.xy + (0.06 - st.face_offset_m) * st.facing
        assert clearance(approach) >= 0.27, n


# ── evidence structure ───────────────────────────────────────────────────────────


def test_ambiguous_house_on_route_probes_are_aliased():
    """Every on-route leg-0 station reads identically under both registrations
    -- the branch cannot be resolved without leaving the route."""
    sc = build_scene("ambiguous_landmark_house")
    sep = sc.prior_modes[1][0] - sc.prior_modes[0][0]
    for n, st in sc.stations.items():
        if st.leg == 0 and st.purpose == "wall":
            assert st.discrimination(sc.plan, 0.0, sep) == 0.0, n
    assert sc.withholds_terminus_evidence


def test_ambiguous_house_only_the_slab_discriminates():
    sc = build_scene("ambiguous_landmark_house")
    sep = sc.prior_modes[1][0] - sc.prior_modes[0][0]
    discs = {n: sc.stations[n].discrimination(sc.plan, 0.0, sep)
             for n in sc.disambiguating_stations}
    assert max(discs.values()) >= 0.9, discs
    best = max(discs, key=discs.get)
    st = sc.stations[best]
    assert st.s_m < 0.0, "reaching the landmark means leaving the route"
    assert st.reachable(sc.plan, 0.0)
    assert st.face_offset_m > 0.0, "the slab face stands proud; approaches account for it"


def test_multi_decision_house_offers_two_evidence_types():
    """Decision 1: leg-0 terminus brackets (wall-end evidence).  Decision 2:
    leg 2's own brackets are withheld and the mid-leg bump steps the contact
    range at its edges -- a different evidence type at a different place."""
    sc = build_scene("multi_decision_house")
    term0 = float(sc.legs[0]["terminus"])
    brackets0 = [n for n, st in sc.stations.items()
                 if st.leg == 0 and abs(st.s_m - term0) <= 0.30]
    assert brackets0, "leg-0 terminus is bracketed"
    bump = [n for n, st in sc.stations.items() if n.startswith("BUMP")]
    assert bump
    for n in bump:
        rays = sc.stations[n].predict(sc.plan, 0.0)
        ranges = [r["range"] for r in rays if r["hit"]]
        assert max(ranges) - min(ranges) >= 0.06, \
            "the bump edge must step the contact range"


# ── behaviour on the frozen loop (stand-in) ──────────────────────────────────────


def test_nominal_multiroom_completes_three_turns():
    best = 0
    for seed in range(6):
        _, r = _run("nominal_multiroom", seed=seed)
        best = max(best, r["n_route_decisions"])
        assert not r["wrong_turn"]
        if r["arrived"] and r["n_route_decisions"] >= 3:
            return
    pytest.fail(f"no seed of 6 completed 3 turns (best {best})")


def test_ambiguous_house_belief_aware_beats_guessing():
    """ACCEPTANCE (logic half): same map, same prior -- belief-aware detours,
    resolves, arrives; no-touch guesses and wrong-turns."""
    ba = [_run("ambiguous_landmark_house", "belief_aware", s)[1] for s in range(6)]
    nt = [_run("ambiguous_landmark_house", "no_touch", s)[1] for s in range(6)]
    assert np.mean([r["arrived"] for r in ba]) >= 0.6
    assert not any(r["wrong_turn"] for r in ba)
    assert all(r["landmark_resolved"] for r in ba if r["arrived"])
    assert all(r["detour_path_m"] >= 0.5 for r in ba if r["arrived"])
    assert np.mean([r["wrong_turn"] for r in nt]) >= 0.5
    assert all(r["n_probes"] == 0 for r in nt)


def test_multi_decision_house_needs_two_tactile_decisions():
    """At least one seed shows the full pattern: an early probe visit whose
    evidence unblocks turn 1, and a later leg-2 visit whose evidence unblocks
    a later turn -- plan, move, touch, update, replan, twice."""
    for seed in range(6):
        _, r = _run("multi_decision_house", seed=seed)
        if not (r["arrived"] and r["n_route_decisions"] >= 4):
            continue
        probes_by_leg = {}
        for p in r["probes"]:
            probes_by_leg.setdefault(p["leg"], []).append(p)
        if 0 not in probes_by_leg or 2 not in probes_by_leg:
            continue
        sc = build_scene("multi_decision_house")
        bump_region = [p for p in probes_by_leg[2]
                       if sc.stations.get(p["station"]) is not None
                       and sc.stations[p["station"]].face_offset_m > 0]
        if bump_region:
            return                # the second decision used the bump's evidence
    pytest.fail("no seed of 6 arrived via leg-0 evidence AND bump-region evidence")
