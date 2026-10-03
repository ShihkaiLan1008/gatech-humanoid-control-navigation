"""The base never goes back along the route."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from demo_map.executor import (DemoExecCfg, RouteRatchet, leg_is_free_space,
                               tag_gaits)
from demo_map.landmark_ate import AteProbe
from demo_map.scene_demo import build_demo_scene, demo_route, plan_steps_demo_room


def _probe(seg, s):
    return AteProbe(seg, float(s), np.zeros(2), np.array([1.0, 0.0]))


def test_a_target_behind_the_furthest_reached_is_refused():
    r = RouteRatchet()
    assert r.allows("north", 1.0)
    r.advance("north", 1.0)
    assert r.allows("north", 1.4)
    assert r.allows("north", 1.0)              # standing still is not reversing
    assert not r.allows("north", 0.6)
    r.advance("north", 1.4)
    assert not r.allows("north", 1.0)


def test_the_tolerance_cannot_re_admit_the_previous_station():
    """It is slack for arithmetic -- a station and the probe that lands on it
    are the same place computed two ways -- not slack for the planner."""
    r = RouteRatchet()
    r.advance("north", 2.0)
    assert r.allows("north", 2.0 - r.tol + 1e-9)
    assert r.tol <= 0.25 * DemoExecCfg().press_spacing_m
    assert not r.allows("north", 2.0 - DemoExecCfg().press_spacing_m)


def test_a_face_the_route_has_left_is_closed():
    """Rank-1 is all a flat face gives and the rhythm already took it every
    20 cm, so going back for more is walking for nothing."""
    r = RouteRatchet()
    r.advance("west", 3.0)
    r.advance("north", 0.5)
    assert not r.allows("west", 9.9)
    assert r.allows("north", 0.7)


def test_it_filters_probes_before_anything_scores_them():
    """Dropped as INFEASIBLE, not overruled after ranking -- so the EIG still
    picks the most informative of what is reachable."""
    r = RouteRatchet()
    r.advance("north", 2.0)
    keep = r.filter([_probe("north", 1.0), _probe("north", 2.5),
                     _probe("north", 3.0)])
    assert [q.s for q in keep] == [2.5, 3.0]


def test_the_ratchet_never_slips_when_the_belief_does():
    """THE CASE THE OLD RULE MISSED.  The rhythm refused to walk back by
    comparing to where the belief says it is now; let the belief slip
    backwards and that comparison slips with it.  A max cannot."""
    r = RouteRatchet()
    for s in (1.0, 1.2, 1.4):
        r.advance("north", s)
    believed_now = 0.9                          # the belief slid back 0.5 m
    assert not r.allows("north", believed_now + 0.2)
    assert r.reached("north") == 1.4


def test_every_leg_of_the_demo_room_gets_a_gait_and_only_the_mouth_walks():
    scene = build_demo_scene()
    steps, _ = plan_steps_demo_room(DemoExecCfg(), scene, demo_route())
    gaits = {s.get("label"): s.get("gait")
             for s in steps if s["kind"] == "goto"}
    assert None not in gaits.values(), gaits
    walked = [k for k, v in gaits.items() if v == "walk"]
    # R5 may be split by a route_around via (the 0.61 tip+wander clearance
    # lifts one); the via is the same mouth leg and rightly walks too
    assert walked and all(k.startswith("R5") for k in walked), gaits


def test_a_leg_beside_a_wall_is_not_free_space():
    scene = build_demo_scene()
    bodies = tuple(scene.plan.bodies)
    steps, _ = plan_steps_demo_room(DemoExecCfg(), scene, demo_route())
    seg = next(s["seg"] for s in steps if s["kind"] == "follow")
    a = np.asarray(seg.lane_point(0.5, 0.49), float)
    b = np.asarray(seg.lane_point(min(seg.length, 3.0), 0.49), float)
    assert not leg_is_free_space(a, b, bodies)
    # ...and a leg out into the middle of nothing is
    far = a + np.array([0.0, 40.0])
    assert leg_is_free_space(a + np.array([0.0, 30.0]), far, bodies)


def test_tagging_is_idempotent():
    scene = build_demo_scene()
    steps, _ = plan_steps_demo_room(DemoExecCfg(), scene, demo_route())
    before = [s.get("gait") for s in steps if s["kind"] == "goto"]
    tag_gaits(steps, scene)
    assert [s.get("gait") for s in steps if s["kind"] == "goto"] == before


def test_a_landmark_visit_may_bracket_its_body_back_to_front():
    """A visit is ONE point on the route, not a walk along it.

    Its probes bracket the body -- past the apex pins the face normal, before
    it pins the along-face axis -- and in EIG order the second is often
    behind the first.  Advancing the ratchet between them refuses that as
    reversing, which it is not: the base has returned to no ground it had
    covered when the visit began.  Measured cost of getting it wrong: the
    along-face sigma stopped converging, 0.092 m against 0.02.
    """
    r = RouteRatchet()
    r.advance("north", 4.0)               # the route has reached s = 4.0
    gate = r.gate("north")

    assert gate(_probe("north", 4.6))     # past the apex, taken first
    assert gate(_probe("north", 4.2))     # ...then BEHIND it, still admitted
    assert not gate(_probe("north", 3.5))  # but not behind where we started
    assert not gate(_probe("west", 9.0))   # nor on another face

    # taking them does not move the route until the visit says so
    assert r.reached("north") == 4.0
    r.advance("north", 4.6)
    assert r.reached("north") == 4.6
    assert not r.gate("north")(_probe("north", 4.2))


def test_a_landmark_visit_may_bracket_its_body_back_to_front():
    """A visit is ONE point on the route, not a walk along it.

    Its probes bracket the body -- past the apex pins the face normal, before
    it pins the along-face axis -- and in EIG order the second is often
    behind the first.  Advancing the ratchet between them refuses that as
    reversing, which it is not: the base has returned to no ground it had
    covered when the visit began.  Measured cost of getting it wrong: the
    along-face sigma stopped converging, 0.092 m against 0.02.
    """
    r = RouteRatchet()
    r.advance("north", 4.0)               # the route has reached s = 4.0
    gate = r.gate("north")

    assert gate(_probe("north", 4.6))     # past the apex, taken first
    assert gate(_probe("north", 4.2))     # ...then BEHIND it, still admitted
    assert not gate(_probe("north", 3.5))  # but not behind where we started
    assert not gate(_probe("west", 9.0))   # nor on another face

    # taking them does not move the route until the visit says so
    assert r.reached("north") == 4.0
    r.advance("north", 4.6)
    assert r.reached("north") == 4.6
    assert not r.gate("north")(_probe("north", 4.2))
