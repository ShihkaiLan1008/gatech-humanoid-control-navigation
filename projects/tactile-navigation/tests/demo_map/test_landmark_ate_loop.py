"""A landmark visit is driven by the EIG, and stops when the EIG says so.

The behaviour under test is the one the old loop did not have.  It ran
``while not any(press contacted)``: the first successful touch ended the
visit, `worth_pressing` was never called, and `rank` was handed a ONE-element
list -- so neither half of the ATE was consulted, and the recorded run
`out/demo_map/run/demo_run_s0.json` shows all eight landmarks at exactly
``tries=1 contacts=1``.

What is asserted here is that the visit now (a) offers a real candidate set,
(b) keeps going after a successful contact, (c) stops on the information
floor rather than on the budget, and (d) never presses on the strength of a
contact it could not physically make.

NO SIMULATOR.  The decision half of the visit -- `landmark_probes`,
`landmark_next_probe`, `feed_contact`/`feed_miss` -- is shipped code and runs
here against the real scanned map; the motion half (`_reacquire`,
`_press_station`, the force shield, the stance ladder) needs MuJoCo and is
NOT covered by these tests.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from demo_map.executor import (FOLLOW_STANDOFF, LANDMARK_MIN_SEPARATION_M,
                               DemoExecCfg, landmark_half_along,
                               landmark_next_probe, landmark_probes,
                               landmark_segment)
from demo_map.landmark_ate import AteProbe, LandmarkATE
from demo_map.scene_demo import build_demo_scene, demo_route, plan_steps_demo_room

ROOT = Path(__file__).resolve().parents[2]
ERR = np.array([0.06, -0.03])      # the robot pose error the oracle simulates


@pytest.fixture(scope="module")
def plan():
    """Compiled WITH the box landmarks, which the shipped scene no longer has.

    `mapv3.BOX_LANDMARKS` is False by owner directive -- a landmark is a
    corner or a cylinder, everything else is wall -- so `prot_*` is gone from
    the route.  The machinery it exercised is not: `LandmarkATE.box`, the
    reach guard, the multi-press stopping rule and the minimum separation all
    still ship, and a finite rectangular body is the case that discriminates
    them (a cylinder is the same from every angle, so it cannot show that a
    second press pins the along-face axis).  So these tests build the scene
    the flag would build, and are the reason the construction was kept."""
    from demo_map import mapv3, scene_demo
    was = mapv3.BOX_LANDMARKS
    mapv3.BOX_LANDMARKS = True
    try:
        # the PRE-MARKER room: the owner's goal pillar stands where the
        # end-room boxes' stances were (prot_10/12/14/16 all lose their
        # lane or base clearance to it), so the box machinery is measured
        # in the room whose boxes are still reachable
        scene = scene_demo.build_demo_scene(
            scene_demo.translated_scan_map(scene_demo.build_scan_map()))
        steps, _ = plan_steps_demo_room(DemoExecCfg(), scene, demo_route())
    finally:
        mapv3.BOX_LANDMARKS = was
    return scene, [s for s in steps if s["kind"] == "ate_landmark"]


def test_the_shipped_scene_has_only_corners_and_cylinders():
    """The directive itself, pinned: no box body reaches the route."""
    scene = build_demo_scene()
    kinds = {b.name.split("_")[0] for b in scene.plan.bodies}
    assert kinds == {"wall", "bulge"}, kinds
    assert not scene.spec.prots
    steps, _ = plan_steps_demo_room(DemoExecCfg(), scene, demo_route())
    for st in steps:
        if st["kind"] == "ate_landmark":
            assert st.get("shape") == "cylinder", st["label"]


@pytest.fixture(scope="module")
def ate_plan_module():
    """The offline validator, loaded so its episode simulator is the one
    under test -- a second copy of the loop would test itself, not the ship."""
    spec = importlib.util.spec_from_file_location(
        "ate_plan_loop", ROOT / "scripts" / "demo_room_ate_plan.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _step(steps, label):
    return next(s for s in steps if s["label"] == label)


# ── the candidate set ───────────────────────────────────────────────────────

def test_the_candidate_set_is_never_a_single_probe(plan):
    """`rank([probe])` is a no-op sort: with one candidate the EIG cannot
    express a preference, which is what the old pillar path handed it."""
    scene, lms = plan
    bodies = tuple(scene.plan.bodies)
    for st in lms:
        seg, s0 = landmark_segment(st)
        cands = landmark_probes(st, seg, s0, bodies,
                                standoff_m=FOLLOW_STANDOFF,
                                search_span_m=0.45)
        kinds = {c.kind for c in cands}
        assert len(cands) >= 10, (st["label"], len(cands))
        assert "press" in kinds and "edge" in kinds, (st["label"], kinds)
        # spatially distinct, and spanning the body rather than one point
        offs = sorted(c.s - s0 for c in cands)
        assert offs[-1] - offs[0] >= 2 * landmark_half_along(st, seg.tangent)


def test_the_ranking_discriminates_by_more_than_the_tie_break_bin(plan):
    """`LandmarkATE.rank` bins ties at 0.08 nats.  A candidate set whose
    scores all fall inside one bin is decided by the proximity tie-break, not
    by information -- that is the WALL result, and the whole claim of a
    landmark is that it is not a wall."""
    scene, lms = plan
    bodies = tuple(scene.plan.bodies)
    discriminating = 0
    for st in lms:
        seg, s0 = landmark_segment(st)
        ate = (LandmarkATE.box(st["label"], st["centre"], st["half_xy"],
                               st["yaw"])
               if st["shape"] == "box"
               else LandmarkATE.cylinder(st["label"], st["centre"],
                                         st["radius"]))
        cands = landmark_probes(st, seg, s0, bodies,
                                standoff_m=FOLLOW_STANDOFF,
                                search_span_m=0.45)
        ranked = ate.rank(list(cands), tie_break_xy=st["centre"])
        scoring = [q.score_nats for q in ranked if q.score_nats > 0.0]
        # a site can be left with a SINGLE reachable face -- the coherent-
        # walls flatten (2026-08-27) sank prot_26's backing board into the
        # south wall, so its edge probes' backstop now sits past the 0.43 m
        # reach and only the direct press scores.  One press is still ranked
        # correctly; it just cannot demonstrate rank-2, so the
        # discrimination claim is asserted on the sites that offer it...
        if len(scoring) >= 2:
            discriminating += 1
            assert max(scoring) - min(scoring) > 0.08, (st["label"],
                                                        scoring[:5])
    # ...and nearly every site must offer it, or the fixture has decayed
    # into wall-pressing
    assert discriminating >= len(lms) - 1, discriminating


def test_a_probe_is_not_scored_for_a_face_it_cannot_reach(plan):
    """When the mean pose says a probe misses, the scorer falls back to the
    closest approach -- which on a finite body can land on the box's SIDE and
    hand the press a normal square to its own approach.  Measured on
    `prot_10`: such a candidate scored 0.99 nats against 0.49 for a genuinely
    on-body one, and the pick it produced swept air."""
    scene, lms = plan
    st = _step(lms, "prot_10")
    seg, s0 = landmark_segment(st)
    ate = LandmarkATE.box(st["label"], st["centre"], st["half_xy"], st["yaw"])
    half = landmark_half_along(st, seg.tangent)
    t = np.asarray(seg.tangent, float)
    apex = seg.surface_point(s0 + 0.18)

    on_body = AteProbe("a", s0, apex, -np.asarray(seg.normal, float))
    far_off = AteProbe("a", s0 + 3 * half, apex + 3 * half * t,
                       -np.asarray(seg.normal, float))
    ate.rank([on_body, far_off])
    assert on_body.score_nats > 1.0
    assert far_off.score_nats == 0.0
    assert ate.reaches_what_it_is_scored_for(on_body)
    assert not ate.reaches_what_it_is_scored_for(far_off)

    # THE GRAZE IS NOT THROWN OUT WITH IT.  The fallback exists for probes
    # whose contact is uncertain, and that band is exactly where the contact
    # boolean is most informative -- so the guard has to cut the unreachable
    # tail without cutting the uncertain one.  Measured on `prot_10`: the
    # scoring band runs down to p_hit 0.55, and every zeroed candidate is one
    # the belief says the hand cannot reach.
    cands = landmark_probes(st, seg, s0, tuple(scene.plan.bodies),
                            standoff_m=FOLLOW_STANDOFF, search_span_m=0.45)
    ate.rank(list(cands), tie_break_xy=st["centre"])
    scoring = [q for q in cands if q.score_nats > 0.0]
    assert scoring, "the guard zeroed the whole set"
    assert min(q.terms.p_hit for q in scoring) < 0.7
    assert all(ate.reaches_what_it_is_scored_for(q) for q in scoring)
    assert all(not ate.reaches_what_it_is_scored_for(q)
               for q in cands if q.score_nats == 0.0)


# ── the loop ────────────────────────────────────────────────────────────────

def test_a_visit_does_not_stop_at_its_first_contact(plan, ate_plan_module):
    """THE REGRESSION THIS CHANGE IS ABOUT.  `prot_27` is 0.30 x 0.16: the
    first press pins the face normal and leaves the along-face axis at its
    prior, and the SECOND press is the one that pins it."""
    scene, lms = plan
    st = _step(lms, "prot_27")
    ep = ate_plan_module.simulate_episode(st, tuple(scene.plan.bodies),
                                          DemoExecCfg(), ERR, seed=0)
    assert ep["n_contacts"] >= 1
    assert ep["n_presses"] >= 2, ep
    first = ep["presses"][0]
    assert first["outcome"] == "contact"      # and the visit continued anyway
    assert len(ep["presses"]) > 1
    # the second probe is a different probe, by at least the separation the
    # sensor's own correlation length demands
    assert abs(ep["presses"][1]["offset_m"] - first["offset_m"]) >= \
        LANDMARK_MIN_SEPARATION_M
    # and the along-face axis, which one press cannot pin, ends up pinned
    assert first["sigma_before"][0] >= 0.09
    assert ep["sigma_final"][0] <= 0.02, ep["sigma_final"]


def test_every_visit_stops_on_information_not_on_the_budget(plan,
                                                            ate_plan_module):
    scene, lms = plan
    cfg = DemoExecCfg()
    eps = [ate_plan_module.simulate_episode(st, tuple(scene.plan.bodies),
                                            cfg, ERR, seed=0) for st in lms]
    assert all(e["stop"] != "budget_exhausted" for e in eps), \
        [(e["label"], e["stop"]) for e in eps]
    assert all(e["n_presses"] < max(cfg.landmark_press_budget,
                                    st["max_tries"])
               for e, st in zip(eps, lms)), \
        [(e["label"], e["n_presses"]) for e in eps]
    # the floor is what ends a visit that found its landmark
    felt = [e for e in eps if e["n_contacts"]]
    assert felt and all(e["stop"] == "eig_below_threshold" for e in felt)
    # and it is not a one-press rule dressed up: several visits press on
    assert max(e["n_presses"] for e in eps) >= 3


def test_max_tries_bounds_failures_and_not_successes(plan, ate_plan_module):
    """A successful informative probe must not consume the failure budget --
    `bulge_0` presses four times, all of them contacts, against a
    ``max_tries`` of 3."""
    scene, lms = plan
    st = _step(lms, "bulge_0")
    ep = ate_plan_module.simulate_episode(st, tuple(scene.plan.bodies),
                                          DemoExecCfg(), ERR, seed=0)
    assert st["max_tries"] == 3
    assert ep["n_contacts"] == ep["n_presses"] > st["max_tries"]
    assert ep["stop"] == "eig_below_threshold"


def test_a_visit_that_has_felt_nothing_keeps_looking(plan):
    """The information floor may not end a visit that has never touched its
    landmark: a body the belief has mislocated by more than its own
    half-width puts every candidate off it, and every off-body candidate
    scores zero once the reachability guard is in.  That reading is "look
    harder", not "nothing here"."""
    scene, lms = plan
    st = _step(lms, "prot_16")
    seg, s0 = landmark_segment(st)
    ate = LandmarkATE.box(st["label"], st["centre"], st["half_xy"], st["yaw"])
    # a belief displaced far enough that nothing on the body scores
    ate.belief.poses["box"].mean[:2] += 3.0 * np.asarray(seg.tangent, float)
    cands = landmark_probes(st, seg, s0, tuple(scene.plan.bodies),
                            standoff_m=FOLLOW_STANDOFF, search_span_m=0.45)
    best, ranked, why = landmark_next_probe(ate, cands, st["centre"],
                                            contacted=False)
    assert best is not None and why == "acquiring"
    assert max(q.score_nats for q in ranked) < ate.economy.landmark_floor_nats
    # once something HAS been felt, the same set ends the visit
    best2, _r2, why2 = landmark_next_probe(ate, cands, st["centre"],
                                           contacted=True)
    assert best2 is None and why2 == "eig_below_threshold"


# ── what must NOT have changed ──────────────────────────────────────────────

def test_flat_wall_following_is_still_rule_based():
    """A wall is rank 1, so its stations stay on the cheap cadence: no part
    of the followed-wall path may consult the landmark ATE."""
    import inspect

    from demo_map import executor as ex

    for fn in (ex.DemoRouteExecutor._follow, ex.DemoRouteExecutor._follow_gen,
               ex._stations_for):
        src = inspect.getsource(fn)
        assert "LandmarkATE" not in src and "landmark_probes" not in src
        assert "worth_pressing" not in src
    # and the cadence itself is untouched arithmetic: first station at the
    # acquire gate's 1 m run-in, then every 0.2 m up to `length - 0.45`
    assert ex._stations_for(6.0, 0.2) == [round(1.0 + k * 0.2, 2)
                                          for k in range(23)]


def test_the_ate_belief_is_fed_in_map_coordinates():
    """A tap reports its contact in the ROBOT's frame; a LandmarkATE belief
    lives in the map's.  The pillar path used to feed the body-frame point
    straight in, which is why every estimate in the recorded run collapsed
    onto the hand (~0.5, 0.0) for bodies spread from x = 1.6 m to 16.0 m."""
    import inspect
    import math
    from types import SimpleNamespace

    from demo_map.executor import DemoRouteExecutor, VisitCtx

    # the visit loop lives in VisitCtx now (P4 carve); `_ate_pillar` is the
    # degenerate propose/execute wrapper over the same implementation
    src = inspect.getsource(VisitCtx)
    assert "_map_frame_touch" in src
    assert 'ate.feed_contact(np.asarray(tap["p_noisy"]' not in src

    stub = SimpleNamespace(
        _believed_xy=lambda: np.array([10.0, 4.0]),
        _believed_yaw=lambda: math.pi / 2.0)
    tap = {"contacted": True, "p_body": (0.4, 0.0), "p_noisy": [0.4, 0.0],
           "n_noisy": [1.0, 0.0], "reason": "felt_on_approach"}
    p_map, n_map = DemoRouteExecutor._map_frame_touch(stub, tap)
    # facing +y, a hand 0.4 m ahead of the base is 0.4 m north of it
    assert np.allclose(p_map, [10.0, 4.4], atol=1e-9)
    assert np.allclose(n_map, [0.0, 1.0], atol=1e-9)


# ── the shipped loop, driven without a simulator ────────────────────────────

def _stub_executor(scene, oracle):
    """`DemoRouteExecutor._ate_pillar` with the motion half replaced.

    Everything the loop DECIDES with is real -- the candidate generator, the
    ranking, the stop rule, `_map_frame_touch`, `_swept_free`, the record it
    writes.  Only `_reacquire` and `_press_station`, which need MuJoCo, are
    stubbed.  The believed pose is put at the origin facing +x so the
    body-frame point a tap reports and the map point it composes to are the
    same number, and the frame composition is still the shipped one."""
    from types import SimpleNamespace

    from demo_map.executor import DemoRun, DemoRouteExecutor

    ex = object.__new__(DemoRouteExecutor)
    # the loop under test is the analytic one; the WM default would price
    # its probes from a checkpoint and a belief this stub does not carry
    ex.cfg = DemoExecCfg(ig_backend="analytic_marginal")
    ex.scene = scene
    ex.run_rec = DemoRun()
    ex.be = SimpleNamespace(last_press_f_mean_n=0.8, time_s=0.0)
    ex._sigma = lambda: 0.10
    ex._believed_xy = lambda: np.zeros(2)
    ex._believed_yaw = lambda: 0.0
    ex._reacquire = lambda *a, **k: "arrived"
    ex._press_station = oracle
    return ex


def _press_oracle(scene, err, seed=0):
    """A kinematic press: the hand commanded to a map point arrives ``err``
    short of it, and what the belief reads is composed back through the
    believed pose -- so a contact reads shifted by ``+err``."""
    from execution.wall_lane import raycast

    rng = np.random.default_rng(seed)
    bodies = tuple(scene.plan.bodies)
    calls = []

    def press(cand, label, standoff, style, **kw):
        from demo_map.executor import HAND_LEAD_M

        seg, s_station = cand.target.ref, cand.target.s_m
        n = np.asarray(seg.normal, float)
        lane = seg.lane_point(float(s_station) + HAND_LEAD_M, standoff) - err
        d, nn, hit = raycast(lane, -n, standoff + 0.08, bodies)
        body = label.split(":", 1)[-1]
        calls.append(float(s_station))
        if d is None or bodies[hit].name != body:
            return ({"contacted": False, "free_body": [(0.4, 0.0)],
                     "press_pose_map": (0.0, 0.0, 0.0), "peak_force_n": 1.0,
                     "reason": "bounded_free_sweep"}, "ok")
        p = lane - float(d) * n + err
        p = p + rng.normal(0.0, 0.005, size=2)
        return ({"contacted": True, "p_body": tuple(p), "p_noisy": list(p),
                 "n_noisy": list(np.asarray(nn, float)),
                 "reason": "felt_on_approach", "peak_force_n": 9.0}, "ok")

    return press, calls


def test_the_shipped_loop_presses_prot_27_more_than_once(plan):
    """END TO END on the real geometry, with only the motion stubbed: the
    visit must not end at its first contact, and the record must say the EIG
    ended it."""
    scene, lms = plan
    st = _step(lms, "prot_27")
    oracle, calls = _press_oracle(scene, ERR)
    ex = _stub_executor(scene, oracle)

    assert ex._ate_pillar(st) == "ok"
    rec = ex.run_rec.steps[-1]
    assert rec["status"] == "confirmed"
    assert rec["stop"] == "eig_below_threshold"
    assert rec["n_presses"] >= 2 and rec["n_contacts"] >= 1
    assert rec["presses"][0]["ate"] == "contact"     # and it kept going
    assert len(calls) == rec["n_presses"] >= 2
    assert abs(calls[1] - calls[0]) >= LANDMARK_MIN_SEPARATION_M - 1e-9
    # Every press was chosen from a real set, and the set discriminated.
    # The set SHRINKS as the visit advances and that is the constraint
    # working: `RouteRatchet` drops candidates behind the furthest point
    # already reached, so the second press cannot be offered the first
    # press's ground.  What must survive is the original regression -- a
    # one-element list, which consults neither half of the ATE.
    counts = [e["n_candidates"] for e in rec["presses"]]
    assert counts[0] >= 8, counts
    assert all(c >= 2 for c in counts), counts
    assert counts == sorted(counts, reverse=True), (
        "a later press may not be offered MORE ground than an earlier one")
    assert any(e["nats_spread"] > 0.08 for e in rec["presses"])
    # the axis one press cannot pin
    assert rec["sigma0"][0] >= 0.09
    assert rec["estimate"]["sigma"][0] <= 0.02


def test_the_shipped_loop_stops_itself_on_every_demo_room_landmark(plan):
    scene, lms = plan
    cfg = DemoExecCfg()
    seen = {}
    for st in lms:
        budget = max(cfg.landmark_press_budget, st["max_tries"])
        oracle, _calls = _press_oracle(scene, ERR)
        ex = _stub_executor(scene, oracle)
        ex._ate_pillar(st)
        rec = ex.run_rec.steps[-1]
        seen[st["label"]] = (rec["n_presses"], rec["stop"])
        assert rec["stop"] != "budget_exhausted", (st["label"], rec["stop"])
        assert rec["n_presses"] < budget, (st["label"], rec["n_presses"])
    assert max(n for n, _ in seen.values()) >= 3, seen
    assert any(s == "eig_below_threshold" for _n, s in seen.values())
