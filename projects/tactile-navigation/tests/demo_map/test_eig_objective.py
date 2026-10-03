"""P2: the landmark loop is an objective, not a script."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from demo_map.executor import (BASE_CLEAR_M, TURN_RADIUS_M, corner_turn_path,
                               stance_feasible)
from demo_map.economy import drift_sigma
from demo_map.landmark_ate import AteProbe, LandmarkATE
from tactile_explore.infogain import EIGTerms


def _probe(seg, s, score):
    """A probe carrying REAL terms: `score_nats` reads the outcome-marginal
    total, so a stub that only carried the hit branch would drift out of
    sync with the scorer it stands in for.  ig_miss = 0 keeps these
    fixtures' scores exactly the `score` each test asks for."""
    q = AteProbe(seg, float(s), np.zeros(2), np.array([1.0, 0.0]))
    q.terms = EIGTerms(ig_pose=score, ig_hyp=0.0, cost_s=1.0, p_hit=0.9,
                       ig_miss=0.0)
    return q


POSE_SIGMA_M = 0.12          # the robot's own spread at the stance


def _utilities(ate, probes, progress):
    """The REAL cost arithmetic, not a copy of it.

    This helper used to re-implement `rank`'s motion/backtrack/risk terms, so
    the tests could go on passing while the scorer they stand for changed
    underneath them -- which is exactly what happened when the drift price
    moved onto the robot's pose sigma.  It now calls `rank` and only supplies
    the inputs, with the probe geometry bypassed (`_probe` already carries the
    terms these tests want to reason about)."""
    ate.reaches_what_it_is_scored_for = lambda p: True
    ate.score_probe = lambda p: (p.candidate, p.terms)
    return ate.rank(probes, progress_from=progress,
                    pose_sigma_m=POSE_SIGMA_M)


def test_backtracking_pays_and_equal_information_prefers_forward():
    ate = LandmarkATE.cylinder("x", (0.0, 0.0), 0.2)
    fwd = _probe("f", 1.4, 2.0)
    back = _probe("f", 0.6, 2.0)
    _utilities(ate, [fwd, back], {"f": 1.0})
    assert fwd.utility > back.utility          # same nats, forward wins
    # ...but information that CLEARLY justifies it still buys a backtrack
    rich_back = _probe("f", 0.6, 2.0 + 1.2)
    _utilities(ate, [rich_back], {"f": 1.0})
    assert rich_back.utility > fwd.utility


def test_the_stop_rule_is_utility_in_the_black():
    ate = LandmarkATE.cylinder("x", (0.0, 0.0), 0.2)
    q = _probe("f", 1.0, ate.economy.landmark_floor_nats + 0.05)
    q.utility = -0.01                          # information, but not worth it
    assert not ate.worth_pressing(q)
    q.utility = 0.4
    assert ate.worth_pressing(q)


def test_the_concave_window_is_discovered_by_the_base_circle():
    """No CORNER_REACH_M: what stops candidates near a concave vertex is
    that the base circle no longer fits between the two walls.  Checked on a
    synthetic L so the boundary is the term's own doing -- room2's compiled
    face is pre-trimmed 0.58 m short of its vertex (a legacy clearance that
    this term makes redundant), so nothing binds there."""
    from global_nav.geometry import ConvexPolygon

    wall_a = ConvexPolygon([(-4, -0.1), (0.1, -0.1), (0.1, 0), (-4, 0)],
                           "wall_a", "surface")     # face y=0, vertex at 0,0
    wall_b = ConvexPolygon([(0, -0.1), (0.1, -0.1), (0.1, 4), (0, 4)],
                           "wall_b", "surface")     # face x=0, rising north
    feas = [bool(stance_feasible((x, 0.38), (wall_a, wall_b), {"wall_a"},
                                 BASE_CLEAR_M))
            for x in np.arange(-2.0, -0.02, 0.11)]
    assert feas[0], "far from the vertex the stance must be fine"
    assert not feas[-1], "hard against the other wall it must not be"
    k = feas.index(False)
    assert all(not f for f in feas[k:]), "one boundary, discovered once"
    # and the discovered boundary is the geometry, not a tuned number
    x_bound = -2.0 + 0.11 * k
    assert abs(abs(x_bound) - BASE_CLEAR_M) < 0.12


def test_the_corner_turn_is_a_rolling_arc_on_both_corner_types():
    for na, nb in (((1, 0), (0, -1)),          # room2's concave L
                   ((0, -1), (-1, 0))):        # the demo room's convex branch
        path = corner_turn_path((0.0, 0.0), na, nb, off=0.40)
        assert len(path) >= 3
        pts = np.array([p for p, _ in path])
        steps = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        assert steps.min() > 0.05, "every waypoint moves the base"
        for (p0, y0), (p1, y1) in zip(path, path[1:]):
            d = p1 - p0                        # heading tracks travel:
            travel = math.atan2(d[1], d[0])    # never an in-place spin
            mid = 0.5 * (y0 + y1) if abs(y1 - y0) < math.pi else travel
            assert abs((travel - mid + math.pi) % (2 * math.pi) - math.pi) \
                < 0.45
        # and it really is an arc of at least the minimum turn radius:
        # every chord step subtends radius >= TURN_RADIUS_M
        for (p0, y0), (p1, y1) in zip(path, path[1:]):
            dyaw = abs((y1 - y0 + math.pi) % (2 * math.pi) - math.pi)
            if dyaw > 1e-6:
                r = np.linalg.norm(p1 - p0) / (2 * math.sin(dyaw / 2))
                assert r >= TURN_RADIUS_M - 1e-6


def test_wall_touch_spacing_emerges_from_the_objective():
    """P3: no 20 cm schedule.  The utility must (a) have an interior optimum
    -- the risk term prices walking long with a drifting belief, so neither
    'press immediately, always' nor 'one distant touch' wins -- and (b) push
    the press FORWARD when the local arc is already covered."""
    from types import SimpleNamespace

    from demo_map.executor import DemoExecCfg, DemoRouteExecutor
    from execution.wall_lane import WallSegment

    ex = object.__new__(DemoRouteExecutor)
    # the ANALYTIC objective is what this file asserts; the deployment
    # default is the WM, which would price these stations from a checkpoint
    # and a belief this stub does not carry
    ex.cfg = DemoExecCfg(ig_backend="analytic_marginal")
    ex.scene = SimpleNamespace(plan=SimpleNamespace(bodies=()))
    ex.be = SimpleNamespace(mj=SimpleNamespace(
        vf_touch_target_m=0.43, vf_press_stance_m=0.36,
        probe_hand_z_m=0.25, park_x_m=0.14))
    ex._cov_xy = lambda: np.array([[0.013 ** 2, 0.0], [0.0, 0.013 ** 2]])
    seg = WallSegment(p0=np.zeros(2), tangent=np.array([1.0, 0.0]),
                      normal=np.array([0.0, -1.0]), length=8.0, name="w")
    import demo_map.executor as X
    orig = X.face_present
    X.face_present = lambda *a, **k: True
    try:
        cand, j, trace = ex._next_wall_press(seg, 1.0, covered=[])
        assert cand is not None and j > ex.cfg.economy.wall_floor_nats
        # the wall generator emits the SAME structure the corner and the
        # landmark do -- a stance, an arm, a tip target and what it is about
        assert cand.target.primitive == "wall_face"
        assert cand.end_effector in ("left", "right")
        assert cand.approach_dir_body == (1.0, 0.0)
        s_c = float(cand.target.s_m)
        gap = s_c - 1.0
        # <= 1.6: bounded by the objective's own curvature, not by a
        # schedule.  The unified lambda_risk = 2.5 (16 priced a miss risk
        # feeler-first removed, and kills every visit's rank-2 second
        # press) stretches THIS scenario's optimum to 1.25 m because its
        # post-contact sigma is tiny (0.013); live runs re-decide from
        # real sigma each time and their measured spacing barely moves
        # (lr3_s0: 0.64-0.82 m).  The far-edge assertion below is the
        # real invariant: the horizon must not win.
        assert 0.15 <= gap <= 1.6, f"spacing {gap} came from the objective"
        js = dict(trace)
        far = max(js)
        assert js[far] < j, "the far edge must NOT win: risk prices it"
        # a covered stretch pushes the choice forward
        c2, _, _ = ex._next_wall_press(seg, 1.0,
                                       covered=[(0.9, s_c + 0.10)])
        assert c2 is None or float(c2.target.s_m) > s_c + 0.05
    finally:
        X.face_present = orig


def _wall_j(sigma_m, reach, stance):
    """J of the immediate station under one belief sigma and one reach."""
    from types import SimpleNamespace

    import demo_map.executor as X
    from demo_map.executor import DemoExecCfg, DemoRouteExecutor
    from execution.wall_lane import WallSegment

    ex = object.__new__(DemoRouteExecutor)
    # the ANALYTIC objective is what this file asserts; the deployment
    # default is the WM, which would price these stations from a checkpoint
    # and a belief this stub does not carry
    ex.cfg = DemoExecCfg(ig_backend="analytic_marginal")
    ex.scene = SimpleNamespace(plan=SimpleNamespace(bodies=()))
    ex.be = SimpleNamespace(mj=SimpleNamespace(vf_touch_target_m=reach,
                                               vf_press_stance_m=stance))
    ex._cov_xy = lambda: (sigma_m ** 2) * np.eye(2)
    seg = WallSegment(p0=np.zeros(2), tangent=np.array([1.0, 0.0]),
                      normal=np.array([0.0, -1.0]), length=8.0, name="w")
    orig = X.face_present
    X.face_present = lambda *a, **k: True
    try:
        return ex._next_wall_press(seg, 1.0, covered=[])
    finally:
        X.face_present = orig


def test_a_press_the_belief_expects_to_miss_is_not_priced_as_a_hit():
    """The correction itself: EIG = p_hit*IG_hit + (1-p_hit)*IG_miss.

    demo_room v14 stood with the wall 1.5 m away and swept air twenty-three
    times while this objective kept returning ~2.4 nats, because the score
    asked only "if I touch, what is this worth?".  Reach enters the score
    now, so a station the belief cannot reach is worth strictly less than
    the same station in reach -- with the SAME sigma, so the difference is
    the outcome model and nothing else."""
    _c_in, j_in, _ = _wall_j(0.28, reach=0.43, stance=0.36)      # margin +0.07
    _c_out, j_out, _ = _wall_j(0.28, reach=0.43, stance=0.60)    # margin -0.17
    assert j_out < j_in, (j_out, j_in)


def test_a_certain_miss_is_worth_nothing_and_a_certain_hit_is_the_hit_branch():
    """The two limits that say the weighting is a probability, not a fudge."""
    from demo_map.economy import miss_truncation_ig, unified_ig

    n = np.array([0.0, 1.0])
    sig, sig_t = 0.05, 0.018
    hit = unified_ig(n, sig ** 2 * np.eye(2), sig_t ** 2)

    def eig(margin):
        p = 0.5 * (1.0 + math.erf(margin / (sig * math.sqrt(2.0))))
        return p * hit + (1.0 - p) * miss_truncation_ig(margin, sig)

    # a -> -inf: the sweep certainly ends in air, and "air where I already
    # knew there was air" truncates nothing -- the whole score goes to zero
    assert miss_truncation_ig(-1.0, sig) < 1e-6
    assert eig(-1.0) < 1e-6
    # a -> +inf: p_hit -> 1, so the marginal score collapses onto IG_hit
    assert abs(eig(1.0) - hit) < 1e-6
    # and in between it is a genuine average: bounded by the branches
    assert 0.0 < eig(0.0) < hit


def _rung_ex(sigma_m, reach=0.43, feel_cap=0.35):
    from types import SimpleNamespace

    from demo_map.executor import DemoExecCfg, DemoRouteExecutor

    ex = object.__new__(DemoRouteExecutor)
    ex.cfg = DemoExecCfg()
    ex.be = SimpleNamespace(mj=SimpleNamespace(vf_touch_target_m=reach,
                                               vf_feel_cap_m=feel_cap))
    ex._cov_xy = lambda: (sigma_m ** 2) * np.eye(2)
    return ex


def _seg_y():
    from execution.wall_lane import WallSegment

    return WallSegment(p0=np.zeros(2), tangent=np.array([1.0, 0.0]),
                       normal=np.array([0.0, -1.0]), length=8.0, name="w")


def test_one_press_already_sweeps_the_cap_plus_the_reach():
    """The geometry the ladder was reasoning without: the tip creeps to the
    reach and then the BASE creeps in by the feel cap behind it, so a single
    press covers ~0.78 m of belief error and a 0.035 m rung adds 4%."""
    ex = _rung_ex(0.2)
    assert ex._sweep_margin_m(0.485) == pytest.approx(0.43 - 0.135, abs=1e-9)
    # the 0.05 believed-gap floor: starting closer cannot sweep deeper
    assert ex._sweep_margin_m(0.30) == pytest.approx(0.38, abs=1e-9)
    # one press's total depth against the rung's step
    assert ex._sweep_margin_m(0.485) + 0.485 > 0.75


def test_a_rung_that_re_sweeps_cleared_ground_is_worth_nothing():
    """v14/v16's whole ladder in one assertion.  The press that just missed
    started at believed 0.485 and cleared out to margin 0.295; the base then
    wandered back out, so the next rung would sweep to 0.331 -- and after
    the drift it re-sweeps 0.295 or less.  Conditioned on the miss, that
    rung cannot contact, so its J is zero and the ladder must stop."""
    ex, seg = _rung_ex(0.20), _seg_y()
    ruled = ex._sweep_margin_m(0.485)
    # the base drifted back OUT: the next press starts no closer than the
    # last one did, so its sweep ends no deeper
    j, p, m = ex._rung_worth(seg, 0.52, ruled)
    assert m <= ruled + 1e-9
    assert p == 0.0
    assert j < ex.cfg.economy.wall_floor_nats
    # ...while a rung that genuinely reaches past what is cleared is worth
    # taking, on the SAME belief -- the difference is the sweep, not sigma
    shallow = ex._sweep_margin_m(0.60)          # a ladder that cleared less
    j2, p2, m2 = ex._rung_worth(seg, 0.45, shallow)
    assert m2 > shallow and p2 > 0.0
    assert j2 > ex.cfg.economy.wall_floor_nats


def test_the_plain_probability_is_the_one_that_lies():
    """Why the loop looked worth continuing.  These are v14's rung 5,
    verbatim from out/demo_room_v14: the first press cleared out to margin
    0.295, the base drifted back, and the rung would sweep to 0.297 -- two
    millimetres of new ground.  The plain Phi read 0.50 and the loop ran
    five more times; the sliver it would actually add holds 0.1%."""
    from demo_map.economy import unswept_hit_mass

    sigma, ruled, margin = 0.200, 0.295, 0.297
    plain = 0.5 * (1.0 + math.erf(margin / (sigma * math.sqrt(2.0))))
    assert plain > 0.4                        # what the retry loop assumed
    assert unswept_hit_mass(margin, ruled, sigma) < 0.005  # what is true
    # a first rung, with nothing yet ruled out, keeps the full value
    assert unswept_hit_mass(margin, -9.9, sigma) == pytest.approx(plain,
                                                                 abs=1e-9)


def test_a_ladder_past_the_beliefs_own_envelope_has_nothing_left_to_spend():
    """The renormalisation that was tried and refuted.  v17 swept to 2.4
    sigma over ten rungs, so the belief kept 0.9% of its mass out there --
    but dividing by that surviving tail ("given the face IS deeper, will I
    find it") read p = 0.25-0.91 and the ladder ran to its cap anyway.  The
    unnormalised mass is what says "nothing left out here"."""
    from demo_map.economy import unswept_hit_mass

    sigma, ruled, margin = 0.12, 0.285, 0.331     # v17's rung 2, verbatim
    mass = unswept_hit_mass(margin, ruled, sigma)
    tail = 1.0 - 0.5 * (1.0 + math.erf(ruled / (sigma * math.sqrt(2.0))))
    assert tail < 0.02                            # the belief is spent
    assert mass < 0.01                            # so the rung is worthless
    assert mass / tail > 0.4                      # ...but renormalised it
    #                                               reads as a coin flip


def test_lost_wall_triggers_a_deep_recovery_feel_then_gives_up_honestly():
    """demo_room_v2 measured the missing behavior: belief broke after the
    corner and R6 was pressed 53 times over ~900 s with the true wall
    1.8-2.3 m away.  Two straight misses must now buy ONE deep feel whose
    cap the belief authorises (3 sigma + lane + margin), and two failed
    recoveries must end the segment instead of grinding."""
    from types import SimpleNamespace

    import numpy as np

    from demo_map.executor import DemoExecCfg, DemoRouteExecutor
    from execution.wall_lane import WallSegment

    ex = object.__new__(DemoRouteExecutor)
    ex.cfg = DemoExecCfg()
    ex.run_rec = SimpleNamespace(touches=[], replans=[])
    ex.be = SimpleNamespace(time_s=0.0, mj=SimpleNamespace(
        vf_touch_target_m=0.43, probe_hand_z_m=0.25, park_x_m=0.14))
    ex.scene = SimpleNamespace(plan=SimpleNamespace(bodies=()))
    ex._wall_cover = {}
    ex._believed_xy = lambda: np.array([0.0, 0.0])
    ex._cov_xy = lambda: np.array([[0.25, 0.0], [0.0, 0.25]])  # sigma 0.5
    seg = WallSegment(p0=np.zeros(2), tangent=np.array([1.0, 0.0]),
                      normal=np.array([0.0, -1.0]), length=8.0, name="w")
    calls = {"press": 0, "recover": 0}

    def fake_next(seg_, s_now, covered):
        return ex._candidate(seg_, 1.0 + 0.7 * calls["press"]), 1.0, []
    def fake_follow(seg_, stations, label, **kw):
        if stations:
            calls["press"] += 1
            ex.run_rec.touches.append({"contacted": False})
        return "arrived"
    def fake_press_station(cand, label, *a, **kw):
        calls["recover"] += 1
        assert label.startswith("recover:")
        cap = cand.meta.get("feel_cap_m")
        assert cap and cap >= 3 * 0.5
        # the recovery is an ACQUISITION rung, and the one representation
        # has to say so rather than leaving it in a loose kwarg
        assert cand.target.kind == "search"
        assert cand.meta["press_standoff_m"] == 0.05
        ex.run_rec.touches.append({"contacted": False})   # recovery fails
        return None, "ok"
    ex._next_wall_press = fake_next
    ex._follow = fake_follow
    ex._press_station = fake_press_station

    ex._follow_eig(seg, "w:rhythm")
    reasons = [r["reason"] for r in ex.run_rec.replans]
    assert reasons.count("lost_wall_recovery") == 2
    assert "wall_lost_giving_up" in reasons
    # the loop pressed 2 before the first recovery and 2 between them,
    # never the 53 the measured run burned
    assert calls["press"] <= 5 and calls["recover"] == 2
