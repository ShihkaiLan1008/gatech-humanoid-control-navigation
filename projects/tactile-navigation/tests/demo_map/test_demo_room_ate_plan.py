"""The demo room compiled with room2's planner logic.

room2's compiler says: ATE owns the landmarks, flat runs keep the rule-based
cadence, arrival is confirmed by touching something that is not a wall.  These
tests hold that sentence to the scanned demo room, where "the landmarks" is 36
non-wall bodies instead of two and has to be derived rather than named.
"""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from demo_map import landmarks as LM
from demo_map.executor import FOLLOW_STANDOFF, DemoExecCfg, DemoRouteExecutor
from demo_map.landmark_ate import AteProbe, LandmarkATE
from demo_map.scene_demo import build_demo_scene, demo_route, plan_steps_demo_room
from execution.wall_lane import face_present, raycast

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def compiled():
    scene = build_demo_scene()
    route = demo_route()
    steps, report = plan_steps_demo_room(DemoExecCfg(), scene, route)
    return scene, route, steps, report


@pytest.fixture(scope="module")
def compiled_with_boxes():
    """The scene `mapv3.BOX_LANDMARKS` would build.

    The shipped scene has only corners and cylinders now, so a test about how
    a BOX's press reach is derived has nothing to look at -- and this file's
    own guard says so out loud rather than passing vacuously.  A cylinder
    cannot stand in: it is the same from every angle, which is exactly the
    property the box case does not have."""
    from demo_map import mapv3, scene_demo
    was = mapv3.BOX_LANDMARKS
    mapv3.BOX_LANDMARKS = True
    try:
        # the PRE-MARKER room: the owner's goal pillar stands where the
        # end-room boxes' stances were, so the box machinery is measured in
        # the room whose boxes are still reachable
        scene = scene_demo.build_demo_scene(
            scene_demo.translated_scan_map(scene_demo.build_scan_map()))
        route = demo_route()
        steps, report = plan_steps_demo_room(DemoExecCfg(), scene, route)
    finally:
        mapv3.BOX_LANDMARKS = was
    return scene, route, steps, report


# ── the harvester ────────────────────────────────────────────────────────────

def test_harvest_takes_only_non_wall_bodies(compiled):
    scene, route, _steps, _rep = compiled
    sites, skipped = LM.harvest(scene, route.polyline(0.05),
                                standoff_m=FOLLOW_STANDOFF)
    assert sites, "the demo room has reachable landmarks"
    seen = {s.body for s in sites} | {x["body"] for x in skipped}
    assert not any(n.startswith("wall_") for n in seen)
    # every non-wall body is accounted for: visited, or skipped WITH a reason
    non_wall = {b.name for b in scene.plan.bodies if not b.name.startswith("wall_")}
    assert seen == non_wall
    assert all(x["reason"] for x in skipped)


def test_every_harvested_site_can_actually_be_pressed(compiled):
    scene, route, _steps, _rep = compiled
    bodies = tuple(scene.plan.bodies)
    names = [b.name for b in bodies]
    sites, _ = LM.harvest(scene, route.polyline(0.05), standoff_m=FOLLOW_STANDOFF)
    for s in sites:
        n = np.asarray(s.n_out, float)
        d, _nn, hit = raycast(np.asarray(s.stance_xy, float), -n,
                              FOLLOW_STANDOFF + 0.08, bodies)
        assert d is not None and names[hit] == s.body, s.body
        assert s.detour_m <= LM.MAX_DETOUR_M
        assert s.lane_m >= 0.90, f"{s.body} has no lane to press from"


def test_the_press_reach_is_measured_not_derived_from_half_extents(
        compiled_with_boxes):
    """A box's half-DEPTH is its reach only when the box is square to the
    wall, and a scanned protrusion rarely is.  Deriving it aimed prot_10's
    whole apex segment 71 mm inside the body."""
    scene, _route, steps, _rep = compiled_with_boxes
    bodies = tuple(scene.plan.bodies)
    names = [b.name for b in bodies]
    off = []
    for st in steps:
        if st["kind"] != "ate_landmark":
            continue
        n = np.asarray(st["normal"], float)
        lane = np.asarray(st["centre"], float) + st["reach_m"] * n \
            + FOLLOW_STANDOFF * n
        d, _nn, hit = raycast(lane, -n, FOLLOW_STANDOFF + 0.08, bodies)
        assert d is not None and names[hit] == st["label"], st["label"]
        # the stance really does sit one standoff off the face it aims at
        assert abs(float(d) - FOLLOW_STANDOFF) < 0.01, \
            f"{st['label']} presses from {float(d):.3f} m, not {FOLLOW_STANDOFF}"
        if st["shape"] == "box":
            off.append(abs(st["reach_m"] - float(st["half_xy"][1])))
    assert off and max(off) > 0.02, \
        "at least one scanned box is not square to its wall -- if none is, " \
        "this test has stopped guarding anything"


def test_a_bulge_is_pressed_on_a_face_whose_lane_reaches_it(compiled):
    """The route BOWS around a bulge, so its flank is nearer the drawn line
    than its apex -- ranking stances by detour picked the flank, whose press
    lane runs INTO the wall the bulge grows out of.

    Asserted as the lesson rather than as a compass direction.  This used to
    demand `n_out == -y` for every cylinder, which held only while the lane
    stood at 0.49; at the deployed 0.38 the harvester reaches `bulge_2` from
    a different face, and that is a legitimate re-selection, not the defect.
    What must never come back is a stance whose ray meets something else
    first."""
    scene, route, _steps, _rep = compiled
    sites, _ = LM.harvest(scene, route.polyline(0.05), standoff_m=FOLLOW_STANDOFF)
    bodies = tuple(scene.plan.bodies)
    names = [b.name for b in bodies]
    seen = 0
    for s in sites:
        if s.kind != "cylinder":
            continue
        seen += 1
        n = np.asarray(s.n_out, float)
        lane = np.asarray(s.centre, float) + (s.reach_m + FOLLOW_STANDOFF) * n
        d, _nn, hit = raycast(lane, -n, FOLLOW_STANDOFF + s.reach_m + 0.08,
                              bodies)
        assert d is not None, f"{s.body}: nothing under the press ray"
        assert names[hit] == s.body, (
            f"{s.body} pressed on a flank -- the ray meets {names[hit]} first")
    # three scanned cylinders plus the owner-placed goal_post at the goal
    assert seen == 4, f"expected the room's four cylinders, saw {seen}"


# ── the compiled plan ────────────────────────────────────────────────────────

def test_every_reachable_landmark_is_visited_exactly_once(compiled):
    scene, route, steps, report = compiled
    sites, _ = LM.harvest(scene, route.polyline(0.05), standoff_m=FOLLOW_STANDOFF)
    visits = [s["label"] for s in steps if s["kind"] == "ate_landmark"]
    assert sorted(visits) == sorted(s.body for s in sites)
    assert len(visits) == len(set(visits)), "a landmark scheduled twice"
    assert report["goal_landmark"] in visits


def test_visits_run_in_route_order(compiled):
    scene, route, steps, _rep = compiled
    sites, _ = LM.harvest(scene, route.polyline(0.05), standoff_m=FOLLOW_STANDOFF)
    order = {s.body: s.s_route for s in sites}
    seen = [order[s["label"]] for s in steps if s["kind"] == "ate_landmark"]
    assert seen == sorted(seen)


def test_exactly_one_landmark_is_required_and_it_is_the_goal_one(compiled):
    _scene, _route, steps, report = compiled
    req = [s["label"] for s in steps
           if s["kind"] == "ate_landmark" and s["required"]]
    assert req == [report["goal_landmark"]]


def test_identifying_the_goal_marker_is_the_last_touch(compiled):
    """The marker is a HALF-CYLINDER PROUD OF THE DESTINATION WALL, abeam the
    goal (owner, 2026-08-31), where it used to stand on the goal centre.

    What survives the move is the ordering claim, which is the one that
    matters: the required visit is the last TOUCH the route takes, so arrival
    is declared after a measurement and never from dead reckoning.  What
    changes is that the goal centre is no longer occupied, so the short walk
    to it comes back -- exactly what `marker_on_goal` decides, and room2 has
    always ended this way."""
    scene, _route, steps, report = compiled
    i_last = max(i for i, s in enumerate(steps)
                 if s["kind"] in ("ate_landmark", "ate_corner", "follow"))
    last = steps[i_last]
    assert last["kind"] == "ate_landmark" and last["required"]
    assert last["label"] == report["goal_landmark"]
    assert last["shape"] == "cylinder"
    # abeam the goal, ON the wall face: half of it stands in the room
    centre = np.asarray(last["centre"], float)
    goal = np.asarray(scene.goal_map_xy, float)
    assert abs(centre[0] - goal[0]) < 1e-6, "the marker is abeam the goal"
    assert 0.4 < float(goal[1] - centre[1]) < 1.0, \
        "the marker sits on the wall the goal stands off, not on the goal"
    # ...and everything after it is the walk to the now-unoccupied centre
    for s in steps[i_last + 1:]:
        assert s["kind"] == "goto", s["kind"]


def test_the_route_corner_is_compiled_as_a_convex_landmark(compiled):
    _scene, _route, steps, _rep = compiled
    corners = [s for s in steps if s["kind"] == "ate_corner"]
    assert len(corners) == 1
    c = corners[0]
    assert c["convex"] is True, "the demo room's route corner is convex"
    assert len(c["faces"]) == 2
    a, b = (np.asarray(f["face_n"], float) for f in c["faces"])
    assert abs(float(a @ b)) < 1e-6, "the two faces must be orthogonal"
    assert c["between"], "the turn still has to be walked"


def test_every_rhythm_station_has_a_face_to_press(compiled):
    scene, _route, steps, _rep = compiled
    bodies = tuple(scene.plan.bodies)
    for st in steps:
        if st["kind"] != "follow":
            continue
        for s in st["stations"]:
            assert face_present(st["seg"], float(s), bodies,
                                standoff_m=FOLLOW_STANDOFF), \
                f"{st['label']} station {s} presses air"


def test_the_flat_runs_still_keep_the_rule_based_cadence(compiled):
    _scene, _route, steps, report = compiled
    assert report["n_rhythm"] > 20, "the feeling-forward cadence must survive"
    spacings = []
    for st in steps:
        if st["kind"] == "follow" and len(st["stations"]) > 1:
            spacings += list(np.diff(st["stations"]))
    assert spacings and max(spacings) <= 0.25 + 1e-6


def test_the_plan_validator_finds_nothing_wrong():
    spec = importlib.util.spec_from_file_location(
        "ate_plan", ROOT / "scripts" / "demo_room_ate_plan.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.main(["--spacing", "0.2"]) == 0


# ── the ATE models the room needed ───────────────────────────────────────────

def test_convex_corner_puts_the_material_where_the_material_is():
    """`LandmarkATE.corner`'s own docstring warns that its concave model
    "put both pressable faces on the wrong side" of a convex vertex.  The demo
    room's only route corner IS convex, so the claim is checkable here: ask
    each model whether the two places the robot actually stands are solid."""
    from tactile_explore.belief import shape_sd

    vertex = np.array([14.027, 11.183])
    n_a, n_b = np.array([-1.0, 0.0]), np.array([0.0, 1.0])
    wall = np.array([14.30, 10.90, 0.15])        # inside the branch wall slab
    corridor = np.array([13.60, 10.90, 0.15])    # the lane the robot walks up
    end_room = np.array([14.30, 11.60, 0.15])    # the lane it leaves along

    def sd_at(ate, p):
        g = ate.belief.poses["corner"]
        sp = ate.belief.specs["corner"]
        return float(shape_sd(sp.shape, g.mean, sp.template_half_xy, p))

    cvx = LandmarkATE.convex_corner("c", vertex, n_a, n_b)
    cav = LandmarkATE.corner("c", vertex, n_a, n_b)
    assert sd_at(cvx, wall) < 0.0
    assert sd_at(cvx, corridor) > 0.0 and sd_at(cvx, end_room) > 0.0
    # the concave model calls BOTH lanes solid -- there is no free space left
    # for a hand to reach the faces through
    assert sd_at(cav, corridor) < 0.0 and sd_at(cav, end_room) < 0.0


def test_convex_corner_candidates_clear_the_information_floor():
    vertex = np.array([14.027, 11.183])
    n_a, n_b = np.array([-1.0, 0.0]), np.array([0.0, 1.0])
    ate = LandmarkATE.convex_corner("c", vertex, n_a, n_b)
    probes = ([AteProbe("A", d, vertex + np.array([0.0, -d]), -n_a)
               for d in (0.15, 0.30, 0.45)]
              + [AteProbe("B", d, vertex + np.array([d, 0.0]), -n_b)
                 for d in (0.15, 0.30, 0.45)])
    ranked = ate.rank(probes, tie_break_xy=vertex)
    assert ate.worth_pressing(ranked[0]) and ranked[0].score_nats > 1.0
    # inside the corner reach the scores tie, so the proximity tie-break still
    # pulls the pick toward the vertex rather than out along a face
    assert ranked[0].s <= 0.30


def test_convex_corner_reports_the_vertex_not_the_box_centre():
    vertex = np.array([14.027, 11.183])
    a = LandmarkATE.convex_corner("c", vertex, (-1.0, 0.0), (0.0, 1.0))
    assert np.allclose(a.offset_estimate()["mean"][:2], vertex, atol=1e-6)


def test_a_box_gains_more_than_a_flat_face_would():
    """A wall is rank 1.  A box has edges, so one contact pins both axes --
    which is the whole reason a box is worth an ATE visit."""
    c, half = np.array([3.03, 0.73]), (0.15, 0.08)
    ate = LandmarkATE.box("p", c, half, np.deg2rad(24.0))
    h0 = ate.entropy()
    ate.feed_contact(c + np.array([0.0, -0.09]), np.array([0.0, -1.0]))
    assert ate.entropy() < h0 - 1.0
    sig = ate.offset_estimate()["sigma"]
    assert sig[0] < 0.10 and sig[1] < 0.10


def test_box_eig_prefers_the_face_over_a_press_past_its_edge():
    c, half = np.array([3.03, 0.73]), (0.15, 0.08)
    ate = LandmarkATE.box("p", c, half, 0.0)
    n = np.array([0.0, -1.0])
    on = AteProbe("f", 0.0, c + np.array([0.0, -0.08]), -n)
    off = AteProbe("f", 0.30, c + np.array([0.30, -0.08]), -n)
    r = ate.rank([on, off], tie_break_xy=c)
    assert r[0].s == 0.0 and r[0].score_nats > r[1].score_nats


def test_the_stance_probe_reads_the_face_the_hand_will_meet(compiled):
    """`_stance_for` corrects the press depth by raycasting the map -- and it
    must probe where the HAND lands, `HAND_LEAD_M` along the lane, not where
    the base stops.

    On a flat wall the two agree, which is why this hid until a curved body
    was pressed.  On C1 -- a 0.30 m bulge -- the base-probe lands 0.18 m off
    the apex where the cylinder has already fallen away, and once the search
    ladder steps further it misses the bulge entirely and finds the wall
    behind it.  Over the three stations the failed run actually tried
    (0.95 / 1.25 / 0.65) the base-probe commanded 0.392 / 0.359 / 0.440 while
    the hand-probe commands the intended 0.33 at all three.  In MuJoCo that
    was 0/3 contacts at 0.0-2.8 N against an arm reaching ~0.43; probing at
    the hand's arc length landed 1/1 at 14.4 N on the first try."""
    from demo_map.executor import (APEX_PRESS_STANCE, CRAB_PRESS_STANCE,
                                   HAND_LEAD_M)
    from execution.wall_lane import WallSegment

    scene, _route, steps, _rep = compiled
    bodies = tuple(scene.plan.bodies)

    def stance(seg, s, want, lead):
        lane = seg.lane_point(float(s) + lead, FOLLOW_STANDOFF)
        d, _n, _h = raycast(lane, -np.asarray(seg.normal, float),
                            FOLLOW_STANDOFF + 0.45, bodies)
        off = 0.0 if d is None else float(d) - FOLLOW_STANDOFF
        return float(np.clip(want + off, 0.30, 0.44))

    cyl = next(s for s in steps if s["kind"] == "ate_landmark"
               and s["shape"] == "cylinder")
    n = np.asarray(cyl["normal"], float)
    t = np.array([-n[1], n[0]])
    apex = np.asarray(cyl["centre"], float) + float(cyl["reach_m"]) * n
    seg = WallSegment(p0=apex - (0.95 + HAND_LEAD_M) * t, tangent=t, normal=n,
                      length=0.95 + HAND_LEAD_M + 0.9, name=cyl["label"])
    # AT THE APEX STATION, which is the one the ladder starts from and the
    # only one where "find the apex" is even a well-posed request.  The
    # flanks are no longer part of the claim: since the hallway wall was made
    # continuous behind the curved features (2026-08-31) a probe 0.30 m off a
    # 0.2985 m cylinder correctly reads the WALL, 0.29 m further back, where
    # it used to read nothing at all through the gap the chord left.  Reading
    # the wall there is the map telling the truth, not the probe failing.
    at_hand = stance(seg, 0.95, APEX_PRESS_STANCE, HAND_LEAD_M)
    at_base = stance(seg, 0.95, APEX_PRESS_STANCE, 0.0)
    assert at_hand == pytest.approx(APEX_PRESS_STANCE), \
        "the hand-probe must find the apex (face_off = 0)"
    assert at_base > at_hand + 0.02, \
        "the base-probe must read the flank -- if it no longer does, this " \
        "test has stopped guarding anything"
    # ...and on a flat wall the correction is identical either way
    flat = next(s for s in steps if s["kind"] == "follow" and s["stations"])
    for st in flat["stations"][:6]:
        assert stance(flat["seg"], st, CRAB_PRESS_STANCE, 0.0) == \
            pytest.approx(stance(flat["seg"], st, CRAB_PRESS_STANCE,
                                 HAND_LEAD_M))


# ── what the video draws ─────────────────────────────────────────────────────

def _cov_stub_executor(seed=3, n=500):
    """An executor whose belief is a real `ParticleBelief`, so the covariance
    under test is the shipped accessor and not a re-implementation."""
    from global_nav.belief import BeliefConfig, ParticleBelief
    from demo_map.scene_demo import build_demo_scene
    rng = np.random.default_rng(seed)
    plan = build_demo_scene().plan
    xy = rng.normal(0.0, 1.0, size=(n, 2)) * np.array([0.30, 0.05])
    b = ParticleBelief(plan, BeliefConfig(n_particles=n), xy,
                       rng.normal(0.0, 0.05, n), np.zeros(n, dtype=int),
                       np.zeros((n, 2)), rng)
    w = rng.random(n)
    b.logw = np.log(w / w.sum())
    ex = DemoRouteExecutor.__new__(DemoRouteExecutor)
    ex.mb = SimpleNamespace(b=b)
    return ex, b


def test_recorded_covariance_is_the_same_measurement_as_sigma():
    """`_cov_xy` is the position block of the ONE planner-facing SE(2)
    covariance, and that one drops the weighted-sample bias correction on
    purpose so the HUD number and the plotted ellipse can never be two
    different things."""
    ex, _b = _cov_stub_executor()
    cov = ex._cov_xy()
    assert cov.shape == (2, 2)
    assert np.allclose(cov, cov.T)
    assert np.isclose(np.trace(cov), ex._sigma() ** 2)
    assert cov[0, 0] > cov[1, 1]          # the injected anisotropy survives


def test_the_planner_covariance_is_one_object_and_the_views_agree():
    """Every planner-facing spread is a VIEW of `_pose_cov`, not a second
    moment computed alongside it: the position block, its trace's root, and
    the heading term must come out of the same 3x3."""
    ex, b = _cov_stub_executor(seed=11)
    cov3 = ex._pose_cov()
    assert cov3.shape == (3, 3)
    assert np.allclose(cov3, cov3.T)
    assert np.allclose(ex._cov_xy(), cov3[:2, :2])
    assert np.isclose(ex._sigma_yaw() ** 2, cov3[2, 2])
    assert np.isclose(ex._sigma() ** 2, cov3[0, 0] + cov3[1, 1])
    # ...and it is the RAW moment, not the reporting one: the two differ by
    # 1/(1 - sum w^2), which is where an A/B between two scorers reading
    # different accessors would have been measuring the accessors.
    corrected = np.asarray(b.pose_covariance(), float)
    w = np.asarray(b.weights, float)
    assert np.allclose(corrected, cov3 / (1.0 - float(w @ w)))
    assert not np.allclose(corrected, cov3)


def test_ellipse_axes_and_the_pre_covariance_fallback():
    spec = importlib.util.spec_from_file_location(
        "dv", ROOT / "scripts" / "demo_room_video.py")
    dv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dv)
    cov = np.diag([0.09, 0.01])                    # sigma 0.30 x 0.10
    w, h, ang, area = dv._ellipse_axes(cov, n_sigma=2.0)
    assert np.isclose(w, 1.2) and np.isclose(h, 0.4)
    assert abs(ang) < 1e-6 or abs(abs(ang) - 180.0) < 1e-6
    assert np.isclose(area, np.pi * 0.6 * 0.2)
    # a run recorded before cov_xy existed still plays, as the old circle
    doc = {"nav_trace": [{"t": 0.0, "bel_xy": [0.0, 0.0], "sigma_m": 0.2}]}
    import json
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump({**doc, "touches": []}, f)
        path = Path(f.name)
    _doc, _nav, _t, _bel, sig, cov2 = dv._load_run(path)
    assert np.isclose(np.trace(cov2[0]), sig[0] ** 2)
    assert np.isclose(cov2[0][0, 0], cov2[0][1, 1])   # isotropic fallback
    path.unlink()
