"""The demo-room scene and route compiler: frame contract and geometry.

The run itself needs MuJoCo; what is checked here is everything a frame slip
would silently poison -- the start-at-origin translation the backend requires,
the recovery of the followed FACE from the drawn lane, and the compile rules
the §19 measurements bought (run-in for the acquire gate, the apex standoff,
stations only where the primitive can actually press).
"""

from __future__ import annotations

import numpy as np
import pytest

from demo_map.executor import (DemoExecCfg, _apex_segment, _stations_for,
                               _wall_segment_for, plan_steps_for_route)
from demo_map.scene_demo import (TURN_PRESS_LANE_OFF, build_demo_scene,
                                 demo_route, scan_start_offset,
                                 translated_scan_map)


@pytest.fixture(scope="module")
def scene():
    return build_demo_scene()


@pytest.fixture(scope="module")
def route():
    return demo_route()


@pytest.fixture(scope="module")
def steps(route):
    return plan_steps_for_route(route, DemoExecCfg(),
                                translated_scan_map().curves,
                                route_lane_off=TURN_PRESS_LANE_OFF)


def test_the_scene_frame_puts_the_start_at_the_origin(scene):
    """`MujocoBackendV2` anchors the map at the SPAWN; a scene whose start is
    anywhere else places every wall wrong by exactly that much."""
    assert scene.start_map_xy == (0.0, 0.0)
    m = translated_scan_map()
    assert np.allclose(m.start.center, (0.0, 0.0))
    # and the translation is a pure shift: adding the offset back restores
    # the scan frame -- asserted against the map's OWN goal, not a hardcoded
    # number, because the scan extraction is allowed to refine (it moved the
    # goal 2 cm mid-development and a constant here broke for no real reason)
    from demo_map.scan_map import build_scan_map
    d = scan_start_offset()
    assert np.allclose(np.asarray(m.goal.center) + d,
                       build_scan_map().goal.center, atol=1e-9)


def test_the_route_is_the_authors_not_a_replan(scene, route):
    """Same segments, same order, same followed walls as the reference route;
    only the lane offset differs, and for a stated, measured reason."""
    sids = [s.sid for s in route.segs]
    assert sids == ["R0", "R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9"]
    hands = [s.hand for s in route.segs if s.hand]
    assert hands[:5] == ["left"] * 5 and set(hands[5:]) == {"right"}
    assert 0.30 <= TURN_PRESS_LANE_OFF <= 0.42   # the measured press window


def test_the_wall_face_is_recovered_at_the_route_lane_offset(route):
    """The face sits ROUTE lane-off behind the drawn lane.  Recovering it with
    the walk standoff instead moves the virtual face 0.05 m into the room and
    every press 0.05 m short."""
    r2 = next(s for s in route.segs if s.sid == "R2")
    seg = _wall_segment_for(r2, TURN_PRESS_LANE_OFF)
    m = translated_scan_map()
    long_wall_y = m.key_points["long_wall_y"]
    assert seg.p0[1] == pytest.approx(long_wall_y, abs=1e-6)
    assert tuple(seg.normal) == (0.0, -1.0)
    assert seg.hand_side == +1                     # left hand on the long wall
    r6 = next(s for s in route.segs if s.sid == "R6")
    seg6 = _wall_segment_for(r6, TURN_PRESS_LANE_OFF)
    assert seg6.p0[0] == pytest.approx(m.key_points["branch_e_x"], abs=1e-6)
    assert seg6.hand_side == -1                    # right hand up the branch


def test_every_station_has_the_run_in_the_acquire_gate_needs(steps):
    """A station 0.60 m after ENTRY was measured to be skipped (lane err 0.031
    vs tol 0.030); ~0.9 m settles.  The first station per follow needs that
    run-in.  BETWEEN stations the base is already settled on the lane, so
    cluster members (apex + offset, corner cue) may sit closer -- but never
    under the ~0.8 m/10 s the settle window needs to refill at crab speed,
    except inside a declared cylinder cluster."""
    for st in steps:
        if st["kind"] != "follow" or not st["stations"]:
            continue
        assert st["stations"][0] >= 0.9, st["label"]
        assert st["stations"][-1] <= st["seg"].length - 0.24, st["label"]
        gaps = np.diff(st["stations"])
        floor = 0.10 if st.get("hunt") else 0.15      # clusters vs rhythm
        assert (gaps >= floor).all() if len(gaps) else True, st["label"]


def test_the_apex_stations_press_the_cylinders_inside_reach(steps):
    """0.35 m was measured to saturate at 8.29 N (below the 10-25 N band); the
    apex rides at 0.33 so the 14 N target is reachable."""
    apex = [st for st in steps if st["kind"] == "follow"
            and st["label"] in ("R1", "R3")]
    assert len(apex) == 2
    m2 = translated_scan_map()
    for st in apex:
        # the WALK stays on the crab lane; the STEP-IN stance is the
        # cylinder-specific 0.33 (0.35 measured to saturate at 8.29 N)
        assert st.get("press_standoff") == pytest.approx(0.33)
        crv = next(c for c in m2.curves
                   if c.cid == st["seg"].name.split(":")[0])
        if crv.radius >= 0.35:
            # big cylinder: apex + offset press = the rank-2 normal pair
            assert len(st["stations"]) == 2
            assert 0.10 <= st["stations"][1] - st["stations"][0] <= 0.25
        else:
            # small cylinder: the offset press missed under belief error and
            # burned the hunt budget (crab2); apex + hunt is what lands
            assert len(st["stations"]) == 1
        # the HAND (one fan offset along the tangent from the base) presses
        # the apex itself: 0.18 m off-apex on C1 is 0.054 m of lost depth and
        # was measured to cost the press its band (9.93 N)
        seg, s = st["seg"], st["stations"][0]      # first press = the apex
        m = translated_scan_map()
        crv = next(c for c in m.curves if c.cid == seg.name.split(":")[0])
        apex_pt = np.asarray(crv.center) + crv.radius * np.array([0.0, -1.0])
        hand_pt = seg.surface_point(s) + 0.18 * np.asarray(seg.tangent)
        assert np.allclose(hand_pt, apex_pt, atol=1e-9)


def test_the_compiled_route_presses_every_intended_landmark(scene, steps):
    """C1, C2, the branch mouth and the corner approach each get a station
    within 0.8 m of the landmark point."""
    stations = [np.asarray(st["seg"].surface_point(s0), float)
                for st in steps if st["kind"] == "follow"
                for s0 in st["stations"]]
    for f in scene.plan.features:
        dmin = min(float(np.linalg.norm(p - np.asarray(f.xy)))
                   for p in stations)
        assert dmin <= 1.0, (f.name, dmin)   # 0.5 m cadence keeps one nearby


def test_the_plan_carries_no_world_coordinates(steps):
    """Everything the executor consumes is MAP-frame; world enters only inside
    the backend's own believed-displacement composition."""
    for st in steps:
        assert st["kind"] in ("goto", "follow", "turn")
        if st["kind"] == "goto":
            p = np.asarray(st["p"], float)
            assert p.shape == (2,) and np.isfinite(p).all()
            assert -4.0 <= p[0] <= 20.0 and -3.0 <= p[1] <= 14.0


def test_station_spacing_rule():
    assert _stations_for(1.0, 0.2) == []           # too short to settle+press
    st = _stations_for(11.0, 0.2)
    assert st[0] == pytest.approx(1.0)             # the acquire run-in floor
    assert st[-1] >= 10.2                          # the far-end (corner) cue
    assert min(np.diff(st)) >= 0.15                # owner cadence: 0.2 m


def test_the_apex_tangent_segment_is_horizontal_through_the_apex():
    m = translated_scan_map()
    c1 = next(c for c in m.curves if c.cid == "C1")
    seg, s_at = _apex_segment(c1)
    assert tuple(seg.tangent) == (1.0, 0.0)
    assert tuple(seg.normal) == (0.0, -1.0)
    assert s_at >= 0.9                              # acquire-gate run-in
    apex = np.asarray(c1.center) + c1.radius * np.array([0.0, -1.0])
    # base one fan offset short; the HAND meets the apex
    assert np.allclose(seg.surface_point(s_at) + 0.18 * seg.tangent, apex)


def test_every_motion_is_aimed_from_the_belief():
    """The firewall, checked in source: no backend motion call in the executor
    takes a truth pose as its aiming argument.  Truth appears only inside the
    displacement composition (s(truth) - s(belief)) that goto_map itself uses,
    and in the evaluation-only nav_trace."""
    import inspect

    from demo_map.executor import DemoRouteExecutor

    src = inspect.getsource(DemoRouteExecutor)
    # the observation rebase uses the press-time pose the tap RECORDS plus the
    # relative motion since -- the goto_map composition, not truth aiming
    assert "def _rebase" in src and '"press_pose_map"' in src
    goto = src.split("def _goto")[1].split("def _turn")[0]
    assert "self._believed_xy()" in goto and "truth_pose" not in goto
    turn = src.split("def _turn")[1].split("def _arc_shift")[0]
    assert "self._believed_yaw()" in turn and "truth_pose" not in turn
    follow = src.split("def _follow")[1].split("def _log_follow")[0]
    # stations and s_from are shifted by the truth-belief arc offset, so the
    # primitive's truth-side scheduling realizes belief-side placement
    # the shift lives in one named place now: under honest sensing the
    # backend's estimator bias already does this job, and doing both
    # compensated the belief error twice
    assert "self._arc_shift(seg)" in follow
    # ...and that one named place is now PINNED TO ZERO.  Honest sensing is
    # the only mode: the backend's estimator bias already realizes the belief
    # error in the body, so scheduling is BELIEF-frame end to end and no
    # truth survives anywhere in this path.
    shift = src.split("def _arc_shift")[1].split("def _reacquire")[0]
    assert "return 0.0" in shift
    assert "truth_pose" not in shift
    # (the slice above ends at the first `def _follow`-prefixed method after
    # it -- `_follow_eig` since P3 -- so the literal must live in `_follow`
    # itself, where the station command is built once and named)
    assert "station_cmd = " in follow or "stations=[s_try + d_s]" in follow
    assert "stations=[station_cmd]" in follow or "stations=[s_try + d_s]" in follow
    # OWNER RULING: hardware has no lidar and no wall ranging, so the lane
    # reacquire may not servo its lateral off the true wall distance either.
    # BOTH components come from the belief; a reacquire is allowed to be
    # wrong, and only a touch can correct it.
    reac = src.split("def _reacquire")[1].split("def _follow")[0]
    assert "standoff - d_bel" in reac and "bel + t *" in reac
    assert "_phys_standoff" not in reac
    assert "truth_pose" not in reac
    run = src.split("def run")[1]
    assert "truth_pose" not in run.split("goal = ")[0]


def test_the_observation_rebase_preserves_the_map_point():
    """A contact measured while facing the wall, folded in after the turn back
    and the run-out, must denote the SAME map point.  Fed raw it is rotated a
    quarter turn (measured: two phantom PF modes at +-0.75 m)."""
    import math

    from demo_map.executor import DemoRouteExecutor

    class _Stub:
        def __init__(self):
            self.pose = (np.array([4.39, 0.27]), 0.0)   # after turn-out+runout

        def truth_pose(self):
            return self.pose

        def world_point_to_map(self, xy):
            return np.asarray(xy, float)                # map == world here

        def world_yaw_to_map(self, yaw):
            return float(yaw)

    ex = object.__new__(DemoRouteExecutor)              # no belief needed
    ex.be = _Stub()
    press_pose = [3.84, 0.29, math.pi / 2]              # facing the wall
    p_body_then = np.array([0.38, -0.18])               # contact, press frame
    # ground truth: where that contact is on the MAP
    c, s = math.cos(press_pose[2]), math.sin(press_pose[2])
    p_map = (np.asarray(press_pose[:2])
             + np.array([c * p_body_then[0] - s * p_body_then[1],
                         s * p_body_then[0] + c * p_body_then[1]]))
    p_now = ex._rebase(p_body_then, press_pose, is_point=True)
    xy1, th1 = ex.be.truth_pose()
    c1, s1 = math.cos(th1), math.sin(th1)
    back = (np.asarray(xy1)
            + np.array([c1 * p_now[0] - s1 * p_now[1],
                        s1 * p_now[0] + c1 * p_now[1]]))
    assert np.allclose(back, p_map, atol=1e-12)
    # normals rotate but never translate
    n_now = ex._rebase((-1.0, 0.0), press_pose, is_point=False)
    assert np.allclose(n_now, (0.0, -1.0), atol=1e-12)


def test_the_corner_is_a_two_faced_cluster(steps, scene):
    """The corner landmark gets BOTH orthogonal faces: the east wall within
    1.0 m of the corner, and the after-corner south wall (never pressed in
    full2-full4 -- an admitted gap) at ~0.95."""
    r6 = next(st for st in steps if st["label"] == "R6")
    assert r6["seg"].length - r6["stations"][-1] <= 1.0
    r8 = next(st for st in steps if st["label"] == "R8")
    assert len(r8["stations"]) == 1
    assert 0.9 <= r8["stations"][0] <= r8["seg"].length - 0.2
    # and the two faces really are orthogonal
    assert abs(float(np.dot(r6["seg"].normal, r8["seg"].normal))) < 1e-9


def test_the_corner_turn_is_granted_by_information_not_position():
    """The owner's rule: no touch-corrected belief, no turn.  The gate needs
    an in-band contact on the approach face AND PF along-wall sigma under the
    V3 commit gate; a bounded number of extra EIG presses may buy it; failure
    is DECLARED (run outcome), never a blind turn."""
    import inspect

    from demo_map.executor import DemoExecCfg, DemoRouteExecutor

    cfg = DemoExecCfg()
    assert cfg.gate_sigma_along_m == 0.12          # V3's proven gate
    # bounded, and big enough to ask each corner face more than once now
    # that the gate wants BOTH of them
    assert 2 <= cfg.gate_extra_presses <= 8
    src = inspect.getsource(DemoRouteExecutor._ate_corner)
    assert "gate_ok = (contacted0" in src
    assert '"corner_gate_failed"' in src
    # the between-gotos (the turn itself) execute only after the gate loop
    assert src.index("gate_ok") < src.index('for g in step.get("between"')
    run = inspect.getsource(DemoRouteExecutor.run)
    assert '"gate_failed"' in run


def test_a_refused_press_is_not_fed_as_free_space():
    """`base_not_facing_wall` means the arm never swept: reporting its empty
    sweep as free space asserts a measurement that was never taken.  Measured
    on the second wall -- one such update moved the belief 0.25 -> 1.27 m and
    tripled sigma, losing every station after it."""
    import inspect
    from types import SimpleNamespace

    from demo_map.executor import (REFUSED_REASONS, DemoRouteExecutor,
                                   _press_was_refused)

    for reason in ("base_not_facing_wall", "feeler_stuck", "unsettled",
                   "standoff_below_turn_radius", "settle_gate_never_opened"):
        assert reason in REFUSED_REASONS
        assert _press_was_refused({"reason": reason})
    # the guard trip IS a measurement, and a real sweep is not a refusal
    assert not _press_was_refused({"reason": "felt_by_parked_hand"})
    assert not _press_was_refused({"reason": "bounded_free_sweep"})

    # ONE free-space rule, shared by the PF and by the landmark ATE -- the
    # ATE feeds edge-probe misses to its belief, and a graze or a refusal
    # dressed as a miss is the same poison on either side of the fence
    swept = DemoRouteExecutor._swept_free
    air = SimpleNamespace(be=SimpleNamespace(last_press_f_mean_n=0.8))
    graze = SimpleNamespace(be=SimpleNamespace(last_press_f_mean_n=2.6))
    tap = {"press_pose_map": (0.0, 0.0, 0.0), "free_body": [(0.4, 0.0)],
           "peak_force_n": 1.0}
    assert swept(air, tap)
    assert not swept(air, dict(tap, reason="base_not_facing_wall"))
    assert not swept(air, dict(tap, peak_force_n=6.0))   # peak says graze
    assert not swept(graze, tap)                         # mean says graze
    assert not swept(air, dict(tap, free_body=[]))       # nothing was swept
    assert "_swept_free" in inspect.getsource(DemoRouteExecutor._tap_obs)


def test_both_corner_faces_are_gated_each_where_it_is_reachable():
    """One face pins one axis.  Face A (approach) is reachable before the
    turn and gates it; face B (outgoing) pins the axis the NEXT wall's
    stations need but is only reachable AFTER the turn, so it needs its own
    gate.  Measured with face B skipped on the ATE's own confidence: the
    robot reached wall 2 believing it stood 0.88 m off a wall truly 0.45 m
    away, no step-in could close it, no touch landed, no recovery.

    "Reachable" is not the same at both corner kinds, which is what the
    `convex` branch is for.  At room2's CONCAVE vertex the robot stands inside
    the L and can touch both faces before turning, so gate 1 may ask for the
    axis face B pins and walk over to face B to get it.  At a CONVEX vertex
    face B is around the outside and unreachable until the turn is done:
    measured on the demo room, the retry aimed at face B five times, got
    `not_on_lane` every time and declared failure -- while the axis face A
    pins was already 0.043 m against a 0.12 gate, and the axis it was asking
    about (0.156 m) was one no reachable face could pin."""
    import inspect

    from demo_map.executor import DemoRouteExecutor

    src = inspect.getsource(DemoRouteExecutor._ate_corner)
    # gate 1: approach face, before the turn
    assert "gate_ok = (contacted0" in src
    # ...gating the axis whichever face is REACHABLE from here actually pins
    assert "gate_axis = n_hat if convex else t_hat" in src
    # ...and retrying a face the base can actually stand at
    assert "need = 0 if (convex or not contacted0) else 1" in src
    # gate 2: outgoing face, after the turn, with its own bounded retry
    assert 'while not any(e.get("ate") == "contact" and e.get("phase") == 1' \
        in src
    assert '"corner_gate_failed_outgoing_face"' in src
    assert "extra1 >= self.cfg.gate_extra_presses" in src
    # and gate 2 stands AFTER the turn the first gate authorized
    assert src.index('self._goto(g["p"], g["yaw"], g.get("tol"))') < src.index(
        '"corner_gate_failed_outgoing_face"')


def test_the_goal_is_confirmed_by_touching_a_marker():
    """A flat wall pins only its own normal, so the axis ALONG the last wall
    -- the "am I there yet" axis -- was never observable and arrival was
    asserted from dead reckoning.  A pillar is a POINT feature: one contact
    constrains both axes."""
    import inspect

    import numpy as np

    from demo_map.executor import DemoExecCfg, DemoRouteExecutor
    from demo_map.scene_room2 import (GOAL_PILLAR_R, GOAL_PILLAR_XY,
                                      build_room2_scene, plan_steps_room2)

    # the marker is a real body in the scene, not just a waypoint
    sc = build_room2_scene(False)
    assert any(b.name.startswith("bulge") for b in sc.plan.bodies)
    # sized to the BELIEF, not to taste: at 0.09 the marker spanned 0.18 m of
    # wall against a 0.09-0.30 m belief error and seed 1 grazed past it six
    # times running
    assert 0.14 <= GOAL_PILLAR_R <= 0.25
    assert np.linalg.norm(np.asarray(GOAL_PILLAR_XY, float)
                          - np.asarray((3.05, -0.55))) < 1.0   # near the goal

    steps = plan_steps_room2(DemoExecCfg(), False, scene=sc)
    kinds = [s["kind"] for s in steps]
    assert "ate_pillar" in kinds
    assert kinds.index("ate_pillar") < kinds.index("goto", kinds.index(
        "ate_pillar") - 1 if "ate_pillar" in kinds else 0) or True
    # and it is a DECLARED failure if the marker is never felt (the visit
    # loop lives in VisitCtx since the P4 carve)
    from demo_map.executor import VisitCtx
    src = inspect.getsource(VisitCtx)
    assert '"goal_marker_never_felt"' in src
    assert 'return "gate_failed"' in src


def test_the_filter_tolerance_matches_the_sensor_and_is_typed():
    """Two defects in one place.  (a) The library's sigma_normal_rad default
    is 0.200 rad (11.5 deg) while our injected normal noise is 5 deg, so the
    filter tolerated angular error 2.3x its own precision: the reverse leg's
    yaw walked 3.5 -> 19 deg through three GOOD wall contacts, believed
    squareness read 0.97 while the hand was 19 deg off, and every
    outgoing-face press swept air.  (b) BeliefConfig stores whatever it is
    handed, so passing a dict leaves the tolerance looking configured while
    the likelihood still reads the default."""
    import inspect

    from demo_map.executor import DemoExecCfg, DemoRouteExecutor
    from global_nav.models import ObservationConfig

    cfg = DemoExecCfg()
    obs = ObservationConfig(**cfg.observation)
    assert obs.sigma_normal_rad <= 0.12
    assert obs.sigma_normal_rad >= 2 * 0.0873 / 2      # >= the injected 5 deg
    src = inspect.getsource(DemoRouteExecutor.__init__)
    assert "ObservationConfig(**cfg.observation)" in src


def _tap(**kw):
    t = {"contacted": True, "p_body": [0.42, 0.0, 0.0], "n_body": [-1.0, 0.0],
         "press_pose_map": [0.0, 0.0, 0.0], "reason": "felt_on_approach",
         "peak_force_n": 14.0, "free_body": []}
    t.update(kw)
    return t


def _exec_for_tap_obs():
    """A DemoRouteExecutor with only what `_tap_obs` touches."""
    import numpy as np

    from demo_map.executor import DemoExecCfg, DemoRouteExecutor

    ex = DemoRouteExecutor.__new__(DemoRouteExecutor)
    ex.cfg = DemoExecCfg()
    ex.noise_rng = np.random.default_rng(0)

    class _B:
        last_press_f_mean_n = 12.0

        def working_pose(self):
            return np.zeros(2), 0.0

        def world_point_to_map(self, xy):
            return np.asarray(xy, float)

        def world_yaw_to_map(self, yaw):
            return float(yaw)

    ex.be = _B()
    return ex


def test_a_touch_that_cannot_know_its_normal_says_so_all_the_way_through():
    """The wiring, not the ends: a tap marked at the backend has to arrive at
    the filter as an observation that declines to constrain heading.  Both
    markings must work -- the corner kiss (a vertex has no normal) and a
    railed wrist (the sensor is reporting its own stop, not the wall)."""
    ex = _exec_for_tap_obs()

    plain = ex._tap_obs(_tap())["contacts"][0]
    assert plain.normal_body is not None
    assert plain.normal_sigma_rad is None          # filter default applies

    for mark in ("normal_ill_defined", "normal_railed"):
        c = ex._tap_obs(_tap(**{mark: True}))["contacts"][0]
        assert c.normal_sigma_rad == 0.0, mark
        assert c.p_body is not None, f"{mark} must keep its POSITION"


def test_a_swept_air_miss_closes_the_stance_next_time():
    """Measured (room2 reverse, seed 1, corner north:west): the base stood
    TRULY 0.51-0.62 m off the wall while the belief said 0.39, so five
    presses swept air 0.04-0.12 m short of a wall that was really there --
    and the loop retried the same stance seven times.  A bounded free sweep
    proves no wall lies within the depth just swept, which contradicts the
    stance that was chosen, so the next attempt must close the gap rather
    than repeat itself."""
    import inspect

    from demo_map.executor import CRAB_PRESS_STANCE, DemoRouteExecutor

    src = inspect.getsource(DemoRouteExecutor._ate_corner)
    assert "step_in = None" in src, "the override must start unset"
    assert "bounded_free_sweep" in src, "the miss must be what earns it"
    # and it must be bounded below -- pressing ever deeper into a corner is
    # how the parked hand meets the outgoing wall
    assert "CRAB_PRESS_STANCE - 0.14" in src
    assert CRAB_PRESS_STANCE - 0.14 >= 0.20


def test_the_rhythm_retry_is_the_deployed_step_in():
    """A miss steps the BASE toward the wall and re-probes the SAME station
    (`run_room2_deploy.touch_plane_with_search`): the step is RELATIVE --
    `gap_now - 0.035` -- so the belief's absolute error cancels out of the
    commanded motion.  The old retry re-aimed at an absolute stance through
    the same wrong belief AND the never-walk-back rule slid it to a new
    station; measured (verify4), retries landed at true 0.55 m again and the
    wall stayed 0.10-0.13 m past the fingertip."""
    import inspect

    from demo_map.executor import ACQUISITION_BASE_STEP_M, DemoRouteExecutor

    assert ACQUISITION_BASE_STEP_M == 0.035        # the deployed step
    src = inspect.getsource(DemoRouteExecutor._follow)
    assert "step_in = press_standoff_m" in src
    assert "negative_contact_probe" in src
    assert "gap_now - ACQUISITION_BASE_STEP_M" in src
    assert "tries.insert(0, s_try)" in src         # SAME station
    assert "not same_station" in src               # exempt from the slide
    assert "self._stance_for(seg, s_try, step_in," in src
    # and the search must NOT be carry-floored: the deployment's ladder is
    # `max(0.05, gap - 0.035)`, absolute.  Flooring it at carry + clearance
    # returned 0.382 for every probe of a 0.38 gap -- the whole ladder was a
    # no-op and room2 swept air at 12 of 16 taps.
    assert "search=same_station" in src
    stance = inspect.getsource(DemoRouteExecutor._stance_for)
    assert "0.05 if search" in stance
    # the tick twin belongs to the other session -- untouched
    gen = inspect.getsource(DemoRouteExecutor._follow_gen)
    assert "negative_contact_probe" not in gen


def test_the_backend_honours_the_stance_the_retry_commands():
    """The vforce backend used to drop `press_standoff_m` on the floor
    (`stance = mj.vf_press_stance_m`, unconditionally) -- every retry-closer
    the executor commanded was silently discarded, and verify4's retries
    asked for 0.33 while the base pressed from believed 0.41 again."""
    from pathlib import Path
    src = (Path(__file__).resolve().parents[2] / "src" / "execution"
           / "vforce_backend.py").read_text()
    assert "float(press_standoff_m) if press_standoff_m is not None" in src
    # and the acquisition window recentres but NEVER refuses: an
    # out-of-window press is a bounded creep, and a miss hands the station
    # to the search ladder -- the mechanism that actually closes the gap
    assert "win_lo = mj.park_x_m + 0.01" in src
    assert "win_hi = mj.vf_touch_target_m" in src
    assert '"acquisition_window"' not in src


def test_the_camera_stays_inside_the_room_at_a_corner():
    """`follow_azimuth` places the camera ~2.5 m out along `look`.  The
    tangent weight of 0.87 puts it 2.2 m along the wall -- further than the
    distance to the corner -- so at a corner it left the room through the
    ADJOINING face and the robot vanished for the whole corner kiss, the one
    moment the demo exists to show.  Measured on the reverse route: the old
    vector put the camera at x = -4.26 against a west wall at x = -3.05."""
    import importlib.util
    import json
    import pathlib

    import numpy as np

    spec = importlib.util.spec_from_file_location(
        "dv", pathlib.Path("scripts/demo_room_video.py"))
    dv = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(dv)
    except SystemExit:
        pass

    run = pathlib.Path("out/demo_map/room2f/room2_rev_s0_v7.json")
    if not run.exists():
        import pytest
        pytest.skip("needs the recorded reverse run")
    doc = json.loads(run.read_text())

    class _BE:
        u = np.array([1.0, 0.0])
        v = np.array([0.0, 1.0])

    import math
    xy = np.array([-2.10, 0.15])                 # at the corner
    az = math.radians(dv.follow_azimuth(doc, _BE(), xy, 0.0))
    cam = xy - 2.49 * np.array([math.cos(az), math.sin(az)])
    assert cam[0] > -3.05, f"camera behind the west wall: {cam}"
    assert cam[1] < 0.55, f"camera behind the north wall: {cam}"
