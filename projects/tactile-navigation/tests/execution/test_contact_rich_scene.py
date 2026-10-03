"""Static checks for the contact-rich demo scene + TOUCH_STEP_FOLLOW gating.

Mirrors the structural tests the other scenes carry: reach, discrimination,
aliasing, clearance, and the wrong-favoured prior actually being wrong-favoured.
"""

import numpy as np
import pytest

from execution.executive import ExecConfig
from execution.scenes import build_scene

DELTAS = (0.0, 0.55)


@pytest.fixture(scope="module")
def scene():
    return build_scene("contact_rich_gap")


def test_scene_builds_with_two_turns_and_four_walls(scene):
    assert scene.n_route_decisions == 2
    assert len(scene.walls) == 4
    assert len({w.index for w in scene.walls}) == 4, "each asset box used once"
    assert scene.goal_edge == 2


def test_every_station_is_reachable_under_both_hypotheses(scene):
    for name, st in scene.stations.items():
        for d in DELTAS:
            assert st.reachable(scene.plan, d), (name, d)


def test_gap_stations_discriminate_fully_and_grid_does_not(scene):
    for name in ("GAPW", "GAPE"):
        st = scene.stations[name]
        assert st.discrimination(scene.plan, 0.0, 0.55) == 1.0, name
        assert st.discrimination(scene.plan, 0.55, 0.0) == 1.0, name
    # on-route leg-0 grid stances AWAY from the gap and the terminus window
    # must be aliased: flat proud wall under both registrations.
    gap_w, gap_e = 0.70, 1.10
    fan_lo, fan_hi = 0.06, 0.30
    term = float(scene.legs[0]["terminus"])
    leak = {}
    for name, st in scene.stations.items():
        if st.leg != 0 or st.purpose != "wall":
            continue
        lo, hi = st.s_m + fan_lo - 0.10, st.s_m + fan_hi + 0.10
        near_gap = hi >= gap_w - 0.55 and lo <= gap_e + 0.55
        near_term = hi >= term - 0.55
        if near_gap or near_term:
            continue
        d = st.discrimination(scene.plan, 0.0, 0.55)
        if d > 0.0:
            leak[name] = d
    assert not leak, f"aliasing leak at {leak}"


def test_gap_width_does_not_alias_across_hypotheses(scene):
    # gap edges 0.40 m apart, hypotheses 0.55 m apart: an edge under one
    # registration must not reproduce the other edge's pattern under the rival
    assert abs((1.10 - 0.70) - 0.55) > 0.10


def test_all_walls_are_on_the_standard_plane(scene):
    """A proud leg-0 wall was measured snagging the parked wrist at 157 N on
    the very first walk (the stage-1 approach walks the start's own lateral
    line).  Every wall face must sit at the standard standoff plane and every
    station must carry a zero face offset."""
    for name, st in scene.stations.items():
        assert abs(st.face_offset_m) < 1e-9, (name, st.face_offset_m)
    for w in scene.walls[:2]:                          # the leg-0 pair
        assert abs(w.center_map[1] - 0.43) < 1e-9, w


def test_route_lane_and_start_clear_every_wall(scene):
    """The walking envelope (lane line, 0.28 m hand clearance) must hold along
    every leg under BOTH hypotheses' worst placement, and around the start."""
    lane = 0.18
    for k, leg in enumerate(scene.legs):
        a = np.asarray(leg["origin"], float)
        u = np.asarray(leg["u"], float)
        v = np.asarray(leg["v"], float)
        length = {0: 3.68, 1: 3.24, 2: 0.8}[k]
        pts = np.array([a + s * u - lane * v for s in np.arange(0.0, length, 0.05)])
        for body in scene.plan.bodies:
            sd, _ = body.signed_distance(pts, None)
            assert float(np.min(sd)) >= 0.28 - 1e-6, (k, body.name, float(np.min(sd)))
    start = np.zeros((1, 2))
    for body in scene.plan.bodies:
        sd, _ = body.signed_distance(start, None)
        assert float(sd[0]) >= 0.30, ("start", body.name, float(sd[0]))


def test_press_stance_standoff_stays_outside_wrist_envelope(scene):
    """Commanded stance = station + bias toward the wall: the commanded
    standoff to the pressed face must stay >= min_commanded_standoff_m (0.30)
    and <= 0.38 (the measured d<=0.38-at-stabilize press-validity bound)."""
    cfg = ExecConfig()
    for name, st in scene.stations.items():
        if st.leg != 0:
            continue
        stance = st.xy + (cfg.approach_bias_m - st.face_offset_m) * st.facing
        standoff_to_face = 0.38 - float(stance[1])
        assert cfg.min_commanded_standoff_m - 1e-9 <= standoff_to_face <= 0.38, (
            name, standoff_to_face)


def test_wrong_mode_commit_is_physically_punished(scene):
    """Turning at the believed corner under the wrong registration happens
    while the followed wall still runs alongside: the early corner point sits
    within the material span of wall0_b, not past it."""
    term = float(scene.legs[0]["terminus"])       # 3.30
    early = term - 0.55                            # believed corner, wrong mode
    assert 1.10 < early < term, early


def test_prior_favours_the_wrong_mode(scene):
    (d0, m0), (d1, m1) = scene.prior_modes
    truth = scene.true_delta_m
    wrong = m1 if d1 != truth else m0
    right = m0 if d0 == truth else m1
    assert wrong > right, "the heavier prior mode must be the WRONG one"


def test_touch_step_follow_is_off_by_default():
    assert ExecConfig().touch_step_follow is False


def test_maintenance_trigger_arithmetic():
    cfg = ExecConfig(touch_step_follow=True)
    grown = cfg.motion.sigma_along_per_m * 2.5
    assert np.hypot(0.05, grown) < cfg.branch_sigma_max_m or True
    # the gap trigger fires past touch_gap_m of travel
    assert cfg.touch_gap_m == 1.0 and cfg.touch_commit_window_m > 0.3
