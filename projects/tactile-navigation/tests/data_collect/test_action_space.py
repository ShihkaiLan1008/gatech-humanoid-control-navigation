"""The free target space: the controller's bands, and bins that all exist."""

import math

import numpy as np
import pytest

from data_collect.action_space import (PREDICT_PAD_M, TouchWorkspace,
                                       body_clearance, swept_hit)
from data_collect.rooms import build_scene
from execution.vforce_backend import (TRAINED_TIP_X_M, TRAINED_TIP_Y_M,
                                      TRAINED_TIP_Z_M, VForceConfig)


@pytest.fixture(scope="module")
def ws():
    return TouchWorkspace.from_controller()


def test_the_box_is_the_checkpoints_own_trained_band(ws):
    """Section 2: "use the current robot/controller limits ... rather than
    introducing arbitrary hard-coded bounds"."""
    mj = VForceConfig()
    assert (ws.x_lo, ws.y_lo, ws.y_hi, ws.z_lo, ws.z_hi) == (
        TRAINED_TIP_X_M[0], TRAINED_TIP_Y_M[0], TRAINED_TIP_Y_M[1],
        TRAINED_TIP_Z_M[0], TRAINED_TIP_Z_M[1])
    assert ws.park == (mj.park_x_m, mj.park_y_m, mj.park_z_m)
    # ...and every knob the deployed press uses sits inside it, or the
    # collector and the demo would be exercising different arms
    assert ws.x_lo <= mj.vf_press_stance_m <= ws.x_hi
    assert ws.y_lo <= mj.vf_hand_y_m <= ws.y_hi
    assert ws.z_lo <= mj.vf_hand_z_m <= ws.z_hi


def test_the_reach_is_capped_by_the_travel_limit_too(ws):
    """A tightened `vf_touch_travel_cap_m` must tighten the sampler, not be
    silently out-ranged by the trained band."""
    mj = VForceConfig()
    assert ws.x_hi == min(TRAINED_TIP_X_M[1], mj.park_x_m + mj.vf_touch_travel_cap_m)
    tight = TouchWorkspace.from_controller(
        __import__("dataclasses").replace(mj, vf_touch_travel_cap_m=0.05))
    assert tight.x_hi == pytest.approx(mj.park_x_m + 0.05)


def test_every_one_of_the_forty_five_bins_is_reachable(ws):
    """Equal-WIDTH angular bins leave a middle bin 0.8 deg wide, because
    |y| >= 0.08 puts a deadband around the body centreline.  Quantile edges
    make all five equally likely, which is what stratifying is for."""
    assert (ws.n_height, ws.n_angle, ws.n_reach) == (3, 5, 3)
    assert len(ws.all_bin_keys()) == 45
    assert ws.feasible_bins() == set(ws.all_bin_keys())


def test_a_uniform_draw_spreads_over_every_bin(ws):
    rng = np.random.default_rng(0)
    from collections import Counter
    c = Counter(ws.bin_key(ws.sample(rng)) for _ in range(45000))
    assert len(c) == 45
    assert min(c.values()) > 0.25 * (45000 / 45)


def test_sampling_inside_a_bin_lands_in_that_bin_and_stays_in_the_box(ws):
    rng = np.random.default_rng(3)
    for key in ws.all_bin_keys():
        t = ws.sample_in_bin(rng, key)
        assert t is not None, key
        assert ws.bin_key(t) == key
        assert ws.contains(t)


def test_the_hand_is_the_sign_of_y_and_the_approach_is_generated(ws):
    """Section 2: if the controller takes only a target point, save the
    approach direction that was actually generated."""
    t = np.array([0.46, -0.20, 0.20])
    assert ws.hand_for(t) == "right"
    d = ws.approach(t, "right")
    assert float(np.linalg.norm(d)) == pytest.approx(1.0)
    expect = t - ws.park_tip("right")
    assert np.allclose(d, expect / np.linalg.norm(expect))


def test_contains_rejects_the_body_centreline_deadband(ws):
    assert not ws.contains([0.40, 0.0, 0.18])
    assert not ws.contains([0.40, 0.05, 0.18])
    assert ws.contains([0.40, 0.09, 0.18])


def test_the_balancing_ray_is_conservative_by_default(ws):
    """PREDICT_PAD_M is 0 on a measurement: a 0.035 m pad called 16 of 20
    draws hits at a 0.476 m standoff and none of them contacted."""
    assert PREDICT_PAD_M == 0.0
    scene = build_scene("room2")
    bodies = tuple(scene.plan.bodies)
    from data_collect.route import RouteRef
    from data_collect.rooms import planned_centreline
    ref = RouteRef.from_polyline(planned_centreline(scene), bodies)
    pose = ref.pose_at(1.0, -0.02)                     # 0.476 m off the face
    park = ws.park_tip("left")
    # a target short of the face is a miss; one past it is a hit
    short = np.array([0.44, 0.12, 0.15])
    past = np.array([0.48, 0.12, 0.15])
    assert not swept_hit(pose.xy_map, pose.yaw_map, park, short, bodies)[0]
    assert swept_hit(pose.xy_map, pose.yaw_map, park, past, bodies)[0]
    # ...and the pad is what moves that boundary, nothing else
    assert swept_hit(pose.xy_map, pose.yaw_map, park, short, bodies,
                     pad_m=0.05)[0]


def test_the_predicted_body_is_the_one_the_ray_meets(ws):
    scene = build_scene("room2")
    bodies = tuple(scene.plan.bodies)
    from data_collect.route import RouteRef
    from data_collect.rooms import planned_centreline
    ref = RouteRef.from_polyline(planned_centreline(scene), bodies)
    pose = ref.pose_at(1.0, -0.12)
    hit, dist, name = swept_hit(pose.xy_map, pose.yaw_map,
                                ws.park_tip("left"),
                                np.array([0.47, 0.10, 0.15]), bodies)
    assert hit and name == pose.wall_body
    assert 0.0 < dist < 0.2


def test_clearance_is_measured_in_the_body_frame(ws):
    """A carried hand is 0.33 m in FRONT of the base, so "is the base clear"
    and "is the hand clear" are different questions."""
    scene = build_scene("room2")
    bodies = tuple(scene.plan.bodies)
    from data_collect.route import RouteRef
    from data_collect.rooms import planned_centreline
    ref = RouteRef.from_polyline(planned_centreline(scene), bodies)
    pose = ref.pose_at(1.0, -0.12)                      # 0.376 m off the face
    base = body_clearance(pose.xy_map, pose.yaw_map, (0.0, 0.0), bodies)
    hand = body_clearance(pose.xy_map, pose.yaw_map,
                          ws.park_tip("left"), bodies)
    assert base == pytest.approx(0.376, abs=0.01)
    assert hand == pytest.approx(0.376 - ws.park[0], abs=0.01)
    assert hand < 0.045, "this is exactly the pose that must be rejected"
