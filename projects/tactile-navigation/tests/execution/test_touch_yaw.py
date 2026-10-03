"""Yaw from touch: two fists on one face measure its orientation, and the IMU
is a measurement rather than an overwrite (`execution.punch_belief`)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from execution.punch_belief import LocalizerConfig, PunchLocalizer
from lane_graph.scene3 import build_v3_scene


class FakeCtrl:
    """Odometry that reports no motion and an IMU with a chosen bias."""

    def __init__(self, imu_yaw):
        self.imu_yaw = imu_yaw
        self.resets = []

    def odometry_increment(self):
        return np.zeros(2), 0.0

    def heading_since_reset(self):
        return self.imu_yaw

    def reset_estimator_to(self, xy, yaw):
        self.resets.append((np.asarray(xy, float), float(yaw)))


def _facing_wall(scene):
    """A pose 0.40 m from a wall of the L-room, facing it: (xy, yaw, normal)."""
    w = scene.spec.walls[0]
    a, b = np.asarray(w.p0, float), np.asarray(w.p1, float)
    t = (b - a) / np.linalg.norm(b - a)
    n = np.array([-t[1], t[0]])
    mid = 0.5 * (a + b)
    # stand on the side that is inside the room: try both
    for sign in (1.0, -1.0):
        xy = mid + sign * n * 0.45
        yaw = math.atan2(-sign * n[1], -sign * n[0])
        return xy, yaw, sign * n
    raise AssertionError


def _contacts(loc, dist_m, offset_m):
    """Body-frame contact points of the left and right fists on a flat wall
    ``dist_m`` ahead, hands ``offset_m`` to each side."""
    return (loc.contact_obs((dist_m, +offset_m), (1.0, 0.0)),
            loc.contact_obs((dist_m, -offset_m), (1.0, 0.0)))


def _make(scene, xy, yaw, cfg):
    loc = PunchLocalizer(scene, yaw, seed=0, cfg=cfg)
    # place the cloud at the pose under test (the scene's own start is elsewhere)
    loc.mb.b.xy[:] = xy + 0.02 * np.random.default_rng(1).standard_normal(loc.mb.b.xy.shape)
    loc.mb.b.theta[:] = yaw + np.random.default_rng(2).standard_normal(loc.mb.b.theta.shape) * math.radians(6)
    return loc


def test_two_fists_on_one_face_measure_yaw():
    scene = build_v3_scene("r2_lbend")
    xy, yaw, n = _facing_wall(scene)
    cfg = LocalizerConfig(imu_fuse=True, imu_yaw_sigma0_rad=math.radians(20))   # weak IMU
    loc = _make(scene, xy, yaw, cfg)
    yaw_before = loc.pose()[2]
    sig_before = loc.sigma_yaw()
    ctrl = FakeCtrl(imu_yaw=yaw + math.radians(15))                            # biased IMU
    left, right = _contacts(loc, 0.40, 0.15)
    loc.observe(ctrl, contacts=[left])
    assert loc.n_pair_yaw == 0
    loc.observe(ctrl, contacts=[right])
    assert loc.n_pair_yaw == 1
    assert loc.last_report["pair_yaw_deg"] == pytest.approx(math.degrees(yaw), abs=3.0)
    assert loc.last_report["pair_yaw_sigma_deg"] == pytest.approx(math.degrees(math.sqrt(2) * 0.018 / 0.30), abs=0.1)
    # the pair pulled the heading to the wall's truth, not to the biased IMU
    err_after = abs(math.atan2(math.sin(loc.pose()[2] - yaw), math.cos(loc.pose()[2] - yaw)))
    assert err_after < math.radians(6)
    assert loc.sigma_yaw() < sig_before


def test_hard_anchor_still_available_and_discards_touch_yaw():
    scene = build_v3_scene("r2_lbend")
    xy, yaw, n = _facing_wall(scene)
    loc = _make(scene, xy, yaw, LocalizerConfig(imu_fuse=False))
    ctrl = FakeCtrl(imu_yaw=yaw + math.radians(15))
    left, right = _contacts(loc, 0.40, 0.15)
    loc.observe(ctrl, contacts=[left])
    loc.observe(ctrl, contacts=[right])
    d = math.atan2(math.sin(loc.pose()[2] - ctrl.imu_yaw), math.cos(loc.pose()[2] - ctrl.imu_yaw))
    assert abs(d) < 1e-6                     # mean sits exactly on the IMU


def test_a_move_or_a_miss_breaks_the_pair():
    scene = build_v3_scene("r2_lbend")
    xy, yaw, n = _facing_wall(scene)
    loc = _make(scene, xy, yaw, LocalizerConfig())
    ctrl = FakeCtrl(imu_yaw=yaw)
    left, right = _contacts(loc, 0.40, 0.15)
    loc.observe(ctrl, contacts=[left])
    loc.observe(ctrl, contacts=[], frees=[loc.free_obs((0.3, -0.15), (0.6, -0.15), 0.045)])
    loc.observe(ctrl, contacts=[right])
    assert loc.n_pair_yaw == 0              # the miss in between ended the pair

    class Moving(FakeCtrl):
        def odometry_increment(self):
            return np.array([0.3, 0.0]), 0.0
    loc._pending = None
    loc.observe(ctrl, contacts=[left])
    loc.observe(Moving(yaw), contacts=[right])
    assert loc.n_pair_yaw == 0              # a step between the two hands: new station


def test_imu_sigma_grows_with_turning():
    scene = build_v3_scene("r2_lbend")
    xy, yaw, n = _facing_wall(scene)
    loc = _make(scene, xy, yaw, LocalizerConfig(imu_yaw_sigma0_rad=0.02, imu_yaw_sigma_per_rad=0.01))
    assert loc.imu_sigma() == pytest.approx(0.02)
    loc.turned_rad = 2.0
    assert loc.imu_sigma() == pytest.approx(math.sqrt(0.02 ** 2 + 0.02 ** 2))
