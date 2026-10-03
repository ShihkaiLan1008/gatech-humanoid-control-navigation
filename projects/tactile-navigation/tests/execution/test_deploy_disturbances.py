"""The deploy stack's measured disturbances and the filter's answer to them.

Measured on run out/punch/room2_plan_contact_recovery.json (G1 deploy stack,
MuJoCo vforce): contact points reported 3-5 cm off the face (IKFK tip vs
fist surface, base slide during the hold), short forward steps delivered at
~45 % of the command while leg odometry reports the command, and the base
creeping 3-8 cm toward the wall per punch hold, unseen by odometry.  Under
the library's ideal numbers (contact sigma 1.8 cm, no standing noise) a
touch 8 cm off left 15 of 600 particles and moved the belief 0.2 m ALONG
the wall, a direction no wall touch informs.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from execution.punch_belief import LocalizerConfig, PunchLocalizer
from execution.punch_kinematic import KinematicConfig, KinematicPunchController
from execution.punch_tasks import TASKS


class _Still:
    def __init__(self, yaw):
        self.yaw = yaw

    def odometry_increment(self):
        return (0.0, 0.0), 0.0

    def heading_since_reset(self):
        return self.yaw

    def reset_estimator_to(self, xy, yaw):
        pass


def _biased_cloud(cfg, seed, x_bias=-0.08):
    """room2's west wall (face x = -0.59): the belief 0.08 m too close to it,
    wide along the wall, as recorded before the leg-1 touch."""
    task = TASKS["room2_turn"](standoff_m=0.35)
    yaw = 2.9843
    loc = PunchLocalizer(task.scene, yaw, seed, cfg=cfg)
    b = loc.mb.b
    rng = np.random.default_rng(seed)
    truth = np.array([-0.1626, 1.2796])
    b.xy[:] = rng.multivariate_normal(truth + [x_bias, 0.0],
                                      [[0.03 ** 2, 0.0], [0.0, 0.35 ** 2]], size=b.n)
    b.theta[:] = yaw + rng.standard_normal(b.n) * math.radians(0.8)
    b.logw[:] = 0.0
    # the recorded touch: left fist, press axis 20 deg right of forward, tip on the face
    a = math.radians(-20.0)
    obs = loc.contact_obs((0.445, -0.005), (math.cos(a), math.sin(a)))
    return loc, obs, yaw, truth


def _touch(loc, obs, yaw):
    before = loc.pose()[:2].copy()
    sy0 = math.sqrt(loc.cov_xy()[1, 1])
    rep = loc.observe(_Still(yaw), contacts=[obs])
    after = loc.pose()[:2]
    return after - before, sy0, math.sqrt(loc.cov_xy()[1, 1]), rep["ess_before_resample"], rep["ess_guard"]


def test_ideal_numbers_collapse_the_cloud_on_a_biased_touch():
    loc, obs, yaw, _ = _biased_cloud(LocalizerConfig(prior_sigma_xy_m=0.30), seed=1)
    _d, _sy0, _sy1, ess, guard = _touch(loc, obs, yaw)
    assert guard == 0
    assert ess < 40, ess                     # the recorded failure: ~15 of 600 survive


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_ess_guard_keeps_the_along_wall_marginal(seed):
    cfg = LocalizerConfig(prior_sigma_xy_m=0.30, contact_ess_floor=0.15)
    loc, obs, yaw, truth = _biased_cloud(cfg, seed)
    d, sy0, sy1, ess, guard = _touch(loc, obs, yaw)
    assert guard >= 1
    assert ess >= 0.15 * loc.mb.b.n, ess
    # the touch corrects ACROSS the wall (toward the truth) and leaves ALONG alone
    assert d[0] > 0.04, d
    assert abs(d[1]) < 0.12, (d, ess)        # sampling noise of ~90 survivors (0.35/sqrt(90) = 0.04), not a collapse
    assert 0.85 * sy0 < sy1 < 1.15 * sy0, (sy0, sy1)
    assert abs(loc.pose()[0] - truth[0]) < 0.04


def test_standing_noise_spreads_the_cloud_in_the_body_frame():
    cfg = LocalizerConfig(prior_sigma_xy_m=0.001, prior_sigma_theta_rad=0.001,
                          stand_sigma_along_m=0.05, stand_sigma_cross_m=0.0)
    task = TASKS["room2_turn"](standoff_m=0.35)
    loc = PunchLocalizer(task.scene, math.pi, 0, cfg=cfg)
    loc._standing_noise()
    cov = loc.cov_xy()
    # body forward at yaw pi is map -x: the spread lands on x only
    assert 0.04 < math.sqrt(cov[0, 0]) < 0.06
    assert math.sqrt(cov[1, 1]) < 0.005


def test_kinematic_deploy_profile_under_delivers_forward_and_over_reports_it():
    from execution.punch_actions import HandCmd, MoveCmd, PunchAction
    from execution.punch_rules import PARALLEL_GUARD
    task = TASKS["room2_turn"](standoff_m=0.35)
    cfg = KinematicConfig(forward_deliver_frac=0.45, odom_forward_gain=2.0,
                          odom_sigma_along_per_m=0.0, odom_sigma_cross_per_m=0.0,
                          odom_sigma_yaw_per_m=0.0, odom_sigma_yaw_per_rad=0.0)
    c = KinematicPunchController(task.scene, seed=0, face_yaw_map=task.face_yaw_map, cfg=cfg)
    guards = {h: HandCmd(PARALLEL_GUARD[h]) for h in ("left", "right")}
    xy0, yaw0 = c.truth_pose()
    mv = MoveCmd(None, yaw0, rel=(0.20, -0.30), move_id=1)
    for _ in range(int(6.0 / c.dt)):
        c.apply(PunchAction(mv, guards["left"], guards["right"]))
        c.step()
        if not c.moving():
            break
    assert not c.moving()
    xy1, _ = c.truth_pose()
    cth, sth = math.cos(yaw0), math.sin(yaw0)
    d = xy1 - xy0
    fwd, lat = cth * d[0] + sth * d[1], -sth * d[0] + cth * d[1]
    assert abs(fwd - 0.20 / cfg.odom_forward_gain) < 0.02, fwd  # servo stops on reported odometry
    assert abs(lat - (-0.30)) < 0.03, lat              # ... and all of lateral (arrive_tol 0.02)
    (of, ol), _dyaw = c.odometry_increment()
    assert abs(of - fwd * 2.0) < 0.01, of              # odometry reports twice the true forward
    assert abs(ol - lat) < 0.01, ol


def test_kinematic_press_slide_moves_the_base_unseen_by_odometry():
    from execution.punch_actions import HandCmd, HandPose, MoveCmd, PunchAction
    from execution.punch_rules import PARALLEL_GUARD
    task = TASKS["room2_turn"](standoff_m=0.35)
    cfg = KinematicConfig(press_slide_m_s=0.02, odom_sigma_along_per_m=0.0, odom_sigma_cross_per_m=0.0,
                          odom_sigma_yaw_per_m=0.0, odom_sigma_yaw_per_rad=0.0)
    c = KinematicPunchController(task.scene, seed=0, face_yaw_map=task.face_yaw_map, cfg=cfg)
    xy0, yaw0 = c.truth_pose()
    hold = MoveCmd(None, yaw0)                         # stand
    left = HandCmd(HandPose.solve("left", (0.9, 0.15, 0.18)))   # a press through the west face
    right = HandCmd(PARALLEL_GUARD["right"])
    for _ in range(int(2.0 / c.dt)):
        c.apply(PunchAction(hold, left, right))
        c.step()
    assert c.hand_force_n("left") > 1.0
    xy1, _ = c.truth_pose()
    d = xy1 - xy0
    fwd = math.cos(yaw0) * d[0] + math.sin(yaw0) * d[1]
    assert 0.02 < fwd < 0.04, fwd                     # ~2 s of creep toward the wall
    (of, _ol), _ = c.odometry_increment()
    assert abs(of) < 0.005, of                         # odometry saw none of it


def _north_cloud(cfg, seed=0, interior=False):
    """The deploy run's leg-3 belief: 0.36 m from the north face (y = 3.75)
    with a 0.31 m along-heading sigma, so a quarter of the cloud sits beyond
    the wall; the truth was 1.1 m from it."""
    task = TASKS["room2_turn"](standoff_m=0.35)
    yaw = math.pi / 2
    loc = PunchLocalizer(task.scene, yaw, seed, cfg=cfg)
    b = loc.mb.b
    rng = np.random.default_rng(seed)
    b.xy[:] = rng.multivariate_normal([0.49, 3.39], [[0.13 ** 2, 0.0], [0.0, 0.31 ** 2]], size=b.n)
    if interior:
        # the cloud the prune keeps: nothing beyond the face (a real cloud gets
        # there only by walking through the wall, which predict now refutes)
        b.xy[:, 1] = np.where(b.xy[:, 1] > 3.70, 3.70 - (b.xy[:, 1] - 3.70), b.xy[:, 1])
    b.theta[:] = yaw + rng.standard_normal(b.n) * math.radians(1.5)
    b.logw[:] = 0.0
    return loc, yaw


def test_a_miss_refutes_the_particles_whose_sweep_would_have_hit_the_wall():
    cfg = LocalizerConfig(prior_sigma_xy_m=0.30, prior_sigma_theta_rad=0.05)
    loc, yaw = _north_cloud(cfg, interior=True)
    obs = loc.free_obs((0.30, 0.15), (0.54, 0.15), cfg.free_radius_m)
    for _ in range(2):
        loc.observe(_Still(yaw), frees=[obs])
    y = loc.pose()[1]
    b = loc.mb.b
    w = b.weights
    beyond = float(w[b.xy[:, 1] > 3.75].sum())
    # two sigma_free into the band the sweep demonstrably cleared
    reach = float(w[b.xy[:, 1] > 3.75 - 0.54 - cfg.free_radius_m + 2 * cfg.sigma_free_m].sum())
    assert y < 3.15, y                       # the mean moved back from the wall
    assert reach < 0.10, reach               # ... because the reachable band is refuted
    assert beyond < 0.05, beyond             # ... and the far side does not hold it


def test_the_far_side_of_a_wall_is_pruned_at_predict():
    cfg = LocalizerConfig(prior_sigma_xy_m=0.30, prior_sigma_theta_rad=0.05,
                          sigma_along_per_m=0.0, sigma_cross_per_m=0.0, sigma_yaw_per_m=0.0)
    loc, yaw = _north_cloud(cfg, interior=True)
    b = loc.mb.b
    y0 = b.xy[:, 1].copy()

    class Step(_Still):
        def odometry_increment(self):
            return (0.30, 0.0), 0.0          # 0.30 m forward: the near particles walk into the slab

    loc.predict(Step(yaw))
    w = b.weights
    would_cross = y0 + 0.30 > 3.75           # a third of the interior cloud reaches the slab
    assert np.mean(would_cross) > 0.2
    inside = float(w[(b.xy[:, 1] > 3.75) & (b.xy[:, 1] < 3.85)].sum())
    through = float(w[would_cross & (b.xy[:, 1] > 3.85)].sum())
    assert inside < 0.01, inside
    assert through < 0.01, through           # nothing walks through the wall
    assert float(w[~would_cross].sum()) > 0.99
