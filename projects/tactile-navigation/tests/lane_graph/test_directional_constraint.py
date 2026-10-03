"""Turning into a poorly localized cross-track axis requires an earlier observation."""
import math
from types import SimpleNamespace as NS

import numpy as np
import pytest

from lane_graph.constrained import belief_step, motion_position_sigma
from lane_graph.uncertainty import DEFAULT


def step(S, tangent, ds=.2, contact="none", normal=None, mode="cross_track"):
    cfg = DEFAULT.merged(position_constraint=mode, position_growth_per_step_m=0.,
                         yaw_growth_per_step_rad=0., drift_yaw_per_rad=0.,
                         min_axis_sigma_m=0., min_yaw_sigma_rad=0.)
    return belief_step(S, S[2, 2], 0., True, True, ds, np.asarray(tangent), 0.,
                       contact, normal, cfg, .1, math.pi)


def test_uncertain_x_allows_x_translation_but_blocks_y():
    S = np.diag([.2**2, .02**2, .01**2])
    assert step(S, [1., 0.])[-1]["ok"]
    assert not step(S, [0., 1.])[-1]["ok"]
    assert not step(S, [1., 0.], mode="max_axis")[-1]["ok"]


def test_touch_after_unsafe_turn_cannot_justify_that_move():
    S = np.diag([.2**2, .02**2, .01**2])
    after = step(S, [0., 1.], contact="wall", normal=[1., 0.])
    assert after[-1]["pos_sigma"] < .1
    assert not after[-1]["ok"]
    before = step(S, [1., 0.], contact="wall", normal=[1., 0.])
    assert before[-1]["ok"]
    assert step(before[0], [0., 1.])[-1]["ok"]


@pytest.mark.parametrize("corr", [-.001, .001])
def test_endpoint_includes_yaw_and_position_yaw_correlation(corr):
    S = np.diag([.03**2, .05**2, .1**2])
    S[1, 2] = S[2, 1] = corr
    ell = .7
    info = step(S, [1., 0.], ds=ell)[-1]
    expected = S[1, 1] + 2 * ell * corr + ell**2 * S[2, 2]
    assert info["pre_pos_sigma"]**2 == pytest.approx(expected)


def test_yaw_limits_long_moves_even_with_small_position_sigma():
    S = np.diag([.02**2, .02**2, .2**2])
    assert step(S, [1., 0.], ds=.1)[-1]["ok"]
    assert not step(S, [1., 0.], ds=1.)[-1]["ok"]


def test_projection_uses_off_diagonal_position_covariance():
    P = np.array([[.04, .018], [.018, .01]])
    d = np.array([1., 1.]) / math.sqrt(2)
    assert motion_position_sigma(P, d)**2 == pytest.approx(.007)


def test_feedback_reference_does_not_change_actual_motion_propagation():
    S = np.diag([.03**2, .16**2, .1**2])
    cfg = DEFAULT.merged(position_growth_per_step_m=0., yaw_growth_per_step_rad=0.,
                         drift_yaw_per_rad=0., min_axis_sigma_m=0., min_yaw_sigma_rad=0.,
                         position_constraint='cross_track')
    args = (S, S[2,2], 0., True, True, .1, np.array([1.,0.]), 0., 'none', None, cfg, .14, math.pi)
    ordinary = belief_step(*args)
    feedback = belief_step(*args, constraint_tangent=np.array([0.,1.]))
    assert not ordinary[-1]['ok'] and feedback[-1]['ok']
    assert np.allclose(ordinary[0], feedback[0])
    assert feedback[0][1,2] == pytest.approx(.1 * S[2,2])
    assert feedback[-1]['pre_pos_sigma'] == pytest.approx(.03)


def test_executor_blocks_translation_before_issuing_move():
    from execution.global_route import RouteFollowPolicy
    policy = object.__new__(RouteFollowPolicy)
    S = np.diag([.2**2, .02**2, .01**2])
    loc = NS(pose=lambda: np.zeros(3), cov3=lambda: S)
    def unexpected_move(*args, **kwargs):
        raise AssertionError("uncertain translation must not be issued")
    policy.io = NS(localizer=loc, time_s=0., move=unexpected_move)
    policy.route = NS(cfg=DEFAULT.merged(position_constraint="cross_track"),
                      position_bound_m=.1, yaw_bound_rad=math.pi,
                      grid=NS(segment_clear=lambda a,b: True))
    policy._yaw_cmd = 0.
    policy._log = lambda *args, **kwargs: None
    policy.motion_checks = []
    assert policy._goto(np.array([0., .2]), 0., {}, "test") == "motion_uncertainty"
    assert not policy.motion_checks[-1]["allowed"]
