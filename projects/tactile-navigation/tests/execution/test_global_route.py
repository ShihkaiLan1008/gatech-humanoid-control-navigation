"""The global plan -> local policy hand-off (`execution.global_route`)."""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

from execution.global_route import (filter_consistent_config, legs_from_path, plan_global,
                                    route_record)
from execution.punch_belief import LocalizerConfig
from lane_graph.scene3 import build_v3_scene


@pytest.fixture(scope="module")
def route():
    scene = build_v3_scene("r2_lbend")
    cfg = filter_consistent_config(LocalizerConfig(), ["max_explored_states=100000", "goal_sigma_multiplier=0"])
    r = plan_global(scene, math.pi / 2, cfg, 0.5, math.radians(30), 0.03, 0.02)
    assert r is not None, "loose bounds must be feasible on the L-room"
    return r


def test_legs_cover_the_path_and_turn_only_between_legs(route):
    total = sum(l.s_end for l in route.legs)
    assert total == pytest.approx(route.metrics["distance_m"], abs=1e-6)
    assert route.legs[-1].kind == "stop" and all(l.kind == "turn" for l in route.legs[:-1])
    for a, b in zip(route.legs[:-1], route.legs[1:]):
        # consecutive legs differ in travel direction or commanded heading
        assert (abs(float(a.along @ b.along) - 1) > 1e-6) or abs(a.yaw - b.yaw) > 1e-6
        # the next leg starts where this one ends
        assert np.allclose(a.pt + a.s_end * a.along, b.pt, atol=1e-9)


def test_touch_legs_face_the_surface_and_free_legs_face_travel(route):
    for l in route.legs:
        if l.touch:
            n = np.asarray(l.normal)
            assert abs(math.atan2(-n[1], -n[0]) - l.yaw) < 1e-9
        else:
            assert abs(math.atan2(l.along[1], l.along[0]) - l.yaw) < 1e-9


def test_stations_sit_on_touch_cells_in_route_order(route):
    g = [s.g for s in route.stations]
    assert g == sorted(g)
    for s in route.stations:
        cell = route.grid.cell_of(s.xy)
        assert cell in route.grid.contacts
        assert 0 <= s.leg < len(route.legs)
        assert -1e-6 <= s.s <= route.legs[s.leg].s_end + 1e-6
    assert len(route.stations) == route.metrics["probe_count"]


def test_record_is_json_and_the_plan_is_within_its_bound(route):
    rec = route_record(route)
    json.dumps(rec, allow_nan=False)
    assert rec["metrics"]["peak_position_sigma_m"] <= route.position_bound_m + 1e-9
    assert len(rec["path_xy"]) == len(route.plan.path)


def test_anisotropic_prior_agrees_between_planner_and_pf_in_map_frame():
    from execution.punch_belief import PunchLocalizer
    from execution.punch_tasks import TASKS
    task = TASKS['room2']()
    loc_cfg = LocalizerConfig(prior_sigma_xy_m=(0.03, 0.10))
    loc = PunchLocalizer(task.scene, task.face_yaw_map, seed=0, cfg=loc_cfg)
    cfg = filter_consistent_config(loc_cfg, ['contact_standoff_m=0.4'])
    route = plan_global(task.scene, task.face_yaw_map, cfg, .10, math.radians(15),
                        prior_sigma_xy_m=loc_cfg.prior_sigma_xy_m)
    assert route is not None
    assert np.diag(route.prior.cov) == pytest.approx([.03**2, .10**2])
    assert np.sqrt(np.diag(loc.cov_xy())) == pytest.approx([.03, .10], rel=.10)
    assert loc.sigma_yaw() == pytest.approx(loc_cfg.prior_sigma_theta_rad, rel=.10)


def test_dense_schedule_gives_a_station_per_touch_cell_visited(route):
    dense = frozenset((i, 1) for i, (a, b) in enumerate(zip(route.plan.path[:-1], route.plan.path[1:]))
                      if b in route.grid.contacts)
    legs, stations = legs_from_path(route.grid, route.plan.path, dense, route.cfg)
    assert len(stations) == len(dense)
    assert len(legs) == len(route.legs)


def test_default_retains_every_available_contact(route):
    available = sum(bool(route.grid.contacts.get(c)) for c in route.plan.path[1:])
    assert len(route.stations) == available


def test_scheduled_touch_does_not_override_eig_gate(route):
    from types import SimpleNamespace as NS
    from execution.global_route import RouteFollowPolicy, MIN_TOUCH_IG_NATS
    localizer = NS(cov_xy=lambda: np.eye(2)*.1**2, sigma_yaw=lambda: .05)
    selector = NS(choose_single=lambda **kw: {'expected_ig': MIN_TOUCH_IG_NATS / 2})
    io = NS(localizer=localizer)
    policy = RouteFollowPolicy(io, route, selector)
    out = policy._touch_until(route.legs[0], 0., True, {})
    assert out['touches'] == [] and out['reason'] == 'no touch worth its time'


@pytest.mark.parametrize('informative, expected_probes', [(False, 1), (True, 3)])
def test_miss_must_reduce_uncertainty_to_continue_probing(route, informative, expected_probes):
    from types import SimpleNamespace as NS
    from execution.global_route import RouteFollowPolicy, MIN_TOUCH_IG_NATS

    sigmas = [.2, .1]
    probes = []
    localizer = NS(cov_xy=lambda: np.eye(2) * sigmas[0]**2,
                   sigma_yaw=lambda: sigmas[1])
    pick = dict(expected_ig=2 * MIN_TOUCH_IG_NATS, hand='left',
                spec=NS(guard=None), alpha_deg=0., p_hit=.5, margin_m=0.)
    selector = NS(choose_single=lambda **kw: pick)

    def punch(*args):
        probes.append(args)
        if informative:
            sigmas[0] *= .8
        return NS(contact=False, status='complete')

    policy = RouteFollowPolicy(NS(localizer=localizer, punch=punch), route,
                               selector, max_touches=3)
    out = policy._touch_until(route.legs[0], 0., True, {})
    assert len(probes) == expected_probes
    assert all(not t['hit'] for t in out['touches'])
    assert out['reason'] == ('touch limit' if informative
                             else 'no progress: probe did not reduce uncertainty')


def test_timeout_does_not_punch_and_sideways_pose_is_not_complete(route):
    from types import SimpleNamespace as NS
    from execution.global_route import RouteFollowPolicy
    seg = route.legs[-1]
    endpoint = seg.pt + seg.s_end * seg.along
    sideways = np.array([-seg.along[1], seg.along[0]]) * 3.
    localizer = NS(pose=lambda: np.r_[endpoint + sideways, seg.yaw],
                   sigma_m=lambda: .1, sigma_yaw=lambda: .05)
    policy = RouteFollowPolicy(NS(localizer=localizer), route, NS())
    policy._n_stations = 0; policy._guards = {}; policy.k = len(route.legs)-1
    policy._goto = lambda *a: 'timeout'
    def no_punch(*a):
        raise AssertionError('a timed-out move must not punch')
    policy._touch_until = no_punch
    assert policy.step_station() != 'route_complete'
    assert not policy.touches


def test_replayed_route_preserves_free_travel_metric(route):
    scene = build_v3_scene('r2_lbend')
    replay = plan_global(scene, route.yaw0, route.cfg, route.position_bound_m,
                         route.yaw_bound_rad, path=route.plan.path)
    assert replay.plan.free_distance_m == pytest.approx(route.plan.free_distance_m, abs=1e-9)


@pytest.mark.parametrize('G', [np.eye(3), [[0., 0., 1.]], [[0., 1., 0.], [0., 0., 1.]]])
def test_projected_eig_matches_gaussian_posterior_entropy(G):
    from types import SimpleNamespace as NS
    from execution.global_route import _expected_ig

    A = np.array([[.1, .02, .01], [0., .08, -.02], [.01, 0., .07]])
    P, H = A @ A.T, np.array([1., 0., -.3])
    v = P @ H
    # Independent reference: form both conditional covariances explicitly.
    posteriors = [P - np.outer(v, v) / (H @ v + .018**2),
                  P - .35 * np.outer(v, v) / (H @ v)]
    def gain(prior, posterior):
        return .5 * (np.linalg.slogdet(prior)[1] - np.linalg.slogdet(posterior)[1])
    h = NS(body_k=0, H=H, p_hit=.7,
           ig_hit=gain(P, posteriors[0]), ig_miss=gain(P, posteriors[1]))
    G = np.atleast_2d(G)
    expected = sum(w * gain(G @ P @ G.T, G @ Q @ G.T)
                   for w, Q in zip([.7, .3], posteriors))
    assert _expected_ig(h, P, G) == pytest.approx(expected)


def test_yaw_information_favours_a_separated_contact_after_first_hit():
    from types import SimpleNamespace as NS
    from execution.global_route import _expected_ig

    P = np.diag([.1**2, .1**2, .15**2])
    first, other = np.array([1., 0., -.15]), np.array([1., 0., .15])
    v = P @ first
    P -= np.outer(v, v) / (first @ v + .018**2)
    def score(H):
        h = NS(body_k=0, H=H, p_hit=1., ig_miss=0.,
               ig_hit=.5 * np.log1p(H @ P @ H / .018**2))
        return _expected_ig(h, P, [[0., 0., 1.]])
    assert score(other) > score(first)


def test_followup_stops_when_same_hit_only_sharpens_sufficient_position(route, monkeypatch):
    from types import SimpleNamespace as NS
    from execution import global_route as gr

    seg = route.legs[0]
    normal = np.array([-seg.along[1], seg.along[0], 0.])
    P = np.diag([.1**2, .1**2, .04**2])
    loc = NS(pose=lambda: np.zeros(3), cov3=lambda: P,
             cov_xy=lambda: P[:2, :2], sigma_yaw=lambda: np.sqrt(P[2, 2]))
    selector = gr.SingleTouchSelector(None, loc)
    evaluations = []
    def evaluate(*args, meta):
        ig = .5 * np.log1p(normal @ P @ normal / .018**2)
        h = NS(body_k=0, H=normal, alpha=0., p_hit=1., ig_hit=ig, ig_miss=0., margin_m=.1)
        ev = NS(hands={'left': [h], 'right': [h]}, best_ij=(0, 0), meta=meta)
        evaluations.append(ev)
        return ev
    selector.eig.evaluate = evaluate
    monkeypatch.setattr(gr, 'spec_at', lambda *a: NS(guard=None))
    punches = []
    def punch(*args):
        v = P @ normal
        P[:] -= np.outer(v, v) / (normal @ v + .018**2)
        punches.append({})
        return NS(contact=True, status='complete')
    policy = gr.RouteFollowPolicy(NS(localizer=loc, punch=punch, punches=punches), route, selector)
    policy._planned_sigma = lambda g: (.05, .02)
    out = policy._touch_until(seg, 0., True, {})
    assert len(punches) == 1
    assert evaluations[-1].meta['single_pose_ig'] > gr.MIN_TOUCH_IG_NATS
    assert evaluations[-1].meta['single_expected_ig'] == 0.
    assert evaluations[-1].meta['information_projection'] == [[0., 0., 1.]]
    assert out['reason'] == 'remaining touches not worth their time'

    # After motion replenishes cross-track uncertainty, this angle remains
    # eligible. No action blacklist or hand alternation has been introduced.
    P[:2, :2] = np.eye(2) * .1**2
    out = policy._touch_until(seg, .4, True, {})
    assert len(out['touches']) == 1 and len(punches) == 2
