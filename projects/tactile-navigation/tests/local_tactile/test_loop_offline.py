"""Planner + route + stance-placement loop against oracle geometry.

No simulator: a probe at along-edge coordinate ``s`` hits iff ``s <= 1.00``
(the true wall terminus).  This pins the decision loop the MuJoCo episode
drives — how many probes and stations it needs, that it recovers from an
over-optimistic prior by walking BACK, and that the turn it commands comes
from the map — without a GPU.
"""

import math

import pytest

from local_tactile.ate import LocalGeometryATE, ProbeStance
from local_tactile.floorplan import RouteDecision, RouteState, l_corner_floorplan
from local_tactile.pipeline import IntermittentConfig, plan_stance_s

TRUE_TERMINUS = 1.0
HL0, HL1 = 1.0, 1.2


def build(travel_sign: float = 1.0, **overrides):
    cfg = IntermittentConfig(**overrides)
    plan = l_corner_floorplan(
        origin_xy=(0.0, 0.0),
        travel_xy=(0.0, travel_sign),
        facing_xy=(1.0, 0.0),
        standoff_m=cfg.wall_standoff_m,
        first_wall_half_length_m=HL0,
        second_wall_half_length_m=HL1,
        wall_thickness_m=0.10,
    )
    route = RouteState(
        plan=plan, edge_index=0,
        delta_m=cfg.map_registration_offset_m,
        delta_sigma_m=cfg.map_registration_sigma_m,
    )
    ate = LocalGeometryATE(
        prior_mean_m=route.expected_terminus_s_m(),
        prior_sigma_m=route.expected_terminus_sigma_m(),
        p_detect=cfg.p_detect,
        p_false_alarm=cfg.p_false_alarm,
        edge_softness_m=cfg.edge_softness_m,
        id_entropy_tol_nats=cfg.id_entropy_tol_nats,
        pose_variance_tol_m2=cfg.pose_variance_tol_m2,
        decision_probability=cfg.decision_probability,
    )
    return cfg, plan, route, ate


def drive(cfg, plan, route, ate, *, stance_error_m: float = 0.0):
    """Run the loop; return a record of what happened."""
    s_now = 0.0
    stances, decisions, probes = [], [], 0
    for _station in range(cfg.max_probe_stations):
        planned = plan_stance_s(
            s_now_m=s_now,
            terminus_mean_m=ate.terminus_mean_m,
            terminus_sigma_m=math.sqrt(max(ate.terminus_variance_m2, 0.0)),
            cfg=cfg,
        )
        if planned is None:
            break
        s_stance = planned + stance_error_m
        stances.append(s_stance)
        stance = ProbeStance(edge_index=0, s_m=s_stance, fan_offsets_m=cfg.probe_fan_offsets_m)
        ate.begin_stance(stance)
        for _probe in range(cfg.max_probes_per_stance):
            candidate = ate.select(stance)
            if candidate is None:
                break
            probes += 1
            if candidate.s_probe_m <= TRUE_TERMINUS:
                ate.update_contact(contact_s_m=candidate.s_probe_m)
            else:
                ate.update_swept_free(probe_s_m=candidate.s_probe_m, swept_past_face_m=0.05)
            ate.mark_executed(candidate)
        outcome = route.decide(
            ate.snapshot(stance), stance_s_m=s_stance,
            pose_variance_tol_m2=cfg.pose_variance_tol_m2,
            gate_chi2=cfg.association_gate_chi2,
        )
        decisions.append(outcome)
        s_now = s_stance
        if outcome.decision is RouteDecision.TURN:
            route.fuse_terminus(
                vertex_index=outcome.vertex_index,
                s_hat_m=ate.terminus_mean_m,
                s_sigma_m=math.sqrt(max(ate.terminus_variance_m2, 1e-12)),
            )
            break
    return {"stances": stances, "decisions": decisions, "probes": probes}


def test_nominal_loop_continues_once_then_turns():
    cfg, plan, route, ate = build()
    result = drive(cfg, plan, route, ate)
    kinds = [d.decision for d in result["decisions"]]
    assert kinds[0] is RouteDecision.CONTINUE_EDGE
    assert kinds[-1] is RouteDecision.TURN
    assert result["probes"] <= cfg.max_probe_stations * cfg.max_probes_per_stance
    assert ate.terminus_mean_m == pytest.approx(TRUE_TERMINUS, abs=0.06)
    assert route.edge_index == 0  # advance_to is the executive's job, not decide's


def test_turn_is_read_off_the_map_and_mirrors_with_travel_direction():
    outcomes = {}
    for name, sign in (("left", 1.0), ("right", -1.0)):
        cfg, plan, route, ate = build(sign)
        result = drive(cfg, plan, route, ate)
        turn = result["decisions"][-1]
        assert turn.decision is RouteDecision.TURN
        assert turn.next_edge == 1
        outcomes[name] = turn.turn_angle_rad
    assert outcomes["left"] == pytest.approx(-math.pi / 2)
    assert outcomes["right"] == pytest.approx(+math.pi / 2)


def test_registration_prior_error_is_removed_by_touch():
    cfg, plan, route, ate = build()
    assert route.delta_m == pytest.approx(cfg.map_registration_offset_m)
    drive(cfg, plan, route, ate)
    assert abs(route.delta_m) < 0.06  # started 0.18 m out
    assert route.delta_sigma_m < cfg.map_registration_sigma_m / 2.0
    goal = plan.path_point_w(1, cfg.goal_leg_m, route.delta_m)
    true_goal = plan.path_point_w(1, cfg.goal_leg_m, 0.0)
    assert float(((goal - true_goal) ** 2).sum()) ** 0.5 < 0.10


@pytest.mark.parametrize("stance_error_m", [-0.09, -0.05, 0.0, 0.05, 0.09, 0.15])
def test_loop_converges_under_realistic_stance_error(stance_error_m):
    cfg, plan, route, ate = build()
    result = drive(cfg, plan, route, ate, stance_error_m=stance_error_m)
    assert result["decisions"][-1].decision is RouteDecision.TURN
    assert ate.terminus_mean_m == pytest.approx(TRUE_TERMINUS, abs=0.06)


def test_an_overshooting_stance_is_recovered_by_walking_back():
    """The whole fan lands past the wall, every probe misses, and the belief
    then puts the terminus behind the robot: the next stance must go
    backwards, not forwards."""
    cfg, plan, route, ate = build()
    result = drive(cfg, plan, route, ate, stance_error_m=0.09)
    stances = result["stances"]
    assert len(stances) >= 3
    assert stances[2] < stances[1]  # walked back
    assert result["decisions"][1].decision is RouteDecision.NEED_MORE_EVIDENCE
    assert result["decisions"][-1].decision is RouteDecision.TURN


def test_stance_placement_guards_are_hard():
    """Every returned stance respects all three bounds; when they cannot be
    met together the planner reports no stance rather than a bad one."""
    cfg = IntermittentConfig()
    admissible = refused = 0
    for mean in (0.6, 1.0, 1.4, 2.0):
        for sigma in (0.01, 0.05, 0.2):
            for s_now in (0.0, 0.5, 1.2):
                s = plan_stance_s(s_now_m=s_now, terminus_mean_m=mean,
                                  terminus_sigma_m=sigma, cfg=cfg)
                if s is None:
                    refused += 1
                    continue
                admissible += 1
                assert s <= mean - cfg.stance_base_clearance_m + 1e-9
                assert s <= s_now + cfg.max_leg_m + 1e-9
                assert s >= s_now - cfg.max_backup_m - 1e-9
    assert admissible and refused  # both branches are exercised


def test_stance_placement_refuses_when_the_backup_budget_is_too_small():
    cfg = IntermittentConfig(max_backup_m=0.10)
    assert plan_stance_s(s_now_m=1.2, terminus_mean_m=0.6,
                         terminus_sigma_m=0.02, cfg=cfg) is None
    assert plan_stance_s(s_now_m=0.5, terminus_mean_m=1.18,
                         terminus_sigma_m=0.20, cfg=cfg) is not None
