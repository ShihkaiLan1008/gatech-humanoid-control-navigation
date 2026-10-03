import math

import numpy as np
import pytest

from local_tactile.floorplan import (
    RouteDecision,
    RouteState,
    associate_terminus,
    l_corner_floorplan,
    wrap_angle,
)


def make_plan(travel_sign: float = 1.0):
    # Start at the origin facing +x (the wall), travelling +y or -y.
    return l_corner_floorplan(
        origin_xy=(0.0, 0.0),
        travel_xy=(0.0, travel_sign),
        facing_xy=(1.0, 0.0),
        standoff_m=0.40,
        first_wall_half_length_m=1.0,
        second_wall_half_length_m=1.2,
        wall_thickness_m=0.10,
    )


def test_edge_geometry_places_face_at_the_standoff():
    plan = make_plan()
    edge0 = plan.edge(0)
    assert np.allclose(edge0.path_point(0.5), (0.0, 0.5))
    assert np.allclose(edge0.face_point(0.5), (0.40, 0.5))
    assert edge0.facing_yaw == pytest.approx(0.0)
    edge1 = plan.edge(1)
    # Edge 1's path line sits one standoff past the corner and faces back.
    assert np.allclose(edge1.path_point(0.0), (0.0, 1.40))
    assert np.allclose(edge1.face_point(0.0), (0.0, 1.00))
    assert edge1.facing_yaw == pytest.approx(-math.pi / 2)


def test_turn_direction_is_read_off_the_map_and_flips_with_travel():
    left = make_plan(+1.0)
    right = make_plan(-1.0)
    # Travelling to the robot's LEFT round a convex corner requires a RIGHT
    # turn; mirroring the scene flips the sign.  Nothing in the pipeline
    # assumes either: both come from the two edges' facing yaws.
    assert left.turn_angle_rad(1) == pytest.approx(-math.pi / 2)
    assert right.turn_angle_rad(1) == pytest.approx(math.pi / 2)
    assert left.turn_angle_rad(1) == pytest.approx(-right.turn_angle_rad(1))


def test_turn_point_is_the_intersection_of_the_two_path_lines():
    plan = make_plan()
    assert np.allclose(plan.turn_point(1, 0.0), (0.0, 1.40))
    # A registration offset slides the whole map along edge 0's tangent.
    assert np.allclose(plan.turn_point(1, 0.25), (0.0, 1.65))


def test_registration_gain_is_one_along_edge0_and_zero_across_edge1():
    plan = make_plan()
    assert plan.registration_gain(0) == pytest.approx(1.0)
    assert plan.registration_gain(1) == pytest.approx(0.0)


def test_dead_end_vertices_cannot_be_turned_at():
    plan = make_plan()
    with pytest.raises(ValueError):
        plan.turn_angle_rad(0)


def test_association_gates_out_a_terminus_that_matches_nothing():
    plan = make_plan()
    good = associate_terminus(
        plan, edge_index=0, s_hat_m=1.01, s_sigma_m=0.02,
        delta_m=0.18, delta_sigma_m=0.30, ahead_of_s_m=0.7,
    )
    assert good.status == "associated" and good.vertex_index == 1
    bad = associate_terminus(
        plan, edge_index=0, s_hat_m=3.20, s_sigma_m=0.02,
        delta_m=0.18, delta_sigma_m=0.30, ahead_of_s_m=0.7,
    )
    assert bad.status == "no_candidate"


def test_association_drops_candidates_behind_the_robot():
    plan = make_plan()
    # Wall 0's other end (a dead end) sits at s = -1.0.  Measured on its own it
    # associates; from a stance at s = 0.7 it is behind the robot and cannot be
    # the terminus being approached, so it is not even scored.
    behind = associate_terminus(
        plan, edge_index=0, s_hat_m=-0.95, s_sigma_m=0.02,
        delta_m=0.0, delta_sigma_m=0.20,
    )
    assert behind.status == "associated" and behind.vertex_index == 0
    ahead = associate_terminus(
        plan, edge_index=0, s_hat_m=-0.95, s_sigma_m=0.02,
        delta_m=0.0, delta_sigma_m=0.20, ahead_of_s_m=0.7,
    )
    assert ahead.status == "no_candidate"
    assert 0 not in dict(ahead.scores)


def test_route_continues_turns_and_asks_for_more_evidence():
    plan = make_plan()
    route = RouteState(plan=plan, edge_index=0, delta_m=0.18, delta_sigma_m=0.30)
    tol = 0.0036

    undecided = {"decided": False, "decision": None, "terminus_mean_m": 1.2,
                 "terminus_variance_m2": 0.09}
    assert route.decide(undecided, stance_s_m=0.5, pose_variance_tol_m2=tol).decision \
        is RouteDecision.NEED_MORE_EVIDENCE

    continues = {"decided": True, "decision": "CONTINUES", "terminus_mean_m": 1.25,
                 "terminus_variance_m2": 0.07}
    assert route.decide(continues, stance_s_m=0.5, pose_variance_tol_m2=tol).decision \
        is RouteDecision.CONTINUE_EDGE

    loose = {"decided": True, "decision": "ENDS", "terminus_mean_m": 1.02,
             "terminus_variance_m2": 0.02}
    assert route.decide(loose, stance_s_m=0.8, pose_variance_tol_m2=tol).decision \
        is RouteDecision.NEED_MORE_EVIDENCE

    tight = {"decided": True, "decision": "ENDS", "terminus_mean_m": 1.02,
             "terminus_variance_m2": 0.0004}
    outcome = route.decide(tight, stance_s_m=0.8, pose_variance_tol_m2=tol)
    assert outcome.decision is RouteDecision.TURN
    assert outcome.vertex_index == 1
    assert outcome.next_edge == 1
    assert outcome.turn_angle_rad == pytest.approx(-math.pi / 2)


def test_fusing_a_measured_terminus_reregisters_the_map():
    plan = make_plan()
    route = RouteState(plan=plan, edge_index=0, delta_m=0.18, delta_sigma_m=0.30)
    assert route.expected_terminus_s_m() == pytest.approx(1.18)
    fusion = route.fuse_terminus(vertex_index=1, s_hat_m=1.005, s_sigma_m=0.02)
    assert fusion["innovation_m"] == pytest.approx(-0.175, abs=1e-3)
    assert abs(route.delta_m) < 0.02  # prior error largely removed
    assert route.delta_sigma_m < 0.03
    assert route.expected_terminus_s_m() == pytest.approx(1.0, abs=0.02)


def test_advance_moves_the_route_to_the_mapped_successor():
    plan = make_plan()
    route = RouteState(plan=plan, edge_index=0)
    route.advance_to(1)
    assert route.edge_index == 1
    assert route.visited_vertices == [1]


def test_wrap_angle_normalises():
    assert wrap_angle(3 * math.pi) == pytest.approx(math.pi, abs=1e-9)
    assert wrap_angle(-3 * math.pi / 2) == pytest.approx(math.pi / 2)
