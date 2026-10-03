"""The kinematic punch controller (`execution.punch_kinematic`): fists stop at
map surfaces and read a force, the base servos to relative moves, and the
onboard odometry drifts but is re-anchored by the localizer."""
from __future__ import annotations

import math

import numpy as np
import pytest

from execution.punch_actions import HandCmd, HandPose, MoveCmd, PunchAction, PARALLEL_GUARD
from execution.punch_kinematic import KinematicConfig, KinematicPunchController
from execution.punch_tasks import TASKS


@pytest.fixture(scope="module")
def task():
    return TASKS["room2"]()


def _hold(c, yaw):
    return MoveCmd(None, yaw)


def test_fist_stops_at_the_wall_and_reads_force(task):
    c = KinematicPunchController(task.scene, seed=0, face_yaw_map=task.face_yaw_map)
    guards = {h: HandCmd(PARALLEL_GUARD[h]) for h in ("left", "right")}
    # drive the left fist 1.5 m straight ahead: the west wall is ~0.55 m away
    far = HandPose((1.5, 0.15, 0.18), 0.0, PARALLEL_GUARD["left"].anchor_q)
    for _ in range(int(3.0 / c.dt)):
        c.apply(PunchAction(_hold(c, task.face_yaw_map), HandCmd(far, True, 5.0), guards["right"]))
        c.step()
    tip = c.tip_pos_b("left")
    assert 0.3 < tip[0] < 1.0, "the fist must be clamped short of the command"
    assert c.hand_force_n("left") > 3.0
    assert c.hand_contacts("left") and c.hand_contacts("left")[0]["force_n"] > 3.0
    assert c.hand_force_n("right") == 0.0


def test_relative_move_arrives_and_odometry_matches_truth_in_expectation(task):
    cfg = KinematicConfig(odom_sigma_along_per_m=0.0, odom_sigma_cross_per_m=0.0,
                          odom_sigma_yaw_per_rad=0.0, odom_sigma_yaw_per_m=0.0)
    c = KinematicPunchController(task.scene, seed=0, face_yaw_map=task.face_yaw_map, cfg=cfg)
    guards = {h: HandCmd(PARALLEL_GUARD[h]) for h in ("left", "right")}
    xy0, yaw0 = c.truth_pose()
    mv = MoveCmd(None, yaw0, rel=(0.4, -0.3), move_id=1)
    for _ in range(int(6.0 / c.dt)):
        c.apply(PunchAction(mv, guards["left"], guards["right"]))
        c.step()
        if not c.moving():
            break
    assert not c.moving()
    xy1, _ = c.truth_pose()
    cth, sth = math.cos(yaw0), math.sin(yaw0)
    expected = xy0 + np.array([cth * 0.4 + sth * 0.3, sth * 0.4 - cth * 0.3])
    assert np.allclose(xy1, expected, atol=0.03)
    d_body, d_yaw = c.odometry_increment()
    assert d_body == pytest.approx((0.4, -0.3), abs=0.03)
    assert abs(d_yaw) < 0.03


def test_reset_re_marks_the_odometry(task):
    c = KinematicPunchController(task.scene, seed=0, face_yaw_map=task.face_yaw_map)
    c.reset_estimator_to((1.0, 2.0), 0.5)
    assert c.working_pose()[0] == pytest.approx((1.0, 2.0))
    d_body, d_yaw = c.odometry_increment()
    assert np.allclose(d_body, 0.0) and d_yaw == 0.0


def _action(move):
    return PunchAction(move, HandCmd(PARALLEL_GUARD["left"]),
                       HandCmd(PARALLEL_GUARD["right"]))


def test_commands_and_arrival_cannot_observe_true_pose(task):
    c = KinematicPunchController(task.scene, seed=0, face_yaw_map=0.0)
    c.reset_estimator_to((10.0, 20.0), math.pi / 2)
    action = _action(MoveCmd(None, 0.0, rel=(0.4, -0.3), move_id=1))
    c.apply(action)
    command, turn = c._base_command()
    assert c._target_xy == pytest.approx((10.3, 20.4))
    assert turn < 0.0  # estimated yaw needs a turn even though true yaw is zero
    assert c.moving()
    c._xy, c._yaw = c._target_xy.copy(), 0.0
    c.apply(action)  # repeated ID must not latch again, including at the true target
    assert c._base_command()[0] == pytest.approx(command)
    assert c._base_command()[1] == turn
    assert c.moving()
    c.reset_estimator_to(c._target_xy, 0.0)
    c._xy, c._yaw = np.array([-100.0, -200.0]), 1.0
    assert not c.moving()
    assert c._base_command()[0] == pytest.approx((0.0, 0.0))
    assert c._base_command()[1] == 0.0


@pytest.mark.parametrize("plant_gain,odom_gain", [(1.0, 2.0), (0.45, 2.0), (0.45, 1.0)])
def test_odometry_gain_controls_endpoint_while_plant_gain_changes_duration(task, plant_gain, odom_gain):
    cfg = KinematicConfig(odom_forward_gain=odom_gain, forward_deliver_frac=plant_gain,
                          odom_sigma_along_per_m=0.0, odom_sigma_cross_per_m=0.0,
                          odom_sigma_yaw_per_rad=0.0, odom_sigma_yaw_per_m=0.0)
    c = KinematicPunchController(task.scene, seed=0, face_yaw_map=0.0, cfg=cfg)
    start, _ = c.truth_pose()
    action = _action(MoveCmd(None, 0.0, rel=(1.0, 0.0), move_id=1))
    for _ in range(600):
        c.apply(action)
        c.step()
        if not c.moving():
            break
    assert not c.moving()
    assert c.working_pose()[0][0] - start[0] == pytest.approx(1.0, abs=0.02)
    assert c.truth_pose()[0][0] - start[0] == pytest.approx(1.0 / odom_gain, abs=0.02)


def test_yaw_feedback_preserves_heading_error_in_physical_motion(task):
    cfg = KinematicConfig(odom_sigma_along_per_m=0.0, odom_sigma_cross_per_m=0.0,
                          odom_sigma_yaw_per_rad=0.0, odom_sigma_yaw_per_m=0.0)
    c = KinematicPunchController(task.scene, seed=0, face_yaw_map=0.0, cfg=cfg)
    start, _ = c.truth_pose()
    c.reset_estimator_to(start, math.pi / 2)
    action = _action(MoveCmd(None, math.pi / 2, rel=(0.4, 0.0), move_id=1))
    for _ in range(200):
        c.apply(action)
        c.step()
        if not c.moving():
            break
    assert not c.moving()
    assert c.working_pose()[0] - start == pytest.approx((0.0, 0.4), abs=0.02)
    assert c.truth_pose()[0] - start == pytest.approx((0.4, 0.0), abs=0.02)
    assert c.truth_pose()[1] == 0.0  # no oracle yaw correction


def test_hold_uses_estimated_position(task):
    c = KinematicPunchController(task.scene, seed=0, face_yaw_map=0.0)
    c.reset_estimator_to((5.0, 6.0), 0.7)
    c.apply(_action(MoveCmd(None, 0.7)))
    assert not c.moving()
    assert c._target_xy == pytest.approx((5.0, 6.0))


def test_a_fist_leaving_a_wall_does_not_read_a_contact(task):
    """Press forward into the west wall, then command the same fist sideways:
    the departure must read zero force (a phantom contact here poisoned the
    filter with a wrong press axis)."""
    c = KinematicPunchController(task.scene, seed=0, face_yaw_map=task.face_yaw_map)
    guards = {h: HandCmd(PARALLEL_GUARD[h]) for h in ("left", "right")}
    fwd = HandPose((1.2, -0.15, 0.18), 0.0, PARALLEL_GUARD["right"].anchor_q)
    for _ in range(int(3.0 / c.dt)):
        c.apply(PunchAction(_hold(c, task.face_yaw_map), guards["left"], HandCmd(fwd, True, 5.0)))
        c.step()
    assert c.hand_force_n("right") > 3.0, "the forward press must land first"
    side = HandPose((0.0, -0.45, 0.18), -math.pi / 2, PARALLEL_GUARD["right"].anchor_q)
    forces = []
    for _ in range(int(0.5 / c.dt)):
        c.apply(PunchAction(_hold(c, task.face_yaw_map), guards["left"], HandCmd(side, True, 5.0)))
        c.step()
        forces.append(c.hand_force_n("right"))
    assert max(forces[1:]) == 0.0, f"leaving the wall read a force: {max(forces[1:])}"
