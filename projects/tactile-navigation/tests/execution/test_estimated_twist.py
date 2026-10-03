"""Servo behavior without a MuJoCo rollout: true pose must be inaccessible."""
from types import SimpleNamespace
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("mjlab")
pytest.importorskip("rl.eval.mdp.maze_twist")

from execution.estimated_twist import EstimatedPoseTwistCommand, EstimatedPoseTwistCfg
from execution.punch_actions import HandCmd, MoveCmd, PunchAction, PARALLEL_GUARD
from execution.punch_vforce import VForcePunchController


class NoTruth:
    def __getattr__(self, name):
        raise AssertionError(f"controller accessed simulator state: {name}")


def _servo():
    servo = object.__new__(EstimatedPoseTwistCommand)
    servo._env = SimpleNamespace(device="cpu", step_dt=0.02)
    servo.robot = NoTruth()
    servo.cfg = SimpleNamespace(pos_tol=0.05, yaw_tol=0.05, arrive_hysteresis=2.0,
                                decel=0.5, slew=2.0, yaw_stiffness=1.5,
                                ranges=SimpleNamespace(lin_vel_x=(-0.5, 0.5),
                                                       lin_vel_y=(-0.3, 0.3),
                                                       ang_vel_z=(-1.0, 1.0)))
    servo.pose_target_w = torch.tensor([[1.0, 0.0, 0.0]])
    servo.arrived = torch.tensor([False])
    servo.speed_scale = torch.ones(1)
    servo.vel_command_b = torch.zeros(1, 3)
    servo.time_left = torch.ones(1)
    return servo


def test_tracking_yaw_and_arrival_use_estimate_without_simulator_state():
    servo = _servo()
    servo.pose_provider = lambda: (np.array([0.0, 0.0]), math.pi / 2)
    servo._update_command()
    assert abs(float(servo.command[0, 0])) < 1e-6
    assert servo.command[0, 1] < 0  # map +x is body right at estimated +90 deg
    assert servo.command[0, 2] < 0
    assert not servo.arrived[0]
    servo.pose_provider = lambda: (np.array([1.0, 0.0]), 0.0)
    for _ in range(20):
        servo._update_command()
    assert servo.arrived[0]
    assert torch.equal(servo.command, torch.zeros(1, 3))


def test_reset_and_timer_do_not_create_a_truth_based_goal():
    servo = _servo()
    servo.pose_provider = None
    servo._resample_command(torch.tensor([0]))
    servo._update_command()
    assert torch.equal(servo.pose_target_w, torch.tensor([[1.0, 0.0, 0.0]]))
    assert torch.equal(servo.command, torch.zeros(1, 3))


def test_vforce_relative_latch_hold_recovery_and_stop_need_no_truth():
    c = object.__new__(VForcePunchController)
    c.torch, c.device = torch, "cpu"
    c.fsm = _servo()
    c.robot = NoTruth()
    c._est_xy, c._est_yaw = np.array([10.0, 20.0]), math.pi / 2
    c._estimated_speed = 0.0
    c.map_yaw_to_world = lambda yaw: yaw
    c.mj = SimpleNamespace(speed_scale_locomotion=0.8)
    c.hands = {"left": None, "right": None}
    c._hand = lambda *args: None
    hand = HandCmd(PARALLEL_GUARD["left"])
    action = PunchAction(MoveCmd(None, 0.0, rel=(0.4, -0.3), move_id=1), hand, hand)
    c.apply(action)
    assert c._rel_target == pytest.approx((10.3, 20.4))
    c._est_xy += 0.2
    c.apply(action)
    assert c._rel_target == pytest.approx((10.3, 20.4))
    c._hold_here()
    assert c.fsm.pose_target_w[0, :2].numpy() == pytest.approx(c._est_xy)
    c._set_goal((11.0, 22.0), 0.3, 0.5, hold=False, standing=False)
    assert c.fsm.pose_target_w[0].numpy() == pytest.approx((11.0, 22.0, 0.3))
    assert c.moving()
    c.fsm.arrived[:] = True
    assert not c.moving()
    c._estimated_speed = 0.1
    assert c.moving()


def test_actual_vforce_config_selects_estimated_servo():
    import mjlab.tasks
    from mjlab.tasks.registry import load_env_cfg
    from execution.vforce_backend import VForceConfig
    cfg = load_env_cfg(VForceConfig().task, play=True).commands["twist"]
    converted = VForcePunchController._configure_twist(None, cfg)
    assert isinstance(converted, EstimatedPoseTwistCfg)
    assert converted.ranges == cfg.ranges
    assert converted.decel == cfg.decel
