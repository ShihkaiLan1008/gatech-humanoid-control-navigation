"""Stance-foot holonomic rewards on the planner's latched anchor (robot_rl
``holonomic_constraint`` / ``holonomic_constraint_vel``)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from rl.decoupled.mdp.commands.foot_placement import _wrap, _yaw_from_quat_wxyz
from rl.decoupled.mdp.commands.hlip import _rot2

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def _stance(env: ManagerBasedRlEnv, command_name: str):
    P = env.command_manager.get_term(command_name)
    rows = torch.arange(env.num_envs, device=env.device)
    stance = 1 - P.swing
    return P, rows, stance


def holonomic_constraint(env: ManagerBasedRlEnv, command_name: str,
                         sigma_pose: float, z_offset: float = 0.0,
                         foot_body_names: tuple[str, str] = ("left_ankle_roll_link", "right_ankle_roll_link"),
                         ) -> torch.Tensor:
    """exp(-‖e‖²/σ²) of the STANCE foot vs the pose latched at the step event:
    e = [Δx, Δy, z_site − z_offset, roll, Δyaw]. Shape (B,), in (0, 1]."""
    P, rows, stance = _stance(env, command_name)
    site = P.feet_site_pos_w[rows, stance]
    anchor = P.ref_feet_pos_w[rows, stance]
    psi = P.ref_feet_yaw[rows, stance]
    feet_ids = P.robot.find_bodies(list(foot_body_names), preserve_order=True)[0]
    q = P.robot.data.body_link_quat_w[:, feet_ids][rows, stance]
    w, x, y, z = q.unbind(-1)
    roll = torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    dyaw = _wrap(_yaw_from_quat_wxyz(q) - psi)
    dxy = _rot2(-psi, site[:, :2] - anchor[:, :2])
    e2 = (dxy.square().sum(-1) + (site[:, 2] - z_offset).square() + roll.square() + dyaw.square())
    return torch.exp(-e2 / sigma_pose ** 2)


def holonomic_constraint_vel(env: ManagerBasedRlEnv, command_name: str, sigma_vel: float,
                             foot_body_names: tuple[str, str] = ("left_ankle_roll_link", "right_ankle_roll_link"),
                             ) -> torch.Tensor:
    """exp(-(‖v‖² + ω_z²)/σ²) of the STANCE ankle link (world linear velocity
    and yaw rate). Shape (B,), in (0, 1]."""
    P, rows, stance = _stance(env, command_name)
    feet_ids = P.robot.find_bodies(list(foot_body_names), preserve_order=True)[0]
    v = P.robot.data.body_link_lin_vel_w[:, feet_ids][rows, stance]
    wz = P.robot.data.body_link_ang_vel_w[:, feet_ids][rows, stance, 2]
    return torch.exp(-(v.square().sum(-1) + wz.square()) / sigma_vel ** 2)
