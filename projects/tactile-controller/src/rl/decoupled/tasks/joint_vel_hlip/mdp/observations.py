"""Critic-side views of the H-LIP reference term and the feet."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def hlip_ref_traj(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Reference ``y_out`` of the H-LIP term. Shape (B, 15)."""
    return env.command_manager.get_term(command_name).y_out


def hlip_act_traj(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Measured ``y_act`` of the H-LIP term. Shape (B, 15)."""
    return env.command_manager.get_term(command_name).y_act


def hlip_ref_vel(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Reference ``dy_out`` of the H-LIP term. Shape (B, 15)."""
    return env.command_manager.get_term(command_name).dy_out


def hlip_act_vel(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Measured ``dy_act`` of the H-LIP term. Shape (B, 15)."""
    return env.command_manager.get_term(command_name).dy_act


def feet_lin_vel_w(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Linear velocities of ``asset_cfg.body_names``, world frame, flattened. Shape (B, 3*Nb), m/s."""
    robot = env.scene[asset_cfg.name]
    return robot.data.body_link_lin_vel_w[:, asset_cfg.body_ids].flatten(1)


def feet_ang_vel_w(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Angular velocities of ``asset_cfg.body_names``, world frame, flattened. Shape (B, 3*Nb), rad/s."""
    robot = env.scene[asset_cfg.name]
    return robot.data.body_link_ang_vel_w[:, asset_cfg.body_ids].flatten(1)
