"""Generic gait helpers shared by the MPC-guided tasks (ex-env_cfgs locals)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def feet_too_near(
    env: "ManagerBasedRlEnv",
    min_distance: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalise feet being closer than *min_distance* (XY plane)."""
    asset = env.scene[asset_cfg.name]
    foot_pos = asset.data.site_pos_w[:, asset_cfg.site_ids, :]  # [B, 2, 3]
    diff_xy = foot_pos[:, 0, :2] - foot_pos[:, 1, :2]
    dist_xy = torch.norm(diff_xy, dim=-1)
    return torch.clamp(min_distance - dist_xy, min=0.0)


def gait_symmetry(
    env: "ManagerBasedRlEnv",
    left_sensor_name: str,
    right_sensor_name: str,
    command_name: str = "twist",
    command_threshold: float = 0.1,
) -> torch.Tensor:
    """Reward alternating biped gait (L/R feet anti-phase)."""
    left_sensor: ContactSensor = env.scene[left_sensor_name]
    right_sensor: ContactSensor = env.scene[right_sensor_name]
    left_air_time = left_sensor.data.current_air_time
    right_air_time = right_sensor.data.current_air_time
    assert left_air_time is not None and right_air_time is not None
    left_in_air = left_air_time[:, 0] > 0
    right_in_air = right_air_time[:, 0] > 0
    alternating = (left_in_air ^ right_in_air).float()
    command = env.command_manager.get_command(command_name)
    if command is not None:
        cmd_mag = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
        alternating *= (cmd_mag > command_threshold).float()
    return alternating
