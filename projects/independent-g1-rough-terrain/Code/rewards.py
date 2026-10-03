"""Energy-related reward terms for the G1 policy."""

from __future__ import annotations

import torch

from isaaclab.managers import SceneEntityCfg


def mechanical_power_l1(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Approximate positive mechanical power in kW.

    Absolute joint power is used because the simulator does not contain enough
    battery/motor data to claim real electrical or regenerative energy values.
    """
    asset = env.scene[asset_cfg.name]
    torque = asset.data.applied_torque.torch[:, asset_cfg.joint_ids]
    velocity = asset.data.joint_vel.torch[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(torque * velocity), dim=1) / 1000.0


def lower_body_power_l1(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Approximate leg mechanical power in kW for locomotion efficiency."""
    return mechanical_power_l1(env, asset_cfg)

