from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_apply_inverse

from .commands import MPCMotionCommand
from .rewards import _get_body_indexes, _normal_force, _sensor_indexes_for_frames

if TYPE_CHECKING:
    from mjlab.entity import Entity
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.managers.scene_entity_config import SceneEntityCfg


def bad_anchor_pos(
    env: ManagerBasedRlEnv, command_name: str, threshold: float
) -> torch.Tensor:
    command = cast(MPCMotionCommand, env.command_manager.get_term(command_name))
    return (
        torch.norm(command.anchor_pos_w - command.robot_anchor_pos_w, dim=1) > threshold
    )


def bad_anchor_pos_z_only(
    env: ManagerBasedRlEnv, command_name: str, threshold: float
) -> torch.Tensor:
    command = cast(MPCMotionCommand, env.command_manager.get_term(command_name))
    return (
        torch.abs(command.anchor_pos_w[:, -1] - command.robot_anchor_pos_w[:, -1])
        > threshold
    )


def bad_anchor_ori(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    threshold: float,
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]

    command = cast(MPCMotionCommand, env.command_manager.get_term(command_name))
    motion_projected_gravity_b = quat_apply_inverse(
        command.anchor_quat_w, asset.data.gravity_vec_w
    )

    robot_projected_gravity_b = quat_apply_inverse(
        command.robot_anchor_quat_w, asset.data.gravity_vec_w
    )

    return (
        motion_projected_gravity_b[:, 2] - robot_projected_gravity_b[:, 2]
    ).abs() > threshold


def bad_motion_body_pos(
    env: ManagerBasedRlEnv,
    command_name: str,
    threshold: float,
    body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
    command = cast(MPCMotionCommand, env.command_manager.get_term(command_name))

    body_indexes = _get_body_indexes(command, body_names)
    error = torch.norm(
        command.body_pos_relative_w[:, body_indexes]
        - command.robot_body_pos_w[:, body_indexes],
        dim=-1,
    )
    return torch.any(error > threshold, dim=-1)


def bad_motion_body_pos_z_only(
    env: ManagerBasedRlEnv,
    command_name: str,
    threshold: float,
    body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
    command = cast(MPCMotionCommand, env.command_manager.get_term(command_name))

    body_indexes = _get_body_indexes(command, body_names)
    error = torch.abs(
        command.body_pos_relative_w[:, body_indexes, -1]
        - command.robot_body_pos_w[:, body_indexes, -1]
    )
    return torch.any(error > threshold, dim=-1)


def bad_global_body_pos(
    env: ManagerBasedRlEnv,
    command_name: str,
    threshold: float,
    body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
    """WORLD-frame body position kill, restricted to body_names (the hands).

    The static replacement for the GMT adaptive termination in the
    planner-relative design: the base is free to drift, so the one
    world-anchored relationship left is hand <-> wall. Once the commanded
    target (the clip's world EE point) is this far from the hand, the episode
    has left the local reach envelope the design assumes ("wall always
    reachable") and its data is useless — kill instead of dragging a doomed
    rollout to time-out."""
    command = cast(MPCMotionCommand, env.command_manager.get_term(command_name))
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.norm(
        command.body_pos_w[:, body_indexes]
        - command.robot_body_pos_w[:, body_indexes],
        dim=-1,
    )
    return torch.any(error > threshold, dim=-1)


def bad_gmt_tracking(
    env: ManagerBasedRlEnv,
    command_name: str,
) -> torch.Tensor:
    """GMT adaptive tracking termination: terminate when the GLOBAL max
    key-body position error exceeds the env's clip's threshold E_i.

    E_i tightens from gmt_e_range[1] (fresh clip) to gmt_e_range[0] (converged)
    as the clip's completion level c_i decays — the termination half of the GMT
    curriculum (the sampling half lives in MPCMotionCommand._gmt_probs). Under
    clip_sampling_mode="uniform" c never decays, so this degrades to a static
    loose cutoff at gmt_e_range[1]."""
    command = cast(MPCMotionCommand, env.command_manager.get_term(command_name))
    return command.gmt_tracking_error() > command.gmt_error_threshold


def bad_hand_contact(
    env: ManagerBasedRlEnv,
    command_name: str,
    sensor_name: str,
    force_threshold: float = 5.0,
) -> torch.Tensor:
    """Terminate when a hand presses the wall (force magnitude above threshold)
    while the schedule says airborne. Grazing touches below the threshold are
    left to penalty_hand_contact instead of ending the episode."""
    command = cast(MPCMotionCommand, env.command_manager.get_term(command_name))
    sensor: ContactSensor = env.scene[sensor_name]
    indexes = _sensor_indexes_for_frames(sensor, command.motion.hand_frames)
    data = sensor.data
    assert data.force is not None, "sensor must include 'force' in fields"
    force_mag = torch.norm(data.force[:, indexes], dim=-1)  # [B, H]
    bad = (command.hand_contact == 0) & (force_mag > force_threshold)
    return bad.any(dim=-1)


def bad_hand_contact_force(
    env: ManagerBasedRlEnv,
    command_name: str,
    sensor_name: str,
    force_threshold: float = 30.0,
) -> torch.Tensor:
    """Terminate when the measured hand normal force deviates from the reference
    by more than force_threshold during scheduled contact (slam guard). The
    reference is negative = pressing, the sensor normal is >= 0, so the target
    is -hand_forces. Threshold should sit well above the reference band (5-20 N)
    so brief touchdown transients don't kill the episode."""
    command = cast(MPCMotionCommand, env.command_manager.get_term(command_name))
    sensor: ContactSensor = env.scene[sensor_name]
    indexes = _sensor_indexes_for_frames(sensor, command.motion.hand_frames)
    measured = _normal_force(sensor, indexes)  # [B, H]
    error = torch.abs(measured - (-command.hand_forces))
    bad = (command.hand_contact > 0) & (error > force_threshold)
    return bad.any(dim=-1)
