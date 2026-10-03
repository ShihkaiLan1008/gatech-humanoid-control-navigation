from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from mjlab.entity import Entity
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import (
    matrix_from_quat,
    quat_apply_inverse,
    subtract_frame_transforms,
)

from .commands import MPCMotionCommand

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def motion_anchor_pos_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Anchor (motion anchor) position in the robot's current anchor frame. (N, 3)"""
    command = cast(MPCMotionCommand, env.command_manager.get_term(command_name))

    pos, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.anchor_pos_w,
        command.anchor_quat_w,
    )

    return pos.view(env.num_envs, -1)


def motion_anchor_ori_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Anchor (motion anchor) orientation in the robot's current anchor frame. (N, 4)"""
    command = cast(MPCMotionCommand, env.command_manager.get_term(command_name))

    _, ori = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.anchor_pos_w,
        command.anchor_quat_w,
    )
    mat = matrix_from_quat(ori)
    return mat[..., :2].reshape(mat.shape[0], -1)


def robot_body_pos_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Robot body positions in the robot's current anchor frame. (N, B, 3)"""
    command = cast(MPCMotionCommand, env.command_manager.get_term(command_name))

    num_bodies = len(command.cfg.body_names)
    pos_b, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_body_pos_w,
        command.robot_body_quat_w,
    )

    return pos_b.view(env.num_envs, -1)


def robot_body_ori_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Robot body orientations in the robot's current anchor frame. (N, B, 4)"""
    command = cast(MPCMotionCommand, env.command_manager.get_term(command_name))

    num_bodies = len(command.cfg.body_names)
    _, ori_b = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_body_pos_w,
        command.robot_body_quat_w,
    )
    mat = matrix_from_quat(ori_b)
    return mat[..., :2].reshape(mat.shape[0], -1)


def robot_base_pos_w(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Robot base position in world frame. (N, 3)"""
    asset: Entity = env.scene["robot"]
    return asset.data.root_link_pos_w


def robot_base_ori_w(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Robot base orientation in world frame. (N, 4)"""
    asset: Entity = env.scene["robot"]
    return asset.data.root_link_quat_w


def robot_joint_torque(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Measured joint-space actuator torque, canonical joint order. (N, J)"""
    asset: Entity = env.scene["robot"]
    return asset.data.qfrc_actuator


def motion_anchor_lin_vel_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Reference anchor linear velocity, expressed in the ROBOT's current
    anchor (pelvis/base) frame. (N, 3)

    World-frame velocities are not heading-invariant — the same "walk forward
    at 1 m/s" reads differently for every yaw — so the reference velocity is
    rotated into the robot's own frame, matching the pelvis-frame IMU obs."""
    command = cast(MPCMotionCommand, env.command_manager.get_term(command_name))
    return quat_apply_inverse(command.robot_anchor_quat_w, command.anchor_lin_vel_w)


def motion_anchor_ang_vel_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Reference anchor angular velocity in the ROBOT's current anchor
    (pelvis/base) frame. (N, 3)"""
    command = cast(MPCMotionCommand, env.command_manager.get_term(command_name))
    return quat_apply_inverse(command.robot_anchor_quat_w, command.anchor_ang_vel_w)

def motion_hand_force(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Reference hand contact force in world frame. (N, 3)"""
    command = cast(MPCMotionCommand, env.command_manager.get_term(command_name))
    return command.hand_forces

def motion_hand_contact(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Reference hand contact boolean (1 if in contact, 0 if not). (N,)"""
    command = cast(MPCMotionCommand, env.command_manager.get_term(command_name))
    return command.hand_contact

def motion_joint_torque(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Reference joint-space torque, canonical joint order. (N, J)"""
    command = cast(MPCMotionCommand, env.command_manager.get_term(command_name))
    return command.joint_torque

def motion_joint_pos(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Reference joint-space position, canonical joint order. (N, J)"""
    command = cast(MPCMotionCommand, env.command_manager.get_term(command_name))
    return command.joint_pos

def motion_joint_vel(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Reference joint-space velocity, canonical joint order. (N, J)"""
    command = cast(MPCMotionCommand, env.command_manager.get_term(command_name))
    return command.joint_vel

def motion_feet_contact(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Reference feet contact boolean (1 if in contact, 0 if not). (N, F)"""
    command = cast(MPCMotionCommand, env.command_manager.get_term(command_name))
    return command.feet_contact

def motion_feet_force(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Reference feet contact forces in world frame. (N, F, 3)"""
    command = cast(MPCMotionCommand, env.command_manager.get_term(command_name))
    return command.feet_grf


# --- Opt2Skill-style asymmetric AC additions -----------------------------------


def _ee_pose_b(
    command: MPCMotionCommand,
    body_names: tuple[str, ...],
    pos_w: torch.Tensor,
    quat_w: torch.Tensor,
) -> torch.Tensor:
    """World body poses -> the robot's CURRENT anchor frame, pos(3) ++ 6D ori(6)
    per body, flattened. (N, len(body_names) * 9)"""
    cols = [command.cfg.body_names.index(n) for n in body_names]
    n = len(cols)
    pos, ori = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, n, 1),
        command.robot_anchor_quat_w[:, None, :].repeat(1, n, 1),
        pos_w[:, cols],
        quat_w[:, cols],
    )
    mat = matrix_from_quat(ori)
    return torch.cat(
        [pos.reshape(pos.shape[0], -1), mat[..., :2].reshape(mat.shape[0], -1)],
        dim=-1,
    )


def motion_ee_pose_b(
    env: ManagerBasedRlEnv, command_name: str, body_names: tuple[str, ...]
) -> torch.Tensor:
    """Reference EE pose expressed in the robot's CURRENT anchor frame — the
    actor's task command. (N, len(body_names) * 9), pos(3) ++ 6D ori(6) per EE.

    Deliberately NOT the clip's EE-in-reference-anchor value: the world
    reference (where the wall is) is re-expressed against the robot's actual
    anchor pose every step, so base drift moves the commanded target in the
    policy's frame exactly the way a perception-driven planner would command
    it at deployment. The policy closes the loop on "target in my frame" and
    never needs its own world pose."""
    command = cast(MPCMotionCommand, env.command_manager.get_term(command_name))
    return _ee_pose_b(command, body_names, command.body_pos_w, command.body_quat_w)


def robot_ee_pose_b(
    env: ManagerBasedRlEnv, command_name: str, body_names: tuple[str, ...]
) -> torch.Tensor:
    """Measured EE pose in the robot's current anchor frame (pure FK), same
    layout as motion_ee_pose_b. (N, len(body_names) * 9)"""
    command = cast(MPCMotionCommand, env.command_manager.get_term(command_name))
    return _ee_pose_b(
        command, body_names, command.robot_body_pos_w, command.robot_body_quat_w
    )


def robot_hand_force(
    env: ManagerBasedRlEnv, command_name: str, sensor_name: str
) -> torch.Tensor:
    """MEASURED hand-wall normal force, (N, 2) = [left, right], >= 0 pressing.

    Critic-only privileged signal for now: the hardware has no wrist/tactile
    force sensor yet. When it lands, append this to the actor obs (with noise
    + delay DR) to close the force loop against planner perception error."""
    # lazy import (rewards imports commands; keep module import order flat)
    from .rewards import _normal_force, _sensor_indexes_for_frames

    command = cast(MPCMotionCommand, env.command_manager.get_term(command_name))
    sensor: ContactSensor = env.scene[sensor_name]
    indexes = _sensor_indexes_for_frames(sensor, command.motion.hand_frames)
    return _normal_force(sensor, indexes)


def robot_feet_grf(
    env: ManagerBasedRlEnv, command_name: str, sensor_name: str
) -> torch.Tensor:
    """MEASURED vertical foot GRF, (N, 2) = [left, right], positive = supporting.

    Needs the reduce="netforce" feet sensor (global-frame net wrench); world z
    of a supporting contact is negative, hence the sign flip."""
    from .rewards import _sensor_indexes_for_frames

    command = cast(MPCMotionCommand, env.command_manager.get_term(command_name))
    sensor: ContactSensor = env.scene[sensor_name]
    indexes = _sensor_indexes_for_frames(sensor, command.motion.feet_frames)
    data = sensor.data
    assert data.force is not None, "sensor must include 'force' in fields"
    return -data.force[:, indexes, 2]