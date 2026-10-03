# TODO: joint-torque tracking reward (reference lives on the clip's joint_torque).
# Hand force/contact + feet GRF/stance rewards below consume
# MPCMotionCommand.{hand_forces,hand_contact,feet_grf,feet_contact}.
from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_error_magnitude

from .commands import MPCMotionCommand

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def _get_body_indexes(
    command: MPCMotionCommand, body_names: tuple[str, ...] | None
) -> list[int]:
    return [
        i
        for i, name in enumerate(command.cfg.body_names)
        if (body_names is None) or (name in body_names)
    ]


def motion_global_anchor_position_error_exp(
    env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
    command = cast(MPCMotionCommand, env.command_manager.get_term(command_name))
    error = torch.sum(
        torch.square(command.anchor_pos_w - command.robot_anchor_pos_w), dim=-1
    )
    return torch.exp(-error / std**2)


def motion_global_anchor_orientation_error_exp(
    env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
    command = cast(MPCMotionCommand, env.command_manager.get_term(command_name))
    error = (
        quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w) ** 2
    )
    return torch.exp(-error / std**2)


def motion_relative_body_position_error_exp(
    env: ManagerBasedRlEnv,
    command_name: str,
    std: float,
    body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
    command = cast(MPCMotionCommand, env.command_manager.get_term(command_name))
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(
            command.body_pos_relative_w[:, body_indexes]
            - command.robot_body_pos_w[:, body_indexes]
        ),
        dim=-1,
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_relative_body_orientation_error_exp(
    env: ManagerBasedRlEnv,
    command_name: str,
    std: float,
    body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
    command = cast(MPCMotionCommand, env.command_manager.get_term(command_name))
    body_indexes = _get_body_indexes(command, body_names)
    error = (
        quat_error_magnitude(
            command.body_quat_relative_w[:, body_indexes],
            command.robot_body_quat_w[:, body_indexes],
        )
        ** 2
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_linear_velocity_error_exp(
    env: ManagerBasedRlEnv,
    command_name: str,
    std: float,
    body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
    command = cast(MPCMotionCommand, env.command_manager.get_term(command_name))
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(
            command.body_lin_vel_w[:, body_indexes]
            - command.robot_body_lin_vel_w[:, body_indexes]
        ),
        dim=-1,
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_angular_velocity_error_exp(
    env: ManagerBasedRlEnv,
    command_name: str,
    std: float,
    body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
    command = cast(MPCMotionCommand, env.command_manager.get_term(command_name))
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(
            command.body_ang_vel_w[:, body_indexes]
            - command.robot_body_ang_vel_w[:, body_indexes]
        ),
        dim=-1,
    )
    return torch.exp(-error.mean(-1) / std**2)


def self_collision_cost(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    force_threshold: float = 10.0,
) -> torch.Tensor:
    """Penalize self-collisions.

    When the sensor provides force history (from ``history_length > 0``),
    counts substeps where any contact force exceeds *force_threshold*.
    Falls back to the instantaneous ``found`` count otherwise.
    """
    sensor: ContactSensor = env.scene[sensor_name]
    data = sensor.data
    if data.force_history is not None:
        # force_history: [B, N, H, 3]
        force_mag = torch.norm(data.force_history, dim=-1)  # [B, N, H]
        hit = (force_mag > force_threshold).any(dim=1)  # [B, H]
        return hit.sum(dim=-1).float()  # [B]
    assert data.found is not None
    return data.found.squeeze(-1)


def _sensor_indexes_for_frames(sensor: ContactSensor, frames: list[str]) -> list[int]:
    """Map clip frame names (e.g. hand_frames / feet_frames columns) to the
    sensor's primary indexes, so [B, N] sensor data can be gathered into the
    clip's column order regardless of how the sensor pattern expanded.

    Exact name match first; falls back to substring containment so geom-mode
    primaries (e.g. `left_wrist_yaw_link_collision`) still resolve. Assumes
    num_slots=1 (N == P), the mjlab default."""
    names = sensor.primary_names
    indexes = []
    for frame in frames:
        if frame in names:
            indexes.append(names.index(frame))
            continue
        hits = [i for i, n in enumerate(names) if frame in n or n in frame]
        if len(hits) != 1:
            # last resort: match by side ("left_wrist_yaw_link" ->
            # "left_wrist_collision" — the wall-contact geom lives on
            # wrist_pitch_link, so names share only the side prefix)
            side = frame.split("_")[0]
            hits = [i for i, n in enumerate(names) if n.split("_")[0] == side]
        if len(hits) != 1:
            raise ValueError(
                f"sensor '{sensor.cfg.name}': clip frame '{frame}' matches "
                f"{len(hits)} primaries in {names}; need exactly 1"
            )
        indexes.append(hits[0])
    return indexes


def _normal_force(sensor: ContactSensor, indexes: list[int]) -> torch.Tensor:
    """Measured normal force [B, len(indexes)], >= 0 in compression.

    MuJoCo's contact sensor reports force in the CONTACT frame whose x-axis is
    the normal, so the normal component is index 0 (not 2). Requires the sensor
    to keep contact-frame output: reduce != "netforce" and global_frame=False."""
    if sensor.cfg.reduce == "netforce" or sensor.cfg.global_frame:
        raise ValueError(
            f"sensor '{sensor.cfg.name}' outputs global-frame force; force-tracking "
            "rewards need the contact-frame normal (component 0) — use "
            'reduce="maxforce" with global_frame=False'
        )
    data = sensor.data
    assert data.force is not None, "sensor must include 'force' in fields"
    return data.force[:, indexes, 0]


def reward_hand_contact_force_exp(
    env: ManagerBasedRlEnv,
    command_name: str,
    sensor_name: str,
    std: float,
) -> torch.Tensor:
    """Track the reference hand normal force, gated on the contact schedule.

    The reference (command.hand_forces) is contact-local z, NEGATIVE = pressing;
    the sensor's normal component is >= 0 in compression, so the target is
    -hand_forces. Frames with hand_contact == 0 contribute no error (force 0
    there means deliberately airborne; unwanted touching is penalized by
    penalty_hand_contact)."""
    command = cast(MPCMotionCommand, env.command_manager.get_term(command_name))
    sensor: ContactSensor = env.scene[sensor_name]
    indexes = _sensor_indexes_for_frames(sensor, command.motion.hand_frames)
    measured = _normal_force(sensor, indexes)  # [B, H]
    error = command.hand_contact * torch.square(measured - (-command.hand_forces))
    return torch.exp(-error.sum(-1) / std**2)


def penalty_hand_contact_force_l2(
    env: ManagerBasedRlEnv,
    command_name: str,
    sensor_name: str,
    huber_delta: float = 20.0,
    err_max: float = 200.0,
) -> torch.Tensor:
    """Huber normal-force deviation penalty (quadratic below huber_delta,
    LINEAR above, error clamped at err_max), active in every schedule phase:
    pressing -> |measured - ref|, airborne -> |measured| (ref is 0 by design).

    Why not plain squared error: a stiff wall contact under exploration noise
    spikes to ~1 kN on most steps somewhere in the batch (measured: per-step
    max over 64 random-action envs has p50 = 1040 N). Squared, one spike is a
    -27/step reward outlier (normal step total ~ +0.11 after dt) -> value
    targets +-50 -> the value-loss explosion of run 12-16-06. Huber + clamp
    bounds the worst step at 0.5*delta^2 + delta*(err_max - delta) (=3800
    here, i.e. -0.76/step at weight 2e-4) while keeping a monotone "press
    softer / stop dragging" gradient everywhere the exp force reward is flat."""
    command = cast(MPCMotionCommand, env.command_manager.get_term(command_name))
    sensor: ContactSensor = env.scene[sensor_name]
    indexes = _sensor_indexes_for_frames(sensor, command.motion.hand_frames)
    measured = _normal_force(sensor, indexes)  # [B, H], pressing > 0
    ref = -command.hand_forces  # [B, H], pressing > 0, 0 when airborne
    err = (measured - ref).abs().clamp(max=err_max)
    quad = err.clamp(max=huber_delta)
    return (0.5 * quad**2 + huber_delta * (err - quad)).sum(-1)


def reward_hand_contact(
    env: ManagerBasedRlEnv,
    command_name: str,
    sensor_name: str,
) -> torch.Tensor:
    """Fraction of hands whose measured contact state matches the schedule."""
    command = cast(MPCMotionCommand, env.command_manager.get_term(command_name))
    sensor: ContactSensor = env.scene[sensor_name]
    indexes = _sensor_indexes_for_frames(sensor, command.motion.hand_frames)
    data = sensor.data
    assert data.found is not None, "sensor must include 'found' in fields"
    measured = data.found[:, indexes] > 0  # [B, H]
    return (measured == (command.hand_contact > 0)).float().mean(-1)


def penalty_hand_contact(
    env: ManagerBasedRlEnv,
    command_name: str,
    sensor_name: str,
) -> torch.Tensor:
    """Count hands in contact while the schedule says airborne."""
    command = cast(MPCMotionCommand, env.command_manager.get_term(command_name))
    sensor: ContactSensor = env.scene[sensor_name]
    indexes = _sensor_indexes_for_frames(sensor, command.motion.hand_frames)
    data = sensor.data
    assert data.found is not None, "sensor must include 'found' in fields"
    measured = (data.found[:, indexes] > 0).float()  # [B, H]
    return ((1.0 - command.hand_contact) * measured).sum(-1)


def reward_feet_grf_exp(
    env: ManagerBasedRlEnv,
    command_name: str,
    sensor_name: str,
    std: float,
) -> torch.Tensor:
    """Track the reference vertical foot GRF, gated on the stance schedule.

    Requires a reduce="netforce" sensor: each foot has 7 contact capsules, so
    the total GRF is the NET wrench summed over them, which netforce reports
    in the GLOBAL frame. Sign (measured): supporting contact gives NEGATIVE z
    (verified: standing ~-200 N per foot), so the upward GRF is -force_z —
    positive = supporting, same sign as the feet_grf reference. Swing frames
    (feet_contact == 0) contribute no error."""
    command = cast(MPCMotionCommand, env.command_manager.get_term(command_name))
    sensor: ContactSensor = env.scene[sensor_name]
    if sensor.cfg.reduce != "netforce":
        raise ValueError(
            f"sensor '{sensor.cfg.name}': feet GRF tracking needs the per-foot "
            'NET force in the global frame — use reduce="netforce"'
        )
    indexes = _sensor_indexes_for_frames(sensor, command.motion.feet_frames)
    data = sensor.data
    assert data.force is not None, "sensor must include 'force' in fields"
    measured = -data.force[:, indexes, 2]  # [B, 2] upward GRF (see docstring)
    error = command.feet_contact * torch.square(measured - command.feet_grf)
    return torch.exp(-error.sum(-1) / std**2)


def reward_feet_contact(
    env: ManagerBasedRlEnv,
    command_name: str,
    sensor_name: str,
) -> torch.Tensor:
    """Fraction of feet whose measured contact state matches the stance schedule."""
    command = cast(MPCMotionCommand, env.command_manager.get_term(command_name))
    sensor: ContactSensor = env.scene[sensor_name]
    indexes = _sensor_indexes_for_frames(sensor, command.motion.feet_frames)
    data = sensor.data
    assert data.found is not None, "sensor must include 'found' in fields"
    measured = data.found[:, indexes] > 0  # [B, 2]
    return (measured == (command.feet_contact > 0)).float().mean(-1)


def motion_feet_pos_error_exp(
    env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
    """Anchor-relative FEET position tracking (the two ankle_roll bodies only).

    Anti-skating: in the 14-body mean (motion_body_pos, std 0.3) the feet are
    2/14 — checkpoint 12-43-46 tracked bodies at 3.7 cm mean while the feet
    sat at 7.1 cm with half the reference swing height, and skating won. This
    term re-applies the same anchor-relative error at foot scale."""
    command = cast(MPCMotionCommand, env.command_manager.get_term(command_name))
    cols = [command.cfg.body_names.index(f) for f in command.motion.feet_frames]
    error = torch.sum(
        torch.square(
            command.body_pos_relative_w[:, cols] - command.robot_body_pos_w[:, cols]
        ),
        dim=-1,
    )
    return torch.exp(-error.mean(-1) / std**2)


def penalty_feet_slide(
    env: ManagerBasedRlEnv,
    command_name: str,
    sensor_name: str,
) -> torch.Tensor:
    """Foot xy velocity^2 while measured in contact (anti-skating).

    Adapted from mjlab's velocity-task feet_slip, minus its gate on the
    velocity command's magnitude (a tracking task has no such command, and a
    planted foot should never slide here — standing clips included), and
    reading foot velocity from the tracked ankle_roll bodies instead of foot
    sites."""
    command = cast(MPCMotionCommand, env.command_manager.get_term(command_name))
    sensor: ContactSensor = env.scene[sensor_name]
    indexes = _sensor_indexes_for_frames(sensor, command.motion.feet_frames)
    data = sensor.data
    assert data.found is not None, "sensor must include 'found' in fields"
    in_contact = (data.found[:, indexes] > 0).float()  # [B, 2]
    cols = [command.cfg.body_names.index(f) for f in command.motion.feet_frames]
    vel_xy_sq = torch.sum(
        torch.square(command.robot_body_lin_vel_w[:, cols, :2]), dim=-1
    )  # [B, 2]
    return torch.sum(vel_xy_sq * in_contact, dim=-1)


# --- Opt2Skill-style joint-space tracking (arXiv:2409.20514, Table I) ---------


def motion_joint_pos_error_exp(
    env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
    """Track reference joint positions: exp(-sum_sq_err / std^2).

    Opt2Skill's exp(-5*||q_ref - q||^2) corresponds to std = 1/sqrt(5) ~ 0.45."""
    command = cast(MPCMotionCommand, env.command_manager.get_term(command_name))
    error = torch.sum(torch.square(command.joint_pos - command.robot_joint_pos), dim=-1)
    return torch.exp(-error / std**2)


def motion_global_body_position_error_exp(
    env: ManagerBasedRlEnv,
    command_name: str,
    std: float,
    body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
    """WORLD-frame body position tracking, restricted to body_names.

    For the wall EEs this is deliberately world-anchored (unlike the dropped
    mimic/anchor terms): within an episode the wall is world-fixed, so pulling
    the hand to the clip's world EE point IS reaching the commanded
    target-in-body-frame (motion_ee_pose_b) — same error, different frame.
    It is the only reward tying the hand to the actual wall surface; base
    drift itself stays unpunished."""
    command = cast(MPCMotionCommand, env.command_manager.get_term(command_name))
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(
            command.body_pos_w[:, body_indexes]
            - command.robot_body_pos_w[:, body_indexes]
        ),
        dim=-1,
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_orientation_error_exp(
    env: ManagerBasedRlEnv,
    command_name: str,
    std: float,
    body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
    """WORLD-frame body orientation tracking, restricted to body_names (for the
    wall EEs: keeps the sensor face pointed into the wall)."""
    command = cast(MPCMotionCommand, env.command_manager.get_term(command_name))
    body_indexes = _get_body_indexes(command, body_names)
    error = (
        quat_error_magnitude(
            command.body_quat_w[:, body_indexes],
            command.robot_body_quat_w[:, body_indexes],
        )
        ** 2
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_joint_torque_error_exp(
    env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
    """Track the TO's reference joint torques: exp(-sum_sq_err / std^2).

    Opt2Skill's exp(-0.01*||u_ref - u||^2) corresponds to std = 10. Measured
    torque = qfrc_actuator (joint-space actuator force, canonical joint order —
    NOT actuator_force, which is in ctrl order). Opt2Skill's ablation: torque
    reference teaches WHEN/HOW to apply contact force; together with the force
    reward it gives the best contact-force tracking (Pos+F+T)."""
    command = cast(MPCMotionCommand, env.command_manager.get_term(command_name))
    measured = command.robot.data.qfrc_actuator  # [N, J]
    error = torch.sum(torch.square(command.joint_torque - measured), dim=-1)
    return torch.exp(-error / std**2)
