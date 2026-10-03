from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.utils.lab_api.math import quat_apply_inverse, quat_error_magnitude

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

    from .commands import UniformEePoseCommand


def _pose_errors(
    env: "ManagerBasedRlEnv", command_names: tuple[str, str]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-hand position error [m] and geodesic angle [rad]. Each (B, n_hands)."""
    pos_err, ang_err = [], []
    for name in command_names:
        term: "UniformEePoseCommand" = env.command_manager.get_term(name)
        tgt_pos_w, tgt_quat_w = term.target_pose_w()
        ee_pos_w, ee_quat_w = term.measured_ee_pose_w()
        pos_err.append((tgt_pos_w - ee_pos_w).norm(dim=-1))
        ang_err.append(quat_error_magnitude(tgt_quat_w, ee_quat_w))
    return torch.stack(pos_err, dim=-1), torch.stack(ang_err, dim=-1)


def ee_pos_l2(env: "ManagerBasedRlEnv", command_names: tuple[str, str]) -> torch.Tensor:
    """Sum over hands of squared position error [m^2]. Negative weight."""
    pos_err, _ = _pose_errors(env, command_names)
    return pos_err.pow(2).sum(-1)


def ee_ori_l2(env: "ManagerBasedRlEnv", command_names: tuple[str, str]) -> torch.Tensor:
    """Sum over hands of squared geodesic angle [rad^2]. Negative weight."""
    _, ang_err = _pose_errors(env, command_names)
    return ang_err.pow(2).sum(-1)


def ee_pos_tracking(
    env: "ManagerBasedRlEnv", std: float, command_names: tuple[str, str]
) -> torch.Tensor:
    """exp(-(|p_tgt - p_ee| / std)^2), SUM over hands. In [0, 2]."""
    pos_err, _ = _pose_errors(env, command_names)
    return torch.exp(-((pos_err / std) ** 2)).sum(-1)


def ee_ori_tracking(
    env: "ManagerBasedRlEnv", std: float, command_names: tuple[str, str]
) -> torch.Tensor:
    """exp(-(geodesic angle / std)^2), SUM over hands. In [0, 2]."""
    _, ang_err = _pose_errors(env, command_names)
    return torch.exp(-((ang_err / std) ** 2)).sum(-1)


def arm_mimic_l2(
    env: "ManagerBasedRlEnv", command_names: tuple[str, str]
) -> torch.Tensor:
    total = None
    for name in command_names:
        term = env.command_manager.get_term(name)
        q = term.robot.data.joint_pos[:, term.arm_joint_ids]
        err = (q - term.ik_joint_ref).pow(2).sum(-1)
        total = err if total is None else total + err
    return total


def arm_mimic_exp(
    env: "ManagerBasedRlEnv", std: float, command_names: tuple[str, str]
) -> torch.Tensor:
    out = None
    for name in command_names:
        term = env.command_manager.get_term(name)
        q = term.robot.data.joint_pos[:, term.arm_joint_ids]
        r = torch.exp(-(q - term.ik_joint_ref).pow(2).sum(-1) / std**2)
        out = r if out is None else out + r
    return out


def joint_deviation_l1(env: "ManagerBasedRlEnv", asset_cfg) -> torch.Tensor:
    """Sum of |q - q_default| over the selected joints [rad]. Negative weight.
    (IsaacLab's joint_deviation_l1; used to lock the waist so the pelvis-frame
    sample boxes can't be tilted by bending the torso — the run-4 lesson.)"""
    robot = env.scene[asset_cfg.name]
    q = robot.data.joint_pos[:, asset_cfg.joint_ids]
    q0 = robot.data.default_joint_pos[:, asset_cfg.joint_ids]
    return (q - q0).abs().sum(-1)


def body_flat_orientation_l2(env: "ManagerBasedRlEnv", asset_cfg) -> torch.Tensor:
    """Squared xy gravity projection in a BODY's own frame (torso) — the
    steady-tray recipe. mjlab's flat_orientation_l2 reads the ROOT (pelvis);
    the torso needs its own term (the exp `upright` kernel is gradient-flat
    near vertical). Belt-and-braces: the 2026-07-28 lean-back root cause was
    the Se2 attractor as training command (v5.2), not torso statics — but an
    unpriced torso stays a free null space. Negative weight."""
    robot = env.scene[asset_cfg.name]
    quat = robot.data.body_link_quat_w[:, asset_cfg.body_ids[0]]
    g_b = quat_apply_inverse(quat, torch.tensor([0.0, 0.0, -1.0], device=quat.device).expand(quat.shape[0], 3))
    return g_b[:, :2].pow(2).sum(-1)


def base_height_l2(env: "ManagerBasedRlEnv", target_height: float) -> torch.Tensor:
    """Squared pelvis-height error [m^2] — anti-squat. Negative weight.
    (Nothing else in the stack pins standing height: the MPC CoM reference
    re-seeds from the measured state each solve, and the posture reward is
    outbid — runs 2/3 settled into a half-squat.)"""
    robot = env.scene["robot"]
    return (robot.data.root_link_pos_w[:, 2] - target_height).pow(2)
