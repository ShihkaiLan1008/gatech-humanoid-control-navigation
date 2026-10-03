"""Observation terms the built-ins don't cover (REFINE-DP Table I extras).

The command itself (swing one-hot | countdown | target pose) comes through the
stock `generated_commands`; here live only the critic's privileged reference /
foot channels, expressed in the BASE frame so they are invariant to the
per-episode world transform.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mjlab.managers import SceneEntityCfg
import torch

from mjlab.utils.lab_api.math import (
    matrix_from_quat,
    quat_apply_inverse,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    sample_uniform,
)
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

# Runtime import: used in isinstance() below, not just in annotations.
from .commands.locomotion_mpc import LocoMPCCommand

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

    from rl.decoupled.mdp.commands import (
        EeContactTargetCommand,
        EeTargetCommand,
        FootPlacementCommand,
        EEQposeCommand,
    )


def _cmd(env: ManagerBasedRlEnv, name: str) -> FootPlacementCommand | EeTargetCommand:
    return env.command_manager.get_term(name)  # type: ignore[return-value]


def _to_base(robot, p_w: torch.Tensor) -> torch.Tensor:
    """World points [B, 2, 3] -> base frame, flattened [B, 6]."""
    root_pos = robot.data.root_link_pos_w[:, None, :]
    root_quat = robot.data.root_link_quat_w[:, None, :].expand(-1, p_w.shape[1], -1)
    return quat_apply_inverse(root_quat, p_w - root_pos).flatten(1)


def ref_feet_pos_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Reference feet positions in the base frame (critic-only)."""
    cmd = _cmd(env, command_name)
    return _to_base(cmd.robot, cmd.ref_feet_pos_w)


def feet_pos_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Measured feet positions in the base frame (critic-only)."""
    cmd = _cmd(env, command_name)
    return _to_base(cmd.robot, cmd.robot_feet_pos_w)

def ee_pose_b(env: ManagerBasedRlEnv, command_names: tuple[str, str],
              pos_noise: float = 0.0, rot_noise: float = 0.0) -> torch.Tensor:
    """Measured end-effector POSES in the command anchor frame: per hand
    [pos (3), quat wxyz (4)] — same frame and encoding as the command channel
    (REFINE-DP's "current hand poses"). Quat w-sign normalized so the pose→
    vector map is deterministic. [B, 7·n_hands].

    FK-error DR as in `ee_pose_err`: the measured pose is perturbed
    on-manifold (never additive on quat components) — NOT gated by
    ``enable_corruption``; the play assembly must zero the params.
    """
    outs = []
    for name in command_names:
        term: EeTargetCommand = _cmd(env, name)
        robot = term.robot
        p_w, q_w = term.measured_ee_pose_w()
        if pos_noise > 0.0:
            p_w = p_w + sample_uniform(
                -pos_noise, pos_noise, p_w.shape, p_w.device)
        if rot_noise > 0.0:
            rpy = sample_uniform(
                -rot_noise, rot_noise, (q_w.shape[0], 3), q_w.device)
            q_w = quat_mul(q_w, quat_from_euler_xyz(
                rpy[:, 0], rpy[:, 1], rpy[:, 2]))
        aq = term.anchor_quat()
        p_b = quat_apply_inverse(aq, p_w - robot.data.root_link_pos_w)
        q_b = quat_mul(quat_inv(aq), q_w)
        q_b = torch.where(q_b[:, :1] < 0.0, -q_b, q_b)
        outs.append(torch.cat([p_b, q_b], dim=-1))
    return torch.cat(outs, dim=-1)


def ee_pose_err(env: ManagerBasedRlEnv, command_names: tuple[str, str],
                pos_noise: float = 0.0, rot_noise: float = 0.0
                ) -> torch.Tensor:
    """Per-hand EE tracking error, HERO's residual-obs recipe (their Eq. 2:
    ΔE = current ⊖ target): position gap expressed in the command ANCHOR frame
    (pelvis — same frame as the command itself; HERO expresses it in the
    relative-transform frame, information-equivalent) + the current→target
    relative rotation in the continuous 6D representation (HERO's stated
    encoding; first two columns of the rotation matrix — identity at goal).
    [B, 9·n_hands]. Deployable: measured pose = encoder FK, target = command.

    FK-error DR (`pos_noise` [m], `rot_noise` [rad]): the MEASURED pose is
    perturbed before differencing — position by U(±pos_noise), orientation by
    an EE-local random rotation from U(±rot_noise) rpy (additive noise on the
    quat/6D components would leave the manifold, and perturbing the measured
    pose keeps the position/rotation channels coherently corrupted, like a
    real FK bias). Both papers train these obs CLEAN and eat the real FK error
    at deploy (HERO: learned residual FK); the noise here substitutes for the
    residual model we don't have. NB in-term noise is NOT gated by the group's
    ``enable_corruption`` — the play assembly must zero these params itself.
    """
    outs = []
    for name in command_names:
        term: EeTargetCommand = _cmd(env, name)
        tgt_p, tgt_q = term.target_pose_w()
        ee_p, ee_q = term.measured_ee_pose_w()
        if pos_noise > 0.0:
            ee_p = ee_p + sample_uniform(
                -pos_noise, pos_noise, ee_p.shape, ee_p.device)
        if rot_noise > 0.0:
            rpy = sample_uniform(
                -rot_noise, rot_noise, (ee_q.shape[0], 3), ee_q.device)
            ee_q = quat_mul(ee_q, quat_from_euler_xyz(
                rpy[:, 0], rpy[:, 1], rpy[:, 2]))
        dp = quat_apply_inverse(term.anchor_quat(), tgt_p - ee_p)
        rel = matrix_from_quat(quat_mul(quat_inv(ee_q), tgt_q))
        outs.append(torch.cat([dp, rel[:, :, 0], rel[:, :, 1]], dim=-1))
    return torch.cat(outs, dim=-1)

def ee_ref_joint_pos(env: ManagerBasedRlEnv, command_names: str) -> torch.Tensor:
    command_term: EEQposeCommand = env.command_manager.get_term(command_names)
    return command_term.ref_joint_pos

def zero_obs(env: ManagerBasedRlEnv, dim: int) -> torch.Tensor:
    """Frozen zeros — reserves obs slots a later training stage fills in place
    (the whole-body stack's `hand_stage2_zeros` recipe: identical dims across
    stages ⇒ whole-checkpoint resume with no input-layer surgery)."""
    return torch.zeros(env.num_envs, dim, device=env.device)

def _body_state_b(robot, p_w: torch.Tensor, q_w: torch.Tensor, v_w: torch.Tensor,
                  w_w: torch.Tensor) -> torch.Tensor:
    """Per-link [pos (3), quat wxyz (4), lin vel (3), ang vel (3)] in the base
    frame, flattened [B, 13·Nb]. Quat w-sign normalized."""
    root_pos = robot.data.root_link_pos_w[:, None, :]
    root_quat = robot.data.root_link_quat_w[:, None, :].expand(-1, p_w.shape[1], -1)
    p_b = quat_apply_inverse(root_quat, p_w - root_pos)
    q_b = quat_mul(quat_inv(root_quat), q_w)
    q_b = torch.where(q_b[..., :1] < 0.0, -q_b, q_b)
    v_b = quat_apply_inverse(root_quat, v_w)
    w_b = quat_apply_inverse(root_quat, w_w)
    return torch.cat([p_b, q_b, v_b, w_b], dim=-1).flatten(1)


def ref_body_state_b(env: ManagerBasedRlEnv, command_name: str,
                     body_names: tuple[str, ...]) -> torch.Tensor:
    """Reference link states (pos, quat, lin vel, ang vel) of `body_names` in
    the base frame, [B, 13·Nb] (critic-only)."""
    cmd: FootPlacementCommand = _cmd(env, command_name)
    c = cmd.body_cols(body_names)
    return _body_state_b(cmd.robot, cmd.ref_body_pos_w[:, c], cmd.ref_body_quat_w[:, c],
                         cmd.ref_body_lin_vel_w[:, c], cmd.ref_body_ang_vel_w[:, c])


def body_state_b(env: ManagerBasedRlEnv, command_name: str,
                 body_names: tuple[str, ...]) -> torch.Tensor:
    """Measured link states (pos, quat, lin vel, ang vel) of `body_names` in
    the base frame, [B, 13·Nb] (critic-only)."""
    cmd: FootPlacementCommand = _cmd(env, command_name)
    c = cmd.body_cols(body_names)
    return _body_state_b(cmd.robot, cmd.robot_body_pos_w[:, c], cmd.robot_body_quat_w[:, c],
                         cmd.robot_body_lin_vel_w[:, c], cmd.robot_body_ang_vel_w[:, c])


def clf_state(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """[V, Vdot] of the command term's CLF, [B, 2] (critic-only; zeros when
    the term carries no CLF)."""
    cmd = _cmd(env, command_name)
    return torch.stack([cmd.v, cmd.vdot], dim=-1)


def ref_joint_torque(env: ManagerBasedRlEnv, command_name: str,
                     scale: float = 0.05) -> torch.Tensor:
    """Clip actuator-torque reference, model joint order, times `scale`.
    Shape (B, 29) (critic-only; zeros on a kinematic dataset)."""
    cmd = _cmd(env, command_name)
    return cmd.ref_joint_torque * scale


def ref_feet_grf(env: ManagerBasedRlEnv, command_name: str,
                 scale: float = 0.01) -> torch.Tensor:
    """Clip ground-on-foot force reference, world frame, [L, R] flattened,
    times `scale`. Shape (B, 6) (critic-only)."""
    cmd = _cmd(env, command_name)
    return cmd.ref_feet_grf_w.flatten(1) * scale


def ref_joint_pos(env: ManagerBasedRlEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reference joint positions over the selected joints (critic-only)."""
    cmd: FootPlacementCommand = _cmd(env, command_name)
    return cmd.ref_joint_pos[:, asset_cfg.joint_ids]

def ref_joint_torques(env: ManagerBasedRlEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reference joint torques over the selected joints (critic-only)."""
    cmd: FootPlacementCommand = _cmd(env, command_name)
    return cmd.ref_joint_torque[:, asset_cfg.joint_ids]

def ref_joint_vel(env: ManagerBasedRlEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reference joint velocities over the selected joints (critic-only)."""
    cmd: FootPlacementCommand = _cmd(env, command_name)
    return cmd.ref_joint_vel[:, asset_cfg.joint_ids]


def foot_height(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.site_pos_w[:, asset_cfg.site_ids, 2]  # (num_envs, num_sites)


# foot_air_time / foot_contact / foot_contact_forces: use the mjlab builtins
# (`mjlab.tasks.velocity.mdp`) — our copies were byte-identical.


def phase(env: ManagerBasedRlEnv, period: float, command_name: str) -> torch.Tensor:
    global_phase = (env.episode_length_buf * env.step_dt) % period / period
    phase = torch.zeros(env.num_envs, 2, device=env.device)
    phase[:, 0] = torch.sin(global_phase * torch.pi * 2.0)
    phase[:, 1] = torch.cos(global_phase * torch.pi * 2.0)
    stand_mask = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) < 0.1
    phase = torch.where(stand_mask.unsqueeze(1), torch.zeros_like(phase), phase)
    return phase

def mpc_com_ref(
    env: "ManagerBasedRlEnv",
    command_name: str = "loco_mpc",
) -> torch.Tensor:
    """Expose the MPC CoM-position target to the policy."""
    term = env.command_manager.get_term(command_name)
    if term is None or not isinstance(term, LocoMPCCommand):
        return torch.zeros(env.num_envs, 3, device=env.device)
    c_cur = term._robot.data.root_link_pos_w
    return term._com_mpc_target - c_cur


def mpc_ang_mom_ref(
    env: "ManagerBasedRlEnv",
    command_name: str = "loco_mpc",
) -> torch.Tensor:
    """Expose the MPC angular-momentum target ``k_t^mpc`` to the policy."""
    term = env.command_manager.get_term(command_name)
    if term is None or not isinstance(term, LocoMPCCommand):
        return torch.zeros(env.num_envs, 3, device=env.device)
    return term._k_mpc_target


def mpc_com_vel_ref(
    env: "ManagerBasedRlEnv",
    command_name: str = "loco_mpc",
) -> torch.Tensor:
    """MPC CoM-velocity target for now minus the measured pelvis velocity
    (the plan's l/m), world frame. Shape (B, 3), m/s."""
    term = env.command_manager.get_term(command_name)
    if term is None or not isinstance(term, LocoMPCCommand):
        return torch.zeros(env.num_envs, 3, device=env.device)
    v_cur = term._robot.data.root_link_lin_vel_w
    return term._com_vel_mpc_target - v_cur


# ── hand-force observations ─────────────────────────────────────────────────

_FORCE_SCALE = 50.0
"""N, obs normalization = f_hand_max (the QP's hand-force cap): commands
(≤ 25 N) land ≤ 0.5, measured spikes up to the cap land ≤ 1."""


def hand_contact_force_command(
    env: ManagerBasedRlEnv, command_names: tuple[str, str]
) -> torch.Tensor:
    """(B, 8): per hand [contact_mask(1), force_cmd_b(3)/50], L then R.
    Command channels — actor AND critic."""
    parts = []
    for name in command_names:
        term: EeContactTargetCommand = env.command_manager.get_term(name)
        parts.append(term.contact_mask.unsqueeze(-1))
        parts.append(term.force_cmd_b / _FORCE_SCALE)
    return torch.cat(parts, dim=-1)


def hand_ext_force_meas(
    env: ManagerBasedRlEnv, command_names: tuple[str, str]
) -> torch.Tensor:
    """(B, 6): per hand the measured EXTERNAL force on the hand (= −force on
    surface), /50. CRITIC ONLY — never wire into an actor group."""
    parts = []
    for name in command_names:
        term: EeContactTargetCommand = env.command_manager.get_term(name)
        parts.append(-term.measured_force_w() / _FORCE_SCALE)
    return torch.cat(parts, dim=-1)