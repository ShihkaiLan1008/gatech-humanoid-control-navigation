"""Reward terms — REFINE-DP Table II, adapted to the step-clip references.

Tracking incomes are exp kernels against the command term's references; the
kernels use the SUM of squared errors over the tracked dims (their ‖·‖²
formulation) — remember the repo's sum-kernel reading trap when eyeballing raw
values. Regularizers that mjlab ships (action rate, joint acc, joint limits)
are declared directly from built-ins in reward_cfgs; only what mjlab lacks
lives here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mjlab.sensor import ContactSensor
import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import (
    matrix_from_quat,
    quat_apply,
    quat_apply_inverse,
    quat_error_magnitude,
)
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.sensor import BuiltinSensor
from mjlab.utils.lab_api.string import (
  resolve_matching_names_values,
)

if TYPE_CHECKING:
    from mjlab.entity import Entity
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.managers import CommandTerm
    from rl.decoupled.mdp import (
        EeContactTargetCommand,
        EeTargetCommand,
        FootPlacementCommand,
        EEQposeCommand,
    )

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")

def _cmd(env: ManagerBasedRlEnv, name: str) -> CommandTerm:
    return env.command_manager.get_term(name)  # type: ignore[return-value]


def _foot_column(env: ManagerBasedRlEnv, cmd, role: str) -> torch.Tensor:
    """Per-env column of the commanded swing / stance foot ([B] long).

    THE STANCE FOOT'S TRACKING SCORE IS NEARLY BEHAVIOUR-INDEPENDENT: chaining
    re-anchors each clip onto the robot's ACTUAL stance foot, so that foot's
    reference sits under it by construction (measured kernel 0.957 standing vs
    0.968 walking). Averaging the two feet therefore spends most of the weight
    on a constant — split the roles and put the weight on the swing foot.
    """
    c = cmd.command                              # [B, 7] onehot | countdown | …
    swing = (c[:, 1] > c[:, 0]).long()
    return swing if role == "swing" else 1 - swing


def _foot_contact(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    """Ground-contact flags of the two feet from `sensor_name`, columns in
    model-body order (0 = left, 1 = right). Shape (num_envs, 2), bool."""
    sensor: ContactSensor = env.scene[sensor_name]
    return (sensor.data.found.view(env.num_envs, -1) > 0)[:, :2]


def _foot_kernel_gate(env: ManagerBasedRlEnv, cmd, k: torch.Tensor, role: str,
                      hold_neutral: bool, airborne_only: bool,
                      sensor_name: str | None,
                      airborne_release: float | None = None) -> torch.Tensor:
    """`role` column of a per-foot kernel `k` [B, 2] ("both" = mean; "arc" =
    swing foot while the swing window is open; "pin" = mean over the feet
    whose reference is stationary — the two are complementary), with the
    optional gates: `hold_neutral` pays 0 while the command term is standing
    (`cmd.standing`); `airborne_only` (swing role only) pays 0 unless the
    swing window is open (`cmd.swing_active`) AND the swing foot is out of
    ground contact per `sensor_name` — a foot hovering during the scheduled
    double support earns nothing; `airborne_release` waives the contact
    condition once `cmd.swing_frac` >= that value (the arc's final descent,
    where touching down is the tracked behaviour). Shape (num_envs,)."""
    if role == "both":
        assert not airborne_only, "airborne_only needs role='swing'"
        out = k.mean(dim=-1)
    elif role in ("arc", "pin"):
        # complementary clock partition: "arc" = the swing foot while its
        # reference moves (swing window open); "pin" = every foot whose
        # reference is stationary (stance foot, both feet in DSP / stands)
        assert not airborne_only, "arc/pin are clock-gated, not contact-gated"
        moving = torch.zeros_like(k, dtype=torch.bool)
        idx = torch.arange(env.num_envs, device=env.device)
        moving[idx, _foot_column(env, cmd, "swing")] = cmd.swing_active & ~cmd.standing
        if role == "arc":
            out = (k * moving.float()).sum(dim=-1)
        else:
            pinned = (~moving).float()
            out = (k * pinned).sum(dim=-1) / pinned.sum(dim=-1).clamp(min=1.0)
    else:
        idx = torch.arange(env.num_envs, device=env.device)
        col = _foot_column(env, cmd, role)
        out = k[idx, col]
        if airborne_only:
            assert role == "swing" and sensor_name is not None
            airborne = ~_foot_contact(env, sensor_name)[idx, col]
            if airborne_release is not None:
                airborne = airborne | (cmd.swing_frac >= airborne_release)
            out = out * (airborne & cmd.swing_active).float()
    if hold_neutral:
        out = out * (~cmd.standing).float()
    return out


def track_feet_pos(env: ManagerBasedRlEnv, command_name: str, std: float,
                   role: str = "both", hold_neutral: bool = False,
                   airborne_only: bool = False,
                   sensor_name: str | None = None,
                   airborne_release: float | None = None) -> torch.Tensor:
    """exp(-‖p - p_ref‖²/σ²) on the commanded swing foot, the stance foot, or
    the mean of both (see `_foot_column` for why the split matters), gated as
    in `_foot_kernel_gate`. Shape (num_envs,), in [0, 1]."""
    cmd: FootPlacementCommand = _cmd(env, command_name)
    err2 = torch.sum(
        torch.square(cmd.ref_feet_pos_w - cmd.robot_feet_pos_w), dim=-1)
    k = torch.exp(-err2 / std**2)
    return _foot_kernel_gate(env, cmd, k, role, hold_neutral, airborne_only,
                             sensor_name, airborne_release)


def track_feet_yaw(env: ManagerBasedRlEnv, command_name: str, std: float,
                   role: str = "both", hold_neutral: bool = False,
                   airborne_only: bool = False,
                   sensor_name: str | None = None,
                   airborne_release: float | None = None) -> torch.Tensor:
    """exp(-Δyaw²/σ²) of the foot SITE yaw vs its reference, per role and
    with the same gates as `track_feet_pos`. Shape (num_envs,)."""
    cmd: FootPlacementCommand = _cmd(env, command_name)
    dyaw = cmd.ref_feet_yaw - cmd.robot_feet_yaw
    dyaw = torch.atan2(torch.sin(dyaw), torch.cos(dyaw))
    k = torch.exp(-torch.square(dyaw) / std**2)
    return _foot_kernel_gate(env, cmd, k, role, hold_neutral, airborne_only,
                             sensor_name, airborne_release)


def track_base_lin_vel(env: ManagerBasedRlEnv, command_name: str, std: float
                       ) -> torch.Tensor:
    """Body-frame base linear velocity vs the clip's realized base twist."""
    cmd: FootPlacementCommand = _cmd(env, command_name)
    robot = cmd.robot
    lin_b = quat_apply_inverse(
        robot.data.root_link_quat_w, robot.data.root_link_lin_vel_w)
    err2 = torch.sum(torch.square(cmd.ref_root_lin_vel_b - lin_b), dim=-1)
    return torch.exp(-err2 / std**2)

def track_base_ang_vel(env: ManagerBasedRlEnv, command_name: str, std: float
                       ) -> torch.Tensor:
    """Body-frame base angular velocity vs the clip's realized base twist."""
    cmd: FootPlacementCommand = _cmd(env, command_name)
    robot = cmd.robot
    ang_b = quat_apply_inverse(
        robot.data.root_link_quat_w, robot.data.root_link_ang_vel_w)
    err2 = torch.sum(torch.square(cmd.ref_root_ang_vel_b - ang_b), dim=-1)
    return torch.exp(-err2 / std**2)


class _joint_selected:
    """Base for reference-tracking terms over joints selected BY NAME:
    `joint_names` (names/regex) resolve once at init via `find_joints`
    (preserve_order=True). Valid to index the ref arrays directly — the
    command term asserts entity joint order == the clips' PIN order."""

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        robot = env.scene[cfg.params.get("entity_name", "robot")]
        ids, _ = robot.find_joints(
            list(cfg.params["joint_names"]), preserve_order=True)
        self.ids = torch.tensor(ids, dtype=torch.long, device=env.device)


class track_joint_pos(_joint_selected):
    """exp(-Σ‖q - q_ref‖²/σ²) over the selected joints."""

    def __call__(self, env: ManagerBasedRlEnv, command_name: str, std: float,
                 joint_names: tuple[str, ...], entity_name: str = "robot"
                 ) -> torch.Tensor:
        cmd: FootPlacementCommand = _cmd(env, command_name)
        err2 = torch.sum(torch.square(
            (cmd.ref_joint_pos - cmd.robot.data.joint_pos)[:, self.ids]), dim=-1)
        return torch.exp(-err2 / std**2)


class track_joint_vel(_joint_selected):
    """exp(-Σ‖q̇ - q̇_ref‖²/σ²) over the selected joints."""

    def __call__(self, env: ManagerBasedRlEnv, command_name: str, std: float,
                 joint_names: tuple[str, ...], entity_name: str = "robot"
                 ) -> torch.Tensor:
        cmd: FootPlacementCommand = _cmd(env, command_name)
        err2 = torch.sum(torch.square(
            (cmd.ref_joint_vel - cmd.robot.data.joint_vel)[:, self.ids]), dim=-1)
        return torch.exp(-err2 / std**2)


class track_joint_torque(_joint_selected):
    """exp(-Σ‖τ - τ_ref‖²/σ²) over the selected joints [Nm].

    τ_ref = the clip's TO torque channel; measured = qfrc_actuator (joint-space
    actuator force, joint order — actuator_force is ctrl order). Opt2Skill's
    ablation: the torque reference teaches WHEN/HOW to apply contact force.
    Their exp(-0.01‖·‖²) ⇒ σ=10 over 29 joints; ours defaults looser (σ~30
    over 15 joints ≈ 7.7 Nm/joint before saturation) — a fresh policy is far
    outside σ=10 and the kernel would be gradient-dead."""

    def __call__(self, env: ManagerBasedRlEnv, command_name: str, std: float,
                 joint_names: tuple[str, ...], entity_name: str = "robot"
                 ) -> torch.Tensor:
        cmd = _cmd(env, command_name)
        assert cmd.lib.has_dynamics, \
            "torque tracking on a kinematic dataset (refs are zeros)"
        err2 = torch.sum(torch.square(
            (cmd.ref_joint_torque
             - cmd.robot.data.qfrc_actuator)[:, self.ids]), dim=-1)
        return torch.exp(-err2 / std**2)


class track_joint_torque_l2(_joint_selected):
    """Unsaturating companion of `track_joint_torque` (penalties don't
    saturate; exp kernels do). Per-joint error clamped [Nm] — never square an
    unbounded torque: impact frames spike qfrc_actuator to the actuator
    limits (~90-140 Nm)."""

    def __call__(self, env: ManagerBasedRlEnv, command_name: str,
                 joint_names: tuple[str, ...], clamp: float | None = 30.0,
                 entity_name: str = "robot") -> torch.Tensor:
        cmd = _cmd(env, command_name)
        err = (cmd.ref_joint_torque
               - cmd.robot.data.qfrc_actuator)[:, self.ids]
        if clamp is not None:
            err = err.clamp(min=-clamp, max=clamp)
        return torch.sum(torch.square(err), dim=-1)





def _feet_grf_err(env: ManagerBasedRlEnv, command_name: str,
                  sensor_name: str) -> torch.Tensor:
    """Ground-on-foot force error vs the clip reference, world frame.
    Shape (num_envs, 2, 3), N. The feet ContactSensor reports the force the
    FOOT applies to the ground (support fz < 0), hence the negation; columns
    are model-body order [L, R], same as the reference."""
    cmd = _cmd(env, command_name)
    sensor: ContactSensor = env.scene[sensor_name]
    force = sensor.data.force
    assert force is not None
    grf = -force.view(env.num_envs, -1, 3)[:, :2]
    return cmd.ref_feet_grf_w - grf


def track_feet_grf(env: ManagerBasedRlEnv, command_name: str, std: float,
                   sensor_name: str) -> torch.Tensor:
    """exp(-Σ‖f - f_ref‖²/σ²) over both feet, world frame [N]. Shape
    (num_envs,), in [0, 1]. f_ref = the clip's contact-force channel (zero on
    the swing foot); f = the feet ground ContactSensor as ground-on-foot.
    Requires a dynamics dataset (`lib.has_dynamics`)."""
    cmd = _cmd(env, command_name)
    assert cmd.lib.has_dynamics, \
        "GRF tracking on a kinematic dataset (refs are zeros)"
    err = _feet_grf_err(env, command_name, sensor_name)
    return torch.exp(-torch.sum(torch.square(err), dim=(-2, -1)) / std**2)


def track_feet_grf_l1(env: ManagerBasedRlEnv, command_name: str,
                      sensor_name: str, clamp: float = 300.0) -> torch.Tensor:
    """Unsaturating companion of `track_feet_grf`: Σ|err| clamped per axis
    at `clamp` [N]. Shape (num_envs,), N. Linear on purpose: landing spikes
    reach kN."""
    err = _feet_grf_err(env, command_name, sensor_name)
    return torch.sum(torch.abs(err).clamp(max=clamp), dim=(-2, -1))


def track_feet_pos_l1(env: ManagerBasedRlEnv, command_name: str,
                      clamp: float | None = 0.3, role: str = "both") -> torch.Tensor:
    """Per-foot ‖p - p_ref‖ clamped at `clamp`, selected by `role` as in
    `_foot_kernel_gate` ("arc" = the swing foot while its window is open,
    "both" = mean of both feet). Negative weight: the far-field charge that
    stays linear where an exp kernel and a squared error have both gone
    flat. Shape (num_envs,), m."""
    cmd: FootPlacementCommand = _cmd(env, command_name)
    err = (cmd.ref_feet_pos_w - cmd.robot_feet_pos_w).norm(dim=-1)   # [B, 2]
    if clamp is not None:
        err = err.clamp(max=clamp)
    return _foot_kernel_gate(env, cmd, err, role, False, False, None)


def track_feet_yaw_l2(env: ManagerBasedRlEnv, command_name: str,
                      clamp: float | None = 1.0) -> torch.Tensor:
    """Unsaturating companion of `track_feet_yaw`: wrapped foot-yaw error
    squared, clamped per foot, summed over both feet. Shape (num_envs,), rad²."""
    cmd: FootPlacementCommand = _cmd(env, command_name)
    dyaw = cmd.ref_feet_yaw - cmd.robot_feet_yaw
    dyaw = torch.atan2(torch.sin(dyaw), torch.cos(dyaw)).abs()
    if clamp is not None:
        dyaw = dyaw.clamp(max=clamp)
    return torch.sum(torch.square(dyaw), dim=-1)


def body_ang_vel_l2(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """‖ω‖² of the selected body's LINK frame, world axes, ALL three axes
    (yaw included). Shape (num_envs,), (rad/s)²."""
    asset = env.scene[asset_cfg.name]
    w = asset.data.body_link_ang_vel_w[:, asset_cfg.body_ids, :].squeeze(1)
    return torch.sum(torch.square(w), dim=-1)


def track_feet_pos_l2(env: ManagerBasedRlEnv, command_name: str,
                      clamp: float | None = 0.3,
                      hold_neutral: bool = False) -> torch.Tensor:
    """Unsaturating companion of `track_feet_pos`, summed over both feet.
    Shape (num_envs,), m². Clamped per axis so a foot thrown by a solver
    divergence pays a bounded tax instead of an unbounded squared one;
    `hold_neutral` charges 0 while the command term is standing."""
    cmd: FootPlacementCommand = _cmd(env, command_name)
    err = cmd.ref_feet_pos_w - cmd.robot_feet_pos_w          # [B, 2, 3]
    if clamp is not None:
        err = err.clamp(min=-clamp, max=clamp)
    out = torch.sum(torch.square(err), dim=(-2, -1))
    if hold_neutral:
        out = out * (~cmd.standing).float()
    return out


def joint_acc_l2(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg,
                 clamp: float | None = None) -> torch.Tensor:
    """L2 on joint accelerations over the selected joints. Shape (num_envs,),
    (rad/s²)². `clamp` bounds |acc| PER JOINT before squaring; None reproduces
    the mjlab builtin exactly."""
    acc = env.scene[asset_cfg.name].data.joint_acc[:, asset_cfg.joint_ids]
    if clamp is not None:
        acc = acc.clamp(min=-clamp, max=clamp)
    return torch.sum(torch.square(acc), dim=1)


def _measured_com_w(env: ManagerBasedRlEnv, entity_name: str) -> torch.Tensor:
    """Whole-body CoM in world frame: mujoco's subtree_com at the root body
    (= pinocchio centerOfMass, the v2 dataset's com channel convention)."""
    data = env.scene[entity_name].data
    return data.data.subtree_com[:, data.indexing.root_body_id]


def track_com_pos(env: ManagerBasedRlEnv, command_name: str, std: float,
                  entity_name: str = "robot") -> torch.Tensor:
    """exp(-‖com_xy - ref_com_xy‖²/σ²) — HORIZONTAL CoM against the clip's
    re-anchored com channel (planner play: mid-feet synthesis).

    Prices the posture foot tracking leaves free: feet/vel/attitude can all
    score while the CoM parks at the support-polygon edge (the tilted-stand
    failure). Horizontal only — z is priced by base_height. σ must tolerate
    the gait's real lateral transfer (0.33-0.46 of half-width ≈ 3-6 cm toward
    the stance foot, measured in the clips), hence 0.08-0.10, NOT the feet's
    0.05."""
    cmd: FootPlacementCommand = _cmd(env, command_name)
    err = _measured_com_w(env, entity_name)[:, :2] - cmd.ref_com_pos_w[:, :2]
    return torch.exp(-torch.sum(torch.square(err), dim=-1) / std**2)


def track_com_pos_l2(env: ManagerBasedRlEnv, command_name: str,
                     clamp: float | None = 0.3,
                     entity_name: str = "robot") -> torch.Tensor:
    """Unsaturating companion of `track_com_pos` (penalties don't saturate;
    exp kernels do). Clamped so a falling robot pays a bounded tax."""
    cmd: FootPlacementCommand = _cmd(env, command_name)
    err = _measured_com_w(env, entity_name)[:, :2] - cmd.ref_com_pos_w[:, :2]
    if clamp is not None:
        err = err.clamp(min=-clamp, max=clamp)
    return torch.sum(torch.square(err), dim=-1)


def track_root_pos(env: ManagerBasedRlEnv, command_name: str, std: float
                   ) -> torch.Tensor:
    """exp(-‖pelvis_xy - ref_pelvis_xy‖²/σ²) against the clip's own root
    channel. Shape (num_envs,), unitless.

    The pelvis is a floating-base coordinate, NOT a function of the robot's own
    joints: neither a torso lean nor an arm swing moves it, so unlike
    `track_com_pos` this objective cannot be bought with posture. Horizontal
    only; the clip's pelvis and CoM trajectories differ by a near-constant
    offset, so the reference shape is the same one `track_com_pos` used.
    """
    cmd: FootPlacementCommand = _cmd(env, command_name)
    err = cmd.robot.data.root_link_pos_w[:, :2] - cmd.ref_root_pos_w[:, :2]
    return torch.exp(-torch.sum(torch.square(err), dim=-1) / std**2)


def track_root_pos_l2(env: ManagerBasedRlEnv, command_name: str,
                      clamp: float | None = 0.3) -> torch.Tensor:
    """Unsaturating companion of `track_root_pos`. Shape (num_envs,), m².
    Clamped per axis so a falling robot pays a bounded tax."""
    cmd: FootPlacementCommand = _cmd(env, command_name)
    err = cmd.robot.data.root_link_pos_w[:, :2] - cmd.ref_root_pos_w[:, :2]
    if clamp is not None:
        err = err.clamp(min=-clamp, max=clamp)
    return torch.sum(torch.square(err), dim=-1)


def _torso_point_err(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg,
                     offset: float, origin_b: tuple[float, float, float]
                     ) -> torch.Tensor:
    """Displacement of a body-fixed point from its target, PELVIS frame.
    Shape (num_envs, 3), m.

    The point sits `offset` metres along the body's OWN z axis. The waist
    joints all anchor at the torso_link origin, so the origin itself does not
    translate when the waist bends — the offset is what gives the term a lever
    (measured: 49.7 mm per 0.20 rad at offset 0.25). Expressed in the pelvis
    frame, so it prices the waist alone and is blind to where the pelvis is.
    A point on the body z axis is also blind to waist YAW by construction.

    Args:
        offset: distance along the body's own +z, m. The ONLY lever knob.
        origin_b: the body's ORIGIN in the pelvis frame with the waist at
            zero, m — a fixed model geometry, independent of `offset`. The
            target is `origin_b + (0, 0, offset)`, exact because the body and
            pelvis frames are aligned at zero waist.
    """
    robot: Entity = env.scene[asset_cfg.name]
    bid = asset_cfg.body_ids[0]
    vec = torch.zeros(env.num_envs, 3, device=env.device)
    vec[:, 2] = offset
    pt_w = robot.data.body_link_pos_w[:, bid] + quat_apply(
        robot.data.body_link_quat_w[:, bid], vec)
    pt_b = quat_apply_inverse(
        robot.data.root_link_quat_w, pt_w - robot.data.root_link_pos_w)
    target = torch.tensor(origin_b, device=env.device).clone()
    target[2] += offset
    return pt_b - target


def track_torso_point(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg,
                      offset: float, origin_b: tuple[float, float, float],
                      std: float) -> torch.Tensor:
    """exp(-‖p_b - p_b_nominal‖²/σ²) for a torso-fixed point. Shape
    (num_envs,), unitless.

    Prices trunk lean in METRES, the same currency as the position objective
    the policy would otherwise buy with it. Immune to arm motion (the point is
    on the torso, not in any CoM), so it charges the lower body only for what
    the lower body did.
    """
    err = _torso_point_err(env, asset_cfg, offset, origin_b)
    return torch.exp(-torch.sum(torch.square(err), dim=-1) / std**2)


def track_torso_point_l2(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg,
                         offset: float, origin_b: tuple[float, float, float],
                         clamp: float | None = 0.3) -> torch.Tensor:
    """Unsaturating companion of `track_torso_point`. Shape (num_envs,), m²."""
    err = _torso_point_err(env, asset_cfg, offset, origin_b)
    if clamp is not None:
        err = err.clamp(min=-clamp, max=clamp)
    return torch.sum(torch.square(err), dim=-1)


class track_joint_pos_l2(_joint_selected):
    """L2 companion of the joint imitation kernel over the selected joints;
    optional per-joint `clamp` [rad] bounds the tax under falls."""

    def __call__(self, env: ManagerBasedRlEnv, command_name: str,
                 joint_names: tuple[str, ...], clamp: float | None = None,
                 entity_name: str = "robot") -> torch.Tensor:
        cmd: FootPlacementCommand = _cmd(env, command_name)
        err = (cmd.ref_joint_pos - cmd.robot.data.joint_pos)[:, self.ids]
        if clamp is not None:
            err = err.clamp(min=-clamp, max=clamp)
        return torch.sum(torch.square(err), dim=-1)

class track_joint_vel_l2(_joint_selected):
    """L2 companion of the joint imitation kernel over the selected joints;
    optional per-joint `clamp` [rad/s] bounds the tax under falls."""

    def __call__(self, env: ManagerBasedRlEnv, command_name: str,
                 joint_names: tuple[str, ...], clamp: float | None = None,
                 entity_name: str = "robot") -> torch.Tensor:
        cmd: FootPlacementCommand = _cmd(env, command_name)
        err = (cmd.ref_joint_vel - cmd.robot.data.joint_vel)[:, self.ids]
        if clamp is not None:
            err = err.clamp(min=-clamp, max=clamp)
        return torch.sum(torch.square(err), dim=-1)

class ankle_posture_l2:
    """‖q_ankle - q_target‖² — REFINE-DP's flat-foot bias, with EXPLICIT
    targets: the shared KNEES_BENT keyframe's ankle default (-0.363) is not
    the clips' nominal ankle (-0.2), so pulling toward `default_joint_pos`
    would fight the tracking income."""

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        from mjlab.envs.mdp import resolve_matching_names_values
        asset = env.scene[cfg.params["asset_cfg"].name]
        _, joint_names = asset.find_joints(cfg.params["asset_cfg"].joint_names)
        _, _, vals = resolve_matching_names_values(
            data=cfg.params["targets"], list_of_strings=joint_names)
        self.target = torch.tensor(vals, dtype=torch.float32, device=env.device)

    def __call__(self, env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg,
                 targets: dict) -> torch.Tensor:
        del targets
        asset = env.scene[asset_cfg.name]
        err = asset.data.joint_pos[:, asset_cfg.joint_ids] - self.target
        return torch.sum(torch.square(err), dim=1)


class _cop_forward:
    """Base for the sagittal CoP terms. Per LOADED foot (Σ|f| > ``fz_gate``),
    ``cop_x`` = the |force|-weighted centroid of that foot's ground contacts
    in the foot body frame (m, + = toe side of the ankle_roll origin; sole
    centre +0.039, heel −0.054, toe +0.132), then ``err = cop_x − target``
    with the HEEL side (err < 0) scaled by ``heel_scale``. Unloaded feet:
    err 0, not loaded.

    ``sensor_name``: PER-CONTACT sensor (``reduce="none"``, fields
    ``found/force/pos``; |force| is frame-invariant so no ``global_frame``),
    primaries = the two foot bodies in model-body order;
    ``asset_cfg.body_names`` = the same two bodies (left, right)."""

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        robot = env.scene[cfg.params["asset_cfg"].name]
        ids, names = robot.find_bodies(
            list(cfg.params["asset_cfg"].body_names), preserve_order=True)
        assert len(ids) == 2 and names[0].startswith("left"), names
        self.ids = torch.tensor(ids, dtype=torch.long, device=env.device)

    def _err(self, env: ManagerBasedRlEnv, sensor_name: str,
             asset_cfg: SceneEntityCfg, target: float, heel_scale: float,
             fz_gate: float) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns ``(err [B,2], loaded [B,2] bool)``."""
        sensor: ContactSensor = env.scene[sensor_name]
        data = sensor.data
        assert data.force is not None and data.pos is not None
        B = env.num_envs
        found = (data.found.view(B, 2, -1) > 0).float()
        w = data.force.view(B, 2, -1, 3).norm(dim=-1) * found        # [B,2,S]
        W = w.sum(-1)                                                # [B,2]
        cop_w = (data.pos.view(B, 2, -1, 3) * w[..., None]).sum(2) / W.clamp(min=fz_gate)[..., None]
        robot = env.scene[asset_cfg.name]
        rel = cop_w - robot.data.body_link_pos_w[:, self.ids]
        q = robot.data.body_link_quat_w[:, self.ids]
        cop_x = quat_apply_inverse(q.reshape(-1, 4), rel.reshape(-1, 3)).view(B, 2, 3)[..., 0]
        loaded = W > fz_gate
        err = cop_x - target
        err = torch.where(err < 0.0, heel_scale * err, err)
        return torch.where(loaded, err, torch.zeros_like(err)), loaded


class cop_forward_exp(_cop_forward):
    """Mean over LOADED feet of ``exp(−err²/std²)``, 0 with no foot loaded
    (see `_cop_forward`). Positive weight; pair with `cop_forward_l2`."""

    def __call__(self, env: ManagerBasedRlEnv, sensor_name: str,
                 asset_cfg: SceneEntityCfg, target: float, std: float,
                 heel_scale: float = 2.0, fz_gate: float = 50.0) -> torch.Tensor:
        err, loaded = self._err(env, sensor_name, asset_cfg, target, heel_scale, fz_gate)
        k = torch.exp(-torch.square(err) / std**2) * loaded
        return k.sum(dim=1) / loaded.sum(dim=1).clamp(min=1)


class cop_forward_l2(_cop_forward):
    """Σ over loaded feet of ``min(err², clamp²)`` (see `_cop_forward`).
    Negative weight; the gradient beyond the exp kernel's reach."""

    def __call__(self, env: ManagerBasedRlEnv, sensor_name: str,
                 asset_cfg: SceneEntityCfg, target: float, clamp: float,
                 heel_scale: float = 2.0, fz_gate: float = 50.0) -> torch.Tensor:
        err, _ = self._err(env, sensor_name, asset_cfg, target, heel_scale, fz_gate)
        return torch.square(err.clamp(-clamp, clamp)).sum(dim=1)


def base_height_range(env: ManagerBasedRlEnv, min_height: float,
                      max_height: float, asset_cfg: SceneEntityCfg | None = None,
                      squared: bool = True) -> torch.Tensor:
    """Distance of the root LINK ORIGIN height (world z) outside the band
    [min_height, max_height]: zero inside, [max(0, h_min - h)]² + [max(0,
    h - h_max)]² (m²) when `squared`, else the plain distance (m, constant
    gradient). Shape (num_envs,). Negative weight."""
    asset = env.scene[asset_cfg.name if asset_cfg else "robot"]
    h = asset.data.root_link_pos_w[:, 2]
    low = torch.clamp(min_height - h, min=0.0)
    high = torch.clamp(h - max_height, min=0.0)
    if squared:
        return torch.square(low) + torch.square(high)
    return low + high

def base_height_l2(env: ManagerBasedRlEnv, target_height: float) -> torch.Tensor:
    """(h - h_target)² on the root height."""
    robot = env.scene["robot"]
    return (robot.data.root_link_pos_w[:, 2] - target_height).pow(2)

def swing_no_lift(env: ManagerBasedRlEnv, command_name: str, sensor_name: str,
                  window: tuple[float, float] = (0.40, 0.70)) -> torch.Tensor:
    """1 while the COMMANDED swing foot is still in ground contact during the
    swing window of the countdown. Negative weight — prices the 'never lift
    the commanded foot' hack DIRECTLY (three runs converged to it: stance/DS
    income is free and both imitation kernels are gradient-dead by then).

    The clips' fixed phase fractions (ds 25 / swing 45 / ds 30) put the swing
    at countdown 0.75→0.30; the default window (0.40, 0.70) sits strictly
    inside it, clear of lift-off timing slack and of the last-third landing
    ramp, so a correctly-tracking policy pays nothing. Reads only the command
    (one-hot + countdown — works for the clip AND planner terms; countdown is
    forced 0 during pauses/stand, outside the window) and foot contact —
    deployment-consistent.
    """
    cmd = _cmd(env, command_name)
    c = cmd.command                              # [B, 7] onehot | countdown | …
    cd = c[:, 2]
    in_swing = (cd > window[0]) & (cd < window[1])
    sensor = env.scene[sensor_name]
    contact = (sensor.data.found.view(env.num_envs, -1) > 0)[:, :2]
    swing_idx = (c[:, 1] > c[:, 0]).long()       # 0 = left, 1 = right
    idx = torch.arange(env.num_envs, device=env.device)
    return (in_swing & contact[idx, swing_idx]).float()


def landing_air_time(env: ManagerBasedRlEnv, sensor_name: str,
                     threshold: float,
                     command_name: str | None = None) -> torch.Tensor:
    """REFINE-DP's foot-airtime credit: on each touchdown, (air_time - τ) —
    shuffle steps (short flights) earn negative credit.

    With `command_name` set, ONLY the commanded swing foot's touchdowns are
    settled. Measured exploit (2026-08-04 joint run 4): the foot-agnostic sum
    PAID the stance foot for hopping — stance touchdowns p50 1/step with air
    p90 0.775 s ⇒ +2..+7.5 credit each — funding the both-feet tap-dance while
    tracking stayed good (feet err 0.018 m). The single-step references keep
    the stance foot planted; its flight must never earn."""
    sensor = env.scene[sensor_name]
    data = sensor.data
    assert data.current_contact_time is not None
    first_contact = (data.current_contact_time > 0.0) & (
        data.current_contact_time <= env.step_dt + 1e-6)
    credit = first_contact.float() * (data.last_air_time - threshold)
    if command_name is None:
        return torch.sum(credit, dim=1)
    c = _cmd(env, command_name).command
    swing_idx = (c[:, 1] > c[:, 0]).long()
    idx = torch.arange(env.num_envs, device=env.device)
    return credit[idx, swing_idx]


def stance_no_air(env: ManagerBasedRlEnv, command_name: str, sensor_name: str,
                  window: tuple[float, float] = (0.02, 0.98)) -> torch.Tensor:
    """1 while the STANCE (non-commanded) foot is out of ground contact during
    an active step. Negative weight — the companion of `swing_no_lift`: the
    references keep the stance foot planted from lift-off to landing, and its
    flight was an unpriced quantity (worse: the airtime credit subsidized it).
    Same deployment-consistent inputs (command one-hot + countdown + contact;
    countdown is 0 during pauses/stand, outside the window)."""
    cmd = _cmd(env, command_name)
    c = cmd.command
    cd = c[:, 2]
    in_step = (cd > window[0]) & (cd < window[1])
    sensor = env.scene[sensor_name]
    contact = (sensor.data.found.view(env.num_envs, -1) > 0)[:, :2]
    stance_idx = (c[:, 0] > c[:, 1]).long()      # opposite of the swing foot
    idx = torch.arange(env.num_envs, device=env.device)
    return (in_step & ~contact[idx, stance_idx]).float()


def _heading_err(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """wrap(pelvis yaw - the foothold command's stance-foot yaw
    (`cmd.stance_yaw_w`)), world frame. Shape (num_envs,), rad."""
    cmd = _cmd(env, command_name)
    q = cmd.robot.data.root_link_quat_w
    yaw = torch.atan2(2.0 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]),
                      1.0 - 2.0 * (q[:, 2] * q[:, 2] + q[:, 3] * q[:, 3]))
    e = yaw - cmd.stance_yaw_w
    return torch.atan2(torch.sin(e), torch.cos(e))


def foothold_heading(env: ManagerBasedRlEnv, command_name: str,
                     std: float) -> torch.Tensor:
    """exp(-e²/σ²) on the pelvis-vs-stance-foot heading error of
    `_heading_err`. Shape (num_envs,), in [0, 1]. Positive weight; active in
    holds too (an aligned stander earns it)."""
    return torch.exp(-_heading_err(env, command_name).square() / std**2)


def foothold_heading_l2(env: ManagerBasedRlEnv, command_name: str,
                        clamp: float = 1.5) -> torch.Tensor:
    """min(|e|, clamp)² on the heading error of `_heading_err`. Shape
    (num_envs,), rad². Negative weight — the unsaturating pair of
    `foothold_heading`."""
    return _heading_err(env, command_name).abs().clamp(max=clamp).square()


def swing_airborne(env: ManagerBasedRlEnv, command_name: str,
                   sensor_name: str) -> torch.Tensor:
    """1 while the commanded swing foot is out of ground contact inside its
    swing window (`cmd.swing_active`), else 0 — including all of standing
    and double support, which earn nothing here. Shape (num_envs,).
    Positive weight; independent of the commanded step length."""
    cmd = _cmd(env, command_name)
    contact = _foot_contact(env, sensor_name)
    idx = torch.arange(env.num_envs, device=env.device)
    airborne = ~contact[idx, cmd.swing]
    return (airborne & cmd.swing_active & ~cmd.standing).float()


def foothold_posture_prior(env: ManagerBasedRlEnv, command_name: str,
                           std: float) -> torch.Tensor:
    """exp(-Σ_mask (q - q_prior)² / σ²) over the foothold command's prior
    joints (``prior_joint_mask``: waist + the swing leg while stepping).
    Shape (num_envs,), in (0, 1]. Requires ``posture_prior`` on the command."""
    cmd = _cmd(env, command_name)
    err = (cmd.ref_joint_pos_prior - cmd.robot.data.joint_pos[:, :15]) * cmd.prior_joint_mask
    return torch.exp(-err.square().sum(dim=-1) / std**2)


def feet_contact_schedule(env: ManagerBasedRlEnv, command_name: str,
                          sensor_name: str) -> torch.Tensor:
    """Fraction of feet whose ground contact matches the foothold command's
    schedule: the stance foot in contact throughout, the swing foot airborne
    inside the swing window (`cmd.swing_active`) and in contact outside it,
    both in contact while standing. Shape (num_envs,), in [0, 1]. Positive
    weight; independent of the commanded step length."""
    cmd = _cmd(env, command_name)
    contact = _foot_contact(env, sensor_name)
    want = torch.ones_like(contact)
    idx = torch.arange(env.num_envs, device=env.device)
    want[idx, cmd.swing] = ~cmd.swing_active
    return (contact == want).float().mean(dim=-1)


class torque_soft_limits:
    """max(0, |τ| - ratio·τ_max) summed over actuators. Actuator force limits
    come from the compiled model's forcerange (mujoco clamps AT the limit, so
    the REFINE-DP literal max(0, |τ|-τ_max) would always be 0 — the soft ratio
    restores the gradient below saturation).

    `actuator_names` (name/regex tuple, G1 actuator names == joint names)
    restricts the term to a subset — resolved BY NAME via `find_actuators`
    into the entity's actuator axis (the ctrl vector is actuator-DECLARATION
    order, never index it with joint indices)."""

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        robot = env.scene[cfg.params.get("asset_cfg", SceneEntityCfg("robot")).name]
        names = cfg.params.get("actuator_names")
        if names is None:
            local = list(range(len(robot.actuator_names)))
        else:
            local, _ = robot.find_actuators(list(names))
        ctrl_ids = robot.indexing.ctrl_ids.cpu().numpy()[local]
        force_range = env.sim.mj_model.actuator_forcerange[ctrl_ids]
        self.limit = torch.tensor(
            force_range[:, 1], dtype=torch.float32, device=env.device)
        self.local = torch.tensor(local, dtype=torch.long, device=env.device)
        self.robot = robot

    def __call__(self, env: ManagerBasedRlEnv, ratio: float,
                 actuator_names: tuple[str, ...] | None = None,
                 asset_cfg: SceneEntityCfg | None = None) -> torch.Tensor:
        force = self.robot.data.actuator_force[:, self.local]
        excess = torch.clamp(torch.abs(force) - ratio * self.limit, min=0.0)
        return torch.sum(excess, dim=1)


class posture_exp:
    """exp(-mean_j((q_j - q_default_j)² / std_j²)) over the joints of
    `asset_cfg`, with a FIXED per-joint `std` ({name/regex: rad}; every
    selected joint must match exactly one key). Shape (num_envs,), in [0, 1].
    No command dependence — the same tolerance in stands and steps."""

    def __init__(self, cfg: RewardTermCfg, env: "ManagerBasedRlEnv"):
        asset: Entity = env.scene[cfg.params["asset_cfg"].name]
        _, joint_names = asset.find_joints(cfg.params["asset_cfg"].joint_names)
        _, _, std = resolve_matching_names_values(
            data=cfg.params["std"], list_of_strings=joint_names)
        self.std = torch.tensor(std, device=env.device, dtype=torch.float32)

    def __call__(self, env: "ManagerBasedRlEnv", asset_cfg: SceneEntityCfg,
                 std) -> torch.Tensor:
        del std  # resolved once in __init__
        asset: Entity = env.scene[asset_cfg.name]
        err2 = torch.square(asset.data.joint_pos[:, asset_cfg.joint_ids]
                            - asset.data.default_joint_pos[:, asset_cfg.joint_ids])
        return torch.exp(-torch.mean(err2 / self.std**2, dim=1))


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

def _pose_errors(
    env: "ManagerBasedRlEnv", command_names: tuple[str, str]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-hand position error [m] and geodesic angle [rad]. Each (B, n_hands)."""
    pos_err, ang_err = [], []
    for name in command_names:
        term: EeTargetCommand = env.command_manager.get_term(name)
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

def _wrap_pi(x: torch.Tensor) -> torch.Tensor:
    return torch.remainder(x + torch.pi, 2.0 * torch.pi) - torch.pi

def ee_ref_joint_pos_tracking(env: ManagerBasedRlEnv, command_names: str, std: float) -> torch.Tensor:
    command_term: EEQposeCommand = env.command_manager.get_term(command_names)
    robot: Entity = env.scene['robot']
    ref_joint_pos = command_term.ref_joint_pos
    joint_pos = robot.data.joint_pos[:, command_term.arm_joint_ids]
    error = torch.sum(torch.square(ref_joint_pos - joint_pos), dim=-1)
    return torch.exp(-error / std ** 2)

def ee_ref_joint_pos_l2(env: ManagerBasedRlEnv, command_names: str) -> torch.Tensor:
    command_term: EEQposeCommand = env.command_manager.get_term(command_names)
    robot: Entity = env.scene['robot']
    ref_joint_pos = command_term.ref_joint_pos
    joint_pos = robot.data.joint_pos[:, command_term.arm_joint_ids]
    error = torch.sum(torch.square(ref_joint_pos - joint_pos), dim=-1)
    return error

class _arm_swivel:
    """Per-hand SEW swivel angle ψ [rad] of the elbow about the shoulder→wrist
    axis, measured from link origins (`*_shoulder_roll_link`, `*_elbow_link`,
    `*_wrist_pitch_link` = wrist centre). ψ = 0 when the elbow lies in the
    half-plane of that axis and the root frame's −z; positive toward the
    OUTSIDE of the body (+y left hand, −y right). This is the single
    redundant coordinate the tip pose leaves free, so pricing it never
    trades against EE tracking. Hands resolved from `command_names` order."""

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        self._ids: list[tuple[int, int, int]] | None = None
        self._sign: torch.Tensor | None = None

    def _psi(self, env: ManagerBasedRlEnv, command_names: tuple[str, str]
             ) -> torch.Tensor:
        term: EeTargetCommand = env.command_manager.get_term(command_names[0])
        robot = term.robot
        if self._ids is None:
            ids, signs = [], []
            for name in command_names:
                side = env.command_manager.get_term(name).cfg.body_name.split("_")[0]
                assert side in ("left", "right"), side
                ids.append(tuple(
                    robot.find_bodies(f"{side}_{b}")[0][0]
                    for b in ("shoulder_roll_link", "elbow_link", "wrist_pitch_link")))
                signs.append(1.0 if side == "left" else -1.0)
            self._ids = ids
            self._sign = torch.tensor(signs, device=env.device)
        R = matrix_from_quat(robot.data.root_link_quat_w)          # [B,3,3]
        down = -R[:, :, 2]
        out = []
        for (s_id, e_id, w_id), sgn in zip(self._ids, self._sign.tolist()):
            p = robot.data.body_link_pos_w
            s, e, w = p[:, s_id], p[:, e_id], p[:, w_id]
            a = w - s
            a = a / a.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            b1 = down - (down * a).sum(-1, keepdim=True) * a
            b1 = b1 / b1.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            b2 = torch.cross(a, b1, dim=-1)
            b2 = torch.where((b2 * (sgn * R[:, :, 1])).sum(-1, keepdim=True) < 0, -b2, b2)
            r = e - s
            r = r - (r * a).sum(-1, keepdim=True) * a
            out.append(torch.atan2((r * b2).sum(-1), (r * b1).sum(-1)))
        return torch.stack(out, dim=-1)


class arm_swivel_exp(_arm_swivel):
    """exp(−(wrap(ψ − psi_target) / std)²), SUM over hands. In [0, 2]."""

    def __call__(self, env: ManagerBasedRlEnv, std: float, psi_target: float,
                 command_names: tuple[str, str]) -> torch.Tensor:
        d = _wrap_pi(self._psi(env, command_names) - psi_target)
        return torch.exp(-((d / std) ** 2)).sum(-1)


class arm_swivel_l1(_arm_swivel):
    """Σ|wrap(ψ − psi_target)| over hands [rad]. Negative weight — the slope
    that stays alive where the exp partner is dead."""

    def __call__(self, env: ManagerBasedRlEnv, psi_target: float,
                 command_names: tuple[str, str]) -> torch.Tensor:
        return _wrap_pi(self._psi(env, command_names) - psi_target).abs().sum(-1)


class term_action_rate_l2:
    """Action-rate L2 over ONE action term's slice of the concatenated action
    vector. The slice is resolved BY NAME from the ActionManager layout
    (`active_terms` declaration order × `action_dim`), lazily on first call —
    no hardcoded indices, robust to term reordering or new terms."""

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        self._sl: slice | None = None

    def __call__(self, env: ManagerBasedRlEnv, action_term_name: str
                 ) -> torch.Tensor:
        am = env.action_manager
        if self._sl is None:
            start = 0
            for n in am.active_terms:
                dim = am.get_term(n).action_dim
                if n == action_term_name:
                    self._sl = slice(start, start + dim)
                    break
                start += dim
            else:
                raise KeyError(
                    f"action term {action_term_name!r} not in {am.active_terms}")
        d = am.action[:, self._sl] - am.prev_action[:, self._sl]
        return torch.sum(torch.square(d), dim=1)


class ee_acc_l2:
    """Finite-difference EE acceleration penalty: sum over the selected bodies
    of, clamped per body. mjlab exposes body
    velocities but no body accelerations (mujoco's cacc is not computed), so
    the derivative is numerical. The clamp caps the per-body acc norm BEFORE
    squaring (contact impacts spike Δv — never square an unbounded impact
    quantity), and the first frame after each reset contributes 0 (no phantom
    spike from the reset teleport)."""

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        asset_cfg = cfg.params["asset_cfg"]
        self.robot: Entity = env.scene[asset_cfg.name]
        ids, _ = self.robot.find_bodies(asset_cfg.body_names,
                                        preserve_order=True)
        self.body_ids = torch.tensor(ids, dtype=torch.long, device=env.device)
        self.vel_type = cfg.params.get("vel_type", "lin")
        self._prev = torch.zeros(env.num_envs, len(ids), 3, device=env.device)
        self._fresh = torch.ones(env.num_envs, dtype=torch.bool,
                                 device=env.device)

    def __call__(self, env: ManagerBasedRlEnv,
                 asset_cfg: SceneEntityCfg | None = None,
                 vel_type: str = "lin", clamp: float = 20.0) -> torch.Tensor:
        data = self.robot.data
        v = (data.body_link_lin_vel_w if self.vel_type == "lin"
             else data.body_link_ang_vel_w)[:, self.body_ids]
        acc = (v - self._prev).norm(dim=-1) / env.step_dt     # [B, n_bodies]
        self._prev = v.clone()
        acc = torch.where(self._fresh.unsqueeze(-1),
                          torch.zeros_like(acc), acc.clamp(max=clamp))
        self._fresh[:] = False
        return torch.sum(torch.square(acc), dim=1)

    def reset(self, env_ids=None):
        if env_ids is None:
            self._fresh[:] = True
        else:
            self._fresh[env_ids] = True


def feet_stumble(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    threshold: float = 4.0,
) -> torch.Tensor:
    """Toe-catch / edge-shear event: any foot whose HORIZONTAL contact force
    exceeds `threshold`x the vertical one. `threshold` is a force RATIO and
    must stay above the friction-cone ceiling (μ DR reaches 1.2 — legitimate
    push-off shear is up to 1.2x vertical); the legged-gym convention is 4."""
    sensor: ContactSensor = env.scene[sensor_name]
    force = sensor.data.force
    assert force is not None
    horizontal = torch.norm(force[..., :2], dim=-1)
    vertical = torch.abs(force[..., 2])
    return torch.any(horizontal > threshold * vertical, dim=1).float()


def _clf_term(env: ManagerBasedRlEnv, command_name: str):
    """The command term IF it carries a CLF, else None (the deploy planner and
    the tasks that declare no `clf` cfg share these reward declarations)."""
    term = env.command_manager.get_term(command_name)
    return term if getattr(term, "clf", None) is not None else None


def clf_reward(env: ManagerBasedRlEnv, command_name: str,
               eta_max: float) -> torch.Tensor:
    """exp(-V / sigma_v), in (0, 1]. POSITIVE weight.

    `sigma_v = mu_max(P) * eta_max^2` is the worst-case V over the declared
    operating range, so the kernel spans [e^-1, 1] across it — the
    normalization is DERIVED from `eta_max`, not tuned. `eta_max` is therefore
    the real knob: errors beyond it are off-scale and the kernel saturates.
    V and Vdot are computed once per step by the command term; this only reads.
    """
    term = _clf_term(env, command_name)
    if term is None:
        return torch.zeros(env.num_envs, device=env.device)
    return torch.exp(-term.v / term.clf.sigma_v(eta_max))


def clf_decreasing_condition(env: ManagerBasedRlEnv, command_name: str,
                             eta_max: float, eta_dot_max: float,
                             lam: float) -> torch.Tensor:
    """clamp(Vdot + lam*V, min=0) / sigma_vdot, in [0, 1]. NEGATIVE weight.

    The CLF decrease condition Vdot <= -lam*V asks the tracking error to decay
    like e^(-lam t); this charges the shortfall. It is the one term in the
    table that is a function of the ACTION (through Vdot) and not of the state
    alone — two states with equal error but opposite trends score differently,
    which no state-only kernel can express. It also stays linear inside the
    clip, so unlike an exp kernel its gradient does not die as the error grows.
    """
    term = _clf_term(env, command_name)
    if term is None:
        return torch.zeros(env.num_envs, device=env.device)
    violation = torch.clamp(term.vdot + lam * term.v, min=0.0)
    sigma = term.clf.sigma_vdot(eta_max, eta_dot_max, lam)
    return torch.clamp(violation / sigma, max=1.0)


def feet_slide(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    asset_cfg: SceneEntityCfg,
    force_threshold: float = 1.0,
) -> torch.Tensor:
    """Σ over feet of ‖v_xy‖ · 1{in contact} — IsaacLab's `feet_slide`, ported.
    Negative weight.

    Same form as the original: a BINARY contact mask times the HORIZONTAL body
    linear-velocity NORM (not squared), summed over feet, so only a foot that
    is on the ground pays. Three mjlab API differences:
      * the contact sensor exposes `data.force` [B, n_feet, 3] directly — there
        is no `net_forces_w_history` ring buffer, so no max-over-history
        debounce (the threshold is applied to the current frame);
      * the entity attribute is `body_link_lin_vel_w`, not `body_lin_vel_w`;
      * the sensor is addressed by NAME, matching every other contact term here
        (`swing_no_lift`, `stance_no_air`, `feet_stumble`).

    Column order: the sensor's primary pattern and `asset_cfg.body_names` both
    resolve in model-body order, so column i is the same foot on both sides —
    the assumption `swing_no_lift` already makes (0 = left, 1 = right).

    NB the foot BODY frame is the ankle_roll_link origin, ~25 mm above the sole
    capsules, so a foot rocking about a planted sole shows a lever-arm velocity
    here even at zero true slip. The IsaacLab term carries the same bias; it is
    also why the term doubles as a foot-rocking penalty.
    """
    sensor: ContactSensor = env.scene[sensor_name]
    force = sensor.data.force
    assert force is not None
    contact = force.view(env.num_envs, -1, 3).norm(dim=-1) > force_threshold
    asset: Entity = env.scene[asset_cfg.name]
    vel_xy = asset.data.body_link_lin_vel_w[:, asset_cfg.body_ids, :2]
    return torch.sum(vel_xy.norm(dim=-1) * contact[:, : vel_xy.shape[1]].float(),
                     dim=1)


def track_linear_velocity(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward for tracking the commanded base linear velocity.

  The commanded z velocity is assumed to be zero.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  actual = asset.data.root_link_lin_vel_b
  xy_error = torch.sum(torch.square(command[:, :2] - actual[:, :2]), dim=1)
  z_error = torch.square(actual[:, 2])
  lin_vel_error = xy_error + (2 * z_error)
  return torch.exp(-lin_vel_error / std**2)


def track_angular_velocity(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward heading error for heading-controlled envs, angular velocity for others.

  The commanded xy angular velocities are assumed to be zero.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  actual = asset.data.root_link_ang_vel_b
  z_error = torch.square(command[:, 2] - actual[:, 2])
  xy_error = torch.sum(torch.square(actual[:, :2]), dim=1)
  ang_vel_error = z_error + (0.05 * xy_error)
  return torch.exp(-ang_vel_error / std**2)


def body_orientation_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward flat base orientation (robot being upright).

  If asset_cfg has body_ids specified, computes the projected gravity
  for that specific body. Otherwise, uses the root link projected gravity.
  """
  asset: Entity = env.scene[asset_cfg.name]

  # If body_ids are specified, compute projected gravity for that body.
  if asset_cfg.body_ids:
    body_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :]  # [B, N, 4]
    body_quat_w = body_quat_w.squeeze(1)  # [B, 4]
    gravity_w = asset.data.gravity_vec_w  # [3]
    projected_gravity_b = quat_apply_inverse(body_quat_w, gravity_w)  # [B, 3]
    xy_squared = torch.sum(torch.square(projected_gravity_b[:, :2]), dim=1)
  else:
    # Use root link projected gravity.
    xy_squared = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
  return xy_squared


# self_collision_cost: use the mjlab builtin (`mjlab.tasks.velocity.mdp`) —
# our copy was byte-identical.


def feet_air_time(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  threshold: float = 0.4,
  command_name: str | None = None,
  command_threshold: float = 0.1,
) -> torch.Tensor:
  """Reward feet air time."""
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  air_time = sensor_data.current_air_time
  contact_time = sensor_data.current_contact_time
  in_contact = contact_time > 0.0
  in_mode_time = torch.where(in_contact, contact_time, air_time)
  single_stance = torch.mean(in_contact.float(), dim=1) == 0.5
  mode_time = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
  error = torch.abs(mode_time - threshold)
  reward = torch.clamp(threshold - error, min=0.0)
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      scale = (total_command > command_threshold).float()
      reward *= scale
  return reward


def feet_clearance(
  env: ManagerBasedRlEnv,
  target_height: float,
  command_name: str | None = None,
  command_threshold: float = 0.1,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize deviation from target clearance height, weighted by foot velocity."""
  asset: Entity = env.scene[asset_cfg.name]
  foot_z = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]  # [B, N]
  foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]  # [B, N, 2]
  vel_norm = torch.norm(foot_vel_xy, dim=-1)  # [B, N]
  delta = torch.abs(foot_z - target_height)  # [B, N]
  cost = torch.sum(delta * vel_norm, dim=1)  # [B]
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      active = (total_command > command_threshold).float()
      cost = cost * active
  return cost


def feet_gait(
        env: ManagerBasedRlEnv,
        period: float,
        offset: list[float],
        threshold: float,
        command_threshold: float,
        command_name: str,
        sensor_name: str,
) -> torch.Tensor:
    sensor: ContactSensor = env.scene[sensor_name]
    is_contact = sensor.data.current_contact_time > 0
    global_phase = ((env.episode_length_buf * env.step_dt) / period).unsqueeze(1)
    offsets = torch.as_tensor(offset, device=env.device, dtype=global_phase.dtype).view(1, -1)
    leg_phase = (global_phase + offsets) % 1.0
    is_stance = (leg_phase < threshold)
    reward = (is_stance == is_contact).float().mean(dim=1)
    if command_name is not None:
        command = env.command_manager.get_command(command_name)
        if command is not None:
            linear_norm = torch.norm(command[:, :2], dim=1)
            angular_norm = torch.abs(command[:, 2])
            total_command = linear_norm + angular_norm
            scale = (total_command > command_threshold).float()
            reward *= scale
    return reward


class feet_swing_height:
  """Penalize deviation from target swing height, evaluated at landing."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    self.sensor_name = cfg.params["sensor_name"]
    self.site_names = cfg.params["asset_cfg"].site_names
    self.peak_heights = torch.zeros(
      (env.num_envs, len(self.site_names)), device=env.device, dtype=torch.float32
    )
    self.step_dt = env.step_dt

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    target_height: float,
    command_name: str,
    command_threshold: float,
    asset_cfg: SceneEntityCfg,
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene[sensor_name]
    command = env.command_manager.get_command(command_name)
    assert command is not None
    foot_heights = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]
    in_air = contact_sensor.data.found == 0
    self.peak_heights = torch.where(
      in_air,
      torch.maximum(self.peak_heights, foot_heights),
      self.peak_heights,
    )
    first_contact = contact_sensor.compute_first_contact(dt=self.step_dt)
    linear_norm = torch.norm(command[:, :2], dim=1)
    angular_norm = torch.abs(command[:, 2])
    total_command = linear_norm + angular_norm
    active = (total_command > command_threshold).float()
    error = self.peak_heights / target_height - 1.0
    cost = torch.sum(torch.square(error) * first_contact.float(), dim=1) * active
    num_landings = torch.sum(first_contact.float())
    peak_heights_at_landing = self.peak_heights * first_contact.float()
    mean_peak_height = torch.sum(peak_heights_at_landing) / torch.clamp(
      num_landings, min=1
    )
    env.extras["log"]["Metrics/peak_height_mean"] = mean_peak_height
    self.peak_heights = torch.where(
      first_contact,
      torch.zeros_like(self.peak_heights),
      self.peak_heights,
    )
    return cost



def stand_still(
        env: ManagerBasedRlEnv,
        command_name: str,
        command_threshold: float = 0.1,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    diff_angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    reward = torch.sum(torch.square(diff_angle), dim=1)
    if command_name is not None:
        command = env.command_manager.get_command(command_name)
        if command is not None:
            linear_norm = torch.norm(command[:, :2], dim=1)
            angular_norm = torch.abs(command[:, 2])
            total_command = linear_norm + angular_norm
            scale = (total_command <= command_threshold).float()
            reward *= scale
    return reward

def foot_flat_orientation(
    env: "ManagerBasedRlEnv",
    asset_cfg: "SceneEntityCfg",
    sigma: float = 0.15,
) -> torch.Tensor:
    """Reward flat foot orientation — active in both swing and stance."""
    robot = env.scene[asset_cfg.name]
    q = robot.data.site_quat_w[:, asset_cfg.site_ids, :]
    tilt = 2.0 * (q[:, :, 1].pow(2) + q[:, :, 2].pow(2))
    return torch.exp(-tilt / sigma).mean(dim=-1)


def foot_tilt_l2(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    sensor_name: str | None = None,
) -> torch.Tensor:
    """Σ over the selected foot SITES of the squared tilt angle between the
    site z axis and world up (4·(qx²+qy²) = θ² to second order), counted only
    while that foot is in ground contact when `sensor_name` is given (site
    order must match the sensor's foot columns: left, right). Shape
    (num_envs,), rad². Negative weight."""
    robot = env.scene[asset_cfg.name]
    q = robot.data.site_quat_w[:, asset_cfg.site_ids, :]
    tilt2 = 4.0 * (q[:, :, 1].pow(2) + q[:, :, 2].pow(2))
    if sensor_name is not None:
        tilt2 = tilt2 * _foot_contact(env, sensor_name)[:, : tilt2.shape[1]].float()
    return tilt2.sum(dim=-1)


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


# ── hand-force rewards (measured whole-body hand_force recipe) ──────────────
# PRIORITY ORDER (user decision on the source stack): signal CONTINUITY first,
# magnitude accuracy second — the tactile deliverable is an UNBROKEN force
# signal; chattering contact is worse than a steady press at the wrong
# magnitude. `hand_contact_hold` charges dropout DURATION, `HandForceRate`
# dropout FREQUENCY; tracking/l1 are demoted accuracy shaping (never square an
# unbounded contact force). Phantom windows take the idle/free-pass branch
# everywhere: their press error is ‖f_cmd‖ regardless of behavior.


def _force_errors(
    env: ManagerBasedRlEnv, command_names: tuple[str, str]
) -> torch.Tensor:
    """(B, n_hands) masked-branch force-vector error norm [N]."""
    errs = []
    for name in command_names:
        term: EeContactTargetCommand = env.command_manager.get_term(name)
        masked = (term.contact_mask > 0.5) & (term.phantom < 0.5)
        f_meas = term.measured_force_w()
        err_press = (f_meas - term.force_cmd_w).norm(dim=-1)
        err_idle = f_meas.norm(dim=-1)
        errs.append(torch.where(masked, err_press, err_idle))
    return torch.stack(errs, dim=-1)


def hand_force_tracking(
    env: ManagerBasedRlEnv, std: float, command_names: tuple[str, str]
) -> torch.Tensor:
    """exp(-(err/std)^2) per hand, SUM over hands. In [0, 2]."""
    err = _force_errors(env, command_names)
    return torch.exp(-((err / std) ** 2)).sum(-1)


def hand_force_l1(
    env: ManagerBasedRlEnv, clamp: float, command_names: tuple[str, str]
) -> torch.Tensor:
    """Clamped force-error norm, SUM over hands [N]. Negative weight."""
    err = _force_errors(env, command_names)
    return err.clamp(max=clamp).sum(-1)


def hand_contact_hold(
    env: ManagerBasedRlEnv,
    command_names: tuple[str, str],
    floor: float = 3.0,
    grace_s: float = 1.0,
) -> torch.Tensor:
    """PRIMARY continuity term: saturating normal-force ramp vs a low floor.

    Free passes (reward 1): commanded magnitude below the floor (unmasked,
    ramp edges), the first ``grace_s`` of a window (reach transit), and
    phantom windows. SUM over hands, in [0, 2]."""
    out = None
    for name in command_names:
        term: EeContactTargetCommand = env.command_manager.get_term(name)
        f_n = (
            (term.measured_force_w() * term.press_dir_w).sum(-1).clamp(min=0.0)
        )
        r = (f_n / floor).clamp(0.0, 1.0)
        free = (
            (term.force_cmd_mag < floor)
            | (term._contact_age < grace_s)
            | (term.phantom > 0.5)
        )
        r = torch.where(free, torch.ones_like(r), r)
        out = r if out is None else out + r
    return out


def contact_alignment(
    env: ManagerBasedRlEnv, command_names: tuple[str, str]
) -> torch.Tensor:
    """mask=1 → measured contact (0/1); mask=0 → 1. SUM over hands, [0, 2]."""
    out = None
    for name in command_names:
        term: EeContactTargetCommand = env.command_manager.get_term(name)
        masked = (term.contact_mask > 0.5) & (term.phantom < 0.5)
        contact = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        for sensor in term._sensors:
            found = sensor.data.found
            if found is not None:
                contact |= (found > 0).any(dim=-1)
        contact = contact.float()
        r = torch.where(masked, contact, torch.ones_like(contact))
        out = r if out is None else out + r
    return out


def hand_force_press_depth(
    env: ManagerBasedRlEnv,
    command_names: tuple[str, str],
    k_virt: float = 500.0,
    std: float = 0.03,
) -> torch.Tensor:
    """Force-error → position-target reparameterization (arXiv:2505.20829).

    Never scores force directly: the commanded tip position is displaced along
    the press axis by the NORMAL-force error through a virtual stiffness,
    ``target = cmd_pos + ((f_cmd − f_meas·n̂)/k_virt)·n̂``, and the tip is
    scored with an exp position kernel. Pressing too lightly moves the target
    INTO the surface (a dense positional "press deeper"), too hard moves it
    out; at f_meas = f_cmd the target is exactly the command — no forgiveness.
    Continuous across the touch boundary (f_meas = 0 → a target f_cmd/k_virt
    inside the surface drives approach + loading with no gate flip).

    ``k_virt`` is reward-geometry-calibrated (20 N error ↔ 4 cm shift ≈ the
    arm kernels' scale), NOT the cube's contact stiffness — recalibrate if
    force_range or the tracking σ change. Phantom + unmasked free-pass (1).
    SUM over hands, in [0, 2]."""
    out = None
    for name in command_names:
        term: EeContactTargetCommand = env.command_manager.get_term(name)
        masked = (term.contact_mask > 0.5) & (term.phantom < 0.5)
        n_hat = term.press_dir_w
        f_n = (term.measured_force_w() * n_hat).sum(-1).clamp(min=0.0)
        depth = ((term.force_cmd_mag - f_n) / k_virt).unsqueeze(-1)
        # The surface rides the gliding reference (whole-window semantics).
        target = term.target_pose_w()[0] + depth * n_hat
        tip = term.measured_ee_pose_w()[0]
        r = torch.exp(-(((tip - target).norm(dim=-1)) / std) ** 2)
        r = torch.where(masked, r, torch.ones_like(r))
        out = r if out is None else out + r
    return out


# The press load selects near-singular arm BRANCHES (a locked/extended arm
# resists the normal force with ~zero joint torque, and k_ext turns pose noise
# into force noise, so the singular branch minimizes force-kernel income
# variance) and the corruption generalizes to FREE windows (user-observed
# 2026-08-15: post-force checkpoints hold weird right-arm poses at the bare
# nominal command; the resumed joint_vel_mpc_v1 does not). Deviation pulls
# cannot fix this — a pull cannot cross basins (the online-ik verdict) — so
# the posture is priced by a per-draw joint-space ANCHOR mimic: the pool's
# natural-branch generating posture, exposed as `anchor_q` and gliding with
# the pose transition. Exp+L2 pair per repo rule.


def _press_posture_scale(term, press_scale: float) -> torch.Tensor:
    """Per-hand posture-pricing gate: 1 on free/phantom windows, ``press_scale``
    while THAT hand's spring window is live. The weird-branch pose problem
    lives in the FREE windows (user-observed at the bare nominal command), so
    posture is priced fully there; a live press keeps only ``press_scale`` of
    it so force modulation (±f_cmd/k_ext along the normal) is never taxed into
    inaccuracy — the vel-command-scaled foot_clearance pattern."""
    live = (term.contact_mask > 0.5) & (term.phantom < 0.5)
    return torch.where(
        live,
        torch.full_like(term.contact_mask, press_scale),
        torch.ones_like(term.contact_mask),
    )


def arm_anchor_mimic_exp(
    env: ManagerBasedRlEnv,
    std: float,
    command_names: tuple[str, str],
    press_scale: float = 1.0,
) -> torch.Tensor:
    """exp(−Σ(q − anchor_q)² / std²) per hand, press-gated SUM over hands.
    In [0, 2]."""
    out = None
    for name in command_names:
        term = env.command_manager.get_term(name)
        q = term.robot.data.joint_pos[:, term.arm_joint_ids]
        r = torch.exp(-(q - term.anchor_q).pow(2).sum(-1) / std**2)
        r = r * _press_posture_scale(term, press_scale)
        out = r if out is None else out + r
    return out


def arm_anchor_mimic_l2(
    env: ManagerBasedRlEnv,
    command_names: tuple[str, str],
    press_scale: float = 1.0,
) -> torch.Tensor:
    """Σ(q − anchor_q)² per hand, press-gated SUM over hands [rad²]. Negative
    weight — the exp partner above is gradient-dead in a wrong basin; this
    term is the slope that pays the crossing."""
    out = None
    for name in command_names:
        term = env.command_manager.get_term(name)
        q = term.robot.data.joint_pos[:, term.arm_joint_ids]
        err = (q - term.anchor_q).pow(2).sum(-1)
        err = err * _press_posture_scale(term, press_scale)
        out = err if out is None else out + err
    return out


class HandForceRate:
    """Clamped per-step measured-force change, SUM over hands [N]. Negative
    weight — tactile-hardware protection (smooth force modulation)."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv) -> None:
        self._prev = torch.zeros(env.num_envs, 2, 3, device=env.device)

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        command_names: tuple[str, str],
        clamp: float = 20.0,
    ) -> torch.Tensor:
        f = torch.stack(
            [
                env.command_manager.get_term(n).measured_force_w()
                for n in command_names
            ],
            dim=1,
        )
        rate = (f - self._prev).norm(dim=-1).clamp(max=clamp).sum(-1)
        self._prev = f.detach()
        return rate

    def reset(self, env_ids: torch.Tensor) -> None:
        self._prev[env_ids] = 0.0


# ---------------------------------------------------------------------------
# Body-link tracking against the command term's `ref_body_* / robot_body_*`
# views (FootPlacementCommand and its play-only stand-ins).
# ---------------------------------------------------------------------------

class _body_selected:
    """Base for tracking terms over command-term links selected BY NAME
    (`body_names`, resolved against the command's `body_names` on first call)."""

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        self._names = tuple(cfg.params["body_names"])
        self._cols: torch.Tensor | None = None

    def cols(self, cmd) -> torch.Tensor:
        if self._cols is None:
            self._cols = cmd.body_cols(self._names)
        return self._cols

    @staticmethod
    def _reduce(err2: torch.Tensor, reduce: str) -> torch.Tensor:
        """Per-link squared errors [B, Nb] -> [B]: "sum" or "mean" over links."""
        if reduce == "sum":
            return err2.sum(dim=-1)
        if reduce == "mean":
            return err2.mean(dim=-1)
        raise ValueError(f"reduce must be 'sum' or 'mean', got {reduce!r}")


class track_body_pos(_body_selected):
    """exp(-Σ_b‖p_b - p_b,ref‖²/σ²) over the selected links (`reduce` = sum | mean), link origins in
    world frame [m]. Shape (num_envs,), in [0, 1]."""

    def __call__(self, env: ManagerBasedRlEnv, command_name: str, std: float,
                 body_names: tuple[str, ...], reduce: str = "sum") -> torch.Tensor:
        cmd = _cmd(env, command_name)
        c = self.cols(cmd)
        err2 = self._reduce(torch.square(
            cmd.ref_body_pos_w[:, c] - cmd.robot_body_pos_w[:, c]).sum(dim=-1), reduce)
        return torch.exp(-err2 / std**2)


class track_body_ori(_body_selected):
    """exp(-Σ_b θ_b²/σ²) (`reduce` = sum | mean over links), θ_b the rotation angle between the link's measured
    and reference orientation [rad]. Shape (num_envs,), in [0, 1]."""

    def __call__(self, env: ManagerBasedRlEnv, command_name: str, std: float,
                 body_names: tuple[str, ...], reduce: str = "sum") -> torch.Tensor:
        cmd = _cmd(env, command_name)
        c = self.cols(cmd)
        err2 = self._reduce(quat_error_magnitude(
            cmd.ref_body_quat_w[:, c], cmd.robot_body_quat_w[:, c]) ** 2, reduce)
        return torch.exp(-err2 / std**2)


class track_body_lin_vel(_body_selected):
    """exp(-Σ_b‖v_b - v_b,ref‖²/σ²) over the selected links (`reduce` = sum | mean), link-origin
    linear velocity in world frame [m/s]. Shape (num_envs,), in [0, 1]."""

    def __call__(self, env: ManagerBasedRlEnv, command_name: str, std: float,
                 body_names: tuple[str, ...], reduce: str = "sum") -> torch.Tensor:
        cmd = _cmd(env, command_name)
        c = self.cols(cmd)
        err2 = self._reduce(torch.square(
            cmd.ref_body_lin_vel_w[:, c] - cmd.robot_body_lin_vel_w[:, c]).sum(dim=-1), reduce)
        return torch.exp(-err2 / std**2)


class track_body_ang_vel(_body_selected):
    """exp(-Σ_b‖ω_b - ω_b,ref‖²/σ²) over the selected links (`reduce` = sum | mean), angular velocity
    in world frame [rad/s]. Shape (num_envs,), in [0, 1]."""

    def __call__(self, env: ManagerBasedRlEnv, command_name: str, std: float,
                 body_names: tuple[str, ...], reduce: str = "sum") -> torch.Tensor:
        cmd = _cmd(env, command_name)
        c = self.cols(cmd)
        err2 = self._reduce(torch.square(
            cmd.ref_body_ang_vel_w[:, c] - cmd.robot_body_ang_vel_w[:, c]).sum(dim=-1), reduce)
        return torch.exp(-err2 / std**2)


def stance_foot_still(env: ManagerBasedRlEnv, command_name: str, std: float
                      ) -> torch.Tensor:
    """exp(-(‖v‖² + ω_z²)/σ²) of the commanded STANCE foot: link-origin
    linear velocity [m/s] and yaw rate [rad/s], world frame. Shape (num_envs,),
    in [0, 1]."""
    cmd = _cmd(env, command_name)
    idx = torch.arange(env.num_envs, device=env.device)
    col = _foot_column(env, cmd, "stance")
    v = cmd.robot.data.body_link_lin_vel_w[:, cmd.feet_ids][idx, col]
    wz = cmd.robot.data.body_link_ang_vel_w[:, cmd.feet_ids][idx, col, 2]
    return torch.exp(-(torch.sum(torch.square(v), dim=-1) + wz ** 2) / std**2)
