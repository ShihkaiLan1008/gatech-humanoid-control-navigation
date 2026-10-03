from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import torch
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import (
    axis_angle_from_quat,
    matrix_from_quat,
    quat_apply,
    quat_apply_inverse,
    quat_conjugate,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_from_matrix,
    quat_mul,
    quat_unique,
    yaw_quat,
)
from torch import Tensor

import assets

if TYPE_CHECKING:
    from mjlab.entity import Entity
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.viewer.debug_visualizer import DebugVisualizer

# Per-hand debug-vis colors: left blue, right orange.
_HAND_VIS = ((0, "L", (0.20, 0.45, 0.95, 0.9)), (1, "R", (0.95, 0.55, 0.10, 0.9)))
_FORCE_VIS_COLOR = (0.75, 0.10, 0.80, 0.9)
# Sample-space volume: faint cyan shell/lids, warm inner keep-out column.
_SPACE_SHELL_COLOR = (0.25, 0.65, 0.90, 0.08)
_SPACE_LID_COLOR = (0.25, 0.65, 0.90, 0.25)
_SPACE_KEEPOUT_COLOR = (0.90, 0.35, 0.25, 0.12)


def _hquat_apply(q: Tensor, v: Tensor) -> Tensor:
    """quat_apply over (B, 2, ·) per-hand tensors."""
    return quat_apply(q.reshape(-1, 4), v.reshape(-1, 3)).reshape(v.shape)


def _hquat_mul(q1: Tensor, q2: Tensor) -> Tensor:
    return quat_mul(q1.reshape(-1, 4), q2.reshape(-1, 4)).reshape(q1.shape)

class UniformEePoseCommand(CommandTerm):

    cfg: UniformEePoseCommandCfg

    def __init__(self, cfg: UniformEePoseCommandCfg, env: ManagerBasedRlEnv) -> None:
        super().__init__(cfg, env)
        self.robot: Entity = env.scene[cfg.entity_name]
        self.body_idx = self.robot.find_bodies(cfg.body_name)[0][0]
        self.pose_command_b = torch.zeros(self.num_envs, 7, device=self.device)
        self.pose_command_b[:, 3] = 1.0
        self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["orientation_error"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> Tensor:
        """(B, 7): [pos(3), quat(4, w>=0)] — held in the sample frame."""
        return self.pose_command_b

    def anchor_quat(self) -> Tensor:
        """World orientation of the sample frame: yaw-only or full base."""
        root_q = self.robot.data.root_link_quat_w
        return yaw_quat(root_q) if self.cfg.use_yaw_align_frame else root_q

    def target_pose_w(self) -> tuple[Tensor, Tensor]:
        """Live world target (rides the current anchor). pos (B,3), quat (B,4)."""
        root_pos = self.robot.data.root_link_pos_w
        aq = self.anchor_quat()
        pos_w = root_pos + quat_apply(aq, self.pose_command_b[:, :3])
        quat_w = quat_mul(aq, self.pose_command_b[:, 3:])
        return pos_w, quat_w

    def measured_ee_pose_w(self) -> tuple[Tensor, Tensor]:
        return (
            self.robot.data.body_link_pos_w[:, self.body_idx],
            self.robot.data.body_link_quat_w[:, self.body_idx],
        )

    def _resample_command(self, env_ids: Tensor) -> None:
        rng = self.cfg.ranges
        n = len(env_ids)
        r = torch.empty(n, device=self.device)
        self.pose_command_b[env_ids, 0] = r.uniform_(*rng.pos_x)
        self.pose_command_b[env_ids, 1] = r.uniform_(*rng.pos_y)
        self.pose_command_b[env_ids, 2] = r.uniform_(*rng.pos_z)
        euler = torch.zeros(n, 3, device=self.device)
        euler[:, 0].uniform_(*rng.roll)
        euler[:, 1].uniform_(*rng.pitch)
        euler[:, 2].uniform_(*rng.yaw)
        quat = quat_from_euler_xyz(euler[:, 0], euler[:, 1], euler[:, 2])
        self.pose_command_b[env_ids, 3:] = quat_unique(quat)

    def _update_command(self) -> None:
        pass

    def _update_metrics(self) -> None:
        tgt_pos_w, tgt_quat_w = self.target_pose_w()
        ee_pos_w, ee_quat_w = self.measured_ee_pose_w()
        self.metrics["position_error"] = (tgt_pos_w - ee_pos_w).norm(dim=-1)
        self.metrics["orientation_error"] = quat_error_magnitude(tgt_quat_w, ee_quat_w)

    def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
        env_indices = visualizer.get_env_indices(self.num_envs)
        if not env_indices:
            return
        tgt_pos_w, tgt_quat_w = self.target_pose_w()
        pos_np = tgt_pos_w.cpu().numpy()
        rot_np = matrix_from_quat(tgt_quat_w).cpu().numpy()
        tag = self.cfg.body_name
        for b in env_indices:
            visualizer.add_frame(
                position=pos_np[b],
                rotation_matrix=rot_np[b],
                scale=0.15,
                axis_radius=0.008,
                label=f"ee_goal_{tag}_{b}",
            )


class ArmChain(NamedTuple):
    """Constants of a fused n-R serial arm chain (see :func:`build_arm_chain`)."""

    joint_names: tuple[str, ...]
    T_off: Tensor  # (n, 4, 4) fixed parent offsets, leading fixed frames folded
    T_post: Tensor  # (4, 4) trailing fixed transform (tool/tip frame)
    axes: Tensor  # (n, 3)
    K: Tensor  # (n, 3, 3) axis skew matrices
    K2: Tensor  # (n, 3, 3) K @ K
    q_lo: Tensor  # (n,)
    q_hi: Tensor  # (n,)
    root_t: Tensor  # (3,) base_link -> chain-root translation at q = 0
    root_R: Tensor  # (3, 3) base_link -> chain-root rotation at q = 0


def build_arm_chain(
    urdf_text: str,
    body_name: str,
    root_link: str,
    device: torch.device,
    base_link: str = "pelvis",
) -> ArmChain:
    """Parse the URDF ONCE (pytorch-kinematics) into fused-chain constants.

    pk is used here only: its runtime FK/jacobian are per-joint python loops —
    hundreds of micro kernels per call, launch-bound in the env loop. The
    fused path (:func:`fused_fk_jac`) is ~20 kernels and is cross-validated
    against pk before returning. Also captures the FIXED base_link->root_link
    transform at the zero configuration (the command frame is the nominal
    gravity-upright pelvis frame)."""
    import pytorch_kinematics as pk

    chain = pk.build_chain_from_urdf(urdf_text)
    sc = pk.SerialChain(chain, body_name, root_link).to(
        device=device, dtype=torch.float32
    )
    joint_names = tuple(sc.get_joint_parameter_names())
    lo, hi = sc.get_joint_limits()
    q_lo = torch.tensor(lo, device=device, dtype=torch.float32)
    q_hi = torch.tensor(hi, device=device, dtype=torch.float32)

    # Fixed frames accumulate into `pending`, which folds into the NEXT
    # revolute joint's parent offset (T_off_i sits BEFORE R_i, so
    # pre-composition is exact). Fixed frames AFTER the last revolute
    # (e.g. the tactile tip on wrist_yaw) must instead POST-multiply the
    # whole chain (T_post): folding them back into T_off_n would put the
    # tip translation before R_n — off by exactly the tip offset the
    # moment the wrist yaws (caught by the init assert on the tactile
    # chain switch, 2026-07-29).
    eye = torch.eye(4, device=device)
    T_offs, axes = [], []
    pending = eye.clone()
    for f in sc._serial_frames:
        j = f.joint
        off = (
            j.offset.get_matrix()[0].to(device, torch.float32)
            if j.offset is not None
            else eye
        )
        if j.joint_type == "fixed":
            pending = pending @ off
            continue
        assert j.joint_type == "revolute", j.joint_type
        T_offs.append(pending @ off)
        axes.append(j.axis.to(device).float())
        pending = eye.clone()
    a = torch.stack(axes)
    K = torch.zeros(len(T_offs), 3, 3, device=device)
    K[:, 0, 1], K[:, 0, 2] = -a[:, 2], a[:, 1]
    K[:, 1, 0], K[:, 1, 2] = a[:, 2], -a[:, 0]
    K[:, 2, 0], K[:, 2, 1] = -a[:, 1], a[:, 0]

    base_sc = pk.SerialChain(chain, root_link, base_link).to(
        device=device, dtype=torch.float32
    )
    n_base = len(base_sc.get_joint_parameter_names())
    M0 = base_sc.forward_kinematics(
        torch.zeros(1, n_base, device=device)
    ).get_matrix()

    consts = ArmChain(
        joint_names=joint_names,
        T_off=torch.stack(T_offs),
        T_post=pending,
        axes=a,
        K=K,
        K2=K @ K,
        q_lo=q_lo,
        q_hi=q_hi,
        root_t=M0[0, :3, 3],
        root_R=M0[0, :3, :3],
    )

    # Cross-validate the fused FK/jacobian against pk once. Tolerance is
    # float32 rounding, which scales with chain length: the wrist chain sat
    # at ~1e-5; the tactile TIP chain's extra 13 cm lever arm amplifies it to
    # ~5e-4 (0.5 mm — noise vs commanded-pose scales). Seeded draws keep the
    # check deterministic across builds.
    gen = torch.Generator(device=device).manual_seed(0)
    q_test = (
        torch.rand(64, len(T_offs), device=device, generator=gen) - 0.5
    ) * 1.5
    T_f, J_f = fused_fk_jac(consts, q_test)
    err_T = (T_f - sc.forward_kinematics(q_test).get_matrix()).abs().max()
    err_J = (J_f - sc.jacobian(q_test)).abs().max()
    assert err_T < 2e-3 and err_J < 2e-3, (err_T, err_J)
    return consts


def fused_fk_jac(chain: ArmChain, q: Tensor) -> tuple[Tensor, Tensor]:
    """Fused FK + geometric jacobian for the n-R serial chain.

    Rotations for all joints are built in one batched Rodrigues op; the
    chain is a short sequence of (B,4,4) matmuls. Joint i's world origin
    equals prefix P_i's origin (rotation adds no translation) and its
    world axis is invariant under its own rotation, so the prefixes give
    the jacobian directly. Validated against pk at build."""
    B, n = q.shape
    s = torch.sin(q)[:, :, None, None]
    c = torch.cos(q)[:, :, None, None]
    R = torch.eye(3, device=q.device) + s * chain.K + (1.0 - c) * chain.K2
    A = chain.T_off.expand(B, n, 4, 4).clone()
    A[:, :, :3, :3] = chain.T_off[None, :, :3, :3] @ R
    P = torch.eye(4, device=q.device).expand(B, 4, 4)
    origins, waxes = [], []
    for i in range(n):
        P = P @ A[:, i]
        origins.append(P[:, :3, 3])
        waxes.append(P[:, :3, :3] @ chain.axes[i])
    # Tool/tip frame: trailing fixed transform after the last joint. The
    # position jacobian must use the TIP point (it orbits every joint
    # axis with the tip's lever arm); the rotation jacobian is unchanged
    # (a fixed post-rotation adds no angular velocity).
    P = P @ chain.T_post
    p_ee = P[:, :3, 3]
    w = torch.stack(waxes, dim=-1)  # (B, 3, n)
    o = torch.stack(origins, dim=-1)
    Jp = torch.cross(w, p_ee.unsqueeze(-1) - o, dim=1)
    return P, torch.cat([Jp, w], dim=1)


def fused_ik_solve(
    chain: ArmChain,
    p_goal: Tensor,
    R_goal: Tensor,
    q0: Tensor,
    q_pull: Tensor,
    pos_weight: float = 1.0,
    ori_weight: float = 0.5,
    damping: float = 0.05,
    posture_gain: float = 0.1,
    max_dq: float = 0.3,
    iters: int = 300,
) -> tuple[Tensor, Tensor]:
    """Batch-solve the chain IK to convergence (build-time use, no shaping).

    Same math as the runtime solver in ``UniformEePoseIkCommand``: damped
    normal equations, posture pull strictly in the task null space (tracked
    by a warm-started inverse power iteration). Goals are in the CHAIN ROOT
    frame. Returns (q, final position-error norm)."""
    q = q0.clone()
    null_v = torch.full_like(q, q.shape[1] ** -0.5)
    wp2, wo2 = pos_weight**2, ori_weight**2
    for _ in range(iters):
        M, J = fused_fk_jac(chain, q)
        e_p = p_goal - M[:, :3, 3]
        e_r = axis_angle_from_quat(
            quat_from_matrix(R_goal @ M[:, :3, :3].transpose(-1, -2))
        )
        Jp, Jr = J[:, :3], J[:, 3:]
        JTJ = wp2 * Jp.transpose(-1, -2) @ Jp + wo2 * Jr.transpose(-1, -2) @ Jr
        JTe = wp2 * (Jp.transpose(-1, -2) @ e_p.unsqueeze(-1)).squeeze(-1) + wo2 * (
            Jr.transpose(-1, -2) @ e_r.unsqueeze(-1)
        ).squeeze(-1)
        JTJ.diagonal(dim1=-2, dim2=-1).add_(damping**2)
        sol = torch.linalg.solve(JTJ, torch.stack([JTe, null_v], dim=-1))
        dq, v = sol[..., 0], sol[..., 1]
        null_v = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        dq = dq + posture_gain * null_v * ((q_pull - q) * null_v).sum(-1, keepdim=True)
        q = (q + dq.clamp(-max_dq, max_dq)).clamp(chain.q_lo, chain.q_hi)
    M, _ = fused_fk_jac(chain, q)
    return q, (p_goal - M[:, :3, 3]).norm(dim=-1)


class UniformEePoseIkCommand(UniformEePoseCommand):

    cfg: UniformEePoseIkCommandCfg

    def __init__(self, cfg: UniformEePoseIkCommandCfg, env: ManagerBasedRlEnv) -> None:
        super().__init__(cfg, env)
        # URDF parsing, fused-chain constants, pk cross-validation and the
        # fixed pelvis→ik_root transform all live in build_arm_chain (shared
        # with the build-time anchor solves in reach_pool).
        self._chain = build_arm_chain(
            Path(cfg.urdf_path).read_text(),
            cfg.body_name,
            cfg.ik_root_link,
            self.device,
        )
        self.arm_joint_ids, _ = self.robot.find_joints(
            list(self._chain.joint_names), preserve_order=True
        )
        self._n_dof = len(self._chain.joint_names)
        self._q_lo, self._q_hi = self._chain.q_lo, self._chain.q_hi
        self._root_t, self._root_R = self._chain.root_t, self._chain.root_R

        # Default arms (natural elbows) = reset state and the DEFAULT posture
        # pull target. `_q_post` is the LIVE pull target: per-draw anchors
        # (seed_anchor) re-aim it; reset restores the default.
        self._q_default = self.robot.data.default_joint_pos[
            :, self.arm_joint_ids
        ].clone()
        self._q_post = self._q_default.clone()
        # Solver state (`_q_ik`) and the emitted, trapezoid-shaped reference
        # (`ik_joint_ref`) are SEPARATE: anchors teleport the former, the
        # latter glides over (see _update_command).
        self._q_ik = self._q_default.clone()
        self.ik_joint_ref = self._q_default.clone()
        # Warm-started inverse-power-iteration state for the task null
        # direction (any generic unit vector converges in ~2 steps), and the
        # previous emitted step for the accel clamp.
        self._null_init = torch.full(
            (self._n_dof,), self._n_dof**-0.5, device=self.device
        )
        self._null_v = self._null_init.expand(self.num_envs, -1).clone()
        self._dq_prev = torch.zeros(self.num_envs, self._n_dof, device=self.device)
        self.metrics["ik_pos_err"] = torch.zeros(self.num_envs, device=self.device)

    def reset(self, env_ids) -> dict[str, float]:
        self._q_post[env_ids] = self._q_default[env_ids]
        self._q_ik[env_ids] = self._q_default[env_ids]
        self.ik_joint_ref[env_ids] = self._q_default[env_ids]
        self._null_v[env_ids] = self._null_init
        self._dq_prev[env_ids] = 0.0
        return super().reset(env_ids)

    def seed_anchor(self, env_ids: Tensor, q_anchor: Tensor) -> None:
        self._q_ik[env_ids] = q_anchor
        self._q_post[env_ids] = q_anchor

    def _fk_jac(self, q: Tensor) -> tuple[Tensor, Tensor]:
        return fused_fk_jac(self._chain, q)

    def _update_command(self) -> None:
        cfg = self.cfg
        cmd = self.pose_command_b
        R_cmd = matrix_from_quat(cmd[:, 3:])
        p_goal = (cmd[:, :3] - self._root_t) @ self._root_R
        R_goal = self._root_R.transpose(0, 1) @ R_cmd

        q = self._q_ik
        M, J = self._fk_jac(q)
        e_p = p_goal - M[:, :3, 3]
        R_err = R_goal @ M[:, :3, :3].transpose(-1, -2)
        e_r = axis_angle_from_quat(quat_from_matrix(R_err))
        Jp, Jr = J[:, :3], J[:, 3:]
        wp2, wo2 = cfg.ik_pos_weight**2, cfg.ik_ori_weight**2
        JTJ = wp2 * Jp.transpose(-1, -2) @ Jp + wo2 * Jr.transpose(-1, -2) @ Jr
        JTe = wp2 * (Jp.transpose(-1, -2) @ e_p.unsqueeze(-1)).squeeze(-1) + wo2 * (
            Jr.transpose(-1, -2) @ e_r.unsqueeze(-1)
        ).squeeze(-1)
        JTJ.diagonal(dim1=-2, dim2=-1).add_(cfg.ik_damping**2)

        rhs = torch.stack([JTe, self._null_v], dim=-1)
        sol = torch.linalg.solve(JTJ, rhs)
        dq = sol[..., 0]
        v = sol[..., 1]
        v = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        self._null_v = v
        dq = dq + cfg.ik_posture_gain * v * ((self._q_post - q) * v).sum(
            -1, keepdim=True
        )
        self._q_ik = (q + dq.clamp(-cfg.ik_max_dq, cfg.ik_max_dq)).clamp(
            self._q_lo, self._q_hi
        )

        d = self._q_ik - self.ik_joint_ref
        v_cap = torch.minimum(
            torch.full_like(d, cfg.ik_max_dq),
            (2.0 * cfg.ik_max_ddq * d.abs()).sqrt(),
        )
        step = d.clamp(-v_cap, v_cap).clamp(
            self._dq_prev - cfg.ik_max_ddq, self._dq_prev + cfg.ik_max_ddq
        )
        ref = (self.ik_joint_ref + step).clamp(self._q_lo, self._q_hi)
        self._dq_prev = ref - self.ik_joint_ref
        self.ik_joint_ref = ref
        self.metrics["ik_pos_err"] = e_p.norm(dim=-1)


@dataclass(kw_only=True)
class UniformEePoseCommandCfg(CommandTermCfg):
    """Config for :class:`UniformEePoseCommand` (per-hand box)."""

    @dataclass
    class Ranges:
        pos_x: tuple[float, float]
        pos_y: tuple[float, float]
        pos_z: tuple[float, float]
        roll: tuple[float, float]
        pitch: tuple[float, float]
        yaw: tuple[float, float]

    entity_name: str = "robot"
    body_name: str = "left_wrist_yaw_link"
    ranges: Ranges = None  # type: ignore[assignment]  # required, set in cfgs
    use_yaw_align_frame: bool = False

    def build(self, env: ManagerBasedRlEnv) -> UniformEePoseCommand:
        return UniformEePoseCommand(self, env)



_G1_URDF = str(assets.ASSET_DIR / assets.TACTILE_URDF)


@dataclass(kw_only=True)
class UniformEePoseIkCommandCfg(UniformEePoseCommandCfg):
    """Config for :class:`UniformEePoseIkCommand` (adds the arm-joint IK ref)."""

    urdf_path: str = _G1_URDF
    ik_root_link: str = "torso_link"
    """IK chain root — torso keeps waist/legs OUT of the chain: the reference
    can only ever ask for arm motion."""
    ik_pos_weight: float = 1.0
    ik_ori_weight: float = 0.5
    """Position prioritized over orientation when the target is not exactly
    reachable (best-effort tradeoff of the reference)."""
    ik_damping: float = 0.05
    ik_posture_gain: float = 0.1
    """Per-step pull rate toward the default arm pose (fraction of the
    remaining posture error per step), applied STRICTLY inside the task null
    space — it steers the elbow swivel but never trades pose accuracy (the
    old `ik_posture_weight` penalty did, parking references 13 mm short p50
    on exactly-reachable pool targets)."""
    ik_max_dq: float = 0.3
    """Per-step reference rate limit [rad] — one DLS iteration runs per env
    step (amortized solve), so this also smooths reference transitions."""
    ik_max_ddq: float = 0.02
    """Per-step reference rate CHANGE limit [rad/step²] (~50 rad/s² at the
    50 Hz env step). With the stopping-distance velocity cap this shapes each
    resample transient into a trapezoid instead of a velocity step."""

    def build(self, env: ManagerBasedRlEnv) -> UniformEePoseIkCommand:
        return UniformEePoseIkCommand(self, env)
