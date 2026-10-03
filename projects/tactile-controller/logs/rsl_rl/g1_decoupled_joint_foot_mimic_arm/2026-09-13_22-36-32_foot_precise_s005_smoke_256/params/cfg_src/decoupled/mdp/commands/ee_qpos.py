"""ONE hand's tactile-tip pose command drawn from the ee_qpos dataset, with
the arm joint positions that realize it. Interface = `EeTargetCommand`'s
(`command`, `pose_command_b`, `anchor_quat`, `target_pose_w`,
`measured_ee_pose_w`, `body_idx`, `arm_joint_ids`), so the shared EE
observation and reward terms apply unchanged; adds `ref_joint_pos`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import Tensor

from mjlab.managers import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import (
    matrix_from_quat,
    quat_apply,
    quat_error_magnitude,
    quat_inv,
    quat_mul,
    yaw_quat,
)
import mujoco

if TYPE_CHECKING:
    from mjlab.entity import Entity
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.viewer.debug_visualizer import DebugVisualizer


@dataclass(frozen=True)
class EEQposeData:
    """One hand's samples, torch tensors on one device.

    Attributes:
        pos: Tip positions, yaw-aligned pelvis frame. Shape (N, 3), m.
        quat: Tip orientations, yaw-aligned pelvis frame, wxyz. Shape (N, 4).
        ref_joint_pos: Arm joint positions that put the tip at (`pos`, `quat`).
            Shape (N, 7), rad, columns in ascending mjlab joint order: aligned
            with `robot.data.joint_pos[:, joint_ids]`.
        joint_ids: mjlab joint indices of the columns, ascending. Shape (7,), long.
        joint_names: mjlab joint names of the columns.
    """

    pos: Tensor
    quat: Tensor
    ref_joint_pos: Tensor
    joint_ids: Tensor
    joint_names: tuple[str, ...]


def load_ee_qpose(file_path: str | Path, body_name: str, robot: Entity,
                  device: torch.device | str) -> EEQposeData:
    """The `body_name` hand's samples of an `ee_qpos.npz`
    (`<body_name>__pos/quat/anchor_q/joint_names`), with the `ref_joint_pos`
    columns reordered to `robot`'s ascending mjlab joint order.

    Raises:
        FileNotFoundError: The file is missing.
        KeyError: The file has no `body_name` hand.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found -- generate it with `uv run python -m data_gen.ee_qpos.generate`")
    with np.load(path, allow_pickle=False) as z:
        names = [str(j) for j in z[f"{body_name}__joint_names"]]
        ids = np.asarray(robot.find_joints(names, preserve_order=True)[0])  # mjlab ids, file column order
        order = np.argsort(ids)  # file columns -> ascending mjlab order
        return EEQposeData(
            pos=torch.as_tensor(z[f"{body_name}__pos"], dtype=torch.float32, device=device),
            quat=torch.as_tensor(z[f"{body_name}__quat"], dtype=torch.float32, device=device),
            ref_joint_pos=torch.as_tensor(z[f"{body_name}__anchor_q"][:, order], dtype=torch.float32, device=device),
            joint_ids=torch.as_tensor(ids[order], dtype=torch.long, device=device),
            joint_names=tuple(names[i] for i in order),
        )


def _transform(pose: Tensor, quat: Tensor, trans: Tensor | None = None) -> Tensor:
    """`pose` (B, 7) [pos, quat wxyz] moved by the transform (`trans` (B, 3),
    `quat` (B, 4)); the quaternion sign is normalized to w >= 0."""
    pos = quat_apply(quat, pose[:, :3])
    if trans is not None:
        pos = pos + trans
    q = quat_mul(quat, pose[:, 3:])
    q = torch.where(q[:, :1] < 0.0, -q, q)
    return torch.cat([pos, q], dim=-1)


class ArmChainFK:
    """Batched FK of the pelvis -> tip chain of one arm, read from the MJCF:
    the waist hinges (held at 0 unless given) then the 7 arm hinges, all at
    their bodies' origins. Poses come out in the PELVIS frame.

    Attributes:
        arm_joint_names: The arm joints in chain order (shoulder -> wrist).
        waist_joint_names: The waist joints in chain order.
    """

    @staticmethod
    def _body_id(mj_model: mujoco.MjModel, name: str) -> int:
        """Body id by name, accepting the scene's entity prefix (`<entity>/<name>`)."""
        for i in range(mj_model.nbody):
            n = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, i) or ""
            if n == name or n.endswith("/" + name):
                return i
        raise KeyError(f"body {name!r} not in the scene model")

    def __init__(self, mj_model: mujoco.MjModel, tip_body: str, device: torch.device | str) -> None:
        tip = self._body_id(mj_model, tip_body)
        pelvis = self._body_id(mj_model, "pelvis")
        chain, b = [], tip
        while b != pelvis:
            chain.append(b)
            b = mj_model.body_parentid[b]
        chain.reverse()
        pos, quat, axes, names = [], [], [], []
        for b in chain:
            pos.append(mj_model.body_pos[b]); quat.append(mj_model.body_quat[b])
            n = int(mj_model.body_jntnum[b])
            assert n <= 1, f"{tip_body} chain: body {b} has {n} joints"
            if n == 1:
                j = int(mj_model.body_jntadr[b])
                assert int(mj_model.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_HINGE)
                assert not mj_model.jnt_pos[j].any(), "hinge off the body origin is not handled"
                axes.append(mj_model.jnt_axis[j])
                names.append((mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, j) or "").split("/")[-1])
            else:
                axes.append(None); names.append(None)
        self._pos = torch.as_tensor(np.stack(pos), dtype=torch.float32, device=device)     # (L, 3)
        self._quat = torch.as_tensor(np.stack(quat), dtype=torch.float32, device=device)   # (L, 4)
        self._axis = [None if a is None else torch.as_tensor(a, dtype=torch.float32, device=device) for a in axes]
        joint_names = [n for n in names if n is not None]
        self.waist_joint_names = tuple(n for n in joint_names if "waist" in n)
        self.arm_joint_names = tuple(n for n in joint_names if "waist" not in n)
        assert len(self.arm_joint_names) == 7, self.arm_joint_names

    def tip_pose(self, q_arm: Tensor, q_waist: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Tip pose in the pelvis frame for `q_arm` (B, 7) in chain order and
        `q_waist` (B, 3) in chain order (zeros when None): pos (B, 3), quat
        wxyz (B, 4) with w >= 0."""
        B = q_arm.shape[0]
        q_all = torch.cat([torch.zeros(B, 3, device=q_arm.device) if q_waist is None else q_waist, q_arm], dim=-1)
        p = torch.zeros(B, 3, device=q_arm.device)
        r = torch.zeros(B, 4, device=q_arm.device); r[:, 0] = 1.0
        k = 0
        for i in range(self._pos.shape[0]):
            p = p + quat_apply(r, self._pos[i].expand(B, 3))
            r = quat_mul(r, self._quat[i].expand(B, 4))
            if self._axis[i] is not None:
                half = 0.5 * q_all[:, k]
                rj = torch.cat([torch.cos(half).unsqueeze(-1), torch.sin(half).unsqueeze(-1) * self._axis[i]], dim=-1)
                r = quat_mul(r, rj)
                k += 1
        r = torch.where(r[:, :1] < 0.0, -r, r)
        return p, r


class EEQposeCommand(CommandTerm):
    """Tip pose of ONE hand from the ee_qpos dataset, drawn uniformly at every
    resample and held in between, plus the arm joint positions that realize
    it. `command` (B, 7) = [pos (3), quat wxyz (4)] in the command frame:
    the yaw-aligned pelvis frame when `use_yaw_align_frame`, else the pelvis
    frame."""

    cfg: EEQposeCommandCfg

    def __init__(self, cfg: EEQposeCommandCfg, env: ManagerBasedRlEnv) -> None:
        super().__init__(cfg, env)
        self.robot: Entity = env.scene[cfg.entity_name]
        self.body_idx: int = self.robot.find_bodies(cfg.body_name)[0][0]
        self.data = load_ee_qpose(cfg.ee_qpos_path, cfg.body_name, self.robot, self.device)

        # the drawn sample, in the dataset's yaw-aligned pelvis frame
        self.pose_yaw = torch.zeros(self.num_envs, 7, device=self.device)
        self.pose_yaw[:, 3] = 1.0
        self.ref_joint_pos = torch.zeros(self.num_envs, 7, device=self.device)
        """(B, 7): arm joint positions that put the tip at the command, rad,
        columns aligned with `arm_joint_ids`."""

        self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["orientation_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["qpos_error"] = torch.zeros(self.num_envs, device=self.device)

        # transition governor (cfg.glide): joint-space min-jerk glide from the
        # previous reference to the new draw, the pose target = FK of the glide
        self._fk = ArmChainFK(env.sim.mj_model, cfg.body_name, self.device)
        self._chain_cols = torch.as_tensor(
            [self.data.joint_names.index(n) for n in self._fk.arm_joint_names], dtype=torch.long, device=self.device)
        self._q_start = torch.zeros(self.num_envs, 7, device=self.device)
        self._q_goal = torch.zeros(self.num_envs, 7, device=self.device)
        self._T = torch.zeros(self.num_envs, device=self.device)
        self._t = torch.zeros(self.num_envs, device=self.device)
        self._fresh = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

    @property
    def arm_joint_ids(self) -> Tensor:
        """mjlab joint ids aligned column-for-column with `ref_joint_pos`. Shape (7,), long."""
        return self.data.joint_ids

    def ref_tip_pose(self, q: Tensor) -> tuple[Tensor, Tensor]:
        """Tip pose of arm joint positions `q` (B, 7, `arm_joint_ids` order) in
        the dataset's frame (pelvis frame, waist at 0): pos (B, 3), quat wxyz (B, 4)."""
        return self._fk.tip_pose(q[:, self._chain_cols])

    def anchor_quat(self) -> Tensor:
        """World orientation of the command frame: yaw-only or full base. Shape (B, 4)."""
        root_q = self.robot.data.root_link_quat_w
        return yaw_quat(root_q) if self.cfg.use_yaw_align_frame else root_q

    @property
    def command(self) -> Tensor:
        """(B, 7): `pose_command_b`."""
        if self.cfg.use_yaw_align_frame:
            return self.pose_yaw
        root_q = self.robot.data.root_link_quat_w
        return _transform(self.pose_yaw, quat_mul(quat_inv(root_q), yaw_quat(root_q)))

    def target_pose_w(self) -> tuple[Tensor, Tensor]:
        """Live world target. pos (B, 3), quat wxyz (B, 4)."""
        pose_w = _transform(self.pose_yaw, yaw_quat(self.robot.data.root_link_quat_w),
                            self.robot.data.root_link_pos_w)
        return pose_w[:, :3], pose_w[:, 3:]

    def measured_ee_pose_w(self) -> tuple[Tensor, Tensor]:
        """Measured tip pose, world frame. pos (B, 3), quat wxyz (B, 4)."""
        return (
            self.robot.data.body_link_pos_w[:, self.body_idx],
            self.robot.data.body_link_quat_w[:, self.body_idx],
        )

    # -- sampling --------------------------------------------------------------

    def _resample_command(self, env_ids: Tensor) -> None:
        n = len(env_ids)
        idx = torch.randint(0, self.data.pos.shape[0], (n,), device=self.device)
        goal_pos, goal_quat, goal_q = self.data.pos[idx], self.data.quat[idx], self.data.ref_joint_pos[idx]
        if not self.cfg.glide:
            self.pose_yaw[env_ids, :3] = goal_pos
            self.pose_yaw[env_ids, 3:] = goal_quat
            self.ref_joint_pos[env_ids] = goal_q
            return
        # start = the previous reference; a fresh episode has none and jumps to its first draw
        # (the reset posture is outside the dataset region)
        fresh = self._fresh[env_ids]
        start_q = torch.where(fresh.unsqueeze(-1), goal_q, self.ref_joint_pos[env_ids])
        self._fresh[env_ids] = False
        start_pos, start_quat = self.ref_tip_pose(start_q)
        # duration from per-draw mean speed caps on the endpoint poses, clamped; jump_prob teleports
        v_lo, v_hi = self.cfg.transition_lin_speed
        w_lo, w_hi = self.cfg.transition_ang_speed
        u = torch.rand(n, 2, device=self.device)
        v = v_lo + (v_hi - v_lo) * u[:, 0]
        w = w_lo + (w_hi - w_lo) * u[:, 1]
        dist = (goal_pos - start_pos).norm(dim=-1)
        ang = quat_error_magnitude(start_quat, goal_quat)
        T = torch.maximum(dist / v, ang / w).clamp(*self.cfg.transition_time_range)
        jump = (torch.rand(n, device=self.device) < self.cfg.jump_prob) | fresh
        T = torch.where(jump, torch.zeros_like(T), T)
        self._q_start[env_ids] = start_q
        self._q_goal[env_ids] = goal_q
        self._T[env_ids] = T
        self._t[env_ids] = 0.0
        # jumps land immediately; glides start at the previous reference
        self.ref_joint_pos[env_ids] = torch.where(jump.unsqueeze(-1), goal_q, start_q)
        self.pose_yaw[env_ids, :3] = torch.where(jump.unsqueeze(-1), goal_pos, start_pos)
        self.pose_yaw[env_ids, 3:] = torch.where(jump.unsqueeze(-1), goal_quat, start_quat)

    def _update_command(self) -> None:
        if not self.cfg.glide:
            return  # held until the next resample
        self._t += self._env.step_dt
        s = (self._t / self._T.clamp_min(1e-6)).clamp(0.0, 1.0)
        s = s * s * s * (s * (6.0 * s - 15.0) + 10.0)   # min-jerk profile
        self.ref_joint_pos = self._q_start + (self._q_goal - self._q_start) * s.unsqueeze(-1)
        pos, quat = self.ref_tip_pose(self.ref_joint_pos)
        self.pose_yaw = torch.cat([pos, quat], dim=-1)

    def reset(self, env_ids: Tensor | None = None) -> dict[str, float]:
        assert isinstance(env_ids, Tensor)
        self._fresh[env_ids] = True
        return super().reset(env_ids)

    # -- metrics / vis ---------------------------------------------------------

    def _update_metrics(self) -> None:
        tgt_pos_w, tgt_quat_w = self.target_pose_w()
        ee_pos_w, ee_quat_w = self.measured_ee_pose_w()
        self.metrics["position_error"] = (tgt_pos_w - ee_pos_w).norm(dim=-1)
        self.metrics["orientation_error"] = quat_error_magnitude(tgt_quat_w, ee_quat_w)
        self.metrics["qpos_error"] = (
            self.robot.data.joint_pos[:, self.data.joint_ids] - self.ref_joint_pos).norm(dim=-1)

    def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
        env_indices = visualizer.get_env_indices(self.num_envs)
        if not env_indices:
            return
        tgt_pos_w, tgt_quat_w = self.target_pose_w()
        pos_np = tgt_pos_w.cpu().numpy()
        rot_np = matrix_from_quat(tgt_quat_w).cpu().numpy()
        for b in env_indices:
            visualizer.add_frame(position=pos_np[b], rotation_matrix=rot_np[b], scale=0.15,
                                 axis_radius=0.008, label=f"ee_ref_{self.cfg.body_name}_{b}")


@dataclass(kw_only=True)
class EEQposeCommandCfg(CommandTermCfg):
    """Config for `EEQposeCommand`."""

    entity_name: str = "robot"
    """Robot name, e.g. `g1`."""

    body_name: str = "left_tactile"
    """End-effector body name; also the hand's key in the dataset file."""

    use_yaw_align_frame: bool = False
    """If True, the command frame is the yaw-aligned pelvis frame, else the pelvis frame."""

    ee_qpos_path: str = "data/ee_qpos_v1/ee_qpos.npz"
    """Path to the `ee_qpos.npz` dataset."""

    glide: bool = False
    """True: glide from the previous reference to each new draw in joint space
    (min-jerk), the pose target following by FK; False: every draw is a jump."""
    transition_lin_speed: tuple[float, float] = (0.15, 0.40)
    """Per-draw mean tip speed range of a glide, m/s; min-jerk peak = 1.875x."""
    transition_ang_speed: tuple[float, float] = (0.5, 1.2)
    """Per-draw mean tip angular speed range of a glide, rad/s."""
    transition_time_range: tuple[float, float] = (0.4, 2.5)
    """Clamp on the glide duration, s."""
    jump_prob: float = 0.1
    """Fraction of draws that teleport instead of gliding (glide=True only)."""

    def build(self, env: ManagerBasedRlEnv) -> EEQposeCommand:
        return EEQposeCommand(self, env)
