"""H-LIP reference term for the velocity line (robot_rl ``HLIPCommandTerm``).

Every tick it builds the tracked output vector ``y_out`` / ``dy_out`` and the
measured ``y_act`` / ``dy_act`` in the stance-site yaw frame that ``foot_plan``
latched at the step event, and evaluates the CLF on them.  Layout of ``y``
(15): com (x, y, z) | pelvis (roll, pitch, yaw) | swing site (x, y, z) |
swing site (roll, pitch, yaw) | waist joints (yaw, roll, pitch).  Yaws are
relative to the latched stance yaw.  ``dy`` pairs each entry with its rate:
CoM velocity (stance frame), pelvis angular velocity (body frame), swing site
linear velocity (stance frame), swing ankle angular velocity (ankle frame),
waist joint velocities.  Declare this term AFTER ``foot_plan``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np
import torch

from mjlab.managers import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import quat_apply_inverse

from rl.decoupled.mdp.clf import CLF
from rl.decoupled.mdp.commands.foot_placement import _wrap
from rl.decoupled.mdp.commands.hlip import HLIPCommandTerm, _rot2

if TYPE_CHECKING:
    from mjlab.entity import Entity
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.viewer.debug_visualizer import DebugVisualizer

_G = 9.81
_SWING_REF_VIS = (0.9, 0.2, 0.9, 1.0)

Y_NAMES: tuple[str, ...] = (
    "com:x", "com:y", "com:z",
    "pelvis:roll", "pelvis:pitch", "pelvis:yaw",
    "swing:x", "swing:y", "swing:z",
    "swing:roll", "swing:pitch", "swing:yaw",
    "joint:waist_yaw", "joint:waist_roll", "joint:waist_pitch",
)
"""Channel names of ``y`` / ``dy`` in layout order (CLF weight keys)."""

_H_CTRL = (0.0, 0.0, 1.0, 1.0, 1.0)
"""Horizontal swing Bezier control points on the step progress, degree 4."""


def _euler_xyz_from_quat_wxyz(q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(roll, pitch, yaw) of wxyz quaternions (..., 4), rad."""
    w, x, y, z = q.unbind(-1)
    roll = torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = torch.asin((2.0 * (w * y - z * x)).clamp(-1.0, 1.0))
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw


def _euler_rates_to_omega(eul: torch.Tensor, rates: torch.Tensor) -> torch.Tensor:
    """Body angular velocity (..., 3) from Z-Y-X Euler angles and their rates (..., 3)."""
    _, theta, psi = eul.unbind(-1)
    c_th, s_th, c_ps, s_ps = torch.cos(theta), torch.sin(theta), torch.cos(psi), torch.sin(psi)
    zeros, ones = torch.zeros_like(theta), torch.ones_like(theta)
    m = torch.stack([
        torch.stack([c_th * c_ps, s_ps, zeros], dim=-1),
        torch.stack([-c_th * s_ps, c_ps, zeros], dim=-1),
        torch.stack([s_th, zeros, ones], dim=-1),
    ], dim=-2)
    return torch.einsum("...ij,...j->...i", m, rates)


def _bezier(ctrl: torch.Tensor, tau: torch.Tensor, order: int) -> torch.Tensor:
    """Bernstein curve (order 0) or its d/dtau (order 1) at ``tau`` in [0, 1].
    ``ctrl`` (B, n+1) -> (B,)."""
    tau = tau.clamp(0.0, 1.0).unsqueeze(-1)
    if order == 1:
        n = ctrl.shape[1] - 1
        ctrl = n * (ctrl[:, 1:] - ctrl[:, :-1])
    n = ctrl.shape[1] - 1
    i = torch.arange(n + 1, device=ctrl.device, dtype=ctrl.dtype)
    coef = torch.tensor([math.comb(n, k) for k in range(n + 1)], device=ctrl.device, dtype=ctrl.dtype)
    return (ctrl * coef * (1.0 - tau) ** (n - i) * tau ** i).sum(dim=1)


class HlipReferenceCommand(CommandTerm):
    """robot_rl-style H-LIP reference + CLF over the 15 ``Y_NAMES`` channels.

    Reads ``foot_plan`` (stance anchor, target, orbit CoM, CoM state),
    ``gait`` (phase, period, moving) and ``twist``.  ``command`` = ``y_out``
    [B, 15].  ``v`` / ``vdot`` are the CLF value and its finite-difference
    rate, recomputed once per tick.
    """

    cfg: HlipReferenceCommandCfg
    _env: ManagerBasedRlEnv

    def __init__(self, cfg: HlipReferenceCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self.robot: Entity = env.scene[cfg.entity_name]
        dev = self.device
        self._feet_ids = torch.tensor(
            self.robot.find_bodies(list(cfg.foot_body_names), preserve_order=True)[0],
            dtype=torch.long, device=dev)
        self._site_ids = torch.tensor(
            self.robot.find_sites(list(cfg.site_names), preserve_order=True)[0],
            dtype=torch.long, device=dev)
        self._waist_ids = torch.tensor(
            self.robot.find_joints(list(cfg.waist_joint_names), preserve_order=True)[0],
            dtype=torch.long, device=dev)
        B, n = self.num_envs, len(Y_NAMES)
        self.y_out = torch.zeros(B, n, device=dev)
        self.dy_out = torch.zeros(B, n, device=dev)
        self.y_act = torch.zeros(B, n, device=dev)
        self.dy_act = torch.zeros(B, n, device=dev)
        self.v = torch.zeros(B, device=dev)
        self.vdot = torch.zeros(B, device=dev)
        self._prev_swing = torch.full((B,), -1, dtype=torch.long, device=dev)
        self._prev_standing = torch.ones(B, dtype=torch.bool, device=dev)
        self._swing_start = torch.zeros(B, 2, device=dev)
        self._swing_ref_w = torch.zeros(B, 3, device=dev)
        self._h_ctrl = torch.tensor(_H_CTRL, device=dev).expand(B, -1)
        z = cfg.z_sw_max
        self._z_ctrl = torch.tensor(
            [0.0, 0.2 * z, 0.6 * z, z, 0.5 * z, 0.05 * z, 0.0], device=dev).expand(B, -1)
        self.clf = CLF(
            sim_dt=env.step_dt, batch_size=B, pos_output_names=Y_NAMES,
            vel_output_names=Y_NAMES, Q_weights=cfg.clf["Q_weights"],
            R_weights=cfg.clf["R_weights"], device=dev)
        for k in ("error_com_x", "error_com_y", "error_com_z", "error_sw_x", "error_sw_y",
                  "error_sw_z", "error_pelvis_roll", "error_pelvis_pitch", "error_pelvis_yaw",
                  "v", "vdot"):
            self.metrics[k] = torch.zeros(B, device=dev)

    # ------------------------------------------------------------ views
    @property
    def command(self) -> torch.Tensor:
        """The reference ``y_out``. Shape (B, 15)."""
        return self.y_out

    @property
    def planner(self) -> HLIPCommandTerm:
        """The ``foot_plan`` term this reference is built on."""
        return self._env.command_manager.get_term(self.cfg.planner_name)

    # ------------------------------------------------------------ lifecycle
    def _resample_command(self, env_ids: torch.Tensor) -> None:
        self.clf.invalidate(env_ids)
        self._prev_swing[env_ids] = -1
        self._prev_standing[env_ids] = True

    def _update_command(self) -> None:
        cfg = self.cfg
        P = self.planner
        gait = self._env.command_manager.get_term(cfg.gait_name)
        tw = self._env.command_manager.get_command(cfg.twist_name)
        d = self.robot.data
        B = self.num_envs
        rows = torch.arange(B, device=self.device)
        swing, standing = P.swing, P.standing
        stance = 1 - swing
        moving = (~standing).float()
        p_st = P.ref_feet_pos_w[rows, stance]
        psi = P.ref_feet_yaw[rows, stance]

        def to_stance(v_w: torch.Tensor) -> torch.Tensor:
            return torch.cat([_rot2(-psi, v_w[:, :2]), v_w[:, 2:3]], dim=-1)

        site_p = d.site_pos_w[:, self._site_ids]
        sw_act = to_stance(site_p[rows, swing] - p_st)

        event = (swing != self._prev_swing) | (standing != self._prev_standing)
        if event.any():
            ids = torch.where(event)[0]
            self._swing_start[ids] = sw_act[ids, :2]
            if cfg.invalidate_on_event:
                self.clf.invalidate(ids)
        self._prev_swing = swing.clone()
        self._prev_standing = standing.clone()

        T = 0.5 * gait.period
        tau = (gait.phase % math.pi) / math.pi
        t = tau * T
        tp = gait.phase / (2.0 * math.pi)
        s2, c2 = torch.sin(2.0 * math.pi * tp), torch.cos(2.0 * math.pi * tp)
        dtp = moving / (2.0 * T)
        vx, vy, wz = tw[:, 0], tw[:, 1], tw[:, 2] * moving

        # --- CoM (orbit flow vs measured), stance frame
        com_ref = to_stance(P.ref_com_pos_w - p_st)
        if cfg.com_z_ref is not None:
            com_ref = torch.cat([com_ref[:, :2], torch.full_like(com_ref[:, 2:3], cfg.com_z_ref)], dim=-1)
        com_ref_v = to_stance(P.ref_com_vel_w)
        com_act = to_stance(P.com_pos_w - p_st)
        com_act_v = to_stance(P.com_vel_w)

        # --- pelvis orientation (Euler, yaw relative to the stance yaw)
        roll, pitch, yaw = _euler_xyz_from_quat_wxyz(d.root_link_quat_w)
        pelvis_act = torch.stack([roll, pitch, _wrap(yaw - psi)], dim=-1)
        roll_ref = (cfg.roll_asym_amp * s2
                    + torch.atan(vy / _G).clamp(-cfg.roll_bias_lat_max, cfg.roll_bias_lat_max)
                    + torch.atan(vx * wz / _G).clamp(-cfg.roll_bias_yaw_max, cfg.roll_bias_yaw_max))
        pitch_ref = cfg.pitch_ref + cfg.pitch_amp * s2
        yaw_ref = wz * t
        eul_ref = torch.stack([roll_ref, pitch_ref, yaw_ref], dim=-1)
        rates_ref = torch.stack([cfg.roll_asym_amp * 2.0 * math.pi * c2 * dtp,
                                 cfg.pitch_amp * 2.0 * math.pi * c2 * dtp, wz], dim=-1)
        omega_ref = _euler_rates_to_omega(eul_ref, rates_ref)
        omega_act = d.root_link_ang_vel_b

        # --- swing site: Bezier from the step start to the committed target
        tgt = to_stance(P.target_pos_w - p_st)
        s = torch.where(swing == 0, 1.0, -1.0)
        if cfg.swing_start == "nominal":
            x0, y0 = -tgt[:, 0], s * cfg.stance_width
        else:
            x0, y0 = self._swing_start[:, 0], self._swing_start[:, 1]
        bht = _bezier(self._h_ctrl, tau, 0)
        dbht = _bezier(self._h_ctrl, tau, 1) / T
        sw_ref = torch.stack([(1.0 - bht) * x0 + bht * tgt[:, 0],
                              (1.0 - bht) * y0 + bht * tgt[:, 1],
                              _bezier(self._z_ctrl, tau, 0)], dim=-1)
        sw_ref_v = torch.stack([dbht * (tgt[:, 0] - x0), dbht * (tgt[:, 1] - y0),
                                _bezier(self._z_ctrl, tau, 1) / T], dim=-1)
        hold = standing.unsqueeze(-1)
        sw_ref = torch.where(hold, tgt, sw_ref)
        sw_ref_v = torch.where(hold, torch.zeros_like(sw_ref_v), sw_ref_v)
        link_p = d.body_link_pos_w[:, self._feet_ids][rows, swing]
        link_v = d.body_link_lin_vel_w[:, self._feet_ids][rows, swing]
        link_w = d.body_link_ang_vel_w[:, self._feet_ids][rows, swing]
        site_v = link_v + torch.cross(link_w, site_p[rows, swing] - link_p, dim=-1)
        sw_act_v = to_stance(site_v)

        # --- swing ankle orientation
        qf = d.body_link_quat_w[:, self._feet_ids][rows, swing]
        f_roll, f_pitch, f_yaw = _euler_xyz_from_quat_wxyz(qf)
        fo_act = torch.stack([f_roll, f_pitch, _wrap(f_yaw - psi)], dim=-1)
        fo_ref = torch.stack([torch.zeros_like(yaw_ref), torch.zeros_like(yaw_ref), yaw_ref], dim=-1)
        fo_act_w = quat_apply_inverse(qf, link_w)
        fo_ref_w = torch.stack([torch.zeros_like(wz), torch.zeros_like(wz), wz], dim=-1)

        # --- waist joints
        wj = d.joint_pos[:, self._waist_ids]
        wj_ref = d.default_joint_pos[:, self._waist_ids]
        wjv = d.joint_vel[:, self._waist_ids]

        self.y_out = torch.cat([com_ref, eul_ref, sw_ref, fo_ref, wj_ref], dim=-1)
        self.dy_out = torch.cat([com_ref_v, omega_ref, sw_ref_v, fo_ref_w, torch.zeros_like(wjv)], dim=-1)
        self.y_act = torch.cat([com_act, pelvis_act, sw_act, fo_act, wj], dim=-1)
        self.dy_act = torch.cat([com_act_v, omega_act, sw_act_v, fo_act_w, wjv], dim=-1)
        self.v, self.vdot = self.clf.compute(self.y_act, self.y_out, self.dy_act, self.dy_out)
        self._swing_ref_w = torch.cat([p_st[:, :2] + _rot2(psi, sw_ref[:, :2]), sw_ref[:, 2:3]], dim=-1)

    def _update_metrics(self) -> None:
        err = (self.y_act - self.y_out).abs()
        for i, k in ((0, "error_com_x"), (1, "error_com_y"), (2, "error_com_z"),
                     (3, "error_pelvis_roll"), (4, "error_pelvis_pitch"), (5, "error_pelvis_yaw"),
                     (6, "error_sw_x"), (7, "error_sw_y"), (8, "error_sw_z")):
            self.metrics[k] = err[:, i]
        self.metrics["v"] = self.v
        self.metrics["vdot"] = self.vdot

    # ------------------------------------------------------------- debug vis
    def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
        """MAGENTA sphere = the swing-site reference point on its Bezier arc."""
        env_indices = visualizer.get_env_indices(self.num_envs)
        if not env_indices:
            return
        ref = self._swing_ref_w.cpu().numpy()
        for b in env_indices:
            visualizer.add_sphere(center=ref[b], radius=0.02, color=_SWING_REF_VIS,
                                  label=f"hlip_swing_ref_{b}")


@dataclass(kw_only=True)
class HlipReferenceCommandCfg(CommandTermCfg):
    """Configuration for :class:`HlipReferenceCommand`."""

    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)
    """Never resampled mid-episode; the clock and the planner drive it."""

    entity_name: str = "robot"
    """Scene entity whose feet, pelvis and waist are read."""

    planner_name: str = "foot_plan"
    """``HLIPCommandTerm`` (anchor, target, orbit CoM), declared before this one."""

    gait_name: str = "gait"
    """Gait-clock command term."""

    twist_name: str = "twist"
    """Twist command term (vx, vy, wz in the pelvis frame)."""

    site_names: tuple[str, str] = ("left_foot", "right_foot")
    """Foot SITE names in [L, R] order (the tracked swing point)."""

    foot_body_names: tuple[str, str] = ("left_ankle_roll_link", "right_ankle_roll_link")
    """Ankle bodies in [L, R] order (orientation and angular velocity of the feet)."""

    waist_joint_names: tuple[str, ...] = ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint")
    """Waist joints held at their default positions, in ``Y_NAMES`` order."""

    stance_width: float = 0.25
    """Nominal lateral foot offset y_nom used for the nominal swing start, m."""

    swing_start: Literal["nominal", "measured"] = "nominal"
    """Horizontal start of the swing Bezier: the mirrored nominal foothold
    (-x_target, s*y_nom) or the swing site measured at the step event."""

    z_sw_max: float = 0.065
    """Peak height of the swing SITE reference above the contact plane, m."""

    com_z_ref: float | None = None
    """Constant CoM height reference, m; None = the height measured at each step event."""

    pitch_ref: float = 0.0
    """Pelvis pitch reference offset, rad."""

    pitch_amp: float = 0.02
    """Pelvis pitch oscillation amplitude over the gait cycle, rad."""

    roll_asym_amp: float = -0.05
    """Pelvis roll oscillation amplitude over the gait cycle, rad."""

    roll_bias_lat_max: float = 0.15
    """Clamp on the lateral-velocity roll bias atan(vy/g), rad."""

    roll_bias_yaw_max: float = 0.2
    """Clamp on the turning roll bias atan(vx*wz/g), rad."""

    invalidate_on_event: bool = True
    """Drop the CLF finite-difference history at each step event."""

    clf: dict = field(default_factory=dict)
    """``{"Q_weights": {name: (q_pos, q_vel)}, "R_weights": {name: r}}`` over ``Y_NAMES``."""

    def build(self, env: ManagerBasedRlEnv) -> HlipReferenceCommand:
        return HlipReferenceCommand(self, env)
