"""`EEQposeCommand` + contact mask + force realized by the unilateral virtual
spring of `ee_spring.py` (Portela et al. 2024), riding the dataset target:
anchor = target − (F_n / k_ext)·n̂ with n̂ = the target's +x, so the tip ON the
target realizes exactly the commanded force. Same accessor surface as
`EeSpringContactCommand` (`contact_mask`, `phantom`, `force_cmd_*`,
`press_dir_*`, `measured_force_w`), so the force observations, rewards and
the loco-manip MPC consume it unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from mjlab.utils.lab_api.math import quat_apply

from .ee_qpos import EEQposeCommand, EEQposeCommandCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.viewer.debug_visualizer import DebugVisualizer

_CMD_FORCE_VIS = (0.10, 0.90, 0.20, 0.95)   # commanded
_MEAS_FORCE_VIS = (0.95, 0.20, 0.85, 0.95)  # spring (virtual surface)


class EEQposeSpringCommand(EEQposeCommand):
    """Dataset-drawn tip pose + reference joint positions + contact mask +
    3-D force command, the force realized by a virtual spring on the tip.

    Attributes:
        contact_mask: 1 while the window is a contact command. Shape (B,).
        phantom: 1 on contact windows that carry NO spring. Shape (B,).
        force_mag: Drawn normal force of the window, N. Shape (B,).
    """

    cfg: EEQposeSpringCommandCfg

    def __init__(self, cfg: EEQposeSpringCommandCfg, env: ManagerBasedRlEnv) -> None:
        super().__init__(cfg, env)
        B, dev = self.num_envs, self.device
        self.contact_mask = torch.zeros(B, device=dev)
        self.force_mag = torch.zeros(B, device=dev)
        self.phantom = torch.zeros(B, device=dev)
        self._force_cmd_mag = torch.zeros(B, device=dev)
        self._force_cmd_local = torch.zeros(B, 3, device=dev)
        self._tan_u = torch.zeros(B, 2, device=dev)  # (ratio·cosθ, ratio·sinθ)
        self._contact_age = torch.zeros(B, device=dev)
        self._just_reset = torch.zeros(B, dtype=torch.bool, device=dev)
        self._spring_force_w = torch.zeros(B, 3, device=dev)  # hand ON surface
        self._anchor_prev = torch.zeros(B, 3, device=dev)
        self._anchor_fresh = torch.ones(B, dtype=torch.bool, device=dev)
        self._zero3 = torch.zeros(B, 1, 3, device=dev)

        self.metrics["contact_duty"] = torch.zeros(B, device=dev)
        self.metrics["phantom_duty"] = torch.zeros(B, device=dev)
        self.metrics["force_track_err"] = torch.zeros(B, device=dev)
        self.metrics["spring_engaged"] = torch.zeros(B, device=dev)

    # -- command tuple accessors -------------------------------------------------

    @property
    def press_dir_b(self) -> Tensor:
        """Commanded press axis (+x of the target frame), command frame. Shape (B, 3)."""
        ex = torch.zeros(self.num_envs, 3, device=self.device)
        ex[:, 0] = 1.0
        return quat_apply(self.command[:, 3:], ex)

    @property
    def press_dir_w(self) -> Tensor:
        """Commanded press axis (+x of the target frame), world frame. Shape (B, 3)."""
        _, quat_w = self.target_pose_w()
        ex = torch.zeros(self.num_envs, 3, device=self.device)
        ex[:, 0] = 1.0
        return quat_apply(quat_w, ex)

    @property
    def force_cmd_mag(self) -> Tensor:
        """|F_cmd| after the window ramps, N. Shape (B,)."""
        return self._force_cmd_mag

    @property
    def force_cmd_b(self) -> Tensor:
        """Commanded force vector, command frame, N. Shape (B, 3)."""
        return quat_apply(self.command[:, 3:], self._force_cmd_local)

    @property
    def force_cmd_w(self) -> Tensor:
        """Commanded force vector, world frame, N. Shape (B, 3)."""
        _, quat_w = self.target_pose_w()
        return quat_apply(quat_w, self._force_cmd_local)

    def measured_force_w(self) -> Tensor:
        """Hand-on-surface force (= −applied wrench), world frame, N. Shape (B, 3)."""
        return self._spring_force_w

    # -- lifecycle -------------------------------------------------------------

    def _resample_command(self, env_ids: Tensor) -> None:
        super()._resample_command(env_ids)  # dataset draw: pose + ref joint pos

        n = len(env_ids)
        draw = torch.rand(n, device=self.device) < self.cfg.contact_prob
        draw &= ~self._just_reset[env_ids]
        self._just_reset[env_ids] = False

        self.contact_mask[env_ids] = 0.0
        self.force_mag[env_ids] = 0.0
        self.phantom[env_ids] = 0.0
        self._force_cmd_mag[env_ids] = 0.0
        self._force_cmd_local[env_ids] = 0.0
        self._tan_u[env_ids] = 0.0
        self._contact_age[env_ids] = 0.0
        self._anchor_fresh[env_ids] = True

        ids = env_ids[draw]
        if len(ids) == 0:
            return
        m = len(ids)
        self.contact_mask[ids] = 1.0
        self.force_mag[ids] = torch.empty(m, device=self.device).uniform_(*self.cfg.force_range)
        # contact windows hold contact_resampling_time_range, free draws resampling_time_range
        self.time_left[ids] = torch.empty(m, device=self.device).uniform_(
            *self.cfg.contact_resampling_time_range)
        self.phantom[ids] = (torch.rand(m, device=self.device) < self.cfg.phantom_prob).float()
        # 3-D force: tangential fraction + in-plane angle, fixed per window
        ratio = torch.empty(m, device=self.device).uniform_(0.0, self.cfg.tangential_ratio_max)
        ang = torch.empty(m, device=self.device).uniform_(0.0, 2.0 * torch.pi)
        self._tan_u[ids, 0] = ratio * torch.cos(ang)
        self._tan_u[ids, 1] = ratio * torch.sin(ang)

    def _update_command(self) -> None:
        super()._update_command()
        cfg = self.cfg
        dt = self._env.step_dt
        masked = self.contact_mask > 0.5

        self._contact_age += dt * masked.float()
        ramp_in = (self._contact_age / cfg.force_ramp_time).clamp(0.0, 1.0)
        ramp_out = (self.time_left / cfg.force_ramp_time).clamp(0.0, 1.0)
        f_cmd_n = self.force_mag * torch.minimum(ramp_in, ramp_out) * masked.float()
        self._force_cmd_local = torch.stack(
            (f_cmd_n, f_cmd_n * self._tan_u[:, 0], f_cmd_n * self._tan_u[:, 1]), dim=-1)
        self._force_cmd_mag = self._force_cmd_local.norm(dim=-1)

        # unilateral normal spring riding the target
        tgt_p, tgt_q = self.target_pose_w()
        ex = torch.zeros(self.num_envs, 3, device=self.device)
        ex[:, 0] = 1.0
        n_hat = quat_apply(tgt_q, ex)
        anchor = tgt_p - (f_cmd_n / cfg.k_ext).unsqueeze(-1) * n_hat
        anchor_v = torch.where(
            self._anchor_fresh.unsqueeze(-1), torch.zeros_like(anchor), (anchor - self._anchor_prev) / dt)
        self._anchor_prev = anchor.detach().clone()
        self._anchor_fresh[:] = False

        tip = self.robot.data.body_link_pos_w[:, self.body_idx]
        tip_v = self.robot.data.body_link_lin_vel_w[:, self.body_idx]
        pen = ((tip - anchor) * n_hat).sum(-1)
        pen_rate = ((tip_v - anchor_v) * n_hat).sum(-1)

        # command-scaled cap: ramps from 0 with the command, never above the QP's cone headroom
        cap = (cfg.f_headroom * f_cmd_n).clamp(max=cfg.f_cap)
        f_n = (cfg.k_ext * pen + cfg.d_ext * pen_rate).clamp(min=0.0)
        f_n = torch.minimum(f_n, cap)

        live = masked & (self.phantom < 0.5)
        f_n = f_n * live.float()
        # tangential rides the realized normal: the commanded VECTOR scaled by f_n / F_n
        scale = f_n / f_cmd_n.clamp(min=1e-6)
        f_cmd_w = quat_apply(tgt_q, self._force_cmd_local)
        self._spring_force_w = scale.unsqueeze(-1) * f_cmd_w  # hand ON surface

        # reaction ON the robot; xfrc persists until the next write, so every env is written every step
        self.robot.write_external_wrench_to_sim(
            forces=(-self._spring_force_w).unsqueeze(1), torques=self._zero3, body_ids=[self.body_idx])

    def reset(self, env_ids) -> dict[str, float]:
        self.contact_mask[env_ids] = 0.0
        self.force_mag[env_ids] = 0.0
        self.phantom[env_ids] = 0.0
        self._force_cmd_mag[env_ids] = 0.0
        self._force_cmd_local[env_ids] = 0.0
        self._tan_u[env_ids] = 0.0
        self._contact_age[env_ids] = 0.0
        self._spring_force_w[env_ids] = 0.0
        self._anchor_fresh[env_ids] = True
        self._just_reset[env_ids] = True
        self.robot.write_external_wrench_to_sim(
            forces=torch.zeros(len(env_ids), 1, 3, device=self.device),
            torques=torch.zeros(len(env_ids), 1, 3, device=self.device),
            env_ids=env_ids, body_ids=[self.body_idx])
        return super().reset(env_ids)

    def _update_metrics(self) -> None:
        super()._update_metrics()
        masked = self.contact_mask > 0.5
        live = masked & (self.phantom < 0.5)
        self.metrics["contact_duty"] = self.contact_mask.clone()
        self.metrics["phantom_duty"] = self.phantom.clone()
        err = (self.measured_force_w() - self.force_cmd_w).norm(dim=-1)
        self.metrics["force_track_err"] = err * live.float()
        engaged = self._spring_force_w.norm(dim=-1) > 0.5
        self.metrics["spring_engaged"] = (engaged & live).float()

    def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
        super()._debug_vis_impl(visualizer)
        env_indices = visualizer.get_env_indices(self.num_envs)
        if not env_indices:
            return
        import numpy as np

        masked_np = (self.contact_mask > 0.5).cpu().numpy()
        tgt_np = self.target_pose_w()[0].cpu().numpy()
        fcmd_np = self.force_cmd_w.cpu().numpy()
        fmeas_np = self.measured_force_w().cpu().numpy()
        tip_np = self.measured_ee_pose_w()[0].cpu().numpy()
        tag = self.cfg.body_name
        _SCALE = 0.01  # m per N
        for b in env_indices:
            if not masked_np[b]:
                continue
            for kind, start, f, color in (
                ("cmd", tgt_np[b], fcmd_np[b], _CMD_FORCE_VIS),
                ("spring", tip_np[b], fmeas_np[b], _MEAS_FORCE_VIS),
            ):
                if float(np.linalg.norm(f)) > 1.0:
                    visualizer.add_arrow(start=start, end=start + _SCALE * f, color=color,
                                         width=0.010, label=f"spring_force_{kind}_{tag}_{b}")


@dataclass(kw_only=True)
class EEQposeSpringCommandCfg(EEQposeCommandCfg):
    """Config for `EEQposeSpringCommand`; the spring knobs match `EeSpringContactCommandCfg`."""

    contact_prob: float = 0.3
    """Per-resample probability that the new draw is a contact command."""
    phantom_prob: float = 0.10
    """Fraction of contact draws with a live mask/force but NO spring."""
    contact_resampling_time_range: tuple[float, float] = (3.0, 6.0)
    """Hold time of contact windows, s (free draws keep `resampling_time_range`)."""
    force_range: tuple[float, float] = (5.0, 25.0)
    """Normal-component range, N; |F_cmd| ≤ F_n·sqrt(1 + tangential_ratio²)."""
    tangential_ratio_max: float = 0.5
    """Max |F_t| / F_n; ratio uniform in [0, max], random in-plane angle, fixed per window.
    Keep below the QP's mu_hand so every command lies inside its hand cone."""
    force_ramp_time: float = 0.6
    """Force-command ramp at both window edges, s; also ramps the force cap from zero."""
    k_ext: float = 1000.0
    """Virtual surface stiffness, N/m."""
    d_ext: float = 30.0
    """Virtual surface damping on the tip-vs-anchor relative velocity, N·s/m."""
    f_headroom: float = 1.25
    """Spring force cap = f_headroom · F_cmd(t)."""
    f_cap: float = 100.0
    """Absolute backstop on the spring force, N."""

    def build(self, env: ManagerBasedRlEnv) -> EEQposeSpringCommand:
        return EEQposeSpringCommand(self, env)
