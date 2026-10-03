"""Hand-force loco-manipulation MPC command term (CURRENT GOAL consumer).

``HandForceMPCCommand`` runs the vendored 18-D ``LocoManipMPC`` and feeds it
the sparse command tuple emitted by the two ``HandForceEePoseCommand`` terms
— it CONSUMES commands (like it consumes the twist); it owns no schedule of
its own:

* QP hand gate = commanded mask ∧ debounced measured contact (a scheduled
  press with no palm on the surface would demand an impossible force from the
  balance plan; 0.1 s debounce keeps the 10 Hz gate from chattering at the
  zero-force boundary).
* ``u_ref`` hand slots = the world-frame REACTION of the commanded
  hand-on-surface force (−f_cmd_w), tracked by ``R_f_hand = 1e-2`` — against
  a stiff surface the press exists only through this tracking term binding
  against the CoM/momentum costs (measured on the wall-press experiments; a
  1e-4 regularizer will not press).
* ``R_hand_balance = 0``: per-hand commands are independent; the L-R symmetry
  cost is a pure ‖f‖² penalty against any single-hand press.
* Per-env WORLD-frame hand-force bounds = the AABB of the friction cone
  around each commanded press direction (truncated at ``f_hand_max``), fed
  through ``LocoManipMPCInput.hand_force_lb/ub`` — replaces the pimpc/jax
  backends' baked-in world-axis box (x ∈ [−f_h, 0]), whose sign assumption
  only held for a robot facing +x (the reason the legacy wall task clamped
  spawn yaw). ``f_hand_max`` stays close to the command range: a loose cap
  opens a phantom tangential channel through the box corners (measured).

The CoM / angular-momentum / GRF / foot-placement reward references thereby
become press-consistent: the plan leans and re-distributes GRF for the
commanded force, and the ``mpc_*`` tracking rewards teach the policy that
whole-body bracing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from torch import Tensor

from rl.mpc_online_tracking.mdp.mpc_command import (
    LocoMPCCommand,
    LocoMPCCommandCfg,
    _quat_to_rot,
)
from rl.mpc_online_tracking.mpc_training.cd_mpc.contact_schedule import (
    make_walking_schedule,
)
from rl.mpc_online_tracking.mpc_training.cd_mpc.loco_manip_mpc import (
    LocoManipMPC,
    LocoManipMPCConfig,
    LocoManipMPCInput,
)

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.viewer.debug_visualizer import DebugVisualizer

    from .commands import HandForceEePoseCommand


def _cone_aabb(press_dir_w: Tensor, mu: float, cap: Tensor) -> tuple[Tensor, Tensor]:
    """World-axis AABB of the reaction-force cone, truncated at ``cap``.

    The hand presses along ``press_dir_w`` (unit, world); the force ON THE
    ROBOT is −f·(p̂ + μ·t̂) with f ∈ [0, cap], t̂ ⊥ p̂, |t̂| ≤ 1. Per world
    axis the direction factor spans p̂_i ± μ·√(1−p̂_i²); the box is the convex
    hull of the corner values with 0 (f = 0).

    ``cap`` is PER-ENV and COMMAND-SCALED by the caller (f_cmd + slack, not
    the global f_hand_max): a box decouples tangential from normal, so a
    command-independent cap hands the QP a phantom
    tangential-without-pressing channel (measured: 13 N of vertical
    crutch-support at ZERO commanded force). Scaling the box with the command
    collapses that channel when nothing is commanded. Returns (lb, ub),
    each (B, 3).
    """
    p = press_dir_w
    s = (1.0 - p.pow(2)).clamp(min=0.0).sqrt()
    c = cap.unsqueeze(-1)
    c1 = -c * (p - mu * s)
    c2 = -c * (p + mu * s)
    lb = torch.minimum(c1, c2).clamp(max=0.0)
    ub = torch.maximum(c1, c2).clamp(min=0.0)
    return lb, ub


@dataclass(kw_only=True)
class HandForceMPCCommandCfg(LocoMPCCommandCfg):
    """Configuration for :class:`HandForceMPCCommand`."""

    hand_site_names: tuple[str, str] = ("left_palm", "right_palm")
    ee_command_names: tuple[str, str] = ("left_ee_pose", "right_ee_pose")
    """Hand command terms (L, R order) the MPC reads mask/force from."""
    lhand_sensor_name: str | tuple[str, ...] = "lhand_cube_contact"
    rhand_sensor_name: str | tuple[str, ...] = "rhand_cube_contact"
    """Measured-contact sensor(s) per hand for the debounced gate — a tuple
    is OR-combined (wall_following: one sensor per wall segment)."""

    mu_hand: float = 0.6
    f_hand_max: float = 50.0
    f_hand_headroom: float = 1.25
    """MULTIPLICATIVE headroom on the commanded magnitude in the
    command-scaled force box (cap = headroom · f_cmd, clamped at f_hand_max).
    Multiplicative on purpose: any additive slack is a small phantom
    tangential channel that survives at zero command (measured 1.9 N vertical
    crutch with +5 N slack); zero command must mean a ZERO box."""
    R_f_hand: float = 1e-1
    """u_ref tracking weight. 1e-2 tracks but still lets the QP re-optimize
    the hand as a balance actuator (measured: 13 N vertical crutch at zero
    command, 30% off-command deviations); at 1e-1 the commanded force is
    effectively a boundary condition and the QP's freedom is HOW the feet/CoM
    accommodate it — the intended semantics of a commanded-force interface."""
    R_hand_balance: float = 0.0
    contact_debounce_steps: int = 5
    """Env steps (50 Hz) a lost contact is still treated as present (0.1 s)."""

    def build(self, env: ManagerBasedRlEnv) -> HandForceMPCCommand:
        return HandForceMPCCommand(self, env)


class HandForceMPCCommand(LocoMPCCommand):
    """Loco-manip MPC consuming the sparse hand command tuple."""

    cfg: HandForceMPCCommandCfg

    def __init__(self, cfg: HandForceMPCCommandCfg, env: ManagerBasedRlEnv) -> None:
        super().__init__(cfg, env)

        manip_cfg = LocoManipMPCConfig(
            N=cfg.mpc_horizon,
            dt=cfg.mpc_dt,
            mass=cfg.mass,
            mu_hand=cfg.mu_hand,
            f_hand_max=cfg.f_hand_max,
            R_f_hand=cfg.R_f_hand,
            R_hand_balance=cfg.R_hand_balance,
            solver_type=cfg.solver_type,
        )
        self._mpc = LocoManipMPC(manip_cfg, device=env.device)

        B = env.num_envs
        device = env.device
        self._u_prev = torch.zeros(B, 18, device=device)

        hand_site_cfg = SceneEntityCfg("robot", site_names=cfg.hand_site_names)
        hand_site_cfg.resolve(env.scene)
        self._hand_site_ids: list[int] = hand_site_cfg.site_ids
        _, _hand_names = self._robot.find_sites(
            cfg.hand_site_names, preserve_order=False
        )
        self._lh_local_idx = next(
            i for i, n in enumerate(_hand_names) if "left" in n.lower()
        )
        self._rh_local_idx = next(
            i for i, n in enumerate(_hand_names) if "right" in n.lower()
        )

        def _sensor_list(names: str | tuple[str, ...]) -> list[ContactSensor]:
            if isinstance(names, str):
                names = (names,)
            return [env.scene[n] for n in names]

        self._lhand_sensors = _sensor_list(cfg.lhand_sensor_name)
        self._rhand_sensors = _sensor_list(cfg.rhand_sensor_name)
        self._last_contact_step = torch.full(
            (B, 2), -(10**9), dtype=torch.long, device=device
        )

        # Reward/vis buffers: commanded force ON SURFACE and the QP's planned
        # hand force ON ROBOT, both world frame.
        self._hand_cmd_w = torch.zeros(B, 6, device=device)
        self._hand_u_star = torch.zeros(B, 6, device=device)
        self._gate = torch.zeros(B, 2, device=device)
        self._vis_r_h = torch.zeros(B, 2, 3, device=device)

    # ── helpers ─────────────────────────────────────────────────────────────

    def _ee_terms(self) -> tuple[HandForceEePoseCommand, HandForceEePoseCommand]:
        mgr = self._env.command_manager
        return (
            mgr.get_term(self.cfg.ee_command_names[0]),
            mgr.get_term(self.cfg.ee_command_names[1]),
        )

    def _debounced_contact(self) -> Tensor:
        """(B, 2) float: measured palm-surface contact with dropout debounce
        (OR over the hand's sensor list)."""
        for h, sensors in ((0, self._lhand_sensors), (1, self._rhand_sensors)):
            raw = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            for sensor in sensors:
                found = sensor.data.found
                if found is not None:
                    raw |= (found > 0).any(dim=-1)
            self._last_contact_step[:, h] = torch.where(
                raw,
                torch.full_like(self._last_contact_step[:, h], self._step_count),
                self._last_contact_step[:, h],
            )
        return (
            (self._step_count - self._last_contact_step)
            <= self.cfg.contact_debounce_steps
        ).float()

    # ── solve ───────────────────────────────────────────────────────────────

    def _update_command(self) -> None:
        self._step_count += 1
        self._traj_step += 1
        contact_meas = self._debounced_contact()  # every step: debounce fidelity
        self._interpolate_traj_refs()

        if self._step_count % self.cfg.run_every_n_steps != 0:
            return

        cfg = self.cfg
        B = self.num_envs
        N = cfg.mpc_horizon
        dt = cfg.mpc_dt
        device = self.device
        robot = self._robot

        c = robot.data.root_link_pos_w
        lv = robot.data.root_link_lin_vel_w
        av = robot.data.root_link_ang_vel_w
        l = lv * cfg.mass
        k = av @ self._I_approx
        x0 = torch.cat([c, l, k], dim=-1)

        site_pos = robot.data.site_pos_w[:, self._site_ids, :]
        site_quat = robot.data.site_quat_w[:, self._site_ids, :]
        r_lf = site_pos[:, self._lf_site_local_idx, :]
        r_rf = site_pos[:, self._rf_site_local_idx, :]
        R_lf = _quat_to_rot(site_quat[:, self._lf_site_local_idx, :])
        R_rf = _quat_to_rot(site_quat[:, self._rf_site_local_idx, :])

        hand_pos = robot.data.site_pos_w[:, self._hand_site_ids, :]
        r_lh = hand_pos[:, self._lh_local_idx, :]
        r_rh = hand_pos[:, self._rh_local_idx, :]

        # ── sparse hand commands ────────────────────────────────────────────
        lh_term, rh_term = self._ee_terms()
        mask_cmd = torch.stack(
            [lh_term.contact_mask, rh_term.contact_mask], dim=-1
        )
        hand_contact = mask_cmd * contact_meas
        self._gate = hand_contact.detach().clone()

        f_cmd_lh_w = lh_term.force_cmd_w  # force ON SURFACE
        f_cmd_rh_w = rh_term.force_cmd_w
        self._hand_cmd_w = torch.cat([f_cmd_lh_w, f_cmd_rh_w], dim=-1).detach()

        # Cone frames for the sparse-ADMM path: cone about +x of the given
        # frame must be the REACTION direction (−x̂_cmd) — same 180°-about-z
        # flip as the box/wall terms.
        flip = torch.diag(torch.tensor([-1.0, -1.0, 1.0], device=device))
        _, lh_quat_w = lh_term.target_pose_w()
        _, rh_quat_w = rh_term.target_pose_w()
        R_lh = _quat_to_rot(lh_quat_w) @ flip
        R_rh = _quat_to_rot(rh_quat_w) @ flip

        # Per-env world-axis bounds: friction-cone AABB per hand (pimpc/jax),
        # command-scaled (see _cone_aabb) and capped at f_hand_max.
        cap_lh = (cfg.f_hand_headroom * lh_term.force_cmd_mag).clamp(max=cfg.f_hand_max)
        cap_rh = (cfg.f_hand_headroom * rh_term.force_cmd_mag).clamp(max=cfg.f_hand_max)
        lb_lh, ub_lh = _cone_aabb(lh_term.press_dir_w, cfg.mu_hand, cap_lh)
        lb_rh, ub_rh = _cone_aabb(rh_term.press_dir_w, cfg.mu_hand, cap_rh)
        hand_force_lb = torch.cat([lb_lh, lb_rh], dim=-1)
        hand_force_ub = torch.cat([ub_lh, ub_rh], dim=-1)

        # ── twist → state reference (LocoMPCCommand recipe, unchanged) ──────
        vel_cmd = self._env.command_manager.get_command(cfg.vel_cmd_name)
        if vel_cmd is not None:
            vx_body = vel_cmd[:, 0]
            vy_body = vel_cmd[:, 1]
            wz = vel_cmd[:, 2]
        else:
            vx_body = torch.zeros(B, device=device)
            vy_body = torch.zeros(B, device=device)
            wz = torch.zeros(B, device=device)

        quat_w = robot.data.root_link_quat_w
        q_w, q_x, q_y, q_z = quat_w[:, 0], quat_w[:, 1], quat_w[:, 2], quat_w[:, 3]
        yaw = torch.atan2(
            2.0 * (q_w * q_z + q_x * q_y), 1.0 - 2.0 * (q_y * q_y + q_z * q_z)
        )
        cos_y, sin_y = yaw.cos(), yaw.sin()
        vx = cos_y * vx_body - sin_y * vy_body
        vy = sin_y * vx_body + cos_y * vy_body

        k_steps = torch.arange(N + 1, device=device, dtype=torch.float32)
        yaw_k = yaw.unsqueeze(1) + wz.unsqueeze(1) * k_steps * dt
        cos_k, sin_k = yaw_k.cos(), yaw_k.sin()
        vx_w_k = cos_k * vx_body.unsqueeze(1) - sin_k * vy_body.unsqueeze(1)
        vy_w_k = sin_k * vx_body.unsqueeze(1) + cos_k * vy_body.unsqueeze(1)

        x_ref = x0.unsqueeze(1).expand(B, N + 1, -1).clone()
        x_ref[:, 0, 0] = c[:, 0]
        x_ref[:, 0, 1] = c[:, 1]
        x_ref[:, 1:, 0] = c[:, 0:1] + torch.cumsum(vx_w_k[:, :-1] * dt, dim=1)
        x_ref[:, 1:, 1] = c[:, 1:2] + torch.cumsum(vy_w_k[:, :-1] * dt, dim=1)
        x_ref[:, :, 2] = robot.data.default_root_state[:, 2:3]
        x_ref[:, :, 3] = vx_w_k * cfg.mass
        x_ref[:, :, 4] = vy_w_k * cfg.mass
        x_ref[:, :, 5] = 0.0
        I_zz = float(self._I_approx[2, 2])
        x_ref[:, :, 6:8] = 0.0
        x_ref[:, :, 8] = (I_zz * wz).unsqueeze(1)

        gait_phase = getattr(self._env, "_gait_phase", torch.zeros(B, device=device))
        v_cmd_3d = torch.zeros(B, 3, device=device)
        v_cmd_3d[:, 0] = vx
        v_cmd_3d[:, 1] = vy
        schedule = make_walking_schedule(
            B=B, N=N, r_LF=r_lf, r_RF=r_rf,
            gait_phase=gait_phase, period=cfg.gait_period,
            dt=dt, duty_factor=cfg.duty_factor,
            com_pos=c, v_cmd=v_cmd_3d, yaw=yaw, yaw_rate=wz,
            hip_width=cfg.hip_width,
            R_LF_rot=R_lf, R_RF_rot=R_rf, device=device,
        )

        # ── press reference: u_ref hand slots = reaction of the command ─────
        u_ref = torch.zeros(B, N, 18, device=device)
        u_ref[:, :, 12:15] = -f_cmd_lh_w.unsqueeze(1)
        u_ref[:, :, 15:18] = -f_cmd_rh_w.unsqueeze(1)

        mpc_in = LocoManipMPCInput(
            x0=x0,
            schedule=schedule,
            x_ref=x_ref,
            u_ref=u_ref,
            u_prev=self._u_prev,
            c_bar=None,
            R_LH=R_lh,
            R_RH=R_rh,
            r_LH=r_lh,
            r_RH=r_rh,
            hand_contact=hand_contact,
            other_body_force=torch.zeros(B, 3, device=device),
            other_body_torque=torch.zeros(B, 3, device=device),
            box_resist_force=torch.zeros(B, 3, device=device),
            hand_force_lb=hand_force_lb,
            hand_force_ub=hand_force_ub,
        )
        with torch.no_grad():
            mpc_out = self._mpc.solve(mpc_in)

        self._grf_ref = torch.cat(
            [mpc_out.u_star[:, 0:3], mpc_out.u_star[:, 6:9]], dim=-1
        )
        self._hand_u_star = mpc_out.u_star[:, 12:18].detach()
        self._u_prev = mpc_out.u_star.detach()

        x_pred = mpc_out.x_pred.detach()
        self._com_traj = x_pred[:, :, 0:3]
        self._vel_traj = x_pred[:, :, 3:6] / cfg.mass
        k_pred = x_pred[:, :, 6:9]
        self._k_traj = k_pred
        self._kdot_traj = torch.zeros_like(k_pred)
        if N >= 2:
            self._kdot_traj[:, :-1] = (k_pred[:, 1:] - k_pred[:, :-1]) / dt
            self._kdot_traj[:, -1] = self._kdot_traj[:, -2]

        self._traj_step = 0
        self._interpolate_traj_refs()

        self._vis_com = self._com_traj
        self._vis_ang_mom = self._k_traj
        self._vis_r_lf = schedule.r_LF.detach()
        self._vis_r_rf = schedule.r_RF.detach()
        self._vis_sigma_lf = schedule.sigma[:, :, 0].detach()
        self._vis_sigma_rf = schedule.sigma[:, :, 1].detach()
        self._vis_r_h = torch.stack([r_lh, r_rh], dim=1).detach()

        sigma_lf = schedule.sigma[:, :, 0]
        sigma_rf = schedule.sigma[:, :, 1]
        lf_prev = torch.cat([sigma_lf[:, :1], sigma_lf[:, :-1]], dim=1)
        rf_prev = torch.cat([sigma_rf[:, :1], sigma_rf[:, :-1]], dim=1)
        lf_td = (sigma_lf > 0.5) & (lf_prev < 0.5)
        rf_td = (sigma_rf > 0.5) & (rf_prev < 0.5)
        self._lf_landing_valid = lf_td.any(dim=1)
        self._rf_landing_valid = rf_td.any(dim=1)
        lf_idx = lf_td.float().argmax(dim=1)
        rf_idx = rf_td.float().argmax(dim=1)
        batch_idx = torch.arange(B, device=device)
        self._lf_landing_target = torch.where(
            self._lf_landing_valid.unsqueeze(-1),
            schedule.r_LF[batch_idx, lf_idx],
            r_lf,
        )
        self._rf_landing_target = torch.where(
            self._rf_landing_valid.unsqueeze(-1),
            schedule.r_RF[batch_idx, rf_idx],
            r_rf,
        )

    # ── vis ─────────────────────────────────────────────────────────────────

    def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
        import numpy as np

        super()._debug_vis_impl(visualizer)
        env_indices = visualizer.get_env_indices(self.num_envs)
        if not env_indices:
            return
        r_np = self._vis_r_h.cpu().numpy()
        gate_np = self._gate.cpu().numpy()
        plan_np = (-self._hand_u_star).cpu().numpy()  # ON SURFACE = -reaction
        cmd_np = self._hand_cmd_w.cpu().numpy()
        _SCALE = 0.01
        for b in env_indices:
            for h, tag in ((0, "lh"), (1, "rh")):
                rgb = (0.10, 0.90, 0.20) if gate_np[b, h] > 0.5 else (0.55, 0.55, 0.55)
                visualizer.add_sphere(
                    center=r_np[b, h],
                    radius=0.03,
                    color=(*rgb, 0.85),
                    label=f"hf_mpc_{tag}_{b}",
                )
                f_plan = plan_np[b, h * 3 : h * 3 + 3]
                if float(np.linalg.norm(f_plan)) > 1.0:
                    visualizer.add_arrow(
                        start=r_np[b, h],
                        end=r_np[b, h] + _SCALE * f_plan,
                        color=(0.20, 0.45, 1.00, 0.95),
                        width=0.014,
                        label=f"hf_mpc_{tag}_plan_{b}",
                    )
                f_cmd = cmd_np[b, h * 3 : h * 3 + 3]
                if float(np.linalg.norm(f_cmd)) > 1.0:
                    visualizer.add_arrow(
                        start=r_np[b, h],
                        end=r_np[b, h] + _SCALE * f_cmd,
                        color=(0.10, 0.90, 0.20, 0.85),
                        width=0.009,
                        label=f"hf_mpc_{tag}_cmd_{b}",
                    )

    def _resample_command(self, env_ids: Tensor) -> None:
        super()._resample_command(env_ids)
        self._u_prev[env_ids] = 0.0
        self._hand_cmd_w[env_ids] = 0.0
        self._hand_u_star[env_ids] = 0.0
        self._gate[env_ids] = 0.0
        self._last_contact_step[env_ids] = -(10**9)
