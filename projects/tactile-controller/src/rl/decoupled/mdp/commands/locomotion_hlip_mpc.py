from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from rl.decoupled.cd_mpc.contact_schedule import ContactSchedule
from rl.decoupled.mdp.commands.locomotion_mpc import (
    LocoMPCCommand,
    LocoMPCCommandCfg,
    _quat_to_rot,
)
from rl.decoupled.mdp.commands.hlip import HLIPCommandTerm, _rot2

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


class LocoHlipMPCCommand(LocoMPCCommand):
    """Centroidal MPC on the H-LIP planner's footholds and orbit CoM reference."""

    cfg: LocoHlipMPCCommandCfg

    @property
    def planner(self) -> HLIPCommandTerm:
        """The ``foot_plan`` term this MPC consumes."""
        return self._env.command_manager.get_term(self.cfg.planner_name)

    def _build_refs_and_schedule(
        self, *, c: Tensor, lv: Tensor, r_lf: Tensor, r_rf: Tensor
    ) -> tuple[Tensor, Tensor, ContactSchedule]:
        cfg = self.cfg
        B, N, dt = self.num_envs, cfg.mpc_horizon, cfg.mpc_dt
        device = self.device
        robot = self._robot
        P = self.planner
        gait = self._env.command_manager.get_term("gait")
        rows = torch.arange(B, device=device)
        T = 0.5 * gait.period

        # ── sigma from the clock; a holding env is double support ───────────
        k_n = torch.arange(N, device=device, dtype=torch.float32)
        phase_traj = gait.phase.unsqueeze(1) + (2.0 * math.pi / gait.period) * dt * k_n
        thr = math.cos(math.pi * cfg.duty_factor)
        rf_stance = phase_traj.sin() < thr
        lf_stance = (phase_traj + math.pi).sin() < thr
        hold = P.standing.unsqueeze(1)
        lf_stance = lf_stance | hold
        rf_stance = rf_stance | hold
        sigma = torch.zeros(B, N, 4, device=device)
        sigma[:, :, 0] = lf_stance.float()
        sigma[:, :, 1] = rf_stance.float()

        # ── landings: swing foot -> committed target, stance foot -> the
        #    orbit's next step from the target (mirrored lateral) ───────────
        swing = P.swing
        tgt = P.target_pos_w.clone()
        psi_t = P.target_yaw_w
        u = P.orbit_step
        s = u[:, 2]
        u_next = torch.stack([u[:, 0], u[:, 1] - s * P.cfg.stance_width], dim=-1)
        nxt = torch.cat([tgt[:, :2] + _rot2(psi_t, u_next), tgt[:, 2:3]], dim=-1)
        land_l = torch.where((swing == 0).unsqueeze(-1), tgt, nxt)
        land_r = torch.where((swing == 1).unsqueeze(-1), tgt, nxt)
        land_l[:, 2] = r_lf[:, 2]
        land_r[:, 2] = r_rf[:, 2]

        def _traj(r0: Tensor, stance: Tensor, land: Tensor) -> Tensor:
            traj = torch.zeros(B, N, 3, device=device)
            cur = r0.clone()
            for k in range(N):
                if k > 0:
                    new = stance[:, k] & ~stance[:, k - 1]
                    cur = torch.where(new.unsqueeze(-1), land, cur)
                traj[:, k] = cur
            return traj

        r_lf_traj = _traj(r_lf, lf_stance, land_l)
        r_rf_traj = _traj(r_rf, rf_stance, land_r)

        site_quat = robot.data.site_quat_w[:, self._site_ids, :]
        R_lf = _quat_to_rot(site_quat[:, self._lf_site_local_idx, :])
        R_rf = _quat_to_rot(site_quat[:, self._rf_site_local_idx, :])
        schedule = ContactSchedule(
            sigma=sigma,
            r_LF=r_lf_traj.contiguous(),
            r_RF=r_rf_traj.contiguous(),
            r_LH=torch.zeros(B, N, 3, device=device),
            r_RH=torch.zeros(B, N, 3, device=device),
            R_LF=R_lf.contiguous(),
            R_RF=R_rf.contiguous(),
        )

        # ── x_ref: orbit CoM flow at the node times, re-anchored on every
        #    step boundary the horizon crosses (at most two) ────────────────
        t0 = (gait.phase % math.pi) / math.pi * T
        t_k = t0.unsqueeze(1) + dt * torch.arange(N + 1, device=device, dtype=torch.float32)
        seg = (t_k / T).floor().clamp(max=2.0)                       # (B, N+1)
        tau = t_k - seg * T
        stance = 1 - swing
        p_st = P.ref_feet_pos_w[rows, stance, :2]
        psi_st = P.ref_feet_yaw[rows, stance]
        psi_2 = psi_t + torch.atan2(torch.sin(psi_t - psi_st), torch.cos(psi_t - psi_st))
        x0n, y0n = P.next_orbit_init(T)
        pos_even, vel_even = P.orbit_flow(tau)                       # segments 0 and 2
        lam = P.orbit_lam.unsqueeze(-1)
        ch, sh = torch.cosh(lam * tau), torch.sinh(lam * tau)
        pos_odd = torch.stack([x0n[:, 0:1] * ch + x0n[:, 1:2] / lam * sh,
                               y0n[:, 0:1] * ch + y0n[:, 1:2] / lam * sh], dim=-1)
        vel_odd = torch.stack([x0n[:, 0:1] * lam * sh + x0n[:, 1:2] * ch,
                               y0n[:, 0:1] * lam * sh + y0n[:, 1:2] * ch], dim=-1)
        anchors = torch.stack([p_st, tgt[:, :2], nxt[:, :2]], dim=1)          # (B, 3, 2)
        yaws = torch.stack([psi_st, psi_t, psi_2], dim=1)                     # (B, 3)
        seg_i = seg.long()
        anc = torch.gather(anchors, 1, seg_i.unsqueeze(-1).expand(-1, -1, 2))
        yaw = torch.gather(yaws, 1, seg_i)
        odd = (seg_i == 1).unsqueeze(-1)
        pos_loc = torch.where(odd, pos_odd, pos_even)
        vel_loc = torch.where(odd, vel_odd, vel_even)
        pos_w = anc + _rot2(yaw, pos_loc)
        vel_w = _rot2(yaw, vel_loc)
        hold3 = P.standing.view(B, 1, 1)
        pos_w = torch.where(hold3, P.ref_com_pos_w[:, None, :2].expand(B, N + 1, 2), pos_w)
        vel_w = torch.where(hold3, torch.zeros_like(vel_w), vel_w)

        vel_cmd = self._env.command_manager.get_command(cfg.vel_cmd_name)
        wz = vel_cmd[:, 2] * (~P.standing).float()
        x_ref = torch.zeros(B, N + 1, 9, device=device)
        x_ref[:, :, 0:2] = pos_w
        x_ref[:, :, 2] = cfg.com_height
        x_ref[:, :, 3:5] = vel_w * cfg.mass
        x_ref[:, :, 8] = (float(self._I_approx[2, 2]) * wz).unsqueeze(1)

        # ── c_bar: the base's prediction (measured velocity, measured height)
        t_lin = torch.arange(N + 1, device=device, dtype=torch.float32) * dt
        c_bar = c.unsqueeze(1).expand(B, N + 1, 3).clone()
        c_bar[:, :, 0] += lv[:, 0:1] * t_lin.unsqueeze(0)
        c_bar[:, :, 1] += lv[:, 1:2] * t_lin.unsqueeze(0)

        self._x_ref_dbg = x_ref
        return x_ref, c_bar, schedule


@dataclass(kw_only=True)
class LocoHlipMPCCommandCfg(LocoMPCCommandCfg):
    """Config for :class:`LocoHlipMPCCommand`."""

    planner_name: str = "foot_plan"
    """The ``HLIPCommandTerm`` whose targets and orbit feed the MPC."""

    mpc_impl: str = "cd"

    def build(self, env: "ManagerBasedRlEnv") -> LocoHlipMPCCommand:
        return LocoHlipMPCCommand(self, env)