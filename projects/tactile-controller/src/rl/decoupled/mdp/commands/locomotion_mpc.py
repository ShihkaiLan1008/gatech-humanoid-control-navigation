"""Centroidal MPC command terms + MPC-plan tracking rewards (framework-shared).

The core of the "mpc online tracking" framework, shared by every task:
``LocoMPCCommand`` / ``LocoManipMPCCommand`` run the vendored batched
centroidal MPC (``rl.mpc_online_tracking.mpc_training.cd_mpc``) as command
terms; the ``mpc_*`` reward functions score the policy against the MPC plan.

Port of ``themis_training.mpc_grf_mdp`` from mpc-rl-main (arXiv:2606.05687),
re-targeted to the Unitree G1: trunk-inertia approximation, mass, hip width,
gait period, and default com height. Task-specific command variants (e.g. the
SE(2) pose-goal twist) live in their task's ``mdp/``.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import Tensor

from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.tasks.velocity.mdp.velocity_command import (
    UniformVelocityCommand,
    UniformVelocityCommandCfg,
)
from mjlab.utils.lab_api.math import wrap_to_pi

from rl.decoupled.cd_mpc import cd as cd_mpc
from rl.decoupled.cd_mpc.centroidal_mpc import CentroidalMPC, MPCConfig, MPCInput
from rl.decoupled.cd_mpc.contact_schedule import ContactSchedule, make_walking_schedule
from rl.g1_assets.g1_constants import G1_HOME_INERTIA, PELVIS_HOME_Z

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.managers.reward_manager import RewardTermCfg
    from mjlab.viewer.debug_visualizer import DebugVisualizer

def _quat_to_rot(q: Tensor) -> Tensor:
    """Convert unit quaternions (B, 4) [w, x, y, z] to rotation matrices (B, 3, 3)."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return torch.stack([
        1 - 2*(y*y + z*z),  2*(x*y - w*z),      2*(x*z + w*y),
        2*(x*y + w*z),      1 - 2*(x*x + z*z),  2*(y*z - w*x),
        2*(x*z - w*y),      2*(y*z + w*x),       1 - 2*(x*x + y*y),
    ], dim=-1).reshape(q.shape[0], 3, 3)

@dataclass(kw_only=True)
class LocoMPCCommandCfg(CommandTermCfg):
    """Configuration for :class:`LocoMPCCommand`."""

    asset_cfg: SceneEntityCfg = field(
        default_factory=lambda: SceneEntityCfg(
            "robot",
            site_names=("left_foot", "right_foot"),
        )
    )

    mpc_dt: float = 0.07
    mpc_horizon: int = 10
    mass: float = 33.34  # G1 total mass (mj_getTotalmass on g1.xml)
    hip_width: float = 0.12  # G1 ankle_roll y offset is +-0.119 m

    gait_period: float = 0.8
    duty_factor: float = 0.5

    com_height: float = PELVIS_HOME_Z

    body_inertia: tuple[tuple[float, ...], ...] = G1_HOME_INERTIA
    vel_cmd_name: str = "twist"

    grf_sensor_name: str = "feet_ground_contact"

    run_every_n_steps: int = 5

    tracking_lookahead_frac: float = 0

    resampling_time_range: tuple[float, float] = (1e6, 1e6)

    unconstrained: bool = False

    solver_type: str = "pimpc"

    vis_footplan: bool = True
    """Debug vis: draw the planned foot-placement spheres.  The foothold
    line turns this off (its own arrows show placement; the spheres drown
    them) while keeping the CoM / angular-momentum glyphs."""

    mpc_impl: str = "centroidal"
    """Which MPC implementation backs this command: "centroidal" (vendored
    multi-backend ``centroidal_mpc.py``) or "cd" (the merged jax-PiMPC-only
    ``cd.py`` — bit-exact equivalent on the loco path, proven by
    ``scripts/rl/test/cd_mpc_equivalence.py``). ``solver_type`` /
    ``unconstrained`` only apply to "centroidal"."""

    def build(self, env: "ManagerBasedRlEnv") -> "LocoMPCCommand":
        return LocoMPCCommand(self, env)

class LocoMPCCommand(CommandTerm):
    """Centroidal locomotion MPC runner as a CommandTerm."""

    cfg: LocoMPCCommandCfg

    def __init__(self, cfg: LocoMPCCommandCfg, env: "ManagerBasedRlEnv") -> None:
        super().__init__(cfg, env)

        self._asset_cfg = copy.deepcopy(cfg.asset_cfg)
        self._asset_cfg.resolve(env.scene)
        self._robot = env.scene[self._asset_cfg.name]
        self._site_ids: list[int] = self._asset_cfg.site_ids

        _, _resolved_names = self._robot.find_sites(
            cfg.asset_cfg.site_names, preserve_order=False
        )
        self._lf_site_local_idx = next(
            i for i, n in enumerate(_resolved_names) if n == "left_foot"
        )
        self._rf_site_local_idx = next(
            i for i, n in enumerate(_resolved_names) if n == "right_foot"
        )

        self._I_approx = torch.tensor(
            cfg.body_inertia, device=env.device, dtype=torch.float32
        )

        self._build_mpc(cfg, env)
        self._nu: int = getattr(self._mpc, "N_U", 12)

        B = env.num_envs
        N = cfg.mpc_horizon
        self._grf_ref: Tensor = torch.zeros(B, 6, device=env.device)
        self._u_prev: Tensor = torch.zeros(B, self._nu, device=env.device)
        self._step_count: int = 0

        self._com_traj:  Tensor = torch.zeros(B, N, 3, device=env.device)
        self._vel_traj:  Tensor = torch.zeros(B, N, 3, device=env.device)
        self._k_traj:    Tensor = torch.zeros(B, N, 3, device=env.device)
        self._kdot_traj: Tensor = torch.zeros(B, N, 3, device=env.device)
        self._traj_step: int = 0

        self._com_mpc_target:      Tensor = torch.zeros(B, 3, device=env.device)
        self._com_vel_mpc_target:  Tensor = torch.zeros(B, 3, device=env.device)
        self._k_mpc_target:        Tensor = torch.zeros(B, 3, device=env.device)
        self._k_dot_mpc_target:    Tensor = torch.zeros(B, 3, device=env.device)

        self._vis_com:      Tensor = torch.zeros(B, N, 3, device=env.device)
        self._vis_ang_mom:  Tensor = torch.zeros(B, N, 3, device=env.device)
        self._vis_r_lf:     Tensor = torch.zeros(B, N, 3, device=env.device)
        self._vis_r_rf:     Tensor = torch.zeros(B, N, 3, device=env.device)
        self._vis_sigma_lf: Tensor = torch.zeros(B, N, device=env.device)
        self._vis_sigma_rf: Tensor = torch.zeros(B, N, device=env.device)

        self._lf_landing_target: Tensor = torch.zeros(B, 3, device=env.device)
        self._rf_landing_target: Tensor = torch.zeros(B, 3, device=env.device)
        self._lf_landing_valid:  Tensor = torch.zeros(B, dtype=torch.bool, device=env.device)
        self._rf_landing_valid:  Tensor = torch.zeros(B, dtype=torch.bool, device=env.device)

    # ── subclass hooks (loco-manip extends these; base = pure loco) ─────────

    def _build_mpc(self, cfg: LocoMPCCommandCfg, env: "ManagerBasedRlEnv") -> None:
        """Construct self._mpc / self._mpc_input_cls / self._extra_mpc_kwargs."""
        if cfg.mpc_impl == "cd":
            self._mpc = cd_mpc.CentroidalMPC(
                cd_mpc.MPCConfig(N=cfg.mpc_horizon, dt=cfg.mpc_dt, mass=cfg.mass),
                device=env.device,
            )
            self._mpc_input_cls = cd_mpc.MPCInput
            # cd's MPCInput requires the external-wrench slots; loco has none.
            self._extra_mpc_kwargs = {
                "other_body_force": torch.zeros(env.num_envs, 3, device=env.device),
                "other_body_torque": torch.zeros(env.num_envs, 3, device=env.device),
            }
        elif cfg.mpc_impl == "centroidal":
            mpc_cfg = MPCConfig(
                N=cfg.mpc_horizon,
                dt=cfg.mpc_dt,
                mass=cfg.mass,
                solver_type=cfg.solver_type,
                unconstrained=cfg.unconstrained,
            )
            self._mpc = CentroidalMPC(mpc_cfg, device=env.device)
            self._mpc_input_cls = MPCInput
            self._extra_mpc_kwargs = {}
        else:
            raise ValueError(
                f"Unknown mpc_impl: {cfg.mpc_impl!r} (expected 'centroidal' or 'cd')"
            )

    def _per_step_update(self) -> None:
        """Runs EVERY policy step, before the run_every_n_steps gate."""

    def _build_refs_and_schedule(
        self, *, c: Tensor, lv: Tensor, r_lf: Tensor, r_rf: Tensor
    ) -> tuple[Tensor, Tensor, ContactSchedule]:
        """Returns (x_ref, c_bar, schedule) for this solve.

        Base = twist-integrated straight-line x_ref + Raibert walking
        schedule (moved verbatim from _update_command).  The foothold line
        overrides this with committed-queue schedule + support-centroid
        x_ref."""
        cfg = self.cfg
        B = self.num_envs
        N = cfg.mpc_horizon
        dt = cfg.mpc_dt
        device = self.device
        robot = self._robot

        vel_cmd = self._env.command_manager.get_command(cfg.vel_cmd_name)
        if vel_cmd is not None:
            vx_body = vel_cmd[:, 0]
            vy_body = vel_cmd[:, 1]
            wz      = vel_cmd[:, 2]
        else:
            vx_body = torch.zeros(B, device=device)
            vy_body = torch.zeros(B, device=device)
            wz      = torch.zeros(B, device=device)

        quat_w = robot.data.root_link_quat_w
        q_w, q_x, q_y, q_z = quat_w[:, 0], quat_w[:, 1], quat_w[:, 2], quat_w[:, 3]
        yaw   = torch.atan2(2.0 * (q_w * q_z + q_x * q_y),
                             1.0 - 2.0 * (q_y * q_y + q_z * q_z))
        cos_y = yaw.cos()
        sin_y = yaw.sin()
        vx    = cos_y * vx_body - sin_y * vy_body
        vy    = sin_y * vx_body + cos_y * vy_body

        site_quat = robot.data.site_quat_w[:, self._site_ids, :]
        R_lf = _quat_to_rot(site_quat[:, self._lf_site_local_idx, :])
        R_rf = _quat_to_rot(site_quat[:, self._rf_site_local_idx, :])

        k_steps = torch.arange(N + 1, device=device, dtype=torch.float32)
        yaw_k   = yaw.unsqueeze(1) + wz.unsqueeze(1) * k_steps * dt
        cos_k   = yaw_k.cos()
        sin_k   = yaw_k.sin()

        vx_w_k = cos_k * vx_body.unsqueeze(1) - sin_k * vy_body.unsqueeze(1)
        vy_w_k = sin_k * vx_body.unsqueeze(1) + cos_k * vy_body.unsqueeze(1)

        x0 = torch.cat([c, lv * cfg.mass, torch.zeros(B, 3, device=device)], dim=-1)
        x_ref = x0.unsqueeze(1).expand(B, N + 1, -1).clone()
        x_ref[:, 0, 0] = c[:, 0]
        x_ref[:, 0, 1] = c[:, 1]
        x_ref[:, 1:, 0] = c[:, 0:1] + torch.cumsum(vx_w_k[:, :-1] * dt, dim=1)
        x_ref[:, 1:, 1] = c[:, 1:2] + torch.cumsum(vy_w_k[:, :-1] * dt, dim=1)
        x_ref[:, :, 2] = cfg.com_height

        x_ref[:, :, 3] = vx_w_k * cfg.mass
        x_ref[:, :, 4] = vy_w_k * cfg.mass
        x_ref[:, :, 5] = 0.0

        I_zz = float(self._I_approx[2, 2])
        x_ref[:, :, 6] = 0.0
        x_ref[:, :, 7] = 0.0
        x_ref[:, :, 8] = (I_zz * wz).unsqueeze(1)

        # Linearization point for the (r - c) x f bilinear term: a PREDICTION
        # of the pelvis path, so it extrapolates the MEASURED velocity and
        # holds the MEASURED height. x_ref above is the DEMAND and must keep
        # the COMMANDED velocity — a reference that follows the measurement
        # stops charging the tracking deficit (the standing-optimum shape).
        # (An OU/EMA blend toward a cycle-mean velocity for the far nodes was
        # measured 2026-08-11: NO gain — realized |e| sits at a ~0.045 m @
        # 0.63 s predictability floor — and it dilutes nodes 1-2. Don't re-add.)
        t_k = torch.arange(N + 1, device=device, dtype=torch.float32) * dt
        c_bar = c.unsqueeze(1).expand(B, N + 1, 3).clone()
        c_bar[:, :, 0] += lv[:, 0:1] * t_k.unsqueeze(0)
        c_bar[:, :, 1] += lv[:, 1:2] * t_k.unsqueeze(0)
        # z (row 2) keeps the MEASURED pelvis height from the expand above,
        # constant along the horizon — x_ref keeps demanding the nominal one.

        gait_phase = self._env.command_manager.get_term("gait").phase
        v_cmd_3d = torch.zeros(B, 3, device=device)
        v_cmd_3d[:, 0] = vx
        v_cmd_3d[:, 1] = vy
        schedule = make_walking_schedule(
            B=B,
            N=N,
            r_LF=r_lf,
            r_RF=r_rf,
            gait_phase=gait_phase,
            period=cfg.gait_period,
            dt=dt,
            duty_factor=cfg.duty_factor,
            com_pos=c,
            v_cmd=v_cmd_3d,
            yaw=yaw,
            yaw_rate=wz,
            hip_width=cfg.hip_width,
            R_LF_rot=R_lf,
            R_RF_rot=R_rf,
            device=device,
        )
        return x_ref, c_bar, schedule

    def _build_mpc_input(self, *, x0: Tensor, schedule, x_ref: Tensor, c_bar: Tensor):
        u_ref = torch.zeros(
            self.num_envs, self.cfg.mpc_horizon, self._nu, device=self.device
        )
        return self._mpc_input_cls(
            x0=x0,
            schedule=schedule,
            x_ref=x_ref,
            u_ref=u_ref,
            u_prev=self._u_prev,
            c_bar=c_bar,
            **self._extra_mpc_kwargs,
        )

    def _post_solve(self, mpc_out) -> None:
        """Hook after the QP solve (loco-manip stores its hand slots)."""

    @property
    def command(self) -> Tensor:
        """Optimal first-step foot forces [B, 6]: [fLF(3), fRF(3)]."""
        return self._grf_ref

    def _interpolate_traj_refs(self) -> None:
        """Interpolate current references from stored MPC trajectories."""
        N = self.cfg.mpc_horizon
        elapsed = self._traj_step * self._env.step_dt
        t_frac_slide = elapsed / self.cfg.mpc_dt
        base = self.cfg.tracking_lookahead_frac * (N - 1)
        t_frac = base + t_frac_slide

        idx = min(max(int(t_frac), 0), N - 2)
        alpha = min(max(t_frac - idx, 0.0), 1.0)

        self._com_mpc_target     = (1 - alpha) * self._com_traj[:, idx]  + alpha * self._com_traj[:, idx + 1]
        self._com_vel_mpc_target = (1 - alpha) * self._vel_traj[:, idx]  + alpha * self._vel_traj[:, idx + 1]
        self._k_mpc_target       = (1 - alpha) * self._k_traj[:, idx]    + alpha * self._k_traj[:, idx + 1]
        self._k_dot_mpc_target   = (1 - alpha) * self._kdot_traj[:, idx] + alpha * self._kdot_traj[:, idx + 1]

    def _update_command(self) -> None:
        """Solve the centroidal MPC and update the GRF reference."""
        self._step_count += 1
        self._traj_step += 1
        self._interpolate_traj_refs()
        self._per_step_update()

        if self._step_count % self.cfg.run_every_n_steps != 0:
            return

        cfg = self.cfg
        B = self.num_envs
        N = cfg.mpc_horizon
        dt = cfg.mpc_dt
        device = self.device
        robot = self._robot

        c  = robot.data.root_link_pos_w
        lv = robot.data.root_link_lin_vel_w
        av = robot.data.root_link_ang_vel_w
        l  = lv * cfg.mass
        k  = av @ self._I_approx
        x0 = torch.cat([c, l, k], dim=-1)

        site_pos = robot.data.site_pos_w[:, self._site_ids, :]
        r_lf = site_pos[:, self._lf_site_local_idx, :]
        r_rf = site_pos[:, self._rf_site_local_idx, :]

        x_ref, c_bar, schedule = self._build_refs_and_schedule(
            c=c, lv=lv, r_lf=r_lf, r_rf=r_rf
        )

        mpc_in = self._build_mpc_input(
            x0=x0, schedule=schedule, x_ref=x_ref, c_bar=c_bar
        )
        with torch.no_grad():
            mpc_out = self._mpc.solve(mpc_in)
        self._post_solve(mpc_out)

        self._grf_ref = torch.cat([
            mpc_out.u_star[:, 0:3],
            mpc_out.u_star[:, 6:9],
        ], dim=-1)

        self._u_prev = mpc_out.u_star.detach()

        x_pred = mpc_out.x_pred.detach()
        self._com_traj = x_pred[:, :, 0:3]
        l_traj = x_pred[:, :, 3:6]
        self._vel_traj = l_traj / cfg.mass
        k_pred = x_pred[:, :, 6:9]
        self._k_traj = k_pred
        self._kdot_traj = torch.zeros_like(k_pred)
        if N >= 2:
            self._kdot_traj[:, :-1] = (k_pred[:, 1:] - k_pred[:, :-1]) / cfg.mpc_dt
            self._kdot_traj[:, -1] = self._kdot_traj[:, -2]

        self._traj_step = 0
        self._interpolate_traj_refs()

        self._vis_com      = self._com_traj
        self._vis_ang_mom  = self._k_traj
        self._vis_r_lf     = schedule.r_LF.detach()
        self._vis_r_rf     = schedule.r_RF.detach()
        self._vis_sigma_lf = schedule.sigma[:, :, 0].detach()
        self._vis_sigma_rf = schedule.sigma[:, :, 1].detach()

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

    def _debug_vis_impl(self, visualizer: "DebugVisualizer") -> None:
        """Draw MPC horizon visualizations."""
        import numpy as np

        B = self.num_envs
        N = self.cfg.mpc_horizon
        env_indices = visualizer.get_env_indices(B)
        if not env_indices:
            return

        com_np  = self._vis_com.cpu().numpy()
        k_np    = self._vis_ang_mom.cpu().numpy()
        r_lf_np = self._vis_r_lf.cpu().numpy()
        r_rf_np = self._vis_r_rf.cpu().numpy()
        s_lf_np = self._vis_sigma_lf.cpu().numpy()
        s_rf_np = self._vis_sigma_rf.cpu().numpy()

        _EPS      = 1e-4
        _S_MIN    = 0.08
        _ALPHA    = 0.05
        _S_PERP   = 0.12
        _GAMMA    = 0.08
        _K_NODES  = [0, N // 2, N - 1]
        _K_ALPHAS = [0.75, 0.45, 0.15]

        for b in env_indices:
            for k in range(N):
                alpha = 0.85 * (1.0 - k / N) + 0.05

                visualizer.add_sphere(
                    center=com_np[b, k],
                    radius=0.03,
                    color=(0.85, 0.15, 0.15, alpha),
                    label=f"mpc_com_{b}_{k}",
                )
                if k < N - 1:
                    visualizer.add_cylinder(
                        start=com_np[b, k],
                        end=com_np[b, k + 1],
                        radius=0.012,
                        color=(0.85, 0.15, 0.15, alpha * 0.6),
                        label=f"mpc_com_line_{b}_{k}",
                    )

            for foot, r_np, s_np, rgb in (
                ("lf", r_lf_np, s_lf_np, (0.2, 0.5, 1.0)),
                ("rf", r_rf_np, s_rf_np, (1.0, 0.45, 0.15)),
            ) if self.cfg.vis_footplan else ():
                if s_np[b, 0] > 0.5:
                    pos = r_np[b, 0].copy()
                    pos[2] = max(pos[2], 0.01)
                    visualizer.add_sphere(
                        center=pos,
                        radius=0.045,
                        color=(*rgb, 0.9),
                        label=f"mpc_{foot}_cur_{b}",
                    )
                for k in range(1, N):
                    if s_np[b, k] > 0.5 and s_np[b, k - 1] < 0.5:
                        pos = r_np[b, k].copy()
                        pos[2] = max(pos[2], 0.01)
                        visualizer.add_sphere(
                            center=pos,
                            radius=0.045,
                            color=(*rgb, 0.5),
                            label=f"mpc_{foot}_td{k}_{b}",
                        )

            for ki, a_glyph in zip(_K_NODES, _K_ALPHAS):
                c_i = com_np[b, ki]
                k_i = k_np[b, ki]
                norm_k = float(np.linalg.norm(k_i))

                if norm_k > _EPS:
                    e1 = k_i / norm_k
                    ref = (np.array([0.0, 0.0, 1.0])
                           if abs(e1 @ np.array([0.0, 0.0, 1.0])) < 0.9
                           else np.array([1.0, 0.0, 0.0]))
                    e2_raw = ref - (e1 @ ref) * e1
                    e2 = e2_raw / np.linalg.norm(e2_raw)
                    e3 = np.cross(e1, e2)
                else:
                    e1 = np.array([1.0, 0.0, 0.0])
                    e2 = np.array([0.0, 1.0, 0.0])
                    e3 = np.array([0.0, 0.0, 1.0])

                R = np.stack([e1, e2, e3], axis=1)

                s1 = _S_MIN + _ALPHA * norm_k
                size = np.array([s1, _S_PERP, _S_PERP], dtype=np.float64)

                visualizer.add_ellipsoid(
                    center=c_i,
                    size=size,
                    mat=R,
                    color=(0.7, 0.3, 1.0, a_glyph),
                    label=f"mpc_k_ellipsoid_{b}_{ki}",
                )

                if norm_k > _EPS:
                    visualizer.add_arrow(
                        start=c_i,
                        end=c_i + _GAMMA * k_i,
                        color=(0.7, 0.3, 1.0, min(a_glyph + 0.2, 1.0)),
                        width=0.012,
                        label=f"mpc_k_arrow_{b}_{ki}",
                    )

    def _update_metrics(self) -> None:
        pass

    def _resample_command(self, env_ids: Tensor) -> None:
        """Reset MPC warm start for terminated / reset environments."""
        self._grf_ref[env_ids]             = 0.0
        self._u_prev[env_ids]              = 0.0
        self._com_traj[env_ids]            = 0.0
        self._vel_traj[env_ids]            = 0.0
        self._k_traj[env_ids]              = 0.0
        self._kdot_traj[env_ids]           = 0.0
        self._com_mpc_target[env_ids]      = 0.0
        self._com_vel_mpc_target[env_ids]  = 0.0
        self._k_mpc_target[env_ids]        = 0.0
        self._k_dot_mpc_target[env_ids]    = 0.0
        self._vis_ang_mom[env_ids]         = 0.0
        self._mpc.reset(env_ids)

