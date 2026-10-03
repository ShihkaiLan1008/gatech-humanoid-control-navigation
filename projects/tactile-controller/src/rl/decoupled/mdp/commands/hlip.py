"""H-LIP foothold + CoM-orbit planner on the twist and gait-clock commands.

robot_rl's ``HLIP`` step-to-step math under a per-env command term: one
landing pose per gait-clock event for the swing foot the clock names, fixed
in world frame until the next event, the orbit's CoM flow for that step, and
a hold while the clock is stopped.  Declare this term AFTER ``twist`` and
``gait``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Tuple

import numpy as np
import torch
from torch import Tensor

from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import sample_uniform

from rl.decoupled.mdp.commands.foot_placement import (
    _rel_swing_pose,
    _wrap,
    _yaw_from_quat_wxyz,
)
from rl.g1_assets.g1_constants import FEET_Y_DIST, PELVIS_KNEES_BENT_Z

if TYPE_CHECKING:
    from mjlab.entity import Entity
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.viewer.debug_visualizer import DebugVisualizer

_G = 9.81
_STANCE_VIS = (1.0, 0.6, 0.1, 0.9)
_TARGET_VIS = (0.9, 0.1, 0.1, 0.9)
_DCM_VIS = (0.2, 0.4, 1.0, 1.0)
_COM_REF_VIS = (0.1, 0.8, 0.2, 1.0)
_SOLE_HALF = (0.09, 0.03, 0.003)
"""Half-extents of the drawn sole box about the foot SITE, m."""


def _rot2(yaw: Tensor, v: Tensor) -> Tensor:
    """Planar vectors ``v`` (..., 2) rotated by ``yaw`` (...)."""
    c, s = torch.cos(yaw), torch.sin(yaw)
    return torch.stack([c * v[..., 0] - s * v[..., 1],
                        s * v[..., 0] + c * v[..., 1]], dim=-1)


class HLIP(torch.nn.Module):
    """Hybrid Linear Inverted Pendulum step-to-step orbits (robot_rl ``ref_gen.HLIP``).

    Args:
        z: LIP height, m.
        T_ds: double-support duration per step, s.
        T: step duration, s.
        y_nom: nominal lateral foot offset, m.
        device: torch device of every tensor this module owns.
    """

    def __init__(self, z: float, T_ds: float, T: float, y_nom: float, device: torch.device | str):
        super().__init__()
        self.grav = _G
        self.z = z
        self.y_nom = y_nom
        self.device = torch.device(device)
        self.lambda_ = torch.sqrt(torch.tensor(self.grav / self.z, device=self.device))
        self.A_ss = torch.tensor([[0.0, 1.0], [self.grav / self.z, 0.0]], device=self.device)
        self.A_ds = torch.tensor([[0.0, 1.0], [0.0, 0.0]], device=self.device)
        self.B_usw = torch.tensor([-1.0, 0.0], device=self.device)
        self.T_ds = T_ds
        self.T = T
        self._compute_s2s_matrices()

    def _compute_s2s_matrices(self) -> None:
        """Step-to-step (A, B): SSP flow over T - T_ds after the DSP glide over T_ds."""
        exp_ss = torch.matrix_exp(self.A_ss * (self.T - self.T_ds))
        exp_ds = torch.matrix_exp(self.A_ds * self.T_ds)
        self.A_s2s = exp_ss @ exp_ds
        self.B_s2s = exp_ss @ self.B_usw

    def _remap_for_init_stance_state(
        self, X_des_p1: Tensor, Y_des_p2: Tensor, Ux: Tensor, Uy: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Pre-impact orbit states -> post-impact (start-of-stance) states.

        Returns X0 (B, 2) and Y0 (B, 2, 2) with Y0[:, 0] the LEFT-stance and
        Y0[:, 1] the RIGHT-stance start state, each = the other side's
        pre-impact state shifted by the step that landed this foot."""
        Y_left = torch.cat([
            (Y_des_p2[:, 1, 0] - Uy[:, 1]).unsqueeze(-1),
            Y_des_p2[:, 1, 1].unsqueeze(-1)
        ], dim=-1)
        Y_right = torch.cat([
            (Y_des_p2[:, 0, 0] - Uy[:, 0]).unsqueeze(-1),
            Y_des_p2[:, 0, 1].unsqueeze(-1)
        ], dim=-1)
        X0 = torch.cat([
            (X_des_p1[:, 0] - Ux).unsqueeze(-1),
            X_des_p1[:, 1].unsqueeze(-1)
        ], dim=-1)
        return X0, torch.cat([Y_left.unsqueeze(1), Y_right.unsqueeze(1)], dim=1)

    def _compute_desire_com_trajectory(
        self, cur_time: Tensor, Xdesire: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """LIP flow of a start state (B, 2) at times ``cur_time`` (B,) or
        (B, M) since the step start. Returns (pos, vel) shaped like ``cur_time``."""
        x0, v0 = Xdesire[:, 0], Xdesire[:, 1]
        while x0.dim() < cur_time.dim():
            x0, v0 = x0.unsqueeze(-1), v0.unsqueeze(-1)
        lam = self.lambda_
        pos = x0 * torch.cosh(lam * cur_time) + (v0 / lam) * torch.sinh(lam * cur_time)
        vel = x0 * lam * torch.sinh(lam * cur_time) + v0 * torch.cosh(lam * cur_time)
        return pos, vel

    def compute_desired_orbit(
        self, vel: Tensor, T: float
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Periodic orbits for a stance-frame velocity ``vel`` (B, 2).

        Returns the P1 pre-impact state X- (B, 2), its step Ux (B,), the P2
        pre-impact states Y- (B, 2, 2) [left stance, right stance] and their
        steps Uy (B, 2) = vy*T -/+ y_nom."""
        B = vel.shape[0]
        U_des_p1 = vel[:, 0] * T
        eye = torch.eye(2, device=vel.device).unsqueeze(0).expand(B, -1, -1)
        X_des_p1 = torch.linalg.solve(eye - self.A_s2s, self.B_s2s * U_des_p1.unsqueeze(-1))
        U_left = vel[:, 1] * T - self.y_nom
        U_right = vel[:, 1] * T + self.y_nom
        A_squared = self.A_s2s @ self.A_s2s
        B_term = self.A_s2s @ self.B_s2s
        Y_left = torch.linalg.solve(
            eye - A_squared, B_term * U_left.unsqueeze(-1) + self.B_s2s * U_right.unsqueeze(-1))
        Y_right = torch.linalg.solve(
            eye - A_squared, B_term * U_right.unsqueeze(-1) + self.B_s2s * U_left.unsqueeze(-1))
        return X_des_p1, U_des_p1, torch.stack([Y_left, Y_right], dim=1), torch.stack([U_left, U_right], dim=1)

    def compute_orbit(self, T: float, cmd: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Orbits for ``cmd`` (B, >=2) stance-frame velocities; also stores the
        start-of-stance states as ``x_init`` (B, 2) and ``y_init`` (B, 2, 2)."""
        Xdes, Ux, Ydes, Uy = self.compute_desired_orbit(cmd[:, :2], T)
        self.x_init, self.y_init = self._remap_for_init_stance_state(Xdes, Ydes, Ux, Uy)
        return Xdes, Ux, Ydes, Uy


class HLIPCommandTerm(CommandTerm):
    """H-LIP foothold planner on the twist and gait-clock commands.

    ``command`` [B, 7] = swing one-hot (L, R) | countdown | target (x, y_m,
    z, yaw_m): the committed landing pose of the swing foot SITE in the
    MIRRORED stance-site yaw frame (y_m = s*y, yaw_m = s*yaw, s = +1 for a
    left swing), countdown = remaining fraction of the step.  Holding (clock
    stopped): countdown 0, target = the swing site's pose at hold entry.

    World-frame views: ``target_pos_w`` / ``target_yaw_w`` (swing site),
    ``ref_feet_pos_w`` / ``ref_feet_yaw`` in [L, R] order (stance = measured
    at the event) and ``ref_com_pos_w`` / ``ref_com_vel_w`` (the orbit flow
    of the step).  The twist is read in the pelvis frame; the CoM source is
    ``cfg.com_source``; the stance point is the foot SITE.
    """

    cfg: HLIPCommandTermCfg
    _env: ManagerBasedRlEnv

    def __init__(self, cfg: HLIPCommandTermCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self.robot: Entity = env.scene[cfg.entity_name]
        ids, _ = self.robot.find_sites(list(cfg.site_names), preserve_order=True)
        self._site_ids = torch.tensor(ids, dtype=torch.long, device=self.device)
        self._root_body_id = self.robot.data.indexing.root_body_id
        self._body_ids = self.robot.data.indexing.body_ids
        self._T = 0.5 * env.cfg.commands[cfg.gait_name].period
        self.hlip = HLIP(cfg.com_height_ref, 0.0, self._T, cfg.stance_width, self.device)

        B, dev = self.num_envs, self.device
        self.swing = torch.zeros(B, dtype=torch.long, device=dev)
        self.standing = torch.ones(B, dtype=torch.bool, device=dev)
        self._pending = torch.ones(B, dtype=torch.bool, device=dev)
        self._countdown = torch.zeros(B, device=dev)
        self.target_m = torch.zeros(B, 4, device=dev)
        self._target_pos_w = torch.zeros(B, 3, device=dev)
        self._target_yaw_w = torch.zeros(B, device=dev)
        self._ref_feet_pos = torch.zeros(B, 2, 3, device=dev)
        self._ref_feet_yaw = torch.zeros(B, 2, device=dev)
        self._x_init = torch.zeros(B, 2, device=dev)
        self._y_init = torch.zeros(B, 2, 2, device=dev)
        self._orbit_u = torch.zeros(B, 3, device=dev)
        self._hold_com = torch.zeros(B, 3, device=dev)
        for k in ("td_err", "plan_x", "plan_y", "plan_yaw", "plan_fb", "standing"):
            self.metrics[k] = torch.zeros(B, device=dev)

    # ------------------------------------------------------------ views
    @property
    def command(self) -> Tensor:
        """[B, 7] = swing one-hot (L, R) | countdown | target (x, y_m, z, yaw_m)."""
        one_hot = torch.nn.functional.one_hot(self.swing, 2).float()
        return torch.cat([one_hot, self._countdown.unsqueeze(-1), self.target_m], dim=-1)

    @property
    def target_pos_w(self) -> Tensor:
        """Committed landing position of the swing SITE, world frame. Shape (B, 3), m."""
        return self._target_pos_w

    @property
    def target_yaw_w(self) -> Tensor:
        """Committed landing yaw of the swing SITE, world frame. Shape (B,), rad."""
        return self._target_yaw_w

    @property
    def ref_feet_pos_w(self) -> Tensor:
        """Reference foot SITE positions [L, R], world frame. Shape (B, 2, 3), m.

        Swing = the committed target, stance = measured at the event; holding:
        both measured at hold entry."""
        return self._ref_feet_pos

    @property
    def ref_feet_yaw(self) -> Tensor:
        """Reference foot SITE yaws [L, R], world frame. Shape (B, 2), rad."""
        return self._ref_feet_yaw

    @property
    def ref_com_pos_w(self) -> Tensor:
        """Orbit CoM reference, world frame. Shape (B, 3), m.

        Stepping: the orbit flow from the step's start state at the clock's
        time-in-step; holding: mid-feet at ``com_height_ref``."""
        return self._orbit_ref()[0]

    @property
    def ref_com_vel_w(self) -> Tensor:
        """Orbit CoM velocity reference, world frame. Shape (B, 3), m/s. Zero while holding."""
        return self._orbit_ref()[1]

    @property
    def orbit_lam(self) -> Tensor:
        """LIP frequency of the orbit. Shape (B,), 1/s."""
        return self.hlip.lambda_.expand(self.num_envs)

    @property
    def orbit_step(self) -> Tensor:
        """Nominal step of the committed orbit (ell_x, ell_y, s), stance yaw
        frame, s = +1 for a left swing. Shape (B, 3)."""
        return self._orbit_u

    def _stance_init(self) -> Tuple[Tensor, Tensor]:
        """Start-of-stance orbit states of the committed step: sagittal (B, 2)
        and lateral (B, 2), stance frame."""
        rows = torch.arange(self.num_envs, device=self.device)
        return self._x_init, self._y_init[rows, 1 - self.swing]

    def orbit_flow(self, t: Tensor) -> Tuple[Tensor, Tensor]:
        """Committed orbit CoM (position, velocity) at times ``t`` (B, M) after
        the step start, relative to the stance site in its yaw frame.
        Shapes (B, M, 2), m and m/s. Not clamped to the step: callers pass
        t <= T and re-anchor beyond it."""
        x0, y0 = self._stance_init()
        px, vx = self.hlip._compute_desire_com_trajectory(t, x0)
        py, vy = self.hlip._compute_desire_com_trajectory(t, y0)
        return torch.stack([px, py], dim=-1), torch.stack([vx, vy], dim=-1)

    def next_orbit_init(self, T: float) -> Tuple[Tensor, Tensor]:
        """Start-of-stance orbit states of the step AFTER the committed one
        (same command, the other foot), relative to the committed target foot
        in its yaw frame. Shapes (B, 2) each. ``T`` must be the clock's step time."""
        rows = torch.arange(self.num_envs, device=self.device)
        return self._x_init, self._y_init[rows, self.swing]

    def _orbit_ref(self) -> Tuple[Tensor, Tensor]:
        gait = self._env.command_manager.get_term(self.cfg.gait_name)
        t = (gait.phase % math.pi) / math.pi * self._T
        x0, y0 = self._stance_init()
        px, vx = self.hlip._compute_desire_com_trajectory(t, x0)
        py, vy = self.hlip._compute_desire_com_trajectory(t, y0)
        rel_p = torch.stack([px, py], dim=-1)
        rel_v = torch.stack([vx, vy], dim=-1)
        rows = torch.arange(self.num_envs, device=self.device)
        stance = 1 - self.swing
        p_st = self._ref_feet_pos[rows, stance, :2]
        psi_st = self._ref_feet_yaw[rows, stance]
        z = torch.full_like(t, self.cfg.com_height_ref).unsqueeze(-1)
        pos = torch.cat([p_st + _rot2(psi_st, rel_p), z], dim=-1)
        vel = torch.cat([_rot2(psi_st, rel_v), torch.zeros_like(z)], dim=-1)
        hold = self.standing.unsqueeze(-1)
        return torch.where(hold, self._hold_com, pos), torch.where(hold, torch.zeros_like(vel), vel)

    @property
    def feet_site_pos_w(self) -> Tensor:
        """Measured foot SITE positions [L, R], world frame. Shape (B, 2, 3), m."""
        return self.robot.data.site_pos_w[:, self._site_ids]

    @property
    def feet_site_yaw(self) -> Tensor:
        """Measured foot SITE yaws [L, R], world frame. Shape (B, 2), rad."""
        return _yaw_from_quat_wxyz(self.robot.data.site_quat_w[:, self._site_ids])

    @property
    def com_pos_w(self) -> Tensor:
        """CoM position per ``cfg.com_source``, world frame. Shape (B, 3), m.

        ``"subtree"``: whole-body CoM (mujoco root subtree_com); ``"pelvis"``:
        the pelvis BODY's own CoM (root_com), a proxy that ignores limb pose."""
        if self.cfg.com_source == "pelvis":
            return self.robot.data.root_com_pos_w
        return self.robot.data.data.subtree_com[:, self._root_body_id]

    @property
    def com_vel_w(self) -> Tensor:
        """CoM velocity per ``cfg.com_source``, world frame. Shape (B, 3), m/s.

        ``"subtree"``: mass-weighted body CoM velocities (= whole-body CoM
        velocity); ``"pelvis"``: the pelvis body CoM velocity."""
        if self.cfg.com_source == "pelvis":
            return self.robot.data.root_com_lin_vel_w
        v = self.robot.data.body_com_lin_vel_w
        m = self.robot.data.model.body_mass[:, self._body_ids]
        return (m.unsqueeze(-1) * v).sum(dim=1) / m.sum(dim=1, keepdim=True)

    @property
    def dcm_w(self) -> Tensor:
        """Divergent component of motion xi = c + v/lambda, world frame. Shape (B, 2), m."""
        c, v = self.com_pos_w, self.com_vel_w
        return c[:, :2] + v[:, :2] / self.hlip.lambda_

    # ------------------------------------------------------------ lifecycle
    def _resample_command(self, env_ids: Tensor) -> None:
        # Body poses are stale here (reset events wrote them, no forward yet):
        # the first update after reset initialises from fresh kinematics.
        self._pending[env_ids] = True
        self.standing[env_ids] = True
        self._countdown[env_ids] = 0.0

    def _update_command(self) -> None:
        gait = self._env.command_manager.get_term(self.cfg.gait_name)
        phase, moving = gait.phase, gait.moving
        # schedule convention: left stance on (0, pi) -> the right foot swings
        swing_clock = (phase < math.pi).long()
        pending = self._pending
        enter_hold = ~moving & (~self.standing | pending)
        event = moving & (self.standing | pending | (swing_clock != self.swing))
        ended = event & ~self.standing & ~pending
        if ended.any():
            ids = torch.where(ended)[0]
            rows = torch.arange(len(ids), device=self.device)
            foot = self.feet_site_pos_w[ids][rows, self.swing[ids], :2]
            self.metrics["td_err"][ids] = (foot - self._target_pos_w[ids, :2]).norm(dim=-1)
        if enter_hold.any():
            ids = torch.where(enter_hold)[0]
            self._enter_hold(ids, swing_clock[ids])
        if event.any():
            ids = torch.where(event)[0]
            self._plan(ids, swing_clock[ids], phase[ids])
        self._pending[:] = False
        frac = (phase % math.pi) / math.pi
        self._countdown = torch.where(self.standing, torch.zeros_like(frac), 1.0 - frac)
        self.metrics["standing"] = self.standing.float()

    def _update_metrics(self) -> None:
        pass

    # ------------------------------------------------------------ planning
    def _enter_hold(self, ids: Tensor, swing: Tensor) -> None:
        """Freeze both feet where they are; the swing site's pose is the target."""
        rows = torch.arange(len(ids), device=self.device)
        feet_p = self.feet_site_pos_w[ids]
        feet_yaw = self.feet_site_yaw[ids]
        self.swing[ids] = swing
        self.standing[ids] = True
        self.target_m[ids] = _rel_swing_pose(feet_p, feet_yaw, swing)
        ref = feet_p.clone()
        ref[..., 2] = self.cfg.foot_z
        self._ref_feet_pos[ids] = ref
        self._ref_feet_yaw[ids] = feet_yaw
        self._target_pos_w[ids] = ref[rows, swing]
        self._target_yaw_w[ids] = feet_yaw[rows, swing]
        z = torch.full((len(ids), 1), self.cfg.com_height_ref, device=self.device)
        self._hold_com[ids] = torch.cat([feet_p[:, :, :2].mean(dim=1), z], dim=-1)

    def _plan(self, ids: Tensor, swing: Tensor, phase: Tensor) -> None:
        """Commit the swing site's landing pose and the orbit start states for
        the step that just began.

        Stance-site yaw frame, s = +1 for a left swing, T = step time,
        E = exp(lambda T), E_rem = exp(lambda (T - t)), r = DCM - stance site,
        b_k = the orbit start state's DCM:
            u = (Ux, Uy[stance]) + k*(E_rem*r - E*b_k)     k = feedback_gain
        then the per-step perturbation and the coverage-box clamp."""
        cfg = self.cfg
        K = len(ids)
        T = self._T
        rows = torch.arange(K, device=self.device)
        stance = 1 - swing
        s = torch.where(swing == 0, 1.0, -1.0)
        site_p = self.feet_site_pos_w[ids]
        site_yaw = self.feet_site_yaw[ids]
        p_st = site_p[rows, stance, :2]
        psi_st = site_yaw[rows, stance]

        tw = self._env.command_manager.get_command(cfg.twist_name)[ids]
        dpsi = tw[:, 2] * T
        psi_b = _yaw_from_quat_wxyz(self.robot.data.root_link_quat_w[ids])
        ell = _rot2(psi_b - psi_st + 0.5 * dpsi, tw[:, :2] * T)

        # H-LIP orbit of the NOMINAL step, stance-frame velocity ell / T
        _, Ux, _, Uy = self.hlip.compute_orbit(T, ell / T)
        x_init, y_init = self.hlip.x_init, self.hlip.y_init
        self._x_init[ids] = x_init
        self._y_init[ids] = y_init
        self._orbit_u[ids] = torch.stack([ell[:, 0], ell[:, 1], s], dim=-1)
        u_orbit = torch.stack([Ux, Uy[rows, stance]], dim=-1)

        # one-step DCM correction toward the orbit (off at feedback_gain 0)
        lam = self.hlip.lambda_
        com = self.com_pos_w[ids]
        xi = com[:, :2] + self.com_vel_w[ids][:, :2] / lam
        r = _rot2(-psi_st, xi - p_st)
        y0 = y_init[rows, stance]
        b_k = torch.stack([x_init[:, 0] + x_init[:, 1] / lam, y0[:, 0] + y0[:, 1] / lam], dim=-1)
        E = math.exp(float(lam) * T)
        t_in = (phase % math.pi) / math.pi * T
        E_rem = torch.exp(lam * (T - t_in))
        fb = cfg.feedback_gain * (E_rem.unsqueeze(-1) * r - E * b_k)
        u = u_orbit + fb
        yaw_rel = dpsi

        px, py = cfg.perturb_xy
        u = u + torch.stack([sample_uniform(-px, px, (K,), self.device),
                             sample_uniform(-py, py, (K,), self.device)], dim=-1)
        yaw_rel = yaw_rel + sample_uniform(-cfg.perturb_yaw, cfg.perturb_yaw, (K,), self.device)

        w = cfg.stance_width
        x = u[:, 0].clamp(*cfg.landing_x_range)
        w_abs = (s * u[:, 1]).clamp(w + cfg.landing_dy_range[0], w + cfg.landing_dy_range[1])
        yaw_rel = yaw_rel.clamp(*cfg.landing_yaw_range)
        y = s * w_abs

        land_xy = p_st + _rot2(psi_st, torch.stack([x, y], dim=-1))
        land_yaw = _wrap(psi_st + yaw_rel)
        z = torch.full_like(x, cfg.foot_z)

        self.swing[ids] = swing
        self.standing[ids] = False
        self.target_m[ids] = torch.stack([x, w_abs, torch.zeros_like(x), s * yaw_rel], dim=-1)
        self._target_pos_w[ids] = torch.stack([land_xy[:, 0], land_xy[:, 1], z], dim=-1)
        self._target_yaw_w[ids] = land_yaw
        self._ref_feet_pos[ids[rows], swing] = self._target_pos_w[ids]
        self._ref_feet_pos[ids[rows], stance] = torch.stack([p_st[:, 0], p_st[:, 1], z], dim=-1)
        self._ref_feet_yaw[ids[rows], swing] = land_yaw
        self._ref_feet_yaw[ids[rows], stance] = psi_st
        self.metrics["plan_x"][ids] = x
        self.metrics["plan_y"][ids] = y
        self.metrics["plan_yaw"][ids] = yaw_rel
        self.metrics["plan_fb"][ids] = fb.norm(dim=-1)

    # ------------------------------------------------------------- debug vis
    def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
        """ORANGE sole + small frame = the stance reference, RED sole + large
        frame = the committed target, BLUE sphere = the DCM on the ground,
        GREEN sphere = the orbit CoM reference."""
        env_indices = visualizer.get_env_indices(self.num_envs)
        if not env_indices:
            return
        rows = torch.arange(self.num_envs, device=self.device)
        stance = 1 - self.swing
        h = {
            k: v.cpu().numpy() for k, v in dict(
                tgt=self._target_pos_w, tgt_yaw=self._target_yaw_w,
                st=self._ref_feet_pos[rows, stance], st_yaw=self._ref_feet_yaw[rows, stance],
                xi=self.dcm_w, com_ref=self.ref_com_pos_w,
            ).items()
        }
        for b in env_indices:
            self._draw_foot(visualizer, h["st"][b], h["st_yaw"][b], _STANCE_VIS, 0.08,
                            f"footplan_stance_{b}")
            self._draw_foot(visualizer, h["tgt"][b], h["tgt_yaw"][b], _TARGET_VIS, 0.15,
                            f"footplan_target_{b}")
            visualizer.add_sphere(
                center=np.array([h["xi"][b, 0], h["xi"][b, 1], 0.02]), radius=0.02,
                color=_DCM_VIS, label=f"footplan_dcm_{b}")
            visualizer.add_sphere(center=h["com_ref"][b], radius=0.03, color=_COM_REF_VIS,
                                  label=f"footplan_com_ref_{b}")

    @staticmethod
    def _draw_foot(visualizer: DebugVisualizer, pos: np.ndarray, yaw: float,
                   color: tuple, scale: float, label: str) -> None:
        c, s = math.cos(yaw), math.sin(yaw)
        R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        visualizer.add_box(
            center=pos + np.array([0.0, 0.0, _SOLE_HALF[2]]), size=np.array(_SOLE_HALF),
            mat=R, color=color, label=f"{label}_sole")
        visualizer.add_frame(position=pos, rotation_matrix=R, scale=scale, label=label)


@dataclass(kw_only=True)
class HLIPCommandTermCfg(CommandTermCfg):
    """Configuration for :class:`HLIPCommandTerm`."""

    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)
    """Never resampled mid-episode; the clock drives every step."""

    entity_name: str = "robot"
    """Scene entity the feet and CoM are read from."""

    twist_name: str = "twist"
    """Twist command term (vx, vy, wz in the pelvis frame), declared before this one."""

    gait_name: str = "gait"
    """Gait-clock command term (phase, period, moving), declared before this one."""

    site_names: tuple[str, str] = ("left_foot", "right_foot")
    """Foot SITE names in [L, R] order — the stance point and the landing point."""

    stance_width: float = FEET_Y_DIST
    """Nominal lateral distance between the foot sites w (y_nom), m."""

    feedback_gain: float = 0.0
    """Scale on the DCM correction: 0 = the orbit step only (robot_rl / MARCH),
    1 = one-step deadbeat on the measured DCM (Narrow Path)."""

    com_source: Literal["subtree", "pelvis"] = "pelvis"
    """CoM state fed to the DCM correction and the vis: ``"subtree"`` =
    whole-body CoM (exact in sim), ``"pelvis"`` = the pelvis body's CoM."""

    com_height_ref: float = PELVIS_KNEES_BENT_Z
    """Constant LIP height, m: sets lambda, the orbit CoM z and the hold CoM z."""

    perturb_xy: tuple[float, float] = (0.05, 0.05)
    """Half-widths of the per-step uniform target perturbation (x, y), stance frame, m."""

    perturb_yaw: float = 0.35
    """Half-width of the per-step uniform target yaw perturbation, rad."""

    landing_x_range: tuple[float, float] = (-0.20, 0.25)
    """Clamp on the target x relative to the stance site, stance frame, m."""

    landing_dy_range: tuple[float, float] = (-0.08, 0.18)
    """Clamp on the target lateral offset from ``stance_width`` (absolute, swing side), m."""

    landing_yaw_range: tuple[float, float] = (-0.30, 0.30)
    """Clamp on the target yaw relative to the stance site, rad."""

    foot_z: float = 0.0
    """Height of the landing target (foot SITE on the contact plane), m."""

    def build(self, env: ManagerBasedRlEnv) -> HLIPCommandTerm:
        return HLIPCommandTerm(self, env)
