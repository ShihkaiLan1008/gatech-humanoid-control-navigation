"""H-LIP CoM reference riding the foothold windows (the joint_foot_hlip line).

``FootHoldHLIPCommand`` reads :class:`FootHoldCommand` every step and owns
nothing about footholds. In the STANCE-SITE yaw frame (x forward, y left,
origin = the stance foot site the foothold term froze at issue / stand entry)
it provides:

* a CoM-PROXY reference (pelvis + ``proxy_offset_x`` along the stance heading)
  on the H-LIP orbit of the CURRENT step — P1 sagittal with u = the landing x,
  P2 lateral with u = +-w (Xiong & Ames 2021, eqs. 5/8/19/22): DSP = cubic
  from the proxy state MEASURED at issue to the orbit's SSP entry (rate-
  limited); SSP = the LIP path from the proxy position MEASURED at lift-off
  to the orbit's pre-impact state (its initial velocity is the velocity that
  position needs to get there); standing = the proxy measured at stand entry,
  then the wind-up cubic over the last ``apa_time`` before lift-off;
* the orbital energy E = v_x^2 - lambda^2 x^2 of the reference (constant in
  SSP) and of the measured proxy;
* optionally a CLF (V, Vdot) over (x, y) position + velocity (``cfg.clf``),
  read by ``clf_reward`` / ``clf_decreasing_condition``.

Command tensor [B, 12] = ref (x, y, vx, vy) | measured (x, y, vx, vy) |
E_ref, E_meas | t_lift, t_td [s from now] — a critic-side quantity.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg

from ..clf import CLF
from .foot_hold import FootHoldCommand

if TYPE_CHECKING:
    from mjlab.entity import Entity
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.viewer.debug_visualizer import DebugVisualizer

_REF_VIS = (0.95, 0.15, 0.15, 1.0)
_MEAS_VIS = (0.15, 0.15, 0.95, 1.0)


def _hermite(p0, v0, p1, v1, s, T):
    """Cubic Hermite position and velocity at fraction ``s`` in [0, 1] of a
    segment of duration ``T`` with end states (p0, v0) -> (p1, v1)."""
    s2, s3 = s * s, s * s * s
    h00, h10, h01, h11 = 2 * s3 - 3 * s2 + 1, s3 - 2 * s2 + s, -2 * s3 + 3 * s2, s3 - s2
    d00, d10, d01, d11 = 6 * s2 - 6 * s, 3 * s2 - 4 * s + 1, -6 * s2 + 6 * s, 3 * s2 - 2 * s
    p = h00 * p0 + h10 * T * v0 + h01 * p1 + h11 * T * v1
    v = (d00 * p0 + d10 * T * v0 + d01 * p1 + d11 * T * v1) / T
    return p, v


class FootHoldHLIPCommand(CommandTerm):
    """H-LIP CoM-proxy reference + orbital energy for the foothold line."""

    cfg: FootHoldHLIPCommandCfg

    def __init__(self, cfg: FootHoldHLIPCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self.robot: Entity = env.scene[cfg.entity_name]
        B, dev = self.num_envs, self.device
        self.lam = math.sqrt(9.81 / cfg.com_height)

        # measured proxy state captured at the last anchor (issue / stand entry)
        self._m_pos = torch.zeros(B, 2, device=dev)
        self._m_vel = torch.zeros(B, 2, device=dev)
        self._prev_standing = torch.zeros(B, dtype=torch.bool, device=dev)
        self._prev_ssp = torch.zeros(B, dtype=torch.bool, device=dev)
        self._need_anchor = torch.ones(B, dtype=torch.bool, device=dev)

        self._cmd = torch.zeros(B, 12, device=dev)
        self.ref_pos = torch.zeros(B, 2, device=dev)
        self.ref_vel = torch.zeros(B, 2, device=dev)
        self.meas_pos = torch.zeros(B, 2, device=dev)
        self.meas_vel = torch.zeros(B, 2, device=dev)
        self.E_ref = torch.zeros(B, device=dev)
        self.E_meas = torch.zeros(B, device=dev)
        self.ssp_active = torch.zeros(B, dtype=torch.bool, device=dev)
        self.liftoff_event = torch.zeros(B, dtype=torch.bool, device=dev)
        self._stance_xy = torch.zeros(B, 2, device=dev)
        self._stance_yaw = torch.zeros(B, device=dev)
        # reference state at lift-off: the SSP flow starts here (= the orbit
        # entry when the wind-up caught up, else wherever the rate-limited
        # wind-up got to), and its E is E_ref for the swing
        self._ssp0_pos = torch.zeros(B, 2, device=dev)
        self._ssp0_vel = torch.zeros(B, 2, device=dev)
        self._E_ssp = torch.zeros(B, device=dev)

        self.clf: CLF | None = None
        self.v = torch.zeros(B, device=dev)
        self.vdot = torch.zeros(B, device=dev)
        if cfg.clf is not None:
            self.clf = CLF(
                sim_dt=env.step_dt, batch_size=B,
                pos_output_names=("com:x", "com:y"), vel_output_names=("com:x", "com:y"),
                Q_weights=cfg.clf["Q_weights"], R_weights=cfg.clf["R_weights"],
                device=dev, vdot_clamp=cfg.clf.get("vdot_clamp", 1.0e4),
            )
        for k in ("com_err", "vel_err", "E_err_ssp", "ssp_exit_x"):
            self.metrics[k] = torch.zeros(B, device=dev)
        self._orbit_mats_ready = False

    # ------------------------------------------------------------ orbit
    def _orbit_mats(self) -> None:
        """S2S matrices for the foothold timing (constants): A = e^{A_ss T_ssp}
        e^{A_ds T_dsp}, B = e^{A_ss T_ssp} [-1, 0]; P1/P2 fixed-point vectors."""
        fh = self._fh
        tm = fh.cfg.timing
        T_ssp = (1.0 - tm.lift_frac) * tm.step_period
        T_dsp = tm.lift_frac * tm.step_period
        lam = self.lam
        c, s = math.cosh(lam * T_ssp), math.sinh(lam * T_ssp)
        exp_ss = torch.tensor([[c, s / lam], [lam * s, c]], device=self.device)
        exp_ds = torch.tensor([[1.0, T_dsp], [0.0, 1.0]], device=self.device)
        A = exp_ss @ exp_ds
        Bv = exp_ss @ torch.tensor([-1.0, 0.0], device=self.device)
        I2 = torch.eye(2, device=self.device)
        inv2 = torch.linalg.inv(I2 - A @ A)
        # prev_pre = N1 u_in + N2 u_end ; our_pre = N1 u_end + N2 u_in (P2);
        # u_in = u_end reduces to the P1 fixed point (I - A)^-1 B u.
        self._N1 = inv2 @ A @ Bv
        self._N2 = inv2 @ Bv
        self.T_ssp, self.T_dsp = T_ssp, T_dsp
        self._orbit_mats_ready = True

    def _orbit(self, u_in: torch.Tensor, u_end: torch.Tensor):
        """Per-axis orbit states in the CURRENT stance frame. Shapes (B,).
        Returns (p_issue, v_issue, p_entry, v_entry, p_exit, v_exit): the
        state at issue (DSP start), at SSP entry and at touchdown."""
        N1, N2 = self._N1, self._N2
        prev_p = N1[0] * u_in + N2[0] * u_end
        prev_v = N1[1] * u_in + N2[1] * u_end
        p_issue = prev_p - u_in
        v_issue = prev_v
        p_entry = p_issue + v_issue * self.T_dsp
        v_entry = v_issue
        p_exit = N1[0] * u_end + N2[0] * u_in
        v_exit = N1[1] * u_end + N2[1] * u_in
        return p_issue, v_issue, p_entry, v_entry, p_exit, v_exit

    # ------------------------------------------------------------ frames
    @property
    def _fh(self) -> FootHoldCommand:
        return self._env.command_manager.get_term(self.cfg.foothold_name)

    def _proxy_state(self, stance_xy, yaw):
        """Measured CoM proxy (pos, vel) in the stance frame. Shapes (B, 2)."""
        c, s = yaw.cos(), yaw.sin()
        pel = self.robot.data.root_link_pos_w[:, :2]
        vel = self.robot.data.root_link_lin_vel_w[:, :2]
        d = pel - stance_xy
        x = c * d[:, 0] + s * d[:, 1] + self.cfg.proxy_offset_x
        y = -s * d[:, 0] + c * d[:, 1]
        vx = c * vel[:, 0] + s * vel[:, 1]
        vy = -s * vel[:, 0] + c * vel[:, 1]
        return torch.stack([x, y], -1), torch.stack([vx, vy], -1)

    def _to_world(self, p_local: torch.Tensor) -> torch.Tensor:
        """Stance-frame xy (B, 2) -> world xy (B, 2)."""
        c, s = self._stance_yaw.cos(), self._stance_yaw.sin()
        return torch.stack([self._stance_xy[:, 0] + c * p_local[:, 0] - s * p_local[:, 1],
                            self._stance_xy[:, 1] + s * p_local[:, 0] + c * p_local[:, 1]], -1)

    # ------------------------------------------------------------ hooks
    @property
    def command(self) -> torch.Tensor:
        return self._cmd

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        self._need_anchor[env_ids] = True
        self._prev_standing[env_ids] = False
        self._prev_ssp[env_ids] = False
        if self.clf is not None:
            self.clf.invalidate(env_ids)

    def _update_command(self) -> None:
        if not self._orbit_mats_ready:
            self._orbit_mats()
        fh = self._fh
        B, dev = self.num_envs, self.device
        rows = torch.arange(B, device=dev)
        standing = fh.standing
        swing = fh.foot_q[:, 0]
        stance = 1 - swing
        stance_xy = fh.ref_feet_pos_w[rows, stance, :2]
        yaw = fh.stance_yaw_w
        self._stance_xy, self._stance_yaw = stance_xy, yaw
        c, s = yaw.cos(), yaw.sin()

        # step geometry in the stance frame: landing x (= u_end,x) and signed w
        d = fh.target_pos_w_q[:, 0, :2] - stance_xy
        ell = c * d[:, 0] + s * d[:, 1]
        w = -s * d[:, 0] + c * d[:, 1]
        t_lift = fh.t_lift_q[:, 0]
        t_td = fh.t_td_q[:, 0]

        m_pos, m_vel = self._proxy_state(stance_xy, yaw)
        self.meas_pos, self.meas_vel = m_pos, m_vel

        # anchors: a new step was issued (t == 0) or a stand was entered
        issued = ~standing & (fh.t <= 0.0)
        entered = standing & ~self._prev_standing
        anchor = issued | entered | self._need_anchor
        if anchor.any():
            self._m_pos[anchor] = m_pos[anchor]
            self._m_vel[anchor] = torch.where(
                standing[anchor].unsqueeze(-1), torch.zeros_like(m_vel[anchor]), m_vel[anchor])
            self._need_anchor[anchor] = False
            if self.clf is not None:
                self.clf.invalidate(anchor.nonzero(as_tuple=False).squeeze(-1))
        self._prev_standing = standing.clone()

        # orbit of the current step: P1 in x (u = ell), P2 in y (u = -w, +w)
        _, _, xe, xve, xx, _ = self._orbit(ell, ell)
        _, _, ye, yve, yx, _ = self._orbit(-w, w)
        p_entry = torch.stack([xe, ye], -1)
        v_entry = torch.stack([xve, yve], -1)

        in_ssp = ~standing & (t_lift < 0.0)
        lam2 = self.lam ** 2

        # DSP / wind-up: cubic from the anchored measured state to the orbit
        # entry, rate-limited: a far-off anchor stretches the horizon instead
        # of demanding a dash, and the SSP flow then starts from wherever the
        # cubic is at lift-off.
        T_pre = torch.where(standing, torch.full_like(t_lift, self.cfg.apa_time),
                            torch.full_like(t_lift, self.T_dsp))
        gap = (p_entry - self._m_pos).norm(dim=-1)
        T_eff = torch.maximum(T_pre, gap / self.cfg.windup_vel_cap)
        frac = (1.0 - t_lift / T_eff).clamp(0.0, 1.0)
        p_pre, v_pre = _hermite(self._m_pos, self._m_vel, p_entry, v_entry,
                                frac.unsqueeze(-1), T_eff.unsqueeze(-1))

        self.liftoff_event = in_ssp & ~self._prev_ssp
        if self.liftoff_event.any():
            # SSP path: position re-anchored on the MEASURED proxy at lift-off,
            # exit pinned on the orbit's pre-impact state; the LIP boundary
            # value problem fixes the initial velocity = the velocity this
            # position needs to reach the exit (E_ref = its orbital energy).
            lo = self.liftoff_event
            p_exit = torch.stack([xx, yx], -1)
            chT, shT = math.cosh(self.lam * self.T_ssp), math.sinh(self.lam * self.T_ssp)
            p0 = m_pos
            v0 = self.lam * (p_exit - p0 * chT) / shT
            self._ssp0_pos[lo], self._ssp0_vel[lo] = p0[lo], v0[lo]
            self._E_ssp[lo] = (v0[lo, 0] ** 2 - lam2 * p0[lo, 0] ** 2)
        self._prev_ssp = in_ssp
        self.ssp_active = in_ssp

        # SSP: LIP flow from the lift-off reference state
        tau = (-t_lift).clamp(min=0.0)
        ch, sh = torch.cosh(self.lam * tau), torch.sinh(self.lam * tau)
        p_ssp = self._ssp0_pos * ch.unsqueeze(-1) + self._ssp0_vel / self.lam * sh.unsqueeze(-1)
        v_ssp = self._ssp0_pos * self.lam * sh.unsqueeze(-1) + self._ssp0_vel * ch.unsqueeze(-1)
        self.ref_pos = torch.where(in_ssp.unsqueeze(-1), p_ssp, p_pre)
        self.ref_vel = torch.where(in_ssp.unsqueeze(-1), v_ssp, v_pre)

        E_entry = xve ** 2 - lam2 * xe ** 2
        self.E_ref = torch.where(in_ssp, self._E_ssp, E_entry)
        self.E_meas = m_vel[:, 0] ** 2 - lam2 * m_pos[:, 0] ** 2

        if self.clf is not None:
            self.v, self.vdot = self.clf.compute(m_pos, self.ref_pos, m_vel, self.ref_vel)

        self._cmd = torch.cat([self.ref_pos, self.ref_vel, m_pos, m_vel,
                               self.E_ref.unsqueeze(-1), self.E_meas.unsqueeze(-1),
                               t_lift.clamp(-5.0, 5.0).unsqueeze(-1),
                               t_td.clamp(-5.0, 5.0).unsqueeze(-1)], -1)

    def _update_metrics(self) -> None:
        err = self.ref_pos - self.meas_pos
        self.metrics["com_err"] = err.norm(dim=-1)
        self.metrics["vel_err"] = (self.ref_vel - self.meas_vel).norm(dim=-1)
        e = (self.E_ref - self.E_meas).clamp(min=0.0)
        self.metrics["E_err_ssp"] = torch.where(self.ssp_active, e, torch.zeros_like(e))
        last = self.ssp_active & (self._fh.t_td_q[:, 0] <= self._env.step_dt + 1e-6)
        self.metrics["ssp_exit_x"] = torch.where(last, self.meas_pos[:, 0], self.metrics["ssp_exit_x"])

    # ------------------------------------------------------------ views
    @property
    def ref_com_pos_w(self) -> torch.Tensor:
        """Reference proxy position, world xy. Shape (B, 2), m."""
        return self._to_world(self.ref_pos)

    @property
    def com_pos_w(self) -> torch.Tensor:
        """Measured proxy position, world xy. Shape (B, 2), m."""
        return self._to_world(self.meas_pos)

    def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
        env_indices = visualizer.get_env_indices(self.num_envs)
        if not env_indices:
            return
        z = self.robot.data.root_link_pos_w[:, 2]
        ref = torch.cat([self.ref_com_pos_w, z.unsqueeze(-1)], -1).cpu().numpy()
        meas = torch.cat([self.com_pos_w, z.unsqueeze(-1)], -1).cpu().numpy()
        for b in env_indices:
            visualizer.add_sphere(center=ref[b], radius=0.03, color=_REF_VIS, label=f"hlip_ref_{b}")
            visualizer.add_sphere(center=meas[b], radius=0.02, color=_MEAS_VIS, label=f"hlip_meas_{b}")


@dataclass(kw_only=True)
class FootHoldHLIPCommandCfg(CommandTermCfg):
    """Configuration for :class:`FootHoldHLIPCommand`. Declare it AFTER the
    foothold term: command terms update in declaration order."""

    foothold_name: str = "foothold"
    entity_name: str = "robot"
    com_height: float = 0.678
    """LIP height z_c [m] (lambda = sqrt(g / z_c))."""
    proxy_offset_x: float = 0.04
    """Tracked point = pelvis + this along the stance heading [m] (CoM proxy)."""
    apa_time: float = 0.30
    """Wind-up duration before the first lift-off out of a stand [s]."""
    windup_vel_cap: float = 0.4
    """Mean proxy speed the wind-up cubic may demand [m/s]; a larger gap to
    the orbit entry stretches the wind-up horizon instead (the SSP flow then
    starts short of the orbit)."""
    clf: dict | None = None
    """None = no CLF. Else {"Q_weights": {"com:x": (q_pos, q_vel), "com:y": ...},
    "R_weights": {"com:x": r, "com:y": r}, "vdot_clamp": float}."""
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)
    debug_vis: bool = False

    def build(self, env: ManagerBasedRlEnv) -> FootHoldHLIPCommand:
        return FootHoldHLIPCommand(self, env)
