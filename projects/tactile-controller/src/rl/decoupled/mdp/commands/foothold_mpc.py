"""Foothold-conditioned centroidal MPC command (docs/foothold_command_redesign.md).

``FootholdMPCCommand`` overrides ONE hook of :class:`LocoMPCCommand`
(``_build_refs_and_schedule``): instead of the twist-integrated straight-line
x_ref + Raibert schedule, it QUERIES the :class:`FootHoldCommand`'s
windowed timeline (2026-08-23 redesign — no gait clock exists in this line):

* **schedule**: for each horizon node time t_k, ask the queue which entry's
  window covers it (that foot swings; none => both feet in contact) and how
  many entries have closed (feet advanced along the committed chain).  A gap
  node is just double support — the old hold/walk-out special cases are
  gone.  Future stance positions are the queue's world targets VERBATIM (no
  Raibert, no state feedback); node 0 uses the MEASURED feet (rule); sigma
  NEVER gates on measured contact.
* **x_ref** (``cfg.x_ref_mode``): ``"hlip"`` (default) rolls the H-LIP
  orbit of each queue entry out over the horizon (Xiong & Ames 2021 P1
  sagittal / P2 lateral, in the entry's stance-site yaw frame): SSP = the
  LIP flow from the orbit's SSP entry, DSP = the orbit's constant-velocity
  glide, deeper pre-lift (gaps, stands, absent entries) = a wind-up cubic
  from the feet midpoint at rest.  Purely queue-driven — no measured CoM
  enters; momentum rows carry the analytic velocity.  ``"centroid"`` keeps
  the CCC recipe (RA-L 2022): piecewise support-centroid position with
  constant average-velocity momentum, smoothing left to the QP.
* **c_bar**: the reference CoM path — the moment arms are linearized about
  the demanded motion.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from mjlab.managers import SceneEntityCfg
import torch
from torch import Tensor

from rl.decoupled.cd_mpc.contact_schedule import ContactSchedule
from rl.decoupled.cd_mpc import cd as cd_mpc
from .foot_hold import FootHoldCommand
from .foot_hold_hlip import _hermite
from .locomotion_manip_mpc import _cone_aabb
from .locomotion_mpc import LocoMPCCommand, LocoMPCCommandCfg, _quat_to_rot

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.sensor import ContactSensor
    from .ee_contact import EeContactTargetCommand



class FootholdMPCCommand(LocoMPCCommand):
    """Centroidal MPC whose schedule and reference come from foothold windows."""

    cfg: FootholdMPCCommandCfg

    def _build_refs_and_schedule(
        self, *, c: Tensor, lv: Tensor, r_lf: Tensor, r_rf: Tensor
    ) -> tuple[Tensor, Tensor, ContactSchedule]:
        cfg = self.cfg
        B = self.num_envs
        N = cfg.mpc_horizon
        dt = cfg.mpc_dt
        device = self.device
        robot = self._robot

        fh: FootHoldCommand = self._env.command_manager.get_term(
            cfg.foothold_name
        )
        K = fh.cfg.queue_len

        # ── window queries over N+1 node times (x_ref needs one extra) ──────
        t_k = fh.t_now.unsqueeze(1) + dt * torch.arange(
            N + 1, device=device, dtype=torch.float32
        ).unsqueeze(0)                                     # (B, N+1)
        tk = t_k.unsqueeze(2)                              # (B, N+1, 1)
        tl = fh.t_lift_q.unsqueeze(1)                      # (B, 1, K)
        td = fh.t_td_q.unsqueeze(1)
        covered = (tk >= tl) & (tk < td)                   # (B, N+1, K)
        swinging = covered.any(dim=-1)                     # (B, N+1)
        left_swings = (covered & (fh.foot_q == 0).unsqueeze(1)).any(dim=-1)
        closed = (td <= tk).long().sum(dim=-1)             # (B, N+1) in 0..K

        # ── feet state after j closed entries: entry j-1 lands foot_q[j-1] ──
        feetL = r_lf.unsqueeze(1).repeat(1, K + 1, 1)
        feetR = r_rf.unsqueeze(1).repeat(1, K + 1, 1)
        for j in range(1, K + 1):
            tgt = fh.target_pos_w_q[:, j - 1, :]
            land_left = (fh.foot_q[:, j - 1] == 0).unsqueeze(-1)
            tgtL = tgt.clone()
            tgtL[:, 2] = feetL[:, j - 1, 2]  # keep the measured foot-site z
            tgtR = tgt.clone()
            tgtR[:, 2] = feetR[:, j - 1, 2]
            feetL[:, j] = torch.where(land_left, tgtL, feetL[:, j - 1])
            feetR[:, j] = torch.where(land_left, feetR[:, j - 1], tgtR)

        gather3 = closed.clamp(max=K).unsqueeze(-1).expand(B, N + 1, 3)
        feetL_k = torch.gather(feetL, 1, gather3)          # (B, N+1, 3)
        feetR_k = torch.gather(feetR, 1, gather3)

        # ── sigma: a node covered by no window = both feet in contact ───────
        sw = swinging[:, :N]
        lsw = left_swings[:, :N]
        sigma = torch.zeros(B, N, 4, device=device)
        sigma[:, :, 0] = (~(sw & lsw)).float()
        sigma[:, :, 1] = (~(sw & ~lsw)).float()

        site_quat = robot.data.site_quat_w[:, self._site_ids, :]
        R_lf = _quat_to_rot(site_quat[:, self._lf_site_local_idx, :])
        R_rf = _quat_to_rot(site_quat[:, self._rf_site_local_idx, :])

        schedule = ContactSchedule(
            sigma=sigma,
            r_LF=feetL_k[:, :N].contiguous(),
            r_RF=feetR_k[:, :N].contiguous(),
            r_LH=torch.zeros(B, N, 3, device=device),
            r_RH=torch.zeros(B, N, 3, device=device),
            R_LF=R_lf.contiguous(),
            R_RF=R_rf.contiguous(),
        )

        # ── x_ref ───────────────────────────────────────────────────────────
        x_ref = torch.zeros(B, N + 1, 9, device=device)
        x_ref[:, :, 2] = cfg.com_height

        if cfg.x_ref_mode == "hlip":
            pos_w, vel_w = self._hlip_ref(fh, t_k, closed, feetL, feetR,
                                          feetL_k, feetR_k)
            x_ref[:, :, 0:2] = pos_w
            x_ref[:, :, 3] = vel_w[:, :, 0] * cfg.mass
            x_ref[:, :, 4] = vel_w[:, :, 1] * cfg.mass
        else:
            # crude piecewise support centroid (gap = midpoint), constant
            # average-velocity momentum (a piecewise position's derivative is
            # an impulse train and must NOT be the momentum reference).
            stance_pos = torch.where(left_swings.unsqueeze(-1), feetR_k, feetL_k)
            mid = 0.5 * (feetL_k + feetR_k)
            cent = torch.where(swinging.unsqueeze(-1), stance_pos, mid)[:, :, 0:2]
            x_ref[:, :, 0:2] = cent
            v_avg = (cent[:, -1] - cent[:, 0]) / (N * dt)  # (B, 2)
            x_ref[:, :, 3] = (v_avg[:, 0] * cfg.mass).unsqueeze(1)
            x_ref[:, :, 4] = (v_avg[:, 1] * cfg.mass).unsqueeze(1)

        I_zz = float(self._I_approx[2, 2])
        dyaw = fh.target_yaw_w_q[:, 0] - fh.stance_yaw_w
        head_dyaw = torch.atan2(dyaw.sin(), dyaw.cos())
        T_head = (fh.t_td_q[:, 0] - fh.t_lift_q[:, 0]).clamp(min=1e-6)
        wz_cmd = torch.where(
            fh.swing_active, head_dyaw / T_head, torch.zeros_like(head_dyaw)
        )
        x_ref[:, :, 8] = (I_zz * wz_cmd).unsqueeze(1)

        self._x_ref_dbg = x_ref  # probe/vis access

        c_bar = x_ref[:, :, 0:3].clone()

        return x_ref, c_bar, schedule

    # ------------------------------------------------------------ H-LIP ref
    def _hlip_consts(self, fh: FootHoldCommand) -> None:
        """S2S constants for the foothold timing: N1/N2 fixed-point vectors
        (Xiong & Ames 2021), lambda, T_ssp/T_dsp. Cached on first use."""
        tm = fh.cfg.timing
        T_ssp = (1.0 - tm.lift_frac) * tm.step_period
        T_dsp = tm.lift_frac * tm.step_period
        lam = math.sqrt(9.81 / self.cfg.com_height)
        c, s = math.cosh(lam * T_ssp), math.sinh(lam * T_ssp)
        exp_ss = torch.tensor([[c, s / lam], [lam * s, c]], device=self.device)
        exp_ds = torch.tensor([[1.0, T_dsp], [0.0, 1.0]], device=self.device)
        A = exp_ss @ exp_ds
        Bv = exp_ss @ torch.tensor([-1.0, 0.0], device=self.device)
        inv2 = torch.linalg.inv(torch.eye(2, device=self.device) - A @ A)
        self._N1, self._N2 = inv2 @ A @ Bv, inv2 @ Bv
        self._lam, self._T_dsp = lam, T_dsp

    def _hlip_ref(
        self, fh: FootHoldCommand, t_k: Tensor, closed: Tensor,
        feetL: Tensor, feetR: Tensor, feetL_k: Tensor, feetR_k: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """H-LIP orbit CoM reference at the horizon node times.

        Per queue entry: the P1 (sagittal) / P2 (lateral) orbit of its
        (l, w) in its stance-site yaw frame; nodes map to SSP flow /
        DSP glide / midpoint wind-up by their time to that entry's lift.
        Purely queue-driven — no measured CoM enters. Returns world-frame
        position and velocity, each shape (B, N+1, 2)."""
        cfg = self.cfg
        K = fh.cfg.queue_len
        if not hasattr(self, "_N1"):
            self._hlip_consts(fh)
        lam, T_dsp = self._lam, self._T_dsp
        N1, N2 = self._N1, self._N2

        # per-segment stance frame; feet chain index j = before entry j lands
        stance_left = (fh.foot_q == 1).unsqueeze(-1)              # (B, K, 1)
        p_st = torch.where(stance_left, feetL[:, :K, :2], feetR[:, :K, :2])
        yaw_st = torch.cat(
            [fh.stance_yaw_w.unsqueeze(1), fh.target_yaw_w_q[:, : K - 1]], dim=1)
        cy, sy = yaw_st.cos(), yaw_st.sin()
        d = fh.target_pos_w_q[:, :, :2] - p_st
        ell = cy * d[:, :, 0] + sy * d[:, :, 1]
        w = -sy * d[:, :, 0] + cy * d[:, :, 1]

        # orbit boundary states: P1 in x (u_in = u_end = ell),
        # P2 in y (u_in = -w, u_end = w); issue = DSP start, entry = SSP start
        p_iss = torch.stack([(N1[0] + N2[0]) * ell - ell,
                             (N2[0] - N1[0]) * w + w], dim=-1)    # (B, K, 2)
        v_iss = torch.stack([(N1[1] + N2[1]) * ell,
                             (N2[1] - N1[1]) * w], dim=-1)
        p_ent = p_iss + v_iss * T_dsp
        v_ent = v_iss

        # gather per node by segment (entries whose td passed are closed)
        seg = closed.clamp(max=K - 1)                             # (B, N+1)
        g2 = seg.unsqueeze(-1).expand(-1, -1, 2)
        tl_n = torch.gather(fh.t_lift_q, 1, seg)
        td_n = torch.gather(fh.t_td_q, 1, seg)
        p_st_n = torch.gather(p_st, 1, g2)
        cy_n, sy_n = torch.gather(cy, 1, seg), torch.gather(sy, 1, seg)
        p_ent_n = torch.gather(p_ent, 1, g2)
        v_ent_n = torch.gather(v_ent, 1, g2)
        p_iss_n = torch.gather(p_iss, 1, g2)
        v_iss_n = torch.gather(v_iss, 1, g2)

        # wind-up anchor: feet midpoint in the segment's stance frame, at rest
        mid = 0.5 * (feetL_k + feetR_k)[:, :, 0:2] - p_st_n
        m_loc = torch.stack([cy_n * mid[..., 0] + sy_n * mid[..., 1],
                             -sy_n * mid[..., 0] + cy_n * mid[..., 1]], dim=-1)

        tau = (t_k - tl_n).unsqueeze(-1)                          # (B, N+1, 1)
        # SSP: LIP flow from the entry state, frozen at the pre-impact state
        # for nodes past the window's end (chain exhausted)
        tau_s = torch.minimum(tau.clamp(min=0.0), (td_n - tl_n).unsqueeze(-1))
        ch, sh = torch.cosh(lam * tau_s), torch.sinh(lam * tau_s)
        p_ssp = p_ent_n * ch + v_ent_n / lam * sh
        v_ssp = p_ent_n * lam * sh + v_ent_n * ch
        # DSP: the orbit's constant-velocity glide over the last T_dsp
        p_dsp = p_ent_n + v_ent_n * tau.clamp(min=-T_dsp, max=0.0)
        # deeper pre-lift (gap / stand / 1e9-sentinel entry): cubic wind-up
        # from (midpoint, rest) to the issue state; the clamp holds the
        # midpoint until windup_time before the glide begins
        s_wu = ((tau + T_dsp + cfg.windup_time) / cfg.windup_time).clamp(0.0, 1.0)
        p_wu, v_wu = _hermite(m_loc, torch.zeros_like(v_iss_n), p_iss_n,
                              v_iss_n, s_wu, cfg.windup_time)
        in_ssp, in_dsp = tau >= 0.0, tau >= -T_dsp
        p_loc = torch.where(in_ssp, p_ssp, torch.where(in_dsp, p_dsp, p_wu))
        v_loc = torch.where(in_ssp, v_ssp, torch.where(in_dsp, v_ent_n, v_wu))

        pos_w = p_st_n + torch.stack(
            [cy_n * p_loc[..., 0] - sy_n * p_loc[..., 1],
             sy_n * p_loc[..., 0] + cy_n * p_loc[..., 1]], dim=-1)
        vel_w = torch.stack(
            [cy_n * v_loc[..., 0] - sy_n * v_loc[..., 1],
             sy_n * v_loc[..., 0] + cy_n * v_loc[..., 1]], dim=-1)
        return pos_w, vel_w


@dataclass(kw_only=True)
class FootholdMPCCommandCfg(LocoMPCCommandCfg):
    """Config for :class:`FootholdMPCCommand`."""

    foothold_name: str = "foothold"
    """The FootHold command term whose windows drive schedule + reference."""

    x_ref_mode: str = "hlip"
    """CoM reference shape: "hlip" = per-entry H-LIP orbit rollout with
    analytic momentum; "centroid" = piecewise support centroid with
    average-velocity momentum (the CCC recipe)."""

    windup_time: float = 0.30
    """Pre-lift wind-up duration [s] from the feet midpoint at rest to the
    orbit's DSP entry — shapes stands, gaps and walk-outs ("hlip" only)."""

    mpc_impl: str = "cd"

    vis_footplan: bool = False
    """The foothold term's own arrows show placement — the plan spheres off
    by default here (CoM / momentum glyphs stay)."""

    vel_cmd_name: str = "twist"
    """Unused by the foothold hook (kept for the base cfg contract)."""

    def build(self, env: "ManagerBasedRlEnv") -> FootholdMPCCommand:
        return FootholdMPCCommand(self, env)



# ============================================================
# Foothold Manip MPC
# ============================================================

class FootholdManipMPCCommand(FootholdMPCCommand):
    """Foothold Manipulation MPC Command."""
    cfg: FootholdManipMPCCommandCfg

    def __init__(self, cfg: FootholdManipMPCCommandCfg, env: "ManagerBasedRlEnv") -> None:
        super().__init__(cfg, env)
        B = env.num_envs
        device = env.device

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

        def _sensor_list(names) -> list[ContactSensor]:
            if cfg.gate_mode != "sensor":
                return []
            if isinstance(names, str):
                names = (names,)
            return [env.scene[n] for n in (names or ())]

        self._lhand_sensors = _sensor_list(cfg.lhand_sensor_name)
        self._rhand_sensors = _sensor_list(cfg.rhand_sensor_name)

        self._last_contact_step = torch.full(
            (B, 2), -(10**6), dtype=torch.long, device=device
        )
        self._gate = torch.zeros(B, 2, device=device)
        self._hand_cmd_w = torch.zeros(B, 6, device=device)
        self._hand_u_star = torch.zeros(B, 6, device=device)

    def _build_mpc(self, cfg: FootholdManipMPCCommandCfg, env: "ManagerBasedRlEnv") -> None:
        if cfg.mpc_impl != "cd":
            raise ValueError(
                "FootholdManipMPCCommand requires mpc_impl='cd' — the vendored "
                "centroidal_mpc has no hand slots."
            )
        self._mpc = cd_mpc.CentroidalMPC(
            cd_mpc.MPCConfig(
                N=cfg.mpc_horizon,
                dt=cfg.mpc_dt,
                mass=cfg.mass,
                locomanipulation=True,
                mu_hand=cfg.mu_hand,
                f_hand_max=cfg.f_hand_max,
                R_f_hand=cfg.R_f_hand,
            ),
            device=env.device,
        )
        self._mpc_input_cls = cd_mpc.MPCInput
        self._extra_mpc_kwargs = {}  # hand kwargs are built per solve

    # ── hand layer ──────────────────────────────────────────────────────────

    def _ee_terms(self) -> tuple["EeContactTargetCommand", "EeContactTargetCommand"]:
        mgr = self._env.command_manager
        return (
            mgr.get_term(self.cfg.ee_command_names[0]),
            mgr.get_term(self.cfg.ee_command_names[1]),
        )

    def _per_step_update(self) -> None:
        """Debounce bookkeeping must run at the POLICY rate, not the solve rate."""
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

    def _debounced_contact(self) -> Tensor:
        """(B, 2) float: measured surface contact with dropout debounce."""
        return (
            (self._step_count - self._last_contact_step)
            <= self.cfg.contact_debounce_steps
        ).float()

    def _build_mpc_input(self, *, x0: Tensor, schedule, x_ref: Tensor, c_bar: Tensor):
        cfg = self.cfg
        B = self.num_envs
        N = cfg.mpc_horizon
        device = self.device

        hand_pos = self._robot.data.site_pos_w[:, self._hand_site_ids, :]
        r_lh = hand_pos[:, self._lh_local_idx, :]
        r_rh = hand_pos[:, self._rh_local_idx, :]

        lh_term, rh_term = self._ee_terms()
        mask_cmd = torch.stack(
            [lh_term.contact_mask, rh_term.contact_mask], dim=-1
        )
        if cfg.gate_mode == "command":
            # No contact sensors (virtual-spring task): gate on the command's
            # own state — window live ∧ not phantom (whole-window semantics).
            live = torch.stack(
                [(t.phantom < 0.5).float() for t in (lh_term, rh_term)], dim=-1
            )
            hand_contact = mask_cmd * live
        else:
            hand_contact = mask_cmd * self._debounced_contact()
        self._gate = hand_contact.detach().clone()

        f_cmd_lh_w = lh_term.force_cmd_w  # force ON SURFACE
        f_cmd_rh_w = rh_term.force_cmd_w
        self._hand_cmd_w = torch.cat([f_cmd_lh_w, f_cmd_rh_w], dim=-1).detach()

        cap_lh = (cfg.f_hand_headroom * lh_term.force_cmd_mag).clamp(max=cfg.f_hand_max)
        cap_rh = (cfg.f_hand_headroom * rh_term.force_cmd_mag).clamp(max=cfg.f_hand_max)
        lb_lh, ub_lh = _cone_aabb(lh_term.press_dir_w, cfg.mu_hand, cap_lh)
        lb_rh, ub_rh = _cone_aabb(rh_term.press_dir_w, cfg.mu_hand, cap_rh)
        hand_force_lb = torch.cat([lb_lh, lb_rh], dim=-1)
        hand_force_ub = torch.cat([ub_lh, ub_rh], dim=-1)

        u_ref = torch.zeros(B, N, 18, device=device)
        u_ref[:, :, 12:15] = -f_cmd_lh_w.unsqueeze(1)
        u_ref[:, :, 15:18] = -f_cmd_rh_w.unsqueeze(1)

        return cd_mpc.MPCInput(
            x0=x0,
            schedule=schedule,
            x_ref=x_ref,
            u_ref=u_ref,
            u_prev=self._u_prev,
            other_body_force=torch.zeros(B, 3, device=device),
            other_body_torque=torch.zeros(B, 3, device=device),
            c_bar=c_bar,
            r_LH=r_lh,
            r_RH=r_rh,
            hand_contact=hand_contact,
            hand_force_lb=hand_force_lb,
            hand_force_ub=hand_force_ub,
        )

    def _post_solve(self, mpc_out) -> None:
        self._hand_u_star = mpc_out.u_star[:, 12:18].detach()

    def _resample_command(self, env_ids: Tensor) -> None:
        super()._resample_command(env_ids)
        self._last_contact_step[env_ids] = -(10**6)
        self._gate[env_ids] = 0.0
        self._hand_cmd_w[env_ids] = 0.0
        self._hand_u_star[env_ids] = 0.0


@dataclass(kw_only=True)
class FootholdManipMPCCommandCfg(FootholdMPCCommandCfg):
    """Config for :class:`FootholdManipMPCCommand`."""

    mpc_impl: str = "cd"
    
    ee_command_names: tuple[str, str] = ("left_ee_cmd", "right_ee_cmd")
    """The two :class:`EeContactTargetCommand` terms carrying mask + force."""

    hand_site_names: tuple[str, str] = ("left_tactile_site", "right_tactile_site")
    """Hand contact points fed to the QP = the tactile APEX sites."""

    gate_mode: str = "sensor"
    """QP hand gate source: "sensor" = mask ∧ debounced measured contact (the
    cube task); "command" = mask ∧ not phantom (the virtual-spring task,
    which has no contact sensors)."""

    lhand_sensor_name: str | tuple[str, ...] = "lhand_cube_contact"
    rhand_sensor_name: str | tuple[str, ...] = "rhand_cube_contact"

    mu_hand: float = 0.6
    f_hand_max: float = 50.0
    f_hand_headroom: float = 1.25
    R_f_hand: float = 1e-1
    """At 1e-2 the QP re-optimizes the hand as a balance actuator (measured on
    the source stack — hand force tracking dies)."""

    contact_debounce_steps: int = 5
    """Measured-contact dropout tolerance in POLICY steps (0.1 s at 50 Hz)."""

    def build(self, env: "ManagerBasedRlEnv") -> FootholdManipMPCCommand:
        return FootholdManipMPCCommand(self, env)
    