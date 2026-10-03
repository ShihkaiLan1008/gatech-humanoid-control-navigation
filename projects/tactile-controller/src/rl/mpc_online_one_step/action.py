"""OneStepQpAction — the 1-step QP + WBC as an mjlab ActionTerm (Stage A1).

Policy action (18-D) = [delta_a (6): base lin/ang accel residual,
delta_q (12): leg joint-position residuals]. Per policy step (50 Hz):
gait clock -> Raibert footholds -> weight-shift fractions -> folded-PD refs ->
batched 1-step QP (ADMM) -> stance torque J^T(-u) + swing spline/IK targets.
Per physics substep the targets are re-written to the IdealPd leg actuators
as (q_des, qd_des, tau_ff) — RAMBO's three-channel write — and the arm/waist
Builtin position actuators are held at the default pose (A1: arms nominal).

The QP hand slots exist but are pinned (A1); Stage A2 unpins them, adds palm
targets/forces and the reaction-force application.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import mujoco
import numpy as np
import torch

from mjlab.managers.action_manager import ActionTerm, ActionTermCfg

from .clock import BatchedGait, GaitParams, raibert_foothold, swing_pos_vel
from .kin import LEG_JOINTS, G1LegKin, quat_to_rot, rot_z
from .qp1 import OneStepParams, OneStepQP

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

FOOT_SITES = ("left_foot", "right_foot")
HIP_LINKS = ("left_hip_pitch_link", "right_hip_pitch_link")
ANKLE_LINKS = ("left_ankle_roll_link", "right_ankle_roll_link")


def whole_body_inertia_and_mass(
    mj_model: mujoco.MjModel, body_ids: list[int]
) -> tuple[np.ndarray, float]:
    """Whole-body inertia about the CoM in the model's default configuration."""
    d = mujoco.MjData(mj_model)
    if mj_model.nkey > 0:
        d.qpos[:] = mj_model.key_qpos[0]
    mujoco.mj_forward(mj_model, d)
    ids = np.asarray(body_ids)
    m = mj_model.body_mass[ids]
    com = (d.xipos[ids] * m[:, None]).sum(axis=0) / m.sum()
    inertia = np.zeros((3, 3))
    for k, b in enumerate(ids):
        R = d.ximat[b].reshape(3, 3)
        I_local = np.diag(mj_model.body_inertia[b])
        dvec = d.xipos[b] - com
        inertia += R @ I_local @ R.T + m[k] * (
            np.dot(dvec, dvec) * np.eye(3) - np.outer(dvec, dvec)
        )
    return inertia, float(m.sum())


@dataclass(kw_only=True)
class OneStepQpActionCfg(ActionTermCfg):
    delta_a_lin_scale: float = 2.0  # m/s^2 per unit action
    delta_a_ang_scale: float = 5.0  # rad/s^2 per unit action
    delta_q_scale: float = 0.15  # rad per unit action (RAMBO value)
    gait_freq_scale: float = 0.3  # +-30% clock rate (g1_wall 9-D channel)
    apex_scale: float = 0.04  # m, swing-apex residual (g1_wall 9-D channel)
    raw_clip: float = 5.0  # hard clip on raw policy actions (blow-up guard)
    ema_alpha: float = 0.8  # EMA on the joint-position residual
    enable_residuals: bool = True  # False -> pure model-based (smoke tests)
    gait: GaitParams = field(default_factory=GaitParams)
    qp: OneStepParams = field(default_factory=OneStepParams)
    stance_kp: float = 40.0
    stance_kd: float = 4.0
    swing_kp: float = 250.0
    swing_kd: float = 8.0
    lpf_cutoff_hz: float = 6.0
    command_name: str = "twist"

    def build(self, env: ManagerBasedRlEnv) -> OneStepQpAction:
        return OneStepQpAction(self, env)


class OneStepQpAction(ActionTerm):
    cfg: OneStepQpActionCfg

    def __init__(self, cfg: OneStepQpActionCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        dev = self.device
        B = self.num_envs
        ent = self._entity
        self.step_dt = env.step_dt

        # --- joint / site / body indexing (order: [left(6), right(6)])
        leg_names = LEG_JOINTS[0] + LEG_JOINTS[1]
        ids, names = ent.find_joints(leg_names, preserve_order=True)
        assert list(names) == leg_names, f"leg joint resolution mismatch: {names}"
        self.leg_ids = torch.as_tensor(ids, dtype=torch.long, device=dev)
        all_ids = list(range(ent.data.default_joint_pos.shape[1]))
        self.other_ids = torch.as_tensor(
            [i for i in all_ids if i not in set(ids)], dtype=torch.long, device=dev
        )
        site_ids, site_names = ent.find_sites(FOOT_SITES, preserve_order=True)
        assert list(site_names) == list(FOOT_SITES)
        self.foot_site_ids = torch.as_tensor(site_ids, dtype=torch.long, device=dev)
        hip_ids, _ = ent.find_bodies(HIP_LINKS, preserve_order=True)
        self.hip_body_ids = torch.as_tensor(hip_ids, dtype=torch.long, device=dev)
        ankle_ids, _ = ent.find_bodies(ANKLE_LINKS, preserve_order=True)
        self.ankle_body_ids = torch.as_tensor(ankle_ids, dtype=torch.long, device=dev)

        # --- model constants
        mj_model = env.sim.mj_model
        global_body_ids = list(ent.indexing.body_ids.cpu().numpy())
        inertia, mass = whole_body_inertia_and_mass(mj_model, global_body_ids)
        cfg.qp.mass = mass
        cfg.qp.inertia_body = inertia
        cfg.qp.dt = self.step_dt
        body_mass = mj_model.body_mass[np.asarray(global_body_ids)]
        self.body_mass = torch.as_tensor(body_mass, dtype=torch.float32, device=dev)
        self.total_mass = mass

        # --- modules
        self.kin = G1LegKin(dev)
        self.gait = BatchedGait(cfg.gait, B, dev)
        self.qp = OneStepQP(cfg.qp, B, dev)

        # --- buffers
        self._raw = torch.zeros(B, self.action_dim, device=dev)
        self.delta_q = torch.zeros(B, 12, device=dev)
        self.foothold = torch.zeros(B, 2, 3, device=dev)
        self.liftoff = torch.zeros(B, 2, 3, device=dev)
        self.prev_contact = torch.ones(B, 2, dtype=torch.bool, device=dev)
        self.anchor_xy = torch.zeros(B, 2, device=dev)
        self.anchor_yaw = torch.zeros(B, device=dev)
        self.height_ref = torch.full((B,), 0.72, device=dev)
        self.com_offset = torch.zeros(B, 2, device=dev)
        self._pending_init = torch.ones(B, dtype=torch.bool, device=dev)
        self._omega_f = torch.zeros(B, 3, device=dev)
        self._v_f = torch.zeros(B, 3, device=dev)
        self._lpf_alpha = 1.0 - float(
            np.exp(-2 * np.pi * cfg.lpf_cutoff_hz * self.step_dt)
        )
        # 50 Hz-latched kinematic plan, consumed per substep by apply_actions
        self._swing_pos = torch.zeros(B, 12, device=dev)
        self._swing_vel = torch.zeros(B, 12, device=dev)
        self._stance_pos = torch.zeros(B, 12, device=dev)
        self._sched = torch.ones(B, 2, dtype=torch.bool, device=dev)
        self._swing_ph = torch.full((B, 2), -1.0, device=dev)
        self._eff_target = torch.zeros(B, 12, device=dev)
        # 50 Hz-latched QP inputs (solved at 100 Hz in apply_actions)
        self._substep = 0
        self._delta_a = None
        self._apex_r = torch.zeros(B, device=dev)
        self._w_frac = torch.full((B, 2), 0.5, device=dev)
        self._pos_ref_xy = torch.zeros(B, 2, device=dev)
        self._v_cmd_w = torch.zeros(B, 3, device=dev)
        self._wz_cmd = torch.zeros(B, device=dev)

        # qfrc_bias exposure is optional in the data bridge; fall back to zero
        try:
            _ = ent.data._joint_dof_field("qfrc_bias")
            self._has_bias = True
        except Exception:
            self._has_bias = False
            print("[g1_onestep] WARNING: qfrc_bias not exposed; tau_ff omits bias")

        # IdealPd leg actuator groups (runtime stance/swing gain switching).
        # Legs span three groups; map each group's local columns to our
        # [left(6), right(6)] ordering.
        self._leg_act_maps: list[tuple[object, torch.Tensor, torch.Tensor]] = []
        leg_col = {n: i for i, n in enumerate(leg_names)}
        for act in list(getattr(ent, "actuators", []) or []):
            tnames = list(
                getattr(act, "_target_names", None)
                or getattr(act, "target_names", None)
                or []
            )
            pairs = [(j, leg_col[n]) for j, n in enumerate(tnames) if n in leg_col]
            if pairs and hasattr(act, "set_gains"):
                loc = torch.as_tensor(
                    [p[0] for p in pairs], dtype=torch.long, device=dev
                )
                mine = torch.as_tensor(
                    [p[1] for p in pairs], dtype=torch.long, device=dev
                )
                self._leg_act_maps.append((act, loc, mine))

        self.last_u = torch.zeros(B, 18, device=dev)
        self.last_a_des = torch.zeros(B, 3, device=dev)

    # ------------------------------------------------------------------ api
    @property
    def action_dim(self) -> int:
        # [delta_a(6), delta_q_legs(12), gait_freq(1), swing_apex(1)]
        return 20

    @property
    def raw_action(self) -> torch.Tensor:
        return self._raw

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._raw[env_ids] = 0.0
        self.delta_q[env_ids] = 0.0
        self.gait.reset(env_ids)
        self.qp.reset(env_ids)
        self._pending_init[env_ids] = True
        self._omega_f[env_ids] = 0.0
        self._v_f[env_ids] = 0.0
        self.prev_contact[env_ids] = True

    # ------------------------------------------------------------------ state
    def _fetch_state(self):
        ent = self._entity
        d = ent.data
        body_com = d.body_com_pose_w[:, :, :3]  # [B, nb, 3]
        w = (self.body_mass / self.total_mass).view(1, -1, 1)
        com = (body_com * w).sum(dim=1)
        body_com_vel = d.body_com_vel_w[:, :, :3]
        v_com = (body_com_vel * w).sum(dim=1)
        root_quat = d.root_link_quat_w
        omega = d.root_link_ang_vel_w
        q_leg = d.joint_pos[:, self.leg_ids]
        qd_leg = d.joint_vel[:, self.leg_ids]
        return com, v_com, root_quat, omega, q_leg, qd_leg

    @staticmethod
    def _quat_to_rpy(q: torch.Tensor) -> torch.Tensor:
        w, x, y, z = q.unbind(-1)
        roll = torch.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        pitch = torch.asin((2 * (w * y - z * x)).clamp(-1, 1))
        yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        return torch.stack([roll, pitch, yaw], dim=-1)

    def _measured_contact(self) -> torch.Tensor:
        try:
            sensor = self._env.scene["feet_ground_contact"]
            found = sensor.data.found
            return found.reshape(self.num_envs, -1)[:, :2] > 0
        except Exception:
            return torch.ones(self.num_envs, 2, dtype=torch.bool, device=self.device)

    # ------------------------------------------------------------------ core
    def process_actions(self, actions: torch.Tensor) -> None:
        cfg = self.cfg
        B = self.num_envs
        dev = self.device
        actions = actions.clamp(-cfg.raw_clip, cfg.raw_clip)
        self._raw[:] = actions

        if cfg.enable_residuals:
            delta_a = torch.cat(
                [
                    (actions[:, 0:3] * cfg.delta_a_lin_scale).clamp(
                        -2 * cfg.delta_a_lin_scale, 2 * cfg.delta_a_lin_scale
                    ),
                    (actions[:, 3:6] * cfg.delta_a_ang_scale).clamp(
                        -2 * cfg.delta_a_ang_scale, 2 * cfg.delta_a_ang_scale
                    ),
                ],
                dim=-1,
            )
            dq_new = actions[:, 6:18] * cfg.delta_q_scale
            dq_new = dq_new.clamp(-2 * cfg.delta_q_scale, 2 * cfg.delta_q_scale)
            self.delta_q = cfg.ema_alpha * dq_new + (1 - cfg.ema_alpha) * self.delta_q
            freq_r = actions[:, 18].clamp(-1.0, 1.0) * cfg.gait_freq_scale
            apex_r = actions[:, 19].clamp(-1.0, 1.0) * cfg.apex_scale
        else:
            delta_a = None
            self.delta_q[:] = 0.0
            freq_r = torch.zeros(B, device=dev)
            apex_r = torch.zeros(B, device=dev)
        self._apex_r = apex_r

        com, v_com, root_quat, omega, q_leg, qd_leg = self._fetch_state()
        rpy = self._quat_to_rpy(root_quat)
        yaw = rpy[:, 2]
        # (omega/v LPF states are maintained at 100 Hz in apply_actions)

        foot_pos = self._entity.data.site_pose_w[:, self.foot_site_ids, :3]
        hip_pos = self._entity.data.body_link_pos_w[:, self.hip_body_ids]

        # --- deferred per-env init (after reset teleports)
        init = self._pending_init
        if init.any():
            self.anchor_xy[init] = com[init, :2]
            self.anchor_yaw[init] = yaw[init]
            self.height_ref[init] = com[init, 2]
            self.com_offset[init] = com[init, :2] - foot_pos[init, :, :2].mean(dim=1)
            self.foothold[init] = foot_pos[init]
            self.liftoff[init] = foot_pos[init]
            self._pending_init[init] = False

        cmd = self._env.command_manager.get_command(cfg.command_name)
        v_cmd_body = torch.cat([cmd[:, 0:2], torch.zeros(B, 1, device=dev)], dim=-1)
        wz_cmd = cmd[:, 2]

        # --- clock (freq residual + reactive stepping on measured velocity)
        self.gait.update(self.step_dt, cmd, rate_scale=freq_r, v_meas_xy=self._v_f[:, :2])
        contact_sched = self.gait.contact_state(shift=0.5 * self.step_dt)
        swing_ph = self.gait.swing_phase()
        w_frac = self.gait.load_fractions(shift=0.5 * self.step_dt)

        # liftoff bookkeeping
        lifting = self.prev_contact & ~contact_sched
        if lifting.any():
            self.liftoff = torch.where(lifting.unsqueeze(-1), foot_pos, self.liftoff)
        self.prev_contact = contact_sched.clone()

        # --- anchors advance with the command, clamped near the CoM
        Rz3 = rot_z(self.anchor_yaw)
        v_cmd_w = torch.einsum("bij,bj->bi", Rz3, v_cmd_body)
        self.anchor_xy += v_cmd_w[:, :2] * self.step_dt
        self.anchor_yaw += wz_cmd * self.step_dt
        self.anchor_xy = self.anchor_xy.clamp(com[:, :2] - 0.10, com[:, :2] + 0.10)
        dyaw = (
            torch.remainder(self.anchor_yaw - yaw + torch.pi, 2 * torch.pi) - torch.pi
        )
        self.anchor_yaw = yaw + dyaw.clamp(-0.3, 0.3)

        # --- Raibert footholds for swinging feet (axis-split gains, yaw frame)
        yaw_rot2 = Rz3[:, :2, :2]
        gains = torch.tensor([cfg.gait.raibert_x, cfg.gait.raibert_y], device=dev)
        t_stance = 2 * cfg.gait.t_ds + cfg.gait.t_swing
        for f in range(2):
            swinging = swing_ph[:, f] >= 0.0
            if not swinging.any():
                continue
            fh_xy = raibert_foothold(
                hip_pos[:, f, :2],
                self._v_f[:, :2],
                v_cmd_w[:, :2],
                t_stance,
                gains,
                yaw_rot2,
            )
            rel = torch.einsum(
                "bij,bj->bi", yaw_rot2.transpose(1, 2), fh_xy - hip_pos[:, f, :2]
            )
            rel[:, 0] = rel[:, 0].clamp(-0.25, 0.25)
            if f == 0:
                rel[:, 1] = rel[:, 1].clamp(-0.01, 0.15)
            else:
                rel[:, 1] = rel[:, 1].clamp(-0.15, 0.01)
            fh_xy = hip_pos[:, f, :2] + torch.einsum("bij,bj->bi", yaw_rot2, rel)
            fh = torch.cat([fh_xy, torch.zeros(B, 1, device=dev)], dim=-1)
            self.foothold[:, f] = torch.where(
                swinging.unsqueeze(-1), fh, self.foothold[:, f]
            )

        # --- references: support centroid while quasi-static, anchor otherwise
        centroid = (foot_pos[:, :, :2] * w_frac.unsqueeze(-1)).sum(dim=1)
        pos_ref_xy = torch.where(
            self.gait.active.unsqueeze(-1),
            self.anchor_xy,
            centroid + self.com_offset,
        )

        # latch the solve inputs that are clock/command-shaped; the QP itself
        # runs at 100 Hz inside apply_actions on FRESH state — solving at the
        # 50 Hz policy rate doubled the feedback phase lag vs Stage 0 and
        # produced an 8 Hz pitch/bounce limit cycle (measured during bring-up)
        self._delta_a = delta_a
        self._w_frac = w_frac
        self._pos_ref_xy = pos_ref_xy
        self._v_cmd_w = v_cmd_w
        self._wz_cmd = wz_cmd
        self._substep = 0

        # --- swing kinematic targets (50 Hz, ZOH between policy steps)
        R_wb = quat_to_rot(root_quat)
        base_pos = self._entity.data.root_link_pos_w
        base_vel = self._entity.data.root_link_lin_vel_w
        measured = self._measured_contact()

        nominal_leg = self._entity.data.default_joint_pos[:, self.leg_ids]
        pos_t = torch.zeros(B, 12, device=dev)
        vel_t = torch.zeros(B, 12, device=dev)

        for f in range(2):
            sl = slice(6 * f, 6 * f + 6)
            q_f = q_leg[:, sl]
            ph_f = swing_ph[:, f]
            late_land = contact_sched[:, f] & ~measured[:, f] & (ph_f < 0.0)

            s = ph_f.clamp(min=0.0)
            p_sw, v_sw = swing_pos_vel(
                self.liftoff[:, f],
                self.foothold[:, f],
                cfg.gait.swing_height + self._apex_r,
                s,
                cfg.gait.t_swing,
                cfg.gait.touchdown_overdrive,
            )
            # late landing: press straight down at the foothold
            p_lat = self.foothold[:, f].clone()
            p_lat[:, 2] -= cfg.gait.touchdown_overdrive
            p_des_w = torch.where(late_land.unsqueeze(-1), p_lat, p_sw)
            v_des_w = torch.where(late_land.unsqueeze(-1), torch.zeros_like(v_sw), v_sw)
            p_des_b = torch.einsum(
                "bij,bj->bi", R_wb.transpose(1, 2), p_des_w - base_pos
            )
            R_des_b = torch.bmm(R_wb.transpose(1, 2), rot_z(yaw))
            q_ik = self.kin.ik_6dof(f, q_f, p_des_b, R_des_b, iters=2)
            # velocity feedforward (linear rows, base-motion compensated)
            J_p = self.kin.jac(f, q_f)[:, :3]
            v_rel_b = torch.einsum(
                "bij,bj->bi", R_wb.transpose(1, 2), v_des_w - base_vel
            )
            JJt = torch.bmm(J_p, J_p.transpose(1, 2))
            JJt[:, range(3), range(3)] += 1e-3
            qd_ik = (
                torch.bmm(
                    J_p.transpose(1, 2), torch.linalg.solve(JJt, v_rel_b.unsqueeze(-1))
                )
                .squeeze(-1)
                .clamp(-8.0, 8.0)
            )

            pos_t[:, sl] = q_ik + self.delta_q[:, sl]
            vel_t[:, sl] = qd_ik

        self._swing_pos = pos_t
        self._swing_vel = vel_t
        self._stance_pos = nominal_leg + self.delta_q
        self._sched = contact_sched
        self._swing_ph = swing_ph

    def apply_actions(self) -> None:
        """Per physics substep: fresh J/bias/gating so the 160 N-scale tau_ff
        is never 20 ms stale (a 50 Hz ZOH here rang the contact sensor into
        on/off flicker and toppled the robot — measured during bring-up).
        The QP re-solves every 2nd substep (100 Hz, Stage-0 parity)."""
        cfg = self.cfg
        B = self.num_envs
        dev = self.device
        ent = self._entity

        root_quat = ent.data.root_link_quat_w
        R_wb = quat_to_rot(root_quat)
        q_leg = ent.data.joint_pos[:, self.leg_ids]
        foot_pos = ent.data.site_pose_w[:, self.foot_site_ids, :3]
        ankle_pos = ent.data.body_link_pos_w[:, self.ankle_body_ids]

        if self._substep % 2 == 0:
            com, v_com, _, omega, _, _ = self._fetch_state()
            rpy = self._quat_to_rpy(root_quat)
            alpha = 1.0 - float(
                np.exp(-2 * np.pi * cfg.lpf_cutoff_hz * self.step_dt / 2)
            )
            self._omega_f += alpha * (omega - self._omega_f)
            self._v_f += alpha * (v_com - self._v_f)
            u, a_des = self.qp.solve(
                com=com,
                v=self._v_f,
                omega=self._omega_f,
                rpy=rpy,
                r_feet=foot_pos - com.unsqueeze(1),
                contact=self._sched,
                w_frac=self._w_frac,
                pos_ref_xy=self._pos_ref_xy,
                height_ref=self.height_ref,
                yaw_ref=self.anchor_yaw,
                v_cmd_w=self._v_cmd_w,
                wz_cmd=self._wz_cmd,
                delta_a=self._delta_a,
            )
            self.last_u = u
            self.last_a_des = a_des
        self._substep += 1
        measured = self._measured_contact()
        # sensor-flicker fallback: a foot whose sole sits at ground level is
        # grounded regardless of the instantaneous contact count
        grounded = measured | (foot_pos[:, :, 2] < 0.01)

        if self._has_bias:
            bias = ent.data._joint_dof_field("qfrc_bias")[:, self.leg_ids]
        else:
            bias = torch.zeros(B, 12, device=dev)

        pos_t = torch.zeros(B, 12, device=dev)
        vel_t = torch.zeros(B, 12, device=dev)
        eff_t = torch.zeros(B, 12, device=dev)
        kp_t = torch.full((B, 12), cfg.stance_kp, device=dev)
        kd_t = torch.full((B, 12), cfg.stance_kd, device=dev)

        for f in range(2):
            sl = slice(6 * f, 6 * f + 6)
            q_f = q_leg[:, sl]
            ph_f = self._swing_ph[:, f]
            early_land = (ph_f > 0.6) & grounded[:, f]
            stance_like = (self._sched[:, f] & grounded[:, f]) | early_land

            r_point_b = torch.einsum(
                "bij,bj->bi",
                R_wb.transpose(1, 2),
                foot_pos[:, f] - ankle_pos[:, f],
            )
            J_w = self.kin.jac_world(f, q_f, R_wb, r_point_b)
            wrench = self.last_u[:, 6 * f : 6 * f + 6]
            tau_st = (
                torch.bmm(J_w.transpose(1, 2), (-wrench).unsqueeze(-1)).squeeze(-1)
                + bias[:, sl]
            )

            st = stance_like.unsqueeze(-1)
            pos_t[:, sl] = torch.where(
                st, self._stance_pos[:, sl], self._swing_pos[:, sl]
            )
            vel_t[:, sl] = torch.where(
                st, torch.zeros_like(q_f), self._swing_vel[:, sl]
            )
            eff_t[:, sl] = torch.where(st, tau_st, bias[:, sl])
            kp_t[:, sl] = torch.where(
                st,
                torch.full_like(q_f, cfg.stance_kp),
                torch.full_like(q_f, cfg.swing_kp),
            )
            kd_t[:, sl] = torch.where(
                st,
                torch.full_like(q_f, cfg.stance_kd),
                torch.full_like(q_f, cfg.swing_kd),
            )

        self._eff_target = eff_t
        for act, loc, mine in self._leg_act_maps:
            kp_g = act.stiffness.clone()
            kd_g = act.damping.clone()
            kp_g[:, loc] = kp_t[:, mine]
            kd_g[:, loc] = kd_t[:, mine]
            act.set_gains(slice(None), kp=kp_g, kd=kd_g)

        encoder = ent.data.encoder_bias[:, self.leg_ids]
        ent.set_joint_position_target(pos_t - encoder, joint_ids=self.leg_ids)
        ent.set_joint_velocity_target(vel_t, joint_ids=self.leg_ids)
        ent.set_joint_effort_target(eff_t, joint_ids=self.leg_ids)
        # arms + waist held at the default pose (A1: upper body nominal)
        hold = ent.data.default_joint_pos[:, self.other_ids]
        ent.set_joint_position_target(hold, joint_ids=self.other_ids)
