"""MuJoCo driver for the tick-inverted planner core (M2 stages 1-2).

Wraps an existing ``MujocoBackend`` (V2/V3): builds ``Obs`` from exactly the
reads the blocking primitives used, applies ``Cmd`` through the same write
paths (``_set_goal`` / ``_park_hand`` / shield methods / ``_enter``), and
advances the sim with ``_step()``. The write→step→read order matches the
blocking loops. The shield rate-limiter and the 200 Hz substep brake stay
with the backend, exactly as in the blocking architecture; ``HandCmd``
carries intent and ``HandObs.cmd_b`` is the rate-limited readback.
"""
from __future__ import annotations

import numpy as np

from execution.tick_core import Cmd, HandObs, Obs, PlannerCore, TickParams


def params_from_backend(be) -> TickParams:
    mj = be.mj
    return TickParams(
        speed_scale_locomotion=mj.speed_scale_locomotion,
        speed_scale_turn=mj.speed_scale_turn,
        speed_scale_min=mj.speed_scale_min,
        dt=float(be.dt),
        locomotion_tolerance_m=mj.locomotion_tolerance_m,
        locomotion_timeout_s=mj.locomotion_timeout_s,
        turn_yaw_tolerance_rad=mj.turn_yaw_tolerance_rad,
        turn_timeout_s=mj.turn_timeout_s,
        episode_budget_s=mj.episode_budget_s,
        attractor_pos_tol_m=float(be.fsm.cfg.pos_tol),
        guard_abort_force_n=mj.guard_abort_force_n,
        safety_clear_force_n=mj.safety_clear_force_n,
        stable_speed_m_s=mj.stable_speed_m_s,
        safety_settle_timeout_s=mj.safety_settle_timeout_s,
        park_x_m=mj.park_x_m,
        wp_crab_step_tol_m=mj.wp_crab_step_tol_m,
        wp_crab_step_timeout_s=mj.wp_crab_step_timeout_s,
        stabilize_timeout_s=mj.stabilize_timeout_s,
        probe_hand_z_m=mj.probe_hand_z_m,
        reach_speed_m_s=mj.reach_speed_m_s,
        press_far_speed_m_s=mj.press_far_speed_m_s,
        press_start_x_m=mj.press_start_x_m,
        press_overlap_m=mj.press_overlap_m,
        reach_tolerance_m=mj.reach_tolerance_m,
        reach_timeout_s=mj.reach_timeout_s,
        press_timeout_s=mj.press_timeout_s,
        press_settle_s=mj.press_settle_s,
        retract_timeout_s=mj.retract_timeout_s,
        force_limit_n=mj.force_limit_n,
        contact_threshold_n=be.cfg.contact_threshold_n,
        press_target_n=be._press_target_n(),
        free_sweep_margin_m=be.cfg.free_sweep_margin_m,
        park_rate_m_s=mj.park_rate_m_s,
    )


class MujocoTickDriver:
    def __init__(self, be):
        self.be = be
        self._last_mode = None

    # ── Obs ────────────────────────────────────────────────────────────────
    def make_obs(self) -> Obs:
        be = self.be
        # THE DEPLOYMENT INTERFACE.  `Obs.pose_w` is documented as the
        # "executor working pose", and on the real G1 there is no odometry to
        # supply it -- so it must be the onboard ESTIMATE, not the
        # simulator's truth.  `working_pose()` returns the drifting estimate
        # under honest sensing and truth when that flag is off, so the M2
        # regression's pinned legacy path is unchanged.
        xy, yaw = (be.working_pose() if hasattr(be, "working_pose")
                   else be.truth_pose())
        hands = {}
        for name, term in be.hands.items():
            f_w = term.measured_force_w()[0].detach().cpu().numpy()
            a = term.press_dir_w[0].detach().cpu().numpy()
            hands[name] = HandObs(
                ee_w=term.measured_ee_pose_w()[0][0].detach().cpu().numpy(),
                force_w=f_w,
                force_total_n=float(np.linalg.norm(f_w)),
                axis_n=abs(float(f_w @ a)),
                cmd_b=np.array([float(v) for v in term.pose_command_b[0, :3]]),
                kiss_x_m=float(term._kiss_x(be.env_ids)[0]),
                peak_total_n=float(be.shield.state(term).peak_total_n),
            )
        lin = be.robot.data.root_link_lin_vel_w[0]
        ang = be.robot.data.root_link_ang_vel_w[0]
        return Obs(
            t=float(be._t),
            pose_w=(float(xy[0]), float(xy[1]), float(yaw)),
            base_z=float(be.robot.data.root_link_pos_w[0, 2]),
            twist_w=(float(lin[0]), float(lin[1]), float(ang[2])),
            feet=be._feet(),
            hands=hands,
            substep_peak_total_n=float(be._substep_peak["total"]),
            guard_pending=bool(be.force_abort_pending),
        )

    # ── Cmd ────────────────────────────────────────────────────────────────
    def apply(self, cmd: Cmd) -> None:
        be = self.be
        if cmd.mode is not self._last_mode:
            be._enter(cmd.mode)
            self._last_mode = cmd.mode
        if cmd.ack_guard:
            be.force_abort_pending = False
        if cmd.reset_substep_peak:
            be._substep_peak.update({"axis": 0.0, "total": 0.0})
        if cmd.base is not None:
            be._set_goal(np.asarray(cmd.base.xy_w, float), float(cmd.base.yaw_w),
                         float(cmd.base.speed), hold=cmd.base.hold,
                         standing=cmd.base.standing)
        for name, h in cmd.hands.items():
            term = be.hands[name]
            if h.open_window:
                be.shield.open_window(term, be.env_ids)
                be.shield.reset_peaks(term)
            if h.reset_peaks and not h.open_window:
                be.shield.reset_peaks(term)
            if h.contact_mask is not None:
                term.contact_mask[:] = float(h.contact_mask)
            if h.park:
                be._park_hand(term)
            if h.lateral is not None:
                y, z, r = h.lateral
                be.shield.set_lateral(term, y_m=float(y), z_m=float(z), rate_m_s=float(r))
            if h.normal is not None:
                x, r = h.normal
                be.shield.set_normal(term, x_m=float(x), rate_m_s=float(r))
            if h.face_quat:
                term.pose_command_b[0, 3:] = be._face_quat(term)
            if h.close_window:
                be.shield.close_window(term)

    # ── loop ───────────────────────────────────────────────────────────────
    def run(self, core: PlannerCore, max_ticks: int = 500_000):
        """Blocking regression harness: drive the core to completion."""
        obs = None
        for _ in range(max_ticks):
            cmd = core.tick(obs)
            if cmd is None:
                break
            self.apply(cmd)
            self.be._step()
            obs = self.make_obs()
        return core.result
