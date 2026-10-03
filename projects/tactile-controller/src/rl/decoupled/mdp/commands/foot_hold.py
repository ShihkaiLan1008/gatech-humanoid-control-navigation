"""Sampled single-step foothold command with a swing-arc reference.

Command atom (the same ``[B, 7]`` layout as the foot_mimic line, so
``ManualFootstepCommand`` / ``VelocityFootstepCommand`` and the mimic reward
family consume it unchanged)::

    [swing one-hot (L, R) | countdown | target (x, w, z, yaw)]

Target is the swing FOOT SITE pose in the MIRRORED stance-site frame: x
forward along the stance foot, w = lateral distance toward the swing side
(always positive), yaw = heading change signed toward the swing side.
Standing = countdown 0 with the swing foot's held pose as target.

Per step the term also owns a swing-arc reference from the swing foot's
measured pose at issue to the target (``ref_feet_pos_w`` / ``ref_feet_yaw``).

The entry after the current one is pre-drawn at every transition and exposed
only through the MPC timeline views (``t_lift_q`` / ``t_td_q`` / ``foot_q`` /
``target_pos_w_q`` / ``target_yaw_w_q``, depth ``cfg.queue_len`` = 2, times
relative to now); the policy command never shows it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import sample_uniform, wrap_to_pi

from rl.g1_assets.g1_constants import FEET_Y_DIST

from .foot_hold_prior import LOWER_JOINT_NAMES, OrbitClipLibrary, PosturePriorCfg

if TYPE_CHECKING:
    from mjlab.entity import Entity
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.viewer.debug_visualizer import DebugVisualizer

FOOT_SITE_NAMES = ("left_foot", "right_foot")
"""Sole-contact sites, local index 0 = left, 1 = right."""

_TARGET_VIS = (0.90, 0.10, 0.10, 1.0)
_STANCE_VIS = (1.00, 0.65, 0.00, 1.0)
_ARC_VIS = (0.05, 0.20, 0.90, 1.0)
_ARC_NOW_VIS = (0.70, 0.20, 0.95, 1.0)
_FOOT_HALF = (0.09, 0.03)
"""Support-rectangle half-extents about the foot site [m]."""
_FOOT_BEADS = 20


def _yaw_from_quat_wxyz(q: torch.Tensor) -> torch.Tensor:
    """Yaw [rad] of wxyz quaternions. Shape (..., 4) -> (...)."""
    w, x, y, z = q.unbind(-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class FootHoldCommand(CommandTerm):
    """Sampled footholds + swing arc.

    Does not touch the robot state: reset events own it. On reset the term
    stands and issues its first step ``reset_delay`` seconds later."""

    cfg: FootHoldCommandCfg

    def __init__(self, cfg: FootHoldCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self.robot: Entity = env.scene[cfg.entity_name]
        self.feet_ids = torch.tensor(
            self.robot.find_sites(list(FOOT_SITE_NAMES), preserve_order=True)[0],
            dtype=torch.long, device=self.device)

        B, dev = self.num_envs, self.device
        self.swing = torch.zeros(B, dtype=torch.long, device=dev)
        self.t = torch.zeros(B, device=dev)
        self.standing = torch.ones(B, dtype=torch.bool, device=dev)
        self._fresh = torch.ones(B, dtype=torch.bool, device=dev)
        self._hold_left = torch.zeros(B, device=dev)
        self._nominal = torch.zeros(B, 3, device=dev)
        self._nominal_left = torch.zeros(B, device=dev)
        self.target_m = torch.zeros(B, 4, device=dev)
        self._anchor_pos = torch.zeros(B, 3, device=dev)
        self._anchor_yaw = torch.zeros(B, device=dev)
        self._start_pos = torch.zeros(B, 3, device=dev)
        self._start_yaw = torch.zeros(B, device=dev)
        self._target_pos_w = torch.zeros(B, 3, device=dev)
        self._target_yaw_w = torch.zeros(B, device=dev)
        self._apex = torch.zeros(B, device=dev)
        self._next_foot = torch.zeros(B, dtype=torch.long, device=dev)
        self._next_m = torch.zeros(B, 4, device=dev)
        self._next_apex = torch.zeros(B, device=dev)
        self._next_hold = torch.zeros(B, dtype=torch.bool, device=dev)
        self._next_hold_time = torch.zeros(B, device=dev)
        self._ref_feet_pos = torch.zeros(B, 2, 3, device=dev)
        self._ref_feet_yaw = torch.zeros(B, 2, device=dev)
        self._bead_local = self._bead_ring(_FOOT_BEADS, *_FOOT_HALF, device=dev)
        for k in ("td_err", "swing_peak_z", "step_travel", "hold_slide", "standing"):
            self.metrics[k] = torch.zeros(B, device=dev)
        # per-step / per-hold metric accumulators
        self._peak_z = torch.zeros(B, device=dev)
        self._issue_pelvis_xy = torch.zeros(B, 2, device=dev)
        self._hold_feet_xy = torch.zeros(B, 2, 2, device=dev)
        # optional joint-posture prior (orbit clips); None = A behaviour
        self._prior: OrbitClipLibrary | None = None
        if cfg.posture_prior is not None:
            assert tuple(self.robot.joint_names[:15]) == LOWER_JOINT_NAMES, (
                "entity joint order != clip LOWER order")
            self._prior = OrbitClipLibrary(cfg.posture_prior.dataset_path, dev,
                                           cfg.posture_prior.yaw_weight)
            assert abs(self._prior.step_period - cfg.timing.step_period) < 1e-6, (
                "orbit clips were built for another step_period")
            self._prior_idx = torch.zeros(B, dtype=torch.long, device=dev)
            self._prior_ref = torch.zeros(B, 15, device=dev)
            self._prior_mask = torch.zeros(B, 15, dtype=torch.bool, device=dev)
            self.metrics["prior_err"] = torch.zeros(B, device=dev)

    # ------------------------------------------------------------ views
    @property
    def command(self) -> torch.Tensor:
        """[B, 7] = swing one-hot (L, R) | countdown | target (x, w, z, yaw)."""
        one_hot = torch.nn.functional.one_hot(self.swing, 2).float()
        cd = torch.where(self.standing, 0.0, 1.0 - self.t / self.cfg.timing.step_period)
        return torch.cat([one_hot, cd.clamp(min=0.0).unsqueeze(1), self.target_m], dim=-1)

    @property
    def swing_active(self) -> torch.Tensor:
        """[B] bool: inside the swing window of an active step."""
        tm = self.cfg.timing
        return ~self.standing & (self.t >= tm.lift_frac * tm.step_period)

    @property
    def swing_frac(self) -> torch.Tensor:
        """[B] swing progress in [0, 1]; 0 before lift-off and while standing."""
        T, f = self.cfg.timing.step_period, self.cfg.timing.lift_frac
        s = (self.t - f * T) / ((1.0 - f) * T)
        return torch.where(self.standing, 0.0, s.clamp(0.0, 1.0))

    @property
    def ref_feet_pos_w(self) -> torch.Tensor:
        """Reference foot SITE positions, world frame. Shape (B, 2, 3), m.

        Swing foot = the arc at the current progress; stance foot = the pose
        measured at issue. Standing: both frozen at stand entry."""
        return self._ref_feet_pos

    @property
    def ref_feet_yaw(self) -> torch.Tensor:
        """Reference foot SITE yaws, world frame. Shape (B, 2), rad."""
        return self._ref_feet_yaw

    @property
    def target_pos_w(self) -> torch.Tensor:
        """Committed landing target of the swing site, world frame. Shape (B, 3), m."""
        return self._target_pos_w

    @property
    def target_yaw_w(self) -> torch.Tensor:
        """Committed landing yaw of the swing site, world frame. Shape (B,), rad."""
        return self._target_yaw_w

    @property
    def ref_joint_pos_prior(self) -> torch.Tensor:
        """Prior joint positions, LOWER order (legs + waist). Shape (B, 15), rad.

        Stepping: the matched orbit clip at the clock frame. Standing: the
        clip's standing pose over the current feet, blended into its issue
        frame over the last ``windup_time`` of the stand. Zeros without a prior."""
        return self._prior_ref

    @property
    def prior_joint_mask(self) -> torch.Tensor:
        """Joints the prior applies to. Shape (B, 15), bool: the waist always,
        the SWING leg while stepping; the stance leg is never priced."""
        return self._prior_mask

    def swing_arc(self, u: torch.Tensor) -> torch.Tensor:
        """Swing-arc reference at progress ``u`` in [0, 1], world frame.

        Zero-end-velocity cubic in xy from the lift-off pose to the target,
        sin^2 bump in z with the step's sampled apex. Shape (B, K) -> (B, K, 3), m."""
        h = (u * u * (3.0 - 2.0 * u)).unsqueeze(-1)
        p0 = self._start_pos.unsqueeze(1)
        p1 = self._target_pos_w.unsqueeze(1)
        pos = p0 + (p1 - p0) * h
        pos[..., 2] += self._apex.unsqueeze(1) * torch.sin(math.pi * u) ** 2
        return pos

    # ------------------------------------------------------------ MPC timeline
    @property
    def t_now(self) -> torch.Tensor:
        """[B] clock the timeline windows are expressed in (0 = now)."""
        return torch.zeros_like(self.t)

    @property
    def stance_yaw_w(self) -> torch.Tensor:
        """[B] world yaw of entry 0's stance foot [rad]."""
        rows = torch.arange(self.num_envs, device=self.device)
        stand_st = self._ref_feet_yaw[rows, 1 - self._next_foot]
        return torch.where(self.standing, stand_st, self._anchor_yaw)

    def _timeline(self):
        """Windows [t_lift, t_td) from now, feet and world targets of entry 0
        (current step, or the step that ends a stand) and entry 1 (the
        pre-drawn follower; absent while standing). Shapes (B, 2[, 3])."""
        tm = self.cfg.timing
        T, f = tm.step_period, tm.lift_frac
        rows = torch.arange(self.num_envs, device=self.device)
        inf = torch.full_like(self.t, 1.0e9)

        # stepping: entry 0 = current, entry 1 = follower (after an optional stand)
        td0_s = T - self.t
        gap = torch.where(self._next_hold, self._next_hold_time, torch.zeros_like(self.t))
        lift1_s = td0_s + gap + f * T
        pos1_s, yaw1_s = self._compose(
            self._target_pos_w, self._target_yaw_w, self._next_m, self._next_foot)

        # standing: entry 0 = the wake step about the frozen stance foot
        st = 1 - self._next_foot
        pos0_h, yaw0_h = self._compose(
            self._ref_feet_pos[rows, st], self._ref_feet_yaw[rows, st],
            self._next_m, self._next_foot)
        hold = self._hold_left.clamp(min=0.0)

        S = self.standing
        Sv = S.unsqueeze(1)
        t_lift = torch.stack([torch.where(S, hold + f * T, f * T - self.t),
                              torch.where(S, inf, lift1_s)], dim=1)
        t_td = torch.stack([torch.where(S, hold + T, td0_s),
                            torch.where(S, inf, lift1_s + (1.0 - f) * T)], dim=1)
        foot = torch.stack([torch.where(S, self._next_foot, self.swing),
                            torch.where(S, 1 - self._next_foot, self._next_foot)], dim=1)
        pos = torch.stack([torch.where(Sv, pos0_h, self._target_pos_w),
                           torch.where(Sv, pos0_h, pos1_s)], dim=1)
        yaw = torch.stack([torch.where(S, yaw0_h, self._target_yaw_w),
                           torch.where(S, yaw0_h, yaw1_s)], dim=1)
        return t_lift, t_td, foot, pos, yaw

    @property
    def t_lift_q(self) -> torch.Tensor:
        """[B, 2] lift-off times from now [s]; 1e9 = no entry."""
        return self._timeline()[0]

    @property
    def t_td_q(self) -> torch.Tensor:
        """[B, 2] touchdown times from now [s]; 1e9 = no entry."""
        return self._timeline()[1]

    @property
    def foot_q(self) -> torch.Tensor:
        """[B, 2] swing foot per entry (0 = left, 1 = right)."""
        return self._timeline()[2]

    @property
    def target_pos_w_q(self) -> torch.Tensor:
        """[B, 2, 3] landing targets per entry, world frame [m]."""
        return self._timeline()[3]

    @property
    def target_yaw_w_q(self) -> torch.Tensor:
        """[B, 2] landing yaws per entry, world frame [rad]."""
        return self._timeline()[4]

    @property
    def robot_feet_pos_w(self) -> torch.Tensor:
        """Measured foot SITE positions, world frame. Shape (B, 2, 3), m."""
        return self.robot.data.site_pos_w[:, self.feet_ids, :]

    @property
    def robot_feet_yaw(self) -> torch.Tensor:
        """Measured foot SITE yaws, world frame. Shape (B, 2), rad."""
        return _yaw_from_quat_wxyz(self.robot.data.site_quat_w[:, self.feet_ids, :])

    # ------------------------------------------------------------ sampling
    def _draw_nominal(self, ids: torch.Tensor) -> None:
        n, dev, sp = len(ids), self.device, self.cfg.sampler
        probs = torch.tensor(sp.x_probs, device=dev)
        bins = torch.tensor(sp.x_bins, device=dev)
        k = torch.multinomial(probs, n, replacement=True)
        self._nominal[ids, 0] = bins[k, 0] + torch.rand(n, device=dev) * (bins[k, 1] - bins[k, 0])
        self._nominal[ids, 1] = sample_uniform(*sp.dw_range, (n,), dev)
        self._nominal[ids, 2] = sample_uniform(*sp.yaw_range, (n,), dev)
        if sp.nominal_time is not None:
            self._nominal_left[ids] = sample_uniform(*sp.nominal_time, (n,), dev)

    def _draw_target_m(self, ids: torch.Tensor, foot: torch.Tensor) -> torch.Tensor:
        """Mirrored-frame target (x, w, 0, yaw_m) for ``foot`` and the arc
        apex [m]. Shapes (n, 4), (n,)."""
        n, dev, sp = len(ids), self.device, self.cfg.sampler
        nom = self._nominal[ids]
        s = torch.where(foot == 0, 1.0, -1.0)
        yaw = (nom[:, 2] + sample_uniform(*sp.jitter_yaw, (n,), dev)).clamp(*sp.yaw_clamp)
        x = (nom[:, 0] + sample_uniform(*sp.jitter_x, (n,), dev)).clamp(*sp.x_clamp)
        w_min = sp.w_min_base + sp.w_min_yaw_gain * yaw.abs()
        w = FEET_Y_DIST + s * nom[:, 1] + sample_uniform(*sp.jitter_w, (n,), dev)
        w = torch.maximum(w, w_min)
        apex = sample_uniform(*sp.apex_range, (n,), dev)
        return torch.stack([x, w, torch.zeros_like(x), s * yaw], dim=-1), apex

    # ------------------------------------------------------------ transitions
    def _compose(self, a_pos: torch.Tensor, a_yaw: torch.Tensor,
                 tm: torch.Tensor, foot: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Mirrored target ``tm`` about the stance pose -> world (pos on
        ``ground_z``, yaw). Shapes (n, 3), (n,)."""
        s = torch.where(foot == 0, 1.0, -1.0)
        c, sn = a_yaw.cos(), a_yaw.sin()
        x, y = tm[:, 0], s * tm[:, 1]
        pos = torch.stack([a_pos[:, 0] + c * x - sn * y,
                           a_pos[:, 1] + sn * x + c * y,
                           torch.full_like(x, self.cfg.ground_z)], dim=-1)
        return pos, wrap_to_pi(a_yaw + s * tm[:, 3])

    def _plan_next(self, ids: torch.Tensor, foot: torch.Tensor, may_hold: bool) -> None:
        """Pre-draw the entry after the current one (side, relative target,
        apex, and whether a stand precedes it)."""
        n, dev = len(ids), self.device
        sp, tm = self.cfg.sampler, self.cfg.timing
        if sp.nominal_time is not None:
            expired = ids[self._nominal_left[ids] <= 0.0]
            if len(expired) > 0:
                self._draw_nominal(expired)
        m, apex = self._draw_target_m(ids, foot)
        self._next_foot[ids] = foot
        self._next_m[ids] = m
        self._next_apex[ids] = apex
        hold = torch.rand(n, device=dev) < tm.p_hold if may_hold else torch.zeros(n, dtype=torch.bool, device=dev)
        self._next_hold[ids] = hold
        self._next_hold_time[ids] = sample_uniform(*tm.hold_time, (n,), dev)

    def _prior_lookup(self, ids: torch.Tensor, foot: torch.Tensor, m: torch.Tensor,
                      apex: torch.Tensor, feet_pos: torch.Tensor,
                      feet_yaw: torch.Tensor) -> None:
        """Match the pre-drawn step (mirrored target ``m``, ``apex``) of the
        swing ``foot`` starting from the given measured feet to its nearest
        orbit clip; stores the index for ``ids``."""
        rows = torch.arange(len(ids), device=self.device)
        st = 1 - foot
        s = torch.where(foot == 0, 1.0, -1.0)
        d = feet_pos[rows, foot, :2] - feet_pos[rows, st, :2]
        a = feet_yaw[rows, st]
        c, sn = a.cos(), a.sin()
        lx = c * d[:, 0] + sn * d[:, 1]
        lw = s * (-sn * d[:, 0] + c * d[:, 1])
        lyaw = s * wrap_to_pi(feet_yaw[rows, foot] - a)
        feat = torch.stack([m[:, 0], m[:, 1], m[:, 3], lx, lw, lyaw, apex], dim=-1)
        self._prior_idx[ids] = self._prior.lookup(feat, foot)

    def _update_prior(self) -> None:
        """Prior reference + mask for the current clock (see ``ref_joint_pos_prior``)."""
        lib = self._prior
        idx = self._prior_idx
        k = torch.round(self.t / lib.dt).long().clamp(0, lib.n_frames - 1)
        step_ref = lib.joint_pos[idx, k]
        rest = lib.q_rest_start[idx]
        wu = self.cfg.posture_prior.windup_time
        blend = (1.0 - self._hold_left / wu).clamp(0.0, 1.0).unsqueeze(-1)
        hold_ref = rest + (lib.joint_pos[idx, 0] - rest) * blend
        self._prior_ref = torch.where(self.standing.unsqueeze(-1), hold_ref, step_ref)
        mask = torch.zeros_like(self._prior_mask)
        mask[:, 12:] = True
        leg = torch.arange(6, device=self.device).unsqueeze(0) + 6 * self.swing.unsqueeze(1)
        mask.scatter_(1, leg, (~self.standing).unsqueeze(-1).expand(-1, 6))
        self._prior_mask = mask

    def _issue_step(self, ids: torch.Tensor) -> None:
        """Start the pre-drawn step from the MEASURED feet, then pre-draw its follower."""
        pos, yaw = self.robot_feet_pos_w[ids], self.robot_feet_yaw[ids]
        rows = torch.arange(len(ids), device=self.device)
        foot = self._next_foot[ids]
        stance = 1 - foot
        a_pos, a_yaw = pos[rows, stance].clone(), yaw[rows, stance]
        a_pos[:, 2] = self.cfg.ground_z
        tgt, tgt_yaw = self._compose(a_pos, a_yaw, self._next_m[ids], foot)

        self.swing[ids] = foot
        self.t[ids] = 0.0
        self.standing[ids] = False
        self.target_m[ids] = self._next_m[ids]
        self._apex[ids] = self._next_apex[ids]
        self._anchor_pos[ids], self._anchor_yaw[ids] = a_pos, a_yaw
        self._start_pos[ids], self._start_yaw[ids] = pos[rows, foot], yaw[rows, foot]
        self._target_pos_w[ids] = tgt
        self._target_yaw_w[ids] = tgt_yaw
        self._ref_feet_pos[ids] = pos
        self._ref_feet_pos[ids[rows], stance] = a_pos
        self._ref_feet_yaw[ids] = yaw
        self._peak_z[ids] = 0.0
        self._issue_pelvis_xy[ids] = self.robot.data.root_link_pos_w[ids, :2]
        if self._prior is not None:
            self._prior_lookup(ids, foot, self._next_m[ids], self._next_apex[ids], pos, yaw)
        self._plan_next(ids, 1 - foot, may_hold=True)

    def _enter_stand(self, ids: torch.Tensor, duration: torch.Tensor,
                     feet_pos: torch.Tensor, feet_yaw: torch.Tensor) -> None:
        """Hold for ``duration`` [s]; target = swing pose relative to stance,
        frozen; the wake step (random side) is pre-drawn."""
        rows = torch.arange(len(ids), device=self.device)
        sw = self.swing[ids]
        st = 1 - sw
        s = torch.where(sw == 0, 1.0, -1.0)
        d = feet_pos[rows, sw, :2] - feet_pos[rows, st, :2]
        a = feet_yaw[rows, st]
        c, sn = a.cos(), a.sin()
        x = c * d[:, 0] + sn * d[:, 1]
        y = -sn * d[:, 0] + c * d[:, 1]
        dyaw = wrap_to_pi(feet_yaw[rows, sw] - a)
        self.target_m[ids] = torch.stack(
            [x, s * y, torch.zeros_like(x), s * dyaw], dim=-1)
        self.standing[ids] = True
        self._hold_left[ids] = duration
        self._hold_feet_xy[ids] = feet_pos[:, :, :2]
        feet_pos = feet_pos.clone()
        feet_pos[..., 2] = self.cfg.ground_z
        self._ref_feet_pos[ids] = feet_pos
        self._ref_feet_yaw[ids] = feet_yaw
        self._target_pos_w[ids] = feet_pos[rows, sw]
        self._target_yaw_w[ids] = feet_yaw[rows, sw]
        wake = torch.randint(0, 2, (len(ids),), device=self.device)
        self._plan_next(ids, wake, may_hold=False)
        if self._prior is not None:
            self._prior_lookup(ids, wake, self._next_m[ids], self._next_apex[ids],
                               feet_pos, feet_yaw)

    # ------------------------------------------------------------ hooks
    def _resample_command(self, env_ids: torch.Tensor) -> None:
        """Episode reset: draw the nominal; the stand is entered from the
        measured feet on the next update (sim poses are stale here)."""
        n, dev = len(env_ids), self.device
        self._draw_nominal(env_ids)
        self.swing[env_ids] = torch.randint(0, 2, (n,), device=dev)
        self.t[env_ids] = 0.0
        self.standing[env_ids] = True
        self._fresh[env_ids] = True
        self._hold_left[env_ids] = sample_uniform(*self.cfg.timing.reset_delay, (n,), dev)
        for k in ("td_err", "swing_peak_z", "step_travel", "hold_slide"):
            self.metrics[k][env_ids] = 0.0

    def _update_command(self) -> None:
        dt, tm = self._env.step_dt, self.cfg.timing
        if self._fresh.any():
            ids = self._fresh.nonzero(as_tuple=False).squeeze(-1)
            self._enter_stand(ids, self._hold_left[ids],
                              self.robot_feet_pos_w[ids], self.robot_feet_yaw[ids])
            self._fresh[ids] = False
        self.t += (~self.standing).float() * dt
        self._hold_left -= self.standing.float() * dt
        if self.cfg.sampler.nominal_time is not None:
            self._nominal_left -= dt

        landed = ~self.standing & (self.t >= tm.step_period - 1e-6)
        if landed.any():
            ids = landed.nonzero(as_tuple=False).squeeze(-1)
            rows = torch.arange(len(ids), device=self.device)
            sw = self.swing[ids]
            err = (self.robot_feet_pos_w[ids][rows, sw, :2]
                   - self._target_pos_w[ids, :2]).norm(dim=-1)
            self.metrics["td_err"][ids] = err
            self.metrics["swing_peak_z"][ids] = self._peak_z[ids]
            self.metrics["step_travel"][ids] = (
                self.robot.data.root_link_pos_w[ids, :2] - self._issue_pelvis_xy[ids]
            ).norm(dim=-1)
            hold = self._next_hold[ids]
            if hold.any():
                h = ids[hold]
                self._enter_stand(h, self._next_hold_time[h],
                                  self.robot_feet_pos_w[h], self.robot_feet_yaw[h])
            if (~hold).any():
                self._issue_step(ids[~hold])

        wake = self.standing & (self._hold_left <= 0.0)
        if wake.any():
            ids = wake.nonzero(as_tuple=False).squeeze(-1)
            self.metrics["hold_slide"][ids] = (
                self.robot_feet_pos_w[ids, :, :2] - self._hold_feet_xy[ids]
            ).norm(dim=-1).amax(dim=-1)
            self._issue_step(ids)

        rows = torch.arange(self.num_envs, device=self.device)
        z = self.robot_feet_pos_w[rows, self.swing, 2] - self.cfg.ground_z
        self._peak_z = torch.where(self.swing_active, torch.maximum(self._peak_z, z), self._peak_z)

        self._update_arc()
        if self._prior is not None:
            self._update_prior()

    def _update_arc(self) -> None:
        """Swing-foot arc: zero-end-velocity cubic in xy/yaw, sin^2 bump in z."""
        active = ~self.standing
        if not active.any():
            return
        ids = active.nonzero(as_tuple=False).squeeze(-1)
        s = self.swing_frac
        pos = self.swing_arc(s.unsqueeze(1))[ids, 0]
        h = (s * s * (3.0 - 2.0 * s))[ids]
        yaw = self._start_yaw[ids] + wrap_to_pi(self._target_yaw_w[ids] - self._start_yaw[ids]) * h
        rows = torch.arange(len(ids), device=self.device)
        sw = self.swing[ids]
        self._ref_feet_pos[ids[rows], sw] = pos
        self._ref_feet_yaw[ids[rows], sw] = wrap_to_pi(yaw)
        self._ref_feet_pos[ids[rows], 1 - sw] = self._anchor_pos[ids]
        self._ref_feet_yaw[ids[rows], 1 - sw] = self._anchor_yaw[ids]

    def _update_metrics(self) -> None:
        """td_err [m], swing_peak_z [m] and step_travel [m] latch at each
        landing; hold_slide [m] latches when a stand ends; standing = mode."""
        self.metrics["standing"] = self.standing.float()
        if self._prior is not None:
            err = (self._prior_ref - self.robot.data.joint_pos[:, :15]) * self._prior_mask
            n = self._prior_mask.sum(dim=-1).clamp(min=1)
            self.metrics["prior_err"] = (err.square().sum(dim=-1) / n).sqrt()

    def _debug_vis_impl(self, visualizer: "DebugVisualizer") -> None:
        """Footprints = centre sphere + bead ring on the foot-site support
        rectangle: ORANGE = the stance reference, RED = the swing target.
        While stepping: BLUE = the swing arc, PURPLE = the current reference
        point. Arc segments are boxes (the viser backend batches each
        primitive type under the alpha of its last queued instance)."""
        env_indices = visualizer.get_env_indices(self.num_envs)
        if not env_indices:
            return
        B, dev = self.num_envs, self.device
        rows = torch.arange(B, device=dev)

        K = 16
        u = torch.linspace(0.0, 1.0, K + 1, device=dev).expand(B, K + 1)
        arc = self.swing_arc(u)
        d = arc[:, 1:] - arc[:, :-1]
        seg_len = d.norm(dim=-1)
        seg_mat = self._x_aligned_rotation(d / seg_len.clamp(min=1e-9).unsqueeze(-1))
        stance = 1 - self.swing
        anc_c, anc_ring = self._footprint(
            self._ref_feet_pos[rows, stance, 0:2], self._ref_feet_yaw[rows, stance])
        tgt_c, tgt_ring = self._footprint(self._target_pos_w[:, 0:2], self._target_yaw_w)

        h = {
            k: v.cpu().numpy() for k, v in dict(
                seg_c=0.5 * (arc[:, 1:] + arc[:, :-1]), seg_len=seg_len, seg_mat=seg_mat,
                arc_now=self._ref_feet_pos[rows, self.swing], active=~self.standing,
                tgt_c=tgt_c, tgt_ring=tgt_ring, anc_c=anc_c, anc_ring=anc_ring,
            ).items()
        }
        for b in env_indices:
            self._draw_footprint(visualizer, h["anc_c"][b], h["anc_ring"][b], _STANCE_VIS, f"anc_{b}")
            self._draw_footprint(visualizer, h["tgt_c"][b], h["tgt_ring"][b], _TARGET_VIS, f"tgt_{b}")
            if not h["active"][b]:
                continue
            for k in range(K):
                if h["seg_len"][b, k] < 1e-6:
                    continue
                visualizer.add_box(
                    center=h["seg_c"][b, k],
                    size=(0.5 * h["seg_len"][b, k] + 0.004, 0.008, 0.008),
                    mat=h["seg_mat"][b, k],
                    color=_ARC_VIS, label=f"footplan_arc_{b}_{k}",
                )
            visualizer.add_sphere(
                center=h["arc_now"][b], radius=0.02, color=_ARC_NOW_VIS,
                label=f"footplan_arc_now_{b}",
            )

    def _footprint(
        self, pos_xy: torch.Tensor, yaw: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Centre (B, 3) and ring beads (B, _FOOT_BEADS, 3) of the foot-site
        support rectangle at ``pos_xy`` (B, 2) rotated by ``yaw`` (B,), at ground level."""
        c, s = yaw.unsqueeze(1).cos(), yaw.unsqueeze(1).sin()
        lx, ly = self._bead_local[:, 0], self._bead_local[:, 1]
        x = pos_xy[:, 0:1] + c * lx - s * ly
        y = pos_xy[:, 1:2] + s * lx + c * ly
        ring = torch.stack([x, y, torch.full_like(x, 0.01)], dim=-1)
        centre = torch.cat([pos_xy, torch.full_like(pos_xy[:, :1], 0.02)], dim=-1)
        return centre, ring

    @staticmethod
    def _bead_ring(n: int, hx: float, hy: float, device) -> torch.Tensor:
        """(n, 2) points equally spaced along the perimeter of the rectangle
        with half-extents (hx, hy), local frame."""
        corners = torch.tensor(
            [[hx, hy], [-hx, hy], [-hx, -hy], [hx, -hy], [hx, hy]], device=device
        )
        seg = corners[1:] - corners[:-1]
        seg_len = seg.norm(dim=-1)
        cum = torch.cat([torch.zeros(1, device=device), seg_len.cumsum(0)])
        dist = torch.linspace(0.0, float(cum[-1]), n + 1, device=device)[:-1]
        j = (torch.searchsorted(cum, dist, right=True) - 1).clamp(max=3)
        f = ((dist - cum[j]) / seg_len[j]).unsqueeze(1)
        return corners[j] + f * seg[j]

    @staticmethod
    def _x_aligned_rotation(x: torch.Tensor) -> torch.Tensor:
        """(..., 3, 3) rotation whose local x axis is the unit vector ``x``
        (..., 3); columns x, y, z with y horizontal where possible."""
        up = torch.where(
            x[..., 2:3].abs() < 0.99,
            torch.tensor([0.0, 0.0, 1.0], device=x.device),
            torch.tensor([0.0, 1.0, 0.0], device=x.device),
        ).expand_as(x)
        y = torch.cross(up, x, dim=-1)
        y = y / y.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        z = torch.cross(x, y, dim=-1)
        return torch.stack([x, y, z], dim=-1)

    @staticmethod
    def _draw_footprint(
        visualizer: "DebugVisualizer", centre, ring, color, tag: str
    ) -> None:
        visualizer.add_sphere(center=centre, radius=0.025, color=color, label=f"footplan_{tag}_c")
        for i in range(ring.shape[0]):
            visualizer.add_sphere(center=ring[i], radius=0.008, color=color, label=f"footplan_{tag}_{i}")


@dataclass(kw_only=True)
class FootHoldCommandCfg(CommandTermCfg):
    """Configuration for :class:`FootHoldCommand`."""

    @dataclass(kw_only=True)
    class Sampler:
        """Everything a draw reads. Nominal = once per episode (or every
        ``nominal_time``); the rest = per step, mirrored stance-site frame."""

        x_bins: tuple[tuple[float, float], ...] = (
            (-0.20, -0.08), (-0.08, 0.08), (0.08, 0.20), (0.20, 0.30))
        """Nominal step length x* [m] is drawn inside one of these bins."""
        x_probs: tuple[float, ...] = (0.25, 0.20, 0.30, 0.25)
        """Bin quotas for ``x_bins``."""
        dw_range: tuple[float, float] = (-0.07, 0.07)
        """Nominal width drift dw* [m]: left lands at w0+dw*, right at w0-dw* (sustained lateral)."""
        yaw_range: tuple[float, float] = (-0.25, 0.25)
        """Nominal heading change per step yaw* [rad], positive = left turn."""
        nominal_time: tuple[float, float] | None = None
        """If set, the nominal is redrawn after this many seconds (at a step boundary)."""

        jitter_x: tuple[float, float] = (-0.04, 0.04)
        jitter_w: tuple[float, float] = (-0.04, 0.04)
        jitter_yaw: tuple[float, float] = (-0.08, 0.08)
        """Per-step uniform jitter around the nominal [m, m, rad]."""
        x_clamp: tuple[float, float] = (-0.20, 0.30)
        """Landing x range [m] = the deploy coverage box."""
        yaw_clamp: tuple[float, float] = (-0.30, 0.30)
        """Landing yaw range [rad] = the deploy coverage box."""
        w_min_base: float = 0.14
        w_min_yaw_gain: float = 0.15
        """Width floor w >= w_min_base + w_min_yaw_gain * |yaw| [m]."""
        apex_range: tuple[float, float] = (0.05, 0.09)
        """Swing-arc peak height above the lift-off site [m], drawn per step."""

    @dataclass(kw_only=True)
    class Timing:
        """Step clock and stand transitions."""

        step_period: float = 0.5
        """Step duration [s] from issue to scheduled landing (= half the gait cycle)."""
        lift_frac: float = 0.25
        """Fraction of ``step_period`` before lift-off; the swing spans the rest."""
        p_hold: float = 0.15
        """Probability of entering a stand at each landing."""
        hold_time: tuple[float, float] = (1.0, 4.0)
        """Stand duration [s]."""
        reset_delay: tuple[float, float] = (0.3, 1.0)
        """Stand duration [s] before the first step of an episode."""

    entity_name: str = "robot"
    ground_z: float = 0.0
    """Contact-plane height of the foot sites [m]; landing targets and stance
    references are pinned to it, never to the measured stance z."""
    sampler: Sampler = field(default_factory=Sampler)
    timing: Timing = field(default_factory=Timing)
    queue_len: int = 2
    """MPC preview depth: the current entry + one pre-drawn follower (fixed)."""
    posture_prior: PosturePriorCfg | None = None
    """Orbit-clip joint prior (``ref_joint_pos_prior`` / ``prior_joint_mask``);
    None = no prior, the term behaves exactly as without it."""

    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)
    """Never fires: transitions are step-driven."""
    debug_vis: bool = False

    def build(self, env: ManagerBasedRlEnv) -> FootHoldCommand:
        return FootHoldCommand(self, env)
