"""Tactile-tip SE(3) target + contact mask + force via a VIRTUAL SPRING
(no press cubes, no contact geometry — fully independent of the cube stack).

``EeSpringContactCommand`` wraps the reach-pool :class:`EeTargetCommand`
REFERENCE with the Portela-style force field (Learning Force Control for
Legged Manipulation, arXiv:2405.01402): a unilateral normal spring-damper
anchored just behind the GLIDING TARGET's face — the transition is a goal
too, the surface rides the reference, there is no separate arrival phase
(user design 2026-08-11). The command is a full 3-D force in the TARGET
frame (user decision 2026-08-12): normal component F_n along the target's +x
plus a tangential component in its y-z plane with |F_t| ≤
tangential_ratio_max·F_n — strictly inside the QP's μ_hand friction cone:

    F⃗_cmd  = F_n·n̂ + F⃗_t                       (n̂ = target +x, press dir)
    anchor  = target(t) − (F_n(t) / k_ext) · n̂
    pen     = (tip − anchor) · n̂
    f_n     = min(clamp(k_ext·pen + d_ext·(ṫip − ȧnchor)·n̂, 0),
                  f_headroom · F_n(t), f_cap)
    force   = (f_n / F_n) · F⃗_cmd     (tangential RIDES the realized normal,
    wrench  = −force                    friction-like — reaction on the robot)

The tangential channel exists only in proportion to the realized press
(f_n/F_n ∈ [0, f_headroom]): no pressing ⇒ no lateral force, so the spring
can never act as a free position controller tugging the hand toward the
target; at f_n = F_n the emitted force is exactly F⃗_cmd.

Properties this buys (why a spring, not a rigid cube):

* The force is a FUNCTION OF THE TIP POSITION — the action→force law
  (press deeper ⇔ more force) exists in the environment and is smooth
  (k_ext ≈ 1e3 N/m: 25 N ⇔ 2.5 cm), so DIRECT force rewards are learnable.
  The rigid cube's mm-wide force cliff was measured unlearnable (run 1).
* ``tip on the reference ⇔ f_n = F_cmd(t)`` at EVERY instant of the window —
  glides train "hold the press on a moving surface" (the wipe skill), and
  the pose kernels and the force rewards share one equilibrium throughout.
* Glide draws are intrinsically safe: the anchor departs FROM the reference
  (≈ the hand) and never teleports. For jump draws (T = 0) the
  COMMAND-SCALED cap ``f_headroom·F_cmd(t)`` — ramping from 0 with the
  command — turns a tip-inside-the-new-face overlap into a gentle push
  instead of a full-penetration shove, and keeps the plant inside the QP's
  own hand cone bound (the same 1.25·f_cmd): physics never exceeds the model.
* The spring is deliberately K·x + D·ẋ ONLY — a passive massless surface has
  no M·ẍ term (that belongs to virtual-inertia impedance rendering,
  FACET-style, which this is not), and a finite-differenced ẍ would inject
  noise straight into the applied force.
* Damping acts on the RELATIVE velocity: the anchor moves with the glide
  (up to ~0.75 m/s) — tip-only damping would fake up to ~7 N of drag.

The force this field emits is ALREADY modelled by the loco-manip QP's hand
slots (u_ref = −f_cmd, cone box) — never feed it to ``other_body_force``
too, that double-counts the wrench (the push_box rule).

Phantom windows (mask + force command live, NO spring) keep the deploy
discrimination task. There is no arrival concept anywhere — rewards and the
MPC command-gate cover the whole window (mask ∧ ¬phantom). Duck-typed to
the press-cube command's accessor surface
(mask/phantom/force_cmd/press_dir/measured_force), so rewards, observations
and the loco-manip MPC consume either implementation unchanged. Deliberately
independent of ``ee_contact`` (user decision: the cube stack and the spring
stack must not share code).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from mjlab.utils.lab_api.math import quat_apply

from .ee_tartget import EeTargetCommand, EeTargetCommandCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.viewer.debug_visualizer import DebugVisualizer

_CMD_FORCE_VIS = (0.10, 0.90, 0.20, 0.95)   # commanded
_MEAS_FORCE_VIS = (0.95, 0.20, 0.85, 0.95)  # spring (virtual surface)


class EeSpringContactCommand(EeTargetCommand):
    """Pool-drawn tactile-tip pose + contact mask + force, realized by a
    unilateral virtual spring riding the gliding reference."""

    cfg: "EeSpringContactCommandCfg"

    def __init__(self, cfg: "EeSpringContactCommandCfg", env: "ManagerBasedRlEnv") -> None:
        super().__init__(cfg, env)
        B = self.num_envs
        dev = self.device

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

    # ── command tuple accessors (duck interface shared with the cube stack) ──

    @property
    def press_dir_b(self) -> Tensor:
        """Commanded press axis (+x of the GLIDING TARGET frame), robot frame."""
        ex = torch.zeros(self.num_envs, 3, device=self.device)
        ex[:, 0] = 1.0
        return quat_apply(self.pose_command_b[:, 3:], ex)

    @property
    def press_dir_w(self) -> Tensor:
        """Commanded press axis in the world frame (target frame's +x)."""
        _, quat_w = self.target_pose_w()
        ex = torch.zeros(self.num_envs, 3, device=self.device)
        ex[:, 0] = 1.0
        return quat_apply(quat_w, ex)

    @property
    def force_cmd_mag(self) -> Tensor:
        return self._force_cmd_mag

    @property
    def force_cmd_b(self) -> Tensor:
        return quat_apply(self.pose_command_b[:, 3:], self._force_cmd_local)

    @property
    def force_cmd_w(self) -> Tensor:
        _, quat_w = self.target_pose_w()
        return quat_apply(quat_w, self._force_cmd_local)

    def measured_force_w(self) -> Tensor:
        """Hand-on-surface force (= −applied wrench): the spring's ground
        truth. Same sign convention as the cube stack's contact sensors."""
        return self._spring_force_w

    # ── lifecycle ───────────────────────────────────────────────────────────

    def _resample_command(self, env_ids: Tensor) -> None:
        super()._resample_command(env_ids)  # pool draw (+ nominal blend)

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
        self.force_mag[ids] = torch.empty(m, device=self.device).uniform_(
            *self.cfg.force_range
        )
        # Contact windows hold longer than free draws (steady-state force
        # tracking needs sustained windows).
        self.time_left[ids] = torch.empty(m, device=self.device).uniform_(
            *self.cfg.contact_resampling_time_range
        )
        self.phantom[ids] = (
            torch.rand(m, device=self.device) < self.cfg.phantom_prob
        ).float()
        # 3-D force: tangential fraction + in-plane angle, fixed per window
        # (the WORLD direction still rotates with the gliding target frame).
        ratio = torch.empty(m, device=self.device).uniform_(
            0.0, self.cfg.tangential_ratio_max
        )
        ang = torch.empty(m, device=self.device).uniform_(0.0, 2.0 * torch.pi)
        self._tan_u[ids, 0] = ratio * torch.cos(ang)
        self._tan_u[ids, 1] = ratio * torch.sin(ang)

    def _update_command(self) -> None:
        super()._update_command()
        cfg = self.cfg
        dt = self._env.step_dt
        masked = self.contact_mask > 0.5

        # Whole-window press semantics: the force command, the field and the
        # rewards all live on the full window — at deploy the sender issues
        # (pose, mask, force) together; no transition/arrival concept exists.
        self._contact_age += dt * masked.float()
        ramp_in = (self._contact_age / cfg.force_ramp_time).clamp(0.0, 1.0)
        ramp_out = (self.time_left / cfg.force_ramp_time).clamp(0.0, 1.0)
        f_cmd_n = (
            self.force_mag * torch.minimum(ramp_in, ramp_out) * masked.float()
        )
        self._force_cmd_local = torch.stack(
            (f_cmd_n, f_cmd_n * self._tan_u[:, 0], f_cmd_n * self._tan_u[:, 1]),
            dim=-1,
        )
        self._force_cmd_mag = self._force_cmd_local.norm(dim=-1)

        # ── unilateral normal spring riding the gliding reference ───────────
        tgt_p, tgt_q = self.target_pose_w()
        ex = torch.zeros(self.num_envs, 3, device=self.device)
        ex[:, 0] = 1.0
        n_hat = quat_apply(tgt_q, ex)
        anchor = tgt_p - (f_cmd_n / cfg.k_ext).unsqueeze(-1) * n_hat
        anchor_v = torch.where(
            self._anchor_fresh.unsqueeze(-1),
            torch.zeros_like(anchor),
            (anchor - self._anchor_prev) / dt,
        )
        self._anchor_prev = anchor.detach().clone()
        self._anchor_fresh[:] = False

        tip = self.robot.data.body_link_pos_w[:, self.body_idx]
        tip_v = self.robot.data.body_link_lin_vel_w[:, self.body_idx]
        pen = ((tip - anchor) * n_hat).sum(-1)
        pen_rate = ((tip_v - anchor_v) * n_hat).sum(-1)

        # Command-scaled cap (ramps from 0 with the command): jump-draw
        # overlaps load gently instead of firing at full penetration, and the
        # plant stays inside the QP's hand cone bound (same 1.25·f_cmd).
        cap = (cfg.f_headroom * f_cmd_n).clamp(max=cfg.f_cap)
        f_n = (cfg.k_ext * pen + cfg.d_ext * pen_rate).clamp(min=0.0)
        f_n = torch.minimum(f_n, cap)

        live = masked & (self.phantom < 0.5)
        f_n = f_n * live.float()
        # Tangential rides the realized normal (friction-like): the emitted
        # force is the commanded VECTOR scaled by f_n/F_n ∈ [0, f_headroom].
        scale = f_n / f_cmd_n.clamp(min=1e-6)
        f_cmd_w = quat_apply(tgt_q, self._force_cmd_local)
        self._spring_force_w = scale.unsqueeze(-1) * f_cmd_w  # hand ON surface

        # Reaction ON the robot. xfrc persists until the next write, so every
        # env is written every step (zeros where the spring is inactive).
        self.robot.write_external_wrench_to_sim(
            forces=(-self._spring_force_w).unsqueeze(1),
            torques=self._zero3,
            body_ids=[self.body_idx],
        )

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
            env_ids=env_ids,
            body_ids=[self.body_idx],
        )
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

    def _debug_vis_impl(self, visualizer: "DebugVisualizer") -> None:
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
                    visualizer.add_arrow(
                        start=start,
                        end=start + _SCALE * f,
                        color=color,
                        width=0.010,
                        label=f"spring_force_{kind}_{tag}_{b}",
                    )


@dataclass(kw_only=True)
class EeSpringContactCommandCfg(EeTargetCommandCfg):
    """Config for :class:`EeSpringContactCommand`."""

    contact_prob: float = 0.3
    """Per-resample probability that the new target is a contact command."""
    phantom_prob: float = 0.10
    """Fraction of CONTACT draws with a live mask/force but NO spring."""
    contact_resampling_time_range: tuple[float, float] = (3.0, 6.0)
    """Hold time of masked windows (free draws keep resampling_time_range)."""
    force_range: tuple[float, float] = (5.0, 25.0)
    """NORMAL-component range [N]; |F⃗_cmd| ≤ F_n·√(1 + tangential_ratio²)."""
    tangential_ratio_max: float = 0.5
    """Max |F_t|/F_n of the 3-D force command (ratio uniform in [0, max],
    random in-plane angle, fixed per window). Keep < mu_hand (0.6) so every
    commanded vector lies strictly inside the QP's hand cone-AABB bounds."""
    force_ramp_time: float = 0.6
    """Force-command ramp at window edges [s] — physically, the anchor's
    travel time; also ramps the command-scaled force cap from zero."""

    k_ext: float = 1000.0
    """Virtual surface stiffness [N/m]; 25 N ⇔ 2.5 cm penetration. Swept
    2026-08-11 over {500, 1000, 2000}: engagement transient scales with k
    (~7 → ~13-22 N/step) and the untrained arm holds LESS force the stiffer
    the field (f/f_cmd 0.34 → 0.13) — softer = gentler early learning.
    Re-anchor to the MEASURED real-dome stiffness when available."""
    d_ext: float = 30.0
    """Virtual surface damping [N·s/m]. Swept 2026-08-11 at k=1000: d=30
    dominates d=10 (glide ring 6.9 vs 7.1, steady noise 0.23 vs 1.34 N; jump
    ring 9.1 vs 11.5) — ζ ≈ 0.3 vs the underdamped 0.1."""
    f_headroom: float = 1.25
    """Spring force cap = f_headroom · F_cmd(t) — mirrors the QP's cone-AABB
    headroom, so the plant never exceeds what the QP models."""
    f_cap: float = 100.0
    """Absolute backstop on the spring force [N]."""

    def build(self, env: "ManagerBasedRlEnv") -> EeSpringContactCommand:
        return EeSpringContactCommand(self, env)
