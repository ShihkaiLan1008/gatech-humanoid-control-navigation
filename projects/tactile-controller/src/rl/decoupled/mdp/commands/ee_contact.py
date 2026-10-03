"""Tactile-tip SE(3) target + contact mask + force command (joint_mpc layer).

``EeContactTargetCommand`` extends the reach-pool :class:`EeTargetCommand`
with the hand-force layer ported from the whole-body stack's
``HandForceEePoseCommand`` (mpc_online_tracking/tasks/hand_force) — MINUS all
of that stack's IK machinery: the decoupled upper policy tracks the EE target
directly, so there is no DLS arm reference here.

Mechanics (all measured/decided on the source stack, carried over verbatim):

* Contact is a MASK on the shared pose draw — the pose comes from the pool
  (reachable by construction), a Bernoulli draw per resample decides whether a
  press cube materializes THERE. Independent per hand (bimanual ~p²).
* The press cube RIDES THE GLIDING REFERENCE for the whole window (user
  decision 2026-08-11, reinstating the original behavior after a
  goal-anchored detour measured worse): the surface is always at the pose
  target — same semantics as the (independent) virtual-spring task. It rides the BASE
  frame too (walking never carries the hand away from it) and parks
  underground when unmasked. NEVER re-place a surface into a pressing hand
  (mocap overlap resolves with ~7 kN — repo rule).
* Force command ramps 0→f over ``force_ramp_time`` from WINDOW OPEN and back
  to 0 before the window closes, so the command hits zero before the cube
  parks. There is no arrival concept anywhere (transition is a goal too).
* Phantom windows (``phantom_prob`` of contact draws): mask/force live, cube
  stays parked — the deploy case "commanded contact where nothing is". Force
  rewards free-pass phantom envs; discrimination must come from proprio
  (measured force never enters the actor obs — user rule).
* The first draw after an episode reset is forced free (no contact).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import Tensor

from mjlab.utils.lab_api.math import quat_apply

from rl.g1_assets.press_cube import (
    CUBE_FACE,
    CUBE_THICKNESS,
    FACE_OFFSET,
    PARK_OFFSETS,
)

from .ee_tartget import EeTargetCommand, EeTargetCommandCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.sensor import ContactSensor
    from mjlab.viewer.debug_visualizer import DebugVisualizer

_CMD_FORCE_VIS = (0.10, 0.90, 0.20, 0.95)   # commanded (on surface)
_MEAS_FORCE_VIS = (1.00, 0.35, 0.10, 0.95)  # measured (on surface)


class EeContactTargetCommand(EeTargetCommand):
    """Pool-drawn tactile-tip pose + contact mask + force, materialized by a
    per-hand press cube riding the command."""

    cfg: EeContactTargetCommandCfg

    def __init__(self, cfg: EeContactTargetCommandCfg, env: ManagerBasedRlEnv) -> None:
        super().__init__(cfg, env)
        B = self.num_envs
        dev = self.device
        self._cube = env.scene[cfg.cube_entity_name] if cfg.cube_entity_name else None
        names = cfg.contact_sensor_name
        if isinstance(names, str):
            names = (names,)
        self._sensors: list[ContactSensor] = [env.scene[n] for n in (names or ())]

        self.contact_mask = torch.zeros(B, device=dev)
        self.force_mag = torch.zeros(B, device=dev)
        self._contact_age = torch.zeros(B, device=dev)
        self._force_cmd_mag = torch.zeros(B, device=dev)
        self.phantom = torch.zeros(B, device=dev)
        self._just_reset = torch.zeros(B, dtype=torch.bool, device=dev)

        self._park_pose = torch.zeros(B, 7, device=dev)
        self._park_pose[:, 3] = 1.0
        if self._cube is not None:
            side = "left" if "left" in cfg.cube_entity_name else "right"
            self._park_pose[:, :3] = env.scene.env_origins + torch.tensor(
                PARK_OFFSETS[side], device=dev
            )
        self._all_ids = torch.arange(B, device=dev)

        # Spawn-overlap guard state: body ids of BOTH tactile tips (a cube
        # pairs with BOTH domes — the collision bits carry no left/right).
        self._tip_body_ids = [self.robot.find_bodies(f)[0][0] for f in cfg.ee_frames]
        self._materialized = torch.zeros(B, dtype=torch.bool, device=dev)

        self.metrics["contact_duty"] = torch.zeros(B, device=dev)
        self.metrics["phantom_duty"] = torch.zeros(B, device=dev)
        self.metrics["force_track_err"] = torch.zeros(B, device=dev)
        self.metrics["spawn_demoted"] = torch.zeros(B, device=dev)

    # ── command tuple accessors ─────────────────────────────────────────────

    @property
    def press_dir_b(self) -> Tensor:
        """Commanded press axis (+x of the GLIDING TARGET frame), robot frame."""
        ex = torch.zeros(self.num_envs, 3, device=self.device)
        ex[:, 0] = 1.0
        return quat_apply(self.pose_command_b[:, 3:], ex)

    @property
    def press_dir_w(self) -> Tensor:
        """Commanded press axis in the world frame (target frame's +x). (B, 3)."""
        _, quat_w = self.target_pose_w()
        ex = torch.zeros(self.num_envs, 3, device=self.device)
        ex[:, 0] = 1.0
        return quat_apply(quat_w, ex)

    @property
    def force_cmd_mag(self) -> Tensor:
        """Current commanded force magnitude (ramps applied). (B,)."""
        return self._force_cmd_mag

    @property
    def force_cmd_b(self) -> Tensor:
        """Commanded hand-on-surface force, robot frame. (B, 3)."""
        return self._force_cmd_mag.unsqueeze(-1) * self.press_dir_b

    @property
    def force_cmd_w(self) -> Tensor:
        """Commanded hand-on-surface force, world frame. (B, 3)."""
        return self._force_cmd_mag.unsqueeze(-1) * self.press_dir_w

    def measured_force_w(self) -> Tensor:
        """Measured hand-on-surface force, summed over the surface contact
        sensors. (B, 3)."""
        out = torch.zeros(self.num_envs, 3, device=self.device)
        for sensor in self._sensors:
            f = sensor.data.force
            if f is not None:
                out = out + f.sum(dim=1)
        return out

    # ── lifecycle ───────────────────────────────────────────────────────────

    def _resample_command(self, env_ids: Tensor) -> None:
        super()._resample_command(env_ids)  # pool draw (+ nominal_blend)

        n = len(env_ids)
        draw = torch.rand(n, device=self.device) < self.cfg.contact_prob
        draw &= ~self._just_reset[env_ids]
        self._just_reset[env_ids] = False

        self.contact_mask[env_ids] = 0.0
        self.force_mag[env_ids] = 0.0
        self._contact_age[env_ids] = 0.0
        self._force_cmd_mag[env_ids] = 0.0
        self.phantom[env_ids] = 0.0
        self._materialized[env_ids] = False

        ids = env_ids[draw]
        if len(ids) == 0:
            return
        m = len(ids)
        self.contact_mask[ids] = 1.0
        self.force_mag[ids] = torch.empty(m, device=self.device).uniform_(
            *self.cfg.force_range
        )
        # Contact windows hold longer than free draws: overwrite the base
        # resample timer (steady-state force tracking needs sustained windows).
        self.time_left[ids] = torch.empty(m, device=self.device).uniform_(
            *self.cfg.contact_resampling_time_range
        )
        self.phantom[ids] = (
            torch.rand(m, device=self.device) < self.cfg.phantom_prob
        ).float()

    def _update_command(self) -> None:
        super()._update_command()
        cfg = self.cfg
        dt = self._env.step_dt
        masked = self.contact_mask > 0.5

        # Whole-window press semantics (no arrival concept anywhere): the
        # cube rides the gliding reference, so contact is available from
        # window open and the force ramp runs on window age.
        self._contact_age += dt * masked.float()
        ramp_in = (self._contact_age / cfg.force_ramp_time).clamp(0.0, 1.0)
        ramp_out = (self.time_left / cfg.force_ramp_time).clamp(0.0, 1.0)
        self._force_cmd_mag = (
            self.force_mag * torch.minimum(ramp_in, ramp_out) * masked.float()
        )

        if self._cube is not None:
            # Cube rides the GLIDING REFERENCE (the always-present surface —
            # measured better than a goal-anchored one, 2026-08-11).
            pos_w, quat_w = self.target_pose_w()
            center = pos_w + (FACE_OFFSET + CUBE_THICKNESS / 2) * self.press_dir_w
            real = masked & (self.phantom < 0.5)
            # Spawn guard: materializing the cube around a tip ALREADY inside
            # its volume resolves the mocap overlap with ~kN forces (the 7 kN
            # class). Pool draws can land near where a hand currently is, so a
            # window whose cube would swallow EITHER tip demotes to phantom.
            newly = real & ~self._materialized
            if newly.any():
                demote = newly & self._tip_inside_cube(center, quat_w)
                if demote.any():
                    # Demote to a FREE window (mask off) — NOT phantom: phantom
                    # free-passes the whole force family at full pay, which
                    # made spawn overlaps a farmable jackpot (measured on the
                    # first force run: demotion rate ROSE 0.15→0.23/episode
                    # while every force reward sat flat).
                    self.contact_mask[demote] = 0.0
                    self.force_mag[demote] = 0.0
                    self._force_cmd_mag[demote] = 0.0
                    self.metrics["spawn_demoted"][demote] += 1.0
                    real = (self.contact_mask > 0.5) & (self.phantom < 0.5)
            self._materialized |= real
            poses = torch.cat([center, quat_w], dim=-1)
            poses = torch.where(real.unsqueeze(-1), poses, self._park_pose)
            self._cube.write_mocap_pose_to_sim(poses, env_ids=self._all_ids)

    def _tip_inside_cube(
        self,
        center: Tensor,
        quat_w: Tensor,
        pen_margin: float = 0.02,
        dome_r: float = 0.05,
    ) -> Tensor:
        """(B,) True where ANY tactile tip sits DEEP inside the would-be cube.

        The legitimate press puts the apex ON the −x face plane (window opens
        with the target at the hand — that is the design, not a hazard), so
        the test is asymmetric: only a tip more than ``pen_margin`` PAST the
        face plane, laterally within the face, counts — that spawn resolves a
        deep dome↔box overlap with ~kN forces."""
        q_inv = quat_w.clone()
        q_inv[:, 1:] = -q_inv[:, 1:]
        inside = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        for bid in self._tip_body_ids:
            tip = self.robot.data.body_link_pos_w[:, bid]
            d = quat_apply(q_inv, tip - center)
            inside |= (
                (d[:, 0] > -CUBE_THICKNESS / 2 + pen_margin)
                & (d[:, 0] < CUBE_THICKNESS / 2 + dome_r)
                & (d[:, 1].abs() < CUBE_FACE / 2 + dome_r)
                & (d[:, 2].abs() < CUBE_FACE / 2 + dome_r)
            )
        return inside

    def reset(self, env_ids) -> dict[str, float]:
        self.contact_mask[env_ids] = 0.0
        self.force_mag[env_ids] = 0.0
        self._contact_age[env_ids] = 0.0
        self._force_cmd_mag[env_ids] = 0.0
        self.phantom[env_ids] = 0.0
        self._materialized[env_ids] = False
        self._just_reset[env_ids] = True
        if self._cube is not None:
            self._cube.write_mocap_pose_to_sim(
                self._park_pose[env_ids], env_ids=env_ids
            )
        return super().reset(env_ids)

    def _update_metrics(self) -> None:
        super()._update_metrics()
        masked = self.contact_mask > 0.5
        self.metrics["contact_duty"] = self.contact_mask.clone()
        self.metrics["phantom_duty"] = self.phantom.clone()
        err = (self.measured_force_w() - self.force_cmd_w).norm(dim=-1)
        # Phantom windows excluded: their "error" is ‖f_cmd‖ by construction.
        self.metrics["force_track_err"] = err * (masked & (self.phantom < 0.5)).float()

    def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
        super()._debug_vis_impl(visualizer)  # goal frame
        env_indices = visualizer.get_env_indices(self.num_envs)
        if not env_indices:
            return
        masked_np = (self.contact_mask > 0.5).cpu().numpy()
        tgt_np = self.target_pose_w()[0].cpu().numpy()
        fcmd_np = self.force_cmd_w.cpu().numpy()
        fmeas_np = self.measured_force_w().cpu().numpy()
        ee_np = self.measured_ee_pose_w()[0].cpu().numpy()
        tag = self.cfg.body_name
        _SCALE = 0.01  # m per N
        for b in env_indices:
            if not masked_np[b]:
                continue
            for kind, start, f, color in (
                ("cmd", tgt_np[b], fcmd_np[b], _CMD_FORCE_VIS),
                ("meas", ee_np[b], fmeas_np[b], _MEAS_FORCE_VIS),
            ):
                if float(np.linalg.norm(f)) > 1.0:
                    visualizer.add_arrow(
                        start=start,
                        end=start + _SCALE * f,
                        color=color,
                        width=0.010,
                        label=f"hand_force_{kind}_{tag}_{b}",
                    )


@dataclass(kw_only=True)
class EeContactTargetCommandCfg(EeTargetCommandCfg):
    """Config for :class:`EeContactTargetCommand`."""

    contact_prob: float = 0.3
    """Per-resample probability that the new target is a contact command."""
    phantom_prob: float = 0.10
    """Fraction of CONTACT draws with a live mask/force but NO materialized
    surface (cube stays parked)."""
    contact_resampling_time_range: tuple[float, float] = (3.0, 6.0)
    """Hold time of masked windows (free draws keep resampling_time_range)."""
    force_range: tuple[float, float] = (5.0, 20.0)
    force_ramp_time: float = 0.6
    """Linear ramp-in/out of the force command at window edges [s]."""
    cube_entity_name: str | None = "left_press_cube"
    """Riding-surface mocap prop; None = no cube."""
    contact_sensor_name: str | tuple[str, ...] | None = "lhand_cube_contact"
    """Surface contact sensor(s); measured_force_w sums over them."""

    def build(self, env: ManagerBasedRlEnv) -> EeContactTargetCommand:
        return EeContactTargetCommand(self, env)
