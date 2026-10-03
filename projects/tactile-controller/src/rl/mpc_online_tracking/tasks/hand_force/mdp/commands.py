from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
from mjlab.utils.lab_api.math import quat_apply, quat_from_euler_xyz, quat_unique
from torch import Tensor

from rl.g1_assets.press_cube import CUBE_THICKNESS, FACE_OFFSET, PARK_OFFSETS

from ...hand_pose.mdp.commands import (
    UniformEePoseCommandCfg,
    UniformEePoseIkCommand,
    UniformEePoseIkCommandCfg,
)
from ...hand_pose.mdp.reach_pool import (
    ReachPoolEePoseIkCommand,
    ReachPoolEePoseIkCommandCfg,
)

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.sensor import ContactSensor
    from mjlab.viewer.debug_visualizer import DebugVisualizer

_CMD_FORCE_VIS = (0.10, 0.90, 0.20, 0.95)  # commanded (on surface)
_MEAS_FORCE_VIS = (1.00, 0.35, 0.10, 0.95)  # measured (on surface)


class HandForceEePoseCommand(UniformEePoseIkCommand):
    """Stage-2 hand command: EE pose (stage-1 machinery untouched) + mask +
    force, materialized by the per-hand press cube."""

    cfg: HandForceEePoseCommandCfg

    def __init__(self, cfg: HandForceEePoseCommandCfg, env: ManagerBasedRlEnv) -> None:
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

        self.metrics["contact_duty"] = torch.zeros(B, device=dev)
        self.metrics["phantom_duty"] = torch.zeros(B, device=dev)
        self.metrics["force_track_err"] = torch.zeros(B, device=dev)

    # ── command tuple accessors ─────────────────────────────────────────────

    @property
    def press_dir_b(self) -> Tensor:
        """Commanded press axis (+x of the command) in the sample frame. (B, 3)."""
        ex = torch.zeros(self.num_envs, 3, device=self.device)
        ex[:, 0] = 1.0
        return quat_apply(self.pose_command_b[:, 3:], ex)

    @property
    def press_dir_w(self) -> Tensor:
        """Commanded press axis in the world frame. (B, 3)."""
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
        """Commanded hand-on-surface force, sample frame. (B, 3)."""
        return self._force_cmd_mag.unsqueeze(-1) * self.press_dir_b

    @property
    def force_cmd_w(self) -> Tensor:
        """Commanded hand-on-surface force, world frame. (B, 3)."""
        return self._force_cmd_mag.unsqueeze(-1) * self.press_dir_w

    def measured_force_w(self) -> Tensor:
        """Measured hand-on-surface force, summed over the surface contact
        sensors (one cube sensor here; per-wall sensors in wall_following).
        (B, 3)."""
        out = torch.zeros(self.num_envs, 3, device=self.device)
        for sensor in self._sensors:
            f = sensor.data.force
            if f is not None:
                out = out + f.sum(dim=1)
        return out


    def _resample_command(self, env_ids: Tensor) -> None:
        super()._resample_command(env_ids)  # free-space box, all envs

        n = len(env_ids)
        draw = torch.rand(n, device=self.device) < self.cfg.contact_prob
        draw &= ~self._just_reset[env_ids]
        self._just_reset[env_ids] = False

        self.contact_mask[env_ids] = 0.0
        self.force_mag[env_ids] = 0.0
        self._contact_age[env_ids] = 0.0
        self._force_cmd_mag[env_ids] = 0.0
        self.phantom[env_ids] = 0.0

        ids = env_ids[draw]
        if len(ids) == 0:
            return
        m = len(ids)
        # Contact is a MASK on the shared pose draw (2026-08-01): the command
        # keeps whatever pose super() sampled and the cube face materializes
        # THERE. The old `contact_ranges` sub-box overwrite is gone — it was a
        # crude reachability guarantee (its size was even calibrated through
        # the biased penalty-IK) that the reach pool now provides exactly, and
        # an overwrite would de-sync the pool's per-pose solver anchor.
        # (`cfg.contact_ranges` itself stays: wall_following samples its wall
        # spot bands from it.)
        self.contact_mask[ids] = 1.0
        self.force_mag[ids] = torch.empty(m, device=self.device).uniform_(
            *self.cfg.force_range
        )
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

        self._contact_age += dt * masked.float()
        ramp_in = (self._contact_age / cfg.force_ramp_time).clamp(0.0, 1.0)
        ramp_out = (self.time_left / cfg.force_ramp_time).clamp(0.0, 1.0)
        self._force_cmd_mag = (
            self.force_mag * torch.minimum(ramp_in, ramp_out) * masked.float()
        )

        if self._cube is not None:
            pos_w, quat_w = self.target_pose_w()
            center = pos_w + (FACE_OFFSET + CUBE_THICKNESS / 2) * self.press_dir_w
            poses = torch.cat([center, quat_w], dim=-1)
            real = masked & (self.phantom < 0.5)
            poses = torch.where(real.unsqueeze(-1), poses, self._park_pose)
            self._cube.write_mocap_pose_to_sim(poses, env_ids=self._all_ids)

    def reset(self, env_ids) -> dict[str, float]:
        self.contact_mask[env_ids] = 0.0
        self.force_mag[env_ids] = 0.0
        self._contact_age[env_ids] = 0.0
        self._force_cmd_mag[env_ids] = 0.0
        self.phantom[env_ids] = 0.0
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
class HandForceEePoseCommandCfg(UniformEePoseIkCommandCfg):
    """Config for :class:`HandForceEePoseCommand`."""

    contact_prob: float = 0.3
    """Per-resample probability that the new target is a contact command.
    Independent per hand: bimanual presses arise with p^2 (~9%)."""
    phantom_prob: float = 0.0
    """Fraction of CONTACT draws that are PHANTOM: mask/force command live,
    surface never materializes (cube stays parked). Trains reach-and-probe
    without blind weight-commit for the deploy case "commanded contact where
    nothing is". Force rewards free-pass phantom envs; the actor must
    discriminate via proprio (measured force never enters the actor obs —
    user rule). Default 0 (wall_following keeps it off: its walls are real)."""
    contact_resampling_time_range: tuple[float, float] = (3.0, 6.0)
    """Hold time of masked windows (free draws keep resampling_time_range) —
    steady-state force tracking needs sustained windows."""
    contact_ranges: UniformEePoseCommandCfg.Ranges = None  # type: ignore[assignment]
    """NOT read by this task since the reach-pool switch (2026-08-01) —
    contact draws keep the shared pose draw, whose reachability the pool
    guarantees. Still consumed by wall_following (wall-spot y/z bands) and
    the anchor-press demo."""
    force_range: tuple[float, float] = (5.0, 40.0)
    force_ramp_time: float = 0.6
    """Linear ramp-in/out of the force command at window edges [s] — overlaps
    the reach transit at window open and softens the release before the cube
    parks."""
    cube_entity_name: str | None = "left_press_cube"
    """Riding-surface prop; None = no cube (wall_following: real walls
    materialize the masked commands)."""
    contact_sensor_name: str | tuple[str, ...] | None = "lhand_cube_contact"
    """Surface contact sensor(s); measured_force_w sums over them."""

    def build(self, env: ManagerBasedRlEnv) -> HandForceEePoseCommand:
        return HandForceEePoseCommand(self, env)


class ReachPoolHandForceEePoseCommand(
    HandForceEePoseCommand, ReachPoolEePoseIkCommand
):
    """Hand-force command sampling from the reachable-by-construction pool.

    Cooperative MI, no new logic: `HandForceEePoseCommand._resample_command`'s
    ``super()`` call resolves to :class:`ReachPoolEePoseIkCommand` here, so
    every draw (free AND contact — contact is just a mask on the shared draw)
    is a pool pose with its exact solver anchor seeded. Deliberately a NEW
    class instead of rebasing ``HandForceEePoseCommand``:
    wall_following's ``WallEePoseCommand`` skips its parent's resample via
    ``super(HandForceEePoseCommand, self)`` and writes wall-derived poses — a
    rebased parent would hand it pool draws + STALE anchors."""

    cfg: ReachPoolHandForceEePoseCommandCfg


@dataclass(kw_only=True)
class ReachPoolHandForceEePoseCommandCfg(
    HandForceEePoseCommandCfg, ReachPoolEePoseIkCommandCfg
):
    """Config for :class:`ReachPoolHandForceEePoseCommand` (field union of the
    two bases; `ranges`/`contact_ranges` are inherited but ignored)."""

    def build(self, env: ManagerBasedRlEnv) -> ReachPoolHandForceEePoseCommand:
        return ReachPoolHandForceEePoseCommand(self, env)
