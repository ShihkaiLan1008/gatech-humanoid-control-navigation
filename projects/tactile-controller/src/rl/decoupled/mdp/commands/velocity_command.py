from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch

from mjlab.tasks.velocity.mdp.velocity_command import (
    UniformVelocityCommand,
    UniformVelocityCommandCfg,
)
if TYPE_CHECKING:
  import viser

  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer


class TrackingMetricsVelocityCommand(UniformVelocityCommand):
    """:class:`UniformVelocityCommand` augmented with exp-kernel tracking"""

    cfg: "TrackingMetricsVelocityCommandCfg"

    def __init__(
        self,
        cfg: "TrackingMetricsVelocityCommandCfg",
        env: "ManagerBasedRlEnv",
    ) -> None:
        super().__init__(cfg, env)
        self.metrics["tracking_lin_score"] = torch.zeros(
            self.num_envs, device=self.device
        )
        self.metrics["tracking_ang_score"] = torch.zeros(
            self.num_envs, device=self.device
        )

    def create_gui(self, name, server, get_env_idx, on_change=None, request_action=None) -> None:
        """Viser joystick of the base class; a degenerate range (max < 0.1)
        would fail the "Max" slider's bounds, so the GUI sees it as 0.1."""
        import copy
        ranges = copy.deepcopy(self.cfg.ranges)
        cfg_ranges = self.cfg.ranges
        for axis in ("lin_vel_x", "lin_vel_y", "ang_vel_z"):
            lo, hi = getattr(ranges, axis)
            setattr(ranges, axis, (lo, max(hi, 0.1)))
        self.cfg.ranges = ranges
        try:
            super().create_gui(name, server, get_env_idx, on_change=on_change,
                               request_action=request_action)
        finally:
            self.cfg.ranges = cfg_ranges

    def _update_metrics(self) -> None:
        super()._update_metrics()

        actual_lin = self.robot.data.root_link_lin_vel_b
        xy_err_sq = torch.sum(
            torch.square(self.vel_command_b[:, :2] - actual_lin[:, :2]), dim=1
        )
        z_err_sq = torch.square(actual_lin[:, 2])
        lin_score = torch.exp(
            -(xy_err_sq + z_err_sq) / (self.cfg.metric_std_lin ** 2)
        )
        self.metrics["tracking_lin_score"] += lin_score

        actual_ang = self.robot.data.root_link_ang_vel_b
        yaw_err_sq = torch.square(self.vel_command_b[:, 2] - actual_ang[:, 2])
        xy_ang_sq = torch.sum(torch.square(actual_ang[:, :2]), dim=1)
        ang_score = torch.exp(
            -(yaw_err_sq + xy_ang_sq) / (self.cfg.metric_std_ang ** 2)
        )
        self.metrics["tracking_ang_score"] += ang_score

    def reset(self, env_ids: torch.Tensor | slice | None) -> dict[str, float]:
        """Normalise the accumulated tracking-score sums by ``N_max`` so the"""
        if isinstance(env_ids, torch.Tensor):
            n_max = max(int(self._env.max_episode_length), 1)
            self.metrics["tracking_lin_score"][env_ids] = (
                self.metrics["tracking_lin_score"][env_ids] / n_max
            )
            self.metrics["tracking_ang_score"][env_ids] = (
                self.metrics["tracking_ang_score"][env_ids] / n_max
            )
        return super().reset(env_ids)

@dataclass(kw_only=True)
class TrackingMetricsVelocityCommandCfg(UniformVelocityCommandCfg):
    """Drop-in replacement for :class:`UniformVelocityCommandCfg` that emits"""

    metric_std_lin: float = 0.5
    metric_std_ang: float = 0.7071067811865476

    def build(self, env: "ManagerBasedRlEnv") -> TrackingMetricsVelocityCommand:
        return TrackingMetricsVelocityCommand(self, env)

    @classmethod
    def from_uniform(
        cls,
        base: UniformVelocityCommandCfg,
        **overrides,
    ) -> "TrackingMetricsVelocityCommandCfg":
        """Construct from an existing :class:`UniformVelocityCommandCfg`,"""
        import dataclasses
        fields = {
            f.name: getattr(base, f.name) for f in dataclasses.fields(base)
        }
        fields.update(overrides)
        return cls(**fields)

