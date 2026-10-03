"""Command generators for terrain-aware G1 locomotion."""

from __future__ import annotations

import math

import torch

from isaaclab.envs.mdp.commands.commands_cfg import UniformVelocityCommandCfg
from isaaclab.envs.mdp.commands.velocity_command import UniformVelocityCommand
from isaaclab.utils.configclass import configclass


class EdgeReturnVelocityCommand(UniformVelocityCommand):
    """Random velocity commands that turn the robot back toward its spawn point near an edge."""

    cfg: "EdgeReturnVelocityCommandCfg"

    def __init__(self, cfg: "EdgeReturnVelocityCommandCfg", env):
        super().__init__(cfg, env)
        self._returning_home = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _update_command(self):
        super()._update_command()

        # Each generated terrain tile is 8 x 8 m. Use the environment origin as
        # the robot's home and turn around before it reaches the tile boundary.
        displacement_w = self.robot.data.root_pos_w.torch[:, :2] - self._env.scene.env_origins[:, :2]
        distance = torch.linalg.norm(displacement_w, dim=-1)
        self._returning_home |= distance >= self.cfg.return_trigger_distance
        self._returning_home &= distance > self.cfg.return_release_distance

        env_ids = self._returning_home.nonzero(as_tuple=False).flatten()
        if len(env_ids) == 0:
            return

        # Desired heading points from the current location back to the spawn point.
        to_home_w = -displacement_w[env_ids]
        target_heading = torch.atan2(to_home_w[:, 1], to_home_w[:, 0])
        heading_error = torch.atan2(
            torch.sin(target_heading - self.robot.data.heading_w.torch[env_ids]),
            torch.cos(target_heading - self.robot.data.heading_w.torch[env_ids]),
        )

        # Slow forward motion while making a large turn, then walk home faster
        # once the robot is facing the safe direction.
        alignment = torch.clamp(torch.cos(heading_error), min=0.0)
        self.vel_command_b[env_ids, 0] = self.cfg.return_speed * alignment
        self.vel_command_b[env_ids, 1] = 0.0
        self.vel_command_b[env_ids, 2] = torch.clamp(
            self.cfg.return_heading_gain * heading_error,
            min=self.cfg.ranges.ang_vel_z[0],
            max=self.cfg.ranges.ang_vel_z[1],
        )


@configclass
class EdgeReturnVelocityCommandCfg(UniformVelocityCommandCfg):
    """Configuration for :class:`EdgeReturnVelocityCommand`."""

    class_type: type = EdgeReturnVelocityCommand
    return_trigger_distance: float = 3.0
    return_release_distance: float = 1.0
    return_speed: float = 0.7
    return_heading_gain: float = 1.5

