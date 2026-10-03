"""Executor-owned pose servo using onboard odometry, with the trained twist limits.

Maze geometry/reset support is inherited; its truth-fed itinerary and pose
feedback are replaced. Targets and feedback share the estimated world frame.
"""
from dataclasses import dataclass

import torch
from mjlab.utils.lab_api.math import wrap_to_pi
from rl.eval.mdp.maze_twist import MazeFollowTwistCommand, MazeFollowTwistCommandCfg


class EstimatedPoseTwistCommand(MazeFollowTwistCommand):
    pose_provider = None

    def _resample_command(self, env_ids):
        # The executor owns every target. No random goal or wall-follow hop.
        pass

    def _update_command(self):
        if self.pose_provider is None:
            # Environment reset precedes construction of the onboard estimator.
            self.vel_command_b.zero_()
            return
        xy, yaw = self.pose_provider()
        pose = torch.as_tensor([float(xy[0]), float(xy[1]), float(yaw)],
                               device=self.device, dtype=self.pose_target_w.dtype)
        cfg = self.cfg
        error = self.pose_target_w[:, :2] - pose[:2]
        dist = torch.linalg.vector_norm(error, dim=-1)
        yaw_error = wrap_to_pi(self.pose_target_w[:, 2] - pose[2])
        self.arrived |= (dist < cfg.pos_tol) & (yaw_error.abs() < cfg.yaw_tol)
        self.arrived &= ((dist < cfg.arrive_hysteresis * cfg.pos_tol)
                         & (yaw_error.abs() < cfg.arrive_hysteresis * cfg.yaw_tol))

        # Same deceleration, direction-preserving velocity limits and slew
        # as the checkpoint's maze attractor; only the feedback source changes.
        speed = torch.sqrt(2.0 * cfg.decel * (dist - cfg.pos_tol).clamp(min=0.0))
        direction = error / dist.clamp(min=1e-6).unsqueeze(-1)
        c, s = pose[2].cos(), pose[2].sin()
        vx = speed * (c * direction[:, 0] + s * direction[:, 1])
        vy = speed * (-s * direction[:, 0] + c * direction[:, 1])
        x_lo, x_hi = cfg.ranges.lin_vel_x
        y_lo, y_hi = cfg.ranges.lin_vel_y
        lim_x = torch.where(vx >= 0, x_hi, max(-x_lo, 0.0)) * self.speed_scale
        lim_y = torch.where(vy >= 0, y_hi, max(-y_lo, 0.0)) * self.speed_scale
        over = torch.maximum(vx.abs() / lim_x.clamp(min=1e-6),
                             vy.abs() / lim_y.clamp(min=1e-6)).clamp(min=1.0)
        vx, vy = vx / over, vy / over
        wz = (cfg.yaw_stiffness * yaw_error).clamp(*cfg.ranges.ang_vel_z)
        desired = torch.stack([vx, vy, wz], dim=-1)
        desired[self.arrived] = 0.0
        dv = cfg.slew * self._env.step_dt
        self.vel_command_b[:, :2] += (desired[:, :2] - self.vel_command_b[:, :2]).clamp(-dv, dv)
        self.vel_command_b[:, 2] = desired[:, 2]


@dataclass(kw_only=True)
class EstimatedPoseTwistCfg(MazeFollowTwistCommandCfg):
    def build(self, env):
        return EstimatedPoseTwistCommand(self, env)
