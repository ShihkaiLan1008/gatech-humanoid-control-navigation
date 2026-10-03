"""SE(2) pose-goal command (CURRENT GOAL step 1) — locomotion_pos's own mdp.

``Se2PoseVelocityCommand`` fills the "twist" slot with a goal-attractor twist
computed from a world-latched SE(2) pelvis target, so the shared MPC command
term, the phase clock, the gait rewards, and a trained vel-follower policy all
consume it unchanged.

DESIGN DECISION (2026-07-25): the policy stays a goal-blind vel-follower —
its obs keeps the 3-dim v_des this term emits; the goal exists only in this
planner layer. Consequently the emitted-command DISTRIBUTION must cover the
vel-follower's training distribution: ``ranges`` is pinned to the trained box,
and ``rel_twist_envs`` mixes in plain sampled twists to guarantee coverage of
the sustained extreme combos the attractor rarely produces (fast+max-turn
arcs, long pure-lateral/backward legs).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import Tensor

from mjlab.tasks.velocity.mdp.velocity_command import (
    UniformVelocityCommand,
    UniformVelocityCommandCfg,
)
from mjlab.utils.lab_api.math import wrap_to_pi

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.viewer.debug_visualizer import DebugVisualizer

# Goal-frame gizmo height: nominal G1 pelvis height.
_GOAL_VIZ_Z = 0.79


class Se2PoseVelocityCommand(UniformVelocityCommand):
    """World-latched SE(2) pose goal → goal-attractor body-frame twist.

    Drop-in replacement for the sampled "twist" command: the policy obs, the
    phase clock, the gait rewards, and ``LocoMPCCommand`` all keep reading a
    body-frame (vx, vy, wz) from this term, so a trained vel-follower
    checkpoint runs unchanged while this layer closes the position loop.

    Semantics: the goal is latched to the WORLD frame at resample time — a
    goal re-read in the current robot frame every step would move with the
    robot and never be reached (a velocity command in disguise).

    The G1 walks omnidirectionally, so goal reaching needs no path planner:
    a constant-deceleration attractor (v = sqrt(2 a d), the mobile-robot
    "ramped goto" law — finite-time arrival, no P-creep) plus a heading P law
    is sufficient. The inherited ``ranges`` are the attractor's velocity CLIP
    box and the ``rel_twist_envs`` sampling box — keep them equal to the
    checkpoint's TRAINED distribution (the command_vel curriculum stage), not
    a play-widened box, so the emitted twist stays in-distribution.

    Env modes at resample:
    - goal envs (default): attractor toward a fresh world-latched target;
      on arrival, stand for ``arrive_dwell_range`` then resample early.
    - twist envs (``rel_twist_envs``): a plain uniform twist held to the next
      resample — the original training distribution, coverage guarantee.
    - standing envs (``rel_standing_envs``): zero command (overrides both).
    """

    cfg: "Se2PoseVelocityCommandCfg"

    def __init__(self, cfg: "Se2PoseVelocityCommandCfg", env: "ManagerBasedRlEnv") -> None:
        super().__init__(cfg, env)
        B = self.num_envs
        self.pose_target_w = torch.zeros(B, 3, device=self.device)  # world x, y, yaw
        self.arrived = torch.zeros(B, dtype=torch.bool, device=self.device)
        self.is_twist_env = torch.zeros(B, dtype=torch.bool, device=self.device)
        self.speed_scale = torch.ones(B, device=self.device)
        self._twist_cmd = torch.zeros(B, 3, device=self.device)
        # lin_vel_min bookkeeping: how often the deadband floor actually bites
        self.floor_up_ticks = 0
        self.floor_zero_ticks = 0
        self.lin_cmd_ticks = 0
        self.metrics["dist_to_goal"] = torch.zeros(B, device=self.device)
        self.metrics["yaw_to_goal"] = torch.zeros(B, device=self.device)
        self.metrics["arrived_frac"] = torch.zeros(B, device=self.device)

    # ── sampling ────────────────────────────────────────────────────────────

    def _resample_command(self, env_ids: Tensor) -> None:
        cfg = self.cfg
        n = len(env_ids)
        dev = self.device
        pos = self.robot.data.root_link_pos_w[env_ids, :2]
        heading = self.robot.data.heading_w[env_ids]

        # Env-mode split.
        r = torch.empty(n, device=dev)
        self.is_twist_env[env_ids] = r.uniform_(0.0, 1.0) <= cfg.rel_twist_envs
        self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= cfg.rel_standing_envs

        # World-latched goal (twist envs latch one too but ignore it).
        radius = torch.empty(n, device=dev).uniform_(*cfg.radius_range)
        bearing = heading + torch.empty(n, device=dev).uniform_(*cfg.bearing_range)
        rel_yaw = torch.empty(n, device=dev).uniform_(*cfg.rel_yaw_range)
        self.pose_target_w[env_ids, 0] = pos[:, 0] + radius * bearing.cos()
        self.pose_target_w[env_ids, 1] = pos[:, 1] + radius * bearing.sin()
        self.pose_target_w[env_ids, 2] = wrap_to_pi(heading + rel_yaw)
        self.arrived[env_ids] = False

        # Per-env approach-speed scale (DR: "how fast to reach the target").
        self.speed_scale[env_ids] = torch.empty(n, device=dev).uniform_(
            *cfg.speed_scale_range
        )

        # Held twist for twist envs — uniform over the FULL trained box.
        self._twist_cmd[env_ids, 0] = torch.empty(n, device=dev).uniform_(*cfg.ranges.lin_vel_x)
        self._twist_cmd[env_ids, 1] = torch.empty(n, device=dev).uniform_(*cfg.ranges.lin_vel_y)
        self._twist_cmd[env_ids, 2] = torch.empty(n, device=dev).uniform_(*cfg.ranges.ang_vel_z)

        # Seed the slew filter from the measured twist (no command step at
        # resample), clamped into the trained box so the emitted command never
        # leaves it even when the robot was pushed beyond it.
        vb = self.robot.data.root_link_lin_vel_b[env_ids, 0:2]
        self.vel_command_b[env_ids, 0] = vb[:, 0].clamp(*cfg.ranges.lin_vel_x)
        self.vel_command_b[env_ids, 1] = vb[:, 1].clamp(*cfg.ranges.lin_vel_y)
        self.vel_command_b[env_ids, 2] = self.robot.data.root_link_ang_vel_b[
            env_ids, 2
        ].clamp(*cfg.ranges.ang_vel_z)

    # ── per-step update ─────────────────────────────────────────────────────

    def _update_command(self) -> None:
        cfg = self.cfg
        pos = self.robot.data.root_link_pos_w[:, :2]
        heading = self.robot.data.heading_w

        e_w = self.pose_target_w[:, :2] - pos
        dist = torch.linalg.norm(e_w, dim=-1)
        e_yaw = wrap_to_pi(self.pose_target_w[:, 2] - heading)

        # Arrival latch with hysteresis — goal envs only.
        goal_env = ~self.is_twist_env
        in_tol = (dist < cfg.pos_tol) & (e_yaw.abs() < cfg.yaw_tol) & goal_env
        newly_arrived = in_tol & ~self.arrived
        self.arrived |= in_tol
        self.arrived &= dist < cfg.arrive_hysteresis * cfg.pos_tol

        # Stand for a sampled dwell, then resample a fresh goal early (caps
        # the resampling timer; the base-class timer machinery does the rest).
        if newly_arrived.any():
            ids = newly_arrived.nonzero(as_tuple=False).flatten()
            dwell = torch.empty(len(ids), device=self.device).uniform_(
                *cfg.arrive_dwell_range
            )
            self.time_left[ids] = torch.minimum(self.time_left[ids], dwell)

        # Constant-deceleration approach speed toward the goal.
        v_cap = torch.sqrt(2.0 * cfg.decel * (dist - cfg.pos_tol).clamp(min=0.0))
        dir_w = e_w / dist.clamp(min=1e-6).unsqueeze(-1)
        cos_h, sin_h = heading.cos(), heading.sin()
        vx = v_cap * (cos_h * dir_w[:, 0] + sin_h * dir_w[:, 1])
        vy = v_cap * (-sin_h * dir_w[:, 0] + cos_h * dir_w[:, 1])

        # Direction-preserving clamp into the per-env SCALED trained box
        # (asymmetric per sign: backward/lateral caps differ from forward).
        # speed_scale lowers the cruise cap; the decel curve is unchanged.
        x_lo, x_hi = cfg.ranges.lin_vel_x
        y_lo, y_hi = cfg.ranges.lin_vel_y
        lim_x = torch.where(
            vx >= 0, torch.full_like(vx, x_hi), torch.full_like(vx, max(-x_lo, 0.0))
        ) * self.speed_scale
        lim_y = torch.where(
            vy >= 0, torch.full_like(vy, y_hi), torch.full_like(vy, max(-y_lo, 0.0))
        ) * self.speed_scale
        over = torch.maximum(
            vx.abs() / lim_x.clamp(min=1e-6), vy.abs() / lim_y.clamp(min=1e-6)
        ).clamp(min=1.0)
        vx = vx / over
        vy = vy / over

        z_lo, z_hi = cfg.ranges.ang_vel_z
        wz = (cfg.yaw_stiffness * e_yaw).clamp(min=z_lo, max=z_hi)

        stop = self.arrived | self.is_standing_env
        zeros = torch.zeros_like(vx)
        vx = torch.where(stop, zeros, vx)
        vy = torch.where(stop, zeros, vy)
        wz = torch.where(stop, zeros, wz)

        # Twist envs: the held sampled twist (step changes at resample are the
        # original training distribution — deliberately un-slewed).
        twist_out = torch.where(
            self.is_standing_env.unsqueeze(-1),
            torch.zeros_like(self._twist_cmd),
            self._twist_cmd,
        )

        # Goal envs: slew-limit the linear command (the attractor re-plans
        # every step; without a rate cap it can flip sign across the goal).
        dv = cfg.slew * self._env.step_dt
        slew_x = self.vel_command_b[:, 0] + (vx - self.vel_command_b[:, 0]).clamp(-dv, dv)
        slew_y = self.vel_command_b[:, 1] + (vy - self.vel_command_b[:, 1]).clamp(-dv, dv)

        tw = self.is_twist_env
        self.vel_command_b[:, 0] = torch.where(tw, twist_out[:, 0], slew_x)
        self.vel_command_b[:, 1] = torch.where(tw, twist_out[:, 1], slew_y)
        self.vel_command_b[:, 2] = torch.where(tw, twist_out[:, 2], wz)

        # Deadband floor (sim2real): the EMITTED linear command of a goal env
        # must never lie in (0, lin_vel_min) -- a real base does not move for
        # it.  While the goal still pulls, a small command is raised to the
        # floor (direction kept); once arrived, the slewed residue is snapped
        # to zero instead of being kept alive at the floor.
        if getattr(cfg, "lin_vel_min", 0.0) > 0.0:
            lin = self.vel_command_b[:, :2]
            speed = torch.linalg.norm(lin, dim=-1)
            live = ~tw & (speed > 1e-6)
            small = live & (speed < cfg.lin_vel_min)
            up, down = small & ~self.arrived, small & self.arrived
            if up.any():
                lin[up] *= (cfg.lin_vel_min / speed[up]).unsqueeze(-1)
            if down.any():
                lin[down] = 0.0
            self.floor_up_ticks += int(up.sum())
            self.floor_zero_ticks += int(down.sum())
            self.lin_cmd_ticks += int(live.sum())

    # ── metrics / vis ───────────────────────────────────────────────────────

    def _update_metrics(self) -> None:
        super()._update_metrics()
        pos = self.robot.data.root_link_pos_w[:, :2]
        heading = self.robot.data.heading_w
        # Instantaneous snapshots (reported as the episode mean by reset()).
        # NB in training these mix twist envs (whose goal is inert) — judge
        # goal quality in play (rel_twist_envs=0) or via goal_probe.
        self.metrics["dist_to_goal"] = torch.linalg.norm(
            self.pose_target_w[:, :2] - pos, dim=-1
        )
        self.metrics["yaw_to_goal"] = wrap_to_pi(self.pose_target_w[:, 2] - heading).abs()
        self.metrics["arrived_frac"] = self.arrived.float()

    def _debug_vis_impl(self, visualizer: "DebugVisualizer") -> None:
        super()._debug_vis_impl(visualizer)  # command/actual velocity arrows
        env_indices = visualizer.get_env_indices(self.num_envs)
        if not env_indices:
            return
        tgt = self.pose_target_w.cpu().numpy()
        arrived = self.arrived.cpu().numpy()
        twist = self.is_twist_env.cpu().numpy()
        for b in env_indices:
            if twist[b]:
                continue  # twist envs have no goal semantics
            x, y, psi = float(tgt[b, 0]), float(tgt[b, 1]), float(tgt[b, 2])
            c, s = math.cos(psi), math.sin(psi)
            rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
            # Goal frame at pelvis height: x-axis (red) = target yaw.
            visualizer.add_frame(
                position=np.array([x, y, _GOAL_VIZ_Z]),
                rotation_matrix=rot,
                scale=0.35,
                axis_radius=0.012,
                label=f"se2_goal_frame_{b}",
            )
            color = (0.1, 0.9, 0.2, 0.9) if arrived[b] else (0.95, 0.15, 0.15, 0.9)
            visualizer.add_sphere(
                center=np.array([x, y, _GOAL_VIZ_Z]),
                radius=0.05,
                color=color,
                label=f"se2_goal_state_{b}",
            )


@dataclass(kw_only=True)
class Se2PoseVelocityCommandCfg(UniformVelocityCommandCfg):
    """Config for :class:`Se2PoseVelocityCommand`.

    ``ranges`` (inherited) are BOTH the attractor's velocity clip limits and
    the twist-env sampling box — pin them to the distribution the vel-follower
    policy actually TRAINED under (the command_vel curriculum stage), never a
    play-widened box, or the emitted commands go OOD.
    """

    radius_range: tuple[float, float] = (1.0, 2.5)
    """Goal distance from the robot's pose at resample time [m]."""
    bearing_range: tuple[float, float] = (-math.pi, math.pi)
    """Goal direction relative to the robot's heading at resample time [rad]."""
    rel_yaw_range: tuple[float, float] = (-math.pi, math.pi)
    """Goal yaw relative to the robot's heading at resample time [rad]."""

    decel: float = 0.5
    """Approach deceleration a in v = sqrt(2 a (d - pos_tol)) [m/s^2]."""
    slew: float = 2.0
    """Max rate of change of the commanded linear velocity [m/s^2]."""
    lin_vel_min: float = 0.0
    """Floor on the commanded linear SPEED of goal envs [m/s]; 0 disables.
    A real base cannot track a creeping command, so the band (0, lin_vel_min)
    is forbidden: a nonzero command below the floor is raised to it (direction
    kept) while the goal still pulls, and snapped to zero once arrived."""
    yaw_stiffness: float = 1.5
    """P gain from yaw error to commanded yaw rate [1/s]."""

    pos_tol: float = 0.15
    """Position tolerance for arrival [m]."""
    yaw_tol: float = 0.20
    """Yaw tolerance for arrival [rad]."""
    arrive_hysteresis: float = 2.0
    """Un-arrive only when dist > arrive_hysteresis * pos_tol (no chatter)."""
    arrive_dwell_range: tuple[float, float] = (1.0, 2.0)
    """Stand this long after arrival, then resample a fresh goal early [s]."""

    rel_twist_envs: float = 0.0
    """Fraction of envs that get a plain sampled twist (uniform in ``ranges``,
    held to the next resample) instead of the goal attractor. Training device:
    guarantees the emitted-command distribution covers the vel-follower's
    original sampling distribution (sustained fast+max-turn arcs, long
    pure-lateral/backward legs — combos the attractor rarely sustains).
    Set 0.0 in play/deployment."""

    speed_scale_range: tuple[float, float] = (1.0, 1.0)
    """Per-env scale on the LINEAR clip box, sampled at resample — domain
    randomization for "how fast to approach the target" (deployment planners
    may run gentler profiles). wz cap and the decel curve are unscaled."""

    def build(self, env: "ManagerBasedRlEnv") -> Se2PoseVelocityCommand:
        return Se2PoseVelocityCommand(self, env)
