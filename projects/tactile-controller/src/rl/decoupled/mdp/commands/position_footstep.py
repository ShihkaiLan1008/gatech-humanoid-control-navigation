"""PLAY-ONLY: SE(2) goal -> footstep plan for the foot_mimic line.

Fourth source for the same ``step_cmd`` interface (clips / Raibert planner /
operator / here: a GOAL).  The term holds one world-frame pelvis SE(2) goal per
env (drawn around the robot when ``auto_goals``, else given by ``set_goal``)
and plans one step per chain switch straight in foot space: the swing foot's
pose at the planned stance, expressed in the measured stance foot's frame and
clamped into the trained landing box.  Heading law: while the goal is far the
planned yaw is the bearing to it (walk facing the goal), inside
``face_goal_dist`` it is the goal's own yaw; a yaw error beyond
``turn_in_place_yaw`` makes the step a pure turn.  No velocity is formed
anywhere; the feedback is the replan from the MEASURED feet at every step.
Arrival (both feet inside the tolerance of their goal poses) enters a stand
that is left only once a foot is past the departure tolerance, whether the
goal moved or the robot drifted.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

from mjlab.utils.lab_api.math import sample_uniform

from .foot_placement import _wrap
from .raibert_footstep import VelocityFootstepCommand, VelocityFootstepCommandCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


class PositionFootstepCommand(VelocityFootstepCommand):
    """Footsteps planned from a world-frame pelvis SE(2) goal.

    Attributes:
        pose_target_w: Current goal (x, y, yaw) in world frame. Shape (B, 3),
            m / rad.
        facing: True while the planned yaw is the bearing to the goal (heading
            law, far from the goal). Shape (B,).
    """

    cfg: "PositionFootstepCommandCfg"

    def __init__(self, cfg: "PositionFootstepCommandCfg", env: ManagerBasedRlEnv):
        assert cfg.fixed_twist == (0.0, 0.0, 0.0), (
            "the goal term owns stepping — the twist must stay zero so the "
            "planner's reset comes up standing"
        )
        super().__init__(cfg, env)
        B = self.num_envs
        self.pose_target_w = torch.zeros(B, 3, device=self.device)
        self.facing = torch.zeros(B, dtype=torch.bool, device=self.device)
        self._need_goal = torch.ones(B, dtype=torch.bool, device=self.device)
        self._turn_sign = torch.zeros(B, device=self.device)
        self._hold = torch.zeros(B, dtype=torch.long, device=self.device)
        self._goal_age = torch.zeros(B, dtype=torch.long, device=self.device)
        for k in ("dist_to_goal", "yaw_to_goal", "goals_reached", "facing", "turning"):
            self.metrics[k] = torch.zeros(B, device=self.device)

    # ── goal ────────────────────────────────────────────────────────────────

    def set_goal(self, env_idx: int, x: float, y: float, yaw: float) -> None:
        """Overwrite one env's goal (world frame, m / rad).  Takes effect at the
        next step boundary; a standing env leaves its stand once a foot is past
        the departure tolerance, so re-sending an unchanged goal is a no-op."""
        self.pose_target_w[env_idx] = torch.tensor(
            [x, y, yaw], device=self.device)
        self._need_goal[env_idx] = False
        self._goal_age[env_idx] = 0

    def _sample_goal(self, ids: torch.Tensor) -> None:
        """Polar draw around the current pelvis pose, clamped to the arena."""
        cfg = self.cfg
        n = len(ids)
        p = self.robot.data.root_link_pos_w[ids, :2]
        heading = self.robot.data.heading_w[ids]
        radius = sample_uniform(*cfg.radius_range, (n,), self.device)
        bearing = heading + sample_uniform(*cfg.bearing_range, (n,), self.device)
        goal = p + radius.unsqueeze(-1) * torch.stack(
            [torch.cos(bearing), torch.sin(bearing)], dim=-1)
        origin = self._env.scene.env_origins[ids, :2]
        rel = goal - origin
        r = rel.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        goal = origin + rel * (r.clamp(max=cfg.arena_radius) / r)
        self.pose_target_w[ids, :2] = goal
        self.pose_target_w[ids, 2] = _wrap(
            heading + sample_uniform(*cfg.rel_yaw_range, (n,), self.device))
        self._need_goal[ids] = False
        self._turn_sign[ids] = 0.0
        self._goal_age[ids] = 0

    # ── geometry ────────────────────────────────────────────────────────────

    @staticmethod
    def _feet_xy_of(pose: torch.Tensor, width: float) -> torch.Tensor:
        """Stance feet (left, right) of pelvis SE(2) poses ``pose`` [K, 3],
        world xy. Shape (K, 2, 2), m."""
        lat = 0.5 * width * torch.stack(
            [-torch.sin(pose[:, 2]), torch.cos(pose[:, 2])], dim=-1)      # +y of pose
        return torch.stack([pose[:, :2] + lat, pose[:, :2] - lat], dim=1)

    def _goal_feet_xy(self, ids: torch.Tensor) -> torch.Tensor:
        """Goal stance feet (left, right) in world xy. Shape (K, 2, 2), m."""
        return self._feet_xy_of(self.pose_target_w[ids], self.cfg.nominal_width)

    def _feet_mid_xy(self, ids: torch.Tensor) -> torch.Tensor:
        """Midpoint of the measured feet, world xy. Shape (K, 2), m."""
        return self.robot_feet_pos_w[ids, :, :2].mean(dim=1)

    def _plan_pose_w(self, ids: torch.Tensor) -> torch.Tensor:
        """Pose the next step aims at: the goal xy with the bearing to it as
        yaw while ``facing``, the goal's own yaw otherwise. Shape (K, 3), m / rad."""
        pose = self.pose_target_w[ids].clone()
        f = self.facing[ids]
        if bool(f.any()):
            d = pose[:, :2] - self._feet_mid_xy(ids)
            pose[:, 2] = torch.where(f, torch.atan2(d[:, 1], d[:, 0]), pose[:, 2])
        return pose

    def _pose_err(self, ids: torch.Tensor, pose: torch.Tensor
                  ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-foot position / yaw error to the stance of ``pose`` [K, 3].
        Shapes (K, 2), m / rad."""
        pos = (self.robot_feet_pos_w[ids, :, :2]
               - self._feet_xy_of(pose, self.cfg.nominal_width)).norm(dim=-1)
        yaw = _wrap(self.robot_feet_yaw[ids] - pose[:, 2].unsqueeze(-1)).abs()
        return pos, yaw

    def _arrived(self, ids: torch.Tensor) -> torch.Tensor:
        pos, yaw = self._pose_err(ids, self.pose_target_w[ids])
        return ((pos <= self.cfg.arrive_pos_tol) & (yaw <= self.cfg.arrive_yaw_tol)).all(dim=-1)

    def _departed(self, ids: torch.Tensor) -> torch.Tensor:
        pos, yaw = self._pose_err(ids, self.pose_target_w[ids])
        return ((pos > self.cfg.depart_pos_tol) | (yaw > self.cfg.depart_yaw_tol)).any(dim=-1)

    def _update_facing(self, ids: torch.Tensor) -> None:
        """Heading-law mode with hysteresis on the feet-midpoint distance."""
        if not self.cfg.heading_law:
            self.facing[ids] = False
            return
        dist = (self.pose_target_w[ids, :2] - self._feet_mid_xy(ids)).norm(dim=-1)
        f = self.facing[ids]
        f = f & ~(dist < self.cfg.face_goal_dist)
        f = f | (dist > self.cfg.face_goal_dist + self.cfg.face_goal_hysteresis)
        self.facing[ids] = f

    def _latched_sign(self, ids: torch.Tensor, err: torch.Tensor) -> torch.Tensor:
        """Turn direction for the signed yaw errors ``err``: inside the ±π band
        the direction chosen when the turn began is kept; the latch clears once
        the error is below ``turn_in_place_yaw``. Shape (K,), ±1."""
        sign = torch.where(err < 0.0, -torch.ones_like(err), torch.ones_like(err))
        amb = err.abs() > math.pi - self.cfg.turn_latch_band
        latched = self._turn_sign[ids]
        sign = torch.where(amb & (latched != 0.0), latched, sign)
        turning = err.abs() > self.cfg.turn_in_place_yaw
        self._turn_sign[ids] = torch.where(turning, sign, torch.zeros_like(sign))
        return sign

    # ── the goal replaces the planner ───────────────────────────────────────

    def _plan_target(self, env_ids, swing, stance_yaw, s, T):
        """The swing foot's pose at the planned stance (goal xy, heading-law
        yaw) in the MEASURED stance foot's frame (mirrored, unclamped); a pure
        turn (x = dy = 0) when the yaw error exceeds ``turn_in_place_yaw``."""
        del T
        idx = torch.arange(len(env_ids), device=self.device)
        plan = self._plan_pose_w(env_ids)
        stance_xy = self.robot_feet_pos_w[env_ids][idx, 1 - swing, :2]
        goal_xy = self._feet_xy_of(plan, self.cfg.nominal_width)[idx, swing]
        d = goal_xy - stance_xy
        c, sn = torch.cos(stance_yaw), torch.sin(stance_yaw)
        x = c * d[:, 0] + sn * d[:, 1]
        y = -sn * d[:, 0] + c * d[:, 1]
        dy = s * y - self.cfg.nominal_width
        err = _wrap(plan[:, 2] - stance_yaw)
        err = self._latched_sign(env_ids, err) * err.abs()
        turning = err.abs() > self.cfg.turn_in_place_yaw
        x = torch.where(turning, torch.zeros_like(x), x)
        dy = torch.where(turning, torch.zeros_like(dy), dy)
        self.metrics["turning"][env_ids] = turning.float()
        return x, dy, s * err

    def _next_step(self, env_ids: torch.Tensor):
        """Alternate feet while walking; out of a stand lead with the foot on
        the turning side for a pure turn, else with the foot farther from its
        planned pose."""
        new_swing = 1 - self.swing[env_ids]
        was_standing = self.standing[env_ids]
        if bool(was_standing.any()):
            sub = env_ids[was_standing]
            plan = self._plan_pose_w(sub)
            pos, yaw = self._pose_err(sub, plan)
            lead = ((pos[:, 1] + 0.3 * yaw[:, 1]) > (pos[:, 0] + 0.3 * yaw[:, 0])).long()
            err = _wrap(plan[:, 2] - self.robot.data.heading_w[sub])
            sign = self._latched_sign(sub, err)
            turning = err.abs() > self.cfg.turn_in_place_yaw
            lead = torch.where(turning, (sign < 0.0).long(), lead)
            new_swing[was_standing] = lead
        stance_side = 1 - new_swing
        idx = torch.arange(len(env_ids), device=self.device)
        self._start_step(
            env_ids, new_swing,
            self.robot_feet_pos_w[env_ids][idx, stance_side],
            self.robot_feet_yaw[env_ids][idx, stance_side],
        )

    def _update_command(self):
        active = ~self.standing
        self.t[active] += 1
        self._goal_age += 1
        self._hold[self.standing] = (self._hold[self.standing] - 1).clamp(min=0)
        boundary = (active & (self.t >= self.step_frames)) | self.standing
        if not bool(boundary.any()):
            return
        ids = boundary.nonzero(as_tuple=False).squeeze(-1)

        if self.cfg.auto_goals:
            timeout = self._goal_age[ids].float() * self._env.step_dt >= self.cfg.goal_timeout_s
            hold_done = self.standing[ids] & (self._hold[ids] <= 0)
            redraw = ids[self._need_goal[ids] | timeout | hold_done]
            if len(redraw) > 0:
                self._sample_goal(redraw)

        self._update_facing(ids)
        has_goal = ~self._need_goal[ids]
        standing = self.standing[ids]
        arrived = self._arrived(ids)
        walk = ids[has_goal & ((~standing & ~arrived) | (standing & self._departed(ids)))]
        if len(walk) > 0:
            self._next_step(walk)
        enter = ids[~standing & arrived]
        if len(enter) > 0:
            self._enter_stand(enter)
            self._hold[enter] = (
                sample_uniform(*self.cfg.stand_duration_range, (len(enter),), self.device)
                / self._env.step_dt).long()
            self.metrics["goals_reached"][enter] += 1.0
            self.metrics["turning"][enter] = 0.0

    def _resample_command(self, env_ids: torch.Tensor):
        super()._resample_command(env_ids)      # standing start, zero twist
        # body poses are stale until the next physics step: draw the goal at
        # the first boundary instead of here
        self._need_goal[env_ids] = True
        self.facing[env_ids] = False
        self._turn_sign[env_ids] = 0.0
        self._hold[env_ids] = 0
        self._goal_age[env_ids] = 0
        self.metrics["goals_reached"][env_ids] = 0.0
        self.metrics["turning"][env_ids] = 0.0

    def _update_metrics(self):
        super()._update_metrics()
        d = self.pose_target_w[:, :2] - self.robot.data.root_link_pos_w[:, :2]
        self.metrics["dist_to_goal"] = d.norm(dim=-1)
        self.metrics["yaw_to_goal"] = _wrap(
            self.pose_target_w[:, 2] - self.robot.data.heading_w).abs()
        self.metrics["facing"] = self.facing.float()

    # ── debug vis: goal frame + goal feet + planned heading ─────────────────

    def _debug_vis_impl(self, visualizer) -> None:
        super()._debug_vis_impl(visualizer)
        env_indices = visualizer.get_env_indices(self.num_envs)
        if not env_indices:
            return
        all_ids = torch.arange(self.num_envs, device=self.device)
        goal = self.pose_target_w.cpu().numpy()
        feet = self._goal_feet_xy(all_ids).cpu().numpy()
        plan_yaw = self._plan_pose_w(all_ids)[:, 2].cpu().numpy()
        mid = self._feet_mid_xy(all_ids).cpu().numpy()
        facing = self.facing.cpu().numpy()
        for b in env_indices:
            yaw = goal[b, 2]
            c, s = np.cos(yaw), np.sin(yaw)
            start = np.array([goal[b, 0], goal[b, 1], 0.03])
            visualizer.add_arrow(
                start=start, end=start + 0.3 * np.array([c, s, 0.0]),
                color=(0.95, 0.75, 0.10, 0.95), width=0.02, label=f"goal_{b}")
            R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
            for i, name in enumerate(("left", "right")):
                visualizer.add_frame(
                    position=np.array([feet[b, i, 0], feet[b, i, 1], 0.0]),
                    rotation_matrix=R, scale=0.06, label=f"goal_{name}_foot_{b}")
            if facing[b]:
                c, s = np.cos(plan_yaw[b]), np.sin(plan_yaw[b])
                start = np.array([mid[b, 0], mid[b, 1], 0.05])
                visualizer.add_arrow(
                    start=start, end=start + 0.25 * np.array([c, s, 0.0]),
                    color=(0.10, 0.80, 0.95, 0.95), width=0.015, label=f"plan_yaw_{b}")


@dataclass(kw_only=True)
class PositionFootstepCommandCfg(VelocityFootstepCommandCfg):
    """PLAY-ONLY: the planner's geometry and reset, goal-planned targets."""

    fixed_twist: tuple[float, float, float] | None = (0.0, 0.0, 0.0)
    """Zero and unused: stepping is the goal's.  Keeps the inherited reset
    standing and the reference twist views honest."""
    twist_resample_s: float = 0.0

    auto_goals: bool = True
    """Draw goals on the term's own (first boundary, stand done, timeout);
    False = stand until ``set_goal``."""
    radius_range: tuple[float, float] = (0.5, 2.0)
    """Goal distance from the pelvis at draw time, m."""
    bearing_range: tuple[float, float] = (-3.14159, 3.14159)
    """Goal bearing relative to the current heading, rad."""
    rel_yaw_range: tuple[float, float] = (-1.5708, 1.5708)
    """Goal yaw relative to the current heading, rad."""
    arena_radius: float = 3.0
    """Goals are clamped inside this radius of the env origin, m."""
    goal_timeout_s: float = 30.0
    """A goal not reached within this time is redrawn, s (``auto_goals`` only)."""
    stand_duration_range: tuple[float, float] = (1.0, 3.0)
    """Stand at the goal for this long before the next draw, s (``auto_goals`` only)."""

    heading_law: bool = True
    """Plan the bearing to the goal as yaw while the feet midpoint is farther
    than ``face_goal_dist`` (walk facing the goal), the goal's own yaw inside;
    False = the goal's yaw from the first step."""
    face_goal_dist: float = 0.6
    """Feet-midpoint distance below which the goal's own yaw is planned, m."""
    face_goal_hysteresis: float = 0.3
    """Extra distance before the bearing is planned again, m."""
    turn_in_place_yaw: float = 1.047
    """|planned yaw − stance yaw| above which the step is a pure turn
    (x = dy = 0), rad."""
    turn_latch_band: float = 0.5
    """Yaw errors within this band of ±π keep the turn direction chosen when
    the turn began, rad."""

    arrive_pos_tol: float = 0.05
    """Per-foot position tolerance to the goal stance, m."""
    arrive_yaw_tol: float = 0.15
    """Per-foot yaw tolerance to the goal yaw, rad."""
    depart_pos_tol: float = 0.10
    """Per-foot position error past which a standing env walks again, m."""
    depart_yaw_tol: float = 0.30
    """Per-foot yaw error past which a standing env walks again, rad."""

    def build(self, env: ManagerBasedRlEnv) -> PositionFootstepCommand:
        return PositionFootstepCommand(self, env)
