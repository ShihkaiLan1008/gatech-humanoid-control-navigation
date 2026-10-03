"""Maze-owning SE(2) twist FSM — the family-agnostic half of the honest
force-eval environment.

THE HONEST-EVAL CONTRACT (user decree 2026-08-11). The final command
interface treats the pelvis twist and the per-hand contact tuple as
INDEPENDENT channels, and the policies trained that way (the press cube RIDES
the base — twist stays freely sampled through every contact window). The eval
env must therefore never couple the two: this term NEVER reads the hand
command terms. There is no press-hold, no stand-at-wall latch when a window
opens, no ``hold_during_press`` — the knobs the mpc_online wall_following FSM
used to park the base for its park-and-press checkpoint are deliberately
absent. FOLLOW keeps issuing lateral hops (sustained wipe twist) while the
hand presses; move-and-press under a live twist is exactly what the eval
exists to measure.

The FSM is the mpc_online wall_following three-state machine (STAND / RANDOM
/ APPROACH→FOLLOW) merged with the locomotion_pos SE(2) goal attractor
(constant-decel speed law + yaw P + slew), re-implemented on mjlab built-ins
+ ``rl.g1_assets`` only, so ANY policy family's eval task can consume it
(decoupled today, whole-body later). ``ranges`` stays pinned to the evaluated
lower policy's TRAINED box; the Se2 attractor remains a play/eval shim —
NEVER train against this term without re-arguing the attractor rule
(CLAUDE.md).

This term also OWNS the maze: the wall map is re-randomized per env reset and
the mocap wall entities are re-posed from it (spawn keep-out replaces
free-space-aware robot resets — the stock reset event stays untouched).
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

from rl.g1_assets.maze_wall import (
    MAZE_WALL_Z,
    maze_wall_entity_name,
    maze_wall_park_offset,
)
from rl.g1_assets.wall_map import WallMap

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.viewer.debug_visualizer import DebugVisualizer

STATE_STAND = 0
STATE_RANDOM = 1
STATE_APPROACH = 2
STATE_FOLLOW = 3

# Goal-frame gizmo height: nominal G1 pelvis height.
_GOAL_VIZ_Z = 0.78


class MazeFollowTwistCommand(UniformVelocityCommand):
    """Maze-owning SE(2) goal FSM → goal-attractor body-frame twist.

    Fills the "twist" slot so the lower policy, the gait clock and the MPC
    all keep reading a body-frame (vx, vy, wz). States:

    * STAND — goal = current pose, held for ``stand_duration_range``.
    * RANDOM — a line-of-sight-checked free-space goal (the attractor is a
      straight-line law with NO obstacle avoidance).
    * APPROACH → FOLLOW — a goal in the wall-edge band (standoff in
      ``standoff_range``, yaw FACING the wall); on arrival the env hops
      laterally along the face, occasionally standing at the wall or leaving.

    Honesty: this term owns the maze and the base itinerary and NOTHING else
    — it never queries the hand terms and keeps commanding through every
    contact window (see module docstring).
    """

    cfg: "MazeFollowTwistCommandCfg"

    def __init__(self, cfg: "MazeFollowTwistCommandCfg", env: "ManagerBasedRlEnv") -> None:
        super().__init__(cfg, env)
        B = self.num_envs
        dev = self.device
        self.wall_map = WallMap(B, env.scene.env_origins[:, :2].clone(), dev)
        self._walls = [
            env.scene[maze_wall_entity_name(i)] for i in range(self.wall_map.N)
        ]
        self._park_w = torch.zeros(self.wall_map.N, B, 3, device=dev)
        for i in range(self.wall_map.N):
            self._park_w[i] = env.scene.env_origins + torch.tensor(
                maze_wall_park_offset(i), device=dev
            )
        self._all_ids = torch.arange(B, device=dev)
        # lin_vel_min bookkeeping: how often the deadband floor actually bites
        self.floor_up_ticks = 0
        self.floor_zero_ticks = 0
        self.lin_cmd_ticks = 0

        self.pose_target_w = torch.zeros(B, 3, device=dev)  # world x, y, yaw
        self.arrived = torch.zeros(B, dtype=torch.bool, device=dev)
        self.speed_scale = torch.ones(B, device=dev)
        self.fsm_state = torch.full((B,), STATE_RANDOM, dtype=torch.long, device=dev)
        self.follow_wall = torch.zeros(B, dtype=torch.long, device=dev)
        self.follow_side = torch.ones(B, device=dev)
        self.follow_dir = torch.ones(B, device=dev)

        self.metrics["dist_to_goal"] = torch.zeros(B, device=dev)
        self.metrics["arrived_frac"] = torch.zeros(B, device=dev)
        self.metrics["follow_frac"] = torch.zeros(B, device=dev)
        self.metrics["stand_frac"] = torch.zeros(B, device=dev)
        self.metrics["wall_standoff"] = torch.zeros(B, device=dev)

    # ── maze ────────────────────────────────────────────────────────────────

    def _write_wall_poses(self, env_ids: Tensor) -> None:
        m = self.wall_map
        for i, ent in enumerate(self._walls):
            pose = torch.zeros(len(env_ids), 7, device=self.device)
            # Map convention: th = direction of the wall's LONG (tangent)
            # axis. The geom's long axis is local Y and its THICKNESS is
            # local X, so the entity yaw must be th + 90° to put the length
            # along t̂ and the thickness along n̂ (writing yaw = th rotates
            # every wall 90° — the 2026-07-29 blade-through-the-robot bug).
            psi = m.th[env_ids, i] + torch.pi / 2
            act = m.active[env_ids, i]
            pos_active = torch.cat(
                [
                    m.c[env_ids, i],
                    torch.full((len(env_ids), 1), MAZE_WALL_Z, device=self.device),
                ],
                dim=-1,
            )
            pose[:, :3] = torch.where(
                act.unsqueeze(-1), pos_active, self._park_w[i, env_ids]
            )
            pose[:, 3] = torch.where(act, (psi / 2).cos(), torch.ones_like(psi))
            pose[:, 6] = torch.where(act, (psi / 2).sin(), torch.zeros_like(psi))
            ent.write_mocap_pose_to_sim(pose, env_ids=env_ids)

    def reset(self, env_ids) -> dict[str, float]:
        ids = self._all_ids[env_ids] if not isinstance(env_ids, Tensor) else env_ids
        cfg = self.cfg
        self.wall_map.randomize(
            ids, cfg.arena_radius, cfg.wall_gap_min, cfg.spawn_clear_radius
        )
        self._write_wall_poses(ids)
        self.fsm_state[ids] = STATE_RANDOM
        return super().reset(env_ids)

    # ── FSM helpers ─────────────────────────────────────────────────────────

    def face_standoff(self) -> tuple[Tensor, Tensor, Tensor]:
        """(d, yaw_err, t_margin) of the base vs the FOLLOWED wall face, all
        (B,): standoff distance to the face PLANE, |heading − face-normal
        heading|, and the tangential margin ``hl − |t_par|`` (NEGATIVE =
        beyond the segment end — the plane there is PHANTOM). Only meaningful
        where fsm_state == FOLLOW (junk elsewhere)."""
        p = self.robot.data.root_link_pos_w[:, :2]
        c, t_hat, n_s, hl = self.wall_map.face_frame(
            self._all_ids, self.follow_wall, self.follow_side
        )
        rel = p - c
        d = (rel * n_s).sum(-1) - self.wall_map.thickness / 2
        t_margin = hl - (rel * t_hat).sum(-1).abs()
        face_yaw = torch.atan2(-n_s[:, 1], -n_s[:, 0])
        yaw_err = wrap_to_pi(face_yaw - self.robot.data.heading_w).abs()
        return d, yaw_err, t_margin

    def _stand_at(self, ids: Tensor) -> None:
        cfg = self.cfg
        p = self.robot.data.root_link_pos_w[ids, :2]
        self.pose_target_w[ids, :2] = p
        self.pose_target_w[ids, 2] = self.robot.data.heading_w[ids]
        # arrived=True from the start: time_left (the stand duration) governs.
        self.arrived[ids] = True
        self.time_left[ids] = torch.empty(len(ids), device=self.device).uniform_(
            *cfg.stand_duration_range
        )

    def _sample_random_goals(self, ids: Tensor) -> Tensor:
        """LOS + clearance-checked free-space goals; envs with no valid
        candidate fall back to STAND. Returns the ids that got a goal."""
        cfg = self.cfg
        E = len(ids)
        dev = self.device
        K = 12
        p = self.robot.data.root_link_pos_w[ids, :2]
        heading = self.robot.data.heading_w[ids]

        radius = torch.empty(E, K, device=dev).uniform_(*cfg.radius_range)
        bearing = heading.unsqueeze(1) + torch.empty(E, K, device=dev).uniform_(
            *cfg.bearing_range
        )
        goal = p.unsqueeze(1) + radius.unsqueeze(-1) * torch.stack(
            [bearing.cos(), bearing.sin()], dim=-1
        )
        ids_rep = ids.repeat_interleave(K)
        flat = goal.reshape(-1, 2)
        ok = self.wall_map.clearance(flat, ids_rep) >= cfg.goal_clearance
        ok &= (flat - self.wall_map.origins[ids_rep]).norm(dim=-1) <= cfg.arena_radius
        ok &= ~self.wall_map.seg_blocked(
            p.repeat_interleave(K, 0), flat, ids_rep, cfg.los_margin
        )
        ok = ok.reshape(E, K)

        cand = torch.arange(K, device=dev).expand(E, K)
        pick = (
            torch.where(ok, cand, torch.full_like(cand, K))
            .min(dim=1)
            .values.clamp(max=K - 1)
        )
        any_ok = ok.any(dim=1)
        got = ids[any_ok]
        ar = torch.arange(E, device=dev)[any_ok]
        self.pose_target_w[got, :2] = goal[ar, pick[any_ok]]
        rel_yaw = torch.empty(len(got), device=dev).uniform_(*cfg.rel_yaw_range)
        self.pose_target_w[got, 2] = wrap_to_pi(heading[any_ok] + rel_yaw)
        self.arrived[got] = False
        self._stand_at(ids[~any_ok])
        return got

    def _sample_wall_goals(self, ids: Tensor) -> Tensor:
        """Wall-edge goals (standoff band, facing the wall, LOS-valid).
        Returns the ids that found one; the rest keep needing a fallback."""
        cfg = self.cfg
        m = self.wall_map
        E = len(ids)
        dev = self.device
        K = 12
        p = self.robot.data.root_link_pos_w[ids, :2]

        wall = torch.randint(0, m.N, (E, K), device=dev)
        u = torch.rand(E, K, device=dev) * 2.0 - 1.0
        t_par = u * (m.hl[wall] - cfg.wall_end_margin)
        standoff = torch.empty(E, K, device=dev).uniform_(*cfg.standoff_range)

        ids_rep = ids.repeat_interleave(K)
        wall_f = wall.reshape(-1)
        ar = torch.arange(E * K, device=dev)
        c = m.c[ids_rep, wall_f]
        t_hat = m.t_hat(ids_rep)[ar, wall_f]
        n_hat = m.n_hat(ids_rep)[ar, wall_f]
        active = m.active[ids_rep, wall_f]
        # Robot-side face.
        side = torch.where(
            ((p.repeat_interleave(K, 0) - c) * n_hat).sum(-1) >= 0, 1.0, -1.0
        )
        n_s = n_hat * side.unsqueeze(-1)
        goal = (
            c
            + t_par.reshape(-1, 1) * t_hat
            + (m.thickness / 2 + standoff.reshape(-1)).unsqueeze(-1) * n_s
        )
        ok = active
        ok &= ~m.seg_blocked(p.repeat_interleave(K, 0), goal, ids_rep, cfg.los_margin)
        ok &= (goal - m.origins[ids_rep]).norm(dim=-1) <= cfg.arena_radius
        ok = ok.reshape(E, K)

        # Among valid candidates take the NEAREST goal — short approach legs
        # actually get walked to completion; far ones time out.
        goal_dist = (goal.reshape(E, K, 2) - p.unsqueeze(1)).norm(dim=-1)
        goal_dist = torch.where(ok, goal_dist, torch.full_like(goal_dist, 1e6))
        pick = goal_dist.argmin(dim=1)
        any_ok = ok.any(dim=1)
        got = ids[any_ok]
        arE = torch.arange(E, device=dev)[any_ok]
        flatpick = arE * K + pick[any_ok]
        self.pose_target_w[got, :2] = goal.reshape(E * K, 2)[flatpick]
        ns_pick = n_s.reshape(E * K, 2)[flatpick]
        self.pose_target_w[got, 2] = torch.atan2(-ns_pick[:, 1], -ns_pick[:, 0])
        self.arrived[got] = False
        self.follow_wall[got] = wall.reshape(-1)[flatpick]
        self.follow_side[got] = side.reshape(E, K)[arE, pick[any_ok]]
        self.follow_dir[got] = torch.where(
            torch.rand(len(got), device=dev) < 0.5, 1.0, -1.0
        )
        return got

    def _sample_follow_hop(self, ids: Tensor) -> None:
        """Next lateral hop along the followed face (direction persistent,
        flipped at segment ends or with flip_prob; standoff re-drawn in-band)."""
        cfg = self.cfg
        m = self.wall_map
        E = len(ids)
        dev = self.device
        p = self.robot.data.root_link_pos_w[ids, :2]
        c, t_hat, n_s, hl = m.face_frame(ids, self.follow_wall[ids], self.follow_side[ids])

        flip = torch.rand(E, device=dev) < cfg.follow_flip_prob
        self.follow_dir[ids] = torch.where(
            flip, -self.follow_dir[ids], self.follow_dir[ids]
        )
        t_now = ((p - c) * t_hat).sum(-1)
        ds = torch.empty(E, device=dev).uniform_(*cfg.follow_hop_range)
        t_max = hl - cfg.wall_end_margin
        t_goal = t_now + self.follow_dir[ids] * ds
        # At a segment end, reverse and hop the other way.
        hit_end = t_goal.abs() > t_max
        self.follow_dir[ids] = torch.where(
            hit_end, -self.follow_dir[ids], self.follow_dir[ids]
        )
        t_goal = torch.where(hit_end, t_now + self.follow_dir[ids] * ds, t_goal)
        t_goal = t_goal.clamp(-t_max, t_max)

        standoff = torch.empty(E, device=dev).uniform_(*cfg.standoff_range)
        goal = c + t_goal.unsqueeze(-1) * t_hat + (
            m.thickness / 2 + standoff
        ).unsqueeze(-1) * n_s
        self.pose_target_w[ids, :2] = goal
        self.pose_target_w[ids, 2] = torch.atan2(-n_s[:, 1], -n_s[:, 0])
        self.arrived[ids] = False

    # ── FSM resample ────────────────────────────────────────────────────────

    def _resample_command(self, env_ids: Tensor) -> None:
        cfg = self.cfg
        dev = self.device

        # Pure goal FSM — no twist/standing training modes.
        self.is_standing_env[env_ids] = False
        self.speed_scale[env_ids] = 1.0
        # Seed the slew filter from the measured twist (no command step at
        # resample), clamped into the trained box.
        vb = self.robot.data.root_link_lin_vel_b[env_ids, 0:2]
        self.vel_command_b[env_ids, 0] = vb[:, 0].clamp(*cfg.ranges.lin_vel_x)
        self.vel_command_b[env_ids, 1] = vb[:, 1].clamp(*cfg.ranges.lin_vel_y)
        self.vel_command_b[env_ids, 2] = self.robot.data.root_link_ang_vel_b[
            env_ids, 2
        ].clamp(*cfg.ranges.ang_vel_z)

        st = self.fsm_state[env_ids]
        arrived_prev = self.arrived[env_ids]

        # FOLLOW continuation draws — NO press coupling (honest-eval decree):
        # whether a hand window is open never enters this draw.
        follow_ids = env_ids[st == STATE_FOLLOW]
        stay_follow = torch.zeros(0, dtype=torch.long, device=dev)
        wall_stand = torch.zeros(0, dtype=torch.long, device=dev)
        if len(follow_ids):
            u = torch.rand(len(follow_ids), device=dev)
            stay_follow = follow_ids[u < cfg.follow_p_continue]
            wall_stand = follow_ids[
                (u >= cfg.follow_p_continue)
                & (u < cfg.follow_p_continue + cfg.follow_p_stand)
            ]
        leave_ids = follow_ids[
            ~torch.isin(follow_ids, torch.cat([stay_follow, wall_stand]))
        ]

        # APPROACH: arrived → FOLLOW hop; timed out → top-level redraw.
        appr = env_ids[(st == STATE_APPROACH)]
        appr_arrived = appr[arrived_prev[(st == STATE_APPROACH)]]
        appr_failed = appr[~arrived_prev[(st == STATE_APPROACH)]]

        stay_follow = torch.cat([stay_follow, appr_arrived])
        if len(stay_follow):
            self.fsm_state[stay_follow] = STATE_FOLLOW
            self._sample_follow_hop(stay_follow)
            self.speed_scale[stay_follow] = cfg.follow_speed_scale
        if len(wall_stand):
            self.fsm_state[wall_stand] = STATE_FOLLOW  # stand AT the wall
            self._stand_at(wall_stand)

        # Top-level draw for everyone else.
        top = torch.cat(
            [env_ids[(st == STATE_STAND) | (st == STATE_RANDOM)], appr_failed, leave_ids]
        )
        if len(top):
            u = torch.rand(len(top), device=dev)
            stand_ids = top[u < cfg.p_stand]
            appr_ids = top[u >= 1.0 - cfg.p_approach]
            rand_ids = top[(u >= cfg.p_stand) & (u < 1.0 - cfg.p_approach)]

            if len(stand_ids):
                self.fsm_state[stand_ids] = STATE_STAND
                self._stand_at(stand_ids)
            if len(appr_ids):
                got = self._sample_wall_goals(appr_ids)
                self.fsm_state[got] = STATE_APPROACH
                # Arrive gently: overshooting a wall goal is a step INTO the
                # wall (obstacle-blind policy trips).
                self.speed_scale[got] = cfg.approach_speed_scale
                rand_ids = torch.cat(
                    [rand_ids, appr_ids[~torch.isin(appr_ids, got)]]
                )
            if len(rand_ids):
                got = self._sample_random_goals(rand_ids)
                self.fsm_state[got] = STATE_RANDOM
                self.fsm_state[rand_ids[~torch.isin(rand_ids, got)]] = STATE_STAND

    # ── per-step attractor ──────────────────────────────────────────────────

    def _update_command(self) -> None:
        cfg = self.cfg
        pos = self.robot.data.root_link_pos_w[:, :2]
        heading = self.robot.data.heading_w

        e_w = self.pose_target_w[:, :2] - pos
        dist = torch.linalg.norm(e_w, dim=-1)
        e_yaw = wrap_to_pi(self.pose_target_w[:, 2] - heading)

        # Arrival latch with hysteresis + dwell-then-early-resample.
        in_tol = (dist < cfg.pos_tol) & (e_yaw.abs() < cfg.yaw_tol)
        newly_arrived = in_tol & ~self.arrived
        self.arrived |= in_tol
        self.arrived &= dist < cfg.arrive_hysteresis * cfg.pos_tol
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

        # Direction-preserving clamp into the per-env SCALED trained box.
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

        stop = self.arrived
        zeros = torch.zeros_like(vx)
        vx = torch.where(stop, zeros, vx)
        vy = torch.where(stop, zeros, vy)
        wz = torch.where(stop, zeros, wz)

        # Slew-limit the linear command (the attractor re-plans every step;
        # without a rate cap it can flip sign across the goal).
        dv = cfg.slew * self._env.step_dt
        self.vel_command_b[:, 0] += (vx - self.vel_command_b[:, 0]).clamp(-dv, dv)
        self.vel_command_b[:, 1] += (vy - self.vel_command_b[:, 1]).clamp(-dv, dv)
        self.vel_command_b[:, 2] = wz

        # Deadband floor (sim2real): the EMITTED linear command must never
        # lie in (0, lin_vel_min) -- a real base does not move for it.  While
        # the goal still pulls, a small command is raised to the floor
        # (direction kept); once arrived, the slewed residue is snapped to
        # zero instead of being kept alive at the floor.
        if getattr(cfg, "lin_vel_min", 0.0) > 0.0:
            lin = self.vel_command_b[:, :2]
            speed = torch.linalg.norm(lin, dim=-1)
            live = speed > 1e-6
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
        self.metrics["dist_to_goal"] = torch.linalg.norm(
            self.pose_target_w[:, :2] - pos, dim=-1
        )
        self.metrics["arrived_frac"] = self.arrived.float()
        follow = self.fsm_state == STATE_FOLLOW
        self.metrics["follow_frac"] = follow.float()
        self.metrics["stand_frac"] = (self.fsm_state == STATE_STAND).float()
        d, _, _ = self.face_standoff()
        self.metrics["wall_standoff"] = torch.where(follow, d, torch.zeros_like(d))

    def _debug_vis_impl(self, visualizer: "DebugVisualizer") -> None:
        super()._debug_vis_impl(visualizer)  # command/actual velocity arrows
        env_indices = visualizer.get_env_indices(self.num_envs)
        if not env_indices:
            return
        tgt = self.pose_target_w.cpu().numpy()
        arrived = self.arrived.cpu().numpy()
        for b in env_indices:
            x, y, psi = float(tgt[b, 0]), float(tgt[b, 1]), float(tgt[b, 2])
            c, s = math.cos(psi), math.sin(psi)
            rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
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
class MazeFollowTwistCommandCfg(UniformVelocityCommandCfg):
    """Config for :class:`MazeFollowTwistCommand`.

    ``ranges`` (inherited) are the attractor's velocity clip box — pin them to
    the box the evaluated lower policy TRAINED under, never a widened one.
    """

    # Maze layout.
    arena_radius: float = 5.0
    wall_gap_min: float = 1.8
    spawn_clear_radius: float = 1.3
    """Wall keep-out around the env origin: covers the stock reset event's
    ±0.5 m xy scatter plus robot clearance."""

    # Attractor.
    radius_range: tuple[float, float] = (1.0, 2.5)
    bearing_range: tuple[float, float] = (-math.pi, math.pi)
    rel_yaw_range: tuple[float, float] = (-math.pi, math.pi)
    decel: float = 0.5
    slew: float = 2.0
    yaw_stiffness: float = 1.5
    lin_vel_min: float = 0.0
    """Floor on the commanded linear SPEED [m/s]; 0 disables.  A real base
    cannot track a creeping command, so the band (0, lin_vel_min) is
    forbidden: a nonzero command below the floor is raised to it (direction
    kept) while the goal still pulls, and snapped to zero once arrived."""
    pos_tol: float = 0.12
    yaw_tol: float = 0.25
    arrive_hysteresis: float = 2.0
    arrive_dwell_range: tuple[float, float] = (1.0, 2.0)

    # State probabilities (top-level draw / FOLLOW continuation draw).
    p_stand: float = 0.10
    p_approach: float = 0.35
    follow_p_continue: float = 0.70
    follow_p_stand: float = 0.10

    # Wall-edge geometry.
    standoff_range: tuple[float, float] = (0.36, 0.44)
    """Base-to-face band: the command frame IS the tactile tip (kiss-x = d),
    so the band centers the trained press-x box (0.33, 0.45)."""
    wall_end_margin: float = 0.45
    follow_hop_range: tuple[float, float] = (0.5, 1.5)
    follow_flip_prob: float = 0.15
    stand_duration_range: tuple[float, float] = (2.0, 4.0)
    approach_speed_scale: float = 0.6
    follow_speed_scale: float = 0.7

    # Sampling validity.
    goal_clearance: float = 0.5
    los_margin: float = 0.22
    """Wall-surface inflation for line-of-sight checks [m] — must stay below
    standoff_range[0] so a wall-edge goal is not blocked by its own wall."""

    def build(self, env: "ManagerBasedRlEnv") -> MazeFollowTwistCommand:
        return MazeFollowTwistCommand(self, env)
