"""wall_following commands: maze-aware SE(2) FSM + on-wall hand commands.

``WallFollowSe2Command`` (the "twist" slot) runs the user's three-state
machine on top of the goal attractor:

* STAND — goal = current pose, held for ``stand_duration_range``.
* RANDOM — a line-of-sight-checked free-space goal (the attractor is a
  straight-line P law with NO obstacle avoidance, so every goal must be
  LOS-valid and clear of walls; the generous corridors make rejection
  sampling cheap).
* APPROACH → FOLLOW — a goal in the "wall edge" band (standoff ∈
  ``standoff_range`` from a wall face, yaw FACING the wall); on arrival the
  env enters FOLLOW and hops laterally along the face (side-step wipe
  geometry — the trained press direction is command +x ≈ base-forward, so
  contact requires facing the wall), occasionally standing at the wall or
  leaving back to the free states.

It also OWNS the maze: the wall map is re-randomized per env reset and the
mocap wall entities are re-posed from it (spawn keep-out replaces
free-space-aware robot resets — the stock reset event stays untouched).

``WallEePoseCommand`` (per hand) is the stage-2 hand command against REAL
walls instead of riding cubes (``cube_entity_name=None``):

* contact windows exist ONLY in FOLLOW and open OPPORTUNISTICALLY: the base
  must be measured in the press-ready envelope (standoff band, facing,
  in-segment, slow, upright) — "no wall, no contact" (user semantics); on
  open, ``press_hold`` parks the base on the spot (the current checkpoint
  presses only from a parked base) and ``hold_during_press`` keeps it parked
  for the window's life. Masked draws mix with free reaches
  (``wall_contact_prob``).
* a masked target's tangential/vertical components ride the base exactly as
  trained (on a PLANE, a base-riding constant command + lateral base motion
  IS the wipe); the wall-normal component is solved to KISS the face at open
  and then HELD, floored at the live kiss value and capped at
  ``press_max_overlap`` past it (see ``_apply_masked_x``).
* free targets are CLIPPED in front of wall faces (an unmasked wrist jammed
  into a wall is penalized ghost force).
* windows ramp out (never chop) when the press stops being viable: FSM left
  the wall, base falling, wall out of reach, base past the segment end
  (phantom plane) or at/behind the face.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from mjlab.utils.lab_api.math import (
    quat_apply_inverse,
    quat_from_euler_xyz,
    quat_unique,
    wrap_to_pi,
)
from torch import Tensor

from rl.g1_assets.maze_wall import (
    MAZE_WALL_Z,
    maze_wall_entity_name,
    maze_wall_park_offset,
)
from rl.g1_assets.press_cube import FACE_OFFSET

from ...hand_force.mdp.commands import (
    HandForceEePoseCommand,
    HandForceEePoseCommandCfg,
)
from ...locomotion_pos.mdp.commands import (
    Se2PoseVelocityCommand,
    Se2PoseVelocityCommandCfg,
)
from .wall_map import _INF, WallMap, _pt_seg_dist

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

STATE_STAND = 0
STATE_RANDOM = 1
STATE_APPROACH = 2
STATE_FOLLOW = 3


class WallFollowSe2Command(Se2PoseVelocityCommand):
    """Maze-aware SE(2) goal FSM (owns the wall map; attractor inherited)."""

    cfg: WallFollowSe2CommandCfg

    def __init__(self, cfg: WallFollowSe2CommandCfg, env: ManagerBasedRlEnv) -> None:
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

        self.fsm_state = torch.full((B,), STATE_RANDOM, dtype=torch.long, device=dev)
        self.follow_wall = torch.zeros(B, dtype=torch.long, device=dev)
        self.follow_side = torch.ones(B, device=dev)
        self.follow_dir = torch.ones(B, device=dev)

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
            # along t̂ and the thickness along n̂. Writing yaw = th (the
            # 2026-07-29 bug) rotates every wall 90°: a 10 cm blade running
            # ALONG the intended normal — straight through the robot at the
            # commanded standoff (raw-contact dump: pelvis/torso/head
            # penetrating 3–6 cm while the map said d = 0.30).
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
        beyond the segment end — the plane there is PHANTOM: no physical
        wall. The 2026-07-29 stagger trace: a window held open off-end had
        the policy pressing air at d −1.5…+0.4 for seconds). Only meaningful
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
        # arrived=True from the start: the dwell-early-resample path never
        # fires, so time_left (the stand duration) governs.
        self.arrived[ids] = True
        self.time_left[ids] = torch.empty(len(ids), device=self.device).uniform_(
            *cfg.stand_duration_range
        )

    def press_hold(self, ids: Tensor) -> None:
        """Called by the hand command term the instant a window OPENS: park
        the base right where it is (goal = current pose, immediate stand).
        Press initiation must not depend on the attractor's arrival latch —
        with a vel-degraded policy hops mostly time out and arrival-parking
        lands outside the reach band; instead windows open opportunistically
        whenever the base PASSES through the press-ready envelope, and this
        latch freezes it there. ``hold_during_press`` then keeps it parked
        across resamples for the window's whole life."""
        if len(ids) == 0:
            return
        self._stand_at(ids)

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

        # Among valid candidates take the NEAREST goal — approach legs across
        # the whole maze time out with a slow policy; short legs enter FOLLOW.
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
        n = len(env_ids)
        dev = self.device

        # Pure goal FSM: no twist envs, no standing-env mode (stands are a
        # state with the goal at the current pose).
        self.is_twist_env[env_ids] = False
        self.is_standing_env[env_ids] = False
        self.speed_scale[env_ids] = torch.empty(n, device=dev).uniform_(
            *cfg.speed_scale_range
        )
        vb = self.robot.data.root_link_lin_vel_b[env_ids, 0:2]
        self.vel_command_b[env_ids, 0] = vb[:, 0].clamp(*cfg.ranges.lin_vel_x)
        self.vel_command_b[env_ids, 1] = vb[:, 1].clamp(*cfg.ranges.lin_vel_y)
        self.vel_command_b[env_ids, 2] = self.robot.data.root_link_ang_vel_b[
            env_ids, 2
        ].clamp(*cfg.ranges.ang_vel_z)

        st = self.fsm_state[env_ids]
        arrived_prev = self.arrived[env_ids]

        # FOLLOW continuation draws.
        follow_ids = env_ids[st == STATE_FOLLOW]
        stay_follow = torch.zeros(0, dtype=torch.long, device=dev)
        wall_stand = torch.zeros(0, dtype=torch.long, device=dev)
        if len(follow_ids):
            pressing = torch.zeros(len(follow_ids), dtype=torch.bool, device=dev)
            if cfg.hold_during_press:
                # An open hand window forces the stand-at-wall draw: the
                # current checkpoint can only press from a parked base (its
                # cube-land behavior — the actor has NO tactile input and
                # leans on faith; a hop mid-window swings the standoff out of
                # reach and it topples through the phantom support). Future
                # move-and-press policies set this False.
                for name in cfg.ee_command_names:
                    term = self._env.command_manager.get_term(name)
                    pressing |= term.contact_mask[follow_ids] > 0.5
            rest = follow_ids[~pressing]
            u = torch.rand(len(rest), device=dev)
            stay_follow = rest[u < cfg.follow_p_continue]
            wall_stand = torch.cat(
                [
                    follow_ids[pressing],
                    rest[
                        (u >= cfg.follow_p_continue)
                        & (u < cfg.follow_p_continue + cfg.follow_p_stand)
                    ],
                ]
            )
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
                # Arrive gently: overshooting an approach goal means walking
                # INTO the wall — trip risk with an obstacle-blind policy.
                self.speed_scale[got] = cfg.approach_speed_scale
                rand_ids = torch.cat(
                    [rand_ids, appr_ids[~torch.isin(appr_ids, got)]]
                )
            if len(rand_ids):
                got = self._sample_random_goals(rand_ids)
                self.fsm_state[got] = STATE_RANDOM
                self.fsm_state[rand_ids[~torch.isin(rand_ids, got)]] = STATE_STAND

    def _update_metrics(self) -> None:
        super()._update_metrics()
        follow = self.fsm_state == STATE_FOLLOW
        self.metrics["follow_frac"] = follow.float()
        self.metrics["stand_frac"] = (self.fsm_state == STATE_STAND).float()
        d, _, _ = self.face_standoff()
        self.metrics["wall_standoff"] = torch.where(
            follow, d, torch.zeros_like(d)
        )


@dataclass(kw_only=True)
class WallFollowSe2CommandCfg(Se2PoseVelocityCommandCfg):
    """Config for :class:`WallFollowSe2Command`."""

    # Maze layout.
    arena_radius: float = 5.0
    """Walls (and goals) confined to this disc around the env origin [m]."""
    wall_gap_min: float = 1.8
    """Minimum surface-to-surface corridor between wall segments [m]."""
    spawn_clear_radius: float = 1.3
    """Wall keep-out around the env origin: covers the stock reset event's
    ±0.5 m xy scatter plus robot clearance — the robot always spawns in free
    space without a map-aware reset."""

    # State probabilities (top-level draw / FOLLOW continuation draw).
    p_stand: float = 0.10
    p_approach: float = 0.35
    """Top-level draw: stand / approach-a-wall; the rest is RANDOM."""
    follow_p_continue: float = 0.70
    follow_p_stand: float = 0.10
    """FOLLOW draw: next lateral hop / stand at the wall; the rest LEAVES."""

    # Wall-edge geometry.
    standoff_range: tuple[float, float] = (0.36, 0.44)
    """Base-to-face distance band ("wall edge"). The command frame IS the
    tactile tip (kiss-x = d exactly): reachable faces sit at the contact
    box x (0.33, 0.45); the band centers it and keeps the toes clear."""
    wall_end_margin: float = 0.45
    """Goals stay this far from segment ends (room for the ±0.3 m lateral
    hand box)."""
    follow_hop_range: tuple[float, float] = (0.5, 1.5)
    follow_flip_prob: float = 0.15
    stand_duration_range: tuple[float, float] = (2.0, 4.0)
    approach_speed_scale: float = 0.6
    """Linear-clip scale on approach legs — arrive gently: overshoot at a
    wall goal is a step INTO the wall (obstacle-blind policy trips)."""
    follow_speed_scale: float = 0.7
    """Linear-clip scale on lateral follow hops."""
    hold_during_press: bool = True
    """FOLLOW resamples draw stand-at-wall while a hand window is open (hop
    and press are mutually exclusive). Required by the current checkpoint:
    the actor has no tactile obs and leans on the mask alone — a hop swings
    the standoff out of reach mid-press and it topples through the phantom
    support. Set False for future move-and-press policies (the wipe)."""
    ee_command_names: tuple[str, str] = ("left_ee_pose", "right_ee_pose")

    # Sampling validity.
    goal_clearance: float = 0.5
    """Free-space clearance of RANDOM goals from every wall surface [m]."""
    los_margin: float = 0.22
    """Wall-surface inflation for line-of-sight checks [m] — must stay below
    standoff_range[0] so a wall-edge goal is not blocked by its own wall."""

    def build(self, env: ManagerBasedRlEnv) -> WallFollowSe2Command:
        return WallFollowSe2Command(self, env)


class WallEePoseCommand(HandForceEePoseCommand):
    """Stage-2 hand command against the maze walls (no cube)."""

    cfg: WallEePoseCommandCfg

    def __init__(self, cfg: WallEePoseCommandCfg, env: ManagerBasedRlEnv) -> None:
        assert cfg.cube_entity_name is None, "wall_following has no press cubes"
        super().__init__(cfg, env)
        self._fsm: WallFollowSe2Command | None = None  # lazy (build order)
        # Wall-normal command component HELD from window open (the trained
        # piecewise-constant semantics — see _apply_masked_x).
        self._x_hold = torch.zeros(self.num_envs, device=self.device)

    @property
    def fsm(self) -> WallFollowSe2Command:
        if self._fsm is None:
            self._fsm = self._env.command_manager.get_term(
                self.cfg.twist_command_name
            )
        return self._fsm

    # ── wall geometry helpers ───────────────────────────────────────────────

    def _followed_face(self) -> tuple[Tensor, Tensor, Tensor]:
        """(c (B,2), n̂_s (B,2), valid) of each env's FOLLOWED wall face."""
        fsm = self.fsm
        c, _t, n_s, _hl = fsm.wall_map.face_frame(
            fsm._all_ids, fsm.follow_wall, fsm.follow_side
        )
        return c, n_s, fsm.fsm_state == STATE_FOLLOW

    def _kiss_x(self, ids: Tensor) -> Tensor:
        """The commanded-x value (yaw frame) at which the commanded wrist
        origin sits FACE_OFFSET in front of the followed face — capsule tip
        exactly ON the face — solved from the MEASURED base pose."""
        fsm = self.fsm
        c, n_s, _ = self._followed_face()
        c, n_s = c[ids], n_s[ids]
        base = self.robot.data.root_link_pos_w[ids, :2]
        h = self.robot.data.heading_w[ids]
        u = torch.stack([h.cos(), h.sin()], dim=-1)  # yaw-frame x̂
        v = torch.stack([-h.sin(), h.cos()], dim=-1)  # yaw-frame ŷ
        y_b = self.pose_command_b[ids, 1]
        T2 = fsm.wall_map.thickness / 2
        denom = (u * n_s).sum(-1)  # ≈ −1 facing the wall
        rhs = (
            T2
            + FACE_OFFSET
            - ((base - c) * n_s).sum(-1)
            - y_b * (v * n_s).sum(-1)
        )
        return rhs / denom.clamp(max=-0.5)

    def _apply_masked_x(self, ids: Tensor, open_window: bool) -> None:
        """Wall-normal command component of masked envs.

        The trained (riding-cube) semantics is PIECEWISE-CONSTANT x: the
        policy presses by driving the arm past the commanded pose, and
        leaning the base toward the surface must LOAD the hand. An
        always-kiss live solve retracts the command exactly as fast as the
        base advances — the press cancels, the expected support never comes,
        and the policy topples forward through the phantom surface (measured
        2026-07-29: the charge-and-crash trace, vx 2.0 into the wall with
        0 N on the wrist). So: HOLD x from window open; use the live kiss
        value only as a FLOOR (extend when the wall recedes — stay in
        contact) and cap at ``press_max_overlap`` past it (arm-PD spring
        with force saturation — bounded jam when the base parks deep)."""
        if len(ids) == 0:
            return
        k = self._kiss_x(ids)
        if open_window:
            self._x_hold[ids] = k
        x = torch.minimum(
            torch.maximum(self._x_hold[ids], k), k + self.cfg.press_max_overlap
        )
        self.pose_command_b[ids, 0] = x.clamp(*self.cfg.cmd_x_clamp)

    def _clip_free_targets(self, env_ids: Tensor) -> None:
        """Push free (unmasked) targets out of wall faces: the commanded
        point must keep FACE_OFFSET + free_face_clearance of air in front of
        every wall surface."""
        free = env_ids[(self.contact_mask[env_ids] < 0.5)]
        if len(free) == 0:
            return
        fsm = self.fsm
        pos_w, _ = self.target_pose_w()
        p = pos_w[free, :2]
        need = (
            fsm.wall_map.thickness / 2
            + FACE_OFFSET
            + self.cfg.free_face_clearance
        )
        s0, s1 = fsm.wall_map.endpoints(free)
        d = _pt_seg_dist(p.unsqueeze(1), s0, s1)
        d = torch.where(fsm.wall_map.active[free], d, torch.full_like(d, _INF))
        dmin, widx = d.min(dim=1)
        bad = dmin < need
        if not bad.any():
            return
        ids = free[bad]
        w = widx[bad]
        ar = torch.arange(len(free), device=self.device)[bad]
        # Push along the offending wall's robot-side normal to the needed
        # clearance, then re-express in the sample frame.
        n = fsm.wall_map.n_hat(ids)[torch.arange(len(ids), device=self.device), w]
        c = fsm.wall_map.c[ids, w]
        dp = ((p[ar] - c) * n).sum(-1)
        side = torch.where(dp >= 0, 1.0, -1.0)
        # Push by (need − |dp|) along the TARGET-side normal n̂·side.
        p_new = p[ar] + (need - dp.abs()).unsqueeze(-1) * (n * side.unsqueeze(-1))
        new_w = pos_w[ids]
        new_w[:, :2] = p_new
        root_pos = self.robot.data.root_link_pos_w[ids]
        aq = self.anchor_quat()[ids]
        self.pose_command_b[ids, :3] = quat_apply_inverse(aq, new_w - root_pos)

    # ── sampling ────────────────────────────────────────────────────────────

    def _resample_command(self, env_ids: Tensor) -> None:
        # Free-space box sample for everyone (grandparent), then the
        # wall-aware contact logic replaces HandForceEePoseCommand's
        # cube-Bernoulli block.
        super(HandForceEePoseCommand, self)._resample_command(env_ids)

        self.contact_mask[env_ids] = 0.0
        self.force_mag[env_ids] = 0.0
        self._contact_age[env_ids] = 0.0
        self._force_cmd_mag[env_ids] = 0.0

        fsm = self.fsm
        d, yaw_err, t_margin = fsm.face_standoff()
        base_z = self.robot.data.root_link_pos_w[env_ids, 2]
        base_speed = self.robot.data.root_link_lin_vel_w[env_ids, :2].norm(dim=-1)
        in_band = (
            (fsm.fsm_state[env_ids] == STATE_FOLLOW)
            & (d[env_ids] >= self.cfg.gate_dist_range[0])
            & (d[env_ids] <= self.cfg.gate_dist_range[1])
            & (yaw_err[env_ids] <= self.cfg.gate_yaw_tol)
            # In front of the SEGMENT, not just the plane — beyond the end
            # the plane is phantom (no wall there to press).
            & (t_margin[env_ids] >= fsm.cfg.wall_end_margin)
            # Opportunistic, not arrival-latched: the base just has to PASS
            # through the press-ready envelope slowly — press_hold() below
            # parks it there (the attractor's arrival latch almost never
            # fires with a vel-degraded policy, and its parking lands
            # outside the reach band).
            & (base_speed <= self.cfg.gate_max_speed)
            # Health gate: never open a window on a falling/fallen base (the
            # FSM is fall-blind; play has no fell_over termination).
            & (base_z >= self.cfg.gate_min_base_z)
        )
        draw = torch.rand(len(env_ids), device=self.device) < self.cfg.wall_contact_prob
        draw &= in_band & ~self._just_reset[env_ids]
        self._just_reset[env_ids] = False

        ids = env_ids[draw]
        fsm.press_hold(ids)
        if len(ids):
            m = len(ids)
            rng = self.cfg.contact_ranges
            r = torch.empty(m, device=self.device)
            self.pose_command_b[ids, 1] = r.uniform_(*rng.pos_y)
            self.pose_command_b[ids, 2] = r.uniform_(*rng.pos_z)
            # Orientation: press exactly along the face inward normal,
            # expressed as a (clamped) heading-relative yaw — roll/pitch 0.
            _c, n_s, _ = self._followed_face()
            face_yaw = torch.atan2(-n_s[ids, 1], -n_s[ids, 0])
            dyaw = wrap_to_pi(face_yaw - self.robot.data.heading_w[ids]).clamp(
                -self.cfg.cmd_yaw_clamp, self.cfg.cmd_yaw_clamp
            )
            zeros = torch.zeros_like(dyaw)
            self.pose_command_b[ids, 3:] = quat_unique(
                quat_from_euler_xyz(zeros, zeros, dyaw)
            )
            self._apply_masked_x(ids, open_window=True)
            self.contact_mask[ids] = 1.0
            self.force_mag[ids] = torch.empty(m, device=self.device).uniform_(
                *self.cfg.force_range
            )
            self.time_left[ids] = torch.empty(m, device=self.device).uniform_(
                *self.cfg.contact_resampling_time_range
            )
        self._clip_free_targets(env_ids)

    # ── per-step update ─────────────────────────────────────────────────────

    def _update_command(self) -> None:
        masked = self.contact_mask > 0.5
        if masked.any():
            fsm = self.fsm
            ids = self._all_ids[masked]
            # Ramp the window out (instead of chopping) when the press is no
            # longer viable: FSM left the wall, the base is falling, the wall
            # receded beyond the reachable command box, the base slid past
            # the segment end (the face plane there is PHANTOM — pressing
            # air), or the base is at/behind the face.
            k = self._kiss_x(ids)
            d_all, _yaw, t_margin = fsm.face_standoff()
            left = (
                (fsm.fsm_state[ids] != STATE_FOLLOW)
                | (self.robot.data.root_link_pos_w[ids, 2] < self.cfg.gate_min_base_z)
                | (k > self.cfg.cmd_x_clamp[1] + 0.03)
                | (t_margin[ids] < 0.5 * fsm.cfg.wall_end_margin)
                | (d_all[ids] < 0.05)
            )
            if left.any():
                lids = ids[left]
                self.time_left[lids] = torch.minimum(
                    self.time_left[lids],
                    torch.full_like(self.time_left[lids], self.cfg.force_ramp_time),
                )
            # Wall-normal component: held from open, floored at live kiss,
            # overlap-capped (see _apply_masked_x).
            self._apply_masked_x(ids[~left], open_window=False)
        # Free targets ride the base too — keep them clear of walls as the
        # robot moves (a resample-time clip alone goes stale mid-walk).
        self._clip_free_targets(self._all_ids)
        super()._update_command()  # IK + ramps (+ no cube)
        # Windows that ramped out on wall-leave (or falling) release the mask
        # once the commanded force reaches zero.
        stale = (
            (self.contact_mask > 0.5)
            & (
                (self.fsm.fsm_state != STATE_FOLLOW)
                | (self.robot.data.root_link_pos_w[:, 2] < self.cfg.gate_min_base_z)
            )
            & (self._force_cmd_mag <= 1e-6)
        )
        if stale.any():
            self.contact_mask[stale] = 0.0
            self.force_mag[stale] = 0.0


@dataclass(kw_only=True)
class WallEePoseCommandCfg(HandForceEePoseCommandCfg):
    """Config for :class:`WallEePoseCommand`."""

    twist_command_name: str = "twist"
    wall_contact_prob: float = 0.6
    """Per-resample contact probability while in FOLLOW and in-band (the
    free-reach mixture keeps the remaining draws; ``contact_prob`` of the
    cube stage is unused here)."""
    gate_dist_range: tuple[float, float] = (0.34, 0.45)
    """Measured base-to-face band that allows opening a window. Faithful to
    the TRAINED contact sub-box: with tip-frame commands kiss-x = d, which
    must stay within the contact box x (0.33, 0.45) — beyond the upper end
    the press is at/over the arm-envelope edge (stage-1's free box is
    deliberately half-unreachable, the contact box is NOT), and an
    unreachable press means phantom support → the charge-and-crash
    failure."""
    gate_yaw_tol: float = 0.15
    """Max |heading − face normal| to open a window [rad]. Kept well inside
    the friction cone: at 0.30 the press obliquity approaches tan≈0.31 ≈
    μ/2 and the hand starts sliding along the face."""
    gate_min_base_z: float = 0.55
    """Min base height to open a window [m] — no presses on a falling base."""
    gate_max_speed: float = 0.35
    """Max planar base speed to open a window [m/s] — windows open on a
    slowly-passing base (press_hold parks it), never on a sprint/stagger."""
    cmd_x_clamp: tuple[float, float] = (0.33, 0.45)
    """Clamp of the wall-normal command component — the TRAINED contact
    sub-box x range (never the free box: its far half is outside the arm
    envelope)."""
    press_max_overlap: float = 0.08
    """Cap of the commanded tip's depth PAST the live face (arm-PD spring
    saturation — bounds the jam force when the base parks deep) [m]."""
    cmd_yaw_clamp: float = 0.12
    """Clamp of the command's heading-relative yaw (trained box ±0.1; the
    residual press-direction error stays well inside the friction cone)."""
    free_face_clearance: float = 0.05
    """Air kept between a FREE target's implied capsule tip and any wall."""

    def build(self, env: ManagerBasedRlEnv) -> WallEePoseCommand:
        return WallEePoseCommand(self, env)
