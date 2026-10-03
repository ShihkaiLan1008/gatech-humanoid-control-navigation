"""On-wall hand press command — the DECOUPLED-family adapter of force_maze.

Deployment-faithful and OPERATOR-DRIVEN (user decree 2026-08-11, v2):

* NO transition governor: every draw is a JUMP (``jump_prob`` must be 1.0) —
  a deploy command stream steps, it never glides.
* Pose sampling is wall-biased: whenever the base fronts a wall face whose
  kiss-x lands in ``wall_x_range``, the draw goes ON that face with
  ``wall_target_prob`` (default 0.9); otherwise (and for the remaining
  draws) it is a plain pool draw, clipped clear of wall surfaces.
* Contact mask and force magnitude are controlled by the OPERATOR — a viser
  checkbox + slider per hand (``create_gui``); there is no automatic window
  scheduling and no geometric open/close judgment. Whether a press makes
  sense is the operator's call, exactly like a deploy command stream. While
  the mask is ON the pose draw is FROZEN (the surface being pressed must not
  teleport); headless scripts drive the same fields directly
  (``contact_mask`` / ``force_mag``).
* The twist channel is UNTOUCHED from here (honest-eval decree — see
  rl.eval.mdp.maze_twist): this term only reads the twist term's wall map.

Everything else — measured force from the per-hand-per-wall tactile↔wall
sensors, the duck-typed accessor surface (mask / force_cmd / press_dir /
measured_force_w) consumed by the decoupled obs/rewards/loco-MPC — is
inherited from :class:`EeContactTargetCommand` with ``cube_entity_name=None``.
``phantom_prob`` must stay 0: a masked window against a real wall cannot be
honestly phantom.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

import torch
from torch import Tensor

from mjlab.utils.lab_api.math import (
    matrix_from_quat,
    quat_apply,
    quat_apply_inverse,
    quat_from_euler_xyz,
    quat_unique,
    wrap_to_pi,
)

from rl.decoupled.mdp.commands.ee_contact import (
    EeContactTargetCommand,
    EeContactTargetCommandCfg,
)
from rl.decoupled.mdp.commands.ee_tartget import EeTargetCommand
from rl.eval.mdp.maze_twist import MazeFollowTwistCommand
from rl.g1_assets.press_cube import FACE_OFFSET
from rl.g1_assets.wall_map import _INF, _pt_seg_dist

if TYPE_CHECKING:
    import viser

    from mjlab.envs import ManagerBasedRlEnv


class WallEePressCommand(EeContactTargetCommand):
    """Wall-biased jump pose draws + operator-driven contact/force."""

    cfg: "WallEePressCommandCfg"

    def __init__(self, cfg: "WallEePressCommandCfg", env: "ManagerBasedRlEnv") -> None:
        assert cfg.cube_entity_name is None, "force_maze has no press cubes"
        assert cfg.phantom_prob == 0.0, (
            "phantom windows cannot be materialized honestly against real walls"
        )
        assert cfg.jump_prob == 1.0, (
            "force_maze commands must JUMP — deploy streams have no transition"
        )
        super().__init__(cfg, env)
        self._fsm: MazeFollowTwistCommand | None = None  # lazy (build order)
        # Wall draws are INTENTIONAL face targets — the free-goal clip must
        # not push them back off the wall (mask is operator-owned and mostly
        # 0, so "unmasked" cannot mean "keep clear of walls" here).
        self._on_wall = torch.zeros(self.num_envs, dtype=torch.bool, device=env.device)
        # Viser handles (set by create_gui when the viewer is active).
        self._gui_contact: "viser.GuiCheckboxHandle | None" = None
        self._gui_force: "viser.GuiSliderHandle | None" = None
        self._gui_get_env_idx: Callable[[], int] | None = None

    @property
    def fsm(self) -> MazeFollowTwistCommand:
        if self._fsm is None:
            self._fsm = self._env.command_manager.get_term(
                self.cfg.twist_command_name
            )
        return self._fsm

    # ── operator GUI ────────────────────────────────────────────────────────

    def create_gui(
        self,
        name: str,
        server: "viser.ViserServer",
        get_env_idx: Callable[[], int],
        on_change: Callable[[], None] | None = None,
        request_action: Callable[[str, Any], None] | None = None,
    ) -> None:
        del on_change, request_action
        f_lo, f_hi = self.cfg.force_range
        with server.gui.add_folder(name.capitalize()):
            self._gui_contact = server.gui.add_checkbox(
                "contact", initial_value=False
            )
            self._gui_force = server.gui.add_slider(
                "force [N]",
                min=0.0,
                max=f_hi,
                step=0.5,
                initial_value=min(max(10.0, f_lo), f_hi),
            )
        self._gui_get_env_idx = get_env_idx

    def _apply_gui(self) -> None:
        if self._gui_contact is None or self._gui_get_env_idx is None:
            return
        idx = self._gui_get_env_idx()
        self.contact_mask[idx] = 1.0 if self._gui_contact.value else 0.0
        self.force_mag[idx] = float(self._gui_force.value)

    # ── wall geometry (sampling only — no window judgment) ──────────────────

    def _kiss_x(self, ids: Tensor, c: Tensor, n_s: Tensor) -> Tensor:
        """Goal-x (BASE frame) putting the commanded tip exactly ON the face
        plane (c, n̂_s), solved from the measured base pose and goal y/z."""
        root_pos = self.robot.data.root_link_pos_w[ids]
        R = matrix_from_quat(self.robot.data.root_link_quat_w[ids])  # (n,3,3)
        y_b = self._goal_pos[ids, 1]
        z_b = self._goal_pos[ids, 2]
        T2 = self.fsm.wall_map.thickness / 2
        ex_n = (R[:, :2, 0] * n_s).sum(-1)  # ≈ −1 facing the wall
        ey_n = (R[:, :2, 1] * n_s).sum(-1)
        ez_n = (R[:, :2, 2] * n_s).sum(-1)
        rhs = (
            T2
            + FACE_OFFSET
            - ((root_pos[:, :2] - c) * n_s).sum(-1)
            - y_b * ey_n
            - z_b * ez_n
        )
        return rhs / ex_n.clamp(max=-0.5)

    def _push_clear(self, pos_b: Tensor, ids: Tensor) -> Tensor:
        """Return pos_b (base frame, (n,3)) pushed so its world point keeps
        FACE_OFFSET + free_face_clearance of air in front of every wall (an
        unmasked target inside a wall is unreachable ghost-force bait)."""
        fsm = self.fsm
        root_pos = self.robot.data.root_link_pos_w[ids]
        aq = self.anchor_quat()[ids]
        pos_w = root_pos + quat_apply(aq, pos_b)
        p = pos_w[:, :2]
        need = (
            fsm.wall_map.thickness / 2
            + FACE_OFFSET
            + self.cfg.free_face_clearance
        )
        s0, s1 = fsm.wall_map.endpoints(ids)
        d = _pt_seg_dist(p.unsqueeze(1), s0, s1)
        d = torch.where(fsm.wall_map.active[ids], d, torch.full_like(d, _INF))
        dmin, widx = d.min(dim=1)
        bad = dmin < need
        if not bad.any():
            return pos_b
        bids = ids[bad]
        w = widx[bad]
        ar = torch.arange(len(ids), device=self.device)[bad]
        n = fsm.wall_map.n_hat(bids)[torch.arange(len(bids), device=self.device), w]
        c = fsm.wall_map.c[bids, w]
        dp = ((p[ar] - c) * n).sum(-1)
        side = torch.where(dp >= 0, 1.0, -1.0)
        p_new = p[ar] + (need - dp.abs()).unsqueeze(-1) * (n * side.unsqueeze(-1))
        new_w = pos_w.clone()
        new_w[ar, :2] = p_new
        out = pos_b.clone()
        out[ar] = quat_apply_inverse(aq[ar], new_w[ar] - root_pos[ar])
        return out

    def _clip_free_goals(self) -> None:
        free = self._all_ids[(self.contact_mask < 0.5) & ~self._on_wall]
        if len(free):
            self._goal_pos[free] = self._push_clear(self._goal_pos[free], free)

    # ── sampling ────────────────────────────────────────────────────────────

    def _resample_command(self, env_ids: Tensor) -> None:
        cfg = self.cfg
        self._just_reset[env_ids] = False
        # A live mask FREEZES the pose draw: the surface the operator is
        # pressing must not teleport. The timer re-arms upstream; the goal
        # holds until the operator releases contact.
        env_ids = env_ids[self.contact_mask[env_ids] < 0.5]
        if len(env_ids) == 0:
            return

        # Pool draw (jump) for everyone; wall-biased overwrite below.
        EeTargetCommand._resample_command(self, env_ids)
        self.phantom[env_ids] = 0.0

        # Nearest fronting wall face of the base; a wall target exists iff
        # its kiss-x lands in wall_x_range (implicitly: facing + in reach).
        m = self.fsm.wall_map
        p = self.robot.data.root_link_pos_w[env_ids, :2]
        _dist, wall, side = m.nearest_face(p, env_ids, cfg.wall_end_margin)
        c, _t, n_s, _hl = m.face_frame(env_ids, wall, side)

        n = len(env_ids)
        dev = self.device
        y = torch.empty(n, device=dev).uniform_(*cfg.contact_pos_y)
        z = torch.empty(n, device=dev).uniform_(*cfg.contact_pos_z)
        # Kiss uses the goal's y/z — stage them before solving.
        pool_goal = self._goal_pos[env_ids].clone()
        self._goal_pos[env_ids, 1] = y
        self._goal_pos[env_ids, 2] = z
        k = self._kiss_x(env_ids, c, n_s)
        valid = (k >= cfg.wall_x_range[0]) & (k <= cfg.wall_x_range[1])
        draw = valid & (
            torch.rand(n, device=dev) < cfg.wall_target_prob
        )
        # Non-wall draws keep the pool goal.
        self._goal_pos[env_ids] = torch.where(
            draw.unsqueeze(-1), self._goal_pos[env_ids], pool_goal
        )
        self._goal_pos[env_ids, 0] = torch.where(draw, k, pool_goal[:, 0])
        self._on_wall[env_ids] = draw

        ids = env_ids[draw]
        if len(ids):
            # Press along the face inward normal (heading-relative yaw,
            # clamped to the trained box; roll/pitch 0).
            ns_d = n_s[draw]
            face_yaw = torch.atan2(-ns_d[:, 1], -ns_d[:, 0])
            dyaw = wrap_to_pi(face_yaw - self.robot.data.heading_w[ids]).clamp(
                -cfg.cmd_yaw_clamp, cfg.cmd_yaw_clamp
            )
            zeros = torch.zeros_like(dyaw)
            self._goal_quat[ids] = quat_unique(
                quat_from_euler_xyz(zeros, zeros, dyaw)
            )

        # Jump semantics for the (possibly replaced) goal.
        self._start_pos[env_ids] = self._goal_pos[env_ids]
        self._start_quat[env_ids] = self._goal_quat[env_ids]
        self.pose_command_b[env_ids, :3] = self._goal_pos[env_ids]
        self.pose_command_b[env_ids, 3:] = self._goal_quat[env_ids]

    # ── per-step update ─────────────────────────────────────────────────────

    def _update_command(self) -> None:
        cfg = self.cfg
        self._apply_gui()  # operator: mask + force of the viewed env
        # Free goals ride the base — keep them clear of walls as it walks.
        self._clip_free_goals()
        EeTargetCommand._update_command(self)  # jump: pose ≡ goal
        # Commanded force: slew toward mask · force_mag (smooth ramp at the
        # operator's toggle; live slider tracking while pressed).
        target = self.force_mag * (self.contact_mask > 0.5).float()
        dv = cfg.force_slew_rate * self._env.step_dt
        self._force_cmd_mag += (target - self._force_cmd_mag).clamp(-dv, dv)

    def reset(self, env_ids) -> dict[str, float]:
        # Release the operator's contact on a fallen/reset env — the fresh
        # maze has no wall where the old press was.
        if (
            self._gui_contact is not None
            and self._gui_get_env_idx is not None
            and bool((env_ids == self._gui_get_env_idx()).any())
        ):
            self._gui_contact.value = False
        self._on_wall[env_ids] = False
        return super().reset(env_ids)


@dataclass(kw_only=True)
class WallEePressCommandCfg(EeContactTargetCommandCfg):
    """Config for :class:`WallEePressCommand`."""

    twist_command_name: str = "twist"
    """The maze-owning twist term (wall map access only — never written)."""
    wall_target_prob: float = 0.9
    """Probability that a draw goes ON the fronting wall face when one is in
    reach (kiss-x inside wall_x_range); the rest stay pool draws."""
    wall_x_range: tuple[float, float] = (0.30, 0.48)
    """Kiss-x acceptance band [m]: a wall target exists only where the
    solved wall-normal command lands here (around the trained press-x box —
    the nominal tip sits at x 0.332)."""
    wall_end_margin: float = 0.30
    """Faces count as "fronting" only within the segment minus this margin
    [m] — beyond the end the plane is phantom (no wall to press)."""
    contact_pos_y: tuple[float, float] = (0.08, 0.35)
    """Base-frame y of wall targets — PER HAND (right cfg passes negatives).
    Inside the pool cuts (y_margin 0.05, y_reach 0.3 from the shoulder)."""
    contact_pos_z: tuple[float, float] = (0.10, 0.25)
    """Base-frame z of wall targets — inside the pool cut (0.1, 0.3)."""
    cmd_yaw_clamp: float = 0.12
    """Clamp of the wall target's heading-relative yaw (trained box ±0.1)."""
    free_face_clearance: float = 0.05
    """Air kept between a FREE target and any wall surface [m]."""
    force_slew_rate: float = 40.0
    """Commanded-force slew toward the operator's slider [N/s] (~0.6 s over
    the trained 25 N window — matches the trained trapezoid ramps)."""

    def build(self, env: "ManagerBasedRlEnv") -> WallEePressCommand:
        return WallEePressCommand(self, env)
