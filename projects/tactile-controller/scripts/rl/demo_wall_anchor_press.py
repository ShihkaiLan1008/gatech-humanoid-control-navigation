"""DEMO ONLY — anchor press + wall tapping, driven scripted on top of a
trained hand_force checkpoint.

Scenario (user spec): the robot stands still at a wall (no locomotion), ONE
hand holds a steady press on the wall, the OTHER cycles press -> lift off ->
move -> press at a fresh spot, ~1-2 s per spot, moderate force.

Tap spots are a hardcoded PLUS pattern (up / down / near / far / centre)
visited so consecutive presses are always on opposite sides of the patch —
uniform sampling in the reach box kept putting two presses next to each other
and read as "tapping the same place".

NB `--seed` only controls this script's spot/force draws. The env's own
start-up domain randomization and spawn reset are NOT seeded here, so runs
differ regardless; base drift in particular varies 0.25-0.9 m per 35 s.

This is a pure demo harness: it touches NOTHING in src/. Everything is a
runtime override on the loaded env cfg + command-term state, exactly like the
probe scripts. The env is `Mjlab-MPC-Guided-Wall-Following-G1` in PLAY mode;
its own FSM/sampling never runs (every command term's `time_left` is pinned,
so `_resample_command` is never reached) — this script IS the command source.

What is still the ENV's job (deliberately not re-implemented here):
  * `_kiss_x` / `_apply_masked_x`: where "on the wall" is, in command space.
  * force ramp in/out (`force_ramp_time`, `_contact_age`, `time_left`).
  * the IK arm reference, the loco-manip MPC, and the contact sensors.

Run (viewer):
  uv run python scripts/rl/demo_wall_anchor_press.py \
      --checkpoint logs/rsl_rl/g1_mpc_hand_force/hand_force_v1/model_74499.pt

Run (headless sanity check, prints press stats):
  uv run python scripts/rl/demo_wall_anchor_press.py --headless 30
"""

from __future__ import annotations

import argparse
import math
from dataclasses import asdict
from pathlib import Path

import torch

# Phases of the tapping hand.
_PRESS, _LIFT, _MOVE = 0, 1, 2


class WallTapDemo:
    """Scripted command source: standing base, anchor hand, tapping hand."""

    def __init__(
        self,
        env,
        anchor_side: str,
        standoff: float,
        anchor_force: float,
        tap_force_range: tuple[float, float],
        press_time_range: tuple[float, float],
        lift_time: float,
        move_time: float,
        lift_back: float,
        lift_up: float,
        cmd_x_max: float,
        press_overlap: float,
        spread_y: float,
        spread_z: float,
        jitter: float,
        random_spots: bool,
        corners: bool,
        trace: bool,
        seed: int,
    ) -> None:
        from rl.g1_assets.maze_wall import MAZE_WALL_HALF_LENGTHS

        self.env = env
        self.dt = env.step_dt
        self.dev = env.device
        self.B = env.num_envs
        self.ids = torch.arange(self.B, device=self.dev)
        self.gen = torch.Generator(device=self.dev).manual_seed(seed)

        self.fsm = env.command_manager.get_term("twist")
        self.anchor = env.command_manager.get_term(f"{anchor_side}_ee_pose")
        tap_side = "left" if anchor_side == "right" else "right"
        self.tap = env.command_manager.get_term(f"{tap_side}_ee_pose")
        self.tap_side = tap_side
        self.robot = env.scene["robot"]

        # Widest wall segment = the nicest backdrop; the rest are parked.
        self.wall = int(
            max(range(len(MAZE_WALL_HALF_LENGTHS)), key=MAZE_WALL_HALF_LENGTHS.__getitem__)
        )
        self.d0 = standoff

        self.anchor_force = anchor_force
        self.tap_force_range = tap_force_range
        self.press_time_range = press_time_range
        self.lift_time = lift_time
        self.move_time = move_time
        self.lift_back = lift_back
        self.lift_up = lift_up
        self.want_trace = trace

        # Tap spots: a hardcoded 3x3 grid on the wall, visited in a JUMP
        # order so consecutive presses are always far apart — uniform random
        # sampling in the patch kept putting two presses side by side and the
        # motion read as "tapping the same place". Columns are |y| (lateral),
        # rows are z (height), both in the command (yaw) frame; x is solved
        # onto the wall by the env. Grid extents come from the trained
        # contact sub-box, optionally widened for visibility.
        cr = self.tap.cfg.contact_ranges
        lo_y, hi_y = min(cr.pos_y), max(cr.pos_y)
        mid_y, half_y = (lo_y + hi_y) / 2, (hi_y - lo_y) / 2
        mid_z, half_z = sum(cr.pos_z) / 2, (cr.pos_z[1] - cr.pos_z[0]) / 2
        self.grid_y = tuple(mid_y + f * half_y * spread_y for f in (-1.0, 0.0, 1.0))
        self.grid_z = tuple(mid_z + f * half_z * spread_z for f in (-1.0, 0.0, 1.0))
        self.jitter = jitter
        self.sign_y = 1.0 if tap_side == "left" else -1.0
        self.random_spots = random_spots
        # Cell ids are row-major (row = z index, col = |y| index):
        #   0 1 2   (top)
        #   3 4 5   (middle)
        #   6 7 8   (bottom)
        # DEFAULT = the PLUS pattern (up, down, near, far, centre), visited so
        # every hop crosses the patch. The four CORNERS are deliberately
        # skipped: at the demo's braced standoff (~0.55 m, past the trained
        # band) a far-lateral + low-z target is well outside the arm envelope
        # — the arm strains for it and drags the whole body off station
        # (measured: 3x3-with-corners drifted ~1.7 m in 35 s, and no press
        # depth or force setting rescued it). `--corners` restores the 3x3.
        self.order = (0, 8, 2, 6, 4, 1, 7, 3, 5) if corners else (1, 7, 3, 5, 4)
        self.k = torch.randint(
            0, len(self.order), (self.B,), device=self.dev, generator=self.gen
        )

        # Anchor spot: mid-patch on its own side.
        acr = self.anchor.cfg.contact_ranges
        self.anchor_y = (min(acr.pos_y) + max(acr.pos_y)) / 2.0
        self.anchor_z = (acr.pos_z[0] + acr.pos_z[1]) / 2.0

        # Snappier ramps than training (demo readability); per-term cfg
        # objects belong to this process only.
        self.anchor.cfg.force_ramp_time = 0.4
        self.tap.cfg.force_ramp_time = 0.3
        # The press reaction pushes the base ~0.15 m past the trained
        # standoff band. Let the commanded target follow the wall out there,
        # otherwise x saturates at the training clamp and the arm has to
        # overreach ~0.1 m past its own command to stay in contact.
        for term in (self.anchor, self.tap):
            lo, _hi = term.cfg.cmd_x_clamp
            term.cfg.cmd_x_clamp = (lo, cmd_x_max)
            # NEGATIVE overlap = command the target a few mm SHORT of the
            # face. `_apply_masked_x` floors at the live kiss and caps at
            # kiss+overlap, so a negative cap parks the command just off the
            # wall; this checkpoint's habitual overreach still lands the
            # contact, but shallower — less penetration, less reaction
            # shoving the base backwards.
            term.cfg.press_max_overlap = press_overlap

        # Tapping-hand phase state.
        self.phase = torch.full((self.B,), _PRESS, dtype=torch.long, device=self.dev)
        self.t_phase = torch.zeros(self.B, device=self.dev)
        self.press_T = torch.zeros(self.B, device=self.dev)
        self.spot = torch.zeros(self.B, 2, device=self.dev)
        self.next_spot = torch.zeros(self.B, 2, device=self.dev)
        self.tap_force = torch.zeros(self.B, device=self.dev)
        self._placed = False
        self._prev_ep = torch.zeros(self.B, device=self.dev)

        # Stats (headless mode).
        self.stat = {"n": 0, "tap_touch": 0, "tap_f": 0.0, "tap_n": 0,
                     "anc_touch": 0, "anc_f": 0.0, "drift": 0.0,
                     "lift_n": 0, "lift_touch": 0, "cycles": 0}
        self.trace: list[tuple[float, float, float, float, float]] = []
        self.spot_log: list[tuple[float, float]] = []

    # ── helpers ─────────────────────────────────────────────────────────────

    def _u(self, n: int, lo: float, hi: float) -> torch.Tensor:
        return torch.rand(n, device=self.dev, generator=self.gen) * (hi - lo) + lo

    def _sample_spot(self, sel: torch.Tensor) -> torch.Tensor:
        """Next grid spot for the selected envs (advances their cursor)."""
        n = int(sel.sum()) if sel.dtype == torch.bool else len(sel)
        if self.random_spots:
            y = self.sign_y * self._u(
                n, min(self.grid_y), max(self.grid_y)
            )
            z = self._u(n, min(self.grid_z), max(self.grid_z))
            return torch.stack([y, z], dim=-1)
        self.k[sel] = (self.k[sel] + 1) % len(self.order)
        cells = torch.tensor(self.order, device=self.dev)[self.k[sel]]
        gy = torch.tensor(self.grid_y, device=self.dev)[cells % 3]
        gz = torch.tensor(self.grid_z, device=self.dev)[2 - cells // 3]
        j = self.jitter
        return torch.stack(
            [
                self.sign_y * (gy + self._u(n, -j, j)),
                gz + self._u(n, -j, j),
            ],
            dim=-1,
        )

    def _place_wall(self) -> None:
        """Park every wall but one, and put that one squarely in front of the
        robot at the standoff the trained press geometry expects."""
        m = self.fsm.wall_map
        base = self.robot.data.root_link_pos_w[:, :2]
        heading = self.robot.data.heading_w
        u = torch.stack([heading.cos(), heading.sin()], dim=-1)
        m.active[:] = False
        m.active[:, self.wall] = True
        m.c[:, self.wall] = base + (self.d0 + m.thickness / 2) * u
        m.th[:, self.wall] = torch.atan2(u[:, 1], u[:, 0]) + math.pi / 2
        self.fsm._write_wall_poses(self.ids)
        self._start_xy = base.clone()
        self._start_yaw = heading.clone()
        self.fsm.arrived[:] = True  # start parked; the latch takes over

    def _reset_hands(self) -> None:
        self.phase[:] = _PRESS
        self.t_phase[:] = 0.0
        all_sel = torch.ones(self.B, dtype=torch.bool, device=self.dev)
        self.press_T[:] = self._u(self.B, *self.press_time_range)
        self.spot[:] = self._sample_spot(all_sel)
        self.next_spot[:] = self._sample_spot(all_sel)
        self.tap_force[:] = self._u(self.B, *self.tap_force_range)

    # ── per-step command source ─────────────────────────────────────────────

    def __call__(self) -> None:
        from rl.mpc_online_tracking.tasks.wall_following.mdp.commands import (
            STATE_FOLLOW,
        )

        # (Re)build the scene after any reset (the viewer's reset button
        # re-randomizes the maze and teleports the robot).
        ep = self.env.episode_length_buf.float()
        if not self._placed or bool((ep < self._prev_ep).any()):
            self._place_wall()
            self._reset_hands()
            self._placed = True
        self._prev_ep = ep.clone()
        # NB never re-place the wall mid-run: the slab would teleport INTO the
        # pressing hands (extended, touching the old face) and MuJoCo resolves
        # that overlap with ~7 kN. Measured 2026-07-29.

        m = self.fsm.wall_map
        base = self.robot.data.root_link_pos_w[:, :2]
        heading = self.robot.data.heading_w

        # ── base: stand, hard zero twist ─────────────────────────────────────
        # Station keeping via the Se2 attractor was TRIED and is WORSE
        # (measured): the press reaction pushes the base back ~0.15 m, the
        # attractor then commands a corrective walk, and this checkpoint
        # cannot walk while pressing (the known stop-to-press habit) — it
        # shuffles in place and the LATERAL drift grows monotonically
        # (-0.45 m in 40 s vs -0.30 m with zero twist). A demo wants no
        # locomotion commands at all; the settle-then-reanchor step below
        # handles the one-time backward push instead.
        self.fsm.fsm_state[:] = STATE_FOLLOW
        self.fsm.arrived[:] = True
        self.fsm.is_standing_env[:] = True
        self.fsm.is_twist_env[:] = False
        self.fsm.follow_wall[:] = self.wall
        _t, d_perp = m.coords(base, self.ids)
        self.fsm.follow_side[:] = torch.where(d_perp[:, self.wall] >= 0, 1.0, -1.0)
        self.fsm.pose_target_w[:, :2] = self._start_xy
        self.fsm.pose_target_w[:, 2] = self._start_yaw
        self.fsm.time_left[:] = 1e6
        self.fsm._write_wall_poses(self.ids)

        # Command yaw that squares the press to the wall face (same clamp the
        # env's own sampler uses).
        from mjlab.utils.lab_api.math import (
            quat_from_euler_xyz,
            quat_unique,
            wrap_to_pi,
        )

        _c, _t2, n_s, _hl = m.face_frame(self.ids, self.fsm.follow_wall, self.fsm.follow_side)
        face_yaw = torch.atan2(-n_s[:, 1], -n_s[:, 0])
        zeros = torch.zeros(self.B, device=self.dev)

        def set_pose(term, y: torch.Tensor, z: torch.Tensor) -> None:
            dyaw = wrap_to_pi(face_yaw - heading).clamp(
                -term.cfg.cmd_yaw_clamp, term.cfg.cmd_yaw_clamp
            )
            term.pose_command_b[:, 1] = y
            term.pose_command_b[:, 2] = z
            term.pose_command_b[:, 3:] = quat_unique(
                quat_from_euler_xyz(zeros, zeros, dyaw)
            )
            term.phantom[:] = 0.0
            term.time_left[:] = 1e6

        # ── anchor hand: one steady press, never released ────────────────────
        a = self.anchor
        set_pose(
            a,
            torch.full((self.B,), self.anchor_y, device=self.dev),
            torch.full((self.B,), self.anchor_z, device=self.dev),
        )
        first_anchor = bool((a.contact_mask < 0.5).any())
        a.contact_mask[:] = 1.0
        a.force_mag[:] = self.anchor_force
        a._contact_age[:] = 10.0  # past ramp-in and the reward grace window
        if first_anchor:
            a._apply_masked_x(self.ids, open_window=True)

        # ── tapping hand: press -> lift -> move -> press ─────────────────────
        t = self.tap
        self.t_phase += self.dt
        remaining = self.press_T - self.t_phase

        # Transitions (evaluated before writing this step's command).
        to_lift = (self.phase == _PRESS) & (remaining <= 2.0 * self.dt)
        to_move = (self.phase == _LIFT) & (self.t_phase >= self.lift_time)
        to_press = (self.phase == _MOVE) & (self.t_phase >= self.move_time)
        if bool(to_lift.any()):
            n = int(to_lift.sum())
            self.next_spot[to_lift] = self._sample_spot(to_lift)
            self.tap_force[to_lift] = self._u(n, *self.tap_force_range)
            self.phase[to_lift] = _LIFT
            self.t_phase[to_lift] = 0.0
        if bool(to_move.any()):
            self.phase[to_move] = _MOVE
            self.t_phase[to_move] = 0.0
        if bool(to_press.any()):
            self.spot[to_press] = self.next_spot[to_press]
            self.press_T[to_press] = self._u(int(to_press.sum()), *self.press_time_range)
            self.phase[to_press] = _PRESS
            self.t_phase[to_press] = 0.0
            self.stat["cycles"] += int(to_press.sum())
            self.spot_log.append(
                (float(self.spot[0, 0]), float(self.spot[0, 1]))
            )
        remaining = self.press_T - self.t_phase

        pressing = self.phase == _PRESS
        # Transit target: lifted off the wall, arcing over to the next spot.
        s = (self.t_phase / max(self.move_time, 1e-6)).clamp(0.0, 1.0)
        s = s * s * (3.0 - 2.0 * s)  # smoothstep
        moving = self.phase == _MOVE
        yz = torch.where(
            moving.unsqueeze(-1),
            self.spot + (self.next_spot - self.spot) * s.unsqueeze(-1),
            self.spot,
        )
        # Lift height: full during LIFT/MOVE, tapering back in as MOVE ends.
        up = torch.where(
            self.phase == _LIFT,
            (self.t_phase / max(self.lift_time, 1e-6)).clamp(0.0, 1.0),
            torch.where(moving, torch.ones_like(s), torch.zeros_like(s)),
        )
        set_pose(t, yz[:, 0], yz[:, 1] + self.lift_up * up)

        t.contact_mask[:] = pressing.float()
        t.force_mag[:] = torch.where(pressing, self.tap_force, torch.zeros_like(self.tap_force))
        t._contact_age[:] = torch.where(pressing, self.t_phase, torch.zeros_like(self.t_phase))
        t.time_left[:] = torch.where(
            pressing, remaining.clamp(min=0.06), torch.full_like(remaining, 1e6)
        )
        # Free (lifted) phases: pull the target back off the face so the hand
        # visibly clears the wall before translating.
        kiss = t._kiss_x(self.ids)
        t.pose_command_b[:, 0] = torch.where(
            pressing, t.pose_command_b[:, 0], kiss - self.lift_back
        )
        if bool(to_press.any()):
            # Latch the press depth at the fresh spot.
            t._apply_masked_x(self.ids[to_press], open_window=True)

        self._collect_stats()

    def _collect_stats(self) -> None:
        s = self.stat
        s["n"] += self.B
        af = self.anchor.measured_force_w().norm(dim=-1)
        s["anc_touch"] += int((af > 0.5).sum())
        s["anc_f"] += float(af.sum())
        lifted = self.phase != _PRESS
        if bool(lifted.any()):
            lf = self.tap.measured_force_w().norm(dim=-1)[lifted]
            s["lift_n"] += int(lifted.sum())
            s["lift_touch"] += int((lf > 0.5).sum())
        pressing = (self.tap.contact_mask > 0.5) & (self.tap._contact_age > 0.4)
        if bool(pressing.any()):
            tf = self.tap.measured_force_w().norm(dim=-1)[pressing]
            s["tap_n"] += int(pressing.sum())
            s["tap_touch"] += int((tf > 0.5).sum())
            s["tap_f"] += float(tf.sum())
        rel = self.robot.data.root_link_pos_w[:, :2] - self._start_xy
        s["drift"] = float(rel.norm(dim=-1).max())
        # Split the drift into wall-normal (backward = positive) and
        # tangential, and log the live standoff.
        u = torch.stack([self._start_yaw.cos(), self._start_yaw.sin()], dim=-1)
        v = torch.stack([-self._start_yaw.sin(), self._start_yaw.cos()], dim=-1)
        d_live, _yaw, _tm = self.fsm.face_standoff()
        if self.want_trace and s["n"] % 50 == 0:
            self.trace.append(
                (
                    s["n"] * self.dt / max(self.B, 1),
                    float(-(rel * u).sum(-1)[0]),
                    float((rel * v).sum(-1)[0]),
                    float(d_live[0]),
                    float(self.anchor.measured_force_w().norm(dim=-1)[0]),
                )
            )

    def report(self) -> None:
        s = self.stat
        n = max(s["n"], 1)
        tn = max(s["tap_n"], 1)
        print(f"\nsteps            : {s['n']}")
        print(f"anchor contact   : {s['anc_touch'] / n:.1%}   mean |f| {s['anc_f'] / n:5.1f} N")
        print(f"tap contact      : {s['tap_touch'] / tn:.1%}   mean |f| {s['tap_f'] / tn:5.1f} N"
              f"   (settled press steps: {s['tap_n']})")
        ln = max(s["lift_n"], 1)
        print(f"lift-phase touch : {s['lift_touch'] / ln:.1%}  (should be ~0: hand clears the wall)")
        print(f"press cycles     : {s['cycles']}")
        if self.spot_log:
            seq = "  ".join(f"({y:+.2f},{z:.2f})" for y, z in self.spot_log)
            print(f"tap spots (y,z)  : {seq}")
            hops = [
                math.hypot(b[0] - a[0], b[1] - a[1])
                for a, b in zip(self.spot_log, self.spot_log[1:])
            ]
            if hops:
                print(f"hop distance     : min {min(hops):.3f}  mean "
                      f"{sum(hops) / len(hops):.3f}  max {max(hops):.3f} m")
        print(f"base drift       : {s['drift']:.3f} m")
        print("   t[s]  back[m]  lat[m]  standoff  |f_anchor|")
        for row in self.trace[::4]:
            print(f"  {row[0]:5.1f}  {row[1]:+6.3f}  {row[2]:+6.3f}   {row[3]:6.3f}   {row[4]:6.1f}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--checkpoint",
        default="logs/rsl_rl/g1_mpc_hand_force/hand_force_v1/model_74499.pt",
    )
    p.add_argument("--task", default="Mjlab-MPC-Guided-Wall-Following-G1")
    p.add_argument("--num_envs", type=int, default=1)
    p.add_argument("--anchor", choices=("left", "right"), default="right")
    p.add_argument("--standoff", type=float, default=0.33)
    p.add_argument("--anchor_force", type=float, default=8.0)
    p.add_argument("--tap_force", type=float, nargs=2, default=(6.0, 12.0))
    p.add_argument("--press_time", type=float, nargs=2, default=(1.2, 2.0))
    p.add_argument("--lift_time", type=float, default=0.45)
    p.add_argument("--move_time", type=float, default=0.55)
    p.add_argument("--lift_back", type=float, default=0.16, help="retreat off the wall [m]")
    p.add_argument("--lift_up", type=float, default=0.03, help="arc height while moving [m]")
    p.add_argument("--cmd_x_max", type=float, default=0.60,
                   help="upper clamp on the commanded press depth [m]")
    p.add_argument("--press_overlap", type=float, default=0.08,
                   help="commanded depth past the wall face [m]; negative = short of it")
    p.add_argument("--spread_y", type=float, default=1.0,
                   help="lateral grid extent, x the trained box half-width")
    p.add_argument("--spread_z", type=float, default=1.4,
                   help="vertical grid extent, x the trained box half-height")
    p.add_argument("--jitter", type=float, default=0.012,
                   help="per-press jitter on the grid spot [m]")
    p.add_argument("--corners", action="store_true",
                   help="full 3x3 grid incl. the (hard-to-reach) corners")
    p.add_argument("--random_spots", action="store_true",
                   help="uniform random spots instead of the jump-order grid")
    p.add_argument("--trace", action="store_true",
                   help="headless: print a per-second drift/standoff/force trace")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--headless", type=float, default=0.0, help="sim seconds, no viewer")
    p.add_argument("--viewer", choices=("auto", "native", "viser"), default="auto")
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
    from mjlab.utils.torch import configure_torch_backends

    configure_torch_backends()

    env_cfg = load_env_cfg(args.task, play=True)
    env_cfg.scene.num_envs = args.num_envs
    for name in ("left_ee_pose", "right_ee_pose"):
        env_cfg.commands[name].debug_vis = True
    env_cfg.commands["twist"].debug_vis = False  # no goal gizmo: it stands
    agent_cfg = load_rl_cfg(args.task)

    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    wenv = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner_cls = load_runner_cls(args.task) or MjlabOnPolicyRunner
    runner = runner_cls(wenv, asdict(agent_cfg), device=device)
    runner.load(
        str(Path(args.checkpoint).resolve()),
        load_cfg={"actor": True},
        strict=True,
        map_location=device,
    )
    policy = runner.get_inference_policy(device=device)

    demo = WallTapDemo(
        env,
        anchor_side=args.anchor,
        standoff=args.standoff,
        anchor_force=args.anchor_force,
        tap_force_range=tuple(args.tap_force),
        press_time_range=tuple(args.press_time),
        lift_time=args.lift_time,
        move_time=args.move_time,
        lift_back=args.lift_back,
        lift_up=args.lift_up,
        cmd_x_max=args.cmd_x_max,
        press_overlap=args.press_overlap,
        spread_y=args.spread_y,
        spread_z=args.spread_z,
        jitter=args.jitter,
        random_spots=args.random_spots,
        corners=args.corners,
        trace=args.trace,
        seed=args.seed,
    )

    def demo_policy(obs):
        # Runs immediately before each env.step(): the pinned commands are what
        # the command manager, the MPC and the physics see for that step.
        demo()
        return policy(obs)

    print(
        f"[demo] anchor={args.anchor} hand holds {args.anchor_force:.0f} N; "
        f"{demo.tap_side} hand taps {args.tap_force[0]:.0f}-{args.tap_force[1]:.0f} N "
        f"every {args.press_time[0]:.1f}-{args.press_time[1]:.1f} s"
    )

    if args.headless > 0:
        obs, _ = env.reset()
        steps = int(args.headless / env.step_dt)
        with torch.no_grad():
            for _ in range(steps):
                action = demo_policy(obs)
                obs, _, _, _ = wenv.step(action)
        demo.report()
        env.close()
        return

    import os

    from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

    if args.viewer == "auto":
        has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
        backend = "native" if has_display else "viser"
    else:
        backend = args.viewer

    if backend == "native":
        NativeMujocoViewer(env, demo_policy).run()
    else:
        ViserPlayViewer(env, demo_policy).run()
    env.close()


if __name__ == "__main__":
    main()
