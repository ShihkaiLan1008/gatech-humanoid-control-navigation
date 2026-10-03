"""N independent Room 2 worlds that stand still and touch.

WHY THIS EXISTS.  `VForceBackend` is written for one environment: every read
is `[0]`, and every primitive is a blocking `while` loop that calls `_step()`.
That is right for a demo, which is one robot walking one route.  It is wrong
for a collector, because the measurement said so:

    n_envs      1      16      64     128     256     512
    ms/step  41.9    47.4    49.1    47.7    50.6    52.6
    realtime 0.48x   6.8x    26x     54x     101x    195x

mujoco-warp compiles ONE model and batches its data across worlds, so 512 G1s
cost 25 % more per step than one.  A single-environment collector spends 57 h
on 12 000 transitions; 128 environments spend well under an hour.

WHAT THIS IS NOT.  Not a general backend.  There is no walking, no turning, no
wall following, no lane, no landmark, no planner -- the base is placed and
held, and the only thing that moves is a hand.  Everything physical is taken
from `VForceConfig` so the touch the collector logs is the touch the demo
performs: the same carry, the same creep rate, the same contact intent, the
same debounce, the same guard.

WORLDS ARE INDEPENDENT.  Each env is its own mujoco-warp world, so the rooms
cannot collide with one another and `env_spacing` is a placement convenience
rather than a safety constraint.  Each env carries the room at its own
randomized pose, written per-env through the wall map and the obstacle mocap
poses.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from execution.vforce_backend import (CARRY_ANCHOR_Q, VForceConfig,
                                      _inject_force_family,
                                      _stub_reach_pool_if_no_curobo)
from global_nav.geometry import wrap_angle
from lane_graph.mj_assets import build_obstacles_v3
from local_tactile.fsm import Mode

HANDS = ("left", "right")


@dataclass
class VecCfg:
    n_envs: int = 128
    seed: int = 0
    env_spacing: float = 12.0
    """Room 2 is ~4 x 4 m.  Worlds are independent so this cannot cause a
    cross-env collision; it is set generously anyway so a rendered or
    debugged scene is legible."""


class VecTouchEnv:
    """The batched simulator, addressed per environment."""

    def __init__(self, scene, cfg: VecCfg, mj: VForceConfig | None = None,
                 device: str | None = None):
        import torch
        from mjlab.tasks.registry import (load_env_cfg, load_rl_cfg,
                                          load_runner_cls)
        from mjlab.utils.torch import configure_torch_backends
        import mjlab.tasks  # noqa: F401
        import dataclasses

        from local_tactile.pipeline import resolve_checkpoint

        self.torch = torch
        self.scene = scene
        self.vcfg = cfg
        self.mj = mj or VForceConfig(seed=11 + cfg.seed,
                                     episode_budget_s=5.0e5)
        self.N = int(cfg.n_envs)
        self.rng = np.random.default_rng(cfg.seed)

        configure_torch_backends()
        self.device = device or ("cuda:0" if torch.cuda.is_available()
                                 else "cpu")
        _stub_reach_pool_if_no_curobo()
        ecfg = load_env_cfg(self.mj.task, play=True)
        # Hardware-scale sensor noise on both actor groups -- the backend's
        # own contract, and the only mode there is.
        from mjlab.utils.noise import UniformNoiseCfg as _U
        real = {"base_ang_vel": 0.02, "projected_gravity": 0.01,
                "joint_pos": 0.001, "joint_vel": 0.15}
        for grp in ("upper_body_actor", "lower_body_actor"):
            g = ecfg.observations[grp]
            g.enable_corruption = True
            for name, term in (g.terms or {}).items():
                if not hasattr(term, "noise"):
                    continue
                term.noise = (_U(n_min=-real[name], n_max=real[name])
                              if name in real else None)
        ecfg.seed = self.mj.seed
        ecfg.scene.num_envs = self.N
        ecfg.scene.env_spacing = float(cfg.env_spacing)
        ecfg.episode_length_s = self.mj.episode_budget_s + 120.0
        ecfg.auto_reset = False
        tw = ecfg.commands["twist"]
        tw.resampling_time_range = (1e6, 1e6)
        tw.debug_vis = False
        tw.pos_tol = self.mj.attractor_pos_tol_m
        tw.yaw_tol = self.mj.attractor_yaw_tol_rad
        if self.mj.wp_yaw_stiffness is not None:
            tw.yaw_stiffness = float(self.mj.wp_yaw_stiffness)
        for side in HANDS:
            t = ecfg.commands[f"{side}_ee_cmd"]
            t.resampling_time_range = (1e6, 1e6)
            t.debug_vis = False
        ecfg.commands["loco_mpc"].debug_vis = False

        self.obstacles = build_obstacles_v3(scene.spec)
        _inject_force_family(ecfg, self.obstacles)

        agent = load_rl_cfg(self.mj.task)
        runner_cls = load_runner_cls(self.mj.task)
        self.env = getattr(runner_cls, "env_cls")(cfg=ecfg, device=self.device,
                                                  render_mode=None)
        self.wenv = getattr(runner_cls, "vec_wrapper_cls")(
            self.env, clip_actions=agent.clip_actions)
        runner = runner_cls(self.wenv, dataclasses.asdict(agent),
                            device=self.device)
        runner.load(str(resolve_checkpoint(self.mj.checkpoint)),
                    load_cfg={"actor": True}, strict=True,
                    map_location=self.device)
        self.policy = runner.get_inference_policy(device=self.device)

        self.obs, _ = self.wenv.reset()
        self.fsm = self.env.command_manager.get_term("twist")
        self.robot = self.env.scene["robot"]
        self.wall_map = self.fsm.wall_map
        self.hands = {n: self.env.command_manager.get_term(f"{n}_ee_cmd")
                      for n in HANDS}
        self.env_ids = torch.arange(self.N, device=self.device,
                                    dtype=torch.long)
        self.dt = float(self.env.step_dt)
        self._t = 0.0
        self._tick = 0

        # ── per-env map frame, seeded from each env's own spawn ─────────────
        p = self.robot.data.root_link_pos_w[:, :2].detach().cpu().numpy().copy()
        yaw = self.robot.data.heading_w.detach().cpu().numpy().copy()
        self.base0 = p.astype(float)                      # (N, 2)
        self.yaw0 = yaw.astype(float)                     # (N,)
        self._axes_from_yaw()

        # the maze walls the reset randomized into the arena
        self.wall_map.active[:] = False
        self.fsm._write_wall_poses(self.env_ids)

        self._carry_anchor = {
            s: torch.tensor(v, device=self.device, dtype=torch.float32)
            for s, v in CARRY_ANCHOR_Q.items()}
        self.park_tip = {
            "left": np.array([self.mj.park_x_m, self.mj.park_y_m,
                              self.mj.park_z_m]),
            "right": np.array([self.mj.park_x_m, -self.mj.park_y_m,
                               self.mj.park_z_m])}
        for name, term in self.hands.items():
            term.contact_mask[:] = 0.0
            term.force_mag[:] = 0.0
            term._force_cmd_mag[:] = 0.0
            term._contact_age[:] = 0.0
            term.phantom[:] = 0.0
            term.time_left[:] = 1e6
            term._on_wall[:] = False
            self.set_tip(None, name, np.tile(self.park_tip[name], (self.N, 1)))

        self.install_world()
        self._sens: dict = {}
        self.peak_force_n = 0.0
        self.true_peak_force_n = 0.0
        self.force_aborts = 0
        self.ticks_over_limit = 0
        self.ticks_over_guard = 0
        self._obst_sensors = self._find_obstacle_sensors()
        self.mode = Mode.LOCOMOTION
        self.fsm._update_command()
        self.obs = self.wenv.get_observations()

    # ── frames ───────────────────────────────────────────────────────────────

    def _axes_from_yaw(self) -> None:
        self.v = np.column_stack([np.cos(self.yaw0), np.sin(self.yaw0)])
        self.u = np.column_stack([self.v[:, 1], -self.v[:, 0]])

    def place(self, env_ids, xy_map, yaw_map) -> None:
        """Redefine each env's map frame so `(xy_map, yaw_map)` IS where its
        robot stands.  The robot is not moved; the room is.

        Every wall and obstacle is then re-written for those envs.  Callers
        validate the pose geometrically first -- nothing here checks whether a
        wall is about to land on a robot."""
        e = np.asarray(env_ids, dtype=int)
        xy = np.asarray(xy_map, float).reshape(-1, 2)
        yaw = np.asarray(yaw_map, float).reshape(-1)
        xy_w, yaw_w = self.truth_pose()
        self.yaw0[e] = wrap_angle(yaw_w[e] - (yaw - math.pi / 2.0))
        self._axes_from_yaw()
        self.base0[e] = xy_w[e] - (xy[:, :1] * self.u[e] + xy[:, 1:] * self.v[e])
        self.install_world()

    def map_to_world(self, env_ids, xy_map) -> np.ndarray:
        e = np.asarray(env_ids, dtype=int)
        p = np.asarray(xy_map, float).reshape(-1, 2)
        return self.base0[e] + p[:, :1] * self.u[e] + p[:, 1:] * self.v[e]

    def world_to_map(self, env_ids, xy_w) -> np.ndarray:
        e = np.asarray(env_ids, dtype=int)
        d = np.asarray(xy_w, float).reshape(-1, 2) - self.base0[e]
        return np.column_stack([np.einsum("ij,ij->i", d, self.u[e]),
                                np.einsum("ij,ij->i", d, self.v[e])])

    def map_yaw_to_world(self, env_ids, yaw_map) -> np.ndarray:
        e = np.asarray(env_ids, dtype=int)
        return wrap_angle(self.yaw0[e] + np.asarray(yaw_map, float)
                          - math.pi / 2.0)

    def world_yaw_to_map(self, env_ids, yaw_w) -> np.ndarray:
        e = np.asarray(env_ids, dtype=int)
        return wrap_angle(np.asarray(yaw_w, float) - self.yaw0[e]
                          + math.pi / 2.0)

    # ── the room, per env ────────────────────────────────────────────────────

    def install_world(self) -> None:
        """`VForceBackend._install_world`, batched over every environment."""
        torch = self.torch
        delta = float(getattr(self.scene, "true_delta_m", 0.0) or 0.0)
        walls = [o for o in self.obstacles if o.kind == "wall"]
        others = [o for o in self.obstacles if o.kind != "wall"]
        wm = self.wall_map
        dev, cdt = wm.c.device, wm.c.dtype
        Nw = len(walls)
        hl = torch.tensor([o.size[0] for o in walls], device=dev, dtype=cdt)
        c = torch.zeros((self.N, Nw, 2), device=dev, dtype=cdt)
        th = torch.zeros((self.N, Nw), device=dev, dtype=cdt)
        for i, o in enumerate(walls):
            cm = np.asarray(o.center_map, float) + np.array([delta, 0.0])
            cw = self.base0 + cm[0] * self.u + cm[1] * self.v          # (N, 2)
            d_map = np.array([math.cos(o.yaw_map), math.sin(o.yaw_map)])
            d_w = d_map[0] * self.u + d_map[1] * self.v
            c[:, i] = torch.as_tensor(cw, device=dev, dtype=cdt)
            th[:, i] = torch.as_tensor(np.arctan2(d_w[:, 1], d_w[:, 0]),
                                       device=dev, dtype=cdt)
        wm.N, wm.hl, wm.c, wm.th = Nw, hl, c, th
        wm.active = torch.ones((self.N, Nw), device=dev, dtype=wm.active.dtype)
        self.fsm._walls = [self.env.scene[o.name] for o in walls]
        park = torch.zeros((Nw, self.N, 3), device=dev, dtype=cdt)
        for i, o in enumerate(walls):
            park[i, :] = torch.tensor([2.0 * o.k - 10.0, 9.0, 25.0 + 0.5 * o.k],
                                      device=dev, dtype=cdt)
        self.fsm._park_w = park
        self.fsm._write_wall_poses(self.env_ids)
        # THE FSM'S STORED WALL INDEX MUST NOT OUTLIVE THE MAZE.
        # `maze_twist` keeps a per-env `follow_wall` (long) chosen at reset
        # from the SIX maze walls, and `face_standoff` indexes `wall_map.hl`
        # with it every tick to report a metric.  Room 2 installs four walls,
        # so any env whose reset drew 4 or 5 indexes out of bounds -- a
        # device-side assert on the first step, and one that only appears once
        # enough environments exist for some of them to have drawn a high
        # index (measured: fine at 8 envs, dead at 32).  Nothing here follows
        # a wall, so 0 is as meaningful as any other value and it is in range.
        if hasattr(self.fsm, "follow_wall"):
            self.fsm.follow_wall[:] = 0
        for o in others:
            cm = np.asarray(o.center_map, float) + np.array([delta, 0.0])
            cw = self.base0 + cm[0] * self.u + cm[1] * self.v
            if o.kind == "cabinet":
                d_map = np.array([math.cos(o.yaw_map), math.sin(o.yaw_map)])
                d_w = d_map[0] * self.u + d_map[1] * self.v
                ang = np.arctan2(d_w[:, 1], d_w[:, 0])
            else:
                ang = np.zeros(self.N)
            pose = np.column_stack([
                cw, np.full(self.N, o.half_height + 0.02),
                np.cos(ang / 2.0), np.zeros(self.N), np.zeros(self.N),
                np.sin(ang / 2.0)])
            self.env.scene[o.name].write_mocap_pose_to_sim(
                self.torch.as_tensor(pose, device=self.env.device,
                                     dtype=self.torch.float32),
                env_ids=self.env_ids)

    # ── commands ─────────────────────────────────────────────────────────────

    def set_tip(self, env_ids, hand: str, pos_b) -> None:
        """Write a body-frame tip target, with the arm posture that reaches it.

        `anchor_q` is the policy's arm-joint mimic reference; writing a tip
        without it tells the policy to hold the arms DOWN and put the tip
        0.33 m forward at the same time, and it fights itself."""
        torch = self.torch
        term = self.hands[hand]
        p = torch.as_tensor(np.asarray(pos_b, float).reshape(-1, 3),
                            device=self.device, dtype=torch.float32)
        idx = (self.env_ids if env_ids is None
               else torch.as_tensor(np.asarray(env_ids, dtype=np.int64),
                                    device=self.device))
        q = torch.zeros((len(idx), 4), device=self.device)
        q[:, 0] = 1.0
        term._goal_pos[idx] = p
        term._goal_quat[idx] = q
        term._start_pos[idx] = p
        term._start_quat[idx] = q
        term.pose_command_b[idx, :3] = p
        term.pose_command_b[idx, 3:] = q
        aq = self._carry_anchor[hand]
        term.anchor_q[idx] = aq
        term._anchor_start_q[idx] = aq
        term._anchor_goal_q[idx] = aq

    def park(self, env_ids=None, hand: str | None = None) -> None:
        for h in ([hand] if hand else HANDS):
            n = self.N if env_ids is None else len(np.asarray(env_ids))
            term = self.hands[h]
            idx = (slice(None) if env_ids is None
                   else self.torch.as_tensor(np.asarray(env_ids, np.int64),
                                             device=self.device))
            term.contact_mask[idx] = 0.0
            term.force_mag[idx] = 0.0
            term._on_wall[idx] = False
            self.set_tip(env_ids, h, np.tile(self.park_tip[h], (n, 1)))

    def press_intent(self, env_ids, hand: str, on: bool) -> None:
        """The deployed contact intent -- `touch_primitive.intent_value`."""
        term = self.hands[hand]
        idx = (slice(None) if env_ids is None
               else self.torch.as_tensor(np.asarray(env_ids, np.int64),
                                         device=self.device))
        term._on_wall[idx] = bool(on)
        term.contact_mask[idx] = 1.0 if on else 0.0
        term.force_mag[idx] = float(self.mj.vf_press_intent_n) if on else 0.0

    def hold(self, xy_w, yaw_w, *, speed: float | None = None) -> None:
        """Freeze every base at its own hold pose.  No walking, ever."""
        torch = self.torch
        self.fsm.pose_target_w[:, :2] = torch.as_tensor(
            np.asarray(xy_w, float).reshape(self.N, 2), device=self.device,
            dtype=torch.float32)
        self.fsm.pose_target_w[:, 2] = torch.as_tensor(
            wrap_angle(np.asarray(yaw_w, float).reshape(self.N)),
            device=self.device, dtype=torch.float32)
        self.fsm.speed_scale[:] = float(
            self.mj.speed_scale_locomotion if speed is None else speed)
        self.fsm.arrived[:] = True
        if hasattr(self.fsm, "is_standing_env"):
            self.fsm.is_standing_env[:] = True
        self.fsm.time_left[:] = 1e6

    # ── stepping and reading ─────────────────────────────────────────────────

    def step(self) -> None:
        torch = self.torch
        with torch.no_grad():
            self.obs, _, _, _ = self.wenv.step(self.policy(self.obs))
        self._t += self.dt
        self._tick += 1

    @property
    def time_s(self) -> float:
        return self._t

    def truth_pose(self):
        d = self.robot.data
        return (d.root_link_pos_w[:, :2].detach().cpu().numpy().copy(),
                d.heading_w.detach().cpu().numpy().copy())

    def base_height(self) -> np.ndarray:
        return self.robot.data.root_link_pos_w[:, 2].detach().cpu().numpy().copy()

    def qpos_all(self) -> np.ndarray:
        """`(N, nq)` full generalized coordinates -- one row per world.

        The single-world backend reads `qpos[0]`; this is the same array, and
        `qpos[ei]` is the same access.  (An earlier version of the batched
        collector returned None here with a comment claiming qpos was "not
        per-env in the batch API".  That was wrong: mujoco-warp batches the
        data, so the leading axis IS the world.)"""
        try:
            return self.env.sim.data.qpos.detach().cpu().numpy().copy()
        except Exception:                       # noqa: BLE001 - optional
            return None

    def joint_state(self):
        d = self.robot.data
        return (d.joint_pos.detach().cpu().numpy().copy(),
                d.joint_vel.detach().cpu().numpy().copy())

    def base_vel(self):
        d = self.robot.data
        return (d.root_link_lin_vel_w.detach().cpu().numpy().copy(),
                d.root_link_ang_vel_w.detach().cpu().numpy().copy())

    def body_speed(self) -> np.ndarray:
        return np.linalg.norm(
            self.robot.data.root_link_lin_vel_w[:, :2].detach().cpu().numpy(),
            axis=1)

    def yaw_rate(self) -> np.ndarray:
        return self.robot.data.root_link_ang_vel_w[:, 2].detach().cpu().numpy()

    def ee_world(self, hand: str) -> np.ndarray:
        return (self.hands[hand].measured_ee_pose_w()[0]
                .detach().cpu().numpy().copy())

    def tip_body(self, hand: str) -> np.ndarray:
        """Measured tip in each TRUE body frame -- pelvis FK, no leak."""
        p_w = self.ee_world(hand)
        xy, yaw = self.truth_pose()
        z = self.base_height()
        c, s = np.cos(-yaw), np.sin(-yaw)
        d = p_w[:, :2] - xy
        return np.column_stack([c * d[:, 0] - s * d[:, 1],
                                s * d[:, 0] + c * d[:, 1],
                                p_w[:, 2] - z])

    def force_w(self, hand: str) -> np.ndarray:
        return (self.hands[hand].measured_force_w()
                .detach().cpu().numpy().copy())

    def true_axis_force(self, hand: str) -> np.ndarray:
        term = self.hands[hand]
        f = term.measured_force_w()
        a = term.press_dir_w
        return np.abs((f * a).sum(dim=1).detach().cpu().numpy())

    def true_total_force(self, hand: str) -> np.ndarray:
        return np.linalg.norm(self.force_w(hand), axis=1)

    def sensed(self, true_n: np.ndarray, key: str) -> np.ndarray:
        """The SensX model, batched: saturate, corrupt, hold at 100 Hz, and
        deliver late.  One clock for every env, so the hold test is shared and
        only the values are per-env."""
        mj = self.mj
        st = self._sens.setdefault(key, {"t": None, "ring": []})
        if st["t"] is None or self._t - st["t"] >= 1.0 / mj.force_rate_hz:
            v = np.minimum(np.asarray(true_n, float), mj.force_sat_n)
            v = np.maximum(0.0, v + self.rng.standard_normal(self.N)
                           * mj.force_sigma_n)
            st["ring"].append((self._t, v))
            st["t"] = self._t
            if len(st["ring"]) > 64:
                del st["ring"][:-16]
        lag = mj.plan_force_latency_s
        if lag <= 0.0 or not st["ring"]:
            return st["ring"][-1][1].copy()
        t_read = self._t - lag
        out = st["ring"][0][1]
        for ts, v in st["ring"]:
            if ts > t_read:
                break
            out = v
        return out.copy()

    def probe_force(self, hand: str) -> np.ndarray:
        true_n = self.true_axis_force(hand)
        self.true_peak_force_n = max(self.true_peak_force_n,
                                     float(true_n.max()))
        f = self.sensed(true_n, f"probe_{hand}")
        self.peak_force_n = max(self.peak_force_n, float(f.max()))
        return f

    def hand_force(self, hand: str) -> np.ndarray:
        return self.sensed(self.true_total_force(hand), f"total_{hand}")

    def measured_normal_body(self, hand: str):
        """Touched face's outward normal from the MEASURED contact force.

        The vforce wrist tracks its command stiffly, so an encoder normal is
        the commanded axis in disguise; the per-wall netforce vector is what
        this family actually measures.  NaN where the force is too small to
        direct (< 3 N), which is how the demo reports it too."""
        f = self.force_w(hand)
        L = np.hypot(f[:, 0], f[:, 1])
        _xy, yaw = self.truth_pose()
        c, s = np.cos(-yaw), np.sin(-yaw)
        n = -np.column_stack([c * f[:, 0] - s * f[:, 1],
                              s * f[:, 0] + c * f[:, 1]])
        with np.errstate(invalid="ignore", divide="ignore"):
            n = n / L[:, None]
        n[L < 3.0] = np.nan
        return n

    def _find_obstacle_sensors(self) -> dict:
        """`{hand: [(k, sensor), ...]}` -- the real (hand, obstacle) pairs."""
        out = {}
        for hand, prefix in (("left", "lhand"), ("right", "rhand")):
            found = []
            for k in range(len(self.obstacles)):
                # `_inject_force_family` names them `..._vfobst{k}_...`; the
                # V2 injector uses `v2obst`, and a wrong infix does not raise
                # -- the scan just stops at k = 0 and reports "nothing is
                # touching", indistinguishable from a real miss.
                nm = f"{prefix}_vfobst{k}_contact"
                try:
                    found.append((k, self.env.scene[nm]))
                except (KeyError, TypeError):
                    break
            out[hand] = found
        return out

    def obstacle_contacts(self, hand: str) -> np.ndarray:
        """`(N,)` index of the obstacle each env's hand is touching, or -1."""
        best = np.full(self.N, -1, dtype=int)
        best_f = np.zeros(self.N)
        for k, sen in self._obst_sensors.get(hand, ()):
            found = getattr(sen.data, "found", None)
            if found is None:
                continue
            hit = found.reshape(self.N, -1)[:, 0].detach().cpu().numpy() > 0
            force = getattr(sen.data, "force", None)
            f = (np.linalg.norm(force.reshape(self.N, -1)[:, :3]
                                .detach().cpu().numpy(), axis=1)
                 if force is not None else np.ones(self.N))
            take = hit & (f >= best_f)
            best[take] = k
            best_f[take] = f[take]
        return best

    def physical_summary(self) -> dict:
        """The safety numbers, the way `VForceBackend` reports them.

        `peak_force_n` is what the SENSOR reported and SensX clips at 20 N, so
        the true physical peak is kept beside it -- judging "no sample over
        25 N" on a clipped channel would be vacuous."""
        return {
            "environment_instances": self.N,
            "controller": "vforce_decoupled",
            "checkpoint": str(self.mj.checkpoint),
            "peak_force_n": self.peak_force_n,
            "true_peak_force_n": self.true_peak_force_n,
            "force_limit_n": self.mj.force_limit_n,
            "guard_abort_force_n": self.mj.guard_abort_force_n,
            "force_aborts": self.force_aborts,
            "over_limit_env_ticks": self.ticks_over_limit,
            "over_guard_env_ticks": self.ticks_over_guard,
            "ticks": self._tick,
            "sim_seconds": round(self._t, 1),
        }

    def close(self) -> None:
        try:
            self.env.close()
        except Exception:                       # noqa: BLE001 - shutdown only
            pass
