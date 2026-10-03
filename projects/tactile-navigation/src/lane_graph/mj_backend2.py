"""MujocoBackendV2: the V1 physical backend driving V2 in-room scenes.

Reuses everything measured and frozen in V1 (press shield, force guard,
safety stop, odometry contract, recorder) and changes exactly three things:

1. `_customize_env_cfg` injects the V2 obstacle entities + contact sensors
   (mj_assets) before the env is built;
2. after the one reset, WALL segments are written into a WIDENED FSM wall
   map (so the kiss plane -- press validity's reference -- stays correct
   over arbitrary segments) and the FSM places them; pillars/cabinets are
   placed directly by mocap write and deliberately stay OUT of the wall
   map (kiss then reads the wall behind them: the validated proud-face
   geometry);
3. probe-side selection is frame-driven (`set_press_frame`), not
   leg-indexed -- the V2 executive states the travel/facing frame before
   each press.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np

from execution.mujoco_backend import MujocoBackend, MujocoConfig  # noqa: F401

from .mj_assets import (build_obstacles, build_obstacles_v3,
                        inject_into_env_cfg, place_obstacles)
from .scene2 import V2Scene


class MujocoBackendV2(MujocoBackend):
    def __init__(self, scene: V2Scene, cfg, rng, mj, *, render=False,
                 video_path=None, view: bool = False):
        self._v2_obstacles = (build_obstacles_v3(scene.spec)
                              if hasattr(scene.spec, "bulges")
                              and hasattr(scene.spec, "prots")
                              else build_obstacles(scene.spec))
        super().__init__(scene, cfg, rng, mj, render=render,
                         video_path=video_path, view=view)
        self._install_v2_world()

    def _customize_env_cfg(self, env_cfg) -> None:
        inject_into_env_cfg(env_cfg, self._v2_obstacles)

    # ── world installation (post-reset: base0/u/v are known) ────────────────────

    def _install_v2_world(self) -> None:
        import torch

        # REGISTRATION OFFSET (sim2real): place the PHYSICAL room shifted by
        # `scene.true_delta_m` along map +x relative to where the belief's map
        # says it is.  The kinematic backend has honoured this field forever
        # (backends.py); the MuJoCo path silently ignored it, which meant the
        # robot always spawned EXACTLY where its prior said -- the one mercy a
        # real deployment never grants.
        delta = float(getattr(self.scene, "true_delta_m", 0.0) or 0.0)
        walls = [o for o in self._v2_obstacles if o.kind == "wall"]
        others = [o for o in self._v2_obstacles if o.kind != "wall"]
        wm = self.wall_map
        B = wm.c.shape[0]
        N = len(walls)
        dev, cdt = wm.c.device, wm.c.dtype
        hl = torch.tensor([o.size[0] for o in walls], device=dev, dtype=cdt)
        c = torch.zeros((B, N, 2), device=dev, dtype=cdt)
        th = torch.zeros((B, N), device=dev, dtype=cdt)
        active = torch.ones((B, N), device=dev, dtype=wm.active.dtype)
        for i, o in enumerate(walls):
            cm = np.asarray(o.center_map, float) + np.array([delta, 0.0])
            cw = self.base0 + cm[0] * self.u + cm[1] * self.v
            d_map = np.array([math.cos(o.yaw_map), math.sin(o.yaw_map)])
            d_w = d_map[0] * self.u + d_map[1] * self.v
            c[0, i] = torch.tensor(cw, device=dev, dtype=cdt)
            th[0, i] = float(math.atan2(d_w[1], d_w[0]))
        wm.N, wm.hl, wm.c, wm.th, wm.active = N, hl, c, th, active
        ents = [self.env.scene[o.name] for o in walls]
        self.fsm._walls = ents
        park = torch.zeros((N, B, 3), device=dev, dtype=cdt)
        for i, o in enumerate(walls):
            park[i, 0] = torch.tensor(
                [2.0 * o.k - 10.0, 9.0, 25.0 + 0.5 * o.k],
                device=dev, dtype=cdt)
        self.fsm._park_w = park
        self.fsm._write_wall_poses(self.env_ids)
        others_d = [dataclasses.replace(
                        o, center_map=(float(o.center_map[0]) + delta,
                                       float(o.center_map[1])))
                    for o in others] if delta else others
        place_obstacles(self.env, self.env_ids, others_d,
                        self.base0, self.u, self.v)
        self._wall_world = [(np.asarray(self.base0
                                        + (np.asarray(o.center_map)[0] + delta) * self.u
                                        + np.asarray(o.center_map)[1] * self.v),
                             float(th[0, i]), float(o.size[0]))
                            for i, o in enumerate(walls)]
        self.leg_wall_idx = {0: 0}
        self._update_followed_wall()

    def _update_followed_wall(self) -> None:
        """The FSM's followed-wall slot = the segment nearest a probe point
        half a metre ahead of the base along the current facing.  V1 mapped
        legs to wall slots; V2 has no legs, and the slot only truly matters
        while pressing (it anchors the kiss plane)."""
        base = self.robot.data.root_link_pos_w[0, :2].detach().cpu().numpy()
        yaw = float(self.robot.data.heading_w[0])
        probe = base + 0.5 * np.array([math.cos(yaw), math.sin(yaw)])
        best, best_d = 0, float("inf")
        for i, (c, th_i, hl) in enumerate(self._wall_world):
            d_dir = np.array([math.cos(th_i), math.sin(th_i)])
            rel = probe - c
            along = float(np.clip(rel @ d_dir, -hl, hl))
            dist = float(np.linalg.norm(rel - along * d_dir))
            if dist < best_d:
                best, best_d = i, dist
        self.leg_wall_idx = {self.leg: best}

    def set_followed_wall_map(self, point_map, normal_map) -> int:
        """Pin the FSM's followed-wall slot to a stated face.

        `_update_followed_wall`'s nearest-to-a-point-ahead heuristic is right for
        a facing press but ambiguous while travelling ALONG a wall, where the
        perpendicular wall at the far end can win.  A wall-parallel walk knows
        which face it is tracking, so it says so: the slot anchors the policy's
        own wall relation (`face_standoff`), which is what the base holds its
        standoff against.
        """
        p_w = (self.base0 + float(point_map[0]) * self.u
               + float(point_map[1]) * self.v)
        n_w = float(normal_map[0]) * self.u + float(normal_map[1]) * self.v
        best, best_score = 0, float("inf")
        for i, (c, th_i, hl) in enumerate(self._wall_world):
            d_dir = np.array([math.cos(th_i), math.sin(th_i)])
            rel = p_w - c
            along = float(np.clip(rel @ d_dir, -hl, hl))
            dist = float(np.linalg.norm(rel - along * d_dir))
            nrm = np.array([-d_dir[1], d_dir[0]])
            score = dist + 2.0 * (1.0 - abs(float(nrm @ n_w)))
            if score < best_score:
                best, best_score = i, score
        self.leg_wall_idx = {self.leg: best}
        return best

    # ── frame-driven probe side ─────────────────────────────────────────────────

    def set_press_frame(self, u_map, v_map) -> None:
        """The V2 executive states the press frame explicitly: ``u_map`` is
        the fan travel direction, ``v_map`` the facing (toward the pressed
        face), both in MAP coordinates.  Same hand-choice rule as V1's
        leg-indexed version, then the substep safety monitor re-installs on
        the chosen hand pair."""
        u_w = float(u_map[0]) * self.u + float(u_map[1]) * self.v
        v_w = float(v_map[0]) * self.u + float(v_map[1]) * self.v
        left_w = np.array([-v_w[1], v_w[0]])
        self.probe_side = 1.0 if float(left_w @ u_w) >= 0.0 else -1.0
        self.probe_name = "left" if self.probe_side > 0 else "right"
        self.park_name = "right" if self.probe_side > 0 else "left"
        self.probe_term = self.hands[self.probe_name]
        self.park_term = self.hands[self.park_name]
        self.shield.install_substep_monitor(
            self.env, {"probe": self.probe_term, "park": self.park_term},
            on_sample=self._on_substep)
        self._update_followed_wall()
