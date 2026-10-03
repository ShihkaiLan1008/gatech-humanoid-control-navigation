"""A `VForceBackend` stand-in with real geometry and no simulator.

The collector's contract with the backend is small and entirely mechanical:
place the room, freeze a stance, creep a commanded tip, read a force, read a
pose.  All of that can be answered from the map's own convex bodies, so the
whole pipeline -- proposal, coverage, execution, records, diagnostics -- runs
in a unit test with the SAME geometry the simulator would use.  What the fake
cannot test is the physics, and nothing here pretends to: force is a linear
function of how far the commanded tip has been pushed past a surface.

The frame contract is the real one: `v` is the facing axis at spawn, `u` is
`(v_y, -v_x)`, a map yaw of pi/2 is `yaw0`, and every pose conversion goes
through `base0/u/v` -- which is exactly what `FreeTouchProbe.place_base`
rewrites, so the reframing is genuinely exercised.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np

from execution.vforce_backend import VForceConfig
from global_nav.geometry import wrap_angle

FORCE_PER_M = 400.0          # N per metre of commanded penetration
AIR_N = 0.3


class _Hand:
    def __init__(self, name):
        self.name = name
        self.cmd = np.zeros(3)
        self.force_w = np.zeros(3)
        self.contact_mask = np.zeros(1)
        self.force_mag = np.zeros(1)
        self._force_cmd_mag = np.zeros(1)
        self._on_wall = np.zeros(1, dtype=bool)
        self.pose_command_b = np.zeros((1, 7))

    def measured_ee_pose_w(self):
        return ([_T(self._w)], None)

    def set_world_tip(self, w):
        self._w = np.asarray(w, float)

    def measured_force_w(self):
        return [self.force_w]




class FakeBackend:
    """Everything `FreeTouchProbe` and `records.state_record` ask for."""

    def __init__(self, scene, *, seed: int = 0, n_joints: int = 29,
                 fail_settle_every: int | None = None,
                 guard_every: int | None = None):
        self.scene = scene
        self.bodies = tuple(scene.plan.bodies)
        self.mj = VForceConfig(seed=seed, episode_budget_s=1.0e7)
        self.cfg = SimpleNamespace(contact_threshold_n=2.0)
        self.dt = 0.02
        self._t = 0.0
        self._tick = 0
        self.rng = np.random.default_rng(seed)

        self.yaw0 = math.pi / 2.0
        self.v = np.array([math.cos(self.yaw0), math.sin(self.yaw0)])
        self.u = np.array([self.v[1], -self.v[0]])
        self.base0 = np.zeros(2)
        self._true_xy = np.zeros(2)
        self._true_yaw = self.yaw0

        self.hands = {"left": _Hand("left"), "right": _Hand("right")}
        self._park_tip = {
            "left": np.array([self.mj.park_x_m, self.mj.park_y_m,
                              self.mj.park_z_m]),
            "right": np.array([self.mj.park_x_m, -self.mj.park_y_m,
                               self.mj.park_z_m])}
        for n, h in self.hands.items():
            h.cmd = self._park_tip[n].copy()
            h.set_world_tip(np.zeros(3))
        self.set_touch_hand(-1)
        self._sync_ee()

        self.trace: list = []
        self.state_log: list = []
        self.force_abort_pending = False
        self.force_aborts = 0
        self.peak_force_n = 0.0
        self.true_axis_force_n = 0.0
        self.true_peak_force_n = 0.0
        self._press_f_sum, self._press_f_n = 0.0, 0
        self._hold_bias = None
        self._est_xy = self._true_xy.copy()
        self._est_yaw = float(self._true_yaw)
        self.mode = "LOCOMOTION"
        self.install_count = 0
        self.n_joints = n_joints
        self._settle_calls = 0
        self._fail_settle_every = fail_settle_every
        self._guard_every = guard_every
        self._touches = 0
        self._vf_obstacles = tuple(
            SimpleNamespace(k=i, kind=("wall" if b.name.startswith("wall")
                                       else "pillar"), name=b.name)
            for i, b in enumerate(self.bodies))
        self.robot = SimpleNamespace(data=_RobotData(n_joints))
        self.env = SimpleNamespace(
            sim=SimpleNamespace(data=SimpleNamespace(
                qpos=_T(np.zeros((1, 7 + n_joints))))))

    # ── frames (the real implementations, copied) ────────────────────────────

    def map_point_to_world(self, xy_map):
        p = np.asarray(xy_map, float).reshape(2)
        return self.base0 + p[0] * self.u + p[1] * self.v

    def world_point_to_map(self, xy_w):
        d = np.asarray(xy_w, float).reshape(2) - self.base0
        return np.array([float(d @ self.u), float(d @ self.v)])

    def map_yaw_to_world(self, yaw_map):
        return float(wrap_angle(self.yaw0 + float(yaw_map) - math.pi / 2.0))

    def world_yaw_to_map(self, yaw_w):
        return float(wrap_angle(float(yaw_w) - self.yaw0 + math.pi / 2.0))

    def truth_pose(self):
        return self._true_xy.copy(), float(self._true_yaw)

    def working_pose(self):
        return self._est_xy.copy(), float(self._est_yaw)

    @property
    def time_s(self):
        return self._t

    def reset_estimator_to(self, xy_map, yaw_map):
        self._est_xy = np.asarray(self.map_point_to_world(xy_map), float)
        self._est_yaw = float(self.map_yaw_to_world(float(yaw_map)))
        self._odom_mark = (self._est_xy.copy(), self._est_yaw)

    def odometry_increment(self):
        """Body-frame displacement since the last mark.  The base is held, so
        this is the millimetres the estimator drifted -- the same quantity the
        real backend hands the filter's motion model."""
        prev_xy, prev_yaw = getattr(
            self, "_odom_mark", (self._est_xy.copy(), self._est_yaw))
        d_w = self._est_xy - np.asarray(prev_xy, float)
        c, s = math.cos(prev_yaw), math.sin(prev_yaw)
        d_body = np.array([c * d_w[0] + s * d_w[1], -s * d_w[0] + c * d_w[1]])
        d_yaw = float(wrap_angle(self._est_yaw - float(prev_yaw)))
        self._odom_mark = (self._est_xy.copy(), self._est_yaw)
        return tuple(d_body), d_yaw

    def _install_world(self):
        self.install_count += 1

    # ── hands ────────────────────────────────────────────────────────────────

    def set_touch_hand(self, side: int) -> str:
        self.probe_side = 1.0 if side > 0 else -1.0
        self.probe_name = "left" if side > 0 else "right"
        self.park_name = "right" if side > 0 else "left"
        self.probe_term = self.hands[self.probe_name]
        self.park_term = self.hands[self.park_name]
        return self.probe_name

    def _jump_hand(self, term, pos_b, dyaw: float = 0.0):
        term.cmd = np.asarray(pos_b, float).reshape(3).copy()
        term.pose_command_b[0, :3] = term.cmd

    def _park_hand(self, term):
        self._jump_hand(term, self._park_tip[term.name])

    def park_hands(self):
        for h in self.hands.values():
            self._park_hand(h)

    def _set_goal(self, xy_w, yaw_w, speed, *, hold, standing):
        pass

    def _enter(self, mode):
        self.mode = mode

    # ── the "physics" ────────────────────────────────────────────────────────

    def _tip_map(self, term) -> np.ndarray:
        xy_map = self.world_point_to_map(self._true_xy)
        yaw = self.world_yaw_to_map(self._true_yaw)
        c, s = math.cos(yaw), math.sin(yaw)
        return xy_map + np.array([c * term.cmd[0] - s * term.cmd[1],
                                  s * term.cmd[0] + c * term.cmd[1]])

    def _penetration(self, term):
        p = self._tip_map(term).reshape(1, 2)
        best = (0.0, None, None)
        for i, b in enumerate(self.bodies):
            d = float(b.signed_distance(p)[0][0])
            if d < 0.0 and -d > best[0]:
                n = np.asarray(b.normals, float)
                v = np.asarray(b.verts, float)
                k = int(np.argmax([float(nn @ (p[0] - vv))
                                   for nn, vv in zip(n, v)]))
                best = (-d, i, n[k])
        return best

    def _sync_ee(self):
        xy, yaw = self.truth_pose()
        c, s = math.cos(yaw), math.sin(yaw)
        for h in self.hands.values():
            h.set_world_tip([xy[0] + c * h.cmd[0] - s * h.cmd[1],
                             xy[1] + s * h.cmd[0] + c * h.cmd[1],
                             0.74 + h.cmd[2]])

    def _step(self):
        self._t += self.dt
        self._tick += 1
        self._sync_ee()
        for name, h in self.hands.items():
            depth, _i, _n = self._penetration(h)
            f = FORCE_PER_M * depth
            h.force_w = np.array([0.0, 0.0, 0.0]) if depth <= 0 else \
                np.array([f, 0.0, 0.0])
        if (self._guard_every and self._touches
                and self._touches % self._guard_every == 0
                and self._tick % 17 == 0):
            self.force_abort_pending = True
        self.trace.append({"t": self._t})
        self.state_log.append((self._t, np.zeros(4, np.float32)))

    def _guard_tripped(self) -> bool:
        if not self.force_abort_pending:
            return False
        self.force_abort_pending = False
        self.force_aborts += 1
        return True

    def _probe_force_n(self) -> float:
        depth, _i, _n = self._penetration(self.probe_term)
        true_n = FORCE_PER_M * depth
        self.true_axis_force_n = true_n
        self.true_peak_force_n = max(self.true_peak_force_n, true_n)
        v = min(true_n, self.mj.force_sat_n)
        if v <= 0.0:
            v = AIR_N
        self.peak_force_n = max(self.peak_force_n, v)
        return float(v)

    def _tip_body(self, term) -> np.ndarray:
        return np.asarray(term.cmd[:2], float)

    def _measured_normal_body(self, term):
        depth, _i, n = self._penetration(term)
        if depth <= 0.0 or n is None:
            return None
        yaw = self.world_yaw_to_map(self._true_yaw)
        c, s = math.cos(-yaw), math.sin(-yaw)
        b = np.array([c * n[0] - s * n[1], s * n[0] + c * n[1]])
        return (float(b[0]), float(b[1]))

    def hand_obstacle_contacts(self, side: str) -> list:
        depth, i, _n = self._penetration(self.hands[side])
        if depth <= 0.0 or i is None:
            return []
        return [{"obstacle_k": i, "sensor": f"{side}_v2obst{i}_contact",
                 "force_n": FORCE_PER_M * depth}]

    def _retract(self):
        self._park_hand(self.probe_term)
        for _ in range(3):
            self._step()
        self._touches += 1

    def _settle(self, hold_w, yaw_w, cap_s: float) -> bool:
        self._settle_calls += 1
        for _ in range(3):
            self._step()
        if (self._fail_settle_every
                and self._settle_calls % self._fail_settle_every == 0):
            return False
        return True

    def physical_summary(self) -> dict:
        return {"peak_force_n": self.peak_force_n,
                "force_aborts": self.force_aborts}

    def close(self):
        pass


class _RobotData:
    def __init__(self, n):
        self.joint_pos = _T(np.linspace(-0.3, 0.3, n)[None, :])
        self.joint_vel = _T(np.zeros((1, n)))
        self.root_link_pos_w = _T(np.array([[0.0, 0.0, 0.74]]))
        self.root_link_lin_vel_w = _T(np.zeros((1, 3)))
        self.root_link_ang_vel_w = _T(np.zeros((1, 3)))


class _T:
    """The `.detach().cpu().numpy()` chain, on plain numpy."""

    def __init__(self, a):
        self._a = np.asarray(a, float)

    def __getitem__(self, k):
        v = self._a[k]
        return _T(v) if isinstance(v, np.ndarray) and v.ndim else float(v)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._a

    def __float__(self):
        return float(self._a)
