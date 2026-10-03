"""A kinematic punch controller: the same `PunchController` interface the
MuJoCo stacks honour, with no physics.

For iterating on the planners.  The base is a point that servos to its
commanded target at a constant speed and turn rate; each fist is a point
that lags its commanded body-frame target and is clamped at the first map
surface along its motion, reading a force proportional to how far the
command drives it into that surface.  Odometry is dead-reckoned with the
same per-metre drift constants the belief's motion model uses, and the
IMU channel is a gyro random walk per radian turned, as in `MujocoBackend`.

What it does NOT model: arm reach limits and IK (any commanded tip is
reached), body collisions with walls (the planners' clearance constraint
is what keeps the base off them), foot placement, falls, and force
transients.  A run here says whether the planning and estimation logic
holds together, not whether the robot can do it.

World frame == map frame: ``base0 = 0``, ``u = x``, ``v = y``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from global_nav.geometry import raycast, wrap_angle
from global_nav.models import nearest_surface

from .punch_actions import PunchAction


@dataclass(frozen=True)
class KinematicConfig:
    dt: float = 0.02
    base_speed_m_s: float = 0.25
    base_turn_rad_s: float = 1.0
    arrive_tol_m: float = 0.02
    arrive_tol_rad: float = 0.02
    hand_tau_s: float = 0.06
    """First-order lag of the fist behind its commanded tip."""
    dome_r_m: float = 0.0
    """Where the reported tip stops short of the surface.  The filter scores
    the reported tip AS the contact point (MuJoCo reports the tip site, which
    is the pad), so a nonzero value here is a systematic 'contact 4 cm off
    the wall' that every touch would push the belief with."""
    force_n_per_m: float = 400.0
    """Reading per metre the command pushes past the surface, clipped."""
    force_clip_n: float = 20.0
    force_noise_n: float = 0.3
    odom_sigma_along_per_m: float = 0.104
    odom_sigma_cross_per_m: float = 0.104
    odom_sigma_yaw_per_rad: float = 0.060
    odom_sigma_yaw_per_m: float = 0.020
    imu_yaw_sigma_per_rad: float = 0.01
    # ── the deploy stack's measured disturbances (all off by default) ──────
    forward_deliver_frac: float = 1.0
    """Gain on the plant's commanded forward velocity. With accurate
    odometry the servo compensates by moving longer; an odometry gain
    error can still make it stop short."""
    odom_forward_gain: float = 1.0
    """Odometry reports this multiple of the TRUE body-forward increment.
    Leg odometry on the deploy stack reports about the command, i.e.
    ``1 / forward_deliver_frac``."""
    press_slide_m_s: float = 0.0
    """While a fist is loaded the base creeps along body-forward at this
    rate, unseen by odometry (deploy: 3-8 cm toward the wall per punch
    hold)."""


class KinematicPunchController:
    """See the module docstring."""

    def __init__(self, scene, *, seed: int, face_yaw_map: float,
                 cfg: KinematicConfig = KinematicConfig(), **_ignored):
        self.scene = scene
        self.plan = scene.plan
        self.bodies = tuple(scene.plan.bodies)
        self.cfg = cfg
        self.dt = cfg.dt
        self.rng = np.random.default_rng(seed + 4242)
        self.state_log: list = []               # no qpos: no video
        self.base0 = np.zeros(2)
        self.u, self.v = np.array([1.0, 0.0]), np.array([0.0, 1.0])
        self.yaw0 = float(face_yaw_map)
        self.arrive_tol_m = cfg.arrive_tol_m
        self._t = 0.0
        self._xy = np.asarray(scene.start_map_xy, float).copy()
        self._yaw = float(face_yaw_map)
        self._target_xy = self._xy.copy()
        self._target_yaw = self._yaw
        self._rel_id = None
        self._tip_b = {"left": np.array([0.30, 0.15, 0.18]), "right": np.array([0.30, -0.15, 0.18])}
        self._force = {"left": 0.0, "right": 0.0}
        self._hit_k = {"left": None, "right": None}
        # onboard estimator (drifts) and its odometry mark
        self._est_xy, self._est_yaw = self._xy.copy(), self._yaw
        self._odom_mark = (self._est_xy.copy(), self._est_yaw)
        self._imu_yaw = self._yaw
        self.true_peak_force_n = 0.0

    # ── PunchController ────────────────────────────────────────────────────

    @property
    def time_s(self) -> float:
        return self._t

    def truth_pose(self):
        return self._xy.copy(), float(self._yaw)

    def map_pose(self):
        return self.truth_pose()

    def working_pose(self):
        return self._est_xy.copy(), float(self._est_yaw)

    def moving(self) -> bool:
        return (float(np.linalg.norm(self._target_xy - self._est_xy)) > self.cfg.arrive_tol_m
                or abs(wrap_angle(self._target_yaw - self._est_yaw)) > self.cfg.arrive_tol_rad)

    def apply(self, action: PunchAction) -> None:
        mv = action.move
        if mv.rel is not None:
            if self._rel_id != mv.move_id:
                c, s = math.cos(self._est_yaw), math.sin(self._est_yaw)
                f, l = float(mv.rel[0]), float(mv.rel[1])
                self._target_xy = self._est_xy + np.array([c * f - s * l, s * f + c * l])
                self._rel_id = mv.move_id
        elif mv.target_map is not None:
            self._target_xy = np.asarray(mv.target_map, float)
        else:
            self._target_xy = self._est_xy.copy()      # hold in the estimated frame
        self._target_yaw = float(mv.yaw_map)
        self._cmd = {"left": action.left, "right": action.right}

    def _base_command(self):
        """Body displacement and yaw increment from estimated feedback only."""
        cfg, dt = self.cfg, self.dt
        d = self._target_xy - self._est_xy
        dist = float(np.linalg.norm(d))
        d = (d / dist * min(dist, cfg.base_speed_m_s * dt)
             if dist > cfg.arrive_tol_m else np.zeros(2))
        c, s = math.cos(self._est_yaw), math.sin(self._est_yaw)
        body = np.array([c * d[0] + s * d[1], -s * d[0] + c * d[1]])
        e = wrap_angle(self._target_yaw - self._est_yaw)
        turn = (float(np.clip(e, -cfg.base_turn_rad_s * dt, cfg.base_turn_rad_s * dt))
                if abs(e) > cfg.arrive_tol_rad else 0.0)
        return body, turn

    def step(self) -> None:
        cfg, dt = self.cfg, self.dt
        body, turn = self._base_command()
        # Plant: execute the body command in the physical body's frame.
        # Truth is used here and for sensor synthesis, never by the servo.
        prev_xy, prev_yaw = self._xy.copy(), self._yaw
        body[0] *= cfg.forward_deliver_frac
        c, s = math.cos(self._yaw), math.sin(self._yaw)
        self._xy += np.array([c * body[0] - s * body[1], s * body[0] + c * body[1]])
        self._yaw = wrap_angle(self._yaw + turn)
        self._integrate_estimator(prev_xy, prev_yaw)
        # hands: lag toward the command, clamp at the first surface
        cmds = getattr(self, "_cmd", None)
        if cmds is not None:
            for side in ("left", "right"):
                self._hand(side, np.asarray(cmds[side].pose.pos, float))
        # a loaded fist drags the base along the press axis, unseen by the
        # odometry (the estimator integrated before this)
        if cfg.press_slide_m_s > 0.0 and max(self._force.values()) > 1.0:
            c, s = math.cos(self._yaw), math.sin(self._yaw)
            slide = np.array([c, s]) * cfg.press_slide_m_s * dt
            self._xy = self._xy + slide
        self._t += dt

    def _hand(self, side: str, target_b: np.ndarray) -> None:
        cfg = self.cfg
        tip = self._tip_b[side]
        a = min(1.0, self.dt / max(cfg.hand_tau_s, 1e-6))
        want = tip + a * (target_b - tip)
        # the fist's path in the map, this tick
        c, s = math.cos(self._yaw), math.sin(self._yaw)
        R = np.array([[c, -s], [s, c]])
        p0 = self._xy + R @ tip[:2]
        p1 = self._xy + R @ want[:2]
        seg = p1 - p0
        L = float(np.linalg.norm(seg))
        force, hit_k = 0.0, None
        # already within the dome of a surface: stay on it and read the push
        d0, n0 = nearest_surface(self.plan, p0[None, :], None)
        d0 = float(d0[0])
        if L > 1e-9:
            dist, n_hit, k = raycast(p0, seg / L, L + cfg.dome_r_m, self.bodies)
            # A hit counts only when the fist is moving INTO the face it meets.
            # A fist resting on a wall and commanded away from it starts on the
            # boundary: the ray reports an entry at distance zero, which is not
            # a press.  Without this a sideways command after a forward press
            # produced a phantom contact with the new press axis.
            # (A ray starting on or a hair inside the body reports an entry at
            # distance zero through its BACK face; judge the direction against
            # the face the fist is actually resting on, the nearest one.)
            n_ref = None if dist is None else (np.asarray(n_hit, float) if dist > 1e-6 else np.asarray(n0[0], float))
            into = n_ref is not None and float((seg / L) @ n_ref) < -1e-9
            if into and dist - cfg.dome_r_m < L:
                # clamp the fist at the dome distance from the surface
                reach = max(0.0, dist - cfg.dome_r_m)
                p_c = p0 + seg / L * reach
                want = np.array([*(R.T @ (p_c - self._xy)), want[2]])
                push = L - reach                         # how far the command drives in
                force = min(cfg.force_clip_n, cfg.force_n_per_m * push)
                hit_k = int(k)
        if force == 0.0 and d0 <= cfg.dome_r_m + 1e-3:
            # resting on a surface: the command's remaining push along the normal
            push = max(0.0, float((p1 - p0) @ (-np.asarray(n0[0], float))))
            force = min(cfg.force_clip_n, cfg.force_n_per_m * push) if push > 1e-4 else 0.0
            if force > 0.0:
                want = tip.copy()
                _d, _n, k = raycast(p0, -np.asarray(n0[0], float), cfg.dome_r_m + 0.05, self.bodies)
                hit_k = int(k) if k is not None else None
        self._tip_b[side] = want
        self._force[side] = force
        self._hit_k[side] = hit_k
        self.true_peak_force_n = max(self.true_peak_force_n, force)

    def tip_pos_b(self, side: str) -> np.ndarray:
        return self._tip_b[side].copy()

    def hand_force_n(self, side: str) -> float:
        return float(self._force[side])

    def sensed_force_n(self, side: str) -> float:
        return max(0.0, float(self._force[side]) + float(self.rng.standard_normal()) * self.cfg.force_noise_n)

    def hand_contacts(self, side: str) -> list:
        k = self._hit_k[side]
        return [] if k is None or self._force[side] <= 0.0 else [{"obstacle_k": k, "force_n": self._force[side]}]

    def fallen(self) -> bool:
        return False

    def close(self) -> None:
        pass

    # ── the onboard estimator and the IMU (what the localizer reads) ───────

    def _integrate_estimator(self, prev_xy, prev_yaw) -> None:
        k = self.cfg
        d_w = self._xy - prev_xy
        d_yaw = wrap_angle(self._yaw - prev_yaw)
        c, s = math.cos(prev_yaw), math.sin(prev_yaw)
        d_b = np.array([c * d_w[0] + s * d_w[1], -s * d_w[0] + c * d_w[1]])
        dist = float(np.linalg.norm(d_b))
        d_b = np.array([d_b[0] * k.odom_forward_gain, d_b[1]])
        if dist > 1e-9 or abs(d_yaw) > 1e-9:
            # random-walk drift per increment: sigma proportional to the increment
            d_b = d_b + self.rng.standard_normal(2) * np.array(
                [k.odom_sigma_along_per_m * dist, k.odom_sigma_cross_per_m * dist])
            d_yaw = d_yaw + float(self.rng.standard_normal()) * math.hypot(
                k.odom_sigma_yaw_per_rad * abs(d_yaw), k.odom_sigma_yaw_per_m * dist)
            self._imu_yaw = wrap_angle(self._imu_yaw + wrap_angle(self._yaw - prev_yaw)
                                       + float(self.rng.standard_normal()) * k.imu_yaw_sigma_per_rad
                                       * abs(wrap_angle(self._yaw - prev_yaw)))
        ce, se = math.cos(self._est_yaw), math.sin(self._est_yaw)
        self._est_xy = self._est_xy + np.array([ce * d_b[0] - se * d_b[1], se * d_b[0] + ce * d_b[1]])
        self._est_yaw = wrap_angle(self._est_yaw + d_yaw)

    def odometry_increment(self):
        prev_xy, prev_yaw = self._odom_mark
        d_w = self._est_xy - prev_xy
        c, s = math.cos(prev_yaw), math.sin(prev_yaw)
        d_body = np.array([c * d_w[0] + s * d_w[1], -s * d_w[0] + c * d_w[1]])
        d_yaw = float(wrap_angle(self._est_yaw - prev_yaw))
        self._odom_mark = (self._est_xy.copy(), self._est_yaw)
        return tuple(float(v) for v in d_body), d_yaw

    def heading_since_reset(self) -> float:
        return float(self._imu_yaw)

    def reset_estimator_to(self, xy_map, yaw_map: float) -> None:
        self._est_xy = np.asarray(xy_map, float).copy()
        self._est_yaw = float(yaw_map)
        self._odom_mark = (self._est_xy.copy(), self._est_yaw)
