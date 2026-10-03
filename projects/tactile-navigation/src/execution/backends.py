"""Two robots behind one interface: the real G1, and a kinematic stand-in.

WHY A STAND-IN EXISTS.  One MuJoCo episode of the shipped intermittent scene
takes 5 min 22 s of wall clock, and the Batch-3A scenes are 2-3x longer.  An
executive debugged only against the simulator would cost ten minutes per
iteration, so the mode machine, the planner coupling and the belief updates are
developed and TESTED against ``KinematicBackend`` -- same interface, same
reach limits, same contact-by-pressing semantics, no physics.  What the stand-in
cannot tell you is whether the frozen controller tracks the command, whether the
base drifts while parked, and what the contact forces are; those need the real
thing, and that is exactly what the physical runs are for.

THE ODOMETRY CONTRACT.  Both backends expose ``odometry_increment()`` -- the
robot's own estimate of how far it moved since the last call -- and, separately,
``truth_pose()``, which the evaluator reads and the belief never does.  A
relative increment with noise IS what an odometry sensor gives; an absolute pose
is what it does not.  ``LocalReport`` can carry the first and has no field for
the second.

COMMANDING FROM A BELIEF.  ``goto_map`` takes a MAP-frame target and converts it
with the BELIEVED map pose, not the true one::

    world_target = truth_xy + R(yaw_truth - yaw_believed) . (P_map - xy_believed)

Only the belief sets the displacement; truth supplies the frame the controller
lives in, exactly as a real robot's own odometry would.  So a belief that is
0.55 m wrong about where it stands on the map walks to a place 0.55 m from where
it meant to -- which is how a wrong hypothesis becomes a wrong turn.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from global_nav.geometry import raycast, wrap_angle

# MujocoBackend deliberately lives in mujoco_backend.py: importing it here
# would pull mjlab/torch into every kinematic-only run.
__all__ = ["BackendConfig", "ProbeReading", "KinematicBackend"]


@dataclass(frozen=True)
class BackendConfig:
    reach_m: float = 0.45
    contact_threshold_n: float = 2.0
    probe_force_n: float = 6.0
    force_limit_n: float = 25.0
    force_abort_n: float = 22.0
    free_sweep_margin_m: float = 0.02
    park_x_m: float = 0.14
    probe_hand_z_m: float = 0.15

    odom_sigma_along_per_m: float = 0.030
    odom_sigma_cross_per_m: float = 0.020
    odom_sigma_yaw_per_rad: float = 0.030
    odom_sigma_yaw_per_m: float = 0.010
    """Odometry noise.  The belief's motion model assumes slightly more (Batch 1
    measured that a filter assuming LESS noise than reality diverges, while the
    reverse only costs sharpness)."""

    fk_sigma_m: float = 0.012
    normal_sigma_rad: float = 0.10


@dataclass
class ProbeReading:
    """What one executed press returned, in the BODY frame."""

    contacted: bool
    p_body: tuple[float, float] | None
    normal_body: tuple[float, float] | None
    swept_free_body: tuple
    peak_force_n: float
    swept_past_face_m: float
    valid: bool = True
    reason: str = ""


# ── the fast stand-in ────────────────────────────────────────────────────────────


class KinematicBackend:
    """Perfect tracking, no dynamics; real reach limits and real ray geometry.

    The base teleports along its commanded path at a fixed speed, which is
    exactly the assumption the real one violates -- so every number this backend
    produces about TRACKING is worthless and none is reported.  What it does
    faithfully is the geometry of what the hand can and cannot touch.
    """

    is_physical = False

    def __init__(self, scene, cfg: BackendConfig, rng: np.random.Generator,
                 speed_m_s: float = 0.45, yaw_rate_rad_s: float = 0.9,
                 dt: float = 0.02):
        self.scene = scene
        self.cfg = cfg
        self.rng = rng
        self.dt = dt
        self.speed = speed_m_s
        self.yaw_rate = yaw_rate_rad_s
        # world == map frame for the stand-in, with the TRUE delta applied to
        # the walls (so the map the belief holds is genuinely mis-registered)
        self._xy = np.zeros(2)
        self._yaw = math.pi / 2.0                       # facing +y, the wall
        self._t = 0.0
        self._last_odom = (self._xy.copy(), self._yaw)
        self.true_plan = self._shifted_plan()
        self.peak_force_n = 0.0
        self.n_steps = 0

    def _shifted_plan(self):
        from global_nav.floorplan import Floorplan
        d = np.array([self.scene.true_delta_m, 0.0])
        return Floorplan(tuple(b.translated(d) for b in self.scene.plan.bodies),
                         self.scene.plan.features, self.scene.plan.edges,
                         self.scene.plan.route, self.scene.plan.name)

    # -- state ------------------------------------------------------------------

    def truth_pose(self):
        """EVALUATION ONLY."""
        return self._xy.copy(), float(self._yaw)

    @property
    def time_s(self) -> float:
        return self._t

    def odometry_increment(self):
        prev_xy, prev_yaw = self._last_odom
        d_w = self._xy - prev_xy
        c, s = math.cos(prev_yaw), math.sin(prev_yaw)
        d_body = np.array([c * d_w[0] + s * d_w[1], -s * d_w[0] + c * d_w[1]])
        d_yaw = float(wrap_angle(self._yaw - prev_yaw))
        dist = float(np.linalg.norm(d_body))
        cfg = self.cfg
        d_body = d_body + self.rng.standard_normal(2) * np.array(
            [cfg.odom_sigma_along_per_m * dist, cfg.odom_sigma_cross_per_m * dist])
        d_yaw += self.rng.standard_normal() * math.hypot(
            cfg.odom_sigma_yaw_per_rad * abs(d_yaw), cfg.odom_sigma_yaw_per_m * dist)
        self._last_odom = (self._xy.copy(), self._yaw)
        return tuple(d_body), d_yaw

    def stable(self) -> bool:
        return True

    # -- motion -----------------------------------------------------------------

    # -- map-frame commanding (the belief sets the displacement) -----------------

    def map_point_to_world(self, xy_map) -> np.ndarray:
        return np.asarray(xy_map, float).reshape(2)     # stand-in: world == map

    def world_point_to_map(self, xy_w) -> np.ndarray:
        return np.asarray(xy_w, float).reshape(2)

    def map_yaw_to_world(self, yaw_map: float) -> float:
        return float(yaw_map)                           # stand-in: world == map

    def world_yaw_to_map(self, yaw_w: float) -> float:
        """Identity here, and that is exactly why the physics/kinematic split in
        PF_AUDIT.md 4 was an artefact: the stand-in spawns at yaw pi/2 in a frame
        that IS the map, so a harness that forgot to map the yaw was still right
        on this backend and wrong on MuJoCo."""
        return float(yaw_w)

    def goto_map(self, target_map, yaw_map: float, believed_xy, believed_yaw: float,
                 tol_m: float | None = None, speed_scale: float | None = None) -> str:
        """Walk the displacement the BELIEF asks for, in the frame truth supplies.

        ``speed_scale`` is accepted for interface parity and ignored: the
        stand-in has no dynamics, so speed carries no safety meaning here.

        ``world = truth + R(yaw_truth - yaw_believed) . (target_map - believed)``
        -- only the belief determines where the robot ends up relative to where
        it is, which is what makes a wrong hypothesis a wrong place.
        """
        xy_w, yaw_w = self.truth_pose()
        rel = np.asarray(target_map, float).reshape(2) - np.asarray(believed_xy, float).reshape(2)
        a = yaw_w - float(believed_yaw)
        c, s = math.cos(a), math.sin(a)
        world_target = xy_w + np.array([c * rel[0] - s * rel[1], s * rel[0] + c * rel[1]])
        return self.goto_world(world_target, yaw_w + wrap_angle(float(yaw_map) - float(believed_yaw)))

    def turn_to_map(self, target_yaw_map: float, believed_yaw: float) -> str:
        # the stand-in tracks perfectly, so the map heading IS reachable exactly;
        # the real backend has to verify this against realized yaw
        return self.turn_to(float(target_yaw_map))

    def heading_error_rad(self, yaw_map: float) -> float:
        return float(abs(wrap_angle(float(yaw_map) - self._yaw)))

    def stabilize(self, yaw_map: float) -> bool:
        return True

    def goto_world(self, xy_w, yaw_w, *, max_s: float = 60.0) -> str:
        target = np.asarray(xy_w, float).reshape(2)
        while float(np.linalg.norm(target - self._xy)) > 1e-3:
            step = min(self.speed * self.dt, float(np.linalg.norm(target - self._xy)))
            direction = (target - self._xy) / max(float(np.linalg.norm(target - self._xy)), 1e-12)
            nxt = self._xy + step * direction
            if self._blocked(nxt):
                return "blocked"
            self._xy = nxt
            self._t += self.dt
            self.n_steps += 1
            if self._t > max_s * 40:
                return "timeout"
        return self.turn_to(yaw_w)

    def turn_to(self, yaw_w: float) -> str:
        while abs(wrap_angle(yaw_w - self._yaw)) > 1e-3:
            err = wrap_angle(yaw_w - self._yaw)
            self._yaw = float(wrap_angle(
                self._yaw + math.copysign(min(self.yaw_rate * self.dt, abs(err)), err)))
            self._t += self.dt
            self.n_steps += 1
        return "arrived"

    body_radius_m = 0.17

    def _blocked(self, xy) -> bool:
        """Signed distance, not penetration: penetration is zero everywhere
        outside a body, so a penetration test says 'blocked' in open floor."""
        p = np.asarray(xy, float).reshape(1, 2)
        for body in self.true_plan.bodies:
            sd, _ = body.signed_distance(p, None)
            if float(sd[0]) < self.body_radius_m:
                return True
        return False

    # -- probing ----------------------------------------------------------------

    def press(self, fan_offset_m: float, face_offset_m: float = 0.0) -> ProbeReading:
        """Press straight forward at a lateral offset.  Ray geometry only.
        ``face_offset_m`` is accepted for interface parity; the stand-in's
        raycast measures true geometry directly, so no reference correction is
        needed."""
        cfg = self.cfg
        f = np.array([math.cos(self._yaw), math.sin(self._yaw)])
        t = np.array([-f[1], f[0]])                       # +offset is to the left
        origin = self._xy + fan_offset_m * t
        dist, normal, _ = raycast(origin, f, cfg.reach_m, self.true_plan.bodies)
        self._t += 1.2                                    # a press costs time
        R = np.array([[math.cos(-self._yaw), -math.sin(-self._yaw)],
                      [math.sin(-self._yaw), math.cos(-self._yaw)]])
        if dist is not None:
            p_w = origin + dist * f
            p_body = R @ (p_w - self._xy)
            p_body = p_body + cfg.fk_sigma_m * self.rng.standard_normal(2)
            n_body = R @ np.asarray(normal, float)
            phi = math.atan2(n_body[1], n_body[0]) + cfg.normal_sigma_rad * self.rng.standard_normal()
            free = tuple((float(a), float(fan_offset_m)) for a in
                         np.arange(0.10, max(dist - 0.04, 0.11), 0.05))
            self.peak_force_n = max(self.peak_force_n, cfg.probe_force_n)
            return ProbeReading(True, tuple(p_body), (math.cos(phi), math.sin(phi)),
                                free, cfg.probe_force_n, 0.0)
        swept = cfg.reach_m
        free = tuple((float(a), float(fan_offset_m)) for a in
                     np.arange(0.10, swept, 0.05))
        return ProbeReading(False, None, None, free, 0.0, cfg.free_sweep_margin_m + 0.01)

    def set_precision_arrival(self, on: bool) -> None:
        pass                                   # the stand-in tracks perfectly

    def park_hands(self) -> None:
        pass

    def close(self) -> None:
        pass
