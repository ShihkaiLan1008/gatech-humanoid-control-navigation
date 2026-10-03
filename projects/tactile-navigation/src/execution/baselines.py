"""Explicit opt-in baselines for the existing global-route execution pipeline.

No room-specific tuning. The controller, map loader, contact primitive and
evaluation observer are shared. Importing this module changes no defaults.
"""
from __future__ import annotations

import hashlib
import heapq
import json
import math

import numpy as np

from .global_route import GlobalRoute, RouteFollowPolicy, legs_from_path


def describe(preset, route):
    odom = preset in ("odom_astar", "odom_global")
    return dict(preset=preset, estimator="odometry" if odom else "tactile_pf",
                selector="none" if odom else preset if preset in ("random_touch", "sweep4") else "eig",
                path_planner="geometric_astar" if preset == "odom_astar" else "current",
                recovery="none" if preset == "no_recovery" else "local_reconnect",
                uncertainty_enforced=not odom,
                route_sha256=hashlib.sha256(json.dumps(dict(
                    xy=[route.grid.cell_xy(c).tolist() for c in route.plan.path],
                    headings=[leg.yaw for leg in route.legs]), sort_keys=True).encode()).hexdigest())


class ProbeSelector:
    """Random order or a four-direction sweep, with no EIG computation.

    Feasibility excludes a guard inside a wall and probes across open ends;
    a predicted miss is still an action. Each action is used once per batch.
    """
    uses_eig = False
    last = None  # no EIG surface prediction for TactileIO's fast approach

    def __init__(self, mode, plan, localizer, seed, n_grid=31, open_boundaries=(), reach_profile=None):
        if mode not in ("random_touch", "sweep4"):
            raise ValueError(mode)
        self.mode, self.plan, self.loc = mode, plan, localizer
        self.open_boundaries = tuple(open_boundaries)
        self.reach_profile = reach_profile
        self.rng = np.random.default_rng(np.random.SeedSequence([seed, 104729]))
        if mode == "sweep4":
            # Body-frame yaw -110, -36.67, +36.67, +110 degrees.
            lo, hi = (-110., 110.) if reach_profile is None else (
                -reach_profile.band_deg("right")[1], reach_profile.band_deg("left")[1])
            self.actions = [("right" if yaw < 0 else "left", math.radians(abs(yaw)))
                            for yaw in np.linspace(lo, hi, 4)]
        else:
            self.actions = [(hand, math.radians(alpha)) for hand in ("left", "right")
                            for alpha in np.linspace(*((-40., 110.) if reach_profile is None
                                                       else reach_profile.band_deg(hand)), n_grid)]
        self.pending = []

    def begin_batch(self):
        order = (self.rng.permutation(len(self.actions)) if self.mode == "random_touch"
                 else range(len(self.actions)))
        self.pending = [self.actions[i] for i in order]

    def feasible(self, hand, alpha):
        from global_nav.models import nearest_surface
        from .punch_eig import arm_geometry, TRAVEL_M
        from .punch_eig_step import GUARD_CLEAR_M
        pose = self.loc.pose()
        c, s = math.cos(pose[2]), math.sin(pose[2])
        R = np.array([[c, -s], [s, c]])
        start, direction, _ = arm_geometry(hand, alpha)
        ends = np.array([start, start + TRAVEL_M * direction]) @ R.T + pose[:2]
        if self.reach_profile is not None:
            ends = np.asarray(self.reach_profile.geometry(hand, alpha))[:, :2] @ R.T + pose[:2]
        distance, _ = nearest_surface(self.plan, ends[:1], None)
        return (float(distance[0]) >= GUARD_CLEAR_M and not any(
            np.max(ends @ np.array([nx, ny])) > offset
            for nx, ny, offset in self.open_boundaries))

    def choose_single(self, meta=None, projection=None):
        from .punch_eig import spec_at
        while self.pending:
            hand, alpha = self.pending.pop(0)
            if self.feasible(hand, alpha):
                spec = (spec_at(hand, alpha) if self.reach_profile is None
                        else spec_at(hand, alpha, reach_profile=self.reach_profile))
                return dict(hand=hand, spec=spec, alpha_deg=math.degrees(alpha),
                            expected_ig=None, p_hit=None, margin_m=None)
        return None


class OdometryLocalizer:
    """Live onboard pose with prediction-only covariance; no map or PF update.

    Covariance is diagnostic. Neither it nor contact data changes the onboard
    pose. Passive force stops still work through the existing TactileIO.
    """
    def __init__(self, controller, cfg):
        self.controller, self.cfg = controller, cfg
        sig = np.broadcast_to(np.asarray(cfg.prior_sigma_xy_m), (2,))
        self.P = np.diag([*sig**2, cfg.prior_sigma_theta_rad**2])
        self.n_observe = 0

    def pose(self):
        from .demo_logging import estimated_map_pose
        return estimated_map_pose(self.controller)

    def predict(self, controller):
        delta, yaw = controller.odometry_increment()
        delta = np.asarray(delta)
        theta = self.pose()[2] - yaw
        c, s = math.cos(theta), math.sin(theta)
        R = np.array([[c, -s], [s, c]])
        J = np.eye(3)
        J[:2, 2] = R @ np.array([-delta[1], delta[0]])
        distance = float(np.linalg.norm(delta))
        Q = np.zeros((3, 3))
        Q[:2, :2] = R @ np.diag([(self.cfg.sigma_along_per_m * distance)**2,
                                 (self.cfg.sigma_cross_per_m * distance)**2]) @ R.T
        Q[2, 2] = (self.cfg.sigma_yaw_per_m * distance)**2 + (self.cfg.sigma_yaw_per_rad * yaw)**2
        self.P = J @ self.P @ J.T + Q

    def observe(self, controller, contacts=(), frees=()):
        self.predict(controller)
        return dict(ignored_contacts=len(contacts), ignored_frees=len(frees))

    def cov3(self):
        return self.P.copy()

    def cov_xy(self):
        return self.P[:2, :2].copy()

    def sigma_m(self):
        return float(np.sqrt(np.trace(self.P[:2, :2])))

    def sigma_yaw(self):
        return float(np.sqrt(self.P[2, 2]))


class GeometricArrival:
    def __init__(self, route, endpoint_tolerance):
        self.route, self.endpoint_tolerance = route, endpoint_tolerance

    def arrival(self, localizer, leg_index):
        leg = self.route.legs[leg_index]
        center = self.route.grid.goal_xy if leg.kind == "stop" else leg.pt + leg.s_end * leg.along
        radius = self.route.grid.goal_radius if leg.kind == "stop" else self.endpoint_tolerance
        pose = localizer.pose()
        yaw_error = math.atan2(math.sin(pose[2] - leg.yaw), math.cos(pose[2] - leg.yaw))
        return dict(allowed=bool(np.linalg.norm(pose[:2] - center) <= radius and
                                 abs(yaw_error) <= self.route.yaw_bound_rad),
                    center=np.asarray(center).tolist(), radius_m=radius, mode="odometry")


class OdometryRouteFollowPolicy(RouteFollowPolicy):
    def __init__(self, *args, **kwargs):
        # Retain the existing endpoint-attempt ordering, with geometric arrival.
        kwargs["guard_mode"] = "belief"
        super().__init__(*args, **kwargs)
        self.guard = GeometricArrival(self.route, self.END_TOL_M)

    def _touch_until(self, *args, **kwargs):
        return dict(touches=[], reason="odometry only", target_sigma_m=None,
                    target_yaw_sigma_deg=None)

    def _motion_prediction(self, target_xy, yaw_map, recovery=False):
        xy = self.io.localizer.pose()[:2]
        clear = bool(self.route.grid.segment_clear(xy, target_xy, allow_escape=recovery))
        return dict(t=self.io.time_s, xy=xy.tolist(), target_xy=np.asarray(target_xy).tolist(),
                    distance_m=float(np.linalg.norm(np.asarray(target_xy) - xy)),
                    allowed=clear, geometry_allowed=clear, position_allowed=True, yaw_allowed=True,
                    uncertainty_enforced=False, sigma_m=self.io.localizer.sigma_m(),
                    yaw_sigma_rad=self.io.localizer.sigma_yaw())


def geometric_astar(grid):
    """Shortest distance on the existing collision-checked grid, to the goal disk."""
    def h(cell):
        return max(0., float(np.linalg.norm(grid.cell_xy(cell) - grid.goal_xy)) - grid.goal_radius)
    queue = [(h(grid.start), 0., grid.start)]
    best, parent, explored = {grid.start: 0.}, {}, 0
    while queue:
        _, cost, cell = heapq.heappop(queue)
        if cost > best[cell]:
            continue
        explored += 1
        if grid.is_goal(cell):
            path = [cell]
            while path[-1] in parent:
                path.append(parent[path[-1]])
            return path[::-1], explored
        for i, (dx, dy, length, _) in enumerate(grid.moves):
            if not grid.move_ok.get((*cell, i)):
                continue
            nxt, value = (cell[0] + dx, cell[1] + dy), cost + length
            if value < best.get(nxt, math.inf):
                best[nxt], parent[nxt] = value, cell
                heapq.heappush(queue, (value + h(nxt), value, nxt))
    return None, explored


def plan_astar(scene, yaw0, cfg, position_bound_m, yaw_bound_rad,
               prior_sigma_xy_m=.03, prior_sigma_theta_rad=.02, log=None,
               align_start=False, **_unused):
    from lane_graph.constrained import PlannerBelief
    from lane_graph.grid_constrained import GridBackend, GridPlan, build_grid, free_travel_distance
    from lane_graph.tradeoff import Timing, evaluate_route
    # A no-sensing route holds the travel heading, without wall-facing rewards.
    cfg = cfg.merged(face_surface_on_touch=False)
    grid = build_grid(scene, cfg)
    path, explored = geometric_astar(grid)
    if log:
        log("GLOBAL_PLAN", feasible=path is not None, strategy="geometric_astar", explored=explored)
    if path is None:
        return None
    if align_start and len(path) > 1:
        delta = grid.cell_xy(path[1]) - grid.cell_xy(path[0])
        initial_yaw = math.atan2(delta[1], delta[0])
        if log:
            log("START_HEADING", scene_yaw_deg=math.degrees(yaw0),
                spawn_yaw_deg=math.degrees(initial_yaw), mode="route")
        yaw0 = initial_yaw
    sig = np.broadcast_to(np.asarray(prior_sigma_xy_m), (2,))
    prior = PlannerBelief(cov=np.diag(sig**2), yaw_sigma=prior_sigma_theta_rad, minor=0.)
    plan = GridPlan(path, grid.path_length(path), grid.path_length(path), True, 0,
                    position_bound_m, yaw_bound_rad, explored, strategy="geometric_astar",
                    free_distance_m=sum(free_travel_distance(grid, a, b) for a, b in zip(path, path[1:])))
    schedule = frozenset()
    metrics, trace = evaluate_route(GridBackend(grid), path, prior, cfg, yaw0, Timing(), schedule)
    legs, stations = legs_from_path(grid, path, schedule, cfg, trace)
    return GlobalRoute(grid, plan, cfg, position_bound_m, yaw_bound_rad, schedule,
                       metrics, trace, legs, stations, yaw0, prior)
