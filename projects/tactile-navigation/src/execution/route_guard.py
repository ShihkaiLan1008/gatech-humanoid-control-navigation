"""Planner-derived arrival regions and no-observation departure predictions.

The PF is the only state estimator. These guards neither update its belief nor
credit hypothetical contacts. The route and its ordering remain fixed.
"""
from __future__ import annotations

import heapq
import math
import numpy as np

from lane_graph.constrained import belief_step


class BeliefRouteGuard:
    def __init__(self, route, confidence=.95):
        if not 0 < confidence < 1:
            raise ValueError("arrival confidence must lie between zero and one")
        self.route, self.confidence = route, float(confidence)
        self.points = np.array([route.grid.cell_xy(c) for c in route.plan.path])
        self.distances = np.r_[0., np.cumsum(np.linalg.norm(np.diff(self.points, axis=0), axis=1))]
        self.contact_indices = {i + 1 for i, _ in route.schedule}
        # The exact chi-square quantile with two degrees of freedom.
        self.chi2 = -2. * math.log(1. - confidence)
        self.ends = np.cumsum([leg.s_end for leg in route.legs])

    def arrival(self, localizer, leg_index):
        """PF probability in the region compiled from the planned endpoint belief."""
        leg = self.route.legs[leg_index]
        if leg.kind == "stop":
            center = np.asarray(self.route.grid.goal_xy)
            T = np.eye(2) * self.route.grid.goal_radius**2
        else:
            center = leg.pt + leg.s_end * leg.along
            idx = int(np.argmin(abs(self.distances[1:] - self.ends[leg_index])))
            # Include the spatial resolution of the plan, rather than a
            # hand-authored room-specific waypoint radius.
            T = self.chi2 * np.array(self.route.trace[idx]["cov"]) + np.eye(2) * (self.route.grid.res / 2)**2
        pf = localizer.mb.b
        delta = pf.xy - center
        inside = np.einsum('ni,ij,nj->n', delta, np.linalg.inv(T), delta) <= 1.
        yaw_error = np.arctan2(np.sin(pf.theta - leg.yaw), np.cos(pf.theta - leg.yaw))
        inside &= abs(yaw_error) <= self.route.yaw_bound_rad
        mass = float(np.asarray(pf.weights)[inside].sum())
        return dict(probability=mass, required_probability=self.confidence,
                    center=center.tolist(), region_matrix=T.tolist(),
                    allowed=mass >= self.confidence)

    def horizon(self, target, target_g):
        """From a proposed command endpoint through the next contact opportunity."""
        out = [(np.asarray(target, float), None)]
        for i in range(1, len(self.points)):
            if self.distances[i] < target_g - 1e-6:
                continue
            if np.linalg.norm(self.points[i] - out[-1][0]) > 1e-6:
                out.append((self.points[i], i))
            else:
                out[-1] = (out[-1][0], i)
            if i in self.contact_indices:
                break
        return out

    def departure(self, xy, S, heading, target, target_g, yaw_cmd, recovery=False,
                  correction_tangent=None, position_bound_m=None):
        route = self.route
        pos_bound = route.position_bound_m if position_bound_m is None else position_bound_m
        xy, S = np.asarray(xy, float), np.array(S, float, copy=True)
        rows, total = [], 0.
        for end, index in self.horizon(target, target_g):
            d = end - xy
            length = float(np.linalg.norm(d))
            if length < 1e-9:
                continue
            command = yaw_cmd if index is None else float(route.trace[index - 1]["heading_rad"])
            clear = (route.grid.segment_clear(xy, end, allow_escape=True) if recovery
                     else route.grid.segment_clear(xy, end))
            reference = None
            if correction_tangent is not None:
                # Restore the active leg first. Subsequent route vertices use
                # their own planned direction, so a future turn remains gated.
                reference = (correction_tangent if not rows or index is None
                             else route.trace[index - 1]["motion_tangent"])
            S, _, heading, _, _, info = belief_step(
                S, S[2, 2], heading, True, True, length, d / length, command,
                "none", None, route.cfg, pos_bound, route.yaw_bound_rad,
                constraint_tangent=reference)
            total += length
            rows.append(dict(xy=end.tolist(), distance_m=total,
                             sigma_m=info["max_pos_sigma"], yaw_sigma_rad=info["max_yaw_sigma"],
                             constraint_tangent=info["constraint_tangent"],
                             position_allowed=bool(info["max_pos_sigma"] <= pos_bound + 1e-12),
                             yaw_allowed=bool(info["max_yaw_sigma"] <= route.yaw_bound_rad + 1e-12),
                             geometry_clear=clear, allowed=bool(info["ok"] and clear)))
            xy = end
        return dict(allowed=all(r["allowed"] for r in rows), horizon_m=total,
                    position_allowed=all(r["position_allowed"] for r in rows),
                    yaw_allowed=all(r["yaw_allowed"] for r in rows),
                    geometry_allowed=all(r["geometry_clear"] for r in rows),
                    sigma_m=max((r["sigma_m"] for r in rows), default=0.),
                    yaw_sigma_rad=max((r["yaw_sigma_rad"] for r in rows), default=0.),
                    rows=rows, terminal="contact_opportunity" if rows else "current_stance")


def local_connector(grid, xy, target):
    """Reconnect to a specified nearest route point; never change the global path."""
    xy, target = np.asarray(xy, float), np.asarray(target, float)
    if grid.segment_clear(xy, target, allow_escape=True):
        return [target]
    cells = [tuple(v) for v in np.argwhere(grid.free)]
    starts = sorted(cells, key=lambda c: np.linalg.norm(grid.cell_xy(c) - xy))[:16]
    ends = sorted(cells, key=lambda c: np.linalg.norm(grid.cell_xy(c) - target))[:16]
    goals = {c for c in ends if grid.segment_clear(grid.cell_xy(c), target)}
    queue, best = [], {}
    for c in starts:
        if grid.segment_clear(xy, grid.cell_xy(c), allow_escape=True):
            g = float(np.linalg.norm(grid.cell_xy(c) - xy))
            heapq.heappush(queue, (g + float(np.linalg.norm(grid.cell_xy(c) - target)), g, c, (c,)))
    while queue:
        _, cost, c, path = heapq.heappop(queue)
        if cost >= best.get(c, math.inf):
            continue
        best[c] = cost
        if c in goals:
            return [grid.cell_xy(v) for v in path] + [target]
        for i, (dx, dy, length, _) in enumerate(grid.moves):
            if grid.move_ok.get((*c, i)):
                nxt = (c[0] + dx, c[1] + dy)
                new = cost + length
                heapq.heappush(queue, (new + float(np.linalg.norm(grid.cell_xy(nxt) - target)), new, nxt, path + (nxt,)))
    return None
