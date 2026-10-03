"""Geometric route generation: known map + start + goal -> smooth, clear route.

Subtask 5.  The V3 stack already derives wall-follow lanes automatically from a
wall list (`maps_v3.followable`), so lanes are not the gap.  The gap is
free-space travel between them, which `graph3.build_graph3` covers with a
1.15 m grid of waypoints placed only inside hand-declared room rectangles,
joined by straight hops capped at 2.6 m.  That fails on a new floorplan in
three ways a benchmark will hit immediately:

* a doorway narrower than the grid pitch can receive no waypoint at all, so
  there is no route through it;
* free space outside every declared room rectangle is unreachable, so the
  rectangles are load-bearing hand authoring;
* the result is a hop chain, not a path -- it has no smoothing, so the base
  trajectory zig-zags between waypoints.

This module replaces that layer with the simple reliable method: inflate, A*,
shortcut, smooth.  It deliberately does NOT decide tactile behaviour.  It
answers one question -- "what collision-free routes exist from here to there" --
and hands the belief-aware planner a set of them to choose between.

The map is supplied as a **clearance function**, ``clearance(points) -> metres
to the nearest obstacle``, so the same planner serves the `MapV3` specs of the
lane-graph stack and the `Floorplan` of the global belief without either
importing the other.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np

# 8-connected moves and their step costs.
_NEIGHBOURS = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
               (-1, -1, 1.4142135623730951), (-1, 1, 1.4142135623730951),
               (1, -1, 1.4142135623730951), (1, 1, 1.4142135623730951)]


@dataclass(frozen=True)
class RouteConfig:
    cell_m: float = 0.08
    """Grid pitch.  0.08 m resolves a 0.7 m doorway into ~9 cells, which is
    enough for A* to find the gap that the old 1.15 m connector grid missed."""

    robot_clearance_m: float = 0.34
    """Hard minimum clearance from the base centre to any surface.  The G1's
    measured stance half-width plus margin; a cell closer than this is not
    traversable at all."""

    prefer_clearance_m: float = 0.60
    """Clearance beyond which no penalty applies.  Between the two the planner
    pays to hug a wall, so routes prefer the middle of a corridor without
    refusing a tight one."""

    clearance_weight: float = 2.0
    """Cost per metre of travel at zero preferred-clearance margin."""

    touch_reach_m: float = 0.0
    """Arm reach [m].  ZERO KEEPS THE SIGHTED-ROBOT PREFERENCE above, where
    cost falls monotonically with clearance and routes avoid walls -- which is
    what a free-space connector between two lanes wants (`graph3`).

    POSITIVE TURNS THE PREFERENCE INSIDE OUT, into the one a robot that
    navigates BY TOUCH actually has.  `prefer_clearance_m` charges its maximum
    at `robot_clearance_m` and nothing past 0.60 m, so with a 0.45 m arm it
    prices the only band where anything is touchable at 3x and the middle of
    an empty room at 1x.  A route planned under that preference crosses open
    space by choice and feels nothing on the way -- room2's A* route ran
    1.45 m from the nearest surface for its whole length.

    With this set, travel is free anywhere a surface is within reach and pays
    only for going out of reach, so the cheap route is the one that stays
    where its own localization comes from."""

    touch_span_m: float = 1.2
    """Distance beyond `touch_reach_m` over which the out-of-reach penalty
    saturates.  It has to saturate: an unbounded price on open space makes
    crossing a doorway or a room arbitrarily expensive and the planner
    refuses geometry it should simply pay for once."""

    feature_bonus: float = 0.0
    """DEPRECATED shape of the feature preference: a discount per metre for
    travelling within reach of a mapped feature.  Kept at zero and not
    recommended -- see `plan_route_via_features`, which is the form that
    works.  A per-metre discount cannot attract a route to a SPARSE feature:
    room2 has one corner 1.5 m off the direct line, so reaching it costs a
    ~2 m detour while the discount can only save `bonus` x the ~1 m spent
    beside it.  Breaking even needs `bonus >= 2`, which is the entire
    out-of-reach penalty, i.e. the knob only works at a setting that
    destroys the term it is riding on.  Visiting a sparse feature is an
    orienteering decision, not a shortest-path cost tweak.

    What the discount WAS meant to price, and still records:

    `touch_reach_m` alone is reach-aware, not information-aware: it prices a
    flat face and a corner identically, and they are not worth the same.  A
    press on a flat face is rank 1 by the observation model's construction --
    it measures distance TO the wall and says nothing about position ALONG
    it -- and PF_AUDIT measured what that costs: 5 of 11 tactile updates
    carried exactly zero information, while the one touch that met real
    corner geometry carried +27 nats and pulled the error 0.061 -> 0.007 m in
    a single update.

    Zero by default.  It is a DIFFERENT route preference, not a better
    setting of the same one: the hand-designed routes cut their corners too
    (room2's leaves the west wall 1.6 m before the corner and crosses to the
    north wall's lane), so turning this on moves the planner away from the
    hand reference on purpose.  Kept as its own knob so the two can be
    compared rather than conflated."""

    feature_reach_m: float = 0.0
    """How close to a feature counts as passing it.  Zero takes
    `robot_clearance_m + touch_reach_m` -- a stance the BASE can legally
    stand in, from which the arm still reaches the feature.

    `touch_reach_m` alone is the wrong default and silently does nothing: a
    corner is where two faces meet, so the nearest cell the base may occupy
    is out along the diagonal at `robot_clearance_m * sqrt(2)` ~ 0.48 m, and
    a 0.45 m radius around room2's only corner contains ZERO free cells.  The
    bonus region is then unreachable, the surcharge on everywhere-else
    applies uniformly, and its only effect is to dilute the touch band --
    measured: the route moved 1.39 -> 1.72 m away from the very corner the
    bonus was meant to attract it to."""

    smooth_window: int = 3
    smooth_passes: int = 2

    max_cells: int = 400_000
    """Refuse rather than thrash on an accidentally huge bounding box."""


@dataclass
class Route:
    """A collision-free base route, plus what the caller needs to judge it."""

    points: np.ndarray            # (N,2) smoothed polyline in map coordinates
    length_m: float
    min_clearance_m: float
    found: bool = True
    reason: str = ""

    def __bool__(self) -> bool:
        return self.found


class OccupancyGrid:
    """Inflated free-space grid sampled from a clearance function."""

    def __init__(self, clearance_fn, bounds, cfg: RouteConfig,
                 feature_xy=None):
        """``feature_xy`` is an (F,2) array of mapped rank-2 features -- the
        corners, wall ends and point landmarks a touch there would actually
        discriminate on.  Supplied as bare points for the same reason the map
        arrives as a clearance function: the planner must not need either
        caller's floorplan type to answer a route query."""
        (x0, y0, x1, y1) = bounds
        self.cfg = cfg
        self.origin = np.array([x0, y0], float)
        self.nx = max(2, int(np.ceil((x1 - x0) / cfg.cell_m)) + 1)
        self.ny = max(2, int(np.ceil((y1 - y0) / cfg.cell_m)) + 1)
        if self.nx * self.ny > cfg.max_cells:
            raise ValueError(
                f"grid {self.nx}x{self.ny} exceeds max_cells={cfg.max_cells}; "
                f"raise cell_m or tighten the bounds")
        gx = self.origin[0] + np.arange(self.nx) * cfg.cell_m
        gy = self.origin[1] + np.arange(self.ny) * cfg.cell_m
        mesh = np.stack(np.meshgrid(gx, gy, indexing="ij"), axis=-1)
        self.clear = np.asarray(
            clearance_fn(mesh.reshape(-1, 2)), float).reshape(self.nx, self.ny)
        self.free = self.clear >= cfg.robot_clearance_m
        self.near_feature = np.zeros_like(self.free)
        f = np.asarray([] if feature_xy is None else feature_xy,
                       float).reshape(-1, 2)
        if len(f) and cfg.feature_bonus > 0.0:
            reach = cfg.feature_reach_m or (cfg.robot_clearance_m
                                            + (cfg.touch_reach_m
                                               or cfg.prefer_clearance_m))
            d = np.linalg.norm(
                mesh.reshape(-1, 1, 2) - f.reshape(1, -1, 2), axis=2).min(1)
            self.near_feature = (d.reshape(self.nx, self.ny) <= reach)

    # ── index <-> world ─────────────────────────────────────────────────────────

    def to_cell(self, p) -> tuple[int, int]:
        q = (np.asarray(p, float) - self.origin) / self.cfg.cell_m
        return (int(np.clip(round(q[0]), 0, self.nx - 1)),
                int(np.clip(round(q[1]), 0, self.ny - 1)))

    def to_world(self, i: int, j: int) -> np.ndarray:
        return self.origin + np.array([i, j], float) * self.cfg.cell_m

    def nearest_free(self, p, max_radius_cells: int = 40) -> tuple[int, int] | None:
        """Snap a start/goal that sits inside the inflation band onto free space.

        A goal 0.30 m from a wall is legitimate -- the robot is meant to end up
        near things -- but the planner cannot route to a blocked cell, so it
        routes to the closest free one and lets the executive close the gap.
        """
        i0, j0 = self.to_cell(p)
        if self.free[i0, j0]:
            return i0, j0
        for r in range(1, max_radius_cells + 1):
            best, best_d = None, np.inf
            for di in range(-r, r + 1):
                for dj in (-r, r) if abs(di) < r else range(-r, r + 1):
                    i, j = i0 + di, j0 + dj
                    if 0 <= i < self.nx and 0 <= j < self.ny and self.free[i, j]:
                        d = di * di + dj * dj
                        if d < best_d:
                            best, best_d = (i, j), d
            if best is not None:
                return best
        return None

    def line_is_clear(self, a, b) -> bool:
        """Sample a straight segment at half-cell pitch."""
        a, b = np.asarray(a, float), np.asarray(b, float)
        n = max(2, int(np.linalg.norm(b - a) / (0.5 * self.cfg.cell_m)) + 1)
        for t in np.linspace(0.0, 1.0, n):
            i, j = self.to_cell(a + t * (b - a))
            if not self.free[i, j]:
                return False
        return True

    def clearance_at(self, p) -> float:
        i, j = self.to_cell(p)
        return float(self.clear[i, j])

    def arm_can_see(self, a, b) -> bool:
        """Does the straight segment a->b stay out of MATERIAL?

        `line_is_clear` is the BASE's question and tests `free`, i.e. at
        least `robot_clearance_m` from every surface.  An arm reaching for a
        surface necessarily ends inside that margin, so the base test is
        vacuously false for every stance and cannot be reused here.  What an
        arm needs is only that nothing solid lies between it and the point."""
        a, b = np.asarray(a, float), np.asarray(b, float)
        n = max(2, int(np.linalg.norm(b - a) / (0.5 * self.cfg.cell_m)) + 1)
        for t in np.linspace(0.0, 1.0, n):
            i, j = self.to_cell(a + t * (b - a))
            if not (0 <= i < self.nx and 0 <= j < self.ny):
                return False
            if self.clear[i, j] <= 0.0:
                return False
        return True


def _cell_penalty(grid: OccupancyGrid, i: int, j: int) -> float:
    """Extra cost per metre for travelling this far from a surface (touch
    mode) or this close to one (the default, sighted preference)."""
    cfg = grid.cfg
    clear = float(grid.clear[i, j])
    if cfg.touch_reach_m > 0.0:
        # out of reach is the only thing that costs; inside the band every
        # cell is equally cheap, so the tie is broken by distance and the
        # route hugs whichever surface is on its way
        excess = max(clear - cfg.touch_reach_m, 0.0)
        pen = cfg.clearance_weight * float(
            np.clip(excess / max(cfg.touch_span_m, 1e-6), 0.0, 1.0))
        # a discount expressed as a surcharge everywhere ELSE, so no step
        # cost can go negative and the octile heuristic stays admissible
        if cfg.feature_bonus > 0.0 and not grid.near_feature[i, j]:
            pen += cfg.feature_bonus
        return pen
    margin = clear - cfg.robot_clearance_m
    span = max(cfg.prefer_clearance_m - cfg.robot_clearance_m, 1e-6)
    return cfg.clearance_weight * float(np.clip(1.0 - margin / span, 0.0, 1.0))


def astar(grid: OccupancyGrid, start_cell, goal_cell) -> list[tuple[int, int]]:
    """8-connected A* with an octile heuristic; ties broken toward the goal."""
    cell = grid.cfg.cell_m
    si, sj = start_cell
    gi, gj = goal_cell

    def h(i, j):
        dx, dy = abs(i - gi), abs(j - gj)
        return cell * (max(dx, dy) + 0.4142135623730951 * min(dx, dy))

    open_heap = [(h(si, sj), 0.0, si, sj)]
    came: dict[tuple[int, int], tuple[int, int]] = {}
    best = {(si, sj): 0.0}
    seen = set()
    while open_heap:
        _, g, i, j = heapq.heappop(open_heap)
        if (i, j) in seen:
            continue
        seen.add((i, j))
        if (i, j) == (gi, gj):
            path = [(i, j)]
            while (i, j) in came:
                i, j = came[(i, j)]
                path.append((i, j))
            return path[::-1]
        for di, dj, w in _NEIGHBOURS:
            ni, nj = i + di, j + dj
            if not (0 <= ni < grid.nx and 0 <= nj < grid.ny) or not grid.free[ni, nj]:
                continue
            if di and dj and not (grid.free[i + di, j] and grid.free[i, j + dj]):
                continue      # no corner-cutting through a diagonal gap
            step = cell * w * (1.0 + _cell_penalty(grid, ni, nj))
            ng = g + step
            if ng < best.get((ni, nj), np.inf) - 1e-12:
                best[(ni, nj)] = ng
                came[(ni, nj)] = (i, j)
                heapq.heappush(open_heap, (ng + h(ni, nj), ng, ni, nj))
    return []


def _penalized_length(grid: OccupancyGrid, a, b) -> float:
    """Length of the segment a->b in the SAME currency `astar` minimized."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = float(np.linalg.norm(b - a))
    n = max(2, int(d / grid.cfg.cell_m) + 1)
    step = d / (n - 1)
    tot = 0.0
    for t in np.linspace(0.0, 1.0, n)[1:]:
        i, j = grid.to_cell(a + t * (b - a))
        i = int(np.clip(i, 0, grid.nx - 1))
        j = int(np.clip(j, 0, grid.ny - 1))
        tot += step * (1.0 + _cell_penalty(grid, i, j))
    return tot


def shortcut(grid: OccupancyGrid, pts: np.ndarray) -> np.ndarray:
    """Greedy line-of-sight shortcutting: the grid path becomes a polyline.

    A chord is taken only when it is no more expensive than the path it
    replaces, measured with the SAME cell penalty A* just minimized.  A
    collision-only test is not enough: it accepts any chord that fits, which
    on a touch-band cost throws away the whole route.  Measured on room2 --
    A* returned the wall-following L, the collision-only shortcut replaced it
    with the open-space diagonal, and the planner emitted the one route from
    which the robot can feel nothing."""
    if len(pts) < 3:
        return pts
    seg = [_penalized_length(grid, pts[k], pts[k + 1])
           for k in range(len(pts) - 1)]
    out = [pts[0]]
    i = 0
    while i < len(pts) - 1:
        j = len(pts) - 1
        while j > i + 1 and not (
                grid.line_is_clear(pts[i], pts[j])
                and _penalized_length(grid, pts[i], pts[j])
                <= sum(seg[i:j]) + 1e-9):
            j -= 1
        out.append(pts[j])
        i = j
    return np.asarray(out, float)


def smooth(grid: OccupancyGrid, pts: np.ndarray, cfg: RouteConfig) -> np.ndarray:
    """Moving-average corner rounding, rejected wherever it would clip a wall.

    A tactile route must stay smooth: the base trajectory is what the controller
    tracks, and station hopping that re-breaks it into zig-zags is exactly what
    Subtask 5 forbids.
    """
    if len(pts) < 3:
        return pts
    out = np.asarray(pts, float).copy()
    k = max(1, cfg.smooth_window // 2)
    for _ in range(cfg.smooth_passes):
        cand = out.copy()
        for i in range(1, len(out) - 1):
            lo, hi = max(0, i - k), min(len(out), i + k + 1)
            cand[i] = out[lo:hi].mean(axis=0)
        for i in range(1, len(out) - 1):
            if (grid.line_is_clear(cand[i - 1], cand[i])
                    and grid.line_is_clear(cand[i], cand[i + 1])):
                out[i] = cand[i]
    return out


def _resample(pts: np.ndarray, pitch_m: float) -> np.ndarray:
    """Even spacing, so downstream consumers see a uniform polyline."""
    if len(pts) < 2:
        return pts
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.r_[0.0, np.cumsum(seg)]
    if s[-1] < 1e-9:
        return pts
    n = max(2, int(round(s[-1] / pitch_m)) + 1)
    t = np.linspace(0.0, s[-1], n)
    return np.stack([np.interp(t, s, pts[:, 0]), np.interp(t, s, pts[:, 1])], axis=1)


def plan_route_on(grid: OccupancyGrid, start, goal,
                  cfg: RouteConfig | None = None) -> Route:
    """The same route query against a grid the caller already built.

    Sampling the clearance function over the whole map is the expensive half of
    a route query and it does not depend on the endpoints, so a caller asking
    many questions of one map -- ``build_graph3`` asks one per blocked node pair
    -- must be able to pay for it once.
    """
    cfg = cfg or grid.cfg
    s = grid.nearest_free(start)
    g = grid.nearest_free(goal)
    if s is None or g is None:
        which = "start" if s is None else "goal"
        return Route(np.empty((0, 2)), 0.0, 0.0, False,
                     f"{which} has no free cell within reach of it")
    cells = astar(grid, s, g)
    if not cells:
        return Route(np.empty((0, 2)), 0.0, 0.0, False,
                     "no collision-free route exists at this clearance")

    raw = np.array([grid.to_world(i, j) for i, j in cells], float)
    # keep the caller's exact endpoints; A* can only reach cell centres
    raw[0], raw[-1] = np.asarray(start, float), np.asarray(goal, float)
    pts = _resample(smooth(grid, shortcut(grid, raw), cfg), 3.0 * cfg.cell_m)
    length = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
    clr = float(min(grid.clearance_at(p) for p in pts))
    return Route(pts, length, clr, True, "")


def plan_route(clearance_fn, bounds, start, goal,
               cfg: RouteConfig = RouteConfig()) -> Route:
    """known map + start + goal -> one smooth collision-free base route."""
    return plan_route_on(OccupancyGrid(clearance_fn, bounds, cfg), start, goal, cfg)


def plan_alternatives(clearance_fn, bounds, start, goal, n: int = 3,
                      cfg: RouteConfig = RouteConfig()) -> list[Route]:
    """Several genuinely different routes, for the belief-aware planner to choose
    between.

    Shortest-path cost must not decide tactile behaviour, so this returns a set
    rather than a winner: each subsequent route is forced to avoid the corridor
    already taken, which is what surfaces the "long way round past a landmark"
    option that a pure distance planner would never offer.
    """
    routes: list[Route] = []
    grid = OccupancyGrid(clearance_fn, bounds, cfg)
    blocked = np.zeros_like(grid.free)
    for _ in range(max(1, n)):
        g2 = object.__new__(OccupancyGrid)
        g2.__dict__.update(grid.__dict__)
        g2.free = grid.free & ~blocked
        s, t = g2.nearest_free(start), g2.nearest_free(goal)
        if s is None or t is None:
            break
        cells = astar(g2, s, t)
        if not cells:
            break
        raw = np.array([g2.to_world(i, j) for i, j in cells], float)
        raw[0], raw[-1] = np.asarray(start, float), np.asarray(goal, float)
        pts = _resample(smooth(g2, shortcut(g2, raw), cfg), 3.0 * cfg.cell_m)
        routes.append(Route(
            pts, float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1))),
            float(min(grid.clearance_at(p) for p in pts)), True, ""))
        # block a corridor around this route's interior, never its endpoints
        r_cells = max(1, int(round(0.45 / cfg.cell_m)))
        for p in pts[len(pts) // 8: -max(1, len(pts) // 8)]:
            i, j = grid.to_cell(p)
            blocked[max(0, i - r_cells):i + r_cells + 1,
                    max(0, j - r_cells):j + r_cells + 1] = True
    return routes


def feature_stances(grid: OccupancyGrid, feature_xy, cfg: RouteConfig | None = None,
                    near=None):
    """The cheapest legal stance beside each feature, or None where there is
    none.  A stance is a free cell from which the arm reaches the feature
    ALONG A CLEAR LINE.

    The line-of-sight test is what makes it a stance rather than a nearby
    cell.  A clearance function says only how far the nearest surface is, and
    the far side of a wall is as free as the near side, so nearest-cell alone
    picks stances THROUGH the geometry: on room2 it chose the outside of the
    room and the route grew 5.1 -> 13.6 m walking all the way around to
    reach a corner that was 1.5 m away.  The segment is tested to just short
    of the feature, because the feature itself sits ON the material."""
    cfg = cfg or grid.cfg
    reach = cfg.feature_reach_m or (cfg.robot_clearance_m
                                    + (cfg.touch_reach_m
                                       or cfg.prefer_clearance_m))
    ref = None if near is None else np.asarray(near, float).reshape(-1, 2)
    out = []
    for f in np.asarray(feature_xy, float).reshape(-1, 2):
        lo = grid.to_cell(f - reach)
        hi = grid.to_cell(f + reach)
        best, best_key = None, None
        for i in range(max(0, lo[0]), min(grid.nx, hi[0] + 1)):
            for j in range(max(0, lo[1]), min(grid.ny, hi[1] + 1)):
                if not grid.free[i, j]:
                    continue
                q = grid.to_world(i, j)
                d = float(np.linalg.norm(q - f))
                if d > reach:
                    continue
                if not grid.arm_can_see(q, q + 0.85 * (f - q)):
                    continue
                # WHAT A STANCE COSTS, not how near it is.  Both sides of a
                # corner can see it and the OUTSIDE one is often nearer --
                # room2's outside stance sits 0.449 m from the corner against
                # the inside one's 0.48 m, and routing to it meant walking
                # around the whole room (5.1 -> 13.6 m).  Ranking by distance
                # to the route being detoured picks the side the robot is
                # already on.
                key = (d if ref is None
                       else (float(np.linalg.norm(ref - q, axis=1).min()), d))
                if best_key is None or key < best_key:
                    best, best_key = q, key
        out.append(best)
    return out


def plan_route_via_features(grid: OccupancyGrid, start, goal, feature_xy,
                            cfg: RouteConfig | None = None,
                            detour_budget: float = 1.6) -> Route:
    """Route from start to goal, going PAST the mapped features on the way.

    Visiting a sparse feature is an orienteering decision, not a cost tweak
    on the shortest path (see `RouteConfig.feature_bonus` for the measurement
    that settles this).  So it is decided as one: plan the direct route, then
    plan a route that threads the feature stances in route order, and take
    the threaded one only if it costs no more than `detour_budget` times the
    direct route.

    WHY A BUDGET RATHER THAN A PRICE.  A detour to a corner buys rank-2
    information -- the one channel that constrains position ALONG a wall,
    which no amount of flat-face pressing can -- but the belief only has so
    much drift to spend on getting there, and a route that visits every
    feature in a large room arrives worse than one that visits none.  The
    budget is the honest form of that: it says how much travel the operator
    will pay for the information, and leaves the planner to spend it.

    Features are threaded in the order their stances project onto the direct
    route, which keeps the result a route rather than a tour: this is not a
    TSP, and a feature that would need doubling back is simply not worth its
    own detour under any budget a tactile robot can afford.
    """
    cfg = cfg or grid.cfg
    direct = plan_route_on(grid, start, goal, cfg)
    if not direct.found:
        return direct
    stances = [s for s in feature_stances(grid, feature_xy, cfg,
                                          near=direct.points)
               if s is not None]
    if not stances:
        return direct

    # order by projection onto the direct route: the leg a stance belongs to
    d = np.asarray(direct.points, float)
    order = sorted(stances,
                   key=lambda q: int(np.argmin(np.linalg.norm(d - q, axis=1))))

    legs, total, ok = [], 0.0, True
    for a, b in zip([np.asarray(start, float)] + order,
                    order + [np.asarray(goal, float)]):
        leg = plan_route_on(grid, a, b, cfg)
        if not leg.found:
            ok = False
            break
        legs.append(leg.points)
        total += leg.length_m
    if not ok or total > detour_budget * direct.length_m:
        return direct

    pts = np.vstack([legs[0]] + [p[1:] for p in legs[1:]])
    clr = float(min(grid.clearance_at(p) for p in pts))
    return Route(pts, total, clr, True, "via features")
