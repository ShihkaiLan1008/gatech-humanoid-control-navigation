"""Grid formulation of the constrained tactile global planner.

The roadmap planner (`constrained.py`) searches a sparse lane graph.  This
module keeps the same belief model, hard bounds and localize-first rule,
but plans on a regular grid over the free floor. By default it minimizes
travel outside contact reach first; ``route_objective=distance_and_rotation``
uses distance plus commanded rotation as the primary cost:

* a node is a grid cell of ``grid_resolution_m``; a move joins two cells at
  most ``max_move_m`` apart (16 headings at 0.20 m / 0.50 m);
* a cell is traversable only if its centre keeps at least
  ``contact_standoff_m - contact_band_m`` of clearance from every surface,
  and every sample along a move does too (collision constraint);
* a cell lies in the *touch region* of a surface if that surface is inside
  ``contact_standoff_m +/- contact_band_m`` of the cell centre.  Walls found
  there give the rank-one normal update; landmarks give the two-axis update.
  Regions are precomputed once per map by ray casting from every free cell,
  so a corner is simply a cell inside two wall regions and a pillar is a ring;
* the search credits the most informative available contact at every cell it
  passes through (dense sensing); `tradeoff.prune_schedule` later thins that
  to the contacts actually needed.
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field

import numpy as np

from global_nav.geometry import raycast
from lane_graph.maps_v3 import clearance, in_rooms

from .constrained import (_State, _pre_limits, belief_step, pos_block, position_sigma_max,
                          to_se2, trace_row)
from .uncertainty import UncertaintyConfig, bin_key
from .uncertainty import DEFAULT as UNCERTAINTY

__all__ = ["TactileGrid", "GridPlan", "build_grid", "plan_grid", "replay_grid",
           "GridBackend", "commanded_heading", "best_contact"]


def _moves(cfg: UncertaintyConfig):
    """Integer offsets within ``max_move_m``, one per distinct heading."""
    r = int(math.ceil(cfg.max_move_m / cfg.grid_resolution_m))
    out = {}
    for dx in range(-r, r + 1):
        for dy in range(-r, r + 1):
            if dx == 0 and dy == 0:
                continue
            length = math.hypot(dx, dy) * cfg.grid_resolution_m
            if length > cfg.max_move_m + 1e-9:
                continue
            g = math.gcd(abs(dx), abs(dy))
            if g != 1:            # a multiple of a shorter move: same heading
                continue
            out[(dx, dy)] = (length, math.atan2(dy, dx))
    return [(dx, dy, L, h) for (dx, dy), (L, h) in sorted(out.items())]


@dataclass
class TactileGrid:
    res: float
    x0: float
    y0: float
    nx: int
    ny: int
    free: np.ndarray                     # (nx, ny) bool
    clear: np.ndarray                    # (nx, ny) clearance in metres
    contacts: dict                       # (ix, iy) -> [dict(kind, normal, distance, body)]
    moves: list                          # (dx, dy, length, heading)
    move_ok: dict                        # (ix, iy, move index) -> bool
    start: tuple
    goal_xy: np.ndarray
    goal_radius: float
    bodies: tuple
    cfg: UncertaintyConfig
    spec: object = None

    def segment_clear(self, a, b, allow_escape=False):
        """Continuous connector check, without rounding poses onto blocked cells."""
        a, b = np.asarray(a, float), np.asarray(b, float)
        n = max(2, int(math.ceil(np.linalg.norm(b - a) / (self.res / 2))) + 1)
        points = np.linspace(a, b, n)
        margin = self.cfg.contact_standoff_m - self.cfg.contact_band_m
        boundary_ok = all(np.all(points @ np.array([nx, ny]) <= offset - margin + 1e-9)
                          for nx, ny, offset in self.spec.open_boundaries)
        clear = clearance(points, self.spec)
        obstacle_ok = np.all(clear >= margin - 1e-9)
        if allow_escape and 0. < clear[0] < margin and clear[-1] > clear[0] + 1e-9:
            # A posterior correction can place the mean inside the inflated
            # safety buffer. Permit only an outward connector whose clearance
            # never decreases until the normal margin has been restored.
            # Small feedback steps may need several commands to restore it.
            safe = np.flatnonzero(clear >= margin)
            first_safe = int(safe[0]) if len(safe) else len(clear) - 1
            obstacle_ok = (np.all(np.diff(clear[:first_safe + 1]) >= -1e-9) and
                           (not len(safe) or np.all(clear[first_safe:] >= margin - 1e-9)))
        return bool(boundary_ok and np.all(in_rooms(points, self.spec)) and obstacle_ok)

    def cell_xy(self, cell):
        return np.array([self.x0 + cell[0] * self.res, self.y0 + cell[1] * self.res])

    def cell_of(self, xy):
        return (int(round((xy[0] - self.x0) / self.res)), int(round((xy[1] - self.y0) / self.res)))

    def inside(self, cell):
        return 0 <= cell[0] < self.nx and 0 <= cell[1] < self.ny

    def is_goal(self, cell):
        return float(np.linalg.norm(self.cell_xy(cell) - self.goal_xy)) <= self.goal_radius + 1e-9

    def region_cells(self):
        """Touch-region cells by kind, as scene-frame centres (for plots)."""
        out = {"wall": [], "landmark": []}
        for cell, cs in self.contacts.items():
            kind = "landmark" if any(c["kind"] == "landmark" for c in cs) else "wall"
            out[kind].append(self.cell_xy(cell).tolist())
        return out

    def path_length(self, path):
        return float(sum(np.linalg.norm(self.cell_xy(b) - self.cell_xy(a))
                         for a, b in zip(path[:-1], path[1:])))


def _contact_kind(bodies, plan, hit_i, hit_xy):
    body = bodies[int(hit_i)]
    kind = "wall" if str(getattr(body, "name", "")).startswith("wall") else "landmark"
    for feature in getattr(plan, "features", ()):
        if float(np.linalg.norm(hit_xy - np.asarray(feature.position, float))) <= 0.25:
            kind = "landmark"
    return kind


def build_grid(scene, cfg: UncertaintyConfig | None = None) -> TactileGrid:
    cfg = cfg or UNCERTAINTY
    spec = scene.spec
    res = cfg.grid_resolution_m
    min_clear = cfg.contact_standoff_m - cfg.contact_band_m
    # The follower executes the cell centre; it does not move to an implicit
    # closer stance inside that cell. Credit contact only within the actual
    # standoff band, independent of grid resolution.
    max_touch = cfg.contact_standoff_m + cfg.contact_band_m
    xs = [v for r in spec.rooms for v in (r[0], r[2])]
    ys = [v for r in spec.rooms for v in (r[1], r[3])]
    x0, y0 = min(xs), min(ys)
    nx = int(math.floor((max(xs) - x0) / res)) + 1
    ny = int(math.floor((max(ys) - y0) / res)) + 1
    ix, iy = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    centres = np.stack([x0 + ix.ravel() * res, y0 + iy.ravel() * res], axis=1)
    clear = clearance(centres, spec).reshape(nx, ny)
    for bx, by, limit in spec.open_boundaries:
        clear = np.minimum(clear, (limit - centres @ np.array([bx, by])).reshape(nx, ny))
    free = (in_rooms(centres, spec).reshape(nx, ny)) & (clear >= min_clear)

    bodies = tuple(scene.plan.bodies)
    angles = np.linspace(0.0, 2 * math.pi, cfg.contact_ray_count, endpoint=False)
    dirs = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    contacts = {}
    for cx in range(nx):
        for cy in range(ny):
            if not free[cx, cy] or clear[cx, cy] > max_touch + 0.05:
                continue
            origin = np.array([x0 + cx * res, y0 + cy * res])
            found = []
            for d in dirs:
                dist, normal, hit_i = raycast(origin, d, max_touch + 0.02, bodies)
                if dist is None or not (min_clear - 1e-9 <= dist <= max_touch + 1e-9):
                    continue
                n = np.asarray(normal, float)
                n = n / max(float(np.linalg.norm(n)), 1e-12)
                hit = origin + dist * d
                if any(abs(np.dot(hit, [bx, by]) - limit) < 0.02 and
                       np.dot(n, [bx, by]) > .9 for bx, by, limit in spec.open_boundaries):
                    continue  # an artificial end cap is not a sensed surface
                kind = _contact_kind(bodies, scene.plan, hit_i, origin + dist * d)
                # Group by surface direction: one entry per distinct normal.
                for c in found:
                    if abs(float(c["normal"] @ n)) > math.cos(math.radians(20)) and c["kind"] == kind:
                        if dist < c["distance"]:
                            c.update(distance=float(dist), normal=n, body=int(hit_i))
                        break
                else:
                    found.append(dict(kind=kind, normal=n, distance=float(dist), body=int(hit_i)))
            if found:
                contacts[(cx, cy)] = found

    moves = _moves(cfg)
    move_ok = {}
    samples = max(2, int(math.ceil(cfg.max_move_m / 0.1)))
    for cx in range(nx):
        for cy in range(ny):
            if not free[cx, cy]:
                continue
            for mi, (dx, dy, L, _h) in enumerate(moves):
                tx, ty = cx + dx, cy + dy
                if not (0 <= tx < nx and 0 <= ty < ny) or not free[tx, ty]:
                    continue
                a, b = np.array([x0 + cx * res, y0 + cy * res]), np.array([x0 + tx * res, y0 + ty * res])
                pts = a[None, :] + np.linspace(0, 1, samples + 1)[1:-1, None] * (b - a)[None, :]
                ok = bool(np.all(clearance(pts, spec) >= min_clear)) and bool(np.all(in_rooms(pts, spec)))
                if ok:
                    move_ok[(cx, cy, mi)] = True
    grid = TactileGrid(res=res, x0=x0, y0=y0, nx=nx, ny=ny, free=free, clear=clear,
                       contacts=contacts, moves=moves, move_ok=move_ok, spec=spec,
                       start=(0, 0), goal_xy=np.asarray(scene.goal_map_xy, float),
                       goal_radius=float(scene.goal_radius_m), bodies=bodies, cfg=cfg)
    start = grid.cell_of(scene.start_map_xy)
    if not (grid.inside(start) and free[start]):
        # Snap to the nearest free cell so a start with marginal clearance still plans.
        cand = np.argwhere(free)
        if not len(cand):
            raise ValueError("no free cells")
        d = np.linalg.norm(cand * res + [x0, y0] - np.asarray(scene.start_map_xy, float), axis=1)
        start = tuple(int(v) for v in cand[int(np.argmin(d))])
    grid.start = start
    return grid


def commanded_heading(grid: TactileGrid, cell, travel_heading: float, cfg: UncertaintyConfig) -> float:
    """Heading the base holds after a move into ``cell``: facing the nearest
    touchable surface when the cell is a touch cell (and the option is on),
    else the travel direction."""
    cs = grid.contacts.get(cell)
    if not cs or not cfg.face_surface_on_touch:
        return travel_heading
    c = min(cs, key=lambda c: c["distance"])
    n = c["normal"]                       # outward from the body, toward the robot
    return math.atan2(-n[1], -n[0])


def best_contact(P, contacts):
    """Landmark if available, else the wall whose normal is least certain."""
    if not contacts:
        return "none", None, None
    lm = [c for c in contacts if c["kind"] == "landmark"]
    if lm:
        c = min(lm, key=lambda c: c["distance"])
        return "landmark", c["normal"], c["distance"]
    B = pos_block(P)
    c = max(contacts, key=lambda c: float(c["normal"] @ B @ c["normal"]))
    return "wall", c["normal"], c["distance"]


@dataclass
class GridPlan:
    path: list | None
    distance_m: float
    objective_cost: float
    feasible: bool
    relaxation_level: int
    position_threshold_m: float
    yaw_threshold_rad: float
    explored_states: int
    trace: list = field(default_factory=list)
    reason: str = ""
    free_distance_m: float = 0.0
    strategy: str = "belief_astar"
    strategy_stats: dict = field(default_factory=dict)


def free_travel_distance(grid, a, b):
    """Distance outside contact regions, integrated at half-cell spacing."""
    start, delta = grid.cell_xy(a), grid.cell_xy(b) - grid.cell_xy(a)
    length = float(np.linalg.norm(delta))
    count = max(1, int(math.ceil(length / (grid.res / 2))))
    fractions = (np.arange(count) + 0.5) / count
    points = start + fractions[:, None] * delta
    return length * sum(grid.cell_of(p) not in grid.contacts for p in points) / count


def _free_travel(grid):
    """Free-travel edge costs and optimistic distances to the goal."""
    costs, reverse = {}, {}
    for (x, y, mi) in grid.move_ok:
        dx, dy, length, _heading = grid.moves[mi]
        a, b = (x, y), (x + dx, y + dy)
        # Midpoint integration also sees gaps between two touchable endpoints.
        free = free_travel_distance(grid, a, b)
        costs[x, y, mi] = free
        reverse.setdefault(b, []).append((a, free))
    # Optimistic remaining free travel, ignoring uncertainty and rotation.
    remaining = {tuple(c): 0.0 for c in np.argwhere(grid.free) if grid.is_goal(tuple(c))}
    queue = [(0.0, c) for c in remaining]
    heapq.heapify(queue)
    while queue:
        cost, cell = heapq.heappop(queue)
        if cost != remaining[cell]:
            continue
        for prev, length in reverse.get(cell, ()):
            value = round(cost + length, 12)
            if value < remaining.get(prev, math.inf):
                remaining[prev] = value
                heapq.heappush(queue, (value, prev))
    return costs, remaining


def _key(cell, heading_i, state: _State, cfg):
    yaw_sigma = math.sqrt(max(state.yaw_var, 0.0))
    yaw_bin = min(int(yaw_sigma / max(cfg.yaw_bin_rad, 1e-9)), cfg.max_yaw_bins)
    return (cell, bin_key(pos_block(state.P), cfg), yaw_bin, heading_i, state.resolved, state.localized)


def plan_grid(grid: TactileGrid, belief, cfg: UncertaintyConfig | None = None,
              yaw0: float | None = None) -> GridPlan:
    """Search under belief bounds using the configured route objective."""
    cfg = cfg or grid.cfg
    if cfg.route_objective not in ("free_distance_then_distance_and_rotation", "distance_and_rotation"):
        raise ValueError(f"unknown grid route objective: {cfg.route_objective!r}")
    if cfg.route_strategy == "seed_refine":
        from .grid_seed_refine import plan_seed_refine
        return plan_seed_refine(grid, belief, cfg, yaw0)
    if cfg.route_strategy != "belief_astar":
        raise ValueError(f"unknown grid route strategy: {cfg.route_strategy!r}")
    free_first = cfg.route_objective == "free_distance_then_distance_and_rotation"
    pos_limit, yaw_limit = cfg.position_threshold_m, cfg.yaw_threshold_rad
    _loc0, pre_pos, pre_yaw = _pre_limits(belief, cfg, pos_limit, yaw_limit)
    yaw_sigma = float(getattr(belief, "yaw_sigma", 0.0))
    state0 = _State(to_se2(belief.cov, yaw_sigma ** 2), yaw_sigma ** 2,
                    None if yaw0 is None else float(yaw0),
                    float(getattr(belief, "minor", 0.0)) <= 0.10, False)
    goal = grid.goal_xy
    free_cost, free_remaining = _free_travel(grid)

    def h(cell):
        return max(0.0, float(np.linalg.norm(grid.cell_xy(cell) - goal)) - grid.goal_radius)

    tie = 0
    pq = [(free_remaining.get(grid.start, math.inf) if free_first else 0., h(grid.start),
           0.0, 0.0, 0.0, tie, grid.start, -1, state0, (grid.start,))]
    best, explored = {}, 0
    while pq:
        _free_f, _f, free_distance, objective, distance, _, cell, heading_i, state, path = heapq.heappop(pq)
        key = _key(cell, heading_i, state, cfg)
        cost = (free_distance if free_first else 0., objective)
        if best.get(key, (math.inf, math.inf)) <= cost:
            continue
        best[key] = cost
        explored += 1
        if cfg.max_explored_states and explored > cfg.max_explored_states:
            return GridPlan(None, math.inf, math.inf, False, 0, pos_limit, yaw_limit, explored,
                            reason=f"search budget of {cfg.max_explored_states} states exhausted")
        goal_margin = cfg.goal_sigma_multiplier * position_sigma_max(state.P)
        at_goal = np.linalg.norm(grid.cell_xy(cell) - grid.goal_xy) + goal_margin <= grid.goal_radius + 1e-9
        if at_goal and state.localized and (
                state.resolved or not cfg.require_landmark_for_mode_resolution):
            return GridPlan(list(path), distance, objective, True, 0, pos_limit, yaw_limit,
                            explored, reason="strict", free_distance_m=free_distance)
        for mi, (dx, dy, L, head) in enumerate(grid.moves):
            if not grid.move_ok.get((cell[0], cell[1], mi)):
                continue
            nxt = (cell[0] + dx, cell[1] + dy)
            if len(path) >= 2 and nxt == path[-2]:
                continue                       # no immediate reversal
            tangent = np.array([math.cos(head), math.sin(head)])
            contact, normal, _standoff = best_contact(state.P, grid.contacts.get(nxt, ()))
            yaw_cmd = commanded_heading(grid, nxt, head, cfg)
            P, yaw_var, heading, resolved, localized, info = belief_step(
                state.P, state.yaw_var, state.heading, state.resolved, state.localized,
                L, tangent, yaw_cmd, contact, normal, cfg, pos_limit, yaw_limit, pre_pos, pre_yaw)
            if not info["ok"]:
                continue
            step_cost = L + cfg.turn_cost_per_rad * info["turn"]
            next_free = round(free_distance + free_cost[cell[0], cell[1], mi], 12)
            tie += 1
            free_priority = round(next_free + free_remaining.get(nxt, math.inf), 12) if free_first else 0.
            heapq.heappush(pq, (free_priority,
                                objective + step_cost + h(nxt), next_free, objective + step_cost,
                                distance + L, tie, nxt, mi,
                                _State(P, yaw_var, heading, resolved, localized), path + (nxt,)))
    return GridPlan(None, math.inf, math.inf, False, 0, pos_limit, yaw_limit, explored,
                    reason="no grid route satisfies the bounds")


def replay_grid(grid: TactileGrid, path, belief, cfg: UncertaintyConfig | None = None,
                yaw0: float | None = None, position_threshold_m: float = math.inf,
                yaw_threshold_rad: float = math.inf,
                probe_schedule: frozenset | None = None):
    """Replay a cell path; probes are (move index, 1). None = dense, empty = none."""
    cfg = cfg or grid.cfg
    _loc0, pre_pos, pre_yaw = _pre_limits(belief, cfg, position_threshold_m, yaw_threshold_rad)
    yaw_var = float(getattr(belief, "yaw_sigma", 0.0)) ** 2
    P = to_se2(belief.cov, yaw_var)
    heading = yaw0
    resolved = float(getattr(belief, "minor", 0.0)) <= 0.10
    localized = False
    allowed = None if probe_schedule is None else {i for i, _s in probe_schedule}
    rows, distance = [], 0.0
    for i, (a, b) in enumerate(zip(path[:-1], path[1:])):
        d = grid.cell_xy(b) - grid.cell_xy(a)
        L = float(np.linalg.norm(d))
        head = math.atan2(d[1], d[0])
        tangent = d / max(L, 1e-12)
        contact, normal, standoff = "none", None, None
        if allowed is None or i in allowed:
            contact, normal, standoff = best_contact(P, grid.contacts.get(b, ()))
        yaw_cmd = commanded_heading(grid, b, head, cfg)
        P, yaw_var, heading, resolved, localized, info = belief_step(
            P, yaw_var, heading, resolved, localized, L, tangent, yaw_cmd, contact, normal, cfg,
            position_threshold_m, yaw_threshold_rad, pre_pos, pre_yaw)
        distance += L
        row = trace_row(P, heading, info, edge=i, edge_kind="GRID", step=1, distance=distance,
                        point=grid.cell_xy(b), contact=contact, standoff=standoff,
                        localized=localized, pos_limit=position_threshold_m,
                        yaw_limit=yaw_threshold_rad)
        row["edge_index"] = i
        row["contact_normal"] = None if normal is None else np.asarray(normal).tolist()
        matching = [c for c in grid.contacts.get(b, ())
                    if normal is not None and np.allclose(c["normal"], normal)]
        row["contact_body"] = matching[0].get("body") if matching else None
        rows.append(row)
        if not info["ok"]:
            break
    return rows


class GridBackend:
    """The planner interface `tradeoff.run_sweep` drives."""
    name = "grid"

    def __init__(self, grid: TactileGrid):
        self.grid = grid

    def plan(self, belief, cfg, yaw0):
        return plan_grid(self.grid, belief, cfg, yaw0)

    def replay(self, path, belief, cfg, yaw0, schedule):
        return replay_grid(self.grid, path, belief, cfg, yaw0, probe_schedule=schedule)

    def length(self, path):
        return self.grid.path_length(path)

    def describe(self):
        g = self.grid
        return dict(planner="grid", grid_resolution_m=g.res, max_move_m=g.cfg.max_move_m,
                    contact_standoff_m=g.cfg.contact_standoff_m, contact_band_m=g.cfg.contact_band_m,
                    free_cells=int(g.free.sum()), touch_cells=len(g.contacts),
                    moves=len(g.moves), start_xy=g.cell_xy(g.start).tolist(),
                    goal_xy=g.goal_xy.tolist(), regions=g.region_cells())
