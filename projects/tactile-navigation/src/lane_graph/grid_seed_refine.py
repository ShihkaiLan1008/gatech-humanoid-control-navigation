"""Geometric seeds, wall attraction, bounded belief repair, and shortcuts.

This is discrete path refinement on the same collision-checked grid as the
belief planner. Wall attraction only proposes paths: every accepted route
uses the shared belief_step model, including the source/pre-contact bounds
and terminal confidence disk. No continuous optimizer or changed noise model
is involved. Failed refinement can fall back to the original belief search.
"""
from __future__ import annotations

from dataclasses import dataclass
import heapq
import itertools
import math

import numpy as np

from .constrained import _State, _pre_limits, belief_step, position_sigma_max, to_se2
from .grid_constrained import (GridPlan, _key, best_contact, commanded_heading,
                               free_travel_distance)


def _turn(a, b):
    return 0. if a is None else abs((b - a + math.pi) % (2 * math.pi) - math.pi)


def _unwind(parents, key):
    path = [key[0]]
    while key in parents:
        key = parents[key]
        path.append(key[0])
    return list(reversed(path))


@dataclass
class _Candidate:
    path: list
    states: list                   # prior and each strictly valid prefix state
    feasible: bool
    distance: float
    objective: float
    free_distance: float


class _Refiner:
    def __init__(self, grid, belief, cfg, yaw0):
        self.grid, self.belief, self.cfg, self.yaw0 = grid, belief, cfg, yaw0
        self.pos, self.yaw = cfg.position_threshold_m, cfg.yaw_threshold_rad
        _, self.pre_pos, self.pre_yaw = _pre_limits(belief, cfg, self.pos, self.yaw)
        var = float(getattr(belief, "yaw_sigma", 0.)) ** 2
        self.prior = _State(to_se2(belief.cov, var), var, yaw0,
                             float(getattr(belief, "minor", 0.)) <= .10, False)
        self.free_first = cfg.route_objective == "free_distance_then_distance_and_rotation"
        self.edges, self.reverse, self.free_cost = {}, {}, {}
        self.stats = dict(geometric_expansions=0, repair_expansions=0,
                          seeds=0, repair_attempts=0, shortcuts_tried=0,
                          shortcuts_accepted=0, candidates_evaluated=0, fallback=False)
        self.cache = {}
        for (x, y, mi), ok in grid.move_ok.items():
            if not ok:
                continue
            dx, dy, length, head = grid.moves[mi]
            a, b = (x, y), (x + dx, y + dy)
            yaw = commanded_heading(grid, b, head, cfg)
            self.edges.setdefault(a, []).append((b, length, head, yaw, mi))
            self.reverse.setdefault(b, []).append((a, length))
        self.goals = {tuple(c) for c in np.argwhere(grid.free) if grid.is_goal(tuple(c))}
        # Distance-to-go includes obstacles and ignores belief/rotation. It
        # remains a lower bound for both geometric and repair search.
        self.remaining = self.distances(self.goals)

    def distances(self, goals):
        best = {c: 0. for c in goals}
        q = [(0., c) for c in goals]
        heapq.heapify(q)
        while q:
            cost, c = heapq.heappop(q)
            if cost != best[c]:
                continue
            for prev, length in self.reverse.get(c, ()):
                v = cost + length
                if v < best.get(prev, math.inf):
                    best[prev] = v
                    heapq.heappush(q, (v, prev))
        return best

    def free_length(self, a, b):
        key = a, b
        if key not in self.free_cost:
            self.free_cost[key] = free_travel_distance(self.grid, a, b)
        return self.free_cost[key]

    def geometric(self, start, goals, yaw0, wall_weight=0., remaining=None):
        """A* without belief constraints; positive weight attracts to walls.

        The attraction penalizes travel ending outside contact reach. Grid
        clearance prevents pulling a waypoint into or through a surface.
        Heading is retained so wall-facing turns have their actual cost.
        """
        remaining = self.distances(goals) if remaining is None else remaining
        if start not in remaining:
            return None
        heuristic = {c: self.grid.res * min(math.hypot(c[0] - g[0], c[1] - g[1]) for g in goals)
                     for c in remaining}
        serial = itertools.count()
        key0 = start, yaw0
        q = [(heuristic[start], 0., next(serial), key0)]
        best, parents = {key0: 0.}, {}
        while q:
            _, cost, _, key = heapq.heappop(q)
            if cost != best[key]:
                continue
            self.stats["geometric_expansions"] += 1
            cell, heading = key
            if cell in goals:
                return _unwind(parents, key)
            for nxt, length, _head, yaw, _mi in self.edges.get(cell, ()):
                if nxt not in remaining:
                    continue
                value = cost + length + self.cfg.turn_cost_per_rad * _turn(heading, yaw)
                value += wall_weight * length * (nxt not in self.grid.contacts)
                other = nxt, yaw
                if value + 1e-12 >= best.get(other, math.inf):
                    continue
                best[other], parents[other] = value, key
                heapq.heappush(q, (value + heuristic[nxt], value, next(serial), other))
        return None

    def advance(self, state, a, b):
        d = self.grid.cell_xy(b) - self.grid.cell_xy(a)
        length = float(np.linalg.norm(d))
        head = math.atan2(d[1], d[0])
        contact, normal, _ = best_contact(state.P, self.grid.contacts.get(b, ()))
        out = belief_step(state.P, state.yaw_var, state.heading, state.resolved, state.localized,
                          length, d / length, commanded_heading(self.grid, b, head, self.cfg),
                          contact, normal, self.cfg, self.pos, self.yaw, self.pre_pos, self.pre_yaw)
        return _State(*out[:5]), out[5]

    def at_goal(self, cell, state):
        return bool(state.localized and (state.resolved or not self.cfg.require_landmark_for_mode_resolution)
                    and np.linalg.norm(self.grid.cell_xy(cell) - self.grid.goal_xy)
                    + self.cfg.goal_sigma_multiplier * position_sigma_max(state.P)
                    <= self.grid.goal_radius + 1e-9)

    def evaluate(self, path):
        key = tuple(path)
        if key in self.cache:
            return self.cache[key]
        self.stats["candidates_evaluated"] += 1
        states = [self.prior]
        length = objective = free = 0.
        heading = self.yaw0
        prefix_ok = True
        for a, b in zip(path, path[1:]):
            d = self.grid.cell_xy(b) - self.grid.cell_xy(a)
            ds = float(np.linalg.norm(d))
            yaw = commanded_heading(self.grid, b, math.atan2(d[1], d[0]), self.cfg)
            length += ds
            objective += ds + self.cfg.turn_cost_per_rad * _turn(heading, yaw)
            free += self.free_length(a, b)
            heading = yaw
            if prefix_ok:
                state, info = self.advance(states[-1], a, b)
                prefix_ok = bool(info["ok"])
                if prefix_ok:
                    states.append(state)
        valid = prefix_ok and self.at_goal(path[-1], states[-1])
        result = _Candidate(path, states, valid, length, objective, free)
        self.cache[key] = result
        return result

    def cost(self, candidate):
        return (round(candidate.free_distance, 10) if self.free_first else 0., candidate.objective)

    def surface_rejoins(self, path):
        """Shortcut anchors bracketing an excursion from a reachable surface.

        Match body AND face normal: opposite sides of an obstacle are not a
        continuous wall. Ignore contact kind because a wall end can be tagged
        as a landmark on the same physical surface. Availability only proposes
        a bridge; full replay decides whether its contacts are sufficient.
        """
        runs = {}
        for i, cell in enumerate(path):
            faces = {(c["body"], tuple(np.round(c["normal"], 6)))
                     for c in self.grid.contacts.get(cell, ()) if c.get("body") is not None}
            for face in faces:
                groups = runs.setdefault(face, [])
                if groups and groups[-1][1] == i - 1:
                    groups[-1][1] = i
                else:
                    groups.append([i, i])
        pairs = set()
        for groups in runs.values():
            for (a, i), (j, b) in zip(groups, groups[1:]):
                # Also try anchors inside each contact run: arriving at its
                # boundary may leave too little room for the next turn.
                for pad in (0, 1, 2):
                    pairs.add((max(a, i - pad), min(b, j + pad)))
        return pairs

    def repair(self, candidate, backtrack_m, budget):
        """Preserve a valid prefix; search a corridor around the failed suffix.

        Backtracking creates an opportunity for contact BEFORE the first
        violation. A different wall/normal is selected by best_contact as
        each state propagates. The terminal test includes the confidence disk.
        """
        self.stats["repair_attempts"] += 1
        anchor = len(candidate.states) - 1
        distance = 0.
        while anchor > 0 and distance < backtrack_m:
            distance += float(np.linalg.norm(self.grid.cell_xy(candidate.path[anchor])
                                             - self.grid.cell_xy(candidate.path[anchor - 1])))
            anchor -= 1
        # The tube can reach nearby contact regions but cannot jump obstacles:
        # all expansions still use the precomputed collision-checked edges.
        suffix = np.array(candidate.path[anchor:])
        radius = max(1.2, 3 * self.cfg.contact_standoff_m) / self.grid.res
        allowed = {c for c in self.remaining
                   if np.min(np.sum((suffix - c) ** 2, axis=1)) <= radius ** 2}
        start, state0 = candidate.path[anchor], candidate.states[anchor]
        key0 = _key(start, -1, state0, self.cfg)
        serial = itertools.count()
        q = [(0., 2. * self.remaining.get(start, math.inf), next(serial), 0., 0., start, -1, state0, 0)]
        # Search nodes keep their own parents: covariance-bin replacement
        # must never splice a path onto the state of another arrival.
        nodes = [(start, None)]
        best, expanded = {}, 0
        while q and expanded < budget:
            _, _, _, free, cost, cell, mi0, state, node = heapq.heappop(q)
            key = key0 if node == 0 else _key(cell, mi0, state, self.cfg)
            value = (round(free, 10) if self.free_first else 0., cost)
            if best.get(key, (math.inf, math.inf)) <= value:
                continue
            best[key] = value
            expanded += 1
            self.stats["repair_expansions"] += 1
            if self.at_goal(cell, state):
                tail = []
                while node is not None:
                    c, node = nodes[node]
                    tail.append(c)
                return self.evaluate(candidate.path[:anchor] + list(reversed(tail)))
            previous = nodes[nodes[node][1]][0] if nodes[node][1] is not None else (
                candidate.path[anchor - 1] if anchor else None)
            for nxt, length, _head, _yaw, mi in self.edges.get(cell, ()):
                if nxt not in allowed or nxt == previous:
                    continue
                ns, info = self.advance(state, cell, nxt)
                if not info["ok"]:
                    continue
                nc = cost + length + self.cfg.turn_cost_per_rad * info["turn"]
                nf = free + (self.free_length(cell, nxt) if self.free_first else 0.)
                nk = _key(nxt, mi, ns, self.cfg)
                if best.get(nk, (math.inf, math.inf)) <= (round(nf, 10) if self.free_first else 0., nc):
                    continue
                nid = len(nodes)
                nodes.append((nxt, node))
                # Repairs seek a feasible suffix quickly; weighted A* is
                # deliberate here. The final shortcut pass improves cost.
                heapq.heappush(q, (round(nf, 10) if self.free_first else 0.,
                                  nc + 2. * self.remaining[nxt], next(serial), nf, nc, nxt, mi, ns, nid))
        return None

    def shorten(self, candidate):
        """Try geometric bridges, keeping only fully feasible improvements."""
        tried = set()
        while self.stats["shortcuts_tried"] < self.cfg.seed_shortcut_max_trials:
            path = candidate.path
            points = np.array([self.grid.cell_xy(c) for c in path])
            cumulative = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
            turns = np.r_[0., np.cumsum([_turn(a.heading, b.heading)
                                        for a, b in zip(candidate.states, candidate.states[1:])])]
            rejoins = []
            for i, j in self.surface_rejoins(path):
                if (path[i], path[j]) in tried:
                    continue
                saving = (cumulative[j] - cumulative[i]
                          + self.cfg.turn_cost_per_rad * (turns[j] - turns[i])
                          - float(np.linalg.norm(points[j] - points[i])))
                if saving > 1e-9:
                    rejoins.append((-saving, i, j))
            # Include turns/contact changes and a regular sample along long
            # walls. Long detours are attempted first; accept by route_objective.
            indices = set(range(0, len(path), 3)) | {len(path) - 1}
            for i in range(1, len(path) - 1):
                if (_turn(candidate.states[i].heading, candidate.states[i + 1].heading) > 1e-9
                        or (path[i] in self.grid.contacts) != (path[i + 1] in self.grid.contacts)):
                    indices.add(i)
            indices = sorted(indices)
            buckets = [[], [], []]
            for i in indices:
                for j in indices:
                    if j < i + 3 or (path[i], path[j]) in tried:
                        continue
                    saving = cumulative[j] - cumulative[i] - float(np.linalg.norm(points[j] - points[i]))
                    if saving > self.grid.res / 2:
                        span = cumulative[j] - cumulative[i]
                        buckets[0 if span <= 1.5 else 1 if span <= 3. else 2].append((-saving, i, j))
            # Interleave local and long bridges so large blocked detours do
            # not consume the entire budget before small improvements are tried.
            generic = [p for group in itertools.zip_longest(*(sorted(b) for b in buckets))
                       for p in group if p is not None]
            # Try same-surface returns before spending the budget on generic
            # bridges. A long route can otherwise hide a useful local rejoin
            # behind hundreds of higher-ranked, infeasible open-space bridges.
            pairs = sorted(rejoins) + generic
            changed = False
            for _, i, j in pairs:
                if self.stats["shortcuts_tried"] >= self.cfg.seed_shortcut_max_trials:
                    break
                if (path[i], path[j]) in tried:
                    continue
                tried.add((path[i], path[j]))
                self.stats["shortcuts_tried"] += 1
                bridge = self.geometric(path[i], {path[j]}, candidate.states[i].heading)
                if bridge is None or bridge == path[i:j + 1]:
                    continue
                other = self.evaluate(path[:i] + bridge + path[j + 1:])
                if other.feasible and self.cost(other) < self.cost(candidate):
                    candidate = other
                    self.stats["shortcuts_accepted"] += 1
                    tried.clear()  # a changed prefix changes later uncertainty
                    changed = True
                    break
            if not changed:
                break
        return candidate


def plan_seed_refine(grid, belief, cfg, yaw0=None):
    if cfg.seed_repair_max_states < 0 or cfg.seed_shortcut_max_trials < 0:
        raise ValueError("seed/refine budgets must be nonnegative")
    refiner = _Refiner(grid, belief, cfg, yaw0)
    seeds, feasible = [], []
    if refiner.goals:
        # Aim geometric seeds near the centre; merely entering the goal disk
        # leaves no room for uncertainty. Repair can terminate at any valid cell.
        errors = {c: float(np.linalg.norm(grid.cell_xy(c) - grid.goal_xy)) for c in refiner.goals}
        nearest = min(errors.values())
        goals = {c for c, error in errors.items() if error <= nearest + 1e-9}
        remaining = refiner.distances(goals)
        # Zero is the unconstrained geometric solution. Increasing attraction
        # supplies alternatives without hard-coding any map or contact body.
        for weight in (0., 1., 2., 4., 8., 16.):
            path = refiner.geometric(grid.start, goals, yaw0, weight, remaining)
            if path is None or any(path == c.path for c in seeds):
                continue
            candidate = refiner.evaluate(path)
            seeds.append(candidate)
            refiner.stats["seeds"] += 1
            if candidate.feasible:
                feasible.append(candidate)
                # A feasible geometric solution already minimizes its seed
                # cost; other wall seeds still matter for the free-first objective.
                if weight == 0. and not refiner.free_first:
                    break
    if not feasible:
        # Prefer candidates that retain more of their route under strict
        # replay. Try short and longer backtracking within a shared budget.
        ranked = sorted(seeds, key=lambda c: (-(len(c.states) - 1) / max(1, len(c.path) - 1),
                                               refiner.cost(c)))
        attempts = [(c, back) for c in ranked[:3] for back in (1.2, 3.)]
        for attempt, (candidate, backtrack) in enumerate(attempts):
            left = cfg.seed_repair_max_states - refiner.stats["repair_expansions"]
            if left <= 0:
                break
            repaired = refiner.repair(candidate, backtrack, max(1, left // (len(attempts) - attempt)))
            if repaired is not None and repaired.feasible:
                feasible.append(repaired)
                break
    if feasible:
        best = refiner.shorten(min(feasible, key=refiner.cost))
        return GridPlan(best.path, best.distance, best.objective, True, 0,
                        cfg.position_threshold_m, cfg.yaw_threshold_rad,
                        refiner.stats["repair_expansions"], reason="strict seed/refine",
                        free_distance_m=best.free_distance, strategy="seed_refine",
                        strategy_stats=refiner.stats)
    if cfg.seed_fallback:
        from .grid_constrained import plan_grid
        result = plan_grid(grid, belief, cfg.merged(route_strategy="belief_astar"), yaw0)
        refiner.stats["fallback"] = True
        result.strategy_stats = refiner.stats
        result.reason = "seed/refine fallback: " + result.reason
        result.explored_states += refiner.stats["repair_expansions"]
        return result
    return GridPlan(None, math.inf, math.inf, False, 0,
                    cfg.position_threshold_m, cfg.yaw_threshold_rad,
                    refiner.stats["repair_expansions"], reason="seed/refine found no feasible candidate (fallback disabled)",
                    strategy="seed_refine", strategy_stats=refiner.stats)
