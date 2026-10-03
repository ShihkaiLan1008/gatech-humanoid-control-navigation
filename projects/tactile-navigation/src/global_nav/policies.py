"""Six ways to act on the same belief.  Only one of them asks what a probe is FOR.

Every policy runs the identical filter, executor, bump-and-slide competence and
budget; they differ ONLY in routing and in when they choose to probe.  That is
what makes the comparison an ablation of POLICY rather than of plumbing:

    shortest_astar   never probes.  What acting on the prior costs.
    fixed_interval   probes on a schedule along a wall lane.  Touch without
                     thought -- the control that separates "touching" from
                     "touching the right thing".
    max_wall         hugs walls and probes densely.  Maximal contact time.
    max_landmark     visits every feature station.  Maximal landmark count.
    belief_aware     the planner: near-shortest travel subject to being
                     CONFIDENT ENOUGH AT EACH BRANCH.  Probes are pure cost --
                     nothing rewards touching -- so it probes exactly when a
                     branch gate cannot otherwise be passed.
    oracle           truth-fed localization.  The physics ceiling.

The belief-aware search runs over (graph node x alive-hypotheses x sigma-bin).
A branch edge is traversable only when the minor-hypothesis mass is below delta
and sigma_s is below the junction's spacing-derived bound; a probe action
collapses exactly the hypothesis pairs whose ANALYTIC station signatures
disagree, and re-anchors sigma only where the observation actually varies along
the route (anchors_s).  The planner therefore inherits the physics: a station on
a featureless wall cannot reduce along-route uncertainty in the plan because it
cannot in the world.
"""

from __future__ import annotations

import heapq
import math
from collections import Counter

import numpy as np

from .affordance import PROBE_ANCHOR_SIGMA_M, AffordanceGraph, Station
from .episode import PROBE_COST_M, Action
from .managed import BeliefState
from .stressmaps import Scenario

__all__ = ["POLICIES", "make_policy"]

BRANCH_DELTA = 0.10          # max minor-hypothesis mass to commit a branch
DISCRIM_MIN = 0.60           # station resolves a pair if disagreement >= this
SBINS = (0.15, 0.30, 0.50, 0.80, 1.20, 2.00, 4.00)


def _sbin(sigma: float) -> int:
    for i, edge in enumerate(SBINS):
        if sigma <= edge:
            return i
    return len(SBINS) - 1


class BasePolicy:
    feeds_bumps = True
    uses_truth = False

    def __init__(self, sc: Scenario):
        self.sc = sc
        self.g: AffordanceGraph = sc.graph
        self.queue: list[Action] = []
        self.replans = 0
        self.finished = False

    # -- default queue-driven behaviour -------------------------------------------

    def next_action(self, ep) -> Action:
        if self.finished:
            return Action("done")
        if not self.queue:
            self.plan(ep)
            if not self.queue:
                self.finished = True
                return Action("done")
        return self.queue.pop(0)

    def on_result(self, ep, action: Action, status: str) -> None:
        if status in ("blocked", "stuck"):
            # skip the unreachable waypoint; replan from scratch when that empties us
            if not self.queue:
                self.replans += 1
                if self.replans > 3:
                    self.finished = True

    def plan(self, ep) -> None:
        raise NotImplementedError

    # -- shared helpers -----------------------------------------------------------

    def start_node(self, ep) -> str:
        xy, _ = ep.believed_pose()
        return self.g.nearest_node(xy)

    def route_to(self, ep, target_node: str, wall_weight: float = 1.0) -> list[str]:
        path, _ = self.g.shortest(self.start_node(ep), target_node,
                                  wall_weight=wall_weight)
        return path

    def push_moves(self, path: list[str], probe_at=(), label_prefix: str = "") -> None:
        for node in path[1:]:
            self.queue.append(Action("move", target_xy=self.g.nodes[node].xy,
                                     label=node))
            if node in probe_at:
                st = self.g.stations[node]
                self.queue.append(Action("probe", fan_name=st.fan_name,
                                         facing_yaw=st.facing_yaw,
                                         label=f"{label_prefix}{node}"))


# ── baselines ────────────────────────────────────────────────────────────────────


class ShortestPolicy(BasePolicy):
    """No-touch A*: commit to the MAP hypothesis and walk.  The wrong-turn rate
    this realizes is the number the whole benchmark exists to expose."""

    feeds_bumps = False

    def plan(self, ep) -> None:
        if self.replans == 0 or self.queue == []:
            path = self.route_to(ep, self.sc.goal_node)
            self.push_moves(path)
            if self.replans > 0:
                return
            self.replans += 1
        else:
            self.finished = True


class FixedIntervalPolicy(BasePolicy):
    """Wall-lane route, probe at every station passed.  Touch on a clock."""

    def __init__(self, sc: Scenario, interval_m: float = 1.0):
        super().__init__(sc)
        self.interval = interval_m
        self._planned = False

    def plan(self, ep) -> None:
        if self._planned:
            self.finished = True
            return
        self._planned = True
        path = self.route_to(ep, self.sc.goal_node, wall_weight=0.35)
        last = -1e9
        probe_at = []
        for node in path:
            if node in self.g.stations:
                x = self.g.nodes[node].position[0]
                if x - last >= self.interval - 1e-9:
                    probe_at.append(node)
                    last = x
        self.push_moves(path, probe_at=set(probe_at))


class MaxWallPolicy(FixedIntervalPolicy):
    """Maximum wall-adjacent distance: hug walls end to end, probe every station."""

    def __init__(self, sc: Scenario):
        super().__init__(sc, interval_m=0.0)

    def plan(self, ep) -> None:
        if self._planned:
            self.finished = True
            return
        self._planned = True
        path = self.route_to(ep, self.sc.goal_node, wall_weight=0.10)
        self.push_moves(path, probe_at={n for n in path if n in self.g.stations})


class MaxLandmarkPolicy(BasePolicy):
    """Visit every LANDMARK station (features and junction mouths), in map order,
    probing each; then the goal.  Maximal landmark count with no notion of which
    landmarks pay -- it touches the disambiguator eventually, along with every
    aliased mouth on the map."""

    def __init__(self, sc: Scenario):
        super().__init__(sc)
        self._planned = False

    def plan(self, ep) -> None:
        if self._planned:
            self.finished = True
            return
        self._planned = True
        stations = sorted(
            (n for n, st in self.g.stations.items() if st.purpose in ("feature", "mouth")),
            key=lambda n: self.g.nodes[n].position[0])
        here = self.start_node(ep)
        for target in stations:
            path, _ = self.g.shortest(here, target, allow_branch=False)
            self.push_moves(path, probe_at={target})
            here = target
        path, _ = self.g.shortest(here, self.sc.goal_node)
        self.push_moves(path)


class OraclePolicy(BasePolicy):
    """Truth-fed localization: the executor hands it ground truth as its belief.
    Upper bound on success, lower bound on distance."""

    feeds_bumps = False
    uses_truth = True

    def plan(self, ep) -> None:
        if self.replans == 0:
            self.replans += 1
            self.push_moves(self.route_to(ep, self.sc.goal_node))
        else:
            self.finished = True


# ── the belief-aware planner ─────────────────────────────────────────────────────


class BeliefAwarePolicy(BasePolicy):
    """Receding horizon: replan after every completed action, execute one step.

    NORMAL: graph search over (node, alive hypotheses, sigma bin); branch edges
    gated; probes inserted by the search only where a gate cannot otherwise pass.
    SUSPECT_LOST / RELOCALIZE: no branch may be committed -- go to the nearest
    s-anchoring station and probe until the belief earns NORMAL back.
    """

    def __init__(self, sc: Scenario):
        super().__init__(sc)
        self.spent: Counter = Counter()
        self.recovery_probes = 0
        self._anchor_cache: dict[str, bool] = {}
        self._disc_cache: dict[tuple[str, float], float] = {}

    def next_action(self, ep) -> Action:
        if self.finished:
            return Action("done")
        if ep.mb.state is not BeliefState.NORMAL:
            self.queue.clear()
            return self._recovery_action(ep)
        self.recovery_probes = 0
        # Receding horizon with HYSTERESIS: replan on information events (probe
        # fired, state change, blocked, hypothesis count changed), not on every
        # arrival.  Replanning per-arrival was measured to oscillate between two
        # near-equal-cost anchor plans as the sigma estimate jittered across a
        # bin edge -- the robot walked node-to-node between the two probe targets
        # without ever probing either.
        n_hyps = len(ep.mb.hypotheses())
        if n_hyps != getattr(self, "_last_n_hyps", -1):
            self.queue.clear()
            self._last_n_hyps = n_hyps
        # surprise -> verify: one unexplained contact does not make the belief
        # lost, but it does make the next probe worth its 0.7 m.  Without this
        # the minimal prober sails past the first evidence of a slip and the
        # SUSPECT streak never accumulates (measured: trigger rate 0.33 on
        # gauntlet_slip; the dense fixed-interval schedule triggered 1.0).
        low_streak = getattr(ep.mb, "_low_streak", 0)
        if low_streak >= 1 and low_streak > getattr(self, "_verified_at_streak", 0):
            self._verified_at_streak = low_streak
            self.queue.clear()
            self._fallback(ep, ep.mb.hypotheses())
        elif low_streak == 0:
            self._verified_at_streak = 0
        if not self.queue:
            self.plan(ep)
        # drop leading moves that are already satisfied -- a plan that starts at
        # the node the robot stands on must not emit a zero-length move, which
        # "arrives" instantly, clears the queue, and replans forever
        bel_xy, _ = ep.believed_pose()
        while self.queue and self.queue[0].kind == "move":
            head = self.queue[0]
            tol = head.tol_m if head.tol_m is not None else ep.cfg.arrive_tol_m
            if float(np.linalg.norm(np.asarray(head.target_xy) - bel_xy)) >= tol:
                break
            self.queue.pop(0)
        if not self.queue:
            self.finished = True
            return Action("done")
        act = self.queue.pop(0)
        if act.kind == "probe":
            self.spent[act.label] += 1
        return act

    def on_result(self, ep, action: Action, status: str) -> None:
        if action.kind == "probe" or status in ("blocked", "stuck"):
            self.queue.clear()                  # information event: fresh plan
        if status in ("blocked", "stuck"):
            self.replans += 1
            if self.replans > 12:
                self.finished = True
            else:
                # A wall the plan says is not there is evidence, not an obstacle:
                # probe the most informative reachable station before retrying.
                # Quitting here was measured to end episodes as wrong_stop before
                # the low-support streak could ever trigger relocalization.
                self._fallback(ep, ep.mb.hypotheses())

    # -- lost-belief recovery -----------------------------------------------------

    def _recovery_action(self, ep) -> Action:
        """Relocalization probing: aim at the stations that DISAMBIGUATE the
        reseeded cloud, not merely the nearest anchor.  The reseed band spans
        several alias basins of the junction furniture by construction; a probe
        that reads the same under a +-1.2 m shift teaches nothing, and recovery
        that only used nearest-anchor was measured to grind 30+ m re-confirming
        whichever basin it fell into first."""
        self.recovery_probes += 1
        if self.recovery_probes > 14:
            self.finished = True
            return Action("give_up")
        hyps = ep.mb.hypotheses()
        xy, _ = ep.believed_pose()
        here = self.g.nearest_node(xy)
        best, best_d = None, math.inf
        for name, st in self.g.stations.items():
            score = 0.5 if self._anchors(st) else 0.0
            if len(hyps) > 1:
                score += max(self._disc(st, h["s"] - hyps[0]["s"]) for h in hyps[1:])
            if score <= 0.0:
                continue
            _, d = self.g.shortest(here, name, allow_branch=False)
            # spread the recovery evidence: an alias explains any ONE local
            # contact, so re-probing the same station only re-confirms the alias
            d += 1.6 * self.spent[f"recover:{name}"] - 1.2 * score
            if d < best_d:
                best, best_d = name, d
        if best is None:
            return Action("give_up")
        node = self.g.nodes[best]
        if float(np.linalg.norm(np.asarray(node.xy) - xy)) > 0.10:
            path, _ = self.g.shortest(here, best, allow_branch=False)
            target = path[1] if len(path) > 1 else best
            return Action("move", target_xy=self.g.nodes[target].xy, label=target,
                          tol_m=0.08 if target == best else None)
        st = self.g.stations[best]
        self.spent[f"recover:{best}"] += 1
        return Action("probe", fan_name=st.fan_name, facing_yaw=st.facing_yaw,
                      label=f"recover:{best}")

    # -- planning-time belief model ----------------------------------------------

    def _anchors(self, st: Station) -> bool:
        if st.node not in self._anchor_cache:
            self._anchor_cache[st.node] = st.anchors_s(self.sc.plan, (1.0, 0.0))
        return self._anchor_cache[st.node]

    def _disc(self, st: Station, delta_s: float) -> float:
        key = (st.node, round(delta_s, 2))
        if key not in self._disc_cache:
            self._disc_cache[key] = st.discrimination(self.sc.plan, (delta_s, 0.0))
        return self._disc_cache[key]

    def _collapse(self, st: Station, mask: frozenset, hyps: list[dict]) -> frozenset:
        """Which hypotheses survive a probe here.  The planner assumes the MAP
        hypothesis survives; execution replans on whatever actually happened.

        Two mechanisms, matching two facts about the filter:
        * DISCRIMINATION kills a hypothesis whose predicted observation here
          differs from the MAP one's -- separation >= the jitter-robust floor.
        * An ANCHOR merges pseudo-hypotheses closer together than its own
          resolution: a wide cloud fragments into clusters a few tenths apart,
          nothing can 'discriminate' them because they are not really distinct
          places, and a rank-2 anchoring touch collapses them onto one s.
          Without this the planner had no move at all against a fragmented
          cloud: gates closed, no probe useful, search dead-ends.
        """
        ref = max(mask, key=lambda i: hyps[i]["mass"])
        seps = [abs(hyps[i]["s"] - hyps[ref]["s"]) for i in mask if i != ref]
        if seps and max(seps) <= 0.5 and self._anchors(st):
            return frozenset({ref})
        alive = {ref}
        for i in mask:
            if i != ref and self._disc(st, hyps[i]["s"] - hyps[ref]["s"]) < DISCRIM_MIN:
                alive.add(i)
        return frozenset(alive)

    def plan(self, ep) -> None:
        hyps = ep.mb.hypotheses()
        s_sigma = ep.mb.route_s_within()      # per-mode spread; the mask owns the rest
        start = self.start_node(ep)
        mask0 = frozenset(range(len(hyps)))
        drift = self.sc.drift_per_m

        # Dijkstra over (node, mask, sigma-bin); probes are extra edges
        start_state = (start, mask0, _sbin(s_sigma))
        dist: dict = {start_state: 0.0}
        prev: dict = {}
        pq = [(0.0, start_state, None)]
        goal_state = None
        while pq:
            d, state, via = heapq.heappop(pq)
            if d > dist.get(state, math.inf):
                continue
            node, mask, sb = state
            if node == self.sc.goal_node:
                goal_state = state
                break
            sigma = SBINS[sb]
            minor = 1.0 - max(hyps[i]["mass"] for i in mask) / max(
                sum(hyps[i]["mass"] for i in mask), 1e-12)

            for e in self.g.adj[node]:
                if e.branch_junction is not None:
                    if minor > BRANCH_DELTA or sigma > self.sc.sigma_branch_max_m:
                        continue                     # the gate
                ns = math.sqrt(sigma * sigma + (drift * e.length_m) ** 2)
                nstate = (e.b, mask, _sbin(ns))
                nd = d + e.length_m
                if nd < dist.get(nstate, math.inf):
                    dist[nstate] = nd
                    prev[nstate] = (state, ("move", e.b))
                    heapq.heappush(pq, (nd, nstate, None))

            if node in self.g.stations and self.spent[node] < 4:
                st = self.g.stations[node]
                new_mask = self._collapse(st, mask, hyps) if len(mask) > 1 else mask
                new_sigma = PROBE_ANCHOR_SIGMA_M if self._anchors(st) else sigma
                if new_mask != mask or _sbin(new_sigma) < sb:
                    ref = max(mask, key=lambda i: hyps[i]["mass"])
                    renav = sum(hyps[i]["mass"] * abs(hyps[i]["s"] - hyps[ref]["s"])
                                for i in mask - new_mask)
                    nstate = (node, new_mask, _sbin(new_sigma))
                    nd = d + PROBE_COST_M + renav
                    if nd < dist.get(nstate, math.inf):
                        dist[nstate] = nd
                        prev[nstate] = (state, ("probe", node))
                        heapq.heappush(pq, (nd, nstate, None))

        if goal_state is None:
            self._fallback(ep, hyps)
            return

        steps = []
        state = goal_state
        while state in prev:
            state, step = prev[state]
            steps.append(step)
        steps.reverse()
        for i, (kind, name) in enumerate(steps):
            if kind == "move":
                # tight tolerance when this move is the approach to a probe:
                # swept-free evidence lives in the last decimetre of reach
                next_is_probe = i + 1 < len(steps) and steps[i + 1][0] == "probe"
                self.queue.append(Action("move", target_xy=self.g.nodes[name].xy,
                                         label=name,
                                         tol_m=0.08 if next_is_probe else None))
            else:
                st = self.g.stations[name]
                self.queue.append(Action("probe", fan_name=st.fan_name,
                                         facing_yaw=st.facing_yaw, label=name))

    def _fallback(self, ep, hyps) -> None:
        """No gated path exists (all discriminators spent, or none reachable).
        Probe the best unspent station once; give up honestly if there is none."""
        xy, _ = ep.believed_pose()
        here = self.g.nearest_node(xy)
        options = []
        for name, st in self.g.stations.items():
            if self.spent[name] >= 3:
                continue
            score = 0.0
            if len(hyps) > 1:
                score = max(self._disc(st, h["s"] - hyps[0]["s"]) for h in hyps[1:])
            if self._anchors(st):
                score += 0.5
            if score <= 0.0:
                continue
            _, d = self.g.shortest(here, name, allow_branch=False)
            options.append((d - score, name))
        if not options:
            self.finished = True
            return
        _, best = min(options)
        path, _ = self.g.shortest(here, best, allow_branch=False)
        if len(path) <= 1:
            # already standing on the station: probe in place -- push_moves only
            # attaches probes to moves, and a fallback that pushes nothing reads
            # as 'finished' and quits mid-corridor (measured on gauntlet_slip)
            st = self.g.stations[best]
            self.queue.append(Action("probe", fan_name=st.fan_name,
                                     facing_yaw=st.facing_yaw, label=best))
        else:
            self.push_moves(path, probe_at={best})


POLICIES = {
    "shortest_astar": ShortestPolicy,
    "fixed_interval": FixedIntervalPolicy,
    "max_wall": MaxWallPolicy,
    "max_landmark": MaxLandmarkPolicy,
    "belief_aware": BeliefAwarePolicy,
    "oracle": OraclePolicy,
}


def make_policy(name: str):
    return POLICIES[name]
