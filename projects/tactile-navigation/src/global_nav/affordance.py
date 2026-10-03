"""The tactile-affordance graph: where the robot can go, and what it could feel there.

REACH IS THE ORGANIZING CONSTRAINT.  Batch 1's fans reached 1.10 m; the platform's
measured reach is ~0.45 m (hand_force_v2, INTERMITTENT_TACNAV_REPORT.md).  At
0.45 m, nothing is touchable from the corridor centerline of a 1.6 m corridor --
every touch requires being in a wall-adjacent lane or on a deliberate spur.  So
"where can I probe" stops being a detail of the sensor model and becomes graph
structure: lanes, spurs, and stations are first-class, each with a travel cost,
and probing is an action with a cost of its own.

ANALYTIC SIGNATURES.  The map is known, so what a probe at station S would return
is exact geometry, not a filter rollout: cast the station's rays against the map.
The planner's question is always comparative -- the robot navigates by one
hypothesis, and if another hypothesis is true it is PHYSICALLY displaced by the
inter-hypothesis offset when it stands "at" S.  So the value of a station against
a hypothesis pair is the disagreement between the ray-casts at S and at S+delta:
a station on a featureless wall scores zero (aliased -- rank 1 says so), a station
facing the pillar scores 1.0 (contact under one hypothesis, swept-free under the
other).  Negative information discriminates exactly as well as contact does, and
the planner gets that for free because absence-of-hit is a prediction too.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .floorplan import Floorplan, ProbeFan
from .geometry import raycast

__all__ = [
    "REACH_M", "LANE_OFFSET_M", "g1_fan", "SIDE_LEFT_G1", "SIDE_RIGHT_G1", "AHEAD_G1",
    "predict_rays", "ray_disagreement", "Station", "Node", "GraphEdge",
    "AffordanceGraph",
]

REACH_M = 0.45
"""Measured G1 probe reach (hand_force_v2).  Every Batch-2 fan is capped here."""

LANE_OFFSET_M = 0.28
"""Wall-adjacent lane: robot center to wall face.  Sized so the +-40 deg oblique
rays still reach the wall: 0.28 / sin(50 deg) = 0.366 m <= 0.45 m, leaving ~18%
margin for lateral tracking error.  The obliques matter because bracketing a wall
terminus IS the disagreement between adjacent rays."""

PROBE_ANCHOR_SIGMA_M = 0.14
"""What sigma_s an anchoring probe leaves behind, for PLANNING.  Measured from
the filter: a feature-bracketing station collapses along-route spread to about
2-4x the landmark sigma.  The planner predicting less would gate branches it
cannot actually earn; predicting more only costs an occasional extra probe."""


def g1_fan(name: str, degrees, reach: float = REACH_M) -> ProbeFan:
    if reach > REACH_M + 1e-9:
        raise ValueError(f"fan {name!r} reach {reach} exceeds the measured limit {REACH_M}")
    return ProbeFan(tuple(math.radians(d) for d in degrees), reach, (0.0, 0.0), name)


SIDE_LEFT_G1 = g1_fan("left_g1", (50.0, 90.0, 130.0))
SIDE_RIGHT_G1 = g1_fan("right_g1", (-50.0, -90.0, -130.0))
AHEAD_G1 = g1_fan("ahead_g1", (-25.0, 0.0, 25.0))
FANS = {f.label: f for f in (SIDE_LEFT_G1, SIDE_RIGHT_G1, AHEAD_G1)}


# ── analytic observation signatures ──────────────────────────────────────────────


def predict_rays(plan: Floorplan, xy, yaw: float, fan: ProbeFan,
                 offset=(0.0, 0.0)) -> list[dict]:
    """What each ray of ``fan`` would measure, cast from ``xy + offset`` at ``yaw``.

    ``offset`` is the hypothesis displacement: the robot BELIEVES it stands at
    ``xy``; if a rival hypothesis is true it physically stands at ``xy + offset``.
    Signature relabelling near features mirrors observations.simulate_fan, so the
    planner predicts what the sensor will actually report.
    """
    o = np.asarray(xy, float).reshape(2) + np.asarray(offset, float).reshape(2)
    out = []
    for bearing in fan.bearings_rad:
        a = yaw + bearing
        d, n, i = raycast(o, np.array([math.cos(a), math.sin(a)]), fan.max_reach_m,
                          plan.bodies)
        if d is None:
            out.append({"hit": False, "range": fan.max_reach_m, "normal": None,
                        "signature": None})
            continue
        p = o + d * np.array([math.cos(a), math.sin(a)])
        sig = plan.bodies[i].signature
        for f in plan.features:
            if float(np.linalg.norm(p - f.position)) <= 0.15:
                sig = f.signature
                break
        out.append({"hit": True, "range": float(d), "normal": n, "signature": sig})
    return out


def ray_disagreement(a: dict, b: dict) -> float:
    """How distinguishable two predicted ray outcomes are, in [0, 1].

    Contact-versus-free is maximal -- that is the swept-free channel doing planner
    work.  Signature mismatch is maximal too (pillar vs plain wall).  Range and
    normal thresholds sit at ~4x their sensor sigmas so the planner only counts on
    differences the filter can actually resolve.
    """
    if a["hit"] != b["hit"]:
        return 1.0
    if not a["hit"]:
        return 0.0
    if a["signature"] != b["signature"]:
        return 1.0
    if abs(a["range"] - b["range"]) > 0.12:
        return 0.8
    cosang = float(np.clip(np.dot(a["normal"], b["normal"]), -1.0, 1.0))
    if math.degrees(math.acos(cosang)) > 40.0:
        return 0.6
    return 0.0


@dataclass(frozen=True)
class Station:
    """A reachable stop-and-probe pose.  Everything about it is precomputable."""

    node: str
    xy: tuple[float, float]
    facing_yaw: float
    fan_name: str
    purpose: str = ""            # "feature" | "wall" | "mouth"

    def fan(self) -> ProbeFan:
        return FANS[self.fan_name]

    def predictions(self, plan: Floorplan, offset=(0.0, 0.0)) -> list[dict]:
        return predict_rays(plan, self.xy, self.facing_yaw, self.fan(), offset)

    def nominal(self, plan: Floorplan) -> dict:
        """The analytic signature at zero offset: per-ray (hit, range, normal,
        signature) plus the surfaces reachable from here."""
        rays = self.predictions(plan)
        return {"rays": rays,
                "reachable_surfaces": sorted({r["signature"] for r in rays if r["hit"]})}

    def discrimination(self, plan: Floorplan, delta, jitter_m: float = 0.15) -> float:
        """Disagreement between the ray-casts at S and at S + delta -- the WORST
        CASE over a jitter ball around delta.

        The min matters: hypothesis positions carry the mode's own spread
        (~0.12 m), and a raycast can flip exactly at a slab boundary.  Measured
        failure without it: a station at a gap edge read disagreement 1.0 at
        delta = 3.01 and 0.0 at 3.00, so the planner bet a branch commit on the
        third decimal of a Monte-Carlo mode estimate.  Discrimination the plan
        may rely on must survive the hypothesis being anywhere plausible.
        """
        d = np.asarray(delta, float).reshape(2)
        u = d / max(float(np.linalg.norm(d)), 1e-9)
        here = self.predictions(plan)
        worst = 1.0
        for shift in (-jitter_m, 0.0, jitter_m):
            there = self.predictions(plan, offset=d + shift * u)
            score = max((ray_disagreement(a, b) for a, b in zip(here, there)),
                        default=0.0)
            worst = min(worst, score)
        return worst

    def anchors_s(self, plan: Floorplan, tangent, probe_m: float = 0.40) -> bool:
        """Does the observation change along the route direction?  A station on a
        featureless wall does not (rank 1: sliding is free); one facing a terminus,
        box or pillar does.  This is the planner's word for 'this touch fixes
        along-route position'."""
        t = np.asarray(tangent, float).reshape(2)
        return (self.discrimination(plan, probe_m * t) >= 0.6
                or self.discrimination(plan, -probe_m * t) >= 0.6)


# ── the graph ────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Node:
    name: str
    xy: tuple[float, float]
    kind: str = "way"            # "way" | "station" | "junction" | "goal"

    @property
    def position(self) -> np.ndarray:
        return np.asarray(self.xy, float).reshape(2)


@dataclass(frozen=True)
class GraphEdge:
    a: str
    b: str
    length_m: float
    kind: str = "transit"        # "transit" | "wall" | "detour" | "branch"
    branch_junction: str | None = None
    """Set on the directed turn-in edge at a junction: traversing it COMMITS the
    robot to 'this physical opening is junction <name>'.  The belief-aware gate
    lives on exactly these edges."""


class AffordanceGraph:
    def __init__(self, plan: Floorplan):
        self.plan = plan
        self.nodes: dict[str, Node] = {}
        self.edges: list[GraphEdge] = []
        self.adj: dict[str, list[GraphEdge]] = {}
        self.stations: dict[str, Station] = {}

    # -- construction -------------------------------------------------------------

    def add_node(self, name: str, xy, kind: str = "way") -> None:
        self.nodes[name] = Node(name, tuple(np.asarray(xy, float)), kind)
        self.adj.setdefault(name, [])

    def connect(self, a: str, b: str, kind: str = "transit",
                branch_junction: str | None = None) -> None:
        """Add a<->b.  A branch tag applies to the a->b direction only: turning in
        commits; walking back out does not."""
        length = float(np.linalg.norm(self.nodes[a].position - self.nodes[b].position))
        fwd = GraphEdge(a, b, length, kind, branch_junction)
        rev = GraphEdge(b, a, length, kind, None)
        self.edges += [fwd, rev]
        self.adj[a].append(fwd)
        self.adj[b].append(rev)

    def add_station(self, node: str, facing_yaw: float, fan_name: str,
                    purpose: str) -> None:
        n = self.nodes[node]
        self.stations[node] = Station(node, n.xy, facing_yaw, fan_name, purpose)

    # -- queries ------------------------------------------------------------------

    def nearest_node(self, xy, kinds: tuple[str, ...] | None = None) -> str:
        p = np.asarray(xy, float).reshape(2)
        best, best_d = None, math.inf
        for n in self.nodes.values():
            if kinds and n.kind not in kinds:
                continue
            d = float(np.linalg.norm(n.position - p))
            if d < best_d:
                best, best_d = n.name, d
        return best

    def shortest(self, a: str, b: str, *, wall_weight: float = 1.0,
                 allow_branch: bool = True) -> tuple[list[str], float]:
        """Plain Dijkstra over nodes.  The baselines route with this; the
        belief-aware planner runs its own search over (node x belief)."""
        import heapq

        dist = {a: 0.0}
        prev: dict[str, str] = {}
        pq = [(0.0, a)]
        while pq:
            d, u = heapq.heappop(pq)
            if u == b:
                break
            if d > dist.get(u, math.inf):
                continue
            for e in self.adj[u]:
                if e.branch_junction is not None and not allow_branch:
                    continue
                w = e.length_m * (wall_weight if e.kind == "wall" else 1.0)
                nd = d + w
                if nd < dist.get(e.b, math.inf):
                    dist[e.b] = nd
                    prev[e.b] = u
                    heapq.heappush(pq, (nd, e.b))
        if b not in dist:
            return [], math.inf
        path = [b]
        while path[-1] != a:
            path.append(prev[path[-1]])
        return path[::-1], dist[b]

    def path_length(self, path: list[str]) -> float:
        return float(sum(np.linalg.norm(self.nodes[a].position - self.nodes[b].position)
                         for a, b in zip(path, path[1:])))

    def edge_between(self, a: str, b: str) -> GraphEdge | None:
        for e in self.adj[a]:
            if e.b == b:
                return e
        return None
