"""The tactile-aware global route graph (V2 Subtask 2).

Not a shortest-path grid: nodes are decision places (start, goal, doorway
mouths, lane endpoints, landmark stances) and edges are TYPED --

    FREE_TRANSIT     straight collision-free hops through open space
    TOUCH_FOLLOW     the wall-offset stance lanes (the green stripes of the
                     approval figures), where maintenance touches are cheap
    LANDMARK_DETOUR  a stance facing a landmark whose contact/free answer
                     separates the CURRENT registration hypotheses

A* is only the solver; the cost is belief-aware and re-evaluated at plan
time from the live belief summary:

    travel            edge length (transit walks a bit faster than lanes)
    predicted sigma   along-drift grown by the motion model over distance
                      walked since the last informative touch, charged at
                      every DECISION node (door/landmark/goal) that the
                      commit gate would refuse
    touch starvation  distance beyond touch_gap_m without a touch
    probing           ~0.9 m-equivalent per expected press on a lane
    landmark value    a NEGATIVE term: minor-mass x discrimination, only
                      while the belief actually carries a rival hypothesis

Replanning after every informative tactile update is the executive's job;
the graph is rebuilt never, re-searched cheaply.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field

import numpy as np

from .maps import MapSpec, followable_stretches
from .scene2 import V2Scene

LANE_OFF = 0.56
DRIFT_PER_M = 0.045          # the belief motion model's along-drift
SIGMA_GATE = 0.12
TOUCH_GAP_M = 1.0
PRESS_COST_M = 0.9
TOUCH_PITCH_M = 0.40


@dataclass(frozen=True)
class NodeV2:
    nid: int
    xy: tuple[float, float]
    kind: str                       # start|goal|door|lane|landmark


@dataclass(frozen=True)
class EdgeV2:
    eid: int
    a: int
    b: int
    kind: str                       # FREE_TRANSIT|TOUCH_FOLLOW|LANDMARK_DETOUR
    length: float
    face_normal: tuple[float, float] | None = None   # toward the wall (touch)
    landmark: str | None = None


@dataclass
class RouteGraph:
    scene: V2Scene
    nodes: list[NodeV2] = field(default_factory=list)
    edges: list[EdgeV2] = field(default_factory=list)
    adj: dict = field(default_factory=dict)

    def add_node(self, xy, kind) -> int:
        nid = len(self.nodes)
        self.nodes.append(NodeV2(nid, (float(xy[0]), float(xy[1])), kind))
        self.adj.setdefault(nid, [])
        return nid

    def add_edge(self, a, b, kind, face_normal=None, landmark=None):
        L = float(np.linalg.norm(np.asarray(self.nodes[a].xy)
                                 - np.asarray(self.nodes[b].xy)))
        e = EdgeV2(len(self.edges), a, b, kind, L,
                   tuple(face_normal) if face_normal is not None else None,
                   landmark)
        self.edges.append(e)
        self.adj[a].append(e.eid)
        self.adj[b].append(e.eid)


def _clear_segment(spec: MapSpec, p, q, clearance=0.28, step=0.05) -> bool:
    from .maps import _clearance, _in_rooms
    p, q = np.asarray(p, float), np.asarray(q, float)
    n = max(2, int(np.linalg.norm(q - p) / step) + 1)
    pts = p[None, :] + np.linspace(0, 1, n)[:, None] * (q - p)[None, :]
    return bool(np.all(_clearance(pts, spec) >= clearance)
                and np.all(_in_rooms(pts, spec)))


def build_graph(scene: V2Scene) -> RouteGraph:
    spec = scene.spec
    g = RouteGraph(scene)
    g.add_node(scene.start_map_xy, "start")
    g.add_node(scene.goal_map_xy, "goal")

    # TOUCH_FOLLOW lanes from the computed followable stretches: endpoint
    # nodes on the lane line, one edge per stretch (subdivided ~2 m so
    # transit can join mid-lane)
    for s0, s1 in followable_stretches(spec):
        s0, s1 = np.asarray(s0), np.asarray(s1)
        u = s1 - s0
        L = float(np.linalg.norm(u))
        u = u / max(L, 1e-9)
        n_wall = np.array([-u[1], u[0]])
        # face points (s0/s1 sit 0.02 off the face): the LANE is 0.56 out.
        # Determine the outward side: the followable computation guarantees
        # the lane side; probe both and keep the clear one.
        for sgn in (+1.0, -1.0):
            lane0 = s0 + sgn * (LANE_OFF - 0.02) * n_wall
            lane1 = s1 + sgn * (LANE_OFF - 0.02) * n_wall
            if _clear_segment(spec, lane0, lane1):
                break
        else:
            continue
        nseg = max(1, int(L // 2.0))
        pts = [lane0 + (lane1 - lane0) * k / nseg for k in range(nseg + 1)]
        nids = [g.add_node(p, "lane") for p in pts]
        for a, b in zip(nids, nids[1:]):
            g.add_edge(a, b, "TOUCH_FOLLOW", face_normal=-sgn * n_wall)

    # doorway nodes (transit waypoints at the labeled door anchors, pulled
    # to the free side)
    for x, y, lab in spec.doors:
        from .maps import _clearance as _cl
        for off in ((0, 0), (0, 0.45), (0, -0.45), (0.45, 0), (-0.45, 0)):
            p = np.array([x + off[0], y + off[1]])
            if _cl(p[None, :], spec)[0] >= 0.28:
                g.add_node(p, "door")
                break

    # landmark stance nodes: 0.32 + r off each pillar/cabinet face toward
    # free space
    # each landmark gets up to THREE clear ring stances: one is the probe
    # stance, and together they are the rounding waypoints that let transit
    # pass a wall-grown obstacle (a single stance left M1 disconnected --
    # the lane-break endpoints could not link straight past the column)
    for i, p in enumerate(spec.pillars):
        c = np.asarray(p.center, float)
        kept = 0
        for ang in np.linspace(0, 2 * math.pi, 16, endpoint=False):
            d = np.array([math.cos(ang), math.sin(ang)])
            stance = c + (p.radius + 0.34) * d
            if _clear_segment(spec, stance, stance) and kept < 3:
                g.add_node(stance, "landmark")
                kept += 1
    for i, bx in enumerate(spec.boxes):
        c = np.asarray(bx.center, float)
        kept = 0
        for d in (np.array([0, 1.0]), np.array([0, -1.0]),
                  np.array([1.0, 0]), np.array([-1.0, 0]),
                  np.array([0.707, 0.707]), np.array([-0.707, 0.707])):
            stance = c + d * (np.asarray(bx.half) + 0.36)
            if _clear_segment(spec, stance, stance) and kept < 3:
                g.add_node(stance, "landmark")
                kept += 1

    # FREE_TRANSIT: connect every node pair whose straight segment is clear
    # (bounded length keeps the graph sparse and the hops short)
    N = len(g.nodes)
    for a in range(N):
        for b in range(a + 1, N):
            pa, pb = g.nodes[a].xy, g.nodes[b].xy
            L = float(np.linalg.norm(np.asarray(pa) - np.asarray(pb)))
            if L < 0.15 or L > 4.5:
                continue
            already = any(g.edges[e].kind == "TOUCH_FOLLOW"
                          and {g.edges[e].a, g.edges[e].b} == {a, b}
                          for e in g.adj[a])
            if already:
                continue
            if _clear_segment(spec, pa, pb):
                lm = (g.nodes[a].kind == "landmark"
                      or g.nodes[b].kind == "landmark")
                g.add_edge(a, b, "LANDMARK_DETOUR" if lm else "FREE_TRANSIT")
    return g


@dataclass
class BeliefSummary:
    """What route choice needs from the live belief."""
    sigma_along: float
    minor_mass: float
    dist_since_touch: float
    hyp_separation_m: float = 0.55


def edge_cost(g: RouteGraph, e: EdgeV2, bs: BeliefSummary,
              sigma_in: float, gap_in: float, resolved: bool):
    """Cost + outgoing (sigma, gap) for traversing e given incoming state."""
    sigma = math.hypot(sigma_in, DRIFT_PER_M * e.length)
    gap = gap_in + e.length
    cost = e.length * (1.0 if e.kind == "FREE_TRANSIT" else 1.15)
    if e.kind == "TOUCH_FOLLOW":
        n_press = max(1, int(e.length / TOUCH_PITCH_M))
        # touching resets drift growth laterally and the starvation clock;
        # along-sigma only resets at EDGES (landmarks) -- the V1 lesson --
        # so lanes cap sigma growth rather than erase it
        cost += 0.25 * PRESS_COST_M * n_press
        sigma = min(sigma, math.hypot(sigma_in, 0.5 * DRIFT_PER_M * e.length))
        gap = min(gap, TOUCH_PITCH_M)
    # NO negative costs: landmark value is a one-shot STATE CHANGE
    # (resolved=True strips wrong-mode charges from every later decision),
    # never a discount -- a discounted landmark edge is a negative cycle
    # and the search never terminates (measured: v1 of this file hung).
    if e.kind == "LANDMARK_DETOUR" and not resolved \
            and (bs.minor_mass > 0.10 or sigma > SIGMA_GATE):
        cost += PRESS_COST_M * 2.0               # the disambiguating fan
        sigma = 0.06
        gap = 0.0
        resolved = True
    # touch starvation: strong enough that a touch-capable lane beats a
    # touchless shortcut once the gap is real -- at 0.8 the planner walked
    # 2.9 touchless metres to the goal and the arrival anchor inherited a
    # 0.66 m error it could not press through (attempt 7)
    cost += max(0.0, gap - TOUCH_GAP_M) * 2.5
    return cost, sigma, gap, resolved


def astar(g: RouteGraph, start_nid: int, goal_nid: int,
          bs: BeliefSummary):
    """Dijkstra over (node, sigma-bin, gap-bin); returns edge id list."""
    def key(nid, sigma, gap, res):
        return (nid, min(int(sigma / 0.03), 12), min(int(gap / 0.5), 8), res)

    resolved0 = bs.minor_mass <= 0.10
    pq = [(0.0, 0, start_nid, bs.sigma_along, bs.dist_since_touch,
           resolved0, ())]
    best = {}
    tie = 1
    while pq:
        c, _, nid, sigma, gap, res, path = heapq.heappop(pq)
        if nid == goal_nid:
            # goal decision: unresolved spread AND unresolved wrong-mode
            # risk are both charged -- a route may not arrive confidently
            # wrong (V1's no_touch failure mode, priced into the search)
            return list(path), (c + max(0.0, sigma - SIGMA_GATE) * 25.0
                                + (0.0 if res else bs.minor_mass * 20.0))
        k = key(nid, sigma, gap, res)
        if k in best and best[k] <= c:
            continue
        best[k] = c
        for eid in g.adj[nid]:
            e = g.edges[eid]
            nxt = e.b if e.a == nid else e.a
            dc, s2, g2, r2 = edge_cost(g, e, bs, sigma, gap, res)
            # decision-node uncertainty charges: where a wrong belief
            # becomes a wrong commitment
            if g.nodes[nxt].kind in ("door", "goal", "landmark"):
                dc += max(0.0, s2 - SIGMA_GATE) * 8.0
            if g.nodes[nxt].kind == "door" and not r2:
                dc += bs.minor_mass * 2.0
            tie += 1
            heapq.heappush(pq, (c + dc, tie, nxt, s2, g2, r2, path + (eid,)))
    return None, math.inf
