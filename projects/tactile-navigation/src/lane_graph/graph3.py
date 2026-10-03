"""The V3 route graph: TOUCH_FOLLOW edges carry a LANE POLYLINE.

An edge is not a straight hop between stations -- it is a stretch of base
lane the robot rides continuously, touching before every increment.  The
search is Dijkstra over (node, sigma-bin, resolved) with the same no-negative
-cost discipline V2 needed (a discounted landmark edge is a negative cycle);
what changes is the pricing: TOUCH_FOLLOW is only mildly more expensive per
metre than transit and it BOUNDS sigma growth, so on any route long enough
for drift to matter the touch lane wins on its own merits -- exactly the
"prefer a wall lane when maintenance is useful, cross open space when it is
not" rule, expressed as arithmetic rather than a switch.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field

import numpy as np

from planning.geometric import OccupancyGrid, RouteConfig, plan_route_on

from .maps_v3 import LANE_CLEAR, TOUCH_PITCH, Lane, clearance, in_rooms
from .scene3 import V3Scene

SIGMA_GATE = 0.12
DRIFT_PER_M = 0.045
RISK_PER_M = 10.0        # price of CARRYING uncertainty, per metre per metre
TRANSIT_PENALTY_PER_M = 1.5
"""Standing cost of UNREFERENCED travel, per metre.

Not an uncertainty term -- a mission term.  Belief arithmetic alone stops
valuing touch the moment sigma dips under the commit gate, and the planner
then crosses open floor for the rest of the route; but every V1/V2 episode
that died, died on an unreferenced stretch (2.9 touchless metres let an
arrival anchor inherit an error it could not press through; 0.3 m of
unreferenced drift clipped a doorway jamb).  A tactile robot values staying
referenced even while it is momentarily confident, so open-space travel is
worth about 1.7x wall-referenced travel to this planner.  Transit remains
available wherever no lane is reachable -- which is the spec's own rule."""
TOUCH_SIGMA_FLOOR = 0.075     # what per-increment touching holds sigma near
CHUNK_M = 1.5                 # lane split length: fine enough that a route can
                              # join a lane where it MEETS it, not 2.6 m later
MAX_TRANSIT_M = 2.6           # FREE_TRANSIT is a CONNECTOR, not a mode
FREE_GRID_M = 1.15            # open-space waypoints so long crossings chain

GEOM_DETOUR_MAX = 1.9
"""How far round an obstacle a BENT connector may go, as a multiple of the
straight-line distance it replaces.

The straight-hop test (`_clear_seg`) answers one question -- is the segment
clear -- and where the answer is no it produces no edge at all.  That is the
`build_graph3` gap PF_AUDIT.md 6 names: a doorway narrower than the 1.15 m
waypoint grid receives no waypoint, so nothing routes through it, and a
connector that would need to bend 0.3 m round a jamb is simply absent.  The
geometric layer answers the harder question -- what is the shortest clear PATH
-- and hands back a polyline the executive walks.

Bounded, because FREE_TRANSIT is still a connector and not a travel mode: a
route that has to go 1.9x the straight line is not a connector, it is a
detour, and the planner should be made to buy it as a sequence of edges whose
carried risk it pays for per metre."""


@dataclass(frozen=True)
class Node3:
    nid: int
    xy: tuple
    kind: str                 # start|goal|lane|door


@dataclass
class Edge3:
    eid: int
    a: int
    b: int
    kind: str                 # TOUCH_FOLLOW|FREE_TRANSIT
    length: float
    poly: np.ndarray | None = None      # lane polyline a->b (TOUCH_FOLLOW)
    normal: np.ndarray | None = None    # per-point unit normal toward the wall
    informative: bool = False           # the lane passes a wall FEATURE


@dataclass
class Graph3:
    scene: V3Scene
    nodes: list = field(default_factory=list)
    edges: list = field(default_factory=list)
    adj: dict = field(default_factory=dict)

    def add_node(self, xy, kind) -> int:
        nid = len(self.nodes)
        self.nodes.append(Node3(nid, (float(xy[0]), float(xy[1])), kind))
        self.adj.setdefault(nid, [])
        return nid

    def add_edge(self, a, b, kind, poly=None, normal=None):
        if poly is not None:
            L = float(np.sum(np.linalg.norm(np.diff(poly, axis=0), axis=1)))
        else:
            L = float(np.linalg.norm(np.asarray(self.nodes[a].xy)
                                     - np.asarray(self.nodes[b].xy)))
        e = Edge3(len(self.edges), a, b, kind, L, poly, normal)
        self.edges.append(e)
        self.adj[a].append(e.eid)
        self.adj[b].append(e.eid)
        return e


def _clear_seg(spec, p, q, clear=LANE_CLEAR, step=0.06) -> bool:
    p, q = np.asarray(p, float), np.asarray(q, float)
    n = max(2, int(np.linalg.norm(q - p) / step) + 1)
    pts = p[None, :] + np.linspace(0, 1, n)[:, None] * (q - p)[None, :]
    return bool(np.all(clearance(pts, spec) >= clear)
                and np.all(in_rooms(pts, spec)))


def _clear_poly(spec, poly, clear=LANE_CLEAR, step=0.06) -> bool:
    """`_clear_seg`, applied to every leg of a polyline.

    The occupancy grid answers at CELL resolution: `line_is_clear` rounds each
    sample to a 0.08 m cell, so a route it accepts can still graze ~0.06 m
    inside the band on the continuous clearance function.  A bent connector has
    to meet the same contract a straight one does, verified the same way, or
    "the layer is additive" would be true of the edge list and false of the
    thing the robot walks.
    """
    poly = np.asarray(poly, float)
    return all(_clear_seg(spec, a, b, clear, step)
               for a, b in zip(poly[:-1], poly[1:]))


def _transitable(spec, p) -> bool:
    """May a connector START or END here?  A lane node can sit closer to the
    wall it follows than LANE_CLEAR (that is what a lane is for), and no
    straight connector is offered from such a node either -- `_clear_seg`
    samples its endpoints too.  Bent connectors inherit the same rule."""
    q = np.asarray(p, float).reshape(1, 2)
    return bool(clearance(q, spec)[0] >= LANE_CLEAR and in_rooms(q, spec)[0])


def transit_grid(spec) -> OccupancyGrid:
    """The free-space the BASE may travel, sampled once for a whole map.

    Clearance is the wall/bulge/protrusion distance INSIDE the declared rooms
    and zero outside them.  Keeping `in_rooms` is deliberate: the room list is
    the only statement this map format makes about where the building ends, so
    dropping it to "free space is wherever no wall is near" would let a route
    leave through the gap between two wall segments.  What the geometric layer
    removes is the 1.15 m waypoint pitch, not the boundary.
    """
    def clr(pts):
        pts = np.asarray(pts, float)
        return np.where(in_rooms(pts, spec), clearance(pts, spec), 0.0)

    xs = [v for r in spec.rooms for v in (r[0], r[2])]
    ys = [v for r in spec.rooms for v in (r[1], r[3])]
    pad = 2.0 * RouteConfig.cell_m
    bounds = (min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad)
    # LANE_CLEAR, not the RouteConfig default: a connector must be held to the
    # SAME clearance the straight-hop test used, or the new layer would refuse
    # segments the shipped maps already walk.
    return OccupancyGrid(clr, bounds, RouteConfig(robot_clearance_m=LANE_CLEAR))


def _lane_normals(ln: Lane) -> np.ndarray:
    """Unit vectors from each lane point toward the wall it follows."""
    d = ln.face - ln.lane
    n = np.linalg.norm(d, axis=1, keepdims=True)
    return d / np.maximum(n, 1e-9)


def build_graph3(scene: V3Scene, geometric_transit: bool = True) -> Graph3:
    """The route graph.  ``geometric_transit`` adds the bent connectors of
    `planning.geometric`; pass False to reproduce the pre-wiring hop chain."""
    spec = scene.spec
    g = Graph3(scene)
    g.add_node(scene.start_map_xy, "start")     # 0
    g.add_node(scene.goal_map_xy, "goal")       # 1

    for ln in scene.lanes:
        nrm = _lane_normals(ln)
        seglen = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(ln.lane, axis=0),
                                                     axis=1))]
        nchunk = max(1, int(round(ln.length / CHUNK_M)))
        bounds = np.linspace(0.0, seglen[-1], nchunk + 1)
        idx = [int(np.argmin(np.abs(seglen - b))) for b in bounds]
        nids = [g.add_node(ln.lane[i], "lane") for i in idx]
        for k in range(nchunk):
            i0, i1 = idx[k], idx[k + 1]
            if i1 - i0 < 2:
                continue
            e = g.add_edge(nids[k], nids[k + 1], "TOUCH_FOLLOW",
                           poly=ln.lane[i0:i1 + 1].copy(),
                           normal=nrm[i0:i1 + 1].copy())
            # does this stretch pass a wall feature?  (bulge cap, grown
            # protrusion, or a recess edge -- the along-axis evidence)
            seg_face = ln.face[i0:i1 + 1]
            feat = False
            for bg in spec.bulges:
                if np.min(np.linalg.norm(seg_face - np.asarray(bg.center),
                                         axis=1)) <= bg.radius + 0.25:
                    feat = True
            for pr in spec.prots:
                if np.min(np.linalg.norm(seg_face - np.asarray(pr.center),
                                         axis=1)) <= pr.half_along + 0.45:
                    feat = True
            gaps = np.linalg.norm(np.diff(seg_face, axis=0), axis=1)
            if len(gaps) and float(np.max(gaps)) > 0.12:
                feat = True                      # a recess breaks the face line
            e.informative = feat

    for x, y, lab in spec.doors:
        for off in ((0, 0), (0, .5), (0, -.5), (.5, 0), (-.5, 0)):
            p = np.array([x + off[0], y + off[1]])
            if float(clearance(p[None, :], spec)[0]) >= LANE_CLEAR \
                    and bool(in_rooms(p[None, :], spec)[0]):
                g.add_node(p, "door")
                break

    # open-space waypoints: transit is capped at connector length (the spec's
    # own rule -- transit exists where no wall lane is reachable), so long
    # crossings must chain through free-space nodes and thereby pay the
    # carried-risk cost per metre instead of arriving as one cheap diagonal
    xs = [v for r in spec.rooms for v in (r[0], r[2])]
    ys = [v for r in spec.rooms for v in (r[1], r[3])]
    gx = np.arange(min(xs), max(xs) + 1e-9, FREE_GRID_M)
    gy = np.arange(min(ys), max(ys) + 1e-9, FREE_GRID_M)
    gp = np.array([[x, y] for x in gx for y in gy])
    if len(gp):
        okg = (clearance(gp, spec) >= 0.42) & in_rooms(gp, spec)
        have = [np.asarray(n.xy) for n in g.nodes]
        for p in gp[okg]:
            if min((float(np.linalg.norm(p - h)) for h in have), default=9) > 0.75:
                g.add_node(p, "free")
                have.append(p)

    N = len(g.nodes)
    grid = transit_grid(spec) if geometric_transit else None
    ok = [_transitable(spec, n.xy) for n in g.nodes] if grid is not None else []
    for a in range(N):
        for b in range(a + 1, N):
            pa, pb = g.nodes[a].xy, g.nodes[b].xy
            L = float(np.linalg.norm(np.asarray(pa) - np.asarray(pb)))
            if L < 0.12 or L > MAX_TRANSIT_M:
                continue
            if any(g.edges[e].kind == "TOUCH_FOLLOW"
                   and {g.edges[e].a, g.edges[e].b} == {a, b} for e in g.adj[a]):
                continue
            if _clear_seg(spec, pa, pb):
                g.add_edge(a, b, "FREE_TRANSIT")
            elif grid is not None and ok[a] and ok[b]:
                # blocked straight, but perhaps not blocked.  Strictly additive:
                # every edge the straight test produced is untouched, and a bent
                # connector appears only where there was no edge at all.
                r = plan_route_on(grid, pa, pb)
                if r.found and r.length_m <= GEOM_DETOUR_MAX * L \
                        and _clear_poly(spec, r.points):
                    g.add_edge(a, b, "FREE_TRANSIT", poly=r.points)
    return g


@dataclass
class Belief3:
    sigma: float
    minor: float


def edge_cost(e: Edge3, bs: Belief3, sigma_in: float, resolved: bool):
    """No negative costs anywhere; touching earns its keep by BOUNDING sigma,
    which the decision charges then reward."""
    if e.kind == "TOUCH_FOLLOW":
        n_press = max(1, int(e.length / TOUCH_PITCH))
        cost = e.length * 1.15 + 0.10 * n_press
        # per-increment touching does not merely BOUND sigma, it tightens it:
        # ~3 presses per metre on a shaped wall is the whole premise of the
        # mode, so a lane is an investment that pays downstream
        # a flat stretch bounds the PERPENDICULAR error but not the
        # along-wall coordinate (Batch 1's rank-1 result); only a lane that
        # passes a wall FEATURE pins the along axis and truly shrinks sigma
        shrink = 0.55 if getattr(e, "informative", False) else 0.85
        sigma = max(TOUCH_SIGMA_FLOOR, sigma_in * shrink)
        # a lane long enough to read a whole wall feature also resolves the
        # registration: several tens of presses on a shaped wall is what the
        # multi-hypothesis prior cannot survive
        res = resolved or (e.length >= 2.0)
    else:
        cost = e.length * (1.0 + TRANSIT_PENALTY_PER_M)
        sigma = math.hypot(sigma_in, DRIFT_PER_M * e.length)
        res = resolved
    # CARRIED RISK: uncertainty is not only charged where a decision lands --
    # it is expensive over every metre it is carried, because that is what
    # actually costs an episode (the belief that arrives 0.5 m off).  Without
    # this term a straight 8 m diagonal beat every wall lane and the planner
    # never touched anything (measured on R2).
    cost += RISK_PER_M * max(0.0, 0.5 * (sigma + sigma_in) - SIGMA_GATE) \
        * e.length
    return cost, sigma, res


def astar3(g: Graph3, src: int, dst: int, bs: Belief3):
    def key(nid, sigma, res):
        return (nid, min(int(sigma / 0.03), 14), res)

    pq = [(0.0, 0, src, bs.sigma, bs.minor <= 0.10, ())]
    best, tie = {}, 1
    while pq:
        c, _, nid, sigma, res, path = heapq.heappop(pq)
        if nid == dst:
            return list(path), c + max(0.0, sigma - SIGMA_GATE) * 25.0 \
                + (0.0 if res else bs.minor * 20.0)
        k = key(nid, sigma, res)
        if k in best and best[k] <= c:
            continue
        best[k] = c
        for eid in g.adj[nid]:
            e = g.edges[eid]
            nxt = e.b if e.a == nid else e.a
            dc, s2, r2 = edge_cost(e, bs, sigma, res)
            if g.nodes[nxt].kind in ("door", "goal"):
                dc += max(0.0, s2 - SIGMA_GATE) * 8.0
                if not r2:
                    dc += bs.minor * 2.0
            tie += 1
            heapq.heappush(pq, (c + dc, tie, nxt, s2, r2, path + (eid,)))
    return None, math.inf
