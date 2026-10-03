"""Known floorplan: rigid bodies, point features, and the route graph over them.

The map is KNOWN -- its shape is given.  Three things about it are not:

* where the robot is on it (the SE(2) pose);
* which corridor the robot is currently in (the route edge), because a junction
  offers more than one successor and a drifted heading can take the wrong one;
* how the drawing is registered to the world (a local rigid offset), because a
  floorplan is a drawing and drawings are a few centimetres wrong.

Perceptual aliasing is expressed here rather than bolted on: bodies and features
carry a ``signature``, which is what touch can distinguish.  Two corners sharing a
signature ARE the same measurement, so a filter that localizes through them has to
carry both hypotheses rather than pick one.  ``benchmark_floorplan`` puts two such
corners 3 m apart on the same corridor precisely so that the ambiguity is real.

Nothing here plans.  The route is a nominal SEQUENCE used to define what "wrong
turn" means; choosing an action is out of scope.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .geometry import ConvexPolygon, rect

__all__ = [
    "MapFeature",
    "RouteEdge",
    "Floorplan",
    "ProbeFan",
    "TactileStation",
    "benchmark_floorplan",
    "default_stations",
]


@dataclass(frozen=True)
class MapFeature:
    """A point landmark: a wall terminus, a convex corner, a pillar vertex.

    Touching one is a RANK-2 constraint (it pins both position components), against
    the rank-1 constraint of touching a featureless wall.  That step-change is why
    the association latent exists at all -- and why an ambiguous signature is
    expensive: it buys rank 2 only if you bet on the right candidate.
    """

    index: int
    xy: tuple[float, float]
    signature: str
    name: str = ""

    @property
    def position(self) -> np.ndarray:
        return np.asarray(self.xy, dtype=float).reshape(2)


@dataclass(frozen=True)
class RouteEdge:
    """One corridor the robot can be travelling down, as a directed centerline."""

    index: int
    a_xy: tuple[float, float]
    b_xy: tuple[float, float]
    half_width_m: float
    successors: tuple[int, ...] = ()
    name: str = ""

    @property
    def a(self) -> np.ndarray:
        return np.asarray(self.a_xy, dtype=float).reshape(2)

    @property
    def b(self) -> np.ndarray:
        return np.asarray(self.b_xy, dtype=float).reshape(2)

    @property
    def length_m(self) -> float:
        return float(np.linalg.norm(self.b - self.a))

    @property
    def tangent(self) -> np.ndarray:
        d = self.b - self.a
        return d / max(float(np.linalg.norm(d)), 1e-12)

    @property
    def bearing_rad(self) -> float:
        t = self.tangent
        return math.atan2(float(t[1]), float(t[0]))

    def project(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Along-edge coordinate and SIGNED lateral offset of (K,2) points."""
        p = np.asarray(points, dtype=float).reshape(-1, 2) - self.a
        t = self.tangent
        n = np.array([-t[1], t[0]])
        return p @ t, p @ n


@dataclass(frozen=True)
class Floorplan:
    bodies: tuple[ConvexPolygon, ...]
    features: tuple[MapFeature, ...]
    edges: tuple[RouteEdge, ...]
    route: tuple[int, ...]
    name: str = ""

    def __post_init__(self) -> None:
        for position, edge in enumerate(self.edges):
            if edge.index != position:
                raise ValueError("route edges must be indexed 0..N-1 in order")
            for s in edge.successors:
                if not 0 <= s < len(self.edges):
                    raise ValueError(f"edge {edge.index} names an unknown successor {s}")
        for position, feature in enumerate(self.features):
            if feature.index != position:
                raise ValueError("features must be indexed 0..N-1 in order")
        for a, b in zip(self.route, self.route[1:]):
            if b not in self.edges[a].successors:
                raise ValueError(f"nominal route step {a}->{b} is not an edge successor")

    # -- lookups -----------------------------------------------------------------

    def edge(self, index: int) -> RouteEdge:
        return self.edges[int(index)]

    def candidates(self, signature: str) -> tuple[tuple[str, int], ...]:
        """Map entities a contact with this signature could have come from.

        Point features win over bodies when both match: a corner is strictly more
        informative than the slab it belongs to.  An empty result means the map has
        nothing that could have produced this measurement -- the case
        ``update_contact`` must reject rather than force-fit.
        """
        points = tuple(("point", f.index) for f in self.features if f.signature == signature)
        if points:
            return points
        return tuple(("body", i) for i, b in enumerate(self.bodies) if b.signature == signature)

    def feature_positions(self, indices) -> np.ndarray:
        return np.array([self.features[int(i)].position for i in indices], dtype=float).reshape(-1, 2)

    # Associations range over BOTH point features and whole bodies, in one index
    # space: features first, then bodies.  "I touched the pillar" and "I touched the
    # pillar's north-west corner" are different claims of different rank, and the
    # belief has to be able to report either.

    @property
    def n_entities(self) -> int:
        return len(self.features) + len(self.bodies)

    def entity_id(self, kind: str, index: int) -> int:
        return int(index) if kind == "point" else len(self.features) + int(index)

    def entity_name(self, entity: int) -> str:
        e = int(entity)
        if e < len(self.features):
            return self.features[e].name or f"feature_{e}"
        return self.bodies[e - len(self.features)].name or f"body_{e}"

    def is_route_successor(self, a: int, b: int) -> bool:
        for u, v in zip(self.route, self.route[1:]):
            if u == int(a) and v == int(b):
                return True
        return False

    def translated(self, offset) -> "Floorplan":
        """The same plan drawn somewhere else -- how ground truth is built from a map."""
        d = np.asarray(offset, dtype=float).reshape(2)
        return Floorplan(
            bodies=tuple(b.translated(d) for b in self.bodies),
            features=tuple(
                MapFeature(f.index, tuple(f.position + d), f.signature, f.name)
                for f in self.features
            ),
            edges=tuple(
                RouteEdge(e.index, tuple(e.a + d), tuple(e.b + d), e.half_width_m,
                          e.successors, e.name)
                for e in self.edges
            ),
            route=self.route,
            name=self.name,
        )

    # -- the nominal route as a path ---------------------------------------------

    @property
    def route_length_m(self) -> float:
        return float(sum(self.edge(i).length_m for i in self.route))

    def route_pose(self, s_m: float) -> tuple[np.ndarray, float, int]:
        """Pose at arclength ``s`` along the NOMINAL route: ``(xy, heading, edge)``.

        Used to script ground truth and to place tactile stations.  It is a
        parameterisation of the map, not a plan: nothing consumes it at run time.
        """
        s = float(s_m)
        for i in self.route:
            edge = self.edge(i)
            if s <= edge.length_m or i == self.route[-1]:
                s = min(max(s, 0.0), edge.length_m)
                return edge.a + s * edge.tangent, edge.bearing_rad, i
            s -= edge.length_m
        raise AssertionError("unreachable")

    def bounds(self, pad: float = 0.3) -> tuple[float, float, float, float]:
        v = np.vstack([b.verts for b in self.bodies])
        return (float(v[:, 0].min()) - pad, float(v[:, 0].max()) + pad,
                float(v[:, 1].min()) - pad, float(v[:, 1].max()) + pad)


# ── tactile stations ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ProbeFan:
    """A sweep of the arm: rays from a body-frame origin, at body-frame bearings.

    One primitive yields both observation kinds.  A ray that touches something is a
    realized contact; a ray that reaches its limit having touched nothing certifies
    its whole length as swept-free.  Treating the second as a measurement rather
    than a failure is the point -- negative information is what kills a hypothesis
    that puts a wall where the arm just swept.
    """

    bearings_rad: tuple[float, ...]
    max_reach_m: float = 0.90
    origin_body: tuple[float, float] = (0.0, 0.0)
    label: str = ""


@dataclass(frozen=True)
class TactileStation:
    """Where the robot stops, and what it sweeps there.  Fully configurable."""

    s_m: float
    fans: tuple[ProbeFan, ...]
    label: str = ""


def _fan(name: str, degrees, reach: float) -> ProbeFan:
    return ProbeFan(tuple(math.radians(d) for d in degrees), reach, (0.0, 0.0), name)


# Reach 1.10 m so the +-60 deg rays still meet a wall 0.80 m away (0.80/sin 60 = 0.92 m).
# That matters: it is the DISAGREEMENT between the 60 deg and 90 deg rays that brackets a
# wall terminus, and a fan whose oblique rays all fall short degenerates to a single ray.
# See the report -- 1.10 m exceeds this platform's measured arm reach, and station
# placement (not the belief) is what has to absorb that.
LEFT_FAN = _fan("left", (60.0, 90.0, 120.0), reach=1.10)
RIGHT_FAN = _fan("right", (-60.0, -90.0, -120.0), reach=1.10)
AHEAD_FAN = _fan("ahead", (-20.0, 0.0, 20.0), reach=1.20)


# ── the benchmark scene ──────────────────────────────────────────────────────────


def benchmark_floorplan() -> Floorplan:
    """A staircase corridor with two turns, a decoy branch, a pillar, and aliasing.

    Layout (metres, corridor half-width 0.8, wall slabs 0.15 thick)::

                                          y=4.8 +--------------------------+
          corridor C (edge 4)                   |   -> -> -> -> -> goal    |
                                          y=3.2 +-----+--------------------+
                                                |  ^  |
          corridor B (edge 2)                   |  ^  |            J2 = (6,4) RIGHT turn
                                                |  ^  |
                             y=0.8  +-----+ +---+  ^  +   <- north wall has TWO gaps
          corridor A (edges 0,1)    |     | |       ^
                                    |  D  | |    o  ^      D = dead-end decoy alcove
                             y=-0.8 +-----+-+-------+      o = pillar (unique landmark)
                                   x=2.2  3.8   4.5 6.0    J1 = (6,0) LEFT turn

    Three properties are deliberate, and each one is a test:

    * the north wall TERMINATES at x=2.2 and again at x=5.2, and those two termini
      are geometrically identical -- same signature ``north_terminus``.  A robot
      that has drifted 3 m along the corridor cannot tell them apart by touch, so
      the belief must go bimodal instead of picking one;
    * the first terminus opens onto a DEAD END.  Turning there is the wrong turn
      the benchmark measures, and it is reachable: it is a legal successor of
      edge 0 in the route graph;
    * the pillar sits between them at x=4.5 and is the only body with signature
      ``pillar``.  It is the disambiguator, and it is off the centerline so that
      reaching it costs a deliberate probe.
    """
    hw, t = 0.8, 0.15
    bodies = (
        rect(-0.75, 6.95, -0.95, -hw, "south_A"),
        rect(-0.75, -0.60, -hw, hw, "west_cap"),
        rect(-0.75, 2.20, hw, hw + t, "north_A_west"),      # terminates at x=2.2
        rect(3.80, 5.20, hw, hw + t, "north_A_mid"),        # terminates at x=5.2
        rect(6.80, 6.95, -hw, 3.05, "east_AB"),
        rect(5.05, 5.20, hw + t, 4.80, "west_B"),
        rect(6.80, 11.15, 3.05, 3.20, "south_C"),
        rect(5.05, 11.15, 4.80, 4.95, "north_C"),
        rect(11.00, 11.15, 3.20, 4.80, "east_cap"),
        rect(2.05, 2.20, hw + t, 2.55, "west_D"),
        rect(3.80, 3.95, hw + t, 2.55, "east_D"),
        rect(2.20, 3.80, 2.40, 2.55, "north_D"),
        rect(4.30, 4.70, -0.65, -0.25, "pillar", signature="pillar"),
    )
    features = (
        # THE ALIASED PAIR -- same signature, 3.0 m apart, both on corridor A.
        MapFeature(0, (2.20, 0.80), "north_terminus", "decoy_opening_west"),
        MapFeature(1, (5.20, 0.80), "north_terminus", "junction_J1_opening_west"),
        MapFeature(2, (3.80, 0.80), "north_resume", "decoy_opening_east"),
        # The pillar's two front corners: co-located, so a 2-way gate that still
        # resolves the 3 m ambiguity.  A gate with >1 candidate must not deadlock.
        MapFeature(3, (4.30, -0.25), "pillar_corner", "pillar_nw"),
        MapFeature(4, (4.70, -0.25), "pillar_corner", "pillar_ne"),
        MapFeature(5, (6.80, 3.05), "east_terminus", "junction_J2_corner"),
    )
    edges = (
        RouteEdge(0, (0.0, 0.0), (3.0, 0.0), hw, (1, 3), "A_west"),
        RouteEdge(1, (3.0, 0.0), (6.0, 0.0), hw, (2,), "A_east"),
        RouteEdge(2, (6.0, 0.0), (6.0, 4.0), hw, (4,), "B_north"),
        RouteEdge(3, (3.0, 0.0), (3.0, 2.0), hw, (), "D_decoy"),
        RouteEdge(4, (6.0, 4.0), (11.0, 4.0), hw, (), "C_east"),
    )
    return Floorplan(bodies, features, edges, route=(0, 1, 2, 4), name="staircase_decoy")


def default_stations(spacing_m: float = 1.0, start_m: float = 0.5) -> tuple[TactileStation, ...]:
    """Stop every ``spacing_m`` and sweep left, right and ahead.

    Deliberately naive and route-agnostic: the belief must earn its accuracy from a
    dumb, evenly spaced sweep schedule.  Choosing WHERE to probe is the planner's
    job and is out of scope; ``TactileStation`` is the interface it would write to.
    """
    plan = benchmark_floorplan()
    stations, s = [], float(start_m)
    while s < plan.route_length_m:
        stations.append(TactileStation(s, (LEFT_FAN, RIGHT_FAN, AHEAD_FAN), f"s{len(stations)}"))
        s += float(spacing_m)
    return tuple(stations)


