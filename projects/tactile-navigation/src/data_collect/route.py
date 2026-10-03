"""The planned route, frozen once into a reference centreline.

THE PLANNER RUNS ONCE, HERE, AND NEVER AGAIN.  Section 1 of the collection
spec asks for base poses sampled along "the existing Room 2 planned route",
and section 9 forbids the planner any part in data collection.  Both hold if
the route is computed a single time at start-up, written into the run
manifest as a fixed polyline, and consulted afterwards only as geometry --
which is what `RouteRef` is.  Nothing downstream of this module imports
`lane_graph.graph3`; `tests/data_collect/test_no_planner_in_the_loop.py`
pins that.

What the reference has to answer, per sampled arc length `s`:

    p(s)        where on the centreline
    t(s)        which way the route runs there
    n(s)        the perpendicular the +-0.20 m jitter runs along
    yaw(s)      the heading the base holds
    region(s)   horizontal / corner / vertical

`n(s)` and `yaw(s)` are decided by the wall the route is following, found by
casting a ray from the centreline in both perpendicular directions against
the MAP's own bodies -- the map the robot carries, not a simulator read.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from global_nav.geometry import raycast

RESAMPLE_M = 0.05
"""Centreline node spacing.  The planner emits 0.05 m lane polylines and
1.0 m transit hops in the same route; resampling makes arc length uniform so
`s ~ U(0, L)` is uniform on the ROUTE rather than on its node list."""

CORNER_TURN_RAD = math.radians(30.0)
"""Heading change across a +-`CORNER_WINDOW_M` window that marks a turn node."""

CORNER_WINDOW_M = 0.15
CORNER_MERGE_M = 1.20
"""Turn nodes this close belong to ONE turn.

The room2 route does not round its corner in a single sweep: it jogs 0.23 m
BACKWARD off the west lane at s = 2.35 (a 148 deg segment between the lane's
last node and the corner-cut edge's first), runs 1.0 m of 44 deg diagonal, and
straightens onto the north lane at s = 3.67.  Read as three separate turns
that is three corner regions covering 55 % of the route; read as one it is the
90 deg turn a human sees."""

CORNER_PAD_M = 0.30
CORNER_MIN_WIDTH_M = 1.00
"""How much route counts as "the corner".

The turn's own extent plus `CORNER_PAD_M` each side, widened to at least
`CORNER_MIN_WIDTH_M` so a sharp kink still gets a neighbourhood.  Once the
centreline hugs the corner the turn itself is short -- a 0.24 m fillet -- so
the span is what this constant makes it, and narrow is right: the spec
oversamples the corner *because* its geometry is informative, and spreading
that 20 % over 1.8 m put most of it on plain wall a metre from the vertex.

At 1.00 m the whole corner quota lands inside s in [2.80, 3.80] on room2 --
0.5 m of west face and 0.45 m of north face either side of the elbow, both
within the arm's reach, so a touch anywhere in the region can land within
0.11-0.16 m of the vertex.  Measured before narrowing: the nearest contact to
the vertex was 0.63 m."""

WALL_SEARCH_M = 3.0

VERTEX_ALONG_M = 0.45
"""How far along each face a corner-VERTEX pose may stand from the vertex.

The centreline TURNS at the corner, so no pose on it ever stands alongside the
vertex on either face: the elbow is 0.656 m from it and the hand, leading
+-0.35 m along the face, reached y = 3.487 on the west face and x = -0.415 on
the north.  The last ~0.14 m of each face went untouched.

A pose `a` metres from the vertex along one face, at the corner standoff, puts
the hand anywhere in `a +- 0.35` along that face -- so `a` up to 0.45 covers
the vertex itself and overlaps the region the centreline already reaches.
What bounds it from below is not the standoff but the OTHER wall: facing one
face, the base's own footprint (0.28 m) is what limits how close to the corner
it can stand, because the carried hands point along the face it is facing."""


@dataclass(frozen=True)
class CornerVertex:
    """A corner of the map, and the two faces that meet at it."""
    xy: np.ndarray
    normals: tuple          # outward normal of each face
    tangents: tuple         # along each face, pointing AWAY from the vertex
    bodies: tuple

    def as_dict(self) -> dict:
        return {"xy_map": [round(float(v), 5) for v in self.xy],
                "faces": [{"body": b,
                           "normal": [round(float(x), 5) for x in n],
                           "tangent": [round(float(x), 5) for x in t]}
                          for b, n, t in zip(self.bodies, self.normals,
                                             self.tangents)]}


@dataclass(frozen=True)
class RoutePose:
    """One sampled base pose, in MAP coordinates."""
    s_m: float
    lateral_offset_m: float
    region: str
    xy_map: np.ndarray
    yaw_map: float
    route_xy_map: np.ndarray
    route_tangent: np.ndarray
    route_normal: np.ndarray          # unit, points AWAY from the followed wall
    wall_dist_m: float                # centreline -> followed face, along -n
    stand_dist_m: float               # this pose -> the first surface AHEAD
    wall_body: str

    def as_dict(self) -> dict:
        return {
            "s_m": round(float(self.s_m), 4),
            "lateral_offset_m": round(float(self.lateral_offset_m), 4),
            "region": self.region,
            "xy_map": [round(float(v), 5) for v in self.xy_map],
            "yaw_map": round(float(self.yaw_map), 5),
            "route_xy_map": [round(float(v), 5) for v in self.route_xy_map],
            "route_tangent": [round(float(v), 5) for v in self.route_tangent],
            "route_normal": [round(float(v), 5) for v in self.route_normal],
            "route_wall_dist_m": round(float(self.wall_dist_m), 4),
            # the distance a REACH has to cover: cast along the pose's own
            # facing direction, not the centreline's perpendicular.  At a
            # corner the two differ by up to sqrt(2), and it is this one that
            # decides whether a touch can land.
            "reach_standoff_m": round(float(self.stand_dist_m), 4),
            "followed_body": self.wall_body,
        }


@dataclass
class RouteRef:
    """A resampled route polyline plus everything sampling asks of it."""

    xy: np.ndarray                     # (N, 2) map frame, uniform spacing
    bodies: tuple
    s: np.ndarray = field(init=False)
    tangent: np.ndarray = field(init=False)
    normal: np.ndarray = field(init=False)     # away from the followed wall
    wall_dist: np.ndarray = field(init=False)
    wall_body: list = field(init=False)
    region_of_node: list = field(init=False)
    corner_centres_m: list = field(init=False)
    corner_spans_m: list = field(init=False)

    def __post_init__(self):
        p = np.asarray(self.xy, float)
        seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
        self.s = np.concatenate([[0.0], np.cumsum(seg)])
        d = np.gradient(p, axis=0)
        n = np.linalg.norm(d, axis=1, keepdims=True)
        self.tangent = d / np.maximum(n, 1e-12)
        self._spans = None
        self._resolve_walls()
        self._classify()

    # ── construction ─────────────────────────────────────────────────────────

    @classmethod
    def from_polyline(cls, poly, bodies, *, resample_m: float = RESAMPLE_M):
        p = _dedupe(np.asarray(poly, float))
        return cls(xy=_resample(p, float(resample_m)), bodies=tuple(bodies))

    @property
    def length_m(self) -> float:
        return float(self.s[-1])

    # ── which wall is this stretch following ────────────────────────────────

    def _resolve_walls(self) -> None:
        """The followed FACE and its outward normal, from the MAP.

        Not the route's own perpendicular.  On a straight leg the two agree,
        but where the route turns a corner the perpendicular sweeps through the
        diagonal, and a pose facing the diagonal of a concave corner is
        `sqrt(2)` times its perpendicular standoff away from anything it could
        touch: measured on the corner elbow, 0.46 m from each face and 0.59 m
        along its own reach axis, against an arm that reaches 0.48 m.  Every
        pose in the turn was therefore a guaranteed miss -- which is what left
        the vertex empty even after the centreline was brought in to hug it.

        The face normal is the gradient of the min-signed-distance field, so
        the yaw snaps from square-to-one-wall to square-to-the-other at the
        bisector, and every pose faces a surface the way
        `wall_parallel_touch` faces one: `yaw = atan2(-n_y, -n_x)` on the
        FACE's normal.  `normal` points away from that face, so a positive
        lateral offset always means "further into the room".
        """
        nrm, dist, body = [], [], []
        for p in self.xy:
            d, n, k = _nearest_face(p, self.bodies)
            nrm.append(n)
            dist.append(d)
            body.append(self.bodies[k].name if k is not None else "")
        self.normal = np.asarray(nrm, float)
        self.wall_dist = np.asarray(dist, float)
        self.wall_body = body

    # ── horizontal / corner / vertical ──────────────────────────────────────

    def _classify(self) -> None:
        """A turn is a heading change, not a hard-coded waypoint.

        Heading change across a +-`CORNER_WINDOW_M` window marks turn nodes;
        nodes within `CORNER_MERGE_M` belong to one turn (see that constant --
        room2's route takes its 90 deg in three pieces).  Each merged turn
        becomes a corner span, and everything outside every span is horizontal
        or vertical by whichever map axis its tangent lies closer to -- the
        two names the spec uses for room2's north and west legs, and a rule
        that keeps meaning something on a room with more than one turn."""
        th = np.arctan2(self.tangent[:, 1], self.tangent[:, 0])
        step = max(self._step(), 1e-9)
        half = max(1, int(round(CORNER_WINDOW_M / step)))
        turning = np.zeros(len(th), dtype=bool)
        for i in range(len(th)):
            a, b = max(0, i - half), min(len(th) - 1, i + half)
            turning[i] = abs(_wrap(th[b] - th[a])) >= CORNER_TURN_RAD
        clusters = _merge(_runs(turning), self.s, CORNER_MERGE_M)
        spans, centres = [], []
        for lo, hi in clusters:
            a, b = float(self.s[lo]), float(self.s[hi - 1])
            c = 0.5 * (a + b)
            w = max(b - a + 2 * CORNER_PAD_M, CORNER_MIN_WIDTH_M)
            spans.append((c - 0.5 * w, c + 0.5 * w))
            centres.append(c)
        self.corner_centres_m = centres
        self.corner_spans_m = spans
        reg = []
        for i, si in enumerate(self.s):
            if any(a <= si <= b for a, b in spans):
                reg.append("corner")
            else:
                t = self.tangent[i]
                reg.append("horizontal" if abs(t[0]) >= abs(t[1])
                           else "vertical")
        self.region_of_node = reg

    def _step(self) -> float:
        return float(np.median(np.diff(self.s))) if len(self.s) > 1 else 1.0

    # ── queries ──────────────────────────────────────────────────────────────

    def index_of(self, s_m: float) -> int:
        """The node AT OR BEFORE `s_m`.

        `searchsorted` with its default `side="left"` returns the node AFTER
        an interior arc length, which made `pose_at` interpolate backwards
        from it (a negative fraction) and `region_at` answer with the next
        node's region -- so an `s` drawn inside the vertical leg's own span
        could come back labelled "corner"."""
        return int(np.clip(np.searchsorted(self.s, float(s_m), side="right") - 1,
                           0, len(self.s) - 1))

    def region_at(self, s_m: float) -> str:
        return self.region_of_node[self.index_of(s_m)]

    def spans(self) -> dict:
        """Arc-length intervals per region, as `[(s_lo, s_hi), ...]`.

        Cached: a pose draw asks for this, and a run makes hundreds of
        thousands of pose draws over a node list that never changes."""
        if getattr(self, "_spans", None) is not None:
            return self._spans
        out: dict = {}
        cur, lo = self.region_of_node[0], self.s[0]
        for i in range(1, len(self.s)):
            if self.region_of_node[i] != cur:
                out.setdefault(cur, []).append((float(lo), float(self.s[i])))
                cur, lo = self.region_of_node[i], self.s[i]
        out.setdefault(cur, []).append((float(lo), float(self.s[-1])))
        self._spans = out
        return out

    def region_length_m(self, region: str) -> float:
        return float(sum(b - a for a, b in self.spans().get(region, ())))

    def sample_s(self, rng, region: str | None = None) -> float:
        """`s ~ U(0, L)`, restricted to one region's spans when asked."""
        if region is None:
            return float(rng.uniform(0.0, self.length_m))
        spans = self.spans().get(region, ())
        if not spans:
            raise ValueError(f"route has no {region!r} stretch")
        w = np.array([b - a for a, b in spans], float)
        k = int(rng.choice(len(spans), p=w / w.sum()))
        return float(rng.uniform(*spans[k]))

    def pose_at(self, s_m: float, lateral_offset_m: float,
                yaw_mode: str = "face_wall") -> RoutePose:
        """Centreline point pushed `lateral_offset_m` along `+n` (away from
        the followed wall), with the heading the spec fixes.

        `face_wall` is the default and it is the heading this controller's
        press actually needs: `wall_parallel_touch` faces the surface
        (`yaw = atan2(-n_y, -n_x)`) and creeps the tip along body +x, so the
        reach axis and the surface normal are the same line.  Facing ALONG
        the route instead puts the wall on the body's +-y, where the trained
        lateral band tops out at 0.35 m against a centreline standing 0.50 m
        off it -- a dataset of misses.  `along_route` is kept because the
        spec's words allow reading it that way and an ablation may want it."""
        i = self.index_of(s_m)
        j = min(i + 1, len(self.s) - 1)
        frac = 0.0 if j == i else float(
            (float(s_m) - self.s[i]) / max(self.s[j] - self.s[i], 1e-12))
        p = self.xy[i] + frac * (self.xy[j] - self.xy[i])
        t = self.tangent[i]
        n = self.normal[i]
        xy = p + float(lateral_offset_m) * n
        if yaw_mode == "along_route":
            yaw = math.atan2(t[1], t[0])
        elif yaw_mode == "face_wall":
            yaw = math.atan2(-n[1], -n[0])
        else:
            raise ValueError(f"unknown yaw_mode {yaw_mode!r}")
        d0 = float(self.wall_dist[i])
        ahead, _n, _k = raycast(xy, np.array([math.cos(yaw), math.sin(yaw)]),
                                WALL_SEARCH_M, self.bodies)
        return RoutePose(
            s_m=float(s_m), lateral_offset_m=float(lateral_offset_m),
            region=self.region_of_node[i], xy_map=xy, yaw_map=float(yaw),
            route_xy_map=p, route_tangent=t.copy(), route_normal=n.copy(),
            wall_dist_m=d0,
            stand_dist_m=(float(ahead) if ahead is not None else float("inf")),
            wall_body=self.wall_body[i])

    # ── the corner vertices themselves ───────────────────────────────────────

    def corner_vertices(self) -> list:
        """Where the faces on either side of each corner actually meet.

        Recovered from the route: the node before the turn follows one face at
        a known standoff, the node after follows another, and a face is the
        line `{x : (x - p)·n = -d}`.  Two such lines meet at the vertex."""
        out = []
        for lo, hi in getattr(self, "corner_spans_m", ()):
            i = self.index_of(lo)
            j = self.index_of(hi)
            a, b = self.wall_body[i], self.wall_body[j]
            if not a or not b or a == b:
                continue
            na, nb = self.normal[i], self.normal[j]
            if abs(float(np.cross(na, nb))) < 0.3:
                continue
            m = np.stack([na, nb])
            rhs = np.array([float(na @ self.xy[i]) - float(self.wall_dist[i]),
                            float(nb @ self.xy[j]) - float(self.wall_dist[j])])
            try:
                v = np.linalg.solve(m, rhs)
            except np.linalg.LinAlgError:
                continue
            # along each face, pointing back down its own leg (away from the
            # vertex): the component of that leg's node offset along the face
            ts = []
            for n, q in ((na, self.xy[i]), (nb, self.xy[j])):
                t = np.array([-n[1], n[0]])
                ts.append(t if float(t @ (q - v)) > 0 else -t)
            out.append(CornerVertex(xy=v, normals=(na, nb),
                                    tangents=tuple(ts), bodies=(a, b)))
        return out

    def vertex_pose(self, vertex, face: int, along_m: float,
                    lateral_offset_m: float, standoff_m: float,
                    yaw_mode: str = "face_wall") -> RoutePose:
        """A pose standing `along_m` from the vertex, square to one face.

        Parameterised exactly as a route pose is -- an along-coordinate and a
        lateral offset from a nominal standoff -- so the same lateral-bin
        histogram, the same validity check and the same records apply."""
        n = np.asarray(vertex.normals[face], float)
        t = np.asarray(vertex.tangents[face], float)
        d = float(standoff_m) + float(lateral_offset_m)
        xy = np.asarray(vertex.xy, float) + float(along_m) * t + d * n
        yaw = (math.atan2(t[1], t[0]) if yaw_mode == "along_route"
               else math.atan2(-n[1], -n[0]))
        ahead, _n, _k = raycast(xy, np.array([math.cos(yaw), math.sin(yaw)]),
                                WALL_SEARCH_M, self.bodies)
        i = int(np.argmin(np.linalg.norm(self.xy - xy, axis=1)))
        return RoutePose(
            s_m=float(self.s[i]), lateral_offset_m=float(lateral_offset_m),
            region="vertex", xy_map=xy, yaw_map=float(yaw),
            route_xy_map=np.asarray(vertex.xy, float) + float(along_m) * t,
            route_tangent=t, route_normal=n, wall_dist_m=float(standoff_m),
            stand_dist_m=(float(ahead) if ahead is not None else float("inf")),
            wall_body=vertex.bodies[face])

    def as_dict(self) -> dict:
        return {"polyline_map": [[round(float(x), 4), round(float(y), 4)]
                                 for x, y in self.xy],
                "length_m": round(self.length_m, 4),
                "resample_m": round(self._step(), 4),
                "corner_centres_m": [round(c, 3) for c in self.corner_centres_m],
                "corner_spans_m": [[round(a, 3), round(b, 3)]
                                   for a, b in self.corner_spans_m],
                "region_spans_m": {k: [[round(a, 3), round(b, 3)]
                                       for a, b in v]
                                   for k, v in self.spans().items()},
                "region_length_m": {k: round(self.region_length_m(k), 3)
                                    for k in self.spans()},
                "corner_vertices": [v.as_dict() for v in
                                    self.corner_vertices()]}


# ── helpers ──────────────────────────────────────────────────────────────────


def _nearest_face(q, bodies, h: float = 1e-4):
    """Distance to the nearest body surface, its outward unit normal, and it.

    The normal is the gradient of the min-signed-distance field: exact for a
    face, and on the diagonal between two faces it points along the bisector
    -- which is why callers take the NEARER body's own normal rather than the
    gradient at a tie.  Ties are measure-zero on a sampled polyline."""
    def f(x):
        x = np.asarray(x, float).reshape(1, 2)
        return min(float(b.signed_distance(x)[0][0]) for b in bodies)

    q = np.asarray(q, float).reshape(2)
    ds = [float(b.signed_distance(q.reshape(1, 2))[0][0]) for b in bodies]
    if not ds:
        return float("nan"), np.array([1.0, 0.0]), None
    k = int(np.argmin(ds))
    g = np.array([(f(q + [h, 0.0]) - f(q - [h, 0.0])) / (2 * h),
                  (f(q + [0.0, h]) - f(q - [0.0, h])) / (2 * h)])
    n = float(np.linalg.norm(g))
    return ds[k], (g / n if n > 1e-9 else np.array([1.0, 0.0])), k


def _wrap(a: float) -> float:
    return float((float(a) + math.pi) % (2 * math.pi) - math.pi)


def _dedupe(p: np.ndarray, tol: float = 1e-6) -> np.ndarray:
    keep = [p[0]]
    for q in p[1:]:
        if float(np.linalg.norm(q - keep[-1])) > tol:
            keep.append(q)
    return np.asarray(keep, float)


def _resample(p: np.ndarray, step_m: float) -> np.ndarray:
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    L = float(s[-1])
    if L <= step_m:
        return p
    m = max(2, int(round(L / step_m)) + 1)
    ss = np.linspace(0.0, L, m)
    return np.column_stack([np.interp(ss, s, p[:, 0]),
                            np.interp(ss, s, p[:, 1])])


def _merge(runs: list, s_arr: np.ndarray, gap_m: float) -> list:
    """Fuse index runs whose arc-length gap is under `gap_m`."""
    out: list = []
    for lo, hi in runs:
        if out and float(s_arr[lo]) - float(s_arr[out[-1][1] - 1]) <= gap_m:
            out[-1] = (out[-1][0], hi)
        else:
            out.append((lo, hi))
    return out


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    out, start = [], None
    for i in range(len(mask) + 1):
        if i < len(mask) and mask[i]:
            start = i if start is None else start
        elif start is not None:
            out.append((start, i))
            start = None
    return out
