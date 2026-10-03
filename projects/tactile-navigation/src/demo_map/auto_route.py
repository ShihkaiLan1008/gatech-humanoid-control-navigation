"""Map in, route out: the reference route, derived rather than drawn.

`route.build_reference_route` and `scan_map.build_scan_route` are the same ten
segments written by hand -- which wall is followed, on which hand, in what
order.  Everything downstream is already automatic: `landmarks.harvest` finds
the visitable bodies from a polyline and the map, `_rhythm` places the cadence,
the leg arbiter chooses the touches.  This module closes the one remaining
hand-authored link, so that a NEW room needs a map and nothing else.

HOW.  The geometric planner answers where the base may go; it says nothing
about what is followed.  So the polyline is classified against the map:

  1. plan it under the TOUCH BAND, so the route already runs where a surface
     is in reach (`RouteConfig.touch_reach_m`);
  2. at every sample, cast left and right for the nearest face inside the
     lane window -- that pair (body, side) is what "following" means;
  3. group maximal runs of one (body, side).  A run shorter than the acquire
     gate's own run-in floor is not a followable wall and stays transit;
  4. re-project each run onto the line exactly `lane_off` from ITS face.
     This step is not cosmetic: `executor._wall_segment_for` recovers the
     surface by offsetting the segment back inward, so a segment merely NEAR
     the lane puts the virtual face -- and every press -- off by the error;
  5. where two runs meet on near-perpendicular faces whose surface lines
     intersect within reach, emit the `corner_turn` the compiler needs to
     schedule the rank-2 bracketing pair, instead of two walls and a gap.

WHAT IS DELIBERATELY NOT DERIVED.  Curved features get no `curve_follow`
segment.  The compiler visits a bulge because `harvest` found it beside the
route, not because the route bowed around it, and a drawn bow would only
re-introduce an authored arc the planner already routes past.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from execution.wall_lane import raycast
from planning.geometric import (OccupancyGrid, RouteConfig, plan_route_on,
                                plan_route_via_features)

from .route import Route, RouteSeg

REACH_M = 0.45
"""Arm reach; the touch band the route is planned inside."""

MIN_RUN_M = 0.95
"""Shortest wall run worth following: the acquire gate's run-in floor
(`scene_demo.emit_follow` drops anything under it, so emitting one here would
only produce a segment the compiler discards)."""

LANE_WINDOW_M = 0.30
"""How far either side of `lane_off` a face still counts as the one being
followed.  Wide enough for the planner's own band (it runs 0.34-0.45 where the
lane is 0.40), tight enough that the far wall of a corridor is not claimed."""

CORNER_MAX_GAP_M = 2.0
"""Two runs further apart than this are two walls, not a corner.  Two lanes
meeting at a right angle are already `2 * lane_off` apart where their faces
touch, so the bound has to clear that plus the run trimming."""

CORNER_PERP = 0.35
"""|n_a . n_b| below this counts as perpendicular faces."""


@dataclass
class _Run:
    body: int
    hand: str                 # which side the wall is on: left | right
    i0: int
    i1: int
    normal: np.ndarray        # the face's outward normal (toward the lane)
    surf_p: np.ndarray        # a point ON the face



def _cross2(u, v) -> float:
    """2-D cross product.  numpy 2 removed the 2-vector form of `np.cross`."""
    return float(u[0] * v[1] - u[1] * v[0])


def clearance_fn(scene):
    bodies = tuple(scene.plan.bodies)

    def clearance(pts):
        pts = np.asarray(pts, float).reshape(-1, 2)
        out = np.full(len(pts), np.inf)
        for b in bodies:
            sd, _n = b.signed_distance(pts, None)
            out = np.minimum(out, np.asarray(sd, float))
        return out
    return clearance


def _bounds(scene, pad: float = 1.2):
    v = np.vstack([np.asarray(b.verts, float) for b in scene.plan.bodies])
    lo, hi = v.min(0) - pad, v.max(0) + pad
    return (lo[0], lo[1], hi[0], hi[1])


def _resample(pts: np.ndarray, ds: float) -> np.ndarray:
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.r_[0.0, np.cumsum(seg)]
    if s[-1] < 1e-9:
        return pts
    t = np.linspace(0.0, s[-1], max(2, int(s[-1] / ds) + 1))
    return np.stack([np.interp(t, s, pts[:, 0]),
                     np.interp(t, s, pts[:, 1])], axis=1)


def _classify(poly: np.ndarray, bodies, lane_off: float) -> list:
    """(body, hand, face normal, face point) at every sample, or None."""
    lo, hi = lane_off - LANE_WINDOW_M, lane_off + LANE_WINDOW_M
    # only WALLS are followed.  A bulge or a post is a landmark: `harvest`
    # schedules a visit to it, and following one would compile a rhythm
    # along a 0.3 m body that the acquire gate cannot run in to anyway.
    walls = {i for i, b in enumerate(bodies) if b.name.startswith("wall_")}
    out = []
    for i, p in enumerate(poly):
        j = min(i + 1, len(poly) - 1)
        k = max(i - 1, 0)
        u = poly[j] - poly[k]
        n = float(np.linalg.norm(u))
        if n < 1e-9:
            out.append(None)
            continue
        u = u / n
        best = None
        for hand, s in (("left", np.array([-u[1], u[0]])),
                        ("right", np.array([u[1], -u[0]]))):
            d, n_surf, bi = raycast(p, s, hi, tuple(bodies))
            if d is None or not (lo <= d <= hi) or bi is None:
                continue
            if bi not in walls:
                continue
            if best is None or d < best[0]:
                best = (d, hand, bi, np.asarray(n_surf, float), p + d * s)
        out.append(None if best is None
                   else (best[2], best[1], best[3], best[4]))
    return out


FACE_DOT = 0.999
"""How parallel two hits must be to be the SAME face -- `face_present`'s own
tolerance (2.6 deg), reused rather than restated.

A run must be one face, and the downstream test is the definition of one.
Averaging the hit normals across a run instead produced a segment tilted 7.5
deg off the wall it named; every station on it then failed `face_present`, the
corner's whole exit face came back with no candidates, and the gate failed
with a perfectly good wall in front of the robot."""


def _runs(tags: list, poly: np.ndarray) -> list:
    """Maximal runs of one (body, hand, FACE), long enough to be followed."""
    runs, i = [], 0
    while i < len(tags):
        if tags[i] is None:
            i += 1
            continue
        bi, hand, n0 = tags[i][0], tags[i][1], tags[i][2]
        j = i
        while j + 1 < len(tags) and tags[j + 1] is not None \
                and tags[j + 1][0] == bi and tags[j + 1][1] == hand \
                and float(tags[j + 1][2] @ n0) >= FACE_DOT:
            j += 1
        if float(np.linalg.norm(poly[j] - poly[i])) >= MIN_RUN_M:
            # the FACET's own normal, not an average over the run: the mean of
            # two facets is parallel to neither
            ns = n0 / max(float(np.linalg.norm(n0)), 1e-9)
            sp = np.mean([tags[k][3] for k in range(i, j + 1)], axis=0)
            runs.append(_Run(bi, hand, i, j, ns, sp))
        i = j + 1
    return runs


def _lane_segment(run: _Run, poly: np.ndarray, lane_off: float):
    """The run, re-projected onto the line exactly `lane_off` from its face."""
    n = run.normal
    # the face tangent, ORIENTED BY TRAVEL.  Taking it as perp-to-normal and
    # swapping afterwards reverses runs whose face normal happens to point
    # the other way, and a reversed segment reads as a transit the length of
    # the wall -- demo_room's route came out 44.7 m against a planned 25.5.
    t = np.array([-n[1], n[0]])
    if float((poly[run.i1] - poly[run.i0]) @ t) < 0.0:
        t = -t
    origin = run.surf_p + lane_off * n
    p0 = origin + float((poly[run.i0] - origin) @ t) * t
    p1 = origin + float((poly[run.i1] - origin) @ t) * t
    return p0, p1, n, t


def _corner(a: _Run, b: _Run, pa1, pb0, lane_off: float):
    """The vertex two runs turn about, or None if they are just two walls."""
    if a.body == b.body or abs(float(a.normal @ b.normal)) > CORNER_PERP:
        return None
    if float(np.linalg.norm(np.asarray(pb0) - np.asarray(pa1))) \
            > CORNER_MAX_GAP_M:
        return None
    # intersect the two FACE lines: n . x = n . surf_p
    A = np.stack([a.normal, b.normal])
    rhs = np.array([a.normal @ a.surf_p, b.normal @ b.surf_p])
    if abs(float(np.linalg.det(A))) < 1e-6:
        return None
    return np.linalg.solve(A, rhs)


def _rdp(pts: np.ndarray, tol: float) -> np.ndarray:
    """Douglas-Peucker: the polyline's shape in as few vertices as carry it."""
    if len(pts) < 3:
        return pts
    a, b = pts[0], pts[-1]
    ab = b - a
    L = float(np.linalg.norm(ab))
    if L < 1e-9:
        d = np.linalg.norm(pts - a, axis=1)
    else:
        u, rel = ab / L, pts - a
        d = np.abs(u[0] * rel[:, 1] - u[1] * rel[:, 0])   # 2-D cross
    i = int(np.argmax(d))
    if d[i] <= tol:
        return np.stack([a, b])
    left = _rdp(pts[:i + 1], tol)
    return np.vstack([left[:-1], _rdp(pts[i:], tol)])


def _connector(poly: np.ndarray, a, b, tol: float = 0.03) -> list:
    """Waypoints the PLANNER'S OWN PATH takes from a to b.

    A free-space connector used to be emitted as the straight chord between
    its endpoints, which throws away exactly what the planner was run for:
    the chord from one wall run to the next cuts the arc A* drew around a
    bulge (measured: the auto route's two bulge transits ran a median 0.11
    and 0.18 m from the material -- through it, against a 0.34 m base
    clearance -- degrading odometry and contact rate until the corner gate
    at the end of the route failed), and on an open floor a 6.2 m chord
    walked into walls the planned polyline had gone around (room_8_30: two
    force-aborts, zero touches).  The polyline between the endpoints IS the
    collision-free answer; this recovers it and hands back its corners.

    ``tol`` is 0.03, not the 0.10 a plot would get away with: the bow around
    a small bulge deviates less than 0.10 m from its own chord, so a coarser
    simplification flattens exactly the detour that mattered (measured on
    bulge_0: the slice's own min clearance is 0.36 m, the simplified chord's
    0.13)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    i = int(np.argmin(np.linalg.norm(poly - a, axis=1)))
    j = int(np.argmin(np.linalg.norm(poly - b, axis=1)))
    if j <= i + 1:
        return []
    keep = _rdp(poly[i:j + 1], tol)
    return [np.asarray(q, float) for q in keep[1:-1]]


def plan_auto_route(scene, *, lane_off: float, cfg: RouteConfig | None = None,
                    via_features: bool = False,
                    detour_budget: float = 1.6) -> Route:
    """A `Route` for this scene, from the map alone."""
    cfg = cfg or RouteConfig(touch_reach_m=REACH_M)
    bodies = tuple(scene.plan.bodies)
    start = np.asarray(getattr(scene, "start_map_xy", (0.0, 0.0)), float)
    goal = np.asarray(scene.goal_map_xy, float)
    grid = OccupancyGrid(clearance_fn(scene), _bounds(scene), cfg)
    fx = np.array([f.xy for f in scene.plan.features], float)
    r = (plan_route_via_features(grid, start, goal, fx, cfg, detour_budget)
         if via_features and len(fx) else plan_route_on(grid, start, goal, cfg))
    if not r.found:
        raise RuntimeError(f"no route on this map: {r.reason}")

    poly = _resample(np.asarray(r.points, float), 0.05)
    runs = _runs(_classify(poly, bodies, lane_off), poly)

    segs, k = [], 0
    cf = clearance_fn(scene)

    def lift(q, keep=False):
        """Push a waypoint out to base clearance along the clearance gradient.

        The connector endpoints are LANE-LINE projections of polyline points,
        and beside a bulge the projection pulls the path inward off the bow
        A* drew -- measured 0.108 m from the material against a 0.34 m base.
        ``keep`` pins the mission goal, which stands ON its marker by design."""
        q = np.asarray(q, float)
        if keep:
            return q
        for _ in range(8):
            c = float(cf(q[None, :])[0])
            if c >= 0.36:
                return q
            e = 0.02
            g = np.array([float(cf((q + [e, 0])[None, :])[0]) - c,
                          float(cf((q + [0, e])[None, :])[0]) - c]) / e
            n = float(np.linalg.norm(g))
            if n < 1e-6:
                return q
            q = q + g / n * min(0.36 - c + 0.02, 0.12)
        return q

    def chain(kind, a, b, follows="", hand="", note="", pin_end=False):
        """One connector, threaded through the planner's polyline."""
        nonlocal k
        pts = [lift(a)] + [lift(q) for q in _connector(poly, a, b)] \
            + [lift(b, keep=pin_end)]
        for q0, q1 in zip(pts[:-1], pts[1:]):
            segs.append(RouteSeg(f"A{k}", kind, tuple(q0), tuple(q1),
                                 follows, hand, note=note))
            k += 1

    prev_run = prev_p1 = None
    for run in runs:
        p0, p1, n, _t = _lane_segment(run, poly, lane_off)
        if prev_p1 is None:
            if float(np.linalg.norm(p0 - start)) > 0.05:
                chain("approach", start, p0, bodies[run.body].name, run.hand,
                      note="spawn and acquire, along the planned path")
        else:
            vtx = _corner(prev_run, run, prev_p1, p0, lane_off)
            if vtx is not None:
                # THE LANE RUNS TO THE CORNER, HOWEVER FAR A* WALKED.  The
                # planner's path cuts corners -- inside the touch band the
                # diagonal is simply shorter -- so the runs it induces end
                # early (room2's west-wall run stopped 1.65 m before its own
                # corner) and the "corner_turn" bridging them was an arc
                # whose endpoints sat 1.6 m from its radius-0.40 circle: a
                # drawn diagonal wearing an arc's name.  But where the lane
                # goes at a corner is not the walk's to decide: the two
                # faces meet at the vertex, so the two LANE lines meet too,
                # and that meeting is derived from the map alone.
                #
                #   concave: the lane lines intersect at
                #            vertex + lane_off*(n_a + n_b); both follows
                #            extend to that point and the turn happens there.
                #   convex:  the lane rounds the vertex on a true fillet of
                #            radius lane_off; the tangent points
                #            vertex + lane_off*n are ON the arc's circle.
                n_a = np.asarray(prev_run.normal, float)
                n_b = np.asarray(run.normal, float)
                # convex or concave, read off the TURN -- the same rule the
                # compiler derives it from.  Rounding a corner whose material
                # is inside the L turns the robot toward its followed wall; an
                # inside corner turns it away.  (Testing which quadrant run
                # B's lane start sits in fails on BOTH kinds: a convex face's
                # lane extends past the vertex on the outside and lands in
                # the concave quadrant too.)
                t_a = np.array([-n_a[1], n_a[0]])
                if float((np.asarray(vtx) - np.asarray(prev_p1)) @ t_a) < 0:
                    t_a = -t_a                      # oriented by travel
                t_b = np.array([-n_b[1], n_b[0]])
                if float((p1 - p0) @ t_b) < 0:
                    t_b = -t_b
                turn = float(t_a[0] * t_b[1] - t_a[1] * t_b[0])   # 2-D cross
                convex = turn > 0.0 if run.hand == "left" else turn < 0.0
                if convex:
                    qa = np.asarray(vtx) + lane_off * n_a
                    qb = np.asarray(vtx) + lane_off * n_b
                else:
                    qa = np.asarray(vtx) + lane_off * (n_a + n_b)
                    # a point turn; nudged along the exit lane so the arc's
                    # sweep is a sliver, not the full circle p0 == p1 draws
                    qb = qa + 0.05 * t_b
                segs[-1] = RouteSeg(segs[-1].sid, "wall_follow",
                                    segs[-1].p0, tuple(qa),
                                    segs[-1].follows, segs[-1].hand,
                                    note=segs[-1].note + "; run to the corner")
                segs.append(RouteSeg(
                    f"A{k}", "corner_turn", tuple(qa), tuple(qb),
                    f"{bodies[prev_run.body].name}->{bodies[run.body].name}",
                    run.hand, center=tuple(vtx), radius=lane_off,
                    # the short way round, read off the endpoints themselves:
                    # RouteSeg sweeps ccw-or-cw the whole way, and a sliver
                    # turn with the wrong sign is drawn as the other 356 deg
                    ccw=bool(_cross2(qa - np.asarray(vtx), qb - np.asarray(vtx)) > 0),
                    note="faces meet here"))
                k += 1
                p0 = qb
            elif float(np.linalg.norm(p0 - prev_p1)) > 0.05:
                chain("transit", prev_p1, p0,
                      note="nothing in reach; shape kept from the planner")
        segs.append(RouteSeg(f"A{k}", "wall_follow", tuple(p0), tuple(p1),
                             bodies[run.body].name, run.hand,
                             note=f"followed on the {run.hand}"))
        k += 1
        prev_run, prev_p1 = run, p1

    if prev_p1 is None:
        chain("transit", start, goal, pin_end=True,
              note="no followable wall; the planned path verbatim")
        if segs:
            segs[-1] = RouteSeg(segs[-1].sid, "goal_approach", segs[-1].p0,
                                segs[-1].p1, "", "", note=segs[-1].note)
    elif float(np.linalg.norm(np.asarray(goal) - prev_p1)) > 0.05:
        chain("goal_approach", prev_p1, goal, note="step to the goal",
              pin_end=True)

    segs = [s for s in segs
            if s.length > 0.05 or s.kind == "corner_turn"]
    return Route(segs, tuple(start), tuple(goal))


def route_summary(route: Route) -> str:
    parts = []
    for s in route.segs:
        parts.append(f"{s.sid}:{s.kind}"
                     + (f"[{s.follows}/{s.hand}]" if s.follows else "")
                     + f" {s.length:.2f}m")
    return f"{len(route.segs)} segs, {route.length:.1f} m | " + " ".join(parts)
