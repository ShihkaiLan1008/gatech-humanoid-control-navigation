"""Which room, and where its planned route runs.

One registry entry per scanned room, so `--room room2` today and
`--room demo_room` tomorrow are the same code path.  An entry knows three
things: how to build the `V3Scene` the demo already uses, how to build the
backend the demo already uses, and how to turn the planner's route into the
frozen centreline `route.RouteRef` consumes.

THE PLANNER IS CALLED HERE AND ONLY HERE.  `planned_centreline` runs A* once
per process, at start-up, to produce a polyline; every later decision reads
that polyline as geometry.  See `route.py`'s header.
"""

from __future__ import annotations

import numpy as np

from .action_space import corner_standoff_m  # re-exported for callers


def _astar_polyline(scene) -> tuple[np.ndarray, list]:
    """The planner's own route as `(points, edge_kinds_per_point)`."""
    from lane_graph.graph3 import Belief3, astar3, build_graph3

    g = build_graph3(scene)
    src = next(n.nid for n in g.nodes if n.kind == "start")
    dst = next(n.nid for n in g.nodes if n.kind == "goal")
    eids, _cost = astar3(g, src, dst,
                         Belief3(sigma=scene.prior_sigma_along_m, minor=0.0))
    if eids is None:
        raise RuntimeError(f"{scene.name}: the planner found no route")
    pts, kinds, nid = [], [], src
    for eid in eids:
        e = g.edges[eid]
        poly = (e.poly if e.poly is not None
                else np.array([g.nodes[e.a].xy, g.nodes[e.b].xy]))
        poly = np.asarray(poly, float)
        if e.b == nid:
            poly = poly[::-1]
        nid = e.b if e.a == nid else e.a
        pts.append(poly)
        kinds.append((e.kind, len(poly)))
    return np.vstack(pts), kinds


def _drop_leading_transit(pts: np.ndarray, kinds: list) -> np.ndarray:
    """Drop the walk-onto-the-lane stub the route always opens with.

    Every planned route starts at the start POSE and the first edges are the
    `FREE_TRANSIT` hops that carry the base onto a followable lane.  On room2
    that is 0.48 m of polyline whose heading points nowhere near a wall --
    (0,0) -> (0.01,-0.238) -> (-0.054,-0.004) -- and a base pose sampled on it
    faces open floor.  The trailing transit is NOT dropped: on room2 it runs
    (1.45, 3.19) -> (2.5, 3.2), straight along the north wall at the same
    0.56 m standoff as the lane before it, and it is a metre of perfectly
    good horizontal centreline.  Refused entirely if the trim would leave
    less than a metre of route."""
    n = 0
    for kind, count in kinds:
        if kind != "FREE_TRANSIT":
            break
        n += count
    if n == 0:
        return pts
    rest = pts[max(0, n - 1):]
    if len(rest) < 2:
        return pts
    L = float(np.sum(np.linalg.norm(np.diff(rest, axis=0), axis=1)))
    return rest if L >= 1.0 else pts


HUG_TOL_M = 0.08
HUG_FILLET_M = 0.12
HUG_STEP_M = 0.05


def _nearest_surface(q, bodies, h: float = 1e-4):
    """Distance to the nearest body surface and the outward unit normal there.

    Taken as the gradient of the min-signed-distance field, so it works for
    any convex body and needs no per-shape special case."""
    def f(x):
        x = np.asarray(x, float).reshape(1, 2)
        return min(float(b.signed_distance(x)[0][0]) for b in bodies)

    q = np.asarray(q, float).reshape(2)
    ds = [float(b.signed_distance(q.reshape(1, 2))[0][0]) for b in bodies]
    k = int(np.argmin(ds))
    d = ds[k]
    g = np.array([(f(q + [h, 0.0]) - f(q - [h, 0.0])) / (2 * h),
                  (f(q + [0.0, h]) - f(q - [0.0, h])) / (2 * h)])
    n = float(np.linalg.norm(g))
    return d, (g / n if n > 1e-9 else np.array([1.0, 0.0])), k


def hug_walls(poly, bodies, *, tol_m: float = HUG_TOL_M,
              fillet_m: float = HUG_FILLET_M,
              step_m: float = HUG_STEP_M) -> np.ndarray:
    """Replace the route's corner shortcut with the wall-following elbow.

    WHY.  The planner rounds Room 2's corner on a diagonal cut: it leaves the
    west lane at y = 2.4, crosses open floor, and rejoins the north lane at
    x = 0.5.  Measured, that cut's closest approach to the corner VERTEX is
    1.13 m and to any FACE is 0.79 m -- outside the arm's 0.48 m reach.  Base
    poses are drawn along this centreline +-0.20 m, so the corner itself --
    the one landmark in the room, and the only place a touch can feel two
    faces -- got no contact data at all.  It shows as an empty quadrant in
    `topdown.png`.

    WHAT.  The nodes that are genuinely following a wall are kept exactly
    where the planner put them, so the straight legs keep their measured
    0.496 m and 0.560 m standoffs.  Everything between two legs that follow
    DIFFERENT bodies is replaced by the elbow those two legs' lines meet at --
    the point that is each leg's own standoff from each of the two faces --
    with a short fillet so the heading, and therefore the yaw a pose is given,
    sweeps through the diagonal instead of jumping.  A concave corner's
    constant-standoff locus IS a sharp elbow, so the fillet is kept small:
    every millimetre of rounding moves the path away from both faces.

    This changes the REFERENCE, not the planner.  A* is still what produced
    the route; nothing here consults it.
    """
    p = _resample(_dedupe(np.asarray(poly, float)), step_m)
    if len(p) < 5:
        return p
    d = np.zeros(len(p))
    nrm = np.zeros((len(p), 2))
    who = np.full(len(p), -1)
    for i, q in enumerate(p):
        d[i], nrm[i], who[i] = _nearest_surface(q, bodies)
    t = np.gradient(p, axis=0)
    t = t / np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-12)
    par = np.abs(np.einsum("ij,ij->i", t, nrm)) < 0.25
    target = float(np.median(d[par])) if par.any() else float(np.median(d))
    follow = par & (d <= target + tol_m)
    legs = _runs(follow, who, min_len=max(4, int(0.25 / step_m)))
    if len(legs) < 2:
        return p

    out = [p[:legs[0][1]]]
    for (a0, a1, _), (b0, b1, _) in zip(legs, legs[1:]):
        pa, ta = p[a1 - 1], t[a1 - 1]
        pb, tb = p[b0], t[b0]
        e = _intersect(pa, ta, pb, tb)
        if e is None:
            out.append(p[a1:b1])                # parallel: leave it alone
            continue
        c = _corner_point(pa, nrm[a1 - 1], d[a1 - 1],
                          pb, nrm[b0], d[b0])
        if c is not None:
            e = c
        out.append(_elbow(pa, e, pb, fillet_m, step_m))
        out.append(p[b0:b1])
    out.append(p[legs[-1][1]:])
    return _resample(_dedupe(np.vstack([q for q in out if len(q)])), step_m)


def _runs(mask, who, *, min_len: int):
    """Maximal runs of `mask` that stay on one body, as `(lo, hi, body)`."""
    out, lo = [], None
    for i in range(len(mask) + 1):
        end = (i == len(mask) or not mask[i]
               or (lo is not None and who[i] != who[lo]))
        if lo is not None and end:
            if i - lo >= min_len:
                out.append((lo, i, int(who[lo])))
            lo = None
        if i < len(mask) and mask[i] and lo is None:
            lo = i
    return out


def _corner_point(pa, na, da, pb, nb, db):
    """The point `corner_standoff_m()` from BOTH faces the two legs follow.

    The faces are recovered from each leg: it sits `d` along the outward
    normal `n` from its own face, so the face line is
    `{x : (x - p)·n = -d}`.  Their intersection is the vertex, and the point
    at `s` from each is `vertex + s·(na + nb)` for perpendicular faces --
    which is what a room corner is.  None when the two faces are parallel."""
    s = corner_standoff_m()
    na = np.asarray(na, float)
    nb = np.asarray(nb, float)
    if abs(float(np.cross(na, nb))) < 0.3:      # not a corner, a kink
        return None
    a = np.stack([na, nb])
    rhs = np.array([float(na @ np.asarray(pa, float)) - float(da),
                    float(nb @ np.asarray(pb, float)) - float(db)])
    try:
        vertex = np.linalg.solve(a, rhs)
    except np.linalg.LinAlgError:
        return None
    return vertex + s * (na + nb)


def _intersect(p0, u0, p1, u1):
    """Where two lines meet, or None when they are parallel."""
    a = np.array([[u0[0], -u1[0]], [u0[1], -u1[1]]], float)
    if abs(float(np.linalg.det(a))) < 1e-9:
        return None
    st = np.linalg.solve(a, np.asarray(p1, float) - np.asarray(p0, float))
    return np.asarray(p0, float) + st[0] * np.asarray(u0, float)


def _elbow(pa, e, pb, fillet_m: float, step_m: float) -> np.ndarray:
    """`pa -> e -> pb` as two straight runs with a quadratic fillet at `e`.

    Straight to and from the corner point, and back onto the second leg's own
    line at `pb`.  An earlier version ran OUT of the corner along the second
    leg's direction instead of toward its start, which left the whole
    remaining leg on a line 0.13 m inside the one the planner walked -- the
    profile ran 0.40 m from the north face where the leg holds 0.56 m."""
    pa = np.asarray(pa, float)
    e = np.asarray(e, float)
    pb = np.asarray(pb, float)
    ua, la = _unit(e - pa)
    ub, lb = _unit(pb - e)
    f = float(min(fillet_m, 0.45 * la, 0.45 * lb))
    c0, c2 = e - f * ua, e + f * ub
    n_in = max(2, int((la - f) / step_m) + 1)
    n_out = max(2, int((lb - f) / step_m) + 1)
    seg_in = pa + np.linspace(0.0, la - f, n_in)[:, None] * ua
    u = np.linspace(0.0, 1.0, max(3, int(2 * f / step_m) + 2))[:, None]
    arc = (1 - u) ** 2 * c0 + 2 * (1 - u) * u * e + u ** 2 * c2
    seg_out = c2 + np.linspace(0.0, lb - f, n_out)[:, None] * ub
    return np.vstack([seg_in, arc, seg_out])


def _unit(v):
    v = np.asarray(v, float)
    n = float(np.linalg.norm(v))
    return (v / n, n) if n > 1e-9 else (np.array([1.0, 0.0]), 0.0)


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


def _room2(reverse: bool = False):
    from demo_map.scene_room2 import build_room2_scene
    return build_room2_scene(reverse)


def _demo_room(reverse: bool = False):
    from demo_map.scene_demo import build_demo_scene
    if reverse:
        raise ValueError("demo_room has no reverse scene")
    return build_demo_scene()


ROOMS = {
    "room2": {
        "map_id": "room2_8_11_scan",
        "build_scene": _room2,
        "scan_offset": lambda reverse: __import__(
            "demo_map.scene_room2", fromlist=["room2_scan_offset"]
        ).room2_scan_offset(reverse),
        "walls_json": "docs/demo_maps/room2_scan_walls.json",
    },
    "demo_room": {
        "map_id": "demo_room_scan",
        "build_scene": _demo_room,
        "scan_offset": lambda reverse: __import__(
            "demo_map.scene_demo", fromlist=["scan_start_offset"]
        ).scan_start_offset(),
        "walls_json": "docs/demo_maps/demo_room_scan_walls.json",
    },
}


def build_scene(room: str, reverse: bool = False):
    if room not in ROOMS:
        raise ValueError(f"unknown room {room!r}; have {sorted(ROOMS)}")
    return ROOMS[room]["build_scene"](reverse)


def planned_centreline(scene, *, hug_corner: bool = True) -> np.ndarray:
    """The frozen reference polyline, in the scene (map) frame.

    `hug_corner` pulls the planner's corner shortcut back onto the wall-
    following standoff -- see `hug_walls` for what that buys and why the
    planner's own route does not."""
    pts, kinds = _astar_polyline(scene)
    poly = _drop_leading_transit(pts, kinds)
    return hug_walls(poly, tuple(scene.plan.bodies)) if hug_corner else poly


def map_identity(room: str, scene, reverse: bool = False) -> dict:
    """Enough to rebuild this room's exact geometry without duplicating it.

    Room 2 is fixed, so a transition stores this identifier and not the map:
    `build_scene(room, reverse)` plus the recorded body table reproduces the
    same `V3Scene` the collection ran against."""
    e = ROOMS[room]
    bodies = [{"name": b.name,
               "verts": [[round(float(x), 5), round(float(y), 5)]
                         for x, y in np.asarray(b.verts, float)]}
              for b in scene.plan.bodies]
    return {
        "room": room, "map_id": e["map_id"], "reverse": bool(reverse),
        "scene_name": scene.name,
        "walls_json": e["walls_json"],
        "scan_offset_map_xy": [float(v) for v in e["scan_offset"](reverse)],
        "start_map_xy": [float(v) for v in scene.start_map_xy],
        "goal_map_xy": [float(v) for v in scene.goal_map_xy],
        "goal_radius_m": float(scene.goal_radius_m),
        "true_delta_m": float(getattr(scene, "true_delta_m", 0.0) or 0.0),
        "bodies": bodies,
    }
