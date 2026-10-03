"""Every non-wall rigid body the route can touch, harvested from the map.

WHY THIS EXISTS.  room2 has exactly two landmarks -- one corner and one
owner-placed goal marker -- so its compiler could name them by hand.  The
scanned demo room has 36 non-wall bodies (3 curved bulges, 33 box
protrusions), and hand-picking from that list would be a judgement the map
should be making.  This module states the rule instead:

    A LANDMARK IS ANY RIGID BODY THAT IS NOT A WALL, and it is IN SCOPE when
    a stance exists from which the arm can press one of its faces.

WHY NOT WALLS.  A flat face pins one axis -- its own normal -- and nothing
along itself.  That is the rank-1 result this whole demo exists to show, and
it is why the flat runs keep the cheap rule-based "feeling forward" cadence.
A body with EDGES is different: a press that lands and a sweep that misses
0.15 m further on bracket the body's along-face coordinate, so two touches on
one face of a box pin BOTH axes.  A cylinder does it in one touch, because
its normal turns with position.  Those are the bodies worth spending ATE's
information budget on.

WHAT "REACHABLE" MEANS, MEASURED.  The stance is the lane point one press
standoff out along the face's outward normal.  It is feasible when a ray cast
back from it along that normal hits THIS body at that standoff (which rejects
faces that are occluded, buried in a wall, or facing away), when the base fits
there, and when it lies within a bounded detour of the drawn route.  Every
body that fails is returned with the reason -- a landmark dropped silently
reads as a landmark that was never there.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from execution.wall_lane import raycast

BASE_CLEAR_M = 0.30
"""Free radius a stance needs around the base, excluding the pressed face."""

MAX_DETOUR_M = 0.90
"""How far off the drawn route a landmark's stance may sit.

Not a taste judgement: the demo room's corridor is 1.68 m wide between the
long wall (y = 0.671) and its opposite face (y = -1.011), so a stance on the
FAR side sits ~0.65 m off the lane.  0.90 admits that sidestep and excludes
anything that would be a second route rather than a detour.  Every landmark
beyond it is reported as `too_far`, with its distance, so the cap is visible
rather than silent."""


@dataclass(frozen=True)
class LandmarkSite:
    """One pressable landmark face, in the scene frame."""

    body: str
    kind: str                      # "cylinder" | "box"
    centre: tuple
    n_out: tuple                   # the pressed face's outward normal
    face_xy: tuple                 # the point on that face the stance aims at
    stance_xy: tuple
    s_route: float                 # arc position along the drawn route
    detour_m: float
    lane_m: float = 0.0            # clear press-lane through the stance
    reach_m: float = 0.0           # centre -> pressed face, along ``n_out``
    radius: float = 0.0            # cylinders
    half_xy: tuple = (0.0, 0.0)    # boxes, in the body's own axes
    yaw: float = 0.0               # boxes


def _is_cylinder(verts: np.ndarray) -> bool:
    """The scene builds bulges as 20-gons and protrusions as quads."""
    return len(verts) >= 12


def _box_axes(verts: np.ndarray):
    """(centre, half_xy, yaw) of a quad, from its own two edge directions."""
    c = verts.mean(axis=0)
    e0 = verts[1] - verts[0]
    u = e0 / max(float(np.linalg.norm(e0)), 1e-12)
    v = np.array([-u[1], u[0]])
    hu = float(np.max(np.abs((verts - c) @ u)))
    hv = float(np.max(np.abs((verts - c) @ v)))
    if hv > hu:                      # keep +x' the long axis, as a box spec does
        u, v = v, -u
        hu, hv = hv, hu
    return c, (hu, hv), float(np.arctan2(u[1], u[0]))


def _face_normals(verts: np.ndarray, centre: np.ndarray):
    """Candidate press directions offered by the body's own shape.

    A quad offers its four edge normals.  A polygonal cylinder offers a
    continuum, so it is sampled on a 24-way fan -- finer than the hand aims
    anyway."""
    if _is_cylinder(verts):
        for a in np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False):
            yield np.array([np.cos(a), np.sin(a)])
        return
    n_v = len(verts)
    for i in range(n_v):
        e = verts[(i + 1) % n_v] - verts[i]
        L = float(np.linalg.norm(e))
        if L < 1e-9:
            continue
        n = np.array([e[1], -e[0]]) / L
        mid = 0.5 * (verts[i] + verts[(i + 1) % n_v])
        yield n if float(n @ (mid - centre)) >= 0.0 else -n


def _centre_line_face(bodies, bi: int, c: np.ndarray, r_max: float,
                      n: np.ndarray):
    """Where the press ray through the body CENTRE leaves the body.

    Not the supporting plane's touch point (the body's extreme vertex along
    ``n``), which is where the first version aimed: on a body whose scan
    reconstruction is a degree or two off axis those are different points, and
    the difference is exactly the press-depth error that made the stance's own
    verification ray miss.  Measured on prot_27, a 0.30x0.16 box lying 24 deg
    off the long wall: extreme vertex 0.134 m out, centre-line exit 0.088 m --
    a 47 mm disagreement against a 30 mm face tolerance, so the body was
    rejected as unreachable while sitting 0.33 m off the route.

    The ray starts just clear of the body's own bounding circle, NOT from far
    away: most of these bodies are lumps embedded in a wall, and a ray thrown
    in from across the room hits the wall first and reports the lump occluded
    -- which deleted both curved features and every wall protrusion.  Starting
    at the bounding circle keeps the test local, so the only thing that can
    occlude a face is something genuinely standing in front of it.

    Returns None when the ray reaches some other body first."""
    start = c + (float(r_max) + 0.02) * n
    d, _nn, hit = raycast(start, -n, float(r_max) + 0.04, bodies)
    if d is None or hit != bi:
        return None
    return start - float(d) * n


def _route_projection(poly: np.ndarray, ds: np.ndarray, p: np.ndarray):
    """(arc length of the nearest route sample, distance to it)."""
    k = int(np.argmin(np.linalg.norm(poly - p, axis=1)))
    return float(ds[k]), float(np.linalg.norm(poly[k] - p))


def _clearance(bodies, pts: np.ndarray, skip: int) -> np.ndarray:
    """Distance from each point to the nearest body surface (skipping one).

    Surface distance, not nearest-vertex distance: a 6 m wall's nearest vertex
    can be metres away while its face is 0.2 m from the base."""
    d = np.full(len(pts), 9.9)
    for i, b in enumerate(bodies):
        if i == skip:
            continue
        d = np.minimum(d, np.asarray(b.signed_distance(pts)[0], float))
    return d


def _lane_run_m(bodies, skip: int, face_xy, n, standoff_m: float,
                run_in: float, run_out: float, step: float = 0.12) -> float:
    """Longest clear stretch of the apex-tangent lane THROUGH the stance.

    `_ate_landmark` does not press from wherever it happens to stand: it
    builds a segment tangent to the pressed face, reacquires the lane ~0.75 m
    before the station and walks in.  So a face is only usable if that lane
    exists.  Contiguous-through-the-stance rather than clear-everywhere,
    because a lane blocked on ONE side is still a lane -- the run-in simply
    comes from the other end.  Measured: the all-or-nothing version rejected
    every protrusion in the end room, including ones with 1.4 m of open floor
    in front of them, because a wall clipped the far tail of the sample."""
    t = np.array([-n[1], n[0]])
    lane0 = np.asarray(face_xy, float) + float(standoff_m) * np.asarray(n, float)
    a = np.arange(-run_in, run_out + 1e-9, step)
    i0 = int(np.argmin(np.abs(a)))
    ok = _clearance(bodies, lane0 + a[:, None] * t, skip) >= BASE_CLEAR_M
    if not ok[i0]:
        return 0.0
    lo = i0
    while lo > 0 and ok[lo - 1]:
        lo -= 1
    hi = i0
    while hi < len(ok) - 1 and ok[hi + 1]:
        hi += 1
    return float(a[hi] - a[lo])


def _nearby_wall_normals(bodies, verts: np.ndarray, attach_m: float):
    """Outward normals of every wall face close enough to have grown this body.

    A protrusion is a lump ON a wall, and the robot presses it from the wall's
    own lane -- travelling parallel to the wall with the hand out sideways.
    Its reconstructed front facet is a degree or two off the wall (scan
    noise), and building the press lane from THAT tilt swings the lane into
    the wall over the metre of run-in the station needs.

    All four faces of a near wall, not the nearest one: a lump is usually
    EMBEDDED in its wall, so "nearest face" is a coin flip between the face it
    protrudes through and the one behind it -- and behind it the stance is
    inside the wall.  Offering both and letting the feasibility test choose is
    both simpler and right.

    Nearness is measured from the body's SURFACE, not its centre.  The scan
    carves the wall away where a big feature grows out of it, so C2 -- a
    0.38 m bulge sitting in a 0.76 m gap in the long wall -- has no wall
    within 0.40 m of its centre at all, and the centre-based test reported it
    free-standing and pressed its flank."""
    out = []
    for b in bodies:
        if not b.name.startswith("wall_"):
            continue
        if float(np.min(b.signed_distance(verts)[0])) <= attach_m:
            out.extend(np.asarray(n, float) for n in b.normals)
    return out


def harvest(scene, route_poly, *, standoff_m: float,
            max_detour_m: float = MAX_DETOUR_M,
            face_tol_m: float = 0.03, attach_m: float = 0.40,
            min_lane_m: float = 0.90,
            run_in_m: float = 1.13, run_out_m: float = 0.9) -> tuple[list, list]:
    """Landmark sites in route order, plus everything rejected and why."""
    bodies = tuple(scene.plan.bodies)
    poly = np.asarray(route_poly, float)
    ds = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(poly, axis=0),
                                                         axis=1))])
    sites, skipped = [], []
    for bi, b in enumerate(bodies):
        if b.name.startswith("wall_"):
            continue
        V = np.asarray(b.verts, float)
        c = V.mean(axis=0)
        cyl = _is_cylinder(V)
        # The wall this body sits on, first: pressing a lump the way its wall
        # is pressed is both what the machinery can do and what the demo is
        # about.  The body's own facets are the fallback for free-standing
        # bodies (the branch-mouth post) and for lumps whose wall face is
        # occluded.
        cands = [(0, n) for n in _nearby_wall_normals(bodies, V, attach_m)]
        cands += [(1, n) for n in _face_normals(V, c)]

        best, why = None, "no_exposed_face"
        r_max = float(np.max(np.linalg.norm(V - c, axis=1)))
        for rank, n in cands:
            q = _centre_line_face(bodies, bi, c, r_max, n)
            if q is None:
                continue
            stance = q + float(standoff_m) * n
            d, nn, hit = raycast(stance, -n, float(standoff_m) + 0.06, bodies)
            if d is None or hit != bi or abs(float(d) - standoff_m) > face_tol_m:
                continue                             # occluded, or not this body
            if float(_clearance(bodies, stance[None, :], bi)[0]) < BASE_CLEAR_M:
                why = "base_does_not_fit"
                continue
            lane = _lane_run_m(bodies, bi, q, n, standoff_m, run_in_m, run_out_m)
            if lane < min_lane_m:
                why = "no_lane_to_press_from"
                continue
            s_r, detour = _route_projection(poly, ds, stance)
            if detour > max_detour_m:
                # rejected per FACE, not per body: a wall normal that is
                # feasible only from the room on the other side must not win
                # the ranking and take the whole landmark down with it
                why = "too_far"
                continue
            # DETOUR, then lane, then where the normal came from -- each
            # coarsely binned so a hair's difference cannot outvote the next
            # criterion.  Order matters and was measured:
            #   * the WALL's normal outranks the body's own facets, because
            #     that is the direction the lump stands proud in -- the apex
            #     of a bulge, the front of a box -- and it keeps the press
            #     lane parallel to the wall the station machinery already
            #     follows.  Demoting it to a tie-break picked a tangential
            #     stance on both curved features: nearer the DRAWN line (the
            #     route bows around a bulge, so its flank is closer to the bow
            #     than its apex is) and a press whose lane runs into the wall
            #     the bulge grows out of;
            #   * lane length ahead of detour put prot_27's stance on the FAR
            #     side of the long wall -- a longer lane, in the next room --
            #     and the body was then dropped as `too_far` while sitting
            #     0.33 m off the route.
            key = (rank, round(detour / 0.15), -round(lane / 0.25))
            if best is None or key < best[0]:
                best = (key, n, q, stance, s_r, detour, lane)
        if best is None:
            skipped.append({"body": b.name, "reason": why})
            continue
        _key, n, q, stance, s_r, detour, lane = best
        if cyl:
            r = r_max
            sites.append(LandmarkSite(
                body=b.name, kind="cylinder", centre=tuple(c),
                n_out=tuple(n), face_xy=tuple(q), stance_xy=tuple(stance),
                s_route=s_r, detour_m=detour, lane_m=lane,
                reach_m=float(np.linalg.norm(q - c)), radius=r))
        else:
            bc, half, yaw = _box_axes(V)
            sites.append(LandmarkSite(
                body=b.name, kind="box", centre=tuple(bc),
                n_out=tuple(n), face_xy=tuple(q), stance_xy=tuple(stance),
                s_route=s_r, detour_m=detour, lane_m=lane,
                reach_m=float(np.linalg.norm(q - bc)),
                half_xy=half, yaw=yaw))
    sites.sort(key=lambda s: s.s_route)
    return sites, skipped
