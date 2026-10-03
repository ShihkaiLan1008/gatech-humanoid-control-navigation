"""The demo room as the iPhone scan actually measured it.

TWO MAPS, ON PURPOSE.  ``demo_map.geometry`` is the tape reconstruction and stays
exactly as it was -- it is the record of what the drawing says.  This module is
the scan-derived map: same frame, same primitives, but the walls come from
``docs/demo_maps/demo_map_scan_walls.json``, which
``scripts/extract_scan_walls.py`` produced from the mesh.

WHAT THE SCAN CHANGED, AND WHAT IT DID NOT.

  kept from the tape   the two curved features.  A phone mesh smooths a 0.3 m
                       bump -- it reads C1 as 0.24 m proud where the tape says
                       0.2921 -- so the tape's chord+sagitta circles are the
                       better geometry, and the scan's job was to confirm they
                       exist where predicted.  It does.
  taken from the scan  every wall position and extent.  The scan agrees with
                       the tape to <= 70 mm on all eleven labelled dimensions,
                       and it supplies the four extents the tape never measured
                       (the west stub, the upper room, the east stub) plus the
                       real building around them.
  corrected            the drawing's 1054.1 cm datum puts the upper room's south
                       wall 1.44 m south of where the scan finds the branch
                       corridor's end.  The scan wins: it agrees with every
                       other dimension, so the tape reading is the outlier.

The scan map is NOT a drop-in replacement for the tape map in the regression
tests: those assert the drawing, and the drawing has not changed.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .geometry import (GOAL_RADIUS, HEIGHT, LANE_OFF, SPAWN_OFF, START_RADIUS,
                       THICKNESS, BoxObs, CurveFeat, CylObs, DemoMap, HullObs,
                       Opening, Region, WallSeg, build_demo_map)

SCAN_WALLS = Path("docs/demo_maps/demo_map_scan_walls.json")

# Start unchanged from the tape map: the scan confirms open corridor there, and
# keeping it makes the two maps directly comparable.  The goal moves, because
# the branch corridor is 1.4 m longer than the drawing's datum implied.
GOAL_MIN_CLEAR = 0.60


def load_scan_walls(path: Path | str = SCAN_WALLS) -> dict:
    return json.loads(Path(path).read_text())


def build_scan_map(path: Path | str = SCAN_WALLS,
                   min_length: float = 0.30,
                   flatten_steps: bool = True) -> DemoMap:
    doc = load_scan_walls(path)
    tape = build_demo_map()

    walls = []
    faces = _dedupe(sorted(doc["faces"], key=lambda f: -f["length"]))
    if flatten_steps:
        faces = _close_corners(_join_runs(_flatten_boards(faces)))
    for i, f in enumerate(faces):
        if f["length"] < min_length:
            continue
        pos, s0, s1 = f["pos"], f["s0"], f["s1"]
        if f["axis"] == "x":
            p0, p1 = (pos, s0), (pos, s1)
        else:
            p0, p1 = (s0, pos), (s1, pos)
        walls.append(WallSeg(
            f"S{i:02d}", p0, p1, tuple(f["inward"]),
            source=f"scan face, {f['n_points']} samples, "
                   f"walkable side {f['walkable_frac']:.2f}",
            thickness=THICKNESS, height=HEIGHT,
            note=f"{f['axis']}-aligned at {pos:+.4f}"))

    # The curved features keep their tape geometry (see the module docstring),
    # but only for a scan registered onto the measured room -- a self-framed
    # scan of somewhere else has no C1/C2 to inherit.
    curves = ([CurveFeat(c.cid, c.center, c.radius, c.chord0, c.chord1, c.proud,
                         source=c.source + " (tape; confirmed by the scan)",
                         height=HEIGHT, note=c.note) for c in tape.curves]
              if doc.get("frame", "tape") == "tape" else [])

    boxes, pillars, hulls = [], [], []
    for i, b in enumerate(doc.get("clutter", [])):
        src = f"scan clutter cluster, {b['n_points']} samples"
        kind = b.get("kind", "box")
        if kind == "cyl":
            pillars.append(CylObs(
                f"P{i:02d}", tuple(b["center"]), b["radius"], b["z0"], b["z1"],
                source=src, note="round in plan: column, pole or bin"))
        elif kind == "hull":
            hulls.append(HullObs(
                f"H{i:02d}", tuple(tuple(v) for v in b["verts"]),
                tuple(b["center"]), tuple(b["half"]), b["yaw"],
                b["z0"], b["z1"], source=src,
                note="footprint a box would misstate by >18%: kept as a "
                     "convex prism"))
        else:
            boxes.append(BoxObs(
                f"B{i:02d}", tuple(b["center"]), tuple(b["half"]), b["yaw"],
                b["z0"], b["z1"], source=src,
                note="surface no wall face claimed: door leaf, jamb or "
                     "furniture"))

    openings = [Opening(f"O{i:02d}", tuple(o["p0"]), tuple(o["p1"]),
                        source="scan gap with walkable floor through it",
                        kind="doorway")
                for i, o in enumerate(doc.get("openings", []))]

    rooms = [tuple(r) for r in doc["free_rects"]]
    # anchored on the SAME faces the walls are built from: the route is
    # offset from these lines, and anchors read from the raw doc while the
    # walls flatten left the route 0.14 m off the end-room south wall --
    # every corner candidate there failed face_present at plan time
    anchors = _anchors(faces) if doc.get("frame", "tape") == "tape" else {}
    if anchors:
        start = Region("START", tape.start.center, START_RADIUS, 0.0,
                       "unchanged from the tape map; the scan confirms open "
                       "corridor")
        goal = _pick_goal(walls, rooms, anchors, boxes, pillars, hulls)
    else:
        # A scan the tape never measured: nothing external says where a demo
        # should begin or end, so take the longest traverse the room's own free
        # space allows at spawn clearance.
        start, goal = _pick_traverse(walls, curves, boxes, rooms,
                                     pillars, hulls)
    return DemoMap(walls=walls, curves=curves, openings=openings, rooms=rooms,
                   start=start, goal=goal, key_points=anchors, boxes=boxes,
                   pillars=pillars, hulls=hulls,
                   box_landmarks=bool(doc.get("boxes_are_landmarks")))


def _free_grid(m, rooms, step=0.05, min_clear=GOAL_MIN_CLEAR):
    """Grid of points inside the declared free space with room for the robot."""
    from .geometry import clearance, in_rooms

    r = np.asarray(rooms, float)
    xs = np.arange(r[:, 0].min(), r[:, 2].max() + step, step)
    ys = np.arange(r[:, 1].min(), r[:, 3].max() + step, step)
    G = np.stack(np.meshgrid(xs, ys), -1).reshape(-1, 2)
    ok = in_rooms(G, m) & (clearance(G, m) >= min_clear)
    return G[ok]


def _pick_traverse(walls, curves, boxes, rooms, pillars=(), hulls=()):
    """Start and goal for a room with no external datum: the two clear points
    furthest apart, so the demo crosses the space rather than sitting in it."""
    probe = DemoMap(walls=walls, curves=curves, openings=[], rooms=list(rooms),
                    start=Region("s", (0, 0), 0.1, 0.0),
                    goal=Region("g", (0, 0), 0.1, 0.0), boxes=list(boxes),
                    pillars=list(pillars), hulls=list(hulls))
    G = _free_grid(probe, rooms)
    if len(G) < 2:
        c = np.asarray(rooms, float).mean(0)
        p = (float(c[0]), float(c[1]))
        return (Region("START", p, START_RADIUS, 0.0, "free space too tight"),
                Region("GOAL", p, GOAL_RADIUS, 0.0, "free space too tight"))
    # an obstacle's clearance disc can sever a pocket of free cells from the
    # rest; a traverse must live inside ONE connected piece, so keep the
    # largest component before choosing the farthest pair
    from scipy import ndimage
    step = 0.05
    ij = np.round(G / step).astype(int)
    ij -= ij.min(0)
    raster = np.zeros(ij.max(0) + 1, bool)
    raster[ij[:, 0], ij[:, 1]] = True
    lab, n_lab = ndimage.label(raster, structure=np.ones((3, 3), int))
    if n_lab > 1:
        cell_lab = lab[ij[:, 0], ij[:, 1]]
        G = G[cell_lab == np.argmax(np.bincount(cell_lab)[1:]) + 1]
    # farthest pair, found via the two-sweep diameter heuristic on the point set
    a = G[np.argmax(np.linalg.norm(G - G.mean(0), axis=1))]
    b = G[np.argmax(np.linalg.norm(G - a, axis=1))]
    a = G[np.argmax(np.linalg.norm(G - b, axis=1))]
    yaw = float(np.arctan2(b[1] - a[1], b[0] - a[0]))
    return (Region("START", (float(a[0]), float(a[1])), START_RADIUS, yaw,
                   "longest clear traverse the scanned free space allows"),
            Region("GOAL", (float(b[0]), float(b[1])), GOAL_RADIUS, yaw,
                   "far end of that traverse"))


def build_scan_traverse(m: DemoMap, step: float = 0.05):
    """A collision-free reference path across a self-framed room.

    The demo route is specific to the measured corridor; a room the tape never
    saw gets the honest generic thing instead -- a shortest path over the free
    grid, thinned to the corners that actually turn."""
    from .geometry import clearance
    from .route import Route, RouteSeg

    G = _free_grid(m, m.rooms, step=step)
    key = {(round(p[0] / step), round(p[1] / step)): i for i, p in enumerate(G)}
    s = min(key, key=lambda k: (G[key[k]][0] - m.start.center[0]) ** 2
            + (G[key[k]][1] - m.start.center[1]) ** 2)
    g = min(key, key=lambda k: (G[key[k]][0] - m.goal.center[0]) ** 2
            + (G[key[k]][1] - m.goal.center[1]) ** 2)
    prev, frontier, seen = {s: None}, [s], {s}
    while frontier and g not in seen:
        nxt = []
        for cx, cy in frontier:
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1),
                           (1, 1), (1, -1), (-1, 1), (-1, -1)):
                n = (cx + dx, cy + dy)
                if n in key and n not in seen:
                    seen.add(n)
                    prev[n] = (cx, cy)
                    nxt.append(n)
        frontier = nxt
    if g not in seen:
        return None
    path, cur = [], g
    while cur is not None:
        path.append(G[key[cur]])
        cur = prev[cur]
    path = np.array(path[::-1])

    def thin(pts, tol=0.06):                       # Douglas-Peucker
        if len(pts) < 3:
            return pts
        a, b = pts[0], pts[-1]
        ab = b - a
        L = float(np.linalg.norm(ab))
        rel = pts - a
        d = (np.abs(ab[0] * rel[:, 1] - ab[1] * rel[:, 0]) / L if L > 1e-9   # 2-D cross
             else np.linalg.norm(pts - a, axis=1))
        i = int(np.argmax(d))
        if d[i] <= tol:
            return np.array([a, b])
        return np.vstack([thin(pts[:i + 1], tol)[:-1], thin(pts[i:], tol)])

    way = thin(path)
    segs = [RouteSeg(f"T{i}", "transit", tuple(way[i]), tuple(way[i + 1]), "", "",
                     note="shortest clear path across the scanned room")
            for i in range(len(way) - 1)]
    segs = [x for x in segs if x.length > 1e-6]
    return Route(segs, m.start.center, m.goal.center)


def _merge_pair(a, b):
    """Union two faces of one surface: extents unite, position is the
    sample-weighted mean, and the spread of contributing positions is tracked so
    a chain cannot walk a wall across its own drift."""
    wa, wb = a["n_points"], b["n_points"]
    lo = min(a.get("pos_lo", a["pos"]), b.get("pos_lo", b["pos"]))
    hi = max(a.get("pos_hi", a["pos"]), b.get("pos_hi", b["pos"]))
    return dict(a, pos=(a["pos"] * wa + b["pos"] * wb) / (wa + wb),
                pos_lo=lo, pos_hi=hi,
                s0=min(a["s0"], b["s0"]), s1=max(a["s1"], b["s1"]),
                length=max(a["s1"], b["s1"]) - min(a["s0"], b["s0"]),
                n_points=wa + wb, merged=a.get("merged", 1) + b.get("merged", 1))


def _flatten_boards(faces, depth_tol=0.25, join_gap=0.15, drift_tol=0.15):
    """One plane per contiguous wall run -- the demo ruling (2026-08-27).

    ``_dedupe`` deliberately keeps steps, recesses and bowed surfaces as
    separate boards: they are real structure the scan resolved.  This pass,
    run AFTER it, deliberately flattens them anyway -- the owner ruled the
    demo room's walls must read as ONE coherent surface each, not a stack
    of staggered panels.  The stagger is genuine scan output but it is
    noise at the scale the robot presses: the east wall's 4.6 cm step
    forced ``_stance_for`` to carry a raycast correction, and the east
    stub's single wall stood as five boards spread over 0.27 m.

    Two boards of one facing merge when their extents overlap -- the same
    stretch of wall resolved at two depths, up to ``depth_tol`` apart --
    or nearly touch (one wall drifting through panel joints: gap at most
    ``join_gap``, depth at most ``drift_tol``).  Each (axis, facing)
    group runs to a fixpoint; a merged board sits at the length-weighted
    depth of its parts.  Real openings -- doorways, the curved features'
    chords -- are 0.7 m gaps or wider and never bridge."""
    groups = {}
    for f in faces:
        sign = 1 if (f["inward"][0] + f["inward"][1]) > 0 else -1
        groups.setdefault((f["axis"], sign), []).append(dict(f))
    out = []
    for grp in groups.values():
        merged = True
        while merged:
            merged = False
            for i in range(len(grp)):
                for j in range(i + 1, len(grp)):
                    a, b = grp[i], grp[j]
                    ov = min(a["s1"], b["s1"]) - max(a["s0"], b["s0"])
                    dp = abs(a["pos"] - b["pos"])
                    if not ((ov > 0.02 and dp <= depth_tol)
                            or (-join_gap <= ov <= 0.02
                                and dp <= drift_tol)):
                        continue
                    la, lb = a["length"], b["length"]
                    a["pos"] = (a["pos"] * la + b["pos"] * lb) / (la + lb)
                    a["s0"] = min(a["s0"], b["s0"])
                    a["s1"] = max(a["s1"], b["s1"])
                    a["length"] = a["s1"] - a["s0"]
                    a["n_points"] += b["n_points"]
                    a["walkable_frac"] = (a["walkable_frac"] * la
                                          + b["walkable_frac"] * lb) / (la + lb)
                    del grp[j]
                    merged = True
                    break
                if merged:
                    break
        out.extend(grp)
    return out


def _join_runs(faces, max_gap: float = 1.0, depth_tol: float = 0.40,
               swallow_tol: float = 0.50):
    """One board per wall: bridge the gaps a wall is still a wall through, and
    drop the boards another board already spans.

    ``_flatten_boards`` merges only boards that OVERLAP or nearly touch
    (``join_gap`` 0.15 m), so it leaves a plane broken wherever the scan lost
    a stretch of it -- behind a curved feature's chord (0.74 m and 0.84 m in
    this room), across a panel joint the phone never resolved.  Those are not
    openings.  The openings this room has are the branch mouth and the corner
    into the end room, both 2.39 m, so a gap up to ``max_gap`` between boards
    of one facing at one depth is wall.  Bridging it is what makes the long
    hallway wall ONE surface with its two curved features standing proud of
    it, instead of five staggered panels with the features in the holes.

    The second pass says the same thing about depth.  A board whose whole
    extent lies inside a longer board of the same facing, within
    ``swallow_tol`` of its plane, is one wall the scan resolved twice -- a
    recess behind the surface in front of it, or a 0.44 m panel standing 0.37
    m proud of the plane that spans it.  The robot can only ever touch one of
    them; the longer board, which is the one the scan sampled most, is the one
    that survives.  ``depth_tol`` is 0.40 rather than the flatten's 0.25
    because the step between the branch-top wall's two panels is 0.39 m; at
    0.40 and at 0.50 this room reduces to the same twelve boards, so the
    threshold sits on a plateau rather than on a cliff.  ``swallow_tol`` is deliberately far smaller than the
    separation between DIFFERENT walls that happen to share an extent -- the
    hallway's two long walls are 12 m apart -- so containment alone never
    deletes a wall.
    """
    groups: dict = {}
    for f in faces:
        sign = 1 if (f["inward"][0] + f["inward"][1]) > 0 else -1
        groups.setdefault((f["axis"], sign), []).append(dict(f))
    out = []
    for grp in groups.values():
        merged = True
        while merged:
            merged = False
            for i in range(len(grp)):
                for j in range(i + 1, len(grp)):
                    a, b = grp[i], grp[j]
                    gap = max(a["s0"], b["s0"]) - min(a["s1"], b["s1"])
                    if gap > max_gap or abs(a["pos"] - b["pos"]) > depth_tol:
                        continue
                    la, lb = a["length"], b["length"]
                    a["pos"] = (a["pos"] * la + b["pos"] * lb) / (la + lb)
                    a["s0"] = min(a["s0"], b["s0"])
                    a["s1"] = max(a["s1"], b["s1"])
                    a["length"] = a["s1"] - a["s0"]
                    a["n_points"] += b["n_points"]
                    a["walkable_frac"] = (a["walkable_frac"] * la
                                          + b["walkable_frac"] * lb) / (la + lb)
                    del grp[j]
                    merged = True
                    break
                if merged:
                    break
        keep: list = []
        for f in sorted(grp, key=lambda g: -g["length"]):
            if any(g["s0"] - 1e-9 <= f["s0"] and f["s1"] <= g["s1"] + 1e-9
                   and abs(g["pos"] - f["pos"]) <= swallow_tol for g in keep):
                continue
            keep.append(f)
        out.extend(keep)
    return out


def _close_corners(faces, snap_tol: float = 0.15):
    """Re-square the corners the flatten just moved.

    A board that ended ON a neighbouring plane before ``_flatten_boards``
    shifted that plane now either overhangs it (an 8 cm hangnail past the
    branch mouth blocked every corner candidate on the outgoing face) or
    stops short of it.  Snap each board end to the nearest perpendicular
    plane within ``snap_tol`` -- sized to the largest plane shift the
    flatten produces here (the end-room south wall moved 0.14 m) -- provided
    that plane actually runs past the board's own line.  Ends further than
    ``snap_tol`` from any plane (scan edges, doorway sides) stay where the
    scan put them."""
    for f in faces:
        for end in ("s0", "s1"):
            best = None
            for g in faces:
                if g["axis"] == f["axis"]:
                    continue
                if not (g["s0"] - snap_tol <= f["pos"] <= g["s1"] + snap_tol):
                    continue
                d = abs(f[end] - g["pos"])
                if d <= snap_tol and (best is None or d < best[0]):
                    best = (d, g["pos"])
            if best is not None:
                f[end] = best[1]
        f["length"] = f["s1"] - f["s0"]
    return faces


def _dedupe(faces, near_tol: float = 0.16, join_gap: float = 0.30,
            far_tol: float = 0.045, overlap_frac: float = 0.5,
            max_spread: float = 0.06, same_surface: float = 0.025,
            step_depth: float = 0.030, step_overlap: float = 0.25):
    """Reduce the extractor's lines to one board per wall surface -- and ONLY
    per surface: two boards at different depths are a step, a recess or a
    proud strip, and averaging them away flattens exactly the structure a
    tactile robot navigates by.

    Three passes per (axis, facing) group, each run to a fixpoint:

      MERGE      lines within ``near_tol`` whose extents touch (or nearly) are
                 one drifting surface: unite them.  ``max_spread`` caps how far
                 a chain of small steps may drag the result, and two lines
                 that OVERLAP by more than ``step_overlap`` while sitting more
                 than ``step_depth`` apart are parallel structure -- a recess
                 face behind the wall plane -- and are never merged.
      DUPLICATE  lines within ``far_tol`` that cover EACH OTHER -- overlap
                 greater than ``overlap_frac`` of the LONGER -- are the same
                 wall found twice through its own scan thickness; keep the one
                 nearest the free space.  ``far_tol`` is the scan's surface
                 thickness (~4 cm), NOT a merge radius: anything deeper is a
                 real second surface (the old 0.28 m value ate a 6 cm recess
                 whole).
      INVARIANT  boards on the SAME plane (within ``same_surface``) may not
                 overlap: merge them.  Boards 2.5-10 cm apart that overlap are
                 left alone -- that is a step, and ``refine_runs`` in the
                 extractor re-derives each board's true extent so the step's
                 boards end where their surface ends.
      SHADOW     a deeper board within ~4.5 cm of a shallower one keeps only
                 the stretches the shallower does not cover.  That close, the
                 two planes are one bowed surface (the corridor west wall
                 bends 3 cm over its run); the front plane is the touchable
                 one where both exist, and averaging them instead would drag
                 anchors the corroboration tests pin to the tape.

    A feature standing PROUD of a wall is deliberately left alone: it is a
    different surface and it must intersect the wall it grows from.
    """
    def sign(f):
        return f["inward"][0 if f["axis"] == "x" else 1]

    def overlap(a, b):
        return min(a["s1"], b["s1"]) - max(a["s0"], b["s0"])

    groups: dict = {}
    for f in faces:
        groups.setdefault((f["axis"], tuple(f["inward"])), []).append(dict(f))

    out: list[dict] = []
    for g in groups.values():
        changed = True
        while changed:                                    # MERGE
            changed = False
            for i in range(len(g)):
                for j in range(i + 1, len(g)):
                    a, b = g[i], g[j]
                    if abs(a["pos"] - b["pos"]) > near_tol:
                        continue
                    if -overlap(a, b) > join_gap:
                        continue
                    if (overlap(a, b) > step_overlap
                            and abs(a["pos"] - b["pos"]) > step_depth):
                        continue           # parallel structure, not drift
                    lo = min(a.get("pos_lo", a["pos"]), b.get("pos_lo", b["pos"]))
                    hi = max(a.get("pos_hi", a["pos"]), b.get("pos_hi", b["pos"]))
                    if hi - lo > max_spread:
                        continue
                    g[i] = _merge_pair(a, b)
                    g.pop(j)
                    changed = True
                    break
                if changed:
                    break

        g.sort(key=lambda f: -sign(f) * f["pos"])         # nearest free space first
        keep: list[dict] = []
        for f in g:                                       # DUPLICATE
            if any(abs(k["pos"] - f["pos"]) <= far_tol
                   and overlap(k, f) > overlap_frac * max(k["length"], f["length"])
                   for k in keep):
                continue
            keep.append(f)

        changed = True
        while changed:                                    # INVARIANT
            changed = False
            for i in range(len(keep)):
                for j in range(i + 1, len(keep)):
                    if (abs(keep[i]["pos"] - keep[j]["pos"]) <= same_surface
                            and overlap(keep[i], keep[j]) > 1e-6):
                        keep[i] = _merge_pair(keep[i], keep[j])
                        keep.pop(j)
                        changed = True
                        break
                if changed:
                    break

        trimmed: list[dict] = []                          # SHADOW
        for f in keep:
            pieces = [(f["s0"], f["s1"])]
            for g in keep:
                if g is f:
                    continue
                dpos = abs(f["pos"] - g["pos"])
                # drift-scale only: two planes 2.5-4.5 cm apart are one bowed
                # surface and the front one is the touchable line.  Deeper
                # separations stay whole -- a recess plane wrongly removed
                # costs the robot 6-11 cm of surprise, a buried board costs
                # nothing (it sits inside the front slab's volume).
                if not (same_surface < dpos <= far_tol):
                    continue
                if sign(f) * f["pos"] >= sign(g) * g["pos"]:
                    continue                  # f is the shallower: keep whole
                shade = (g["s0"] - 0.02, g["s1"] + 0.02)
                nxt = []
                for a, b in pieces:
                    lo, hi = max(a, shade[0]), min(b, shade[1])
                    if hi - lo <= 1e-6:
                        nxt.append((a, b))
                        continue
                    if lo - a > 0.25:
                        nxt.append((a, lo))
                    if b - hi > 0.25:
                        nxt.append((hi, b))
                pieces = nxt
            for a, b in pieces:
                trimmed.append(dict(f, s0=a, s1=b, length=b - a))
        out += trimmed

    for f in out:
        f["length"] = f["s1"] - f["s0"]
    return sorted(out, key=lambda f: -f["length"])


def _pick_goal(walls, rooms, anchors, boxes=(), pillars=(), hulls=()) -> Region:
    """The drawing ends the route east along the far wall after the corridor.
    The scan's analogue is the end room's open space east of the branch: take
    the roomiest point there rather than asserting a coordinate."""
    from .geometry import clearance, in_rooms

    probe = DemoMap(walls=walls, curves=[], openings=[], rooms=list(rooms),
                    start=Region("s", (0, 0), 0.1, 0.0),
                    goal=Region("g", (0, 0), 0.1, 0.0), boxes=list(boxes),
                    pillars=list(pillars), hulls=list(hulls))
    # Search only the strip the robot can reach by following the end room's
    # south wall east -- the drawing's own final move.  A goal ON the travel
    # lane could never clear 0.60 m (the lane runs at 0.56 m off the followed
    # face), so the strip starts one step further out.
    xs = np.arange(anchors["branch_e_x"] + 0.4, anchors["room_e_x"], 0.02)
    ys = np.arange(anchors["room_s_y"] + 0.55, anchors["room_s_y"] + 1.35, 0.02)
    G = np.stack(np.meshgrid(xs, ys), -1).reshape(-1, 2)
    c = clearance(G, probe)
    ok = in_rooms(G, probe) & (c >= GOAL_MIN_CLEAR)
    if not ok.any():
        ok = c >= c.max() - 1e-9
    # as far east as the followed wall allows, then closest to that wall
    cand = G[ok]
    east = cand[cand[:, 0] >= cand[:, 0].max() - 1e-9]
    pick = east[np.argmin(east[:, 1])]
    return Region("GOAL", (float(pick[0]), float(pick[1])), GOAL_RADIUS, 0.0,
                  "end room east of the branch corridor: the roomiest point the "
                  "scan's own free space offers")


def _anchors(faces: list) -> dict:
    """The wall lines the route is offset from, chosen by length so that drift
    along a 26 m wall cannot pick a stub.  Fed the PROCESSED faces (deduped,
    flattened, corners closed), never the raw doc: the walls and the route
    must move together or the route presses air where a plane shifted."""
    F = [f for f in faces if f["length"] >= 0.5]

    def pick(axis, lo, hi, s_lo=-1e9, s_hi=1e9, key="length"):
        c = [f for f in F if f["axis"] == axis and lo <= f["pos"] <= hi
             and f["s1"] > s_lo and f["s0"] < s_hi]
        return max(c, key=lambda f: f[key]) if c else None

    long_w = pick("y", -0.2, 0.2, 0.0, 10.0)
    bot = pick("y", -1.95, -1.45, -3.0, 11.0)
    bw = pick("x", 10.0, 10.7)
    be = pick("x", 12.4, 13.1)
    if None in (long_w, bot, bw, be):
        return {}          # not the measured corridor; no demo anchors exist
    # the end room's south wall, east of the branch: northernmost such face,
    # because the lane must clear the one that sticks out furthest
    south = [f for f in F if f["axis"] == "y" and 10.2 <= f["pos"] <= 10.9
             and f["s0"] > be["pos"] - 0.2]
    room_s = max(south, key=lambda f: f["pos"]) if south else None
    room_e = pick("x", 15.4, 16.1)
    room_n = pick("y", 13.0, 14.0)
    return dict(
        long_wall_y=long_w["pos"], long_wall_s1=long_w["s1"],
        bottom_y=bot["pos"], branch_w_x=bw["pos"],
        branch_e_x=be["pos"], branch_end_y=be["s1"],
        room_s_y=room_s["pos"] if room_s else be["s1"],
        room_s_x1=max((f["s1"] for f in south), default=be["pos"] + 2.0),
        room_e_x=room_e["pos"] if room_e else be["pos"] + 3.0,
        room_n_y=room_n["pos"] if room_n else be["s1"] + 3.0)


def build_scan_route(m: DemoMap, lane_off: float = LANE_OFF):
    """The same ten-segment demo route, re-anchored on the scanned walls."""
    from .route import CORNER_R, Route, RouteSeg, _bow

    a = m.key_points
    c1, c2 = m.curve("C1"), m.curve("C2")
    lane_y = a["long_wall_y"] - lane_off
    lane_x = a["branch_e_x"] - lane_off
    lane_y_up = a["room_s_y"] + lane_off
    c1_in, c1_out, r1 = _bow(c1, lane_off, face_y=a["long_wall_y"])
    c2_in, c2_out, r2 = _bow(c2, lane_off, face_y=a["long_wall_y"])
    corner = (a["branch_e_x"], a["branch_end_y"])
    goal = m.goal.center

    segs = [
        RouteSeg("R0", "approach", m.start.center, c1_in, "long wall", "left",
                 note="spawn and acquire the hallway wall"),
        RouteSeg("R1", "curve_follow", c1_in, c1_out, "C1", "left",
                 center=c1.center, radius=r1, ccw=True, note="bow around C1"),
        RouteSeg("R2", "wall_follow", c1_out, c2_in, "long wall", "left",
                 note="the straight run between the curved features"),
        RouteSeg("R3", "curve_follow", c2_in, c2_out, "C2", "left",
                 center=c2.center, radius=r2, ccw=True, note="bow around C2"),
        RouteSeg("R4", "wall_follow", c2_out, (a["branch_w_x"], lane_y),
                 "long wall", "left", note="run out to the branch mouth"),
        RouteSeg("R5", "transit", (a["branch_w_x"], lane_y), (lane_x, lane_y),
                 "", "", note="free crossing of the branch mouth; the followed "
                              "hand switches LEFT -> RIGHT here"),
        RouteSeg("R6", "wall_follow", (lane_x, lane_y), (lane_x, a["branch_end_y"]),
                 "branch east wall", "right",
                 note="up the branch corridor, the longest followed run"),
        RouteSeg("R7", "corner_turn", (lane_x, a["branch_end_y"]),
                 (a["branch_e_x"], lane_y_up), "branch->end room", "right",
                 center=corner, radius=CORNER_R, ccw=False,
                 note="convex corner into the end room"),
        RouteSeg("R8", "wall_follow", (a["branch_e_x"], lane_y_up),
                 (goal[0], lane_y_up), "end room south wall", "right",
                 note="east along the end room's south wall"),
        RouteSeg("R9", "goal_approach", (goal[0], lane_y_up), goal, "", "",
                 note="step to the goal"),
    ]
    segs = [s for s in segs if s.length > 0.05]   # drop degenerate stubs
    return Route(segs, m.start.center, m.goal.center)
