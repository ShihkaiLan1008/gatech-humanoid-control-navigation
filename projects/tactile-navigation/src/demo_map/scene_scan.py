"""Any scanned room, as a scene the automatic pipeline can drive.

`scene_demo` and `scene_room2` each hard-wire one room: which walls, which
start, which goal, which corner.  This module takes the SCAN and a pair of
LANDMARK NAMES, and everything else follows -- the walls come from
`build_scan_map` (which already flattens a stack of shingled boards into one
coherent face per wall), the route from `auto_route`, the touches from the
compiler's own harvest.

THE MISSION IS A PAIR OF LANDMARKS, not a pair of coordinates.  "Start beside
the pole, finish at the box on the north wall" is a thing an owner can say
about a room they have walked through; a pair of metre coordinates is not.
The stances are then derived: the roomiest reachable point within pressing
distance of each, which is where the robot would have to stand anyway.
"""
from __future__ import annotations

import dataclasses
import itertools
import math

import numpy as np

from .geometry import Region, clearance, in_rooms
from .scan_map import build_scan_map

STANCE_CLEAR_M = 0.36
"""Clearance a spawn or goal stance needs: the base circle plus a little."""

STANCE_BAND_M = (0.45, 1.10)
"""How far from the landmark a stance may sit -- near enough that the visit is
about THIS body, far enough that the base is not inside its guard band."""


def landmarks(m) -> dict:
    """Every pressable non-wall body in the room, by name."""
    out = {}
    for b in m.boxes:
        out[b.bid] = (np.asarray(b.center, float),
                      f"box {2 * b.half[0]:.2f}x{2 * b.half[1]:.2f} m")
    for p in m.pillars:
        out[p.pid] = (np.asarray(p.center, float),
                      f"pillar r={p.radius:.2f} m")
    for h in getattr(m, "hulls", ()) or ():
        v = np.asarray(h.verts, float)
        out[h.hid] = (v.mean(axis=0), f"hull {len(v)} verts")
    return out


def stance_near(m, centre, *, rng_step: float = 0.05) -> np.ndarray | None:
    """The roomiest stance within pressing distance of a landmark.

    Derived rather than declared: the search is the annulus the arm can work
    in, the test is the map's own clearance and free-space membership, and the
    winner is the point with the most room.  A stance chosen any other way is
    a coordinate somebody typed."""
    lo, hi = STANCE_BAND_M
    c = np.asarray(centre, float)
    xs = np.arange(c[0] - hi, c[0] + hi + 1e-9, rng_step)
    ys = np.arange(c[1] - hi, c[1] + hi + 1e-9, rng_step)
    G = np.stack(np.meshgrid(xs, ys), -1).reshape(-1, 2)
    d = np.linalg.norm(G - c, axis=1)
    G = G[(d >= lo) & (d <= hi)]
    if not len(G):
        return None
    cl = clearance(G, m)
    ok = in_rooms(G, m) & (cl >= STANCE_CLEAR_M)
    if not ok.any():
        return None
    return G[ok][int(np.argmax(cl[ok]))]


def build_scan_room(walls_json, start_landmark: str, goal_landmark: str,
                    *, goal_radius_m: float = 0.40):
    """(map, scene) for a scanned room, on a landmark -> landmark mission."""
    from .scene_demo import build_demo_scene, translated_scan_map

    m = build_scan_map(walls_json)
    lm = landmarks(m)
    for name in (start_landmark, goal_landmark):
        if name not in lm:
            raise KeyError(f"{name!r} is not a landmark of this room; "
                           f"have {sorted(lm)[:12]}...")
    s_xy = stance_near(m, lm[start_landmark][0])
    g_xy = stance_near(m, lm[goal_landmark][0])
    for name, xy in ((start_landmark, s_xy), (goal_landmark, g_xy)):
        if xy is None:
            raise RuntimeError(f"no stance the base can take beside {name}")

    head = math.atan2(*(np.asarray(g_xy) - np.asarray(s_xy))[::-1])
    m = dataclasses.replace(
        m,
        start=Region("START", tuple(float(v) for v in s_xy), 0.30, head,
                     note=f"beside {start_landmark} ({lm[start_landmark][1]})"),
        goal=Region("GOAL", tuple(float(v) for v in g_xy), goal_radius_m, head,
                    note=f"at {goal_landmark} ({lm[goal_landmark][1]})"))
    mt = translated_scan_map(m)
    return mt, build_demo_scene(mt)


def corners(m, *, perp_dot: float = 0.35, meet_tol_m: float = 0.35) -> list:
    """Every vertex two wall faces actually make, from the map alone.

    A landmark is a corner or a cylinder (`mapv3.BOX_LANDMARKS`), and a
    scanned room may hold no cylinder at all -- 8_30_2026 has one 4.7 cm pole
    and nothing else -- so the corners ARE the landmarks there.  They are not
    stored anywhere: two faces meet or they do not, and this asks.

    ``meet_tol_m`` is how far past a face's own drawn end the intersection may
    fall and still count.  It has to allow some: a scan ends a face where the
    samples ran out, which is a little short of where the surfaces really
    meet, and demanding an exact touch would drop most real corners."""
    out = []
    for a, b in itertools.combinations(m.walls, 2):
        pa0, pa1 = np.asarray(a.p0, float), np.asarray(a.p1, float)
        pb0, pb1 = np.asarray(b.p0, float), np.asarray(b.p1, float)
        ta = pa1 - pa0
        tb = pb1 - pb0
        la, lb = np.linalg.norm(ta), np.linalg.norm(tb)
        if la < 1e-6 or lb < 1e-6:
            continue
        ta, tb = ta / la, tb / lb
        if abs(float(ta @ tb)) > perp_dot:
            continue
        A = np.stack([ta, -tb], axis=1)
        if abs(float(np.linalg.det(A))) < 1e-9:
            continue
        s, t = np.linalg.solve(A, pb0 - pa0)
        if not (-meet_tol_m <= s <= la + meet_tol_m):
            continue
        if not (-meet_tol_m <= t <= lb + meet_tol_m):
            continue
        out.append({"name": f"{a.wid}x{b.wid}", "xy": pa0 + s * ta,
                    "walls": (a.wid, b.wid)})
    return out


def corner_mission(walls_json, *, min_separation_m: float = 6.0):
    """Pick the two corners furthest apart that the base can stand beside.

    Furthest, because a mission that crosses the room exercises the pipeline;
    both stances legal, because a landmark the robot cannot stand beside is
    not one it can be sent to."""
    m = build_scan_map(walls_json)
    cs = [c for c in corners(m) if stance_near(m, c["xy"]) is not None]
    if len(cs) < 2:
        raise RuntimeError(f"only {len(cs)} reachable corners in this room")
    best, best_d = None, -1.0
    for a, b in itertools.combinations(cs, 2):
        d = float(np.linalg.norm(a["xy"] - b["xy"]))
        if d > best_d:
            best, best_d = (a, b), d
    return m, best, best_d


def build_corner_room(walls_json, *, goal_radius_m: float = 0.40):
    """(map, scene, mission) for a scanned room.

    A curated room states its own named-landmark mission in the JSON beside
    the solids it names, and that wins.  Without one, the mission falls back
    to the two wall corners furthest apart that the base can stand beside."""
    import json as _json
    from pathlib import Path as _Path

    from .scene_demo import build_demo_scene, translated_scan_map

    doc = _json.loads(_Path(walls_json).read_text())
    if doc.get("mission"):
        mi = doc["mission"]
        m = build_scan_map(walls_json)
        a = {"name": mi["start"]["name"], "xy": np.asarray(mi["start"]["xy"])}
        b = {"name": mi["goal"]["name"], "xy": np.asarray(mi["goal"]["xy"])}
        d = float(np.linalg.norm(a["xy"] - b["xy"]))
    else:
        m, (a, b), d = corner_mission(walls_json)
    s_xy, g_xy = stance_near(m, a["xy"]), stance_near(m, b["xy"])
    head = math.atan2(*(np.asarray(g_xy) - np.asarray(s_xy))[::-1])
    m = dataclasses.replace(
        m,
        start=Region("START", tuple(float(v) for v in s_xy), 0.30, head,
                     note=f"beside corner {a['name']}"),
        goal=Region("GOAL", tuple(float(v) for v in g_xy), goal_radius_m, head,
                    note=f"at corner {b['name']}"))
    mt = translated_scan_map(m)
    return mt, build_demo_scene(mt), {"from": a, "to": b, "apart_m": d}
