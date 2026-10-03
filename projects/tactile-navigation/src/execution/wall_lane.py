"""Wall-parallel lanes derived from the CONTACTED SURFACE.

A `WallSegment` is a straight stretch of real wall face plus the constant-offset
lane the base travels along it.  Everything here is arithmetic on the map's own
bodies -- no simulator, no controller, no scene names -- because the one lesson
of `docs/HW_ROBUSTNESS.md` section 10 is that the navigation-lane polyline is
the wrong source for a wall-follow heading (up to 44 deg off the surface it is
meant to parallel).  The heading a wall-parallel primitive travels on comes from
the surface tangent, and it comes from here.

Two questions are kept separate, because the answers differ along one wall:

  WALKABLE   `wall_segments` -- the lane point clears every body that is not
             part of the followed run, and it is enclosed by the map.  A bay,
             a doorway or a recess does not stop the base walking past it.
  TAPPABLE   `tap_stations` -- a ray cast from the lane at the wall hits the
             followed run at the requested standoff.  Bay mouths and doorways
             fail this and are simply skipped, which is why no map needs a
             hand-written station list.

Enclosure is what removes the lane on a wall's OUTWARD face: it passes every
other test (nothing near it, wall exactly at the standoff) and is outside the
building.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from global_nav.geometry import raycast

__all__ = ["WallSegment", "wall_segments", "tap_stations", "face_present"]

LANE_CLEAR_M = 0.28
"""Base half-width enforced everywhere else in the stack (maps_v3.LANE_CLEAR)."""


@dataclass(frozen=True)
class WallSegment:
    """A straight run of wall face, in MAP coordinates.

    ``p0`` is the surface point at arc length 0, ``tangent`` the travel
    direction along the face, ``normal`` the outward face normal (wall -> the
    side the robot travels on).  Arc length ``s`` runs 0 .. ``length``.
    """

    p0: np.ndarray
    tangent: np.ndarray
    normal: np.ndarray
    length: float
    bodies: tuple = ()
    name: str = ""

    def surface_point(self, s: float) -> np.ndarray:
        return np.asarray(self.p0, float) + float(s) * np.asarray(self.tangent, float)

    def lane_point(self, s: float, standoff_m: float) -> np.ndarray:
        return self.surface_point(s) + float(standoff_m) * np.asarray(self.normal, float)

    @property
    def yaw(self) -> float:
        """Map yaw of the travel direction -- the heading a wall-parallel walk
        holds.  Taken from the SURFACE, never from a lane polyline."""
        t = np.asarray(self.tangent, float)
        return float(math.atan2(t[1], t[0]))

    @property
    def hand_side(self) -> int:
        """+1 (left hand) when the wall lies on the body's +y while travelling
        along ``tangent``, -1 (right hand) otherwise.  Chosen from geometry, so
        one continuous segment always uses one hand."""
        t = np.asarray(self.tangent, float)
        lat = np.array([-t[1], t[0]])          # body +y at this heading
        return 1 if float(np.asarray(self.normal, float) @ lat) < 0.0 else -1

    def arc_length_of(self, xy_map) -> float:
        """Projection of a map point onto the segment's arc length."""
        d = np.asarray(xy_map, float).reshape(2) - np.asarray(self.p0, float)
        return float(d @ np.asarray(self.tangent, float))

    def standoff_of(self, xy_map) -> float:
        """Signed distance of a map point from the wall FACE, positive on the
        travelled side.  This is the quantity a constant-standoff walk holds."""
        d = np.asarray(xy_map, float).reshape(2) - np.asarray(self.p0, float)
        return float(d @ np.asarray(self.normal, float))

    def reversed(self) -> "WallSegment":
        """The same face travelled the other way (and therefore the other hand)."""
        return WallSegment(
            p0=self.surface_point(self.length), tangent=-np.asarray(self.tangent, float),
            normal=np.asarray(self.normal, float), length=self.length,
            bodies=self.bodies, name=self.name)

    def as_dict(self) -> dict:
        return {"p0": [float(v) for v in self.p0],
                "tangent": [float(v) for v in self.tangent],
                "normal": [float(v) for v in self.normal],
                "length_m": float(self.length),
                "yaw_rad": self.yaw, "hand": "left" if self.hand_side > 0 else "right",
                "bodies": list(self.bodies), "name": self.name}


# ── face extraction ──────────────────────────────────────────────────────────────


def _faces(bodies) -> list[tuple]:
    """Every outward polygon edge as (normal, tangent, offset, s0, s1, index)."""
    out = []
    for i, b in enumerate(bodies):
        v = np.asarray(b.verts, float)
        n = np.asarray(b.normals, float)
        for k in range(len(v)):
            a, c = v[k], v[(k + 1) % len(v)]
            e = c - a
            L = float(np.linalg.norm(e))
            if L < 1e-9:
                continue
            t = e / L
            out.append((n[k], t, float(a @ n[k]), float(a @ t), float(c @ t), i))
    return out


def _runs(faces, *, angle_tol_rad: float = 1e-3, offset_tol_m: float = 1e-3) -> list[dict]:
    """Merge coplanar, collinear faces into runs.

    An alcove splits its wall into three polygons that share one plane; grouping
    them keeps a bay from cutting the lane that passes it (the same grouping
    `maps_v3.followable` performs, done here on the bodies the belief carries
    rather than on the map spec).
    """
    runs: list[dict] = []
    for nk, t, off, s0, s1, idx in faces:
        lo, hi = min(s0, s1), max(s0, s1)
        for r in runs:
            if (float(r["n"] @ nk) > math.cos(angle_tol_rad)
                    and abs(r["off"] - off) <= offset_tol_m):
                r["smin"] = min(r["smin"], lo)
                r["smax"] = max(r["smax"], hi)
                r["bodies"].add(idx)
                break
        else:
            runs.append({"n": nk, "t": t, "off": off, "smin": lo, "smax": hi,
                         "bodies": {idx}})
    return runs


def _enclosed(point, bodies, *, max_range_m: float = 20.0) -> bool:
    """True when a ray in each axis direction terminates on a body.

    Every wall carries an outward face as well as an inward one, and the lane
    beside the outward face passes every other test: nothing is near it, and the
    wall is exactly at the requested standoff.  It is also outside the building.
    Enclosure is the map-independent way to say so.
    """
    for d in ((1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)):
        if raycast(point, np.asarray(d), max_range_m, bodies)[0] is None:
            return False
    return True


def _stretches(ok: np.ndarray) -> list[tuple[int, int]]:
    """Maximal runs of True as half-open index intervals."""
    out, start = [], None
    for j in range(len(ok) + 1):
        if j < len(ok) and ok[j]:
            start = j if start is None else start
        elif start is not None:
            out.append((start, j))
            start = None
    return out


def wall_segments(bodies, *, standoff_m: float, clearance_m: float = LANE_CLEAR_M,
                  min_length_m: float = 1.5, step_m: float = 0.05,
                  both_directions: bool = False) -> list[WallSegment]:
    """Straight wall stretches whose constant-standoff lane is WALKABLE.

    A stretch qualifies where the lane point clears every body outside the
    followed run by ``clearance_m`` and is enclosed by the map.  Face presence is
    NOT required here -- the base walks past a bay; only a tap needs a face.

    Returns every qualifying stretch, longest first.  With ``both_directions``
    each stretch is returned twice, once per travel direction (hence per hand).
    """
    bodies = tuple(bodies)
    segs: list[WallSegment] = []
    for r in _runs(_faces(bodies)):
        n, t = np.asarray(r["n"], float), np.asarray(r["t"], float)
        span = r["smax"] - r["smin"]
        if span < min_length_m:
            continue
        origin = t * r["smin"] + n * r["off"]      # surface point at s = 0
        m = max(2, int(round(span / step_m)) + 1)
        ss = np.linspace(0.0, span, m)
        others = [b for i, b in enumerate(bodies) if i not in r["bodies"]]
        ok = np.zeros(m, dtype=bool)
        for j, s in enumerate(ss):
            lane = origin + float(s) * t + float(standoff_m) * n
            clear = min((float(b.signed_distance(lane.reshape(1, 2))[0][0])
                         for b in others), default=float("inf"))
            ok[j] = clear >= clearance_m and _enclosed(lane, bodies)
        for a, b_ in _stretches(ok):
            s_a, s_b = float(ss[a]), float(ss[b_ - 1])
            if s_b - s_a < min_length_m:
                continue
            seg = WallSegment(
                p0=origin + s_a * t, tangent=t, normal=n, length=s_b - s_a,
                bodies=tuple(sorted(bodies[i].name for i in r["bodies"])),
                name=f"face@{r['off']:+.2f}/"
                     f"{math.degrees(math.atan2(n[1], n[0])):+.0f}")
            segs.append(seg)
            if both_directions:
                segs.append(seg.reversed())
    segs.sort(key=lambda s: -s.length)
    return segs


def face_present(seg: WallSegment, s: float, bodies, *, standoff_m: float,
                 tol_m: float = 0.02) -> bool:
    """Does a ray from the lane at arc length ``s`` hit the followed face?

    This is what separates a tappable station from a bay mouth, a doorway or the
    end of a run -- measured against the map the belief carries, so the same
    rule holds on every scene.
    """
    n = np.asarray(seg.normal, float)
    lane = seg.lane_point(s, standoff_m)
    d, nn, hit = raycast(lane, -n, float(standoff_m) + float(tol_m) + 0.30,
                         tuple(bodies))
    if d is None or abs(float(d) - float(standoff_m)) > tol_m:
        return False
    if float(np.asarray(nn, float) @ n) < 0.999:
        return False
    return not seg.bodies or bodies[hit].name in seg.bodies


def tap_stations(seg: WallSegment, spacing_m: float, bodies, *, standoff_m: float,
                 first_offset_m: float = 0.6, end_margin_m: float = 0.3,
                 hand_half_width_m: float = 0.06, tol_m: float = 0.02) -> list[float]:
    """Arc lengths at which a touch is both DUE and possible.

    Scheduling is by travelled arc length along the wall -- not by clock time,
    not by a map-specific station list -- so the same spacing gives the same
    rhythm on any segment of any map.  A station whose face is missing (bay
    mouth, doorway, run end) is dropped rather than special-cased.
    """
    spacing = max(float(spacing_m), 1e-3)
    limit = seg.length - float(end_margin_m)
    out, s = [], float(first_offset_m)
    while s <= limit + 1e-9:
        if s >= -1e-9 and all(
                face_present(seg, s + d, bodies, standoff_m=standoff_m, tol_m=tol_m)
                for d in (-hand_half_width_m, 0.0, hand_half_width_m)):
            out.append(round(float(s), 6))
        s += spacing
    return out
