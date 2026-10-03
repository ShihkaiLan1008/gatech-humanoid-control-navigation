"""The sparse reference route through the reconstructed room.

WHAT THIS IS.  A demo route, expressed the way the map expresses itself: as a
handful of geometric segments tied to named walls and features, not as a dense
list of poses.  Ten segments carry the whole run; the only arcs are the three
places where the geometry genuinely bends -- the two curved wall features and
the convex corner into the upper room.  ``sample()`` densifies a segment, but
only for plotting and clearance checks; nothing downstream stores the samples.

WHY THE LANE BOWS.  A followed lane is the offset curve of the wall ENVELOPE at
``LANE_OFF``, not a straight line at a fixed y.  It has to be: C2 stands 0.47 m
proud of the face, so a straight 0.56 m lane would pass 0.09 m from its apex --
a collision.  Offsetting the envelope instead, the lane swings out to 1.03 m
where C2 is and comes back, keeping every surface at a constant 0.56 m so the
press stance is a fixed step-in the whole way.  The offset of a half-space is a
half-space and the offset of a disc is a disc, so each bow is exactly a circular
arc of radius ``R + LANE_OFF`` -- closed form, no sampling.

WHICH HAND.  The route follows the long wall with the wall on the LEFT, then
crosses the corridor and follows the east wall and the upper room's south wall
with the wall on the RIGHT.  The hand therefore switches once, at R5.  That is
a property of the drawn route, not a choice made here, and it is the one thing
in this file a wall-following controller has to be told about.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .geometry import LANE_OFF, CurveFeat, DemoMap

# Radius of the lane bow around the convex corner into the upper room: the
# offset of a corner point is a circular arc of exactly the lane offset.
CORNER_R = LANE_OFF


@dataclass(frozen=True)
class RouteSeg:
    """One reference segment.  ``kind`` says what the robot is doing; ``follows``
    names the wall or feature the lane is offset from (empty when in free space)."""

    sid: str
    kind: str                 # approach | wall_follow | curve_follow | transit
                              # | corner_turn | goal_approach
                              # (only wall_follow/curve_follow/corner_turn hold
                              #  a constant offset to a followed surface)
    p0: tuple[float, float]
    p1: tuple[float, float]
    follows: str = ""
    hand: str = ""            # left | right | "" (no wall in reach)
    center: tuple[float, float] | None = None   # arcs only
    radius: float = 0.0                          # arcs only
    ccw: bool = True                             # arcs only
    note: str = ""

    @property
    def is_arc(self) -> bool:
        return self.center is not None

    @property
    def length(self) -> float:
        if not self.is_arc:
            return float(np.linalg.norm(np.subtract(self.p1, self.p0)))
        return self.radius * abs(self._sweep())

    def _sweep(self) -> float:
        c = np.asarray(self.center, float)
        a0 = math.atan2(self.p0[1] - c[1], self.p0[0] - c[0])
        a1 = math.atan2(self.p1[1] - c[1], self.p1[0] - c[0])
        d = a1 - a0
        if self.ccw:
            while d <= 0.0:
                d += 2.0 * math.pi
        else:
            while d >= 0.0:
                d -= 2.0 * math.pi
        return d

    def sample(self, ds: float = 0.02) -> np.ndarray:
        """Densify -- for plotting and clearance checks only."""
        n = max(2, int(math.ceil(self.length / ds)) + 1)
        if not self.is_arc:
            t = np.linspace(0.0, 1.0, n)[:, None]
            return np.asarray(self.p0, float) + t * np.subtract(self.p1, self.p0)
        c = np.asarray(self.center, float)
        a0 = math.atan2(self.p0[1] - c[1], self.p0[0] - c[0])
        a = a0 + np.linspace(0.0, self._sweep(), n)
        return c + self.radius * np.c_[np.cos(a), np.sin(a)]

    def heading0(self) -> float:
        p = self.sample(ds=max(self.length / 4.0, 1e-3))
        return math.atan2(*(p[1] - p[0])[::-1])

    def heading1(self) -> float:
        p = self.sample(ds=max(self.length / 4.0, 1e-3))
        return math.atan2(*(p[-1] - p[-2])[::-1])


@dataclass
class Route:
    segs: list[RouteSeg]
    start: tuple[float, float]
    goal: tuple[float, float]

    @property
    def length(self) -> float:
        return sum(s.length for s in self.segs)

    def waypoints(self) -> list[tuple[str, tuple[float, float]]]:
        """The sparse waypoint list: segment ends, in order.  Named P* so they
        never read as the map's W* wall IDs."""
        out = [("P0", self.segs[0].p0)]
        out += [(f"P{i + 1}", s.p1) for i, s in enumerate(self.segs)]
        return out

    def polyline(self, ds: float = 0.02) -> np.ndarray:
        return np.vstack([s.sample(ds) for s in self.segs])


def _bow(curve: CurveFeat, off: float, face_y: float = 0.0):
    """Where the offset curve leaves the straight lane and rejoins it.

    The lane is the boundary of (half-space ∪ disc) dilated by ``off``.  The
    straight part sits at ``face_y - off``; the bow is the arc of radius
    ``R + off`` about the curve's centre.  They meet where that circle crosses
    the straight lane."""
    cx, cy = curve.center
    rr = curve.radius + off
    dy = (face_y - off) - cy                 # signed drop from centre to the lane
    dx = math.sqrt(max(rr * rr - dy * dy, 0.0))
    return (cx - dx, face_y - off), (cx + dx, face_y - off), rr


def build_reference_route(m: DemoMap, lane_off: float = LANE_OFF) -> Route:
    """Green region -> long wall -> both curved features -> corridor transit ->
    east corridor wall -> convex corner -> upper south wall -> doorway."""
    kp = m.key_points
    c1, c2 = m.curve("C1"), m.curve("C2")
    lane_y = -lane_off                                   # lower corridor lane
    lane_x = kp["x_corr_e"] - lane_off                   # vertical corridor lane
    lane_y_up = kp["y_upper_s"] + lane_off               # upper room lane

    c1_in, c1_out, r1 = _bow(c1, lane_off)
    c2_in, c2_out, r2 = _bow(c2, lane_off)
    corner = (kp["x_corr_e"], kp["y_upper_s"])           # convex corner point
    door_w = (kp["x_door_w"], lane_y_up)

    segs = [
        RouteSeg("R0", "approach", m.start.center, c1_in, "W1", "left",
                 note="spawn and acquire the long wall; the only segment that "
                      "does NOT hold a constant offset -- it eases from the "
                      "0.65 m spawn standoff onto the 0.56 m lane"),
        # Passing a bump that lies to the LEFT while heading +x sweeps the lane
        # counter-clockwise about that bump's centre.
        RouteSeg("R1", "curve_follow", c1_in, c1_out, "C1", "left",
                 center=c1.center, radius=r1, ccw=True,
                 note="bow around C1 (0.29 m proud); apex lane depth "
                      f"{lane_off + c1.proud:.2f} m"),
        RouteSeg("R2", "wall_follow", c1_out, c2_in, "W2", "left",
                 note="the long straight run between the two curved features"),
        RouteSeg("R3", "curve_follow", c2_in, c2_out, "C2", "left",
                 center=c2.center, radius=r2, ccw=True,
                 note="bow around C2 (0.47 m proud); apex lane depth "
                      f"{lane_off + c2.proud:.2f} m"),
        RouteSeg("R4", "wall_follow", c2_out, (kp["x_corner"], lane_y), "W3", "left",
                 note="run out to the inside corner, where the long wall ends"),
        RouteSeg("R5", "transit", (kp["x_corner"], lane_y), (lane_x, lane_y), "",
                 "", note="free crossing of the corridor mouth: leave the long "
                          "wall on the left, arrive at the east wall on the "
                          "right. THE FOLLOWED HAND SWITCHES HERE."),
        RouteSeg("R6", "wall_follow", (lane_x, lane_y), (lane_x, kp["y_upper_s"]),
                 "W6", "right",
                 note="up the corridor's east wall, the longest followed run"),
        RouteSeg("R7", "corner_turn", (lane_x, kp["y_upper_s"]),
                 (kp["x_corr_e"], lane_y_up), "W6->W8", "right",
                 center=corner, radius=CORNER_R, ccw=False,   # a right turn
                 note="convex corner into the upper room: the followed wall "
                      "turns east, the lane rounds it at the lane offset"),
        RouteSeg("R8", "wall_follow", (kp["x_corr_e"], lane_y_up), door_w, "W8",
                 "right",
                 note="east along the upper room's south wall; the wall ending "
                      "at the doorway's west jamb is the arrival cue"),
        RouteSeg("R9", "goal_approach", door_w, m.goal.center, "DOOR", "",
                 note="step across the doorway opening to its centre"),
    ]
    return Route(segs, m.start.center, m.goal.center)


def route_report(m: DemoMap, r: Route) -> dict:
    """Numbers Phase 5 asks for: clearance, reach and continuity."""
    from .geometry import clearance, in_rooms

    pts = r.polyline(0.02)
    clr = clearance(pts, m)
    rows = []
    for s in r.segs:
        p = s.sample(0.02)
        c = clearance(p, m)
        rows.append(dict(sid=s.sid, kind=s.kind, follows=s.follows, hand=s.hand,
                         length_m=s.length, min_clear_m=float(c.min()),
                         max_clear_m=float(c.max())))
    gaps = [float(np.linalg.norm(np.subtract(b.p0, a.p1)))
            for a, b in zip(r.segs[:-1], r.segs[1:])]
    return dict(
        total_length_m=r.length,
        n_segments=len(r.segs),
        min_clearance_m=float(clr.min()),
        max_continuity_gap_m=max(gaps) if gaps else 0.0,
        all_inside_free_space=bool(in_rooms(pts, m).all()),
        segments=rows)
