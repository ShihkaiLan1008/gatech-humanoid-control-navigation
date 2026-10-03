"""V3 candidate maps: true indoor rooms whose tactile features all GROW FROM
the walls.  STAGE 1 ONLY -- specs and approval figures; nothing here runs a
planner or a simulator.

DESIGN DRIVER: the new TOUCH_FOLLOW contract wants one valid contact before
EVERY forward increment (~0.35 m), so a map is only useful if it offers long
continuous wall lanes.  Each map is therefore built around 3-7 m followable
runs, and the figures report lane length and the touch increments it affords.

THE TACTILE VOCABULARY (all wall-attached, all validated by V1 measurements):

  alcove / bay   a recess: presses opposite it sweep FREE where the wall
                 would be -- the contact-vs-free channel that carried the
                 V1 gap landmark (valid frees, valid contacts, 11 touches)
  bulge          ahalf-embedded cylinder standing <= 0.15 m proud of the face:
                 a curved range step; shallow enough that the 0.56 m lane
                 still clears it by >= 0.28 m, so the lane stays SMOOTH
  protrusion     a box grown from the face (<= 0.25 m proud); two facing
                 protrusions make a narrow wall-grown passage
  corner         convex (wall ends, free past it) and concave (the L/T
                 bends) -- the turn evidence
  doorway        a cut in a run: transit only, no lane through it

GEOMETRY CONSTRAINTS (measured, V1):
  stance standoff to the pressed face 0.30-0.38 m, wrist reach cap ~0.43 m,
  parked-wrist envelope 0.26 m, lane 0.56 m off the face with >= 0.28 m
  clearance to everything else, spawn at map (0,0) facing +y needs 0.6 m.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

THICK = 0.10
LANE_OFF = 0.56
LANE_CLEAR = 0.28
MIN_STRETCH = 0.9
TOUCH_PITCH = 0.35


# ── primitives ───────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Wall:
    p0: tuple[float, float]
    p1: tuple[float, float]

    def poly(self) -> np.ndarray:
        a, b = np.asarray(self.p0, float), np.asarray(self.p1, float)
        u = b - a
        u = u / max(float(np.linalg.norm(u)), 1e-12)
        n = np.array([-u[1], u[0]])
        h = THICK / 2.0
        return np.array([a - h * n, b - h * n, b + h * n, a + h * n])


@dataclass(frozen=True)
class Bulge:
    """Half-embedded cylinder: attached to a wall, apex `proud` past the face."""
    center: tuple[float, float]
    radius: float
    label: str = ""


@dataclass(frozen=True)
class Prot:
    """Oriented box grown from a wall face."""
    center: tuple[float, float]
    half_along: float
    half_out: float
    u: tuple[float, float]
    label: str = ""
    half_height: float | None = None
    """Injected half height [m]; None = the scene default."""

    def poly(self) -> np.ndarray:
        c = np.asarray(self.center, float)
        u = np.asarray(self.u, float)
        u = u / max(float(np.linalg.norm(u)), 1e-12)
        n = np.array([-u[1], u[0]])
        return np.array([c - self.half_along * u - self.half_out * n,
                         c + self.half_along * u - self.half_out * n,
                         c + self.half_along * u + self.half_out * n,
                         c - self.half_along * u + self.half_out * n])


@dataclass(frozen=True)
class Feat:
    kind: str            # alcove | bulge | prot | door
    s: float             # centre along the run
    width: float = 0.8   # along-run extent (alcove/prot/door)
    depth: float = 0.5   # alcove recess depth; bulge/prot proud amount
    radius: float = 0.30 # bulge only
    label: str = ""


@dataclass
class MapV3:
    name: str
    title: str
    walls: list[Wall]
    bulges: list[Bulge]
    prots: list[Prot]
    rooms: list[tuple[float, float, float, float]]
    start: tuple[float, float]
    goal: tuple[float, float]
    goal_r: float
    doors: list[tuple[float, float, str]]
    landmarks: list[tuple[float, float, str]]
    topology: str
    story: str
    notes: list[str] = field(default_factory=list)
    open_boundaries: list = field(default_factory=list)


def run(a, b, inward, feats=()) -> tuple:
    """One wall run with wall-attached features.  Returns
    (walls, bulges, prots, landmark_anchors, door_anchors)."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    u = b - a
    L = float(np.linalg.norm(u))
    u = u / max(L, 1e-12)
    n = np.asarray(inward, float)
    n = n / max(float(np.linalg.norm(n)), 1e-12)
    walls, bulges, prots, lms, doors = [], [], [], [], []
    cuts: list[tuple[float, float]] = []
    for f in feats:
        s0, s1 = f.s - f.width / 2.0, f.s + f.width / 2.0
        if f.kind == "alcove":
            cuts.append((s0, s1))
            walls += [
                Wall(tuple(a + s0 * u - f.depth * n), tuple(a + s1 * u - f.depth * n)),
                Wall(tuple(a + s0 * u), tuple(a + s0 * u - f.depth * n)),
                Wall(tuple(a + s1 * u), tuple(a + s1 * u - f.depth * n))]
            lms.append((tuple(a + f.s * u - 0.45 * f.depth * n),
                        f.label or "alcove"))
        elif f.kind == "door":
            cuts.append((s0, s1))
            doors.append((tuple(a + f.s * u), f.label or "doorway"))
        elif f.kind == "bulge":
            c = a + f.s * u + (THICK / 2.0 + f.depth - f.radius) * n
            bulges.append(Bulge(tuple(c), f.radius, f.label))
            lms.append((tuple(a + f.s * u + (THICK / 2.0 + f.depth) * n),
                        f.label or "wall bulge"))
        elif f.kind == "prot":
            c = a + f.s * u + (THICK / 2.0 + f.depth / 2.0) * n
            prots.append(Prot(tuple(c), f.width / 2.0, f.depth / 2.0,
                              tuple(u), f.label))
            lms.append((tuple(a + f.s * u + (THICK / 2.0 + f.depth) * n),
                        f.label or "protrusion"))
    cuts.sort()
    s = 0.0
    for s0, s1 in cuts:
        if s0 - s > 0.05:
            walls.append(Wall(tuple(a + s * u), tuple(a + s0 * u)))
        s = s1
    if L - s > 0.05:
        walls.append(Wall(tuple(a + s * u), tuple(a + L * u)))
    return walls, bulges, prots, lms, doors


def _collect(*runs):
    W, B, P, L, D = [], [], [], [], []
    for w, b, p, l, d in runs:
        W += w; B += b; P += p; L += l; D += d
    return W, B, P, L, D


# ── geometry queries ─────────────────────────────────────────────────────────────


def _seg_dist(pts, a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ab = b - a
    t = np.clip(((pts - a) @ ab) / max(float(ab @ ab), 1e-12), 0.0, 1.0)
    return np.linalg.norm(pts - (a + t[:, None] * ab), axis=1)


def clearance(pts, m: MapV3, skip=None):
    d = np.full(len(pts), np.inf)
    for w in m.walls:
        if w is skip:
            continue
        d = np.minimum(d, _seg_dist(pts, w.p0, w.p1) - THICK / 2.0)
    for bg in m.bulges:
        d = np.minimum(d, np.linalg.norm(pts - np.asarray(bg.center), axis=1)
                       - bg.radius)
    for pr in m.prots:
        c = np.asarray(pr.center, float)
        u = np.asarray(pr.u, float)
        n = np.array([-u[1], u[0]])
        rel = pts - c
        q = np.abs(np.c_[rel @ u, rel @ n]) - np.array([pr.half_along, pr.half_out])
        d = np.minimum(d, np.linalg.norm(np.maximum(q, 0.0), axis=1)
                       + np.minimum(np.max(q, axis=1), 0.0))
    return d


def in_rooms(pts, m: MapV3):
    ok = np.zeros(len(pts), dtype=bool)
    for x0, y0, x1, y1 in m.rooms:
        ok |= ((pts[:, 0] >= x0) & (pts[:, 0] <= x1)
               & (pts[:, 1] >= y0) & (pts[:, 1] <= y1))
    for nx, ny, offset in m.open_boundaries:
        ok &= np.asarray(pts) @ np.array([nx, ny]) <= offset + 1e-9
    return ok


@dataclass
class Lane:
    """A base lane: a POLYLINE offset from the wall ENVELOPE, not a straight
    line at a fixed offset.  Where a bulge or jaw protrudes, the lane bows
    smoothly around it -- the way a person walks past a column -- which is
    what lets features be strongly convex while the base path stays smooth
    and every surface stays pressable at a 0.30-0.38 m stance."""
    face: np.ndarray
    lane: np.ndarray
    length: float


PRESS_GAP = 0.48     # lane-to-surface gap that keeps a 0.32 m stance a short hop away
SMOOTH_M = 1.10      # lane offset smoothing window


def followable(m: MapV3):
    """Lanes along each COLLINEAR RUN (grouped, so an alcove does not split
    its wall), inflated off the envelope and smoothed."""
    groups: dict = {}
    for w in m.walls:
        a, b = np.asarray(w.p0, float), np.asarray(w.p1, float)
        u = b - a
        L = float(np.linalg.norm(u))
        if L < 1e-6:
            continue
        u = u / L
        if u[0] < -1e-9 or (abs(u[0]) < 1e-9 and u[1] < 0):
            u = -u
        nrm = np.array([-u[1], u[0]])
        key = (round(float(math.atan2(u[1], u[0])), 3), round(float(a @ nrm), 3))
        g = groups.setdefault(key, {"u": u, "n": nrm, "walls": [],
                                    "smin": np.inf, "smax": -np.inf})
        g["walls"].append(w)
        for p in (a, b):
            sp = float(p @ u)
            g["smin"] = min(g["smin"], sp)
            g["smax"] = max(g["smax"], sp)

    out = []
    for g in groups.values():
        u, nrm = g["u"], g["n"]
        span = g["smax"] - g["smin"]
        if span < MIN_STRETCH:
            continue
        base = u * g["smin"] + nrm * (np.asarray(g["walls"][0].p0, float) @ nrm)
        others = [w for w in m.walls if w not in g["walls"]]
        for side in (+1.0, -1.0):
            ss = np.arange(0.0, span + 1e-9, 0.05)
            face = base + ss[:, None] * u + side * (THICK / 2.0 + 0.02) * nrm

            def clr(pts):
                d = np.full(len(pts), np.inf)
                for w in others:
                    d = np.minimum(d, _seg_dist(pts, w.p0, w.p1) - THICK / 2.0)
                for bg in m.bulges:
                    d = np.minimum(d, np.linalg.norm(
                        pts - np.asarray(bg.center), axis=1) - bg.radius)
                for pr in m.prots:
                    c = np.asarray(pr.center, float)
                    uu = np.asarray(pr.u, float)
                    nn = np.array([-uu[1], uu[0]])
                    rel = pts - c
                    q = np.abs(np.c_[rel @ uu, rel @ nn]) - np.array(
                        [pr.half_along, pr.half_out])
                    d = np.minimum(d, np.linalg.norm(np.maximum(q, 0.0), axis=1)
                                   + np.minimum(np.max(q, axis=1), 0.0))
                return d

            off = np.full(len(ss), LANE_OFF)
            for _ in range(40):
                pts = base + ss[:, None] * u + side * (off + THICK / 2.0)[:, None] * nrm
                need = clr(pts) < PRESS_GAP
                if not np.any(need) or float(off.max()) > LANE_OFF + 1.10:
                    break
                off = off + np.where(need, 0.03, 0.0)
            k = max(3, int(SMOOTH_M / 0.05) | 1)
            pad = k // 2
            off = np.convolve(np.pad(off, pad, mode="edge"),
                              np.ones(k) / k, "valid")[:len(ss)]
            lane = base + ss[:, None] * u + side * (off + THICK / 2.0)[:, None] * nrm
            ok = (clr(lane) >= LANE_CLEAR) & in_rooms(lane, m)
            s0 = None
            for i, gd in enumerate(list(ok) + [False]):
                if gd and s0 is None:
                    s0 = i
                elif not gd and s0 is not None:
                    i1 = max(i - 2, s0)
                    if (ss[i1] - ss[s0]) >= MIN_STRETCH:
                        out.append(Lane(face[s0:i1 + 1], lane[s0:i1 + 1],
                                        float(ss[i1] - ss[s0])))
                    s0 = None
    return out


def lane_stats(m: MapV3):
    st = followable(m)
    total = sum(l.length for l in st)
    return {"lanes": len(st), "total_m": total,
            "longest_m": max([l.length for l in st], default=0.0),
            "touch_increments": int(total / TOUCH_PITCH)}


def validate(m: MapV3) -> list[str]:
    """Buildability/safety checks every map must pass before approval."""
    bad = []
    for nm, p, need in (("start", m.start, 0.60), ("goal", m.goal, 0.60)):
        d = float(clearance(np.asarray([p], float), m)[0])
        if d < need:
            bad.append(f"{nm} clearance {d:.2f} < {need}")
        if not bool(in_rooms(np.asarray([p], float), m)[0]):
            bad.append(f"{nm} outside declared free space")
    st = lane_stats(m)
    if st["longest_m"] < 2.5:
        bad.append(f"longest lane {st['longest_m']:.1f} m < 2.5")
    if st["touch_increments"] < 20:
        bad.append(f"only {st['touch_increments']} touch increments")
    return bad


# ── the five candidate maps ──────────────────────────────────────────────────────


def r1_gallery() -> MapV3:
    """Rectangular gallery: one 7 m north wall carrying a rhythm of alcoves
    plus one distinctive double-width bay."""
    N = run((-1.0, 3.60), (7.2, 3.60), (0, -1), [
        Feat("alcove", 2.2, 0.85, 0.55, label="bay 1"),
        Feat("alcove", 3.7, 0.85, 0.55, label="bay 2"),
        Feat("alcove", 5.2, 0.85, 0.55, label="bay 3"),
        Feat("alcove", 7.15, 1.70, 0.55, label="DOUBLE BAY (unique)")])
    S = run((-1.0, -1.10), (7.2, -1.10), (0, 1),
            [Feat("door", 1.4, 0.95, label="entry")])
    Wl = run((-1.0, -1.10), (-1.0, 3.60), (1, 0))
    E = run((7.2, -1.10), (7.2, 3.60), (-1, 0),
            [Feat("door", 3.15, 0.95, label="exit (goal)")])
    W, B, P, L, D = _collect(N, S, Wl, E)
    return MapV3(
        name="r1_gallery", title="R1 - gallery: alcove rhythm on a 7 m wall",
        walls=W, bulges=B, prots=P,
        rooms=[(-0.95, -1.05, 7.15, 3.55)],
        start=(0.0, 0.0), goal=(6.35, 2.05), goal_r=0.22,
        doors=[(x, y, lab) for (x, y), lab in D],
        landmarks=[(x, y, lab) for (x, y), lab in L],
        topology="rectangular, single long touch wall",
        story=("The purest touch-step-touch demo: transit to the north lane, "
               "then follow ~7 m of wall pressing before every increment.  "
               "Three identical bays give a repeating contact/free rhythm; "
               "the double-width bay near the exit is the unique signature "
               "that pins the along-wall position before the door."),
        notes=["~7 m lane = ~20 touch increments at 0.35 m pitch.",
               "Bays are 0.55 m deep: a press opposite one sweeps free "
               "(beyond the 0.43 m reach) -- the validated V1 gap channel.",
               "Interior free space 8.2 x 4.7 m: transit is real, not a "
               "corridor walk."])


def r2_lbend() -> MapV3:
    """L room: continuous followable wall through a CONCAVE bend, with two
    strongly CONVEX curved bulges and a deep alcove."""
    S = run((-1.0, -1.10), (7.2, -1.10), (0, 1),
            [Feat("door", 1.4, 0.95, label="entry")])
    Wl = run((-1.0, -1.10), (-1.0, 2.30), (1, 0))
    Ninner = run((-1.0, 2.30), (4.30, 2.30), (0, -1), [
        Feat("bulge", 1.45, depth=0.48, radius=0.80,
             label="BULGE A - 1.47 m face, 0.48 m proud (~4 presses)"),
        Feat("alcove", 3.70, 1.60, 0.80,
             label="DEEP ALCOVE 1.6 x 0.8 (~4 free presses)")])
    Wupper = run((4.30, 2.30), (4.30, 5.60), (1, 0), [
        Feat("bulge", 1.70, depth=0.45, radius=0.75,
             label="BULGE B - 1.37 m face, 0.45 m proud")])
    E = run((7.2, -1.10), (7.2, 5.60), (-1, 0))
    Nup = run((4.30, 5.60), (7.2, 5.60), (0, -1),
              [Feat("door", 1.45, 0.95, label="exit (goal)")])
    W, B, P, L, D = _collect(S, Wl, Ninner, Wupper, E, Nup)
    return MapV3(
        name="r2_lbend", title="R2 - L room: concave bend + convex wall bulges",
        walls=W, bulges=B, prots=P,
        rooms=[(-0.95, -1.05, 7.15, 2.25), (4.35, 2.25, 7.15, 5.55)],
        start=(0.0, 0.0), goal=(5.75, 4.95), goal_r=0.22,
        doors=[(x, y, lab) for (x, y), lab in D],
        landmarks=[(x, y, lab) for (x, y), lab in L],
        topology="L-shaped, concave bend mid-follow",
        story=("Follow the inner wall west-to-east over a bulge that stands "
               "half a metre out of the wall, past a 1.6 x 0.8 m alcove, "
               "round the CONCAVE corner at (4.3, 2.3), reacquire the upper "
               "arm's west wall over a second bulge, exit north.  Each "
               "feature spans 3-5 touch increments AND deflects the lane, "
               "so it is felt as a shape, not a blip."),
        notes=["Bulges: r=0.80/0.75 cylinders whose centres sit only 0.32/"
               "0.30 m behind the face -- strongly convex caps 1.47/1.37 m "
               "wide standing 0.48/0.45 m proud.",
               "The base lane is offset from the wall ENVELOPE and smoothed, "
               "so it bows around each bulge instead of running straight -- "
               "smooth base path, deep features, no compromise."])


def r3_tpinch() -> MapV3:
    """T junction through a deep wall-grown pinch; goal west, decoy east."""
    Wcor = run((-1.15, -1.10), (-1.15, 3.80), (1, 0), [
        Feat("prot", 1.60, 1.40, 0.42,
             label="PINCH JAW W - 1.4 m x 0.42 proud")])
    Ecor = run((1.15, -1.10), (1.15, 3.80), (-1, 0), [
        Feat("prot", 1.60, 1.40, 0.42, label="PINCH JAW E"),
        Feat("alcove", 3.60, 1.60, 0.80,
             label="JUNCTION ALCOVE 1.6 x 0.8 (unique)")])
    Scor = run((-1.15, -1.10), (1.15, -1.10), (0, 1))
    Nhall = run((-3.90, 6.30), (3.90, 6.30), (0, -1), [
        Feat("bulge", 1.90, depth=0.48, radius=0.80,
             label="HALL BULGE - 1.47 m face, 0.48 m proud")])
    Whall = run((-3.90, 3.80), (-3.90, 6.30), (1, 0),
                [Feat("door", 1.30, 0.95, label="exit (goal)")])
    Ehall = run((3.90, 3.80), (3.90, 6.30), (-1, 0))
    Shall_w = run((-3.90, 3.80), (-1.15, 3.80), (0, 1))
    Shall_e = run((1.15, 3.80), (3.90, 3.80), (0, 1))
    W, B, P, L, D = _collect(Wcor, Ecor, Scor, Nhall, Whall, Ehall,
                             Shall_w, Shall_e)
    return MapV3(
        name="r3_tpinch", title="R3 - T junction through a deep wall-grown pinch",
        walls=W, bulges=B, prots=P,
        rooms=[(-1.10, -1.05, 1.10, 3.95), (-3.85, 3.75, 3.85, 6.25)],
        start=(0.0, 0.0), goal=(-3.15, 5.10), goal_r=0.22,
        doors=[(x, y, lab) for (x, y), lab in D],
        landmarks=[(x, y, lab) for (x, y), lab in L],
        topology="T junction, 2.3 m corridor, real decoy arm",
        story=("Touch-follow north up the corridor and squeeze the pinch: "
               "two 1.4 m jaws standing 0.42 m out of facing walls narrow "
               "2.3 m to 1.36 m, and the lane visibly bends through them.  "
               "The 1.6 x 0.8 m junction alcove then says WHICH side of the "
               "T the robot is on; it commits WEST (the east arm is real "
               "geometry) and follows the hall's north wall -- over a "
               "0.48 m bulge -- to the exit."),
        notes=["Corridor widened to 2.3 m so 0.42 m jaws still leave a "
               "1.36 m passage the base lane can bow through.",
               "Jaw, alcove and bulge are each 3-4 presses wide: the robot "
               "reads a SHAPE at every one."])


def r4_winding() -> MapV3:
    """Winding Z room: three bends on one continuous outer wall, one deep
    feature per leg.  Every passage is >= 1.34 m wide."""
    S = run((-0.95, -1.10), (0.95, -1.10), (0, 1))
    El = run((0.95, -1.10), (0.95, 0.50), (-1, 0))
    Sm = run((0.95, 0.50), (3.75, 0.50), (0, 1))
    Em = run((3.75, 0.50), (3.75, 3.25), (-1, 0))
    Su = run((3.75, 3.25), (6.05, 3.25), (0, 1))
    Eu = run((6.05, 3.25), (6.05, 5.05), (-1, 0),
             [Feat("door", 0.90, 0.95, label="exit (goal)")])
    N = run((1.80, 5.05), (6.05, 5.05), (0, -1))
    Wu = run((1.80, 2.35), (1.80, 5.05), (1, 0), [
        Feat("prot", 1.55, 1.30, 0.45,
             label="STEP LANDMARK - 1.3 m x 0.45 proud")])
    Nm = run((-0.95, 2.35), (1.80, 2.35), (0, -1), [
        Feat("alcove", 1.20, 1.50, 0.75, label="ALCOVE 1.5 x 0.75")])
    W = run((-0.95, -1.10), (-0.95, 2.35), (1, 0), [
        Feat("bulge", 2.30, depth=0.46, radius=0.78,
             label="BULGE - 1.42 m face, 0.46 m proud")])
    Wl, B, P, L, D = _collect(S, El, Sm, Em, Su, Eu, N, Wu, Nm, W)
    return MapV3(
        name="r4_winding", title="R4 - winding Z room: three bends, one wall",
        walls=Wl, bulges=B, prots=P,
        rooms=[(-0.90, -1.05, 0.90, 2.30), (-0.90, 0.55, 3.70, 2.30),
               (1.85, 0.55, 3.70, 5.00), (1.85, 3.30, 6.00, 5.00)],
        start=(0.0, 0.0), goal=(5.35, 4.15), goal_r=0.22,
        doors=[(x, y, lab) for (x, y), lab in D],
        landmarks=[(x, y, lab) for (x, y), lab in L],
        topology="winding Z, three consecutive bends",
        story=("The hardest turn test: one continuous outer wall (west -> "
               "north -> east) carries the robot through three bends without "
               "losing the followed surface.  A 0.46 m convex bulge, a "
               "1.5 x 0.75 m alcove and a 0.45 m step sit one per leg, each "
               "read over three to four presses -- and each deflecting the "
               "lane -- before its bend."),
        notes=["Rebuilt geometry: the earlier version had a 0.70 m neck the "
               "robot could not physically pass; every passage is now "
               ">= 1.34 m even with the deepest feature in it.",
               "Three lanes chained by turns: reacquire-after-turn, three "
               "times in one episode."])


def r5_twinbay() -> MapV3:
    """Repeated wall structure (two identical bays at 3.4 m pitch) resolved
    only by a distinctive stepped protrusion on the opposite wall."""
    S = run((-1.10, -1.20), (7.60, -1.20), (0, 1), [
        Feat("alcove", 2.10, 1.20, 0.70, label="BAY A"),
        Feat("alcove", 5.50, 1.20, 0.70, label="BAY B (identical)")])
    N = run((-1.10, 3.30), (7.60, 3.30), (0, -1), [
        Feat("prot", 3.60, 0.55, 0.24, label="STEP (unique, 0.24)"),
        Feat("prot", 4.25, 0.55, 0.12, label="step shoulder (0.12)")])
    Wl = run((-1.10, -1.20), (-1.10, 3.30), (1, 0))
    E = run((7.60, -1.20), (7.60, 3.30), (-1, 0),
            [Feat("door", 2.05, 0.95, label="exit (goal)")])
    W, B, P, L, D = _collect(S, N, Wl, E)
    return MapV3(
        name="r5_twinbay", title="R5 - twin bays (aliased) + unique wall step",
        walls=W, bulges=B, prots=P,
        rooms=[(-1.05, -1.15, 7.55, 3.25)],
        start=(0.0, 0.0), goal=(6.85, 0.90), goal_r=0.22,
        doors=[(x, y, lab) for (x, y), lab in D],
        landmarks=[(x, y, lab) for (x, y), lab in L],
        topology="rectangular, two touch walls, structural aliasing",
        story=("Follow the south wall east: the two IDENTICAL bays 3.4 m "
               "apart make the touch signature bimodal by construction -- "
               "the same contact/free pattern explains both registrations.  "
               "Crossing to the north wall and touching the two-step "
               "protrusion (0.24 then 0.12 proud) is the only evidence that "
               "separates them before the exit commitment."),
        notes=["Two long lanes (~8 m each side): the touch budget is the "
               "constraint, not the geometry.",
               "Bay pitch 3.4 m IS the ambiguity; the step is unique."])


MAPS3 = {f.__name__.split("_", 1)[0] + "_" + f.__name__.split("_", 1)[1]: f
         for f in (r1_gallery, r2_lbend, r3_tpinch, r4_winding, r5_twinbay)}


# ── rendering ────────────────────────────────────────────────────────────────────


def render(m: MapV3, out_png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Polygon as MplPoly

    st = followable(m)
    stats = lane_stats(m)
    fig, ax = plt.subplots(figsize=(10.0, 7.4), dpi=125)
    for w in m.walls:
        ax.add_patch(MplPoly(w.poly(), closed=True, facecolor="#b9b9b9",
                             edgecolor="#5a5a5a", lw=0.8, zorder=2))
    for bg in m.bulges:
        # draw the cylinder faintly and the PROTRUDING CAP solid: what the
        # room actually offers the hand is the cap; the rest is embedded in
        # the wall (attached by construction, never free-standing)
        ax.add_patch(Circle(bg.center, bg.radius, facecolor="none",
                            edgecolor="#c51b8a", lw=0.6, ls=":", zorder=2,
                            alpha=0.22))
        c = np.asarray(bg.center, float)
        near, nd = None, 1e9
        for w in m.walls:
            d = float(_seg_dist(c[None, :], w.p0, w.p1)[0])
            if d < nd:
                nd, near = d, w
        if near is not None:
            a0, b0 = np.asarray(near.p0, float), np.asarray(near.p1, float)
            uu = (b0 - a0) / max(float(np.linalg.norm(b0 - a0)), 1e-12)
            nn = np.array([-uu[1], uu[0]])
            if float((c - a0) @ nn) > 0:
                nn = -nn                       # nn points from wall to room
            cut = float((a0 - c) @ nn) + THICK / 2.0   # face offset along nn
            if abs(cut) < bg.radius:
                th = math.acos(np.clip(cut / bg.radius, -1, 1))
                base = math.atan2(nn[1], nn[0])
                ang = np.linspace(base - th, base + th, 24)
                cap = np.c_[c[0] + bg.radius * np.cos(ang),
                            c[1] + bg.radius * np.sin(ang)]
                ax.add_patch(MplPoly(cap, closed=True, facecolor="#d9a7c7",
                                     edgecolor="#c51b8a", lw=1.4, zorder=3))
    for pr in m.prots:
        ax.add_patch(MplPoly(pr.poly(), closed=True, facecolor="#c8a06a",
                             edgecolor="#7a5c33", lw=1.2, zorder=3))
    for ln in st:
        ax.plot(ln.face[:, 0], ln.face[:, 1], "-", color="#2ca25f", lw=3.4,
                alpha=0.8, zorder=4, solid_capstyle="butt")
        ax.plot(ln.lane[:, 0], ln.lane[:, 1], ":", color="#2ca25f", lw=1.5,
                alpha=0.95, zorder=4)
    ax.plot([], [], "-", color="#2ca25f", lw=3.4, alpha=0.8,
            label="touch-followable wall face")
    ax.plot([], [], ":", color="#2ca25f", lw=1.3, label="base lane (bows around features)")
    ax.plot([m.start[0]], [m.start[1]], "s", color="#111111", ms=10, zorder=6,
            label="start (faces +y)")
    ax.add_patch(Circle(m.goal, m.goal_r, facecolor="none", edgecolor="#31a354",
                        lw=2.2, zorder=5))
    ax.annotate("GOAL", np.asarray(m.goal) + [0.0, m.goal_r + 0.10], fontsize=9,
                color="#31a354", fontweight="bold", ha="center")
    for x, y, lab in m.doors:
        ax.plot([x], [y], "^", color="#2b8cbe", ms=9, zorder=6)
        ax.annotate(lab, (x, y), textcoords="offset points", xytext=(0, -15),
                    fontsize=7.5, color="#2b8cbe", ha="center", fontweight="bold")
    for x, y, lab in m.landmarks:
        ax.annotate(lab, (x, y), textcoords="offset points", xytext=(0, 8),
                    fontsize=7, color="#c51b8a", ha="center", fontweight="bold")
    ax.set_aspect("equal")
    ax.set_title(f"{m.title}\n{m.story}", fontsize=9, wrap=True)
    xs = [v[0] for w in m.walls for v in w.poly()]
    ys = [v[1] for w in m.walls for v in w.poly()]
    ax.set_xlim(min(xs) - 0.6, max(xs) + 0.6)
    ax.set_ylim(min(ys) - 0.9, max(ys) + 0.8)
    ax.legend(fontsize=7.5, loc="lower left")
    foot = (f"topology: {m.topology}\n"
            f"touch lanes: {stats['lanes']}  total {stats['total_m']:.1f} m  "
            f"longest {stats['longest_m']:.1f} m  "
            f"=> ~{stats['touch_increments']} touch increments @ 0.35 m"
            + "".join(f"\nnote: {n}" for n in m.notes))
    ax.text(0.0, -0.055, foot, transform=ax.transAxes, fontsize=7.5,
            color="#555555", va="top")
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    print(f"{m.name}: lanes={stats['lanes']} total={stats['total_m']:.1f}m "
          f"longest={stats['longest_m']:.1f}m touches~{stats['touch_increments']}"
          f" -> {out_png}", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    for fn in (r1_gallery, r2_lbend, r3_tpinch, r4_winding, r5_twinbay):
        m = fn()
        render(m, args.out / f"{m.name}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
