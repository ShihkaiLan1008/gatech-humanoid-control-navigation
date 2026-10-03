"""Metric reconstruction of the measured demo room (docs/demo_maps/measured_map.jpg).

WHAT IS AUTHORITATIVE.  Every blue handwritten number on the sketch is a tape
measurement and is reproduced here verbatim, in metres, as a ``D_*`` constant.
Nothing else in this module is measured: every coordinate is *derived* from
those constants by the arithmetic written next to it, and the handful of
quantities the sketch never labels are collected in ``CHOSEN`` where they can
be read, argued with and changed in one place.

THE SKETCH IS NOT TO SCALE, so no coordinate may be taken from it by eye.  The
proof is in the drawing's own pixels: the 59.69 cm chord is drawn 113 px wide
while the 675.64 cm run beside it is drawn 173 px -- a local scale of 0.53 vs
3.91 cm/px, a factor of 7.4.  The vertical scale varies by a factor of 2.3
(1.11 cm/px across the lower corridor, 2.55 cm/px down the 1054.1 cm datum).
Reconstruction therefore chains labelled dimensions and nothing else.

FRAME.  Right-handed, floor at z = 0, metres.

    origin      the west end of curved feature C1's chord, ON the room-side
                face of the long wall.  This is the first tick of the sketch's
                dimension chain and the only unambiguous datum on that wall.
    +x          east, along the long wall, the direction of travel from start
    +y          north, out of the lower corridor and up toward the upper room
                (i.e. "up" in the drawing)

The long wall's room-side face IS the line y = 0.  Consequences worth stating
because they make the spec readable: the lower corridor occupies y in
[-1.6256, 0], every horizontal dimension on the long wall is directly an
x-coordinate, and the 1054.1 cm datum is the full span from the bottom wall
face to the upper room's south wall face.

FACES, NOT CENTRELINES.  A tape measure in a real room spans wall *surfaces*,
so every point stored here lies on the room-side face and walls are extruded
``THICKNESS`` away from the free space.  The MuJoCo/MapV3 adapter converts to
the centreline convention those APIs use; see ``demo_map.mapv3``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# ── labelled dimensions: the blue numbers, cm -> m ────────────────────────────
# Each is read directly off docs/demo_maps/measured_map.jpg.  The comment gives
# the two ticks the dimension arrow actually spans, verified against the black
# line work at pixel level (see the module docstring).

D_C1_CHORD = 0.5969      # 59.69 cm  C1 west tick -> C1 east tick (chord/opening)
D_C1_PROUD = 0.2921      # 29.21 cm  C1 apex depth into the corridor (sagitta)
D_RUN_C1_C2 = 6.7564     # 675.64 cm C1 east tick -> C2 west tick (clear run)
D_C2_CHORD = 0.7366      # 73.66 cm  C2 west tick -> C2 east tick (chord/opening)
D_C2_PROUD = 0.4699      # 46.99 cm  C2 apex depth into the corridor (sagitta)
D_RUN_C2_CORNER = 2.2606  # 226.06 cm C2 east tick -> inside corner
D_CORRIDOR_W = 2.4130    # 241.3 cm  corridor clear width, west face -> east face
D_LOWER_W = 1.6256       # 162.56 cm long wall face -> bottom wall face
D_BOTTOM_TO_UPPER = 10.5410  # 1054.1 cm bottom wall face -> upper room south face
D_CORNER_TO_DOOR = 1.4630    # 146.3 cm  corridor east face -> doorway west jamb
D_DOOR_W = 1.6256        # 162.56 cm doorway clear width

LABELLED = {
    "D_C1_CHORD": D_C1_CHORD,
    "D_C1_PROUD": D_C1_PROUD,
    "D_RUN_C1_C2": D_RUN_C1_C2,
    "D_C2_CHORD": D_C2_CHORD,
    "D_C2_PROUD": D_C2_PROUD,
    "D_RUN_C2_CORNER": D_RUN_C2_CORNER,
    "D_CORRIDOR_W": D_CORRIDOR_W,
    "D_LOWER_W": D_LOWER_W,
    "D_BOTTOM_TO_UPPER": D_BOTTOM_TO_UPPER,
    "D_CORNER_TO_DOOR": D_CORNER_TO_DOOR,
    "D_DOOR_W": D_DOOR_W,
}

# ── quantities the drawing never labels ───────────────────────────────────────
# Listed before use, as required.  Each records what the sketch's own local
# pixel scale would suggest, so the size of the licence taken is visible.

CHOSEN = {
    "west_stub_m": dict(
        value=1.50,
        what="long + bottom wall extent WEST of C1's west chord end",
        sketch_says="~0.25 m (51 px at C1's local 0.53 cm/px)",
        why="the dimension chain starts AT C1, so the tape never measured this "
            "stub; 1.50 m gives the robot a straight run to acquire the wall "
            "before the first curved feature (spawn needs 0.60 m clear, and "
            "TOUCH_PITCH is 0.35 m). The lower corridor is left OPEN at its "
            "west end, exactly as drawn -- no wall closes it."),
    "east_stub_m": dict(
        value=0.70,
        what="upper-room south wall from the doorway's east jamb to the east wall",
        sketch_says="0.61-0.80 m (43 px at the 1.43-1.85 cm/px local scale)",
        why="drawn but unlabelled; the mid-range value is taken."),
    "upper_room_west_ext_m": dict(
        value=5.50,
        what="how far the upper room reaches WEST of the corridor's west wall",
        sketch_says="~5.5 m (292 px at the corridor's 1.87 cm/px)",
        why="unlabelled and not route-relevant -- the route turns east on "
            "entering the room. Taken at the nearest labelled x-scale."),
    "upper_room_depth_m": dict(
        value=4.00,
        what="upper room north wall, i.e. its depth north of the south wall",
        sketch_says="3.4-7.7 m, depending which labelled datum sets the local "
                    "y-scale (162.56 cm gives 3.4, 1054.1 cm gives 7.7)",
        why="unlabelled, and the sketch cannot narrow it to better than 2x. "
            "Not route-relevant; 4.00 m keeps the scene compact."),
    "thickness_m": dict(
        value=0.10,
        what="wall thickness",
        sketch_says="nothing -- single-line sketch",
        why="the project standard (lane_graph.maps_v3.THICK, "
            "execution.scenes.THICKNESS, and the shipped maze-wall asset)."),
    "height_m": dict(
        value=1.80,
        what="wall height",
        sketch_says="nothing -- the sketch is a plan view",
        why="specified by the task."),
}

WEST_STUB = CHOSEN["west_stub_m"]["value"]
EAST_STUB = CHOSEN["east_stub_m"]["value"]
UPPER_WEST_EXT = CHOSEN["upper_room_west_ext_m"]["value"]
UPPER_DEPTH = CHOSEN["upper_room_depth_m"]["value"]
THICKNESS = CHOSEN["thickness_m"]["value"]
HEIGHT = CHOSEN["height_m"]["value"]

# ── route/demo parameters (not geometry: they describe how the robot uses it) ──
LANE_OFF = 0.56          # travel lane offset from a followed face (maps_v3.LANE_OFF)
SPAWN_OFF = 0.65         # spawn standoff; maps_v3.validate wants >= 0.60 m clear
START_RADIUS = 0.35
GOAL_RADIUS = 0.40


# ── primitives ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class WallSeg:
    """A straight wall.  ``p0``/``p1`` lie on the ROOM-SIDE FACE; the slab is
    extruded ``thickness`` in ``-inward``, away from the free space."""

    wid: str
    p0: tuple[float, float]
    p1: tuple[float, float]
    inward: tuple[float, float]
    source: str
    thickness: float = THICKNESS
    height: float = HEIGHT
    note: str = ""

    @property
    def length(self) -> float:
        return float(np.linalg.norm(np.subtract(self.p1, self.p0)))

    @property
    def centreline(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """Endpoints shifted half a thickness away from the room: the
        convention MapV3.Wall and the MuJoCo box geoms use."""
        n = np.asarray(self.inward, float)
        d = -0.5 * self.thickness * n
        return (tuple(np.add(self.p0, d)), tuple(np.add(self.p1, d)))


@dataclass(frozen=True)
class CurveFeat:
    """A curved wall feature standing proud of a straight run, reconstructed as
    the unique circle through its chord ends and its apex.  ``proud`` is the
    sagitta -- the labelled depth -- and ``radius`` follows from chord+sagitta,
    so BOTH labelled numbers are honoured exactly.  ``proud > chord/2`` (C2)
    puts the centre on the free-space side: a major arc, wider below the face
    than at it."""

    cid: str
    center: tuple[float, float]
    radius: float
    chord0: tuple[float, float]
    chord1: tuple[float, float]
    proud: float
    source: str
    height: float = HEIGHT
    note: str = ""

    @property
    def chord_len(self) -> float:
        return float(np.linalg.norm(np.subtract(self.chord1, self.chord0)))


@dataclass(frozen=True)
class BoxObs:
    """An oriented box for scanned structure that is not a long straight wall:
    door leaves, jambs, columns, furniture, oblique panels.

    A wall extractor emits infinite-thin faces with a free side; that model has
    nothing to say about an object standing in the room, so those surfaces used
    to be dropped on the floor.  This is the primitive that holds them, and it
    maps onto the ``Prot`` / "cabinet" obstacle the MuJoCo injection path
    already supports."""

    bid: str
    center: tuple[float, float]
    half: tuple[float, float]        # (half length along `yaw`, half across)
    yaw: float                       # radians
    z0: float
    z1: float
    source: str
    note: str = ""

    @property
    def height(self) -> float:
        return self.z1 - self.z0

    @property
    def u(self) -> tuple[float, float]:
        return (math.cos(self.yaw), math.sin(self.yaw))

    def poly(self) -> np.ndarray:
        c = np.asarray(self.center, float)
        u = np.asarray(self.u, float)
        n = np.array([-u[1], u[0]])
        a, b = self.half
        return np.array([c - a * u - b * n, c + a * u - b * n,
                         c + a * u + b * n, c - a * u + b * n])


@dataclass(frozen=True)
class CylObs:
    """A freestanding round obstacle read out of the scan -- a column, a pole,
    a bin.  Kept as the cylinder it is: a box would move its touchable surface
    by up to (sqrt(2)-1)r and break the surface normal everywhere but four
    points."""

    pid: str
    center: tuple[float, float]
    radius: float
    z0: float
    z1: float
    source: str
    note: str = ""

    @property
    def height(self) -> float:
        return self.z1 - self.z0


@dataclass(frozen=True)
class HullObs:
    """A low-poly convex prism for scanned structure a box would misrepresent:
    curved seating, oblique panels, shaped furniture.  ``verts`` trace the
    footprint; ``center``/``half``/``yaw`` carry the minimum-area rectangle as
    the fallback for consumers that only speak boxes (the V3 injection path)."""

    hid: str
    verts: tuple                      # ((x, y), ...) convex, <= ~10 corners
    center: tuple[float, float]
    half: tuple[float, float]
    yaw: float
    z0: float
    z1: float
    source: str
    note: str = ""

    @property
    def height(self) -> float:
        return self.z1 - self.z0

    @property
    def u(self) -> tuple[float, float]:
        return (math.cos(self.yaw), math.sin(self.yaw))

    def poly(self) -> np.ndarray:
        return np.asarray(self.verts, float)


@dataclass(frozen=True)
class Opening:
    """A gap in a wall run: transit only, no wall built across it."""

    oid: str
    p0: tuple[float, float]
    p1: tuple[float, float]
    source: str
    kind: str = "doorway"

    @property
    def width(self) -> float:
        return float(np.linalg.norm(np.subtract(self.p1, self.p0)))

    @property
    def mid(self) -> tuple[float, float]:
        return tuple(0.5 * (np.asarray(self.p0, float) + np.asarray(self.p1, float)))


@dataclass(frozen=True)
class Region:
    rid: str
    center: tuple[float, float]
    radius: float
    heading_rad: float
    note: str = ""


@dataclass
class DemoMap:
    walls: list[WallSeg]
    curves: list[CurveFeat]
    openings: list[Opening]
    rooms: list[tuple[float, float, float, float]]   # free-space rectangles
    start: Region
    goal: Region
    key_points: dict = field(default_factory=dict)
    boxes: list = field(default_factory=list)
    pillars: list = field(default_factory=list)   # CylObs
    hulls: list = field(default_factory=list)     # HullObs
    box_landmarks: bool = False
    open_boundaries: list = field(default_factory=list)
    """Navigation-only halfspaces (nx, ny, offset): n dot xy <= offset. No tactile surface."""
    """Whether this map's boxes and hulls are DELIBERATE bodies the scene
    must carry.  False is the demo room's owner directive (a landmark is a
    corner or a cylinder; scanned clutter is noise there).  A CURATED room
    sets True: its solids were placed on purpose -- a furniture island, a
    marker slab -- and dropping them would delete the mission."""

    def wall(self, wid: str) -> WallSeg:
        return next(w for w in self.walls if w.wid == wid)

    def curve(self, cid: str) -> CurveFeat:
        return next(c for c in self.curves if c.cid == cid)

    def opening(self, oid: str) -> Opening:
        return next(o for o in self.openings if o.oid == oid)


# ── derivation ────────────────────────────────────────────────────────────────


def arc_from_chord_sagitta(chord: float, sagitta: float) -> tuple[float, float]:
    """Circle through a chord of length ``chord`` whose apex stands ``sagitta``
    proud of it.  Returns (radius, back_off), where ``back_off`` is how far the
    centre sits BEHIND the chord line (away from the apex); it is negative when
    sagitta > chord/2 and the centre crosses to the apex side."""
    radius = (chord * chord / 4.0 + sagitta * sagitta) / (2.0 * sagitta)
    return radius, radius - sagitta


def build_demo_map() -> DemoMap:
    """Chain the labelled dimensions into coordinates.  Read top to bottom: no
    number appears that is not either a ``D_*`` above or arithmetic on ones
    already derived."""

    # -- the long wall's dimension chain, west to east along y = 0 -------------
    x_c1_w = 0.0                                  # origin, by definition
    x_c1_e = x_c1_w + D_C1_CHORD                  # 0.5969
    x_c2_w = x_c1_e + D_RUN_C1_C2                 # 7.3533
    x_c2_e = x_c2_w + D_C2_CHORD                  # 8.0899
    x_corner = x_c2_e + D_RUN_C2_CORNER           # 10.3505  (inside corner)
    x_west = x_c1_w - WEST_STUB                   # chosen

    # -- the corridor and the lower room --------------------------------------
    x_corr_e = x_corner + D_CORRIDOR_W            # 12.7635
    y_bottom = -D_LOWER_W                         # -1.6256
    y_upper_s = y_bottom + D_BOTTOM_TO_UPPER      # 8.9154

    # -- the upper room and its doorway ---------------------------------------
    x_door_w = x_corr_e + D_CORNER_TO_DOOR        # 14.2265
    x_door_e = x_door_w + D_DOOR_W                # 15.8521
    x_upper_e = x_door_e + EAST_STUB              # chosen
    x_upper_w = x_corner - UPPER_WEST_EXT         # chosen
    y_upper_n = y_upper_s + UPPER_DEPTH           # chosen

    # `inward` points from the face INTO the free space that face bounds.
    TO_N, TO_S, TO_E, TO_W = (0.0, 1.0), (0.0, -1.0), (1.0, 0.0), (-1.0, 0.0)

    walls = [
        WallSeg("W1", (x_west, 0.0), (x_c1_w, 0.0), TO_S,
                "chain start; west extent CHOSEN (west_stub_m)",
                note="long wall, west of C1"),
        WallSeg("W2", (x_c1_e, 0.0), (x_c2_w, 0.0), TO_S,
                "D_RUN_C1_C2 = 675.64 cm", note="long wall, between C1 and C2"),
        WallSeg("W3", (x_c2_e, 0.0), (x_corner, 0.0), TO_S,
                "D_RUN_C2_CORNER = 226.06 cm", note="long wall, C2 to the corner"),
        WallSeg("W4", (x_west, y_bottom), (x_corr_e, y_bottom), TO_N,
                "D_LOWER_W = 162.56 cm sets y; east end at the corridor east wall",
                note="bottom wall of the lower corridor"),
        WallSeg("W5", (x_corner, 0.0), (x_corner, y_upper_s), TO_E,
                "x from the long wall chain; y spans corridor mouth to upper room",
                note="corridor WEST wall"),
        WallSeg("W6", (x_corr_e, y_bottom), (x_corr_e, y_upper_s), TO_W,
                "D_CORRIDOR_W = 241.3 cm off W5", note="corridor EAST wall"),
        WallSeg("W7", (x_upper_w, y_upper_s), (x_corner, y_upper_s), TO_N,
                "D_BOTTOM_TO_UPPER = 1054.1 cm sets y; west extent CHOSEN",
                note="upper room south wall, WEST of the corridor mouth"),
        WallSeg("W8", (x_corr_e, y_upper_s), (x_door_w, y_upper_s), TO_N,
                "D_CORNER_TO_DOOR = 146.3 cm",
                note="upper room south wall, corner to the doorway"),
        WallSeg("W9", (x_door_e, y_upper_s), (x_upper_e, y_upper_s), TO_N,
                "D_DOOR_W = 162.56 cm then east extent CHOSEN",
                note="upper room south wall, doorway to the east wall"),
        WallSeg("W10", (x_upper_w, y_upper_s), (x_upper_w, y_upper_n), TO_E,
                "both extents CHOSEN", note="upper room west wall"),
        WallSeg("W11", (x_upper_w, y_upper_n), (x_upper_e, y_upper_n), TO_S,
                "depth CHOSEN (upper_room_depth_m)", note="upper room north wall"),
        WallSeg("W12", (x_upper_e, y_upper_s), (x_upper_e, y_upper_n), TO_W,
                "east extent CHOSEN", note="upper room east wall"),
    ]

    curves = []
    for cid, xw, xe, chord, proud, src in (
        ("C1", x_c1_w, x_c1_e, D_C1_CHORD, D_C1_PROUD,
         "D_C1_CHORD = 59.69 cm, D_C1_PROUD = 29.21 cm"),
        ("C2", x_c2_w, x_c2_e, D_C2_CHORD, D_C2_PROUD,
         "D_C2_CHORD = 73.66 cm, D_C2_PROUD = 46.99 cm"),
    ):
        radius, back = arc_from_chord_sagitta(chord, proud)
        # apex points into the corridor (-y), so the centre sits at +back.
        curves.append(CurveFeat(
            cid, (0.5 * (xw + xe), back), radius, (xw, 0.0), (xe, 0.0), proud, src,
            note=("minor arc: centre behind the wall face"
                  if back > 0 else
                  "major arc: centre INSIDE the corridor, widest below the face")))

    openings = [Opening("DOOR", (x_door_w, y_upper_s), (x_door_e, y_upper_s),
                        "D_CORNER_TO_DOOR then D_DOOR_W = 162.56 cm")]

    rooms = [
        (x_west, y_bottom, x_corr_e, 0.0),          # lower corridor
        (x_corner, 0.0, x_corr_e, y_upper_s),       # vertical corridor
        (x_upper_w, y_upper_s, x_upper_e, y_upper_n),  # upper room
    ]

    start = Region("START", (x_west + 0.20, -SPAWN_OFF), START_RADIUS, 0.0,
                   "green region: west end of the long wall, facing +x")
    goal = Region("GOAL", (openings[0].mid[0], y_upper_s + LANE_OFF),
                  GOAL_RADIUS, 0.0,
                  "red region: centred on the doorway, on the room side")

    return DemoMap(
        walls=walls, curves=curves, openings=openings, rooms=rooms,
        start=start, goal=goal,
        key_points=dict(
            x_c1_w=x_c1_w, x_c1_e=x_c1_e, x_c2_w=x_c2_w, x_c2_e=x_c2_e,
            x_corner=x_corner, x_west=x_west, x_corr_e=x_corr_e,
            y_bottom=y_bottom, y_upper_s=y_upper_s, y_upper_n=y_upper_n,
            x_door_w=x_door_w, x_door_e=x_door_e,
            x_upper_w=x_upper_w, x_upper_e=x_upper_e))


# ── measure the reconstruction back (Phase 2 validation) ──────────────────────


def remeasure(m: DemoMap) -> list[dict]:
    """Recover each labelled dimension FROM the built geometry, so the check is
    a measurement of the reconstruction and not a restatement of its inputs."""
    c1, c2 = m.curve("C1"), m.curve("C2")
    door = m.opening("DOOR")
    kp = m.key_points

    def circle_chord_on_face(c: CurveFeat, face_y: float = 0.0) -> float:
        """Width of the curve's opening where its circle crosses the wall face."""
        d = abs(c.center[1] - face_y)
        return 2.0 * math.sqrt(max(c.radius ** 2 - d ** 2, 0.0))

    def circle_proud(c: CurveFeat, face_y: float = 0.0) -> float:
        """How far the circle reaches past the face into the corridor."""
        return face_y - (c.center[1] - c.radius)

    return [
        dict(feature="C1 opening (chord)", labelled=D_C1_CHORD,
             recon=circle_chord_on_face(c1),
             how="circle C1 intersected with the wall face y=0"),
        dict(feature="C1 depth (proud of face)", labelled=D_C1_PROUD,
             recon=circle_proud(c1), how="face y=0 minus circle C1's lowest point"),
        dict(feature="C1->C2 clear run", labelled=D_RUN_C1_C2,
             recon=m.wall("W2").length, how="length of wall W2"),
        dict(feature="C2 opening (chord)", labelled=D_C2_CHORD,
             recon=circle_chord_on_face(c2),
             how="circle C2 intersected with the wall face y=0"),
        dict(feature="C2 depth (proud of face)", labelled=D_C2_PROUD,
             recon=circle_proud(c2), how="face y=0 minus circle C2's lowest point"),
        dict(feature="C2->corner run", labelled=D_RUN_C2_CORNER,
             recon=m.wall("W3").length, how="length of wall W3"),
        dict(feature="corridor clear width", labelled=D_CORRIDOR_W,
             recon=kp["x_corr_e"] - kp["x_corner"],
             how="W6 face x minus W5 face x"),
        dict(feature="lower corridor clear width", labelled=D_LOWER_W,
             recon=0.0 - kp["y_bottom"], how="long wall face y minus W4 face y"),
        dict(feature="bottom wall -> upper room south wall", labelled=D_BOTTOM_TO_UPPER,
             recon=kp["y_upper_s"] - kp["y_bottom"], how="W7/W8 face y minus W4 face y"),
        dict(feature="corridor east wall -> doorway", labelled=D_CORNER_TO_DOOR,
             recon=m.wall("W8").length, how="length of wall W8"),
        dict(feature="doorway clear width", labelled=D_DOOR_W,
             recon=door.width, how="gap between W8's east end and W9's west end"),
        dict(feature="long wall, C1 west end -> corner",
             labelled=D_C1_CHORD + D_RUN_C1_C2 + D_C2_CHORD + D_RUN_C2_CORNER,
             recon=kp["x_corner"] - kp["x_c1_w"],
             how="closure of the whole dimension chain"),
    ]


# ── distance field (used for clearance checks and plots) ──────────────────────


def _seg_dist(pts: np.ndarray, a, b) -> np.ndarray:
    a, b = np.asarray(a, float), np.asarray(b, float)
    ab = b - a
    t = np.clip(((pts - a) @ ab) / max(float(ab @ ab), 1e-12), 0.0, 1.0)
    return np.linalg.norm(pts - (a + t[:, None] * ab), axis=1)


def clearance(pts, m: DemoMap) -> np.ndarray:
    """Distance from each point to the nearest built SURFACE (negative inside).

    Walls are slabs, so the distance is taken to the centreline segment less
    half the thickness -- the same convention as ``lane_graph.maps_v3``."""
    pts = np.atleast_2d(np.asarray(pts, float))
    d = np.full(len(pts), np.inf)
    for w in m.walls:
        a, b = w.centreline
        d = np.minimum(d, _seg_dist(pts, a, b) - 0.5 * w.thickness)
    for c in m.curves:
        d = np.minimum(d, np.linalg.norm(pts - np.asarray(c.center), axis=1)
                       - c.radius)
    for b in m.boxes:
        u = np.asarray(b.u, float)
        n = np.array([-u[1], u[0]])
        rel = pts - np.asarray(b.center, float)
        q = np.abs(np.c_[rel @ u, rel @ n]) - np.asarray(b.half, float)
        d = np.minimum(d, np.linalg.norm(np.maximum(q, 0.0), axis=1)
                       + np.minimum(np.max(q, axis=1), 0.0))
    for p in m.pillars:
        d = np.minimum(d, np.linalg.norm(pts - np.asarray(p.center), axis=1)
                       - p.radius)
    for h in m.hulls:
        d = np.minimum(d, _convex_dist(pts, h.poly()))
    return d


def _convex_dist(pts, poly) -> np.ndarray:
    """Signed distance to a convex polygon: negative inside."""
    poly = np.asarray(poly, float)
    if len(poly) >= 3:
        # 2-D cross product, written out: numpy 2 removed the 2-vector form of
        # `np.cross` (it raises "must be 3-dimensional vectors"), which broke
        # every scene built from a hull -- demo_room among them.
        u, v = poly[1] - poly[0], poly[2] - poly[1]
        if float(u[0] * v[1] - u[1] * v[0]) < 0:
            poly = poly[::-1]              # ensure CCW
    dmin = np.full(len(pts), np.inf)
    inside = np.ones(len(pts), bool)
    for k in range(len(poly)):
        a, b = poly[k], poly[(k + 1) % len(poly)]
        dmin = np.minimum(dmin, _seg_dist(pts, a, b))
        e = b - a
        inside &= (pts - a) @ np.array([-e[1], e[0]]) >= 0.0
    return np.where(inside, -dmin, dmin)


def in_rooms(pts, m: DemoMap) -> np.ndarray:
    pts = np.atleast_2d(np.asarray(pts, float))
    ok = np.zeros(len(pts), dtype=bool)
    for x0, y0, x1, y1 in m.rooms:
        ok |= ((pts[:, 0] >= x0) & (pts[:, 0] <= x1)
               & (pts[:, 1] >= y0) & (pts[:, 1] <= y1))
    return ok
