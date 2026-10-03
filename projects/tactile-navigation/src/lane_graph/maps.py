"""V2 MVP candidate maps: generalized in-room layouts for contact-rich
tactile navigation.  STAGE 1 ONLY -- map specs and approval figures.  No
planner, no episodes; nothing here runs navigation.

Every layout is SEEDED-PARAMETRIC (jitter door positions, pillar and cabinet
placements with ``seed``; seed 0 = the canonical figures) and is designed
against the measured V1 constraints:

- press stance: commanded standoff 0.30-0.38 m to the pressed face, wrist
  cap ~0.43 m -> a face is TOUCH-FOLLOWABLE only where a walking lane
  ~0.56 m off the face keeps >= 0.28 m clearance from everything else;
- discrimination: contact-vs-free EDGES (jambs, wall ends, pillar
  silhouettes) are the physically robust channel; 0.08 m range steps are
  marginal on the real robot;
- chain-corner masking: a wall end flush with a perpendicular wall's face
  is invisible from that side (measured in V1); corner observability is a
  design consideration, not an assumption;
- spawn: the robot appears at map (0,0) facing +y with wobble -> every
  face must be >= 0.6 m from the start.

BUILDABILITY: the current MuJoCo task asset ships SIX fixed-length wall
boxes and no cylinder.  Map 1 builds with it today; maps 2-5 need the
Stage-2 asset extension (N generic wall segments + 1 cylinder + cabinet
boxes), counted per map in ``MapSpec.asset_note``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

THICK = 0.10
LANE_OFF = 0.56          # stance lane distance from a followed face
LANE_CLEAR = 0.28        # min clearance the lane needs from everything else
MIN_STRETCH = 0.7        # shortest face stretch worth following (stance+fan)


@dataclass(frozen=True)
class Wall:
    """Axis-aligned wall segment: p0->p1 along its axis, THICK wide."""
    p0: tuple[float, float]
    p1: tuple[float, float]

    def poly(self) -> np.ndarray:
        a, b = np.asarray(self.p0, float), np.asarray(self.p1, float)
        u = b - a
        u = u / max(np.linalg.norm(u), 1e-12)
        n = np.array([-u[1], u[0]])
        h = THICK / 2.0
        return np.array([a - h * n, b - h * n, b + h * n, a + h * n])


@dataclass(frozen=True)
class Pillar:
    center: tuple[float, float]
    radius: float = 0.18


@dataclass(frozen=True)
class Box:
    center: tuple[float, float]
    half: tuple[float, float]
    label: str = "cabinet"

    def poly(self) -> np.ndarray:
        c = np.asarray(self.center, float)
        hx, hy = self.half
        return np.array([c + [-hx, -hy], c + [hx, -hy], c + [hx, hy], c + [-hx, hy]])


@dataclass
class MapSpec:
    name: str
    title: str
    walls: list[Wall]
    pillars: list[Pillar]
    boxes: list[Box]
    rooms: list[tuple[float, float, float, float]]      # free-space rects (x0,y0,x1,y1)
    start: tuple[float, float]
    goal: tuple[float, float]
    goal_r: float
    doors: list[tuple[float, float, str]]               # label anchors
    landmarks: list[tuple[float, float, str]]
    story: str
    asset_note: str
    ambiguity: str
    seed: int = 0
    notes: list[str] = field(default_factory=list)


# ── geometry helpers ─────────────────────────────────────────────────────────────


def _seg_dist(pts: np.ndarray, a, b) -> np.ndarray:
    a, b = np.asarray(a, float), np.asarray(b, float)
    ab = b - a
    tt = np.clip(((pts - a) @ ab) / max(float(ab @ ab), 1e-12), 0.0, 1.0)
    proj = a + tt[:, None] * ab
    return np.linalg.norm(pts - proj, axis=1)


def _clearance(pts: np.ndarray, spec: MapSpec, skip=None) -> np.ndarray:
    d = np.full(len(pts), np.inf)
    for w in spec.walls:
        if w is skip:
            continue
        d = np.minimum(d, _seg_dist(pts, w.p0, w.p1) - THICK / 2.0)
    for p in spec.pillars:
        if p is skip:
            continue
        d = np.minimum(d, np.linalg.norm(pts - np.asarray(p.center), axis=1) - p.radius)
    for bx in spec.boxes:
        if bx is skip:
            continue
        c = np.asarray(bx.center, float)
        q = np.abs(pts - c) - np.asarray(bx.half, float)
        d = np.minimum(d, np.linalg.norm(np.maximum(q, 0.0), axis=1)
                       + np.minimum(np.maximum(q[:, 0], q[:, 1]), 0.0))
    return d


def _in_rooms(pts: np.ndarray, spec: MapSpec) -> np.ndarray:
    ok = np.zeros(len(pts), dtype=bool)
    for x0, y0, x1, y1 in spec.rooms:
        ok |= ((pts[:, 0] >= x0) & (pts[:, 0] <= x1)
               & (pts[:, 1] >= y0) & (pts[:, 1] <= y1))
    return ok


def followable_stretches(spec: MapSpec):
    """For every wall face side, the stretches where a 0.56 m stance lane is
    inside free space with >= 0.28 m clearance.  These are the candidate
    TOUCH_FOLLOW lanes stage 2 will turn into graph edges."""
    out = []
    for w in spec.walls:
        a, b = np.asarray(w.p0, float), np.asarray(w.p1, float)
        L = float(np.linalg.norm(b - a))
        if L < MIN_STRETCH:
            continue
        u = (b - a) / L
        n = np.array([-u[1], u[0]])
        for side in (+1.0, -1.0):
            ss = np.arange(0.0, L + 1e-9, 0.05)
            pts = a + ss[:, None] * u + side * (LANE_OFF + THICK / 2.0) * n
            good = (_clearance(pts, spec, skip=w) >= LANE_CLEAR) & _in_rooms(pts, spec)
            s0 = None
            for k, g in enumerate(list(good) + [False]):
                if g and s0 is None:
                    s0 = ss[min(k, len(ss) - 1)]
                elif not g and s0 is not None:
                    # the LAST GOOD sample ends the stretch (recording the
                    # first bad one put a failing endpoint on every lane and
                    # broke lane reconstruction); one further sample of
                    # margin at each end keeps endpoints strictly interior
                    s1 = ss[max(k - 2, 0)]
                    s0m = s0 + 0.05
                    if s1 - s0m >= MIN_STRETCH:
                        face = a + side * (THICK / 2.0 + 0.02) * n
                        out.append((face + s0m * u, face + s1 * u))
                    s0 = None
    return out


# ── the five candidate maps ──────────────────────────────────────────────────────


def m1_legacy_perimeter(seed: int = 0) -> MapSpec:
    """The V1 baseline kept as-is: contact_rich_gap's wall chain."""
    walls = [Wall((-2.5, 0.43), (3.3, 0.43)),    # wall0, now CONTINUOUS
             Wall((3.25, 0.48), (3.25, 2.48)),   # wall_1
             Wall((0.80, 2.43), (3.20, 2.43))]   # wall_2
    return MapSpec(
        name="m1_legacy_perimeter",
        title="Map 1 - legacy perimeter, gap replaced by a wall column",
        walls=walls, pillars=[Pillar((0.9, 0.43), 0.35)], boxes=[],
        rooms=[(-2.9, -1.2, 4.4, 0.38), (2.9, -1.2, 4.4, 3.4),
               (0.4, 2.48, 4.4, 3.4)],
        start=(0.0, 0.0), goal=(2.88, 2.86), goal_r=0.22,
        doors=[],
        landmarks=[(0.9, 0.43, "WALL COLUMN r=0.35 (bulges 0.35 into the lane)")],
        story=("The V1 contour kept, the landmark changed per review: the "
               "followed wall is continuous and a large half-embedded "
               "cylinder column bulges 0.35 m into the corridor exactly on "
               "the follow lane -- the robot must round it, and its "
               "silhouette edges + range bulge replace the gap as the "
               "registration evidence."),
        asset_note=("Needs Stage-2 assets: 3 wall segments + 1 cylinder "
                    "(the V1 boxes cover the walls; the column is new)."),
        ambiguity="Along-wall registration {0, +0.55 m}, wrong mode 0.55 mass.",
        seed=seed,
        notes=["Column surface to the nominal lane: 0.26 m -- the follow "
               "lane BREAKS around it (computed) and transit rounds it.",
               "Contour identical to V1 for comparability."])


def m2_pillar_room(seed: int = 0) -> MapSpec:
    rng = np.random.default_rng(seed)
    dj = rng.uniform(-0.15, 0.15, size=3) if seed else np.zeros(3)
    x1, y1 = 5.2, 3.8                                  # interior NE corner
    ds0, ds1 = 1.2 + dj[0], 2.1 + dj[0]                # south door
    de0, de1 = 1.4 + dj[1], 2.3 + dj[1]                # east door (goal exit)
    pil = (2.9 + dj[2], -0.85)                         # ON the south wall
    walls = [
        Wall((-0.85, -0.85), (ds0, -0.85)), Wall((ds1, -0.85), (5.25, -0.85)),
        Wall((-0.85, 3.85), (5.25, 3.85)),
        Wall((-0.85, -0.85), (-0.85, 3.85)),
        Wall((5.25, -0.85), (5.25, de0)), Wall((5.25, de1), (5.25, 3.85)),
    ]
    return MapSpec(
        name="m2_pillar_room", title="Map 2 - rectangular room, two doorways, pillar",
        walls=walls, pillars=[Pillar(pil, 0.40)], boxes=[],
        rooms=[(-0.8, -0.8, x1, y1)],
        start=(0.0, 0.0), goal=(5.0, 0.5 * (de0 + de1)), goal_r=0.22,
        doors=[(0.5 * (ds0 + ds1), -0.85, "entry door"),
               (5.25, 0.5 * (de0 + de1), "exit door (goal)")],
        landmarks=[(pil[0], pil[1],
                    "WALL COLUMN r=0.40 (grows from the south wall)")],
        story=("The normal room: free interior plus a LARGE cylindrical "
               "column growing out of the south wall, directly on the "
               "south-lane path from the entry side toward the east exit -- "
               "the follow lane must round it, and its silhouette/range "
               "bulge is the registration landmark."),
        asset_note=("Needs Stage-2 assets: 6 wall segments (2 door cuts) "
                    "+ 1 cylinder."),
        ambiguity=("Along-x registration {0, +0.55 m}: both doorway jambs and "
                   "the pillar discriminate; flat N/S faces alias."),
        seed=seed,
        notes=["Doorway width 0.9 m: transit only, no touch lane through it.",
               "Column bulges 0.35 m past the wall face; the S lane breaks "
               "around it (computed) -- rounding it is the obstacle "
               "transition."])


def m3_l_alcove(seed: int = 0) -> MapSpec:
    rng = np.random.default_rng(seed)
    dj = rng.uniform(-0.1, 0.1, size=2) if seed else np.zeros(2)
    stub_x = 1.6 + dj[0]
    cab = (3.1 + dj[1], -0.55)                         # flush against S wall
    walls = [
        Wall((-0.85, -0.85), (4.65, -0.85)),           # S of main arm
        Wall((-0.85, -0.85), (-0.85, 2.25)),           # W
        Wall((-0.85, 2.25), (2.55, 2.25)),             # N of main arm (west part)
        Wall((2.55, 2.25), (2.55, 4.85)),              # W of side arm
        Wall((4.65, -0.85), (4.65, 4.85)),             # E (shared)
        Wall((2.55, 4.85), (3.15, 4.85)),              # N of side arm w/ door
        Wall((4.05, 4.85), (4.65, 4.85)),
        Wall((stub_x, 2.25), (stub_x, 1.25)),          # alcove stub (internal)
    ]
    return MapSpec(
        name="m3_l_alcove", title="Map 3 - L-shaped room, alcove stub, cabinet",
        walls=walls, pillars=[],
        boxes=[Box(cab, (0.60, 0.35))],
        rooms=[(-0.8, -0.8, 4.6, 2.2), (2.6, 2.2, 4.6, 4.8)],
        start=(0.0, 0.0), goal=(3.6, 4.6), goal_r=0.22,
        doors=[(3.6, 4.85, "exit door (goal)")],
        landmarks=[(stub_x, 1.75, "alcove stub 1.0 m (free-standing edge)"),
                   (cab[0], cab[1], "WALL CABINET 1.2 x 0.7 (grows from S wall)")],
        story=("L topology, obstacles grown from the walls per review: the "
               "internal stub hangs 1.0 m off the north wall, and a LARGE "
               "cabinet (1.2 x 0.7) sits flush against the south wall right "
               "on the way to the elbow -- the S follow lane breaks around "
               "it and its two proud corners are the mid-leg evidence."),
        asset_note=("Needs Stage-2 assets: 8 wall segments + 1 cabinet box."),
        ambiguity=("Along-main-arm {0, +0.55 m}: the stub edge and cabinet "
                   "corners discriminate; the S face aliases."),
        seed=seed,
        notes=["Cabinet front face to stub tip: ~1.15 m transit corridor.",
               "Stub tip is a FREE-standing edge (not flush-masked)."])


def m4_twin_alcoves(seed: int = 0) -> MapSpec:
    rng = np.random.default_rng(seed)
    dj = float(rng.uniform(-0.1, 0.1)) if seed else 0.0
    a1, a2, aw, ad = 0.6, 3.4, 0.9, 0.6                # alcove x-starts, width, depth
    pil = (1.9 + dj, -0.85)                            # ON the south wall
    de0, de1 = 1.0, 1.9
    walls = [
        Wall((-0.85, -0.85), (6.25, -0.85)),
        Wall((-0.85, -0.85), (-0.85, 3.45)),
        Wall((6.25, -0.85), (6.25, de0)), Wall((6.25, de1), (6.25, 3.45)),
        # north wall with two identical alcove recesses
        Wall((-0.85, 3.45), (a1, 3.45)),
        Wall((a1 + aw, 3.45), (a2, 3.45)),
        Wall((a2 + aw, 3.45), (6.25, 3.45)),
        Wall((a1, 3.45), (a1, 3.45 + ad)), Wall((a1 + aw, 3.45), (a1 + aw, 3.45 + ad)),
        Wall((a1, 3.45 + ad), (a1 + aw, 3.45 + ad)),
        Wall((a2, 3.45), (a2, 3.45 + ad)), Wall((a2 + aw, 3.45), (a2 + aw, 3.45 + ad)),
        Wall((a2, 3.45 + ad), (a2 + aw, 3.45 + ad)),
    ]
    return MapSpec(
        name="m4_twin_alcoves", title="Map 4 - aliased twin alcoves + pillar",
        walls=walls, pillars=[Pillar(pil, 0.35)], boxes=[],
        rooms=[(-0.8, -0.8, 6.2, 3.4),
               (a1, 3.4, a1 + aw, 3.4 + ad), (a2, 3.4, a2 + aw, 3.4 + ad)],
        start=(0.0, 0.0), goal=(6.0, 0.5 * (de0 + de1)), goal_r=0.22,
        doors=[(6.25, 0.5 * (de0 + de1), "exit door (goal)")],
        landmarks=[(pil[0], pil[1],
                    "WALL COLUMN r=0.35 - the ONLY asymmetry"),
                   (a1 + aw / 2, 3.7, "alcove A"), (a2 + aw / 2, 3.7, "alcove B")],
        story=("The aliasing scene: two IDENTICAL alcoves 2.8 m apart mean "
               "every alcove-local touch pattern is explained by both "
               "registration modes.  A LARGE column grows from the south "
               "wall at an asymmetric position on the south follow lane -- "
               "contact where one mode predicts it and free under the other "
               "is the only decisive evidence."),
        asset_note=("Needs Stage-2 assets: 13 wall segments + 1 cylinder."),
        ambiguity=("Along-x registration {0, +2.8 m} (the alcove pitch): "
                   "structurally aliased by construction; pillar "
                   "discrimination 1.0."),
        seed=seed,
        notes=["Alcove jambs (4 edges each) are probe-reachable from the "
               "room-side lane; no lane fits inside the 0.9 m pockets.",
               "Prior separation equals the alcove pitch, not the V1 0.55."])


def m5_t_junction(seed: int = 0) -> MapSpec:
    rng = np.random.default_rng(seed)
    dj = rng.uniform(-0.1, 0.1, size=2) if seed else np.zeros(2)
    pil = (0.0 + dj[0], 4.45)                          # ON the north hall wall
    cab = (2.5 + dj[1], 4.0)                           # flush under N wall, east
    walls = [
        # south corridor (width 1.6), open at the top into the hall
        Wall((-0.85, -0.85), (0.85, -0.85)),
        Wall((-0.85, -0.85), (-0.85, 1.95)),
        Wall((0.85, -0.85), (0.85, 1.95)),
        # hall perimeter
        Wall((-3.25, 1.95), (-0.85, 1.95)),
        Wall((0.85, 1.95), (3.65, 1.95)),
        Wall((-3.25, 1.95), (-3.25, 4.45)),
        Wall((3.65, 1.95), (3.65, 4.45)),
        Wall((-3.25, 4.45), (3.65, 4.45)),
        # internal partition: narrow passage to the west goal room
        Wall((-1.6, 1.95), (-1.6, 3.5)),
    ]
    return MapSpec(
        name="m5_t_junction", title="Map 5 - T junction, partitioned goal, distractor",
        walls=walls, pillars=[Pillar(pil, 0.35)],
        boxes=[Box(cab, (0.70, 0.30))],
        rooms=[(-0.8, -0.8, 0.8, 2.0), (-3.2, 2.0, 3.6, 4.4)],
        start=(0.0, 0.0), goal=(-2.7, 3.2), goal_r=0.22,
        doors=[(-1.6, 3.95, "0.95 m passage (partition end)")],
        landmarks=[(pil[0], pil[1],
                    "WALL COLUMN r=0.35 (faces the corridor mouth)"),
                   (cab[0], cab[1], "WALL CABINET 1.4 x 0.6 (east branch)")],
        story=("Different topology: corridor north, T into a hall; the goal "
               "sits WEST behind an internal partition through a 0.95 m "
               "passage, the east hall is a physically real wrong branch "
               "with a LARGE wall cabinet.  A column grows from the north "
               "hall wall directly opposite the corridor mouth: the robot "
               "emerging from the corridor faces it, and touching it "
               "confirms the along-corridor registration before the "
               "west-vs-east commitment."),
        asset_note=("Needs Stage-2 assets: 9 wall segments + 1 cylinder "
                    "+ 1 cabinet box."),
        ambiguity=("Along-corridor {0, +0.55 m}: the junction mouth's two "
                   "jambs alias under the shift; pillar range/edge at a "
                   "known offset discriminates."),
        seed=seed,
        notes=["Wrong branch (east hall) is REAL geometry, not open floor.",
               "Partition end is a free-standing edge; its far side is the "
               "goal room's followable face."])


MAPS = {f.__name__: f for f in
        (m1_legacy_perimeter, m2_pillar_room, m3_l_alcove,
         m4_twin_alcoves, m5_t_junction)}


# ── rendering ────────────────────────────────────────────────────────────────────


def render_map(spec: MapSpec, out_png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Polygon as MplPoly

    fig, ax = plt.subplots(figsize=(9.2, 7.4), dpi=125)
    for w in spec.walls:
        ax.add_patch(MplPoly(w.poly(), closed=True, facecolor="#b9b9b9",
                             edgecolor="#5a5a5a", lw=0.9, zorder=2))
    for bx in spec.boxes:
        ax.add_patch(MplPoly(bx.poly(), closed=True, facecolor="#c8a06a",
                             edgecolor="#7a5c33", lw=1.0, zorder=2))
        ax.annotate(bx.label, bx.center, fontsize=7, ha="center", va="center",
                    color="#5a3f1e", zorder=6)
    for p in spec.pillars:
        ax.add_patch(Circle(p.center, p.radius, facecolor="#e7b3d4",
                            edgecolor="#c51b8a", lw=1.8, zorder=3))
    for s0, s1 in followable_stretches(spec):
        ax.plot([s0[0], s1[0]], [s0[1], s1[1]], "-", color="#2ca25f", lw=3.2,
                alpha=0.75, zorder=4, solid_capstyle="butt")
    ax.plot([], [], "-", color="#2ca25f", lw=3.2, alpha=0.75,
            label="touch-followable face (0.56 m lane fits, 0.28 m clear)")
    ax.plot([spec.start[0]], [spec.start[1]], "s", color="#111111", ms=10,
            zorder=6, label="start (spawn, faces +y)")
    ax.add_patch(Circle(spec.goal, spec.goal_r, facecolor="none",
                        edgecolor="#31a354", lw=2.2, zorder=5))
    ax.annotate("GOAL", np.asarray(spec.goal) + [0.05, spec.goal_r + 0.05],
                fontsize=9, color="#31a354", fontweight="bold", ha="center")
    for x, y, lab in spec.doors:
        ax.annotate(lab, (x, y), textcoords="offset points", xytext=(0, -14),
                    fontsize=7.5, color="#2b8cbe", ha="center",
                    fontweight="bold")
        ax.plot([x], [y], "^", color="#2b8cbe", ms=8, zorder=5)
    for x, y, lab in spec.landmarks:
        ax.annotate(lab, (x, y), textcoords="offset points", xytext=(0, 12),
                    fontsize=7.5, color="#c51b8a", ha="center",
                    fontweight="bold")
    ax.set_aspect("equal")
    ax.set_title(f"{spec.title}\n{spec.story}", fontsize=9, wrap=True)
    xs = [v[0] for w in spec.walls for v in w.poly()]
    ys = [v[1] for w in spec.walls for v in w.poly()]
    ax.set_xlim(min(xs) - 0.6, max(xs) + 0.6)
    ax.set_ylim(min(ys) - 0.9, max(ys) + 0.7)
    ax.legend(fontsize=7.5, loc="lower left")
    foot = (f"ambiguity: {spec.ambiguity}\nbuildability: {spec.asset_note}"
            + ("".join(f"\nnote: {n}" for n in spec.notes)))
    ax.text(0.0, -0.06, foot, transform=ax.transAxes, fontsize=7.5,
            color="#555555", va="top")
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    print(f"map -> {out_png}", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    for name, fn in MAPS.items():
        spec = fn(args.seed)
        render_map(spec, args.out / f"{spec.name}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
