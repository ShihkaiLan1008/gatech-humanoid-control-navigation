"""The two physical scenes, and the map the belief navigates them with.

FRAME.  Everything global is in the MAP frame: ``x`` runs along the robot's
travel direction on edge 0, ``y`` runs from the robot's path toward the wall it
faces.  The executive converts map<->world through the robot's start pose, which
is the one privileged piece of state a real robot also has ("you start at the
door, facing it").  After that the belief is on its own.

SCENE GEOMETRY IS CONSTRAINED BY THE ASSET.  The task ships six wall boxes with
FIXED half-lengths (1.0, 1.2, 1.4, 1.6, 1.1, 1.5), 0.10 m thick; a scene is a
choice of which to activate and where to put them.  Two consequences shape both
scenes: a follow-on wall must start at the corner (it would otherwise block the
leg the robot just walked), so every leg after the first is at least
``2 * half_length + standoff`` long; and no localized bump can be smaller than
the 0.10 m box thickness.

THE ROBOT STRAFES.  It faces the wall and walks sideways along it, hands parked,
then stops and presses.  So "the route" is a chain of wall faces, and a route
decision is: does this wall continue, or has it ended and the map says turn?

WHY THE AMBIGUOUS SCENE IS AMBIGUOUS.  Along-wall registration is bimodal:
delta in {0, +0.55} m -- the map's drawing of wall 0 slides along its own axis by
a junction-relevant amount, and the robot does not know which.  Both hypotheses
predict exactly the same thing for every forward probe near the start (flat wall
at the standoff, rank 1 -- Batch 1 measured that a flat-wall touch says how far
the wall is and nothing about where you are along it).  What separates them is
wall 0's NEAR EDGE, behind the robot: at a stance 0.75 m back the fan reads all
contact under one hypothesis and all free under the other -- the strongest
discrimination there is, and the same contact-vs-free channel the analytic
station model scores 1.0.

That evidence is BEHIND, i.e. away from the goal, and that is the whole point.
The forward alternative -- walk to the believed terminus and probe there -- does
resolve the ambiguity, but it delivers the answer AT the commit point rather
than before it: the fan reaches only 0.06-0.30 m ahead, so a stance that
straddles the terminus is already within a fan-length of the corner the robot
would have to start turning at.  The branch gate demands resolution BEFORE
committing, so the planner pays for the backward detour.  The no-touch baseline
commits on the prior instead, and turns at the wrong place when the prior's
heavier mode is the wrong one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from global_nav.floorplan import Floorplan, MapFeature, RouteEdge
from global_nav.geometry import ConvexPolygon
from .stations import G1Station

__all__ = ["SceneSpec", "WallPlacement", "build_scene", "SCENES"]

THICKNESS = 0.10
"""MAZE_WALL_THICKNESS -- asserted against the asset in test_scenes.py."""

HALF_LENGTHS = (1.0, 1.2, 1.4, 1.6, 1.1, 1.5)


@dataclass(frozen=True)
class WallPlacement:
    """One maze-wall box, in the MAP frame.  ``along`` is its long axis."""

    index: int
    center_map: tuple[float, float]
    along_map: tuple[float, float]

    @property
    def half_length(self) -> float:
        return HALF_LENGTHS[self.index]

    def polygon(self, name: str, signature: str = "surface") -> ConvexPolygon:
        c = np.asarray(self.center_map, float)
        a = np.asarray(self.along_map, float)
        a = a / max(float(np.linalg.norm(a)), 1e-12)
        n = np.array([-a[1], a[0]])
        hl, ht = self.half_length, THICKNESS / 2.0
        verts = np.array([c - hl * a - ht * n, c + hl * a - ht * n,
                          c + hl * a + ht * n, c - hl * a + ht * n])
        return ConvexPolygon(verts, name, signature)

    @property
    def angle_map(self) -> float:
        a = self.along_map
        return math.atan2(float(a[1]), float(a[0]))


@dataclass(frozen=True)
class SceneSpec:
    name: str
    walls: tuple[WallPlacement, ...]
    plan: Floorplan
    stations: dict
    standoff_m: float
    goal_map_xy: tuple[float, float]
    goal_radius_m: float
    goal_edge: int

    prior_modes: tuple | None = None
    """``((delta_m, mass), ...)`` -- bimodal ALONG-WALL registration of the map.
    ``None`` means a unimodal prior at delta = 0."""

    prior_sigma_along_m: float = 0.14
    prior_sigma_lateral_m: float = 0.06
    true_delta_m: float = 0.0
    """The registration the WORLD is actually built with.  Evaluation only --
    the walls are placed with it, the belief never sees it."""

    max_backup_m: float = 0.30
    n_route_decisions: int = 1
    """Turns the route requires.  Acceptance counts these."""

    disambiguating_stations: tuple[str, ...] = ()
    legs: tuple = ()
    fan_offsets_m: tuple = ()

    withholds_terminus_evidence: bool = False
    """This scene deliberately has NO station whose fan brackets a leg terminus.
    Declared, not accidental: such a stance sits within a fan-length of the
    commit point, so its evidence arrives at the decision rather than before it.
    A scene that withholds it forces the planner to find evidence elsewhere."""

    def wall_world(self, base0: np.ndarray, u: np.ndarray, v: np.ndarray, w: WallPlacement):
        """Map -> world for one wall: centre and angle, with the TRUE delta
        applied along edge 0's travel axis."""
        c = np.asarray(w.center_map, float) + np.array([self.true_delta_m, 0.0])
        centre = base0 + c[0] * u + c[1] * v
        a = np.asarray(w.along_map, float)
        along_w = a[0] * u + a[1] * v
        return centre, math.atan2(float(along_w[1]), float(along_w[0]))

    def map_to_world(self, base0: np.ndarray, u: np.ndarray, v: np.ndarray, xy_map) -> np.ndarray:
        p = np.asarray(xy_map, float).reshape(2)
        return np.asarray(base0, float) + p[0] * np.asarray(u, float) + p[1] * np.asarray(v, float)

    def world_to_map(self, base0: np.ndarray, u: np.ndarray, v: np.ndarray, xy_w) -> np.ndarray:
        d = np.asarray(xy_w, float).reshape(2) - np.asarray(base0, float)
        return np.array([float(d @ np.asarray(u, float)), float(d @ np.asarray(v, float))])


# ── shared construction ──────────────────────────────────────────────────────────


def _chain(standoff: float, wall_indices, first_near_m: float):
    """Place a chain of perpendicular walls, each starting at the previous
    wall's terminus, and return (placements, route edges, features, legs).

    Leg k is followed with the robot facing ``v_k`` and travelling ``u_k``; the
    next wall stands off the terminus in ``+v_k``, so the robot walks one
    standoff past the corner and turns to face back at it.  That is the geometry
    the frozen controller was trained on and the intermittent pipeline proved.
    """
    face = standoff + THICKNESS / 2.0
    u = np.array([1.0, 0.0])
    v = np.array([0.0, 1.0])
    origin = np.array([0.0, 0.0])

    placements, edges, features, legs = [], [], [], []
    near = float(first_near_m)
    for k, idx in enumerate(wall_indices):
        hl = HALF_LENGTHS[idx]
        # material spans [near, near + 2 hl] in this leg's along-coordinate
        terminus = near + 2.0 * hl
        centre_along = near + hl
        centre = origin + centre_along * u + face * v
        placements.append(WallPlacement(idx, tuple(centre), tuple(u)))
        features.append((origin + near * u + standoff * v, f"wall{k}_near", "wall_end"))
        features.append((origin + terminus * u + standoff * v, f"wall{k}_far", "wall_end"))
        legs.append({"origin": origin.copy(), "u": u.copy(), "v": v.copy(),
                     "near": near, "terminus": terminus, "index": idx})
        # next leg: past the corner by one standoff, facing back along -u
        origin = origin + (terminus + standoff) * u
        u, v = v.copy(), -u
        near = face + THICKNESS / 2.0
    return placements, edges, features, legs


def _build(name: str, wall_indices, *, standoff: float, first_near_m: float,
           goal_leg_m: float, half_width: float = 0.55, **kw) -> SceneSpec:
    placements, _, features, legs = _chain(standoff, wall_indices, first_near_m)

    bodies = tuple(p.polygon(f"wall_{k}") for k, p in enumerate(placements))
    feats = tuple(MapFeature(i, tuple(np.asarray(xy, float)), sig, nm)
                  for i, (xy, nm, sig) in enumerate(features))

    edges = []
    for k, leg in enumerate(legs):
        a = leg["origin"]
        span = (leg["terminus"] + standoff) if k + 1 < len(legs) else goal_leg_m
        b = a + span * leg["u"]
        succ = (k + 1,) if k + 1 < len(legs) else ()
        edges.append(RouteEdge(k, tuple(a), tuple(b), half_width, succ, f"leg{k}"))
    plan = Floorplan(bodies, feats, tuple(edges), route=tuple(range(len(legs))), name=name)

    fan = tuple(kw.get("fan_offsets_m", (0.06, 0.12, 0.18, 0.24, 0.30)))
    stations: dict[str, G1Station] = {}
    for k, leg in enumerate(legs):
        a, u, v = leg["origin"], leg["u"], leg["v"]
        length = float(np.linalg.norm(np.asarray(edges[k].b) - np.asarray(edges[k].a)))
        # stations every ~0.35 m along the leg, plus one BEHIND the start of
        # leg 0 aimed at the wall's near edge (the ambiguous scene's detour)
        offsets = [round(float(s), 3) for s in np.arange(0.0, length + 1e-6, 0.35)]
        # Stations that BRACKET the terminus.  A uniform grid does not: the fan
        # spans [0.06, 0.30] ahead, so a stance only straddles a terminus from a
        # 0.24 m-wide window, and a 0.35 m grid can step right over it -- measured,
        # and it left the belief permanently one-sided (all-contact then all-free,
        # sigma stuck at 0.13 m, the branch gate never opening).
        term = float(leg["terminus"])
        mid = 0.5 * (min(fan) + max(fan))
        if k not in kw.get("no_bracket_legs", ()):
            for back in (mid, mid + 0.10, mid - 0.10):
                s = round(term - back, 3)
                if -0.05 <= s <= length + 0.35:
                    offsets.append(s)
        if k == 0 and kw.get("back_station_m"):
            offsets = [-float(kw["back_station_m"])] + offsets
        offsets = sorted(dict.fromkeys(offsets))
        for j, s in enumerate(offsets):
            node = f"L{k}s{j}"
            stations[node] = G1Station(
                name=node, leg=k, s_m=float(s), xy_map=tuple(a + s * u),
                facing_map=tuple(v), travel_map=tuple(u), fan_offsets_m=fan,
                purpose="disambiguator" if s < 0.0 else "wall")

    goal_xy = tuple(np.asarray(edges[-1].b, float))
    disambig = tuple(n for n, st in stations.items() if st.purpose == "disambiguator")
    return SceneSpec(
        name=name, walls=tuple(placements), plan=plan, stations=stations,
        standoff_m=standoff, goal_map_xy=goal_xy, goal_radius_m=0.22,
        goal_edge=len(legs) - 1, n_route_decisions=len(legs) - 1,
        disambiguating_stations=disambig, legs=tuple(legs), fan_offsets_m=fan,
        **{k: v for k, v in kw.items() if k in SceneSpec.__dataclass_fields__},
    )


# ── the two scenes ───────────────────────────────────────────────────────────────


def two_turns() -> SceneSpec:
    """Unambiguous, THREE walls, TWO consecutive physical turns.

    Wall half-lengths are the smallest the asset offers (1.0, 1.1) precisely
    because a follow-on leg costs ``2*hl + standoff``: the episode length is set
    by the boxes, not by the route.
    """
    return _build("two_turns", (0, 4, 1), standoff=0.38, first_near_m=-0.55,
                  goal_leg_m=0.55)


def ambiguous_near_edge() -> SceneSpec:
    """Bimodal along-wall registration; the disambiguator is BEHIND the robot.

    delta = 0 is the truth.  Under delta = +0.55 every forward probe near the
    start reads identically (flat wall at the standoff) and the terminus sits
    0.55 m further on, so committing the turn on the wrong mode turns 0.55 m
    early -- into open floor, missing the next wall entirely.
    """
    scene = _build("ambiguous_near_edge", (0, 4), standoff=0.38, first_near_m=-0.70,
                   goal_leg_m=0.55, back_station_m=0.75,
                   prior_modes=((0.0, 0.5), (0.55, 0.5)),
                   prior_sigma_along_m=0.10, true_delta_m=0.0, max_backup_m=0.95)
    return scene


SCENES = {"two_turns": two_turns, "ambiguous_near_edge": ambiguous_near_edge}


def build_scene(name: str) -> SceneSpec:
    return SCENES[name]()


def detour_box() -> SceneSpec:
    """A route whose ONLY disambiguating evidence is an off-route box.

    Two hypotheses about where the robot stands along wall 0, 0.55 m apart.
    Wall 0 is long and unbroken over the whole approach, so **every on-route
    station reads flat wall at the standoff under both hypotheses** -- rank 1,
    zero discrimination, checked station by station in the tests.

    The stations deliberately EXCLUDE the terminus brackets that the other
    scenes get.  That is not a convenience: the fan reaches 0.06-0.30 m ahead,
    so a stance that straddles the corner is already within a fan-length of the
    point at which the robot must commit -- the evidence arrives AT the decision,
    not before it, which is the same limitation Batch 3A documented.  A stop that
    can only pay off after the commit is not a station on this route.

    The one thing that does discriminate is a free-standing box 1.00 m BEHIND
    the start, on the wall line.  Under the near hypothesis a press there
    contacts it; under the far one the same press finds open air, because the
    wall itself has not started yet.  Contact versus free: discrimination 1.0.
    Reaching it means strafing ~1.2 m AWAY from the goal and back.

    Guessing is punished physically: navigating by the wrong hypothesis commits
    the turn 0.55 m early, while wall 0 is still alongside, so the robot turns
    into the wall it was following.
    """
    standoff, hl0, hl1 = 0.38, HALF_LENGTHS[0], HALF_LENGTHS[4]
    face = standoff + THICKNESS / 2.0
    near0, delta, box_s = -0.70, 0.55, -1.00
    # The landmark PROTRUDES 0.08 m proud of the wall plane.  Coplanar with it,
    # the only evidence a miss carries is a 0.10 m penetration -- 0.6 nats, and
    # fifteen presses moved the hypothesis masses from 0.51 to 0.64 without
    # deciding anything.  Proud of the plane, a hit reads 0.30 m against the
    # wall's 0.38 m and a miss sweeps 0.15 m through where the box should be,
    # which saturates the free-ray term.  0.30 m is the floor: closer than that
    # and a parked wrist reaches it.
    box_face = standoff - 0.08
    term0 = near0 + 2.0 * hl0                                   # 1.30
    u, v = np.array([1.0, 0.0]), np.array([0.0, 1.0])

    walls = (
        WallPlacement(0, (near0 + hl0, face), tuple(u)),        # the long wall 0
        WallPlacement(4, ((term0 + standoff) - THICKNESS / 2.0 - hl1 + hl1,
                          face + hl1), tuple(v)),               # the corner wall
        # the landmark: a box turned across the wall line, so what the hand meets
        # is a 0.10 m-wide face at the same standoff -- the narrowest localized
        # feature the six-box asset can express
        WallPlacement(2, (box_s, box_face + HALF_LENGTHS[2]), tuple(v)),
    )
    walls = (walls[0],
             WallPlacement(4, (term0 - THICKNESS / 2.0, face + hl1), tuple(v)),
             walls[2])

    # EVERY body carries signature "surface".  A tactile contact enters the
    # belief as an anonymous "surface" touch, and the association model scores
    # it against the bodies SHARING its signature -- a body with any other
    # signature is invisible to it.  With the box marked "box", a physically
    # correct press on the landmark could not be explained by the landmark:
    # the correct particles were executed by their own contact (nearest
    # same-signature wall 0.3+ m away) and the belief resolved 0.47 m WRONG,
    # systematically, on 8 of 12 seeds.  The map may know the body is a box;
    # the touch channel cannot, so the map must not partition bodies into
    # signature classes touch cannot distinguish.
    bodies = tuple(w.polygon(n, s) for w, n, s in
                   ((walls[0], "wall_0", "surface"),
                    (walls[1], "wall_1", "surface"),
                    (walls[2], "landmark_box", "surface")))
    feats = (MapFeature(0, (near0, standoff), "wall_end", "wall0_near"),
             MapFeature(1, (term0, standoff), "wall_end", "wall0_far"),
             MapFeature(2, (box_s, box_face), "box_face", "landmark"))

    leg0 = {"origin": np.zeros(2), "u": u.copy(), "v": v.copy(),
            "near": near0, "terminus": term0, "index": 0}
    o1 = np.array([term0 + standoff, 0.0])
    leg1 = {"origin": o1, "u": v.copy(), "v": -u, "near": face + THICKNESS / 2.0,
            "terminus": face + THICKNESS / 2.0 + 2.0 * hl1, "index": 4}
    legs = (leg0, leg1)
    goal_leg = 0.55
    edges = (RouteEdge(0, (0.0, 0.0), tuple(o1), 0.55, (1,), "leg0"),
             RouteEdge(1, tuple(o1), tuple(o1 + goal_leg * leg1["u"]), 0.55, (), "leg1"))
    plan = Floorplan(bodies, feats, edges, route=(0, 1), name="detour_box")

    fan = (0.06, 0.12, 0.18, 0.24, 0.30)
    stations = {}
    # On-route stations stop BEFORE the terminus-bracketing window.  The fan
    # spans [0.06, 0.30] ahead, so any stance with s > term0 - 0.30 straddles the
    # corner and would discriminate -- which is precisely the evidence this scene
    # exists to withhold, and it arrives only at the commit point anyway.
    last_on_route = term0 - max(fan) - 0.05
    for j, s in enumerate([round(float(x), 3) for x in
                           np.arange(0.0, last_on_route + 1e-6, 0.30)]):
        stations[f"L0s{j}"] = G1Station(f"L0s{j}", 0, s, tuple(leg0["origin"] + s * u),
                                        tuple(v), tuple(u), fan, purpose="wall")
    stations["BOX"] = G1Station("BOX", 0, box_s - 0.18,
                                tuple(leg0["origin"] + (box_s - 0.18) * u),
                                tuple(v), tuple(u), fan, purpose="disambiguator",
                                face_offset_m=standoff - box_face)
    for j, s in enumerate([0.0, 0.35]):
        stations[f"L1s{j}"] = G1Station(f"L1s{j}", 1, s, tuple(o1 + s * leg1["u"]),
                                        tuple(leg1["v"]), tuple(leg1["u"]), fan, purpose="wall")

    return SceneSpec(
        name="detour_box", walls=walls, plan=plan, stations=stations,
        standoff_m=standoff, goal_map_xy=tuple(o1 + goal_leg * leg1["u"]),
        goal_radius_m=0.22, goal_edge=1,
        prior_modes=((0.0, 0.5), (delta, 0.5)), prior_sigma_along_m=0.10,
        prior_sigma_lateral_m=0.06, true_delta_m=0.0, max_backup_m=1.60,
        n_route_decisions=1, disambiguating_stations=("BOX",),
        legs=legs, fan_offsets_m=fan, withholds_terminus_evidence=True)


SCENES["detour_box"] = detour_box

# ── the whole-house scenes (Batch 3A.2 demo pack) ────────────────────────────────


def _with_landmark(scene: SceneSpec, placement: WallPlacement, body_name: str,
                   feature: tuple, extra_stations: dict,
                   mark_disambiguating: bool = False) -> SceneSpec:
    """Return a copy of ``scene`` with one landmark body, its map feature, and
    its hand-built stations added.  The landmark is a real wall-map box (it is
    placed physically), a real map body (the belief and the clearance checks
    see it), and its stations carry ``face_offset_m`` so approaches keep the
    wrist envelope."""
    from dataclasses import replace as _replace
    plan = scene.plan
    bodies = plan.bodies + (placement.polygon(body_name, "surface"),)
    fx, fname, fsig = feature
    feats = plan.features + (MapFeature(len(plan.features),
                                        tuple(np.asarray(fx, float)), fsig, fname),)
    new_plan = Floorplan(bodies, feats, plan.edges, plan.route, plan.name)
    stations = dict(scene.stations)
    stations.update(extra_stations)
    disambig = scene.disambiguating_stations + tuple(
        n for n, st in extra_stations.items() if st.purpose == "disambiguator")
    return _replace(scene, walls=scene.walls + (placement,), plan=new_plan,
                    stations=stations,
                    disambiguating_stations=disambig if mark_disambiguating
                    else scene.disambiguating_stations)


def nominal_multiroom() -> SceneSpec:
    """A. Four corridor sections, three physical turns, 10.4 m of route, a
    mostly unimodal prior, and the ordinary terminus-bracketing anchors on
    every leg.  The long-horizon stability scene: no ambiguity to resolve,
    just drift to manage across four walls of the six-box asset
    (half-lengths 1.6, 1.4, 1.5, 1.2)."""
    return _build("nominal_multiroom", (3, 2, 5, 1), standoff=0.38,
                  first_near_m=-0.55, goal_leg_m=0.8)


def ambiguous_landmark_house() -> SceneSpec:
    """B. Four corridor sections, three branch decisions, and a bimodal
    along-registration (delta in {0, +0.55} m) that every on-route probe
    before the FIRST branch cannot resolve: leg 0's wall is long and
    unbroken, its terminus brackets are withheld, so every on-route fan
    reads flat wall at the standoff under both hypotheses.

    The only discriminating evidence is a 2.0 m LANDMARK SLAB (box 0 laid
    parallel to wall 0) standing in OPEN FLOOR behind the start: its west
    edge has nothing behind it, so a fan there reads slab-contact under one
    registration and full-reach free under the other -- the strongest channel
    the hand has, and 2 m of face makes it far more observable than the
    0.10 m sliver 3A.2 measured at the edge of physical observability.
    Reaching it means walking ~1-3 m AWAY from the goal and back."""
    # wall 0 extends 1.3 m BEHIND the start so that no on-route fan can see
    # its near edge under either registration (fans reach 0.30 ahead and the
    # hypotheses sit 0.55 apart), and the on-route grid is trimmed the same
    # margin short of the terminus -- both were measured leaking discrimination
    # (1.0 at the first and last grid stances) before this.
    # wall assignment (3, 1, 2, 4): the chain spirals one turn direction, and
    # with a 1.5 m fourth wall its slab was measured crossing the detour
    # corridor at y = -0.42 -- the short 1.1 m wall keeps leg 3 above y = 0.58
    scene = _build("ambiguous_landmark_house", (3, 1, 2, 4), standoff=0.38,
                   first_near_m=-1.30, goal_leg_m=0.8,
                   no_bracket_legs=(0,),
                   prior_modes=((0.0, 0.5), (0.55, 0.5)),
                   prior_sigma_along_m=0.10, true_delta_m=0.0,
                   max_backup_m=3.4, withholds_terminus_evidence=True)
    from dataclasses import replace as _replace
    sep, fan_max = 0.55, 0.30
    cutoff = float(scene.legs[0]["terminus"]) - sep - fan_max - 0.05
    scene = _replace(scene, stations={
        n: st for n, st in scene.stations.items()
        if not (st.leg == 0 and st.purpose == "wall" and st.s_m > cutoff)})
    leg0 = scene.legs[0]
    u, v = np.asarray(leg0["u"], float), np.asarray(leg0["v"], float)
    face_y = 0.30                       # 0.08 proud of the 0.38 wall plane
    west, east = -2.6, -0.6             # a 2.0 m slab: 2 * HALF_LENGTHS[0]
    centre = np.array([0.5 * (west + east), face_y + THICKNESS / 2.0])
    slab = WallPlacement(0, tuple(centre), tuple(u))
    fan = scene.fan_offsets_m
    off = scene.standoff_m - face_y
    extra = {}
    for tag, s in (("W", west - 0.18), ("M", 0.5 * (west + east)),
                   ("E", east - 0.18)):
        name = f"SLAB{tag}"
        extra[name] = G1Station(
            name=name, leg=0, s_m=float(s), xy_map=tuple(s * u),
            facing_map=tuple(v), travel_map=tuple(u), fan_offsets_m=fan,
            purpose="disambiguator", face_offset_m=off)
    return _with_landmark(
        scene, slab, "landmark_slab",
        ((west, face_y), "slab_west_edge", "wall_end"),
        extra, mark_disambiguating=True)


def multi_decision_house() -> SceneSpec:
    """C. Five corridor sections, four physical turns, 12.2 m of route, and
    TWO separate tactile decisions with different evidence types:

    1. the prior's along-spread (0.14 m) exceeds the 0.12 m branch gate, so
       the FIRST turn cannot be committed until the leg-0 terminus brackets
       are pressed -- wall-end (contact/free-past-the-end) evidence;
    2. leg 2's own terminus brackets are withheld, and after ~6 m and two
       turns the accumulated drift again exceeds the gate -- the only
       along-anchor within reach is a 2.0 m PROUD BUMP (box 0, 0.08 m proud
       of wall 2's face, the detour_box protrusion the stance machinery is
       validated on) whose edges step the contact range mid-leg.

    Each decision is measurable in the directive log: PROBE directives, the
    gate opening, then the TURN that the evidence unblocked."""
    # wall assignment (1, 4, 5, 3, 2): the five-leg spiral wraps leg 4 back
    # under leg 0, and with term3 - term1 = 0.60 its wall passed 0.12 m from
    # the START; the 1.1/1.6 pairing pushes leg 4 down to y = -0.57, clearing
    # the start by 0.52 m and the leg-0 lane by 0.34 m
    # leg 2 keeps its terminus brackets (the frozen bracket invariant demands
    # them); the BUMP sits mid-leg, over a metre CLOSER to where the drift
    # crosses the commit gate, so the cost model prefers it -- the second
    # tactile decision uses bump-edge range steps, verified behaviourally in
    # the whole-house tests rather than by deleting the brackets
    from dataclasses import replace as _replace
    scene = _build("multi_decision_house", (1, 4, 5, 3, 2), standoff=0.38,
                   first_near_m=-0.55, goal_leg_m=0.8,
                   prior_sigma_along_m=0.16)
    # the prior's spread must sit CLEARLY over the 0.12 commit gate, or the
    # first tactile decision becomes optional at small particle counts --
    # decision 1 is the point of the scene, not a coin flip
    leg2 = scene.legs[2]
    a2 = np.asarray(leg2["origin"], float)
    u2, v2 = np.asarray(leg2["u"], float), np.asarray(leg2["v"], float)
    near2, term2 = float(leg2["near"]), float(leg2["terminus"])
    face_y = scene.standoff_m - 0.08
    hl = HALF_LENGTHS[0]
    mid_s = 0.5 * (near2 + term2)
    west, east = mid_s - hl, mid_s + hl
    centre = a2 + mid_s * u2 + (face_y + THICKNESS / 2.0) * v2
    bump = WallPlacement(0, tuple(centre), tuple(u2))
    fan = scene.fan_offsets_m
    off = scene.standoff_m - face_y
    # every leg-2 grid stance whose fan overlaps the bump span must carry the
    # bump's face offset too, or its approach recreates the exact stance the
    # 42.5 N wrist clip came from (bias computed against the wall plane, face
    # 0.08 m prouder)
    from dataclasses import replace as _st_replace
    scene = _replace(scene, stations={
        n: (_st_replace(st, face_offset_m=off)
            if (st.leg == 2 and st.purpose == "wall"
                and st.s_m + max(fan) >= west - 0.05
                and st.s_m + min(fan) <= east + 0.05)
            else st)
        for n, st in scene.stations.items()})
    extra = {}
    for tag, s in (("W", west - 0.18), ("E", east - 0.18)):
        name = f"BUMP{tag}"
        extra[name] = G1Station(
            name=name, leg=2, s_m=float(s), xy_map=tuple(a2 + s * u2),
            facing_map=tuple(v2), travel_map=tuple(u2), fan_offsets_m=fan,
            purpose="wall", face_offset_m=off)
    return _with_landmark(
        scene, bump, "leg2_bump",
        (tuple(a2 + west * u2 + face_y * v2), "bump_west_edge", "wall_end"),
        extra)


SCENES["nominal_multiroom"] = nominal_multiroom
SCENES["ambiguous_landmark_house"] = ambiguous_landmark_house
SCENES["multi_decision_house"] = multi_decision_house


# ── the contact-rich demo scene (TOUCH_STEP_FOLLOW) ──────────────────────────────


def contact_rich_gap() -> SceneSpec:
    """D. Contact-rich demo: touch-maintained travel plus an on-route wall-gap
    landmark that flips a wrong-favoured prior.

    Three corridor sections, two physical turns, ~7.3 m of route.

    LEG 0'S WALL IS THE STANDARD PLANE.  A first cut placed it 0.08 m proud
    (face at 0.30 m) to buy free-certification margin -- and the very first
    physical walk snagged the parked left wrist at 157 N at t = 1.5 s: the
    station approach's stage-1 leg walks the START's own lateral line (y = 0),
    which a proud wall puts 0.30 m from the face, inside spawn-wobble range
    of the 0.26 m tucked-wrist envelope.  Every validated scene gives that
    line 0.43 m.  The proud margin is also unnecessary: the press-validity
    diagnosis on the whole-house episode showed correct stances at a realized
    0.32-0.38 m standoff produce valid contacts AND valid swept-frees on a
    standard wall (station L2s13, 8/8) -- the invalid fans came from stances
    displaced to 0.47-0.51 m by the clearance-shift path, which this scene's
    early-resolving, wall-parallel hypothesis layout does not trigger.

    THE GAP IS THE LANDMARK.  Between the boxes: a 0.40 m gap, edges at
    s = 0.70 and s = 1.10, both ``wall_end`` features -- the contact-vs-free
    channel, discrimination 1.0 under the +-0.08 m jitter ball, sitting ON
    the route ~2.2 m before the first corner.  The gap width is deliberately
    NOT the 0.55 m hypothesis separation, so an edge observed under one
    registration cannot alias to the other edge under the rival.

    THE PRIOR FAVOURS THE WRONG MODE.  Bimodal along-registration
    {0.0: 0.45, +0.55: 0.55}, truth at 0.0.  A no-touch commit tracks the
    heavier (wrong) mode and turns 0.55 m early -- into the wall it was
    following, since the followed box still runs beside it there.  The prior
    alone cannot choose the branch; the gap evidence flips the tracked mode,
    which moves the believed corner by 0.55 m and changes the turn the
    planner commits.  Leg 0 keeps its terminus brackets: they sharpen sigma
    near the corner but cannot pick the mode from under the wrong tracked
    registration without the gap flip happening first (checked in tests:
    their discrimination exists only inside the bracket window, which the
    commit-gated approach reaches after the gap stations).
    """
    standoff = 0.38
    face = standoff + THICKNESS / 2.0                 # 0.43: standard wall plane
    fan = (0.06, 0.12, 0.18, 0.24, 0.30)
    u0, v0 = np.array([1.0, 0.0]), np.array([0.0, 1.0])

    # leg 0: wall pair A [-2.5, 0.7] + 0.40 m gap + B [1.1, 3.3], standard plane
    face0_y = standoff                                 # face at 0.38: y = 0.38
    off0 = 0.0
    hlA, hlB = HALF_LENGTHS[3], HALF_LENGTHS[4]        # 1.6, 1.1
    a_w = -2.5
    a_e = a_w + 2.0 * hlA                              # 0.70
    b_w = a_e + 0.40                                   # 1.10; 0.40 m gap A->B
    b_e = b_w + 2.0 * hlB                              # 3.30
    term0 = b_e
    wallA = WallPlacement(3, (a_w + hlA, face0_y + THICKNESS / 2.0), tuple(u0))
    wallB = WallPlacement(4, (b_w + hlB, face0_y + THICKNESS / 2.0), tuple(u0))

    # legs 1-2: standard chain geometry, standard 0.38 m walls
    o1 = np.array([term0 + standoff, 0.0])             # (3.68, 0)
    u1, v1 = v0.copy(), -u0                            # travel +y, face -x
    near1 = face + THICKNESS / 2.0                     # 0.48
    hlC = HALF_LENGTHS[0]                              # 1.0
    term1 = near1 + 2.0 * hlC                          # 2.48
    wallC = WallPlacement(0, tuple(o1 + (near1 + hlC) * u1 + face * v1), tuple(u1))
    o2 = o1 + (term1 + standoff) * u1                  # (3.68, 2.86)
    u2, v2 = v1.copy(), -u1                            # travel -x, face -y
    near2 = face + THICKNESS / 2.0
    hlD = HALF_LENGTHS[1]                              # 1.2
    wallD = WallPlacement(1, tuple(o2 + (near2 + hlD) * u2 + face * v2), tuple(u2))
    goal_leg = 0.8
    goal_xy = o2 + goal_leg * u2                       # (2.88, 2.86)

    walls = (wallA, wallB, wallC, wallD)
    bodies = (wallA.polygon("wall0_a", "surface"), wallB.polygon("wall0_b", "surface"),
              wallC.polygon("wall_1", "surface"), wallD.polygon("wall_2", "surface"))
    feats = tuple(MapFeature(i, xy, "wall_end", nm) for i, (xy, nm) in enumerate((
        ((a_w, face0_y), "wall0a_near"), ((a_e, face0_y), "gap_west"),
        ((b_w, face0_y), "gap_east"), ((b_e, face0_y), "wall0b_far"),
        (tuple(o1 + near1 * u1 + standoff * v1), "wall1_near"),
        (tuple(o1 + term1 * u1 + standoff * v1), "wall1_far"),
    )))

    legs = ({"origin": np.zeros(2), "u": u0.copy(), "v": v0.copy(),
             "near": a_w, "terminus": term0, "index": 3},
            {"origin": o1.copy(), "u": u1.copy(), "v": v1.copy(),
             "near": near1, "terminus": term1, "index": 0},
            {"origin": o2.copy(), "u": u2.copy(), "v": v2.copy(),
             "near": near2, "terminus": near2 + 2.0 * hlD, "index": 1})
    edges = (RouteEdge(0, (0.0, 0.0), tuple(o1), 0.55, (1,), "leg0"),
             RouteEdge(1, tuple(o1), tuple(o2), 0.55, (2,), "leg1"),
             RouteEdge(2, tuple(o2), tuple(goal_xy), 0.55, (), "leg2"))
    plan = Floorplan(bodies, feats, edges, route=(0, 1, 2), name="contact_rich_gap")

    stations: dict[str, G1Station] = {}
    mid = 0.5 * (min(fan) + max(fan))                  # 0.18
    for k, (leg, span, foff) in enumerate((
            (legs[0], float(np.linalg.norm(o1)), off0),
            (legs[1], float(np.linalg.norm(o2 - o1)), 0.0),
            (legs[2], goal_leg, 0.0))):
        a, u, v = leg["origin"], leg["u"], leg["v"]
        offsets = [round(float(s), 3) for s in np.arange(0.0, span + 1e-6, 0.35)]
        if k < 2:                                       # brackets on the turn legs
            term = float(leg["terminus"])
            for back in (mid, mid + 0.10, mid - 0.10):
                s = round(term - back, 3)
                if -0.05 <= s <= span + 0.35:
                    offsets.append(s)
        for j, s in enumerate(sorted(dict.fromkeys(offsets))):
            node = f"L{k}s{j}"
            stations[node] = G1Station(
                name=node, leg=k, s_m=float(s), xy_map=tuple(a + s * u),
                facing_map=tuple(v), travel_map=tuple(u), fan_offsets_m=fan,
                purpose="wall", face_offset_m=foff)
    for tag, edge_s in (("W", a_e), ("E", b_w)):
        name = f"GAP{tag}"
        stations[name] = G1Station(
            name=name, leg=0, s_m=float(edge_s - mid),
            xy_map=tuple(legs[0]["origin"] + (edge_s - mid) * u0),
            facing_map=tuple(v0), travel_map=tuple(u0), fan_offsets_m=fan,
            purpose="disambiguator", face_offset_m=off0)

    # THE GOAL LEG NEEDS ITS OWN ALONG-ANCHOR.  From leg-2 stances, wall_1's
    # end cap is COPLANAR with wall_2's face (the chain corner puts both at
    # the same 0.38 m range), so the wall_2 near edge is masked: the visible
    # surface is one flat wall from the cap's far flank (leg-2 s = 0.38)
    # all the way past the goal, and the 0.35 m grid steps right over that
    # flank -- L2s0 reads all-free, L2s1 all-contact, nothing mixed.
    # Measured consequence (seed-0 attempt 2): 0.24 m of post-turn along-goal
    # drift, an ARRIVE spin at the believed goal, and a forced anchor that
    # found no station.  These two stances bracket the cap flank so their
    # fans read free-then-contact: the arrival anchor.
    cap_flank_x = float(o1[0]) - face + THICKNESS / 2.0
    cap_s = float(o2[0]) - cap_flank_x             # 3.68 - 3.30 = 0.38
    for tag, s in (("0", cap_s - mid), ("1", cap_s - mid + 0.10)):
        name = f"L2A{tag}"
        stations[name] = G1Station(
            name=name, leg=2, s_m=float(s),
            xy_map=tuple(legs[2]["origin"] + s * u2),
            facing_map=tuple(v2), travel_map=tuple(u2), fan_offsets_m=fan,
            purpose="wall", face_offset_m=0.0)

    return SceneSpec(
        name="contact_rich_gap", walls=walls, plan=plan, stations=stations,
        standoff_m=standoff, goal_map_xy=tuple(goal_xy), goal_radius_m=0.22,
        goal_edge=2, n_route_decisions=2,
        prior_modes=((0.0, 0.45), (0.55, 0.55)),
        prior_sigma_along_m=0.10, prior_sigma_lateral_m=0.06, true_delta_m=0.0,
        max_backup_m=0.30, disambiguating_stations=("GAPW", "GAPE"),
        legs=legs, fan_offsets_m=fan)


SCENES["contact_rich_gap"] = contact_rich_gap
