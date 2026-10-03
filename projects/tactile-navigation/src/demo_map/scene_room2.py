"""Room2 (the 8_11 scan): scene + routes, both directions, corner-ATE demo.

THE ROOM.  An L of two walls meeting at exactly (0, 0) in the scan frame: the
west wall's inner face on x = 0 (with two recessed panels behind it) and the
north wall's inner face on y = 0.  One shadowed board on the north wall and a
small post at (3.57, -0.25) are the only clutter.

THE DEMO.  Start and goal both sit ~0.55 m off a wall (owner spec).  The route
follows one wall to the corner, rounds it, and follows the other wall out --
and the CORNER IS THE ONLY LANDMARK: the ported ATE decides where on its two
faces to press, how many times, and when it has learned enough.  Flat-wall
rhythm presses remain rule-based (the "feeling forward" cadence).  Both
directions use the same geometry mirrored.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np

from execution.wall_lane import WallSegment
from global_nav.floorplan import Floorplan, MapFeature, RouteEdge
from global_nav.geometry import ConvexPolygon
from lane_graph.maps_v3 import followable
from lane_graph.scene3 import V3Scene, _ngon

from .executor import (FOLLOW_STANDOFF, STATION_END_MARGIN_M,
                       corner_turn_path, tag_gaits,
                       _stations_for)
from .mapv3 import demo_map_v3
from .geometry import CylObs
from .scan_map import build_scan_map
from .scene_demo import translated_scan_map


def _walls_hugged(poly, spec_walls, min_run: int = 3) -> list:
    """Which walls a followed edge actually runs along, in order.

    Read from the EDGE'S OWN SAMPLES against the wall faces, not from the sign
    of the edge's dominant axis.  The two are the same for a straight run and
    differ exactly where it matters: a corner-rounding edge is one
    ``TOUCH_FOLLOW`` that leaves one wall and arrives at the next, and its
    dominant axis reports whichever end happens to be longer.

    Measured (2026-08-31): with the west wall reconstructed as three
    overlapping boards the corner edge ran (-0.25, 2.47) -> (1.45, 3.19) and
    the dominant axis called it "north"; with the boards merged into the one
    continuous wall the owner asked for, the SAME corner runs (-0.03, 1.71) ->
    (0.64, 3.41) and the dominant axis calls it "west" -- so the route report
    flipped while the route did not.  Sample-wise, that edge hugs the west
    face for 25 of 35 samples and closes to 0.34 m of the north face for the
    last 10: west then north, which is what the route does.

    A wall is only credited with a run of at least ``min_run`` samples, so a
    single sample equidistant between two faces cannot invent a leg."""
    P = np.asarray(poly, float).reshape(-1, 2)
    faces = []
    for w in spec_walls:
        a, b = np.asarray(w.p0, float), np.asarray(w.p1, float)
        u = b - a
        faces.append((a, u, "west" if abs(u[1]) >= abs(u[0]) else "north"))
    names = []
    for p in P:
        best = None
        for a, u, name in faces:
            t = float(np.clip((p - a) @ u / max(u @ u, 1e-12), 0.0, 1.0))
            d = float(np.linalg.norm(p - (a + t * u)))
            if best is None or d < best[0]:
                best = (d, name)
        names.append(best[1])
    out, run = [], []
    for name in names + [None]:
        if run and name != run[0]:
            if len(run) >= min_run:
                out.append(run[0])
            run = []
        if name is not None:
            run.append(name)
    return out


def plan_room2_route(scene) -> dict:
    """Run the REAL path planner on the room and report its route.

    The owner's rule: the corner is a landmark the PLANNER routes through,
    not a scripted waypoint.  In an L-room every start->goal route passes the
    corner topologically, so what this buys is honesty, not drama: the
    executor's compiled steps must MATCH the planner's wall order and corner
    passage, and `plan_steps_room2` refuses to compile a route the planner
    did not choose."""
    from lane_graph.graph3 import Belief3, astar3, build_graph3

    g = build_graph3(scene)
    src = next(n.nid for n in g.nodes if n.kind == "start")
    dst = next(n.nid for n in g.nodes if n.kind == "goal")
    eids, cost = astar3(g, src, dst,
                        Belief3(sigma=scene.prior_sigma_along_m, minor=0.0))
    if eids is None:
        return {"ok": False, "reason": "no_route"}
    corner_sc = np.asarray(scene.plan.features[0].xy, float)
    walls, d_corner, nid = [], np.inf, src
    for eid in eids:
        e = g.edges[eid]
        poly = (e.poly if e.poly is not None
                else np.array([g.nodes[e.a].xy, g.nodes[e.b].xy]))
        if e.b == nid:
            poly = poly[::-1]
        nid = e.b if e.a == nid else e.a
        d_corner = min(d_corner, float(np.min(np.linalg.norm(
            np.asarray(poly, float) - corner_sc, axis=1))))
        if e.kind == "TOUCH_FOLLOW":
            for wall in _walls_hugged(poly, scene.spec.walls):
                if not walls or walls[-1] != wall:
                    walls.append(wall)
    # the followable lanes run at 0.56 m and round the corner at lane
    # offset, so the planned polyline's closest approach to the corner POINT
    # is ~0.8-1.2 m by construction; within 1.4 m = the route rounds this
    # corner rather than crossing the room
    return {"ok": True, "walls": walls, "corner_dist_m": round(d_corner, 3),
            "n_edges": len(eids), "cost": round(float(cost), 2),
            "passes_corner": d_corner <= 1.4}

ROOM2_WALLS = Path("docs/demo_maps/room2_scan_walls.json")

CORNER = np.array([0.0, 0.0])          # scan frame; exact by construction
CORNER_EXIT_REACH_M = 1.00
"""How far along the OUTGOING face a corner press may sit.

Asymmetric with the approach face on purpose: the turn drops the base at arc
0.42 on this segment, and the never-walk-back floor then starts candidates at
`s_now + 0.25 = 0.67` -- past a 0.60 limit, so the ATE reported
`no_candidates` and the outgoing-face gate failed with the corner press on
the approach face already banked.  1.00 leaves real choice after the turn;
the proximity tie-break still pulls the pick to the low end, so presses land
near the vertex in practice rather than at the limit."""
CORNER_STATION_ROOM_M = STATION_END_MARGIN_M
"""Room the station machinery needs past the last corner candidate.

Tightening the corner pushed the ATE's range to `len - 0.15`, and the
proximity tie-break then always picked the LAST candidate -- the one with no
room left.  `_press_station` gives a chunk `s_to = station + 0.55` clipped to
the segment, and refuses a station past `length - 0.25`, so at 0.15 m of
run-out the settle gate could never open: measured `unsettled` on the corner
press in five consecutive runs, which failed the turn gate every time.  The
hand lands 0.18 m further along than the station, so a press at
`len - 0.32` still contacts ~0.14 m from the vertex.

0.32 was sized for a station that meant SETTLE-then-press.  Feeling needs no
settle, so the only run-out still required is the one the station machinery
itself enforces: it DROPS a station past `length - STATION_END_MARGIN_M`.
Deriving this from that rule instead of guessing removes a duplicated magic
number -- guessing 0.10 put the ATE's pick one centimetre inside the refusal
boundary and the corner press was silently dropped.  The
remaining distance is the platform's own floor: the wall segments stop 0.58 m
(approach face) and 0.30 m (outgoing face) short of the vertex because a base
facing one wall while sidling toward the other cannot occupy the corner, and
the hand only reaches 0.18 m past the base."""
CORNER_REACH_M = 0.60
"""How much wall on each side counts as THE CORNER.

Was 1.4 m / 1.2 m, which handed the ATE over a metre of plain face per side:
its presses landed that far from the vertex and the demo read as ordinary
wall touching rather than as understanding a landmark (owner).  0.60 m keeps
every corner press within two station spacings of the vertex while still
leaving the ATE two or three candidates per face to choose between."""
N_WEST = np.array([1.0, 0.0])          # west wall inward normal (east)
N_NORTH = np.array([0.0, -1.0])        # north wall inward normal (south)
START_FWD = (0.55, -3.75)              # 0.55 off the west wall, south end
GOAL_FWD = (3.05, -0.55)               # 0.55 off the north wall, clear of the
                                       # post at (3.57, -0.25)

# OWNER-PLACED GOAL MARKER, EMBEDDED IN THE WALL (owner, 2026-08-18).
#
# "You are here" cannot be dead-reckoned: the axis ALONG a wall is exactly
# what a flat face cannot pin, so arrival used to be declared on a belief
# nothing had corrected since the corner.  A cylinder fixes that because its
# normal TURNS with position -- one touch constrains both axes.
#
# Half-embedded in the wall rather than free-standing, which is strictly
# better here and not merely more convenient:
#   * it sits ON the wall-following path, so it is reached by an ordinary
#     station -- no turn, no separate apex lane, no goal-side detour;
#   * it escapes the concave-corner clearance problem entirely (a base facing
#     one wall while sidling toward another cannot occupy a vertex, which is
#     why corner presses bottom out ~0.40 m away);
#   * `bulge` -- "a half-embedded cylinder standing proud of the face" -- is
#     already a first-class primitive in this codebase, injected into MuJoCo
#     and understood by the V3 planner.
# Placed directly abeam the goal, so feeling it pins the very coordinate that
# says "stop here".
#
# SIZE IS A MEASUREMENT, not a preference.  At radius 0.09 the marker spanned
# 0.18 m of wall while the belief error ran 0.09-0.30 m, so the hand kept
# landing on the flat wall beside it and grazing at 2.9-3.9 N -- seed 1
# missed six tries running and correctly refused to declare arrival.  0.16 m
# spans 0.32 m, wider than the belief error, and is what a real pillar or
# door frame actually measures.  The apex then stands 0.16 m off the wall:
# the lane sits at 0.49 and the arm reaches 0.42, so it is comfortably in
# reach and still nowhere near the 0.32 m parked-hand sweep.
GOAL_PILLAR_XY = (3.05, 0.0)           # forward goal: embedded in the NORTH wall
GOAL_PILLAR_XY_REV = (0.0, -3.75)      # reverse goal: embedded in the WEST wall,
                                       # abeam START_FWD -- the reverse leg ends
                                       # there, and an unmarked goal cannot be
                                       # CONFIRMED by touch (the whole point of
                                       # the marker)
GOAL_PILLAR_R = 0.16


def build_room2_map(reverse: bool = False):
    # flatten_steps=True since 2026-08-31, at the owner's instruction: the
    # west wall was THREE overlapping boards -- the 4.29 m outer plane plus
    # recesses 64 mm and 111 mm behind it, spanning the same stretch -- and
    # the owner asked for one continuous wall before the repeatability batch.
    # This is a real change to the room the deploy-parity numbers were earned
    # in: the merged plane sits at the length-weighted depth of the three,
    # 36 mm behind the outer face, so the west wall moves that far and the
    # lane with it.
    m = build_scan_map(ROOM2_WALLS, flatten_steps=True)
    start, goal = (GOAL_FWD, START_FWD) if reverse else (START_FWD, GOAL_FWD)
    pillars = list(getattr(m, "pillars", ())) + [
        CylObs(pid="goal_post",
               center=(GOAL_PILLAR_XY_REV if reverse else GOAL_PILLAR_XY),
               radius=GOAL_PILLAR_R,
               z0=0.0, z1=1.2, source="owner-placed goal marker",
               note="point landmark: one touch pins BOTH axes")]
    return dataclasses.replace(
        m, pillars=pillars,
        start=dataclasses.replace(m.start, center=tuple(start),
                                  note="owner-placed, 0.55 m off the wall"),
        goal=dataclasses.replace(m.goal, center=tuple(goal),
                                 note="owner-placed, 0.55 m off the wall"))


def build_room2_scene(reverse: bool = False) -> V3Scene:
    """Same contract as ``build_demo_scene``: translated so start = (0, 0)."""
    mt = translated_scan_map(build_room2_map(reverse))
    spec = demo_map_v3(mt)
    bodies = [ConvexPolygon(w.poly(), f"wall_{i}", "surface")
              for i, w in enumerate(spec.walls)]
    bodies += [_ngon(b.center, b.radius, name=f"bulge_{i}")
               for i, b in enumerate(spec.bulges)]
    bodies += [ConvexPolygon(p.poly(), f"prot_{i}", "surface")
               for i, p in enumerate(spec.prots)]
    feats = (MapFeature(0, _tr(CORNER, reverse), "landmark", "corner"),)
    xs = [v[0] for b in bodies for v in np.asarray(b.verts)]
    ys = [v[1] for b in bodies for v in np.asarray(b.verts)]
    lo = (min(xs) - 1.0, min(ys) - 1.0)
    hi = (max(xs) + 1.0, max(ys) + 1.0)
    edges = (RouteEdge(0, lo, (hi[0], lo[1]), half_width_m=(hi[1] - lo[1]),
                       successors=(), name="freespace"),)
    plan = Floorplan(tuple(bodies), feats, edges,
                     route=(0,), name=f"room2_{'rev' if reverse else 'fwd'}")
    return V3Scene(
        name=f"room2_{'rev' if reverse else 'fwd'}", spec=spec, plan=plan,
        lanes=followable(spec), start_map_xy=(0.0, 0.0),
        goal_map_xy=tuple(spec.goal), goal_radius_m=spec.goal_r,
        standoff_m=FOLLOW_STANDOFF,
        prior_components=(((0.0, 0.0), np.pi / 2.0, 0, 1.0),),
        prior_sigma_along_m=0.10, prior_sigma_lateral_m=0.06)


def _tr(p_scan, reverse: bool) -> tuple:
    """Scan frame -> scene frame (start at the origin)."""
    s = np.asarray(START_FWD if not reverse else GOAL_FWD, float)
    q = np.asarray(p_scan, float) - s
    return (float(q[0]), float(q[1]))


def room2_scan_offset(reverse: bool) -> np.ndarray:
    """Scene -> scan frame shift (what plots add back)."""
    return np.asarray(START_FWD if not reverse else GOAL_FWD, float)


def _merge_stations(cadence, guaranteed, spacing_m: float) -> list:
    """The cadence, plus one guaranteed station.  DEPLOYED BEHAVIOUR, VERBATIM.

    The home stretch may never go silent, so a station is forced at
    `s_hi + 0.95` when the ATE range would otherwise leave the run to the goal
    untouched.

    It can land beside a cadence station: room2's north:east leg gets
    `1.4 1.6 1.8 1.95 2.0 2.2 2.4`, so two presses sit 0.05 m apart, inside
    the sensor's own correlation length.  I deduped that, and then found the
    same 1.95 in `locomanip_deploy`'s shipped mission -- this is the route
    that was deployed, so the dedupe is reverted and the mission conformance
    check (`scripts/check_deploy_mission.py`) pins the whole station list to
    it.  Change it there first if it should change at all.
    """
    out = sorted(float(s) for s in cadence)
    if guaranteed is None:
        return out
    return sorted(set(out + [float(guaranteed)]))


def plan_steps_room2(cfg, reverse: bool = False, scene=None) -> list:
    """The planner's route, compiled to executor steps.

    The wall order and corner passage come from ``plan_room2_route``; this
    function only turns that choice into executable geometry, and REFUSES to
    compile if the planner chose differently (the compile may not override
    the planner -- owner directive)."""
    scene = scene or build_room2_scene(reverse)
    planned = plan_room2_route(scene)
    want = ["north", "west"] if reverse else ["west", "north"]
    if not (planned.get("ok") and planned.get("passes_corner")
            and planned.get("walls") == want):
        raise RuntimeError(f"planner route disagrees with the compile: "
                           f"{planned} vs walls {want}")
    lane = cfg.standoff_m

    if cfg.touch_style in ("lateral_stop", "turn_press_exact",
                           "crab_press_exact"):
        if reverse:
            raise ValueError("dense lateral room2 showcase is forward-only")
        corner_sc = np.asarray(_tr(CORNER, False), float)
        pre = WallSegment(
            p0=np.asarray(_tr((0.0, -4.05), False)),
            tangent=np.array([0.0, 1.0]), normal=N_WEST.copy(),
            # Ten centimetres of virtual run-out lets the arrival watchdog
            # reach the 4.05 m endpoint station before declaring the segment
            # complete; the physical wall itself still ends at the corner.
            length=4.15, name="west:north:dense")
        post = WallSegment(
            p0=np.asarray(_tr((0.0, 0.0), False)),
            tangent=np.array([1.0, 0.0]), normal=N_NORTH.copy(),
            length=3.25, name="north:east:dense")
        if cfg.touch_style == "crab_press_exact":
            # With 0.20 m counter-veer the measured contact lands on the
            # scheduled mark in the calibration probe.  Deliver the
            # high-value sequence first: five 0.20 m stations approach the
            # vertex closely enough for the hand footprint to cover it, then
            # plan six more stations on the outgoing face.
            pre_stations = [round(3.05 + 0.20 * k, 2) for k in range(5)]
            post_stations = [round(0.90 + 0.20 * k, 2) for k in range(6)]
        else:
            pre_stations = ([round(0.91 + 0.20 * k, 2) for k in range(15)]
                            + [3.90])
            post_stations = [round(0.50 + 0.20 * k, 2) for k in range(13)]
        return [
            {"kind": "goto", "p": np.asarray(pre.lane_point(
                2.70 if cfg.touch_style == "crab_press_exact" else 0.30,
                lane)),
             "yaw": pre.yaw, "label": "approach", "tol": 0.03},
            {"kind": "follow", "seg": pre, "stations": pre_stations,
             "label": "west:north:dense", "style": cfg.touch_style,
             "standoff": lane, "run_out": False},
            {"kind": "corner_gate", "corner": corner_sc,
             "tol": 0.065 if cfg.touch_style == "crab_press_exact" else 0.10},
            {"kind": "goto", "p": np.asarray(_tr((0.38, -0.38), False)),
             "yaw": 0.65, "label": "corner:arc", "tol": 0.12},
            {"kind": "goto", "p": np.asarray(post.lane_point(0.50, lane)),
             "yaw": post.yaw, "label": "corner:exit", "tol": 0.12},
            {"kind": "follow", "seg": post, "stations": post_stations,
             "label": "north:east:dense", "style": cfg.touch_style,
             "standoff": lane},
            {"kind": "goto", "p": np.asarray(_tr(GOAL_FWD, False)),
             "yaw": post.yaw, "label": "goal", "tol": 0.20},
        ]

    # the two faces as WallSegments in the SCENE frame, oriented along travel
    if not reverse:
        # west wall northbound: surface x=0, y -4.0 .. -0.75
        # the lane may run to 0.58 short of the corner (base clearance to the
        # facing north wall); the HAND then reaches to ~0.4 of the corner
        a0, a1 = _tr((0.0, -4.05), reverse), _tr((0.0, -0.58), reverse)
        seg_a = WallSegment(p0=np.asarray(a0), tangent=np.array([0.0, 1.0]),
                            normal=N_WEST.copy(),
                            length=float(a1[1] - a0[1]), name="west:north")
        # north wall eastbound: starts 0.30 past the corner so the first
        # press can land its hand ~0.6 from the corner
        b0 = _tr((0.30, 0.0), reverse)
        seg_b = WallSegment(p0=np.asarray(b0), tangent=np.array([1.0, 0.0]),
                            normal=N_NORTH.copy(), length=2.95,
                            name="north:east")
        corner_sc = _tr(CORNER, reverse)
        pillar_sc = _tr(np.asarray(
            GOAL_PILLAR_XY_REV if reverse else GOAL_PILLAR_XY, float),
            reverse)
        arc = corner_turn_path(corner_sc, N_WEST, N_NORTH, cfg.standoff_m)
        ate_faces = [
            {"seg": seg_a, "s_lo": seg_a.length - CORNER_REACH_M,
             "s_hi": seg_a.length - CORNER_STATION_ROOM_M, "face_n": N_WEST},
            {"seg": seg_b, "s_lo": 0.12, "s_hi": CORNER_EXIT_REACH_M,
             "face_n": N_NORTH}]
        goal_sc = _tr(GOAL_FWD, reverse)
        pre = seg_a
        post = seg_b
    else:
        # north wall westbound: surface y=0, x 3.25 .. 0.55
        b0 = _tr((3.25, 0.0), reverse)
        seg_a = WallSegment(p0=np.asarray(b0), tangent=np.array([-1.0, 0.0]),
                            normal=N_NORTH.copy(), length=2.95,
                            name="north:west")
        # west wall southbound: surface x=0, y -0.58 .. -4.05
        a0 = _tr((0.0, -0.58), reverse)
        seg_b = WallSegment(p0=np.asarray(a0), tangent=np.array([0.0, -1.0]),
                            normal=N_WEST.copy(), length=3.47,
                            name="west:south")
        corner_sc = _tr(CORNER, reverse)
        pillar_sc = _tr(np.asarray(
            GOAL_PILLAR_XY_REV if reverse else GOAL_PILLAR_XY, float),
            reverse)
        arc = corner_turn_path(corner_sc, N_NORTH, N_WEST, cfg.standoff_m)
        ate_faces = [
            {"seg": seg_a, "s_lo": seg_a.length - CORNER_REACH_M,
             "s_hi": seg_a.length - CORNER_STATION_ROOM_M, "face_n": N_NORTH},
            {"seg": seg_b, "s_lo": 0.12, "s_hi": CORNER_EXIT_REACH_M,
             "face_n": N_WEST}]
        goal_sc = _tr(START_FWD, reverse)
        pre = seg_a
        post = seg_b

    steps = [
        {"kind": "goto", "p": np.asarray(pre.lane_point(0.25, lane)),
         "yaw": pre.yaw, "label": "approach", "tol": 0.12},
        # rhythm presses on the approach wall, stopping short of the ATE
        # range -- and the follow segment itself is TRIMMED there, because a
        # follow runs out to its segment end and the first room2 run walked
        # clean past every face-A candidate before the ATE could ask for one
        {"kind": "follow",
         "seg": WallSegment(p0=pre.p0, tangent=pre.tangent, normal=pre.normal,
                            length=float(ate_faces[0]["s_lo"] - 0.05),
                            name=pre.name),
         "stations": [s for s in _stations_for(pre.length, cfg.press_spacing_m)
                      if s <= ate_faces[0]["s_lo"] - 0.29],
         "label": f"{pre.name}:rhythm"},
        # the landmark: ATE owns everything between here and the far face
        {"kind": "ate_corner", "faces": ate_faces, "corner": corner_sc,
         "between": [{"kind": "goto", "p": np.asarray(p), "yaw": y,
                      "label": "corner:arc", "tol": 0.14}
                     for p, y in arc],
         "label": "corner"},
        # rhythm presses out to the goal on the far wall; always at least one
        # between the ATE range and the goal, so the "feel forward" cadence
        # never goes silent on the home stretch
        {"kind": "follow", "seg": post,
         "stations": _merge_stations(
             [s for s in _stations_for(post.length, cfg.press_spacing_m)
              if s > ate_faces[1]["s_hi"] + 0.3],
             round(ate_faces[1]["s_hi"] + 0.95, 2)
             if ate_faces[1]["s_hi"] + 0.95 <= post.length - 0.3 else None,
             cfg.press_spacing_m),
         "label": f"{post.name}:rhythm"},
        # THE GOAL MARKER.  A flat wall pins only its own normal, so the axis
        # ALONG the last wall -- the one that says "you have arrived" -- was
        # never observable and arrival was being asserted from dead reckoning
        # alone.  A pillar is a POINT: one touch pins both axes at once.  The
        # ported ATE already models cylinders, so the marker is a landmark
        # visit like the corner, not a special case.
        {"kind": "ate_pillar", "centre": np.asarray(pillar_sc, float),
         "radius": float(GOAL_PILLAR_R), "label": "goal_post",
         # the face the marker protrudes FROM decides the approach side; a
         # hardcoded "apex faces -y" served the fwd (north-wall) marker by
         # coincidence and aimed the reverse (west-wall) probes at the wall's
         # inside
         "normal": np.asarray(N_NORTH if not reverse else N_WEST,
                              float)},   # frames differ by translation only
        {"kind": "goto", "p": np.asarray(goal_sc), "yaw": post.yaw,
         "label": "goal", "tol": 0.20},
    ]
    return tag_gaits(steps, scene)
