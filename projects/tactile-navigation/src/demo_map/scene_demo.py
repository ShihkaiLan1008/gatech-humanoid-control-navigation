"""The scanned demo room as a V3Scene: what the planner, belief and MuJoCo
backend all consume.

FRAME.  ``MujocoBackendV2`` anchors the map origin at the SPAWN -- ``base0`` is
wherever the robot resets and ``v`` is its facing -- so a scene's coordinates
must put the start at (0, 0).  The scan map keeps its own frame (origin at C1's
west chord end, the datum the tape used); this module translates everything by
``-start`` once, here, and nowhere else.  ``TO_SCAN`` restores scan/drawing
coordinates for plots, so every figure still matches the reference route figure.

The spawn FACES map +y (the backend's convention), which in this room is the
long wall 0.67 m north of the start.  The route heads east, so a run begins
with a turn in place -- visible, and honest about how the backend frames the
world.

PRIOR.  Unimodal, elongated along +x: the long wall's normal pins y and yaw
from the first press, but x -- position ALONG the wall -- is exactly what a
flat touch cannot pin (the rank-1 result).  What pins x is the map's own
landmarks: the two curved features and the wall ends the route was drawn
through.  That is the story this demo exists to show, so the prior starts with
the uncertainty those landmarks remove.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from global_nav.floorplan import Floorplan, MapFeature, RouteEdge
from global_nav.geometry import ConvexPolygon
from lane_graph.maps_v3 import followable
from lane_graph.scene3 import V3Scene, _ngon

from execution.wall_lane import WallSegment, face_present, raycast

from .executor import (feasible_stations,
                       STATION_END_MARGIN_M, clearance_lift, route_around,
                       corner_turn_path, tag_gaits,
                       _stations_for, _wall_segment_for)
from .geometry import CylObs
from .landmarks import harvest
from .mapv3 import demo_map_v3
from .scan_map import build_scan_map, build_scan_route

DEMO_SCENE_NAME = "demo_room_scan"

# Followed-lane offset for EXECUTION, not the 0.56 m of the drawn reference.
# The turn-press cycle was measured to need the face at 0.36-0.42 m when the
# base turns in place (docs/HW_ROBUSTNESS.md 19): the parked hands sweep a
# 0.32 m circle no command shrinks, and the arm's forward reach asymptotes at
# ~0.43 m.  0.56 m is unreachable without the approach step the demo forbids.
TURN_PRESS_LANE_OFF = 0.40


# OWNER-PLACED GOAL PILLAR AT THE DESTINATION (owner, 2026-08-29).
#
# Four runs (v10-v13) each died differently in the end room, and every death
# traces to the same axis: position ALONG the last walls, which no flat face
# observes.  room2 solved exactly this with an owner-placed goal marker --
# "'You are here' cannot be dead-reckoned" -- and the owner has now placed the
# same marker here: a pillar standing ON the goal, so the route ends at a body
# whose normal turns with position.  IDENTIFYING THE PILLAR IS THE STOP: the
# confirming visit is the last step, and there is no walk to the goal centre
# afterwards, because the marker occupies it (see `plan_steps_demo_room`).
#
# Radius is room2's measured 0.16 (a 0.09 marker was narrower than the belief
# error and the hand kept missing it; this room's end-game error runs larger
# still).  Height is the G1's own standing height, so the marker reads as
# "robot-sized post" in the video and stands far above press height.
GOAL_PILLAR_R = 0.16
GOAL_PILLAR_Z1 = 1.32              # the G1's standing height


def build_demo_room_map():
    """The scanned demo map plus the owner-placed goal pillar.

    Appended AFTER `build_scan_map` on purpose: `_pick_goal` chooses the
    roomiest free-space point, and handing it a pillar already standing there
    would push the goal away from its own marker."""
    m = build_scan_map()
    # Owner correction, 2026-09-09: the eastern corridor end is open and
    # cannot be traversed. Its map cutoff is not a wall-end landmark.
    lower = max((w for w in m.walls if w.inward[1] > .9), key=lambda w: w.length)
    east = max(lower.p0[0], lower.p1[0])
    return dataclasses.replace(m, pillars=list(m.pillars) + [_goal_marker(m)],
                               open_boundaries=[(1., 0., float(east))])


def _goal_marker(m) -> CylObs:
    """The goal marker as a half-cylinder standing proud of the wall the route
    ends along, abeam the goal.

    Placed on the END ROOM'S SOUTH WALL because that is the face the home
    stretch follows (route ``R8``), and abeam the goal because the marker's
    job is to say "the goal is here", not to be somewhere else.  ``WallSeg``
    endpoints lie on the room-side face, so ``room_s_y`` IS the face: a
    cylinder centred there shows exactly half of itself, which is the shape
    the owner asked for and the one room2's marker already has.

    Derived from the map's own anchors, never written down: the wall moves
    when the scan is reprocessed and the marker has to move with it."""
    return CylObs(
        pid="goal_post", center=(float(m.goal.center[0]),
                                 float(m.key_points["room_s_y"])),
        radius=GOAL_PILLAR_R, z0=0.0, z1=GOAL_PILLAR_Z1,
        source="owner-placed goal marker",
        note="half-cylinder proud of the end room's south wall, abeam the "
             "goal: a curved face whose normal turns with position, so one "
             "touch pins both axes")


def scan_start_offset() -> np.ndarray:
    """Scan-frame coordinates of the map origin (the start)."""
    return np.asarray(build_scan_map().start.center, float)


def translated_scan_map(m=None):
    """The scan map with the start at (0, 0): the backend's frame contract."""
    m = m or build_demo_room_map()
    d = -np.asarray(m.start.center, float)

    def pt(p):
        return (float(p[0] + d[0]), float(p[1] + d[1]))

    walls = [dataclasses.replace(w, p0=pt(w.p0), p1=pt(w.p1)) for w in m.walls]
    curves = [dataclasses.replace(c, center=pt(c.center), chord0=pt(c.chord0),
                                  chord1=pt(c.chord1)) for c in m.curves]
    boxes = [dataclasses.replace(b, center=pt(b.center)) for b in m.boxes]
    pillars = [dataclasses.replace(p, center=pt(p.center)) for p in m.pillars]
    hulls = [dataclasses.replace(h, center=pt(h.center),
                                 verts=tuple(pt(v) for v in h.verts))
             for h in m.hulls]
    openings = [dataclasses.replace(o, p0=pt(o.p0), p1=pt(o.p1))
                for o in m.openings]
    rooms = [(r[0] + d[0], r[1] + d[1], r[2] + d[0], r[3] + d[1])
             for r in m.rooms]
    start = dataclasses.replace(m.start, center=pt(m.start.center))
    goal = dataclasses.replace(m.goal, center=pt(m.goal.center))
    kp = {k: (v + (d[0] if k.endswith(("_x", "_s1", "_x1")) else d[1]))
          for k, v in m.key_points.items()}
    return dataclasses.replace(m, walls=walls, curves=curves, boxes=boxes,
                               pillars=pillars, hulls=hulls, openings=openings,
                               rooms=rooms, start=start, goal=goal,
                               key_points=kp,
                               open_boundaries=[(nx, ny, offset + nx*d[0] + ny*d[1])
                                                for nx, ny, offset in m.open_boundaries])


def demo_route(lane_off: float = TURN_PRESS_LANE_OFF):
    """The MANUAL reference route, in the scene frame, at the execution lane."""
    return build_scan_route(translated_scan_map(), lane_off=lane_off)


def build_demo_scene(m=None) -> V3Scene:
    """The shipped scene.  ``m`` admits a different map -- the box-era tests
    measure the flagged BOX_LANDMARKS machinery against the pre-marker room,
    because the goal pillar stands where the end-room boxes' stances were."""
    m = m if m is not None else translated_scan_map()
    spec = demo_map_v3(m)
    bodies = [ConvexPolygon(w.poly(), f"wall_{i}", "surface")
              for i, w in enumerate(spec.walls)]
    bodies += [_ngon(b.center, b.radius, name=f"bulge_{i}")
               for i, b in enumerate(spec.bulges)]
    bodies += [ConvexPolygon(p.poly(), f"prot_{i}", "surface")
               for i, p in enumerate(spec.prots)]
    feats = tuple(MapFeature(i, (float(x), float(y)), "landmark", lab)
                  for i, (x, y, lab) in enumerate(spec.landmarks))
    xs = [v[0] for b in bodies for v in np.asarray(b.verts)]
    ys = [v[1] for b in bodies for v in np.asarray(b.verts)]
    lo = (min(xs) - 1.0, min(ys) - 1.0)
    hi = (max(xs) + 1.0, max(ys) + 1.0)
    edges = (RouteEdge(0, lo, (hi[0], lo[1]), half_width_m=(hi[1] - lo[1]),
                       successors=(), name="freespace"),)
    plan = Floorplan(tuple(bodies), feats, edges, route=(0,),
                     name=DEMO_SCENE_NAME)
    return V3Scene(
        name=DEMO_SCENE_NAME, spec=spec, plan=plan, lanes=followable(spec),
        start_map_xy=(0.0, 0.0), goal_map_xy=tuple(spec.goal),
        goal_radius_m=spec.goal_r, standoff_m=TURN_PRESS_LANE_OFF,
        prior_components=(((0.0, 0.0), np.pi / 2.0, 0, 1.0),),
        prior_sigma_along_m=0.14, prior_sigma_lateral_m=0.06)


# ── the demo room compiled with room2's planner logic ──────────────────────
#
# room2's compiler is one sentence long: ATE OWNS THE LANDMARKS, the flat runs
# keep the cheap rule-based "feeling forward" cadence, and arrival is confirmed
# by touching something that is not a wall.  Everything below is that sentence
# applied to a room with 36 non-wall bodies instead of two, so the bodies are
# HARVESTED (`demo_map.landmarks`) rather than named by hand.
#
# The three differences from room2 are all facts about this room, not changes
# of policy, and each is measured:
#
#   1. The corner is CONVEX.  room2's L is concave and `LandmarkATE.corner`
#      models only that; its own docstring records that pointing it at a convex
#      vertex scores every candidate ~0.  `LandmarkATE.convex_corner` is the
#      box model the same docstring prescribes.
#   2. The goal marker STANDS ON THE GOAL rather than half-embedded in a wall
#      (room2's placement).  The end room's own protrusions were tried as the
#      confirming landmark first and four runs died in four ways, all on the
#      unobservable along-wall axis; the owner then placed `goal_post` at the
#      destination itself (2026-08-29).  Because the marker occupies the goal
#      centre, the confirming visit IS arrival and the final walk to the goal
#      is dropped -- walking there would walk into the marker.
#   3. The scan reconstructs the end room's south wall as TWO slabs 0.17 m
#      apart in y -- the near one stops at x = 14.61 -- so the home stretch
#      follows the far slab's own face.  Scheduling the drawn route's face
#      there was already pressing air: the pre-existing compiler's single
#      station on that wall lands at x = 14.98, where `face_present` is False.

LANDMARK_GUARD_M = 0.30
"""Rhythm stations are dropped this close to a landmark visit.

room2 spells the same rule out per face (`s <= s_lo - 0.29` before the corner,
`s > s_hi + 0.3` after it).  The point is not tidiness: a rhythm press and a
landmark press at the same arc position are the same touch scheduled twice,
and the second one arrives after the first has already moved the belief out
from under its station."""

def _face_runs(bodies, p_start, tangent, normal, *, span_m: float,
               step: float = 0.06, tol_m: float = 0.02, min_run_m: float = 0.55):
    """Every contiguous stretch of wall face a ray finds under a lane.

    Returns ``[(surface_point_at_run_start, run_length, face_offset), ...]``.

    Why this is measured rather than taken from the drawn route: a drawn
    segment carries ONE constant lane offset, and this room's end-room south
    wall is reconstructed as two slabs whose faces differ by 0.17 m in y --
    the near one stopping at x = 14.61.  The pre-existing compiler put its one
    station on that wall at x = 14.98, where `face_present` is False, so the
    press swept air.  Sweeping a ray under the lane asks the same question
    `face_present` asks at every station, and asks it before scheduling."""
    n = np.asarray(normal, float)
    t = np.asarray(tangent, float)
    p0 = np.asarray(p_start, float)
    probe = 0.9
    runs, cur = [], None
    for a in np.arange(0.0, span_m + 1e-9, step):
        d, nn, _hit = raycast(p0 + a * t + probe * n, -n, probe + 0.1,
                              tuple(bodies))
        off = None
        if d is not None and float(np.asarray(nn, float) @ n) > 0.999:
            off = probe - float(d)
        if off is None:
            if cur is not None:
                runs.append(cur)
            cur = None
        elif cur is not None and abs(off - cur[2]) <= tol_m:
            cur = (cur[0], a, cur[2])
        else:
            if cur is not None:
                runs.append(cur)
            cur = (a, a, off)
    if cur is not None:
        runs.append(cur)
    return [(p0 + a0 * t + off * n, float(a1 - a0), float(off))
            for a0, a1, off in runs if a1 - a0 >= min_run_m]


def _slice(seg, a: float, b: float, name: str | None = None):
    """The sub-segment of ``seg`` between arc lengths a and b."""
    return WallSegment(p0=seg.surface_point(float(a)),
                       tangent=np.asarray(seg.tangent, float).copy(),
                       normal=np.asarray(seg.normal, float).copy(),
                       length=float(b - a), name=name or seg.name)


def _rhythm(seg, spacing: float, bodies, standoff_m: float, blocked=()):
    """Rule-based cadence stations on a flat run, minus the impossible ones.

    Two filters on top of `executor._stations_for`'s even rhythm, and the
    demo room needs both where room2 needed neither:

      * a station whose face the map says is ABSENT is dropped -- this room's
        long wall is interrupted by two curved features, a branch mouth and a
        3 m stretch the scan never saw, and the pre-existing compiler
        scheduled presses into all of them;
      * a station inside a landmark's guard band is dropped, because the
        landmark visit is already pressing there and better."""
    out = []
    for s in _stations_for(seg.length, spacing):
        if any(lo <= s <= hi for lo, hi in blocked):
            continue
        if not face_present(seg, float(s), tuple(bodies),
                            standoff_m=float(standoff_m)):
            continue
        out.append(s)
    return out


def _landmark_step(site, cfg, *, required: bool):
    """One harvested body -> one executor landmark visit."""
    step = {"kind": "ate_landmark", "shape": site.kind,
            "centre": np.asarray(site.centre, float),
            "normal": np.asarray(site.n_out, float),
            # HOW FAR THE PRESSED FACE STANDS OFF THE CENTRE, measured by the
            # ray that found it -- not derived from a half-extent.  A box's
            # half-depth is only its reach when the box is square to the wall,
            # and these are scanned: prot_10's front face sits 0.126 m out
            # while its short half-extent is 0.055, so deriving the reach
            # would aim the whole apex-segment 71 mm inside the body.
            "reach_m": float(site.reach_m),
            "label": site.body, "required": required,
            "max_tries": 3 if not required else cfg.gate_extra_presses}
    if site.kind == "cylinder":
        step["radius"] = float(site.radius)
    else:
        step["half_xy"] = tuple(float(v) for v in site.half_xy)
        step["yaw"] = float(site.yaw)
    return step


def plan_steps_demo_room(cfg, scene=None, route=None,
                         route_lane_off: float = TURN_PRESS_LANE_OFF) -> tuple:
    """The author's route, compiled with ATE at every reachable landmark.

    Returns ``(steps, report)``.  The report names every non-wall body the
    room contains, whether it was visited, and -- for the ones that were not
    -- why not, because a landmark dropped silently reads as a landmark that
    was never there.

    The compile does not re-route.  Every step is one of the author's own
    segments in the author's order, exactly as `plan_steps_for_route` does;
    what changes is WHO chooses the touches on it."""
    scene = scene or build_demo_scene()
    route = route or demo_route(route_lane_off)
    bodies = tuple(scene.plan.bodies)
    lane = float(cfg.standoff_m)
    poly = route.polyline(0.05)
    sites, skipped = harvest(scene, poly, standoff_m=lane)
    used, steps = set(), []

    # arc position of each route segment, so a harvested site can be placed
    # in the leg it belongs to rather than by nearest-anything
    bounds, acc = [], 0.0
    for rs in route.segs:
        bounds.append((acc, acc + rs.length))
        acc += rs.length

    # the goal-confirming landmark: the one nearest the goal, which is what
    # room2's owner-placed marker was FOR.  Named by distance, not by hand.
    goal_xy = np.asarray(route.goal, float)
    goal_site = min(sites, key=lambda s: float(np.linalg.norm(
        np.asarray(s.centre, float) - goal_xy)), default=None)
    tail_goto = None

    def _is_goal(site):
        return goal_site is not None and site.body == goal_site.body

    def visit(site, required=False):
        used.add(site.body)
        steps.append(_landmark_step(site, cfg, required=required))

    def emit_follow(seg, a, b, sid):
        """One rhythm chunk, if there is enough wall left to be worth one."""
        if b - a < 0.95:            # under the acquire gate's run-in floor
            return
        sub = _slice(seg, a, b, f"{seg.name}@{a:.1f}")
        steps.append({"kind": "follow", "seg": sub,
                      "stations": _rhythm(sub, cfg.press_spacing_m, bodies,
                                          lane),
                      "label": f"{sid}:rhythm"})

    def _goto(p, yaw, label, tol):
        """Skip a waypoint the previous step already sent the robot to."""
        p = np.asarray(p, float)
        for prev in reversed(steps):
            if prev["kind"] == "goto":
                if float(np.linalg.norm(np.asarray(prev["p"], float) - p)) < 0.02:
                    return
                break
            if prev["kind"] in ("follow", "ate_corner"):
                break
        # transit only: stations keep their press stances, but a waypoint
        # the robot merely walks through must not park the carried tip
        # inside a mapped body -- and neither may the LEG that reaches it
        # (endpoint-only lifting was measured insufficient: the straight
        # path still cut bulge envelopes and the carried hand ground them)
        p = clearance_lift(p, scene.plan.bodies)
        prev = None
        for st in reversed(steps):
            if st["kind"] == "goto":
                prev = np.asarray(st["p"], float)
                break
            if st["kind"] == "follow":
                sseg = st["seg"]
                prev = np.asarray(sseg.lane_point(
                    sseg.length, TURN_PRESS_LANE_OFF), float)
                break
        if prev is not None:
            last = prev
            for via in route_around(prev, p, float(yaw), scene.plan.bodies):
                d = np.asarray(via, float) - np.asarray(last, float)
                travel = float(np.arctan2(d[1], d[0]))
                steps.append({"kind": "goto", "p": np.asarray(via, float),
                              "yaw": travel, "label": f"{label}:via",
                              "tol": 0.18, "gait": "walk"})
                last = via
        steps.append({"kind": "goto", "p": p, "yaw": yaw, "label": label,
                      "tol": tol})

    def sites_in(s0, s1):
        return [s for s in sites
                if s.body not in used and s0 - 1e-6 <= s.s_route < s1 + 1e-6]

    for idx, rs in enumerate(route.segs):
        s0, s1 = bounds[idx]
        here = sites_in(s0, s1)
        if rs.kind in ("approach", "transit", "goal_approach"):
            # LANDMARKS FIRST, then the leg's endpoint.  The other order sends
            # the robot across the branch mouth and then walks it back 1.7 m to
            # press the post standing in it -- and "no walking back" is the
            # rule the station machinery itself enforces everywhere else.
            for st in here:                       # e.g. the branch-mouth post
                visit(st, required=_is_goal(st))
            if rs.kind == "goal_approach":
                tail_goto = (np.asarray(rs.p1, float), rs.heading1(), rs.sid)
            else:
                _goto(rs.p1, rs.heading1(), rs.sid, 0.10)
        elif rs.kind == "curve_follow":
            # the drawn bow's entry and exit are kept -- the author's path --
            # but the touching on it is the landmark visit, not two stations
            # at a hardcoded apex offset
            _goto(rs.p0, rs.heading0(), f"{rs.sid}:in", 0.12)
            for st in here:
                visit(st, required=_is_goal(st))
            _goto(rs.p1, rs.heading1(), f"{rs.sid}:out", 0.14)
        elif rs.kind == "corner_turn":
            seg_a = steps_face_a = None
            for prev in reversed(steps):           # the wall we came up
                if prev["kind"] == "follow":
                    seg_a = prev["seg"]
                    steps_face_a = prev
                    break
            vertex = np.asarray(rs.center, float)
            # face B: the wall being entered, taken from the NEXT route
            # segment so its direction is the author's, its length padded so
            # the station machinery has run-out past the last candidate
            nxt = route.segs[idx + 1]
            seg_b_full = _wall_segment_for(nxt, route_lane_off)
            # FACE B STARTS AT THE VERTEX, not where the route resumed.  A
            # route that cuts the corner rejoins the far wall some way past
            # it, and measuring the exit window from there aims it down the
            # wall instead of at the corner.
            s_vtx_b = float((vertex - seg_b_full.p0) @ seg_b_full.tangent)
            seg_b = WallSegment(
                p0=seg_b_full.p0 + s_vtx_b * seg_b_full.tangent,
                tangent=seg_b_full.tangent.copy(),
                normal=seg_b_full.normal.copy(),
                length=max(seg_b_full.length - s_vtx_b, 1.20),
                name=f"{nxt.sid}:corner")
            if seg_a is not None:
                # THE PRESSED WALL IS THE MAP'S, NOT THE WALK'S.  The corner
                # needs wall on both sides of the vertex to bracket it, and
                # the map says that wall is there -- a vertex IS where two
                # faces meet.  How far the ROUTE happened to travel along it
                # is a separate question: room2's own hand-written compile
                # pressed a 3.47 m face while its route left the wall at
                # 2.1 m, and the ATE reached the corner from an excursion.
                #
                # Taking the walked run instead is what broke: the automatic
                # route cuts the corner, so `s_vtx` landed 1.65 m PAST the
                # run's end, `hi_a` clamped back to the run, and the window
                # came out [3.15, 1.85] -- negative width, no candidate can
                # fall inside it, and the corner gate failed with nothing to
                # press.  Extending face A to the vertex is what makes the
                # window well-formed BY CONSTRUCTION rather than by the
                # caller happening to have walked far enough.
                s_vtx = float((vertex - seg_a.p0) @ seg_a.tangent)
                if s_vtx > seg_a.length:
                    seg_a = WallSegment(p0=seg_a.p0.copy(),
                                        tangent=seg_a.tangent.copy(),
                                        normal=seg_a.normal.copy(),
                                        length=s_vtx, name=seg_a.name)
                # NO WINDOW.  What counts as "the corner" on each face is
                # discovered by `feasible_stations` at execution time; the
                # compile's job is only to say WHICH faces meet and where.
                # The band the constants used to name is a consequence of
                # the base circle and the arm, and it differs by vertex
                # angle, by room, and by whatever is parked in the wedge.
                #
                # The rhythm is trimmed against that same discovery, so the
                # cadence stops exactly where the corner's own presses can
                # begin rather than `CORNER_REACH_M` before wherever the
                # window happened to be put.
                band = feasible_stations(seg_a, bodies, lane)
                corner_lo = (min((b for b in band if b >= s_vtx - 1.6),
                                 default=seg_a.length)
                             if band else seg_a.length)
                steps_face_a["stations"] = [
                    s for s in steps_face_a["stations"]
                    if s <= corner_lo - LANDMARK_GUARD_M]
                faces = [{"seg": seg_a,
                          "face_n": np.asarray(seg_a.normal, float)},
                         {"seg": seg_b,
                          "face_n": np.asarray(seg_b.normal, float)}]
                # CONVEX OR CONCAVE, READ OFF THE TURN.  Rounding a corner the
                # material sits inside of turns the robot TOWARD the wall it
                # is following; an inside corner turns it away.  This used to
                # be the constant True, which is right for this room and
                # wrong for room2 -- and pointing the convex gate at a
                # concave vertex fails it (`gate_failed:convex_corner` on the
                # first end-to-end room2 run off the automatic route).
                ta, tb = seg_a.tangent, seg_b.tangent
                turn = float(ta[0] * tb[1] - ta[1] * tb[0])   # 2-D cross
                convex = turn > 0.0 if rs.hand == "left" else turn < 0.0
                steps.append({
                    "kind": "ate_corner", "faces": faces, "corner": vertex,
                    "convex": convex,
                    "label": "convex_corner" if convex else "concave_corner",
                    # a rolling fillet, not route samples at a FIXED yaw:
                    # constant-yaw waypoints made the turn an in-place spin
                    # at each point, which the G1 cannot do reliably
                    "between": [{"kind": "goto", "p": np.asarray(q, float),
                                 "yaw": float(y), "label": f"{rs.sid}:arc",
                                 "tol": 0.15}
                                for q, y in corner_turn_path(
                                    vertex, faces[0]["face_n"],
                                    faces[1]["face_n"], cfg.standoff_m)]})
            else:
                for q in rs.sample(ds=cfg.arc_step_m)[1:]:
                    _goto(q, rs.heading1(), f"{rs.sid}:corner", 0.15)
        else:                                              # wall_follow
            drawn = _wall_segment_for(rs, route_lane_off)
            runs = _face_runs(bodies, drawn.p0, drawn.tangent, drawn.normal,
                              span_m=drawn.length)
            claimed = set()
            for p_run, len_run, off in runs:
                sub0 = WallSegment(p0=p_run, tangent=drawn.tangent.copy(),
                                   normal=drawn.normal.copy(), length=len_run,
                                   name=(drawn.name if abs(off) <= 0.02
                                         else f"{drawn.name}:felt"))
                # split the run at each landmark on it, so the visit happens
                # where the body is rather than after the whole wall is walked
                cuts = sorted(
                    (float((np.asarray(st.face_xy, float) - sub0.p0)
                           @ sub0.tangent), st) for st in here
                    if -LANDMARK_GUARD_M <= float(
                        (np.asarray(st.face_xy, float) - sub0.p0)
                        @ sub0.tangent) <= len_run + LANDMARK_GUARD_M)
                a = 0.0
                for at, st in cuts:
                    if st.body in used:
                        # a body standing PROUD of its wall breaks the face
                        # into a run before it and a run after, and it sits
                        # inside both runs' guard bands.  Visit it once.
                        a = max(a, min(at + LANDMARK_GUARD_M, len_run))
                        continue
                    b = max(a, min(at - LANDMARK_GUARD_M, len_run))
                    emit_follow(sub0, a, b, rs.sid)
                    visit(st, required=_is_goal(st))
                    claimed.add(st.body)
                    a = min(at + LANDMARK_GUARD_M, len_run)
                emit_follow(sub0, a, len_run, rs.sid)
            for st in here:            # a landmark on a stretch with no face
                if st.body not in claimed:
                    visit(st, required=_is_goal(st))

    for st in sites:                       # anything the legs did not claim
        if st.body not in used:
            visit(st, required=_is_goal(st))
    # a marker standing ON the goal replaces the walk to the goal centre:
    # its required visit is the arrival, and the centre itself is occupied.
    # "On" is literal -- the goal centre lies inside the body's footprint --
    # not "near": a landmark merely abeam the goal leaves the centre free,
    # and the walk there stays.
    marker_on_goal = goal_site is not None and float(np.linalg.norm(
        np.asarray(goal_site.centre, float) - goal_xy)) <= goal_site.reach_m
    if tail_goto is not None and not marker_on_goal:
        # the goal step LAST, after the landmark that confirms arrival --
        # room2's order, and the reason for it: a goal declared before the
        # confirming touch is a goal declared from dead reckoning
        _goto(tail_goto[0], tail_goto[1], tail_goto[2], 0.20)

    report = {
        "n_bodies": sum(1 for b in bodies if not b.name.startswith("wall_")),
        "visited": [s.body for s in sites],
        "goal_landmark": None if goal_site is None else goal_site.body,
        "skipped": skipped,
        "n_rhythm": sum(len(s["stations"]) for s in steps
                        if s["kind"] == "follow"),
        # which route legs got no rule-based cadence at all, and why: a leg
        # whose reconstructed face is shorter than the acquire gate's own
        # run-in cannot host a station, and saying so beats an empty list
        "legs_without_rhythm": sorted(
            {rs.sid for rs in route.segs if rs.kind == "wall_follow"}
            - {s["label"].split(":")[0] for s in steps
               if s["kind"] == "follow"}),
    }
    return tag_gaits(steps, scene), report
