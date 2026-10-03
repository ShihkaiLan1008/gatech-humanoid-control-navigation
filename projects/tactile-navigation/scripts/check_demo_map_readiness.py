"""Phase 5 -- demo readiness of the reconstructed room.

    PYTHONPATH=src python3 scripts/check_demo_map_readiness.py

Checks only what the task asks to check first: route geometry, G1 body
clearance, tactile reachability, start/goal placement, and whether anything
about the map obviously disagrees with the controller stack.  It does NOT run
an episode and introduces no controller work.

The G1 footprint is MEASURED off the shipped model rather than assumed, and the
spawn check compiles the robot into the reconstructed room and asks MuJoCo for
the contacts.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from demo_map import build_demo_map, build_reference_route
from demo_map.geometry import LANE_OFF, clearance, in_rooms

G1_SCENE = Path("tactile_controller/src/assets/g1_description/g1_tactile_scene.xml")

# Measured V1 stance geometry, quoted from lane_graph.maps_v3's header.
PRESS_STANCE = (0.30, 0.38)   # standoff to the pressed face
WRIST_REACH = 0.43            # wrist reach cap

_PASS, _FAIL, _NOTE = "PASS", "FAIL", "NOTE"
_results: list[tuple[str, str, str]] = []


def check(tag: str, ok: bool, msg: str) -> bool:
    _results.append((_PASS if ok else _FAIL, tag, msg))
    return ok


def note(tag: str, msg: str) -> None:
    _results.append((_NOTE, tag, msg))


# ── G1 footprint, measured off the shipped model ──────────────────────────────


def g1_footprint() -> dict | None:
    """Horizontal half-extent of the G1's collision geoms about its root, at the
    model's own default pose."""
    try:
        import mujoco
    except ImportError:
        return None
    if not G1_SCENE.exists():
        return None
    model = mujoco.MjModel.from_xml_path(str(G1_SCENE))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    root = np.array(data.qpos[:3])
    r = 0.0
    for i in range(model.ngeom):
        if model.geom_contype[i] == 0 and model.geom_conaffinity[i] == 0:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
        if name in ("floor", None):
            continue
        d = float(np.linalg.norm(data.geom_xpos[i][:2] - root[:2]))
        r = max(r, d + float(model.geom_rbound[i]))
    return dict(radius=r, root_z=float(root[2]), ngeom=int(model.ngeom))


def _room_and_robot_geoms(model):
    import mujoco

    room, robot = [], []
    for i in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
        if name.startswith(("wall_", "curve_", "box_")):
            room.append(i)
        elif name.startswith("g1_") and name != "g1_floor":
            if model.geom_contype[i] or model.geom_conaffinity[i]:
                robot.append(i)
    return room, robot


def sweep_route(m, r, step: float = 0.5, distmax: float = 1.5) -> dict | None:
    """Stand the G1 at poses along the reference route and ask MuJoCo for the
    true geom-to-geom separation between the robot and the room.

    This is the body-clearance check the disc approximation only estimates: the
    robot is its real collision hull, at its nominal standing pose, yawed along
    the route tangent.  It is a static envelope check, not a gait."""
    try:
        import mujoco
    except ImportError:
        return None
    if not G1_SCENE.exists():
        return None

    from demo_map.mujoco_scene import combined_with_g1

    model = combined_with_g1(m, at=m.start.center, yaw=m.start.heading_rad,
                             g1_scene=str(G1_SCENE))
    data = mujoco.MjData(model)
    room_g, robot_g = _room_and_robot_geoms(model)

    poly = r.polyline(0.02)
    keep = [0]
    for i in range(1, len(poly)):
        if float(np.linalg.norm(poly[i] - poly[keep[-1]])) >= step:
            keep.append(i)
    if keep[-1] != len(poly) - 1:
        keep.append(len(poly) - 1)

    worst = (np.inf, None, None, None)
    contacts = 0
    for i in keep:
        j = min(i + 1, len(poly) - 1)
        k = i - 1 if j == i else i
        tangent = poly[j] - poly[k]
        yaw = math.atan2(float(tangent[1]), float(tangent[0])) if np.any(tangent) else 0.0
        data.qpos[:] = model.qpos0
        data.qpos[0:2] = poly[i]
        data.qpos[3:7] = [math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw)]
        mujoco.mj_forward(model, data)
        for a in robot_g:
            for b in room_g:
                d = mujoco.mj_geomDistance(model, data, a, b, distmax, None)
                if d < worst[0]:
                    worst = (float(d), tuple(poly[i]),
                             mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, a),
                             mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, b))
                if d <= 0.0:
                    contacts += 1
    return dict(n_poses=len(keep), min_sep=worst[0], at_xy=worst[1],
                robot_geom=worst[2], room_geom=worst[3], penetrations=contacts,
                nbody=int(model.nbody), ngeom=int(model.ngeom),
                n_robot_geoms=len(robot_g), n_room_geoms=len(room_g))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lane-off", type=float, default=LANE_OFF)
    ap.add_argument("--step", type=float, default=0.5,
                    help="spacing of the swept G1 poses along the route [m]")
    ap.add_argument("--walls", type=Path, default=None,
                    help="scan walls JSON (implies --map scan)")
    ap.add_argument("--map", choices=("tape", "scan"), default="tape",
                    help="tape = the drawing's reconstruction; "
                         "scan = the map extracted from the iPhone mesh")
    args = ap.parse_args()

    if args.walls:
        args.map = "scan"
    if args.map == "scan":
        from demo_map.scan_map import (build_scan_map, build_scan_route,
                                       build_scan_traverse)
        m = build_scan_map(args.walls) if args.walls else build_scan_map()
        # the ten-segment demo route belongs to the measured corridor; a room
        # with no such anchors gets the generic clear traverse instead
        r = (build_scan_route(m, lane_off=args.lane_off) if m.key_points
             else build_scan_traverse(m))
        if r is None:
            raise SystemExit("no collision-free path across this map")
    else:
        m = build_demo_map()
        r = build_reference_route(m, lane_off=args.lane_off)
    print(f"map: {args.map}  ({len(m.walls)} walls, {len(m.curves)} curves, "
          f"{len(m.boxes)} clutter boxes)")

    # 1 ── collision-free route geometry ------------------------------------
    pts = r.polyline(0.01)
    clr = clearance(pts, m)
    check("route/collision-free", float(clr.min()) > 0.0,
          f"min surface clearance {clr.min():.3f} m over {r.length:.2f} m, "
          f"{len(r.segs)} segments")
    check("route/continuous", all(
        np.allclose(a.p1, b.p0, atol=1e-9) for a, b in zip(r.segs[:-1], r.segs[1:])),
        "segment ends coincide exactly")
    check("route/inside-free-space", bool(in_rooms(pts, m).all()),
          "every sampled point lies in a declared free-space rectangle")

    # 2 ── G1 body clearance -------------------------------------------------
    fp = g1_footprint()
    if fp is None:
        note("g1/footprint", "G1 model unavailable -- skipped")
    else:
        check("g1/body-clearance", float(clr.min()) > fp["radius"],
              f"route min clearance {clr.min():.3f} m vs measured G1 footprint "
              f"radius {fp['radius']:.3f} m "
              f"(margin {clr.min() - fp['radius']:+.3f} m)")

    # 3 ── reachable tactile walls ------------------------------------------
    lo, hi = PRESS_STANCE
    for s in r.segs:
        if s.kind not in ("wall_follow", "curve_follow", "corner_turn"):
            continue
        d = clearance(s.sample(0.01), m)
        check(f"reach/{s.sid}", float(d.max()) <= WRIST_REACH + args.lane_off,
              f"follows {s.follows or '-':8s} lane gap "
              f"{d.min():.3f}-{d.max():.3f} m; step in "
              f"{d.max() - hi:.2f}-{d.max() - lo:.2f} m to press")
    if any(s.kind in ("wall_follow", "curve_follow", "corner_turn") for s in r.segs):
        note("reach/stance",
             f"the lane runs at {args.lane_off:.2f} m; a press needs "
             f"{lo:.2f}-{hi:.2f} m, so every press is a "
             f"{args.lane_off - hi:.2f}-{args.lane_off - lo:.2f} m step in -- "
             "the standard V3 contract, unchanged by this map")

    # 4 ── start / goal placement -------------------------------------------
    for name, reg in (("start", m.start), ("goal", m.goal)):
        c = float(clearance([reg.center], m)[0])
        check(f"{name}/clearance", c >= 0.60,
              f"{c:.3f} m clear (spawn needs 0.60 m), inside free space: "
              f"{bool(in_rooms([reg.center], m)[0])}")
    if m.openings:
        check("goal/on-doorway",
              abs(m.goal.center[0] - m.opening('DOOR').mid[0]) < 1e-9,
              "goal is centred on the doorway, on the room side")
    else:
        note("goal/placement", m.goal.note)

    sw = sweep_route(m, r, step=args.step)
    if sw is None:
        note("g1/sweep", "G1 model unavailable -- swept body check skipped")
    else:
        check("g1/route-body-clearance", sw["penetrations"] == 0,
              f"G1 stood at {sw['n_poses']} poses along the route "
              f"({sw['n_robot_geoms']} robot geoms vs {sw['n_room_geoms']} room "
              f"geoms): {sw['penetrations']} penetrations, closest approach "
              f"{sw['min_sep']:.3f} m ({sw['robot_geom']} -> {sw['room_geom']} "
              f"at x={sw['at_xy'][0]:.2f}, y={sw['at_xy'][1]:.2f})")
        note("g1/sweep-caveat",
             "nominal standing pose yawed along the route tangent -- a static "
             "envelope check, not a gait; swing-leg and arm excursions are the "
             "controller's business")

    # 5 ── map / controller compatibility -----------------------------------
    from demo_map.geometry import HEIGHT, THICKNESS
    from demo_map.mapv3 import demo_map_obstacles, demo_map_v3

    v3 = demo_map_v3(m)
    obs = demo_map_obstacles(m)
    try:
        from lane_graph.maps_v3 import validate
        check("compat/v3-validate", validate(v3) == [],
              "lane_graph.maps_v3.validate reports nothing")
    except ImportError:
        pass
    check("compat/wall-thickness", abs(THICKNESS - 0.10) < 1e-12,
          f"{THICKNESS} m matches MAZE_WALL_THICKNESS and lane_graph THICK")
    note("compat/wall-height",
         f"{HEIGHT} m; the shipped asset default is 1.5 m. mj_assets now carries "
         "a per-obstacle half_height, so this scene is taller without moving "
         "the shared default")
    note("compat/obstacle-count",
         f"{len(obs)} injected entities ({sum(o.kind == 'wall' for o in obs)} walls, "
         f"{sum(o.kind == 'pillar' for o in obs)} cylinders) -> "
         f"{2 * len(obs)} hand contact sensors. The V2/V3 backend rewrites the "
         f"FSM wall map wholesale, so {sum(o.kind == 'wall' for o in obs)} walls "
         "is not a capacity problem, but the sensor count is the thing to watch "
         "when this map first meets the mjlab env")

    pts_all = np.array([p for w in m.walls for p in (w.p0, w.p1)], float)
    reach_m = float(np.linalg.norm(pts_all, axis=1).max())
    note("compat/scene-extent",
         f"{np.ptp(pts_all[:, 0]):.1f} x {np.ptp(pts_all[:, 1]):.1f} m, farthest wall "
         f"point {reach_m:.1f} m from the map origin. The V3 rooms run 7-9 m and "
         "the wall-following task's arena_radius default is 5.0 m -- placement "
         "bypasses it, but nothing in this stack has been driven this far out")
    if m.curves:
        note("compat/curved-features",
             "C1/C2 stand 0.29 m and 0.47 m proud; the V3 bulge envelope assumes "
             "<=0.15 m so a straight lane stays smooth. Here the lane must bow, "
             "which the reference route does analytically")
    hands = {s.hand for s in r.segs if s.hand}
    if len(hands) > 1:
        # the switch straddles the hand-less transit, so compare each handed
        # segment with the LAST handed one, not with its immediate neighbour
        sw, last = None, None
        for sg in r.segs:
            if not sg.hand:
                continue
            if last is not None and sg.hand != last:
                sw = sg.sid
                break
            last = sg.hand
        note("compat/hand-switch",
             f"the route follows with the LEFT hand, then the RIGHT after "
             f"{sw}. A follower that pins one hand will need the switch before "
             "this route can be executed end to end")
    elif not hands:
        note("route/kind",
             "this map has no demo route: the path is the generic clear "
             "traverse across the scanned free space, so nothing is followed "
             "by hand and no press stance applies")

    # ── report ---------------------------------------------------------------
    w = max(len(t) for _, t, _ in _results)
    print()
    for status, tag, msg in _results:
        print(f"[{status:4s}] {tag:{w}s}  {msg}")
    bad = [t for s, t, _ in _results if s == _FAIL]
    n_pass = sum(s == _PASS for s, _, _ in _results)
    print(f"\n{n_pass} checks passed, {len(bad)} failed"
          + (f": {', '.join(bad)}" if bad else ""))
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
