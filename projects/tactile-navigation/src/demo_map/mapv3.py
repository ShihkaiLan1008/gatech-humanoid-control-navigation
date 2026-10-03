"""Adapter: the reconstructed demo room -> the project's existing map/obstacle types.

The reconstruction stores wall FACES, because that is what a tape measure spans.
``lane_graph.maps_v3`` stores wall CENTRELINES, because that is what a MuJoCo box
geom needs.  This module is the one place that conversion happens, so the metric
spec never has to know about slab thickness and the MuJoCo path never has to know
about the drawing.

Producing a real ``MapV3`` buys the whole V3 toolchain unchanged: ``clearance``,
``followable``/``lane_stats``, ``validate``, and -- the point of the exercise --
``mj_assets.build_obstacles_v3`` + ``inject_into_env_cfg``, which is how walls
reach MuJoCo in this project.

The two curved features become ``Bulge`` cylinders.  That is exact, not an
approximation: a chord and a sagitta define a unique circle, and both labelled
numbers are recovered from the cylinder that results (see ``geometry.remeasure``).
C2's centre lands 9.1 cm INSIDE the corridor, so its cylinder is wider below the
wall face than at it -- a real undercut, and the reason the lane has to bow.
"""

from __future__ import annotations

from lane_graph.maps_v3 import THICK, Bulge, MapV3, Prot, Wall

from .geometry import HEIGHT, THICKNESS, DemoMap, build_demo_map

MAP_NAME = "demo_room"
MAP_TITLE = "Measured demo room (docs/demo_maps/measured_map.jpg)"


BOX_LANDMARKS = False
"""Do boxes and hulls reach the simulated scene at all?

False by owner directive: a landmark is a corner or a cylinder, and every
other non-wall body is removed so the surface there is plain wall.  Setting it
True restores the scanned clutter -- and with it the 33 demo-room bodies the
ATE used to rank, so expect a different route.
"""


def demo_map_v3(m: DemoMap | None = None) -> MapV3:
    """The reconstruction as a ``MapV3``, in the same map frame."""
    m = m or build_demo_map()
    if abs(THICKNESS - THICK) > 1e-12:
        raise ValueError(
            f"demo map thickness {THICKNESS} != lane_graph.maps_v3.THICK {THICK}; "
            "MapV3 extrudes walls with its own module constant, so the two must "
            "agree or the built slabs will not match the metric spec")

    walls = [Wall(*w.centreline) for w in m.walls]
    # scanned pillars are cylinders exactly like the curved features, so they
    # ride the bulge channel; V3 sees one vocabulary either way.
    bulges = ([Bulge(c.center, c.radius, c.cid) for c in m.curves]
              + [Bulge(p.center, p.radius, p.pid)
                 for p in getattr(m, "pillars", ())])
    # LANDMARKS ARE CORNERS AND CYLINDERS.  Owner's directive: every other
    # non-wall body goes -- the ones on the route "去掉变成墙", so what the
    # hand meets there is the flat wall behind them, and the ones off it
    # deleted outright.  Both are the same edit, because walls are their own
    # bodies: dropping a protrusion cannot open a hole in one.
    #
    # `prots` is the channel that carried the scan's boxes and hulls -- desks,
    # door leaves, clutter.  It was 33 of the demo room's 36 non-wall bodies
    # (the ATE visited 5) and 2 of room2's 3 (the ATE visited neither).  What
    # survives is `bulges`: the curved wall features and the scanned pillars,
    # which are the cylinders, plus the convex corner, which is wall geometry
    # and was never a body.
    #
    # The construction is kept behind the flag rather than deleted, because a
    # rescan re-derives boxes and hulls whether or not this planner wants
    # them, and code that is deleted cannot be measured against.
    prots = ([Prot(b.center, b.half[0], b.half[1], b.u, b.bid)
              for b in m.boxes]
             + [Prot(h.center, h.half[0], h.half[1], h.u, h.hid)
                for h in getattr(m, "hulls", ())]) \
        if (BOX_LANDMARKS or getattr(m, "box_landmarks", False)) else []

    # Landmarks come from the map's own primitives so this works for the tape
    # map and the scan-derived map alike; the named corners are added only when
    # the map carries those key points.
    landmarks = [(c.center[0], c.center[1] - c.radius, f"{c.cid} apex")
                 for c in m.curves]
    kp = m.key_points
    if {"x_corner", "x_corr_e", "y_upper_s"} <= set(kp):
        landmarks += [(kp["x_corner"], 0.0, "long wall ends (corridor mouth)"),
                      (kp["x_corr_e"], kp["y_upper_s"], "convex corner")]
    elif {"branch_w_x", "branch_e_x", "branch_end_y"} <= set(kp):
        landmarks += [(kp["branch_w_x"], kp.get("long_wall_y", 0.0),
                       "long wall ends (branch mouth)"),
                      (kp["branch_e_x"], kp["branch_end_y"], "convex corner")]
    doors = [(o.mid[0], o.mid[1], o.oid) for o in m.openings]

    return MapV3(
        name=MAP_NAME, title=MAP_TITLE, walls=walls, bulges=bulges, prots=prots,
        rooms=list(m.rooms), open_boundaries=list(m.open_boundaries),
        start=tuple(m.start.center), goal=tuple(m.goal.center),
        goal_r=m.goal.radius, doors=doors, landmarks=landmarks,
        topology="lower corridor -> vertical corridor -> upper room",
        story="Follow the long wall past two curved features, cross the corridor "
              "mouth to the opposite wall, run up it, round the convex corner "
              "into the upper room and stop at the doorway.",
        notes=[f"wall height {HEIGHT} m (not maps_v3's default 1.5 m)",
               "curved features stand 0.29 m and 0.47 m proud -- far outside the "
               "<=0.15 m V3 bulge envelope, so a followed lane must bow",
               "not registered in scene3.PRIORS: the belief planner is out of "
               "scope for this map"])


def demo_map_obstacles(m: DemoMap | None = None):
    """MuJoCo obstacle descriptors for ``mj_assets.inject_into_env_cfg``, at the
    demo map's 1.8 m wall height rather than the shared 1.5 m default."""
    from lane_graph.mj_assets import build_obstacles_v3

    m = m or build_demo_map()
    return build_obstacles_v3(demo_map_v3(m), half_height=0.5 * HEIGHT)
