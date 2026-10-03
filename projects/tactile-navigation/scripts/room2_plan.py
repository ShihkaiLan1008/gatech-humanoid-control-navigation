"""Room2 (8_11_2026.glb) planned by the V3 planner, BOTH directions.

    PYTHONPATH=src python3 scripts/room2_plan.py --out out/demo_map

This is the real-robot test room, so the routes come from the actual stack:
``build_scan_map`` -> ``demo_map_v3`` -> ``followable`` lanes ->
``build_graph3`` + ``astar3``.  Nothing in the planner is touched; this
script only builds the scene the planner already consumes -- exactly the
``build_demo_scene`` pattern, minus the demo room's manual-route baggage.

Two scenes are planned, start->goal and goal->start, each translated so ITS
OWN start sits at (0, 0) -- the MujocoBackendV2 frame contract -- and drawn
back in the room2 map frame on one figure.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from demo_map.mapv3 import demo_map_v3
from demo_map.scan_map import build_scan_map
from demo_map.scene_demo import TURN_PRESS_LANE_OFF, translated_scan_map
from global_nav.floorplan import Floorplan, MapFeature, RouteEdge
from global_nav.geometry import ConvexPolygon
from lane_graph.graph3 import Belief3, astar3, build_graph3
from lane_graph.maps_v3 import followable
from lane_graph.scene3 import V3Scene, _ngon

sys.path.insert(0, "scripts")

WALLS = Path("docs/demo_maps/room2_scan_walls.json")


def room2_scene(m, name: str) -> V3Scene:
    """``build_demo_scene`` for a self-framed room: same contract, no demo
    anchors, unimodal prior at the spawn (this room carries no registration
    ambiguity story -- the robot starts where it is put)."""
    mt = translated_scan_map(m)
    spec = demo_map_v3(mt)
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
    plan = Floorplan(tuple(bodies), feats, edges, route=(0,), name=name)
    return V3Scene(
        name=name, spec=spec, plan=plan, lanes=followable(spec),
        start_map_xy=(0.0, 0.0), goal_map_xy=tuple(spec.goal),
        goal_radius_m=spec.goal_r, standoff_m=TURN_PRESS_LANE_OFF,
        prior_components=(((0.0, 0.0), np.pi / 2.0, 0, 1.0),),
        prior_sigma_along_m=0.14, prior_sigma_lateral_m=0.06)


def plan(scene):
    """Graph, A*, edge polylines -- the planner's own route, nothing else."""
    g = build_graph3(scene)
    src = next(n.nid for n in g.nodes if n.kind == "start")
    dst = next(n.nid for n in g.nodes if n.kind == "goal")
    eids, cost = astar3(g, src, dst,
                        Belief3(sigma=scene.prior_sigma_along_m, minor=0.0))
    if eids is None:
        return None, None, np.inf
    pts, kinds, nid = [], [], src
    for eid in eids:
        e = g.edges[eid]
        poly = (e.poly if e.poly is not None else
                np.array([g.nodes[e.a].xy, g.nodes[e.b].xy]))
        if e.b == nid:
            poly = poly[::-1]
        nid = e.b if e.a == nid else e.a
        pts.append(np.asarray(poly, float))
        kinds.append(e.kind)
    return pts, kinds, cost


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("out/demo_map"))
    a = ap.parse_args(argv)

    m = build_scan_map(WALLS)
    fwd = m
    rev = dataclasses.replace(
        m,
        start=dataclasses.replace(m.goal, rid="START", radius=m.start.radius,
                                  note="reverse leg: the forward goal"),
        goal=dataclasses.replace(m.start, rid="GOAL", radius=m.goal.radius,
                                 note="reverse leg: the forward start"))

    fig, ax = plt.subplots(figsize=(11, 11))
    doc = {}
    colors = {"start->goal": "#1f77b4", "goal->start": "#d95f02"}
    for label, mm in (("start->goal", fwd), ("goal->start", rev)):
        scene = room2_scene(mm, f"room2_{label}")
        pts, kinds, cost = plan(scene)
        d = np.asarray(mm.start.center, float)   # scene frame -> map frame
        if pts is None:
            print(f"{label}: NO ROUTE")
            doc[label] = None
            continue
        L = float(sum(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1))
                      for p in pts))
        n_follow = sum(k == "TOUCH_FOLLOW" for k in kinds)
        follow_len = float(sum(
            np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1))
            for p, k in zip(pts, kinds) if k == "TOUCH_FOLLOW"))
        print(f"{label}: {L:.2f} m, cost {cost:.1f}, "
              f"{len(kinds)} edges ({n_follow} TOUCH_FOLLOW, "
              f"{follow_len:.2f} m followed = {100 * follow_len / L:.0f}%)")
        print(f"  kinds: {kinds}")
        doc[label] = dict(
            length_m=round(L, 3), cost=round(float(cost), 2),
            followed_m=round(follow_len, 3),
            start_map=[round(v, 3) for v in mm.start.center],
            goal_map=[round(v, 3) for v in mm.goal.center],
            edges=[dict(kind=k,
                        poly=[[round(float(x + d[0]), 3),
                               round(float(y + d[1]), 3)] for x, y in p])
                   for p, k in zip(pts, kinds)])
        for p, k in zip(pts, kinds):
            q = p + d
            ax.plot(q[:, 0], q[:, 1], color=colors[label], lw=2.6,
                    ls="-" if k == "TOUCH_FOLLOW" else (0, (3, 2)),
                    alpha=0.95, zorder=6)
        ax.plot([], [], color=colors[label], lw=2.6,
                label=f"{label}: {L:.1f} m, {100 * follow_len / L:.0f}% "
                      "wall-followed (solid=TOUCH_FOLLOW, dashed=transit)")

    # the room, in its own frame
    from demo_map.plots import _box_patches, _wall_patches
    _wall_patches(ax, m, ids=True, fs=7)
    _box_patches(ax, m, ids=False)
    s, g2 = np.asarray(m.start.center), np.asarray(m.goal.center)
    ax.plot(*s, "o", ms=13, color="#2ca02c", zorder=7)
    ax.annotate("START", s, textcoords="offset points", xytext=(-14, -18),
                color="#1a7a1a", fontweight="bold")
    ax.plot(*g2, "o", ms=13, color="#d62728", zorder=7)
    ax.annotate("GOAL", g2, textcoords="offset points", xytext=(8, 8),
                color="#a11", fontweight="bold")
    ax.annotate("corner (0,0)", (0, 0), textcoords="offset points",
                xytext=(10, 12), fontsize=9)
    ax.plot(0, 0, "k+", ms=12, zorder=7)
    ax.set_aspect("equal")
    ax.grid(alpha=0.25)
    ax.set_xlabel("x [m, room2 map frame]")
    ax.set_ylabel("y [m]")
    ax.set_title("room2 (8_11 scan) -- V3 planner routes, both directions "
                 "(planner untouched)")
    ax.legend(loc="lower right", fontsize=9)
    a.out.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(a.out / "room2_planned_routes.png", dpi=130)
    (a.out / "room2_planned_routes.json").write_text(json.dumps(doc, indent=1))
    print(f"wrote {a.out}/room2_planned_routes.png, "
          f"{a.out}/room2_planned_routes.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
