"""Demo geometry and initial conditions, independent of navigation policies.

Coordinates use the existing start-at-origin map frame. Initial headings are
scene presets; turn decisions and sensing locations belong to the planner.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class DemoSceneSetup:
    room: str
    scene: object
    face_yaw_map: float
    standoff_m: float

    def describe(self) -> dict:
        return dict(room=self.room, face_yaw_map=self.face_yaw_map,
                    standoff_m=self.standoff_m,
                    start_map_xy=list(self.scene.start_map_xy),
                    goal_map_xy=list(self.scene.goal_map_xy),
                    goal_radius_m=self.scene.goal_radius_m)


def _room2(standoff_m):
    from demo_map.scene_room2 import build_room2_scene
    return build_room2_scene(False), math.pi


def _demo_room(standoff_m):
    from demo_map.scene_demo import build_demo_scene
    return build_demo_scene(), math.pi / 2


def build_room830_scene(standoff_m=0.40, *,
                        walls_json="docs/demo_maps/room_8_30_2026_curated.json",
                        island_half_height_m=0.65, start_in_from_end_m=0.45):
    """Preserve the island scene and spawn; return scene, island, north normal.

    The legacy loop policy also uses the island geometry to construct its
    feature sequence. Scene loading itself never constructs that sequence.
    """
    from demo_map.geometry import Region
    from demo_map.scan_map import build_scan_map
    from demo_map.scene_demo import build_demo_scene, translated_scan_map

    m = build_scan_map(walls_json)
    boxes = list(m.boxes)

    def touches(a, b):
        d = np.asarray(a.center, float) - np.asarray(b.center, float)
        return (abs(d[0]) < a.half[0] + b.half[0] + 0.05
                and abs(d[1]) < a.half[1] + b.half[1] + 0.05)

    free = [b for b in boxes
            if not any(touches(b, o) for o in boxes if o is not b)]
    island = max(free, key=lambda b: b.half[0] * b.half[1])
    c = np.asarray(island.center, float)
    u = np.asarray(island.u, float) / np.linalg.norm(island.u)
    n = np.array([-u[1], u[0]])
    if n[1] < 0:
        n, u = -n, -u
    ha, ho = float(island.half[0]), float(island.half[1])
    start_scan = c + (ha - start_in_from_end_m) * u + (ho + standoff_m) * n
    m = replace(m, start=Region(
        "START", tuple(float(v) for v in start_scan), 0.30,
        float(math.atan2(-n[1], -n[0])), note="beside the island's north face"))
    scene = build_demo_scene(translated_scan_map(m))
    prots = []
    for pr in scene.spec.prots:
        if abs(pr.half_along - ha) < 1e-6 and abs(pr.half_out - ho) < 1e-6:
            pr = replace(pr, half_height=island_half_height_m)
            island = pr
        prots.append(pr)
    scene = replace(scene, spec=replace(scene.spec, prots=prots))
    return scene, island, n


def _room830(standoff_m):
    scene, _, n = build_room830_scene(standoff_m)
    return scene, float(math.atan2(-n[1], -n[0]))


def _room830_modified(standoff_m):
    """The approved proposal, translated once into the spawn-at-origin frame.

    Keep simulator body order (walls, cylinders, boxes) identical to the
    analytic floorplan so recorded contact body indices refer to the same wall.
    """
    import json
    from global_nav.geometry import ConvexPolygon
    from lane_graph.maps_v3 import THICK, Wall, followable

    root = Path(__file__).resolve().parents[2]
    doc = json.loads((root / "maps/room830_modified.json").read_text())
    # The authored modification is expressed in the original 0.40 m spawn
    # frame. A planner standoff change must not translate these map bodies.
    scene, _, n = build_room830_scene(0.40, walls_json=str(root / doc["base_map"]))
    yaw = float(math.atan2(-n[1], -n[0]))
    start = np.asarray(doc["start_xy"], float)
    def pt(p):
        return tuple(np.asarray(p, float) - start)
    north = float(next(b for b in scene.plan.bodies
                       if b.name == doc["north_alignment_body"]).verts[:, 1].max())
    x0, x1 = doc["north_aligned_wall_x"]
    added = [Wall(*map(tuple, doc["vertical_wall"])),
             Wall((x0, north - THICK / 2), (x1, north - THICK / 2))]
    original = scene.spec
    count = len(original.walls)
    walls = [replace(w, p0=pt(w.p0), p1=pt(w.p1)) for w in [*original.walls, *added]]
    spec = replace(original, name=doc["name"], title=doc["name"], walls=walls,
                   bulges=[replace(b, center=pt(b.center)) for b in original.bulges],
                   prots=[replace(p, center=pt(p.center)) for p in original.prots],
                   rooms=[(*pt((a, b)), *pt((c, d))) for a, b, c, d in original.rooms],
                   doors=[(*pt((x, y)), label) for x, y, label in original.doors],
                   landmarks=[(*pt((x, y)), label) for x, y, label in original.landmarks],
                   open_boundaries=[(nx, ny, c - np.dot((nx, ny), start))
                                    for nx, ny, c in original.open_boundaries],
                   start=(0., 0.), goal=pt(doc["goal_xy"]))
    old = [ConvexPolygon(np.asarray(b.verts) - start, b.name, b.signature)
           for b in scene.plan.bodies]
    new = [ConvexPolygon(w.poly(), f"wall_{count+i}", "surface")
           for i, w in enumerate(walls[count:])]
    plan = replace(scene.plan, name=spec.name, bodies=tuple(old[:count] + new + old[count:]),
                   features=tuple(replace(f, xy=pt(f.xy)) for f in scene.plan.features),
                   edges=tuple(replace(e, a_xy=pt(e.a_xy), b_xy=pt(e.b_xy))
                               for e in scene.plan.edges))
    return replace(scene, name=spec.name, spec=spec, plan=plan, lanes=followable(spec),
                   start_map_xy=spec.start, goal_map_xy=spec.goal,
                   prior_components=(((0., 0.), yaw, 0, 1.),)), yaw


DEMO_ROOMS = {"room2": _room2, "room2_turn": _room2,
              "demo_room": _demo_room, "room830": _room830,
              "room830_modified": _room830_modified}

REFERENCE_MAPS = {name: Path(__file__).resolve().parents[2] / "maps" / f"{name}.json"
                  for name in ("ref1_bend", "ref2_hall", "ref3_zsuite", "ref4_room",
                               "sky_offset_gate", "sky_staggered_baffles", "sky_island_choice")}


def _reference(path, standoff_m):
    from demo_map.manifest import build_manifest_scene
    # The manifest loader rotates its authored start heading onto map +y.
    return build_manifest_scene(path), math.pi / 2


DEMO_ROOMS.update({name: partial(_reference, path) for name, path in REFERENCE_MAPS.items()})


def load_demo_scene(room: str, standoff_m: float = 0.40) -> DemoSceneSetup:
    scene, yaw = DEMO_ROOMS[room](float(standoff_m))
    return DemoSceneSetup(room, scene, yaw, float(standoff_m))
