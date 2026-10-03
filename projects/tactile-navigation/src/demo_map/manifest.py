"""Load map_gen JSON geometry for the current planner and MuJoCo controllers.

The geometry conversion is adapted from scripts/map_gen/polymap.py on
jjang318/global (99fffc4). No sampling or navigation policy is included.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from global_nav.floorplan import Floorplan, MapFeature, RouteEdge
from global_nav.geometry import ConvexPolygon
from lane_graph.maps_v3 import THICK, Bulge, MapV3, Prot, Wall, followable
from lane_graph.mj_assets import OBST_HEIGHT_HALF
from lane_graph.scene3 import STANDOFF, V3Scene, _ngon

FORMAT = "tactile-navigation/map_gen v1"


def room_rotation(doc):
    """Rotate the authored starting heading onto map +y, as map_gen does."""
    quadrant = float(doc.get("start_heading_deg", 90.0)) / 90.0
    if abs(quadrant - round(quadrant)) > 1e-6:
        raise ValueError("start_heading_deg must be a multiple of 90: free-space rectangles are axis-aligned")
    angle = math.pi / 2 - math.radians(90.0 * (int(round(quadrant)) % 4))
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s], [s, c]])


def room_to_map(doc, points):
    return (np.asarray(points, float) - np.asarray(doc["start_xy"], float)) @ room_rotation(doc).T


def _room_centrelines(vertices, half_thickness):
    """Offset the inside outline outward; intersect adjacent wall lines."""
    V = np.asarray(vertices, float)
    if len(V) < 3:
        raise ValueError("a room outline needs at least three corners")
    directions, starts = [], []
    for k in range(len(V)):
        d = V[(k + 1) % len(V)] - V[k]
        length = float(np.linalg.norm(d))
        if length < 1e-9:
            raise ValueError(f"room outline corner {k} repeats its neighbour")
        u = d / length
        directions.append(u)
        starts.append(V[k] + half_thickness * np.array([u[1], -u[0]]))
    corners = []
    for k in range(len(V)):
        a0, u0 = starts[k - 1], directions[k - 1]
        a1, u1 = starts[k], directions[k]
        den = float(u0[0] * (-u1[1]) - u0[1] * (-u1[0]))
        if abs(den) < 1e-9:
            corners.append(a1.copy())
        else:
            rhs = a1 - a0
            t = float(rhs[0] * (-u1[1]) - rhs[1] * (-u1[0])) / den
            corners.append(a0 + t * u0)
    return np.asarray(corners)


def _polygon_ring(vertices, half_thickness):
    """Inset a convex outer footprint to the centrelines of its wall slabs."""
    V = np.asarray(vertices, float)
    edges = np.roll(V, -1, axis=0) - V
    lengths = np.linalg.norm(edges, axis=1)
    if len(V) < 3 or float(lengths.min()) < 1e-9:
        raise ValueError("polygon obstacle needs three distinct corners")
    U = edges / lengths[:, None]
    N = np.stack([U[:, 1], -U[:, 0]], axis=1)
    C = np.sum(N * V, axis=1) - half_thickness
    out = []
    for i in range(len(V)):
        j = (i - 1) % len(V)
        A = np.array([N[j], N[i]])
        if abs(float(np.linalg.det(A))) < 1e-9:
            raise ValueError("polygon obstacle has collinear adjacent edges")
        out.append(np.linalg.solve(A, np.array([C[j], C[i]])))
    P = np.asarray(out)
    if float(np.max(P @ N.T - C)) > 1e-7:
        raise ValueError("polygon obstacle collapses at the configured wall thickness")
    return P


def build_manifest_scene(manifest: str | Path | dict) -> V3Scene:
    doc = json.loads(Path(manifest).read_text()) if isinstance(manifest, (str, Path)) else manifest
    if doc.get("format") != FORMAT:
        raise ValueError(f"expected map format {FORMAT!r}")
    if abs(float(doc.get("wall_thickness_m", THICK)) - THICK) > 1e-12:
        raise ValueError(f"wall_thickness_m must match the simulator's {THICK} m walls")
    if abs(float(doc.get("body_height_m", 2 * OBST_HEIGHT_HALF)) - 2 * OBST_HEIGHT_HALF) > 1e-12:
        raise ValueError(f"body_height_m must match the simulator's {2 * OBST_HEIGHT_HALF} m bodies")
    R = room_rotation(doc)
    W, D = float(doc["room"]["width_m"]), float(doc["room"]["depth_m"])
    h = THICK / 2
    walls, bulges, prots, wall_names, landmarks = [], [], [], [], []

    def wall(a, b, name):
        q = room_to_map(doc, [a, b])
        walls.append(Wall(tuple(q[0]), tuple(q[1])))
        wall_names.append(name)

    if doc.get("outline"):
        C = _room_centrelines(doc["outline"], h)
        for k in range(len(C)):
            wall(C[k], C[(k + 1) % len(C)], f"wall_e{k}")
    else:
        for a, b, name in [((-h, -h), (W + h, -h), "south"),
                           ((-h, D + h), (W + h, D + h), "north"),
                           ((-h, -h), (-h, D + h), "west"),
                           ((W + h, -h), (W + h, D + h), "east")]:
            wall(a, b, f"wall_{name}")

    for obstacle in doc["obstacles"]:
        name, kind = obstacle["id"], obstacle.get("kind", "polygon")
        if kind == "cylinder":
            bulges.append(Bulge(tuple(room_to_map(doc, obstacle["center"])),
                                float(obstacle["radius_m"]), name))
        elif kind == "box":
            yaw = math.radians(float(obstacle.get("yaw_deg", 0.0)))
            u = np.array([math.cos(yaw), math.sin(yaw)])
            n = np.array([-u[1], u[0]])
            c = np.asarray(obstacle["center"], float)
            a, b = float(obstacle["half_along_m"]), float(obstacle["half_out_m"])
            prots.append(Prot(tuple(room_to_map(doc, c)), a, b, tuple(u @ R.T), name))
            corners = room_to_map(doc, [c - a*u - b*n, c + a*u - b*n,
                                       c + a*u + b*n, c - a*u + b*n])
            landmarks.extend((float(p[0]), float(p[1]), f"{name} corner {k}")
                             for k, p in enumerate(corners))
        elif kind == "wall":
            pts = obstacle["pts"] if "pts" in obstacle else [obstacle["p0"], obstacle["p1"]]
            for k, (a, b) in enumerate(zip(pts[:-1], pts[1:])):
                wall(a, b, name if len(pts) == 2 else f"{name}_s{k}")
            for k, p in enumerate(room_to_map(doc, pts)):
                tag = "bend" if 0 < k < len(pts) - 1 else "end"
                landmarks.append((float(p[0]), float(p[1]), f"{name} {tag} {k}"))
        elif kind == "polygon":
            ring = _polygon_ring(obstacle["verts"], h)
            for k in range(len(ring)):
                wall(ring[k], ring[(k + 1) % len(ring)], f"{name}_f{k}")
            landmarks.extend((float(p[0]), float(p[1]), f"{name} corner {k}")
                             for k, p in enumerate(room_to_map(doc, obstacle["verts"])))
        else:
            raise ValueError(f"{name}: unknown obstacle kind {kind!r}")

    rooms = []
    for x0, y0, x1, y1 in doc.get("free_space_rects", [(0., 0., W, D)]):
        p = room_to_map(doc, [(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
        rooms.append((float(p[:, 0].min()), float(p[:, 1].min()),
                      float(p[:, 0].max()), float(p[:, 1].max())))
    goal = tuple(room_to_map(doc, doc["goal_xy"]))
    m = MapV3(name=doc["name"], title=doc["name"], walls=walls, bulges=bulges,
              prots=prots, rooms=rooms, start=(0., 0.), goal=goal,
              goal_r=float(doc.get("goal_radius_m", .30)), doors=[], landmarks=landmarks,
              topology="manifest", story="Authored map loaded from JSON.")
    # Body ordering is shared with build_obstacles_v3 and its contact sensors.
    bodies = tuple([ConvexPolygon(w.poly(), name, "surface") for w, name in zip(walls, wall_names)]
                   + [_ngon(b.center, b.radius, name=b.label) for b in bulges]
                   + [ConvexPolygon(p.poly(), p.label, "surface") for p in prots])
    features = tuple(MapFeature(i, (x, y), "landmark", label)
                     for i, (x, y, label) in enumerate(landmarks))
    vertices = np.concatenate([b.verts for b in bodies])
    lo, hi = vertices.min(axis=0) - 1., vertices.max(axis=0) + 1.
    edges = (RouteEdge(0, tuple(lo), (hi[0], lo[1]), half_width_m=hi[1] - lo[1],
                       successors=(), name="freespace"),)
    plan = Floorplan(bodies, features, edges, route=(0,), name=m.name)
    return V3Scene(name=m.name, spec=m, plan=plan, lanes=followable(m),
                   start_map_xy=(0., 0.), goal_map_xy=goal, goal_radius_m=m.goal_r,
                   standoff_m=STANDOFF, prior_components=(((0., 0.), math.pi/2, 0, 1.),),
                   prior_sigma_along_m=.10, prior_sigma_lateral_m=.06)
