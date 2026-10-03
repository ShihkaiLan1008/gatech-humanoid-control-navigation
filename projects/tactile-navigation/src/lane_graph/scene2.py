"""MapSpec -> the belief/backend world: Floorplan bodies, registration
priors, and the minimal scene contract the backends read (``plan``,
``true_delta_m``) plus what the V2 executive needs (start/goal, standoff).

Pillars enter the ANALYTIC map as regular 16-gons (max radial error at
r=0.40 is ~0.8 cm, inside the 3.5 cm contact sigma); the physical build
uses a true cylinder.  Every body carries signature "surface" -- the V1
association lesson: the map must not partition bodies into classes the
touch channel cannot distinguish.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from global_nav.floorplan import Floorplan, MapFeature, RouteEdge
from global_nav.geometry import ConvexPolygon

from .maps import MAPS, MapSpec

STANDOFF = 0.38
PRIORS = {
    # name: (modes ((delta, mass), ...), sigma_along, sigma_lat, true_delta)
    "m1_legacy_perimeter": (((0.0, 0.45), (0.55, 0.55)), 0.10, 0.06, 0.0),
    "m2_pillar_room":      (((0.0, 0.45), (0.55, 0.55)), 0.10, 0.06, 0.0),
    "m3_l_alcove":         (((0.0, 0.45), (0.55, 0.55)), 0.10, 0.06, 0.0),
    "m4_twin_alcoves":     (((0.0, 0.45), (2.80, 0.55)), 0.12, 0.06, 0.0),
    "m5_t_junction":       (((0.0, 0.45), (0.55, 0.55)), 0.10, 0.06, 0.0),
}


def _pillar_poly(center, radius, name) -> ConvexPolygon:
    ang = np.linspace(0.0, 2 * math.pi, 16, endpoint=False)
    verts = np.c_[center[0] + radius * np.cos(ang),
                  center[1] + radius * np.sin(ang)]
    return ConvexPolygon(verts, name, "surface")


@dataclass
class V2Scene:
    """The scene contract: what backends + belief + executive2 consume."""
    name: str
    spec: MapSpec
    plan: Floorplan
    start_map_xy: tuple[float, float]
    goal_map_xy: tuple[float, float]
    goal_radius_m: float
    standoff_m: float
    prior_modes: tuple
    prior_sigma_along_m: float
    prior_sigma_lateral_m: float
    true_delta_m: float
    # legacy attrs the V1 backend ctor touches: walls=() no-ops the maze
    # placement block; the one-leg shim satisfies _set_probe_side(0) until
    # the executive states real press frames via set_press_frame
    walls: tuple = ()
    stations: dict = None
    legs: tuple = ()
    fan_offsets_m: tuple = (0.06, 0.12, 0.18, 0.24, 0.30)

    def __post_init__(self):
        if self.stations is None:
            self.stations = {}
        if not self.legs:
            self.legs = ({"origin": np.zeros(2), "u": np.array([1.0, 0.0]),
                          "v": np.array([0.0, 1.0]), "near": 0.0,
                          "terminus": 1.0, "index": 0},)


def build_v2_scene(name: str, seed: int = 0) -> V2Scene:
    spec = MAPS[name](seed)
    bodies = []
    for i, w in enumerate(spec.walls):
        bodies.append(ConvexPolygon(w.poly(), f"wall_{i}", "surface"))
    for i, p in enumerate(spec.pillars):
        bodies.append(_pillar_poly(p.center, p.radius, f"pillar_{i}"))
    for i, bx in enumerate(spec.boxes):
        bodies.append(ConvexPolygon(bx.poly(), f"cabinet_{i}", "surface"))
    feats = tuple(MapFeature(i, (float(x), float(y)), "landmark", lab)
                  for i, (x, y, lab) in enumerate(spec.landmarks))
    # ONE whole-map route edge: V2 navigates a graph, not a corridor chain,
    # so the belief's edge component is deliberately inert (documented MVP
    # simplification -- pose/registration/association carry the work).
    xs = [v[0] for b in bodies for v in np.asarray(b.verts)]
    ys = [v[1] for b in bodies for v in np.asarray(b.verts)]
    lo = (min(xs) - 1.0, min(ys) - 1.0)
    hi = (max(xs) + 1.0, max(ys) + 1.0)
    edges = (RouteEdge(0, lo, (hi[0], lo[1]),
                       half_width_m=(hi[1] - lo[1]), successors=(),
                       name="freespace"),)
    plan = Floorplan(tuple(bodies), feats, edges, route=(0,), name=name)
    modes, s_along, s_lat, true_d = PRIORS[name]
    return V2Scene(
        name=name, spec=spec, plan=plan,
        start_map_xy=tuple(spec.start), goal_map_xy=tuple(spec.goal),
        goal_radius_m=spec.goal_r, standoff_m=STANDOFF,
        prior_modes=modes, prior_sigma_along_m=s_along,
        prior_sigma_lateral_m=s_lat, true_delta_m=true_d)
