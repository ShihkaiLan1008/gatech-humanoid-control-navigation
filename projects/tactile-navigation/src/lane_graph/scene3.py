"""MapV3 -> the scene contract the belief, graph and backends consume.

Bodies carry signature "surface" without exception (the V1 association
lesson: the map must not partition bodies into classes the touch channel
cannot distinguish).  Bulges enter the analytic map as 20-gons -- max radial
error 0.6 cm at r=0.80, well inside the 3.5 cm contact sigma -- while the
physical build uses a true cylinder.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from global_nav.floorplan import Floorplan, MapFeature, RouteEdge
from global_nav.geometry import ConvexPolygon

from .maps_v3 import (MapV3, followable, r1_gallery, r2_lbend, r3_tpinch,
                      r4_winding, r5_twinbay)

MAPS3 = {f.__name__: f for f in (r1_gallery, r2_lbend, r3_tpinch,
                                 r4_winding, r5_twinbay)}

STANDOFF = 0.38

# Registration ambiguity per map: the direction is the FIRST followed wall's
# axis, because that is the coordinate a flat touch cannot pin (Batch 1's
# rank-1 result).  The heavier mode is always the WRONG one, so the prior
# alone cannot pick the route.
PRIORS = {
    "r1_gallery":  dict(dir=(1.0, 0.0), modes=((0.0, 0.45), (0.55, 0.55))),
    # R2 is UNIMODAL by necessity, not preference: its route turns, so some
    # followed wall always has its normal ALONG the ambiguity axis, and there
    # the worst-case stance sits up to a full separation beyond the 0.43 m
    # reach -- the press can never land, so the ambiguity can never resolve
    # (measured: 12 increments, 0 contacts, peak 0.0 N).  R3 keeps the bimodal
    # branch story because its ambiguity runs parallel to every wall it
    # follows.  A prior spread of 0.14 m still exceeds the 0.12 m commit gate,
    # so touch is still what unlocks the turns.
    "r2_lbend":    dict(dir=(0.0, 1.0), modes=((0.0, 1.0),)),
    "r3_tpinch":   dict(dir=(0.0, 1.0), modes=((0.0, 0.45), (0.55, 0.55))),
    "r4_winding":  dict(dir=(0.0, 1.0), modes=((0.0, 0.45), (0.55, 0.55))),
    "r5_twinbay":  dict(dir=(1.0, 0.0), modes=((0.0, 0.45), (3.40, 0.55))),
}


def _ngon(c, r, n=20, name="bulge") -> ConvexPolygon:
    a = np.linspace(0.0, 2 * math.pi, n, endpoint=False)
    return ConvexPolygon(np.c_[c[0] + r * np.cos(a), c[1] + r * np.sin(a)],
                         name, "surface")


@dataclass
class V3Scene:
    name: str
    spec: MapV3
    plan: Floorplan
    lanes: list
    start_map_xy: tuple
    goal_map_xy: tuple
    goal_radius_m: float
    standoff_m: float
    prior_components: tuple
    prior_sigma_along_m: float
    prior_sigma_lateral_m: float
    true_delta_m: float = 0.0
    # shims the V1 MuJoCo backend ctor touches
    walls: tuple = ()
    stations: dict = None
    legs: tuple = ()
    fan_offsets_m: tuple = (-0.24, -0.12, 0.0, 0.12, 0.24)

    def __post_init__(self):
        if self.stations is None:
            self.stations = {}
        if not self.legs:
            self.legs = ({"origin": np.zeros(2), "u": np.array([1.0, 0.0]),
                          "v": np.array([0.0, 1.0]), "near": 0.0,
                          "terminus": 1.0, "index": 0},)


def build_v3_scene(name: str) -> V3Scene:
    spec = MAPS3[name]()
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

    pr = PRIORS[name]
    d = np.asarray(pr["dir"], float)
    comps = tuple(((float(d[0] * delta), float(d[1] * delta)), math.pi / 2.0,
                   0, float(mass)) for delta, mass in pr["modes"])
    return V3Scene(
        name=name, spec=spec, plan=plan, lanes=followable(spec),
        start_map_xy=tuple(spec.start), goal_map_xy=tuple(spec.goal),
        goal_radius_m=spec.goal_r, standoff_m=STANDOFF,
        prior_components=comps,
        prior_sigma_along_m=(0.14 if len(pr["modes"]) == 1 else 0.10),
        prior_sigma_lateral_m=0.06)
