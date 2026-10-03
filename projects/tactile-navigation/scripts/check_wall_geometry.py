"""Two static checks before any further wall-follow work. No controller, no MuJoCo.

    PYTHONPATH=src python3 scripts/check_wall_geometry.py --scene r5_twinbay

CHECK 1 -- the wall-follow heading must come from the WALL SURFACE tangent, not
from the lane polyline's local tangent.  V3 lanes bow around features, so the
polyline tangent is not even perpendicular to the lane's own normal; the failed
static gate commanded a heading 25.8 deg off the wall (docs/HW_ROBUSTNESS.md 9).

CHECK 2 -- the belief's `Floorplan` polygon and the injected MuJoCo entity must
describe the SAME plane.  `place_obstacles` writes a mocap pose per obstacle
with its own yaw convention (`th + pi/2` for wall-like boxes, thickness along
the entity's local +x); if that disagrees with `Wall.poly()` the mapped wall
distance is not the physical one, which is the outstanding candidate for the
0.077 m residual overlap the gate could not explain.

Everything here is derived arithmetic on the map spec and the placement code,
so it runs offline and deterministically.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from global_nav.geometry import raycast                  # noqa: E402
from lane_graph.graph3 import build_graph3               # noqa: E402
from lane_graph.mj_assets import (OBST_HEIGHT_HALF, WALL_THICK_HALF,  # noqa: E402
                                  build_obstacles_v3)
from lane_graph.maps_v3 import THICK                     # noqa: E402
from lane_graph.scene3 import build_v3_scene             # noqa: E402


def wall_tangent_at(plan, origin, n_probe):
    """The tangent of the SURFACE a press ray hits -- the correct heading source.

    Returns (tangent, normal, range) in the same frame as `origin`, or None.
    """
    d, nn, idx = raycast(np.asarray(origin, float), np.asarray(n_probe, float),
                         2.0, plan.bodies)
    if d is None:
        return None
    nn = np.asarray(nn, float)
    return np.array([-nn[1], nn[0]]), nn, float(d), int(idx)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="r5_twinbay")
    ap.add_argument("--base0", type=float, nargs=2, default=[0.0, 0.0])
    ap.add_argument("--yaw0", type=float, default=0.0,
                    help="spawn heading; map +y is the start heading")
    a = ap.parse_args(argv)

    scene = build_v3_scene(a.scene)
    graph = build_graph3(scene)
    plan = scene.plan

    # ── CHECK 1 ─────────────────────────────────────────────────────────────
    print(f"# CHECK 1 -- heading source, {a.scene}\n")
    print(f"{'edge':>5s} {'lane-tangent yaw':>17s} {'WALL-tangent yaw':>17s}"
          f" {'error':>8s} {'u.n_lane':>9s}")
    worst = 0.0
    for e in graph.edges:
        if e.kind != "TOUCH_FOLLOW" or e.poly is None:
            continue
        poly, nrm = np.asarray(e.poly, float), np.asarray(e.normal, float)
        i = len(poly) // 2
        u_lane = poly[min(len(poly) - 1, i + 1)] - poly[max(0, i - 1)]
        u_lane = u_lane / max(float(np.linalg.norm(u_lane)), 1e-9)
        got = wall_tangent_at(plan, poly[i], nrm[i])
        if got is None:
            continue
        t_wall, n_surf, rng, _idx = got
        ang = math.degrees(math.acos(min(1.0, abs(float(u_lane @ t_wall)))))
        worst = max(worst, ang)
        print(f"{e.eid:5d} {math.degrees(math.atan2(u_lane[1], u_lane[0])):16.1f}d"
              f" {math.degrees(math.atan2(t_wall[1], t_wall[0])):16.1f}d"
              f" {ang:7.1f}d {float(u_lane @ nrm[i]):9.4f}")
    print(f"\nworst lane-vs-wall heading error: {worst:.1f} deg")
    print("=> the wall tangent must come from the SURFACE NORMAL of the body the"
          "\n   press ray hits, never from the lane polyline."
          if worst > 2.0 else "=> lane tangent is adequate here")

    # ── CHECK 2 ─────────────────────────────────────────────────────────────
    print(f"\n\n# CHECK 2 -- mapped plane vs injected MuJoCo plane, {a.scene}\n")
    base0 = np.asarray(a.base0, float)
    v = np.array([math.cos(a.yaw0), math.sin(a.yaw0)])
    u = np.array([v[1], -v[0]])

    def to_world(p):
        p = np.asarray(p, float)
        return base0 + p[0] * u + p[1] * v

    obst = [o for o in build_obstacles_v3(scene.spec) if o.kind == "wall"]
    print(f"map half-thickness THICK/2 = {THICK / 2:.4f} m   "
          f"injected WALL_THICK_HALF = {WALL_THICK_HALF:.4f} m   "
          f"{'MATCH' if abs(THICK / 2 - WALL_THICK_HALF) < 1e-9 else 'MISMATCH'}")
    print(f"injected z centre = {OBST_HEIGHT_HALF + 0.02:.3f} m, half-height "
          f"{OBST_HEIGHT_HALF:.3f}  -> spans z [{0.02:.3f}, "
          f"{2 * OBST_HEIGHT_HALF + 0.02:.3f}] m\n")
    print(f"{'k':>3s} {'centre err (m)':>15s} {'normal err (deg)':>17s}"
          f" {'thickness err (m)':>18s}")
    max_c = max_n = 0.0
    for o in obst:
        w = scene.spec.walls[o.k]
        p0, p1 = np.asarray(w.p0, float), np.asarray(w.p1, float)
        c_map = 0.5 * (p0 + p1)
        d_map = p1 - p0
        phi_map = math.atan2(float(d_map[1]), float(d_map[0]))
        n_map = np.array([-math.sin(phi_map), math.cos(phi_map)])

        # what place_obstacles writes
        cw = to_world(o.center_map)
        dm = np.array([math.cos(o.yaw_map), math.sin(o.yaw_map)])
        d_w = dm[0] * u + dm[1] * v
        th = math.atan2(float(d_w[1]), float(d_w[0])) + math.pi / 2.0
        # the box's LOCAL +x is its thickness axis == the wall normal
        n_inj_w = np.array([math.cos(th), math.sin(th)])
        # bring both back to the map frame for comparison
        n_inj_map = np.array([float(n_inj_w @ u), float(n_inj_w @ v)])
        c_inj_map = np.array([float((cw - base0) @ u), float((cw - base0) @ v)])

        ec = float(np.linalg.norm(c_inj_map - c_map))
        en = math.degrees(math.acos(min(1.0, abs(float(n_inj_map @ n_map)))))
        et = abs(THICK / 2 - WALL_THICK_HALF)
        max_c, max_n = max(max_c, ec), max(max_n, en)
        print(f"{o.k:3d} {ec:15.6f} {en:17.3f} {et:18.6f}")

    print(f"\nworst centre offset  {max_c:.6f} m")
    print(f"worst normal error   {max_n:.4f} deg")
    ok = max_c < 1e-6 and max_n < 1e-3 and abs(THICK / 2 - WALL_THICK_HALF) < 1e-9
    print(f"\nCORE SANITY CHECK: mapped and injected wall planes are "
          f"{'COINCIDENT' if ok else 'NOT COINCIDENT'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
