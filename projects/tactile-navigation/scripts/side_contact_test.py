"""Subtask 1 (corrected) -- static LATERAL wall contact, heading from the surface.

    PYTHONPATH=src:tactile_controller/src MUJOCO_GL=egl \
      tactile_controller/.venv/bin/python scripts/side_contact_test.py \
      --standoff 0.32 0.29 0.26 0.23 --out out/hw/side2

The first version of this test commanded a heading taken from the LANE
POLYLINE's local tangent.  V3 lanes bow around features, so that tangent is up
to **44 deg** off the wall it is meant to parallel (docs/HW_ROBUSTNESS.md 10),
and the whole gate was invalid: the hand ended ~7 cm short of the wall and read
0.00 N, correctly.

Here the heading comes from the CONTACTED SURFACE: cast the press ray, take the
surface normal the ray returns, and use its perpendicular as the travel
direction.  No lane tangent, no `_kiss_x`.  The lateral `side_contact` primitive
is unchanged.

Logged per standoff: commanded and realized base yaw, the wall tangent/normal,
EE world position, the SIGNED EE-to-wall-plane distance (mapped == injected,
verified in 10), the actual MuJoCo contact pair from the per-obstacle sensors,
raw hand force, coverage and peak.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from global_nav.geometry import raycast              # noqa: E402
from lane_graph.graph3 import build_graph3           # noqa: E402
from lane_graph.run_v3 import _v3_backend            # noqa: E402
from lane_graph.scene3 import build_v3_scene         # noqa: E402


def pick_wall_site(scene, graph):
    """A STRAIGHT stretch of ordinary wall, and its surface frame.

    Returns (surface_point_map, outward_normal_map, tangent_map).  The site is
    chosen where the lane's own tangent already agrees with the surface, so the
    test is about contact rather than about lane curvature.
    """
    best = None
    for e in graph.edges:
        if e.kind != "TOUCH_FOLLOW" or e.poly is None:
            continue
        poly, nrm = np.asarray(e.poly, float), np.asarray(e.normal, float)
        for i in range(1, len(poly) - 1):
            d, nn, _idx = raycast(poly[i], nrm[i], 2.0, scene.plan.bodies)
            if d is None:
                continue
            nn = np.asarray(nn, float)
            tang = np.array([-nn[1], nn[0]])
            u_lane = poly[i + 1] - poly[i - 1]
            u_lane = u_lane / max(float(np.linalg.norm(u_lane)), 1e-9)
            err = math.degrees(math.acos(min(1.0, abs(float(u_lane @ tang)))))
            surf = poly[i] + float(d) * np.asarray(nrm[i], float)
            cand = (err, float(e.length), surf, nn, tang, e.eid, float(d))
            if best is None or cand[0] < best[0]:
                best = cand
    return best


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="r5_twinbay")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--standoff", type=float, nargs="+",
                    default=[0.32, 0.29, 0.26, 0.23])
    ap.add_argument("--force", type=float, default=6.0)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)

    scene = build_v3_scene(a.scene)
    graph = build_graph3(scene)
    site = pick_wall_site(scene, graph)
    if site is None:
        print("no wall site found")
        return 1
    err_deg, _elen, surf, n_out, tang, eid, _rng = site
    yaw_cmd = math.atan2(float(tang[1]), float(tang[0]))
    lat = np.array([-math.sin(yaw_cmd), math.cos(yaw_cmd)])   # body +y in map
    # the wall lies opposite the outward normal
    side = 1 if float(np.dot(-n_out, lat)) >= 0 else -1

    print(f"# site: edge e{eid}, surface point {np.round(surf, 3)} (map)")
    print(f"  outward normal {np.round(n_out, 4)}  tangent {np.round(tang, 4)}")
    print(f"  commanded yaw {yaw_cmd:+.4f} rad ({math.degrees(yaw_cmd):+.1f} deg)"
          f"   lane-vs-surface residual at this site {err_deg:.2f} deg")
    print(f"  wall is on the body's {'LEFT (+y)' if side > 0 else 'RIGHT (-y)'}"
          f" -> {'left' if side > 0 else 'right'} hand\n")

    be = _v3_backend(scene, np.random.default_rng(a.seed), a.seed)
    out = {"scene": a.scene, "edge": eid, "surface_map": surf.tolist(),
           "normal_map": n_out.tolist(), "tangent_map": tang.tolist(),
           "yaw_cmd": yaw_cmd, "side": "left" if side > 0 else "right",
           "lane_surface_residual_deg": err_deg, "trials": []}
    try:
        for d in a.standoff:
            stance = surf + float(d) * n_out       # d measured from the SURFACE
            here = be.world_point_to_map(be.truth_pose()[0])
            st = be.goto_map(stance, yaw_cmd, here,
                             be.world_yaw_to_map(be.truth_pose()[1]))
            be.stabilize(yaw_cmd)
            xy_map = be.world_point_to_map(be.truth_pose()[0])
            yaw_real = be.world_yaw_to_map(be.truth_pose()[1])
            base_off = float((xy_map - surf) @ n_out)   # realized standoff

            r = be.side_contact(side, float(d), force_n=a.force)

            # signed EE-to-wall-plane distance, per tick, in the map frame
            sd = []
            for t in r["ticks"]:
                ee_map = be.world_point_to_map(np.asarray(t["ee_w"][:2], float))
                sd.append(float((ee_map - surf) @ n_out))
            pairs = sorted({p["sensor"] for t in r["ticks"] for p in t["pairs"]})
            trial = {k: v for k, v in r.items() if k != "ticks"}
            trial.update({
                "standoff_cmd_m": float(d), "goto": st,
                "yaw_cmd": yaw_cmd, "yaw_realized": float(yaw_real),
                "yaw_err_rad": float(abs(math.atan2(
                    math.sin(yaw_real - yaw_cmd), math.cos(yaw_real - yaw_cmd)))),
                "base_standoff_realized_m": base_off,
                "ee_signed_dist_min_m": float(min(sd)) if sd else None,
                "ee_signed_dist_end_m": float(sd[-1]) if sd else None,
                "contact_pairs": pairs,
                "ticks": r["ticks"][::6]})
            out["trials"].append(trial)
            print(f"standoff {d:.2f} m | goto {st} | yaw err "
                  f"{trial['yaw_err_rad']:.4f} rad | realized standoff "
                  f"{base_off:+.3f} m")
            print(f"   coverage {trial['coverage']:.3f}  peak {trial['peak_force_n']:.2f} N"
                  f"  guard {trial['guard']}  status {trial['status']}")
            print(f"   EE signed dist to wall: min {trial['ee_signed_dist_min_m']}"
                  f"  end {trial['ee_signed_dist_end_m']}   pairs {pairs}\n")
    finally:
        be.close()

    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / f"side_{a.scene}_s{a.seed}.json").write_text(
        json.dumps(out, indent=1, default=float))
    good = [t for t in out["trials"]
            if t["coverage"] >= 0.5 and not t["guard"]
            and 3.0 <= t["peak_force_n"] <= 9.0]
    print(f"GATE: {len(good)}/{len(out['trials'])} standoffs gave sustained "
          f"4-8 N lateral contact at safe force")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
