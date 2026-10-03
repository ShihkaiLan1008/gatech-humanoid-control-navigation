"""Can the base be commanded to, and hold, a CLOSER wall standoff?

    PYTHONPATH=src:tactile_controller/src MUJOCO_GL=egl \
      tactile_controller/.venv/bin/python scripts/standoff_tracking_test.py \
      --targets 0.44 0.40 0.36 0.32 0.28 0.25 --out out/hw/standoff

The lateral-contact gate failed on a 0.124 m gap between where the base stands
(0.39-0.53 m) and how far the arm reaches sideways (0.263 m), and the base is
the larger share.  This isolates the base half: command a descending sequence of
base-to-face standoffs on a straight wall and measure what is realized -- right
after the walk, and again after a settle.  No arm, no contact.

`WallFollowSe2CommandCfg.standoff_range` (trained 0.36-0.44 m) is set to the
requested band through `MujocoConfig.wall_standoff_range`, so the controller's
own notion of a legal wall edge matches the request.  Note that our `_set_goal`
overwrites `pose_target_w` every tick, so that band governs only the FSM's own
resampling -- the commanded stance here is ours either way.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from execution.backends import BackendConfig                 # noqa: E402
from execution.mujoco_backend import MujocoConfig            # noqa: E402
from lane_graph.graph3 import build_graph3                   # noqa: E402
from lane_graph.mj_backend2 import MujocoBackendV2           # noqa: E402
from lane_graph.run_v3 import PRESS_TARGET_N                 # noqa: E402
from lane_graph.scene3 import build_v3_scene                 # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from side_contact_test import pick_wall_site                 # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="r5_twinbay")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--targets", type=float, nargs="+",
                    default=[0.44, 0.40, 0.36, 0.32, 0.28, 0.25])
    ap.add_argument("--band", type=float, nargs=2, default=[0.24, 0.44])
    ap.add_argument("--settle-s", type=float, default=2.0)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)

    scene = build_v3_scene(a.scene)
    graph = build_graph3(scene)
    site = pick_wall_site(scene, graph)
    err_deg, _elen, surf, n_out, tang, eid, _rng = site
    yaw = math.atan2(float(tang[1]), float(tang[0]))

    class _V3(MujocoBackendV2):
        def _press_target_n(self):
            return PRESS_TARGET_N

    be = _V3(scene, BackendConfig(probe_force_n=PRESS_TARGET_N),
             np.random.default_rng(a.seed),
             MujocoConfig(seed=11 + a.seed, episode_budget_s=2000.0,
                          press_overlap_m=0.20,
                          wall_standoff_range=tuple(a.band)),
             render=False, video_path=None)

    out = {"scene": a.scene, "edge": eid, "surface_map": surf.tolist(),
           "normal_map": n_out.tolist(), "yaw_cmd": yaw,
           "standoff_band_set": list(a.band),
           "trained_band": [0.36, 0.44], "trials": []}
    print(f"# standoff tracking -- {a.scene} edge e{eid}, surface "
          f"{np.round(surf, 3)}, yaw {math.degrees(yaw):+.1f} deg")
    print(f"  controller standoff_range set to {tuple(a.band)} "
          f"(trained 0.36-0.44)\n")
    print(f"{'target':>7s} {'goto':>10s} {'realized':>9s} {'err':>7s}"
          f" {'after settle':>13s} {'drift':>7s} {'yaw err':>8s}")
    try:
        for d in a.targets:
            stance = surf + float(d) * n_out
            here = be.world_point_to_map(be.truth_pose()[0])
            st = be.goto_map(stance, yaw, here,
                             be.world_yaw_to_map(be.truth_pose()[1]))
            xy1 = be.world_point_to_map(be.truth_pose()[0])
            r1 = float((xy1 - surf) @ n_out)
            be.stabilize(yaw)
            t0 = be.time_s
            while be.time_s - t0 < a.settle_s:
                be._set_goal(be.truth_pose()[0], be.map_yaw_to_world(yaw),
                             be.mj.speed_scale_min, hold=True, standing=True)
                be._step()
            xy2 = be.world_point_to_map(be.truth_pose()[0])
            r2 = float((xy2 - surf) @ n_out)
            yaw2 = be.world_yaw_to_map(be.truth_pose()[1])
            ye = abs(math.atan2(math.sin(yaw2 - yaw), math.cos(yaw2 - yaw)))
            out["trials"].append({
                "target_m": float(d), "goto": st,
                "realized_m": r1, "err_m": r1 - float(d),
                "after_settle_m": r2, "settle_drift_m": r2 - r1,
                "yaw_err_rad": ye})
            print(f"{d:7.2f} {st:>10s} {r1:9.3f} {r1 - d:+7.3f}"
                  f" {r2:13.3f} {r2 - r1:+7.3f} {ye:8.4f}")
    finally:
        be.close()

    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / f"standoff_{a.scene}_s{a.seed}.json").write_text(
        json.dumps(out, indent=1, default=float))
    tr = out["trials"]
    if tr:
        closest = min(t["after_settle_m"] for t in tr)
        tracked = [t for t in tr if abs(t["err_m"]) <= 0.09]
        print(f"\nclosest standoff HELD: {closest:.3f} m")
        print(f"targets tracked within the 0.09 m locomotion tolerance: "
              f"{len(tracked)}/{len(tr)}"
              f"  ({[t['target_m'] for t in tracked]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
