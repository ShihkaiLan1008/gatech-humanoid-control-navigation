"""Arm A of the comparison: the OLD contact-rich cadence, on a stated segment.

    PYTHONPATH=src:tactile_controller/src MUJOCO_GL=egl \
      tactile_controller/.venv/bin/python scripts/wall_touch_old_cadence.py \
        --scene r5_twinbay --seed 0 --segment 0 --out out/hw/abc

`ExecutiveV3._run_touch_follow` drives its own belief, its own route and its own
station pitch, so running it directly would not be a controlled comparison.  This
reproduces its CADENCE -- and calls the same unmodified `MujocoBackend.press` --
over the SAME `WallSegment`, seed and stations the new behaviour uses:

    stance = lane point moved to `press_standoff_m` off the surface
    goto(stance, yaw FACING the wall)      <- the approach
    stabilize()                            <- the stop
    press()                                <- forward press, `_kiss_x` geometry
    goto(lane point, same facing yaw)      <- the retreat, and the step

That is the behaviour docs/HW_ROBUSTNESS.md 16 diagnoses.  It is reproduced here
to be MEASURED against, not to be restored: nothing in this file is imported by
`wall_parallel_touch`, which keeps its lateral geometry and never faces a wall.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from execution.wall_lane import tap_stations                     # noqa: E402
from lane_graph.scene3 import build_v3_scene                     # noqa: E402
from wall_parallel_demo import backend, candidates               # noqa: E402

PRESS_STANDOFF_M = 0.33      # ExecV3Config.press_standoff_m
LANE_STANDOFF_M = 0.38       # scene3.STANDOFF -- where the step returns to


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene", default="r5_twinbay")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--segment", type=int, default=0)
    ap.add_argument("--standoff", type=float, default=0.28)
    ap.add_argument("--spacing", type=float, default=0.6)
    ap.add_argument("--min-length", type=float, default=1.8)
    ap.add_argument("--max-taps", type=int, default=6)
    ap.add_argument("--fan-offset", type=float, default=0.0)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)

    scene = build_v3_scene(a.scene)
    cands = candidates(scene, a.standoff, a.spacing,
                       min_length_m=a.min_length, min_taps=2)
    seg, stations, _reach = cands[a.segment]
    stations = stations[:a.max_taps]
    bodies = scene.plan.bodies
    # the old cadence faces the wall: yaw points along the INWARD normal
    yaw_face = math.atan2(-seg.normal[1], -seg.normal[0])
    print(f"# old cadence on {a.scene} seg {a.segment}: {len(stations)} stations, "
          f"facing yaw {math.degrees(yaw_face):+.1f} deg "
          f"(travel tangent is {math.degrees(seg.yaw):+.1f})")

    t_wall = time.time()
    be = backend(scene, a.seed, a.standoff)
    out = {"scene": a.scene, "seed": a.seed, "segment_index": a.segment,
           "cadence": "old_touch_follow", "stations_m": stations,
           "segment": seg.as_dict(), "taps": [], "trace": []}
    try:
        for k, s_m in enumerate(stations):
            lane_pt = seg.lane_point(s_m, LANE_STANDOFF_M)
            stance = seg.lane_point(s_m, PRESS_STANDOFF_M)
            here = be.world_point_to_map(be.truth_pose()[0])
            # APPROACH
            g1 = be.goto_map(stance, yaw_face, here,
                             be.world_yaw_to_map(be.truth_pose()[1]))
            d_stance = seg.standoff_of(be.world_point_to_map(be.truth_pose()[0]))
            # STOP
            be.stabilize(yaw_face)
            # PRESS (forward, body +x, the unmodified primitive)
            if hasattr(be, "set_press_frame"):
                be.set_press_frame(seg.tangent, -np.asarray(seg.normal, float))
            t0 = be.time_s
            r = be.press(float(a.fan_offset))
            d_press = seg.standoff_of(be.world_point_to_map(be.truth_pose()[0]))
            # RETREAT + STEP
            g2 = be.goto_map(lane_pt, yaw_face,
                             be.world_point_to_map(be.truth_pose()[0]),
                             be.world_yaw_to_map(be.truth_pose()[1]))
            d_lane = seg.standoff_of(be.world_point_to_map(be.truth_pose()[0]))
            tap = {"station_m": s_m, "goto_stance": g1, "goto_lane": g2,
                   "standoff_at_stance_m": d_stance,
                   "standoff_after_press_m": d_press,
                   "standoff_at_lane_m": d_lane,
                   "cycle_excursion_m": abs(d_lane - d_stance),
                   "peak_force_n": float(r.peak_force_n),
                   "contacted": bool(r.contacted), "valid": bool(r.valid),
                   "reason": str(getattr(r, "reason", "") or ""),
                   "duration_s": round(float(be.time_s - t0), 2)}
            out["taps"].append(tap)
            print(f"  station {s_m:.2f}: stance {d_stance:.3f} -> press "
                  f"{d_press:.3f} -> lane {d_lane:.3f} m "
                  f"(base excursion {tap['cycle_excursion_m']:+.3f})  "
                  f"press {tap['duration_s']:.1f} s peak "
                  f"{tap['peak_force_n']:.2f} N contacted {tap['contacted']}")
    finally:
        out["physical"] = be.physical_summary()
        out["trace"] = [{k: t[k] for k in ("t", "xy", "yaw", "mode",
                                           "force_l", "force_r")}
                        for t in be.trace]
        a.out.mkdir(parents=True, exist_ok=True)
        if getattr(be, "state_log", None):
            ts = np.asarray([t for t, _ in be.state_log], dtype=np.float64)
            qs = np.stack([q for _, q in be.state_log]).astype(np.float32)
            np.savez_compressed(a.out / f"states_{a.scene}_s{a.seed}_A.npz",
                                t=ts, qpos=qs)
        be.close()
    out["wall_clock_s"] = round(time.time() - t_wall, 1)
    (a.out / f"wp_{a.scene}_s{a.seed}_A.json").write_text(
        json.dumps(out, indent=1, default=float))
    exc = [t["cycle_excursion_m"] for t in out["taps"]]
    print(f"\nbase excursion per touch cycle: median "
          f"{float(np.median(exc)) if exc else 0.0:.3f} m")
    print(f"wrote {a.out}  ({out['wall_clock_s']} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
