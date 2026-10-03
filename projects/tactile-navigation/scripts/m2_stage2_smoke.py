"""M2 stage-2 smoke: blocking vs tick-inverted PRESS (bounded free sweep).

At spawn, faces map +y (open space on the demo scene start), sets the press
frame, and presses with a small travel cap -> both paths must deliver the same
bounded-free-sweep/short-sweep outcome, peaks, and timing. Wall-contact press
equivalence is covered wholesale by the stage-4 demo regression.

    PYTHONPATH=src:tactile_controller/src MUJOCO_GL=egl \
      tactile_controller/.venv/bin/python scripts/m2_stage2_smoke.py --mode blocking|tick
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def build(seed: int):
    import sys
    sys.path.insert(0, "scripts")
    from demo_room_run import backend
    from demo_map.scene_demo import build_demo_scene

    return backend(build_demo_scene(), seed, 0.49)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["blocking", "tick"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="out/m2")
    a = ap.parse_args()
    be = build(a.seed)
    try:
        be.set_press_frame((1.0, 0.0), (0.0, 1.0))   # travel +x_map, face +y_map
        xy0, yaw0 = be.truth_pose()
        cap = 0.45
        if a.mode == "blocking":
            r = be.press(0.18, max_normal_x_m=cap, anchor_xy_w=xy0)
        else:
            from execution.tick_core import PlannerCore, press
            from execution.tick_driver import MujocoTickDriver, params_from_backend

            drv = MujocoTickDriver(be)
            p = params_from_backend(be)
            core = PlannerCore(press(p, be.probe_name, 0.18,
                                     max_normal_x_m=cap, anchor_xy_w=xy0), p)
            r = drv.run(core)
        rec = {"mode": a.mode, "probe_hand": be.probe_name,
               "contacted": bool(r.contacted), "reason": r.reason,
               "valid": bool(r.valid), "peak_n": float(r.peak_force_n),
               "swept_past_m": float(r.swept_past_face_m),
               "n_free": len(r.swept_free_body), "t": float(be._t)}
    finally:
        be.close()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / f"stage2_{a.mode}_seed{a.seed}.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec))


if __name__ == "__main__":
    main()
