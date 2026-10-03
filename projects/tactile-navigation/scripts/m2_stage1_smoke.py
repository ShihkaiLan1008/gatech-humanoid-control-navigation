"""M2 stage-1 smoke: blocking vs tick-inverted goto+turn on the demo scene.

Runs the SAME short mission (goto a point 1.2 m ahead-left, then turn 90° in
place) through (a) the blocking primitives and (b) PlannerCore + tick driver,
in two separate processes with identical seeds, and compares final poses and
statuses. Faithful inversion ⇒ near-identical trajectories (same env build,
same command sequence, deterministic sim).

    PYTHONPATH=src:tactile_controller/src MUJOCO_GL=egl \
      tactile_controller/.venv/bin/python scripts/m2_stage1_smoke.py --mode blocking
      ... --mode tick
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def build(seed: int):
    import sys
    sys.path.insert(0, "scripts")
    from demo_room_run import backend
    from demo_map.scene_demo import build_demo_scene

    scene = build_demo_scene()
    return backend(scene, seed, 0.49)


def mission(be, mode: str) -> dict:
    xy0, yaw0 = be.truth_pose()
    target_w = xy0 + np.array([math.cos(yaw0), math.sin(yaw0)]) * 1.0 + \
        np.array([-math.sin(yaw0), math.cos(yaw0)]) * 0.6
    yaw_goal = yaw0 + math.pi / 2

    if mode == "blocking":
        # world-frame equivalents of goto_map/turn: believed == truth here
        s1 = be.goto_map(be.world_point_to_map(target_w), be.world_yaw_to_map(yaw0),
                         be.world_point_to_map(xy0), be.world_yaw_to_map(yaw0))
        xy_mid, _ = be.truth_pose()
        s2 = be._turn_in_place(yaw_goal, xy_mid)
    else:
        from execution.tick_core import (BaseGoal, Cmd, PlannerCore, goto,
                                         observe, turn_in_place)
        from execution.tick_driver import MujocoTickDriver, params_from_backend
        from local_tactile.fsm import Mode

        drv = MujocoTickDriver(be)
        p = params_from_backend(be)

        def gen():
            s1 = yield from goto(p, target_w, yaw0)
            # one observation tick to freeze the turn's hold point (blocking
            # path reads truth right before _turn_in_place)
            obs = yield from observe(Cmd(
                mode=Mode.LOCOMOTION,
                base=BaseGoal(obs_hold := target_w, yaw0, p.speed_scale_min,
                              hold=True, standing=True)))
            s2 = yield from turn_in_place(p, yaw_goal, obs.xy.copy())
            return {"s1": s1, "s2": s2}

        core = PlannerCore(gen())
        core.set_safety_params(p)
        res = drv.run(core)
        s1, s2 = res["s1"], res["s2"]

    xy1, yaw1 = be.truth_pose()
    return {"mode": mode, "s1": s1, "s2": s2,
            "end_xy": [float(xy1[0]), float(xy1[1])], "end_yaw": float(yaw1),
            "target": [float(target_w[0]), float(target_w[1])],
            "yaw_goal": float(yaw_goal), "t": float(be._t)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["blocking", "tick"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="out/m2")
    a = ap.parse_args()
    be = build(a.seed)
    try:
        rec = mission(be, a.mode)
    finally:
        be.close()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    p = Path(a.out) / f"stage1_{a.mode}_seed{a.seed}.json"
    p.write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec))


if __name__ == "__main__":
    main()
