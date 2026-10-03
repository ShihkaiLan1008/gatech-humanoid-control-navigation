"""Room2 demo run: corner-only landmark, ATE picks the presses.

    PYTHONPATH=src:tactile_controller/src MUJOCO_GL=egl \\
      tactile_controller/.venv/bin/python scripts/room2_run.py \\
        --seed 0 --out out/demo_map/room2            # start -> goal
    ... --reverse                                     # goal -> start
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import numpy as np

from demo_map.executor import DemoExecCfg, DemoRouteExecutor
from demo_map.run_support import guarded_run, render_panel
from demo_map.executor import lane_standoff
from demo_map.deploy_mission import compile_steps as compile_mission
from demo_map.deploy_mission import load as load_mission
from demo_map.scene_room2 import (build_room2_scene, plan_room2_route,
                                  plan_steps_room2, room2_scan_offset)
from execution.run_batch3a import _jsonable


def backend(scene, seed: int, standoff_m: float, crab_ff_m: float = 0.0, *,
            render: bool = False, video_path=None, view: bool = False,
            controller: str = "vforce", **cfg_over):
    """See `demo_room_run.backend`: the replay MUST rebuild the same frame."""
    from execution.backends import BackendConfig
    from execution.mujoco_backend import MujocoConfig
    from lane_graph.mj_backend2 import MujocoBackendV2
    from lane_graph.run_v3 import PRESS_TARGET_N

    if controller == "vforce":
        from execution.vforce_backend import VForceBackend, VForceConfig

        class _VF(VForceBackend):
            def _press_target_n(self):
                return PRESS_TARGET_N

        return _VF(scene, BackendConfig(probe_force_n=PRESS_TARGET_N),
                   np.random.default_rng(seed),
                   VForceConfig(seed=11 + seed, episode_budget_s=4000.0,
                                **cfg_over),
                   render=render, video_path=video_path, view=view)

    class _V3(MujocoBackendV2):
        def _press_target_n(self):
            return PRESS_TARGET_N

    return _V3(scene, BackendConfig(probe_force_n=PRESS_TARGET_N),
               np.random.default_rng(seed),
               MujocoConfig(seed=11 + seed, episode_budget_s=4000.0,
                            press_overlap_m=0.20, wp_yaw_stiffness=3.0,
                            wp_press_speed_m_s=0.035,
                            wp_crab_step_veer_ff_m=float(crab_ff_m),
                            # The dense showcase supplies 0.20 m of
                            # counter-veer so the physical contacts, rather
                            # than only their planned stations, retain the
                            # requested cadence near the corner.
                            wall_standoff_range=(max(0.10, standoff_m - 0.10),
                                                 standoff_m + 0.10),
                            **cfg_over),
               render=render, video_path=video_path, view=view)


MISSION = load_mission()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reverse", action="store_true")
    ap.add_argument("--dense-showcase", action="store_true",
                    help="0.2 m physical contact cadence + hand-footprint corner gate")
    ap.add_argument("--true-delta", type=float, default=0.12,
                    help="registration offset [m] along map +x (sim2real)")
    ap.add_argument("--blind", action="store_true",
                    help="ABLATION: identical route, noise and seed, but no "
                         "press ever happens and the corner is traversed "
                         "open-loop (no information gate).  Answers 'can it "
                         "arrive without touching at all'.")
    ap.add_argument("--controller", choices=("vforce", "hand_force"),
                    default="vforce")
    ap.add_argument("--plan", choices=("deploy", "ate", "auto"), default="ate",
                    help="'deploy' executes "
                         "configs/room2_deploy_mission.json LITERALLY -- the "
                         "steps locomanip_deploy ran, with no EIG landmark "
                         "visits and no monotone-route ratchet.  'ate' is "
                         "this repo's own compile, which adds both.")
    ap.add_argument("--view", action="store_true",
                    help="watch the physics live in a MuJoCo window.  Needs a "
                         "display: run WITHOUT MUJOCO_GL=egl.  It is a passive "
                         "reader of the run being recorded, not a second "
                         "simulation.")
    ap.add_argument("--no-video", action="store_true",
                    help="skip the replay video (the run record is written "
                         "either way)")
    ap.add_argument("--stride", type=int, default=1,
                    help="video: keep every Nth recorded frame")
    ap.add_argument("--cam-distance", type=float, default=None,
                    help="video: tracking-camera distance [m] "
                         "(default fits the room)")
    ap.add_argument("--cam-elevation", type=float, default=None,
                    help="video: tracking-camera elevation [deg]")
    ap.add_argument("--pf-log", action="store_true",
                    help="dump the whole particle set at every absorb to "
                         "pf_<record>.npz -- the full (x, y, theta) belief "
                         "covariance the nav_trace's 2x2 block cannot carry. "
                         "Read by scripts/room2_belief_prior.py.")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--override", action="append", default=[],
                    metavar="FIELD=VALUE",
                    help="Override one DemoExecCfg or VForceConfig field by "
                         "name (repeatable).  Exists so parameter experiments "
                         "run the SAME code with different numbers instead of "
                         "editing source between launches.")
    a = ap.parse_args(argv)

    def _parse_val(s: str):
        if s.lower() in ("true", "false"):
            return s.lower() == "true"
        try:
            return int(s)
        except ValueError:
            pass
        try:
            return float(s)
        except ValueError:
            return s

    overrides = {}
    for item in a.override:
        k, _, v = item.partition("=")
        if not _:
            ap.error(f"--override needs FIELD=VALUE, got {item!r}")
        overrides[k.strip()] = _parse_val(v.strip())

    cfg = DemoExecCfg(seed=a.seed)
    if a.dense_showcase:
        if a.reverse:
            ap.error("--dense-showcase is currently forward-only")
        cfg = DemoExecCfg(seed=a.seed, standoff_m=0.49,
                          touch_style="crab_press_exact", press_spacing_m=0.20)
    scene = build_room2_scene(a.reverse)
    # THE LANE IS A PHYSICAL STANDOFF.  `true_delta_m` shifts the real walls
    # off the map ones, so a lane pinned to the deployed 0.38 in MAP units puts
    # the base 0.38 - delta from the wall it actually walks beside -- and the
    # hands, carried 0.3323 forward, inside it.  Measured at delta 0.12: base
    # 0.27 physical, both tips -0.002..0.06 of the face for a whole run.
    if a.plan == "deploy":
        # The deployed standoffs are PHYSICAL: that stack's map is its world.
        # Under our injected registration offset the mission's own base poses
        # stand the robot `delta` closer to the real wall than the deployment
        # ever did -- measured at 0.12: base 0.27 m instead of 0.38, and both
        # carried hands at or inside the wall face for a whole run.  A literal
        # mission therefore runs at the deployment's own condition.
        a.true_delta = 0.0
        cfg = dataclasses.replace(cfg,
                                  standoff_m=float(MISSION["standoff_m"]),
                                  press_spacing_m=float(MISSION["spacing_m"]),
                                  monotone_route=False,
                                  eig_wall=False)   # verbatim stations ARE
                                                    # the literal mission
    else:
        cfg = dataclasses.replace(cfg, standoff_m=lane_standoff(a.true_delta))
    if a.true_delta:
        scene = __import__("dataclasses").replace(scene,
                                                  true_delta_m=a.true_delta)
    eco_over = {k: v for k, v in overrides.items()
                if hasattr(cfg.economy, k)}
    cfg_over = {k: v for k, v in overrides.items()
                if hasattr(cfg, k) and k not in eco_over}
    vf_over = {k: v for k, v in overrides.items()
               if not hasattr(cfg, k) and k not in eco_over}
    if eco_over:
        cfg = dataclasses.replace(
            cfg, economy=dataclasses.replace(cfg.economy, **eco_over))
        print("economy overrides:", eco_over)
    if cfg_over:
        cfg = dataclasses.replace(cfg, **cfg_over)
        print("cfg overrides:", cfg_over)
    if vf_over:
        print("backend overrides:", vf_over)
    planned = plan_room2_route(scene)
    print("planner route:", planned)
    if a.plan == "auto":
        # the SAME map-in/route-out path the demo room takes: no room2
        # geometry is written down here either
        from demo_map.auto_route import plan_auto_route, route_summary
        from demo_map.scene_demo import (TURN_PRESS_LANE_OFF,
                                         plan_steps_demo_room)
        route = plan_auto_route(scene, lane_off=TURN_PRESS_LANE_OFF)
        print("auto route:", route_summary(route))
        steps, plan_report = plan_steps_demo_room(
            cfg, scene, route, route_lane_off=TURN_PRESS_LANE_OFF)
        print(f"auto plan: {len(steps)} steps, visited {plan_report['visited']},"
              f" skipped {plan_report['skipped']}")
    else:
        steps = plan_steps_room2(cfg, a.reverse, scene=scene)
    if a.plan == "deploy":
        if a.reverse:
            ap.error("--plan deploy is forward-only (the mission is)")
        steps = compile_mission(MISSION, steps)
        print(f"deploy mission: {len(steps)} steps, "
              f"{sum(len(s['stations']) for s in steps if s['kind'] == 'follow')}"
              f" presses, standoff {cfg.standoff_m}, spacing "
              f"{cfg.press_spacing_m}  (no EIG, no ratchet)")
    if a.blind:
        # Strip the touching, keep everything else byte-identical: same
        # planner route, same noise draws, same PF, same goal tolerance.
        blind = []
        for st in steps:
            if st["kind"] == "follow":
                blind.append({**st, "stations": []})
            elif st["kind"] == "ate_corner":
                blind.extend(st.get("between", []))   # open-loop turn
            elif st["kind"] == "corner_gate":
                continue                              # nothing to gate on
            else:
                blind.append(st)
        steps = blind
    d = room2_scan_offset(a.reverse)
    leg = "rev" if a.reverse else "fwd"
    n_r = sum(len(s["stations"]) for s in steps if s["kind"] == "follow")
    print(f"room2 {leg}: {len(steps)} steps, {n_r} rhythm presses + ATE corner")

    t0 = time.time()
    be = backend(scene, a.seed, cfg.standoff_m,
                 crab_ff_m=0.20 if a.dense_showcase else 0.0,
                 controller=a.controller, view=a.view, **vf_over)
    ex = DemoRouteExecutor(scene, be, None, cfg)
    if a.pf_log:
        ex.pf_log = []
    rec, how = guarded_run(ex, lambda: ex.run(steps))

    states_npz = a.out / f"states_room2_{leg}_s{a.seed}{a.tag}.npz"
    summary = be.physical_summary()
    a.out.mkdir(parents=True, exist_ok=True)
    if getattr(be, "state_log", None):
        ts = np.asarray([t for t, _ in be.state_log], dtype=np.float64)
        qs = np.stack([q for _, q in be.state_log]).astype(np.float32)
        np.savez_compressed(states_npz, t=ts, qpos=qs)
    be.close()

    # THE DRAWN ROUTE IS WHAT THE ROBOT WILL VISIT, in order: each goto's
    # point and each station's lane point.  Sweeping `0 .. seg.length` per
    # follow step drew whole walls instead -- and once the literal deploy
    # mission split a wall across `rhythm` and `corner:pre/post`, it drew each
    # wall TWICE and ran a line straight across the goal pillar (the synthetic
    # tangent face the press uses is a `follow` segment too).
    route_poly = []
    for st in steps:
        if st["kind"] == "goto":
            route_poly.append([float(v) for v in np.asarray(st["p"])])
        elif st["kind"] == "follow":
            seg = st["seg"]
            off = float(st.get("standoff") or cfg.standoff_m)
            for s0 in st["stations"]:
                route_poly.append([float(v) for v in
                                   seg.lane_point(float(s0), off)])
    doc = {
        "scene": scene.name, "seed": a.seed, "reverse": bool(a.reverse),
        # the replay rebuilds the backend from this: the dense showcase adds
        # crab counter-veer, which is a MujocoConfig field and so part of the
        # spawn-deciding config
        "dense_showcase": bool(a.dense_showcase),
        "controller": a.controller,
        # the map frame itself, so a replay never has to re-derive it
        "spawn_frame": {"base0": [float(v) for v in np.asarray(be.base0)],
                        "u": [float(v) for v in np.asarray(be.u)],
                        "v": [float(v) for v in np.asarray(be.v)]},
        "standoff_m": cfg.standoff_m,
        "scan_start_offset": [float(v) for v in d],
        "true_delta_m": float(a.true_delta),
        "planner_route": planned,
        "goal_map": [float(v) for v in scene.goal_map_xy],
        "route_polyline_map": route_poly,
        "planned_steps": [
            {k: (v.as_dict() if hasattr(v, "as_dict") else
                 v.tolist() if isinstance(v, np.ndarray) else
                 v if not isinstance(v, list) else
                 [(x if not isinstance(x, dict) else
                   {kk: (vv.as_dict() if hasattr(vv, "as_dict") else
                         vv.tolist() if isinstance(vv, np.ndarray) else vv)
                    for kk, vv in x.items()}) for x in v])
             for k, v in s.items() if k != "faces"}
            for s in steps],
        "outcome": rec.outcome if rec else "crashed",
        "goal_dist_true_m": rec.goal_dist_true_m if rec else None,
        "goal_dist_believed_m": rec.goal_dist_believed_m if rec else None,
        "touches": rec.touches if rec else [],
        "replans": rec.replans if rec else [],
        "ig_decisions": getattr(rec, "ig_decisions", []) if rec else [],
        "steps_log": rec.steps if rec else [],
        "nav_trace": rec.nav_trace if rec else [],
        "physical": summary,
        "wall_clock_s": round(time.time() - t0, 1),
    }
    out_json = a.out / f"room2_{leg}_s{a.seed}{a.tag}.json"
    out_json.write_text(json.dumps(_jsonable(doc), indent=1, default=float))
    if getattr(ex, "pf_log", None):
        # beside the record, named so `belief_prior.harvest` finds it
        pf = {"n_absorbs": len(ex.pf_log),
              "t": np.asarray([e["t"] for e in ex.pf_log], float),
              "err_m": np.asarray([e["err_m"] for e in ex.pf_log], float),
              "truth_xy": np.asarray([e["truth_xy"] for e in ex.pf_log], float),
              "truth_yaw": np.asarray([e["truth_yaw"] for e in ex.pf_log],
                                      float)}
        for k, e in enumerate(ex.pf_log):
            pf[f"xy{k}"] = e["xy"].astype(np.float32)
            pf[f"theta{k}"] = e["theta"].astype(np.float32)
            pf[f"w{k}"] = e["w"].astype(np.float32)
        np.savez_compressed(a.out / f"pf_{out_json.stem}.npz", **pf)
        print(f"wrote {a.out}/pf_{out_json.stem}.npz "
              f"({len(ex.pf_log)} particle snapshots)")

    if rec:
        hits = [t for t in rec.touches if t.get("contacted")]
        band = [t for t in hits if 10.0 <= (t.get("peak_force_n") or 0) <= 25.0]
        print(f"\noutcome {rec.outcome}  goal(true) {rec.goal_dist_true_m:.2f}"
              f"  goal(bel) {rec.goal_dist_believed_m:.2f}")
        print(f"touches {len(hits)}/{len(rec.touches)} contacted, "
              f"{len(band)} in band")
        for st in rec.steps:
            if st.get("kind") == "ate":
                print("ATE corner:", json.dumps(_jsonable(st), indent=1)[:900])
        errs = [x["err_m"] for x in rec.nav_trace]
        print(f"belief err median {np.median(errs):.3f} final {errs[-1]:.3f}")
        print(f"aborts {summary.get('force_aborts')}  peak "
              f"{summary.get('peak_force_n'):.2f} N")
    print(f"wrote {out_json}  ({doc['wall_clock_s']} s)")
    # EVERY ending gets its video -- arrived, stuck, or Ctrl-C.
    if not a.no_video:
        render_panel(out_json, states_npz,
                     a.out / f"room2_{leg}_s{a.seed}{a.tag}.mp4",
                     stride=a.stride, distance=a.cam_distance,
                     elevation=a.cam_elevation)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
