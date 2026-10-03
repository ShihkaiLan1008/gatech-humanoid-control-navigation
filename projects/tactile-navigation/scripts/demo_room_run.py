"""Run the manual demo route on the scanned room with turn-press touches.

    PYTHONPATH=src:tactile_controller/src MUJOCO_GL=egl \\
      tactile_controller/.venv/bin/python scripts/demo_room_run.py \\
        --seed 0 --out out/demo_map/run

Everything the demo video needs is logged: true pose (evaluation only),
belief/PF pose and modes, the drawn route, the realized trajectory, every
touch with its force, and recoveries.  ``--max-steps`` truncates the compiled
step list for smoke tests.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


from demo_map.executor import (DemoExecCfg, DemoRouteExecutor,
                               plan_steps_for_route)
from demo_map.run_support import guarded_run, render_panel
from demo_map.executor import FOLLOW_STANDOFF, lane_standoff
from demo_map.scene_demo import (TURN_PRESS_LANE_OFF, build_demo_scene,
                                 demo_route, plan_steps_demo_room,
                                 scan_start_offset, translated_scan_map)
from execution.run_batch3a import _jsonable


def _plan_jsonable(o):
    """Recursive serializer for a compiled step list.

    The ATE compile nests further than the old one did -- an `ate_corner`
    carries a `faces` list of dicts each holding a WallSegment, and a `between`
    list of goto dicts holding arrays -- so a one-level comprehension leaves
    live objects in the doc and `json.dumps(default=float)` dies on them.  The
    video reads `planned_steps` for its camera and its touch arrows, so this
    has to survive every step kind, not just `follow`."""
    if hasattr(o, "as_dict"):
        return o.as_dict()
    if isinstance(o, dict):
        return {k: _plan_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_plan_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return o


def backend(scene, seed: int, standoff_m: float, *, render: bool = False,
            video_path=None, view: bool = False, controller: str = "vforce",
            **cfg_over):
    """The demo room's backend.  ALSO the replay's -- and that is the point.

    The map frame is the SPAWN: `base0` is wherever the env resets and `u`/`v`
    is its facing, so every world<->map conversion depends on it.  The reset
    pose is sampled from the env config, so a replay that rebuilds the backend
    with different `MujocoConfig` fields gets a different frame: measured, the
    video's own config put base0 0.64 m away and the heading 173 deg around,
    which aimed the tracking camera through a wall and drew the realized
    trajectory in the wrong place.  One factory, `cfg_over` for the things the
    replay genuinely needs to change (renderer size, camera), and the fields
    that decide the spawn stay identical by construction."""
    from execution.backends import BackendConfig
    from execution.mujoco_backend import MujocoConfig
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

    from lane_graph.mj_backend2 import MujocoBackendV2

    class _V3(MujocoBackendV2):
        def _press_target_n(self):
            return PRESS_TARGET_N

    return _V3(scene, BackendConfig(probe_force_n=PRESS_TARGET_N),
               np.random.default_rng(seed),
               MujocoConfig(seed=11 + seed, episode_budget_s=4000.0,
                            press_overlap_m=0.20,
                            # the 19 acceptance runs all turned with this;
                            # without it the in-place turn back to tangent was
                            # measured to time out (30 s) at every station
                            wp_yaw_stiffness=3.0,
                            # 0.035 m/s verified on r5 (CM: 12.9-14.3 N, no
                            # aborts); saves ~2 s per press over 0.025
                            wp_press_speed_m_s=0.035,
                            wall_standoff_range=(max(0.10, standoff_m - 0.08),
                                                 standoff_m + 0.08),
                            **cfg_over),
               render=render, video_path=video_path, view=view)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--standoff", type=float, default=None)
    ap.add_argument("--spacing", type=float, default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--start-step", type=int, default=None,
                    help="drop the plan's first N steps and begin at N.  The "
                         "robot cannot walk there, so this needs --start-from "
                         "/ --start-t: the frame is moved so the robot stands "
                         "where the reference run stood, and the belief is "
                         "seeded with the error the reference carried.")
    ap.add_argument("--start-from", type=Path, default=None,
                    help="a previous demo_run_sN.json to take that state from")
    ap.add_argument("--start-t", type=float, default=None,
                    help="reference-run time [s]; the LAST nav_trace entry at "
                         "or before it supplies the pose and the belief")
    ap.add_argument("--tick", action="store_true",
                    help="drive the route through PlannerCore.tick (M2)")
    ap.add_argument("--controller", choices=("vforce", "hand_force"),
                    default="vforce",
                    help="'vforce' drives the decoupled vel-force checkpoint "
                         "(vforce_model_60000_v1, force-commanded presses, "
                         "honest sensing) in its own force_maze env family; "
                         "'hand_force' is the wall-following stack the "
                         "earlier demos ran (truth-driven motion layer).")
    ap.add_argument("--walls-json", type=Path, default=None,
                    help="drive a DIFFERENT scanned room: the wall JSON from "
                         "scripts/extract_scan_walls.py.  The mission is then "
                         "corner -> corner, the two furthest apart the base "
                         "can stand beside, and the route is the planner's. "
                         "Implies --plan auto: there is no drawn route for a "
                         "room nobody has drawn.")
    ap.add_argument("--plan", choices=("ate", "rhythm", "auto"), default="ate",
                    help="'ate' compiles room2's planner logic -- ATE owns "
                         "every reachable non-wall body and the corner, the "
                         "flat runs keep the rule-based cadence.  'rhythm' is "
                         "the older compile: cadence everywhere, no landmark "
                         "episodes, hardcoded apex stations at the curves.")
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
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--override", action="append", default=[],
                    metavar="FIELD=VALUE",
                    help="Override one DemoExecCfg field by name "
                         "(repeatable), exactly as room2_run does.")
    a = ap.parse_args(argv)

    def _parse_val(v: str):
        if v.lower() in ("true", "false"):
            return v.lower() == "true"
        for cast in (int, float):
            try:
                return cast(v)
            except ValueError:
                pass
        return v

    cfg = DemoExecCfg(seed=a.seed)
    # the demo-room-trained checkpoint, which scanned rooms (room830) borrow
    # too: the WM's covariance head is map-independent, so no per-room
    # training is needed (--override wm_run_dir=... still wins below)
    cfg.wm_run_dir = "models/wm/demo_room"
    for item in a.override:
        k, sep, v = item.partition("=")
        if not sep or not hasattr(cfg, k.strip()):
            ap.error(f"--override needs a DemoExecCfg FIELD=VALUE, got {item!r}")
        setattr(cfg, k.strip(), _parse_val(v.strip()))
    if a.spacing is not None:
        cfg.press_spacing_m = a.spacing

    mission = None
    if a.walls_json is not None:
        from demo_map.scene_scan import build_corner_room
        _m, scene, mission = build_corner_room(a.walls_json)
        a.plan = "auto"
        print(f"room {a.walls_json.stem}: corner {mission['from']['name']} -> "
              f"corner {mission['to']['name']}, {mission['apart_m']:.1f} m apart")
    else:
        scene = build_demo_scene()
    # the lane realises the deployed standoff PHYSICALLY -- see room2_run.
    # AFTER the scene exists: it is the scene that carries `true_delta_m`.
    cfg.standoff_m = lane_standoff(getattr(scene, "true_delta_m", 0.0) or 0.0,
                                   base_m=FOLLOW_STANDOFF)
    if a.standoff is not None:
        cfg.standoff_m = a.standoff
    m = translated_scan_map()
    plan_report = None
    if a.plan == "auto":
        # END TO END FROM THE MAP: the route is the geometric planner's, its
        # segments classified against the map, and the same compiler turns
        # it into steps.  Nothing about this room is written down.
        from demo_map.auto_route import plan_auto_route, route_summary
        route = plan_auto_route(scene, lane_off=TURN_PRESS_LANE_OFF)
        print("auto route:", route_summary(route))
    else:
        route = demo_route()
    if a.plan in ("ate", "auto"):
        steps, plan_report = plan_steps_demo_room(
            cfg, scene, route, route_lane_off=TURN_PRESS_LANE_OFF)
    else:
        steps = plan_steps_for_route(route, cfg, m.curves,
                                     route_lane_off=TURN_PRESS_LANE_OFF)
    if a.max_steps is not None:
        steps = steps[:a.max_steps]
    seed_row = None
    if a.start_step is not None:
        if a.start_from is None or a.start_t is None:
            ap.error("--start-step needs --start-from and --start-t")
        ref = json.loads(Path(a.start_from).read_text())
        rows = [e for e in ref["nav_trace"] if e["t"] <= a.start_t + 1e-9]
        if not rows:
            ap.error(f"no nav_trace entry at or before t={a.start_t}")
        seed_row = rows[-1]
        cov = np.asarray(seed_row["cov_xy"], float)
        # THE BELIEF IS THE POINT.  Starting mid-route with a fresh, tight
        # prior would not reproduce the situation being tested -- the whole
        # question is what the robot does while holding the error it had.
        scene.prior_components = ((tuple(float(v) for v in seed_row["bel_xy"]),
                                   float(seed_row["bel_yaw"]), 0, 1.0),)
        scene.prior_sigma_along_m = float(np.sqrt(max(cov[0, 0], 1e-8)))
        scene.prior_sigma_lateral_m = float(np.sqrt(max(cov[1, 1], 1e-8)))
        scene.prior_sigma_theta_rad = float(seed_row.get("sigma_yaw_rad", 0.03))
        steps = steps[a.start_step:]
        print(f"start: step {a.start_step} from {a.start_from} @ "
              f"t={seed_row['t']:.1f}  bel={seed_row['bel_xy']} "
              f"truth={seed_row['truth_xy']} err={seed_row['err_m']:.3f} "
              f"sigma={seed_row['sigma_m']:.3f}")
    n_press = sum(len(s["stations"]) for s in steps if s["kind"] == "follow")
    n_ate = sum(1 for s in steps
                if s["kind"] in ("ate_landmark", "ate_pillar", "ate_corner"))
    print(f"route: {len(steps)} steps, {n_press} rhythm presses, "
          f"{n_ate} landmark episodes ({a.plan} compile)")
    if plan_report:
        print(f"  landmarks: {', '.join(plan_report['visited'])}"
              f"  (goal-confirming: {plan_report['goal_landmark']})")

    t0 = time.time()
    be = backend(scene, a.seed, cfg.standoff_m, controller=a.controller,
                 view=a.view)
    if seed_row is not None:
        # move the ROOM, not the robot: see `place_map_frame`
        be.place_map_frame(seed_row["truth_xy"], seed_row["truth_yaw"])
    ex = DemoRouteExecutor(scene, be, route, cfg)

    def _route():
        if a.tick:
            from execution.tick_core import PlannerCore
            from execution.tick_driver import MujocoTickDriver, params_from_backend
            p = params_from_backend(be)
            return MujocoTickDriver(be).run(PlannerCore(ex.run_gen(steps, p), p))
        return ex.run(steps)

    rec, how = guarded_run(ex, _route)
    states_npz = a.out / f"states_demo_s{a.seed}{a.tag}.npz"
    summary = be.physical_summary()
    a.out.mkdir(parents=True, exist_ok=True)
    if getattr(be, "state_log", None):
        ts = np.asarray([t for t, _ in be.state_log], dtype=np.float64)
        qs = np.stack([q for _, q in be.state_log]).astype(np.float32)
        np.savez_compressed(states_npz, t=ts, qpos=qs)
    be.close()

    doc = {
        "scene": scene.name, "seed": a.seed, "plan": a.plan,
        "walls_json": None if a.walls_json is None else str(a.walls_json),
        "controller": a.controller,
        # the map frame itself, so a replay never has to re-derive it
        "spawn_frame": {"base0": [float(v) for v in np.asarray(be.base0)],
                        "u": [float(v) for v in np.asarray(be.u)],
                        "v": [float(v) for v in np.asarray(be.v)]},
        "plan_report": plan_report,
        "standoff_m": cfg.standoff_m, "lane_off_m": TURN_PRESS_LANE_OFF,
        "scan_start_offset": ([float(v) for v in scan_start_offset()]
                              if a.walls_json is None else [0.0, 0.0]),
        "mission": (None if mission is None else
                    {"from": mission["from"]["name"],
                     "to": mission["to"]["name"],
                     "apart_m": mission["apart_m"]}),
        "goal_map": [float(v) for v in scene.goal_map_xy],
        "route_drawn": [s.as_dict() if hasattr(s, "as_dict") else None
                        for s in []],
        "route_polyline_map": route.polyline(0.05).tolist(),
        "planned_steps": _plan_jsonable(steps),
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
    out_json = a.out / f"demo_run_s{a.seed}{a.tag}.json"
    out_json.write_text(json.dumps(_jsonable(doc), indent=1, default=float))

    if rec:
        hits = [t for t in rec.touches if t.get("contacted")]
        in_band = [t for t in hits if 10.0 <= (t.get("peak_force_n") or 0) <= 25.0]
        print(f"\noutcome {rec.outcome}  goal(true) {rec.goal_dist_true_m:.2f} m"
              f"  goal(believed) {rec.goal_dist_believed_m:.2f} m")
        print(f"touches: {len(hits)}/{len(rec.touches)} contacted, "
              f"{len(in_band)} in 10-25 N")
        for t in rec.touches:
            print(f"  {t['label']:8s} s={t['station_m']:5.2f}  "
                  f"{t['status']:10s} {t.get('peak_force_n', 0) or 0:6.2f} N  "
                  f"{t.get('reason', '')}")
        errs = [x["err_m"] for x in rec.nav_trace]
        sig = [x["sigma_m"] for x in rec.nav_trace]
        print(f"belief err: median {np.median(errs):.3f}  final {errs[-1]:.3f} m"
              f"   sigma: start {sig[0]:.3f}  final {sig[-1]:.3f} m")
        print(f"aborts {summary.get('force_aborts')}  peak "
              f"{summary.get('peak_hand_force_n', summary.get('peak_force_n'))} N")
    print(f"wrote {out_json}  ({doc['wall_clock_s']} s)")
    # EVERY ending gets its video -- arrived, stuck, or Ctrl-C.
    if not a.no_video:
        render_panel(out_json, states_npz,
                     a.out / f"demo_room_s{a.seed}{a.tag}.mp4",
                     stride=a.stride, distance=a.cam_distance,
                     elevation=a.cam_elevation)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
