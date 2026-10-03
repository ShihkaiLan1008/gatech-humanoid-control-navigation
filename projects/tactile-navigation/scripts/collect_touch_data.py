#!/usr/bin/env python3
"""Collect belief-transition data for a tactile world model.  DATA ONLY.

    scripts/collect_touch_data.py --room room2 --smoke      # ~200, validates
    scripts/collect_touch_data.py --room room2              # the full 12 000
    scripts/collect_touch_data.py --room room2 --n-envs 1    # single world

The training unit is one step of belief evolution:

    (b_t, c_t, M)  ->  (o_{t+1}, b_{t+1})

A LOCAL SEGMENT places the base at a random point near the room's planned
route, seeds a particle filter from the covariance that filter has ACTUALLY
carried in that route region, then takes 10-20 freely sampled touches, folding
each real observation into the filter with one ordinary update.  Segments are
independent: there is no navigation between them, and no planner anywhere.

Each run leaves, in `--out`:

    manifest.json      map identity, route, workspace bins, belief prior,
                       controller config and digest, seed
    transitions.jsonl  one line per attempt: b_t, c_t, o_{t+1}, b_{t+1}, and
                       the privileged simulator state beside them
    beliefs/*.npz      the particle sets -- section 8 forbids reducing a
                       belief to (mu, Sigma)
    qpos.npz           full generalized coordinates (single-world runs)
    summary.json/.txt  every section-8 diagnostic and the pass/fail checks
    topdown.png        route, base poses, contacts, misses -- the bias check
    coverage.png       the action/offset/force histograms
    beliefs.png        particle clouds before and after several touches

Ctrl-C is caught: the JSONL is appended as the run happens, and the summary
and the plots are produced from whatever was collected.

Dispatched into `tactile_controller/.venv` in a SUBPROCESS, for the reason
`run_demo.py` documents: two processes racing to initialise the same GPU/warp
context deadlock on this machine.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import json
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROOMS = ("room2", "demo_room")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Collect (x_t, M, c_t, o_t+1, x_t+1) touch transitions.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--room", choices=ROOMS, required=True,
                    help="which scanned room to collect in")
    ap.add_argument("--reverse", action="store_true",
                    help="room2 only: the goal->start scene")
    ap.add_argument("--n-envs", type=int, default=128,
                    help="parallel simulated worlds (default 128).  1 uses the "
                         "single-environment path.  Measured on an RTX 4090: "
                         "512 envs cost 25%% more per step than one, so this "
                         "is the difference between ~1 h and ~57 h for 12 000.")
    ap.add_argument("--append", action="store_true",
                    help="EXTEND the dataset already in --out instead of "
                         "replacing it: rows are appended, belief shards "
                         "continue, segment ids continue (so existing splits "
                         "are unchanged), and --target counts the COMBINED "
                         "dataset.")
    ap.add_argument("--vertex-share", type=float, default=0.0,
                    help="share of the COMBINED dataset spent standing beside "
                         "a corner VERTEX -- its own region, because the "
                         "centreline turns at the corner and no pose on it "
                         "ever stands beside the vertex on either face.")
    ap.add_argument("--no-hug-corner", action="store_true",
                    help="use the planner's corner shortcut verbatim as the "
                         "reference centreline.  By default the shortcut is "
                         "replaced by the wall-following elbow, because the "
                         "cut passes 1.13 m from the corner vertex and 0.79 m "
                         "from any face -- outside the arm's reach -- so the "
                         "one landmark in the room collects no contact data.")
    ap.add_argument("--steps-per-segment", default="10-20",
                    help="transitions per local segment, as LO-HI "
                         "(section 9: 10-20, varied for feasibility)")
    ap.add_argument("--controller", choices=("vforce",), default="vforce",
                    help="vforce: the deployed decoupled vel-force checkpoint. "
                         "The free-target press is built on its JUMP EE "
                         "semantics, so hand_force is not wired up here.")
    ap.add_argument("--target", type=int, default=12000,
                    help="valid transitions to collect (default 12000 -- the "
                         "number every per-region and per-bin quota in the "
                         "spec adds up to)")
    ap.add_argument("--smoke", action="store_true",
                    help="a ~200-transition validation run: 10 base poses, "
                         "plots and checks, nothing else changed")
    ap.add_argument("--n-particles", type=int, default=600,
                    help="PF size per segment (the demo executor's own)")
    ap.add_argument("--no-particles", action="store_true",
                    help="store only the Gaussian summary of each belief, not "
                         "the particle sets.  Section 8 asks for the sets; "
                         "this is for a quick shakedown only.")
    ap.add_argument("--belief-prior", type=Path, default=None,
                    help="the empirical covariance pool "
                         "(default out/<room>_belief_prior/"
                         "belief_prior_pool.npz; built if missing)")
    ap.add_argument("--lateral-span", type=float, default=0.20,
                    help="perpendicular jitter, +-metres (default 0.20)")
    ap.add_argument("--yaw-mode", choices=("face_wall", "along_route"),
                    default="face_wall",
                    help="face_wall (default): the heading the route implies "
                         "for a robot that touches the wall it follows -- and "
                         "the only one this controller can press along. "
                         "along_route: the travel direction itself.")
    ap.add_argument("--press-rate", type=float, default=None,
                    help="tip creep [m/s]; default the controller's own "
                         "vf_press_rate_m_s")
    ap.add_argument("--settle-cap", type=float, default=3.0)
    ap.add_argument("--predict-pad", type=float, default=None,
                    help="how far past a target the BALANCER's swept ray may "
                         "reach when it guesses hit/miss (default 0: guess "
                         "conservatively). Never touches the label.")
    ap.add_argument("--true-delta", type=float, default=0.0,
                    help="registration offset [m]: 0 for a dataset, so map "
                         "and world agree")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-attempts", type=int, default=None,
                    help="hard stop, whatever the coverage says")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--out", type=Path, default=None,
                    help="default: out/<room>_touch")
    ap.add_argument("--no-qpos", action="store_true",
                    help="skip the per-transition qpos.npz")
    ap.add_argument("--no-state-log", action="store_true",
                    help="skip states/*.npz -- the per-tick qpos a REPLAY "
                         "needs.  Without it a trajectory can only be "
                         "re-executed, not replayed.")
    ap.add_argument("--state-stride", type=int, default=2,
                    help="log qpos every Nth control tick (default 2 = 25 Hz, "
                         "the demo's own video rate)")
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--report-only", action="store_true",
                    help="re-run the diagnostics on an existing --out, no sim")
    ap.add_argument("--in-venv", action="store_true",
                    help=argparse.SUPPRESS)
    return ap


def project_python() -> str:
    venv = ROOT / "tactile_controller" / ".venv" / "bin" / "python"
    if venv.exists():
        return str(venv)
    print(f"[warn] {venv} not found -- falling back to {sys.executable}.\n"
          f"       Run ./scripts/setup_env.sh first.", file=sys.stderr)
    return sys.executable


def relaunch(argv) -> int:
    cmd = [project_python(), "-u", str(Path(__file__).resolve()),
           "--in-venv", *argv]
    env = dict(os.environ)
    env["PYTHONPATH"] = "src:tactile_controller/src"
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    env["MUJOCO_GL"] = env.get("MUJOCO_GL", "egl")   # headless: no window
    print("+", " ".join(cmd), flush=True)
    proc = subprocess.Popen(cmd, cwd=str(ROOT), env=env)
    while True:
        try:
            return proc.wait()
        except KeyboardInterrupt:
            print("\n[collect] Ctrl-C -- waiting for the summary and plots "
                  "(Ctrl-C again to abandon them)", flush=True)
            try:
                return proc.wait()
            except KeyboardInterrupt:
                proc.terminate()
                return 130


def collect(a) -> int:
    import numpy as np

    from data_collect.action_space import TouchWorkspace
    from data_collect.belief_prior import CovariancePool
    from data_collect.collector import CollectCfg, Collector
    from data_collect.coverage import lateral_bin_edges
    from data_collect.diagnostics import report
    from data_collect.probe import FreeTouchProbe
    from data_collect.records import (BeliefStore, StateLog, TransitionWriter,
                                      config_digest)
    from data_collect.rooms import build_scene, map_identity, planned_centreline
    from data_collect.route import RouteRef

    out = a.out or (ROOT / "out" / f"{a.room}_touch")
    out = Path(out)
    if a.report_only:
        report(out, plots=not a.no_plots)
        return 0

    run_id = a.run_id or f"{a.room}-{uuid.uuid4().hex[:8]}"
    target = 200 if a.smoke else a.target

    scene = build_scene(a.room, a.reverse)
    if a.true_delta:
        import dataclasses
        scene = dataclasses.replace(scene, true_delta_m=float(a.true_delta))
    route = RouteRef.from_polyline(
        planned_centreline(scene, hug_corner=not a.no_hug_corner),
        scene.plan.bodies)
    ws = TouchWorkspace.from_controller()

    # THE BELIEF PRIOR IS MEASURED.  Section 3 forbids choosing a covariance
    # before checking the existing logs, so `b_0` is bootstrapped from what
    # the Room 2 filter really carried, bucketed by route region.
    prior = a.belief_prior or (ROOT / "out" / f"{a.room}_belief_prior"
                               / "belief_prior_pool.npz")
    if not Path(prior).exists():
        print(f"[collect] no belief prior at {prior} -- building it from the "
              f"run records under out/", flush=True)
        import subprocess
        r = subprocess.run([sys.executable, str(ROOT / "scripts"
                                                / "room2_belief_prior.py"),
                            "--room", a.room], cwd=str(ROOT))
        if r.returncode != 0 or not Path(prior).exists():
            print(f"[collect] could not build {prior}; run "
                  f"scripts/room2_belief_prior.py first", file=sys.stderr)
            return 2
    pool = CovariancePool.from_npz(prior)
    print(f"[collect] belief prior: {pool.as_dict()}", flush=True)

    print(f"[collect] {a.room}: route {route.length_m:.2f} m, regions "
          + ", ".join(f"{k} {route.region_length_m(k):.2f} m"
                      for k in route.spans()), flush=True)

    lo_hi = str(a.steps_per_segment).split("-")
    steps_lo, steps_hi = (int(lo_hi[0]), int(lo_hi[-1]))
    prior_rows = []
    if a.append:
        jl = out / "transitions.jsonl"
        if not jl.exists():
            print(f"[collect] --append but no dataset at {jl}",
                  file=sys.stderr)
            return 2
        prior_rows = [json.loads(l) for l in jl.read_text().splitlines()
                      if l.strip()]
        print(f"[collect] appending to {len(prior_rows)} existing transitions",
              flush=True)
    writer = TransitionWriter(out, run_id, store_qpos=not a.no_qpos,
                              append=a.append)

    from execution.backends import BackendConfig
    from execution.vforce_backend import VForceBackend, VForceConfig
    # The demo's 4000 s budget ends a ROUTE; this one has to outlast a
    # DATASET.  12 000 touches at ~7 s of settle, creep and retract each is
    # ~24 h of simulated time, so the budget sits an order above that -- and
    # the collector stops cleanly on it rather than turning every remaining
    # attempt into a `budget` invalid.
    mj = VForceConfig(seed=11 + a.seed, episode_budget_s=5.0e5)
    controller = {
        "name": a.controller, "task": mj.task, "checkpoint": str(mj.checkpoint),
        "press": {"rate_m_s": float(a.press_rate or mj.vf_press_rate_m_s),
                  "intent_n": mj.vf_press_intent_n,
                  "contact_threshold_n": mj.wp_tap_contact_n,
                  "debounce_samples": mj.contact_debounce_samples,
                  "contact_backoff_m": mj.vf_contact_backoff_m,
                  "guard_abort_force_n": mj.guard_abort_force_n,
                  "guard_abort_hold_s": mj.guard_abort_hold_s,
                  "force_sat_n": mj.force_sat_n},
    }
    cfg = CollectCfg(target_n=target, touches_per_pose=steps_hi,
                     n_particles=a.n_particles,
                     store_particles=not a.no_particles,
                     lateral_span_m=a.lateral_span, yaw_mode=a.yaw_mode,
                     seed=a.seed, settle_cap_s=a.settle_cap,
                     max_attempts=a.max_attempts,
                     progress_every=1 if a.smoke else 5,
                     vertex_share=float(a.vertex_share),
                     **({} if a.predict_pad is None
                        else {"predict_pad_m": float(a.predict_pad)}))
    beliefs = BeliefStore(out, enabled=not a.no_particles, append=a.append)
    states = StateLog(out, stride=a.state_stride,
                      enabled=not a.no_state_log, append=a.append)
    map_id = map_identity(a.room, scene, a.reverse)["map_id"]

    if a.n_envs > 1:
        from data_collect.vec_collector import VecCollector
        from data_collect.vec_env import VecCfg, VecTouchEnv
        from data_collect.vec_touch import VecTouchDriver
        env = VecTouchEnv(scene, VecCfg(n_envs=a.n_envs, seed=a.seed), mj)
        drv = VecTouchDriver(env, ws, press_rate_m_s=a.press_rate)
        controller["press"]["control_dt_s"] = env.dt
        controller["n_envs"] = a.n_envs
        controller["config_digest"] = config_digest(controller)
        col = VecCollector(env=env, driver=drv, route=route, workspace=ws,
                           bodies=scene.plan.bodies, writer=writer,
                           beliefs=beliefs, cfg=cfg, run_id=run_id,
                           map_id=map_id, controller=controller, scene=scene,
                           cov_pool=pool,
                           steps_per_segment=(steps_lo, steps_hi),
                           states=states)
        be, probe = env, drv
        if prior_rows:
            col.resume_from(prior_rows)
            print(f"[collect] resumed: segments continue at "
                  f"{col.next_seg_id}, quotas describe {cfg.target_n} "
                  f"combined", flush=True)
        print(f"[collect] {a.n_envs} parallel worlds, dt {env.dt:.3f} s",
              flush=True)
    else:
        be = VForceBackend(scene, BackendConfig(),
                           np.random.default_rng(a.seed), mj,
                           render=False, video_path=None, view=False)
        probe = FreeTouchProbe(be, ws, press_rate_m_s=a.press_rate)
        controller["press"]["control_dt_s"] = be.dt
        controller["n_envs"] = 1
        controller["config_digest"] = config_digest(controller)
        col = Collector(probe=probe, route=route, workspace=ws,
                        bodies=scene.plan.bodies, writer=writer, cfg=cfg,
                        run_id=run_id, map_id=map_id, controller=controller,
                        scene=scene, cov_pool=pool, beliefs=beliefs)

    writer.write_manifest({
        "room": a.room, "target_n": target,
        "cfg": {"steps_per_segment": [steps_lo, steps_hi],
                "n_envs": a.n_envs, "lateral_span_m": a.lateral_span,
                "yaw_mode": a.yaw_mode, "seed": a.seed,
                "n_particles": a.n_particles,
                "hug_corner": not a.no_hug_corner,
                "store_particles": not a.no_particles,
                "state_log": not a.no_state_log,
                "state_stride": a.state_stride,
                "true_delta_m": float(a.true_delta), "smoke": bool(a.smoke)},
        "belief_prior": {"path": str(prior), **pool.as_dict()},
        # the REPLAY format, in the manifest so a consumer knows it up front;
        # the shard list lands in run_state.json once the run has closed them
        "state_log": states.as_dict(),
        "map": map_identity(a.room, scene, a.reverse),
        "route": route.as_dict(),
        "workspace": {**ws.as_dict(), "all_bin_keys": ws.all_bin_keys()},
        "feasible_action_bins": sorted(ws.feasible_bins()),
        "lateral_bin_edges_m": lateral_bin_edges(a.lateral_span),
        "controller": controller,
        "appended_to": len(prior_rows) or None,
        "quotas": {"region": col.cov.region_share,
                   "outcome": {"contact": 0.50, "no_contact": 0.50},
                   "lateral_bin_min_share": 0.15,
                   "action_bin_min": col.cov.action_min()},
    })

    how = "completed"
    try:
        col.run()
    except KeyboardInterrupt:
        how = "interrupted"
        print("\n[collect] Ctrl-C -- finalizing what was collected", flush=True)
    finally:
        writer.close()
        beliefs.close()
        states.close()
        try:
            phys = be.physical_summary()
        except Exception as exc:               # noqa: BLE001
            phys = {"physical_summary_failed": f"{type(exc).__name__}: {exc}"}
        try:
            be.close()
        except Exception as exc:               # noqa: BLE001 - shutdown only
            print(f"[collect] backend close failed: {type(exc).__name__}: "
                  f"{exc}", flush=True)

    # The run state goes down BEFORE the report: it is the record of how the
    # run ended and what the safety stack saw, and it must survive a report
    # that trips over its own data.
    (out / "run_state.json").write_text(json.dumps({
        "how": how, "episodes": col.episode,
        "pose_rejects": col.pose_rejects,
        "coverage": col.cov.as_dict(),
        "pf_updates": col.n_pf_updates,
        "beliefs": beliefs.as_dict(),
        "states": states.as_dict(),
        "physical": phys,
    }, indent=1, default=float))
    s = report(out, plots=not a.no_plots)
    print(f"\n[collect] {how}: {s['totals']['valid']} valid transitions -> "
          f"{out}", flush=True)
    bad = {k: v for k, v in s["checks"].items()
           if v is False or (isinstance(v, list) and v)}
    if bad:
        # Not an error: a short or interrupted run cannot meet quotas sized
        # for 12 000, and a full run that misses one is a result to read, not
        # a crash.  Named here so it is not only in summary.json.
        print(f"[collect] coverage checks not met: {sorted(bad)}", flush=True)
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    a = build_parser().parse_args(argv)
    if not a.in_venv:
        # `--report-only` goes through the venv too: it needs no GPU, but the
        # shebang interpreter is whatever is first on PATH (a conda base here)
        # and the plots die on `No module named matplotlib` while the text
        # report still prints -- a silent half-failure.
        return relaunch(list(argv))
    sys.path.insert(0, str(ROOT / "src"))
    return collect(a)


if __name__ == "__main__":
    raise SystemExit(main())
