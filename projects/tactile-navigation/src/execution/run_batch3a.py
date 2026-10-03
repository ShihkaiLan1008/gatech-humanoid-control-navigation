"""Run Batch 3A episodes -- physical (MuJoCo/G1) or kinematic.

    # one physical belief-aware episode with video
    PYTHONPATH=src:tactile_controller/src MUJOCO_GL=egl \
      uv run --project tactile_controller python -m execution.run_batch3a \
        --scene two_turns --policy belief_aware --seed 0 \
        --out out/batch3a/two_turns_belief_s0

    # the fast stand-in (no GPU, no simulator)
    python3 -m execution.run_batch3a --backend kinematic --scene ambiguous_near_edge \
        --policy belief_aware --seeds 8 --out /tmp/b3a
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .backends import BackendConfig, KinematicBackend
from .executive import ExecConfig, Executive
from .scenes import SCENES, build_scene


def _jsonable(o):
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


def run_one(scene_name: str, policy: str, seed: int, out: Path, *, backend: str,
            render: bool, particles: int, exec_kw: dict | None = None) -> dict:
    scene = build_scene(scene_name)
    rng = np.random.default_rng(seed)
    out.mkdir(parents=True, exist_ok=True)

    if backend == "mujoco":
        from .mujoco_backend import MujocoBackend, MujocoConfig
        mj_kw = {}
        budget = (exec_kw or {}).get("episode_budget_s")
        if budget:
            # the backend's own budget must outlast the executive's, or long
            # whole-house episodes end as backend 'budget' statuses instead of
            # planner decisions
            mj_kw["episode_budget_s"] = float(budget) + 200.0
        be = MujocoBackend(scene, BackendConfig(), rng,
                           MujocoConfig(seed=11 + seed, **mj_kw),
                           render=render,
                           video_path=(out / "episode.mp4") if render else None)
    else:
        be = KinematicBackend(scene, BackendConfig(), rng)

    try:
        result = Executive(scene, be, ExecConfig(seed=seed, policy=policy,
                                                 n_particles=particles,
                                                 **(exec_kw or {}))).run()
        if hasattr(be, "physical_summary"):
            result["physical"] = be.physical_summary()
            result["tick_trace"] = be.trace
        if getattr(be, "state_log", None):
            ts = np.asarray([t for t, _ in be.state_log], dtype=np.float64)
            qs = np.stack([q for _, q in be.state_log]).astype(np.float32)
            np.savez_compressed(out / "states.npz", t=ts, qpos=qs)
    finally:
        be.close()

    (out / "episode.json").write_text(json.dumps(_jsonable(result), indent=1))
    try:
        from .plots_batch3a import plot_episode
        plot_episode(out, _jsonable(result), scene)
    except Exception as exc:                      # a figure must never lose a run
        print(f"  (figure skipped: {exc})", flush=True)
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene", default="two_turns", choices=sorted(SCENES))
    ap.add_argument("--policy", default="belief_aware",
                    choices=["belief_aware", "no_touch"])
    ap.add_argument("--backend", default="mujoco", choices=["mujoco", "kinematic"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=0,
                    help="run seeds 0..N-1 instead of a single --seed")
    ap.add_argument("--particles", type=int, default=900)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--budget-s", type=float, default=None,
                    help="episode time budget override (whole-house scenes)")
    ap.add_argument("--max-probes", type=int, default=None)
    ap.add_argument("--touch-step-follow", action="store_true",
                    help="enable TOUCH_STEP_FOLLOW maintenance probing")
    args = ap.parse_args(argv)

    exec_kw = {}
    if args.budget_s is not None:
        exec_kw["episode_budget_s"] = args.budget_s
    if args.max_probes is not None:
        exec_kw["max_probes_total"] = args.max_probes
    if args.touch_step_follow:
        exec_kw["touch_step_follow"] = True
    seeds = range(args.seeds) if args.seeds else [args.seed]
    rows = []
    for s in seeds:
        sub = args.out if len(list(seeds)) == 1 else args.out / f"seed{s}"
        r = run_one(args.scene, args.policy, int(s), sub, backend=args.backend,
                    render=not args.no_render, particles=args.particles,
                    exec_kw=exec_kw)
        rows.append(r)
        print(f"[{args.scene}/{args.policy}/s{s}] {r['outcome']}  "
              f"decisions={r['n_route_decisions']}/{r['route_decisions_required']}  "
              f"wrong_turn={r['wrong_turn']}  probes={r['n_probes']}  "
              f"stations={len(r['stations_visited'])}  "
              f"pos_err={r['global_pos_err_m']:.3f} m  "
              f"goal_dist={r['goal_dist_m']:.3f} m  "
              f"peak_force={r['peak_force_n']:.2f} N  "
              f"sim_t={r['sim_time_s']:.0f}s", flush=True)
        if "physical" in r:
            p = r["physical"]
            print(f"    physical: envs={p['environment_instances']} resets={p['reset_count']} "
                  f"max_base_step={p['max_base_step_m']*1000:.1f} mm "
                  f"exclusion_clean={p['exclusion']['clean']}", flush=True)
    if len(rows) > 1:
        agg = {"scene": args.scene, "policy": args.policy, "n": len(rows),
               "arrived": float(np.mean([r["arrived"] for r in rows])),
               "wrong_turn": float(np.mean([r["wrong_turn"] for r in rows])),
               "probes": float(np.mean([r["n_probes"] for r in rows])),
               "pos_err_m": float(np.mean([r["global_pos_err_m"] for r in rows]))}
        (args.out / "aggregate.json").write_text(json.dumps(agg, indent=1))
        print(json.dumps(agg, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
