"""Run a V3 contact-rich episode (kinematic stand-in or MuJoCo G1).

    PYTHONPATH=src python3 -m lane_graph.run_v3 --map r2_lbend --out DIR
    ... --backend mujoco   (with MUJOCO_GL=egl under uv)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from execution.backends import BackendConfig, KinematicBackend
from execution.run_batch3a import _jsonable

from .executive3 import ExecutiveV3, ExecV3Config
from .graph3 import build_graph3
from .scene3 import build_v3_scene

PRESS_TARGET_N = 14.0     # mid of the 10-25 N contract window


def _v3_backend(scene, rng, seed: int, render=False):
    """MuJoCo backend for V3: injected room + a force-TARGETED press."""
    from execution.mujoco_backend import MujocoConfig
    from .mj_backend2 import MujocoBackendV2

    class MujocoBackendV3(MujocoBackendV2):
        def _press_target_n(self) -> float:
            return PRESS_TARGET_N

    return MujocoBackendV3(
        scene, BackendConfig(probe_force_n=PRESS_TARGET_N), rng,
        MujocoConfig(seed=11 + seed, episode_budget_s=2000.0,
                     press_overlap_m=0.20),
        render=render, video_path=None)


def run_one(map_name: str, seed: int, out: Path, backend="kinematic",
            exec_kw: dict | None = None) -> dict:
    scene = build_v3_scene(map_name)
    graph = build_graph3(scene)
    rng = np.random.default_rng(seed)
    out.mkdir(parents=True, exist_ok=True)
    be = (KinematicBackend(scene, BackendConfig(probe_force_n=PRESS_TARGET_N), rng)
          if backend == "kinematic" else _v3_backend(scene, rng, seed))
    try:
        ex = ExecutiveV3(scene, be, graph,
                         ExecV3Config(seed=seed, **(exec_kw or {})))
        result = ex.run()
        if getattr(be, "state_log", None):
            ts = np.asarray([t for t, _ in be.state_log], dtype=np.float64)
            qs = np.stack([q for _, q in be.state_log]).astype(np.float32)
            np.savez_compressed(out / "states.npz", t=ts, qpos=qs)
    finally:
        be.close()
    (out / "episode.json").write_text(json.dumps(_jsonable(result), indent=1))
    print(f"[{map_name}/s{seed}] {result['outcome']}  "
          f"goal={result['goal_dist_m']:.3f}m  pos_err={result['global_pos_err_m']:.3f}m  "
          f"steps_TF={result['steps_touch_follow']} "
          f"touched={result['steps_with_valid_touch']} "
          f"({result['touch_coverage']*100:.0f}%)  "
          f"valid={result['n_valid_obs']} in10-25N={result['n_in_window']}  "
          f"peak={result['peak_force_n']:.1f}N  turns={result['turns']}  "
          f"path={result['path_len_m']:.1f}m  sim={result['sim_time_s']:.0f}s",
          flush=True)
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--map", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--backend", default="kinematic")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    run_one(args.map, args.seed, args.out, backend=args.backend)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
