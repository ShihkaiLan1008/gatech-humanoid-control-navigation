"""Run a V2 episode (kinematic stand-in now; MuJoCo once assets land).

    PYTHONPATH=src python3 -m lane_graph.run_v2 \
        --map m2_pillar_room --seed 0 --out out/tacnav_v2/kin_m2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from execution.backends import BackendConfig, KinematicBackend
from execution.run_batch3a import _jsonable

from .executive2 import ExecutiveV2, ExecV2Config
from .graph import build_graph
from .scene2 import build_v2_scene


def run_one(map_name: str, seed: int, out: Path, backend: str = "kinematic",
            exec_kw: dict | None = None) -> dict:
    scene = build_v2_scene(map_name, seed=0)      # map seed 0 = approved figs
    graph = build_graph(scene)
    rng = np.random.default_rng(seed)
    out.mkdir(parents=True, exist_ok=True)
    if backend == "kinematic":
        be = KinematicBackend(scene, BackendConfig(), rng)
    else:
        from execution.mujoco_backend import MujocoConfig
        from .mj_backend2 import MujocoBackendV2
        be = MujocoBackendV2(scene, BackendConfig(), rng,
                             MujocoConfig(seed=11 + seed,
                                          episode_budget_s=1400.0),
                             render=False, video_path=None)
    try:
        ex = ExecutiveV2(scene, be, graph,
                         ExecV2Config(seed=seed, **(exec_kw or {})))
        result = ex.run()
        if getattr(be, "state_log", None):
            ts = np.asarray([t for t, _ in be.state_log], dtype=np.float64)
            qs = np.stack([q for _, q in be.state_log]).astype(np.float32)
            np.savez_compressed(out / "states.npz", t=ts, qpos=qs)
    finally:
        be.close()
    (out / "episode.json").write_text(json.dumps(_jsonable(result), indent=1))
    print(f"[{map_name}/s{seed}] {result['outcome']}  "
          f"probes={result['n_probes']} replans={result['replans']}  "
          f"pos_err={result['global_pos_err_m']:.3f} m  "
          f"goal_dist={result['goal_dist_m']:.3f} m  "
          f"path={result['path_len_m']:.1f} m  sim={result['sim_time_s']:.0f}s",
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
