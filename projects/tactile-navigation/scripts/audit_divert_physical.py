"""Subtask 1 -- reproduce r5 diversion failures in MuJoCo and explain them.

    PYTHONPATH=src:tactile_controller/src MUJOCO_GL=egl \
      tactile_controller/.venv/bin/python scripts/audit_divert_physical.py \
      --scene r5_twinbay --seed 0 --out out/hw/divert_audit

One episode per invocation (one env, one reset).  Prints the per-attempt record
Subtask 1 asks for and the aggregate that answers the question.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from execution.backends import BackendConfig, KinematicBackend      # noqa: E402
from lane_graph.divert_audit import DivertAuditExec                 # noqa: E402
from lane_graph.executive3 import ExecV3Config                      # noqa: E402
from lane_graph.graph3 import build_graph3                          # noqa: E402
from lane_graph.run_v3 import PRESS_TARGET_N, _v3_backend           # noqa: E402
from lane_graph.scene3 import build_v3_scene                        # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="r5_twinbay")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--backend", default="mujoco", choices=("mujoco", "kinematic"))
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)

    scene = build_v3_scene(a.scene)
    graph = build_graph3(scene)
    rng = np.random.default_rng(a.seed)
    be = (KinematicBackend(scene, BackendConfig(probe_force_n=PRESS_TARGET_N), rng)
          if a.backend == "kinematic" else _v3_backend(scene, rng, a.seed))
    try:
        ex = DivertAuditExec(scene, be, graph, ExecV3Config(seed=a.seed))
        res = ex.run()
    finally:
        be.close()

    a.out.mkdir(parents=True, exist_ok=True)
    tag = f"{a.scene}_s{a.seed}_{a.backend}"
    (a.out / f"{tag}.json").write_text(json.dumps(
        {"attempts": ex.attempts,
         "outcome": res["outcome"], "arrived": res["arrived"],
         "arrived_truly": res.get("arrived_truly"),
         "belief_err_m": res["global_pos_err_m"],
         "n_diversions": res.get("n_diversions", 0)},
        indent=1, default=float))

    print(f"\n# diversion audit -- {tag}")
    print(f"outcome {res['outcome']}  arrived={res['arrived']}"
          f" truly={res.get('arrived_truly')}  err={res['global_pos_err_m']:.3f} m\n")
    for r in ex.attempts:
        g = r["goto"]
        print(f"attempt t={r['t']:.1f}s  edge e{r['edge']}"
              f"{' INFORMATIVE' if r['informative'] else ''}"
              f"  value {r['value']:.3f}  travel {r['travel_m']:.2f} m")
        print(f"  RESULT              {r['result']}"
              f"   (force {r.get('force_n')})")
        print(f"  goto                status={g.get('status')!r}"
              f"  tol={g.get('tol_m')}  attractor floor={r['attractor_floor_m']:.3f} m")
        print(f"  stance planned      {np.round(r['stance_planned'], 3)}"
              f"   clearance {r['stance_clearance_m']:.3f} m")
        print(f"  base BELIEVED after {np.round(r['bel_after'], 3)}"
              f"   err vs stance {r['bel_stance_err_m']:.3f} m")
        print(f"  base REALIZED after {np.round(r['truth_after_map'], 3)}"
              f"   err vs stance {r['truth_stance_err_m']:.3f} m")
        print(f"  yaw planned {r['yaw_planned']:+.3f} realized {r['yaw_realized']:+.3f}"
              f"  err {r['yaw_err_rad']:+.3f} rad")
        print(f"  surface range from planned {r['reach_from_planned_m']}"
              f"  from REALIZED {r['reach_from_realized_m']}"
              f"  (reach limit {r['reach_limit_m']})")
        print(f"  REACH MARGIN        {r['reach_margin_m']}\n")

    n = len(ex.attempts)
    if n:
        ok = [r for r in ex.attempts if r["result"] == "contact"]
        bad = [r for r in ex.attempts if r["result"] != "contact"]
        print(f"SUMMARY  {len(ok)}/{n} delivered contact")
        st = [r["goto"].get("status") for r in ex.attempts]
        print(f"  goto statuses      {dict((s, st.count(s)) for s in set(st))}")
        tm = [r["truth_stance_err_m"] for r in ex.attempts]
        bm = [r["bel_stance_err_m"] for r in ex.attempts]
        print(f"  |realized - planned stance|  median {np.median(tm):.3f} m,"
              f" max {max(tm):.3f}")
        print(f"  |believed - planned stance|  median {np.median(bm):.3f} m")
        mg = [r["reach_margin_m"] for r in ex.attempts
              if r["reach_margin_m"] is not None]
        if mg:
            print(f"  reach margin at the REALIZED pose: median {np.median(mg):+.3f} m,"
                  f" min {min(mg):+.3f}  ({sum(1 for m in mg if m < 0)} of {len(mg)}"
                  f" out of reach)")
        yw = [r["yaw_err_rad"] for r in ex.attempts]
        print(f"  yaw error          median {np.median(yw):.3f} rad")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
