"""Localization-critical benchmark for the V3 lane-graph stack.

    PYTHONPATH=src python3 scripts/bench_v3.py --seeds 20 --out out/bench_v3

Scored on the quantity `PF_AUDIT.md` 2 established is the real one.  The old
harness stopped the episode when the TRUE pose entered the goal radius, which
made every belief error invisible provided the robot's true path happened to
pass close enough; `arrived` now means "the robot concluded it was there" and
`arrived_truly` is the truth-scored answer.  The gap between them is what this
programme has to close, so it is the headline column here.

A scene is called LOCALIZATION-CRITICAL when tactile localization decides the
outcome on it -- measured, not asserted, by running the same seeds with the
probe budget set to zero.  Where open-loop odometry arrives just as often, the
touching is not what the scene tests, and tuning against it teaches nothing.

Three arms per (scene, seed):

    full        touch-follow + INSPECT      (the shipped stack, today)
    no_inspect  touch-follow only           (the R2/R3 demo cadence)
    no_touch    probe budget 0              (open-loop odometry)
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from execution.backends import BackendConfig, KinematicBackend   # noqa: E402
from lane_graph.executive3 import ExecutiveV3, ExecV3Config      # noqa: E402
from lane_graph.graph3 import build_graph3                       # noqa: E402
from lane_graph.maps_v3 import MAPS3                             # noqa: E402
from lane_graph.run_v3 import PRESS_TARGET_N                     # noqa: E402
from lane_graph.scene3 import build_v3_scene                     # noqa: E402

ARMS = {
    # A: the frozen classical baseline
    "baseline": dict(inspect=False, skip_uninformative_probes=True, divert=False),
    # B: + divert to geometry that can separate the live hypotheses
    "divert":   dict(inspect=False, skip_uninformative_probes=True, divert=True),
    # the ablation that says whether any of the touching mattered
    "no_touch": dict(inspect=False, skip_uninformative_probes=False, divert=False,
                     max_probes_total=0),
}


def run_one(name: str, seed: int, **kw) -> dict:
    sc = build_v3_scene(name)
    be = KinematicBackend(sc, BackendConfig(probe_force_n=PRESS_TARGET_N),
                          np.random.default_rng(seed))
    try:
        r = ExecutiveV3(sc, be, build_graph3(sc), ExecV3Config(seed=seed, **kw)).run()
    finally:
        be.close()
    hyp = r.get("hypothesis_trace") or [{}]
    last = hyp[-1]
    masses = last.get("masses") or [1.0]
    return {"scene": name, "seed": seed,
            "false_arrival": bool(r["arrived"] and not r.get("arrived_truly")),
            "wrong_turn": bool(r.get("wrong_turn")),
            "n_diversions": int(r.get("n_diversions", 0)),
            "path_len_m": float(r.get("path_len_m", float("nan"))),
            "minor_mass": float(1.0 - max(masses)),
            "n_modes": len(masses),
            "arrived": bool(r["arrived"]),
            "arrived_truly": bool(r.get("arrived_truly")),
            "goal_dist_m": float(r.get("goal_dist_m", float("nan"))),
            "belief_err_m": float(r["global_pos_err_m"]),
            "n_probes": int(r["n_probes"]),
            "n_inspects": int(r.get("n_inspects", 0)),
            "sim_time_s": float(r["sim_time_s"]),
            "outcome": str(r["outcome"])}


def _med(rows, key):
    v = [r[key] for r in rows if np.isfinite(r[key])]
    return float(statistics.median(v)) if v else float("nan")


def summarize(rows, arm, scene=None):
    sel = [r for r in rows if r["arm"] == arm and (scene is None or r["scene"] == scene)]
    if not sel:
        return None
    return {"n": len(sel),
            "arrived": sum(r["arrived"] for r in sel),
            "arrived_truly": sum(r["arrived_truly"] for r in sel),
            "false_arrival": sum(r["false_arrival"] for r in sel),
            "wrong_turn": sum(r["wrong_turn"] for r in sel),
            "diversions": sum(r["n_diversions"] for r in sel),
            "minor_mass": _med(sel, "minor_mass"),
            "path_len_m": _med(sel, "path_len_m"),
            "belief_err_m": _med(sel, "belief_err_m"),
            "goal_dist_m": _med(sel, "goal_dist_m"),
            "n_probes": _med(sel, "n_probes"),
            "sim_time_s": _med(sel, "sim_time_s"),
            "inspects": sum(r["n_inspects"] for r in sel)}


def report(rows, seeds: int) -> str:
    scenes = list(dict.fromkeys(r["scene"] for r in rows))
    L = ["# V3 localization benchmark", "",
         f"{len(scenes)} scenes x {seeds} seeds x {len(ARMS)} arms, kinematic backend.",
         "", "`arrived` is the robot's own conclusion; `arrived_truly` is the",
         "truth-scored answer.  The gap between the two columns is the error the",
         "belief is carrying at the moment it decides it is finished.", ""]

    L += ["## Per scene", "",
          "| scene | arm | truly | FALSE arr | wrong turn | belief err | minor mass |"
          " probes | diversions | path | sim |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    critical = []
    for sc in scenes:
        for arm in ARMS:
            s = summarize(rows, arm, sc)
            L.append(f"| {sc} | `{arm}` | **{s['arrived_truly']}/{s['n']}** |"
                     f" **{s['false_arrival']}** | {s['wrong_turn']} |"
                     f" {s['belief_err_m']:.3f} m | {s['minor_mass']:.3f} |"
                     f" {s['n_probes']:.0f} | {s['diversions']} |"
                     f" {s['path_len_m']:.1f} m | {s['sim_time_s']:.0f} s |")
        full = summarize(rows, "divert", sc)
        blind = summarize(rows, "no_touch", sc)
        if full["arrived_truly"] > blind["arrived_truly"]:
            critical.append((sc, full["arrived_truly"], blind["arrived_truly"]))

    L += ["", "## Overall", "",
          "| arm | truly | FALSE arrivals | median belief err | median minor mass |"
          " median probes | diversions | median sim |",
          "|---|---|---|---|---|---|---|---|"]
    for arm in ARMS:
        s = summarize(rows, arm)
        L.append(f"| `{arm}` | **{s['arrived_truly']}/{s['n']}** |"
                 f" **{s['false_arrival']}/{s['n']}** | {s['belief_err_m']:.3f} m |"
                 f" {s['minor_mass']:.3f} | {s['n_probes']:.0f} | {s['diversions']} |"
                 f" {s['sim_time_s']:.0f} s |")

    L += ["", "## Which scenes does localization actually decide?", ""]
    if critical:
        L.append("Touch beats open-loop odometry on these, so they are the ones"
                 " worth tuning against:")
        L.append("")
        L.append("| scene | truly arrived, full | truly arrived, no_touch |")
        L.append("|---|---|---|")
        for sc, a, b in critical:
            L.append(f"| **{sc}** | {a} | {b} |")
    else:
        L.append("None.  On every scene here, open-loop odometry arrives as often"
                 " as the tactile stack -- so this scene set does not yet measure"
                 " what the programme is for.")
    others = [sc for sc in scenes if sc not in {c[0] for c in critical}]
    if others:
        L += ["", "Not localization-critical (open-loop does as well or better): "
              + ", ".join(f"`{s}`" for s in others) + ".  A change that only moves"
              " these numbers has not been shown to help."]
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--scenes", nargs="*", default=list(MAPS3))
    ap.add_argument("--out", type=Path, default=Path("out/bench_v3"))
    a = ap.parse_args(argv)

    rows = []
    for name in a.scenes:
        for seed in range(a.seeds):
            for arm, kw in ARMS.items():
                r = run_one(name, seed, **kw)
                r["arm"] = arm
                rows.append(r)
        print(f"  {name}: done", flush=True)

    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "rows.json").write_text(json.dumps(rows, indent=2))
    txt = report(rows, a.seeds)
    (a.out / "bench_v3.md").write_text(txt)
    print(txt)
    print(f"wrote {a.out}/bench_v3.md, rows.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
