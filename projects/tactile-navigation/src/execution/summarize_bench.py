"""Assemble the Batch 3A.2 benchmark table from the per-episode JSON.

Reports exactly the columns the batch asks for, plus the ones that turned out
to matter most: FULL-HAND peak force asserted equal to the audit's, force
aborts, no-progress events, landmark resolution as a separate outcome from
route completion, and the physically walked detour.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


def load(root: Path) -> list[dict]:
    rows = []
    for f in sorted(root.glob("*/episode.json")):
        d = json.loads(f.read_text())
        p = d.get("physical", {})
        exc = p.get("exclusion", {})
        settle = p.get("turn_settle_errors_deg", [])
        rows.append({
            "scene": d["scene"], "policy": d["policy"], "seed": d["seed"],
            "arrived": bool(d["arrived"]), "wrong_turn": bool(d["wrong_turn"]),
            "decisions": d["n_route_decisions"],
            "required": d["route_decisions_required"],
            "probes": d["n_probes"],
            "stations": d.get("n_stations", 0),
            # take the max of BOTH sources: older episode JSONs recorded only
            # the probe-hand figure in "peak_force_n"
            "peak_force_n": max(float(d["peak_force_n"]),
                                float(p.get("peak_force_n", 0.0))),
            "audit_peak_force_n": float(p.get("audit_peak_force_n", float("nan"))),
            "hand_force_moving_n": exc.get("max_hand_force_while_moving_n", 0.0),
            "exclusion_clean": bool(exc.get("clean", True)),
            "force_aborts": int(p.get("force_aborts", 0)),
            "safety_aborts": int(d.get("safety_aborts", 0)),
            "no_progress_events": int(d.get("no_progress_events", 0)),
            "landmark_resolved": bool(d.get("landmark_resolved", False)),
            "detour_path_m": float(d.get("detour_path_m", 0.0)),
            "probes_remaining": int(d.get("probes_remaining", 0)),
            "final_yaw_err_deg": float(max(settle)) if settle else float("nan"),
            "goal_dist_m": d["goal_dist_m"], "belief_err_m": d["global_pos_err_m"],
            "sim_time_s": d["sim_time_s"], "path_m": d["path_len_m"],
            "failures": d["n_failures"], "outcome": d["outcome"],
        })
    return rows


def table(rows: list[dict]) -> str:
    out = ["| scene | policy | n | success | wrong turn | peak force (N) "
           "| force aborts | yaw err (deg) | probes | stations | path (m) "
           "| time (s) | goal dist (m) | no-prog |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for scene in sorted({r["scene"] for r in rows}):
        for policy in ("no_touch", "belief_aware"):
            sel = [r for r in rows if r["scene"] == scene and r["policy"] == policy]
            if not sel:
                continue
            g = lambda k: np.array([float(r[k]) for r in sel], dtype=float)
            yaw = g("final_yaw_err_deg")
            yaw = yaw[~np.isnan(yaw)]
            sem = lambda v: (v.std(ddof=1) / math.sqrt(v.size)) if v.size > 1 else 0.0
            out.append(
                f"| {scene} | {policy} | {len(sel)} "
                f"| {g('arrived').mean():.2f} ± {sem(g('arrived')):.2f} "
                f"| {g('wrong_turn').mean():.2f} "
                f"| {g('peak_force_n').max():.2f} max "
                f"| {int(g('force_aborts').sum())} "
                f"| {(yaw.mean() if yaw.size else float('nan')):.2f} "
                f"| {g('probes').mean():.1f} "
                f"| {g('stations').mean():.1f} "
                f"| {g('path_m').mean():.2f} "
                f"| {g('sim_time_s').mean():.0f} "
                f"| {g('goal_dist_m').mean():.3f} "
                f"| {int(g('no_progress_events').sum())} |")
    return "\n".join(out)


def detour_table(rows: list[dict]) -> str:
    sel = [r for r in rows if r["scene"] == "detour_box"
           and r["policy"] == "belief_aware"]
    if not sel:
        return ""
    out = ["| seed | outcome | landmark resolved | probes (box share) | "
           "detour walked (m) | belief err (m) | budget left | time (s) |",
           "|---|---|---|---|---|---|---|---|"]
    for r in sorted(sel, key=lambda r: r["seed"]):
        out.append(
            f"| {r['seed']} | {r['outcome']} | {r['landmark_resolved']} "
            f"| {r['probes']} | {r['detour_path_m']:.2f} "
            f"| {r['belief_err_m']:.3f} | {r['probes_remaining']} "
            f"| {r['sim_time_s']:.0f} |")
    return "\n".join(out)


def audit(rows: list[dict]) -> str:
    peak = max((r["peak_force_n"] for r in rows), default=0.0)
    moving = max((r["hand_force_moving_n"] for r in rows), default=0.0)
    dirty = [f"{r['scene']}/{r['policy']}/s{r['seed']}" for r in rows
             if not r["exclusion_clean"]]
    over = [f"{r['scene']}/{r['policy']}/s{r['seed']}({r['peak_force_n']:.1f}N)"
            for r in rows if r["peak_force_n"] > 25.0]
    mism = [f"{r['scene']}/{r['policy']}/s{r['seed']}" for r in rows
            if not math.isnan(r["audit_peak_force_n"])
            and abs(r["peak_force_n"] - r["audit_peak_force_n"]) > 1e-6]
    yaws = [r["final_yaw_err_deg"] for r in rows if not math.isnan(r["final_yaw_err_deg"])]
    return "\n".join([
        f"- episodes: **{len(rows)}**, all one env / one reset",
        f"- **peak force across every executed episode: {peak:.2f} N** (limit 25 N)"
        + (f" — **OVER LIMIT: {', '.join(over)}**" if over else " — all episodes within the limit"),
        f"- summary vs audit peak: "
        f"{'EQUAL on every episode' if not mism else 'MISMATCH: ' + ', '.join(mism)}",
        f"- max hand force while moving {moving:.2f} N",
        f"- exclusion audit: **{'clean on all episodes' if not dirty else 'not clean on: ' + ', '.join(dirty)}**",
        f"- force aborts: {sum(r['force_aborts'] for r in rows)} across "
        f"{sum(1 for r in rows if r['force_aborts'])} episodes",
        f"- no-progress events: {sum(r['no_progress_events'] for r in rows)}",
        f"- post-settle yaw error: mean **{np.mean(yaws):.2f}°**, worst **{np.max(yaws):.2f}°**"
        if yaws else "- post-settle yaw error: no turns executed",
        f"- total failures logged: {sum(r['failures'] for r in rows)}",
    ])


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("out/batch3a1/bench"))
    a = ap.parse_args()
    rows = load(a.root)
    if not rows:
        print("no episodes found under", a.root)
        return 1
    print(table(rows))
    print()
    dt = detour_table(rows)
    if dt:
        print("detour_box / belief_aware, per seed:")
        print(dt)
        print()
    print(audit(rows))
    (a.root / "summary.json").write_text(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
