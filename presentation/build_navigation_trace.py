"""Build a static GitHub figure from a preserved simulation, without running it.

Chart contract: Show how planar localization error varied during the completed
offset-opening seed-0 run, with a separate event strip for actual contacts and
completed misses. Use every recorded sample; break lines at missing values.
Takeaway: the archive permits inspection of the episode beyond its final pose;
event proximity alone cannot establish that a probe caused an error change.
Renderer/surface: Matplotlib PNG + SVG embedded in GitHub Markdown (not HTML).
Palette: two roots, blue and orange, plus neutral axes; circle/x markers and
separate rows distinguish outcomes without relying on color. No partner logo.
Footprint: 11.6 x 5.8 inches; inspect PNG and GitHub rendering before delivery.
"""
from pathlib import Path
import ast
import hashlib
import json
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NAV = ROOT / "projects/tactile-navigation"
OUTPUT = ROOT / "assets"


def main():
    summary = json.loads((NAV / "outputs/sky_results.json").read_text())
    row = next(r for r in summary["evaluation_records"] if r["room"] == "sky_offset_gate")
    source = NAV / row["record"]
    raw = source.read_bytes()
    record = json.loads(raw)
    log_source = source.parent / "run.log"
    plan_line = next(line for line in log_source.read_text().splitlines() if "GLOBAL_PLAN " in line)
    planned = ast.literal_eval(plan_line.split("GLOBAL_PLAN ", 1)[1])
    assert planned["feasible"] and planned["strategy"] == "belief_astar" and planned["strategy_stats"]["fallback"]
    samples, probes = record["samples"], record["probes"]
    t = np.array([s["t"] for s in samples])
    e = np.array([math.dist(s["estimate"][:2], s["truth"][:2]) * 100
                  if s["estimate"] is not None and s["truth"] is not None else np.nan
                  for s in samples])
    hits = [p["t"] for p in probes if p["contact"]]
    misses = [p["t"] for p in probes if not p["contact"] and p["status"] == "complete"]
    interrupted = [p["t"] for p in probes if p["status"] != "complete"]
    assert len(probes) == row["probe_attempts"] == 29
    assert len(hits) == row["touch_contacts"] == 12
    assert len(misses) == row["probe_misses"] == 17
    assert not interrupted
    assert len(samples) == row["trajectory_samples"] == 15483
    dt = np.diff(t)
    valid = (dt > 0) & np.isfinite(e[:-1]) & np.isfinite(e[1:])
    scored_duration = float(dt[valid].sum())
    rmse_cm = float(np.sqrt(np.sum(dt[valid] * (e[:-1][valid]**2 + e[1:][valid]**2) / 2) / scored_duration))
    assert math.isclose(rmse_cm / 100, row["position_error_rmse_m"], abs_tol=1e-9)
    assert math.isclose(scored_duration, row["estimation_scored_duration_s"], abs_tol=1e-8)
    assert np.isfinite(e).all()
    path_length_m = sum(math.dist(a["truth"][:2], b["truth"][:2])
                        for a, b in zip(samples, samples[1:])
                        if b["t"] > a["t"] and a["truth"] is not None and b["truth"] is not None)
    assert math.isclose(path_length_m, row["path_length_m"], abs_tol=1e-8)
    goal_cm = math.dist(samples[-1]["truth"][:2], record["goal_xy"]) * 100
    assert math.isclose(goal_cm / 100, row["final_goal_error_m"], abs_tol=1e-10)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "svg.fonttype": "none", "path.simplify": False,
                         "axes.edgecolor": "#6c7178",
                         "axes.labelcolor": "#28323f", "text.color": "#202a35"})
    fig, axes = plt.subplots(2, 1, figsize=(11.6, 5.8), sharex=True,
                             gridspec_kw={"height_ratios": [3.1, 1], "hspace": .18})
    fig.subplots_adjust(left=.12, right=.97, top=.79, bottom=.22)
    fig.suptitle("Offset-opening run: localization error and probe outcomes", x=.12,
                 y=.97, ha="left", fontsize=15, fontweight="bold")
    fig.text(.12, .89, "Completed simulation · seed 0 · model 35500 · 15,483 samples · 308.40 s", fontsize=10)
    axes[0].plot(t, e, color="#2765a2", linewidth=1.05, label="Planar estimate–truth error")
    axes[0].axhline(rmse_cm, color="#555555", linestyle="--", linewidth=1,
                    label=f"Time-weighted RMSE: {rmse_cm:.2f} cm")
    axes[0].set_ylabel("Localization error (cm)")
    axes[0].set_ylim(bottom=0)
    axes[0].legend(loc="lower left", bbox_to_anchor=(0, 1.025), ncol=2,
                   fontsize=9, frameon=False, borderaxespad=0)
    axes[0].grid(axis="y", alpha=.22)
    axes[1].scatter(hits, [1]*len(hits), marker="o", s=29, facecolors="none", edgecolors="#2765a2")
    axes[1].scatter(misses, [0]*len(misses), marker="x", s=29, color="#be7026", linewidths=1.3)
    axes[1].set_yticks([0, 1], ["Miss (17)", "Contact (12)"])
    axes[1].set_ylim(-.55, 1.55)
    axes[1].set_xlim(0, record["end_t"])
    axes[1].set_xlabel("Simulation time (s)")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.text(.12, .095, "Final true goal distance: 4.07 cm within a 30 cm goal radius. Arrival belief: 91.89% < 95% required.", fontsize=9.5)
    fig.text(.12, .047, "Probe markers use recorded event times. Timing association does not establish a causal effect; a miss can still inform belief.", fontsize=9)
    OUTPUT.mkdir(exist_ok=True)
    fig.savefig(OUTPUT / "offset-opening-trace.png", dpi=180, facecolor="white")
    fig.savefig(OUTPUT / "offset-opening-trace.svg", facecolor="white")
    svg = OUTPUT / "offset-opening-trace.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)
    audit = dict(source=source.relative_to(ROOT).as_posix(), source_sha256=hashlib.sha256(raw).hexdigest(),
                 samples=len(samples), probe_attempts=len(probes), contacts=len(hits), misses=len(misses),
                 simulation_duration_s=record["end_t"], scored_duration_s=scored_duration,
                 actual_planar_path_length_m=path_length_m,
                 planned_route_distance_m=planned["distance_m"],
                 requested_planner_strategy=planned["requested_strategy"],
                 executed_planner_strategy=planned["strategy"],
                 planner_used_fallback=planned["strategy_stats"]["fallback"],
                 plan_log=log_source.relative_to(ROOT).as_posix(),
                 plan_log_sha256=hashlib.sha256(log_source.read_bytes()).hexdigest(),
                 localization_rmse_cm=rmse_cm, final_goal_error_cm=goal_cm,
                 final_arrival_belief=row["final_arrival_probability"],
                 no_new_simulation=True, all_samples_plotted=True,
                 note="Event alignment is descriptive, not an estimate of a probe's causal effect.")
    (ROOT / "presentation/navigation_trace_audit.json").write_text(json.dumps(audit, indent=2)+"\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
