"""Summarize Sky's local campaign from canonical evaluation records."""
from pathlib import Path
import csv
import hashlib
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from execution.evaluation import compute_metrics
from execution.demo_scenes import load_demo_scene


def revision(path):
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    import numpy as np

    campaign = ROOT / "out/sky_evaluation_20261003"
    output = ROOT / "outputs"
    output.mkdir(exist_ok=True)
    plans = json.loads((campaign / "new_map_plans/summary.json").read_text())
    rows, records, interrupted = [], {}, []
    for suite in ("original_room2", "new_maps_physics", "new_maps_physics_restart"):
        for path in sorted((campaign / suite).glob("*/*/seed_*/*.eval.json")):
            doc = json.loads(path.read_text())
            metrics = compute_metrics(doc)
            row = {"room": doc["metadata"]["room"], "seed": doc["metadata"]["seed"],
                   "controller": doc["metadata"]["controller"], **metrics,
                   "record": str(path.relative_to(ROOT))}
            if metrics["outcome"] == "system_restart":
                interrupted.append(row)
                continue
            replay = path.with_name(path.name.replace(".eval.json", ".json"))
            arrival = None
            if replay.exists():
                transitions = json.loads(replay.read_text()).get("transitions", [])
                checks = [t["arrival"] for t in transitions if t.get("arrival") is not None]
                arrival = checks[-1] if checks else None
            row["final_arrival_probability"] = None if arrival is None else arrival["probability"]
            row["required_arrival_probability"] = None if arrival is None else arrival["required_probability"]
            rows.append(row)
            records[row["room"]] = doc
    source_files = [ROOT / "src/execution/demo_scenes.py", *sorted((ROOT / "maps").glob("sky_*.json"))]
    checkpoint = ROOT / "models/sky_local/model_35500.pt"
    provenance = {"navigation_commit": revision(ROOT), "controller_commit": revision(ROOT / "tactile_controller"),
                  "checkpoint": str(checkpoint.resolve()),
                  "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                  "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in source_files}, "evaluation_records": rows,
                  "interrupted_attempts": interrupted}
    (output / "sky_results.json").write_text(json.dumps(provenance, indent=2) + "\n")
    if rows:
        with (output / "sky_results.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    names = ["room2", "sky_island_choice", "sky_offset_gate", "sky_staggered_baffles"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    for ax, name in zip(axes.flat, names):
        scene = load_demo_scene(name, .30).scene
        for body in scene.plan.bodies:
            v = np.asarray(body.verts)
            ax.fill(v[:, 0], v[:, 1], color="#c8ced6", edgecolor="#647080", linewidth=.6)
        doc = records.get(name)
        if doc:
            actual = np.array([s["truth"][:2] for s in doc["samples"] if s["truth"] is not None])
            estimated = np.array([s["estimate"][:2] for s in doc["samples"] if s["estimate"] is not None])
            if len(actual): ax.plot(*actual.T, color="#153e63", linewidth=1.5, label="True path")
            if len(estimated): ax.plot(*estimated.T, color="#008c95", linewidth=1, alpha=.8, label="Estimated path")
            metric = compute_metrics(doc)
            ax.set_title(f"{name}\n{metric['outcome']}; goal={metric['goal_reached']}, fall={metric['fallen']}", fontsize=10)
        else:
            ax.set_title(f"{name}\nPhysical trial pending", fontsize=10)
        ax.plot(*scene.start_map_xy, "s", color="#13814d", markersize=7, label="Start")
        ax.plot(*scene.goal_map_xy, "*", color="#c63b36", markersize=11, label="Goal")
        ax.add_patch(Circle(scene.goal_map_xy, scene.goal_radius_m, fill=False, color="#c63b36", linewidth=1))
        ax.set_aspect("equal")
        ax.set_xlabel("Map x [m]")
        ax.set_ylabel("Map y [m]")
        ax.legend(fontsize=7, loc="best")
        ax.grid(alpha=.15)
    fig.suptitle("Tactile navigation: original and newly authored maps\nExecuted simulation trajectories; seed 0", fontsize=15)
    fig.savefig(output / "sky_navigation_trajectories.png", dpi=160)
    plt.close(fig)

    lines = ["# Sky: tactile-navigation experiment report", "",
             "Prepared 3 October 2026, Asia/Taipei. This report concerns the current navigation-map mission.", "",
             "## Outcome", "",
             f"Three new maps passed the global planner's feasibility checks. {len(rows)} physical trials have completed. "
             f"Of those, {sum(r['goal_reached'] is True for r in rows)} ended inside the physical goal region, "
             f"{sum(r['success'] is True for r in rows)} met framework success, and {sum(r['fallen'] for r in rows)} reported a fall. "
             "These are individual pilot trials, not a population success-rate estimate.", "",
             "## Physical results", "",
             "| Map | Goal reached | Framework success | Fell | Outcome | Goal error [m] | Localization RMSE [m] | Arrival probability | Contacts/attempts | Sim time [s] |",
             "|---|---|---|---|---|---:|---:|---:|---:|---:|"]
    for name in names:
        row = next((r for r in rows if r["room"] == name), None)
        if row:
            probability = ("Unknown" if row["final_arrival_probability"] is None
                           else f"{100*row['final_arrival_probability']:.2f}%")
            lines.append(f"| {name} | {row['goal_reached']} | {row['success']} | {row['fallen']} | {row['outcome']} | "
                         f"{row['final_goal_error_m']:.3f} | {row['position_error_rmse_m']:.3f} | "
                         f"{probability} | {row['touch_contacts']}/{row['probe_attempts']} | {row['duration_s']:.1f} |")
        else:
            lines.append(f"| {name} | Pending | Pending | Pending | Pending | — | — | — | — | — |")
    lines += ["", "`goal_reached` checks the final true position against the goal disk. Framework success also requires "
              "reported completion and no fall. Localization RMSE compares estimated position with simulator truth; it is not "
              "footstep error. True state is used for evaluation, not supplied to navigation. Preserve arrival uncertainty and failures.", "",
              "The arrival guard sums particle-filter weight jointly inside the goal disk and within the final yaw bound. "
              "It requires at least 95% mass. Original room2 had 93.35% and island choice 92.41%, explaining their "
              "`arrival_uncertain` outcomes despite physical goal membership. Offset gate had 91.89%, while staggered "
              "baffles reached 96.04% and reported `route_complete`. These probabilities come from the saved "
              "runtime transitions, not a Gaussian approximation. The experiment does not loosen the threshold to make results pass.", "",
              "## New maps and planner checks", "",
              "| Map | Test purpose | Planned distance [m] | Strategy used | Feasible |", "|---|---|---:|---|---|"]
    purposes = {"sky_offset_gate": "Detour through an offset opening", "sky_staggered_baffles": "Repeated turns and accumulated drift",
                "sky_island_choice": "Route around a central island"}
    # global_plan.py writes the per-room summary as a list.
    for p in plans:
        lines.append(f"| {p['room']} | {purposes[p['room']]} | {p['distance_m']:.2f} | {p['strategy']} | {p['feasible']} |")
    lines += ["", "A feasible route passed segment clearance, uncertainty bounds, and the goal confidence check. "
              "Planner feasibility is separate from actual G1 execution.", "",
              "## What was done", "",
              "- Cloned the requested `sky/evaluation` branch into a separate checkout.",
              "- Read the README, evaluation instructions, map loader, and physical controller code.",
              "- Connected the existing WSL controller repository/environment without overwriting its training artifacts.",
              "- Added three JSON maps and registered them with the existing loader.",
              "- Passed 12 focused geometry/import tests, including endpoint clearance and planner/simulator obstacle agreement.",
              "- Ran global planning on all three new maps with the shared default configuration.",
              "- Ran the actual learned G1 controller and recorded robot states, commands, localization, probes, and termination outcomes.",
              "- Produced CSV/JSON results, trajectory figures, a launcher, and replay videos when rendering completed.", "",
              "## Settings and limits", "",
              "GPU: NVIDIA GeForce RTX 5070 Laptop GPU, 8 GiB. WSL Ubuntu 24.04. "
              "Controller: `footstep`; seed 0; PF + EIG global navigation; position bound/trigger 0.18 m; yaw bound 20 degrees; "
              "standoff 0.30 m; grid 0.10 m; maximum planner move 0.375 m; route-aligned spawn; common angle-dependent reach profile. "
              "The original room2 pilot has a 240 s simulation cap and new maps 900 s. Room2 ended before its cap. "
              "Avoid pooling differently capped trials as equal-budget robustness statistics.", "",
              "The README's default model 34000 is missing from the downloaded checkout. Every local run instead uses the existing "
              "team `foot_mimic_noforce_arm_model_35500.pt`, explicitly recorded as an override. "
              "Results are not an exact reproduction of the team's default-checkpoint experiments. "
              "No training, cloud upload, cluster submission, or GitHub publishing was performed.", "",
              "## System restart and recovery", "",
              "Windows rebooted after a recorded 0x139 bugcheck with parameter 1 equal to 3. "
              "The error identifies kernel list corruption, not its responsible driver. "
              "The crash dump exists at `C:\\Windows\\Minidump\\100326-22312-01.dmp`, "
              "but this session cannot read it because Windows denies access. No driver culprit or permanent fix is claimed. "
              "The Windows event record is saved under `outputs/crash_diagnostics/`. "
              "Completed room2 and island-choice results and videos were preserved. "
              "The interrupted offset-gate telemetry was recovered with outcome `system_restart` and is recorded separately in "
              "`sky_results.json`; it is not presented as a navigation failure or successful episode. "
              "Offset gate and staggered baffles restarted in `new_maps_physics_restart/` with the same navigation settings and seed. "
              "The restarted campaign uses two CPU cores and a 4 GiB process RSS watchdog, with no concurrent video rendering. "
              "These are precautionary resource changes, not proof that excessive load caused the kernel crash.", "",
              "One seed per map does not establish robustness. Follow-up should repeat the same settings on several seeds, "
              "then vary corridor width and obstacle placement systematically without silently tuning per map. "
              "Inspect failures using localization, recovery, contacts, and video before changing the framework.", "",
              "## Inspect and reproduce", "",
              "![Executed trajectories](sky_navigation_trajectories.png)", "",
              "Metrics and provenance: `sky_results.csv`, `sky_results.json`. Original records and videos: "
              "`../out/sky_evaluation_20261003/original_room2/`, `../out/sky_evaluation_20261003/new_maps_physics/`, "
              "and `../out/sky_evaluation_20261003/new_maps_physics_restart/`. "
              "Planner plots: `../out/sky_evaluation_20261003/new_map_plans/<map>/route.png`.", "",
              "```bash", "bash scripts/sky_evaluate.sh test", "bash scripts/sky_evaluate.sh plan out/repeat_plans",
              "bash scripts/sky_evaluate.sh run out/repeat_physics", "bash scripts/sky_evaluate.sh render out/repeat_physics", "```", "",
              "The launcher uses the existing WSL environment and ignored controller/checkpoint links; setup details are in "
              "`docs/SKY_MAP_EXPERIMENT.md`. Each run needs a fresh output directory. Replay video samples recorded physical "
              "states rather than re-running the controller.", "",
              "## Credits and application use", "",
              "Underlying navigation and controller framework: Huaze Liu, Zhenyu Wu, and the research team. "
              "This extension supplies new map designs, registration, test execution, evidence collection, and analysis, "
              "prepared with Codex assistance for Sky's review. It does not claim original authorship of the team's framework. "
              "The user's screenshots permit inclusion of team work in a report; the user separately reports GitHub permission. "
              "Preserve attribution and the actual permission scope when selecting public materials.", "",
              f"Navigation base commit: `{provenance['navigation_commit']}`. Controller base commit: `{provenance['controller_commit']}`. "
              "The metrics JSON records checkpoint and map/source SHA-256 hashes.", ""]
    (output / "Sky Navigation Experiment Report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"completed_trials": len(rows), "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
