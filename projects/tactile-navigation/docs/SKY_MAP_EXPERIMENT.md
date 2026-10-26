# Sky: navigation on new simulation maps

## Assignment and contribution

The supplied September Slack screenshots ask Sky to first run the existing navigation framework on its original maps with a UI or video, then create new maps to test generality and robustness. This is a navigation experiment, separate from the earlier foot-tracking reward task.

The navigation framework, planner, particle filter, EIG sensing, robot controller, and map manifest format are the team's foundation. This extension adds three authored JSON maps, registry entries, geometry checks, a local evaluation launcher, and a report of experiments. It does not modify or retrain the team's navigation policy or tune a different controller for each room. Sky led this assigned evaluation workstream and participated in interpreting and reporting its results, using Codex for substantial map/script implementation, simulation operation and calculations. The team's underlying framework remains credited to its authors.

| Map | New geometry | Research question |
|---|---|---|
| `sky_offset_gate` | One divider, with an opening at its northern end | Does navigation handle an indirect route through an offset opening? |
| `sky_staggered_baffles` | Alternating south/north dividers | Do repeated turns and accumulated odometry drift break navigation? |
| `sky_island_choice` | Solid central rectangular island | Does navigation select and execute a route around an obstacle with two alternatives? |

All dimensions and endpoints are in the authored room frame in metres. The existing manifest loader transforms the robot's start to the origin and its authored heading onto +y. Island choice deliberately uses an eastward authored heading, exercising that transform. The maps have non-intersecting, physically clear starts and goals. The central box is a solid obstacle, not a wall ring.

## Settings

Use the existing default global pipeline: 0.18 m position bound and execution trigger, 20 degree yaw bound, 0.30 m standoff, 0.10 m grid, 0.375 m maximum planner movement, seed/refine with belief-A* fallback, route-aligned initial heading, PF localization, and EIG probes. Enable the same `configs/probe_reach_ee_qpos_v1.json` on every physical run. Use seed 0 for initial physical trials; several further seeds are needed for robustness estimates.

The README's default `models/0911_footmimic_noforce/model_34000.pt` is absent from this checkout. Initial local experiments therefore explicitly override the checkpoint to the existing team `foot_mimic_noforce_arm_model_35500.pt` from the separate tactile_controller checkout. Every run saves this override. Do not call these an exact reproduction of the team's default-checkpoint campaign. Use the same checkpoint for all local comparisons.

Local WSL links point `tactile_controller` to the existing controller repository and `models/sky_local/model_35500.pt` to that checkpoint. They are ignored by Git. The verified Python environment is `/home/sky/.venvs/tactile_controller/bin/python`; pytest was added there to run the geometry checks. No model training or cluster submission is involved.

## Checks and outputs

1. Map tests check endpoint clearance, frame transformation, and matching obstacle counts in the analytic planner and simulator.
2. `global_plan.py` checks uncertainty bounds, collision-free segments, and the goal confidence disk. A route is not evidence of physical navigation success.
3. `punch_eval.py` executes the actual GPU footstep controller and saves replay states, per-run records, logs, summary JSON, and CSV.
4. `render_eval.py` renders those recorded robot poses in MuJoCo alongside estimated/true trajectories. The video is a replay of the executed run, not a second simulation.

Report `goal_reached`, framework `success`, termination outcome, and falls separately. A robot can physically enter the goal region while the framework declines to report completion. Timeouts and failures must remain in the results. Single-seed trials establish only what happened in those trials.

The first original-room `room2` pilot uses a 240-second simulation cap. New-map trials use 900 seconds. Record that difference; do not pool them into a claim of equal-budget success rates. The launcher below uses a common 900-second cap for future baseline/new-map comparisons.

## Reproduce locally

From WSL in this repository:

```bash
bash scripts/sky_evaluate.sh test
bash scripts/sky_evaluate.sh plan out/sky_plans_repeat
bash scripts/sky_evaluate.sh run out/sky_physics_repeat
bash scripts/sky_evaluate.sh render out/sky_physics_repeat
```

Use a fresh output folder on each run. Set `SKY_PYTHON` to another compatible controller Python environment if needed. The commands require the ignored controller/checkpoint links described above; they are not standalone dependencies bundled into this map extension.

The renderer's stride 10 samples every tenth recorded pose, preserving simulation time at a lower video frame rate. Results from the initial campaign are saved under `out/sky_evaluation_20261003/` and will be summarized in `outputs/Sky Navigation Experiment Report.md`.

After a Windows kernel crash interrupted offset gate, the remaining two maps restarted in `new_maps_physics_restart/`. Completed trials and videos remain in their original folders. The launcher now limits execution to two CPU cores with a 4 GiB RSS watchdog. Run simulation and rendering sequentially; these resource precautions do not identify or repair the underlying Windows driver fault.

## College application presentation

Use a `tactile-navigation-map-study` case study with the research question, original versus new map images, actual navigation videos, comparable settings, measured outcomes, and an explanation of failures. Credit Huaze Liu, Zhenyu Wu, and the team for the underlying framework. Describe Sky's direction of the assigned evaluation, team discussions, work with the recorded data, and reporting alongside the substantial AI implementation and operation assistance. Do not claim independent authorship of code or analyses that Sky cannot explain. The screenshot grants inclusion of team work in a report; the user separately reports permission to show it on GitHub. Preserve authorship and document the actual scope before publishing team source.
