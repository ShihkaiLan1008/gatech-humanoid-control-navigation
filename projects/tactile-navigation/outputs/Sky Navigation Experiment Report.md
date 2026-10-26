# Sky: tactile-navigation experiment report

Prepared 3 October 2026, Asia/Taipei. This report concerns the current navigation-map mission.

## Outcome

Three new maps passed the global planner's feasibility checks. 4 physical trials have completed. Of those, 4 ended inside the physical goal region, 1 met framework success, and 0 reported a fall. These are individual pilot trials, not a population success-rate estimate.

## Physical results

| Map | Goal reached | Framework success | Fell | Outcome | Goal error [m] | Localization RMSE [m] | Arrival probability | Contacts/attempts | Sim time [s] |
|---|---|---|---|---|---:|---:|---:|---:|---:|
| room2 | True | False | False | arrival_uncertain | 0.073 | 0.035 | 93.35% | 13/17 | 187.5 |
| sky_island_choice | True | False | False | arrival_uncertain | 0.064 | 0.066 | 92.41% | 12/23 | 256.7 |
| sky_offset_gate | True | False | False | arrival_uncertain | 0.041 | 0.066 | 91.89% | 12/29 | 308.4 |
| sky_staggered_baffles | True | True | False | route_complete | 0.042 | 0.068 | 96.04% | 17/39 | 423.9 |

`goal_reached` checks the final true position against the goal disk. Framework success also requires reported completion and no fall. Localization RMSE compares estimated position with simulator truth; it is not footstep error. True state is used for evaluation, not supplied to navigation. Preserve arrival uncertainty and failures.

The arrival guard sums particle-filter weight jointly inside the goal disk and within the final yaw bound. It requires at least 95% mass. Original room2 had 93.35% and island choice 92.41%, explaining their `arrival_uncertain` outcomes despite physical goal membership. Offset gate had 91.89%, while staggered baffles reached 96.04% and reported `route_complete`. These probabilities come from the saved runtime transitions, not a Gaussian approximation. The experiment does not loosen the threshold to make results pass.

## New maps and planner checks

| Map | Test purpose | Planned distance [m] | Strategy used | Feasible |
|---|---|---:|---|---|
| sky_offset_gate | Detour through an offset opening | 9.80 | belief_astar | True |
| sky_staggered_baffles | Repeated turns and accumulated drift | 12.18 | seed_refine | True |
| sky_island_choice | Route around a central island | 8.18 | seed_refine | True |

A feasible route passed segment clearance, uncertainty bounds, and the goal confidence check. Planner feasibility is separate from actual G1 execution.

## What was done

- Cloned the requested `sky/evaluation` branch into a separate checkout.
- Read the README, evaluation instructions, map loader, and physical controller code.
- Connected the existing WSL controller repository/environment without overwriting its training artifacts.
- Added three JSON maps and registered them with the existing loader.
- Passed 12 focused geometry/import tests, including endpoint clearance and planner/simulator obstacle agreement.
- Ran global planning on all three new maps with the shared default configuration.
- Ran the actual learned G1 controller and recorded robot states, commands, localization, probes, and termination outcomes.
- Produced CSV/JSON results, trajectory figures, a launcher, and replay videos when rendering completed.

## Settings and limits

GPU: NVIDIA GeForce RTX 5070 Laptop GPU, 8 GiB. WSL Ubuntu 24.04. Controller: `footstep`; seed 0; PF + EIG global navigation; position bound/trigger 0.18 m; yaw bound 20 degrees; standoff 0.30 m; grid 0.10 m; maximum planner move 0.375 m; route-aligned spawn; common angle-dependent reach profile. The original room2 pilot has a 240 s simulation cap and new maps 900 s. Room2 ended before its cap. Avoid pooling differently capped trials as equal-budget robustness statistics.

The README's default model 34000 is missing from the downloaded checkout. Every local run instead uses the existing team `foot_mimic_noforce_arm_model_35500.pt`, explicitly recorded as an override. Results are not an exact reproduction of the team's default-checkpoint experiments. No training, cloud upload, cluster submission, or GitHub publishing was performed.

## System restart and recovery

Windows rebooted after a recorded 0x139 bugcheck with parameter 1 equal to 3. The error identifies kernel list corruption, not its responsible driver. The crash dump exists at `C:\Windows\Minidump\100326-22312-01.dmp`, but this session cannot read it because Windows denies access. No driver culprit or permanent fix is claimed. The Windows event record is saved under `outputs/crash_diagnostics/`. Completed room2 and island-choice results and videos were preserved. The interrupted offset-gate telemetry was recovered with outcome `system_restart` and is recorded separately in `sky_results.json`; it is not presented as a navigation failure or successful episode. Offset gate and staggered baffles restarted in `new_maps_physics_restart/` with the same navigation settings and seed. The restarted campaign uses two CPU cores and a 4 GiB process RSS watchdog, with no concurrent video rendering. These are precautionary resource changes, not proof that excessive load caused the kernel crash.

One seed per map does not establish robustness. Follow-up should repeat the same settings on several seeds, then vary corridor width and obstacle placement systematically without silently tuning per map. Inspect failures using localization, recovery, contacts, and video before changing the framework.

## Inspect and reproduce

![Executed trajectories](sky_navigation_trajectories.png)

Metrics and provenance: `sky_results.csv`, `sky_results.json`. Original records and videos: `../out/sky_evaluation_20261003/original_room2/`, `../out/sky_evaluation_20261003/new_maps_physics/`, and `../out/sky_evaluation_20261003/new_maps_physics_restart/`. Planner plots: `../out/sky_evaluation_20261003/new_map_plans/<map>/route.png`.

```bash
bash scripts/sky_evaluate.sh test
bash scripts/sky_evaluate.sh plan out/repeat_plans
bash scripts/sky_evaluate.sh run out/repeat_physics
bash scripts/sky_evaluate.sh render out/repeat_physics
```

The launcher uses the existing WSL environment and ignored controller/checkpoint links; setup details are in `docs/SKY_MAP_EXPERIMENT.md`. Each run needs a fresh output directory. Replay video samples recorded physical states rather than re-running the controller.

## Credits and application use

Underlying navigation and controller framework: Huaze Liu, Zhenyu Wu, and the research team. Sky led the assigned evaluation workstream, participated in reviewing the recordings and metrics, and shaped the report. Codex provided substantial assistance with new map implementation, registration, simulation execution, calculations and writing. This extension does not claim original authorship of the team's framework. The user's screenshots permit inclusion of team work in a report; the user separately reports GitHub permission. Preserve attribution and the actual permission scope when selecting public materials.

Navigation base commit: `f6ba1df1ed45f6f482eadfe2f84b2182692abd60`. Controller base commit: `21aed054f51a926904bafc2e5db7fdace6f6660c`. The metrics JSON records checkpoint and map/source SHA-256 hashes.
