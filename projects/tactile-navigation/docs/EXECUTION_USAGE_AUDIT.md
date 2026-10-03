Execution usage audit — jjang318/punch+global

Update, 2026-09-11: the footstep path was reduced to actor inference, measured
FK/IMU estimation, goal-to-footstep geometry and separate simulation telemetry.
It no longer imports the vforce/MujocoBackend helpers or the removed vendor
planner. See [PIPELINE_MAP.md](PIPELINE_MAP.md) for the current path. The module
counts below describe the earlier audit; other controllers still have callers
and were not removed as "unused".

Audited checkout: `/home/jjang318/tacnav/tactile-navigation`, HEAD `c3fc6a9`, with the existing five script deletions and quickstart edit. Scope: all 32 Python files under `src/execution`, their imports from retained source/scripts/tests, documented command-line entry points, current room construction, and the PACE/deploy boundary. This audit adds documentation only; it deletes no source or rooms.

**Findings**

There are 31 implementation modules plus `__init__.py`. Importing the current global policy, constructing every registered map, and loading the selected controller module brings in 18 execution modules for vforce, 15 for kinematic, and 17 for footstep. Their union is 20 modules. The other 11 support older execution, scenes, reports, replay or diagnostics; all have retained callers or documented CLI usage. No whole file in this directory is established as globally unused.

These counts measure import reachability, not executed policy branches. The probe loads modules and constructs all four task scenes; it does not initialize a physical controller or run a simulation. Conditional controller-constructor helpers were included by source inspection. This distinction matters: importing `punch_rules` does not run ContactPolicy, and importing `tick_core` does not activate PlannerCore.

**Commands: current full map missions**

The following uses the current checkout and controller environment. It runs each registered task from its configured start to its configured goal with vforce physics, global movement, PF localization, dense planned contact opportunities and EIG-selected punches. These are navigation missions, not area-coverage tours. Bounds are explicit per map; the planner can report infeasibility instead of silently relaxing them. Room830's 0.14 m / 20 degree setting previously produced a plan, but this command is not evidence of a completed physical room830 run.

```bash
cd /home/jjang318/tacnav/tactile-navigation
export PYTHONPATH="$PWD/src:$PWD/tactile_controller/src"
export MUJOCO_GL=egl
RUN_DIR="out/all_rooms_$(date +%Y%m%d_%H%M%S)"

for room in room2 room2_turn demo_room room830; do
  case "$room" in
    room2*) pos=0.10; yaw=15 ;;
    *)      pos=0.14; yaw=20 ;;
  esac
  tactile_controller/.venv/bin/python -u scripts/punch_demo.py \
    --policy global --room "$room" --controller vforce \
    --bound-pos "$pos" --bound-yaw-deg "$yaw" \
    --route-guard belief --thin dense --seed 0 --max-s 1800 \
    --out "$RUN_DIR/$room"
done
```

For a fast execution check, change `--controller vforce` to `--controller kinematic` and add `--no-video`. For a live physical view, use `MUJOCO_GL=glfw` and add `--view` on a machine with a display.

A completed recording writes `<room>_vforce_pf_global_s0.json`, `states_<room>_vforce_pf_global_s0.npz`, a route PNG, and a replay MP4 when rendering succeeds. Check the JSON `outcome` and `stop`, not only the process exit code: punch_demo.main returns 0 after recording several stopped outcomes. `route_complete` is a belief-based goal condition; use the recorded true endpoint for evaluation of actual arrival.

`--full-route` is accepted only for supported `eig`/`eig_step` task routes; global mode already searches from the scene start to its goal. `room2_turn` is a task variant of room2 with the same global scene/endpoints. The current CLI has no start/goal override for the earlier requested room830 (-4,-2.5) -> (6,1) mission; the command above uses the stored island-side start instead.

**Commands: tracker comparison and plots**

Using the same working directory and PYTHONPATH:

```bash
CMP_DIR="out/tracker_comparison_$(date +%Y%m%d_%H%M%S)"
tactile_controller/.venv/bin/python scripts/compare_route_guards.py \
  --rooms room2 demo_room --seeds 0 1 --jobs 3 --out "$CMP_DIR"
tactile_controller/.venv/bin/python scripts/plot_route_guard_comparison.py "$CMP_DIR"
```

This is 48 kinematic runs: two maps x three arrival trackers (distance/contact/belief) x two seeds x four conditions (nominal, shortened strokes, initial offset, yaw drift). Each room's plan is computed once and reused within that comparison. Outputs include `comparison.png`, `movement.png`, `touches.png`, `results.json`, and per-run records. The comparison parser and plotter support only room2/demo_room. Their two-room scope is not a reason to delete room830 or any other room.

For the historical `jjang318/eval` comparison, `compare_route_guards.py --eval-repo /home/jjang318/tacnav/demo-room-repro/tactile-navigation` loads the complete baseline checkout. Run the current side with `--modes belief`. Exact commands and the saved MuJoCo baseline are in [EVAL_BRANCH_COMPARISON.md](EVAL_BRANCH_COMPARISON.md). Historical/current comparison is a comparison of complete implementations: planner bounds, objective, map boundary and PF behavior differ. It is not the matched three-tracker experiment above. The current historical plotter also expects MuJoCo records; two new kinematic directories alone are insufficient for that plotter.

**Current execution path**

```text
scripts/punch_demo.py --policy global
  -> punch_tasks.TASKS[room] -> scene + map + goal
  -> controllers.make_controller(vforce | kinematic | footstep)
  -> PunchLocalizer + TactileIO
  -> global_route.plan_global -> lane_graph grid/covariance/planning code
  -> RouteFollowPolicy
       -> BeliefRouteGuard: arrival, predicted departure, local recovery
       -> SingleTouchSelector -> ClearedArmEIG -> ArmAngleEIG
       -> TactileIO.move / punch -> selected controller -> PF update
  -> JSON + route PNG + recorded states
  -> scripts/punch_video.py -> selected controller setup + recorded-state replay
```

Movement in global mode is selected by RouteFollowPolicy. The `punch_eig_policy` and `punch_eig_step` imports supply helpers; their alternative movement policies are not selected. The `contact` and `stations` choices remain separate CLI branches.

**Complete file inventory**

`Y` means imported in the inspected global/map/controller setup. `—` means absent from that setup, not unused across the repository. `I` highlights the indirect tick-core import, whose older planner is not selected. Caller counts are static Python import sites grouped as source/scripts/tests; CLI documentation and runtime selection are described separately.

| File | Vforce | Kinematic | Footstep | Role and reason to retain | Import sites S/C/T |
| --- | --- | --- | --- | --- | --- |
| [arm_ik.py](/home/jjang318/tacnav/tactile-navigation/src/execution/arm_ik.py) | Y | Y | Y | **Shared punch geometry.** FK and IK for scored punch directions and arm anchors. | 3/0/0 |
| [backends.py](/home/jjang318/tacnav/tactile-navigation/src/execution/backends.py) | Y | Y | Y | **Shared definitions; mixed legacy code.** BackendConfig for vforce; ProbeReading imported through tick_core. Its KinematicBackend is the older backend, distinct from PunchController kinematics. | 13/11/12 |
| [controllers.py](/home/jjang318/tacnav/tactile-navigation/src/execution/controllers.py) | Y | Y | Y | **Local controller selection.** Selects vforce, kinematic, or footstep lazily; also used by replay. | 0/3/0 |
| [executive.py](/home/jjang318/tacnav/tactile-navigation/src/execution/executive.py) | — | — | — | **Older Batch-3A executor.** Used by run_batch3a and pf_audit, and indirectly by callers importing run_batch3a._jsonable. Not the global route follower. | 1/1/11 |
| [global_route.py](/home/jjang318/tacnav/tactile-navigation/src/execution/global_route.py) | Y | Y | Y | **Current navigation.** plan_global, RouteFollowPolicy, SingleTouchSelector, route records and plots. | 0/2/6 |
| [injected_contact_check.py](/home/jjang318/tacnav/tactile-navigation/src/execution/injected_contact_check.py) | — | — | — | **Documented standalone diagnostic.** No Python import caller; explicit physical-contact check in docs/BATCH3A2_REPORT.md:379. Not an orphan. | 0/0/0 |
| [interface.py](/home/jjang318/tacnav/tactile-navigation/src/execution/interface.py) | — | — | — | **Older executor contract.** Executive imports its commands, reports and truth firewall. Retained tests also import it. | 1/0/3 |
| [mujoco_backend.py](/home/jjang318/tacnav/tactile-navigation/src/execution/mujoco_backend.py) | Y | — | Y | **Inherited physical infrastructure.** VForceBackend subclasses MujocoBackend. Live inherited methods include odometry, coordinates, truth recording, shutdown and simulation helpers. | 11/6/27 |
| [plots_batch3a.py](/home/jjang318/tacnav/tactile-navigation/src/execution/plots_batch3a.py) | — | — | — | **Older episode plotting.** run_batch3a.run_one imports plot_episode when rendering is requested. | 1/0/0 |
| [punch_actions.py](/home/jjang318/tacnav/tactile-navigation/src/execution/punch_actions.py) | Y | Y | Y | **Current action representation.** MoveCmd, PunchAction, hand poses and punch specifications; used by local and ROS controllers. | 11/2/5 |
| [punch_belief.py](/home/jjang318/tacnav/tactile-navigation/src/execution/punch_belief.py) | Y | Y | Y | **Current estimator.** PunchLocalizer and LocalizerConfig; odometry/contact/miss updates over the shared PF implementation. | 0/2/3 |
| [punch_eig.py](/home/jjang318/tacnav/tactile-navigation/src/execution/punch_eig.py) | Y | Y | Y | **Current touch scoring.** Arm-angle information gain, spec_at, arm_geometry and reach geometry; includes the alternative paired EigSelector. | 6/1/0 |
| [punch_eig_policy.py](/home/jjang318/tacnav/tactile-navigation/src/execution/punch_eig_policy.py) | Y | Y | Y | **Shared helpers plus alternative policy.** Global uses Segment and _wrap. EigRoutePolicy is selected only by --policy eig, not global. | 3/2/1 |
| [punch_eig_step.py](/home/jjang318/tacnav/tactile-navigation/src/execution/punch_eig_step.py) | Y | Y | Y | **Shared scoring plus alternative policy.** Global uses ClearedArmEIG and LAMBDA_TIME. StepEigPolicy/StepEigSelector belong to --policy eig_step. | 1/1/1 |
| [punch_footstep.py](/home/jjang318/tacnav/tactile-navigation/src/execution/punch_footstep.py) | — | — | Y | **Optional current controller.** Selected by --controller footstep. Also imports vforce helpers during construction. | 1/0/0 |
| [punch_kinematic.py](/home/jjang318/tacnav/tactile-navigation/src/execution/punch_kinematic.py) | — | Y | — | **Current comparison controller.** Selected by --controller kinematic; compare_route_guards subclasses it to inject faults. | 1/2/2 |
| [punch_rules.py](/home/jjang318/tacnav/tactile-navigation/src/execution/punch_rules.py) | Y | Y | Y | **Shared I/O plus alternative rule policy.** Global uses TactileIO and shared feature/action definitions. ContactPolicy runs only in the contact policy branch. | 7/1/3 |
| [punch_tasks.py](/home/jjang318/tacnav/tactile-navigation/src/execution/punch_tasks.py) | Y | Y | Y | **Protected room/task registry.** All four current task identifiers, map construction and starting orientation. Shared by execution, plotting and replay. | 0/7/3 |
| [punch_vforce.py](/home/jjang318/tacnav/tactile-navigation/src/execution/punch_vforce.py) | Y | — | — | **Current physical controller adapter.** VForcePunchController applies route moves and hand actions over VForceBackend. | 1/0/1 |
| [replay_video.py](/home/jjang318/tacnav/tactile-navigation/src/execution/replay_video.py) | — | — | — | **Documented older replay CLI.** No Python import caller; used by documented Batch-3A/contact-rich replay and hstack commands. | 0/0/0 |
| [replay_video_v2.py](/home/jjang318/tacnav/tactile-navigation/src/execution/replay_video_v2.py) | — | — | — | **Shared older video panels.** lane_graph.replay_v2 and replay_v3 import its drawing helpers. Not used by punch_video. | 2/0/0 |
| [route_guard.py](/home/jjang318/tacnav/tactile-navigation/src/execution/route_guard.py) | Y | Y | Y | **Current arrival/recovery checks.** BeliefRouteGuard and local_connector, called by RouteFollowPolicy. | 1/0/1 |
| [run_batch3a.py](/home/jjang318/tacnav/tactile-navigation/src/execution/run_batch3a.py) | — | — | — | **Older CLI plus shared serializer.** The older execution CLI; _jsonable is imported by both older room runners and lane_graph.run_v2/run_v3. | 2/2/0 |
| [scenes.py](/home/jjang318/tacnav/tactile-navigation/src/execution/scenes.py) | — | — | — | **Protected older room definitions.** Seven benchmark scenes; required by older runners and their retained tests. Preserve the rooms even if executors are reorganized. | 4/3/9 |
| [sensx_force.py](/home/jjang318/tacnav/tactile-navigation/src/execution/sensx_force.py) | Y | — | — | **Current vforce simulated sensing.** SenseX-style tactile force/contact model used by VForcePunchController.step. | 1/0/0 |
| [stations.py](/home/jjang318/tacnav/tactile-navigation/src/execution/stations.py) | — | — | — | **Protected scene support.** G1Station and reach geometry used by execution.scenes and tests. | 1/0/3 |
| [summarize_bench.py](/home/jjang318/tacnav/tactile-navigation/src/execution/summarize_bench.py) | — | — | — | **Documented standalone report CLI.** No Python import caller; docs/BATCH3A2_REPORT.md:398 invokes its benchmark summarizer. | 0/0/0 |
| [tick_core.py](/home/jjang318/tacnav/tactile-navigation/src/execution/tick_core.py) | I | I | I | **Indirect import plus older tick execution.** Map builders import demo_map.executor, which imports tick_core. The current global follower does not instantiate PlannerCore. | 2/3/0 |
| [tick_driver.py](/home/jjang318/tacnav/tactile-navigation/src/execution/tick_driver.py) | — | — | — | **Older tick driver.** Used by demo_room_run and both M2 smoke scripts; not selected by punch_demo. | 0/3/0 |
| [vforce_backend.py](/home/jjang318/tacnav/tactile-navigation/src/execution/vforce_backend.py) | Y | — | Y | **Current physical environment; mixed legacy code.** Builds the vforce environment, loads the checkpoint and inherits MuJoCo infrastructure. Also shared by data collection and footstep setup. | 7/4/8 |
| [wall_lane.py](/home/jjang318/tacnav/tactile-navigation/src/execution/wall_lane.py) | Y | Y | Y | **Protected shared map geometry.** WallSegment, raycast and face helpers used by room2/demo_room scene builders, older execution and data collection. | 9/5/12 |
| [__init__.py](/home/jjang318/tacnav/tactile-navigation/src/execution/__init__.py) | Y | Y | Y | Package initializer; preserve the package. | — |

**Couplings that prevent deletion by filename**

1. **Shared tactile I/O is inside the rule-policy file.** `global_route.py:26` imports TactileIO; the class starts at `punch_rules.py:152`, while ContactPolicy starts at 525. Removing the file would break global execution. A future separation should move shared I/O/action results/features while preserving the rule policy's own callers.
2. **The global route shares data types with the alternative EIG follower.** `global_route.py:24` imports Segment and _wrap from `punch_eig_policy.py`; RouteLeg subclasses Segment. Deleting the file because its EigRoutePolicy is not selected would break the route representation.
3. **Single-punch EIG shares a file with joint movement/punch EIG.** `global_route.py:25` imports ClearedArmEIG and LAMBDA_TIME from `punch_eig_step.py`. ClearedArmEIG starts at line 135; StepEigSelector and StepEigPolicy start at 160 and 295. Only the latter implement the alternative movement policy.
4. **Map construction pulls in an older executor.** `scene_room2.py:29` and `scene_demo.py:37` import helpers/constants from `demo_map.executor`. That module imports tick_core at line 41, which imports backends.ProbeReading at line 39. The current global loop does not drive PlannerCore. Removing tick_core would nevertheless break imports until scene geometry/helpers are separated from executor code.
5. **The vforce backend is not standalone.** VForcePunchController subclasses VForceBackend, which subclasses MujocoBackend. Inherited live methods include `truth_pose` (mujoco_backend.py:1029), `odometry_increment` (1231), frame transforms (1004/1008), and `close` (3913). The physical environment setup lives in VForceBackend even though the old executor's press API is bypassed. Whole-file removal is invalid; unused inherited methods need a separate method-level study before trimming.
6. **A serializer imports a whole older pipeline.** Older `room2_run.py:26`, `demo_room_run.py:30`, and `lane_graph/run_v2.py:16` / `run_v3.py:16` import `_jsonable` from `execution.run_batch3a`. That module imports Executive and SCENES at module scope. These older executors are outside punch_demo's global path, but removing them breaks retained room runners unless the serializer is first separated.
7. **Old video modules contain shared drawing code.** `lane_graph.replay_v2` and `replay_v3` import panels from `execution.replay_video_v2`; only `replay_video.py` is a standalone older CLI. Current punch replay is `scripts/punch_video.py`.
8. **A redundant selector is constructed on the global path.** `punch_demo.run_contact` constructs a paired EigSelector in the general `a.select == "eig"` block (line 274 onward), then replaces it with SingleTouchSelector in the global branch. This is a candidate for a narrow code cleanup; it does not make the old paired selector choose global movement.

**Concrete callers for every execution module**

Representative source/script import sites below make the matrix reviewable. Files with zero Python import callers are documented CLIs, listed with their invocation evidence. Full test counts are in the matrix; this is not a claim that every test was run during the audit.

- **arm_ik**: [src/execution/punch_actions.py](/home/jjang318/tacnav/tactile-navigation/src/execution/punch_actions.py:20) (`module` imports `SEED_FORWARD,solve_anchor`); [src/execution/punch_eig.py](/home/jjang318/tacnav/tactile-navigation/src/execution/punch_eig.py:348) (`spec_at` imports `SEED_FORWARD`); [src/execution/punch_eig.py](/home/jjang318/tacnav/tactile-navigation/src/execution/punch_eig.py:189) (`ArmAngleEIG.reach_at` imports `_fk`).
- **backends**: [scripts/audit_divert_physical.py](/home/jjang318/tacnav/tactile-navigation/scripts/audit_divert_physical.py:22) (`module` imports `BackendConfig,KinematicBackend`); [scripts/bench_v3.py](/home/jjang318/tacnav/tactile-navigation/scripts/bench_v3.py:36) (`module` imports `BackendConfig,KinematicBackend`); [scripts/collect_touch_data.py](/home/jjang318/tacnav/tactile-navigation/scripts/collect_touch_data.py:252) (`collect` imports `BackendConfig`); [scripts/demo_room_run.py](/home/jjang318/tacnav/tactile-navigation/scripts/demo_room_run.py:69) (`backend` imports `BackendConfig`); 20 additional source/script import sites.
- **controllers**: [notebooks/run_walkthrough_headless.py](/home/jjang318/tacnav/tactile-navigation/notebooks/run_walkthrough_headless.py:167) (`module` imports `make_controller`); [scripts/compare_route_guards.py](/home/jjang318/tacnav/tactile-navigation/scripts/compare_route_guards.py:31) (`module` imports `CONTROLLERS`); [scripts/punch_demo.py](/home/jjang318/tacnav/tactile-navigation/scripts/punch_demo.py:50) (`module` imports `CONTROLLERS,make_controller`); [scripts/punch_video.py](/home/jjang318/tacnav/tactile-navigation/scripts/punch_video.py:168) (`main` imports `make_controller`).
- **executive**: [scripts/pf_audit.py](/home/jjang318/tacnav/tactile-navigation/scripts/pf_audit.py:27) (`module` imports `ExecConfig,Executive`); [src/execution/run_batch3a.py](/home/jjang318/tacnav/tactile-navigation/src/execution/run_batch3a.py:23) (`module` imports `ExecConfig,Executive`).
- **global_route**: [notebooks/run_walkthrough_headless.py](/home/jjang318/tacnav/tactile-navigation/notebooks/run_walkthrough_headless.py:48) (`module` imports `filter_consistent_config`); [notebooks/run_walkthrough_headless.py](/home/jjang318/tacnav/tactile-navigation/notebooks/run_walkthrough_headless.py:156) (`module` imports `plan_global,route_record,legs_from_path`); [notebooks/run_walkthrough_headless.py](/home/jjang318/tacnav/tactile-navigation/notebooks/run_walkthrough_headless.py:170) (`module` imports `SingleTouchSelector,RouteFollowPolicy`); [scripts/compare_route_guards.py](/home/jjang318/tacnav/tactile-navigation/scripts/compare_route_guards.py:30) (`module` imports `global_route`); 1 additional source/script import sites.
- **injected_contact_check**: documented CLI: [BATCH3A2_REPORT.md](/home/jjang318/tacnav/tactile-navigation/docs/BATCH3A2_REPORT.md:379).
- **interface**: [src/execution/executive.py](/home/jjang318/tacnav/tactile-navigation/src/execution/executive.py:46) (`module` imports `ExecStatus,GlobalDirective,LocalReport,ProbeOutcome,RouteDecision,TruthLog,assert_no_truth_leak`).
- **mujoco_backend**: [scripts/demo_room_run.py](/home/jjang318/tacnav/tactile-navigation/scripts/demo_room_run.py:70) (`backend` imports `MujocoConfig`); [scripts/pf_audit.py](/home/jjang318/tacnav/tactile-navigation/scripts/pf_audit.py:99) (`run_episode` imports `MujocoBackend,MujocoConfig`); [scripts/room2_run.py](/home/jjang318/tacnav/tactile-navigation/scripts/room2_run.py:34) (`backend` imports `MujocoConfig`); [scripts/standoff_tracking_test.py](/home/jjang318/tacnav/tactile-navigation/scripts/standoff_tracking_test.py:33) (`module` imports `MujocoConfig`); 13 additional source/script import sites.
- **plots_batch3a**: [src/execution/run_batch3a.py](/home/jjang318/tacnav/tactile-navigation/src/execution/run_batch3a.py:77) (`run_one` imports `plot_episode`).
- **punch_actions**: [scripts/punch_demo.py](/home/jjang318/tacnav/tactile-navigation/scripts/punch_demo.py:51) (`module` imports `HandCmd,MoveCmd,PunchAction`); [scripts/punch_demo.py](/home/jjang318/tacnav/tactile-navigation/scripts/punch_demo.py:579) (`main` imports `PARALLEL_GUARD`); [src/execution/controllers.py](/home/jjang318/tacnav/tactile-navigation/src/execution/controllers.py:27) (`module` imports `PunchAction`); [src/execution/global_route.py](/home/jjang318/tacnav/tactile-navigation/src/execution/global_route.py:698) (`RouteFollowPolicy.start` imports `PARALLEL_GUARD`); 9 additional source/script import sites.
- **punch_belief**: [notebooks/run_walkthrough_headless.py](/home/jjang318/tacnav/tactile-navigation/notebooks/run_walkthrough_headless.py:47) (`module` imports `LocalizerConfig`); [notebooks/run_walkthrough_headless.py](/home/jjang318/tacnav/tactile-navigation/notebooks/run_walkthrough_headless.py:168) (`module` imports `PunchLocalizer`); [scripts/compare_route_guards.py](/home/jjang318/tacnav/tactile-navigation/scripts/compare_route_guards.py:32) (`module` imports `LocalizerConfig`); [scripts/punch_demo.py](/home/jjang318/tacnav/tactile-navigation/scripts/punch_demo.py:191) (`run_contact` imports `LocalizerConfig,PunchLocalizer`).
- **punch_eig**: [scripts/punch_demo.py](/home/jjang318/tacnav/tactile-navigation/scripts/punch_demo.py:274) (`run_contact` imports `EigSelector`); [src/execution/global_route.py](/home/jjang318/tacnav/tactile-navigation/src/execution/global_route.py:23) (`module` imports `spec_at,arm_geometry,TRAVEL_M`); [src/execution/global_route.py](/home/jjang318/tacnav/tactile-navigation/src/execution/global_route.py:488) (`RouteFollowPolicy._expect_travel` imports `DOME_R_M`); [src/execution/punch_eig_policy.py](/home/jjang318/tacnav/tactile-navigation/src/execution/punch_eig_policy.py:38) (`module` imports `DOME_R_M`); 3 additional source/script import sites.
- **punch_eig_policy**: [scripts/punch_demo.py](/home/jjang318/tacnav/tactile-navigation/scripts/punch_demo.py:359) (`run_contact` imports `EigRoutePolicy`); [scripts/punch_route_preview.py](/home/jjang318/tacnav/tactile-navigation/scripts/punch_route_preview.py:11) (`module` imports `EigRoutePolicy`); [src/execution/global_route.py](/home/jjang318/tacnav/tactile-navigation/src/execution/global_route.py:24) (`module` imports `Segment,_wrap`); [src/execution/punch_eig_step.py](/home/jjang318/tacnav/tactile-navigation/src/execution/punch_eig_step.py:65) (`module` imports `Segment,_wrap`); 1 additional source/script import sites.
- **punch_eig_step**: [scripts/punch_demo.py](/home/jjang318/tacnav/tactile-navigation/scripts/punch_demo.py:334) (`run_contact` imports `StepEigPolicy,StepEigSelector`); [src/execution/global_route.py](/home/jjang318/tacnav/tactile-navigation/src/execution/global_route.py:25) (`module` imports `ClearedArmEIG,LAMBDA_TIME`).
- **punch_footstep**: [src/execution/controllers.py](/home/jjang318/tacnav/tactile-navigation/src/execution/controllers.py:72) (`_make_footstep` imports `FootstepPunchController`).
- **punch_kinematic**: [scripts/compare_route_guards.py](/home/jjang318/tacnav/tactile-navigation/scripts/compare_route_guards.py:33) (`module` imports `KinematicConfig,KinematicPunchController`); [scripts/punch_demo.py](/home/jjang318/tacnav/tactile-navigation/scripts/punch_demo.py:530) (`main` imports `KinematicConfig`); [src/execution/controllers.py](/home/jjang318/tacnav/tactile-navigation/src/execution/controllers.py:79) (`_make_kinematic` imports `KinematicConfig,KinematicPunchController`).
- **punch_rules**: [notebooks/run_walkthrough_headless.py](/home/jjang318/tacnav/tactile-navigation/notebooks/run_walkthrough_headless.py:169) (`module` imports `TactileIO`); [scripts/punch_demo.py](/home/jjang318/tacnav/tactile-navigation/scripts/punch_demo.py:180) (`run_contact` imports `ContactPolicy,TactileIO`); [src/execution/global_route.py](/home/jjang318/tacnav/tactile-navigation/src/execution/global_route.py:26) (`module` imports `TactileIO`); [src/execution/punch_eig_policy.py](/home/jjang318/tacnav/tactile-navigation/src/execution/punch_eig_policy.py:39) (`module` imports `Feature,TactileIO`); 5 additional source/script import sites.
- **punch_tasks**: [notebooks/run_walkthrough_headless.py](/home/jjang318/tacnav/tactile-navigation/notebooks/run_walkthrough_headless.py:28) (`module` imports `TASKS`); [scripts/compare_route_guards.py](/home/jjang318/tacnav/tactile-navigation/scripts/compare_route_guards.py:34) (`module` imports `TASKS`); [scripts/plot_eval_comparison.py](/home/jjang318/tacnav/tactile-navigation/scripts/plot_eval_comparison.py:15) (`module` imports `TASKS`); [scripts/plot_room_plan.py](/home/jjang318/tacnav/tactile-navigation/scripts/plot_room_plan.py:32) (`main` imports `TASKS`); 4 additional source/script import sites.
- **punch_vforce**: [src/execution/controllers.py](/home/jjang318/tacnav/tactile-navigation/src/execution/controllers.py:62) (`_make_vforce` imports `VForcePunchController`).
- **replay_video**: documented CLI: [CONTACT_RICH_DEMO_REPORT.md](/home/jjang318/tacnav/tactile-navigation/docs/CONTACT_RICH_DEMO_REPORT.md:213).
- **replay_video_v2**: [src/lane_graph/replay_v2.py](/home/jjang318/tacnav/tactile-navigation/src/lane_graph/replay_v2.py:21) (`module` imports `_draw_left,_draw_right,_left_panel,_right_panel`); [src/lane_graph/replay_v3.py](/home/jjang318/tacnav/tactile-navigation/src/lane_graph/replay_v3.py:16) (`module` imports `_draw_left,_draw_right,_left_panel,_right_panel`).
- **route_guard**: [src/execution/global_route.py](/home/jjang318/tacnav/tactile-navigation/src/execution/global_route.py:27) (`module` imports `BeliefRouteGuard,local_connector`).
- **run_batch3a**: [scripts/demo_room_run.py](/home/jjang318/tacnav/tactile-navigation/scripts/demo_room_run.py:30) (`module` imports `_jsonable`); [scripts/room2_run.py](/home/jjang318/tacnav/tactile-navigation/scripts/room2_run.py:26) (`module` imports `_jsonable`); [src/lane_graph/run_v2.py](/home/jjang318/tacnav/tactile-navigation/src/lane_graph/run_v2.py:16) (`module` imports `_jsonable`); [src/lane_graph/run_v3.py](/home/jjang318/tacnav/tactile-navigation/src/lane_graph/run_v3.py:16) (`module` imports `_jsonable`).
- **scenes**: [scripts/accept_contact_rich.py](/home/jjang318/tacnav/tactile-navigation/scripts/accept_contact_rich.py:37) (`module` imports `build_scene`); [scripts/pf_audit.py](/home/jjang318/tacnav/tactile-navigation/scripts/pf_audit.py:28) (`module` imports `build_scene`); [scripts/plots_contact_rich.py](/home/jjang318/tacnav/tactile-navigation/scripts/plots_contact_rich.py:14) (`module` imports `build_scene`); [src/execution/injected_contact_check.py](/home/jjang318/tacnav/tactile-navigation/src/execution/injected_contact_check.py:41) (`main` imports `build_scene`); 3 additional source/script import sites.
- **sensx_force**: [src/execution/punch_vforce.py](/home/jjang318/tacnav/tactile-navigation/src/execution/punch_vforce.py:23) (`module` imports `SensxPad`).
- **stations**: [src/execution/scenes.py](/home/jjang318/tacnav/tactile-navigation/src/execution/scenes.py:52) (`module` imports `G1Station`).
- **summarize_bench**: documented CLI: [BATCH3A2_REPORT.md](/home/jjang318/tacnav/tactile-navigation/docs/BATCH3A2_REPORT.md:398).
- **tick_core**: [scripts/demo_room_run.py](/home/jjang318/tacnav/tactile-navigation/scripts/demo_room_run.py:270) (`main._route` imports `PlannerCore`); [scripts/m2_stage1_smoke.py](/home/jjang318/tacnav/tactile-navigation/scripts/m2_stage1_smoke.py:46) (`mission` imports `BaseGoal,Cmd,PlannerCore,goto,observe,turn_in_place`); [scripts/m2_stage2_smoke.py](/home/jjang318/tacnav/tactile-navigation/scripts/m2_stage2_smoke.py:43) (`main` imports `PlannerCore,press`); [src/demo_map/executor.py](/home/jjang318/tacnav/tactile-navigation/src/demo_map/executor.py:41) (`module` imports `tick_core`); 1 additional source/script import sites.
- **tick_driver**: [scripts/demo_room_run.py](/home/jjang318/tacnav/tactile-navigation/scripts/demo_room_run.py:271) (`main._route` imports `MujocoTickDriver,params_from_backend`); [scripts/m2_stage1_smoke.py](/home/jjang318/tacnav/tactile-navigation/scripts/m2_stage1_smoke.py:48) (`mission` imports `MujocoTickDriver,params_from_backend`); [scripts/m2_stage2_smoke.py](/home/jjang318/tacnav/tactile-navigation/scripts/m2_stage2_smoke.py:44) (`main` imports `MujocoTickDriver,params_from_backend`).
- **vforce_backend**: [scripts/collect_touch_data.py](/home/jjang318/tacnav/tactile-navigation/scripts/collect_touch_data.py:253) (`collect` imports `VForceBackend,VForceConfig`); [scripts/demo_room_run.py](/home/jjang318/tacnav/tactile-navigation/scripts/demo_room_run.py:74) (`backend` imports `VForceBackend,VForceConfig`); [scripts/render_dataset_trajectories.py](/home/jjang318/tacnav/tactile-navigation/scripts/render_dataset_trajectories.py:266) (`render_segment` imports `VForceBackend,VForceConfig`); [scripts/room2_run.py](/home/jjang318/tacnav/tactile-navigation/scripts/room2_run.py:39) (`backend` imports `VForceBackend,VForceConfig`); 7 additional source/script import sites.
- **wall_lane**: [scripts/demo_room_ate_plan.py](/home/jjang318/tacnav/tactile-navigation/scripts/demo_room_ate_plan.py:49) (`module` imports `WallSegment,face_present,raycast`); [scripts/demo_room_video.py](/home/jjang318/tacnav/tactile-navigation/scripts/demo_room_video.py:434) (`follow_azimuth` imports `raycast`); [scripts/wall_parallel_demo.py](/home/jjang318/tacnav/tactile-navigation/scripts/wall_parallel_demo.py:25) (`module` imports `tap_stations,wall_segments`); [scripts/wall_touch_compare.py](/home/jjang318/tacnav/tactile-navigation/scripts/wall_touch_compare.py:28) (`module` imports `WallSegment`); 10 additional source/script import sites.

**Rooms and map data: preserve all**

| Current task identifier | Runtime scene label | Stored start -> goal, map frame (m) | Map source |
| --- | --- | --- | --- |
| room2 | room2_fwd | (0,0) -> (2.5,3.2), radius 0.4 | docs/demo_maps/room2_scan_walls.json |
| room2_turn | room2_fwd | (0,0) -> (2.5,3.2), radius 0.4 | Same room2 geometry; distinct task behavior in alternative policies |
| demo_room | demo_room_scan | (0,0) -> (15.2200,11.7765), radius 0.4 | docs/demo_maps/demo_map_scan_walls.json plus scene construction |
| room830 | demo_room_scan | (0,0) -> (4.894,-3.246), radius 0.4 | docs/demo_maps/room_8_30_2026_curated.json, translated to island-side start |

Use the record's `room`/`task` fields to distinguish room830 from demo_room; they currently share the scene-name string `demo_room_scan`. Room830 has its own map geometry despite that label. Its curated JSON is regenerated from `room_8_30_2026_walls.json` by `scripts/curate_room830.py`; this script is retained even without a Python import caller.

Also preserve:

- `execution.scenes` and its seven scenes: ambiguous_landmark_house, ambiguous_near_edge, contact_rich_gap, detour_box, multi_decision_house, nominal_multiroom, two_turns; `execution.stations` supports them.
- `lane_graph.maps` / `scene2`: m1_legacy_perimeter, m2_pillar_room, m3_l_alcove, m4_twin_alcoves, m5_t_junction.
- `lane_graph.maps_v3` / `scene3`: r1_gallery, r2_lbend, r3_tpinch, r4_winding, r5_twinbay.
- `global_nav.stressmaps`, other scene/map generators, all files under `docs/demo_maps`, mission files under `configs`, their reconstruction/extraction scripts, controller assets, and saved run outputs.

Being outside the four-task registry does not make a room or its support code expendable. Moving room definitions into a dedicated package could preserve them during a later refactor; deleting room definitions is outside this cleanup.

**PACE is a separate execution boundary**

`/home/jjang318/tacnav/locomanip-deploy-recovery/scripts/pace/submit_eval.sh` syncs the deploy repository's `src_ext`, builds scenes with `make_planner_scene_mjcf.py`, and invokes `scripts/pace/run_job.py`. That runner launches the ROS controller stack and `tacnav_punch/punch_node.py`. The library dependency in deploy `pixi.toml:213` is `src_ext/tactile_navigation` (checkout 1721b49), not this sibling punch+global checkout. ROS uses RosPunchController; it does not load the standalone VForcePunchController simulation wrapper. Its recording/rendering scripts also live in the deploy repository.

The current deploy adapter is not API-compatible with merely swapping in this branch. At `punch_node.py:445` it imports GlobalRoutePolicy, which this branch no longer defines. Its SingleTouchSelector call also passes the old `motion` argument, and its RouteFollowPolicy call uses old lane parameters. These are concrete integration differences, independent of the script cleanup. The local commands above run this branch; submitting a PACE job with the existing deploy checkout does not establish that the server ran it. This audit does not switch either checkout or modify the ROS adapter.

**Cleanup disposition**

- Keep the current global/PF/action/geometry/controller modules and every room/map dependency.
- Separate shared I/O, Segment/angle helpers, cleared-arm EIG, map construction helpers and the JSON serializer from their older policy/runner files before considering relocation.
- The 11 modules outside current controller setups are `executive`, `injected_contact_check`, `interface`, `plots_batch3a`, `replay_video`, `replay_video_v2`, `run_batch3a`, `scenes`, `stations`, `summarize_bench`, and `tick_driver`. They are candidates for organization by workflow, not established dead files. In particular, scenes/stations must survive.
- Preserve documented older execution commands and their tests unless retiring a workflow is separately requested. Updating or relocating an import path requires updating its callers.

**Verification performed**

All four TASKS factories constructed successfully. Fresh-process module probes confirmed the per-controller import sets above. Static AST inspection found source/script/test import sites across the repository; documentation searches caught standalone `python -m` entry points that an import-only scan would miss. Controller method inspection confirmed the vforce inheritance dependencies. The current environment reports NumPy 2.4.6, MuJoCo 3.10.0, mjlab 1.5.3 and Torch 2.13.0. These are file-usage and command checks, not new full-room outcome measurements.
