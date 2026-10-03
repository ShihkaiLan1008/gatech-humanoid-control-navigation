Pipeline and modularity review — jjang318/punch+global

For the current module diagram, interfaces and candidate status, see
[PIPELINE_MAP.md](PIPELINE_MAP.md). The dated analysis below includes historical
implementations and experiment results.

Update, 2026-09-11: footstep now uses encoder FK and gyro/accelerometer estimation
for feet, hands and base odometry. Simulator pose/velocity is isolated to
evaluation/replay. The goal-to-footstep geometry remains, with unused random
goals, heading law and vendor duplication removed. Only the two actors run;
critics, rewards and random hand-target generation are excluded. Vforce retains
its separate synthetic odometry. [Current validation](../out/footstep_proprio/).

Update, 2026-09-10: the kinematic/vforce global execution controllers now latch targets, track translation/yaw and decide arrival using onboard estimated pose. Vforce uses [EstimatedPoseTwistCommand](/home/jjang318/tacnav/tactile-navigation/src/execution/estimated_twist.py); its inherited recovery target setter also bypasses the old truth-minus-estimate compensation. Maze goal resampling is disabled for this executor-owned command stream. The shared tactile_controller checkout is unchanged.

The corrected target composition is `estimated_start_world + R(estimated_start_yaw) * delta_body`; the servo transforms estimated position error into a body command. Physical motion still uses the actual body frame, so localization errors remain visible. Kinematic `forward_deliver_frac` now scales plant velocity: accurate odometry compensates by moving longer, while inaccurate odometry changes the physical endpoint.

Seed-0 kinematic checks completed all four tasks with unchanged planner/guard bounds. Records and movement comparison: [estimated_feedback/20260910_143410](/home/jjang318/tacnav/tactile-navigation/out/estimated_feedback/20260910_143410). Focused tests cover inaccessible simulator pose, incorrect yaw, odometry gain error, arrival, latching and the vforce configuration. No full vforce rollout was performed for this check.

Subsequent physics check, 2026-09-10: Room2 completed with vforce and estimated-pose control, seed 0, one global plan, 19 punches/18 contacts. The right-hand north-wall punch ran from 56.26 to 58.32 s (physical obstacle ID 1, contact hold 0.56 s); the -90-degree turn started at 58.32 s. Y-position sigma fell from 0.097 to 0.021 m. Route completion was at 105.86 s, true goal error 0.189 m inside the 0.4 m goal radius. [Record, plot and replay](/home/jjang318/tacnav/tactile-navigation/out/north_touch_vforce/20260910_145303).

This observed ordering is not an explicit north-contact precondition. `step_station.satisfied` in belief mode checks PF arrival probability, and `_motion_prediction` checks geometry/uncertainty. EIG chose the side touch in this run. An offline prediction using the recorded pre-punch mean/covariance still permits the turning move (sigma 0.08175 m < 0.10 m); see `without_north_touch_prediction.json` in the run directory. The contact baseline checks the last scheduled contact of the current leg (west wall here), so switching to that baseline also does not express a required next-wall touch.

The recorded pre-touch particle cloud also passes the arrival test (97.07% probability versus 95% required). Thus both runtime predicates can pass without this touch; a strict touch-before-turn contract would need an explicit confirmed-contact precondition naming the planned surface.

The analysis and instrumented examples below describe the **pre-fix revision**.

Reviewed revision: `387b837`. Current source was exercised with instrumented kinematic runs of all four registered tasks, seed 0. Instrumentation recorded commands at RouteFollowPolicy, TactileIO and the controller; it delegated to the original implementations and did not change action selection or constraints. Runtime files and room/map assets were not edited.

The central finding: the global planner returns a complete route object, not a stream of controller waypoints. RouteFollowPolicy owns next-point selection and recovery; TactileIO turns that decision into a blocking motion/punch primitive; the controller owns the actual base/hand servo. These layers exist, but their contracts are incomplete and several responsibilities cross the boundaries.

**How the next location reaches the robot**

| Stage | Actual operation | Source |
| --- | --- | --- |
| Global planning | Search once and return GlobalRoute: legs, grid/path, headings, contact schedule, uncertainty trace and goal | [plan_global](/home/jjang318/tacnav/tactile-navigation/src/execution/global_route.py:173) |
| Route progress | Project PF mean onto current leg; choose s_next = min(max(s,0)+0.40, leg_length); target = leg_origin + s_next * tangent | [step_station](/home/jjang318/tacnav/tactile-navigation/src/execution/global_route.py:719) |
| Local correction | If off route, find a connector to the current-leg projection; wall-distance corrections are capped at 0.10 m | [local recovery](/home/jjang318/tacnav/tactile-navigation/src/execution/global_route.py:775) |
| Admission | Check segment geometry, cross-track/yaw uncertainty and, in belief mode, prediction through the next contact opportunity | [_goto](/home/jjang318/tacnav/tactile-navigation/src/execution/global_route.py:656) |
| Map -> body | delta_body = R(-PF_yaw) * (target_map - PF_xy); split targets over 0.60 m and re-read PF after each primitive | [_to_body](/home/jjang318/tacnav/tactile-navigation/src/execution/global_route.py:650) |
| Primitive | Increment move_id; create MoveCmd(target_map=None, rel=(forward,left), yaw_map=commanded heading); apply it until completion/timeout/interruption | [TactileIO.move](/home/jjang318/tacnav/tactile-navigation/src/execution/punch_rules.py:464) |
| Tick | Package the base command and both hand commands as PunchAction; call controller.apply, then controller.step | [TactileIO._tick](/home/jjang318/tacnav/tactile-navigation/src/execution/punch_rules.py:234) |
| Controller | Latch a local target on the first occurrence of move_id; repeated ticks must not add the displacement again | [VForcePunchController.apply](/home/jjang318/tacnav/tactile-navigation/src/execution/punch_vforce.py:104) |
| Feedback | TactileIO predicts PF from odometry after a move; a punch updates it with contact/miss evidence; route following then chooses again | [PF predict](/home/jjang318/tacnav/tactile-navigation/src/execution/punch_belief.py:147) |

For one latched command, the simulation controller computes:

```text
delta_body = R(-estimated_yaw) * (map_target - estimated_position)
servo_target_world = actual_start_world + R(actual_start_yaw) * delta_body
```

The second equation uses simulator state inside the simulation controller. ROS instead uses local odometry. Thus the controller's physical endpoint need not equal the global map target: localization error is retained. The planner does not get a perfect correction to the desired map point. Heading is separately carried as an absolute map-frame yaw in MoveCmd; translation and heading currently have different frame semantics.

The base trajectory is closed loop inside the controller, but the route follower normally re-evaluates PF only after a blocking primitive, not on every servo tick. A motion's relative displacement is latched in its starting body frame even when yaw changes during the move.

**An observed turn, not just a diagram**

Room2 at t=39.68 s, move_id=11:

```text
PF position/yaw: (-0.174200, 3.102089), 179.971 degrees
Global/local target: (0.030000, 3.312000)
Body displacement passed to TactileIO: (-0.204096, -0.210013) m
Heading change: -90 degrees; commanded map heading: +90 degrees
Actual simulated start: (-0.168249, 3.099709), 180 degrees
Controller's latched target: (0.035847, 3.309722)
Returned motion status: done
PF after motion: (0.030192, 3.309336), 89.568 degrees
```

The difference between global target and latched simulated target comes from the estimated-versus-actual starting pose. The frame conversion itself matches to numerical precision.

**What the physical vforce controller actually does**

VForcePunchController.apply writes a world pose target into the controller environment's twist command term. [MazeTwist._update_command](/home/jjang318/tacnav/tactile-navigation/tactile_controller/src/rl/eval/mdp/maze_twist.py:395) converts position/yaw error into body velocities using a distance-dependent deceleration law, velocity limits, yaw feedback and slew limiting. The neural policy consumes the environment observations/commands and outputs actions; [MujocoBackend._step](/home/jjang318/tacnav/tactile-navigation/src/execution/mujoco_backend.py:1564) advances the simulation. VForceBackend loads the policy and environment at [lines 513 onward](/home/jjang318/tacnav/tactile-navigation/src/execution/vforce_backend.py:513).

MazeTwist also contains standalone goal-sampling behavior, but this wrapper writes the goal and resets its time_left on each apply. The global mission's waypoint source is RouteFollowPolicy. The nearest-surface facing behavior comes from [commanded_heading](/home/jjang318/tacnav/tactile-navigation/src/lane_graph/grid_constrained.py:220), not the EIG selector.

**The different meanings of guard**

| Name / owner | What it means in this workflow | Consequence and review |
| --- | --- | --- |
| `guards`, `_guards`, `PARALLEL_GUARD` | Dictionary of left/right hand carry or pre-punch poses | Not a blocking rule. Rename conceptually to carry_poses or ready_poses to avoid confusion. |
| Geometry admission | Sample the proposed base segment against obstacle clearance and room/open-boundary limits | A physical feasibility check, but currently only for the proposed straight segment; controller path shape is not part of the contract. |
| One-step uncertainty admission | Propagate joint x/y/yaw covariance and require cross-track sigma and yaw sigma to fit the bounds | An execution policy implementing the selected uncertainty objective. Present in distance, contact and belief modes. Rotation-only moves skip the position bound but retain the yaw bound. |
| Belief departure check | Predict without hypothetical observations from the proposed target through the next planned contact opportunity | Additional gate enforced only by `--route-guard belief`. It can reject a step that passes the one-step check. |
| Arrival check | Before changing legs, require 95% PF mass inside a planner-derived endpoint region and heading band; final leg uses the goal disk | A progress/arrival policy. Distance mode uses mean endpoint distance; contact mode also requires a matching observed surface. None of these is a motor-level force stop. |
| EIG/time threshold | Reject a touch below 0.065 nats/s * 2.3 s = 0.1495 nats expected information | A heuristic time-price decision. It applies even when a planned contact is marked required and when motion is waiting for localization. |
| Touch/retry limits | Up to 4 touches per batch, 3% progress rule, at most 3 arrival recenterings, 400 station iterations; move timeout 15 s or 20 s with a turn | Bounded execution heuristics. They need explicit failure/recovery policy rather than being treated as physical feasibility laws. |
| Punch candidate clearance | A candidate's ready hand position must be >=0.06 m from a mapped surface; candidates crossing an open navigation boundary are excluded | Part of local touch feasibility; keep it separate from the EIG utility. |
| Force interruption | Detect an unexpectedly loaded carried hand; record passive contact before consuming the stop | Controller-level protection. During a commanded probe the probing hand is exempt from this carried-hand check; the punch primitive separately detects/brakes/regulates contact. |

The selection modes are not a master switch for all checks. `--route-guard distance` removes the probabilistic arrival requirement and the additional departure-horizon veto, but retains one-step uncertainty, geometry, touch economics and physical force handling. There is no general guard-off mode.

The sigma bound and EIG time gate can disagree: motion asks for localization, but every available touch scores below 0.1495 nats. Current code returns motion_uncertainty after its touch attempt if the prediction remains inadmissible. This is a policy conflict; disabling force protection would not resolve it. Recovery needs its own decision objective or admissible small feedback motions, not a blanket force-guard bypass.

The allowed wall correction already changes the constrained direction to the active leg's normal for bounded error-reducing feedback. It still propagates the actual displacement and still checks geometry. Global route search is not rerun. The `REPLAN` event emitted after an interrupted primitive means local reselection on the existing route.

**Concrete implementation findings**

1. **Stopped is incorrectly usable as arrived.** [TactileIO.move](/home/jjang318/tacnav/tactile-navigation/src/execution/punch_rules.py:499) declares done when `moving()` becomes false. It does not read a reached/stalled/cancelled outcome. The deploy ROS servo sets last_move_status='stalled' and clears its move on a stall ([ros_controller.py](/home/jjang318/tacnav/locomanip-deploy-recovery/src/planner/tacnav_punch/tacnav_punch/ros_controller.py:532)), and moving() then returns false. A direct probe of the real TactileIO.move with a controller reporting stalled reproduced a done return. This is a concrete contract defect, not an observed physical run in this review. Use explicit motion status plus endpoint errors.

2. **The vforce force-hold setting is bypassed by another path.** VForceConfig sets guard_abort_hold_s=0.10 ([config](/home/jjang318/tacnav/tactile-navigation/src/execution/vforce_backend.py:262)); the per-tick path honors it, but [_on_substep](/home/jjang318/tacnav/tactile-navigation/src/execution/mujoco_backend.py:1349) triggers immediately on >=12 N. Vforce installs that callback through set_touch_hand. A direct callback probe with one 13 N sensed sample set force_abort_pending while loaded_s was still 0.0. Thus the setting does not debounce the complete force guard. Both sampling paths must share one declared force policy, or separately expose and name an instantaneous limit.

3. **The checked path and controller path can differ.** `_goto` calls grid.segment_clear on a straight segment. The ROS velocity servo deliberately finishes forward motion before lateral motion ([servo](/home/jjang318/tacnav/locomanip-deploy-recovery/src/planner/tacnav_punch/tacnav_punch/ros_controller.py:522)); a simultaneous turn can change the swept path further. A clear straight connector is not evidence that this controller's executed path clears obstacles. This is a source-level mismatch; no collision from this mismatch was simulated here. A replacement controller must expose its path/footprint envelope, or the local executor must generate and check the actual trajectory it will follow.

4. **An uncleared stop has no global-executor readiness gate.** The physical safety stop can time out with last_safety_stop_cleared=False ([exit](/home/jjang318/tacnav/tactile-navigation/src/execution/mujoco_backend.py:1760)). TactileIO logs this in last_guard_cleared, but RouteFollowPolicy's interrupted-move path resumes the station loop without consulting it ([resume](/home/jjang318/tacnav/tactile-navigation/src/execution/global_route.py:823)). The lower force monitor can trip again, but that is not an explicit ready-to-resume contract. Feedback needs an interrupted status and a separate ready/clear condition; recovery motion can then be chosen deliberately.

5. **The controller Protocol is incomplete.** It omits odometry_increment and reset_estimator_to, although PunchLocalizer calls them, and omits optional probe/force-stop/cancel/carry hooks discovered with getattr by TactileIO. It also mixes simulator recording fields (truth_pose, state_log, frame vectors) with actuation. A controller satisfying the declared interface can still fail at runtime, or silently lack a hook. Sensor access, command execution and evaluator-only recording should have separate explicit contracts.

6. **The planner interface exposes its internals.** GlobalRoute carries a live grid, the grid search's cell path, its schedule indices and a trace of dictionaries. RouteFollowPolicy and BeliefRouteGuard directly read those fields; local_connector requires grid.free, grid.moves and grid.move_ok. A replacement planner cannot currently return only a new waypoint path. It must recreate these implementation-specific objects or supply an adapter.

7. **Failure reasons merge unrelated causes.** BeliefRouteGuard.departure combines geometry_clear with covariance feasibility into allowed; `_goto` labels a failed combined prediction motion_uncertainty. A future-segment geometry failure can therefore lead to a localization attempt. Return the failed constraint and segment separately so recovery can distinguish blocked geometry, uncertain arrival, insufficient information and controller failure.

8. **Arrival confidence controls two things at once.** BeliefRouteGuard uses the same confidence both as required PF mass and to size intermediate endpoint ellipses via the chi-square quantile. Increasing it raises the mass threshold while enlarging the region, so it is not a simple stricter-tolerance knob. Separate the planner's arrival-region definition from the executor's probability threshold if those are to be compared independently.

**How replaceable each layer is today**

| Layer | Existing seam | What prevents independent replacement |
| --- | --- | --- |
| Controller | CONTROLLERS registry, make_controller, PunchAction | Incomplete sensor/status contract, mixed coordinate frames, simulator fields, backend-specific carry/force semantics, no guaranteed trajectory shape |
| Local executor | RouteFollowPolicy can be instantiated or stepped; TactileIO exposes move/punch | Planner grid/trace dependencies, concrete PF internals, blocking I/O owns PF updates, concrete EIG selector cache used for expected punch travel |
| Global planner | plan_global is called once and returns GlobalRoute | Executor consumes grid implementation and prediction-trace indexing directly; no planner-independent route contract |
| EIG touch selector | SingleTouchSelector.choose_single returns one hand/spec | _expect_travel reads selector.last, best_ij and hands internals; returning a different selector result alone is insufficient |
| Estimator | PunchLocalizer exposes pose/covariance | Arrival reads localizer.mb.b particles directly; PF also resets a working estimate inside the controller |
| ROS adapter | RosPunchController uses similar commands | Deploy uses another library checkout and old global-route APIs; it is not currently a drop-in adapter for this branch |

**Minimal modular design to aim for — proposed, not implemented**

```text
Global planner
  input: map, initial belief, goal, motion/observation model
  output: RoutePlan (map-frame path/legs, heading intent, goal/arrival regions,
                     contact opportunities and predicted beliefs)
          |
          v
Local executor
  input: RoutePlan + BeliefSnapshot + MotionFeedback
  owns: current leg/progress, local reconnect, motion feasibility,
        replaceable arrival/departure policy and EIG touch selection
  output: motion or touch request; wait/recover/finished with a reason
          |
          v
Controller adapter
  accepts: command id + explicitly framed target/trajectory + hand request
  owns: servo/actuation, contact limits and stopping
  returns: reached/running/stalled/interrupted/timeout/cancelled,
           measured endpoint errors, odometry and passive-contact events
```

Keep the map/collision queries and shared uncertainty model as injected services, not grid-search internals inside RoutePlan. A new global planner then supplies the same route data without replacing the controller. A new local executor consumes the same route and feedback without rewriting the PF or simulator. A new controller satisfies the same framed command/status contract without changing global planning.

Use one explicit execution frame for both position and heading, or document and convert mixed frames at exactly one adapter boundary. A command must be latched once by id; feedback must distinguish reaching the goal from merely stopping. A body displacement alone must not imply a straight collision-free trajectory unless the adapter promises that tracking behavior.

Represent physical force limits separately from navigation decisions. The latter can be independently swapped for an experimental objective: endpoint-distance arrival, PF arrival probability, a predictive uncertainty bound, or another local recovery policy. Each returns a structured reason and the next viable action; a bare boolean with a catch-all guard name is insufficient.

A selector should return the touch specification and its expected travel/uncertainty together, so TactileIO need not inspect private EIG caches. BeliefSnapshot should provide pose/covariance and a region-probability query or samples, without requiring `.mb.b` layout. The estimator consumes sensor events; an evaluator consumes truth/state recordings through a separate optional interface.

The first refactor should define and validate command/status/frame semantics, then separate the shared I/O and selector helpers, then introduce a planner-independent route representation. Moving files before fixing these contracts would leave the same coupling under different names. Preserve every task, scene and map throughout.

**Observed executions**

Kinematic, global policy, belief arrival, dense contacts, seed 0; bounds 0.10 m / 15 degrees for room2 variants and 0.14 m / 20 degrees for demo_room/room830. These runs check route/executor/frame behavior; the kinematic controller has no physical force-guard implementation and cannot validate the robot's tracking or force limits.

| Room | Outcome | Sim seconds | Issued moves | Punches / hits | True goal error (m) | Global planner calls |
| --- | --- | --- | --- | --- | --- | --- |
| room2 | route_complete | 75.68 | 19 | 19 / 19 | 0.129 | 1 |
| room2_turn | route_complete | 75.68 | 19 | 19 / 19 | 0.129 | 1 |
| demo_room | route_complete | 471.90 | 91 | 141 / 86 | 0.165 | 1 |
| room830 | route_complete | 137.70 | 32 | 37 / 26 | 0.208 | 1 |

All 161 issued commands reconstructed their checked map target within 1.11e-16 m; all controller target latches matched the starting-body-frame displacement exactly. Every recorded allowed transition had PF mass above 0.95. No uncertainty veto occurred in these nominal seeds, so these runs alone do not establish that the uncertainty gate is necessary or sufficient. Demo_room rejected one straight connector geometrically and recovered locally; its global planner call count stayed one. Room2 and room2_turn produced identical global missions, as expected from their shared scene/goal.

Artifacts: [summary](/home/jjang318/tacnav/tactile-navigation/out/pipeline_review/20260910_121521/summary.json), [contract probes](/home/jjang318/tacnav/tactile-navigation/out/pipeline_review/20260910_121521/contract_probes.json), [room2 command trace](/home/jjang318/tacnav/tactile-navigation/out/pipeline_review/20260910_121521/room2/command_trace.json), [demo_room command trace](/home/jjang318/tacnav/tactile-navigation/out/pipeline_review/20260910_121521/demo_room/command_trace.json), [room830 command trace](/home/jjang318/tacnav/tactile-navigation/out/pipeline_review/20260910_121521/room830/command_trace.json). Each room folder also contains the original run JSON, route plot and run log. The two small contract probes call the real methods with controlled inputs; they are reproductions of the status/timer defects, not physical safety tests.
