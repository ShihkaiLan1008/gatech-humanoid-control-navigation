# Comparison with jjang318/eval

Reference: clean `jjang318/eval` / `origin/jjang318/eval` at
`707387fcbdc4cef72a343d6fb0d931ef5ceb5599`, fetched for this comparison.
It preserves the September 7 navigation implementation and includes the
completed September 9 MuJoCo reproduction. The comparison checkout remains
unchanged at `/home/jjang318/tacnav/demo-room-repro/tactile-navigation`.
The original tables below describe the implementation on `jjang318/punch+global`
before bounded wall feedback. The follow-up section records that update.

## Actual execution differences

| Component | eval | Current |
|---|---|---|
| Global movement | Fixed route, next point from PF progress | Fixed route, next point from PF progress |
| Planning objective | Distance plus rotation | Minimize distance outside contact reach first, then distance plus rotation |
| Position constraint | Largest position axis sigma | Cross-track sigma for each translation |
| Demo bound | 0.40 m / 20 degrees | 0.14 m / 20 degrees |
| Room2 bound | 0.30 m / 15 degrees | 0.10 m / 15 degrees |
| Contact schedule | Balanced pruning | Dense |
| Scheduled first punch | Can bypass minimum EIG/time threshold | Must pass EIG/time threshold |
| Leg/goal advancement | Longitudinal progress >= leg end minus 0.10 m | 95% PF mass in planned arrival region / actual goal disk |
| Execution admission | No predictive covariance or continuous-map gate | Covariance prediction through next contact opportunity and map clearance |
| Recovery | Next point on current leg from updated pose | Local connector to current leg, with bounded arrival recentering |
| PF | Range contact/recovery model | Also penalizes wall-crossing particles and refutes wall-entry miss rays |
| Demo right opening | No navigation cutoff | Halfspace cutoff at x=17.744 m, without contact credit |

Both already propagate joint position/yaw covariance in the global planner.
Both use the same nearest-touchable-surface heading heuristic. EIG selects
hand and punch angle; it does not select base heading in either route follower.
The vforce controller, `TactileIO`, punch actions and MuJoCo backend source
have no diff between these versions. The additional current kinematic
disturbance options are off by default; nominal controller behavior matches.

Exact references in the eval checkout:

- `src/execution/global_route.py:188`: balanced pruning default.
- `src/execution/global_route.py:593`: scheduled-first-punch EIG exception.
- `src/execution/global_route.py:631`: movement without the new belief gate.
- `src/execution/global_route.py:678`: longitudinal stage completion.
- `src/lane_graph/grid_constrained.py:192`: wall-facing heading heuristic.

Current equivalents are `src/execution/global_route.py` (`plan_global`,
`_touch_until`, `_motion_prediction`, `step_station`) and
`src/execution/route_guard.py` (`arrival`, `departure`, `local_connector`).

The scheduled-punch EIG exception exists in eval's source, but no recorded
punch fell below the threshold in its 16 new kinematic runs or saved MuJoCo
run. It does not explain the observed completion difference. Similarly,
pruning predicts fewer contacts; runtime EIG still produces many actual
touches (23 planned versus 110 actual in the saved eval MuJoCo run).

## New kinematic runs

32 runs: both complete versions × room2/demo_room × seeds 0/1 × nominal,
short strokes, initial offset and yaw drift. Both used the same Python 3.13
controller environment and `--touch-target 0.08`. Disturbances match the
[route-guard benchmark](ROUTE_GUARD_COMPARISON.md). Each version imports its
own unmodified executor, filter, map and planner modules. This compares the
complete versions; differences cannot be attributed solely to the tracker.

The eval demo path is replayed from its verified MuJoCo record, including
balanced pruning. Native room2 is planned from scratch. Current uses its
current planner in both rooms. Each path is shared across that version's
eight room trials. No global replanning occurs during execution.

| Room | Version | Reports complete | True goal reached | Median time (s) | Median punches | Route (m) / planned contacts |
|---|---|---:|---:|---:|---:|---:|
| room2 | eval | 8/8 | 8/8 | 73.73 | 20 | 5.083 / 1 |
| room2 | current | 8/8 | 8/8 | 76.36 | 19 | 6.130 / 29 |
| demo_room | eval | 8/8 | 7/8 | 402.07 | 132 | 25.510 / 23 |
| demo_room | current | 8/8 | 8/8 | 503.97 | 151.5 | 27.666 / 116 |

Eval is a working nominal baseline. In the yaw-drift/seed-1 kinematic case,
it reports completion 1.76 m from the goal and reaches 4.65 m localization
error. Its point-base trajectory crosses wall geometry; the kinematic
backend cannot physically block the base. Current reaches the goal in that
case. Current is slower and still breaches its own 0.35 m clearance margin
in four demo trials (minimum clearance 0.226 m). Eval's planning margin is
0.30 m; current's is 0.35 m, so the plots show both thresholds explicitly.

The prior 48-run comparison only compared three current-branch guards. It
did not establish superiority to eval or validate MuJoCo performance.

## MuJoCo comparison and the current regression

Both use controller commit `3396d5a5c20124b9e85e285b344f25f29fad41b2`, the
bundled `vforce_model_60000_v1.pt`, seed 0, and `--touch-target 0.08`.
Eval is its saved September 9 reproduction; current is a new run. This is
one run per version, with each version's own plan, bounds and filter.

| Result | eval | Current |
|---|---:|---:|
| Outcome | route_complete | motion_uncertainty |
| Recorded simulated time (s) | 523.70 | 449.22 |
| Punches / landed | 110 / 90 | 91 / 56 |
| Final true distance to goal centre (m) | 0.515 | 7.304 |
| Median localization error (m) | 0.082 | 0.063 |
| Maximum localization error (m) | 0.438 | 0.308 |
| Minimum true wall clearance (m) | 0.336 | 0.276 |
| Fallen | No | No |

Eval traversed the room. Its longitudinal completion rule declared success
0.115 m outside the actual 0.40 m goal radius. Current stopped midway up the
vertical corridor. Its shorter duration and smaller observed maximum error
are not completion advantages: it executed only part of the route.

At the current stop, t=448.22 s (the record includes one final settling second):

1. Estimated pose is (13.363, 4.606); true pose is (13.411, 4.700).
2. The fixed route is x=13.5, travelling +y. Its 0.137 m lateral offset exceeds
   half a grid cell, so `step_station` selects the nearest route point:
   (13.5, 4.606), a pure +x correction.
3. That correction makes y the cross-track direction. The PF has
   `P_yy=0.024254 m²`, so its source sigma is 0.155738 m. Endpoint prediction
   is 0.155771 m, exceeding the 0.14 m bound. Geometry passes; yaw sigma is
   only about 1.5 degrees. The translation is never issued.
4. The best available punch has EIG 0.087 nats, below the 0.1495 nats time
   threshold, so it is skipped and execution returns `motion_uncertainty`.

The last leg transition had already passed at probability 0.966. The failure
is the recovery target's direction combined with accumulated along-wall
uncertainty. Applying the directional bound to the instantaneous pure-lateral
correction creates a deadlock here.

The corrective displacement restores the existing leg's wall-distance error;
it does not start a new leg. Its constraint should therefore use the active
leg's normal, while covariance propagation still uses the actual displacement.
Selecting a farther point just to rotate the constraint axis was rejected.

Records: `current_vforce/demo_room_vforce_pf_global_s0.json`, its last
`motion_checks`, and `stop_snapshot_predictions.json` under the comparison
output directory. Figures: `mujoco_comparison.png/pdf`.

## Bounded wall-feedback update

Wall-parallel recovery now targets the closest point on the active leg in
steps of at most 0.10 m, re-evaluating the posterior and EIG after each step.
It starts above 0.10 m lateral error (half a grid cell) and continues until
the error is at most 0.05 m. Small ordinary tracking/endpoint corrections
also use the active leg's normal. Actual SE(2) covariance propagation is
unchanged, including yaw effects from the actual displacement. The next leg
uses its own normal; arrival, geometry and EIG/time gates remain in force.

`belief_step(..., constraint_tangent=...)` separates the measurement frame
from motion propagation. `_wall_tracking_tangent` restricts its runtime use
to wall-parallel legs, targets within that leg, non-increasing lateral error,
and at most 0.10 m change normal to the leg. An outward step may incrementally
restore an inflated clearance margin but may not cross actual wall geometry
or the open navigation boundary.

All 16 follow-up kinematic runs reached the true goal (8/8 in each room).
Room2's median time/punch count remained 76.36 s / 19. Demo_room's was
507.13 s / 155.5. All issued wall-recovery commands were at most 0.10 m.
The minimum true demo clearance remained below the 0.35 m planned margin
(0.227 m); small corrections do not make the PF mean a clearance guarantee.

The new MuJoCo run completed and reached the true goal. The before/after
global paths are identical, with the same 0.14 m bound and controller.
These are full runs, not identical physical-state replays; equal seeds did
not produce identical trajectories. The saved blocked posterior is also
covered directly by a regression test.

| Result | Before wall feedback | Small wall feedback |
|---|---:|---:|
| Outcome | motion_uncertainty | route_complete |
| Recorded simulated time (s) | 449.22 | 813.40 |
| Punches / landed | 91 / 56 | 189 / 96 |
| True goal distance (m), radius 0.40 | 7.304 | 0.188 |
| Maximum localization error (m) | 0.308 | 0.208 |
| Minimum true wall clearance (m) | 0.276 | 0.326 |
| Fallen | No | No |

At t=664.40 s the new run issued one 0.10 m wall correction from estimated
(13.3932, 9.7898). The old motion-normal source sigma was 0.20246 m; the
active-route source sigma was 0.05413 m and the full guarded prediction
was 0.05993 m, below 0.14 m. At t=666.14 s, estimated lateral error had
fallen from 0.1068 m to 0.0184 m. The true base moved 0.056 m toward the
wall. The next EIG-selected punch landed at t=668.22 s, and forward
tracking resumed. No issued command failed its recorded uncertainty guard.

The run remained slower than eval (813.40 versus 523.70 s). A separate
clearance-restoration branch issued 39 very short corrections near the
goal, taking about 24 simulated seconds before tracking resumed. These
commands were around the controller deadband; this residual hesitation
is not fixed by the bounded wall-error correction. True clearance also
briefly fell below the 0.35 m planning margin. This is one successful
MuJoCo run, not a robustness claim.

Validation: 48 focused tests passed, including unchanged covariance under
a different projection, rejection of actual turns, future-turn gating,
bounded correction commands, geometry escape, and probabilistic arrival.

Follow-up records and figures are under `out/wall_feedback/validated/`.
Reproduce with the commands below, replacing the current kinematic output
directory by `out/wall_feedback/validated/kinematic` and the current vforce
output directory by `out/wall_feedback/validated/vforce`. Add
`--wall-feedback out/wall_feedback/validated` to the plotting command to
compare against both preserved baselines.

## Reproduce

From the current repository root:

```bash
PYTHONPATH=src ../demo-room-repro/tactile_controller/.venv/bin/python \
  scripts/compare_route_guards.py \
  --eval-repo ../demo-room-repro/tactile-navigation \
  --eval-demo-record ../demo-room-repro/tactile-navigation/out/punch/merged/demo_room_vforce/demo_room_vforce_pf_global_s0.json \
  --touch-target 0.08 --seeds 0 1 --jobs 2 \
  --out out/eval_branch_comparison/eval_kinematic

PYTHONPATH=src ../demo-room-repro/tactile_controller/.venv/bin/python \
  scripts/compare_route_guards.py --modes belief \
  --touch-target 0.08 --seeds 0 1 --jobs 2 \
  --out out/eval_branch_comparison/current_kinematic

PYTHONPATH=src:../demo-room-repro/tactile_controller/src MUJOCO_GL=egl \
  ../demo-room-repro/tactile_controller/.venv/bin/python -u scripts/punch_demo.py \
  --room demo_room --policy global --controller vforce --seed 0 \
  --bound-pos 0.14 --bound-yaw-deg 20 --route-guard belief \
  --touch-target 0.08 --planner-set max_explored_states=800000 \
  --override checkpoint=../demo-room-repro/tactile_controller/checkpoints/decoupled_vel_force_mpc/vforce_model_60000_v1.pt \
  --no-video --out out/eval_branch_comparison/current_vforce

PYTHONPATH=src ../locomanip-deploy-recovery/.pixi/envs/default/bin/python \
  scripts/plot_eval_comparison.py \
  --eval-vforce ../demo-room-repro/tactile-navigation/out/punch/merged/demo_room_vforce/demo_room_vforce_pf_global_s0.json
```

Records and plots: `out/eval_branch_comparison/`. Each kinematic directory
includes `source.json` with the revision, source paths and exact arguments.
`comparison_metrics.json` contains the actual goal distances, clearance and
localization errors. `movement.png` shows seed 0; `yaw_drift_seed1.png` shows
the failing eval stress case explicitly.
