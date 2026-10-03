# Global route execution

## Map evaluation entry points

| Purpose | Script |
| --- | --- |
| Execute a global route on a map | `scripts/punch_demo.py --policy global --room <map>` |
| Replay a recorded run as video | `scripts/punch_video.py` |
| Plot map geometry and an optional recorded trajectory | `scripts/plot_room_plan.py` |
| Run matched route-guard or historical eval comparisons | `scripts/compare_route_guards.py` |
| Plot comparison results | `scripts/plot_route_guard_comparison.py`, `scripts/plot_eval_comparison.py` |

Map extraction/building scripts and their input data are retained to rebuild
the scenes. `punch_demo.py` saves the planned route plot and run record, and
invokes `punch_video.py` for recorded simulation states unless `--no-video` is set.

See [the execution usage audit](EXECUTION_USAGE_AUDIT.md) for all-room commands,
comparison commands, every `src/execution` file's callers, and the PACE boundary.

Run from the tactile-navigation repository root:

```bash
PYTHONPATH=src tactile_controller/.venv/bin/python scripts/punch_demo.py \
  --policy global --room room2 --controller kinematic --out out/punch/room2_kin

PYTHONPATH=src:tactile_controller/src MUJOCO_GL=egl tactile_controller/.venv/bin/python \
  scripts/punch_demo.py --policy global --room room2 --controller vforce \
  --out out/punch/room2_vforce
```

Use `--room demo_room` for the larger map; global defaults are 0.18 m and 20°.
Earlier checks of the corrected open-end map found no route at 0.10 or 0.12 m. Its eastern
navigation boundary is x=17.744 m in the spawn frame; it is not a contact wall.
The `tactile_controller` symlink
must point to the controller repository and its environment.

## What runs

`punch_demo.py` and `global_plan.py` default to `--standoff 0.30`. This is
the nominal robot-base distance to a contact surface. With the default
`contact_band_m=0.05`, contact credit requires the executed grid centre to
be 0.25–0.35 m from that surface. Grid resolution does not enlarge this band;
the executor does not refine a cell into a hypothetical closer contact stance.
Cells farther away remain traversable when geometry permits, but receive no
contact credit for that surface. On a grid, curved or offset walls can produce
distances within the band rather than exactly 0.30 m. Use a finer grid when
the band contains too few usable cells.
The default global grid is 0.10 m, with a 0.375 m maximum planner move.
Both entry points use the first planned heading as the starting yaw;
`--start-heading scene` retains the scene heading for comparison.

1. `lane_graph.grid_constrained.plan_grid`: plan under cross-track/yaw
   uncertainty bounds. Default to geometric seeding and refinement, with
   belief-search fallback. The default objective prioritizes distance outside
   contact reach, then distance plus rotation. Contact reach uses the grid's
   sampled map regions, including walls and landmarks.
2. `execution.global_route.plan_global`: keep all available contact stations
   by default (`--thin dense`). `--thin balanced` and `--thin time` explicitly
   opt into pruning for experiments.
3. `RouteFollowPolicy`: move along the fixed global route from the PF pose.
   Intermediate legs advance by endpoint distance. Final arrival requires 95%
   PF mass in the goal disk and yaw range. Before translation, check the joint
   position/yaw prediction through the next contact opportunity without crediting
   that future contact. Excess uncertainty triggers bounded sensing. After the
   belief update, one primitive can proceed with a positional exceedance if
   geometry and yaw pass; each subsequent primitive is checked again.
   Recovery reconnects locally to the current leg. On a wall-parallel leg,
   lateral error over half a grid cell triggers corrections of at most 0.10 m,
   with a fresh PF/EIG evaluation after each. Correction continues until the
   lateral error is at most 0.05 m, then forward tracking resumes. It never
   replans the global route. An outward-only connector can restore clearance after the estimated
   pose moves inside the inflated wall buffer; it cannot cross a wall or the
   open navigation boundary.
4. `SingleTouchSelector`: select one hand/angle by EIG. Skip a touch if its
   expected information does not justify its time, including scheduled
   contacts. Stop repeated touches at the sigma target or touch limit.
5. `TactileIO`: execute moves/punches and update the PF from odometry and
   contact/free-space observations. A failed move does not trigger punching.

The global plan's dense contact updates are predictions. Actual EIG gating,
misses and contact geometry can produce a different covariance trajectory.
The bound allows gap crossings only when their cross-track prediction fits.

To prioritize shorter routes, add
`--planner-set route_objective=distance_and_rotation`. The grid search then
minimizes distance plus `turn_cost_per_rad` times commanded rotation, subject
to the same position/yaw bounds. It can choose a gap crossing even when a
longer route stays within contact reach. This changes the search objective;
the path and dense contact schedule are still generated by the planner.
The default is `route_objective=free_distance_then_distance_and_rotation`.

## Global planning strategy

Global planning defaults to `route_strategy=seed_refine` with
`seed_fallback=true` in `punch_demo.py` and `scripts/global_plan.py`.
Select `--planner-set route_strategy=belief_astar` for the original search.
Strategy and objective are independent; use
`route_objective=distance_and_rotation` to favor short routes.

The new strategy:

1. Runs geometric A* with the same collision clearance and commanded-heading
   turn cost, ignoring uncertainty. Seeds aim near the goal centre to leave
   room for the confidence disk.
2. Generates wall-attracted alternatives by increasing the penalty for travel
   outside contact reach. These use the grid's reachable contact regions at
   the configured standoff; waypoints are never projected through obstacles.
3. Replays each candidate with the existing `belief_step` model. If none is
   feasible, it backtracks before the first failing step and searches for a
   replacement suffix in a corridor around the seed. This bounded repair uses
   weighted A* (distance heuristic weight 2), informative contact normals,
   strict source/pre-contact/post-contact checks, and the goal confidence test.
4. Tries geometric bridges to shorten a feasible path, prioritizing excursions
   that leave and return to the same reachable wall face. This removes
   avoidable wall switching before generic bridges consume the trial budget.
   A surface is identified by body and normal, including wall ends tagged as
   landmarks; required detours still survive the uncertainty checks. Each accepted bridge
   improves the selected route objective and passes complete belief replay
   through the goal. Dense contact scheduling and optional thinning then use
   the usual global-route pipeline; thinning retains dense contacts if the
   reduced schedule would fail terminal confidence.

This is discrete refinement, with no guarantee of finding the shortest route.
It can produce longer routes or miss a route available to the full search.
By default, failed refinement falls back to `belief_astar` with the same bounds.
Use `seed_fallback=false` when measuring refinement alone. Failure with
fallback disabled means no feasible candidate was found, not a proof that
the map is infeasible.

The defaults allow `seed_repair_max_states=12000` total belief expansions for
repairs and `seed_shortcut_max_trials=80` bridges. Zero disables the respective
stage. Fallback has its own `max_explored_states` budget. `GLOBAL_PLAN` and the
saved route record include `requested_strategy`, actual `strategy`, and
`strategy_stats` (geometric expansions, repair expansions, accepted shortcuts,
and whether fallback ran). The existing `explored_states` count refers to
belief expansions, including fallback, rather than geometric expansions.

Run all four reference maps with the shared bounds and grid:

```bash
tactile_controller/.venv/bin/python -u scripts/global_plan.py \
  --rooms ref1_bend ref2_hall ref3_zsuite ref4_room \
  --bound-pos .14 --bound-yaw-deg 20 --thin dense \
  --planner-set route_strategy=seed_refine \
    route_objective=distance_and_rotation \
    grid_resolution_m=.15 max_move_m=.375 \
  --out out/reference_maps/seed_refine
```

Implementation: `src/lane_graph/grid_seed_refine.py`; dispatch and original
belief search: `src/lane_graph/grid_constrained.py`.

## Directional constraint

Default: `--position-constraint cross_track --bound-pos 0.18` (one sigma).
For a translation of length `l` along unit tangent `t`, use `n = (-t_y, t_x)`.
The first-order SE(2) endpoint variance is

```text
sigma_perp^2 = n' P_xy n + 2 l n' P_xy,yaw + l^2 P_yaw,yaw + process noise
```

`belief_step` propagates the full 3x3 covariance, including turn/process noise.
It checks the source and the endpoint before crediting a planned contact.
Yaw uncertainty is counted once through this propagation; the separate yaw
bound also remains. Runtime uses the PF's full covariance in the same model.
The existing model floor (`min_axis_sigma_m=0.053`) also applies to the
prediction; recorded PF axis sigmas can be smaller than this floor.
Large x uncertainty permits x travel when y/yaw are sufficiently certain,
but blocks y travel until an earlier observation reduces x uncertainty.
This is a standard-deviation bound, not a guaranteed landing-error radius.

For bounded wall-distance feedback, `n` is the normal of the **active route
leg**, rather than the normal of the corrective displacement. This also
applies to small finishing moves on that leg. The target must reduce or
preserve lateral error, lie longitudinally within the leg, and change the
wall-normal position by at most 0.10 m. The covariance always propagates the
actual translation and commanded yaw; only the projection axis changes.
The general calculation is `n' (F P F' + Q)_xy n`, with `F` built from the
actual displacement. On entering the next leg its new direction is checked;
the arrival probability, geometry checks and EIG/time threshold still apply.
The constants are `RouteFollowPolicy.WALL_CORRECTION_M` and
`WALL_CORRECTION_TOL_M`. The 0.10 m command exceeds the base controller's
0.03 m per-foot arrival tolerance while remaining below the normal 0.40 m step.

`--position-constraint max_axis` retains the old metric for explicit comparison.
Use the same numerical bound to isolate the metric change.

`--standoff` sets the task distance and the grid's contact clearance.
`--bound-pos` and `--bound-yaw-deg` set the planning bounds.
`--step`, `--max-touches`, `--touch-target` and `--touch-target-yaw-deg`
control execution cadence and stopping targets.

`--route-guard belief` is the default. `--arrival-confidence 0.95` requires
that fraction of PF mass in the final goal disk and configured heading-error
range. Intermediate endpoints are distance-based tracking targets, not
separate belief-region goals.
The condition is sufficient localization for the next motion, not a mandatory
touch event. A miss observation can also satisfy it. EIG still chooses punches.

For matched ablations, `--route-guard distance` uses mean endpoint distance;
`--route-guard contact` additionally requires the expected surface event.
Both retain the same one-step uncertainty and geometry checks. The belief mode
adds the final PF arrival condition and prediction through the next observation gap.
`--execution-pos-policy strict` retains hard positional admission; the default
`probe_then_continue` uses positional exceedance as a sensing trigger.
See [the execution uncertainty contract](UNCERTAINTY_EXECUTION.md).

The global movement/EIG ablation (`--global-motion eig`) was removed. Global
execution has one route follower. Original `punching` comparison modes remain:
`--policy eig` follows feature-derived lanes; `--policy eig_step` jointly
selects movement and punches. Both are distinct from the global planner.

## Results

The JSON records `route` (including the objective and free-travel distance),
`step_stations`, `touches_log`, `nav_trace`, `motion_checks`, `transitions` and `outcome`.
`motion_checks` records the PF covariance, translation, source/endpoint sigma,
bound and admission result. `constraint_frame` and `constraint_tangent`
identify which axis was checked, separately from the actual motion `tangent`.
`wall_correction` marks recovery evaluated in the active route frame.
`route.predicted` records the same directional
quantities for the plan. The route PNG shows
planning predictions. In belief mode, `route_complete` requires the configured
PF probability inside the goal region. `transitions` records the arrival
probability and whether the leg advanced; `motion_checks.gap` records the
departure prediction. These model checks do not guarantee true robot clearance.

Kinematic and vforce use different simulation dynamics and synthetic sensor
models. Their shared route is not a validation of deployed odometry or base
control. See [comparison with punching](PUNCHING_COMPARISON.md).
The matched 48-run tracker experiment, commands and plots are in
[the route-guard comparison](ROUTE_GUARD_COMPARISON.md).
The subsequent [eval branch comparison](EVAL_BRANCH_COMPARISON.md) includes
the MuJoCo local-reconnection deadlock and the bounded wall-feedback update.
