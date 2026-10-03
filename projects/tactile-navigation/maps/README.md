# Reference maps

## Sky's new evaluation maps

Three new authored layouts use the same JSON manifest loader and unchanged
navigation framework: `sky_offset_gate` (offset opening),
`sky_staggered_baffles` (alternating dividers), and `sky_island_choice`
(central solid box with two alternative routes). Their JSON files live in
this folder and are registered in `execution.demo_scenes`.

Run `bash scripts/sky_evaluate.sh test`, then `plan` or `run` with a fresh
output directory. The local launcher expects the existing WSL controller
environment and checkpoint links. See [experiment design and setup](../docs/SKY_MAP_EXPERIMENT.md)
for the checkpoint override, shared settings, and interpretation limits.

After the Windows restart, remaining physical trials run sequentially with
two CPU cores and a 4 GiB RSS watchdog. Render videos after simulation ends.
Keep interrupted journals separate from completed outcomes.

Current planning defaults and four-map execution commands are in
[EVALUATION.md](../docs/EVALUATION.md#current-four-map-configuration).
Global execution and `global_plan.py` default to a route-aligned starting yaw;
`--start-heading scene` retains the authored heading. The geometry transform
below still uses the authored map heading.

`room830_modified.json` adds the approved barrier through x=-4 and the recess
wall from x=1.9 to x=3.4, aligned with the adjacent blocks' north edges.
Its authored start is (-2.5, -2) and goal is (4.4, 1.5). It is a registered
global room, independent of the unmodified `room830`.

## Reference-map import and earlier checks

These four JSON maps were copied from `tactile-navigation-global-planner`,
branch `jjang318/global`, `out/map_gen/`. Geometry and initial conditions are
unchanged; only the `_readme` link now points here.

| Map | Start (m) | Initial heading | Goal (m) |
| --- | --- | --- | --- |
| `ref1_bend` | (6, 1) | 180 degrees | (0.75, 8) |
| `ref2_hall` | (1, 1) | 90 degrees | (10, 0.75) |
| `ref3_zsuite` | (1.3, 8.4) | 0 degrees | (11.2, 1.3) |
| `ref4_room` | (1, 1) | 90 degrees | (8, 5.7) |

The JSON is the map input: `outline`, `free_space_rects`, and `obstacles`
define geometry; `start_xy`, `start_heading_deg`, `goal_xy`, and
`goal_radius_m` define the mission. Coordinates are in the authored room
frame, in metres. Headings are counterclockwise from +x. Initial headings
must be multiples of 90 degrees to preserve axis-aligned free-space rectangles.

`src/demo_map/manifest.py` adapts the original `polymap.py` geometry conversion
without its sampler or executor. It translates the start to (0,0) and rotates
the starting heading to +y. Every obstacle and the goal use the same transform.
Walls, cylinders, solid boxes, and polygon rings keep their original geometry.
Wall thickness is 0.10 m and body height is 1.50 m, matching the simulator.

The loader produces `V3Scene`, used by the current global planner, estimator,
and controllers. The vforce backend injects its bodies into MuJoCo through
`build_obstacles_v3`; a separate XML file is not needed to execute these maps.

Run one map with the actual controller from the repository root:

```bash
PYTHONPATH=src:tactile_controller/src MUJOCO_GL=egl \
  tactile_controller/.venv/bin/python scripts/punch_demo.py \
  --room ref1_bend --policy global --controller vforce --seed 0 \
  --bound-pos .14 --bound-yaw-deg 20 --no-video \
  --out out/reference_maps/ref1_bend
```

To search all four maps with shared parameters before starting a controller
(seed/refine with automatic belief-search fallback is the default):

```bash
tactile_controller/.venv/bin/python -u scripts/global_plan.py \
  --rooms ref1_bend ref2_hall ref3_zsuite ref4_room \
  --bound-pos .14 --bound-yaw-deg 20 --thin dense \
  --planner-set route_objective=distance_and_rotation \
    grid_resolution_m=.15 max_move_m=.375 \
  --out out/reference_maps/shared_plans
```

This runs `plan_global` on each map with its JSON start, heading and goal.
It saves `config.json`, `summary.csv`, `summary.json`, and each map's
`result.json` and successful `route.png`. The summary checks geometry,
stepwise uncertainty bounds and the goal confidence radius. A failed map
does not stop the batch; the command exits nonzero if any map fails.
Search-budget exhaustion is recorded separately from no route satisfying
the bounds. Use a different `--out` for each parameter set when comparing.

With `route_strategy=belief_astar`, this shared configuration passed all four
global searches on 2026-09-11, with the default filter noise, initial belief,
0.40 m standoff and 95% goal
confidence check. The 0.15 m grid gives finer arrival positions; limiting
moves to 0.375 m retains 16 movement directions. The complete batch took
about seven minutes. These are planner checks; controller execution is a
separate evaluation. The same `--planner-set` values can be forwarded to
`punch_demo.py` or `punch_eval.py` for execution.

The default `route_strategy=seed_refine` starts with geometric A*,
attracts alternatives to reachable wall contact regions,
repairs uncertainty failures and validates shortcuts. The bounds, noise model,
clearance and goal confidence check stay the same. It is a heuristic and may
return longer routes. Failed refinement falls back to the existing
`route_strategy=belief_astar` search; `seed_fallback=false` disables fallback
for comparison. Saved results report which strategy actually produced the
route. See [strategy details](../docs/GLOBAL_PLANNER_QUICKSTART.md#global-planning-strategy).

With these shared parameters, the 2026-09-11 seed/refine batch passed all four
maps in 115 seconds total versus 425 seconds for the earlier belief search.
The first three maps refined successfully; `ref4_room` used fallback. The
first three routes were longer (12.17 / 15.04 / 18.81 m versus 11.07 / 14.65 /
15.52 m), illustrating the planning-time versus route-quality tradeoff.
These are individual planner runs, including grid construction, without
controller execution. Results and per-map route plots are in
`out/reference_maps/planner_shared/seed_refine_pos014_yaw20_grid015_move0375/`.

Evaluate all four through the same execution pipeline:

```bash
tactile_controller/.venv/bin/python scripts/punch_eval.py run \
  --rooms ref1_bend ref2_hall ref3_zsuite ref4_room \
  --controllers vforce --seeds 0 --out out/reference_maps/evaluation \
  -- --policy global --bound-pos .14 --bound-yaw-deg 20
```

`--rooms all` now includes these maps and the original four room presets.
The new maps use `--policy global`; they have no legacy scripted task sequences.
Use `--controller kinematic` for a lightweight execution check.

Import validation: all four maps match the original geometry. With the default
initial belief and bounds of 0.14 m / 20 degrees, the current global planner
finds routes for `ref1_bend` and `ref2_hall`; it reports no feasible route for
`ref3_zsuite` and `ref4_room`. The imported maps and goals are unchanged.

Existing scanned room files remain in `docs/demo_maps/`. Registered JSON paths
are listed in `execution.demo_scenes.REFERENCE_MAPS`.
