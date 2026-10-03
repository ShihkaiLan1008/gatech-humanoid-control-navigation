# Tactile navigation

Current workflow on `jjang318/punch+global`: a global route supplies motion
subgoals, EIG selects hand probes, and a particle filter updates pose from
odometry and touch. **Movement follows the route; EIG selects probes only.**

Global execution uses a **0.18 m** planning bound and the same execution
sensing threshold. After bounded sensing, positional uncertainty alone does not
terminate a movement; geometry and yaw remain checked. The fleet runner enables
the [angle-dependent probe profile](docs/PROBE_REACH.md). For direct runs, append
`--probe-reach-profile configs/probe_reach_ee_qpos_v1.json`.
See [execution settings and manuscript wording](docs/UNCERTAINTY_EXECUTION.md).
The nominal contact standoff is **0.30 m**, with a 0.25–0.35 m contact band.
Global runs spawn at the planned starting heading. The evaluated four-map
configuration and its explicit overrides are in [EVALUATION.md](docs/EVALUATION.md#current-four-map-configuration).
Global planning uses a 0.10 m grid, 0.375 m maximum move and 20° yaw bound.

## Evaluate

Run these from the repository root with the existing
`tactile_controller/.venv` environment. The `footstep` controller uses
`models/0911_footmimic_noforce/model_34000.pt`. Use an NVIDIA GPU for the learned
controllers; `kinematic` is the CPU alternative for planner checks.
Footstep starts and walks with hands at `(0.20, ±0.15, 0.10)` m in pelvis axes.
Arm actions default to 2× speed (`arm_speed_scale=2`): transitions ease in/out
over at least 0.3 s, yaw is limited to 1.0 rad/s, and probing uses 0.16 m/s.
Arm holds/waits also halve; base timing stays the same. These settings are printed at startup
and saved in the run JSON. Older videos and PACE snapshots retain their old settings.

**Local:**

```bash
tactile_controller/.venv/bin/python scripts/punch_eval.py run \
  --controllers footstep --rooms room2 --seeds 0 --out out/evaluation/room2
```

**PACE — submit from this PC:**

```bash
scripts/pace/submit_eval.sh --controllers footstep --rooms room2 --seeds 0
```

Both commands run the same global/PF/EIG pipeline. Replace `room2` with
`demo_room`, `room830`, or another registered room. Use `--rooms all` for all
maps and `--seeds 0 1 2` for three concurrent seed processes. One seed launches
one process; maps/controllers still run in order. Choose a new local `--out` for each
comparison; existing trials are not overwritten.

**Four seeds per GPU job, split across PACE A100 and SKYNET 2080Ti:**

```bash
tactile_controller/.venv/bin/python scripts/cluster/submit_batch.py \
  --pace-rooms ref1_bend demo_room \
  --skynet-rooms ref2_hall room830_modified --seeds 0 1 2 3
```

Uploads one frozen source snapshot and uses [fixed evaluation settings](configs/evaluation_batch.json).
Each map gets one GPU, 12 CPUs, 32 GB RAM and eight hours. It saves metrics,
recorded states and videos in the fleet folder format, then copies and verifies
them on Dropbox. A CPU follow-up retries uploads after each GPU job ends.
Local receipts are under `~/tacnav-evaluation/<timestamp>_batch4/`.
Use `--source /path/to/tactile-navigation` to pin an existing evaluated checkout.
Each seed has a 6 GB RSS watchdog. The 32 GB Slurm request is a hard limit only
on clusters with memory enforcement enabled; SKYNET currently does not enforce it.

Track both clusters (add `--collect` to download and combine their metrics):

```bash
tactile_controller/.venv/bin/python scripts/cluster/track_batch.py \
  ~/tacnav-evaluation/<timestamp>_batch4
```

For the optional angle-dependent probe reach, append
`-- --probe-reach-profile configs/probe_reach_ee_qpos_v1.json` to either command.
EIG and the hand command share the same tip endpoint after a 3 cm margin.
Omitting this option uses uniform probe geometry. See the
[reach mapping and update instructions](docs/PROBE_REACH.md).

Local results are in `out/evaluation/room2/`: `summary.json` and `runs.csv`
report success rate, state-estimation error, touch counts and trajectory metrics.
Each `<controller>/<room>/seed_<N>/` contains `run.log`, `.eval.json`, the replay
JSON and saved robot states. Local batch evaluation skips video; render it later:

```bash
PYTHONPATH=src:tactile_controller/src MUJOCO_GL=egl \
  tactile_controller/.venv/bin/python scripts/pace/render_eval.py out/evaluation/room2
```

PACE rsyncs the current working tree, including uncommitted source, maps and
selected checkpoints, then submits evaluation, video rendering and Dropbox
upload. It reuses the configured PACE environment. Submission prints job IDs and
remote paths; local receipts are in `out/pace_submissions/`. Results and videos
upload to `TacNav/tactile-navigation/runs/pace_<evaluation-job-id>/` on Dropbox.
Use `--endpoint local` to keep them on PACE, `--no-video` to skip rendering, or
`--dry-run` to preview submission. PACE defaults to one A100 and one hour for
evaluation; increase `--time` for larger suites.

## Module swaps and ablations

Options after `--` are forwarded to the demo by **both** commands above.
Keep maps, seeds and scoring the same when comparing changes.

**Opt-in baselines:** append `-- --preset NAME`. Omitting it keeps the original
execution, defaults and output layout. Use a separate local `--out` per preset.

| Preset | Change from the full pipeline |
| --- | --- |
| `full` | Original PF + EIG + constrained global route + local recovery. |
| `random_touch` | Uniform random feasible hand/angle, without replacement within each sensing batch. |
| `sweep4` | Four directions from rightmost to leftmost, stopping early at the uncertainty target. |
| `odom_astar` | Odometry only, no probes, geometric shortest-path A*. |
| `odom_global` | Odometry only, no probes, same initial constrained route and headings as `full`. |
| `no_recovery` | Full pipeline with dedicated reconnects and recovery retries disabled. |

```bash
# Local
tactile_controller/.venv/bin/python scripts/punch_eval.py run \
  --controllers footstep --rooms room2 --seeds 0 --out out/evaluation/sweep4 \
  -- --preset sweep4

# PACE, submitted from this PC
scripts/pace/submit_eval.sh --controllers footstep --rooms room2 --seeds 0 \
  -- --preset sweep4
```

Keep the same controller, checkpoint, parameters and seed list across maps;
change only `--rooms`. Presets contain no room-specific tuning. Random/sweep
share the existing uncertainty targets and four-attempt default, without EIG
scoring or its early exit after an uninformative probe. Details:
[baseline definitions](docs/EVALUATION.md#opt-in-baselines).

| Change | Option / implementation |
| --- | --- |
| Controller | `--controllers footstep vforce kinematic` runs each controller. Add a new implementation through [`CONTROLLERS` / `PunchController`](src/execution/controllers.py). |
| Checkpoint or arm settings | Append `-- --override checkpoint=/path/to/model.pt`; `arm_speed_scale=1` restores the previous arm timing (default: `2`). Other footstep overrides include nominal `hand_speed_m_s`, `hand_yaw_rate_rad_s`, `hand_transition_min_s` and `carry_forward_m`. |
| Global search | Append `-- --planner-set route_strategy=belief_astar` to compare with default `seed_refine`. Set `seed_fallback=false` when testing seed/refine without fallback. |
| Route progression | Append `-- --route-guard distance` or `-- --route-guard contact` to compare with default `belief`. |
| Uncertainty constraint | Append `-- --position-constraint max_axis` to compare with default `cross_track`. |

For a new global planner, adapt its output to [`GlobalRoute`](src/execution/global_route.py).
[`RouteFollowPolicy`](src/execution/global_route.py) selects the next route/local
recovery target; [`TactileIO`](src/execution/punch_rules.py) converts it to
body-relative `MoveCmd` plus hand commands for the controller. Replace these
classes for a different local execution method. Opt-in selectors and the odometry
adapter live in [`baselines.py`](src/execution/baselines.py); the original EIG
selector stays in `global_route.py`. These are code boundaries, not independent
plugin registries. Without a preset, global mode still forces PF and EIG;
`--select fixed` or `--estimator none` alone is not an ablation.
Keep [`DemoRunLog`](src/execution/demo_logging.py) and
[`RunRecorder`](src/execution/evaluation.py) connected so every variant produces
the same metrics.

Maps: [map folder and formats](maps/README.md).
More detail: [evaluation and PACE](docs/EVALUATION.md) ·
[module interfaces and diagram](docs/PIPELINE_MAP.md) ·
[global planner](docs/GLOBAL_PLANNER_QUICKSTART.md).
