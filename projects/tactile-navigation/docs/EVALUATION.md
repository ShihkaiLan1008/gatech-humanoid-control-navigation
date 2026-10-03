# Demo evaluation and logging

Module diagram, data passed between modules, and controller candidates:
[PIPELINE_MAP.md](PIPELINE_MAP.md).

The existing `scripts/punch_demo.py` remains the execution entry point. All
registered tasks/maps remain available. Evaluation observes the run; it does
not choose moves, probes, or recovery actions.

Global runs load maps and initial conditions through `execution.demo_scenes`.
They do not import `punch_tasks` or compute its scripted feature sequences.
The evaluation runner and replay also use this scene registry. `punch_tasks`
is loaded only for the legacy stations/contact/eig/eig_step policies.

The four reference rooms (`ref1_bend`, `ref2_hall`, `ref3_zsuite`, `ref4_room`)
are available through the same `--room`/`--rooms` options with `--policy global`.
`--rooms all` includes them. Their editable JSON maps and commands are in
[`maps/README.md`](../maps/README.md).

## Current four-map configuration

The main workspace integrates evaluated navigation commit `75ca4fa` while
retaining the baseline presets and angle-dependent reach implementation.
`room830_modified` is registered alongside the original rooms and reference maps.
Use the same settings as the four workstation trials:

```bash
PYTHONPATH=src:tactile_controller/src MUJOCO_GL=egl \
  tactile_controller/.venv/bin/python scripts/punch_eval.py run \
  --controllers footstep \
  --rooms ref2_hall ref3_zsuite room830_modified ref1_bend \
  --seeds 1 --out out/evaluation/current_four_maps \
  -- --bound-pos .18 --bound-yaw-deg 20 \
  --execution-pos-trigger .18 --execution-pos-policy probe_then_continue \
  --route-guard belief --start-heading route --thin dense --standoff .30 \
  --probe-reach-profile configs/probe_reach_ee_qpos_v1.json \
  --override arrive_pos_tol_m=.03 --override diagnose_moves=true --max-s 1800 \
  --planner-set route_strategy=seed_refine seed_fallback=true \
    grid_resolution_m=.10 max_move_m=.375 \
    route_objective=free_distance_then_distance_and_rotation
```

This command runs the maps sequentially on the local PC. For a distributed
campaign, assign one room per workstation. `scripts/fleet_eval.py` applies
these shared settings, defaults to seeds 0–19, writes crash-recoverable
telemetry, and uploads each completed run to the supplied `--dropbox` prefix.
It records the actual source commit and working-file hashes; a separate
`host_provenance.json` is optional. Use `--diagnose-moves` for per-foot diagnostics.
Fleet folders use `[room]_original_[seed]_[MM_DD_HH]` in America/New_York time.
Fleet execution saves replay states with video disabled; render those separately
with `scripts/pace/render_eval.py`. No new run starts merely by updating source.

The global defaults are 0.18 m position bound/trigger, 20° yaw bound, 0.30 m
nominal standoff, 0.10 m grid, 0.375 m maximum planner move, route-aligned
spawn, and seed/refine with fallback. Explicit planner overrides still apply.
The reach profile remains an explicit option.
Contact credit is granted only within 0.25–0.35 m of an executed grid point;
free transitions can be farther from walls. Clearance corrections request at
least 0.08 m and wall-distance corrections are capped at 0.10 m. Footstep
arrival uses a 0.03 m tolerance per foot; a clearance timeout is logged and
following resumes from the updated belief.

The original four seed-1 trials reached all four physical goal regions without
falling. Three reported `route_complete`; modified room830 reported
`arrival_uncertain`. Physical goal membership (`goal_reached` with no fall)
and framework `success` must remain separately reported. These four trials
used the isolated evaluated source; the integrated baseline code needs its
own trials before claiming equivalent physical outcomes.

## Opt-in baselines

Pass `-- --preset full|random_touch|sweep4|odom_astar|odom_global|no_recovery`
to either `punch_eval.py run` or `scripts/pace/submit_eval.sh` (choose one name).
For `punch_demo.py`, pass `--preset NAME` directly. Without this argument,
the original path, defaults, filenames and settings remain unchanged. Existing
remote source snapshots and their queued runs are not updated by local edits.
Use a fresh output directory for each baseline campaign.

Presets change modules, not map-dependent parameters. Keep the checkpoint,
controller, bounds, grid resolution, noise, probe budget, arm timing, timeouts
and seed schedule fixed when changing rooms. A room supplies geometry and its
authored start, goal and heading. Existing explicit CLI overrides still apply.

- `full`: existing behavior, including at most four attempts per sensing batch
  (`--max-touches`), EIG utility rejection and the measured-progress early exit.
- `random_touch`: sample without replacement from the same hand/angle grid as
  EIG. Its RNG is independent of the controller and PF RNGs. Safe air probes
  remain eligible; there is no EIG or predicted-hit preference.
- `sweep4`: body-frame directions -110°, -36.67°, +36.67°, +110°, with the right
  hand for the first two and the left for the last two. Unsafe guard positions
  and probes crossing a declared open end are skipped. At most four probes
  constitute one sweep; the sequence restarts on the next sensing batch.
- Both probe baselines check the same position/yaw targets, departure prediction
  or arrival condition as EIG, and stop as soon as that condition is satisfied
  (including before the first probe). They do not stop after one uninformative
  miss. Exhausting the budget does not satisfy a departure or arrival condition.
  Under the shared `probe_then_continue` policy, a remaining positional
  exceedance can permit one primitive after sensing; geometry and yaw must pass.
- `odom_astar`: distance-only A* on the same collision-checked grid. The body
  follows travel headings; there are no sensing stations or tactile rewards.
- `odom_global`: the same deterministic initial planner, configuration, path
  and headings as `full`; runtime does not credit the planned contacts. Compare
  the recorded route hash to verify matching routes for matching inputs.
- Both odometry baselines read live onboard encoder/IMU odometry. They neither
  instantiate the PF nor apply active/passive tactile corrections or map-based
  pose corrections. They follow geometric arrival conditions without uncertainty
  gating. Diagnostic covariance propagates without observations. Physical force
  stops and geometric local reconnects remain active.
- `no_recovery`: keep ordinary route tracking, EIG/PF, uncertainty checks and
  physical stops; disable dedicated wall/route/clearance connectors and arrival
  recentering. An incomplete or obstructed move terminates instead of retrying.
  This does not disable the PF's existing outlier/observation handling.

All presets use the existing `DemoRunLog`/`RunRecorder` metrics. Explicit presets
add their name, resolved modules and route hash to metadata, plus a `BASELINE`
event. Probe batch reasons remain in `step_stations` and `touches_log`; movement
checks identify reconnect commands. Pure odometry has no PF snapshots; its live
estimate and ground-truth scoring still appear in the canonical `.eval.json`.
Replay filenames include the explicit preset. No preset is inferred from a room.

## Logging layers

| Layer | Responsibility | Replacement contract |
| --- | --- | --- |
| `execution.evaluation.RunRecorder` | Timestamped poses, probe outcomes, generic events, goal and run metadata | Plain data in one declared map frame; no controller, PF, planner, or simulator imports |
| `execution.evaluation.compute_metrics`, `aggregate_runs`, `write_report` | Recompute per-run metrics and aggregate independent trials | Versioned evaluation records; identical definitions for every implementation |
| `execution.demo_logging.DemoRunLog` | Adapt control ticks, PF updates, EIG decisions, and probe results; retain existing replay fields | The current demo's observer callbacks; backend/frame conversions live here |
| `scripts/punch_eval.py` | Run the existing demo across maps, controllers, and seeds; produce CSV/JSON | Controller/task registries plus forwarded demo arguments; no duplicate execution loop |

`TactileIO.on_probe` reports each completed or interrupted attempt once.
Passive contacts are recorded as events and do not inflate commanded probe
counts. New executors can call the same recorder methods directly.

## Metrics and definitions

- **Position estimation error:** map-frame Euclidean distance between the
  estimated and true base positions. Mean and RMSE are time-weighted over the
  trajectory; maximum and final error are also reported.
- **Yaw estimation error:** absolute wrapped angular difference, reported in
  degrees, with time-weighted mean and RMSE.
- **Success:** execution reports completion, the true final base position is
  inside the scene's goal disk, and the robot has not fallen. Reported completion
  and goal membership remain separate fields. Timeout, planning failure, and
  process startup failure remain in the attempted-trial count.
- **Touch counts:** commanded attempts, landed contacts, completed misses,
  interrupted attempts, and contacts/attempts. An interrupted attempt that already
  contacted can contribute to both contact and interruption counts.
- **Efficiency:** execution duration, wall-clock duration, actual planar travel
  distance, and final goal distance. The extra second used for replay after
  termination is excluded from execution metrics.
- **Coverage:** fraction of execution time with valid truth/estimate pairs, and
  fraction with truth for travel distance. Missing values are `null`, never zero
  error. Runs without enough data for goal verification have unknown success;
  the report exposes the success denominator and unknown-run count.

Mean error is the trapezoidal integral of sampled error divided by scored
duration. Missing intervals are not bridged. Same-timestamp samples before and
after a belief correction preserve the instantaneous jump without adding time.
Aggregation gives each trial equal weight, includes failure outcomes, and reports
the number of runs contributing to each metric. Groups retain all experimental
settings except seed and output/video options; different settings are not pooled.

The PF updates at primitive boundaries. Between updates the adapter propagates
its last posterior mean using the live onboard odometry pose. This avoids scoring
a stale stationary PF mean while the base moves. It **does not** call the consuming
`odometry_increment`, run filter prediction, reset the estimator, or sample noise.
Truth is read separately for scoring and is never supplied to action selection.
Full covariance snapshots are saved at belief-update events.
Non-finite diagnostic scores use the strings `Infinity`, `-Infinity`, and `NaN`
in JSON, preserving rejected-candidate scores without nonstandard JSON numbers.

## Run or summarize

From the repository root, a normal demo run now also writes `<stem>.eval.json`:

```bash
PYTHONPATH=src:tactile_controller/src MUJOCO_GL=egl \
  tactile_controller/.venv/bin/python scripts/punch_demo.py \
  --room room2 --policy global --controller vforce --seed 0 --no-video \
  --out out/evaluation/room2
```

Run every registered room with three seeds through the same demo:

```bash
PYTHONPATH=src:tactile_controller/src MUJOCO_GL=egl \
  tactile_controller/.venv/bin/python scripts/punch_eval.py run \
  --rooms all --controllers vforce --seeds 0 1 2 --out out/evaluation/all_rooms \
  -- --policy global --bound-pos .14 --bound-yaw-deg 20
```

Use `--controllers kinematic vforce` to run both, or forward a different
`--policy`, `--planner-set`, `--loc-set`, or controller `--override` after `--`.
The `footstep` controller loads `models/0911_footmimic_noforce/model_34000.pt`
from this repository by default; the checkpoint can stay in that location.
Use `--override checkpoint=/path/to/model.pt` after `--` to select another
compatible joint foot-mimic checkpoint. Both actors are loaded strictly.
Arm actions default to `arm_speed_scale=2`: translation is capped at 0.32 m/s
and yaw at 1.0 rad/s. This multiplies the nominal `hand_speed_m_s` and
`hand_yaw_rate_rad_s` and divides `hand_transition_min_s` and arm-only waits;
`arm_speed_scale=1` restores the previous timing. Base timing is independent.
New arm actions use a smoothstep profile with zero starting/ending speed and
at least 0.3 s of transition; larger changes take longer to respect both rate
limits. Startup and walking carry use the same transition. Each probe first transitions to
its guard, then approaches at 0.16 m/s. Contact discards the unachieved target
and holds the measured encoder-FK pose for 0.1 s, allowing retreat under load
but no force-seeking re-approach, before withdrawing at up to 0.32 m/s.
The initial and walking hand targets are `(0.20, ±0.15, 0.10)` m in the pelvis
frame (`carry_forward_m` overrides the forward offset). Before every base
move, both hands return to these targets while the base holds; the movement
then uses this compact carry. The EIG sensing rays keep their full reach.
Contact during a guard transition stops that transition and reverses it;
localization receives the contact point without an assumed surface normal.
The run and evaluation metadata record the resolved checkpoint and hand-motion
settings. `ARM_TRANSITION` events record the requested poses and transition time.
The footstep path now estimates base motion from encoder FK of the supporting
foot and the IMU gyro. Roll/pitch come from gyro/accelerometer fusion; actor hand
poses/errors also come from FK. No simulator pose or velocity enters execution.
Simulator truth is retained only for evaluation/replay. The estimator assumes
flat ground and a nonslipping support foot; it does not receive synthetic
truth-based odometry. The removed `odom_sigma_*`, `odom_bias_frac`, `odom_rw_gain`
and `imu_yaw_sigma_per_rad` footstep overrides are therefore no longer accepted.

Only the two trained actors run: no critic observations, rewards, training
metrics, random hand targets or automatic goal/heading selection. All map and
controller choices remain available. Vforce still uses its existing synthetic
odometry, so this privilege-free execution claim specifically covers footstep.
See [the module diagram](PIPELINE_MAP.md) and
[proprioceptive validation artifacts](../out/footstep_proprio/).

Odometry repair, 2026-09-12: support is selected from sole heights at both ends
of each sensor interval before integrating displacement. The previous delayed
switch integrated a lifting foot's swing into the base estimate. Isolated
backward/lateral/turn checks now report completion, with 1.5/2.5/1.9 cm endpoint
displacement-estimation errors. Full Room2/seed 0 mean position error decreased
from 0.360 m to 0.080 m, but the run fell during corner probing (6/10 contacts),
so full navigation success remains unvalidated. [Records and comparison](../out/odom_repair_20260912T013034Z/).
Replay now draws the continuously logged estimate in cyan separately from the
blue PF snapshots/uncertainty ellipse. PF updates still occur at primitive
boundaries; rendering does not change the estimator.

Validation on 2026-09-11: 108 focused tests passed. A full Room2/seed-0
global run completed with 30 simulator pose/velocity properties blocked, but
navigation **failed** (`reconnect_infeasible`, 0/5 contacts, 0.157 m mean
position error, no fall). The current footstep path is not yet validated for
successful room execution. See [the recorded trajectory](../out/footstep_proprio/room2_sole_support_clock/trajectory.png).
The `stations` policy remains a legacy oracle comparison and reads true pose;
use `--policy global` for the execution path audited here.

The suite writes per-trial `command.json`, `run.log`, replay data and evaluation
data, plus `runs.csv`, `summary.json` and `attempts.json` at the suite root.
Use a new output directory for a new suite; existing trials are not overwritten.

Recompute or combine reports without running a controller:

```bash
PYTHONPATH=src python scripts/punch_eval.py summarize \
  out/evaluation/room2 out/evaluation/all_rooms --out out/evaluation/report
```

The summarizer reads `*.eval.json`; old replay JSONs have only event-sampled
beliefs and cannot reconstruct the new continuous estimation metric. The legacy
replay JSON, state archive, and video paths remain available. `--no-eval` disables
the new recorder for an execution-equivalence check.

## PACE batch jobs

Logging, metric computation and the kinematic controller run on CPU. For the
current `vforce` environment, request one NVIDIA GPU: a CPU-only initialization
check failed in the controller's cuRobo reach-pool setup despite the backend's
top-level CPU fallback.

Submit **from your workstation**, in this repository:

```bash
scripts/pace/submit_eval.sh --controllers footstep --rooms room2 --seeds 0
scripts/pace/submit_eval.sh --controllers vforce footstep --rooms all \
  --seeds 0 1 2 --time 04:00:00 \
  -- --policy global --bound-pos .14 --bound-yaw-deg 20
```

The script rsyncs the **current working tree, including uncommitted changes**:
navigation source/scripts/configs, all maps (including `docs/demo_maps`), tactile
world-model files, matching controller source/robot assets, cuRobo source, and
the selected controller checkpoint(s). It excludes virtualenvs, Git history,
training logs and previous evaluation outputs. A checkpoint supplied through
`-- --override checkpoint=/local/path/model.pt` is copied and its argument is
rewritten to the uploaded location.

Every submission gets a new directory under
`Pace-ICE:/storage/ice1/2/1/jjang318/tacnav/punch-eval/<timestamp-id>/`.
Later submissions do not alter earlier jobs' source. `submission.json` records
the branches, commits, dirty status, runtime versions and submitted command;
`SHA256SUMS` records the actual uploaded files and is checked before execution.
Both are copied into the evaluation output. Local submission receipts are in
`out/pace_submissions/`.

The existing PACE Python environment is reused; no workstation virtualenv is
copied and no environment is rebuilt. The script checks simulator package
versions against the workstation and verifies that imports select the uploaded
source. A PACE-only Python startup hook removes older editable source paths and
import finders from each job process; shared binary packages remain installed.
This also prevents deleted namespace modules from falling back to an older
checkout. The default interpreter is
`<remote-root>/tactile_controller/.venv/bin/python`; override it with
`--remote-python`. `--host`, `--remote-root` and `--curobo` select a different
SSH alias, remote base directory or local cuRobo checkout. Environment defaults
are `TACNAV_HOST`, `TACNAV_ROOT`, `TACNAV_PACE_PYTHON`, and (for the local
interpreter) `TACNAV_LOCAL_PYTHON`.

```bash
# Preview only: no network calls or file transfer.
scripts/pace/submit_eval.sh --dry-run --controllers footstep --rooms demo_room
# Upload and validate the runtime/Slurm request without queuing a job.
scripts/pace/submit_eval.sh --test-only --controllers vforce footstep --rooms room2
```

Defaults: `vforce`, global/PF/EIG, one A100, four CPUs, 16 GB RAM, one hour.
`--time`, `--cpus`, `--mem`, `--partition`, `--qos`, and `--gres` override Slurm
resources. Seeds run concurrently (one process per seed); controller/map groups
run sequentially in one evaluation job with logging. Allocate sufficient CPUs
and memory for the seed count. The script prints the job ID, remote log and
results paths. Under the snapshot, results are in
`tactile-navigation/out/evaluation/pace_<job-id>/`; the log is `slurm-<job-id>.log`.

**Video and Dropbox upload are enabled by default.** After the evaluation, a
separate GPU job calls the existing `punch_video.py` on each trial's saved
`states_*.npz` and replay JSON. It replays the recorded joint/base poses without
rerunning navigation or the locomotion policy. The MP4 shows the robot alongside
the belief/trajectory panel. Each MP4 and `video.log` live beside the trial's
metrics; `videos.json` reports completed, skipped and failed renders. Kinematic
trials and trials without saved robot poses have no 3D replay. A kinematic-only
suite does not queue a video job.

`--no-video` (before `--`) disables rendering; `--video-stride N` renders every
Nth recorded frame while keeping real-time playback. `--video-time` sets the
render job's wall time (default one hour). `--endpoint local` still renders video
but keeps results on PACE. The local receipt records the evaluation, video and
upload job IDs; render job output is `video-<video-job-id>.log`.

A separate CPU upload job waits for rendering (or evaluation with `--no-video`)
using Slurm `afterany`, so it also uploads diagnostic files after a failure or
timeout. It copies the results, MP4s, evaluation Slurm log and submission
provenance to `dropbox:TacNav/tactile-navigation/runs/pace_<evaluation-job-id>/`,
then runs `rclone check`. Transfer/verification failure makes the upload job fail;
the original PACE files remain available. The submission prints both job IDs,
the Dropbox destination and `upload-<upload-job-id>.log`; the local receipt also
records the upload job ID. No simulation is rerun by the upload job.

Use `--dropbox-path "TacNav/my-experiment"` to change the destination,
`--rclone /remote/path/rclone` to change the remote executable, or
`--endpoint local` to keep results only on PACE. The default executable is
`<remote-root>/bin/rclone`, using its existing `dropbox:` configuration.
`--test-only` validates all requested jobs without queuing them.

`punch_eval.sbatch` remains the remote worker and can still be submitted directly
from a configured checkout on PACE; direct `sbatch` does not queue the upload
job. The deployment repository's similarly named
`scripts/pace/submit_eval.sh` runs the separate ROS pipeline. Use the one in
**tactile-navigation** for this demo. All eight maps remain included in
`--rooms all`, including maps whose default planning bounds currently yield no
feasible route (see [`maps/README.md`](../maps/README.md)).

## Using a different implementation

```python
from execution.evaluation import RunRecorder

log = RunRecorder(
    {"controller": "my_controller", "planner": "my_planner", "seed": 0},
    goal_xy=[6., 1.], goal_radius_m=.4, frame="map")
log.sample(t, estimate=estimated_xyyaw, truth=reference_xyyaw)
log.probe(t, hand="left", contact=True, status="complete")
log.event(t, "recovery", target_xy=[1., 2.])
log.finish(t, outcome="done", completed=True, fallen=False)
log.write("trial.eval.json")
```

Supply `truth=None` when no external reference is available. All poses must use
the declared frame and yaw in radians. A replacement controller using the demo
adapter can provide `estimated_map_pose()` returning `[x, y, yaw]`; existing
kinematic/vforce controllers are adapted through their read-only `working_pose`
and frame-conversion methods.

## Execution modularity that remains

The controller already has a factory registry and `PunchAction` command interface;
`TactileIO` owns motion/probe primitives and observer hooks. `RouteFollowPolicy`
owns fixed-route progress and local recovery, and `SingleTouchSelector` chooses
probes. `plan_global` supplies the route.

The planner/executor boundary still carries `GlobalRoute` grid and uncertainty
trace internals, and the follower reads the selector's EIG cache. Replacing those
implementations still needs an execution adapter. The shared evaluation schema
and metric functions do not depend on those internals and need no changes.
