# Deployability audit — can this planner run on the real G1?

**Answer: no, not yet.**  A systematic inventory of every simulator
ground-truth read on the control path found **twelve blocker classes**.  This
document lists them so they can be closed, and states plainly which of my
earlier claims it corrects.

Rule used: a quantity may be read from the simulator only if it lands in a
log.  If it decides, gates, targets or bounds anything, it must be obtainable
on hardware.  Hardware (`DEPLOY_INTEGRATION_PLAN.md`) has **no odometry of any
kind**; the real sensors are the IMU, the joint encoders, and SensX per-hand
force (0-20 N, ±1-3 N, ~100 Hz).  No lidar, no wall ranging, no camera.

## What the honest-sensing work did and did not cover

`honest_sensing` closed the four leaks it was built for: observation noise,
the truth-driven locomotion attractor, the wall-distance sensor the *lateral
servo* used, and the unmodelled force channel.  Nineteen branches are gated
by it and behave correctly.

It did **not** cover the rest of the stack, and I previously described the
sim as honest without checking the rest.  That was wrong, and the following
list is the correction.

## The blockers

### 1. `_kiss_x` is a lidar the robot does not have — and it commands the press
The FSM solves base-to-wall distance from `root_link_pos_w`/`heading_w`
against wall slots placed at their *physical* (registration-shifted) poses.
That value is not telemetry: it **is** the press command
(`mujoco_backend.py:3110/3120`), the reach depth (`:3082`), the saturation
latch that ends the press (`:3155`), the valid-sweep test (`:3178`), and the
hand clamp in `wall_follow_map` (`:2862-2870`).  Worse, `_apply_masked_x`
(`commands.py:566/693`) **overwrites** the shield's x command from the live
kiss solve whenever `contact_mask > 0.5`.  No `honest_sensing` branch touches
any of it.

This is the largest single blocker: the deployed answer has to be "derive
the kiss plane from the believed pose and the map", which is what the deploy
plan already prescribes and what has not been done.

### 2. The tick path — the actual deployment target — is not honest at all
`tick_core.py` has **no `honest_sensing` awareness** (grep-verified) and runs
seven gating call sites on `_phys_standoff`: the `not_on_lane` refusal
(`:625`), the cross-track feedback and settle gate (`:659-670`), the
turn-radius skip (`:713`), the hand-to-wall scheduling (`:690`), the crab
step-in arrival (`:742/748`), and the press travel cap (`:807-813`).  Since
M2 is the path meant to reach hardware, this matters more than the blocking
twin.

### 3. Along-wall progress `s` is read from truth
`wall_parallel_touch` (`:2053`, `:2604`) takes `s` from the true pose, and
`s` drives the carrot, every station-due test, the skip decision, lane
arrival, and the stall watchdog.  Only `d_base` was made honest.  The tick
twin `_follow_gen` (`executor.py:1074-1077`) computes `arc(truth) −
arc(belief)` inline and never calls `_arc_shift`, so the guard that protects
the blocking twin does not exist there.

### 4. Twelve "hold where I stand" reads, each worth one estimator-error
`_set_goal` adds `(truth − estimate)` to whatever target it is handed.  So
`_set_goal(truth_pose()[0], ...)` commands `truth + (truth − est)` — a
commanded step of exactly one estimator error at the moment the code wants
the base to stand still.  Sites: `:1387`, `:1415`, `:1424`, `:1532`, `:1707`,
`:2085` (stand-to-settle), `:2245`, `:2255`, `:2295` (the press anchor),
`:2877`, `:2956`, `:2977`, `:3078`, `:3106`, `:3277`.  All must become
`working_pose()`.

### 5. Three displacement anchors double-count the bias
`goto_map` (`:1458`), `wall_follow_map` (`:2844`) and `_goto_gen`
(`executor.py:1049`) anchor a relative displacement on truth, and then
`_set_goal` adds the bias again.

### 6. Eight terminations decide on true pose or true yaw
`goto_map` arrival (`:1477`), `side_contact` arrival (`:1586`),
`_turn_in_place` (`:1908`), `_turn_once` (`:2963`), `turn_to_map`
verification (`:2929`), `_stabilize_world` (`:2985`), `heading_error_rad`
(`:2950`), `wall_follow_map` arrival (`:2882`).  Heading is available from
the IMU, so these are fixable — but as written they read the simulator.

### 7. Two channels that do not exist on hardware at all
`root_link_lin_vel_w` gates four decisions — the safety-stop still test
(`:1439`), the feeler's "body has stopped" break (`:1725`), **the crab settle
gate before every press** (`:2283`), and `StabilityGate.max_speed_m_s`
(`:2986`).  And `_feet()` (`:1260`) reads a foot-contact sensor the plan says
hardware does not expose, yet `StabilityGate` refuses to pass below two feet
in contact.

### 8. The force model is bypassed on the press latch
`_probe_force_n` (`:1230-1234`) returns the raw contact force, and the press
latch and the 14 N target test read it (`:3133-3194`).  `:2600-2601` mixes
the shield's raw 200 Hz peak into the tap trigger and the 25 N abort.  So the
SensX clip that the whole §26 analysis is about does not actually apply where
the press decides.

### 9. `face_dot` vetoes the press using truth yaw and the true wall normal
`press_snapshot` (`:3032`) → `rec_t["face_dot"]` (`:2322`) → the
`base_not_facing_wall` refusal (`:2377`).

### 10. `corner_gate` aborts the mission on a truth-derived absolute position
`contact_xy_map` comes from the true EE pose; `executor.py:964-977` compares
it to the map corner and can abort the route.

### 11. An `# eval only` label that is not eval only
`executor.py:383-385` computes `_phys_standoff`, comments it "eval only", and
then passes it as `standoff_m` into `wall_parallel_touch`, where it becomes
the commanded lane.

### 12. My own mitigation is dead config
`wp_press_cap_margin_honest_m = 0.10` was added as the tightened travel cap
for exactly the 20 N clip problem, and is **never read** — both press paths
still use `wp_press_cap_margin_m = 0.16`.

## Gaps in the sensor models (not blockers, but not honest either)

* **Observation and action latency: zero.**  No obs buffer, no actuation lag,
  no transport delay anywhere.
* **Contact detection at 200 Hz inside the physics step** — a rate the deploy
  plan says hardware does not have.
* **Planner-side proprioception is noise-free**: the noise applies only to the
  policy's actor group, so `measured_ee_pose_w()` and the brake's
  `joint_pos` read are exact.
* **`_rebase` is perfect short-baseline odometry**: it composes press-time and
  current truth poses exactly, across a turn and a walk, with no drift.
* No force bias, drift, hysteresis or dropout; σ = 1 N against a stated
  ±1-3 N.

## What is genuinely clean

* `scene_room2.py` and `landmark_ate.py` contain **zero** ground-truth reads.
* Nothing fed to the particle filter or the ATE is an absolute truth pose —
  every input is map data, pelvis-frame FK, a commanded-axis assumption, or a
  relative pose composition.  The belief itself is honestly sourced; its
  inputs are simply cleaner than hardware's would be.
* `estimator_bias()` reaches **every** base command, including the tick path
  (`_set_goal` is the only writer of `pose_target_w`).

## Order to fix

1. `_kiss_x` → believed pose + map (blocker 1) — unblocks the press command,
   the biggest single dependency.
2. `tick_core.py` honest branch (blocker 2) — it is the deployment path.
3. `s` from `working_pose()` and `_arc_shift` in the generator twin (3).
4. The mechanical sweep: `working_pose()` at the fifteen hold/anchor sites
   and the eight terminations (4, 5, 6).
5. Replace the two impossible channels (7) — base speed from the estimator,
   foot contact from the gait clock or drop the gate.
6. Route the press latch through the force model, and wire up the dead cap
   (8, 12).
7. `face_dot` from belief, `corner_gate` from believed contact position
   (9, 10), and fix the mislabelled lane (11).
8. Then add latency, and only then claim deployability.

---

## Status after the fix sweep (2026-08-18, commits 86df5e0..4273d51)

| blocker | status |
|---|---|
| 1 `_kiss_x` ranging | **fixed** — `believed_kiss_x()` (estimate + map, controller's own clamp); press paths banned from the raw solve by test |
| 2 tick_core honest branch | **handed to the M2 session** (their file, by agreement), with the interfaces to consume; also 3 more tick-side findings |
| 3 `s` from truth | **fixed** — `working_pose()` in both loop reads |
| 4 hold-where-I-stand ×12 | **fixed** — all `working_pose()` |
| 5 displacement anchors ×3 | **fixed** (backend two; `_goto_gen` is tick-side) |
| 6 truth terminations ×8 | **fixed** — arrival/turn/settle judge the estimate |
| 7 impossible channels | **fixed** — stillness from a CALIBRATED leg-encoder proxy (mean, 0.73; calibration table in source); feet nominal under honest sensing |
| 8 force model bypassed | **partly fixed 2026-08-18, closed 2026-08-19** — the substep peak and tap trigger were routed then; `_probe_force_n` was MISSED and kept returning raw MuJoCo contact force into the press latch and the press-mean sum (see below) |
| 9 `face_dot` from truth | **fixed** — believed heading × map normal; truth demoted to eval |
| 10 corner gate on truth position | **fixed** — `_believed_contact_map()` |
| 11 mislabelled eval standoff | **fixed** — `_est_standoff` |
| 12 dead cap config | **fixed** — wired, tested tighter-than-legacy |

Two regressions surfaced by the sweep's own verification runs, both fixed:

* **misassociation**: a feel aimed at corner face B touched the WEST wall
  (13.1 N, believed face_dot 0.21) and was hard-associated to face B by the
  ATE — belief jumped 0.20 → 0.34.  Felt contacts failing the squareness
  gate are now `felt_off_face`: the PF may eat them (per-particle ray match
  is association-free), the ATE and the wall-confirmed latch may not.
* **stillness proxy miscalibrated**: 0.30 × max leg speed put STANDING at a
  0.108 m/s equivalent — above every settle threshold — so no gate opened
  and the goal phase burned 18 aborts.  Recalibrated from a measured
  stand/walk_min/walk table to 0.73 × mean (3.5× separation).

**Post-sweep verification (`room2_fwd_s0_b3`): arrived 0.126 m, 8/8 contacts
(7 in band), marker confirmed 20.1 N, corner gate on evidence, every
face_dot 1.0.**  Belief error median 0.185 (up from 0.09 — the observation
composition now carries honest odometry drift instead of a perfect rebase;
that is the sim getting more honest, not the planner getting worse).

## Second pass, 2026-08-19 — what the first pass missed

The table above was written from the fix commits rather than from the code,
and three of its rows were more optimistic than the source.

### Blocker 8 was half fixed

`_probe_force_n` carried a docstring saying it went through the SensX model
and did not call it.  It fed `contacted = axis_n >= 2 N` -- the press latch --
and the press-mean sum with exact, unbounded MuJoCo contact force at the
physics rate.  Consequence beyond tidiness: the air-vs-graze mean gate was
calibrated on the SENSED distribution (air ~0.8 N, because zero-clipped
sigma=1 N noise has mean 0.8) and was being applied to a raw signal whose air
mean is 0.0.  There was no test.  Now routed, with the true value kept under
`force_axis_true_n` for evaluation, and a test that asserts the decision path
reads the model.

### Blocker 13 (new): the wrist was aimed with the simulator's answer

`_face_quat` yawed the hand toward the pressed face using
`clamp(face_dir - heading_w, +-0.12)`.  `heading_w` is the TRUE base heading.
Besides being a truth read on the control path, it was self-fulfilling: the
hand was aimed by the very quantity the felt normal exists to estimate, so
the sim's normal channel was better than hardware's can be.  Now
`working_pose()` under honest sensing.

### Blocker 14 (new): the 100 Hz guard was running at 50 Hz

`_sensed_force_n` timestamped samples with `self._t`, which is not advanced
until after the physics step, so every substep of a control tick shared one
timestamp and the hold returned a single sample per tick.  The guard was
therefore blind to any spike landing in the second half of a tick -- half the
rate the deploy plan specifies.  It now uses the substep clock it is handed.

### Latency: partly modelled now

`plan_force_latency_s = 0.010`.  The deploy plan runs sensor drivers and
controllers onboard the Jetson and the planner offboard on the workstation
over a measured ~5 ms link, so a force the PLANNER acts on is a round trip
old, while the 100 Hz guard node is onboard and reads at zero lag.  One tick
of action latency was already modelled implicitly (a decision made after
`_step` applies on the next one).  Observation and action latency on the POSE
channels are still zero.

### A note on what "honest" bought and cost

Honest sensing is now good enough that it exposes planner defects instead of
hiding them.  The heading-for-position trade documented in HW_ROBUSTNESS 29
was only findable because the belief and the truth are genuinely separate
quantities here.  It also means every gate the route runs -- `face_dot`, the
corner information gates, lane acquire -- is computed FROM the belief, so a
belief that is confidently wrong is invisible on board.  That is a real
deployment property, not a sim artefact, and it argues for at least one
consistency check that does not come from the belief.

## Still open before "deployable" can be claimed

1. tick-side blockers (with the M2 session);
2. latency on the POSE channels — observation and action are still
   instantaneous; the force channel now carries the planner's transport lag
   and the guard runs at its true rate;
3. planner-side proprioception is noise-free (EE pose, brake joint reads);
4. force bias/drift/dropout beyond σ=1 N; per-hand polarity.
