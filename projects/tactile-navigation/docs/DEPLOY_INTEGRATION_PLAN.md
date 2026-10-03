# TacNav → locomanip_deploy Integration Plan

Date: 2026-08-12 · Updated: 2026-08-14 (controller decision) · Status: IN EXECUTION (M0/M1/M2-s1-s2 done; not yet reviewed with the deploy-repo owner)
Scope: put the TacNav planner (PF belief + route planner + touch scheduler + press primitives, currently a MuJoCo research loop in `tacnav_analysis`) behind a ROS 2 node inside the `locomanip_deploy` workspace, sim-first, then on the real G1.

Repos:
- Planner: `~/Desktop/tacnav_analysis` (branch `hw-robustness`, planner core under `src/`)
- Deployment: `~/Desktop/locomanip_deploy` (ROS 2 Jazzy + pixi, branch `jjang318/docker` = default)

> **Four controllers — never conflate (owner decisions 2026-08-12 / 2026-08-14):**
> 1. **GR00T Walk/Balance** (`lower_body_controller` ONNX) + **Pink IK** arms — the deploy repo's stack. Was the MVP target; **now the fallback** (weights still not on this machine).
> 2. **TacNav's MPC-RL** (`hand_force_v2`, Mjlab wall-following) — exists only in tacnav's own MuJoCo research loop. Its measured limits do **not** transfer to other controllers.
> 3. **Deploy repo's tracking-policy / MPC** (`g1_mpc_tracking`, clip replay, 29-joint) — the owner's separate line of work. Out of MVP scope, but its `tracking_policy_node.py` + `export_tracking_onnx.py` are the deployment template for #4.
> 4. **Decoupled vel-force-MPC** (`tactile_controller` branch `decoupled-policy`, `vforce_model_52000.pt`, two-actor 29-joint @50 Hz) — **M3 priority candidate (user-approved 2026-08-14)** after passing the maze eval: 0 falls, 2.5 cm anchor hold under 8–22 N press, R-hand 22/22; force under-delivery ~40 % at ≥15 N handled by an outer measured-force servo on f_cmd. Evidence: `docs/DECOUPLED_CONTROLLER_EVAL.md`. **Sim2real: CONFIRMED on the real G1** (user, 2026-08-17: “vforce那个上过真机了” — the same `vforce_model_52000.pt`). Deployment workspace identified: `locomanip_deploy` branch `jjang318/wall_mpc` — tip commit adds `tactile_controller` as a submodule (`git@github.com:ZhenyuWu323/tactile_controller.git`); its `docs/mpc_tracking.md` + `tracking_policy_node.py` + `g1_mpc_tracking.yaml` define the runner pattern: 50 Hz python node → `whole_body_controller` position forwarding @500 Hz, PDs from `src/hardware/unitree_description/config/<pids>.yaml` (e.g. `g1_beyond_mimic.yaml`) matched to the mjlab training stiffness/damping the action scale was derived from. Zhenyu-repo branches `jjang318/decoupled`/`jjang318/anchor` are earlier dev lines with no deploy tooling. Remaining narrowed ask: which pids yaml + export format the vforce HW run used (if none is pushed, we export ourselves and derive PDs from the decoupled train cfg).

---

## 1. What the deploy pipeline is (review summary)

ROS 2 Jazzy workspace, `ros2_control` at 500 Hz, everything run under `pixi run` (`ROS_DOMAIN_ID=42`). Sim and hardware share a byte-identical joint/IMU interface set: `simulation:=true` swaps the hardware plugin (`g1_hardware_interface/G1HardwareInterface`, low-level DDS `rt/lowstate`/`rt/lowcmd`) for `mujoco_ros2_control/MujocoSystemInterface` via one xacro conditional.

**Base locomotion** — `lower_body_controller` (`velocity_controller/VelocityController`, subclass of the generic `onnx_controller`): GR00T-WholeBodyControl Walk/Balance ONNX at 50 Hz inference, owns 15 joints (12 legs + 3 waist). Commands:
- `/lower_body_controller/cmd_vel` (`geometry_msgs/Twist`) — only `linear.x`, `linear.y`, `angular.z` are read. **No clamp inside the controller** — the publisher must limit itself.
- `/lower_body_controller/cmd_height` (`Float64`, teleop range 0.5–0.9, default 0.75) and `cmd_rpy` (`Vector3`).
- Auto policy switch: `‖(vx,vy,wz)‖ ≥ 0.05` → walking, below → standing; `‖v‖ < 0.01` freezes gait (standing dead-zone).

**Upper body** — `g1_upper_pink_controller` (Pink IK, Python, 200 Hz), owns the 14 arm joints (waist belongs to the walk policy):
- Targets: `/g1_upper_pink_controller/{left,right}_hand` (`PoseStamped`), interpreted **directly as T_pelvis→{left,right}_rubber_hand** — pelvis frame, no TF lookup, palm frame not the sensor TCP.
- Output: `Float64MultiArray` → `/upper_body_controller/commands` (position) + `/upper_body_effort_controller/commands` (feedforward torque on top of hardware joint PD).
- Force feedforward: `WrenchStamped` on `/g1_upper_pink_controller/external_wrench/g1robot_vitac_{left,right}` → `τ += Jᵀ·W`. **Off by default** (`feedforward.external_wrench_enabled: false`); enable via `/g1_upper_pink_controller/set_feedforward` (`deploy_transport/srv/IKFeedforward`). The same wrench magnitude (‖W‖² > 1) is what *arms* the tactile contact-depth regulation.
- Tactile admittance (off by default): per-hand contact-area regulation (push in / back off on `[area_fraction, centroid]` from `/sensor{1,2}/contact`) + wrist-compliance PD switch via `/g1_hardware_interface/upper_body_pd_gains`.
- Useful read-backs: `/tf_server_{left,right}/pose_measured` (measured palm pose in pelvis, 250 Hz), `~/ik_metrics`, `~/move_to_init`, jump clamp (`jump_clamping.*`, off by default).

**Whole-body alternative** — `g1_mpc_tracking.launch.py`: a 29-joint motion-imitation policy (mjlab `Unitree-G1-MPC-Tracking`, 1002-d obs, `scripts/tracking_policy_node.py` streams an `.npz` clip at 50 Hz into a `forward_command_controller`). Trained for walk-to-wall-and-press; hardware-verified ("Hardware working" commit). **No velocity-command interface — it replays clips.** `deploy_transport/msg/MotionReference` exists for streaming references but has zero users today. This stack and the decoupled stack are mutually exclusive controller sets.

**State available on hardware** — IMU (orientation, gyro, accel; `/imu_broadcaster/imu`, frame `pelvis`, 500 Hz) + 29× joint (q, dq, τ_est; `/joint_states`, 500 Hz) + commanded torques (`/joint_states_commanded`). **No absolute pose on hardware** — no world-frame position and no foot-contact interface. **Closed 2026-08-19** — the deploy repo closes it with `leg_odom` (stance-foot FK on `/joint_states` + `/imu_broadcaster/imu` and a latched URDF, Pinocchio, sole-frame anchored) → `sportmode_odom_bridge(source: "topic")` → `/tacnav/odom`, a drifting RELATIVE estimate, which is what the PF eats. Unitree's own `rt/odommodestate` sits behind the bridge's `"dds"` branch — also an onboard estimator, not truth — and is not enabled on either hardware launch. `/simulator/floating_base_state` (`nav_msgs/Odometry`, ground truth) is sim-only and is now forbidden on the hardware path: readiness fails closed if any publisher of it, or of `/tacnav/eval/*`, appears in the graph.

**Force sensing on hardware** — no wrist F/T, no torque-residual observer. What exists:

| Signal | Topic | Range | Rate | Trust |
|---|---|---|---|---|
| SensX per-hand (calibrated) | `/sensor{1,2}/sensor{1,2}/force` (`Float32`) | 0–20 N (13.8 N peaks observed) | ~100 Hz | ±1–3 N bench accuracy — **the planner's force signal** |
| SensX contact stats | `/sensor{1,2}/contact` (`[area_frac, centroid r,c]`) | — | 100 Hz | good binary/positional cue |
| force_observer (training parity) | `/force_observer/measured_force_{right,left}` | **hard-capped [0, 5] N** | 50 Hz | never use for control/safety |
| ViTac per-hand | `/vitac_sensor{1,2}/force` | 0–5 N heuristic | serial | binary contact switch only |
| Policy force head | `/tactile_policy_server/force_pred*` | — | ~20 Hz | image feature, r ≈ 0.08 — not a force estimate |
| Torque residual (buildable) | `/joint_states.effort` − `/joint_states_commanded.effort` | — | 500 Hz | would need a J⁻ᵀ projection node (does not exist) |

⚠️ Known polarity trap (documented pitfall #7): sensor1↔right/left comments disagree across configs; the mapping is set by USB cabling and must be verified by squeezing the sensor while echoing the topic.

**Superseded 2026-08-19 — the shipped hardware path takes NO force input.** The row above marked "the planner's force signal" is the 08-12 plan, not what was built: contact is confirmed from the TactX *image* (`sensx_pub` → `sensx_contact_projector` → `ContactObs.v1`), the touch primitive is touch-and-retract rather than force-regulated, and the palm wrench exists only in sim on `/tacnav/eval/*`. The table stands as the sensor inventory; it no longer describes the control path.

**Sim specifics** — MJCF `src/hardware/unitree_description/mjcf/g1.xml` (2000 Hz physics). No touch/force sensors in the MJCF today ("no tactile model in sim"), but the plugin already supports `mujoco_type: "fts"` → a wrist F/T sensor in sim is a **config-only** change. Obstacle injection precedent: `scripts/make_wall_mjcf.py` (splices a static box before `</worldbody>`, wired through the `mjcf:=` launch arg). `/mujoco_node/reset_world` for episodic resets. **Elastic band defaults ON** (kp=100 tether on torso) — must be off for any locomotion.

**Ops** — `pixi run deploy` rsyncs to the Jetson (controllers + sensor drivers run onboard; policy/planner nodes run offboard on the workstation); PTP clock sync; measured link ~5 ms / 20 Mbps. As shipped, cross-machine DDS discovery is **disabled** (`ROS_STATIC_PEERS=127.0.0.1`; the robot-hostname peer line in `pixi.toml` is commented out).

**Prior wall work in that lab** (branch `jjang318/wall`, `tasks/lessons.md`, 2026-08): real-G1 kinesthetic wall probing achieved ±2–3 cm wall reconstruction; **IMU yaw beat stance-foot kinematic yaw** (feet pivot during lateral side-steps); repo docs drift — verify every service/msg name with `ros2 interface show` (Jazzy uses `controller_manager_msgs/srv/SwitchController` with `activate_controllers`/`strictness`, not the older names in README). The deploy repo already vendors our `tactile_controller` as a submodule at `src_ext/tactile_controller` — the vendoring path is established.

---

## 2. What the planner brings, and the seam on our side

Planner stack on `hw-robustness` (all paths relative to `tacnav_analysis`):

- **Pure-NumPy, lift verbatim** (zero MuJoCo imports, verified): `src/global_nav/` (ParticleBelief/ManagedBelief, observations, floorplan, models, affordance), `src/execution/{interface,wall_lane,stations}.py`, `src/local_tactile/fsm.py`, `src/demo_map/{route,geometry,mapv3,scan_map,scene_demo}.py`, `src/lane_graph/{maps_v3,scene3,graph3}.py`, `src/planning/geometric.py`.
- **Belief I/O**: `predict(delta_body, delta_yaw)` (body-frame odometry increments) + `update(contacts: ContactObs[], frees: SweptFreeObs[])` (body-frame contact points + swept-free rays). Outputs pose estimate/covariance/hypotheses/wrong-turn prob. It never sees an absolute pose — exactly what a drifting hardware odometry needs.
- **The one structural refactor**: today's executors are *blocking primitives* — `goto_map()` / `press()` / `wall_parallel_touch()` own hundreds of 50 Hz ticks inside `src/execution/mujoco_backend.py`. A ROS node must own the clock, so each `while True:` loop becomes a resumable step function: `PlannerCore.tick(obs) -> cmd`. The transition predicates already read only pose/standoff/forces/time, so the inversion is mechanical (big, but mechanical).
- **Command layer**: port `Se2PoseVelocityCommand._update_command` (~35 lines of NumPy: decel-limited carrot → body-frame `(vx, vy, wz)`, slew-limited) into PlannerCore so the emitted command is a plain **body-frame twist** — which is precisely `/lower_body_controller/cmd_vel`. Velocity clamps become a config owned by the planner (the deployed GR00T box must be characterized in M0; our sim policy's box was vx∈[−0.5,1.0], vy∈[−0.75,0.75], wz∈[−1,1]).
- **Press primitive to port**: `crab_press` station style (face the wall → settle → rate-limited EE creep along +x to a force target → retract), including its safety stack: travel cap = kiss + margin (this is what kept sim presses at ~13–14 N instead of 400+ N), 25 N hard brake, 12 N parked-hand guard, `_safety_stop` back-off along the measured reaction force. `turn_press` and taps-while-walking (`lateral`) stay sim-only for now — pressing while walking is out-of-distribution for the deployed walk policy, and our own FSM already treats contact-at-standstill as the norm.
- **Keep the truth firewall**: `src/execution/interface.py::assert_no_truth_leak` + `tests/global_nav/test_truth_firewall.py` move into the deployment package. On hardware "truth" becomes mocap/lidar/sim-GT — eval-only, never a belief input.

---

## 3. Gap analysis (planner needs vs deploy provides)

| Planner need | Deploy today | Gap / decision |
|---|---|---|
| Body-frame odometry increments | ❌ nothing consumed on hw; GT in sim | **Check before building** (owner decision 2026-08-12): Unitree's own estimator may or may not keep publishing (SportModeState-like topics) once the built-in motion service is off and our low-level stack drives `rt/lowcmd` — sim keeps it, hardware may not. M0 hardware step (see `docs/M0_HARDWARE_CHECKLIST.md`): with GR00T running, `ros2 topic list \| grep -Ei 'odom\|sport\|loco\|state'` → echo/hz the candidates while walking. **Readable & tracking** → light `unitree_odom_bridge` → `/tacnav/odom`, measure crab-step lateral drift, skip leg_odom. **Absent/frozen** → build `leg_odom` (stance-foot FK + IMU yaw; IMU yaw authoritative per the wall-branch lesson). Either way it is a *relative* estimate, not absolute map pose — PF + wall-touch correction stays load-bearing. Retune `MotionConfig` σ up, never down. |
| Base→wall physical range (acquire gate, cross-track, crab arrival) | ❌ no range sensor in the pipeline | MVP: derive from **believed pose + map**; re-base the acquire gate on belief σ + force events. Lidar (MID-360 is mounted but unused) is a later, *eval-only* option. Biggest ported-behavior change — flagged as top risk. |
| Per-hand contact force, ~5–25 N band | SensX 0–20 N calibrated (if hand-mounted); ViTac binary | Subscribe **uncapped SensX** directly; press target ~8–12 N; brake ~15 N (sensor clips at 20). Never use the 5 N-capped force_observer. Optional upgrade: torque-residual EE wrench node (500 Hz, directional). |
| Contact point/normal for `ContactObs` | hand pose from `/tf_server_*/pose_measured` (pelvis, 250 Hz) | De-tilt pelvis→yaw-aligned base frame using IMU roll/pitch; normal from commanded press axis (same simplification as sim, which measured only 8° error). |
| EE press motion | Pink IK PoseStamped + wrench feedforward + wrist compliance + optional contact-depth regulation | Port `ProbeShield.set_normal/set_lateral` rate-limiting onto PoseStamped targets. Enable `ff_external_force` + publish desired-normal-force wrench. |
| 200 Hz force brake | ❌ (sim substep hook) | Approximate: 100 Hz SensX guard loop + conservative creep rate + wrist compliance gains + Pink jump clamp. Accept higher latency; compensate with slower `rate_m_s`. |
| Walking policy tolerant of our command style (0.1–0.25 m/s creep, crab lateral, precise stops) | Unknown — GR00T box/behavior uncharacterized; auto standing↔walking switch at 0.05 may chatter | M0 characterization; planner-side hysteresis around the switch threshold. |
| Scene in sim (scanned room) | one hardcoded wall (`make_wall_mjcf.py`) | Generalize to `make_scene_mjcf.py` reading our scan-map JSON (`docs/demo_maps/*.json`) → N static boxes. |
| Wrist force in sim | ❌ no touch sensors in MJCF | Add `mujoco_type: "fts"` wrist sensor (config-only) + a bridge node republishing to the **hardware topic names**, so the planner node is byte-identical sim/hw. |
| Offboard planner ↔ onboard controllers | DDS discovery localhost-only as shipped | Uncomment the static-peer line (or run the planner on the Jetson — decide in M0 with the owner). |

---

## 4. Target architecture

```
                          ┌────────────────────────────────────────────┐
                          │  tacnav_planner_node  (50 Hz tick, Python) │
                          │  PlannerCore = belief + route + scheduler  │
                          │  + press/goto/crab FSMs (ported, tick-ized)│
                          └────┬───────────┬───────────┬───────────────┘
        subs                   │ pubs      │ pubs      │ pubs
  /tacnav/odom  (leg_odom) ───►│ /lower_body_controller/cmd_vel   (Twist, clamped)
  /imu_broadcaster/imu     ───►│ /g1_upper_pink_controller/{left,right}_hand (PoseStamped, pelvis)
  /joint_states            ───►│ /g1_upper_pink_controller/external_wrench/g1robot_vitac_* (WrenchStamped)
  /tf_server_*/pose_measured──►│ /tacnav/belief  (markers + snapshot for Foxglove)
  /sensor{1,2}/sensor{1,2}/force ─► services: ~/start ~/stop ~/enable; calls switch_controller,
  /sensor{1,2}/contact     ───►            set_feedforward, toggle_contact_regulation
  /vitac_sensor{1,2}/force ───►
                          ┌────────────────────────────────────────────┐
                          │  tacnav_guard_node (100 Hz, independent)   │
                          │  unexpected-contact + over-force latch →   │
                          │  zero twist, retract, idle_controller      │
                          └────────────────────────────────────────────┘
  sim only: make_scene_mjcf.py → mjcf:=…  ·  sim_touch_bridge (fts → hardware force topic names)
            /simulator/floating_base_state → eval + leg_odom validation only (truth firewall)
```

Frames: `map` (scan-map convention: origin = spawn, +y = spawn heading) · `odom` (leg-odom, drifting) · `pelvis`. The PF owns odom→map. Commands: twist in yaw-aligned base frame; EE targets converted to pelvis frame (re-apply IMU roll/pitch).

Packaging (mirrors the repo's own convention):
- **`tacnav_core`** — new pure-Python package extracted from `tacnav_analysis/src/` (no ROS, no MuJoCo), vendored into the deploy repo as a `src_ext/` submodule (same as `tactile_controller`). Fallback if a separate repo is unwanted: copy into the ROS package.
- **`src/planner/tacnav_planner/`** (ament_python) — the two nodes + launch `g1_tacnav.launch.py` + config `g1_tacnav.yaml` + pixi task `launch_tacnav`.

---

## 5. Workstreams

### W-A. `tacnav_analysis`: tick-ification + core extraction (largest single chunk, L)
1. Define `Obs`/`Cmd` dataclasses (fields per §4; `obs` = t, pose_odom, twist_body, base_z, feet, ee_pose_b, hand forces incl. per-tick peak; `cmd` = mode, twist or goal, ee targets + rate limits, contact-enable flags).
2. Invert the blocking primitives in `src/execution/mujoco_backend.py` (`goto_map:1101`, `_turn_in_place:1375`, `_crab_step:1320`, `_stabilize_world:2267`, `press:2340`, `wall_parallel_touch:1414`) into resumable step FSMs; `MujocoBackend` becomes one consumer of `PlannerCore.tick()`.
3. Replace the seven `truth_pose()` reads in `src/demo_map/executor.py` with estimator pose / obs fields.
4. **Regression gate**: re-run the crab-press demo route through the tick API in our own MuJoCo loop — same stations touched, same acceptance stats as `demo_room_run.py` today. This proves the refactor before any ROS code exists.
5. Extract `tacnav_core` package (the verbatim-liftable modules + the new tick FSMs + ported Se2 attractor).

### W-B. `locomanip_deploy`: sim scaffolding (M)
1. `make_scene_mjcf.py` — scan-map JSON → static boxes (generalize `make_wall_mjcf.py`); launch arg `scene:=`.
2. Sim wrist force: `fts` sensor entries (sim-guarded in `ros2_control.xacro`) + `sim_touch_bridge` node republishing to `/sensor{1,2}/sensor{1,2}/force`-compatible names.
3. `/tacnav/odom` contract (`nav_msgs/Odometry`) + sim source (relay of `/simulator/floating_base_state`, optional noise injection). The **hardware** source is decided by the M0 hardware check: `unitree_odom_bridge` if Unitree's estimator survives low-level mode, else `leg_odom` (stance-foot FK + IMU yaw). Whichever wins runs in sim too where possible and is validated against sim GT.
4. Launch/config/pixi plumbing; elastic band off; controller activation sequence scripted (Jazzy `controller_manager_msgs` form).

### W-C. `locomanip_deploy`: planner + guard nodes (M)
1. `tacnav_planner_node` wrapping `PlannerCore` (50 Hz timer; topic I/O per §4; belief snapshot → Foxglove markers; start/stop/enable services; explicit `switch_policy` hysteresis around the 0.05 threshold).
2. `tacnav_guard_node`: independent 100 Hz loop on uncapped SensX + ViTac binary; latches abort → zero twist once, retract via `~/move_to_init`, worst case `idle_controller` switch. Mirrors sim guard semantics (12 N parked-hand, level-triggered).
3. Press executor details: PoseStamped creep at planner-configured `rate_m_s`, travel cap from believed kiss + `press_cap_margin`, force target from SensX, desired-force wrench published for feedforward + regulation arming.

### W-D. Hardware bring-up (with the deploy-repo owner)
Sensor mounting decision, USB polarity verification, network peer config, Jetson-vs-offboard placement, PTP, deadman via Unitree remote.

---

## 6. Milestones and gates

| M | Content | Gate (measurable) |
|---|---|---|
| **M0** | Build + run deploy sim; drive `cmd_vel` and Pink targets from CLI; characterize walk policy (min stable speed, lateral tracking, stop precision, switch chatter) and a scripted wall press with the fts sensor. **Plus the hardware odom check** (`docs/M0_HARDWARE_CHECKLIST.md`) deciding bridge-vs-leg_odom. Known blocker: `export/` is gitignored — **GR00T ONNX weights must come from the deploy-repo owner/robot** before the walk-policy half can run in sim. | Characterization doc with measured numbers (the deploy-stack analogue of our "four uncommandable policy limits") + filled-in hardware checklist |
| **M1** | W-B complete | Our scanned room renders in deploy sim; scripted press shows a plausible force profile on the hardware-named topic; leg_odom vs GT drift quantified (m per m travelled, yaw) |
| **M2** | W-A complete | Crab-press demo route reproduced through `PlannerCore.tick()` in tacnav's own sim — no acceptance regression |
| **M3** | W-C in deploy **sim**, end-to-end scan-room route, **decoupled controller first** (revised 2026-08-14): export vforce two-actor ONNX → `decoupled_policy_node` (mirror of `tracking_policy_node.py`: obs assembly incl. gait clock + FK tip pose, 29 joint targets → whole-body forward controller, training-matched PDs) → planner node emits twist + tip-frame pose + mask + f_cmd (planner-side rate limiter + outer force servo). GR00T+Pink path stays as fallback if the decoupled line stalls. | Goal reached; all presses in 8–12 N band; zero guard violations; belief σ bounded; N≥5 seeds |
| **M4** | Hardware **shadow mode** (planner estimates only, operator teleops route), then standstill press-only at a hand-placed station | Belief consistent with taped ground truth ≤ agreed bound; ≥10 presses in band, zero over-limit, abort drill passes |
| **M5** | Hardware closed loop, staged: single goto+press → multi-station crab follow → full route with PF corrections | Per-stage acceptance, deadman always live |
| M6 (later) | Options: lidar eval-truth; torque-residual wrench; whole-body tracking-policy press (`MotionReference` streaming — we would be its first user) | — |

Ordering note: M0/M1 (deploy repo) and M2 (tacnav repo) are independent and can run in parallel; M3 needs both.

**M3 progress (2026-08-17):**
- **M2 CLOSED** — all four tick-ification stages pass (docs/M2_TICK_DESIGN.md).
- **M3(a) DONE — vforce two-actor ONNX export + validation.**
  Script: `tacnav_wt/dp_eval/scripts/rl/test/export_decoupled_onnx.py`. Artifacts in
  `tacnav_wt/dp_eval/out/onnx_export/`: `vforce_model_52000_lower_body.onnx` (obs[478]→15),
  `vforce_model_52000_upper_body.onnx` (obs[519]→14), `vforce_model_52000_specs.json`
  (full deploy contract: per-term obs layout w/ history-5 flattening, action order
  lower→upper = 29, per-joint scales, default-pos offsets, joint name order,
  `target = raw*scale + offset`, dt 0.02 / decimation 4). Validated vs the torch policy
  over 250 live env steps: max|Δaction| = 1.5e-05 (tol 1e-4) — PASS.
- Training PDs for the deploy pids yaml (from `g1_constants.py`, kp = armature*(2π*10Hz)^2,
  kd = 2*2.0*armature*ω): 4010 wrists 16.78/1.07 · 5020 shoulders/elbows 14.25/0.91 ·
  7520_14 40.18/2.56 · 7520_22 99.10/6.31 · ankle 28.50/1.81 · waist 28.50/1.81.
  M3(b) generates the full 29-joint pids yaml from the env's resolved actuator arrays.
- **M3(b) REVISED (2026-08-18) — 学长's deployed joint stack found on `origin/foot_trajectory`;
  nothing needs inventing.** `tasks/todo.md` there is the authoritative design record. Facts
  (all committed, several verified here by measurement):
  - Runner is the C++ `onnx_controller` (500 Hz RT / 50 Hz inference) with three
    backwards-compatible extensions (per-key obs history, `action_offset_list`,
    `~/cmd_vector` Float64MultiArray) — NOT a python tracking node. Zero C++ changes needed
    for new policies of this family.
  - ONE merged/adapted ONNX per policy (each actor observes the full 29-d last_action, so two
    separate controllers cannot reproduce it). `scripts/export_joint_decoupled_onnx.py` +
    `scripts/adapt_velocity_onnx.py` build it; deployed wire contract obs[518] =
    shared465 | velocity_command[sin,cos,vx,vy,wz,0,0] | upper_command[46]; 53-d cmd_vector.
  - Gait clock lives in `ee_command_manager` (`lower_command_mode: velocity`,
    `velocity_phase_period 0.6`, `velocity_stand_threshold 0.1`); tip FK via pinocchio
    (wrist_yaw_link + (0.1325, ±0.003, 0)); HOME-keyframe zero-points; tips at HOME =
    (0.3323, ±0.1516, 0.0952) — matches our vforce specs exactly.
  - PDs doctrine (his words): `pids:=g1_beyond_mimic` REQUIRED — those gains ARE the mjlab
    stiffness/damping the action scales (0.25·effort/kp) derive from.
  - Verification harness exists: `scripts/check_joint_stack.py` (17 sim2sim checks, EE
    tracking 2.0 cm) + `check_decoupled_wbc_regression.py`.
  - **Measured (2026-08-18): the committed `export/velocity_policy.onnx` is the same
    architecture (478/519→29) but NOT vforce_52000's weights (0/9 matrices equal)** — the
    Aug-9 HW deployment used a sibling checkpoint with the 8-d force channel BAKED TO ZERO
    (adapter does this by design; 学长's note: "when a real force command lands later,
    extend cmd_vector instead").
  - Slot-order delta: the deployed sibling's lower command is [sin,cos,vx,vy,wz]; OUR
    vforce_52000's trained order is [vx,vy,wz,sin,cos] (from the live eval env, validated by
    the whole maze eval). Our adapter permutes at the seam; wire contract unchanged.
  - M3(b) plan: run his export+adapt pipeline on vforce_52000; extend cmd_vector +8 slots for
    the LIVE hand_contact_force_cmd (per his note); keep pids g1_beyond_mimic; gate on his 17
    checks + a behavioral parity harness (adapted graph vs our torch policy, force live and
    zeroed). Remaining 学长-question (non-blocking, provenance only): which checkpoint the HW
    run actually used / whether a newer uncommitted export exists.

- **M3(b) DONE (2026-08-18) — full record in
  `locomanip_wt/m3b/docs/M3B_VFORCE_PORT.md`** (worktree branch `tacnav/m3b-vforce`
  off `origin/foot_trajectory`, sim-only, known-good models untouched).
  Wire model `export/tactile_controller/vforce_52000_policy.onnx` (obs[526]→29,
  sha256 84419117…): 61-d cmd_vector (53 + live force8, /50 scaling), internal
  [sin,cos|v] → [v|sin,cos] permutation, vforce crouch keyframe (−0.312/0.669/−0.363),
  pids g1_beyond_mimic verified == training constants, training-exact integrator gait
  clock (period 0.8, decay 0.95, standing (0,1)) as additive manager params.
  **Gates: torch↔ONNX parity 1.1–1.3e-05 (250 live steps, force channel exercised);
  sim2sim ALL 21 CHECKS PASSED** (17 mapped from the committed checker + 4 vforce
  checks; hand EE tracking 1.7 cm vs the committed stack's 2.0 cm benchmark).
  Known limitation (environment, not port): deploy-sim velocity tracking is
  value-dependent (0.2→0.12, 0.5→1.7 m/s runaway; mjlab reference 80% at 0.33) —
  M3(c) planner must close position loops and cap |vx|≲0.4 in this sim.
  NEXT: M3(c)/(d) — see below.

- **M3(c)+(d) DONE (2026-08-18), simulation only — record in
  `locomanip_wt/m3b/docs/M3CD_TACNAV_SIM.md`.** Guard node
  (`tacnav_bringup/guard_node.py`) is the sole writer of the manager's cmd_vel/force
  topics: envelope clamp+slew, latched trips on overspeed / height / tilt / stale
  inputs / per-hand force limit / force-feedback-lost, planner deadman, arm+reset
  services. **Guard gate: 14/14 checks pass** (isolated-domain fault injection);
  it found a real defect — a `max()` over both hands' sample ages meant one dead
  force sensor silently disabled the limit on the other hand (now per-hand).
  **End-to-end mission gate (3 runs, fresh stack each) — owner-ruled verdict:
  navigation PASS (3/3)**, goto arrived, return arrived, standing throughout, ZERO
  guard trips; **sustained-force regulation FAIL (0/3)** — no run held 8-12 N for the
  required 0.8 s (band reached: 12.09 N end, 10.21 N peak; not holdable).
  **M3 IS NOT CLOSED. The 0.8 s force-hold criterion STANDS** and is not replaced by a
  tap criterion; options are compared in `docs/M3_PRESS_OPTIONS.md` and any change is an
  explicit owner decision.
  Per-axis characterization (the envelope's basis): vx dead zone below ~0.2,
  non-monotonic (0.2->0.086, 0.3->0.018, 0.4->0.095, 0.5->0.252 m/s, with a 1.7 m/s
  runaway seen at 0.5); vy tracks ~1:1 at 0.2-0.3; wz delivers ~25% and couples
  ~0.25 m/s of parasitic translation. Envelope: vx [-0.15,0.40], |vy|<=0.30,
  |wz|<=0.40.
  **Key finding: the press-force problem and the velocity-response problem are the
  same blocker.** The base complies backward under press load (-0.04 to -0.09 m), so
  the wall recedes in pelvis frame and the servo chases it out of arm reach;
  station-keeping needs centimetre corrections, but commanded vx below ~0.2 produces
  no motion at all. Independently reproduces the TacNav MuJoCo result that sustained
  press holds are unavailable on this platform while brief taps work.
  **HARDWARE REMAINS STOPPED** (owner ruling); the velocity dead zone and the absence
  of base station-keeping remain hardware blockers. Blockers before any hw step: (1) the
  nonlinear/unstable velocity response — the guard can only catch a runaway after
  ~0.5 m of travel; (2) no base station-keeping mode; ~~(3) the guard's odometry source
  is sim truth~~ **CLOSED 2026-08-19** — the hardware launches run `leg_odom` (encoders +
  IMU, relative) and the truth firewall fails readiness closed if sim truth is present;
  ~~(4) the SensX 20 N clip caps any enforceable force limit at ~18 N~~ **NO LONGER
  APPLIES** — the hardware path carries no force input at all: contact is confirmed from
  the TactX image (`sensx_pub` → `sensx_contact_projector` → `ContactObs.v1`) and the
  palm wrench is evaluator-only, so the touch primitive is touch-and-retract rather than
  force-regulated. Blockers (1) and (2) stand, and both were measured in the deploy SIM;
  neither has been characterized on the robot.


---

## 7. Top risks & open questions for the deploy-repo owner

1. **Odometry quality is the program risk.** Order of attack (owner decision): first the M0 hardware check for a surviving Unitree estimator (bridge if alive); only then leg_odom. Feet pivot during lateral side-steps (their measured lesson) and crab-follow is exactly lateral stepping — whichever source wins gets a crab-step lateral-drift measurement before trust. Mitigations: IMU yaw as heading, PF σ tuned high, touch cadence from our `touch_gap_m` maintenance trigger. If both sources are unusable, fall back to cmd-vel dead reckoning + denser touching before considering exteroception.
2. **Which surface presses the wall, with which sensor?** Hand SensX blocks are commented out in the live bringup config (only chest is live); ViTac is binary. The press-validity logic needs a calibrated ≥10 N-range signal on the pressing face. → Ask: planned hand sensor mounting; if none, priority of the torque-residual observer.
3. **GR00T walk policy behavior at our operating point** (0.1–0.25 m/s creep, lateral crab, frequent stand↔walk transitions near a wall). Unknown until M0; if hostile, the planner's speed floor/hysteresis absorbs some, but re-training/policy swap is the owner's call.
4. **Press-force fidelity**: their sim wall-press measured ~90 N vs an 18 N reference (rigid geom + no cap). Our cap+rate+brake stack is the direct fix and should be the default press path in both sim and hw.
5. Sensor polarity (pitfall #7), doc drift (verify interfaces live), network peer config, Jetson headroom for an offboard-vs-onboard planner, `odom_to_tf_ros2` availability in sim.
6. **Coordination**: `src_ext/tactile_controller` submodule is pinned at an old commit; the `wall_mpc`/tracking-policy line is the owner's active work — agree on whether TacNav targets the decoupled stack only (this plan) or should eventually feed the tracking stack (M6).

## 8. Quick reference (verified topic/service names)

```
# command
/lower_body_controller/cmd_vel                geometry_msgs/Twist        (vx, vy, wz only)
/lower_body_controller/cmd_height             std_msgs/Float64
/g1_upper_pink_controller/left_hand           geometry_msgs/PoseStamped  (pelvis → left_rubber_hand)
/g1_upper_pink_controller/right_hand          geometry_msgs/PoseStamped
/g1_upper_pink_controller/external_wrench/g1robot_vitac_{left,right}   geometry_msgs/WrenchStamped
# state
/imu_broadcaster/imu                          sensor_msgs/Imu            500 Hz, frame pelvis
/joint_states                                 sensor_msgs/JointState     500 Hz (29, incl. tau_est)
/tf_server_{left,right}/pose_measured         geometry_msgs/PoseStamped  250 Hz, pelvis frame
/sensor{1,2}/sensor{1,2}/force                std_msgs/Float32           ~100 Hz, 0–20 N calibrated
/sensor{1,2}/contact                          std_msgs/Float64MultiArray [area_frac, centroid r, c]
/simulator/floating_base_state                nav_msgs/Odometry          SIM ONLY, ground truth
# services
/controller_manager/switch_controller         controller_manager_msgs/srv/SwitchController  (activate_/deactivate_/strictness)
/g1_upper_pink_controller/set_feedforward     deploy_transport/srv/IKFeedforward
/g1_upper_pink_controller/toggle_contact_regulation   std_srvs/srv/Trigger
/lower_body_controller/switch_policy          deploy_transport/srv/SwitchPolicy
/mujoco_node/reset_world · /mujoco_node/elastic_band_toggle              (sim)
```
