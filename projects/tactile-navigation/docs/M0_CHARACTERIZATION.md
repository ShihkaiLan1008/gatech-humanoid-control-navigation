# M0 Characterization — deploy stack (GR00T + Pink), sim

Status: **press phase DONE · walk phase BLOCKED on GR00T ONNX weights · hardware checklist PENDING robot access**
Setup: locomanip_deploy @ `jjang318/docker` (37934f8), sim via `g1_velocity_kinesthetic.launch.py simulation:=true` (headless xvfb, RTX 4090 host), config `g1_decoupled_wbc_admittance`. Tool: `scripts/m0_characterize.py` (untracked), data `out/m0/press_0812_163754.jsonl`.
All numbers are for THIS stack. Nothing is carried over from tacnav's `hand_force_v2` (different controller family).

## Press/arm phase — measured results (left hand, pelvis frame, z = 0.25)

| Quantity | Value | Notes |
|---|---|---|
| End-to-end tracking (IK solution vs realized hand, whole run) | **rms 3 / 1 / 6 mm (x/y/z)** | joints track commanded within ~0.02 rad; sim PD healthy |
| Forward reach saturation | **x ≈ 0.395 m** | commanded 0.50; IK solution stops at 0.395 (joint-limit saturation, tightened elbow/shoulder limits in pink node). Deploy-stack analogue of tacnav's 0.45 REACH_M — **planner press cap must be `min(kiss+margin, ~0.39)`** and treat IK saturation as its own press outcome |
| Lateral reach saturation | **y ≈ 0.375 m** (at x = 0.35) | commanded 0.40 → IK 0.374, realized 0.377. Wider than hand_force_v2's ~0.29 — do NOT reuse old lane constants |
| Tracking lag during 0.05 m/s creep | **≤ 240 ms** (mm-level error mid-sweep) | planner-side creep at 50 Hz publish is well inside the servo bandwidth |
| Height hold | z 0.244–0.247 vs cmd 0.25 | few-mm sag, constant |
| **Jump test: 0.10 m instantaneous target step** | **peak hand speed 4.22 m/s** | `jump_clamping.enable: false` by default — pink slews immediately. **Confirms planner-side rate limiting is mandatory** (and/or enable pink jump clamp) before any wall contact |

## Bring-up defects found (each cost real debugging; write-ups for the PR later)

1. **Spawner cascade on missing weights.** All inactive controllers share ONE spawner process (`spawner idle upper_body upper_body_effort lower_body lower_body_pd --inactive`). `lower_body_controller` fails to configure (no `export/GR00T-*.onnx` — `export/` is gitignored) → the spawner process dies → **`lower_body_pd_controller` is never even loaded**. Fix used: re-spawn it manually with the same param file. Real fix candidate: one spawner per controller, or order the fallible one last.
2. **JTC capture-and-hold freezes garbage postures.** With legs/waist limp (defect 1), the dangling robot wound `waist_yaw` to **−2.26 rad (−129°)**; activating `lower_body_pd_controller` then held that twist. Pink's reduced model locks the waist at 0 → every "pelvis-frame" EE pose was the intended pose rotated by the twist (verified numerically: |t| matched, R_z(−129°) mapped ikfk→measured). **Bring-up invariant: activate the leg/waist hold only from a known posture** (sim: `/mujoco_node/reset_world` first; hardware: from idle/damp stand). Same failure class as tacnav's "anchor press at stance, not at drawn pose".
3. **`/tf_server_left/pose_measured` verified correct** (= `tf2_echo pelvis left_rubber_hand`, 250 Hz) once posture is sane — earlier mismatch was defect 2, not a frame bug.
4. `odom_to_tf` IS installed (the survey's caveat is resolved); `/simulator/floating_base_state` live at ~500 Hz.
5. Env note: private gatech submodule `diffusion_humanoid` blocks `pixi install`'s solve; local stub (`name=diffusion-policy 1.0.0`) + `pixi run --frozen` everywhere works and touches no tracked files. MVP never imports it.

## Walk phase — blocked, protocol ready

`lower_body_controller` is `unconfigured`: `Load model from export/GR00T-WholeBodyControl-Balance.onnx failed. File doesn't exist`. The moment the two GR00T onnx files land in `export/`, run:
`pixi run --frozen python scripts/m0_characterize.py --phase walk`
(profiles already implemented: vx steps 0.05–0.5, crab vy ±0.1/±0.2, wz, 5× stop-precision, low-speed floor 0.02–0.08 around the 0.05 auto-switch threshold; elastic band toggled off automatically).

## Hardware checklist — pending robot access

`docs/M0_HARDWARE_CHECKLIST.md`: Unitree odom availability under low-level control (decides `unitree_odom_bridge` vs `leg_odom`), SensX mounting/polarity, press end-effector face.

---

# M1 — sim scaffolding results (2026-08-12, all gates PASS)

Branch `tacnav-integration` in locomanip_deploy. New: `scripts/make_scene_mjcf.py` (scan-JSON → walls + palm fts sensors + `--test-wall` + `--weld-base`), sim-only `palm_fts_sensors` xacro macro, vendored `force_torque_sensor_broadcaster` **4.40.0** (jazzy tip breaks against installed ros2_control 4.43 API; version must match robostack), `src/planner/tacnav_bringup` (sim_touch_bridge, odom_shim, `g1_tacnav_sim.launch.py` with one-spawner-per-controller), `scripts/m1_press_gate.py` + `scripts/contact_probe.py`.

**Gates:** room2 (4 faces) and the 51-face scanned map both load + render in deploy sim (`out/m1/scene_*.png`); `/tacnav/odom` live at 100 Hz; force-servoed press on `/sensor2/sensor2/force` hit the 10 N target (stop at 10.12 N, clean 0 N retract, no self-collision, no cross-hand contamination).

**Measured constants (welded-base rig, left hand, natural orientation):**

| Quantity | Value |
|---|---|
| **Fingertip offset beyond `rubber_hand` frame origin** | **0.103 m** — the planner's kiss model on this stack MUST add this (tacnav's tip-frame convention does not carry over) |
| Contact-onset sharpness | 0→14 N within ~60 ms at 0.03 m/s creep; at 0.02 m/s stop lands 10.1 N on a 10 N threshold (bridge 100 Hz + smoothing) |
| Position-hold force relaxation | 10.1 → 6.32 ± 0.23 N over ~1.5 s, then stable — holding a force target needs active regulation (port `shield.advance` force-gain) |
| Penetration at stop | meas 1.0 cm past command (compliance) |

**Traps found (each verified by experiment):**
1. **Identity-quat EE commands fold the wrist into self-collision** — hand pressed its own forearm at 26–52 N, read by the palm sensor as phantom contact ("stuck at x=0.174, steady 8.8 N"). Press commands must hold the arm's natural/latched quat (`hold_quat`), never a blind identity.
2. **A full-width test wall sits inside the RIGHT hand's default-pose envelope** (~2 N permanent graze contaminating sensor1) — test obstacles must stay out of the idle hand's reach.
3. **fts baseline is pose-dependent** — recalibrate after posture setup, out of contact (bridge `~/recalibrate`), never at node start.
4. **Elastic band kp=100 N/m is soft**: a 10 N press displaces the hanging robot ~10 cm and it swings back (phantom re-contact). Arm-only force work needs `--weld-base`; locomotion work needs the band OFF.
5. MJCF offscreen framebuffer is 640×480 — python-Renderer requests above that fail (misleading `_gl_context` error); one Renderer per process.
6. The mujoco camera comment in g1.xml notwithstanding, cameras are NOT auto-published as ROS topics in this launch path — use offline mujoco rendering or Foxglove for visuals.
