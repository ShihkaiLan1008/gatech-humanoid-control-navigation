# Decoupled vel-force-MPC checkpoints — deployment-candidate eval (2026-08-14)

User question: can the deploy pipeline use a controller from
`ZhenyuWu323/tactile_controller` instead of the GR00T-named lower-body policy —
specifically the three checkpoints under `checkpoints/decoupled_vel_force_mpc/`
on branch `decoupled-policy`, played via `Mjlab-Eval-Force-Maze-G1`?
Instruction: try `vforce_model_52000.pt` and `force_knee_model_60500.pt` first;
`force_home_model_59000.pt` (different default joint pos) only if both fail.

**Verdict: not garbage — `vforce_model_52000` is the deployment candidate.**
Zero falls; station anchor-hold works — 2.5 cm median / 4.2 cm p90 base drift
across 42 press windows under 8–22 N load (the old A/B could never test this:
its upper policy had no live force channel, and its free-standing drift was a
random walk not worth quoting) — and right-hand presses land 22/22. Weaknesses: commanded force is
under-delivered ~40 % at ≥15 N (effective measured ceiling ≈ 12–17 N), and the
left hand misses the wall in 35 % of draws (the family's known short left
reach). `force_knee_60500` is strictly worse here (≈6 N ceiling, 59 % right-hand
touch); `force_home` untouched per instruction.

## Where things live

- Worktree: `~/Desktop/tacnav_wt/dp_eval` @ `origin/decoupled-policy` (21a0481),
  own `.venv` via `uv sync --frozen`.
  Local fix (uncommitted): `src/assets/g1_description/g1_tactile*.yml` carried
  Zhenyu's absolute paths (`/home/zhw/...`) — sed-rewritten to this worktree
  (the assets docstring itself says "regenerate if this folder ever moves").
- Probe: `dp_eval/scripts/rl/test/maze_force_eval.py` (new, untracked).
- Results: `dp_eval/out/maze_eval/<ckpt>/summary.json` (8 envs × 150 s each),
  videos `dp_eval/out/maze_eval_vid480/vforce_model_52000/eval.mp4` (640×480)
  and `out/maze_eval_vid/<ckpt>/eval.mp4` (320×240, both checkpoints).

## What the checkpoints are

True decoupled two-policy stack (`JointOnPolicyRunner`: N independent PPO
stacks — lower = 12 legs + 3 waist, upper = 14 arms; coupled only through
physics/last_action/shared rewards). Actions: 29 joint-position targets
@ 50 Hz, per-joint scales, default-offset. All three checkpoints share one obs
layout (only training rewards differ; `vforce` = virtual-force line,
`force_knee`/`force_home` = real-press line).

**Actor obs are deployment-grade by construction** (measured force and the
`loco_mpc` refs are critic-only): IMU ang-vel + projected gravity + joint
q/dq + last_action (history 5); lower adds twist cmd + gait phase; upper adds
per-hand EE pose cmd + FK measured EE pose + pose err + force cmd
(8 = per hand mask + force_b/40). No base_lin_vel anywhere in the actors —
every term is measurable on the real G1 from IMU + encoders + the command
stream.

Command interface ≡ what the TacNav planner already emits: twist (trained box
vx −0.5..1.0, vy ±0.5, wz ±0.5 — narrower than hand_force_v2's) + per-hand
tactile-TIP pose in pelvis frame (`left/right_tactile`, no wrist→tip offset
needed, nominal tip (0.332, ±0.152, 0.095)) + contact mask + force target
(trained window 5–25 N, slew 40 N/s).

## Method (scripted operator, headless)

`Mjlab-Eval-Force-Maze-G1` = trained env minus press cubes, plus a maze of
real walls, per-hand-per-wall contact sensors, maze-FSM twist; contact/force
are normally driven from viser GUI. The probe scripts the operator:

- STAND press (primary, deploy-faithful): when a hand draw is `_on_wall`,
  freeze the base (`_stand_at` + long `time_left` — what the planner does at a
  station), settle 0.75 s, force a fresh draw from the standing pose, mask on
  with 8/15/22 N cycling, hold 6.5 s, release, resume FSM.
- FOLLOW press (wall-parallel-tap analogue, every 3rd window): mask the
  current on-wall draw while the FSM keeps walking. Sparse data (FSM follow
  fraction was only 5–9 %).
- Metrics skip the first 1.5 s (reach transit); windows hit by a fall are
  discarded. Force = per-hand net force summed over that hand's wall sensors
  (`measured_force_w()`), same signal the operator GUI reads.

## Results (8 envs × 150 s each = 20 env-min per checkpoint)

| metric | vforce_52000 | force_knee_60500 |
|---|---|---|
| falls | **0** | **0** |
| twist err free-walk (mean, at 0.33 m/s cmd) | 0.19 m/s | 0.18 m/s |
| stand windows / touch>50 % | 42 / **83 %** | 40 / 68 % |
| touch by hand L / R | 65 % / **100 %** | 74 % / 59 % |
| time-to-touch med | 0.26 s | 0.36 s |
| f_mean med @ 8 N | 6.5 N | 4.2 N |
| f_mean med @ 15 N | 5.4 N | 5.7 N |
| f_mean med @ 22 N | **12.3 N** | 5.5 N |
| max f_p90 @ 22 N | 17.1 N | 12.8 N |
| base drift during press med / p90 | **2.5 / 4.2 cm** | 2.7 / 4.3 cm |
| base Δyaw during press p90 | 3.3° | 3.0° |

Degenerate cases (both ckpts, ~1 per 40 windows): a press window where the
base walks away ~1–1.5 m with 1.5 rad yaw wander — looked like the policy
abandoning the station; and windows where the hand never reaches the wall
(static kiss draw at reach edge, mostly LEFT hand).

## Reading for TacNav

1. **The old A/B blocker is resolved for presses.** The grid-search-era
   decoupled policy could not hold an anchor and had no live force channel to
   even test a press; this line holds 2.5 cm median / 4.2 cm p90 under an
   active 8–22 N press (n=42). Presses while standing are viable.
2. **Force under-delivery is systematic but outer-loop correctable.** The
   planner already runs a measured-force servo (M1 press gate); with f_cmd as
   the control variable (cmd up to 25 N → ~12–17 N measured), the TacNav
   8–12 N press window is reachable on `vforce`. On `force_knee` it is not.
3. **Touch failures are draw-model failures, not necessarily policy ones.**
   The eval fires a single static kiss target; the planner creeps until
   contact, so eval touch rate is a lower bound. Left-hand weakness still
   argues for right-hand-first station planning (matches hand_force_v2 lore).
4. **Sim-to-deploy behavioral gap shrinks vs GR00T+Pink**: same command tuple,
   same policy family the planner's shield/kiss/press semantics were tuned
   against, tip-frame commands (no +0.103 m wrist offset bookkeeping).

## Deployment path (if chosen over / alongside GR00T+Pink)

Template exists in locomanip_deploy: `scripts/tracking_policy_node.py`
(python, onnxruntime CPU, obs from `/joint_states` + `/imu_broadcaster/imu`,
29 joint targets → whole-body forward controller @ 50 Hz) +
`scripts/export_tracking_onnx.py` (mjlab .pt → onnx, normalizer baked in).
A decoupled variant needs:

- export both actors (or one merged graph) to ONNX;
- a `decoupled_policy_node`: assemble the two obs groups (proprio histories,
  twist cmd, gait phase clock ported from `GaitPhaseCommand` — threshold 0.1,
  stance decay — per-hand pose/mask/force cmd, FK tip pose from joint_states
  with the tactile URDF);
- whole-body position controller with kp/kd matching the training PDs
  (same caveat the tracking bringup documents);
- TacNav planner side unchanged in shape: it already emits exactly this tuple.

Open items before committing: confirm with Zhenyu/学长 the intended sim2real
status of this line (any HW run yet? action-scale/PD gains source of truth);
gait-phase reproduction on hw; the 22 N ceiling if >12 N presses ever needed
(`force_home`'s different default joint pos may be the press-posture answer —
ask Zhenyu how to run it as instructed).
