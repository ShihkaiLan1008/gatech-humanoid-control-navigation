# M0 Hardware Checklist — Unitree odom availability under low-level control

Purpose: decide **bridge vs leg_odom** before writing any odometry code.
Run ON the real G1 setup **while the deploy pipeline's low-level stack is active**
(GR00T `lower_body_controller` running through `g1_hardware_interface`, i.e. the
built-in Unitree motion service is OFF / preempted by `rt/lowcmd`). Sim does not
answer this question — MuJoCo keeps a floating-base state that hardware may not have.

Background: `g1_hardware_interface` subscribes **only** `rt/lowstate` (low-level).
Unitree's high-level `SportModeState` (carries `position[3]`, `velocity[3]` odometry)
is not consumed anywhere in the deploy repo, and per Unitree's docs it may stop
publishing once the built-in motion service is disabled. Whether *some* odometry
DDS topic survives low-level mode is exactly what this checklist measures.

## Procedure (all under `pixi run`, on the machine that sees the robot's DDS)

1. Bring up the normal stack on hardware:
   ```bash
   pixi run launch_g1_kinesthetic simulation:=false
   # activate lower_body_controller as usual (robot standing under GR00T)
   ```
2. Enumerate candidate topics:
   ```bash
   pixi run ros2 topic list | grep -Ei 'odom|sport|loco|state'
   ```
3. For every candidate (typical names: `/sportmodestate`, `/odommodestate`,
   `/lf/sportmodestate`, `rt/odommodestate` via the DDS bridge):
   ```bash
   pixi run ros2 topic info /sportmodestate
   pixi run ros2 topic echo /sportmodestate --once
   ```
4. If a topic exists and echoes: check it **keeps updating while the robot walks
   under GR00T** (not a frozen last value):
   ```bash
   pixi run ros2 topic hz /sportmodestate        # expect a steady rate
   # drive the robot ~2 m with teleop, confirm position[] moves accordingly
   ```
5. Record: topic name, message type, rate, and whether position/velocity track
   actual motion. Paste the raw outputs into this file under "Results".

## Decision rule

- **Readable + tracks motion** → write a light `unitree_odom_bridge` node
  (SportModeState-like → `nav_msgs/Odometry` on `/tacnav/odom`), then measure
  **lateral-drift** during crab-stepping (the known failure mode: feet pivot in
  side-steps). leg_odom is NOT built.
- **Absent or frozen under low-level control** → build the planned `leg_odom`
  (stance-foot FK + IMU yaw; IMU yaw authoritative per the 2026-07/08 wall-branch
  lesson).
- Either way the signal is a **relative** motion estimate, not absolute map pose —
  the PF + wall-touch correction stays load-bearing regardless of which source wins.

## Also verify in the same session (cheap while you're there)

- `ros2 topic hz /imu_broadcaster/imu` and `/joint_states` (expect ~500 Hz).
- SensX hand sensors: mounted? `ros2 topic list | grep sensor` — squeeze each pad,
  watch `/sensor{1,2}/sensor{1,2}/force` to confirm the right↔left USB mapping
  (documented pitfall: comments disagree across configs; cabling decides).
- Note which end-effector face is planned for wall pressing (rubber-hand palm vs
  sensor pad) — decides where the press force reading comes from.

## Results (fill in)

- Date / operator:
- Topics found:
- Verdict (bridge | leg_odom):
- Lateral-drift measurement (if bridge):
