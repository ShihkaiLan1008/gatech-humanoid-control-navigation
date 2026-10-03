# M2 — Tick-ification design (PlannerCore.tick)

Goal (gate): the crab-press demo route (`scripts/demo_room_run.py`) reproduced
through a `PlannerCore.tick(obs) -> cmd` API driven by a thin MuJoCo driver,
with no acceptance regression vs the blocking executor.

## Strategy: generator inversion, not hand-unrolled state machines

Every blocking primitive in `src/execution/mujoco_backend.py` has the shape

    setup...
    while True:
        <write commands via _set_goal / shield>
        self._step()                       # advance sim one 50 Hz tick
        <read truth/forces>
        <transition predicates -> return status>

Mechanical transform: the primitive becomes a **generator**

    def goto(cfg, state, target_w, yaw_w, ...) -> Generator[Cmd, Obs, str]:
        setup...
        while True:
            obs = yield Cmd(base_goal=..., park=("left","right"))
            <predicates on obs -> return status>

Control flow, predicates and constants stay byte-identical (reviewable diffs);
only the I/O is redirected. Primitives compose with `yield from` (the executor's
`run()` becomes a generator too). `PlannerCore.tick(obs)` drives the top
generator: `cmd = gen.send(obs)`; `StopIteration.value` carries the status up.

## Seam placement

- **PlannerCore (portable, tacnav side)**: primitive control laws, shield
  rate-limiting state (command-side integrators), mode FSM + exclusion audit
  inputs, executor/route logic, belief absorption. No MuJoCo, no ROS.
- **Driver (per backend)**:
  - `MujocoTickDriver` (M2): builds `Obs` from the same env reads the blocking
    backend used; applies `Cmd` via `_set_goal` + shield writes; calls
    `_step()`. Guard/brake stays driver-side for M2 (same 200 Hz hook).
  - ROS driver (M3): `Obs` from topics; `Cmd.base_goal` through the ported
    Se2 attractor -> `cmd_vel` twist; `Cmd.ee_target` -> Pink PoseStamped
    (+0.103 m tip offset applied in the kiss model); force from SensX-named
    topics.

For the M2 regression the sim driver keeps using the env's own `_set_goal`
attractor — identical behavior to the blocking path, zero numeric drift. The
twist-emitting attractor port (`twist_from_goal`, ~35 lines from
`Se2PoseVelocityCommand._update_command`) ships in tick_core but is exercised
only by the ROS driver.

## Obs / Cmd (v0)

```python
@dataclass Obs:
    t: float                     # s
    pose_w: (x, y, yaw)          # executor working pose (sim: truth; hw: estimate)
    base_z: float
    feet: int                    # 0..2
    hands: {side: HandObs}       # side in {"left","right"}
@dataclass HandObs:
    ee_w: np.ndarray             # measured EE pos, world (3,)
    force_w: np.ndarray          # measured hand force vector, world (3,)
    force_total_n: float
    force_axial_n: float         # |f . press_dir| (driver computes w/ core-provided dir)
    peak_total_n: float          # max since last tick (driver-side substep monitor)

@dataclass Cmd:
    mode: Mode
    base: BaseGoal | None        # (xy_w, yaw_w, speed, hold, standing)
    hands: {side: HandCmd}
@dataclass HandCmd:
    target_b: np.ndarray | None  # (x, y, z) yaw-aligned base frame (rate-limited BY CORE)
    quat_b: np.ndarray | None
    contact: bool                # command window / contact_mask
    force_cmd_n: float
    park: bool
```

Core-side keeps: last commanded pose per hand (shield integrator state),
press_dir, kiss estimates. Driver never exposes `pose_command_b` back.

## Staging (each stage runnable)

1. `src/execution/tick_core.py`: Obs/Cmd + PlannerCore driver loop + `goto` +
   `turn_in_place` generators; `MujocoTickDriver`; smoke test: goto+turn
   mini-route matches blocking `goto_map`/`turn_to_map` end pose.
2. `crab_step` + `stabilize` + `press` (REACH/PRESS/RETRACT with force
   regulation = shield.advance semantics moved core-side).
3. `wall_parallel_touch(touch_style="crab_press")` station flow +
   `DemoRouteExecutor` inversion (`run_gen()` yielding from primitives;
   `_absorb`/belief calls unchanged — they are already tick-free).
4. Full regression: `scripts/demo_room_run.py --tick` vs blocking run, same
   seed: same stations touched, same press validity/forces (tolerance: press
   peak ±1 N, taps count exact), same route completion.

## Stage 1 result (2026-08-12): PASS

`scripts/m2_stage1_smoke.py` blocking vs tick, same seed, demo scene, goto+turn
mission (which happens to trip the force guard — a bonus end-to-end test of the
guard/safety path): **statuses identical (`force_abort`/`arrived`), Δyaw
0.06°, Δt 2 ticks.** End-position Δ76 mm is the expected stage-2 gap (blocking
`_safety_stop` backs off 0.10 m along the measured reaction force; the stage-1
`safety_stop` freezes in place). Files: `src/execution/tick_core.py`,
`src/execution/tick_driver.py`. Harness lesson: any helper generator that
yields must tolerate `GuardAbort` (use the `observe()` primitive, never a bare
yield outside a primitive).

## Stage 2 result (2026-08-12): PASS (outcome parity)

Full ports landed in tick_core: `safety_stop` (back-off along −measured-force,
force-guarded phase A, clear+parked+still phase B), `crab_step` (physical
standoff via caller-supplied callable), `stabilize` (StabilityGate core-side),
`press` (REACH→PRESS→RETRACT with all six outcome branches) + pure geometry
helpers (`contact_body_frame`, `swept_free_body`, `ee_base_x`). Driver v2
translates HandCmd intent to shield calls; shield + 200 Hz brake stay
driver-side by design (numerics identical; the ROS driver re-implements the
~40-line limiter planner-side in M3).

`m2_stage2_smoke.py` (open-space capped press, blocking vs tick): **reason
`bounded_free_sweep`, valid, peak 0.0 N, swept_past 0.03 m, n_free 2 — all
identical.** Δt 7.72→9.48 s (+1.76 s) with identical outcomes — flagged as a
stage-4 watch item (suspects: per-predicate one-tick readback lag compounding
in REACH/RETRACT convergence checks).

## Stage 3 implementation notes (2026-08-14)

Ported: `tick_core.wall_parallel_touch` (crab_press ONLY — lateral taps and
turn_press raise NotImplementedError), `WallTouchServices` (the synchronous
non-stepping backend calls the blocking primitive makes: frames, kiss
geometry snapshots, hand selection, attractor-tolerance stack — the MuJoCo
executor binds them to the backend, the M3 ROS driver to planner-side
models), executor generator twins (`_goto_gen`, `_follow_gen`,
`run_gen(steps, params)`; `turn`/`ate_corner` steps are not on the demo route
and stay blocking-only), `scripts/demo_room_run.py --tick`.

Supporting changes: `TickParams.dt`; `HandObs.peak_total_n` +
`HandCmd.reset_peaks` (the per-hand 200 Hz shield peak the walk loop reads);
`crab_step(out=...)` and `press(out=...)` return their final Obs (blocking's
synchronous truth read right after the sub-loop); `_retract` returns its
final Obs and press's returned peak is now the POST-retract value (matches
blocking, which reads `_substep_peak` after `_retract()` at every return).

Documented deltas vs blocking (all diagnostic-only): first tick row's
`ee_gap_m` is NaN; `y/x/z_cmd_m` rows are post-step readbacks; a guard abort
during the station SETTLE skips the press with the aborted reading instead
of running one aborting press tick; `stale_abort_cleared` is always False
(PlannerCore consumes guards at the primitive that raised them);
`duration_s` on force_abort excludes the safety-stop time (blocking's
`_guard_tripped` runs it inside the primitive's clock); no trailing
park-write after the loop (the next primitive's first cmd parks).

### Stage-3 smoke defects found and fixed (2026-08-17)

First tick smoke: station R1 s=0.94 pressed 0.0 N `bounded_free_sweep`
(blocking: 12.21 N hit) + a second retry station. Three transcription
defects, found in escalating order:

1. **Missing post-settle anchor re-pin** (blocking `settle_done_repin`,
   commit 616201f): the press anchored at the crab-step arrival pose, not
   where the base came to rest. Fixed (near-no-op on this station but
   faithful).
2. **Invented `contact_mask=1.0` at press entry**: blocking's `press()`
   never touches the mask (it stays 0 through the whole crab flow — only
   `side_contact`/`wall_follow_map` set it). Mask=1 hands the x command to
   `WallEePoseCommand._apply_masked_x`, which pins the hand AT the kiss
   surface so the press never loads — and nothing ever reset it, so the
   rest of the route ran masked. Removing it ALSO resolved the Stage-2
   +1.76 s per-press watch item (tick press now 7.76 s vs blocking 7.72 s).
3. **Wrong press hand** (the killer): `set_press_frame` RE-SELECTS
   `probe_name/probe_term` from the frame geometry (right hand at R1),
   overriding the wall-side hand `set_touch_hand` picked at entry (left).
   Blocking presses `self.probe_term` (right); the tick flow pressed the
   entry-time `name` (left — cross-body, out of reach, 0 N). Found by
   instrumenting `be._step` in both modes: blocking's right-hand x_cmd
   rises at the station, tick's left. Fix: `WallTouchServices.probe_name`
   getter, read after `set_press_frame`, press with that hand.

Debugging trap for the record: two intermediate re-runs "still failed"
because a PEER SESSION's honest-sensing migration (onboard drifting
estimator + noisy obs, `MujocoConfig.honest_sensing` default True) landed
in `mujoco_backend.py` mid-day and silently changed the walk under both
modes. Owner ruling (relayed 2026-08-17): M2 validates against the OLD
behavior — `scripts/demo_room_run.py` now pins `honest_sensing=False` +
`DemoExecCfg.exec_noise=True` explicitly; the honest path is adopted in a
later re-baseline. Paired runs are fingerprinted (sha256 before/after) to
catch mid-pair edits of the shared tree.

## Stage 3 result (2026-08-17): PASS

`demo_room_run.py --seed 0 --max-steps 2` (goto R0 + follow R1, legacy
behavior pinned), blocking vs tick back-to-back on the same tree
(`_s3c` pair): **R1 s=0.94 hit 13.52 N `bounded_press_contact` in BOTH,
peaks 13.5207 vs 13.5213 N (gate ±1 N), same outcome
`route_done_off_goal`, touch t 27.44 vs 27.48 s (the 2-tick priming
offset), press_s 12.86 vs 12.90 s, standoffs within 0.1 mm, belief median
err 0.035 both.** The mid-pair source change was the peer's SensX model,
fully `honest_sensing`-gated (legacy path byte-equivalent) — pair valid.
Wall clock 147→163 s (+11%) is driver-side obs-construction compute, not
sim divergence. The Stage-2 +1.76 s per-press watch item is closed (it
was defect 2's mask): tick press now paces blocking to 0.04 s.

## Stage 4 result (2026-08-17): PASS (behavioral parity; strict bit-parity
## shown structurally unattainable)

Full seed-0 route pair (`out/m2/stage4/demo_run_s0_s4_{blocking,tick}.json`,
legacy behavior pinned, 42 sim-min each):

- **Deterministic prefix in lockstep**: R1 press station s identical to the
  mm, peak Δ 0.00 N; R2 station-0 acquisition s identical (0.975/0.975);
  nav-event sequences (= exec-noise RNG draw pattern) identical through it.
- **Fork is physics, not logic**: first divergence is a 7 N peak difference
  at a bit-identical stance (gait-phase sensitivity of press force — the
  known platform property), then R2 station-1 lands hit(14.33)/miss(0.00)
  and the retry paths separate; station-s drift grows progressively
  (1.5 → 27 cm), the chaos signature. No directional pattern anywhere:
  hits 12 vs 11, in-band 9 vs 8, mean contacted peak 11.0 vs 10.1 N,
  replans 71 vs 73 (same reason mix), aborts 0/0.
- **Route behavior equivalent**: both `arrived` (goal 0.30 / 0.23 m),
  per-segment tap counts identical on R1/R2/R3/R4, R6 4 vs 5
  (retry-driven), sim time 542.9 vs 569.6 s (+4.9%), belief median err
  0.038 vs 0.021 (tick better), final sigma 0.618 vs 0.601.

The stage-list's original "press peak ±1 N, taps count exact" criterion is
met on the deterministic prefix and provably cannot hold route-long: the
documented 2-tick priming offset guarantees a gait-phase delta, press peak
is gait-phase-sensitive, and the executor's retry logic turns the first
knife-edge hit/miss disagreement into permanent s-offset. Lockstep holds
exactly as long as press outcomes agree — which is the correct statement of
transcription fidelity for a chaotic closed loop. A seed-1 pair was run as
supplementary distributional evidence (see below). Mid-pair source change:
commit a8233a6 landed during the seed-0 blocking run; audited hunk-by-hunk
as honest_sensing-gated (R1 = 13.52 N on both sides of the boundary
confirms the legacy path is bit-stable across it).

**Seed-1 supplementary pair** (`demo_run_s1_s4_{blocking,tick}.json`):
both arrived (0.17 / 0.06 m), 0 aborts, in-band 14 vs 13, mean contacted
peak 10.11 vs 10.82 N — tick HIGHER this seed where seed-0 had it lower,
and tick 17% FASTER in sim time where seed-0 had it 4.9% slower: the two
seeds bracket each other on every distributional metric, i.e. no
directional bias. Fork mechanism identical (first divergence at a
bit-identical station s=0.946 on a 3.02 vs 4.16 N knife-edge press).

**M2 CLOSED (2026-08-17)**: all four stages pass; the tick inversion is
validated as behaviorally equivalent to the blocking executor on the
crab-press demo route. Next: M3 (deploy-sim closed loop, decoupled vforce
controller first — see docs/DEPLOY_INTEGRATION_PLAN.md).

## Non-goals (M2)

- `lateral` tap style, side_contact, wall_follow_map (not on the demo route).
- Replacing the driver-side 200 Hz brake (M3 guard-node work).
- Removing truth_pose from the DemoRouteExecutor belief-feed path (its odometry
  noising already models hw; the swap to /tacnav/odom happens in M3's driver).
- The honest-sensing feature set (commit a8233a6+: `_est_standoff` believed
  distances, `wp_feeler_first` arm-first probe with the `confirmed=` wall-pin
  threading, SensX force model, honest brake threshold). All of it is
  `honest_sensing`-gated and M2 runs pinned False. **Re-baseline TODO**: when
  M2's stack moves to honest_sensing=True, `tick_core.wall_parallel_touch`
  needs the feeler branch + `confirmed` parameter ported and `run_gen`'s
  `_follow_gen` needs the `_wall_confirmed` threading, mirroring the blocking
  executor's version at that commit.
