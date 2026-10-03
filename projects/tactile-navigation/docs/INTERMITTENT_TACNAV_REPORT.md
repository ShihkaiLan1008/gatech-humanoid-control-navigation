# Persistent intermittent tactile navigation — first physical pipeline baseline

**Verdict: PASS, with caveats that matter.** One persistent MuJoCo episode, one
reset, the frozen MPC-RL checkpoint, all eight acceptance criteria met on both
the left- and right-turning scenes, zero failures logged in the reported runs.
Episodes are **not bit-reproducible**, and one earlier run of this same code did
fail — see §7.

`LOCOMOTION` and `CONTACT` never overlap. The G1 faces the wall, walks sideways
with both hands parked, stops, waits for both feet and a low base velocity,
probes, updates a belief from realized contacts and swept free space, then
either resumes the same wall or executes the turn the **map** demands at the
junction its **touches** located.

---

## 1. Reproduction

```bash
cd <repo root>
PYTHONPATH=src:tactile_controller/src MUJOCO_GL=egl \
  uv run --project tactile_controller \
  python -m local_tactile.pipeline \
    --config configs/intermittent_tacnav_corner_v1.json \
    --out    out/intermittent_tacnav_seed11
```

Mirrored scene — identical config except `travel: "right"` — which turns the
other way:

```bash
PYTHONPATH=src:tactile_controller/src MUJOCO_GL=egl \
  uv run --project tactile_controller \
  python -m local_tactile.pipeline \
    --config configs/intermittent_tacnav_corner_v1_mirror.json \
    --out    out/intermittent_tacnav_seed11_mirror --no-render
```

Tests (numpy only, no GPU, no simulator):

```bash
PYTHONPATH=src python3 -m pytest tests/local_tactile -q
# 51 passed
```

Task `Mjlab-MPC-Guided-Wall-Following-G1`, seed 11, checkpoint
`tactile_controller/checkpoints/hand_force_v2/model_57000.pt`, RTX 4090.

## 2. Acceptance — measured

| criterion | primary (`travel: left`) | mirror (`travel: right`) |
|---|---|---|
| one persistent MuJoCo episode | **PASS** — 1 env, 1 `reset` | **PASS** |
| zero teleportation / stance reset | **PASS** — max base step **8.0 mm** | **PASS** — 10.0 mm |
| peak contact force < 25 N | **PASS** — **7.20 N** total (4.83 N axial) | **PASS** — 3.99 N |
| yaw change consistent with the mapped corner | **PASS** — **−97.8°** vs map **−90.0°** | **PASS** — **+94.6°** vs map **+90.0°** |
| correct next-wall selection | **PASS** — vertex 1 → edge 1 | **PASS** |
| physical arrival within 0.20 m | **PASS** — **0.135 m** | **PASS** — 0.104 m |
| LOCOMOTION ⊥ CONTACT | **PASS** — **0.00 N** hand force in non-contact modes; 0.000 m/s commanded twist while parked (after the 0.4 s slew drain) | **PASS** |
| clear failure attribution | **PASS** — CSV emitted, **0 failures**; owners `planner`/`estimator`/`controller`/`route` | **PASS** |

Realized sequence (`intermittent_phase_timeline.json`, primary):

```
LOCOMOTION → STABILIZE → (standoff out of band) → LOCOMOTION → STABILIZE
→ PROBE (station 0)  → route: CONTINUE_EDGE
→ LOCOMOTION → STABILIZE → PROBE (station 1) → route: TURN
→ LOCOMOTION (turn point) → STABILIZE → TURN (−90°) → STABILIZE
→ LOCOMOTION (wall 2) → STABILIZE → PROBE (confirm wall 2) → ARRIVED
```

112 s of simulated time for one corner.

## 3. What the touches bought

The map handed to the route layer is exact in **shape** and wrong in
**registration**: the corner's along-wall coordinate carries a prior of
**+0.18 m** (σ 0.20 m) against a true 1.00 m. Nothing but touch can close it.

| quantity | primary | mirror |
|---|---|---|
| probes | 5 over 2 stations (3 contacts, 2 swept-free) | 9 over 2 stations (7 contacts, 2 swept-free) |
| terminus estimate | **0.983 ± 0.058 m** (err **−17 mm**) | 1.069 ± 0.052 m (err +69 mm) |
| map registration | prior **+0.180** → posterior **−0.002 ± 0.056 m** | +0.180 → +0.076 ± 0.050 m |
| association | vertex 1, Mahalanobis² **0.89** (χ²₁ gate 3.84), one candidate gated | vertex 1 |
| fusion | innovation −0.197 m, gain 0.92 | — |

Probe-by-probe (primary), every belief update sourced from realized
end-effector state:

| t [s] | station | ATE phase | commanded s | **realized** s | outcome | peak force |
|---|---|---|---|---|---|---|
| 14.6 | 0 | identity | 0.660 | 0.646 | contact | 2.72 N |
| 34.8 | 1 | identity | 1.131 | 1.108 | swept free | 0.00 N |
| 50.6 | 1 | identity | 1.071 | 1.037 | swept free | 0.71 N |
| 62.4 | 1 | **pose** | 0.891 | 0.902 | contact | 2.13 N |
| 109.2 | 1 (edge 1) | confirmation | 0.650 | 0.626 | contact | 2.05 N |

The frozen identity → pose → coverage schedule is visible in that column: three
identity probes settle *whether* the wall ends in reach, one guarded pose probe
localises *where*, coverage stops the fan. Station 0's single probe decided
`CONTINUES` and the route resumed the same edge without turning.

## 4. What is physically executed, and what is not

**Physically executed in MuJoCo, by the frozen policy, in one unbroken episode:**

- **Every metre of base motion.** Sideways wall-facing locomotion, the approach
  to each probe stance, the walk past the corner to the turn point, the 90°
  turn, and the second-wall leg are all policy actions stepping the simulator.
  The 8 mm bound on per-step base displacement is the evidence that no pose was
  ever written.
- **Every contact.** Wrist-capsule/wall collisions reported by the
  per-hand-per-wall contact sensors; forces measured, never modelled. The two
  "swept free" observations are real arm sweeps that carried the realized wrist
  origin ≥20 mm past the mapped face plane and found nothing.
- **The physical turn**, and the stability gate (foot loading from the
  `feet_ground_contact` sensor, base velocity from simulator state).

**Not physical — stated plainly:**

- **Base pose is read from simulator state**, standing in for an onboard state
  estimator. No odometry drift, IMU noise, or slip. This is the largest gap to
  hardware. Note what it does *not* buy: the corner's along-wall location is
  withheld via the registration prior, so knowing the base pose does not reveal
  the corner.
- **The floorplan is given** — edge directions, standoffs, wall lengths, the
  junction list, and therefore the turn angle. What is estimated is where that
  map sits along the wall being followed.
- **The wall face's lateral line is map data**, used by the kiss solve and the
  free-space test. Only the along-wall terminus is estimated.
- **Which wall geom a contact came from is recorded but never used** by the
  belief; updates take only the realized end-effector position and the measured
  force.
- **Sensing is a wrist force/contact sensor, not a tactile skin** — no contact
  normal, no contact patch.
- **One scene, one seed, two mirror variants.** A baseline, not a statistic.

## 5. Design decisions worth flagging

**The validation corner is convex, and measurement forced that.** At a
*concave* corner the second wall closes across the path 0.23 m ahead of a base
already at its wall standoff, and the probe arm's lateral sweep enters that wall
before any forward press can sense it. With a press-along-`+x` hand and no base
yaw permitted during `PROBE`, a concave corner is **not sensable from a parked
stance**. The earlier `persistent_real_task2` pipeline solved this by yawing the
base ±70° to aim the probe — which this design forbids. A convex corner is
sensable exactly as the frozen interface allows: probes past the terminus sweep
into free space. Concave corners need a lateral (along-travel) press or a
permitted probe yaw; neither is in scope, and both are the obvious next step.

**The ATE estimates geometry, not turns.** `LocalGeometryATE` carries a grid
posterior over one latent — the along-edge coordinate of the followed wall's
terminus — plus a stance-local binary identity (`CONTINUES`/`ENDS`). It never
mentions left or right. `floorplan.py` computes the required turn as
`wrap(facing_yaw(next_edge) − facing_yaw(this_edge))`; mirroring the scene flips
it from −90° to +90° with no code change, confirmed by both the offline tests
and the mirrored episode.

**"Physical yaw change" is measured leg-to-leg, and the two obvious
alternatives are both wrong.** The heading change *across the TURN mode*
undercounts — the base drifts several degrees toward the target before the turn
is commanded (mirror: 71.1° of TURN-mode change for a 90° corner). The
*end-of-episode* heading overcounts — it carries the second leg's heading
tracking error (mirror: 101.2°). What the criterion is about is the heading held
while parked on wall 2 minus the heading held while parked on wall 1, so that is
what `leg_heading_change_rad` computes and what the acceptance uses. All three
are in every summary.

**One documented deviation from the frozen phase rule.** The pose phase
localises a terminus inside the probe fan, so it is entered only once the
identity decision is `ENDS`. A decided `CONTINUES` goes straight to coverage —
no terminus in reach to localise, and probing on would be motion for its own
sake. Tested (`test_continue_decision_skips_the_pose_phase`).

**A miss only counts as free space if the sweep earned it.** The test compares
the realized wrist origin against the live kiss solution *in the same frame*, so
base drift and the wrist/tip convention cancel. A sweep that stalls short is a
controller failure barred from touching the belief —
`update_swept_free` raises rather than accept it.

## 6. Four defects the runs found, and the fixes

1. **Every probe stalled short of the wall.** The base parks ~0.50 m from the
   face against a 0.40 m nominal, and a dedicated reach sweep showed the
   realized wrist origin **saturates at 0.45 m** of base-frame x across the whole
   lateral fan whatever is commanded. The press target was simply unreachable.
   Fixed by closing the loop on the *measured* standoff: the stability gate now
   also requires the base inside a measured reach band (0.30–0.43 m), with the
   residual fed back into the approach goal, re-checked before every probe.

2. **37.7 N through a parked hand.** The first version of that feedback had no
   floor, drove the goal to a 0.24 m standoff, and walked a tucked wrist into
   the wall. A tucked hand settles near **0.26 m** of base-frame x however it is
   commanded, so the commanded standoff now has a hard floor at 0.30 m.

3. **The physical turn stalled 14° short.** The goal attractor's own arrival
   latch (`yaw_tol` 0.25 rad) fires mid-turn and zeroes the yaw command. This
   executive never uses that latch — it owns `arrived` — so both `pos_tol` and
   `yaw_tol` are tightened. The policy, MPC term, action space, and force
   objective are untouched.

4. **A post-turn stabilize timeout, caught by the attribution machinery.** One
   run terminated with `controller / stabilize_timeout:stable` at 89.2 s: the
   gate condition was momentarily true but never held 0.6 s within the 15 s
   budget. Measured settle times are 0.6–3.4 s at every stance, but the
   post-turn settle is heavy-tailed — usually ~3 s, occasionally past 15 s while
   the policy takes recovery steps. The timeout had no headroom over its own
   distribution and is now 40 s. The gate itself was **not** loosened. This is
   the one real failure the pipeline produced, and it named the right owner.

## 7. Honest limitations

- **Episodes are not bit-reproducible.** Same seed, same config, same code: two
  runs of the mirrored scene diverged after ~99 s and ended 5° apart in heading,
  and the primary scene passed, then failed on the stabilize timeout, then
  passed. Every number here is one sample. The thin margins below should be read
  with that in mind.
- **A parked base is not a still base.** Peak displacement during a mode that
  claims to be parked is **18.8 cm** (11.3 cm during `PROBE`); the 90° turn
  translates the base **24.2 cm**. The commanded twist is exactly zero
  throughout — this is the frozen policy's standing behaviour, not a command. It
  is why the standoff loop exists and why the runs need re-approaches.
- **Re-approaches and the shielded press dominate the clock.** 112 s for one
  corner, most of it in the 10 mm/s near-surface press. That speed is the
  calibrated protection against the 51 N transient the shield exists for; it was
  not relaxed.
- **The association test is weak by construction.** The MVP floorplan has one
  junction and one dead end on the followed edge, so association is a
  two-hypothesis gate. It does verify that a bad estimate is *rejected* rather
  than silently accepted and that the turn comes from the map — not that data
  association scales.
- **Terminus σ sits just under tolerance** (0.058 m vs a 0.060 m bar). The
  schedule stops as soon as the guard is met; another station would tighten it.
- **The mirror's terminus is 69 mm long** (vs 17 mm primary) and that error
  propagates straight to the goal. Its arrival margin is real but thin.
- **Yaw agreement is ±8°** on a ±11.5° tolerance (primary −97.8° vs −90°). The
  margin is not comfortable.

## 8. Files

| what | where |
|---|---|
| package | `src/local_tactile/` — `floorplan.py`, `ate.py`, `fsm.py`, `shield.py`, `pipeline.py`, `plots.py` |
| config | `configs/intermittent_tacnav_corner_v1.json` (+ `_mirror`) |
| tests | `tests/local_tactile/` — 51 tests, numpy only |
| summary / traces | `out/intermittent_tacnav_seed11/intermittent_summary.json`, `intermittent_trace.jsonl` (50 Hz), `intermittent_force_200hz.jsonl` |
| probes, route, timeline | `intermittent_probes.json`, `intermittent_route_events.json`, `intermittent_phase_timeline.json` |
| failure attribution | `intermittent_failure_attribution.csv` |
| synchronized plot | `intermittent_tacnav_synchronized.png` |
| MuJoCo demo | `intermittent_tacnav_demo.mp4` — 960×720, 2834 frames, 113 s |
| mirrored episode | `out/intermittent_tacnav_seed11_mirror/` |

The predecessor persistent-corner Task-2 pipeline was **unmodified** (it is
not carried in this repository). The force
shield was lifted into `src/local_tactile/shield.py` with its calibrated
constants intact and its provenance in the module docstring. Nothing in the
frozen controller stack was retrained or edited.
