# Batch 3A — nominal closed-loop global tactile navigation on G1

`src/execution/` · integrates the Batch-1 global particle belief and the Batch-2
belief-aware planner with the persistent intermittent G1 pipeline. Neither of those
is edited: the Batch-1/2 suites (83 tests) and the intermittent suite (51 tests)
still pass unchanged.

---

## 1. The interface (`interface.py`)

| GLOBAL → LOCAL (`GlobalDirective`) | LOCAL → GLOBAL (`LocalReport`) |
|---|---|
| `route_edge` | `odom_delta_body`, `odom_delta_yaw` |
| `waypoint_map_xy`, `waypoint_yaw_map` | `probes: tuple[ProbeOutcome]` — realized contact points **and** realized swept-free paths, body frame |
| `station` (+ `station_map_xy`) | `ate_belief` — the local geometry belief |
| `hypotheses` (mass, s, xy) | `status` — execution and safety, with an owner |
| `probe_budget` | |
| `allowed_decisions` ⊆ {CONTINUE, TURN, PROBE, ARRIVE, ABORT} | |

**The ground-truth firewall is structural, not a promise.** `LocalReport` has no field
that could carry simulator state: a realized contact is a **body-frame** point (what
forward kinematics gives a real robot) plus the odometry increment that produced the
body frame — turning that into a world pose needs something the local side is never
given. `TruthLog` holds world state, is written by the executive and read only by the
evaluator, and `_update_global` accepts a `LocalReport` and asserts its type, so the
backend and the truth log have no path in. `assert_no_truth_leak` walks every report
and raises on any field name matching a truth vocabulary (`true_*`, `world_*`, `sim_*`,
`gt_*`, `oracle*`, `wall_pose`), so *adding* a leaking field breaks a test rather than
silently working. Seven tests cover this, including one that confirms the guard
actually fires.

**Odometry is a sensor, not a leak.** Both backends expose `odometry_increment()` — the
true relative motion plus drift noise — and separately `truth_pose()`. A relative
increment with noise *is* what an odometry sensor gives; an absolute pose is what it
does not.

**Commanding from a belief.** Locomotion targets are computed as
`world = truth ⊕ R(yaw_truth − yaw_believed)·(P_map − xy_believed)`: only the belief
determines the displacement, truth supplies the frame the controller lives in. So a
belief 0.55 m wrong about where it stands walks to a place 0.55 m from where it meant
to — which is how a wrong hypothesis becomes a wrong turn instead of being hidden.

---

## 2. The persistent loop (`executive.py`)

```
GLOBAL PLAN → LOCOMOTION → STABILIZE → ATE PROBE
            → GLOBAL BELIEF UPDATE → GLOBAL REPLAN → CONTINUE | PHYSICAL TURN
```

One environment, one `reset` (in `MujocoBackend.__init__`); nothing teleports the base,
re-places a wall, or re-initialises a stance thereafter, and `max_base_step_m` is
measured every tick so a teleport would appear in the summary.

**LOCOMOTION ⊥ CONTACT lives in the backend**, because that is where base twists and
hand poses are written. Each blocking action (`goto_map`, `stabilize`, `press`,
`turn_to_map`) runs its ticks under one `Mode`, and the ported `ExclusionAudit`
re-measures the realized twist and the realized hand force every tick. The executive
cannot violate the rule because it never writes a command.

**ATE selects the probes; the frozen controller executes them.** `LocalGeometryATE`
(unchanged, from the intermittent pipeline) picks the next fan offset at a stance; the
shielded REACH → PRESS → RETRACT sequence is ported unchanged, including the
`kiss + overlap` press target, the force threshold, and the rule that a sweep which
stopped short of the mapped face is a **controller event, not a measurement** and must
not touch either belief.

**Only realized data updates the global belief.** Analytic ray-casts against the known
map choose *where to stand* (`G1Station.discrimination`) and never produce an outcome.
Every belief update is driven by a `ProbeOutcome` built from measured force and measured
wrist pose. Contacts enter as rank-1 `ContactObs` with signature `surface` — a tactile
sensor cannot tell "this is a wall end" from one press, so the terminus is inferred from
the **contact/free pattern across the fan**, which is exactly what Batch 1's swept-free
channel encodes.

---

## 3. The scenes (`scenes.py`)

Constrained by the shipped asset: six wall boxes, fixed half-lengths
(1.0, 1.2, 1.4, 1.6, 1.1, 1.5), 0.10 m thick. A follow-on wall must start at the corner
(it would otherwise block the leg just walked), so **every leg after the first costs at
least 2·half-length + standoff** — the episode length is set by the boxes, not the route.

**`two_turns`** — three walls, **two consecutive physical turns**, 5.3 m of route,
unimodal prior. The unambiguous scenario.

**`ambiguous_near_edge`** — two walls, one turn, with a **bimodal along-wall
registration** δ ∈ {0, +0.55} m. Measured station by station
(`test_the_approach_is_aliased_and_the_near_edge_is_not`): every station on the approach
scores discrimination **0.0** — both registrations predict the same flat wall at the same
standoff, which is Batch 1's rank-1 result — while the station **behind the start**, at
wall 0's near edge, scores **1.0** (contact under one hypothesis, free under the other).

**Reach is the organizing constraint.** All fans are capped at the measured
`REACH_M = 0.45 m` and the constructor raises above it.
`test_every_station_is_within_the_measured_reach` checks every ray of every station
against the map geometry, and `test_every_probe_the_executive_actually_ran_was_reachable`
re-checks the stations episodes really visited.

---

## 4. Results

### 4.1 Logic, on the kinematic stand-in (12 seeds, ± 1 s.e.m.)

Perfect tracking, real reach limits, real ray geometry. These isolate the
**decision** behaviour from the controller; §4.2 is the physical run.

| scene | policy | arrived | wrong turn | decisions | probes | stations | path (m) | belief error (m) |
|---|---|---|---|---|---|---|---|---|
| two_turns | no-touch | 0.92 ± 0.08 | 0.00 | 2.00 / 2 | 0 | 0 | 5.62 | 0.099 ± 0.020 |
| two_turns | belief-aware | 0.75 ± 0.13 | 0.00 | 1.75 / 2 | 7.1 | 3.9 | 8.29 | 0.125 ± 0.026 |
| ambiguous_near_edge | no-touch | **0.00 ± 0.00** | 0.08 | 0.17 / 1 | 0 | 0 | 1.92 | **0.554 ± 0.009** |
| ambiguous_near_edge | belief-aware | **1.00 ± 0.00** | 0.00 | 1.00 / 1 | 4.2 | 1.2 | 2.30 | **0.055 ± 0.008** |

**The ambiguous scene is the result.** With a 50/50 prior over two registrations
0.55 m apart, the no-touch baseline commits on the prior, turns at the wrong place, and
arrives **0 / 12** times with a final belief error of 0.55 m — i.e. it ends convinced of
the wrong hypothesis. The belief-aware planner probes 4.2 times, resolves, and arrives
**12 / 12** with a 10× smaller belief error, for 0.4 m of extra path.

**On the unambiguous scene, probing is overhead and the table says so.** `two_turns`
has a good unimodal prior, so no-touch reaches the goal more often (0.92 vs 0.75) on a
shorter path. The belief-aware losses are budget exhaustion after re-probing, not wrong
turns — its wrong-turn rate is 0.00 on both scenes. That is the honest shape of the
result: confidence-gated probing buys nothing when the prior is already good enough to
commit, and everything when it is not.

Planning cost: 0.5–2.1 ms per replan.

### 4.2 The physical episode (MuJoCo / G1, frozen `hand_force_v2`)

`ambiguous_near_edge`, belief-aware, seed 0. **One env, one reset, 2368 control ticks,
47 s of simulated time, 1185 video frames** (`out/batch3a/ambiguous_belief_s0/`).

| | measured |
|---|---|
| outcome | **arrived** |
| route decisions | **1 / 1** |
| wrong turn | **no** |
| probes | 4, **all real contacts** (8.2, 7.9, 8.8, 6.9 N) |
| stations visited | 1 (`L0s4`) |
| global belief error, final | **0.034 m** |
| distance to goal | **0.083 m** |
| path length | 2.85 m |
| peak **probe** force | 8.84 N — **not** the peak force; see §4.3 |
| environments / resets | **1 / 1** |
| max base step per tick | **10.3 mm** (no teleport) |
| LOCOMOTION ⊥ CONTACT | **clean** — 0.000 m/s commanded twist while parked (after settle), 0.000 N hand force while moving |
| failures | **0** |
| planning time | 1.3 ms mean, 5.3 ms max |

**The loop, as it actually ran:**

```
 0.0 s  PLAN leg0 [PROBE]     masses=[0.52, 0.48]  σ=0.29   → resolve before the branch
 4.8 s  LOCOMOTION → arrived at station L0s4
14.5 s  PRESS +0.24 → CONTACT  8.2 N
19.8 s  PRESS +0.30 → CONTACT  7.9 N
26.1 s  PRESS +0.18 → CONTACT  8.8 N
33.0 s  PRESS +0.12 → CONTACT  6.9 N
33.0 s  PLAN leg0 [CONTINUE]  masses=[1.00]                ← touch collapsed the belief
35.7 s  PLAN leg0 [TURN]      minor=0.00  σ=0.08           ← the gate opens
43.0 s  PLAN leg1 [ARRIVE]                                  ← after the physical turn
        arrived, 0.083 m from goal
```

That sequence is the acceptance criterion "realized touch changes the global hypothesis
weights and route plan", on the real robot: the planner **refused to commit the turn**
while the belief was split 0.52/0.48, spent four physical presses, and only issued TURN
once the rival mass reached 0.00 and σ fell to 0.08 m.

**One metric was invalid on this run and is reported as such.** `turn_errors_deg` came
out 58.3°, which is impossible for an episode that completed the turn and arrived within
0.083 m — the metric compared a *belief-frame* commanded turn against a *world-frame*
realized turn, so it measured the belief's heading offset rather than the controller's
tracking. The definition is fixed in code (realized yaw change vs the angle the map
requires, both in the world frame) but **the corrected figure has not been re-measured
on hardware**; the 0.083 m arrival is the only physical-turn evidence this run supports.

### 4.3 The `two_turns` physical episode FAILED — and it is the most useful run

`two_turns`, belief-aware, seed 0. **Outcome: `planner_abort`, 1 / 2 route decisions,
0.519 m from goal, 243 s, 12 144 ticks.** Two independent defects, both invisible in the
stand-in and in the ambiguous scene:

**1. A parked hand was dragged along a wall at 35.2 N — above the 25 N safety limit.**
The exclusion audit caught it (5 255 samples from t = 53 s, `LOCOMOTION`,
`hand_contact_force_outside_contact_mode`), and `max_commanded_contact_while_moving` is
**0.0** — the executive never commanded contact; the base walked with a tucked wrist
against the wall. This is the failure the intermittent pipeline's
`min_commanded_standoff_m = 0.30` guard exists to prevent (it measured 37.7 N the same
way), and **Batch 3A does not port that guard**: the 0.06 m approach bias added in §5 is
capped against the scene's nominal standoff, not against the *realized* one, so the
realized standoff drifted toward the ~0.26 m at which a parked wrist reaches the face.

*The reported peak force was wrong, and that is the worse half of this finding.* The
summary said **8.58 N** because `peak_force_n` was fed only by the shield's substep
monitor, which peaks the **probe** hand. A parked hand crushed against a wall never
reached it. Fixed — every hand is now sampled every tick — but **the 8.84 N figure in
§4.2 carries the same defect and should be read as "peak PROBE force", not peak force.**
Neither episode's true peak force is established.

**2. The planner starved on unreachable backward stations.** On leg 1, odometry drift
split the belief 0.86 / 0.14 — above the 0.10 branch gate — so the planner correctly
refused the second turn and asked for a probe. But it selected stations *behind* the
robot (L1s10 … L1s7 at s ≈ 2.36–2.56 while the robot stood at s ≈ 2.9), and every
approach hit the 45 s locomotion timeout: four attempts, 180 s of the episode, no probe
executed, minor mass drifting 0.14 → 0.18, then `ABORT`. The station cost function ranks
by `|s_station − s_now|` with no penalty for reversing, and — most likely — the robot
could not walk at all while pinned against the wall by defect 1.

Failure attribution worked: four `controller / locomotion_timeout_to_station` and one
terminal `global_planner / no_reachable_station_can_resolve_the_branch`.

**So, plainly: two acceptance criteria are NOT met on hardware.** "One persistent
physical episode completes at least two route decisions" — the `two_turns` run reached
1 / 2. "Peak force remains below the existing safety limit" — 35.2 N was measured, above
the 25 N limit. Both hold in the stand-in and neither holds on the G1, and the gap is a
guard I did not port plus a metric that could not see its own violation.

### 4.4 Both fixes VALIDATED on hardware — `two_turns` re-run

Same scene, same seed, after the standoff guard and the reversal penalty:

| | before | after |
|---|---|---|
| outcome | `planner_abort` | **arrived** |
| route decisions | 1 / 2 | **2 / 2** |
| peak force, **all hands** | **35.2 N** (over the 25 N limit) | **8.61 N** |
| exclusion audit | **not clean** — 5 255 violating samples | **clean**, 0.00 N hand force while moving |
| failures | 5 (`controller`, `global_planner`) | **0** |
| goal distance | 0.519 m | 0.205 m |
| sim time / path | 243 s / 9.21 m | 67 s / 6.66 m |
| env / resets / max base step | 1 / 1 / 10.3 mm | 1 / 1 / 10.3 mm (3 326 ticks) |

**8.61 N is the first TRUE peak force reported for Batch 3A** — every hand, every tick.
The 8.84 N and 8.58 N figures above it are probe-hand-only and are not peak forces.

The two previously-failing acceptance criteria now hold on the G1: two route decisions
in one persistent episode, and peak force under the safety limit with a clean exclusion
audit. Removing the parked-hand collision also made the episode 3.6× faster and 2.5 m
shorter — the robot had been pinned against the wall, which is why four station
approaches timed out.

**Turn error, under the corrected metric: −19.1° and −21.2°** against the map's 90°
corners. That is a consistent physical under-turn by the frozen controller, not noise;
the belief and the arrival tolerance absorb it here, but it is real and unfixed.

### 4.5 The failure that produced those fixes

The
standoff guard caps the approach bias against the standoff the *belief* implies
(`_believed_standoff`, belief-only — no truth) rather than the scene's nominal value, so
it tracks the drift that caused the collision; the station cost charges 1.5× for
reversing. The stand-in results in §4.1 are unchanged by both (12 seeds, identical to
three decimal places) — the fixes are inert where the failures did not occur. §4.4 is
the hardware validation.

**What was not run.** The physical no-touch baseline and multi-seed physical statistics.
§4.1 covers those in the stand-in only.

---

## 5. What went wrong on the way (each is now a test)

| defect (measured) | fix |
|---|---|
| The fan pointed **backward** down the wall — a body-frame sign error, since the robot travels to its *right* while facing the wall, so `+y = left` inverted every offset. Every press landed on wall material, no probe could ever report swept-free, the terminus was unobservable and the branch gate never opened. | `G1Station.lateral_sign` / `body_offset`, derived from `cross(facing, travel)` |
| Batch 2's hypothesis **merge scale (0.60 m) exceeded this scene's ambiguity (0.55 m)**, fusing the two hypotheses into a phantom singleton — the gate then opened on a belief that was still split. | `hyp_cluster_radius_m` / `hyp_merge_s_m` made configurable (defaults unchanged, so Batch 2 is untouched); Batch 3A uses 0.25 m |
| A uniform 0.35 m station grid **stepped over the terminus-straddling window**: the fan spans 0.06–0.30 m ahead, so a stance only brackets a terminus from a 0.24 m window. The belief stayed one-sided (all-contact, then all-free), σ stuck at 0.13 m, gate never opened. | terminus-relative stations, plus `test_every_leg_terminus_is_bracketed_by_some_station` |
| `_choose_station` asked only for **discrimination**, so when two modes were 0.2 m apart nothing qualified and the planner aborted on a belief that merely needed sharpening. | a station may DISCRIMINATE rivals **or** ANCHOR the survivor by bracketing the terminus |
| The kinematic backend's collision test used **penetration** (zero everywhere outside a body), so it reported "blocked" in open floor. | signed distance against each body |

---

## 6. Figures and artefacts

| file | what it shows |
|---|---|
| `out/batch3a/ambiguous_belief_s0/episode.mp4` | the physical episode, 960×720, 47.4 s |
| `out/batch3a/ambiguous_belief_s0/episode_*.png` | the synchronized figure: trajectory by mode, believed position vs the mapped corner, rival-hypothesis mass against the branch gate, σ against the commit tolerance, and probe force against the 25 N limit — all on one clock, so the probe that opened the gate and the turn that followed line up |
| `out/batch3a/kinematic/comparison.png` | belief-aware vs no-touch on both scenes, 12 seeds |
| `out/batch3a/kinematic/kinematic_sweep.json` | every episode, every metric |
| `out/batch3a/*/episode.json` | full per-episode record incl. hypothesis trace and directives |

## 7. Reproduction

```bash
cd <repo root>

# offline tests: interface, firewall, scenes, closed loop  (no GPU, ~40 s)
PYTHONPATH=src:tactile_controller/src python3 -m pytest tests/execution -q

# the suites this batch must not break
PYTHONPATH=src python3 -m pytest tests/global_nav tests/local_tactile -q

# one physical episode, with video
PYTHONPATH=src:tactile_controller/src MUJOCO_GL=egl \
  uv run --project tactile_controller python -m execution.run_batch3a \
    --scene ambiguous_near_edge --policy belief_aware --seed 0 \
    --out out/batch3a/ambiguous_belief_s0

# the fast stand-in, many seeds
PYTHONPATH=src python3 -m execution.run_batch3a --backend kinematic \
    --scene ambiguous_near_edge --policy belief_aware --seeds 8 --out /tmp/b3a
```

---

## 8. Scope

No learning, no world model, no integrity-aware planning, no controller retraining, no
continuous wall contact, no graph auto-generation. No large slips and no global
relocalization — the belief-state machine from Batch 2 is present and wired, but this
batch never drives it out of `NORMAL`.
