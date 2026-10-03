# Batch 3A.1 — arrival, turning, and a verified MuJoCo benchmark

Closes out Batch 3A. Diagnoses and fixes the 0.205 m arrival gap and the turn
under-rotation, adds a true landmark-detour scene, and freezes the batch behind a
pre-freeze audit. 58 offline tests; Batch-1/2 (83) and intermittent (51) unchanged.

---

## 1. The 0.205 m arrival discrepancy — diagnosed

Decomposed against the map frame on the failing episode:

| component | value |
|---|---|
| along-leg | **+0.167 m** |
| cross-leg (lateral) | −0.119 m |
| final belief error | 0.136 m |
| approach tolerance in belief space | 0.120 m |

**It was not turn error.** The dominant term is along-leg, and it is almost exactly
`belief error + approach tolerance`. Leg 1 of `two_turns` is 2.98 m long and was walked
with **no anchoring probe at all** — the planner had nothing to spend a probe on there,
so odometry drift accumulated unopposed, and the executive then stopped as soon as the
*belief* said it was within 0.12 m of the goal.

**Fix — arrival is a route decision too.** Declaring arrival on a belief the planner
would refuse to *turn* on was simply inconsistent. `ARRIVE` now carries the same bar as
`TURN`: if σ_s exceeds `arrive_sigma_max_m` (0.12 m, equal to the branch gate) or the
minor hypothesis holds real mass, the planner emits a `PROBE` directive to anchor first.
The final approach also uses a tighter band (`goal_tol_m = 0.06` against a waypoint's
0.12), because on the last leg the approach band adds directly to the arrival error.

Measured on hardware, same scene and seed: **0.205 m → 0.147 m**, with the final belief
error down from 0.136 m to 0.096 m.

---

## 2. Turn under-rotation — diagnosed

Absolute world yaw, read off the tick trace of the failing episode:

| turn | required | realized | short by |
|---|---|---|---|
| 1 | +72.6° | +61.4° | **11.2°** |
| 2 | +162.6° | +151.4° | **11.2°** |

Both turns ended short by **the same 11.2°**, and `turn_yaw_tolerance_rad` was
0.20 rad = **11.46°**. The loop exited the instant it entered the tolerance band, always
approaching from the same side — *the tolerance was the error*. The post-turn settle then
drifted a further +2.7° and +5.5°, so even the in-band result was not what the robot
ended up holding.

A second, independent contributor: heading drifted **+7 to +10° on average per leg**
during strafing locomotion, so the robot entered each turn already partly rotated. That
is also why the naive metric "realized yaw change vs the map's 90°" read −19 to −21°: it
was measuring accumulated drift, not the turn.

**Fix — closed-loop turn completion on realized yaw.** Controller-side first: the drive
band is tightened to 0.045 rad. Then, because tightening a band a settle can undo is not
a fix, the turn **settles and re-verifies**: drive → `STABILIZE` → measure realized yaw →
re-enter the turn if it is outside `turn_settle_tolerance_rad` (0.09 rad), up to
`max_turn_corrections`. A turn is finished when the robot is **standing still at the
required heading, measured**. The timeout is a failure exit (`"timeout"`,
`"turn_incomplete"`) and never a completion path —
`test_turn_completion_is_verified_against_realized_yaw_not_duration` asserts there is no
clock in the outer loop at all.

Separately, `_hold_heading` re-squares the base to the leg's facing before any probe or
commitment when the IMU-measured heading error exceeds 0.12 rad.

**Measured on hardware after the fix — absolute heading error after settle:**

| turn | before | after |
|---|---|---|
| 1 | 11.2° short | **3.14°** |
| 2 | 11.2° short | **0.13°** |

---

## 3. Benchmark — 2 scenes × 2 policies × 5 seeds, real MuJoCo

Identical initial conditions and controller settings throughout: one env, one reset per
episode, frozen `hand_force_v2`, same `ExecConfig`, same `MujocoConfig`, seeds 0–4.

| scene | policy | n | success | wrong turn | peak force (N, all hands) | final yaw err (deg) | probes | episode time (s) | goal dist (m) |
|---|---|---|---|---|---|---|---|---|---|
| ambiguous_near_edge | no_touch | 5 | 0.20 ± 0.20 | 0.00 | 50.21 max | 1.66 | 0.0 | 201 | 0.625 |
| ambiguous_near_edge | belief_aware | 5 | 0.80 ± 0.20 | 0.00 | 16.11 max | 2.08 | 7.4 | 148 | 0.346 |
| two_turns | no_touch | 5 | 0.60 ± 0.24 | 0.00 | 35.44 max | 3.68 | 0.0 | 70 | 1.335 |
| two_turns | belief_aware | 5 | 0.60 ± 0.24 | 0.00 | 36.64 max | 2.31 | 6.6 | 176 | 0.734 |

- episodes: **20**, all one env / one reset
- **peak force across every executed leg: 50.21 N** (limit 25 N) — max hand force while moving 49.17 N
- exclusion audit: **VIOLATIONS: ambiguous_near_edge/no_touch/s1, ambiguous_near_edge/no_touch/s3, two_turns/belief_aware/s3, two_turns/no_touch/s4**
- post-settle yaw error: mean **2.45°**, worst **7.39°**
- total failures logged: 56

**What the table says.** On the ambiguous scene the belief-aware planner arrives
**0.80 ± 0.20** against the baseline's **0.20 ± 0.20**, and ends 0.346 m from goal against
0.625 m — touch is doing the work it was built to do. On `two_turns` both reach 0.60: the
prior is good enough to commit there, so probing buys nothing, exactly as the stand-in
predicted. Zero wrong turns anywhere. Post-settle yaw error averages **2.45°** (worst
7.39°) against the 11.2° systematic under-rotation this batch started with.

### THE SAFETY CRITERION FAILS: 50.21 N against a 25 N limit

Four of twenty episodes exceeded the limit, all by the same mechanism — a **parked** hand
loaded while the base walked (26.4, 35.4, 36.6, 50.2 N), with
`max_commanded_contact_while_moving = 0`: the executive never asked for contact, the
robot simply walked into a wall and leaned on its tucked wrist. Three of the four are
`no_touch` episodes, which never press at all.

**And the number I would have reported was 17.40 N.** The summary field quotes the
*executive's* counter, which only records forces from presses; on a no-probe episode it
read **0.00 N** while the robot pushed 50 N. This is the identical blind-metric defect
Batch 3A found and I believed fixed — the all-hands sampling went into the backend, but
the field the report quotes came from the executive. Both are now reconciled
(`peak_force_n` = max of both; the probe-only figure is kept as `peak_probe_force_n`).

**Root cause: locomotion had no force limit at all.** The shield guards presses; a
walking leg had nothing. Batch 3A's `_believed_standoff` guard caps the probe *approach
bias* and does nothing about simply walking into a wall. A `locomotion_abort_force_n`
(12 N) now ends a leg when a parked hand takes load — the wall is not going to move.

**The guard was re-validated on hardware and IT DID NOT WORK.** Re-running
`ambiguous_near_edge/no_touch/s1` with it: `force_aborts = 0` — the guard never fired —
and peak force went **up**, 26.43 N → **43.24 N**. The check was placed inside
`goto_map`, but the audit shows the load accumulating in `STABILIZE`, `TURN` and the
bump-slide loops as well, none of which `goto_map` sees. It has been moved to `_step`,
the single per-tick path every mode passes through, where it also cannot be bypassed by
adding a new motion primitive later.

**That second placement is itself unvalidated.** Two unvalidated safety fixes in a row
have now been shipped in this program and the first one measurably did nothing, so the
correct reading is: **Batch 3A.1 does not pass its safety criterion, and no force guard
in this batch has been demonstrated to work.** The batch is frozen with that stated,
not with it resolved. Re-running the four violating episodes is the first thing Batch 3B
should do, before anything else.

### A second defect the benchmark exposed: planner livelock

`ambiguous_near_edge/no_touch/s2` burned all 120 actions in **5.2 s of simulated time**,
emitting an identical `CONTINUE` directive every iteration. The belief is already within
tolerance of the waypoint, so the move returns `"arrived"` without commanding motion,
`_at_goal()` is still false, and the planner re-emits the same directive. There is no
no-forward-progress detector. It does not corrupt the results — these are baseline
episodes that are *supposed* to fail — but it means the baseline's episode-time column
(201 s on the ambiguous scene) reflects spinning, not navigating.

---

## 4. The landmark-detour scene (`detour_box`)

### Real-MuJoCo demo — measured

`detour_box`, belief-aware, seed 0, with video (`out/batch3a1/detour_demo/`):
**25 presses across 5 stations, 187 s, one env / one reset, peak probe force 9.40 N,
max base step 10.4 mm.** The robot strafed backward to the off-route box, swept it, and
drove its global belief error to **0.024 m** — on the real robot, from a 50/50 prior over
two registrations 0.55 m apart. That is the landmark detour demonstrated physically.

**It did not finish the route:** `planner_abort`, 0/1 route decisions, 2.205 m from goal,
and the exclusion audit was **not clean**. So the video shows the detour and the
disambiguation, which is what the deliverable asks of it, and does **not** show the
"return to the correct route and physically arrive" half of the scene specification.
That half is unmet, on hardware and in the stand-in (0.33 of seeds).

Built to the specification and **verified structurally by test**, not by assertion:

- **All on-route touches are aliased.** Wall 0 is long and unbroken across the whole
  approach, so every on-route station reads flat wall at the standoff under *both*
  hypotheses. `test_detour_scene_has_no_on_route_evidence_at_all` checks every on-route
  station, every ray, both hypotheses: **max discrimination 0.0**.
- **Only the off-route box disambiguates.** A free-standing box 1.00 m *behind* the
  start, protruding 0.08 m proud of the wall plane: a press reads 0.30 m against the
  wall's 0.38 m, and a miss sweeps through where the box should be.
  `test_detour_scene_is_resolvable_only_off_route`: **discrimination 1.0**, reachable,
  and `s = −1.18 m` — reaching it means strafing away from the goal.
- **Guessing is punished physically.** Committing on the wrong hypothesis turns 0.55 m
  early, while wall 0 is still alongside — the robot turns into the wall it was
  following. Asserted geometrically.
- The scene **declares** `withholds_terminus_evidence`: it deliberately has no
  terminus-bracketing station, because such a stance sits within a fan-length of the
  commit point and its evidence arrives *at* the decision rather than before it.

**Measured behaviour (stand-in, 12 seeds):** the planner **detours to the box on 83% of
seeds**, and the detour works as evidence — final belief error **0.082 m against the
no-touch baseline's 0.555 m**, a 6.8× reduction, with zero wrong turns.

**But it completes the full route on only 0.33 of seeds, and that is an honest failure.**
The belief resolves; the robot then runs out of probe budget before walking the route.
The cost is structural: the robot probes where its *tracked* hypothesis puts the box, and
when it is tracking the wrong one it stands a full 0.55 m separation away and sweeps
empty floor. The mitigation shipped — mirrored "rival" stances that ask the complementary
question — resolves the belief but costs ~24 presses. **The detour scene is delivered as
a verified benchmark and a demonstrated behaviour, not as a solved navigation task.**

Two measured facts behind that, both worth carrying forward:

1. **A thin landmark caps its own evidence.** The swept-free term saturates at the
   available penetration, and a 0.10 m box — the narrowest the six-box asset can express
   — can never offer more than its half-width, 0.05 m. At the original `sigma_free_m`
   of 0.045 that is 0.6 nats: fifteen presses moved the hypothesis masses from 0.51 to
   0.64 and decided nothing. Setting `sigma_free_m` from FK precision (0.022 m) makes
   the same 0.05 m worth 2.6 nats.
2. **Probing a landmark you cannot yet locate is circular.** To touch a 0.10 m box at a
   known map spot you must already know where you are to better than the fan's 0.24 m
   span — which is the thing the probe is for.

---

## 5. Defects found and fixed in 3A.1

| defect (measured) | fix |
|---|---|
| Turn tolerance 0.20 rad **was** the 11.2° under-rotation | drive band 0.045 rad + settle-and-verify on realized yaw |
| Heading drifted 7–10°/leg while strafing | `_hold_heading` re-squares before probes and commits |
| Arrival declared on a belief too weak to turn on | `ARRIVE` gated at the branch bar; tighter final approach |
| **Route arclength clipped at the origin**, so every position *behind* the start mapped to s = 0 — two hypotheses 0.54 m apart read as 0.00 apart and **merged into one mode at the midpoint**; the robot then stood 0.27 m off-station and all five presses missed the landmark | un-clip the first segment backward only (un-clipping the last too was measured to split end-of-route clouds and drop no-touch arrival 1.00 → 0.58) |
| Local ATE **vetoed a probe the global planner had already decided on** — seeded from a tight global σ it called the stance decided and returned nothing, so four PROBE directives produced zero presses and the gate never opened | ATE chooses *which* offset; the global planner decides *that* a probe happens — fallback candidate on veto |
| Arrival gate leaked probes into the **no-touch baseline** (0.9 probes) | baseline is structurally probe-free; asserted per scene per seed |
| Rival stances minted a fresh name each replan, so the spend cap never applied (22 presses in one place under 12 labels) | shift quantized to 0.10 m so the name is stable |

---

## 6. Pre-freeze audit

`tests/execution/test_freeze_checks.py` — 13 checks, all passing:

- **Interfaces**: `_update_global` accepts a `LocalReport` and nothing else (backend,
  dict, `None`, float all raise); every directive carries `route_edge` and
  `allowed_decisions`.
- **Truth firewall**: every report built during real runs on every scene is intercepted
  and walked; probes are body-frame only; contacts enter as `signature="surface"` with
  no map identity; `G1Station.predict` is asserted absent from the observation path, so
  an analytic prediction cannot become a probe outcome.
- **Safety across all executed legs**: every station actually visited (including
  dynamically-generated rival stances) is within the 0.45 m reach; probe budgets bind;
  every failure carries an owner from the fixed set.
- **Realized base state**: arrival is judged on the realized pose against the goal
  radius; turn completion settles and verifies on realized yaw with no clock in the
  outer loop.
- **Determinism**: every scene reproduces exactly given a seed.

---

## 7. Reproduction

```bash
cd <repo root>

# all offline tests incl. the pre-freeze audit  (no GPU, ~2 min)
PYTHONPATH=src:tactile_controller/src python3 -m pytest tests/execution -q

# the suites this batch must not break
PYTHONPATH=src python3 -m pytest tests/global_nav tests/local_tactile -q

cd <repo root>
# the benchmark: 2 scenes x 2 policies x 5 seeds  (~80 min)
for scene in two_turns ambiguous_near_edge; do
  for policy in belief_aware no_touch; do
    for seed in 0 1 2 3 4; do
      PYTHONPATH=src:tactile_controller/src MUJOCO_GL=egl \
        uv run --project tactile_controller python -m execution.run_batch3a \
        --scene $scene --policy $policy --seed $seed \
        --out out/batch3a1/bench/${scene}_${policy}_s${seed} --no-render
    done
  done
done

# the landmark-detour demo, with video
PYTHONPATH=src:tactile_controller/src MUJOCO_GL=egl \
  uv run --project tactile_controller python -m execution.run_batch3a \
    --scene detour_box --policy belief_aware --seed 0 \
    --out out/batch3a1/detour_demo
```

Frozen on branch `master`, tagged **`batch3a1-freeze`** (`git show batch3a1-freeze`), clean working tree.
