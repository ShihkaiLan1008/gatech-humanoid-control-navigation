# Batch 3A.2 — the frozen nominal global tactile-navigation baseline

Closes the safety, progress, and landmark-detour failures Batch 3A.1 froze
with open defects, and freezes the nominal baseline. Phases in order, each
its own commit: all-mode safety (`cad5b71`), planner livelock (`bffadc7`),
landmark detour (`b0fdc8b`), and the freeze-benchmark hardening the physical
runs demanded (`13c9446`, `78c6ab1`, `b390d58`), frozen at tag
**`batch3a2-freeze`**. 30-episode MuJoCo benchmark in `out/batch3a2/bench/`,
safety artifacts (injected-contact video, trace, figure) in
`out/batch3a2/safety/injected/`.

---

## 1. Phase 1 — all-mode safety

Batch 3A.1 ended with: *"no force guard in this batch has been demonstrated to
work"* — two unvalidated guard placements shipped, the first measurably inert
(re-run with it: force_aborts = 0, peak **up** 26.4 → 43.2 N), and four of
twenty benchmark episodes above the 25 N limit (26.4, 35.4, 36.6, 50.2 N), all
by the same mechanism: a **parked** hand loaded while the base moved, in modes
the old check never saw.

### 1.1 What the guard is now

- **Detection lives in the single per-tick path.** `_step` samples **every
  hand, every tick, in every mode**, and additionally at the 200 Hz physics
  substep through the shield monitor. In contact modes (PROBE) the probing
  hand is exempt — the shield regulates it against a force target — the parked
  hand never is.
- **Level-triggered, not edge-triggered.** An edge latch was measured to
  disarm the guard: a safety stop that timed out with force still high left
  the latch set, and the next TURN ground to 36 N with no trip. Any sample
  over 12 N outside the safety stop demands consumption; one *event* is
  recorded per trip cycle, so `force_aborts` counts events, not ticks.
- **Every loop that steps physics consumes the trip.** `goto_map`,
  `_turn_once`, `_stabilize_world`, both press loops and `_retract` all bail
  out through `_guard_tripped()` within one control tick. A structural test
  parses the module and fails if any current *or future* primitive calls
  `_step` without consuming the trip.
- **The response is in the backend, before control returns.** `_safety_stop`
  freezes a hold pose at entry (re-targeting the current pose each tick was
  measured letting a creeping base drag its own target 0.05 m into the wall),
  backs off 0.10 m along the **negative** of the measured hand-on-surface
  force (the first implementation used +f and ratcheted 21 → 31.8 N — the
  "retreat" walked into the wall), retracts both hands, and exits only after
  force < 2 N, hands parked, and base speed < 0.06 m/s **held 0.5 s**. A
  single zero-force sample is not "clear": 0 N was measured between 13.8 N
  and 15.2 N brushes.
- **The executive owns attribution and recovery.** Every force abort is a
  `safety`-owned failure; each grows a standoff retreat margin (0.04 m/event,
  cap 0.12 m) applied to every later approach; the third is terminal
  (`safety_abort`) — repeating aborts mean the recovery is not working.

### 1.2 Prevention: clearance and approach energy

- **Translation clearance** (belief + known map only): before any walk, the
  believed path is swept against every mapped body under **every live
  hypothesis** (the same displacement starts from each hypothesis's position);
  targets closer than 0.28 m are shifted off the wall, capped at 0.15 m. The
  0.28 m floor sits between the measured wrist-contact standoff (~0.26 m) and
  the nominal station approach (0.32 m).
- **Turn sweep clearance**: a turn sweeps the parked hands through a circle;
  below 0.32 m of believed room the executive steps away from the nearest
  body before turning.
- **The locomotion speed governor.** The 25 N limit cannot be held by
  reaction alone: a wrist **snagging** the wall while the base strafed was
  measured rising 0 → 44.6 N inside one 5 ms substep — the guard's first
  over-threshold sample was already 44.6 N. The only lever on that first
  sample is approach energy. Locomotion speed now scales with the believed
  clearance of the commanded path: full 0.35 with ≥ 0.55 m of room, floored
  at 0.12 below 0.25 m. At the scenes' own 0.38 m standoff legs walk at
  ~0.43× — the price of keeping a snag under the limit.

### 1.3 Unified metrics

`peak_force_n` (headline) = all hands, all ticks **and** all substep samples.
The exclusion audit now carries `peak_hand_force_n` fed from exactly the same
samples, and `physical_summary()` **asserts the two are equal** — a summary
that cannot see a collision the audit saw (shipped twice in this program) now
fails loudly instead of printing 8.58 N under a 35 N drag. The probe-only
figure is kept separately as `peak_probe_force_n`, a diagnostic, never the
headline.

### 1.4 The injected parked-hand contact test (measured, MuJoCo)

`execution.injected_contact_check` — deliberately walks the base at a mapped
wall with both hands parked, then turns against it. All 13 checks PASS
(`out/batch3a2/safety/injected/`, with video, synchronized trace and figure):

| check | measured |
|---|---|
| LOCOMOTION trip | fired at 21.09 N (substep sample), `goto` → `force_abort` |
| hands after abort | both parked, 0.00 N residual |
| base after abort | 0.002 m/s over the last 1 s of a 2 s watch |
| TURN trip | parallel approach clean; the 90° sweep tripped at **16.5 N in TURN mode** |
| recovery | post-abort walk-away arrives, no new events |
| peak force, whole session | **21.09 N** ≤ 25 N |
| summary vs audit peak | **equal**, 21.089 == 21.089 |

The TURN-mode trip is the exact mechanism 3A.1 measured accumulating load
outside `goto_map`'s sight.

### 1.5 The four violation episodes, re-run (freeze benchmark cells)

The four 3A.1 cells re-ran physically as part of the freeze benchmark, on the
frozen code:

| episode | 3A.1 peak (sustained lean) | 3A.2 peak | dwell > 25 N | outcome |
|---|---|---|---|---|
| ambiguous_near_edge / no_touch / s1 | 26.43 N | **0.00 N** | 0 | arrived |
| ambiguous_near_edge / no_touch / s3 | 50.21 N | **23.36 N** | 0 | arrived |
| two_turns / belief_aware / s3 | 36.64 N | **9.02 N** | 0 | stabilize_failed (controller) |
| two_turns / no_touch / s4 | 35.44 N | **0.00 N** | 0 | arrived |

All four are ≤ 25 N; summary peak equals the audit peak on every episode. The
3A.1 figures were **sustained leans** (s1's ran 5 255 audit samples); the one
remaining contact in the set (s3) is a single guard-tripped transient during
the scene's deliberate wrong-commit punishment.

### 1.5b The stance window (found by the freeze benchmark itself)

Two further mechanisms surfaced during benchmark iteration and are fixed in
the frozen code, each with its own commit:

- **Tolerance inversions**: the backend's arrival band sat below the goal
  attractor's own 0.05 m deadband (five consecutive 45 s approach timeouts,
  225 s standing still), and later `goal_tol` sat below the corrected
  backend floor. Both bands are now floored above the deadband, and probe
  stances get a **precision hop** (deadband tightened to 0.02 m for the
  final wall-ward move only).
- **The proud-face clearance hole**: the approach bias is computed against
  the wall plane, so a landmark face 0.08 m proud put the stance 0.24 m from
  what it presses — inside the parked wrist's 0.26 m envelope by
  construction; measured as a 42.5 N wrist clip while walking between box
  stances. Stations now declare `face_offset_m` and approaches subtract it.

### 1.6 What reaction alone could not do (measured, kept honest)

The first governed-less re-run of `ambiguous_near_edge/no_touch/s1` produced
force-abort events at **44.6 N and 58.2 N** (peak 62.4 N) despite a working
guard: the wrist snag develops faster than any sampling rate — 0 → 44.6 N
within 5 ms. Detection latency was not the failure; impact energy was. That
measurement is what put the speed governor in. Guard-only safety at full
locomotion speed **does not meet the 25 N criterion on this robot**.

---

## 2. Phase 2 — planner livelock

### 2.1 The mechanism, diagnosed

`ambiguous_near_edge/no_touch/s2` burned all 120 actions in 5.2 s of simulated
time on an identical zero-motion CONTINUE. Root cause: **geometric decisions
gated on the weighted MEAN over a still-bimodal belief while locomotion walked
the TRACKED mode** — 0.28 m apart at a 0.55 m ambiguity. The tracked mode
stood "arrived" at a waypoint the mean said was still ahead, so no motion was
commanded, no odometry was absorbed, the belief could not change, and the
identical directive re-issued forever. A bimodal mean is a place no particle
believes in; gating on it while walking on the mode is incoherent.

### 2.2 Two independent fixes

- **`_tracked_s()`** — distance-to-corner commits, CONTINUE steps, station
  travel costs and probe stances now use the along-leg position of the mode
  locomotion actually navigates by. Confidence still comes from the whole
  belief (minor mass + full spread). Side effect, deliberate: `no_touch`
  becomes a genuine commit-on-the-prior baseline — it walks its tracked mode
  to the believed corner and turns there, paying with **real wrong turns**
  instead of stalls, which finally exercises the benchmark's wrong-turn
  column.
- **A no-forward-progress detector** backstops any future zero-progress
  cycle: the same directive on the same belief with < 3 cm of realized
  odometry travel, three times in a row, is a detection. `belief_aware` gets
  one escalation (a forced anchor probe); a second detection without progress
  is a terminal `global_planner / no_forward_progress_livelock`. `no_touch`
  fails immediately — it structurally cannot probe its way out. All detector
  inputs are robot-side (directive, odometry, belief moments), so failure
  time and ownership stay deterministic and firewall-clean. Asserted silent
  on healthy runs (both original scenes × 4 seeds: zero events).

The distilled livelock substrate (a backend that reports "arrived" without
moving and returns zero odometry) terminates in ≤ 10 directives with the
explicit failure on both policies; 12/12 `no_touch` seeds on the ambiguous
scene finish or fail explicitly without ever touching the action cap.

---

---

## 3. Phase 3 — landmark detour

Baseline measured before the fix (stand-in, 12 seeds): arrival **0.25**, mean
probes 25.0; dominant failure `planner_abort` with the belief already resolved
(final position error 0.02–0.11 m) but the probe budget exhausted at the box —
full-fan sweeps repeated per visit — followed by zero-press station visits
grinding the spend caps. Two of the three "arrivals" never visited the box at
all: with zero discriminating evidence, resampling noise collapsed the 50/50
prior by luck.

### 3.1 Four measured defects, four fixes

1. **Full-fan visits pressed on after the answer.** Every press now ships its
   own single-outcome `LocalReport` into the global belief immediately (the
   firewall walks each one), and the fan **stops when the station's question
   is answered** — rival mass collapsed for a pure disambiguator, mass *and*
   spread for one that can also anchor.
2. **Branch probing could starve the arrival.** A reserve of 4 presses is
   held back from branch probing; exhausting the spendable budget is an
   explicit `probe_budget_exhausted_branch_unresolved` abort, never a
   zero-press station grind. Station choice now prices **expected cost**
   (travel + reversing penalty + the return leg an off-route stance commits
   to + ~2 presses at the gate-stop expectation) and **worst case** (the
   whole fan).
3. **The mirrored "rival" stances were measured harmful.** The shifted map
   point lands where BOTH hypotheses predict flat-wall contact at the same
   range: 20 presses, masses wobbling 0.55–0.77, nothing decided — while
   diverting the visits that would have converged. Hypothesis-conditioned
   stances are now **re-approaches**: a disambiguator visit that swept only
   free space reconditions the navigation reference to the rival mode, so
   the next visit to the *same map station* physically stands a whole
   hypothesis-separation away — where the rival predicts the landmark — and
   contacts.
4. **An association bug resolved the belief 0.47 m WRONG on 8 of 12 seeds.**
   `plan.candidates("surface")` returns only same-signature bodies, so an
   anonymous surface contact could not be explained by the `"box"`-signed
   landmark: the physically correct particles were *executed by their own
   contact* (nearest same-signature wall 0.3+ m away) while mass fled to the
   +0.55 cluster that explained the touch as wall 0. Every `detour_box` body
   now carries signature `"surface"` — the map must not partition bodies into
   signature classes the touch channel cannot distinguish. The scene test
   identifies the landmark geometrically (its proud face) instead.

Two supporting changes: a station whose predicted fan pattern is **mixed**
(contact and free, or a range step) counts as an along-anchor — which is what
lets a landmark revisit sharpen σ in a scene that deliberately withholds
terminus evidence (measured without it: four seeds resolved the branch to
2–3 cm and then aborted at σ = 0.128 against the 0.12 gate with 15 probes
unspent); and the disambiguator visit cap is 5 (erode → recondition →
contact → sharpen is 3–4 visits by construction) now that the reserve bounds
total spend.

### 3.2 Stand-in result (same 12 seeds as the baseline)

| | before | after |
|---|---|---|
| full route completed | 0.25 | **0.92** (11/12) |
| wrong turns | 0.00 | **0.00** |
| mean probes | 25.0 | **12.1** |
| mean final belief error | 0.319 m | **0.068 m** |
| landmark resolved (flagged at the box) | — | 0.67 |
| detour walked (arriving box seeds) | — | 3.6–4.5 m |

The scene's defining constraints are untouched and asserted: every on-route
station still discriminates 0.0, `withholds_terminus_evidence` still holds,
and `no_touch` still cannot probe. The one failing seed is a three-way
particle splinter under weak erosion evidence whose minor mass hovers just
above the 0.10 gate until the visit cap — reported, not patched around.

### 3.3 What the thin box taught (kept honest)

The free-ray channel against a 0.10 m slab is structurally weak: penetration
is capped by `half-thickness − |ray-to-axis distance|`, so an all-free visit
is worth ~0.4–1.6 nats total, not the 2.6 nats/ray the proud-face design
aimed at. Erosion alone therefore cannot decide the branch inside the visit
cap — which is *why* the recondition-to-rival move matters: it converts the
next visit into a contact test, and one or two contacts on the proud face
(range 0.30 vs the wall's 0.38) are decisive.

### 3.4 On MuJoCo (the honest half)

**The full landmark-detour route completed on the physical G1**
(`detour_box/belief_aware/s3`, `out/batch3a2/bench/`): detour →
disambiguate → return to route → turn → arrive, with 11 probes across 7
stations, 8.32 m of off-route walking, a final belief error of **0.041 m**,
peak force 15.70 N, 19 of 30 probes left in budget, in 551 s of simulated
time. That is the acceptance sequence, physically.

**It is 1 of 5 seeds, and the others fail honestly.** Seed 4 is the
env-seed-15 controller runaway (§6.3, 0.00 N). Seed 2 resolves
the branch correctly (final belief error **0.019 m**) but exhausts its
spendable budget before σ sharpens past the commit gate; seeds 0 and 1
resolve to the *wrong* mode. The mechanism is measured: the box-vs-wall
contact-range difference (0.08 m, the proud face) is only ~2.3σ of contact
noise, and the real robot's stance error (±0.065 m arrival + drift) blurs it
further — the range channel that cleanly separates the hypotheses in the
stand-in sits at the edge of physical observability. The robust channel is
the contact/free **edge pattern** across the fan, which the stance-window
fixes (§1.5b) restored but which remains seed-sensitive. The stand-in
completes 0.83; MuJoCo completes 0.20 (1/5, with one seed lost to the
controller runaway). The thin box stays the hard case, exactly as
specified, and these numbers are reported as they fell.

Per-seed (`execution.summarize_bench` emits this table):

| seed | outcome | probes | detour walked (m) | belief err (m) | budget left | time (s) |
|---|---|---|---|---|---|---|
| 0 | planner_abort (wrong-mode resolve) | 25 | 3.07 | 0.584 | 5 | 305 |
| 1 | planner_abort (wrong-mode resolve) | 25 | 3.15 | 0.530 | 5 | 291 |
| 2 | planner_abort (resolved RIGHT, σ starved) | 26 | 3.19 | **0.019** | 4 | 228 |
| 3 | **arrived — the full route** | 11 | 8.32 | **0.041** | 19 | 551 |
| 4 | time_budget (controller runaway, §6.3) | 4 | — | 1.941 | 26 | 954 |

---

---

## 4. Phase 4 — the frozen benchmark

3 scenes × 2 policies × 5 seeds, real MuJoCo, one env / one reset per
episode, frozen `hand_force_v2`, identical `ExecConfig`/`MujocoConfig`
throughout (`out/batch3a2/bench/`, tables from `execution.summarize_bench`):

| scene | policy | n | success | wrong turn | peak force (N) | force aborts | yaw err (°) | probes | time (s) | goal dist (m) |
|---|---|---|---|---|---|---|---|---|---|---|
| ambiguous_near_edge | no_touch | 5 | 1.00 | 0.20 | 23.36 max | 1 | 3.33 | 0.0 | 37 | 0.084 |
| ambiguous_near_edge | belief_aware | 5 | 0.80 | 0.00 | 19.20 max | 0 | 1.96 | 4.2 | 238 | 3.257* |
| detour_box | no_touch | 5 | 1.00 | 0.20 | 29.48 max† | 1 | 2.96 | 0.0 | 37 | 0.083 |
| detour_box | belief_aware | 5 | 0.20 | 0.00 | 18.79 max | 1 | 2.93 | 18.2 | 466 | 5.309* |
| two_turns | no_touch | 5 | 0.80 | 0.00 | 0.00 max | 0 | 3.17 | 0.0 | 59 | 0.244 |
| two_turns | belief_aware | 5 | 0.60 | 0.00 | 9.02 max | 0 | 1.92 | 6.2 | 299 | 4.462* |

- 30 episodes; **summary peak equals audit peak on every episode**; zero
  no-progress events; post-settle yaw error mean 2.78°, worst 8.48°.
- `*` the goal-distance means are polluted by the env-seed-15 runaway (see
  §6): both belief_aware seed-4 episodes walked 15–18 m off-map under a
  controller-level pathology at 0.00 N and ended `time_budget`.
- `†` **one episode exceeds 25 N**: `detour_box/no_touch/s3` at 29.48 N —
  the *no-touch ablation* committing on its coin-flip prior and colliding at
  the scene's designed wrong-commit punishment. Dwell above 25 N: **one
  control tick (0.02 s)**, guard-tripped, safety-aborted. The 3A.1
  violations this batch set out to close were multi-second 26–50 N leans;
  every remaining contact in 30 episodes is a transient of this kind, and
  **every belief-aware episode stays ≤ 19.2 N**. Preventing the ablation's
  wrong-commit collision would mean correcting a belief the ablation exists
  to leave uncorrected; it is reported, not patched around.
- **Wrong turns are finally real**: the no-touch baseline pays for guessing
  with measured wrong turns (0.20 on both ambiguous scenes) instead of the
  stalls and 50 N leans it used to dissolve into.

**What the table says.** On the ambiguous scene the belief-aware planner
resolves by touch and never turns wrong (0.80 arrival, 0 wrong turns, ≤
19.2 N) against a baseline that guesses (0.20 wrong-turn rate). two_turns
needs no touch — the prior is good enough — so probing is overhead there,
as the stand-in has always predicted. The detour scene stays the hard case
(§3.4). The no_touch arrival rates on the ambiguous scenes look high (1.00)
because the goal disc (0.22 m) forgives a 0.55 m-early turn on these short
final legs about half the time — the wrong-turn and belief-error columns,
not arrival, carry the diagnosis there.

### 4.1 Freeze checklist

- Safety: guard demonstrated (injected test, 13/13, re-validated on the
  frozen code), four violation episodes ≤ 25 N, summary == audit everywhere.
- Livelock: zero no-progress events across 30 episodes; the s2 livelock
  episode arrives in 24 s.
- Detour: one persistent G1 episode completes the full route (§3.4).
- Suites: execution (95), global_nav (83), local_tactile (51) — all green
  at the freeze commit.
- Firewall / one-env-one-reset / reach / ownership: enforced by the same
  tests as 3A.1, unchanged.

---

---

## 5. Reproduction

```bash
cd <repo root>

# all offline tests (no GPU)
PYTHONPATH=src:tactile_controller/src python3 -m pytest tests/execution -q
PYTHONPATH=src python3 -m pytest tests/global_nav tests/local_tactile -q

cd <repo root>

# the injected parked-hand contact test (MuJoCo, ~3 min, writes video+trace)
PYTHONPATH=src:tactile_controller/src MUJOCO_GL=egl \
  uv run --project tactile_controller python -m execution.injected_contact_check \
    --out out/batch3a2/safety/injected

# the full freeze benchmark: 3 scenes x 2 policies x 5 seeds (~6 h; the four
# 3A.1 violation cells are the ambiguous/no_touch s1+s3, two_turns/bel s3,
# two_turns/no_touch s4 cells of this grid)
for scene in two_turns ambiguous_near_edge detour_box; do
  for policy in belief_aware no_touch; do
    for seed in 0 1 2 3 4; do
      PYTHONPATH=src:tactile_controller/src MUJOCO_GL=egl \
        uv run --project tactile_controller python -m execution.run_batch3a \
        --scene $scene --policy $policy --seed $seed \
        --out out/batch3a2/bench/${scene}_${policy}_s${seed} --no-render
    done
  done
done

# tables + audit from the episode JSON
PYTHONPATH=src:tactile_controller/src \
  python3 -m execution.summarize_bench --root out/batch3a2/bench
```

## 6. Unresolved limitations

1. **The no-touch ablation can still breach 25 N transiently.** Its
   wrong-commit collisions are the scene's designed punishment; the guard
   bounds them to single-tick transients (worst measured 29.5 N, 0.02 s
   dwell) and safety-aborts. Holding the hard 25 N line through a
   deliberately-punished wrong commit is physically impossible without
   deleting the punishment. Belief-aware episodes stay ≤ 19.2 N.
2. **Impact peaks are physics-limited.** A wrist snag develops 0 → 44.6 N
   inside one 5 ms physics step and does not scale linearly with commanded
   speed (gait-cycle wrist oscillation dominates). The working protection is
   the lane + governor + face-offset **envelope**; the guard is a backstop
   that caps dwell, not peaks. A novel geometry outside the measured
   envelope could still spike within one step.
3. **Env-seed-15 runaway.** With `MujocoConfig.seed = 15` (benchmark seed
   4), both belief-aware seed-4 episodes marched 15–18 m on a fixed heading
   at 0.00 N contact, never arriving at any station: a frozen-controller /
   goal-attractor pathology at this initialization, deterministic,
   controller-owned, ended by the episode time budget. The no-progress
   detector does not fire because odometry travel is large. Undiagnosed
   beyond attribution.
4. **The goal leg cannot observe its own along-axis.** After the last turn,
   accumulated lateral drift becomes along-goal error that no on-leg
   evidence can reduce (flat wall, rank 1); the belief can stand "at the
   goal" 0.3 m from it. Such episodes end as explicit no-progress or
   planner aborts. A lateral anchor on the goal leg is future work (and out
   of scope here).
5. **Detour on MuJoCo completes 1/4 seeds** (§3.4): the proud-face range
   channel sits at the edge of physical observability, and two seeds resolve
   to the wrong mode. The stand-in (0.83) overestimates the real
   observability; the scene is kept at its designed hardness.
6. **`stabilize_failed` at two_turns seed 3**: the stability gate times out
   at the second corner on this seed (both policies, pre-existing in 3A.1's
   runs). Controller-owned, unfixed.
7. **The turn sweep is not speed-governed** (turns run at the fixed turn
   scale; the pre-turn clearance retreat and the guard cover it — worst
   measured TURN-mode trip 16.5 N).
8. **Clearance shifting is one-axis** (away from the followed wall);
   perpendicular bodies rely on the cap + governor + guard.
9. **An episode can end `safety_abort` while physically inside the goal
   disc** (the third abort breaks the loop before the arrival check runs);
   ambiguous/no_touch/s3 ended 0.07 m from goal this way in an intermediate
   run.
10. **Rendering changes the physics trajectory, and the safety tail is
   longer than the benchmark shows.** Re-running the arriving detour cell
   (seed 3) WITH video produced a *different* episode — `render_mode`
   changes environment construction, so render and no-render runs of the
   same seed are distinct trajectory draws (MuJoCo determinism holds only
   within a rendering mode; the kinematic determinism test does not cover
   this). That re-run ended `safety_abort` at a **66.77 N** substep spike
   (per-tick trips at 25.2/19.0/26.0 N; total dwell above 25 N: 0.06 s)
   while walking between box-area stances — the same off-route slab-passing
   geometry as §1.5b, beyond the stance fixes' coverage. The run is kept at
   `out/batch3a2/detour_demo_attempt_66N/` (with video). Consequence,
   stated plainly: the 30-episode benchmark's peak-force column samples a
   distribution whose tail exceeds it; what the frozen system guarantees is
   bounded **dwell** (guard-tripped transients, three-strike abort), not a
   hard 25 N peak, and the completed full-route episode is documented by
   its synchronized figures and JSON rather than a video.

---

## 7. What 3A.2 changed, in one paragraph

Batch 3A.1 froze with an unvalidated guard, four 26–50 N sustained
collisions, a livelock, and a detour benchmark no run had ever completed.
3A.2 ships a validated all-mode force guard (injected test 13/13, trip ==
peak on the walk-in), the prevention stack that actually holds the limit
(lane, governor, face-offset, precision stances), a livelock-free planner
(tracked-mode geometry + a no-progress detector, zero events in 30
episodes), a detour planner that completes the stand-in at 0.92 and the
real robot once — with the association bug that used to execute the correct
particles fixed — and a 30-episode frozen benchmark in which every
belief-aware episode stays under 19.2 N and every remaining contact is a
guard-capped transient. The honest residue is enumerated above.
