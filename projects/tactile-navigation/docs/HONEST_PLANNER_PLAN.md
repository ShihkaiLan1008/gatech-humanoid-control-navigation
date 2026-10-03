# Getting to a deployable planner

**Goal (owner, 2026-08-18):** a planner that can be deployed on the real G1
and completes the navigation task nearly perfectly, **with as little
hard-coded behaviour as possible**.

That last clause is the ranking rule.  A constant that happens to work in
room2 is worth less than a rule the robot derives from what it feels, even
when the constant scores better today — because the constant will not
survive the first real room.  Every subtask below is judged against it.

Reverse leg is out of scope for now (owner).

## The hard-coded things this program currently depends on

| what | where | why it is fragile |
|---|---|---|
| probe depth `wp_feeler_standoff_m = 0.50` | `MujocoConfig` | tuned to room2's measured registration bias; the bias is WALL-DEPENDENT (west face 0.12 m nearer than the map, north face not displaced), so one constant provably cannot serve both faces |
| corner extent `CORNER_REACH_M`, `CORNER_EXIT_REACH_M`, `CORNER_STATION_ROOM_M` | `scene_room2` | hand-placed geometry per room |
| station cadence `press_spacing_m` | `DemoExecCfg` | a schedule, not a decision |
| press travel margin | `MujocoConfig` | re-tuned twice as the belief quality changed |

## Subtasks, in order (revised 2026-08-18 after the arriving run)

- **S1 closed** — `room2_fwd_s0_s2g` arrives 0.044 m from the goal under the
  full honest contract.  See HW_ROBUSTNESS 27.
- **S2 closed** — the guessed approach depth is gone; contacts are
  `felt_on_approach`.
- **S2.1 — press the corner as close to the vertex as the platform allows.**
  Measured: the ATE asked for a press 0.9 m from the vertex and got 1.13 m,
  which reads on video as ordinary wall touching (owner).  The wall segments
  themselves stop short of the vertex -- 0.58 m on the approach face, 0.30 m
  on the outgoing one -- and those are MEASURED base-clearance limits, not
  arbitrary: facing one wall while sidling toward the other, the base and its
  parked hands cannot occupy the vertex.  The hand reaches 0.18 m further
  than the base, so ~0.40 m from the vertex is the platform's floor.  What is
  removable is the extra 0.32 m of station run-out I added when a station
  still meant settle-then-press; feeling needs no settle.
- **S2.2 — a wall-embedded landmark (owner's proposal).**  A half-embedded
  cylinder standing proud of a flat wall is a POINT feature -- its normal
  turns with position, so one touch constrains the along-wall axis a flat
  face never can -- and it sits on the wall-following path, so it needs no
  turn, no separate lane, and none of the vertex clearance above.  The
  codebase already models exactly this as a `bulge`.
- **the rest, unchanged:**

- ~~**S1 — close the loop.**~~ DONE. One forward run that completes: wall touches, ATE
  on both corner faces, information gate, turn, second wall, goal marker
  felt.  Blocking everything else, because nothing downstream can be
  measured against a route that never finishes.
- ~~**S2 — remove the biggest hard-coded number.**~~ DONE.  Replace the guessed probe
  depth with feeling: the probe hand rides ahead of the body and the
  approach stops when the HAND reports contact.  Tried once and reverted
  (the base target stayed at the press stance and kept driving past contact:
  134.6 N true peak, 6 aborts).  Doing it properly needs (a) the base target
  to stop AT the hand's reach, not past it, and (b) the felt contact to
  BECOME the measurement instead of being followed by a second press.
- **S3 — re-render the demo video** on the finished pipeline.
- **S4 — multi-seed forward validation** (3 seeds), reporting arrivals,
  in-band touches, aborts, true force peaks and belief error.
- **S5 — decide the controller question** with evidence rather than
  preference: whether to port the decoupled vel-force-MPC actor
  (`vforce_model_52000`, already sim2real-confirmed on hardware) into this
  backend.  Known blockers are structural, not cosmetic: two actor groups
  (519 + 478) against our single 492, renamed hand command terms, a FSM that
  deliberately drops the press-hold latch we depend on, and a ProbeShield
  that cannot install its force brake on those command objects.
- **S6 — write it down**: docs + memory, so the next session starts from the
  measurements and not from the story.

---

## S4 result — three forward seeds, honest contract

| seed | outcome | goal (true) | contacts | in band | felt | belief err med/max | corner gate | marker | aborts | TRUE peak |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | **arrived** | 0.10 | 8/13 | 7 | 8 | 0.088 / 0.135 | PASS | confirmed | 8 | 30.1 N |
| 1 | gate_failed:goal_post | 0.17 | 6/15 | 6 | 6 | 0.143 / 0.297 | PASS | never felt | 6 | 28.2 N |
| 2 | **arrived** | 0.28 | 9/10 | 7 | 6 | 0.136 / 0.190 | PASS | confirmed | 6 | 53.7 N |

**2 of 3 arrive; the third is a DECLARED failure**, which was the acceptance
rule.  Seed 1 stood 0.17 m from the goal and refused to call it arrival
because it could not feel the marker -- the behaviour the marker exists to
produce.  The corner information gate passed on evidence in all three.

Reproducible across seeds, not luck: every seed touches by feeling
(6-8 `felt_on_approach` contacts), keeps belief error under 0.30 m, and
passes the corner gate.

### The two defects the seeds expose

1. **The marker is too small for the belief.**  Radius 0.09 m spans 0.18 m of
   wall; belief error runs 0.09-0.30 m, so the hand often lands on the flat
   wall beside it and grazes at 2.9-3.9 N.  Seed 1 failed exactly this way,
   six tries running.  A real pillar or door frame is 0.15-0.25 m; 0.09 m was
   my number, not a physical one.
2. **Force is still not deployment-clean.**  6-8 aborts per run and a true
   peak of 28-54 N.  The aborts are parked hands scraping a wall the belief
   has misplaced, not the touches themselves.

## S4b — the same three seeds with the marker sized to the belief

| seed | outcome | goal (true) | contacts | in band | felt | belief err med/max | gate | marker | aborts | TRUE peak |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | **arrived** | 0.07 | 8/11 | 7 | 7 | 0.089 / 0.131 | PASS | confirmed | 6 | 30.1 N |
| 1 | **arrived** | 0.20 | 6/13 | 6 | 6 | 0.107 / 0.253 | PASS | confirmed | 6 | 40.6 N |
| 2 | **arrived** | 0.12 | 8/11 | 7 | 4 | 0.112 / 0.307 | PASS | confirmed | 6 | 47.7 N |

**3/3 arrive, 3/3 confirm the goal by touching it, 3/3 pass the corner gate
on evidence.**  The one change from S4 was the marker's radius, 0.09 -> 0.16,
chosen because the belief error is 0.09-0.31 m and a 0.18 m-wide marker
cannot be found reliably inside that.  Seed 1, which had failed six tries in
a row, now arrives.

Every run still carries 6 aborts and a 30-48 N true peak.  That is the open
item, and the newly added per-contact instrumentation (base speed and hand
tracking error at the instant of contact) exists to say whether it belongs to
the planner or to the controller.

---

## Owner TODO, 2026-08-18

1. **Fix the 30-48 N true peaks first.**
2. **Test the three vforce checkpoints anyway** — owner's reason: their EE
   tracking is more accurate.  Noting the tension honestly: our instrumented
   contacts show arm tracking error FLAT at 2.0-2.4 cm while base speed
   varies 3x and tracks the force, so better EE tracking is not predicted to
   move this number.  Worth testing regardless — n=7 on one approach style is
   thin, better arm tracking has value elsewhere, and `vforce_52000` is the
   only actor here confirmed sim2real on hardware.
3. **Prove the planner is deployable on the real G1** — every observation and
   every noise channel accounted for, nothing the hardware cannot supply.
