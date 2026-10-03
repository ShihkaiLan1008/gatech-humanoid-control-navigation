# Hardware-oriented classical stack — the feasibility mismatch and its fix

Branch `hw-robustness`, from `classical-freeze-2`. No learning. PF semantics
unchanged: flat-wall contact stabilizes locally, landmark contact localizes
globally.

## 1. Physical failure audit — why a legal diversion fails on the G1

`scripts/audit_divert_physical.py` + `lane_graph/divert_audit.py` re-run r5's
diversion in MuJoCo and log, per attempt, planned-vs-realized base pose and yaw
(sampled at ARRIVAL, before the press moves the base), the surface range and
reach margin from the realized pose, the controller's own termination string,
and the outcome. Five seeds, eight attempts:

| seed | goto status | result | peak force | believed err | **realized err vs stance** | **surface in reach?** |
|---|---|---|---|---|---|---|
| 0 | arrived | invalid | **584 N** | 0.515 m | 3.386 m | **none** |
| 3 | arrived | invalid | **426 N** | 0.668 m | 3.467 m | **none** |
| 3 | arrived | invalid | **594 N** | 1.409 m | 3.303 m | **none** |
| 5 | timeout | unreachable | — | 0.224 m | 3.502 m | none |
| 5 | timeout | unreachable | — | 0.293 m | 3.551 m | none |
| 8 | timeout | unreachable | — | 0.957 m | 3.860 m | none |
| 11 | timeout | unreachable | — | 0.715 m | 3.844 m | none |
| 11 | timeout | unreachable | — | 0.800 m | 3.849 m | none |

**0 of 8 delivered a contact.**

### The mismatch, stated exactly

It is *not* an execution-error problem, and it is not the attractor deadband.
The realized base sits **3.30–3.86 m** from the planned stance in every attempt
— that is the belief's own mode separation (the 3.4 m bay pitch), not tracking
error. The robot walks the displacement its belief asks for; the belief is on
the wrong mode; so it arrives 3.4 m away in map coordinates. That is by design.

`_safe_for_every_hypothesis` **already evaluates that point**: it checks
`stance + (hyp_i − bel)` for every live hypothesis, and (3.82, −0.10) — where
the robot actually ends up — is in-room with 0.33 m clearance. The check is
correct about what it checks.

**What neither check verifies is that the PRESS is executable from there.**
`reach_from_realized` is `None` in all eight attempts: from the pose the robot
actually reaches, no mapped surface lies within 2 m along the press axis, against
a 0.43 m hand. The candidate is a legal place to *stand* and an impossible place
to *touch*.

`_separating_value` does model this — under the correct hypothesis the ray finds
nothing, and it scores contact-vs-free as separating. Two things make that
score wrong in practice:

1. **the free branch is the weak channel** (capped at `free_cap_nats` = 4, and
   measured near-inert), so the "separation" it promises is mostly unavailable;
2. **the controller has no safe "press at nothing" behaviour.** The shielded
   press advances toward `kiss + overlap` and regulates toward a force target.
   With no surface in front of it, the arm extends to its clamp and the policy
   leans the base into a reaction force that never arrives.

### The safety finding

Three of eight attempts recorded peak hand forces of **426 N, 584 N and 594 N**
against a 25 N probe limit and a 12 N parked-hand guard. These come from the
substep monitor that samples every hand every tick — the sensor this programme
added precisely because a slower one could not see the collision it was meant to
catch. A press commanded into open space is not benign: it ends in a hard strike.

That is the single most important result of this audit, and it is a *hardware
blocker*, not a planning inefficiency.

### The second failure mode

The five `timeout` attempts never arrive at all: `goto_map` exhausts its 45 s
`locomotion_timeout_s` with the believed pose still 0.22–0.96 m from the stance,
and `_divert` records the whole attempt as `unreachable`. The stance tolerance
demanded (`stance_tol_m` = 0.07 m) sits below the attractor's own arrival floor
(`attractor_pos_tol_m + 0.015` = 0.065 m) with almost no headroom, so the walk
has to close to a band the controller barely resolves — and when the target lies
across the room it does not close it inside the timeout.

### Why the kinematic stand-in cannot show any of this

It teleports to the commanded displacement exactly, its press is a raycast that
cannot strike anything, and its base never moves during a press. All three
failure mechanisms above are invisible to it — which is why the same candidates
score 6/20 true arrivals kinematically and 0/8 contacts physically.

## 2. Margin-aware reachability — and what it says about the platform

`_contact_feasible_under_error` casts the press ray from every live
hypothesis' projected stance, perturbed by the base position and yaw error the
controller actually delivers, and asks whether a mapped surface stays inside the
hand's usable band. Bounds are measured, not guessed: yaw error at arrival
<= 0.079 rad over the audited attempts, base drift during a press <= 0.014 m over
the r3 demo's 13 probes, attractor arrival floor 0.065 m, locomotion tolerance
0.09 m.

**The result is a platform statement, and it is the durable finding of this
work.** Two constraints meet and nearly exclude each other:

| constraint | band |
|---|---|
| pressable — `reach_min + e <= d <= reach - safety - e` | [0.22, 0.30] m at e = 0.07 |
| standable — `LANE_CLEAR`, the base half-width enforced everywhere else | d >= 0.28 m |
| **intersection** | **[0.285, 0.30] m — 1.5 cm** |

The frozen stance of 0.33 m sits **outside** it. At e = 0.12 m the pressable band
is **empty**. So on this platform a landmark press is only both reliable and
legal inside a 1.5 cm window of standoff, and open-loop "walk to a computed
stance and press once" cannot hold that window.

**The gate is kept, measured, and defaulted OFF.** As a pre-rejection test it
removes every candidate on all five maps — because a stance only DISCRIMINATES
when the hypotheses disagree about whether a surface is there, and the
hypothesis that predicts nothing is the informative half, not an infeasible one.
What makes that half safe is capping the press, not refusing the candidate.

## 3. Closed-loop contact acquisition

`_divert` no longer means "go to one exact stance and press once":

1. walk to the stance REGION at `acquire_walk_tol_m` = 0.12 m — above the
   attractor's own floor, because the frozen 0.07 m demand timed out 5 of 8
   physical attempts;
2. read the REALIZED pose, raycast the map from there;
3. if nothing is pressable, step along the press axis toward the robust standoff
   — bounded by `acquire_step_m`, clearance- and room-checked;
4. press with the extension **capped at the map-predicted range**
   (`bounded_press`), so a press that finds nothing returns a bounded free sweep
   instead of hunting force;
5. a miss adjusts laterally and retries, up to `acquire_max_tries`.

## 4. Physical validation on r5 — same 5 seeds, same audit

| | frozen baseline | robust stack |
|---|---|---|
| truly arrived | **0/5** | **2/5** |
| contacts delivered | 0/8 | 0/8 |
| dangerous `invalid` presses | 3 | **0** |
| valid bounded free sweeps | 0 | **4** |
| **peak hand force** | **426 / 584 / 594 N** | **0.0 N** |
| walk timeouts (`unreachable`) | 5 | 4 |

**The safety failure is eliminated** — every press now stops at the
map-predicted depth, and the three >400 N strikes are gone. The three dangerous
presses became four *valid* bounded free sweeps, and that evidence was enough to
flip seeds 0 and 3 from confidently-wrong arrivals to true ones.

**Contact delivery is still 0/8**, and four walks still time out. Landmark
acquisition on this platform remains unreliable; what changed is that failing is
now safe and sometimes still informative.

### The kinematic benchmark is the wrong instrument here

Kinematically the same change reads as a **regression**: r5 goes 6/20 -> 3/20
truly arrived, r1-r4 byte-identical, overall 71 -> 68/100. Physically it goes
0/5 -> 2/5 with the force violations removed. The stand-in teleports to the
commanded pose, its press cannot strike anything and its base never moves during
a press, so it scores exactly the failure modes this work fixes as free. Both
r5 numbers — the frozen 6/20 and this 3/20 — are kinematic artifacts.

## 5. Not delivered

Stated plainly rather than half-built:

- **Subtask 4 (wall-follow execution mode)** — not implemented. The measurement
  above says why it should be attempted next: lateral drift is what consumes the
  1.5 cm standoff window, and a wall-referenced lateral constraint is the only
  lever that attacks it without changing the arm.
- **Subtask 5 (WALL_FOLLOW / LANDMARK_ACQUIRE as planner interaction types)** —
  the acquisition half exists as behaviour but is not exposed as a planner-level
  interaction type.
- **Subtask 6 (full scene set physically)** — only r5 was validated on hardware;
  r1-r4 are kinematic only.

## 6. The missing controller capability, stated exactly

Reliable landmark acquisition needs **either**:

- base position tracking at the stance of ~+-0.05 m or better (against the
  0.065-0.09 m delivered), which would open the pressable band to ~5 cm; **or**
- more arm reach relative to the base footprint — the binding constraint is that
  `LANE_CLEAR` (0.28 m) and `reach - safety - e` (0.30 m) leave 1.5 cm between
  them.

No planner change manufactures either. Until one exists, the honest hardware
configuration is `divert=False` for global localization, with the bounded press
kept unconditionally as a safety property.

## 7. WALL_FOLLOW — attempted, and blocked by the controller's press geometry

`MujocoBackend.wall_follow_map` puts the wall-side hand in a contact window at a
5 N light target and drives the base tangentially, with guard, substep monitor
and exclusion audit live.  It is opt-in; `goto_map` is untouched.

**It never touches the wall.** Physically, r5, one 3 m wall segment: contact
coverage **0.000**, peak hand force **0.0 N** over 996 ticks.  The command trace
says why, and the number is unambiguous:

| t (s) | `kiss` | commanded x | measured wrist x | force |
|---|---|---|---|---|
| 38.0 | **13.04 m** | 12.98 | 0.242 | 0.00 |
| 47.6 | **14.89 m** | 14.83 | 0.416 | 0.00 |
| 57.2 | **16.17 m** | 16.11 | 0.442 | 0.00 |

`_kiss_x` — the controller's own solve for "where must the wrist be to touch the
followed face" — returns **13–16 m** against a true standoff of ~0.3 m, so the
command saturates at the arm clamp and the wrist stalls at its 0.444 m forward
reach with the wall never approached.

### The limitation, stated exactly

`_kiss_x` divides by `denom = u . n_s`, the projection of the robot's HEADING on
the followed face's normal, and clamps that divisor at `-0.5`.  The solve is
therefore only valid when the robot **faces** the wall (`u . n_s ~ -1`).
Travelling tangentially is precisely the case `u . n_s ~ 0`: the divisor hits its
clamp, and the returned kiss is meaningless.

Underneath that is the real constraint: **the press axis is body +x**.  The
shield, `_kiss_x`, `_face_quat` and `WallEePoseCommand` are all built to press
FORWARD at a wall the robot is squared up to.  A wall the robot is travelling
ALONG sits at body +-y, and this controller has **no lateral-reach press
geometry** to put a hand there.

**Literal continuous contact is therefore not available on this platform**, and
the safest near-continuous approximation is the one the frozen stack already
implements: face the wall, press, step along — TOUCH_FOLLOW.  What
`wall_follow_map` delivers as written is slow locomotion with an extended arm.

### What must NOT be read as a wall-follow result

The same run showed lateral error 0.243 m (odometry) against 0.071 m
(wall-follow).  That is **not** evidence for wall-follow and is not claimed as
such: contact coverage was zero, so no wall reference existed; the two arms ran
at different speeds (0.35 vs 0.18 speed scale); and the odometry arm contributed
a single end-of-segment sample against 996 per-tick samples.  The comparison is
confounded in three ways at once and is reported only to say so.

A third arm — the frozen stop-and-probe cadence over the same segment — ended in
an mjlab environment termination (**the robot fell**), which the single-env
backend surfaces as a crash rather than a scored failure.

### The capability gap, quantitatively

For a wall-referenced lateral constraint during travel this controller needs one
of:

1. **a lateral press geometry** — a hand command that reaches along body +-y with
   its own kiss solve, so the wall can be touched while the heading is
   tangential.  Today `_kiss_x` degenerates exactly there (`u . n_s -> 0`,
   divisor clamped at -0.5, kiss 13-16 m);
2. **or a strafing wall-follow** — keep the wall in front (`u . n_s ~ -1`) and
   translate sideways, which the trained command box supports only weakly (chunk
   steps are strafes and realize a fraction of the commanded speed);
3. **or continuous-contact training** — the policy is trained under
   LOCOMOTION xor CONTACT, and the audit exists because a parked hand dragged
   along a wall was measured at 35.2 N while the reporting path showed 8.58 N.

None of these is a planner change, and per the Subtask 5 stop rule none is
compensated for here.

## 8. Status against the acceptance criteria

| criterion | status |
|---|---|
| WALL_FOLLOW reduces lateral/heading drift | **not demonstrated** — zero contact; the drift numbers are confounded |
| no dangerous force events | **met** — 0.0 N peak in wall-follow; bounded press eliminated the 426/584/594 N strikes |
| planner switches WALL_FOLLOW / LANDMARK_ACQUIRE / FREE_TRANSIT | **not delivered** — blocked on a wall-follow that touches |
| landmark contact delivery improves physically | **not met** — still 0/8 contacts; failure is now safe (4 bounded free sweeps) and 0/5 -> 2/5 true arrivals |
| no scene-specific logic | met |
| all tests pass | met |

## 9. Lateral (side) contact — Subtask 1 gate FAILED

`MujocoBackend.side_contact` is a direct lateral primitive: no `_kiss_x`, no
press cadence.  `pose_command_b` is a position in the base's yaw-aligned frame,
so for a wall PARALLEL to the heading the lateral base-to-hand distance is
simply the command's **y** component; the forward `x` offset is irrelevant
because the wall is a plane running along x.  That matters because the trained
box is `pos_x (0.33, 0.53)` with `cmd_x_clamp (0.33, 0.45)` — the hand can never
sit beside the base, and for this geometry it does not need to.

Static test, r5's longest ordinary wall, robot parallel to it, wall-side hand
walked outward at a 6 N target:

| commanded standoff | realized standoff | max \|hand y\| | peak force | coverage | guard |
|---|---|---|---|---|---|
| 0.30 m | 0.187 m | 0.261 m | **0.00 N** | 0.000 | no |
| 0.27 m | 0.221 m | 0.331 m | **0.00 N** | 0.000 | no |
| 0.24 m | 0.223 m | 0.220 m | **0.00 N** | 0.000 | no |

**0 of 3 produced sustained light lateral contact.** Per the Subtask 1 stop rule
("stop if static lateral contact cannot be made reliable"), everything
downstream — locomotion-plus-contact, planner WALL_FOLLOW, the r5 re-test — is
not attempted.

### An inconsistency I could not resolve, stated rather than buried

The gate failed, but it is **not yet cleanly attributable to the controller**.
In trials 1 and 2 the measured hand offset exceeds the realized wall distance —
0.261 m of hand against a 0.187 m wall, and 0.331 m against 0.221 m — which
would put the hand 7-11 cm INSIDE the wall while the force sensor reads exactly
0.00 N.  That is not a physical result; it means one of my own measurements is
wrong.  Three candidates, in the order worth testing:

1. **hand-to-wall alignment** — `_ee_base_y` projects on the base's +y axis
   using the realized yaw; if the base did not settle at the tangential heading,
   that axis is not the wall normal and the two distances are not comparable;
2. **z height** — the hand rides `probe_hand_z_m` = 0.15 m; if the injected wall
   geometry does not span that height at this lane, the hand passes under it and
   0.00 N is correct while the lateral numbers are irrelevant;
3. **force term** — `_hand_total_n` is read on the hand being commanded, but the
   substep monitor is installed for the probe/park pair, so a hand outside that
   pair may not be sampled the way a press is.

**The honest status is therefore: the gate is failed, and the blocker is not yet
isolated between "the controller cannot" and "my primitive is mis-instrumented".**
The next diagnostic is cheap and specific — log the realized base yaw against
the lane tangent, the wall's z extent at the test lane, and the same force read
through the substep monitor rather than the per-tick term — and it should be run
before any claim about controller capability is made from this section.

What §7 established independently still stands: `_kiss_x` degenerates when
travelling tangentially (13-16 m against a ~0.3 m standoff), so the *existing*
press path cannot serve wall-follow regardless of how this resolves.

## 10. Two static checks — and the gate failure resolved

No controller, no MuJoCo: `scripts/check_wall_geometry.py` is arithmetic on the
map spec and on `place_obstacles` itself.

### CHECK 1 — heading source: **the lane polyline is wrong, by up to 44 deg**

Across r5's TOUCH_FOLLOW edges, the lane polyline's local tangent differs from
the tangent of the wall surface a press ray actually hits:

| edge | lane tangent | wall tangent | error | `u . n_lane` |
|---|---|---|---|---|
| 4, 5, 6, 11, 14 | 0 / 90 deg | 0 / 90 deg | **0.0** | 0.000 |
| 9 | -7.4 deg | 0 deg | 7.4 | -0.129 |
| 10 | 8.9 deg | 0 deg | 8.9 | 0.155 |
| 7, 12 | +-25.8 deg | 180 / 0 deg | 25.8 | -0.435 |
| 3, 8, 13, 15, 16, 18 | ... | ... | **44.0** | +-0.695 |

V3 lanes bow around features by construction, so on a bowed segment the
polyline tangent is not even perpendicular to the lane's own normal
(`u . n_lane` reaches +-0.695 where it should be 0).

**The wall-follow heading must be derived from the surface normal of the body
the press ray hits** — `wall_tangent_at()` in the check script — never from the
lane polyline. Straight segments are unaffected (error 0.0), which is why this
never showed up in the frozen TOUCH_FOLLOW cadence: that cadence faces the wall
per-station rather than travelling along it.

### CHECK 2 — mapped vs injected plane: **coincident, exactly**

All 13 r5 walls, comparing the belief's `Floorplan` polygon against the mocap
pose `place_obstacles` writes (its `th + pi/2` convention puts the box's local
+x, the thickness axis, on the wall normal):

| quantity | worst over 13 walls |
|---|---|
| centre offset | **0.000000 m** |
| normal error | **0.0000 deg** |
| half-thickness | map `THICK/2` = 0.0500 == injected `WALL_THICK_HALF` = 0.0500 |
| injected z span | [0.020, 1.520] m |

**The map-vs-physical offset hypothesis is rejected.** The mapped wall distance
IS the physical one, and the hand's ~0.90 m working height sits well inside the
wall's vertical extent.

### The residual is resolved, and it was mine

Section 9 left 0.077 m of unexplained overlap. That arithmetic was wrong: it
projected only the hand's lateral `y` onto the wall normal and ignored the
**mandatory 0.34 m forward offset** (`cmd_x_clamp` floors x at 0.33). With a
25.8 deg heading error the forward offset has a normal component of its own,
0.34 sin(25.8 deg) = 0.148 m:

| | normal component | vs wall 0.221 m |
|---|---|---|
| forward offset tilts toward the wall | 0.446 m | inside by 0.225 m |
| **forward offset tilts away** | **0.150 m** | **short by 0.071 m** |

Only the second branch is consistent with a 0.00 N reading, and it explains it
exactly: **the hand stopped ~7 cm short of the wall.** No contact was made
because none was geometrically possible at that heading.

### Consequence

The Subtask 1 gate failure is attributable to **my test's heading derivation**,
not to the controller. The G1's lateral contact capability is therefore still
**untested** — section 9's table must not be read as a controller limit. What
does stand independently is section 7: `_kiss_x` degenerates when travelling
tangentially, so the existing press path cannot serve wall-follow regardless.

Re-running the gate needs the heading taken from `wall_tangent_at()`; that run
has not been made.

## 11. Static lateral contact, corrected — gate FAILED, blocker identified

Heading now comes from the CONTACTED SURFACE (`wall_tangent_at`), not the lane
polyline.  Site: r5 edge e0, surface point (0.95, -1.85), outward normal (0, 1),
tangent (-1, 0), commanded yaw 180 deg, **lane-vs-surface residual 0.00 deg** —
so the invalid heading of 9 is gone.  Wall on the body's LEFT, left hand.
`side_contact` unchanged, `_kiss_x` unused.

| commanded standoff | realized base standoff | lateral EE reach | EE-to-wall (min) | coverage | peak force |
|---|---|---|---|---|---|
| 0.32 m | **0.388 m** | 0.222 m | **+0.136 m** | 0.000 | 0.00 N |
| 0.29 m | **0.480 m** | 0.141 m | +0.312 m | 0.000 | 0.00 N |
| 0.26 m | **0.527 m** | 0.263 m | +0.251 m | 0.000 | 0.00 N |
| 0.23 m | **0.420 m** | 0.142 m | +0.214 m | 0.000 | 0.00 N |

**0/4.  Every EE-to-wall distance is POSITIVE** (0.136-0.312 m) — the hand
stopped short every time, and no MuJoCo contact pair ever registered
(`contact_pairs` empty).  0.00 N is now the *correct* reading, and the
measurements are self-consistent: the contradiction of 9 is gone.

### The blocker, decomposed

Two independent limits bind, and neither is force control:

1. **Policy capability -- the base will not stand where it is told.**
   Commanded 0.23-0.32 m from the wall, the base realized **0.388-0.527 m**, an
   error of 0.10-0.27 m against an arrival band of 0.065-0.09 m.  `_set_goal`
   holds the FSM in STATE_FOLLOW, and a wall-FOLLOWING policy maintains its own
   trained standoff; the realized values cluster at 0.39-0.53 m regardless of
   what is commanded.  The pose target does not override it.
2. **Arm lateral reach -- 0.263 m, against 0.388 m needed.**
   The largest lateral EE offset achieved was **0.263 m** (commanded 0.256 m, so
   the arm tracked its command; this is not an EE-tracking failure).  The
   closest the base ever stood was 0.388 m.

        irreducible gap = 0.388 - 0.263 = **+0.124 m**

   The hand cannot cross it, at any standoff in the sweep.

**Not the blocker:** EE tracking (commanded 0.256 m, realized 0.263 m -- it goes
where it is told); force control (never exercised -- nothing was ever touched);
map/physics geometry (10: planes coincident to 0.000000 m).

### What low-level capability is required

Closing a 0.124 m gap needs one of:

- **a wall-follow standoff command the policy honours** -- the ability to hold
  0.25-0.30 m from the followed wall instead of its trained 0.39-0.53 m.  This
  is the larger and more tractable half: 0.124 m of the gap is base standoff,
  only ~0.03 m is arm;
- **or ~0.13 m more lateral arm reach** at the 0.15 m working height, against a
  measured 0.263 m;
- **or a compliant lateral contact mode** that lets the base approach under
  force control rather than position control, so the wall itself sets the
  standoff.

Retraining or re-commanding the base standoff is the single change that would
most likely open this gate.  None of it is a planner change, and per the stop
rule none is compensated for here.

**Subtask 2 (contact + locomotion) is therefore not attempted**: there is no
static contact to carry into motion.

## 12. Closer wall standoff — measured, calibrated, and mostly solved

The 11 gate failed on a 0.124 m gap, of which the base standoff was the larger
share.  `scripts/standoff_tracking_test.py` isolates that half: command a
descending sequence of base-to-face standoffs on a straight wall, measure what
is realized at arrival and after a settle.  No arm, no contact.

### Where 0.36-0.44 m comes from

`WallFollowSe2CommandCfg.standoff_range = (0.36, 0.44)`, documented as
*"reachable faces sit at the contact box x (0.33, 0.45); the band centers it"* —
it exists to centre the **forward** press, and it is the distribution the policy
trained on.  It feeds `_resample_follow`, which writes `pose_target_w`; our
`_set_goal` overwrites that every tick, so the band never governed our commanded
stance.  It is now settable via `MujocoConfig.wall_standoff_range` so the
controller's notion of a legal wall edge matches ours.

### The base tracks a closer standoff — with a constant bias

| commanded | realized at arrival | error | after settle |
|---|---|---|---|
| 0.44 m | 0.516 | +0.076 | 0.580 |
| 0.40 m | 0.481 | +0.081 | 0.531 |
| 0.36 m | 0.436 | +0.076 | 0.509 |
| 0.32 m | 0.391 | +0.071 | 0.429 |
| 0.28 m | 0.366 | +0.086 | 0.418 |
| 0.25 m | 0.328 | +0.078 | 0.396 |
| 0.22 m | 0.293 | +0.073 | 0.386 |
| **0.18 m** | 0.256 | +0.076 | **0.302** |
| 0.14 m | 0.219 | +0.079 | 0.391 |
| 0.10 m | **force_abort** | — | — |
| 0.06 m | **force_abort** | — | — |

Three facts, and the first is the useful one:

1. **Tracking gain is unity.** A 0.19 m change in command moved the realized
   standoff 0.188 m.  The base goes where it is told.
2. **There is a constant +0.078 m OUTWARD bias** (+0.071 to +0.086 over eight
   standoffs from 0.14 to 0.44 m, +-0.007).  It survives because it sits just
   under `locomotion_tolerance_m` = 0.09 m: `goto_map` reports `arrived` and
   never corrects it.  This is calibration, not a heuristic —
   `standoff_command_m()` inverts it.
3. **The floor is ~0.14 m commanded.**  At 0.10 m and 0.06 m the guard trips
   (`force_abort`) with the base held out at ~0.30 m.

So the answer to "make the controller hold 0.25-0.30 m" is: **command 0.17-0.22 m.**
Commanding 0.18 m held **0.302 m**, inside the requested band.

    hold 0.30 m  ->  command 0.222 m
    hold 0.28 m  ->  command 0.202 m
    hold 0.26 m  ->  command 0.182 m

### What this does to the lateral gate

    closest standoff HELD   0.296-0.302 m   (was 0.388 m)
    arm lateral reach       0.263 m
    remaining gap           +0.036 m        (was +0.124 m)

**The base share of the blocker is largely solved by calibration.**  What is
left is ~3.6 cm, and that is the arm's share, not the base's.

### Two honest caveats

- **Settle drift is not constant** (+0.038 to +0.172 m, median ~0.058) and is
  therefore reported rather than compensated.  It puts roughly +-0.05 m of
  spread on any *held* standoff, so 0.302 m is a sample and not a guarantee.
- **n = 1 per target, one wall, one seed.**  The arrival bias is well supported
  (eight consistent samples); the held values are not.  A repeat on a second
  wall and seed is cheap insurance before this is relied on.

## 13. Contact at arrival — reach solved, force regulation is the new blocker

Sequence changed to the one geometry allows: command the calibrated standoff,
measure the realized base-wall distance IMMEDIATELY, acquire contact at arrival
(one bounded corrective step if outside reach), then hold.  No free settle.
Sweep: 2 maps x 3 seeds x 3 target standoffs = **18 acquisitions**.

### The reach gap is closed

| | before (11) | now |
|---|---|---|
| trials that touched the wall | **0 / 4** | **9 / 18** |
| real MuJoCo contact pairs | 0 | **9** |
| EE-to-wall distance | +0.136 to +0.312 m (always short) | contact achieved |

Reaching at ARRIVAL (0.23-0.30 m) instead of after the free settle (0.39-0.53 m)
is what did it, together with the 0.078 m command calibration.  **The 0.124 m
gap of 11, and the 0.036 m that remained after 12, are no longer the blocker.**

### But the contact cannot be REGULATED — the gate fails 0/18

| scene | seed | target | coverage | peak N | in-band | drift | status |
|---|---|---|---|---|---|---|---|
| r1_gallery | 1 | 0.27 | 0.03 | **47.10** | 0.00 | +0.027 | force_abort |
| r1_gallery | 2 | 0.25 | 0.03 | **18.22** | 0.00 | +0.047 | force_abort |
| r5_twinbay | 1 | 0.25 | 0.03 | **14.56** | 0.00 | +0.178 | force_abort |
| r5_twinbay | 2 | 0.25 | 0.03 | 9.78 | 0.00 | +0.148 | force_abort |
| r5_twinbay | 0 | 0.25 | **0.16** | 8.07 | **0.08** | -0.015 | held |
| r5_twinbay | 1 | 0.27 | 0.09 | 4.96 | 0.00 | -0.035 | held |
| r1_gallery | 2 | 0.27 | 0.04 | 2.54 | 0.00 | +0.050 | held |

Median force is **0.00 N in every trial**: contact is a spike, not a hold.  Best
coverage 0.16, best in-band fraction 0.08, four force aborts.

### Why, quantitatively

Measured contact stiffness at the moment of touch:

    dF / dy_cmd  =  median 2283 N/m   (max 2531 N/m)

The lateral regulator advances the commanded offset at `side_slew_m` = 1.2 mm
per tick, so **one tick of overshoot is 2.7 N**.  A 4-8 N band is 4 N wide —
**about 1.5 ticks of positional resolution.**  Holding it would need ~1.75 mm of
steady-state accuracy at the hand (4 N / 2283 N/m), which is finer than either
the arm's tracking or the base's micro-drift.

That is also why the existing forward press works: its contract band is
10-25 N — **15 N wide, ~5.5 ticks** — so the same position-regulation scheme has
room.  The light-touch band is the problem, not the mechanism.

### The blocker, named

Not reach (solved), not base standoff (calibrated, 12), not EE tracking, not
geometry (10).  It is **compliance / force control**: a POSITION-commanded hand
against a 2283 N/m contact cannot hold a 4 N-wide force band.

Two ways forward, neither requiring locomotion retraining:

1. **Use the policy's existing hand-FORCE channel.**  `HandForceEePoseCommand`
   carries `force_cmd_b` / `_force_cmd_mag`, and the actor observes
   `hand_contact_force_cmd` (8,) — the checkpoint was trained with force
   commands.  Commanding a force setpoint instead of position-regulating `y` is
   the compliant approach, and it uses a channel that already exists.
2. **Widen the light-contact band.**  At 2.7 N per tick, a 5-15 N band is ~3.7
   ticks wide and within reach of the present regulator.  Cheapest experiment,
   and a 5-15 N lateral reference is still far below the 25 N limit.

### A defect in this test, not in the robot

On r1_gallery seeds 0 and 1 the walk arrived at **0.73-0.96 m** instead of
~0.25 m, and the corrective step made it worse (0.730 -> 0.766, 0.746 -> 0.871).
The step is computed from the bad arrival and re-issues a `goto` carrying the
same +0.078 m bias, so it cannot recover a gross miss.  Those four trials are
not evidence about contact; r5 and r1 seed 2 (arrivals 0.21-0.30 m) are.

## 14. 10-25 N static lateral contact — gate FAILS, and the blocker inverts

Band widened to 10-25 N with a hard abort above 25 N, targets 12/16/20 N,
2 maps x 3 seeds x 3 targets = 18 acquisitions.  **0/18.**

### Subtask 1 fix works, and it disqualifies r1 at this site

The corrective step is now ABSOLUTE — the stance is recomputed from the wall
surface with the +0.078 m bias removed, instead of a relative step off a bad
arrival that re-introduced the same bias.  It correctly refuses to pretend:
r1_gallery seeds 0 and 1 arrive at **0.728-0.875 m** and all six trials are
voided as `walk_failed` rather than "corrected" (previously 0.730 -> 0.766).
r1 seed 2 arrives at 0.262-0.271 m and is usable.  **r1's site is not reachable
from two of three spawns**, so r1 is excluded from the contact statistics until
a reachable site is chosen.

### The prediction was wrong: widening the band did not help

At 2283 N/m and 1.2 mm/tick the regulator quantum is 2.7 N, so a 15 N band
should have been ~5.5 quanta — the same headroom the working forward press has.
It made no difference: coverage never exceeded 0.26, median force was **0.00 N
in every trial**, in-band fraction <= 0.03.  Band width was not the blocker.

### The blocker: contact PUSHES THE BASE AWAY

| peak force | n | mean base drift | mean coverage |
|---|---|---|---|
| no contact | 13 | **+0.002 m** | 0.000 |
| 1-10 N | 11 | **+0.027 m** | 0.070 |
| **> 10 N** | 6 | **+0.113 m** | 0.026 |

    corr(peak force, base drift) = +0.427   (n = 30, gate3 + gate4 pooled)

A dose-response: the firmer the contact, the further the base is displaced
outward.  **The contact does not act as a lateral constraint — the reaction
displaces the base, and the base retreat breaks the contact.**  That is the
direct answer to the question this sweep was built to ask, and it is the
opposite of the hypothesis.

The numbers close on themselves: at >10 N the base is pushed **+0.113 m**, while
the arm's lateral margin after calibration is only **0.036 m** (12).  Any
contact firm enough to be a useful reference displaces the base by three times
the margin it has, so it breaks itself within a few ticks — hence coverage
<= 0.26 and a median of 0.00 N.

Why the forward press does not suffer this: pressing along body +x, the reaction
runs through the facing direction and the controller explicitly *leans into* it
(`_apply_masked_x` holds x and lets the base advance).  Laterally there is no
equivalent bracing — the SE(2) attractor holds a position target against gravity
and its own dynamics, not against a sustained external side load.

### An experimental confound I introduced, and what survives it

Force targets were run in fixed order 12 -> 16 -> 20 N within each episode, so
**target and trial order are perfectly confounded**:

| trial # | mean peak | aborts |
|---|---|---|
| 0 (12 N) | 14.06 N | **7/10** |
| 1 (16 N) | 6.89 N | 1/10 |
| 2 (20 N) | 0.51 N | 0/10 |

The monotonic decay is the push-away effect poisoning later trials in the same
episode: trial 0 contacts, the base is displaced ~0.11 m, and trials 1-2 can no
longer reach.  **No claim about force target is supportable from this sweep** —
that comparison must be re-run one target per episode.  The push-away finding
survives, because it is a dose-response across contact magnitude with a clean
zero (no contact -> +0.002 m drift) rather than an order effect.

### Subtask 5 — the remaining low-level blocker, quantitatively

Not reach (solved, 13).  Not base standoff (calibrated, 12).  Not band width
(tested here).  Not geometry (10).

**It is base impedance under lateral load.** To use wall contact as a lateral
reference the base must react a sustained side force without receding.
Measured requirement: hold station to better than the arm's **0.036 m** margin
while absorbing a 10-25 N lateral load; the current controller recedes
**0.113 m** at >10 N.

Candidate low-level capabilities, none of them planner changes:

1. **a braced/stiff stance mode** for the base under lateral load — the single
   most direct fix, and the analogue of the lean the forward press already
   performs;
2. **force-command contact** via `hand_contact_force_cmd`, which would at least
   stop the position regulator chasing a receding wall (Subtask 3, untested —
   static position-regulated contact never passed, so its precondition failed);
3. **more lateral arm reach**, so the arm can follow a receding base — needs
   ~0.15 m more to cover the measured displacement, against 0.263 m today.

**Subtask 4 (contact + locomotion) is not attempted.**  Contact cannot be held
for four seconds standing still; walking with it is not a meaningful next test.
WALL_FOLLOW is **not** physically feasible on this controller as it stands.

## 15. WALL_PARALLEL_INTERMITTENT_TOUCH — constant standoff, brief taps, and it works

Section 14 ended by saying wall-follow is not feasible on this controller.  That
conclusion was about a **sustained** lateral contact, and it stands: a >10 N
hold pushes the base out 0.113 m against a 0.036 m arm margin, so the contact
breaks itself.  What section 14 did not test is the contact as an **event**.

`MujocoBackend.wall_parallel_touch(segment, standoff, spacing)` walks a
constant-standoff lane along a wall and makes a brief tap every `spacing` metres
of travelled arc length.  It is not WALL_FOLLOW and it is not the frozen
stop-and-probe cadence: the base command is one expression that never changes
when a touch begins, ends, or fails, and the touch is over in two to five ticks.

### What is different from every previous attempt

| | 7 / 11 / 13 / 14 | this |
|---|---|---|
| heading | lane polyline, then surface normal | surface tangent (`execution.wall_lane`) |
| reach solve | `_kiss_x` (degenerates travelling tangentially) | direct body ±y, measured EE-to-face gap |
| `contact_mask` | 1 (trained press semantics) | **0** |
| force | regulated toward a target | **not regulated** — creep, latch, release |
| base under load | expected to brace | expected to keep walking its lane |

Leaving `contact_mask` at 0 is not a detail.  With it set,
`WallEePoseCommand._apply_masked_x` overwrites the forward reference with its
kiss solve every tick; on the first tick of a lateral acquisition that steps the
arm 0.19 m and the wall is struck at **37.5 N**.  With it clear the hand is an
ordinary free-space pose target inside the trained box
(`pos_x` 0.33-0.53, `pos_y` 0.05-0.40, `pos_z` 0.10-0.20), and the shield's
rate limiter owns every millimetre it moves.

### Subtask 1 — the lane, alone

`--no-taps`, r5's 8 m south wall, standoff 0.28 m commanded:

| | |
|---|---|
| travelled | 7.9 m, 35 s (0.22 m/s) |
| standoff | mean **0.266 m**, sd **0.013 m**, range [0.245, 0.293] |
| heading error vs the surface tangent | mean 4.1 deg, max 4.8 |
| peak hand force | **0.00 N** |
| falls / timeouts / aborts | 0 / 0 / 0 |

Two calibrations came out of this, and both contradict what section 12 would
have predicted:

1. **A lateral lane does not carry the +0.078 m arrival bias.**  That bias was
   measured arriving at a stance the robot FACES.  Beside a wall the command is
   realized directly, so `standoff_command_m` must NOT be applied here -- doing
   so walks the robot 8 cm closer than asked.
2. **Pure pursuit alone does not close a lateral error.**  With a 1 m lookahead
   the attractor asks for 0.013 m/s of lateral speed to correct 0.06 m, and the
   policy does not move for that: commanding 0.222 m realized 0.218, commanding
   0.300 m realized 0.240, and neither converged over 4 m -- the base simply held
   whatever standoff it arrived at.  A 0.17 m error DID close, so this is a vy
   deadband, not a broken loop.  A cross-track term on the carrot's offset
   (gain 2.5, clamped at 0.10 m) turns a small error into a command the policy
   acts on, and the residual is the 0.014 m sd above.

The 4 deg heading offset is the same 7-10 deg per leg that
`heading_hold_tolerance_rad` exists for.  Raising the attractor's yaw gain from
1.5 to 3.0 halved it (8.1 -> 4.1 deg), which is what a fixed yaw-rate deadband of
~0.2 rad/s predicts.  It is direction-dependent: travelling the other way along
the same wall the error is 0.9 deg.

### Subtask 2/3 — the tap, and the question section 14 asked

The tap machine, layered on the SAME base command:

    FREE          hand carried at body y = 0.12 m, x = 0.34, z = 0.15 -- inside
                  the trained free box, so the only visible motion per tap is a
                  sideways extension
    PREPOSITION   from 0.55 m before the station, close the MEASURED hand-to-face
                  gap to 0.050 m, slowing as it closes
    TAP           creep the last centimetre at 5 mm/s, latching on the first
                  200 Hz sample at or above 10 N
    RELEASE       retract 0.08 m at 0.60 m/s, immediately
    FREE          return to the carry pose and keep walking the same lane

Nothing in it writes a base command.  A station is only attempted from a settled
lane, and a station whose gate never opens is recorded as skipped.

**The answer to section 14's question is that a brief contact does NOT displace
the base.**  Over 31 taps on four maps:

| contact regime | base displacement |
|---|---|
| sustained >10 N (14, standing still) | **+0.113 m** mean outward |
| brief tap (here, walking) | **+0.001 m** mean, range **-0.008 to +0.009 m** |

That is two orders of magnitude, and it is the whole reason this behaviour is
available when WALL_FOLLOW is not.  The reaction impulse of a 2-5 tick contact
is simply too small to move a walking base; the displacement that killed the
hold is an integral over seconds.

### What actually sets the peak force — and it is not the hand

The tap latches on the first 200 Hz sample at or above the trigger and releases
the same tick, but the force keeps rising for the two to four ticks the arm
takes to move.  Slowing the approach should have shrunk that overshoot, and it
did not:

| creep rate | n | overshoot mean | median | p90 | max |
|---|---|---|---|---|---|
| 0.012 m/s | 44 | 8.3 N | 9.3 | 14.9 | 16.5 |
| 0.008 m/s | 42 | 6.4 N | 6.0 | 11.2 | 15.1 |
| 0.005 m/s | 31 | 8.2 N | 8.3 | 15.0 | 20.8 |

The reason is in the base, not the arm.  Measured over the six ticks before each
contact, the base is closing on the wall at **0.036-0.154 m/s** in every single
tap -- the gait's own lateral sway, ten to thirty times the hand's creep -- and

    corr(base closing speed, peak force) = +0.653   (n = 73)

The sign is not an accident: a creeping hand can only meet the wall on the
sway-IN half of the gait cycle, so every tap lands on an approaching wall.
**Overshoot is a property of the gait**, and the trigger is the only term the
primitive chooses.  It is set to 8 N so that trigger plus worst observed
overshoot stays under the 25 N limit.

Two more things were tried against it and both are reported because both
failed:

- **lowering the trigger 10 -> 8 N** moved the median peak the wrong way
  (16.0 -> 19.1 N) and produced four taps over the limit instead of one.  If the
  peak were `trigger + overshoot` this could not happen; it means the loading is
  essentially over before the release can act;
- **tracking the hand-to-wall GAP instead of creeping the hand**, so the arm
  cancels the sway, was worse still: 43.1 N peak, four force aborts, and half the
  taps missed.  Given a command rate high enough to follow the base, the hand
  becomes the fast-moving body and hits the wall itself.

So the shipped configuration is the best measured one, and the residual is
named rather than tuned away: **peak tap force on this platform is set by the
gait, not by the tap.**  Making it tighter needs either a compliant hand or a
gait-phase-aware trigger, and neither is a planner change.

### Subtask 4 — the demo, on every map

Standoff 0.28 m, spacing 1.0 m, 5 maps, 3 seeds, both hands, no scene-specific
rule anywhere.  `scripts/wall_parallel_report.py out/hw/wp_final` regenerates
this from the recorded runs.

| scene | seed | seg | hand | travelled | standoff mean+-sd | range | heading | taps | peak N |
|---|---|---|---|---|---|---|---|---|---|
| r1_gallery | 0 | 0 | right | 5.87 m | 0.271 +- 0.014 | [0.244, 0.301] | 4.0 deg | 6/6 | 19.6 |
| r1_gallery | 0 | 1 | left | 5.86 m | 0.257 +- 0.017 | [0.235, 0.336] | 0.9 deg | 6/6 | 15.9 |
| r2_lbend | 0 | 0 | right | 5.91 m | 0.272 +- 0.014 | [0.246, 0.300] | 4.0 deg | 6/6 | 23.3 |
| r2_lbend | 0 | 1 | right | 4.27 m | 0.271 +- 0.014 | [0.244, 0.298] | 4.0 deg | 4/4 | 20.4 |
| r3_tpinch | 0 | 0 | left | 4.56 m | 0.258 +- 0.019 | [0.235, 0.339] | 0.9 deg | 3/4 | 29.0 |
| r3_tpinch | 0 | 1 | right | 2.19 m | 0.271 +- 0.016 | [0.224, 0.299] | 3.8 deg | 2/2 | 20.5 |
| r4_winding | 0 | 1 | left | 2.29 m | 0.261 +- 0.018 | [0.236, 0.333] | 0.8 deg | 2/2 | 22.1 |
| r5_twinbay | 0 | 0 | right | 7.92 m | 0.269 +- 0.014 | [0.246, 0.296] | 4.1 deg | 5/5 | 24.8 |
| r5_twinbay | 0 | 1 | left | 7.88 m | 0.255 +- 0.016 | [0.235, 0.334] | 0.9 deg | 5/5 | 20.4 |
| r5_twinbay | 1 | 0 | right | 7.91 m | 0.270 +- 0.014 | [0.241, 0.296] | 3.7 deg | 5/5 | 22.9 |
| r5_twinbay | 2 | 0 | right | 7.92 m | 0.257 +- 0.016 | [0.233, 0.290] | 3.0 deg | 5/5 | 21.3 |

**11 runs, 62.6 m of wall-parallel travel, 50 taps.**

| | |
|---|---|
| taps that made a real MuJoCo contact | **50 / 50** |
| taps carrying a named per-obstacle contact pair | 48 / 50 |
| peak force | 10.2 - 29.0 N, median **16.2**, p90 21.6 |
| in the 10-25 N band | **49 / 50** |
| over the 25 N limit | **1** (29.0 N, r3 seed 0) |
| contact duration | median **0.04 s**, max 0.14 s |
| base displacement from first contact | mean **+0.0015 m**, sd 0.0039, range **-0.010 to +0.009** |
| stations skipped as off-lane | 0 |
| bounded misses (swept free, released) | 0 |
| force aborts / falls / guard trips | **0 / 0 / 0** |

The two hands behave the same: right n = 33, median 18.0 N; left n = 17,
median 14.5 N.

### Status against the acceptance criteria

| criterion | status |
|---|---|
| travels 1-2 m approximately parallel to the wall | **met** — 2.2-7.9 m per run, 62.6 m total |
| standoff visually and quantitatively stable | **met** — sd 0.014-0.019 m every run |
| multiple real lateral tactile contacts | **met** — 50/50, 48 with a named contact pair |
| each valid contact 10-25 N, never above 25 | **not fully met** — 49/50 in band, one at 29.0 N |
| contact brief, not sustained | **met** — median 0.04 s, max 0.14 s |
| never deliberately retreats after a touch | **met** — one lane command, mean displacement +0.0015 m |
| forward motion smooth | **met** — 0.20-0.22 m/s, no stop-start, no arrival deadband stalls |
| no falls, no unsafe aborts | **met** — 0 falls, 0 force aborts across 11 runs |
| works on multiple segments, no scene-specific rules | **met** — 5 maps, 11 segments, both hands, geometry-selected |
| all tests pass | **met** |

The one criterion not met is the 25 N ceiling, and section "what actually sets
the peak force" says why: it is the gait's lateral sway, it is not controlled by
any parameter of the tap, and the three things that should have fixed it were
tried and measured making it worse.  1 tap in 50 exceeded the limit, by 4 N,
for one 5 ms sample.

### Subtask 5 — the integration boundary

`wall_parallel_touch(segment, standoff_m, spacing_m)` is the primitive.  It takes
a `WallSegment` in map coordinates and a spacing, and it does NOT route to the
segment: if the base is not already on the lane it returns `not_on_lane` without
walking.  Choosing the segment, the travel direction, the hand and the stations
is `execution.wall_lane`, which is arithmetic on the map bodies the belief
already carries.

The global planner is untouched, and one limitation belongs to it: `goto_map` is
a goal attractor, so the demo runner only offers segments whose STRAIGHT approach
from the spawn is clear.  On r3 and r4 that excludes the longest walls (the
approach walked into an intervening wall: 0.00-0.03 m travelled, force abort),
and on r4 an approach with 0.212 m of clearance still ended in contact -- so the
corridor test needs 0.24 m.  Routing to the rest is the planner's job.

### One defect this work found and fixed, which is not about wall-following

`ProbeShield.install_substep_monitor` captured whatever `compute_substep`
happened to be, so every re-install STACKED a wrapper and left the previous
hand mapping live underneath.  Measured cost: a legal 15.17 N tap on the LEFT
hand raised a force abort attributed to the RIGHT hand, which was reading zero.
Every primitive that changes the probing hand re-installs the monitor
(`set_leg`, `set_press_frame`, `set_touch_hand`), so this was latent in the
frozen stack too.  The pristine hook is now remembered on the manager and an
install always wraps that; `tests/local_tactile/test_shield_install.py` pins it.

### The mode machine says what this does

`Mode.TOUCH_WALK` is in both `MOVING_MODES` and `CONTACT_MODES`, and it is the
only member of a new `OVERLAP_MODES`.  `modes_are_mutually_exclusive()` still
holds for every other mode, and `ExclusionAudit` counts the overlap in its own
fields, so no report can quote a LOCOMOTION-xor-CONTACT number that silently
includes a tap.  The parked hand keeps its 12 N guard in every mode, including
this one; the touching hand keeps the 25 N abort.

### Reproducing it

```bash
# list the segments a map offers at this standoff (no simulator)
PYTHONPATH=src python3 scripts/wall_parallel_demo.py \
    --scene r5_twinbay --standoff 0.28 --spacing 1.0 --list --out /tmp/x

# one physical run
PYTHONPATH=src:tactile_controller/src MUJOCO_GL=egl \
  tactile_controller/.venv/bin/python scripts/wall_parallel_demo.py \
    --scene r5_twinbay --seed 0 --standoff 0.28 --spacing 1.0 --segment 0 \
    --yaw-stiffness 3.0 --out out/hw/wp_final        # --no-taps for the lane alone

# the table and the acceptance roll-up above
python3 scripts/wall_parallel_report.py out/hw/wp_final

# replay a recorded run to video (physics does not re-run)
PYTHONPATH=src:tactile_controller/src MUJOCO_GL=egl \
  tactile_controller/.venv/bin/python scripts/wall_parallel_video.py \
    --states  out/hw/wp_final/states_r5_twinbay_s0_s0g0.npz \
    --episode out/hw/wp_final/wp_r5_twinbay_s0_s0g0.json \
    --out     out/hw/wp_final/video/r5_twinbay_seg0_right.mp4
```

Recorded demos, one per hand and per map:

| video | what it shows |
|---|---|
| `out/hw/wp_final/video/r5_twinbay_seg0_right.mp4` | 7.9 m of r5's south wall, right hand, 5/5 taps |
| `out/hw/wp_final/video/r5_twinbay_seg1_left.mp4` | the same wall travelled the other way, left hand, 5/5 |
| `out/hw/wp_final/video/r1_gallery_seg0_right.mp4` | r1's south wall past the entry doorway, 6/6 |

## 16. Contact-rich again — the diagnosis before the change

Section 15's base motion is what the demo wants and its arm motion is not: the
lane is held to 0.014 m but a viewer cannot see the robot touching anything.
This section measures BOTH behaviours on their own recorded runs before
changing either, because the two failures are opposite and it matters which
part of each is worth keeping.

### What the old contact-rich demo actually did

`ExecutiveV3._run_touch_follow` + `MujocoBackend.press`, measured on the shipped
r1_gallery V3 demo (`out/demo_v3/r1_gallery/`, 22 TOUCH probes):

| | old (V3 TOUCH_FOLLOW) | current (15, `wall_parallel_touch`) |
|---|---|---|
| heading vs the surface tangent | **88.6 deg** — nose to the wall | **4.0 deg** — sideways |
| standoff sd over the segment | **0.136 m** | **0.014 m** |
| per-cycle standoff excursion | **0.351 m** median (max 1.345) | -- |
| base speed during touching | 0.068 m/s, **43% of the time stopped** | 0.22 m/s, never stopped |
| arm excursion per touch | x 0.14 -> ~0.45, y to +-0.24, z 0.25 -> 0.15 | **y 0.12 -> 0.22 only** |
| probe duration | **11.6 - 15.3 s** | 1.3 - 1.9 s |

The four mechanisms, named exactly:

1. **What drove the base TOWARD the wall before a contact.**
   `ExecutiveV3._stance()`: `stance = lane_pt + back * n_wall`, where
   `back = min(ray distance over live hypotheses) - press_standoff_m (0.33)`
   clipped to [0, 0.30], followed by `_goto(stance, yaw_face)`.  The stance is a
   *different point* from the lane point, and it is closer to the wall.
2. **What drove it AWAY after the contact.**  The increment step,
   `_goto(lane_pt, yaw_face, hop_tol_m)`: the lane polyline sits at
   `scene3.STANDOFF = 0.38 m`, i.e. `back` metres further out than the stance
   the press was taken from.  Approach and retreat are the same two `goto`
   calls running once per 0.45 m increment -- that is the sawtooth.
3. **What made it stop.**  `be.stabilize(yaw_face)` before every press, and
   again in the retract-confirm loop: `StabilityGate` demands both feet loaded,
   <= 0.06 m/s and <= 0.20 rad/s held continuously for 0.60 s.  Plus the arrival
   latch of each `goto`.  Measured consequence: 43% of the touch phase standing
   still.  And `yaw_face = atan2(n_wall)` turns the robot to FACE the wall at
   every station, so the old demo strafes rather than walks.
4. **What made the contact visible.**  `press()` alone: REACH walks the hand
   from park (x 0.14, y +-0.10, z 0.25) out to the chosen fan offset
   (y +-0.24, z 0.15) at 0.12 m/s, PRESS drives x to `kiss + press_overlap`
   (~0.45 m) at 0.06 m/s, RETRACT returns it all to park.  That is ~0.3 m of
   forward swing plus ~0.14 m of lateral and ~0.10 m of vertical, taking
   **11.6-15.3 s**.  Nothing about it is subtle.

**The visibility came from the ARM, and the base oscillation bought nothing for
it.**  The 0.351 m sawtooth exists to put the base at `press_standoff_m` for a
FORWARD press; a lateral touch does not need it at all.  So the fix is not to
restore any of 1-3, and section 15's `contact_mask = 0` / no-`_kiss_x` geometry
stays exactly as it is.  What has to come back is 4, on an unchanged lane.

### Why the current tap is invisible, in one number

The hand command moves **0.122 m, in `y` only**: `x` is pinned at
`side_x_m = 0.34` and `z` at `probe_hand_z_m = 0.15` for the entire run, and the
carry pose (0.12) is barely inboard of the contact pose (0.22).  There is no
reach and no retract to see -- only a wrist drifting sideways by a hand's width.

### The fix: an arm-only cycle on an unchanged lane

`wall_parallel_touch` gains three phases and loses none of its base behaviour.
The lane command is still one expression, issued once per tick BEFORE the arm
machine runs, and no branch of the cycle writes it
(`tests/execution/test_contact_rich_arm.py` pins that):

    CARRY        hand at (0.34, 0.06, 0.15) -- inboard, clear of the wall
    REACH        swing out to y = 0.16 at 0.30 m/s          <- visible
    HOLD         keep the extended pose while the lane re-settles around it
    PREPOSITION  close the measured hand-to-wall gap to 0.055 m
    TAP          creep the last centimetre at 8 mm/s, pausing whenever the base
                 is swaying INTO the wall faster than 0.06 m/s
    RELEASE      unload to CARRY at 0.90 m/s, ending on a MEASURED gap
    RETRACT      restore the carry pose                      <- visible
    CARRY        continue, and repeat at the next arc-length station

Four of those details are not preferences; each replaced something that was
measured failing, and the failures are worth more than the fix:

1. **The arm pose moves the base.**  A properly tucked carry (0.20, 0.07, 0.24)
   is far more visible and put the realized lane at **0.290 m** where the same
   segment, seed and speed with the hand carried at the contact pose held
   **0.255 m**.  Every touch then landed while the cross-track loop was hauling
   the base back in: **5 of 8 taps over the 25 N limit, up to 40.6 N**, against
   **8 of 8 inside it at 10.2-18.4 N** for the flat arm on that same run.  The
   shipped carry therefore pins x and z to the contact pose and spends the whole
   excursion on the axis that points at the wall -- which is also the axis a
   viewer reads as reaching.
2. **A reference on the COMMAND runs ahead of the arm.**  Closing the gap from
   the commanded lateral let the reference lead by whatever the arm lagged, and
   the arm then cleared that debt at whatever speed it liked: measured **0.52 m/s
   of gap closure while the command sat still for five ticks**, ending in a
   41.5 N contact.  Every approach phase now references
   `abs(_ee_base_y(term))`, so the closing speed is bounded by the rate limit.
3. **A partial retract cannot unload a hand a drifting base is pressing.**
   Ending RELEASE on a command value gave 4-7 loaded ticks and 29-42 N whenever
   the lane happened to be converging inward.  It now ends on the measured gap.
4. **The gait closes the wall faster than the hand does.**  Section 15 measured
   `corr(base closing speed, peak force) = +0.653`; TAP now simply PAUSES while
   the base is swaying in.  It is a hold, never a retreat -- the base command is
   untouched and a tap that never finds a quiet phase ends as a bounded miss.
   Adding it took one segment from 3 over-limit taps to 0.

### A / B / C on one segment, seed and speed

r5_twinbay segment 0, seed 0, speed scale 0.22.  `scripts/wall_touch_compare.py`
regenerates the figure (`out/hw/abc/compare.png`) and this table from the
recorded runs.

| arm | heading vs tangent | standoff mean+-sd | per-cycle base excursion | contacts | peak N | visible swing |
|---|---|---|---|---|---|---|
| A old TOUCH_FOLLOW | **84.0 deg** | 0.479 +- 0.069 | 0.051 m | 3/6 | 7.1-10.9 | press, 13.3 s median |
| B flat-arm tap | 3.9 deg | **0.274 +- 0.015** | one lane | 5/7 | 12.3-18.3 | 0.130 m |
| C contact-rich | 4.2 deg | **0.269 +- 0.016** | one lane | 6/7 | 17.3-24.5 | **0.171 m** |

A is reproduced by `scripts/wall_touch_old_cadence.py`, which calls the same
unmodified `press()` from the same stations.  Two caveats on it, stated because
they flatter A: its stance/lane pair are only 0.05 m apart here, where the
belief-driven original moved **0.351 m** per cycle (measured on the shipped
demo); and its first approach recorded a **44 N** hand force, which the figure
plots rather than hides.

### Where it stands, measured over 6 runs and 4 segments

Standoff 0.28 m, spacing 0.8 m, 3 map/segment pairs plus a second seed, both
hands, 41.3 m of wall-parallel travel (`out/hw/rich_final`):

| | |
|---|---|
| lane standoff sd | **0.014-0.016 m** every run |
| heading vs the surface tangent | 0.7-4.2 deg -- sideways throughout |
| base displacement caused by a touch | mean **+0.001 m**, range -0.012 to +0.011 |
| real MuJoCo contacts | 25 of 35 scheduled (10 bounded misses, 0 skipped) |
| peak force | 8.2-27.3 N, median **11.8**; 16/25 in the 10-25 N band |
| over the 25 N limit | **3** (25.7, 26.7, 27.3) |
| contact duration | median **0.04 s**, max 0.06 s |
| visible swing / reach / unload | **0.15-0.17 m** / 0.9-2.3 s / 0.17 s |
| falls, force aborts, guard trips | **0 / 0 / 0** |

### What this cost, stated plainly

Against section 15's flat arm (11 runs, 50 taps: 49/50 in band, 0 misses), the
contact-rich arm is **more visible and less reliable**: the swing grows
0.122 -> 0.171 m and the cadence tightens 1.0 -> 0.8 m, and in exchange the miss
rate goes 0/50 -> 10/35 and the over-limit rate 1/50 -> 3/25.  Both regressions
have the same cause and it is the one section 15 already named: the base closes
on the wall faster than the hand does, and a bigger, later swing gives the gait
more of the stride to interfere with.  The phase hold recovers most of it but
trades misses for it -- a tap that waits for a quiet stride sometimes runs out
of station.

Three things were tried against the force tail and are reported because they did
not work: lowering the trigger (10 -> 8 N moved the median the WRONG way,
16.0 -> 19.1, on the flat arm), tracking the hand-to-wall gap with a rate high
enough to cancel the sway (43.1 N peak, four aborts), and walking slower
(speed 0.22 -> 0.14 collapsed the lane to 0.345 m, because the cross-track
correction scales with the commanded speed).

The honest configuration knob is the SPACING: 1.0 m restores section 15's
reliability, 0.8 m is what a contact-rich demo wants, and the difference is
measured above rather than hidden.

### Reproducing 16

```bash
# C -- the shipped contact-rich behaviour
PYTHONPATH=src:tactile_controller/src MUJOCO_GL=egl \
  tactile_controller/.venv/bin/python scripts/wall_parallel_demo.py \
    --scene r5_twinbay --seed 0 --standoff 0.28 --spacing 0.8 --segment 0 \
    --yaw-stiffness 3.0 --out out/hw/rich_final

# B -- the same code path with the pre-16 arm
... scripts/wall_parallel_demo.py --arm-style flat ...

# A -- the old approach/press/retreat cadence, same segment and stations
... scripts/wall_touch_old_cadence.py --scene r5_twinbay --seed 0 --segment 0 \
      --out out/hw/abc

python3 scripts/wall_touch_compare.py out/hw/abc      # figure + the A/B/C table
python3 scripts/wall_parallel_report.py out/hw/rich_final
```

| video | what it shows |
|---|---|
| `out/hw/abc/video/A_old_touch_follow.mp4` | face the wall, stop, press 13 s, step |
| `out/hw/abc/video/B_flat_arm_tap.mp4` | smooth lane, wrist drifting 0.13 m |
| `out/hw/abc/video/C_contact_rich.mp4` | same lane, hand reaching out 0.17 m and back |

## 17. What actually made demo_v3 readable — measured in JOINT space

Section 16 made the touch bigger in the hand's LATERAL axis and it is still hard
to see.  This section starts by identifying the demo it is trying to match,
rather than assuming, and then measures it in the space the eye actually reads.

### Which implementation produced demo_v3

`out/demo_v3/r1_gallery/episode.json` records `policy: touch_follow_v3`, seed 17
(env seed 28), with `steps_touch_follow` / `touch_coverage` / `n_inspects` in the
record -- the fields only `ExecutiveV3` writes.  So demo_v3 IS
`ExecutiveV3._run_touch_follow` + the unmodified `MujocoBackend.press`, run
through `run_v3.run_one(..., backend="mujoco")` and rendered by
`replay_v3.render`.  Section 16's diagnosis was aimed at the right code.

### The gesture, per touch, from the recorded states

`states.npz` carries full `qpos`, so the arm trajectory is recoverable exactly.
Over four TOUCH probes, the LEFT (probing) arm:

| joint | demo_v3 | section 16 behaviour | this section |
|---|---|---|---|
| shoulder pitch | **69 - 80 deg** | 4 - 6 deg | **19 - 20 deg** |
| elbow | **119 - 131 deg** | 7 deg | **44 deg** |
| wrist pitch | 48 - 49 deg | 5 deg | **31 deg** |
| shoulder roll | 15 - 27 deg | 15 - 17 deg | 14 deg |
| **waist yaw** | **4.2 deg** | 4.0 deg | 4.6 deg |

Two things fall out of that table and neither was visible in EE-command space:

1. **The torso contributes nothing.**  Waist yaw moves 4.2 deg during a demo_v3
   touch and 4.0 deg during a section-16 touch -- that is the gait, identically,
   in both.  No part of the old gesture came from turning the body.
2. **Section 16 was spending its whole excursion on shoulder ROLL**, which is
   exactly the axis that drives the hand into the wall.  Its elbow moved 7 deg.
   demo_v3's moved 125.  Making the lateral excursion bigger could never have
   fixed this, which is why 0.122 -> 0.171 m of hand travel bought so little.

Shoulder pitch and elbow are driven by the hand's FORWARD and VERTICAL position,
both orthogonal to the wall normal.  They can be swung freely at zero contact
risk -- which is the whole design rule: **make the gesture large, keep the
contact displacement small.**

The old poses, for the record: park `(0.14, +-0.10, 0.25)` -> fan offset
`(., +-0.24, 0.15)` -> press to `kiss + overlap` (x ~ 0.45) -> park, at
0.06-0.12 m/s, **11.6-15.3 s per probe**.

### The hybrid, and the ceiling it hit

CARRY is now a genuinely tucked pose `(0.16, 0.10, 0.26)` and REACH swings the
hand forward and down to the touch pose; the lateral axis still only moves the
last few centimetres.  The base lane is untouched -- one carrot command per tick,
issued before the arm machine, exactly as in 15.

Reaching all the way to the old x = 0.45 was tried first and is reported because
it FAILED for a reason worth keeping:

| touch pose | elbow swing | shoulder pitch | force |
|---|---|---|---|
| x = 0.45 (arm nearly straight) | **71 deg** | 37 deg | **3 of 5 taps at 28-38 N** |
| x = 0.45, held 2 s before the touch | 71 deg | 37 deg | 3 of 5 at 30-38 N |
| **x = 0.34 (elbow bent)** | 44 deg | 20 deg | **4 of 5 at 16-24 N** |

Holding the extended pose for 2 s of travel first changed nothing, so it is not a
lane transient: **a nearly-straight arm is a stiff arm**, and the same few
millimetres of penetration that give 10-18 N from the bent-elbow pose give
28-38 N from the extended one.  The gesture therefore ends at the compliant pose
and the touch is taken from there.  That caps the recovered swing at about a
third of demo_v3's -- six times section 16's, and bounded by arm stiffness rather
than by tuning.

### Physical gate, 6 runs / 4 segments / both hands / 41.3 m

Standoff 0.28 m, spacing 0.8 m (`out/hw/gesture_final`):

| | |
|---|---|
| lane standoff sd | **0.014-0.017 m** every run -- unchanged from 15 |
| heading vs the surface tangent | 1.5-3.2 deg, sideways throughout |
| base displacement caused by a touch | mean **+0.001 m**, range -0.005 to +0.007 |
| real MuJoCo contacts | 22, of which 20 carry a named contact pair |
| peak force | 8.6-27.5 N, median 22.8; **16/22 in the 10-25 N band** |
| over the 25 N limit | 4 (25.4-27.5), all on the two right-hand runs |
| contact duration | median **0.04 s** |
| joint swing per touch | shoulder pitch 20 deg, **elbow 44 deg**, wrist pitch 31 deg |
| falls, force aborts | **0 / 0** |

`wp_acquire_tol_m` was tightened 0.05 -> 0.03 m in the same pass, and it is doing
real work: the two LEFT-hand runs sit at a realized lane of 0.326 m (the tucked
carry shifts that direction further out than the right), so all 13 of their
stations are **recorded as skipped rather than touched from the wrong standoff**.
That is the intended behaviour of the gate and it converted what were 4-of-6
over-limit taps in the previous pass into zero taps at all.  It also means the
left-hand direction currently gets a clean lane and NO touches, which is a real
limitation and not a wash: the fix is a lane-offset compensation for the carry
pose, and it belongs to the base controller this task was told not to redesign.

### Where the visual gate stands

`out/hw/gesture/video/side_by_side_demo_v3_vs_hybrid.mp4` puts one demo_v3 press
next to the hybrid at the same wall-clock rate.  What the hybrid recovers: an
obvious tucked CARRY, a forward-and-down REACH that visibly straightens the
elbow, the touch, and a RETRACT back to the tuck -- with the base tracing one
smooth lane through all of it at 0.22 m/s, where demo_v3 stops for 43% of its
touch phase and stands nose-on to the wall.

What it does not recover is the amplitude: 44 deg of elbow against 125.  The
measured reason is in the table above -- the extra swing is only available from a
near-straight arm, and a near-straight arm doubles the contact force for the same
penetration.  Buying the rest of the gesture means a compliant hand or a
force-controlled touch, not more tuning.

## 18. Turn-to-press on a constant-standoff lane — built, not yet working

The behaviour asked for: walk the lane sideways, stop, **yaw in place** to face
the wall, run the ORIGINAL demo_v3 `press()`, retract, yaw back, continue on the
same lane.  It removes demo_v3's base approach/retreat while keeping its press
verbatim.  `wall_parallel_touch(..., touch_style="turn_press")`.

**The core invariant is implemented and holds.**  Every `_set_goal` in the
primitive now targets one of exactly two things -- `target_w`, the carrot on the
constant-standoff lane, or `hold_xy`, the pose the base already occupies -- and a
test asserts that set is not exceeded.  `_turn_in_place` commands the FROZEN
position rather than the live one, so a turn cannot walk the base off its lane a
tick at a time.  Measured on the first run that reached a station:

    standoff 0.246 -> 0.235 (turn in) -> 0.247 (press) -> 0.247 (turn out)
    NET wall-normal displacement over the whole cycle: +0.001 m

**It does not yet complete a touch, and the failure is reproducible.**  Three
identical runs (T3/T4/T5, deterministic):

| | |
|---|---|
| `turn_in` | `force_abort` |
| `press` | `sweep_stopped_short_of_mapped_face`, 0.00 N, 11.1 s |
| run status | `stalled` after 0.59 m |

Two defects were found and one was fixed:

1. **Stale abort attribution (fixed).**  A guard trip at t = 4.9 s during the
   caller's APPROACH turn stayed pending and surfaced as the in-place turn's own
   failure at t = 15 s.  `_turn_in_place` now compares the abort-event COUNT
   rather than trusting the pending flag, and the primitive consumes a stale
   pending abort at entry.  This did not change the outcome, so it was not the
   cause -- it was a reporting fault sitting on top of one.
2. **Wrong followed wall during the press (candidate, UNVERIFIED).**
   `set_press_frame` re-derives the followed-wall slot from a probe point 0.5 m
   ahead of the base along the facing.  Facing a wall 0.28-0.31 m away, that
   point is 0.2 m BEHIND the surface, so the slot can latch a different segment
   and `_kiss_x` then solves against the wrong plane -- which is exactly the
   signature `sweep_stopped_short_of_mapped_face` reports while the hand is
   commanded past the real face.  The primitive now pins the slot with
   `set_followed_wall_map(seg.surface_point(station), seg.normal)` immediately
   before the press.  **No run has exercised that line.  It is a hypothesis with
   a mechanism, not a result.**

One thing the runs did establish independently: **at a 0.25 m lane the robot
cannot turn to face the wall at all** -- the parked hand loads and the guard
fires.  demo_v3 pressed from `press_standoff_m` = 0.33 m, and the lane for this
behaviour must therefore be set to ~0.34 m ONCE, for the whole segment, rather
than changed per touch.  That is consistent with the invariant and is what T3-T5
used.

### Next step, stated exactly

Re-run `--touch-style turn_press --standoff 0.34` on r5 segment 0 and read
`taps[].reason`.  If it still returns `sweep_stopped_short_of_mapped_face`, the
followed-wall hypothesis is wrong and the next instrument is `_kiss_x` logged
per press tick against the true base-to-face distance.

## 19. Turn-to-press — instrumented, diagnosed, and made to touch

§18 shipped the cycle with the wrong diagnosis attached to it.  One instrumented
failure replaced the guess.

### The instrument

`press_snapshot()` reports, per tick, every term the kiss model depends on: the
FOLLOWED wall slot and its plane, the base pose, the true base-to-face distance,
`|heading · face normal|`, `_kiss_x`, the commanded and measured hand poses, the
hand-to-face distance, the per-obstacle contact sensors and the force.
`_turn_snapshot()` does the same for both hands during a turn.  Both are off
unless `--press-trace` is passed; `scripts/wall_press_diagnose.py` reads them.

### The hypothesis in §18 was wrong

`set_press_frame` selects the **right** wall.  Measured at the failing station:
its re-derivation moved the slot from 7 to 6, and slot 6 is on the segment's own
plane, with the base inside its extent (`along` 0.07 m of a 0.75 m half-length)
and `base_to_face` 0.294 m against a true standoff of 0.296 m.  Pinning it
changes nothing.  The pin is kept only because stating the wall costs nothing.

### What was actually wrong

Two faults, both measured, in sequence:

1. **The parked hand's realized radius exceeds the lane.**  `_park_hand`
   commands x = 0.14 m, |y| = 0.10 m — a 0.172 m radius.  Over 100 ticks of
   turning the hands sat at **0.276–0.317 m** (mean 0.290), because `pos_x`'s
   trained box starts at 0.33 and the arm has no such posture.  Rotating the
   base sweeps that radius through a circle, and the worst point is mid-turn,
   where the hand's bearing aligns with the wall normal.  Measured: the right
   hand hit `v2_obst_6` at **20.7 N, radius 0.298 m, standoff 0.296 m, 33° into
   a 90° turn**, tripping the 12 N guard.  No rotation direction avoids it — the
   other hand aligns instead.  There is no park command that shrinks it.
2. **The press then ran from a base 62° off square.**  `_kiss_x` divides by
   `(heading · n̂)` clamped at −0.5, so at the measured 0.47 it returned
   **0.565 m for a 0.325 m wall**.  The hand swept diagonally past the face's
   side into open air and reported `sweep_stopped_short_of_mapped_face` with
   0.00 N — a geometry error wearing a reach error's name.

A third fault was found in the *timestamps*: the substep monitor stamps abort
events with its own clock, so the 20.7 N trip was recorded at "t = 4.86" while
the audit put the same event at t = 14.48.  §18's "stale abort" was that, not a
stale abort.  The exclusion audit is the timestamp to trust.

### What changed

| change | why |
|---|---|
| `turn_press_min_standoff_m()` = `wp_turn_hand_radius_m` (0.32) + `wp_turn_clearance_m` | a lane below the parked hand's sweep cannot host a turn; the station is **skipped** with `standoff_below_turn_radius`, not attempted |
| press is skipped unless `turn_in == "arrived"` and `\|heading · n̂\| ≥ wp_press_face_dot_min` (0.90) | a press from an unsquared base is not a weak press, it is a press in the wrong direction |
| `max_normal_x_m = true_standoff + wp_press_cap_margin_m` | uncapped, a wrong kiss plane extends the arm until something stops it — 426, 584, 594 N in §1 |
| `hold_xy` = the LANE POINT at the station, not the measured pose | the frozen pose lets the standoff sag; the lane point is the one distance the whole run holds |
| `wp_station_pos_tol_m` = 0.03 while holding a station | the shipped attractor has `pos_tol = 0.12`, so any error under 12 cm commands *zero* translation |
| `wp_press_fan_offset_m` 0.0 → 0.18, signed by `probe_side` | y = 0 is a cross-body reach and is outside the trained `pos_y (0.05, 0.40)` box |
| capped press with force on the hand returns a CONTACT | `bounded_free_sweep` was discarding a genuine 10.5 N touch on the mapped face |
| the stall watchdog is reset after a station | a station stands still for 15–20 s; that is not a failure to walk |

### The force is set by the cap, and the cap by the geometry

At a 0.41 m face the hand reaches the surface at a command of ~0.42 m, and the
contact force then climbs **linearly with the overcommand at ≈116 N/m**:

| `wp_press_cap_margin_m` | peak force |
|---|---|
| 0.10 | 10.5 N (cap-limited, still climbing) |
| 0.16 | **12.2 N, `hit`** |

### The measured budget, and why the window is narrow

| quantity | measured |
|---|---|
| parked-hand sweep radius | 0.276–0.317 m — not commandable |
| wall-normal drift during turn-in | +0.035 … +0.078 m |
| wall-normal drift during the press | +0.015 … +0.032 m |
| arm forward reach (free air) | ee_x ≈ 0.43 m asymptote |
| lane 0.33 m ⇒ press runs at | 0.41–0.42 m |
| lane 0.296 m | turn aborts on the parked hand |

The turn's own outward drift is what makes this tight: it costs 6–8 cm of the
arm's reach, and the attractor cannot take them back because the error is inside
its 0.12 m dead zone (tightening it to 0.03 recovered 0.011 m — measured, not
assumed).  **The usable lane is ≈ 0.33 m: wide enough to turn on, close enough
that the arm still reaches the wall after the drift.**  It is reported here as a
measured window, not as a tuned constant.

### It works — 3 seeds, 8 presses, r5 segment 2

`--scene r5_twinbay --segment 2 --standoff 0.33 --spacing 0.9 --s-to 3.0
--touch-style turn_press --max-taps 3`, seeds 0/1/2, ~2.9 m of lane each:

| | |
|---|---|
| both turns arrived | **8/8** |
| real MuJoCo contact on the mapped face | **8/8** |
| peak force in 10–25 N | **8/8** — min 12.80, median 13.59, max 14.31 N |
| \|NET wall-normal displacement per cycle\| | min 0.002, median 0.009, **max 0.023 m** |
| \|heading · face normal\| at the press | 1.000 (worst 1.3° off square) |
| force aborts / falls / samples over 25 N | **0 / 0 / 0** |

One station of nine was skipped, by the acquire gate rather than by a failure:
at s = 0.60 m — the first station after the approach — the lane had converged to
0.031 m of error against a 0.030 m tolerance.  That is the gate doing its job.

### The one over-limit press, and what it cost

Before the approach rate was separated out, seed 2 station 2.40 peaked at
**31.06 N against the 25 N hard limit**.  It is reported here because it
happened, not because it survived: the press whose face sat at 0.375 m was
reached *early* in the command ramp and closed fast, and it lasted 2.9 s where
an in-band press lasts 8.  Dropping the normal-command rate from the shipped
0.06 m/s to `wp_press_speed_m_s` = 0.025 turned that same seed and station into
**13.23 N**, and no press since has exceeded 14.31 N.  This is §15's finding
again from the other axis: the peak is set by the closing speed at contact, and
the force target is not what controls it.

With the segment budget sized for the touches it schedules (a station is 15–20 s
of deliberate standing still, so a budget derived from walking speed alone times
out after the touches have already landed), seed 0 completes the full behaviour:
`arrived`, 2.90 m, **3/3 presses at 13.81 / 14.27 / 14.30 N**, both turns arrived
at every station, NET wall-normal −0.017 / −0.006 / −0.012 m, zero aborts.
Video: `out/hw/turn/turn_press_3touch_s0.mp4` (normal speed, camera tracking the
base from the room side).

## 20. The scanned demo room: route verified, planner compared, turn-press integrated

### Subtask 1 — the map and the manual route, verified

`scripts/check_demo_map_readiness.py --map scan` (16 PASS / 0 FAIL after two
harness fixes: a NumPy-2 `ptp` call, and a hand-switch scan that compared
neighbours across the hand-less transit and so never found the switch):

* geometry: 29 scan walls + C1/C2 (tape geometry, scan-confirmed) + 17 clutter
  boxes; `maps_v3.validate` clean; G1 compiled INTO the room at 54 poses along
  the route -- 0 penetrations, closest approach 0.226 m.
* start (-1.30, -0.65) and goal (13.93, 11.08) as drawn; goal clearance 0.615 m.
* the manual route: 9 segments, 27.2 m, min clearance 0.457 m, continuous,
  entirely inside declared free space -- identical to the reference figure
  (`out/demo_map/scan_reference_route.png`).
* the route passes all four landmarks (C1 apex, C2 apex, branch mouth,
  convex corner) at 0.40 m.

**One deliberate execution change, stated up front: the followed lane runs at
0.40 m, not the drawn 0.56 m.**  The turn-press cycle physically requires the
face at 0.36-0.42 m when the base turns in place (§19): the parked hands sweep
an uncommandable 0.32 m circle and the arm's forward reach asymptotes at
~0.43 m.  At 0.56 m the press cannot land without the approach step this demo
forbids.  Same walls, same order, same landmarks, same topology -- only the
offset changes, and the readiness geometry re-verifies at 0.40 m.

### Subtask 2 — the planner, run cold on the same start/goal

`build_graph3` + `astar3` on the scanned scene, planner untouched
(`out/demo_map/route_vs_planner.png`, `route_vs_planner.json`):

| | manual | planner |
|---|---|---|
| length | 27.4 m | 26.7 m |
| edge kinds | -- | 14 TOUCH_FOLLOW + 4 FREE_TRANSIT |
| C1 apex | ✓ 0.40 m | ✓ 0.49 m |
| C2 apex | ✓ 0.40 m | ✓ 0.43 m |
| branch mouth | ✓ 0.40 m | ✓ 0.56 m |
| convex corner | ✓ 0.40 m | ✓ 0.39 m |

**The planner naturally chooses the landmark-rich route.**  It follows the
same long wall, bows around both curved features, crosses the same branch
mouth, rides the same east wall and rounds the same corner -- every landmark
passed within 0.6 m, unprompted.  This is the touch-lane economics doing what
they were priced to do (`TRANSIT_PENALTY_PER_M` = 1.5: unreferenced travel
costs 1.7x wall-referenced travel).

Where it deviates, and why:

1. **The branch-mouth crossing is diagonal.**  The manual route crosses at a
   right angle and picks up the east wall at the same y; the planner's
   FREE_TRANSIT is a straight connector from where its long-wall lane ends to
   where its east-wall lane begins (~0.6 m further north), because a connector
   joins lane ENDS and the east lane's clear stretch starts there.  Net: the
   planner trades ~1.8 m of wall-referenced travel for ~1.4 m of diagonal
   transit -- allowed, since the crossing stays under `MAX_TRANSIT_M`.
2. **Its lanes sit at the `followable` envelope offset (0.56 m, bowing to
   ~0.9 m at C2)** where the manual route holds 0.40 m constant.  Same shape,
   different offset family -- the planner inherits `LANE_OFF` from maps_v3.
3. **The start join**: the planner transits from the spawn directly onto the
   lane just past C1's west chord end rather than easing onto the wall before
   C1 -- the spawn sits closer to the lane than `MIN_STRETCH` allows a lane to
   begin.

None of this is a defect to fix for the demo; the demo runs the AUTHOR's
route, and the comparison stands as evidence the planner would have found the
same corridor on its own.

### Subtasks 3-4 — the full route, and what three runs taught

The executor (`demo_map/executor.py`) compiles the manual route into gotos,
arc chains and followed walls with §19 turn-press stations, runs the V3
particle filter beside it, and aims every motion from the belief.  Three full
runs, three mechanisms:

**Run 1 — stuck at C2, and the frame bug.**  The PF diverged to 0.95 m at the
first flat-wall station and split into two phantom modes at ±0.75 m.  Cause:
the contact is MEASURED while the base faces the wall, but the observation was
folded in after the turn back and the run-out -- rotated a quarter turn, it
matched the map against both corridor walls at once, and the two modes sit
exactly where the mis-rotated point meets each.  (`executive3` never sees this
because it observes while still facing the wall.)  Fix: the tap records the
press-time pose, and the executor re-expresses `p_body`/`n_body`/free rays in
the absorb-time base frame -- the same relative-displacement composition
`goto_map` performs.  A stub-backend test pins the math.

**Run 2 (`arrived`) — the demo works end to end.**  27 m, 10 stations, 8
contacts, 0 aborts, nothing over 25 N, both turns arrived everywhere, NET
wall-normal <= 0.027 m per cycle, belief err median 0.034 m.  The map's design
story showed up in the data: the C2 press cut err 0.163 -> 0.018 and sigma
0.231 -> 0.120 (the curved feature pinning the along-axis a flat wall cannot),
and the corner cue re-pinned the east-wall run at its end (err -> 0.008).
Blemishes: C1's press was a real 14.45 N touch *classified* free -- the press
cap referenced the FSM wall slot (the wall BEHIND the cylinder, 0.64 m) so the
hand had 15 s to slide off the flank until its force read fully lateral -- and
the east-wall lane realized wide (presses reach-limited at 8.5-11.9 N).

**Run 3 — cap and standoff fixes, and the two miss mechanisms.**  Bounding
the press by the SEGMENT the caller stated (`d_faced` + 0.16) fixed C1
outright (13.98 N, in band) and every contacted press landed 12.0-14.3 N.
Two stations still missed, for measured and *different* reasons: C2's station
was displaced by the belief's along-wall error (0.22 m -- the station the
robot believed was the apex put the hand on the flank), and one east-wall
press drifted +0.08 m outward DURING the press (0.379 faced -> 0.46 pressed,
a §19 press-drift outlier) and swept free of a wall that is demonstrably
there.  Both misses were honest negative observations; both walls' realized
lanes (0.34-0.40) confirmed the 0.33 standoff target centres the press
window on every followed wall of this map.

Fixes for run 4: a bounded retry budget per wall (2) -- walls retry the same
lane point once (drift is one-off), cylinder apexes hunt ±0.28 m along the
lane (the belief error is along-wall) -- and lane reacquisition that servos
its lateral off the physically measured wall distance instead of repeating
the belief's own lateral error (run 3 lost two stations to two identical
`not_on_lane` rejections).

**Run 4 — the demo run.**  `arrived`, 0.14 m true / 0.20 m believed from the
goal, 465 s sim.  **All four intended landmarks contacted in band**: C1
13.98 N, C2 14.24 N (the apex hunt recovered it: first press at the believed
apex swept free, the +0.28 m retry landed on the cylinder), branch mouth
13.53 N, corner cue 12.34 N.  9 contacts over 11 unique stations, 8 of 9 in
10-25 N (one at 9.89), peak anywhere 14.28 N, 0 force aborts, 0 samples over
25 N, both turns arrived at all 13 attempts, NET wall-normal per cycle
<= 0.009 m.  Belief err median 0.109 m, final 0.102 m, unimodal throughout.

**The one open limitation, measured precisely.**  At R6 s≈3.5 the press was
attempted three times and swept free three times: the base pressed from
0.436-0.459 m where every hit presses from 0.39-0.41 -- the press-time
outward drift reaches +0.06-0.08 m at that stance (vs +0.02-0.03 normally)
and pushes the face past the arm's ~0.43 m reach.  It is location-systematic
on the RIGHT-hand (east) wall -- full2 and full3 each lost a different
east-wall station the same way -- and a same-spot retry repeats the same
policy reaction.  Every such miss is a safe bounded free sweep banked as a
negative observation; the touch cadence loses one maintenance press.  This is
§19's press-drift tail, not a new mechanism, and it does not touch the
landmarks: those all landed.

| run | outcome | landmarks in band | contacts in band | aborts / >25 N | belief err med | goal (true) |
|---|---|---|---|---|---|---|
| full2 | arrived | 3 of 4 | 6/8 | 0 / 0 | 0.034 m | 0.25 m |
| full3 | arrived | 3 of 4 | 6/6 | 0 / 0 | 0.184 m | 0.23 m |
| **full4** | **arrived** | **4 of 4** | 8/9 | 0 / 0 | 0.109 m | **0.14 m** |

## 21. Crab mode: the facing walk, measured, and the step-in cycle

Owner decision: the demo switches from turn-to-press to a FACE-THE-WALL demo --
sidle along the wall, press without turning, "feel the way forward" -- and the
§18-19 invariant that a touch never changes the base-wall distance is
deliberately retired for this style.  Getting there took four smoke iterations,
each of which measured something:

1. **The lateral-tap branch leaked into crab mode** (its guard tested
   `not turn_press`, and crab is not turn_press): the arm ran the side-tap
   machine while the base faced the wall, reaching ALONG the wall -- and the
   outstretched arm alone dragged the lane (§16's arm-pose effect).  Fixed by
   gating on `station_press`.
2. **The facing walk will not hold a commanded lane.**  Hands parked, entered
   at 0.39 it drifted out and held 0.48-0.50; entered at 0.49 it drifted to
   0.59 (range up to 0.64) -- with the cross-track term saturated at 0.25 m
   the whole way, i.e. a carrot up to 0.35 m INSIDE the realized lane changed
   nothing.  This is not a constant bias to feed forward (that model predicted
   0.27 from a 0.23 command and instead put the parked hands -- realized reach
   0.28-0.31 m, §19 -- into the wall: guard trip in 3.9 s).  It is an outward
   DRIFT of roughly +0.04 per metre walked that position feedback cannot
   arrest: the policy's own foot placement owns the wall-normal axis when it
   faces a wall.  Travel speed while crabbing is fine (0.165 m/s ~ tangent
   speed) and the walk is locally steady (band 0.019 m).
3. **Therefore: ride the wander, gate on steadiness, and STEP IN to press.**
   A crab station drops the lane-error term from the acquire gate (drift and
   band remain -- the walk must be steady, not on-line), walks ~0.2 m in to a
   physically-measured press stance (0.36 m; cylinders 0.33), presses with the
   segment-bounded cap, and walks back out.  This is demo_v3's stance cycle,
   reinstated knowingly.

Accepted on r5 segment 2, 3 stations x 2 runs (press rates 0.025 and 0.035):
**6/6 presses hit, 12.9-20.2 N, all step-ins arrived at 0.38-0.40 m, 0 force
aborts.**  Rate 0.035 showed no force penalty (max 14.30 N) and saves ~2 s per
press; adopted for the demo.  The 20.2 N outlier sat at the deepest realized
stance (0.380 m) at rate 0.025 -- the spread is stance-depth-driven, not
rate-driven, and stays under the 25 N line with margin.

### The crab full runs: five iterations, each one mechanism

| run | outcome | contacts | in band | >25 N | aborts | belief err med/final | what it taught |
|---|---|---|---|---|---|---|---|
| crab1 | arrived 0.18 m | 14/15 | 8 | **6** (max 36.4) | 0 | 0.100 / 0.099 | step-in arrives with momentum; 0.4 s settle is not stillness |
| crab2 | arrived 0.32 m | 13/19 | 9 | 1 (27.96, braked) | 0 | 0.184 / 0.234 | velocity settle works but the LIVE-pose press hold follows drift back out (3-6 N presses) |
| crab3 | stuck at corner | 9/17 | 8 | 0 | 2 | 0.146 / 0.946 | anchoring the press on the DRAWN stance yanks the base a belief-error along the wall; six presses swept air |
| crab4 | **arrived 0.075 m** | 8/15 | 7 | 0 | 0 | 0.157 / 0.267 | freeze-at-arrival anchor fixes forces (peak 14.60); an 11.9 N graze fed to the PF as "free space" blew sigma to 1.36 |
| crab5 | arrived 0.093 m | 9/17 | 8 | 0 | 0 | 0.152 / 0.107 | loaded-free guard + raycast stance depth; safety clean twice running |

Safety, settled: the last two runs have **0 aborts, 0 samples over 25 N, peak
14.6 N**, with three independent mechanisms holding the line -- the velocity-
gated settle (stillness before pressing), the frozen-at-arrival anchor (no
drift-following, no belief in the hold), and the in-press 25 N brake (the
probing hand is guard-exempt by design, so the press protects itself).

Landmarks: C1, C2 (via the +-0.28 hunt) and the corner's east face land in
band in both clean runs.  Two stations stay lost: the branch mouth and the
corner's second (south) face, both to a mid-run belief excursion that starts
in the C2 apex region -- the observation model's association is fragile where
a curved surface stands close in front of a wall, and both a flank contact
and an honest free sweep there can move mass to the wrong basin before later
flat-wall touches pull it back (final err 0.107).  That is follow-on work on
the observation model (or an information-driven scheduler that declines to
press where association is fragile -- the ATE integration), not an executor
patch.

Also measured this cycle, for the record: the scan re-extraction (concurrent
work) grew the room from 48 to 80 obstacles between runs and put a +0.046 m
step in the east wall's first 2 m; the per-station stance now comes from a
map-side raycast rather than one anchor line, and two 45-49 N spawn-settle
transients against the (now 4.6 cm closer) start wall were caught by the
guard exactly as designed.

## 22. ATE as the local planner, and room2 both ways

Owner direction: use the validated ATE program (wt/9.4-tp, DAIRLab-style EIG)
as the LOCAL PLANNER at landmarks -- not a homegrown heuristic -- and deliver
room2 (the real-robot test room) in both directions with its corner as the
ONLY landmark.

**The port.**  `tactile_explore/{belief,infogain,noise,rng}` copied verbatim
from commit 9a17a4f (byte-identical; one documented import inline).  Their
belief/EIG unit tests pass on the source tree (the single failure there is a
TP-01 data-condenser test over episode files this pipeline does not use).
`LandmarkATE` wraps it: one hypothesis (the landmark's known shape), prior
sigma from the navigation PF, candidates = stances the crab machinery can
take, `eig_closed_form` orders them, `worth_pressing` ends the episode.  The
PF remains the navigator; ATE chooses the touches; both consume the same
observations (composed with the BELIEVED pose -- no truth in either).

**The concave corner.**  A room corner is not any solid in the source shape
library: modelled as a box, both pressable faces land on the wrong side of the
corner and the EIG correctly scored every candidate ~0 (p_hit 1e-10, measured
on the first room2 run -- the scorer refused a geometrically impossible probe,
which is the right failure).  The ported copy gains a `ccorner` shape
(material on the far side of both faces, state = the corner point itself),
with face association BY NORMAL, the source program's own lesson.  Unit tests
pin the behaviours that make it a planner: after one face-A touch the best
next press flips to face B; two orthogonal touches converge the corner to the
true offset within 2 cm; a redundant re-press scores below a fresh face; a
miss never manufactures uncertainty.

**Room2, forward (start -> goal, both ~0.55 m off their walls).**  `arrived`
0.06 m true / 0.11 believed, 3/3 touches in band, 0 aborts, peak 14.41 N.
The ATE pressed the corner exactly as designed: west face at its top-EIG
candidate (2.66 nats planned -> 14.41 N), rounded the corner, north face
(1.89 nats -> 12.0 N); episode entropy -5.35 -> -11.98 nats.  Corner estimate
(-0.569, 3.710) vs map (-0.55, 3.75): the offset equals the robot's own pose
error -- the object-belief/pose-belief duality doing the localizing.

**Room2, reverse -- and the drift tail finally attributed.**  First reverse
run arrived (0.12 m, 0 aborts) but pressed poorly: the misses pressed from
0.44-0.49 m while every hit pressed from 0.39-0.41 -- and lining up EVERY
miss across every crab run: **all of them sit on RIGHT-hand walls; left-hand
walls never missed.**  The press-time outward drift is side-specific gait
asymmetry, not a per-station lottery.  Fix: right-hand segments pre-pay the
median drift (stance 0.035 m deeper).  A rerun validates it below.

**The rerun's verdict on the drift fix: refuted, and the finding sharpened.**
With the stance 0.035 m deeper, the cursed stations pressed from 0.477-0.486
-- the same absolute depth as before -- while the good station held its
stance to 2 mm (0.357 -> 0.355, 14.69 N).  The right-hand drift at those
stations is an ABSOLUTE ~+0.12 m during the press, independent of entry
depth; pre-paying it cannot work, so the compensation is reverted.  What
remains true and useful: the tail is side-specific (left-hand walls have
never missed, across every run) and station-systematic (the same three
stations, twice each).  The lever is the press hold's authority at those
stances -- follow-on work, not a stance knob.

**Room2 reverse, as delivered** (`room2_rev_s0_c`): `arrived` 0.11 m true /
0.20 believed, 0 aborts, peak 14.69 N.  The corner ATE: face-A press at its
top candidate swept free from 0.477 (the drift tail) and was fed to the
belief as the negative evidence it is (entropy -5.59 -> -8.47), face-B press
landed 14.69 N (-> -11.93 nats); corner estimate (-3.115, 0.635) vs map
(-3.05, 0.55), sigma (0.069, 0.028) -- the estimate carries the robot's own
pose error, as designed.  Two tail rhythm stations pressed air from ~0.48
(safe bounded sweeps, banked as free-space evidence).  Videos:
`out/demo_map/room2/room2_fwd.mp4`, `room2_rev.mp4`.

## 23. The sim2real batch: room2 both ways, all noise on, 6/6 arrived

Owner directive: everything must be felt out.  The contract, stated precisely
-- global position enters ONLY through noisy touch (point sigma 5 mm, normal
5 deg, the ATE program's registered sensor model, one noisy copy consumed by
BOTH the PF and the ATE); motion executes only through noisy dead-reckoning
(targets corrupted 3%/m along, 2%/m cross, plus heading noise -- the "ideal
IMU / ideal displacement" gray channels are gone); the physical room sits
**0.12 m off the belief's map** (`true_delta`, newly plumbed into the MuJoCo
path, with every wall-distance read split into map-side vs PHYSICAL);
wall-relative ranging and the force sensor remain, declared, as the two real
onboard senses.

The pipeline, per the owner's four layers: graph3 plans the route (the
compile REFUSES a route the planner did not choose); the PF navigates; the
rule-based crab machinery follows walls at a 0.5 m press cadence; the ported
ATE owns the corner -- and the TURN GATE converts "reach the landmark" from a
scripted waypoint into a hard information requirement: no in-band contact on
the approach face + PF sigma_along under 0.12, no turn; bounded extra EIG
presses may buy it; otherwise the run DECLARES failure and stops.

| run | outcome | goal (true) | contacts | in band | >25 N | aborts | err med | ATE presses | gate |
|---|---|---|---|---|---|---|---|---|---|
| fwd s0 | arrived | 0.14 | 5/7 | 5 | 0 | 0 | 0.082 | 3 | PASS |
| fwd s1 | arrived | 0.07 | 4/6 | 2 | 0 | 0 | 0.028 | 2 | PASS |
| fwd s2 | arrived | 0.12 | 5/6 | 3 | 1 (35.9, braked) | 1 (guard) | 0.086 | 2 | PASS |
| rev s0 | arrived | 0.16 | 5/7 | 3 | 0 | 0 | 0.041 | 3 | PASS |
| rev s1 | arrived | 0.17 | 5/6 | 4 | 0 | 0 | 0.054 | 3 | PASS |
| rev s2 | arrived | 0.16 | 7/7 | 5 | 0 | 0 | 0.040 | 3 | PASS |

**6/6 arrivals under a 0.12 m registration offset**, every turn granted by
the information gate on real evidence, ATE 2-3 presses per corner with its
estimate collapsing to sigma 1-4 cm.  fwd s2's two events both sit at the
first stations after the approach -- the hand-over's execution-noise scatter
-- and both safety layers did their bounded jobs (the in-press brake capped a
5 ms spike; the parked-hand guard aborted a settle clip and the run
recovered).  Fix applied (first station moved from 1.0 to 1.4 m of run-in)
and that seed rerun; see the amendment below.

Getting the batch to run at all killed one more bug worth recording: the
first stand-to-settle implementation ran a PRIVATE stand loop that cleared
the settle window it was waiting on and skipped the budget checks -- a
livelock that burned 2.5 h of GPU before the batch timeout cut it (earlier
identical runs took 3-5 minutes).  The fix holds the base per-TICK inside the
main loop, so the window fills while standing and every watchdog stays live;
an unsettleable station now stalls out in 8 s, honestly.

**Amendment: the impact tail, three refutations, and the honest verdict.**
fwd s2 was rerun twice with candidate fixes and the single-sample spike
survived both: moving the first station (1.0 -> 1.4 m run-in) relocated it
(35.1 N at the new first station, 48.5 N at an ATE press); halving the reach
speed did nothing (54.0 N).  Three refuted levers -- entry depth, station
placement, commanded hand speed -- pin the carrier: **base gait sway**
(+-0.15 m/s, §15's own measurement, uncommandable) closing a 5-8 cm hand-wall
margin under execution noise.  Every occurrence is a SINGLE 200 Hz sample
(over-limit dwell 0.00-0.04 s), braked, with zero aborts-to-fall and every
run arriving; but the literal "no sample above 25 N" criterion is NOT met on
this seed, and this section says so rather than tuning the number.  The
transferable prior for the real fix is §15's gait-phase hold, which took the
lateral taps' identical impact tail from 3 over-limit to 0: gate the press's
contact approach on the sway phase.  Follow-on work.

Final tally under the full noise contract: **8/8 runs arrived** (six batch +
two reruns), all gates passed on evidence, all failures of the force
criterion confined to seed-2-forward's braked 5 ms transients.

## 24. The 0.2 m cadence: five variants measured, the platform limit named

Owner directive: press every 0.2 m.  The station machinery schedules 0.2 m
marks correctly; what defeats them is the crab STEP-IN's tangential veer --
the facing gait slides +0.40..+0.53 m ALONG the wall while closing its 0.13 m
of depth, every station, so the base ends ~0.45 past its mark and the next
two marks are already behind it.  Five configurations, all smoked on the same
seed and leg:

| variant | arc accuracy | press quality | verdict |
|---|---|---|---|
| gait step-in, depth-only arrival | +0.4-0.5 m slip/station -> realized ~0.6 m | 12-15 N in band, arrives | **the working point** |
| + tangential arrival requirement | slip unchanged | fought its own veer 10 s, drifted out, pressed air; 28.9 N + 2 aborts | refuted |
| standing-mode step-in | -- | cannot close 0.13 m depth at all; 0/5 pressed, gate declared failure | refuted |
| + veer feedforward (stance 0.45 behind) | **stations land on their 0.2 m marks** | press-phase outward bleed 0.39 -> 0.45-0.54, forces 3-9 N, 1/8 in band | refuted |
| + settle-cap 6 s + repin anchor at rest | unchanged | unchanged (bleed is DURING the press, not before it) | refuted |

The feedforward result is the sharpest: it proves the veer is compensable in
POSITION, and that doing so couples straight into the §22 press-drift tail --
after a long diagonal step-in the anchored hold cannot keep the base on the
stance through an 11 s press on ANY wall (previously right-hand walls only).
Position and press quality currently trade against each other through the
same mechanism: **the facing gait's base authority during and after lateral
maneuvers.**  That is the platform limit; the delivery runs at the working
point (~0.6 m realized cadence, in-band forces), and the follow-on that can
break the trade is press-phase gait shaping (§15's gait-phase hold family),
not another stance knob.

## 25. The no-touch ablation: arriving does not need touch, KNOWING does

Asked whether the demo had ever been run against a no-contact control, the
answer was no -- so `scripts/room2_run.py --blind` was added: identical
planner route, identical noise draws, identical seed, identical PF and goal
test, but **no press ever happens** and the corner is traversed open-loop
(the information gate is removed too -- gating a run that cannot touch would
be circular).  Three seeds, forward:

| run | outcome | physical error to goal | belief error | touches | wall clock |
|---|---|---|---|---|---|
| blind s0 | arrived | 0.063 | 0.126 | 0 | 1.6 min |
| blind s1 | arrived | 0.063 | 0.134 | 0 | 1.7 min |
| blind s2 | arrived | 0.075 | 0.115 | 0 | 1.7 min |
| touching s0 | arrived | 0.141 | 0.087 | 5 | 6.5 min |
| touching s1 | arrived | 0.068 | 0.026 | 4 | 5.9 min |
| touching s2 | arrived | 0.117 | 0.048 | 5 | 5.3 min |

**3/3 blind runs arrive.**  Touch halves the BELIEF error (0.115-0.134 ->
0.026-0.087) and does nothing for the physical arrival.  The blind belief
error is 0.115-0.134 -- i.e. essentially the 0.12 m registration offset it can
never discover, which is exactly what the touching runs discover.

The mechanism is structural, not a defect: **the wall-follow controller
servos on the PHYSICAL wall distance.**  Along wall 1 that pins the
wall-normal axis; after the corner, wall 2 pins the axis that was free
before.  Both axes are held by real geometry regardless of what the belief
thinks, and room2's goal sits 0.55 m off both walls -- the room catches the
robot.  Ranging is a declared real sensor (§23), so this is honest; it just
means the demo currently demonstrates *localization by touch*, not *touch as
a precondition for arrival*.

To make touch load-bearing for ARRIVAL the task must change, not the
controller.  Three candidates, in increasing fidelity to the owner's
"everything must be felt" framing:

1. **Remove wall ranging** -- distance to the wall is known only when felt,
   so wall-following itself becomes touch-driven.
2. **Put the goal away from any followed wall** -- cross open space at the
   end, where nothing catches the along-wall error.
3. **Add ambiguity** -- several similar corners/doorways where only touch
   says which one was reached.  This is the case the EIG/ATE machinery is
   actually built for, and the one the global-belief work already prices.

## 26. Honest sensing: taking away what hardware cannot have

Owner ruling, in their words: the simulation must be 100% of the real
situation, and information the real robot cannot get must not be handed to
G1 -- specifically, **there will be no lidar and no wall ranging on the
deployed machine**.  An audit found three leaks and one absence:

1. **Observation noise was OFF.**  The policy was trained with gyro
   +-0.2 rad/s, projected gravity +-0.05, joint pos +-0.01 rad, joint vel
   +-1.5 rad/s and EE-command +-0.02, but every play cfg sets
   `enable_corruption = False` and mjlab then NULLS `term_cfg.noise`.  Every
   run in this document before this section used perfectly clean,
   zero-latency proprioception.  There is still no latency model anywhere.
2. **The locomotion loop closed on ground truth.**  The 492-D network itself
   is deployment-clean (no `base_lin_vel`, no world pose, no terrain scan --
   verified term by term), but the 3-D twist it consumes comes from
   `WallFollowSe2Command`, which computes its error from `root_link_pos_w` /
   `heading_w` every control step against a world target this backend writes.
   The BELIEF had been firewalled since 20; the BODY never was.
3. **Wall distance was a perfect sensor.**  `_phys_standoff` returned the
   exact analytic distance from the true pose to the exact wall plane, with
   perfect knowledge of WHICH wall, and drove the lateral servo, the step-in
   arrival test, the press cap and several gates.
4. **Force had no sensor model.**  Raw MuJoCo contact-force sum: unbounded,
   noiseless, at the 200 Hz substep rate.  Hardware has SensX: 0-20 N,
   +-1-3 N, ~100 Hz (`docs/DEPLOY_INTEGRATION_PLAN.md`).

`MujocoConfig.honest_sensing` (default true) closes all four.  The backend
carries a drifting onboard estimator and biases every attractor target by
(truth - estimate), so the truth-fed controller realizes the estimate's
error without the policy or the FSM being touched; the estimate is corrected
ONLY when the particle filter speaks.  `honest_sensing=False` restores the
old behaviour exactly, for the M2 tick regression running in parallel.

### What the honest run measured

The chain the owner asked for -- follow wall, ATE understands the corner,
turn, follow the next wall -- **does execute**: wall contacts at 18.4-20.9 N
in band, the ATE pressing the corner and driving its entropy -5.67 -> -8.6
-> -10.3 and stopping itself, and the turn gate granting the turn on a real
contact plus sigma_along 0.09.  Two runs reached the goal leg (0.18 m and
0.38 m true from the goal).  It is not yet reliable end to end.

Six defects were found and fixed on the way, four of them mine:

| defect | evidence |
|---|---|
| odometry drift modelled as FULLY systematic | 2.7x the repo's per-segment constants over 7 m (0.21 vs 0.076); belief 0.74 m off by wall 2.  Split 35% bias + random walk -> 0.147 m, belief error 0.09-0.11 |
| the hold chased its own drift | the bias was recomputed per tick, so a held base slid its own target and never settled; the ATE's corner press came back `unsettled`.  Latch the bias while holding |
| the belief error was compensated TWICE | the executor's `s(truth)-s(belief)` station shift plus the new estimator bias; `_arc_shift` returns 0 under honest sensing |
| a REFUSED press was fed as free space | `base_not_facing_wall` means the arm never swept; one such non-observation moved the belief 0.25 -> 1.27 m and tripled sigma |
| a safety gate judged a LABEL, not a measurement | `feeler_stuck` (depth not reached) vetoed a press from a base truly 0.30 m from the wall -- well in reach -- so the belief that caused the veto could never be corrected |
| the turn was granted on ONE corner face | face A pins the axis normal to the wall being left; the axis the NEXT wall needs is pinned only by face B, and the ATE skips it on its own confidence (0.007 nats).  Two gates now, each where its face is reachable |

### The open blocker, stated exactly

After the corner the belief is 0.19-0.43 m wrong on the axis normal to the
wall being entered -- and that axis is observable only by touching that very
wall.  The probe must therefore commit an approach depth using the number it
exists to measure.  Measured from both sides at a believed 0.40 m: truly
0.71 m (nothing within the arm's 0.43 m reach) and truly 0.18 m (inside the
parked hands' 0.32 m sweep, every press refused as unsquare).

Hand-ahead feeling -- hold the probe hand out front so it meets the wall
before the body does -- is the right mechanism and was tried: it made the arm
a battering ram, because the base target stayed at the 0.36 m stance and kept
driving 0.04 m past contact (134.6 N true peak, 6 aborts).  REVERTED.  Doing
it properly needs two more things this branch did not have: the base target
must stop AT the hand's reach rather than past it, and the felt contact must
itself become the tap instead of being followed by a normal press.

## 27. The honest run closes: felt, not guessed

`room2_fwd_s0_s2g` **arrives, 0.044 m from the goal**, with no wall ranging,
no truth-driven position loop, hardware-scale IMU/encoder noise, a SensX
force model (0-20 N, ~100 Hz) and a 0.12 m registration offset the robot
must discover for itself.  The chain the owner asked for runs end to end:
follow a wall by touch, let the ATE understand the corner, turn on
information, follow the next wall, and CONFIRM arrival by feeling a marker.

| | value |
|---|---|
| outcome | arrived, 0.044 m true (0.227 believed) |
| belief error | median 0.062 m, max 0.183 m |
| touches | 7 of 13 contacted, 6 in band |
| corner | face A 19.5 N, face B 15.7 N, gate sigma_along 0.076 |
| goal marker | **confirmed**, 14.6 N |
| force | sensed peak 20.8 N, TRUE peak 36.9 N, 6 aborts |

### What made it work: feeling instead of guessing

Every approach depth used to be a constant I tuned.  That constant provably
could not work: the registration offset lies along map +x, so the west face
sits 0.12 m nearer than the map says, the north face is not displaced, and
the goal pillar is different again.  Tuned for the west wall (0.50) the same
number left the pillar 0.51-0.61 m away, past the arm, and every marker
press grazed at 3 N.

It is replaced by an ordering that holds whatever the belief says:

    hand reach 0.42  >  press stance 0.36  >  parked sweep 0.32

With the hand held out ahead, the HAND always meets the wall first, so the
approach aims deep and lets contact stop it.  A wrong belief can then only
make the stop happen sooner, never later.  Most contacts in the arriving run
are `felt_on_approach` -- the robot never pressed at a place it had guessed.

Six iterations were needed, and three of them failed the same way: **the
safety of a contact was left to detection after the fact, and detection is
always a tick late.**

1. base target stayed at the press stance and drove past contact -- 134.6 N
   true, 6 aborts;
2. hand and base were commanded together, but the hand needs 2.3 s to extend
   and a 0.13 m step-in finishes sooner, so the BASE arrived first and the
   press drove a half-extended arm into the wall -- 114.6 N;
3. the 12 N guard test ran BEFORE the 4 N felt test, and force crosses both
   inside one 50 Hz tick, so `felt` never fired once in two runs while the
   approach kept pushing -- 109.4 N.

The fix that finally bounded it is the one the interactive-perception
literature states plainly: **a guarded move is a SLOW move.**  Force limiting
alone does not bound impact, because the guard reports 12 N while the base
carries its momentum through.  Approaching at the minimum speed took the
true peak from 135 N to 36.9 N.

### The recurring bug class, stated once

Three separate defects, one shape -- **a gate blocking the very measurement
that would resolve the uncertainty the gate exists to guard against**:

* `feeler_stuck` (a LABEL, not a measurement) vetoed a press from a base
  truly 0.30 m from the wall, well within reach;
* a REFUSED press was fed to the filter as free space -- a measurement that
  never happened, which moved the belief 0.25 -> 1.27 m;
* the lane-acquire gate refused every station on the second wall because the
  belief was 0.47 m out, so nothing was touched and both error and sigma
  FROZE for nine stations.

And two of the same-source kind: odometry must have ONE source (the
estimator, not a second draw), and a contact normal must have ONE convention
(`_contact_body_frame`'s, not a second one derived from the segment -- that
one jumped the belief 0.3 m on the outgoing corner face and diverged it to
1.3 m with sigma growing to 0.54).

### Still open

* 6 aborts and a 36.9 N true peak: better than 135 N but not a number to
  deploy.  The remaining spikes are parked hands scraping while walking a
  lane the belief has misplaced.
* One seed, forward only.
* The 20 N SensX clip still hides real force from the control path; joint
  torque / tracking error is an independent contact channel worth adding.

## 28. The contact peak is an impact, and the planner cannot reach it

Owner TODO 1 was to fix the 30-48 N true peaks.  Three attempts, all
measured, and the answer is that this number does not belong to the planner.

**First, the number is less alarming than it reads.**  Every "abort" in a
clean run is the 12 N guard noticing a touch we deliberately made -- the
abort forces (18.1, 21.9, 21.9, 19.5, 15.8 N) are the same numbers as the
felt contacts.  They are not collisions.  **Time spent above the 25 N limit
is 0.0 s.**  Contact forces themselves are in band: 7.2-21.9 N.

What remains is real but narrow: a deliberate touch reaches ~34 N TRUE while
SensX reports 21.9 -- an overshoot the hardware channel cannot see, because
it clips at 20 N.  Invisible over-force cannot be detected on the robot, so
it has to be prevented.

**Three levers tried:**

| lever | result |
|---|---|
| approach slower (feel scale 0.12 -> 0.06) | REFUTED, and it broke the demo: 0.045 m/s is ~6% of the policy's trained lateral box, the gait does not engage, the approach times out and sweeps short -- contacts 7 -> 1.  **This platform has a slowest walk, and it is faster than a gentle contact wants.** |
| stop the body before moving in | no effect on contact speed (median 0.110 -> 0.105 m/s).  The momentum is not carried in from the lane walk; the base RE-ACCELERATES over the 0.13 m inward move |
| unload in the same tick the hand feels | REVERTED.  True peak 34.1 -> 30.8 N (inside run-to-run noise) and it cost a corner failure |

**Conclusion: the peak is a sub-tick impact.**  Its magnitude is set by
closing speed and contact stiffness, it is over before the next 50 Hz
control tick, and the closing speed cannot be lowered on this platform --
which is the same result 15 recorded for lateral taps ("peak force is
gait-limited and NOT tunable").  Nothing above the controller can bound it.

The levers that remain are all below the planner: arm compliance during the
approach, or a low-level force loop that yields on contact.  That is the one
honest argument for evaluating the decoupled vel-force checkpoints, which
carry a force command channel -- not the EE-tracking argument, since
instrumented tracking error is flat at 2.0-2.4 cm while the force varies 2.4x.

## 29. The belief sold its heading to buy position

Tightening the particle filter's normal tolerance to match the sensor
(`sigma_normal_rad` 0.200 -> 0.100 rad, against ~5 deg of injected normal
noise) made the forward leg **fail**, and the failure is worth more than the
fix was.

| | sigma 0.200 (arrived, 0.09 m) | sigma 0.100 (failed, 0.33 m) |
|---|---|---|
| yaw error, first wall | 1.2 - 7.5 deg | **0.16 - 5.4 deg** |
| position error at the corner | 0.092 m | **0.035 m** |
| yaw error after `ate:corner:kiss` | 5.3 deg | **28.5 deg** |
| position error, next wall | 0.19 m | 0.94 -> 1.29 m |

The tighter tolerance was *better everywhere it was measured against a flat
wall*, and then lost the run at one event.  Position error hit its best value
in the history of this route (0.035 m) at the exact moment heading error hit
its worst (28.5 deg).  That is not a coincidence: the two are exchangeable
against a wall, and the filter took the trade.  Nothing detected it, because
every gate the route runs -- `face_dot`, the corner information gates, the
lane-acquire test -- is computed *from the belief*, so all of them read
"square to the wall, 0.035 m of doubt" while the robot was 28 deg off.  The
error was invisible until the robot walked 1.7 m, at which point 28 deg of
heading became 0.94 m of position in a single station.

### Why the corner kiss is the poison

The kiss aims at the VERTEX.  A vertex has no surface normal.  But
`contact_loglik` scores the felt normal against `nearest_surface()`, which
for a vertex touch returns one of the two flat faces -- 45 deg from the
bisector the hand actually met.  At 0.200 rad that mismatch is 2.7 sigma and
gets absorbed; at 0.100 it is 7.9 sigma, and the only way the filter can
reconcile it is to rotate the particle cloud.

**Fix**: `ContactObs.normal_sigma_rad`, a per-touch tolerance.  The kiss
passes 0.0 and contributes position only.  It loses nothing: the vertex lies
on both faces, so it pins position as well as any touch.  It simply stops
claiming to know something it cannot.

### The defect underneath: a sensor that rails

`_face_quat` commanded the wrist `dyaw = clamp(face_dir - heading, +-0.12)`.
So the felt normal is the true body-frame normal **clipped at 6.9 deg**.
Measured on this run: the base sat 36 deg off square, the wrist railed at
6.9 deg, and the filter -- told the surface was 6.9 deg off the robot's nose
-- concluded the robot was 6.9 deg off square and rotated the belief by the
29 deg difference.  A saturated sensor read as a measurement is worse than no
measurement: it is confidently wrong, and consistently so.

The robot can see this without privileged information, from its own commanded
and measured wrist angle, so `normal_railed` is now reported per contact and a
railed touch declines its normal the same way the vertex does.

### A third truth read, found by pulling the same thread (audit blocker 13)

`_face_quat` aimed the hand using `heading_w` -- the simulator's true base
heading.  Besides being a truth read on the control path, it was
self-fulfilling: the hand was being aimed by the very quantity the felt
normal exists to estimate, which made the sim's normal channel better than
hardware's can ever be.  It now uses `working_pose()` under honest sensing.

### What this says about the normal channel generally

A wrist that rails at 6.9 deg is a poor heading sensor.  Contact POSITIONS
along a wall are a far better one: successive touches lie on a line whose
direction is the wall's, so the heading accuracy is
(contact noise)/(baseline) -- 8 mm over 1.5 m is ~0.3 deg, an order better
than the wrist.  The peer session measured the same conclusion from the other
end (a constant 3.17 deg frame rotation their position-only filter could not
observe, costing 0.20 m of lateral at 3.6 m and 0-for-22 marker touches) and
fixed it with wall-line registration.  That is the direction to take this,
rather than asking more of the normals.

### Also fixed here: the press latch was reading a force hardware cannot supply

Audit blocker 8 was recorded fixed when the substep brake and the tap trigger
were routed through the SensX model.  `_probe_force_n` was missed and still
returned raw MuJoCo contact force -- exact, unbounded, at the physics rate --
into `contacted = axis_n >= 2 N` and into the press-mean sum.  That also meant
the air-vs-graze mean gate, calibrated on the SENSED distribution (air ~0.8 N,
because zero-clipped sigma=1 noise has mean 0.8), was being applied to a raw
signal whose air mean is 0.0.  Now routed, with the true value kept for
evaluation only.

Two smaller honesty gaps closed alongside it:

* the 100 Hz guard was collapsing to 50 Hz.  `_sensed_force_n` timestamped
  samples with `self._t`, which is not advanced until after the physics step,
  so every substep of a control tick shared one timestamp and the hold
  returned one sample per tick -- a guard blind to any spike landing in the
  second half of a tick.  It now uses the substep clock it is handed.
* `plan_force_latency_s = 0.010`: the deploy plan runs sensors and controllers
  onboard the Jetson and the planner offboard over a measured ~5 ms link, so
  a force the PLANNER acts on is a round trip old.  The guard node is onboard
  and reads at zero lag.

## 30. The reverse leg fails for a different reason, and the sign is the whole story

Forward and reverse were failing together, so they looked like one problem.
They are two.  Reverse (seed 1, corner `north:west`) never gets past the
first corner: it retries `ate:north:west` seven times, sigma grows
0.117 -> 0.263, and every press sweeps air at 2.5-3.8 N.

Measured, true against believed standoff at each corner press:

| TRUE before | TRUE after | believed | real drift | belief error |
|---|---|---|---|---|
| 0.575 | 0.592 | 0.390 | +0.017 | **+0.186** |
| 0.619 | 0.615 | 0.395 | −0.003 | **+0.223** |
| 0.512 | 0.527 | 0.392 | +0.015 | **+0.120** |
| 0.559 | 0.532 | 0.394 | −0.028 | **+0.166** |
| 0.428 | 0.444 | 0.393 | +0.016 | +0.035 |

**Real drift during the press is negligible** (−0.03 to +0.02 m): the base
holds its stance fine, and the "+0.12 m absolute press drift" recorded in
`_stance_for` is not what is happening here.  What is happening is that near
the corner the base stands **truly 0.51-0.62 m from the wall while believing
0.39 m**.  The hand reaches ~0.42 m and the press travel cap is
`believed + 0.10`, so the sweep stops 0.04-0.12 m short of a wall that is
really there.

Forward has the same kind of error with the **opposite sign** (−0.04 to
−0.25 m: "I think I am farther than I am"), which puts every touch safely
inside the hand's reach.  That sign, not any difference in the planner, is
why forward arrives and reverse does not.

Two things this corrects in earlier notes:

* `true_standoff_m` does **not** hold the true standoff.  It is
  `_est_standoff(seg)` -- the BELIEVED depth -- and the true value is in
  `true_standoff_eval_m` (before) and `standoff_pressed_m` (after).  Reading
  the name instead of the assignment turns a belief error into a phantom
  0.20 m of motion; the first draft of this section did exactly that.
* The press travel cap is therefore honest (believed + margin), not a truth
  read.  Blocker 8's neighbourhood is clean.

### The fix: a missed sweep is a measurement

The loop retried the identical stance seven times because nothing in it asked
why the sweep came back empty.  A bounded free sweep proves there is no wall
within the depth just swept, which contradicts the stance that was chosen --
and the robot needs no truth to notice.  `_ate_corner` now closes the stance
by 0.05 m after each swept-air miss, floored at `CRAB_PRESS_STANCE - 0.14`
(the existing clearance clip, not a new constant), and resets the override as
soon as a press contacts.

Worth stating plainly: the commanded stance was already <= 0.44 m while the
base delivered 0.51-0.62 m, so this asks the controller for something it was
already failing to deliver.  It is the only lever the planner has, it is
bounded, and it is earned by evidence rather than tuned -- but if the base
still cannot close the gap, the answer is a controller limit and the honest
outcome is a declared failure, not a planner constant.

## 31. The step-in stopped because the belief said so

Forward arrives (0.27 m, 9/11 touches, marker confirmed at 19.0 N) while
reverse fails at the same corner run after run.  Section 30 named the
symptom -- the base stands 0.5-0.75 m from a wall it believes it is 0.39 m
from -- but not the cause.  The cause is one line.

`_crab_step` ended the inward step on

    d = self._est_standoff(seg)
    if d <= want_off + tol:  return "stepped_in"

`_est_standoff` is the **believed** distance to the wall.  So the robot steps
in until it *believes* it has arrived -- and the belief is the one quantity
in the whole loop that can be wrong.  The feeler riding ahead of the body
exists to catch exactly this, and its verdict was not consulted.

The evidence is in the `turn_in` field, which records how the step ended:

| leg | `felt` (hand found the wall) | `feeler` (reached, felt nothing) |
|---|---|---|
| forward (arrives) | 9 of 11 | 2 |
| reverse (fails) | 2 of 10 | **8** |

And every `feeler` station reports a believed standoff of 0.389-0.395 m.
That constancy is the tell: it is not a measurement of anything, it is the
loop's own termination condition read back.  Truth at those same stations was
0.525-0.745 m.  The robot reached out 0.42 m, felt nothing, and recorded "I
am 0.39 m from the wall" -- the exact opposite of what it had just learned.
Every press after that capped its travel at `believed + 0.10` and swept air,
and the six retries all re-ran the same unreachable geometry.

### The fix

With a feeler leading, the believed target becomes a **floor to close past**
rather than a place to stop: the step may continue a bounded
`wp_feel_overstep_m = 0.20` further, and CONTACT ends it.

It is self-limiting, which is why the bound can be generous.  If the wall
really is where the belief says, the leading hand meets it first and the step
ends `felt` exactly as before -- the extra travel is only ever spent when the
belief was wrong, which is precisely when it is needed.  This is the
feel-your-way-in ordering (hand reach 0.42 > press stance 0.36 > parked sweep
0.32) finally applied to the step-in itself, rather than only to the press.

### Two supporting fixes from the same investigation

* **A missed sweep earns a closer stance.**  Both the corner ATE and the
  rhythm stations retried a stance that a swept-air miss had already proven
  unreachable.  A bounded free sweep proves there is no wall within the depth
  just swept, which contradicts the stance that was chosen, so the retry now
  closes by 0.05 m (floored at the existing clearance clip) and resets on any
  contact.
* **A miss where the map promised a wall counts as an unexplained station.**
  `ManagedBelief` only stepped its health monitor on CONTACTS, so a run whose
  every press missed never advanced the streak and stayed `NORMAL` while
  being 0.8 m wrong -- six times the cloud's own sigma, an error no
  reweighting could reach because there were no particles out there.  Such a
  miss is the strongest pose evidence available (no pose the cloud holds
  explains it) and now counts toward `RELOCALIZE`, which re-seeds the
  population wide enough to contain the truth again.

### Measured caution on the third one

On the reverse run that *did* reach the goal post, belief error was only
0.20-0.33 m while the base was physically unable to close the lane.  That is
a REACH failure, not a lost belief, and re-seeding the cloud would have
thrown away good localization to fix something localization did not cause.
The two are distinguishable on board -- a sweep that reached past the
believed face contradicts the belief, one that stopped short says nothing --
and the health signal is deliberately wired only where the map promised a
wall and the hand went past where it should have been.

## 32. Both directions arrive

| | forward (`room2_fwd_s0_v5`) | reverse (`room2_rev_s0_v7`) |
|---|---|---|
| outcome | **arrived** | **arrived** |
| goal, true / believed | 0.27 / 0.05 m | 0.26 / 0.18 m |
| contacts | 9 of 11 | 9 of 12, all 9 in band |
| contact force | 5.8 - 21.1 N | 19.1 - 21.4 N |
| goal marker | confirmed, 19.0 N | confirmed, 19.3 N |
| heading error, median | +1.58 deg | +0.18 deg |
| air-level (noise-latched) contacts | 0 | 0 |
| sensed / true peak | 21.1 / 41.8 N | 21.9 / 43.2 N |
| force aborts | 11 | 10 |

Reverse had never arrived before -- 0 for 9, best 0.86 m.  Every one of its
nine contacts is `felt_on_approach` with `turn_in=felt`, against 8-of-10
`feeler` (reached, felt nothing) in the failing runs.  That is the section-31
fix doing exactly what it was built to do.

### A correlation that did not survive the fix

Before the cause was found, the arrival record split perfectly along
`hand_side`: forward runs both walls at +1 and arrived 4 times in 5; reverse
runs both at -1 and had never arrived in 9.  `_stance_for` carried an older
note saying the same thing.  It was a **confound**: reverse always carries
-1, so that split is equally a split by direction, and the two cannot be
separated by this route.  Reverse now arrives with `hand_side` still -1.

`force_probe_hand` was built to test the hypothesis and is kept as a lever
for a genuine hand asymmetry, documented as unvalidated end to end.  The
lesson is the ordering: the confounded hypothesis was cheap to test and got
tested first, when the known causal defect was still unfixed.

### Two video defects fixed alongside

* **Force arrows were drawn at the shin.**  `probe_hand_z_m = 0.15` is a
  PELVIS-frame offset and was used as an absolute world height, so every
  touch arrow sat 0.75 m below the hand that made it.
* **The camera lost the robot at corners.**  `follow_azimuth` took the FIRST
  wall in range; at a corner the robot is near both, so it framed from the
  room side of one and through the face of the other.  The reverse demo hid
  the robot for two stretches, one of them the corner kiss -- the moment the
  video exists to show.  It now averages the normals of every wall in range,
  which points into the open corner.
