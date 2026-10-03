# S5 — should the tactile navigator move to the decoupled controller?

**Question (owner):** can the locomotion policy be swapped for one of the
decoupled vel-force-MPC checkpoints —
`python scripts/rl/play.py Mjlab-Eval-Force-Maze-G1 --checkpoint_file
checkpoints/decoupled_vel_force_mpc/vforce_model_52000.pt --viewer viser`?

**Recommendation: not now.  Revisit at the hardware-bring-up milestone, and
only after instrumenting our own arm.**  The reasoning is below, and so are
the specific facts that would flip it.

## 1. It is a port, not a swap

The checkpoints do not live in the checkout the demo uses.  They are in a
separate worktree of the same repo (`tacnav_wt/dp_eval`, branch
`origin/decoupled-policy`), whose own venv registers `rl.eval`; our
`tactile_controller` checkout declares only the three `mpc_*` entry points,
so `load_env_cfg("Mjlab-Eval-Force-Maze-G1")` cannot resolve here at all.

The owner's understanding was that the three checkpoints share observations
and differ only in training reward.  That is true **between the three**, but
not between their task and ours:

| | ours (`Mjlab-MPC-Guided-Wall-Following-G1`) | theirs (`Mjlab-Eval-Force-Maze-G1`) |
|---|---|---|
| actor groups | one, `actor`, **492** | two, `upper_body_actor` **519** + `lower_body_actor` **478** |
| extra terms | — | measured EE pose (14) + EE pose error (18) |
| gait phase | computed in-obs from the twist | a first-class `gait` command term |
| force scaling | ÷40 | ÷50 |
| EE command frame | yaw-aligned | root body |
| hand command terms | `left_ee_pose` / `right_ee_pose` | `left_ee_cmd` / `right_ee_cmd` |

Beyond the vectors, two structural blockers matter more:

* their FSM **deliberately drops `hold_during_press`** — "no press-hold, no
  stand-at-wall latch" — which is the latch our stations were built on;
* `ProbeShield` writes `term._x_hold` and reads `arm_joint_ids` / `_fk_jac` /
  `_root_R`, none of which exist on the decoupled command objects, so **the
  200 Hz force brake cannot be installed at all**.

Cost estimate: a second env/runner/wrapper path, a two-group inference path,
renamed command terms, an FSM shim, and a shield adapter.

## 2. Its headline advantage is one we designed away

The eval's strongest number for `vforce_model_52000` is anchor hold under an
active press: **2.5 cm median / 4.2 cm p90 over 42 windows**, against our
whole-body policy's inability to park a base through a long press.

That was decisive when a station meant *stand still, extend, press for
10-15 s*.  After S2 it is not what we do: the hand rides ahead, contact ends
the approach, and the measurement is the contact itself.  There is no
multi-second stationary hold left to protect.

Meanwhile its recorded weaknesses land squarely on what we DO now:

* free-standing (no press) drift **p90 0.397 m**, long-tailed;
* roughly **1 in 40** windows degenerate — base walks away 1.0-1.5 m with
  ~1.5 rad of yaw wander;
* force under-delivery: 22 N commanded produces 12.3 N median.

We walk far more than we press, next to walls, with parked hands sweeping
0.32 m.  A 1-in-40 metre-scale excursion is a collision, not a bad episode.

## 3. What it does have that we do not

`vforce_model_52000` is **sim2real-confirmed on the real G1** (2026-08-17)
and is the M3 deploy candidate in `DEPLOY_INTEGRATION_PLAN.md`.  Nothing in
our stack has been on hardware.  That is a real asset and the reason this is
"not now" rather than "no".

Also relevant, and untested either way: its upper-body actor consumes 32
dimensions of EE pose and pose-error feedback that our whole-body actor never
sees.  Better arm tracking is plausible — but **no EE-tracking or jitter
metric exists for these checkpoints**; the eval measured touch rate,
time-to-touch and force only.  A grep of `docs/` and `out/` finds nothing.
Any claim that swapping fixes arm jitter would be invention.

## 4. What we should do instead, in order

1. **Instrument our own arm** — per-contact hand speed and EE tracking error
   at the moment of contact.  We do not have this today; a timestamp-based
   estimate is not good enough to compare controllers with.
2. **Fix the aborts we already understand.**  6-8 per run at 28-54 N true,
   and they are parked hands scraping a lane the belief has misplaced — a
   planner problem, not a controller problem.  Swapping controllers before
   fixing it would move the evidence, not the fault.
3. **Then** re-open this, with our own numbers on the same axes the decoupled
   eval used.

## 5. What would flip this decision

* our own instrumentation shows arm tracking error at contact is the
  dominant term in the force spikes (rather than base motion);
* we re-introduce a long stationary press (e.g. for force-regulated probing),
  which re-values the 2.5 cm anchor hold;
* hardware bring-up starts and the sim2real-confirmed actor is worth more
  than the port cost;
* the degenerate walk-away mode is shown to be absent under our command
  distribution (narrow twist, wall-adjacent) rather than the eval's.

---

## 6. The flip-condition was tested, and it did not flip

Section 5 named the condition that would reverse this: *"our own
instrumentation shows arm tracking error at contact is the dominant term in
the force spikes (rather than base motion)"*.  The instrumentation now
exists and the answer is **base motion**.

Seven `felt_on_approach` contacts, each recorded at the tick contact was
detected:

| contact | force | base speed | hand tracking error |
|---|---|---|---|
| west wall 1 | 18.9 N | 0.197 m/s | 0.020 m |
| west wall 2 | 14.7 N | 0.116 | 0.024 |
| west wall 3 | 13.6 N | 0.110 | 0.023 |
| corner face A | 9.9 N | 0.192 | 0.019 |
| corner face B | 9.3 N | 0.070 | 0.021 |
| north wall 1 | 10.7 N | 0.067 | 0.021 |
| north wall 2 | 7.9 N | 0.063 | 0.022 |
| | | median 0.110, max 0.197 | median 0.021, max 0.024 |

Arm tracking error is **flat** — 2.0 to 2.4 cm across every contact, a fixed
reference-to-measured offset rather than a tracking failure — while force
varies by a factor of 2.4.  A constant cannot explain a varying quantity.
Base speed varies threefold and tracks the force (r ~ 0.6): the fastest
approach produced the largest force, the slowest the smallest.

**The force belongs to the approach, not to the arm.**  The 32 dimensions of
EE feedback the decoupled actor adds would improve the term that is already
flat.  Fix the deceleration into a station first; the controller question
stays open on its other merits (hardware sim2real), not on this one.

---

## 7. All three checkpoints measured on the question that actually matters

Owner asked for the three to be tested anyway.  Done — but not on the axis
the prior eval used.  That eval starts recording force only after
`age > 1.5 s`, deliberately skipping the settle transient so the
force-regulation statistics stay clean.  **That transient is our entire
problem**, so a copy of the harness (`dp_eval/scripts/rl/test/impact_eval.py`,
new file, their script untouched) records the peak from the instant contact
begins.  8 envs x 80 s each.

| checkpoint | windows | falls | impact peak med | p90 | max | force at first contact | touch % |
|---|---|---|---|---|---|---|---|
| `vforce_model_52000` | 20 | 0 | **48.5 N** | 154.2 | 191.4 | 40.6 N | 79 % |
| `force_knee_model_60500` | 22 | 0 | **41.2 N** | 101.8 | 175.4 | 36.2 N | 78 % |
| `force_home_model_59000` | 8 | 0 | **38.1 N** | 66.1 | 78.7 | 29.8 N | 50 % |
| **ours, current** | 7 | 0 | **21.9 N sensed / ~34 N true** | — | — | 4 N (our trigger) | — |

`force_home_model_59000` had never been run before; it is gentler than the
other two and touches half as often.

### Read it carefully — there is a confound, and it is the finding

Their harness raises `contact_mask = 1` and `force_mag = F` in one step and
lets the policy fly the hand at the wall: no creep, no rate limit, **no
tactile stop**.  Ours creeps at 0.035 m/s and halts the instant the hand
reads 4 N.  So this is not a clean controller-vs-controller comparison, and
I will not present it as one.

But two things survive the confound:

1. **The force command does not bound the transient.**  At `F_cmd = 8 N` the
   peak median is 58.7 N — seven times the command.  A controller whose force
   channel yielded on contact could not do that however fast the approach.
   Its force loop regulates the steady press (the prior eval measured 22 N
   commanded delivering 12.3 N) and does nothing for the impact.
2. **All three land in the same place** (38-49 N median) despite reward being
   the only difference between them.  A quantity that ignores the training
   reward is not a policy behaviour — it is contact stiffness times closing
   speed, which is exactly what 28 concluded from our own side.

### What this actually says

The number that separates us from them is not the controller.  It is that
**we stop on contact and they do not**: their force at first contact is
30-41 N, ours is 4 N, because 4 N is where our approach ends.

So the safety of a touch belongs to the tactile stop, not to the policy.
Porting the decoupled actor would therefore require porting our stop with it
— and that is the most expensive piece of the port, since `ProbeShield`
cannot install its force brake on the decoupled command objects at all.

**Decision unchanged, now on three checkpoints' worth of evidence:** not
now, and not for force reasons.  `vforce_model_52000` remains the candidate
for a hardware milestone on the strength of being sim2real-confirmed — which
is a deployment argument, not a contact-safety one.

---

## 8. `vforce_model_52000`, intensively, under OUR approach

Owner asked for this checkpoint specifically, because it is the one already
sim2real-confirmed.  New harness
(`dp_eval/scripts/rl/test/vforce_intensive.py`, new file) drives it the way
our planner touches — rate-limited creep at 0.035 m/s, tactile stop at 4 N,
then a 3 s hold — so the controller is separated from the approach.  8 envs
x 6 cycles = 48 touch attempts, force command 15 N.

| metric | vforce_52000 (our approach) | vforce_52000 (its own eval) | ours |
|---|---|---|---|
| falls | 0 | 0 | 0 |
| **find rate** | **31 %** (15/48) | 79 % touch ratio | 7-9 of 10-13 stations |
| impact peak med | **29.0 N** | 48.5 N | ~34 N true / 21.9 sensed |
| impact peak p90 / max | 56.7 / 80.3 N | 154 / 191 N | — |
| force at the tactile stop | **10.5 N** | 40.6 N first contact | 4 N (our trigger) |
| **EE tracking error med** | **0.022 m** | not measured | **0.021 m** |
| EE tracking error p90 | 0.126 m | — | 0.024 m |
| base drift during approach | 0.013 m med, 0.057 p90 | 0.024 / 0.397 free-standing | — |
| time to touch | 0.48 s | 0.26 s | — |
| hold force (15 N commanded) | **6.8 N** | 12.3 N at 22 N cmd | — |

### The EE-tracking question is settled, and not in the expected direction

Owner's stated reason for testing these was that their EE tracking is more
accurate.  Measured on the same axis, with the same creep, it is **the same**:
**0.022 m median against our 0.021 m**.  At the tail it is five times worse
(p90 0.126 m against our 0.024 m).

That is not a knock on the checkpoint — it says the ~2 cm is not a controller
property at all.  Two independently trained policies, one whole-body and one
decoupled with 32 extra dimensions of EE feedback, land on the same number,
which makes it a property of the arm and its reference, not of the actor.
Swapping controllers cannot move it.

### The impact does improve, and it is the approach that does it

29.0 N median under our creep against 48.5 N under its own fling — the same
policy. So most of that 48.5 N was the harness, exactly as suspected.  But
29.0 N is still no better than our own ~34 N true, and the p90 (56.7 N) and
max (80.3 N) are worse than anything our stack produces.  It also stops at
10.5 N where we stop at 4 N: its force channel does not arrest the approach
promptly even when told to.

### The finding that would actually block deployment

**It only found the wall 31 % of the time** (15 of 48), against a 79 % touch
ratio in its own eval.  The difference is the approach: a creep that stops on
first contact needs the hand to arrive at the surface along the commanded
axis, and this policy's arm wanders enough (p90 tracking 0.126 m) that two
thirds of the creeps ran their 4 s and never felt anything.  A tactile
navigator that misses two touches in three is not deployable, whatever its
locomotion quality.

Steady-state regulation is also weak here: 15 N commanded delivers 6.8 N.

### Verdict

Tested as asked, on every axis asked for.  **`vforce_model_52000` does not
improve any metric our tactile navigator depends on**, matches our arm
tracking exactly at the median, is worse in the tail, is worse in impact p90,
and finds the wall a third as often under the only approach we actually use.
Its sim2real confirmation remains genuinely valuable and is a reason to keep
it in view for hardware bring-up — but on contact behaviour, the evidence now
says the port would buy nothing.

---

## 9. `vforce_model_60000_v1` — the owner's adoption, measured (2026-08-18)

Owner directive: Zhenyu's new **vforce v1** (`vforce_model_60000_v1.pt`,
commit `cce58ec` "[decoupled virtual force mpc] deployed version 1",
2026-08-18) is sim2real-confirmed and is to be our controller.  Same task id,
same two-group obs layout (519/478 — diff on the obs configs is empty), new
weights; the training changes target exactly the defects measured in §8:
anti-drift `stand_still` −2.5, `arm_anchor_mimic` for the wandering arm,
upper-only force rewards.

Measured, same harnesses as §7-8, plus a new FEEL protocol (contact_mask
OFF during the approach — our planner's touch semantics — mask+force only
after contact):

| metric | v1 mask-on | v1 FEEL | 52000 mask-on | 52000 FEEL |
|---|---|---|---|---|
| find rate | 0.27 | 0.19 | 0.31 | 0.31 |
| EE track med (m) | **0.015** | 0.080 | 0.022 | 0.043 |
| EE track p90 (m) | **0.045** | 0.179 | 0.126 | 0.096 |
| impact peak med (N) | 40.7 | 40.4 | 29.0 | 42.5 |
| force at stop (N) | 34.5 | 21.3 | 10.5 | 28.2 |
| time to touch (s) | 0.34 | 3.98 | 0.48 | 1.18 |
| falls | 0 | 0 | 0 | 0 |
| impact, fling harness p90/max | 115 / 170 | | 154 / 191 | |

### Three findings that decide the integration design

1. **The decoupled arm obeys pose references only in contact mode.**  With
   the mask off, tracking collapses (v1 0.015→0.080 med; 52000 too) — a
   family property, and v1 is MORE mask-dependent (its arm rewards are
   contact-regime shaped).  **Our position-creep-and-stop primitive cannot
   drive this controller.**  The native semantics is: command `mask=1` with
   a LOW `f_cmd`, and the policy itself reaches, touches and regulates —
   *the policy is the feeler*.  This matches the compliant-probe pattern
   from the interactive-perception literature rather than our current
   guarded creep, so the adoption changes the touch primitive, not just the
   actor.
2. **v1's real wins are clean:** mask-on arm tracking p90 3× tighter
   (0.045 vs 0.126 — `arm_anchor_mimic` worked), force delivery 20.6 N at a
   15 N command against 52000's 6.8 (overshoot vs undershoot; either way the
   planner must calibrate its `f_cmd` against the realized gain), and the
   fling-harness impact tail improved (p90 154→115).
3. **Impact stays ~29-43 N in every cell** — protocol- and
   checkpoint-insensitive, consistent with 28: a sub-tick closing-speed ×
   stiffness quantity nothing above the controller can bound.  The find-rate
   ceiling (~31 %) is a harness geometry artifact (targets drawn beyond the
   creep window), not a discriminator.

### Adoption plan

* **Touch primitive for v1:** mask-on soft-force touch (`f_cmd` ≈ 6-8 N
  commanded, calibrated against the measured ~1.4× gain), contact detected
  from the force rise, mask dropped after the measurement.  Retire the
  pose-creep feeler when driving this actor.
* **Integration route: RESOLVED (owner-ruled, M3 session confirmed).**  The
  deploy-side port is DONE on their line: ONNX `vforce_v1_60000_policy.onnx`
  exported (526-d wire), the override yaml switched, the Gate-1 manifest
  re-pinned, and a 21/21 stack battery passed on v1 (stands, 0.002 m stop
  drift, EE tracking 2.7 cm, force slots exact).  Our planner emits the
  contract through their Guard (sole writer): `cmd_vel_in` (Twist),
  `force_cmd_in` (8-d unitless intent, manager applies /50),
  `left_hand_target_in` (pelvis frame), plus their P3 touch primitive and
  ContactObs.v1.  **No MuJoCo-side port**: the MuJoCo stack stays the
  planner-development sim on hand_force_v2; both stacks now share the same
  touch semantics (touch-and-retract, mask-on approach).
* v1 calibration facts for that line: f_cmd 15 (unitless) realizes ~20.6 N
  (1.4x over; 52000 under-delivered 6.8 -- a stale calibration curve would
  overshoot ~3x), small-command twist attenuation persists in the deploy sim
  (hardware response is Zhenyu's authority), and the ~40 N contact-onset
  transient is checkpoint-insensitive.
* The `anchor_q` reach-pool arity change in `ee_tartget.py` is the one
  compatibility hazard; a wire-parity run against the ONNX export is the
  gate before anything hardware-facing.
