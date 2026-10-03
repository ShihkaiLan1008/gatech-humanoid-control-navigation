# M3 press options: brief-tap MVP vs sustained pressing — decision brief

Prepared 2026-08-18 at the owner's request, **before** any change to the acceptance
criterion. The standing criterion is unchanged: *a press holds 8–12 N for 0.8 s.*
M3(c) is recorded as **navigation PASS / sustained-force regulation FAIL**; M3 is **not
closed**. Nothing here has run on hardware.

## The measurement that frames both options

Sustained contact is not failing for want of tuning. Two independent controllers, in two
independent simulators, show the same mechanism: **contact reaction displaces the base,
and the base retreat breaks the contact.**

| where | controller | result |
|---|---|---|
| deploy sim, this M3(c) work | vforce decoupled (velocity base) | base drifts −0.04 to −0.09 m during a press; servo chases the receding wall until the arm runs out of reach (tip reference driven to 0.505 m, force decayed to 1 N) |
| TacNav research loop, `docs/HW_ROBUSTNESS.md` §14 | hand_force_v2 MPC-RL | dose-response: >10 N contact ⇒ **+0.113 m** base drift (n=6), corr(force, drift)=+0.427; sustained lateral gate FAILED |

**Correction (2026-08-18, prompted by the owner's question "was the earlier MPC-RL
controller's leg control also velocity-command-only?"). It was — and this overturns the
interface explanation I first wrote here.** Verified in source:

* `locomotion_vel/env_configs/observation_cfgs.py`: the MPC-RL actor's command slot is
  `generated_commands("twist")` — a 3-d v_des plus a gait phase, nothing else.
* `locomotion_pos/.../observation_cfgs.py` states the design decision outright: *"the
  actor's 3-dim command slot carries the attractor's v_des permanently — the policy is a
  goal-blind vel-follower and the goal lives only in the planner layer."*
* The Se2 attractor is a planner-layer outer loop that converts pose error → v_des; the
  policy never sees a pose.
* And during a TacNav press the base command is `_set_goal(anchor, hold=True,
  **standing=True**)`, which sets `is_standing_env` → the attractor outputs **zero
  twist**. Exactly what the deploy mission sends (`send(0,0,0)`).

So **both controllers are velocity-only at the leg interface, and both receive zero twist
during a press.** The deployed stack is not missing an interface the research loop has.

**What actually differs is how stiffly the policy holds station at zero command** — and
that is measurable, not speculative:

| policy | sim | base drift during a press |
|---|---|---|
| vforce decoupled | its OWN mjlab training env (maze eval, n=42) | **2.5 cm median / 4.2 cm p90** |
| vforce decoupled | deploy sim (M3c) | 4.0 / 6.8 / 9.3 cm |
| hand_force_v2 MPC-RL | TacNav research sim | forward presses accepted at 12.8–14.3 N |

vforce gives up 2.5–4.2 cm under press **in its own training simulator**, so the
compliance is substantially a property of that policy; the deploy model gap widens it to
4–9 cm. That is enough to walk the wall out of the arm's usable reach, which is how every
press here died.

The dead zone remains a real and separate blocker — it is why no outer loop can trim
centimetre-scale drift once it starts — but it is not the reason the press regime differs
between the two stacks.

## Option A — brief-tap TacNav MVP

Replace the sustained hold with a tap: creep in, latch the contact event on the first
sample in band, release. Acceptance becomes *peak tap force in band + a named contact
pair*, not a 0.8 s hold.

- **Evidence it works** (different controller and simulator — TacNav research loop,
  `HW_ROBUSTNESS.md` §15): 50/50 taps produced real lateral contacts, **49/50 in the
  10–25 N band** (one at 29.0 N), base drift +0.001 m mean. The whole V3 demo route and
  the belief/PF pipeline already consume tap records (point + normal), so the planner
  needs no redesign.
- **Cost**: small. The press phase becomes a tap cycle; the clear-standoff, unmasked
  creep and live-face logic already written for M3(c) carry over. ~1 day to implement and
  gate in the deploy sim.
- **Risks**: tap peak force is *gait-limited and not tunable* on this platform, so the
  band is a distribution, not a setpoint — with the deploy sim's 20 N jam artifacts, the
  guard's 18 N limit will trip on outliers. Information per touch is lower than a
  sustained hold (no persistent normal estimate).
- **What it does NOT fix**: the hardware blockers stay. The velocity dead zone still
  prevents fine positioning for navigation, and the overspeed runaway class (0.5 → 1.7
  m/s) is untouched. Option A buys a demonstrable MVP, not a safe robot.

## Option B — sustained pressing, with station-keeping

Give the base a position-referenced stance during contact. Three paths, very different
costs:

- **B0 — buy reach margin instead of stiffness (cheapest, no controller change, no
  criterion change).** Every press here died the same way: the base gave up 4–9 cm and the
  wall walked out of the arm's reach (tip reference driven to 0.505 m, its limit). The
  mission approached to a 0.35 m standoff, which leaves almost no margin once the tip's
  ~7 cm collision offset is taken out. Approaching to ~0.25 m puts contact well inside the
  arm's stiff region with ~10 cm of slack to absorb exactly the drift we measured. **~1
  hour, same controller, same criterion** — and it directly tests whether sustained force
  is reachable without any new control work. This is now the first thing to run.
- **B1 — the committed positional step interface (still worth testing, weaker rationale
  than I first claimed).** 学长's joint decoupled stack has `lower_command_mode:
  footplacement`: a 7-d step command in the stance-foot frame where standing = "countdown
  0, target held". Note this is a *new* mechanism, **not** parity with the research loop
  (which is velocity-only, as corrected above), so it is a genuine experiment rather than
  a restoration. Caveat: that checkpoint's 8-d force channel is baked to zero, so force
  would be regulated by EE position alone — which our data suggests is happening anyway
  (we drove `f_cmd` to 16 N and the measured force did not track it). ~0.5 day, committed
  assets only.
- **B2 — add a base admittance/position loop.** The repo's admittance is EE-side (Pink
  tactile ball-follow) and `cartesian_impedance_controller` is an arm controller; there
  is no base equivalent. New control work, weeks, and it must be validated against the
  same velocity blocker.
- **B3 — upstream checkpoint with a stance-hold or position command channel** (Zhenyu).
  Weeks, and a schedule dependency outside our control.

- **What it does fix**: station-keeping also gives fine positioning *during contact*,
  which is half of blocker #1. Navigation's dead zone still needs its own fix.

## B0 RESULT (run 2026-08-18)

**Sweep 1 — approach target 0.45 m (base parked 0.39–0.405, standoff ~0.35 m,
i.e. ~4 cm closer than the failing runs). 1 of 3 held the band for 0.8 s — the
program's first sustained 8–12 N hold.** Drift is the discriminator exactly as
predicted:

| run | base drift during press | peak | longest continuous time in band | verdict |
|---|---|---|---|---|
| 1 | −0.069 m | 10.24 N | 0.11 s | timeout |
| 2 | **−0.019 m** | 20.0 N (spike) | **0.80 s** | **HELD** |
| 3 | −0.061 m | 7.45 N | 0.00 s | timeout |

**Sweep 2 — approach target 0.53 m: cannot get there.** Both valid runs timed out
stalled at x ≈ 0.40–0.42 (the third run's stack failed to launch; note the batch
script reuses `out/mission_run$i.json`, so sweep-1 files 1 and 3 were overwritten —
the console record is the source for those).

**Why it cannot be pushed further** (`scripts/probe_approach_limit.py`, constant
vx = 0.40 through the guard, no yaw or lateral mixing, 40 s): the base advanced
0 → 0.393 m in the first 6 s, then **stalled in x for ~24 s while y drifted from
0.04 to 0.72 m**, finally reaching x = 0.536. Net forward progress 0.013 m/s
against **0.60 m of uncommanded lateral drift**. A pure straight-ahead command
does not produce straight-ahead motion on this platform.

So B0 is **partially validated and capped**: more reach margin does convert
presses into holds, but the margin cannot be increased further because
*approaching* is itself unreliable — the same blocker, one level up. This is a
new measured fact for blocker #1: uncommanded lateral drift comparable in
magnitude to the commanded forward motion.

## B1 RESULT (run 2026-08-18): hypothesis REFUTED

Controlled A/B — same scene (wall placed at the press standoff so neither arm walks),
same press logic, same criterion; only the lower-body policy and stance command differ.
n = 3 per arm. Base drift during the press:

| arm | stance command | run 1 | run 2 | run 3 |
|---|---|---|---|---|
| **A** vforce decoupled | velocity (zero twist) | **+0.0003 m** | **−0.0053 m** | **−0.0138 m** |
| **B** joint decoupled | footplacement (positional, "target held") | force-guard trip before contact | **−0.1505 m** | **−0.2191 m** |

**The positional stance is an order of magnitude worse** (15–22 cm vs ≤1.4 cm). B1 is
refuted: a positional stance command does not reduce press drift on this platform, it
amplifies it. Arm B also struck the wall on contact (20 N) where arm A crept in at 3–4 N,
and needed a hands-clear step at bringup because its HOME arm pose comes up already
loaded against a wall at this standoff.

**The by-product is more useful than the test.** Arm A's drift *when it never walked*
is ≤1.4 cm, against 4–9 cm for the same press after walking to the stance. So drift is
dominated by having recently walked, not by the press itself.

### Follow-up: does settling before the press fix it? No.

TacNav's crab press runs a velocity-gated settle before pressing; the deploy mission had
none. Added one (still < 0.02 m/s for 1.5 s, min 3 s, cap 15 s) and re-ran the full
walking mission ×3: the gate fired properly (3.7–4.5 s), and drift was **−0.302, −0.037,
+0.007 m** — no improvement, and run 1 is the worst drift recorded in the whole program.
Run-to-run variance is now the headline: the same commanded press produces anywhere from
+0.7 cm to −30 cm of base motion.

## Servo-deadband fix + wrench instrumentation (2026-08-18)

Owner-directed, criterion untouched. Two changes: the force servo now aims at the band
**centre** with its own deadband (`--servo-deadband`, default 0.3 N, **configurable, not
hardcoded**; `--f-tol` 2.0 N and `--hold-s` 0.8 s are exposed separately and unchanged —
they are the acceptance criterion), and the full baseline-corrected wrench vector is
logged every tick alongside the scalar magnitude.

**Result of the fix: it works, but it is not sufficient.** Of 4 attempts:
1 **HELD** (0.80 s in band, peak 9.9 N, drift −1.75 cm), 1 timeout, 1 guard trip,
1 stack-launch failure. Before the fix the same configuration held 0 of 3.

**What the wrench vector revealed** (the timeout run): after one brief 3.4 N touch whose
force was mostly *lateral* — wrench (−1.66, −1.97, +2.28) N, i.e. the pressing axis was
the smallest component — the palm force sat at **0.1–0.4 N for the entire remaining 25 s
while the hand swept 10.4 cm forward**. It glanced off and then pressed air. The scalar
magnitude topic could never have shown this.

**The quantified problem underneath** — contact-onset geometry is not repeatable. Across
12 recorded presses, the gap between the modeled wall face and the FK tip position at
first contact:

| | value |
|---|---|
| median | 0.094 m |
| range | −0.002 … 0.131 m |
| **spread** | **133 mm** |
| what the press model assumes | 0.070 m, and it lets the reference travel to face + 0.12 m |

Both runs that ever held had the *smallest* gaps (0.053, −0.002 m: contact essentially at
the modeled face); the failures cluster at 0.076–0.131 m, where the servo's reference cap
sits 20–25 cm beyond the real contact point, so it drives deep, jams (20 N) and loses
contact. **A 133 mm geometric uncertainty is wider than the press's entire working range,
and no force servo can be made reliable on top of it.** Cause is not yet isolated —
candidates are the hand's radial orientation law, elbow configuration moving the collision
capsule relative to the wrist FK point, and base yaw error (measured up to 6°).

**Guard false-positive found in the same batch**: an odometry gap (`stale(odom,0.40s)`)
made the finite-difference speed estimate read 0.73 m/s and latch an `overspeed` trip. The
guard's speed estimator must ignore stale gaps rather than differencing across them.

## Recommendation

**Both cheap experiments have now run, and this is the decision point I said I would
bring back rather than pre-empt.** Summary of what is known:

* the 0.8 s / 8–12 N criterion is **achievable but rare** on this stack: 1 hold in 9
  mission-style presses, and only in the closer-stance configuration;
* **B0 (more reach margin) helps but is capped** — the approach cannot be tightened
  further because straight-ahead commands do not produce straight-ahead motion
  (0.60 m lateral drift per 0.54 m forward);
* **B1 (positional stance) is refuted** — 15–22 cm of drift versus ≤1.4 cm;
* **a settle gate does not help**, and press behaviour is not repeatable run to run
  (+0.7 cm to −30 cm of base motion for the same commanded press).

No cheap fix remains. The realistic paths are now **Option A (tap MVP, ~1 day, buys a
demonstrable MVP but fixes none of the hardware blockers)** or **Option B2/B3 (base
admittance loop, or an upstream checkpoint with a stance-hold channel — weeks, and B3
depends on Zhenyu)**. Both are program-level calls, and both leave the hardware blockers
where they are. I am not proposing a criterion change; the criterion stands until an
explicit decision.

Either way the hardware blockers are unchanged and hardware stays stopped.
