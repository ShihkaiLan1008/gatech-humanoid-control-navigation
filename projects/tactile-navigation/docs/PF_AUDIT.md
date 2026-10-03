# Subtask 1 — global PF audit

Instrumented every tactile update on the R2, R3 and `ambiguous_near_edge`
episodes, and audited both executives for truth leakage.

Reproduce:

```bash
# kinematic (seconds)
PYTHONPATH=src python3 scripts/pf_audit.py --stack v3 --scene r2_lbend --seed 0 \
    --out out/pf_audit/v3_r2_kin
PYTHONPATH=src python3 scripts/plot_pf_audit.py out/pf_audit/v3_r2_kin

# physics (30-55 min per episode; two fit on one RTX 4090)
PYTHONPATH=src:tactile_controller/src MUJOCO_GL=egl \
    tactile_controller/.venv/bin/python scripts/pf_audit.py --stack v3 \
    --scene r3_tpinch --seed 0 --backend mujoco --out out/pf_audit/v3_r3_mj_fixed

# the contact chain, offline, from any of the above
PYTHONPATH=src python3 scripts/contact_frame_audit.py out/pf_audit/v3_r3_mj_fixed

# the localization benchmark (5 maps x 20 seeds x 4 arms, ~25 min)
PYTHONPATH=src python3 scripts/bench_v3.py --seeds 20 --out out/bench_v3
```

## 1. The belief errors are not a filter bug

For every update the recorder evaluates the log-likelihood **the true pose would
have earned**, next to what the filter's own leading mode earned, on identical
evidence (`src/global_nav/pf_diagnostics.py`). That single number separates the
four candidate causes.

| | R2 (`r2_lbend`) | R3 (`r3_tpinch`) |
|---|---|---|
| tactile updates | 11 | 10 |
| truth outside particle support | **0** | **0** |
| min ESS (of 900) | 459 | 462 |
| median ll(truth) − ll(leading) | **0.00 nats** | **0.03 nats** |
| updates carrying zero information | **5 (45%)** | 1 (10%) |
| updates that made the error worse | 2 | 4 |
| final belief error | 0.121 m | 0.057 m |

The truth was inside the particle support on **every single update** of both
episodes, and ESS never fell below half. Nothing was depleted, nothing was
resampled away, and the estimate is in the same frame as the truth.

So the answer to "where does the 0.12 m come from" is none of the four
candidates in the brief. It is **evidence starvation**:

- the median likelihood margin is ~0 nats — the evidence does not distinguish
  the true pose from the filter's answer at all;
- in R2, 5 of 11 updates returned a swept-free ray that penetrated no
  hypothesised wall for *any* pose, so the likelihood was exactly flat;
- the **structural channel contributed 0.000 nats on every update of every
  episode** — it is inert on these maps;
- one touch in eleven (R2, t≈27 s) carried +27 nats and pulled the error from
  0.061 m to 0.007 m in a single update.

Between those rare informative touches, odometry drift accumulates unopposed:
in R2 the error climbs 0.020 → 0.104 m across three consecutive updates that
each changed nothing. `out/pf_audit/v3_r2_kin/pf_audit.png` shows this on one
clock — panel 2's shaded band is the "no information" zone, and 8 of 11 bars sit
inside it.

**Mechanism.** A press on a featureless wall is rank-1 by the observation
model's own construction: it constrains distance *to* the wall and says nothing
about position *along* it. Touch-follow presses flat wall almost exclusively, so
it re-measures the one component that was never in doubt. This is what motivates
Subtask 2: informative geometry (corners, wall ends, bulges, doorway edges) is
where rank-2 evidence exists.

## 2. Truth leakage — one real breach, found and fixed

`tests/global_nav/test_truth_firewall.py` now checks 36 decision methods across
both executives, plus an allowlist pinning exactly which methods may read truth
(logging, odometry bookkeeping, after-the-fact scoring).

It found that **the `arrived` outcome was decided by ground truth in both
executives**:

```python
# Executive._at_goal  (and the same test inside ExecutiveV3.run)
xy_w, _ = self.be.truth_pose()
return norm(xy_w - goal_w) <= self.scene.goal_radius_m
```

The episode terminated the instant the robot's **true** pose entered the goal
radius. A real robot has no such signal. Any belief error large enough to stop
it in the wrong place was masked, provided its true path happened to pass close
enough to the goal.

Measured on 4 scenes × 5 seeds, kinematic, `belief_aware`:

| rule | `arrived` | actually at the goal |
|---|---|---|
| old — stop when the TRUE pose enters the goal radius | 16/20 | not measurable |
| new — stop when the BELIEF enters the goal radius | 16/20 | **8/20** |

The headline count is unchanged; its meaning is not. The pipeline's true
arrival rate on these scenes is **40%, not 80%**. Median final goal distance
0.157 m → 0.239 m, median belief error 0.092 m → 0.134 m: the earlier numbers
were partly an artefact of stopping the clock at the moment of truth-alignment.

`arrived` now means "the robot concluded it was there"; `arrived_truly` in the
summary is the truth-scored answer. The gap between them is the quantity this
programme has to close.

**This overrules a prior deliberate decision.** `test_arrival_is_decided_on_
realized_base_state` previously *required* `truth_pose()` in `_at_goal`, on the
reading that it is the harness's success detector rather than a robot decision.
That test has been replaced, with the measurement above recorded in its
docstring.

## 3. What was NOT wrong

- No frame or registration mismatch: estimate and truth are in the same frame,
  and `truth → nearest particle` stays below 0.015 m throughout R2.
- No particle depletion: ESS/N ∈ [0.51, 1.0] across both episodes.
- No resampling loss: 5 resamples in each episode, none of which moved the truth
  outside support.
- No mode-clustering artefact: both episodes are single-mode after the first
  station.

## 4. On physics the verdict changed — and the physics was never the problem

**This section previously reported that the MuJoCo runs were "actively misled"
by geometry misregistered by roughly half a metre. That was wrong, and the
defect was in this audit's own instrumentation.** The corrected result is below;
the original reading is kept at the end so the retraction is legible.

### The bug

`scripts/pf_audit.py::_install_recorder` carried the truth **position** into the
belief's map frame and handed over the truth **yaw** in world:

```python
xy, yaw = be.truth_pose()
if hasattr(be, "world_point_to_map"):
    xy = be.world_point_to_map(np.asarray(xy, float))
return np.asarray(xy, float), float(yaw)     # <- yaw never converted
```

A pose whose position is mapped and whose heading is not is the true pose
**rotated about itself** by `pi/2 − yaw0`. On `r3_tpinch`, seed 0, the spawn
heading is `yaw0 = −0.302 rad`, so every truth pose the recorder scored was
swung **+1.873 rad = 107.3°**. A contact reached 0.44 m in front of the base is
therefore reconstructed `2·0.44·sin(1.873/2) = 0.71 m` away from where the hand
actually touched. Its distance to the *nearest* mapped surface is smaller than
that displacement — the swung point often lands near some other wall — and
measures 0.33–0.67 m, which is the 0.48–0.62 m residual §4 read as a
registration fault.

Three things pin it, none of them requiring a re-run:

- the world→map rotation recovered by Procrustes from the six recorded contact
  poses of `v3_r3_mj2` is **exact** — 0.00000 m fit residual on all six — and
  equals **+1.8730 rad**;
- `yaw_err_after_rad` was **+1.87 rad on every single update** of the episode
  while `truth_nn_dist` stayed at 0.001–0.009 m. A filter 3 mm right in
  position and 107° wrong in heading, consistently, for a whole episode it
  completes successfully, is not a filter;
- `KinematicBackend.world_point_to_map` is the **identity** and it spawns at
  map yaw `pi/2`, so the missing conversion is a no-op there. That is the whole
  of the "kinematic says starvation, physics says misregistration" split — the
  bug could only bite the backend that has a non-trivial map frame.

Fixed by adding the missing inverse, `world_yaw_to_map`, to both backends and
using it in the harness (`tests/execution/test_frame_transforms.py`, 29 cases).

### The corrected physics result

Both scenes re-run on MuJoCo with the fix — `out/pf_audit/v3_r3_mj_fixed/` and
`out/pf_audit/v3_r2_mj_fixed/`:

| | r3 before (bug) | **r3 after** | **r2 after** | kinematic r3 |
|---|---|---|---|---|
| updates | 6 | 5 | 6 | 10 |
| median ll(truth) − ll(leading) | **−136.6 nats** | **−0.20 nats** | **+0.57 nats** | 0.03 nats |
| `model_prefers_wrong_pose` | **6 of 6** | **0** | **0** | 0 |
| `evidence_uninformative` | 0 | 4 of 5 | 4 of 6 | 7 of 10 |
| truth → nearest particle | 0.003–0.009 m | 0.001–0.007 m | 0.001–0.004 m | ≤ 0.015 m |
| final belief error | 0.057 m | 0.244 m | 0.035 m | 0.057 m |

The measurement model does **not** prefer a wrong pose on physics. It never did.
The physical verdict is now the same as the kinematic one — **evidence
starvation** — which makes §1 the single reading for both backends, and makes
the physical belief numbers in this programme sound.

R2's two `filter_ignored_better_pose` updates (margins +31.6 and +73.9 nats)
are not a filter defect: the verdict is scored on the state *before* the
update, and both updates then moved the estimate onto the truth (0.107 → 0.035
and 0.107 → 0.022 m). They are the informative touches §1 described — the ones
that pull the error down in a single step — seen from the wrong side of the
clock.

### The contact chain, end to end

`scripts/contact_frame_audit.py` re-derives every stage from a recorded episode
(no simulator, no GPU), medians over the contacts of each fixed physics run:

| | reach \|p_body\| | residual at truth | residual at leading | normal error | residual the yaw bug made |
|---|---|---|---|---|---|
| **r3_tpinch** | 0.441 m | **0.009 m** | 0.029 m | 11.7° | 0.496 m |
| **r2_lbend** | 0.441 m | **0.007 m** | 0.018 m | 8.5° | 0.536 m |

Per-contact the residual at truth spans 0.004–0.022 m across both episodes, and
the residual the unmapped yaw manufactured spans 0.333–0.666 m — reproducing,
from the same data, the 0.48–0.62 m band this section originally attributed to
misregistered geometry.

**The `_probe_ee_w()` hypothesis is rejected, with a number.** `_probe_ee_w()`
returns the *wrist link origin*, not the surface contact point, and
`FACE_OFFSET = 0.0` means the controller's kiss model treats the two as the
same. That approximation is real but small: reconstructed from the true pose,
those wrist points land a median **9 mm** from the mapped surface, against
`sigma_contact_m = 0.035`. Replacing it with the true MuJoCo surface contact
location can buy at most 9 mm on a 35 mm noise scale, so it is not what was
costing 144 nats and it is not worth the plumbing. The reach is 0.39–0.45 m
throughout, inside the measured 0.45 m hand_force_v2 limit.

One item does deserve attention: `_contact_body_frame` asserts a hardcoded
body-frame normal `(-1, 0)` on every contact, and the measured angle between
that and the mapped surface normal is **6–14°** (median 11.7° on r3, 8.5° on
r2) against `sigma_normal_rad = 0.22` (12.6°). It fits, but it is spending
essentially the whole normal budget on a constant that could be measured.

### What the original section said

> So on physics the belief is not evidence-starved, it is **actively misled**:
> the measurement model prefers a wrong pose, consistently, on every contact.
> Something between the injected MuJoCo geometry, the `world_point_to_map`
> transform, and the body-frame contact point is misregistered by roughly half
> a metre.

The instinct that named the three suspects was right; the fault was in none of
them, because the pose being scored was not the pose the harness thought it was.
The lesson worth keeping: an instrument that reads a 144-nat anomaly should be
audited before the system it is pointed at.

## 5. Where the seconds go (Subtask 3, profiling)

`scripts/profile_episode.py`, on the shipped physical episodes:

| episode | total | LOCO | STAB | PROBE | valid/attempts | wasted on invalid | nav |
|---|---|---|---|---|---|---|---|
| `mj_r3` | 434 s | 144 s | 61 s | **229 s (53%)** | 11/18 | **117 s** | 0.07 m/s |
| `mj_r2` | 474 s | 153 s | 57 s | **263 s (56%)** | 5/17 | **222 s** | 0.08 m/s |
| `mj_m2` | 84 s | 52 s | 7 s | 26 s (30%) | 2/4 | 18 s | 0.12 m/s |

Inside PROBE, split by whether the hand was loaded:

| episode | pressing + regulating | arm in transit (approach + retract) |
|---|---|---|
| `mj_r3` | 52 s | **177 s** |
| `mj_r2` | 21 s | **242 s** |

Three conclusions, in priority order:

1. **The arm is in transit, not pressing.** 86% (r3) and 92% (r2) of probe time
   is approach and retract with an unloaded hand, while the base stands still.
   Pre-positioning the hand during locomotion is the single largest available
   saving — roughly 40% of total episode time on r2.
2. **Invalid probes cost 47% of the r2 episode** (222 s of 474 s) and returned
   nothing. Rejecting unreachable and grazing stances *before* execution is the
   second lever.
3. **Force regulation is not the bottleneck** (21–52 s). Do not optimise it, and
   do not touch walking speed: at 0.07–0.08 m/s locomotion is only a third of
   the clock, and the brief's own rule is that probe overhead comes first.

## 6. Geometric route generation (Subtask 5)

**Audit.** Wall-follow lanes are already derived automatically from the wall list
(`maps_v3.followable`), so lanes were never the gap. The gap is free-space
travel: `graph3.build_graph3` places waypoints on a 1.15 m grid, only inside
hand-declared room rectangles, joined by straight hops capped at 2.6 m, with no
smoothing. On a new floorplan that fails when a passage is narrower than the grid
pitch, when free space lies outside every declared rectangle, or whenever the
route needs to be a path rather than a hop chain.

**Added.** `src/planning/geometric.py` — inflate, A*, line-of-sight shortcut,
smooth, resample. The map enters as a *clearance function*, so the same planner
serves the lane-graph `MapV3` specs and the belief's `Floorplan` without either
importing the other. `plan_alternatives()` returns several genuinely different
routes rather than a winner, so shortest-path cost does not decide tactile
behaviour.

Verified (12 tests, `tests/planning/test_geometric.py`): routes found on all
three shipped maps with clearance ≥ 0.34 m; a 1.20 m doorway is routed and a
0.72 m one is **refused** (0.62 m of free width after wall thickness is less than
the G1 needs — refusing is correct); maximum turn angle < 40° so the route stays
smooth; endpoints inside the inflation band snap to free space.

**Wired in.** `build_graph3` now samples the map's free space once
(`transit_grid`) and, wherever the straight-hop test refuses a node pair, asks
`plan_route_on` for a path instead. The resulting `FREE_TRANSIT` edge carries a
polyline, and `ExecutiveV3._run_transit` walks its interior waypoints rather
than driving straight at the far node.

It is **strictly additive**, and that is tested rather than asserted: every
straight connector on all five shipped maps survives unchanged, and the route
`astar3` plans is identical in kind and in cost on every one of them. Across
5 maps × 2 seeds, kinematic, every episode is bit-identical — same outcome,
same belief error, same probe count, same sim time. 70 bent connectors are
added across the five maps (r4_winding alone gains 38) and none of them changes
a shipped answer.

What it buys, measured on a one-room map with a centred doorway:

| free width | hop chain | geometric |
|---|---|---|
| 0.50 m | refused | refused (correct — the base does not fit) |
| **0.56 m** | **refused** | **routed** (exactly `2 × LANE_CLEAR`) |
| 0.70 m | refused | routed |
| 0.80 m | routed | routed |

The hop chain needs 0.80 m of free width to admit a doorway the base walks
through at 0.56 m — 43% more conservative than the clearance contract, not
because the geometry is tight but because a 1.15 m waypoint pitch has to get
lucky and land a straight segment inside the gap. The geometric layer admits
exactly the contract and nothing looser.

Two things the wiring had to fix to be honest about that: a bent connector is
verified against the **continuous** clearance function before it is added (the
occupancy grid answers at 0.08 m cell resolution, and a route it accepts can
graze ~0.06 m inside the band), and a connector may not start or end at a node
that is itself inside the inflation band — which lane nodes can be, since a
lane hugs the wall it follows. Without the second rule r4_winding gained a
connector with 0.275 m clearance against a 0.28 m contract.

Still hand-authored, deliberately: the room rectangles. They are the only
statement this map format makes about where the building ends, so `in_rooms`
stays inside the clearance function — dropping it would let a route leave
through the gap between two wall segments. What the geometric layer removes is
the waypoint pitch, not the boundary.

## 7. Probe latency: 21% of the probes were provably worthless

§5 ranked the two available savings. The second one — "rejecting unreachable and
grazing stances *before* execution" — turns out to have a form that needs no
physics change at all and no new machinery.

`_choose_offset` already predicts, for every fan offset and every live
hypothesis, whether the press ray reaches a mapped surface. When the answer is
**no for every offset and every hypothesis**, it scored that case 0.25 ("a
certified free ray") and pressed anyway. That press is provably uninformative:
the swept-free ray reaches no further than the hand does, so if no hypothesis
puts a surface within reach then the ray penetrates no hypothesised wall for
any pose and its likelihood is exactly flat. This is the same event §1 counted
— "5 of 11 updates returned a swept-free ray that penetrated no hypothesised
wall for *any* pose" — and §5 priced at 222 s of R2's 474 s.

`skip_uninformative_probes` (default on) declines those.

**Correction (subtask 3).** "Provably uninformative" was wrong, and the error is
instructive: it argues over the handful of live mode CENTRES, while
`swept_free_loglik` is applied to every PARTICLE and the cloud is spread around
those centres. Measured with `scripts/diagnose_free_tail.py` on three maps, a
ray that every mode agrees about still spans a **median 2.53-2.57 nats** across
the particles and penalises **9-14%** of them.

The tail-pruning hypothesis -- that such a ray quietly eats the truth-supporting
tail -- was tested and **disconfirmed**. The truth-nearest particle is penalised
a median of **0.000 nats** (worst -1.398 over 119 r1 updates), and truth-mode
mass moves by mean **+0.006** (r1), **+0.047** (r3), **+0.279** (r5). The free
channel is weak, high-variance and slightly *helpful*.

So the rule is a latency trade, correctly stated: it buys 21.4% of the probe
budget and costs a little evidence. On the benchmark that is +3 true arrivals
overall and **-1 on r1_gallery**, which is exactly the map whose belief stays
broad (minor mass 0.48 at the end) and therefore has the most to gain from
within-mode sharpening. No gating fix is justified; the premise needed fixing,
not the filter.

Measured on
`scripts/bench_v3.py`, 5 maps × 20 seeds, paired against the same seeds:

| | shipped | **skip** | delta |
|---|---|---|---|
| probes, total | 1031 | **810** | **−21.4%** |
| sim time, total | 4750 s | 4479 s | −5.7% |
| truly arrived | 65/100 | 68/100 | +10 / −7 |
| median belief error | 0.089 m | 0.092 m | −0.006 m (median paired) |

**21% of every probe budget was buying nothing**, and declining it costs no
accuracy. The sim-time saving is only 5.7% on the kinematic stand-in because
there a probe is nearly free; on physics, where §5 measured probes at 53–56% of
the episode and ~12 s each, the same 21% is worth roughly 11% of wall-clock.

One caveat worth stating: this is a *latency* result, not an accuracy one. The
true-arrival change (+10/−7) and the belief-error change (−0.006 m) are both
inside the noise of 100 paired episodes. The claim is "a fifth of the probes
were free to delete", not "deleting them helps".

The mechanism is not fully isolated. Path length does **not** track the
improvement — on `r1_gallery` seed 2 the skipping run walked *further*
(13.40 → 14.29 m) and still improved (0.266 → 0.055 m) — which rules out "less
walking, less odometry drift". What does track it is the number of swept-free
observations ingested (6–10 → 1–2). The remaining hypothesis, untested, is that
a free ray which discriminates nothing among the *live modes* still reweights
the whole particle cloud, and prunes tail particles that sometimes hold the
truth.

§5's first and larger lever — the arm is in transit 86–92% of probe time —
is **not** addressed here. The transit is almost entirely wall-normal travel
(park 0.14 m → contact ≈0.53 m → park, at 0.06 m/s out and 0.125 m/s back
≈ 9.6 s, against the 9.8 s measured), so the only ways to cut it are to hold
the hand out between presses or to approach faster. Both change what the hand
does while the base moves, which is the exact failure mode this programme has
already paid for (a parked hand dragged along a wall reported 8.58 N while the
audit saw 35.2 N). Neither should land without a physics run to validate it.

## 8. Active landmark INSPECT: built, measured, does not pay

§1's remedy for rank-1 evidence was informative geometry. INSPECT implements
the smallest version of it: at a stance whose press fan straddles a mapped
discontinuity, spend a *second* press on the far side, so the pair (contact,
free) carries the along-wall coordinate that neither half carries alone.

It works as designed and it does not help. Two triggers, both measured on
5 maps × 20 seeds, paired:

| trigger | fires on | belief error | truly arrived | probes |
|---|---|---|---|---|
| the fan straddles an edge | 70/100 episodes | 33 better, 37 worse (median +0.010 m) | +5 / −2 | +10% |
| ...**and** the live hypotheses disagree about it | 7/100 episodes | unchanged on 93/100 | +0 / −1 | +0.6% |

Loose, it fires nearly everywhere and is a coin flip — which is the definition
of a behaviour that is not active sensing but simply "press twice". Tight, it is
correct and almost never applicable. All of the loose trigger's gain was on
`r5_twinbay`, the one genuinely ambiguous map.

Default **off**, code and flag kept (`ExecV3Config.inspect`).

The benchmark says what is actually missing, and it is not a better local
chooser. On `r5_twinbay` — 2/20 true arrivals, 3.4 m median belief error, the
map where localization fails hardest — the stack takes **three probes in a
whole episode** and finishes early (36 s against a 44 s median). It is not
choosing badly among the presses it makes; it is not making any. The behaviour
worth building is one that **diverts** to mapped geometry when the belief is
bad. `graph3` already carries the raw material (`Edge3.informative` marks lane
chunks that pass a wall feature, and `edge_cost` already prices them at
`shrink = 0.55` against `0.85`), so the missing piece is a station that acts on
it rather than a fan offset that reads it.

## 9. What the scene set actually measures

`scripts/bench_v3.py` scores 5 maps × 20 seeds × 4 arms and calls a scene
localization-critical when tactile localization beats open-loop odometry on it.

| arm | arrived | truly arrived | median belief err | median probes | median sim |
|---|---|---|---|---|---|
| `shipped` | 93/100 | **65/100** | 0.089 m | 10 | 44 s |
| `skip` | 95/100 | **68/100** | 0.092 m | 7 | 43 s |
| `full` (skip+INSPECT) | 95/100 | **67/100** | 0.091 m | 7 | 43 s |
| `no_touch` | 70/100 | **4/100** | 0.465 m | 0 | 32 s |

**All five scenes are localization-critical** — open-loop odometry truly
arrives 4 times in 100 against 65–68. So the scene set does measure what the
programme is for, and §2's arrived-vs-truly gap is the metric to move.

That was not visible before this run, because the ablation did not ablate.
`max_probes_total` was enforced in the touch-follow cadence but **not** in
`_confirm_corner`, so a "no touch" arm still took three corner presses per
episode — enough to localize, and enough to make touching look unnecessary. An
ablation that does not ablate cannot answer the question it is asked.

Two scenes are where the remaining error lives:

- `r5_twinbay`: 1–3/20 truly arrived, 3.4 m belief error, and the belief is
  *confident* — 20/20 episodes report `arrived`. This is the bimodal twin-bay
  ambiguity resolving to the wrong bay and never being challenged.
- `r1_gallery`: 11–12/20, and the only scene where the skip rule costs
  something (12 → 11 truly arrived, 0.070 → 0.147 m median belief error). Worth
  understanding before the rule is taken as universal.

## 11. r5_twinbay: diagnosed, and the last classical mechanism

`r5_twinbay` was the outstanding failure of §9: 3/20 truly arrived, 17/20
false arrivals, 3.4 m belief error, and `arrived` reported 20/20.  Traced over
20 seeds (`scripts/bench_v3.py` and the diagnosis recorded here):

| | |
|---|---|
| false arrivals | **17/20** |
| truth-nearest mode ever lost | **1/20** |
| collapsed to one mode | 1/20 |
| median final mass on the TRUE mode | 0.372, median 3 modes still alive |
| wrong mode leads | from t=0 on every seed (the prior gives it 0.55) |
| median probes | **3** |
| informative edges on the planned route | **0 of the 2 in the graph** |

**It is not mode collapse, not route gating, not unreachability.**  The chain:
the prior puts 0.55 on a pose 3.4 m ahead, so the planner plans from *there* and
the route is short -> 3 presses, all on a south wall that is aliased by
construction (two identical bays at exactly the 3.4 m mode pitch) -> nothing
separates the modes -> `_relatch()` hands navigation to the heaviest mode at
every edge boundary -> on seed 0 the robot is physically AT the true goal
(x = 6.93) at t = 28 s, relatches to the wrong mode at 10.25, concludes it
overshot, and walks **back 3.4 m** to arrive confidently wrong.

The disambiguating geometry exists and is decisive: graph edges 9 and 10 are
already marked `informative`, both reachable, and a press from the north lane
predicts 0.415/0.535 m under the correct hypothesis against 0.655 m under the
+3.4 m one -- 3-7 sigma at `sigma_contact_m = 0.035`.  The planner refuses the
detour because `edge_cost` shrinks sigma multiplicatively for *any* touch-follow
edge (0.85 plain, 0.55 informative), so it expects three plain south presses to
take sigma 1.69 -> 0.61.  It has no notion of *which* geometry separates the
live hypotheses.

**Landmark diversion** supplies exactly that, and is accepted: four of five maps
are identical on every column (zero diversions), r5 improves on every axis and
gets cheaper.  Mechanism, results and the seed-0 trace are in
`CLASSICAL.md` §5.

Three defects the measurements caught while building it, each worth keeping:

1. the hypothesis offset must be **added** to a commanded stance
   (`stance + (hyp_i - bel)`); `_ray(delta=)` shifts the map the other way, and
   with that sign the aliased south wall scored a perfect 0.25;
2. the top-scoring stances put a minority hypothesis **2 m outside the
   building** -- target *and path* must be legal under every live hypothesis;
3. scoring refutation in metres preferred stances whose only evidence is a free
   sweep.  Priced in nats per channel, a free sweep can never clear the bar on
   its own -- which is right, because it cannot.

## 10. Open items

Closed since the first pass: the MuJoCo re-runs of R2 and R3 are done and both
pass (§4); 45% of R2's updates returning no information is now declined rather
than executed (§7); the geometric layer is wired in (§6).

Still open, in the order the benchmark ranks them:

- **`r5_twinbay` is improved but not solved: 6/20.** Diagnosed in §11 and
  addressed by landmark diversion. The remaining ceiling is precise: the
  diversion resolves the ambiguity only when its press lands a **contact**
  (9 of 21 diversions). When the correct hypothesis is true the robot ends
  mid-room and sweeps free, and the capped free channel does not refute the
  wrong mode — on seed 3 it moved mass *onto* it (0.547 → 0.821). 6 of 21
  diversions are also still `unreachable`. Obtaining a contact in **both**
  branches is the smallest next step.
- **`r1_gallery` is where the probe-skip rule costs something** (12 → 11 truly
  arrived, median belief error 0.070 → 0.147 m) while helping `r4_winding`
  (12 → 15). The rule is a net win overall and is provably deleting flat
  evidence, so the r1 regression needs a cause before the rule is treated as
  universal.
- **The swept-free channel is not merely inert — it may be mildly harmful.**
  Skipping the presses that produce uninformative free rays changed belief error
  on 100 of 100 episodes and *improved* it slightly, and path length does not
  explain it (§7). The untested hypothesis is tail pruning: a ray that
  discriminates nothing among the live modes still reweights the whole cloud.
  This is a cheap experiment on the kinematic stack and it has not been run.
- **The structural channel is still fully inert** (0.000 nats on every update of
  every episode, both backends). It has not been removed, on the argument that
  it may matter on tighter geometry — but nothing in this scene set tests that.
- **The contact normal is a constant.** `_contact_body_frame` asserts
  `(-1, 0)`, and the measured error against the mapped surface normal is 6–14°
  against a 12.6° sigma (§4). It fits today; it will not on a map with
  shallower incidence, and the wrist quaternion needed to measure it is already
  returned by `measured_ee_pose_w`.
- **§5's larger latency lever is untouched.** 86–92% of probe time is
  wall-normal arm transit, and every way to cut it changes what the hand does
  while the base moves. It needs a physics run to validate, and none was spent
  on it here.
