# Map-aligned global tactile localization belief — V0

`src/global_nav/` · 2 507 lines of module, 755 lines of test, 48 tests passing.

A multi-hypothesis SE(2) navigation belief over a **known** floorplan, driven by noisy
odometry, realized tactile contacts, and swept-free volumes. It estimates; it does not
decide. No path planning, no MuJoCo, no learning, no world model — as scoped.

**Read section 7 before building a planner on this.** Four of the seven items there
change what a planner would be entitled to assume.

---

## 1. What a particle is

Each particle carries four things, and the reason this is a particle filter rather than
an EKF is that two are discrete and one of the continuous pairs is not jointly
identifiable from a single measurement:

| field | meaning |
|---|---|
| `xy`, `theta` | SE(2) pose in the map frame |
| `edge` | which route-graph corridor the robot is in (discrete) |
| `assoc` | which map entity the last distinctive contact came from (discrete) |
| `reg` | local registration offset of the drawing from the world (continuous) |

Touching a flat wall constrains **one linear combination** of pose and registration:
*"I am 5 cm nearer than I thought"* and *"the wall is drawn 5 cm nearer"* are the same
measurement. A Gaussian filter must resolve that split immediately and will resolve it
wrongly. Measured, in `test_flat_wall_leaves_pose_and_registration_ambiguous`: with the
offset frozen a wall touch cuts the along-normal pose spread from 0.20 m to ~0.03 m;
with the offset free it cuts it only to 0.141 m — a factor of √2, exactly what a ridge
posterior predicts — and leaves a +0.6 correlation between pose error and offset.

---

## 2. The benchmark floorplan

`benchmark_floorplan()` — a staircase corridor, 1.6 m wide, 0.15 m slabs, 15 m of route:

```
                                      y=4.8 +--------------------------+
      corridor C (edge 4)                   |   -> -> -> -> -> goal    |
                                      y=3.2 +-----+--------------------+
                                            |  ^  |
      corridor B (edge 2)                   |  ^  |        J2 = (6,4) RIGHT turn
                                            |  ^  |
                         y=0.8  +-----+ +---+  ^  +   <- north wall has TWO gaps
      corridor A (edges 0,1)    |     | |       ^
                                |  D  | |    o  ^      D = dead-end decoy alcove
                         y=-0.8 +-----+-+-------+      o = pillar (unique landmark)
                               x=2.2  3.8   4.5 6.0    J1 = (6,0) LEFT turn
```

Three properties are deliberate, and each is a test:

- **Two corridor turns.** J1 is a left turn, J2 a right turn; the signs come from the
  edges' bearings, never from a hard-coded "left".
- **Repeated / ambiguous corners.** The north wall terminates at x = 2.2 **and** at
  x = 5.2. Both carry signature `north_terminus`, so under touch they are the same
  measurement, 3.0 m apart on the same corridor.
- **A wrong turn that is reachable.** The first terminus opens onto a dead-end alcove
  that is a legal successor of edge 0 in the route graph — so a robot that thinks it is
  3 m further along turns into it.
- **One box/pillar landmark**, at x = 4.5, the only body with signature `pillar`, sitting
  off the centerline so reaching it costs a deliberate probe.

**Tactile stations are configurable** (`TactileStation`, `ProbeFan`): an arclength, and a
set of ray fans in the body frame. The benchmark fills them from a fixed, route-agnostic
schedule — every 1.0 m, sweeping left/right/ahead — because choosing *where* to probe is
the planner's job and is out of scope. `TactileStation` is the interface it would write to.

---

## 3. The three measurement channels

**Realized contact.** A probe ray that touches something returns the contact point in the
*body* frame plus the measured surface normal and a `signature` — what touch could tell
about the thing it hit. Signature `surface` (a featureless wall) is scored against the
nearest surface anywhere: **rank 1**, it says how far the wall is and nothing about where
you are along it. Any other signature is scored against the map entities that share it: a
point feature gives **rank 2**, pinning both position components, but only for particles
that bet on the right candidate.

**Swept-free.** A ray that reaches its limit having touched nothing certifies its length as
free, and penalizes any particle whose hypothesized map intrudes into it. This is the only
channel that can rule a hypothesis out without touching anything, and on its own it cuts
the mean position error by a factor of 2.5 (section 5).

**Structural prior.** A robot is not inside a wall, and a robot that believes it is in
corridor D is not standing in corridor A. This is not a measurement; it is the coupling
that lets continuous evidence correct the discrete `edge` latent. It is ablated separately
so the tactile results are attributable — see `structural_only`.

### Three modelling decisions worth naming

**One ray is one measurement.** A swept-free ray is scored by its *worst* violation, not by
summing over its samples. Summing lets the sample spacing — a filter parameter with no
information in it — inflate confidence. `test_one_ray_is_one_measurement_not_one_per_sample`
doubles a ray's samples and asserts the log-likelihood is bit-identical.

**An ambiguous gate is not a deadlock.** When several map features could have produced a
contact, every particle *commits* to one candidate sampled from the candidate posterior,
and its weight is multiplied by the **marginal** over candidates. The population carries
the ambiguity. The alternative that suggests itself — *"more than one candidate in the
gate, so make no update"* — discards the measurement exactly when it is most informative;
it is a failure mode this codebase has paid for before, and
`test_an_ambiguous_gate_does_not_deadlock_it_splits_the_population` is the guard.

**Multi-hypothesis preservation is bought, not assumed.** Resample only below
`ess_frac · N`; systematic resampling (guarantees a mode holding fraction *f* at least
`floor(f·N)` survivors, where multinomial resampling occasionally deletes it outright);
roughening afterwards. `test_symmetric_evidence_never_breaks_a_fifty_fifty_split` feeds
12 updates that are provably identical under both hypotheses and asserts the split does
not move.

---

## 4. Outputs

Per step, `ParticleBelief.snapshot()` logs: particles (sampled for the figures), pose mean
and 3×3 covariance, per-axis and yaw std, **route-edge posterior**, **landmark/association
posterior**, **registration estimate and error**, **wrong-turn probability**, ESS, mode
count and per-mode mass, resample and outlier counts.

The mean and covariance are reported because a consumer will ask for them, **not** because
they are a sufficient summary. When the belief is bimodal the mean sits between the modes,
where no particle lives, and the covariance reports a confidence the belief does not have.
Measured: 1.59 m mean error while both modes are individually within 0.1 m of a
hypothesis. `cluster_modes` / `n_modes` exist so a consumer can tell when that is happening
instead of being handed a Gaussian and left to trust it.

---

## 5. Validation

12 seeds, 2000 particles, identical scenario except for the enabled channels. Position
error in metres, ± 1 s.e.m.

| condition | mean error over run | final error | final 1σ | reg. error | min ESS |
|---|---|---|---|---|---|
| `odometry_only` | 0.1890 ± 0.0295 | 0.5596 ± 0.1140 | 0.913 | 0.072 | 2000 |
| `structural_only` | 0.1893 ± 0.0299 | 0.5609 ± 0.1145 | 0.749 | 0.070 | 1709 |
| `contact_only` | 0.0733 ± 0.0039 | 0.0818 ± 0.0110 | 0.122 | 0.077 | 1293 |
| `swept_free_only` | 0.0768 ± 0.0081 | 0.1241 ± 0.0266 | 0.182 | 0.056 | 1104 |
| `combined` | **0.0620 ± 0.0055** | **0.0630 ± 0.0087** | 0.132 | 0.058 | 1406 |

The channels share seeds, so the comparisons that matter are **paired**:

| paired difference (mean error) | Δ | significance |
|---|---|---|
| `combined` − `odometry_only` | −0.1270 ± 0.0317 | **−4.0 s.e.** |
| `combined` − `contact_only` | −0.0113 ± 0.0040 | **−2.8 s.e.** |
| `combined` − `swept_free_only` | −0.0148 ± 0.0101 | −1.5 s.e. (not resolved) |
| `contact_only` − `swept_free_only` | −0.0036 ± 0.0096 | −0.4 s.e. (not resolved) |
| `structural_only` − `odometry_only` | +0.0004 ± 0.0022 | +0.2 s.e. (**no effect**) |

**1 · Odometry-only drift.** Error grows from ~0.005 m (the Monte-Carlo error of the prior
mean) to 0.56 m over 15 m, and the reported spread grows with it to 0.91 m final 1σ; the
3σ envelope holds on >95 % of steps across 4 seeds with a median error/σ ratio below 1.5.
It drifts, and it knows it.

**2 · Contact-only.** 2.6× better than dead reckoning on mean error, 6.8× on final error,
and the reported spread drops 7.5×. Isolated: the rank-1 collapse of a flat-wall touch, and
the rank-2 collapse of a corner touch, are asserted directly in `test_belief.py`.

**3 · Swept-free-only.** 2.5× better than dead reckoning on mean error. **Touching nothing
localizes**, and on this floorplan it is statistically indistinguishable from touching
something.

**4 · Combined.** Best on both metrics. The gain over `contact_only` is real but small
(−0.011 m, 2.8 s.e. paired) — the two channels are largely *redundant here*, because a
swept-free ray finding no wall where one was expected and a contact 20 cm away say nearly
the same thing. Separating them properly needs a scene where they do not.

**The `structural_only` control earns its place.** Knowing you are inside a corridor,
without touching it, buys **nothing at all** (+0.2 s.e. — a null result to three decimal
places). Every metre of the improvement above is bought by touch, not by the map.

**5 · Ambiguous landmarks.** Prior: 50/50 over the two identical wall termini — not a wide
Gaussian centred on the gap between them, where the robot certainly is not. Truth is at
x = 5.2. The belief carries both hypotheses at 52 % / 42 % and then 51 % / 48 % across two
stations and 1.0 m of travel, with full ESS and a mean 1.59 m from truth, then collapses to
one mode within 0.04 m of truth the moment evidence distinguishes them.

*What* resolves it is identified, not assumed: with the full fan set, the **forward** probe
finds corridor A's end wall at 0.83 m and settles it in one station; withhold that fan
(`ambiguous_side_only`) and the ambiguity survives longer and is broken instead by the east
probe at the turn. `test_the_forward_probe_is_what_resolves_the_ambiguity` asserts the
causal direction by comparison.

**6 · Incorrect association rejection.** Contacts carrying a signature the map has no entity
for are rejected **2/2 on every one of 6 seeds**, and the run under attack ends within
0.017 m of the clean run — the phantom changes nothing. Meanwhile `n_outliers == 0` across
`combined`, `contact_only` and `ambiguous_side_only`: the gate fires on impossible
measurements, not merely ambiguous ones.

The gate is **mass-based**, and this was a real bug found in development: testing the single
best-fitting particle accepts almost anything, because with N particles the best one is a
1-in-N extreme. A phantom contact 0.78 m from any wall was accepted at N = 1000 by the one
particle 3.5σ out. The gate now asks how much belief *mass* passes, at a 5 % threshold —
1 % admitted a phantom that a diffuse (σ = 0.25 m) belief gave 1.4 % support, and admitting
it would have moved the other 98.6 % of the mass onto that tail.

### Registration observability

12 seeds, 1000 particles, a known offset fed in and measured back out:

| true ‖r‖ (m) | residual (m) | recovered |
|---|---|---|
| 0.00 | 0.075 ± 0.037 | — |
| 0.05 | 0.083 ± 0.047 | **worse than doing nothing** |
| 0.10 | 0.053 ± 0.037 | 47 % |
| 0.20 | 0.080 ± 0.056 | 60 % |
| 0.30 | 0.057 ± 0.026 | 81 % |
| 0.40 | 0.057 ± 0.040 | 86 % |

The residual sits at 0.05–0.08 m **regardless of the true offset**: the estimator has a
floor, and it crosses the do-nothing line at ‖r‖ ≈ 0.07 m — below that it adds noise rather
than removing bias. This is the ridge from
section 1 showing up as a number — contacts constrain pose ⊖ offset, only odometry
separates them, and odometry gets worse with distance walked.

---

## 6. Reproduction

```bash
export PYTHONPATH=src           # or: pip install -e .

# tests (48, ~3 min)
python3 -m pytest tests/global_nav/ -q

# full benchmark: all 8 scenarios, traces, summaries, figures  (~6 min)
python3 -m global_nav.run_benchmark --out out/global_tacnav \
        --particles 2000 --seeds 0 1 2 3 4 5 6 7 8 9 11 12

# quick pass, no multi-seed sweeps (~40 s)
python3 -m global_nav.run_benchmark --out /tmp/gt --particles 800 --skip-sweeps
```

Deterministic given `--seed`: seed 0 reproduces bit-identically across the single-scenario
and multi-seed code paths.

Outputs in `out/global_tacnav/`:

| file | what it shows |
|---|---|
| `ablation.png` | the channel ablation, s.e.m. error bars |
| `registration_observability.png` | recovered offset vs true offset, against the do-nothing line |
| `ambiguity_ambiguous_side_only.png` | the bimodal cloud, its mass split, and the 2σ ellipse spanning both modes |
| `map_<scenario>.png` | particles, estimate, truth and the true (offset) walls, per scenario |
| `timeseries_<scenario>.png` | error vs claimed 1σ, registration error, wrong-turn probability, route-edge posterior, ESS — one shared arclength axis |
| `trace_<scenario>.json`, `summary.json` | every logged quantity |

---

## 7. Unresolved modelling assumptions — settle these before a planner

Ordered by how much they would change a planner's design.

**7.1 · The registration offset is not observable below ~0.07 m, and V0 estimates it
anyway.** Measured above. A planner that reads `registration_mean_m` to decide where a
junction is will, for small true offsets, be acting on noise it could have avoided by
assuming zero. *Unresolved:* whether to gate the offset estimate on a significance test,
whether to make it per-region rather than one globally random-walking vector, and whether
to admit a rotation component (V0 is translation-only, so a map drawn at a slight angle is
unmodelled).

**7.2 · The outlier gate cannot distinguish "the sensor lied" from "the filter is lost".**
Both present as *no belief mass explains this contact*. V0 always chooses to protect the
belief, which is right for a phantom and exactly wrong for a genuinely lost robot — the
measurement that would relocalize it is the one that gets rejected. *Unresolved:* a
relocalization trigger. Nothing in V0 counts consecutive rejections or reinitializes; the
`support` value is logged for every contact precisely so that such a trigger has something
to fire on.

**7.3 · Wrong-turn probability is a required output that this benchmark never stresses.**
It never exceeded **0.05** in any configuration measured (0.045 over 8 scenarios × 6 seeds
at 1500 particles; 0.023 at 2000). The mechanism is verified in unit
tests — a north-facing particle past the junction takes the decoy ~98 % of the time, the
label is sticky, and a particle that leaves the corridor *sideways* at a junction still
transitions (trigger on arclength alone and it never does, so the wrong turn goes
uncounted). But end-to-end, the tactile evidence resolves along-corridor position *before*
the junction is reached, so the number never grows. *Unresolved:* whether that is a
property of the belief or of this floorplan. A scene with a genuine tactile blackout
through a junction would answer it; V0 does not contain one, and the number should not be
trusted as validated until it does.

**7.4 · Probe reach is 1.10 m; this platform's measured arm reach is ~0.45 m.** The
oblique rays in a fan must reach a wall 0.80 m away (0.80 / sin 60° = 0.92 m), and it is
the *disagreement* between the 60° and 90° rays that brackets a wall terminus — a fan whose
oblique rays fall short degenerates to a single ray and stops localizing along the
corridor. The belief code does not care (reach is config), but **station placement does**:
on hardware the robot must walk within ~0.45 m of a wall or the fans return only
swept-free. *Unresolved:* whether the planner hugs walls, or whether the fan geometry is
redesigned for short reach.

**7.5 · The surface normal is not optional.** A wall is a slab with two faces, and a
contact scored on position alone fits either — `test_position_alone_cannot_tell_which_face_of_a_thin_slab_was_touched`
measures the rank-1 collapse degrading by more than 2× when the normal is withheld. V0
assumes the normal is always measured and reasonably accurate (σ = 0.12 rad).
*Unresolved:* behaviour with a position-only sensor, and whether a wall's far face should
simply be excluded from scoring.

**7.6 · Signatures are given noiselessly by ground truth.** The simulator tells the belief
that it touched *a corner* or *the pillar*. A real tactile system infers that from contact
geometry and will sometimes be wrong, and a wrong signature is worse than no signature —
it routes the contact to the wrong candidate set with full confidence. *Unresolved:* a
confusion model over signatures, and whether association should range over signature
hypotheses too.

**7.7 · Two sampling limits, both bounded and both real.** (a) Swept-free rays are scored
on the sampled worst violation, so the spacing must be finer than the thinnest slab — at
0.10 m against 0.15 m slabs there is margin, but it is a constraint on the sensor model,
not a free parameter (`test_swept_free_sampling_must_be_finer_than_the_thinnest_wall`).
(b) Contacts are scored against the nearest *body* boundary, not the boundary of the bodies'
union, so an internal seam between two abutting slabs reads as a surface; every slab here is
≤ 0.15 m thick, which bounds the artefact to 0.075 m and confines it to points already deep
inside wall material.

### Also worth knowing

- **Particle count.** ESS bottoms out around 55–70 % of N, and the ordering of the tactile
  channels is unstable below ~1500 particles. 2000 is where the numbers above were taken.
- **The filter assumes more drift than the truth generates** (by design — a filter that
  assumes less is overconfident and diverges). Behaviour under the reverse mismatch is
  untested.
- **Redundancy, not composition.** The small `combined` − `contact_only` gap is a fact about
  this floorplan. Do not generalize it to scenes where free space and contact carry
  different information.
