# Room 2 belief transitions — a dataset for a tactile world model

The training unit is one step of localization-belief evolution:

```
(b_t, c_t, M)  ->  (o_{t+1}, b_{t+1})
```

* `b_t` — the particle-filter belief before a touch, stored **in full**
* `c_t` — a freely sampled tactile action
* `o_{t+1}` — what the controller actually felt
* `b_{t+1}` — the posterior after exactly one ordinary PF update
* `M` — the fixed Room 2 map

**The planner is not part of this pipeline.** No route following, no expected
information gain, no policy selects a touch. Base poses are uniform draws
along a frozen route polyline; targets are uniform draws from the
controller's own trained workspace; the label is whatever the hand felt.

```bash
scripts/room2_belief_prior.py --room room2       # section 3, run this first
scripts/collect_touch_data.py --room room2 --smoke    # ~200, validates
scripts/collect_touch_data.py --room room2            # the full 12 000
```

`--room demo_room` runs the same code on the other scan.
`--n-envs 1` uses the single-world path; the default is 128.

---

## 1. Local segments, never whole-room trajectories

A **local segment** is: place the base somewhere near the route, seed a
particle filter, take 10–20 freely sampled touches, throw it away, place
somewhere else. There is no navigation between segments — the base is
*placed*, not driven.

The chain `b_0 → b_1 → … → b_T` exists only to make `b_t` realistic. It is
written out as independent one-step rows sharing a `segment` id and a split;
no sequence model is asked for.

## 2a. The corner the planner drives around

The planner rounds Room 2's corner on a diagonal across open floor: it leaves
the west lane at y = 2.4, crosses, and rejoins the north lane at x = 0.5.
Measured on that route, the cut's closest approach to the corner **vertex** is
1.13 m and to any **face** is 0.79 m — against an arm that reaches 0.48 m. With
base poses drawn ±0.20 m about the centreline, that means the one landmark in
the room, and the only place a touch can feel two faces at once, collected
**no contact data at all**. It showed as an empty quadrant in `topdown.png`.

So the reference centreline replaces the shortcut with the wall-following
elbow (`rooms.hug_walls`, `--no-hug-corner` to disable). The nodes genuinely
following a wall are kept exactly where the planner put them — the straight
legs keep their measured 0.496 m and 0.560 m — and only the crossing changes:

| | planner's cut | wall-following elbow |
| --- | --- | --- |
| max distance to any face | 0.792 m | **0.560 m** |
| closest approach to the vertex | 1.132 m | **0.656 m** |
| at the elbow: west face / north face | 0.78 / 1.06 m | **0.44 / 0.47 m** |

**Where the elbow goes is derived, not chosen.** It sits at
`corner_standoff_m()` from both faces — the midpoint of the contact band,
whose floor is what the carried hands need (`park_x_m` + the deployed 0.045 m
clearance) and whose ceiling is the far edge of the checkpoint's trained
reach. That midpoint, 0.429 m, is the only standoff from which a single stance
reaches *either* face, which is the whole reason a corner is a landmark rather
than two walls. Placing the elbow where the two legs' own lines intersect
instead leaves one face 0.02–0.08 m outside reach.

**A pose faces its wall squarely, not the route's perpendicular.** On a
straight leg those agree, but through a turn the perpendicular sweeps across
the diagonal — and a pose facing the diagonal of a *concave* corner stands
`sqrt(2)` times its perpendicular standoff from anything it can touch.
Measured on the elbow: 0.46 m from each face and **0.59 m along its own reach
axis**, against an arm that reaches 0.48 m. Every pose in the turn was a
guaranteed miss, which left the vertex empty even after the centreline came in
to hug it. `route._resolve_walls` now takes the nearest *face's* own outward
normal (the gradient of the min-signed-distance field), so the yaw snaps from
square-to-one-wall to square-to-the-other at the bisector — the way
`wall_parallel_touch` faces a surface. The route's worst-case reach distance
drops from 0.79 m to 0.560 m.

`segment_meta.reach_standoff_m` is that reach distance — a ray cast along the
pose's own facing direction — not the centreline's perpendicular. At a corner
the two differ by up to `sqrt(2)`, and it is this one that decides whether a
touch can land.

**The corner region is 1.05 m, not 1.8 m.** The spec oversamples the corner
*because* its geometry is informative; spreading that 20 % over 1.8 m put most
of it on plain wall a metre from the vertex, and the nearest contact to the
vertex was 0.63 m. Narrowed to `[2.80, 3.85]` the whole quota lands on 0.5 m
of west face and 0.45 m of north face either side of the elbow — both within
reach, so a touch anywhere in the region can land 0.11–0.16 m from the vertex.

This changes the reference, not the planner: A\* is still what produced the
route, and nothing downstream consults it.

## 2. Where the base goes

`s ~ U(0, L)` along the frozen route, then `δ⊥ ~ U(−0.20, +0.20)` m
perpendicular to the local route direction, with the heading the route
implies. Region quotas: 40 % horizontal / 20 % corner / 40 % vertical.

A pose is rejected before the walls are placed if the base footprint or
**either carried hand** would penetrate. The hand check is the binding one:
the hands are carried 0.3323 m in front of the base, and the deployed lane
leaves them 0.048 m of air, so a base closer than ~0.38 m to a face ploughs it.

## 3. The belief prior is measured, not chosen

`scripts/room2_belief_prior.py` reads every Room 2 run record under `out/`,
buckets each logged belief by route region, and reports the empirical
distribution. Two sources, ranked:

| source | what it is |
| --- | --- |
| `particles` | a run recorded with `scripts/room2_run.py --pf-log` dumps the whole particle set per absorb, so `μ = Σ wᵢxᵢ` and `Σ = Σ wᵢ(xᵢ−μ)(xᵢ−μ)ᵀ` are computed over the full `(x, y, θ)` — **cross terms and all** |
| `nav_trace` | every record ever written carries the filter's own 2×2 position covariance and a scalar heading σ. A real belief covariance with its position–heading cross terms thrown away; marked `cross_terms: false` |

A third, the **labelled fallback** of the spec, is reported beside them and
never mixed into the pool: the covariance of `(believed − true)` pose across
runs. It describes how wrong the filter was over a population, not how wide
any one belief was.

Measured on 333 logged Room 2 beliefs (43 of them from real particle sets):

| region | σx median | σy median | σθ median | n |
| --- | --- | --- | --- | --- |
| horizontal | 0.115 m | 0.157 m | 6.5° | 135 |
| corner | 0.063 m | 0.085 m | 4.0° | 146 |
| vertical | 0.098 m | 0.070 m | 2.1° | 52 |

The corner is the tightest — it is the one landmark the demo's filter gets to
press repeatedly — and the horizontal leg carries the long tail (σy p90 = 0.34 m).

**Sampling is a bootstrap, not a draw from a fitted prior.** `b_0` takes an
actual observed covariance for its region and multiplies it by a single scalar
`exp(N(0, 0.15))`, so the *shape* is always one the filter really had and only
its size moves by about ±16 %. `--jitter 0` makes it a pure bootstrap.

## 4. How `b_0` is placed

Ground truth is used **once**, to decide where a plausible estimate would sit:

```
Σ_0 ~ bootstrap(region)
μ_0 = x_GT + ε,   ε ~ N(0, Σ_0)
particles ~ N(μ_0, Σ_0)
```

Drawing the offset from the same covariance the particles get is what keeps
the initial error consistent with the initial confidence — a belief 0.3 m
wrong with a 0.05 m σ is not one this filter has ever held. After that the
filter sees only odometry increments and tactile observations;
`tests/data_collect/test_belief_transitions.py` parses `belief_segment.py` and
fails if `gt_pose_map` is read anywhere but the line that forms `μ_0`.

## 5. One update per touch

```
b_t  --record--  c_t  --execute-->  o_{t+1}  --predict+update-->  b_{t+1}
```

`ParticleBelief.predict` (odometry) then `ParticleBelief.update` (the
observation) — the standard recursive form, so `b_t` already carries every
earlier measurement and no history is replayed. A contact becomes a
`ContactObs` (point + normal, through the registered 5 mm / 5° sensor model);
a valid miss becomes a `SweptFreeObs` over the span the tip crossed.

The `ManagedBelief` wrapper is deliberately **not** used: its relocalize and
reseed machinery can replace a posterior wholesale, and a sample whose
`b_{t+1}` is a reseed rather than a Bayes update of `b_t` is not the transition
this dataset claims to hold.

**An `invalid_attempt` advances nothing.** Nothing was measured, so nothing is
folded in; `belief_after` is `null` and the next step starts from the same
belief and the same particle row.

## 6. What one action is

`_jump_hand` writes a tip position in the **body** frame and the press creeps
the command to it, so

```
c_t = (p_target_body_xyz, d_approach_body)      d = unit(p_target − park_tip)
```

The approach direction is *generated*, not commanded, and saved afterwards.
Targets are uniform draws from the checkpoint's own trained bands
(`x ∈ [0.30, 0.48]`, `|y| ∈ [0.08, 0.35]`, `z ∈ [0.10, 0.25]`, also capped by
`vf_touch_travel_cap_m`), stratified over 3 heights × 5 angles × 3 reach
distances. The bins are **quantile** cuts of the reachable set, not equal
widths: `|y| ≥ 0.08` puts a deadband around the body centreline, and five
equal-width angular bins leave a middle bin 0.8° wide.

## 7. Outcomes

| termination | label |
| --- | --- |
| debounced sensed contact | `contact` |
| tip reached the target and held 0.3 s, nothing felt | `no_contact` |
| force-guard safety stop | `invalid_attempt` / `safety_abort` |
| creep budget exceeded | `invalid_attempt` / `controller_timeout` |
| settle gate never opened | `invalid_attempt` / `unsettled` |
| target outside the workspace | `invalid_attempt` / `target_outside_workspace` |

Balancing to 45–55 % contact uses the one ground-truth read the spec permits
the **sampler**: `action_space.swept_hit`, a ray from the carry to the target
against the map's convex bodies. It has exactly one caller — the proposal
scorer — and the execution path cannot reach it. The label always comes from
the controller.

### Three requirements that cannot all be maximal

Worth the arithmetic, because it decides what the collector optimises.

A contact needs the base 0.38–0.48 m off a face — closer and the carried hands
plough it, further and the tip cannot reach. The centreline stands at
0.50/0.56 m and the jitter is ±0.20 m, so **contacts live in the two lateral
bins nearest the wall**. Measured over 184 segments, capability is close to
bimodal:

| pose capability | segments | contact rate |
| --- | --- | --- |
| 0 (reaches nothing) | 93 | 2.8 % |
| 0.05–0.2 | 20 | ~68 % |
| 0.2–0.5 | 39 | ~89 % |
| > 0.5 | 32 | 98.4 % |

If the three far lateral bins must each hold 15 % — 45 % of the data at a
standoff where the contact rate is 2.8 % — then the two near bins carry 55 %
and must convert at **88 %** for the run to reach 50 % overall. That is at the
very top of what a capable pose delivers, and getting it means the target
sampler choosing reaching targets nearly every time in those poses, which
starves the near-*reach* action bins.

So the weights choose, and they say so: outcome balance is the requirement with
a stated acceptable range (45–55 %), the lateral spread and the action bins are
**floors**. `W_OUTCOME` is 4.0 against `W_LATERAL`'s 1.0 — at 2.0 the floor won
every tie and the contact rate settled near 42 %.

### The quota engine is a controller, and it was tuned like one

128 segments are committed before a single row exists, each worth 10–20
touches, so the naive schedule is 1 900 identical decisions. Four faults, each
found by measurement:

| fault | symptom |
| --- | --- |
| deficits counted only accepted rows | all 128 opening segments chose the same region — 36/33/32 against a 40/20/40 target |
| the outcome preference was an argmax | tied deficits sent every opening segment one way: 91 % contact, then a long slide |
| capability made a decisive preference | every pose went to the closest legal standoff (0.38–0.40 m), where even a target that stops short still contacts |
| the outcome axis compared raw counts | two thirds through a run *both* deficits are large (0.69 and 0.58 measured), so a pose that can reach a wall outscored one that cannot by 0.09 out of 1.2 — 59 % → 41 % with every lever pointing the right way |

The cures: count what live segments still owe (per region *and* per outcome
class, credited from each target's predicted class, which converts 98.3 % of
the time); sample the preference from the deficits rather than argmax it; draw
the lateral bin **uniformly** and let one score make the trade; and judge the
outcome axis on **share** rather than count, so `0.50 − 0.414` reads as a
signal instead of washing out.

**The detector's own false positives are reported, not corrected.** Contact
latches on 3 sensed samples over 2 N on a channel whose air noise is σ = 1 N,
so about 0.4 % of presses latch on nothing. That is the deployed detector, and
a world model should see the observation distribution the real system has.
Every one is identifiable by `observation.max_force_true_n == 0`, and the
summary reports the rate as `phantom_contacts`.

## 8. On disk

| file | contents |
| --- | --- |
| `manifest.json` | map identity + body table, the frozen route, the workspace bins, the belief prior's provenance, the controller config and digest, the seed, the quotas |
| `transitions.jsonl` | one line per attempt |
| `beliefs/particles_*.npz` | the particle sets, `(row, n_particles, 4)` of `(x, y, θ, w)` in map coordinates |
| `qpos.npz` | full generalized coordinates (single-world runs) |
| `summary.json` / `.txt` | every diagnostic and the pass/fail checks |
| `topdown.png` | route by region, base poses, contacts, misses |
| `coverage.png` | offset, height, angle, reach, the 45 action bins, contact force |
| `beliefs.png` | particle clouds before and after several touches |
| `run_state.json` | how the run ended, rejected poses by reason, the safety summary |

One row:

```jsonc
{
  "run_id": "...", "segment": 41, "step": 7, "uid": ".../s00041/t07",
  "split": "train", "map_id": "room2_8_11_scan", "env_id": 93,
  "segment_meta":   { "s_m":…, "lateral_offset_m":…, "region":…,
                      "belief_init": { "cov0":…, "mu0_map":…,
                                       "offset_from_gt_m":…,
                                       "provenance": { "source":…, … } } },
  "belief_before":  { "mean":[x,y,θ], "cov":[3×3], "cov_eigenvalues":…,
                      "cov_eigenvectors":…, "ess":…, "n_particles":…,
                      "n_updates": 7, "particles_row": 812 },
  "action":         { "target_body":…, "target_map_xy":…, "target_world_xy":…,
                      "approach_body":…, "hand":…, "action_bin":…,
                      "commanded_speed_m_s":…, "desired_contact_force_n":…,
                      "predicted_hit":… },
  "observation":    { "label": "contact", "contact_point_body":…,
                      "contact_normal_body":…, "contact_force_n":…,
                      "max_force_n":…, "max_force_true_n":…,
                      "time_to_contact_s":…, "contact_object":…,
                      "contact_link":…, "termination_reason":…,
                      "p_noisy_body":…, "pf_update": {…} },
  "belief_after":   { …same shape as belief_before…, "particles_row": 813 },
  "state_before":   { "base_pose_map":…, "base_pose_world":…, "joint_pos":…,
                      "joint_vel":…, "ee_pose_body":…, … },
  "state_after":    { … }
}
```

`belief_after.particles_row` of step *t* **is** `belief_before.particles_row`
of step *t+1* — one array on disk per belief, which is what makes the
recursive chain checkable rather than merely asserted.

`state_before`/`state_after` are **privileged simulator state**: ground-truth
pose, joint state, EE pose. They are for debugging and for evaluating a
learned model. They must not become deployment-time world-model inputs.

## 9. Splits

By **local segment** (`records.split_of` hashes `(run_id, segment)` → 80/10/10),
never by transition: the beliefs inside a segment are the same filter at
consecutive instants and the poses are the same base pose.

## 10. Running it: 128 worlds

`mujoco-warp` compiles one model and batches its data across worlds, so the
cost of many robots is nearly the cost of one. Measured on an RTX 4090 with
this exact scene, checkpoint and obstacle injection:

| `num_envs` | ms/step | realtime |
| --- | --- | --- |
| 1 | 41.9 | 0.48× |
| 16 | 47.4 | 6.8× |
| 64 | 49.1 | 26× |
| **128** | **47.7** | **54×** |
| 256 | 50.6 | 101× |
| 512 | 52.6 | 195× |

512 worlds cost 25 % more per step than one. At `--n-envs 128` the collector
sustains **~17 000 valid transitions per hour**, so the full 12 000 takes
under an hour against the ~57 h a single world would need.

Each world runs its own segment, out of phase with the others; one
`driver.tick()` advances every state machine by one control step and every
touch that *ends* on that tick becomes a row. Worlds are independent
mujoco-warp worlds, so rooms in different worlds cannot collide.

**One real bug this uncovered.** `maze_twist` keeps a per-environment
`follow_wall` index chosen at reset from the six maze walls, and reads
`wall_map.hl[follow_wall]` every tick for a metric. Room 2 installs four
walls, so any environment whose reset drew 4 or 5 indexes out of bounds — a
device-side assert on the first step. It is latent in the single-world demo
(one environment has been getting away with it) and fatal at 32. Both paths
now clear the index when the world is installed.

## 11. What the validation checks

`summary.json`'s `checks` block, computed from the written file:

* `pf_chain_is_recursive` — `n_updates` increments by exactly one per valid
  row, and an invalid attempt advances nothing
* `every_valid_row_is_a_one_step_transition` — each row's `b_t` is the particle
  row its predecessor left as `b_{t+1}`
* `no_gt_in_the_filter` — no belief mean is within 1 mm of the true pose
* `split_is_segment_disjoint`
* `contact_balance_45_55`, `invalid_rate_below_5pct`
* per-region, per-lateral-bin and per-action-bin coverage

Offline, against a fake backend that answers from the map's real convex
bodies — no simulator, no GPU:

```bash
python3 -m pytest tests/data_collect -q
```
