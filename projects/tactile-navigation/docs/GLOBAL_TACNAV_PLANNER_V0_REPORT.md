# Belief-aware global tactile route planner — V0

`src/global_nav/` (Batch 2) · builds on the Batch-1 belief, which is not edited:
its 48 tests still pass unchanged. 35 new tests. All numbers below: 20 seeds, 800
particles, identical budget per seed for every policy.

The goal as specified: choose a near-shortest route while actively visiting reachable
tactile landmarks **when the current belief is not reliable enough for an upcoming route
decision** — and only then. Probes are pure cost in the objective; nothing rewards
contact time or landmark count. Confidence at a branch is a constraint, not a reward.

---

## 1. Making the belief planner-ready (`managed.py`, `affordance.py`)

**Reach.** Every Batch-2 fan is capped at the measured G1 range, `REACH_M = 0.45 m`
(hand_force_v2; the fan constructor raises above it). At 0.45 m **nothing is touchable
from the centerline of a 1.6 m corridor**, so reachability becomes *graph structure*:
wall-adjacent lanes at 0.28 m from the face (sized so the ±40° oblique rays still reach:
0.28/sin 50° = 0.37 m), feature spurs, and stop-and-probe stations. Station approaches
use a tight 0.08 m tolerance because swept-free evidence lives in the last decimetre of
reach — a probe executed 0.11 m off-station was measured to thin the decisive 7-nat
free-ray penetration to a 0.8-nat shrug. `test_every_station_probe_is_reachable` checks
every ray of every station on every map against the map geometry.

**Registration gate.** The Batch-1 estimator has a measured ~0.07 m observability floor.
The particles keep carrying `reg` (the ridge is real); the *point estimate the planner
consumes* is gated: zero until the raw estimate clears 0.10 m, hysteresis at 0.06 m so an
opened gate doesn't chatter. Validation in §5.

**Belief states.** `NORMAL → SUSPECT_LOST (2 consecutive unexplained stations) →
RELOCALIZE (4)`, closing Batch-1 §7.2 (the outlier gate that couldn't tell "sensor lied"
from "filter lost" and always protected the belief). RELOCALIZE re-seeds 70 % of the
population along the route within ±1.8 m of the current arclength estimate **and stops
rejecting low-support contacts** — during relocalization the measurements are trusted
over the belief, which is the actual fix. Recovery: three consecutive well-explained
stations while tight (three, not two: an alias of the true position explains the first
local contact perfectly — measured, §6) plus a re-concentrated cloud. One well-explained
contact clears SUSPECT: it was a phantom after all.

---

## 2. The stressed benchmarks (`stressmaps.py`)

**`twin_junctions`** — aliasing all the way to the commit point. Three *identical*
junction mouths at x = 4, 7, 10 on a 1.6 m corridor; the route takes the middle one; the
outer two are dead-end decoys, both graph-legal turns. The prior is 50/50 over two
positions **3.0 m apart — exactly the mouth spacing** — so every north-wall probe is
aliased: gap answers gap, wall answers wall, all the way to the junction
(`test_twin_approach_is_aliased_and_the_pillar_is_not` checks it station by station).
The one de-aliasing surface is a pillar on the south wall: unique signature, off the
shortest path, **out of reach from the centerline** (0.55 m > 0.45 m). Probing where the
pillar *should* be resolves both ways: contact kills the far hypothesis, swept-free kills
the near one — the robot can also probe where the *rival* hypothesis puts the pillar,
which the planner discovers on its own.

**`drift_gauntlet`** — uncertainty outgrows the junction spacing. A ~12 m approach under
a loose along-corridor prior (σ_x = 0.6 m; the truth is drawn from the same prior), ending
at two openings 1.6 m apart; the first is a dead end. Arrival σ_s ≈ 1.0 m against a 0.8 m
half-spacing. A box on the south wall and the junction furniture itself are the only
along-route anchors; everything between is featureless and rank-1.

**`gauntlet_slip`** — the same corridor, tight prior (σ_x = 0.12), then a **0.9 m
un-odometered slip** after s = 8.5. Rank-1 physics delays detection: side probes on a
uniform wall explain themselves whatever x is, so nothing can fire until the junction
furniture disagrees. 0.9 m is deliberately *off* the furniture's alias spectrum
(pairwise gap-edge deltas 0.4/1.2/1.6 m) — see §6 for what happens on it.

---

## 3. The tactile-affordance graph (`affordance.py`)

Nodes: lane waypoints, junction nodes, stations, goal. Edges: transit / wall-adjacent /
detour, plus **branch edges** — the directed turn-in edge at each junction, the only
places a commitment happens and therefore the only places the confidence gate lives.
Stations store travel-adjacent cost, probe cost (0.7 m-equivalent per fan), reachable
surfaces, and **analytic observation signatures**: per-ray (hit, range, normal,
signature) cast against the known map, exactly — no filter rollout.

The planner's use of a signature is comparative. Navigating by hypothesis A, if
hypothesis B is true the robot physically stands displaced by the inter-hypothesis
offset Δ when it reaches a station; the station's value against that pair is the
disagreement between the ray-casts at S and S ⊕ Δ. Contact-vs-free counts 1.0 — negative
information discriminates as well as touch. Two hardening details, both bought with
measured failures: discrimination is the **worst case over a ±0.15 m jitter ball**
(a knife-edge raycast difference at a slab boundary once existed at Δ = 3.01 m and not at
3.00, and the planner bet a branch on the third decimal of a Monte-Carlo mode estimate);
and `anchors_s` — does the observation vary along the route at all — encodes rank-1
physics: a featureless-wall station *cannot* reduce along-route uncertainty in the plan
because it cannot in the world.

---

## 4. The planner (`policies.py::BeliefAwarePolicy`)

Receding-horizon Dijkstra over **(graph node × alive-hypothesis mask × σ_s bin)**, with
NORMAL / SUSPECT / RELOCALIZE handled at the policy layer. Branch edges are traversable
only when minor-hypothesis mass ≤ δ = 0.10 **and** the within-mode σ_s is below the
junction's spacing-derived bound and the state is NORMAL. A probe action at a station
collapses exactly the hypothesis pairs whose analytic signatures disagree (planner
assumes the MAP hypothesis survives; execution replans on whatever actually happened,
with the dropped mass's re-navigation distance charged to the probe), merges
pseudo-hypotheses closer than the anchor resolution, and re-anchors σ only where
`anchors_s` holds. Probes cost 0.7 m; travel costs its length; **that is the whole
objective** — probing emerges only where a gate cannot otherwise pass.

Design decisions that came out of measured failures, not foresight (each carries its
test):

| failure (measured) | fix |
|---|---|
| route coordinate clamped at corners → a 3.0 m mode separation read as 0.03 m at the turn, phantom singleton, gate opened with the ambiguity intact | route frame extends through every off-route continuation (`test_route_arclength_extends_past_corners`) |
| total σ_s conflates mode spread with mode separation → planner bought anchors that could not pay, 3× per station | gate on **within-mode** σ; the mask owns separation (`test_route_s_within_...`) |
| navigation reference latched onto a dying 10 %-mass ghost after its own probe refuted it → drove into the decoy | track nearest mode, snap to heaviest when tracked mass < 0.25 |
| clusterer split tight clouds into phantom micro-modes the gate then blocked on | hypotheses merge at route scale (0.6 m) |
| replan-per-arrival oscillated between two equal-cost anchor plans without probing either | plan hysteresis: replan on information events only |
| a blocked-slide loop fed dozens of identical bump contacts → resampling churn laundered into σ_s 0.15 → 0.02 of fake confidence | bump observations debounced to one per move leg |
| repeatedly bumping a wall the plan says isn't there ended as give-up | blocked ⇒ probe the most informative reachable station: surprise is evidence |

Planning time: mean 5–10 ms per replan, worst observed 62 ms, pure Python.

---

## 5. Results

20 seeds, 800 particles, ± is 1 s.e.m.  Excess is cost (travel + probes + turns) minus
the oracle path *over all episodes*; note failed no-touch runs stop early and cheap, so
their excess under-states the failure — success rate is the primary column.

**twin_junctions** (50/50 prior over two identical positions; every lane probe aliased):

| policy | success | wrong turn | excess (m) | probes | plan (ms) |
|---|---|---|---|---|---|
| no-touch shortest | 0.40 ± 0.11 | 0.50 ± 0.11 | 5.6 | 0.0 | — |
| fixed-interval | 0.40 ± 0.11 | 0.55 ± 0.11 | 5.0 | 3.4 | — |
| max wall-adjacent | 0.45 ± 0.11 | 0.55 ± 0.11 | 5.2 | 4.2 | — |
| max landmark | 0.20 ± 0.09 | 0.10 ± 0.07 | 16.4 | 3.9 | — |
| **belief-aware** | **0.65 ± 0.11** | **0.00 ± 0.00** | 9.3 (6.4 on successes) | **2.5** | 8.7 / 44 max |
| oracle | 1.00 | 0.00 | 0.5 | 0.0 | — |

The no-touch baseline realizes the constructed risk: half its runs turn into a decoy.
Fixed-interval and max-wall — *touch without thought* — keep the **same** wrong-turn rate
as never touching (0.55 vs 0.50): every probe on their lane is aliased, which is the
causal isolation the benchmark was built for: touching doesn't help; touching the right
thing does. The belief-aware planner detoured to the pillar on **20/20 seeds** (or probed
where the rival hypothesis puts it — swept-free resolving both ways), eliminated wrong
turns entirely, and lifted success to 0.65 at 2.5 probes. Its residual failures are
budget deaths on seeds where the swept-free side of the resolution took 3 probes plus
re-navigation. Max-landmark also mostly avoids wrong turns — it does touch the pillar
eventually — but burns 16.4 m of excess and times out on the tight-budget seeds:
landmark *count* is the wrong objective, and the shared budget makes that visible.

**drift_gauntlet** (σ_s ≈ 1.0 m at a 1.6 m junction pair):

| policy | success | wrong turn | excess (m) | probes |
|---|---|---|---|---|
| no-touch shortest | 0.30 ± 0.11 | 0.30 ± 0.11 | 12.4 | 0.0 |
| fixed-interval | 0.95 ± 0.05 | 0.00 | 7.1 | 7.0 |
| max wall-adjacent | 1.00 | 0.00 | 8.1 | 9.0 |
| max landmark | 0.95 ± 0.05 | 0.00 | 11.8 | 3.0 |
| **belief-aware** | 0.85 ± 0.08 | 0.00 | 12.3 | **4.3** |
| oracle | 1.00 | 0.00 | 0.7 | 0.0 |

Under a wide prior *any* competent anchoring works, and the honest reading is that the
belief-aware planner does **not** win here: fixed-interval beats it on success (0.95 vs
0.85) and excess (7.1 vs 12.3) at the price of 7 probes to its 4.3. When anchors are
cheap, lane-adjacent, and everywhere — the junction furniture itself anchors within
reach — a dumb schedule amortizes better than reactive minimalism. The planner is right
about one thing the schedule isn't: it skips the deliberate box detour because the
affordance graph says the junction furniture suffices, and it does.

**gauntlet_slip** (tight prior, 0.9 m un-odometered slip):

| policy | success | wrong turn | excess (m) | probes | reloc trig/rec |
|---|---|---|---|---|---|
| no-touch shortest | 0.20 ± 0.09 | 0.20 | 9.6 | 0.0 | 0 / 0 |
| fixed-interval | 0.55 ± 0.11 | 0.00 | 10.3 | 7.0 | 0 / 0 |
| max wall-adjacent | **0.70 ± 0.11** | 0.00 | 11.1 | 9.0 | 0.10 / 0 |
| max landmark | 0.40 ± 0.11 | 0.05 | 14.5 | 3.0 | 0.35 / 0 |
| belief-aware | 0.40 ± 0.11 | 0.10 | 30.7 | 7.3 | 0.20 / 0 |
| oracle | 1.00 | 0.00 | 2.1 | 0.0 | — |

The honest one, and the planner loses it. Detection latency is physics — rank-1
blindness keeps every side probe self-consistent until the junction furniture — and
after that the outcomes split: **dense probers mostly *absorb* the slip through
association** (prevention rather than recovery: max-wall reaches 0.70 with barely a
relocalization), while the minimal prober meets the slip late, with a tight belief, and
pays for it in recovery grind (30.7 m mean excess). The probes that would have noticed
the slip early are exactly the probes the optimizer proved unnecessary — probing for
*decision confidence* and probing for *integrity* are different objectives, and V0
implements only the first (§7). Relocalization at this scale: triggers on 0.20 of
belief-aware seeds (0.35 max-landmark); **no triggered episode completed recovery at
N = 800** — completions were observed at N = 600 (0.17–0.42 of triggers) and the full
NORMAL → SUSPECT → RELOCALIZE → recovered arc is asserted deterministically in
`test_managed.py` — so recovery *works as machinery* but is not yet *effective in the
loop* at benchmark scale. False relocalizations on the clean scenarios: **0 in 240
episodes**.

### Registration gate (10 seeds per point, dense touch)

| true ‖r‖ (m) | raw estimate err | gated err | assume-zero err |
|---|---|---|---|
| 0.00 | 0.031 | **0.000** | 0.000 |
| 0.02 | 0.043 | **0.020** | 0.020 |
| 0.05 | 0.049 | 0.050 | 0.050 |
| 0.10 | 0.074 | 0.098 | 0.100 |
| 0.20 | 0.158 | 0.186 | 0.200 |
| 0.30 | 0.230 | 0.256 | 0.300 |

Below the floor the gate is exactly the do-nothing line — zero noise injected, while the
raw estimator would have doubled to tripled the error at ‖r‖ ≤ 0.02. Above the floor the
gate keeps a weak recovery (0.256 vs 0.300 at 0.30) but costs a deadband premium over
the raw estimate (+0.02–0.03): it zeroes the seeds whose evidence hadn't yet crossed the
open threshold. Both curves sit at or below assume-zero everywhere. The deeper limit:
under 0.45 m single-lane touch the estimator recovers ~30–50 % of a large offset per
component at best — the x-component is confounded with pose along an aliased corridor
(Batch 1's pose⊖offset ridge, with strictly weaker evidence than Batch 1's 1.10 m fans
measured). The Batch-1 "~0.07 m floor" reads as ~0.10–0.15 m effective under G1 reach.

---

## 6. Negative results and measured limitations

Reported as required — none of these are tuned away:

1. **A slip at an alias distance of the junction furniture is near-unrecoverable from
   local evidence.** With a 1.2 m slip (= gap-edge spacing), the belief re-anchors
   exactly one junction off, every cluster probe confirms the alias, and relocalization
   "recovers" onto it — declared NORMAL, confidently wrong, 0/12 success. The de-aliasing
   evidence is a wall reading *east of the cluster* (station n135), and V0's recovery
   only partially exploits multi-station patterns. The shipped slip scenario uses 0.9 m
   (off the alias spectrum) and states so.
2. **Slip detection under minimal probing is late and partial** (trigger rate 0.20), and
   **recovery completion at benchmark scale is 0/4 triggered episodes** (N = 800; it
   completed on 0.17–0.42 of triggers at N = 600, and deterministically in the unit
   tests). The shipped mitigation — one verification probe after any single unexplained
   contact — helps but does not close it. A probing-for-*integrity* term (distinct from
   probing-for-*decision*) is Batch-3 material, and so is the particle-count sensitivity
   of the recovery loop.
3. **Large registration offsets are only fractionally recoverable at G1 reach** (§5),
   and the gate charges a small deadband premium over the raw estimate above the floor.
4. **belief-aware loses drift_gauntlet on distance and gauntlet_slip on success** (§5):
   when anchors are cheap and plentiful, or when failure detection depends on probes the
   optimizer proved unnecessary, a dumb dense schedule is the better policy. The planner
   wins where information is scarce and priced (twin_junctions); it does not win
   everywhere, and the benchmark was not tuned until it did.
5. **max_landmark under a shared budget is worse than no-touch on success** on
   twin_junctions — visiting everything is not a strategy, and the shared-budget rule
   makes that visible instead of hiding it behind an uncapped run.

## 7. Assumptions a Batch 3 must revisit

- The planner's probe-outcome model is deterministic (MAP survives, discrimination
  resolves in one probe); reality took 1–3 probes on the swept-free side. Receding
  horizon absorbs the error, but an expected-probe-count cost would plan detours more
  honestly.
- Hypothesis identity across replans is positional (nearest-mode tracking with a
  0.25-mass floor). A kidnapped robot whose true position coincides with a decoy mode
  would be mis-tracked; nothing in these scenarios exercises it.
- The affordance graph is hand-built per floorplan. Auto-extraction (lanes from corridor
  medial axes, stations from feature adjacency + reach) is mechanical but unwritten.
- Slides during blocked motion cost budget but produce only debounced touch; a wall-following
  *primitive* that feeds the filter continuously would both localize and move.
- `route_s` for decoy interiors reuses the mouth's arclength anchor; adequate here
  (decoys are short), degenerate for deep off-route excursions.

## 8. Reproduction

```bash
export PYTHONPATH=src           # or: pip install -e .

# full test suite: 48 Batch-1 + 35 Batch-2                     (~8 min)
python3 -m pytest tests/global_nav/ -q

# the benchmark behind every number and figure in this report  (~15 min)
python3 -m global_nav.run_planner_bench --out out/global_tacnav_planner \
        --seeds 20 --particles 800

# quick pass                                                    (~3 min)
python3 -m global_nav.run_planner_bench --out /tmp/gtp --quick
```

Outputs in `out/global_tacnav_planner/`: `planner_bench.json` (every episode,
every metric), `headline_<scenario>.png`, `trajectories_<scenario>.png`
(wrong turn vs pillar detour, same seed), `risk_twin_junctions.png` (minor-hypothesis
mass vs cost; the probe is where the risk dies), `registration_gate.png`,
`reloc_timeline.png` (slip → SUSPECT → RELOCALIZE → recovery).
