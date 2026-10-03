# The classical tactile-navigation stack — frozen architecture

What a blind humanoid does with a known floorplan, an uncertain pose, and a
hand: walk wall lanes, press before every increment, update a particle belief
from the contacts, and decide the route from the belief rather than from the
truth. This document is the architecture as frozen, the measurements that
justify each piece, and the limits that are still open.

Frozen at tag `classical-freeze`.  Companion documents: `PF_AUDIT.md` (how the belief was audited and what it
measured), `README.md` (how to run things).

## 1. The pipeline

```
        MapV3 spec                    scene3.build_v3_scene
   walls + bulges + prots  ------->   Floorplan (belief's map)
   + rooms + start/goal               lanes    (followable(): one per wall run)
                                      prior    (bimodal, by construction)
                 |
                 v
        graph3.build_graph3                     planning.geometric
   lane chunks -> TOUCH_FOLLOW edges   <----->  bent FREE_TRANSIT connectors
   waypoints   -> FREE_TRANSIT edges            (A* on a 0.08 m clearance grid)
   astar3 over (node, sigma-bin, resolved)
                 |
                 v
        ExecutiveV3                             global_nav.ParticleBelief
   TOUCH_FOLLOW: stance -> ATE fan ->  ------>  contact / swept-free / structural
   press -> PF update -> step                   ManagedBelief -> hypotheses()
   FREE_TRANSIT: hop, or walk a polyline
   DIVERT: go read geometry that separates
           the live hypotheses
                 |
                 v
        backend (duck-typed)
   KinematicBackend   ray geometry, world == map, seconds per episode
   MujocoBackendV3    G1 + hand_force_v2 policy, 30-55 min per episode
```

The backends are interchangeable behind `truth_pose` / `odometry_increment` /
`goto_map` / `press` / `stabilize`, plus the four frame converters
(`map_point_to_world`, `world_point_to_map`, `map_yaw_to_world`,
`world_yaw_to_map`). Everything above the backend is physics-agnostic.

## 2. The rules that make it general

**Stance from a map raycast, not a constant.** The press stance is 0.33 m off
whatever surface the press ray finds from the believed lane point, worst-cased
over live hypotheses. One rule covers flat wall, bulge cap, jaw face and empty
recess on every map.

**ATE picks the point, not the path.** Among the five fan offsets, the local
chooser takes the one whose predicted outcome differs most across the live
hypotheses and refuses grazing incidence beyond 55°. The base path stays the
graph's lane polyline.

**Decisions are made on the belief; truth only scores.** `arrived` means the
robot concluded it was there; `arrived_truly` is the truth-scored answer.
`tests/global_nav/test_truth_firewall.py` checks 36 decision methods across
both executives against an allowlist of the methods that may read truth
(logging, odometry bookkeeping, after-the-fact scoring). It has caught two real
breaches: the arrival test itself (PF_AUDIT §2) and the diversion logger.

**Every claimed improvement is a paired measurement.** `scripts/bench_v3.py`
runs 5 maps × N seeds × arms and reports truly-arrived and false arrivals, not
just `arrived`.

## 3. What the audit changed

The single largest correction was not in the robot. `scripts/pf_audit.py`
mapped the truth position into the belief frame and left the truth yaw in
world, which rotated every scored pose by `pi/2 - yaw0` (107.3° on r3) and made
the physics contact likelihood look misregistered by half a metre. Fixing it
took the median likelihood margin from **-136.6 nats to -0.20 nats** with zero
`model_prefers_wrong_pose` verdicts, and made the physical verdict agree with
the kinematic one. Full account in PF_AUDIT §4.

Three smaller measured results shaped the frozen defaults:

- **`skip_uninformative_probes` (on).** A press where no live hypothesis puts a
  surface within reach is provably flat — the swept-free ray reaches no further
  than the hand. Declining those removed **21.4%** of all probes with no
  accuracy change.
- **`inspect` (off).** Pressing a second fan offset that straddles a mapped edge
  is rank-2 in principle and paid nothing in practice: loose trigger fired on
  70/100 episodes and was a coin flip; tight trigger fired on 7/100.
- **Geometric transit (on).** Bent connectors are strictly additive — every
  shipped route identical in kind and cost — and admit a doorway at exactly
  `2 × LANE_CLEAR` (0.56 m free width) where the 1.15 m hop grid needed 0.80 m.

## 4. The observation model, and what it can and cannot do

| channel | what it constrains | measured contribution |
|---|---|---|
| contact | distance TO a surface; rank 1 on flat wall | the only channel that resolves anything |
| swept-free | "no surface here"; capped at `free_cap_nats` = 4 | near-inert, and possibly mildly harmful |
| structural | edge/route consistency | **0.000 nats on every update of every episode** |

This table is the reason the stack behaves as it does. A press on featureless
wall re-measures the one component that was never in doubt, so between rare
informative touches the odometry drifts unopposed. Informative geometry —
corners, wall ends, bulge shoulders, doorway jambs — is where rank-2 evidence
exists, and going to it is what the diversion mechanism is for.

## 5. Landmark diversion — the r5 mechanism

`r5_twinbay` was the last classical failure: 3/20 truly arrived, 17/20
**false arrivals** (`arrived && !arrived_truly`), 3.4 m belief error, and the
robot confident every time. The diagnosis (`scripts/diagnose_r5.py`, recorded
in PF_AUDIT §11) ruled out the obvious causes — the true mode survives in 19/20
seeds, the belief stays multi-modal, and the disambiguating geometry is
reachable and in the graph. What was missing was an action that goes to it.

The rule, general and belief-side:

```
decision-critical ambiguity            two live modes further apart than
  (modes > 4 goal radii apart,         4 x goal_radius -- they cannot both be
   both above 0.15 mass)               right about where to stop
        |
        v
score every reachable lane stance      EXPECTED refutable mass, in NATS:
  (sampled off the graph's own            contact -> nearest-surface residual
   TOUCH_FOLLOW polylines)                free    -> penetration, capped at 4
        |
        v
reject unless legal under EVERY         stance AND path, in-room and clear --
  live hypothesis                       a diversion is a commitment under
        |                               ambiguity, so it must be safe under
        v                               every hypothesis, not just the tracked
walk there -> ATE picks the offset ->   one
press -> PF update -> REPLAN
```

Nothing in it names r5. It fires where the map offers separating geometry and
the belief needs it, and nowhere else.

### Accepted, on this measurement

`scripts/bench_v3.py`, 5 maps × 20 seeds, paired, `out/bench_v3_final/`:

| scene | truly arrived | false arrivals | belief err | minor mass | probes | path | sim | diversions |
|---|---|---|---|---|---|---|---|---|
| r1_gallery | 11 → **11** | 9 → **9** | 0.147 → **0.147** | 0.480 → 0.480 | 15 → 15 | 13.6 → 13.6 | 62 → 62 | **0** |
| r2_lbend | 19 → **19** | 0 → **0** | 0.051 → **0.051** | 0.000 → 0.000 | 7 → 7 | 9.7 → 9.7 | 42 → 42 | **0** |
| r3_tpinch | 20 → **20** | 0 → **0** | 0.066 → **0.066** | 0.000 → 0.000 | 10 → 10 | 9.4 → 9.4 | 43 → 43 | **0** |
| r4_winding | 15 → **15** | 1 → **1** | 0.107 → **0.107** | 0.000 → 0.000 | 6 → 6 | 10.6 → 10.6 | 45 → 45 | **0** |
| **r5_twinbay** | 3 → **6** | 17 → **14** | 3.414 → **1.761** | 0.502 → **0.222** | 3 → 1 | 10.7 → **7.2** | 36 → **29** | 21 |
| **overall** | 68 → **71** | 27 → **24** | 0.092 → 0.096 | — | 7 → 7 | — | 43 → 43 | 21 |

Four of five maps are **identical on every column** — the trigger does not fire
there at all. r5 improves on every axis *and* gets cheaper: it stops walking
7 m of aliased wall it could not learn from.

Wrong turns are 0 everywhere: r5's failure was never a wrong turn, it was
stopping at the wrong point along the right lane.

### The trace, seed 0

```
belief BEFORE   hyp_x [3.398, 0.006]  masses [0.549, 0.451]  tracked 3.40  truth 0.00
LANDMARK        edge e16, stance (7.22, -0.10), value 0.247, travel 2.84 m
OBSERVATION     contact, 14.0 N
PF CHANGE       masses [0.549, 0.451] -> [1.0]
belief AFTER    hyp_x [7.208]         masses [1.0]           tracked 7.21  truth 7.22
ROUTE DECISION  initial cost 117.6  ->  post-diversion cost 10.0
```

### The ceiling, stated

The diversion resolves the ambiguity **only when the press lands a contact**.
Of 21 diversions: 9 contact (belief collapses to one mode — these are the
wins), 6 sweep free, 6 are unreachable. The free branch is the blocker, and it
is the same one §4 names: with the correct hypothesis true the robot ends
mid-room, sweeps free, and the capped free channel does not refute the wrong
mode — on r5 seed 3 it moved mass **onto** it (0.547 → 0.821). Until a contact
observation can be obtained in both branches, r5 stays at 6/20.

## 6. The replay frame contract

Three recorded streams carry truth, and **they are not in the same frame**:

| stream | written by | frame |
|---|---|---|
| `tick_trace[*].xy` | `MujocoBackend._step` | **world** |
| `probes[*].truth_xy` | `ExecutiveV3._press` | **world** |
| `hypothesis_trace[*].truth_xy` | `ExecutiveV3._snap` via `_truth_map()` | **map** |

`TelemetryV2` applied the world→map transform to all three, so the hypothesis
samples were mapped twice.  On the r1_gallery demo the panel read
"believed vs TRUE (along-route)" up to **+12.10 m** (median +8.38) for an
episode whose belief was never worse than **0.496 m** and which ended
`arrived` with `global_pos_err_m` 0.098.  Double-mapping rotates an
already-mapped point about the spawn by `pi/2 - yaw0`, so the artefact grows
with distance travelled and reads exactly like accumulating route drift.

It was replay-only: nothing the executive decides ever consulted it, which is
why arrival still worked.  `scripts/audit_belief_viz.py` reproduces the
attribution from any recorded episode, and
`tests/execution/test_belief_viz.py` pins the contract — the pose error must
not depend on the world frame the robot happened to spawn in.

The panels now draw the PF where it actually is: every live mode at its own map
xy (area ~ mass), the leading mode as the believed pose, truth as a marker
labelled evaluation-only, `|belief − TRUE|` as the headline number, and the
map-x component demoted to a secondary line that says what it is.

## 7. Final cleanup findings

**Contact normal — measured, and the constant kept.** `_contact_body_frame`
asserts `(-1, 0)`. Scoring every measured wrist axis against the surface each
contact landed on (`scripts/audit_contact_normal.py`, instrumented r3_tpinch,
7 contacts) verifies the convention -- wrist **-x** is palm-forward, +x is
172 deg away, y is perpendicular -- and shows it buys nothing: 8.1 deg median
for both, and the wrist is marginally worse on mean (9.1 vs 8.9) and max
(14.3 vs 12.3). `_face_quat` commands the wrist square to the body, so it
carries no information the constant lacks. The residual ~8 deg is the BASE
standing oblique to the wall; the sensor that would see it is the contact force
direction, not the wrist pose. `MujocoConfig.use_measured_normal` exists,
defaults off, and is documented with the numbers.

**Swept-free -- the premise was wrong, the filter is not.** See PF_AUDIT §7.
`skip_uninformative_probes` was justified as "provably flat"; that argues over
the live mode CENTRES while `swept_free_loglik` is applied to every PARTICLE.
Measured over three maps, a ray every mode agrees about still spans **2.53-2.57
nats** across the cloud and penalises **9-14%** of particles. Tail pruning was
tested and **disconfirmed**: the truth-nearest particle is penalised a median
**0.000 nats**, and truth-mode mass moves by mean +0.006 (r1), +0.047 (r3),
+0.279 (r5). The rule is a latency trade -- 21.4% of the probe budget for a
little evidence -- and r1_gallery is the map that pays, because its belief stays
broad and has the most to gain from within-mode sharpening. No gating fix is
justified.

**Belief visualisation.** Corrected (§6) and all four demos re-rendered from
recorded states with the honest overlay.

The 5-map benchmark after all of this is **byte-identical** to the frozen
baseline: 71/100 truly arrived, 24/100 false arrivals. Everything above is
logging, documentation, or a flag that defaults off.

## 8. Running it

```bash
# tests (382, ~2 min)
PYTHONPATH=src python3 -m pytest tests/ -q

# one kinematic episode
PYTHONPATH=src python3 -m lane_graph.run_v3 --map r3_tpinch --out out/ep

# one physics episode (30-55 min; two fit on one RTX 4090)
PYTHONPATH=src:tactile_controller/src MUJOCO_GL=egl \
  tactile_controller/.venv/bin/python -m lane_graph.run_v3 \
  --map r3_tpinch --seed 0 --backend mujoco --out out/ep_mj

# replay a physics episode to video (states.npz -> mp4)
PYTHONPATH=src:tactile_controller/src MUJOCO_GL=egl \
  tactile_controller/.venv/bin/python -m lane_graph.replay_v3 \
  --episode out/ep_mj --map r3_tpinch --out out/ep_mj/demo.mp4

# the benchmark that decides whether a change helped
PYTHONPATH=src python3 scripts/bench_v3.py --seeds 20 --out out/bench_v3

# the belief audit, and the contact chain
PYTHONPATH=src python3 scripts/pf_audit.py --stack v3 --scene r3_tpinch \
    --seed 0 --out out/pf_audit/kin
PYTHONPATH=src python3 scripts/contact_frame_audit.py out/pf_audit/kin
```
