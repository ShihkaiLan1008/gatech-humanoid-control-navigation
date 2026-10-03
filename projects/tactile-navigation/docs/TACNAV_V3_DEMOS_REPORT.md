# V3 contact-rich in-room navigation — two MuJoCo G1 demos

One shared implementation, no scene-name special-casing, one fixed seed (0)
per map.  Both demos are single persistent MuJoCo episodes: one environment,
one reset, no teleportation.

## The two demos

| | **R3 — T junction** | **R2 — L bend** |
|---|---|---|
| topology | 2.3 m corridor → wall-grown pinch → T, real decoy arm | L room, concave bend mid-follow, two convex wall bulges |
| outcome | **arrived**, 0.075 m from goal | **arrived**, 0.035 m from goal |
| belief error at arrival | 0.085 m | 0.120 m |
| path / sim time | 10.6 m / 434 s | 12.4 m / 474 s |
| forward increments on TOUCH_FOLLOW | 12 | 9 |
| increments preceded by a valid touch | **10 (83 %)** | **5 (56 %)** |
| total valid tactile observations | 11 | 5 |
| contacts inside 10–25 N | **11 of 11** | **5 of 5** |
| contact force range | 10.4 – 14.8 N | 12.0 – 15.7 N |
| peak force (all hands, all ticks + substeps) | 14.8 N | 15.7 N |
| dwell above 25 N | **0.00 s** | **0.00 s** |
| force aborts / safety aborts | 0 / 0 | 0 / 0 |
| turns | 3 | 3 |
| wrong turns | **0** | **0** |
| PF masses, start → end | [1.0] → [1.0] | [1.0] → [0.66, 0.23, 0.12] |
| PF σ, start → end | 0.302 → 0.130 m | 0.153 → 0.270 m |
| env / resets | 1 / 1 | 1 / 1 |

Deliverables per demo in `out/tacnav_v3/_keep/mj_{r3,r2}/`:
`demo_*_full.mp4` (full speed), `demo_*_4x.mp4`, `episode.json`,
`states.npz`, `route_contacts_belief.png`, and the run log.  Videos are
offline replays of the recorded states — physics never re-runs in render
mode (the V1 measurement that render mode changes the trajectory still
holds).

## What TOUCH_FOLLOW is now

A persistent locomotion mode, not an uncertainty-triggered event.  Per
forward increment, unconditionally:

    STABILIZE → TOUCH (press to a 14 N target, 10–25 N accepted)
              → PF UPDATE from the realized contact / swept-free observation
              → RETRACT + hand-clearance confirm
              → STEP one increment along the SAME lane polyline

The base rides one lane the whole edge; it never hops between independently
generated stations.  Three rules make it general rather than tuned:

1. **Stance from a map raycast.**  The stance is not "wall plane minus a
   constant": the executive casts the press ray against the known map from
   the believed lane point and stands 0.33 m off whatever surface that ray
   finds — flat wall, bulge cap, jaw face — or stays on the lane when the
   ray finds nothing (a recess, where the press *should* sweep free).  One
   rule covers every feature in every map, belief-side, firewall intact.
2. **ATE picks the point, not the path.**  Among reachable fan offsets it
   takes the most informative (predicted outcome differs most across live
   hypotheses) and rejects grazing incidence beyond 55° — the tangent
   presses that wrist-grind.  It never edits the base path.
3. **Corners are verified before they are committed**, from recent tactile
   evidence plus the commit gate, with one confirming press when the last
   stretch was uninformative; after the turn the next lane is reacquired and
   the cadence resumes.

Route choice is Dijkstra over (node, σ-bin, resolved) with no negative costs.
Two terms were added so wall-following is *chosen*, not accidental: carried
uncertainty per metre (without it an 8 m diagonal beat every wall lane), and
a standing cost for **unreferenced travel** — belief arithmetic stops valuing
touch the moment σ dips under the commit gate, yet every V1/V2 episode that
died, died on an unreferenced stretch.  Routes now run 59–67 % on touch
lanes.

## What changed from V2

| | V2 | V3 |
|---|---|---|
| touching | landmark visits + maintenance when σ or travel-gap demanded it | **every forward increment on a TOUCH_FOLLOW edge** |
| contact force | whatever the 2 N latch produced (5–10 N) | driven to a 14 N target; **10–25 N contract, 16/16 contacts in window** |
| lane | straight, fixed 0.56 m offset | offset from the wall **envelope** and smoothed — bows around deep features |
| stance | wall plane minus a constant, tracked mode | **map raycast, worst case over live hypotheses** |
| features | ≤0.10 m proud (deeper broke the lane) | **0.42–0.48 m proud, 1.3–1.6 m wide** — 3–4 presses each |
| edges | station-to-station hops | **lane polylines** the base rides continuously |

The press change is the one that unlocked the force window: V1/V2 latched a
contact at `contact_threshold_n` (2 N), which is exactly why their contacts
were 5–10 N.  A `_press_target_n()` hook drives to 14 N; the base class keeps
the old threshold so V1 behaviour is byte-identical (its suite is green,
135 tests).

## Rejected run — R4 winding (reported, not shipped)

R4 **failed** and is not a demo: `time_budget` 2.67 m from the goal, 3 of 24
increments touched (12 %), and **peak 295.8 N with 0.4 s of dwell above
25 N** — a hard violation of the acceptance criteria.

Diagnosis: 44 presses returned `sweep_stopped_short_of_mapped_face`.  In
R4's narrow Z passages the arm could not reach the wall, so no evidence
entered the belief, the error grew to 0.53 m, and the robot wandered 34.5 m
through 6 turns colliding with geometry.  The 295.8 N spike happened inside
a PROBE: the arm drove into a surface at full extension before the shield
could regulate — the "impact beats detection" class V1 documented, at higher
energy now that the press target is 14 N rather than 6 N.  **Raising the
press target raises the cost of a mis-aimed press**, and R4 is the evidence.

## Honest limitations

- **The bimodal-prior deadlock.**  The stance stands off the *nearest*
  surface any live hypothesis predicts — safe against collision, but on a
  wall whose normal lies along the ambiguity axis that stance can sit a full
  separation (0.55 m) beyond the 0.43 m arm reach, so the press can never
  land and the ambiguity can never resolve.  Measured on R2: 12 increments,
  0 contacts, peak 0.0 N.  **R3 keeps the bimodal branch-decision story only
  because its ambiguity runs parallel to every wall it follows**; R2 was
  therefore run with a unimodal prior (σ 0.14 m, still above the 0.12 m
  commit gate, so touch still unlocks its turns).  The two demos do not test
  the same thing and are not presented as if they did.
- **R2's coverage is 56 %, not ~100 %.**  Four of nine increments produced a
  stopped-short press even after the stand-closer retry.
- **R2's belief widens** (σ 0.153 → 0.270, masses splinter to
  [0.66, 0.23, 0.12]) while still arriving at 0.035 m: arrival is
  truth-in-disc, and the belief is not fully collapsed at the end.
- **Cost.** One touch per 0.45 m increment costs ~30 s of sim time, and this
  box runs ~6 s of sim per wall-minute, so a 12 m contact-rich episode is
  ~75 minutes of wall clock.  R4 ran 8 h before hitting its budget.
- Single seed per map, by instruction.  No multi-seed statistics, no world
  model, no controller retraining.

## Reproduction

```bash
cd ~/Desktop/tacnav_wt/contact_rich
export PYTHONPATH=src:tactile_controller/src
python3 -m lane_graph.maps_v3 --out out/tacnav_v3/maps       # the maps
python3 -m lane_graph.run_v3 --map r3_tpinch --seed 0 --out DIR      # kinematic
MUJOCO_GL=egl uv run --project ~/Desktop/tacnav_analysis/tactile_controller \
  python -m lane_graph.run_v3 --map r3_tpinch --seed 0 --backend mujoco --out DIR
MUJOCO_GL=egl uv run --project ... \
  python -m lane_graph.replay_v3 --episode DIR --map r3_tpinch --out DIR/demo.mp4
python3 src/lane_graph/report_v3.py DIR r3_tpinch                # figure+metrics
```
