# Whole-house tactile navigation — demo pack on `batch3a2-freeze`

Persistent known-map tactile navigation across multi-room routes, on the
frozen Batch 3A.2 stack: the global particle belief, the belief-aware
planner, local ATE, the frozen `hand_force_v2` controller, the all-mode
force guard, the truth firewall, and the execution interfaces. No new
planning method. Scenes, tests, episodes and videos in `src/execution/`
(scenes), `tests/execution/test_whole_house.py`, and
`out/whole_house/`.

**Verdict up front.** Two of the three houses complete end-to-end on the
physical G1 with validated, recorded, replay-rendered episodes:
`nominal_multiroom` (3/3 turns, arrived) and `multi_decision_house` (4/4
turns, two tactile decisions, arrived). The ambiguous house demonstrates
its detour and belief resolution physically (same map and prior as the
guessing baseline, which wrong-turns and fails), but across eight seeds no
episode completes the full route — three failure classes are characterized
in §4 and reported per the task's own rule, instead of weakening the
scene's ambiguity or safety conditions.

---

## 1. The three houses (six boxes, hand-built graphs)

Everything is built from the shipped six-box wall asset (fixed half-lengths
1.0/1.2/1.4/1.6/1.1/1.5 m, 0.10 m thick); every probe stays inside the
measured 0.45 m reach; station aliasing and discrimination are verified
analytically in the regression tests, not asserted.

| | walls used | sections | turns | route |
|---|---|---|---|---|
| `nominal_multiroom` | 4 | 4 | 3 | 11.4 m |
| `ambiguous_landmark_house` | 4 + 2 m landmark slab | 4 | 3 | 9.8 m |
| `multi_decision_house` | 5 + 2 m proud bump | 5 | 4 | 13.6 m |

- **A. nominal_multiroom** — mostly unimodal prior (σ 0.14 m, just over the
  0.12 m commit gate, so the terminus anchors are *useful*, not decorative);
  ordinary bracket stations on every leg.
- **B. ambiguous_landmark_house** — bimodal along-registration (δ ∈ {0,
  +0.55} m). Wall 0 extends 1.3 m behind the start and leg 0's grid stops
  short of the δ-shifted terminus window, so **every on-route probe before
  branch 1 reads identically under both hypotheses** (asserted: max
  discrimination 0.0; leaks of 1.0 were measured at both grid ends before
  the trim). The only discriminator is a **2.0 m landmark slab in open
  floor behind the start** (analytic discrimination 1.0 at its west edge,
  0.9 at its east edge; free sweeps past its edges certify in open air —
  the 3A.2 observability lesson applied). Reaching it means physically
  walking away from the goal.
- **C. multi_decision_house** — two tactile decisions with different
  evidence types at different places: the prior (σ 0.16) blocks turn 1
  until the **leg-0 terminus brackets** are pressed (wall-end evidence),
  and by mid-leg-2 the accumulated drift again exceeds the gate, where the
  nearest along-anchor is the **2.0 m proud bump** (0.08 m, the protrusion
  the 3A.2 stance machinery is validated on) whose edges step the contact
  range.

**Chain geometry bites at this scale and is regression-tested**: the 4-leg
spiral initially put a wall across the detour corridor at y = −0.42, and
the 5-leg spiral wrapped a wall 0.12 m from the start. Wall-length
assignment fixes both; `test_the_chain_does_not_wrap_onto_the_route` sweeps
the start, every route lane, and every station approach against every body.

## 2. What the scale broke, measured, and the repairs

Four defects surfaced only at 10–14 m / 3–4 turns, each measured before
being fixed (commits `8137858`, `3cb6a22`, `acd7217`):

1. **Sweep validity used the wrong reference at proud faces.** The
   controller's kiss model fits the followed wall as an infinite plane, so
   at a slab stance every open-air free was judged against a plane 0.08 m
   too far: 28 of 36 presses were discarded as "stopped short" while
   genuinely sweeping 0.15 m of open air, and the branch never resolved.
   `press()` now takes the station's `face_offset_m`; zero-offset stations
   are bit-identical.
2. **The gate-stop left no margin for the walk home.** A fan that stops
   pressing the tick σ dips under 0.12 hands the return walk zero margin
   (measured: σ 0.115 at the slab, 0.120 three CONTINUE hops later, a
   forced 3 m round trip, budget dead). Along-anchoring visits now press to
   0.75× the bar.
3. **The commit gate double-counted the minor mass.** Its spread term used
   the *mixture* σ, and a surviving 6.6 % ghost cluster 0.55 m away
   contributes 0.137 m of mixture spread by itself — the gate topped out at
   0.167 against a 0.12 bar and five identical landmark visits changed
   nothing. The spread term is now the **tracked cluster's own σ** (the
   same coherence repair 3A.2's Phase 2 made for position); the minor-mass
   term keeps owning mode risk. Unimodal scenes are unchanged, and the full
   3A.2 suite stays green.
4. **Boundary hover on the minor mass**: at 900 particles the resampling
   noise of the minor mass (±0.01) oscillates it across the 0.10 gate
   between hops; demo episodes on the ambiguous house run at 2000 particles
   (planning cost stays ≈10 ms per replan).

Stand-in fleets after the repairs (8 seeds each): nominal 0.75, ambiguous
0.88 belief-aware — with **0.00 arrivals and 0.88 wrong turns for
no-touch** — multi-decision 1.00 with 4.0/4 decisions.

## 3. Validated MuJoCo episodes (one env, one reset, no teleport, recorded)

Every episode runs **without rendering** and records full generalized
coordinates at video rate; the loop is the frozen persistent one
(plan → walk → stabilize → probe → belief update → replan), all evidence
realized (odometry, contact, swept-free), every hand sampled at every tick
and 200 Hz substep, failure ownership and the no-progress detector intact.

### 3.1 The accepted demo episodes

| episode | outcome | turns | probes / stations | path | peak force | dwell >25 N | belief err | goal err | time | plan ms |
|---|---|---|---|---|---|---|---|---|---|---|
| **nominal / belief_aware / s0** | **arrived** | 3/3 | 21 / 6 | 15.4 m | 15.31 N (1 guard trip, disclosed) | 0 | 0.167 m | 0.217 m | 308 s | 8.6 |
| **multi_decision / belief_aware / s1** | **arrived** | 4/4 | 58 / 24 | 34.0 m | 13.40 N (1 trip) | 0 | 0.168 m | 0.116 m | 1027 s | 10.4 |
| nominal / no_touch / s0 (cost ref) | arrived | 3/3 | 0 / 0 | 14.1 m | 0.00 N | 0 | 0.098 m | 0.109 m | 96 s | 2.4 |
| ambiguous / no_touch / s0 (baseline) | no_progress, **wrong turn** | 3/3 | 0 / 0 | 16.3 m | 20.04 N (1 trip) | 0 | 0.397 m | 0.406 m | 126 s | 2.6 |
| ambiguous / belief_aware / s2 (partial) | safety_abort | 2/3 | 44 / 10 | 14.1 m | 52.89 N | 0.04 s | 0.272 m | 0.83 m | 353 s | 12.3 |

- **multi_decision's two tactile decisions, from its own logs**: 6 leg-0
  presses before turn 1 (t = 184 s) — the terminus anchor that unblocked
  it — then 5 leg-2 presses (4 in the bump region) at t = 584–650 s,
  immediately unblocking turn 3 at t = 653 s. Two separate locations, two
  evidence types, each changing the navigation action that followed.
- **The comparison pair shares the same map and the same 50/50 prior**:
  no-touch commits on its coin-flip mode, wrong-turns (flagged by the truth
  evaluator), drags a wall at 20 N (guard-caught) and ends 0.41 m out with
  an explicit failure; belief-aware walks *away from the goal* to the slab,
  resolves the branch in 4 presses (minor mass 0.06 at t = 134 s), returns,
  and completes two turns before its own failure (§4). Only one of them
  ever collected evidence, and their route decisions diverge exactly there.
- No accepted episode contains a wrong turn, a livelock (zero no-progress
  events), a truth leak (the firewall walks every report), a sustained
  force violation (all dwell-over-25 N zero), an unexplained reset (one
  env / one reset each), or a controller runaway. The disclosed guard trips
  are single transients with immediate stop-and-retreat.
- Nominal cost reference: touch costs 3.2× the task time of blind transit
  (308 s vs 96 s) and 21 presses; what it buys on this unimodal scene is
  bounded belief error at the corners rather than arrival (no-touch's
  drift happened to stay benign on this seed).

## 4. Honest failures (the ambiguous house does not complete)

Eight belief-aware seeds on `ambiguous_landmark_house` (900 and 2000
particles): zero full completions. The *belief* side works — four seeds
resolved the landmark (s2 in 4 presses; s3 to minor 0.00) and s2 executed
two of three turns after resolving — and no seed ever wrong-turned. The
kills are three characterized classes:

1. **Slab-region wrist grinds** (s2, s7: 52.9 N single-substep spikes,
   dwell ≤ 0.04 s, safety-aborted after three trips): walking between slab
   stances under lateral belief error crosses the wrist envelope; the same
   impact-physics tail 3A.2 documented — the guard bounds dwell, not peaks.
2. **Lateral-belief excursion shuttle** (s5, s6: ~1 600 s time-budget, zero
   probes): a ~0.5 m lateral belief excursion makes every wall-ward hop
   command a through-the-wall target; the controller's own wall gate clips
   it, the backend times out, and with zero valid presses there is no
   contact to correct the lateral belief — unrecoverable belief-blind.
3. **Commit-boundary hover / spent-cap starvation** (s0, s1 at 900
   particles): resampling noise oscillates the minor mass across the 0.10
   gate between hops until the landmark's visit caps exhaust (2 000
   particles fixed this class; the others then dominate).

The stand-in completes this house at 0.88 — the gap is physical execution
(stance scatter, drift, impact physics), not planning logic. Per the task
instruction, the scene ships at its designed ambiguity with these numbers,
not weakened until they look better.

### 3.2 Delivered artifacts

- `out/whole_house/<episode>/`: `episode.json` (full record: directives,
  hypothesis trace, probes, tick trace with per-hand forces, physical
  summary with audit equality and dwell), `states.npz` (recorded
  generalized coordinates), `episode_*.png` (synchronized plots), and
  `demo.mp4` for the four demo episodes.
- `out/whole_house/ambiguous_comparison.mp4`: belief-aware vs no-touch side
  by side, same map, same prior, labeled.

## 5. Videos — replayed, never re-simulated

Rendering during an episode changes the physics trajectory (measured in
3A.2: `render_mode` changes environment construction), so every demo video
is produced by `execution.replay_video`: the recorded states are written into
a render environment frame by frame (`forward` only — the policy and the
physics never run) and composited with a synchronized panel: floorplan +
planned route + goal, trajectory, mass-sized pose hypotheses, the selected
tactile station, executed presses, WALK/STABILIZE/PROBE/TURN/ARRIVE mode,
all-hand force against the 12 N guard and 25 N limit lines, and a
route-decision/press ticker. The side-by-side comparison composes two
finished replays with end-padding.

## 6. Reproduction

```bash
cd <repo root>

# offline: scenes, aliasing, chain clearance, stand-in behaviour  (~4 min)
PYTHONPATH=src:tactile_controller/src \
  python3 -m pytest tests/execution/test_whole_house.py -q && cd ..

# one validated episode (records states.npz; NO rendering)
PYTHONPATH=src:tactile_controller/src MUJOCO_GL=egl \
  uv run --project tactile_controller python -m execution.run_batch3a \
    --scene nominal_multiroom --policy belief_aware --seed 0 \
    --out out/whole_house/nominal_multiroom_belief_aware_s0 \
    --no-render --budget-s 1600 --max-probes 60

# the offline replay video for that episode
PYTHONPATH=src:tactile_controller/src MUJOCO_GL=egl \
  uv run --project tactile_controller python -m execution.replay_video \
    --episode out/whole_house/nominal_multiroom_belief_aware_s0 \
    --scene nominal_multiroom \
    --out out/whole_house/nominal_multiroom_belief_aware_s0/demo.mp4

# the side-by-side comparison from two finished replays
python3 -m execution.replay_video --hstack \
  out/whole_house/ambiguous_landmark_house_belief_aware_s2/demo.mp4 \
  out/whole_house/ambiguous_landmark_house_no_touch_s0/demo.mp4 \
  --out out/whole_house/ambiguous_comparison.mp4
```

## 7. Scope kept

No learning, no world model, no integrity-aware planning, no artificial
slip, no global relocalization, no controller retraining, no continuous
wall following, no graph auto-generation. Scenes that the frozen method
cannot complete are reported as such below rather than weakened.
