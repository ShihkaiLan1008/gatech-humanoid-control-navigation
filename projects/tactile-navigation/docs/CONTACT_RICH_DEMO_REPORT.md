# Contact-rich tactile-navigation demo — one persistent MuJoCo episode

Branch `contact_rich_demo` (worktree, base `acd7217`). One G1 episode on the
new `contact_rich_gap` scene with the new feature-gated `TOUCH_STEP_FOLLOW`
option: **arrived, 0.191 m inside the 0.22 m goal disc, 2/2 route turns, no
wrong turn, 14 presses / 11 valid tactile observations at 8 stations across
all three legs, peak force 16.10 N, zero dwell above 25 N, zero livelock
events, one env / one reset / no teleport** (max base step 9.7 mm), 239 s of
simulated time, seed 0. All 14 hard acceptance checks pass
(`execution.accept_contact_rich`); the soft "< 20 N peak" preference is met.

Deliverables, all under `out/contact_rich/_keep/mj_s0/`:
`demo_full.mp4` (real-time), `demo_4x.mp4`, `episode.json`, `states.npz`,
`floorplan_labeled.png`, `belief_timeline.png`, `contact_map.png`, and the
episode log.  Videos are produced by **offline replay** of the recorded
states (`execution.replay_video`) — physics never re-ran in render mode.

---

## 1. How touch maintains localization during travel

`TOUCH_STEP_FOLLOW` (default **off**; the frozen benchmark scenes reproduce
untouched — suite green at 135) lets the planner replace a CONTINUE with a
**one-press PROBE at the next predeclared grid stance** when either
robot-side trigger fires:

- **gap** — realized odometry travel since the last *valid* tactile
  observation exceeds `touch_gap_m` (1.0 m);
- **predicted_sigma** — the tracked cluster's spread, grown by the motion
  model's per-metre drift over the distance left to the coming route
  decision, would land over the 0.12 m commit gate.

Every stop is the intermittent cadence this controller demands (park →
STABILIZE → shielded press → retract; LOCOMOTION xor CONTACT is untouched),
runs through the existing ATE offset selection, per-press immediate global
belief update, and gate-stop, and uses only stances whose reach and approach
geometry the scene tests verify.  In the episode: maintenance touches at
`L0s3` (t=53), `L0s7` (t=89), `L1s0/L1s1/L1s2` (t=116–140), and the
**arrival anchor** `L2A0` (t=181) — the goal leg runs the same triggers with
no commit window and the arrive reserve spendable, preferring a stance whose
fan pins the along-axis.  What flat-wall touches buy is bounded cross-track
error and registration (rank-1 evidence); what they cannot buy is along-leg
position — that comes from edges, which is why the goal-leg anchor stance
brackets one.

Two scheduling rules keep the option from ever probing the force guard,
both measured in on this scene (§4): **the attempt consumes the trigger**
(a failed stop re-arms only after another full gap of travel), and the
**gap trigger stands down while the safety retreat margin is grown**.

## 2. How the landmark changes the belief

The scene's along-wall registration prior is bimodal {0.0, +0.55 m} with
**the wrong mode deliberately heavier (0.55 vs 0.45)** — the no-touch
ablation tracks the wrong registration and wrong-turns into the wall it is
following (kinematic check: `wrong_turn=True`, blocked 2.75 m from goal).
Every flat stretch of wall 0 reads identically under both modes (rank 1,
aliased — asserted per station).  The only discriminator is the **0.40 m
gap** between the two leg-0 wall boxes: its edges are wall-end
contact-vs-free evidence, discrimination 1.0 under the station model's
±0.08 m jitter ball, and the gap width is deliberately ≠ the 0.55 m mode
separation so one edge cannot alias to the other under the rival.

Measured flip, two stages: the **first press of the episode** (t=12.9,
`L0s2`, whose fan straddles the gap's west edge) contacts wall material
where the wrong-favoured mode predicts open gap — leader mass flips
0.52 → **0.85** onto the true registration in one update.  The planner then
walks the dedicated disambiguator `GAPW` and its four contacts (6.8–8.7 N,
t=25–35) consolidate the flip to **0.92** and sharpen the along-position.
The surviving 0.08 ghost dies later by resampling; every commit thereafter
carries minor mass ≤ 0.08 against the 0.10 gate.

## 3. How the belief change determines the selected branch

The planner walks and turns on its **tracked mode**: the physical point
where it will commit turn 1 is its physical position plus the believed
remaining distance to the corner.  Under the wrong-favoured prior that
point sits **0.55 m early** — at map x ≈ 3.13, where the followed wall
still runs alongside (the labeled floorplan marks it; committing there is
the scene's physical punishment).  The moment the leader flips at t=12.9,
the believed corner jumps 0.55 m forward to the true corner (x = 3.68) —
the video overlay's "planned turn" diamond makes the jump visible — and
turn 1 later commits there with minor 0.00–0.08 and σ ≈ 0.08–0.09, turning
into the corridor instead of the wall.  Same gate, different evidence,
different physical decision: that is the landmark changing a route decision.

## 4. The three failed attempts (same seed), and what each fixed

1. **157 N wrist snag at t=1.5.** First scene cut placed leg-0's wall
   0.08 m proud (free-certification insurance).  The station approach's
   stage-1 leg walks the start's own lateral line, which the proud face put
   0.30 m out — inside spawn-wobble range of the 0.26 m tucked-wrist
   envelope.  Reverted to the standard 0.38 m plane; the press-validity
   diagnosis (below) shows the insurance was unnecessary.
2. **Arrival death at 0.278 m.** Both turns clean, then 0.24 m of
   along-goal drift under a confident σ, an ARRIVE spin, and a forced
   anchor finding **no station**: from goal-leg stances, wall_1's end cap
   is *coplanar* with wall_2's face (chain-corner geometry), masking the
   near edge, and the 0.35 m grid steps over the cap's far flank — the only
   real along-edge.  Fixed with bracket stances `L2A0/L2A1` (predicted
   free-free-contact-contact-contact) and the goal-leg maintenance hook.
3. **Terminal safety abort, three trips in 11 s.** A maintenance approach
   late on leg 1 force-aborted at 22.2 N (true lateral drift the belief
   could not see); the gap trigger — reset only by *valid* observations —
   immediately rescheduled into the same geometry twice more (19.4, 15.0 N).
   The guard worked; the scheduler kept walking back into it.  Fixed with
   the two scheduling rules of §1.  In the accepted episode the same
   geometry produced exactly one 16.1 N trip, the stand-down held, and the
   episode continued to arrival.

Carried from the whole-house diagnosis (and relied on here): a press is
valid only if the realized wrist clears `kiss − face_offset` by 0.02 m with
a ~0.43 m arm cap, so stances must stabilize at ≤ 0.38 m to the face —
every stance in this scene commands 0.32 m; 11 of 14 presses were valid
(the whole-house wave runs ~30 %).

## 5. Honest limitations

- **One guard trip remains in the accepted episode** (16.1 N, LOCOMOTION,
  right wrist, t=157.5): late-leg-1 lateral drift eats the wall-ward
  approach margin, deterministically across attempts.  It is a
  guard-bounded transient (zero ticks above 25 N; the exclusion audit
  records it, `exclusion_clean=False`), and the recovery works — but the
  *prevention* envelope does not cover this geometry.  A lateral
  re-anchor before late-leg wall-ward hops would close it; out of scope
  for a demo.
- `pos_err` at arrival is 0.227 m — inside the goal disc but the same
  order as the along-goal observability floor; the L2A0 anchor's free rays
  bound one side only (a contact on the cap flank would pin both).
- **The goal-leg anchor briefly splinters the belief** (visible in
  `belief_timeline.png`, t ≈ 185–210): free-swept evidence is one-sided,
  and combined with post-turn drift it raised a rival cluster to 0.39 mass
  before resampling re-collapsed it to a single mode ahead of the ARRIVE
  commit.  The arrival was gated on the recovered belief, not the splinter
  — but a cap-flank *contact* anchor would avoid the excursion entirely.
- The minor mode survives at 0.08 mass for most of leg 0, just under the
  0.10 gate; a stingier prior (0.60/0.40) would have needed a second
  landmark visit.
- Single seed, single scene, by design — this is a demo, not a benchmark.

## 5b. Clarity re-render (v2) — recorded states only

`src/execution/replay_video_v2.py` re-renders the SAME accepted episode (no physics
re-run, no result changes) into a three-panel 1920x720 layout: overhead
floorplan with planned route + raw and smoothed paths + numbered touch
flashes + old/new planned-turn points (left), a close-follow tracking camera
(mjlab `mjCAMERA_TRACKING` on the robot root, distance 1.9 m, elevation
-16 deg, azimuth slewed at <=4 deg/frame toward the robot's facing so the
camera looks over its shoulder at the wall) (center), and a simplified
status panel: mode, belief masses + sigma, planned turn, and the "TOUCH
CONTRADICTS WRONG HYPOTHESIS" banner (right).

Per-event emphasis: every touch gets an expanding flash ring (1.6 s) and a
persistent number — **valid observations (1–11) and invalid attempts (1–3,
grey x) are numbered and counted separately** (the episode's `probes` list
records valid presses only; invalid attempts are recovered from their
controller-owned failure records).  A planned-turn jump > 0.20 m holds the
old point (red X), the new point (green diamond) and the changed route
segment (amber) on screen for 2 s with the moved distance printed; a press
that dropped the minor mass >= 0.05 (or sits within 2 s of the jump) raises
the contradiction banner.

**Smoothing method (visualization only; raw data untouched and drawn
alongside, labeled):** the tick-trace xy is resampled uniformly at 10 Hz
and convolved with a centered 2.5 s moving-average window (25 samples,
edge-padded).  The window is an order of magnitude shorter than any leg or
stop, so turns, stops and route changes survive while ~1–2 Hz gait
oscillation and stabilize wiggle are suppressed.  `overhead_routes.png`
shows raw and smoothed together.

Exact commands (from `~/Desktop/tacnav_wt/contact_rich`):

```bash
# three-panel video + overhead_routes.png (offline replay)
PYTHONPATH=src:tactile_controller/src MUJOCO_GL=egl \
  uv run --project ~/Desktop/tacnav_analysis/tactile_controller \
  python -m execution.replay_video_v2 \
    --episode out/contact_rich/_keep/mj_s0 --scene contact_rich_gap \
    --out out/contact_rich/_keep/mj_s0/demo_v2_full.mp4

# the still alone, no GPU:  add --still-only
# 4x cut
ffmpeg -i demo_v2_full.mp4 -filter:v "setpts=PTS/4" -an \
  -c:v libx264 -preset medium -crf 25 -pix_fmt yuv420p demo_v2_4x.mp4
```

## 6. Reproduction

```bash
cd ~/Desktop/tacnav_wt/contact_rich

# static checks + the suite (no GPU)
PYTHONPATH=src:tactile_controller/src \
  python3 -m pytest tests/execution -q

# kinematic smoke (both policies)
PYTHONPATH=src:tactile_controller/src \
  python3 -m execution.run_batch3a --backend kinematic --scene contact_rich_gap \
    --policy belief_aware --seed 0 --out /tmp/crg_kin --no-render \
    --max-probes 40 --touch-step-follow

# the episode (MuJoCo, ~35 min) -- run_batch3a records states.npz no-render
PYTHONPATH=src:tactile_controller/src MUJOCO_GL=egl \
  uv run --project ~/Desktop/tacnav_analysis/tactile_controller \
  python -m execution.run_batch3a --scene contact_rich_gap --policy belief_aware \
    --seed 0 --out out/contact_rich/mj_s0 --no-render \
    --max-probes 40 --budget-s 1200 --touch-step-follow

# acceptance, figures, video (offline replay -- physics never re-runs)
PYTHONPATH=src:... python3 scripts/accept_contact_rich.py <out>
PYTHONPATH=src:... python3 scripts/plots_contact_rich.py <out>
PYTHONPATH=src:... MUJOCO_GL=egl uv run --project ... \
  python -m execution.replay_video --episode <out> --scene contact_rich_gap \
    --out <out>/demo_full.mp4
```
