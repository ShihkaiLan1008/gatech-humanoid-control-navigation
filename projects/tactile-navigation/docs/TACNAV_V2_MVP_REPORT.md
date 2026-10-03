# TacNav V2 MVP — generalized in-room contact-rich navigation

Branch `contact_rich_demo` (worktree), modules in `src/lane_graph/`.
First physical episode: **`m2_pillar_room` seed 0 — arrived 0.150 m inside
the 0.22 m goal disc**, belief error 0.166 m, 4 presses (2 pillar contacts
7.3/9.8 N), 6.0 m path, 84 s sim, zero force aborts, peak 23.4 N (an
invalid press the shield force-limited; zero dwell over 25 N).  Recorded
episode + `states.npz` + figures in `out/tacnav_v2/_keep/mj_m2/`;
replay video `demo_v2mvp_full.mp4` (+ 4x cut) rendered offline from the
recorded states — physics never re-ran.

## What changed from V1

| V1 (wall-chain) | V2 (in-room) |
|---|---|
| fixed leg chain, one followed wall per leg | typed route GRAPH: FREE_TRANSIT, TOUCH_FOLLOW lanes, LANDMARK_DETOUR stances |
| stations predeclared per leg | lanes computed from map geometry (0.56 m stance lane, 0.28 m clearance); landmark ring stances double as obstacle-rounding waypoints |
| leg-indexed probe side + kiss wall | frame-driven `set_press_frame`; followed-wall slot = nearest segment ahead (anchors the kiss plane) |
| 6 fixed-length maze boxes | runtime-injected entities: arbitrary wall segments, TRUE CYLINDER pillars, cabinets — zero controller-repo edits, observation space unchanged |
| plan = leg gates on tracked σ/minor | belief-aware Dijkstra over (node, σ, gap, resolved); landmark value is a one-shot RESOLVED state change (a cost discount is a negative cycle — measured hang); wrong-mode risk charged at doors and the goal |
| turn commits | replan after every informative update; lane-clean touch-step-touch (perpendicular hop in → press → same lane point) |

## Stage 1 (approved rev 2)

Five seeded-parametric maps (`src/lane_graph/maps.py`), contours per review with
wall-grown, on-path, large obstacles; computed touch-followable faces on
every figure; buildability + ambiguity story per map.

## The measured fix log (kinematic 5 + physical 11 attempts, all same seed)

Every fix below came from a failed run's instrumented trace, V1-style:

1. lane off-by-one (stretch endpoints recorded the first FAILING sample)
2. negative-cost landmark edges = Dijkstra negative cycle (hang) → resolved-state
3. nav-reference LATCH (mode-thrash killed goto convergence)
4. no fallthrough "arrived" (pressed 0.5 m off-stance)
5. landmark approach WITH press-facing yaw (`stabilize()` does not turn)
6. single-shot relative re-stance along the hypothesis-separation vector
7. `leg_wall_idx` → frame-driven followed-wall slot (kiss anchor)
8. three-strike force-abort terminal in the V2 executive (a wall was ground
   20+ times, guard bailing each time)
9. frame-correct evaluation AND arrival check (world truth vs map belief
   reported a phantom 8.8 m error on a 0.31 m run)
10. hop-chunked governed walks, PHYSICAL only (single-shot 3.5 m transits
    timed out → landmark leg abandoned → zero touches → drift death; in the
    stand-in the extra absorbs only diffuse the belief)
11. believed-clearance speed governor (V1 numbers) on every hop
12. touch starvation 0.8→2.5/m (a touchless shortcut let the arrival anchor
    inherit a 0.66 m error it could not press through: reach 0.43 vs 1.07 m)
13. arrival anchor from the lane stance nearest the goal, with the
    stand-closer retry (V1's repair, reachable this time)
14. ONE committed door pass (the replanning goal loop oscillated the error
    0.14→0.93 across five doorway passes; once standing IN the door at
    0.04 m belief error without credit)
15. pass axis from the ROBOT'S side (a door-anchored axis put the pre-point
    0.55 m outside the building)
16. **silhouette fan** — the decisive one: a 0.8 m column self-aliases
    under the 0.55 m hypothesis shift when pressed mid-body (physical fans
    left a 3-cluster mush; the latch then teleported through the endgame).
    Rays bracketing the silhouette edge collapse the belief; the next
    attempt arrived.

## First-demo checklist (spec Subtask 4), honest

- in-room free-space navigation ✓ (transit legs, no wall chain)
- doorway/obstacle transition ✓ (exits through the east doorway; rounds the
  wall-grown pillar)
- landmark observation changes/confirms a later route decision ✓ (the
  pillar-silhouette contacts collapse the wrong-favoured prior; the route
  then commits straight to the exit)
- online particle-belief updates ✓ (per-press, V1 firewall unchanged)
- arrival ✓ (0.150 m, disc 0.22)
- distributed wall contacts during travel — **WEAK**: the belief resolved
  so early that no maintenance touches were needed; both valid contacts are
  at the pillar.  Flagged as the first goal for demos 2–3 (M3/M5 routes
  ride lanes and will show the touch-step cadence).

## Open items

- M4 (kinematic): the 2.8 m-pitch aliased map never engages its landmark
  (0 probes) — planner_abort.
- M5 (kinematic): replan oscillation after the cost retune (explicit
  `replan_livelock` breaker at 12) — needs a cost/hysteresis pass.
- Belief at arrival is not collapsed (masses [0.58, 0.36, 0.07],
  σ 0.36 m): arrival is truth-in-disc with believed-arrival, honest but the
  endgame runs on a still-multimodal belief.
- One 23.4 N invalid press at the pillar (shield-limited, no dwell): the
  silhouette ray grazes at near-tangent incidence — worth a shallower
  force target at high-|offset| rays.
- The V1 no-touch ablation comparison and demos 2–3 are not run yet.

## Reproduction

```bash
cd ~/Desktop/tacnav_wt/contact_rich
export PYTHONPATH=src:tactile_controller/src
# maps + graphs (no GPU): python3 -m lane_graph.maps --out /tmp/maps
# kinematic:  python3 -m lane_graph.run_v2 --map m2_pillar_room --seed 0 \
#               --out /tmp/kin
# physical:   MUJOCO_GL=egl uv run --project ~/Desktop/tacnav_analysis/tactile_controller \
#   python -m lane_graph.run_v2 --map m2_pillar_room --seed 0 --backend mujoco --out <dir>
# replay:     python -m lane_graph.replay_v2 --episode <dir> --map m2_pillar_room \
#               --out <dir>/demo_v2mvp_full.mp4
# figures:    python3 scripts/plots_v2.py <dir> m2_pillar_room
```
