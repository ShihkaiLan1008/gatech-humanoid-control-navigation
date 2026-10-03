# e2e WM vs analytic EIG, the velocity floor, and the registration offset

Date: 2026-09-02.  Branch: `wm-selection` (both arms run here: `main` cannot
drive room830 or the auto-route pipeline, and `ig_backend=analytic` IS main's
EIG formula inside the identical pipeline, so the comparison swaps only the
information term).  Controller: vforce.  Plans: fully automatic from the map.

Runs: 3 rooms x 3 seeds x {`wm_direct` (e2e WM), `analytic` (EIG)} = 18, plus
3 room2 seeds with the velocity floor off (A/B).  All 21 records, states, logs
and the 21 per-run figures live under `out/demo_map/e2e_report/`.

## 1. The registration offset is redundant for accuracy; it only buys honesty

Test bed: `global_nav`'s open-loop benchmark (the world is the drawing
translated by `reg_true`), 16 seeds per cell, `live` = the shipped filter
(reg prior 0.05 m, walk 0.004 /m) vs `frozen` = reg pinned to zero.

| arm | true offset | final err [m] | reg-estimate err [m] | sigma [m] | z = err/sigma |
|---|---|---|---|---|---|
| live   | 0 cm  | 0.033 +-0.020 | 0.028 | 0.076 | 0.46 |
| live   | 5 cm  | 0.056 +-0.017 | 0.050 | 0.073 | 0.72 |
| live   | 10 cm | 0.115 +-0.024 | 0.109 | 0.073 | 1.18 |
| frozen | 0 cm  | 0.024 +-0.010 | (0)   | 0.037 | 0.61 |
| frozen | 5 cm  | 0.060 +-0.013 | (0.050) | 0.038 | 1.09 |
| frozen | 10 cm | 0.103 +-0.014 | (0.100) | 0.039 | 1.87 |

Paired per-seed wins for `live`: 4/16, 8/16, 5/16 -- coin flips at every
offset.  Three findings:

1. **No accuracy gain, ever** -- even with the map truly 10 cm wrong.
2. **The reg dimension never estimates the offset**: its posterior error
   equals the offset itself (flat walls leave pose vs registration
   unidentifiable -- `test_flat_wall_leaves_pose_registration_ambiguous`
   documents exactly this).
3. **Its one real effect is doubling sigma** (0.073 vs 0.038), which keeps the
   filter honest when the map IS wrong (z 1.18 vs 1.87 at 10 cm) and slightly
   underconfident when it is right (z 0.46 vs 0.61).

Verdict: as an estimator it is dead weight; as a variance-inflation prior for
"the map may be wrong" it earns its keep only on hardware, where map error is
real.  In sim (map == world) it could be dropped with no accuracy cost; keep
it for deployment honesty, or replace it with an equivalent fixed inflation.

## 2. The 0.02 m/s command floor changes nothing in sim

`lin_vel_min` is now enforced at the attractor in both command generators
(`maze_twist.py` for the vforce task, `locomotion_pos/mdp/commands.py` for the
hand-force task), wired through `MujocoConfig.base_lin_vel_min_m_s = 0.02`:
the emitted linear command never lies in (0, 0.02) m/s -- below-floor commands
are raised to the floor while the goal pulls and snapped to zero once arrived.
Engagement is counted per run and reported as `physical.cmd_vel_floor`.

* Across all 18 floored runs the floor touched **337 of 186,516 linear-command
  ticks (0.18%; worst run 0.37%)**.
* room2 A/B, e2e arm, 3 seeds each: floored final errors {0.164, 0.083,
  0.046} m vs no-floor {0.064, 0.244, 0.058} m -- same distribution, all 6
  arrived, same touch counts (11-12/12).

The attractor's decel curve spends almost no time below 2 cm/s (it commands
zero inside `pos_tol` anyway); the floor exists purely so the sim never relies
on a command the real base cannot track.  Sim behaviour: unchanged.

## 3. e2e WM (`wm_direct`) vs analytic EIG, 3 rooms x 3 seeds

| room | arm | arrived | final true-dist [m] | touch hit rate |
|---|---|---|---|---|
| demo_room | wm_direct | 2/3 | 0.97, (1.83), 0.60 | 105/131 |
| demo_room | analytic  | 1/3 | (0.81), (1.97), 0.60 | 71/105 |
| room2     | wm_direct | 3/3 | 0.16, 0.08, 0.05 | 35/35 |
| room2     | analytic  | 3/3 | 0.13, 0.10, 0.40 | 25/26 |
| room_8_30 | wm_direct | 3/3 | 0.06, 0.12, 0.43 | 30/36 |
| room_8_30 | analytic  | 2/3 | 0.28, (1.32), 0.09 | 25/50 |

**e2e WM: 8/9.  Analytic EIG: 6/9.**  Overall touch hit rate 84% (WM) vs 67%
(analytic): the learned model picks presses that actually land, and that is
where both analytic failures come from:

* `room_8_30 analytic s1` (`stuck:A7`): 4/14 touches contacted, two
  `ate_pillar not_felt`, sigma grew to 1.38 m, a transit walked 0.6 m off and
  force-aborted on an island, the recovery goto timed out.
* `demo_room s1` fails the convex-corner gate on BOTH arms (six bounded
  outgoing-face presses missed) -- the known corner-gate robustness issue, not
  an information-term difference.  demo_room `analytic s0` failed the same
  gate; `wm_direct s0` got through.

The e2e WM needs no per-room training: room_8_30 runs demo_room's `direct`
checkpoint (`out/wm/out/runs_demo_relabelled/direct`), exactly as the
factorized runs did -- the WM's covariance head was shown to be
map-independent (feature-permutation test: rank correlation 1.000 under
permuted or 8 m-shifted absolute coordinates; 0.02 under permuted covariance
features).

Figures (one per run: room + planned route + GT vs belief + touches, and the
sigma / |truth-belief| strip): `out/demo_map/e2e_report/figs/
{room}_{arm}_s{seed}.png`, 21 files.

## Reproduction

```bash
PYTHONPATH=src:tactile_controller/src MUJOCO_GL=egl \
  tactile_controller/.venv/bin/python scripts/room2_run.py \
  --seed S --plan auto --no-video --override ig_backend=wm_direct \
  --out out/demo_map/e2e_report/room2/wm_direct
# demo_room: scripts/demo_room_run.py --plan auto
# room_8_30: scripts/demo_room_run.py --walls-json docs/demo_maps/room_8_30_2026_curated.json
# analytic arm: --override ig_backend=analytic
# floor off:    --override base_lin_vel_min_m_s=0.0
```
