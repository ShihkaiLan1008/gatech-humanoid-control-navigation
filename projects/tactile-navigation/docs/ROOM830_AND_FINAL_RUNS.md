# Room 830 curation and final three-room runs

Date: 2026-09-01  
Branch: `wm-selection`  
Seeds: 0 and 1

## What changed

The Room 830 scan was curated into the geometry shown by the owner's sketch.
Fragmented scan surfaces were replaced by three upper-row cuboids, one lower
island cuboid, a cylindrical pillar, and the thin marker slab in the lower-right
alcove. The small scanned shelf on the right remains present. Curated cuboids
are deliberately exposed as landmarks rather than discarded as scan clutter.

The mission now starts beside the long upper-left cuboid, not beside the
pillar:

- start landmark: `upper_left_long_block`, centre `[-7.150, 4.225]`
- goal landmark: `marker_slab`, centre `[-0.845, -1.110]`
- landmark-to-landmark separation: 8.259 m

`build_corner_room` reads this mission from the curated JSON and chooses a
collision-free stance beside each named body. The marker slab remains a
required goal landmark: a confirmed touch is the arrival event. Goal radius is
only a fallback for missions without a required marker.

The route planner changes already on the branch extend wall-follow lanes to the
intersection of adjacent lane lines and use tangent points at convex rounded
corners. Connectors retain the collision-free shape produced by the planner.

## Reproduction

The runs used the vforce controller, auto planning, no video, and the normal
4000 s simulated episode budget. Room 830 was run with:

```bash
PYTHONPATH=src:tactile_controller/src MUJOCO_GL=egl \
  tactile_controller/.venv/bin/python scripts/demo_room_run.py \
  --seed 0 --walls-json docs/demo_maps/room_8_30_2026_curated.json \
  --no-video --out out/demo_map/final/room_8_30
```

The same command was repeated with seed 1. `demo_room_run.py --plan auto` and
`room2_run.py --plan auto` were likewise run for seeds 0 and 1.

## Results

| Room | Seed | Outcome | True goal distance | Believed goal distance | Final belief error | Contacts | Force aborts |
|---|---:|---|---:|---:|---:|---:|---:|
| demo_room | 0 | arrived | 0.579 m | 0.662 m | 0.084 m | 37/42 | 1 |
| demo_room | 1 | gate_failed:convex_corner | 1.929 m | 2.117 m | 0.189 m | 41/54 | 1 |
| room2 | 0 | arrived | 0.098 m | 0.185 m | 0.283 m | 11/11 | 0 |
| room2 | 1 | arrived | 0.055 m | 0.143 m | 0.140 m | 11/11 | 0 |
| room830 | 0 | arrived | 0.007 m | 0.127 m | 0.134 m | 11/12 | 0 |
| room830 | 1 | arrived | 0.218 m | 0.164 m | 0.072 m | 10/10 | 0 |

Five of six runs arrived. Both Room 830 runs confirmed the required
`marker_slab` landmark and therefore ended with the intended arrival event.
Their peak hand forces were 21.03 N and 21.86 N, with no force aborts.

The sole failure was `demo_room` seed 1. It reached the convex corner, confirmed
the incoming face after one retry, then missed all six bounded outgoing-face
presses (roughly 2.35–3.94 N). The executor correctly stopped with
`corner_gate_failed_outgoing_face`; it did not claim arrival from dead
reckoning. This is a remaining robustness issue, not a Room 830 regression.

## Generated artifacts

- `out/demo_map/final/demo_room_gt_vs_belief.png`
- `out/demo_map/final/room2_gt_vs_belief.png`
- `out/demo_map/final/room_8_30_gt_vs_belief.png`
- per-seed JSON, state NPZ, and logs under the corresponding directories in
  `out/demo_map/final/`

The three figures were regenerated from the new run records with
`scripts/plot_gt_vs_belief.py`. Solid lines are simulator ground truth, dashed
lines are the online belief, circles mark starts, and crosses mark endpoints.

## Validation

- curated mission loads as `upper_left_long_block -> marker_slab`
- Room 830 seeds 0 and 1 both contain a required, confirmed goal-landmark step
- `tests/demo_map/test_final_phase.py` and `tests/demo_map/test_scan_map.py` pass
- the final full test invocation is recorded in the commit handoff
