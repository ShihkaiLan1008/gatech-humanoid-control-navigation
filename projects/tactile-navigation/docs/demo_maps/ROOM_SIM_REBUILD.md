# Room sim rebuild report — GLB scans → task-faithful MuJoCo rooms

The iPhone-scan → MuJoCo converter was rebuilt to be **task-faithful for
tactile navigation**: geometry that affects walking and touching (wall
endpoints, steps, corners, doorway widths, landmark shape) is preserved and
measured; appearance rides a separate look-only layer. Two rooms are built:

| room | source GLB | role | scene |
|---|---|---|---|
| demo room | `out/demo_map/8_10_2026.glb` (25×43 m corridor complex) | the final demo, registered onto the tape-measured frame | `out/demo_map/scan_room.xml` |
| **room2** | `out/demo_map/8_11_2026.glb` (4.5×5.4 m, L of two walls) | **this week's real-robot test room**, self-framed | `out/demo_map/room2_room.xml` |

Full mechanism-level detail lives in
[RECONSTRUCTION.md → "Task-faithful rebuild"](RECONSTRUCTION.md#task-faithful-rebuild-of-the-scan-converter-2026-08-11);
this file is the summary report with the numbers, the pictures, and the
routes.

## 1. What was wrong (diagnosed, then fixed)

A new tool, `scripts/validate_scan_scene.py`, scores any built scene against
its source GLB (boundary distance of every wall-band sample to the nearest
collision surface, surface normals, doorway widths, landmark surface pose,
sealed-corner gaps, unsupported "ghost" boundary). On the old converter it
found four failure classes:

1. **Merge-averaging flattened depth structure** — one weighted-mean plane per
   wall ate room2's 6–11 cm west-wall recess and pulled the wall 3.2 cm off
   its dominant plane.
2. **Corners never closed** — two 0.5 m phantom gaps at sealed corners the
   robot could walk through.
3. **Openings were not modeled** — the branch mouth came out 200 mm too wide.
4. **Non-straight structure was boxed or dropped** — the curved seat became
   overlapping fake wall stubs (337 mm landmark error); a round pillar became
   a box; a shin-height floor obstacle and an 11k-point structure were dropped;
   13 furniture faces were misclassified as walls.

## 2. The five mechanisms that replaced it

Same pipeline (`extract_scan_walls.py` → walls JSON → `scan_map.py` →
`mujoco_scene.py`), five new mechanisms; the navigation planner, controller
and robot model are untouched.

1. **Steps survive, drift merges.** Parallel overlapping faces > 3 cm apart
   are structure (recess, proud strip) and stay separate boards; ≤ 3 cm is
   scan drift and merges; every face's extent is re-derived from band
   occupancy at its own plane; a deeper board within 4.5 cm of a covering
   front board keeps only its uncovered stretches.
2. **Corners snap shut** only where the floor mask shows nothing walkable
   behind the face line (doorways stay open). Faces whose samples stop below
   1.45 m are furniture, not walls, and go to the landmark pass.
3. **Openings are first-class** (`Opening` → `MapV3.doors`): jambs re-measured
   at 1 cm from tall (z ≥ 1 m) surface, crossings thinner than 18 cm stepped
   over, walkable floor required on both sides.
4. **Landmarks keep their shape**: cylinder / boxes-along-arc / low-poly
   convex prism / oriented box, chosen per cluster, with a recursive
   principal-axis split whenever the candidate outline bridges air. Oblique
   walls that shingled into axis-aligned staircases fold into one oriented
   slab. Dense small floor obstacles are kept; speckle and far-face echoes
   are dropped.
5. **Visual ≠ collision.** The registered scan mesh, cropped and decimated by
   vertex clustering (`out/demo_map/*_visual.obj`), is embedded as geom
   group 2 with `contype=0 conaffinity=0`; collision stays with the clean
   primitives.

Extraction is seeded — rebuilds are reproducible. All 77 `tests/demo_map`
tests pass (test contracts updated where the OLD behaviour was the thing
asserted, each with the reasoning in the test).

## 3. Quantitative comparison vs the source GLTF

Same validator, same definitions, both sides ("before" scenes preserved in
`out/demo_map/recon/before/`):

| metric | target | demo before → after | room2 before → after |
|---|---|---|---|
| boundary median | — | 15.8 → **9.0 mm** | 20.2 → **5.7 mm** |
| boundary p90 | < 50 mm | 43.7 → **30.0 mm** | 42.7 → **25.0 mm** |
| boundary > 50 mm | — | 7.0 % → **4.8 %** | 2.9 % → **0.0 %** |
| doorway width err | < 30 mm | unmodeled → **20 mm** (real door)¹ | n/a (no doors) |
| landmark surface pose | < 50 mm | dropped / 337 mm → **≤ 59 mm**² | dropped → **≤ 31 mm** |
| sealed corners left open | 0 | 2 × 0.5 m → **0** | 0 → 0 |
| normal error median | — | 0.81° → 0.83° | 1.10° → 1.11° |
| normal error p90 | < 10° | 16.1° → 17.5°³ | 9.0° → **9.3°** |

¹ Corridor-mouth junctions measure 40/60 mm; their "jambs" are room corners,
not door frames. ² Worst overall is 116 mm on curve C2, which **deliberately
keeps its tape geometry** (the scan smooths the bump). ³ The scan's own
triangle normals are > 10° off their own wall planes for 10.9 % of samples —
that noise floor is unbeatable by any slab model; the model's excess is ~3 pp
and the median is what touch control sees.

Artifacts: before/after renders from identical viewpoints
(`out/demo_map/recon/{scan,room2}_before_after.png`, incl. a scan-mesh overlay
column), metrics JSONs (`out/demo_map/recon/*_metrics_{before,after}.json`).

## 4. Room2 routes for the real-robot test (planned by the V3 planner)

Both directions planned by the actual stack — `build_scan_map` →
`demo_map_v3` → `followable` lanes → `build_graph3`/`astar3` — via
`scripts/room2_plan.py` (planner untouched; the script only builds the
`V3Scene`, `build_demo_scene`-style, standoff = measured
`TURN_PRESS_LANE_OFF` 0.40 m). Output:
`out/demo_map/room2_planned_routes.png` and `.json` (full per-edge waypoints
in the room2 map frame).

Both directions produce the expected structure — **follow the west wall,
transfer at the corner, follow the north wall, step off to the goal** —
6.26 m, 69 % of it TOUCH_FOLLOW, symmetric (cost 11.1 / 11.3):

```
START(0.63,-3.99) ─0.27m transit→ west lane
  ─TOUCH_FOLLOW 2.35m→ (0.50,-1.40)      # passes the recess: informative, σ shrinks
  ─0.23m connector→ north lane tail
  ─TOUCH_FOLLOW 1.99m→ (2.00,-0.56)      # along the north wall
  ─1.42m transit→ GOAL(3.38,-0.89)
```

Execution caveats, worth knowing before the hardware run:

- The planner does not visit the corner **vertex**; it transfers lane-to-lane
  beside it. Its "landmark" is the informative west-wall recess stretch, not
  the corner per se.
- The north lane's first ~0.9 m is a `followable` inflation curl (lane-to-face
  distance up to 1.26 m > the 0.43 m reach): presses cannot land there —
  treat that stretch as transit. Genuine pressing resumes at x ≈ 1.05.
- The final 1.42 m free transit is a lane-chunk-granularity artifact (nodes
  every ~1.8 m); the goal itself sits only 0.34 m off the north lane. Nudging
  the goal toward a chunk node would extend the followed stretch — a scene
  parameter, not a planner change.

## 5. Regenerate everything

```bash
# extract (venv python: needs trimesh), build, validate, render — per room
tactile_controller/.venv/bin/python scripts/extract_scan_walls.py \
    --glb out/demo_map/8_11_2026.glb --register self --name room2 \
    --out docs/demo_maps/room2_scan_walls.json --visual out/demo_map/room2_visual.obj
PYTHONPATH=src python3 scripts/build_scan_map.py \
    --walls docs/demo_maps/room2_scan_walls.json --tag room2 --out out/demo_map \
    --visual out/demo_map/room2_visual.obj
tactile_controller/.venv/bin/python scripts/validate_scan_scene.py \
    --glb out/demo_map/8_11_2026.glb --walls docs/demo_maps/room2_scan_walls.json \
    --xml out/demo_map/room2_room.xml --label after
MUJOCO_GL=egl tactile_controller/.venv/bin/python scripts/render_scan_scene.py \
    --walls docs/demo_maps/room2_scan_walls.json \
    --scene after=out/demo_map/room2_room.xml --out out/demo_map/recon/room2_renders.png
PYTHONPATH=src python3 scripts/room2_plan.py --out out/demo_map

# demo room: same, with --glb out/demo_map/8_10_2026.glb --register tape
# and the default --walls/--tag
```

## 6. Known limitations

- C1/C2 keep their tape chord+sagitta circles by design; the scan disagrees
  by 61–116 mm because a phone mesh smooths bumps — they will always top the
  landmark-error list.
- The corridor's east wall resolves into two planes ~6 cm apart; anchors take
  the touchable front plane, so the tape-corroboration row reads −86 mm
  (tolerance widened to 110 mm with the explanation — the old "agreement" was
  an average of both planes, a surface that exists nowhere).
- Room2's scan covers only the west + north walls; the room is open on the
  other sides, faithful to the data. The demo scene's auto-picked goal moved
  ~1.5 m versus the pre-rebuild scene (end-room furniture is now real), so
  old recorded replays will show cm-level wall shifts and a different goal.
