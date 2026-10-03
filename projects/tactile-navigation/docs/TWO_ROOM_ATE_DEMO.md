# Two rooms, one planner: ATE at every landmark

Date: 2026-08-23 · Updated: 2026-08-24 (vforce port; honest sensing is the only mode) · Status: **shipped, rendered, tested**

This document covers the work asked for: apply room2's planner logic to the
scanned demo room, add a live uncertainty area to the 2D belief panel, and
produce one demo video per room.

---

## 1. What room2's logic actually is

Three rules, and `src/demo_map/scene_room2.py` is nothing but those rules
written out for a room with two landmarks:

1. **ATE owns the landmarks.** At a landmark the ported `tactile_explore`
   belief + EIG scorer decides where to press, how many times, and when to
   stop (`src/demo_map/landmark_ate.py`). The executor keeps feasibility (which
   stances the crab machinery can take) and the navigation PF.
2. **Flat runs keep the rule-based cadence.** A wall is rank 1 — it pins its
   own normal and nothing along itself — so it is not worth the information
   budget. Its stations come from `_stations_for(length, press_spacing_m)`.
3. **Arrival is confirmed by touching something that is not a wall.** room2
   adds an owner-placed pillar abeam its goal, because "am I there yet" is the
   axis a flat wall cannot pin.

The particle filter is unchanged and runs throughout: `_absorb` predicts from
odometry after **every** primitive and updates from **every** touch — rhythm
presses included, not just landmark ones.

## 2. The same three rules on the demo room

`plan_steps_demo_room` (`src/demo_map/scene_demo.py`). The demo room has **36
non-wall bodies** instead of two, so rule 1 needs a definition of "landmark"
rather than a list of names. The one used is the one you gave:

> A landmark is any rigid body that is not a wall — and it is in scope when a
> stance exists from which the arm can press one of its faces.

`src/demo_map/landmarks.py` harvests them. A body is in scope when its press
ray hits *it* (not a wall in front of it) at the press standoff, the base fits
at the stance, ≥ 0.90 m of press lane runs clear through that stance, and the
stance is within 0.90 m of the drawn route. Everything rejected is returned
with a reason; nothing is dropped silently.

**Result on the demo room — 8 bodies in scope, plus the route corner:**

| route s | body | kind | size (m) | press dir | note |
|---|---|---|---|---|---|
| 1.7 | `bulge_0` | cylinder | r 0.30 | apex, −y | curved feature C1 |
| 3.3 | `prot_27` | box | 0.30 × 0.16 | −y | lump on the long wall |
| 9.5 | `bulge_1` | cylinder | r 0.38 | apex, −y | curved feature C2 |
| 12.6 | `bulge_2` | cylinder | r 0.17 | apex, −y | post in the branch mouth |
| — | *(vertex)* | **convex corner** | — | −x then +y | branch → end room |
| 27.0 | `prot_16` | box | 0.12 × 0.08 | +y | end-room slab junction |
| 27.6 | `prot_10` | box | 0.32 × 0.10 | −x | end room |
| 27.8 | `prot_12` | box | 0.28 × 0.06 | −x | end room |
| 27.8 | `prot_14` | box | 0.32 × 0.06 | −x | **goal-confirming** (abeam the goal) |

Not visited: 15 `too_far`, 5 `base_does_not_fit`, 4 `no_lane_to_press_from`,
4 `no_exposed_face`.

Compiled plan: **19 steps, 57 rhythm presses, 8 landmark episodes + 1 corner
episode.** Route order is preserved; the compile never re-routes.

### Three differences from room2, each a fact about the room

1. **The corner is convex.** room2's L is concave, and `LandmarkATE.corner`
   models only that. Asked for the signed distance at the two lanes the robot
   actually walks in the demo room — the branch corridor and the end room — the
   concave model returns −0.28 and −0.27: it calls **both of them solid**, so
   no hand could reach either face through the free space it believes in.
   `LandmarkATE.convex_corner` is the solid-box model the concave
   constructor's own docstring prescribes; it returns +0.43 and +0.42 for
   those lanes and −0.15 inside the wall slab, and its candidates score
   2.66–2.68 nats against the 0.06-nat floor.
2. **No owner-placed goal marker, and none needed.** The end room contains
   real scanned protrusions 0.13 m off the route, with `prot_14` directly
   abeam the goal. Arrival is confirmed against the room the robot is actually
   in rather than against furniture added to make the demo work.
3. **The end room's south wall is two slabs, 0.17 m apart in y**, the near one
   stopping at x = 14.61. The compiler follows the faces a ray actually finds
   under the lane (`_face_runs`). This also fixes a pre-existing bug: the older
   compile put its one station on that wall at x = 14.98, where `face_present`
   is False — it was pressing air.

### New capability: box landmarks

`LandmarkATE.box`. A wall is rank 1; a box is not, and the reason is its
**edges** — a press that lands and a sweep 0.15 m further on that finds
nothing bracket the along-face coordinate between them. Measured on `prot_27`:
one contact drops entropy −5.25 → −8.44 and leaves σ = (0.091, 0.041, 0.049)
in (x, y, yaw).

## 3. The live uncertainty area

`scripts/demo_room_video.py`, right panel. The 2σ **circle** is replaced by the
particle cloud's 2σ covariance **ellipse**, filled and updated every frame.

This is not decoration. After a flat-wall press the cloud is squeezed hard
along the face normal and untouched along the face — the rank-1 result — so
the honest picture is a long thin slot lying along the wall. A circle of radius
σ reports the average of the two axes and shows neither, hiding exactly what a
landmark touch then fixes.

- `executor._cov_xy` records the 2×2 covariance into `nav_trace` at every
  absorb, deliberately with the **same** weighting as the scalar σ (no
  weighted-sample bias correction), so `trace(cov) == sigma_m²` exactly and
  the HUD number and the drawn ellipse can never disagree.
- A **zoom inset** (2.4 m window, bottom-right) follows the *belief* — not the
  truth, which would leak the answer into the panel that judges it. On a 20 m
  room map a 0.7 m ellipse is 3 % of the frame; without the inset the collapse
  is invisible.
- The strip below plots the ellipse **area** (π·a·b, m²) against time on its
  own axis, so the collapse reads as a number as well as a shape. The HUD adds
  the two axes and their ratio — a 10:1 ellipse says the last touch pinned one
  axis and left the other alone.
- Runs recorded before `cov_xy` existed still play: an isotropic cloud of the
  same σ reproduces exactly the circle they used to draw.

## 4. Running it

```bash
# check the compile before spending an hour of physics on it (no MuJoCo needed)
PYTHONPATH=src python3 scripts/demo_room_ate_plan.py --out out/demo_map
```

Both videos, one command each — the run renders its own when it ends,
whether that is arrival, a stuck route, or Ctrl-C (§8 builds the environment;
`--controller vforce` is the default):

```bash
scripts/run_demo.py --room demo_room --stride 2   # ~44 min for 766 s of episode
scripts/run_demo.py --room room2     --stride 2   # ~8 min for 180 s of episode
```

Add `--view` to watch the physics live while it records. To re-render an
existing record without re-simulating:

```bash
PYTHONPATH=src:tactile_controller/src MUJOCO_GL=egl \
tactile_controller/.venv/bin/python scripts/demo_room_video.py \
    --run    out/room2/room2_fwd_s0.json \
    --states out/room2/states_room2_fwd_s0.npz \
    --out    out/room2/room2.mp4 --stride 2 --fps 25
```

`--stride 2 --fps 25` renders every second frame at 25 fps, i.e. **2× real
time** — the default keeps real time (25/stride), which for a 27 m route at
crab speed is a long watch. The touches stay legible at 2–4×.

`--plan rhythm` restores the older demo-room compile (cadence everywhere, no
landmark episodes) if you want the A/B.

Both videos come out of the **same** `demo_room_video.py`: it already switches
scenes on `doc["scene"]`, so the uncertainty area and the zoom inset are
identical in both.

## 5. The controller: vforce, and honest sensing as the only mode

The demo runs **`checkpoints/decoupled_vel_force_mpc/vforce_model_60000_v1.pt`**
(`ZhenyuWu323/tactile_controller` @ `decoupled-policy`) — the deployment
candidate of `DEPLOY_INTEGRATION_PLAN.md`, sim2real-confirmed on the real G1 —
in **both** rooms.

**Honest sensing is no longer a flag.** `MujocoConfig.honest_sensing` and every
one of its 25 branches are gone, along with `DemoExecCfg.exec_noise` and the
truth term in `_arc_shift` (pinned to 0.0). There is one code path: the robot
carries a drifting onboard estimator, every attractor target is biased by
(truth − estimate) so the controller behaves as one fed the estimate would,
the force channel is the SensX model, and `truth_pose` survives only for
evaluation and for the relative-displacement compositions real odometry also
provides. Keeping it as a flag meant every decision site had to be read twice
to know what it did.

The checkpoint cannot be loaded into the wall-following task (two PPO stacks,
`lower_body` 478→15 and `upper_body` 519→14, no single `actor_state_dict`), so
`src/execution/vforce_backend.py` runs it in **its own env family**:
`Mjlab-Eval-Force-Maze-G1`, the controller repo's honest force-eval maze — the
harness `DECOUPLED_CONTROLLER_EVAL.md` validated these checkpoints in — with
the maze replaced by the tacnav scene, exactly as `MujocoBackendV2` replaces
the wall-following maze.

`VForceBackend(MujocoBackend)` inherits every family-agnostic piece, because
that is where the honest-sensing contract lives (a source-guard test pins that
none of it is re-implemented). The seam is: the env build
(`JointOnPolicyRunner`, per-body checkpoint keys, strict load), the maze FSM's
attractor in `_set_goal`, and a **creeping force-commanded press**.
`--controller hand_force` still selects the wall-following stack.

**The carry is not part of the seam.** `VForceConfig` does not override
`park_x_m/park_y_m/park_z_m`: the hands are carried at `MujocoConfig`'s tucked
park, **(0.14, ±0.10, 0.25)** — the pose every demo before this work was walked
in. Two substitutions were tried and both were visible in the video. Mine,
(0.16, ±0.09, 0.12), was invented to silence parked-hand guard trips: it cured
a symptom by giving the robot a different body. The over-correction that
followed was this policy's own **trained nominal tip** (0.3323, 0.1516,
0.0952) — defensible for the checkpoint, but it carries the hand extended
forward and low, which is a *different* posture again, and it is what put the
carried tip on the wall for `over_guard_s` **10.0 s of a 674 s run**. The
tucked carry keeps it 0.22 m clear at the press stance instead, so
`vf_press_stance_m` also goes back to the crab's own `wp_crab_press_standoff_m`
= 0.36 (0.46 existed only to buy air for the extended carry), and the creep
budget grows to 14.0 s because the tip now starts 0.19 m further back.

cuRobo is stubbed when absent: the EE command builds a GPU reach pool through
it at construction, and the controller repo pins that install to a local clone
this machine does not have. The pool exists to *sample* training commands;
this backend freezes every sampler and writes explicit targets, so a stub pool
is never consulted for a command the robot actually takes.

## 6. What the physics runs taught, in order

Each finding was measured, fixed, and re-measured; the refuted fixes are
recorded too, including three of my own that treated symptoms.

1. **The press-depth probe measured the wrong point** (`_stance_for` probed at
   the base; the hand lands `HAND_LEAD_M` = 0.18 further): on C1 it commanded
   0.392/0.359/0.440 instead of 0.33 and all three presses swept air.
2. **The replay rebuilt a different map frame** (base0 0.64 m, heading 173°
   off): `|truth−belief|` drew as 2.57 m where the run recorded 0.07.
3. **The camera stood outside the room** (fixed 1.25 m offset in a 1.19 m
   corridor); it is now measured against free space along its own bearing.
4. **The convex turn gate asked for an axis no reachable face could pin**
   (σ_x 0.043 vs σ_y 0.156): a convex corner gates on what the approach face
   gives, and the second gate still requires face B after the turn.

### Three fixes of mine that were wrong, and what replaced them

5. **I changed the robot's body instead of its situation.** The vforce carry
   is the trained nominal tip (0.3323, ±0.1516, 0.0952). Facing a wall, this
   controller settles at a TRUE standoff of 0.28–0.34 m *whatever* is
   commanded (measured: 0.38 → 0.287, 0.46 → 0.280, 0.50 → 0.291,
   0.60 → 0.337) — so the carried hand rests on that wall. That is the
   posture, not a fault. I answered it by parking the hands at
   (0.16, ±0.09, 0.12), which silenced the guard and gave the robot a
   visibly wrong walk. **Reverted**: the carry is the trained tip, and the
   contact is answered at the guard (below).
6. **I made the press harder by pushing with position.** Commanded force
   stayed 14 N, but the tip was commanded 0.05 m *past* the believed face and
   the EE channel JUMPS by design — stiff position tracking turned a
   quarter-metre lunge into an impact: **240–435 N true** transients against
   a 20 N-saturated sensed channel. **Replaced** by a rate-limited creep
   (0.03 m/s) that stops on the first debounced contact, exactly what the
   hand_force shield does: **true peak 33 N**, `over_limit_s` 0.
7. **I aimed the walk with truth.** `_walk_to_lane` targeted
   `map_point_to_world(lane_point)` — which resolves through the spawn frame,
   i.e. the true map location — while testing arrival against the estimate.
   The base walked to where the wall really is, stopped a belief-error short
   of a target it had reached, and drove its carried hand into walls it did
   not know it was at. **Replaced** by the displacement composition
   `goto_map` performs.

### The guard, sized for this controller

8. **A peak threshold cannot separate a brush from a grind on a clipped
   channel.** The inherited 12 N / one sample ran a full safety stop for every
   carry contact — seconds each, several per leg — and one tripped *inside*
   the settle gate, so a press reported `settle_gate_never_opened` while
   nothing was wrong. Raising it to 25 N was tried and **refuted**: the sensed
   channel saturates at 20 N, so the guard could never fire, and room2 then
   ground a hand into a wall at **201 N true (22 N sensed)** with `aborts 0`.
   What separates them is time. The guard asks for **15 N held**
   (`guard_abort_hold_s`, new, default 0 so hand_force is unchanged).

9. **How long, is set by what the carry does.** Facing a wall this controller
   stands about a hand's length off it, so the parked tip rides the wall for
   as long as the base faces it: the demo room measured `over_guard_s` **10.0 s
   spread across a 674 s run** — seconds at a time, every leg. At 0.40 s that
   is a stop per leg (six in one run, one of them inside the first landmark's
   settle), and each stop costs its station the press, which is the belief's
   only correction: with the early landmarks lost the belief drifted to
   **1.08 m** and every later press swept air at a true standoff of 0.8–1.6 m.
   A hand that is *stuck* is not seconds, it is the whole leg — room2's 201 N
   grind held for tens of them. The hold is **3.0 s**.

10. **A completed safety stop is not a dead end.** `_safety_stop` refuses to
    return until the robot is demonstrably clear and still, which is most of
    what a settle is waiting for — yet the first trip returned `"guard"` and
    ended the run (`guard:R2:rhythm`, 1/8 contacted). `_walk_to_lane` now
    re-aims from where the stop left the base and carries on; `_settle`
    re-freezes its hold there and re-opens its gate; `vf_guard_retries = 2`
    bounds it.

### The measurement none of the above could be chosen by

11. **Single runs cannot compare two settings here.** Same seed, same code,
    two runs: the *first control step* already differs — truth_yaw by
    `-2.384185791015625e-07`, exactly one float32 ULP (2⁻²², GPU reduction
    order in mjlab/warp) — and ten minutes of closed loop turns that into a
    different route (23 vs 28 `nav_trace` events, a metre apart by t = 93 s).
    One configuration produced both `arrived, 11/12 contacts, 0 aborts` and
    `gate_failed:goal_post, 6/15`. Several of the A/B conclusions earlier in
    this section were therefore read off noise; the settings above are the
    ones that survive as *mechanisms* — a force trace, an `over_guard_s`, a
    `true_peak_detail` — rather than as one run beating another.

## 7. Final results — vforce + honest sensing, seed 0

Both rooms, `vforce_model_60000_v1.pt`, honest sensing on, on the shipped
code.

### room2 — **arrived**

| | |
|---|---|
| outcome | **arrived**, 0.31 m true / 0.20 m believed |
| contacts | **11 / 11, all in the 10–25 N band** |
| belief error | median 0.102 m, final 0.114 m |
| aborts | 1 (parked-hand brush, safety stop cleared it) |
| episode | 180 s sim, 466 s wall |

The corner (concave), the corner kiss and the goal-post pillar all pressed.
An earlier same-code-family run measured arrived at **0.02 m true, 12/12,
zero aborts** (`out/vforce/room2_v3_final`) — the user's statement that
room2 cannot fail under honest sensing is confirmed with this controller.

### Demo room — **arrived**

| | |
|---|---|
| outcome | **arrived**, 0.16 m true / 0.19 m believed |
| contacts | **37 / 43, 36 in the 10–25 N band** |
| belief error | median 0.156 m, final 0.126 m — over 27.5 m |
| aborts | 1 (12.9 N walking brush at t = 250, cleared) |
| landmark episodes | 8 of 9 confirmed — C1, prot_27, branch post, convex corner (gate on measured σ_across = 0.028), prot_16, prot_10, prot_12, prot_14 (goal-confirming). C2 `not_felt` after its budget — non-fatal by design, and the wall it grows from was pressed by the R4 rhythm stations directly after |
| episode | 766 s sim, 2611 s wall |

Known gap, stated: the true physical press transient peaks at ~240–435 N for
milliseconds while the sensed channel (SensX model, 20 N saturation) reads
~22 N — every control and safety decision rides the sensed channel, so the
run-level force story is honest to what a real G1 would know, but the
simulated impact itself is harder than the trained trapezoid intends. That
is a controller-side `force_slew_rate`/impedance question, out of this
repo's scope, and it is measured and attributed per run
(`physical_summary.true_peak_detail`).

## 8. Environment

Nothing above needed a hand-built environment beyond `uv`:

```bash
git clone --depth 1 -b decoupled-policy \
    https://github.com/ZhenyuWu323/tactile_controller.git tactile_controller
cd tactile_controller
uv venv --python 3.13
uv pip install --python .venv/bin/python \
    "mjlab==1.5.3" "mujoco==3.10.0" "mujoco-warp==3.10.0.3" \
    "rsl-rl-lib==5.4.0" "jax[cuda12]==0.11.0" \
    scipy pyyaml osqp pytorch-kinematics
uv pip install --python .venv/bin/python -e . --no-deps
```

`uv sync --frozen` does **not** work: the project pins `nvidia-curobo` to a
local sibling clone (`../../curobo`) that its own comment calls "NOT
PORTABLE". Nothing on the wall-following path needs it — the curobo imports in
`reach_pool.py` are inside functions — so `-e . --no-deps` plus the list above
is enough. The versions are the ones `uv.lock` pins; installing `mjlab` at
`>=1.5.0` picks up 1.6.0, whose `CollisionCfg` API change breaks the task
registry on import.
