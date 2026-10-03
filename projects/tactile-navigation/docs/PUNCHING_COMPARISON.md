# Comparison with latest punching

Compared on 2026-09-09 after `git fetch origin punching`:

- Latest remote `punching`: `f1f52751f38fad0e7b7705635e800ce15dbd87f4`.
- Starting `jjang318/punch+global`: `bd2b17ebb209a0bac9c5a151f356a1500ffa2e25`.
- The local branch named `punching` was 12 commits behind its remote. The remote
  revision above is the comparison baseline.

## Active execution

| Component | Latest punching | Global branch after cleanup |
|---|---|---|
| Route | Feature-derived lanes for `--policy eig`; movement/punch argmax for `eig_step` | Constrained grid path, minimizing free travel before distance/rotation |
| Movement | `EigRoutePolicy` or `StepEigPolicy` | One `RouteFollowPolicy`; no EIG movement selection |
| Touch | Full-angle punch pairs in `eig`; joint utility in `eig_step` | Single-hand EIG over both hands; EIG/time gate always applies |
| Global contact schedule | None | Dense default; pruning only via explicit `--thin balanced/time` |
| Shared primitives | `TactileIO` | Same move/punch primitives |
| Physics controller | `VForcePunchController` | Same controller implementation |
| Kinematic backend | Not provided by punching | Added point-base/point-fist simulation |
| Belief | Odometry + contact/free-space PF | Also includes contact recovery, press-range penetration scoring and wall-crossing penalties |

The shared `punch_rules.py`, `punch_vforce.py`, `punch_eig_policy.py`, and
`punch_eig_step.py` match the fetched punching revision after cleanup. The
removed global-only `force_press` extension had bypassed its EIG/time gate.

## Removed from global execution

- `GlobalRoutePolicy`: the duplicate global step-and-punch EIG executor.
- `TouchSelector`: unused pair-touch selection inherited by the single selector.
- `--global-motion`: global execution now has one meaning.
- Inheritance from movement-EIG classes in the global follower and selector.
- Forced first touches at scheduled stations despite insufficient EIG.

Pruning functions remain because they have explicit experiment callers and
tests. Original `contact`, `eig`, `eig_step`, and `stations` CLI policies remain
as comparison modes. Shared map/filter/robot code was not deleted merely
because another entrypoint uses it.

## Execution corrections

The global follower now handles incomplete movement before selecting a touch,
matching the intent of punching's timeout handling. The initial cleanup replaced
longitudinal-only completion with Euclidean endpoint distance; the current
default additionally requires PF goal-region probability (see the route-guard
comparison). `--standoff` is passed to the global planner clearance configuration.
The scene geometry's removed NumPy 2D cross-product operation is expressed
as the equivalent scalar determinant, so demo_room can load in the current env.

## Interpretation of comparisons

Both controllers can follow the same path because navigation shares the same
map, PF and policy. This does not mean their physical behavior is equivalent:
vforce executes an RL checkpoint in MuJoCo, while kinematic directly servos a
point base and clamps fists to surfaces. Vforce's base servo is truth-backed
and its odometry is synthetic. Kinematic noise is sampled per simulation tick,
while PF motion noise is sampled per observed increment; equal coefficients
are not equal accumulated noise. These sensor/controller modeling issues are
unchanged by this planner/cleanup change.

Dense contact predictions do not guarantee every contact will happen. EIG
can skip a low-value touch and a selected touch can miss. Runtime covariance
must therefore be read from `nav_trace`, separately from the planned trace.

## Earlier wall-preference validation (before directional constraint)

Full kinematic CLI runs, seed 0, default dense schedule:

| Scene | Position/yaw bounds | Route | Outside contact reach | Predicted contacts | Actual punches / hits | Outcome |
|---|---|---:|---:|---:|---:|---|
| room2 | 0.30 m / 15 deg | 5.766 m | 0.094 m | 28 | 28 / 28 | route_complete |
| demo_room | 0.40 m / 20 deg | 27.946 m | 1.524 m | 124 | 162 / 100 | route_complete |

Records: `out/global_dense_review/{room}/{room}_kinematic_pf_global_s0.json`.
Outside-contact distances use the planner's grid contact regions and midpoint
sampling, not ground-truth measurements of the physical arm's reach.
Regression tests cover dense defaults, the EIG gate on scheduled contacts,
wall-route preference under loose bounds, unavoidable gaps, move timeouts,
and endpoint completion. The deployed ROS checkout is separate and has not
been updated to this branch's cleaned interface.

## Historical directional constraint runs (old demo boundary)

The demo_room result below is invalid for the intended task: it traversed the
open right end using an artificial wall cap. The corrected map forbids that
traversal and gives no contact credit for the cap. See the current
[route-guard comparison](ROUTE_GUARD_COMPARISON.md) for valid-map runs.

Global execution now defaults to `cross_track`: the source and predicted
endpoint uncertainty perpendicular to the next translation must fit the
bound. The endpoint uses joint SE(2) covariance, including yaw and its
correlation with position. `max_axis` remains an explicit comparison option.

Kinematic, seed 0, dense planned contacts and unchanged EIG/time gate:

| Scene | Cross-track / yaw bound | Route | Planned peak cross-track sigma | Peak runtime motion-check sigma | Punches / hits | Outcome |
|---|---|---:|---:|---:|---:|---|
| room2 | 0.10 m / 15 deg | 5.883 m | 0.0929 m | 0.0567 m | 17 / 17 | route_complete |
| demo_room | 0.12 m / 20 deg | 37.006 m | 0.1150 m | 0.0953 m | 168 / 113 | route_complete |

Room2 touched the second wall at 39.68 s before changing translation
direction. Demo_room visited wall-end geometry near (18.1, -0.75) before
crossing back to the upper wall and returning to the vertical route. Its
three landed probes there reduced PF x sigma from 0.120 to 0.079 m; nearby
miss observations also updated the belief. Runtime EIG actions need not match
the planner's conceptual contact normal. All 117 translation checks passed;
the blocked-move path was exercised by a regression test, not by these runs.
The 0.10 m demo_room search found no feasible route; 0.12 m was an explicit
run setting, not automatic relaxation. Compared with the earlier records,
both the metric and its numerical bound changed.

Records and plots: `out/global_directional/`. The reproduction commands are
in [the quickstart](GLOBAL_PLANNER_QUICKSTART.md). The 35 relevant tests passed,
including cross-track projection, prior-to-turn contact, yaw propagation,
signed position/yaw correlation, and the executor's translation gate.
