# Passive-contact interruption and replanning

Branch: `punching`; builds on guard-only revision `3424ae9`.

Final physical source: `3d53adc`. Both requested seed-30 routes completed. This
verifies continuation in these episodes, not a statistically established gain
in navigation performance or contact safety.

## Scope

A force guard interrupts an action, not automatically the whole route. The
existing force thresholds and safety-stop controller are unchanged. No calibrated
execution model, world model, or geometric recovery planner is added.

- Before consuming the safety stop, read the guard's loaded-hand identity and
  measured body-frame hand position. Integrate odometry to that instant, observe
  a position-only contact (no inferred normal or obstacle ID), then run the stop.
- Integrate the stop's subsequent odometry separately; discard the old relative
  goal and anchor the held heading to the existing sensed-heading channel.
- Resume EIG selection from the updated belief after the existing safety stop.
  Its clearance/stability result is diagnostic, not an episode-termination gate:
  an action that leaves contact must be plannable while contact persists. Every
  newly issued action remains subject to the unchanged force guard.
- Keep the safety stop's parked wrist poses AND joint anchors for the next move;
  only at the actual landing belief re-evaluate the punch pair. An uncleared
  guarded turn first reselects a step, retaining the pending corner, rather than
  repeatedly rotating at the same contact.
- Skip the interrupted action's old punch plan. A pending corner survives a
  stop that backs below the corner threshold; only its remaining heading change
  is requested, and the route index advances only after `done`.
- An individual action timeout also cancels its old target and requests a new
  decision; it is not an episode timeout. The overall time/station limits remain.
- The same guard consumption applies during punches: a guarded other hand is
  not fabricated as the probing hand's contact/free sweep. No stale second punch
  is issued. Repeated guards on a continuously loaded hand are not duplicate
  contact observations.

## Routes and rendering

The room830 route preview was shown before simulation: north face westbound,
west face southbound, south face eastbound, east face northbound, then westbound
on the north face to the existing endpoint near (-0.55, 0) in map coordinates.

`punch_demo.py --full-route` enables demo_room's two complete connected walls,
ending at the second wall's mapped far endpoint instead of the legacy one-metre
post-corner segment. This is opt-in and supported for EIG policies. It is not an
invented whole-building perimeter route. `punch_route_preview.py` displays either
task before simulation; the plotted offset is illustrative, not a commanded path.

Replay displays passive contacts, guards and replans. The camera follows recorded
realized yaw, so interrupted/remainder turn commands cannot over-rotate it.

## Validation

28 targeted tests pass, including pre-stop observation ordering, post-stop
odometry, loaded-contact deduplication, partial-turn continuation, stale-punch
suppression, ordinary move timing, and full-route endpoint geometry.
Execution suite: 308 passed, 3 skipped (`/tmp/passive_contact_park_replan_tests.xml`).

### Final experiments at `3d53adc`

Both rooms were run in parallel with VForce + PF + EIG-step, grid 25, seed 30,
1200 s episode budget, and no learned execution model. demo_room used
`--full-route`; room830 used the unchanged route shown before its first experiment.

| Room | Outcome | Policy stop [sim s] | Endpoint error [m] | Median belief error [m] | Full-run hand peak [N] |
|---|---|---:|---:|---:|---:|
| demo_room, both connected walls | route_complete | 578.80 | 0.096 | 0.057 | 50.0 |
| room830, four island faces + final segment | route_complete | 303.18 | 0.094 | 0.053 | 74.8 |

Neither run fell. Endpoint error is distance from the recorded stop pose to the
last segment's map endpoint plus the configured standoff. Full recordings include
the demo script's existing final one-second cleanup, outside policy stop time.

- demo_room: passive contact at 197.94 s, guard at 202.86 s, then a newly selected
  step with backward component (`forward=-0.2`, `left=-0.346`). It completed at
  205.86 s and the punch was re-evaluated there. A separate action timeout at
  232.54 s also replanned instead of ending the route. The robot turned at
  311.44 s and traversed the full second wall, finishing near (12.096, 11.192).
- room830: passive contact during the fourth corner at 263.64 s, guard at
  265.22 s, then only the remaining 1.4187 rad (~81.3 degrees) was requested.
  The pending corner was completed before entering the final segment. The robot
  finished near (-0.559, 0.094).

Both final force stops reported `cleared=true`; continuation after an uncleared
stop is covered by unit tests, **not** by these two final physical episodes.
The peaks above show that this patch is not a force-limit/safety-performance fix.
Do not compare demo_room time directly with the previous one-metre-final-leg
task. Same-seed GPU runs can diverge, and these are regression episodes rather
than an untouched or statistically powered before/after evaluation.

Machine-readable outcomes and events: [summary.json](artifacts/2026-09-06_passive_contact_replanning/summary.json).
Raw JSON, full state recordings and replay outputs are under
`out/punch/passive_contact_3d53adc/{demo_room,room830}/`.

### Videos

Both videos contain the entire newly simulated episode, replayed from recorded
states without re-simulation, at 5x speed (1680 x 720, 25 fps). Both passed full
FFmpeg decode checks with exit code 0. Visual inspection covered demo_room's
second wall and final `route_complete` frame, and room830's interrupted/remainder
turn. No segments were removed to conceal an interruption.

- [demo_room: complete two-wall route, 115.96 s](../../out/punch/passive_contact_3d53adc/demo_room/demo_room_full_s30.mp4)
- [room830: complete island route, 60.84 s](../../out/punch/passive_contact_3d53adc/room830/room830_full_s30.mp4)

Repository-relative video paths above require the retained local `out/` files;
raw recordings and videos are ignored by Git. The recorded command protocol and
outcomes are retained in the tracked summary.

### Development run at `483a210`

The first room830 run stopped at 217.40 s because a 15 s move timeout was still
propagated as episode failure (`move_timeout`). It had not exercised a force guard.
This exposed a caller-lifecycle error, not a reason to loosen arrival or force
thresholds. Action timeouts now replan from actual state as well. The concurrent
demo_room run was deliberately interrupted at about 251 s before rerunning the
corrected implementation. Both diagnostic recordings are retained separately in
`out/punch/passive_contact_483a210/`; neither is presented as a complete route.

### Development run at `0b011c3`

Both rooms then exercised passive-contact capture correctly, but stopped after
the pre-existing safety stop exhausted its six-second clear/still dwell budget:
demo_room at 203.28 s, room830 at 80.50 s. The added upper-layer `guard_unsettled`
exit still prevented planning an action to leave contact. That exit has been
removed; the low-level force limit and stop remain unchanged. These diagnostic
records are retained under `out/punch/passive_contact_0b011c3/`.
