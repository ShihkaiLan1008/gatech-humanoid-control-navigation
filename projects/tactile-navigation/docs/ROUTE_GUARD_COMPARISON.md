# Global route tracker comparison

Implemented on `jjang318/punch+global`, 2026-09-09. The belief guard is the
default: advance when the PF reaches a planned belief region, then admit
movement only if prediction through the next contact opportunity fits the
position/yaw bounds. Future contacts receive no execution credit.

Subsequent [comparison with jjang318/eval](EVAL_BRANCH_COMPARISON.md) found a
MuJoCo recovery deadlock in the current follower. These kinematic results
predate the bounded wall-feedback update documented in that comparison.
They do not establish readiness to replace the working eval execution.

This encodes the purpose of “touch before turning”: acquire enough information
to execute the turn and next translation. It does not require a particular
touch if the posterior already satisfies that condition. EIG selects punches;
the global route selects movement. The stage index tracks order, with guards
generated from the plan instead of room-specific transition rules.

## Matched experiment

48 kinematic runs: 2 rooms × 3 trackers × seeds 0–1 × 4 conditions.
Each room uses one shared, dense global plan for all 24 runs. All trackers
share the controller, PF, EIG/time gate, motion limits, one-step uncertainty
check, and geometry/reconnection code.

| Tracker | Leg transition | Departure check |
|---|---|---|
| Distance | Mean within 0.10 m of endpoint | One step |
| Contact event | Distance plus expected mapped surface contact | One step |
| Belief | At least 95% PF mass inside planned arrival region | Through next contact opportunity |

Conditions: nominal noise; every fourth commanded contact stroke shortened to
0.035 m; unreported initial displacement (+0.08, +0.08) m; unmodeled yaw-sensor
bias +0.01 rad per travelled metre. Faults are injected into the simulator or
sensors; execution receives neither truth nor fault labels. Equal seeds give
equal initial random streams, but different actions consume them differently.

Bounds: room2 0.10 m / 15 degrees; demo_room 0.14 m / 20 degrees. The corrected
demo navigation boundary is x=17.744 m at the lower wall's east endpoint. It
is not a closing wall. The old 0.12 m wall-end detour is invalid; all trackers
here use the same corrected map and 0.14 m bound, without automatic relaxation.

## Results

“Completed” also requires the simulated true endpoint to lie in the goal disk.
Time and punch medians include completed runs only.

| Room | Tracker | Completed | Advances below 95% arrival | Median time (s) | Median punches | Runs below clearance margin |
|---|---|---:|---:|---:|---:|---:|
| room2 | Distance | 8/8 | 5 | 72.71 | 18 | 0/8 |
| room2 | Contact event | 8/8 | 3 | 73.30 | 18 | 0/8 |
| room2 | Belief | 8/8 | 0 | 76.36 | 19 | 0/8 |
| demo_room | Distance | 7/8 | 13 | 459.54 | 134 | 5/8 |
| demo_room | Contact event | 0/8 | 1 | — | — | 2/8 |
| demo_room | Belief | 8/8 | 0 | 503.76 | 151.5 | 4/8 |

The belief tracker performed best on completion and arrival qualification in
this sample, with more time and punches. The tested contact-event guard
stopped when its expected surface remained unconfirmed despite EIG-selected
attempts; misses still updated the PF. Distance tracking stopped on demo yaw
drift, seed 1, at the motion-uncertainty gate.

No issued translation failed the audited gap prediction for any method in
these runs. Thus these runs distinguish arrival tracking, but do not establish
an empirical benefit from the additional gap check. A regression test covers
a first step that passes while the complete observation gap violates the bound.

Clearance failures remain: the nominal margin is 0.35 m, with a 0.01 m metric
tolerance. The minimum true wall clearance was 0.210 m for distance and 0.226 m
for belief in demo_room. No run crossed the open-boundary margin. Passing a PF
guard is not a true-clearance guarantee under estimation error; this is not a
certified belief-space controller or a hardware validation.

The shared recovery fix permits outward motion from an inflated obstacle
buffer after a posterior correction, restoring the normal margin. It does
not permit crossing a physical wall or an open navigation boundary. Recovery
targets the nearest point on the current leg; the global plan remains fixed.

## Reproduce and inspect

From the repository root, using the environment available for these runs:

```bash
PYTHONPATH=src ../locomanip-deploy-recovery/.pixi/envs/default/bin/python \
  scripts/compare_route_guards.py --seeds 0 1 --jobs 3 \
  --out out/route_guard_comparison/validated
PYTHONPATH=src ../locomanip-deploy-recovery/.pixi/envs/default/bin/python \
  scripts/plot_route_guard_comparison.py out/route_guard_comparison/validated
```

Artifacts in `out/route_guard_comparison/validated/`:

- `comparison.png/pdf`: completion, arrival/departure checks, run duration.
- `movement.png/pdf`: actual base trajectories for all four conditions, seed 0.
- `touches.png/pdf`: turns and actual punch locations, nominal seed 0.
- `results.json`, `aggregate.json`: all 48 outcomes and aggregate metrics.
- `runs/{room}/{condition}/s{seed}/{tracker}/`: full CLI log and sensor/belief record.
- `{room}_plan.json/png`: the shared plan and its predicted belief.

Implementation: `src/execution/route_guard.py` defines PF arrival membership,
departure prediction and local connectors; `RouteFollowPolicy.step_station`
enforces them. CLI: `--route-guard belief --arrival-confidence 0.95`.
Relevant verification: 118 tests passed, 1 skipped; `git diff --check` passed.
