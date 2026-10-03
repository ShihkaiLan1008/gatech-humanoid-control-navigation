# Planning and execution uncertainty

Global planning uses a 0.18 m position-standard-deviation bound by default.
Execution in belief mode uses the same 0.18 m sensing threshold by default. The
prediction includes motion through the next contact opportunity, without
crediting future observations.

When the prediction exceeds the threshold, translation waits for the existing
bounded sensing batch (up to four probes, with its existing stopping rules).
The follower then recomputes the displacement from the updated PF pose and
checks geometry and yaw again. One primitive may proceed despite remaining
position uncertainty. The next primitive must be checked and, when needed,
sensed again. A force-guard interruption does not authorize continuation.

Use these options with `scripts/punch_demo.py --policy global` to select the
default uncertainty settings and enable the angle-dependent probe profile:

```bash
--bound-pos .18 --execution-pos-policy probe_then_continue --route-guard belief \
--probe-reach-profile configs/probe_reach_ee_qpos_v1.json
```

The execution trigger inherits `--bound-pos`; an explicit
`--execution-pos-trigger` can override it for comparisons.
`--execution-pos-policy strict` retains hard execution admission at the chosen
execution threshold. Distance/contact guard comparisons retain
strict motion admission. `--uncertainty-scale` scales both position settings.
Geometry, yaw, physical safety, episode limits, and final arrival checks retain
their existing roles. Foot-placement completion criteria are separate.

A clearance-correction timeout is logged as `REPLAN` with
`kind=reconnect:clearance` and `status=timeout`, and does not by itself end the
episode. The primitive is cancelled and odometry updates the PF before the
follower chooses its next target. That movement must pass geometry and belief
admission again. Repeated attempts still count toward the existing station and
episode budgets; a missing connector or a guard interruption is not excused by
this timeout handling. Per-foot completion tolerances are unchanged.

Replay JSON records `position_recoveries`, the position/yaw/geometry components
of `motion_checks`, and `execution_position`. Each issued primitive identifies
whether a positional exceedance was accepted and its duration. The aggregate
`override_motion_duration_s` is time executing such primitives; it is not a
continuous measurement of the PF's time above threshold. Threshold and policy
arguments are also saved in evaluation settings. Run logs include
`MOTION_SENSING` and `POSITION_BOUND_CONTINUE` events.

Suggested manuscript text:

> We plan routes under a positional uncertainty bound and use predicted
> uncertainty to guide tactile sensing during execution. Contact and miss
> observations update the belief before motion resumes, with the execution
> threshold serving as a sensing criterion rather than a termination condition.

The angle-dependent profile is used for probe selection and commanded
endpoints, including recovery probes. Expected contact travel uses the same
tactile-tip convention without an additional dome offset. The global planner's
contact opportunities remain a geometric approximation and do not incorporate
this per-angle reach table.

Experimental settings: the planning bound is 0.18 m and the execution sensing
threshold is also 0.18 m. The method does not guarantee that execution uncertainty
stays below either value, or that every sensing attempt yields a contact.
