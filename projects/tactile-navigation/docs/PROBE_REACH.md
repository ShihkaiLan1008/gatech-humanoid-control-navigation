# Per-angle probing reach

Enable explicitly; omitted means the original action geometry and EIG model:

```bash
tactile_controller/.venv/bin/python scripts/punch_eval.py run \
  --controllers footstep --rooms room2 --seeds 0 --out out/evaluation/reach_profile \
  -- --probe-reach-profile configs/probe_reach_ee_qpos_v1.json
```

PACE accepts the same options after `--`. Keep the profile under `configs/`
so source snapshots include it. Existing running snapshots are unaffected.

The editable mapping is [`configs/probe_reach_ee_qpos_v1.json`](../configs/probe_reach_ee_qpos_v1.json).
Each hand has **signed pelvis-frame yaw → `max_radius_m`** knots, with a
shared `margin_m` (default 0.03). An explicit `--probe-margin-m 0.02` overrides
that margin. No controller-specific copy of the mapping is needed.

The ray starts from `(0, ±0.15, 0.10)` m beside the pelvis. The guard is
0.30 m along this ray. Command radius = maximum radius − margin; stroke =
command radius − 0.30 m. All positions refer to the **tactile tip/apex**;
there is no additional 4 cm dome allowance. EIG casts the exact rounded
guard-to-endpoint segment used by `PunchSpec` and the existing interpolator.
Prediction retains its existing execution-noise term, so this is a reach
margin, not a guarantee of contact.

At z=0.10 m, representative supported limits and commands are:

| Target yaw | Maximum tip `(x, y)` [cm] | EIG / command tip `(x, y)` [cm] |
| --- | --- | --- |
| 0° | (44.78, ±15.00) | (41.78, ±15.00) |
| ±30° | (37.84, ±36.85) | (35.24, ±35.34) |
| ±60° | (20.15, ±49.90) | (18.65, ±47.30) |
| ±90° | (0.00, ±57.48) | (0.00, ±54.48) |

Positive outward yaw uses the left hand; negative outward yaw uses the right.
These maxima are projected **sampled IK workspace support**, not measured
absolute limits of the active policy. The YAML's 0.48 m is a horizontal
shoulder-relative cap; it is not a universal 48 cm pelvis-relative endpoint.

Source: `tactile_controller`, `origin/decoupled-policy` at
`25f384979410e7181e86f94cac7dda5120299bfe`, `data/ee_qpos_v1/ee_qpos.npz`
(dataset generator commit recorded internally as `14a820c`). Its full SHA256
is stored in the profile. Extraction uses accepted cell centres at z=0.10 m,
pitch=roll=0, target yaw matching the knot, and within 1.25 cm of the existing
probe ray. The maximum is the FK-tip projection, capped at the target
projection. A sparse sampled workspace does not establish an exact continuous
kinematic boundary or prove this checkpoint's training provenance.

Between knots, use the **smaller adjacent maximum**, then subtract the margin.
There is no extrapolation. For this profile, candidates span outward alpha
−10°…90° per hand: farther inward samples cannot extend past the current guard
after the margin; >90° has no training support. The EIG and random selectors
retain 31 candidates per hand over that interval. With this profile, `sweep4`
uses body yaws −90°, −30°, +30°, +90°. Without it, all presets retain their
original angle ranges. Global route planning and the touch stopping criteria
are unchanged; this option applies to global execution with tactile sensing.

To update, edit each hand's `max_radius_m` values and/or `margin_m`, or rebuild
from a new accepted dataset:

```bash
tactile_controller/.venv/bin/python scripts/build_probe_reach.py \
  /path/to/ee_qpos.npz --out configs/probe_reach_ee_qpos_v1.json --margin-m 0.03
```

The runtime saves the complete profile, its file hash and the effective margin
in replay/evaluation metadata. EIG records also contain each candidate's
stroke and the selected tip endpoint. Early wall contact still stops a probe;
PF contact/miss updates continue to use measured FK positions.

Implementation: [`ProbeReachProfile`](../src/execution/probe_reach.py) owns
lookup/geometry; [`ArmAngleEIG` and `spec_at`](../src/execution/punch_eig.py)
consume it; [`SingleTouchSelector`](../src/execution/global_route.py) and
[`ProbeSelector`](../src/execution/baselines.py) pass it through. No change to
the learned controller or interpolation timing is required.

Validation: eight isolated production footstep probes (both hands, outward
0°/30°/60°/90°, seed 0, checkpoint `0911_footmimic_noforce/model_34000.pt`)
finished without falling. The existing endpoint dwell supplied 14 command
ticks including arrival; final FK XY errors were 2.8–8.7 mm (vertical error
0.9–17.6 mm). A separate Room2 kinematic seed-0 run reached the
goal; it verifies integration, not physical navigation success.
