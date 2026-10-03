![Research portfolio section](../assets/section-technical.svg)

[← Portfolio home](../README.md) · [Research overview](ADMISSIONS_RESEARCH_OVERVIEW.md) · [Technical depth](TECHNICAL_DEPTH.md) · [Video gallery](VIDEOS.md)

# Technical depth: motion, belief and completion

## How the parts work together

1. The authored map loader transforms room geometry and endpoints into the navigation frame. The island layout exercises a different authored initial heading.
2. The global planner searches for a route respecting obstacles and uncertainty bounds, with seed/refine planning and a belief-A* fallback. A planned route passes segment clearance and confidence checks.
3. The learned G1 footstep controller executes movement in MuJoCo/Warp. No new controller was trained for these maps.
4. A particle-filter belief tracks possible robot poses. Movement prediction propagates uncertainty; touch observations can constrain the pose estimate. The expected-information-gain sensing strategy chooses probes within the shared reach profile.
5. Execution and final-arrival guards use belief conditions. Simulator truth is logged for evaluation, not supplied to the high-level navigation estimate.

## Keep the four questions distinct

| Question | Evidence |
|---|---|
| Is the planned route feasible? | Planner checks of collision clearance and uncertainty bounds |
| Did the robot physically arrive? | Final simulator-truth position lies inside the configured goal disk |
| Was localization accurate throughout? | Time-weighted RMSE between estimated and true planar position |
| Did the framework certify completion? | Runtime completion outcome, belief-based arrival requirement, physical goal membership and no fall |

## Metric definitions

For final true planar position `p_T`, goal center `g` and radius `r`, physical arrival means `||p_T - g|| <= r`. Goal distance is the Euclidean distance `||p_T - g||`, measured in metres.

The final runtime particle-filter arrival probability is the sum of particle weights that jointly satisfy the goal disk and final yaw bound. The required mass is 0.95. This value is an internal belief under the filter's model; it is not a statistical confidence interval and the experiment does not establish its calibration as a real-world success probability.

Localization RMSE uses time integration: `sqrt(integral ||estimate_xy(t) - truth_xy(t)||^2 dt / scored_duration)`. The saved samples are integrated with trapezoidal weights. Instantaneous belief corrections can have duplicate timestamps and zero duration. Missing intervals are excluded with coverage reported. It is not footstep or landing error.

Canonical framework success requires reported completion, physical goal membership and no fall. In these records, the final arrival guard explains the `arrival_uncertain` outcomes. Contact attempts and actual contacts are recorded separately; a failed contact attempt is not counted as a sensing success.

## Controlled settings and provenance

Seed 0, the same team model 35500, PF localization, EIG probes and angle-dependent reach profile were used. Shared settings include a 0.18 m position bound/trigger, 20 degree yaw bound, 0.30 m standoff, 0.10 m planner grid and 0.375 m maximum planner movement. The 240/900 s cap difference and restart resource precautions are disclosed in the full report.

`PROVENANCE.json` gives source revisions and remotes; `SOURCE_MANIFEST.json` records archived file SHA-256 hashes. The navigation metrics record also identifies the checkpoint and map/source hashes. `scripts/sky_validate_results.py` independently recomputes headline metrics from raw samples, and `sky_validation.json` preserves the results. Geometry tests and physical execution are separate checks.

## Research questions with depth

- **Calibration:** Across repeated trials, does 95% arrival belief actually correspond to correct joint position/heading membership at the expected frequency? A single seed cannot answer this.
- **Sensing:** Does an extra final probe improve confidence and calibrated completion enough to justify its time cost? Compare policies while holding map, seed and checkpoint fixed.
- **Geometry:** How do opening width and turn count affect execution outcomes and localization error? Change one factor at a time and replicate trials.
- **Failure attribution:** Is uncertainty due to position, yaw, missed contacts or filter assumptions? Inspect the joint arrival condition and contact history rather than lowering the threshold solely to pass.
- **Transfer:** Does the behavior remain stable with real sensing noise and contact mechanics? Simulation results alone cannot establish this.

The difficult technical lesson is that perception, planning and control must be evaluated together, while their failure criteria remain distinguishable.
