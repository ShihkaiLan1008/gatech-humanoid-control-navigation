![Research portfolio section](../assets/section-technical.svg)

[← Portfolio home](../README.md) · [Research overview](ADMISSIONS_RESEARCH_OVERVIEW.md) · [Technical depth](TECHNICAL_DEPTH.md) · [Video gallery](VIDEOS.md)

# Technical depth: motion, belief and completion

The technical difficulty spans representation, learning, and evaluation: making several systems describe the same physical task, choosing a training signal, and deciding what a recorded run actually proves. This page explains five concrete challenges using preserved code and results. The team supplied the underlying research methods; map extensions, implementation, and analysis received substantial Codex assistance under Sky's direction.

For the complete system diagram, the role of contact and miss information, the execution-policy tradeoff, and a recorded episode timeline, start with [Inside the research](RESEARCH_WALKTHROUGH.md).

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

## Five challenges behind the results

### 1. A new map must remain the same room in every representation

The [central-island manifest](../projects/tactile-navigation/maps/sky_island_choice.json) specifies a start at `(0.7, 2.0)`, a goal at `(4.3, 2.0)`, and a starting heading of `0°` in the authored room. The existing loader translates the start to the origin and rotates the heading to navigation `+y`. Consequently, the authored goal displacement `(3.6, 0)` becomes `(0, 3.6)` in the navigation frame.

This is not just a plotting choice. Endpoints, obstacles, and free space must undergo compatible transforms so that a feasible path in the planner corresponds to an unobstructed path in the simulator. Wall thickness and obstacle identities also matter when the robot uses contact to infer where it is. A visually plausible room alone cannot validate these relationships.

The [new-map tests](../projects/tactile-navigation/tests/demo_map/test_sky_maps.py) check transformed endpoints, at least 0.6 m endpoint clearance, and agreement in planner/simulator obstacle counts. The existing [manifest tests](../projects/tactile-navigation/tests/demo_map/test_manifest.py) also enforce assumptions such as axis-aligned free-space rectangles. These tests validate specific invariants; they do not prove that every possible geometry is correct. The three layouts exercised the team's [existing loader](../projects/tactile-navigation/src/demo_map/manifest.py), rather than introducing a new coordinate-transform algorithm.

**Next experiment:** vary opening width while keeping the coordinate convention, obstacle construction, checkpoint, and other settings fixed. Check geometry first, then compare actual execution. This helps distinguish a difficult navigation case from an invalid map.

### 2. A sharper reward changes the learning problem, not the outcome by definition

The earlier foot-tracking experiment added `track_feet_pos_precise` with weight `1.0` and scale `0.05 m`, alongside the original `0.20 m` position term. The [reward configuration](../projects/tactile-controller/src/rl/decoupled/tasks/joint_foot_mimic_arm/env_configs/reward_cfgs.py) uses the family's exponential form:

`r(e) = exp(-sum(e²) / σ²)`

A smaller scale makes the reward more selective near the target, but gives much less reward at larger error. For an illustrative aggregate error norm of 0.10 m, the kernel is about **0.78** with `σ = 0.20 m` and **0.018** with `σ = 0.05 m`. These are mathematical examples, not measured episode rewards. Keeping the broader term preserves a coarse signal, while the extra term expresses a hypothesis about improving precision. It does not guarantee the optimizer will find a better policy within a particular training budget.

The later [screening report](../projects/tactile-controller/outputs/Mission%20Review%20and%20Work%20Report.md) measured **9.62 cm** mean reference-foot error for the reference checkpoint and **33.02 cm** for the tuned checkpoint. This was reference tracking across frames, not a measure of landing accuracy. The screening used seed 42, 32 environments, and 500 steps, but the checkpoints had different training histories; resets can also lead to divergent command trajectories. It did not establish improvement or isolate the reward change as the cause of the difference.

Resource limits make that comparison harder. The [saved training configuration](../projects/tactile-controller/logs/rsl_rl/g1_decoupled_joint_foot_mimic_arm/2026-09-13_22-38-08_foot_precise_s005_500/params/agent.yaml) specifies 24 steps per environment and 500 iterations, with [1,024 environments](../projects/tactile-controller/logs/rsl_rl/g1_decoupled_joint_foot_mimic_arm/2026-09-13_22-38-08_foot_precise_s005_500/params/env.yaml). At those settings, the nominal rollout budget is `1,024 × 24 × 500 = 12,288,000` transitions. A run with 8,192 environments at the same steps and iterations would collect eight times as many. Iteration counts alone therefore cannot establish equal training exposure. Equal transition counts would still not guarantee convergence or control all other differences.

**Next experiment:** start both reward variants from the same checkpoint, match training exposure and evaluation commands, repeat seeds, and report stability and episode outcomes alongside tracking error. Keep the unfavorable screening visible while testing the hypothesis more fairly.

### 3. Accurate movement and justified arrival are different conditions

The offset-opening trial ended **4.07 cm** from the goal with **91.89%** arrival belief. The staggered-baffle trial ended **4.16 cm** away with **96.04%** belief. The slightly closer robot failed the 95% criterion, while the other passed. These [validated results](../projects/tactile-navigation/outputs/sky_validation.json) show why final distance alone cannot explain formal completion.

The team's [arrival guard](../projects/tactile-navigation/src/execution/route_guard.py) sums the weights of particles that satisfy both the positional goal region and the wrapped yaw-error bound:

`arrival_mass = Σ_i w_i · 1[position_i inside goal AND abs(wrapped_yaw_error_i) <= yaw_bound]`

The same particle must satisfy both conditions. Position confidence and heading confidence cannot simply be checked separately and assumed to establish their joint condition. The value also describes the filter's own modeled belief. It does not become a calibrated probability of real-world success merely because the threshold is 95%.

The current pilot does not establish whether position spread, heading spread, unsuccessful touches, or a filter-model mismatch explains the remaining uncertainty. Lowering the threshold until every run passes would change the success definition without resolving that question.

**Next experiment:** record positional mass, heading mass, and joint mass separately around arrival; compare an added final sensing action against the unchanged policy using matched seeds. Check whether any increased confidence corresponds to correct position and heading against simulator truth. Calibration requires repeated trials, not these four observations alone.

### 4. The sampling pattern must not silently redefine the metric

The [independent validation script](../projects/tactile-navigation/scripts/sky_validate_results.py) recomputes planar localization error from logged simulator-truth and estimated positions. Logging is not necessarily uniform in time: motion steps and instantaneous belief corrections can generate different sampling densities, including duplicate timestamps.

Simply averaging squared error across rows could give a frequently logged moment disproportionate influence. The audit therefore integrates squared error over elapsed time using trapezoidal weights, then divides by scored duration before taking the square root. Duplicate timestamps add zero elapsed time. Intervals lacking the required positions are excluded. The campaign's [evaluation module](../projects/tactile-navigation/src/execution/evaluation.py) separately reports estimation and truth coverage in the [saved results](../projects/tactile-navigation/outputs/sky_results.json), so readers can see how much of each trace was scored.

This measurement answers how accurately the system localized itself over time. It does not measure foot placement, establish physical arrival, or replace the runtime stopping rule. The evaluation also keeps attempted probes separate from actual contacts. These distinctions make the saved evidence interpretable beyond the replay video.

**Next experiment:** examine error and belief immediately before and after contacts, while reporting coverage and the same metric definitions across runs. This could identify where sensing helps, without claiming from contact counts alone that more touches cause better localization.

### 5. Recovery must preserve what an experiment can still prove

A separate Windows restart interrupted the first offset-opening attempt. The [recovery record](../projects/tactile-navigation/outputs/crash_diagnostics/Crash%20and%20Recovery.md) preserves that attempt as `system_restart`; it is not an additional completed trial. Finished runs were retained, and reruns went into a new directory. The campaign consistently used the available team checkpoint `35500` because the documented default `34000` was absent. Reporting that substitution makes this an evaluation of a specified checkpoint, not an exact replication of an unavailable default.

Restart precautions limited physical simulations to one at a time, used two CPU cores and a 4 GiB process-group memory watchdog, and rendered video afterward. These were workload controls, not proof of a repaired driver or protection against every hardware failure. The earlier board failure described in [Sky's personal account](SETBACK_AND_REBUILD.md) is a distinct event whose cause was not established.

**Next step:** retain a run manifest, configuration, checkpoint identity, logs, and intermediate outputs together; test restoration from a separate backup before the next long campaign. A checkpoint and a conversation serve different recovery purposes. Reconstructing setup instructions cannot reconstruct lost learned parameters.

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
