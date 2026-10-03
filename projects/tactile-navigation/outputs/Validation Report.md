# Validation report

## Overall assessment: Share with caveats

The 3 October 2026 (Asia/Taipei) pilot answers whether the existing tactile-navigation framework can execute on three new map layouts. Four completed physical-simulation records support four physical goal arrivals, one framework success, and no recorded falls. These counts describe the recorded trials; they are not estimates of a general success rate.

## Methodology review

The campaign uses one original room and three authored maps, seed 0, the same local model 35500 and shared navigation settings. The earlier interrupted offset-gate record is preserved separately as `system_restart`, not counted twice with its restarted run. Room2 has a 240 s cap and new maps a 900 s cap; each completed before its cap. No statistical comparison between map populations or causal inference is justified.

## Issues and caveats

1. Single seed per map: insufficient for a robustness estimate or evidence that one geometry is easier than another.
2. Default model 34000 was missing; model 35500 was used consistently and recorded. This is not exact reproduction of the default-checkpoint campaign.
3. Physical arrival and confident completion differ. The three `arrival_uncertain` episodes remain visible. A 95% belief-mass threshold is not a statistical confidence interval or demonstrated real-world success probability.
4. All experiments were simulation. No real robot trial, new model training or publication occurred.
5. Windows crash evidence identifies a kernel bugcheck category, not a specific responsible driver or a permanent repair. Resource precautions do not prove crash causation.

## Calculation spot-checks

`scripts/sky_validate_results.py` independently computes final Euclidean goal distance and time-integrated localization RMSE from raw samples, without importing the report's metrics function. It also checks physical goal membership, completed/no-fall success, contact counts, arrival probabilities from replay transitions, unique map/seed pairs and separate interrupted telemetry.

All four records passed. Exact audit values and sample counts are saved in `sky_validation.json`. Goal errors span 4.07–7.31 cm; localization RMSE spans 3.48–6.83 cm. Final arrival probabilities are 93.35%, 92.41%, 91.89% and 96.04% in report order. The last meets the common 95% requirement.

Reproduce the check from the existing WSL environment:

```bash
/home/sky/.venvs/tactile_controller/bin/python scripts/sky_validate_results.py
```

The focused geometry/import suite passed all 12 checks. It verifies map loading, transformed endpoints/clearance and agreement of analytic and simulator obstacle counts; it does not by itself prove navigation success.

## Visualization review

The refreshed four-panel trajectory figure was visually inspected. It contains all four completed trials, true and estimated paths, obstacles, goal disks, labeled axes in metres, and outcome labels. Each panel uses equal x/y aspect; rooms have different displayed extents, so avoid judging relative distance solely by image size. Recorded high-frequency root path length includes oscillations and should not be compared directly to planned distance as a route-efficiency score.

## Suggested improvements

Repeat the fixed configuration across seeds, then vary one geometric factor at a time. Inspect saved belief distributions, contact attempts and final yaw/position uncertainty before suggesting changes to arrival logic. Preserve all outcomes and use matched budgets for future comparisons.

## Handoff

No missing metrics block sharing this pilot with its caveats. Video integrity and sampled-frame inspection are documented separately in `Research Completion.md` once rendering finishes. The college-application guide contains evidence-based technical wording and prompts for the student's own reflection; personal authorship and experiences must be supplied by the student.
