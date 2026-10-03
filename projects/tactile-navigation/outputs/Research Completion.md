# Navigation-map pilot: completed handoff

Completed 3 October 2026, Asia/Taipei, after the user requested resumption. This supersedes the historical pause instructions in `Research Paused.md`.

## Completed work

- Added and checked offset-gate, staggered-baffle and island-choice maps on the requested `sky/evaluation` branch.
- Completed one original-map trial and three new-map trials with the existing physical G1 simulation controller.
- Refreshed the report, CSV/JSON metrics, hashes and four-panel executed-trajectory figure.
- Independently recomputed key results from all four raw records. All checks passed; interrupted telemetry remains separate.
- Passed all 12 focused map geometry/import checks.
- Rendered all four full replay videos, completely decoded them and visually inspected start/middle/end sheets. Robot, room geometry, trajectory/diagnostic panel and terminal outcomes appeared correctly. Walls occasionally occlude the robot in the camera view; the trajectory panel remains available.
- Prepared the GitHub portfolio brainstorming and a college-application writing guide with section instructions, a verified results table, an English conclusion draft, contribution guidance and personal reflection prompts.

## Result

All four completed trials ended inside their configured physical goal region, with no recorded falls. Staggered baffles met formal framework success; the other three ended `arrival_uncertain` below the common 95% arrival-confidence requirement. These are single-seed pilot observations, not a demonstrated general robustness rate.

## Start here

- [College admission writing instructions](College%20Admission%20Research%20Writing%20Guide.md)
- [Full experiment report](Sky%20Navigation%20Experiment%20Report.md)
- [GitHub portfolio plan](GitHub%20Navigation%20Portfolio%20Plan.md)
- [Validation report](Validation%20Report.md)
- [Independent numeric audit](sky_validation.json)
- [Video integrity audit](sky_video_validation.json)
- [Trajectory figure](sky_navigation_trajectories.png)

The writing guide links all four videos. Full-decoded frame counts in report order: 943, 645, 774 and 1,063. Video playback durations are 188.6, 258.0, 309.6 and 425.2 seconds; use canonical evaluation durations for quantitative simulation time, as replay sampling and end frames affect video length.

## What remains outside this completed pilot

The student should review the work and add their actual personal contribution and reflection before writing a first-person application essay. A future robustness study requires more seeds and systematic geometry variation. The Windows driver responsible for the earlier kernel crash remains unidentified; no permanent repair is claimed. No GitHub publication, Slack message, model training or real-robot testing was performed.

All simulation, rendering and video-audit jobs for this handoff have finished. The original controller checkout and checkpoints remain preserved.
