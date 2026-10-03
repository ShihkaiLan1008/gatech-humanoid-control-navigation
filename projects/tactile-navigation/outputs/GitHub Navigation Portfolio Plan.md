# GitHub portfolio: tactile navigation on new maps

## Main idea

Present a focused research case study: **Can a humanoid navigation framework using touch generalize to layouts beyond the team's original maps?** Show the question, your map designs, the experiment, results, and what you learned. Keep the first screen readable for an admissions reviewer, with links to technical evidence for readers who want more detail.

## Profile and project

Use a short GitHub profile README linking to one pinned project, `tactile-navigation-map-study`. Describe your interests and role accurately. The project's opening should contain a map comparison, a short physical-simulation video, a contribution statement, and the main finding. GitHub supports profile READMEs and pinned repositories: https://docs.github.com/en/account-and-profile/how-tos/profile-customization

Suggested structure:

```text
tactile-navigation-map-study/
  README.md
  CONTRIBUTIONS.md
  ACKNOWLEDGEMENTS.md
  maps/
    sky_offset_gate.json
    sky_staggered_baffles.json
    sky_island_choice.json
  experiments/
    settings.json
    reproduction.md
  evaluation/
    scripts/
  results/
    metrics.csv
    trajectories.png
  media/
    navigation-demo.mp4
  docs/
    map-design.md
    failure-analysis.md
    research-report.md
```

Include credited team source only within the permission you described. Keep the case study focused even if a separate approved fork provides the full framework.

## Contribution statement

Separate the team's foundation from your extension:

| Team foundation | This extension |
|---|---|
| Tactile-navigation framework and existing maps | Three new layouts testing detours, repeated turns, and route choice |
| Global planner, particle filter, EIG sensing | Evaluation under common settings across the new maps |
| Trained G1 controller and MuJoCo integration | Execution records, videos, quantitative results, and analysis |

The maps, scripts, and analysis here were prepared with Codex assistance. Review them, rerun an experiment, and make sure you can explain the choices before describing them as your work in an application. A useful disclosure is: “I used AI assistance to inspect the codebase, implement map files and evaluation tools, and organize results; I reviewed the designs and verified the experiments.” Adapt that to what you actually do.

Credit Huaze Liu, Zhenyu Wu, and the research team by their preferred names and affiliation. Preserve original authorship and license notices for approved reused source. Your statement that Zhengyu allows GitHub sharing is the basis for this plan; the screenshots additionally show permission to include team work in a report. Keep private chat screenshots outside the public portfolio.

## Evidence reviewers can understand

Lead with a 30–60 second video excerpt showing a robot navigating one new layout, with a clear “simulation” label and the actual run identifier. Link the full replay and results. Do not replace failed episodes with staged successful motion.

Show a table with final goal error, physical goal membership, framework success, falls, localization RMSE, and contact attempts. Explain the distinction between physically reaching the goal and satisfying the framework's confidence criterion. An `arrival_uncertain` result is a finding, not something to hide.

Include exact map files, shared settings, seed, checkpoint identity, code revision, and commands. The current local runs use a model 35500 override because the README's default model 34000 was not included; make that visible. One seed per map supports a pilot study, not a general robustness claim.

## Research narrative

1. **Motivation:** The team had tested only a small set of layouts; new geometry can reveal generalization limits.
2. **Design:** Explain why you chose an offset divider, alternating baffles, and a central island.
3. **Method:** Hold the navigation framework and controller settings fixed while changing the maps.
4. **Observation:** Use measured results and videos to describe what worked and what did not.
5. **Interpretation:** Separate planner feasibility, physical execution, localization error, and completion confidence.
6. **Next experiment:** Repeat multiple seeds, then systematically change opening widths or obstacle locations.

## Application wording draft

> Evaluated a team's touch-based humanoid navigation framework on newly designed simulation maps. Built three layouts to test detours, repeated turns, and route choice; executed the existing G1 controller and analyzed localization, contact sensing, and arrival outcomes using recorded metrics and videos.

Use this wording only after reviewing and understanding the work. Add quantitative improvements only when an appropriate comparison supports them. Avoid claiming that you invented the navigation framework or established robustness from these initial trials.

## Build order

Finish and review the pilot report, select one clear video, and write your contribution statement. Assemble the case-study repository with the permitted materials. Add the profile README and pin the project. A portfolio website can follow once the evidence and project narrative are ready. Nothing in this plan has been published to GitHub.
