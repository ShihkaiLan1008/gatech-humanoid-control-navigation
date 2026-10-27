# Humanoid Control and Navigation: Training and Evaluation

![Sky Lan's humanoid robotics portfolio, with a trajectory from a recorded simulation.](assets/research-cover.svg)

**Shih-Kai Lan (Sky) · Summer 2026–present · Ongoing research in simulation**  
Mentored by **Professor Ye Zhao**, Georgia Tech · Directly guided by **Zhenyu Wu**

[Research overview](docs/ADMISSIONS_RESEARCH_OVERVIEW.md) · [My contribution](CONTRIBUTIONS.md) · [Recorded runs](docs/VIDEOS.md) · [AI and source credits](docs/AI_AND_SOURCE_DISCLOSURE.md)

## The research question

**Can a humanoid reach a destination and know that it has arrived?**

The team's research connects learned walking control, tactile sensing, localization and motion planning. My assigned navigation study examined that existing system on new layouts. The recorded pilot exposed a useful distinction: the robot could physically enter its goal region while its navigation system remained uncertain about arrival.

## My contribution

I am a student participant in Georgia Tech's LIDAR research group. **Under my mentors' guidance, I directed the workflow and reporting for my assigned evaluation:** keeping the task aligned with Zhenyu's instructions, supplying context and constraints, examining outputs, requesting explanations and corrections, and discussing progress with the group.

- **Implementation and persistence:** I rebuilt the training environment, worked on a foot-tracking reward modification and completed a 500-iteration diagnostic run with mentor guidance and AI assistance.
- **Experimental judgment:** I identified unequal training sample budgets and proposed matched conditions before claiming improved tracking.
- **Evaluation and communication:** I helped carry the existing-map-then-new-maps assignment through to three new layouts, recordings, measurements and a documented report, using substantial AI implementation and analysis assistance.

[See each contribution alongside its evidence and the division of work →](CONTRIBUTIONS.md)

## The recorded navigation pilot

[![Recorded central-island simulation with the robot, executed path and localization panel. Open the full MP4.](projects/tactile-navigation/outputs/island_video_frame.png)](https://github.com/ShihkaiLan1008/gatech-humanoid-control-navigation/raw/refs/heads/main/projects/tactile-navigation/out/sky_evaluation_20261003/new_maps_physics/footstep/sky_island_choice/seed_0/sky_island_choice_footstep_pf_global_s0.mp4)

*Actual simulation replay, using the team's trained controller. [Download the central-island MP4 (13.3 MiB)](https://github.com/ShihkaiLan1008/gatech-humanoid-control-navigation/raw/refs/heads/main/projects/tactile-navigation/out/sky_evaluation_20261003/new_maps_physics/footstep/sky_island_choice/seed_0/sky_island_choice_footstep_pf_global_s0.mp4) · [All four runs and frame sheets](docs/VIDEOS.md)*

**Four completed trials physically reached the goal region; one met the framework's completion condition. No fall was reported.**

| Layout | Goal distance | Arrival belief | Framework decision |
| :--- | ---: | ---: | :--- |
| Original room | 7.31 cm | 93.35% | Arrival uncertain |
| Central island | 6.41 cm | 92.41% | Arrival uncertain |
| Offset opening | 4.07 cm | 91.89% | Arrival uncertain |
| Staggered baffles | 4.16 cm | 96.04% | Complete |

The system requires at least **95% joint position-and-heading belief** to confirm arrival. That is an internal localization belief, not a measured real-world success probability. Each layout has one seed; these are pilot observations.

[Methods, full results and limits →](docs/ADMISSIONS_RESEARCH_OVERVIEW.md) · [Inspect one complete experiment →](docs/RESEARCH_WALKTHROUGH.md)

## What I learned

The earlier foot-tracking screening did not demonstrate improvement. Recognizing unequal training exposure taught me to distinguish a completed implementation from a supported conclusion. The navigation study added another lesson: visible movement, physical arrival and justified confidence answer different questions.

These are useful questions for robots that may eventually carry supplies, assist people or inspect hazardous spaces. Real-world reliability remains to be tested.

## Research continues

This experience has reaffirmed my love for mechanical engineering by connecting mechanics, control, sensing and computation. I continue conducting experiments and want to contribute more deeply to robotics. My next priorities are repeated trials, matched training comparisons and understanding uncertain arrivals.

[Motivation and future work](docs/MY_RESEARCH_DIRECTION.md) · [Learning to rebuild after a setback](docs/SETBACK_AND_REBUILD.md)

## Credits and AI use

Professor Ye Zhao guided my research direction; Zhenyu Wu provided direct mentorship and assigned the evaluation. Huaze Liu and the team supplied the underlying methods, models and reference maps.

**I supplied prompts, priorities and revision requests. ChatGPT integrated source material and generated much of this portfolio's explanatory text. Codex provided substantial code, simulation-operation, analysis, visualization and documentation assistance.** [Detailed roles and disclosure](docs/AI_AND_SOURCE_DISCLOSURE.md)

Team data, content and images supplied through Zhenyu Wu are included with his permission, as confirmed by me. Original authorship and upstream notices remain credited. [Permission and attribution](CONFIDENTIALITY.md)

## Explore the evidence

[Technical methods](docs/TECHNICAL_DEPTH.md) · [Mentor and assignment context](docs/MENTOR_ALIGNMENT.md) · [Visual sources](assets/SOURCES.md) · [Review and verification](docs/PORTFOLIO_REVIEW.md)

<details>
<summary>Research archive and reproduction</summary>

The archive preserves code snapshots, configurations, models, logs and recordings. Large files use Git LFS:

```bash
git clone https://github.com/ShihkaiLan1008/gatech-humanoid-control-navigation.git
cd gatech-humanoid-control-navigation
git lfs pull
```

Historical reports and writing drafts remain preserved; the current contribution statement and result records define the claims made by this portfolio.

Reproduction also requires the compatible simulation environment described in the [experiment report](projects/tactile-navigation/outputs/Sky%20Navigation%20Experiment%20Report.md). This repository is a preserved evaluation and an ongoing research portfolio, not a one-command simulator installation.

</details>
