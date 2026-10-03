![Sky Lan — Walking toward understanding. Robotics research portfolio with a recorded simulation trajectory.](assets/research-cover.svg)

<p align="center">
  <strong>Humanoid robotics &nbsp;·&nbsp; Mechanical engineering &nbsp;·&nbsp; Research in simulation</strong><br>
  <sub>Private portfolio · team-supported work · AI assistance disclosed</sub>
</p>

<p align="center">
  <a href="docs/ADMISSIONS_RESEARCH_OVERVIEW.md"><strong>START HERE</strong></a> &nbsp; / &nbsp;
  <a href="docs/VIDEOS.md">WATCH THE ROBOT</a> &nbsp; / &nbsp;
  <a href="docs/TECHNICAL_DEPTH.md">GO DEEPER</a> &nbsp; / &nbsp;
  <a href="docs/MY_RESEARCH_DIRECTION.md">WHY I CARE</a>
</p>

---

## Can a robot know when it has arrived?

A robot has a map and a destination—but its estimate of its own position can drift as it walks. Touching a wall can help it correct that estimate. The challenge is to move successfully **and gather enough evidence to confirm arrival**.

Zhenyu's assignment was to test the team's navigation framework on new layouts. This extension introduced three maps, checked their geometry, ran the existing humanoid controller and preserved the results for analysis.

![Pilot study: three new maps, four completed trials, four physical arrivals, one formal success. One seed per map.](assets/pilot-at-a-glance.svg)

**The finding:** all four completed trials entered the physical goal region, but three ended with insufficient arrival confidence. That gap between motion and certainty is the research question worth pursuing.

[Read the accessible research overview →](docs/ADMISSIONS_RESEARCH_OVERVIEW.md)

## From code to movement

[![A recorded G1 simulation navigating the central-island map, beside its trajectory and localization diagnostics. Click to open the full replay.](projects/tactile-navigation/outputs/island_video_frame.png)](projects/tactile-navigation/out/sky_evaluation_20261003/new_maps_physics/footstep/sky_island_choice/seed_0/sky_island_choice_footstep_pf_global_s0.mp4)

<p align="center"><sub>Central-island experiment · actual recorded simulation · full replay linked above</sub></p>

<p align="center">
  <a href="docs/VIDEOS.md"><strong>Explore all four replays →</strong></a> &nbsp; · &nbsp;
  <a href="projects/tactile-navigation/outputs/sky_navigation_trajectories.png">Compare the executed paths →</a>
</p>

## What the results reveal

![Final arrival beliefs: original room 93.35%, central island 92.41%, offset opening 91.89%, staggered baffles 96.04%. Only staggered baffles clears the common 95% criterion.](assets/arrival-confidence.svg)

The threshold requires enough localization belief to satisfy **both position and heading**. It is an internal belief criterion. Each map was tested with one seed, so these observations are a pilot study rather than a general success-rate estimate.

[Methods and metric definitions →](docs/TECHNICAL_DEPTH.md) &nbsp; · &nbsp; [Full report and limitations →](projects/tactile-navigation/outputs/Sky%20Navigation%20Experiment%20Report.md)

## My contribution within the team project

| The team's foundation | This research extension |
| :--- | :--- |
| Navigation algorithms, localization and sensing strategy | Three new layouts testing detours, repeated turns and route choice |
| Trained G1 walking controller and simulator integration | Execution, recordings, geometry checks and quantitative evaluation |
| Original maps and research code | A reproducible case study with videos, analysis and open questions |

**Credit:** Zhenyu Wu, Huaze Liu and the team provided the framework and models. Codex assisted with code inspection, map/script implementation, experiment operation, analysis and writing. [Detailed attribution →](CONTRIBUTIONS.md)

## The curiosity behind the work

At a 2024 Georgia Tech / CIC Asia Future Tech Forum event, I saw professors, university students and high-school students using engineering to contribute to the world. Seeing students close to my age participate made research feel possible for me.

> Watching a robot learn to walk felt like a parent watching a child take its first steps.

That connection between an input and a physical movement drew me toward mechanical engineering. I want to explore more adaptive walking, movement knowledge that can transfer between robots, and cleaner, more efficient use of energy.

[My story and the questions I want to pursue →](docs/MY_RESEARCH_DIRECTION.md)

## A growing research journey

<table>
<tr>
<td width="33%" valign="top">
<h3>01 · Locomotion</h3>
<p><strong>Independent G1 exploration</strong></p>
<p>Rough-terrain training and a custom command that guides the robot toward its environment origin.</p>
<p><a href="projects/independent-g1-rough-terrain/README.md">Code, model &amp; training record →</a></p>
</td>
<td width="33%" valign="top">
<h3>02 · Measurement</h3>
<p><strong>Tactile controller experiments</strong></p>
<p>Reward changes, training artifacts and an unfavorable foot-tracking result preserved for learning.</p>
<p><a href="projects/tactile-controller/RETROSPECTIVE.md">Experiment retrospective →</a></p>
</td>
<td width="33%" valign="top">
<h3>03 · Navigation</h3>
<p><strong>New maps, new questions</strong></p>
<p>Different obstacle layouts reveal the relationship between physical arrival and localization confidence.</p>
<p><a href="docs/ADMISSIONS_RESEARCH_OVERVIEW.md">Featured case study →</a></p>
</td>
</tr>
</table>

## Turning ambition into experiments

| Next step | The question it can answer |
| :--- | :--- |
| **Repeat** · more seeds, matched budgets | Which findings persist beyond a single run? |
| **Explain** · inspect belief and contact records | What causes arrival uncertainty, and can final sensing help? |
| **Transfer** · another robot body or terrain | How much movement knowledge can be reused with less retraining? |
| **Measure** · energy alongside stability | What tradeoffs connect useful movement and energy consumption? |

Adaptive walking, broad transfer and cleaner energy are future directions. The current evidence establishes a simulation case study and a starting point for those questions.

---

<details>
<summary><strong>Research evidence, limitations and application writing</strong></summary>

- [Full navigation report](projects/tactile-navigation/outputs/Sky%20Navigation%20Experiment%20Report.md) · [Numeric audit](projects/tactile-navigation/outputs/sky_validation.json) · [Video audit](projects/tactile-navigation/outputs/sky_video_validation.json)
- [Source inventory and hashes](SOURCE_MANIFEST.json) · [Source revisions](PROVENANCE.json) · [Graphic sources](assets/SOURCES.md)
- [College-application report and conclusion guide](projects/tactile-navigation/outputs/College%20Admission%20Research%20Writing%20Guide.md)
- Scope: one seed per map, simulation only, team model 35500 substituted for the missing default 34000. The original room and new maps had different time caps; the system-interrupted attempt remains separate.
- The earlier controller screening did not demonstrate improvement. The independent G1 package does not establish guaranteed fall prevention or energy-efficiency gains.

</details>

<details>
<summary><strong>Private archive and reproduction</strong></summary>

This repository preserves source snapshots, models, datasets, logs, reports and recordings. Large artifacts use Git LFS:

```bash
git clone https://github.com/azureowl372/ShihKai-Lan-Research-Project-with-Georgia-Tech-LIDAR-LAB..git research-retrospective
cd research-retrospective
git lfs pull
```

Dependencies and original local paths are documented by project. The archive is private and read-only; authorized updates require unarchiving it. Admissions readers need an approved excerpt or explicitly granted access. [Confidentiality and access →](CONFIDENTIALITY.md)

</details>

<p align="center"><sub>Ask a question. Build an experiment. Keep the evidence. Continue learning.</sub></p>
