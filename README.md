# Humanoid Control and Navigation: Training and Evaluation

![Sky Lan — Walking toward understanding. Robotics research portfolio with a recorded simulation trajectory.](assets/research-cover.svg)

<p align="center">
  <strong>Humanoid robotics &nbsp;·&nbsp; Mechanical engineering &nbsp;·&nbsp; Research in simulation</strong><br>
  <sub>Shih-Kai Lan (Sky) · student team member · simulation research portfolio</sub>
</p>

<p align="center">
  <a href="docs/ADMISSIONS_RESEARCH_OVERVIEW.md"><strong>START HERE</strong></a> &nbsp; / &nbsp;
  <a href="CONTRIBUTIONS.md"><strong>MY CONTRIBUTION</strong></a> &nbsp; / &nbsp;
  <a href="docs/RESEARCH_WALKTHROUGH.md"><strong>INSIDE THE RESEARCH</strong></a> &nbsp; / &nbsp;
  <a href="docs/VIDEOS.md">WATCH THE ROBOT</a> &nbsp; / &nbsp;
  <a href="docs/MY_RESEARCH_DIRECTION.md">WHY I CARE</a> &nbsp; / &nbsp;
  <a href="docs/AI_AND_SOURCE_DISCLOSURE.md">AI &amp; SOURCE CREDITS</a>
</p>

---

## Research status

**Summer 2026–present · Ongoing research and experiments.**

I am a student participant in Georgia Tech's LIDAR research group under **Professor Ye Zhao's mentorship and Zhenyu Wu's direct guidance**. This portfolio records the experiments preserved so far; my research continues.

## The assignment and my role

**How can a humanoid reach a destination and know that it has arrived?** My assigned navigation study tested the team's existing system on an original map and three new layouts.

**I directed the workflow and reporting for my assigned evaluation under my mentors' guidance.** I kept the work focused on the assignment, supplied the research context and constraints, examined the recordings and measurements, and requested follow-up explanations and corrections. I discussed training quality and research scope with the group and shared a weekly report. [Assignment and participation record →](docs/MENTOR_ALIGNMENT.md)

Earlier, I rebuilt the training environment, worked on a foot-tracking reward modification and completed a 500-iteration diagnostic run. I identified an unequal training sample budget and proposed matched conditions before claiming improvement. These contributions combined implementation with experimental judgment, using mentor guidance and AI assistance. [Contribution evidence →](CONTRIBUTIONS.md)

The team supplied the controller, navigation methods, trained models and reference materials. ChatGPT and Codex provided substantial writing and technical assistance; their specific roles are stated in the [AI and source disclosure](docs/AI_AND_SOURCE_DISCLOSURE.md).

## My research contributions and why they matter

- **Turning an assignment into a documented evaluation:** I took on Zhenyu's existing-map-then-new-maps task, directed the AI-assisted workflow, examined the outcomes and shaped the report. The resulting baseline and three new-map trials provide specific cases for further investigation. [Study and recorded results →](docs/ADMISSIONS_RESEARCH_OVERVIEW.md)
- **Making experiments runnable:** I rebuilt a failed software environment, worked through compatibility issues and checked command and data pipelines with guidance and AI assistance. This restored a working basis for training and evaluation. [Environment and recovery record →](docs/SETBACK_AND_REBUILD.md)
- **Implementing a testable change:** I added a narrower foot-position reward and completed a diagnostic training run within an 8 GB GPU constraint. The [reward change](projects/tactile-controller/local-reward-change.patch), [training measurements](projects/tactile-controller/report-evidence/experiment_evidence.json) and saved model make the experiment inspectable.
- **Judging what the evidence supports:** I recognized that 1,024 parallel environments supplied one eighth of the samples of an 8,192-environment setup at the same iteration count. I kept the improvement question open and proposed matched training conditions. [Experimental reasoning and evidence →](CONTRIBUTIONS.md#controller-implementation-and-experimental-judgment)

## Why this research matters

I want to contribute to a future in which robots can help people with everyday physical tasks and take on work in places that are difficult or dangerous to reach. For that future to be useful, robots need to move through different layouts and recognize when their position estimates are uncertain. My current contribution is a simulation pilot toward that ambition.

This research has reaffirmed my love for mechanical engineering by connecting mechanics, control, sensing and computation. I continue conducting experiments and want to develop the knowledge and judgment to contribute more deeply to robotics.

[The future contribution that motivates this work →](docs/ADMISSIONS_RESEARCH_OVERVIEW.md#why-study-this)

## The completed navigation pilot

![Pilot study: three new maps, four completed trials, four physical arrivals, one formal success. One seed per map.](assets/pilot-at-a-glance.svg)

**The finding:** all four completed trials entered the physical goal region, but three ended with insufficient arrival confidence. The study raised a question about what evidence the system needs to confirm arrival; four single-seed runs do not establish general robustness.

[Read the research overview →](docs/ADMISSIONS_RESEARCH_OVERVIEW.md)

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

## Who contributed what

| Contributor | Role across the work shown here |
| :--- | :--- |
| **Me — Shih-Kai Lan (Sky)** | Directed the workflow and reporting for my assigned evaluation under mentor guidance; contributed environment reconstruction, preliminary G1 training, a foot-tracking reward modification and diagnostic run, unequal-sample-budget analysis, result review and team communication. Implementation and analysis used AI assistance. |
| **Professor Ye Zhao — research mentor** | Guidance on research direction, learning resources and a suitable opportunity within the team. |
| **Zhenyu Wu — direct mentor** | Guided controller training and foot-tracking evaluation, assigned the new-map study, and provided practical instructions, setup help and research resources. |
| **Huaze Liu and the research team** | The existing navigation framework, trained controller, shared code and reference maps. |
| **ChatGPT and Codex — tools I directed** | ChatGPT integrated source material and generated much of the explanatory text from my prompts. Codex performed substantial map/script implementation, code inspection, simulation operation, quantitative analysis, visualization and documentation. |

## How this portfolio was produced

I supplied the task context, prompts, priorities and revision requests, examined outputs and shaped the reporting. **The portfolio's narrative was substantially generated with ChatGPT assistance; its technical work also received substantial Codex assistance.** The [detailed contribution and source record](docs/AI_AND_SOURCE_DISCLOSURE.md) explains this workflow alongside my documented participation.

Team data, reference content and images supplied through **Zhenyu Wu** are included with his permission, as confirmed by me, for this public GitHub portfolio. Original authorship and upstream notices remain credited. [Permission and attribution →](CONFIDENTIALITY.md)

## Follow one recorded experiment

The system connects a known map, a global route, a learned walking controller, a particle-filter position estimate, and hand probes selected for information. A failed touch attempt can still constrain the estimate. Final completion then requires enough belief to satisfy position and heading together.

![Recorded localization error and contact/miss events during the completed offset-opening trial](assets/offset-opening-trace.png)

*One executed seed-0 trial, using the team checkpoint. The figure shows recorded localization error and probe outcomes; temporal association alone does not establish a probe's effect.*

The offset-opening robot ended **4.07 cm** from its goal but failed the 95% joint-arrival criterion. In everyday terms, it arrived physically, but its estimate of position and direction was not confident enough for the software to declare success. Comparing the video with these measurements makes the evaluation useful: it identifies a case the team can investigate further. See the [short interpretation and proposed follow-up](docs/ADMISSIONS_RESEARCH_OVERVIEW.md#reading-one-result-and-planning-the-next-test). The [complete walkthrough](docs/RESEARCH_WALKTHROUGH.md) follows that trial through the system diagram, map design, raw records, measurements, unanswered questions, and a controlled next experiment.

## The curiosity behind the work

At a Georgia Tech / CIC Asia Future Tech Forum event, I saw professors, university students and high-school students using engineering to contribute to the world. Seeing students close to my age participate made research feel possible for me.

> Watching a robot learn to walk felt like a parent watching a child take its first steps.

That connection between an input and a physical movement drew me toward mechanical engineering. I want to explore more adaptive walking, movement knowledge that can transfer between robots, and cleaner, more efficient use of energy.

[My story and the questions I want to pursue →](docs/MY_RESEARCH_DIRECTION.md)

## A growing research journey

### The work I had to rebuild

Before the new-map study, I used AI tools to learn how to set up a simulation environment and begin robot training. My computer later suffered an internal board failure, and I lost the training work on that machine. I suspected the workload might have contributed, but its cause was never established. On a better laptop, I used previous conversations and the context I could reconstruct to direct AI through rebuilding the environment and finding settings that better fit the available resources.

This setback is part of my personal account. The surviving records show training, checkpoints, and a separate later navigation restart; they do not establish the cause of the earlier hardware failure. Rebuilding the environment also did not mean recovering every lost file. The experience gives me a concrete engineering goal: make the process understandable, measurable, and recoverable.

[Read the setback, recovery, and lessons →](docs/SETBACK_AND_REBUILD.md)

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

### Where the technical difficulty lies

The work raises questions beyond making a robot move. A map must mean the same thing to the planner and the simulator. A narrower foot-tracking reward needs a fair comparison before anyone can claim it helps. A robot that ends **4.07 cm** from its target can still fail the arrival-confidence test while another **4.16 cm** away passes. These examples connect geometry, learning, measurement, and uncertainty to specific evidence in the archive.

[Five technical challenges, their evidence, and the next experiments →](docs/TECHNICAL_DEPTH.md#five-challenges-behind-the-results)

## Turning ambition into experiments

| Next step | The question it can answer |
| :--- | :--- |
| **Repeat** · more seeds, matched budgets | Which findings persist beyond a single run? |
| **Explain** · inspect belief and contact records | What causes arrival uncertainty, and can final sensing help? |
| **Transfer** · another robot body or terrain | How much movement knowledge can be reused with less retraining? |
| **Measure** · energy alongside stability | What tradeoffs connect useful movement and energy consumption? |

Adaptive walking, broad transfer and cleaner energy are future directions. The current evidence establishes a simulation case study and a starting point for those questions.

[A concrete next study: repeated trials, paired sensing comparisons, and final-belief diagnostics →](docs/RESEARCH_WALKTHROUGH.md#what-a-stronger-next-study-would-test)

---

## Acknowledgments

I thank [Professor Ye Zhao](https://people.research.gatech.edu/ye-zhao) for his feedback, engineering advice and help finding a research direction. Zhenyu Wu gave me sustained practical guidance, setup help, instructions and access to the team's resources; he also introduced me to Huaze Liu, who generously shared maps and research work as references. Rui Yan, Feiyang and Junnosuke helped with early onboarding and evaluation questions. Their support made my participation possible. [Full credits →](CONTRIBUTIONS.md)

---

<details>
<summary><strong>Research evidence, limitations and application writing</strong></summary>

- [Full navigation report](projects/tactile-navigation/outputs/Sky%20Navigation%20Experiment%20Report.md) · [Numeric audit](projects/tactile-navigation/outputs/sky_validation.json) · [Video audit](projects/tactile-navigation/outputs/sky_video_validation.json)
- [Source inventory and hashes](SOURCE_MANIFEST.json) · [Source revisions](PROVENANCE.json) · [Graphic sources](assets/SOURCES.md)
- [Portfolio review and sharing caveats](docs/PORTFOLIO_REVIEW.md) · [Repeatable content audit](presentation/audit_portfolio.py)
- [Historical application-writing guide (see the current contribution record above)](projects/tactile-navigation/outputs/College%20Admission%20Research%20Writing%20Guide.md)
- Scope: one seed per map, simulation only, team model 35500 substituted for the missing default 34000. The original room and new maps had different time caps; the system-interrupted attempt remains separate.
- The earlier controller screening did not demonstrate improvement. The independent G1 package does not establish guaranteed fall prevention or energy-efficiency gains.

</details>

<details>
<summary><strong>Public access and reproduction</strong></summary>

This repository preserves source snapshots, models, datasets, logs, reports and recordings. Large artifacts use Git LFS:

```bash
git clone https://github.com/ShihkaiLan1008/gatech-humanoid-control-navigation.git research-retrospective
cd research-retrospective
git lfs pull
```

This repository is public and can be read without a GitHub account. Dependencies and original local paths are documented by project. Large models and recordings use Git LFS. [Access and attribution →](CONFIDENTIALITY.md)

</details>

<p align="center"><sub>Ask a question. Build an experiment. Keep the evidence. Continue learning.</sub></p>
