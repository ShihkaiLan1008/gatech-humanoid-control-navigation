# With gratitude

I am deeply grateful to **[Professor Ye Zhao](https://people.research.gatech.edu/ye-zhao) at Georgia Tech** for his generous advice and for introducing me to his student, Zhenyu Wu. His guidance opened the door for me to learn from the team's research.

I owe particular thanks to **Zhenyu Wu**, who was my main source of practical guidance throughout this project. He gave me detailed instructions for approaching the work, provided extensive help with setting up the research environment, and shared the code and resources I needed to begin. His support was central to my ability to take part in the research and continue learning. He also introduced me to Huaze Liu. I am deeply grateful for the time and effort he devoted to helping me through this experience.

I also sincerely thank **Huaze Liu** for generously sharing his maps and research work as references for my learning and experiments. These materials gave me a foundation from which to explore new maps and questions.

Their time, guidance, and generosity have meant a great deal to me. This portfolio reflects both what I have learned and my respect for the people whose support helped me get here.

*With sincere thanks and respect, Shih-Kai Lan (Sky)*

---

![Sky Lan — Walking toward understanding. Robotics research portfolio with a recorded simulation trajectory.](assets/research-cover.svg)

<p align="center">
  <strong>Humanoid robotics &nbsp;·&nbsp; Mechanical engineering &nbsp;·&nbsp; Research in simulation</strong><br>
  <sub>Private portfolio · team-supported work · AI assistance disclosed</sub>
</p>

<p align="center">
  <a href="docs/ADMISSIONS_RESEARCH_OVERVIEW.md"><strong>START HERE</strong></a> &nbsp; / &nbsp;
  <a href="docs/HOW_I_DIRECTED_AI.md"><strong>MY ROLE</strong></a> &nbsp; / &nbsp;
  <a href="docs/VIDEOS.md">WATCH THE ROBOT</a> &nbsp; / &nbsp;
  <a href="docs/TECHNICAL_DEPTH.md">GO DEEPER</a> &nbsp; / &nbsp;
  <a href="docs/MY_RESEARCH_DIRECTION.md">WHY I CARE</a>
</p>

---

## Can a robot know when it has arrived?

*Shih-Kai Lan (Sky) · Personal student research portfolio · Work supported by Georgia Tech LIDAR mentors*

A robot has a map and a destination—but its estimate of its own position can drift as it walks. Touching a wall can help it correct that estimate. The challenge is to move successfully **and gather enough evidence to confirm arrival**.

Zhenyu's assignment was to test the team's navigation framework on new layouts. This extension introduced three maps, checked their geometry, ran the existing humanoid controller and preserved the results for analysis.

**My role:** I directed the AI-assisted workflow around that assignment: clarifying the goal, keeping the work focused on new-map testing, asking for an account of what had been done, and shaping how the research would be explained. [How I turned the task into directions for AI tools →](docs/HOW_I_DIRECTED_AI.md)

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

### How I directed the work

My contribution included interpreting the research assignment and giving AI tools a clear purpose. When I needed clarity about the testing, I asked where it had been performed and restated the goal: use the team's existing work as a reference, create new maps, and test the navigation system on them.

I also asked for investigation after the computer restarted unexpectedly, decided when to pause and resume the work, and requested an explanation of the experiments, their significance, and future possibilities. For this portfolio, I specified that readers should understand both the research and my role, with clear credit for my mentors and AI assistance.

These decisions show how I used AI deliberately: defining the task, questioning whether the work matched it, communicating what I needed, and refining the deliverable. [Concrete examples of my directions and the resulting work →](docs/HOW_I_DIRECTED_AI.md)

### How the contributions fit together

| Contributor | Role in this study |
| :--- | :--- |
| **Professor Zhao and the research team** | Advice, the new-map assignment, research resources, existing algorithms, maps, trained controller and simulator integration |
| **Me — Shih-Kai Lan (Sky)** | Task interpretation, direction of the AI-assisted workflow, scope clarification, questions about testing, and decisions about presentation and attribution |
| **ChatGPT and Codex** | Concept explanations and resource organization; substantial code/map implementation, simulation operation, analysis, visualization and drafting |

Together, this work produced three new map layouts and a documented simulation case study with recordings, quantitative results, and open questions.

**Credit:** Professor Ye Zhao provided advice and connected me with the team. Zhenyu Wu provided extensive practical guidance, instructions, setup help, and research resources; Huaze Liu generously shared maps and research work as references. The framework and models remain credited to their team authors. Codex assisted with code inspection, map/script implementation, experiment operation, analysis and writing. [Detailed attribution →](CONTRIBUTIONS.md)

**Learning with ChatGPT:** As a high-school student, I used ChatGPT to organize research resources, clarify unfamiliar concepts, and get quick advice on small questions while learning. I supplied the assignment, context, priorities, and follow-up directions that guided the assistance. This support complemented the guidance I received from Professor Zhao, Zhenyu Wu, and Huaze Liu.

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
- [Portfolio review and sharing caveats](docs/PORTFOLIO_REVIEW.md) · [Repeatable content audit](presentation/audit_portfolio.py)
- [College-application report and conclusion guide](projects/tactile-navigation/outputs/College%20Admission%20Research%20Writing%20Guide.md)
- Scope: one seed per map, simulation only, team model 35500 substituted for the missing default 34000. The original room and new maps had different time caps; the system-interrupted attempt remains separate.
- The earlier controller screening did not demonstrate improvement. The independent G1 package does not establish guaranteed fall prevention or energy-efficiency gains.

</details>

<details>
<summary><strong>Private archive and reproduction</strong></summary>

This repository preserves source snapshots, models, datasets, logs, reports and recordings. Large artifacts use Git LFS:

```bash
git clone https://github.com/ShihkaiLan1008/gatech-lidar-research-assistant.git research-retrospective
cd research-retrospective
git lfs pull
```

Dependencies and original local paths are documented by project. The archive is private and read-only; authorized updates require unarchiving it. Admissions readers need an approved excerpt or explicitly granted access. [Confidentiality and access →](CONFIDENTIALITY.md)

</details>

<p align="center"><sub>Ask a question. Build an experiment. Keep the evidence. Continue learning.</sub></p>
