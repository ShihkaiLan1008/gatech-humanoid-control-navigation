![Research portfolio section](../assets/section-study.svg)

[← Portfolio home](../README.md) · [Research overview](ADMISSIONS_RESEARCH_OVERVIEW.md) · [Technical depth](TECHNICAL_DEPTH.md) · [Video gallery](VIDEOS.md)

# Can a humanoid know when it has arrived?

## Assignment, research question and my role

As a student member of an ongoing Georgia Tech LIDAR team project, I took on a navigation evaluation assigned by Zhenyu Wu. The team had already developed the humanoid walking controller, map-based planner, localization and touch-sensing framework. Zhenyu asked me to run the existing navigation code with visible evidence, then test it on new maps to explore generality and robustness. Huaze Liu's shared maps and research work provided references. [Dated task record →](MENTOR_ALIGNMENT.md)

I supplied the team's repository and assignment to AI tools, clarified that new-map testing was the task, and directed the evaluation and its explanation. The pilot preserved a run on an existing map and tested three newly authored layouts with geometry checks, simulation recordings and numerical results. When a later run was interrupted by a computer restart, I requested investigation and continuation, then chose when to pause and resume. I also asked for the report to explain the purpose, results and limits in accessible language. [My role and dated evidence →](HOW_I_DIRECTED_AI.md)

The resulting question is whether physical arrival and confidence in arrival agree. In four completed single-seed simulations, the robot entered the goal region each time, but the navigation system formally confirmed arrival only once. The team's algorithms and trained model remain team work; AI tools provided substantial implementation, operation, analysis and writing support. [Detailed attribution →](../CONTRIBUTIONS.md)

Earlier G1 setup and training, an independent rough-terrain package, and foot-tracking work were separate stages of my participation. They did not constitute this new-map assignment. The three new layouts were deliberately authored cases, not a randomized or repeated robustness benchmark. [Research timeline →](MENTOR_ALIGNMENT.md) · [Setback and rebuild →](SETBACK_AND_REBUILD.md)

## What was changed and why

The offset-opening map forces an indirect route through an opening at one end of a divider. The staggered-baffle map adds repeated turns around alternating dividers. The island map presents two possible routes around a central solid obstacle and exercises the map-coordinate transformation with a different starting heading.

These layouts introduce different navigation demands while leaving the underlying controller and navigation settings shared. Twelve focused checks passed for loading, endpoints, clearance and obstacle representation. All three planned routes passed the framework's feasibility checks; actual physical-simulation execution was assessed separately.

## What the experiments showed

| Layout | Physical goal reached | Fell | Formal completion | Final distance to goal | Arrival belief |
|---|---|---|---|---:|---:|
| Original room2 | Yes | No | Arrival uncertain | 7.31 cm | 93.35% |
| Central island | Yes | No | Arrival uncertain | 6.41 cm | 92.41% |
| Offset opening | Yes | No | Arrival uncertain | 4.07 cm | 91.89% |
| Staggered dividers | Yes | No | Route complete | 4.16 cm | 96.04% |

The system requires at least 95% of its localization belief to satisfy both final position and heading conditions before declaring arrival. The simulator's true final position was inside the goal region in all four trials; that ground truth was used to evaluate execution. Three internal beliefs remained insufficiently confident. The distinction exposes a meaningful research question about how sensing, uncertainty and completion interact.

A more complicated-looking map did not necessarily produce a worse completion outcome. One trial per layout cannot explain why the staggered map passed or establish which geometry is easier. Contact count alone also does not establish that more touches cause higher confidence.

![Executed simulation paths](../projects/tactile-navigation/outputs/sky_navigation_trajectories.png)

*True and estimated trajectories from the four completed seed-0 trials. Axes are in metres; obstacles and goal disks are shown. These are executed simulation records.*

## What the work can do now

It can serve as a new-map evaluation suite and a reproducible diagnostic case study. The saved maps, recordings, contact events and metric definitions let a researcher examine how the same framework behaves on different layouts. Replay videos show the executed behavior rather than a hand-animated successful route.

Potential longer-term value is better navigation when position estimates drift or sensing is uncertain. Application to dark spaces, visually difficult environments or assistive robots would require further evidence; it is not demonstrated by these trials.

## The difficult questions underneath the demonstration

[Inside the research](RESEARCH_WALKTHROUGH.md) explains the system as a whole and opens one recorded trial, including the timeline of error, contacts, and misses. It also gives a concrete plan for a controlled next comparison.

Three examples show why this work goes beyond drawing a room and playing a robot animation:

- **Geometry must agree across systems.** The island map's authored start heading is rotated into the navigation frame. Its goal changes from an authored displacement of `(3.6, 0)` to `(0, 3.6)`. The planner and simulated obstacles must still represent the same room.
- **A plausible training idea can fail to establish improvement.** Earlier work added a narrower foot-position reward, but later screening measured 33.02 cm error for the tuned checkpoint versus 9.62 cm for the reference. Different training histories prevent a causal conclusion about the reward; a matched experiment is needed.
- **Closeness is not certainty.** The offset-opening robot finished slightly closer to its goal than the staggered-baffle robot, yet only the latter passed the joint position-and-heading belief threshold. The next question is what evidence the system lacked, rather than whether the final video looked successful.

[Technical depth](TECHNICAL_DEPTH.md) explains the mechanisms, measurements, source evidence, and controlled follow-up experiments. These explanations were developed with AI assistance; the team supplied the underlying methods.

## What it has taught at the project level

Useful robotics results include unsuccessful criteria and uncertainty, not just an appealing video. Here, a robot can finish close to its target while the system still declines to certify arrival. A careful evaluation separates geometry feasibility, physical movement, localization accuracy and confidence. It also preserves an unfavorable earlier foot-tracking result rather than reporting improvement without support.

Sky describes a 2024 forum experience as the origin of their interest in mechanical engineering, and the sight of robot movement emerging from changed inputs as a source of excitement. Their longer-term interests are more adaptive walking, transferable robot knowledge and cleaner energy. [Personal motivation and research path](MY_RESEARCH_DIRECTION.md) connects those interests to concrete next experiments while keeping untested ambitions distinct from results.

## Limits and next questions

Only one seed was run per layout, all trials were simulation, and a team model-35500 checkpoint replaced the absent README default model 34000. The original map had a 240 s cap while new maps had 900 s caps. A Windows-interrupted attempt was recorded separately and restarted; it is not counted as an extra independent completed trial. These are pilot observations, not a population success-rate estimate or an exact replication of the team's default experiments.

The next study should match budgets and repeat seeds before making robustness claims. Then vary opening widths, compare sensing policies or final confirmation probes, and inspect whether confidence improvements also remain calibrated against simulator truth. Real-hardware transfer is a future team research step.

For mathematical definitions and the particle-filter/planner/controller relationship, see [Technical depth](TECHNICAL_DEPTH.md). For records, exact settings, checkpoint identity and hashes, see the [full experiment report](../projects/tactile-navigation/outputs/Sky%20Navigation%20Experiment%20Report.md).

## Acknowledgments

I thank [Professor Ye Zhao](https://people.research.gatech.edu/ye-zhao) for his feedback and help finding a suitable research direction. Zhenyu Wu provided sustained practical mentorship, instructions, setup help and research resources. Huaze Liu generously shared maps and research work as references. I also appreciate the early guidance of Rui Yan, Feiyang and Junnosuke. [Full credits and contribution boundaries →](../CONTRIBUTIONS.md)
