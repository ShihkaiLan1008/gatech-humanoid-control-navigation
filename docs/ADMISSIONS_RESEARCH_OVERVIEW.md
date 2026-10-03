![Research portfolio section](../assets/section-study.svg)

[← Portfolio home](../README.md) · [Research overview](ADMISSIONS_RESEARCH_OVERVIEW.md) · [Technical depth](TECHNICAL_DEPTH.md) · [Video gallery](VIDEOS.md)

# Can a humanoid know when it has arrived?

## Acknowledgments

I sincerely thank **[Professor Ye Zhao](https://people.research.gatech.edu/ye-zhao) at Georgia Tech** for his advice and for introducing me to **Zhenyu Wu**.

**Zhenyu was my main source of practical guidance throughout the project.** He provided detailed instructions, extensive setup help, and the code and research resources I needed. His support was central to my participation and learning, and I am deeply grateful for the time and effort he devoted to helping me. He also introduced me to **Huaze Liu**, whom I sincerely thank for generously sharing his maps and research work as references for my experiments. [Full acknowledgments and contribution details →](../CONTRIBUTIONS.md)

## Question and context

This team-supported simulation study explores a robot-navigation problem that is easy to overlook: reaching a destination and knowing that it has reached it are not the same. The team's system combines a trained humanoid walking controller, a map-based route planner, an estimate of the robot's position and touch observations. Zhenyu's assignment was to run the existing navigation code with visible evidence, then create new maps to explore generality and robustness.

The study extends that testing to three newly authored layouts. It is an evaluation contribution to an existing research framework, with AI assistance in implementation, execution and reporting. The team's algorithms and trained model remain credited team work.

## My role: turning the assignment into directions

I understood Zhenyu's assignment as a question about an existing system: how would it behave on new maps? I provided the team's repository and mentor context to AI tools, asked where testing had been done, and restated that new-map evaluation was the task. This kept the requested work tied to the research assignment.

When the computer restarted unexpectedly, I asked for investigation and continuation, then decided when to pause and resume. I also directed the report to explain the study's purpose, results, significance, and future questions in language an admissions reader could understand while retaining technical depth. I requested explicit credit for the team's support and AI assistance.

This role required communicating the goal, clarifying scope, asking follow-up questions, and refining the output. The resulting study used substantial AI assistance for implementation and analysis. [Concrete examples of my directions →](HOW_I_DIRECTED_AI.md)

**AI-assisted learning:** As a high-school student, I used ChatGPT to organize research resources, clarify unfamiliar concepts, and get quick advice on small questions while learning. I supplied the context, priorities, and follow-up directions. Codex also assisted with implementation, experiment operation, analysis, visualization, and writing. [Full AI-use and contribution disclosure →](../CONTRIBUTIONS.md#ai-assisted-learning-and-implementation)

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

## What it has taught at the project level

Useful robotics results include unsuccessful criteria and uncertainty, not just an appealing video. Here, a robot can finish close to its target while the system still declines to certify arrival. A careful evaluation separates geometry feasibility, physical movement, localization accuracy and confidence. It also preserves an unfavorable earlier foot-tracking result rather than reporting improvement without support.

Sky describes a 2024 forum experience as the origin of their interest in mechanical engineering, and the sight of robot movement emerging from changed inputs as a source of excitement. Their longer-term interests are more adaptive walking, transferable robot knowledge and cleaner energy. [Personal motivation and research path](MY_RESEARCH_DIRECTION.md) connects those interests to concrete next experiments while keeping untested ambitions distinct from results.

## Limits and next questions

Only one seed was run per layout, all trials were simulation, and a team model-35500 checkpoint replaced the absent README default model 34000. The original map had a 240 s cap while new maps had 900 s caps. A Windows-interrupted attempt was recorded separately and restarted; it is not counted as an extra independent completed trial. These are pilot observations, not a population success-rate estimate or an exact replication of the team's default experiments.

The next study should match budgets and repeat seeds before making robustness claims. Then vary opening widths, compare sensing policies or final confirmation probes, and inspect whether confidence improvements also remain calibrated against simulator truth. Real-hardware transfer is a future team research step.

For mathematical definitions and the particle-filter/planner/controller relationship, see [Technical depth](TECHNICAL_DEPTH.md). For records, exact settings, checkpoint identity and hashes, see the [full experiment report](../projects/tactile-navigation/outputs/Sky%20Navigation%20Experiment%20Report.md).
