![Research study](../assets/section-study.svg)

[Portfolio home](../README.md) · [My contribution](../CONTRIBUTIONS.md) · [Recorded runs](VIDEOS.md) · [Technical methods](TECHNICAL_DEPTH.md)

# Can a humanoid know when it has arrived?

**A student research case study · Summer 2026–present · Ongoing work**

The team's system combines learned walking, tactile sensing, localization and planning. My assigned evaluation asked how it behaved on new layouts. The recorded pilot showed that physical arrival and the system's confidence in arrival could disagree.

## Why study this?

I want to contribute to robots that can help with everyday physical tasks or work in hazardous spaces. Dependable movement requires a robot to recognize when its estimate of its surroundings is uncertain. Testing that problem under Professor Ye Zhao's mentorship and Zhenyu Wu's guidance gives me a concrete starting point in mechanical engineering and robotics.

## Assignment, research question and my role

Zhenyu assigned me to run an existing map with visible evidence, then create and test new maps. The team supplied the controller, navigation framework and pretrained model; Huaze Liu shared maps and research materials as references.

**I directed the workflow and reporting for my assigned evaluation under mentor guidance.** I kept the task aligned with the instructions, supplied context and constraints, examined recordings and measurements, requested explanations and corrections, discussed evaluation with the group and shared a weekly report.

Codex provided substantial map/script implementation, geometry checks, simulation operation and calculations. ChatGPT helped integrate source material and draft explanations from my prompts. [Contribution evidence](../CONTRIBUTIONS.md) · [Detailed division of work](AI_AND_SOURCE_DISCLOSURE.md)

## Research skills behind the evaluation

Earlier, I rebuilt a training environment, implemented a foot-tracking reward modification and completed a 500-iteration diagnostic run with guidance and AI assistance. I identified that the local setup supplied one eighth of the samples of the larger reference configuration at the same iteration count and proposed matched training conditions.

A later screening did not demonstrate improved tracking: the tuned checkpoint's mean reference-foot error was 33.02 cm versus 9.62 cm for the reference. Different training histories prevented a causal conclusion about the reward change. That experience informed how I approached the navigation study: preserve the result and explain what comparison is still needed.

[Controller experiment and evidence](../CONTRIBUTIONS.md#controller-implementation-and-experimental-judgment)

## What was changed and why

| New layout | Purpose |
| :--- | :--- |
| Offset opening | Require an indirect route through an opening away from the direct path |
| Staggered baffles | Introduce repeated turns and examine behavior as movement and uncertainty accumulate |
| Central island | Require routing around an obstacle and exercise the existing map-coordinate transformation |

All three maps passed planner feasibility checks. Twelve focused geometry/import checks covered loading, endpoints, clearance and obstacle representation. Those checks are separate from evaluating actual movement with the learned controller. The layouts were deliberately authored cases, not a randomized-map benchmark.

## What the experiments showed

Four completed seed-0 trials entered the physical goal region without a reported fall.

| Layout | Goal distance | Arrival belief | Framework decision |
| :--- | ---: | ---: | :--- |
| Original room | 7.31 cm | 93.35% | Arrival uncertain |
| Central island | 6.41 cm | 92.41% | Arrival uncertain |
| Offset opening | 4.07 cm | 91.89% | Arrival uncertain |
| Staggered baffles | 4.16 cm | 96.04% | Complete |

The framework requires at least 95% of its particle-filter weight to jointly satisfy the final position region and heading bound. Three trials ended as `arrival_uncertain`; the staggered-baffle trial reported completion.

Simulator truth was used to evaluate movement rather than supplied to high-level navigation. Arrival belief is an internal filter quantity, not a validated probability of real-world success.

![True and estimated trajectories from the four completed simulation trials.](../projects/tactile-navigation/outputs/sky_navigation_trajectories.png)

[Recorded runs and frame sheets](VIDEOS.md) · [Canonical result records](../projects/tactile-navigation/outputs/sky_results.json)

## Reading one result and planning the next test

The offset-opening trial ended **4.07 cm** from its target with **91.89%** arrival belief. The staggered-baffle trial ended slightly farther away, at **4.16 cm**, but passed with **96.04%** belief. Final distance alone does not explain the different decisions.

This gives the team a specific case to inspect: what uncertainty remained when the robot physically reached the goal? The present records do not isolate position spread, heading spread, missed contacts or model assumptions as the cause. More contacts alone do not establish better localization.

A proposed next study is to repeat the offset-opening case using predefined seeds and a common time budget, recording position, heading and joint arrival confidence separately. A paired test could then compare an additional final sensing action with the unchanged policy. These follow-ups remain unrun in this preserved pilot.

[Open the full experiment and follow-up outline](RESEARCH_WALKTHROUGH.md)

## Limits and next questions

- **Replication:** One seed per map cannot establish a general success rate or broad robustness.
- **Time budgets:** The original room had a 240-second cap; new maps had 900-second caps.
- **Model:** All local navigation trials used available team checkpoint 35500 because documented default 34000 was absent. This was not an exact default-checkpoint replication.
- **Recovery:** One interrupted attempt was preserved separately as `system_restart` and excluded from the four completed trials.
- **Transfer:** All results are in simulation. Hardware reliability, energy efficiency and cross-robot transfer remain unestablished.

[Exact settings and recovery record](../projects/tactile-navigation/outputs/Sky%20Navigation%20Experiment%20Report.md) · [Metric definitions](TECHNICAL_DEPTH.md)

## What it has taught me

I learned to distinguish a runnable implementation from a supported research conclusion, and physical success from the evidence needed to certify it. This work has further affirmed my love for mechanical engineering. Research is ongoing, and I want to continue developing the technical knowledge and experimental judgment to contribute in this field.

[Personal motivation and future direction](MY_RESEARCH_DIRECTION.md)

## Sources and credits

Professor Zhao guided the research opportunity; Zhenyu provided direct mentorship and the assignment; Huaze Liu and the team supplied the research foundation. Team data, content and images supplied through Zhenyu are included with permission confirmed by me. AI implementation and drafting remain disclosed. [Roles, sources and permission](AI_AND_SOURCE_DISCLOSURE.md)
