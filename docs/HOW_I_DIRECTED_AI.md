[Portfolio home](../README.md) · [Contribution evidence](../CONTRIBUTIONS.md) · [AI and source credits](AI_AND_SOURCE_DISCLOSURE.md)

# How I directed my assigned evaluation

**Shih-Kai Lan (Sky) · Mentor-guided research, summer 2026–present**

My responsibility was to carry the assigned evaluation forward under Professor Ye Zhao's mentorship and Zhenyu Wu's direct guidance. The practical distinction is between my task direction and research participation, the team's methods, and the technical work performed with AI.

## The workflow

1. **Clarify the task.** I sought a manageable contribution and asked that changing tasks remain connected. Zhenyu's navigation instructions were to run an existing map with visible evidence, then test new maps.
2. **Supply context and constraints.** I provided the assignment, repository context and priorities to the AI-assisted workflow. I corrected scope when needed and asked for results that addressed the mentor's task.
3. **Follow execution and recovery.** Codex implemented substantial map and script changes, checks and simulation operations. During the later computer restart, I requested investigation and continuation and chose when to pause and resume.
4. **Examine the outputs.** I worked through recordings and measurements, asked what the results supported, and helped organize explanations of physical arrival, arrival confidence and the comparison limits.
5. **Communicate.** I discussed training and evaluation with the group, shared a weekly report, and supplied prompts and corrections for the portfolio's presentation.

[Assignment and discussion context](MENTOR_ALIGNMENT.md) · [Full experiment report](../projects/tactile-navigation/outputs/Sky%20Navigation%20Experiment%20Report.md)

## Technical work and experimental judgment

My earlier controller work included rebuilding the environment, a foot-tracking reward modification and a 500-iteration diagnostic run, with mentor guidance and AI assistance.

The important comparison issue was training exposure: 1,024 environments supplied one eighth of the transitions of an 8,192-environment setup at the same iteration count. I identified that mismatch and proposed controlled training conditions before claiming improvement.

The later screening did not demonstrate improved tracking. Keeping that outcome in the record makes the open question specific: what happens when initialization, sample budget, commands and evaluation are matched?

[Reward and training evidence](../CONTRIBUTIONS.md#controller-implementation-and-experimental-judgment)

## What directing AI means here

I supplied prompts, source context, priorities and revision requests; I examined outputs and requested explanations and corrections. ChatGPT integrated material and generated much of the explanatory text. Codex performed substantial implementation, experiment operation, analysis, visualization and documentation.

This is a record of directing tools for my assigned research participation. The team retains credit for its methods and models, and AI drafting is disclosed as such. It is not a reconstructed verbatim prompt log.

## What remains to be tested

The new-map pilot raised the gap between physical arrival and confidence in arrival. Repeated seeds, matched time budgets and a controlled final-sensing comparison could investigate that gap. Those are proposed next experiments, not completed results.

[One trial and the proposed next study](RESEARCH_WALKTHROUGH.md#what-a-stronger-next-study-would-test)

## Portfolio and application writing

The portfolio is substantially AI-assisted in both technical execution and presentation. Its explanations are not ready-to-submit original application responses. Being able to explain a generated passage does not by itself make its wording my original writing.

[AI-use disclosure and application-writing distinction](AI_AND_SOURCE_DISCLOSURE.md#portfolio-production-and-application-writing)
