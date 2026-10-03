[← Portfolio home](../README.md) · [Research overview](ADMISSIONS_RESEARCH_OVERVIEW.md) · [Contributions and acknowledgments](../CONTRIBUTIONS.md)

# How I directed AI-assisted research

*Shih-Kai Lan (Sky) · Student direction, team mentorship, and disclosed AI assistance*

My role was to translate a mentor's assignment into a clear purpose for AI tools, keep the requested work aligned with that purpose, and decide how to communicate the experience. As a high-school student, I used ChatGPT to organize resources, clarify concepts, and get quick advice. I also directed Codex to help carry out the research workflow, including substantial implementation, simulation operation, analysis, and writing.

## Decisions that show my role

The examples below paraphrase my instructions during the project. They describe the directions I gave; the linked reports and artifacts document the resulting work.

| Decision | Direction I gave | Why it mattered |
| :--- | :--- | :--- |
| **Identify the task** | Use Zhenyu's assignment and the team's repository as the starting point for testing new maps. | Connected the assistance to the mentor's research question and existing framework. |
| **Clarify scope** | Ask where testing had happened and restate that the mission was to make and test new maps based on the team's work. | Made the requested experiment explicit and distinguished it from other robotics work in the archive. |
| **Respond to interruption** | Ask for investigation after the unexpected computer restart, then direct when to pause and resume the research. | Addressed a practical obstacle and controlled when work continued. |
| **Specify the explanation** | Ask what was done, what the experiments mean, what they can do, and what questions come next; request both accessibility and technical depth. | Defined the purpose and audience of the research report. |
| **Set attribution expectations** | Request prominent thanks to Professor Zhao, Zhenyu Wu, and Huaze Liu, and disclosure of ChatGPT's help. | Made mentorship, shared resources, and AI assistance visible to readers. |

## From direction to research evidence

The new-map request led to three authored layouts evaluated with the team's navigation framework. The archive preserves the maps, executed simulations, recordings, and analysis:

- [Research overview](ADMISSIONS_RESEARCH_OVERVIEW.md): the question, map changes, results, and limitations.
- [Video gallery](VIDEOS.md): recordings of the four completed simulation trials.
- [Technical explanation](TECHNICAL_DEPTH.md): how planning, localization, walking, and arrival criteria fit together.
- [Full experiment report](../projects/tactile-navigation/outputs/Sky%20Navigation%20Experiment%20Report.md): settings, provenance, measurements, and the interrupted attempt.

The team's trained controller and navigation algorithms provided the research foundation. Codex developed the map variants and carried out substantial implementation, execution, validation, and analysis in response to the requested scope. My contribution was directing that workflow through the assignment, context, questions, priorities, and revisions described above.

## The research question behind the directions

The purpose of the assignment was to explore how an existing navigation framework behaves beyond its original maps. That requires testing executed movement as well as describing a proposed layout. The completed pilot then raised a more specific question: why did all four trials physically reach the goal region, while only one met the framework's formal completion criterion?

The reports explain the distinction between physical arrival and the system's belief about arrival. One seed per map provides an initial case study; repeated trials are needed to support broader claims about robustness. These findings came from the AI-assisted analysis and are presented with their limits.

## A concise account of my contribution

> With advice from Professor Ye Zhao, extensive practical guidance from Zhenyu Wu, and reference materials generously shared by Huaze Liu, I directed an AI-assisted evaluation of humanoid navigation on new maps. I clarified the assignment, provided the team's code and mentor context, and questioned whether the work addressed the requested testing. I used ChatGPT to organize resources and understand unfamiliar concepts, while Codex performed substantial implementation and analysis. I also shaped the research report to explain the experiments, their significance, and future questions, with explicit credit for the team and AI assistance. My contribution was connecting the research task to clear directions and refining how the resulting work was communicated.

[Full contribution and AI-use disclosure →](../CONTRIBUTIONS.md)
