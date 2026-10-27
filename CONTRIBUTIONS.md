# Contributions and provenance

This portfolio records my participation as a student member of an ongoing Georgia Tech LIDAR research project, beginning in summer 2026. I continue conducting experiments and plan to pursue further projects; this repository preserves the implementation and results of the project presented here. Professor Ye Zhao is my research mentor; Zhenyu Wu is my direct mentor and assigned the navigation evaluation. Under their guidance, I used and directed AI tools to help carry out my assigned work and collaborated with the group on questions, results and reporting. The existing framework, navigation model and reference materials remain credited to their authors.

## My contribution within the mentor-guided project

**I directed the workflow and reporting for my assigned evaluation under my mentors' guidance.** My responsibilities connected three parts of the work:

- **Keeping the task coherent:** I asked for a meaningful experimental contribution, clarified the new-map assignment and supplied its context, priorities and constraints to the AI-assisted workflow.
- **Examining what the results supported:** I discussed training quality and learning curves with the group, identified the unequal training sample budget, examined navigation recordings and measurements, and kept unproven improvement claims open.
- **Following through and communicating:** I shared a weekly report, requested explanations and corrections, and shaped a documented account of the results and limitations.

The mentor defined the assigned study and the team supplied its methods and models. The code changes, training measurements and reports linked below connect my participation to inspectable artifacts; substantial AI implementation, analysis and drafting are credited in the sections that follow.

My participation began with setting up and training a preliminary G1 simulation model with AI assistance; the archived configuration, log and checkpoint document that work. As described in my [personal account of the setback and rebuild](docs/SETBACK_AND_REBUILD.md), I later lost work after an internal computer failure and used earlier conversations and reconstructed context to direct a fresh installation on a better laptop. I asked for resource-conscious settings so that I could continue. The hardware cause remains unconfirmed; the surviving artifacts do not establish the full chronology or recover every lost file.

I interpreted Zhenyu's assignment as an evaluation task: build on the team's existing navigation work and test it on new maps. I supplied the repository and mentor context, kept the work focused on that mission, and used AI-assisted code and analysis tools to examine recorded outcomes and metrics. During the later navigation evaluation, when the computer restarted unexpectedly, I requested investigation and continuation, then chose when to pause and resume the research.

I also directed how the results should be communicated: explain what the experiments mean, make the work accessible to admissions readers while retaining technical depth, connect it with my motivation and future interests, and clearly credit the mentors, team resources, and AI tools involved.

My contribution includes preliminary model training, team discussions, a shared weekly report, carrying out my assigned navigation evaluation with AI assistance, examination of its quantitative results, and communication of the research. The detailed technical explanations describe the project and my developing research questions; they do not imply that I independently invented the team's methods or implemented every component.

## Controller implementation and experimental judgment

Under Professor Ye Zhao's supervision and Zhenyu Wu's direct guidance, I reconstructed the training environment, resolved CUDA, compiler and library compatibility issues, checked command and data pipelines, and implemented an additional foot-position reward with AI assistance. I completed a 500-iteration diagnostic run within an 8 GB GPU constraint and saved the trained model. The [reward diff](projects/tactile-controller/local-reward-change.patch) and [training record](projects/tactile-controller/report-evidence/experiment_evidence.json) preserve the implemented change and its measurements.

I identified that the local run's 1,024 environments supplied one eighth of the samples of an 8,192-environment setup at the same iteration count. I therefore kept the accuracy claim open and proposed matched training conditions. My contribution was both technical and analytical: implement a testable change, inspect the evidence, and explain why it did not yet establish an improvement. AI helped with implementation and troubleshooting; I checked suggestions against source code, system behavior and measured results. [Experiment settings, results and proposed controlled comparison →](projects/tactile-controller/outputs/Mission%20Review%20and%20Work%20Report.md)

## Preliminary independent exploration

The G1 rough-terrain package contains custom Isaac Lab task configuration, rewards, edge-return commands, a saved model and scripts from Sky's earlier independent exploration. It builds on Isaac Lab, Unitree assets and the RL framework; it is not an original general-purpose simulator or robot controller. Implementation and setup received AI assistance. The historical task package is preserved as found; no new training or playback validation was performed during archiving.

## Team foundation and extensions

Zhenyu Wu, Huaze Liu and the research team provided the tactile-controller and tactile-navigation foundations, learned checkpoints, localization, global planner, sensing strategy and existing maps. Their source snapshots remain team work. Sky's assigned extension concerns three new map layouts, experiment execution, geometry checks, recordings, quantitative analysis and reporting. Codex performed substantial map/script implementation, code inspection, experiment operation, analysis, visualization and writing in response to that direction. Sky participated in the mentor-guided evaluation, directed AI tools for his assigned tasks, and helped interpret and report the results, using AI as a substantial technical tool; the artifacts retain the team and AI attribution stated here.

The foot-reference screening is a separate earlier experiment, not Zhenyu's new-map assignment. The reference checkpoint measured about 9.62 cm mean reference foot error; the local tuned checkpoint measured about 33.02 cm in that screening. It did not demonstrate improvement. Single-seed screening and different training histories prevent a matched-training causal conclusion. The unfavorable result is retained in this portfolio.

Sky confirms that Zhenyu Wu authorized inclusion of the team data, reference content and images he supplied in the original public GitHub research portfolio. This private copy preserves those materials and their credits. Zhenyu is credited as the supplying mentor; the underlying authors, including Huaze Liu and other team or upstream contributors, retain their respective authorship and rights. This permission statement records Sky's confirmation, rather than a separately published license or a claim of independently verified permission for every upstream item. Existing licenses and notices remain applicable; this portfolio does not grant a blanket license over team materials. See [material permissions and provenance](docs/AI_AND_SOURCE_DISCLOSURE.md#material-permissions-and-provenance).

## AI-assisted learning and implementation

I supplied **ChatGPT** with the research context, source materials, prompts, priorities and follow-up directions. In response, ChatGPT helped integrate and organize the material and generated much of the portfolio's explanatory and presentation text. I examined outputs, asked follow-up questions and requested revisions to reflect my intended scope and account of the research. ChatGPT also helped explain unfamiliar concepts and troubleshoot setup while I learned alongside guidance from Professor Zhao, Zhenyu Wu and Huaze Liu. First-person portfolio passages can therefore be AI-drafted accounts of my documented participation; they are not presented as prose that I independently wrote without assistance.

**Codex** also assisted with code inspection, map/script implementation, experiment operation, quantitative analysis, visualization, and writing. These uses are disclosed so readers can distinguish the team's research foundation, my learning and evaluation work, and the assistance provided by AI tools. Providing prompts and reviewing outputs is part of my documented workflow; it is not evidence of sole code authorship, independent invention of the team's methods, or mastery of every component. The [AI and source disclosure](docs/AI_AND_SOURCE_DISCLOSURE.md) links contribution claims to existing records and explains why this portfolio is not a substitute for independently authored application writing.

## Acknowledgments

I sincerely thank **[Professor Ye Zhao](https://people.research.gatech.edu/ye-zhao) at Georgia Tech** for his quick feedback, engineering advice, and active help finding a suitable research direction. He arranged learning resources, helped connect my humanoid interests to the team's work, and introduced me to **Zhenyu Wu**. His guidance has strongly influenced the engineer I want to become.

**Zhenyu Wu was my main source of practical guidance throughout this project.** He provided detailed instructions, extensive help with setting up the research environment, and the code and resources I needed to begin. His guidance and support were central to my participation and learning. I am especially grateful for the time and effort he devoted to helping me. He also introduced me to Huaze Liu.

I also thank **Huaze Liu** for generously sharing his maps and research work as references for my learning and experiments. His materials supported the new-map evaluation described in this portfolio.

Their guidance and generosity made this learning experience possible. I am grateful for their time and support, and I credit their research contributions throughout this portfolio.

I also appreciate the early onboarding and evaluation guidance from **Rui Yan, Feiyang, and Junnosuke** in the research group. The [mentor guidance record](docs/MENTOR_ALIGNMENT.md) describes these distinct forms of support.

*Shih-Kai Lan (Sky)*
