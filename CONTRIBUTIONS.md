# Contributions and provenance

This portfolio records my participation as a student member of an ongoing Georgia Tech LIDAR research project. Professor Ye Zhao was my research mentor; Zhenyu Wu was my direct mentor and assigned the navigation evaluation. Under their guidance, I used and directed AI tools to help carry out my assigned work and collaborated with the group on questions, results and reporting. The existing framework, navigation model and reference materials remain credited to their authors.

## My contribution within the mentor-guided project

The research conversations document more than my use of AI. I asked how to interpret training quality and curves on July 23, requested a meaningful experimental role on August 24 and September 6, sent a weekly report to both mentors on September 14, and asked on September 25 that changing tasks stay within a coherent research area. These actions support initiative, communication, and developing experimental judgment. [Dated examples and their meaning](docs/HOW_I_DIRECTED_AI.md) connect that role to the preserved work.

My participation began with setting up and training a preliminary G1 simulation model with AI assistance; the archived configuration, log and checkpoint document that work. As described in my [personal account of the setback and rebuild](docs/SETBACK_AND_REBUILD.md), I later lost work after an internal computer failure and used earlier conversations and reconstructed context to direct a fresh installation on a better laptop. I asked for resource-conscious settings so that I could continue. The hardware cause remains unconfirmed; the surviving artifacts do not establish the full chronology or recover every lost file.

I interpreted Zhenyu's assignment as an evaluation task: build on the team's existing navigation work and test it on new maps. I supplied the repository and mentor context, kept the work focused on that mission, and used AI-assisted code and analysis tools to examine recorded outcomes and metrics. During the later navigation evaluation, when the computer restarted unexpectedly, I requested investigation and continuation, then chose when to pause and resume the research.

I also directed how the results should be communicated: explain what the experiments mean, make the work accessible to admissions readers while retaining technical depth, connect it with my motivation and future interests, and clearly credit the mentors, team resources, and AI tools involved.

My contribution includes preliminary model training, team discussions, a shared weekly report, carrying out my assigned navigation evaluation with AI assistance, examination of its quantitative results, and communication of the research. [Examples of my instructions and the work they guided](docs/HOW_I_DIRECTED_AI.md) make that role concrete. The detailed technical explanations describe the project and my developing research questions; they do not imply that I independently invented the team's methods or implemented every component.

## Preliminary independent exploration

The G1 rough-terrain package contains custom Isaac Lab task configuration, rewards, edge-return commands, a saved model and scripts from Sky's earlier independent exploration. It builds on Isaac Lab, Unitree assets and the RL framework; it is not an original general-purpose simulator or robot controller. Implementation and setup received AI assistance. The historical task package is preserved as found; no new training or playback validation was performed during archiving.

## Team foundation and extensions

Zhenyu Wu, Huaze Liu and the research team provided the tactile-controller and tactile-navigation foundations, learned checkpoints, localization, global planner, sensing strategy and existing maps. Their source snapshots remain team work. Sky's assigned extension concerns three new map layouts, experiment execution, geometry checks, recordings, quantitative analysis and reporting. Codex performed substantial map/script implementation, code inspection, experiment operation, analysis, visualization and writing in response to that direction. Sky participated in the mentor-guided evaluation, directed AI tools for his assigned tasks, and helped interpret and report the results, using AI as a substantial technical tool; the artifacts retain the team and AI attribution stated here.

The foot-reference screening is a separate earlier experiment, not Zhenyu's new-map assignment. The reference checkpoint measured about 9.62 cm mean reference foot error; the local tuned checkpoint measured about 33.02 cm in that screening. It did not demonstrate improvement. Single-seed screening and different training histories prevent a matched-training causal conclusion. The unfavorable result is retained in this portfolio.

The mentor conversation provided for this portfolio permits including team work in a report, and Sky separately reported permission to include the team's work in this private GitHub archive. Existing ownership, upstream license notices and other rights remain with their respective authors. Listing material here does not transfer the team's authorship to Sky or grant third parties reuse rights.

## AI-assisted learning and implementation

As a high-school student, I used **ChatGPT** to organize research resources, clarify unfamiliar concepts and help with setup while I learned. I supplied the research context, priorities, questions and follow-up directions, participated in discussions, and examined the outputs. The tool helped me work through gaps in my experience alongside guidance from Professor Zhao, Zhenyu Wu and Huaze Liu.

**Codex** also assisted with code inspection, map/script implementation, experiment operation, quantitative analysis, visualization, and writing. These uses are disclosed so readers can distinguish the team's research foundation, my learning and evaluation work, and the assistance provided by AI tools.

## Acknowledgments

I sincerely thank **[Professor Ye Zhao](https://people.research.gatech.edu/ye-zhao) at Georgia Tech** for his quick feedback, engineering advice, and active help finding a suitable research direction. He arranged learning resources, helped connect my humanoid interests to the team's work, and introduced me to **Zhenyu Wu**. His guidance has strongly influenced the engineer I want to become.

**Zhenyu Wu was my main source of practical guidance throughout this project.** He provided detailed instructions, extensive help with setting up the research environment, and the code and resources I needed to begin. His guidance and support were central to my participation and learning. I am especially grateful for the time and effort he devoted to helping me. He also introduced me to Huaze Liu.

I also thank **Huaze Liu** for generously sharing his maps and research work as references for my learning and experiments. His materials supported the new-map evaluation described in this portfolio.

Their guidance and generosity made this learning experience possible. I am grateful for their time and support, and I credit their research contributions throughout this portfolio.

I also appreciate the early onboarding and evaluation guidance from **Rui Yan, Feiyang, and Junnosuke** in the research group. The [dated mentor-alignment record](docs/MENTOR_ALIGNMENT.md) describes these distinct forms of support.

*Shih-Kai Lan (Sky)*
