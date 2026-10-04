# Contributions and provenance

## Acknowledgments

I sincerely thank **[Professor Ye Zhao](https://people.research.gatech.edu/ye-zhao) at Georgia Tech** for his advice and for introducing me to his student **Zhenyu Wu**, opening the door for me to learn from the team.

**Zhenyu Wu was my main source of practical guidance throughout this project.** He provided detailed instructions, extensive help with setting up the research environment, and the code and resources I needed to begin. His guidance and support were central to my participation and learning. I am especially grateful for the time and effort he devoted to helping me. He also introduced me to Huaze Liu.

I also thank **Huaze Liu** for generously sharing his maps and research work as references for my learning and experiments. His materials supported the new-map evaluation described in this portfolio.

Their guidance and generosity made this learning experience possible. I am grateful for their time and support, and I credit their research contributions throughout this portfolio.

*Shih-Kai Lan (Sky)*

## My role in directing the research workflow

I interpreted Zhenyu's assignment as an evaluation task: build on the team's existing navigation work and test it on new maps. I supplied the repository and mentor context to AI tools, asked where testing had occurred, and explicitly brought the work back to the new-map mission. When the computer restarted unexpectedly, I requested investigation and continuation, then chose when to pause and resume the research.

I also directed how the results should be communicated: explain what the experiments mean, make the work accessible to admissions readers while retaining technical depth, connect it with my motivation and future interests, and clearly credit the mentors, team resources, and AI tools involved.

My contribution here is task interpretation, scope clarification, direction of AI assistance, and research communication. [Examples of my instructions and the work they guided](docs/HOW_I_DIRECTED_AI.md) make that role concrete.

## AI-assisted learning and implementation

As a high-school student, I used **ChatGPT** to help organize research resources, clarify unfamiliar concepts, and get quick advice on small questions as I learned. I supplied the research context, priorities, questions, and follow-up directions. This assistance supported my learning alongside the guidance provided by Professor Zhao, Zhenyu Wu, and Huaze Liu.

**Codex** also assisted with code inspection, map/script implementation, experiment operation, quantitative analysis, visualization, and writing. These uses are disclosed so readers can distinguish the team's research foundation, my learning and evaluation work, and the assistance provided by AI tools.

## Preliminary independent exploration

The G1 rough-terrain package contains custom Isaac Lab task configuration, rewards, edge-return commands, a saved model and scripts from Sky's earlier independent exploration. It builds on Isaac Lab, Unitree assets and the RL framework; it is not an original general-purpose simulator or robot controller. Implementation and setup received AI assistance. The historical task package is preserved as found; no new training or playback validation was performed during archiving.

## Team foundation and extensions

Zhenyu Wu, Huaze Liu and the research team provided the tactile-controller and tactile-navigation foundations, learned checkpoints, localization, global planner, sensing strategy and existing maps. Their source snapshots remain team work. The extension directed by Sky concerns three new map layouts, experiment execution, geometry checks, recordings, quantitative analysis and reporting. Codex performed substantial map/script implementation, code inspection, experiment operation, analysis, visualization and writing in response to that direction. The contribution described above concerns directing and communicating this AI-assisted evaluation; the technical artifacts retain the team and AI attribution stated here.

The foot-reference screening is a separate earlier experiment, not Zhenyu's new-map assignment. The reference checkpoint measured about 9.62 cm mean reference foot error; the local tuned checkpoint measured about 33.02 cm in that screening. It did not demonstrate improvement. Single-seed screening and different training histories prevent a matched-training causal conclusion. The unfavorable result is retained in this portfolio.

The mentor conversation provided for this portfolio permits including team work in a report, and Sky separately reported permission to include the team's work in this private GitHub archive. Existing ownership, upstream license notices and other rights remain with their respective authors. Listing material here does not transfer the team's authorship to Sky or grant third parties reuse rights.
