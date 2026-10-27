# Contributions and evidence

[Portfolio home](README.md) · [Research overview](docs/ADMISSIONS_RESEARCH_OVERVIEW.md) · [AI and source credits](docs/AI_AND_SOURCE_DISCLOSURE.md)

**Shih-Kai Lan (Sky) · Student research participant · Summer 2026–present**

I participate under Professor Ye Zhao's mentorship and Zhenyu Wu's direct guidance. **I directed the workflow and reporting for my assigned evaluation.** Zhenyu defined the navigation assignment; I kept the work focused on it, supplied context and constraints, examined results, asked follow-up questions and shaped the report.

The record supports that responsibility within a mentor-guided project. The team's underlying research methods and the AI tools' substantial implementation, execution and writing contributions remain credited below.

## What I contributed

| Contribution | Evidence to inspect | What it establishes |
| :--- | :--- | :--- |
| Rebuilding a runnable environment and continuing after setbacks | [Recovery account](docs/SETBACK_AND_REBUILD.md) | A documented reconstruction effort with mentor guidance and AI help; the early hardware-failure cause remains unknown |
| A foot-tracking reward change and a completed diagnostic training run | [Reward diff](projects/tactile-controller/local-reward-change.patch), [training record](projects/tactile-controller/report-evidence/experiment_evidence.json), [saved model](projects/tactile-controller/logs/rsl_rl/g1_decoupled_joint_foot_mimic_arm/2026-09-13_22-38-08_foot_precise_s005_500/model_499.pt) | A specific implemented change and a preserved trained artifact |
| Identifying unequal training exposure and proposing a fairer comparison | [Settings and comparison limits](projects/tactile-controller/outputs/Mission%20Review%20and%20Work%20Report.md) | Experimental judgment: the improvement question remained open |
| Carrying the navigation assignment into a documented new-map evaluation | [Task context](docs/MENTOR_ALIGNMENT.md), [full report](projects/tactile-navigation/outputs/Sky%20Navigation%20Experiment%20Report.md), [recordings](docs/VIDEOS.md) | Participation in the assigned study, with substantial Codex map implementation, operation and analysis |
| Discussing scope and evaluation, reviewing outputs and reporting progress | [Mentor discussion summary](docs/MENTOR_ALIGNMENT.md), [workflow account](docs/HOW_I_DIRECTED_AI.md) | Initiative, follow-through and communication; mentor acknowledgment is distinct from validation of every result |

## Controller implementation and experimental judgment

With mentor guidance and AI assistance, I reconstructed the software environment, worked through CUDA, compiler and library issues, checked the command and data pipelines, added a narrower foot-position reward and completed a **500-iteration diagnostic run** within an 8 GB GPU constraint.

The local run used 1,024 parallel environments rather than 8,192. At 24 rollout steps and 500 iterations, those configurations supply 12,288,000 versus 98,304,000 transitions. I recognized that iteration count alone did not make the training exposure equal and proposed matched conditions.

A later single-seed screening measured **33.02 cm mean reference-foot error for the tuned checkpoint versus 9.62 cm for the reference**. It did not establish improvement. The checkpoints had different training histories; the screening does not isolate the reward change as the cause. Reference-foot tracking is also distinct from landing accuracy.

[Reference screening output](projects/tactile-controller/outputs/mission_review/reference_evaluation.json) · [Tuned screening output](projects/tactile-controller/outputs/mission_review/tuned_evaluation.json)

## Navigation assignment and findings

Zhenyu assigned the sequence: run an existing map, then create and test new maps. I supplied that mission and the team's resources to the AI-assisted workflow, clarified scope, examined recordings and measurements, requested explanations and shaped the report. The study includes three authored layouts: an offset opening, staggered baffles and a central island.

Four completed seed-0 trials physically reached their goals; only one passed the framework's arrival-confidence condition. The pilot provides cases to inspect rather than a general robustness result. [Full study](docs/ADMISSIONS_RESEARCH_OVERVIEW.md)

## Preliminary independent exploration

My earlier [G1 rough-terrain package](projects/independent-g1-rough-terrain/README.md) preserves custom task configuration, a command modification, logs and a model from AI-assisted exploration built on Isaac Lab, Unitree assets and an existing RL framework. It is a separate stage from the assigned navigation study and does not demonstrate energy savings or guaranteed fall prevention.

## Team foundation and extensions

Professor Zhao shaped the opportunity and research direction. Zhenyu provided direct guidance, instructions and resources. Huaze Liu and the team supplied the navigation foundation, trained controller, reference maps, localization, planning and sensing methods.

My assigned extension concerns new-map evaluation and reporting within that foundation. **Codex performed substantial map/script implementation, code inspection, experiment operation, quantitative analysis and visualization in response to my directions.** A contribution is credited according to the work performed, including the assistance received.

## AI-assisted learning and implementation

I supplied prompts, research context, priorities and revision requests, examined outputs and asked for corrections. **ChatGPT integrated source materials and generated much of the portfolio's explanatory and presentation text.** It also supported learning and troubleshooting. **Codex** provided the technical assistance described above and helped draft documentation.

[The full disclosure](docs/AI_AND_SOURCE_DISCLOSURE.md) distinguishes my research participation, AI-produced text, team methods and application-writing requirements. Saved artifacts support specific contributions; they do not establish that I independently authored every line.

## Permission for shared materials

I confirm that **Zhenyu Wu authorized inclusion of the team data, reference content and images he supplied in this public GitHub research portfolio**. He is credited as the supplying mentor; original authors, including Huaze Liu and other team or upstream contributors, retain their respective credits and rights.

Existing licenses and notices remain applicable. This permission to include materials does not transfer ownership or create a blanket third-party reuse license. [Permission scope](CONFIDENTIALITY.md) · [Visual sources](assets/SOURCES.md)

## Acknowledgments

I sincerely thank **[Professor Ye Zhao](https://people.research.gatech.edu/ye-zhao) at Georgia Tech** for his quick feedback, engineering advice, and active help finding a suitable research direction. He arranged learning resources, helped connect my humanoid interests to the team's work, and introduced me to **Zhenyu Wu**. His guidance has strongly influenced the engineer I want to become.

**Zhenyu Wu was my main source of practical guidance throughout this project.** He provided detailed instructions, extensive help with setting up the research environment, and the code and resources I needed to begin. His guidance and support were central to my participation and learning. I am especially grateful for the time and effort he devoted to helping me. He also introduced me to Huaze Liu.

I also thank **Huaze Liu** for generously sharing his maps and research work as references for my learning and experiments. His materials supported the new-map evaluation described in this portfolio.

Their guidance and generosity made this learning experience possible. I am grateful for their time and support, and I credit their research contributions throughout this portfolio.

I also appreciate the early onboarding and evaluation guidance from **Rui Yan, Feiyang, and Junnosuke** in the research group. The [mentor guidance record](docs/MENTOR_ALIGNMENT.md) describes these distinct forms of support.

*Shih-Kai Lan (Sky)*
