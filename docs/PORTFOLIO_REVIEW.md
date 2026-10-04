[← Portfolio home](../README.md) · [My role](HOW_I_DIRECTED_AI.md) · [Contributions](../CONTRIBUTIONS.md)

# Portfolio review

*Reviewed October 4, 2026 · Asia/Taipei · AI-assisted content and evidence audit*

## Overall assessment: share with research caveats

This is suitable as **Shih-Kai Lan's private student research portfolio**. It explains the assigned new-map evaluation, the team's research foundation, Sky's direction of the AI-assisted workflow, the experiment results, and future interests. The study is a simulation pilot with one seed per map. Broader robustness, hardware performance, and energy-efficiency claims require further experiments.

## What the portfolio shows

| Reader's question | Where to look |
| :--- | :--- |
| Who helped, and how? | The acknowledgments at the very top of the [README](../README.md) and the [contribution record](../CONTRIBUTIONS.md) |
| What did Sky contribute? | [How I directed AI-assisted research](HOW_I_DIRECTED_AI.md): task interpretation, scope clarification, questions about testing, and research communication |
| What did the team provide? | [Contributions and provenance](../CONTRIBUTIONS.md): advice, practical guidance, maps, research code, models, and resources |
| What did AI tools do? | The [AI-use disclosure](../CONTRIBUTIONS.md#ai-assisted-learning-and-implementation): learning support and substantial implementation, execution, analysis, and drafting |
| What was tested and why? | The [research overview](ADMISSIONS_RESEARCH_OVERVIEW.md) |
| What happened? | The [video gallery](VIDEOS.md), [saved results](../projects/tactile-navigation/outputs/sky_results.json), and [full report](../projects/tactile-navigation/outputs/Sky%20Navigation%20Experiment%20Report.md) |
| Is there technical depth? | [Planning, localization, control, metrics, and limitations](TECHNICAL_DEPTH.md) |
| Why does this matter personally? | [Motivation and future research questions](MY_RESEARCH_DIRECTION.md) |
| How did Sky respond to a setback? | [Learning to rebuild the research](SETBACK_AND_REBUILD.md): early setup and training, reported hardware loss, reconstruction, and surviving evidence |

## Methodology and calculation checks

The audit reads the four canonical evaluation records, final simulator-truth poses, saved localization estimates, contact records, and runtime arrival beliefs. It independently recomputes goal distance, physical arrival, time-weighted localization RMSE, and formal completion. The interrupted attempt remains separate from the four completed trials. No new training or simulation was needed for this review.

| Check | Result |
| :--- | :--- |
| Original-source integrity | All 10,101 files match their recorded byte sizes and SHA-256 hashes; 1,590,881,755 source bytes preserved |
| Large-file availability | All 28 unique Git LFS objects were confirmed available through GitHub's authenticated LFS API |
| Navigation counts | Four completed trials; four physical arrivals; one formal success; no recorded falls |
| Final goal errors | 7.31, 6.41, 4.07, and 4.16 cm for original room, island, offset opening, and staggered baffles |
| Final arrival beliefs | 93.35%, 92.41%, 91.89%, and 96.04%; the common runtime threshold is 95% |
| Earlier foot-tracking screening | Saved summaries report 9.62 cm reference versus 33.02 cm tuned mean error; the portfolio correctly retains the unfavorable result |
| Credential-pattern check | No known-pattern matches in scanned source text, presentation files, or Office XML; this is a scoped scan, not proof that every possible secret is absent |

Reproduce the local integrity and navigation calculations with `python presentation/audit_portfolio.py` from the repository root. Results are stored in [PORTFOLIO_AUDIT.json](../PORTFOLIO_AUDIT.json). The [link audit](../ADMISSIONS_PRESENTATION_AUDIT.json) covers relative file links in the student-facing Markdown pages and embedded HTML. Large-file availability was checked separately against GitHub and is time-dependent.

## Presentation and attribution review

- Professor Ye Zhao's quick feedback, advice on becoming a good engineer, personal influence, and introduction remain prominent in concise acknowledgments. Zhenyu Wu is explicitly described as Sky's main source of practical guidance, with extensive instructions, setup help, and resources. Huaze Liu is thanked specifically for generously sharing maps and research work as references.
- Student contribution statements use examples of Sky's actual directions. They credit interpretation of the assignment and direction of the AI-assisted workflow. Team algorithms and AI-generated implementation remain attributed.
- The main navigation includes a direct **My role** link. The personal narrative presents Sky's account of the motivating event without claiming that Sky won an award.
- The cover trajectory and numerical chart derive from saved records. The arrival chart starts at zero, shows the common threshold, and identifies the single-seed scope. Its percentages are internal beliefs, not confidence intervals or population success rates.
- Original reports are preserved as historical artifacts. The earlier foot-tracking retrospective explains the unfavorable later screening and prevents the old report from serving as proof of improvement.
- GitHub's contributor graph reflects commits to this archive. The written acknowledgment and contribution record explain research support that a commit graph cannot represent.
- The new recovery narrative identifies the early board failure and loss as Sky's account, without assigning a date or verified cause. It distinguishes that event from the documented October 3 Windows restart and does not claim all lost files were restored or future damage was prevented.
- The expanded technical discussion links coordinate transformations, reward design, training exposure, joint arrival belief, time-weighted metrics, and recovery semantics to preserved records. Mathematical reward examples and proposed follow-up experiments are labeled accordingly. Prior WSL setup checks are historical records, not fresh validation of the current environment.

## GitHub description and topics

The [metadata definition](../presentation/github_metadata.json) identifies this as a personal student research portfolio and classifies it with general subjects: robotics, humanoid robotics, robot navigation, tactile sensing, localization, motion planning, MuJoCo, simulation, student research, research portfolio, and AI-assisted research.

These labels describe the material in the repository. They do not imply an official appointment, institutional endorsement, or a completed research breakthrough. Only general subject labels are used because [GitHub topic names are public even when the repository is private](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/classifying-your-repository-with-topics).

## Caveats for readers and sharing

1. **Pilot scope:** One seed per map, simulation only, a substituted team checkpoint, and different original/new-map time caps limit the conclusions. The interrupted attempt is excluded from completed-trial counts.
2. **Future directions:** Adaptive walking with less retraining, transfer across robot bodies, and cleaner energy are interests for further study. They are not established outcomes of these experiments.
3. **Contribution evidence:** Personal experiences and mentor support are based on Sky's account and supplied conversation. The documented directions support the stated workflow role; the artifact audit does not independently establish mastery of every algorithm or mentor endorsement of an application.
4. **Private access:** An admissions reader cannot open a private repository without access. Use an appropriately approved excerpt or arrange access deliberately while respecting the confidentiality of team materials.
5. **Team materials:** Inclusion follows the sharing permission reported by Sky. This private portfolio does not grant a new license over the team's work or establish permission for public release.

## Issues corrected in this review

The generic GitHub description and missing topic labels were replaced with descriptive metadata. A direct contribution link was added to the main navigation. Editing instructions were removed from the personal narrative, and the attribution text was polished for readers. No change to the research results or underlying source snapshots was required.

The subsequent narrative revision added the early setup/training and rebuilding experience, strengthened Professor Zhao's acknowledgment, and expanded the technical discussion with source links and concrete comparisons. This was a documentation revision: no new training, simulation, or hardware diagnosis was performed. Relative links and the changed-file scope were checked again; the earlier full source and metric audit remains the numerical evidence for the unchanged experiments.
