[← Portfolio home](../README.md) · [My role](HOW_I_DIRECTED_AI.md) · [Contributions](../CONTRIBUTIONS.md)

# Portfolio review

*This review describes the preserved project snapshot. Numerical audit results apply to the recorded artifacts; presentation updates do not constitute new experimental validation. Sky reports continuing experiments and plans for further projects.*

## Overall assessment: share with research caveats

This is suitable as **Shih-Kai Lan's student research portfolio**. It explains the assigned new-map evaluation, the team's research foundation, Sky's participation in that evaluation under Professor Ye Zhao's mentorship and Zhenyu Wu's direct guidance, the experiment results, and future interests. The study is a simulation pilot with one seed per map. Broader robustness, hardware performance, and energy-efficiency claims require further experiments.

## What the portfolio shows

| Reader's question | Where to look |
| :--- | :--- |
| Who helped, and how? | The acknowledgments later in the [README](../README.md) and the [contribution record](../CONTRIBUTIONS.md) |
| What did Sky contribute? | [My role in the team research](HOW_I_DIRECTED_AI.md): preliminary G1 training, the assigned navigation evaluation, data review, team discussions and reports, with supporting evidence |
| What did the team provide? | [Contributions and provenance](../CONTRIBUTIONS.md): advice, practical guidance, maps, research code, models, and resources |
| What did AI tools do? | The [AI-use disclosure](../CONTRIBUTIONS.md#ai-assisted-learning-and-implementation): learning support and substantial implementation, execution, analysis, and drafting |
| What was tested and why? | The [research overview](ADMISSIONS_RESEARCH_OVERVIEW.md) |
| What happened? | The [video gallery](VIDEOS.md), [saved results](../projects/tactile-navigation/outputs/sky_results.json), and [full report](../projects/tactile-navigation/outputs/Sky%20Navigation%20Experiment%20Report.md) |
| Is there technical depth? | [The system and a complete trial](RESEARCH_WALKTHROUGH.md), followed by [technical definitions and five challenges](TECHNICAL_DEPTH.md) |
| Why does this matter personally? | [Motivation and future research questions](MY_RESEARCH_DIRECTION.md) |
| How did Sky respond to a setback? | [Learning to rebuild the research](SETBACK_AND_REBUILD.md): early setup and training, reported hardware loss, reconstruction, and surviving evidence |
| Does this match the mentor's task? | [Mentor alignment](MENTOR_ALIGNMENT.md): task evolution, completed deliverables, and the remaining robustness question |

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
| Offset-opening episode figure | 15,483 samples; 29 probe attempts comprising 12 contacts and 17 completed misses; independently recomputed RMSE 6.64 cm; source hash and checks saved in the trace audit |
| Credential-pattern check | No known-pattern matches in scanned source text, presentation files, or Office XML; this is a scoped scan, not proof that every possible secret is absent |

Reproduce the local integrity and navigation calculations with `python presentation/audit_portfolio.py` from the repository root. Results are stored in [PORTFOLIO_AUDIT.json](../PORTFOLIO_AUDIT.json). The [link audit](../ADMISSIONS_PRESENTATION_AUDIT.json) covers relative file links in the student-facing Markdown pages and embedded HTML. Large-file availability was checked separately against GitHub and is time-dependent.

The new figure is reproduced by [build_navigation_trace.py](../presentation/build_navigation_trace.py), which reads the completed offset-opening record and verifies sample/probe counts, duration, goal error, and time-weighted RMSE against the saved results. Its [audit](../presentation/navigation_trace_audit.json) records the source hash. The plotting work reanalyzes existing data; it does not add another trial.

## Presentation and attribution review

- Professor Ye Zhao's feedback, engineering advice, personal influence, and active help matching research opportunities to Sky's interests remain prominent in concise acknowledgments. The group record supports his requests for resources, an accessible pipeline, and a suitable humanoid topic. Zhenyu remains the main practical mentor, and Huaze is thanked for reference maps/work. Early guidance from other group members is also acknowledged.
- Student contribution statements distinguish Sky's participation in the assigned evaluation and direction of AI tools for his tasks from the team's underlying algorithms and the substantial tool-assisted implementation. The supporting record and artifacts support the stated participation without assigning Sky sole authorship of the code or methods.
- The main navigation includes **My contribution** and **Inside the research**. The personal narrative presents Sky's account of the motivating event without claiming that Sky won an award.
- The cover trajectory and numerical chart derive from saved records. The arrival chart starts at zero, shows the common threshold, and identifies the single-seed scope. Its percentages are internal beliefs, not confidence intervals or population success rates.
- Original reports are preserved as historical artifacts. The earlier foot-tracking retrospective explains the unfavorable later screening and prevents the old report from serving as proof of improvement.
- GitHub's contributor graph reflects commits to this archive. The written acknowledgment and contribution record explain research support that a commit graph cannot represent.
- The new recovery narrative identifies the early board failure and loss as Sky's account, without assigning a date or verified cause. It distinguishes that event from the documented Windows restart and does not claim all lost files were restored or future damage was prevented.
- The expanded technical discussion links coordinate transformations, reward design, training exposure, joint arrival belief, time-weighted metrics, and recovery semantics to preserved records. Mathematical reward examples and proposed follow-up experiments are labeled accordingly. Prior WSL setup checks are historical records, not fresh validation of the current environment.
- The Slack review adds evidence of Sky's initiative, questions about evaluation, report communication, and scope negotiation. Paraphrases exclude unrelated personal details and raw chat screenshots. Mentor acknowledgment of a report is not presented as scientific validation or application endorsement.
- The walkthrough makes the known-map premise, route versus sensing roles, miss observations, and execution-policy limits visible. Its new figure plots all recorded error samples with separate contact/miss markers and does not attribute error changes causally to nearby probes.

## GitHub description and topics

The [metadata definition](../presentation/github_metadata.json) identifies this as a personal student research portfolio and classifies it with general subjects: robotics, humanoid robotics, robot navigation, tactile sensing, localization, motion planning, MuJoCo, simulation, student research, research portfolio, AI-assisted research, reinforcement learning, Isaac Lab, and experiment design.

These labels describe the material in the repository. They do not imply an official appointment, institutional endorsement, or a completed research breakthrough. Only general subject labels are used because [GitHub topic names are public even when the repository is private](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/classifying-your-repository-with-topics).

## Caveats for readers and sharing

1. **Pilot scope:** One seed per map, simulation only, a substituted team checkpoint, and different original/new-map time caps limit the conclusions. The interrupted attempt is excluded from completed-trial counts.
2. **Future directions:** Adaptive walking with less retraining, transfer across robot bodies, and cleaner energy are interests for further study. They are not established outcomes of these experiments.
3. **Contribution evidence:** Personal experiences are based on Sky's account; mentor interactions were also reviewed in the visible Slack conversations. The documented actions support the stated participation and workflow role; the audit does not independently establish mastery of every algorithm or mentor endorsement of an application.
4. **Public access:** The repository is public and readable without a GitHub account. Large artifacts use Git LFS.
5. **Team materials:** Sky confirms permission from Zhenyu Wu to include the team data, reference content and images he supplied in this public GitHub portfolio. Zhenyu's provider role, the original authors' credits and applicable upstream notices remain distinct; no blanket relicensing is asserted.
6. **AI-generated presentation:** ChatGPT integrated source material and generated much of the narrative in response to Sky's prompts and revisions. Codex supplied substantial technical and presentation assistance. The disclosure does not establish independent student authorship of that prose or guarantee that it can be submitted as an original application response. See [AI and source disclosure](AI_AND_SOURCE_DISCLOSURE.md).

## Issues corrected in this review

The generic GitHub description and missing topic labels were replaced with descriptive metadata. A direct contribution link was added to the main navigation. Editing instructions were removed from the personal narrative, and the attribution text was polished for readers. No change to the research results or underlying source snapshots was required.

The subsequent narrative revision added the early setup/training and rebuilding experience, strengthened Professor Zhao's acknowledgment, and expanded the technical discussion with source links and concrete comparisons. This was a documentation revision: no new training, simulation, or hardware diagnosis was performed. Relative links and the changed-file scope were checked again; the earlier full source and metric audit remains the numerical evidence for the unchanged experiments.

The presentation revision reorganized the entry points around student participation and a complete research walkthrough. It added the mentor/task record, a source-backed episode plot, and controlled next-study questions. The initial upper-body topic, later foot-tracking work, and subsequent new-map assignment are now distinguished. A planned baseline comparison is clearly labeled unrun, and authored-map pilot coverage is distinguished from a randomized robustness benchmark. Source integrity, calculations, and relative links were rechecked for this revision.
