[← Portfolio home](../README.md) · [My contribution](HOW_I_DIRECTED_AI.md) · [Inside the research](RESEARCH_WALKTHROUGH.md)

# How this portfolio connects to the team's work

*A thematic summary of research-related messages reviewed in Sky's Georgia Tech Slack group and the Zhenyu Wu and Ye Zhao direct conversations. This is a paraphrased source record, not a mentor endorsement of the portfolio.*

Sky began participating in summer 2026 and reports that he continues conducting experiments, with further projects planned. This repository documents the implementation and results of the project presented here.

## How team guidance shaped the work

| Topic and source | What the discussion established | What belongs in this portfolio |
| :--- | :--- | :--- |
| **Introductions and learning resources · research group** | Professor Zhao arranged introductions and requested Isaac Sim / reinforcement-learning learning resources. Rui Yan shared onboarding notes. | The work began with guided learning and access to a research community. |
| **Research interests · research group** | Sky described an interest in learned locomotion and behavior beyond familiar training conditions, asked for a manageable starting project, and reported reading the onboarding material. Professor Zhao asked the team for an accessible RL pipeline. | A specific interest and a request to move from reading into experiments. |
| **Setup and training questions · research group** | Sky reported setting up Isaac Sim / Isaac Lab and posted training visuals, then asked what good training and learning curves should look like. Rui advised beginning with defaults; Feiyang emphasized task-specific tracking metrics. Junnosuke supplied setup guidance. | Early setup, a running-training report, and the beginning of a question about measurement. The message itself does not validate a trained policy's quality. |
| **G1 exploration · research group** | Sky shared an independent G1 package and requested a small, complete contribution involving controlled experiments, comparisons, and reporting. | The [rough-terrain package](../projects/independent-g1-rough-terrain/README.md) is prior exploration. Its energy-related intent is not evidence of measured energy savings. |
| **Research fit and mentorship · research group** | Professor Zhao brought Zhenyu into the conversation, asked for a manageable topic, and steered the introduction toward humanoid/tactile-navigation work in response to Sky's interests. | Professor Zhao helped shape the opportunity and research fit, beyond making an introduction. |
| **Controller onboarding · research group** | Zhenyu supplied access to `feat/playground`, setup documentation, and pretrained examples, and recommended learning the MJLab tracking workflow. | Practical onboarding and team-provided infrastructure, credited to Zhenyu and the team. |
| **Tracking task · Zhenyu DM** | An initial upper-body orientation-tracking topic evolved into concrete instructions for foot-tracking reward tuning, a reference-checkpoint comparison, and AI-assisted use of the README. | The foot-tracking study is a separate earlier task. Do not present it as completed upper-body tracking research or as the later navigation assignment. |
| **Progress reporting · both mentor DMs** | Sky sent a weekly report and disclosed AI help with preparing it. Zhenyu acknowledged the training work and offered to review it; Professor Zhao acknowledged the report and asked whether the mentoring interaction was helpful. | Communication and follow-through are evidenced. Receipt or an encouraging response does not certify the report's technical claims. |
| **Navigation scope · Zhenyu DM** | Zhenyu said he and Professor Zhao had discussed a suitable task: run tactile navigation on different maps, building beyond the team's existing four maps. | The featured study evaluates an existing team system on new layouts. |
| **Coherent research questions · Zhenyu DM** | Sky asked to keep changing tasks within a coherent research area so the work could accumulate depth. Zhenyu described new-map testing as a way to examine generality and robustness. | Evidence of scope negotiation and a research purpose beyond collecting demonstrations. |
| **Evaluation instructions · Zhenyu DM** | Zhenyu supplied `sky/evaluation` and specified the order: run an existing map with a visible UI/video, then create new maps and run experiments. | The current baseline replay, three new maps, saved videos, and result records address the concrete assignment. |

The reviewed conversations establish guidance and assignment context; they do not establish mentor approval of every evaluation result. Raw chat screenshots and unrelated personal details are not reproduced here. The topics above summarize the relevant discussions while preserving the distinction between mentorship and validation of results.

## Does the completed work fit the assignment?

**Yes, as an initial simulation evaluation.** The archive contains an existing-map trial, three newly authored layouts, geometry checks, executed simulation recordings, and a quantitative account of the outcomes. Those deliverables fit the run-existing-then-test-new-maps sequence. The custom maps were implemented with Codex assistance in response to Sky's direction.

**The broader robustness question remains open.** Four completed seed-0 trials do not establish a general success rate or demonstrate performance across a randomized map distribution. Zhenyu mentioned customized/random maps; this pilot uses three deliberately authored layouts. It exercises new geometry, but is not a randomized-map benchmark. Only one of the four completed trials met the framework's formal completion criterion.

| Assignment or ambition | Present status | Next evidence needed |
| :--- | :--- | :--- |
| Run the existing framework and see its behavior | Completed simulation with replay and logs | Mentor review of whether the chosen checkpoint/settings are the preferred baseline |
| Add different maps | Three authored layouts and geometry checks | Systematic variation or a reproducible map generator to test broader generality |
| Evaluate robustness | Pilot observations and identifiable uncertain arrivals | Repeated seeds, matched budgets, and explicit failure categories |
| Improve foot tracking | Earlier reward experiment; later screening did not establish improvement | Matched starting conditions and repeated controlled comparisons |
| Adaptive walking across robots / energy efficiency | Personal future directions | New experiments and suitable transfer/energy metrics |

## Contribution boundaries that make the portfolio credible

Professor Zhao's guidance and help shaping an appropriate research opportunity are distinct from Zhenyu's sustained practical mentoring. Huaze Liu's maps and shared research work provide a reference foundation. The group's earlier onboarding help also matters: Rui Yan, Feiyang, and Junnosuke helped orient Sky toward the tools and evaluation questions.

Sky's documented actions include setting up preliminary G1 training, asking how to evaluate it, seeking a meaningful team task, reporting progress, and keeping the assigned navigation study coherent. The preserved experiments also show the baseline and new-map runs, measurements, and report produced through Sky's participation in the assigned evaluation with AI assistance. The team's algorithms and trained navigation model remain team work; ChatGPT and Codex assistance is described in [Contributions](../CONTRIBUTIONS.md).

This review supports a student research portfolio with clear limits. It does not establish an official appointment, paper authorship, a recommendation from the mentors, or permission to make private team material public.
