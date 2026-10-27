[← Portfolio home](../README.md) · [Team assignment and source record](MENTOR_ALIGNMENT.md) · [Inside the research](RESEARCH_WALKTHROUGH.md)

# My role in the team research

*Shih-Kai Lan (Sky) · assignment, participation, evidence and contribution boundaries*

I participated as a student member of an ongoing humanoid robotics project under Professor Ye Zhao's mentorship and Zhenyu Wu's direct guidance. After preliminary G1 simulation training, I worked on the navigation evaluation Zhenyu assigned to me, directing AI tools to help with my tasks and collaborating with the team. I brought the task and team code into the workflow, participated in discussions, examined results and shared reports. The [dated mentor record](MENTOR_ALIGNMENT.md) and artifacts below show the basis and limits of these claims.

## Work I contributed

- **Model and training work:** With AI assistance, I set up and trained a preliminary G1 simulation model. The [archived package](../projects/independent-g1-rough-terrain/README.md) preserves the configuration, training log, checkpoint and edge-return command. I did not build the team's later navigation controller from scratch.
- **Controller implementation and evaluation:** I reconstructed the training environment, modified the foot-tracking reward and completed a 500-iteration diagnostic run with mentor guidance and AI assistance. I identified the unequal sample budget and proposed matched training conditions; the [technical evidence below](#technical-work-and-experimental-judgment) connects these contributions to saved artifacts.
- **Experiments and data:** I took on Zhenyu's new-map evaluation, supplied the team's code and task context, and used AI-assisted scripts to run tests and calculate and inspect outcomes such as final goal distance and arrival belief. The [study](ADMISSIONS_RESEARCH_OVERVIEW.md) keeps the baseline, three new layouts, recordings, numerical results and limits together.
- **Team communication:** I discussed training quality and experimental scope with the group, shared a weekly report with mentors, and asked for the results to be explained clearly. The [dated record](MENTOR_ALIGNMENT.md) distinguishes those conversations from validation of every technical claim.

## Technical work and experimental judgment

My technical contribution included rebuilding the software environment, working through CUDA, compiler and library compatibility problems, checking the command and data pipelines, modifying a foot-tracking reward, and completing a diagnostic training run. I did this under Professor Ye Zhao's supervision and Zhenyu Wu's direct guidance, using AI for reading, implementation and troubleshooting and checking its suggestions against the code, observed behavior and measurements.

| Contribution | Preserved evidence | Research value |
| :--- | :--- | :--- |
| Reconstructing a working training environment | [Environment and recovery account](SETBACK_AND_REBUILD.md), including the Windows-to-WSL compatibility issue and recorded setup checks | Makes it possible to execute and inspect experiments under limited computing resources |
| Implementing a foot-tracking reward modification | [Reward diff](../projects/tactile-controller/local-reward-change.patch): an additional 0.05 m kernel, retaining the original 0.20 m signal | Turns a proposed precision incentive into a specific change that can be tested |
| Completing a 500-iteration diagnostic run | [Recorded training evidence](../projects/tactile-controller/report-evidence/experiment_evidence.json) and [model_499.pt](../projects/tactile-controller/logs/rsl_rl/g1_decoupled_joint_foot_mimic_arm/2026-09-13_22-38-08_foot_precise_s005_500/model_499.pt) | Preserves a trained artifact and measurements for later examination; iteration numbering begins at zero |
| Identifying unequal sample budgets and proposing a fairer comparison | The evidence records 1,024 versus 8,192 environments and 24 rollout steps; the [experiment report](../projects/tactile-controller/outputs/Mission%20Review%20and%20Work%20Report.md) specifies a matched-training follow-up | Prevents attributing an outcome to the reward change when training exposure and other conditions differ |

At 500 iterations, the local configuration collected 12,288,000 samples, compared with 98,304,000 for 8,192 environments at the same iteration count: a ratio of 1:8. I recognized that this was not a fair basis for claiming improved accuracy and proposed matching training conditions before drawing that conclusion. Matching sample counts alone would still not make this short run equivalent to the mature reference checkpoint; initialization, command distribution, evaluation conditions and repeated seeds also matter.

The later screening measured 33.02 cm mean reference-foot error for the tuned checkpoint versus 9.62 cm for the supplied reference. I retain that unfavorable result because it keeps the open research question honest. The value of this work includes a runnable modification, a saved diagnostic model, and a concrete account of what a stronger comparison would require. It does not establish improved foot-placement accuracy, balance or hand tracking.

## My participation and its evidence

| Strength | Concrete example | Why it matters in research |
| :--- | :--- | :--- |
| **Initiative with a realistic scope** | On August 24 and September 6, I asked for a manageable but meaningful task: specific experiments, data analysis, or a technical report that could contribute to the ongoing work. | I tried to find a useful contribution within my experience and available time. |
| **Curiosity about evidence** | After reporting running training on July 23, I asked what good training looked like and how to interpret reward and learning curves. The group directed me toward task-specific metrics. | I was beginning to ask what the output demonstrated, beyond whether the program ran. This is evidence of asking the question, not proof that I already mastered evaluation. |
| **Coherence and scope judgment** | On September 25, as tasks changed, I asked that they stay within a related area so the research could accumulate depth. I later clarified that the assignment was new-map navigation testing. | I communicated the need for a connected research question and a deliverable that addressed it. |
| **Communication and follow-through** | I shared my earlier G1 work, requested guidance, and sent a weekly report to both mentors on September 14, explicitly acknowledging AI help. | Progress became something the team could inspect and respond to. An acknowledged report is evidence of communication, not validation of all its claims. |
| **Persistence and practical direction** | In my personal account, I rebuilt after losing work to a computer failure. During the later navigation restart, I requested investigation, controlled when work resumed, and asked for an explanation of what had been tested. | I supplied context and constraints that guided recovery and continuation. Codex implemented the technical recovery workflow. |

These strengths describe how I participated. The team's methods and substantial AI implementation remain credited alongside that participation.

## From direction to research evidence

The learning process connects to three tangible stages: an [earlier G1 task package](../projects/independent-g1-rough-terrain/README.md), a [foot-tracking reward experiment with an unfavorable later screening](../projects/tactile-controller/RETROSPECTIVE.md), and the three authored maps evaluated with the team's navigation framework. The archive preserves the maps, executed simulations, recordings, and analysis:

- [Research overview](ADMISSIONS_RESEARCH_OVERVIEW.md): the question, map changes, results, and limitations.
- [Video gallery](VIDEOS.md): recordings of the four completed simulation trials.
- [Technical explanation](TECHNICAL_DEPTH.md): how planning, localization, walking, and arrival criteria fit together.
- [Inside the research](RESEARCH_WALKTHROUGH.md): the system, one complete trial, and a controlled follow-up study outline.
- [Setback and rebuild](SETBACK_AND_REBUILD.md): my early experience, the surviving training artifacts, and the separate later recovery record.
- [Full experiment report](../projects/tactile-navigation/outputs/Sky%20Navigation%20Experiment%20Report.md): settings, provenance, measurements, and the interrupted attempt.

The team's trained controller and navigation algorithms provided the research foundation. For my mentor-assigned evaluation, I helped carry out and document three new-map trials with recordings and measurements, using Codex for substantial map implementation, simulation execution, validation and calculations. My contribution also included the research context, discussions, questions, review of results and reporting described above. This division records my participation under my mentors' guidance alongside the tool's technical contribution.

## The research question behind the directions

The purpose of the assignment was to explore how an existing navigation framework behaves beyond its original maps. That requires testing executed movement as well as describing a proposed layout. The completed pilot then raised a more specific question: why did all four trials physically reach the goal region, while only one met the framework's formal completion criterion?

The reports explain the distinction between physical arrival and the system's belief about arrival. One seed per map provides an initial case study; repeated trials are needed to support broader claims about robustness. These findings came from the AI-assisted analysis and are presented with their limits.

The [short result interpretation](ADMISSIONS_RESEARCH_OVERVIEW.md#reading-one-result-and-planning-the-next-test) connects two recorded trials to a concrete follow-up proposal. My next step would be to discuss that proposal with my mentors, record a prediction before testing, and compare it with repeated results. The [follow-up study outline](RESEARCH_WALKTHROUGH.md#what-a-stronger-next-study-would-test) provides a concrete opportunity. These are proposed next actions, not completed achievements.

## AI assistance and attribution

As a student still learning the tools, I used ChatGPT for explanations and setup guidance and Codex for substantial help with map/script implementation, simulation runs, calculations, visualization and drafting. I supplied the mentor's assignment and repository, corrected the scope when needed, examined the outputs and asked what the results could support. Zhenyu also encouraged AI-assisted reading of the README and parameter experimentation. These tools extended my ability to participate; their technical work and the team's original methods remain credited. [Full contribution boundaries →](../CONTRIBUTIONS.md)

## A concise account of my contribution

> As a student participant in a Georgia Tech robotics project, mentored by Professor Ye Zhao and directly guided by Zhenyu Wu, I worked on environment reconstruction, preliminary G1 training, and a foot-tracking reward modification with AI assistance. I completed a 500-iteration diagnostic run, identified an unequal sample budget, and proposed matched training conditions before claiming improvement. I also discussed the work with the team and shared a weekly report. For Zhenyu Wu's navigation assignment, I used the team's system and AI-assisted tools to test new maps, calculate and examine results, and shape a documented pilot. Four completed trials physically reached their goals, but only one met the system's formal arrival criterion. That gap led me to ask what evidence a robot needs to know it has succeeded. The team's methods and models, mentor guidance and substantial AI implementation are credited separately.

*This paragraph is an AI-assisted draft based on the record. It should be used in an application only in wording that I can personally explain and stand behind.*

[Full contribution and AI-use disclosure →](../CONTRIBUTIONS.md)
