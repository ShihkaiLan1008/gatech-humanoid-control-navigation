[← Portfolio home](../README.md) · [Team assignment and source record](MENTOR_ALIGNMENT.md) · [Inside the research](RESEARCH_WALKTHROUGH.md)

# My role in the team research

*Shih-Kai Lan (Sky) · assignment, participation, evidence and contribution boundaries*

I worked as a student research assistant in an ongoing humanoid robotics team. After preliminary G1 simulation training, I led the AI-assisted work on Zhenyu Wu's assigned navigation evaluation, working with mentors and team members rather than building their framework anew. I brought the task and team code into the workflow, participated in discussions, examined results and shared reports. The [dated mentor record](MENTOR_ALIGNMENT.md) and artifacts below show the basis and limits of these claims.

## Work I contributed

- **Model and training work:** With AI assistance, I set up and trained a preliminary G1 simulation model. The [archived package](../projects/independent-g1-rough-terrain/README.md) preserves the configuration, training log, checkpoint and edge-return command. I did not build the team's later navigation controller from scratch.
- **Experiments and data:** I took on Zhenyu's new-map evaluation, supplied the team's code and task context, and used AI-assisted scripts to run tests and calculate and inspect outcomes such as final goal distance and arrival belief. The [study](ADMISSIONS_RESEARCH_OVERVIEW.md) keeps the baseline, three new layouts, recordings, numerical results and limits together.
- **Team communication:** I discussed training quality and experimental scope with the group, shared a weekly report with mentors, and asked for the results to be explained clearly. The [dated record](MENTOR_ALIGNMENT.md) distinguishes those conversations from validation of every technical claim.

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

The team's trained controller and navigation algorithms provided the research foundation. I led my assigned evaluation workstream through three new-map trials, recordings and measurements, using Codex for substantial map implementation, simulation execution, validation and calculations. My contribution also included the research context, discussions, questions, review of results and reporting described above. This division records my research direction and participation alongside the tool's technical contribution.

## The research question behind the directions

The purpose of the assignment was to explore how an existing navigation framework behaves beyond its original maps. That requires testing executed movement as well as describing a proposed layout. The completed pilot then raised a more specific question: why did all four trials physically reach the goal region, while only one met the framework's formal completion criterion?

The reports explain the distinction between physical arrival and the system's belief about arrival. One seed per map provides an initial case study; repeated trials are needed to support broader claims about robustness. These findings came from the AI-assisted analysis and are presented with their limits.

The next step in developing my ability is to make more of my understanding directly visible: explain one complete run in my own words, predict what a controlled change should do, and compare that prediction with the result. The [follow-up study outline](RESEARCH_WALKTHROUGH.md#what-a-stronger-next-study-would-test) provides a concrete opportunity. These are proposed next actions, not completed achievements.

## AI assistance and attribution

As a student still learning the tools, I used ChatGPT for explanations and setup guidance and Codex for substantial help with map/script implementation, simulation runs, calculations, visualization and drafting. I supplied the mentor's assignment and repository, corrected the scope when needed, examined the outputs and asked what the results could support. Zhenyu also encouraged AI-assisted reading of the README and parameter experimentation. These tools extended my ability to participate; their technical work and the team's original methods remain credited. [Full contribution boundaries →](../CONTRIBUTIONS.md)

## A concise account of my contribution

> As a student member of an ongoing Georgia Tech robotics project, I began by setting up and training a preliminary G1 simulation model with AI assistance, then joined discussions about how to evaluate the work and shared a weekly report. For Zhenyu Wu's navigation assignment, I used the team's system and AI-assisted tools to test new maps, calculate and examine results, and shape a documented pilot. Four completed trials physically reached their goals, but only one met the system's formal arrival criterion. That gap led me to ask what evidence a robot needs to know it has succeeded. The team's methods and models, mentor guidance and substantial AI implementation are credited separately.

*This paragraph is an AI-assisted draft based on the record. It should be used in an application only in wording that I can personally explain and stand behind.*

[Full contribution and AI-use disclosure →](../CONTRIBUTIONS.md)
