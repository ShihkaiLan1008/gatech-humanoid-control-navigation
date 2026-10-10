[← Portfolio home](../README.md) · [Team assignment and source record](MENTOR_ALIGNMENT.md) · [Inside the research](RESEARCH_WALKTHROUGH.md)

# My contribution: questions, initiative, and follow-through

*Shih-Kai Lan (Sky) · Student direction, team mentorship, and disclosed AI assistance*

My contribution developed through learning the tools, seeking a useful role in an existing team project, asking how to judge results, and directing AI assistance toward a specific experiment. The strongest evidence is a sequence of actions that a reader can examine. The dated [mentor-alignment record](MENTOR_ALIGNMENT.md) connects the conversation evidence with the technical artifacts.

## What my actions show

| Strength | Concrete example | Why it matters in research |
| :--- | :--- | :--- |
| **Initiative with a realistic scope** | On August 24 and September 6, I asked for a manageable but meaningful task: specific experiments, data analysis, or a technical report that could contribute to the ongoing work. | I tried to find a useful contribution within my experience and available time. |
| **Curiosity about evidence** | After reporting running training on July 23, I asked what good training looked like and how to interpret reward and learning curves. The group directed me toward task-specific metrics. | I was beginning to ask what the output demonstrated, beyond whether the program ran. This is evidence of asking the question, not proof that I already mastered evaluation. |
| **Coherence and scope judgment** | On September 25, as tasks changed, I asked that they stay within a related area so the research could accumulate depth. I later clarified that the assignment was new-map navigation testing. | I communicated the need for a connected research question and a deliverable that addressed it. |
| **Communication and follow-through** | I shared my earlier G1 work, requested guidance, and sent a weekly report to both mentors on September 14, explicitly acknowledging AI help. | Progress became something the team could inspect and respond to. An acknowledged report is evidence of communication, not validation of all its claims. |
| **Persistence and practical direction** | In my personal account, I rebuilt after losing work to a computer failure. During the later navigation restart, I requested investigation, controlled when work resumed, and asked for an explanation of what had been tested. | I supplied context and constraints that guided recovery and continuation. Codex implemented the technical recovery workflow. |

These strengths describe how I participated. The team's methods and substantial AI implementation remain credited alongside that participation.

## How I used AI deliberately

ChatGPT helped me organize resources, clarify unfamiliar concepts, and work through setup questions. I gave AI tools the mentor's repository and assignment, corrected the scope when necessary, asked where testing had happened, and requested explanations of the results and limitations. Zhenyu explicitly encouraged AI-assisted reading of the README and parameter experimentation; AI use was part of the learning approach discussed with him.

The examples below combine my personal account of early setup and rebuilding with my recorded directions for the new-map study. The early failure's date and hardware cause are not established by the surviving artifacts.

| Decision | Direction I gave | Why it mattered |
| :--- | :--- | :--- |
| **Learn the starting process** | Ask AI to teach me how to install and configure the tools, then help me undertake robot training. | Made environment setup part of my learning, rather than assuming a working research system already existed. This early experience is described in my personal account. |
| **Rebuild with constraints** | Use previous conversations and reconstructed context to request a fresh setup on a better laptop, with settings suited to the resources available. | Gave AI a recovery goal and operating constraints. Recreating setup decisions does not recreate lost weights or establish that hardware cannot fail again. |
| **Identify the task** | Use Zhenyu's assignment and the team's repository as the starting point for testing new maps. | Connected the assistance to the mentor's research question and existing framework. |
| **Clarify scope** | Ask where testing had happened and restate that the mission was to make and test new maps based on the team's work. | Made the requested experiment explicit and distinguished it from other robotics work in the archive. |
| **Respond to interruption** | Ask for investigation after the later navigation-run restart, then direct when to pause and resume the research. | Addressed a practical obstacle and controlled when work continued. |
| **Specify the explanation** | Ask what was done, what the experiments mean, what they can do, and what questions come next; request both accessibility and technical depth. | Defined the purpose and audience of the research report. |
| **Set attribution expectations** | Request prominent thanks to Professor Zhao, Zhenyu Wu, and Huaze Liu, and disclosure of ChatGPT's help. | Made mentorship, shared resources, and AI assistance visible to readers. |

## From direction to research evidence

The learning process connects to three tangible stages: an [earlier G1 task package](../projects/independent-g1-rough-terrain/README.md), a [foot-tracking reward experiment with an unfavorable later screening](../projects/tactile-controller/RETROSPECTIVE.md), and the three authored maps evaluated with the team's navigation framework. The archive preserves the maps, executed simulations, recordings, and analysis:

- [Research overview](ADMISSIONS_RESEARCH_OVERVIEW.md): the question, map changes, results, and limitations.
- [Video gallery](VIDEOS.md): recordings of the four completed simulation trials.
- [Technical explanation](TECHNICAL_DEPTH.md): how planning, localization, walking, and arrival criteria fit together.
- [Inside the research](RESEARCH_WALKTHROUGH.md): the system, one complete trial, and a controlled follow-up study outline.
- [Setback and rebuild](SETBACK_AND_REBUILD.md): my early experience, the surviving training artifacts, and the separate later recovery record.
- [Full experiment report](../projects/tactile-navigation/outputs/Sky%20Navigation%20Experiment%20Report.md): settings, provenance, measurements, and the interrupted attempt.

The team's trained controller and navigation algorithms provided the research foundation. Codex developed the map variants and carried out substantial implementation, execution, validation, and analysis in response to the requested scope. My contribution was directing that workflow through the assignment, context, questions, priorities, and revisions described above.

## The research question behind the directions

The purpose of the assignment was to explore how an existing navigation framework behaves beyond its original maps. That requires testing executed movement as well as describing a proposed layout. The completed pilot then raised a more specific question: why did all four trials physically reach the goal region, while only one met the framework's formal completion criterion?

The reports explain the distinction between physical arrival and the system's belief about arrival. One seed per map provides an initial case study; repeated trials are needed to support broader claims about robustness. These findings came from the AI-assisted analysis and are presented with their limits.

The next step in developing my ability is to make more of my understanding directly visible: explain one complete run in my own words, predict what a controlled change should do, and compare that prediction with the result. The [follow-up study outline](RESEARCH_WALKTHROUGH.md#what-a-stronger-next-study-would-test) provides a concrete opportunity. These are proposed next actions, not completed achievements.

## A concise account of my contribution

> With guidance from Professor Ye Zhao and extensive practical mentoring from Zhenyu Wu, I progressed from learning simulation tools to participating in an evaluation of humanoid tactile navigation. I asked how to judge training quality, sought a useful experimental role, and kept changing tasks focused on a coherent research direction. Using team resources and substantial AI assistance, the work produced new maps and a documented simulation pilot. Four trials physically reached their goals, but only one met the system's completion criterion. The experience gives me a sharper research question: what evidence does a robot need to know that its action succeeded? I want to continue by making controlled comparisons, explaining their limits, and developing more of the technical work myself.

*This paragraph is an AI-assisted draft based on the record. It should be used in an application only in wording that I can personally explain and stand behind.*

[Full contribution and AI-use disclosure →](../CONTRIBUTIONS.md)
