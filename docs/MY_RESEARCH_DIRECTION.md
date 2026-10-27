![Research portfolio section](../assets/section-direction.svg)

[← Portfolio home](../README.md) · [Research overview](ADMISSIONS_RESEARCH_OVERVIEW.md) · [Technical depth](TECHNICAL_DEPTH.md) · [Video gallery](VIDEOS.md)

# Why I care, and where I want to go

*A personal reflection by Shih-Kai Lan (Sky), prepared with AI writing assistance from my account of the experience.*

## Where the interest began

I attended a Georgia Tech / CIC Asia Future Tech Forum event. I saw robotics work involving professors, university students and high-school students like me, including students receiving recognition for their work. I was surprised by how much people at different stages of education could contribute. Technology began to mean more to me than an impressive machine: it could be a way to do something useful for the world.

That experience made me want to pursue mechanical engineering and create things with practical value. It also changed who I thought could participate in research. Seeing students close to my age take part made the possibility feel real.

## What makes the work exciting to me

In the robotics work, I became especially excited by the connection between data inputs and physical behavior: changing what was provided to a robot and then watching it find a route and walk. The feeling was like a parent watching a child learn to walk. A line of code or an input was no longer just an abstraction; it had become movement.

I did undertake earlier robot-training work with AI assistance. The featured new-map study then used the team's already-trained controller; I did not train that navigation controller from scratch. My excitement about movement is not a claim that every change improved performance. The portfolio's contribution statement keeps those phases and the assistance visible.

## When continuing meant starting again

My early work involved asking AI to teach me how to install the tools, set up the training environment, and begin training robots. After my computer suffered an internal board failure, I lost the training work I had on that machine. I wondered whether the workload was too demanding, but the cause was never established.

I kept trying. On a better laptop, I used previous conversations and whatever context I could reconstruct to direct AI through installing the tools again. I also wanted settings that better respected the computer's resources. An old conversation could help rebuild the setup; it could not bring back exact training weights that had been lost.

This experience gives a practical meaning to the engineer I want to become. I want to understand the conditions under which work can run, know what has actually been saved, and make it possible to resume after an interruption. [The full recovery reflection](SETBACK_AND_REBUILD.md) separates my early experience from the later documented navigation crash and explains what the surviving records show.

Professor Ye Zhao's quick feedback and advice about becoming a good engineer have strongly influenced me. Zhenyu Wu's extensive practical guidance helped me work through setup and begin participating in the research. I value both the opportunity they gave me and the example of taking engineering seriously while continuing to learn.

## How the experiment deepens that interest

The new-map study shows why robotics asks for more than excitement about visible movement. All four completed trials ended inside the goal region without a recorded fall, yet three were still classified as arrival uncertain. The robot's true position and the system's confidence about its position answered different questions.

That distinction gives me a concrete direction for further study: how can a robot connect useful action with enough evidence to trust that action? The next step is to understand the saved belief and contact records, repeat trials, and test whether additional final sensing improves correctly justified completion. These are future research tasks, not work already completed.

## The future contribution that motivates me

I want to help build robots that can offer practical physical assistance in the spaces where people live and work. I can imagine them carrying supplies, helping with everyday tasks, or inspecting places that are difficult or dangerous for people to enter. The possibility of reducing physical burdens and human exposure to hazardous work gives my interest in robot movement a purpose.

That ambition draws me toward the less visible questions behind a walking robot: can it handle a different layout, estimate where it is, and recognize when it needs more information? My participation in the team's navigation evaluation lets me begin working on those questions under Professor Ye Zhao's mentorship and Zhenyu Wu's direct guidance. The new-map experiments are a small simulation-based contribution to a much longer research path. I want to keep developing the ability to connect careful testing with technology that people can eventually depend on.

## The longer-term questions I want to pursue

I would like robot walking to become more adaptive, with less need to train again for every new situation. The present study reused a trained walking controller across different maps without per-map retraining. It did not remove the original training requirement or demonstrate training-free locomotion. Investigating how feedback, planning and reusable movement knowledge can reduce retraining is a direction I want to explore.

I would also like useful movement knowledge to be shared across robots. One data file cannot be assumed to work on every robot: bodies, sensors and operating conditions differ. A research path toward that ambition is to define a transferable representation, evaluate it on a second robot or simulated morphology, and document where it fails as carefully as where it works.

Cleaner energy is another long-term motivation. A first measurable connection to walking research would be to examine actuator work, power or energy per distance together with stability and task completion. None of the current experiments establishes an energy-efficiency improvement or clean-energy technology. Making the ambition measurable would let me investigate it honestly.

## Continuing beyond this project

I began participating in the research in summer 2026 and continue conducting experiments. This repository documents the implementation and recorded results of this project within my ongoing research. I plan to continue with further projects and experiments as the work develops; the results presented here refer to the runs preserved in this repository. My participation continues under Professor Ye Zhao's mentorship and Zhenyu Wu's direct guidance. The questions below guide the work I want to develop next.

## My proposed next steps

1. **Understand:** Explain the current map, localization and controller pipeline, and trace one recorded run end to end.
2. **Replicate:** Use matched settings and multiple seeds to see which observations persist.
3. **Investigate:** Compare final sensing strategies and inspect position versus yaw uncertainty.
4. **Adapt:** Study transfer to changed terrain or a second morphology with less additional training, while documenting the existing trained components.
5. **Measure energy:** Add meaningful energy metrics, then evaluate tradeoffs between energy, stability and navigation success.

The reason I want to keep doing this work is that it connects curiosity with something physical and potentially useful. My ambition is broad, but the research path should be specific: ask a question, make a controlled change, measure what happened, acknowledge what is still uncertain and continue.

## How I am learning to direct the work

I used the mentor's assignment to give AI tools a clear goal, asked where testing had happened, and restated the need to create and test new maps using the team's work as a reference. I also asked for explanations of the research's meaning and future possibilities. Those directions connect my curiosity to a specific task and a record that others can examine. [My contributions and experimental evidence](../CONTRIBUTIONS.md) describe my work alongside the team's foundation and AI assistance.

The earlier group conversations give this interest a more concrete history. I expressed curiosity about learned locomotion beyond familiar training conditions and asked how to judge good training and learning curves. I also asked for changing tasks to remain connected so the work could accumulate depth. Those were early questions, not established results. The next step is to carry that curiosity into a prediction and a controlled comparison I can explain. [The recorded task history](MENTOR_ALIGNMENT.md) and [proposed next study](RESEARCH_WALKTHROUGH.md#what-a-stronger-next-study-would-test) connect the motivation to that work.
