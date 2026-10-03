![Research portfolio section](../assets/section-direction.svg)

[← Portfolio home](../README.md) · [Research overview](ADMISSIONS_RESEARCH_OVERVIEW.md) · [Technical depth](TECHNICAL_DEPTH.md) · [Video gallery](VIDEOS.md)

# Why I care, and where I want to go

Personal narrative prepared from Sky's account of their interests and experience. The event description follows that account rather than asserting a separately verified official event title or award record. No award or prize is claimed for Sky.

## Where the interest began

In 2024, I attended a Georgia Tech / CIC Asia Future Tech Forum event. I saw robotics work involving professors, university students and high-school students like me, including students receiving recognition for their work. I was surprised by how much people at different stages of education could contribute. Technology began to mean more to me than an impressive machine: it could be a way to do something useful for the world.

That experience made me want to pursue mechanical engineering and create things with practical value. It also changed who I thought could participate in research. Seeing students close to my age take part made the possibility feel real.

## What makes the work exciting to me

In the robotics work, I became especially excited by the connection between data inputs and physical behavior: changing what was provided to a robot and then watching it find a route and walk. The feeling was like a parent watching a child learn to walk. A line of code or an input was no longer just an abstraction; it had become movement.

That excitement is my personal reaction, not a claim that the current model was trained from scratch by me or that every change improved performance. The team supplied the trained controller, and AI assisted implementation and experiment operation. The portfolio's contribution statement keeps that foundation visible.

## How the experiment deepens that interest

The new-map study shows why robotics asks for more than excitement about visible movement. All four completed trials ended inside the goal region without a recorded fall, yet three were still classified as arrival uncertain. The robot's true position and the system's confidence about its position answered different questions.

That distinction gives me a concrete direction for further study: how can a robot connect useful action with enough evidence to trust that action? The next step is to understand the saved belief and contact records, repeat trials, and test whether additional final sensing improves correctly justified completion. These are future research tasks, not work already completed.

## The longer-term questions I want to pursue

I would like robot walking to become more adaptive, with less need to train again for every new situation. The present study reused a trained walking controller across different maps without per-map retraining. It did not remove the original training requirement or demonstrate training-free locomotion. Investigating how feedback, planning and reusable movement knowledge can reduce retraining is a direction I want to explore.

I would also like useful movement knowledge to be shared across robots. One data file cannot be assumed to work on every robot: bodies, sensors and operating conditions differ. A research path toward that ambition is to define a transferable representation, evaluate it on a second robot or simulated morphology, and document where it fails as carefully as where it works.

Cleaner energy is another long-term motivation. A first measurable connection to walking research would be to examine actuator work, power or energy per distance together with stability and task completion. None of the current experiments establishes an energy-efficiency improvement or clean-energy technology. Making the ambition measurable would let me investigate it honestly.

## My proposed next steps

1. **Understand:** Explain the current map, localization and controller pipeline, and trace one recorded run end to end.
2. **Replicate:** Use matched settings and multiple seeds to see which observations persist.
3. **Investigate:** Compare final sensing strategies and inspect position versus yaw uncertainty.
4. **Adapt:** Study transfer to changed terrain or a second morphology with less additional training, while documenting the existing trained components.
5. **Measure energy:** Add meaningful energy metrics, then evaluate tradeoffs between energy, stability and navigation success.

The reason I want to keep doing this work is that it connects curiosity with something physical and potentially useful. My ambition is broad, but the research path should be specific: ask a question, make a controlled change, measure what happened, acknowledge what is still uncertain and continue.

## Before using this narrative in an application

Read it in your own voice and adjust wording to reflect your actual experience. Add one specific example of an input you personally changed, a design you reviewed or a result you explained. Keep the distinction between personal work, team work and AI assistance. The technical record supports the study; it does not by itself prove every personal contribution or future ambition.
