![Research portfolio section](../assets/section-direction.svg)

[← Portfolio home](../README.md) · [My role](HOW_I_DIRECTED_AI.md) · [Technical depth](TECHNICAL_DEPTH.md) · [Motivation and future work](MY_RESEARCH_DIRECTION.md)

# Learning to rebuild the research

*A reflection by Shih-Kai Lan (Sky), drafted with AI assistance from my account of the experience. The early hardware failure and loss are my personal account; the records below establish specific surviving work, not a diagnosis of that failure.*

## From learning the setup to losing the work

My robotics work began with learning how to create a place where training could happen. As a high-school student, I asked AI tools to teach me how to install and configure the simulation and training environment. With AI assistance, I began training robots and seeing how settings and inputs could become movement.

Then my computer suffered an internal board failure, and I lost the training work and data I had on that machine. I suspected the workload might have been too demanding, but I did not have a diagnosis that established the cause. The machine had stopped working, and the work I wanted to continue was no longer available there.

I tried again on a better laptop. I used previous conversations and the context I could reconstruct to direct AI through reinstalling the tools and rebuilding the environment. I also asked for settings that would better fit the available computer resources. Continuing meant revisiting the setup, recovering what context I could, and accepting that getting the software running again would not automatically recover lost training.

## What rebuilding actually means

A research environment contains several different kinds of state. Source code defines the experiment. Dependencies make the code executable. Configuration specifies the task and training settings. A checkpoint contains learned parameters and, depending on how it was saved, additional state needed to resume training. Logs explain what happened along the way.

An old conversation can help reconstruct installation decisions; it cannot recreate the exact contents of a lost checkpoint or dataset. That distinction matters to my story. I rebuilt a way to continue working. I do not claim to have recovered every lost file or reproduced a lost training trajectory exactly.

The setup history also includes a concrete compatibility obstacle: a frozen CUDA JAX dependency could not install on native Windows, so the workflow moved to WSL Ubuntu. Historical setup checks recorded GPU discovery in both PyTorch and JAX, 37 registered tasks, and loading a supplied policy. Those were environment checks at that time, not evidence that every training task was validated or that the computer could not fail again.

## What the surviving records show

These are separate documented milestones. The archive does not establish exactly where the early hardware failure falls between them.

| Milestone | Preserved evidence | What it lets a reader conclude |
| :--- | :--- | :--- |
| **Earlier G1 training** | [Training log](../projects/independent-g1-rough-terrain/Logs/training.log), [agent configuration](../projects/independent-g1-rough-terrain/Configuration/agent.yaml), and [saved model](../projects/independent-g1-rough-terrain/Model/model_3000.pt) | A run resumed `model_975.pt` with 500 environments; a later checkpoint survives. This does not mean all lost work was recovered. |
| **A concrete behavior change** | [Command implementation](../projects/independent-g1-rough-terrain/Code/commands.py) | The custom edge-return command guides the robot toward its environment origin beyond 3 m and releases inside 1 m. It does not guarantee fall prevention. |
| **Later foot-tracking work** | [Experiment retrospective](../projects/tactile-controller/RETROSPECTIVE.md) and [screening report](../projects/tactile-controller/outputs/Mission%20Review%20and%20Work%20Report.md) | Training and reward changes were followed by a comparison that did not establish improvement. Keeping the unfavorable result is part of the research record. |
| **Navigation interruption and recovery** | [Crash and recovery record](../projects/tactile-navigation/outputs/crash_diagnostics/Crash%20and%20Recovery.md) | A later Windows crash interrupted one map attempt. Its journal was retained, and the remaining simulations ran in a separate restart folder. |

The installation history above comes from earlier setup records; it was not rerun for this portfolio edit. The linked files are surviving artifacts. None identifies the damaged component or proves the cause of the earlier hardware failure.

## A second interruption, with a more recoverable workflow

The documented October 3 navigation restart is a **separate event** from the early board failure in my account. Windows recorded a `0x139` kernel crash; the responsible driver or component was not identified. I asked for investigation and continuation, and chose when to pause and resume the work.

The resulting AI-assisted recovery preserved completed trials, kept the interrupted offset-opening attempt as `system_restart`, and reran the remaining work separately. The interrupted attempt was not counted as another completed experiment. Restart precautions limited execution to one physical simulation at a time, allocated two CPU cores, monitored a 4 GiB process-group memory limit, and deferred video rendering until simulation finished. These measures constrained workload; they were not a proven repair or a guarantee against hardware failure.

This is a more useful engineering lesson than simply saying that I persisted. A result needs enough saved context to explain what ran, where it stopped, and whether the next run is a continuation or a new attempt. My contribution was to supply the context and direct the response; Codex implemented the diagnostic and recovery workflow.

## The harder lesson: effort is not evidence of improvement

Rebuilding the tools gave me a way to continue, but it did not guarantee a better robot. In the earlier foot-tracking screening, the tuned checkpoint had a mean reference-foot error of **33.02 cm**, compared with **9.62 cm** for the reference checkpoint. Different training histories prevent attributing that difference solely to the reward change. The result still does not support a claim of improved tracking.

That unsuccessful comparison belongs beside the later navigation result: four physical arrivals, but only one trial satisfying the framework's formal completion criterion. Both challenge an easy story in which more effort, a more elaborate configuration, or a convincing video must mean success. [The technical discussion](TECHNICAL_DEPTH.md) traces the actual definitions and tradeoffs.

## The engineer I want to become

Professor Ye Zhao's quick feedback and advice have strongly influenced how I think about becoming a good engineer. Zhenyu Wu's detailed instructions, extensive setup help, and continued practical guidance helped me take part in the work and keep learning. Huaze Liu's generosity in sharing maps and research work gave me references to build on. I am grateful for these distinct contributions.

I want the next version of my workflow to make recovery deliberate: record the environment and exact settings, retain checkpoints and logs, keep a separate backup, and verify that a saved run can actually be restored. Backup restoration is a next step, not something this account claims has already been tested. I also want to begin with a small verified workload and increase it only after checking resource use and behavior.

For me, directing AI is becoming a more specific responsibility: provide the goal and constraints, preserve the evidence, ask what a result really establishes, and understand enough of the process to question it. I am still learning the technical foundations. The setback gives that learning a purpose beyond getting the next run to finish.

[Contributions and full AI-use disclosure →](../CONTRIBUTIONS.md)
