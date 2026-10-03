# GitHub portfolio brainstorm for college applications

> Updated scope, 3 October 2026: The current mission is tactile-navigation generalization on newly authored simulation maps. Use a `tactile-navigation-map-study` case study as the main portfolio project: show the team's existing framework, your three new maps, unchanged evaluation settings, physical navigation videos, success/failure results, and analysis. The foot-tracking ideas below describe an earlier project and are optional supporting work. The screenshots authorize inclusion of team work in a report; they do not explicitly establish public redistribution permission for the entire codebase. The user separately reports GitHub permission, which should be recorded with its actual scope before publishing team source.

Prepared 3 October 2026. This is a local plan, not a published repository.

## The story to tell

Make the central story: **I learned to investigate a real robotics problem, designed a controlled experiment, and used evidence to judge the result.** A reviewer should understand your work in one minute, then have a path to inspect the technical details.

Your strongest current candidate is Unitree G1 footstep tracking with simultaneous arm control. The inspected checkout contains a team controller, your existing foot reward modification, training artifacts, and weekly reports. The presence of these files does not establish which parts you personally authored; verify each contribution before using it in an application.

## Permission and attribution

You stated that Zhengyu allows you to put the team's work on your GitHub. Use that permission to show the context needed to explain your work. Add a clear acknowledgement and preserve existing authorship and license notices when copying approved code. Repository access and authorship are different: the portfolio should make both the team's foundation and your contribution visible.

Suggested wording, to complete after confirming contribution details:

> This project builds on the tactile_controller research codebase developed by Zhenyu Wu and the research team. Team materials are included with permission, as communicated to me by Zhenyu. My contribution was [specific work], supported by [commit, configuration, experiment log, or report]. I used AI assistance for [specific tasks] and validated the results through [specific checks].

Use the spelling and affiliation that Zhengyu prefers. Do not claim a formal research appointment unless that is your actual role. Keep private Slack conversations out of the public portfolio; a concise acknowledgement is sufficient. Record the scope of permission privately and use the existing license for team code rather than assigning it a new license yourself.

## Recommended profile and repositories

| Piece | Purpose | Contents |
|---|---|---|
| Profile README | A quick introduction | Name, robotics interests, one-sentence project summary, links to your best work |
| `g1-footstep-tracking-study` | Main technical case study | Project README, contribution statement, experiment configs, evaluation scripts, plots, demo, acknowledgements |
| `robotics-research-notes` (optional) | Show how your understanding developed | A few edited explanations and experiment decisions, each tied to evidence |
| Portfolio website (later) | A readable application link | A short introduction and project cards linking to the evidence repositories |

Start with the profile and one excellent case study. Add another repository when it represents a distinct project. GitHub supports profile READMEs and pinned projects: https://docs.github.com/en/account-and-profile/how-tos/profile-customization

## A one-minute project README

1. **Question:** Can a more precise foot-position reward improve commanded footstep tracking without reducing stability?
2. **Demo:** A short simulation clip with target and actual foot positions visible. Label it as simulation and identify the checkpoint.
3. **My contribution:** Three concrete actions, each linked to a diff, configuration, analysis, or report.
4. **Approach:** Explain the existing upper/lower policy structure, then the reward change and why you chose it.
5. **Results:** A compact baseline/tuned table using physical metrics, with uncertainty and limitations.
6. **Reproduce:** Exact code revision, environment, datasets, seed, commands, and evaluation settings.
7. **What I learned:** One failure or unexpected result and the next experiment it motivated.
8. **Credits:** Team foundation, approved reused assets, dependencies, and AI assistance.

## Experiment ideas that show research judgment

- Baseline versus one additional narrow foot-position reward. Keep environment count, seed, training budget, and other rewards fixed.
- Compare landing error separately from average reference tracking error; they answer different questions.
- Evaluate short, long, sideways, and turning steps separately, so an average cannot hide a weak command region.
- Check stability and hand tracking alongside foot accuracy. A smaller foot error can come with a tradeoff.
- Repeat promising comparisons across several seeds. A single run is preliminary evidence.
- Include a concise failure analysis with a video frame and a proposed explanation that can be tested.

The existing change adds a 0.05 m reward kernel while retaining the 0.20 m term. The earlier saved diagnostic evidence records about 0.259 m final reference-foot error and about 0.243 m over the final 20 iterations. These are historical training metrics, not proof of improved landing accuracy. The 500-iteration run used 1,024 environments, so it cannot be treated as equivalent to a 500-iteration run using 8,192 environments. A matched baseline/tuned comparison is the appropriate test of the reward change; merely extending one run does not establish causality.

## Suggested repository layout

```text
g1-footstep-tracking-study/
  README.md
  CONTRIBUTIONS.md
  ACKNOWLEDGEMENTS.md
  docs/
    experiment-design.md
    results.md
    lessons-learned.md
  experiments/
    baseline-config/
    precise-foot-reward-config/
  evaluation/
    evaluate_foot_tracking.py
  results/
    metrics.csv
    plots/
  media/
    demo.mp4
  reproduction/
    environment.md
    commands.md
```

Add approved team code as a credited fork or selected source files when it helps reproduction. Keep the case study easy to navigate. Store large checkpoints and datasets in an appropriate release or storage location and document how to obtain them. Check the weekly reports for private conversations, unpublished assets, and accurate affiliation wording before selecting public excerpts.

## A contribution statement template

| Work item | Team foundation | My work | Evidence |
|---|---|---|---|
| Controller | Existing two-policy G1 framework | [What I understood, implemented, or changed] | [Source/diff] |
| Data | Existing generation method | [Generation/validation I performed] | [Validation report] |
| Reward experiment | Original tracking rewards | [Precisely describe the modification] | [Diff and saved config] |
| Evaluation | Existing metrics and checkpoints | [Comparison and interpretation I performed] | [Script and result file] |
| Communication | Team research context | [Report/visuals I prepared] | [Public report excerpt] |

## Build order

First confirm the latest assignment in Slack and finish its evaluation. Then select the strongest evidence, write the contribution statement, and assemble the case-study repository. Add the profile README and pin the case study. A website can follow once the project story and results are ready.

For an application description, use a verified version of: “Investigated humanoid footstep tracking in a team research codebase; designed a reward-tuning experiment and evaluated accuracy and stability in simulation.” Add measured improvements only after the comparison supports them. The value of the portfolio is the work and judgment it demonstrates, not the number of repositories or contribution squares.
